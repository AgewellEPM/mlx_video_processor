"""
Base model class and registry for MLX Video Processor.

Defines the abstract interface that all model processors must implement,
ensuring consistency and interoperability across different model types.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Type, Callable
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_video_processor.types import (
    FrameResult,
    ModelType,
    PrecisionType,
    ProcessingConfig,
)


class BaseModel(ABC):
    """
    Abstract base class for all model processors.

    All model implementations must inherit from this class and implement
    the required abstract methods. This ensures a consistent interface
    for the video processing pipeline.
    """

    # Class attributes to be set by subclasses
    model_name: str = "base"
    model_type: ModelType = ModelType.CLASSIFICATION
    default_input_size: tuple[int, int] = (224, 224)  # (height, width)
    supports_batching: bool = False

    def __init__(
        self,
        precision: PrecisionType = PrecisionType.FLOAT16,
        device: Optional[str] = None,
        compile_model: bool = True,
    ):
        """
        Initialize the base model.

        Args:
            precision: Weight precision to use (float16, float32, etc.)
            device: Device to run on (None = default GPU)
            compile_model: Whether to compile the forward pass with mx.compile
        """
        self.precision = precision
        self.device = device
        self.compile_model = compile_model
        self._model: Optional[nn.Module] = None
        self._compiled_forward: Optional[Callable] = None
        self._is_loaded = False

        # Set dtype based on precision
        self.dtype = self._get_dtype(precision)

    def _get_dtype(self, precision: PrecisionType) -> mx.Dtype:
        """Convert precision enum to MLX dtype."""
        dtype_map = {
            PrecisionType.FLOAT32: mx.float32,
            PrecisionType.FLOAT16: mx.float16,
            PrecisionType.BFLOAT16: mx.bfloat16,
            PrecisionType.INT8: mx.int8,
        }
        return dtype_map.get(precision, mx.float16)

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model weights and prepare for inference.

        This method should:
        1. Download or load model weights
        2. Initialize the model architecture
        3. Move weights to the correct precision
        4. Optionally compile the forward pass
        """
        pass

    @abstractmethod
    def preprocess(self, frame: np.ndarray) -> mx.array:
        """
        Preprocess a frame for model input.

        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format

        Returns:
            Preprocessed MLX array ready for model input
        """
        pass

    @abstractmethod
    def forward(self, x: mx.array) -> Any:
        """
        Run forward pass on preprocessed input.

        Args:
            x: Preprocessed input tensor

        Returns:
            Raw model output (model-specific format)
        """
        pass

    @abstractmethod
    def postprocess(self, output: Any, original_size: tuple[int, int]) -> Any:
        """
        Postprocess model output into structured results.

        Args:
            output: Raw model output from forward pass
            original_size: Original frame size (height, width)

        Returns:
            Structured result (ClassificationResult, DetectionResult, etc.)
        """
        pass

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_ms: float,
    ) -> FrameResult:
        """
        Process a single frame through the complete pipeline.

        This is the main entry point for frame processing. It handles
        preprocessing, inference, and postprocessing in one call.

        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format
            frame_index: Index of the frame in the video
            timestamp_ms: Timestamp of the frame in milliseconds

        Returns:
            FrameResult with model outputs
        """
        if not self._is_loaded:
            self.load_model()

        start_time = time.perf_counter()

        # Get original size for postprocessing
        original_size = (frame.shape[0], frame.shape[1])

        # Preprocess
        x = self.preprocess(frame)

        # Forward pass
        if self._compiled_forward is not None:
            output = self._compiled_forward(x)
        else:
            output = self.forward(x)

        # Force evaluation to get accurate timing
        mx.eval(output) if isinstance(output, mx.array) else None

        # Postprocess
        result = self.postprocess(output, original_size)

        processing_time = (time.perf_counter() - start_time) * 1000

        # Create frame result based on model type
        frame_result = FrameResult(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            processing_time_ms=processing_time,
        )

        # Set the appropriate result field based on model type
        if self.model_type == ModelType.CLASSIFICATION:
            frame_result.classification = result
        elif self.model_type == ModelType.OBJECT_DETECTION:
            frame_result.detection = result
        elif self.model_type == ModelType.SEMANTIC_SEGMENTATION:
            frame_result.segmentation = result
        elif self.model_type == ModelType.DEPTH_ESTIMATION:
            frame_result.depth = result
        elif self.model_type == ModelType.PANOPTIC_SEGMENTATION:
            frame_result.panoptic = result

        return frame_result

    def process_batch(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        timestamps_ms: list[float],
    ) -> list[FrameResult]:
        """
        Process a batch of frames.

        Default implementation processes frames sequentially.
        Override for true batch processing where supported.

        Args:
            frames: List of input frames
            frame_indices: Corresponding frame indices
            timestamps_ms: Corresponding timestamps

        Returns:
            List of FrameResult objects
        """
        results = []
        for frame, idx, ts in zip(frames, frame_indices, timestamps_ms):
            results.append(self.process_frame(frame, idx, ts))
        return results

    def warmup(self, input_size: Optional[tuple[int, int]] = None) -> None:
        """
        Warm up the model with a dummy input.

        This triggers JIT compilation and memory allocation,
        ensuring consistent timing for subsequent inference.

        Args:
            input_size: Size of warmup input (height, width), defaults to model default
        """
        if not self._is_loaded:
            self.load_model()

        size = input_size or self.default_input_size
        dummy_frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)

        # Run a few warmup passes
        for _ in range(3):
            _ = self.process_frame(dummy_frame, 0, 0.0)

    def get_info(self) -> dict[str, Any]:
        """Return model information dictionary."""
        return {
            "name": self.model_name,
            "type": self.model_type.name,
            "precision": self.precision.value,
            "input_size": self.default_input_size,
            "supports_batching": self.supports_batching,
            "is_loaded": self._is_loaded,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(precision={self.precision.value}, loaded={self._is_loaded})"


class ModelRegistry:
    """
    Registry for model processors.

    Provides a central place to register and retrieve model implementations
    by name, enabling dynamic model loading based on configuration.
    """

    _models: dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
        """
        Decorator to register a model class.

        Usage:
            @ModelRegistry.register("fastvit")
            class FastViTProcessor(BaseModel):
                ...
        """
        def decorator(model_cls: Type[BaseModel]) -> Type[BaseModel]:
            cls._models[name.lower()] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        """
        Get a model class by name.

        Args:
            name: Model name (case-insensitive)

        Returns:
            Model class

        Raises:
            KeyError: If model not found
        """
        name_lower = name.lower()
        if name_lower not in cls._models:
            available = ", ".join(cls._models.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return cls._models[name_lower]

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model names."""
        return list(cls._models.keys())

    @classmethod
    def create(
        cls,
        name: str,
        precision: PrecisionType = PrecisionType.FLOAT16,
        **kwargs,
    ) -> BaseModel:
        """
        Create a model instance by name.

        Args:
            name: Model name
            precision: Weight precision
            **kwargs: Additional arguments passed to model constructor

        Returns:
            Instantiated model processor
        """
        model_cls = cls.get(name)
        return model_cls(precision=precision, **kwargs)
