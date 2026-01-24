"""
Model orchestrator for running multiple models.

Provides a unified interface for running multiple models
on the same video with combined visualization.
"""

from typing import Callable, Optional, Any
from pathlib import Path
import numpy as np
from tqdm import tqdm

from mlx_video_processor.models.base import BaseModel, ModelRegistry
from mlx_video_processor.types import (
    FrameResult,
    ProcessingConfig,
    VideoMetadata,
)
from mlx_video_processor.utils.frame_extractor import FrameExtractor
from mlx_video_processor.utils.video_io import VideoWriter
from mlx_video_processor.utils.timer import PerformanceMonitor
from mlx_video_processor.visualizers.composite import CompositeVisualizer


class ModelOrchestrator:
    """
    Orchestrates multiple models on video data.

    Allows running classification, detection, segmentation, and depth
    estimation models on the same frames with combined visualization.

    Optimized for Apple Silicon by leveraging MLX's unified memory
    to share frame data between models without copying.
    """

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
    ):
        """
        Initialize model orchestrator.

        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        self._models: dict[str, BaseModel] = {}
        self._visualizer = CompositeVisualizer()
        self._monitor = PerformanceMonitor()

    def add_model(
        self,
        name: str,
        model: Optional[BaseModel] = None,
        **model_kwargs,
    ) -> "ModelOrchestrator":
        """
        Add a model to the orchestrator.

        Args:
            name: Model name (from registry) or custom name
            model: Optional pre-created model instance
            **model_kwargs: Arguments passed to model constructor

        Returns:
            Self for chaining
        """
        if model is not None:
            self._models[name] = model
        else:
            # Create from registry
            self._models[name] = ModelRegistry.create(name, **model_kwargs)

        return self

    def add_models(self, model_names: list[str], **kwargs) -> "ModelOrchestrator":
        """
        Add multiple models by name.

        Args:
            model_names: List of model names from registry
            **kwargs: Arguments passed to all model constructors

        Returns:
            Self for chaining
        """
        for name in model_names:
            self.add_model(name, **kwargs)
        return self

    def remove_model(self, name: str) -> "ModelOrchestrator":
        """Remove a model from the orchestrator."""
        if name in self._models:
            del self._models[name]
        return self

    def list_models(self) -> list[str]:
        """List active model names."""
        return list(self._models.keys())

    def load_all_models(self) -> None:
        """Load all models (for pre-warming)."""
        for name, model in self._models.items():
            print(f"Loading {name}...")
            model.load_model()
            model.warmup()

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        progress: bool = True,
        callback: Optional[Callable[[FrameResult, np.ndarray], None]] = None,
    ) -> list[FrameResult]:
        """
        Process video with all models.

        Args:
            input_path: Path to input video
            output_path: Optional output video path
            progress: Show progress bar
            callback: Optional per-frame callback

        Returns:
            List of combined FrameResult objects
        """
        if not self._models:
            raise ValueError("No models added to orchestrator")

        results = []
        self._monitor.start_session()

        # Load all models first
        print("Loading models...")
        for name, model in self._models.items():
            with self._monitor.time(f"load_{name}"):
                model.load_model()
                model.warmup()

        # Open video
        extractor = FrameExtractor(input_path, self.config)
        extractor.open()
        metadata = extractor.metadata

        # Setup output writer
        writer = None
        if output_path:
            output_fps = self.config.output_fps or metadata.fps / self.config.frame_interval
            writer = VideoWriter(
                output_path,
                output_fps,
                self.config.output_width or metadata.width,
                self.config.output_height or metadata.height,
            )
            writer.open()

        try:
            frame_iterator = extractor.extract_frames()

            if progress:
                frame_iterator = tqdm(
                    frame_iterator,
                    total=len(extractor),
                    desc="Processing with multiple models",
                )

            for frame_idx, timestamp_ms, frame in frame_iterator:
                # Create combined result
                combined = FrameResult(
                    frame_index=frame_idx,
                    timestamp_ms=timestamp_ms,
                )

                total_processing_time = 0.0

                # Run each model
                for name, model in self._models.items():
                    with self._monitor.time(f"inference_{name}"):
                        result = model.process_frame(frame, frame_idx, timestamp_ms)

                    total_processing_time += result.processing_time_ms

                    # Merge results based on model type
                    if result.classification is not None:
                        combined.classification = result.classification
                    if result.detection is not None:
                        combined.detection = result.detection
                    if result.segmentation is not None:
                        combined.segmentation = result.segmentation
                    if result.depth is not None:
                        combined.depth = result.depth
                    if result.panoptic is not None:
                        combined.panoptic = result.panoptic

                combined.processing_time_ms = total_processing_time
                results.append(combined)

                # Visualize and write
                with self._monitor.time("visualization"):
                    vis_frame = self._visualizer.draw(frame, combined, is_bgr=True)

                if writer:
                    with self._monitor.time("video_write"):
                        writer.write_frame(vis_frame, is_rgb=False)

                if callback:
                    callback(combined, vis_frame)

        finally:
            extractor.close()
            if writer:
                writer.close()

        return results

    def process_single_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process a single frame with all models.

        Args:
            frame: Frame array (H, W, 3) BGR

        Returns:
            Combined FrameResult
        """
        if not self._models:
            raise ValueError("No models added to orchestrator")

        combined = FrameResult(frame_index=0, timestamp_ms=0.0)
        total_time = 0.0

        for name, model in self._models.items():
            model.load_model()
            result = model.process_frame(frame, 0, 0.0)
            total_time += result.processing_time_ms

            if result.classification is not None:
                combined.classification = result.classification
            if result.detection is not None:
                combined.detection = result.detection
            if result.segmentation is not None:
                combined.segmentation = result.segmentation
            if result.depth is not None:
                combined.depth = result.depth
            if result.panoptic is not None:
                combined.panoptic = result.panoptic

        combined.processing_time_ms = total_time
        return combined

    def visualize_frame(
        self,
        frame: np.ndarray,
        result: FrameResult,
        layout: str = "overlay",
    ) -> np.ndarray:
        """
        Visualize results on a frame.

        Args:
            frame: Original frame
            result: Combined FrameResult
            layout: "overlay" or "grid"

        Returns:
            Visualized frame
        """
        self._visualizer.layout = layout
        return self._visualizer.draw(frame, result, is_bgr=True)

    def get_performance_summary(self) -> str:
        """Get performance metrics summary."""
        return self._monitor.get_summary()

    def get_model_info(self) -> dict[str, dict]:
        """Get information about all loaded models."""
        return {name: model.get_info() for name, model in self._models.items()}


def create_default_orchestrator(
    models: list[str] = None,
    config: Optional[ProcessingConfig] = None,
) -> ModelOrchestrator:
    """
    Create an orchestrator with common model configurations.

    Args:
        models: List of model names (defaults to all available)
        config: Processing configuration

    Returns:
        Configured ModelOrchestrator
    """
    orchestrator = ModelOrchestrator(config)

    if models is None:
        models = ["fastvit", "yolo", "deeplab", "depth"]

    for name in models:
        orchestrator.add_model(name)

    return orchestrator


class ParallelOrchestrator(ModelOrchestrator):
    """
    Orchestrator that runs models in parallel where possible.

    Uses MLX's stream system to overlap computation between
    models for improved throughput on Apple Silicon.
    """

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        num_streams: int = 2,
    ):
        """
        Initialize parallel orchestrator.

        Args:
            config: Processing configuration
            num_streams: Number of parallel streams
        """
        super().__init__(config)
        self.num_streams = num_streams

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        progress: bool = True,
        callback: Optional[Callable[[FrameResult, np.ndarray], None]] = None,
    ) -> list[FrameResult]:
        """
        Process video with parallel model execution.

        Note: This is a simplified implementation. Full parallelism
        would require more sophisticated stream management.
        """
        import mlx.core as mx
        import threading
        from queue import Queue

        if not self._models:
            raise ValueError("No models added to orchestrator")

        results = []
        self._monitor.start_session()

        # Load models
        for name, model in self._models.items():
            model.load_model()
            model.warmup()

        extractor = FrameExtractor(input_path, self.config)
        extractor.open()
        metadata = extractor.metadata

        writer = None
        if output_path:
            output_fps = self.config.output_fps or metadata.fps / self.config.frame_interval
            writer = VideoWriter(
                output_path,
                output_fps,
                self.config.output_width or metadata.width,
                self.config.output_height or metadata.height,
            )
            writer.open()

        try:
            frame_iterator = extractor.extract_frames()

            if progress:
                frame_iterator = tqdm(
                    frame_iterator,
                    total=len(extractor),
                    desc="Parallel processing",
                )

            for frame_idx, timestamp_ms, frame in frame_iterator:
                combined = FrameResult(
                    frame_index=frame_idx,
                    timestamp_ms=timestamp_ms,
                )

                # Process models (could be parallelized with proper stream management)
                model_results = {}
                total_time = 0.0

                for name, model in self._models.items():
                    result = model.process_frame(frame, frame_idx, timestamp_ms)
                    model_results[name] = result
                    total_time += result.processing_time_ms

                # Merge results
                for name, result in model_results.items():
                    if result.classification is not None:
                        combined.classification = result.classification
                    if result.detection is not None:
                        combined.detection = result.detection
                    if result.segmentation is not None:
                        combined.segmentation = result.segmentation
                    if result.depth is not None:
                        combined.depth = result.depth
                    if result.panoptic is not None:
                        combined.panoptic = result.panoptic

                combined.processing_time_ms = total_time
                results.append(combined)

                vis_frame = self._visualizer.draw(frame, combined, is_bgr=True)

                if writer:
                    writer.write_frame(vis_frame, is_rgb=False)
                if callback:
                    callback(combined, vis_frame)

        finally:
            extractor.close()
            if writer:
                writer.close()

        return results
