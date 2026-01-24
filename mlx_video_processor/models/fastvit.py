"""
FastViT T8 Image Classification Processor.

FastViT is a fast hybrid vision transformer for efficient image classification.
This implementation uses MLX for optimal performance on Apple Silicon.

Model specifications:
- Parameters: 3.6M
- Size: 7.8MB (Float16)
- Input: 256x256 RGB images
- Output: 1000 ImageNet classes
"""

from typing import Any, Optional
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_video_processor.models.base import BaseModel, ModelRegistry
from mlx_video_processor.types import (
    ClassificationResult,
    ModelType,
    PrecisionType,
)


# ImageNet class names (abbreviated - full list would have 1000 entries)
# In production, this would be loaded from a file
IMAGENET_CLASSES = [
    "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead",
    "electric_ray", "stingray", "cock", "hen", "ostrich",
    # ... (truncated for brevity - would include all 1000 classes)
]


class RepMixer(nn.Module):
    """
    Reparameterizable token mixer for FastViT.

    Uses structural reparameterization to convert complex training-time
    architecture into simple inference-time operations.
    """

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Depthwise convolution for spatial mixing
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim
        )
        self.norm = nn.BatchNorm(dim)

    def __call__(self, x: mx.array) -> mx.array:
        # x is in NHWC format (B, H, W, C) - no transpose needed
        out = self.dwconv(x)
        out = self.norm(out)
        return out


class ConvFFN(nn.Module):
    """
    Convolutional Feed-Forward Network.

    Uses large kernel convolutions for efficient local feature extraction.
    """

    def __init__(self, dim: int, hidden_dim: int, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2

        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=kernel_size,
            padding=padding, groups=hidden_dim
        )
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.norm = nn.BatchNorm(hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        # x is in NHWC format (B, H, W, C) - no transposes needed
        x = self.fc1(x)
        x = self.norm(x)
        x = nn.gelu(x)
        x = self.dwconv(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        return x


class FastViTBlock(nn.Module):
    """
    FastViT transformer block.

    Combines RepMixer for token mixing with ConvFFN for channel mixing.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.BatchNorm(dim)
        self.token_mixer = RepMixer(dim)
        self.norm2 = nn.BatchNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvFFN(dim, hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Patch embedding using overlapping convolution."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=7, stride=stride, padding=3
        )
        self.norm = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        # x is in NHWC format (B, H, W, C) - no transpose needed
        x = self.proj(x)
        x = self.norm(x)
        return x


class Downsample(nn.Module):
    """Spatial downsampling layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        # x is in NHWC format (B, H, W, C) - no transpose needed
        x = self.conv(x)
        x = self.norm(x)
        return x


class FastViTT8(nn.Module):
    """
    FastViT-T8 architecture.

    A tiny variant of FastViT optimized for speed while maintaining accuracy.
    Uses hybrid CNN-Transformer design with structural reparameterization.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        dims: tuple[int, ...] = (48, 96, 192, 384),
        depths: tuple[int, ...] = (2, 2, 4, 2),
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Patch embedding
        self.patch_embed = PatchEmbed(in_channels, dims[0], stride=4)

        # Build stages
        self.stages = []
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            stage_blocks = [FastViTBlock(dim, mlp_ratio) for _ in range(depth)]
            self.stages.append(stage_blocks)

            # Add downsampling except for last stage
            if i < len(dims) - 1:
                self.stages.append([Downsample(dim, dims[i + 1])])

        # Flatten nested list for proper module registration
        self.blocks = [block for stage in self.stages for block in stage]

        # Classification head
        self.norm = nn.BatchNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        # Patch embedding
        x = self.patch_embed(x)

        # Apply all blocks
        for block in self.blocks:
            x = block(x)

        # Global average pooling
        x = self.norm(x)
        x = mx.mean(x, axis=(1, 2))  # (B, C)

        # Classification
        x = self.head(x)
        return x


@ModelRegistry.register("fastvit")
class FastViTProcessor(BaseModel):
    """
    FastViT T8 processor for image classification.

    Optimized for Apple Silicon using MLX's unified memory architecture.
    """

    model_name = "FastViT-T8"
    model_type = ModelType.CLASSIFICATION
    default_input_size = (256, 256)
    supports_batching = True

    # ImageNet normalization parameters
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        precision: PrecisionType = PrecisionType.FLOAT16,
        device: Optional[str] = None,
        compile_model: bool = True,
        top_k: int = 5,
        class_names: Optional[list[str]] = None,
    ):
        """
        Initialize FastViT processor.

        Args:
            precision: Weight precision
            device: Device to run on
            compile_model: Whether to compile forward pass
            top_k: Number of top predictions to return
            class_names: Custom class names (defaults to ImageNet)
        """
        super().__init__(precision, device, compile_model)
        self.top_k = top_k
        self.class_names = class_names or self._load_imagenet_classes()

    def _load_imagenet_classes(self) -> list[str]:
        """Load ImageNet class names."""
        # In production, load from file. Using placeholder for now.
        # Would typically load from imagenet_classes.json
        return [f"class_{i}" for i in range(1000)]

    def load_model(self) -> None:
        """Load FastViT-T8 model weights."""
        if self._is_loaded:
            return

        # Initialize model
        self._model = FastViTT8(num_classes=1000)

        # In production, load pretrained weights:
        # weights = mx.load("fastvit_t8.safetensors")
        # self._model.load_weights(list(weights.items()))

        # Compile forward pass for performance
        if self.compile_model:
            self._compiled_forward = mx.compile(self._model.__call__)

        self._is_loaded = True

    def preprocess(self, frame: np.ndarray) -> mx.array:
        """
        Preprocess frame for FastViT input.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            Preprocessed tensor in NHWC format (1, 256, 256, 3)
        """
        import cv2

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        resized = cv2.resize(rgb, (self.default_input_size[1], self.default_input_size[0]))

        # Normalize to [0, 1] then apply ImageNet normalization
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.MEAN) / self.STD

        # Convert to MLX array in NHWC format and add batch dimension
        x = mx.array(normalized, dtype=self.dtype)
        x = mx.expand_dims(x, axis=0)

        return x

    def forward(self, x: mx.array) -> mx.array:
        """Run forward pass."""
        return self._model(x)

    def postprocess(
        self,
        output: mx.array,
        original_size: tuple[int, int],
    ) -> ClassificationResult:
        """
        Postprocess model output to classification result.

        Args:
            output: Model logits (1, 1000)
            original_size: Original frame size (unused for classification)

        Returns:
            ClassificationResult with top-k predictions
        """
        # Apply softmax to get probabilities
        probs = mx.softmax(output, axis=-1)
        probs = np.array(probs[0])

        # Get top-k predictions
        top_indices = np.argsort(probs)[::-1][:self.top_k]

        top_k_results = [
            (int(idx), self.class_names[idx], float(probs[idx]))
            for idx in top_indices
        ]

        # Top prediction
        top_idx = top_indices[0]

        return ClassificationResult(
            class_id=int(top_idx),
            class_name=self.class_names[top_idx],
            confidence=float(probs[top_idx]),
            top_k=top_k_results,
        )

    def process_batch(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        timestamps_ms: list[float],
    ) -> list["FrameResult"]:
        """
        Process batch of frames efficiently.

        FastViT supports true batching for improved throughput.
        """
        from mlx_video_processor.types import FrameResult
        import time

        if not self._is_loaded:
            self.load_model()

        start_time = time.perf_counter()

        # Preprocess all frames
        preprocessed = [self.preprocess(f) for f in frames]
        batch = mx.concatenate(preprocessed, axis=0)

        # Batch forward pass
        outputs = self.forward(batch)
        mx.eval(outputs)

        total_time = (time.perf_counter() - start_time) * 1000
        time_per_frame = total_time / len(frames)

        # Postprocess each result
        results = []
        for i, (frame, idx, ts) in enumerate(zip(frames, frame_indices, timestamps_ms)):
            output_i = outputs[i:i+1]
            classification = self.postprocess(output_i, (frame.shape[0], frame.shape[1]))

            results.append(FrameResult(
                frame_index=idx,
                timestamp_ms=ts,
                classification=classification,
                processing_time_ms=time_per_frame,
            ))

        return results
