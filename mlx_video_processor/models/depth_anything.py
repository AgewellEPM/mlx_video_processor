"""
Depth Anything V2 Monocular Depth Estimation Processor.

Depth Anything V2 estimates depth from single RGB images using a
teacher-student pipeline trained on synthetic and real data.

Model specifications:
- Parameters: 24.8M (Small variant)
- Input: Variable size RGB images
- Output: Relative depth maps
- Architecture: DINOv2 encoder + DPT decoder
"""

from typing import Any, Optional
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_video_processor.models.base import BaseModel, ModelRegistry
from mlx_video_processor.types import (
    DepthResult,
    ModelType,
    PrecisionType,
)


class PatchEmbed(nn.Module):
    """Patch embedding for Vision Transformer."""

    def __init__(self, img_size: int = 518, patch_size: int = 14, in_channels: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) - MLX native NHWC format
        x = self.proj(x)  # (B, H/P, W/P, embed_dim)
        # Flatten spatial dimensions: (B, H/P, W/P, embed_dim) -> (B, N, embed_dim)
        x = x.reshape(x.shape[0], -1, x.shape[3])  # (B, N, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int = 6, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """Feed-forward MLP block."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Vision Transformer block."""

    def __init__(self, embed_dim: int, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DINOv2Encoder(nn.Module):
    """
    DINOv2-style Vision Transformer encoder.

    Simplified version for Depth Anything V2 Small model.
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = (img_size // patch_size) ** 2

        # Learnable CLS token and position embeddings
        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.pos_embed = mx.zeros((1, num_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = [TransformerBlock(embed_dim, num_heads) for _ in range(depth)]
        self.norm = nn.LayerNorm(embed_dim)

        # Indices for intermediate features (for DPT decoder)
        self.out_indices = [2, 5, 8, 11]

    def __call__(self, x: mx.array) -> list[mx.array]:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = mx.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        x = mx.concatenate([cls_tokens, x], axis=1)

        # Add position embeddings
        x = x + self.pos_embed

        # Collect intermediate features
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.out_indices:
                features.append(self.norm(x))

        return features


class ConvBlock(nn.Module):
    """Convolution block for DPT decoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        return x


class FeatureFusion(nn.Module):
    """Feature fusion module for DPT decoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array, skip: Optional[mx.array] = None) -> mx.array:
        if skip is not None:
            x = x + skip
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        return x


class DPTDecoder(nn.Module):
    """
    DPT (Dense Prediction Transformer) decoder.

    Fuses multi-scale features from the encoder for dense prediction.
    """

    def __init__(self, embed_dim: int = 384, features: int = 256, out_channels: int = 1):
        super().__init__()

        # Project encoder features to common dimension
        self.proj1 = nn.Conv2d(embed_dim, features, kernel_size=1)
        self.proj2 = nn.Conv2d(embed_dim, features, kernel_size=1)
        self.proj3 = nn.Conv2d(embed_dim, features, kernel_size=1)
        self.proj4 = nn.Conv2d(embed_dim, features, kernel_size=1)

        # Refinement blocks
        self.refine1 = FeatureFusion(features)
        self.refine2 = FeatureFusion(features)
        self.refine3 = FeatureFusion(features)
        self.refine4 = FeatureFusion(features)

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def _reshape_features(self, x: mx.array, h: int, w: int) -> mx.array:
        """Reshape transformer features to spatial format."""
        B, N, C = x.shape
        # Remove CLS token
        x = x[:, 1:]
        # Reshape to spatial in NHWC format
        x = x.reshape(B, h, w, C)  # (B, H, W, C)
        return x

    def __call__(self, features: list[mx.array], img_size: tuple[int, int]) -> mx.array:
        """
        Args:
            features: List of 4 feature maps from encoder
            img_size: Original input image size (H, W)
        """
        h = img_size[0] // 14  # Patch size
        w = img_size[1] // 14

        # Reshape features to spatial format
        f1 = self._reshape_features(features[0], h, w)
        f2 = self._reshape_features(features[1], h, w)
        f3 = self._reshape_features(features[2], h, w)
        f4 = self._reshape_features(features[3], h, w)

        # Project to common dimension
        f1 = self.proj1(f1)
        f2 = self.proj2(f2)
        f3 = self.proj3(f3)
        f4 = self.proj4(f4)

        # Progressive fusion (coarse to fine)
        # In NHWC format, upsample H (axis 1) and W (axis 2)
        x = self.refine4(f4)
        x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)  # Upsample 2x
        x = self.refine3(x, f3)
        x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
        x = self.refine2(x, f2)
        x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
        x = self.refine1(x, f1)

        # Final upsampling
        x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)

        # Prediction head
        for layer in self.head.layers:
            x = layer(x)
            if not isinstance(layer, nn.Conv2d) or layer != self.head.layers[-1]:
                x = nn.relu(x)

        return x


class DepthAnythingV2Model(nn.Module):
    """
    Depth Anything V2 model.

    Combines DINOv2 encoder with DPT decoder for monocular depth estimation.
    """

    def __init__(self, img_size: int = 518, embed_dim: int = 384, encoder_depth: int = 12):
        super().__init__()
        self.img_size = img_size
        self.encoder = DINOv2Encoder(img_size, embed_dim=embed_dim, depth=encoder_depth)
        self.decoder = DPTDecoder(embed_dim=embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        # x is in NHWC format (B, H, W, C)
        h, w = x.shape[1], x.shape[2]

        # Encoder
        features = self.encoder(x)

        # Decoder
        depth = self.decoder(features, (h, w))

        return depth


@ModelRegistry.register("depth")
@ModelRegistry.register("depth_anything")
@ModelRegistry.register("depth_anything_v2")
class DepthAnythingV2Processor(BaseModel):
    """
    Depth Anything V2 processor for monocular depth estimation.

    Estimates relative depth from single RGB images.
    """

    model_name = "Depth-Anything-V2-Small"
    model_type = ModelType.DEPTH_ESTIMATION
    default_input_size = (518, 518)
    supports_batching = False

    # Normalization parameters (ImageNet)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        precision: PrecisionType = PrecisionType.FLOAT16,
        device: Optional[str] = None,
        compile_model: bool = True,
        normalize_output: bool = True,
    ):
        """
        Initialize Depth Anything V2 processor.

        Args:
            precision: Weight precision
            device: Device to run on
            compile_model: Whether to compile forward pass
            normalize_output: Whether to normalize depth to [0, 1]
        """
        super().__init__(precision, device, compile_model)
        self.normalize_output = normalize_output

    def load_model(self) -> None:
        """Load Depth Anything V2 model weights."""
        if self._is_loaded:
            return

        self._model = DepthAnythingV2Model(
            img_size=self.default_input_size[0],
            embed_dim=384,  # Small model
            encoder_depth=12,
        )

        # In production, load pretrained weights:
        # weights = mx.load("depth_anything_v2_small.safetensors")
        # self._model.load_weights(list(weights.items()))

        if self.compile_model:
            self._compiled_forward = mx.compile(self._model.__call__)

        self._is_loaded = True

    def preprocess(self, frame: np.ndarray) -> mx.array:
        """
        Preprocess frame for Depth Anything input.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            Preprocessed tensor in NHWC format (1, 518, 518, 3)
        """
        import cv2

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to input size
        h, w = self.default_input_size
        resized = cv2.resize(rgb, (w, h))

        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.MEAN) / self.STD

        # Convert to (B, H, W, C) - MLX native NHWC format
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
    ) -> DepthResult:
        """
        Postprocess Depth Anything output to depth result.

        Args:
            output: Raw depth prediction in NHWC format (1, H, W, 1)
            original_size: Original frame size (H, W)

        Returns:
            DepthResult with depth map
        """
        import cv2

        # Remove batch and channel dimensions - output is (B, H, W, C)
        depth = np.array(output[0, :, :, 0])  # (H, W)

        # Resize to original size
        depth = cv2.resize(depth, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

        min_depth = float(depth.min())
        max_depth = float(depth.max())

        # Optionally normalize to [0, 1]
        if self.normalize_output and max_depth > min_depth:
            depth = (depth - min_depth) / (max_depth - min_depth)

        return DepthResult(
            depth_map=mx.array(depth),
            min_depth=min_depth,
            max_depth=max_depth,
            is_metric=False,  # Depth Anything produces relative depth
        )

    def get_colored_depth(self, result: DepthResult, colormap: str = "magma") -> np.ndarray:
        """
        Convert depth map to colored visualization.

        Args:
            result: DepthResult from postprocess
            colormap: Matplotlib colormap name

        Returns:
            RGB colored depth map (H, W, 3)
        """
        import cv2

        depth = np.array(result.depth_map)

        # Normalize if not already
        if not self.normalize_output:
            depth = result.get_normalized()
            depth = np.array(depth)

        # Apply colormap (OpenCV expects 0-255 uint8)
        depth_uint8 = (depth * 255).astype(np.uint8)

        # Use OpenCV colormaps (COLORMAP_MAGMA is 13)
        colormap_dict = {
            "magma": cv2.COLORMAP_MAGMA,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "plasma": cv2.COLORMAP_PLASMA,
            "inferno": cv2.COLORMAP_INFERNO,
            "jet": cv2.COLORMAP_JET,
            "turbo": cv2.COLORMAP_TURBO,
        }

        cm = colormap_dict.get(colormap, cv2.COLORMAP_MAGMA)
        colored = cv2.applyColorMap(depth_uint8, cm)

        # Convert BGR to RGB
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
