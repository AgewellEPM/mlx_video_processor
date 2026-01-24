"""
DeepLabV3 Semantic Segmentation Processor.

DeepLabV3 performs pixel-level semantic segmentation using atrous convolutions
and Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context.

Model specifications:
- Input: Variable size RGB images
- Output: Per-pixel class predictions (21 classes for PASCAL VOC)
- Architecture: ResNet backbone + ASPP + decoder
"""

from typing import Any, Optional
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_video_processor.models.base import BaseModel, ModelRegistry
from mlx_video_processor.types import (
    SegmentationResult,
    ModelType,
    PrecisionType,
)


# PASCAL VOC class names (21 classes including background)
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Class colors for visualization (RGB)
VOC_COLORS = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
    (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
    (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
    (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
    (0, 64, 128)
]


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        # x is in NHWC format - no transpose needed
        x = self.conv(x)
        x = self.bn(x)
        return nn.relu(x)


class Bottleneck(nn.Module):
    """ResNet bottleneck block with optional dilation."""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(out_channels)

        # Apply dilation to 3x3 conv
        padding = dilation
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm(out_channels * self.expansion)

        self.downsample = downsample

    def __call__(self, x: mx.array) -> mx.array:
        # x is in NHWC format - no transposes needed
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return nn.relu(out)


class ResNetBackbone(nn.Module):
    """
    Modified ResNet backbone for DeepLabV3.

    Uses dilated convolutions in later stages to maintain spatial resolution.
    """

    def __init__(self, output_stride: int = 16):
        super().__init__()

        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)
        # Max pooling handled separately

        # Determine dilation rates based on output stride
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]

        # Build stages
        self.layer1 = self._make_layer(64, 64, 3, stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(256, 128, 4, stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(512, 256, 6, stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(1024, 512, 3, stride=strides[3], dilation=dilations[3])

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> list:
        downsample = None
        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm(out_channels * Bottleneck.expansion),
            )

        layers = [Bottleneck(in_channels, out_channels, stride, dilation, downsample)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels * Bottleneck.expansion, out_channels, dilation=dilation))

        return layers

    def __call__(self, x: mx.array) -> mx.array:
        # x is in NHWC format - no transposes needed
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.relu(x)

        # Max pooling (stride 2) - operates on spatial dimensions (H, W) in NHWC
        # NHWC: (B, H, W, C) -> pool over H and W dimensions (axis 1 and 2)
        x = mx.max(x.reshape(x.shape[0], x.shape[1]//2, 2, x.shape[2]//2, 2, x.shape[3]), axis=(2, 4))

        # ResNet stages
        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)

        return x


class ASPPConv(nn.Module):
    """ASPP convolution module with dilation."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        padding = dilation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=False)
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        # x is in NHWC format - no transpose needed
        x = self.conv(x)
        x = self.bn(x)
        return nn.relu(x)


class ASPPPooling(nn.Module):
    """ASPP global pooling module."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        # x is in NHWC format (B, H, W, C)
        size = x.shape[1:3]  # (H, W)

        # Global average pooling - pool over spatial dimensions (H, W)
        x = mx.mean(x, axis=(1, 2), keepdims=True)
        x = self.conv(x)
        x = self.bn(x)
        x = nn.relu(x)

        # Upsample to original size - repeat over H and W dimensions
        x = mx.repeat(mx.repeat(x, size[0], axis=1), size[1], axis=2)
        return x


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.

    Captures multi-scale context using parallel dilated convolutions
    with different rates plus global pooling.
    """

    def __init__(self, in_channels: int, out_channels: int = 256, atrous_rates: tuple = (6, 12, 18)):
        super().__init__()

        modules = []

        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm(out_channels),
        ))

        # Dilated convolutions
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # Global pooling
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = modules

        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm(out_channels),
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x is in NHWC format (B, H, W, C)
        res = []
        for conv in self.convs:
            if isinstance(conv, nn.Sequential):
                # Handle 1x1 conv
                out = conv[0](x)
                out = conv[1](out)
                out = nn.relu(out)
                res.append(out)
            else:
                res.append(conv(x))

        # Concatenate along channel dimension (axis 3 in NHWC)
        x = mx.concatenate(res, axis=3)

        # Project
        x = self.project[0](x)
        x = self.project[1](x)
        return nn.relu(x)


class DeepLabV3Model(nn.Module):
    """
    DeepLabV3 semantic segmentation model.

    Combines ResNet backbone with ASPP module for multi-scale segmentation.
    """

    def __init__(self, num_classes: int = 21, output_stride: int = 16):
        super().__init__()

        self.backbone = ResNetBackbone(output_stride=output_stride)
        self.aspp = ASPP(2048, 256)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        # x is in NHWC format (B, H, W, C)
        input_shape = x.shape[1:3]  # (H, W)

        # Extract features
        x = self.backbone(x)

        # ASPP
        x = self.aspp(x)

        # Classification
        x = self.classifier(x)

        # Bilinear upsampling to input size - operate on H and W dimensions (axis 1, 2)
        scale_h = input_shape[0] // x.shape[1]
        scale_w = input_shape[1] // x.shape[2]
        x = mx.repeat(mx.repeat(x, scale_h, axis=1), scale_w, axis=2)

        return x


@ModelRegistry.register("deeplab")
@ModelRegistry.register("deeplabv3")
class DeepLabV3Processor(BaseModel):
    """
    DeepLabV3 processor for semantic segmentation.

    Assigns class labels to each pixel in the image.
    """

    model_name = "DeepLabV3"
    model_type = ModelType.SEMANTIC_SEGMENTATION
    default_input_size = (513, 513)
    supports_batching = False

    def __init__(
        self,
        precision: PrecisionType = PrecisionType.FLOAT16,
        device: Optional[str] = None,
        compile_model: bool = True,
        class_names: Optional[list[str]] = None,
        output_stride: int = 16,
    ):
        """
        Initialize DeepLabV3 processor.

        Args:
            precision: Weight precision
            device: Device to run on
            compile_model: Whether to compile forward pass
            class_names: Custom class names (defaults to PASCAL VOC)
            output_stride: Backbone output stride (8 or 16)
        """
        super().__init__(precision, device, compile_model)
        self.class_names = class_names or VOC_CLASSES
        self.num_classes = len(self.class_names)
        self.output_stride = output_stride
        self.class_colors = VOC_COLORS

    def load_model(self) -> None:
        """Load DeepLabV3 model weights."""
        if self._is_loaded:
            return

        self._model = DeepLabV3Model(
            num_classes=self.num_classes,
            output_stride=self.output_stride,
        )

        # In production, load pretrained weights:
        # weights = mx.load("deeplabv3_resnet.safetensors")
        # self._model.load_weights(list(weights.items()))

        if self.compile_model:
            self._compiled_forward = mx.compile(self._model.__call__)

        self._is_loaded = True

    def preprocess(self, frame: np.ndarray) -> mx.array:
        """
        Preprocess frame for DeepLabV3 input.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            Preprocessed tensor in NHWC format (1, H, W, 3)
        """
        import cv2

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to input size
        h, w = self.default_input_size
        resized = cv2.resize(rgb, (w, h))

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - mean) / std

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
    ) -> SegmentationResult:
        """
        Postprocess DeepLabV3 output to segmentation result.

        Args:
            output: Model logits in NHWC format (1, H, W, num_classes)
            original_size: Original frame size (H, W)

        Returns:
            SegmentationResult with class mask
        """
        import cv2

        # Get class predictions - output is now (B, H, W, C)
        output = np.array(output[0])  # (H, W, C)

        # Argmax to get class labels - along last axis (channels)
        mask = np.argmax(output, axis=-1).astype(np.uint8)  # (H, W)

        # Get confidence map (softmax probabilities)
        probs = np.exp(output) / np.sum(np.exp(output), axis=-1, keepdims=True)
        confidence = np.max(probs, axis=-1)

        # Resize to original size
        mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        confidence = cv2.resize(confidence, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

        # Get unique classes in mask
        unique_ids = np.unique(mask).tolist()
        class_names = [self.class_names[i] for i in unique_ids]

        return SegmentationResult(
            mask=mx.array(mask),
            class_ids=unique_ids,
            class_names=class_names,
            confidence_map=mx.array(confidence),
        )

    def get_color_mask(self, result: SegmentationResult) -> np.ndarray:
        """
        Convert segmentation mask to color image.

        Args:
            result: SegmentationResult from postprocess

        Returns:
            RGB color mask (H, W, 3)
        """
        mask = np.array(result.mask)
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id in result.class_ids:
            color = self.class_colors[class_id] if class_id < len(self.class_colors) else (128, 128, 128)
            color_mask[mask == class_id] = color

        return color_mask
