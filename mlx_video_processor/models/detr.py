"""
DETR ResNet50 Detection/Segmentation Processor.

DETR (DEtection TRansformer) uses a transformer encoder-decoder architecture
with bipartite matching loss for end-to-end object detection and segmentation.

Model specifications:
- Parameters: 43M
- Input: Variable size RGB images
- Output: Object detections with optional panoptic segmentation
- Architecture: ResNet-50 backbone + Transformer encoder-decoder
"""

from typing import Any, Optional
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_video_processor.models.base import BaseModel, ModelRegistry
from mlx_video_processor.types import (
    BoundingBox,
    DetectionResult,
    SegmentationResult,
    ModelType,
    PrecisionType,
)


# COCO class names (91 classes for DETR)
COCO_CLASSES = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class ConvNormAct(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.norm = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        # x is in NHWC format - no transpose needed
        x = self.conv(x)
        x = self.norm(x)
        return nn.relu(x)


class ResNetBottleneck(nn.Module):
    """ResNet bottleneck block."""

    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm(out_channels * self.expansion)
        self.downsample = downsample

    def __call__(self, x: mx.array) -> mx.array:
        # x is in NHWC format - no transposes needed
        identity = x

        out = self.conv1(x)
        out = nn.relu(self.bn1(out))
        out = self.conv2(out)
        out = nn.relu(self.bn2(out))
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return nn.relu(out + identity)


class ResNet50Backbone(nn.Module):
    """ResNet-50 backbone for DETR."""

    def __init__(self):
        super().__init__()

        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)

        # Build stages
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int) -> list:
        downsample = None
        if stride != 1 or in_channels != out_channels * ResNetBottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * ResNetBottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm(out_channels * ResNetBottleneck.expansion),
            )

        layers = [ResNetBottleneck(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(ResNetBottleneck(out_channels * ResNetBottleneck.expansion, out_channels))

        return layers

    def __call__(self, x: mx.array) -> mx.array:
        # x is in NHWC format - no transposes needed
        x = self.conv1(x)
        x = nn.relu(self.bn1(x))

        # Max pooling - operates on spatial dimensions (H, W) in NHWC
        # NHWC: (B, H, W, C) -> pool over H and W dimensions (axis 1 and 2)
        x = mx.max(x.reshape(x.shape[0], x.shape[1]//2, 2, x.shape[2]//2, 2, x.shape[3]), axis=(2, 4))

        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)

        return x


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for DETR."""

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def __call__(self, x: mx.array) -> mx.array:
        # x is in NHWC format (B, H, W, C)
        _, h, w, _ = x.shape

        # Create position indices
        y_embed = mx.arange(h, dtype=mx.float32).reshape(h, 1)
        x_embed = mx.arange(w, dtype=mx.float32).reshape(1, w)

        y_embed = mx.broadcast_to(y_embed, (h, w))
        x_embed = mx.broadcast_to(x_embed, (h, w))

        # Normalize
        y_embed = y_embed / (h + 1e-6)
        x_embed = x_embed / (w + 1e-6)

        dim_t = mx.arange(self.num_pos_feats, dtype=mx.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = mx.concatenate([mx.sin(pos_x[..., 0::2]), mx.cos(pos_x[..., 1::2])], axis=-1)
        pos_y = mx.concatenate([mx.sin(pos_y[..., 0::2]), mx.cos(pos_y[..., 1::2])], axis=-1)

        pos = mx.concatenate([pos_y, pos_x], axis=-1)  # (H, W, C)

        return pos


class TransformerEncoderLayer(nn.Module):
    """DETR transformer encoder layer."""

    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def __call__(self, src: mx.array, pos: mx.array) -> mx.array:
        q = k = src + pos
        src2 = self.self_attn(q, k, src)
        src = src + src2
        src = self.norm1(src)

        src2 = self.linear2(nn.relu(self.linear1(src)))
        src = src + src2
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):
    """DETR transformer decoder layer."""

    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, nhead)
        self.cross_attn = nn.MultiHeadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def __call__(self, tgt: mx.array, memory: mx.array,
                 query_pos: mx.array, pos: mx.array) -> mx.array:
        # Self attention
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, tgt)
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        # Cross attention
        q = tgt + query_pos
        k = memory + pos
        tgt2 = self.cross_attn(q, k, memory)
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(nn.relu(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)

        return tgt


class DETRTransformer(nn.Module):
    """DETR transformer encoder-decoder."""

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, num_queries: int = 100):
        super().__init__()

        self.d_model = d_model
        self.num_queries = num_queries

        # Encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_encoder_layers)
        ]

        # Decoder layers
        self.decoder_layers = [
            TransformerDecoderLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_decoder_layers)
        ]

        # Learnable query embeddings
        self.query_embed = mx.zeros((num_queries, d_model))

    def __call__(self, src: mx.array, pos: mx.array) -> mx.array:
        # src is in NHWC format (B, H, W, C)
        B = src.shape[0]

        # Flatten spatial dimensions - NHWC: (B, H, W, C) -> (B, HW, C)
        src = src.reshape(B, -1, src.shape[3])  # (B, HW, C)
        pos = pos.reshape(-1, pos.shape[2])  # (HW, C)
        pos = mx.broadcast_to(pos, (B, pos.shape[0], pos.shape[1]))  # (B, HW, C)

        # Encoder
        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory, pos)

        # Decoder
        query_embed = mx.broadcast_to(self.query_embed, (B, self.num_queries, self.d_model))
        tgt = mx.zeros_like(query_embed)

        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, query_embed, pos)

        return tgt


class DETRModel(nn.Module):
    """
    Complete DETR model for object detection.

    Combines ResNet-50 backbone with transformer encoder-decoder
    and prediction heads for bounding boxes and classes.
    """

    def __init__(self, num_classes: int = 91, num_queries: int = 100,
                 hidden_dim: int = 256, nheads: int = 8):
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Backbone
        self.backbone = ResNet50Backbone()
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        # Positional encoding
        self.pos_enc = PositionalEncoding2D(hidden_dim // 2)

        # Transformer
        self.transformer = DETRTransformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_queries=num_queries,
        )

        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 4),
        )

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        # Backbone features
        features = self.backbone(x)
        features = self.input_proj(features)

        # Positional encoding
        pos = self.pos_enc(features)

        # Transformer
        hs = self.transformer(features, pos)

        # Predictions
        outputs_class = self.class_embed(hs)
        outputs_bbox = mx.sigmoid(self._apply_bbox_mlp(hs))

        return outputs_class, outputs_bbox

    def _apply_bbox_mlp(self, x: mx.array) -> mx.array:
        """Apply bbox MLP with ReLU activations."""
        for i, layer in enumerate(self.bbox_embed.layers):
            x = layer(x)
            if i < len(self.bbox_embed.layers) - 1:
                x = nn.relu(x)
        return x


@ModelRegistry.register("detr")
class DETRProcessor(BaseModel):
    """
    DETR processor for object detection.

    Uses transformer-based detection with set prediction.
    """

    model_name = "DETR-ResNet50"
    model_type = ModelType.OBJECT_DETECTION
    default_input_size = (800, 800)
    supports_batching = False

    # ImageNet normalization
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        precision: PrecisionType = PrecisionType.FLOAT16,
        device: Optional[str] = None,
        compile_model: bool = True,
        confidence_threshold: float = 0.7,
        class_names: Optional[list[str]] = None,
        num_queries: int = 100,
    ):
        """
        Initialize DETR processor.

        Args:
            precision: Weight precision
            device: Device to run on
            compile_model: Whether to compile forward pass
            confidence_threshold: Minimum confidence for detections
            class_names: Custom class names (defaults to COCO)
            num_queries: Number of object queries
        """
        super().__init__(precision, device, compile_model)
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names or COCO_CLASSES
        self.num_classes = len(self.class_names)
        self.num_queries = num_queries

    def load_model(self) -> None:
        """Load DETR model weights."""
        if self._is_loaded:
            return

        self._model = DETRModel(
            num_classes=self.num_classes,
            num_queries=self.num_queries,
        )

        # In production, load pretrained weights:
        # weights = mx.load("detr_resnet50.safetensors")
        # self._model.load_weights(list(weights.items()))

        if self.compile_model:
            self._compiled_forward = mx.compile(self._model.__call__)

        self._is_loaded = True

    def preprocess(self, frame: np.ndarray) -> mx.array:
        """
        Preprocess frame for DETR input.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            Preprocessed tensor in NHWC format (1, H, W, 3)
        """
        import cv2

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Store original size for postprocessing
        self._original_size = (frame.shape[0], frame.shape[1])

        # Resize maintaining aspect ratio with max dimension
        h, w = rgb.shape[:2]
        max_size = max(self.default_input_size)

        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Ensure dimensions are divisible by 32 for the backbone
        new_h = (new_h // 32) * 32
        new_w = (new_w // 32) * 32

        resized = cv2.resize(rgb, (new_w, new_h))
        self._resized_size = (new_h, new_w)

        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.MEAN) / self.STD

        # Convert to (B, H, W, C) - MLX native NHWC format
        x = mx.array(normalized, dtype=self.dtype)
        x = mx.expand_dims(x, axis=0)

        return x

    def forward(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Run forward pass."""
        return self._model(x)

    def postprocess(
        self,
        output: tuple[mx.array, mx.array],
        original_size: tuple[int, int],
    ) -> DetectionResult:
        """
        Postprocess DETR output to detection results.

        Args:
            output: (class_logits, bbox_predictions)
            original_size: Original frame size (H, W)

        Returns:
            DetectionResult with bounding boxes
        """
        pred_logits, pred_boxes = output

        # Get probabilities (remove no-object class for confidence)
        probs = np.array(mx.softmax(pred_logits, axis=-1)[0])  # (num_queries, num_classes+1)

        # Filter out no-object predictions (last class)
        scores = probs[:, :-1]  # (num_queries, num_classes)
        max_scores = np.max(scores, axis=-1)
        max_classes = np.argmax(scores, axis=-1)

        # Filter by confidence
        keep = max_scores > self.confidence_threshold

        pred_boxes = np.array(pred_boxes[0])  # (num_queries, 4)

        # Extract kept predictions
        boxes_keep = pred_boxes[keep]
        scores_keep = max_scores[keep]
        classes_keep = max_classes[keep]

        if len(boxes_keep) == 0:
            return DetectionResult.empty()

        # Convert from center format to corner format
        # DETR outputs: (cx, cy, w, h) normalized [0, 1]
        bounding_boxes = []
        for box, score, cls_id in zip(boxes_keep, scores_keep, classes_keep):
            cx, cy, w, h = box
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Clip to valid range
            x1, x2 = np.clip([x1, x2], 0, 1)
            y1, y2 = np.clip([y1, y2], 0, 1)

            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"

            bounding_boxes.append(BoundingBox(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(score),
                class_id=int(cls_id),
                class_name=class_name,
            ))

        return DetectionResult(
            boxes=bounding_boxes,
            num_detections=len(bounding_boxes),
        )
