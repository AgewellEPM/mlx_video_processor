"""
YOLOv3 Object Detection Processor.

YOLOv3 is a real-time object detection model that predicts bounding boxes
and class probabilities in a single forward pass using a DarkNet-53 backbone.

Model specifications:
- Objects detected: 80 COCO classes
- Input: 416x416 RGB images (or 608x608 for higher accuracy)
- Output: Multi-scale predictions at 3 different resolutions
"""

from typing import Any, Optional
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_video_processor.models.base import BaseModel, ModelRegistry
from mlx_video_processor.types import (
    BoundingBox,
    DetectionResult,
    ModelType,
    PrecisionType,
)


# COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# YOLOv3 anchor boxes for three scales
ANCHORS = [
    [(116, 90), (156, 198), (373, 326)],  # Scale 1 (large objects)
    [(30, 61), (62, 45), (59, 119)],       # Scale 2 (medium objects)
    [(10, 13), (16, 30), (33, 23)],        # Scale 3 (small objects)
]


class ConvBNLeaky(nn.Module):
    """Convolution + BatchNorm + LeakyReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = self.bn(x)
        x = nn.leaky_relu(x, negative_slope=0.1)
        return x


class DarkNetBlock(nn.Module):
    """DarkNet residual block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNLeaky(channels, channels // 2, kernel_size=1, padding=0)
        self.conv2 = ConvBNLeaky(channels // 2, channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class DarkNet53(nn.Module):
    """
    DarkNet-53 backbone for YOLOv3.

    53 convolutional layers with residual connections.
    Returns feature maps at three scales for multi-scale detection.
    """

    def __init__(self):
        super().__init__()

        # Initial convolution
        self.conv1 = ConvBNLeaky(3, 32, kernel_size=3, padding=1)

        # Downsample + residual blocks
        # Stage 1: 32 -> 64
        self.down1 = ConvBNLeaky(32, 64, kernel_size=3, stride=2, padding=1)
        self.stage1 = [DarkNetBlock(64) for _ in range(1)]

        # Stage 2: 64 -> 128
        self.down2 = ConvBNLeaky(64, 128, kernel_size=3, stride=2, padding=1)
        self.stage2 = [DarkNetBlock(128) for _ in range(2)]

        # Stage 3: 128 -> 256 (output for small objects)
        self.down3 = ConvBNLeaky(128, 256, kernel_size=3, stride=2, padding=1)
        self.stage3 = [DarkNetBlock(256) for _ in range(8)]

        # Stage 4: 256 -> 512 (output for medium objects)
        self.down4 = ConvBNLeaky(256, 512, kernel_size=3, stride=2, padding=1)
        self.stage4 = [DarkNetBlock(512) for _ in range(8)]

        # Stage 5: 512 -> 1024 (output for large objects)
        self.down5 = ConvBNLeaky(512, 1024, kernel_size=3, stride=2, padding=1)
        self.stage5 = [DarkNetBlock(1024) for _ in range(4)]

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        x = self.conv1(x)

        x = self.down1(x)
        for block in self.stage1:
            x = block(x)

        x = self.down2(x)
        for block in self.stage2:
            x = block(x)

        x = self.down3(x)
        for block in self.stage3:
            x = block(x)
        out3 = x  # 52x52 for small objects

        x = self.down4(x)
        for block in self.stage4:
            x = block(x)
        out2 = x  # 26x26 for medium objects

        x = self.down5(x)
        for block in self.stage5:
            x = block(x)
        out1 = x  # 13x13 for large objects

        return out1, out2, out3


class YOLOHead(nn.Module):
    """YOLO detection head for a single scale."""

    def __init__(self, in_channels: int, num_classes: int = 80, num_anchors: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # 5 = 4 (box coords) + 1 (objectness)
        out_channels = num_anchors * (5 + num_classes)

        self.conv1 = ConvBNLeaky(in_channels, in_channels // 2, kernel_size=1, padding=0)
        self.conv2 = ConvBNLeaky(in_channels // 2, in_channels, kernel_size=3, padding=1)
        self.conv3 = ConvBNLeaky(in_channels, in_channels // 2, kernel_size=1, padding=0)
        self.conv4 = ConvBNLeaky(in_channels // 2, in_channels, kernel_size=3, padding=1)
        self.conv5 = ConvBNLeaky(in_channels, in_channels // 2, kernel_size=1, padding=0)

        # Final prediction layers
        self.conv6 = ConvBNLeaky(in_channels // 2, in_channels, kernel_size=3, padding=1)
        self.pred = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        route = self.conv5(x)

        x = self.conv6(route)
        pred = self.pred(x)

        return pred, route


class YOLOv3Model(nn.Module):
    """
    Complete YOLOv3 model.

    Combines DarkNet-53 backbone with FPN-style feature pyramid
    and three detection heads for multi-scale object detection.
    """

    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes

        # Backbone
        self.backbone = DarkNet53()

        # Detection heads
        self.head1 = YOLOHead(1024, num_classes)  # Large objects
        self.head2 = YOLOHead(512, num_classes)   # Medium objects
        self.head3 = YOLOHead(256, num_classes)   # Small objects

        # Upsampling and merging layers
        self.upsample1_conv = ConvBNLeaky(512, 256, kernel_size=1, padding=0)
        self.upsample2_conv = ConvBNLeaky(256, 128, kernel_size=1, padding=0)

        # Merge convolutions
        self.merge1 = ConvBNLeaky(768, 512, kernel_size=1, padding=0)
        self.merge2 = ConvBNLeaky(384, 256, kernel_size=1, padding=0)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        # x is in NHWC format (B, H, W, C)

        # Backbone features at three scales
        feat1, feat2, feat3 = self.backbone(x)

        # Detection at scale 1 (large objects, 13x13)
        pred1, route1 = self.head1(feat1)

        # Upsample and merge for scale 2
        up1 = self.upsample1_conv(route1)
        # Simple nearest neighbor upsampling - upsample H and W dimensions (axis 1, 2)
        up1 = mx.repeat(mx.repeat(up1, 2, axis=1), 2, axis=2)
        # Concatenate along channel dimension (axis 3 in NHWC)
        merged1 = mx.concatenate([up1, feat2], axis=3)
        merged1 = self.merge1(merged1)

        # Detection at scale 2 (medium objects, 26x26)
        pred2, route2 = self.head2(merged1)

        # Upsample and merge for scale 3
        up2 = self.upsample2_conv(route2)
        up2 = mx.repeat(mx.repeat(up2, 2, axis=1), 2, axis=2)
        merged2 = mx.concatenate([up2, feat3], axis=3)
        merged2 = self.merge2(merged2)

        # Detection at scale 3 (small objects, 52x52)
        pred3, _ = self.head3(merged2)

        return pred1, pred2, pred3


@ModelRegistry.register("yolo")
@ModelRegistry.register("yolov3")
class YOLOv3Processor(BaseModel):
    """
    YOLOv3 processor for real-time object detection.

    Detects 80 COCO classes with bounding boxes and confidence scores.
    """

    model_name = "YOLOv3"
    model_type = ModelType.OBJECT_DETECTION
    default_input_size = (416, 416)
    supports_batching = False  # Sequential for simplicity

    def __init__(
        self,
        precision: PrecisionType = PrecisionType.FLOAT16,
        device: Optional[str] = None,
        compile_model: bool = True,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        class_names: Optional[list[str]] = None,
    ):
        """
        Initialize YOLOv3 processor.

        Args:
            precision: Weight precision
            device: Device to run on
            compile_model: Whether to compile forward pass
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
            class_names: Custom class names (defaults to COCO)
        """
        super().__init__(precision, device, compile_model)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.class_names = class_names or COCO_CLASSES
        self.num_classes = len(self.class_names)
        self.anchors = ANCHORS

    def load_model(self) -> None:
        """Load YOLOv3 model weights."""
        if self._is_loaded:
            return

        self._model = YOLOv3Model(num_classes=self.num_classes)

        # In production, load pretrained weights:
        # weights = mx.load("yolov3.safetensors")
        # self._model.load_weights(list(weights.items()))

        if self.compile_model:
            self._compiled_forward = mx.compile(self._model.__call__)

        self._is_loaded = True

    def preprocess(self, frame: np.ndarray) -> mx.array:
        """
        Preprocess frame for YOLOv3 input.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            Preprocessed tensor (1, 3, 416, 416)
        """
        import cv2

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize with letterbox to maintain aspect ratio
        h, w = frame.shape[:2]
        target_h, target_w = self.default_input_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(rgb, (new_w, new_h))

        # Create letterbox image
        letterbox = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        letterbox[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

        # Store letterbox info for postprocessing
        self._letterbox_info = {
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "original_size": (h, w),
        }

        # Normalize to [0, 1]
        normalized = letterbox.astype(np.float32) / 255.0

        # Convert to (B, H, W, C) - MLX native format
        x = mx.array(normalized, dtype=self.dtype)
        x = mx.expand_dims(x, axis=0)

        return x

    def forward(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Run forward pass."""
        return self._model(x)

    def _decode_predictions(
        self,
        pred: mx.array,
        anchors: list[tuple[int, int]],
        stride: int,
    ) -> np.ndarray:
        """
        Decode YOLO predictions to bounding boxes.

        Args:
            pred: Raw predictions in NHWC format (1, H, W, num_anchors*(5+num_classes))
            anchors: Anchor boxes for this scale
            stride: Stride for this detection scale

        Returns:
            Decoded boxes array (N, 6) with [x1, y1, x2, y2, conf, class]
        """
        pred = np.array(pred)
        batch_size, grid_h, grid_w, _ = pred.shape
        num_anchors = len(anchors)

        # Reshape to (B, H, W, num_anchors, 5+num_classes)
        pred = pred.reshape(batch_size, grid_h, grid_w, num_anchors, 5 + self.num_classes)
        # Reorder to (B, num_anchors, H, W, 5+C) for easier processing
        pred = pred.transpose(0, 3, 1, 2, 4)  # (B, A, H, W, 5+C)

        # Extract components
        box_xy = 1 / (1 + np.exp(-pred[..., :2]))  # Sigmoid
        box_wh = pred[..., 2:4]
        objectness = 1 / (1 + np.exp(-pred[..., 4:5]))
        class_probs = 1 / (1 + np.exp(-pred[..., 5:]))

        # Create grid
        grid_y, grid_x = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
        grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
        grid = grid.reshape(1, 1, grid_h, grid_w, 2)

        # Decode boxes
        box_xy = (box_xy + grid) * stride
        anchors_array = np.array(anchors, dtype=np.float32).reshape(1, num_anchors, 1, 1, 2)
        box_wh = np.exp(box_wh) * anchors_array

        # Convert to x1, y1, x2, y2
        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        boxes = np.concatenate([box_x1y1, box_x2y2], axis=-1)

        # Confidence = objectness * class_prob
        confidence = objectness * class_probs
        class_ids = np.argmax(confidence, axis=-1)
        class_conf = np.max(confidence, axis=-1, keepdims=True)

        # Reshape for output
        boxes = boxes.reshape(-1, 4)
        class_conf = class_conf.reshape(-1, 1)
        class_ids = class_ids.reshape(-1, 1)

        return np.concatenate([boxes, class_conf, class_ids], axis=-1)

    def _non_max_suppression(self, boxes: np.ndarray) -> np.ndarray:
        """
        Apply Non-Maximum Suppression.

        Args:
            boxes: Array of boxes (N, 6) with [x1, y1, x2, y2, conf, class]

        Returns:
            Filtered boxes after NMS
        """
        if len(boxes) == 0:
            return boxes

        # Filter by confidence
        mask = boxes[:, 4] >= self.confidence_threshold
        boxes = boxes[mask]

        if len(boxes) == 0:
            return boxes

        # Sort by confidence
        indices = np.argsort(boxes[:, 4])[::-1]
        boxes = boxes[indices]

        # NMS per class
        keep = []
        unique_classes = np.unique(boxes[:, 5])

        for cls in unique_classes:
            cls_mask = boxes[:, 5] == cls
            cls_boxes = boxes[cls_mask]
            cls_indices = np.where(cls_mask)[0]

            while len(cls_boxes) > 0:
                keep.append(cls_indices[0])

                if len(cls_boxes) == 1:
                    break

                # Calculate IoU with remaining boxes
                x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
                y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
                x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
                y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

                intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

                area1 = (cls_boxes[0, 2] - cls_boxes[0, 0]) * (cls_boxes[0, 3] - cls_boxes[0, 1])
                area2 = (cls_boxes[1:, 2] - cls_boxes[1:, 0]) * (cls_boxes[1:, 3] - cls_boxes[1:, 1])
                union = area1 + area2 - intersection

                iou = intersection / (union + 1e-8)

                # Remove overlapping boxes
                mask = iou < self.nms_threshold
                cls_boxes = cls_boxes[1:][mask]
                cls_indices = cls_indices[1:][mask]

        return boxes[keep] if keep else np.array([])

    def postprocess(
        self,
        output: tuple[mx.array, mx.array, mx.array],
        original_size: tuple[int, int],
    ) -> DetectionResult:
        """
        Postprocess YOLO output to detection results.

        Args:
            output: Predictions from three scales
            original_size: Original frame size (H, W)

        Returns:
            DetectionResult with bounding boxes
        """
        pred1, pred2, pred3 = output
        strides = [32, 16, 8]

        # Decode all scales
        all_boxes = []
        for pred, anchors, stride in zip([pred1, pred2, pred3], self.anchors, strides):
            boxes = self._decode_predictions(pred, anchors, stride)
            all_boxes.append(boxes)

        boxes = np.concatenate(all_boxes, axis=0)

        # Apply NMS
        boxes = self._non_max_suppression(boxes)

        if len(boxes) == 0:
            return DetectionResult.empty()

        # Convert to normalized coordinates and create BoundingBox objects
        h, w = original_size
        input_h, input_w = self.default_input_size

        # Account for letterboxing
        info = self._letterbox_info
        scale = info["scale"]
        offset_x = info["offset_x"]
        offset_y = info["offset_y"]

        bounding_boxes = []
        for box in boxes:
            # Remove letterbox offset and scale
            x1 = (box[0] - offset_x) / scale
            y1 = (box[1] - offset_y) / scale
            x2 = (box[2] - offset_x) / scale
            y2 = (box[3] - offset_y) / scale

            # Normalize to [0, 1]
            x1, x2 = x1 / w, x2 / w
            y1, y2 = y1 / h, y2 / h

            # Clip to valid range
            x1, x2 = np.clip([x1, x2], 0, 1)
            y1, y2 = np.clip([y1, y2], 0, 1)

            class_id = int(box[5])
            bounding_boxes.append(BoundingBox(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(box[4]),
                class_id=class_id,
                class_name=self.class_names[class_id],
            ))

        return DetectionResult(
            boxes=bounding_boxes,
            num_detections=len(bounding_boxes),
        )
