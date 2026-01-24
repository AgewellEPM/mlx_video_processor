"""
Type definitions for MLX Video Processor.

This module defines all the data types used throughout the video processing pipeline,
ensuring type safety and clear interfaces between components.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
import numpy as np
import mlx.core as mx


class ModelType(Enum):
    """Enumeration of supported model types."""
    CLASSIFICATION = auto()
    OBJECT_DETECTION = auto()
    SEMANTIC_SEGMENTATION = auto()
    DEPTH_ESTIMATION = auto()
    PANOPTIC_SEGMENTATION = auto()


class PrecisionType(Enum):
    """Model weight precision options."""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"


@dataclass
class BoundingBox:
    """Represents a bounding box detection result."""
    x1: float  # Top-left x coordinate (normalized 0-1)
    y1: float  # Top-left y coordinate (normalized 0-1)
    x2: float  # Bottom-right x coordinate (normalized 0-1)
    y2: float  # Bottom-right y coordinate (normalized 0-1)
    confidence: float  # Detection confidence score
    class_id: int  # Class index
    class_name: str  # Human-readable class name

    @property
    def width(self) -> float:
        """Calculate box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Calculate box height."""
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        """Calculate box center coordinates."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        """Calculate box area."""
        return self.width * self.height

    def to_pixel_coords(self, img_width: int, img_height: int) -> tuple[int, int, int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        return (
            int(self.x1 * img_width),
            int(self.y1 * img_height),
            int(self.x2 * img_width),
            int(self.y2 * img_height),
        )


@dataclass
class ClassificationResult:
    """Represents an image classification result."""
    class_id: int
    class_name: str
    confidence: float
    top_k: list[tuple[int, str, float]] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"ClassificationResult({self.class_name}: {self.confidence:.2%})"


@dataclass
class SegmentationResult:
    """Represents a semantic segmentation result."""
    mask: mx.array  # Shape: (H, W) with class indices
    class_ids: list[int]  # List of unique class IDs in the mask
    class_names: list[str]  # Corresponding class names
    confidence_map: Optional[mx.array] = None  # Shape: (H, W) with confidence scores

    @property
    def num_classes(self) -> int:
        """Return number of unique classes in segmentation."""
        return len(self.class_ids)


@dataclass
class DepthResult:
    """Represents a depth estimation result."""
    depth_map: mx.array  # Shape: (H, W) with depth values
    min_depth: float  # Minimum depth value
    max_depth: float  # Maximum depth value
    is_metric: bool = False  # Whether depth values are in metric units

    def get_normalized(self) -> mx.array:
        """Return depth map normalized to 0-1 range."""
        return (self.depth_map - self.min_depth) / (self.max_depth - self.min_depth + 1e-8)


@dataclass
class DetectionResult:
    """Represents object detection results for a single frame."""
    boxes: list[BoundingBox]
    num_detections: int

    @classmethod
    def empty(cls) -> "DetectionResult":
        """Create an empty detection result."""
        return cls(boxes=[], num_detections=0)

    def filter_by_confidence(self, threshold: float) -> "DetectionResult":
        """Filter detections by confidence threshold."""
        filtered = [b for b in self.boxes if b.confidence >= threshold]
        return DetectionResult(boxes=filtered, num_detections=len(filtered))

    def filter_by_class(self, class_ids: list[int]) -> "DetectionResult":
        """Filter detections by class IDs."""
        filtered = [b for b in self.boxes if b.class_id in class_ids]
        return DetectionResult(boxes=filtered, num_detections=len(filtered))


@dataclass
class FrameResult:
    """Container for all model outputs for a single frame."""
    frame_index: int
    timestamp_ms: float
    classification: Optional[ClassificationResult] = None
    detection: Optional[DetectionResult] = None
    segmentation: Optional[SegmentationResult] = None
    depth: Optional[DepthResult] = None
    panoptic: Optional[SegmentationResult] = None
    processing_time_ms: float = 0.0

    def has_results(self) -> bool:
        """Check if any model produced results."""
        return any([
            self.classification,
            self.detection,
            self.segmentation,
            self.depth,
            self.panoptic,
        ])


@dataclass
class VideoMetadata:
    """Metadata about the input video."""
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str

    @property
    def aspect_ratio(self) -> float:
        """Calculate video aspect ratio."""
        return self.width / self.height if self.height > 0 else 0.0


@dataclass
class ProcessingConfig:
    """Configuration for video processing pipeline."""
    # Frame extraction settings
    frame_interval: int = 1  # Process every Nth frame
    start_frame: int = 0  # Starting frame index
    end_frame: Optional[int] = None  # Ending frame index (None = end of video)
    max_frames: Optional[int] = None  # Maximum frames to process

    # Model settings
    batch_size: int = 1  # Frames to process in parallel (where supported)
    precision: PrecisionType = PrecisionType.FLOAT16
    confidence_threshold: float = 0.5  # For detection models

    # Output settings
    output_width: Optional[int] = None  # Output video width (None = same as input)
    output_height: Optional[int] = None  # Output video height
    output_fps: Optional[float] = None  # Output FPS (None = same as input)

    # Performance settings
    use_compile: bool = True  # Use mx.compile for optimization
    prefetch_frames: int = 4  # Number of frames to prefetch

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.frame_interval < 1:
            raise ValueError("frame_interval must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.start_frame < 0:
            raise ValueError("start_frame must be >= 0")
