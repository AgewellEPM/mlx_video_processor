"""
Visualization utilities for model outputs.

Provides visualization functions for each model type including
bounding boxes, segmentation masks, and depth maps.
"""

from mlx_video_processor.visualizers.base import BaseVisualizer
from mlx_video_processor.visualizers.detection import DetectionVisualizer
from mlx_video_processor.visualizers.segmentation import SegmentationVisualizer
from mlx_video_processor.visualizers.depth import DepthVisualizer
from mlx_video_processor.visualizers.classification import ClassificationVisualizer
from mlx_video_processor.visualizers.composite import CompositeVisualizer

__all__ = [
    "BaseVisualizer",
    "DetectionVisualizer",
    "SegmentationVisualizer",
    "DepthVisualizer",
    "ClassificationVisualizer",
    "CompositeVisualizer",
]
