"""
Model implementations for MLX Video Processor.

Each model is implemented as a separate module following a common interface,
allowing them to be swapped and composed easily.
"""

from mlx_video_processor.models.base import BaseModel, ModelRegistry
from mlx_video_processor.models.fastvit import FastViTProcessor
from mlx_video_processor.models.yolo import YOLOv3Processor
from mlx_video_processor.models.deeplab import DeepLabV3Processor
from mlx_video_processor.models.depth_anything import DepthAnythingV2Processor
from mlx_video_processor.models.detr import DETRProcessor

__all__ = [
    "BaseModel",
    "ModelRegistry",
    "FastViTProcessor",
    "YOLOv3Processor",
    "DeepLabV3Processor",
    "DepthAnythingV2Processor",
    "DETRProcessor",
]
