"""
Video processing pipelines for MLX Video Processor.

Provides high-level interfaces for processing videos with
multiple models.
"""

from mlx_video_processor.pipelines.video_processor import VideoProcessor
from mlx_video_processor.pipelines.orchestrator import ModelOrchestrator

__all__ = [
    "VideoProcessor",
    "ModelOrchestrator",
]
