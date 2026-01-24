"""
MLX Video Processor - Modular video processing with Apple ML models on Apple Silicon.

This package provides a unified interface for applying various computer vision models
to video files, optimized for Apple Silicon using MLX's unified memory architecture.
"""

__version__ = "0.1.0"
__author__ = "MLX Video Processor Team"

from mlx_video_processor.pipelines.video_processor import VideoProcessor
from mlx_video_processor.pipelines.orchestrator import ModelOrchestrator

__all__ = [
    "VideoProcessor",
    "ModelOrchestrator",
    "__version__",
]
