"""
Base visualizer class for MLX Video Processor.

Defines the interface for all visualization components.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np

from mlx_video_processor.types import FrameResult


class BaseVisualizer(ABC):
    """
    Abstract base class for visualizers.

    All visualizers must implement the draw method to render
    model outputs onto frames.
    """

    def __init__(self, opacity: float = 0.6, show_labels: bool = True):
        """
        Initialize visualizer.

        Args:
            opacity: Opacity for overlays (0-1)
            show_labels: Whether to show text labels
        """
        self.opacity = opacity
        self.show_labels = show_labels

    @abstractmethod
    def draw(
        self,
        frame: np.ndarray,
        result: FrameResult,
        **kwargs,
    ) -> np.ndarray:
        """
        Draw visualization on frame.

        Args:
            frame: Input frame (H, W, 3) in RGB or BGR
            result: Frame processing result
            **kwargs: Additional visualization options

        Returns:
            Frame with visualization overlaid
        """
        pass

    def _ensure_rgb(self, frame: np.ndarray, is_bgr: bool = True) -> np.ndarray:
        """Convert BGR to RGB if needed."""
        import cv2
        if is_bgr:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def _ensure_bgr(self, frame: np.ndarray, is_rgb: bool = True) -> np.ndarray:
        """Convert RGB to BGR if needed."""
        import cv2
        if is_rgb:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def _blend_overlay(
        self,
        background: np.ndarray,
        overlay: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Blend overlay onto background.

        Args:
            background: Background image
            overlay: Overlay image (same size)
            mask: Optional mask for selective blending

        Returns:
            Blended image
        """
        if mask is not None:
            mask = mask.astype(np.float32) / 255.0 if mask.max() > 1 else mask
            mask = mask[..., None] if mask.ndim == 2 else mask
            return (background * (1 - mask * self.opacity) + overlay * mask * self.opacity).astype(np.uint8)
        else:
            return ((1 - self.opacity) * background + self.opacity * overlay).astype(np.uint8)

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple[int, int],
        font_scale: float = 0.6,
        color: tuple[int, int, int] = (255, 255, 255),
        bg_color: Optional[tuple[int, int, int]] = (0, 0, 0),
        thickness: int = 1,
    ) -> np.ndarray:
        """
        Draw text with background on frame.

        Args:
            frame: Input frame
            text: Text to draw
            position: (x, y) position
            font_scale: Font scale
            color: Text color (BGR)
            bg_color: Background color (BGR) or None for no background
            thickness: Text thickness

        Returns:
            Frame with text
        """
        import cv2

        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        x, y = position
        y = max(y, text_h + 5)  # Ensure text fits

        if bg_color is not None:
            # Draw background rectangle
            cv2.rectangle(
                frame,
                (x - 2, y - text_h - 5),
                (x + text_w + 2, y + baseline),
                bg_color,
                -1,
            )

        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
        return frame

    def get_color_palette(self, num_colors: int) -> list[tuple[int, int, int]]:
        """
        Generate a color palette.

        Args:
            num_colors: Number of colors needed

        Returns:
            List of RGB color tuples
        """
        import colorsys

        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(tuple(int(c * 255) for c in rgb))

        return colors
