"""
Depth visualization for depth estimation models.

Renders depth maps using colormaps with optional overlays
for models like Depth Anything V2.
"""

from typing import Optional, Literal
import numpy as np
import cv2

from mlx_video_processor.visualizers.base import BaseVisualizer
from mlx_video_processor.types import FrameResult, DepthResult


ColormapType = Literal["magma", "viridis", "plasma", "inferno", "jet", "turbo", "gray"]


class DepthVisualizer(BaseVisualizer):
    """
    Visualizer for depth estimation results.

    Renders depth maps using various colormaps and blending modes.
    """

    # OpenCV colormap mappings
    COLORMAPS = {
        "magma": cv2.COLORMAP_MAGMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "gray": None,  # Special case for grayscale
    }

    def __init__(
        self,
        opacity: float = 0.7,
        show_labels: bool = True,
        colormap: ColormapType = "magma",
        invert: bool = False,
        show_colorbar: bool = True,
        show_stats: bool = True,
    ):
        """
        Initialize depth visualizer.

        Args:
            opacity: Depth overlay opacity
            show_labels: Show depth value labels
            colormap: Color mapping scheme
            invert: Invert depth values (near=bright vs near=dark)
            show_colorbar: Show depth colorbar
            show_stats: Show min/max depth statistics
        """
        super().__init__(opacity, show_labels)
        self.colormap = colormap
        self.invert = invert
        self.show_colorbar = show_colorbar
        self.show_stats = show_stats

    def draw(
        self,
        frame: np.ndarray,
        result: FrameResult,
        is_bgr: bool = True,
        blend_mode: str = "overlay",
        **kwargs,
    ) -> np.ndarray:
        """
        Draw depth visualization on frame.

        Args:
            frame: Input frame
            result: Frame result with depth data
            is_bgr: Whether frame is BGR format
            blend_mode: "overlay", "side_by_side", or "depth_only"

        Returns:
            Frame with depth visualization
        """
        if result.depth is None:
            return frame

        depth_map = np.array(result.depth.depth_map)

        # Resize to frame size if needed
        if depth_map.shape[:2] != frame.shape[:2]:
            depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

        # Normalize depth
        if result.depth.max_depth > result.depth.min_depth:
            depth_norm = (depth_map - result.depth.min_depth) / (result.depth.max_depth - result.depth.min_depth)
        else:
            depth_norm = np.zeros_like(depth_map)

        if self.invert:
            depth_norm = 1.0 - depth_norm

        # Apply colormap
        colored_depth = self._apply_colormap(depth_norm)

        # Apply blend mode
        if blend_mode == "overlay":
            output = self._blend_overlay(frame, colored_depth)
        elif blend_mode == "side_by_side":
            output = self._create_side_by_side(frame, colored_depth)
        elif blend_mode == "depth_only":
            output = colored_depth
        else:
            output = self._blend_overlay(frame, colored_depth)

        # Draw colorbar
        if self.show_colorbar:
            output = self._draw_colorbar(output, result.depth)

        # Draw stats
        if self.show_stats:
            output = self._draw_stats(output, result.depth)

        return output

    def _apply_colormap(self, depth_norm: np.ndarray) -> np.ndarray:
        """Apply colormap to normalized depth."""
        # Convert to 8-bit
        depth_uint8 = (depth_norm * 255).astype(np.uint8)

        if self.colormap == "gray":
            return cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)

        cm = self.COLORMAPS.get(self.colormap, cv2.COLORMAP_MAGMA)
        return cv2.applyColorMap(depth_uint8, cm)

    def _create_side_by_side(
        self,
        frame: np.ndarray,
        depth_colored: np.ndarray,
    ) -> np.ndarray:
        """Create side-by-side comparison."""
        h, w = frame.shape[:2]

        # Resize both to half width
        frame_half = cv2.resize(frame, (w // 2, h))
        depth_half = cv2.resize(depth_colored, (w // 2, h))

        return np.concatenate([frame_half, depth_half], axis=1)

    def _draw_colorbar(
        self,
        frame: np.ndarray,
        depth_result: DepthResult,
    ) -> np.ndarray:
        """Draw colorbar legend."""
        h, w = frame.shape[:2]

        # Colorbar dimensions
        bar_width = 20
        bar_height = 150
        bar_x = w - bar_width - 30
        bar_y = h - bar_height - 50

        # Create gradient
        gradient = np.linspace(0, 1, bar_height).reshape(-1, 1)
        if not self.invert:
            gradient = 1 - gradient  # Top = far, bottom = near

        gradient = np.tile(gradient, (1, bar_width))
        gradient_uint8 = (gradient * 255).astype(np.uint8)

        if self.colormap == "gray":
            colorbar = cv2.cvtColor(gradient_uint8, cv2.COLOR_GRAY2BGR)
        else:
            cm = self.COLORMAPS.get(self.colormap, cv2.COLORMAP_MAGMA)
            colorbar = cv2.applyColorMap(gradient_uint8, cm)

        # Draw colorbar
        frame[bar_y:bar_y + bar_height, bar_x:bar_x + bar_width] = colorbar

        # Draw border
        cv2.rectangle(
            frame,
            (bar_x - 1, bar_y - 1),
            (bar_x + bar_width + 1, bar_y + bar_height + 1),
            (255, 255, 255),
            1,
        )

        # Draw labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4

        # Far label (top)
        cv2.putText(frame, "Far", (bar_x - 25, bar_y + 10), font, font_scale, (255, 255, 255), 1)

        # Near label (bottom)
        cv2.putText(frame, "Near", (bar_x - 30, bar_y + bar_height - 5), font, font_scale, (255, 255, 255), 1)

        return frame

    def _draw_stats(
        self,
        frame: np.ndarray,
        depth_result: DepthResult,
    ) -> np.ndarray:
        """Draw depth statistics."""
        stats_text = f"Depth: {depth_result.min_depth:.2f} - {depth_result.max_depth:.2f}"
        if depth_result.is_metric:
            stats_text += "m"

        return self._draw_text(
            frame,
            stats_text,
            (10, frame.shape[0] - 20),
            font_scale=0.5,
            color=(255, 255, 255),
            bg_color=(0, 0, 0),
        )

    def depth_to_rgb(
        self,
        depth_map: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Convert depth map to RGB image.

        Args:
            depth_map: Depth values (H, W)
            normalize: Whether to normalize to [0, 1]

        Returns:
            RGB color image (H, W, 3)
        """
        if normalize:
            d_min, d_max = depth_map.min(), depth_map.max()
            if d_max > d_min:
                depth_norm = (depth_map - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth_map)
        else:
            depth_norm = depth_map

        if self.invert:
            depth_norm = 1.0 - depth_norm

        colored = self._apply_colormap(depth_norm)

        # Convert BGR to RGB
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
