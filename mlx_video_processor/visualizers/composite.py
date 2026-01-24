"""
Composite visualizer for combining multiple model outputs.

Provides unified visualization when running multiple models
on the same video frames.
"""

from typing import Optional
import numpy as np
import cv2

from mlx_video_processor.visualizers.base import BaseVisualizer
from mlx_video_processor.visualizers.detection import DetectionVisualizer
from mlx_video_processor.visualizers.segmentation import SegmentationVisualizer
from mlx_video_processor.visualizers.depth import DepthVisualizer
from mlx_video_processor.visualizers.classification import ClassificationVisualizer
from mlx_video_processor.types import FrameResult


class CompositeVisualizer(BaseVisualizer):
    """
    Composite visualizer for multiple model outputs.

    Combines outputs from classification, detection, segmentation,
    and depth models into a single visualization.
    """

    def __init__(
        self,
        opacity: float = 0.6,
        show_labels: bool = True,
        layout: str = "overlay",
        show_fps: bool = True,
        show_frame_info: bool = True,
    ):
        """
        Initialize composite visualizer.

        Args:
            opacity: Overall overlay opacity
            show_labels: Show labels for all visualizers
            layout: "overlay" (all on one frame) or "grid" (2x2 grid)
            show_fps: Show processing FPS
            show_frame_info: Show frame index and timestamp
        """
        super().__init__(opacity, show_labels)
        self.layout = layout
        self.show_fps = show_fps
        self.show_frame_info = show_frame_info

        # Initialize sub-visualizers
        self.detection_viz = DetectionVisualizer(opacity=0.7, show_labels=show_labels)
        self.segmentation_viz = SegmentationVisualizer(opacity=0.4, show_labels=show_labels)
        self.depth_viz = DepthVisualizer(opacity=0.5, show_labels=show_labels)
        self.classification_viz = ClassificationVisualizer(opacity=0.8, show_labels=show_labels)

    def draw(
        self,
        frame: np.ndarray,
        result: FrameResult,
        is_bgr: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Draw composite visualization.

        Args:
            frame: Input frame
            result: Frame result with all model outputs
            is_bgr: Whether frame is BGR format

        Returns:
            Frame with composite visualization
        """
        if self.layout == "grid":
            return self._draw_grid(frame, result, is_bgr)
        else:
            return self._draw_overlay(frame, result, is_bgr)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        result: FrameResult,
        is_bgr: bool,
    ) -> np.ndarray:
        """Draw all visualizations overlaid on single frame."""
        output = frame.copy()

        # Apply visualizations in order (background to foreground)

        # 1. Depth (background layer)
        if result.depth is not None:
            output = self.depth_viz.draw(output, result, is_bgr, blend_mode="overlay")

        # 2. Segmentation
        if result.segmentation is not None or result.panoptic is not None:
            output = self.segmentation_viz.draw(output, result, is_bgr)

        # 3. Detection boxes (foreground)
        if result.detection is not None:
            output = self.detection_viz.draw(output, result, is_bgr)

        # 4. Classification (top corner)
        if result.classification is not None:
            output = self.classification_viz.draw(output, result, is_bgr)

        # 5. Frame info overlay
        if self.show_frame_info or self.show_fps:
            output = self._draw_info_overlay(output, result)

        return output

    def _draw_grid(
        self,
        frame: np.ndarray,
        result: FrameResult,
        is_bgr: bool,
    ) -> np.ndarray:
        """Draw 2x2 grid showing different outputs."""
        h, w = frame.shape[:2]
        half_h, half_w = h // 2, w // 2

        # Create grid panels
        panels = []

        # Top-left: Original with detection
        panel1 = frame.copy()
        if result.detection is not None:
            panel1 = self.detection_viz.draw(panel1, result, is_bgr)
        if result.classification is not None:
            panel1 = self.classification_viz.draw(panel1, result, is_bgr)
        panels.append(cv2.resize(panel1, (half_w, half_h)))

        # Top-right: Segmentation
        panel2 = frame.copy()
        if result.segmentation is not None or result.panoptic is not None:
            panel2 = self.segmentation_viz.draw(panel2, result, is_bgr)
        else:
            self._add_label(panel2, "Segmentation: N/A")
        panels.append(cv2.resize(panel2, (half_w, half_h)))

        # Bottom-left: Depth
        panel3 = frame.copy()
        if result.depth is not None:
            panel3 = self.depth_viz.draw(panel3, result, is_bgr, blend_mode="depth_only")
        else:
            self._add_label(panel3, "Depth: N/A")
        panels.append(cv2.resize(panel3, (half_w, half_h)))

        # Bottom-right: Original
        panel4 = frame.copy()
        self._add_label(panel4, "Original")
        panels.append(cv2.resize(panel4, (half_w, half_h)))

        # Combine into grid
        top_row = np.concatenate([panels[0], panels[1]], axis=1)
        bottom_row = np.concatenate([panels[2], panels[3]], axis=1)
        output = np.concatenate([top_row, bottom_row], axis=0)

        # Draw grid lines
        cv2.line(output, (half_w, 0), (half_w, h), (255, 255, 255), 1)
        cv2.line(output, (0, half_h), (w, half_h), (255, 255, 255), 1)

        # Draw info overlay
        if self.show_frame_info or self.show_fps:
            output = self._draw_info_overlay(output, result)

        return output

    def _add_label(
        self,
        frame: np.ndarray,
        label: str,
    ) -> np.ndarray:
        """Add corner label to frame."""
        cv2.putText(
            frame,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
        return frame

    def _draw_info_overlay(
        self,
        frame: np.ndarray,
        result: FrameResult,
    ) -> np.ndarray:
        """Draw frame info and FPS overlay."""
        h, w = frame.shape[:2]
        info_lines = []

        if self.show_frame_info:
            info_lines.append(f"Frame: {result.frame_index}")
            info_lines.append(f"Time: {result.timestamp_ms / 1000:.2f}s")

        if self.show_fps and result.processing_time_ms > 0:
            fps = 1000 / result.processing_time_ms
            info_lines.append(f"FPS: {fps:.1f}")
            info_lines.append(f"Latency: {result.processing_time_ms:.1f}ms")

        # Draw info box
        if info_lines:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            padding = 5
            line_height = 18

            # Calculate box size
            max_width = 0
            for line in info_lines:
                (tw, _), _ = cv2.getTextSize(line, font, font_scale, 1)
                max_width = max(max_width, tw)

            box_width = max_width + padding * 2
            box_height = len(info_lines) * line_height + padding * 2

            # Draw semi-transparent background
            x, y = w - box_width - 10, 10
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (x, y),
                (x + box_width, y + box_height),
                (30, 30, 30),
                -1,
            )
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Draw text
            for i, line in enumerate(info_lines):
                text_y = y + padding + (i + 1) * line_height - 5
                cv2.putText(
                    frame,
                    line,
                    (x + padding, text_y),
                    font,
                    font_scale,
                    (200, 200, 200),
                    1,
                )

        return frame


def create_comparison_grid(
    frame: np.ndarray,
    results: dict[str, np.ndarray],
    titles: Optional[list[str]] = None,
) -> np.ndarray:
    """
    Create a grid comparing multiple visualization outputs.

    Args:
        frame: Original frame
        results: Dictionary of {name: visualized_frame}
        titles: Optional titles for each panel

    Returns:
        Grid image showing all results
    """
    panels = [frame] + list(results.values())
    panel_titles = ["Original"] + (titles or list(results.keys()))

    # Determine grid dimensions
    n = len(panels)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    h, w = frame.shape[:2]
    cell_h, cell_w = h // rows, w // cols

    # Create output grid
    grid = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)

    for i, (panel, title) in enumerate(zip(panels, panel_titles)):
        row, col = i // cols, i % cols
        y, x = row * cell_h, col * cell_w

        # Resize and place panel
        resized = cv2.resize(panel, (cell_w, cell_h))
        grid[y:y + cell_h, x:x + cell_w] = resized

        # Add title
        cv2.putText(
            grid,
            title,
            (x + 10, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            grid,
            title,
            (x + 10, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )

    return grid
