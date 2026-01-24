"""
Segmentation visualization for semantic masks.

Renders segmentation masks with color-coded class regions
for models like DeepLabV3 and DETR panoptic segmentation.
"""

from typing import Optional
import numpy as np
import cv2

from mlx_video_processor.visualizers.base import BaseVisualizer
from mlx_video_processor.types import FrameResult, SegmentationResult


# PASCAL VOC colors for semantic segmentation
VOC_COLORS = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
], dtype=np.uint8)


class SegmentationVisualizer(BaseVisualizer):
    """
    Visualizer for semantic segmentation results.

    Overlays color-coded segmentation masks on frames with
    optional class labels and legends.
    """

    def __init__(
        self,
        opacity: float = 0.5,
        show_labels: bool = True,
        show_legend: bool = True,
        class_colors: Optional[np.ndarray] = None,
        show_boundaries: bool = True,
    ):
        """
        Initialize segmentation visualizer.

        Args:
            opacity: Mask overlay opacity
            show_labels: Show class labels on regions
            show_legend: Show color legend
            class_colors: Custom color palette (N, 3)
            show_boundaries: Draw boundaries between regions
        """
        super().__init__(opacity, show_labels)
        self.show_legend = show_legend
        self.class_colors = class_colors if class_colors is not None else VOC_COLORS
        self.show_boundaries = show_boundaries

    def draw(
        self,
        frame: np.ndarray,
        result: FrameResult,
        is_bgr: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Draw segmentation mask on frame.

        Args:
            frame: Input frame
            result: Frame result with segmentation data
            is_bgr: Whether frame is BGR format

        Returns:
            Frame with segmentation overlay
        """
        seg_result = result.segmentation or result.panoptic
        if seg_result is None:
            return frame

        output = frame.copy()
        mask = np.array(seg_result.mask)

        # Resize mask if needed
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create color mask
        color_mask = self._create_color_mask(mask)

        # Blend with frame
        output = self._blend_overlay(output, color_mask)

        # Draw boundaries
        if self.show_boundaries:
            output = self._draw_boundaries(output, mask)

        # Draw legend
        if self.show_legend:
            output = self._draw_legend(output, seg_result)

        return output

    def _create_color_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create RGB color mask from class indices."""
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        unique_classes = np.unique(mask)
        for class_id in unique_classes:
            if class_id < len(self.class_colors):
                color = self.class_colors[class_id]
            else:
                # Generate color for unknown classes
                np.random.seed(class_id)
                color = np.random.randint(0, 255, 3)

            color_mask[mask == class_id] = color

        return color_mask

    def _draw_boundaries(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Draw boundaries between segmentation regions."""
        # Find edges using Sobel
        mask_float = mask.astype(np.float32)
        grad_x = cv2.Sobel(mask_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mask_float, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.sqrt(grad_x ** 2 + grad_y ** 2)
        edges = (edges > 0).astype(np.uint8) * 255

        # Draw edges in white
        frame[edges > 0] = [255, 255, 255]

        return frame

    def _draw_legend(
        self,
        frame: np.ndarray,
        result: SegmentationResult,
    ) -> np.ndarray:
        """Draw color legend for present classes."""
        if len(result.class_ids) == 0:
            return frame

        # Calculate legend dimensions
        box_size = 15
        padding = 5
        font_scale = 0.4
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Get text sizes
        max_text_width = 0
        for name in result.class_names:
            (tw, _), _ = cv2.getTextSize(name, font, font_scale, 1)
            max_text_width = max(max_text_width, tw)

        legend_width = box_size + padding * 3 + max_text_width
        legend_height = len(result.class_ids) * (box_size + padding) + padding

        # Draw legend background
        h, w = frame.shape[:2]
        legend_x = w - legend_width - 10
        legend_y = 10

        cv2.rectangle(
            frame,
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y + legend_height),
            (40, 40, 40),
            -1,
        )

        # Draw entries
        for i, (class_id, class_name) in enumerate(zip(result.class_ids, result.class_names)):
            y_pos = legend_y + padding + i * (box_size + padding)

            # Color box
            color = self.class_colors[class_id] if class_id < len(self.class_colors) else [128, 128, 128]
            color = tuple(int(c) for c in color)

            cv2.rectangle(
                frame,
                (legend_x + padding, y_pos),
                (legend_x + padding + box_size, y_pos + box_size),
                color,
                -1,
            )

            # Class name
            cv2.putText(
                frame,
                class_name[:15],  # Truncate long names
                (legend_x + box_size + padding * 2, y_pos + box_size - 3),
                font,
                font_scale,
                (255, 255, 255),
                1,
            )

        return frame

    def mask_to_rgb(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert segmentation mask to RGB image.

        Args:
            mask: Segmentation mask with class indices (H, W)

        Returns:
            RGB color image (H, W, 3)
        """
        return self._create_color_mask(mask)
