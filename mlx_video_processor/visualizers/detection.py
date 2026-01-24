"""
Detection visualization for bounding boxes.

Draws bounding boxes with class labels and confidence scores
for object detection models like YOLOv3 and DETR.
"""

from typing import Optional
import numpy as np
import cv2

from mlx_video_processor.visualizers.base import BaseVisualizer
from mlx_video_processor.types import FrameResult, BoundingBox


class DetectionVisualizer(BaseVisualizer):
    """
    Visualizer for object detection results.

    Draws bounding boxes with class labels and confidence scores.
    """

    # Default color palette for classes
    DEFAULT_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
        (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128),
    ]

    def __init__(
        self,
        opacity: float = 0.6,
        show_labels: bool = True,
        show_confidence: bool = True,
        line_thickness: int = 2,
        font_scale: float = 0.5,
        class_colors: Optional[dict[int, tuple[int, int, int]]] = None,
    ):
        """
        Initialize detection visualizer.

        Args:
            opacity: Fill opacity for boxes
            show_labels: Show class name labels
            show_confidence: Show confidence scores
            line_thickness: Bounding box line thickness
            font_scale: Font scale for labels
            class_colors: Custom colors per class ID
        """
        super().__init__(opacity, show_labels)
        self.show_confidence = show_confidence
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.class_colors = class_colors or {}

    def _get_color(self, class_id: int) -> tuple[int, int, int]:
        """Get color for a class ID."""
        if class_id in self.class_colors:
            return self.class_colors[class_id]
        return self.DEFAULT_COLORS[class_id % len(self.DEFAULT_COLORS)]

    def draw(
        self,
        frame: np.ndarray,
        result: FrameResult,
        is_bgr: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Draw detection boxes on frame.

        Args:
            frame: Input frame
            result: Frame result with detection data
            is_bgr: Whether frame is BGR format

        Returns:
            Frame with detection visualization
        """
        if result.detection is None or result.detection.num_detections == 0:
            return frame

        output = frame.copy()
        h, w = frame.shape[:2]

        for box in result.detection.boxes:
            self._draw_box(output, box, w, h)

        return output

    def _draw_box(
        self,
        frame: np.ndarray,
        box: BoundingBox,
        img_width: int,
        img_height: int,
    ) -> None:
        """Draw a single bounding box."""
        # Convert normalized coords to pixels
        x1, y1, x2, y2 = box.to_pixel_coords(img_width, img_height)

        # Get color for this class
        color = self._get_color(box.class_id)

        # Draw filled rectangle with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, self.opacity * 0.3, frame, 1 - self.opacity * 0.3, 0, frame)

        # Draw border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)

        # Draw label
        if self.show_labels:
            label_parts = [box.class_name]
            if self.show_confidence:
                label_parts.append(f"{box.confidence:.0%}")
            label = " ".join(label_parts)

            # Calculate text size
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )

            # Draw label background
            label_y = max(y1 - 5, text_h + 5)
            cv2.rectangle(
                frame,
                (x1, label_y - text_h - 5),
                (x1 + text_w + 4, label_y + baseline),
                color,
                -1,
            )

            # Draw text
            cv2.putText(
                frame,
                label,
                (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1,
            )

    def draw_boxes_only(
        self,
        frame: np.ndarray,
        boxes: list[BoundingBox],
    ) -> np.ndarray:
        """
        Draw boxes without using FrameResult.

        Args:
            frame: Input frame
            boxes: List of bounding boxes

        Returns:
            Frame with boxes drawn
        """
        output = frame.copy()
        h, w = frame.shape[:2]

        for box in boxes:
            self._draw_box(output, box, w, h)

        return output
