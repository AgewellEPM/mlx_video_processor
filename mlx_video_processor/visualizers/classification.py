"""
Classification visualization for image classification results.

Displays class predictions with confidence bars and top-k results.
"""

from typing import Optional
import numpy as np
import cv2

from mlx_video_processor.visualizers.base import BaseVisualizer
from mlx_video_processor.types import FrameResult, ClassificationResult


class ClassificationVisualizer(BaseVisualizer):
    """
    Visualizer for classification results.

    Displays top-k predictions with confidence bars.
    """

    def __init__(
        self,
        opacity: float = 0.8,
        show_labels: bool = True,
        show_top_k: int = 5,
        bar_width: int = 200,
        position: str = "top_left",
    ):
        """
        Initialize classification visualizer.

        Args:
            opacity: Background opacity for overlay
            show_labels: Show class labels
            show_top_k: Number of top predictions to show
            bar_width: Width of confidence bars in pixels
            position: Position of overlay ("top_left", "top_right", "bottom_left", "bottom_right")
        """
        super().__init__(opacity, show_labels)
        self.show_top_k = show_top_k
        self.bar_width = bar_width
        self.position = position

    def draw(
        self,
        frame: np.ndarray,
        result: FrameResult,
        is_bgr: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Draw classification results on frame.

        Args:
            frame: Input frame
            result: Frame result with classification data
            is_bgr: Whether frame is BGR format

        Returns:
            Frame with classification overlay
        """
        if result.classification is None:
            return frame

        output = frame.copy()
        cls_result = result.classification

        # Get predictions to show
        predictions = cls_result.top_k[:self.show_top_k] if cls_result.top_k else [
            (cls_result.class_id, cls_result.class_name, cls_result.confidence)
        ]

        # Calculate overlay dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        line_height = 25
        padding = 10

        # Find max label width
        max_label_width = 0
        for _, name, _ in predictions:
            (tw, _), _ = cv2.getTextSize(name[:20], font, font_scale, 1)
            max_label_width = max(max_label_width, tw)

        overlay_width = max_label_width + self.bar_width + padding * 4 + 50  # +50 for percentage
        overlay_height = len(predictions) * line_height + padding * 2

        # Calculate position
        h, w = frame.shape[:2]
        if self.position == "top_left":
            x, y = padding, padding
        elif self.position == "top_right":
            x, y = w - overlay_width - padding, padding
        elif self.position == "bottom_left":
            x, y = padding, h - overlay_height - padding
        else:  # bottom_right
            x, y = w - overlay_width - padding, h - overlay_height - padding

        # Draw background
        overlay = output.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + overlay_width, y + overlay_height),
            (30, 30, 30),
            -1,
        )
        cv2.addWeighted(overlay, self.opacity, output, 1 - self.opacity, 0, output)

        # Draw border
        cv2.rectangle(
            output,
            (x, y),
            (x + overlay_width, y + overlay_height),
            (100, 100, 100),
            1,
        )

        # Draw predictions
        for i, (class_id, class_name, confidence) in enumerate(predictions):
            row_y = y + padding + i * line_height

            # Draw label
            label = class_name[:20]  # Truncate long names
            cv2.putText(
                output,
                label,
                (x + padding, row_y + 15),
                font,
                font_scale,
                (255, 255, 255),
                1,
            )

            # Draw confidence bar background
            bar_x = x + max_label_width + padding * 2
            bar_y = row_y + 3
            bar_height = 16

            cv2.rectangle(
                output,
                (bar_x, bar_y),
                (bar_x + self.bar_width, bar_y + bar_height),
                (60, 60, 60),
                -1,
            )

            # Draw confidence bar fill
            fill_width = int(confidence * self.bar_width)
            color = self._get_confidence_color(confidence)

            cv2.rectangle(
                output,
                (bar_x, bar_y),
                (bar_x + fill_width, bar_y + bar_height),
                color,
                -1,
            )

            # Draw percentage
            pct_text = f"{confidence:.1%}"
            cv2.putText(
                output,
                pct_text,
                (bar_x + self.bar_width + 5, row_y + 15),
                font,
                font_scale,
                (255, 255, 255),
                1,
            )

        return output

    def _get_confidence_color(self, confidence: float) -> tuple[int, int, int]:
        """Get color based on confidence level."""
        if confidence >= 0.8:
            return (0, 200, 0)  # Green
        elif confidence >= 0.5:
            return (0, 200, 200)  # Yellow
        elif confidence >= 0.3:
            return (0, 150, 255)  # Orange
        else:
            return (0, 0, 200)  # Red

    def draw_minimal(
        self,
        frame: np.ndarray,
        result: ClassificationResult,
    ) -> np.ndarray:
        """
        Draw minimal classification overlay (just top prediction).

        Args:
            frame: Input frame
            result: Classification result

        Returns:
            Frame with minimal overlay
        """
        output = frame.copy()

        text = f"{result.class_name}: {result.confidence:.1%}"

        return self._draw_text(
            output,
            text,
            (10, 30),
            font_scale=0.7,
            color=(255, 255, 255),
            bg_color=(0, 0, 0),
            thickness=1,
        )
