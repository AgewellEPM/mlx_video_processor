"""
Core video processing pipeline.

Provides the main VideoProcessor class that handles video I/O,
frame extraction, model inference, and result aggregation.
"""

from typing import Callable, Generator, Optional, Any
from pathlib import Path
import numpy as np
from tqdm import tqdm

from mlx_video_processor.models.base import BaseModel
from mlx_video_processor.types import (
    FrameResult,
    ProcessingConfig,
    VideoMetadata,
)
from mlx_video_processor.utils.frame_extractor import FrameExtractor
from mlx_video_processor.utils.video_io import VideoReader, VideoWriter
from mlx_video_processor.utils.timer import PerformanceMonitor
from mlx_video_processor.visualizers.base import BaseVisualizer


class VideoProcessor:
    """
    Main video processing pipeline.

    Handles the complete workflow of reading video, extracting frames,
    running model inference, visualizing results, and writing output.

    Optimized for Apple Silicon using MLX's unified memory architecture.
    """

    def __init__(
        self,
        model: BaseModel,
        visualizer: Optional[BaseVisualizer] = None,
        config: Optional[ProcessingConfig] = None,
    ):
        """
        Initialize video processor.

        Args:
            model: Model processor to use
            visualizer: Optional visualizer for results
            config: Processing configuration
        """
        self.model = model
        self.visualizer = visualizer
        self.config = config or ProcessingConfig()
        self.config.validate()

        self._monitor = PerformanceMonitor()
        self._results: list[FrameResult] = []

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        progress: bool = True,
        callback: Optional[Callable[[FrameResult, np.ndarray], None]] = None,
    ) -> list[FrameResult]:
        """
        Process a video file.

        Args:
            input_path: Path to input video
            output_path: Optional path for output video with visualizations
            progress: Show progress bar
            callback: Optional callback for each processed frame

        Returns:
            List of FrameResult objects for all processed frames
        """
        self._results = []
        self._monitor.start_session()

        # Open video
        extractor = FrameExtractor(input_path, self.config)
        extractor.open()
        metadata = extractor.metadata

        # Initialize model
        with self._monitor.time("model_load"):
            self.model.load_model()

        # Warm up model
        with self._monitor.time("warmup"):
            self.model.warmup()

        # Setup output writer if needed
        writer = None
        if output_path:
            output_fps = self.config.output_fps or metadata.fps / self.config.frame_interval
            output_width = self.config.output_width or metadata.width
            output_height = self.config.output_height or metadata.height

            writer = VideoWriter(output_path, output_fps, output_width, output_height)
            writer.open()

        # Process frames
        try:
            frame_iterator = extractor.extract_frames()

            if progress:
                total_frames = len(extractor)
                frame_iterator = tqdm(
                    frame_iterator,
                    total=total_frames,
                    desc=f"Processing with {self.model.model_name}",
                    unit="frames",
                )

            for frame_idx, timestamp_ms, frame in frame_iterator:
                # Process frame
                with self._monitor.time("inference"):
                    result = self.model.process_frame(frame, frame_idx, timestamp_ms)

                self._results.append(result)
                self._monitor.record("frame_processing", result.processing_time_ms)

                # Visualize and write output
                if writer or callback:
                    with self._monitor.time("visualization"):
                        if self.visualizer:
                            vis_frame = self.visualizer.draw(frame, result, is_bgr=True)
                        else:
                            vis_frame = frame

                    if writer:
                        with self._monitor.time("video_write"):
                            writer.write_frame(vis_frame, is_rgb=False)

                    if callback:
                        callback(result, vis_frame)

        finally:
            extractor.close()
            if writer:
                writer.close()

        return self._results

    def process_frames(
        self,
        frames: list[np.ndarray],
        timestamps_ms: Optional[list[float]] = None,
    ) -> list[FrameResult]:
        """
        Process a list of frames directly.

        Args:
            frames: List of frame arrays (H, W, 3) BGR
            timestamps_ms: Optional list of timestamps

        Returns:
            List of FrameResult objects
        """
        if timestamps_ms is None:
            timestamps_ms = [i * 33.33 for i in range(len(frames))]  # Assume 30fps

        # Ensure model is loaded
        self.model.load_model()

        results = []
        for i, (frame, ts) in enumerate(zip(frames, timestamps_ms)):
            result = self.model.process_frame(frame, i, ts)
            results.append(result)

        return results

    def process_single_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process a single frame.

        Args:
            frame: Frame array (H, W, 3) BGR

        Returns:
            FrameResult for the frame
        """
        self.model.load_model()
        return self.model.process_frame(frame, 0, 0.0)

    def stream_process(
        self,
        input_path: str,
    ) -> Generator[tuple[np.ndarray, FrameResult], None, None]:
        """
        Stream process video, yielding results frame by frame.

        Args:
            input_path: Path to input video

        Yields:
            Tuple of (frame, result) for each processed frame
        """
        self.model.load_model()
        self.model.warmup()

        extractor = FrameExtractor(input_path, self.config)
        try:
            extractor.open()

            for frame_idx, timestamp_ms, frame in extractor.extract_frames():
                result = self.model.process_frame(frame, frame_idx, timestamp_ms)

                if self.visualizer:
                    vis_frame = self.visualizer.draw(frame, result, is_bgr=True)
                else:
                    vis_frame = frame

                yield vis_frame, result

        finally:
            extractor.close()

    def get_performance_summary(self) -> str:
        """Get performance metrics summary."""
        return self._monitor.get_summary()

    def get_results_summary(self) -> dict[str, Any]:
        """
        Get summary of processing results.

        Returns:
            Dictionary with result statistics
        """
        if not self._results:
            return {"status": "no_results"}

        total_time = sum(r.processing_time_ms for r in self._results)
        avg_time = total_time / len(self._results)

        summary = {
            "total_frames": len(self._results),
            "total_time_ms": total_time,
            "avg_time_per_frame_ms": avg_time,
            "fps": 1000 / avg_time if avg_time > 0 else 0,
        }

        # Model-specific summaries
        if self._results[0].detection is not None:
            total_detections = sum(r.detection.num_detections for r in self._results if r.detection)
            summary["total_detections"] = total_detections
            summary["avg_detections_per_frame"] = total_detections / len(self._results)

        if self._results[0].classification is not None:
            # Most common classifications
            from collections import Counter
            classes = [r.classification.class_name for r in self._results if r.classification]
            summary["top_classes"] = Counter(classes).most_common(5)

        return summary

    @property
    def results(self) -> list[FrameResult]:
        """Access processing results."""
        return self._results


class BatchVideoProcessor(VideoProcessor):
    """
    Video processor with batch inference support.

    For models that support batching, this processes multiple
    frames together for improved throughput.
    """

    def __init__(
        self,
        model: BaseModel,
        visualizer: Optional[BaseVisualizer] = None,
        config: Optional[ProcessingConfig] = None,
        batch_size: int = 4,
    ):
        """
        Initialize batch video processor.

        Args:
            model: Model processor to use (should support batching)
            visualizer: Optional visualizer
            config: Processing configuration
            batch_size: Number of frames per batch
        """
        super().__init__(model, visualizer, config)
        self.batch_size = batch_size

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        progress: bool = True,
        callback: Optional[Callable[[FrameResult, np.ndarray], None]] = None,
    ) -> list[FrameResult]:
        """Process video with batch inference."""
        if not self.model.supports_batching:
            # Fall back to single-frame processing
            return super().process_video(input_path, output_path, progress, callback)

        self._results = []
        self._monitor.start_session()

        extractor = FrameExtractor(input_path, self.config)
        extractor.open()
        metadata = extractor.metadata

        # Initialize model
        with self._monitor.time("model_load"):
            self.model.load_model()
            self.model.warmup()

        # Setup output writer
        writer = None
        if output_path:
            output_fps = self.config.output_fps or metadata.fps / self.config.frame_interval
            writer = VideoWriter(
                output_path,
                output_fps,
                self.config.output_width or metadata.width,
                self.config.output_height or metadata.height,
            )
            writer.open()

        try:
            # Collect frames into batches
            batch_frames = []
            batch_indices = []
            batch_timestamps = []

            frame_iterator = extractor.extract_frames()
            if progress:
                frame_iterator = tqdm(
                    frame_iterator,
                    total=len(extractor),
                    desc=f"Batch processing with {self.model.model_name}",
                )

            for frame_idx, timestamp_ms, frame in frame_iterator:
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                batch_timestamps.append(timestamp_ms)

                if len(batch_frames) >= self.batch_size:
                    # Process batch
                    with self._monitor.time("batch_inference"):
                        results = self.model.process_batch(
                            batch_frames, batch_indices, batch_timestamps
                        )

                    self._results.extend(results)

                    # Visualize and write
                    if writer or callback:
                        for frame, result in zip(batch_frames, results):
                            if self.visualizer:
                                vis_frame = self.visualizer.draw(frame, result, is_bgr=True)
                            else:
                                vis_frame = frame

                            if writer:
                                writer.write_frame(vis_frame, is_rgb=False)
                            if callback:
                                callback(result, vis_frame)

                    batch_frames.clear()
                    batch_indices.clear()
                    batch_timestamps.clear()

            # Process remaining frames
            if batch_frames:
                results = self.model.process_batch(
                    batch_frames, batch_indices, batch_timestamps
                )
                self._results.extend(results)

                if writer or callback:
                    for frame, result in zip(batch_frames, results):
                        if self.visualizer:
                            vis_frame = self.visualizer.draw(frame, result, is_bgr=True)
                        else:
                            vis_frame = frame

                        if writer:
                            writer.write_frame(vis_frame, is_rgb=False)
                        if callback:
                            callback(result, vis_frame)

        finally:
            extractor.close()
            if writer:
                writer.close()

        return self._results
