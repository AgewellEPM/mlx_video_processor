"""
Frame extraction utilities for video processing.

Provides efficient frame extraction with configurable sampling rates
and prefetching for optimal performance on Apple Silicon.
"""

from typing import Generator, Optional
import threading
import queue
import numpy as np
import cv2

from mlx_video_processor.types import VideoMetadata, ProcessingConfig


class FrameExtractor:
    """
    Efficient frame extraction from video files.

    Supports configurable frame intervals, prefetching for pipeline
    optimization, and frame-accurate seeking.
    """

    def __init__(self, video_path: str, config: Optional[ProcessingConfig] = None):
        """
        Initialize frame extractor.

        Args:
            video_path: Path to video file
            config: Processing configuration (optional)
        """
        self.video_path = video_path
        self.config = config or ProcessingConfig()
        self.config.validate()

        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self._prefetch_queue: Optional[queue.Queue] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._stop_prefetch = threading.Event()

    def open(self) -> None:
        """Open video file and read metadata."""
        self._cap = cv2.VideoCapture(self.video_path)

        if not self._cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        self._metadata = VideoMetadata(
            path=self.video_path,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self._cap.get(cv2.CAP_PROP_FPS),
            total_frames=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration_seconds=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) / self._cap.get(cv2.CAP_PROP_FPS),
            codec=self._fourcc_to_str(int(self._cap.get(cv2.CAP_PROP_FOURCC))),
        )

    def close(self) -> None:
        """Close video file and stop prefetching."""
        self._stop_prefetch.set()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=2.0)
            self._prefetch_thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def metadata(self) -> VideoMetadata:
        """Get video metadata."""
        if self._metadata is None:
            self.open()
        return self._metadata

    @property
    def is_open(self) -> bool:
        """Check if video is open."""
        return self._cap is not None and self._cap.isOpened()

    def _fourcc_to_str(self, fourcc: int) -> str:
        """Convert FOURCC code to string."""
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    def _get_frame_indices(self) -> list[int]:
        """Calculate frame indices to extract based on config."""
        if self._metadata is None:
            self.open()

        start = self.config.start_frame
        end = self.config.end_frame or self._metadata.total_frames
        interval = self.config.frame_interval

        indices = list(range(start, end, interval))

        if self.config.max_frames is not None:
            indices = indices[:self.config.max_frames]

        return indices

    def extract_frames(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
        """
        Extract frames from video.

        Yields:
            Tuple of (frame_index, timestamp_ms, frame_array)
        """
        if not self.is_open:
            self.open()

        indices = self._get_frame_indices()
        fps = self._metadata.fps

        for frame_idx in indices:
            # Seek to frame if not sequential
            current_pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_pos != frame_idx:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = self._cap.read()

            if not ret:
                break

            timestamp_ms = (frame_idx / fps) * 1000
            yield frame_idx, timestamp_ms, frame

    def extract_frames_prefetch(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
        """
        Extract frames with prefetching for improved throughput.

        Uses a background thread to prefetch frames while processing
        continues, optimizing memory bandwidth on Apple Silicon.

        Yields:
            Tuple of (frame_index, timestamp_ms, frame_array)
        """
        if not self.is_open:
            self.open()

        # Start prefetch thread
        self._prefetch_queue = queue.Queue(maxsize=self.config.prefetch_frames)
        self._stop_prefetch.clear()
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self._prefetch_thread.start()

        try:
            while True:
                try:
                    item = self._prefetch_queue.get(timeout=1.0)
                    if item is None:  # End signal
                        break
                    yield item
                except queue.Empty:
                    if not self._prefetch_thread.is_alive():
                        break
        finally:
            self._stop_prefetch.set()
            if self._prefetch_thread.is_alive():
                self._prefetch_thread.join(timeout=2.0)

    def _prefetch_worker(self) -> None:
        """Background worker for frame prefetching."""
        indices = self._get_frame_indices()
        fps = self._metadata.fps

        for frame_idx in indices:
            if self._stop_prefetch.is_set():
                break

            # Seek and read
            current_pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_pos != frame_idx:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = self._cap.read()

            if not ret:
                break

            timestamp_ms = (frame_idx / fps) * 1000

            try:
                self._prefetch_queue.put((frame_idx, timestamp_ms, frame), timeout=5.0)
            except queue.Full:
                if self._stop_prefetch.is_set():
                    break

        # Signal end
        try:
            self._prefetch_queue.put(None, timeout=1.0)
        except queue.Full:
            pass

    def extract_single_frame(self, frame_index: int) -> tuple[float, np.ndarray]:
        """
        Extract a single frame by index.

        Args:
            frame_index: Frame index to extract

        Returns:
            Tuple of (timestamp_ms, frame_array)
        """
        if not self.is_open:
            self.open()

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self._cap.read()

        if not ret:
            raise ValueError(f"Could not read frame {frame_index}")

        timestamp_ms = (frame_index / self._metadata.fps) * 1000
        return timestamp_ms, frame

    def get_batch(
        self,
        start_index: int,
        batch_size: int,
    ) -> list[tuple[int, float, np.ndarray]]:
        """
        Extract a batch of consecutive frames.

        Args:
            start_index: Starting frame index
            batch_size: Number of frames to extract

        Returns:
            List of (frame_index, timestamp_ms, frame_array) tuples
        """
        if not self.is_open:
            self.open()

        batch = []
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        fps = self._metadata.fps

        for i in range(batch_size):
            ret, frame = self._cap.read()
            if not ret:
                break

            frame_idx = start_index + i
            timestamp_ms = (frame_idx / fps) * 1000
            batch.append((frame_idx, timestamp_ms, frame))

        return batch

    def __enter__(self) -> "FrameExtractor":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __iter__(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
        return self.extract_frames()

    def __len__(self) -> int:
        return len(self._get_frame_indices())
