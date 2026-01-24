"""
Video I/O utilities for reading and writing video files.

Provides high-level interfaces for video reading and writing
with support for various codecs and formats.
"""

from typing import Optional, Iterator
from pathlib import Path
import numpy as np
import cv2

from mlx_video_processor.types import VideoMetadata


class VideoReader:
    """
    High-level video reader with context manager support.

    Provides convenient iteration over video frames with
    automatic resource management.
    """

    def __init__(
        self,
        path: str,
        resize: Optional[tuple[int, int]] = None,
        to_rgb: bool = True,
    ):
        """
        Initialize video reader.

        Args:
            path: Path to video file
            resize: Optional (width, height) to resize frames
            to_rgb: Convert BGR to RGB
        """
        self.path = path
        self.resize = resize
        self.to_rgb = to_rgb
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

    def open(self) -> None:
        """Open video file."""
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video: {self.path}")

        self._metadata = VideoMetadata(
            path=self.path,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self._cap.get(cv2.CAP_PROP_FPS),
            total_frames=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration_seconds=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(self._cap.get(cv2.CAP_PROP_FPS), 1),
            codec=self._get_codec(),
        )

    def close(self) -> None:
        """Close video file."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _get_codec(self) -> str:
        """Get video codec as string."""
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    @property
    def metadata(self) -> VideoMetadata:
        """Get video metadata."""
        if self._metadata is None:
            self.open()
        return self._metadata

    @property
    def width(self) -> int:
        """Get frame width (after resize if specified)."""
        if self.resize:
            return self.resize[0]
        return self.metadata.width

    @property
    def height(self) -> int:
        """Get frame height (after resize if specified)."""
        if self.resize:
            return self.resize[1]
        return self.metadata.height

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read next frame.

        Returns:
            Frame array or None if end of video
        """
        if self._cap is None:
            self.open()

        ret, frame = self._cap.read()
        if not ret:
            return None

        if self.resize:
            frame = cv2.resize(frame, self.resize)

        if self.to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    def seek(self, frame_index: int) -> None:
        """Seek to specific frame."""
        if self._cap is None:
            self.open()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def seek_time(self, time_ms: float) -> None:
        """Seek to specific time in milliseconds."""
        if self._cap is None:
            self.open()
        self._cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """
        Iterate over all frames.

        Yields:
            Tuple of (frame_index, frame_array)
        """
        if self._cap is None:
            self.open()

        frame_idx = 0
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame_idx, frame
            frame_idx += 1

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        return self.frames()

    def __len__(self) -> int:
        return self.metadata.total_frames


class VideoWriter:
    """
    High-level video writer with context manager support.

    Provides convenient frame writing with automatic codec
    selection and format handling.
    """

    # Common codec configurations
    CODECS = {
        ".mp4": ("avc1", "mp4v", "H264"),
        ".avi": ("XVID", "MJPG", "DIVX"),
        ".mov": ("avc1", "mp4v"),
        ".mkv": ("X264", "XVID"),
    }

    def __init__(
        self,
        path: str,
        fps: float,
        width: int,
        height: int,
        codec: Optional[str] = None,
    ):
        """
        Initialize video writer.

        Args:
            path: Output video path
            fps: Frame rate
            width: Frame width
            height: Frame height
            codec: Specific codec to use (auto-detected if None)
        """
        self.path = path
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self._writer: Optional[cv2.VideoWriter] = None
        self._frames_written = 0

    def open(self) -> None:
        """Open video writer."""
        # Determine codec
        suffix = Path(self.path).suffix.lower()
        if self.codec:
            fourcc = cv2.VideoWriter_fourcc(*self.codec[:4])
        else:
            # Try codecs in order of preference
            codecs = self.CODECS.get(suffix, ["mp4v"])
            fourcc = None
            for codec in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec[:4])
                    break
                except Exception:
                    continue
            if fourcc is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self._writer = cv2.VideoWriter(
            self.path,
            fourcc,
            self.fps,
            (self.width, self.height),
        )

        if not self._writer.isOpened():
            raise ValueError(f"Could not open video writer: {self.path}")

    def close(self) -> None:
        """Close video writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def write_frame(self, frame: np.ndarray, is_rgb: bool = True) -> None:
        """
        Write a frame to the video.

        Args:
            frame: Frame array (H, W, 3)
            is_rgb: Whether frame is RGB (will convert to BGR)
        """
        if self._writer is None:
            self.open()

        # Convert RGB to BGR if needed
        if is_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Resize if needed
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        self._writer.write(frame)
        self._frames_written += 1

    def write_frames(self, frames: list[np.ndarray], is_rgb: bool = True) -> None:
        """Write multiple frames."""
        for frame in frames:
            self.write_frame(frame, is_rgb)

    @property
    def frames_written(self) -> int:
        """Number of frames written."""
        return self._frames_written

    def __enter__(self) -> "VideoWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def convert_video(
    input_path: str,
    output_path: str,
    target_fps: Optional[float] = None,
    target_size: Optional[tuple[int, int]] = None,
    codec: Optional[str] = None,
) -> VideoMetadata:
    """
    Convert video to different format/resolution.

    Args:
        input_path: Input video path
        output_path: Output video path
        target_fps: Target frame rate (None = same as input)
        target_size: Target (width, height) (None = same as input)
        codec: Output codec

    Returns:
        Metadata of output video
    """
    with VideoReader(input_path) as reader:
        fps = target_fps or reader.metadata.fps
        width = target_size[0] if target_size else reader.metadata.width
        height = target_size[1] if target_size else reader.metadata.height

        with VideoWriter(output_path, fps, width, height, codec) as writer:
            for _, frame in reader:
                if target_size:
                    frame = cv2.resize(frame, target_size)
                writer.write_frame(frame)

    # Return metadata of output
    with VideoReader(output_path) as out_reader:
        return out_reader.metadata


def extract_frames_to_images(
    video_path: str,
    output_dir: str,
    frame_interval: int = 1,
    image_format: str = "png",
    name_template: str = "frame_{:06d}",
) -> list[str]:
    """
    Extract frames from video to image files.

    Args:
        video_path: Input video path
        output_dir: Output directory for images
        frame_interval: Extract every Nth frame
        image_format: Output image format (png, jpg)
        name_template: Filename template with frame number placeholder

    Returns:
        List of saved image paths
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    with VideoReader(video_path, to_rgb=False) as reader:
        for frame_idx, frame in reader:
            if frame_idx % frame_interval != 0:
                continue

            filename = f"{name_template.format(frame_idx)}.{image_format}"
            filepath = output_dir / filename

            cv2.imwrite(str(filepath), frame)
            saved_paths.append(str(filepath))

    return saved_paths
