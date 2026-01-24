"""
Performance timing utilities.

Provides timing and performance monitoring for the video
processing pipeline.
"""

import time
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TimingStats:
    """Statistics for a timed operation."""
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        """Average time in milliseconds."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    def update(self, duration_ms: float) -> None:
        """Update statistics with new timing."""
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)

    def __repr__(self) -> str:
        return (
            f"TimingStats(count={self.count}, avg={self.avg_ms:.2f}ms, "
            f"min={self.min_ms:.2f}ms, max={self.max_ms:.2f}ms)"
        )


class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        with Timer("operation") as t:
            # code to time
        print(f"Operation took {t.elapsed_ms:.2f}ms")
    """

    def __init__(self, name: Optional[str] = None, print_on_exit: bool = False):
        """
        Initialize timer.

        Args:
            name: Optional name for the timer
            print_on_exit: Print elapsed time when exiting context
        """
        self.name = name
        self.print_on_exit = print_on_exit
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def start(self) -> "Timer":
        """Start the timer."""
        self._start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time in ms."""
        self._end_time = time.perf_counter()
        return self.elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.perf_counter()
        return (end - self._start_time) * 1000

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed_ms / 1000

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
        if self.print_on_exit:
            name = f"[{self.name}] " if self.name else ""
            print(f"{name}Elapsed: {self.elapsed_ms:.2f}ms")


class PerformanceMonitor:
    """
    Monitor and aggregate performance metrics across operations.

    Tracks timing statistics for multiple named operations,
    useful for profiling the video processing pipeline.
    """

    def __init__(self):
        """Initialize performance monitor."""
        self._stats: dict[str, TimingStats] = defaultdict(TimingStats)
        self._start_times: dict[str, float] = {}
        self._total_start: Optional[float] = None

    def start_session(self) -> None:
        """Start a new monitoring session."""
        self._total_start = time.perf_counter()
        self._stats.clear()
        self._start_times.clear()

    def start(self, name: str) -> None:
        """Start timing an operation."""
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """
        Stop timing an operation.

        Args:
            name: Operation name

        Returns:
            Elapsed time in milliseconds
        """
        if name not in self._start_times:
            return 0.0

        elapsed_ms = (time.perf_counter() - self._start_times[name]) * 1000
        self._stats[name].update(elapsed_ms)
        del self._start_times[name]
        return elapsed_ms

    def record(self, name: str, duration_ms: float) -> None:
        """Record a timing directly."""
        self._stats[name].update(duration_ms)

    def time(self, name: str) -> Timer:
        """
        Get a timer context manager that auto-records to this monitor.

        Usage:
            with monitor.time("operation"):
                # code to time
        """
        return _MonitoredTimer(self, name)

    def get_stats(self, name: str) -> TimingStats:
        """Get statistics for an operation."""
        return self._stats.get(name, TimingStats())

    def get_all_stats(self) -> dict[str, TimingStats]:
        """Get all statistics."""
        return dict(self._stats)

    @property
    def total_elapsed_ms(self) -> float:
        """Total elapsed time since session start."""
        if self._total_start is None:
            return 0.0
        return (time.perf_counter() - self._total_start) * 1000

    def get_summary(self) -> str:
        """Get a formatted summary of all timing statistics."""
        lines = ["Performance Summary", "=" * 50]

        if self._total_start:
            lines.append(f"Total Time: {self.total_elapsed_ms:.2f}ms")
            lines.append("")

        # Sort by total time descending
        sorted_stats = sorted(
            self._stats.items(),
            key=lambda x: x[1].total_ms,
            reverse=True,
        )

        for name, stats in sorted_stats:
            pct = (stats.total_ms / self.total_elapsed_ms * 100) if self.total_elapsed_ms > 0 else 0
            lines.append(
                f"{name:30s}: {stats.avg_ms:8.2f}ms avg | "
                f"{stats.count:6d} calls | {stats.total_ms:10.2f}ms total ({pct:5.1f}%)"
            )

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print timing summary to console."""
        print(self.get_summary())


class _MonitoredTimer:
    """Timer that records to a PerformanceMonitor."""

    def __init__(self, monitor: PerformanceMonitor, name: str):
        self.monitor = monitor
        self.name = name
        self._timer: Optional[Timer] = None

    def __enter__(self) -> Timer:
        self._timer = Timer(self.name)
        self._timer.start()
        return self._timer

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._timer:
            self._timer.stop()
            self.monitor.record(self.name, self._timer.elapsed_ms)
