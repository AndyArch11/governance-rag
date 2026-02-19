"""Simple performance monitoring utilities for dashboard rendering.

Tracks named timing segments using a monotonic clock and returns a summary
suitable for display in the UI or logging. Designed to be lightweight
and dependency-free.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PerformanceMonitor:
    """Collects timing information for labeled segments."""

    start_time: float = field(default_factory=time.monotonic)
    segments: Dict[str, float] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)

    def record(self, label: str) -> None:
        """Record the elapsed time since the previous mark or start."""
        now = time.monotonic()
        elapsed = now - self.start_time
        self.segments[label] = elapsed
        self.order.append(label)
        self.start_time = now

    def snapshot(self) -> Dict[str, float]:
        """Return a copy of recorded timings."""
        return dict(self.segments)

    def reset(self) -> None:
        """Clear segments and restart the timer."""
        self.segments.clear()
        self.order.clear()
        self.start_time = time.monotonic()


def format_timings_ms(segments: Dict[str, float]) -> Dict[str, float]:
    """Convert timings to milliseconds for readability."""
    return {label: round(sec * 1000.0, 2) for label, sec in segments.items()}
