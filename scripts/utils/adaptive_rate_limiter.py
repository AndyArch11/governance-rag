"""Adaptive rate limiting (deprecated - use rate_limiter.py instead).

This module is kept for backward compatibility. All functionality has been
consolidated into scripts.utils.rate_limiter.

For new code, import directly from rate_limiter:
    from scripts.utils.rate_limiter import (
        AdaptiveRateLimiter,
        RequestMetrics,
        WindowStats,
        init_adaptive_rate_limiter,
        get_adaptive_rate_limiter,
    )
"""

# Re-export everything from rate_limiter for backward compatibility
from scripts.utils.rate_limiter import (
    AdaptiveRateLimiter,
    RequestMetrics,
    WindowStats,
    get_adaptive_rate_limiter,
    init_adaptive_rate_limiter,
)

__all__ = [
    "AdaptiveRateLimiter",
    "RequestMetrics",
    "WindowStats",
    "init_adaptive_rate_limiter",
    "get_adaptive_rate_limiter",
]
