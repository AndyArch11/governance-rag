"""Unified rate limiting for LLM API calls and external services.

Implements both fixed-rate and adaptive rate limiting:

## Fixed-Rate Limiter (RateLimiter)
Simple token bucket algorithm for smooth rate limiting across concurrent workers.
Thread-safe for use in multi-threaded ingestion pipeline.

## Adaptive Rate Limiter (AdaptiveRateLimiter)
Automatically adjusts concurrency and backoff strategy based on:
- Response latencies (p50, p95, p99)
- Error rates and error types
- Success rates
- Upstream rate limit signals (429, 503)

## Configuration Recommendations

### Ollama LLM Calls
- **Rate Limit:** 10-20 calls/sec (adjust based on GPU)
- **Retries:** 3 attempts
- **Initial Delay:** 1.0 second
- **Backoff:** Exponential 2x

### ChromaDB Operations
- **Retries:** 3-5 attempts (query = 3, write = 5)
- **Initial Delay:** 0.5 seconds
- **Backoff:** Exponential 2x
- **Rate Limit:** None (local service)

### Bitbucket API
- **Rate Limit:** 1000 requests/hour = ~0.28 req/sec
- **Retries:** 3-5 attempts
- **Initial Delay:** 1.0 second
- **Throttle:** 100ms between requests (existing)

### PDF/HTML Parsing
- **Retries:** 2 attempts
- **Initial Delay:** 0.5 seconds
- **Transient Types:** IOError, MemoryError, TimeoutError

"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


# ============================================================================
# Fixed-Rate Limiter
# ============================================================================


class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm.

    Limits the rate of function calls across multiple threads.
    Useful for preventing Ollama GPU saturation when using many workers.

    Attributes:
        rate: Maximum calls per second.
        capacity: Maximum burst size (tokens in bucket).
        tokens: Current available tokens.
        last_update: Timestamp of last token refill.
        lock: Threading lock for thread-safe operations.

    Example:
        >>> limiter = RateLimiter(rate=10.0)  # 10 calls/sec max
        >>>
        >>> @limiter.limit
        ... def expensive_llm_call(text):
        ...     return llm.invoke(text)
        >>>
        >>> # Will block if rate exceeded
        >>> result = expensive_llm_call("prompt")
    """

    def __init__(self, rate: float = 10.0, capacity: Optional[float] = None):
        """Initialise rate limiter.

        Args:
            rate: Maximum calls per second (e.g., 10.0 = 10 calls/sec).
            capacity: Maximum burst size. Defaults to 2x rate.
        """
        self.rate = rate
        self.capacity = capacity if capacity is not None else rate * 2
        self.tokens = self.capacity
        self.last_update = time.time()
        self.lock = threading.Lock()

    def _refill(self) -> None:
        """Refill token bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

    def acquire(self, tokens: float = 1.0, blocking: bool = True) -> bool:
        """Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire (default 1.0).
            blocking: If True, wait until tokens available. If False, return immediately.

        Returns:
            True if tokens acquired, False if not available (only when blocking=False).

        Behaviour:
            - If enough tokens available: consume and return immediately
            - If insufficient tokens (blocking=True): sleep and retry
            - If insufficient tokens (blocking=False): return False
        """
        while True:
            with self.lock:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                if not blocking:
                    return False

                # Calculate sleep time needed for next token
                deficit = tokens - self.tokens
                sleep_time = deficit / self.rate

            # Sleep outside lock to allow other threads
            time.sleep(min(sleep_time, 0.1))

    def limit(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to rate-limit a function.

        Args:
            func: Function to rate-limit.

        Returns:
            Rate-limited wrapper function.

        Example:
            >>> limiter = RateLimiter(rate=5.0)
            >>>
            >>> @limiter.limit
            ... def call_api(data):
            ...     return api.invoke(data)
        """

        def wrapper(*args, **kwargs) -> T:
            self.acquire()
            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    def get_stats(self) -> dict:
        """Get current rate limiter statistics.

        Returns:
            Dictionary with rate, capacity, and available tokens.
        """
        with self.lock:
            self._refill()
            return {
                "rate": self.rate,
                "capacity": self.capacity,
                "available_tokens": self.tokens,
                "utilisation": 1.0 - (self.tokens / self.capacity),
            }


# ============================================================================
# Adaptive Rate Limiter
# ============================================================================


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    timestamp: float
    latency_ms: float
    success: bool
    error_type: Optional[str] = None  # e.g., "timeout", "rate_limited", "server_error"
    status_code: Optional[int] = None  # HTTP status code if applicable


@dataclass
class WindowStats:
    """Statistics for a time window."""

    window_start: float
    window_end: float
    requests: int = 0
    successes: int = 0
    failures: int = 0
    error_rate: float = 0.0
    latencies_ms: list = field(default_factory=list)
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    rate_limited_count: int = 0  # 429 responses
    server_error_count: int = 0  # 5xx responses


class AdaptiveRateLimiter:
    """Automatically tunes concurrency and backoff based on system health.

    Maintains metrics over sliding windows and adjusts rate up/down based on:
    - Latency trends (if increasing, reduce rate)
    - Error rates (if increasing, reduce rate)
    - Rate limit signals (aggressive backoff on 429)
    - Server errors (moderate backoff on 5xx)

    Features:
    - Non-blocking rate checking (returns immediately if over limit)
    - Automatic backoff on rate limit signals
    - Gradual increase when healthy (prevents aggressive scaling)
    - Thread-safe with minimal lock contention
    - Logging of rate adjustments
    """

    def __init__(
        self,
        initial_rate: float = 10.0,
        min_rate: float = 1.0,
        max_rate: float = 50.0,
        sample_window_sec: int = 60,
        latency_target_p95_ms: float = 500.0,
        target_error_rate: float = 0.01,
        adjustment_rate: float = 0.1,
        backoff_multiplier: float = 0.5,
        recovery_rate: float = 0.05,
    ):
        """Initialise adaptive rate limiter.

        Args:
            initial_rate: Starting rate in calls/sec
            min_rate: Minimum rate (safety floor)
            max_rate: Maximum rate (safety ceiling)
            sample_window_sec: Metrics collection window (seconds)
            latency_target_p95_ms: Target p95 latency (ms)
            target_error_rate: Target error rate (0.0-1.0)
            adjustment_rate: Rate of change per window (0.0-1.0)
            backoff_multiplier: Multiplier for rate limit backoff (0.5 = 50%)
            recovery_rate: Rate of increase when healthy (5% per window)
        """
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.current_rate = initial_rate
        self.sample_window_sec = sample_window_sec
        self.latency_target_p95_ms = latency_target_p95_ms
        self.target_error_rate = target_error_rate
        self.adjustment_rate = adjustment_rate
        self.backoff_multiplier = backoff_multiplier
        self.recovery_rate = recovery_rate

        # Metrics tracking
        self.metrics: deque = deque(maxlen=10000)
        self.lock = threading.Lock()
        self.last_adjustment_time = time.time()

        # Token bucket for rate limiting
        self.tokens = self.initial_rate * 2  # Start with 2x capacity
        self.capacity = self.initial_rate * 2
        self.last_refill = time.time()

        # Rate limit detection
        self.rate_limited_until = 0.0  # Timestamp when we can retry after 429
        self.consecutive_rate_limits = 0
        self.consecutive_successes = 0

    def record_request(
        self,
        latency_ms: float,
        success: bool,
        error_type: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        """Record metrics for a request.

        Args:
            latency_ms: Request latency in milliseconds
            success: Whether request succeeded
            error_type: Type of error (timeout, rate_limited, server_error, etc.)
            status_code: HTTP status code if applicable
        """
        with self.lock:
            metric = RequestMetrics(
                timestamp=time.time(),
                latency_ms=latency_ms,
                success=success,
                error_type=error_type,
                status_code=status_code,
            )
            self.metrics.append(metric)

            # Track rate limit detection
            if status_code == 429:
                self.consecutive_rate_limits += 1
                self.consecutive_successes = 0
            elif success:
                self.consecutive_successes += 1
                self.consecutive_rate_limits = 0
            else:
                self.consecutive_rate_limits = 0
                self.consecutive_successes = 0

    def should_backoff_429(self) -> bool:
        """Check if we should backoff due to rate limit signal."""
        return self.consecutive_rate_limits >= 2

    def record_rate_limit_response(self, retry_after_sec: Optional[float] = None) -> None:
        """Record that we received a rate limit response (429, 503).

        Args:
            retry_after_sec: Seconds to wait before retry (from Retry-After header)
        """
        with self.lock:
            # Aggressive backoff on rate limit
            wait_time = retry_after_sec or (60.0 / self.current_rate)
            self.rate_limited_until = time.time() + wait_time
            self.consecutive_rate_limits += 1

            # Reduce rate
            new_rate = max(self.min_rate, self.current_rate * self.backoff_multiplier)
            if new_rate != self.current_rate:
                logger.warning(
                    f"Rate limited (429). Reducing rate: {self.current_rate:.2f} → {new_rate:.2f} req/sec. "
                    f"Backoff until {datetime.fromtimestamp(self.rate_limited_until)}"
                )
                self._update_rate(new_rate)

    def _get_window_stats(self) -> WindowStats:
        """Calculate statistics for current window.

        Returns:
            WindowStats with metrics for last sample_window_sec seconds
        """
        now = time.time()
        window_start = now - self.sample_window_sec

        # Filter metrics in window
        window_metrics = [m for m in self.metrics if m.timestamp >= window_start]

        if not window_metrics:
            return WindowStats(
                window_start=window_start,
                window_end=now,
            )

        # Calculate statistics
        successes = sum(1 for m in window_metrics if m.success)
        failures = len(window_metrics) - successes
        error_rate = failures / len(window_metrics) if window_metrics else 0.0

        latencies = [m.latency_ms for m in window_metrics]
        latencies_sorted = sorted(latencies)

        def percentile(data, p):
            if not data:
                return 0.0
            idx = int(len(data) * p / 100.0)
            return data[min(idx, len(data) - 1)]

        rate_limited_count = sum(1 for m in window_metrics if m.status_code == 429)
        server_error_count = sum(
            1 for m in window_metrics if m.status_code and 500 <= m.status_code < 600
        )

        return WindowStats(
            window_start=window_start,
            window_end=now,
            requests=len(window_metrics),
            successes=successes,
            failures=failures,
            error_rate=error_rate,
            latencies_ms=latencies,
            p50_latency=percentile(latencies_sorted, 50),
            p95_latency=percentile(latencies_sorted, 95),
            p99_latency=percentile(latencies_sorted, 99),
            rate_limited_count=rate_limited_count,
            server_error_count=server_error_count,
        )

    def _update_rate(self, new_rate: float) -> None:
        """Update current rate and token bucket capacity.

        Args:
            new_rate: New rate in calls/sec
        """
        old_rate = self.current_rate
        self.current_rate = max(self.min_rate, min(self.max_rate, new_rate))
        self.capacity = self.current_rate * 2
        self.tokens = min(self.tokens, self.capacity)
        self.last_adjustment_time = time.time()

        if abs(self.current_rate - old_rate) > 0.01:  # Only log significant changes
            logger.info(
                f"Rate adjusted: {old_rate:.2f} → {self.current_rate:.2f} req/sec "
                f"(capacity: {self.capacity:.2f})"
            )

    def adjust_rate_based_on_metrics(self) -> Dict[str, Any]:
        """Analyse metrics and adjust rate based on feedback control.

        Called periodically to tune the rate limiter based on observed behaviour.

        Returns:
            Dictionary with adjustment details for logging
        """
        with self.lock:
            stats = self._get_window_stats()
            now = time.time()

            # Not enough data to adjust
            if stats.requests < 5:
                return {"status": "insufficient_data", "requests": stats.requests}

            # Check if still under backoff from rate limit
            if now < self.rate_limited_until:
                remaining = self.rate_limited_until - now
                return {
                    "status": "backoff_active",
                    "remaining_sec": remaining,
                    "current_rate": self.current_rate,
                }

            old_rate = self.current_rate
            adjustment_reason = []

            # 1. Latency-based adjustment (p95 > target → reduce)
            if stats.p95_latency > self.latency_target_p95_ms:
                reduction = (
                    stats.p95_latency / self.latency_target_p95_ms - 1.0
                ) * self.adjustment_rate
                new_rate = self.current_rate * (1.0 - reduction)
                self._update_rate(new_rate)
                adjustment_reason.append(
                    f"high_latency(p95={stats.p95_latency:.0f}ms, target={self.latency_target_p95_ms:.0f}ms)"
                )

            # 2. Error rate adjustment (error_rate > target → reduce)
            elif stats.error_rate > self.target_error_rate:
                reduction = (stats.error_rate / self.target_error_rate - 1.0) * self.adjustment_rate
                new_rate = self.current_rate * (1.0 - reduction)
                self._update_rate(new_rate)
                adjustment_reason.append(
                    f"high_error_rate(rate={stats.error_rate:.1%}, target={self.target_error_rate:.1%})"
                )

            # 3. Recovery: gradually increase when healthy
            elif (
                stats.error_rate <= self.target_error_rate * 0.5
                and stats.p95_latency <= self.latency_target_p95_ms * 0.8
                and self.current_rate < self.max_rate
            ):
                new_rate = self.current_rate * (1.0 + self.recovery_rate)
                self._update_rate(new_rate)
                adjustment_reason.append(
                    f"healthy_recovery(error={stats.error_rate:.1%}, p95={stats.p95_latency:.0f}ms)"
                )

            return {
                "status": "adjusted" if self.current_rate != old_rate else "stable",
                "previous_rate": old_rate,
                "current_rate": self.current_rate,
                "requests_in_window": stats.requests,
                "error_rate": stats.error_rate,
                "p50_latency_ms": stats.p50_latency,
                "p95_latency_ms": stats.p95_latency,
                "p99_latency_ms": stats.p99_latency,
                "rate_limited_responses": stats.rate_limited_count,
                "server_errors": stats.server_error_count,
                "reason": " | ".join(adjustment_reason) if adjustment_reason else "system_healthy",
            }

    def acquire(self, tokens: float = 1.0, blocking: bool = False) -> bool:
        """Acquire tokens from rate limiter.

        Args:
            tokens: Tokens to acquire (default 1.0)
            blocking: If False, return immediately (non-blocking check)

        Returns:
            True if tokens acquired, False otherwise
        """
        # Check if under backoff from rate limit
        if time.time() < self.rate_limited_until:
            return False

        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill

            # Refill tokens
            self.tokens = min(self.capacity, self.tokens + elapsed * self.current_rate)
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            if not blocking:
                return False

        # For blocking: calculate sleep time
        deficit = tokens - self.tokens
        sleep_time = deficit / self.current_rate
        time.sleep(min(sleep_time, 1.0))
        return self.acquire(tokens, blocking=False)

    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limiter statistics.

        Returns:
            Dictionary with current rate, capacity, and recent metrics
        """
        with self.lock:
            stats = self._get_window_stats()
            return {
                "current_rate": self.current_rate,
                "min_rate": self.min_rate,
                "max_rate": self.max_rate,
                "capacity": self.capacity,
                "available_tokens": self.tokens,
                "utilisation": 1.0 - (self.tokens / self.capacity) if self.capacity > 0 else 0.0,
                "window_requests": stats.requests,
                "window_successes": stats.successes,
                "window_failures": stats.failures,
                "error_rate": stats.error_rate,
                "p50_latency_ms": stats.p50_latency,
                "p95_latency_ms": stats.p95_latency,
                "p99_latency_ms": stats.p99_latency,
                "rate_limited_signals": stats.rate_limited_count,
                "server_errors": stats.server_error_count,
                "backoff_active": time.time() < self.rate_limited_until,
            }


# ============================================================================
# Global Instances and Factory Functions
# ============================================================================

# Global rate limiter instance (initialised in main())
_global_limiter: Optional[RateLimiter] = None

# Global adaptive rate limiter instance
_global_adaptive_limiter: Optional[AdaptiveRateLimiter] = None


def init_rate_limiter(rate: float) -> RateLimiter:
    """Initialise global rate limiter.

    Args:
        rate: Maximum LLM calls per second.

    Returns:
        Initialised RateLimiter instance.
    """
    global _global_limiter
    _global_limiter = RateLimiter(rate=rate)
    return _global_limiter


def get_rate_limiter() -> Optional[RateLimiter]:
    """Get global rate limiter instance.

    Returns:
        Global RateLimiter, or None if not initialised.
    """
    return _global_limiter


def init_adaptive_rate_limiter(
    initial_rate: float = 10.0,
    min_rate: float = 1.0,
    max_rate: float = 50.0,
) -> AdaptiveRateLimiter:
    """Initialise global adaptive rate limiter.

    Args:
        initial_rate: Starting rate in calls/sec
        min_rate: Minimum rate (safety floor)
        max_rate: Maximum rate (safety ceiling)

    Returns:
        Initialised AdaptiveRateLimiter
    """
    global _global_adaptive_limiter
    _global_adaptive_limiter = AdaptiveRateLimiter(
        initial_rate=initial_rate,
        min_rate=min_rate,
        max_rate=max_rate,
    )
    logger.info(
        f"Adaptive rate limiter initialised: "
        f"initial={initial_rate:.1f}, min={min_rate:.1f}, max={max_rate:.1f} req/sec"
    )
    return _global_adaptive_limiter


def get_adaptive_rate_limiter() -> Optional[AdaptiveRateLimiter]:
    """Get global adaptive rate limiter instance.

    Returns:
        Global AdaptiveRateLimiter, or None if not initialised
    """
    return _global_adaptive_limiter


def rate_limited_llm_call(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for rate-limiting LLM calls using global limiter.

    Args:
        func: LLM function to rate-limit.

    Returns:
        Rate-limited wrapper function.

    Example:
        >>> @rate_limited_llm_call
        ... def clean_text_with_llm(text, **kwargs):
        ...     return llm.invoke(text)
    """

    def wrapper(*args, **kwargs) -> T:
        limiter = get_rate_limiter()
        if limiter:
            limiter.acquire()
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
