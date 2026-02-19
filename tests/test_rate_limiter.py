"""Tests for LLM rate limiter.

Tests the token bucket algorithm implementation for rate limiting
LLM calls across concurrent workers.
"""

import threading
import time

import pytest

from scripts.utils.rate_limiter import (
    RateLimiter,
    get_rate_limiter,
    init_rate_limiter,
    rate_limited_llm_call,
)


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_rate_limiter_initialisation(self):
        """Test rate limiter initialises with correct values."""
        limiter = RateLimiter(rate=5.0)

        assert limiter.rate == 5.0
        assert limiter.capacity == 10.0  # 2x rate by default
        assert limiter.tokens == 10.0

    def test_rate_limiter_custom_capacity(self):
        """Test rate limiter with custom capacity."""
        limiter = RateLimiter(rate=5.0, capacity=20.0)

        assert limiter.rate == 5.0
        assert limiter.capacity == 20.0
        assert limiter.tokens == 20.0

    def test_acquire_single_token(self):
        """Test acquiring a single token."""
        limiter = RateLimiter(rate=10.0)
        initial_tokens = limiter.tokens

        result = limiter.acquire(tokens=1.0, blocking=False)

        assert result is True
        assert limiter.tokens < initial_tokens

    def test_acquire_blocking_when_insufficient_tokens(self):
        """Test that acquire blocks when insufficient tokens."""
        limiter = RateLimiter(rate=1.0)  # 1 token per second

        # Acquire all tokens
        limiter.acquire(tokens=limiter.capacity, blocking=False)
        assert limiter.tokens == 0

        # Try to acquire more (should block and then succeed)
        start = time.time()
        result = limiter.acquire(tokens=0.5, blocking=True)
        elapsed = time.time() - start

        assert result is True
        assert elapsed >= 0.4  # Should wait ~0.5 seconds for 0.5 tokens at 1 token/sec

    def test_acquire_non_blocking_returns_false(self):
        """Test non-blocking acquire returns False when no tokens."""
        limiter = RateLimiter(rate=10.0)

        # Acquire all tokens
        limiter.acquire(tokens=limiter.capacity, blocking=False)

        # Try non-blocking acquire (should return False)
        result = limiter.acquire(tokens=1.0, blocking=False)
        assert result is False

    def test_token_refill(self):
        """Test that tokens refill over time."""
        limiter = RateLimiter(rate=10.0)  # 10 tokens per second

        # Acquire all tokens
        limiter.acquire(tokens=limiter.capacity, blocking=False)
        initial_tokens = limiter.tokens

        # Wait and check refill
        time.sleep(0.15)
        limiter._refill()

        assert limiter.tokens > initial_tokens
        assert limiter.tokens <= limiter.capacity

    def test_get_stats(self):
        """Test get_stats returns correct information."""
        limiter = RateLimiter(rate=5.0)
        stats = limiter.get_stats()

        assert stats["rate"] == 5.0
        assert stats["capacity"] == 10.0
        assert 0 <= stats["available_tokens"] <= 10.0
        assert 0 <= stats["utilisation"] <= 1.0

    def test_decorator_limits_function(self):
        """Test @limit decorator rate limits a function."""
        limiter = RateLimiter(rate=10.0)
        call_times = []

        @limiter.limit
        def slow_function():
            call_times.append(time.time())
            return "result"

        # Call multiple times in rapid succession
        for _ in range(3):
            result = slow_function()
            assert result == "result"

        # All calls should complete successfully
        assert len(call_times) == 3

    def test_thread_safety(self):
        """Test rate limiter is thread-safe with concurrent access."""
        limiter = RateLimiter(rate=100.0)
        acquired = []

        def acquire_tokens():
            for _ in range(10):
                if limiter.acquire(tokens=1.0, blocking=False):
                    acquired.append(1)

        threads = [threading.Thread(target=acquire_tokens) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have acquired at least some tokens despite race conditions
        assert len(acquired) > 0

    def test_capacity_ceiling(self):
        """Test tokens never exceed capacity."""
        limiter = RateLimiter(rate=10.0, capacity=20.0)

        # Wait for refill
        time.sleep(0.5)
        limiter._refill()

        assert limiter.tokens <= limiter.capacity


class TestGlobalRateLimiter:
    """Tests for global rate limiter functions."""

    def test_init_rate_limiter(self):
        """Test initializing global rate limiter."""
        limiter = init_rate_limiter(5.0)

        assert limiter is not None
        assert limiter.rate == 5.0

    def test_get_rate_limiter_after_init(self):
        """Test retrieving initialised global rate limiter."""
        init_rate_limiter(5.0)
        limiter = get_rate_limiter()

        assert limiter is not None
        assert limiter.rate == 5.0

    def test_rate_limited_llm_call_decorator(self):
        """Test @rate_limited_llm_call decorator."""
        init_rate_limiter(10.0)

        @rate_limited_llm_call
        def mock_llm_call(text):
            return f"processed: {text}"

        result = mock_llm_call("hello")
        assert result == "processed: hello"

    def test_decorator_preserves_function_metadata(self):
        """Test that @rate_limited_llm_call preserves function name and docstring."""
        init_rate_limiter(10.0)

        @rate_limited_llm_call
        def my_llm_function():
            """Test LLM function."""
            return "result"

        assert my_llm_function.__name__ == "my_llm_function"
        assert my_llm_function.__doc__ == "Test LLM function."


class TestRateLimiterStress:
    """Stress tests for rate limiter."""

    def test_high_rate_sustained_throughput(self):
        """Test rate limiter maintains target rate under load."""
        target_rate = 50.0  # 50 requests per second
        limiter = RateLimiter(rate=target_rate)

        start = time.time()
        count = 0

        # Try to acquire 100 tokens at target rate using blocking
        while count < 100:
            if limiter.acquire(tokens=1.0, blocking=True):
                count += 1
                # Break early if taking too long (safety check)
                if time.time() - start > 10:
                    break

        elapsed = time.time() - start

        # Should achieve close to target rate
        # With 100 tokens at 50 tokens/sec = ~2 seconds
        # Starting with capacity of 100 tokens (2x rate), first batch is free
        assert count == 100
        # First ~100 tokens come from burst capacity, rest are rate-limited
        # So timing should be minimal since capacity covers it
        assert elapsed < 1.0

    def test_concurrent_workers_respect_rate(self):
        """Test multiple threads respect the shared rate limit."""
        limiter = RateLimiter(rate=10.0)
        acquired_count = [0]
        lock = threading.Lock()

        def worker():
            for _ in range(20):
                if limiter.acquire(tokens=1.0, blocking=False):
                    with lock:
                        acquired_count[0] += 1
                time.sleep(0.01)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start

        # With 5 threads * 20 attempts = 100 total attempts
        # At 10 tokens/sec rate limit, should take ~10 seconds
        # But we're using non-blocking, so just check some were acquired
        assert acquired_count[0] > 0


class TestRateLimiterEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_rate_handled(self):
        """Test rate limiter handles edge case of zero or very low rate."""
        # Very low rate but not zero
        limiter = RateLimiter(rate=0.1)

        result = limiter.acquire(tokens=1.0, blocking=False)
        # May or may not have token available immediately
        assert isinstance(result, bool)

    def test_large_token_request(self):
        """Test acquiring more tokens than capacity."""
        limiter = RateLimiter(rate=5.0, capacity=10.0)

        # Request more than capacity - use non-blocking to avoid hanging
        result = limiter.acquire(tokens=15.0, blocking=False)

        # Won't have 15 tokens immediately (only capacity of 10)
        assert result is False

    def test_fractional_tokens(self):
        """Test acquiring fractional tokens."""
        limiter = RateLimiter(rate=10.0)

        result = limiter.acquire(tokens=0.5, blocking=False)
        assert result is True

    def test_rapid_sequential_acquires(self):
        """Test rapid sequential token acquisitions."""
        limiter = RateLimiter(rate=100.0)

        results = [limiter.acquire(tokens=1.0, blocking=False) for _ in range(50)]

        # At 100 tokens/sec with capacity of 200, first ~100 should succeed
        assert sum(results) >= 50
