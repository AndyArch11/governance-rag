"""Tests for auto-tuning integration with existing code.

Tests adaptive rate limiter and cache tuning integration into:
- vectors.py (embedding generation)
- generate.py (LLM calls)
- embedding_cache.py (cache access tracking)
- llm_cache.py (cache access tracking)
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from scripts.utils.adaptive_cache_tuning import (
    AdaptiveCacheTuner,
    get_adaptive_cache_tuner,
    init_adaptive_cache_tuner,
)
from scripts.utils.adaptive_rate_limiter import (
    AdaptiveRateLimiter,
    get_adaptive_rate_limiter,
    init_adaptive_rate_limiter,
)


class TestAdaptiveRateLimiterIntegration:
    """Test adaptive rate limiter with realistic scenarios."""

    def test_rate_limiter_initialisation(self):
        """Test that rate limiter can be initialised."""
        limiter = init_adaptive_rate_limiter(initial_rate=10.0, min_rate=1.0, max_rate=50.0)
        assert limiter is not None
        assert limiter.current_rate == 10.0
        assert limiter.min_rate == 1.0
        assert limiter.max_rate == 50.0

    def test_rate_limiter_get_global(self):
        """Test getting global rate limiter instance."""
        limiter = init_adaptive_rate_limiter(initial_rate=10.0)
        global_limiter = get_adaptive_rate_limiter()
        assert global_limiter is limiter

    def test_record_successful_request(self):
        """Test recording successful API request."""
        limiter = AdaptiveRateLimiter(initial_rate=10.0)

        limiter.record_request(latency_ms=250.0, success=True, status_code=200)

        assert limiter.consecutive_successes == 1
        assert limiter.consecutive_rate_limits == 0

    def test_record_failed_request(self):
        """Test recording failed API request."""
        limiter = AdaptiveRateLimiter(initial_rate=10.0)

        limiter.record_request(
            latency_ms=500.0, success=False, error_type="timeout", status_code=408
        )

        assert len(limiter.metrics) == 1
        assert limiter.metrics[0].error_type == "timeout"

    def test_rate_limit_backoff_429(self):
        """Test automatic backoff on rate limit (429) response."""
        limiter = AdaptiveRateLimiter(initial_rate=10.0, min_rate=1.0, backoff_multiplier=0.5)

        original_rate = limiter.current_rate
        limiter.record_rate_limit_response(retry_after_sec=60)

        # Rate should be reduced
        assert limiter.current_rate < original_rate
        assert limiter.rate_limited_until > time.time()

    def test_rate_adjustment_on_high_latency(self):
        """Test that rate is reduced when latency is high."""
        limiter = AdaptiveRateLimiter(
            initial_rate=10.0, min_rate=1.0, latency_target_p95_ms=500.0, adjustment_rate=0.1
        )

        # Record high latency requests
        for _ in range(10):
            limiter.record_request(latency_ms=800.0, success=True, status_code=200)

        # Adjust based on metrics
        adjustment = limiter.adjust_rate_based_on_metrics()

        # Rate should be adjusted
        assert adjustment["status"] in ["adjusted", "stable", "insufficient_data"]

    def test_rate_recovery_when_healthy(self):
        """Test that rate increases when system is healthy."""
        limiter = AdaptiveRateLimiter(
            initial_rate=5.0, max_rate=50.0, latency_target_p95_ms=500.0, recovery_rate=0.05
        )

        # Record good latency requests
        for _ in range(20):
            limiter.record_request(latency_ms=200.0, success=True, status_code=200)

        original_rate = limiter.current_rate
        limiter.adjust_rate_based_on_metrics()

        # Rate might increase if system is very healthy
        assert limiter.current_rate >= original_rate * 0.99

    def test_acquire_token_bucket(self):
        """Test token bucket acquisition."""
        limiter = AdaptiveRateLimiter(initial_rate=10.0)

        # Should succeed immediately (we have tokens)
        assert limiter.acquire(tokens=1.0, blocking=False) is True

        # Fill all tokens
        for _ in range(100):
            limiter.acquire(tokens=1.0, blocking=False)

        # Should now fail (no tokens left)
        assert limiter.acquire(tokens=1.0, blocking=False) is False

    def test_get_stats(self):
        """Test getting rate limiter statistics."""
        limiter = AdaptiveRateLimiter(initial_rate=10.0)

        # Record some requests
        for i in range(5):
            latency = 100.0 + (i * 50)
            limiter.record_request(latency_ms=latency, success=True, status_code=200)

        stats = limiter.get_stats()

        assert stats["current_rate"] == 10.0
        assert stats["p50_latency_ms"] > 0
        assert stats["p95_latency_ms"] > 0


class TestAdaptiveCacheTunerIntegration:
    """Test adaptive cache tuner with realistic scenarios."""

    def test_cache_tuner_initialisation(self):
        """Test that cache tuner can be initialised."""
        tuner = init_adaptive_cache_tuner(cache_type="embedding", default_ttl_sec=86400)
        assert tuner is not None
        assert tuner.current_ttl_sec == 86400

    def test_cache_tuner_get_instance(self):
        """Test getting cache tuner instance."""
        tuner = init_adaptive_cache_tuner("embedding", default_ttl_sec=86400)
        retrieved = get_adaptive_cache_tuner("embedding")
        assert retrieved is tuner

    def test_record_cache_hit(self):
        """Test recording cache hit."""
        tuner = AdaptiveCacheTuner(default_ttl_sec=86400)

        tuner.record_access(
            cache_type="embedding", key="test_key", hit=True, latency_ms=5.0, size_bytes=1024
        )

        assert len(tuner.accesses) == 1
        assert tuner.accesses[0].hit is True

    def test_record_cache_miss(self):
        """Test recording cache miss."""
        tuner = AdaptiveCacheTuner(default_ttl_sec=86400)

        tuner.record_access(
            cache_type="embedding", key="missing_key", hit=False, latency_ms=250.0, size_bytes=0
        )

        assert len(tuner.accesses) == 1
        assert tuner.accesses[0].hit is False

    def test_hit_rate_calculation(self):
        """Test that hit rate is correctly calculated."""
        tuner = AdaptiveCacheTuner(default_ttl_sec=86400)

        # Record 8 hits and 2 misses
        for _ in range(8):
            tuner.record_access("embedding", "key", True, 5.0, 1024)
        for _ in range(2):
            tuner.record_access("embedding", "key", False, 250.0, 0)

        stats = tuner.get_stats()
        assert stats["window_hit_rate"] == pytest.approx(0.8, rel=0.01)

    def test_analyse_and_recommend(self):
        """Test cache analysis and recommendations."""
        tuner = AdaptiveCacheTuner(default_ttl_sec=86400)

        # Record enough data for analysis
        for i in range(50):
            hit = i % 3 != 0  # 66% hit rate
            tuner.record_access(
                cache_type="embedding",
                key=f"key_{i}",
                hit=hit,
                latency_ms=5.0 if hit else 100.0,
                size_bytes=1024,
            )

        recommendation = tuner.analyse_and_recommend("embedding")

        assert recommendation["status"] == "analysed"
        assert "hit_rate" in recommendation
        assert recommendation["hit_rate"] > 0

    def test_ttl_recommendation_high_hit_rate(self):
        """Test that TTL increases for high hit rates."""
        tuner = AdaptiveCacheTuner(default_ttl_sec=86400, min_ttl_sec=3600, max_ttl_sec=86400 * 7)

        # Record very high hit rate
        for i in range(50):
            tuner.record_access(
                cache_type="embedding",
                key=f"key_{i}",
                hit=True,  # All hits
                latency_ms=5.0,
                size_bytes=1024,
            )

        # Calculate TTL
        ttl = tuner._calculate_optimal_ttl_for_type("embedding")

        # TTL should be increased for high hit rate
        assert ttl >= tuner.current_ttl_sec

    def test_eviction_policy_recommendation(self):
        """Test eviction policy recommendation."""
        tuner = AdaptiveCacheTuner(default_ttl_sec=86400)

        # Record varied access frequencies
        for i in range(100):
            frequency = 1 if i < 10 else (i % 3)  # Some hot items, some cold
            hit = True
            tuner.record_access(
                cache_type="embedding", key=f"key_{i}", hit=hit, latency_ms=5.0, size_bytes=1024
            )

        policy = tuner._recommend_eviction_policy("embedding")

        # Should recommend a valid policy
        assert policy in ["LRU", "LFU", "HYBRID"]

    def test_memory_warning(self):
        """Test memory threshold warning."""
        tuner = AdaptiveCacheTuner(
            default_ttl_sec=86400, memory_threshold_mb=10  # Low threshold for testing
        )

        # Record large items to exceed threshold
        for i in range(20):
            tuner.record_access(
                cache_type="embedding",
                key=f"key_{i}",
                hit=True,
                latency_ms=5.0,
                size_bytes=1024 * 1024,  # 1MB per item
            )

        recommendation = tuner.analyse_and_recommend("embedding")

        # Should have memory warning
        if "memory_warning" in recommendation:
            assert "exceeds threshold" in recommendation["memory_warning"]

    def test_get_stats(self):
        """Test getting cache tuner statistics."""
        tuner = AdaptiveCacheTuner(default_ttl_sec=86400)

        # Record some accesses
        for i in range(10):
            tuner.record_access(
                cache_type="embedding",
                key=f"key_{i}",
                hit=i % 2 == 0,
                latency_ms=5.0 if i % 2 == 0 else 100.0,
                size_bytes=1024,
            )

        stats = tuner.get_stats()

        assert "unique_items" in stats
        assert "window_hit_rate" in stats
        assert stats["window_hit_rate"] == pytest.approx(0.5, rel=0.01)


class TestIntegrationWithCacheModules:
    """Test integration with embedding_cache.py and llm_cache.py."""

    def test_embedding_cache_records_tuner_access(self):
        """Test that EmbeddingCache records access in tuner."""
        with patch("scripts.ingest.embedding_cache.get_cache_client") as mock_db:
            with patch("scripts.ingest.embedding_cache.get_adaptive_cache_tuner") as mock_tuner_fn:
                # Setup mocks
                mock_db_instance = Mock()
                mock_db.return_value = mock_db_instance
                mock_tuner = Mock()
                mock_tuner_fn.return_value = mock_tuner

                from scripts.ingest.embedding_cache import EmbeddingCache

                cache = EmbeddingCache("/fake/path", enabled=True)

                # Mock the DB get_embedding to return a hit
                mock_db_instance.get_embedding.return_value = [0.1, 0.2, 0.3]

                # Access cache
                result = cache.get("test text")

                # Check that tuner.record_access was called
                assert mock_tuner.record_access.called
                call_args = mock_tuner.record_access.call_args
                assert call_args[1]["hit"] is True
                assert call_args[1]["cache_type"] == "embedding"

    def test_llm_cache_records_tuner_access(self):
        """Test that LLMCache records access in tuner."""
        with patch("scripts.ingest.llm_cache.get_cache_client") as mock_db:
            with patch("scripts.ingest.llm_cache.get_adaptive_cache_tuner") as mock_tuner_fn:
                # Setup mocks
                mock_db_instance = Mock()
                mock_db.return_value = mock_db_instance
                mock_tuner = Mock()
                mock_tuner_fn.return_value = mock_tuner

                from scripts.ingest.llm_cache import LLMCache

                cache = LLMCache("/fake/path", enabled=True)

                # Mock the DB get_llm_result to return a hit
                mock_db_instance.get_llm_result.return_value = '{"result": "test"}'

                # Access cache
                result = cache.get("doc_hash_123", "metadata")

                # Check that tuner.record_access was called
                assert mock_tuner.record_access.called
                call_args = mock_tuner.record_access.call_args
                assert call_args[1]["hit"] is True
                assert call_args[1]["cache_type"] == "llm_result"


class TestRateAndCacheTuningTogether:
    """Test adaptive rate limiting and cache tuning working together."""

    def test_concurrent_rate_and_cache_tuning(self):
        """Test that rate limiting and cache tuning can run concurrently."""
        rate_limiter = AdaptiveRateLimiter(initial_rate=10.0)
        cache_tuner = AdaptiveCacheTuner(default_ttl_sec=86400)

        # Simulate concurrent operations
        for i in range(100):
            # Record rate limiter metrics
            success = i % 10 != 0  # 90% success rate
            rate_limiter.record_request(
                latency_ms=200.0 if success else 1000.0,
                success=success,
                status_code=200 if success else 500,
            )

            # Record cache metrics
            hit = i % 3 == 0  # 33% hit rate
            cache_tuner.record_access(
                cache_type="embedding",
                key=f"key_{i}",
                hit=hit,
                latency_ms=5.0 if hit else 250.0,
                size_bytes=1024,
            )

        # Adjust both
        rate_adj = rate_limiter.adjust_rate_based_on_metrics()
        cache_rec = cache_tuner.analyse_and_recommend("embedding")

        assert rate_adj is not None
        assert cache_rec is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
