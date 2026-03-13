"""Unit tests for cache classes.

Tests coverage for BaseCache, SimpleCache, TTLCache, and LRUCache implementations
including persistence, expiration, eviction, and statistics tracking.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.utils.cache import BaseCache, LRUCache, SimpleCache, TTLCache


class TestBaseCache:
    """Test BaseCache abstract base class functionality."""

    def test_compute_hash(self) -> None:
        """Test SHA-256 hash computation."""
        text = "test string"
        hash_value = BaseCache.compute_hash(text)

        # Should produce consistent hash
        assert BaseCache.compute_hash(text) == hash_value
        assert len(hash_value) == 64  # SHA-256 is 64 hex characters

    def test_compute_hash_different_inputs(self) -> None:
        """Test that different inputs produce different hashes."""
        hash1 = BaseCache.compute_hash("text1")
        hash2 = BaseCache.compute_hash("text2")
        assert hash1 != hash2


class TestSimpleCache:
    """Test SimpleCache basic key-value operations."""

    def test_init_creates_cache(self) -> None:
        """Test cache initialisation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = SimpleCache(str(cache_path))

            assert cache.enabled is True
            assert len(cache.cache) == 0
            assert cache.hits == 0
            assert cache.misses == 0

    def test_put_and_get(self) -> None:
        """Test basic put and get operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = SimpleCache(str(cache_path))

            cache.put("key1", "value1")
            assert cache.get("key1") == "value1"

    def test_get_nonexistent_key(self) -> None:
        """Test getting non-existent key returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = SimpleCache(str(cache_path))

            assert cache.get("nonexistent") is None

    def test_hit_miss_tracking(self) -> None:
        """Test cache hit and miss statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = SimpleCache(str(cache_path))

            cache.put("key1", "value1")
            cache.get("key1")  # Hit
            cache.get("key1")  # Hit
            cache.get("missing")  # Miss

            assert cache.hits == 2
            assert cache.misses == 1

    def test_delete(self) -> None:
        """Test delete operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = SimpleCache(str(cache_path))

            cache.put("key1", "value1")
            assert cache.delete("key1") is True
            assert cache.get("key1") is None
            assert cache.delete("key1") is False  # Already deleted

    def test_clear(self) -> None:
        """Test clearing all cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = SimpleCache(str(cache_path))

            cache.put("key1", "value1")
            cache.put("key2", "value2")
            cache.get("key1")  # Generate a hit

            cache.clear()

            assert len(cache.cache) == 0
            assert cache.hits == 0
            assert cache.misses == 0

    def test_persistence_save_load(self) -> None:
        """Test cache persists to disk and loads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"

            # Create cache and add entries
            cache1 = SimpleCache(str(cache_path))
            cache1.put("key1", "value1")
            cache1.put("key2", {"nested": "dict"})
            cache1.flush()

            # Load cache from disk
            cache2 = SimpleCache(str(cache_path))
            assert cache2.get("key1") == "value1"
            assert cache2.get("key2") == {"nested": "dict"}

    def test_auto_save_interval(self) -> None:
        """Test auto-save triggers at interval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = SimpleCache(str(cache_path), auto_save_interval=2)

            # Add entries - should not save yet
            cache.put("key1", "value1")
            cache_path.unlink(missing_ok=True)

            # Add second entry - should trigger auto-save at interval
            cache.put("key2", "value2")
            assert cache_path.exists(), "Auto-save should have triggered"

    def test_disabled_cache(self) -> None:
        """Test cache operations when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = SimpleCache(str(cache_path), enabled=False)

            cache.put("key1", "value1")
            assert cache.get("key1") is None
            assert not cache_path.exists()

    def test_stats(self) -> None:
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = SimpleCache(str(cache_path))

            cache.put("key1", "value1")
            cache.put("key2", "value2")
            cache.get("key1")  # Hit
            cache.get("missing")  # Miss

            stats = cache.stats()
            assert stats["hits"] == 1
            assert stats["misses"] == 1
            assert stats["total_requests"] == 2
            assert stats["hit_rate"] == 0.5
            assert stats["total_entries"] == 2
            assert stats["enabled"] is True


class TestTTLCache:
    """Test TTLCache expiration functionality."""

    def test_entry_not_expired(self) -> None:
        """Test non-expired entry is returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = TTLCache(str(cache_path), default_ttl=10)

            cache.put("key1", "value1")
            assert cache.get("key1") == "value1"

    def test_entry_expired(self) -> None:
        """Test expired entry is not returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            with patch("scripts.utils.cache.time") as mock_time:
                mock_time.time.return_value = 1000.0
                cache = TTLCache(str(cache_path), default_ttl=1)
                cache.put("key1", "value1")

                mock_time.time.return_value = 1002.0  # 2s later, past TTL of 1s
                assert cache.get("key1") is None
                assert cache.misses == 1

    def test_custom_ttl_per_entry(self) -> None:
        """Test custom TTL can be set per entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            with patch("scripts.utils.cache.time") as mock_time:
                mock_time.time.return_value = 1000.0
                cache = TTLCache(str(cache_path), default_ttl=100)
                # Entry with short TTL
                cache.put("short", "value1", ttl=1)
                # Entry with long TTL
                cache.put("long", "value2", ttl=100)

                mock_time.time.return_value = 1002.0  # 2s later: short expired, long still valid
                assert cache.get("short") is None
                assert cache.get("long") == "value2"

    def test_ttl_hit_miss_tracking(self) -> None:
        """Test hit/miss tracking with expiration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            with patch("scripts.utils.cache.time") as mock_time:
                mock_time.time.return_value = 1000.0
                cache = TTLCache(str(cache_path), default_ttl=1)
                cache.put("key1", "value1")
                cache.get("key1")  # Hit before expiration

                mock_time.time.return_value = 1002.0  # 2s later, past TTL of 1s
                cache.get("key1")  # Miss after expiration

                assert cache.hits == 1
                assert cache.misses == 1


class TestLRUCache:
    """Test LRUCache eviction and functionality."""

    def test_lru_eviction_at_capacity(self) -> None:
        """Test LRU eviction when max capacity reached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = LRUCache(str(cache_path), max_entries=3)

            cache.put("key1", "value1")
            cache.put("key2", "value2")
            cache.put("key3", "value3")

            # Access key1 to make it recently used
            cache.get("key1")

            # Add fourth entry - should evict key2 (least recently used)
            cache.put("key4", "value4")

            assert cache.get("key1") == "value1"
            assert cache.get("key2") is None  # Evicted
            assert cache.get("key3") == "value3"
            assert cache.get("key4") == "value4"

    def test_lru_access_updates_timestamp(self) -> None:
        """Test that get operations update access timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = LRUCache(str(cache_path), max_entries=2)

            cache.put("key1", "value1")
            cache.put("key2", "value2")

            # Access key1 to make it recently used
            cache.get("key1")

            # Add key3 - should evict key2 (not accessed recently)
            cache.put("key3", "value3")

            assert cache.get("key1") == "value1"
            assert cache.get("key2") is None  # Evicted
            assert cache.get("key3") == "value3"

    def test_lru_with_ttl_expiration(self) -> None:
        """Test LRU cache with TTL expiration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            with patch("scripts.utils.cache.time") as mock_time:
                mock_time.time.return_value = 1000.0
                cache = LRUCache(str(cache_path), max_entries=10, default_ttl=1)
                cache.put("key1", "value1")
                cache.put("key2", "value2")

                mock_time.time.return_value = 1002.0  # 2s later, past TTL of 1s
                # Both should be expired
                assert cache.get("key1") is None
                assert cache.get("key2") is None

    def test_lru_no_ttl(self) -> None:
        """Test LRU cache without TTL expiration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = LRUCache(str(cache_path), max_entries=10, default_ttl=None)

            cache.put("key1", "value1")
            time.sleep(1)

            # Should still be there (no TTL)
            assert cache.get("key1") == "value1"

    def test_lru_max_entries_respected(self) -> None:
        """Test that max_entries limit is respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = LRUCache(str(cache_path), max_entries=5)

            for i in range(10):
                cache.put(f"key{i}", f"value{i}")

            assert len(cache.cache) <= 5

    def test_lru_access_count_tracking(self) -> None:
        """Test access count is tracked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = LRUCache(str(cache_path), max_entries=10)

            cache.put("key1", "value1")
            cache.get("key1")
            cache.get("key1")
            cache.get("key1")

            entry = cache.cache["key1"]
            assert entry["access_count"] == 4  # 1 from put + 3 from gets

    def test_lru_auto_save_interval(self) -> None:
        """Test auto-save with LRU cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = LRUCache(str(cache_path), auto_save_interval=2)

            cache.put("key1", "value1")
            cache_path.unlink(missing_ok=True)

            cache.put("key2", "value2")
            assert cache_path.exists()

    def test_lru_stats(self) -> None:
        """Test LRU cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = LRUCache(str(cache_path), max_entries=10)

            cache.put("key1", "value1")
            cache.get("key1")
            cache.get("missing")

            stats = cache.stats()
            assert stats["hits"] == 1
            assert stats["misses"] == 1
            assert stats["total_entries"] == 1


class TestCacheThreadSafety:
    """Test thread safety of cache operations."""

    def test_concurrent_puts(self) -> None:
        """Test multiple threads can safely put values."""
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = SimpleCache(str(cache_path))

            def add_entries(start: int, count: int) -> None:
                for i in range(start, start + count):
                    cache.put(f"key{i}", f"value{i}")

            threads = [
                threading.Thread(target=add_entries, args=(0, 100)),
                threading.Thread(target=add_entries, args=(100, 100)),
                threading.Thread(target=add_entries, args=(200, 100)),
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(cache.cache) == 300

    def test_concurrent_gets(self) -> None:
        """Test multiple threads can safely get values."""
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            cache = SimpleCache(str(cache_path))

            # Pre-populate
            for i in range(100):
                cache.put(f"key{i}", f"value{i}")

            results = []

            def read_entries(count: int) -> None:
                for i in range(count):
                    val = cache.get(f"key{i % 100}")
                    results.append(val)

            threads = [
                threading.Thread(target=read_entries, args=(100,)),
                threading.Thread(target=read_entries, args=(100,)),
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All reads should succeed
            assert len(results) == 200
            assert all(v in [f"value{i}" for i in range(100)] for v in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
