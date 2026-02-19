"""Tests for context caching module."""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.rag.context_cache import ContextCache, get_context_cache


@pytest.fixture
def temp_cache_file(tmp_path):
    """Create temporary cache file."""
    return tmp_path / "test_cache.json"


@pytest.fixture
def cache(temp_cache_file):
    """Create test cache instance."""
    return ContextCache(
        max_entries=5,
        default_ttl=1,  # 1 second for fast tests
        cache_file=temp_cache_file,
        enabled=True,
    )


def test_cache_put_and_get(cache):
    """Test basic put/get operations."""
    cache.put(
        entity="test_entity",
        context="Test context content",
        chunk_ids=["chunk1", "chunk2"],
    )

    result = cache.get("test_entity")
    assert result == "Test context content"


def test_cache_miss(cache):
    """Test cache miss returns None."""
    result = cache.get("nonexistent_entity")
    assert result is None


def test_cache_expiration(cache):
    """Test TTL expiration."""
    cache.put(
        entity="expiring_entity",
        context="Will expire soon",
        chunk_ids=["chunk1"],
        ttl=1,  # 1 second
    )

    # Should hit immediately
    assert cache.get("expiring_entity") == "Will expire soon"

    # Wait for expiration
    time.sleep(1.5)

    # Should miss after expiration
    assert cache.get("expiring_entity") is None


def test_cache_lru_eviction(cache):
    """Test LRU eviction when at capacity."""
    # Fill cache to max (5 entries)
    for i in range(5):
        cache.put(
            entity=f"entity_{i}",
            context=f"Content {i}",
            chunk_ids=[f"chunk_{i}"],
        )

    # All should be cached
    assert cache.get("entity_0") is not None

    # Add one more - should evict LRU
    cache.put(
        entity="entity_5",
        context="Content 5",
        chunk_ids=["chunk_5"],
    )

    # entity_1 should be evicted (entity_0 was just accessed above)
    assert cache.get("entity_1") is None
    assert cache.get("entity_5") is not None


def test_cache_access_count(cache):
    """Test access count tracking."""
    cache.put(entity="popular", context="Hot entity", chunk_ids=["c1"])

    # Access multiple times
    for _ in range(5):
        cache.get("popular")

    stats = cache.get_stats()
    assert stats["total_accesses"] >= 5


def test_cache_invalidate(cache):
    """Test manual invalidation."""
    cache.put(entity="removeme", context="Temporary", chunk_ids=["c1"])
    assert cache.get("removeme") is not None

    # Invalidate
    result = cache.invalidate("removeme")
    assert result is True
    assert cache.get("removeme") is None

    # Invalidating again returns False
    assert cache.invalidate("removeme") is False


def test_cache_clear(cache):
    """Test clearing all entries."""
    cache.put(entity="e1", context="Content 1", chunk_ids=["c1"])
    cache.put(entity="e2", context="Content 2", chunk_ids=["c2"])

    cache.clear()

    assert cache.get("e1") is None
    assert cache.get("e2") is None
    assert cache.get_stats()["total_entries"] == 0


def test_cache_persistence(temp_cache_file):
    """Test save/load persistence."""
    # Create cache and add entry
    cache1 = ContextCache(cache_file=temp_cache_file, enabled=True)
    cache1.put(entity="persistent", context="Saved data", chunk_ids=["c1"])

    # Create new cache instance - should load from disk
    cache2 = ContextCache(cache_file=temp_cache_file, enabled=True)
    assert cache2.get("persistent") == "Saved data"


def test_cache_disabled():
    """Test cache with enabled=False."""
    cache = ContextCache(enabled=False)
    cache.put(entity="test", context="Should not cache", chunk_ids=["c1"])

    # Should always miss when disabled
    assert cache.get("test") is None


def test_cache_stats(cache):
    """Test statistics retrieval."""
    cache.put(entity="e1", context="C1", chunk_ids=["c1"])
    cache.put(entity="e2", context="C2", chunk_ids=["c2"])
    cache.get("e1")
    cache.get("e1")
    cache.get("e2")

    stats = cache.get_stats()
    assert stats["total_entries"] == 2
    # Access counts include the initial put (which counts as 1 access)
    # e1: 1 (put) + 2 (gets) = 3, e2: 1 (put) + 1 (get) = 2, total = 5
    assert stats["total_accesses"] >= 3  # At least the explicit gets
    assert stats["cache_enabled"] is True
    assert len(stats["hot_entities"]) == 2
    # e1 should be hotter (more accesses)
    assert stats["hot_entities"][0]["entity"] == "e1"


def test_cache_metadata(cache):
    """Test storing metadata with cache entries."""
    metadata = {"source": "test", "priority": "high"}
    cache.put(
        entity="with_meta",
        context="Content",
        chunk_ids=["c1"],
        metadata=metadata,
    )

    # Metadata is stored internally
    assert cache._cache["with_meta"]["metadata"] == metadata


def test_global_cache_instance(tmp_path):
    """Test global cache singleton."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    cache1 = get_context_cache(cache_dir=cache_dir, enabled=True)
    cache2 = get_context_cache()

    # Should return same instance
    assert cache1 is cache2
