"""Tests for EmbeddingCache utility.

Verifies basic get/put, batch operations, stats, and persistence behaviours.
"""

import os

# Add ingest directory to path
import sys
from pathlib import Path

import pytest

scripts_ingest_path = Path(__file__).parent.parent / "scripts" / "ingest"
if str(scripts_ingest_path) not in sys.path:
    sys.path.insert(0, str(scripts_ingest_path))

from embedding_cache import EmbeddingCache  # type: ignore


def test_get_put_and_stats(tmp_path):
    cache_file = tmp_path / "embedding_cache.json"
    cache = EmbeddingCache(str(cache_file), enabled=True, auto_save_interval=10)
    
    # Clear any existing cache state from previous tests
    cache.clear()

    assert cache.get("hello") is None
    cache.put("hello", [0.1, 0.2])
    assert cache.get("hello") == [0.1, 0.2]

    stats = cache.stats()
    assert stats["total_entries"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert 0.0 <= stats["hit_rate"] <= 1.0
    
    cache.close()


def test_batch_ops(tmp_path):
    cache_file = tmp_path / "embedding_cache.json"
    cache = EmbeddingCache(str(cache_file), enabled=True, auto_save_interval=10)

    cache.put("a", [1.0])
    cache.put("c", [3.0])

    embs, uncached = cache.get_batch(["a", "b", "c"])
    assert embs[0] == [1.0]
    assert embs[1] is None
    assert embs[2] == [3.0]
    assert uncached == ["b"]

    cache.put_batch(["b"], [[2.0]])
    assert cache.get("b") == [2.0]
    
    cache.close()


def test_persistence(tmp_path):
    cache_file = tmp_path / "embedding_cache.json"
    cache = EmbeddingCache(str(cache_file), enabled=True, auto_save_interval=1)
    cache.put("x", [0.5])
    cache.flush()
    cache.close()

    # New instance loads previous entries
    cache2 = EmbeddingCache(str(cache_file), enabled=True)
    assert cache2.get("x") == [0.5]
    cache2.close()
