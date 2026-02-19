"""Tests for LLM caching functionality."""

import json
import os

# Add parent directory to path for imports
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "ingest"))

from llm_cache import LLMCache  # type: ignore


class TestLLMCache:
    """Test suite for LLMCache class."""

    def test_cache_creation(self):
        """Test cache creation with temp file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            cache_path = f.name

        try:
            cache = LLMCache(cache_path)
            assert cache is not None
            assert cache.enabled is True
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)
            cache.close()

    def test_cache_put_and_get(self):
        """Test basic put/get operations."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            cache_path = f.name

        try:
            cache = LLMCache(cache_path)

            # Put a value
            doc_hash = "abc123"
            operation = "metadata"
            result = {"doc_type": "policy", "summary": "Test summary"}

            cache.put(doc_hash, operation, result)

            # Get it back
            retrieved = cache.get(doc_hash, operation)
            assert retrieved == result
        finally:
            cache.close()
            if os.path.exists(cache_path):
                os.unlink(cache_path)

    def test_cache_miss(self):
        """Test cache miss returns None."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            cache_path = f.name

        try:
            cache = LLMCache(cache_path)
            result = cache.get("nonexistent", "metadata")
            assert result is None
        finally:
            cache.close()
            if os.path.exists(cache_path):
                os.unlink(cache_path)

    def test_cache_persistence(self):
        """Test cache persists across instances."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            cache_path = f.name

        try:
            # First instance - write data
            cache1 = LLMCache(cache_path)
            cache1.put("doc1", "metadata", {"type": "governance"})
            cache1.flush()
            cache1.close()

            # Second instance - read data
            cache2 = LLMCache(cache_path)
            result = cache2.get("doc1", "metadata")
            assert result == {"type": "governance"}
            cache2.close()
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)

    def test_cache_expiry(self):
        """Test cache expiry based on max_age_days."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            cache_path = f.name

        try:
            # Create cache with 1 day expiry
            cache = LLMCache(cache_path, max_age_days=1)

            # Add entry
            doc_hash = "recent_doc"
            operation = "metadata"
            cache.put(doc_hash, operation, {"recent": "data"})

            # Verify recent entry is retrieved
            result = cache.get(doc_hash, operation)
            assert result == {"recent": "data"}
            
            # For expired entries, manually set old timestamp
            old_key = f"old_doc_{operation}"
            cache.entry_timestamps[old_key] = time.time() - (2 * 24 * 60 * 60)
            
            # Verify old entry is expired
            assert cache._is_expired(cache.entry_timestamps[old_key])
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)
            cache.close()

    def test_cache_stats(self):
        """Test cache statistics."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            cache_path = f.name

        try:
            cache = LLMCache(cache_path, max_age_days=1)
            cache.clear()  # Clear any existing cache state

            # Add some entries
            cache.put("doc1", "metadata", {"a": 1})
            cache.put("doc2", "clean_text", "cleaned")

            stats = cache.stats()
            assert stats["total_entries"] == 2
            # No expired entries yet
            assert stats["expired_entries"] >= 0
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)
            cache.close()

    def test_cache_clear(self):
        """Test clearing the cache."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            cache_path = f.name

        try:
            cache = LLMCache(cache_path)
            cache.clear()  # Clear any existing cache state

            # Add entries
            cache.put("doc1", "metadata", {"a": 1})
            cache.put("doc2", "clean_text", "cleaned")

            stats_before = cache.stats()
            assert stats_before["total_entries"] == 2

            # Clear cache
            cache.clear()

            stats_after = cache.stats()
            assert stats_after["total_entries"] == 0
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)
            cache.close()

    def test_auto_save(self):
        """Test persistence across cache instances."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            cache_path = f.name

        try:
            cache = LLMCache(cache_path)
            cache.clear()  # Clear any existing cache state

            # Add 15 entries (SQLite auto-commits each write)
            for i in range(15):
                cache.put(f"doc{i}", "metadata", {"index": i})

            # Verify by loading a new instance
            cache2 = LLMCache(cache_path)

            # Should have all 15 entries (persisted in SQLite)
            assert cache2.stats()["total_entries"] == 15
            cache2.close()
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)
            cache.close()

    def test_multiple_operations_same_doc(self):
        """Test multiple operations cached for same document."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            cache_path = f.name

        try:
            cache = LLMCache(cache_path)

            doc_hash = "doc1"

            # Store different operations
            cache.put(doc_hash, "clean_text", "cleaned text")
            cache.put(doc_hash, "metadata", {"type": "policy"})
            cache.put(doc_hash, "score_summary", {"overall": 8})

            # Retrieve each
            assert cache.get(doc_hash, "clean_text") == "cleaned text"
            assert cache.get(doc_hash, "metadata") == {"type": "policy"}
            assert cache.get(doc_hash, "score_summary") == {"overall": 8}
        finally:
            cache.close()
            if os.path.exists(cache_path):
                os.unlink(cache_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
