"""LLM output caching for ingest pipeline using SQLite backend.

Caches expensive LLM calls (metadata extraction, summaries, validations) keyed
by prompt hash to avoid redundant processing. Cache is persisted in SQLite
and shared across ingestion runs.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

from scripts.utils.adaptive_cache_tuning import get_adaptive_cache_tuner
from scripts.utils.db_factory import get_cache_client


class LLMCache:
    """Persistent SQLite cache for LLM outputs during document ingestion.

    Caches results from expensive LLM operations (metadata generation,
    summaries, chunk validation) to skip redundant calls when document
    content hasn't changed between ingestion runs.

    Uses CacheDB SQLite backend for persistence and thread safety.
    Provides TTL-based expiration for cache entries.

    Cache entries are keyed by: {document_hash}_{operation_type}

    Attributes:
        max_age_days: Maximum age of cache entries (0 = no expiry).
    """

    def __init__(self, cache_path: str, enabled: bool = True, max_age_days: int = 0) -> None:
        """Initialise LLM cache using SQLite backend.

        Args:
            cache_path: Ignored (for API compatibility); uses rag_data/cache.db
            enabled: Enable/disable caching (useful for testing).
            max_age_days: Expire entries older than this (0 = no expiry).
        """
        # Extract rag_data_path from cache_path if provided
        # Expected path format: .../rag_data/cache/llm_cache.json
        rag_data_path = None
        if cache_path and "rag_data" in cache_path:
            path_parts = Path(cache_path).parts
            if "rag_data" in path_parts:
                rag_data_idx = path_parts.index("rag_data")
                rag_data_path = Path(*path_parts[: rag_data_idx + 1])

        # Get SQLite cache backend
        self.db = get_cache_client(rag_data_path=rag_data_path, enable_cache=enabled)

        self.enabled = enabled
        self.max_age_days = max_age_days
        self.entry_timestamps = {}  # Track timestamps for TTL

        # Initialise adaptive cache tuner
        self.tuner = get_adaptive_cache_tuner("llm_result")

    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry has expired.

        Args:
            timestamp: Entry creation timestamp.

        Returns:
            True if entry is expired, False otherwise.
        """
        if self.max_age_days == 0:
            return False

        age_seconds = time.time() - timestamp
        age_days = age_seconds / 86400
        return age_days > self.max_age_days

    def get(self, doc_hash: str, operation: str) -> Optional[Any]:
        """Retrieve cached LLM output.

        Args:
            doc_hash: MD5 hash of document content.
            operation: Operation type (e.g., 'metadata', 'summary').

        Returns:
            Cached result if found and not expired, None otherwise.
        """
        if not self.enabled:
            return None

        # Create compound key for LLM cache query
        prompt_key = f"{doc_hash}_{operation}"

        start_ms = time.perf_counter() * 1000
        result_str = self.db.get_llm_result(prompt_key, model="ingest")
        latency_ms = time.perf_counter() * 1000 - start_ms

        hit = result_str is not None

        if result_str is None:
            # Record miss in tuner
            if self.tuner:
                self.tuner.record_access(
                    cache_type="llm_result",
                    key=prompt_key,
                    hit=False,
                    latency_ms=latency_ms,
                    size_bytes=0,
                )
            return None

        # Check TTL if configured
        if prompt_key in self.entry_timestamps:
            if self._is_expired(self.entry_timestamps[prompt_key]):
                # Remove expired entry
                del self.entry_timestamps[prompt_key]
                if self.tuner:
                    self.tuner.record_access(
                        cache_type="llm_result",
                        key=prompt_key,
                        hit=False,
                        latency_ms=latency_ms,
                        size_bytes=0,
                    )
                return None

        # Record hit in tuner
        if self.tuner:
            self.tuner.record_access(
                cache_type="llm_result",
                key=prompt_key,
                hit=True,
                latency_ms=latency_ms,
                size_bytes=len(result_str) if result_str else 0,
            )

        # Try to parse as JSON, fall back to string
        try:
            return json.loads(result_str)
        except (json.JSONDecodeError, TypeError):
            return result_str

    def put(self, doc_hash: str, operation: str, result: Any) -> None:
        """Store LLM output in cache.

        Args:
            doc_hash: MD5 hash of document content.
            operation: Operation type (e.g., 'metadata', 'summary').
            result: LLM output to cache (must be JSON-serialisable).
        """
        if not self.enabled:
            return

        prompt_key = f"{doc_hash}_{operation}"
        timestamp = time.time()

        # Convert result to JSON string for storage
        result_str = json.dumps(result) if not isinstance(result, str) else result

        # Store in SQLite cache
        self.db.put_llm_result(prompt_key, result_str, model="ingest")

        # Track timestamp for TTL
        self.entry_timestamps[prompt_key] = timestamp

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with 'total_entries' count.
        """
        db_stats = self.db.llm_stats()
        return {
            "total_entries": db_stats.get("entries", 0),
            "expired_entries": sum(
                1 for ts in self.entry_timestamps.values() if self._is_expired(ts)
            ),
        }

    def flush(self) -> None:
        """Flush cache to storage (no-op for SQLite; auto-commits)."""
        pass

    def clear(self) -> None:
        """Clear all LLM results from cache."""
        if not self.enabled:
            return
        self.db.clear_llm_cache()
        self.entry_timestamps.clear()

    def close(self) -> None:
        """Close database connection."""
        if self.db:
            self.db.close()
