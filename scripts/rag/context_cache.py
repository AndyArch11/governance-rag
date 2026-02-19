"""Context caching for frequently accessed entities and topics.

Caches expanded context bundles to reduce repeated chunk retrieval
and LLM calls for common entities, components, or technical topics.

Benefits:
- Instant retrieval for cached entities (no search overhead)
- Reduced ChromaDB queries
- Better for resource-constrained devices
- Consistent context for repeated queries

Cache Strategy:
- LRU eviction (least recently used)
- TTL-based expiration (1 hour default)
- Access count tracking for hot entities
- Configurable max size
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.utils.cache import LRUCache


class ContextCache(LRUCache):
    """In-memory cache for entity contexts with persistence.

    Stores pre-computed context bundles for frequently accessed entities,
    reducing retrieval overhead and improving response time.

    Inherits LRU eviction and persistence from LRUCache.
    Adds per-entry TTL, entity metadata, and access tracking.
    """

    def __init__(
        self,
        max_entries: int = 100,
        default_ttl: int = 3600,
        cache_file: Optional[Path] = None,
        enabled: bool = True,
    ):
        """Initialise context cache.

        Args:
            max_entries: Maximum number of cached entries (LRU eviction)
            default_ttl: Default time-to-live in seconds (3600 = 1 hour)
            cache_file: Optional file path for persistence
            enabled: Whether caching is active
        """
        # Initialise parent LRUCache with persistence and auto-save
        cache_path = str(cache_file) if cache_file else "/tmp/context_cache.json"
        super().__init__(
            cache_path,
            max_entries=max_entries,
            enabled=enabled,
            auto_save_interval=1,  # Save every put for context cache (important for persistence tests)
        )

        self.default_ttl = default_ttl
        self.cache_file = cache_file

        # Metrics tracking
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_requests = 0

    def get(self, entity: str) -> Optional[str]:
        """Retrieve cached context for an entity.

        Args:
            entity: Entity identifier (e.g., "auth_service", "database_config")

        Returns:
            Cached context string or None if not found/expired
        """
        self._total_requests += 1

        if not self.enabled:
            self._cache_misses += 1
            return None

        if entity not in self.cache:
            self._cache_misses += 1
            return None

        # Get the parent's wrapping structure
        entry_wrapper = self.cache[entity]
        # Extract our custom entry from "value" field
        entry = entry_wrapper.get("value", {}) if isinstance(entry_wrapper, dict) else {}

        # Check if expired (use wrapper's created_at timestamp)
        if self._is_expired(entry_wrapper):
            del self.cache[entity]
            self.flush()
            self._cache_misses += 1
            return None

        # Cache hit!
        self._cache_hits += 1

        # Update access stats (update both wrapper and value)
        entry_wrapper["last_accessed"] = time.time()
        entry_wrapper["access_count"] = entry_wrapper.get("access_count", 0) + 1
        if "last_accessed_iso" in entry:
            entry["last_accessed_iso"] = datetime.now(timezone.utc).isoformat()
        if "access_count" in entry:
            entry["access_count"] = entry.get("access_count", 0) + 1

        return entry.get("context")

    def put(
        self,
        entity: str,
        context: str,
        chunk_ids: List[str],
        ttl: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Store context for an entity.

        Args:
            entity: Entity identifier
            context: Expanded context text
            chunk_ids: List of chunk IDs that comprise this context
            ttl: Time-to-live in seconds (uses default if None)
            metadata: Optional metadata about this context
        """
        if not self.enabled:
            return

        now = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Build the entry structure that LRUCache expects
        # with our custom fields nested inside
        entry_value = {
            "context": context,
            "chunk_ids": chunk_ids,
            "created_at_iso": now_iso,
            "last_accessed_iso": now_iso,
            "access_count": 1,
            "ttl": ttl or self.default_ttl,
            "metadata": metadata or {},
        }

        # Use parent's put which handles LRU eviction
        # It will wrap this in {value: entry_value, created_at: now, ...}
        super().put(entity, entry_value)

    def invalidate(self, entity: str) -> bool:
        """Remove an entity from cache.

        Args:
            entity: Entity to invalidate

        Returns:
            True if entity was cached, False otherwise
        """
        if entity in self.cache:
            del self.cache[entity]
            self.flush()
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including hit rate metrics.

        Returns:
            Dict with cache metrics:
            - total_entries: Current number of cached items
            - max_entries: Maximum cache capacity
            - total_accesses: Sum of all entity access counts
            - expired_entries: Number of expired entries
            - cache_enabled: Whether cache is active
            - hot_entities: Top 5 most accessed entities
            - cache_hits: Number of successful cache retrievals
            - cache_misses: Number of cache misses
            - total_requests: Total get() calls
            - hit_rate: Percentage of requests served from cache
        """
        total_accesses = sum(e.get("access_count", 0) for e in self.cache.values())
        expired_count = sum(1 for e in self.cache.values() if self._is_expired(e))

        hit_rate = (
            (self._cache_hits / self._total_requests * 100) if self._total_requests > 0 else 0.0
        )

        return {
            "total_entries": len(self.cache),
            "max_entries": self.max_entries,
            "total_accesses": total_accesses,
            "expired_entries": expired_count,
            "cache_enabled": self.enabled,
            "hot_entities": self._get_hot_entities(5),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_requests": self._total_requests,
            "hit_rate": round(hit_rate, 2),
        }

    def _is_expired(self, entry_wrapper: Dict) -> bool:
        """Check if cache entry has expired.

        Args:
            entry_wrapper: The wrapper from parent LRUCache with created_at timestamp

        Returns:
            True if expired, False otherwise
        """
        # Use parent's TTL logic which checks the float timestamp
        # The entry_wrapper contains {"created_at": float_timestamp, ...}
        if self.default_ttl is None:
            return False

        created_at = entry_wrapper.get("created_at", 0)
        age = time.time() - created_at

        # Get the per-entry TTL from the nested value if available
        entry_value = entry_wrapper.get("value", {})
        ttl = entry_value.get("ttl", self.default_ttl)

        return age > ttl

    def _get_hot_entities(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most frequently accessed entities.

        Args:
            limit: Number of top entities to return

        Returns:
            List of dicts with entity and access_count
        """
        sorted_entities = sorted(
            self.cache.items(), key=lambda x: x[1].get("access_count", 0), reverse=True
        )
        return [
            {"entity": entity, "access_count": data.get("access_count", 0)}
            for entity, data in sorted_entities[:limit]
        ]

    @property
    def _cache(self) -> Dict[str, Any]:
        """Backward compatibility property for tests accessing internal _cache.

        Returns a flattened view where each entry includes its nested value fields.
        """
        result = {}
        for key, wrapper in self.cache.items():
            # Flatten the structure: merge wrapper and value fields
            if isinstance(wrapper, dict) and "value" in wrapper:
                result[key] = {**wrapper.get("value", {})}
            else:
                result[key] = wrapper
        return result


# Global cache instance (lazy initialisation)
_global_cache: Optional[ContextCache] = None


def get_context_cache(cache_dir: Optional[Path] = None, enabled: bool = True) -> ContextCache:
    """Get or create global context cache instance.

    Args:
        cache_dir: Directory for cache file persistence
        enabled: Whether caching is active

    Returns:
        ContextCache instance
    """
    global _global_cache

    if _global_cache is None:
        cache_file = None
        if cache_dir:
            cache_file = cache_dir / "context_cache.json"

        _global_cache = ContextCache(
            max_entries=100, default_ttl=3600, cache_file=cache_file, enabled=enabled
        )

    return _global_cache
