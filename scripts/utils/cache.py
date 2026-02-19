"""Base cache classes with common functionality for different cache types.

Provides unified caching infrastructure with:
- Thread-safe operations
- JSON persistence
- TTL expiration
- LRU eviction
- Hit/miss tracking
- Auto-save capabilities

Cache Types:
    BaseCache: Abstract base with core functionality
    SimpleCache: Basic key-value cache with persistence
    TTLCache: Cache with time-to-live expiration
    LRUCache: Cache with least-recently-used eviction
"""

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


class BaseCache(ABC, Generic[T]):
    """Abstract base cache with common functionality.

    Provides thread-safe persistence, statistics tracking, and lifecycle management.
    Subclasses implement specific caching strategies (TTL, LRU, etc.).

    Attributes:
        cache_path: Path to JSON cache file
        enabled: Whether caching is active
        cache: In-memory cache dictionary
        lock: Threading lock for thread-safe operations
        hits: Total cache hits (successful retrievals)
        misses: Total cache misses (not found)
    """

    def __init__(self, cache_path: str, enabled: bool = True):
        """Initialise base cache.

        Args:
            cache_path: Path to JSON cache file
            enabled: Whether to enable caching
        """
        self.cache_path = Path(cache_path)
        self.enabled = enabled
        self.cache: Dict[str, Any] = {}
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

        if enabled and self.cache_path.exists():
            self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.cache = data.get("entries", {})
                self.hits = data.get("stats", {}).get("hits", 0)
                self.misses = data.get("stats", {}).get("misses", 0)
        except Exception:
            # If load fails, start with empty cache
            self.cache = {}

    def _save(self) -> None:
        """Persist cache to disk."""
        if not self.enabled:
            return

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "entries": self.cache,
                "stats": {
                    "hits": self.hits,
                    "misses": self.misses,
                    "total_entries": len(self.cache),
                },
            }
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            # Silent failure - cache is optional
            pass

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Retrieve value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found, None otherwise
        """
        pass

    @abstractmethod
    def put(self, key: str, value: T) -> None:
        """Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        pass

    def delete(self, key: str) -> bool:
        """Remove entry from cache.

        Args:
            key: Cache key to remove

        Returns:
            True if key existed, False otherwise
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
        if self.enabled:
            self._save()

    def flush(self) -> None:
        """Write cache to disk immediately."""
        self._save()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with hit rate, total entries, etc.
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "total_entries": len(self.cache),
            "enabled": self.enabled,
        }

    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute MD5 hash of text for cache key.

        Args:
            text: Text to hash

        Returns:
            Hexadecimal MD5 hash string
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()


class SimpleCache(BaseCache[T]):
    """Simple key-value cache with auto-save.

    Provides basic caching with periodic persistence to disk.
    No eviction or expiration - cache grows unbounded.
    """

    def __init__(
        self,
        cache_path: str,
        enabled: bool = True,
        auto_save_interval: int = 50,
    ):
        """Initialise simple cache.

        Args:
            cache_path: Path to JSON cache file
            enabled: Whether to enable caching
            auto_save_interval: Save to disk every N puts
        """
        super().__init__(cache_path, enabled)
        self.auto_save_interval = auto_save_interval
        self._puts_since_save = 0

    def get(self, key: str) -> Optional[T]:
        """Retrieve value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found, None otherwise
        """
        if not self.enabled:
            return None

        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                entry["hits"] = entry.get("hits", 0) + 1
                self.hits += 1
                return entry.get("value")

            self.misses += 1
            return None

    def put(self, key: str, value: T) -> None:
        """Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if not self.enabled:
            return

        with self.lock:
            self.cache[key] = {
                "value": value,
                "hits": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._puts_since_save += 1

        # Auto-save periodically (outside lock)
        if self._puts_since_save >= self.auto_save_interval:
            self.flush()
            self._puts_since_save = 0


class TTLCache(BaseCache[T]):
    """Cache with time-to-live expiration.

    Entries expire after a configurable TTL period.
    Expired entries are removed on access.
    """

    def __init__(
        self,
        cache_path: str,
        enabled: bool = True,
        default_ttl: int = 3600,
    ):
        """Initialise TTL cache.

        Args:
            cache_path: Path to JSON cache file
            enabled: Whether to enable caching
            default_ttl: Default time-to-live in seconds (default: 1 hour)
        """
        super().__init__(cache_path, enabled)
        self.default_ttl = default_ttl

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if entry has expired.

        Args:
            entry: Cache entry with timestamp and ttl

        Returns:
            True if expired, False otherwise
        """
        timestamp = entry.get("timestamp", 0)
        ttl = entry.get("ttl", self.default_ttl)
        age = time.time() - timestamp
        return age > ttl

    def get(self, key: str) -> Optional[T]:
        """Retrieve value from cache if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        if not self.enabled:
            return None

        with self.lock:
            if key in self.cache:
                entry = self.cache[key]

                # Check expiration
                if self._is_expired(entry):
                    del self.cache[key]
                    self.misses += 1
                    return None

                entry["hits"] = entry.get("hits", 0) + 1
                self.hits += 1
                return entry.get("value")

            self.misses += 1
            return None

    def put(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Store value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        if not self.enabled:
            return

        with self.lock:
            self.cache[key] = {
                "value": value,
                "timestamp": time.time(),
                "ttl": ttl if ttl is not None else self.default_ttl,
                "hits": 0,
            }


class LRUCache(BaseCache[T]):
    """Cache with least-recently-used eviction.

    Maintains a maximum size. When full, evicts least recently accessed entry.
    Tracks access times for LRU eviction policy.
    """

    def __init__(
        self,
        cache_path: str,
        enabled: bool = True,
        max_entries: int = 100,
        default_ttl: Optional[int] = None,
        auto_save_interval: int = 10,
    ):
        """Initialise LRU cache.

        Args:
            cache_path: Path to JSON cache file
            enabled: Whether to enable caching
            max_entries: Maximum number of cached entries
            default_ttl: Optional TTL in seconds (None = no expiration)
            auto_save_interval: Save to disk every N puts
        """
        super().__init__(cache_path, enabled)
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.auto_save_interval = auto_save_interval
        self._puts_since_save = 0

    def _evict_lru(self) -> None:
        """Evict least recently used entry.

        Must be called with lock held.
        """
        if not self.cache:
            return

        # Find entry with oldest last_accessed time
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].get("last_accessed", 0),
        )
        del self.cache[lru_key]

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if entry has expired (if TTL is set).

        Args:
            entry: Cache entry

        Returns:
            True if expired, False otherwise
        """
        if self.default_ttl is None:
            return False

        created_at = entry.get("created_at", 0)
        age = time.time() - created_at
        return age > self.default_ttl

    def get(self, key: str) -> Optional[T]:
        """Retrieve value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        if not self.enabled:
            return None

        with self.lock:
            if key in self.cache:
                entry = self.cache[key]

                # Check expiration (if TTL is set)
                if self._is_expired(entry):
                    del self.cache[key]
                    self.misses += 1
                    return None

                # Update access stats
                entry["last_accessed"] = time.time()
                entry["access_count"] = entry.get("access_count", 0) + 1
                self.hits += 1
                return entry.get("value")

            self.misses += 1
            return None

    def put(self, key: str, value: T) -> None:
        """Store value in cache with LRU eviction.

        Args:
            key: Cache key
            value: Value to cache
        """
        if not self.enabled:
            return

        with self.lock:
            # Evict if at capacity and key is new
            if len(self.cache) >= self.max_entries and key not in self.cache:
                self._evict_lru()

            now = time.time()
            self.cache[key] = {
                "value": value,
                "created_at": now,
                "last_accessed": now,
                "access_count": 1,
            }
            self._puts_since_save += 1

        # Auto-save periodically (outside lock)
        if self._puts_since_save >= self.auto_save_interval:
            self.flush()
            self._puts_since_save = 0
