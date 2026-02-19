"""Adaptive cache TTL and eviction tuning.

Automatically adjusts cache parameters based on observed hit/miss patterns:
- TTL tuning: Extends TTL for frequently-hit items, shortens for rarely-used
- Eviction policy: Prefers LRU (least recently used) vs LFU (least frequently used)
- Cache size: Recommends cache size based on hit rate trends
- Memory monitoring: Warns if cache exceeds memory threshold

Goal: Maximise cache efficiency (hit rate) while minimising memory footprint.

Metrics tracked:
- Hit rate (requests served from cache / total requests)
- Miss rate (requests not in cache / total requests)
- Item frequency distribution
- Item recency distribution
- Memory usage (if available)
- TTL effectiveness (items accessed before expiry)

Tuning strategy:
1. Analyse hit/miss patterns per item type (embedding, llm_result)
2. Calculate optimal TTL based on access patterns
3. Recommend eviction policy (LRU vs LFU vs hybrid)
4. Predict optimal cache size from hit rate trajectory
5. Alert if memory usage exceeds threshold
"""

import heapq
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheAccess:
    """Single cache access record."""

    timestamp: float
    cache_type: str  # "embedding" or "llm_result"
    key: str
    hit: bool  # True if cache hit, False if miss
    latency_ms: float  # Time to serve from cache or generate new


@dataclass
class ItemStats:
    """Statistics for a cached item."""

    key: str
    cache_type: str
    access_count: int = 0
    hit_count: int = 0
    last_access_time: float = 0.0
    creation_time: float = 0.0
    expiry_time: Optional[float] = None
    size_bytes: int = 0


@dataclass
class CacheWindowStats:
    """Cache statistics for a time window."""

    window_start: float
    window_end: float
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    avg_hit_latency_ms: float = 0.0
    avg_miss_latency_ms: float = 0.0
    items_tracked: int = 0
    total_cache_size_bytes: int = 0
    expiry_effectiveness: float = 0.0  # % of items accessed before expiry


class AdaptiveCacheTuner:
    """Automatically optimises cache TTL, eviction policy, and size.

    Features:
    - Tracks hit/miss patterns by item type
    - Calculates optimal TTL based on access frequency and recency
    - Recommends eviction policy (LRU vs LFU)
    - Estimates optimal cache size
    - Memory usage monitoring
    - Per-type tuning (embeddings vs LLM results have different patterns)
    """

    def __init__(
        self,
        min_ttl_sec: int = 3600,  # 1 hour minimum
        max_ttl_sec: int = 86400 * 7,  # 7 days maximum
        default_ttl_sec: int = 86400,  # 1 day default
        sample_window_sec: int = 300,  # 5 minutes
        memory_threshold_mb: int = 1024,  # 1GB threshold
        hit_rate_threshold: float = 0.7,  # 70% target hit rate
    ):
        """Initialise adaptive cache tuner.

        Args:
            min_ttl_sec: Minimum TTL to recommend
            max_ttl_sec: Maximum TTL to recommend
            default_ttl_sec: Current default TTL
            sample_window_sec: Window for metrics aggregation
            memory_threshold_mb: Cache memory warning threshold
            hit_rate_threshold: Target hit rate (0.0-1.0)
        """
        self.min_ttl_sec = min_ttl_sec
        self.max_ttl_sec = max_ttl_sec
        self.current_ttl_sec = default_ttl_sec
        self.sample_window_sec = sample_window_sec
        self.memory_threshold_mb = memory_threshold_mb
        self.hit_rate_threshold = hit_rate_threshold

        # Access tracking
        self.accesses: deque[CacheAccess] = deque(maxlen=100000)
        self.item_stats: Dict[str, ItemStats] = {}
        self.lock = threading.Lock()

        # Per-type tracking for separate analysis
        self.cache_types = {"embedding", "llm_result"}
        self.type_stats: Dict[str, Dict[str, Any]] = {
            cache_type: {
                "ttl_recommendation": default_ttl_sec,
                "hit_rate": 0.0,
                "eviction_policy": "LRU",
            }
            for cache_type in self.cache_types
        }

    def record_access(
        self,
        cache_type: str,
        key: str,
        hit: bool,
        latency_ms: float,
        size_bytes: int = 0,
    ) -> None:
        """Record a cache access event.

        Args:
            cache_type: Type of cache ("embedding" or "llm_result")
            key: Cache key/item identifier
            hit: Whether this was a cache hit
            latency_ms: Time to serve (cache hit) or generate (miss)
            size_bytes: Size of the item (for memory tracking)
        """
        with self.lock:
            access = CacheAccess(
                timestamp=time.time(),
                cache_type=cache_type,
                key=key,
                hit=hit,
                latency_ms=latency_ms,
            )
            self.accesses.append(access)

            # Update item stats
            if key not in self.item_stats:
                self.item_stats[key] = ItemStats(
                    key=key,
                    cache_type=cache_type,
                    creation_time=access.timestamp,
                    size_bytes=size_bytes,
                )

            item = self.item_stats[key]
            item.access_count += 1
            item.last_access_time = access.timestamp
            if hit:
                item.hit_count += 1

    def _get_window_stats(self, cache_type: Optional[str] = None) -> CacheWindowStats:
        """Calculate statistics for current window.

        Args:
            cache_type: Filter to specific cache type (None = all types)

        Returns:
            CacheWindowStats with metrics for last sample_window_sec seconds
        """
        now = time.time()
        window_start = now - self.sample_window_sec

        # Filter accesses in window
        window_accesses = [
            a
            for a in self.accesses
            if a.timestamp >= window_start and (cache_type is None or a.cache_type == cache_type)
        ]

        if not window_accesses:
            return CacheWindowStats(
                window_start=window_start,
                window_end=now,
            )

        # Calculate statistics
        hits = sum(1 for a in window_accesses if a.hit)
        misses = len(window_accesses) - hits
        hit_rate = hits / len(window_accesses) if window_accesses else 0.0

        # Latencies by hit/miss
        hit_latencies = [a.latency_ms for a in window_accesses if a.hit]
        miss_latencies = [a.latency_ms for a in window_accesses if not a.hit]

        avg_hit_latency = sum(hit_latencies) / len(hit_latencies) if hit_latencies else 0.0
        avg_miss_latency = sum(miss_latencies) / len(miss_latencies) if miss_latencies else 0.0

        # Total cache size
        total_size = sum(
            self.item_stats[a.key].size_bytes for a in window_accesses if a.key in self.item_stats
        )

        # Expiry effectiveness: items accessed before they expire
        items_before_expiry = sum(
            1
            for a in window_accesses
            if a.key in self.item_stats
            and self.item_stats[a.key].expiry_time
            and a.timestamp < self.item_stats[a.key].expiry_time
        )
        expiry_effectiveness = (
            (items_before_expiry / len(window_accesses)) if window_accesses else 0.0
        )

        return CacheWindowStats(
            window_start=window_start,
            window_end=now,
            total_requests=len(window_accesses),
            cache_hits=hits,
            cache_misses=misses,
            hit_rate=hit_rate,
            avg_hit_latency_ms=avg_hit_latency,
            avg_miss_latency_ms=avg_miss_latency,
            items_tracked=len(self.item_stats),
            total_cache_size_bytes=int(total_size),
            expiry_effectiveness=expiry_effectiveness,
        )

    def _calculate_optimal_ttl_for_type(self, cache_type: str) -> int:
        """Calculate optimal TTL for a cache type based on access patterns.

        Strategy:
        - Frequent recent accesses → longer TTL
        - Infrequent recent accesses → shorter TTL
        - Items rarely accessed before expiry → reduce TTL

        Args:
            cache_type: Type of cache to analyse

        Returns:
            Recommended TTL in seconds
        """
        # Get items of this type
        type_items = [item for item in self.item_stats.values() if item.cache_type == cache_type]

        if not type_items:
            return self.current_ttl_sec

        # Calculate average item age and hit rate
        now = time.time()
        ages = [(now - item.creation_time) for item in type_items]
        hit_rates = [
            item.hit_count / item.access_count if item.access_count > 0 else 0.0
            for item in type_items
        ]

        avg_age = sum(ages) / len(ages) if ages else 0.0
        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0

        # Base recommendation on recent access activity
        recent_threshold = 300  # 5 minutes
        recent_items = [
            item for item in type_items if now - item.last_access_time < recent_threshold
        ]
        recent_hit_rate = (
            sum(item.hit_count for item in recent_items)
            / sum(item.access_count for item in recent_items)
            if recent_items
            else 0.0
        )

        # Tune TTL based on patterns
        if recent_hit_rate > 0.8:
            # High recent hit rate: increase TTL
            recommended_ttl = int(self.current_ttl_sec * 1.5)
        elif recent_hit_rate > self.hit_rate_threshold:
            # Good hit rate: keep TTL stable
            recommended_ttl = self.current_ttl_sec
        elif recent_hit_rate > 0.5:
            # Moderate hit rate: slightly reduce TTL
            recommended_ttl = int(self.current_ttl_sec * 0.75)
        else:
            # Poor hit rate: significantly reduce TTL
            recommended_ttl = int(self.current_ttl_sec * 0.5)

        # Constrain to min/max
        return max(self.min_ttl_sec, min(self.max_ttl_sec, recommended_ttl))

    def _recommend_eviction_policy(self, cache_type: str) -> str:
        """Recommend eviction policy (LRU vs LFU) for cache type.

        Strategy:
        - If items accessed uniformly: use LRU (simpler)
        - If access frequency varies widely: use LFU (better for hot items)
        - If mixed patterns: hybrid LRU+LFU

        Args:
            cache_type: Type of cache to analyse

        Returns:
            Recommended policy: "LRU", "LFU", or "HYBRID"
        """
        type_items = [item for item in self.item_stats.values() if item.cache_type == cache_type]

        if len(type_items) < 10:
            return "LRU"  # Not enough data

        # Calculate frequency distribution
        frequencies = [item.access_count for item in type_items]
        if not frequencies:
            return "LRU"

        avg_freq = sum(frequencies) / len(frequencies)
        max_freq = max(frequencies)
        min_freq = min(frequencies)

        # Coefficient of variation (measure of spread)
        variance = sum((f - avg_freq) ** 2 for f in frequencies) / len(frequencies)
        std_dev = variance**0.5
        cv = std_dev / avg_freq if avg_freq > 0 else 0.0

        # Decision based on frequency distribution
        if cv > 0.8:  # High variation in access patterns
            return "LFU"
        elif cv > 0.5:  # Moderate variation
            return "HYBRID"
        else:
            return "LRU"  # Uniform access patterns

    def analyse_and_recommend(self, cache_type: Optional[str] = None) -> Dict[str, Any]:
        """Analyse cache performance and recommend optimisations.

        Args:
            cache_type: Specific cache type to analyse (None = all types)

        Returns:
            Dictionary with analysis and recommendations
        """
        with self.lock:
            stats = self._get_window_stats(cache_type)

            if stats.total_requests < 10:
                return {
                    "status": "insufficient_data",
                    "total_requests": stats.total_requests,
                }

            recommendations = {
                "status": "analysed",
                "timestamp": datetime.now().isoformat(),
                "window_seconds": self.sample_window_sec,
                "total_requests": stats.total_requests,
                "hit_rate": stats.hit_rate,
                "avg_hit_latency_ms": stats.avg_hit_latency_ms,
                "avg_miss_latency_ms": stats.avg_miss_latency_ms,
                "latency_improvement_ratio": (
                    stats.avg_miss_latency_ms / stats.avg_hit_latency_ms
                    if stats.avg_hit_latency_ms > 0
                    else 1.0
                ),
                "cache_size_mb": stats.total_cache_size_bytes / (1024 * 1024),
                "items_tracked": stats.items_tracked,
                "expiry_effectiveness": stats.expiry_effectiveness,
            }

            # Per-type analysis
            if cache_type is None:
                for ctype in self.cache_types:
                    type_stats = self._get_window_stats(ctype)
                    if type_stats.total_requests > 0:
                        recommended_ttl = self._calculate_optimal_ttl_for_type(ctype)
                        recommended_policy = self._recommend_eviction_policy(ctype)

                        self.type_stats[ctype]["ttl_recommendation"] = recommended_ttl
                        self.type_stats[ctype]["eviction_policy"] = recommended_policy
                        self.type_stats[ctype]["hit_rate"] = type_stats.hit_rate

                        recommendations[ctype] = {
                            "hit_rate": type_stats.hit_rate,
                            "recommended_ttl_seconds": recommended_ttl,
                            "recommended_ttl_hours": recommended_ttl / 3600,
                            "eviction_policy": recommended_policy,
                            "action": (
                                "increase_ttl"
                                if recommended_ttl > self.current_ttl_sec
                                else (
                                    "decrease_ttl"
                                    if recommended_ttl < self.current_ttl_sec
                                    else "keep_ttl"
                                )
                            ),
                        }
            else:
                recommended_ttl = self._calculate_optimal_ttl_for_type(cache_type)
                recommended_policy = self._recommend_eviction_policy(cache_type)
                recommendations["recommended_ttl_seconds"] = recommended_ttl
                recommendations["recommended_ttl_hours"] = recommended_ttl / 3600
                recommendations["eviction_policy"] = recommended_policy

            # Memory warning
            if stats.total_cache_size_bytes > self.memory_threshold_mb * 1024 * 1024:
                recommendations["memory_warning"] = (
                    f"Cache size ({stats.total_cache_size_bytes / (1024*1024):.1f}MB) "
                    f"exceeds threshold ({self.memory_threshold_mb}MB). "
                    f"Consider reducing TTL or cache size."
                )

            return recommendations

    def get_stats(self) -> Dict[str, Any]:
        """Get current cache statistics.

        Returns:
            Dictionary with cache performance metrics
        """
        with self.lock:
            overall_stats = self._get_window_stats()

            stats_dict = {
                "total_accesses_tracked": len(self.accesses),
                "unique_items": len(self.item_stats),
                "current_ttl_seconds": self.current_ttl_sec,
                "window_hit_rate": overall_stats.hit_rate,
                "window_requests": overall_stats.total_requests,
                "cache_size_mb": overall_stats.total_cache_size_bytes / (1024 * 1024),
                "latency_improvement_ms": (
                    overall_stats.avg_miss_latency_ms - overall_stats.avg_hit_latency_ms
                ),
            }

            # Add per-type stats
            for cache_type in self.cache_types:
                type_stats = self._get_window_stats(cache_type)
                if type_stats.total_requests > 0:
                    stats_dict[f"{cache_type}_hit_rate"] = type_stats.hit_rate
                    stats_dict[f"{cache_type}_recommendations"] = self.type_stats[cache_type]

            return stats_dict


# Global adaptive cache tuner instances
_global_cache_tuners: Dict[str, AdaptiveCacheTuner] = {}


def init_adaptive_cache_tuner(
    cache_type: str = "embedding",
    default_ttl_sec: int = 86400,
) -> AdaptiveCacheTuner:
    """Initialise adaptive cache tuner for a cache type.

    Args:
        cache_type: Type of cache ("embedding" or "llm_result")
        default_ttl_sec: Current TTL in seconds

    Returns:
        Initialised AdaptiveCacheTuner
    """
    global _global_cache_tuners

    if cache_type not in _global_cache_tuners:
        tuner = AdaptiveCacheTuner(default_ttl_sec=default_ttl_sec)
        _global_cache_tuners[cache_type] = tuner
        logger.info(f"Adaptive cache tuner initialised for {cache_type} (TTL: {default_ttl_sec}s)")

    return _global_cache_tuners[cache_type]


def get_adaptive_cache_tuner(cache_type: str = "embedding") -> Optional[AdaptiveCacheTuner]:
    """Get adaptive cache tuner for a cache type.

    Args:
        cache_type: Type of cache ("embedding" or "llm_result")

    Returns:
        AdaptiveCacheTuner or None if not initialised
    """
    return _global_cache_tuners.get(cache_type)
