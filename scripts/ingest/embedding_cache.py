"""Embedding cache for reusing vector embeddings of identical text.

Caches embeddings by text hash in SQLite database to avoid regenerating
embeddings for duplicate chunks across documents or versions. Significantly
reduces embedding API calls and improves throughput.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

from scripts.utils.adaptive_cache_tuning import get_adaptive_cache_tuner
from scripts.utils.db_factory import get_cache_client


class EmbeddingCache:
    """SQLite-backed cache for storing and retrieving vector embeddings.

    Maps text hashes to their vector embeddings, enabling reuse when
    the same chunk text appears across multiple documents or versions.

    Uses CacheDB SQLite backend for persistence and thread safety.
    Provides compatibility with previous JSON-based API.

    Thread Safety:
        CacheDB handles thread-safe access via connection pooling.

    Persistence:
        Automatically persisted in SQLite rag_data/cache.db.

    Attributes:
        enabled: Whether caching is active.
        hits: Total cache hits (successful retrievals).
        misses: Total cache misses (not found).

    Example:
        >>> cache = EmbeddingCache("/path/to/cache.json")
        >>>
        >>> # Try to get cached embedding
        >>> embedding = cache.get("chunk text")
        >>> if embedding is None:
        ...     # Not cached, generate new embedding
        ...     embedding = embed_model.embed_documents(["chunk text"])[0]
        ...     cache.put("chunk text", embedding)
        >>>
        >>> # Get statistics
        >>> stats = cache.stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
    """

    def __init__(self, cache_path: str, enabled: bool = True, auto_save_interval: int = 50):
        """Initialise embedding cache using SQLite backend.

        Args:
            cache_path: Ignored (for API compatibility); uses rag_data/cache.db
            enabled: Whether to enable caching (default True).
            auto_save_interval: Ignored; SQLite auto-commits each write.
        """
        # Extract rag_data_path from cache_path if provided
        # Expected path format: .../rag_data/cache/embedding_cache.json
        rag_data_path = None
        if cache_path and "rag_data" in cache_path:
            path_parts = Path(cache_path).parts
            if "rag_data" in path_parts:
                rag_data_idx = path_parts.index("rag_data")
                rag_data_path = Path(*path_parts[: rag_data_idx + 1])

        # Get SQLite cache backend
        self.db = get_cache_client(rag_data_path=rag_data_path, enable_cache=enabled)

        self.enabled = enabled
        self.hits = 0
        self.misses = 0

        # Initialise adaptive cache tuner
        self.tuner = get_adaptive_cache_tuner("embedding")

    def get(self, text: str) -> Optional[List[float]]:
        """Retrieve cached embedding for text.

        Args:
            text: Text to look up.

        Returns:
            Embedding vector if cached, None if not found.

        Side Effects:
            - Increments hits/misses counter
            - Records access in adaptive cache tuner
        """
        if not self.enabled:
            return None

        start_ms = time.perf_counter() * 1000
        embedding = self.db.get_embedding(text)
        latency_ms = time.perf_counter() * 1000 - start_ms

        hit = embedding is not None
        if hit:
            self.hits += 1
        else:
            self.misses += 1

        # Record in adaptive cache tuner
        if self.tuner:
            self.tuner.record_access(
                cache_type="embedding",
                key=text[:64],  # Truncate key for storage
                hit=hit,
                latency_ms=latency_ms,
                size_bytes=len(embedding) * 4 if embedding else 0,  # 4 bytes per float
            )

        return embedding

    def put(self, text: str, embedding: List[float]) -> None:
        """Store embedding for text in cache.

        Args:
            text: Text that was embedded.
            embedding: Vector embedding to cache.
        """
        if not self.enabled:
            return

        self.db.put_embedding(text, embedding)

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache metrics:
                - total_entries: Number of cached embeddings
                - hits: Successful cache retrievals
                - misses: Cache misses
                - hit_rate: Percentage of successful lookups
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0

        db_stats = self.db.embedding_stats()
        return {
            "total_entries": db_stats.get("entries", 0),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }

    def get_batch(self, texts: List[str]) -> Tuple[List[Optional[List[float]]], List[str]]:
        """Retrieve embeddings for batch of texts.

        Args:
            texts: List of texts to look up.

        Returns:
            Tuple of (embeddings, uncached_texts):
                - embeddings: List of embeddings or None for each text
                - uncached_texts: Texts that were not in cache
        """
        if not self.enabled:
            return [None] * len(texts), texts

        embeddings = []
        uncached = []

        for text in texts:
            emb = self.get(text)
            embeddings.append(emb)
            if emb is None:
                uncached.append(text)

        return embeddings, uncached

    def put_batch(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """Store batch of embeddings in cache.

        Args:
            texts: List of texts that were embedded.
            embeddings: List of corresponding embeddings.
        """
        if not self.enabled or len(texts) != len(embeddings):
            return

        for text, embedding in zip(texts, embeddings):
            self.put(text, embedding)

    def flush(self) -> None:
        """Flush cache to storage (no-op for SQLite; auto-commits)."""
        pass

    def clear(self) -> None:
        """Clear all embeddings from cache."""
        if not self.enabled:
            return
        self.db.clear_embedding_cache()

    def close(self) -> None:
        """Close database connection."""
        if self.db:
            self.db.close()
