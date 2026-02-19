"""Backend factory for vector store clients and cache management.

Provides a single place to select between ChromaDB and the lightweight
SQLite-backed compatibility layer. Defaults to ChromaDB when available,
and falls back to SQLite. Preference can be flipped per call.

Also manages cache database (SQLite) for embeddings, LLM results, graphs,
and document metadata.
"""

import atexit
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Global cache instances keyed by canonical path
_cache_instances: Dict[str, Any] = {}
_cache_lock = threading.Lock()


def get_vector_client(prefer: str = "chroma") -> Tuple[Any, bool]:
    """Return (PersistentClient, using_sqlite) with preference ordering.

    Args:
        prefer: "chroma" (default) or "sqlite" to set the first choice.

    Returns:
        (PersistentClient class, using_sqlite flag)

    Raises:
        ImportError: if neither backend is available.
    """
    prefer = (prefer or "chroma").lower()
    prefer_order = ["sqlite", "chroma"] if prefer == "sqlite" else ["chroma", "sqlite"]

    sqlite_client = None
    chroma_client = None
    errors = []

    # Try SQLite backend
    if "sqlite" in prefer_order:
        try:
            from scripts.ingest.chromadb_sqlite import (  # noqa: WPS433
                PersistentClient as SQLitePersistentClient,
            )

            sqlite_client = SQLitePersistentClient
        except Exception as exc:  # ImportError or other
            errors.append(f"SQLite backend unavailable: {exc}")

    # Try ChromaDB backend
    if "chroma" in prefer_order:
        try:
            from chromadb import (  # type: ignore  # noqa: WPS433
                PersistentClient as ChromaDBPersistentClient,
            )

            chroma_client = ChromaDBPersistentClient
        except Exception as exc:
            errors.append(f"ChromaDB backend unavailable: {exc}")

    # Return in preferred order
    for backend in prefer_order:
        if backend == "chroma" and chroma_client:
            return chroma_client, False
        if backend == "sqlite" and sqlite_client:
            return sqlite_client, True

    raise ImportError("No vector backend available. Details: " + "; ".join(errors))


def get_default_vector_path(rag_data_path: Path, using_sqlite: bool) -> str:
    """Build default path for the chosen backend."""
    return str(Path(rag_data_path) / ("chromadb.db" if using_sqlite else "chromadb"))


def get_cache_client(rag_data_path: Optional[Path] = None, enable_cache: bool = True) -> Any:
    """Return SQLite cache client for embeddings, LLM results, graphs, and documents.

    Uses a singleton pattern per path to avoid creating multiple database connections
    to the same database. Different paths get different instances.
    Instances are automatically closed on program exit.

    Args:
        rag_data_path: Path to rag_data directory; defaults to repo rag_data/
        enable_cache: Whether caching is enabled; if False, returns no-op cache

    Returns:
        CacheDB instance (singleton per path)

    Raises:
        ImportError: if SQLite cache module is not available.
    """
    try:
        from scripts.ingest.cache_db import CacheDB  # noqa: WPS433

        # Determine the canonical path
        if rag_data_path is None:
            repo_root = Path(__file__).parent.parent.parent
            rag_data_path = repo_root / "rag_data"

        # Create canonical path key
        canonical_path = str(Path(rag_data_path).resolve())

        # Return existing instance if available for this path
        with _cache_lock:
            if canonical_path in _cache_instances:
                return _cache_instances[canonical_path]

            # Create new instance for this path
            instance = CacheDB(rag_data_path=rag_data_path, enable_cache=enable_cache)
            _cache_instances[canonical_path] = instance

            # Register cleanup on first instance creation
            if len(_cache_instances) == 1:
                atexit.register(_cleanup_cache_clients)

            return instance
    except Exception as exc:
        raise ImportError(f"Cache backend unavailable: {exc}") from exc


def _cleanup_cache_clients():
    """Clean up all cache instances on program exit."""
    with _cache_lock:
        for instance in _cache_instances.values():
            try:
                instance.close()
            except Exception:
                pass
        _cache_instances.clear()
