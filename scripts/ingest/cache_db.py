"""SQLite-based cache system for graph, embeddings, and LLM results.

Replaces JSON file-based caching with SQLite for better performance,
concurrent access, and structured queries. Provides CRUD operations for:

- Graph Cache: Consistency graph instances with settings mapping
- Embedding Cache: Vector embeddings indexed by text hash
- LLM Cache: LLM outputs indexed by content hash
- Document Cache: Document metadata and chunk information
- BM25 Index Cache: Inverted index for keyword search (term frequencies, IDF, doc lengths)

Architecture:
    - Single database: rag_data/cache.db
    - Tables: graph_cache, embedding_cache, llm_cache, document_cache, bm25_index
    - Auto-initialised on first use
    - Thread-safe with connection pooling
    - Graceful fallback to JSON if SQLite unavailable

Usage:
    from scripts.ingest.cache_db import CacheDB

    # Initialise cache
    cache = CacheDB(rag_data_path=Path("rag_data"))

    # Store/retrieve embeddings
    cache.put_embedding("text_content", [0.1, 0.2, 0.3])
    embedding = cache.get_embedding("text_content")

    # Store/retrieve graph
    cache.put_graph("graph_hash", graph_data, settings)
    graph = cache.get_graph("graph_hash")

    # Store/retrieve LLM results
    cache.put_llm_result("prompt_hash", "response_text")
    response = cache.get_llm_result("prompt_hash")
"""

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class CacheDB:
    """SQLite-based cache for embeddings, LLM results, graphs, and documents."""

    def __init__(
        self,
        rag_data_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        enable_cache: bool = True,
    ):
        """Initialise the SQLite cache database.

        Args:
            rag_data_path: Path to rag_data directory; defaults to repo rag_data/
            db_path: Direct path to cache.db file; overrides rag_data_path if provided
            enable_cache: Whether caching is enabled; if False, all operations are no-ops
        """
        self.enabled = enable_cache
        if not enable_cache:
            return

        # Determine database path
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            if rag_data_path is None:
                repo_root = Path(__file__).parent.parent.parent
                rag_data_path = repo_root / "rag_data"
            self.db_path = Path(rag_data_path) / "cache.db"

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-safe connection pool
        self._local = threading.local()
        self._lock = threading.RLock()
        self._connections: set[sqlite3.Connection] = set()

        # Initialise database schema
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not self.enabled:
            return None

        if not hasattr(self._local, "connection") or self._local.connection is None:
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30,
                isolation_level=None,
            )
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                pass
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")

            with self._lock:
                self._connections.add(conn)

            self._local.connection = conn

        return self._local.connection

    @property
    def conn(self) -> Optional[sqlite3.Connection]:
        """Expose the database connection for direct SQL queries."""
        return self._get_connection()

    @contextmanager
    def _get_cursor(self):
        """Context manager for database cursor."""
        if not self.enabled:
            yield None
            return

        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _init_db(self):
        """Initialise database schema with all required tables."""
        if not self.enabled:
            return

        with self._get_cursor() as cursor:
            # Embedding cache: text hash -> vector embedding
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    text_hash TEXT PRIMARY KEY,
                    embedding TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hits INTEGER DEFAULT 0
                )
            """)

            # LLM cache: content hash -> LLM response
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    content_hash TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    model TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hits INTEGER DEFAULT 0
                )
            """)

            # Graph cache: graph hash -> graph data with settings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_cache (
                    graph_hash TEXT PRIMARY KEY,
                    graph_data TEXT NOT NULL,
                    settings TEXT NOT NULL,
                    instance_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hits INTEGER DEFAULT 0
                )
            """)

            # Settings mapping: settings hash -> instance mapping
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_settings_map (
                    settings_hash TEXT PRIMARY KEY,
                    graph_hash TEXT NOT NULL,
                    settings TEXT NOT NULL,
                    instance_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (graph_hash) REFERENCES graph_cache(graph_hash)
                )
            """)

            # Document cache: doc hash -> document metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_cache (
                    doc_hash TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    chunks_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # BM25 index cache: term statistics and document frequencies
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bm25_index (
                    term TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    term_frequency INTEGER NOT NULL,
                    doc_length INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (term, doc_id)
                )
            """)

            # BM25 corpus statistics (global IDF values)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bm25_corpus_stats (
                    term TEXT PRIMARY KEY,
                    document_frequency INTEGER NOT NULL,
                    idf REAL NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # BM25 document metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bm25_doc_metadata (
                    doc_id TEXT PRIMARY KEY,
                    doc_length INTEGER NOT NULL,
                    original_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Query analytics: Track query performance and retrieval quality
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    k_results INTEGER,
                    retrieval_time_ms REAL,
                    generation_time_ms REAL,
                    total_time_ms REAL,
                    num_chunks_retrieved INTEGER,
                    avg_similarity_score REAL,
                    max_similarity_score REAL,
                    cache_hit BOOLEAN DEFAULT FALSE,
                    is_code_query BOOLEAN DEFAULT FALSE,
                    model_name TEXT,
                    temperature REAL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Cache access patterns: Track for smart prefetching
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_access_patterns (
                    cache_type TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    last_access TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    avg_access_interval_seconds REAL,
                    prefetch_score REAL DEFAULT 0.0,
                    PRIMARY KEY (cache_type, cache_key)
                )
            """)

            # Word frequency: Global word frequency across all ingested documents (for word clouds)
            # TODO: Consider sharding by first letter and limiting to top N words for performance (cleanup after ingestion?)
            # TODO: Add doc_id list for each word to track which documents contain it (for more advanced analytics)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS word_frequency (
                    word TEXT PRIMARY KEY,
                    frequency INTEGER NOT NULL DEFAULT 1,
                    doc_count INTEGER NOT NULL DEFAULT 1,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for faster lookups
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_embedding_hits ON embedding_cache(hits DESC)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_llm_hits ON llm_cache(hits DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_hits ON graph_cache(hits DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON document_cache(document_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bm25_term ON bm25_index(term)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bm25_doc ON bm25_index(doc_id)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_query_hash ON query_analytics(query_hash)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_query_created ON query_analytics(created_at DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_access ON cache_access_patterns(cache_type, last_access DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_word_frequency ON word_frequency(frequency DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_word_doc_count ON word_frequency(doc_count DESC)"
            )

    @staticmethod
    def _compute_text_hash(text: str) -> str:
        """Compute SHA256 hash of text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_settings_hash(settings: Dict[str, Any]) -> str:
        """Compute deterministic hash from settings dict."""
        normalised = {
            "max_neighbours": settings.get("max_neighbours", 20),
            "sim_threshold": round(settings.get("sim_threshold", 0.4), 4),
            "where": (
                json.dumps(settings.get("where"), sort_keys=True) if settings.get("where") else None
            ),
            "include_documents": settings.get("include_documents", True),
        }
        settings_str = json.dumps(normalised, sort_keys=True)
        return hashlib.sha256(settings_str.encode("utf-8")).hexdigest()[:16]

    # ============================================================================
    # EMBEDDING CACHE OPERATIONS
    # ============================================================================

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Retrieve cached embedding for text.

        Args:
            text: Input text to retrieve embedding for

        Returns:
            List of floats (embedding vector), or None if not cached
        """
        if not self.enabled:
            return None

        text_hash = self._compute_text_hash(text)

        with self._get_cursor() as cursor:
            cursor.execute(
                "SELECT embedding FROM embedding_cache WHERE text_hash = ?", (text_hash,)
            )
            row = cursor.fetchone()

            if row:
                # Update hit count
                cursor.execute(
                    "UPDATE embedding_cache SET hits = hits + 1 WHERE text_hash = ?", (text_hash,)
                )
                # Parse JSON embedding
                return json.loads(row[0])

        return None

    def put_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache embedding for text.

        Args:
            text: Input text
            embedding: Vector embedding (list of floats)
        """
        if not self.enabled:
            return

        text_hash = self._compute_text_hash(text)
        embedding_json = json.dumps(embedding)

        with self._get_cursor() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO embedding_cache (text_hash, embedding)
                VALUES (?, ?)
            """,
                (text_hash, embedding_json),
            )

    def embedding_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        if not self.enabled:
            return {"enabled": False}

        with self._get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count, SUM(hits) as total_hits FROM embedding_cache")
            row = cursor.fetchone()
            return {
                "entries": row["count"] or 0,
                "total_hits": row["total_hits"] or 0,
                "db_path": str(self.db_path),
            }

    # ============================================================================
    # LLM CACHE OPERATIONS
    # ============================================================================

    def get_llm_result(self, prompt: str, model: Optional[str] = None) -> Optional[str]:
        """Retrieve cached LLM response.

        Args:
            prompt: Input prompt
            model: Optional model name for filtering

        Returns:
            LLM response text, or None if not cached
        """
        if not self.enabled:
            return None

        content_hash = self._compute_text_hash(prompt)

        with self._get_cursor() as cursor:
            query = "SELECT response FROM llm_cache WHERE content_hash = ?"
            params = [content_hash]

            if model:
                query += " AND model = ?"
                params.append(model)

            cursor.execute(query, params)
            row = cursor.fetchone()

            if row:
                # Update hit count
                cursor.execute(
                    "UPDATE llm_cache SET hits = hits + 1 WHERE content_hash = ?", (content_hash,)
                )
                return row[0]

        return None

    def put_llm_result(self, prompt: str, response: str, model: Optional[str] = None) -> None:
        """Cache LLM response.

        Args:
            prompt: Input prompt
            response: LLM response text
            model: Optional model name
        """
        if not self.enabled:
            return

        content_hash = self._compute_text_hash(prompt)

        with self._get_cursor() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO llm_cache (content_hash, prompt, response, model)
                VALUES (?, ?, ?, ?)
            """,
                (content_hash, prompt, response, model),
            )

    def llm_stats(self) -> Dict[str, Any]:
        """Get LLM cache statistics."""
        if not self.enabled:
            return {"enabled": False}

        with self._get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count, SUM(hits) as total_hits FROM llm_cache")
            row = cursor.fetchone()
            return {
                "entries": row["count"] or 0,
                "total_hits": row["total_hits"] or 0,
                "db_path": str(self.db_path),
            }

    # ============================================================================
    # GRAPH CACHE OPERATIONS
    # ============================================================================

    def get_cached_graph(self, settings: Dict[str, Any]) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Retrieve cached graph for settings.

        Args:
            settings: Graph build settings (max_neighbours, sim_threshold, etc.)

        Returns:
            Tuple of (instance_id, graph_data), or None if not cached
        """
        if not self.enabled:
            return None

        settings_hash = self._compute_settings_hash(settings)

        with self._get_cursor() as cursor:
            cursor.execute(
                """
                SELECT gsm.instance_id, gc.graph_data
                FROM graph_settings_map gsm
                JOIN graph_cache gc ON gsm.graph_hash = gc.graph_hash
                WHERE gsm.settings_hash = ?
            """,
                (settings_hash,),
            )

            row = cursor.fetchone()
            if row:
                instance_id = row[0]
                graph_data = json.loads(row[1])

                # Update hit count and last_used
                cursor.execute(
                    """
                    UPDATE graph_cache
                    SET hits = hits + 1, last_used = CURRENT_TIMESTAMP
                    WHERE instance_id = ?
                """,
                    (instance_id,),
                )

                return instance_id, graph_data

        return None

    def put_graph(self, graph_data: Dict[str, Any], settings: Dict[str, Any]) -> int:
        """Cache graph with settings mapping.

        Args:
            graph_data: Graph data structure (nodes, edges, etc.)
            settings: Graph build settings

        Returns:
            Instance ID assigned to this graph
        """
        if not self.enabled:
            return -1

        settings_hash = self._compute_settings_hash(settings)
        graph_json = json.dumps(graph_data)
        settings_json = json.dumps(settings)

        with self._get_cursor() as cursor:
            # Get next instance ID
            cursor.execute("SELECT MAX(instance_id) as max_id FROM graph_cache")
            row = cursor.fetchone()
            instance_id = (row["max_id"] or 0) + 1

            # Compute graph hash
            graph_hash = hashlib.sha256(graph_json.encode("utf-8")).hexdigest()[:16]

            # Store graph
            cursor.execute(
                """
                INSERT OR REPLACE INTO graph_cache
                (graph_hash, graph_data, settings, instance_id)
                VALUES (?, ?, ?, ?)
            """,
                (graph_hash, graph_json, settings_json, instance_id),
            )

            # Map settings to graph
            cursor.execute(
                """
                INSERT OR REPLACE INTO graph_settings_map
                (settings_hash, graph_hash, settings, instance_id)
                VALUES (?, ?, ?, ?)
            """,
                (settings_hash, graph_hash, settings_json, instance_id),
            )

            return instance_id

    def purge_all_graphs(self) -> None:
        """Remove all cached graphs."""
        if not self.enabled:
            return

        with self._get_cursor() as cursor:
            cursor.execute("DELETE FROM graph_settings_map")
            cursor.execute("DELETE FROM graph_cache")

    def graph_stats(self) -> Dict[str, Any]:
        """Get graph cache statistics."""
        if not self.enabled:
            return {"enabled": False}

        with self._get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count, SUM(hits) as total_hits FROM graph_cache")
            row = cursor.fetchone()
            return {
                "entries": row["count"] or 0,
                "total_hits": row["total_hits"] or 0,
                "db_path": str(self.db_path),
            }

    # ============================================================================
    # DOCUMENT CACHE OPERATIONS
    # ============================================================================

    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached document metadata.

        Args:
            doc_id: Document ID

        Returns:
            Document metadata dict, or None if not cached
        """
        if not self.enabled:
            return None

        with self._get_cursor() as cursor:
            cursor.execute("SELECT metadata FROM document_cache WHERE document_id = ?", (doc_id,))
            row = cursor.fetchone()

            if row:
                return json.loads(row[0])

        return None

    def put_document_metadata(
        self, doc_id: str, metadata: Dict[str, Any], chunks_count: int = 0
    ) -> None:
        """Cache document metadata.

        Args:
            doc_id: Document ID
            metadata: Document metadata
            chunks_count: Number of chunks in document
        """
        if not self.enabled:
            return

        doc_hash = self._compute_text_hash(doc_id)
        metadata_json = json.dumps(metadata)

        with self._get_cursor() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO document_cache
                (doc_hash, document_id, metadata, chunks_count, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (doc_hash, doc_id, metadata_json, chunks_count),
            )

    def document_stats(self) -> Dict[str, Any]:
        """Get document cache statistics."""
        if not self.enabled:
            return {"enabled": False}

        with self._get_cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) as count, SUM(chunks_count) as total_chunks
                FROM document_cache
            """)
            row = cursor.fetchone()
            return {
                "documents": row["count"] or 0,
                "total_chunks": row["total_chunks"] or 0,
                "db_path": str(self.db_path),
            }

    # ============================================================================
    # STATS & CLEANUP
    # ============================================================================

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all cache tables."""
        if not self.enabled:
            return {"enabled": False}

        return {
            "embeddings": self.embedding_stats(),
            "llm": self.llm_stats(),
            "graphs": self.graph_stats(),
            "documents": self.document_stats(),
            "query_analytics": self.query_analytics_stats(),
        }

    def clear_embedding_cache(self) -> None:
        """Clear only embedding cache table."""
        if not self.enabled:
            return

        with self._get_cursor() as cursor:
            cursor.execute("DELETE FROM embedding_cache")

    def clear_llm_cache(self) -> None:
        """Clear only LLM cache table."""
        if not self.enabled:
            return

        with self._get_cursor() as cursor:
            cursor.execute("DELETE FROM llm_cache")

    def clear_all(self) -> None:
        """Clear all cache tables."""
        if not self.enabled:
            return

        with self._get_cursor() as cursor:
            cursor.execute("DELETE FROM embedding_cache")
            cursor.execute("DELETE FROM llm_cache")
            cursor.execute("DELETE FROM graph_cache")
            cursor.execute("DELETE FROM graph_settings_map")
            cursor.execute("DELETE FROM document_cache")
            cursor.execute("DELETE FROM bm25_index")
            cursor.execute("DELETE FROM bm25_corpus_stats")
            cursor.execute("DELETE FROM bm25_doc_metadata")
            cursor.execute("DELETE FROM query_analytics")
            cursor.execute("DELETE FROM cache_access_patterns")

    # ============================================================================
    # QUERY ANALYTICS OPERATIONS
    # ============================================================================

    def log_query_analytics(
        self,
        query_text: str,
        k_results: int,
        retrieval_time_ms: float,
        generation_time_ms: float,
        num_chunks_retrieved: int,
        avg_similarity_score: float,
        max_similarity_score: float,
        cache_hit: bool = False,
        is_code_query: bool = False,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log query analytics for performance tracking.

        Args:
            query_text: The user's query
            k_results: Number of results requested
            retrieval_time_ms: Time to retrieve chunks (ms)
            generation_time_ms: Time to generate answer (ms)
            num_chunks_retrieved: Actual chunks retrieved
            avg_similarity_score: Average similarity across chunks
            max_similarity_score: Best similarity score
            cache_hit: Whether result came from cache
            is_code_query: Whether this was code-aware query
            model_name: LLM model used
            temperature: LLM temperature
            metadata: Additional metadata
        """
        if not self.enabled:
            return

        query_hash = self._compute_text_hash(query_text)
        total_time_ms = retrieval_time_ms + generation_time_ms
        metadata_json = json.dumps(metadata) if metadata else None

        with self._get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO query_analytics (
                    query_text, query_hash, k_results, retrieval_time_ms,
                    generation_time_ms, total_time_ms, num_chunks_retrieved,
                    avg_similarity_score, max_similarity_score, cache_hit,
                    is_code_query, model_name, temperature, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    query_text,
                    query_hash,
                    k_results,
                    retrieval_time_ms,
                    generation_time_ms,
                    total_time_ms,
                    num_chunks_retrieved,
                    avg_similarity_score,
                    max_similarity_score,
                    cache_hit,
                    is_code_query,
                    model_name,
                    temperature,
                    metadata_json,
                ),
            )

    def query_analytics_stats(self) -> Dict[str, Any]:
        """Get query analytics statistics.

        Returns:
            Dict with total queries, avg times, cache hit rate, etc.
        """
        if not self.enabled:
            return {}

        with self._get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(retrieval_time_ms) as avg_retrieval_ms,
                    AVG(generation_time_ms) as avg_generation_ms,
                    AVG(total_time_ms) as avg_total_ms,
                    AVG(num_chunks_retrieved) as avg_chunks,
                    AVG(avg_similarity_score) as avg_similarity,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits,
                    SUM(CASE WHEN is_code_query = 1 THEN 1 ELSE 0 END) as code_queries
                FROM query_analytics
            """)
            result = cursor.fetchone()

            if not result or result["total_queries"] == 0:
                return {
                    "total_queries": 0,
                    "avg_retrieval_ms": 0.0,
                    "avg_generation_ms": 0.0,
                    "avg_total_ms": 0.0,
                    "avg_chunks": 0.0,
                    "avg_similarity": 0.0,
                    "cache_hit_rate": 0.0,
                    "code_query_rate": 0.0,
                }

            total = result["total_queries"]
            return {
                "total_queries": total,
                "avg_retrieval_ms": round(result["avg_retrieval_ms"] or 0.0, 2),
                "avg_generation_ms": round(result["avg_generation_ms"] or 0.0, 2),
                "avg_total_ms": round(result["avg_total_ms"] or 0.0, 2),
                "avg_chunks": round(result["avg_chunks"] or 0.0, 1),
                "avg_similarity": round(result["avg_similarity"] or 0.0, 3),
                "cache_hit_rate": round((result["cache_hits"] or 0) / total * 100, 1),
                "code_query_rate": round((result["code_queries"] or 0) / total * 100, 1),
            }

    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent queries with their analytics.

        Args:
            limit: Number of recent queries to return

        Returns:
            List of query analytics records
        """
        if not self.enabled:
            return []

        with self._get_cursor() as cursor:
            cursor.execute(
                """
                SELECT 
                    query_text, k_results, retrieval_time_ms, generation_time_ms,
                    total_time_ms, num_chunks_retrieved, avg_similarity_score,
                    cache_hit, is_code_query, model_name, created_at
                FROM query_analytics
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (limit,),
            )

            results = cursor.fetchall()
            return [dict(row) for row in results]

    # ============================================================================
    # SMART CACHE OPERATIONS (Access Patterns & Prefetching)
    # ============================================================================

    def track_cache_access(self, cache_type: str, cache_key: str) -> None:
        """Track cache access for smart prefetching.

        Args:
            cache_type: Type of cache (embedding, llm, graph)
            cache_key: Cache key accessed
        """
        if not self.enabled:
            return

        with self._get_cursor() as cursor:
            # Check if pattern exists
            cursor.execute(
                """
                SELECT access_count, last_access, avg_access_interval_seconds
                FROM cache_access_patterns
                WHERE cache_type = ? AND cache_key = ?
            """,
                (cache_type, cache_key),
            )

            result = cursor.fetchone()
            current_time = datetime.now(timezone.utc).isoformat()

            if result:
                # Update existing pattern
                access_count = result["access_count"] + 1
                last_access = datetime.fromisoformat(result["last_access"].replace("Z", "+00:00"))
                current_dt = datetime.now(timezone.utc)
                interval_seconds = (current_dt - last_access).total_seconds()

                # Update average interval (exponential moving average)
                old_avg = result["avg_access_interval_seconds"] or interval_seconds
                new_avg = 0.7 * old_avg + 0.3 * interval_seconds

                # Calculate prefetch score (higher = more predictable access)
                prefetch_score = min(100.0, access_count / (new_avg / 3600.0 + 1.0))

                cursor.execute(
                    """
                    UPDATE cache_access_patterns
                    SET access_count = ?, last_access = ?,
                        avg_access_interval_seconds = ?, prefetch_score = ?
                    WHERE cache_type = ? AND cache_key = ?
                """,
                    (access_count, current_time, new_avg, prefetch_score, cache_type, cache_key),
                )
            else:
                # Insert new pattern
                cursor.execute(
                    """
                    INSERT INTO cache_access_patterns
                    (cache_type, cache_key, access_count, last_access)
                    VALUES (?, ?, 1, ?)
                """,
                    (cache_type, cache_key, current_time),
                )

    def get_prefetch_candidates(self, cache_type: str, limit: int = 10) -> List[str]:
        """Get cache keys that should be prefetched.

        Args:
            cache_type: Type of cache (embedding, llm, graph)
            limit: Max candidates to return

        Returns:
            List of cache keys with highest prefetch scores
        """
        if not self.enabled:
            return []

        with self._get_cursor() as cursor:
            cursor.execute(
                """
                SELECT cache_key, prefetch_score
                FROM cache_access_patterns
                WHERE cache_type = ? AND prefetch_score > 5.0
                ORDER BY prefetch_score DESC
                LIMIT ?
            """,
                (cache_type, limit),
            )

            results = cursor.fetchall()
            return [row["cache_key"] for row in results]

    def evict_lru_cache_entries(self, cache_type: str, keep_count: int = 1000) -> int:
        """Evict least recently used cache entries using LRU policy.

        Args:
            cache_type: Type of cache (embedding, llm, graph)
            keep_count: Number of entries to keep

        Returns:
            Number of entries evicted
        """
        if not self.enabled:
            return 0

        table_map = {"embedding": "embedding_cache", "llm": "llm_cache", "graph": "graph_cache"}

        table = table_map.get(cache_type)
        if not table:
            return 0

        with self._get_cursor() as cursor:
            # Count total entries
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            total = cursor.fetchone()["count"]

            if total <= keep_count:
                return 0

            # Get LRU entries to delete (based on created_at since we don't track last_used everywhere)
            to_delete = total - keep_count

            if cache_type == "embedding":
                cursor.execute(
                    f"""
                    DELETE FROM {table}
                    WHERE text_hash IN (
                        SELECT text_hash FROM {table}
                        ORDER BY created_at ASC
                        LIMIT ?
                    )
                """,
                    (to_delete,),
                )
            elif cache_type == "llm":
                cursor.execute(
                    f"""
                    DELETE FROM {table}
                    WHERE content_hash IN (
                        SELECT content_hash FROM {table}
                        ORDER BY created_at ASC
                        LIMIT ?
                    )
                """,
                    (to_delete,),
                )
            elif cache_type == "graph":
                cursor.execute(
                    f"""
                    DELETE FROM {table}
                    WHERE graph_hash IN (
                        SELECT graph_hash FROM {table}
                        ORDER BY last_used ASC
                        LIMIT ?
                    )
                """,
                    (to_delete,),
                )

            return to_delete

    # ============================================================================
    # BM25 INDEX OPERATIONS
    # ============================================================================

    def put_bm25_document(
        self,
        doc_id: str,
        term_frequencies: Dict[str, int],
        doc_length: int,
        original_text: Optional[str] = None,
    ) -> None:
        """Store BM25 term frequencies for a document.

        Args:
            doc_id: Document identifier
            term_frequencies: Dict of term -> frequency in document
            doc_length: Total token count in document
            original_text: Optional original text for reference
        """
        if not self.enabled:
            return

        with self._get_cursor() as cursor:
            # Store document metadata
            cursor.execute(
                """INSERT OR REPLACE INTO bm25_doc_metadata 
                   (doc_id, doc_length, original_text) 
                   VALUES (?, ?, ?)""",
                (doc_id, doc_length, original_text),
            )

            # Store term frequencies
            for term, freq in term_frequencies.items():
                cursor.execute(
                    """INSERT OR REPLACE INTO bm25_index 
                       (term, doc_id, term_frequency, doc_length) 
                       VALUES (?, ?, ?, ?)""",
                    (term, doc_id, freq, doc_length),
                )

    def update_bm25_corpus_stats(self, total_docs: int) -> None:
        """Compute and store IDF values for all terms in the corpus.

        Args:
            total_docs: Total number of documents in the corpus
        """
        if not self.enabled or total_docs == 0:
            return

        with self._get_cursor() as cursor:
            # Get document frequency for each term
            cursor.execute("""
                SELECT term, COUNT(DISTINCT doc_id) as df
                FROM bm25_index
                GROUP BY term
            """)

            import math

            # Compute and store IDF for each term
            for term, df in cursor.fetchall():
                # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
                idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

                cursor.execute(
                    """INSERT OR REPLACE INTO bm25_corpus_stats 
                       (term, document_frequency, idf, last_updated) 
                       VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
                    (term, df, idf),
                )

    def get_bm25_term_stats(self, term: str) -> Optional[Tuple[int, float]]:
        """Get document frequency and IDF for a term.

        Args:
            term: Search term

        Returns:
            Tuple of (document_frequency, idf) or None if term not indexed
        """
        if not self.enabled:
            return None

        with self._get_cursor() as cursor:
            cursor.execute(
                "SELECT document_frequency, idf FROM bm25_corpus_stats WHERE term = ?", (term,)
            )
            result = cursor.fetchone()
            return (result["document_frequency"], result["idf"]) if result else None

    def get_bm25_doc_terms(self, doc_id: str) -> Dict[str, int]:
        """Get all term frequencies for a document.

        Args:
            doc_id: Document identifier

        Returns:
            Dict of term -> frequency
        """
        if not self.enabled:
            return {}

        with self._get_cursor() as cursor:
            cursor.execute(
                "SELECT term, term_frequency FROM bm25_index WHERE doc_id = ?", (doc_id,)
            )
            return {row["term"]: row["term_frequency"] for row in cursor.fetchall()}

    def get_bm25_docs_with_term(self, term: str) -> List[Tuple[str, int, int]]:
        """Get all documents containing a term with their frequencies.

        Args:
            term: Search term

        Returns:
            List of (doc_id, term_frequency, doc_length) tuples
        """
        if not self.enabled:
            return []

        with self._get_cursor() as cursor:
            cursor.execute(
                """SELECT doc_id, term_frequency, doc_length 
                   FROM bm25_index 
                   WHERE term = ?""",
                (term,),
            )
            return [
                (row["doc_id"], row["term_frequency"], row["doc_length"])
                for row in cursor.fetchall()
            ]

    def get_bm25_avg_doc_length(self) -> float:
        """Get average document length across corpus.

        Returns:
            Average document length (tokens)
        """
        if not self.enabled:
            return 0.0

        with self._get_cursor() as cursor:
            cursor.execute("SELECT AVG(doc_length) as avg_len FROM bm25_doc_metadata")
            result = cursor.fetchone()
            return result["avg_len"] if result and result["avg_len"] else 0.0

    def get_bm25_corpus_size(self) -> int:
        """Get total number of documents in BM25 index.

        Returns:
            Number of indexed documents
        """
        if not self.enabled:
            return 0

        with self._get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM bm25_doc_metadata")
            result = cursor.fetchone()
            return result["count"] if result else 0

    def delete_bm25_document(self, doc_id: str) -> None:
        """Remove a document from BM25 index.

        Args:
            doc_id: Document identifier to remove
        """
        if not self.enabled:
            return

        with self._get_cursor() as cursor:
            cursor.execute("DELETE FROM bm25_index WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM bm25_doc_metadata WHERE doc_id = ?", (doc_id,))

    def put_word_frequencies(
        self, word_freqs: Dict[str, int], doc_count: Optional[Dict[str, int]] = None
    ) -> None:
        """Store or update word frequencies for word cloud generation.

        Accumulates word frequencies from new documents. Tracks both global frequency
        and the number of unique documents containing each word.

        Args:
            word_freqs: Dictionary of word -> frequency (from current document)
            doc_count: Optional dictionary of word -> unique doc count (from current document);
                      if None, assumes 1 document contains each word
        """
        if not self.enabled or not word_freqs:
            return

        if doc_count is None:
            doc_count = {word: 1 for word in word_freqs}

        with self._get_cursor() as cursor:
            for word, freq in word_freqs.items():
                doc_cnt = doc_count.get(word, 1)
                cursor.execute(
                    """
                    INSERT INTO word_frequency (word, frequency, doc_count, last_updated)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(word) DO UPDATE SET
                        frequency = frequency + excluded.frequency,
                        doc_count = doc_count + excluded.doc_count,
                        last_updated = CURRENT_TIMESTAMP
                """,
                    (word, freq, doc_cnt),
                )

    def get_top_words(self, limit: int = 100, min_frequency: int = 1) -> List[Tuple[str, int, int]]:
        """Get top N words by frequency for word cloud generation.

        Args:
            limit: Maximum number of words to return (default 100)
            min_frequency: Minimum frequency threshold for inclusion (default 1)

        Returns:
            List of (word, frequency, doc_count) tuples ordered by frequency descending
        """
        if not self.enabled:
            return []

        with self._get_cursor() as cursor:
            cursor.execute(
                """
                SELECT word, frequency, doc_count
                FROM word_frequency
                WHERE frequency >= ?
                ORDER BY frequency DESC
                LIMIT ?
            """,
                (min_frequency, limit),
            )
            return [(row["word"], row["frequency"], row["doc_count"]) for row in cursor.fetchall()]

    def get_word_frequency_stats(self) -> Dict[str, int]:
        """Get overall word frequency statistics.

        Returns:
            Dictionary with stats: total_unique_words, total_frequency, avg_frequency
        """
        if not self.enabled:
            return {}

        with self._get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM word_frequency")
            unique_words = cursor.fetchone()["count"] or 0

            cursor.execute("SELECT SUM(frequency) as total FROM word_frequency")
            total_freq = cursor.fetchone()["total"] or 0

            avg_freq = total_freq / unique_words if unique_words > 0 else 0

            return {
                "total_unique_words": unique_words,
                "total_frequency": total_freq,
                "avg_frequency": round(avg_freq, 2),
            }

    def clear_word_frequencies(self) -> None:
        """Clear all word frequency data. Useful for rebuilding statistics."""
        if not self.enabled:
            return

        with self._get_cursor() as cursor:
            cursor.execute("DELETE FROM word_frequency")

    def close(self) -> None:
        """Close database connection."""
        if not self.enabled:
            return

        with self._lock:
            for conn in list(self._connections):
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()

        if hasattr(self._local, "connection"):
            self._local.connection = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed."""
        self.close()
        return False

    def __del__(self):
        """Destructor - ensures connection is closed on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
