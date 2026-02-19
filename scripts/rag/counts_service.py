import sqlite3
from typing import Dict, List, Optional, Tuple

try:
    # Prefer the shared cache client if available
    from scripts.utils.db_factory import get_cache_client  # type: ignore
except Exception:
    get_cache_client = None  # Fallback to direct sqlite connection when testing


class CountsService:
    """SQL-backed counts and listing service for Hybrid RAG.

    Provides corpus-aware counts that bypass top-k retrieval limits:
    - count distinct documents mentioning a term or AND-combination
    - total occurrences of a term across corpus
    - list of documents matching a term with pagination
    - breakdowns by metadata (e.g. source_category)

    Relies on BM25 ingestion-time tables:
      bm25_index(term TEXT, doc_id TEXT, term_frequency INTEGER, doc_length INTEGER)
      bm25_corpus_stats(term TEXT PRIMARY KEY, document_frequency INTEGER, idf REAL)
      bm25_doc_metadata(doc_id TEXT PRIMARY KEY, source_category TEXT, doc_length INTEGER, original_text TEXT)
    """

    def __init__(self, conn: Optional[sqlite3.Connection] = None) -> None:
        self._external_conn = conn is not None
        self._cache_client = None  # Track cache client for cleanup
        try:
            self.conn = conn or self._get_connection()
        except Exception as e:
            self.conn = None
            raise e

    def _bm25_index_columns(self) -> List[str]:
        """Return column names for bm25_index, handling environments with differing schemas.

        Some environments use `term_frequency` while others use `tf`. This helper lets us
        select the appropriate column without relying on try/except around query execution.
        """
        try:
            cur = self.conn.cursor()
            cur.execute("PRAGMA table_info('bm25_index')")
            return [row[1] for row in cur.fetchall()]  # row[1] is column name
        except Exception:
            return []

    def _get_connection(self) -> sqlite3.Connection:
        if get_cache_client:
            # The cache client exposes a sqlite3 connection via .conn
            self._cache_client = get_cache_client(enable_cache=True)
            # Defensive: support both .conn and direct connection return
            conn = getattr(self._cache_client, "conn", None)
            if conn is not None:
                return conn
            # If cache client returns connection directly
            if isinstance(self._cache_client, sqlite3.Connection):
                return self._cache_client
        # Fallback: use rag_data/cache.db path
        from pathlib import Path

        from scripts.rag.rag_config import RAGConfig

        config = RAGConfig()
        cache_path = Path(config.rag_data_path) / "cache.db"
        return sqlite3.connect(str(cache_path))

    def close(self) -> None:
        # Close the cache client first if we created one
        if self._cache_client and hasattr(self._cache_client, "close"):
            try:
                self._cache_client.close()
            except Exception:
                pass
        # Then close our connection if it's not external
        if self.conn and not self._external_conn:
            try:
                self.conn.close()
            except Exception:
                pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed."""
        self.close()
        return False

    # ---- Core queries ----
    def count_docs_for_term(self, term: str, source_category: Optional[str] = None) -> int:
        if not term:
            return 0
        cur = self.conn.cursor()
        if source_category:
            cur.execute(
                """
                SELECT COUNT(DISTINCT i.doc_id)
                FROM bm25_index i
                JOIN bm25_doc_metadata m ON m.doc_id = i.doc_id
                WHERE i.term = ? AND LOWER(COALESCE(m.source_category, '')) = LOWER(?)
                """,
                (term, source_category),
            )
        else:
            cur.execute(
                """
                SELECT COUNT(DISTINCT doc_id)
                FROM bm25_index
                WHERE term = ?
                """,
                (term,),
            )
        row = cur.fetchone()
        return int(row[0] or 0) if row else 0

    def count_occurrences_for_term(self, term: str) -> int:
        if not term:
            return 0
        cur = self.conn.cursor()
        # Detect the appropriate term frequency column (`term_frequency` or `tf`)
        cols = self._bm25_index_columns()
        tf_col = "term_frequency" if "term_frequency" in cols else ("tf" if "tf" in cols else None)

        if tf_col is None:
            # Fallback: if no TF column found, approximate by counting rows for the term
            cur.execute(
                """
                SELECT COUNT(*)
                FROM bm25_index
                WHERE term = ?
                """,
                (term,),
            )
            row = cur.fetchone()
            return int(row[0] or 0) if row else 0

        # Sum occurrences using the detected TF column
        cur.execute(
            f"""
            SELECT COALESCE(SUM({tf_col}), 0)
            FROM bm25_index
            WHERE term = ?
            """,
            (term,),
        )
        row = cur.fetchone()
        return int(row[0] or 0) if row else 0

    def total_documents(self) -> int:
        cur = self.conn.cursor()
        # Count all documents from bm25_doc_metadata
        cur.execute("SELECT COUNT(*) FROM bm25_doc_metadata")
        row = cur.fetchone()
        return int(row[0] or 0) if row else 0

    def list_docs_for_term(
        self,
        term: str,
        limit: int = 50,
        offset: int = 0,
        source_category: Optional[str] = None,
    ) -> List[str]:
        """List document IDs containing term. Note: source_category filtering not available in BM25 tables."""
        if not term:
            return []
        cur = self.conn.cursor()
        # Note: source_category not stored in bm25_doc_metadata, so filtering is ignored
        cur.execute(
            """
            SELECT DISTINCT doc_id
            FROM bm25_index
            WHERE term = ?
            ORDER BY doc_id
            LIMIT ? OFFSET ?
            """,
            (term, limit, offset),
        )
        return [row[0] for row in cur.fetchall()]

    def breakdown_by_category(self, term: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get document count breakdown by `source_category` for a term.

        Groups distinct documents mentioning the term by metadata category and returns
        the top categories by document count. If metadata is not present, returns empty list.
        """
        if not term:
            return []
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                SELECT COALESCE(LOWER(m.source_category), 'unknown') AS category,
                       COUNT(DISTINCT i.doc_id) AS doc_count
                FROM bm25_index i
                JOIN bm25_doc_metadata m ON m.doc_id = i.doc_id
                WHERE i.term = ?
                GROUP BY category
                ORDER BY doc_count DESC, category ASC
                LIMIT ?
                """,
                (term, top_n),
            )
            return [(row[0], int(row[1])) for row in cur.fetchall()]
        except Exception:
            # If metadata table or column is unavailable, return empty breakdown gracefully
            return []

    def count_docs_for_and(self, terms: List[str]) -> int:
        if not terms:
            return 0
        # Use a subquery: group by doc_id and require all distinct terms present
        placeholders = ",".join(["?"] * len(terms))
        sql = f"""
            SELECT COUNT(*) FROM (
                SELECT i.doc_id
                FROM bm25_index i
                WHERE i.term IN ({placeholders})
                GROUP BY i.doc_id
                HAVING COUNT(DISTINCT i.term) = ?
            ) AS docs
        """
        cur = self.conn.cursor()
        cur.execute(sql, (*terms, len(terms)))
        row = cur.fetchone()
        return int(row[0] or 0) if row else 0

    def summarise_term(self, term: str, limit: int = 10) -> Dict:
        """Return a structured summary for a term for prompt enrichment."""
        total_docs = self.count_docs_for_term(term)
        total_occ = self.count_occurrences_for_term(term)
        docs = self.list_docs_for_term(term, limit=limit, offset=0)
        breakdown = self.breakdown_by_category(term, top_n=5)
        return {
            "term": term,
            "total_docs": total_docs,
            "total_occurrences": total_occ,
            "sample_docs": docs,
            "category_breakdown": breakdown,
        }

    # ---- Parent-child aggregation (optional) ----
    def count_parent_docs_for_term(self, term: str, source_category: Optional[str] = None) -> int:
        """Count distinct parent documents (via child->parent join) for a term.

        If bm25_chunk_metadata table exists with parent_id column, aggregates child counts
        to parent documents. Otherwise, falls back to chunk-level counts.

        Args:
            term: Search term
            source_category: Optional category filter

        Returns:
            Count of distinct parent documents containing the term
        """
        if not term:
            return 0

        cur = self.conn.cursor()

        # Check if chunk-level table exists (optional feature)
        try:
            cur.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='bm25_chunk_metadata'"
            )
            has_chunk_table = cur.fetchone()[0] > 0
        except Exception:
            has_chunk_table = False

        if has_chunk_table and source_category:
            # With chunk table and category filter
            cur.execute(
                """
                SELECT COUNT(DISTINCT COALESCE(cm.parent_id, i.doc_id))
                FROM bm25_index i
                LEFT JOIN bm25_chunk_metadata cm ON cm.chunk_id = i.doc_id
                JOIN bm25_doc_metadata m ON m.doc_id = COALESCE(cm.parent_id, i.doc_id)
                WHERE i.term = ? AND LOWER(COALESCE(m.source_category, '')) = LOWER(?)
                """,
                (term, source_category),
            )
        elif has_chunk_table:
            # With chunk table but no category filter
            cur.execute(
                """
                SELECT COUNT(DISTINCT COALESCE(cm.parent_id, i.doc_id))
                FROM bm25_index i
                LEFT JOIN bm25_chunk_metadata cm ON cm.chunk_id = i.doc_id
                WHERE i.term = ?
                """,
                (term,),
            )
        else:
            # Fallback to bm25_index only (same as count_docs_for_term)
            return self.count_docs_for_term(term, source_category)

        row = cur.fetchone()
        return int(row[0] or 0) if row else 0

    def list_parent_docs_for_term(
        self,
        term: str,
        limit: int = 50,
        offset: int = 0,
        source_category: Optional[str] = None,
    ) -> List[str]:
        """List distinct parent documents for a term with pagination.

        Args:
            term: Search term
            limit: Max results to return
            offset: Pagination offset
            source_category: Optional category filter

        Returns:
            List of parent document IDs
        """
        if not term:
            return []

        cur = self.conn.cursor()

        # Check if chunk-level table exists
        try:
            cur.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='bm25_chunk_metadata'"
            )
            has_chunk_table = cur.fetchone()[0] > 0
        except Exception:
            has_chunk_table = False

        if has_chunk_table and source_category:
            cur.execute(
                """
                SELECT DISTINCT COALESCE(cm.parent_id, i.doc_id)
                FROM bm25_index i
                LEFT JOIN bm25_chunk_metadata cm ON cm.chunk_id = i.doc_id
                JOIN bm25_doc_metadata m ON m.doc_id = COALESCE(cm.parent_id, i.doc_id)
                WHERE i.term = ? AND LOWER(COALESCE(m.source_category, '')) = LOWER(?)
                ORDER BY COALESCE(cm.parent_id, i.doc_id)
                LIMIT ? OFFSET ?
                """,
                (term, source_category, limit, offset),
            )
        elif has_chunk_table:
            cur.execute(
                """
                SELECT DISTINCT COALESCE(cm.parent_id, i.doc_id)
                FROM bm25_index i
                LEFT JOIN bm25_chunk_metadata cm ON cm.chunk_id = i.doc_id
                WHERE i.term = ?
                ORDER BY COALESCE(cm.parent_id, i.doc_id)
                LIMIT ? OFFSET ?
                """,
                (term, limit, offset),
            )
        else:
            # Fallback
            return self.list_docs_for_term(term, limit, offset, source_category)

        return [row[0] for row in cur.fetchall()]

    def parent_document_summary(self, term: str, limit: int = 10) -> Dict:
        """Return a parent-document-focused summary for a term.

        Useful when parent chunks are preferred (e.g., richer context).

        Args:
            term: Search term
            limit: Max sample parent docs to list

        Returns:
            Dictionary with parent-level counts and sample docs
        """
        total_parent_docs = self.count_parent_docs_for_term(term)
        parent_docs = self.list_parent_docs_for_term(term, limit=limit, offset=0)
        breakdown = self.breakdown_by_category(term, top_n=5)

        return {
            "term": term,
            "total_parent_docs": total_parent_docs,
            "total_occurrences": self.count_occurrences_for_term(term),
            "sample_parent_docs": parent_docs,
            "category_breakdown": breakdown,
        }
