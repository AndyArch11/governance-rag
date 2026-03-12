"""
Cache layer for academic references.

Implements SQLite-based caching to avoid redundant API calls across
multiple document ingestions.
"""

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class ReferenceStatus:
    """Status constants for references."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    CACHED = "cached"


class Reference:
    """Resolved academic reference."""

    def __init__(
        self,
        ref_id: str,
        raw_citation: str = "",
        doi: Optional[str] = None,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        year: Optional[int] = None,
        abstract: Optional[str] = None,
        venue: Optional[str] = None,
        venue_type: Optional[str] = None,
        volume: Optional[str] = None,
        issue: Optional[str] = None,
        pages: Optional[str] = None,
        reference_type: str = "online",
        resolved: bool = False,
        status: str = "unresolved",
        quality_score: float = 0.0,
        metadata_provider: str = "",
        oa_available: bool = False,
        oa_url: Optional[str] = None,
        link_status: str = "available",
        citation_count: Optional[int] = None,
        doc_ids: Optional[List[str]] = None,
        resolved_at: Optional[datetime] = None,
        cached: bool = False,
    ):
        self.ref_id = ref_id
        self.raw_citation = raw_citation
        self.doi = doi
        self.title = title
        self.authors = authors or []
        self.year = year
        self.abstract = abstract
        self.venue = venue
        self.venue_type = venue_type
        self.volume = volume
        self.issue = issue
        self.pages = pages
        self.reference_type = reference_type
        self.resolved = resolved
        self.status = status
        self.quality_score = quality_score
        self.metadata_provider = metadata_provider
        self.oa_available = oa_available
        self.oa_url = oa_url
        self.link_status = link_status
        self.citation_count = citation_count
        self.doc_ids = doc_ids or []
        self.resolved_at = resolved_at or (datetime.now(timezone.utc) if resolved else None)
        self.cached = cached


class ReferenceCache:
    """SQLite-based cache for resolved references."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialise cache database.

        Args:
            db_path: Path to SQLite database (defaults to rag_data/academic_references.db)
        """
        if db_path is None:
            # Default to rag_data directory relative to project root
            repo_root = Path(__file__).parent.parent.parent.parent
            db_path = str(repo_root / "rag_data" / "academic_references.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # References table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cached_references (
                    ref_id TEXT PRIMARY KEY,
                    cache_key TEXT UNIQUE NOT NULL,
                    doi TEXT,
                    title TEXT,
                    authors TEXT,
                    year INTEGER,
                    abstract TEXT,
                    venue TEXT,
                    venue_type TEXT,
                    volume TEXT,
                    issue TEXT,
                    pages TEXT,
                    reference_type TEXT,
                    resolved INTEGER,
                    status TEXT,
                    quality_score REAL,
                    metadata_provider TEXT,
                    oa_available INTEGER,
                    oa_url TEXT,
                    link_status TEXT DEFAULT 'available',
                    citation_count INTEGER,
                    doc_ids TEXT,
                    raw_citation TEXT,
                    resolved_at TIMESTAMP,
                    cached_at TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)

            # Add missing columns to existing tables (backward compatibility migration)
            cursor.execute("PRAGMA table_info(cached_references)")
            existing_columns = {row[1] for row in cursor.fetchall()}

            if "link_status" not in existing_columns:
                try:
                    cursor.execute(
                        "ALTER TABLE cached_references ADD COLUMN link_status TEXT DEFAULT 'available'"
                    )
                    logging.debug("Added link_status column to cached_references table")
                except sqlite3.OperationalError:
                    pass  # Column already exists

            if "citation_count" not in existing_columns:
                try:
                    cursor.execute(
                        "ALTER TABLE cached_references ADD COLUMN citation_count INTEGER"
                    )
                    logging.debug("Added citation_count column to cached_references table")
                except sqlite3.OperationalError:
                    pass  # Column already exists

            # Document citations table (tracks which docs cite which refs)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_citations (
                    id INTEGER PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    ref_id TEXT NOT NULL,
                    raw_citation TEXT,
                    FOREIGN KEY (ref_id) REFERENCES cached_references(ref_id),
                    UNIQUE(doc_id, ref_id)
                )
            """)

            # Cache statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_stats (
                    id INTEGER PRIMARY KEY,
                    date TEXT,
                    cache_hits INTEGER,
                    cache_misses INTEGER,
                    total_resolved INTEGER,
                    total_unresolved INTEGER
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doi ON cached_references(doi)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_title_year ON cached_references(title, year)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_provider ON cached_references(metadata_provider)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON document_citations(doc_id)")

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def compute_cache_key(
        self,
        doi: Optional[str] = None,
        title: Optional[str] = None,
        year: Optional[int] = None,
        authors: Optional[List[str]] = None,
    ) -> str:
        """
        Generate deterministic cache key.

        Priority: DOI > (title, year, first_author_hash)
        """
        if doi:
            doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
            return f"doi_{doi_clean}"

        h = hashlib.sha256()
        if title:
            h.update(title.lower().encode())
        if year:
            h.update(str(year).encode())
        if authors and len(authors) > 0:
            h.update(authors[0].lower().encode())

        return f"cite_{h.hexdigest()}"

    def get(self, cache_key: str) -> Optional[Reference]:
        """
        Retrieve reference from cache.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM cached_references 
                WHERE cache_key = ? 
                AND (expires_at IS NULL OR expires_at > datetime('now'))
            """,
                (cache_key,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            ref = self._row_to_reference(row)
            logging.debug(f"Cache hit for {cache_key}")
            return ref

    def put(self, cache_key: str, reference: Reference, expires_in_days: int = 365):
        """
        Store reference in cache.
        """
        expires_at = (datetime.now(timezone.utc) + timedelta(days=expires_in_days)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO cached_references (
                    ref_id, cache_key, doi, title, authors, year, abstract,
                    venue, venue_type, volume, issue, pages, reference_type,
                    resolved, status, quality_score, metadata_provider,
                    oa_available, oa_url, link_status, citation_count, doc_ids, raw_citation,
                    resolved_at, cached_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    reference.ref_id,
                    cache_key,
                    reference.doi,
                    reference.title,
                    json.dumps(reference.authors),
                    reference.year,
                    reference.abstract,
                    reference.venue,
                    reference.venue_type,
                    reference.volume,
                    reference.issue,
                    reference.pages,
                    reference.reference_type,
                    int(reference.resolved),
                    reference.status,
                    reference.quality_score,
                    reference.metadata_provider,
                    int(reference.oa_available),
                    reference.oa_url,
                    reference.link_status,
                    reference.citation_count,
                    json.dumps(reference.doc_ids),
                    reference.raw_citation,
                    reference.resolved_at.isoformat() if reference.resolved_at else None,
                    datetime.now(timezone.utc).isoformat(),
                    expires_at,
                ),
            )

            conn.commit()

    def add_citation(self, doc_id: str, ref_id: str, raw_citation: str):
        """
        Record that a document cites a reference.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR IGNORE INTO document_citations (doc_id, ref_id, raw_citation)
                VALUES (?, ?, ?)
            """,
                (doc_id, ref_id, raw_citation),
            )

            conn.commit()

    def _row_to_reference(self, row: sqlite3.Row) -> Reference:
        """
        Reconstruct Reference object from database row.
        """

        # Handle backward compatibility with existing caches missing link_status/citation_count
        def get_row_value(row, key, default=None):
            try:
                return row[key]
            except IndexError:
                return default

        return Reference(
            ref_id=row["ref_id"],
            doi=row["doi"],
            title=row["title"],
            authors=json.loads(row["authors"] or "[]"),
            year=row["year"],
            abstract=row["abstract"],
            venue=row["venue"],
            venue_type=row["venue_type"],
            volume=row["volume"],
            issue=row["issue"],
            pages=row["pages"],
            reference_type=row["reference_type"],
            resolved=bool(row["resolved"]),
            status=row["status"],
            quality_score=row["quality_score"],
            metadata_provider=row["metadata_provider"],
            oa_available=bool(row["oa_available"]),
            oa_url=row["oa_url"],
            link_status=get_row_value(row, "link_status") or "available",
            citation_count=get_row_value(row, "citation_count"),
            doc_ids=json.loads(row["doc_ids"] or "[]"),
            raw_citation=row["raw_citation"],
            resolved_at=datetime.fromisoformat(row["resolved_at"]) if row["resolved_at"] else None,
            cached=True,
        )
