"""Unit tests for parent-child aggregation in CountsService.

Tests parent document counting and listing via optional bm25_chunk_metadata table
with parent_id column for chunk-to-parent mappings.
"""

import sqlite3


def _setup_db_with_chunk_metadata():
    """Create in-memory DB with chunk metadata table for parent-child testing."""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    # Core tables
    cur.execute("""
        CREATE TABLE bm25_index (
            doc_id TEXT,
            term TEXT,
            tf INTEGER,
            doc_length INTEGER
        )
        """)
    cur.execute("""
        CREATE TABLE bm25_corpus_stats (
            term TEXT PRIMARY KEY,
            doc_freq INTEGER,
            corpus_term_freq INTEGER,
            total_docs INTEGER
        )
        """)
    cur.execute("""
        CREATE TABLE bm25_doc_metadata (
            doc_id TEXT PRIMARY KEY,
            source_category TEXT,
            repo TEXT,
            project TEXT
        )
        """)

    # Chunk metadata table (optional, parent-child mapping)
    cur.execute("""
        CREATE TABLE bm25_chunk_metadata (
            chunk_id TEXT PRIMARY KEY,
            parent_id TEXT,
            chunk_index INTEGER
        )
        """)

    # Insert document metadata (parents only)
    cur.executemany(
        "INSERT INTO bm25_doc_metadata(doc_id, source_category, repo, project) VALUES(?,?,?,?)",
        [
            ("parent1", "governance", "r1", "p1"),
            ("parent2", "code", "r2", "p2"),
            ("parent3", "patterns", "r3", "p3"),
        ],
    )

    # Insert chunk metadata (chunk -> parent mapping)
    cur.executemany(
        "INSERT INTO bm25_chunk_metadata(chunk_id, parent_id, chunk_index) VALUES(?,?,?)",
        [
            ("chunk1", "parent1", 0),
            ("chunk2", "parent1", 1),
            ("chunk3", "parent2", 0),
            ("chunk4", "parent2", 1),
            ("chunk5", "parent3", 0),
        ],
    )

    # Insert index rows (use chunk_ids from bm25_chunk_metadata)
    cur.executemany(
        "INSERT INTO bm25_index(doc_id, term, tf, doc_length) VALUES(?,?,?,?)",
        [
            ("chunk1", "auth", 2, 100),
            ("chunk2", "auth", 1, 120),
            ("chunk3", "auth", 5, 80),
            ("chunk4", "auth", 3, 110),
            ("chunk5", "login", 2, 90),
        ],
    )

    cur.executemany(
        "INSERT INTO bm25_corpus_stats(term, doc_freq, corpus_term_freq, total_docs) VALUES(?,?,?,?)",
        [
            ("auth", 4, 11, 3),
            ("login", 1, 2, 1),
        ],
    )

    conn.commit()
    return conn


def _setup_db_without_chunk_metadata():
    """Create in-memory DB without chunk metadata (fallback to chunk-level counts)."""

    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE bm25_index (
            doc_id TEXT,
            term TEXT,
            tf INTEGER,
            doc_length INTEGER
        )
        """)
    cur.execute("""
        CREATE TABLE bm25_corpus_stats (
            term TEXT PRIMARY KEY,
            doc_freq INTEGER,
            corpus_term_freq INTEGER,
            total_docs INTEGER
        )
        """)
    cur.execute("""
        CREATE TABLE bm25_doc_metadata (
            doc_id TEXT PRIMARY KEY,
            source_category TEXT,
            repo TEXT,
            project TEXT
        )
        """)

    # Insert document metadata
    cur.executemany(
        "INSERT INTO bm25_doc_metadata(doc_id, source_category, repo, project) VALUES(?,?,?,?)",
        [
            ("chunk1", "governance", "r1", "p1"),
            ("chunk2", "governance", "r1", "p1"),
            ("chunk3", "code", "r2", "p2"),
        ],
    )

    # Insert index rows (no chunk metadata table, so treat chunk_ids as docs)
    cur.executemany(
        "INSERT INTO bm25_index(doc_id, term, tf, doc_length) VALUES(?,?,?,?)",
        [
            ("chunk1", "auth", 2, 100),
            ("chunk2", "auth", 1, 120),
            ("chunk3", "auth", 5, 80),
        ],
    )

    cur.executemany(
        "INSERT INTO bm25_corpus_stats(term, doc_freq, corpus_term_freq, total_docs) VALUES(?,?,?,?)",
        [
            ("auth", 3, 8, 3),
        ],
    )

    conn.commit()
    return conn


def test_parent_docs_with_chunk_metadata():
    from scripts.rag.counts_service import CountsService

    conn = None
    svc = None

    try:

        conn = _setup_db_with_chunk_metadata()
        svc = CountsService(conn=conn)

        # With chunk metadata: 4 chunks mentioning "auth" should aggregate to 2 parents
        # (chunk1, chunk2 -> parent1; chunk3, chunk4 -> parent2)
        parent_count = svc.count_parent_docs_for_term("auth")
        assert parent_count == 2

        # List parent docs
        parent_docs = svc.list_parent_docs_for_term("auth", limit=10)
        assert len(parent_docs) == 2
        assert set(parent_docs) == {"parent1", "parent2"}

        # Category filter
        parent_count_gov = svc.count_parent_docs_for_term("auth", source_category="governance")
        assert parent_count_gov == 1  # Only parent1 is in governance

        parent_docs_gov = svc.list_parent_docs_for_term(
            "auth", source_category="governance", limit=10
        )
        assert parent_docs_gov == ["parent1"]
    finally:
        if conn:
            conn.close()
        if svc:
            svc.close()


def test_parent_document_summary():
    from scripts.rag.counts_service import CountsService

    conn = None
    svc = None
    try:

        conn = _setup_db_with_chunk_metadata()
        svc = CountsService(conn=conn)

        summary = svc.parent_document_summary("auth", limit=2)

        assert summary["term"] == "auth"
        assert summary["total_parent_docs"] == 2
        assert summary["total_occurrences"] == 11
        assert len(summary["sample_parent_docs"]) == 2
        assert set(summary["sample_parent_docs"]) == {"parent1", "parent2"}
        assert isinstance(summary["category_breakdown"], list)
    finally:
        if conn:
            conn.close()
        if svc:
            svc.close()


def test_parent_fallback_without_chunk_metadata():
    from scripts.rag.counts_service import CountsService

    conn = None
    svc = None

    parent_count = 0
    parent_docs = []

    try:
        conn = _setup_db_without_chunk_metadata()
        svc = CountsService(conn=conn)

        # Without chunk_metadata table, should fallback to chunk-level counts
        parent_count = svc.count_parent_docs_for_term("auth")

        parent_docs = svc.list_parent_docs_for_term("auth", limit=10)
        assert parent_count == 3  # Falls back to 3 chunks
        assert len(parent_docs) == 3
        assert set(parent_docs) == {"chunk1", "chunk2", "chunk3"}

    finally:
        if conn:
            conn.close()
        if svc:
            svc.close()


def test_parent_list_pagination():
    from scripts.rag.counts_service import CountsService

    conn = None
    svc = None

    page1 = []
    page2 = []

    try:

        conn = _setup_db_with_chunk_metadata()
        svc = CountsService(conn=conn)

        # Test pagination
        page1 = svc.list_parent_docs_for_term("auth", limit=1, offset=0)
        page2 = svc.list_parent_docs_for_term("auth", limit=1, offset=1)
        assert len(page1) == 1
        assert len(page2) == 1
        assert page1[0] != page2[0]
        assert set(page1 + page2) == {"parent1", "parent2"}

    finally:

        if conn:
            conn.close()
        if svc:
            svc.close()
