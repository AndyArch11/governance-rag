"""Unit tests for scripts.rag.counts_service.CountsService.

Creates an in-memory SQLite schema that mimics BM25 ingestion-time tables
and validates counts, occurrences, listing, breakdown, AND-counts, and summary.
"""

import sqlite3


def _setup_in_memory_db():
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    # Create tables
    cur.execute(
        """
        CREATE TABLE bm25_index (
            doc_id TEXT,
            term TEXT,
            tf INTEGER,
            doc_length INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE bm25_corpus_stats (
            term TEXT PRIMARY KEY,
            doc_freq INTEGER,
            corpus_term_freq INTEGER,
            total_docs INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE bm25_doc_metadata (
            doc_id TEXT PRIMARY KEY,
            source_category TEXT,
            repo TEXT,
            project TEXT
        )
        """
    )

    # Insert metadata
    cur.executemany(
        "INSERT INTO bm25_doc_metadata(doc_id, source_category, repo, project) VALUES(?,?,?,?)",
        [
            ("doc1", "governance", "r1", "p1"),
            ("doc2", "code", "r2", "p2"),
            ("doc3", "patterns", "r3", "p3"),
        ],
    )

    # Insert index rows
    cur.executemany(
        "INSERT INTO bm25_index(doc_id, term, tf, doc_length) VALUES(?,?,?,?)",
        [
            ("doc1", "auth", 2, 100),
            ("doc2", "auth", 1, 120),
            ("doc3", "auth", 5, 80),
            ("doc2", "login", 3, 120),
        ],
    )

    # Insert corpus stats for 'auth'; omit for 'login' to test fallback
    cur.executemany(
        "INSERT INTO bm25_corpus_stats(term, doc_freq, corpus_term_freq, total_docs) VALUES(?,?,?,?)",
        [
            ("auth", 3, 8, 3),
        ],
    )

    conn.commit()
    return conn


def test_counts_service_core_functions():
    from scripts.rag.counts_service import CountsService

    conn = None
    svc = None
    
    try:

        conn = _setup_in_memory_db()
        svc = CountsService(conn=conn)

        # Distinct document counts
        assert svc.count_docs_for_term("auth") == 3
        assert svc.count_docs_for_term("auth", source_category="code") == 1

        # Occurrences via corpus stats
        assert svc.count_occurrences_for_term("auth") == 8

        # Occurrences fallback via bm25_index when stats missing
        assert svc.count_occurrences_for_term("login") == 3

        # Total documents reported via corpus stats table
        assert svc.total_documents() == 3

        # Listing of docs for term
        docs = svc.list_docs_for_term("auth", limit=10, offset=0)
        assert docs == ["doc1", "doc2", "doc3"]

        # Breakdown by category
        breakdown = svc.breakdown_by_category("auth", top_n=5)
        # Expect governance, code, patterns each with 1
        cats = dict(breakdown)
        assert cats.get("governance") == 1
        assert cats.get("code") == 1
        assert cats.get("patterns") == 1

        # AND-term doc counts
        assert svc.count_docs_for_and(["auth", "login"]) == 1

        # Summary structure
        summary = svc.summarise_term("auth", limit=2)
        assert summary["term"] == "auth"
        assert summary["total_docs"] == 3
        assert summary["total_occurrences"] == 8
        assert summary["sample_docs"] == ["doc1", "doc2"]
        assert isinstance(summary["category_breakdown"], list)

    finally:
        if conn:
            conn.close()
        if svc:
            svc.close()
