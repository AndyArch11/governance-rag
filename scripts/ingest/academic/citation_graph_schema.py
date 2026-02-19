"""
SQLite schema for citation graph.

Tables:
- nodes: Citation nodes (document or reference)
- edges: Citation relationships
- metadata: Graph metadata (build time, document info, etc.)
"""

import sqlite3
from typing import Optional


def ensure_schema(conn: sqlite3.Connection, drop_existing: bool = False) -> None:
    """
    Create citation graph schema in SQLite database.

    Args:
        conn: SQLite connection
        drop_existing: If True, drop existing tables first
    """
    cursor = conn.cursor()

    if drop_existing:
        cursor.execute("DROP TABLE IF EXISTS edges")
        cursor.execute("DROP TABLE IF EXISTS nodes")
        cursor.execute("DROP TABLE IF EXISTS metadata")

    # Metadata table - graph-level information
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Nodes table - documents and references
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            title TEXT,
            authors TEXT,
            doi TEXT,
            year INTEGER,
            year_verified INTEGER,
            reference_type TEXT,
            quality_score REAL DEFAULT 0.0,
            link_status TEXT DEFAULT 'available',
            venue_type TEXT,
            venue_rank TEXT,
            source TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Add missing columns for older databases
    cursor.execute("PRAGMA table_info(nodes)")
    node_columns = {row[1] for row in cursor.fetchall()}
    if "quality_score" not in node_columns:
        cursor.execute("ALTER TABLE nodes ADD COLUMN quality_score REAL DEFAULT 0.0")

    # Edges table - citation relationships
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS edges (
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            relation TEXT DEFAULT 'cites',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source, target),
            FOREIGN KEY (source) REFERENCES nodes(node_id),
            FOREIGN KEY (target) REFERENCES nodes(node_id)
        )
    """
    )

    # Indexes for common queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_source ON nodes(source)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_year ON nodes(year)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_doi ON nodes(doi)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_reference_type ON nodes(reference_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_quality_score ON nodes(quality_score)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_link_status ON nodes(link_status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_venue_type ON nodes(venue_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_venue_rank ON nodes(venue_rank)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)")

    conn.commit()


def set_metadata_value(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Store metadata key-value pair."""
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value))
    conn.commit()


def get_metadata_value(conn: sqlite3.Connection, key: str) -> Optional[str]:
    """Retrieve metadata value by key."""
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
    row = cursor.fetchone()
    return row[0] if row else None
