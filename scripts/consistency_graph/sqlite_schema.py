"""
SQLite schema for consistency graph with undirected edges.

Provides schema creation SQL statements and initialisation logic for the
consistency graph SQLite database. Edges are stored canonically (node_a < node_b)
to enforce undirected semantics while preserving raw metrics for query-time filtering.

Features:
- Undirected edge storage with CHECK constraint
- Raw metrics (similarity, confidence, severity) stored exactly as computed
- edges_directed VIEW for compatibility with existing code expecting source/target
- Node metadata with JSON fields for arrays
- Cluster membership tracking (primary + secondary)
- Graph-level metadata table

Usage:
    import sqlite3
    from scripts.consistency_graph.sqlite_schema import ensure_schema

    conn = sqlite3.connect("consistency_graph.sqlite")
    ensure_schema(conn)
"""

import sqlite3
from typing import Optional

# Schema creation statements (DDL)
SCHEMA_SQL = [
    # ========================================================================
    # NODES: Document/code node metadata
    # ========================================================================
    """
    CREATE TABLE IF NOT EXISTS nodes (
        node_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        version INTEGER NOT NULL,
        
        -- Document metadata
        doc_type TEXT,
        timestamp TEXT,
        summary TEXT,
        source_category TEXT,
        repository TEXT,
        health TEXT,  -- JSON blob
        
        -- Code-specific fields (JSON arrays stored as TEXT)
        language TEXT,
        service_name TEXT,
        service_type TEXT,
        dependencies TEXT,      -- JSON array
        internal_calls TEXT,    -- JSON array
        endpoints TEXT,         -- JSON array
        db TEXT,                -- JSON array (database refs)
        queue TEXT,             -- JSON array
        exports TEXT,           -- JSON array
        
        -- Computed aggregates (updated after edge build)
        conflict_score REAL DEFAULT 0.0,
        topic_clusters TEXT,    -- JSON array of cluster IDs
        risk_clusters TEXT,     -- JSON array of cluster IDs
        
        UNIQUE(doc_id, version)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_nodes_doc_id ON nodes(doc_id);",
    "CREATE INDEX IF NOT EXISTS idx_nodes_source_category ON nodes(source_category);",
    "CREATE INDEX IF NOT EXISTS idx_nodes_language ON nodes(language);",
    "CREATE INDEX IF NOT EXISTS idx_nodes_service_name ON nodes(service_name);",
    "CREATE INDEX IF NOT EXISTS idx_nodes_repository ON nodes(repository);",
    "CREATE INDEX IF NOT EXISTS idx_nodes_conflict_score ON nodes(conflict_score DESC);",
    # ========================================================================
    # EDGES: Consistency relationships (undirected)
    # ========================================================================
    """
    CREATE TABLE IF NOT EXISTS edges (
        node_a TEXT NOT NULL,
        node_b TEXT NOT NULL,
        
        -- Raw metrics (exact values from computation)
        similarity REAL NOT NULL,       -- [0, 1]: Embedding similarity
        confidence REAL NOT NULL,       -- [0, 1]: LLM confidence score
        severity REAL NOT NULL,         -- [0, 1]: Computed severity
        
        -- Relationship metadata
        relationship TEXT NOT NULL,     -- conflict | partial_conflict | duplicate | consistent | cross_repo_service
        explanation TEXT,               -- Human-readable reason
        
        -- Version tracking
        version_source INTEGER,
        version_target INTEGER,
        
        -- Quality flags
        interpolated INTEGER DEFAULT 0, -- 1 if edge was interpolated during removal
        
        -- Audit trail
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        
        -- Enforce canonical ordering (lexicographic)
        CHECK (node_a < node_b),
        PRIMARY KEY (node_a, node_b)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_edges_relationship ON edges(relationship);",
    "CREATE INDEX IF NOT EXISTS idx_edges_similarity ON edges(similarity DESC);",
    "CREATE INDEX IF NOT EXISTS idx_edges_confidence ON edges(confidence DESC);",
    "CREATE INDEX IF NOT EXISTS idx_edges_severity ON edges(severity DESC);",
    "CREATE INDEX IF NOT EXISTS idx_edges_node_a ON edges(node_a);",
    "CREATE INDEX IF NOT EXISTS idx_edges_node_b ON edges(node_b);",
    # ========================================================================
    # CLUSTERS: Community detection results
    # ========================================================================
    """
    CREATE TABLE IF NOT EXISTS clusters (
        cluster_id TEXT PRIMARY KEY,    -- Format: {type}_{id} e.g., "risk_1", "topic_3"
        
        -- Cluster identity
        cluster_type TEXT NOT NULL,     -- 'risk' | 'topic'
        cluster_name TEXT,              -- Human-readable label from LLM
        
        -- Cluster metadata
        summary TEXT,                   -- LLM-generated description
        node_count INTEGER,
        edge_count INTEGER,
        metadata TEXT,                  -- JSON: arbitrary cluster-specific data
        
        -- Computed properties
        modularity_contribution REAL,   -- How well this cluster groups together
        primary_relationship TEXT,      -- Most common edge type in cluster
        
        -- Audit trail
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        
        UNIQUE(cluster_type, cluster_id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_clusters_type ON clusters(cluster_type);",
    "CREATE INDEX IF NOT EXISTS idx_clusters_name ON clusters(cluster_name);",
    # ========================================================================
    # NODE_CLUSTERS: Many-to-many node-cluster membership
    # ========================================================================
    """
    CREATE TABLE IF NOT EXISTS node_clusters (
        node_id TEXT NOT NULL,
        cluster_id TEXT NOT NULL,
        
        -- Membership strength (how well node fits in cluster)
        strength REAL DEFAULT 1.0,          -- [0, 1]: Avg edge weight to cluster members
        primary_cluster INTEGER DEFAULT 0,  -- 1 if this is node's primary cluster
        
        PRIMARY KEY (node_id, cluster_id),
        FOREIGN KEY (node_id) REFERENCES nodes(node_id),
        FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_node_clusters_node ON node_clusters(node_id);",
    "CREATE INDEX IF NOT EXISTS idx_node_clusters_cluster ON node_clusters(cluster_id);",
    "CREATE INDEX IF NOT EXISTS idx_node_clusters_strength ON node_clusters(strength DESC);",
    # ========================================================================
    # METADATA: Graph-level metadata and build tracking
    # ========================================================================
    """
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """,
]

# Compatibility view: expose edges in both directions (source/target)
VIEW_SQL = """
CREATE VIEW IF NOT EXISTS edges_directed AS
SELECT
  node_a AS source,
  node_b AS target,
  similarity, confidence, severity,
  relationship, explanation,
  version_source, version_target,
  interpolated, created_at
FROM edges
UNION ALL
SELECT
  node_b AS source,
  node_a AS target,
  similarity, confidence, severity,
  relationship, explanation,
  version_source, version_target,
  interpolated, created_at
FROM edges;
"""


def ensure_schema(conn: sqlite3.Connection, drop_existing: bool = False) -> None:
    """
    Create or verify the consistency graph schema.

    Args:
        conn: Open SQLite connection
        drop_existing: If True, drop all tables before creating (WARNING: destroys data)

    Side effects:
        Creates tables and indexes in the database.
        Creates edges_directed view for compatibility.
    """
    cur = conn.cursor()

    if drop_existing:
        # Drop in reverse order to avoid FK constraints
        cur.execute("DROP VIEW IF EXISTS edges_directed;")
        cur.execute("DROP TABLE IF EXISTS node_clusters;")
        cur.execute("DROP TABLE IF EXISTS clusters;")
        cur.execute("DROP TABLE IF EXISTS edges;")
        cur.execute("DROP TABLE IF EXISTS nodes;")
        cur.execute("DROP TABLE IF EXISTS metadata;")
        conn.commit()

    # Create tables and indexes
    for stmt in SCHEMA_SQL:
        cur.execute(stmt)

    # Create compatibility view
    cur.execute(VIEW_SQL)

    conn.commit()


def get_metadata_value(conn: sqlite3.Connection, key: str) -> Optional[str]:
    """
    Retrieve a single metadata value by key.

    Args:
        conn: Open SQLite connection
        key: Metadata key to look up

    Returns:
        Value string or None if key not found
    """
    cur = conn.cursor()
    row = cur.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_metadata_value(conn: sqlite3.Connection, key: str, value: str) -> None:
    """
    Set a metadata key-value pair (insert or update).

    Args:
        conn: Open SQLite connection
        key: Metadata key
        value: Value to store (converted to string)
    """
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)", (key, str(value)))
    conn.commit()
