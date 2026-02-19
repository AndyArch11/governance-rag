"""
SQLite writer for consistency graph with progressive and incremental writes.

Supports writing graph data to SQLite progressively during build (nodes upfront,
edges as computed) with atomic swap capability to minimise dashboard outage.

Features:
- Progressive writes: insert nodes before edge building, edges as futures complete
- Edge canonicalisation: node_a < node_b enforcement
- Atomic swap: build to temp DB, rename when complete
- Resume support: skip already-processed nodes via metadata tracking
- Thread-safe edge insertion for parallel workers

Usage:
    from scripts.consistency_graph.sqlite_writer import SQLiteGraphWriter

    # Progressive build (nodes first, then edges, then clusters)
    writer = SQLiteGraphWriter("consistency_graph.sqlite.tmp", replace=True)
    writer.insert_nodes_batch(nodes_data)
    writer.insert_edge(source, target, similarity, confidence, severity, relationship)
    writer.insert_clusters(clusters_data)
    writer.set_build_metadata("total_nodes", len(nodes_data))
    writer.atomic_swap("consistency_graph.sqlite")

    # Or, one-shot write for complete graph dictionaries
    from scripts.consistency_graph.sqlite_writer import write_graph_to_sqlite

    write_graph_to_sqlite(graph_dict, "consistency_graph.sqlite", replace=True)
"""

import json
import shutil
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .sqlite_schema import ensure_schema, get_metadata_value, set_metadata_value


def _canon(a: str, b: str) -> Tuple[str, str]:
    """Return canonical ordering of two node IDs (lexicographic)."""
    return (a, b) if a < b else (b, a)


class SQLiteGraphWriter:
    """
    Thread-safe writer for progressive consistency graph persistence.

    Supports building graphs incrementally with:
    - Batch node insertion before edge computation
    - Per-edge insertion as futures complete (thread-safe)
    - Cluster metadata insertion at end
    - Atomic swap to production DB path
    """

    def __init__(self, sqlite_path: Union[str, Path], replace: bool = True):
        """
        Initialise writer with target SQLite path.

        Args:
            sqlite_path: Path to SQLite database file
            replace: If True, drop existing tables before creating schema
        """
        self.sqlite_path = str(sqlite_path)
        self._lock = threading.Lock()  # Thread-safe edge insertion
        self._conn = None
        self._replace = replace
        self._node_count = 0
        self._edge_count = 0

        # Ensure directory exists
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialise schema
        self._init_db()

    def _init_db(self) -> None:
        """Create database connection and schema."""
        self._conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        ensure_schema(self._conn, drop_existing=self._replace)

        # Set pragmas for performance
        self._conn.execute("PRAGMA journal_mode=WAL;")  # Write-ahead logging
        self._conn.execute("PRAGMA synchronous=NORMAL;")  # Faster writes
        self._conn.execute("PRAGMA cache_size=-64000;")  # 64MB cache
        self._conn.commit()

    def insert_nodes_batch(self, nodes: Dict[str, Dict[str, Any]]) -> int:
        """
        Insert all nodes in a single transaction (upfront, before edges).

        Args:
            nodes: Dict mapping node_id -> node metadata dict

        Returns:
            Number of nodes inserted
        """
        if not nodes:
            return 0

        node_insert = """
        INSERT OR REPLACE INTO nodes (
            node_id, doc_id, version, doc_type, timestamp, summary, source_category,
            repository, health, language, service_name, service_type, dependencies,
            internal_calls, endpoints, db, queue, exports, conflict_score,
            topic_clusters, risk_clusters
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """

        def to_json(v):
            """Convert value to JSON string if not None."""
            return json.dumps(v, ensure_ascii=False) if v is not None else None

        rows = []
        for node_id, meta in nodes.items():
            row = (
                node_id,
                meta.get("doc_id", node_id),
                int(meta.get("version", 1)),
                meta.get("doc_type"),
                meta.get("timestamp"),
                meta.get("summary"),
                meta.get("source_category"),
                meta.get("repository"),
                to_json(meta.get("health")),
                meta.get("language"),
                meta.get("service_name") or meta.get("service"),
                meta.get("service_type"),
                to_json(meta.get("dependencies")),
                to_json(meta.get("internal_calls")),
                to_json(meta.get("endpoints")),
                to_json(meta.get("db")),
                to_json(meta.get("queue")),
                to_json(meta.get("exports")),
                float(meta.get("conflict_score", 0.0)),
                to_json(meta.get("topic_clusters")),
                to_json(meta.get("risk_clusters")),
            )
            rows.append(row)

        with self._lock:
            cur = self._conn.cursor()
            cur.executemany(node_insert, rows)
            self._conn.commit()
            self._node_count = len(nodes)

        return len(rows)

    def insert_edge(
        self,
        source: str,
        target: str,
        similarity: float,
        confidence: float,
        severity: float,
        relationship: str,
        explanation: Optional[str] = None,
        version_source: Optional[int] = None,
        version_target: Optional[int] = None,
        interpolated: bool = False,
    ) -> None:
        """
        Insert a single edge (thread-safe, canonicalised).

        Args:
            source: Source node ID
            target: Target node ID
            similarity: Embedding similarity [0, 1]
            confidence: LLM confidence [0, 1]
            severity: Computed severity [0, 1]
            relationship: Edge type (conflict, partial_conflict, duplicate, consistent, etc.)
            explanation: Human-readable explanation
            version_source: Source document version
            version_target: Target document version
            interpolated: True if edge was interpolated (not direct LLM output)
        """
        if not source or not target:
            return

        # Canonicalise: node_a < node_b
        node_a, node_b = _canon(str(source), str(target))

        edge_insert = """
        INSERT OR REPLACE INTO edges (
            node_a, node_b, similarity, confidence, severity, relationship, explanation,
            version_source, version_target, interpolated
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """

        row = (
            node_a,
            node_b,
            float(similarity),
            float(confidence),
            float(severity),
            str(relationship),
            explanation,
            version_source,
            version_target,
            1 if interpolated else 0,
        )

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(edge_insert, row)
            self._conn.commit()
            self._edge_count += 1

    def insert_edges_batch(self, edges: List[Dict[str, Any]]) -> int:
        """
        Insert multiple edges in a single transaction (faster than one-by-one).

        Args:
            edges: List of edge dicts with keys: source, target, similarity, confidence,
                   severity, relationship, explanation, version_source, version_target, interpolated

        Returns:
            Number of edges inserted
        """
        if not edges:
            return 0

        edge_insert = """
        INSERT OR REPLACE INTO edges (
            node_a, node_b, similarity, confidence, severity, relationship, explanation,
            version_source, version_target, interpolated
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """

        rows = []
        for e in edges:
            s = e.get("source")
            t = e.get("target")
            if not s or not t:
                continue

            a, b = _canon(str(s), str(t))
            row = (
                a,
                b,
                float(e.get("similarity", 0.0)),
                float(e.get("confidence", 0.0)),
                float(e.get("severity", 0.0)),
                str(e.get("relationship") or "consistent"),
                e.get("explanation"),
                e.get("version_source"),
                e.get("version_target"),
                1 if e.get("interpolated") else 0,
            )
            rows.append(row)

        with self._lock:
            cur = self._conn.cursor()
            cur.executemany(edge_insert, rows)
            self._conn.commit()
            self._edge_count += len(rows)

        return len(rows)

    def update_node_metrics(
        self,
        node_id: str,
        conflict_score: float,
        topic_clusters: List[int],
        risk_clusters: List[int],
    ) -> None:
        """
        Update node with computed metrics (called after edge building).

        Args:
            node_id: Node to update
            conflict_score: Sum of edge severities
            topic_clusters: List of topic cluster IDs
            risk_clusters: List of risk cluster IDs
        """
        update = """
        UPDATE nodes 
        SET conflict_score = ?, topic_clusters = ?, risk_clusters = ?
        WHERE node_id = ?
        """

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                update,
                (
                    float(conflict_score),
                    json.dumps(topic_clusters, ensure_ascii=False),
                    json.dumps(risk_clusters, ensure_ascii=False),
                    node_id,
                ),
            )
            self._conn.commit()

    def insert_clusters(self, clusters: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Insert cluster metadata and node-cluster memberships.

        Args:
            clusters: Dict with "risk" and "topic" keys, each containing list of cluster dicts
                     with keys: id, type, size, members, label, description, summary
        """
        cluster_insert = """
        INSERT OR REPLACE INTO clusters (
            cluster_id, cluster_type, cluster_name, summary, node_count, edge_count, metadata
        ) VALUES (?,?,?,?,?,?,?)
        """

        node_cluster_insert = """
        INSERT OR REPLACE INTO node_clusters (node_id, cluster_id, strength, primary_cluster)
        VALUES (?,?,?,?)
        """

        with self._lock:
            cur = self._conn.cursor()

            for ctype in ("risk", "topic"):
                for c in clusters.get(ctype, []):
                    cid = f"{ctype}_{c['id']}"
                    meta = {
                        "description": c.get("description"),
                        "label": c.get("label"),
                    }

                    cur.execute(
                        cluster_insert,
                        (
                            cid,
                            ctype,
                            c.get("label"),
                            c.get("summary"),
                            int(c.get("size", len(c.get("members") or []))),
                            None,  # edge_count computed later if needed
                            json.dumps(meta, ensure_ascii=False),
                        ),
                    )

                    # Insert node-cluster memberships (first is primary)
                    for idx, nid in enumerate(c.get("members") or []):
                        cur.execute(node_cluster_insert, (nid, cid, 1.0, 1 if idx == 0 else 0))

            self._conn.commit()

    def set_build_metadata(self, key: str, value: Any) -> None:
        """
        Store graph build metadata (settings, timestamps, progress tracking).

        Args:
            key: Metadata key
            value: Value to store (will be converted to string)
        """
        set_metadata_value(self._conn, key, str(value))

    def get_build_metadata(self, key: str) -> Optional[str]:
        """
        Retrieve build metadata by key.

        Args:
            key: Metadata key

        Returns:
            Value string or None if not found
        """
        return get_metadata_value(self._conn, key)

    def mark_node_processed(self, node_id: str) -> None:
        """
        Mark a node as fully processed (for resume support).

        Updates metadata table with processed_nodes JSON array.
        """
        processed = self.get_build_metadata("processed_nodes")
        if processed:
            nodes_list = json.loads(processed)
        else:
            nodes_list = []

        if node_id not in nodes_list:
            nodes_list.append(node_id)
            self.set_build_metadata("processed_nodes", json.dumps(nodes_list))

    def get_processed_nodes(self) -> List[str]:
        """
        Get list of nodes already processed (for resume).

        Returns:
            List of node IDs that have been fully processed
        """
        processed = self.get_build_metadata("processed_nodes")
        return json.loads(processed) if processed else []

    def close(self) -> None:
        """Close database connection (commits any pending transactions first)."""
        if self._conn:
            try:
                self._conn.commit()  # Ensure all pending writes are committed
            except:
                pass  # Ignore commit errors if transaction is already complete
            self._conn.close()
            self._conn = None

    def atomic_swap(self, target_path: Union[str, Path]) -> None:
        """
        Atomically replace target database with this one.

        Closes connection, moves temp DB to target path.
        Minimises dashboard outage by making swap atomic at filesystem level.

        Args:
            target_path: Production database path to replace
        """
        target_path = str(target_path)

        # Close connection to release file lock
        self.close()

        # Ensure target directory exists
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)

        # Atomically move temp DB to production path
        # (self.sqlite_path already points to temp file like consistency_graph.sqlite.tmp)
        shutil.move(self.sqlite_path, target_path)

        # Update path reference (no need to reopen - we're done writing)
        self.sqlite_path = target_path

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit (closes connection)."""
        self.close()


def write_graph_to_sqlite(
    graph: Dict[str, Any], sqlite_path: Union[str, Path], replace: bool = True
) -> None:
    """
    Convenience function: write a complete graph dict to SQLite in one call.

    Args:
        graph: Graph dictionary with keys: nodes, edges, clusters
        sqlite_path: Path to SQLite database file
        replace: If True, drop existing tables before writing
    """
    with SQLiteGraphWriter(sqlite_path, replace=replace) as writer:
        # Insert nodes
        nodes = graph.get("nodes") or {}
        writer.insert_nodes_batch(nodes)

        # Insert edges
        edges = graph.get("edges") or []
        writer.insert_edges_batch(edges)

        # Insert clusters
        clusters = graph.get("clusters") or {}
        writer.insert_clusters(clusters)

        # Store metadata
        writer.set_build_metadata("total_nodes", len(nodes))
        writer.set_build_metadata("total_edges", len(edges))
