"""
SQLite-backed graph store for dashboard lazy loading.

Supports pagination, filtering, and on-demand data loading.

Features:
- Connection pooling for concurrent dashboard requests
- Lazy loading: metadata loaded immediately, full data on-demand
- Pagination support for large graphs
- ChromaDB integration for document text retrieval
- Undirected edge support via edges_directed view

Usage:
    from scripts.consistency_graph.sqlite_store import SQLiteGraphStore

    store = SQLiteGraphStore("consistency_graph.sqlite")
    store.load_metadata()

    node_ids = store.get_node_ids()
    nodes, total_pages = store.get_nodes_paginated(page=0, page_size=50)
    edges = store.get_edges()
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from .sqlite_schema import ensure_schema


class SQLiteGraphStore:
    """
    SQLite-backed graph store with API.

    Provides efficient read-only access to consistency graph data stored
    in SQLite. Optimised for dashboard querying with connection pooling
    and lazy loading.
    """

    def __init__(self, sqlite_path: str):
        """
        Initialise store with SQLite database path.

        Args:
            sqlite_path: Path to SQLite database file
        """
        self.sqlite_path = sqlite_path
        self._metadata = {}
        self._clusters = {"risk": [], "topic": []}
        self._all_node_ids = []
        self._collection = None
        self._doc_cache = {}
        self._lock = threading.Lock()

        # Track file inode to detect atomic swaps (file replacement)
        self._last_inode = None

        # Connection will be created on-demand per thread
        self._thread_local = threading.local()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection (connection pooling).

        Automatically detects when the database file has been replaced (e.g., atomic_swap)
        and reloads the connection to pick up the new data.
        """
        import os

        # Check if file was replaced by comparing inode
        try:
            current_inode = (
                os.stat(self.sqlite_path).st_ino if os.path.exists(self.sqlite_path) else None
            )
            if current_inode != self._last_inode:
                # File was replaced or doesn't exist yet - close old connection and clear cache
                if hasattr(self._thread_local, "conn") and self._thread_local.conn is not None:
                    try:
                        self._thread_local.conn.close()
                    except:
                        pass
                self._thread_local.conn = None
                self._last_inode = current_inode

                # Clear cached data so load_metadata() will reload from new file
                self._all_node_ids = []
                self._metadata = {}
                self._clusters = {"risk": [], "topic": []}
        except:
            pass  # If stat fails, just proceed with existing connection

        if not hasattr(self._thread_local, "conn") or self._thread_local.conn is None:
            self._thread_local.conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
            self._thread_local.conn.row_factory = sqlite3.Row  # Dict-like access
            ensure_schema(self._thread_local.conn, drop_existing=False)
        return self._thread_local.conn

    def load_metadata(self) -> bool:
        """
        Load graph metadata without loading full node/edge data.

        Returns:
            True if metadata loaded successfully, False otherwise
        """
        try:
            conn = self._get_conn()
            cur = conn.cursor()

            # Load metadata table
            rows = cur.execute("SELECT key, value FROM metadata").fetchall()
            self._metadata = {row["key"]: row["value"] for row in rows}

            # Load all node IDs (lightweight)
            rows = cur.execute("SELECT node_id FROM nodes ORDER BY node_id").fetchall()
            self._all_node_ids = [row["node_id"] for row in rows]

            # Load cluster metadata (lightweight)
            risk_rows = cur.execute(
                """
                SELECT cluster_id, cluster_name, summary, node_count, metadata
                FROM clusters
                WHERE cluster_type = 'risk'
                ORDER BY cluster_id
                """
            ).fetchall()

            topic_rows = cur.execute(
                """
                SELECT cluster_id, cluster_name, summary, node_count, metadata
                FROM clusters
                WHERE cluster_type = 'topic'
                ORDER BY cluster_id
                """
            ).fetchall()

            self._clusters["risk"] = [
                {
                    "id": int(row["cluster_id"].split("_")[1]),
                    "type": "risk",
                    "label": row["cluster_name"],
                    "summary": row["summary"],
                    "size": row["node_count"],
                    **(json.loads(row["metadata"]) if row["metadata"] else {}),
                }
                for row in risk_rows
            ]

            self._clusters["topic"] = [
                {
                    "id": int(row["cluster_id"].split("_")[1]),
                    "type": "topic",
                    "label": row["cluster_name"],
                    "summary": row["summary"],
                    "size": row["node_count"],
                    **(json.loads(row["metadata"]) if row["metadata"] else {}),
                }
                for row in topic_rows
            ]

            return True

        except Exception as e:
            print(f"Error loading graph metadata from SQLite: {e}")
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """Return graph metadata dict."""
        return self._metadata

    def get_node_ids(self) -> List[str]:
        """Return all node IDs (loaded during metadata load)."""
        return self._all_node_ids

    def get_nodes_paginated(
        self, page: int = 0, page_size: int = 50
    ) -> Tuple[List[Tuple[str, Dict]], int]:
        """
        Get paginated nodes with full data.

        Args:
            page: Page number (0-indexed)
            page_size: Number of nodes per page

        Returns:
            Tuple of (nodes_data, total_pages)
            nodes_data is list of (node_id, node_dict) tuples
        """
        total = len(self._all_node_ids)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1

        offset = page * page_size
        limit = page_size

        conn = self._get_conn()
        cur = conn.cursor()

        rows = cur.execute(
            """
            SELECT * FROM nodes
            ORDER BY node_id
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

        nodes_data = []
        for row in rows:
            node_dict = dict(row)
            node_id = node_dict["node_id"]

            # Deserialise JSON fields
            for field in [
                "health",
                "dependencies",
                "internal_calls",
                "endpoints",
                "db",
                "queue",
                "exports",
                "topic_clusters",
                "risk_clusters",
            ]:
                if node_dict.get(field):
                    try:
                        node_dict[field] = json.loads(node_dict[field])
                    except (json.JSONDecodeError, TypeError):
                        pass  # Keep as-is if not valid JSON

            nodes_data.append((node_id, node_dict))

        return nodes_data, total_pages

    def get_edges(self) -> List[Dict[str, Any]]:
        """
        Get all edges (uses edges_directed view for compatibility).

        Returns:
            List of edge dicts with keys: source, target, similarity, confidence,
            severity, relationship, explanation, version_source, version_target, etc.
        """
        conn = self._get_conn()
        cur = conn.cursor()

        rows = cur.execute(
            """
            SELECT source, target, similarity, confidence, severity,
                   relationship, explanation, version_source, version_target,
                   interpolated, created_at
            FROM edges_directed
            ORDER BY source, target
            """
        ).fetchall()

        edges = []
        for row in rows:
            edge = {
                "source": row["source"],
                "target": row["target"],
                "similarity": row["similarity"],
                "confidence": row["confidence"],
                "severity": row["severity"],
                "relationship": row["relationship"],
                "explanation": row["explanation"],
                "version_source": row["version_source"],
                "version_target": row["version_target"],
                "interpolated": bool(row["interpolated"]),
                "created_at": row["created_at"],
            }
            edges.append(edge)

        return edges

    def get_clusters(self) -> Dict[str, List[Dict]]:
        """Return risk and topic clusters (loaded during metadata load)."""
        return self._clusters

    def get_analytics(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve pre-computed advanced graph analytics from metadata.

        Returns:
            Analytics dict with keys: influence_scores, communities, topology, node_clustering
            or None if analytics were not computed during graph build
        """
        if self._metadata.get("analytics"):
            try:
                analytics_json = self._metadata.get("analytics", "{}")
                return (
                    json.loads(analytics_json)
                    if isinstance(analytics_json, str)
                    else analytics_json
                )
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single node by ID with full data.

        Args:
            node_id: Node identifier

        Returns:
            Node dict or None if not found
        """
        conn = self._get_conn()
        cur = conn.cursor()

        row = cur.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()

        if not row:
            return None

        node_dict = dict(row)

        # Deserialise JSON fields
        for field in [
            "health",
            "dependencies",
            "internal_calls",
            "endpoints",
            "db",
            "queue",
            "exports",
            "topic_clusters",
            "risk_clusters",
        ]:
            if node_dict.get(field):
                try:
                    node_dict[field] = json.loads(node_dict[field])
                except (json.JSONDecodeError, TypeError):
                    pass

        return node_dict

    def to_networkx_subgraph(self, node_ids: Set[str]) -> nx.Graph:
        """
        Convert a subset of nodes to NetworkX graph.

        Args:
            node_ids: Set of node IDs to include

        Returns:
            NetworkX Graph with requested nodes and edges between them
        """
        G = nx.Graph()

        if not node_ids:
            return G

        conn = self._get_conn()
        cur = conn.cursor()

        # Build placeholders for IN clause
        placeholders = ",".join("?" * len(node_ids))

        # Add nodes
        rows = cur.execute(
            f"SELECT * FROM nodes WHERE node_id IN ({placeholders})", tuple(node_ids)
        ).fetchall()

        for row in rows:
            node_dict = dict(row)
            node_id = node_dict["node_id"]

            # Deserialise JSON fields
            for field in [
                "health",
                "dependencies",
                "internal_calls",
                "endpoints",
                "db",
                "queue",
                "exports",
                "topic_clusters",
                "risk_clusters",
            ]:
                if node_dict.get(field):
                    try:
                        node_dict[field] = json.loads(node_dict[field])
                    except (json.JSONDecodeError, TypeError):
                        pass

            G.add_node(node_id, **node_dict)

        # Add edges (undirected: only need to check canonical storage)
        edge_rows = cur.execute(
            f"""
            SELECT * FROM edges
            WHERE node_a IN ({placeholders}) AND node_b IN ({placeholders})
            """,
            tuple(node_ids) + tuple(node_ids),
        ).fetchall()

        for row in edge_rows:
            edge_dict = {
                "similarity": row["similarity"],
                "confidence": row["confidence"],
                "severity": row["severity"],
                "relationship": row["relationship"],
                "explanation": row["explanation"],
                "version_source": row["version_source"],
                "version_target": row["version_target"],
                "interpolated": bool(row["interpolated"]),
            }
            # Add edge in both directions for NetworkX undirected graph
            G.add_edge(row["node_a"], row["node_b"], **edge_dict)

        return G

    def set_collection(self, collection):
        """Set ChromaDB collection for document lookups."""
        self._collection = collection

    def get_doc(self, node_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        Get document text and metadata (lazy loaded from ChromaDB).

        Args:
            node_id: Node identifier (may or may not include _vN version suffix)

        Returns:
            Tuple of (document_text, metadata_dict)
        """
        if node_id in self._doc_cache:
            return self._doc_cache[node_id]

        if self._collection is None:
            return "", {}

        try:
            # Parse version from node_id if it has format: doc_id_vN
            # Otherwise use version 1 as default or search for latest
            base = node_id
            version = None

            if "_v" in node_id:
                try:
                    base, v = node_id.rsplit("_v", 1)
                    version = int(v)
                except (ValueError, AttributeError):
                    base = node_id
                    version = None

            # Query ChromaDB for parent chunks
            try:
                if version is not None:
                    # Search for specific version
                    results = self._collection.get(
                        where={
                            "$and": [{"doc_id": base}, {"version": version}, {"is_parent": True}]
                        },
                        include=["documents", "metadatas"],
                    )
                else:
                    # Search for any version (prefer latest)
                    all_results = self._collection.get(
                        where={"$and": [{"doc_id": base}, {"is_parent": True}]},
                        include=["documents", "metadatas"],
                    )
                    # Find highest version
                    if all_results and all_results.get("metadatas"):
                        max_version = 1
                        best_doc_idx = 0
                        for i, meta in enumerate(all_results.get("metadatas", [])):
                            if meta.get("version", 1) > max_version:
                                max_version = meta.get("version", 1)
                                best_doc_idx = i

                        if all_results.get("documents"):
                            doc_text = all_results["documents"][best_doc_idx]
                            metadata = all_results["metadatas"][best_doc_idx]
                            self._doc_cache[node_id] = (doc_text, metadata)
                            return doc_text, metadata
                    results = {"documents": [], "metadatas": []}
            except Exception:
                # Fallback: try without $and
                results = self._collection.get(
                    where={"doc_id": base}, include=["documents", "metadatas"]
                )
                # Filter by is_parent and optionally version
                filtered_docs = []
                filtered_metas = []
                for i, meta in enumerate(results.get("metadatas", [])):
                    if meta.get("is_parent"):
                        if version is None or meta.get("version") == version:
                            if results.get("documents"):
                                filtered_docs.append(results["documents"][i])
                            filtered_metas.append(meta)
                results = {"documents": filtered_docs, "metadatas": filtered_metas}

            if results and results.get("documents") and len(results["documents"]) > 0:
                doc_text = results["documents"][0]
                metadata = results["metadatas"][0] if results.get("metadatas") else {}
                self._doc_cache[node_id] = (doc_text, metadata)
                return doc_text, metadata

            # If no parent chunks found, return empty but with metadata from first chunk
            # This allows metadata display even when full text isn't available
            try:
                meta_results = self._collection.get(
                    where={"doc_id": base}, include=["metadatas"], limit=1
                )
                if meta_results and meta_results.get("metadatas"):
                    metadata = meta_results["metadatas"][0]
                    self._doc_cache[node_id] = ("", metadata)
                    return "", metadata
            except Exception:
                pass

            return "", {}

        except Exception as e:
            print(f"Error retrieving document for {node_id}: {e}")
            return "", {}

    def close(self) -> None:
        """Close thread-local database connection."""
        if hasattr(self._thread_local, "conn") and self._thread_local.conn:
            self._thread_local.conn.close()
            self._thread_local.conn = None
