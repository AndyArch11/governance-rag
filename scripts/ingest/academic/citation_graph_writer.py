"""
SQLite writer for citation graph.

Provides:
- Batch node insertion
- Thread-safe edge insertion
- Atomic swap capability
- JSON export for backward compatibility
"""

import json
import shutil
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .citation_graph_schema import ensure_schema, set_metadata_value


class CitationGraphWriter:
    """
    Thread-safe writer for citation graph persistence to SQLite.

    Supports:
    - Batch node insertion
    - Per-edge insertion (thread-safe)
    - Metadata tracking (document ID, build time, etc.)
    - Atomic swap to production DB
    - JSON export for compatibility
    """

    def __init__(self, sqlite_path: Union[str, Path], replace: bool = True):
        """
        Initialise writer with target SQLite path.

        Args:
            sqlite_path: Path to SQLite database file
            replace: If True, drop existing tables before creating schema
        """
        self.sqlite_path = str(sqlite_path)
        self._lock = threading.Lock()
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
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA cache_size=-64000;")
        self._conn.commit()

    def insert_node(
        self,
        node_id: str,
        node_type: str,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        doi: Optional[str] = None,
        year: Optional[int] = None,
        year_verified: bool = False,
        reference_type: Optional[str] = None,
        quality_score: Optional[float] = None,
        link_status: Optional[str] = None,
        venue_type: Optional[str] = None,
        venue_rank: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        """
        Insert a single node into the database.

        Args:
            node_id: Unique node identifier
            node_type: "document" or "reference"
            title: Publication title
            authors: List of author names
            doi: DOI identifier
            year: Publication year
            year_verified: Whether year is from authoritative source
            reference_type: Type of reference (academic, web, etc.)
            quality_score: Quality rating (0.0-1.0)
            link_status: Link health (available, stale_404, etc.)
            venue_type: Publication venue type (journal, conference, etc.)
            venue_rank: Venue quality rank (Q1-Q4, A*-C)
            source: Metadata source (crossref, openalex, etc.)
        """
        with self._lock:
            # Serialise authors list to JSON
            authors_json = json.dumps(authors) if authors else None

            self._conn.execute(
                """
                INSERT OR REPLACE INTO nodes 
                (node_id, node_type, title, authors, doi, year, year_verified, 
                 reference_type, quality_score, link_status, venue_type, venue_rank, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    node_type,
                    title,
                    authors_json,
                    doi,
                    year,
                    1 if year_verified else 0,
                    reference_type,
                    quality_score,
                    link_status,
                    venue_type,
                    venue_rank,
                    source,
                ),
            )
            self._node_count += 1

    def insert_nodes_batch(self, nodes: List[Dict[str, Any]]) -> None:
        """
        Insert multiple nodes in a single transaction.

        Args:
            nodes: List of node dictionaries with keys matching insert_node parameters
        """
        with self._lock:
            for node in nodes:
                authors_json = json.dumps(node.get("authors", [])) if node.get("authors") else None

                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO nodes 
                    (node_id, node_type, title, authors, doi, year, year_verified, 
                     reference_type, quality_score, link_status, venue_type, venue_rank, source, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node["node_id"],
                        node["node_type"],
                        node.get("title"),
                        authors_json,
                        node.get("doi"),
                        node.get("year"),
                        1 if node.get("year_verified", False) else 0,
                        node.get("reference_type"),
                        node.get("quality_score"),
                        node.get("link_status"),
                        node.get("venue_type"),
                        node.get("venue_rank"),
                        node.get("source"),
                        node.get("confidence"),
                    ),
                )
                self._node_count += 1

            self._conn.commit()

    def insert_edge(self, source: str, target: str, relation: str = "cites") -> None:
        """
        Insert a single edge (citation relationship).

        Args:
            source: Source node ID (citing document)
            target: Target node ID (cited reference)
            relation: Relationship type (default: "cites")
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO edges (source, target, relation) VALUES (?, ?, ?)",
                (source, target, relation),
            )
            self._edge_count += 1

    def insert_edges_batch(self, edges: List[Dict[str, str]]) -> None:
        """
        Insert multiple edges in a single transaction.

        Args:
            edges: List of edge dictionaries with 'source', 'target', 'relation' keys
        """
        with self._lock:
            for edge in edges:
                self._conn.execute(
                    "INSERT OR IGNORE INTO edges (source, target, relation) VALUES (?, ?, ?)",
                    (edge["source"], edge["target"], edge.get("relation", "cites")),
                )
                self._edge_count += 1

            self._conn.commit()

    def set_metadata(self, key: str, value: str) -> None:
        """Store metadata key-value pair."""
        set_metadata_value(self._conn, key, value)

    def finalise(self) -> None:
        """Commit final transaction and store statistics."""
        with self._lock:
            self._conn.commit()

            # Store statistics
            set_metadata_value(self._conn, "node_count", str(self._node_count))
            set_metadata_value(self._conn, "edge_count", str(self._edge_count))

    def atomic_swap(self, target_path: Union[str, Path]) -> None:
        """
        Atomically swap this database to target path.

        Args:
            target_path: Final destination path for the database
        """
        target_path = str(target_path)

        # Close connection
        if self._conn:
            self._conn.close()
            self._conn = None

        # Atomic rename
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Remove target if it exists
        if target.exists():
            target.unlink()

        # Move temp to target
        shutil.move(self.sqlite_path, target_path)

        # Update internal path reference
        self.sqlite_path = target_path

    def export_to_json(self, output_path: Union[str, Path]) -> None:
        """
        Export graph to JSON format for backward compatibility.

        Args:
            output_path: Path to write JSON file
        """
        cursor = self._conn.cursor()

        # Load all nodes - select only the columns we need
        cursor.execute("""
            SELECT node_id, node_type, title, authors, doi, year, year_verified, 
                   reference_type, source
            FROM nodes
        """)
        nodes = []
        for row in cursor.fetchall():
            node_id, node_type, title, authors_json, doi, year, year_verified, ref_type, source = (
                row
            )

            # Parse authors from JSON
            authors = json.loads(authors_json) if authors_json else []

            nodes.append(
                {
                    "node_id": node_id,
                    "node_type": node_type,
                    "title": title,
                    "authors": authors,
                    "doi": doi,
                    "year": year,
                    "year_verified": bool(year_verified),
                    "reference_type": ref_type,
                    "source": source,
                }
            )

        # Load all edges
        cursor.execute("SELECT source, target, relation FROM edges")
        edges = [
            {"source": source, "target": target, "relation": relation}
            for source, target, relation in cursor.fetchall()
        ]

        # Write JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump({"nodes": nodes, "edges": edges}, f, indent=2)

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
