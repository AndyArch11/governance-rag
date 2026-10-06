"""Registry for thesis graph locations and applied cultural lens versions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.ingest.ingest_utils import compute_file_hash
from scripts.thesis_graph.thesis_evidence_graph import (
    ThesisEvidenceGraph,
    build_thesis_evidence_graph,
    get_thesis_graph_path,
)


class ThesisRegistry:
    """Persist thesis identity and pointers to per-thesis graph databases.

    Attributes:
        database_path: The path to the SQLite database file.
    """

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS thesis_registry (
                    thesis_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors_json TEXT NOT NULL DEFAULT '[]',
                    source_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    citation_doc_node_id TEXT,
                    graph_path TEXT NOT NULL,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    built_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    document_role TEXT NOT NULL DEFAULT 'main',
                    cultural_lens_id TEXT,
                    cultural_lens_version TEXT,
                    cultural_lens_status TEXT,
                    confirmed_research_inquiries_json TEXT
                );
                """)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(thesis_registry)")}
            if "confirmed_research_inquiries_json" not in columns:
                connection.execute(
                    "ALTER TABLE thesis_registry ADD COLUMN confirmed_research_inquiries_json TEXT"
                )

    def register_thesis(
        self,
        *,
        thesis_id: str,
        title: str,
        authors: list[str],
        source_path: Path,
        file_hash: str,
        graph_path: Path,
        citation_doc_node_id: str | None = None,
        schema_version: int = 1,
        status: str = "ready",
        document_role: str = "main",
        cultural_lens_id: str | None = None,
        cultural_lens_version: str | None = None,
        cultural_lens_status: str | None = None,
    ) -> None:
        """Insert or update one thesis registry record.

        Args:
            thesis_id: The ID of the thesis.
            title: The title of the thesis.
            authors: A list of authors of the thesis.
            source_path: The path to the source document of the thesis.
            file_hash: The hash of the source file.
            graph_path: The path to the thesis graph.
            citation_doc_node_id: Optional ID of the citation document node.
            schema_version: The schema version of the registry record.
            status: The status of the thesis record.
            document_role: The role of the document in the thesis.
            cultural_lens_id: Optional ID of the associated cultural lens.
            cultural_lens_version: Optional version of the associated cultural lens.
            cultural_lens_status: Optional status of the associated cultural lens.

        Returns:
            None

        Raises:
            sqlite3.DatabaseError: If there is an error executing the SQL statement.
        """
        built_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO thesis_registry (
                    thesis_id, title, authors_json, source_path, file_hash,
                    citation_doc_node_id, graph_path, schema_version, built_at,
                    status, document_role, cultural_lens_id, cultural_lens_version,
                    cultural_lens_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thesis_id) DO UPDATE SET
                    title = excluded.title,
                    authors_json = excluded.authors_json,
                    source_path = excluded.source_path,
                    file_hash = excluded.file_hash,
                    citation_doc_node_id = excluded.citation_doc_node_id,
                    graph_path = excluded.graph_path,
                    schema_version = excluded.schema_version,
                    built_at = excluded.built_at,
                    status = excluded.status,
                    document_role = excluded.document_role,
                    cultural_lens_id = excluded.cultural_lens_id,
                    cultural_lens_version = excluded.cultural_lens_version,
                    cultural_lens_status = excluded.cultural_lens_status
                """,
                (
                    thesis_id,
                    title,
                    json.dumps(authors),
                    str(source_path),
                    file_hash,
                    citation_doc_node_id,
                    str(graph_path),
                    schema_version,
                    built_at,
                    status,
                    document_role,
                    cultural_lens_id,
                    cultural_lens_version,
                    cultural_lens_status,
                ),
            )

    def get_thesis(self, thesis_id: str) -> dict[str, Any] | None:
        """Return one thesis record, decoding its authors list.

        Args:
            thesis_id: The ID of the thesis to retrieve.

        Returns:
            A dictionary representing the thesis record, or None if not found.
        """
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM thesis_registry WHERE thesis_id = ?", (thesis_id,)
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["authors"] = json.loads(record.pop("authors_json"))
        record["confirmed_research_inquiries"] = self._decode_confirmed_inquiries(
            record.pop("confirmed_research_inquiries_json", None)
        )
        return record

    def list_theses(self) -> list[dict[str, Any]]:
        """Return all thesis records ordered by title and ID."""
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM thesis_registry ORDER BY title, thesis_id"
            ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["authors"] = json.loads(record.pop("authors_json"))
            record["confirmed_research_inquiries"] = self._decode_confirmed_inquiries(
                record.pop("confirmed_research_inquiries_json", None)
            )
            records.append(record)
        return records

    @staticmethod
    def _decode_confirmed_inquiries(value: str | None) -> list[dict[str, str]] | None:
        """Decode reviewer-confirmed inquiry records, treating malformed legacy data as absent.

        Args:
            value: JSON-encoded string of confirmed inquiries.

        Returns:
            A list of confirmed inquiry dictionaries, or None if the data is absent or malformed.
        """
        if not value:
            return None
        try:
            records = json.loads(value)
        except json.JSONDecodeError:
            return None
        if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
            return None
        return records

    def set_confirmed_research_inquiries(
        self,
        thesis_id: str,
        inquiries: list[dict[str, str]],
    ) -> bool:
        """Persist a reviewer-confirmed inquiry list without changing thesis identity fields.

        Args:
            thesis_id: The ID of the thesis to update.
            inquiries: A list of confirmed inquiry dictionaries.

        Returns:
            True if the update was successful (exactly one row affected), False otherwise.

        Raises:
            ValueError: If any inquiry has missing or duplicate IDs, or empty text.
        """
        normalised: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for inquiry in inquiries:
            inquiry_id = str(inquiry.get("id", "")).strip()
            text = " ".join(str(inquiry.get("text", "")).split())
            parent_id = str(inquiry.get("parent_id", "")).strip()
            inquiry_type = str(inquiry.get("type", "research_question")).strip()
            if not inquiry_id or not text or inquiry_id in seen_ids:
                raise ValueError("Confirmed inquiries require unique IDs and non-empty text")
            seen_ids.add(inquiry_id)
            normalised.append(
                {
                    "id": inquiry_id,
                    "parent_id": parent_id,
                    "type": inquiry_type or "research_question",
                    "text": text,
                }
            )

        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute(
                "UPDATE thesis_registry SET confirmed_research_inquiries_json = ? WHERE thesis_id = ?",
                (json.dumps(normalised, ensure_ascii=False), thesis_id),
            )
            return cursor.rowcount == 1


def build_and_register_thesis_graph(
    collection: Any,
    *,
    thesis_id: str,
    title: str,
    authors: list[str],
    source_path: Path,
    graphs_dir: Path,
    registry_path: Path,
    citation_doc_node_id: str | None = None,
    cultural_lens_profile: dict[str, Any] | None = None,
    figures: list[dict[str, Any]] | None = None,
) -> ThesisEvidenceGraph:
    """Build a thesis graph and register its source and database locations.

    Args:
        collection: The collection containing the thesis documents.
        thesis_id: The ID of the thesis.
        title: The title of the thesis.
        authors: A list of authors of the thesis.
        source_path: The path to the source document of the thesis.
        graphs_dir: The directory where thesis graphs are stored.
        registry_path: The path to the thesis registry database.
        citation_doc_node_id: Optional ID of the citation document node.
        cultural_lens_profile: Optional cultural lens profile associated with the thesis.
        figures: Optional list of figures associated with the thesis.

    Returns:
        An instance of ThesisEvidenceGraph representing the built thesis graph.
    """
    graph_path = get_thesis_graph_path(graphs_dir, thesis_id)
    graph_options: dict[str, Any] = (
        {"cultural_lens_profile": cultural_lens_profile}
        if cultural_lens_profile is not None
        else {}
    )
    graph_options["citation_graph_path"] = graphs_dir.parent / "academic_citation_graph.db"
    if figures is not None:
        graph_options["figures"] = figures
    graph = build_thesis_evidence_graph(collection, thesis_id, graph_path, **graph_options)
    registry = ThesisRegistry(registry_path)
    registry.register_thesis(
        thesis_id=thesis_id,
        title=title,
        authors=authors,
        source_path=source_path,
        file_hash=compute_file_hash(str(source_path)),
        citation_doc_node_id=citation_doc_node_id,
        graph_path=graph_path,
        status="ready",
        cultural_lens_id=(cultural_lens_profile or {}).get("profile_id"),
        cultural_lens_version=(cultural_lens_profile or {}).get("version"),
        cultural_lens_status=(cultural_lens_profile or {}).get("status"),
    )
    return graph
