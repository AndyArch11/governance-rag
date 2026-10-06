"""Tests for thesis identity and graph registry records."""

import sqlite3
from pathlib import Path

from scripts.thesis_graph.thesis_evidence_graph import ThesisEvidenceGraph
from scripts.thesis_graph.thesis_registry import (
    ThesisRegistry,
    build_and_register_thesis_graph,
)


def test_registry_supports_multiple_theses_and_updates_by_id(tmp_path: Path) -> None:
    registry = ThesisRegistry(tmp_path / "thesis_graphs" / "registry.sqlite")
    graph_path = tmp_path / "thesis_graphs" / "thesis_one.sqlite"

    registry.register_thesis(
        thesis_id="thesis-one",
        title="First Thesis",
        authors=["A. Author"],
        source_path=tmp_path / "first.pdf",
        file_hash="first-hash",
        graph_path=graph_path,
        citation_doc_node_id="thesis-one",
    )
    registry.register_thesis(
        thesis_id="thesis-two",
        title="Second Thesis",
        authors=["B. Author"],
        source_path=tmp_path / "second.pdf",
        file_hash="second-hash",
        graph_path=tmp_path / "thesis_graphs" / "thesis-two.sqlite",
    )
    registry.register_thesis(
        thesis_id="thesis-one",
        title="First Thesis Revised",
        authors=["A. Author", "C. Author"],
        source_path=tmp_path / "first.pdf",
        file_hash="revised-hash",
        graph_path=graph_path,
        citation_doc_node_id="thesis-one",
        cultural_lens_id="lens-draft",
        cultural_lens_version="0.1.0",
        cultural_lens_status="draft",
    )

    first = registry.get_thesis("thesis-one")
    assert first is not None
    assert first["title"] == "First Thesis Revised"
    assert first["authors"] == ["A. Author", "C. Author"]
    assert first["file_hash"] == "revised-hash"
    assert first["citation_doc_node_id"] == "thesis-one"
    assert first["cultural_lens_status"] == "draft"
    assert {record["thesis_id"] for record in registry.list_theses()} == {
        "thesis-one",
        "thesis-two",
    }
    assert registry.get_thesis("missing") is None


def test_build_and_register_thesis_graph_records_canonical_mapping(
    monkeypatch, tmp_path: Path
) -> None:
    source_path = tmp_path / "thesis.pdf"
    source_path.write_bytes(b"thesis fixture")
    graph_path = tmp_path / "thesis_graphs" / "thesis_one.sqlite"
    expected_graph = ThesisEvidenceGraph("thesis-one", graph_path, 3, 2)
    figures = [{"figure_number": 1, "caption": "Figure caption"}]

    def fake_build(
        collection,
        thesis_id: str,
        output_path: Path,
        figures: list[dict[str, object]] | None = None,
        citation_graph_path: Path | None = None,
    ) -> ThesisEvidenceGraph:
        assert collection == "chunks"
        assert thesis_id == "thesis-one"
        assert output_path == graph_path
        assert figures == [{"figure_number": 1, "caption": "Figure caption"}]
        assert citation_graph_path == graph_path.parent.parent / "academic_citation_graph.db"
        return expected_graph

    monkeypatch.setattr(
        "scripts.thesis_graph.thesis_registry.build_thesis_evidence_graph", fake_build
    )

    result = build_and_register_thesis_graph(
        "chunks",
        thesis_id="thesis-one",
        title="First Thesis",
        authors=["A. Author"],
        source_path=source_path,
        graphs_dir=graph_path.parent,
        registry_path=graph_path.parent / "registry.sqlite",
        citation_doc_node_id="thesis-one",
        figures=figures,
    )

    record = ThesisRegistry(graph_path.parent / "registry.sqlite").get_thesis("thesis-one")
    assert result == expected_graph
    assert record is not None
    assert record["citation_doc_node_id"] == "thesis-one"
    assert record["graph_path"] == str(graph_path)


def test_confirmed_inquiries_round_trip_and_survive_reregistration(tmp_path: Path) -> None:
    registry = ThesisRegistry(tmp_path / "registry.sqlite")
    source_path = tmp_path / "thesis.pdf"
    source_path.write_bytes(b"thesis")
    registry.register_thesis(
        thesis_id="thesis-one",
        title="First Thesis",
        authors=[],
        source_path=source_path,
        file_hash="first-hash",
        graph_path=tmp_path / "thesis.sqlite",
    )

    confirmed = [
        {"id": "RQ1", "parent_id": "", "type": "research_question", "text": "Question one?"},
        {"id": "RQ1a", "parent_id": "RQ1", "type": "sub_question", "text": "Sub-question?"},
    ]
    assert registry.set_confirmed_research_inquiries("thesis-one", confirmed)

    registry.register_thesis(
        thesis_id="thesis-one",
        title="First Thesis revised",
        authors=[],
        source_path=source_path,
        file_hash="second-hash",
        graph_path=tmp_path / "thesis.sqlite",
    )

    assert registry.get_thesis("thesis-one")["confirmed_research_inquiries"] == confirmed
    assert not registry.set_confirmed_research_inquiries("missing", confirmed)


def test_registry_adds_confirmed_inquiry_column_to_existing_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy_registry.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("""
            CREATE TABLE thesis_registry (
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
                cultural_lens_status TEXT
            )
            """)

    ThesisRegistry(database_path)
    with sqlite3.connect(database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(thesis_registry)")}
    assert "confirmed_research_inquiries_json" in columns
