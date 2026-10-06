"""Tests for composing thesis and citation graph stores."""

import sqlite3
from pathlib import Path

from scripts.ingest.academic.citation_graph_schema import ensure_schema
from scripts.thesis_graph.thesis_registry import ThesisRegistry
from scripts.thesis_graph.unified_graph import load_academic_graph
from scripts.ui.academic.citation_graph_viz import CitationGraphViz


def _create_thesis_graph(path: Path, thesis_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                thesis_id TEXT NOT NULL,
                label TEXT NOT NULL,
                source_start INTEGER,
                source_end INTEGER,
                sequence_number INTEGER,
                attributes_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE edges (
                source_node_id TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                relation TEXT NOT NULL
            );
            """)
        connection.executemany(
            "INSERT INTO nodes (node_id, node_type, thesis_id, label) VALUES (?, ?, ?, ?)",
            [
                ("thesis-root", "thesis", thesis_id, "Study of Healing"),
                ("chapter-1", "chapter", thesis_id, "Chapter 1: Introduction"),
            ],
        )
        connection.execute(
            "INSERT INTO edges VALUES (?, ?, ?)",
            ("thesis-root", "chapter-1", "contains"),
        )


def _create_citation_graph(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        ensure_schema(connection)
        connection.executemany(
            "INSERT INTO nodes (node_id, node_type, title, source, reference_type) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                ("document-one", "document", "Study of Healing", "document", None),
                ("reference-one", "reference", "Healing Research", "crossref", "academic"),
                ("document-other", "document", "Other Study", "document", None),
                ("reference-other", "reference", "Other Research", "crossref", "academic"),
            ],
        )
        connection.executemany(
            "INSERT INTO edges (source, target, relation) VALUES (?, ?, ?)",
            [
                ("document-one", "reference-one", "cites"),
                ("document-other", "reference-other", "cites"),
            ],
        )


def _register_thesis(tmp_path: Path) -> tuple[Path, Path, str]:
    thesis_id = "thesis-one"
    thesis_graph_path = tmp_path / "thesis_graphs" / "thesis_one.sqlite"
    citation_graph_path = tmp_path / "academic_citation_graph.db"
    registry_path = tmp_path / "thesis_graphs" / "registry.sqlite"
    _create_thesis_graph(thesis_graph_path, thesis_id)
    _create_citation_graph(citation_graph_path)
    registry = ThesisRegistry(registry_path)
    registry.register_thesis(
        thesis_id=thesis_id,
        title="Study of Healing",
        authors=["A. Author"],
        source_path=tmp_path / "thesis.pdf",
        file_hash="fixture-hash",
        graph_path=thesis_graph_path,
        citation_doc_node_id="document-one",
    )
    return registry_path, citation_graph_path, thesis_id


def test_unified_loader_supports_thesis_only_mode(tmp_path: Path) -> None:
    registry_path, citation_graph_path, thesis_id = _register_thesis(tmp_path)

    graph = load_academic_graph(registry_path, citation_graph_path, thesis_id, mode="thesis")

    assert set(graph.nodes) == {"thesis-root", "chapter-1"}
    assert graph.edges["thesis-root", "chapter-1"]["relation"] == "contains"
    assert graph.graph["mode"] == "thesis"


def test_unified_loader_scopes_references_only_to_selected_thesis(tmp_path: Path) -> None:
    registry_path, citation_graph_path, thesis_id = _register_thesis(tmp_path)

    graph = load_academic_graph(registry_path, citation_graph_path, thesis_id, mode="references")

    assert set(graph.nodes) == {"document-one", "reference-one"}
    assert graph.edges["document-one", "reference-one"]["relation"] == "cites"
    assert "document-other" not in graph


def test_unified_loader_applies_reference_filters(tmp_path: Path) -> None:
    registry_path, citation_graph_path, thesis_id = _register_thesis(tmp_path)

    graph = load_academic_graph(
        registry_path,
        citation_graph_path,
        thesis_id,
        mode="references",
        reference_types=["preprint"],
    )

    assert set(graph.nodes) == {"document-one"}


def test_unified_loader_bridges_thesis_root_to_its_references(tmp_path: Path) -> None:
    registry_path, citation_graph_path, thesis_id = _register_thesis(tmp_path)

    graph = load_academic_graph(
        registry_path, citation_graph_path, thesis_id, mode="thesis_and_references"
    )

    assert set(graph.nodes) == {"thesis-root", "chapter-1", "reference-one"}
    assert graph.edges["thesis-root", "chapter-1"]["layer"] == "thesis"
    assert graph.edges["thesis-root", "reference-one"]["relation"] == "cites"
    assert graph.edges["thesis-root", "reference-one"]["layer"] == "bridge"
    assert graph.nodes["thesis-root"]["citation_doc_node_id"] == "document-one"


def test_combined_graph_uses_distinct_thesis_node_type_encodings(tmp_path: Path) -> None:
    registry_path, citation_graph_path, thesis_id = _register_thesis(tmp_path)
    graph = load_academic_graph(
        registry_path, citation_graph_path, thesis_id, mode="thesis_and_references"
    )
    visualiser = CitationGraphViz(db_path=citation_graph_path)
    visualiser._graph = graph
    visualiser._primary_docs = {"thesis-root"}

    root_attributes = visualiser.compute_node_attributes("thesis-root")
    chapter_attributes = visualiser.compute_node_attributes("chapter-1")
    reference_attributes = visualiser.compute_node_attributes("reference-one")

    assert root_attributes.shape == "diamond"
    assert chapter_attributes.shape == "square"
    assert chapter_attributes.color != reference_attributes.color


def test_unified_loader_rejects_unknown_mode(tmp_path: Path) -> None:
    registry_path, citation_graph_path, thesis_id = _register_thesis(tmp_path)

    try:
        load_academic_graph(registry_path, citation_graph_path, thesis_id, mode="all")
    except ValueError as error:
        assert "mode" in str(error)
    else:
        raise AssertionError("Unknown graph mode should be rejected")
