"""Tests for the thesis evidence graph command-line entrypoint."""

import importlib
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

cli = importlib.import_module("scripts.thesis_graph.build_thesis_evidence_graph")


class FakeVectorClient:
    """Minimal vector client used to verify CLI wiring."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.collection_name: str | None = None

    def get_collection(self, collection_name: str) -> object:
        """Return a distinct collection marker for the graph builder."""
        self.collection_name = collection_name
        return {"collection": collection_name}


def test_main_uses_default_graph_path(monkeypatch, tmp_path, capsys) -> None:
    """The CLI builds a graph using the configured thesis graph directory."""
    default_output = tmp_path / "graphs" / "thesis.sqlite"
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "parse_args", lambda: Namespace(thesis_id="thesis-id", output=None))
    monkeypatch.setattr(
        cli,
        "RAGConfig",
        lambda: SimpleNamespace(
            rag_data_path=tmp_path / "rag-data",
            chunk_collection_name="thesis_chunks",
            thesis_graphs_dir=tmp_path / "graphs",
        ),
    )
    monkeypatch.setattr(cli, "get_vector_client", lambda prefer: (FakeVectorClient, False))
    monkeypatch.setattr(cli, "get_default_vector_path", lambda path, using_sqlite: path / "vectors")
    monkeypatch.setattr(cli, "get_thesis_graph_path", lambda directory, thesis_id: default_output)

    def fake_build(collection, thesis_id, output, citation_graph_path):
        captured["citation_graph_path"] = citation_graph_path
        return SimpleNamespace(output_path=output, node_count=4, edge_count=3)

    monkeypatch.setattr(
        cli,
        "build_thesis_evidence_graph",
        fake_build,
    )

    assert cli.main() == 0
    assert captured["citation_graph_path"] == tmp_path / "rag-data" / "academic_citation_graph.db"
    assert "Thesis evidence graph built" in capsys.readouterr().out


def test_main_respects_explicit_output_path(monkeypatch, tmp_path) -> None:
    """An explicit output path is supplied directly to the graph builder."""
    explicit_output = tmp_path / "custom.sqlite"
    captured: dict[str, object] = {}

    class Client:
        def __init__(self, path: Path) -> None:
            captured["vector_path"] = path

        def get_collection(self, collection_name: str) -> object:
            captured["collection_name"] = collection_name
            return "collection"

    monkeypatch.setattr(
        cli, "parse_args", lambda: Namespace(thesis_id="thesis-id", output=explicit_output)
    )
    monkeypatch.setattr(
        cli,
        "RAGConfig",
        lambda: SimpleNamespace(
            rag_data_path=tmp_path,
            chunk_collection_name="thesis_chunks",
            thesis_graphs_dir=tmp_path / "ignored",
        ),
    )
    monkeypatch.setattr(cli, "get_vector_client", lambda prefer: (Client, True))
    monkeypatch.setattr(cli, "get_default_vector_path", lambda path, using_sqlite: path / "sqlite")
    monkeypatch.setattr(
        cli,
        "build_thesis_evidence_graph",
        lambda collection, thesis_id, output, citation_graph_path: captured.update(
            {
                "collection": collection,
                "thesis_id": thesis_id,
                "output": output,
                "citation_graph_path": citation_graph_path,
            }
        )
        or SimpleNamespace(output_path=output, node_count=2, edge_count=1),
    )

    assert cli.main() == 0
    assert captured["collection_name"] == "thesis_chunks"
    assert captured["collection"] == "collection"
    assert captured["thesis_id"] == "thesis-id"
    assert captured["output"] == explicit_output
    assert captured["citation_graph_path"] == tmp_path / "academic_citation_graph.db"
