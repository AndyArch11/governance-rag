"""Command-line builder for persisted thesis evidence graphs."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.rag.rag_config import RAGConfig
from scripts.thesis_graph.thesis_evidence_graph import (
    build_thesis_evidence_graph,
    get_thesis_graph_path,
)
from scripts.utils.db_factory import get_default_vector_path, get_vector_client


def parse_args() -> argparse.Namespace:
    """Parse thesis evidence graph build arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build a deterministic thesis evidence graph from chapter-aware chunks."
    )
    parser.add_argument("thesis_id", help="Canonical thesis ID stored in ChromaDB.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="SQLite output path (default: RAG_THESIS_GRAPHS_DIR/<thesis_id>.sqlite).",
    )
    return parser.parse_args()


def main() -> int:
    """Build and report a thesis evidence graph.

    Returns:
        int: Exit code (0 for success).
    """
    args = parse_args()
    config = RAGConfig()
    client_class, using_sqlite = get_vector_client(prefer="chroma")
    client = client_class(path=get_default_vector_path(Path(config.rag_data_path), using_sqlite))
    collection = client.get_collection(config.chunk_collection_name)
    output_path = args.output or get_thesis_graph_path(
        Path(config.thesis_graphs_dir), args.thesis_id
    )

    graph = build_thesis_evidence_graph(
        collection,
        args.thesis_id,
        output_path,
        citation_graph_path=Path(config.rag_data_path) / "academic_citation_graph.db",
    )
    print(f"Thesis evidence graph built: {graph.output_path}")
    print(f"Nodes: {graph.node_count}, Edges: {graph.edge_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
