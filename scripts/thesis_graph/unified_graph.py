"""Compose per-thesis evidence graphs with the shared citation graph."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Literal

import networkx as nx

from scripts.thesis_graph.thesis_registry import ThesisRegistry

AcademicGraphMode = Literal["thesis", "thesis_and_references", "references"]
_VALID_MODES = {"thesis", "thesis_and_references", "references"}


def load_academic_graph(
    registry_path: Path,
    citation_graph_path: Path,
    thesis_id: str,
    *,
    mode: AcademicGraphMode,
    link_statuses: list[str] | None = None,
    reference_types: list[str] | None = None,
    venue_types: list[str] | None = None,
    venue_ranks: list[str] | None = None,
    sources: list[str] | None = None,
) -> nx.DiGraph:
    """Load a thesis, its citations, or a composed graph for one thesis.

    ``references`` returns the selected citation document and its direct references.
    ``thesis_and_references`` composes the thesis evidence graph and citation edges,
    mapping the citation document onto the thesis root to avoid a duplicate node.

    Args:
        registry_path: Path to the thesis registry.
        citation_graph_path: Path to the shared citation graph.
        thesis_id: ID of the thesis to load.
        mode: Mode of the academic graph to load.
        link_statuses: Filter for citation link statuses.
        reference_types: Filter for reference types.
        venue_types: Filter for venue types.
        venue_ranks: Filter for venue ranks.
        sources: Filter for sources.

    Returns:
        A NetworkX directed graph representing the requested academic graph.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unsupported academic graph mode {mode!r}; expected one of {sorted(_VALID_MODES)}."
        )

    thesis_record = ThesisRegistry(registry_path).get_thesis(thesis_id)
    if thesis_record is None:
        raise KeyError(f"Thesis is not registered: {thesis_id}")

    graph = nx.DiGraph(
        thesis_id=thesis_id,
        mode=mode,
        layers=[],
    )
    thesis_root_id: str | None = None

    if mode in {"thesis", "thesis_and_references"}:
        thesis_graph_path = Path(thesis_record["graph_path"])
        if not thesis_graph_path.exists():
            raise FileNotFoundError(f"Thesis graph not found: {thesis_graph_path}")
        thesis_root_id = _load_thesis_layer(graph, thesis_graph_path, thesis_id)
        graph.graph["layers"].append("thesis")

    if mode in {"references", "thesis_and_references"}:
        citation_doc_node_id = thesis_record.get("citation_doc_node_id") or thesis_id
        _load_citation_layer(
            graph,
            citation_graph_path,
            citation_doc_node_id,
            thesis_root_id=thesis_root_id if mode == "thesis_and_references" else None,
            link_statuses=link_statuses,
            reference_types=reference_types,
            venue_types=venue_types,
            venue_ranks=venue_ranks,
            sources=sources,
        )
        graph.graph["layers"].append("citation" if mode == "references" else "bridge")

    return graph


def _load_thesis_layer(graph: nx.DiGraph, graph_path: Path, thesis_id: str) -> str:
    """Load a thesis graph and return its root node ID .

    Args:
        graph: The NetworkX directed graph to populate.
        graph_path: Path to the thesis graph SQLite database.
        thesis_id: ID of the thesis to load.

    Returns:
        The root node ID of the thesis graph.
    """
    with sqlite3.connect(graph_path) as connection:
        connection.row_factory = sqlite3.Row
        nodes = connection.execute(
            """
            SELECT node_id, node_type, thesis_id, label, source_start, source_end,
                   sequence_number, attributes_json
            FROM nodes
            WHERE thesis_id = ?
            ORDER BY sequence_number, node_id
            """,
            (thesis_id,),
        ).fetchall()
        edges = connection.execute(
            """
            SELECT edges.source_node_id, edges.target_node_id, edges.relation
            FROM edges
            JOIN nodes AS source ON source.node_id = edges.source_node_id
            JOIN nodes AS target ON target.node_id = edges.target_node_id
            WHERE source.thesis_id = ? AND target.thesis_id = ?
            ORDER BY edges.source_node_id, edges.target_node_id, edges.relation
            """,
            (thesis_id, thesis_id),
        ).fetchall()

    root_node_id = None
    for row in nodes:
        attributes = json.loads(row["attributes_json"] or "{}")
        node_attributes: dict[str, Any] = {
            "node_type": row["node_type"],
            "thesis_id": row["thesis_id"],
            "label": row["label"],
            "source_start": row["source_start"],
            "source_end": row["source_end"],
            "sequence_number": row["sequence_number"],
            "layer": "thesis",
        }
        node_attributes.update(attributes)
        graph.add_node(row["node_id"], **node_attributes)
        if row["node_type"] == "thesis":
            root_node_id = row["node_id"]

    for row in edges:
        graph.add_edge(
            row["source_node_id"],
            row["target_node_id"],
            relation=row["relation"],
            layer="thesis",
        )

    if root_node_id is None:
        raise ValueError(f"Thesis graph has no thesis root node for {thesis_id!r}.")
    return root_node_id


def _load_citation_layer(
    graph: nx.DiGraph,
    citation_graph_path: Path,
    citation_doc_node_id: str,
    *,
    thesis_root_id: str | None,
    link_statuses: list[str] | None,
    reference_types: list[str] | None,
    venue_types: list[str] | None,
    venue_ranks: list[str] | None,
    sources: list[str] | None,
) -> None:
    """Load one document's direct references, optionally bridging to thesis root.

    Args:
        graph: The NetworkX directed graph to populate.
        citation_graph_path: Path to the citation graph SQLite database.
        citation_doc_node_id: Node ID of the citation document.
        thesis_root_id: Node ID of the thesis root, if available.
        link_statuses: Filter for citation link statuses.
        reference_types: Filter for reference types.
        venue_types: Filter for venue types.
        venue_ranks: Filter for venue ranks.
        sources: Filter for sources.
    """
    if not citation_graph_path.exists():
        raise FileNotFoundError(f"Citation graph not found: {citation_graph_path}")

    with sqlite3.connect(citation_graph_path) as connection:
        connection.row_factory = sqlite3.Row
        document = connection.execute(
            "SELECT * FROM nodes WHERE node_id = ? AND node_type = 'document'",
            (citation_doc_node_id,),
        ).fetchone()
        references = connection.execute(
            """
            SELECT DISTINCT reference.*
            FROM nodes AS reference
            JOIN edges ON edges.target = reference.node_id
            WHERE edges.source = ? AND reference.node_type = 'reference'
            ORDER BY reference.node_id
            """,
            (citation_doc_node_id,),
        ).fetchall()

    if thesis_root_id is None and document is not None:
        graph.add_node(
            citation_doc_node_id,
            **_citation_node_attributes(document),
        )

    if thesis_root_id is not None and document is not None:
        graph.nodes[thesis_root_id]["citation_doc_node_id"] = citation_doc_node_id
        graph.nodes[thesis_root_id]["citation_document"] = _citation_node_attributes(document)

    for row in references:
        reference_attributes = _citation_node_attributes(row)
        if not _matches_reference_filters(
            reference_attributes,
            link_statuses=link_statuses,
            reference_types=reference_types,
            venue_types=venue_types,
            venue_ranks=venue_ranks,
            sources=sources,
        ):
            continue
        graph.add_node(row["node_id"], **reference_attributes)
        source_node_id = thesis_root_id or citation_doc_node_id
        graph.add_edge(
            source_node_id,
            row["node_id"],
            relation="cites",
            layer="citation" if thesis_root_id is None else "bridge",
        )


def _citation_node_attributes(row: sqlite3.Row) -> dict[str, Any]:
    """Convert one citation database row into visualisation-ready node data.

    Args:
        row: A row from the citation database representing a node.

    Returns:
        A dictionary of node attributes suitable for visualization.
    """
    attributes = dict(row)
    attributes["label"] = attributes.get("title") or attributes["node_id"]
    attributes["layer"] = "citation"
    return attributes


def _matches_reference_filters(
    attributes: dict[str, Any],
    *,
    link_statuses: list[str] | None,
    reference_types: list[str] | None,
    venue_types: list[str] | None,
    venue_ranks: list[str] | None,
    sources: list[str] | None,
) -> bool:
    """Check if a reference node matches the given filter criteria.

    Args:
        attributes: A dictionary of node attributes.
        link_statuses: Filter for citation link statuses.
        reference_types: Filter for reference types.
        venue_types: Filter for venue types.
        venue_ranks: Filter for venue ranks.
        sources: Filter for sources.

    Returns:
        True if the node matches all specified filters, False otherwise.
    """
    filter_pairs = (
        ("link_status", link_statuses),
        ("reference_type", reference_types),
        ("venue_type", venue_types),
        ("venue_rank", venue_ranks),
        ("source", sources),
    )
    return all(
        not selected_values
        or attributes.get(attribute_name) is None
        or attributes.get(attribute_name) in selected_values
        for attribute_name, selected_values in filter_pairs
    )
