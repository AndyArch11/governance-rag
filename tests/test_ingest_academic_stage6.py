"""Tests for academic ingestion stage 6 (citation graph)."""

import json
from pathlib import Path

from scripts.ingest.academic.graph import CitationGraph, add_references_to_graph, reference_id_from_metadata


def test_reference_id_from_metadata_prefers_doi():
    ref = {"doi": "10.1000/182", "title": "Some Title"}
    assert reference_id_from_metadata(ref) == "doi:10.1000/182"


def test_reference_id_from_metadata_fallback_hash():
    ref = {"title": "Some Title"}
    ref_id = reference_id_from_metadata(ref)
    assert ref_id.startswith("ref:")


def test_add_references_to_graph_creates_edges():
    graph = CitationGraph()
    refs = [
        {"title": "Paper A", "doi": "10.1/abc", "reference_type": "academic"},
        {"title": "Paper B", "reference_type": "online"},
    ]
    add_references_to_graph(graph, "doc1", refs)
    assert "doc1" in graph.nodes
    assert len(graph.edges) == 2

