"""Tests for graph-enhanced retrieval module."""

import json
from pathlib import Path

import networkx as nx
import pytest

from scripts.rag.graph_retrieval import GraphEnhancedRetriever, get_graph_retriever


@pytest.fixture
def sample_graph_data():
    """Create sample graph data."""
    return {
        "nodes": {
            "chunk_1": {
                "chunk_id": "chunk_1",
                "text": "Authentication service",
                "technical_entities": ["auth_service", "login_api"],
                "risk_cluster": "cluster_1",
                "topic_cluster": "auth_topic",
            },
            "chunk_2": {
                "chunk_id": "chunk_2",
                "text": "Database configuration",
                "technical_entities": ["database", "config_db"],
                "risk_cluster": "cluster_1",
                "topic_cluster": "config_topic",
            },
            "chunk_3": {
                "chunk_id": "chunk_3",
                "text": "API gateway",
                "technical_entities": ["api_gateway", "login_api"],
                "risk_cluster": "cluster_2",
                "topic_cluster": "auth_topic",
            },
        },
        "edges": [
            {
                "source": "chunk_1",
                "target": "chunk_2",
                "relationship": "related",
                "severity": 0.7,
            },
            {
                "source": "chunk_1",
                "target": "chunk_3",
                "relationship": "conflict",
                "severity": 0.9,
            },
        ],
    }


@pytest.fixture
def graph_file(tmp_path, sample_graph_data):
    """Create temporary graph JSON file."""
    graph_path = tmp_path / "test_graph.json"
    with open(graph_path, "w") as f:
        json.dump(sample_graph_data, f)
    return graph_path


@pytest.fixture
def retriever(graph_file):
    """Create retriever instance with test graph."""
    return GraphEnhancedRetriever(graph_source=graph_file)


def test_load_graph(retriever):
    """Test graph loading from JSON."""
    assert retriever.graph is not None
    assert len(retriever.graph.nodes) == 3
    assert len(retriever.graph.edges) == 2


def test_entity_mappings(retriever):
    """Test entity-to-chunk and chunk-to-entity mappings."""
    # Check entity_to_chunks mapping
    assert "auth_service" in retriever.entity_to_chunks
    assert "chunk_1" in retriever.entity_to_chunks["auth_service"]

    # login_api appears in chunk_1 and chunk_3
    assert len(retriever.entity_to_chunks["login_api"]) == 2

    # Check chunk_to_entities mapping
    assert "auth_service" in retriever.chunk_to_entities["chunk_1"]
    assert "login_api" in retriever.chunk_to_entities["chunk_1"]


def test_expand_with_neighbours(retriever):
    """Test expanding chunks with graph neighbours."""
    # Expand chunk_1 with 1-hop neighbours
    expanded = retriever.expand_with_neighbours(chunk_ids=["chunk_1"], max_hops=1, max_neighbours=5)

    # Should include original + neighbours
    assert "chunk_1" in expanded
    assert "chunk_2" in expanded  # Connected by 'related'
    assert "chunk_3" in expanded  # Connected by 'conflict'


def test_expand_with_neighbours_limited(retriever):
    """Test neighbour expansion with max_neighbours limit."""
    # Limit to 1 neighbour
    expanded = retriever.expand_with_neighbours(chunk_ids=["chunk_1"], max_hops=1, max_neighbours=1)
    # Should include original + top 1 neighbour
    assert "chunk_1" in expanded
    assert len(expanded) == 2  # chunk_1 + 1 neighbour


def test_expand_with_entities(retriever):
    """Test finding chunks by entity."""
    # Find chunks containing login_api
    chunks = retriever.expand_with_entities(entities=["login_api"], max_chunks_per_entity=5)

    assert "chunk_1" in chunks
    assert "chunk_3" in chunks


def test_find_conflicting_chunks(retriever):
    """Test finding conflicting chunks."""
    conflicts = retriever.find_conflicting_chunks(["chunk_1"])

    # Should find conflict with chunk_3
    assert len(conflicts) > 0
    conflict = conflicts[0]
    assert conflict[0] == "chunk_1"
    assert conflict[1] == "chunk_3"
    assert conflict[2] == "conflict"


def test_get_cluster_context(retriever):
    """Test cluster membership retrieval."""
    clusters = retriever.get_cluster_context(["chunk_1", "chunk_2"])

    # Both in cluster_1
    assert "cluster_1" in clusters
    assert "chunk_1" in clusters["cluster_1"]
    assert "chunk_2" in clusters["cluster_1"]


def test_bfs_neighbours(retriever):
    """Test BFS neighbour discovery."""
    neighbours = retriever._bfs_neighbours("chunk_1", max_hops=1)

    assert "chunk_2" in neighbours
    assert "chunk_3" in neighbours
    # chunk_1 itself should not be in neighbours
    assert "chunk_1" not in neighbours


def test_rank_neighbours(retriever):
    """Test neighbour ranking by edge strength."""
    neighbours = ["chunk_2", "chunk_3"]
    ranked = retriever._rank_neighbours("chunk_1", neighbours)

    # chunk_3 has higher severity (0.9) than chunk_2 (0.7)
    assert ranked[0] == "chunk_3"
    assert ranked[1] == "chunk_2"


def test_no_graph_graceful_failure():
    """Test graceful handling when graph file doesn't exist."""
    retriever = GraphEnhancedRetriever(graph_source=Path("/nonexistent/graph.json"))

    assert retriever.graph is None

    # Should return original chunks without expansion
    expanded = retriever.expand_with_neighbours(["chunk_1"], max_hops=1)
    assert expanded == ["chunk_1"]


def test_global_retriever_instance(graph_file):
    """Test global singleton retriever."""
    retriever1 = get_graph_retriever(graph_source=graph_file)
    retriever2 = get_graph_retriever()

    # Should return same instance
    assert retriever1 is retriever2


def test_multi_hop_expansion(retriever):
    """Test multi-hop BFS expansion."""
    # With 2 hops, should reach all connected nodes
    neighbours = retriever._bfs_neighbours("chunk_1", max_hops=2)

    # Direct neighbours + their neighbours
    assert "chunk_2" in neighbours
    assert "chunk_3" in neighbours