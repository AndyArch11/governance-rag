import networkx as nx
import pytest

from scripts.consistency_graph.advanced_analytics import (
    compute_pagerank_influence,
    compute_betweenness_centrality,
    compute_eigenvector_centrality,
    detect_communities_louvain,
    detect_communities_label_propagation,
    compute_relationship_strength,
    compute_network_topology_metrics,
    compute_node_clustering_coefficients,
    compute_advanced_analytics,
    get_node_influence_rank,
)


def test_pagerank_empty_graph():
    G = nx.Graph()
    assert compute_pagerank_influence(G) == {}


def test_pagerank_simple_graph():
    G = nx.path_graph(4)
    pr = compute_pagerank_influence(G)
    assert set(pr.keys()) == set(G.nodes())
    # Values should be probabilities and sum close to 1
    assert all(0.0 <= v <= 1.0 for v in pr.values())
    assert pytest.approx(sum(pr.values()), rel=1e-2) == 1.0


def test_betweenness_path_graph():
    G = nx.path_graph(5)
    bc = compute_betweenness_centrality(G)
    # Center node should have the highest betweenness
    center = max(bc, key=bc.get)
    assert center in (2,)  # node 2 is center in path_graph(5)


def test_eigenvector_star_graph():
    G = nx.star_graph(4)  # center node 0 connected to 1..4
    eig = compute_eigenvector_centrality(G)
    assert set(eig.keys()) == set(G.nodes())
    # Center should have highest eigenvector centrality
    center = max(eig, key=eig.get)
    assert center == 0


def test_communities_louvain_two_cliques():
    G = nx.Graph()
    # Clique 1: 0-1-2
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    # Clique 2: 3-4-5
    G.add_edges_from([(3, 4), (4, 5), (3, 5)])
    # Bridge: 2-3
    G.add_edge(2, 3)

    comms = detect_communities_louvain(G)
    # Expect at least 2 communities
    assert len(comms) >= 2
    # Nodes from first triangle mostly together
    assert any({0, 1, 2}.issubset(c) or len({0,1,2} & c) >= 2 for c in comms)


def test_communities_label_propagation_returns_groups():
    G = nx.erdos_renyi_graph(20, 0.1, seed=42)
    comms = detect_communities_label_propagation(G)
    assert len(comms) >= 1
    # All nodes covered
    covered = set().union(*comms)
    assert covered == set(G.nodes())


def test_relationship_strength_triangle():
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
    rel = compute_relationship_strength(G)
    # Expect metrics per edge
    for edge in [("A","B"), ("B","C"), ("A","C")]:
        assert edge in rel or (edge[1], edge[0]) in rel
    # For existing edges, NetworkX jaccard/resource_allocation return only for non-edges.
    # Our implementation defaults missing coefficients to 0.0; combined_strength == weight.
    # Find one edge entry
    any_edge = next(iter(rel))
    metrics = rel[any_edge]
    assert "edge_weight" in metrics
    assert "jaccard_coefficient" in metrics
    assert "resource_allocation" in metrics
    assert "preferential_attachment" in metrics
    assert "combined_strength" in metrics
    assert metrics["jaccard_coefficient"] == 0.0
    assert metrics["combined_strength"] == pytest.approx(metrics["edge_weight"], rel=1e-6)


def test_network_topology_metrics_path4():
    G = nx.path_graph(4)
    m = compute_network_topology_metrics(G)
    assert m["num_nodes"] == 4
    assert m["num_edges"] == 3
    assert m["density"] == pytest.approx(0.5, rel=1e-3)
    assert m["avg_clustering"] == 0.0
    assert m["num_components"] == 1
    assert m["diameter"] == 3
    assert m["avg_path_length"] == pytest.approx(1.67, rel=1e-2)


def test_node_clustering_coefficients_triangle():
    G = nx.Graph()
    G.add_edges_from([(1,2), (2,3), (1,3)])
    coeffs = compute_node_clustering_coefficients(G)
    assert coeffs[1] == coeffs[2] == coeffs[3] == 1.0


def test_compute_advanced_analytics_star_graph():
    G = nx.star_graph(5)  # center 0
    analytics = compute_advanced_analytics(G)
    # Presence of all sections
    assert "influence_scores" in analytics
    assert "communities" in analytics
    assert "relationship_strength" in analytics
    assert "topology" in analytics
    assert "node_clustering" in analytics
    # Top influencer by PageRank should be center
    top_pr = analytics.get("top_influencers", {}).get("by_pagerank", [])
    if top_pr:
        assert top_pr[0][0] == 0


def test_get_node_influence_rank():
    G = nx.path_graph(5)
    analytics = compute_advanced_analytics(G)
    # Check known node ranks
    node2 = get_node_influence_rank(analytics, 2)
    assert node2["pagerank"]["total_nodes"] == 5
    assert node2["betweenness"]["total_nodes"] == 5
    assert node2["eigenvector"]["total_nodes"] == 5
    # Node 2 should be near top on betweenness in a path of 5
    assert node2["betweenness"]["rank"] in (1, 2)
