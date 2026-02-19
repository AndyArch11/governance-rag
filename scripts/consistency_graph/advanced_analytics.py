"""Advanced Graph Analytics for Consistency Graphs.

Provides sophisticated graph analysis capabilities including:
- Community detection (Louvain, Label Propagation, Greedy Modularity)
- Influence scoring (PageRank, Betweenness Centrality, Eigenvector Centrality)
- Relationship strength analysis (edge weights, clustering coefficients)
- Network topology metrics (diameter, density, connected components)
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx


def compute_pagerank_influence(
    G: nx.Graph, alpha: float = 0.85, max_iter: int = 100
) -> Dict[str, float]:
    """Compute PageRank influence scores for all nodes.

    PageRank measures importance based on the link structure of the graph.
    Nodes with more high-quality connections score higher.

    Args:
        G: NetworkX graph
        alpha: Damping parameter (default: 0.85)
        max_iter: Maximum iterations (default: 100)

    Returns:
        Dict mapping node_id -> PageRank score (0.0-1.0)
    """
    if len(G.nodes()) == 0:
        return {}

    try:
        pagerank = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=1e-6)
        return {node: round(score, 4) for node, score in pagerank.items()}
    except Exception:
        # Fallback for disconnected graphs
        return {node: 1.0 / len(G.nodes()) for node in G.nodes()}


def compute_betweenness_centrality(
    G: nx.Graph, normalised: bool = True, k: Optional[int] = None
) -> Dict[str, float]:
    """Compute betweenness centrality for all nodes.

    Betweenness measures how often a node appears on shortest paths
    between other nodes. High betweenness indicates "bridge" nodes
    that connect different parts of the graph.

    Args:
        G: NetworkX graph
        normalised: Normalise by number of node pairs (default: True)
        k: Use k random samples for approximation (None = exact)

    Returns:
        Dict mapping node_id -> betweenness score
    """
    if len(G.nodes()) == 0:
        return {}

    try:
        betweenness = nx.betweenness_centrality(G, normalized=normalised, k=k)
        return {node: round(score, 4) for node, score in betweenness.items()}
    except Exception:
        return {node: 0.0 for node in G.nodes()}


def compute_eigenvector_centrality(G: nx.Graph, max_iter: int = 100) -> Dict[str, float]:
    """Compute eigenvector centrality for all nodes.

    Eigenvector centrality measures influence based on the importance
    of neighbours. Connections to high-scoring nodes contribute more.

    Args:
        G: NetworkX graph
        max_iter: Maximum iterations (default: 100)

    Returns:
        Dict mapping node_id -> eigenvector centrality score
    """
    if len(G.nodes()) == 0:
        return {}

    try:
        eigen = nx.eigenvector_centrality(G, max_iter=max_iter, tol=1e-6)
        return {node: round(score, 4) for node, score in eigen.items()}
    except Exception:
        # Fallback for disconnected or problematic graphs
        return {node: 0.0 for node in G.nodes()}


def detect_communities_louvain(G: nx.Graph, resolution: float = 1.0) -> List[Set[str]]:
    """Detect communities using Louvain method.

    Louvain algorithm maximises modularity to find densely connected
    communities. Provides hierarchical community structure.

    TODO: NetworkX doesn't have built-in Louvain, consider using python-louvain package for better results.

    Args:
        G: NetworkX graph
        resolution: Resolution parameter for community detection

    Returns:
        List of sets, each containing node IDs in a community
    """
    if len(G.nodes()) == 0:
        return []

    try:
        # NetworkX doesn't have Louvain built-in, use greedy modularity instead
        communities = nx.community.greedy_modularity_communities(G, resolution=resolution)
        return [set(comm) for comm in communities]
    except Exception:
        return [{node} for node in G.nodes()]


def detect_communities_label_propagation(G: nx.Graph) -> List[Set[str]]:
    """Detect communities using label propagation.

    Label propagation is a fast algorithm where nodes adopt the most
    common label among their neighbours. Good for large graphs.

    Args:
        G: NetworkX graph

    Returns:
        List of sets, each containing node IDs in a community
    """
    if len(G.nodes()) == 0:
        return []

    try:
        communities = nx.community.label_propagation_communities(G)
        return [set(comm) for comm in communities]
    except Exception:
        return [{node} for node in G.nodes()]


def compute_relationship_strength(
    G: nx.Graph, edge_weight_key: str = "weight"
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Analyse relationship strength for all edges.

    Computes multiple metrics for each edge to quantify relationship strength:
    - Edge weight
    - Common neighbours (Jaccard coefficient)
    - Resource allocation index
    - Preferential attachment score

    Args:
        G: NetworkX graph
        edge_weight_key: Attribute name for edge weights

    Returns:
        Dict mapping (node1, node2) -> relationship metrics
    """
    if len(G.edges()) == 0:
        return {}

    strength_data = {}

    # Compute Jaccard coefficient (common neighbours)
    try:
        jaccard = nx.jaccard_coefficient(G)
        jaccard_dict = {(u, v): score for u, v, score in jaccard}
    except Exception:
        jaccard_dict = {}

    # Compute resource allocation index
    try:
        resource_allocation = nx.resource_allocation_index(G)
        ra_dict = {(u, v): score for u, v, score in resource_allocation}
    except Exception:
        ra_dict = {}

    # Compute preferential attachment
    try:
        pref_attachment = nx.preferential_attachment(G)
        pa_dict = {(u, v): score for u, v, score in pref_attachment}
    except Exception:
        pa_dict = {}

    # Combine metrics for each edge
    for u, v in G.edges():
        edge_data = G[u][v]
        weight = edge_data.get(edge_weight_key, 1.0)

        # Normalise edge tuple
        edge_key = (u, v) if (u, v) in jaccard_dict else (v, u)

        strength_data[(u, v)] = {
            "edge_weight": weight,
            "jaccard_coefficient": round(jaccard_dict.get(edge_key, 0.0), 4),
            "resource_allocation": round(ra_dict.get(edge_key, 0.0), 4),
            "preferential_attachment": pa_dict.get(edge_key, 0),
            "combined_strength": round(weight * (1 + jaccard_dict.get(edge_key, 0.0)), 4),
        }

    return strength_data


def compute_network_topology_metrics(G: nx.Graph) -> Dict[str, Any]:
    """Compute global network topology metrics.

    Provides overview statistics about the graph structure:
    - Node/edge counts
    - Density
    - Average clustering coefficient
    - Number of connected components
    - Diameter (longest shortest path)
    - Average path length

    Args:
        G: NetworkX graph

    Returns:
        Dict with topology metrics
    """
    if len(G.nodes()) == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "density": 0.0,
            "avg_clustering": 0.0,
            "num_components": 0,
            "diameter": None,
            "avg_path_length": None,
        }

    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": round(nx.density(G), 4),
    }

    # Clustering coefficient
    try:
        metrics["avg_clustering"] = round(nx.average_clustering(G), 4)
    except Exception:
        metrics["avg_clustering"] = 0.0

    # Connected components
    try:
        metrics["num_components"] = nx.number_connected_components(G)
    except Exception:
        metrics["num_components"] = 1

    # Diameter and average path length (only for connected graphs)
    try:
        if nx.is_connected(G):
            metrics["diameter"] = nx.diameter(G)
            metrics["avg_path_length"] = round(nx.average_shortest_path_length(G), 2)
        else:
            # For disconnected graphs, use largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            metrics["diameter"] = nx.diameter(subgraph)
            metrics["avg_path_length"] = round(nx.average_shortest_path_length(subgraph), 2)
    except Exception:
        metrics["diameter"] = None
        metrics["avg_path_length"] = None

    return metrics


def compute_node_clustering_coefficients(G: nx.Graph) -> Dict[str, float]:
    """Compute local clustering coefficient for each node.

    Clustering coefficient measures how tightly clustered a node's
    neighbours are (what fraction of neighbours are also connected).

    Args:
        G: NetworkX graph

    Returns:
        Dict mapping node_id -> clustering coefficient (0.0-1.0)
    """
    if len(G.nodes()) == 0:
        return {}

    try:
        clustering = nx.clustering(G)
        return {node: round(coeff, 4) for node, coeff in clustering.items()}
    except Exception:
        return {node: 0.0 for node in G.nodes()}


def compute_advanced_analytics(G: nx.Graph) -> Dict[str, Any]:
    """Compute comprehensive advanced analytics for a graph.

    Combines all advanced analytics into a single result:
    - Influence scores (PageRank, betweenness, eigenvector centrality)
    - Community detection (Louvain, label propagation)
    - Relationship strength analysis
    - Network topology metrics
    - Node clustering coefficients

    Args:
        G: NetworkX graph

    Returns:
        Dict with comprehensive analytics results
    """
    analytics = {
        "influence_scores": {
            "pagerank": compute_pagerank_influence(G),
            "betweenness_centrality": compute_betweenness_centrality(G),
            "eigenvector_centrality": compute_eigenvector_centrality(G),
        },
        "communities": {
            "louvain": detect_communities_louvain(G),
            "label_propagation": detect_communities_label_propagation(G),
        },
        "relationship_strength": compute_relationship_strength(G),
        "topology": compute_network_topology_metrics(G),
        "node_clustering": compute_node_clustering_coefficients(G),
    }

    # Add top influencers summary
    pagerank = analytics["influence_scores"]["pagerank"]
    if pagerank:
        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        analytics["top_influencers"] = {
            "by_pagerank": [(node, score) for node, score in top_pagerank],
        }

    betweenness = analytics["influence_scores"]["betweenness_centrality"]
    if betweenness:
        top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
        if "top_influencers" not in analytics:
            analytics["top_influencers"] = {}
        analytics["top_influencers"]["by_betweenness"] = [
            (node, score) for node, score in top_betweenness
        ]

    return analytics


def get_node_influence_rank(analytics: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """Get influence rank for a specific node.

    Args:
        analytics: Analytics dict from compute_advanced_analytics
        node_id: Node to analyse

    Returns:
        Dict with node's ranks and scores across different metrics
    """
    pagerank_scores = analytics["influence_scores"]["pagerank"]
    betweenness_scores = analytics["influence_scores"]["betweenness_centrality"]
    eigen_scores = analytics["influence_scores"]["eigenvector_centrality"]

    # Get ranks
    pagerank_sorted = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    betweenness_sorted = sorted(betweenness_scores.items(), key=lambda x: x[1], reverse=True)
    eigen_sorted = sorted(eigen_scores.items(), key=lambda x: x[1], reverse=True)

    pagerank_rank = next((i + 1 for i, (n, _) in enumerate(pagerank_sorted) if n == node_id), None)
    betweenness_rank = next(
        (i + 1 for i, (n, _) in enumerate(betweenness_sorted) if n == node_id), None
    )
    eigen_rank = next((i + 1 for i, (n, _) in enumerate(eigen_sorted) if n == node_id), None)

    return {
        "node_id": node_id,
        "pagerank": {
            "score": pagerank_scores.get(node_id, 0.0),
            "rank": pagerank_rank,
            "total_nodes": len(pagerank_scores),
        },
        "betweenness": {
            "score": betweenness_scores.get(node_id, 0.0),
            "rank": betweenness_rank,
            "total_nodes": len(betweenness_scores),
        },
        "eigenvector": {
            "score": eigen_scores.get(node_id, 0.0),
            "rank": eigen_rank,
            "total_nodes": len(eigen_scores),
        },
    }
