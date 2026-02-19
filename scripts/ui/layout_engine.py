"""
Graph layout engine for Plotly Dash WebGL rendering.

Provides multiple layout algorithms for positioning nodes in 2D space:
- Force-directed (spring-force model with Coulomb repulsion)
- Hierarchical (layered layout for DAGs)
- Circular (nodes arranged in concentric circles by cluster)

Usage:
    from scripts.ui.layout_engine import ForceDirectedLayout

    layout = ForceDirectedLayout(k=1.0, damping=0.5, iterations=100)
    positions = layout.compute_layout(nodes_dict, edges_list)

    # Returns: {node_id: (x, y)}
"""

import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


class LayoutEngine(ABC):
    """Abstract base class for graph layout algorithms."""

    @abstractmethod
    def compute_layout(
        self,
        nodes: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> Dict[str, Tuple[float, float]]:
        """Compute 2D positions for all nodes.

        Args:
            nodes: Dictionary of node_id -> node_data
            edges: List of edge dicts with 'source' and 'target'

        Returns:
            Dictionary mapping node_id -> (x, y) position
        """
        pass

    def _get_node_neighbors(self, node_id: str, edges: List[Dict]) -> List[str]:
        """Get all neighbors of a node."""
        neighbors = []
        for edge in edges:
            if edge.get("source") == node_id:
                neighbors.append(edge.get("target"))
            elif edge.get("target") == node_id:
                neighbors.append(edge.get("source"))
        return neighbors


class ForceDirectedLayout(LayoutEngine):
    """
    Spring-force layout using force-directed graph drawing.

    Nodes repel each other (Coulomb force) while edges act as springs.
    Uses iterative simulation to find equilibrium positions.

    Good for: General graphs, discovering structure
    Time complexity: O(n²) per iteration (can optimise with Barnes-Hut)
    """

    def __init__(
        self,
        k: float = 1.0,
        c_rep: float = 1.0,
        c_spring: float = 0.1,
        damping: float = 0.5,
        iterations: int = 50,
        seed: int = 42,
    ):
        """Initialise force-directed layout.

        Args:
            k: Optimal spring length (distance between nodes)
            c_rep: Coulomb repulsion coefficient (higher = more repulsion)
            c_spring: Spring force coefficient (higher = stronger attraction)
            damping: Velocity damping factor (0-1, higher = slower)
            iterations: Number of simulation iterations
            seed: Random seed for reproducibility
        """
        self.k = k
        self.c_rep = c_rep
        self.c_spring = c_spring
        self.damping = damping
        self.iterations = iterations
        self.seed = seed
        random.seed(seed)

    def compute_layout(
        self,
        nodes: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> Dict[str, Tuple[float, float]]:
        """Compute positions using force-directed algorithm.

        Algorithm:
        1. Initialise random positions
        2. For each iteration:
            a. Calculate repulsive forces (all pairs)
            b. Calculate attractive forces (connected pairs)
            c. Update velocities with damping
            d. Update positions
        3. Return final positions

        Args:
            nodes: Dictionary of node_id -> node_data
            edges: List of edges

        Returns:
            Dictionary of node_id -> (x, y) position
        """
        if not nodes:
            return {}

        node_ids = list(nodes.keys())
        n = len(node_ids)

        # Initialise random positions in [-1, 1] range
        positions = {
            node_id: (
                random.uniform(-1, 1),
                random.uniform(-1, 1),
            )
            for node_id in node_ids
        }

        # Initialise velocities
        velocities = {node_id: (0.0, 0.0) for node_id in node_ids}

        # Precompute adjacency for faster edge lookups
        adjacency = defaultdict(set)
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                adjacency[source].add(target)
                adjacency[target].add(source)

        # Simulation loop
        for iteration in range(self.iterations):
            forces = {node_id: (0.0, 0.0) for node_id in node_ids}

            # Calculate repulsive forces (all pairs)
            for i, node_a in enumerate(node_ids):
                for node_b in node_ids[i + 1 :]:
                    x_a, y_a = positions[node_a]
                    x_b, y_b = positions[node_b]

                    # Distance vector
                    dx = x_b - x_a
                    dy = y_b - y_a
                    dist_sq = dx * dx + dy * dy + 0.01  # Avoid division by zero

                    # Repulsive force magnitude (inverse square)
                    f_rep = self.c_rep * (self.k**2) / math.sqrt(dist_sq)

                    # Normalise and apply
                    if dist_sq > 0:
                        norm = math.sqrt(dist_sq)
                        fx = (dx / norm) * f_rep
                        fy = (dy / norm) * f_rep

                        # Apply to both nodes (Newton's 3rd law)
                        forces[node_a] = (
                            forces[node_a][0] - fx,
                            forces[node_a][1] - fy,
                        )
                        forces[node_b] = (
                            forces[node_b][0] + fx,
                            forces[node_b][1] + fy,
                        )

            # Calculate attractive forces (connected pairs)
            for edge in edges:
                source = edge.get("source")
                target = edge.get("target")
                if not (source and target):
                    continue

                x_a, y_a = positions[source]
                x_b, y_b = positions[target]

                # Distance vector
                dx = x_b - x_a
                dy = y_b - y_a
                dist = math.sqrt(dx * dx + dy * dy) + 0.01

                # Attractive force (Hooke's law)
                f_spring = self.c_spring * (dist - self.k)

                # Normalise and apply
                if dist > 0:
                    fx = (dx / dist) * f_spring
                    fy = (dy / dist) * f_spring

                    # Pull nodes together
                    forces[source] = (
                        forces[source][0] + fx,
                        forces[source][1] + fy,
                    )
                    forces[target] = (
                        forces[target][0] - fx,
                        forces[target][1] - fy,
                    )

            # Update velocities and positions
            for node_id in node_ids:
                fx, fy = forces[node_id]
                vx, vy = velocities[node_id]

                # Apply damping
                vx = (vx + fx) * self.damping
                vy = (vy + fy) * self.damping

                # Limit velocity (prevent explosion) - but allow boundaryless expansion
                max_velocity = 1.0  # Higher limit for boundaryless layout
                speed = math.sqrt(vx * vx + vy * vy)
                if speed > max_velocity:
                    vx = (vx / speed) * max_velocity
                    vy = (vy / speed) * max_velocity

                # Update position (no boundary constraints - let nodes spread naturally)
                x, y = positions[node_id]
                x += vx
                y += vy

                positions[node_id] = (x, y)
                velocities[node_id] = (vx, vy)

        return positions


class HierarchicalLayout(LayoutEngine):
    """
    Layered/hierarchical layout for DAGs and tree structures.

    Uses Sugiyama algorithm:
    1. Layer assignment (topological sort)
    2. Ordering (minimise edge crossings)
    3. Coordinate assignment (minimise total edge length)

    Good for: Hierarchical data, DAGs, tree-like graphs
    Time complexity: O(n) with good heuristics
    """

    def __init__(self, layer_gap: float = 2.0, node_gap: float = 1.0):
        """Initialise hierarchical layout.

        Args:
            layer_gap: Vertical distance between layers
            node_gap: Horizontal distance between nodes in same layer
        """
        self.layer_gap = layer_gap
        self.node_gap = node_gap

    def compute_layout(
        self,
        nodes: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> Dict[str, Tuple[float, float]]:
        """Compute hierarchical layout.

        Args:
            nodes: Dictionary of node_id -> node_data
            edges: List of edges

        Returns:
            Dictionary of node_id -> (x, y) position
        """
        if not nodes:
            return {}

        node_ids = list(nodes.keys())

        # Build adjacency list
        adj = defaultdict(list)
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                adj[source].append(target)

        # Layer assignment (topological sort with BFS)
        layers = self._assign_layers(node_ids, adj)

        # Coordinate assignment
        positions = {}
        for layer_idx, layer_nodes in enumerate(layers):
            y = layer_idx * self.layer_gap
            for node_idx, node_id in enumerate(layer_nodes):
                x = node_idx * self.node_gap - (len(layer_nodes) * self.node_gap) / 2
                positions[node_id] = (x, y)

        return positions

    def _assign_layers(self, node_ids: List[str], adj: Dict) -> List[List[str]]:
        """Assign nodes to layers using topological ordering.

        Args:
            node_ids: List of all node IDs
            adj: Adjacency list (source -> [targets])

        Returns:
            List of layers, each containing node IDs
        """
        # Calculate in-degree
        in_degree = defaultdict(int)
        for source in adj:
            for target in adj[source]:
                in_degree[target] += 1

        # Initialise nodes with no incoming edges
        queue = [n for n in node_ids if in_degree[n] == 0]
        layers = []

        while queue:
            # Current layer
            current_layer = queue[:]
            layers.append(current_layer)

            # Remove processed nodes and update degrees
            next_queue = set()
            for node in current_layer:
                for target in adj.get(node, []):
                    in_degree[target] -= 1
                    if in_degree[target] == 0:
                        next_queue.add(target)

            queue = list(next_queue)

        # Handle isolated nodes
        processed = {n for layer in layers for n in layer}
        remaining = [n for n in node_ids if n not in processed]
        if remaining:
            layers.append(remaining)

        return layers


class CircularLayout(LayoutEngine):
    """
    Circular layout with nodes arranged by cluster/community.

    Algorithm:
    1. Detect clusters (using conflict_score or provided clusters)
    2. Arrange clusters in concentric circles
    3. Place nodes within each cluster circle

    Good for: Discovering clusters, showing relationships
    Time complexity: O(n) after clustering
    """

    def __init__(self, cluster_key: str = "community", radius_gap: float = 1.0):
        """Initialise circular layout.

        Args:
            cluster_key: Node attribute key for cluster assignment
            radius_gap: Distance between cluster circles
        """
        self.cluster_key = cluster_key
        self.radius_gap = radius_gap

    def compute_layout(
        self,
        nodes: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> Dict[str, Tuple[float, float]]:
        """Compute circular layout grouped by clusters.

        Args:
            nodes: Dictionary of node_id -> node_data
            edges: List of edges

        Returns:
            Dictionary of node_id -> (x, y) position
        """
        if not nodes:
            return {}

        # Group nodes by cluster
        clusters = defaultdict(list)
        for node_id, node_data in nodes.items():
            cluster = node_data.get(self.cluster_key, "default")
            clusters[cluster].append(node_id)

        # Sort clusters by size (largest first)
        cluster_list = sorted(clusters.items(), key=lambda x: -len(x[1]))

        positions = {}
        for cluster_idx, (cluster_id, node_ids) in enumerate(cluster_list):
            # Cluster radius increases with each layer
            radius = (cluster_idx + 1) * self.radius_gap

            # Place nodes evenly around circle
            n_nodes = len(node_ids)
            for node_idx, node_id in enumerate(node_ids):
                angle = (2 * math.pi * node_idx) / n_nodes
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                positions[node_id] = (x, y)

        return positions


def get_adaptive_layout_params(num_nodes: int, num_edges: int) -> Dict[str, Any]:
    """
    Determine optimal force-directed layout parameters based on graph size.

    Strategy:
    - Small graphs (< 50 nodes): Fruchterman-Reingold style
      Higher repulsion, more iterations, better for detailed structure
    - Medium graphs (50-200 nodes): Balanced parameters
      Standard force-directed with moderate settings
    - Large graphs (> 200 nodes): Force Atlas 2 style
      Lower repulsion, higher spring strength, boundaryless expansion

    Args:
        num_nodes: Number of nodes in graph
        num_edges: Number of edges in graph

    Returns:
        Dictionary of layout parameters
    """
    density = num_edges / max(num_nodes, 1) if num_nodes > 0 else 0

    if num_nodes < 50:
        # Small graphs: Fruchterman-Reingold (boundaryless)
        return {
            "k": 2.5,  # Larger optimal distance for better spread
            "c_rep": 1.8,  # Strong repulsion for spacing
            "c_spring": 0.05,  # Lighter attraction to allow spreading
            "damping": 0.7,  # Higher damping for stability
            "iterations": 100,  # More iterations for convergence
        }
    elif num_nodes < 200:
        # Medium graphs: Balanced approach (boundaryless)
        return {
            "k": 2.0,  # Increased for better spacing
            "c_rep": 1.2,  # Slightly higher repulsion
            "c_spring": 0.08,  # Lighter springs
            "damping": 0.6,  # Good stability
            "iterations": 80,  # More iterations
        }
    else:
        # Large graphs: Force Atlas 2 style (boundaryless expansion)
        # Optimised for natural clustering without boundary constraints
        scaling_factor = math.sqrt(num_nodes / 200.0)
        return {
            "k": 1.5 / scaling_factor,  # Optimal distance scales with size
            "c_rep": 0.5,  # Moderate repulsion for natural spread
            "c_spring": 0.12 * scaling_factor,  # Balanced spring strength
            "damping": 0.5,  # Balanced damping
            "iterations": min(200, 80 + num_nodes // 8),  # More iterations for large graphs
        }


# Convenience function for computing layout with auto-selection
def compute_layout(
    nodes: Dict[str, Dict[str, Any]],
    edges: List[Dict[str, Any]],
    layout_type: str = "force",
    adaptive: bool = True,
    **kwargs,
) -> Dict[str, Tuple[float, float]]:
    """Compute node positions using specified layout algorithm.

    Args:
        nodes: Dictionary of node_id -> node_data
        edges: List of edges
        layout_type: One of 'force', 'hierarchical', 'circular'
        adaptive: If True and layout_type='force', auto-select optimal parameters
        **kwargs: Additional arguments for the layout engine (override adaptive)

    Returns:
        Dictionary of node_id -> (x, y) position

    Example:
        # Adaptive parameters (recommended for varying graph sizes)
        positions = compute_layout(nodes, edges, layout_type='force', adaptive=True)

        # Manual parameters
        positions = compute_layout(nodes, edges, layout_type='force', k=1.0, iterations=50)
    """
    if layout_type == "force":
        # Apply adaptive parameters if requested and no overrides provided
        if adaptive and not kwargs:
            num_nodes = len(nodes)
            num_edges = len(edges)
            kwargs = get_adaptive_layout_params(num_nodes, num_edges)
        engine = ForceDirectedLayout(**kwargs)
    elif layout_type == "hierarchical":
        engine = HierarchicalLayout(**kwargs)
    elif layout_type == "circular":
        engine = CircularLayout(**kwargs)
    else:
        raise ValueError(f"Unknown layout type: {layout_type}")

    return engine.compute_layout(nodes, edges)
