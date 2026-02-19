"""Graph-enhanced retrieval using consistency graph relationships.

Leverages the consistency graph to expand retrieval results with related
technical entities, components, and documents based on graph structure.

Benefits:
- Discover related concepts not explicitly in query
- Follow technical entity relationships
- Surface conflicting or duplicate information
- Hierarchical context expansion (1-hop, 2-hop neighbours)

Integration:
- Connects to existing consistency_graph infrastructure via SQLite
- Uses NetworkX graph structure
- Expands chunk retrieval with graph-connected entities
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx


class GraphEnhancedRetriever:
    """Enhances semantic retrieval with graph-based relationship expansion.

    Loads consistency graph from SQLite store and provides methods to expand
    chunk retrieval by following technical entity relationships, document
    connections, and semantic links discovered during graph construction.
    """

    def __init__(self, graph_source: Optional[Union[Path, str]] = None):
        """Initialise graph-enhanced retriever.

        Args:
            graph_source: Path to SQLite database file or legacy JSON (auto-detect if None)
        """
        self.graph: Optional[nx.Graph] = None
        self.entity_to_chunks: Dict[str, Set[str]] = {}  # entity -> chunk IDs
        self.chunk_to_entities: Dict[str, Set[str]] = {}  # chunk_id -> entities
        self._sqlite_store = None

        if graph_source:
            self.load_graph(graph_source)

    def load_graph(self, graph_source: Union[Path, str]) -> bool:
        """Load consistency graph from SQLite database or legacy JSON.

        Args:
            graph_source: Path to SQLite database or JSON file

        Returns:
            True if loaded successfully, False otherwise
        """
        graph_path = Path(graph_source) if isinstance(graph_source, str) else graph_source

        if not graph_path.exists():
            return False

        try:
            # Check if it's SQLite or JSON based on extension
            if str(graph_path).endswith(".sqlite"):
                return self._load_from_sqlite(graph_path)
            else:
                return self._load_from_json(graph_path)

        except Exception as e:
            print(f"Warning: Failed to load consistency graph: {e}")
            return False

    def _load_from_sqlite(self, sqlite_path: Path) -> bool:
        """Load graph from SQLite database.

        Args:
            sqlite_path: Path to SQLite database file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Import SQLiteGraphStore
            from scripts.consistency_graph.sqlite_store import SQLiteGraphStore

            # Create store and load metadata
            self._sqlite_store = SQLiteGraphStore(str(sqlite_path))
            if not self._sqlite_store.load_metadata():
                return False

            # Build NetworkX graph from all nodes and edges
            self.graph = nx.Graph()

            # Load all nodes (paginated to avoid memory issues)
            all_node_ids = self._sqlite_store.get_node_ids()
            page_size = 1000
            total_pages = (len(all_node_ids) + page_size - 1) // page_size

            for page in range(total_pages):
                nodes_data, _ = self._sqlite_store.get_nodes_paginated(
                    page=page, page_size=page_size
                )
                for node_id, node_data in nodes_data:
                    self.graph.add_node(node_id, **node_data)

            # Load all edges
            edges = self._sqlite_store.get_edges()
            for edge in edges:
                source = edge.get("source")
                target = edge.get("target")
                if source and target:
                    # Remove source/target from edge dict before adding as attributes
                    edge_attrs = {k: v for k, v in edge.items() if k not in ["source", "target"]}
                    self.graph.add_edge(source, target, **edge_attrs)

            # Build entity mappings from node metadata
            self._build_entity_mappings_from_graph()

            return True

        except Exception as e:
            print(f"Warning: Failed to load graph from SQLite: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _load_from_json(self, json_path: Path) -> bool:
        """Load graph from legacy JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(json_path, "r") as f:
                graph_data = json.load(f)

            # Build NetworkX graph from nodes and edges
            self.graph = nx.Graph()

            # Add nodes
            for node_id, node_data in graph_data.get("nodes", {}).items():
                self.graph.add_node(node_id, **node_data)

            # Add edges
            for edge in graph_data.get("edges", []):
                source = edge.get("source")
                target = edge.get("target")
                relationship = edge.get("relationship", "related")
                severity = edge.get("severity", 0.5)

                if source and target:
                    self.graph.add_edge(
                        source, target, relationship=relationship, severity=severity
                    )

            # Build entity mappings (if available in metadata)
            self._build_entity_mappings(graph_data)

            return True

        except Exception as e:
            print(f"Warning: Failed to load graph from JSON: {e}")
            return False

    def _build_entity_mappings(self, graph_data: Dict) -> None:
        """Build entity-to-chunk mappings from graph metadata (JSON loader).

        Args:
            graph_data: Loaded graph JSON data
        """
        # Extract entities from node metadata
        for node_id, node_data in graph_data.get("nodes", {}).items():
            # Check for technical entities in metadata
            entities = node_data.get("technical_entities", [])
            chunk_id = node_data.get("chunk_id", node_id)

            if entities:
                # Map entity -> chunks
                for entity in entities:
                    if entity not in self.entity_to_chunks:
                        self.entity_to_chunks[entity] = set()
                    self.entity_to_chunks[entity].add(chunk_id)

                # Map chunk -> entities
                if chunk_id not in self.chunk_to_entities:
                    self.chunk_to_entities[chunk_id] = set()
                self.chunk_to_entities[chunk_id].update(entities)

    def _build_entity_mappings_from_graph(self) -> None:
        """Build entity-to-chunk mappings from loaded NetworkX graph."""
        if not self.graph:
            return

        for node_id, node_data in self.graph.nodes(data=True):
            # Check for technical entities in metadata
            entities = node_data.get("technical_entities", [])
            chunk_id = node_data.get("chunk_id", node_id)

            if entities:
                # Map entity -> chunks
                for entity in entities:
                    if entity not in self.entity_to_chunks:
                        self.entity_to_chunks[entity] = set()
                    self.entity_to_chunks[entity].add(chunk_id)

                # Map chunk -> entities
                if chunk_id not in self.chunk_to_entities:
                    self.chunk_to_entities[chunk_id] = set()
                self.chunk_to_entities[chunk_id].update(entities)

    def expand_with_neighbours(
        self, chunk_ids: List[str], max_hops: int = 1, max_neighbours: int = 5
    ) -> List[str]:
        """Expand chunk list with graph neighbours.

        Args:
            chunk_ids: Initial chunk IDs from semantic search
            max_hops: Number of hops to traverse (1 or 2 recommended)
            max_neighbours: Max neighbours to add per chunk

        Returns:
            Expanded list of chunk IDs (includes originals)
        """
        if not self.graph:
            return chunk_ids

        expanded = set(chunk_ids)

        for chunk_id in chunk_ids:
            if chunk_id not in self.graph:
                continue

            # Get neighbours within max_hops
            if max_hops == 1:
                neighbours = list(self.graph.neighbors(chunk_id))
            else:
                # BFS for multi-hop
                neighbours = self._bfs_neighbours(chunk_id, max_hops)

            # Rank neighbours by edge weight/severity
            ranked = self._rank_neighbours(chunk_id, neighbours)

            # Add top neighbours
            expanded.update(ranked[:max_neighbours])

        return list(expanded)

    def expand_with_entities(
        self, entities: List[str], max_chunks_per_entity: int = 3
    ) -> List[str]:
        """Find chunks related to specific entities.

        Args:
            entities: List of technical entities (APIs, functions, etc.)
            max_chunks_per_entity: Max chunks to retrieve per entity

        Returns:
            List of chunk IDs containing these entities
        """
        chunk_ids = set()

        for entity in entities:
            if entity in self.entity_to_chunks:
                entity_chunks = list(self.entity_to_chunks[entity])
                # Take top chunks (could rank by relevance later)
                chunk_ids.update(entity_chunks[:max_chunks_per_entity])

        return list(chunk_ids)

    def find_conflicting_chunks(self, chunk_ids: List[str]) -> List[Tuple[str, str, str]]:
        """Find chunks that conflict with given chunks.

        Args:
            chunk_ids: Chunk IDs to check for conflicts

        Returns:
            List of (chunk_id_1, chunk_id_2, relationship) tuples
        """
        if not self.graph:
            return []

        conflicts = []

        for chunk_id in chunk_ids:
            if chunk_id not in self.graph:
                continue

            for neighbour in self.graph.neighbors(chunk_id):
                edge_data = self.graph[chunk_id][neighbour]
                relationship = edge_data.get("relationship", "")

                if relationship in ["conflict", "partial_conflict"]:
                    conflicts.append((chunk_id, neighbour, relationship))

        return conflicts

    def get_cluster_context(self, chunk_ids: List[str]) -> Dict[str, List[str]]:
        """Get cluster memberships for chunks.

        Args:
            chunk_ids: Chunk IDs to check

        Returns:
            Dict mapping cluster IDs to chunk IDs
        """
        if not self.graph:
            return {}

        clusters: Dict[str, List[str]] = {}

        for chunk_id in chunk_ids:
            if chunk_id not in self.graph:
                continue

            node_data = self.graph.nodes[chunk_id]
            # Check for cluster assignments
            for cluster_key in ["risk_cluster", "topic_cluster"]:
                if cluster_key in node_data:
                    cluster_id = node_data[cluster_key]
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(chunk_id)

        return clusters

    def _bfs_neighbours(self, start_node: str, max_hops: int) -> List[str]:
        """BFS to find neighbours within max_hops.

        Args:
            start_node: Starting node ID
            max_hops: Maximum distance to traverse

        Returns:
            List of neighbour node IDs
        """
        if not self.graph or start_node not in self.graph:
            return []

        visited = {start_node}
        queue = [(start_node, 0)]  # (node, depth)
        neighbours = []

        while queue:
            node, depth = queue.pop(0)

            if depth >= max_hops:
                continue

            for neighbour in self.graph.neighbors(node):
                if neighbour not in visited:
                    visited.add(neighbour)
                    neighbours.append(neighbour)
                    queue.append((neighbour, depth + 1))
        return neighbours

    def _rank_neighbours(self, source: str, neighbours: List[str]) -> List[str]:
        """Rank neighbours by edge strength/severity.

        Args:
            source: Source node
            neighbours: List of neighbour nodes

        Returns:
            Sorted list (strongest connections first)
        """
        if not self.graph:
            return neighbours

        assert self.graph is not None

        def get_edge_strength(neighbour: str) -> float:
            assert self.graph is not None
            if not self.graph.has_edge(source, neighbour):
                return 0.0
            return self.graph[source][neighbour].get("severity", 0.5)

        return sorted(neighbours, key=get_edge_strength, reverse=True)


# Global instance (lazy initialisation)
_global_retriever: Optional[GraphEnhancedRetriever] = None


def get_graph_retriever(graph_source: Optional[Union[Path, str]] = None) -> GraphEnhancedRetriever:
    """Get or create global graph retriever instance.

    Args:
        graph_source: Path to SQLite database or legacy JSON file (auto-detect if None)

    Returns:
        GraphEnhancedRetriever instance
    """
    global _global_retriever

    if _global_retriever is None:
        # Auto-detect graph source if not provided
        if graph_source is None:
            # Try SQLite first (preferred), then fall back to JSON
            from scripts.consistency_graph.consistency_config import get_consistency_config

            config = get_consistency_config()

            sqlite_path = Path(config.output_sqlite)
            if sqlite_path.exists():
                graph_source = sqlite_path
            else:
                # Fall back to legacy JSON in consistency_graph directory
                json_path = (
                    Path(__file__).parent.parent / "consistency_graph" / "consistency_graph.json"
                )
                if json_path.exists():
                    graph_source = json_path

        _global_retriever = GraphEnhancedRetriever(graph_source)

    return _global_retriever
