"""Consistency graph cache manager using SQLite backend.

Manages multiple cached consistency graph instances using SQLite database,
allowing the dashboard to retain and reuse previously built graphs when settings
are toggled back and forth, while standalone builds purge all cached instances.

Architecture:
- Stores graphs in SQLite database: rag_data/cache.db
- Tables: graph_cache, graph_settings_map
- Settings hash computed from: max_neighbours, sim_threshold, where_filter, include_documents

Usage (Dashboard Mode - Retain Existing):
    manager = GraphCacheManager(dashboard_mode=True)
    settings = {'max_neighbours': 20, 'sim_threshold': 0.4}

    # Check if graph exists for these settings
    result = manager.get_cached_graph(settings)
    if result:
        instance_id, graph = result
        graph = manager.load_graph(instance_id)
    else:
        # Build new graph
        graph = build_consistency_graph(...)
        instance_id = manager.save_graph(graph, settings)

Usage (Standalone Mode - Purge All):
    manager = GraphCacheManager(dashboard_mode=False)
    manager.purge_all()  # Remove all cached graphs
    graph = build_consistency_graph(...)
    instance_id = manager.save_graph(graph, settings)
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from scripts.utils.db_factory import get_cache_client


class GraphCacheManager:
    """Manages cached consistency graph instances using SQLite backend."""

    def __init__(self, dashboard_mode: bool = False, rag_data_path: Optional[Path] = None):
        """Initialise the graph cache manager.

        Args:
            dashboard_mode: If True, retain existing graphs; if False, purge on init
            rag_data_path: Path to rag_data directory; defaults to repo rag_data/
        """
        if rag_data_path is None:
            repo_root = Path(__file__).parent.parent.parent
            rag_data_path = repo_root / "rag_data"

        self.rag_data_path = Path(rag_data_path)
        self.dashboard_mode = dashboard_mode

        # Ensure consistency_graphs directory exists
        self.graphs_dir = self.rag_data_path / "consistency_graphs"
        self.graphs_dir.mkdir(parents=True, exist_ok=True)

        # Get SQLite cache client
        self.cache_db = get_cache_client(rag_data_path=self.rag_data_path, enable_cache=True)

        # In standalone mode, purge all existing graphs
        if not dashboard_mode:
            self.purge_all()

    def get_cached_graph(self, settings: Dict[str, Any]) -> Optional[Path]:
        """Check if a cached graph exists for the given settings.

        Args:
            settings: Graph build settings

        Returns:
            Path to cached JSON file if exists, else None
        """
        result = self.cache_db.get_cached_graph(settings)
        if result is None:
            return None

        instance_id, graph_data = result

        # Check if JSON file exists
        json_filename = f"consistency_graph_{instance_id}.json"
        json_path = self.graphs_dir / json_filename

        # If JSON file doesn't exist, create it from cached data
        if not json_path.exists():
            with open(json_path, "w") as f:
                json.dump(graph_data, f, indent=2)

        return json_path

    def load_graph(self, json_path: Path) -> Optional[Dict[str, Any]]:
        """Load graph data from JSON file.

        Args:
            json_path: Path to the JSON file

        Returns:
            Graph data dict, or None if file not found
        """
        if not json_path.exists():
            return None

        with open(json_path, "r") as f:
            return json.load(f)

    def save_graph(self, graph: Dict[str, Any], settings: Dict[str, Any]) -> Path:
        """Save a graph to cache with settings mapping.

        Args:
            graph: Graph dict to save
            settings: Settings used to build the graph

        Returns:
            Path to the saved JSON file
        """
        # Save to SQLite cache
        instance_id = self.cache_db.put_graph(graph, settings)

        # Also write to JSON file for compatibility
        json_filename = f"consistency_graph_{instance_id}.json"
        json_path = self.graphs_dir / json_filename

        with open(json_path, "w") as f:
            json.dump(graph, f, indent=2)

        return json_path

    def purge_all(self) -> int:
        """Remove all cached graphs.

        Returns:
            Number of graphs removed (SQLite count)
        """
        stats_before = self.cache_db.graph_stats()
        count_before = stats_before.get("entries", 0)
        self.cache_db.purge_all_graphs()
        return count_before

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache.

        Returns:
            Dict with cache statistics
        """
        stats = self.cache_db.graph_stats()
        return {
            "total_instances": stats.get("entries", 0),
            "db_path": stats.get("db_path", ""),
        }

    def close(self) -> None:
        """Close the cache database connection."""
        if self.cache_db:
            self.cache_db.close()
