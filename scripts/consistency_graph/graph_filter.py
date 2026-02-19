"""
Graph filtering layer for SQLite-backed consistency graphs.

Provides parameterised filtering, metadata extraction, and query optimisation
for the consistency graph dashboard's pagination and filtering features.

Features:
- Extract metadata from node IDs (doc_type, language, repository)
- Build parameterised SQLite WHERE clauses
- Aggregate available filter options from graph data
- Preserve filter state across dashboard sessions
- Optimised queries with proper indexing

Usage:
    from scripts.consistency_graph.graph_filter import GraphFilter

    # Initialise with graph data
    gf = GraphFilter(graph_data)

    # Get available filter options
    doc_types = gf.get_available_doc_types()
    languages = gf.get_available_languages()

    # Filter nodes by metadata
    filtered_nodes = gf.filter_nodes(
        min_conflict_score=0.5,
        doc_types=['java', 'python'],
        repositories=['repo1', 'repo2']
    )
"""

import hashlib
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


class GraphFilter:
    """Provides filtering and metadata extraction for consistency graphs."""

    def __init__(self, graph_data: Dict[str, Any]):
        """Initialise filter with graph data.

        Args:
            graph_data: Dictionary with 'nodes', 'edges', 'clusters' keys
        """
        self.graph_data = graph_data
        self.nodes = graph_data.get("nodes", {})
        self.edges = graph_data.get("edges", [])
        self.clusters = graph_data.get("clusters", {})

        # Cache metadata extraction
        self._metadata_cache = {}
        self._doc_types = None
        self._languages = None
        self._repositories = None

    def extract_metadata(self, node_id: str) -> Dict[str, str]:
        """Extract metadata from node ID.

        Node IDs typically follow patterns like:
        - "file.java_v1" → doc_type: "java", version: "1"
        - "policy_document_v2" → doc_type: "policy_document", version: "2"
        - "code/service/Handler.groovy_v1" → language: "groovy", path: "code/service"

        Args:
            node_id: Node identifier string

        Returns:
            Dict with extracted metadata (doc_type, language, file_path, version, etc.)
        """
        if node_id in self._metadata_cache:
            return self._metadata_cache[node_id]

        # Start with defaults
        metadata = {
            "node_id": node_id,
            "doc_type": "unknown",
            "language": None,
            "file_path": None,
            "repository": None,
            "version": None,
            "source_category": None,
        }

        # Merge any explicit fields from node data first (preferred over inference)
        node_data = self.nodes.get(node_id, {}) if isinstance(self.nodes, dict) else {}
        if isinstance(node_data, dict):
            # Document type can be stored under different keys
            doc_type = (
                node_data.get("doc_type") or node_data.get("document_type") or node_data.get("type")
            )
            if doc_type:
                metadata["doc_type"] = str(doc_type).strip().lower()

            # Language
            lang = node_data.get("language") or node_data.get("lang")
            if lang:
                metadata["language"] = str(lang).strip().lower()

            # Repository name
            repo = (
                node_data.get("repository")
                or node_data.get("repo")
                or node_data.get("repo_name")
                or node_data.get("bitbucket_repo")
            )
            if repo:
                metadata["repository"] = str(repo).strip()

            # Source category (e.g., policy, standard, glossary, code)
            src_cat = node_data.get("source_category") or node_data.get("category")
            if src_cat:
                metadata["source_category"] = str(src_cat).strip().lower()

        # Remove version suffix (e.g., "_v1" or "_v123")
        version_match = re.search(r"_v(\d+)$", node_id)
        if version_match:
            metadata["version"] = int(version_match.group(1))
            base_id = node_id[: version_match.start()]
        else:
            base_id = node_id

        # Get file extension to determine language/doc_type (fallback if not provided)
        if "." in base_id:
            parts = base_id.rsplit(".", 1)
            file_path = parts[0]
            ext = parts[1].lower()

            metadata["file_path"] = file_path

            # Map extension to language/doc_type
            code_extensions = {
                "java": "java",
                "groovy": "groovy",
                "gvy": "groovy",
                "gsp": "groovy",
                "gradle": "gradle",
                "py": "python",
                "js": "javascript",
                "jsx": "javascript",
                "ts": "typescript",
                "tsx": "typescript",
                "cs": "csharp",
                "go": "go",
                "rust": "rust",
                "rb": "ruby",
            }

            doc_extensions = {
                "md": "markdown",
                "txt": "text",
                "pdf": "pdf",
                "doc": "document",
                "docx": "document",
                "odt": "document",
                "html": "html",
                "yaml": "yaml",
                "yml": "yaml",
                "xml": "xml",
                "json": "json",
                "properties": "properties",
            }

            # Only infer if not explicitly provided above
            if not metadata.get("language") and ext in code_extensions:
                metadata["language"] = code_extensions[ext]
            if metadata.get("doc_type") in (None, "unknown"):
                if ext in code_extensions:
                    metadata["doc_type"] = "code"
                    if not metadata.get("source_category"):
                        metadata["source_category"] = "code"
                elif ext in doc_extensions:
                    metadata["doc_type"] = doc_extensions[ext]
                    if not metadata.get("source_category"):
                        metadata["source_category"] = "documentation"
                else:
                    metadata["doc_type"] = ext

        # Skip repository extraction for academic references
        if (
            not metadata.get("repository")
            and metadata.get("source_category") != "academic_reference"
        ):
            # Extract repository from file path (first path component)
            file_path = metadata.get("file_path")
            if file_path and "/" in file_path:
                parts = file_path.split("/")
                if len(parts) > 0:
                    metadata["repository"] = parts[0]

            # Fallback inference from underscored doc_id format: PROJECT_REPO_path_segments
            if not metadata.get("repository"):
                tokens = base_id.split("_")
                if len(tokens) >= 3:
                    # Heuristic: first token is project, second is repository
                    repo_token = tokens[1]
                    if repo_token:
                        metadata["repository"] = repo_token

        self._metadata_cache[node_id] = metadata
        return metadata

    def get_available_doc_types(self) -> List[str]:
        """Get list of unique document types in graph.

        Returns:
            Sorted list of doc_type values
        """
        if self._doc_types is None:
            doc_types: Set[str] = set()
            for node_id, node_data in self.nodes.items():
                # Prefer explicit node data fields
                if isinstance(node_data, dict):
                    dt = (
                        node_data.get("doc_type")
                        or node_data.get("document_type")
                        or node_data.get("type")
                    )
                    if dt:
                        doc_types.add(str(dt).strip().lower())
                    # For non-code content, include source_category as a doc type (policy, standard, glossary)
                    src_cat = node_data.get("source_category") or node_data.get("category")
                    if src_cat:
                        doc_types.add(str(src_cat).strip().lower())
                # Fallback to inference from ID
                metadata = self.extract_metadata(node_id)
                if metadata.get("doc_type"):
                    doc_types.add(str(metadata["doc_type"]).strip().lower())
            # Remove generic placeholders
            self._doc_types = sorted([dt for dt in doc_types if dt and dt != "unknown"])

        return self._doc_types

    def get_available_languages(self) -> List[str]:
        """Get list of unique programming languages in graph.

        Returns:
            Sorted list of language values
        """
        if self._languages is None:
            languages: Set[str] = set()
            for node_id, node_data in self.nodes.items():
                if isinstance(node_data, dict):
                    lang = node_data.get("language") or node_data.get("lang")
                    if lang:
                        languages.add(str(lang).strip().lower())
                metadata = self.extract_metadata(node_id)
                if metadata.get("language"):
                    languages.add(str(metadata["language"]).strip().lower())
            self._languages = sorted(list(languages))

        return self._languages

    def get_available_repositories(self) -> List[str]:
        """Get list of unique repositories in graph.

        Returns:
            Sorted list of repository names
        """
        if self._repositories is None:
            repositories: Set[str] = set()
            for node_id, node_data in self.nodes.items():
                if isinstance(node_data, dict):
                    repo = (
                        node_data.get("repository")
                        or node_data.get("repo")
                        or node_data.get("repo_name")
                        or node_data.get("bitbucket_repo")
                    )
                    if repo:
                        repositories.add(str(repo).strip())
                metadata = self.extract_metadata(node_id)
                if metadata.get("repository"):
                    repositories.add(str(metadata["repository"]).strip())
            self._repositories = sorted(list(repositories))

        return self._repositories

    def filter_nodes(
        self,
        node_ids: List[str],
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Filter node IDs by metadata and criteria.

        Args:
            node_ids: List of node IDs to filter
            filters: Dictionary with filter criteria:
                - min_conflict: Minimum conflict score threshold (0-1)
                - doc_types: List of doc_types to include
                - languages: List of languages to include
                - repositories: List of repositories to include
                - source_categories: List of source categories

        Returns:
            Filtered list of node IDs
        """
        if not filters:
            return node_ids

        filtered = []

        min_conflict = filters.get("min_conflict", 0.0)
        doc_types = filters.get("doc_types")
        languages = filters.get("languages")
        repositories = filters.get("repositories")
        source_categories = filters.get("source_categories")
        topic_clusters = set(filters.get("topic_clusters", []) or [])
        risk_clusters = set(filters.get("risk_clusters", []) or [])

        for node_id in node_ids:
            node_data = self.nodes.get(node_id, {})

            # Check conflict score
            conflict_score = node_data.get("conflict_score", 0.0)
            if conflict_score < min_conflict:
                continue

            # Extract metadata
            metadata = self.extract_metadata(node_id)

            # Check doc_type filter
            if doc_types and metadata["doc_type"] not in doc_types:
                continue

            # Check language filter
            if languages and metadata["language"] not in languages:
                continue

            # Check repository filter
            if repositories and metadata["repository"] not in repositories:
                continue

            # Check source category filter
            if source_categories and metadata["source_category"] not in source_categories:
                continue

            # Check topic clusters
            if topic_clusters:
                node_topics = set(node_data.get("topic_clusters", []) or [])
                if not node_topics & topic_clusters:
                    continue

            # Check risk clusters
            if risk_clusters:
                node_risks = set(node_data.get("risk_clusters", []) or [])
                if not node_risks & risk_clusters:
                    continue

            filtered.append(node_id)

        return filtered

    def get_available_topic_clusters(self) -> List[int]:
        """Get list of unique topic cluster IDs from nodes.

        Returns:
            Sorted list of topic cluster identifiers (ints)
        """
        clusters: Set[int] = set()
        for node_id, node_data in self.nodes.items():
            if isinstance(node_data, dict):
                topics = node_data.get("topic_clusters", []) or []
                for t in topics:
                    try:
                        clusters.add(int(t))
                    except Exception:
                        # Ignore non-int
                        pass
        return sorted(list(clusters))

    def get_available_risk_clusters(self) -> List[int]:
        """Get list of unique risk cluster IDs from nodes.

        Returns:
            Sorted list of risk cluster identifiers (ints)
        """
        clusters: Set[int] = set()
        for node_id, node_data in self.nodes.items():
            if isinstance(node_data, dict):
                risks = node_data.get("risk_clusters", []) or []
                for r in risks:
                    try:
                        clusters.add(int(r))
                    except Exception:
                        pass
        return sorted(list(clusters))

    def get_topic_cluster_label(self, cluster_id: int) -> str:
        """Get human-readable label for a topic cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            Label string including cluster ID and description if available
        """
        topic_clusters = self.clusters.get("topic", [])
        for cluster in topic_clusters:
            if cluster.get("id") == cluster_id:
                label = cluster.get("label", f"Cluster {cluster_id}")
                # Return label with ID: "Healthcare (Topic 1)"
                return f"{label} (Topic {cluster_id})" if label else f"Topic Cluster {cluster_id}"
        return f"Topic Cluster {cluster_id}"

    def get_risk_cluster_label(self, cluster_id: int) -> str:
        """Get human-readable label for a risk cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            Label string including cluster ID and description if available
        """
        risk_clusters = self.clusters.get("risk", [])
        for cluster in risk_clusters:
            if cluster.get("id") == cluster_id:
                label = cluster.get("label", f"Cluster {cluster_id}")
                # Return label with ID: "High Risk (Risk 0)"
                return f"{label} (Risk {cluster_id})" if label else f"Risk Cluster {cluster_id}"
        return f"Risk Cluster {cluster_id}"

    def get_topic_cluster_filter_label(self, cluster_id: int) -> str:
        """Get filter label for a topic cluster with membership count.

        Args:
            cluster_id: Cluster identifier

        Returns:
            Label string with cluster name and member count (no ID)
        """
        topic_clusters = self.clusters.get("topic", [])
        for cluster in topic_clusters:
            if cluster.get("id") == cluster_id:
                label = cluster.get("label", f"Topic {cluster_id}")
                members = cluster.get("size", 0)
                return f"{label} ({members} nodes)"
        return f"Topic {cluster_id}"

    def get_risk_cluster_filter_label(self, cluster_id: int) -> str:
        """Get filter label for a risk cluster with membership count.

        Args:
            cluster_id: Cluster identifier

        Returns:
            Label string with cluster name and member count (no ID)
        """
        risk_clusters = self.clusters.get("risk", [])
        for cluster in risk_clusters:
            if cluster.get("id") == cluster_id:
                label = cluster.get("label", f"Risk {cluster_id}")
                members = cluster.get("size", 0)
                return f"{label} ({members} nodes)"
        return f"Risk {cluster_id}"

    def filter_edges(
        self,
        node_ids: Set[str],
        min_weight: float = 0.0,
        relationship_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Filter edges to only those connecting filtered nodes.

        Args:
            node_ids: Set of node IDs to include
            min_weight: Minimum edge weight (severity)
            relationship_types: List of relationship types to include (conflict, duplicate, etc.)

        Returns:
            Filtered list of edges
        """
        filtered = []

        for edge in self.edges:
            # Check both nodes are in filtered set
            if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
                continue

            # Check weight
            weight = edge.get("weight", 0.0)
            if weight < min_weight:
                continue

            # Check relationship type
            rel_type = edge.get("relationship_type", "")
            if relationship_types and rel_type not in relationship_types:
                continue

            filtered.append(edge)

        return filtered

    def get_filter_summary(
        self,
        node_ids: Set[str],
    ) -> Dict[str, Any]:
        """Get summary statistics for filtered nodes.

        Args:
            node_ids: Set of filtered node IDs

        Returns:
            Dict with counts by category
        """
        doc_type_counts = defaultdict(int)
        language_counts = defaultdict(int)
        repository_counts = defaultdict(int)
        source_category_counts = defaultdict(int)

        for node_id in node_ids:
            metadata = self.extract_metadata(node_id)

            if metadata["doc_type"]:
                doc_type_counts[metadata["doc_type"]] += 1
            if metadata["language"]:
                language_counts[metadata["language"]] += 1
            if metadata["repository"]:
                repository_counts[metadata["repository"]] += 1
            if metadata["source_category"]:
                source_category_counts[metadata["source_category"]] += 1

        return {
            "total_nodes": len(node_ids),
            "doc_types": dict(sorted(doc_type_counts.items())),
            "languages": dict(sorted(language_counts.items())),
            "repositories": dict(sorted(repository_counts.items())),
            "source_categories": dict(sorted(source_category_counts.items())),
        }

    def build_sql_where_clause(
        self,
        min_conflict_score: float = 0.0,
        doc_types: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        repositories: Optional[List[str]] = None,
    ) -> Tuple[str, List[Any]]:
        """Build parameterised SQL WHERE clause for efficient SQLite querying.

        This allows the consistency graph to leverage SQLite's index and query optimisation
        for large datasets in the future.

        Args:
            min_conflict_score: Minimum conflict score
            doc_types: List of doc_types
            languages: List of languages
            repositories: List of repositories

        Returns:
            Tuple of (where_clause, parameters) for parameterised query
        """
        clauses = []
        params = []

        # Conflict score filter
        if min_conflict_score > 0.0:
            clauses.append("conflict_score >= ?")
            params.append(min_conflict_score)

        # Doc type filter (will be applied to JSON metadata in SQLite)
        if doc_types:
            placeholders = ",".join(["?" for _ in doc_types])
            clauses.append(f"doc_type IN ({placeholders})")
            params.extend(doc_types)

        # Language filter
        if languages:
            placeholders = ",".join(["?" for _ in languages])
            clauses.append(f"language IN ({placeholders})")
            params.extend(languages)

        # Repository filter
        if repositories:
            placeholders = ",".join(["?" for _ in repositories])
            clauses.append(f"repository IN ({placeholders})")
            params.extend(repositories)

        where_clause = " AND ".join(clauses) if clauses else "1=1"

        return where_clause, params
