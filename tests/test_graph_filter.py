"""Tests for graph_filter module.

Covers metadata extraction, filtering, aggregation, and SQL query building.
"""

import pytest
from scripts.consistency_graph.graph_filter import GraphFilter


@pytest.fixture
def sample_graph_data():
    """Create sample graph data for testing."""
    return {
        "nodes": {
            "Handler.java_v1": {
                "conflict_score": 0.8,
                "metadata": {"source": "bitbucket"},
            },
            "service/Controller.groovy_v1": {
                "conflict_score": 0.5,
                "metadata": {"source": "bitbucket"},
            },
            "utils/helper.py_v2": {
                "conflict_score": 0.3,
                "metadata": {"source": "bitbucket"},
            },
            "docs/API.md_v1": {
                "conflict_score": 0.2,
                "metadata": {"source": "docs"},
            },
            "config/app.properties_v3": {
                "conflict_score": 0.6,
                "metadata": {"source": "config"},
            },
            "repo2/service/Auth.java_v1": {
                "conflict_score": 0.7,
                "metadata": {"source": "bitbucket"},
            },
        },
        "edges": [
            {"source": "Handler.java_v1", "target": "service/Controller.groovy_v1", "weight": 0.8, "relationship_type": "conflict"},
            {"source": "Handler.java_v1", "target": "utils/helper.py_v2", "weight": 0.5, "relationship_type": "duplicate"},
            {"source": "service/Controller.groovy_v1", "target": "docs/API.md_v1", "weight": 0.3, "relationship_type": "references"},
        ],
        "clusters": {},
    }


class TestExtractMetadata:
    """Tests for metadata extraction from node IDs."""

    def test_extract_java_file(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        meta = gf.extract_metadata("Handler.java_v1")
        
        assert meta["doc_type"] == "code"
        assert meta["language"] == "java"
        assert meta["file_path"] == "Handler"
        assert meta["version"] == 1
        assert meta["source_category"] == "code"

    def test_extract_groovy_file(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        meta = gf.extract_metadata("service/Controller.groovy_v1")
        
        assert meta["language"] == "groovy"
        assert meta["doc_type"] == "code"
        assert meta["file_path"] == "service/Controller"
        assert meta["repository"] == "service"
        assert meta["version"] == 1

    def test_extract_python_file(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        meta = gf.extract_metadata("utils/helper.py_v2")
        
        assert meta["language"] == "python"
        assert meta["doc_type"] == "code"
        assert meta["version"] == 2
        assert meta["repository"] == "utils"

    def test_extract_markdown_file(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        meta = gf.extract_metadata("docs/API.md_v1")
        
        assert meta["doc_type"] == "markdown"
        assert meta["language"] is None
        assert meta["source_category"] == "documentation"
        assert meta["file_path"] == "docs/API"

    def test_extract_properties_file(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        meta = gf.extract_metadata("config/app.properties_v3")
        
        assert meta["doc_type"] == "properties"
        assert meta["language"] is None
        assert meta["version"] == 3

    def test_extract_no_version_suffix(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        meta = gf.extract_metadata("file.java")
        
        assert meta["version"] is None
        assert meta["language"] == "java"

    def test_extract_unknown_extension(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        meta = gf.extract_metadata("file.xyz_v1")
        
        assert meta["doc_type"] == "xyz"
        assert meta["language"] is None

    def test_extract_no_extension(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        meta = gf.extract_metadata("Makefile_v1")
        
        assert meta["doc_type"] == "unknown"
        assert meta["language"] is None

    def test_extract_metadata_caching(self, sample_graph_data):
        """Metadata extraction results are cached."""
        gf = GraphFilter(sample_graph_data)
        
        meta1 = gf.extract_metadata("Handler.java_v1")
        meta2 = gf.extract_metadata("Handler.java_v1")
        
        # Should return same object from cache
        assert meta1 is meta2

    def test_extract_multiple_path_components(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        meta = gf.extract_metadata("repo2/service/Auth.java_v1")
        
        assert meta["repository"] == "repo2"
        assert meta["file_path"] == "repo2/service/Auth"


class TestGetAvailableFilters:
    """Tests for aggregating available filter options."""

    def test_get_available_doc_types(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        doc_types = gf.get_available_doc_types()
        
        assert "code" in doc_types
        assert "markdown" in doc_types
        assert "properties" in doc_types
        assert doc_types == sorted(doc_types)  # Should be sorted

    def test_get_available_languages(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        languages = gf.get_available_languages()
        
        assert "java" in languages
        assert "groovy" in languages
        assert "python" in languages
        assert languages == sorted(languages)

    def test_get_available_repositories(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        repos = gf.get_available_repositories()
        
        assert "service" in repos
        assert "utils" in repos
        assert "repo2" in repos
        assert "docs" in repos
        assert "config" in repos
        assert repos == sorted(repos)

    def test_get_available_filters_empty_graph(self):
        """Handle empty graph gracefully."""
        gf = GraphFilter({"nodes": {}, "edges": [], "clusters": {}})
        
        assert gf.get_available_doc_types() == []
        assert gf.get_available_languages() == []
        assert gf.get_available_repositories() == []

    def test_get_available_filters_caching(self, sample_graph_data):
        """Filter options are cached."""
        gf = GraphFilter(sample_graph_data)
        
        types1 = gf.get_available_doc_types()
        types2 = gf.get_available_doc_types()
        
        assert types1 is types2  # Same object


class TestFilterNodes:
    """Tests for node filtering by criteria."""

    def test_filter_nodes_no_filters(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        node_ids = list(sample_graph_data["nodes"].keys())
        
        filtered = gf.filter_nodes(node_ids, filters=None)
        
        assert filtered == node_ids

    def test_filter_nodes_empty_filters(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        node_ids = list(sample_graph_data["nodes"].keys())
        
        filtered = gf.filter_nodes(node_ids, filters={})
        
        assert filtered == node_ids

    def test_filter_nodes_by_min_conflict(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        node_ids = list(sample_graph_data["nodes"].keys())
        
        filtered = gf.filter_nodes(node_ids, filters={"min_conflict": 0.5})
        
        # Should include Handler.java_v1 (0.8), service/Controller.groovy_v1 (0.5),
        # config/app.properties_v3 (0.6), repo2/service/Auth.java_v1 (0.7)
        assert len(filtered) == 4
        assert "Handler.java_v1" in filtered
        assert "docs/API.md_v1" not in filtered  # 0.2 < 0.5

    def test_filter_nodes_by_doc_type(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        node_ids = list(sample_graph_data["nodes"].keys())
        
        filtered = gf.filter_nodes(node_ids, filters={"doc_types": ["code"]})
        
        assert "Handler.java_v1" in filtered
        assert "service/Controller.groovy_v1" in filtered
        assert "docs/API.md_v1" not in filtered
        assert len(filtered) == 4  # Four code files (Handler, Controller, helper, Auth)

    def test_filter_nodes_by_language(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        node_ids = list(sample_graph_data["nodes"].keys())
        
        filtered = gf.filter_nodes(node_ids, filters={"languages": ["java"]})
        
        assert "Handler.java_v1" in filtered
        assert "repo2/service/Auth.java_v1" in filtered
        assert "utils/helper.py_v2" not in filtered
        assert len(filtered) == 2

    def test_filter_nodes_by_repository(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        node_ids = list(sample_graph_data["nodes"].keys())
        
        filtered = gf.filter_nodes(node_ids, filters={"repositories": ["repo2"]})
        
        assert "repo2/service/Auth.java_v1" in filtered
        assert len(filtered) == 1

    def test_filter_nodes_by_source_category(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        node_ids = list(sample_graph_data["nodes"].keys())
        
        filtered = gf.filter_nodes(node_ids, filters={"source_categories": ["code"]})
        
        # Should include all code files
        assert "Handler.java_v1" in filtered
        assert "docs/API.md_v1" not in filtered  # documentation category
        assert len(filtered) == 4

    def test_filter_nodes_combined_filters(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        node_ids = list(sample_graph_data["nodes"].keys())
        
        filtered = gf.filter_nodes(
            node_ids,
            filters={
                "min_conflict": 0.6,
                "languages": ["java"],
            }
        )
        
        # Handler.java_v1 (0.8, java) and repo2/service/Auth.java_v1 (0.7, java) pass
        assert "Handler.java_v1" in filtered
        assert "repo2/service/Auth.java_v1" in filtered
        assert len(filtered) == 2

    def test_filter_nodes_empty_input(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        
        filtered = gf.filter_nodes([], filters={"min_conflict": 0.5})
        
        assert filtered == []


class TestFilterEdges:
    """Tests for edge filtering."""

    def test_filter_edges_no_filters(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        node_ids = set(sample_graph_data["nodes"].keys())
        
        filtered = gf.filter_edges(node_ids)
        
        # All edges should be included
        assert len(filtered) == 3

    def test_filter_edges_by_min_weight(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        node_ids = set(sample_graph_data["nodes"].keys())
        
        filtered = gf.filter_edges(node_ids, min_weight=0.5)
        
        # Should include edges with weight >= 0.5
        assert len(filtered) == 2

    def test_filter_edges_by_relationship_type(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        node_ids = set(sample_graph_data["nodes"].keys())
        
        filtered = gf.filter_edges(node_ids, relationship_types=["conflict"])
        
        assert len(filtered) == 1
        assert filtered[0]["source"] == "Handler.java_v1"

    def test_filter_edges_disconnected_nodes(self, sample_graph_data):
        """Edges to filtered-out nodes are excluded."""
        gf = GraphFilter(sample_graph_data)
        # Only include Handler.java_v1
        node_ids = {"Handler.java_v1"}
        
        filtered = gf.filter_edges(node_ids)
        
        # No edges should be included (no both source and target in node_ids)
        assert len(filtered) == 0

    def test_filter_edges_combined_filters(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        node_ids = set(sample_graph_data["nodes"].keys())
        
        filtered = gf.filter_edges(
            node_ids,
            min_weight=0.5,
            relationship_types=["conflict", "duplicate"],
        )
        
        # Should include edges with weight >= 0.5 and type in [conflict, duplicate]
        # conflict edge (0.8) and duplicate edge (0.5) both match
        assert len(filtered) == 2


class TestGetFilterSummary:
    """Tests for filter summary statistics."""

    def test_get_filter_summary_all_nodes(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        node_ids = set(sample_graph_data["nodes"].keys())
        
        summary = gf.get_filter_summary(node_ids)
        
        assert summary["total_nodes"] == 6
        assert summary["doc_types"]["code"] == 4
        assert summary["doc_types"]["markdown"] == 1
        assert summary["doc_types"]["properties"] == 1
        assert summary["languages"]["java"] == 2
        assert summary["languages"]["python"] == 1
        assert summary["source_categories"]["code"] == 4
        assert summary["source_categories"]["documentation"] == 2  # markdown and properties files

    def test_get_filter_summary_empty_set(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        
        summary = gf.get_filter_summary(set())
        
        assert summary["total_nodes"] == 0
        assert summary["doc_types"] == {}
        assert summary["languages"] == {}

    def test_get_filter_summary_partial_nodes(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        node_ids = {"Handler.java_v1", "utils/helper.py_v2"}
        
        summary = gf.get_filter_summary(node_ids)
        
        assert summary["total_nodes"] == 2
        assert summary["languages"]["java"] == 1
        assert summary["languages"]["python"] == 1
        assert "groovy" not in summary["languages"]


class TestBuildSqlWhereClause:
    """Tests for SQL WHERE clause building."""

    def test_build_where_clause_no_filters(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        
        where_clause, params = gf.build_sql_where_clause()
        
        assert where_clause == "1=1"
        assert params == []

    def test_build_where_clause_min_conflict(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        
        where_clause, params = gf.build_sql_where_clause(min_conflict_score=0.5)
        
        assert "conflict_score >= ?" in where_clause
        assert params == [0.5]

    def test_build_where_clause_doc_types(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        
        where_clause, params = gf.build_sql_where_clause(doc_types=["code", "markdown"])
        
        assert "doc_type IN (?,?)" in where_clause
        assert params == ["code", "markdown"]

    def test_build_where_clause_languages(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        
        where_clause, params = gf.build_sql_where_clause(languages=["java", "python"])
        
        assert "language IN (?,?)" in where_clause
        assert params == ["java", "python"]

    def test_build_where_clause_repositories(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        
        where_clause, params = gf.build_sql_where_clause(repositories=["repo1"])
        
        assert "repository IN (?)" in where_clause
        assert params == ["repo1"]

    def test_build_where_clause_combined(self, sample_graph_data):
        gf = GraphFilter(sample_graph_data)
        
        where_clause, params = gf.build_sql_where_clause(
            min_conflict_score=0.5,
            doc_types=["code"],
            languages=["java"],
            repositories=["repo1", "repo2"],
        )
        
        # All clauses should be present and joined with AND
        assert "conflict_score >= ?" in where_clause
        assert "doc_type IN (?)" in where_clause
        assert "language IN (?)" in where_clause
        assert "repository IN (?,?)" in where_clause
        assert where_clause.count(" AND ") == 3
        
        # Parameters should be in order
        assert params[0] == 0.5
        assert params[1] == "code"
        assert params[2] == "java"
        assert "repo1" in params
        assert "repo2" in params

    def test_build_where_clause_parametrised_safe(self, sample_graph_data):
        """Verify no SQL injection via proper parameterisation."""
        gf = GraphFilter(sample_graph_data)
        
        # Try to inject SQL
        where_clause, params = gf.build_sql_where_clause(
            doc_types=["code'; DROP TABLE nodes; --"]
        )
        
        # Should not contain the injection, only parameters
        assert "DROP TABLE" not in where_clause
        assert "code'; DROP TABLE nodes; --" in params
