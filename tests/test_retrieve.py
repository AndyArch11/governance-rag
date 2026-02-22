"""Tests for retrieve module.

Covers validation, Chroma query interaction, auditing, and logging.
"""

import sys
from pathlib import Path

import pytest


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.errors = []
        self.warnings = []
        self.debugs = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg)

    def error(self, msg, *args, **kwargs):
        self.errors.append(msg)

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg)

    def debug(self, msg, *args, **kwargs):
        self.debugs.append(msg)


class DummyCollection:
    def __init__(self, results=None):
        self.results = results or {
            "ids": [["id1", "id2"]],
            "documents": [["chunk1", "chunk2"]],
            "metadatas": [[{"id": "1"}, {"id": "2"}]],
            "distances": [[0.1, 0.2]],
        }
        self.queries = []
        self.gets = []

    def query(
        self, query_embeddings=None, query_texts=None, n_results=None, where=None, include=None
    ):  # noqa: D401
        """Mock vector search."""
        # Record the query call with all parameters
        self.queries.append(
            {
                "query_embeddings": query_embeddings,
                "query_texts": query_texts,
                "n_results": n_results,
                "where": where,
                "include": include,
            }
        )
        return self.results

    def get(self, where=None, limit=None, include=None):
        """Mock keyword search get operation."""
        self.gets.append({"where": where, "limit": limit, "include": include})
        # Return empty results for keyword search by default
        return {"ids": [], "documents": [], "metadatas": []}


def get_where_value(where_clause, key):
    """Extract value from where clause Supporting both direct and $and formats.

    Args:
        where_clause: ChromaDB where clause (dict)
        key: Key to extract (e.g., "language", "source_category")

    Returns:
        Value if found, None otherwise
    """
    if where_clause is None:
        return None

    # Direct key access
    if key in where_clause:
        return where_clause[key]

    # $and format: {'$and': [{'embedding_model': '...'}, {'language': 'java'}, ...]}
    if "$and" in where_clause:
        for condition in where_clause["$and"]:
            if isinstance(condition, dict) and key in condition:
                return condition[key]

    return None


@pytest.fixture()
def retrieve_module(monkeypatch):
    """Load retrieve with mocked logger and audit."""
    import importlib

    sys.modules.pop("scripts.rag.retrieve", None)
    retrieve = importlib.import_module("scripts.rag.retrieve")

    dummy_logger = DummyLogger()
    audit_events = []

    def dummy_audit(event_type, data):
        audit_events.append((event_type, data))

    monkeypatch.setattr(retrieve, "get_logger", lambda: dummy_logger)
    monkeypatch.setattr(retrieve, "audit", dummy_audit)

    return retrieve, dummy_logger, audit_events


class TestRetrieveValidation:
    """Validation tests for retrieve."""

    def test_empty_query_raises(self, retrieve_module):
        retrieve, _, _ = retrieve_module
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retrieve.retrieve("", DummyCollection())

    def test_whitespace_query_raises(self, retrieve_module):
        retrieve, _, _ = retrieve_module
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retrieve.retrieve("   ", DummyCollection())

    def test_invalid_k_raises(self, retrieve_module):
        retrieve, _, _ = retrieve_module
        with pytest.raises(ValueError, match="k must be >= 1"):
            retrieve.retrieve("test", DummyCollection(), k=0)


class TestRetrieveSuccess:
    """Successful retrieval tests."""

    def test_retrieve_returns_chunks_and_metadata(self, retrieve_module):
        retrieve, logger, audit_events = retrieve_module
        collection = DummyCollection()

        chunks, metadata = retrieve.retrieve("test query", collection, k=3)

        assert chunks == ["chunk1", "chunk2"]
        assert len(metadata) == 2
        # Metadata should have retrieval_method field added by hybrid search
        assert all("retrieval_method" in m for m in metadata)

        # Logger should record info
        assert any("Retrieved" in msg or "retrieval" in msg.lower() for msg in logger.infos)

        # Audit should record retrieval with hybrid search fields
        retrieve_events = [e for e in audit_events if e[0] == "retrieve"]
        assert len(retrieve_events) > 0
        # Check that audit event has expected fields
        audit_data = retrieve_events[0][1]
        assert "query_length" in audit_data
        assert "retrieved_count" in audit_data

        # Chroma query called for vector search
        assert len(collection.queries) > 0

    def test_retrieve_handles_empty_results(self, retrieve_module):
        retrieve, logger, audit_events = retrieve_module
        empty_collection = DummyCollection(
            results={"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        )

        chunks, metadata = retrieve.retrieve("test", empty_collection, k=2)

        assert chunks == []
        assert metadata == []
        assert any("Retrieved 0 chunks" in msg for msg in logger.infos)
        # Check retrieve audit event exists with basic fields
        retrieve_events = [e for e in audit_events if e[0] == "retrieve"]
        assert len(retrieve_events) > 0
        assert retrieve_events[0][1]["retrieved_count"] == 0

    def test_retrieve_respects_k(self, retrieve_module):
        retrieve, _, _ = retrieve_module
        collection = DummyCollection()

        retrieve.retrieve("q", collection, k=7)

        # Should have called query with k=7
        assert len(collection.queries) > 0
        assert collection.queries[0]["n_results"] == 7


class TestRetrieveErrors:
    """Error handling tests for retrieve."""

    def test_retrieve_handles_search_failures_gracefully(self, monkeypatch, retrieve_module):
        """Test that retrieve handles both vector and keyword search failures gracefully."""
        retrieve, logger, audit_events = retrieve_module

        def failing_query(*args, **kwargs):
            raise RuntimeError("Vector search failed")

        def failing_get(*args, **kwargs):
            raise RuntimeError("Keyword search failed")

        collection = DummyCollection()
        collection.query = failing_query
        collection.get = failing_get

        # With hybrid search, failures should be caught and logged, returning empty results
        chunks, metadata = retrieve.retrieve("test query", collection, k=1)

        # Should return empty results rather than raising
        assert chunks == []
        assert metadata == []

        # Logger should capture warnings about failures
        assert any(
            "warning" in msg.lower() or "fail" in msg.lower()
            for msg in logger.warnings + logger.errors + logger.debugs
        )

        # Should have regular retrieve audit event (not retrieve_error) since errors were handled
        retrieve_events = [e for e in audit_events if e[0] == "retrieve"]
        assert len(retrieve_events) > 0
        assert retrieve_events[0][1]["retrieved_count"] == 0


class TestCodeFilters:
    """Tests for code-specific filtering in retrieve."""

    def test_language_filter_passed_to_collection(self, retrieve_module):
        """Test that language_filter parameter is passed to _query_collection."""
        retrieve, _, _ = retrieve_module
        collection = DummyCollection()

        retrieve.retrieve("authentication", collection, language_filter="java", k=3)

        # Check that query was called with filters in where clause
        assert len(collection.queries) > 0
        query_call = collection.queries[0]
        assert query_call["where"] is not None
        where_clause = query_call["where"]
        assert get_where_value(where_clause, "language") == "java"

    def test_source_category_filter_passed_to_collection(self, retrieve_module):
        """Test that source_category_filter parameter is passed to _query_collection."""
        retrieve, _, _ = retrieve_module
        collection = DummyCollection()

        retrieve.retrieve("payment API", collection, source_category_filter="code", k=3)

        # Check that query was called with filters in where clause
        assert len(collection.queries) > 0
        query_call = collection.queries[0]
        assert query_call["where"] is not None
        where_clause = query_call["where"]
        assert get_where_value(where_clause, "source_category") == "code"

    def test_multiple_filters_combined(self, retrieve_module):
        """Test that multiple filters are combined properly."""
        retrieve, _, _ = retrieve_module
        collection = DummyCollection()

        retrieve.retrieve(
            "payment service",
            collection,
            language_filter="java",
            source_category_filter="code",
            k=3,
        )

        # Check that query was called with both filters
        assert len(collection.queries) > 0
        query_call = collection.queries[0]
        where_clause = query_call["where"]
        assert get_where_value(where_clause, "language") == "java"
        assert get_where_value(where_clause, "source_category") == "code"
        # embedding_model should also be in there
        assert get_where_value(where_clause, "embedding_model") is not None

    def test_language_filter_case_insensitive(self, retrieve_module):
        """Test that language filter is normalised to lowercase."""
        retrieve, _, _ = retrieve_module
        collection = DummyCollection()

        retrieve.retrieve("auth", collection, language_filter="JAVA", k=3)

        # Check that filter was converted to lowercase
        query_call = collection.queries[0]
        where_clause = query_call["where"]
        assert get_where_value(where_clause, "language") == "java"

    def test_keyword_search_receives_filters(self, retrieve_module, monkeypatch):
        """Test that filters are passed to keyword search."""
        retrieve, logger, audit_events = retrieve_module

        # Disable BM25Retriever so we test the fallback keyword search
        monkeypatch.setattr("scripts.rag.retrieve.BM25Retriever", None)

        # Mock RAGConfig to ensure hybrid search is enabled
        class MockRAGConfig:
            enable_hybrid_search = True
            hybrid_combination_strategy = "sum"

        # Patch RAGConfig at its actual import location
        monkeypatch.setattr("scripts.rag.rag_config.RAGConfig", MockRAGConfig)

        # Mock _embed_query to return 1024D embedding
        def mock_embed_query(query: str, model_name: str):
            return [0.1] * 1024

        monkeypatch.setattr("scripts.rag.retrieve._embed_query", mock_embed_query)

        # Create a collection that returns results from keyword search
        collection = DummyCollection(
            results={
                "ids": [[]],  # Empty vector results
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }
        )

        # Override get to return keyword results and record calls
        original_get = collection.get
        get_calls = []

        def mock_get(where=None, limit=None, include=None, ids=None):
            get_calls.append({"where": where, "limit": limit, "include": include, "ids": ids})
            # If querying by ids (BM25 path), return empty
            if ids:
                return {"ids": [], "documents": [], "metadatas": []}
            # If checking for embeddings (where=None, include=['embeddings']), return with embeddings
            if include and "embeddings" in include:
                return {
                    "ids": ["dummy"],
                    "documents": ["dummy"],
                    "metadatas": [{}],
                    "embeddings": [[0.1] * 1024],
                }
            # Otherwise (keyword fallback path), return java results
            return {
                "ids": ["id1"],
                "documents": ["auth chunk"],
                "metadatas": [{"language": "java", "source_category": "code"}],
            }

        collection.get = mock_get

        chunks, metadata = retrieve.retrieve(
            "authentication", collection, language_filter="java", k=3
        )

        # Verify get was called (keyword search fallback)
        assert len(get_calls) > 0, f"Expected get calls, but got {len(get_calls)}"

        # Find the call from keyword search (should have where clause with language filter)
        keyword_calls = [
            c for c in get_calls if c.get("where") and c["where"].get("language") == "java"
        ]
        assert (
            len(keyword_calls) > 0
        ), f"Expected keyword search call with language filter, all calls: {get_calls}"

        keyword_call = keyword_calls[0]
        where_clause = keyword_call["where"]
        assert where_clause is not None, f"where_clause should not be None"
        assert (
            get_where_value(where_clause, "language") == "java"
        ), f"Expected language='java', got {where_clause}"
        assert (
            get_where_value(where_clause, "chunk_type") == "child"
        ), f"Expected chunk_type='child', got {where_clause}"


class TestDetectFiltersFromQuery:
    """Tests for detect_filters_from_query function."""

    def test_detect_java_language(self, retrieve_module):
        retrieve, _, _ = retrieve_module

        filters = retrieve.detect_filters_from_query("Show me Java services")
        assert filters.get("language") == "java"
        assert filters.get("source_category") == "code"

    def test_detect_groovy_language(self, retrieve_module):
        retrieve, _, _ = retrieve_module

        filters = retrieve.detect_filters_from_query("Where are the Groovy gradle builds?")
        assert filters.get("language") == "groovy"
        assert filters.get("source_category") == "code"

    def test_detect_python_language(self, retrieve_module):
        retrieve, _, _ = retrieve_module

        filters = retrieve.detect_filters_from_query("Show Python Flask endpoints")
        assert filters.get("language") == "python"
        assert filters.get("source_category") == "code"

    def test_detect_api_code_query(self, retrieve_module):
        retrieve, _, _ = retrieve_module

        filters = retrieve.detect_filters_from_query("Find REST API endpoints")
        assert filters.get("source_category") == "code"

    def test_detect_service_dependency_query(self, retrieve_module):
        retrieve, _, _ = retrieve_module

        filters = retrieve.detect_filters_from_query("Which microservices depend on this library?")
        assert filters.get("source_category") == "code"

    def test_no_false_positives_on_governance(self, retrieve_module):
        retrieve, _, _ = retrieve_module

        filters = retrieve.detect_filters_from_query("What are the security policies?")
        # Should detect governance, not code
        assert filters.get("source_category") != "code"
        # or at least not auto-detect as code if no language is mentioned

    def test_combined_language_and_service_detection(self, retrieve_module):
        retrieve, _, _ = retrieve_module

        filters = retrieve.detect_filters_from_query("Which Java services use Spring?")
        assert filters.get("language") == "java"
        assert filters.get("source_category") == "code"


class TestBuildCodeFilters:
    """Tests for build_code_filters helper function."""

    def test_build_filters_with_language(self, retrieve_module):
        retrieve, _, _ = retrieve_module

        filters = retrieve.build_code_filters(language="java")
        assert filters.get("source_category") == "code"
        assert filters.get("language") == "java"

    def test_build_filters_with_dependencies(self, retrieve_module):
        retrieve, _, _ = retrieve_module

        filters = retrieve.build_code_filters(include_dependencies=True)
        assert filters.get("source_category") == "code"
        assert filters.get("has_dependencies") is True

    def test_build_filters_with_endpoints(self, retrieve_module):
        retrieve, _, _ = retrieve_module

        filters = retrieve.build_code_filters(include_endpoints=True)
        assert filters.get("source_category") == "code"
        assert filters.get("has_endpoints") is True

    def test_build_filters_with_services(self, retrieve_module):
        retrieve, _, _ = retrieve_module

        filters = retrieve.build_code_filters(include_services=True)
        assert filters.get("source_category") == "code"
        assert filters.get("is_service") is True

    def test_build_filters_combined_parameters(self, retrieve_module):
        retrieve, _, _ = retrieve_module

        filters = retrieve.build_code_filters(
            language="groovy", include_dependencies=True, include_endpoints=True
        )
        assert filters.get("language") == "groovy"
        assert filters.get("has_dependencies") is True
        assert filters.get("has_endpoints") is True
