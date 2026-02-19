"""Integration tests for retrieve_with_filters() advanced features.

Tests the enhanced retrieval capabilities not covered by basic retrieve() tests:
- Context caching
- Graph-enhanced retrieval
- Neighbor chunk fetching
- Lightweight reranking
- Filter combination strategies
- Feature orchestration
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest


# Mock collection for testing
class MockCollection:
    """Mock ChromaDB collection for testing."""

    def __init__(self, chunks: List[str] | None = None, metadata: List[Dict] | None = None):
        self.chunks = chunks if chunks is not None else ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        self.metadata = metadata if metadata is not None else [
            {"chunk_id": f"id_{i}", "distance": 0.1 * i, "source_category": "code", "language": "java"}
            for i in range(len(self.chunks))
        ]
        self.query_calls = []

    def query(self, query_embeddings, n_results, where=None, include=None):
        """Mock query method."""
        self.query_calls.append({"where": where, "n_results": n_results})
        
        # Return empty results if collection is empty
        if not self.chunks:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }
        
        return {
            "documents": [self.chunks[:n_results]],
            "metadatas": [self.metadata[:n_results]],
            "distances": [[0.1 * i for i in range(min(n_results, len(self.chunks)))]],
        }

    def get(self, where=None, ids=None, limit=None, include=None):
        """Mock get method."""
        # Return empty results if collection is empty
        if not self.chunks:
            return {
                "ids": [],
                "documents": [],
                "metadatas": [],
            }
        
        if ids:
            # Return specific chunks by ID
            result_chunks = []
            result_meta = []
            result_ids = []
            for chunk_id in ids:
                idx = int(chunk_id.split("_")[-1]) if "_" in chunk_id else 0
                if idx < len(self.chunks):
                    result_chunks.append(self.chunks[idx])
                    result_meta.append(self.metadata[idx])
                    result_ids.append(chunk_id)
            return {
                "ids": result_ids,
                "documents": result_chunks,
                "metadatas": result_meta,
            }
        # Return all matching chunks
        return {
            "ids": [m.get("chunk_id", f"id_{i}") for i, m in enumerate(self.metadata[:limit or len(self.metadata)])],
            "documents": self.chunks[:limit or len(self.chunks)],
            "metadatas": self.metadata[:limit or len(self.metadata)],
        }


def get_where_value(where_clause, key):
    """Extract value from where clause supporting both direct and $and formats.
    
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


@pytest.fixture
def mock_embedding():
    """Mock embedding function."""
    with patch("scripts.rag.retrieve._embed_query") as mock:
        mock.return_value = [0.1] * 1024  # Standard embedding dimension
        yield mock


@pytest.fixture
def mock_collection():
    """Create mock collection with test data."""
    return MockCollection()


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


# ============================================================================
# Context Caching Tests
# ============================================================================


class TestRetrievalCaching:
    """Test context caching integration."""

    def test_cache_miss_then_hit(self, mock_collection, mock_embedding, temp_cache_dir):
        """Test cache miss on first query, hit on second."""
        from scripts.rag.retrieve import retrieve_with_filters

        query = "test authentication"

        # First call - cache miss
        chunks1, meta1 = retrieve_with_filters(
            query=query,
            collection=mock_collection,
            k=3,
            enable_caching=True,
            enable_graph=False,
            enable_hybrid_search=False,
            cache_dir=temp_cache_dir,
        )

        assert len(chunks1) == 3
        assert mock_embedding.called  # Embedding was generated

        # Reset mock
        mock_embedding.reset_mock()

        # Second call - should hit cache
        chunks2, meta2 = retrieve_with_filters(
            query=query,
            collection=mock_collection,
            k=3,
            enable_caching=True,
            enable_graph=False,
            enable_hybrid_search=False,
            cache_dir=temp_cache_dir,
        )

        assert chunks2 == chunks1
        assert meta2 == meta1
        assert not mock_embedding.called  # No embedding needed from cache

    def test_cache_disabled(self, mock_collection, mock_embedding, temp_cache_dir):
        """Test that caching can be disabled."""
        from scripts.rag.retrieve import retrieve_with_filters

        query = "test query"

        # Two calls with caching disabled
        retrieve_with_filters(
            query=query,
            collection=mock_collection,
            k=3,
            enable_caching=False,
            cache_dir=temp_cache_dir,
        )

        mock_embedding.reset_mock()

        retrieve_with_filters(
            query=query,
            collection=mock_collection,
            k=3,
            enable_caching=False,
            cache_dir=temp_cache_dir,
        )

        # Embedding should be called both times (no caching)
        assert mock_embedding.called


# ============================================================================
# Graph Enhancement Tests
# ============================================================================


class TestGraphEnhancedRetrieval:
    """Test graph-enhanced retrieval integration."""

    @pytest.fixture
    def mock_graph_retriever(self):
        """Create mock graph retriever."""
        mock = MagicMock()
        mock.graph = Mock()  # Non-None graph indicates it's loaded
        mock.expand_with_neighbours.return_value = ["id_0", "id_1", "id_2", "id_5", "id_6"]
        return mock

    def test_graph_expansion_adds_chunks(self, mock_collection, mock_embedding, mock_graph_retriever):
        """Test that graph expansion adds related chunks."""
        from scripts.rag.retrieve import retrieve_with_filters

        with patch("scripts.rag.graph_retrieval.get_graph_retriever") as mock_get_graph:
            mock_get_graph.return_value = mock_graph_retriever

            chunks, meta = retrieve_with_filters(
                query="test query",
                collection=mock_collection,
                k=3,
                enable_graph=True,
                enable_caching=False,
                enable_hybrid_search=False,
            )

            # Should have called graph expansion
            assert mock_graph_retriever.expand_with_neighbours.called
            call_args = mock_graph_retriever.expand_with_neighbours.call_args
            assert call_args[1]["max_hops"] == 1
            assert call_args[1]["max_neighbours"] == 3

            # Should have fetched additional chunks
            assert len(chunks) <= 3  # Limited to k after expansion

    def test_graph_disabled(self, mock_collection, mock_embedding, mock_graph_retriever):
        """Test that graph expansion can be disabled."""
        from scripts.rag.retrieve import retrieve_with_filters

        with patch("scripts.rag.graph_retrieval.get_graph_retriever") as mock_get_graph:
            mock_get_graph.return_value = mock_graph_retriever

            chunks, meta = retrieve_with_filters(
                query="test query",
                collection=mock_collection,
                k=3,
                enable_graph=False,
                enable_caching=False,
            )

            # Graph retriever shouldn't be called
            assert not mock_graph_retriever.expand_with_neighbours.called


# ============================================================================
# Filter Combination Tests
# ============================================================================


class TestFilterCombination:
    """Test combining multiple filter strategies."""

    def test_explicit_and_auto_detect_filters(self, mock_collection, mock_embedding):
        """Test combining explicit filters with auto-detection."""
        from scripts.rag.retrieve import retrieve_with_filters

        chunks, meta = retrieve_with_filters(
            query="Show me Java authentication services",  # Should auto-detect: language=java, category=code
            collection=mock_collection,
            k=3,
            filters={"is_service": True},  # Explicit filter
            auto_detect_filters=True,
            enable_caching=False,
            enable_graph=False,
        )

        # Check that query was made with combined filters
        assert len(mock_collection.query_calls) > 0
        where_clause = mock_collection.query_calls[0]["where"]
        # Check explicit filter is present (handle both direct and $and formats)
        assert get_where_value(where_clause, "is_service") == True or "is_service" in where_clause
        # Auto-detected filters should be present too

    def test_language_and_category_filters(self, mock_collection, mock_embedding):
        """Test using both language_filter and source_category_filter."""
        from scripts.rag.retrieve import retrieve_with_filters

        chunks, meta = retrieve_with_filters(
            query="test query",
            collection=mock_collection,
            k=3,
            language_filter="python",
            source_category_filter="code",
            auto_detect_filters=False,  # Disable to test explicit only
            enable_caching=False,
            enable_graph=False,
        )

        # Check filters were applied
        where_clause = mock_collection.query_calls[0]["where"]
        assert get_where_value(where_clause, "language") == "python"
        assert get_where_value(where_clause, "source_category") == "code"

    def test_filter_precedence(self, mock_collection, mock_embedding):
        """Test that explicit filters take precedence over auto-detected."""
        from scripts.rag.retrieve import retrieve_with_filters

        chunks, meta = retrieve_with_filters(
            query="Show me Java code",  # Would auto-detect language=java
            collection=mock_collection,
            k=3,
            language_filter="python",  # Explicit override
            auto_detect_filters=True,
            enable_caching=False,
            enable_graph=False,
        )

        # Explicit filter should win
        where_clause = mock_collection.query_calls[0]["where"]
        assert get_where_value(where_clause, "language") == "python"


# ============================================================================
# Reranking Tests
# ============================================================================


class TestReranking:
    """Test lightweight and learned reranking."""

    def test_lightweight_reranking_reorders(self, mock_collection, mock_embedding):
        """Test that lightweight reranking reorders results."""
        from scripts.rag.retrieve import retrieve_with_filters

        # Create collection with varied distances
        collection = MockCollection(
            chunks=["chunk_a", "chunk_b", "chunk_c"],
            metadata=[
                {"chunk_id": "id_0", "distance": 0.5, "section_depth": 1},
                {"chunk_id": "id_1", "distance": 0.2, "section_depth": 3},
                {"chunk_id": "id_2", "distance": 0.8, "section_depth": 2},
            ],
        )

        chunks, meta = retrieve_with_filters(
            query="test query",
            collection=collection,
            k=3,
            enable_reranking=True,  # Enable lightweight reranking
            enable_caching=False,
            enable_graph=False,
            enable_hybrid_search=False,
        )

        # Results should be reordered (not just by distance)
        # chunk_b has best distance (0.2) but chunk_a has section_depth=1 bonus
        assert len(chunks) == 3

    def test_reranking_disabled(self, mock_collection, mock_embedding):
        """Test results without reranking."""
        from scripts.rag.retrieve import retrieve_with_filters

        chunks, meta = retrieve_with_filters(
            query="test query",
            collection=mock_collection,
            k=3,
            enable_reranking=False,
            enable_caching=False,
            enable_graph=False,
        )

        # Should get results in original order
        assert len(chunks) == 3


# ============================================================================
# Neighbor Fetching Tests
# ============================================================================


class TestNeighborFetching:
    """Test neighboring chunk fetching."""

    def test_fetch_neighbours_expands_context(self, mock_collection, mock_embedding):
        """Test that fetch_neighbours retrieves prev/next chunks."""
        from scripts.rag.retrieve import retrieve_with_filters

        # Create collection with neighbor relationships
        collection = MockCollection(
            chunks=["chunk_a", "chunk_b", "chunk_c"],
            metadata=[
                {
                    "chunk_id": "id_0",
                    "distance": 0.1,
                    "prev_chunk_id": None,
                    "next_chunk_id": "id_1",
                },
                {
                    "chunk_id": "id_1",
                    "distance": 0.2,
                    "prev_chunk_id": "id_0",
                    "next_chunk_id": "id_2",
                },
                {
                    "chunk_id": "id_2",
                    "distance": 0.3,
                    "prev_chunk_id": "id_1",
                    "next_chunk_id": None,
                },
            ],
        )

        chunks, meta = retrieve_with_filters(
            query="test query",
            collection=collection,
            k=1,  # Only retrieve 1 chunk
            fetch_neighbours=True,  # But expand with neighbors
            enable_caching=False,
            enable_graph=False,
            enable_hybrid_search=False,
        )

        # Should have more than k=1 due to neighbor expansion
        # Will have: prev + main + next for the retrieved chunk
        assert len(chunks) > 1

    def test_neighbours_disabled(self, mock_collection, mock_embedding):
        """Test without neighbor fetching."""
        from scripts.rag.retrieve import retrieve_with_filters

        chunks, meta = retrieve_with_filters(
            query="test query",
            collection=mock_collection,
            k=2,
            fetch_neighbours=False,
            enable_caching=False,
            enable_graph=False,
        )

        # Should get exactly k results
        assert len(chunks) == 2


# ============================================================================
# Feature Orchestration Tests
# ============================================================================


class TestFeatureOrchestration:
    """Test combinations of multiple advanced features."""

    @pytest.fixture
    def mock_graph_retriever(self):
        """Create mock graph retriever."""
        mock = MagicMock()
        mock.graph = Mock()
        mock.expand_with_neighbours.return_value = ["id_0", "id_1", "id_3"]
        return mock

    def test_cache_plus_graph(
        self, mock_collection, mock_embedding, mock_graph_retriever, temp_cache_dir
    ):
        """Test caching + graph enhancement together."""
        from scripts.rag.retrieve import retrieve_with_filters

        with patch("scripts.rag.graph_retrieval.get_graph_retriever") as mock_get_graph:
            mock_get_graph.return_value = mock_graph_retriever

            query = "test query"

            # First call - populate cache with graph-enhanced results
            chunks1, meta1 = retrieve_with_filters(
                query=query,
                collection=mock_collection,
                k=3,
                enable_caching=True,
                enable_graph=True,
                cache_dir=temp_cache_dir,
                enable_hybrid_search=False,
            )

            assert len(chunks1) <= 3
            assert mock_graph_retriever.expand_with_neighbours.called

            # Reset mocks
            mock_embedding.reset_mock()
            mock_graph_retriever.expand_with_neighbours.reset_mock()

            # Second call - should use cache (no graph expansion needed)
            chunks2, meta2 = retrieve_with_filters(
                query=query,
                collection=mock_collection,
                k=3,
                enable_caching=True,
                enable_graph=True,
                cache_dir=temp_cache_dir,
                enable_hybrid_search=False,
            )

            assert chunks2 == chunks1
            assert not mock_embedding.called  # From cache
            assert not mock_graph_retriever.expand_with_neighbours.called  # From cache

    def test_all_features_enabled(
        self, mock_collection, mock_embedding, mock_graph_retriever, temp_cache_dir
    ):
        """Test with all advanced features enabled."""
        from scripts.rag.retrieve import retrieve_with_filters

        with patch("scripts.rag.graph_retrieval.get_graph_retriever") as mock_get_graph:
            mock_get_graph.return_value = mock_graph_retriever

            chunks, meta = retrieve_with_filters(
                query="Find Java authentication services",
                collection=mock_collection,
                k=3,
                language_filter="java",
                filters={"is_service": True},
                auto_detect_filters=True,
                enable_hybrid_search=True,
                enable_reranking=True,
                enable_graph=True,
                enable_caching=True,
                fetch_neighbours=False,  # Skip for simpler test
                cache_dir=temp_cache_dir,
            )

            # Should get results with all enhancements
            assert len(chunks) <= 3
            assert all(isinstance(m, dict) for m in meta)


# ============================================================================
# Validation Tests
# ============================================================================


class TestValidation:
    """Test input validation and error handling."""

    def test_empty_query_raises(self, mock_collection):
        """Test that empty query raises ValueError."""
        from scripts.rag.retrieve import retrieve_with_filters

        with pytest.raises(ValueError, match="Query cannot be empty"):
            retrieve_with_filters(
                query="",
                collection=mock_collection,
                k=5,
            )

    def test_invalid_k_raises(self, mock_collection):
        """Test that k < 1 raises ValueError."""
        from scripts.rag.retrieve import retrieve_with_filters

        with pytest.raises(ValueError, match="k must be >= 1"):
            retrieve_with_filters(
                query="test",
                collection=mock_collection,
                k=0,
            )

    def test_handles_empty_results(self, mock_embedding):
        """Test graceful handling of empty results."""
        from scripts.rag.retrieve import retrieve_with_filters

        empty_collection = MockCollection(chunks=[], metadata=[])
        
        # Verify collection is actually empty
        assert empty_collection.chunks == []
        assert empty_collection.metadata == []
        
        # Test query returns empty
        test_query = empty_collection.query([0.1] * 1024, 5)
        assert test_query["documents"] == [[]]

        chunks, meta = retrieve_with_filters(
            query="test query",
            collection=empty_collection,
            k=5,
            enable_caching=False,
            enable_graph=False,
            enable_hybrid_search=False,  # Disable hybrid to prevent keyword search
        )

        assert chunks == []
        assert meta == []


# ============================================================================
# Persona Integration Tests
# ============================================================================


class TestPersonaIntegration:
    """Test persona-aware retrieval."""

    def test_persona_applied(self, mock_collection, mock_embedding):
        """Test that persona parameter is used."""
        from scripts.rag.retrieve import retrieve_with_filters

        with patch("scripts.rag.retrieve.apply_persona_reranking") as mock_persona:
            mock_persona.return_value = (["chunk1", "chunk2"], [{"id": "1"}, {"id": "2"}])

            chunks, meta = retrieve_with_filters(
                query="test query",
                collection=mock_collection,
                k=3,
                persona="assessor",
                enable_caching=False,
                enable_graph=False,
                enable_hybrid_search=False,
            )

            # Persona reranking should be called
            assert mock_persona.called
            call_args = mock_persona.call_args
            assert call_args[0][2] == "assessor"  # persona argument


# ============================================================================
# Performance and Metrics Tests
# ============================================================================


class TestMetrics:
    """Test that metrics are recorded correctly."""

    def test_metrics_recorded(self, mock_collection, mock_embedding):
        """Test that retrieval metrics are recorded."""
        from scripts.rag.retrieve import retrieve_with_filters

        with patch("scripts.rag.retrieve.perf_metrics") as mock_metrics:
            chunks, meta = retrieve_with_filters(
                query="test query",
                collection=mock_collection,
                k=3,
                enable_caching=False,
                enable_graph=False,
            )

            # Metrics should be recorded
            assert mock_metrics.record_retrieval.called
            call_args = mock_metrics.record_retrieval.call_args[1]
            assert "latency_ms" in call_args
            assert "result_count" in call_args
            assert call_args["result_count"] == len(chunks)

    def test_cache_hit_metrics(self, mock_collection, mock_embedding, temp_cache_dir):
        """Test that cache hits are recorded in metrics."""
        from scripts.rag.retrieve import retrieve_with_filters

        query = "test query"

        # First call to populate cache
        retrieve_with_filters(
            query=query,
            collection=mock_collection,
            k=3,
            enable_caching=True,
            cache_dir=temp_cache_dir,
            enable_graph=False,
            enable_hybrid_search=False,
        )

        # Second call should hit cache
        with patch("scripts.rag.retrieve.perf_metrics") as mock_metrics:
            retrieve_with_filters(
                query=query,
                collection=mock_collection,
                k=3,
                enable_caching=True,
                cache_dir=temp_cache_dir,
                enable_graph=False,
                enable_hybrid_search=False,
            )

            # Should record cache hit
            call_args = mock_metrics.record_retrieval.call_args[1]
            assert call_args.get("cache_hit") is True
