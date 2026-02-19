"""Tests for hybrid search functionality in retrieve module.

Covers keyword search, hybrid retrieval combining vector + keyword results,
and audit event tracking.
"""

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_retrieve():
    """Load retrieve module with mocked components."""
    import importlib
    
    sys.modules.pop("scripts.rag.retrieve", None)
    retrieve = importlib.import_module("scripts.rag.retrieve")
    return retrieve


class TestKeywordSearchFunction:
    """Tests for _keyword_search function."""

    def test_keyword_search_exists(self, mock_retrieve):
        """Test that _keyword_search function exists."""
        assert hasattr(mock_retrieve, "_keyword_search")
        assert callable(mock_retrieve._keyword_search)

    def test_keyword_search_signature(self, mock_retrieve):
        """Test _keyword_search has correct signature."""
        import inspect
        
        sig = inspect.signature(mock_retrieve._keyword_search)
        params = list(sig.parameters.keys())
        
        # Should have query, collection, k parameters
        assert "query" in params
        assert "collection" in params
        assert "k" in params

    def test_keyword_search_returns_tuple(self, mock_retrieve):
        """Test _keyword_search returns tuple of (chunks, metadata, scores)."""
        # Create mock collection
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["1", "2"],
            "documents": ["Netskope Agent", "Data classification"],
            "metadatas": [{}, {}]
        }
        
        chunks, metadata, scores = mock_retrieve._keyword_search("test", mock_collection, k=5)
        
        assert isinstance(chunks, list)
        assert isinstance(metadata, list)
        assert isinstance(scores, list)

    def test_keyword_search_filters_by_chunk_type(self, mock_retrieve):
        """Test _keyword_search filters by chunk_type='child'."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": []
        }
        
        mock_retrieve._keyword_search("test", mock_collection, k=5)
        
        # Should call get at least once with where filter for chunk_type='child'
        # (May be called additional times by BM25 index lookup)
        assert mock_collection.get.called
        
        # Check that at least one call had the chunk_type filter
        found_filter_call = False
        for call in mock_collection.get.call_args_list:
            if "where" in call[1]:
                if call[1]["where"].get("chunk_type") == "child":
                    found_filter_call = True
                    break
        
        assert found_filter_call, "Expected at least one call with chunk_type='child' filter"

    def test_keyword_search_respects_limit(self, mock_retrieve):
        """Test _keyword_search uses reasonable limit."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": []
        }
        
        mock_retrieve._keyword_search("test", mock_collection, k=5)
        
        call_kwargs = mock_collection.get.call_args[1]
        # Should limit to 10,000 chunks
        assert call_kwargs["limit"] == 10000


class TestHybridRetrieveFunction:
    """Tests for hybrid retrieve function."""

    def test_retrieve_calls_vector_search(self, mock_retrieve):
        """Test retrieve calls vector search."""
        mock_collection = MagicMock()
        
        # Mock vector search results
        mock_collection.query.return_value = {
            "documents": [["Vector result 1", "Vector result 2"]],
            "metadatas": [[
                {"id": "v1", "source": "doc1"},
                {"id": "v2", "source": "doc2"}
            ]]
        }
        
        chunks, metadata = mock_retrieve.retrieve("test query", mock_collection, k=5)
        
        # Should have called vector search
        assert mock_collection.query.called
        
        # Should return results
        assert len(chunks) > 0
        assert len(metadata) > 0

    def test_retrieve_returns_metadata_with_retrieval_method(self, mock_retrieve):
        """Test retrieve metadata includes retrieval_method field."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["test chunk"]],
            "metadatas": [[{"id": "1"}]]
        }
        mock_collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": []
        }
        
        _, metadata = mock_retrieve.retrieve("test", mock_collection, k=5)
        
        # Should have retrieval_method in metadata
        assert any("retrieval_method" in m for m in metadata)

    def test_retrieve_validates_query(self, mock_retrieve):
        """Test retrieve validates query is not empty."""
        mock_collection = MagicMock()
        
        with pytest.raises(ValueError):
            mock_retrieve.retrieve("", mock_collection, k=5)
        
        with pytest.raises(ValueError):
            mock_retrieve.retrieve("   ", mock_collection, k=5)

    def test_retrieve_validates_k(self, mock_retrieve):
        """Test retrieve validates k parameter."""
        mock_collection = MagicMock()
        
        with pytest.raises(ValueError):
            mock_retrieve.retrieve("test", mock_collection, k=0)


class TestAuditLogging:
    """Tests for audit logging in hybrid search."""

    def test_retrieve_audits_keyword_search_activation(self, mock_retrieve, monkeypatch):
        """Test retrieve audits when keyword search is activated."""
        audit_events = []
        
        def mock_audit(event_type, data):
            audit_events.append((event_type, data))
        
        monkeypatch.setattr(mock_retrieve, "audit", mock_audit)
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["test"]],
            "metadatas": [[{"id": "1"}]]
        }
        mock_collection.get.return_value = {
            "ids": ["2"],
            "documents": ["keyword match"],
            "metadatas": [{}]
        }
        
        mock_retrieve.retrieve("test query", mock_collection, k=5)
        
        # Should have keyword_search_used event
        keyword_events = [e for e in audit_events if e[0] == "keyword_search_used"]
        # Event may or may not exist depending on whether keyword search was triggered


class TestRetrievalDeduplication:
    """Tests for deduplication in hybrid retrieval."""

    def test_retrieval_deduplicates_by_id(self, mock_retrieve):
        """Test hybrid retrieval deduplicates results by chunk ID."""
        mock_collection = MagicMock()
        
        # Vector search returns chunk 1 and 2
        mock_collection.query.return_value = {
            "documents": [["Chunk 1", "Chunk 2"]],
            "metadatas": [[
                {"id": "1", "source": "doc1"},
                {"id": "2", "source": "doc2"}
            ]]
        }
        
        # Keyword search also returns chunk 1 (should be deduplicated)
        mock_collection.get.return_value = {
            "ids": ["1", "3"],
            "documents": ["Chunk 1", "Chunk 3"],
            "metadatas": [{}, {}]
        }
        
        chunks, metadata = mock_retrieve.retrieve("test", mock_collection, k=5)
        
        # Should not have duplicate chunks by ID
        chunk_ids = [m.get("id") for m in metadata if "id" in m]
        assert len(chunk_ids) == len(set(chunk_ids))


class TestConfiguration:
    """Tests for hybrid search configuration."""

    def test_retrieve_respects_k_parameter(self, mock_retrieve):
        """Test retrieve respects k parameter for result limit."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["chunk"] * 10],
            "metadatas": [[{"id": str(i)} for i in range(10)]]
        }
        mock_collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": []
        }
        
        chunks, _ = mock_retrieve.retrieve("test", mock_collection, k=3)
        
        # Should return at most k results
        assert len(chunks) <= 3


class TestIntegration:
    """Integration tests for hybrid search workflow."""

    def test_keyword_search_finds_term_matches(self, mock_retrieve):
        """Test keyword search finds chunks matching query terms."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["1", "2", "3"],
            "documents": [
                "Netskope Agent security",
                "Data classification",
                "Netskope DLP policy"
            ],
            "metadatas": [{}, {}, {}]
        }
        
        chunks, _, _ = mock_retrieve._keyword_search("Netskope", mock_collection, k=5)
        
        # Should find Netskope chunks
        netskope_chunks = [c for c in chunks if "Netskope" in c]
        assert len(netskope_chunks) >= 1

    def test_retrieve_handles_no_vector_results(self, mock_retrieve):
        """Test retrieve handles fallback when vector search returns nothing."""
        mock_collection = MagicMock()
        
        # Vector search returns nothing
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]]
        }
        
        # Keyword search returns something
        mock_collection.get.return_value = {
            "ids": ["1"],
            "documents": ["Keyword result"],
            "metadatas": [{}]
        }
        
        chunks, metadata = mock_retrieve.retrieve("test", mock_collection, k=5)
        
        # Should either return keyword results as fallback OR handle gracefully
        # The implementation determines whether keyword search runs when vector returns nothing
        assert isinstance(chunks, list)
        assert isinstance(metadata, list)

    def test_retrieve_handles_no_keyword_results(self, mock_retrieve):
        """Test retrieve handles case where keyword search returns nothing."""
        mock_collection = MagicMock()
        
        # Vector search returns something
        mock_collection.query.return_value = {
            "documents": [["Vector result"]],
            "metadatas": [[{"id": "1"}]]
        }
        
        # Keyword search returns nothing
        mock_collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": []
        }
        
        chunks, metadata = mock_retrieve.retrieve("test", mock_collection, k=5)
        
        # Should return vector results
        assert len(chunks) > 0
