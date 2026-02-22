"""Tests for retrieve module helper functions.

Covers _combine_results, _replace_children_with_parents, and _apply_learned_reranking
with focused, isolated test cases for improved maintainability and coverage.
"""

import sys
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest


class DummyLogger:
    """Minimal logger mock for tests."""

    def __init__(self):
        self.infos = []
        self.debugs = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg)

    def debug(self, msg, *args, **kwargs):
        self.debugs.append(msg)


@pytest.fixture()
def retrieve_helpers(monkeypatch):
    """Load retrieve module with mocked audit and logger."""
    sys.modules.pop("scripts.rag.retrieve", None)
    retrieve = __import__("scripts.rag.retrieve", fromlist=[""])

    audit_events = []

    def dummy_audit(event_type, data):
        audit_events.append((event_type, data))

    logger = DummyLogger()
    monkeypatch.setattr(retrieve, "audit", dummy_audit)
    monkeypatch.setattr(retrieve, "get_logger", lambda: logger)

    return retrieve, logger, audit_events


class TestCombineResults:
    """Tests for _combine_results helper."""

    def test_combine_empty_inputs(self, retrieve_helpers):
        """Combine with all empty inputs returns empty results."""
        retrieve, logger, _ = retrieve_helpers

        chunks, metadata, v_count, k_count = retrieve._combine_results(
            [], [], [], [], [], [], k=5, logger=logger, use_weights=False
        )

        assert chunks == []
        assert metadata == []
        assert v_count == 0
        assert k_count == 0

    def test_combine_counts_first(self, retrieve_helpers):
        """Counts results are prepended first."""
        retrieve, logger, _ = retrieve_helpers

        counts_chunks = ["count_summary"]
        counts_metadata = [{"type": "counts"}]
        vector_chunks = ["vector_result"]
        vector_metadata = [{"id": "v1"}]
        keyword_chunks = []
        keyword_metadata = []

        chunks, metadata, v_count, k_count = retrieve._combine_results(
            vector_chunks,
            vector_metadata,
            keyword_chunks,
            keyword_metadata,
            counts_chunks,
            counts_metadata,
            k=5,
            logger=logger,
            use_weights=False,
        )

        assert chunks[0] == "count_summary"
        assert chunks[1] == "vector_result"
        assert metadata[0]["type"] == "counts"
        assert metadata[1]["retrieval_method"] == "vector"
        assert v_count == 1
        assert k_count == 0

    def test_combine_vector_prioritised(self, retrieve_helpers):
        """Vector results appear after counts, before keywords."""
        retrieve, logger, _ = retrieve_helpers

        vector_chunks = ["vec1", "vec2"]
        vector_metadata = [{"id": "v1"}, {"id": "v2"}]
        keyword_chunks = ["kw1"]
        keyword_metadata = [{"id": "k1"}]

        chunks, metadata, v_count, k_count = retrieve._combine_results(
            vector_chunks,
            vector_metadata,
            keyword_chunks,
            keyword_metadata,
            [],
            [],
            k=5,
            logger=logger,
            use_weights=False,
        )

        # Vector results should come first
        assert chunks[0] == "vec1"
        assert chunks[1] == "vec2"
        assert chunks[2] == "kw1"
        assert v_count == 2
        assert k_count == 1

    def test_combine_deduplication(self, retrieve_helpers):
        """Duplicate chunks are only included once."""
        retrieve, logger, _ = retrieve_helpers

        shared_chunk = "duplicate_content"
        vector_chunks = [shared_chunk, "vec2"]
        vector_metadata = [{"id": "v1"}, {"id": "v2"}]
        keyword_chunks = [shared_chunk, "kw2"]
        keyword_metadata = [{"id": "k1"}, {"id": "k2"}]

        chunks, metadata, v_count, k_count = retrieve._combine_results(
            vector_chunks,
            vector_metadata,
            keyword_chunks,
            keyword_metadata,
            [],
            [],
            k=5,
            logger=logger,
            use_weights=False,
        )

        # Duplicate should not appear twice
        assert chunks.count(shared_chunk) == 1
        assert len(chunks) == 3  # shared, vec2, kw2
        assert v_count == 2
        assert k_count == 1

    def test_combine_respects_k_limit(self, retrieve_helpers):
        """Final result is trimmed to k."""
        retrieve, logger, _ = retrieve_helpers

        vector_chunks = ["v1", "v2", "v3"]
        vector_metadata = [{"id": f"v{i}"} for i in range(1, 4)]
        keyword_chunks = ["k1", "k2"]
        keyword_metadata = [{"id": f"k{i}"} for i in range(1, 3)]

        chunks, metadata, _, _ = retrieve._combine_results(
            vector_chunks,
            vector_metadata,
            keyword_chunks,
            keyword_metadata,
            [],
            [],
            k=3,
            logger=logger,
            use_weights=False,
        )

        assert len(chunks) == 3
        assert len(metadata) == 3

    def test_combine_logs_hybrid_breakdown(self, retrieve_helpers):
        """Logs breakdown when keyword results are included."""
        retrieve, logger, _ = retrieve_helpers

        vector_chunks = ["v1"]
        vector_metadata = [{"id": "v1"}]
        keyword_chunks = ["k1"]
        keyword_metadata = [{"id": "k1"}]

        retrieve._combine_results(
            vector_chunks,
            vector_metadata,
            keyword_chunks,
            keyword_metadata,
            [],
            [],
            k=5,
            logger=logger,
            use_weights=False,
        )

        assert any("Hybrid retrieval" in msg for msg in logger.infos)


class TestReplaceChildrenWithParents:
    """Tests for _replace_children_with_parents helper."""

    def test_parent_replacement_disabled(self, retrieve_helpers):
        """When disabled, no replacement happens."""
        retrieve, logger, _ = retrieve_helpers

        chunks = ["child1", "child2"]
        metadata = [{"id": "c1"}, {"id": "c2"}]
        collection = Mock()

        result_chunks, result_meta, count = retrieve._replace_children_with_parents(
            chunks, metadata, collection, enable_parent_child=False, logger=logger
        )

        assert result_chunks == chunks
        assert count == 0
        assert result_chunks[0] == "child1"

    def test_parent_replacement_no_batch_function(self, retrieve_helpers, monkeypatch):
        """When batch_get_parents_for_children is None, no replacement."""
        retrieve, logger, _ = retrieve_helpers
        monkeypatch.setattr(retrieve, "batch_get_parents_for_children", None)

        chunks = ["child1"]
        metadata = [{"id": "c1"}]
        collection = Mock()

        result_chunks, result_meta, count = retrieve._replace_children_with_parents(
            chunks, metadata, collection, enable_parent_child=True, logger=logger
        )

        assert result_chunks == chunks
        assert count == 0

    def test_parent_replacement_success(self, retrieve_helpers, monkeypatch):
        """Child chunks are replaced with parent chunks when available."""
        retrieve, logger, _ = retrieve_helpers

        chunks = ["child1", "child2"]
        metadata = [{"id": "c1", "chunk_id": "c1"}, {"id": "c2", "chunk_id": "c2"}]
        collection = Mock()

        # Mock batch_get_parents_for_children to return parent map
        parents_map = {
            "c1": {"text": "parent1_text", "id": "p1"},
            "c2": {"text": "parent2_text", "id": "p2"},
        }
        mock_batch_get = Mock(return_value=parents_map)
        monkeypatch.setattr(retrieve, "batch_get_parents_for_children", mock_batch_get)

        result_chunks, result_meta, count = retrieve._replace_children_with_parents(
            chunks, metadata, collection, enable_parent_child=True, logger=logger
        )

        # Chunks should be replaced
        assert result_chunks[0] == "parent1_text"
        assert result_chunks[1] == "parent2_text"
        assert count == 2
        # Metadata should be updated
        assert result_meta[0]["used_parent"] is True
        assert result_meta[0]["parent_id"] == "p1"
        assert result_meta[0]["original_child_id"] == "c1"

    def test_parent_replacement_partial(self, retrieve_helpers, monkeypatch):
        """Partial replacement when only some parents exist."""
        retrieve, logger, _ = retrieve_helpers

        chunks = ["child1", "child2"]
        metadata = [{"id": "c1", "chunk_id": "c1"}, {"id": "c2", "chunk_id": "c2"}]
        collection = Mock()

        # Only c1 has a parent
        parents_map = {
            "c1": {"text": "parent1_text", "id": "p1"},
        }
        mock_batch_get = Mock(return_value=parents_map)
        monkeypatch.setattr(retrieve, "batch_get_parents_for_children", mock_batch_get)

        result_chunks, result_meta, count = retrieve._replace_children_with_parents(
            chunks, metadata, collection, enable_parent_child=True, logger=logger
        )

        # Only first chunk replaced
        assert result_chunks[0] == "parent1_text"
        assert result_chunks[1] == "child2"
        assert count == 1

    def test_parent_replacement_error_handling(self, retrieve_helpers, monkeypatch):
        """Errors during parent fetch are caught and logged."""
        retrieve, logger, _ = retrieve_helpers

        chunks = ["child1"]
        metadata = [{"id": "c1"}]
        collection = Mock()

        # Mock to raise an error
        mock_batch_get = Mock(side_effect=RuntimeError("DB error"))
        monkeypatch.setattr(retrieve, "batch_get_parents_for_children", mock_batch_get)

        result_chunks, result_meta, count = retrieve._replace_children_with_parents(
            chunks, metadata, collection, enable_parent_child=True, logger=logger
        )

        # Should return original chunks and log debug message
        assert result_chunks == chunks
        assert count == 0
        assert any("skipped" in msg for msg in logger.debugs)


class TestApplyLearnedReranking:
    """Tests for _apply_learned_reranking helper."""

    def test_reranking_disabled_returns_originals(self, retrieve_helpers, monkeypatch):
        """When RerankerConfig is None, original results returned unchanged."""
        retrieve, logger, _ = retrieve_helpers
        monkeypatch.setattr(retrieve, "RerankerConfig", None)

        chunks = ["chunk1", "chunk2"]
        metadata = [{"id": "1"}, {"id": "2"}]

        result_chunks, result_meta = retrieve._apply_learned_reranking(
            chunks,
            metadata,
            "query",
            k=2,
            model_name="model",
            top_k=50,
            device="cpu",
            logger=logger,
        )

        assert result_chunks == chunks
        assert result_meta == metadata

    def test_reranking_empty_chunks_returns_empty(self, retrieve_helpers):
        """Empty chunks input returns empty output."""
        retrieve, logger, _ = retrieve_helpers

        result_chunks, result_meta = retrieve._apply_learned_reranking(
            [], [], "query", k=2, model_name="model", top_k=50, device="cpu", logger=logger
        )

        assert result_chunks == []
        assert result_meta == []

    def test_reranking_unavailable_returns_originals(self, retrieve_helpers, monkeypatch):
        """When rerank_results is None, originals returned."""
        retrieve, logger, _ = retrieve_helpers
        monkeypatch.setattr(retrieve, "rerank_results", None)
        # Ensure RerankerConfig exists so it doesn't short-circuit first
        retrieve.RerankerConfig = Mock()

        chunks = ["chunk1"]
        metadata = [{"id": "1"}]

        result_chunks, result_meta = retrieve._apply_learned_reranking(
            chunks,
            metadata,
            "query",
            k=1,
            model_name="model",
            top_k=50,
            device="cpu",
            logger=logger,
        )

        assert result_chunks == chunks

    def test_reranking_success(self, retrieve_helpers, monkeypatch):
        """Successful reranking reorders results."""
        retrieve, logger, audit_events = retrieve_helpers

        chunks = ["chunk1", "chunk2", "chunk3"]
        metadata = [
            {"id": "1", "distance": 0.1},
            {"id": "2", "distance": 0.2},
            {"id": "3", "distance": 0.3},
        ]

        # Mock RerankerConfig and rerank_results
        mock_config = Mock()
        retrieve.RerankerConfig = Mock(return_value=mock_config)

        # Create mock reranked results in reverse order
        mock_ranked_1 = Mock(doc_id="3")
        mock_ranked_2 = Mock(doc_id="2")
        mock_ranked_3 = Mock(doc_id="1")

        retrieve.rerank_results = Mock(return_value=[mock_ranked_1, mock_ranked_2, mock_ranked_3])

        result_chunks, result_meta = retrieve._apply_learned_reranking(
            chunks,
            metadata,
            "query",
            k=3,
            model_name="test_model",
            top_k=50,
            device="cpu",
            logger=logger,
        )

        # Results should be reordered by reranker
        assert result_chunks == ["chunk3", "chunk2", "chunk1"]
        assert result_meta[0]["id"] == "3"
        assert result_meta[1]["id"] == "2"
        assert result_meta[2]["id"] == "1"

        # Audit should be recorded
        assert any(e[0] == "retrieve_reranked" for e in audit_events)

    def test_reranking_error_returns_originals(self, retrieve_helpers, monkeypatch):
        """Errors during reranking are caught, originals returned."""
        retrieve, logger, _ = retrieve_helpers

        chunks = ["chunk1", "chunk2"]
        metadata = [{"id": "1"}, {"id": "2"}]

        # Mock to raise error during reranking
        mock_config = Mock()
        retrieve.RerankerConfig = Mock(return_value=mock_config)
        retrieve.rerank_results = Mock(side_effect=RuntimeError("Rerank failed"))

        result_chunks, result_meta = retrieve._apply_learned_reranking(
            chunks,
            metadata,
            "query",
            k=2,
            model_name="model",
            top_k=50,
            device="cpu",
            logger=logger,
        )

        # Should return originals and log debug
        assert result_chunks == chunks
        assert any("skipped" in msg for msg in logger.debugs)

    def test_reranking_preserves_unranked_results(self, retrieve_helpers, monkeypatch):
        """Results not in reranker output are filtered out."""
        retrieve, logger, _ = retrieve_helpers

        chunks = ["chunk1", "chunk2", "chunk3"]
        metadata = [{"id": "1"}, {"id": "2"}, {"id": "3"}]

        mock_config = Mock()
        retrieve.RerankerConfig = Mock(return_value=mock_config)

        # Reranker only returns 2 results
        mock_ranked_1 = Mock(doc_id="2")
        mock_ranked_2 = Mock(doc_id="1")
        retrieve.rerank_results = Mock(return_value=[mock_ranked_1, mock_ranked_2])

        result_chunks, result_meta = retrieve._apply_learned_reranking(
            chunks,
            metadata,
            "query",
            k=3,
            model_name="model",
            top_k=50,
            device="cpu",
            logger=logger,
        )

        # Only reranked results should be returned
        assert len(result_chunks) == 2
        assert result_chunks == ["chunk2", "chunk1"]

    def test_reranking_logs_success(self, retrieve_helpers, monkeypatch):
        """Successful reranking is logged."""
        retrieve, logger, _ = retrieve_helpers

        chunks = ["chunk1"]
        metadata = [{"id": "1"}]

        mock_config = Mock()
        retrieve.RerankerConfig = Mock(return_value=mock_config)
        mock_ranked = Mock(doc_id="1")
        retrieve.rerank_results = Mock(return_value=[mock_ranked])

        retrieve._apply_learned_reranking(
            chunks,
            metadata,
            "query",
            k=1,
            model_name="model",
            top_k=50,
            device="cpu",
            logger=logger,
        )

        assert any("reranking" in msg.lower() for msg in logger.infos)
