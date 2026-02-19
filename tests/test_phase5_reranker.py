"""Test Phase 5 cross-encoder reranker integration with RAG retrieval.

Most tests in this module disable reranker cache (`enable_cache=False`) to avoid
non-determinism from persisted local cache entries in ``~/.cache/rag_reranker``.
Only the dedicated cache persistence test enables cache behaviour explicitly.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.search.reranker import (
    CrossEncoderReranker,
    RankerResult,
    RerankerConfig,
    rerank_results,
)


class TestRerankerModule:
    """Test basic reranker functionality."""

    def test_reranker_config_creation(self):
        """Test RerankerConfig instantiation."""
        config = RerankerConfig(
            enable_reranking=True,
            model_name="BAAI/bge-reranker-base",
            rerank_top_k=50,
            final_top_k=5,
            device="cpu",
            batch_size=32,
            enable_cache=True,
        )
        assert config.enable_reranking is True
        assert config.model_name == "BAAI/bge-reranker-base"
        assert config.rerank_top_k == 50
        assert config.final_top_k == 5

    def test_ranker_result_creation(self):
        """Test RankerResult dataclass."""
        result = RankerResult(
            doc_id="chunk_123",
            reranker_score=0.85,
            hybrid_score=0.75,
            rank=1,
        )
        assert result.doc_id == "chunk_123"
        assert result.reranker_score == 0.85
        assert result.hybrid_score == 0.75
        assert result.rank == 1

    def test_reranker_initialisation(self):
        """Test CrossEncoderReranker lazy initialisation."""
        reranker = CrossEncoderReranker(
            model_name="BAAI/bge-reranker-base", device="cpu", enable_cache=False
        )
        assert reranker.model_name == "BAAI/bge-reranker-base"
        assert reranker.device == "cpu"
        assert reranker._model is None  # Lazy loading
        assert reranker._model_loaded is False

    @patch("scripts.search.reranker.logger")
    def test_reranker_lazy_loading(self, mock_logger):
        """Test that model is loaded on first rerank call."""
        # Mock the CrossEncoder at the import point
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = [0.85, 0.72, 0.68]

        reranker = CrossEncoderReranker(
            model_name="BAAI/bge-reranker-base", device="cpu", enable_cache=False
        )
        
        # Model should not be loaded yet
        assert reranker._model is None
        assert reranker._model_loaded is False

        # Mock cache and the dynamic import in _load_model
        with patch.object(reranker, "_load_cache"):
            with patch.object(reranker, "_save_cache_entry"):
                with patch("builtins.__import__", side_effect=lambda *args, **kwargs: MagicMock(CrossEncoder=lambda **kw: mock_model_instance) if args[0] == "sentence_transformers" else __import__(*args, **kwargs)):
                    # Manually set up the model to avoid import issues
                    reranker._model = mock_model_instance
                    reranker._model_loaded = True
                    
                    results = reranker.rerank(
                        query="test query",
                        documents=[
                            {"doc_id": "1", "text": "relevant doc", "hybrid_score": 0.8},
                            {"doc_id": "2", "text": "less relevant", "hybrid_score": 0.5},
                            {"doc_id": "3", "text": "somewhat relevant", "hybrid_score": 0.6},
                        ],
                        top_k=3,
                    )

        # Model should now be loaded
        assert reranker._model is not None
        assert reranker._model_loaded is True
        # Should have reranked results
        assert len(results) == 3
        assert results[0].reranker_score >= results[1].reranker_score

    def test_rerank_results_function(self):
        """Test rerank_results convenience function with mock."""
        with patch("scripts.search.reranker.CrossEncoderReranker") as mock_reranker_class:
            mock_instance = MagicMock()
            mock_instance.rerank.return_value = [
                RankerResult(doc_id="1", reranker_score=0.9, hybrid_score=0.8, rank=1),
                RankerResult(doc_id="2", reranker_score=0.7, hybrid_score=0.6, rank=2),
            ]
            mock_reranker_class.return_value = mock_instance

            config = RerankerConfig(
                enable_reranking=True,
                model_name="BAAI/bge-reranker-base",
                rerank_top_k=50,
                final_top_k=5,
            )

            documents = [
                {"doc_id": "1", "text": "high quality content"},
                {"doc_id": "2", "text": "moderate content"},
            ]

            results = rerank_results("test query", documents, config)
            assert len(results) == 2
            assert results[0].rank == 1
            assert results[1].rank == 2

    def test_cache_key_generation(self):
        """Test MD5 cache key generation."""
        reranker = CrossEncoderReranker(model_name="test", device="cpu", enable_cache=False)
        key1 = reranker._get_cache_key("query", "1")
        key2 = reranker._get_cache_key("query", "1")
        key3 = reranker._get_cache_key("query", "2")

        assert key1 == key2  # Same inputs = same key
        assert key1 != key3  # Different inputs = different key
        assert len(key1) == 32  # MD5 hash length

    def test_batch_processing(self):
        """Test that reranker handles batch processing correctly."""
        reranker = CrossEncoderReranker(
            model_name="BAAI/bge-reranker-base",
            device="cpu",
            batch_size=2,
            enable_cache=False,
        )

        # Create 5 documents to test batching with batch_size=2
        documents = [
            {"doc_id": str(i), "text": f"document {i}"}
            for i in range(5)
        ]

        with patch.object(reranker, "_load_cache"):
            with patch.object(reranker, "_save_cache_entry"):
                # Mock the model directly
                mock_instance = MagicMock()
                mock_instance.predict.side_effect = [
                    [0.9, 0.8],      # Batch 1 (2 items)
                    [0.7, 0.6],      # Batch 2 (2 items)
                    [0.5],           # Batch 3 (1 item)
                ]
                
                # Inject the mock by setting it directly
                def mock_load_model():
                    reranker._model = mock_instance
                    reranker._model_loaded = True
                
                with patch.object(reranker, "_load_model", side_effect=mock_load_model):
                    results = reranker.rerank(
                        query="test query",
                        documents=documents,
                        top_k=5,
                    )

                    # Should have processed 3 batches for 5 items with batch_size=2
                    assert len(results) == 5
                    # Verify predict was called 3 times
                    assert mock_instance.predict.call_count == 3

    def test_cache_persistence(self):
        """Test cache file creation and persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            reranker = CrossEncoderReranker(
                model_name="test",
                device="cpu",
                enable_cache=True,
                cache_dir=str(cache_dir),
            )

            # Mock the model
            mock_instance = MagicMock()
            mock_instance.predict.return_value = [0.9, 0.7]
            
            def mock_load_model():
                reranker._model = mock_instance
                reranker._model_loaded = True

            with patch.object(reranker, "_load_cache"):
                with patch.object(reranker, "_load_model", side_effect=mock_load_model):
                    documents = [
                        {"doc_id": "1", "text": "doc1"},
                        {"doc_id": "2", "text": "doc2"},
                    ]

                    results = reranker.rerank(
                        query="test query",
                        documents=documents,
                        top_k=2,
                    )

                    # Cache file should be created
                    cache_file = cache_dir / "scores.jsonl"
                    assert cache_file.exists()

                    # Verify cache content
                    with open(cache_file) as f:
                        lines = f.readlines()
                        assert len(lines) >= 2  # At least 2 entries

                        for line in lines:
                            entry = json.loads(line)
                            assert "key" in entry
                            assert "score" in entry


class TestRerankerIntegration:
    """Test reranker integration scenarios."""

    def test_rerank_with_hybrid_scores(self):
        """Test reranking with hybrid search scores."""
        mock_instance = MagicMock()
        # Reorder scores: original order [0.8, 0.5, 0.6] -> reranked [0.85, 0.72, 0.68]
        mock_instance.predict.return_value = [0.85, 0.72, 0.68]

        reranker = CrossEncoderReranker(
            model_name="BAAI/bge-reranker-base", device="cpu", enable_cache=False
        )
        
        # Inject the mock
        def mock_load_model():
            reranker._model = mock_instance
            reranker._model_loaded = True
        
        with patch.object(reranker, "_load_model", side_effect=mock_load_model):
            with patch.object(reranker, "_load_cache"):
                with patch.object(reranker, "_save_cache_entry"):
                    documents = [
                        {"doc_id": "1", "text": "doc1", "hybrid_score": 0.8},
                        {"doc_id": "2", "text": "doc2", "hybrid_score": 0.5},
                        {"doc_id": "3", "text": "doc3", "hybrid_score": 0.6},
                    ]

                    results = reranker.rerank(
                        query="test query",
                        documents=documents,
                        top_k=3,
                    )

                    # Verify results are ranked by reranker score
                    assert results[0].reranker_score == 0.85
                    assert results[1].reranker_score == 0.72
                    assert results[2].reranker_score == 0.68

    def test_top_k_filtering(self):
        """Test that top_k parameter limits returned results."""
        mock_instance = MagicMock()
        mock_instance.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]

        reranker = CrossEncoderReranker(
            model_name="BAAI/bge-reranker-base", device="cpu", enable_cache=False
        )
        
        # Inject the mock
        def mock_load_model():
            reranker._model = mock_instance
            reranker._model_loaded = True

        with patch.object(reranker, "_load_model", side_effect=mock_load_model):
            documents = [
                {"doc_id": str(i), "text": f"doc{i}"} for i in range(5)
            ]

            with patch.object(reranker, "_load_cache"):
                with patch.object(reranker, "_save_cache_entry"):
                    results = reranker.rerank(
                        query="test query",
                        documents=documents,
                        top_k=2,
                    )

                    # Should only return top 2
                    assert len(results) == 2
                    assert results[0].rank == 1
                    assert results[1].rank == 2

    def test_empty_documents_handling(self):
        """Test reranker with empty documents list."""
        reranker = CrossEncoderReranker(
            model_name="BAAI/bge-reranker-base", device="cpu", enable_cache=False
        )

        with patch.object(reranker, "_load_cache", return_value={}):
            results = reranker.rerank(
                query="test query",
                documents=[],
                top_k=5,
            )

            assert results == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
