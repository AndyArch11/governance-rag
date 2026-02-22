"""Test that retrieve.py uses BM25Retriever when available and falls back to simple search."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.ingest.cache_db import CacheDB
from scripts.rag.retrieve import _bm25_search_with_fallback
from scripts.search.bm25_retrieval import BM25Retriever


def test_bm25_search_uses_pre_built_index():
    """Test that BM25 search uses pre-built index when available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a BM25 index
        cache_db = CacheDB(rag_data_path=Path(tmpdir), enable_cache=True)
        cache_db.put_bm25_document("doc1", {"python": 3, "programming": 2}, 5)
        cache_db.put_bm25_document("doc2", {"java": 2, "programming": 1}, 3)
        cache_db.update_bm25_corpus_stats(2)
        cache_db.close()

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["doc1"],
            "documents": ["Python programming tutorial"],
            "metadatas": [{"source": "test.txt"}],
        }

        # Mock RAGConfig to use tmpdir - patch where it's imported
        with patch("scripts.search.bm25_retrieval.Path") as mock_path:
            # Make Path() return our tmpdir
            mock_path.return_value = Path(tmpdir)

            # Also patch in the _bm25_search_with_fallback function
            with patch("scripts.rag.rag_config.RAGConfig") as mock_config_class:
                mock_config = MagicMock()
                mock_config.rag_data_path = tmpdir
                mock_config_class.return_value = mock_config

                # Perform search
                chunks, metadatas, chunk_ids = _bm25_search_with_fallback(
                    "python programming", mock_collection, k=5
                )

                # Should have results
                assert len(chunks) > 0
                assert chunks[0] == "Python programming tutorial"
                # Should have BM25 score in metadata
                assert "bm25_score" in metadatas[0]


def test_bm25_search_falls_back_when_no_index():
    """Test that BM25 search falls back to simple keyword search when no index exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Don't create any BM25 index - empty database

        # Mock collection for fallback search
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["chunk1", "chunk2"],
            "documents": ["Python is great for programming", "Java is also good"],
            "metadatas": [{"source": "doc1.txt"}, {"source": "doc2.txt"}],
        }

        # Mock RAGConfig to use tmpdir - patch where it's imported
        with patch("scripts.rag.rag_config.RAGConfig") as mock_config_class:
            mock_config = MagicMock()
            mock_config.rag_data_path = tmpdir
            mock_config_class.return_value = mock_config

            # Perform search
            chunks, metadatas, chunk_ids = _bm25_search_with_fallback(
                "python programming", mock_collection, k=5
            )

            # Should still get results from fallback
            assert len(chunks) > 0
            # Fallback doesn't add bm25_score
            assert "bm25_score" not in metadatas[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
