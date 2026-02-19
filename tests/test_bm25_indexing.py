"""Unit tests for BM25 chunk-level indexing utility.

Tests the common index_chunks_in_bm25() function used across all three ingest modules.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from scripts.ingest.bm25_indexing import index_chunks_in_bm25


class TestIndexChunksInBm25:
    """Test chunk-level BM25 indexing."""

    def test_index_regular_chunks(self):
        """Test indexing regular chunks with chunk-level IDs."""
        # Mock dependencies
        mock_cache_db = Mock()
        mock_logger = Mock()
        mock_config = Mock(bm25_index_original_text=False)

        chunks = ["chunk text 1", "chunk text 2"]
        doc_id = "test_doc"

        with patch("scripts.ingest.bm25_indexing.BM25Search") as mock_bm25:
            mock_bm25_instance = Mock()
            mock_bm25.return_value = mock_bm25_instance
            mock_bm25_instance.tokenise.side_effect = [
                ["chunk", "text", "1"],
                ["chunk", "text", "2"],
            ]

            total = index_chunks_in_bm25(
                doc_id=doc_id,
                chunks=chunks,
                config=mock_config,
                cache_db=mock_cache_db,
                logger=mock_logger,
            )

        # Verify correct number of chunks indexed
        assert total == 2

        # Verify cache_db was called twice with correct IDs
        assert mock_cache_db.put_bm25_document.call_count == 2
        calls = mock_cache_db.put_bm25_document.call_args_list
        assert calls[0][1]["doc_id"] == "test_doc-chunk-0"
        assert calls[1][1]["doc_id"] == "test_doc-chunk-1"

    def test_index_child_chunks(self):
        """Test indexing child chunks with child IDs."""
        mock_cache_db = Mock()
        mock_logger = Mock()
        mock_config = Mock(bm25_index_original_text=False)

        chunks = []
        child_chunks = [
            {"id": "child-0", "text": "child text 1"},
            {"id": "child-1", "text": "child text 2"},
        ]
        doc_id = "test_doc"

        with patch("scripts.ingest.bm25_indexing.BM25Search") as mock_bm25:
            mock_bm25_instance = Mock()
            mock_bm25.return_value = mock_bm25_instance
            mock_bm25_instance.tokenise.side_effect = [
                ["child", "text", "1"],
                ["child", "text", "2"],
            ]

            total = index_chunks_in_bm25(
                doc_id=doc_id,
                chunks=chunks,
                child_chunks=child_chunks,
                config=mock_config,
                cache_db=mock_cache_db,
                logger=mock_logger,
            )

        # Verify correct number of chunks indexed
        assert total == 2

        # Verify cache_db was called with correct child IDs
        calls = mock_cache_db.put_bm25_document.call_args_list
        assert calls[0][1]["doc_id"] == "test_doc-child-0"
        assert calls[1][1]["doc_id"] == "test_doc-child-1"

    def test_index_parent_chunks(self):
        """Test indexing parent chunks with parent IDs."""
        mock_cache_db = Mock()
        mock_logger = Mock()
        mock_config = Mock(bm25_index_original_text=False)

        chunks = []
        parent_chunks = [
            {"id": "parent-0", "text": "parent text 1", "child_ids": ["c1", "c2"]},
        ]
        doc_id = "test_doc"

        with patch("scripts.ingest.bm25_indexing.BM25Search") as mock_bm25:
            mock_bm25_instance = Mock()
            mock_bm25.return_value = mock_bm25_instance
            mock_bm25_instance.tokenise.return_value = ["parent", "text", "1"]

            total = index_chunks_in_bm25(
                doc_id=doc_id,
                chunks=chunks,
                parent_chunks=parent_chunks,
                config=mock_config,
                cache_db=mock_cache_db,
                logger=mock_logger,
            )

        # Verify correct number of chunks indexed
        assert total == 1

        # Verify cache_db was called with correct parent ID
        calls = mock_cache_db.put_bm25_document.call_args_list
        assert calls[0][1]["doc_id"] == "test_doc-parent-0"

    def test_index_mixed_chunk_types(self):
        """Test indexing regular, child, and parent chunks together."""
        mock_cache_db = Mock()
        mock_logger = Mock()
        mock_config = Mock(bm25_index_original_text=False)

        chunks = ["regular chunk"]
        child_chunks = [{"id": "child-0", "text": "child chunk"}]
        parent_chunks = [{"id": "parent-0", "text": "parent chunk", "child_ids": []}]
        doc_id = "test_doc"

        with patch("scripts.ingest.bm25_indexing.BM25Search") as mock_bm25:
            mock_bm25_instance = Mock()
            mock_bm25.return_value = mock_bm25_instance
            mock_bm25_instance.tokenise.side_effect = [
                ["regular"],
                ["child"],
                ["parent"],
            ]

            total = index_chunks_in_bm25(
                doc_id=doc_id,
                chunks=chunks,
                child_chunks=child_chunks,
                parent_chunks=parent_chunks,
                config=mock_config,
                cache_db=mock_cache_db,
                logger=mock_logger,
            )

        # Verify all chunks are indexed
        assert total == 3
        assert mock_cache_db.put_bm25_document.call_count == 3

    def test_missing_required_parameters(self):
        """Test that ValueError is raised for missing required parameters."""
        with pytest.raises(ValueError, match="doc_id and chunks are required"):
            index_chunks_in_bm25(doc_id="", chunks=[])

        with pytest.raises(ValueError, match="doc_id and chunks are required"):
            index_chunks_in_bm25(doc_id="test", chunks=None)

    def test_indexing_with_cache_db_auto_creation(self):
        """Test that cache_db is auto-created if not provided."""
        mock_logger = Mock()
        mock_config = Mock(bm25_index_original_text=False)

        chunks = ["test chunk"]
        doc_id = "test_doc"

        with patch(
            "scripts.ingest.bm25_indexing.get_cache_client"
        ) as mock_get_cache, patch(
            "scripts.ingest.bm25_indexing.BM25Search"
        ) as mock_bm25:
            mock_cache_db = Mock()
            mock_get_cache.return_value = mock_cache_db

            mock_bm25_instance = Mock()
            mock_bm25.return_value = mock_bm25_instance
            mock_bm25_instance.tokenise.return_value = ["test"]

            total = index_chunks_in_bm25(
                doc_id=doc_id,
                chunks=chunks,
                config=mock_config,
                logger=mock_logger,
            )

            # Verify cache_db was created
            mock_get_cache.assert_called_once_with(enable_cache=True)
            assert total == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
