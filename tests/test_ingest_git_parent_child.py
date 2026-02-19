"""Tests for parent-child chunking in ingest_git.py

Tests parent-child chunk hierarchy functionality for maintaining relationships
between chunks and providing better context during RAG retrieval.
"""

import hashlib
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call
from typing import List, Dict, Any

from scripts.ingest.git.git_ingest_config import GitIngestConfig


@pytest.fixture
def mock_config():
    """Create mock GitIngestConfig with parent-child enabled."""
    config = MagicMock(spec=GitIngestConfig)
    config.enable_parent_child_chunking = True
    config.enable_dlp = False
    config.generate_summaries = False
    config.bm25_indexing_enabled = False
    config.git_provider = "github"
    config.git_host = "https://github.com"
    config.github_host = "https://github.com"
    config.bitbucket_host = ""
    return config


@pytest.fixture
def mock_collections():
    """Create mock ChromaDB collections."""
    chunk_collection = MagicMock()
    doc_collection = MagicMock()
    chunk_collection.get.return_value = {"ids": [], "metadatas": []}
    return chunk_collection, doc_collection


@pytest.fixture
def mock_parser():
    """Create mock CodeParser."""
    parser = MagicMock()
    mock_result = MagicMock()
    mock_result.language = "python"
    mock_result.file_type = "source"
    mock_result.to_dict.return_value = {
        "language": "python",
        "file_type": "source",
        "service_name": "test_service",
        "exports": ["function1", "function2"],
        "endpoints": [],
        "external_dependencies": ["os", "sys"]
    }
    parser.parse_file.return_value = mock_result
    return parser


@pytest.fixture
def sample_code():
    """Sample Python code for testing."""
    return """
def example_function():
    '''This is a docstring.'''
    x = 1
    y = 2
    return x + y

class ExampleClass:
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1
        return self.value

# Additional content to create multiple chunks
def another_function():
    pass
""" * 10  # Repeat to ensure enough content for multiple chunks


class TestParentChildChunkingEnabled:
    """Tests for parent-child chunking when enabled."""
    
    def test_create_parent_child_chunks_called_when_enabled(
        self, mock_config, mock_collections, mock_parser, sample_code
    ):
        """Test that create_parent_child_chunks is called when enabled."""
        from scripts.ingest.ingest_git import process_code_file
        
        chunk_collection, doc_collection = mock_collections
        
        # Mock create_parent_child_chunks
        child_chunks = [
            {"id": "child1", "text": "chunk 1", "parent_id": "parent1"},
            {"id": "child2", "text": "chunk 2", "parent_id": "parent1"}
        ]
        parent_chunks = [
            {"id": "parent1", "text": "parent chunk 1", "child_ids": ["child1", "child2"]}
        ]
        
        with patch("scripts.ingest.ingest_git.create_parent_child_chunks") as mock_create, \
             patch("scripts.ingest.ingest_git.chunk_text") as mock_chunk, \
             patch("scripts.ingest.ingest_git.store_chunks_in_chroma"), \
             patch("scripts.ingest.ingest_git.store_child_chunks") as mock_child_store, \
             patch("scripts.ingest.ingest_git.store_parent_chunks") as mock_parent_store:
            
            mock_create.return_value = (child_chunks, parent_chunks)
            mock_chunk.return_value = ["regular chunk"]
            
            result = process_code_file(
                file_path="test.py",
                file_content=sample_code,
                project_key="TEST",
                repo_slug="test-repo",                branch="main",                chunk_collection=chunk_collection,
                doc_collection=doc_collection,
                parser=mock_parser,
                llm_cache=None,
                embedding_cache=None,
                config=mock_config,
            )
            
            assert result is True
            mock_create.assert_called_once()
            assert mock_create.call_args[1]["text"] == sample_code
            assert mock_create.call_args[1]["doc_type"] == "python"
    
    def test_child_chunks_stored_before_parent_chunks(
        self, mock_config, mock_collections, mock_parser, sample_code
    ):
        """Test that child chunks are stored BEFORE parent chunks (critical for ChromaDB dimension)."""
        from scripts.ingest.ingest_git import process_code_file
        
        chunk_collection, doc_collection = mock_collections
        
        child_chunks = [
            {"id": "child1", "text": "chunk 1", "parent_id": "parent1"}
        ]
        parent_chunks = [
            {"id": "parent1", "text": "parent chunk 1", "child_ids": ["child1"]}
        ]
        
        call_order = []
        
        def track_child_store(*args, **kwargs):
            call_order.append("child")
        
        def track_parent_store(*args, **kwargs):
            call_order.append("parent")
        
        with patch("scripts.ingest.ingest_git.create_parent_child_chunks") as mock_create, \
             patch("scripts.ingest.ingest_git.chunk_text") as mock_chunk, \
             patch("scripts.ingest.ingest_git.store_chunks_in_chroma"), \
             patch("scripts.ingest.ingest_git.store_child_chunks", side_effect=track_child_store), \
             patch("scripts.ingest.ingest_git.store_parent_chunks", side_effect=track_parent_store):
            
            mock_create.return_value = (child_chunks, parent_chunks)
            mock_chunk.return_value = ["regular chunk"]
            
            process_code_file(
                file_path="test.py",
                file_content=sample_code,
                project_key="TEST",
                repo_slug="test-repo",
                branch="main",
                chunk_collection=chunk_collection,
                doc_collection=doc_collection,
                parser=mock_parser,
                llm_cache=None,
                embedding_cache=None,
                config=mock_config,
            )
            
            # Verify child stored before parent
            assert call_order == ["child", "parent"]
    
    def test_store_child_chunks_receives_correct_parameters(
        self, mock_config, mock_collections, mock_parser, sample_code
    ):
        """Test that store_child_chunks is called with correct parameters."""
        from scripts.ingest.ingest_git import process_code_file
        
        chunk_collection, doc_collection = mock_collections
        
        child_chunks = [
            {"id": "child1", "text": "chunk 1", "parent_id": "parent1"}
        ]
        parent_chunks = [
            {"id": "parent1", "text": "parent chunk 1", "child_ids": ["child1"]}
        ]
        
        with patch("scripts.ingest.ingest_git.create_parent_child_chunks") as mock_create, \
             patch("scripts.ingest.ingest_git.chunk_text") as mock_chunk, \
             patch("scripts.ingest.ingest_git.store_chunks_in_chroma"), \
             patch("scripts.ingest.ingest_git.store_child_chunks") as mock_child_store, \
             patch("scripts.ingest.ingest_git.store_parent_chunks"):
            
            mock_create.return_value = (child_chunks, parent_chunks)
            mock_chunk.return_value = ["regular chunk"]
            
            process_code_file(
                file_path="test.py",
                file_content=sample_code,
                project_key="TEST",
                repo_slug="test-repo",
                branch="main",
                chunk_collection=chunk_collection,
                doc_collection=doc_collection,
                parser=mock_parser,
                llm_cache=None,
                embedding_cache=None,
                config=mock_config,
            )
            
            mock_child_store.assert_called_once()
            call_kwargs = mock_child_store.call_args[1]
            
            assert call_kwargs["doc_id"] == "TEST_test_repo_test_py"
            assert call_kwargs["child_chunks"] == child_chunks
            assert call_kwargs["chunk_collection"] == chunk_collection
            assert call_kwargs["dry_run"] is False
            assert "base_metadata" in call_kwargs
    
    def test_store_parent_chunks_receives_correct_parameters(
        self, mock_config, mock_collections, mock_parser, sample_code
    ):
        """Test that store_parent_chunks is called with correct parameters."""
        from scripts.ingest.ingest_git import process_code_file
        
        chunk_collection, doc_collection = mock_collections
        
        child_chunks = [
            {"id": "child1", "text": "chunk 1", "parent_id": "parent1"}
        ]
        parent_chunks = [
            {"id": "parent1", "text": "parent chunk 1", "child_ids": ["child1"]}
        ]
        
        with patch("scripts.ingest.ingest_git.create_parent_child_chunks") as mock_create, \
             patch("scripts.ingest.ingest_git.chunk_text") as mock_chunk, \
             patch("scripts.ingest.ingest_git.store_chunks_in_chroma"), \
             patch("scripts.ingest.ingest_git.store_child_chunks"), \
             patch("scripts.ingest.ingest_git.store_parent_chunks") as mock_parent_store:
            
            mock_create.return_value = (child_chunks, parent_chunks)
            mock_chunk.return_value = ["regular chunk"]
            
            process_code_file(
                file_path="test.py",
                file_content=sample_code,
                project_key="TEST",
                repo_slug="test-repo",
                branch="main",
                chunk_collection=chunk_collection,
                doc_collection=doc_collection,
                parser=mock_parser,
                llm_cache=None,
                embedding_cache=None,
                config=mock_config,
            )
            
            mock_parent_store.assert_called_once()
            call_kwargs = mock_parent_store.call_args[1]
            
            assert call_kwargs["doc_id"] == "TEST_test_repo_test_py"
            assert call_kwargs["parent_chunks"] == parent_chunks
            assert call_kwargs["chunk_collection"] == chunk_collection
            assert call_kwargs["dry_run"] is False
            assert "base_metadata" in call_kwargs
    
    def test_base_metadata_contains_required_fields(
        self, mock_config, mock_collections, mock_parser, sample_code
    ):
        """Test that base_metadata contains all required fields."""
        from scripts.ingest.ingest_git import process_code_file
        
        chunk_collection, doc_collection = mock_collections
        
        child_chunks = [{"id": "child1", "text": "chunk 1", "parent_id": "parent1"}]
        parent_chunks = [{"id": "parent1", "text": "parent chunk 1", "child_ids": ["child1"]}]
        
        with patch("scripts.ingest.ingest_git.create_parent_child_chunks") as mock_create, \
             patch("scripts.ingest.ingest_git.chunk_text") as mock_chunk, \
             patch("scripts.ingest.ingest_git.store_chunks_in_chroma"), \
             patch("scripts.ingest.ingest_git.store_child_chunks") as mock_child_store, \
             patch("scripts.ingest.ingest_git.store_parent_chunks"):
            
            mock_create.return_value = (child_chunks, parent_chunks)
            mock_chunk.return_value = ["regular chunk"]
            
            process_code_file(
                file_path="test.py",
                file_content=sample_code,
                project_key="TEST",
                repo_slug="test-repo",
                branch="main",
                chunk_collection=chunk_collection,
                doc_collection=doc_collection,
                parser=mock_parser,
                llm_cache=None,
                embedding_cache=None,
                config=mock_config,
            )
            
            call_kwargs = mock_child_store.call_args[1]
            base_metadata = call_kwargs["base_metadata"]
            
            assert "doc_id" in base_metadata
            assert "source" in base_metadata
            assert "version" in base_metadata
            assert "hash" in base_metadata
            assert "doc_type" in base_metadata
            assert "embedding_model" in base_metadata
            
            assert base_metadata["doc_id"] == "TEST_test_repo_test_py"
            assert base_metadata["source"] == "test.py"
            assert base_metadata["version"] == 1
            assert base_metadata["doc_type"] == "python"


class TestParentChildChunkingDisabled:
    """Tests for parent-child chunking when disabled."""
    
    def test_parent_child_not_called_when_disabled(
        self, mock_collections, mock_parser, sample_code
    ):
        """Test that parent-child functions not called when disabled."""
        from scripts.ingest.ingest_git import process_code_file
        
        config = MagicMock(spec=GitIngestConfig)
        config.enable_parent_child_chunking = False
        config.enable_dlp = False
        config.generate_summaries = False
        config.bm25_indexing_enabled = False
        config.git_provider = "github"
        config.git_host = "https://github.com"
        config.github_host = "https://github.com"
        config.bitbucket_host = ""
        
        chunk_collection, doc_collection = mock_collections
        
        with patch("scripts.ingest.ingest_git.create_parent_child_chunks") as mock_create, \
             patch("scripts.ingest.ingest_git.chunk_text") as mock_chunk, \
             patch("scripts.ingest.ingest_git.store_chunks_in_chroma"), \
             patch("scripts.ingest.ingest_git.store_child_chunks") as mock_child_store, \
             patch("scripts.ingest.ingest_git.store_parent_chunks") as mock_parent_store:
            
            mock_chunk.return_value = ["regular chunk"]
            
            result = process_code_file(
                file_path="test.py",
                file_content=sample_code,
                project_key="TEST",
                repo_slug="test-repo",
                branch="main",
                chunk_collection=chunk_collection,
                doc_collection=doc_collection,
                parser=mock_parser,
                llm_cache=None,
                embedding_cache=None,
                config=config,
            )
            
            assert result is True
            mock_create.assert_not_called()
            mock_child_store.assert_not_called()
            mock_parent_store.assert_not_called()
    
    def test_regular_chunks_stored_when_disabled(
        self, mock_collections, mock_parser, sample_code
    ):
        """Test that regular chunks are stored when parent-child disabled."""
        from scripts.ingest.ingest_git import process_code_file
        
        config = MagicMock(spec=GitIngestConfig)
        config.enable_parent_child_chunking = False
        config.enable_dlp = False
        config.generate_summaries = False
        config.bm25_indexing_enabled = False
        config.git_provider = "github"
        config.git_host = "https://github.com"
        config.github_host = "https://github.com"
        config.bitbucket_host = ""
        
        chunk_collection, doc_collection = mock_collections
        
        with patch("scripts.ingest.ingest_git.chunk_text") as mock_chunk, \
             patch("scripts.ingest.ingest_git.store_chunks_in_chroma") as mock_store:
            
            regular_chunks = ["chunk1", "chunk2", "chunk3"]
            mock_chunk.return_value = regular_chunks
            
            process_code_file(
                file_path="test.py",
                file_content=sample_code,
                project_key="TEST",
                repo_slug="test-repo",
                branch="main",
                chunk_collection=chunk_collection,
                doc_collection=doc_collection,
                parser=mock_parser,
                llm_cache=None,
                embedding_cache=None,
                config=config,
            )
            
            mock_store.assert_called_once()
            call_kwargs = mock_store.call_args[1]
            assert call_kwargs["chunks"] == regular_chunks


class TestParentChildErrorHandling:
    """Tests for error handling in parent-child chunking."""
    
    def test_parent_child_creation_failure_logs_warning(
        self, mock_config, mock_collections, mock_parser, sample_code
    ):
        """Test that creation failure logs warning and continues."""
        from scripts.ingest.ingest_git import process_code_file
        
        chunk_collection, doc_collection = mock_collections
        
        with patch("scripts.ingest.ingest_git.create_parent_child_chunks") as mock_create, \
             patch("scripts.ingest.ingest_git.chunk_text") as mock_chunk, \
             patch("scripts.ingest.ingest_git.store_chunks_in_chroma"), \
             patch("scripts.ingest.ingest_git.logger") as mock_logger:
            
            mock_create.side_effect = Exception("Creation failed")
            mock_chunk.return_value = ["regular chunk"]
            
            result = process_code_file(
                file_path="test.py",
                file_content=sample_code,
                project_key="TEST",
                repo_slug="test-repo",
                branch="main",
                chunk_collection=chunk_collection,
                doc_collection=doc_collection,
                parser=mock_parser,
                llm_cache=None,
                embedding_cache=None,
                config=mock_config,
            )
            
            assert result is True
            # Check warning was logged
            warning_calls = [c for c in mock_logger.warning.call_args_list 
                           if "Parent-child chunking failed" in str(c)]
            assert len(warning_calls) > 0
    
    def test_child_storage_failure_raises_exception(
        self, mock_config, mock_collections, mock_parser, sample_code
    ):
        """Test that child chunk storage failure raises (critical path)."""
        from scripts.ingest.ingest_git import process_code_file
        
        chunk_collection, doc_collection = mock_collections
        
        child_chunks = [{"id": "child1", "text": "chunk 1", "parent_id": "parent1"}]
        parent_chunks = [{"id": "parent1", "text": "parent chunk 1", "child_ids": ["child1"]}]
        
        with patch("scripts.ingest.ingest_git.create_parent_child_chunks") as mock_create, \
             patch("scripts.ingest.ingest_git.chunk_text") as mock_chunk, \
             patch("scripts.ingest.ingest_git.store_chunks_in_chroma"), \
             patch("scripts.ingest.ingest_git.store_child_chunks") as mock_child_store, \
             patch("scripts.ingest.ingest_git.store_parent_chunks"), \
             patch("scripts.ingest.ingest_git.audit"):
            
            mock_create.return_value = (child_chunks, parent_chunks)
            mock_chunk.return_value = ["regular chunk"]
            mock_child_store.side_effect = Exception("Child storage failed")
            
            result = process_code_file(
                file_path="test.py",
                file_content=sample_code,
                project_key="TEST",
                repo_slug="test-repo",
                branch="main",
                chunk_collection=chunk_collection,
                doc_collection=doc_collection,
                parser=mock_parser,
                llm_cache=None,
                embedding_cache=None,
                config=mock_config,
            )
            
            # Child storage failure should cause overall failure
            assert result is False
    
    def test_parent_storage_failure_logs_but_continues(
        self, mock_config, mock_collections, mock_parser, sample_code
    ):
        """Test that parent chunk storage failure logs error but doesn't raise (non-fatal)."""
        from scripts.ingest.ingest_git import process_code_file
        
        chunk_collection, doc_collection = mock_collections
        
        child_chunks = [{"id": "child1", "text": "chunk 1", "parent_id": "parent1"}]
        parent_chunks = [{"id": "parent1", "text": "parent chunk 1", "child_ids": ["child1"]}]
        
        with patch("scripts.ingest.ingest_git.create_parent_child_chunks") as mock_create, \
             patch("scripts.ingest.ingest_git.chunk_text") as mock_chunk, \
             patch("scripts.ingest.ingest_git.store_chunks_in_chroma"), \
             patch("scripts.ingest.ingest_git.store_child_chunks"), \
             patch("scripts.ingest.ingest_git.store_parent_chunks") as mock_parent_store, \
             patch("scripts.ingest.ingest_git.logger") as mock_logger, \
             patch("scripts.ingest.ingest_git.audit"):
            
            mock_create.return_value = (child_chunks, parent_chunks)
            mock_chunk.return_value = ["regular chunk"]
            mock_parent_store.side_effect = Exception("Parent storage failed")
            
            result = process_code_file(
                file_path="test.py",
                file_content=sample_code,
                project_key="TEST",
                repo_slug="test-repo",
                branch="main",
                chunk_collection=chunk_collection,
                doc_collection=doc_collection,
                parser=mock_parser,
                llm_cache=None,
                embedding_cache=None,
                config=mock_config,
            )
            
            # Parent storage failure should not cause overall failure
            assert result is True
            # Check error was logged
            error_calls = [c for c in mock_logger.error.call_args_list 
                         if "Failed to store parent chunks" in str(c)]
            assert len(error_calls) > 0
    
    def test_audit_trail_for_child_storage_failure(
        self, mock_config, mock_collections, mock_parser, sample_code
    ):
        """Test that audit trail is recorded for child storage failure."""
        from scripts.ingest.ingest_git import process_code_file
        
        chunk_collection, doc_collection = mock_collections
        
        child_chunks = [{"id": "child1", "text": "chunk 1", "parent_id": "parent1"}]
        parent_chunks = [{"id": "parent1", "text": "parent chunk 1", "child_ids": ["child1"]}]
        
        with patch("scripts.ingest.ingest_git.create_parent_child_chunks") as mock_create, \
             patch("scripts.ingest.ingest_git.chunk_text") as mock_chunk, \
             patch("scripts.ingest.ingest_git.store_chunks_in_chroma"), \
             patch("scripts.ingest.ingest_git.store_child_chunks") as mock_child_store, \
             patch("scripts.ingest.ingest_git.store_parent_chunks"), \
             patch("scripts.ingest.ingest_git.audit") as mock_audit:
            
            mock_create.return_value = (child_chunks, parent_chunks)
            mock_chunk.return_value = ["regular chunk"]
            mock_child_store.side_effect = ValueError("Dimension mismatch")
            
            process_code_file(
                file_path="test.py",
                file_content=sample_code,
                project_key="TEST",
                repo_slug="test-repo",                branch="main",                chunk_collection=chunk_collection,
                doc_collection=doc_collection,
                parser=mock_parser,
                llm_cache=None,
                embedding_cache=None,
                config=mock_config,
            )
            
            # Verify audit was called for failure
            audit_calls = [c for c in mock_audit.call_args_list 
                         if c[0][0] == "child_chunks_storage_failed"]
            assert len(audit_calls) > 0
            audit_data = audit_calls[0][0][1]
            assert "error" in audit_data
            assert "error_type" in audit_data
            assert audit_data["error_type"] == "ValueError"


class TestParentChildIntegration:
    """Integration tests for parent-child chunking."""
    
    def test_empty_chunks_avoid_storage(
        self, mock_config, mock_collections, mock_parser
    ):
        """Test that empty child/parent lists don't trigger storage."""
        from scripts.ingest.ingest_git import process_code_file
        
        chunk_collection, doc_collection = mock_collections
        
        # Empty chunks
        child_chunks = []
        parent_chunks = []
        
        with patch("scripts.ingest.ingest_git.create_parent_child_chunks") as mock_create, \
             patch("scripts.ingest.ingest_git.chunk_text") as mock_chunk, \
             patch("scripts.ingest.ingest_git.store_chunks_in_chroma"), \
             patch("scripts.ingest.ingest_git.store_child_chunks") as mock_child_store, \
             patch("scripts.ingest.ingest_git.store_parent_chunks") as mock_parent_store:
            
            mock_create.return_value = (child_chunks, parent_chunks)
            mock_chunk.return_value = ["regular chunk"]
            
            process_code_file(
                file_path="test.py",
                file_content="short content",
                project_key="TEST",
                repo_slug="test-repo",                branch="main",                chunk_collection=chunk_collection,
                doc_collection=doc_collection,
                parser=mock_parser,
                llm_cache=None,
                embedding_cache=None,
                config=mock_config,
            )
            
            # Empty chunks should not trigger storage
            mock_child_store.assert_not_called()
            mock_parent_store.assert_not_called()
