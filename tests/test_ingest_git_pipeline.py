"""Unit tests for unified Git ingestion pipeline (ingest_git.py).

Tests cover:
- Document ID computation
- File hash computation
- Code file ingestion pipeline
- Integration with ChromaDB
"""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

from scripts.ingest.ingest_git import (
    compute_code_doc_id,
    compute_file_hash,
    process_code_file,
)


# ============================================================================
# Document ID Computation Tests
# ============================================================================


class TestComputeCodeDocId:
    """Tests for document ID generation."""

    def test_compute_doc_id_simple(self):
        """Test basic document ID computation."""
        doc_id = compute_code_doc_id(
            file_path="src/main/java/MyService.java",
            repository="my-service",
            project_key="PROJ",
        )
        assert doc_id == "PROJ_my_service_src_main_java_MyService_java"

    def test_compute_doc_id_with_underscores(self):
        """Test document ID with existing underscores in path."""
        doc_id = compute_code_doc_id(
            file_path="src/my_service/main.java",
            repository="test-repo",
            project_key="TEST",
        )
        assert doc_id == "TEST_test_repo_src_my_service_main_java"

    def test_compute_doc_id_consistency(self):
        """Test that same inputs produce same ID."""
        id1 = compute_code_doc_id(
            file_path="src/main/java/Service.java",
            repository="repo",
            project_key="PROJ",
        )
        id2 = compute_code_doc_id(
            file_path="src/main/java/Service.java",
            repository="repo",
            project_key="PROJ",
        )
        assert id1 == id2

    def test_compute_doc_id_different_files(self):
        """Test that different files produce different IDs."""
        id1 = compute_code_doc_id(
            file_path="src/main/java/Service1.java",
            repository="repo",
            project_key="PROJ",
        )
        id2 = compute_code_doc_id(
            file_path="src/main/java/Service2.java",
            repository="repo",
            project_key="PROJ",
        )
        assert id1 != id2

    def test_compute_doc_id_normalises_backslashes(self):
        """Test that backslashes are normalised to underscores."""
        doc_id = compute_code_doc_id(
            file_path=r"src\main\java\MyService.java",
            repository="repo",
            project_key="PROJ",
        )
        assert "\\" not in doc_id
        assert doc_id.count("_") >= 5  # Multiple separators converted


# ============================================================================
# File Hash Computation Tests
# ============================================================================


class TestComputeFileHash:
    """Tests for file hashing."""

    def test_compute_file_hash_md5(self):
        """Test MD5 hash computation."""
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            f.write("test content")
            f.flush()
            
            hash_value = compute_file_hash(f.name)
            
            # Verify it's valid hex
            assert len(hash_value) == 32
            assert all(c in '0123456789abcdef' for c in hash_value)
            
            Path(f.name).unlink()

    def test_compute_file_hash_consistency(self):
        """Test that same file produces same hash."""
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            f.write("consistent content")
            f.flush()
            
            hash1 = compute_file_hash(f.name)
            hash2 = compute_file_hash(f.name)
            
            assert hash1 == hash2
            
            Path(f.name).unlink()

    def test_compute_file_hash_different_content(self):
        """Test that different content produces different hashes."""
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f1:
            f1.write("content 1")
            f1.flush()
            hash1 = compute_file_hash(f1.name)
            Path(f1.name).unlink()
        
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f2:
            f2.write("content 2")
            f2.flush()
            hash2 = compute_file_hash(f2.name)
            Path(f2.name).unlink()
        
        assert hash1 != hash2

    def test_compute_file_hash_binary_file(self):
        """Test hashing binary file."""
        with tempfile.NamedTemporaryFile(delete=False, mode='wb') as f:
            f.write(b'\x00\x01\x02\x03\x04')
            f.flush()
            
            hash_value = compute_file_hash(f.name)
            
            assert len(hash_value) == 32
            
            Path(f.name).unlink()

    def test_compute_file_hash_nonexistent_file(self):
        """Test that nonexistent file uses path as fallback."""
        fake_path = "/tmp/nonexistent_file_12345.txt"
        hash_value = compute_file_hash(fake_path)
        
        # Should produce hash of the path string
        expected_hash = hashlib.md5(fake_path.encode("utf-8")).hexdigest()
        assert hash_value == expected_hash


# ============================================================================
# Process Code File Tests
# ============================================================================


class TestProcessCodeFile:
    """Tests for code file ingestion pipeline."""

    @patch('scripts.ingest.ingest_git.get_existing_doc_hash')
    @patch('scripts.ingest.ingest_git.CodeParser')
    @patch('scripts.ingest.ingest_git.chunk_text')
    @patch('scripts.ingest.ingest_git.store_chunks_in_chroma')
    def test_process_code_file_success(
        self,
        mock_store,
        mock_chunk,
        mock_parser_class,
        mock_get_hash,
    ):
        """Test successful code file processing."""
        # Setup mocks
        mock_get_hash.return_value = None  # File not seen before
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parse_result = MagicMock()
        mock_parse_result.language = "java"
        mock_parser.parse_file.return_value = mock_parse_result
        mock_chunk.return_value = [
            {"text": "chunk1"},
            {"text": "chunk2"},
        ]
        
        config = MagicMock()
        collection = MagicMock()
        
        # Process file
        result = process_code_file(
            file_path="src/Main.java",
            file_content="public class Main {}",
            project_key="PROJ",
            repo_slug="my-repo",
            branch="main",
            chunk_collection=collection,
            doc_collection=collection,
            parser=mock_parser,
            llm_cache=None,
            embedding_cache=None,
            config=config,
        )
        
        # Verify success
        assert result is True
        mock_chunk.assert_called_once()
        mock_store.assert_called_once()
        
        # Verify metadata
        call_args = mock_store.call_args
        metadata = call_args.kwargs['metadata']
        assert metadata['file_path'] == "src/Main.java"
        assert metadata['repository'] == "my-repo"
        assert metadata['project'] == "PROJ"

    @patch('scripts.ingest.ingest_git.get_existing_doc_hash')
    def test_process_code_file_unchanged(self, mock_get_hash):
        """Test that unchanged files are skipped."""
        # Compute actual file hash
        content = "public class Main {}"
        file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        # Mock returns the same hash so file appears unchanged
        mock_get_hash.return_value = file_hash
        
        config = MagicMock()
        collection = MagicMock()
        parser = MagicMock()
        
        result = process_code_file(
            file_path="src/Main.java",
            file_content=content,
            project_key="PROJ",
            repo_slug="my-repo",
            branch="main",
            chunk_collection=collection,
            doc_collection=collection,
            parser=parser,
            llm_cache=None,
            embedding_cache=None,
            config=config,
        )
        
        # Should return True (skipped is success)
        assert result is True
        # Parser should not be called
        parser.parse_file.assert_not_called()

    @patch('scripts.ingest.ingest_git.get_existing_doc_hash')
    @patch('scripts.ingest.ingest_git.CodeParser')
    @patch('scripts.ingest.ingest_git.chunk_text')
    def test_process_code_file_no_chunks(
        self,
        mock_chunk,
        mock_parser_class,
        mock_get_hash,
    ):
        """Test file with no chunks (should fail)."""
        mock_get_hash.return_value = None
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_chunk.return_value = []  # No chunks
        
        config = MagicMock()
        collection = MagicMock()
        
        result = process_code_file(
            file_path="src/Empty.java",
            file_content="",
            project_key="PROJ",
            repo_slug="my-repo",
            branch="main",
            chunk_collection=collection,
            doc_collection=collection,
            parser=mock_parser,
            llm_cache=None,
            embedding_cache=None,
            config=config,
        )
        
        assert result is False

    @patch('scripts.ingest.ingest_git.get_existing_doc_hash')
    @patch('scripts.ingest.ingest_git.CodeParser')
    def test_process_code_file_parser_error(
        self,
        mock_parser_class,
        mock_get_hash,
    ):
        """Test file processing with parser error."""
        mock_get_hash.return_value = None
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_file.side_effect = Exception("Parse error")
        
        config = MagicMock()
        collection = MagicMock()
        
        result = process_code_file(
            file_path="src/BadCode.java",
            file_content="invalid {{{",
            project_key="PROJ",
            repo_slug="my-repo",
            branch="main",
            chunk_collection=collection,
            doc_collection=collection,
            parser=mock_parser,
            llm_cache=None,
            embedding_cache=None,
            config=config,
        )
        
        assert result is False

    @patch('scripts.ingest.ingest_git.get_existing_doc_hash')
    @patch('scripts.ingest.ingest_git.CodeParser')
    @patch('scripts.ingest.ingest_git.chunk_text')
    @patch('scripts.ingest.ingest_git.store_chunks_in_chroma')
    def test_process_code_file_with_caches(
        self,
        mock_store,
        mock_chunk,
        mock_parser_class,
        mock_get_hash,
    ):
        """Test file processing with LLM and embedding caches."""
        mock_get_hash.return_value = None
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parse_result = MagicMock()
        mock_parse_result.language = "python"
        mock_parser.parse_file.return_value = mock_parse_result
        mock_chunk.return_value = [{"text": "chunk"}]
        
        config = MagicMock()
        collection = MagicMock()
        
        # Create mock caches with cache_path parameter
        llm_cache = MagicMock()
        llm_cache.cache_path = "/tmp/llm_cache"
        embedding_cache = MagicMock()
        embedding_cache.cache_path = "/tmp/embedding_cache"
        
        result = process_code_file(
            file_path="src/script.py",
            file_content="print('hello')",
            project_key="PROJ",
            repo_slug="my-repo",
            branch="main",
            chunk_collection=collection,
            doc_collection=collection,
            parser=mock_parser,
            llm_cache=llm_cache,
            embedding_cache=embedding_cache,
            config=config,
        )
        
        assert result is True
        # Verify embedding_cache was passed to store_chunks_in_chroma
        call_args = mock_store.call_args
        assert call_args.kwargs['embedding_cache'] is embedding_cache


# ============================================================================
# Integration Tests
# ============================================================================


class TestIngestGitIntegration:
    """Integration tests for the full ingestion pipeline."""

    @patch('scripts.ingest.ingest_git.get_existing_doc_hash')
    @patch('scripts.ingest.ingest_git.CodeParser')
    @patch('scripts.ingest.ingest_git.chunk_text')
    @patch('scripts.ingest.ingest_git.store_chunks_in_chroma')
    def test_multiple_files_same_repo(
        self,
        mock_store,
        mock_chunk,
        mock_parser_class,
        mock_get_hash,
    ):
        """Test processing multiple files from same repository."""
        mock_get_hash.return_value = None
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parse_result = MagicMock()
        mock_parse_result.language = "java"
        mock_parser.parse_file.return_value = mock_parse_result
        mock_chunk.return_value = [{"text": "chunk"}]
        
        config = MagicMock()
        collection = MagicMock()
        
        files = [
            ("src/Service1.java", "public class Service1 {}"),
            ("src/Service2.java", "public class Service2 {}"),
            ("test/ServiceTest.java", "public class ServiceTest {}"),
        ]
        
        results = []
        for file_path, content in files:
            result = process_code_file(
                file_path=file_path,
                file_content=content,
                project_key="PROJ",
                repo_slug="my-repo",
                branch="main",
                chunk_collection=collection,
                doc_collection=collection,
                parser=mock_parser,
                llm_cache=None,
                embedding_cache=None,
                config=config,
            )
            results.append(result)
        
        # All should succeed
        assert all(results)
        # Store should be called 3 times
        assert mock_store.call_count == 3

    def test_doc_id_collision_prevention(self):
        """Test that different repos/projects don't create colliding IDs."""
        # Same file in different repos
        id1 = compute_code_doc_id(
            file_path="src/Service.java",
            repository="repo1",
            project_key="PROJ1",
        )
        id2 = compute_code_doc_id(
            file_path="src/Service.java",
            repository="repo2",
            project_key="PROJ2",
        )
        
        assert id1 != id2

    def test_doc_id_collision_prevention_same_repo_different_project(self):
        """Test that same repo in different projects creates different IDs."""
        id1 = compute_code_doc_id(
            file_path="src/Service.java",
            repository="my-repo",
            project_key="PROJ1",
        )
        id2 = compute_code_doc_id(
            file_path="src/Service.java",
            repository="my-repo",
            project_key="PROJ2",
        )
        
        assert id1 != id2
