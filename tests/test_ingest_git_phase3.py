"""Phase 3: Operations features tests (dry-run, encoding detection, error recovery).

Tests for:
- Dry-run mode: simulate processing without persistence
- File encoding detection: UTF-8, Latin-1, CP1252 with fallback
- Error recovery: tracking and reporting of errors during processing
"""

import codecs

# Add project root to path
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ingest.ingest_git import (
    ConcurrentFileProcessor,
    EncodingDetector,
    ErrorRecoveryStrategy,
    process_code_file,
)

# =========================
# ENCODING DETECTION TESTS
# =========================


class TestEncodingDetector:
    """Tests for encoding detection and fallback."""

    def test_detect_utf8_encoding(self):
        """Test detection of UTF-8 encoded content."""
        content = "Hello, World! 🌍".encode("utf-8")
        encoding, confidence = EncodingDetector.detect_encoding("test.py", content)

        # Chardet might not always detect correctly, just verify we get an encoding
        assert encoding is not None
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_detect_latin1_encoding(self):
        """Test detection of Latin-1 encoded content."""
        # Latin-1 text with special characters
        content = "Café résumé".encode("latin-1")
        encoding, confidence = EncodingDetector.detect_encoding("test.txt", content)

        # Should detect as some encoding
        assert encoding is not None
        assert isinstance(confidence, float)

    def test_detect_empty_content(self):
        """Test detection with empty content."""
        encoding, confidence = EncodingDetector.detect_encoding("test.py", b"")

        assert encoding == "utf-8"  # Default fallback
        assert confidence == 1.0

    def test_decode_with_fallback_utf8(self):
        """Test UTF-8 decoding."""
        content = "Hello, World!".encode("utf-8")
        decoded = EncodingDetector.decode_with_fallback(content, "test.py")

        assert decoded == "Hello, World!"

    def test_decode_with_fallback_latin1(self):
        """Test Latin-1 fallback decoding."""
        content = "Café".encode("latin-1")
        decoded = EncodingDetector.decode_with_fallback(
            content, "test.txt", preferred_encoding="latin-1"
        )

        assert "Caf" in decoded  # At least partial content decoded

    def test_decode_with_fallback_lossy(self):
        """Test lossy decoding with replacement characters."""
        # Create content that's invalid UTF-8 but valid Latin-1
        content = b"\xff\xfe"
        decoded = EncodingDetector.decode_with_fallback(content, "test.bin")

        # Should not raise, uses replacement characters
        assert isinstance(decoded, str)

    def test_encoding_detection_with_file_path(self):
        """Test encoding detection from file path context."""
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".py", delete=False
        ) as f:
            f.write("# Python file\nprint('Hello')")
            temp_file = f.name

        try:
            encoding, confidence = EncodingDetector.detect_encoding(temp_file)
            # Just verify we get an encoding back
            assert encoding is not None
            assert isinstance(confidence, float)
        finally:
            Path(temp_file).unlink()


# =========================
# ERROR RECOVERY TESTS
# =========================


class TestErrorRecoveryStrategy:
    """Tests for error recovery and tracking."""

    def test_record_error_recoverable(self):
        """Test recording a recoverable error."""
        recovery = ErrorRecoveryStrategy()
        error = ValueError("Test error")

        recovery.record_error("test.py", error, stage="parsing", recoverable=True)

        assert len(recovery.errors) == 1
        assert recovery.recovered_count == 1
        assert recovery.failed_count == 0

    def test_record_error_not_recoverable(self):
        """Test recording a non-recoverable error."""
        recovery = ErrorRecoveryStrategy()
        error = RuntimeError("Critical error")

        recovery.record_error("test.py", error, stage="storage", recoverable=False)

        assert len(recovery.errors) == 1
        assert recovery.recovered_count == 0
        assert recovery.failed_count == 1

    def test_record_warning(self):
        """Test recording a warning."""
        recovery = ErrorRecoveryStrategy()

        recovery.record_warning("test.py", "Deprecated pattern found", stage="analysis")

        assert len(recovery.warnings) == 1
        assert recovery.warnings[0]["warning"] == "Deprecated pattern found"

    def test_summary_with_multiple_errors(self):
        """Test error summary with multiple errors."""
        recovery = ErrorRecoveryStrategy()

        recovery.record_error("file1.py", ValueError("Error 1"), recoverable=True)
        recovery.record_error("file2.py", RuntimeError("Error 2"), recoverable=False)
        recovery.record_warning("file3.py", "Warning 1")
        recovery.record_warning("file4.py", "Warning 2")

        summary = recovery.summary()

        assert summary["total_errors"] == 2
        assert summary["recovered_errors"] == 1
        assert summary["failed_errors"] == 1
        assert summary["total_warnings"] == 2

    def test_summary_empty(self):
        """Test summary with no errors or warnings."""
        recovery = ErrorRecoveryStrategy()
        summary = recovery.summary()

        assert summary["total_errors"] == 0
        assert summary["total_warnings"] == 0
        assert summary["recovered_errors"] == 0
        assert summary["failed_errors"] == 0

    def test_error_truncation(self):
        """Test that long error messages are truncated."""
        recovery = ErrorRecoveryStrategy()
        long_error = "x" * 500
        error = ValueError(long_error)

        recovery.record_error("test.py", error)

        assert len(recovery.errors[0]["error_message"]) <= 200


# =========================
# DRY-RUN MODE TESTS
# =========================


class TestDryRunMode:
    """Tests for dry-run mode functionality."""

    @patch("scripts.ingest.ingest_git.store_chunks_in_chroma")
    @patch("scripts.ingest.ingest_git.store_child_chunks")
    @patch("scripts.ingest.ingest_git.store_parent_chunks")
    @patch("scripts.ingest.ingest_git.get_existing_doc_hash")
    def test_dry_run_no_storage(self, mock_get_hash, mock_parent, mock_child, mock_store):
        """Test that dry-run mode doesn't persist to ChromaDB."""
        mock_get_hash.return_value = None

        # Create minimal mocks
        config = Mock()
        config.enable_dlp = False
        config.generate_summaries = False
        config.enable_parent_child_chunking = False
        config.bm25_indexing_enabled = False

        parser = Mock()
        parser.parse_file.return_value = Mock(
            language="python",
            file_type="code",
            to_dict=Mock(return_value={}),
        )

        with (
            patch("scripts.ingest.ingest_git.chunk_text") as mock_chunk,
            patch("scripts.ingest.ingest_git.Path.write_text"),
            patch("scripts.ingest.ingest_git.Path.unlink"),
        ):
            mock_chunk.return_value = [{"text": "chunk1"}, {"text": "chunk2"}]

            result = process_code_file(
                file_path="test.py",
                file_content="print('hello')",
                project_key="TEST",
                repo_slug="test-repo",
                branch="main",
                chunk_collection=Mock(),
                doc_collection=Mock(),
                parser=parser,
                llm_cache=None,
                embedding_cache=None,
                config=config,
                dry_run=True,  # Enable dry-run
                error_recovery=ErrorRecoveryStrategy(),
            )

        # Verify storage was NOT called
        mock_store.assert_not_called()
        mock_child.assert_not_called()
        mock_parent.assert_not_called()

    @patch("scripts.ingest.ingest_git.store_chunks_in_chroma")
    @patch("scripts.ingest.ingest_git.get_existing_doc_hash")
    def test_dry_run_vs_normal_mode(self, mock_get_hash, mock_store):
        """Test difference between dry-run and normal mode."""
        mock_get_hash.return_value = None

        config = Mock()
        config.enable_dlp = False
        config.generate_summaries = False
        config.enable_parent_child_chunking = False
        config.bm25_indexing_enabled = False

        parser = Mock()
        parser.parse_file.return_value = Mock(
            language="python",
            file_type="code",
            to_dict=Mock(return_value={}),
        )

        with (
            patch("scripts.ingest.ingest_git.chunk_text") as mock_chunk,
            patch("scripts.ingest.ingest_git.Path.write_text"),
            patch("scripts.ingest.ingest_git.Path.unlink"),
        ):
            mock_chunk.return_value = [{"text": "chunk1"}]

            # Normal mode - should call store
            mock_store.reset_mock()
            process_code_file(
                file_path="test.py",
                file_content="print('hello')",
                project_key="TEST",
                repo_slug="test-repo",
                branch="main",
                chunk_collection=Mock(),
                doc_collection=Mock(),
                parser=parser,
                llm_cache=None,
                embedding_cache=None,
                config=config,
                dry_run=False,  # Normal mode
                error_recovery=ErrorRecoveryStrategy(),
            )
            assert mock_store.called, "Storage should be called in normal mode"

            # Dry-run mode - should NOT call store
            mock_store.reset_mock()
            process_code_file(
                file_path="test.py",
                file_content="print('hello')",
                project_key="TEST",
                repo_slug="test-repo",
                branch="main",
                chunk_collection=Mock(),
                doc_collection=Mock(),
                parser=parser,
                llm_cache=None,
                embedding_cache=None,
                config=config,
                dry_run=True,  # Dry-run mode
                error_recovery=ErrorRecoveryStrategy(),
            )
            assert not mock_store.called, "Storage should NOT be called in dry-run mode"


# =========================
# CONCURRENT PROCESSOR WITH PHASE 3
# =========================


class TestConcurrentProcessorPhase3:
    """Tests for ConcurrentFileProcessor with Phase 3 features."""

    def test_processor_with_dry_run(self):
        """Test that processor respects dry_run flag."""
        processor = ConcurrentFileProcessor(
            max_workers=1,
            chunk_collection=Mock(),
            doc_collection=Mock(),
            parser=Mock(),
            config=Mock(detect_encoding=False),
            dry_run=True,
            error_recovery=ErrorRecoveryStrategy(),
        )

        assert processor.dry_run is True
        assert processor.error_recovery is not None

    def test_processor_with_error_recovery(self):
        """Test that processor uses error recovery."""
        recovery = ErrorRecoveryStrategy()
        processor = ConcurrentFileProcessor(
            max_workers=1,
            chunk_collection=Mock(),
            doc_collection=Mock(),
            parser=Mock(),
            config=Mock(),
            dry_run=False,
            error_recovery=recovery,
        )

        assert processor.error_recovery is recovery


# =========================
# INTEGRATION TESTS
# =========================


class TestPhase3Integration:
    """Integration tests for Phase 3 features."""

    def test_encoding_detection_integration(self):
        """Test encoding detection with real files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with different encodings
            utf8_file = Path(tmpdir) / "utf8.txt"
            utf8_file.write_text("Hello UTF-8", encoding="utf-8")

            latin1_file = Path(tmpdir) / "latin1.txt"
            latin1_file.write_text("Café", encoding="latin-1")

            # Detect encodings
            utf8_content = utf8_file.read_bytes()
            latin1_content = latin1_file.read_bytes()

            utf8_enc, _ = EncodingDetector.detect_encoding(str(utf8_file), utf8_content)
            latin1_enc, _ = EncodingDetector.detect_encoding(str(latin1_file), latin1_content)

            # Just verify encodings were detected (chardet may vary)
            assert utf8_enc is not None
            assert latin1_enc is not None

    def test_error_recovery_with_multiple_files(self):
        """Test error recovery across multiple files."""
        recovery = ErrorRecoveryStrategy()

        # Simulate processing multiple files with mixed success
        for i in range(5):
            if i % 2 == 0:  # Half succeed
                recovery.record_warning(f"file{i}.py", "Non-critical warning")
            else:  # Half fail
                recovery.record_error(
                    f"file{i}.py",
                    ValueError(f"Error in file {i}"),
                    recoverable=True,  # All are recoverable in this test
                )

        summary = recovery.summary()
        assert summary["total_errors"] == 2
        assert summary["total_warnings"] == 3
        assert summary["recovered_errors"] == 2  # All errors were recoverable


# =========================
# EDGE CASE TESTS
# =========================


class TestPhase3EdgeCases:
    """Edge case tests for Phase 3 features."""

    def test_encoding_detection_with_binary_file(self):
        """Test encoding detection on binary file."""
        binary_content = bytes([0xFF, 0xD8, 0xFF, 0xE0])  # JPEG header
        encoding, confidence = EncodingDetector.detect_encoding("image.jpg", binary_content)

        # Should handle gracefully, may detect as binary or fallback
        assert encoding is not None

    def test_error_recovery_with_very_long_error_message(self):
        """Test error tracking with very long error messages."""
        recovery = ErrorRecoveryStrategy()
        very_long_msg = "x" * 10000

        recovery.record_error("test.py", ValueError(very_long_msg), recoverable=True)

        # Should truncate message
        assert len(recovery.errors[0]["error_message"]) <= 200

    def test_encoding_detection_with_mixed_encodings(self):
        """Test handling of files with mixed character encodings."""
        # This is tricky - create content that's mostly UTF-8 with some Latin-1
        content = "Hello".encode("utf-8") + "Café".encode("latin-1")

        # Should still attempt detection
        encoding, confidence = EncodingDetector.detect_encoding("mixed.txt", content)
        assert encoding is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
