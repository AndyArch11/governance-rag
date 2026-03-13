"""Unit tests for scripts.ingest.ingest_utils.

Covers the centralised ingestion helpers that were previously duplicated
across ingest.py, ingest_git.py, and ingest_academic.py.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.ingest.ingest_utils import (
    check_ollama_availability,
    compute_chunk_hash,
    compute_doc_id,
    compute_file_hash,
)


# ============================================================================
# check_ollama_availability
# ============================================================================


class TestCheckOllamaAvailability:
    """Tests for check_ollama_availability."""

    def test_returns_true_when_skip_llm(self) -> None:
        """skip_llm=True bypasses the HTTP check."""
        mock_logger = MagicMock()
        result = check_ollama_availability(mock_logger, skip_llm=True)
        assert result is True
        mock_logger.info.assert_called_once()

    def test_returns_true_when_ollama_responds(self) -> None:
        """Returns True when Ollama responds with HTTP 200."""
        mock_logger = MagicMock()
        with patch("scripts.ingest.ingest_utils.requests.get") as mock_get:
            mock_get.return_value = SimpleNamespace(status_code=200)
            result = check_ollama_availability(mock_logger)
        assert result is True
        mock_logger.info.assert_called()

    def test_returns_false_on_connection_error(self) -> None:
        """Returns False when Ollama is not running."""
        import requests as req

        mock_logger = MagicMock()
        with patch("scripts.ingest.ingest_utils.requests.get", side_effect=req.ConnectionError):
            result = check_ollama_availability(mock_logger)
        assert result is False
        mock_logger.warning.assert_called()

    def test_returns_false_on_timeout(self) -> None:
        """Returns False when Ollama connection times out."""
        import requests as req

        mock_logger = MagicMock()
        with patch("scripts.ingest.ingest_utils.requests.get", side_effect=req.Timeout):
            result = check_ollama_availability(mock_logger)
        assert result is False

    def test_returns_false_on_unexpected_exception(self) -> None:
        """Returns False on any unexpected error."""
        mock_logger = MagicMock()
        with patch("scripts.ingest.ingest_utils.requests.get", side_effect=OSError("boom")):
            result = check_ollama_availability(mock_logger)
        assert result is False

    def test_uses_ollama_host_env_var(self) -> None:
        """Respects OLLAMA_HOST environment variable."""
        mock_logger = MagicMock()
        custom_host = "http://host.docker.internal:11434"
        with patch("scripts.ingest.ingest_utils.requests.get") as mock_get:
            mock_get.return_value = SimpleNamespace(status_code=200)
            with patch.dict("os.environ", {"OLLAMA_HOST": custom_host}):
                check_ollama_availability(mock_logger)
            called_url = mock_get.call_args[0][0]
        assert called_url.startswith(custom_host)

    def test_explicit_host_overrides_env_var(self) -> None:
        """Explicit ollama_host argument takes precedence over OLLAMA_HOST env."""
        mock_logger = MagicMock()
        explicit = "http://myserver:11434"
        with patch("scripts.ingest.ingest_utils.requests.get") as mock_get:
            mock_get.return_value = SimpleNamespace(status_code=200)
            with patch.dict("os.environ", {"OLLAMA_HOST": "http://other:11434"}):
                check_ollama_availability(mock_logger, ollama_host=explicit)
            called_url = mock_get.call_args[0][0]
        assert called_url.startswith(explicit)


# ============================================================================
# compute_doc_id
# ============================================================================


class TestComputeDocId:
    """Tests for compute_doc_id."""

    def test_extracts_filename_without_extension(self) -> None:
        assert compute_doc_id("/path/to/MyDocument.html") == "MyDocument"

    def test_relative_path(self) -> None:
        assert compute_doc_id("folder/subfolder/test_file.html") == "test_file"

    def test_no_extension(self) -> None:
        assert compute_doc_id("/path/to/document") == "document"

    def test_multiple_dots_keeps_all_but_final(self) -> None:
        assert compute_doc_id("/path/my.document.v2.html") == "my.document.v2"

    def test_filename_only(self) -> None:
        assert compute_doc_id("report.pdf") == "report"


# ============================================================================
# compute_file_hash
# ============================================================================


class TestComputeFileHash:
    """Tests for compute_file_hash."""

    def test_returns_sha256_hex_string(self) -> None:
        """Hash is a 64-character hex string."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("hello world")
            path = f.name
        try:
            result = compute_file_hash(path)
            assert len(result) == 64
            assert all(c in "0123456789abcdef" for c in result)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_consistent_for_same_file(self) -> None:
        """Same file always produces the same hash."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("stable content")
            path = f.name
        try:
            assert compute_file_hash(path) == compute_file_hash(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_different_content_gives_different_hash(self) -> None:
        """Different file content yields a different hash."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("content A")
            path_a = f.name
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("content B")
            path_b = f.name
        try:
            assert compute_file_hash(path_a) != compute_file_hash(path_b)
        finally:
            Path(path_a).unlink(missing_ok=True)
            Path(path_b).unlink(missing_ok=True)

    def test_nonexistent_file_falls_back_to_path_hash(self) -> None:
        """Unreadable/missing files fall back to hashing the path string."""
        fake = "/tmp/no_such_file_xyzzy_99999.txt"
        expected = hashlib.sha256(fake.encode("utf-8")).hexdigest()
        assert compute_file_hash(fake) == expected

    def test_binary_file(self) -> None:
        """Binary files are hashed without error."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(bytes(range(256)))
            path = f.name
        try:
            result = compute_file_hash(path)
            assert len(result) == 64
        finally:
            Path(path).unlink(missing_ok=True)


# ============================================================================
# compute_chunk_hash
# ============================================================================


class TestComputeChunkHash:
    """Tests for compute_chunk_hash."""

    def test_returns_64_char_hex(self) -> None:
        result = compute_chunk_hash("some chunk text")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_consistent_for_same_text(self) -> None:
        text = "consistent chunk"
        assert compute_chunk_hash(text) == compute_chunk_hash(text)

    def test_different_text_gives_different_hash(self) -> None:
        assert compute_chunk_hash("alpha") != compute_chunk_hash("beta")

    def test_matches_manual_sha256(self) -> None:
        text = "verify me"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert compute_chunk_hash(text) == expected

    def test_empty_string(self) -> None:
        result = compute_chunk_hash("")
        assert len(result) == 64
