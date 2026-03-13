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
    AuthConfig,
    build_auth_headers,
    check_ollama_availability,
    compute_chunk_hash,
    compute_doc_id,
    compute_file_hash,
    parse_seed_auth,
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


# ============================================================================
# parse_seed_auth
# ============================================================================


class TestParseSeedAuth:
    """Tests for parse_seed_auth."""

    def test_returns_none_for_no_auth(self) -> None:
        """Missing auth block returns None."""
        assert parse_seed_auth(None) is None

    def test_returns_none_for_empty_dict(self) -> None:
        """Empty dict returns None (treated as absent)."""
        assert parse_seed_auth({}) is None

    def test_parses_bearer_token_env(self) -> None:
        """Bearer config is parsed correctly."""
        cfg = parse_seed_auth({"type": "bearer", "token_env": "MY_TOKEN"})
        assert cfg is not None
        assert cfg.auth_type == "bearer"
        assert cfg.token_env == "MY_TOKEN"
        assert cfg.token is None

    def test_parses_bearer_inline_token(self) -> None:
        """Inline bearer token is preserved as-is."""
        cfg = parse_seed_auth({"type": "bearer", "token": "secret123"})
        assert cfg is not None
        assert cfg.token == "secret123"

    def test_parses_basic_auth(self) -> None:
        """Basic auth fields are parsed correctly."""
        cfg = parse_seed_auth({
            "type": "basic",
            "username_env": "INGEST_USER",
            "password_env": "INGEST_PASS",
        })
        assert cfg is not None
        assert cfg.auth_type == "basic"
        assert cfg.username_env == "INGEST_USER"
        assert cfg.password_env == "INGEST_PASS"

    def test_parses_cookie_auth(self) -> None:
        """Cookie auth fields are parsed correctly."""
        cfg = parse_seed_auth({
            "type": "cookie",
            "cookie_name": "sess",
            "cookie_env": "SESSION_COOKIE",
        })
        assert cfg is not None
        assert cfg.auth_type == "cookie"
        assert cfg.cookie_name == "sess"
        assert cfg.cookie_env == "SESSION_COOKIE"

    def test_raises_on_unrecognised_type(self) -> None:
        """Unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported auth type"):
            parse_seed_auth({"type": "oauth2"})

    def test_raises_on_missing_type(self) -> None:
        """Auth block without 'type' raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported auth type"):
            parse_seed_auth({"token_env": "FOO"})

    def test_type_is_case_insensitive(self) -> None:
        """Auth type matching is case-insensitive."""
        cfg = parse_seed_auth({"type": "BEARER", "token_env": "X"})
        assert cfg is not None
        assert cfg.auth_type == "bearer"


# ============================================================================
# build_auth_headers
# ============================================================================


class TestBuildAuthHeaders:
    """Tests for build_auth_headers."""

    def test_returns_empty_dict_for_none(self) -> None:
        """No auth config produces empty headers."""
        assert build_auth_headers(None) == {}

    def test_bearer_from_env(self) -> None:
        """Bearer token is resolved from environment variable."""
        cfg = AuthConfig(auth_type="bearer", token_env="TEST_TOKEN_VAR")
        with patch.dict("os.environ", {"TEST_TOKEN_VAR": "mytoken"}):
            headers = build_auth_headers(cfg)
        assert headers == {"Authorization": "Bearer mytoken"}

    def test_bearer_inline_token(self) -> None:
        """Inline bearer token is used directly."""
        cfg = AuthConfig(auth_type="bearer", token="directtoken")
        headers = build_auth_headers(cfg)
        assert headers == {"Authorization": "Bearer directtoken"}

    def test_bearer_raises_when_no_token(self) -> None:
        """Raises ValueError when bearer token is absent."""
        cfg = AuthConfig(auth_type="bearer")
        with pytest.raises(ValueError, match="Bearer auth configured"):
            build_auth_headers(cfg)

    def test_bearer_env_missing_raises(self) -> None:
        """Raises ValueError when referenced env var is not set."""
        cfg = AuthConfig(auth_type="bearer", token_env="NONEXISTENT_VAR_XYZ")
        import os
        os.environ.pop("NONEXISTENT_VAR_XYZ", None)
        with pytest.raises(ValueError, match="Bearer auth configured"):
            build_auth_headers(cfg)

    def test_basic_auth_from_env(self) -> None:
        """Basic auth credentials are resolved from environment variables."""
        import base64 as b64
        cfg = AuthConfig(auth_type="basic", username_env="TEST_USER", password_env="TEST_PASS")
        with patch.dict("os.environ", {"TEST_USER": "alice", "TEST_PASS": "s3cr3t"}):
            headers = build_auth_headers(cfg)
        expected = b64.b64encode(b"alice:s3cr3t").decode("ascii")
        assert headers == {"Authorization": f"Basic {expected}"}

    def test_basic_inline_credentials(self) -> None:
        """Inline basic auth credentials are used directly."""
        import base64 as b64
        cfg = AuthConfig(auth_type="basic", username="bob", password="pass123")
        headers = build_auth_headers(cfg)
        expected = b64.b64encode(b"bob:pass123").decode("ascii")
        assert headers == {"Authorization": f"Basic {expected}"}

    def test_basic_raises_when_missing_credentials(self) -> None:
        """Raises ValueError when username or password is missing."""
        cfg = AuthConfig(auth_type="basic", username="onlyuser")
        with pytest.raises(ValueError, match="Basic auth configured"):
            build_auth_headers(cfg)

    def test_cookie_from_env(self) -> None:
        """Cookie value is resolved from environment variable."""
        cfg = AuthConfig(auth_type="cookie", cookie_name="session", cookie_env="TEST_COOKIE")
        with patch.dict("os.environ", {"TEST_COOKIE": "abc123"}):
            headers = build_auth_headers(cfg)
        assert headers == {"Cookie": "session=abc123"}

    def test_cookie_inline_value(self) -> None:
        """Inline cookie value is used directly."""
        cfg = AuthConfig(auth_type="cookie", cookie_name="tok", cookie_value="xyz789")
        headers = build_auth_headers(cfg)
        assert headers == {"Cookie": "tok=xyz789"}

    def test_cookie_defaults_name_to_session(self) -> None:
        """Cookie name defaults to 'session' when not specified."""
        cfg = AuthConfig(auth_type="cookie", cookie_value="val")
        headers = build_auth_headers(cfg)
        assert headers == {"Cookie": "session=val"}

    def test_cookie_raises_when_no_value(self) -> None:
        """Raises ValueError when cookie value is absent."""
        cfg = AuthConfig(auth_type="cookie", cookie_name="sess")
        with pytest.raises(ValueError, match="Cookie auth configured"):
            build_auth_headers(cfg)
