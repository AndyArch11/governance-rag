"""Shared ingestion utility functions used across all ingest entrypoints.

Centralises common helpers so that `ingest.py`, `ingest_git.py`, and
`ingest_academic.py` do not duplicate logic.

Provided utilities:
- :func:`check_ollama_availability` — verify Ollama is reachable before LLM calls
- :func:`compute_doc_id` — stable document identifier from file path
- :func:`compute_file_hash` — SHA-256 fingerprint of file contents
- :func:`compute_chunk_hash` — SHA-256 fingerprint of chunk text
- :func:`parse_seed_auth` — parse auth block from a URL seed definition
- :func:`build_auth_headers` — build HTTP auth headers from an AuthConfig
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class AuthConfig:
    """Authentication credentials for a protected URL source.

    Supports three schemes:

    - ``bearer``: Adds ``Authorization: Bearer <token>`` header.
      Provide the token via ``token_env`` (name of an environment variable)
      or directly via ``token``.
    - ``basic``: Adds ``Authorization: Basic <base64(user:pass)>`` header.
      Provide credentials via ``username_env`` / ``password_env`` or
      directly via ``username`` / ``password``.
    - ``cookie``: Adds a ``Cookie: <name>=<value>`` header.
      Provide the cookie value via ``cookie_env`` or directly via
      ``cookie_value``.

    Security note: prefer the ``*_env`` variants so that credentials are
    sourced from environment variables and never embedded in JSON seed files.
    """

    auth_type: str  # "bearer" | "basic" | "cookie"

    # Bearer
    token: Optional[str] = field(default=None, repr=False)
    token_env: Optional[str] = None

    # Basic
    username: Optional[str] = field(default=None, repr=False)
    username_env: Optional[str] = None
    password: Optional[str] = field(default=None, repr=False)
    password_env: Optional[str] = None

    # Cookie
    cookie_name: Optional[str] = None
    cookie_value: Optional[str] = field(default=None, repr=False)
    cookie_env: Optional[str] = None


def parse_seed_auth(auth_block: Optional[Dict[str, Any]]) -> Optional[AuthConfig]:
    """Parse the ``auth`` block from a URL seed definition into an AuthConfig.

    Returns ``None`` when the block is absent or empty, so callers can skip
    authenticated paths cleanly.

    Expected JSON shape (bearer example)::

        {
          "type": "bearer",
          "token_env": "MY_API_TOKEN"
        }

    Basic auth example::

        {
          "type": "basic",
          "username_env": "INGEST_USER",
          "password_env": "INGEST_PASSWORD"
        }

    Cookie example::

        {
          "type": "cookie",
          "cookie_name": "session",
          "cookie_env": "INGEST_SESSION_COOKIE"
        }

    Args:
        auth_block: Mapping extracted from ``seed.get("auth")``.

    Returns:
        Populated :class:`AuthConfig`, or ``None`` if no auth is defined.

    Raises:
        ValueError: If ``type`` is missing or unrecognised.
    """
    if not auth_block:
        return None

    auth_type = auth_block.get("type", "").strip().lower()
    if auth_type not in ("bearer", "basic", "cookie"):
        raise ValueError(
            f"Unsupported auth type '{auth_type}'. "
            "Must be one of: bearer, basic, cookie"
        )

    return AuthConfig(
        auth_type=auth_type,
        token=auth_block.get("token"),
        token_env=auth_block.get("token_env"),
        username=auth_block.get("username"),
        username_env=auth_block.get("username_env"),
        password=auth_block.get("password"),
        password_env=auth_block.get("password_env"),
        cookie_name=auth_block.get("cookie_name"),
        cookie_value=auth_block.get("cookie_value"),
        cookie_env=auth_block.get("cookie_env"),
    )


def build_auth_headers(auth_config: Optional[AuthConfig]) -> Dict[str, str]:
    """Build HTTP request headers for an authenticated source.

    Resolves environment variable references at call time so that secrets
    are never cached in dataclass fields longer than necessary.

    Args:
        auth_config: Populated :class:`AuthConfig`, or ``None`` for no auth.

    Returns:
        Dictionary of HTTP headers to merge into the request.  Returns an
        empty dict when ``auth_config`` is ``None``.

    Raises:
        ValueError: If a required credential is absent from both the config
            and the referenced environment variable.
    """
    if auth_config is None:
        return {}

    if auth_config.auth_type == "bearer":
        token = auth_config.token or (
            os.environ.get(auth_config.token_env, "") if auth_config.token_env else ""
        )
        if not token:
            raise ValueError(
                "Bearer auth configured but no token found. "
                "Set 'token' or 'token_env' in the seed auth block, "
                "and ensure the environment variable is populated."
            )
        return {"Authorization": f"Bearer {token}"}

    if auth_config.auth_type == "basic":
        username = auth_config.username or (
            os.environ.get(auth_config.username_env, "") if auth_config.username_env else ""
        )
        password = auth_config.password or (
            os.environ.get(auth_config.password_env, "") if auth_config.password_env else ""
        )
        if not username or not password:
            raise ValueError(
                "Basic auth configured but username or password is missing. "
                "Set 'username_env' / 'password_env' in the seed auth block."
            )
        credentials = base64.b64encode(
            f"{username}:{password}".encode("utf-8")
        ).decode("ascii")
        return {"Authorization": f"Basic {credentials}"}

    if auth_config.auth_type == "cookie":
        cookie_value = auth_config.cookie_value or (
            os.environ.get(auth_config.cookie_env, "") if auth_config.cookie_env else ""
        )
        if not cookie_value:
            raise ValueError(
                "Cookie auth configured but no cookie value found. "
                "Set 'cookie_value' or 'cookie_env' in the seed auth block."
            )
        cookie_name = auth_config.cookie_name or "session"
        return {"Cookie": f"{cookie_name}={cookie_value}"}

    # Unreachable after parse_seed_auth validation, but guard defensively.
    raise ValueError(f"Unrecognised auth_type: {auth_config.auth_type}")


def check_ollama_availability(
    logger: logging.Logger,
    skip_llm: bool = False,
    ollama_host: Optional[str] = None,
) -> bool:
    """Check if Ollama is running and accessible.

    Attempts to connect to Ollama and verify it is responding. Respects the
    ``OLLAMA_HOST`` environment variable so that dev-container setups pointing
    at a WSL-hosted Ollama instance work without code changes.

    Args:
        logger: Logger instance for output.
        skip_llm: If True, skip this check (the caller has opted out of LLM).
        ollama_host: Override the Ollama base URL. Falls back to the
            ``OLLAMA_HOST`` environment variable, then ``http://localhost:11434``.

    Returns:
        True if Ollama is available or skip_llm is True, False otherwise.
    """
    if skip_llm:
        logger.info("LLM preprocessing skipped (--skip-llm-preprocess flag set)")
        return True

    base = ollama_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    url = base.rstrip("/") + "/api/tags"

    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            logger.info("✓ Ollama is running and responding")
            return True
    except requests.ConnectionError:
        logger.warning(f"✗ Ollama not running: Connection refused to {base}")
        logger.warning("  Start Ollama with: ollama serve")
        logger.warning("  Or use --skip-llm-preprocess to skip LLM-based preprocessing")
        return False
    except requests.Timeout:
        logger.warning(f"✗ Ollama not responding: Connection timeout to {base}")
        logger.warning("  Or use --skip-llm-preprocess to skip LLM-based preprocessing")
        return False
    except Exception as e:
        logger.warning(f"✗ Could not check Ollama availability: {e}")
        logger.warning("  Or use --skip-llm-preprocess to skip LLM-based preprocessing")
        return False

    return False


def compute_doc_id(path: str) -> str:
    """Generate a stable document identifier from a file path.

    Creates a consistent document ID by extracting the filename without its
    extension. This ID is used throughout the system to track document
    versions and relationships.

    Args:
        path: Absolute or relative file path.

    Returns:
        Document identifier (filename without extension).

    Example:
        >>> compute_doc_id("/path/to/MyDocument.html")
        'MyDocument'
    """
    base = os.path.basename(path)
    doc_id, _ = os.path.splitext(base)
    return doc_id


def compute_file_hash(path: str) -> str:
    """Compute SHA-256 hash of file contents for change detection.

    Calculates a fingerprint of the file to detect modifications between
    ingestion runs. Uses 8 KB chunked reading for memory efficiency with
    large files.

    If the file cannot be read (e.g. it does not exist or access is denied),
    the hash of the path string is returned as a stable fallback so that
    callers always receive a valid hex digest.

    Args:
        path: Path to the file to hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except Exception:
        # Fallback: hash the path string so callers always get a valid digest.
        h.update(path.encode("utf-8"))
    return h.hexdigest()


def compute_chunk_hash(chunk_text: str) -> str:
    """Compute SHA-256 hash of chunk text for idempotency checking.

    Enables chunk-level deduplication: skip re-processing chunks whose
    content is identical across document versions.

    Args:
        chunk_text: Text content of the chunk.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).
    """
    h = hashlib.sha256()
    h.update(chunk_text.encode("utf-8"))
    return h.hexdigest()
