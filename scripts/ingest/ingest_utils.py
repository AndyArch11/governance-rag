"""Shared ingestion utility functions used across all ingest entrypoints.

Centralises common helpers so that `ingest.py`, `ingest_git.py`, and
`ingest_academic.py` do not duplicate logic.

Provided utilities:
- :func:`check_ollama_availability` — verify Ollama is reachable before LLM calls
- :func:`compute_doc_id` — stable document identifier from file path
- :func:`compute_file_hash` — SHA-256 fingerprint of file contents
- :func:`compute_chunk_hash` — SHA-256 fingerprint of chunk text
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)


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
