"""Tests for ingestion configuration guardrail settings."""

from __future__ import annotations

from unittest.mock import patch

from scripts.ingest.ingest_config import IngestConfig


def test_table_chunk_max_llm_chars_default() -> None:
    """Default table chunk LLM guardrail is 5000 characters."""
    with patch.dict("os.environ", {}, clear=False):
        cfg = IngestConfig()
    assert cfg.table_chunk_max_llm_chars == 5000


def test_table_chunk_max_llm_chars_env_override() -> None:
    """Environment override is respected for table chunk LLM guardrail."""
    with patch.dict("os.environ", {"TABLE_CHUNK_MAX_LLM_CHARS": "7200"}, clear=False):
        cfg = IngestConfig()
    assert cfg.table_chunk_max_llm_chars == 7200


def test_table_chunk_max_llm_chars_min_floor() -> None:
    """Guardrail applies a minimum floor of 500 characters."""
    with patch.dict("os.environ", {"TABLE_CHUNK_MAX_LLM_CHARS": "120"}, clear=False):
        cfg = IngestConfig()
    assert cfg.table_chunk_max_llm_chars == 500
