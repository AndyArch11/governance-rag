"""Tests for ingestion configuration guardrail settings."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.consistency_graph.consistency_config import ConsistencyConfig
from scripts.ingest.ingest_config import IngestConfig
from scripts.rag.rag_config import RAGConfig
from scripts.utils.embedding_model_config import EmbeddingConfig
from scripts.utils.llm_model_config import get_llm_model_spec


def test_relative_rag_data_path_is_project_root_anchored(monkeypatch) -> None:
    """Relative .env paths are stable when commands run from subdirectories."""
    monkeypatch.setenv("RAG_DATA_PATH", "./rag_data")

    config = IngestConfig()

    assert Path(config.rag_data_path) == Path(__file__).resolve().parents[1] / "rag_data"


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


def test_embedding_model_default() -> None:
    """Embedding configuration defaults to mxbai-embed-large."""
    with patch.dict("os.environ", {}, clear=False):
        cfg = EmbeddingConfig()
    assert cfg.model_name == "mxbai-embed-large"
    assert cfg.model_spec.dimensions == 1024
    assert cfg.model_spec.max_tokens == 512


def test_embedding_model_nomic_override() -> None:
    """Embedding configuration loads the selected model's capabilities."""
    with patch.dict("os.environ", {"EMBEDDING_MODEL_NAME": "nomic-embed-text"}, clear=False):
        cfg = EmbeddingConfig()
    assert cfg.model_name == "nomic-embed-text"
    assert cfg.model_spec.dimensions == 768
    assert cfg.model_spec.max_tokens == 2048


def test_embedding_model_rejects_unsupported_model() -> None:
    """Unsupported models fail early rather than using invalid vector settings."""
    with patch.dict("os.environ", {"EMBEDDING_MODEL_NAME": "unknown-model"}, clear=False):
        with pytest.raises(ValueError, match="Unsupported EMBEDDING_MODEL_NAME"):
            EmbeddingConfig()


def test_ingest_llm_model_default_and_validator_default() -> None:
    """Ingestion and its validator use the current recommended model."""
    with patch.dict(
        "os.environ",
        {
            "INGEST_LLM_MODEL": "qwen3.6:27b",
            "INGEST_VALIDATOR_LLM_MODEL": "qwen3.6:27b",
        },
        clear=False,
    ):
        cfg = IngestConfig()
    assert cfg.llm_model_name == "qwen3.6:27b"
    assert cfg.validator_llm_model_name == "qwen3.6:27b"


def test_ingest_validator_can_use_supported_override() -> None:
    """A validator can use another explicitly supported model."""
    with patch.dict(
        "os.environ",
        {
            "INGEST_LLM_MODEL": "qwen3.6:27b",
            "INGEST_VALIDATOR_LLM_MODEL": "llama3.1:8b",
        },
        clear=False,
    ):
        cfg = IngestConfig()
    assert cfg.validator_llm_model_name == "llama3.1:8b"


def test_ingest_context_windows_respect_model_limits() -> None:
    """Ingestion role windows are capped by their selected model capabilities."""
    with patch.dict(
        "os.environ",
        {
            "INGEST_LLM_MODEL": "qwen2.5:7b-instruct",
            "INGEST_CONTEXT_WINDOW_TOKENS": "100000",
            "INGEST_VALIDATOR_LLM_MODEL": "llama3.1:8b",
            "INGEST_VALIDATOR_CONTEXT_WINDOW_TOKENS": "12000",
        },
        clear=False,
    ):
        cfg = IngestConfig()
    assert cfg.llm_context_window_tokens == 32768
    assert cfg.validator_llm_context_window_tokens == 12000


def test_rag_llm_model_uses_supported_override() -> None:
    """RAG generation uses the centrally validated selected model."""
    with patch.dict("os.environ", {"RAG_MODEL": "gemma3:27b"}, clear=False):
        cfg = RAGConfig()
    assert cfg.model_name == "gemma3:27b"


def test_consistency_llm_model_uses_supported_override() -> None:
    """Graph comparisons use the centrally validated selected model."""
    with patch.dict("os.environ", {"CONSISTENCY_LLM_MODEL": "llama3.1:8b"}, clear=False):
        cfg = ConsistencyConfig()
    assert cfg.llm_model_name == "llama3.1:8b"


def test_consistency_context_window_respects_model_limit() -> None:
    """Graph comparison window is capped by the selected model's capability."""
    with patch.dict(
        "os.environ",
        {
            "CONSISTENCY_LLM_MODEL": "qwen2.5:7b-instruct",
            "CONSISTENCY_CONTEXT_WINDOW_TOKENS": "100000",
        },
        clear=False,
    ):
        cfg = ConsistencyConfig()
    assert cfg.llm_context_window_tokens == 32768


def test_llm_model_rejects_unsupported_value() -> None:
    """Unsupported LLM names fail during configuration rather than generation."""
    with pytest.raises(ValueError, match="Unsupported LLM model"):
        get_llm_model_spec("unknown-model")


def test_llm_model_supports_current_reasoning_models() -> None:
    """The registry includes current general-purpose Ollama models."""
    qwen_spec = get_llm_model_spec("qwen3.6:27b")
    gemma_spec = get_llm_model_spec("gemma4:26b")
    deepseek_spec = get_llm_model_spec("deepseek-r1:32b")

    assert qwen_spec.parameter_scale == "27B"
    assert qwen_spec.max_context_tokens == 262144
    assert gemma_spec.parameter_scale == "26B"
    assert gemma_spec.max_context_tokens == 262144
    assert deepseek_spec.parameter_scale == "32B"
    assert deepseek_spec.max_context_tokens == 131072


def test_rag_context_window_respects_model_limit() -> None:
    """RAG never requests a context window larger than the selected model supports."""
    with patch.dict(
        "os.environ",
        {
            "RAG_MODEL": "qwen2.5:7b-instruct",
            "RAG_CONTEXT_WINDOW_TOKENS": "100000",
            "RAG_RESPONSE_TOKEN_RESERVE": "2048",
        },
        clear=False,
    ):
        cfg = RAGConfig()
    assert cfg.context_window_tokens == 32768
    assert cfg.response_token_reserve == 2048
    assert cfg.max_prompt_tokens == 30720
    assert cfg.max_context_chars == 61440


def test_llm_model_rejects_coding_specialist() -> None:
    """Coding-specialist models are excluded from general application roles."""
    with pytest.raises(ValueError, match="Unsupported LLM model"):
        get_llm_model_spec("qwen3-coder:30b")
