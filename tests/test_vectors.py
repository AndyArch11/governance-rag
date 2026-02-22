"""Tests for vectors module with mocked dependencies.

Covers validation, semantic checks, repair pipeline, health scoring,
previous version lookup, hash retrieval, deletion, and drift detection.
"""

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, *args, **kwargs):
        self.messages.append(("info", args, kwargs))

    def warning(self, *args, **kwargs):
        self.messages.append(("warning", args, kwargs))

    def error(self, *args, **kwargs):
        self.messages.append(("error", args, kwargs))

    def debug(self, *args, **kwargs):
        self.messages.append(("debug", args, kwargs))


class DummyLLM:
    def __init__(self):
        self.responses = []

    def set_responses(self, *responses):
        self.responses = list(responses)

    def invoke(self, prompt: str):  # noqa: D401
        if self.responses:
            return self.responses.pop(0)
        return ""


class DummyCollection:
    def __init__(self, metadatas=None):
        self.metadatas = metadatas or []
        self.add_calls = []
        self.delete_calls = []

    def count(self):  # noqa: D401
        return len(self.metadatas)

    def get(self, where=None, include=None):  # noqa: D401
        return {"metadatas": self.metadatas}

    def add(self, ids=None, documents=None, metadatas=None, embeddings=None):  # noqa: D401
        self.add_calls.append(
            {
                "ids": ids,
                "documents": documents,
                "metadatas": metadatas,
                "embeddings": embeddings,
            }
        )

    def delete(self, where=None):  # noqa: D401
        self.delete_calls.append(where)


@pytest.fixture()
def vectors_module(monkeypatch):
    """Load vectors with mocked dependencies and LLMs."""

    # Ensure ingest package is importable
    scripts_ingest_path = Path(__file__).parent.parent / "scripts" / "ingest"
    sys.path.insert(0, str(scripts_ingest_path))

    # Stub langchain_ollama before import
    dummy_langchain = types.ModuleType("langchain_ollama")

    class DummyOllamaLLM:  # noqa: D401
        def __init__(self, model=None):
            self.model = model

        def invoke(self, prompt: str):
            return ""

    class DummyOllamaEmbeddings:  # noqa: D401
        def __init__(self, model=None):
            self.model = model

        def embed_documents(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    dummy_langchain.OllamaLLM = DummyOllamaLLM
    dummy_langchain.OllamaEmbeddings = DummyOllamaEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_ollama", dummy_langchain)

    # Import vectors fresh
    sys.modules.pop("scripts.ingest.vectors", None)
    vectors = importlib.import_module("scripts.ingest.vectors")

    # Override logger and audit with controllable stubs
    dummy_logger = DummyLogger()
    monkeypatch.setattr(vectors, "get_logger", lambda: dummy_logger)
    monkeypatch.setattr(vectors, "audit", lambda *args, **kwargs: None)

    # Provide controllable LLMs
    vectors.dummy_llm = DummyLLM()  # helper if needed

    return vectors, dummy_logger


def test_validate_chunk_success_and_failure(vectors_module):
    vectors, _ = vectors_module

    good = vectors.validate_chunk("doc-1", "x" * 25, "doc")
    assert good.chunk_id == "doc-1"

    with pytest.raises(Exception):
        vectors.validate_chunk("doc-2", "too short", "doc")


def test_validate_chunk_semantics(vectors_module, monkeypatch):
    vectors, _ = vectors_module

    llm = DummyLLM()
    llm.set_responses('{"is_valid": true, "reason": "ok"}')
    monkeypatch.setattr(vectors, "get_LLM_validator", lambda: llm)
    monkeypatch.setattr(vectors, "extract_first_json_block", json.loads)

    is_valid, reason = vectors.validate_chunk_semantics("content", "policy")
    assert is_valid is True
    assert reason == "ok"


def test_process_and_validate_chunks_repair_flow(vectors_module, monkeypatch):
    vectors, _ = vectors_module

    # Force structural validation to pass
    monkeypatch.setattr(vectors, "validate_chunk", lambda *a, **k: True)

    # First semantic check fails, second succeeds after repair
    calls = iter([(False, "bad"), (True, "fixed")])
    monkeypatch.setattr(vectors, "validate_chunk_semantics", lambda *a, **k: next(calls))

    # Repair returns improved text
    monkeypatch.setattr(
        vectors,
        "repair_chunk_with_llm",
        lambda *a, **k: "repaired chunk content with enough length.",
    )

    processed, valid_count, repaired_count, failed_count = vectors.process_and_validate_chunks(
        ["broken chunk"], "doc1", "policy"
    )

    assert len(processed) == 1
    assert processed[0][1].startswith("repaired chunk content")
    assert valid_count == 0
    assert repaired_count == 1
    assert failed_count == 0


def test_compute_document_health(vectors_module):
    vectors, _ = vectors_module
    health = vectors.compute_document_health(
        summary_score=7,
        total_chunks=10,
        valid_chunks=6,
        repaired_chunks=2,
        failed_chunks=2,
        truncated_chunks=3,
        truncation_loss_avg_pct=12.5,
        truncation_chars_lost=250,
        drift=None,
        preprocess_time=1.2,
        ingest_time=2.5,
    )

    assert health["summary_score"] == 7
    assert health["total_chunks"] == 10
    assert health["chunk_validity_ratio"] == 0.6
    assert health["health_score"] > 0


def test_get_previous_version_metadata(vectors_module):
    vectors, _ = vectors_module
    collection = DummyCollection(
        metadatas=[
            {"version": 1, "summary": "v1", "key_topics": '["a"]'},
            {"version": 2, "summary": "v2", "key_topics": '["a", "b"]'},
        ]
    )

    meta = vectors.get_previous_version_metadata(collection, "doc1")
    assert meta == {
        "version": 1,
        "summary": "v1",
        "key_topics": ["a"],
    }


def test_get_existing_doc_hash(vectors_module):
    vectors, _ = vectors_module
    collection = DummyCollection(metadatas=[{"hash": "abc123"}])
    result = vectors.get_existing_doc_hash("doc1", collection)
    assert result == "abc123"


def test_delete_document_chunks(vectors_module):
    vectors, logger = vectors_module
    collection = DummyCollection()

    # Dry run should not call delete
    vectors.delete_document_chunks("doc1", collection, dry_run=True)
    assert collection.delete_calls == []
    assert any(level == "info" for level, *_ in logger.messages)

    # Real delete should record call
    logger.messages.clear()
    vectors.delete_document_chunks("doc1", collection, dry_run=False)
    assert collection.delete_calls == [{"doc_id": "doc1"}]


def test_detect_semantic_drift(vectors_module, monkeypatch):
    vectors, _ = vectors_module
    llm = DummyLLM()
    llm.set_responses('{"drift_detected": true, "severity": "high", "explanation": "changed"}')
    monkeypatch.setattr(vectors, "get_LLM_validator", lambda: llm)
    monkeypatch.setattr(vectors, "extract_first_json_block", json.loads)

    result = vectors.detect_semantic_drift(
        "old summary",
        "new summary",
        ["a"],
        ["b"],
    )
    assert result["drift_detected"] is True
    assert result["severity"] == "high"


def test_store_chunks_with_source_category(vectors_module, monkeypatch):
    """Test that source_category is included in stored metadata."""
    vectors, _ = vectors_module

    # Mock dependencies
    chunk_collection = DummyCollection()
    doc_collection = DummyCollection()
    monkeypatch.setattr(
        vectors,
        "OllamaEmbeddings",
        lambda **kw: type(
            "obj", (object,), {"embed_documents": lambda self, texts: [[0.1] * 1024 for _ in texts]}
        )(),
    )

    # Mock validation to pass all chunks without semantic validation
    monkeypatch.setattr(
        vectors,
        "validate_chunk_semantics",
        lambda chunk, doc_type, chunk_hash=None, llm_cache=None: (True, "valid"),
    )

    metadata = {
        "cleaned_text": "This is a test document with enough content to pass validation checks.",
        "doc_type": "governance",
        "key_topics": ["security", "compliance"],
        "summary": "A test summary for governance documentation.",
        "summary_scores": {"relevance": 8, "clarity": 9, "completeness": 7, "overall": 8},
        "source_category": "Governance",
    }

    chunks = [
        "First chunk with test content for validation.",
        "Second chunk with additional test content for validation.",
    ]

    vectors.store_chunks_in_chroma(
        doc_id="test-doc",
        file_hash="abc123def456",
        source_path="/data/Governance/test-doc.html",
        version=1,
        chunks=chunks,
        metadata=metadata,
        chunk_collection=chunk_collection,
        doc_collection=doc_collection,
        preprocess_duration=1.5,
        ingest_duration=2.0,
        dry_run=False,
    )

    # Verify chunk metadata includes source_category
    assert chunk_collection.add_calls
    chunk_metadata = chunk_collection.add_calls[0]["metadatas"][0]
    assert chunk_metadata["source_category"] == "Governance"
    assert chunk_metadata["embedding_model"] == vectors.EMBEDDING_MODEL_NAME

    # Verify doc metadata includes source_category
    assert doc_collection.add_calls
    doc_metadata = doc_collection.add_calls[0]["metadatas"][0]
    assert doc_metadata["source_category"] == "Governance"
    assert doc_metadata["embedding_model"] == vectors.EMBEDDING_MODEL_NAME


def test_store_chunks_empty_source_category(vectors_module, monkeypatch):
    """Test that empty source_category is handled correctly."""
    vectors, _ = vectors_module

    chunk_collection = DummyCollection()
    doc_collection = DummyCollection()
    monkeypatch.setattr(
        vectors,
        "OllamaEmbeddings",
        lambda **kw: type(
            "obj", (object,), {"embed_documents": lambda self, texts: [[0.1] * 1024 for _ in texts]}
        )(),
    )

    # Mock validation to pass all chunks without semantic validation
    monkeypatch.setattr(
        vectors,
        "validate_chunk_semantics",
        lambda chunk, doc_type, chunk_hash=None, llm_cache=None: (True, "valid"),
    )

    metadata = {
        "cleaned_text": "This is a test document with enough content to pass validation checks.",
        "doc_type": "general",
        "key_topics": ["test"],
        "summary": "A test summary.",
        "summary_scores": {"relevance": 8, "clarity": 9, "completeness": 7, "overall": 8},
        "source_category": None,
    }

    chunks = ["Test chunk content for validation purposes here."]

    vectors.store_chunks_in_chroma(
        doc_id="test-doc-2",
        file_hash="xyz789",
        source_path="/data/test-doc.html",
        version=1,
        chunks=chunks,
        metadata=metadata,
        chunk_collection=chunk_collection,
        doc_collection=doc_collection,
        preprocess_duration=1.5,
        ingest_duration=2.0,
        dry_run=False,
    )

    # Verify source_category defaults to empty string
    chunk_metadata = chunk_collection.add_calls[0]["metadatas"][0]
    assert chunk_metadata["source_category"] == ""

    doc_metadata = doc_collection.add_calls[0]["metadatas"][0]
    assert doc_metadata["source_category"] == ""


class TestChunkTruncation:
    """Tests for chunk truncation to fit embedding model context limits."""

    def test_truncate_chunk_short_text_no_truncation(self):
        """Short text should not be truncated."""
        from scripts.ingest.vectors import _truncate_chunk_to_token_limit

        short_text = "This is a short text that fits easily within the token limit."
        truncated, was_truncated = _truncate_chunk_to_token_limit(short_text)

        assert not was_truncated, "Short text should not be truncated"
        assert truncated == short_text, "Short text should remain unchanged"

    def test_truncate_chunk_long_text_truncation(self):
        """Long text should be truncated to fit token limit."""
        from scripts.ingest.vectors import (
            EMBEDDING_CONTEXT_SAFETY_MARGIN,
            EMBEDDING_MODEL_MAX_TOKEN_LIMIT,
            _truncate_chunk_to_token_limit,
        )

        # Create a very long text that exceeds token limit
        long_text = "This is a very long text. " * 500
        truncated, was_truncated = _truncate_chunk_to_token_limit(long_text)

        assert was_truncated, "Long text should be truncated"
        assert "[TRUNCATED]" in truncated, "Truncated text should have marker"
        assert len(truncated) < len(long_text), "Truncated text should be shorter"

    def test_truncate_chunk_respects_token_limit(self):
        """Truncated text should respect the token limit."""
        from scripts.ingest.vectors import (
            EMBEDDING_CONTEXT_SAFETY_MARGIN,
            EMBEDDING_USABLE_TOKEN_LIMIT,
            _truncate_chunk_to_token_limit,
        )

        # Create a long text
        long_text = "word " * 2000  # Create 2000 words

        truncated, was_truncated = _truncate_chunk_to_token_limit(long_text)

        # Verify truncation respects token limit
        # Uses conservative 1 token per 3 chars estimate
        max_chars = (EMBEDDING_USABLE_TOKEN_LIMIT * 3) - EMBEDDING_CONTEXT_SAFETY_MARGIN
        assert len(truncated) <= max_chars + 50, (
            f"Truncated text ({len(truncated)} chars) should be close to limit "
            f"({max_chars} chars with safety margin)"
        )

    def test_truncate_chunk_preserves_word_boundaries(self):
        """Truncation should prefer word boundaries."""
        from scripts.ingest.vectors import _truncate_chunk_to_token_limit

        # Create text with clear word boundaries
        words = ["word"] * 1500
        text = " ".join(words)
        truncated, was_truncated = _truncate_chunk_to_token_limit(text)

        # The truncated text should end with the truncation marker
        # and should not have partial words (e.g., "wor" instead of "word")
        if was_truncated:
            # Check that the marker is present
            assert "[TRUNCATED]" in truncated, "Should have truncation marker"
            # Check that we don't have incomplete words before the marker
            text_before_marker = truncated.replace(" [TRUNCATED]", "")
            # The text should be a series of complete words
            last_word = text_before_marker.split()[-1] if text_before_marker.split() else ""
            assert (
                last_word == "word" or last_word == ""
            ), f"Last word should be complete or empty, got '{last_word}'"

    def test_truncate_chunk_normalises_dot_leaders(self):
        """Long dot leaders should be collapsed to a single ellipsis."""
        from scripts.ingest.vectors import _truncate_chunk_to_token_limit

        text = (
            "Chapter 1. Introduction "
            + ". . . . . . . . . . . . . . . . . . . . . . . . . 1\n"
            + "Chapter 2. Methods ................................................ 5"
        )

        truncated, _ = _truncate_chunk_to_token_limit(text)

        # Should normalise dot leaders regardless of truncation
        assert ".." not in truncated, "Dot leaders should be collapsed"
        assert "…" in truncated, "Ellipsis should replace dot leaders"

    def test_truncate_chunk_custom_token_limit(self):
        """Truncation should respect custom token limit parameter."""
        from scripts.ingest.vectors import _truncate_chunk_to_token_limit

        text = "word " * 500
        custom_limit = 100  # Very restrictive limit

        truncated, was_truncated = _truncate_chunk_to_token_limit(text, max_tokens=custom_limit)

        # With 100 tokens and 1 token per 3 chars estimate, should truncate to ~300 chars
        max_chars = (custom_limit * 3) - 200
        assert (
            len(truncated) <= max_chars + 50
        ), f"Truncated text should respect custom token limit (100 tokens = {max_chars} chars)"

    def test_truncate_chunk_empty_text(self):
        """Empty text should not be truncated."""
        from scripts.ingest.vectors import _truncate_chunk_to_token_limit

        empty_text = ""
        truncated, was_truncated = _truncate_chunk_to_token_limit(empty_text)

        assert not was_truncated, "Empty text should not be truncated"
        assert truncated == empty_text, "Empty text should remain unchanged"

    def test_truncate_chunk_whitespace_only(self):
        """Whitespace-only text should not be truncated."""
        from scripts.ingest.vectors import _truncate_chunk_to_token_limit

        whitespace_text = "   \n\t  "
        truncated, was_truncated = _truncate_chunk_to_token_limit(whitespace_text)

        assert not was_truncated, "Whitespace-only text should not be truncated"
        assert truncated == whitespace_text, "Whitespace-only text should remain unchanged"

    def test_truncate_chunk_exactly_at_limit(self):
        """Text exactly at token limit should not be truncated."""
        from scripts.ingest.vectors import (
            EMBEDDING_CONTEXT_SAFETY_MARGIN,
            EMBEDDING_USABLE_TOKEN_LIMIT,
            _truncate_chunk_to_token_limit,
        )

        # Create text that's at the limit (accounting for [TRUNCATED] suffix buffer)
        # Uses conservative 1 token per 3 chars estimate
        # Reserve 12 chars for " [TRUNCATED]" suffix
        max_chars = (EMBEDDING_USABLE_TOKEN_LIMIT * 3) - EMBEDDING_CONTEXT_SAFETY_MARGIN - 12
        text = "x" * (max_chars - 10)  # Slightly under limit

        truncated, was_truncated = _truncate_chunk_to_token_limit(text)

        assert not was_truncated, "Text at/under limit should not be truncated"
        assert truncated == text, "Text at/under limit should remain unchanged"

    def test_truncate_chunk_just_over_limit(self):
        """Text just over token limit should be truncated."""
        from scripts.ingest.vectors import (
            EMBEDDING_CONTEXT_SAFETY_MARGIN,
            EMBEDDING_USABLE_TOKEN_LIMIT,
            _truncate_chunk_to_token_limit,
        )

        # Create text that's just over the limit (accounting for [TRUNCATED] suffix buffer)
        # Uses conservative 1 token per 3 chars estimate
        # Reserve 12 chars for " [TRUNCATED]" suffix
        max_chars = (EMBEDDING_USABLE_TOKEN_LIMIT * 3) - EMBEDDING_CONTEXT_SAFETY_MARGIN - 12
        text = "word " * ((max_chars // 5) + 50)  # Over the limit

        truncated, was_truncated = _truncate_chunk_to_token_limit(text)

        assert was_truncated, "Text over limit should be truncated"
        assert "[TRUNCATED]" in truncated, "Truncated text should have marker"


class TestDocumentSummaryTruncation:
    """Tests for document summary truncation before embedding."""

    def test_summary_short_no_truncation(self):
        """Short document summary should not be truncated."""
        from scripts.ingest.vectors import _truncate_chunk_to_token_limit

        short_summary = "This is a concise academic paper summary with key findings."
        truncated, was_truncated = _truncate_chunk_to_token_limit(short_summary)

        assert not was_truncated, "Short summary should not be truncated"
        assert truncated == short_summary, "Short summary should remain unchanged"

    def test_summary_long_gets_truncated(self):
        """Long document summary should be truncated to fit embedding limit."""
        from scripts.ingest.vectors import _truncate_chunk_to_token_limit

        # Create a very long summary (simulating a poorly extracted metadata)
        long_summary = (
            "This comprehensive doctoral thesis explores the intricate relationships "
            "between distributed systems architecture and microservices patterns, "
            "examining in detail the various trade-offs and considerations that "
            "practitioners must navigate when designing scalable cloud-native applications. "
        ) * 50  # Make it very long

        truncated, was_truncated = _truncate_chunk_to_token_limit(long_summary)

        assert was_truncated, "Long summary should be truncated"
        assert "[TRUNCATED]" in truncated, "Truncated summary should have marker"
        assert len(truncated) < len(long_summary), "Truncated summary should be shorter"

    def test_summary_respects_embedding_limit(self):
        """Truncated summary should fit within embedding model's token limit."""
        from scripts.ingest.vectors import (
            EMBEDDING_CONTEXT_SAFETY_MARGIN,
            EMBEDDING_USABLE_TOKEN_LIMIT,
            _truncate_chunk_to_token_limit,
        )

        # Create a summary that definitely exceeds the limit
        long_summary = "Academic research findings and methodologies. " * 500

        truncated, was_truncated = _truncate_chunk_to_token_limit(long_summary)

        # Verify it fits within the conservative token limit
        max_chars = (EMBEDDING_USABLE_TOKEN_LIMIT * 3) - EMBEDDING_CONTEXT_SAFETY_MARGIN
        assert (
            len(truncated) <= max_chars + 50
        ), f"Truncated summary ({len(truncated)} chars) should fit within limit ({max_chars} chars)"

    def test_summary_empty_string(self):
        """Empty summary should not cause issues."""
        from scripts.ingest.vectors import _truncate_chunk_to_token_limit

        empty_summary = ""
        truncated, was_truncated = _truncate_chunk_to_token_limit(empty_summary)

        assert not was_truncated, "Empty summary should not be truncated"
        assert truncated == "", "Empty summary should remain empty"

    def test_summary_metadata_field(self):
        """Test summary extraction from metadata dict (integration test)."""
        from scripts.ingest.vectors import _truncate_chunk_to_token_limit

        # Simulate metadata from academic ingestion
        metadata = {
            "doc_type": "academic_reference",
            "summary": "This paper discusses machine learning approaches in natural language processing. "
            * 20,
            "summary_scores": {"overall": 0},
        }

        summary_text = metadata.get("summary", "")
        truncated, was_truncated = _truncate_chunk_to_token_limit(summary_text)

        # Summary should be truncated if it exceeds limit
        if len(summary_text) > 721:  # Our conservative limit
            assert was_truncated, "Long metadata summary should be truncated"
            assert len(truncated) < len(summary_text), "Truncated should be shorter"

    def test_summary_preserves_meaning_start(self):
        """Truncated summary should preserve the beginning (most important content)."""
        from scripts.ingest.vectors import _truncate_chunk_to_token_limit

        summary = (
            "IMPORTANT_INTRO: This paper presents groundbreaking research. "
            "The methodology involves advanced techniques. "
        ) * 100

        truncated, was_truncated = _truncate_chunk_to_token_limit(summary)

        if was_truncated:
            # Should preserve the important intro
            assert "IMPORTANT_INTRO" in truncated, "Should preserve beginning of summary"
            assert "[TRUNCATED]" in truncated, "Should have truncation marker"

    def test_summary_academic_paper_realistic(self):
        """Test with realistic academic paper summary."""
        from scripts.ingest.vectors import _truncate_chunk_to_token_limit

        realistic_summary = (
            "This dissertation investigates the application of machine learning techniques "
            "to improve software engineering practices in distributed systems. The research "
            "focuses on three main areas: automated code review, predictive maintenance, "
            "and performance optimisation. Through empirical studies across multiple "
            "industrial case studies, we demonstrate significant improvements in code quality "
            "metrics and system reliability."
        )

        truncated, was_truncated = _truncate_chunk_to_token_limit(realistic_summary)

        # This realistic summary should NOT be truncated (it's reasonable length)
        assert not was_truncated, "Realistic summary should fit within limits"
        assert truncated == realistic_summary, "Realistic summary should be unchanged"
