"""Tests for academic ingestion stage 5 (chunk + store)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from scripts.ingest.ingest_academic import (
    _format_toc_validation_warning,
    stage_chunk_and_store,
    stage_store_figure_chunks,
    validate_thesis_chunk_structure,
)
from scripts.ingest.pdfparser import extract_structure_from_text


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, msg):
        self.infos.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)

    def error(self, msg, exc_info=False):
        self.errors.append(msg)

    def debug(self, msg):
        pass


def test_validate_thesis_chunk_structure_requires_valid_spans_and_chapter_mapping():
    """Thesis chunks require exact, single-chapter source spans before storage."""
    text = "## Chapter 1\n\nFirst text.\n\n## Chapter 2\n\nSecond text."
    structure = extract_structure_from_text(text)
    first_start = text.index("First text.")
    second_start = text.index("Second text.")

    valid, report = validate_thesis_chunk_structure(
        [
            {
                "text": "First text.",
                "source_start": first_start,
                "source_end": first_start + len("First text."),
            },
            {
                "text": "Second text.",
                "source_start": second_start,
                "source_end": second_start + len("Second text."),
            },
        ],
        text,
        structure,
        min_coverage=1.0,
    )

    assert valid is True
    assert report["chapter_coverage"] == 1.0
    assert report["cross_chapter_chunks"] == 0

    invalid, invalid_report = validate_thesis_chunk_structure(
        [
            {
                "text": "First text.\n\n## Chapter 2\n\nSecond text.",
                "source_start": first_start,
                "source_end": second_start + len("Second text."),
            }
        ],
        text,
        structure,
        min_coverage=1.0,
    )

    assert invalid is False
    assert invalid_report["cross_chapter_chunks"] == 1


def test_toc_validation_warning_is_non_blocking_and_threshold_configurable():
    report = {
        "toc_present": True,
        "coverage": 2 / 3,
        "missing_chapters": ["Chapter 2"],
        "unexpected_chapters": ["Chapter 9: Unlisted"],
        "order_matches": False,
    }

    warning = _format_toc_validation_warning(report, minimum_coverage=0.9)

    assert warning is not None
    assert "coverage=66.7%" in warning
    assert "Chapter 2" in warning
    assert "Chapter 9: Unlisted" in warning
    assert _format_toc_validation_warning({"toc_present": False}, 0.95) is None
    assert (
        _format_toc_validation_warning(
            {"toc_present": True, "coverage": 1.0, "order_matches": True}, 0.95
        )
        is None
    )


def test_stage_chunk_and_store_skips_without_text():
    logger = DummyLogger()
    config = SimpleNamespace(dry_run=True)
    ok = stage_chunk_and_store({}, "", None, None, config, logger)
    assert ok is False


def test_stage_chunk_and_store_dry_run(monkeypatch, tmp_path):
    logger = DummyLogger()
    config = SimpleNamespace(dry_run=True)
    artifact = tmp_path / "ref.pdf"
    artifact.write_bytes(b"%PDF-1.4")

    ref = {"artifact_path": str(artifact), "reference_type": "academic"}
    ok = stage_chunk_and_store(ref, "Some reference text", None, None, config, logger)
    assert ok is True
    assert any("DRY_RUN" in msg for msg in logger.infos)


def test_stage_chunk_and_store_calls_store(monkeypatch, tmp_path):
    logger = DummyLogger()
    config = SimpleNamespace(
        dry_run=False,
        enable_parent_child_chunking=False,
        bm25_indexing_enabled=False,
    )
    artifact = tmp_path / "ref.pdf"
    artifact.write_bytes(b"%PDF-1.4")

    ref = {"artifact_path": str(artifact), "reference_type": "academic", "ref_id": "ref-1"}

    called = {"count": 0}
    stored_metadata = {}

    def fake_store(*args, **kwargs):
        called["count"] += 1
        stored_metadata.update(kwargs["metadata"])
        # Mock doesn't raise exceptions
        return None

    # Patch store_chunks_in_chroma from vectors module
    monkeypatch.setattr("scripts.ingest.ingest_academic.store_chunks_in_chroma", fake_store)

    ok = stage_chunk_and_store(ref, "Some reference text", object(), object(), config, logger)
    assert ok is True
    assert called["count"] == 1
    assert stored_metadata["ref_id"] == "ref-1"


def test_stage_store_figure_chunks_preserves_thesis_and_section_metadata(monkeypatch):
    stored = {}

    def fake_store_child_chunks(**kwargs):
        stored.update(kwargs)

    monkeypatch.setattr(
        "scripts.ingest.ingest_academic.store_child_chunks", fake_store_child_chunks
    )
    figures = [
        {
            "figure_number": 2,
            "page_number": 12,
            "chapter": "Chapter 3",
            "heading_path": "Chapter 3 > Findings",
            "caption": "Community governance model",
            "caption_number": "3",
            "alt_text": "Connections between community groups.",
            "description": "A diagram connecting community groups to governance roles.",
            "vision_status": "human_review_required",
            "vision_assessment": {
                "caption_description_alignment": "consistent",
                "alt_text_description_alignment": "inconsistent",
                "body_text_discussion": "interprets",
                "body_text_evidence": "This figure explains the governance model.",
            },
        }
    ]

    stage_store_figure_chunks(
        "collection",
        thesis_id="thesis-001",
        source_path="thesis.pdf",
        file_hash="fixture-hash",
        figures=figures,
        logger=DummyLogger(),
    )

    assert stored["doc_id"] == "thesis-001"
    assert stored["chunk_type"] == "figure"
    assert stored["child_chunks"][0]["metadata"]["heading_path"] == "Chapter 3 > Findings"
    assert stored["child_chunks"][0]["metadata"]["caption_number"] == "3"
    assert stored["child_chunks"][0]["metadata"]["page_number"] == 12
    assert "Community governance model" in stored["child_chunks"][0]["text"]
    assert "A diagram connecting community groups" in stored["child_chunks"][0]["text"]
    assert "Caption-description alignment: consistent" in stored["child_chunks"][0]["text"]
    assert "Alt-text-description alignment: inconsistent" in stored["child_chunks"][0]["text"]
    assert "Body-text discussion: interprets" in stored["child_chunks"][0]["text"]
    assert "This figure explains the governance model" in stored["child_chunks"][0]["text"]


def test_stage_chunk_and_store_preserves_source_thesis_identity(monkeypatch, tmp_path):
    """Source thesis chunks use the citation graph document ID and retain their source kind."""
    logger = DummyLogger()
    config = SimpleNamespace(
        dry_run=False,
        enable_parent_child_chunking=False,
        bm25_indexing_enabled=False,
    )
    artifact = tmp_path / "thesis.pdf"
    artifact.write_bytes(b"%PDF-1.4")
    stored = {}

    def fake_store(**kwargs):
        stored.update(kwargs)

    monkeypatch.setattr("scripts.ingest.ingest_academic.store_chunks_in_chroma", fake_store)

    assert stage_chunk_and_store(
        {
            "ref_id": "source-thesis-id",
            "source": "thesis_document",
            "artifact_path": str(artifact),
        },
        "Source thesis content.",
        object(),
        object(),
        config,
        logger,
    )
    assert stored["doc_id"] == "source-thesis-id"
    assert stored["metadata"]["source_kind"] == "thesis_document"


def test_stage_chunk_and_store_persists_chapter_metadata_for_thesis(monkeypatch, tmp_path):
    """Thesis child chunks retain source spans and mapped chapter metadata at storage."""
    logger = DummyLogger()
    config = SimpleNamespace(
        dry_run=False,
        enable_parent_child_chunking=True,
        bm25_indexing_enabled=False,
        thesis_structure_min_coverage=0.95,
    )
    artifact = tmp_path / "thesis.pdf"
    artifact.write_bytes(b"%PDF-1.4")
    text = (
        "## Chapter 1: Introduction\n\n"
        + ("Opening text. " * 80)
        + "\n\n## Chapter 2: Methods\n\n"
        + ("Method text. " * 80)
    )
    stored_children = []

    monkeypatch.setattr(
        "scripts.ingest.ingest_academic.store_chunks_in_chroma", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "scripts.ingest.ingest_academic.store_child_chunks",
        lambda **kwargs: stored_children.extend(kwargs["child_chunks"]),
    )
    monkeypatch.setattr("scripts.ingest.ingest_academic.store_parent_chunks", lambda **kwargs: None)

    assert stage_chunk_and_store(
        {
            "ref_id": "source-thesis-id",
            "source": "thesis_document",
            "artifact_path": str(artifact),
        },
        text,
        object(),
        object(),
        config,
        logger,
    )
    assert stored_children
    assert all(chunk["source_end"] > chunk["source_start"] >= 0 for chunk in stored_children)


def test_stage_chunk_and_store_refresh_replaces_existing_thesis(monkeypatch, tmp_path):
    """Refresh removes only the selected thesis before storing chapter-aware chunks."""
    logger = DummyLogger()
    config = SimpleNamespace(
        dry_run=False,
        enable_parent_child_chunking=True,
        bm25_indexing_enabled=False,
        thesis_structure_min_coverage=0.95,
        replace_existing_thesis=True,
        rag_data_path=str(tmp_path / "rag_data"),
    )
    artifact = tmp_path / "thesis.pdf"
    artifact.write_bytes(b"%PDF-1.4")
    text = "## Chapter 1\n\n" + ("Thesis text. " * 100)
    deleted_chunk_ids = []

    class DocCollection:
        def __init__(self):
            self.deleted = []

        def delete(self, where):
            self.deleted.append(where)

    class CacheDb:
        def __init__(self):
            self.deleted = []

        def delete_bm25_document(self, doc_id):
            self.deleted.append(doc_id)

        def close(self):
            pass

    doc_collection = DocCollection()
    cache_db = CacheDb()
    monkeypatch.setattr(
        "scripts.ingest.ingest_academic.delete_document_chunks",
        lambda doc_id, collection: deleted_chunk_ids.append(doc_id),
    )
    monkeypatch.setattr(
        "scripts.ingest.ingest_academic.get_cache_client", lambda **kwargs: cache_db
    )
    monkeypatch.setattr(
        "scripts.ingest.ingest_academic.store_chunks_in_chroma", lambda **kwargs: None
    )
    monkeypatch.setattr("scripts.ingest.ingest_academic.store_child_chunks", lambda **kwargs: None)
    monkeypatch.setattr("scripts.ingest.ingest_academic.store_parent_chunks", lambda **kwargs: None)

    assert stage_chunk_and_store(
        {"ref_id": "source-thesis-id", "source": "thesis_document", "artifact_path": str(artifact)},
        text,
        object(),
        doc_collection,
        config,
        logger,
    )
    assert deleted_chunk_ids == ["source-thesis-id"]
    assert doc_collection.deleted == [{"doc_id": "source-thesis-id"}]
    assert cache_db.deleted == ["source-thesis-id"]


def test_stage_chunk_and_store_indexes_bm25_in_configured_rag_data_path(monkeypatch, tmp_path):
    """Academic BM25 writes use the same configured cache as RAG retrieval."""
    logger = DummyLogger()
    configured_rag_data = tmp_path / "configured-rag-data"
    config = SimpleNamespace(
        dry_run=False,
        enable_parent_child_chunking=False,
        bm25_indexing_enabled=True,
        bm25_index_original_text=True,
        rag_data_path=str(configured_rag_data),
    )
    artifact = tmp_path / "reference.pdf"
    artifact.write_bytes(b"%PDF-1.4")
    cache_db = MagicMock()
    cache_paths = []
    indexed = []

    monkeypatch.setattr(
        "scripts.ingest.ingest_academic.store_chunks_in_chroma", lambda **kwargs: None
    )

    def fake_get_cache_client(rag_data_path, enable_cache):
        cache_paths.append((rag_data_path, enable_cache))
        return cache_db

    def fake_index_chunks(**kwargs):
        indexed.append(kwargs)
        return 1

    monkeypatch.setattr("scripts.ingest.ingest_academic.get_cache_client", fake_get_cache_client)
    monkeypatch.setattr("scripts.ingest.ingest_academic.index_chunks_in_bm25", fake_index_chunks)

    assert stage_chunk_and_store(
        {"artifact_path": str(artifact), "reference_type": "academic"},
        "Some reference text with enough content for chunking.",
        object(),
        object(),
        config,
        logger,
    )
    assert cache_paths == [(configured_rag_data, True)]
    assert indexed[0]["cache_db"] is cache_db


def test_stage_chunk_and_store_parent_child_storage(monkeypatch, tmp_path):
    logger = DummyLogger()
    config = SimpleNamespace(
        dry_run=False,
        enable_parent_child_chunking=True,
        bm25_indexing_enabled=False,
    )
    artifact = tmp_path / "ref.pdf"
    artifact.write_bytes(b"%PDF-1.4")

    ref = {"artifact_path": str(artifact), "reference_type": "academic", "ref_id": "ref-1"}

    calls = {
        "store_chunks": 0,
        "store_child": 0,
        "store_parent": 0,
        "chunks_to_store": None,
        "child_chunks": None,
        "parent_chunks": None,
        "child_metadata": None,
        "parent_metadata": None,
    }

    def fake_parent_child(text, doc_type, parent_size=None, child_size=None):
        return ["child-1", "child-2"], ["parent-1"]

    def fake_store(*args, **kwargs):
        calls["store_chunks"] += 1
        calls["chunks_to_store"] = kwargs.get("chunks")
        return None

    def fake_store_child(*args, **kwargs):
        calls["store_child"] += 1
        calls["child_chunks"] = kwargs.get("child_chunks")
        calls["child_metadata"] = kwargs.get("base_metadata")
        return None

    def fake_store_parent(*args, **kwargs):
        calls["store_parent"] += 1
        calls["parent_chunks"] = kwargs.get("parent_chunks")
        calls["parent_metadata"] = kwargs.get("base_metadata")
        return None

    monkeypatch.setattr(
        "scripts.ingest.ingest_academic.create_parent_child_chunks", fake_parent_child
    )
    monkeypatch.setattr("scripts.ingest.ingest_academic.store_chunks_in_chroma", fake_store)
    monkeypatch.setattr("scripts.ingest.ingest_academic.store_child_chunks", fake_store_child)
    monkeypatch.setattr("scripts.ingest.ingest_academic.store_parent_chunks", fake_store_parent)

    ok = stage_chunk_and_store(ref, "Some reference text", object(), object(), config, logger)

    assert ok is True
    assert calls["store_chunks"] == 1
    assert calls["chunks_to_store"] == []
    assert calls["store_child"] == 1
    assert calls["child_chunks"] == ["child-1", "child-2"]
    assert calls["store_parent"] == 1
    assert calls["parent_chunks"] == ["parent-1"]
    assert calls["child_metadata"] is not None
    assert calls["parent_metadata"] is not None
    assert calls["child_metadata"]["doc_type"] == "academic_reference"
    assert calls["parent_metadata"]["doc_type"] == "academic_reference"
    assert calls["child_metadata"]["version"] == 1
    assert calls["parent_metadata"]["version"] == 1
    assert calls["child_metadata"]["source"].endswith("ref.pdf")
    assert calls["parent_metadata"]["source"].endswith("ref.pdf")
    assert calls["child_metadata"]["doc_id"]
    assert calls["parent_metadata"]["doc_id"]
    assert calls["child_metadata"]["hash"]
    assert calls["parent_metadata"]["hash"]
    assert calls["child_metadata"]["ref_id"] == "ref-1"
    assert calls["parent_metadata"]["ref_id"] == "ref-1"
    assert calls["child_metadata"]["embedding_model"]
    assert calls["parent_metadata"]["embedding_model"]


class TestMetadataSanitisation:
    """Test ChromaDB metadata sanitisation logic."""

    def _sanitise_metadata(self, metadata: dict) -> dict:
        """Sanitise metadata for ChromaDB compatibility (mirrors production code)."""
        sanitised = {}
        for key, value in metadata.items():
            if value is None:
                continue  # Skip None values
            elif isinstance(value, (str, int, float, bool)):
                sanitised[key] = value
            elif isinstance(value, (dict, list)):
                # Convert complex types to JSON strings
                sanitised[key] = json.dumps(value)
            else:
                # Convert other types to string
                sanitised[key] = str(value)
        return sanitised

    def test_sanitise_dict_to_json_string(self):
        """Test that dict values are converted to JSON strings."""
        metadata = {
            "summary_scores": {"overall": 0, "confidence": 0.95},
            "name": "test",
        }

        sanitised = self._sanitise_metadata(metadata)

        # Dict should be converted to JSON string
        assert isinstance(sanitised["summary_scores"], str)
        assert sanitised["summary_scores"] == '{"overall": 0, "confidence": 0.95}'
        # String should stay string
        assert sanitised["name"] == "test"

    def test_sanitise_list_to_json_string(self):
        """Test that list values are converted to JSON strings."""
        metadata = {
            "technical_entities": ["ML", "NLP", "RNN"],
            "tags": "important",
        }

        sanitised = self._sanitise_metadata(metadata)

        # List should be converted to JSON string
        assert isinstance(sanitised["technical_entities"], str)
        assert sanitised["technical_entities"] == '["ML", "NLP", "RNN"]'
        # String should stay string
        assert sanitised["tags"] == "important"

    def test_sanitise_primitive_types_unchanged(self):
        """Test that primitive types remain unchanged."""
        metadata = {
            "doc_id": "doc_123",
            "section_depth": 2,
            "timestamp": 1707244500,
            "contains_table": True,
            "score": 0.95,
        }

        sanitised = self._sanitise_metadata(metadata)

        # All primitives should remain unchanged and of same type
        assert sanitised["doc_id"] == "doc_123"
        assert isinstance(sanitised["doc_id"], str)

        assert sanitised["section_depth"] == 2
        assert isinstance(sanitised["section_depth"], int)

        assert sanitised["timestamp"] == 1707244500
        assert isinstance(sanitised["timestamp"], int)

        assert sanitised["contains_table"] is True
        assert isinstance(sanitised["contains_table"], bool)

        assert sanitised["score"] == 0.95
        assert isinstance(sanitised["score"], float)

    def test_sanitise_skip_none_values(self):
        """Test that None values are skipped."""
        metadata = {
            "heading_path": None,
            "chapter": None,
            "doc_id": "doc_123",
            "parent_section": None,
        }

        sanitised = self._sanitise_metadata(metadata)

        # None values should be absent from sanitised dict
        assert "heading_path" not in sanitised
        assert "chapter" not in sanitised
        assert "parent_section" not in sanitised
        # Other values should be present
        assert sanitised["doc_id"] == "doc_123"

    def test_sanitise_chromadb_compatible_output(self):
        """Test that sanitised metadata passes ChromaDB type validation."""
        metadata = {
            "doc_id": "refnew_123_title",
            "summary_scores": {"overall": 0},
            "timestamp": 1707244500,
            "technical_entities": ["term1", "term2"],
            "section_depth": 2,
            "heading_path": None,
            "invalid_type": {"nested": "dict"},
        }

        sanitised = self._sanitise_metadata(metadata)

        # All values should be valid ChromaDB types
        for key, value in sanitised.items():
            assert isinstance(
                value, (str, int, float, bool)
            ), f"Key '{key}' has invalid type {type(value).__name__}: {value}"

        # Verify specific conversions
        assert isinstance(json.loads(sanitised["summary_scores"]), dict)  # Was valid JSON
        assert isinstance(json.loads(sanitised["technical_entities"]), list)  # Was valid JSON

    def test_sanitise_complex_academic_metadata(self):
        """Test sanitisation with realistic academic reference metadata."""
        # This mirrors the actual metadata structure from stage_chunk_and_store
        metadata = {
            "doc_type": "academic_reference",
            "summary": "Abstract text here",
            "summary_scores": json.dumps({"overall": 0}),  # Already JSON from creation
            "source_category": "academic_reference",
            "display_name": "Smith & Jones (2020)",
            "reference_type": "academic",
            "chunk_index": "0",
            "embedding_model": "mxbai-embed-large",
            "doc_id": "Smith_2020_SomeTitle",
            "file_hash": "abc123def456",
            "timestamp": 1707244500,
            "heading_path": None,
            "parent_section": None,
            "section_title": None,
            "chapter": None,
            "section_depth": 0,
            "content_type": "text",
            "contains_table": False,
            "contains_diagram": False,
            "technical_entities": "term1,term2",
            "is_api_reference": False,
            "is_configuration": False,
        }

        sanitised = self._sanitise_metadata(metadata)

        # Verify all values are ChromaDB compatible
        for key, value in sanitised.items():
            assert isinstance(
                value, (str, int, float, bool)
            ), f"Key '{key}' has invalid type {type(value).__name__}"

        # None values should not be present
        assert "heading_path" not in sanitised
        assert "parent_section" not in sanitised
        assert "section_title" not in sanitised
        assert "chapter" not in sanitised

        # Essential fields should be present
        assert sanitised["doc_id"] == "Smith_2020_SomeTitle"
        assert sanitised["doc_type"] == "academic_reference"
        assert sanitised["section_depth"] == 0

    def test_sanitise_empty_dict_and_list(self):
        """Test sanitisation of empty dict and list."""
        metadata = {
            "empty_dict": {},
            "empty_list": [],
            "normal_value": "test",
        }

        sanitised = self._sanitise_metadata(metadata)

        # Empty dict and list should convert to JSON strings
        assert sanitised["empty_dict"] == "{}"
        assert sanitised["empty_list"] == "[]"
        assert sanitised["normal_value"] == "test"

    def test_sanitise_nested_structures(self):
        """Test sanitisation of nested dict/list structures."""
        metadata = {
            "nested_dict": {
                "level1": {
                    "level2": ["value1", "value2"],
                    "score": 0.95,
                },
                "count": 10,
            },
            "list_of_dicts": [
                {"name": "item1", "score": 0.8},
                {"name": "item2", "score": 0.9},
            ],
        }

        sanitised = self._sanitise_metadata(metadata)

        # Both should convert to JSON strings
        assert isinstance(sanitised["nested_dict"], str)
        assert isinstance(sanitised["list_of_dicts"], str)

        # Should be valid JSON
        nested = json.loads(sanitised["nested_dict"])
        assert nested["level1"]["level2"] == ["value1", "value2"]

        list_data = json.loads(sanitised["list_of_dicts"])
        assert list_data[0]["name"] == "item1"
        assert list_data[1]["score"] == 0.9
