"""Tests for academic ingestion stage 5 (chunk + store)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from scripts.ingest.ingest_academic import stage_chunk_and_store


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

    ref = {"artifact_path": str(artifact), "reference_type": "academic"}

    called = {"count": 0}

    def fake_store(*args, **kwargs):
        called["count"] += 1
        # Mock doesn't raise exceptions
        return None

    # Patch store_chunks_in_chroma from vectors module
    monkeypatch.setattr("scripts.ingest.ingest_academic.store_chunks_in_chroma", fake_store)

    ok = stage_chunk_and_store(ref, "Some reference text", object(), object(), config, logger)
    assert ok is True
    assert called["count"] == 1


def test_stage_chunk_and_store_parent_child_storage(monkeypatch, tmp_path):
    logger = DummyLogger()
    config = SimpleNamespace(
        dry_run=False,
        enable_parent_child_chunking=True,
        bm25_indexing_enabled=False,
    )
    artifact = tmp_path / "ref.pdf"
    artifact.write_bytes(b"%PDF-1.4")

    ref = {"artifact_path": str(artifact), "reference_type": "academic"}

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
            "contains_code": True,
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

        assert sanitised["contains_code"] is True
        assert isinstance(sanitised["contains_code"], bool)

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
            "contains_code": False,
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
            "contains_code": False,
            "contains_table": False,
            "contains_diagram": False,
            "technical_entities": "term1,term2",
            "code_language": None,
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
        assert "code_language" not in sanitised

        # Essential fields should be present
        assert sanitised["doc_id"] == "Smith_2020_SomeTitle"
        assert sanitised["doc_type"] == "academic_reference"
        assert sanitised["contains_code"] is False
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
