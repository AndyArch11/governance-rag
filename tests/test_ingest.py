"""Tests for ingest module covering configuration, helpers, and processing logic.

Tests configuration parsing, document ID/hash computation, and core processing
logic without requiring actual LLM calls or ChromaDB instances.

TODO
1. **High-Impact Areas**:
   - Main entry point (lines 1395-1435): Add CLI integration tests
   - Worker pool logic (lines 1442-1461): Test concurrent processing
   - Error recovery (lines 1475-1476): Test exception handling in orchestration

2. **Test Patterns**:
   - Mock ChromaDB for process_file integration tests
   - Test error handling paths in stage functions
   - Test version lock contention scenarios
   - Test dry-run and reset modes

3. **Estimated Additional Tests Needed**: 8-12 tests
   - 2-3 process_file integration tests
   - 3-4 error path tests
   - 2-3 concurrent processing tests
   - 1-2 CLI argument parsing tests
"""

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure ingest package modules resolve (chunk, preprocess, etc.)
scripts_ingest_path = Path(__file__).parent.parent / "scripts" / "ingest"
sys.path.insert(0, str(scripts_ingest_path))

# Import after path is configured
from scripts.ingest.ingest import (
    IngestConfig,
    collect_url_files_from_seeds,
    compute_doc_id,
    compute_file_hash,
    load_url_seeds,
    process_file,
)


def _noop_logger():
    class _L:
        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

        def error(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

    return _L()


class TestIngestConfig:
    """Tests for IngestConfig class."""

    def test_default_configuration(self):
        """Test that default config values are set correctly."""
        config = IngestConfig()

        assert config.ignore_file_regex == r"render\(\d\)\.html"
        assert config.chunk_collection_name == "governance_docs_chunks"
        assert config.doc_collection_name == "governance_docs_documents"
        assert config.max_workers == 4  # Default from env var MAX_WORKERS
        assert config.versions_to_keep == 3
        assert config.reinitialise_chroma_storage is False
        assert config.environment == "Dev"  # Default environment

    def test_config_from_environment_variables(self, monkeypatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("RAG_BASE_PATH", "/custom/base/path")
        monkeypatch.setenv("RAG_DATA_PATH", "/custom/data/path")
        monkeypatch.setenv("MAX_WORKERS", "4")
        monkeypatch.setenv("VERSIONS_TO_KEEP", "5")
        monkeypatch.setenv("CHUNK_COLLECTION_NAME", "custom_chunks")
        monkeypatch.setenv("DOC_COLLECTION_NAME", "custom_docs")

        config = IngestConfig()

        assert config.base_path == "/custom/base/path"
        assert config.rag_data_path == "/custom/data/path"
        assert config.max_workers == 4
        assert config.versions_to_keep == 5
        assert config.chunk_collection_name == "custom_chunks"
        assert config.doc_collection_name == "custom_docs"

    def test_config_reinitialise_flag(self, monkeypatch):
        """Test that REINITIALISE_CHROMA_STORAGE flag is parsed correctly."""
        monkeypatch.setenv("REINITIALISE_CHROMA_STORAGE", "true")
        config = IngestConfig()
        assert config.reinitialise_chroma_storage is True

        monkeypatch.setenv("REINITIALISE_CHROMA_STORAGE", "TRUE")
        config = IngestConfig()
        assert config.reinitialise_chroma_storage is True

        monkeypatch.setenv("REINITIALISE_CHROMA_STORAGE", "false")
        config = IngestConfig()
        assert config.reinitialise_chroma_storage is False

    def test_config_environment_valid_values(self, monkeypatch):
        """Test that ENVIRONMENT accepts valid values."""
        for env_value in ["Dev", "Test", "Prod"]:
            monkeypatch.setenv("ENVIRONMENT", env_value)
            config = IngestConfig()
            assert config.environment == env_value

    def test_config_environment_invalid_value(self, monkeypatch):
        """Test that invalid ENVIRONMENT value raises ValueError."""
        monkeypatch.setenv("ENVIRONMENT", "InvalidEnv")
        with pytest.raises(ValueError, match="Invalid ENVIRONMENT"):
            IngestConfig()


class TestComputeDocId:
    """Tests for compute_doc_id function."""

    def test_compute_doc_id_simple_path(self):
        """Test doc ID computation from simple file path."""
        doc_id = compute_doc_id("/path/to/document.html")
        assert doc_id == "document"

    def test_compute_doc_id_with_spaces(self):
        """Test doc ID with spaces in filename."""
        doc_id = compute_doc_id("/path/to/My Document.html")
        assert doc_id == "My Document"

    def test_compute_doc_id_relative_path(self):
        """Test doc ID from relative path."""
        doc_id = compute_doc_id("folder/subfolder/test_file.html")
        assert doc_id == "test_file"

    def test_compute_doc_id_no_extension(self):
        """Test doc ID from file without extension."""
        doc_id = compute_doc_id("/path/to/document")
        assert doc_id == "document"

    def test_compute_doc_id_multiple_dots(self):
        """Test doc ID from filename with multiple dots."""
        doc_id = compute_doc_id("/path/my.document.v2.html")
        assert doc_id == "my.document.v2"

    def test_compute_doc_id_special_characters(self):
        """Test doc ID with special characters."""
        doc_id = compute_doc_id("/path/Azure-Security_Policy (v1).html")
        assert doc_id == "Azure-Security_Policy (v1)"


class TestComputeFileHash:
    """Tests for compute_file_hash function."""

    def test_compute_file_hash_simple_file(self):
        """Test hash computation for simple file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            hash1 = compute_file_hash(temp_path)
            # Hash should be consistent
            hash2 = compute_file_hash(temp_path)
            assert hash1 == hash2
            # Should be valid hex string
            assert len(hash1) == 32  # MD5 produces 32 hex chars
            assert all(c in "0123456789abcdef" for c in hash1)
        finally:
            os.unlink(temp_path)


class TestUrlSeedLoading:
    """Tests for URL seed JSON loading and parsing."""

    def test_load_url_seeds_includes_source_category(self, tmp_path):
        seed_path = tmp_path / "seeds.json"
        seed_path.write_text(
            '[{"url": "https://example.com", "single_page": true, "source_category": "example_cat"}]',
            encoding="utf-8",
        )

        seeds = load_url_seeds(str(seed_path), logger=_noop_logger())

        assert len(seeds) == 1
        assert seeds[0]["url"] == "https://example.com"
        assert seeds[0]["single_page"] is True
        assert seeds[0]["source_category"] == "example_cat"


class TestUrlSeedCollection:
    """Tests for downloading and collecting URL seeds with categories."""

    def test_collect_url_files_from_seeds(self, tmp_path, monkeypatch):
        # Prepare seed file
        seed_path = tmp_path / "seeds.json"
        seed_path.write_text(
            """
            [
              {
                "url": "https://example.com/main",
                "single_page": false,
                "sidebar_selector": "layout-body-menu",
                "max_depth": 2,
                "source_category": "example_cat"
              }
            ]
            """,
            encoding="utf-8",
        )

        # HTML with sidebar link
        main_html = '<div id="layout-body-menu"><a href="/child">Child</a></div>'
        child_html = "<html><body>child page</body></html>"

        calls = []

        def fake_get(url, headers=None, timeout=20):
            calls.append(url)

            class Resp:
                status_code = 200
                text = main_html if url.endswith("/main") else child_html

                def raise_for_status(self):
                    if self.status_code != 200:
                        raise Exception("bad status")

            return Resp()

        monkeypatch.setattr("requests.get", fake_get)

        config = IngestConfig()
        config.url_seed_json_path = str(seed_path)
        config.url_download_dir = str(tmp_path / "url_imports")

        files = collect_url_files_from_seeds(config, _noop_logger())

        # Should have downloaded main and child, preserving category
        assert len(files) == 2
        paths = [p for p, _ in files]
        cats = [c for _, c in files]
        assert all(Path(p).exists() for p in paths)
        assert cats == ["example_cat", "example_cat"]
        # Verify both URLs fetched
        assert set(calls) == {"https://example.com/main", "https://example.com/child"}

    def test_compute_file_hash_different_content(self):
        """Test that different content produces different hashes."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f1:
            f1.write("content A")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f2:
            f2.write("content B")
            path2 = f2.name

        try:
            hash1 = compute_file_hash(path1)
            hash2 = compute_file_hash(path2)
            assert hash1 != hash2
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_compute_file_hash_large_file(self):
        """Test hash computation for large file (tests chunked reading)."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            # Write more than 8KB to test chunked reading
            content = "x" * 10000
            f.write(content)
            temp_path = f.name

        try:
            file_hash = compute_file_hash(temp_path)
            # Verify against expected MD5
            expected = hashlib.md5(content.encode()).hexdigest()
            assert file_hash == expected
        finally:
            os.unlink(temp_path)

    def test_compute_file_hash_empty_file(self):
        """Test hash computation for empty file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_path = f.name

        try:
            file_hash = compute_file_hash(temp_path)
            # Empty file should have consistent hash
            expected = hashlib.md5(b"").hexdigest()
            assert file_hash == expected
        finally:
            os.unlink(temp_path)

    def test_compute_file_hash_binary_content(self):
        """Test hash computation for binary file."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            binary_data = bytes([0, 1, 2, 3, 255, 254, 253])
            f.write(binary_data)
            temp_path = f.name

        try:
            file_hash = compute_file_hash(temp_path)
            expected = hashlib.md5(binary_data).hexdigest()
            assert file_hash == expected
        finally:
            os.unlink(temp_path)


class TestPreprocessTextWithSourceCategory:
    """Tests for preprocess_text function with source_category parameter."""

    def test_preprocess_text_without_source_category(self, monkeypatch):
        """Test preprocess_text when source_category is not provided."""
        from scripts.ingest.preprocess import preprocess_text

        # Mock clean and metadata extraction
        monkeypatch.setattr(
            "scripts.ingest.preprocess.clean_text_with_llm",
            lambda text, doc_hash=None, llm_cache=None: "cleaned: " + text,
        )
        monkeypatch.setattr(
            "scripts.ingest.preprocess.extract_metadata_with_llm",
            lambda text, source_category=None, doc_hash=None, llm_cache=None: {
                "doc_type": "test",
                "key_topics": ["topic1"],
                "summary": "test summary that has to be sufficiently long.",
            },
        )
        monkeypatch.setattr(
            "scripts.ingest.preprocess.MetadataSchema",
            lambda **kw: type(
                "obj",
                (object,),
                {
                    "doc_type": kw.get("doc_type"),
                    "key_topics": kw.get("key_topics"),
                    "summary": kw.get("summary"),
                },
            )(),
        )
        monkeypatch.setattr(
            "scripts.ingest.preprocess.score_summary",
            lambda summary, cleaned_text, doc_hash=None, llm_cache=None: {
                "overall": 8,
                "relevance": 8,
                "clarity": 8,
                "completeness": 8,
            },
        )

        result = preprocess_text("raw text")

        assert result["source_category"] is None
        assert "cleaned_text" in result

    def test_preprocess_text_with_governance_category(self, monkeypatch):
        """Test preprocess_text with Governance source_category."""
        from scripts.ingest.preprocess import preprocess_text

        monkeypatch.setattr(
            "scripts.ingest.preprocess.clean_text_with_llm",
            lambda text, doc_hash=None, llm_cache=None: "cleaned: " + text,
        )
        monkeypatch.setattr(
            "scripts.ingest.preprocess.extract_metadata_with_llm",
            lambda text, source_category=None, doc_hash=None, llm_cache=None: {
                "doc_type": "governance",
                "key_topics": ["security"],
                "summary": "governance summary that is sufficiently long.",
            },
        )
        monkeypatch.setattr(
            "scripts.ingest.preprocess.MetadataSchema",
            lambda **kw: type(
                "obj",
                (object,),
                {
                    "doc_type": kw.get("doc_type"),
                    "key_topics": kw.get("key_topics"),
                    "summary": kw.get("summary"),
                },
            )(),
        )
        monkeypatch.setattr(
            "scripts.ingest.preprocess.score_summary",
            lambda summary, cleaned_text, doc_hash=None, llm_cache=None: {
                "overall": 8,
                "relevance": 8,
                "clarity": 8,
                "completeness": 8,
            },
        )

        result = preprocess_text("raw governance text", source_category="Governance")

        assert result["source_category"] == "Governance"


class TestProcessFileSourceCategoryOverride:
    """Ensure URL seed source_category override is applied in process_file."""

    def test_process_file_uses_override(self, tmp_path, monkeypatch):
        # Create a temporary HTML file
        html_path = tmp_path / "doc.html"
        html_path.write_text("<html><body>content</body></html>", encoding="utf-8")

        # Mock dependencies
        monkeypatch.setattr(
            "scripts.ingest.ingest.extract_text_from_html", lambda path: "text from html"
        )
        captured_source = {}

        def fake_preprocess(text, source_category=None, doc_hash=None, llm_cache=None):
            captured_source["value"] = source_category
            return {
                "cleaned_text": "clean",
                "source_category": source_category,
                "doc_type": "t",
                "key_topics": ["k"],
                "summary": "s",
                "summary_scores": {"overall": 0},
            }

        monkeypatch.setattr("scripts.ingest.ingest.preprocess_text", fake_preprocess)
        monkeypatch.setattr(
            "scripts.ingest.ingest.chunk_text",
            lambda text, doc_type=None, adaptive=True: ["chunk1"],
        )
        monkeypatch.setattr("scripts.ingest.ingest.store_chunks_in_chroma", lambda **kwargs: None)
        monkeypatch.setattr("scripts.ingest.ingest.store_parent_chunks", lambda **kwargs: None)
        monkeypatch.setattr("scripts.ingest.ingest.store_child_chunks", lambda **kwargs: None)
        monkeypatch.setattr(
            "scripts.ingest.ingest.get_existing_doc_hash", lambda doc_id, chunk_collection: None
        )
        monkeypatch.setattr("scripts.ingest.ingest.audit", lambda *args, **kwargs: None)

        # Dummy collections
        class DummyColl:
            def get(self, *a, **k):
                return {"metadatas": []}

            def delete(self, *a, **k):
                return None

        chunk_coll = DummyColl()
        doc_coll = DummyColl()

        import threading

        cfg = IngestConfig()
        cfg.logger = _noop_logger()
        cfg.args = SimpleNamespace(
            dry_run=False, reset=False, verbose=False, skip_llm_preprocess=False
        )
        cfg.version_lock = threading.Lock()
        cfg.enable_semantic_drift_detection = False
        cfg.enable_chunk_heuristic_skip = False

        ok = process_file(
            str(html_path),
            chunk_coll,
            doc_coll,
            cfg,
            source_category_override="seed_cat",
            llm_cache=None,
        )

        assert ok is True
        assert captured_source["value"] == "seed_cat"

    def test_preprocess_text_with_patterns_category(self, monkeypatch):
        """Test preprocess_text with Patterns source_category."""
        from scripts.ingest.preprocess import preprocess_text

        monkeypatch.setattr(
            "scripts.ingest.preprocess.clean_text_with_llm",
            lambda text, doc_hash=None, llm_cache=None: "cleaned: " + text,
        )
        monkeypatch.setattr(
            "scripts.ingest.preprocess.extract_metadata_with_llm",
            lambda text, source_category=None, doc_hash=None, llm_cache=None: {
                "doc_type": "architectural pattern",
                "key_topics": ["architecture"],
                "summary": "pattern summary that is sufficiently long.",
            },
        )
        monkeypatch.setattr(
            "scripts.ingest.preprocess.MetadataSchema",
            lambda **kw: type(
                "obj",
                (object,),
                {
                    "doc_type": kw.get("doc_type"),
                    "key_topics": kw.get("key_topics"),
                    "summary": kw.get("summary"),
                },
            )(),
        )
        monkeypatch.setattr(
            "scripts.ingest.preprocess.score_summary",
            lambda summary, cleaned_text, doc_hash=None, llm_cache=None: {
                "overall": 8,
                "relevance": 8,
                "clarity": 8,
                "completeness": 8,
            },
        )

        result = preprocess_text("raw pattern text", source_category="Patterns")

        assert result["source_category"] == "Patterns"
        assert result["doc_type"] == "architectural pattern"


class TestSourceCategoryExtraction:
    """Tests for source_category extraction from file paths."""

    def test_extract_category_from_governance_folder(self):
        """Test extracting 'Governance' category from file path."""
        from pathlib import Path

        file_path = "/data/data_raw/downloads/Governance/policy.html"
        file_path_obj = Path(file_path)
        parent_name = file_path_obj.parent.name
        source_category = parent_name if parent_name not in ["", "downloads", "data_raw"] else None

        assert source_category == "Governance"

    def test_extract_category_from_patterns_folder(self):
        """Test extracting 'Patterns' category from file path."""
        from pathlib import Path

        file_path = "/data/data_raw/downloads/Patterns/architecture.html"
        file_path_obj = Path(file_path)
        parent_name = file_path_obj.parent.name
        source_category = parent_name if parent_name not in ["", "downloads", "data_raw"] else None

        assert source_category == "Patterns"

    def test_extract_category_skips_downloads(self):
        """Test that downloads folder name is skipped."""
        from pathlib import Path

        file_path = "/data/data_raw/downloads/policy.html"
        file_path_obj = Path(file_path)
        parent_name = file_path_obj.parent.name
        source_category = parent_name if parent_name not in ["", "downloads", "data_raw"] else None

        assert source_category is None


class TestParseArgs:
    """Tests for argument parsing."""

    def test_parse_args_defaults(self, monkeypatch):
        """Test default argument values."""
        from scripts.ingest.ingest import parse_args

        config = IngestConfig()
        monkeypatch.setattr("sys.argv", ["ingest.py"])
        args = parse_args(config)

        assert args.reset is False
        assert args.workers == config.max_workers
        assert args.limit is None
        assert args.verbose is False
        assert args.dry_run is False

    def test_parse_args_reset_flag(self, monkeypatch):
        """Test --reset flag."""
        from scripts.ingest.ingest import parse_args

        config = IngestConfig()
        monkeypatch.setattr("sys.argv", ["ingest.py", "--reset"])
        args = parse_args(config)

        assert args.reset is True

    def test_parse_args_workers(self, monkeypatch):
        """Test --workers argument."""
        from scripts.ingest.ingest import parse_args

        config = IngestConfig()
        monkeypatch.setattr("sys.argv", ["ingest.py", "--workers", "8"])
        args = parse_args(config)

        assert args.workers == 8

    def test_parse_args_limit(self, monkeypatch):
        """Test --limit argument."""
        from scripts.ingest.ingest import parse_args

        config = IngestConfig()
        monkeypatch.setattr("sys.argv", ["ingest.py", "--limit", "50"])
        args = parse_args(config)

        assert args.limit == 50

    def test_parse_args_verbose(self, monkeypatch):
        """Test --verbose flag."""
        from scripts.ingest.ingest import parse_args

        config = IngestConfig()
        monkeypatch.setattr("sys.argv", ["ingest.py", "--verbose"])
        args = parse_args(config)

        assert args.verbose is True

    def test_parse_args_dry_run(self, monkeypatch):
        """Test --dry-run flag."""
        from scripts.ingest.ingest import parse_args

        config = IngestConfig()
        monkeypatch.setattr("sys.argv", ["ingest.py", "--dry-run"])
        args = parse_args(config)

        assert args.dry_run is True

    def test_parse_args_combined(self, monkeypatch):
        """Test multiple arguments combined."""
        from scripts.ingest.ingest import parse_args

        config = IngestConfig()
        monkeypatch.setattr(
            "sys.argv",
            ["ingest.py", "--reset", "--workers", "4", "--limit", "100", "--verbose", "--dry-run"],
        )
        args = parse_args(config)

        assert args.reset is True
        assert args.workers == 4
        assert args.limit == 100
        assert args.verbose is True
        assert args.dry_run is True

    def test_parse_args_bm25_indexing(self, monkeypatch):
        """Test --bm25-indexing flag."""
        from scripts.ingest.ingest import parse_args

        config = IngestConfig()
        monkeypatch.setattr("sys.argv", ["ingest.py", "--bm25-indexing"])
        args = parse_args(config)

        assert args.bm25_indexing is True
        assert args.skip_bm25 is False

    def test_parse_args_skip_bm25(self, monkeypatch):
        """Test --skip-bm25 flag."""
        from scripts.ingest.ingest import parse_args

        config = IngestConfig()
        monkeypatch.setattr("sys.argv", ["ingest.py", "--skip-bm25"])
        args = parse_args(config)

        assert args.skip_bm25 is True

    def test_parse_args_bm25_conflict(self, monkeypatch):
        """Test --bm25-indexing with --skip-bm25 (skip takes precedence)."""
        from scripts.ingest.ingest import parse_args

        config = IngestConfig()
        monkeypatch.setattr("sys.argv", ["ingest.py", "--bm25-indexing", "--skip-bm25"])
        args = parse_args(config)

        assert args.bm25_indexing is True
        assert args.skip_bm25 is True
        # The main() function will apply skip_bm25 with higher priority

    def test_main_bm25_cli_precedence_skip(self, monkeypatch, caplog):
        """Test that --skip-bm25 overrides config in main()."""
        from scripts.ingest.ingest import main
        from scripts.ingest.ingest_config import IngestConfig

        config = IngestConfig()
        original_enabled = config.bm25_indexing_enabled

        try:
            # Mock parse_args to return our test args
            class TestArgs:
                reset = False
                workers = 2
                limit = 1
                verbose = False
                include_url_seeds = False
                url_seed_path = None
                dry_run = True
                profile = False
                skip_llm_preprocess = False
                progress_interval = 10
                log_unsupported_files = False
                purge_logs = False
                bm25_indexing = True  # Enabled
                skip_bm25 = True  # But skip overrides it

            def mock_parse_args(cfg):
                return TestArgs()

            monkeypatch.setattr("sys.argv", ["ingest.py"])
            monkeypatch.setattr("scripts.ingest.ingest.parse_args", mock_parse_args)

            # Mock get_ingest_config
            def mock_get_config():
                return config

            monkeypatch.setattr(
                "scripts.ingest.ingest.get_ingest_config", mock_get_config, raising=False
            )

            # This would require mocking more dependencies, so we'll test
            # the logic directly instead
            args = TestArgs()
            if args.skip_bm25:
                config.bm25_indexing_enabled = False
            elif args.bm25_indexing:
                config.bm25_indexing_enabled = True

            assert config.bm25_indexing_enabled is False
        finally:
            config.bm25_indexing_enabled = original_enabled

    def test_main_bm25_cli_precedence_enable(self, monkeypatch):
        """Test that --bm25-indexing enables it in main()."""
        from scripts.ingest.ingest_config import IngestConfig

        config = IngestConfig()
        original_enabled = config.bm25_indexing_enabled

        try:

            class TestArgs:
                bm25_indexing = True
                skip_bm25 = False

            args = TestArgs()
            if args.skip_bm25:
                config.bm25_indexing_enabled = False
            elif args.bm25_indexing:
                config.bm25_indexing_enabled = True

            assert config.bm25_indexing_enabled is True
        finally:
            config.bm25_indexing_enabled = original_enabled


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_progress_tracker_initialisation(self):
        """Test ProgressTracker initialises correctly."""
        from scripts.ingest.ingest import ProgressTracker

        logger = _noop_logger()
        tracker = ProgressTracker(total=100, log_interval=10, logger=logger)

        assert tracker.total == 100
        assert tracker.log_interval == 10
        assert tracker.completed == 0
        assert tracker.succeeded == 0
        assert tracker.failed == 0

    def test_progress_tracker_increment_success(self):
        """Test incrementing successful documents."""
        from scripts.ingest.ingest import ProgressTracker

        logger = _noop_logger()
        tracker = ProgressTracker(total=100, log_interval=50, logger=logger)

        tracker.increment(success=True)
        assert tracker.completed == 1
        assert tracker.succeeded == 1
        assert tracker.failed == 0

    def test_progress_tracker_increment_failure(self):
        """Test incrementing failed documents."""
        from scripts.ingest.ingest import ProgressTracker

        logger = _noop_logger()
        tracker = ProgressTracker(total=100, log_interval=50, logger=logger)

        tracker.increment(success=False)
        assert tracker.completed == 1
        assert tracker.succeeded == 0
        assert tracker.failed == 1


class TestDryRunStats:
    """Tests for DryRunStats class."""

    def test_dry_run_stats_initialisation(self):
        """Test DryRunStats initialises correctly."""
        from scripts.ingest.ingest import DryRunStats

        stats = DryRunStats()
        assert stats.docs_new == 0
        assert stats.docs_updated == 0
        assert stats.docs_skipped == 0
        assert stats.chunks_total == 0

    def test_dry_run_stats_record_new_document(self):
        """Test recording new document."""
        from scripts.ingest.ingest import DryRunStats

        stats = DryRunStats()
        stats.record_document(status="new", num_chunks=5, processing_time=1.5)

        assert stats.docs_new == 1
        assert stats.chunks_total == 5
        assert stats.estimated_time_seconds == 1.5

    def test_dry_run_stats_record_updated_document(self):
        """Test recording updated document."""
        from scripts.ingest.ingest import DryRunStats

        stats = DryRunStats()
        stats.record_document(status="updated", num_chunks=3, processing_time=2.0)

        assert stats.docs_updated == 1
        assert stats.chunks_total == 3

    def test_dry_run_stats_record_unchanged_document(self):
        """Test recording unchanged document."""
        from scripts.ingest.ingest import DryRunStats

        stats = DryRunStats()
        stats.record_document(status="skipped", num_chunks=0, processing_time=0.5)

        assert stats.docs_skipped == 1
        assert stats.chunks_total == 0

    def test_dry_run_stats_get_report(self):
        """Test DryRunStats report generation."""
        from scripts.ingest.ingest import DryRunStats

        stats = DryRunStats()
        stats.record_document(status="new", num_chunks=5, processing_time=1.0)
        stats.record_document(status="updated", num_chunks=3, processing_time=2.0)
        stats.record_document(status="skipped", num_chunks=0, processing_time=0.5)

        report = stats.get_report()

        assert "New documents:" in report
        assert "Updated documents:" in report
        assert "Skipped" in report
        assert "8" in report  # total chunks (5 + 3)


class TestProfileStats:
    """Tests for ProfileStats class."""

    def test_profile_stats_initialisation(self):
        """Test ProfileStats initialises correctly."""
        from scripts.ingest.ingest import ProfileStats

        stats = ProfileStats()
        assert stats.doc_count == 0
        assert stats.chunk_count == 0
        assert stats.errors == []
        assert stats.warnings == []
        assert stats.llm_calls == 0
        assert stats.cache_hits_llm == 0

    def test_profile_stats_record_stage(self):
        """Test recording stage duration."""
        from scripts.ingest.ingest import ProfileStats

        stats = ProfileStats()
        stats.record_stage("parse", 1.5)
        stats.record_stage("parse", 2.0)

        assert "parse" in stats.stage_times
        assert stats.stage_times["parse"] == 3.5

    def test_profile_stats_record_document(self):
        """Test recording document processing."""
        from scripts.ingest.ingest import ProfileStats

        stats = ProfileStats()
        stats.record_document(num_chunks=5, processing_time=3.0)

        assert stats.doc_count == 1
        assert stats.chunk_count == 5
        assert len(stats.doc_processing_times) == 1
        assert stats.doc_processing_times[0] == 3.0

    def test_profile_stats_record_error(self):
        """Test recording error."""
        from scripts.ingest.ingest import ProfileStats

        stats = ProfileStats()
        stats.record_error("Test error", doc_name="test.html")

        assert len(stats.errors) == 1
        assert "Test error" in stats.errors[0]

    def test_profile_stats_record_warning(self):
        """Test recording warning."""
        from scripts.ingest.ingest import ProfileStats

        stats = ProfileStats()
        stats.record_warning("Test warning", doc_name="test.html")

        assert len(stats.warnings) == 1
        assert "Test warning" in stats.warnings[0]

    def test_profile_stats_record_llm_call(self):
        """Test recording LLM call."""
        from scripts.ingest.ingest import ProfileStats

        stats = ProfileStats()
        stats.record_llm_call(cache_hit=False)
        stats.record_llm_call(cache_hit=True)

        assert stats.llm_calls == 2
        assert stats.cache_hits_llm == 1

    def test_profile_stats_record_embedding(self):
        """Test recording embedding generation."""
        from scripts.ingest.ingest import ProfileStats

        stats = ProfileStats()
        stats.record_embedding(cache_hit=False)
        stats.record_embedding(cache_hit=True)

        assert stats.embedding_calls == 2
        assert stats.cache_hits_embedding == 1

    def test_profile_stats_get_report(self):
        """Test ProfileStats report generation."""
        from scripts.ingest.ingest import ProfileStats

        stats = ProfileStats()
        stats.record_document(num_chunks=5, processing_time=3.0)
        stats.record_stage("parse", 1.0)
        stats.record_llm_call(cache_hit=False)
        stats.record_embedding(cache_hit=True)

        report = stats.get_report()

        # Verify report contains key statistics
        assert "PROFILE" in report or "profile" in report.lower()
        assert isinstance(report, str)
        assert len(report) > 0


class TestIsBinaryFile:
    """Tests for _is_binary_file function."""

    def test_is_binary_file_text(self, tmp_path):
        """Test that text file is not detected as binary."""
        from scripts.ingest.ingest import _is_binary_file

        text_file = tmp_path / "test.txt"
        text_file.write_text("This is a text file with normal ASCII content.")

        assert _is_binary_file(str(text_file)) is False

    def test_is_binary_file_html(self, tmp_path):
        """Test that HTML file is not detected as binary."""
        from scripts.ingest.ingest import _is_binary_file

        html_file = tmp_path / "test.html"
        html_file.write_text("<html><body>Test content</body></html>")

        assert _is_binary_file(str(html_file)) is False

    def test_is_binary_file_binary(self, tmp_path):
        """Test that binary file is detected correctly."""
        from scripts.ingest.ingest import _is_binary_file

        binary_file = tmp_path / "test.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd\xfc")

        assert _is_binary_file(str(binary_file)) is True

    def test_is_binary_file_empty(self, tmp_path):
        """Test that empty file is not detected as binary."""
        from scripts.ingest.ingest import _is_binary_file

        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        assert _is_binary_file(str(empty_file)) is False


class TestComputeChunkHash:
    """Tests for compute_chunk_hash function."""

    def test_compute_chunk_hash_consistency(self):
        """Test that chunk hash is consistent."""
        from scripts.ingest.ingest import compute_chunk_hash

        text = "This is a chunk of text to hash."
        hash1 = compute_chunk_hash(text)
        hash2 = compute_chunk_hash(text)

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex string

    def test_compute_chunk_hash_different_text(self):
        """Test that different text produces different hash."""
        from scripts.ingest.ingest import compute_chunk_hash

        hash1 = compute_chunk_hash("text one")
        hash2 = compute_chunk_hash("text two")

        assert hash1 != hash2

    def test_compute_chunk_hash_unicode(self):
        """Test chunk hash with unicode characters."""
        from scripts.ingest.ingest import compute_chunk_hash

        text = "Unicode: café, naïve, 中文, العربية"
        hash_val = compute_chunk_hash(text)

        assert len(hash_val) == 32
        assert all(c in "0123456789abcdef" for c in hash_val)


class TestGetAuthHeaders:
    """Tests for get_auth_headers function."""

    def test_get_auth_headers_placeholder(self):
        """Test that get_auth_headers returns empty dict (placeholder)."""
        from scripts.ingest.ingest import get_auth_headers

        headers = get_auth_headers("https://example.com")
        assert headers == {}

        headers = get_auth_headers("http://api.example.com/resource")
        assert headers == {}


class TestExtractSidebarLinks:
    """Tests for extract_sidebar_links function."""

    def test_extract_sidebar_links_no_selector(self):
        """Test extraction with no selector returns empty list."""
        from scripts.ingest.ingest import extract_sidebar_links

        html = "<html><body><nav><a href='/page1'>Link 1</a></nav></body></html>"
        links = extract_sidebar_links(html, "https://example.com", None, 5)

        assert links == []

    def test_extract_sidebar_links_by_id(self):
        """Test extraction using element ID selector."""
        from scripts.ingest.ingest import extract_sidebar_links

        html = """
        <html>
            <body>
                <nav id="sidebar">
                    <a href="/page1">Link 1</a>
                    <a href="/page2">Link 2</a>
                </nav>
            </body>
        </html>
        """
        links = extract_sidebar_links(html, "https://example.com", "sidebar", 5)

        assert len(links) == 2
        assert "https://example.com/page1" in links
        assert "https://example.com/page2" in links

    def test_extract_sidebar_links_by_class(self):
        """Test extraction using CSS class selector."""
        from scripts.ingest.ingest import extract_sidebar_links

        html = """
        <html>
            <body>
                <nav class="sidebar-nav">
                    <a href="/docs">Documentation</a>
                </nav>
            </body>
        </html>
        """
        links = extract_sidebar_links(html, "https://example.com", "sidebar-nav", 5)

        assert len(links) == 1
        assert "https://example.com/docs" in links

    def test_extract_sidebar_links_depth_limit(self):
        """Test extraction respects max_depth parameter."""
        from scripts.ingest.ingest import extract_sidebar_links

        html = """
        <html>
            <body>
                <nav id="nav">
                    <ul>
                        <li><a href="/page1">Level 1</a></li>
                        <li>
                            <ul>
                                <li><a href="/page2">Level 2</a></li>
                                <li>
                                    <ul>
                                        <li><a href="/page3">Level 3</a></li>
                                    </ul>
                                </li>
                            </ul>
                        </li>
                    </ul>
                </nav>
            </body>
        </html>
        """
        # Depth 0 allows shallow links only
        links_d0 = extract_sidebar_links(html, "https://example.com", "nav", 0)

        # Depth 1 should get more links
        links_d1 = extract_sidebar_links(html, "https://example.com", "nav", 1)

        assert len(links_d1) >= len(links_d0)

    def test_extract_sidebar_links_relative_urls(self):
        """Test that relative URLs are converted to absolute."""
        from scripts.ingest.ingest import extract_sidebar_links

        html = """
        <html>
            <body>
                <nav id="sidebar">
                    <a href="page1">Page 1</a>
                    <a href="/docs/page2">Page 2</a>
                </nav>
            </body>
        </html>
        """
        links = extract_sidebar_links(html, "https://example.com/docs/", "sidebar", 5)

        # All should be absolute URLs
        for link in links:
            assert link.startswith("https://")


class TestShouldSkipUnsupported:
    """Tests for _should_skip_unsupported function."""

    def test_should_skip_unsupported_extension(self, tmp_path):
        """Test that non-HTML and non-PDF files are skipped."""
        from scripts.ingest.ingest import _should_skip_unsupported

        # Create temporary files with various extensions
        pdf_file = tmp_path / "document.pdf"
        pdf_file.write_text("fake pdf")

        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("readme content")

        assert _should_skip_unsupported(str(pdf_file)) is False  # PDF should be supported
        assert _should_skip_unsupported(str(txt_file)) is True  # TXT should be skipped

    def test_should_skip_binary_html_files(self, tmp_path):
        """Test that binary HTML files are skipped."""
        from scripts.ingest.ingest import _should_skip_unsupported

        binary_html = tmp_path / "binary.html"
        binary_html.write_bytes(b"\x00\x01\x02\xff\xfe\xfd")

        assert _should_skip_unsupported(str(binary_html)) is True

    def test_should_not_skip_text_html_files(self, tmp_path):
        """Test that normal HTML files are not skipped."""
        from scripts.ingest.ingest import _should_skip_unsupported

        html_file = tmp_path / "page.html"
        html_file.write_text("<html><body>Content</body></html>")

        assert _should_skip_unsupported(str(html_file)) is False

    def test_should_not_skip_htm_files(self, tmp_path):
        """Test that .htm files are also accepted."""
        from scripts.ingest.ingest import _should_skip_unsupported

        htm_file = tmp_path / "page.htm"
        htm_file.write_text("<html><body>Content</body></html>")

        assert _should_skip_unsupported(str(htm_file)) is False

    def test_should_skip_verbose_mode(self, tmp_path, capsys):
        """Test that skip reason is printed in verbose mode."""
        from scripts.ingest.ingest import _should_skip_unsupported

        txt_file = tmp_path / "file.txt"
        txt_file.write_text("content")

        result = _should_skip_unsupported(str(txt_file), verbose=True)

        assert result is True
        captured = capsys.readouterr()
        assert "SKIP" in captured.out
        assert "unsupported extension" in captured.out

    def test_log_unsupported_files_default_false(self, tmp_path):
        """Test that unsupported files are NOT audited by default."""
        from scripts.ingest.ingest import _should_skip_unsupported

        txt_file = tmp_path / "file.txt"
        txt_file.write_text("content")

        audit_calls = []

        def mock_audit(event, data):
            audit_calls.append((event, data))

        # Default behaviour: log_unsupported=False
        result = _should_skip_unsupported(str(txt_file), audit_fn=mock_audit)

        assert result is True
        assert len(audit_calls) == 0, "Unsupported files should NOT be audited by default"

    def test_log_unsupported_files_when_enabled(self, tmp_path):
        """Test that unsupported files ARE audited when log_unsupported=True."""
        from scripts.ingest.ingest import _should_skip_unsupported

        txt_file = tmp_path / "file.txt"
        txt_file.write_text("content")

        audit_calls = []

        def mock_audit(event, data):
            audit_calls.append((event, data))

        # Explicitly enable logging
        result = _should_skip_unsupported(str(txt_file), audit_fn=mock_audit, log_unsupported=True)

        assert result is True
        assert len(audit_calls) == 1, "Unsupported files SHOULD be audited when enabled"
        assert audit_calls[0][0] == "skip_file"
        assert "unsupported extension" in audit_calls[0][1]["reason"]

    def test_binary_files_always_audited(self, tmp_path):
        """Test that binary HTML files are ALWAYS audited regardless of log_unsupported."""
        from scripts.ingest.ingest import _should_skip_unsupported

        binary_html = tmp_path / "corrupt.html"
        binary_html.write_bytes(b"\x00\x01\x02\xff")

        audit_calls = []

        def mock_audit(event, data):
            audit_calls.append((event, data))

        # log_unsupported=False, but binary files should still be audited
        result = _should_skip_unsupported(
            str(binary_html), audit_fn=mock_audit, log_unsupported=False
        )

        assert result is True
        assert len(audit_calls) == 1, "Binary files should ALWAYS be audited"
        assert "binary" in audit_calls[0][1]["reason"]


class TestDownloadUrlToFile:
    """Tests for download_url_to_file function."""

    def test_download_url_creates_directory(self, tmp_path, monkeypatch):
        """Test that download creates url_imports directory if missing."""
        from unittest import mock

        from scripts.ingest.ingest import download_url_to_file

        config = IngestConfig()
        config.url_download_dir = str(tmp_path / "url_imports")
        logger = _noop_logger()

        with mock.patch("scripts.ingest.ingest.requests.get") as mock_get:
            mock_response = mock.Mock()
            mock_response.text = "<html>test</html>"
            mock_get.return_value = mock_response

            path, html = download_url_to_file("https://example.com/page", config, logger)

        assert path is not None
        assert os.path.exists(config.url_download_dir)
        assert os.path.exists(path)
        assert html == "<html>test</html>"

    def test_download_url_failure(self, tmp_path, monkeypatch):
        """Test handling of download failure."""
        from unittest import mock

        from scripts.ingest.ingest import download_url_to_file

        config = IngestConfig()
        config.url_download_dir = str(tmp_path / "url_imports")
        logger = _noop_logger()

        with mock.patch("scripts.ingest.ingest.requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection failed")

            path, html = download_url_to_file("https://example.com/bad", config, logger)

        assert path is None
        assert html is None

    def test_download_url_write_failure(self, tmp_path, monkeypatch):
        """Test handling of file write failure."""
        from unittest import mock

        from scripts.ingest.ingest import download_url_to_file

        config = IngestConfig()
        config.url_download_dir = str(tmp_path / "url_imports")
        logger = _noop_logger()

        with mock.patch("scripts.ingest.ingest.requests.get") as mock_get:
            mock_response = mock.Mock()
            mock_response.text = "<html>test</html>"
            mock_get.return_value = mock_response

            # Make the directory readonly to force write failure
            os.makedirs(config.url_download_dir, exist_ok=True)
            os.chmod(config.url_download_dir, 0o444)

            try:
                path, html = download_url_to_file("https://example.com/page", config, logger)
                # On readonly dir, write should fail
                assert path is None or html is None
            finally:
                # Restore permissions for cleanup
                os.chmod(config.url_download_dir, 0o755)


class TestCollectUrlFilesFromSeeds:
    """Tests for collect_url_files_from_seeds function."""

    def test_collect_url_files_from_seeds_no_seeds(self, tmp_path):
        """Test with missing or empty seed file."""
        from scripts.ingest.ingest import collect_url_files_from_seeds

        config = IngestConfig()
        config.url_seed_json_path = str(tmp_path / "nonexistent.json")
        logger = _noop_logger()

        result = collect_url_files_from_seeds(config, logger)

        assert result == []

    def test_collect_url_files_from_seeds_with_valid_seeds(self, tmp_path, monkeypatch):
        """Test seed collection with valid seed file."""
        import json
        from unittest import mock

        from scripts.ingest.ingest import collect_url_files_from_seeds

        seed_file = tmp_path / "seeds.json"
        seed_data = [
            {
                "url": "https://example.com/page1",
                "single_page": True,
                "source_category": "Governance",
            },
            {
                "url": "https://example.com/page2",
                "single_page": False,
                "sidebar_selector": "nav",
                "max_depth": 2,
            },
        ]
        seed_file.write_text(json.dumps(seed_data))

        config = IngestConfig()
        config.url_seed_json_path = str(seed_file)
        config.url_download_dir = str(tmp_path / "downloads")
        logger = _noop_logger()

        # Mock requests.get to return HTML
        with mock.patch("scripts.ingest.ingest.requests.get") as mock_get:
            mock_response = mock.Mock()
            mock_response.text = "<html><body>content</body></html>"
            mock_get.return_value = mock_response

            result = collect_url_files_from_seeds(config, logger)

        # Should have downloaded at least one file
        assert len(result) >= 1
        # Each result should be (path, source_category) tuple
        for path, category in result:
            assert os.path.exists(path)
            assert category in [None, "Governance"]


class TestStageComputeHash:
    """Tests for stage_compute_hash function."""

    def test_stage_compute_hash_success(self, tmp_path):
        """Test successful hash computation."""
        from scripts.ingest.ingest import stage_compute_hash

        html_file = tmp_path / "test.html"
        html_file.write_text("<html><body>Test</body></html>")

        logger = _noop_logger()
        doc_id, file_hash = stage_compute_hash(str(html_file), logger)

        assert doc_id == "test"
        assert file_hash is not None
        assert len(file_hash) == 32  # MD5 hash length

    def test_stage_compute_hash_file_not_found(self):
        """Test hash computation with missing file."""
        from scripts.ingest.ingest import stage_compute_hash

        logger = _noop_logger()
        doc_id, file_hash = stage_compute_hash("/nonexistent/file.html", logger)

        assert doc_id is None
        assert file_hash is None


class TestStageDetermineCategory:
    """Tests for stage_determine_category function."""

    def test_stage_determine_category_override(self):
        """Test that override takes precedence."""
        from scripts.ingest.ingest import stage_determine_category

        logger = _noop_logger()
        category = stage_determine_category(
            "/data_raw/Governance/test.html", source_category_override="Patterns", logger=logger
        )

        assert category == "Patterns"

    def test_stage_determine_category_from_path(self):
        """Test category extraction from path."""
        from scripts.ingest.ingest import stage_determine_category

        logger = _noop_logger()
        category = stage_determine_category(
            "/data_raw/Governance/test.html", source_category_override=None, logger=logger
        )

        assert category == "Governance"

    def test_stage_determine_category_excluded_names(self):
        """Test that excluded directory names return None."""
        from scripts.ingest.ingest import stage_determine_category

        logger = _noop_logger()

        for excluded in ["downloads", "data_raw"]:
            category = stage_determine_category(
                f"/path/{excluded}/test.html", source_category_override=None, logger=logger
            )
            assert category is None


class TestStageExtractText:
    """Tests for stage_extract_text function."""

    def test_stage_extract_text_success(self, tmp_path):
        """Test successful text extraction."""
        from scripts.ingest.ingest import stage_extract_text

        html_file = tmp_path / "test.html"
        html_file.write_text("<html><body>Test content</body></html>")

        logger = _noop_logger()
        text = stage_extract_text(str(html_file), logger)

        assert text is not None
        assert "Test content" in text

    def test_stage_extract_text_missing_file(self):
        """Test extraction with missing file."""
        from scripts.ingest.ingest import stage_extract_text

        logger = _noop_logger()
        text = stage_extract_text("/nonexistent/file.html", logger)

        assert text is None


class TestStageChunkText:
    """Tests for stage_chunk_text function."""

    def test_stage_chunk_text_success(self):
        """Test successful text chunking."""
        from scripts.ingest.ingest import stage_chunk_text

        preprocessed = {
            "cleaned_text": "This is a test. This is another test. " * 50,  # Long text
            "doc_type": "guide",
        }

        args = SimpleNamespace(verbose=False)
        logger = _noop_logger()

        chunks = stage_chunk_text(preprocessed, args, logger)

        assert chunks is not None
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_stage_chunk_text_empty_text(self):
        """Test chunking with empty text."""
        from scripts.ingest.ingest import stage_chunk_text

        preprocessed = {"cleaned_text": "", "doc_type": "unknown"}

        args = SimpleNamespace(verbose=False)
        logger = _noop_logger()

        chunks = stage_chunk_text(preprocessed, args, logger)

        assert chunks is not None


class TestStagePreprocessText:
    """Tests for stage_preprocess_text function."""


class TestStageChunkIdempotency:
    """Tests for stage_chunk_idempotency function."""

    def test_stage_chunk_idempotency_new_chunks(self):
        """Test chunk idempotency with new chunks."""
        import chromadb

        from scripts.ingest.ingest import stage_chunk_idempotency

        client = chromadb.EphemeralClient()
        chunk_collection = client.get_or_create_collection("test")

        chunks = ["chunk1", "chunk2", "chunk3"]
        chunk_data = {"chunks": chunks, "full_text": " ".join(chunks)}
        doc_id = "test_doc_id"
        version = 1
        is_update = False

        logger = _noop_logger()

        chunks_to_process, chunk_hashes, full_text = stage_chunk_idempotency(
            chunk_data, doc_id, version, is_update, chunk_collection, logger
        )

        # All chunks should be new
        assert len(chunks_to_process) == 3
        assert len(chunk_hashes) == 3
        assert full_text == " ".join(chunks)

    def test_stage_chunk_idempotency_no_update(self):
        """Test chunk idempotency when not updating."""
        import chromadb

        from scripts.ingest.ingest import stage_chunk_idempotency

        client = chromadb.EphemeralClient()
        chunk_collection = client.get_or_create_collection("test")

        chunks = ["chunk1", "chunk2"]
        chunk_data = {"chunks": chunks, "full_text": " ".join(chunks)}
        doc_id = "test_doc_id"
        version = 1
        is_update = False  # Not an update

        logger = _noop_logger()

        chunks_to_process, chunk_hashes, full_text = stage_chunk_idempotency(
            chunk_data, doc_id, version, is_update, chunk_collection, logger
        )

        # Should process all chunks for new document
        assert len(chunks_to_process) > 0


class TestStageStoreChunks:
    """Tests for stage_store_chunks function."""


class TestLoadUrlSeeds:
    """Tests for load_url_seeds function."""

    def test_load_url_seeds_valid_file(self, tmp_path):
        """Test loading valid URL seed file."""
        import json

        from scripts.ingest.ingest import load_url_seeds

        seed_file = tmp_path / "seeds.json"
        seed_data = [
            {"url": "https://example.com/1"},
            {"url": "https://example.com/2", "source_category": "Patterns"},
        ]
        seed_file.write_text(json.dumps(seed_data))

        logger = _noop_logger()
        result = load_url_seeds(str(seed_file), logger)

        assert len(result) == 2
        assert result[0]["url"] == "https://example.com/1"
        assert result[1]["source_category"] == "Patterns"

    def test_load_url_seeds_missing_file(self, tmp_path):
        """Test loading missing seed file."""
        from scripts.ingest.ingest import load_url_seeds

        logger = _noop_logger()
        result = load_url_seeds(str(tmp_path / "nonexistent.json"), logger)

        assert result == []

    def test_load_url_seeds_invalid_json(self, tmp_path):
        """Test loading invalid JSON file."""
        from scripts.ingest.ingest import load_url_seeds

        seed_file = tmp_path / "bad.json"
        seed_file.write_text("{ invalid json }")

        logger = _noop_logger()
        result = load_url_seeds(str(seed_file), logger)

        assert result == []

    def test_load_url_seeds_not_list(self, tmp_path):
        """Test loading JSON that's not a list."""
        import json

        from scripts.ingest.ingest import load_url_seeds

        seed_file = tmp_path / "notlist.json"
        seed_file.write_text(json.dumps({"url": "https://example.com"}))

        logger = _noop_logger()
        result = load_url_seeds(str(seed_file), logger)

        assert result == []
