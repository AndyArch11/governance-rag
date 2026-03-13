"""Unit tests for database clearing utilities.

Tests all clear_* functions with temporary databases to avoid impacting
production data. Uses mock config objects with configurable paths.
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.utils.clear_databases import (
    ResetContext,
    clear_academic_pdf_cache,
    clear_bm25_cache,
    clear_cache_database,
    clear_chromadb,
    clear_citation_graph_database,
    clear_graph_database,
    clear_legacy_artifacts,
    clear_reference_cache,
    clear_terminology_database,
    commit_reset,
    prepare_reset_workspace,
    rollback_reset,
    show_current_state,
)


class MockConfig:
    """Mock config object for testing."""

    def __init__(self, rag_data_path: str):
        """Initialise mock config with test paths.

        Args:
            rag_data_path: Path to temporary rag_data directory
        """
        self.rag_data_path = rag_data_path
        self.cache_path = str(Path(rag_data_path) / "cache.db")
        self.output_sqlite = str(
            Path(rag_data_path) / "consistency_graphs" / "consistency_graph.sqlite"
        )


class TestClearChromadb:
    """Test clearing ChromaDB functionality."""

    def test_clear_chromadb_not_exists(self, capsys) -> None:
        """Test clearing when ChromaDB doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            clear_chromadb(config)

            captured = capsys.readouterr()
            assert "already clean" in captured.out

    def test_clear_chromadb_exists(self, capsys) -> None:
        """Test clearing existing ChromaDB directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            chroma_path = Path(tmpdir) / "chromadb"
            chroma_path.mkdir()
            sqlite_path = Path(tmpdir) / "chromadb.db"

            # Create some files
            (chroma_path / "test.db").write_text("test data")
            sqlite_path.write_text("sqlite data")
            assert chroma_path.exists()
            assert sqlite_path.exists()

            clear_chromadb(config)

            assert not chroma_path.exists()
            assert not sqlite_path.exists()
            captured = capsys.readouterr()
            assert "Cleared ChromaDB" in captured.out

    def test_clear_chromadb_dry_run(self, capsys) -> None:
        """Test dry run doesn't delete ChromaDB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            chroma_path = Path(tmpdir) / "chromadb"
            chroma_path.mkdir()
            (chroma_path / "test.db").write_text("test data")
            sqlite_path = Path(tmpdir) / "chromadb.db"
            sqlite_path.write_text("sqlite data")

            clear_chromadb(config, dry_run=True)

            assert chroma_path.exists()
            assert sqlite_path.exists()
            captured = capsys.readouterr()
            assert "DRY RUN" in captured.out


class TestClearGraphDatabase:
    """Test clearing graph database functionality."""

    def test_clear_graph_database_not_exists(self, capsys) -> None:
        """Test clearing when graph database doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            clear_graph_database(config)

            captured = capsys.readouterr()
            assert "already clean" in captured.out

    def test_clear_graph_database_clears_tables(self, capsys) -> None:
        """Test clearing graph database removes data but keeps schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            graph_dir = Path(tmpdir) / "consistency_graphs"
            graph_dir.mkdir(parents=True)
            db_path = graph_dir / "consistency_graph.sqlite"

            # Create database with tables and data
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, name TEXT)")
            cursor.execute(
                "CREATE TABLE edges (id INTEGER PRIMARY KEY, source INTEGER, target INTEGER)"
            )
            cursor.execute("INSERT INTO nodes (name) VALUES ('node1')")
            cursor.execute("INSERT INTO edges (source, target) VALUES (1, 2)")
            conn.commit()

            # Verify data exists
            cursor.execute("SELECT COUNT(*) FROM nodes")
            assert cursor.fetchone()[0] == 1
            conn.close()

            # Clear database
            clear_graph_database(config)

            # Verify tables are empty but still exist
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nodes")
            assert cursor.fetchone()[0] == 0
            cursor.execute("SELECT COUNT(*) FROM edges")
            assert cursor.fetchone()[0] == 0
            conn.close()

            captured = capsys.readouterr()
            assert "Cleared graph database" in captured.out

    def test_clear_graph_database_dry_run(self) -> None:
        """Test dry run doesn't modify database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            graph_dir = Path(tmpdir) / "consistency_graphs"
            graph_dir.mkdir(parents=True)
            db_path = graph_dir / "consistency_graph.sqlite"

            # Create database with data
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, name TEXT)")
            cursor.execute("INSERT INTO nodes (name) VALUES ('node1')")
            conn.commit()
            conn.close()

            # Dry run should not modify
            clear_graph_database(config, dry_run=True)

            # Verify data still exists
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nodes")
            assert cursor.fetchone()[0] == 1
            conn.close()


class TestClearCacheDatabase:
    """Test clearing cache database functionality."""

    def test_clear_cache_database_not_exists(self, capsys) -> None:
        """Test clearing when cache database doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            clear_cache_database(config)

            captured = capsys.readouterr()
            assert "already clean" in captured.out

    def test_clear_cache_database_clears_tables(self, capsys) -> None:
        """Test clearing cache database removes ingest cache data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            cache_path = Path(config.cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Create cache database with tables
            conn = sqlite3.connect(str(cache_path))
            cursor = conn.cursor()
            cursor.execute(
                "CREATE TABLE embeddings_cache (id INTEGER PRIMARY KEY, text TEXT, embedding BLOB)"
            )
            cursor.execute(
                "CREATE TABLE llm_cache (id INTEGER PRIMARY KEY, prompt TEXT, response TEXT)"
            )
            cursor.execute("CREATE TABLE bm25_cache (id INTEGER PRIMARY KEY, term TEXT)")
            cursor.execute("CREATE TABLE bm25_corpus_stats (id INTEGER PRIMARY KEY, stat TEXT)")
            cursor.execute(
                "CREATE TABLE bm25_doc_metadata (id INTEGER PRIMARY KEY, doc_id INTEGER)"
            )
            cursor.execute("CREATE TABLE bm25_index (id INTEGER PRIMARY KEY, term_id INTEGER)")
            cursor.execute(
                "CREATE TABLE word_frequency (word TEXT PRIMARY KEY, frequency INTEGER, doc_count INTEGER)"
            )
            cursor.execute("CREATE TABLE graph_cache (id INTEGER PRIMARY KEY, data TEXT)")
            cursor.execute("CREATE TABLE graph_settings_map (id INTEGER PRIMARY KEY, setting TEXT)")

            # Insert data into all tables
            cursor.execute("INSERT INTO embeddings_cache (text) VALUES ('text1')")
            cursor.execute("INSERT INTO llm_cache (prompt, response) VALUES ('q', 'a')")
            cursor.execute(
                "INSERT INTO word_frequency (word, frequency, doc_count) VALUES ('test', 5, 2)"
            )
            cursor.execute("INSERT INTO graph_cache (data) VALUES ('cached')")
            conn.commit()
            conn.close()

            # Clear cache database
            clear_cache_database(config)

            # Verify ingest caches are empty
            conn = sqlite3.connect(str(cache_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM embeddings_cache")
            assert cursor.fetchone()[0] == 0
            cursor.execute("SELECT COUNT(*) FROM llm_cache")
            assert cursor.fetchone()[0] == 0
            cursor.execute("SELECT COUNT(*) FROM word_frequency")
            assert cursor.fetchone()[0] == 0

            # Verify graph cache is preserved
            cursor.execute("SELECT COUNT(*) FROM graph_cache")
            assert cursor.fetchone()[0] == 1
            conn.close()

            captured = capsys.readouterr()
            assert "Ingest caches cleared" in captured.out
            assert "graph caches preserved" in captured.out

    def test_clear_cache_database_dry_run(self) -> None:
        """Test dry run doesn't modify cache database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            cache_path = Path(config.cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Create cache database with data
            conn = sqlite3.connect(str(cache_path))
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE embeddings_cache (id INTEGER PRIMARY KEY, text TEXT)")
            cursor.execute("INSERT INTO embeddings_cache (text) VALUES ('text1')")
            conn.commit()
            conn.close()

            # Dry run should not modify
            clear_cache_database(config, dry_run=True)

            # Verify data still exists
            conn = sqlite3.connect(str(cache_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM embeddings_cache")
            assert cursor.fetchone()[0] == 1
            conn.close()


class TestClearBM25Cache:
    """Test clearing BM25 index functionality."""

    def test_clear_bm25_cache_not_exists(self, capsys) -> None:
        """Test clearing when BM25 index doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            clear_bm25_cache(config)

            captured = capsys.readouterr()
            assert "already clean" in captured.out

    def test_clear_bm25_cache_exists(self, capsys) -> None:
        """Test clearing existing BM25 index directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            bm25_path = Path(tmpdir) / "bm25_index"
            bm25_path.mkdir()
            (bm25_path / "index.pkl").write_text("index data")

            clear_bm25_cache(config)

            assert not bm25_path.exists()
            captured = capsys.readouterr()
            assert "Cleared BM25 index" in captured.out

    def test_clear_bm25_cache_dry_run(self, capsys) -> None:
        """Test dry run doesn't delete BM25 index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            bm25_path = Path(tmpdir) / "bm25_index"
            bm25_path.mkdir()
            (bm25_path / "index.pkl").write_text("index data")

            clear_bm25_cache(config, dry_run=True)

            assert bm25_path.exists()
            captured = capsys.readouterr()
            assert "DRY RUN" in captured.out


class TestClearReferenceCache:
    """Test clearing reference cache functionality."""

    def test_clear_reference_cache_not_exists(self, capsys) -> None:
        """Test clearing when reference cache doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            clear_reference_cache(config)

            captured = capsys.readouterr()
            assert "already clean" in captured.out

    def test_clear_reference_cache_exists(self, capsys) -> None:
        """Test clearing existing reference cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            cache_path = Path(tmpdir) / "academic_references.db"
            cache_path.write_text("reference cache data")

            assert cache_path.exists()
            clear_reference_cache(config)

            assert not cache_path.exists()
            captured = capsys.readouterr()
            assert "Cleared reference cache" in captured.out

    def test_clear_reference_cache_dry_run(self, capsys) -> None:
        """Test dry run doesn't delete reference cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            cache_path = Path(tmpdir) / "academic_references.db"
            cache_path.write_text("reference cache data")

            clear_reference_cache(config, dry_run=True)
            assert cache_path.exists()
            captured = capsys.readouterr()
            assert "DRY RUN" in captured.out


class TestClearTerminologyDatabase:
    """Test clearing terminology database functionality."""

    def test_clear_terminology_database_not_exists(self, capsys) -> None:
        """Test clearing when terminology database doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            clear_terminology_database(config)

            captured = capsys.readouterr()
            assert "already clean" in captured.out

    def test_clear_terminology_database_exists(self, capsys) -> None:
        """Test clearing existing terminology database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            term_path = Path(tmpdir) / "academic_terminology.db"
            term_path.write_text("terminology data")

            assert term_path.exists()
            clear_terminology_database(config)

            assert not term_path.exists()
            captured = capsys.readouterr()
            assert "Cleared terminology database" in captured.out

    def test_clear_terminology_database_dry_run(self, capsys) -> None:
        """Test dry run doesn't delete terminology database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            term_path = Path(tmpdir) / "academic_terminology.db"
            term_path.write_text("terminology data")

            clear_terminology_database(config, dry_run=True)

            assert term_path.exists()
            captured = capsys.readouterr()
            assert "DRY RUN" in captured.out


class TestClearCitationGraphDatabase:
    """Test clearing citation graph database functionality."""

    def test_clear_citation_graph_database_not_exists(self, capsys) -> None:
        """Test clearing when citation graph database doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            clear_citation_graph_database(config)

            captured = capsys.readouterr()
            assert "already clean" in captured.out

    def test_clear_citation_graph_database_exists(self, capsys) -> None:
        """Test clearing existing citation graph database and JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            db_path = Path(tmpdir) / "academic_citation_graph.db"
            json_path = Path(tmpdir) / "academic_citation_graph.json"
            temp_db_path = Path(tmpdir) / "academic_citation_graph_temp.db"
            db_path.write_text("graph data")
            json_path.write_text("graph json")
            temp_db_path.write_text("temp data")

            clear_citation_graph_database(config)

            assert not db_path.exists()
            assert not json_path.exists()
            assert not temp_db_path.exists()
            captured = capsys.readouterr()
            assert "Cleared citation graph" in captured.out

    def test_clear_citation_graph_database_dry_run(self, capsys) -> None:
        """Test dry run doesn't delete citation graph database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            db_path = Path(tmpdir) / "academic_citation_graph.db"
            db_path.write_text("graph data")

            clear_citation_graph_database(config, dry_run=True)

            assert db_path.exists()
            captured = capsys.readouterr()
            assert "DRY RUN" in captured.out


class TestClearAcademicPdfCache:
    """Test clearing academic PDF cache functionality."""

    def test_clear_academic_pdf_cache_not_exists(self, capsys) -> None:
        """Test clearing when academic PDF cache doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            clear_academic_pdf_cache(config)

            captured = capsys.readouterr()
            assert "already clean" in captured.out

    def test_clear_academic_pdf_cache_exists(self, capsys) -> None:
        """Test clearing existing academic PDF cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            cache_dir = Path(tmpdir) / "academic_pdfs"
            cache_dir.mkdir()
            (cache_dir / "example.pdf").write_text("pdf data")

            clear_academic_pdf_cache(config)

            assert not cache_dir.exists()
            captured = capsys.readouterr()
            assert "Cleared academic PDF cache" in captured.out

    def test_clear_academic_pdf_cache_dry_run(self, capsys) -> None:
        """Test dry run doesn't delete academic PDF cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            cache_dir = Path(tmpdir) / "academic_pdfs"
            cache_dir.mkdir()
            (cache_dir / "example.pdf").write_text("pdf data")

            clear_academic_pdf_cache(config, dry_run=True)

            assert cache_dir.exists()
            captured = capsys.readouterr()
            assert "DRY RUN" in captured.out


class TestClearLegacyArtifacts:
    """Test clearing legacy artefacts functionality."""

    def test_clear_legacy_artifacts_not_exists(self, capsys) -> None:
        """Test clearing when no legacy artefacts exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            clear_legacy_artifacts(config)

            captured = capsys.readouterr()
            assert "No legacy artefacts" in captured.out

    def test_clear_legacy_artifacts_exists(self, capsys) -> None:
        """Test clearing existing legacy artefacts and hnsw index dirs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)

            legacy_dir = Path(tmpdir) / "chroma_db"
            legacy_dir.mkdir()
            (legacy_dir / "chroma.sqlite3").write_text("db data")

            legacy_file = Path(tmpdir) / "graph.db"
            legacy_file.write_text("graph data")

            hnsw_dir = Path(tmpdir) / "legacy_index"
            hnsw_dir.mkdir()
            for filename in [
                "data_level0.bin",
                "header.bin",
                "index_metadata.pickle",
                "length.bin",
                "link_lists.bin",
            ]:
                (hnsw_dir / filename).write_text("data")

            clear_legacy_artifacts(config)

            assert not legacy_dir.exists()
            assert not legacy_file.exists()
            assert not hnsw_dir.exists()
            captured = capsys.readouterr()
            assert "Cleared legacy artefacts" in captured.out

    def test_clear_legacy_artifacts_dry_run(self, capsys) -> None:
        """Test dry run doesn't delete legacy artefacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)

            legacy_dir = Path(tmpdir) / "chroma_db"
            legacy_dir.mkdir()
            (legacy_dir / "chroma.sqlite3").write_text("db data")

            clear_legacy_artifacts(config, dry_run=True)

            assert legacy_dir.exists()
            captured = capsys.readouterr()
            assert "DRY RUN" in captured.out


class TestShowCurrentState:
    """Test showing current database state."""

    def test_show_empty_state(self, capsys) -> None:
        """Test showing state with no databases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            show_current_state(config)

            captured = capsys.readouterr()
            assert "Current Database State" in captured.out
            assert "Not found" in captured.out

    def test_show_state_with_chromadb(self, capsys) -> None:
        """Test showing state with ChromaDB present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            chroma_path = Path(tmpdir) / "chromadb"
            chroma_path.mkdir()

            # Try to show state (may fail if chromadb not installed, that's ok)
            try:
                show_current_state(config)
                captured = capsys.readouterr()
                assert "ChromaDB" in captured.out
            except ImportError:
                # chromadb not installed, skip this test
                pass

    def test_show_state_with_cache(self, capsys) -> None:
        """Test showing state with cache database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)
            cache_path = Path(config.cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text("cache data")

            show_current_state(config)

            captured = capsys.readouterr()
            assert "Cache Database" in captured.out


class TestIntegration:
    """Integration tests for clear_databases functions."""

    def test_clear_all_with_config(self, capsys) -> None:
        """Test clearing all databases with consistent config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)

            # Set up all databases
            chroma_path = Path(tmpdir) / "chromadb"
            chroma_path.mkdir()
            (chroma_path / "data.db").write_text("chroma data")

            bm25_path = Path(tmpdir) / "bm25_index"
            bm25_path.mkdir()
            (bm25_path / "index.pkl").write_text("bm25 data")

            cache_path = Path(config.cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text("cache data")

            ref_cache = Path(tmpdir) / "academic_references.db"
            ref_cache.write_text("reference data")

            term_db = Path(tmpdir) / "academic_terminology.db"
            term_db.write_text("terminology data")

            citation_db = Path(tmpdir) / "academic_citation_graph.db"
            citation_db.write_text("citation data")

            citation_json = Path(tmpdir) / "academic_citation_graph.json"
            citation_json.write_text("citation json")

            pdf_cache = Path(tmpdir) / "academic_pdfs"
            pdf_cache.mkdir()
            (pdf_cache / "example.pdf").write_text("pdf data")

            legacy_dir = Path(tmpdir) / "chroma_db"
            legacy_dir.mkdir()
            (legacy_dir / "chroma.sqlite3").write_text("legacy data")

            # Clear all
            clear_terminology_database(config)
            clear_citation_graph_database(config)
            clear_academic_pdf_cache(config)
            clear_legacy_artifacts(config)
            clear_chromadb(config)
            clear_bm25_cache(config)
            clear_cache_database(config)
            clear_reference_cache(config)

            # Verify all cleared
            assert not chroma_path.exists()
            assert not bm25_path.exists()
            assert not ref_cache.exists()
            assert not term_db.exists()
            assert not citation_db.exists()
            assert not citation_json.exists()
            assert not pdf_cache.exists()
            assert not legacy_dir.exists()

    def test_config_consistency(self) -> None:
        """Test that config paths are used consistently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MockConfig(tmpdir)

            # All clear functions should use paths from config
            assert Path(config.rag_data_path).exists()
            assert isinstance(config.cache_path, str)
            assert isinstance(config.output_sqlite, str)

            # Paths should be within rag_data
            assert tmpdir in config.rag_data_path
            assert tmpdir in config.cache_path
            assert tmpdir in config.output_sqlite




# =============================================================================
# ResetContext / prepare_reset_workspace / commit_reset / rollback_reset
# =============================================================================


def _make_mock_ingest_config(tmpdir: str):
    """Return a minimal mock IngestConfig-like object for reset tests."""
    config = MockConfig(tmpdir)
    return config


def _make_reset_context(
    tmpdir: str,
    chroma_live: Path,
    chroma_temp: Path,
    using_sqlite: bool,
) -> ResetContext:
    """Build a ResetContext with BM25 cache paths for tests."""
    bm25_stage_dir = Path(tmpdir) / "bm25_reset_stage"
    return ResetContext(
        chroma_path_live=chroma_live,
        chroma_path_temp=chroma_temp,
        bm25_cache_live_db=Path(tmpdir) / "cache.db",
        bm25_cache_temp_db=bm25_stage_dir / "cache.db",
        bm25_stage_dir=bm25_stage_dir,
        using_sqlite=using_sqlite,
    )


class TestResetContext:
    """Tests for the ResetContext dataclass."""

    def test_fields_are_stored(self) -> None:
        """ResetContext stores live and temp paths plus backend flag."""
        live = Path("/tmp/chroma")
        temp = Path("/tmp/chroma.new")
        ctx = ResetContext(
            chroma_path_live=live,
            chroma_path_temp=temp,
            bm25_cache_live_db=Path("/tmp/cache.db"),
            bm25_cache_temp_db=Path("/tmp/bm25_reset_stage/cache.db"),
            bm25_stage_dir=Path("/tmp/bm25_reset_stage"),
            using_sqlite=False,
        )
        assert ctx.chroma_path_live == live
        assert ctx.chroma_path_temp == temp
        assert ctx.bm25_cache_live_db == Path("/tmp/cache.db")
        assert ctx.bm25_stage_dir == Path("/tmp/bm25_reset_stage")
        assert ctx.using_sqlite is False

    def test_sqlite_variant(self) -> None:
        """ResetContext using_sqlite=True stores the flag correctly."""
        ctx = ResetContext(
            chroma_path_live=Path("/tmp/c.db"),
            chroma_path_temp=Path("/tmp/c.db.new"),
            bm25_cache_live_db=Path("/tmp/cache.db"),
            bm25_cache_temp_db=Path("/tmp/bm25_reset_stage/cache.db"),
            bm25_stage_dir=Path("/tmp/bm25_reset_stage"),
            using_sqlite=True,
        )
        assert ctx.using_sqlite is True


class TestPrepareResetWorkspace:
    """Tests for prepare_reset_workspace."""

    def test_returns_reset_context_directory_backend(self) -> None:
        """Returns ResetContext with .new suffix for directory backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_mock_ingest_config(tmpdir)
            ctx = prepare_reset_workspace(config, using_sqlite=False)
            assert isinstance(ctx, ResetContext)
            assert ctx.chroma_path_live == Path(tmpdir) / "chromadb"
            assert ctx.chroma_path_temp == Path(tmpdir) / "chromadb.new"
            assert ctx.using_sqlite is False

    def test_returns_reset_context_sqlite_backend(self) -> None:
        """Returns ResetContext with .db.new suffix for SQLite backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_mock_ingest_config(tmpdir)
            ctx = prepare_reset_workspace(config, using_sqlite=True)
            assert ctx.chroma_path_live == Path(tmpdir) / "chromadb.db"
            assert ctx.chroma_path_temp == Path(tmpdir) / "chromadb.db.new"
            assert ctx.using_sqlite is True

    def test_live_chromadb_is_preserved(self) -> None:
        """The live ChromaDB directory/file is untouched by prepare_reset_workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_mock_ingest_config(tmpdir)
            chroma_live = Path(tmpdir) / "chromadb"
            chroma_live.mkdir()
            (chroma_live / "sentinel.db").write_text("live data")

            prepare_reset_workspace(config, using_sqlite=False)

            assert chroma_live.exists(), "Live ChromaDB must be preserved"
            assert (chroma_live / "sentinel.db").read_text() == "live data"

    def test_stale_temp_is_removed(self, capsys) -> None:
        """A stale temp path left by a previous interrupted reset is cleaned up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_mock_ingest_config(tmpdir)
            stale_temp = Path(tmpdir) / "chromadb.new"
            stale_temp.mkdir()
            (stale_temp / "stale.db").write_text("stale")

            ctx = prepare_reset_workspace(config, using_sqlite=False)

            assert not stale_temp.exists(), "Stale temp must be removed"
            assert ctx.chroma_path_temp == stale_temp

    def test_stale_backup_is_removed(self) -> None:
        """A stale .old backup from a previous interrupted commit is cleaned up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_mock_ingest_config(tmpdir)
            stale_backup = Path(tmpdir) / "chromadb.old"
            stale_backup.mkdir()
            (stale_backup / "old.db").write_text("old")

            prepare_reset_workspace(config, using_sqlite=False)

            assert not stale_backup.exists(), "Stale backup must be removed"

    def test_live_bm25_is_preserved_and_stage_is_created(self) -> None:
        """Live BM25 remains available while an isolated stage DB is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_mock_ingest_config(tmpdir)
            live_cache = Path(tmpdir) / "cache.db"
            with sqlite3.connect(str(live_cache)) as conn:
                conn.execute(
                    "CREATE TABLE bm25_doc_metadata (doc_id TEXT PRIMARY KEY, doc_length INTEGER NOT NULL, original_text TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                )
                conn.execute(
                    "INSERT INTO bm25_doc_metadata (doc_id, doc_length, original_text) VALUES ('live-doc', 10, 'text')"
                )
                conn.commit()

            ctx = prepare_reset_workspace(config, using_sqlite=False)

            with sqlite3.connect(str(live_cache)) as conn:
                rows = conn.execute("SELECT COUNT(*) FROM bm25_doc_metadata").fetchone()[0]
            assert rows == 1, "Live BM25 must stay available during ingestion"
            assert ctx.bm25_stage_dir.exists()
            assert ctx.bm25_cache_temp_db.exists()


class TestCommitReset:
    """Tests for commit_reset."""

    def test_moves_temp_to_live(self) -> None:
        """Temp ChromaDB directory is moved to the live path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chroma_live = Path(tmpdir) / "chromadb"
            chroma_temp = Path(tmpdir) / "chromadb.new"
            chroma_temp.mkdir()
            (chroma_temp / "data.db").write_text("new data")

            bm25_stage = Path(tmpdir) / "bm25_reset_stage"
            bm25_stage.mkdir(parents=True)
            staged_cache = bm25_stage / "cache.db"
            with sqlite3.connect(str(staged_cache)) as conn:
                conn.execute(
                    "CREATE TABLE bm25_index (term TEXT NOT NULL, doc_id TEXT NOT NULL, term_frequency INTEGER NOT NULL, doc_length INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (term, doc_id))"
                )
                conn.execute(
                    "CREATE TABLE bm25_corpus_stats (term TEXT PRIMARY KEY, document_frequency INTEGER NOT NULL, idf REAL NOT NULL, last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                )
                conn.execute(
                    "CREATE TABLE bm25_doc_metadata (doc_id TEXT PRIMARY KEY, doc_length INTEGER NOT NULL, original_text TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                )
                conn.execute(
                    "INSERT INTO bm25_doc_metadata (doc_id, doc_length, original_text) VALUES ('doc-new', 5, 'hello')"
                )
                conn.commit()

            ctx = _make_reset_context(tmpdir, chroma_live, chroma_temp, using_sqlite=False)
            commit_reset(ctx)

            assert chroma_live.exists()
            assert (chroma_live / "data.db").read_text() == "new data"
            assert not chroma_temp.exists()
            assert not bm25_stage.exists()

            with sqlite3.connect(str(Path(tmpdir) / "cache.db")) as conn:
                rows = conn.execute("SELECT COUNT(*) FROM bm25_doc_metadata").fetchone()[0]
            assert rows == 1

    def test_replaces_existing_live(self) -> None:
        """An existing live ChromaDB is replaced by the new temp on commit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chroma_live = Path(tmpdir) / "chromadb"
            chroma_live.mkdir()
            (chroma_live / "old.db").write_text("old data")

            chroma_temp = Path(tmpdir) / "chromadb.new"
            chroma_temp.mkdir()
            (chroma_temp / "new.db").write_text("new data")

            bm25_stage = Path(tmpdir) / "bm25_reset_stage"
            bm25_stage.mkdir(parents=True)
            staged_cache = bm25_stage / "cache.db"
            with sqlite3.connect(str(staged_cache)) as conn:
                conn.execute(
                    "CREATE TABLE bm25_index (term TEXT NOT NULL, doc_id TEXT NOT NULL, term_frequency INTEGER NOT NULL, doc_length INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (term, doc_id))"
                )
                conn.execute(
                    "CREATE TABLE bm25_corpus_stats (term TEXT PRIMARY KEY, document_frequency INTEGER NOT NULL, idf REAL NOT NULL, last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                )
                conn.execute(
                    "CREATE TABLE bm25_doc_metadata (doc_id TEXT PRIMARY KEY, doc_length INTEGER NOT NULL, original_text TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                )
                conn.execute(
                    "INSERT INTO bm25_doc_metadata (doc_id, doc_length, original_text) VALUES ('doc-replaced', 7, 'text')"
                )
                conn.commit()

            ctx = _make_reset_context(tmpdir, chroma_live, chroma_temp, using_sqlite=False)
            commit_reset(ctx)

            assert chroma_live.exists()
            assert not (chroma_live / "old.db").exists()
            assert (chroma_live / "new.db").read_text() == "new data"
            # Backup should be cleaned up
            assert not Path(str(chroma_live) + ".old").exists()

    def test_raises_when_temp_missing(self) -> None:
        """Raises FileNotFoundError if temp ChromaDB was never created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _make_reset_context(
                tmpdir,
                Path(tmpdir) / "chromadb",
                Path(tmpdir) / "chromadb.new",
                using_sqlite=False,
            )
            with pytest.raises(FileNotFoundError, match="Temp ChromaDB not found"):
                commit_reset(ctx)

    def test_restores_live_from_backup_on_move_failure(self) -> None:
        """If the move fails, the backup is restored so live path is never empty."""
        import unittest.mock as mock

        with tempfile.TemporaryDirectory() as tmpdir:
            chroma_live = Path(tmpdir) / "chromadb"
            chroma_live.mkdir()
            (chroma_live / "live.db").write_text("live")

            chroma_temp = Path(tmpdir) / "chromadb.new"
            chroma_temp.mkdir()
            (chroma_temp / "new.db").write_text("new")

            bm25_stage = Path(tmpdir) / "bm25_reset_stage"
            bm25_stage.mkdir(parents=True)
            staged_cache = bm25_stage / "cache.db"
            with sqlite3.connect(str(staged_cache)) as conn:
                conn.execute(
                    "CREATE TABLE bm25_index (term TEXT NOT NULL, doc_id TEXT NOT NULL, term_frequency INTEGER NOT NULL, doc_length INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (term, doc_id))"
                )
                conn.execute(
                    "CREATE TABLE bm25_corpus_stats (term TEXT PRIMARY KEY, document_frequency INTEGER NOT NULL, idf REAL NOT NULL, last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                )
                conn.execute(
                    "CREATE TABLE bm25_doc_metadata (doc_id TEXT PRIMARY KEY, doc_length INTEGER NOT NULL, original_text TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                )
                conn.commit()

            ctx = _make_reset_context(tmpdir, chroma_live, chroma_temp, using_sqlite=False)

            # Patch shutil.move to fail after the backup rename has happened
            def failing_move(src, dst):
                raise OSError("simulated move failure")

            with mock.patch("scripts.utils.clear_databases.shutil.move", side_effect=failing_move):
                with pytest.raises(OSError):
                    commit_reset(ctx)

            # The live path should have been restored from backup
            assert chroma_live.exists(), "Live path must be restored after failed commit"

    def test_sqlite_file_swap(self) -> None:
        """Commit works for the SQLite single-file backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chroma_live = Path(tmpdir) / "chromadb.db"
            chroma_live.write_text("old db")

            chroma_temp = Path(tmpdir) / "chromadb.db.new"
            chroma_temp.write_text("new db")

            bm25_stage = Path(tmpdir) / "bm25_reset_stage"
            bm25_stage.mkdir(parents=True)
            staged_cache = bm25_stage / "cache.db"
            with sqlite3.connect(str(staged_cache)) as conn:
                conn.execute(
                    "CREATE TABLE bm25_index (term TEXT NOT NULL, doc_id TEXT NOT NULL, term_frequency INTEGER NOT NULL, doc_length INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (term, doc_id))"
                )
                conn.execute(
                    "CREATE TABLE bm25_corpus_stats (term TEXT PRIMARY KEY, document_frequency INTEGER NOT NULL, idf REAL NOT NULL, last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                )
                conn.execute(
                    "CREATE TABLE bm25_doc_metadata (doc_id TEXT PRIMARY KEY, doc_length INTEGER NOT NULL, original_text TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                )
                conn.commit()

            ctx = _make_reset_context(tmpdir, chroma_live, chroma_temp, using_sqlite=True)
            commit_reset(ctx)

            assert chroma_live.read_text() == "new db"
            assert not chroma_temp.exists()


class TestRollbackReset:
    """Tests for rollback_reset."""

    def test_removes_temp(self) -> None:
        """Temp ChromaDB is removed, live is untouched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chroma_live = Path(tmpdir) / "chromadb"
            chroma_live.mkdir()
            (chroma_live / "live.db").write_text("live")

            chroma_temp = Path(tmpdir) / "chromadb.new"
            chroma_temp.mkdir()
            (chroma_temp / "partial.db").write_text("partial")

            bm25_stage = Path(tmpdir) / "bm25_reset_stage"
            bm25_stage.mkdir(parents=True)
            (bm25_stage / "cache.db").write_text("placeholder")

            ctx = _make_reset_context(tmpdir, chroma_live, chroma_temp, using_sqlite=False)
            rollback_reset(ctx)

            assert not chroma_temp.exists(), "Temp must be removed on rollback"
            assert not bm25_stage.exists(), "BM25 stage dir must be removed on rollback"
            assert chroma_live.exists(), "Live must be preserved on rollback"
            assert (chroma_live / "live.db").read_text() == "live"

    def test_safe_when_temp_never_created(self) -> None:
        """rollback_reset is a no-op when the temp path was never created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _make_reset_context(
                tmpdir,
                Path(tmpdir) / "chromadb",
                Path(tmpdir) / "chromadb.new",
                using_sqlite=False,
            )
            # Should not raise
            rollback_reset(ctx)

    def test_verbose_output(self, capsys) -> None:
        """Verbose mode prints rollback status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chroma_temp = Path(tmpdir) / "chromadb.new"
            chroma_temp.mkdir()

            ctx = _make_reset_context(
                tmpdir,
                Path(tmpdir) / "chromadb",
                chroma_temp,
                using_sqlite=False,
            )
            rollback_reset(ctx, verbose=True)

            captured = capsys.readouterr()
            assert "Rolled back" in captured.out or "preserved" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
