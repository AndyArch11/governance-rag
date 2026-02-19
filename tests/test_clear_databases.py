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
    clear_academic_pdf_cache,
    clear_bm25_cache,
    clear_cache_database,
    clear_citation_graph_database,
    clear_chromadb,
    clear_graph_database,
    clear_legacy_artifacts,
    clear_reference_cache,
    clear_terminology_database,
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
        self.output_sqlite = str(Path(rag_data_path) / "consistency_graphs" / "consistency_graph.sqlite")


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
            cursor.execute("CREATE TABLE edges (id INTEGER PRIMARY KEY, source INTEGER, target INTEGER)")
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
            cursor.execute("CREATE TABLE embeddings_cache (id INTEGER PRIMARY KEY, text TEXT, embedding BLOB)")
            cursor.execute("CREATE TABLE llm_cache (id INTEGER PRIMARY KEY, prompt TEXT, response TEXT)")
            cursor.execute("CREATE TABLE bm25_cache (id INTEGER PRIMARY KEY, term TEXT)")
            cursor.execute("CREATE TABLE bm25_corpus_stats (id INTEGER PRIMARY KEY, stat TEXT)")
            cursor.execute("CREATE TABLE bm25_doc_metadata (id INTEGER PRIMARY KEY, doc_id INTEGER)")
            cursor.execute("CREATE TABLE bm25_index (id INTEGER PRIMARY KEY, term_id INTEGER)")
            cursor.execute("CREATE TABLE word_frequency (word TEXT PRIMARY KEY, frequency INTEGER, doc_count INTEGER)")
            cursor.execute("CREATE TABLE graph_cache (id INTEGER PRIMARY KEY, data TEXT)")
            cursor.execute("CREATE TABLE graph_settings_map (id INTEGER PRIMARY KEY, setting TEXT)")
            
            # Insert data into all tables
            cursor.execute("INSERT INTO embeddings_cache (text) VALUES ('text1')")
            cursor.execute("INSERT INTO llm_cache (prompt, response) VALUES ('q', 'a')")
            cursor.execute("INSERT INTO word_frequency (word, frequency, doc_count) VALUES ('test', 5, 2)")
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
