#!/usr/bin/env python3
"""
Utility to clear RAG databases for fresh ingestion.

Usage:
    python -m scripts.utils.clear_databases [--all] [--chromadb] [--graph] [--confirm]

Also provides a programmatic function for use in ingest scripts:
    from scripts.utils.clear_databases import clear_for_ingestion
    clear_for_ingestion(verbose=True)
"""

import argparse
import logging
import shutil
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.consistency_graph.consistency_config import get_consistency_config
from scripts.ingest.ingest_config import get_ingest_config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from scripts.consistency_graph.consistency_config import ConsistencyConfig
    from scripts.ingest.ingest_config import IngestConfig


HNSW_INDEX_FILES = {
    "data_level0.bin",
    "header.bin",
    "index_metadata.pickle",
    "length.bin",
    "link_lists.bin",
}


def _is_hnsw_index_dir(path: Path) -> bool:
    if not path.is_dir():
        return False

    try:
        filenames = {child.name for child in path.iterdir() if child.is_file()}
    except OSError:
        return False

    return HNSW_INDEX_FILES.issubset(filenames)


def _delete_path(path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY RUN] Would delete: {path}")
        return

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _iter_existing_paths(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def clear_chromadb(config: "IngestConfig", dry_run: bool = False) -> None:
    """Clear ChromaDB collections.

    Args:
        config: IngestConfig instance containing ChromaDB path
        dry_run: Show what would be deleted without deleting

    Note: On ``--reset`` ingestion runs, prefer :func:`prepare_reset_workspace`
    which preserves the live ChromaDB during ingestion and swaps it atomically
    on success, avoiding any window where the dashboard has no data.
    """
    chroma_dir = Path(config.rag_data_path) / "chromadb"
    sqlite_path = Path(config.rag_data_path) / "chromadb.db"
    existing_paths = _iter_existing_paths([chroma_dir, sqlite_path])

    if not existing_paths:
        print(f"✓ ChromaDB storage not found (already clean)")
        return

    for path in existing_paths:
        if dry_run:
            print(f"[DRY RUN] Would delete: {path}")
            continue

        print(f"Deleting ChromaDB at {path}...")
        _delete_path(path, dry_run=False)

    if not dry_run:
        print(f"✓ Cleared ChromaDB")


def clear_graph_database(config: "ConsistencyConfig", dry_run: bool = False) -> None:
    """Clear graph consistency database nodes and edges."""
    graph_db = Path(config.output_sqlite)

    if not graph_db.exists():
        print(f"✓ Graph database not found at {graph_db} (already clean)")
        return

    if dry_run:
        print(f"[DRY RUN] Would clear tables in: {graph_db}")
        return

    print(f"Clearing graph database at {graph_db}...")
    with closing(sqlite3.connect(str(graph_db))) as conn:
        cursor = conn.cursor()

        # Clear all data but keep schema
        tables = ["edges", "node_clusters", "nodes", "clusters"]
        for table in tables:
            try:
                cursor.execute(f"DELETE FROM {table}")
                print(f"  ✓ Cleared {table}")
            except sqlite3.OperationalError as e:
                print(f"  ⚠ Could not clear {table}: {e}")

        conn.commit()
    print(f"✓ Cleared graph database")


def clear_cache_database(config: "IngestConfig", dry_run: bool = False) -> None:
    """Clear embedding and LLM cache (ingest-related only).

    Preserves graph cache entries so the dashboard continues to work.
    Graph caches are rebuilt by build_consistency_graph.py when needed.

    Args:
        config: IngestConfig instance containing cache path
        dry_run: Show what would be deleted without deleting
    """
    cache_path = Path(config.cache_path)

    if not cache_path.exists():
        print(f"✓ Ingest cache not found (already clean)")
        return

    if dry_run:
        print(f"[DRY RUN] Would clear ingest tables in: {cache_path}")
        return

    print(f"Clearing ingest caches in {cache_path}...")
    try:
        with closing(sqlite3.connect(str(cache_path))) as conn:
            cursor = conn.cursor()

            # Clear ONLY ingest-related cache tables
            # Preserve graph_cache and graph_settings_map for dashboard
            # CRITICAL: Must clear BM25 corpus stats tables to prevent lock conflicts and stale index
            ingest_tables = [
                "embeddings_cache",
                "llm_cache",
                "bm25_cache",
                "bm25_corpus_stats",  # IDF and document frequency - corpus-specific
                "bm25_doc_metadata",  # Document lengths and metadata - corpus-specific
                "bm25_index",  # Inverted index - corpus-specific, must clear to prevent stale results
                "word_frequency",  # Word cloud frequencies - corpus-specific, must clear to prevent stale results
            ]

            for table in ingest_tables:
                try:
                    cursor.execute(f"DELETE FROM {table}")
                    rows_deleted = cursor.rowcount
                    print(f"  ✓ Cleared {table} ({rows_deleted:,} rows)")
                except sqlite3.OperationalError:
                    # Table doesn't exist, that's fine
                    pass

            conn.commit()
        print(f"✓ Ingest caches cleared (graph caches preserved)")
    except Exception as e:
        print(f"✗ Error clearing cache database: {e}")


def clear_bm25_cache(config: "IngestConfig", dry_run: bool = False) -> None:
    """Clear BM25 keyword index cache (needs reindexing on new corpus).

    Args:
        config: IngestConfig instance containing rag_data_path
        dry_run: Show what would be deleted without deleting
    """
    bm25_path = Path(config.rag_data_path) / "bm25_index"

    if not bm25_path.exists():
        print(f"✓ BM25 index not found (already clean)")
        return

    if dry_run:
        print(f"[DRY RUN] Would delete: {bm25_path}")
        return

    print(f"Deleting BM25 index at {bm25_path}...")
    shutil.rmtree(bm25_path)
    print(f"✓ Cleared BM25 index")


def clear_reference_cache(config: "IngestConfig", dry_run: bool = False) -> None:
    """Clear academic reference cache (provider metadata should be re-fetched on reset).

    Args:
        config: IngestConfig instance containing rag_data_path
        dry_run: Show what would be deleted without deleting
    """
    cache_path = Path(config.rag_data_path) / "academic_references.db"

    if not cache_path.exists():
        print(f"✓ Reference cache not found (already clean)")
        return

    if dry_run:
        print(f"[DRY RUN] Would delete: {cache_path}")
        return

    print(f"Deleting reference cache at {cache_path}...")
    try:
        cache_path.unlink()
        print(f"✓ Cleared reference cache")
    except Exception as e:
        print(f"✗ Error clearing reference cache: {e}")


def clear_terminology_database(config: "IngestConfig", dry_run: bool = False) -> None:
    """Clear domain terminology database (must be re-extracted on new corpus).

    Args:
        config: IngestConfig instance containing rag_data_path
        dry_run: Show what would be deleted without deleting
    """
    terminology_path = Path(config.rag_data_path) / "academic_terminology.db"

    if not terminology_path.exists():
        print(f"✓ Terminology database not found (already clean)")
        return

    if dry_run:
        print(f"[DRY RUN] Would delete: {terminology_path}")
        return

    print(f"Deleting terminology database at {terminology_path}...")
    try:
        terminology_path.unlink()
        print(f"✓ Cleared terminology database")
    except Exception as e:
        print(f"✗ Error clearing terminology database: {e}")


def clear_semantic_clustering_cache(config: "IngestConfig", dry_run: bool = False) -> None:
    """Clear semantic clustering cache (must be rebuilt on new corpus).

    Args:
        config: IngestConfig instance containing rag_data_path
        dry_run: Show what would be deleted without deleting
    """
    cache_path = Path(config.rag_data_path) / "cache" / "semantic_clustering.db"

    if not cache_path.exists():
        print(f"✓ Semantic clustering cache not found (already clean)")
        return

    if dry_run:
        print(f"[DRY RUN] Would delete: {cache_path}")
        return

    print(f"Deleting semantic clustering cache at {cache_path}...")
    try:
        cache_path.unlink()
        print(f"✓ Cleared semantic clustering cache")
    except Exception as e:
        print(f"✗ Error clearing semantic clustering cache: {e}")


def clear_citation_graph_database(config: "IngestConfig", dry_run: bool = False) -> None:
    """Clear academic citation graph database and JSON export.

    Args:
        config: IngestConfig instance containing rag_data_path
        dry_run: Show what would be deleted without deleting
    """
    db_path = Path(config.rag_data_path) / "academic_citation_graph.db"
    temp_db_path = Path(config.rag_data_path) / "academic_citation_graph_temp.db"
    json_path = Path(config.rag_data_path) / "academic_citation_graph.json"

    sidecar_suffixes = ["-wal", "-shm", "-journal"]
    db_sidecars = [Path(f"{db_path}{suffix}") for suffix in sidecar_suffixes]
    temp_sidecars = [Path(f"{temp_db_path}{suffix}") for suffix in sidecar_suffixes]

    all_paths = _iter_existing_paths(
        [db_path, temp_db_path, json_path] + db_sidecars + temp_sidecars
    )

    if not all_paths:
        print("✓ Citation graph not found (already clean)")
        return

    for path in all_paths:
        if dry_run:
            print(f"[DRY RUN] Would delete: {path}")
            continue

        _delete_path(path, dry_run=False)

    if not dry_run:
        print("✓ Cleared citation graph")


def clear_academic_pdf_cache(config: "IngestConfig", dry_run: bool = False) -> None:
    """Clear cached academic PDFs downloaded during ingestion.

    Args:
        config: IngestConfig instance containing rag_data_path
        dry_run: Show what would be deleted without deleting
    """
    cache_dir = Path(config.rag_data_path) / "academic_pdfs"

    if not cache_dir.exists():
        print("✓ Academic PDF cache not found (already clean)")
        return

    if dry_run:
        print(f"[DRY RUN] Would delete: {cache_dir}")
        return

    print(f"Deleting academic PDF cache at {cache_dir}...")
    _delete_path(cache_dir, dry_run=False)
    print("✓ Cleared academic PDF cache")


def clear_legacy_artifacts(config: "IngestConfig", dry_run: bool = False) -> None:
    """Clear legacy artefacts that are no longer used by current ingestion.

    Args:
        config: IngestConfig instance containing rag_data_path
        dry_run: Show what would be deleted without deleting
    """
    rag_data_path = Path(config.rag_data_path)
    legacy_paths = [
        rag_data_path / "chroma_db",
        rag_data_path / "embedding_cache",
        rag_data_path / "llm_cache",
        rag_data_path / "graph.db",
        rag_data_path / "graphs.db",
        rag_data_path / "consistency_graph.sqlite",
        rag_data_path / "academic_citations.db",
    ]

    hnsw_dirs = [path for path in rag_data_path.iterdir() if _is_hnsw_index_dir(path)]
    all_paths = _iter_existing_paths(legacy_paths) + hnsw_dirs

    if not all_paths:
        print("✓ No legacy artefacts found (already clean)")
        return

    for path in all_paths:
        if dry_run:
            print(f"[DRY RUN] Would delete: {path}")
            continue

        _delete_path(path, dry_run=False)

    if not dry_run:
        print("✓ Cleared legacy artefacts")


# =============================================================================
# Temp-DB swap helpers (safe reset semantics)
# =============================================================================


@dataclass
class ResetContext:
    """Paths used during a safe-reset ingestion run.

    The ingestion pipeline writes the new ChromaDB to ``chroma_path_temp``
    while leaving ``chroma_path_live`` untouched.  On success, call
    :func:`commit_reset` to atomically promote the temp path to live.  On
    failure, call :func:`rollback_reset` to discard the incomplete temp DB.

    Attributes:
        chroma_path_live: Production ChromaDB path (directory or .db file).
        chroma_path_temp: Shadow path to write the new ChromaDB during ingest.
        bm25_cache_live_db: Live cache.db path serving hybrid search.
        bm25_cache_temp_db: Temp cache.db path used to stage rebuilt BM25 tables.
        bm25_stage_dir: Temporary rag_data-like directory for staged BM25 writes.
        using_sqlite: True when the SQLite-backed vector client is active.
    """

    chroma_path_live: Path
    chroma_path_temp: Path
    bm25_cache_live_db: Path
    bm25_cache_temp_db: Path
    bm25_stage_dir: Path
    using_sqlite: bool


def _remove_path(path: Path) -> None:
    """Remove a file or directory tree, ignoring already-absent paths."""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _ensure_bm25_tables(conn: sqlite3.Connection) -> None:
    """Ensure BM25 tables exist in a cache database connection."""
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS bm25_index (
            term TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            term_frequency INTEGER NOT NULL,
            doc_length INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (term, doc_id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS bm25_corpus_stats (
            term TEXT PRIMARY KEY,
            document_frequency INTEGER NOT NULL,
            idf REAL NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS bm25_doc_metadata (
            doc_id TEXT PRIMARY KEY,
            doc_length INTEGER NOT NULL,
            original_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bm25_term ON bm25_index(term)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bm25_doc ON bm25_index(doc_id)")
    conn.commit()


def _commit_bm25_tables(live_db: Path, staged_db: Path) -> None:
    """Replace live BM25 tables with staged BM25 tables in one transaction."""
    if not staged_db.exists():
        raise FileNotFoundError(f"Staged BM25 cache DB not found: {staged_db}")

    live_db.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(str(live_db))) as live_conn:
        _ensure_bm25_tables(live_conn)

        live_conn.execute("ATTACH DATABASE ? AS staged", (str(staged_db),))
        try:
            live_conn.execute("BEGIN IMMEDIATE")

            live_conn.execute("DELETE FROM bm25_index")
            live_conn.execute("DELETE FROM bm25_corpus_stats")
            live_conn.execute("DELETE FROM bm25_doc_metadata")

            live_conn.execute(
                """
                INSERT INTO bm25_index (term, doc_id, term_frequency, doc_length, created_at)
                SELECT term, doc_id, term_frequency, doc_length, created_at
                FROM staged.bm25_index
                """
            )
            live_conn.execute(
                """
                INSERT INTO bm25_corpus_stats (term, document_frequency, idf, last_updated)
                SELECT term, document_frequency, idf, last_updated
                FROM staged.bm25_corpus_stats
                """
            )
            live_conn.execute(
                """
                INSERT INTO bm25_doc_metadata (doc_id, doc_length, original_text, created_at)
                SELECT doc_id, doc_length, original_text, created_at
                FROM staged.bm25_doc_metadata
                """
            )

            live_conn.commit()
        except Exception:
            live_conn.rollback()
            raise
        finally:
            live_conn.execute("DETACH DATABASE staged")


def prepare_reset_workspace(
    config: "IngestConfig",
    using_sqlite: bool,
    verbose: bool = False,
) -> ResetContext:
    """Clear auxiliary data and set up a shadow ChromaDB path for ingestion.

    Auxiliary databases (caches, BM25 index, reference cache, etc.) are
    cleared immediately — they are derived data that will be repopulated
    during the new ingest run and are not user-facing.

    The live ChromaDB is left completely intact so the dashboard remains
    readable during ingestion.  The ingestion pipeline should write its new
    ChromaDB to ``context.chroma_path_temp`` and call :func:`commit_reset`
    when complete.

    Any stale temp path left behind by a previous interrupted reset is
    cleaned up automatically with a warning.

    Args:
        config: IngestConfig with ``rag_data_path`` and cache paths.
        using_sqlite: Pass the same flag used to construct the vector client.
        verbose: Print progress messages.

    Returns:
        :class:`ResetContext` containing the live and temp ChromaDB paths.
    """
    rag_data_path = Path(config.rag_data_path)

    if using_sqlite:
        chroma_live = rag_data_path / "chromadb.db"
        chroma_temp = rag_data_path / "chromadb.db.new"
    else:
        chroma_live = rag_data_path / "chromadb"
        chroma_temp = rag_data_path / "chromadb.new"

    if verbose:
        print("\n" + "=" * 60)
        print("PREPARING RESET WORKSPACE (safe swap mode)")
        print("=" * 60)
        print(f"  Live ChromaDB : {chroma_live}  (preserved during ingest)")
        print(f"  Temp ChromaDB : {chroma_temp}  (ingest writes here)")
        print()

    # Clean up any stale temp from a previous interrupted reset.
    backup = Path(str(chroma_live) + ".old")
    for stale in (chroma_temp, backup):
        if stale.exists():
            logger.warning(
                "Removing stale path from previous interrupted reset: %s", stale
            )
            if verbose:
                print(f"  ⚠ Removing stale path: {stale}")
            _remove_path(stale)

    bm25_stage_dir = rag_data_path / "bm25_reset_stage"
    bm25_cache_temp_db = bm25_stage_dir / "cache.db"
    bm25_cache_live_db = Path(config.cache_path)

    # Clear all auxiliary derived data (none of these affect dashboard availability).
    # IMPORTANT: Do NOT clear live BM25 tables here. They remain available for
    # hybrid search while ingestion is running. BM25 is rebuilt into a staged DB
    # and merged on commit.
    if verbose:
        print("→ Clearing auxiliary databases and caches...")

    if bm25_stage_dir.exists():
        logger.warning("Removing stale BM25 stage directory: %s", bm25_stage_dir)
        if verbose:
            print(f"  ⚠ Removing stale BM25 stage dir: {bm25_stage_dir}")
        _remove_path(bm25_stage_dir)
    bm25_stage_dir.mkdir(parents=True, exist_ok=True)

    # Initialise staged cache DB schema.
    with closing(sqlite3.connect(str(bm25_cache_temp_db))) as staged_conn:
        _ensure_bm25_tables(staged_conn)

    clear_reference_cache(config, dry_run=False)
    clear_terminology_database(config, dry_run=False)
    clear_semantic_clustering_cache(config, dry_run=False)
    clear_citation_graph_database(config, dry_run=False)
    clear_academic_pdf_cache(config, dry_run=False)
    clear_legacy_artifacts(config, dry_run=False)

    if verbose:
        print(
            "\n✓ Auxiliary data cleared. Ingestion will write to temp ChromaDB.\n"
            "  BM25 writes are staged and live hybrid search remains available.\n"
            "  Dashboard continues to serve from live ChromaDB until commit.\n"
        )

    return ResetContext(
        chroma_path_live=chroma_live,
        chroma_path_temp=chroma_temp,
        bm25_cache_live_db=bm25_cache_live_db,
        bm25_cache_temp_db=bm25_cache_temp_db,
        bm25_stage_dir=bm25_stage_dir,
        using_sqlite=using_sqlite,
    )


def commit_reset(context: ResetContext, verbose: bool = False) -> None:
    """Atomically replace the live ChromaDB with the freshly ingested temp.

    Uses a backup rename (``live → live.old``) before moving ``temp → live``
    so that a crash during the swap leaves a recoverable state: either the
    old ``live`` path still exists (rename not yet applied) or the new
    ``live.old`` path exists and can be restored manually.

    The caller **must** close all open ChromaDB connections pointing at
    ``context.chroma_path_temp`` before calling this function, otherwise
    Windows file locks (and some Linux flock users) will prevent the rename.

    Args:
        context: Reset context returned by :func:`prepare_reset_workspace`.
        verbose: Print progress messages.

    Raises:
        FileNotFoundError: If ``chroma_path_temp`` does not exist (ingestion
            produced no output).
        OSError: If the filesystem rename fails.
    """
    if not context.chroma_path_temp.exists():
        raise FileNotFoundError(
            f"Temp ChromaDB not found at {context.chroma_path_temp}. "
            "Ingestion may not have written any data."
        )

    if verbose:
        print("\n[RESET] Committing temp ChromaDB to live path...")

    backup = Path(str(context.chroma_path_live) + ".old")

    # Step 1: move live → live.old so we have a fallback if step 2 fails.
    if context.chroma_path_live.exists():
        context.chroma_path_live.rename(backup)
        logger.debug("Renamed live ChromaDB to backup: %s", backup)

    try:
        # Step 2: move temp → live (atomic on POSIX, best-effort on Windows).
        shutil.move(str(context.chroma_path_temp), str(context.chroma_path_live))
        logger.info(
            "ChromaDB committed: %s → %s", context.chroma_path_temp, context.chroma_path_live
        )
    except Exception:
        # Restore backup so the live path is not empty.
        if backup.exists() and not context.chroma_path_live.exists():
            backup.rename(context.chroma_path_live)
            logger.warning("Commit failed — original ChromaDB restored from backup")
        raise

    # Step 3: remove backup now that live is confirmed in place.
    if backup.exists():
        _remove_path(backup)
        logger.debug("Backup removed after successful commit")

    # Step 4: atomically replace live BM25 tables with staged BM25 tables.
    _commit_bm25_tables(context.bm25_cache_live_db, context.bm25_cache_temp_db)

    # Step 5: remove staged BM25 directory after successful merge.
    if context.bm25_stage_dir.exists():
        _remove_path(context.bm25_stage_dir)
        logger.debug("BM25 stage directory removed after successful commit")

    if verbose:
        print(f"[RESET] ✓ ChromaDB committed to {context.chroma_path_live}")
        print(f"[RESET] ✓ BM25 cache committed to {context.bm25_cache_live_db}")


def rollback_reset(context: ResetContext, verbose: bool = False) -> None:
    """Discard the incomplete temp ChromaDB; the live DB is left intact.

    Safe to call even if the temp path was never created (e.g. ingest failed
    before any data was written).

    Args:
        context: Reset context returned by :func:`prepare_reset_workspace`.
        verbose: Print progress messages.
    """
    if context.chroma_path_temp.exists():
        _remove_path(context.chroma_path_temp)
        logger.info("Rolled back incomplete temp ChromaDB: %s", context.chroma_path_temp)
        if verbose:
            print(f"[RESET] Rolled back incomplete temp ChromaDB: {context.chroma_path_temp}")

    if context.bm25_stage_dir.exists():
        _remove_path(context.bm25_stage_dir)
        logger.info("Rolled back staged BM25 cache: %s", context.bm25_stage_dir)
        if verbose:
            print(f"[RESET] Rolled back staged BM25 cache: {context.bm25_stage_dir}")

    if verbose:
        print(f"[RESET] ✓ Original ChromaDB preserved at {context.chroma_path_live}")
        print(f"[RESET] ✓ Original BM25 cache preserved at {context.bm25_cache_live_db}")


def clear_for_ingestion(verbose: bool = False, dry_run: bool = False) -> bool:
    """
    Clear databases and caches for a fresh ingestion.

    This is the function used by ingest scripts when --reset is called.

    Clears:
    - ChromaDB (chunks and documents) - corpus changed, old data invalid
    - Reference cache - provider metadata should be re-fetched for new documents
    - Ingest caches (embeddings, LLM results) - will be regenerated
    - BM25 index - MUST clear as it's corpus-specific; stale index causes erroneous results
    - Word frequency cache - corpus-specific; stale frequencies cause incorrect word cloud
    - Terminology database - must be re-extracted with current stop word filters
    - Citation graph - regenerated during academic ingestion
    - Academic PDF cache - re-downloaded as needed
    - Legacy artefacts - stale paths from older storage layouts

    Preserves:
    - Graph caches (rebuilt by build_consistency_graph.py)
    - Graph database (rebuilt by build_consistency_graph.py)

    Args:
        verbose: Print status messages
        dry_run: Show what would be deleted without deleting

    Returns:
        True if successful, False if any errors occurred
    """
    try:
        ingest_config = get_ingest_config()

        if verbose:
            print("\n" + "=" * 60)
            print("CLEARING INGESTION DATABASES & CACHES")
            print("=" * 60)
            if dry_run:
                print("[DRY RUN MODE - No changes will be made]\n")
            else:
                print()

        # Clear ChromaDB (chunks and documents)
        if verbose:
            print("→ Clearing ChromaDB collections...")
        clear_chromadb(ingest_config, dry_run=dry_run)

        # Clear ingest-specific caches
        if verbose:
            print("→ Clearing ingest caches...")
        clear_cache_database(ingest_config, dry_run=dry_run)

        # Clear BM25 index - CRITICAL: must clear as it's corpus-specific
        # Stale BM25 index will return erroneous results for documents that no longer exist
        if verbose:
            print("→ Clearing BM25 keyword index (corpus-specific)...")
        clear_bm25_cache(ingest_config, dry_run=dry_run)

        # Clear reference cache - should be re-fetched when ingesting new documents
        if verbose:
            print("→ Clearing academic reference cache...")
        clear_reference_cache(ingest_config, dry_run=dry_run)

        # Clear terminology database - must be re-extracted with current filters
        if verbose:
            print("→ Clearing terminology database...")
        clear_terminology_database(ingest_config, dry_run=dry_run)

        # Clear semantic clustering cache - must be rebuilt on new corpus
        if verbose:
            print("→ Clearing semantic clustering cache...")
        clear_semantic_clustering_cache(ingest_config, dry_run=dry_run)

        # Clear citation graph artefacts - regenerated during academic ingestion
        if verbose:
            print("→ Clearing citation graph artefacts...")
        clear_citation_graph_database(ingest_config, dry_run=dry_run)

        # Clear academic PDF cache
        if verbose:
            print("→ Clearing academic PDF cache...")
        clear_academic_pdf_cache(ingest_config, dry_run=dry_run)

        # Clear legacy artefacts (old storage layouts)
        if verbose:
            print("→ Clearing legacy artefacts...")
        clear_legacy_artifacts(ingest_config, dry_run=dry_run)

        if verbose:
            print("\n" + "=" * 60)
            if dry_run:
                print("✓ DRY RUN COMPLETE - Ready to proceed with --confirm flag")
            else:
                print("✓ INGESTION DATABASES CLEARED")
                print("  (Graph database and caches preserved - will rebuild)")
                print("  (Query tab will show unavailable until re-ingestion completes)")
            print("=" * 60 + "\n")

        return True
    except Exception as e:
        print(f"✗ Error during database cleanup: {e}")
        return False


def show_current_state(config: Optional["IngestConfig"] = None) -> None:
    """Show current database state.

    Args:
        config: IngestConfig instance (uses defaults if None)
    """
    if config is None:
        config = get_ingest_config()

    print("\n=== Current Database State ===\n")

    # ChromaDB
    chroma_path = Path(config.rag_data_path) / "chromadb"
    sqlite_path = Path(config.rag_data_path) / "chromadb.db"
    if chroma_path.exists():
        client = None
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(chroma_path))
            collections = client.list_collections()
            print(f"ChromaDB ({chroma_path}):")
            if collections:
                for coll in collections:
                    print(f"  - {coll.name}: {coll.count()} documents")
            else:
                print(f"  (empty)")
        except Exception as e:
            print(f"ChromaDB: Error reading - {e}")
        finally:
            if client is not None:
                try:
                    system = getattr(client, "_system", None)
                    stop = getattr(system, "stop", None)
                    if callable(stop):
                        stop()
                except Exception:
                    pass
    elif sqlite_path.exists():
        size_mb = sqlite_path.stat().st_size / 1024 / 1024
        print(f"ChromaDB (sqlite) at {sqlite_path}: {size_mb:.1f} MB")
    else:
        print(f"ChromaDB: Not found")

    # Graph database
    graph_db = Path(config.rag_data_path) / "consistency_graphs" / "consistency_graph.sqlite"
    if graph_db.exists():
        try:
            with closing(sqlite3.connect(str(graph_db))) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM nodes")
                node_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM edges")
                edge_count = cursor.fetchone()[0]
            print(f"\nGraph Database ({graph_db}):")
            print(f"  - Nodes: {node_count}")
            print(f"  - Edges: {edge_count}")
        except Exception as e:
            print(f"Graph Database: Error reading - {e}")
    else:
        print(f"\nGraph Database: Not found")

    # Cache
    cache_path = Path(config.cache_path)
    if cache_path.exists():
        size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"\nCache Database: {size_mb:.1f} MB")
    else:
        print(f"\nCache Database: Not found")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clear RAG databases for fresh ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current state
  python -m scripts.utils.clear_databases --status
  
  # Clear everything (with confirmation)
  python -m scripts.utils.clear_databases --all --confirm
  
  # Clear only ChromaDB
  python -m scripts.utils.clear_databases --chromadb --confirm
  
  # Dry run to see what would be deleted
  python -m scripts.utils.clear_databases --all --dry-run
        """,
    )

    parser.add_argument("--all", action="store_true", help="Clear all databases")
    parser.add_argument("--chromadb", action="store_true", help="Clear ChromaDB only")
    parser.add_argument("--graph", action="store_true", help="Clear graph database only")
    parser.add_argument("--cache", action="store_true", help="Clear cache database only")
    parser.add_argument("--confirm", action="store_true", help="Confirm deletion (required)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be deleted without deleting"
    )
    parser.add_argument("--status", action="store_true", help="Show current database state")

    args = parser.parse_args()

    # Show status if requested
    if args.status or (not any([args.all, args.chromadb, args.graph, args.cache])):
        show_current_state()
        if not args.status:
            print("Use --all, --chromadb, --graph, or --cache to clear databases")
            print("Add --confirm to actually delete (or --dry-run to preview)")
        return

    # Require confirmation unless dry run
    if not args.confirm and not args.dry_run:
        print("⚠ WARNING: This will permanently delete database contents!")
        print("Add --confirm to proceed or --dry-run to preview")
        return

    if args.dry_run:
        print("=== DRY RUN MODE (no changes will be made) ===\n")

    # Load configs
    ingest_config = get_ingest_config()
    graph_config = get_consistency_config()

    # Clear databases
    if args.all or args.chromadb:
        clear_chromadb(ingest_config, dry_run=args.dry_run)

    if args.all or args.graph:
        clear_graph_database(graph_config, dry_run=args.dry_run)

    if args.all or args.cache:
        clear_cache_database(ingest_config, dry_run=args.dry_run)

    if args.all:
        clear_bm25_cache(ingest_config, dry_run=args.dry_run)
        clear_reference_cache(ingest_config, dry_run=args.dry_run)
        clear_terminology_database(ingest_config, dry_run=args.dry_run)
        clear_citation_graph_database(ingest_config, dry_run=args.dry_run)
        clear_academic_pdf_cache(ingest_config, dry_run=args.dry_run)
        clear_legacy_artifacts(ingest_config, dry_run=args.dry_run)

    print("\n✓ Done!")

    if not args.dry_run:
        print("\nYou can now run ingestion with fresh databases.")


if __name__ == "__main__":
    main()
