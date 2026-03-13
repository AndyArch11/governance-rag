"""Document ingestion pipeline for RAG system.

This module implements a multi-threaded document ingestion pipeline that processes
HTML documents from Confluence exports, extracts text, generates metadata, creates
embeddings, and stores versioned documents in ChromaDB for retrieval-augmented
generation (RAG) applications.

Process Overview:
    Phase 1 - Document Ingestion:
        - Discover HTML and PDF documents in source directory
        - Extract and clean text content
        - Generate document metadata and summaries using LLM
        - Create semantic chunks with validation and repair
        - Detect semantic drift between versions
        - Calculate document health metrics
        - Store chunks and embeddings in ChromaDB
        - Maintain version history with automatic pruning

    Phase 2 - Graph Building (separate module):
        - Retrieve documents from ChromaDB
        - Query similar documents using vector indexes
        - Run consistency checks between document pairs
        - Build consistency graph with nodes and edges
        - Persist graph for dashboard visualisation

Configuration:
    Environment variables control all configuration (see IngestConfig class).
    Key settings: RAG_BASE_PATH, RAG_DATA_PATH, MAX_WORKERS, VERSIONS_TO_KEEP

Usage:
    Basic ingestion:
        $ python ingest.py

    Reset collections and reingest:
        $ python ingest.py --reset

    Limit files for testing:
        $ python ingest.py --limit 10 --workers 2

    Dry run to preview changes:
        $ python ingest.py --dry-run --verbose

TODO:
    - Add OpenTelemetry and OTLP export for observability
    - Add ability to interpret diagrams from documents
    - Deduplication across near-identical docs/versions (minhash or embedding similarity) to avoid redundant storage.
    - Improve LLM cost estimates based on real usage data and dynamic pricing models.
    - Implement incremental ingestion for large doc sets with change detection.
    - Include option to estimate LLM and embedding costs during dry-run mode without actual LLM API calls.
    - Extend support for additional document formats (RTF, ODT, ODF, DOCX, PPTX?) with robust parsing and cleaning.
"""

# Standard library imports
import argparse
import getpass
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

# Ensure project root is importable when running as a script path, e.g.:
# python scripts/ingest/ingest.py --help
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

from scripts.utils.clear_databases import clear_for_ingestion
from scripts.utils.db_factory import get_cache_client, get_default_vector_path, get_vector_client
from scripts.utils.logger import _loggers, create_module_logger
from scripts.utils.metrics_export import get_metrics_collector
from scripts.utils.monitoring import get_perf_metrics, init_monitoring
from scripts.utils.rate_limiter import init_rate_limiter
from scripts.utils.resource_monitor import ResourceMonitor
from scripts.ingest.ingest_utils import (
    check_ollama_availability,
    compute_chunk_hash,
    compute_doc_id,
    compute_file_hash,
)

get_logger, audit = create_module_logger("ingest")

try:
    from scripts.rag.domain_terms import DomainType, get_domain_term_manager, resolve_domain_type
except ImportError:
    DomainType = None  # type: ignore[assignment]
    get_domain_term_manager = None  # type: ignore[assignment]
    resolve_domain_type = None  # type: ignore[assignment]

# Optional import for error handling compatibility
try:
    import chromadb  # noqa: WPS433
except ImportError:
    chromadb = None

if chromadb is not None:
    NotFoundError = chromadb.errors.NotFoundError
else:  # Fallback to avoid NameError when ChromaDB is absent

    class NotFoundError(Exception):
        pass


# Centralised backend selection (ChromaDB preferred, SQLite fallback)
PersistentClient, USING_SQLITE = get_vector_client(prefer="chroma")

# Collection typing (best-effort, tolerates missing backends)
from typing import Any, Union

try:
    from chromadb.api.models.Collection import (  # type: ignore  # noqa: WPS433,E402
        Collection as ChromaDBCollection,
    )
except Exception:
    ChromaDBCollection = Any  # type: ignore

try:
    from scripts.ingest.chromadb_sqlite import ChromaSQLiteCollection  # noqa: WPS433,E402
except Exception:
    ChromaSQLiteCollection = Any  # type: ignore

Collection = Union[ChromaDBCollection, ChromaSQLiteCollection, Any]

# Import IngestConfig from proper location
try:
    if __package__:
        from .ingest_config import IngestConfig, build_cli_overrides, get_ingest_config
    else:
        from scripts.ingest.ingest_config import IngestConfig, build_cli_overrides, get_ingest_config
except ImportError:
    # Fallback should not happen in normal operation
    raise ImportError("Cannot import IngestConfig from ingest_config module")

# Handle both package imports and direct script execution
if __name__ == "__main__" and __package__ is None:
    # Running as script
    from scripts.ingest.bm25_indexing import index_chunks_in_bm25
    from scripts.ingest.chunk import chunk_text, create_parent_child_chunks, extract_technical_entities
    from scripts.ingest.embedding_cache import EmbeddingCache
    from scripts.ingest.htmlparser import extract_text_from_html
    from scripts.ingest.llm_cache import LLMCache
    from scripts.ingest.pdfparser import extract_text_from_pdf
    from scripts.ingest.preprocess import preprocess_text, redact_sensitive_text
    from scripts.ingest.vectors import (
        EMBEDDING_MODEL_NAME,
        delete_document_chunks,
        get_existing_doc_hash,
        store_child_chunks,
        store_chunks_in_chroma,
        store_parent_chunks,
    )
    from scripts.utils.retry_utils import retry_chromadb_call
else:
    # Running as module - use relative imports
    from scripts.utils.retry_utils import retry_chromadb_call

    from .bm25_indexing import index_chunks_in_bm25
    from .chunk import chunk_text, create_parent_child_chunks, extract_technical_entities
    from .embedding_cache import EmbeddingCache
    from .htmlparser import extract_text_from_html
    from .llm_cache import LLMCache
    from .pdfparser import extract_text_from_pdf
    from .preprocess import preprocess_text, redact_sensitive_text
    from .vectors import (
        EMBEDDING_MODEL_NAME,
        delete_document_chunks,
        get_existing_doc_hash,
        store_child_chunks,
        store_chunks_in_chroma,
        store_parent_chunks,
    )

# =========================
# CONFIG
# =========================


class ProgressTracker:
    """Lightweight progress tracker for headless runs.

    Provides periodic logging updates for batch processing jobs that run
    in headless environments where tqdm progress bars aren't visible.
    Thread-safe for concurrent document processing.

    Attributes:
        total (int): Total number of items to process.
        completed (int): Number of items completed.
        succeeded (int): Number of successful completions.
        failed (int): Number of failed completions.
        start_time (float): Timestamp when processing started.
        logger: Logger instance for progress updates.
        log_interval (int): Log progress every N items.
        lock: Threading lock for thread-safe updates.

    Usage:
        tracker = ProgressTracker(total=100, log_interval=10, logger=logger)
        tracker.increment(success=True)  # Updates counters and logs periodically
    """

    def __init__(self, total: int, log_interval: int = 10, logger=None):
        """Initialise progress tracker.

        Args:
            total: Total number of items to process.
            log_interval: Log progress every N items (default 10).
            logger: Logger instance for output (optional).
        """
        self.total = total
        self.completed = 0
        self.succeeded = 0
        self.failed = 0
        self.start_time = time.perf_counter()
        self.logger = logger
        self.log_interval = log_interval
        self.lock = threading.Lock()
        self.last_log_count = 0

    def increment(self, success: bool = True) -> None:
        """Increment progress counters and log if interval reached.

        Args:
            success: Whether the operation succeeded (default True).
        """
        with self.lock:
            self.completed += 1
            if success:
                self.succeeded += 1
            else:
                self.failed += 1

            # Log at intervals or on completion
            if (self.completed - self.last_log_count >= self.log_interval) or (
                self.completed == self.total
            ):
                self._log_progress()
                self.last_log_count = self.completed

    def _log_progress(self) -> None:
        """Log current progress with metrics.

        Internal method called by increment() when interval reached.
        """
        elapsed = time.perf_counter() - self.start_time
        percent = (self.completed / self.total * 100) if self.total > 0 else 0

        # Calculate rate and ETA
        rate = self.completed / elapsed if elapsed > 0 else 0
        remaining = self.total - self.completed
        eta_seconds = remaining / rate if rate > 0 else 0

        # Format ETA
        if eta_seconds < 60:
            eta_str = f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            eta_str = f"{eta_seconds/60:.1f}m"
        else:
            eta_str = f"{eta_seconds/3600:.1f}h"

        msg = (
            f"Progress: {self.completed}/{self.total} ({percent:.1f}%) | "
            f"Success: {self.succeeded} | Failed: {self.failed} | "
            f"Rate: {rate:.1f} docs/s | ETA: {eta_str}"
        )

        if self.logger:
            self.logger.info(msg)
        else:
            print(f"[PROGRESS] {msg}")

        # Audit progress checkpoint
        audit(
            "progress_checkpoint",
            {
                "completed": self.completed,
                "total": self.total,
                "percent": percent,
                "succeeded": self.succeeded,
                "failed": self.failed,
                "rate_per_second": rate,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta_seconds,
            },
        )


class DryRunStats:
    """Statistics collector for dry-run preview reports.

    Tracks all operations that would be performed during actual ingestion
    to provide cost and time estimates before processing.
    Thread-safe for concurrent document processing.

    Attributes:
        docs_new (int): Count of new documents to be ingested.
        docs_updated (int): Count of existing documents to be updated.
        docs_skipped (int): Count of unchanged documents (would skip).
        chunks_total (int): Total chunks that would be created.
        llm_calls_metadata (int): LLM calls for metadata generation.
        llm_calls_validation (int): LLM calls for chunk validation.
        llm_calls_repair (int): LLM calls for chunk repair.
        embeddings_chunks (int): Embedding calls for chunks.
        embeddings_docs (int): Embedding calls for documents.
        estimated_time_seconds (float): Estimated total processing time.
        lock: Threading lock for thread-safe updates.
    """

    def __init__(self):
        """Initialise dry-run statistics tracker."""
        self.docs_new = 0
        self.docs_updated = 0
        self.docs_skipped = 0
        self.chunks_total = 0
        self.llm_calls_metadata = 0
        self.llm_calls_validation = 0
        self.llm_calls_repair = 0
        self.embeddings_chunks = 0
        self.embeddings_docs = 0
        self.estimated_time_seconds = 0.0
        self.lock = threading.Lock()

    def record_document(self, status: str, num_chunks: int, processing_time: float):
        """Record document processing statistics.

        Args:
            status: 'new', 'updated', or 'skipped'.
            num_chunks: Number of chunks in document.
            processing_time: Time taken to process document (seconds).
        """
        with self.lock:
            if status == "new":
                self.docs_new += 1
                self.llm_calls_metadata += 1  # Metadata generation
                self.embeddings_docs += 1  # Doc-level embedding
            elif status == "updated":
                self.docs_updated += 1
                self.llm_calls_metadata += 1
                self.embeddings_docs += 1
            elif status == "skipped":
                self.docs_skipped += 1
                return  # No further processing for skipped docs

            self.chunks_total += num_chunks
            # Estimate LLM validation calls (assume 20% need validation)
            self.llm_calls_validation += int(num_chunks * 0.2)
            # Estimate repair calls (assume 5% need repair)
            self.llm_calls_repair += int(num_chunks * 0.05)
            # All chunks need embeddings
            self.embeddings_chunks += num_chunks
            # Add processing time
            self.estimated_time_seconds += processing_time

    def get_report(self) -> str:
        """Generate formatted preview report.

        Returns:
            Multi-line formatted report string.
        """
        total_docs = self.docs_new + self.docs_updated + self.docs_skipped
        docs_to_process = self.docs_new + self.docs_updated

        # Calculate total LLM calls
        total_llm_calls = (
            self.llm_calls_metadata + self.llm_calls_validation + self.llm_calls_repair
        )

        # Calculate total embedding calls
        total_embeddings = self.embeddings_chunks + self.embeddings_docs

        # Estimate costs (approximate pricing)
        # Ollama is free but include cloud API estimates for reference
        # LLM API: ~$0.0002 per 1K tokens, avg 500 tokens/call = $0.0001/call
        # Embedding API: ~$0.0001 per 1K tokens, avg 200 tokens = $0.00002/call
        llm_cost_estimate = total_llm_calls * 0.0001
        embedding_cost_estimate = total_embeddings * 0.00002
        total_cost_estimate = llm_cost_estimate + embedding_cost_estimate

        # Format time estimate
        time_minutes = self.estimated_time_seconds / 60
        time_hours = time_minutes / 60

        if time_hours >= 1:
            time_str = f"{time_hours:.1f} hours"
        elif time_minutes >= 1:
            time_str = f"{time_minutes:.1f} minutes"
        else:
            time_str = f"{self.estimated_time_seconds:.0f} seconds"

        # Estimate parallel time (divide by 4 workers)
        parallel_seconds = self.estimated_time_seconds / 4
        parallel_minutes = parallel_seconds / 60
        parallel_hours = parallel_minutes / 60

        if parallel_hours >= 1:
            parallel_time_str = f"{parallel_hours:.1f} hours"
        elif parallel_minutes >= 1:
            parallel_time_str = f"{parallel_minutes:.1f} minutes"
        else:
            parallel_time_str = f"{parallel_seconds:.0f} seconds"

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                      DRY-RUN PREVIEW REPORT                      ║
╠══════════════════════════════════════════════════════════════════╣
║ DOCUMENT SUMMARY                                                 ║
║   Total documents discovered: {total_docs:>6}                                ║
║   New documents:              {self.docs_new:>6}                                ║
║   Updated documents:          {self.docs_updated:>6}                                ║
║   Skipped (unchanged):        {self.docs_skipped:>6}                                ║
║   Documents to process:       {docs_to_process:>6}                                ║
╠══════════════════════════════════════════════════════════════════╣
║ CHUNK ANALYSIS                                                   ║
║   Total chunks to create:     {self.chunks_total:>6}                                ║
║   Avg chunks per doc:         {(self.chunks_total / docs_to_process if docs_to_process > 0 else 0):>6.1f}                                ║
╠══════════════════════════════════════════════════════════════════╣
║ LLM OPERATIONS                                                   ║
║   Metadata generation:        {self.llm_calls_metadata:>6} calls                          ║
║   Chunk validation:           {self.llm_calls_validation:>6} calls (est. 20%)               ║
║   Chunk repair:               {self.llm_calls_repair:>6} calls (est. 5%)                ║
║   Total LLM calls:            {total_llm_calls:>6}                                ║
╠══════════════════════════════════════════════════════════════════╣
║ EMBEDDING OPERATIONS                                             ║
║   Chunk embeddings:           {self.embeddings_chunks:>6} calls                          ║
║   Document embeddings:        {self.embeddings_docs:>6} calls                          ║
║   Total embeddings:           {total_embeddings:>6}                                ║
╠══════════════════════════════════════════════════════════════════╣
║ ESTIMATED COSTS (Cloud API Reference)                            ║
║   LLM calls:                  ${llm_cost_estimate:>7.2f}                             ║
║   Embeddings:                 ${embedding_cost_estimate:>7.2f}                             ║
║   Total (if using cloud):     ${total_cost_estimate:>7.2f}                             ║
║   Note: Ollama (local) is FREE                                   ║
╠══════════════════════════════════════════════════════════════════╣
║ ESTIMATED PROCESSING TIME                                        ║
║   Sequential processing:      {time_str:<15}                      ║
║   With parallelisation:       {parallel_time_str:<15}   (4 workers)           ║
╚══════════════════════════════════════════════════════════════════╝

To proceed with ingestion, run without --dry-run flag.
"""
        return report


class ProfileStats:
    """Statistics collector for profile mode timing and error analysis.

    Tracks detailed performance metrics during profile runs to identify
    bottlenecks and provide optimisation recommendations.
    Thread-safe for concurrent document processing.

    Attributes:
        doc_count (int): Number of documents processed.
        chunk_count (int): Total chunks created.
        stage_times (dict): Time spent in each processing stage.
        errors (list): Errors encountered during processing.
        warnings (list): Warnings encountered during processing.
        doc_processing_times (list): Individual document processing times.
        llm_calls (int): Total LLM invocations.
        embedding_calls (int): Total embedding generations.
        cache_hits_llm (int): LLM cache hits.
        cache_hits_embedding (int): Embedding cache hits.
        lock: Threading lock for thread-safe updates.
    """

    def __init__(self):
        """Initialise profile statistics tracker."""
        self.doc_count = 0
        self.chunk_count = 0
        self.stage_times = {
            "parse": 0.0,
            "preprocess": 0.0,
            "chunk": 0.0,
            "validate": 0.0,
            "embed": 0.0,
            "store": 0.0,
        }
        self.errors = []
        self.warnings = []
        self.doc_processing_times = []
        self.llm_calls = 0
        self.embedding_calls = 0
        self.cache_hits_llm = 0
        self.cache_hits_embedding = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

    def record_stage(self, stage: str, duration: float):
        """Record time spent in a processing stage.

        Args:
            stage: Stage name ('parse', 'preprocess', 'chunk', etc.).
            duration: Time spent in stage (seconds).
        """
        with self.lock:
            if stage in self.stage_times:
                self.stage_times[stage] += duration

    def record_document(self, num_chunks: int, processing_time: float):
        """Record document processing completion.

        Args:
            num_chunks: Number of chunks created.
            processing_time: Total time to process document (seconds).
        """
        with self.lock:
            self.doc_count += 1
            self.chunk_count += num_chunks
            self.doc_processing_times.append(processing_time)

    def record_error(self, error_msg: str, doc_name: str = None):
        """Record an error during processing.

        Args:
            error_msg: Error description.
            doc_name: Document where error occurred (optional).
        """
        with self.lock:
            entry = f"{doc_name}: {error_msg}" if doc_name else error_msg
            self.errors.append(entry)

    def record_warning(self, warning_msg: str, doc_name: str = None):
        """Record a warning during processing.

        Args:
            warning_msg: Warning description.
            doc_name: Document where warning occurred (optional).
        """
        with self.lock:
            entry = f"{doc_name}: {warning_msg}" if doc_name else warning_msg
            self.warnings.append(entry)

    def record_llm_call(self, cache_hit: bool = False):
        """Record an LLM invocation.

        Args:
            cache_hit: Whether result came from cache.
        """
        with self.lock:
            self.llm_calls += 1
            if cache_hit:
                self.cache_hits_llm += 1

    def record_embedding(self, cache_hit: bool = False):
        """Record an embedding generation.

        Args:
            cache_hit: Whether result came from cache.
        """
        with self.lock:
            self.embedding_calls += 1
            if cache_hit:
                self.cache_hits_embedding += 1

    def get_report(self) -> str:
        """Generate formatted profile analysis report.

        Returns:
            Multi-line formatted report with timing, errors, and recommendations.
        """
        elapsed = time.time() - self.start_time

        # Calculate statistics
        avg_doc_time = (
            sum(self.doc_processing_times) / len(self.doc_processing_times)
            if self.doc_processing_times
            else 0
        )
        min_doc_time = min(self.doc_processing_times) if self.doc_processing_times else 0
        max_doc_time = max(self.doc_processing_times) if self.doc_processing_times else 0

        docs_per_sec = self.doc_count / elapsed if elapsed > 0 else 0
        chunks_per_doc = self.chunk_count / self.doc_count if self.doc_count > 0 else 0
        llm_per_doc = self.llm_calls / self.doc_count if self.doc_count > 0 else 0

        # Find slowest stage
        slowest_stage = max(self.stage_times.items(), key=lambda x: x[1])[0]
        slowest_time = self.stage_times[slowest_stage]
        slowest_pct = (
            slowest_time / sum(self.stage_times.values()) * 100
            if sum(self.stage_times.values()) > 0
            else 0
        )

        # Cache hit rates
        llm_cache_rate = self.cache_hits_llm / self.llm_calls * 100 if self.llm_calls > 0 else 0
        embed_cache_rate = (
            self.cache_hits_embedding / self.embedding_calls * 100
            if self.embedding_calls > 0
            else 0
        )

        # Build recommendations
        recommendations = []
        if slowest_stage == "validate" and slowest_pct > 40:
            recommendations.append(
                "• Enable ENABLE_CHUNK_HEURISTIC_SKIP to reduce validation overhead"
            )
        if llm_cache_rate < 30 and self.llm_calls > 10:
            recommendations.append("• LLM cache hit rate is low - ensure LLM_CACHE_ENABLED=true")
        if embed_cache_rate < 30 and self.embedding_calls > 20:
            recommendations.append(
                "• Embedding cache hit rate is low - ensure EMBEDDING_CACHE_ENABLED=true"
            )
        if chunks_per_doc > 100:
            recommendations.append(
                "• High chunk count - consider increasing chunk size for faster processing"
            )
        if docs_per_sec < 0.5 and self.doc_count > 5:
            recommendations.append(
                "• Slow throughput - consider increasing MAX_WORKERS (GPU permitting)"
            )
        if len(self.errors) > self.doc_count * 0.2:
            recommendations.append("⚠  High error rate - review errors below before full run")

        if not recommendations:
            recommendations.append("✓ Pipeline configuration looks optimal")

        # Format report
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                      PROFILE ANALYSIS REPORT                     ║
╠══════════════════════════════════════════════════════════════════╣
║ PROCESSING SUMMARY                                               ║
║   Documents processed:        {self.doc_count:>6}                                ║
║   Total chunks created:       {self.chunk_count:>6}                                ║
║   Total elapsed time:         {elapsed:>6.1f}s                               ║
║   Throughput:                 {docs_per_sec:>6.2f} docs/sec                        ║
╠══════════════════════════════════════════════════════════════════╣
║ DOCUMENT TIMING (seconds)                                        ║
║   Average per document:       {avg_doc_time:>6.2f}s                               ║
║   Fastest document:           {min_doc_time:>6.2f}s                               ║
║   Slowest document:           {max_doc_time:>6.2f}s                               ║
╠══════════════════════════════════════════════════════════════════╣
║ STAGE BREAKDOWN                                                  ║
║   Parse:                      {self.stage_times['parse']:>6.2f}s                               ║
║   Preprocess:                 {self.stage_times['preprocess']:>6.2f}s                               ║
║   Chunk:                      {self.stage_times['chunk']:>6.2f}s                               ║
║   Validate:                   {self.stage_times['validate']:>6.2f}s                               ║
║   Embed:                      {self.stage_times['embed']:>6.2f}s                               ║
║   Store:                      {self.stage_times['store']:>6.2f}s                               ║
║   Bottleneck:                 {slowest_stage:<10} ({slowest_pct:>5.1f}% of total)       ║
╠══════════════════════════════════════════════════════════════════╣
║ API EFFICIENCY                                                   ║
║   LLM calls:                  {self.llm_calls:>6} ({llm_per_doc:>5.1f} per doc)                ║
║   LLM cache hit rate:         {llm_cache_rate:>6.1f}%                              ║
║   Embedding calls:            {self.embedding_calls:>6}                                ║
║   Embedding cache rate:       {embed_cache_rate:>6.1f}%                              ║
╠══════════════════════════════════════════════════════════════════╣
║ QUALITY METRICS                                                  ║
║   Chunks per document:        {chunks_per_doc:>6.1f}                                ║
║   Errors encountered:         {len(self.errors):>6}                                ║
║   Warnings raised:            {len(self.warnings):>6}                                ║
╠══════════════════════════════════════════════════════════════════╣
║ OPTIMISATION RECOMMENDATIONS                                     ║
"""

        # Add recommendations
        for rec in recommendations:
            # Wrap long recommendations
            if len(rec) <= 62:
                report += f"║ {rec:<64} ║\n"
            else:
                # Split into multiple lines
                words = rec.split()
                line = "║ "
                for word in words:
                    if len(line) + len(word) + 1 <= 67:
                        line += word + " "
                    else:
                        report += f"{line:<68}║\n"
                        line = "║   " + word + " "
                if len(line.strip()) > 1:
                    report += f"{line:<68}║\n"

        report += "╚══════════════════════════════════════════════════════════════════╝\n"

        # Add error/warning details if present
        if self.errors:
            report += "\n⚠️  ERRORS DETECTED:\n"
            for i, error in enumerate(self.errors[:10], 1):  # Show first 10
                report += f"  {i}. {error}\n"
            if len(self.errors) > 10:
                report += f"  ... and {len(self.errors) - 10} more errors\n"

        if self.warnings:
            report += "\n⚠️  WARNINGS:\n"
            for i, warning in enumerate(self.warnings[:5], 1):  # Show first 5
                report += f"  {i}. {warning}\n"
            if len(self.warnings) > 5:
                report += f"  ... and {len(self.warnings) - 5} more warnings\n"

        report += f"\n{'='*68}\n"

        if len(self.errors) == 0:
            report += "✅ READY FOR PRODUCTION RUN\n"
            report += "   No errors detected. Pipeline configured correctly.\n"
            report += "   Remove --profile flag to process full dataset.\n"
        else:
            report += "⚠️  REVIEW REQUIRED\n"
            report += "   Errors detected during profile run.\n"
            report += "   Address issues above before full production run.\n"

        return report


def _is_binary_file(path: str, sniff_bytes: int = 4096) -> bool:
    """Return True if the file appears to be binary.

    Heuristics:
    - Contains NUL bytes in the first few KB
    - Starts with ZIP header (PK..) for non-HTML files
    - Starts with OLE Compound File signature for legacy Office binaries
    """
    try:
        with open(path, "rb") as f:
            head = f.read(sniff_bytes)
        if b"\x00" in head:
            return True
        # ZIP magic (used by docx/xlsx/pptx/odt/ods/odp)
        if head[:2] == b"PK":
            return True
        # OLE Compound File magic for legacy Office docs (doc/xls/ppt)
        if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
            return True
    except OSError:
        return False
    return False


def _should_skip_unsupported(
    path: str, verbose: bool = False, audit_fn: Optional[Any] = None, log_unsupported: bool = False
) -> bool:
    """Return True when a file should be skipped (unsupported extension or binary).

    Args:
        path: File path to check.
        verbose: Print skip messages to console.
        audit_fn: Optional audit function to log skip events.
        log_unsupported: If True, log unsupported files to audit. If False, only log binary files.

    Returns:
        True if file should be skipped, False otherwise.
    """
    filename = os.path.basename(path)
    ext = os.path.splitext(filename)[1].lower()
    reason = None
    is_binary = False

    if ext not in {".html", ".htm", ".pdf"}:
        reason = f"unsupported extension {ext or 'none'}"
    elif ext != ".pdf" and _is_binary_file(path):
        # PDFs are intentionally binary but have a dedicated parser, so exclude them from binary check
        reason = "binary file detected"
        is_binary = True

    if reason:
        if verbose:
            print(f"[SKIP] {path} ({reason})")
        # Only log to audit if: binary file (always log) OR log_unsupported is True
        if audit_fn and (is_binary or log_unsupported):
            try:
                audit_fn("skip_file", {"path": path, "reason": reason})
            except Exception:
                # Skip audit failures to avoid interrupting discovery
                pass
        return True

    return False


# =========================
#  HELPERS: Process CLI arguments
# =========================


def parse_args(config: IngestConfig) -> argparse.Namespace:
    """Parse command-line arguments for the ingestion pipeline.

    Args:
        config: IngestConfig instance providing default values.

    Returns:
        Parsed command-line arguments namespace.

    Available Arguments:
        --reset: Delete all collections before ingesting.
        --workers: Number of concurrent threads (default from config).
        --limit: Maximum number of files to process.
        --verbose: Enable detailed console output.
        --dry-run: Simulate without writing to ChromaDB.
    """
    parser = argparse.ArgumentParser(
        description="Incremental ingestion pipeline for governance documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --reset --workers 4
  %(prog)s --limit 10 --verbose --dry-run
  %(prog)s --workers 2
        """,
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Delete the entire Chroma collection(s) before ingesting",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=config.max_workers,
        help=f"Number of concurrent worker threads (default: {config.max_workers})",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of files to ingest (useful for testing)",
    )

    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Enable verbose logging to stdout"
    )

    parser.add_argument(
        "--include-url-seeds",
        action="store_true",
        default=False,
        help="Fetch and ingest URLs defined in the URL seed JSON file",
    )

    parser.add_argument(
        "--url-seed-path",
        type=str,
        default=None,
        help="Override path to the URL seed JSON file (default from config)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate ingestion without writing to Chroma",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Quick validation run: process sample docs with detailed timing/error analysis (implies --limit 10 --verbose)",
    )

    parser.add_argument(
        "--skip-llm-preprocess",
        action="store_true",
        default=False,
        help="Skip LLM-based text preprocessing (useful for testing when Ollama is unavailable)",
    )

    parser.add_argument(
        "--record-candidate-terms",
        action="store_true",
        default=config.enable_candidate_terms,
        help="Record candidate terms from key topics for human curation",
    )

    parser.add_argument(
        "--candidate-terms-domain",
        type=str,
        default=config.candidate_terms_domain,
        help="Optional domain to associate with recorded candidate terms",
    )

    parser.add_argument(
        "--progress-interval",
        type=int,
        default=None,
        help="Log progress every N documents (default: 10, 0 to disable)",
    )

    parser.add_argument(
        "--log-unsupported-files",
        action="store_true",
        default=False,
        help="Log files skipped due to unsupported extensions (default: False to avoid log noise)",
    )

    parser.add_argument(
        "--purge-logs",
        action="store_true",
        default=False,
        help="Purge all .log and .jsonl files for ingest, rag, and consistency_graph (disabled in Prod environment)",
    )

    parser.add_argument(
        "--bm25-indexing",
        action="store_true",
        default=None,
        help="Enable BM25 keyword indexing during ingestion (default: from BM25_INDEXING_ENABLED env var)",
    )

    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        default=False,
        help="Disable BM25 keyword indexing during ingestion (overrides --bm25-indexing and env var)",
    )

    return parser.parse_args()


# =========================
#  PIPELINE STAGE FUNCTIONS
# =========================


def stage_compute_hash(file_path: str, logger) -> Tuple[Optional[str], Optional[str]]:
    """Compute document ID and file hash.

    Args:
        file_path: Path to the document file.
        logger: Logger instance for output.

    Returns (doc_id, file_hash) or (None, None) on failure.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        doc_id = compute_doc_id(file_path)
        file_hash = compute_file_hash(file_path)
        logger.debug(f"Computed doc_id={doc_id}, hash={file_hash[:8]}...")
        return doc_id, file_hash
    except Exception as e:
        logger.error(f"Stage HASH_COMPUTE failed: {e}")
        audit(
            "stage_error",
            {
                "stage": "hash_compute",
                "path": file_path,
                "error": str(e)[:200],
                "error_type": type(e).__name__,
            },
        )
        return None, None


def stage_check_hash(
    doc_id: str,
    file_hash: str,
    file_path: str,
    chunk_collection: Collection,
    args,
    config: IngestConfig,
    dry_run_stats: Optional["DryRunStats"],
    logger,
) -> Tuple[Optional[bool], Optional[int]]:
    """Check existing hash and assign version.

    Args:
        doc_id: Document ID.
        file_hash: Hash of the file.
        file_path: Path to the document file.
        chunk_collection: Collection of document chunks.
        args: Command-line arguments.
        config: Ingest configuration.
        dry_run_stats: Optional dry run statistics.
        logger: Logger instance for output.

    Returns (is_update, version) or (None, None) on failure.
    If returns (True, None) = document unchanged, skip processing.
    """
    try:
        existing_hash = get_existing_doc_hash(doc_id, chunk_collection)

        if existing_hash == file_hash and not (args.dry_run and args.reset):
            logger.info(f"SKIP {doc_id} - unchanged")
            audit("skip", {"doc_id": doc_id, "path": file_path})
            if args.verbose:
                print(f"[SKIP]  {file_path} (unchanged)")
            if dry_run_stats:
                dry_run_stats.record_document("skipped", 0, 0.0)
            return True, None  # Signal to skip

        is_update = False
        if (
            existing_hash is not None
            and existing_hash != file_hash
            and not (args.dry_run and args.reset)
        ):
            is_update = True
            logger.info(f"UPDATE {doc_id} — hash changed")
            audit(
                "update",
                {
                    "doc_id": doc_id,
                    "path": file_path,
                    "old_hash": existing_hash,
                    "new_hash": file_hash,
                },
            )
            print(f"[UPDATE] {file_path} (hash changed, re-ingesting)")

            with config.version_lock:

                @retry_chromadb_call(
                    max_retries=5, initial_delay=0.5, operation_name=f"get_versions({doc_id})"
                )
                def _get_existing_versions():
                    return chunk_collection.get(where={"doc_id": doc_id}, include=["metadatas"])

                existing_versions = _get_existing_versions()
                if existing_versions["metadatas"]:
                    version = max(m.get("version", 1) for m in existing_versions["metadatas"]) + 1
                else:
                    logger.warning(
                        f"No metadata found for existing doc {doc_id}, treating as version 1"
                    )
                    version = 1
        else:
            logger.info(f"NEW {doc_id}")
            audit("new", {"doc_id": doc_id, "path": file_path})
            version = 1
            print(f"[NEW]  {file_path}")

        return is_update, version
    except Exception as e:
        logger.error(f"Stage VERSION_ASSIGN failed for {doc_id}: {e}")
        audit(
            "stage_error",
            {
                "stage": "version_assign",
                "doc_id": doc_id,
                "path": file_path,
                "error": str(e)[:200],
                "error_type": type(e).__name__,
            },
        )
        return None, None


def stage_prune_versions(
    doc_id: str,
    version: int,
    is_update: bool,
    chunk_collection: Collection,
    doc_collection: Collection,
    config: IngestConfig,
    logger,
) -> bool:
    """Stage 3: Prune old versions (non-fatal on failure).

    Args:
    - doc_id: Document ID.
    - version: Current version number assigned to this document.
    - is_update: Whether this document is an update (True) or new (False).
    - chunk_collection: ChromaDB collection for document chunks.
    - doc_collection: ChromaDB collection for documents.
    - config: Ingest configuration.
    - logger: Logger instance for output.

    Returns True if successful or no pruning needed, False if there's an error.
    """
    try:
        if not is_update or version <= config.versions_to_keep:
            return True

        prune_version = version - config.versions_to_keep
        logger.info(f"Pruning versions <= {prune_version} for {doc_id}")

        prune_filter = {"$and": [{"doc_id": doc_id}, {"version": {"$lte": prune_version}}]}

        @retry_chromadb_call(
            max_retries=5, initial_delay=0.5, operation_name=f"get_chunks_to_prune({doc_id})"
        )
        def _get_chunks_to_delete():
            return chunk_collection.get(where=prune_filter, include=[])

        chunks_to_delete = _get_chunks_to_delete()
        chunk_count = len(chunks_to_delete.get("ids", []))

        @retry_chromadb_call(
            max_retries=5, initial_delay=0.5, operation_name=f"get_docs_to_prune({doc_id})"
        )
        def _get_docs_to_delete():
            return doc_collection.get(where=prune_filter, include=[])

        docs_to_delete = _get_docs_to_delete()
        doc_count = len(docs_to_delete.get("ids", []))

        if chunk_count == 0 and doc_count == 0:
            logger.debug(f"No old versions found for {doc_id}")
            audit(
                "prune_old_versions",
                {
                    "doc_id": doc_id,
                    "prune_version": prune_version,
                    "new_version": version,
                    "chunks_deleted": 0,
                    "docs_deleted": 0,
                    "status": "no_old_versions",
                },
            )
            return True

        chunk_delete_success = False
        doc_delete_success = False

        try:

            @retry_chromadb_call(
                max_retries=5, initial_delay=0.5, operation_name=f"delete_chunks_prune({doc_id})"
            )
            def _delete_chunks():
                chunk_collection.delete(where=prune_filter)

            _delete_chunks()
            chunk_delete_success = True
            logger.debug(f"Deleted {chunk_count} chunks for {doc_id}")
        except Exception as e:
            logger.error(f"Failed to delete {chunk_count} chunks for {doc_id}: {e}")

        try:

            @retry_chromadb_call(
                max_retries=5, initial_delay=0.5, operation_name=f"delete_docs_prune({doc_id})"
            )
            def _delete_docs():
                doc_collection.delete(where=prune_filter)

            _delete_docs()
            doc_delete_success = True
            logger.debug(f"Deleted {doc_count} documents for {doc_id}")
        except Exception as e:
            logger.error(f"Failed to delete {doc_count} documents for {doc_id}: {e}")

        if chunk_delete_success and doc_delete_success:
            logger.info(f"Successfully pruned {chunk_count} chunks + {doc_count} docs for {doc_id}")
            audit(
                "prune_old_versions",
                {
                    "doc_id": doc_id,
                    "prune_version": prune_version,
                    "new_version": version,
                    "chunks_deleted": chunk_count,
                    "docs_deleted": doc_count,
                    "status": "success",
                },
            )
        else:
            warning_msg = []
            if not chunk_delete_success:
                warning_msg.append(f"chunks({chunk_count})")
            if not doc_delete_success:
                warning_msg.append(f"docs({doc_count})")
            logger.warning(
                f"Partial prune failure for {doc_id}: {', '.join(warning_msg)} not deleted"
            )
            audit(
                "prune_old_versions_partial",
                {
                    "doc_id": doc_id,
                    "prune_version": prune_version,
                    "new_version": version,
                    "chunks_deleted": chunk_count if chunk_delete_success else 0,
                    "docs_deleted": doc_count if doc_delete_success else 0,
                    "chunks_delete_failed": not chunk_delete_success,
                    "docs_delete_failed": not doc_delete_success,
                },
            )
        return True
    except Exception as e:
        logger.error(f"Stage PRUNE_VERSIONS failed for {doc_id}: {e}")
        audit(
            "stage_error",
            {
                "stage": "prune_versions",
                "doc_id": doc_id,
                "error": str(e)[:200],
                "error_type": type(e).__name__,
            },
        )
        return True  # Non-fatal


def stage_extract_text(file_path: str, logger) -> Optional[str]:
    """Extract text from HTML or PDF.

    Args:
        file_path: Path to the document file.
        logger: Logger instance for output.

    Returns raw text or None on failure.
    """
    try:
        # Determine file type and use appropriate parser
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            raw_text = extract_text_from_pdf(file_path)
            logger.debug(f"[Extracted PDF Text] {raw_text[:500]}...")
        else:
            raw_text = extract_text_from_html(file_path)
            logger.debug(f"[Extracted HTML Text] {raw_text[:500]}...")
        return raw_text
    except Exception as e:
        logger.error(f"Stage EXTRACT_TEXT failed for {file_path}: {e}")
        audit(
            "stage_error",
            {
                "stage": "extract_text",
                "path": file_path,
                "error": str(e)[:200],
                "error_type": type(e).__name__,
            },
        )
        return None


def stage_determine_category(
    file_path: str, source_category_override: Optional[str], logger
) -> Optional[str]:
    """Stage 5: Determine source category from path or override.

    Args:
        file_path: Path to the document file.
        source_category_override: Optional override for source category.
        logger: Logger instance for output.

    Returns category string or None.
    """
    try:
        if source_category_override:
            return source_category_override

        file_path_obj = Path(file_path)
        parent_name = file_path_obj.parent.name
        source_category = parent_name if parent_name not in ["", "downloads", "data_raw"] else None

        if source_category:
            logger.info(f"Source category detected: {source_category}")
            audit("source_category_detected", {"path": file_path, "category": source_category})

        return source_category
    except Exception as e:
        logger.error(f"Stage SOURCE_CATEGORY failed: {e}")
        audit(
            "stage_error",
            {
                "stage": "source_category",
                "path": file_path,
                "error": str(e)[:200],
                "error_type": type(e).__name__,
            },
        )
        return None


def stage_preprocess_text(
    raw_text: str,
    source_category: Optional[str],
    file_hash: str,
    llm_cache: Optional[LLMCache],
    args,
    logger,
) -> Optional[Dict[str, Any]]:
    """Preprocess text with LLM.

    Returns preprocessed text dict or None on failure.
    """
    try:
        if args.verbose:
            print("Preprocessing Text")

        # If skip-llm-preprocess flag is set, return minimal preprocessing
        if args.skip_llm_preprocess:
            logger.info("Skipping LLM preprocessing (--skip-llm-preprocess flag set)")
            redacted_text, dlp_counts = redact_sensitive_text(raw_text, file_hash)
            return {
                "cleaned_text": redacted_text,
                "doc_type": source_category or "general",
                "key_topics": [],
                "summary": (
                    redacted_text[:200] + "..." if len(redacted_text) > 200 else redacted_text
                ),
                "summary_scores": {"overall": 0},
                "source_category": source_category,
                "dlp_pattern_counts": dlp_counts,
            }

        preprocessed = preprocess_text(
            raw_text, source_category=source_category, doc_hash=file_hash, llm_cache=llm_cache
        )
        logger.debug(f"[PreProcessed Text] {preprocessed}")
        return preprocessed
    except Exception as e:
        logger.error(f"Stage PREPROCESS_TEXT failed: {e}")
        audit(
            "stage_error",
            {"stage": "preprocess_text", "error": str(e)[:200], "error_type": type(e).__name__},
        )
        return None


def stage_record_candidate_terms(
    preprocessed_text: Dict[str, Any],
    doc_id: str,
    args,
    config,
    logger,
    extra_terms: Optional[List[str]] = None,
) -> None:
    """Stage: Record candidate terms for human curation.

    Args:
    - preprocessed_text: Dict containing preprocessed text and metadata, including key topics.
    - doc_id: Document ID to associate with candidate terms.
    - args: Command-line arguments.
    - config: Ingest configuration for defaults.
    - logger: Logger instance for output.
    - extra_terms: Optional list of additional terms to record beyond key topics.

    """
    if getattr(args, "dry_run", False):
        return
    if not getattr(args, "record_candidate_terms", False):
        return

    key_topics = preprocessed_text.get("key_topics") or []
    if not key_topics and not extra_terms:
        return

    if DomainType is None or get_domain_term_manager is None or resolve_domain_type is None:
        logger.debug("Domain term manager not available, skipping candidate term recording")
        return

    domain_value = getattr(args, "candidate_terms_domain", None) or getattr(
        config, "candidate_terms_domain", None
    )
    domain = None
    if domain_value:
        resolved_value, display_name = resolve_domain_type(domain_value)
        if resolved_value:
            try:
                domain = DomainType(resolved_value)
            except ValueError:
                logger.warning(f"Invalid candidate terms domain: {domain_value}")
                domain = DomainType.CUSTOM
        else:
            domain = DomainType.CUSTOM

    manager = get_domain_term_manager()
    summary = preprocessed_text.get("summary")

    terms = {t.strip() for t in key_topics if isinstance(t, str) and t.strip()}
    if extra_terms:
        terms.update(t.strip() for t in extra_terms if isinstance(t, str) and t.strip())

    for topic in terms:
        try:
            manager.record_candidate_term(
                topic,
                domain=domain,
                source_doc_id=doc_id,
                context=summary,
            )
        except Exception as e:
            logger.warning(f"Failed to record candidate term '{topic}': {e}")


def stage_chunk_text(
    preprocessed_text: Dict[str, Any],
    args,
    logger,
    config: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Chunk text.

    Args:
    - preprocessed_text: Dict containing preprocessed text and metadata, including key topics.
    - args: Command-line arguments.
    - logger: Logger instance for output.
    - config: Ingest configuration for defaults.

    Returns dict with chunks, full_text, and optionally parent_chunks, or None on failure.
    """
    try:
        if args.verbose:
            print("Chunking Text")

        doc_type = preprocessed_text.get("doc_type")
        full_text = preprocessed_text["cleaned_text"]

        # Default containers to avoid NameError when parent-child chunking is disabled
        child_chunks: List[Dict[str, Any]] = []

        # Use parent-child chunking if enabled
        parent_chunks = None
        parent_child_enabled = bool(getattr(config, "enable_parent_child_chunking", False))

        if parent_child_enabled:
            try:
                # Create parent-child chunk pairs
                child_chunks, parent_chunks = create_parent_child_chunks(
                    text=full_text, doc_type=doc_type, parent_size=1200, child_size=400
                )
                # Extract just the text from child chunks for storage
                chunks = [c["text"] for c in child_chunks]

                if args.verbose:
                    print(
                        f"Created {len(chunks)} child chunks and {len(parent_chunks)} parent chunks (doc_type: {doc_type})"
                    )
            except Exception as e:
                logger.warning(
                    f"Parent-child chunking unavailable, falling back to standard chunking: {e}"
                )
                parent_chunks = None
                chunks = chunk_text(full_text, doc_type=doc_type, adaptive=True)
                child_chunks = []
        else:
            # Use regular chunking
            chunks = chunk_text(full_text, doc_type=doc_type, adaptive=True)

            if args.verbose:
                print(f"Created {len(chunks)} chunks (doc_type: {doc_type})")
                print(chunks)

        logger.debug(f"[Chunks] {chunks}")
        return {
            "chunks": chunks,
            "full_text": full_text,
            "parent_chunks": parent_chunks,
            "child_chunk_data": child_chunks if parent_chunks else None,
        }
    except Exception as e:
        logger.error(f"Stage CHUNK_TEXT failed: {e}")
        audit(
            "stage_error",
            {"stage": "chunk_text", "error": str(e)[:200], "error_type": type(e).__name__},
        )
        return None


def stage_chunk_idempotency(
    chunk_data: Dict[str, Any],
    doc_id: str,
    version: int,
    is_update: bool,
    chunk_collection: Collection,
    logger,
) -> Tuple[List[str], List[str], str]:
    """Check chunk-level idempotency and deduplication.

    Args:
    - chunk_data: Dict containing chunked text and metadata.
    - doc_id: Document ID.
    - version: Document version.
    - is_update: Whether this is an update operation.
    - chunk_collection: Collection to check for existing chunks.
    - logger: Logger instance for output.

    Returns (chunks_to_process, chunk_hashes, full_text) where chunks_to_process excludes unchanged chunks.
    """
    try:
        chunks = chunk_data["chunks"]
        full_text = chunk_data["full_text"]
        chunk_hashes = [compute_chunk_hash(chunk) for chunk in chunks]
        chunks_to_process = chunks

        if is_update and version > 1:

            @retry_chromadb_call(
                max_retries=5, initial_delay=0.5, operation_name=f"get_existing_chunks({doc_id})"
            )
            def _get_existing_chunks():
                return chunk_collection.get(
                    where={"doc_id": doc_id}, include=["documents", "metadatas"]
                )

            existing_chunks = _get_existing_chunks()

            existing_chunk_hashes = {}
            for metadata in existing_chunks.get("metadatas", []):
                if metadata and "chunk_text_hash" in metadata:
                    h = metadata["chunk_text_hash"]
                    existing_chunk_hashes[h] = existing_chunk_hashes.get(h, 0) + 1

            new_chunks = []
            new_hashes = []
            unchanged_count = 0

            for chunk, chunk_hash in zip(chunks, chunk_hashes):
                if chunk_hash in existing_chunk_hashes and existing_chunk_hashes[chunk_hash] > 0:
                    existing_chunk_hashes[chunk_hash] -= 1
                    unchanged_count += 1
                    logger.debug(f"Skipping unchanged chunk (hash: {chunk_hash[:8]}...)")
                else:
                    new_chunks.append(chunk)
                    new_hashes.append(chunk_hash)

            if unchanged_count > 0:
                logger.info(
                    f"Chunk idempotency: {unchanged_count}/{len(chunks)} unchanged, {len(new_chunks)} new/changed"
                )
                audit(
                    "chunk_idempotency",
                    {
                        "doc_id": doc_id,
                        "version": version,
                        "unchanged_chunks": unchanged_count,
                        "new_changed_chunks": len(new_chunks),
                        "total_chunks": len(chunks),
                    },
                )
                chunks_to_process = new_chunks
                chunk_hashes = new_hashes

        return chunks_to_process, chunk_hashes, full_text
    except Exception as e:
        logger.error(f"Stage CHUNK_IDEMPOTENCY failed: {e}")
        audit(
            "stage_error",
            {"stage": "chunk_idempotency", "error": str(e)[:200], "error_type": type(e).__name__},
        )
        return chunks, chunk_hashes, chunk_data.get("full_text", "")  # Return originals on error


def stage_store_chunks(
    doc_id: str,
    file_hash: str,
    file_path: str,
    version: int,
    chunks: List[str],
    chunk_hashes: List[str],
    full_text: str,
    preprocessed_text: Dict[str, Any],
    chunk_collection: Collection,
    doc_collection: Collection,
    preprocess_duration: float,
    ingest_duration: float,
    is_update: bool,
    config: IngestConfig,
    args,
    logger,
    llm_cache: Optional[LLMCache],
    embedding_cache: Optional[Any],
    parent_chunks: Optional[List[Dict[str, Any]]] = None,
    child_chunks: Optional[List[Dict[str, str]]] = None,
) -> bool:
    """Store chunks in ChromaDB.

    Args:
    - doc_id: Document ID.
    - file_hash: Hash of the source file.
    - file_path: Path to the source file.
    - version: Document version.
    - chunks: List of chunk texts.
    - chunk_hashes: List of chunk hashes.
    - full_text: Full text of the document.
    - preprocessed_text: Dict containing preprocessed text and metadata.
    - chunk_collection: Collection to store chunks.
    - doc_collection: Collection to store document metadata.
    - preprocess_duration: Duration of the preprocessing stage.
    - ingest_duration: Duration of the ingest stage.
    - is_update: Whether this is an update operation.
    - config: Ingest configuration.
    - args: Command-line arguments.
    - logger: Logger instance for output.
    - llm_cache: Optional LLM cache.
    - embedding_cache: Optional embedding cache.
    - parent_chunks: Optional list of parent chunks.
    - child_chunks: Optional list of child chunks.

    Returns True on success, False on failure.
    """
    try:
        # Check if parent-child chunking is active
        using_parent_child_chunking = (
            parent_chunks is not None
            and hasattr(config, "enable_parent_child_chunking")
            and config.enable_parent_child_chunking
        )

        if args.verbose:
            if using_parent_child_chunking:
                print(
                    f"Storing {len(parent_chunks)} parent chunks and {len(child_chunks) if child_chunks else 0} child chunks (parent-child mode)"
                )
            else:
                print("Storing chunks in Chroma")

        # Store chunks in ChromaDB
        # When using parent-child chunking, pass empty chunks list to avoid duplication
        # (child chunks are stored separately below with embeddings and parent links)
        chunks_to_store = [] if using_parent_child_chunking else chunks
        chunk_hashes_to_store = [] if using_parent_child_chunking else chunk_hashes

        store_chunks_in_chroma(
            doc_id=doc_id,
            file_hash=file_hash,
            source_path=file_path,
            version=version,
            chunks=chunks_to_store,
            chunk_hashes=chunk_hashes_to_store,
            full_text=full_text,
            metadata=preprocessed_text,
            chunk_collection=chunk_collection,
            doc_collection=doc_collection,
            preprocess_duration=preprocess_duration,
            ingest_duration=ingest_duration,
            dry_run=args.dry_run,
            llm_cache=llm_cache,
            enable_drift_detection=config.enable_semantic_drift_detection and is_update,
            enable_chunk_heuristic=config.enable_chunk_heuristic_skip,
            embedding_cache=embedding_cache,
            embedding_batch_size=config.embedding_batch_size,
            preserve_domain_keywords=config.preserve_domain_keywords,
        )

        # Store parent and child chunks if using parent-child chunking
        if using_parent_child_chunking:

            filename = os.path.basename(file_path)
            base_metadata = {
                "doc_id": doc_id,
                "source": file_path,
                "filename": filename,
                "version": version,
                "hash": file_hash,
                "doc_type": preprocessed_text.get("doc_type", "unknown"),
                "embedding_model": EMBEDDING_MODEL_NAME,
            }

            # Store child chunks FIRST with real embeddings to establish collection dimension
            # This ensures the collection schema uses the natural embedding dimension (768)
            # rather than being locked to 384 by dummy parent chunk embeddings
            if child_chunks:
                try:
                    store_child_chunks(
                        doc_id=doc_id,
                        child_chunks=child_chunks,
                        chunk_collection=chunk_collection,
                        base_metadata=base_metadata,
                        dry_run=args.dry_run,
                        full_text=full_text,
                        doc_type=preprocessed_text.get("doc_type", "unknown"),
                    )
                except Exception as child_err:
                    logger.error(f"Failed to store child chunks for {doc_id}: {child_err}")
                    audit(
                        "child_chunks_storage_failed",
                        {
                            "doc_id": doc_id,
                            "error": str(child_err)[:200],
                            "error_type": type(child_err).__name__,
                        },
                    )
                    # Re-raise child error since these are the searchable chunks
                    raise

            # Then store parent chunks (with error handling to not block if it fails)
            try:
                store_parent_chunks(
                    doc_id=doc_id,
                    parent_chunks=parent_chunks,
                    chunk_collection=chunk_collection,
                    base_metadata=base_metadata,
                    dry_run=args.dry_run,
                    full_text=full_text,
                    doc_type=preprocessed_text.get("doc_type", "unknown"),
                )
            except Exception as parent_err:
                logger.error(f"Failed to store parent chunks for {doc_id}: {parent_err}")
                audit(
                    "parent_chunks_storage_failed",
                    {
                        "doc_id": doc_id,
                        "error": str(parent_err)[:200],
                        "error_type": type(parent_err).__name__,
                    },
                )
        logger.debug("Storage phase completed")
        return True
    except Exception as e:
        logger.error(f"Stage STORE_CHUNKS failed: {e}")
        audit(
            "stage_error",
            {
                "stage": "store_chunks",
                "doc_id": doc_id,
                "path": file_path,
                "error": str(e)[:200],
                "error_type": type(e).__name__,
            },
        )
        return False


# =========================
#  HELPERS: DOCUMENT ID + HASH
# =========================


# =========================
#  URL SEED INGEST HELPERS
# =========================


def get_auth_headers(url: str) -> Dict[str, str]:
    """Placeholder for auth headers for protected URLs.
    Replace with actual token/cookie retrieval when needed.
    """
    # TODO: Implement auth (e.g., bearer token, cookies, basic auth)
    return {}


def load_url_seeds(path: str, logger) -> List[Dict[str, Any]]:
    """Load URL seed definitions from JSON file.
    Expected schema: list of objects with keys
    - url (str)
    - single_page (bool)
    - sidebar_selector (str, optional)
    - max_depth (int, optional)
    - source_category (str, optional)
    """
    if not os.path.exists(path):
        logger.warning(f"URL seed file not found: {path}")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error("URL seed file must be a list of seed objects")
            return []
        return data
    except Exception as e:
        logger.error(f"Failed to load URL seed file {path}: {e}")
        return []


def extract_sidebar_links(
    html: str, base_url: str, selector: Optional[str], max_depth: int
) -> List[str]:
    """Extract links from a sidebar navigation section.

    selector: id or class name of the sidebar root element.
    max_depth: max nesting depth of links inside the sidebar tree.
    """
    if not selector:
        return []

    soup = BeautifulSoup(html, "html.parser")
    nav = soup.find(id=selector) or soup.find(class_=selector)
    if not nav:
        return []

    links: List[str] = []
    for a in nav.find_all("a", href=True):
        depth = 0
        parent = a.parent
        while parent and parent is not nav:
            if parent.name in ("ul", "ol", "li", "div"):
                depth += 1
            parent = parent.parent
        if depth <= max_depth:
            links.append(urljoin(base_url, a["href"]))
    return links


def download_url_to_file(
    url: str, config: IngestConfig, logger
) -> Tuple[Optional[str], Optional[str]]:
    """Download a URL to the url_imports directory and return (path, html).
    Returns (None, None) on failure.
    """
    try:
        headers = get_auth_headers(url)
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None, None

    os.makedirs(config.url_download_dir, exist_ok=True)
    filename = f"url_{hashlib.sha256(url.encode('utf-8')).hexdigest()}.html"
    out_path = os.path.join(config.url_download_dir, filename)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        logger.error(f"Failed to write downloaded HTML for {url}: {e}")
        return None, None

    return out_path, html


def collect_url_files_from_seeds(config: IngestConfig, logger) -> List[Tuple[str, Optional[str]]]:
    seeds = load_url_seeds(config.url_seed_json_path, logger)
    if not seeds:
        return []

    downloaded: List[Tuple[str, Optional[str]]] = []
    seen_urls = set()

    for seed in seeds:
        url = seed.get("url")
        if not url:
            logger.warning("Skipping URL seed with no 'url' field")
            continue

        single_page = bool(seed.get("single_page", False))
        sidebar_selector = seed.get("sidebar_selector")
        max_depth = int(seed.get("max_depth", 1))
        source_category = seed.get("source_category")

        if url in seen_urls:
            continue

        main_path, main_html = download_url_to_file(url, config, logger)
        if main_path:
            downloaded.append((main_path, source_category))
            seen_urls.add(url)
        if not main_html:
            continue

        if single_page:
            continue

        nav_links = extract_sidebar_links(main_html, url, sidebar_selector, max_depth)
        for link in nav_links:
            if link in seen_urls:
                continue
            # Only follow links on same domain to avoid unintended crawls
            if urlparse(link).netloc != urlparse(url).netloc:
                continue
            path, _ = download_url_to_file(link, config, logger)
            if path:
                downloaded.append((path, source_category))
                seen_urls.add(link)

    return downloaded


def process_file(
    file_path: str,
    chunk_collection: Collection,
    doc_collection: Collection,
    config: IngestConfig,
    source_category_override: Optional[str] = None,
    llm_cache: Optional[LLMCache] = None,
    embedding_cache: Optional[Any] = None,
    dry_run_stats: Optional["DryRunStats"] = None,
    profile_stats: Optional["ProfileStats"] = None,
) -> bool:
    """Orchestrate the complete ingestion workflow for a single document.

    Coordinates execution of 10 pipeline stages:
    1. Compute document ID and file hash
    2. Check existing hash and assign version
    3. Prune old versions
    4. Extract text from HTML
    5. Determine source category
    6. Preprocess text with LLM
    7. Chunk text
    8. Check chunk-level idempotency
    9. Store chunks in ChromaDB
    10. Record completion metrics

    Args:
        file_path: Absolute path to HTML file to process.
        chunk_collection: ChromaDB collection for chunks.
        doc_collection: ChromaDB collection for documents.
        config: Configuration instance.
        source_category_override: Optional override for source category (e.g., from URL seed).
        llm_cache: Optional cache for LLM calls to speed up repeated runs.
        embedding_cache: Optional cache for embeddings to speed up repeated runs.
        dry_run_stats: Optional object to record dry run statistics.
        profile_stats: Optional object to record profiling statistics.

    Returns:
        True if successful, False if any fatal stage failed.
    """
    logger = config.logger
    args = config.args
    start_time = time.perf_counter()
    doc_id = None
    version = None
    is_update = False

    try:
        logger.info(f"START {file_path}")
        audit("start_file", {"path": file_path})
        if args.verbose:
            print(f"[START] {file_path}")

        # Stage 1: Compute hash
        doc_id, file_hash = stage_compute_hash(file_path, logger)
        if doc_id is None or file_hash is None:
            return False

        # Stage 2: Check hash and assign version
        result = stage_check_hash(
            doc_id, file_hash, file_path, chunk_collection, args, config, dry_run_stats, logger
        )
        if result == (True, None):  # Skip signal
            return True
        is_update, version = result
        if is_update is None or version is None:
            return False

        # Stage 3: Prune old versions (non-fatal)
        stage_prune_versions(
            doc_id, version, is_update, chunk_collection, doc_collection, config, logger
        )

        # Stage 4: Extract text
        start_parse_time = time.perf_counter()
        raw_text = stage_extract_text(file_path, logger)
        parse_duration = time.perf_counter() - start_parse_time
        if raw_text is None:
            return False

        # Stage 5: Determine category
        source_category = stage_determine_category(file_path, source_category_override, logger)

        start_preprocess_time = time.perf_counter()

        # Stage 6: Preprocess text
        preprocessed_text = stage_preprocess_text(
            raw_text, source_category, file_hash, llm_cache, args, logger
        )
        if preprocessed_text is None:
            return False

        preprocess_duration = time.perf_counter() - start_preprocess_time

        # Stage 7: Chunk text
        start_chunk_time = time.perf_counter()
        chunk_data = stage_chunk_text(preprocessed_text, args, logger, config)
        chunk_duration = time.perf_counter() - start_chunk_time
        if chunk_data is None:
            return False

        extra_terms: Optional[List[str]] = None
        if getattr(args, "record_candidate_terms", False):
            try:
                extra_terms = extract_technical_entities(chunk_data.get("full_text", ""))
            except Exception as e:
                logger.warning(f"Failed to extract technical entities for candidate terms: {e}")

        stage_record_candidate_terms(
            preprocessed_text,
            doc_id,
            args,
            config,
            logger,
            extra_terms=extra_terms,
        )

        # Stage 8: Check chunk idempotency
        start_idempotency_time = time.perf_counter()
        chunks_to_process, chunk_hashes, full_text = stage_chunk_idempotency(
            chunk_data, doc_id, version, is_update, chunk_collection, logger
        )
        idempotency_duration = time.perf_counter() - start_idempotency_time

        # Stage 9: Store chunks
        start_storage_time = time.perf_counter()
        if not stage_store_chunks(
            doc_id,
            file_hash,
            file_path,
            version,
            chunks_to_process,
            chunk_hashes,
            full_text,
            preprocessed_text,
            chunk_collection,
            doc_collection,
            preprocess_duration,
            time.perf_counter() - start_time,
            is_update,
            config,
            args,
            logger,
            llm_cache,
            embedding_cache,
            chunk_data.get("parent_chunks"),
            chunk_data.get("child_chunk_data"),
        ):
            return False
        storage_duration = time.perf_counter() - start_storage_time

        # Stage 10: BM25 Keyword Indexing (optional)
        # Uses common indexing utility for chunk-level granularity
        # TODO: move db_factory import to top of page
        start_bm25_time = time.perf_counter()
        if config.bm25_indexing_enabled and not args.dry_run:
            try:
                cache_db = get_cache_client(enable_cache=True)
                total_indexed = index_chunks_in_bm25(
                    doc_id=doc_id,
                    chunks=chunks_to_process,
                    child_chunks=chunk_data.get("child_chunk_data"),
                    parent_chunks=chunk_data.get("parent_chunks"),
                    config=config,
                    cache_db=cache_db,
                    logger=logger,
                )

                bm25_duration = time.perf_counter() - start_bm25_time
                audit(
                    "bm25_indexed",
                    {
                        "doc_id": doc_id,
                        "version": version,
                        "chunks_indexed": total_indexed,
                        "granularity": "chunk",
                        "duration_seconds": bm25_duration,
                    },
                )
            except Exception as e:
                # Non-fatal: BM25 indexing failure shouldn't block ingestion
                logger.warning(f"BM25 indexing failed for {doc_id}: {e}")
                audit(
                    "bm25_index_failed",
                    {"doc_id": doc_id, "error": str(e)[:200], "error_type": type(e).__name__},
                )

        # Stage 11: Record completion with all phase timings
        duration = time.perf_counter() - start_time
        logger.info(f"DONE {file_path} in {duration:.2f}s")
        audit(
            "done_file",
            {
                "path": file_path,
                "doc_id": doc_id,
                "duration_seconds": duration,
                "parse_seconds": parse_duration,
                "preprocess_seconds": preprocess_duration,
                "chunk_seconds": chunk_duration,
                "idempotency_seconds": idempotency_duration,
                "storage_seconds": storage_duration,
                "source_category": source_category,
            },
        )

        if args.dry_run and dry_run_stats:
            status = "new" if version == 1 else "updated"
            dry_run_stats.record_document(status, len(chunk_data["chunks"]), duration)

        if profile_stats:
            profile_stats.record_document(len(chunk_data["chunks"]), duration)

        if args.verbose:
            print(f"[DONE] {file_path}")

        return True

    except Exception as e:
        """Catch all exceptions to allow batch processing to continue."""
        duration = time.perf_counter() - start_time
        logger.error(f"ERROR {file_path}: {e}", exc_info=True)
        audit(
            "error",
            {
                "path": file_path,
                "doc_id": doc_id or "unknown",
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_seconds": duration,
            },
        )
        if profile_stats:
            profile_stats.record_error(str(e), file_path)
        print(f"[ERROR] {file_path}: {e}")
        traceback.print_exc(file=sys.stdout)
        return False


def main() -> None:
    """Main entry point for the document ingestion pipeline.

    Coordinates the complete ingestion process:
    1. Initialises configuration and logging
    2. Parses command-line arguments
    3. Sets up ChromaDB client and collections
    4. Discovers HTML files in source directory
    5. Processes files in parallel using thread pool
    6. Reports completion statistics

    The function handles collection reset, dry-run mode, file limiting,
    and graceful error handling for individual document failures.

    Command-line Arguments:
        See parse_args() for available options.

    Environment Variables:
        See IngestConfig for configuration options.

    Exit Codes:
        Does not explicitly exit - returns normally on completion.
        Individual file errors are caught and logged.

    Examples:
        Reset and process all documents:
            $ python ingest.py --reset --workers 4

        Test with limited files:
            $ python ingest.py --limit 5 --verbose --dry-run
    """
    # Get singleton configuration from ingest_config
    config = get_ingest_config()

    start_time = time.perf_counter()
    args = parse_args(config)

    # Rebuild config with CLI overrides (priority: CLI > system env > .env > defaults)
    try:
        overrides = build_cli_overrides(args)
        if overrides:
            config = get_ingest_config(overrides=overrides, reset=True)
    except Exception:
        # Fall back to existing config if overrides fail
        pass

    config.args = args

    # Apply command-line BM25 arguments to config
    # Priority: --skip-bm25 > --bm25-indexing > environment variable
    if config.args.skip_bm25:
        config.bm25_indexing_enabled = False
    elif config.args.bm25_indexing:
        config.bm25_indexing_enabled = True
    # Otherwise use config value from environment variable

    # Handle log purging BEFORE logger initialisation
    # This ensures we purge the old audit log before we write to a new one
    purge_logs_performed = False
    if config.args.purge_logs:
        if config.environment == "Prod":
            print("\n[ERROR] Log purging is disabled in Production environment for safety.")
            print("        Current environment: Prod")
            print("        To purge logs, set ENVIRONMENT=Dev or ENVIRONMENT=Test\n")
            sys.exit(1)
        else:
            # Purge ingest logs ONLY (not rag or consistency_graph logs)
            logs_dir = Path(__file__).parent.parent.parent / "logs"
            if logs_dir.exists():
                ingest_logs = [
                    "ingest.log",
                    "ingest_audit.jsonl",
                ]

                purged_count = 0
                print(f"\n[PURGE LOGS] Environment: {config.environment}")
                for log_name in ingest_logs:
                    log_file = logs_dir / log_name
                    if log_file.exists():
                        try:
                            log_file.unlink()
                            purged_count += 1
                            print(f"  ✓ Removed: {log_file}")
                        except Exception as e:
                            print(f"  ✗ Failed to remove {log_file}: {e}")
                    else:
                        print(f"  - Not found: {log_name}")

                print(f"[PURGE LOGS] Removed {purged_count} ingest log file(s)\n")
                purge_logs_performed = True

                # Clear the logger cache so get_logger() creates fresh handlers
                _loggers.pop("ingest", None)

    # Initialise logger AFTER purging so we don't write to a file we're about to delete
    config.logger = get_logger()
    config.include_url_seeds = config.args.include_url_seeds
    if config.args.url_seed_path:
        config.url_seed_json_path = config.args.url_seed_path
    config.max_workers = config.args.workers
    config.version_lock = threading.Lock()

    # Initialise monitoring infrastructure
    init_monitoring()
    perf_metrics = get_perf_metrics()
    metrics_collector = get_metrics_collector()

    # Get logger reference
    logger = config.logger
    args = config.args

    # Log the purge event as the FIRST audit entry if logs were purged
    if purge_logs_performed:
        audit(
            "purge_logs",
            {
                "environment": config.environment,
                "module": "ingest",
                "user": getpass.getuser(),
                "files_purged": ["ingest.log", "ingest_audit.jsonl"],
            },
        )

    # Handle profile mode auto-configuration
    if args.profile:
        # Profile mode implies verbose and a small sample
        args.verbose = True
        if not args.limit:
            args.limit = 10  # Default sample size for profile
        logger.info("Profile mode enabled: verbose output, sample limit set to 10")
        print("\n[PROFILE MODE] Running quick validation with detailed timing analysis...\n")

    # Override progress log interval from CLI if provided
    if config.args.progress_interval is not None:
        config.progress_log_interval = config.args.progress_interval

    # Initialise LLM cache
    llm_cache = LLMCache(
        cache_path=config.llm_cache_path,
        enabled=config.llm_cache_enabled,
        max_age_days=config.llm_cache_max_age_days,
    )
    cache_stats = llm_cache.stats()
    logger.info(
        f"LLM cache initialised: {cache_stats['total_entries']} entries, {cache_stats['expired_entries']} expired"
    )

    # Initialise embedding cache
    embedding_cache = EmbeddingCache(
        cache_path=config.embedding_cache_path, enabled=config.embedding_cache_enabled
    )
    emb_stats = embedding_cache.stats()
    logger.info(
        f"Embedding cache initialised: {emb_stats['total_entries']} entries, hit rate: {emb_stats['hit_rate']*100:.1f}%"
    )

    # Initialise rate limiter for LLM calls
    init_rate_limiter(config.llm_rate_limit)
    logger.info(f"Rate limiter initialised: {config.llm_rate_limit} requests/second")

    # Check if Ollama is available (unless LLM preprocessing is skipped)
    ollama_available = check_ollama_availability(logger, skip_llm=args.skip_llm_preprocess)
    if not ollama_available and not args.skip_llm_preprocess:
        logger.warning("WARNING: Proceeding without Ollama - LLM preprocessing will fail")
        logger.warning("         Either start Ollama or use --skip-llm-preprocess flag")

    logger.info(f"[START MAIN] begin ingestion")
    # Log config and args at the very start, before file discovery
    # This ensures parameters are captured even if file discovery fails
    audit(
        "start_main",
        {
            "config": config.to_dict(),
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        },
    )

    # Clean Chroma store before ingesting, don't do this if doing incremental loads
    # Preference to use --reset parameter instead.
    # Only set REINITIALISE_CHROMA_STORAGE env var as a last resort
    if config.reinitialise_chroma_storage:
        logger.warning("REINITIALISE_CHROMA_STORAGE is True - deleting entire Chroma storage")
        audit("reinitialise_storage", {})
        shutil.rmtree(config.rag_data_path, ignore_errors=True)
        # Clear LLM cache as well
        llm_cache.clear()
        logger.info("LLM cache cleared due to REINITIALISE_CHROMA_STORAGE")

    # Initialise the vector store path using selected backend
    chroma_path = get_default_vector_path(Path(config.rag_data_path), USING_SQLITE)
    client = PersistentClient(path=chroma_path)

    if args.verbose:
        print(f"Begin ingestion")

    if args.reset:
        if args.dry_run:
            logger.info("[DRY_RUN] Would reset collections and all caches")
            audit("dry_run_reset", {})
            clear_for_ingestion(verbose=True, dry_run=True)
        else:
            logger.info(f"[RESET] Collection Reset - deleting existing collections and caches")
            audit("collection_reset", {})
            if args.verbose:
                print("[RESET] Deleting existing collections and caches...")

            success = clear_for_ingestion(verbose=True, dry_run=False)

            if not success:
                logger.error("Reset failed - aborting ingestion")
                sys.exit(1)

            if args.verbose:
                print("[RESET] Reset complete.\n")

    # Create collections with no auto-embedding (we provide embeddings manually)
    # This ensures consistent use of EMBEDDING_MODEL_NAME instead of ChromaDB's default 384D
    chunk_collection = client.get_or_create_collection(
        name=config.chunk_collection_name, embedding_function=None  # Disable auto-embedding
    )
    doc_collection = client.get_or_create_collection(
        name=config.doc_collection_name, embedding_function=None  # Disable auto-embedding
    )
    if args.reset:
        logger.info(f"[RESET] Collections recreated")
    else:
        logger.info(
            f"Chunk Collection retrieved"
        )  # if first time run, would actually be created rather than retrieved.

    url_seed_files: List[Tuple[str, Optional[str]]] = []
    if config.include_url_seeds:
        logger.info(f"Fetching URL seeds from {config.url_seed_json_path}")
        url_seed_files = collect_url_files_from_seeds(config, logger)
        logger.info(
            f"Downloaded {len(url_seed_files)} URL-backed HTML files to {config.url_download_dir}"
        )

    logger.info(f"Starting file discovery from {config.base_path}")
    audit("start_file_discovery", {"base_path": config.base_path, "user": getpass.getuser()})
    files_to_process = []
    unsupported_count = 0
    filtered_count = 0
    for root, _, files in os.walk(config.base_path, topdown=True):
        if args.verbose:
            print(f"Walking directory: {root}")

        for file in files:
            file_path = os.path.join(root, file)
            # Skip unsupported/binary types early
            if _should_skip_unsupported(
                file_path,
                verbose=args.verbose,
                audit_fn=audit,
                log_unsupported=args.log_unsupported_files,
            ):
                unsupported_count += 1
                continue
            # Filter by regex pattern (applies to both HTML and PDF files)
            if re.fullmatch(config.ignore_file_regex, file.lower()):
                filtered_count += 1
                if args.verbose:
                    print(f"[FILTER] {file_path} (matched ignore pattern)")
                continue
            # Check HTML-specific filename filters
            if file.lower().endswith(".html"):
                if file.lower() in ("saved_resource.html", "render.html"):
                    filtered_count += 1
                    if args.verbose:
                        print(f"[FILTER] {file_path} (excluded by name)")
                    continue
            # File passes all filters - add both HTML and PDF files
            if file.lower().endswith((".html", ".pdf")):
                files_to_process.append(file_path)

    # Add URL-seeded files (may live outside base_path)
    url_source_map: Dict[str, Optional[str]] = {}
    for f, cat in url_seed_files:
        if f not in files_to_process:
            files_to_process.append(f)
        url_source_map[f] = cat

    file_count = len(files_to_process)
    if args.limit:
        print(
            f"[LIMIT] Processing only first {args.limit} files of {file_count} discovered files"
        )  # args.limit may be larger than number of discovered files
        files_to_process = files_to_process[: args.limit]

    logger.info(f"Ingesting {len(files_to_process)} files of {file_count} discovered files")
    logger.info(f"Using {config.max_workers} workers (dry_run={args.dry_run})")
    audit(
        "start_ingestion",
        {
            "workers": config.max_workers,
            "limit": args.limit,
            "files_to_process": len(files_to_process),
            "discovered_files": file_count,
            "unsupported_files": unsupported_count,
            "filtered_files": filtered_count,
            "dry_run": args.dry_run,
            "skip_llm_preprocess": getattr(args, "skip_llm_preprocess", False),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_db_path": config.rag_data_path,
            "chunk_collection": config.chunk_collection_name,
            "doc_collection": config.doc_collection_name,
            "embedding_cache_path": config.embedding_cache_path,
            "user": getpass.getuser(),
        },
    )
    print(f"Found {file_count} files (HTML & PDF), processing {len(files_to_process)} files")
    print(f"Using {config.max_workers} workers")

    # Initialise progress tracker for headless runs
    progress_tracker = None
    if config.progress_log_interval > 0:
        progress_tracker = ProgressTracker(
            total=len(files_to_process), log_interval=config.progress_log_interval, logger=logger
        )
        logger.info(
            f"Progress logging enabled: updates every {config.progress_log_interval} documents"
        )

    # Initialise dry-run stats collector
    dry_run_stats = None
    if args.dry_run:
        dry_run_stats = DryRunStats()
        logger.info("Dry-run mode: collecting preview statistics...")
        print("\n[DRY-RUN MODE] Analysing documents and generating preview report...\n")

    # Initialise profile stats collector
    profile_stats = None
    if args.profile:
        profile_stats = ProfileStats()
        logger.info("Profile mode: collecting detailed timing and error metrics...")

    # Initialise resource monitoring
    resource_monitor = None
    if config.enable_resource_monitoring:
        resource_monitor = ResourceMonitor(
            operation_name="document_ingestion",
            interval=config.resource_monitoring_interval,
            enabled=True,
            monitor_ollama=config.monitor_ollama,
            monitor_chromadb=config.monitor_chromadb,
        )
        resource_monitor.start()
        logger.info("Resource monitoring started")

    try:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = [
                executor.submit(
                    process_file,
                    file_path,
                    chunk_collection,
                    doc_collection,
                    config,
                    url_source_map.get(file_path),
                    llm_cache,
                    embedding_cache,
                    dry_run_stats,
                    profile_stats,
                )
                for file_path in files_to_process
            ]

            # execute process_file in parallel
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Document ingestion and parsing"
            ):
                try:
                    result = future.result()  # triggers exceptions if any
                    # Update progress tracker
                    if progress_tracker:
                        progress_tracker.increment(success=result)
                except Exception as e:
                    print(f"[ERROR] Worker exception: {e}")
                    # Update progress tracker for failure
                    if progress_tracker:
                        progress_tracker.increment(success=False)
    finally:
        # Stop resource monitoring
        if resource_monitor:
            resource_monitor.stop()
            resource_monitor.print_summary()
            stats_file = resource_monitor.export_json()
            logger.info(f"Resource statistics exported to {stats_file}")

    duration = time.perf_counter() - start_time

    # Log final progress summary
    if progress_tracker:
        logger.info(
            f"Final results: {progress_tracker.succeeded} succeeded, {progress_tracker.failed} failed out of {progress_tracker.total} total"
        )
        audit(
            "final_progress_summary",
            {
                "total": progress_tracker.total,
                "succeeded": progress_tracker.succeeded,
                "failed": progress_tracker.failed,
                "success_rate": (
                    (progress_tracker.succeeded / progress_tracker.total * 100)
                    if progress_tracker.total > 0
                    else 0
                ),
            },
        )

    # Flush caches to disk
    llm_cache.flush()
    logger.info(f"LLM cache flushed: {llm_cache.stats()['total_entries']} entries")

    embedding_cache.flush()
    emb_stats = embedding_cache.stats()
    logger.info(
        f"Embedding cache flushed: {emb_stats['total_entries']} entries, hit rate: {emb_stats['hit_rate']*100:.1f}%"
    )

    # Display dry-run preview report
    if args.dry_run and dry_run_stats:
        print("\n" + dry_run_stats.get_report())
        audit(
            "dry_run_preview",
            {
                "docs_new": dry_run_stats.docs_new,
                "docs_updated": dry_run_stats.docs_updated,
                "docs_skipped": dry_run_stats.docs_skipped,
                "chunks_total": dry_run_stats.chunks_total,
                "llm_calls_total": (
                    dry_run_stats.llm_calls_metadata
                    + dry_run_stats.llm_calls_validation
                    + dry_run_stats.llm_calls_repair
                ),
                "embeddings_total": dry_run_stats.embeddings_chunks + dry_run_stats.embeddings_docs,
                "estimated_time_seconds": dry_run_stats.estimated_time_seconds,
            },
        )

    # Display profile analysis report
    if args.profile and profile_stats:
        print("\n" + profile_stats.get_report())
        audit(
            "profile_analysis",
            {
                "docs_processed": profile_stats.doc_count,
                "chunks_created": profile_stats.chunk_count,
                "errors_count": len(profile_stats.errors),
                "warnings_count": len(profile_stats.warnings),
                "llm_calls": profile_stats.llm_calls,
                "llm_cache_rate": (
                    profile_stats.cache_hits_llm / profile_stats.llm_calls * 100
                    if profile_stats.llm_calls > 0
                    else 0
                ),
                "embedding_calls": profile_stats.embedding_calls,
                "embedding_cache_rate": (
                    profile_stats.cache_hits_embedding / profile_stats.embedding_calls * 100
                    if profile_stats.embedding_calls > 0
                    else 0
                ),
                "bottleneck_stage": (
                    max(profile_stats.stage_times.items(), key=lambda x: x[1])[0]
                    if profile_stats.stage_times
                    else "unknown"
                ),
            },
        )

    # Update BM25 corpus statistics (IDF values) after all documents indexed
    if config.bm25_indexing_enabled and not args.dry_run:
        try:
            logger.info("Computing BM25 corpus statistics (IDF values)...")
            cache_db = get_cache_client(enable_cache=True)
            total_docs = cache_db.get_bm25_corpus_size()

            if total_docs > 0:
                cache_db.update_bm25_corpus_stats(total_docs)
                avg_doc_len = cache_db.get_bm25_avg_doc_length()
                logger.info(
                    f"BM25 corpus stats updated: {total_docs} documents, avg length {avg_doc_len:.1f} tokens"
                )
                audit(
                    "bm25_corpus_stats_updated",
                    {"total_documents": total_docs, "avg_doc_length": avg_doc_len},
                )
        except Exception as e:
            logger.warning(f"Failed to update BM25 corpus statistics: {e}")
            audit(
                "bm25_corpus_stats_failed", {"error": str(e)[:200], "error_type": type(e).__name__}
            )

    logger.info(f"[DONE MAIN] Ingestion completed in {duration:.2f}s")
    audit("done_main", {"duration_seconds": duration})
    if args.verbose:
        print(f"Completed ingestion in {duration:.2f}s")


if __name__ == "__main__":
    main()
