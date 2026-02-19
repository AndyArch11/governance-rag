"""
Consistency Graph Builder for Governance Documents.

This module builds a version-aware consistency graph from governance documents stored
in ChromaDB, detecting relationships (conflicts, duplicates, consistency, partial conflicts)
between document versions using an LLM. It supports:

- Parallel LLM-based relationship detection via Ollama
- Thread-safe concurrent processing with ThreadPoolExecutor
- Community detection for risk and topic clusters
- Overlapping cluster membership based on edge strength
- Automatic LLM-generated cluster labels and summaries
- JSON serialisation for visualisation and downstream analysis

Key features:
- Version-aware: Handles multiple versions of the same document
- Conflict detection: Identifies conflicts and partial conflicts between documents
- Community detection: Uses greedy modularity optimisation with multiple weightings
- Cluster labeling: LLM generates human-readable labels for each cluster
- Logging: Comprehensive structured logging via scripts.utils.logger

Usage:
    collection = get_collection()
    graph = build_consistency_graph(
        collection=collection,
        max_neighbours=20,
        sim_threshold=0.4,
        workers=8
    )
    # graph["nodes"]: dict of versioned nodes with metadata and cluster assignments
    # graph["edges"]: list of consistency edges
    # graph["clusters"]: risk and topic cluster metadata with labels
"""

import argparse
import getpass
import json
import os
import re
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from networkx.algorithms.community import greedy_modularity_communities
from tqdm import tqdm

from scripts.consistency_graph.advanced_analytics import compute_advanced_analytics
from scripts.ingest.vectors import EMBEDDING_MODEL_NAME
from scripts.utils.db_factory import get_default_vector_path, get_vector_client
from scripts.utils.metrics_export import get_metrics_collector
from scripts.utils.monitoring import get_perf_metrics, get_token_counter, init_monitoring
from scripts.utils.resource_monitor import ResourceMonitor

# Collection typing (best-effort)
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

# Centralised backend selection (Chroma preferred)
PersistentClient, USING_SQLITE = get_vector_client(prefer="chroma")

# Handle both package imports and direct script execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from scripts.consistency_graph.consistency_config import get_consistency_config
    from scripts.utils.json_utils import extract_first_json_block, repair_json, sanitise_for_json
    from scripts.utils.logger import create_module_logger, flush_all_handlers

    get_logger, audit = create_module_logger("consistency")
    from scripts.consistency_graph.graph_cache_manager import GraphCacheManager
    from scripts.consistency_graph.sqlite_writer import SQLiteGraphWriter
else:
    from scripts.utils.json_utils import extract_first_json_block, repair_json, sanitise_for_json
    from scripts.utils.logger import create_module_logger, flush_all_handlers

    from .consistency_config import get_consistency_config

    get_logger, audit = create_module_logger("consistency")
    from .graph_cache_manager import GraphCacheManager
    from .sqlite_writer import SQLiteGraphWriter

# TODO add logging including Open Telemetry, currently logging minimal info
# TODO add embedding cache

# Initialise monitoring infrastructure
init_monitoring()
token_counter = get_token_counter()
perf_metrics = get_perf_metrics()
metrics_collector = get_metrics_collector()

# ============================================================================
# Configuration
# ============================================================================

# Load .env if present (does not override real environment variables)
load_dotenv(override=False)

# Centralised configuration (dotenv-friendly)
CONFIG = get_consistency_config()

# Vector store path based on selected backend
CHROMA_PATH = get_default_vector_path(Path(CONFIG.rag_data_path), USING_SQLITE)
DOC_COLLECTION_NAME = CONFIG.doc_collection_name
VALIDATOR_LLM_MODEL = CONFIG.llm_model_name

SIMILARITY_THRESHOLD = CONFIG.similarity_threshold  # max distance; lower = more similar
MAX_NEIGHBOURS = CONFIG.max_neighbours  # max neighbours per doc for consistency checks
WORKERS = CONFIG.workers  # parallel LLM workers

RELATIONSHIP_SEVERITY = {
    "conflict": 1.0,
    "partial_conflict": 0.6,
    "duplicate": 0.3,
    "consistent": 0.0,
}

VALID_RELATIONSHIPS = {"consistent", "partial_conflict", "conflict", "duplicate"}

# Thread-local storage for LLM instances to avoid sharing across workers
_thread_local = threading.local()


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
                "rate": rate,
                "eta_seconds": eta_seconds,
            },
        )


class EmbeddingCacheManager:
    """
    In-memory embedding cache to avoid redundant embedding retrievals.

    Caches embeddings by document ID and model, reducing ChromaDB queries
    when the same document is processed multiple times.

    Usage:
        cache = EmbeddingCacheManager()
        embedding = cache.get(doc_id, model_name)  # Returns cached or None
        cache.put(doc_id, model_name, embedding)
    """

    def __init__(self, logger=None):
        """Initialise embedding cache.

        Args:
            logger: Optional logger for debug messages
        """
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.logger = logger or get_logger()

    def _cache_key(self, doc_id: str, model: str) -> str:
        """Generate cache key from doc_id and model."""
        return f"{doc_id}:{model}"

    def get(self, doc_id: str, model: str) -> Optional[List[float]]:
        """Retrieve cached embedding.

        Args:
            doc_id: Document identifier
            model: Embedding model name

        Returns:
            Embedding vector if cached, else None
        """
        key = self._cache_key(doc_id, model)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, doc_id: str, model: str, embedding: List[float]) -> None:
        """Cache an embedding.

        Args:
            doc_id: Document identifier
            model: Embedding model name
            embedding: Embedding vector
        """
        key = self._cache_key(doc_id, model)
        self.cache[key] = embedding

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache_size, hits, misses, hit_rate
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "cache_size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": hit_rate,
        }


class LLMBatcher:
    """
    Batch document pair validation to reduce LLM invocations.

    Groups similar document pairs and submits them for batch LLM inference,
    reducing individual LLM call overhead compared to sequential processing.

    Usage:
        batcher = LLMBatcher(batch_size=10)
        for pair in document_pairs:
            result = batcher.validate_pair(doc_a, doc_b, meta_a, meta_b)
        results = batcher.flush()  # Process remaining items
    """

    def __init__(self, batch_size: int = 10, logger=None):
        """Initialise LLM batcher.

        Args:
            batch_size: Number of pairs to accumulate before batch processing
            logger: Optional logger for debug messages
        """
        self.batch_size = batch_size
        self.pending_pairs = []
        self.results_cache = {}
        self.logger = logger or get_logger()
        self.batch_count = 0

    def validate_pair(
        self, doc_a: str, doc_b: str, meta_a: Dict[str, Any], meta_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Queue a document pair for validation.

        Returns cached result if available, otherwise queues for batch processing.

        Args:
            doc_a, doc_b: Document texts
            meta_a, meta_b: Document metadata

        Returns:
            Validation result dict (may be from cache or batch)
        """
        # Create cache key from documents and metadata
        cache_key = hash(
            (
                doc_a[:100],  # First 100 chars to keep key size manageable
                doc_b[:100],
                meta_a.get("doc_type", ""),
                meta_b.get("doc_type", ""),
            )
        )

        if cache_key in self.results_cache:
            return self.results_cache[cache_key]

        # Queue for batch processing
        self.pending_pairs.append(
            {
                "cache_key": cache_key,
                "doc_a": doc_a,
                "doc_b": doc_b,
                "meta_a": meta_a,
                "meta_b": meta_b,
            }
        )

        # Process batch if full
        if len(self.pending_pairs) >= self.batch_size:
            self._process_batch()

        # Return placeholder; will be updated in cache
        return self.results_cache.get(
            cache_key,
            {
                "relationship": "consistent",
                "confidence": 0.0,
                "explanation": "Pending batch processing",
            },
        )

    def _process_batch(self) -> None:
        """Process accumulated batch of document pairs via LLM."""
        if not self.pending_pairs:
            return

        self.batch_count += 1
        batch_size = len(self.pending_pairs)

        if self.logger:
            self.logger.debug(f"Processing batch {self.batch_count} ({batch_size} pairs)")

        # Process each pair in the batch
        for item in self.pending_pairs:
            try:
                result = enforce_valid_output(
                    item["doc_a"], item["doc_b"], item["meta_a"], item["meta_b"]
                )
                self.results_cache[item["cache_key"]] = result
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Batch validation failed: {e}")
                # Store fallback result
                self.results_cache[item["cache_key"]] = {
                    "relationship": "consistent",
                    "confidence": 0.0,
                    "explanation": f"Validation error: {str(e)}",
                }

        self.pending_pairs = []

    def flush(self) -> None:
        """Process any remaining pending pairs."""
        if self.pending_pairs:
            self._process_batch()


def sample_and_interpolate_graph(
    versioned_docs: List[Dict[str, Any]],
    doc_collection: Collection,
    max_neighbours: int,
    sim_threshold: float,
    sampling_rate: float = 0.1,
    workers: int = WORKERS,
    enable_llm_batching: bool = False,
    enable_embedding_cache: bool = False,
    progress_callback: Optional[callable] = None,
    progress_log_interval: int = 10,
    logger=None,
) -> Dict[str, Any]:
    """
    Build high-quality graph on sampled subset, interpolate remaining nodes.

    This optimisation is useful for very large graphs (>5000 nodes) where
    building the full LLM-validated graph is too expensive. Instead:

    1. Sample ~10% of nodes randomly
    2. Build high-quality edges using LLM validation on sample
    3. For remaining nodes, assign edges based on nearest sample node

    N.B. This approach reduces LLM calls significantly but may introduce noise in interpolated edges.
    It can also be CPU intensive due to nearest neighbour searches, so use with caution on very large graphs, as may lead to long processing times.
    Consider using a more efficient ANN search or reducing the number of neighbours for interpolation.

    Args:
        versioned_docs: All document records
        sampling_rate: Fraction of nodes to sample (0.0-1.0)
        ... (other args same as build_consistency_graph_parallel)

    Returns:
        Complete graph dict with sampled edges + interpolated edges
    """
    if logger is None:
        logger = get_logger()

    import random

    doc_count = len(versioned_docs)
    sample_size = max(1, int(doc_count * sampling_rate))

    logger.info(
        f"Graph sampling enabled: {sample_size}/{doc_count} nodes ({sampling_rate*100:.1f}%)"
    )
    logger.info(f"Sampling strategy: Build LLM-validated edges on sample, interpolate remaining")

    # Sample nodes randomly
    sampled_docs = random.sample(versioned_docs, sample_size)
    remaining_docs = [d for d in versioned_docs if d not in sampled_docs]

    logger.info(f"Sampled {len(sampled_docs)} nodes, {len(remaining_docs)} to interpolate")

    # Build high-quality graph on sample
    logger.info("Building high-quality edges on sampled subset...")
    sampled_graph = build_consistency_graph_parallel(
        versioned_docs=sampled_docs,
        doc_collection=doc_collection,
        max_neighbours=max_neighbours,
        sim_threshold=sim_threshold,
        workers=workers,
        progress_callback=progress_callback,
        progress_log_interval=progress_log_interval,
        enable_llm_batching=enable_llm_batching,
        enable_embedding_cache=enable_embedding_cache,
    )

    # Add all nodes (including remaining ones)
    remaining_nodes = {
        make_node_id(d["doc_id"], d["version"]): {
            "doc_id": d["doc_id"],
            "version": d["version"],
            "timestamp": d.get("timestamp"),
            "doc_type": d.get("doc_type"),
            "summary": d.get("summary"),
            "source_category": d.get("source_category", ""),
            "health": d.get("health", {}),
        }
        for d in remaining_docs
    }

    sampled_graph["nodes"].update(remaining_nodes)

    # Interpolate edges for remaining nodes
    logger.info(f"Interpolating edges for {len(remaining_docs)} remaining nodes...")

    # For each remaining node, find its nearest sampled node and copy edges
    sampled_node_ids = set(make_node_id(d["doc_id"], d["version"]) for d in sampled_docs)

    for rem_doc in remaining_docs:
        rem_node_id = make_node_id(rem_doc["doc_id"], rem_doc["version"])

        # Find nearest sampled node by embedding similarity
        rem_embedding = rem_doc.get("embedding", [])
        if not rem_embedding:
            continue

        # Query for nearest sample nodes
        try:
            results = doc_collection.query(
                query_embeddings=[rem_embedding],
                n_results=5,
                include=["metadatas", "distances"],
            )
            nearest_sample_id = None
            for meta in results["metadatas"][0]:
                candidate_id = make_node_id(meta.get("doc_id"), meta.get("version", 1))
                if candidate_id in sampled_node_ids:
                    nearest_sample_id = candidate_id
                    break

            if nearest_sample_id:
                # Copy edges from nearest sample node
                for edge in sampled_graph["edges"]:
                    if edge["source"] == nearest_sample_id:
                        # Create new edge from remaining node to targets
                        new_edge = edge.copy()
                        new_edge["source"] = rem_node_id
                        new_edge["interpolated"] = True  # Mark as interpolated
                        # Reduce confidence for interpolated edges
                        if "confidence" in new_edge:
                            new_edge["confidence"] *= 0.8
                        sampled_graph["edges"].append(new_edge)
                    elif edge["target"] == nearest_sample_id:
                        # Create new edge from sources to remaining node
                        new_edge = edge.copy()
                        new_edge["target"] = rem_node_id
                        new_edge["interpolated"] = True
                        if "confidence" in new_edge:
                            new_edge["confidence"] *= 0.8
                        sampled_graph["edges"].append(new_edge)
        except Exception as e:
            logger.warning(f"Failed to interpolate edges for {rem_node_id}: {e}")

    logger.info(
        f"Graph sampling complete: {len(sampled_graph['nodes'])} nodes, {len(sampled_graph['edges'])} edges"
    )
    return sampled_graph


def get_validator_llm() -> OllamaLLM:
    """
    Get or create a thread-local LLM instance.

    Ensures each worker thread has its own LLM client to avoid concurrency issues.

    Returns:
        OllamaLLM: Thread-local instance of Ollama LLM (configured model).
    """
    if not hasattr(_thread_local, "llm"):
        _thread_local.llm = OllamaLLM(model=VALIDATOR_LLM_MODEL)
    return _thread_local.llm


logger = get_logger()


# ============================================================================
# Consistency check via LLM
# ============================================================================


def build_prompt(doc_a, doc_b, meta_a, meta_b):
    """Build comparison prompt with code-aware enhancements.

    Detects if both documents are code files (source_category='code')
    and includes code-specific comparison guidance.
    """
    # Check if both documents are code files
    is_code_a = meta_a.get("source_category") == "code"
    is_code_b = meta_b.get("source_category") == "code"
    both_code = is_code_a and is_code_b

    # Base definitions (same for all document types)
    definitions = """
    Definitions (follow these EXACTLY):
    
    - "consistent": The documents do not contradict each other in any way. 
                     Differences in scope or detail are allowed as long as they do not disagree.
    
    - "partial_conflict": The documents overlap in topic AND contain at least one 
                           differing requirement, recommendation, or statement that 
                           could cause confusion or misalignment.

    - "conflict": The documents contain directly opposing requirements, 
                  incompatible instructions, or mutually exclusive statements.
    
    - "duplicate": The documents express the same meaning, intent, or requirements 
                   even if wording differs.
    """

    # Code-specific guidance (only if both are code files)
    code_guidance = ""
    if both_code:
        code_guidance = """
    
    CODE-SPECIFIC COMPARISON GUIDANCE:
    When comparing code files, consider:
    
    - Service Dependencies: Do they depend on the same services or APIs? Are there 
      conflicting versions or incompatible interfaces?
    
    - Language/Framework Consistency: Are both using the same programming language, 
      framework versions, and architectural patterns?
    
    - Function/Class Relationships: Do they export/import the same functions or classes? 
      Are there naming conflicts or incompatible method signatures?
    
    - Configuration & Metadata: Do they specify conflicting configuration values, 
      environment variables, or deployment requirements?
    
    - Database & External Services: Do they reference conflicting database schemas, 
      API endpoints, or external service integrations?
    
    - Code Reuse: Are there duplicate implementations of the same functionality that 
      should be consolidated?
    
    Focus on architectural and operational conflicts, not minor code style differences.
    """

    prompt = f"""
    You are a strict classification engine. 
    You MUST output ONLY one of the following four relationship labels:
    
      - "consistent"
      - "partial_conflict"
      - "conflict"
      - "duplicate"
    
    These labels are EXACT and case-sensitive. 
    You are NOT allowed to invent new labels, paraphrase them, or modify them in any way.
    
    {definitions}{code_guidance}

    Your task:
    Compare the following two governance documents and classify their relationship 
    using ONLY the four allowed labels.

    DOC A ({meta_a["doc_type"]}, v{meta_a["version"]}):
    {doc_a}

    DOC B ({meta_b["doc_type"]}, v{meta_b["version"]}):
    {doc_b}

    Return ONLY valid JSON in the following EXACT structure:
    
    {{
      "relationship": "consistent" | "partial_conflict" | "conflict" | "duplicate",
      "confidence": <float between 0 and 1>,
      "explanation": "<short explanation>"
    }}
    
    If you cannot decide between two categories, choose the one with the higher risk 
    (e.g., partial_conflict instead of consistent).
    
    If your output contains any label other than the four allowed ones, 
    your answer is INVALID.
    """

    return prompt


# GPU concurrency limiter to prevent thrashing
class GPULock:
    """Semaphore to limit concurrent GPU operations and prevent thrashing."""

    def __init__(self, max_concurrent: int = 2):
        self.semaphore = threading.Semaphore(max_concurrent)

    def __enter__(self):
        self.semaphore.acquire()
        return self

    def __exit__(self, *args):
        self.semaphore.release()


# Global GPU lock instance (will be initialised with config value)
_gpu_lock: Optional[GPULock] = None


def get_gpu_lock() -> GPULock:
    """Get or create the global GPU lock with configured concurrency."""
    global _gpu_lock
    if _gpu_lock is None:
        config = get_consistency_config()
        _gpu_lock = GPULock(max_concurrent=config.gpu_concurrency)
    return _gpu_lock


def call_llm(prompt: str) -> str:
    """
    Invoke the thread-local LLM with error handling and logging.

    Uses GPU concurrency limiting to prevent thrashing when multiple
    threads compete for GPU resources.

    Args:
        prompt: The prompt to send to the LLM.

    Returns:
        str: The LLM's response text.

    Raises:
        Exception: If LLM invocation fails (logged before raising).
    """
    with get_gpu_lock():
        llm = get_validator_llm()
        try:
            return llm.invoke(prompt)
        except Exception:
            logger.exception("LLM invocation failed")
            raise


def enforce_valid_output(
    doc_a: str, doc_b: str, meta_a: Dict[str, Any], meta_b: Dict[str, Any], max_retries: int = 3
) -> Dict[str, Any]:
    """
    Call LLM and retry with self-critique if output violates relationship schema.

    Ensures relationship field contains only valid labels. Falls back to "consistent"
    with confidence 0.0 if maximum retries exceeded.

    Args:
        doc_a: First document text.
        doc_b: Second document text.
        meta_a: First document metadata (doc_type, version, etc.).
        meta_b: Second document metadata.
        max_retries: Maximum retry attempts before fallback.

    Returns:
        Dict[str, Any]: Validated JSON response with keys:
            - relationship: One of {"consistent", "partial_conflict", "conflict", "duplicate"}
            - confidence: Float between 0 and 1
            - explanation: Brief explanation of the classification
    """
    llm = get_validator_llm()

    for _ in range(max_retries):
        raw = llm.invoke(build_prompt(doc_a, doc_b, meta_a, meta_b))
        try:
            parsed = extract_first_json_block(raw)
        except Exception:
            parsed = {}

        relationship_raw = parsed.get("relationship", "")
        relationship = normalise_relationship(relationship_raw)

        if validate_relationship(relationship):
            parsed["relationship"] = relationship
            return parsed

        critique_prompt = f"""
Your previous answer used an invalid relationship label: "{relationship_raw}".

You MUST choose ONLY one of:
- "consistent"
- "partial_conflict"
- "conflict"
- "duplicate"

Correct your answer. Output ONLY valid JSON in the required format.
"""
        raw = llm.invoke(critique_prompt)
        try:
            parsed = extract_first_json_block(raw)
        except Exception:
            parsed = {}

        relationship_raw = parsed.get("relationship", "")
        relationship = normalise_relationship(relationship_raw)

        if validate_relationship(relationship):
            parsed["relationship"] = relationship
            return parsed

    # Final fallback
    return {
        "relationship": "consistent",
        "confidence": 0.0,
        "explanation": "Fallback due to repeated invalid outputs.",
    }


def normalise_relationship(rel: str) -> str:
    """
    Normalise fuzzy LLM responses to valid relationship labels.

    Handles common variations and misspellings from LLM output, mapping them
    to one of the four canonical relationship types. Uses exact matches first,
    then fuzzy pattern matching as fallback.

    Args:
        rel: Raw relationship string from LLM output.

    Returns:
        str: One of {"consistent", "partial_conflict", "conflict", "duplicate"}.
             Defaults to "consistent" if no match found.
    """
    if not rel:
        return "consistent"

    r = rel.strip().lower()

    # Exact and near-exact matches
    mapping = {
        "consistent": "consistent",
        "fully consistent": "consistent",
        "mostly consistent": "consistent",
        "aligned": "consistent",
        "fully aligned": "consistent",
        "partial_conflict": "partial_conflict",
        "partially conflicting": "partial_conflict",
        "partially inconsistent": "partial_conflict",
        "partially consistent": "partial_conflict",
        "partially aligned": "partial_conflict",
        "minor conflict": "partial_conflict",
        "minor discrepancy": "partial_conflict",
        "conflict": "conflict",
        "conflicting": "conflict",
        "fully conflicting": "conflict",
        "direct conflict": "conflict",
        "contradiction": "conflict",
        "contradictory": "conflict",
        "duplicate": "duplicate",
        "duplicates": "duplicate",
        "duplicated": "duplicate",
        "near duplicate": "duplicate",
    }

    if r in mapping:
        return mapping[r]

    # Fuzzy fallback rules
    if "duplicate" in r:
        return "duplicate"
    if "conflict" in r or "contradic" in r:
        return "conflict"
    if "partial" in r or "minor" in r:
        return "partial_conflict"
    if "consistent" in r or "aligned" in r:
        return "consistent"

    # Last resort
    # TODO: Review if "consistent" should be the default, as it may mask LLM issues. Consider returning "unknown" or raising an error instead.
    return "consistent"


# ============================================================================
# Helper Functions
# ============================================================================


def get_collection() -> Collection:
    """
    Get the ChromaDB collection for governance documents.

    Returns:
        Collection: The configured document collection.
    """
    client = PersistentClient(path=CHROMA_PATH)
    return client.get_collection(DOC_COLLECTION_NAME)


def make_node_id(doc_id: str, version: int) -> str:
    """
    Create a versioned node ID from document ID and version.

    Args:
        doc_id: Base document identifier.
        version: Version number.

    Returns:
        str: Formatted node ID like "doc123_v2".
    """
    return f"{doc_id}_v{version}"


def compute_edge_severity(relationship: str, confidence: float, similarity: float) -> float:
    """
    Compute edge severity score from relationship type and confidence.

    Severity = base_severity(relationship) x confidence x similarity.
    Used for weighting edges in risk clustering and conflict scoring.

    Args:
        relationship: Type of relationship (conflict, partial_conflict, etc.).
        confidence: LLM confidence score [0, 1].
        similarity: Semantic similarity [0, 1].

    Returns:
        float: Computed severity score [0, 1].
    """
    base = RELATIONSHIP_SEVERITY.get(relationship, 0.0)
    # similarity in [0,1], confidence in [0,1]
    return base * confidence * similarity


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, with fallback.

    Args:
        value: The value to convert (could be str, int, float, etc.)
        default: The default float value to return if conversion fails

    Returns:
        float: The converted float value, or default if conversion fails
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def distance_to_similarity(distance: float) -> float:
    """Convert a distance value to a bounded [0,1] similarity score.

    Args:
        distance: The distance value to convert (expected to be in [0, 1], but will be clamped)

    Returns:
        float: Similarity score in [0, 1], where 1 means identical and 0 means completely different.
    """
    # Clamp distance to [0, 1], then invert: similarity = 1 - distance
    # This ensures: distance=0 -> similarity=1, distance=1 -> similarity=0
    clamped_distance = max(0.0, min(1.0, distance))
    return 1.0 - clamped_distance


def calculate_optimal_neighbours(node_count: int, config_max: int) -> int:
    """Calculate optimal max_neighbours based on graph size.

    Large graphs need fewer neighbours per node to avoid O(n²) explosion.
    Falls back to config value for small graphs.

    Args:
        node_count: Total number of nodes in graph
        config_max: Configured max_neighbours value

    Returns:
        Optimal max_neighbours value
    """
    if node_count < 100:
        return min(config_max, 20)  # Small graph - check many neighbours
    elif node_count < 500:
        return min(config_max, 15)
    elif node_count < 2000:
        return min(config_max, 10)
    else:
        return min(config_max, 5)  # Large graph - only closest neighbours


def should_validate_with_llm(
    meta_a: Dict[str, Any], meta_b: Dict[str, Any], similarity: float, enable_heuristic: bool = True
) -> bool:
    """Reduce LLM calls with heuristic pre-filtering.

    Filters out document pairs that are unlikely to have meaningful
    relationships before expensive LLM validation.

    Args:
        meta_a: First document metadata
        meta_b: Second document metadata
        similarity: Similarity score [0, 1]
        enable_heuristic: Whether to apply heuristic filtering

    Returns:
        True if pair should be validated with LLM, False to skip
    """
    if not enable_heuristic:
        return True  # Heuristic filtering disabled

    # 1. Skip very high similarity - likely duplicates (fast path)
    if similarity > 0.90:
        logger.debug(f"Skipping LLM (very high similarity {similarity:.3f})")
        return False

    # 2. Skip moderate-low similarity - unlikely meaningful conflicts
    # Only validate similarity in "sweet spot" range: 0.5-0.9
    if similarity < 0.50:
        logger.debug(f"Skipping LLM (low similarity {similarity:.3f})")
        return False

    # Code-specific heuristics
    source_cat_a = meta_a.get("source_category", "")
    source_cat_b = meta_b.get("source_category", "")
    both_code = "code" in source_cat_a and "code" in source_cat_b

    if both_code:
        # 3. Code: Skip if different languages (unlikely conflicts)
        lang_a = meta_a.get("language", "").lower()
        lang_b = meta_b.get("language", "").lower()
        if lang_a and lang_b and lang_a != lang_b:
            logger.debug(f"Skipping LLM (different languages: {lang_a} vs {lang_b})")
            return False

        # 4. Code: Skip if no overlapping dependencies
        deps_a_raw = meta_a.get("dependencies", [])
        deps_b_raw = meta_b.get("dependencies", [])

        # Handle JSON strings
        if isinstance(deps_a_raw, str):
            try:
                deps_a = set(json.loads(deps_a_raw) if deps_a_raw else [])
            except (json.JSONDecodeError, TypeError):
                deps_a = set()
        else:
            deps_a = set(deps_a_raw) if deps_a_raw else set()

        if isinstance(deps_b_raw, str):
            try:
                deps_b = set(json.loads(deps_b_raw) if deps_b_raw else [])
            except (json.JSONDecodeError, TypeError):
                deps_b = set()
        else:
            deps_b = set(deps_b_raw) if deps_b_raw else set()

        if deps_a and deps_b and not (deps_a & deps_b):
            logger.debug(f"Skipping LLM (no shared dependencies)")
            return False

    # 5. Different doc types rarely conflict (except config-related)
    doc_type_a = str(meta_a.get("doc_type", "")).lower()
    doc_type_b = str(meta_b.get("doc_type", "")).lower()
    if doc_type_a and doc_type_b and doc_type_a != doc_type_b:
        # Exception: config vs code might conflict
        if not ("config" in doc_type_a or "config" in doc_type_b):
            logger.debug(f"Skipping LLM (different doc types: {doc_type_a} vs {doc_type_b})")
            return False

    return True  # Pass through to LLM
    return max(0.0, min(1.0, 1.0 - distance))


def dedupe_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse duplicate undirected edges, keeping the highest-severity entry.

    Normalises source/target ordering to ensure consistent keys and drops
    reverse duplicates produced by neighbour queries.
    """
    deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for edge in edges:
        u = edge["source"]
        v = edge["target"]
        key = tuple(sorted((u, v)))

        existing = deduped.get(key)
        if existing is None or edge.get("severity", 0.0) > existing.get("severity", 0.0):
            normalised = {**edge, "source": key[0], "target": key[1]}
            deduped[key] = normalised

    return list(deduped.values())


def load_versioned_docs(
    collection: Collection,
    batch_size: int = 500,
    where: Optional[Dict[str, Any]] = None,
    include_documents: bool = True,
) -> List[Dict[str, Any]]:
    """Stream versioned docs from Chroma to avoid loading the entire corpus at once.

    Args:
        collection: Chroma collection to read from.
        batch_size: Number of records per page; keeps memory bounded.
        where: Optional Chroma filter to limit records (e.g., source_category/health).
        include_documents: Whether to fetch full document texts. Set False to reduce memory
            and rely on summaries/metadata when available.

    Returns:
        List of records:
        {
          "doc_id": str,
          "version": int,
          "timestamp": str | None,
          "embedding": list[float],
          "summary": str,
          "doc_type": str,
          "source_category": str,
          "health": dict,
          "text": str
        }
    """
    include = ["metadatas", "embeddings"]
    if include_documents:
        include.append("documents")

    versioned: List[Dict[str, Any]] = []
    offset = 0

    while True:
        # Build parameters without an empty 'where' (Chroma rejects empty filter dicts)
        get_params = {
            "include": include,
            "limit": batch_size,
            "offset": offset,
        }
        # Only add 'where' if it's not None
        if where is not None:
            get_params["where"] = where

        try:
            batch = collection.get(**get_params)
        except Exception:
            # Fallback for backends that don't accept where=None
            batch = collection.get(
                include=include,
                limit=batch_size,
                offset=offset,
            )

        # collection.get returns empty lists when exhausted
        metas = batch.get("metadatas", [])
        embeddings = batch.get("embeddings", [])
        docs = batch.get("documents", []) if include_documents else [None] * len(metas)

        if not metas:
            break

        for meta, emb, text in zip(metas, embeddings, docs):
            doc_id = meta.get("doc_id")
            if not doc_id:
                continue

            health_raw = meta.get("health", "{}")
            try:
                health = json.loads(health_raw) if isinstance(health_raw, str) else health_raw
            except Exception:
                health = {}

            # Build base record
            meta_language = meta.get("language") or meta.get("lang")
            meta_doc_type = meta.get("doc_type") or meta.get("file_type")
            if not meta_doc_type and meta_language:
                meta_doc_type = "code"

            record = {
                "doc_id": doc_id,
                "version": meta.get("version", 1),
                "timestamp": meta.get("timestamp"),
                "embedding": emb,
                "summary": meta.get("summary", ""),
                "doc_type": meta_doc_type or "unknown",
                "source_category": meta.get("source_category", ""),
                "health": health,
                "text": text or "",
                "file_path": meta.get("file_path"),
                "repository": meta.get("repository"),
            }

            # Add code-specific metadata if present (for code documents)
            code_fields = [
                "language",
                "service_name",
                "service",
                "service_type",
                "dependencies",
                "internal_calls",
                "endpoints",
                "db",
                "queue",
                "exports",
                "git_provider",
                "git_host",
                "project",
                "project_key",
                "branch",
                "git_url",
                "bitbucket_url",
            ]
            for field in code_fields:
                if field in meta:
                    record[field] = meta[field]

            versioned.append(record)

        if len(metas) < batch_size:
            break

        offset += batch_size

    return versioned


def validate_relationship(rel: str) -> bool:
    """
    Validate that a relationship label is in the allowed set.

    Args:
        rel: Relationship string to validate.

    Returns:
        bool: True if rel is a valid relationship type.
    """
    return rel in VALID_RELATIONSHIPS


# ============================================================================
# Version-Aware Edge Building
# ============================================================================


def compute_edge_quality_score(similarity: float, confidence: float) -> float:
    """
    Compute combined quality score for an edge.

    Combines semantic similarity and LLM confidence into a normalised [0, 1] score.
    Both factors are important: high similarity without confidence, or confidence
    without similarity, are less reliable.

    Args:
        similarity: Semantic similarity score [0, 1]
        confidence: LLM confidence in the relationship [0, 1]

    Returns:
        Combined quality score [0, 1]
    """
    # Geometric mean gives balanced weight to both factors
    # If either is weak, the overall score is weak
    if similarity <= 0 or confidence <= 0:
        return 0.0
    return (similarity * confidence) ** 0.5


def should_expand_neighbours(
    edges_found: List[Dict[str, Any]],
    base_max_neighbours: int,
    quality_threshold: float = 0.7,
) -> bool:
    """
    Determine if we should expand neighbour search based on edge quality.

    If average quality of discovered edges is high, it suggests the node is
    highly connected and important in the landscape. Expanding the search
    will likely find more valuable relationships.

    Args:
        edges_found: List of edges discovered so far
        base_max_neighbours: Initial max_neighbours value used
        quality_threshold: Minimum average quality to trigger expansion (default: 0.7)

    Returns:
        True if expansion is recommended, False otherwise
    """
    if not edges_found or len(edges_found) < base_max_neighbours * 0.8:
        # Didn't find enough edges to be confident about quality
        return False

    # Compute average quality
    qualities = []
    for edge in edges_found:
        quality = compute_edge_quality_score(
            edge.get("similarity", 0.0), edge.get("confidence", 0.0)
        )
        qualities.append(quality)

    avg_quality = sum(qualities) / len(qualities) if qualities else 0.0
    return avg_quality >= quality_threshold


def expand_neighbour_results(
    doc_collection: Collection,
    query_embedding: List[float],
    query_model: str,
    base_max_neighbours: int,
    expanded_max_neighbours: int,
    sim_threshold: float,
) -> Tuple[List[str], List[str], List[Dict], List[float]]:
    """
    Query ChromaDB with expanded neighbour count.

    Args:
        doc_collection: ChromaDB collection to query
        query_embedding: Embedding vector to query with
        query_model: Embedding model name for filtering
        base_max_neighbours: Original query size
        expanded_max_neighbours: New expanded query size
        sim_threshold: Similarity threshold

    Returns:
        Tuple of (ids, docs, metas, distances) from expanded query
    """
    try:
        results = doc_collection.query(
            query_embeddings=[query_embedding],
            n_results=expanded_max_neighbours,
            where={"embedding_model": query_model},
            include=["documents", "metadatas", "distances"],
        )
    except TypeError:
        # Fallback if 'where' parameter not supported
        results = doc_collection.query(
            query_embeddings=[query_embedding],
            n_results=expanded_max_neighbours,
            include=["documents", "metadatas", "distances"],
        )

    # Return only the new results (skip the ones we already processed)
    ids = results["ids"][0][base_max_neighbours:]
    docs = results["documents"][0][base_max_neighbours:]
    metas = results["metadatas"][0][base_max_neighbours:]
    distances = results["distances"][0][base_max_neighbours:]

    return ids, docs, metas, distances


def process_document_for_graph(
    record: Dict[str, Any],
    doc_collection: Collection,
    max_neighbours: int,
    sim_threshold: float,
    enable_heuristic: bool = True,
    embedding_cache: Optional["EmbeddingCacheManager"] = None,
    llm_batcher: Optional["LLMBatcher"] = None,
    enable_dynamic_expansion: bool = True,
    expansion_quality_threshold: float = 0.7,
    max_expansion_multiplier: float = 1.5,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Build consistency edges for a single versioned document record.

    Queries ChromaDB for nearest neighbours, then uses LLM to classify
    the relationship between this document and each neighbour. Filters
    by similarity threshold and validates all relationships.

    Supports dynamic neighbour expansion: if initial neighbours have high
    quality scores (average above quality_threshold), expands search to find
    more valuable relationships.

    Returns metrics for monitoring performance.

    Args:
        record: Document record dict with keys:
            - doc_id: Base document identifier
            - version: Version number
            - embedding: Pre-computed embedding vector
            - summary: Document summary text
            - doc_type: Type of document
            - health: Health metrics dict
            - text: Full document text
        doc_collection: ChromaDB collection to query.
        max_neighbours: Maximum number of neighbours to consider initially.
        sim_threshold: Maximum distance threshold (lower = more similar).
        enable_heuristic: Enable heuristic filtering before LLM calls.
        enable_dynamic_expansion: Enable adaptive neighbour expansion based on quality.
        expansion_quality_threshold: Min avg quality [0, 1] to trigger expansion.
        max_expansion_multiplier: Max expansion ratio (e.g., 1.5 = 50% more neighbours).

    Returns:
        Tuple of (edges, metrics) where:
            - edges: List of edge dicts
            - metrics: Dict with keys llm_calls, filtered, expansion_applied, expanded_neighbours_found
    """
    # Profiling: Track timing for different operations
    prof_start = time.time()
    edges = []
    metrics = {
        "llm_calls": 0,
        "filtered": 0,
        "expansion_applied": False,
        "expanded_neighbours_found": 0,
        # Profiling metrics
        "time_chromadb_query": 0.0,
        "time_heuristic_filter": 0.0,
        "time_llm_validation": 0.0,
        "time_edge_processing": 0.0,
        "time_total": 0.0,
    }

    doc_id = record["doc_id"]
    version = record["version"]
    source_node = make_node_id(doc_id, version)

    summary_text = record["summary"] or record["text"]

    meta_a = {
        "doc_id": doc_id,
        "summary": record["summary"],
        "doc_type": record["doc_type"],
        "version": version,
        "health": record["health"],
        "source_category": record.get("source_category", ""),
    }

    # Add code-specific metadata for heuristic filtering
    code_fields = ["language", "dependencies", "service", "service_name"]
    for field in code_fields:
        if field in record:
            meta_a[field] = record[field]

    # Use the embedding already stored in Chroma
    query_embedding = record["embedding"]
    query_model = record.get("embedding_model", "mxbai-embed-large")

    # PROFILING: ChromaDB query time
    prof_chroma_start = time.time()
    # Query Chroma for nearest neighbours (filter by embedding model to avoid cross-model mixing)
    try:
        results = doc_collection.query(
            query_embeddings=[query_embedding],
            n_results=max_neighbours,
            where={"embedding_model": query_model},
            include=["documents", "metadatas", "distances"],
        )
    except TypeError:
        # Fallback if 'where' parameter not supported
        results = doc_collection.query(
            query_embeddings=[query_embedding],
            n_results=max_neighbours,
            include=["documents", "metadatas", "distances"],
        )
    metrics["time_chromadb_query"] = time.time() - prof_chroma_start

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(ids)):
        meta_b = metas[i]
        related_id = meta_b.get("doc_id")  # Use doc_id from metadata, not ChromaDB ID
        version_target = meta_b.get("version", 1)

        # Skip self (same doc_id AND same version)
        if related_id == doc_id and version_target == version:
            continue

        # Skip low similarity (distance too high)
        if distances[i] > sim_threshold:
            continue

        # Convert distance → similarity (bounded)
        similarity = distance_to_similarity(distances[i])

        # PROFILING: Heuristic filter time
        prof_heuristic_start = time.time()
        # Apply heuristic filtering before expensive LLM call
        should_validate = should_validate_with_llm(meta_a, meta_b, similarity, enable_heuristic)
        metrics["time_heuristic_filter"] += time.time() - prof_heuristic_start

        if not should_validate:
            metrics["filtered"] += 1
            continue

        target_text = docs[i]

        # PROFILING: LLM validation time
        prof_llm_start = time.time()
        consistency = enforce_valid_output(summary_text, target_text, meta_a, meta_b)
        metrics["time_llm_validation"] += time.time() - prof_llm_start
        metrics["llm_calls"] += 1

        relationship = consistency["relationship"]

        if not validate_relationship(relationship):
            logger.warning(
                "Invalid relationship after normalisation; skipping edge",
                extra={"relationship": relationship},
            )
            continue

        # PROFILING: Edge processing time
        prof_edge_start = time.time()
        confidence = safe_float(consistency.get("confidence", 0.0))
        explanation = consistency.get("explanation", "")

        severity = compute_edge_severity(relationship, confidence, similarity)

        target_node = make_node_id(related_id, version_target)

        # Skip reverse duplicates: only emit edge when source_node sorts before target_node
        if source_node > target_node:
            continue

        edges.append(
            {
                "source": source_node,
                "target": target_node,
                "relationship": relationship,
                "confidence": confidence,
                "explanation": explanation,
                "similarity": similarity,
                "severity": severity,
                "version_source": version,
                "version_target": version_target,
            }
        )
        metrics["time_edge_processing"] += time.time() - prof_edge_start

    # Check if we should dynamically expand neighbour search
    if enable_dynamic_expansion and should_expand_neighbours(
        edges, max_neighbours, expansion_quality_threshold
    ):
        expanded_max = min(
            int(max_neighbours * max_expansion_multiplier),
            max_neighbours + 10,  # Hard cap at +10 more neighbours
        )

        try:
            # Query for additional neighbours
            exp_ids, exp_docs, exp_metas, exp_distances = expand_neighbour_results(
                doc_collection,
                query_embedding,
                query_model,
                max_neighbours,
                expanded_max,
                sim_threshold,
            )

            metrics["expansion_applied"] = True

            # Process expanded results
            for i in range(len(exp_ids)):
                meta_b = exp_metas[i]
                related_id = meta_b.get("doc_id")
                version_target = meta_b.get("version", 1)

                # Skip self
                if related_id == doc_id and version_target == version:
                    continue

                # Skip low similarity
                if exp_distances[i] > sim_threshold:
                    continue

                similarity = distance_to_similarity(exp_distances[i])

                # Apply heuristic filtering
                if not should_validate_with_llm(meta_a, meta_b, similarity, enable_heuristic):
                    metrics["filtered"] += 1
                    continue

                target_text = exp_docs[i]

                consistency = enforce_valid_output(summary_text, target_text, meta_a, meta_b)
                metrics["llm_calls"] += 1

                relationship = consistency["relationship"]

                if not validate_relationship(relationship):
                    continue

                confidence = safe_float(consistency.get("confidence", 0.0))
                explanation = consistency.get("explanation", "")

                severity = compute_edge_severity(relationship, confidence, similarity)

                target_node = make_node_id(related_id, version_target)

                # Skip reverse duplicates
                if source_node > target_node:
                    continue

                edges.append(
                    {
                        "source": source_node,
                        "target": target_node,
                        "relationship": relationship,
                        "confidence": confidence,
                        "explanation": explanation,
                        "similarity": similarity,
                        "severity": severity,
                        "version_source": version,
                        "version_target": version_target,
                    }
                )

                metrics["expanded_neighbours_found"] += 1

        except Exception as e:
            logger.warning(
                f"Failed to expand neighbours for {source_node}: {e}",
                exc_info=False,
            )
            # Continue without expansion; don't fail the entire process

    # PROFILING: Record total time
    metrics["time_total"] = time.time() - prof_start
    return edges, metrics


def build_consistency_graph_parallel(
    versioned_docs: List[Dict[str, Any]],
    doc_collection: Collection,
    max_neighbours: int,
    sim_threshold: float,
    workers: int = WORKERS,
    progress_callback: Optional[callable] = None,
    include_dependency_edges: bool = False,
    enable_llm_batching: bool = False,
    enable_embedding_cache: bool = False,
    enable_graph_sampling: bool = False,
    graph_sampling_rate: float = 0.1,
    progress_log_interval: int = 10,
    enable_dynamic_expansion: bool = True,
    expansion_quality_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Build a version-aware consistency graph using parallel LLM checks.

    Creates versioned nodes for each document version, then processes documents
    in parallel using ThreadPoolExecutor to build edges via LLM classification.

    Args:
        versioned_docs: List of document records from load_versioned_docs().
        doc_collection: ChromaDB collection for similarity queries.
        max_neighbours: Maximum neighbours per document.
        sim_threshold: Similarity threshold for edge creation.
        workers: Number of parallel worker threads.
        progress_callback: Optional callback(step, total, message) for progress updates.

    Returns:
        Dict[str, Any]: Graph dict with keys:
            - nodes: Dict mapping node_id to node attributes
            - edges: List of edge dicts
    """
    graph = {"nodes": {}, "edges": []}

    # Performance monitoring counters
    llm_call_count = 0
    filtered_count = 0
    cache_hit_count = 0
    cache_miss_count = 0
    expansion_applied_count = 0
    expanded_neighbours_total = 0
    start_time = time.time()

    # Profiling aggregators
    total_chromadb_time = 0.0
    total_heuristic_time = 0.0
    total_llm_time = 0.0
    total_edge_time = 0.0
    total_processing_time = 0.0
    profile_sample_count = 0

    # Get config for optimisation settings
    config = get_consistency_config()

    # Calculate optimal max_neighbours based on graph size
    doc_count = len(versioned_docs)
    optimal_max_neighbours = calculate_optimal_neighbours(doc_count, max_neighbours)
    if optimal_max_neighbours != max_neighbours:
        logger.info(
            f"Adjusted max_neighbours from {max_neighbours} to {optimal_max_neighbours} "
            f"for graph with {doc_count} nodes (optimisation for large graphs)"
        )
        max_neighbours = optimal_max_neighbours

    # -----------------------------
    # 1. Add versioned nodes
    # -----------------------------
    for rec in versioned_docs:
        node_id = make_node_id(rec["doc_id"], rec["version"])
        # Start with core fields
        node_data = {
            "doc_id": rec["doc_id"],
            "version": rec["version"],
            "timestamp": rec.get("timestamp"),
            "doc_type": rec.get("doc_type"),
            "summary": rec.get("summary"),
            "source_category": rec.get("source_category", ""),
            "health": rec.get("health", {}),
            "repository": rec.get("repository"),
            "file_path": rec.get("file_path"),
        }

        # Add code-specific metadata fields if present (from code parser)
        code_fields = [
            "language",
            "service_name",
            "service",
            "service_type",
            "dependencies",
            "internal_calls",
            "endpoints",
            "db",
            "queue",
            "exports",
        ]
        for field in code_fields:
            if field in rec:
                # Deserialise JSON strings back to lists for code fields
                value = rec[field]
                if isinstance(value, str) and field in [
                    "dependencies",
                    "internal_calls",
                    "endpoints",
                    "db",
                    "queue",
                    "exports",
                ]:
                    try:
                        node_data[field] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        node_data[field] = value
                else:
                    node_data[field] = value

        # Map database_refs from ChromaDB to db field for consistency
        if "database_refs" in rec and "db" not in node_data:
            value = rec["database_refs"]
            if isinstance(value, str):
                try:
                    node_data["db"] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    node_data["db"] = value
            else:
                node_data["db"] = value

        graph["nodes"][node_id] = node_data

    # -----------------------------
    # 2. Parallel edge building
    # -----------------------------
    doc_count = len(versioned_docs)
    progress_tracker = ProgressTracker(
        total=doc_count, log_interval=progress_log_interval, logger=logger
    )

    # Initialise optional optimisation caches
    embedding_cache = EmbeddingCacheManager(logger=logger) if enable_embedding_cache else None
    llm_batcher = LLMBatcher(batch_size=10, logger=logger) if enable_llm_batching else None

    if enable_llm_batching:
        logger.info("LLM batching enabled: grouping similar documents for batch inference")
    if enable_embedding_cache:
        logger.info("Embedding cache enabled: caching embeddings to reduce ChromaDB queries")
    if enable_graph_sampling:
        logger.info(
            f"Graph sampling enabled: sampling {int(graph_sampling_rate*100)}% of nodes for LLM validation"
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_document_for_graph,
                rec,
                doc_collection,
                optimal_max_neighbours,  # Use adaptive max_neighbours
                sim_threshold,
                config.enable_heuristic_filter,  # Pass heuristic filter flag
                embedding_cache=embedding_cache,
                llm_batcher=llm_batcher,
                enable_dynamic_expansion=enable_dynamic_expansion,
                expansion_quality_threshold=expansion_quality_threshold,
            ): rec
            for rec in versioned_docs
        }

        i = 0
        for future in tqdm(
            as_completed(futures), total=doc_count, desc="Building version-aware consistency graph"
        ):
            rec = futures[future]
            i += 1
            if progress_callback:
                progress_callback(i, doc_count, f"Loading documents ({i}/{doc_count})")

            try:
                edges, metrics = future.result()  # Unpack edges and metrics tuple
                graph["edges"].extend(edges)
                progress_tracker.increment(success=True)

                # Aggregate metrics from worker thread
                llm_call_count += metrics.get("llm_calls", 0)
                filtered_count += metrics.get("filtered", 0)
                cache_hit_count += metrics.get("cache_hits", 0)
                cache_miss_count += metrics.get("cache_misses", 0)

                # Aggregate profiling metrics
                total_chromadb_time += metrics.get("time_chromadb_query", 0.0)
                total_heuristic_time += metrics.get("time_heuristic_filter", 0.0)
                total_llm_time += metrics.get("time_llm_validation", 0.0)
                total_edge_time += metrics.get("time_edge_processing", 0.0)
                total_processing_time += metrics.get("time_total", 0.0)
                profile_sample_count += 1

                if metrics.get("expansion_applied", False):
                    expansion_applied_count += 1
                expanded_neighbours_total += metrics.get("expanded_neighbours_found", 0)

                # Enhanced progress logging every 50 nodes
                if i % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    eta_seconds = (doc_count - i) / rate if rate > 0 else 0
                    eta_minutes = eta_seconds / 60

                    cache_total = cache_hit_count + cache_miss_count
                    cache_hit_rate = (cache_hit_count / cache_total * 100) if cache_total > 0 else 0

                    logger.info(
                        f"Progress: {i}/{doc_count} nodes ({i/doc_count*100:.1f}%) | "
                        f"Edges: {len(graph['edges'])} | "
                        f"LLM calls: {llm_call_count} | "
                        f"Filtered: {filtered_count} | "
                        f"Cache hits: {cache_hit_rate:.1f}% | "
                        f"Rate: {rate:.1f} nodes/sec | "
                        f"ETA: {eta_minutes:.1f}min"
                    )
            except Exception as e:
                print(f"Error processing {rec['doc_id']} v{rec['version']}: {e}")
                traceback.print_exc(file=sys.stdout)
                progress_tracker.increment(success=False)

            # Log profiling data every 10 documents
            if i % 10 == 0 and profile_sample_count > 0:
                avg_chromadb = total_chromadb_time / profile_sample_count
                avg_heuristic = total_heuristic_time / profile_sample_count
                avg_llm = total_llm_time / profile_sample_count
                avg_edge = total_edge_time / profile_sample_count
                avg_total = total_processing_time / profile_sample_count

                # Calculate actual LLM call rate
                avg_llm_calls = llm_call_count / i if i > 0 else 0
                avg_filtered = filtered_count / i if i > 0 else 0

                logger.info(
                    f"PROFILING [{i}/{doc_count}]: Avg per-doc times - "
                    f"ChromaDB: {avg_chromadb:.3f}s, Heuristic: {avg_heuristic:.4f}s, "
                    f"LLM: {avg_llm:.3f}s, Edge: {avg_edge:.4f}s, Total: {avg_total:.3f}s | "
                    f"Avg LLM calls/doc: {avg_llm_calls:.1f}, Avg filtered/doc: {avg_filtered:.1f}"
                )

    # Flush any remaining batched LLM calls
    if llm_batcher:
        logger.info("Flushing remaining LLM batches...")
        llm_batcher.flush()

    # ========================================================================
    # 3. Add cross-repository service clustering edges
    # ========================================================================
    # Detect services appearing in multiple repositories and create clustering edges
    code_nodes = {nid: n for nid, n in graph["nodes"].items() if n.get("source_category") == "code"}
    if code_nodes:
        service_repos = defaultdict(set)
        service_nodes = defaultdict(list)

        # Map services to their repositories and nodes
        for node_id, meta in code_nodes.items():
            service = meta.get("service_name") or meta.get("service")
            repo = meta.get("repository")
            if service and repo:
                service_repos[service].add(repo)
                service_nodes[service].append((node_id, repo))

        # Create edges for services appearing in multiple repos (potential duplication/sharing)
        for service, repos in service_repos.items():
            if len(repos) > 1:
                # This service appears in multiple repositories
                nodes_list = service_nodes[service]
                for i, (node_i, repo_i) in enumerate(nodes_list):
                    for node_j, repo_j in nodes_list[i + 1 :]:
                        if repo_i != repo_j:  # Only cross-repo edges
                            # Avoid duplicates
                            n1, n2 = sorted([node_i, node_j])
                            if not any(
                                (e["source"] == n1 and e["target"] == n2)
                                or (e["source"] == n2 and e["target"] == n1)
                                for e in graph["edges"]
                                if e.get("relationship") == "cross_repo_service"
                            ):
                                graph["edges"].append(
                                    {
                                        "source": n1,
                                        "target": n2,
                                        "relationship": "cross_repo_service",
                                        "service": service,
                                        "repos": list(repos),
                                        "confidence": 1.0,
                                        "severity": 0.5,  # Medium severity for potential duplication
                                        "explanation": f"Service '{service}' appears in multiple repos: {', '.join(sorted(repos))}. Potential code duplication or shared library.",
                                    }
                                )

    # ========================================================================
    # 4. Add dependency-based edges (optional)
    # ========================================================================
    if include_dependency_edges:
        # Index code nodes by service, dependencies, queue, db
        service_map = {}
        queue_map = {}
        db_map = {}
        dep_map = {}
        code_nodes = {
            nid: n for nid, n in graph["nodes"].items() if n.get("source_category") == "code"
        }
        for node_id, meta in code_nodes.items():
            # Service
            service = meta.get("service")
            if service:
                service_map.setdefault(service, set()).add(node_id)
            # message queue (AMQP)
            queue = meta.get("queue") or meta.get("amqp_queue")
            if queue:
                queue_map.setdefault(queue, set()).add(node_id)
            # DB
            db = meta.get("db") or meta.get("database")
            if db:
                db_map.setdefault(db, set()).add(node_id)
            # Dependencies (list or comma-separated string)
            deps = meta.get("dependencies")
            if isinstance(deps, str):
                deps = [d.strip() for d in deps.split(",") if d.strip()]
            if isinstance(deps, list):
                for dep in deps:
                    dep_map.setdefault(dep, set()).add(node_id)

        # Helper to add undirected edge if not already present
        def add_dep_edge(n1, n2, rel, field, value):
            if n1 == n2:
                return
            # Only add if not already present (undirected)
            for e in graph["edges"]:
                if (
                    (
                        (e["source"] == n1 and e["target"] == n2)
                        or (e["source"] == n2 and e["target"] == n1)
                    )
                    and e.get("relationship") == rel
                    and e.get("field") == field
                    and e.get("value") == value
                ):
                    return
            graph["edges"].append(
                {
                    "source": n1,
                    "target": n2,
                    "relationship": rel,
                    "field": field,
                    "value": value,
                    "confidence": 1.0,
                    "explanation": f"Shared {field}: {value}",
                }
            )

        # Add edges for each shared field
        for field, cmap in [
            ("service", service_map),
            ("queue", queue_map),
            ("db", db_map),
            ("dependency", dep_map),
        ]:
            for value, nodes in cmap.items():
                nodes = list(nodes)
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        add_dep_edge(nodes[i], nodes[j], "dependency", field, value)

    # Drop any remaining duplicate undirected edges before returning
    graph["edges"] = dedupe_edges(graph["edges"])

    # Log final optimisation summary
    total_time = time.time() - start_time
    total_comparisons = doc_count * optimal_max_neighbours
    filtered_percentage = (filtered_count / total_comparisons * 100) if total_comparisons > 0 else 0
    cache_total = cache_hit_count + cache_miss_count
    cache_hit_rate = (cache_hit_count / cache_total * 100) if cache_total > 0 else 0

    logger.info("=" * 80)
    logger.info("Consistency Graph Building Complete - Performance Summary")
    logger.info("=" * 80)
    logger.info(f"Total nodes: {doc_count}")
    logger.info(f"Total edges: {len(graph['edges'])}")
    logger.info(f"Max neighbours per node: {optimal_max_neighbours}")
    logger.info(f"Total potential comparisons: {total_comparisons}")
    logger.info(f"Heuristic filtered: {filtered_count} ({filtered_percentage:.1f}%)")
    logger.info(f"LLM calls made: {llm_call_count}")
    logger.info(f"Cache hits: {cache_hit_count} / {cache_total} ({cache_hit_rate:.1f}%)")
    if expansion_applied_count > 0:
        logger.info(f"Dynamic expansion applied: {expansion_applied_count} nodes")
        logger.info(f"Extra neighbours found via expansion: {expanded_neighbours_total}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Processing rate: {doc_count/total_time:.2f} nodes/sec")

    # PROFILING: Log breakdown of time spent
    if profile_sample_count > 0:
        logger.info("=" * 80)
        logger.info("PROFILING SUMMARY (average per document):")
        logger.info(f"  ChromaDB query:     {total_chromadb_time/profile_sample_count:.3f}s")
        logger.info(f"  Heuristic filter:   {total_heuristic_time/profile_sample_count:.4f}s")
        logger.info(f"  LLM validation:     {total_llm_time/profile_sample_count:.3f}s")
        logger.info(f"  Edge processing:    {total_edge_time/profile_sample_count:.4f}s")
        logger.info(f"  TOTAL per doc:      {total_processing_time/profile_sample_count:.3f}s")
        logger.info("=" * 80)
        logger.info("PROFILING SUMMARY (total time breakdown):")
        logger.info(
            f"  ChromaDB query:     {total_chromadb_time/60:.1f}min ({total_chromadb_time/total_time*100:.1f}%)"
        )
        logger.info(
            f"  Heuristic filter:   {total_heuristic_time/60:.1f}min ({total_heuristic_time/total_time*100:.1f}%)"
        )
        logger.info(
            f"  LLM validation:     {total_llm_time/60:.1f}min ({total_llm_time/total_time*100:.1f}%)"
        )
        logger.info(
            f"  Edge processing:    {total_edge_time/60:.1f}min ({total_edge_time/total_time*100:.1f}%)"
        )
        logger.info(
            f"  Other overhead:     {(total_time-total_processing_time)/60:.1f}min ({(total_time-total_processing_time)/total_time*100:.1f}%)"
        )
    logger.info("=" * 80)

    # Add build metadata to graph
    from datetime import datetime, timezone

    graph["metadata"] = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "build_duration_seconds": round(total_time, 2),
        "max_neighbours_configured": (
            max_neighbours if max_neighbours == optimal_max_neighbours else max_neighbours
        ),
        "max_neighbours_used": optimal_max_neighbours,
        "graph_size": doc_count,
        "total_edges": len(graph["edges"]),
        "similarity_threshold": sim_threshold,
        "llm_calls": llm_call_count,
        "heuristic_filtered": filtered_count,
        "cache_hits": cache_hit_count,
        "cache_total": cache_total,
        "cache_hit_rate_percent": round(cache_hit_rate, 2),
        "dynamic_expansion_applied_count": expansion_applied_count,
        "expanded_neighbours_total": expanded_neighbours_total,
    }

    return graph


# ============================================================================
# NetworkX Conversion and Metrics
# ============================================================================


def to_networkx(graph: Dict[str, Any]) -> nx.Graph:
    """
    Convert JSON graph representation to NetworkX graph.

    Merges duplicate edges by taking maximum severity/similarity values.
    All node and edge attributes are preserved.

    Args:
        graph: Graph dict with "nodes" (dict) and "edges" (list).

    Returns:
        nx.Graph: NetworkX undirected graph with all attributes.
    """
    G = nx.Graph()

    # Add nodes
    for node_id, data in graph["nodes"].items():
        G.add_node(node_id, **data)

    # Add edges
    for edge in graph["edges"]:
        u = edge["source"]
        v = edge["target"]

        if G.has_edge(u, v):
            # merge severity/similarity if multiple edges exist
            existing = G[u][v]
            existing["severity"] = max(existing.get("severity", 0.0), edge.get("severity", 0.0))
            existing["similarity"] = max(
                existing.get("similarity", 0.0), edge.get("similarity", 0.0)
            )
        else:
            G.add_edge(u, v, **edge)

    return G


def compute_version_node_conflict(G: nx.Graph) -> None:
    """
    Compute conflict score per versioned node in-place.

    Conflict score = sum of severity weights of all incident edges.
    Higher scores indicate more/stronger conflicts with other documents.

    Args:
        G: NetworkX graph with severity edge attributes.

    Side effects:
        Adds "conflict_score" attribute to each node in G.
    """
    for node in G.nodes():
        edges = G.edges(node, data=True)
        conflict_score = sum(e.get("severity", 0.0) for _, _, e in edges)
        G.nodes[node]["conflict_score"] = conflict_score


def compute_version_node_clusters(G):
    """Compute clusters and return community assignments for risk and topic.

    Allows nodes to belong to multiple clusters based on edge strength to other communities.
    """

    # Helper: assign node to secondary clusters based on edge strength threshold
    def assign_secondary_clusters(G, primary_clusters, weight_attr, strength_threshold=0.5):
        """
        For each node, compute average edge weight to each primary cluster.
        If average weight to cluster > threshold, add node to that cluster too.
        """
        # Map node -> primary cluster
        node_to_primary = {}
        for cid, comm in enumerate(primary_clusters):
            for node in comm:
                node_to_primary[node] = cid

        # For each node, compute avg edge weight to each cluster
        for node in G.nodes():
            primary_cid = node_to_primary.get(node)
            edge_weights_by_cluster = defaultdict(list)

            for neighbour in G.neighbors(node):
                neighbour_cid = node_to_primary.get(neighbour)
                if neighbour_cid is None:
                    continue
                edge_data = G[node][neighbour]
                weight = edge_data.get(weight_attr, 0.0)
                edge_weights_by_cluster[neighbour_cid].append(weight)

            # Assign to secondary clusters if avg weight is strong enough
            for cluster_id, weights in edge_weights_by_cluster.items():
                if cluster_id == primary_cid:
                    continue
                avg_weight = sum(weights) / len(weights) if weights else 0.0
                if avg_weight >= strength_threshold:
                    G.nodes[node].setdefault(
                        "risk_clusters" if weight_attr == "risk_weight" else "topic_clusters", []
                    ).append(cluster_id)

    # -----------------------------
    # Risk clusters (severity-weighted)
    # -----------------------------
    for _, _, data in G.edges(data=True):
        sev = float(data.get("severity", 0.0))
        data["risk_weight"] = max(sev, 1e-6)

    risk_comms = list(greedy_modularity_communities(G, weight="risk_weight"))

    for cid, comm in enumerate(risk_comms):
        for node in comm:
            G.nodes[node].setdefault("risk_clusters", []).append(cid)

    # Assign to secondary risk clusters based on severity strength
    assign_secondary_clusters(G, risk_comms, "risk_weight", strength_threshold=0.5)

    # -----------------------------
    # Topic clusters (similarity-weighted)
    # -----------------------------
    for _, _, data in G.edges(data=True):
        sim = float(data.get("similarity", 0.0))
        data["topic_weight"] = max(sim, 1e-6)

    topic_comms = list(greedy_modularity_communities(G, weight="topic_weight"))

    for cid, comm in enumerate(topic_comms):
        for node in comm:
            G.nodes[node].setdefault("topic_clusters", []).append(cid)

    # Assign to secondary topic clusters based on similarity strength
    assign_secondary_clusters(G, topic_comms, "topic_weight", strength_threshold=0.5)

    return risk_comms, topic_comms


def llm_label_cluster(
    docs: List[Dict[str, Any]],
    severities: List[float],
    topics: List[str],
    cluster_type: str,
    is_code_cluster: bool = False,
    code_languages: Optional[List[str]] = None,
    code_services: Optional[List[str]] = None,
) -> Tuple[str, str, str]:
    """
    Generate human-readable label, description, and summary for a cluster via LLM.

    Uses GPT to analyse cluster contents and generate meaningful metadata.
    Tailors prompts based on cluster_type (risk vs topic).

    Args:
        docs: List of document dicts with doc_id, version, summary.
        severities: List of severity scores for risk weighting.
        topics: List of topic/summary strings.
        cluster_type: Either "risk" or "topic".

    Returns:
        Tuple[str, str, str]: (label, description, summary).
            - label: Short cluster name (5-7 words)
            - description: 1-2 sentence theme description
            - summary: Detailed cluster summary
    """

    # Enhanced doc list for code clusters
    if is_code_cluster:
        doc_list = "\n".join(
            f"- {d['doc_id']} (v{d['version']}): {d.get('summary','')}"
            + (f" [lang: {d.get('language','unknown')}]" if "language" in d else "")
            + (f" [service: {d.get('service','unknown')}]" if "service" in d else "")
            for d in docs
        )
    else:
        doc_list = "\n".join(f"- {d['doc_id']} (v{d['version']}): {d['summary']}" for d in docs)

    avg_severity = sum(severities) / len(severities) if severities else 0.0

    code_context = ""
    if is_code_cluster:
        code_context = f"""
    This cluster is primarily composed of CODE documents.
    Languages present: {', '.join(code_languages) if code_languages else 'unknown'}
    Services represented: {', '.join(code_services) if code_services else 'unknown'}

    When generating the label and description, focus on the shared codebase purpose, service domain (e.g., Authentication, Payments), and architectural role if possible.
    Use code metadata to infer the cluster's function (e.g., 'User Management Services', 'Payment Processing Pipelines', 'API Gateway Components').
    """

    prompt = f"""
    You are analysing a governance cluster of type: {cluster_type.upper()}.
    {code_context}
    Documents in this cluster:
    {doc_list}

    Average severity score: {avg_severity:.3f}

    TASK:
    1. Generate a short, human-readable cluster label (5-7 words).
    2. Provide a 1-2 sentence description of the shared theme.
    3. Provide a summary:
       - For RISK clusters: summarise the governance risks and inconsistencies.
       - For TOPIC clusters: summarise the shared subject matter and themes.

    Return ONLY JSON:
    {{
        "label": "...",
        "description": "...",
        "summary": "..."
    }}
    """

    response = call_llm(prompt)
    try:
        data = extract_first_json_block(response)
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse LLM response for cluster labelling: {e}")
        logger.debug(f"Raw response: {response[:500]}")
        data = {
            "label": "Unknown",
            "description": "Failed to generate cluster label",
            "summary": "",
        }
    return data["label"], data["description"], data["summary"]


def generate_cluster_metadata(
    G: nx.Graph, clusters: List[set], cluster_type: str
) -> Dict[int, Dict[str, Any]]:
    """
    Generate LLM-based metadata for all clusters.

    For each cluster, collects documents and severities, then invokes
    llm_label_cluster to generate label, description, and summary.

    Args:
        G: NetworkX graph with node metadata.
        clusters: List of clusters (each cluster is a set of node IDs).
        cluster_type: Either "risk" or "topic".

    Returns:
        Dict mapping cluster_id to metadata dict with keys:
            - label: Short cluster label
            - description: Theme description
            - summary: Detailed summary
            - size: Number of nodes in cluster
    """
    metadata = {}

    for cid, cluster_nodes in enumerate(clusters):
        docs = []
        severities = []
        topics = []
        code_count = 0
        code_languages = set()
        code_services = set()

        for node in cluster_nodes:
            meta = G.nodes[node]
            doc_entry = {
                "doc_id": meta.get("doc_id", node),
                "version": meta.get("version", 1),
                "summary": meta.get("summary", ""),
            }
            # Add code metadata if present
            if meta.get("source_category") == "code":
                code_count += 1
                if "language" in meta:
                    doc_entry["language"] = meta["language"]
                    code_languages.add(meta["language"])
                if "service" in meta:
                    doc_entry["service"] = meta["service"]
                    code_services.add(meta["service"])
            docs.append(doc_entry)

            if cluster_type == "risk":
                for nbr in G.neighbors(node):
                    severities.append(G[node][nbr].get("severity", 0.0))

            topics.append(meta.get("summary", ""))

        # Detect if this is a code cluster (majority code nodes)
        is_code_cluster = code_count >= max(1, len(cluster_nodes) // 2)

        label, description, summary = llm_label_cluster(
            docs=docs,
            severities=severities,
            topics=topics,
            cluster_type=cluster_type,
            is_code_cluster=is_code_cluster,
            code_languages=list(code_languages),
            code_services=list(code_services),
        )

        metadata[cid] = {
            "label": label,
            "description": description,
            "summary": summary,
            "size": len(cluster_nodes),
        }

    return metadata


def build_consistency_graph(
    collection: Collection,
    max_neighbours: int = MAX_NEIGHBOURS,
    sim_threshold: float = SIMILARITY_THRESHOLD,
    workers: int = WORKERS,
    progress_callback: Optional[callable] = None,
    batch_size: int = 500,
    where: Optional[Dict[str, Any]] = None,
    include_documents: bool = True,
    progress_log_interval: int = 10,
    enable_llm_batching: bool = False,
    enable_embedding_cache: bool = False,
    enable_graph_sampling: bool = False,
    graph_sampling_rate: float = 0.1,
    include_advanced_analytics: bool = False,
) -> Dict[str, Any]:
    """
    Build a complete consistency graph with clusters and metadata.

    This is the main entry point for graph building. It orchestrates:
    1. Loading versioned documents from ChromaDB
    2. Building edges in parallel via LLM classification
    3. Computing conflict scores per node
    4. Detecting risk and topic clusters (with overlapping membership)
    5. Generating LLM-based cluster labels and summaries

    Args:
        collection: ChromaDB collection containing versioned documents.
        max_neighbours: Maximum neighbours to compare per document.
        sim_threshold: Similarity threshold for edge creation (distance metric).
        workers: Number of parallel worker threads.
        progress_callback: Optional callback(step, total, message).
        include_advanced_analytics: If True, compute and include advanced graph analytics
            (PageRank, centrality measures, community detection, topology metrics).

    Returns:
        Dict[str, Any]: Complete graph dict with keys:
            - nodes: Dict of node_id -> node attributes (includes cluster assignments)
            - edges: List of edge dicts with relationship metadata
            - clusters: Dict with "risk" and "topic" cluster metadata including
                       LLM-generated labels, descriptions, and summaries
            - analytics: (optional) Advanced analytics if include_advanced_analytics=True

    Raises:
        ValueError: If max_neighbours < 1 or sim_threshold <= 0.
    """
    if max_neighbours < 1:
        raise ValueError("max_neighbours must be >= 1")
    if sim_threshold <= 0:
        raise ValueError("similarity threshold must be > 0")

    versioned_docs = load_versioned_docs(
        collection=collection,
        batch_size=batch_size,
        where=where,
        include_documents=include_documents,
    )
    logger.info(f"Building consistency graph from {len(versioned_docs)} document versions")

    # Use sampling strategy if enabled, otherwise build full graph
    if enable_graph_sampling:
        graph = sample_and_interpolate_graph(
            versioned_docs=versioned_docs,
            doc_collection=collection,
            max_neighbours=max_neighbours,
            sim_threshold=sim_threshold,
            sampling_rate=graph_sampling_rate,
            workers=workers,
            enable_llm_batching=enable_llm_batching,
            enable_embedding_cache=enable_embedding_cache,
            progress_callback=progress_callback,
            progress_log_interval=progress_log_interval,
            logger=logger,
        )
    else:
        graph = build_consistency_graph_parallel(
            versioned_docs=versioned_docs,
            doc_collection=collection,
            max_neighbours=max_neighbours,
            sim_threshold=sim_threshold,
            workers=workers,
            progress_callback=progress_callback,
            progress_log_interval=progress_log_interval,
            enable_llm_batching=enable_llm_batching,
            enable_embedding_cache=enable_embedding_cache,
            enable_graph_sampling=enable_graph_sampling,
            graph_sampling_rate=graph_sampling_rate,
        )

    # Convert to NetworkX for metrics and visualisation
    G = to_networkx(graph)

    # Per-version conflict scores
    compute_version_node_conflict(G)

    # Per-version clusters
    risk_clusters, topic_clusters = compute_version_node_clusters(G)

    # Push metrics back into JSON graph nodes
    for node_id, data in G.nodes(data=True):
        if node_id in graph["nodes"]:
            graph["nodes"][node_id]["conflict_score"] = data.get("conflict_score", 0.0)
            graph["nodes"][node_id]["risk_clusters"] = data.get("risk_clusters", [])
            graph["nodes"][node_id]["topic_clusters"] = data.get("topic_clusters", [])

    # Cluster metadata with labels/descriptions
    risk_meta = generate_cluster_metadata(G, risk_clusters, "risk")
    topic_meta = generate_cluster_metadata(G, topic_clusters, "topic")

    graph["clusters"] = {
        "risk": [
            {
                "id": cid,
                "type": "risk",
                "size": len(comm),
                "members": sorted(list(comm)),
                "label": risk_meta.get(cid, {}).get("label"),
                "description": risk_meta.get(cid, {}).get("description"),
                "summary": risk_meta.get(cid, {}).get("summary"),
            }
            for cid, comm in enumerate(risk_clusters)
        ],
        "topic": [
            {
                "id": cid,
                "type": "topic",
                "size": len(comm),
                "members": sorted(list(comm)),
                "label": topic_meta.get(cid, {}).get("label"),
                "description": topic_meta.get(cid, {}).get("description"),
                "summary": topic_meta.get(cid, {}).get("summary"),
            }
            for cid, comm in enumerate(topic_clusters)
        ],
    }

    # Optionally enrich with advanced analytics
    if include_advanced_analytics:
        try:
            logger.info("Computing advanced graph analytics...")
            analytics = compute_advanced_analytics(G)
            graph["analytics"] = analytics
            logger.info(
                f"Advanced analytics computed: "
                f"PageRank, betweenness, eigenvector centrality, "
                f"relationship strength, topology metrics"
            )
        except Exception as e:
            logger.warning(
                f"Failed to compute advanced analytics: {e}. "
                f"Graph will be returned without analytics."
            )

    return graph


# ============================================================================
# Command-Line Interface
# ============================================================================


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for graph building.

    TODO: Add options for filtering by document type, enabling optimisations, and resetting state.
    TODO: Review arguments, esp binary flags that are opposite of each other (e.g., --include-documents vs --no-documents) for clarity and ease of use.

    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            - sqlite_output: Path to write the graph SQLite database
            - max_neighbours: Maximum neighbours per document
            - similarity_threshold: Distance threshold for edges
            - workers: Number of parallel worker threads
    """
    parser = argparse.ArgumentParser(
        description="Build a version-aware cross-document consistency graph from an existing Chroma collection."
    )
    parser.add_argument(
        "--include-dependency-edges",
        action="store_true",
        default=False,
        help="Add edges between code nodes that share a dependency, service, or API contract (default: False)",
    )
    parser.add_argument(
        "--sqlite-output",
        type=str,
        default=None,
        help="Path to write the graph SQLite database (default: from config CONSISTENCY_GRAPH_SQLITE)",
    )
    parser.add_argument(
        "--max-neighbours",
        type=int,
        default=MAX_NEIGHBOURS,
        help="Maximum neighbours per document to consider",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help="Maximum distance (lower = more similar)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help="Number of worker threads for building the graph edges in parallel",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Log progress every N documents (default: 10, 0 to disable)",
    )
    parser.add_argument(
        "--include-documents",
        action="store_true",
        default=True,
        help="Include full document bodies when loading from Chroma (default: True).",
    )
    parser.add_argument(
        "--no-documents",
        dest="include_documents",
        action="store_false",
        help="Exclude document bodies to reduce memory; relies on summaries/metadata.",
    )
    parser.add_argument(
        "--purge-logs",
        action="store_true",
        default=False,
        help="Purge all consistency graph log files before starting (disabled in Production environment)",
    )
    parser.add_argument(
        "--filter-doc-type",
        type=str,
        default="all",
        choices=["all", "documentation", "code", "java", "groovy", "gradle", "xml", "properties"],
        help="Filter documents by type: all, documentation, code, java, groovy, gradle, xml, properties (default: all)",
    )
    parser.add_argument(
        "--enable-llm-batching",
        action="store_true",
        default=False,
        help="Enable LLM inference batching: group similar documents for batch inference calls (opt-in optimisation)",
    )
    parser.add_argument(
        "--enable-embedding-cache",
        action="store_true",
        default=False,
        help="Enable embedding cache: avoid re-computing embeddings for same documents (opt-in optimisation)",
    )
    parser.add_argument(
        "--enable-graph-sampling",
        action="store_true",
        default=False,
        help="Enable graph sampling: build high-quality graph on subset, interpolate remaining nodes (opt-in optimisation)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=0.1,
        help="Graph sampling rate (0.0-1.0): fraction of nodes to sample for LLM validation (default: 0.1 = 10%%, used only with --enable-graph-sampling)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Delete previous graph artifacts (SQLite files, cache entries, JSON exports) before building",
    )
    parser.add_argument(
        "--include-advanced-analytics",
        action="store_true",
        default=False,
        help="Compute and include advanced graph analytics (PageRank, centrality, community detection, topology metrics)",
    )
    return parser.parse_args()


def remove_sqlite_artifacts(base_path: Optional[str], logger: Any) -> List[Path]:
    """Remove SQLite artifacts for a given base path.

    Args:
        base_path: Base path of the SQLite database (e.g., "consistency_graph.db").
        logger: Logger instance for logging removals and warnings.

    Returns:
        List[Path]: List of Path objects representing the removed files.
    """
    if not base_path:
        return []

    base = Path(base_path)
    candidates = [base, Path(f"{base}.tmp")]
    candidates.extend(Path(f"{base}{suffix}") for suffix in ("-wal", "-shm", "-journal"))

    removed = []
    for candidate in candidates:
        try:
            if candidate.exists():
                candidate.unlink()
                removed.append(candidate)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to remove %s: %s", candidate, exc)
    return removed


def purge_graph_cache_exports(cache_manager: GraphCacheManager, logger: Any) -> int:
    """Delete any cached JSON exports generated from previous builds.

    Args:
        cache_manager: Instance of GraphCacheManager to access the graphs_dir.
        logger: Logger instance for logging removals and warnings.

    Returns:
        int: Count of JSON export files removed.
    """
    removed = 0
    if not cache_manager.graphs_dir.exists():
        return removed

    for json_path in cache_manager.graphs_dir.glob("consistency_graph_*.json"):
        try:
            json_path.unlink()
            removed += 1
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to remove cached graph export %s: %s", json_path, exc)
    return removed


def reset_graph_state(
    cache_manager: GraphCacheManager, output_sqlite: Optional[str], logger: Any
) -> None:
    """Purge cache and output artifacts so the next build starts clean.

    Args:
        cache_manager: Instance of GraphCacheManager to purge cache entries and exports.
        output_sqlite: Base path of the SQLite database to remove artifacts for.
        logger: Logger instance for logging the reset process.
    """
    logger.info("Reset requested: removing previous consistency graph artifacts")

    sqlite_removed = remove_sqlite_artifacts(output_sqlite, logger)
    cache_purged = cache_manager.purge_all()
    json_removed = purge_graph_cache_exports(cache_manager, logger)

    logger.info(
        "Reset removed %d SQLite artifact(s), purged %d cache entry(ies), deleted %d export(s)",
        len(sqlite_removed),
        cache_purged,
        json_removed,
    )


def main() -> None:
    """
    Main entry point for CLI execution.
    
    Builds consistency graph, generates NetworkX visualisation,
    and writes graph JSON to disk.
    
    Command-line usage:
        python build_consistency_graph.py [--output FILE] [--max-neighbours N] \
                                          [--similarity-threshold T] [--workers W] [--purge-logs]
    
    Exit Codes:
        0: Success - graph and visualisation created
        1: Error during execution
    """
    args = parse_args()

    # Handle log purging BEFORE logger initialisation
    # This ensures we purge the old audit log before we write to a new one
    purge_logs_performed = False
    purge_logs_requested = bool(getattr(args, "purge_logs", False))

    if purge_logs_requested:
        if CONFIG.environment == "Prod":
            print("\n[ERROR] Log purging is disabled in Production environment for safety.")
            print("        Current environment: Prod")
            print("        To purge logs, set ENVIRONMENT=Dev or ENVIRONMENT=Test\n")
            if "logger" in locals() and logger:
                flush_all_handlers(logger)
            sys.exit(1)
        else:
            # Purge consistency_graph logs ONLY (not ingest or rag logs)
            logs_dir = Path(__file__).parent.parent.parent / "logs"
            if logs_dir.exists():
                consistency_logs = [
                    "consistency.log",
                    "consistency_audit.jsonl",
                ]

                purged_count = 0
                print(f"\n[PURGE LOGS] Environment: {CONFIG.environment}")
                for log_name in consistency_logs:
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

                print(f"[PURGE LOGS] Removed {purged_count} consistency graph log file(s)\n")
                purge_logs_performed = True

                # Clear the logger cache so get_logger() creates fresh handlers
                from scripts.utils.logger import _loggers

                _loggers.pop("consistency", None)

    # Initialise logger AFTER purging so we don't write to a file we're about to delete
    logger = get_logger()

    # Initialise monitoring for this session
    init_monitoring()
    token_counter = get_token_counter()
    perf_metrics = get_perf_metrics()
    metrics_collector = get_metrics_collector()

    # Log the purge event as the FIRST audit entry if logs were purged
    if purge_logs_performed:
        audit(
            "purge_logs",
            {
                "environment": CONFIG.environment,
                "module": "consistency_graph",
                "user": getpass.getuser(),
                "files_purged": ["consistency.log", "consistency_audit.jsonl"],
            },
        )

    output_sqlite = args.sqlite_output or CONFIG.output_sqlite
    output_sqlite_temp = f"{output_sqlite}.tmp"

    # PROFILING: Cache manager initialisation time
    prof_cache_start = time.time()
    # Initialise graph cache manager
    cache_manager = GraphCacheManager(rag_data_path=CONFIG.rag_data_path)
    prof_cache_time = time.time() - prof_cache_start
    logger.info(f"PROFILING: GraphCacheManager initialisation took {prof_cache_time:.3f}s")

    if getattr(args, "reset", False):
        reset_graph_state(cache_manager, output_sqlite, logger)

    # Build ChromaDB filter for doc type
    where = None
    if args.filter_doc_type != "all":
        if args.filter_doc_type == "documentation":
            where = {"source_category": {"$ne": "code"}}
        elif args.filter_doc_type == "code":
            where = {"source_category": "code"}
        else:
            # Specific language (java, groovy, etc.)
            where = {"source_category": "code", "doc_type": args.filter_doc_type}

    # Create settings dict for cache lookup
    build_settings = {
        "max_neighbours": args.max_neighbours,
        "sim_threshold": args.similarity_threshold,
        "where": where,
        "include_documents": args.include_documents,
    }

    try:
        logger.info("Starting consistency graph build process")
        try:
            args_dict = {k: v for k, v in vars(args).items()}
        except TypeError:
            args_dict = {}

        audit(
            "graph_build_start",
            {
                "output_sqlite": output_sqlite,
                "max_neighbours": args.max_neighbours,
                "similarity_threshold": args.similarity_threshold,
                "workers": args.workers,
                "user": getpass.getuser(),
                "llm_model": VALIDATOR_LLM_MODEL,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "embedding_db_path": CHROMA_PATH,
                "doc_collection": DOC_COLLECTION_NAME,
                "config": CONFIG.to_dict(),
                "args": args_dict,
            },
        )

        # Get collection
        logger.info("Loading ChromaDB collection")
        prof_collection_start = time.time()
        try:
            collection = get_collection()
            prof_collection_time = time.time() - prof_collection_start
            logger.info(
                f"Collection loaded: {DOC_COLLECTION_NAME} (took {prof_collection_time:.3f}s)"
            )
        except Exception as e:
            logger.error(f"Failed to load ChromaDB collection: {e}", exc_info=True)
            audit(
                "graph_build_error",
                {"stage": "collection_load", "error": str(e), "error_type": type(e).__name__},
            )
            print(f"ERROR: Failed to load ChromaDB collection: {e}")
            sys.exit(1)

        # Build graph
        logger.info("Building consistency graph")

        # Track build start time
        build_start_time = time.time()

        # Initialise resource monitoring
        resource_monitor = None
        if CONFIG.enable_resource_monitoring:
            resource_monitor = ResourceMonitor(
                operation_name="consistency_graph_build",
                interval=CONFIG.resource_monitoring_interval,
                enabled=True,
                monitor_ollama=CONFIG.monitor_ollama,
                monitor_chromadb=CONFIG.monitor_chromadb,
            )
            resource_monitor.start()
            logger.info("Resource monitoring started")

        try:
            graph = build_consistency_graph(
                collection=collection,
                max_neighbours=args.max_neighbours,
                sim_threshold=args.similarity_threshold,
                workers=args.workers,
                include_documents=args.include_documents,
                where=where,
                progress_log_interval=args.progress_interval,
                enable_llm_batching=args.enable_llm_batching,
                enable_embedding_cache=args.enable_embedding_cache,
                enable_graph_sampling=args.enable_graph_sampling,
                graph_sampling_rate=args.sampling_rate,
                include_advanced_analytics=args.include_advanced_analytics,
            )
            build_duration = time.time() - build_start_time
            logger.info(
                f"Graph built: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges (took {build_duration:.1f}s)"
            )
        except Exception as e:
            logger.error(f"Failed to build consistency graph: {e}", exc_info=True)
            audit(
                "graph_build_error",
                {"stage": "graph_construction", "error": str(e), "error_type": type(e).__name__},
            )
            print(f"ERROR: Failed to build consistency graph: {e}")
            sys.exit(1)
        finally:
            # Stop resource monitoring
            if resource_monitor:
                resource_monitor.stop()
                resource_monitor.print_summary()
                stats_file = resource_monitor.export_json()
                logger.info(f"Resource statistics exported to {stats_file}")

        # Write to temporary SQLite database
        logger.info(f"Writing graph to temporary SQLite database: {output_sqlite_temp}")
        try:
            with SQLiteGraphWriter(output_sqlite_temp, replace=True) as writer:
                # Insert all nodes
                writer.insert_nodes_batch(graph["nodes"])
                logger.info(f"Inserted {len(graph['nodes'])} nodes")

                # Insert all edges
                writer.insert_edges_batch(graph["edges"])
                logger.info(f"Inserted {len(graph['edges'])} edges")

                # Insert clusters
                writer.insert_clusters(graph["clusters"])
                logger.info(
                    f"Inserted {len(graph['clusters']['risk'])} risk clusters, {len(graph['clusters']['topic'])} topic clusters"
                )

                # Store build metadata
                build_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                writer.set_build_metadata("built_at", build_timestamp)
                writer.set_build_metadata("build_duration_seconds", f"{int(build_duration)}")
                writer.set_build_metadata("max_neighbours_used", args.max_neighbours)
                writer.set_build_metadata("similarity_threshold", args.similarity_threshold)
                writer.set_build_metadata("total_nodes", len(graph["nodes"]))
                writer.set_build_metadata("total_edges", len(graph["edges"]))

                logger.info("Atomically swapping temp DB to production path")
                # Atomic swap: closes temp DB, copies to production path
                writer.atomic_swap(output_sqlite)

            logger.info(f"Graph successfully written to {output_sqlite}")

            audit(
                "graph_build_complete",
                {
                    "output_sqlite": output_sqlite,
                    "nodes": len(graph["nodes"]),
                    "edges": len(graph["edges"]),
                    "risk_clusters": len(graph["clusters"]["risk"]),
                    "topic_clusters": len(graph["clusters"]["topic"]),
                    "dashboard_mode": False,
                },
            )
        except Exception as e:
            logger.error(f"Failed to write SQLite graph: {e}", exc_info=True)
            audit(
                "graph_build_error",
                {"stage": "sqlite_export", "error": str(e), "error_type": type(e).__name__},
            )
            print(f"ERROR: Failed to write SQLite graph: {e}")
            sys.exit(1)

        # Success summary
        print(f"\n✓ Consistency graph built successfully")
        print(f"  SQLite: {output_sqlite}")
        print(f"  Nodes: {len(graph['nodes'])}, Edges: {len(graph['edges'])}")
        print(
            f"  Risk Clusters: {len(graph['clusters']['risk'])}, Topic Clusters: {len(graph['clusters']['topic'])}"
        )
        logger.info("Graph build process completed successfully")
        flush_all_handlers(logger)

    except KeyboardInterrupt:
        logger.warning("Graph build interrupted by user")
        flush_all_handlers(logger)
        audit("graph_build_interrupted", {})
        print("\nGraph build interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during graph build: {e}", exc_info=True)
        flush_all_handlers(logger)
        audit(
            "graph_build_error",
            {"stage": "unexpected", "error": str(e), "error_type": type(e).__name__},
        )
        print(f"ERROR: Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Close cache manager to prevent resource leaks
        if "cache_manager" in locals():
            cache_manager.close()


if __name__ == "__main__":
    main()
