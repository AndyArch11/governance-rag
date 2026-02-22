"""Configuration singleton for the ingestion pipeline.

Centralises ingestion settings with environment variable overrides and
production friendly dotenv loading.

Usage:
    from ingest_config import get_ingest_config
    cfg = get_ingest_config()
    print(cfg.rag_data_path)
"""

import argparse
import threading
from pathlib import Path
from typing import Any, Optional, Set

from scripts.utils.config import BaseConfig
from scripts.utils.logger import get_logger


class IngestConfig(BaseConfig):
    """Centralised configuration for document ingestion.

    Loads settings from environment variables with sensible defaults and caches
    them in memory. `.env` is loaded with `override=False` to avoid clobbering
    real environment values in production.

    Environment variables (selected):
      - RAG_BASE_PATH: Root directory of source documents
      - RAG_DATA_PATH: ChromaDB persistent storage location
      - CHUNK_COLLECTION_NAME, DOC_COLLECTION_NAME: Chroma collections
      - URL_SEED_JSON_PATH, URL_DOWNLOAD_DIR: URL seed ingestion
      - LLM_CACHE_ENABLED, LLM_CACHE_PATH, LLM_CACHE_MAX_AGE_DAYS
      - ENABLE_SEMANTIC_DRIFT_DETECTION
      - ENABLE_CHUNK_HEURISTIC_SKIP
      - MAX_WORKERS, PROGRESS_LOG_INTERVAL, LLM_RATE_LIMIT
      - INGEST_LLM_MODEL, INGEST_VALIDATOR_LLM_MODEL
      - EMBEDDING_BATCH_SIZE, EMBEDDING_CACHE_ENABLED
      - PRESERVE_DOMAIN_KEYWORDS: Comma-separated keywords to preserve
      - DOMAIN_KEYWORDS_FILE: JSON file with a list of keywords
      - ENABLE_CANDIDATE_TERMS: Record candidate terms from ingestion (default: false)
      - CANDIDATE_TERMS_DOMAIN: Optional domain for candidate terms (e.g., "legal")
    """

    def __init__(self) -> None:
        super().__init__()

        # Paths
        self.base_path = self.get_path(
            "RAG_BASE_PATH",
            str(self.project_root / "data_raw" / "downloads"),
        )

        # URL seed ingestion
        self.url_seed_json_path = self.get_path(
            "URL_SEED_JSON_PATH",
            str(self.project_root / "data_raw" / "url_seeds.json"),
        )
        self.url_download_dir = self.get_str(
            "URL_DOWNLOAD_DIR",
            str(Path(self.base_path) / "url_imports"),
        )

        # LLM cache - stored in cache subdirectory
        self.llm_cache_enabled = self.get_bool("LLM_CACHE_ENABLED", True)
        self.llm_cache_path = self.get_str(
            "LLM_CACHE_PATH",
            str(Path(self.rag_data_path) / "cache" / "llm_cache.json"),
        )
        self.llm_cache_max_age_days = self.get_int("LLM_CACHE_MAX_AGE_DAYS", 0)

        # Semantic drift detection
        self.enable_semantic_drift_detection = self.get_bool(
            "ENABLE_SEMANTIC_DRIFT_DETECTION", True
        )

        # Chunk heuristic skip for high‑confidence chunks
        self.enable_chunk_heuristic_skip = self.get_bool("ENABLE_CHUNK_HEURISTIC_SKIP", True)

        # Parent-child chunking for improved context retrieval
        self.enable_parent_child_chunking = self.get_bool("ENABLE_PARENT_CHILD_CHUNKING", True)

        # Performance / throughput
        self.max_workers = self.get_int("MAX_WORKERS", 4)
        self.progress_log_interval = self.get_int("PROGRESS_LOG_INTERVAL", 10)
        self.llm_rate_limit = self.get_float("LLM_RATE_LIMIT", 10.0)

        # LLM configuration
        self.llm_model_name = self.get_str("INGEST_LLM_MODEL", "mistral")
        self.validator_llm_model_name = self.get_str(
            "INGEST_VALIDATOR_LLM_MODEL",
            self.llm_model_name,
        )

        # Embedding generation
        self.embedding_batch_size = self.get_int("EMBEDDING_BATCH_SIZE", 32)
        self.embedding_cache_enabled = self.get_bool("EMBEDDING_CACHE_ENABLED", True)
        self.embedding_cache_path = self.get_str(
            "EMBEDDING_CACHE_PATH",
            str(Path(self.rag_data_path) / "cache" / "embedding_cache.json"),
        )

        # BM25 keyword indexing (for hybrid search)
        self.bm25_indexing_enabled = self.get_bool("BM25_INDEXING_ENABLED", True)
        self.bm25_index_original_text = self.get_bool("BM25_INDEX_ORIGINAL_TEXT", True)

        # Cache configuration
        self.cache_enabled = self.get_bool("RAG_CACHE_ENABLED", True)
        self.cache_path = self.get_path(
            "RAG_CACHE_PATH",
            str(Path(self.rag_data_path) / "cache.db"),
        )

        # Version management
        self.versions_to_keep = self.get_int("VERSIONS_TO_KEEP", 3)

        # Emergency flag to reset storage (prefer using --reset argument instead)
        self.reinitialise_chroma_storage = self.get_bool("REINITIALISE_CHROMA_STORAGE", False)

        # Deployment environment (Dev, Test, Prod)
        self.environment = self.get_str("ENVIRONMENT", "Dev")
        if self.environment not in ("Dev", "Test", "Prod"):
            raise ValueError(
                f"Invalid ENVIRONMENT '{self.environment}'. Must be one of: Dev, Test, Prod"
            )

        # Resource monitoring
        self.enable_resource_monitoring = self.get_bool("ENABLE_RESOURCE_MONITORING", False)
        self.resource_monitoring_interval = self.get_float("RESOURCE_MONITORING_INTERVAL", 1.0)
        self.monitor_ollama = self.get_bool("MONITOR_OLLAMA", True)
        self.monitor_chromadb = self.get_bool("MONITOR_CHROMADB", True)

        # Regex pattern to ignore/filter files by name (e.g., render files, temp files)
        self.ignore_file_regex = self.get_str("IGNORE_FILE_REGEX", r"render\(\d\)\.html")

        # Domain keywords preservation
        self.preserve_domain_keywords: Optional[Set[str]] = None
        keywords_env = self.get_str("PRESERVE_DOMAIN_KEYWORDS", "").strip()
        keywords_file = self.get_str("DOMAIN_KEYWORDS_FILE", "").strip()

        keywords: Set[str] = set()
        if keywords_env:
            for kw in keywords_env.split(","):
                kw = kw.strip()
                if kw:
                    keywords.add(kw)

        if keywords_file:
            try:
                import json

                p = Path(keywords_file)
                if p.exists():
                    data = json.loads(p.read_text())
                    if isinstance(data, list):
                        for kw in data:
                            if isinstance(kw, str) and kw.strip():
                                keywords.add(kw.strip())
                else:
                    # Log missing file for visibility
                    get_logger("ingest").warning(f"DOMAIN_KEYWORDS_FILE not found: {keywords_file}")
            except Exception as e:
                # Log error instead of failing silently
                get_logger("ingest").warning(
                    f"Failed to load DOMAIN_KEYWORDS_FILE '{keywords_file}': {type(e).__name__}: {e}"
                )

        if keywords:
            self.preserve_domain_keywords = set(keywords)

        # Candidate term recording for unfamiliar domains
        self.enable_candidate_terms = self.get_bool("ENABLE_CANDIDATE_TERMS", False)
        candidate_domain = self.get_str("CANDIDATE_TERMS_DOMAIN", "").strip()
        self.candidate_terms_domain: Optional[str] = candidate_domain or None

        # Runtime attributes (assigned in main())
        self.logger: Optional[Any] = None  # IngestLogger instance
        self.args: Optional[argparse.Namespace] = None  # CLI arguments
        self.include_url_seeds: bool = False  # Set from CLI or config
        self.version_lock: Optional[threading.Lock] = None  # Threading lock for version assignment


_INGEST_CONFIG: Optional[IngestConfig] = None


def build_cli_overrides(args: argparse.Namespace) -> dict:
    """Build config override mapping from CLI args.

    Maps CLI arguments to environment variable names so BaseConfig
    can apply the priority: CLI > system env > .env > defaults.
    """
    overrides = {}

    if getattr(args, "workers", None) is not None:
        overrides["MAX_WORKERS"] = args.workers

    if getattr(args, "url_seed_path", None):
        overrides["URL_SEED_JSON_PATH"] = args.url_seed_path

    if getattr(args, "progress_interval", None) is not None:
        overrides["PROGRESS_LOG_INTERVAL"] = args.progress_interval

    return overrides


def get_ingest_config(overrides: Optional[dict] = None, reset: bool = False) -> IngestConfig:
    """Return a process-wide singleton of `IngestConfig`.

    Args:
        overrides: Optional config overrides (highest priority).
        reset: If True, rebuild the singleton using provided overrides.

    Returns:
        IngestConfig singleton instance
    """
    global _INGEST_CONFIG
    if overrides:
        IngestConfig.set_overrides(overrides)
    if _INGEST_CONFIG is None or reset:
        _INGEST_CONFIG = IngestConfig()
    return _INGEST_CONFIG
