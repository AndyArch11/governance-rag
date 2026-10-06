"""Configuration singleton for the consistency graph module.

Provides centralised, environment driven settings for building and displaying
the consistency graph. Loads `.env` with `override=False` for production friendliness.

Usage:
    from consistency_config import get_consistency_config
    cfg = get_consistency_config()
    print(cfg.output_json)
"""

import os
from pathlib import Path
from typing import Optional

from scripts.utils.config import BaseConfig
from scripts.utils.llm_model_config import get_configured_context_window, get_configured_llm_model


class ConsistencyConfig(BaseConfig):
    """Centralised configuration for consistency graph operations.

    Environment variables (selected):
      - CONSISTENCY_MAX_NEIGHBOURS: Neighbourhood size per node (default: 20)
      - CONSISTENCY_SIMILARITY_THRESHOLD: Edge creation similarity threshold (default: 0.4)
      - CONSISTENCY_WORKERS: Parallel workers for graph build (default falls back to MAX_WORKERS or 4)
      - CONSISTENCY_LOUVAIN_SEED: Seed for Louvain clustering (default: 42)
    - CONSISTENCY_LLM_MODEL: Supported LLM model for graph comparisons
    - CONSISTENCY_CONTEXT_WINDOW_TOKENS: Requested context window for comparisons
      - RAG_DATA_PATH, CHUNK_COLLECTION_NAME, DOC_COLLECTION_NAME: Storage + collections
    """

    def __init__(self) -> None:
        # Initialise base config (loads .env)
        super().__init__()

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

        # Output paths - default to rag_data/consistency_graphs/ directory
        # Primary output is SQLite DB;
        self.output_sqlite = self.get_str(
            "CONSISTENCY_GRAPH_SQLITE",
            str(Path(self.rag_data_path) / "consistency_graphs" / "consistency_graph.sqlite"),
        )

        # Graph build parameters
        self.max_neighbours = self.get_int("CONSISTENCY_MAX_NEIGHBOURS", 20)
        self.similarity_threshold = self.get_float("CONSISTENCY_SIMILARITY_THRESHOLD", 0.4)

        # Performance optimisation settings
        self.gpu_concurrency = self.get_int("CONSISTENCY_GPU_CONCURRENCY", 2)
        self.enable_heuristic_filter = self.get_bool("CONSISTENCY_ENABLE_HEURISTIC_FILTER", True)

        # Advanced optimisation flags (opt-in)
        self.enable_llm_batching = self.get_bool("CONSISTENCY_ENABLE_LLM_BATCHING", False)
        self.enable_embedding_cache = self.get_bool("CONSISTENCY_ENABLE_EMBEDDING_CACHE", False)
        self.enable_graph_sampling = self.get_bool("CONSISTENCY_ENABLE_GRAPH_SAMPLING", False)
        self.graph_sampling_rate = self.get_float(
            "CONSISTENCY_GRAPH_SAMPLING_RATE", 0.1
        )  # 10% sample by default

        # Workers: prefer module‑specific override, else MAX_WORKERS, else 4
        workers_env = os.getenv("CONSISTENCY_WORKERS")
        if workers_env is None:
            workers_env = os.getenv("MAX_WORKERS", "4")
        self.workers = int(workers_env)

        # Clustering determinism
        self.louvain_seed = self.get_int("CONSISTENCY_LOUVAIN_SEED", 42)

        # LLM configuration
        self.llm_model_name = get_configured_llm_model(self, "CONSISTENCY_LLM_MODEL")
        self.llm_context_window_tokens = get_configured_context_window(
            self,
            self.llm_model_name,
            "CONSISTENCY_CONTEXT_WINDOW_TOKENS",
        )


_CONSISTENCY_CONFIG: Optional[ConsistencyConfig] = None


def get_consistency_config() -> ConsistencyConfig:
    """Return a process-wide singleton of `ConsistencyConfig`."""
    global _CONSISTENCY_CONFIG
    if _CONSISTENCY_CONFIG is None:
        _CONSISTENCY_CONFIG = ConsistencyConfig()
    return _CONSISTENCY_CONFIG
