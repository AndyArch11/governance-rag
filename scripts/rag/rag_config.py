"""Configuration for RAG retrieval and generation pipeline.

Manages centralised configuration with environment variable overrides and sensible
defaults. All settings are read at startup and cached in the RAGConfig singleton.
For production deployments, set environment variables before launching the service.
"""

from typing import Optional

from scripts.utils.config import BaseConfig
from scripts.utils.llm_model_config import (
    get_configured_context_window,
    get_configured_llm_model,
)


class RAGConfig(BaseConfig):
    """Centralised configuration for RAG operations.

    Loads configuration from environment variables with sensible defaults.
    All settings are cached in memory at instantiation; for production,
    set environment variables before launching the application.

    Attributes:
        rag_data_path (str): ChromaDB persistent storage location.
        chunk_collection_name (str): Name of chunk collection in ChromaDB.
        doc_collection_name (str): Name of document collection in ChromaDB.
        k_results (int): Default number of chunks to retrieve per query.
        model_name (str): LLM model name (passed to OllamaLLM).
        temperature (float): LLM temperature for generation (0.0-1.0).
        context_window_tokens (int): Requested Ollama context window, capped by the model limit.
        max_prompt_tokens (int): Maximum token budget for prompt input.
        max_context_chars (int): Character budget for retrieved context within prompt input.
        logs_dir (Path): Directory for log files (created if missing).

    Environment Variables (with defaults):
        RAG_DATA_PATH: ChromaDB data directory (default: <project_root>/rag_data)
        CHUNK_COLLECTION_NAME: Chunk collection name (default: governance_docs_chunks)
        DOC_COLLECTION_NAME: Document collection name (default: governance_docs_documents)
        RAG_K_RESULTS: Chunks to retrieve (default: 5, must be >= 1)
        RAG_MODEL: Supported Ollama LLM model name (default: qwen2.5:14b-instruct-q4_K_M)
        RAG_TEMPERATURE: LLM temperature (default: 0.3, range 0.0-1.0)
        RAG_CONTEXT_WINDOW_TOKENS: Requested Ollama context window (default: 8192)
        RAG_RESPONSE_TOKEN_RESERVE: Tokens reserved for generation (default: 1024)
        RAG_MAX_PROMPT_TOKENS: Optional lower cap for prompt input (default: model-aware)

    Example:
        >>> config = RAGConfig()  # Reads env vars
        >>> print(config.model_name)  # qwen2.5:14b-instruct-q4_K_M unless RAG_MODEL set
        >>> config.k_results
        5
        >>> config.context_window_tokens
        8192

    Note: Temperature parameter is loaded but not currently applied by generate.py
    due to LLM being initialised at module import. This is a known limitation.
    """

    def __init__(self) -> None:
        """Initialise configuration from environment variables with defaults.
        Reads all RAG settings from environment or uses sensible defaults.
        Creates logs directory if it doesn't exist.
        """
        # Initialise base config (loads .env)
        super().__init__()

        # Deployment environment (Dev, Test, Prod)
        self.environment = self.get_str("ENVIRONMENT", "Dev")
        if self.environment not in ("Dev", "Test", "Prod"):
            raise ValueError(
                f"Invalid ENVIRONMENT '{self.environment}'. Must be one of: Dev, Test, Prod"
            )

        # Retrieval parameters
        self.k_results = self.get_int("RAG_K_RESULTS", 5)

        # LLM configuration
        self.model_name = get_configured_llm_model(self, "RAG_MODEL")
        self.temperature = self.get_float("RAG_TEMPERATURE", 0.3)

        # Context budgeting is limited by verified model capabilities. The
        # requested runtime window remains deliberately conservative by default.
        self.context_window_tokens = get_configured_context_window(
            self,
            self.model_name,
            "RAG_CONTEXT_WINDOW_TOKENS",
        )
        self.response_token_reserve = min(
            max(256, self.get_int("RAG_RESPONSE_TOKEN_RESERVE", 1024)),
            self.context_window_tokens - 1,
        )
        available_prompt_tokens = self.context_window_tokens - self.response_token_reserve
        configured_prompt_tokens = self.get_int("RAG_MAX_PROMPT_TOKENS", 0)
        self.max_prompt_tokens = (
            min(configured_prompt_tokens, available_prompt_tokens)
            if configured_prompt_tokens > 0
            else available_prompt_tokens
        )

        # Prompt assembly includes labelled and raw context representations.
        # Reserve the other half of the input budget for that duplication,
        # prompts, and the user question. One token is conservatively four chars.
        self.max_context_chars = (self.max_prompt_tokens * 4) // 2

        # Phase 3 Enhancement Features
        # Enable context caching for hot entities (100x faster retrieval)
        self.enable_caching = self.get_bool("RAG_ENABLE_CACHING", True)

        # Enable graph-enhanced retrieval (relationship-based expansion)
        self.enable_graph = self.get_bool("RAG_ENABLE_GRAPH", True)

        # Enable parent-child chunking (precise search, rich context)
        self.enable_parent_child = self.get_bool("RAG_ENABLE_PARENT_CHILD", True)

        # Enable hybrid search (vector + keyword matching for better retrieval)
        self.enable_hybrid_search = self.get_bool("RAG_ENABLE_HYBRID_SEARCH", True)

        # Context cache settings
        self.cache_max_entries = self.get_int("RAG_CACHE_MAX_ENTRIES", 100)
        self.cache_ttl_seconds = self.get_int("RAG_CACHE_TTL_SECONDS", 3600)  # 1 hour

        # Graph expansion settings
        self.graph_max_hops = self.get_int("RAG_GRAPH_MAX_HOPS", 1)
        self.graph_max_neighbours = self.get_int("RAG_GRAPH_MAX_NEIGHBOURS", 3)

        # Graph SQLite database path (defaults to consistency graph location)
        from pathlib import Path

        self.graph_sqlite_path = self.get_str(
            "RAG_GRAPH_SQLITE_PATH",
            str(Path(self.rag_data_path) / "consistency_graphs" / "consistency_graph.sqlite"),
        )
        self.enable_thesis_graph = self.get_bool("RAG_ENABLE_THESIS_GRAPH", False)
        self.thesis_graphs_dir = self.get_path("RAG_THESIS_GRAPHS_DIR", "rag_data/thesis_graphs")
        self.thesis_graph_max_chunks = self.get_int("RAG_THESIS_GRAPH_MAX_CHUNKS", 3)

        # ross-encoder reranking for improved relevance
        self.enable_learned_reranking = self.get_bool("RAG_ENABLE_LEARNED_RERANKING", True)
        self.reranker_model = self.get_str("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")
        self.rerank_top_k = self.get_int("RAG_RERANK_TOP_K", 50)  # Rerank top N from hybrid search
        self.reranker_device = self.get_str("RAG_RERANKER_DEVICE", "cpu")  # Can also be cuda or mps
        self.reranker_batch_size = self.get_int("RAG_RERANKER_BATCH_SIZE", 32)
        self.enable_reranker_cache = self.get_bool("RAG_ENABLE_RERANKER_CACHE", True)
        self.reranker_strict_offline = self.get_bool("RAG_RERANKER_STRICT_OFFLINE", False)

        # Resource monitoring
        self.enable_resource_monitoring = self.get_bool("ENABLE_RESOURCE_MONITORING", False)
        self.resource_monitoring_interval = self.get_float("RESOURCE_MONITORING_INTERVAL", 1.0)
        self.monitor_ollama = self.get_bool("MONITOR_OLLAMA", True)
        self.monitor_chromadb = self.get_bool("MONITOR_CHROMADB", True)

        # Phase 6 Step 3: Auto-tuning Features
        # Enable adaptive rate limiting for API calls (auto-adjusts based on latency/errors)
        self.enable_adaptive_rate_limiting = self.get_bool(
            "RAG_ENABLE_ADAPTIVE_RATE_LIMITING", True
        )

        # Enable adaptive cache tuning (auto-adjusts TTL/eviction based on hit patterns)
        self.enable_adaptive_cache_tuning = self.get_bool("RAG_ENABLE_ADAPTIVE_CACHE_TUNING", True)

        # Hybrid Search Enhancement Features (Phase 7)
        # Vector search weight in hybrid results (0.0-1.0, normalised with keyword_weight)
        self.hybrid_vector_weight = self.get_float("RAG_HYBRID_VECTOR_WEIGHT", 0.6)

        # Keyword search weight in hybrid results (0.0-1.0, normalised with vector_weight)
        self.hybrid_keyword_weight = self.get_float("RAG_HYBRID_KEYWORD_WEIGHT", 0.4)

        # Hybrid result combination strategy: "sum", "rank_fusion", or "top_k"
        self.hybrid_combination_strategy = self.get_str("RAG_HYBRID_COMBINATION_STRATEGY", "sum")

        # Enable query expansion (synonyms, spelling variants, abbreviations)
        self.enable_query_expansion = self.get_bool("RAG_ENABLE_QUERY_EXPANSION", True)

        # Enable US/British spelling variant expansion
        self.enable_spelling_variants = self.get_bool("RAG_ENABLE_SPELLING_VARIANTS", True)

        # Enable domain synonym expansion
        self.enable_domain_synonyms = self.get_bool("RAG_ENABLE_DOMAIN_SYNONYMS", True)
