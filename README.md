# 🏷️ Governance Intelligence Console

The Governance Intelligence Console is an interactive analysis platform that transforms documentation into a version‑aware consistency graph, identifies governance risks, discovers semantic topic clusters, and visualises conflicts, drift, and cluster insights through an intelligent dashboard.

It is designed for technical architects, governance teams, and engineering leaders who need deep visibility into documentation consistency, risk posture, and thematic structure across large document sets, as well as for academics needing to assess academic theses.

Motivation for this work is to serve as a platform to explore various AI related considerations. Created with Copilot in Visual Studio using predominantly Claude Haiku 4.5 as the coding assistant.

## 🚀 Key Features

- **Version‑Aware Consistency Graph** - Builds a graph of document versions using embeddings, similarity edges, and LLM‑based consistency scoring.
- **Risk Clusters** - Severity‑weighted Louvain style clustering with LLM‑generated labels, descriptions, and risk summaries.
- **Topic Clusters** - Semantic similarity clustering with LLM‑generated labels and topic summaries.
- **Interactive Dashboard** - Heatmaps, timelines, cluster exploration, and various filters
- **Rebuild Optimisations** - Cached document loading, progress‑aware graph building, and automatic graph persistence.
- **Context Caching:** LRU + TTL cache for hot entities
- **Graph-Enhanced Retrieval:** Relationship-based chunk expansion
- **Parent-Child Chunking:** Small chunks for search, large chunks for LLM context
- **Persona-Aware Retrieval:** Academic filtering (supervisor/assessor/researcher)

**Ingestion-Time Indexing:**
- BM25 inverted index built automatically during document ingestion
- Indexes chunk text at chunk-level granularity (regular chunks plus parent/child chunks when enabled)
- Pre-computes term frequencies, IDF values, and document lengths
- Stored in SQLite cache database for efficient retrieval
- Storage: ~10-20% of document size in SQLite index

**Testing:** Almost 2,000 tests with ~60% code coverage

### 🔍 Resource Monitoring

**Supports Capacity planning and resource alerting:**

- **Multi-Process Tracking:** Python, Ollama, and ChromaDB resource usage
  - Subject to environment and what can be reported on
  - Graceful degradation without OpenTelemetry
  - Thread-safe metrics collection
- **Metrics:** CPU, RAM, VRAM (GPU), disk I/O (IOPS), network I/O
- **Peak Value Recording:** Track maximum utilisation for infrastructure sizing
- **Export:** Statistics for analysis and trending
- **Flexible Integration:** Context managers, decorators, or manual control

**Use Cases:**
- Server capacity planning and infrastructure sizing
- Container resource limit configuration (Docker/K8s)
- Performance optimisation and regression detection
- Resource alert threshold configuration

**Quick Start:**
Place files to ingest in `rag-project/data_raw/downloads`

```bash
# Enable in .env
ENABLE_RESOURCE_MONITORING=true

# Activate project environment
cd ~/rag-project
source .venv/bin/activate

# Run with monitoring
python scripts/ingest/ingest.py  # Auto-captures stats
```
**Integration Points:**
- RAG pipeline: `query.py`, `generate.py`, `retrieve.py` - LLM latency and token tracking
- Ingestion: `ingest.py`, `ingest_git.py`, `ingest_academic.py`, `vector.py` - Processing metrics
- Consistency graph: `build_consistency_graph.py` - Graph building performance metrics

### 🎯 Adaptive Auto-Tuning ✅

**Automatic optimisation of rate limiting and cache parameters:**

**Adaptive Rate Limiting:**
- Automatic concurrency adjustment based on latency (target: p95 < 500ms)
- Rate limit backoff on 429/503 responses
- Gradual recovery when system is healthy
- Error rate monitoring (target: < 1%)

**Adaptive Cache Tuning:**
- TTL optimisation based on hit/miss patterns (target: 80%+ hit rate)
- Eviction policy recommendations (LRU, LFU, or HYBRID)
- Memory usage monitoring with threshold alerts
- Per-type tuning (embeddings vs LLM results)

**Integration Points:**
- Embedding generation: `vectors.py` - Auto rate-limit API calls
- LLM generation: `generate.py` - Auto adjust for latency
- Unified cache backend: `scripts/ingest/cache_db.py` - Track hit patterns and cache efficiency for embeddings, LLM results, graph cache, and BM25 tables


## 🧠 Architecture Overview


### Environment loading (dev vs prod)

- Load order: code defaults → `.env` (if present) → real environment variables → CLI args (highest).
- `.env` is for local convenience; production should inject env vars via the runtime (containers/CI/systemd/k8s).
- ⚠️ Do not place secrets in `.env` or commit them. Use your environment or ideally a secrets manager instead.

### Environment Variables

Key variables (see .env.example and the respective `*_config.py` modules for full details and defaults):
- Environment: `ENVIRONMENT` (Dev/Test/Prod) - Controls log purging behaviour and deployment settings.
- Paths: `RAG_BASE_PATH` (`~/rag-project/data_raw/downloads/`), `RAG_DATA_PATH` (`~/rag-project/rag_data`), `URL_SEED_JSON_PATH`, `URL_DOWNLOAD_DIR`.
- Collections: `CHUNK_COLLECTION_NAME` (governance_docs_chunks), `DOC_COLLECTION_NAME` (governance_docs_documents).
- Caching: `RAG_CACHE_ENABLED` (true), `RAG_CACHE_PATH` (`~/rag-project/rag_data/cache.db`), `LLM_CACHE_ENABLED` (true), `EMBEDDING_CACHE_ENABLED` (true), `LLM_CACHE_MAX_AGE_DAYS` (0).
- Quality/versions: `ENABLE_SEMANTIC_DRIFT_DETECTION` (true), `ENABLE_CHUNK_HEURISTIC_SKIP` (true), `VERSIONS_TO_KEEP` (3).
- Performance: `MAX_WORKERS` (4), `LLM_RATE_LIMIT` (10.0), `EMBEDDING_BATCH_SIZE` (32), `PROGRESS_LOG_INTERVAL` (10).
- Clean-up: `REINITIALISE_CHROMA_STORAGE` (false).

Legacy compatibility keys such as `LLM_CACHE_PATH` and `EMBEDDING_CACHE_PATH` may still appear in older configs, but active cache storage is SQLite `cache.db`.

Do not store secrets in .env; inject them via the real environment or a secrets manager.

The system is composed of three major layers:

### 1. Ingestion Layer

Loads documents into ChromaDB, including metadata, embeddings, and ingestion version history.

### 2. Graph Builder

Constructs the consistency graph by:

- Computing similarity edges
- Running LLM consistency checks
- Assigning severity
- Generating risk and topic clusters
- Producing LLM‑generated cluster metadata
- Emitting a normalised JSON graph

### 3. Dashboard

A Dash application that visualises:

- Document Similarity Relationships
- Conflict heatmaps
- Version timelines
- Document comparisons
- Cluster insights
- Drift patterns

### Process Flow

                        ┌───────────────────────────────────┐
                        │         Raw Documents             │
                        │ (HTML and PDF files, source code) │
                        └────────────────┬──────────────────┘
                                         │
                                         ▼
                        ┌───────────────────────────────────┐
                        │     Ingestion & Preprocessing     │
                        │    (ChromaDB: text + metadata)    │
                        │-----------------------------------│
                        │                                   │
                        │  - Load versions & metadata       │
                        │  - Load embeddings                │
                        │  - Normalise document structure   │
                        └────────────────┬──────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────────┐
                  │             Consistency Graph Builder            │
                  │                  (SQLite DB)                     │
                  │--------------------------------------------------│
                  │  Similarity Engine                               │
                  │   - Compute cosine similarity                    │
                  │   - Build similarity edges                       │
                  │                                                  │
                  │  LLM Consistency Engine                          │
                  │   - Evaluate semantic consistency                │
                  │   - Assign severity scores                       │
                  │                                                  │
                  │  Graph Construction                              │
                  │   - Create nodes (doc versions)                  │
                  │   - Add edges (similarity + severity)            │
                  └──────────────────────┬───────────────────────────┘
                                         │
                                         ▼
         ┌──────────────────────────────────────────────────────────────────┐
         │                      Cluster Identification                      │
         │------------------------------------------------------------------│
         │  Risk Clusters (severity-weighted Louvain like)                  │
         │   - Detect governance risk groups                                │
         │   - LLM label: risk label + description + risk summary           │
         │                                                                  │
         │  Topic Clusters (semantic similarity)                            │
         │   - Detect thematic groups                                       │
         │   - LLM label: topic label + description + topic summary         │
         │                                                                  │
         │  Normalised Cluster Metadata                                     │
         │   - Stored once in graph.graph["clusters"]                       │
         │   - Document nodes reference cluster IDs                         │
         └────────────────────────────────┬─────────────────────────────────┘
                                          │
                                          ▼
                   ┌──────────────────────────────────────────┐
                   │  consistency_graph.sqlite (normalised)   │
                   │------------------------------------------│
                   │  - nodes                                 │
                   │  - edges                                 │
                   │  - clusters:                             │
                   │      • risk                              │
                   │      • topic                             │
                   └──────────────────────┬───────────────────┘
                                          │
                                          ▼
                  ┌─────────────────────────────────────────────┐
                  │                Dash Dashboard               │
                  │---------------------------------------------│
                  │  Heatmap                                    │
                  │   - Conflict severity matrix                │
                  │   - Cluster-aware ordering                  │
                  │                                             │
                  │  Queries                                    │
                  │   - Templates and Personas                  │
                  │   - Filters                                 │
                  │                                             │
                  │  Graph Analytics                            │
                  │   - Toplogy metrics                         │
                  │   - Clustering distribution                 │
                  │                                             │
                  │  Cluster Overview                           │
                  │   - Risk clusters                           │
                  │   - Topic clusters                          │
                  │   - LLM labels + summaries                  │
                  │                                             │
                  │  Graph Controls                             │
                  │   - Presets                                 │
                  │   - Sliders                                 │
                  │   - Filters                                 │
                  └─────────────────────────────────────────────┘


## 📦 Project Structure

```
rag-project/
│
├── .venv/                              # Python virtual environment
│
├── data_raw/                           # Optional folder from which to ingest documents to process from (subject to env path setting)
│   ├── academic_papers/                # Folder containing pdf theses for processing
│   ├── downloads/                      # Folder containing downloaded html and pdf files for processing
│   ├── public_docs/                    # Folder containing public docs for processing
│   └── url_seeds.json
│
├── docs/
│   └── *.md                            # Topic based README files
│
├── examples/
│   └── *.py                            # Coding examples and various ad-hoc utility files
│
├── logs/
│   ├── resource_stats/                 
│   │   └── resource_stats_*.json       # Resource monitoring statistics (CPU, RAM, VRAM, IOPS, network)
│   │
│   ├── app.log                         # Generated by resource_monitor.py logging resource monitoring activity
│   ├── citation_graph_callbacks.log    # Generated by citation_graph_callbacks.py for debugging dashboard Citation Graph tab
│   ├── citation_graph_viz.log          # Generated by citation_graph_viz.py for debugging dashboard Citation Graph tab
│   ├── consistency_audit.jsonl         # Structured consistency graph trace log generated by build_consistency_graph.py
│   ├── consistency_graph.academic.citation_graph_callbacks.log    # Generated by citation_graph_callbacks.py for debugging dashboard Citation Graph tab
│   ├── consistency_graph.academic.citation_graph_viz.log           # Generated by citation_graph_viz.py for debugging dashboard Citation Graph tab
│   ├── consistency.log                 # Unstructured human readable consistency graph log generated by build_consistency_graph.py
│   ├── embedding.log                   # Unstructured human readable embedding log generated by vectors.py
│   ├── ingest_audit.jsonl              # Structured ingest trace log generated by ingest.py, ingest_git.py, ingest_academic.py
│   ├── ingest.log                      # Unstructured human readable ingest log generated by ingest.py, ingest_git.py, ingest_academic.py
│   ├── perf_metrics.log                # Structured log generated by dashboard.py, entries missing creation timestamps
│   ├── rag_audit.jsonl                 # Structured rag trace log generated by retrieve.py and generate.py
│   ├── rag.log                         # Unstructured human readable rag log generated by retrieve.py and generate.py
│   ├── retry_audit.jsonl               # Structured log recording retries and failures generated by retry_utils.py
│   └── retry.log                       # Unstructured log recording retries and failures generated by retry_utils.py
│
├── notebooks/
│   └── *.ipynb                         # Various Jupyter notebooks to assess health of processing activities
│
├── rag_data/                           # Data storage (subject to env RAG_DATA_PATH setting)
│   ├── academic_pdfs/                  # downloaded citations/reference data
│   │
│   ├── chromadb/                       # ChromaDB persistent storage
│   │   ├── chroma.sqlite3              # ChromaDB database
│   │   └── ...                         # ChromaDB index files
│   │
│   ├── consistency_graphs/             # Consistency graph outputs
│   │   └── consistency_graph.sqlite    # Primary graph store used by the dashboard
│   │
│   ├── domain_terms/                   # Terms that are specific to a domain that should be emphasised during indexing/searches
│   │   └── *.json                      # Domain specific files
│   │
│   ├── learning/                       # Adaptive learning based on search response feedback
│   │   ├── last_recommendation.json    # Records last recommendation used
│   │   └── weight_samples.json         # Tracks query configuration and relevancy of response
│   │
│   ├── academic_citation_graph.db      # Academic citation SQLite database
│   ├── academic_citation_graph.json    # Academic citation JSON file
│   ├── academic_terminology.db         # Academic terminology SQLite database
│   │
│   ├── benchmarks.db                   # Benchmarking database
│   │
│   ├── cache.db                        # Various caches such as embedding and llm caches
│   │
│   ├── conversations.db                # Query conversations database
│   │
│   ├── hybrid_search_weights.json      # vector vs keyword weights and combination strategy for queries
│   │
│   └── query_templates.db              # templates to support queries
│
├── repos/                              # Used by ingest_git.py to clone repos to parse.
│   ├── bitbucket/                      # Bitbucket files
│   │   └── git slug                    # git slug path to cloned files
│   └── github/                         # Github files
│       └── git slug                    # git slug path to cloned files
│
├── scripts/
│   │
│   ├── consistency_graph/
│   │   ├── advanced_analytics.py       # utiltity module with various analytics methods
│   │   ├── build_consistency_graph.py  # Builds the consistency graph, establishing relationships between documents
│   │   ├── consistency_config.py       # Utility class for loading config values
│   │   ├── graph_cache_manager.py      # Cache management for a consistency graph load
│   │   ├── graph_filter.py             # Convenience module for querying consistency graph db
│   │   ├── sqlite_schema.py            # Consistency graph schema
│   │   ├── sqlite_store.py             # Supports connection pooling and paging of consistency graph db
│   │   └── sqlite_writer.py            # Supports progressinve writes and atomic swap of consistency graph db
│   │ 
│   ├── ingest/
│   │   ├── academic/                   # ingest_academic support modules
│   │   │   ├── providers/              # Various online technical paper reference sites
│   │   │   │   └── *.py                # Manages connections to the corresponding site
│   │   │   └── *.py                    # Academic specific modules
│   │   ├── git/                        # ingest_git support modules
│   │   │   └── *.py                    # Git specific modules
│   │   ├── cache_db.py                 # Cache management during ingestion - graph cache, embedding cache, llm cache, document cache, bm25 index
│   │   ├── chromadb_sqlite.py          # Drop in replacement of ChromaDB using a light weight sqlite DB instead
│   │   ├── advanced_analytics.py       # utiltity module with various analytics methods
│   │   ├── chunk.py                    # Senmantic aware text chunking
│   │   ├── embedding_cache.py          # Cache for embedding vectors
│   │   ├── htmlparser.py               # Parses html files
│   │   ├── ingest_academic.py          # Ingests PDF thesis files into the ChromaDB for RAG queries and thesis assessments
│   │   ├── ingest_git.py               # Ingests files from Git providers (Bitbucket/GitHub) into the ChromaDB for RAG queries
│   │   ├── ingest.py                   # Ingests standard html/pdf files into the ChromaDB for RAG queries
│   │   ├── llm_cache.py                # Cache for LLM calls
│   │   ├── academic/                   # Academic ingestion helpers
│   │   ├── pdfparser.py                # Parses pdf files (WIP)
│   │   ├── preprocess.py               # Text processing and LLM metadata
│   │   ├── vectors.py                  # Saves the embedding vectors and metadata to the Chroma DB
│   │   └── word_frequency.py           # calculates word frequency of ingested documents
│   │ 
│   ├── rag/
│   │   ├── adaptive_weighting.py       # Adaptive learning for hybrid search
│   │   ├── assemble.py                 # Constructs prompts for RAG generation
│   │   ├── benchmark_manager.py        # Benchmarks responsiveness of queries
│   │   ├── context_cache.py            # Caches chunk retrieval and LLM calls
│   │   ├── conversation_manager.py     # Manages query conversations
│   │   ├── counts_service.py           # Counts doucments that include a search term
│   │   ├── domain_terms.py             # Manages weights to apply to search terms if used by domain
│   │   ├── generate.py                 # Generates responses from the LLM
│   │   ├── graph_retrieval.py          # Extends semantic retrieval with technical based graph relationship expansion
│   │   ├── hybrid_search_weights.py    # Manages keyword and vector search results with learned weights
│   │   ├── query_expansion.py          # Expands query to find other chunks that might be relevant to the query
│   │   ├── query.py                    # Takes a query as input and coordinates the RAG response
│   │   ├── rag_config.py               # Utility class for loading config values
│   │   ├── rag_logger.py               # Utility class for logging RAG activity
│   │   ├── retrieve.py                 # Retrieve Chroma DB chunks based on semantic similarity search
│   │   └── semantic_clustering.py      # Cluster semantically similar search terms
│   │ 
│   ├── search/
│   │   ├── bm25_retrieval.py           # BM25 retrieval using pre-built index from cache
│   │   ├── bm25_search.py              # BM25 full-text search implementation
│   │   ├── hybrid_search.py            # Hybrid semantic + keyword search
│   │   ├── persona_retrieval.py        # Academic persona aware filtering and reranking logic
│   │   ├── query_expansion.py          # Query expansion for improved retrieval of BM25 hybrid search
│   │   ├── reranker.py                 # Semantic reranking of search results
│   │   └── text_preprocessing.py       # Text normalisation and preprocessing
│   │ 
│   ├── security/
│   │   └── dlp.py                      # Data Loss Prevention (DLP) filtering
│   │ 
│   ├── ui/
│   │   ├── dashboard.py                # Plotly Dash web interface (main dashboard)
│   │   ├── export_manager.py           # Exports conversations
│   │   ├── layout_engine.py            # WebGL layout rendering
│   │   ├── performance_monitor.py      # Performance monitoring utility
│   │   ├── query_templates.py          # Query templates
│   │   ├── spatial_index.py            # Quadtree spatial index to efficiently cull viewport
│   │   └── word_cloud_provider.py      # Supports rendering a word cloud in the dashboard.
│   │ 
│   └── utils/
│       ├── __init__.py                 # exports: logger, retry_utils, rate_limiter, schemas, db_factory
│       ├── adaptive_cache_tuning.py    # Automatic cache optimisation
│       ├── adaptive_rate_limiter.py    # Automatic API tuning
│       ├── cache.py                    # BaseCache, SimpleCache, TTLCache, LRUCache
│       ├── config.py                   # BaseConfig with env var helpers
│       ├── db_factory.py               # Supports using either ChromaDB or SQLlite for the vector DB
│       ├── json_utils.py               # Managing JSON output from LLMs
│       ├── llm_instrumentation.py      # Records LLM activity
│       ├── logger.py                   # get_logger, audit functions
│       ├── metrics_export.py           # Supports export of metrics via the dashboard
│       ├── monitoring.py               # OpenTelemetry instrumentation (work in progress)
│       ├── rate_limiter.py             # RateLimiter with token bucket
│       ├── resource_monitor.py         # ResourceMonitor for capacity planning
│       ├── retry_utils.py              # Exponential_backoff, retry decorators
│       └── schemas.py                  # MetadataSchema, ChunkSchema, etc.
│
├── tests/
│   ├── fixtures/
│   │   └── *.html                      # pytest fixtures to support testing
│   ├── logs/
│   │   └── *.log                       # pytest test output to support log testing
│   ├── confest.py                      # pytest config utility
│   └── test_*.py                       # pytest tests
│
├── AGENTS.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
├── requirements.txt
└── setup.sh
```
N.B. Jupyter libraries to view the notebooks with are optional.

## 🛠️ Environment Setup (WSL)

### 1. Clone the repository

N.B. Solution assumes that application is running from the root of the home directory.

```bash
git clone <repo-url>
cd ~/rag-project
```

### 2. Create and activate the virtual environment

Use the automated setup script:
          
```bash
chmod +x ./setup.sh
./setup.sh
```
- Checks Python version (>= 3.10)
- Creates virtual environment
- Installs dependencies
- Creates project directories
- Checks for Ollama and required models
- Generates .env from template
- Provides next steps

Or use the Makefile:

```bash
make install
```
List of Makefile options (run in this order)
- `make help` show all available commands
- `make install` install dependencies
- `make setup-dirs` create required directories
- `make test` / `make test-cov` run tests to confirm environment is setup
- `make ingest` make ingest-reset and runs ingestion
- `make query QUERY="..."` run RAG queries
- `make graph` build consistency graph
- `make dashboard` launch dashboard
- `make test` / `make test-cov` run tests on making any changes
- `make full-rebuild` complete rebuild workflow

Otherwise if needed, create the Python environment

```bash
python -m venv .venv
```
Then activate the environment

```bash
source .venv/bin/activate
```
Then install the required libraries

```bash
pip install -r requirements.txt
```

### 3. Set the environment parameters

Variables are checked in the following order of priority
1. CLI arguments
2. OS environment (docker, Kubernetes, etc)
3. .env configuration files if present (useful for dev environments)
4. Application sensible defaults

```bash
cp .env.example .env
nano .env
```

### 4. Ensure all of the tests work

```bash
pytest tests
```

### 5. Confirm that you can ingest files and the environment is working

Copy files to `data_raw/downloads`

```bash
python scripts/ingest/ingest.py --dry-run --limit 10 --verbose
```
or

```bash
python scripts/ingest/ingest.py --profile --verbose
```

**CLI Options**:
- `--dry-run` - Preview ingestion without writing to ChromaDB
- `--profile` - Quick validation with detailed timing analysis
- `--limit N` - Process only first N files
- `--reset` - Clear ChromaDB collections before ingestion
- `--verbose` - Enable detailed console output
- `--purge-logs` - Delete all log files before starting (disabled in Prod environment)
- `--log-unsupported-files` - Log files skipped due to unsupported extensions

### 6. Ingest the required files, use parameters appropriate to your environment

Run scripts directly:

```bash
python scripts/ingest/ingest.py
```
or run scripts as modules (generally a better practice)
```bash
python -m scripts.ingest.ingest
```
or use the Makefile
```bash
make ingest
```

### 7. Confirm that files are loaded and can be queried

```bash
python scripts/rag/query.py "natural language query on a topic covered by the loaded documents"
```
or
```bash
make query QUERY="..."
```

**CLI Options**:
- `--k N` - Number of chunks to retrieve (default: 5)
- `--verbose` - Show performance metrics and model info
- `--show-sources` - Include source metadata for retrieved chunks
- `--purge-logs` - Delete all RAG log files before starting (disabled in Prod environment)

### 8. Build the consistency graph

```bash
python scripts/consistency_graph/build_consistency_graph.py
```
or
```bash
make graph
```

**CLI Options**:
- `--max-neighbours N` - Max neighbours per document for consistency checks
- `--similarity-threshold T` - Similarity threshold for edge creation
- `--workers W` - Number of parallel LLM workers
- `--no-documents` - Exclude document bodies to reduce memory
- `--purge-logs` - Delete all consistency graph log files before starting (disabled in Prod environment)

### 9. Start the dashboard

#### Using the main environment (RAG pipeline + Dashboard)

```bash
python scripts/ui/dashboard.py
```
or
```bash
make dashboard
```

Open in browser [http://localhost:8050](http://localhost:8050)

If running in WSL, may need to use the host IP address rather than localhost.

## �️ Uninstalling the Application

To completely remove the Governance Intelligence Console and clean up your environment:

### 1. Deactivate the virtual environment

If the virtual environment is currently active, change directory to the project root:

```bash
deactivate
```

### 2. Remove the project directory

```bash
cd ~
rm -rf rag-project
```

Or if using a different location:

```bash
rm -rf /path/to/rag-project
```

### 3. Clean up environment variables

If you set environment variables in your shell profile (`.bashrc`, `.zshrc`, `.bash_profile`, etc.), remove or comment out:

```bash
# Remove or comment out these lines from ~/.bashrc, ~/.zshrc, or equivalent:
# export RAG_BASE_PATH=...
# export RAG_DATA_PATH=...
# export URL_SEED_JSON_PATH=...
# export CHROMA_DB_PATH=...
# export LLM_CACHE_PATH=...
# export EMBEDDING_CACHE_PATH=...
# Any other RAG related variables
```

Then reload your shell:

```bash
source ~/.bashrc  # or ~/.zshrc for zsh
```

### 4. Remove the .env file (if created separately)

If you created a `.env` file outside the project directory:

```bash
rm ~/.env.rag  # or wherever you stored it
```

### 5. Optional: Remove Ollama (if installed solely for this project)

If you installed Ollama only for this project and no longer need it:

```bash
# On Linux
sudo apt remove ollama

# On macOS
rm -rf ~/Applications/Ollama.app

# Or for Homebrew on macOS
brew uninstall ollama
```

### 6. Optional: Remove ChromaDB data (if stored outside the project)

If you configured ChromaDB to store data outside the project directory:

```bash
rm -rf /path/to/chroma_db_storage
```

### Complete cleanup checklist

- ✅ Deactivated virtual environment
- ✅ Removed `rag-project` directory
- ✅ Removed environment variables from shell profile
- ✅ Deleted any external `.env` files
- ✅ Uninstalled Ollama (if not needed for other projects)
- ✅ Removed ChromaDB data directories (if external)
- ✅ Cleared any project-specific cache directories

## 🔄 Rebuilding the Consistency Graph

The dashboard loads the prebuilt graph from the consistency graph DB on startup. If there are changes to ChromaDB ingestion, the consistency graph DB should be rebuilt.

Manual rebuild of graph:

```bash
python scripts/consistency_graph/build_consistency_graph.py
```
or
```bash
make graph
```

## 📊 Dashboard Features

### Cluster Overview

- Risk clusters
- Topic clusters
- LLM‑generated labels, descriptions, summaries
- Cluster sizes and risk posture

## 🧩 Cluster Metadata Model

The basic structure is made up of nodes (imported document metadata), edges (document similarity relationships), and clusters (documents that share the same risks or are covered by the same topics).


## 🤖 LLM Usage

The LLM is used for:

- Consistency scoring between document pairs
- Cluster label generation
- Cluster descriptions
- Risk summaries (risk clusters)
- Topic summaries (topic clusters)

Prompts are deterministic and auditable.

## 🔁 Switching Embedding Models

You can change the embedding model used across ingestion, retrieval, and the dashboard. The system isolates embeddings per model to avoid cross‑model mixing.

- Where it’s configured: Environment Variable → `EMBEDDING_MODEL_NAME`.
- What’s namespaced: Embedding cache keys and Chroma metadata include the current model name.
- What’s filtered: Retrieval and dashboard queries filter by `embedding_model` to ensure model‑consistent results.

### Steps to switch

1. Update the model name:
   - Set `EMBEDDING_MODEL_NAME` to your target (e.g., `"mxbai-embed-large"`, `"nomic-embed-text"`, `"all-minilm"`).

2. Re‑ingest or re‑embed:
   - For a clean reindex:

```bash
python scripts/ingest/ingest.py --reset
```

   - Or incrementally re‑ingest changed docs only:

```bash
python scripts/ingest/ingest.py
```

3. Clear the embedding cache (optional but recommended)

4. Rebuild the consistency graph (if used)

```bash
python scripts/consistency_graph/build_consistency_graph.py
```

5. Verify retrieval uses the new model
   - CLI:

```bash
python scripts/rag/query.py "What are our security policies?"
```

   - Dashboard semantic search and RAG assistant automatically filter by `embedding_model`.

### Notes

- Mixing embeddings from different models in one index degrades quality; this project avoids that via metadata and query filters.
- If you keep multiple models side‑by‑side, ensure each index and cache uses its own namespace. The current setup handles this automatically when you change `EMBEDDING_MODEL_NAME` and re‑ingest.
- Re-ingestion does lose past ingestion version history and information on semantic drift between ingestion versions. The semantic drift feature is not meant for historical reconstruction, but to alert on potentially unexpected changes on a new load.
- If maintaining semantic drift over ingestion versions is important, consider:
  - Creating snapshot loads distinguished by version number or load date, then re-ingest in the same order as they were previously ingested. The new model will also likely assess the semantic drift between versions differently than the previous model as well. 
  - Export drift data from the audit logs before switching models
  - Keep a separate backup of the old Chroma DB collections
  - Use the dashboard to analyse / visualise drift trends before the migration to the new model


## 🔍 Filtering Test Files (Git Code Ingestion)

When ingesting code from Git providers, you can exclude test files to keep your production corpus focused on implementation code.

- CLI flag: `--exclude-tests`
- Environment variable: `GIT_EXCLUDE_TESTS=true`

Heuristic coverage:
- Directories: `src/test/**`, `test/**`, `tests/**`, `__tests__/**`
- Filenames: `*Test.java`, `*Tests.java`, `*IT.java`, `*Spec.java` and Groovy equivalents
- Patterns: `.test.`, `.spec.`, plus `test.*` and `test_*`

Examples:

```bash
# Exclude tests via CLI flag
python scripts/ingest/ingest_git.py \
  --provider bitbucket \
  --host https://bitbucket.org \
  --project myproject \
  --repo myrepo \
  --branch main \
  --exclude-tests \
  --verbose

# Exclude tests via env var
export GIT_EXCLUDE_TESTS=true
python scripts/ingest/ingest_git.py \
  --provider bitbucket \
  --host https://bitbucket.org \
  --project myproject \
  --repo myrepo \
  --branch main \
  --verbose
```

During ingestion you'll see a summary like: `Excluded N test files`.

### 8. Build the consistency graph following running of ingest_git.py

```bash
python scripts/consistency_graph/build_consistency_graph.py
```
or
```bash
make graph
```

## Ingesting Academic Papers

**Academic Features:**
- 3 personas with distinct retrieval strategies
- Quality score and citation filtering
- Reference type classification (academic/preprint/report)
- Stale link detection and penalty scoring

```bash
# Activate environment
cd ~/rag-project
source .venv/bin/activate

# Ingest a single thesis/paper PDF
python scripts/ingest/ingest_academic.py data_raw/academic_papers/your_thesis_paper.pdf

# Or ingest a batch list of references
python scripts/ingest/ingest_academic.py --batch data_raw/academic_references.txt

# Preview only (no writes)
python scripts/ingest/ingest_academic.py --dry-run --batch data_raw/academic_references.txt
```

### 8. Build the consistency graph following running of ingest_academic.py

```bash
python scripts/consistency_graph/build_consistency_graph.py
```
or
```bash
make graph
```

**Academic Reference Maintenance:**
- Staleness check notebook: `notebooks/academic_reference_staleness_check.ipynb`
- Revalidation commands:
  - `python scripts/ingest/ingest_academic.py --revalidate stale --staleness-threshold 30`
  - `python scripts/ingest/ingest_academic.py --revalidate failed`
  - `python scripts/ingest/ingest_academic.py --revalidate online`
  - `python scripts/ingest/ingest_academic.py --revalidate all`
  - `python scripts/ingest/ingest_academic.py --revalidate ids --ref-ids <space separated ref IDs>`

## 🧪 Testing

### Running Tests

Run all tests:
```bash
pytest tests
```

Run specific test file:
```bash
pytest tests/test_<module>.py
```

Run with coverage report:
```bash
pytest tests/test_<module>.py --cov=scripts --cov-report=term-missing
```

## 🤝 Contributing

Feedback, suggestions, defect reporting, or pull requests are welcome.

If wanting to make changes:

### Tool Chain
Black (Formatting)
    ↓
isort (Import Organisation)
    ↓
pylint (Linting) → --fail-under=8
    ↓
mypy (Type Checking) → --ignore-missing-imports
    ↓
pytest (Testing) 


## 📄 License

Apache 2.0