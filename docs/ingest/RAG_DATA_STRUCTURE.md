# RAG Data Structure

## Overview

The `rag_data/` directory is the persistent storage root for ingestion, retrieval, caching, dashboard analytics, and academic workflows.

This document reflects the **current implementation** (SQLite-first cache architecture).

## Current Directory Structure

Typical structure in this project:

```
rag_data/
├── chromadb/                         # ChromaDB persistent directory (default backend)
├── chroma.sqlite3                    # Chroma local SQLite file (backend artefact)
│
├── cache.db                          # Unified SQLite cache (LLM, embeddings, graph cache, BM25, analytics)
├── benchmarks.db                     # RAG query benchmark database
├── conversations.db                  # Dashboard conversation history
├── query_templates.db                # Dashboard query templates
├── hybrid_search_weights.json        # Hybrid vector/BM25 weighting config
│
├── consistency_graphs/
│   └── consistency_graph.sqlite      # Consistency graph SQLite output
│
├── academic_references.db            # Academic reference cache/state
├── academic_citation_graph.db        # Academic citation graph SQLite database
├── academic_citation_graph.json      # Academic citation graph JSON export
├── academic_terminology.db           # Academic terminology database
├── academic_pdfs/                    # Downloaded/managed academic PDFs
│
├── domain_terms/                     # Domain term artefacts
└── learning/                         # Adaptive learning artefacts
```

## What Is Stored Where

### `chromadb/` and `chroma.sqlite3`
- Vector collections for document and chunk retrieval.
- Managed by the vector backend selected through `scripts.utils.db_factory`.
- Default path resolves from `RAG_DATA_PATH`.

### `cache.db` (Primary Cache)
- Central SQLite cache used across ingestion and retrieval.
- Implemented in `scripts/ingest/cache_db.py`.
- Stores multiple cache domains, including:
    - `llm_cache`
    - `embedding_cache`
    - `graph_cache`
    - `document_cache`
    - BM25 index and corpus statistics tables
    - query analytics tables

### `consistency_graphs/consistency_graph.sqlite`
- Canonical consistency graph output.
- Default from `ConsistencyConfig.output_sqlite` and `RAGConfig.graph_sqlite_path`.

### Academic Datastores
- `academic_references.db`: reference metadata/status cache.
- `academic_citation_graph.db`: citation graph relationships.
- `academic_citation_graph.json`: compatibility/export copy.
- `academic_terminology.db`: extracted academic terminology.
- `academic_pdfs/`: local academic artifact downloads.

### Dashboard/Runtime Datastores
- `benchmarks.db`: performance and quality benchmarking.
- `conversations.db`: conversation persistence.
- `query_templates.db`: reusable dashboard query templates.

### Hybrid Search Configuration
- `hybrid_search_weights.json`: persisted vector/keyword weighting strategy.

## Auto-Creation Behaviour

All major paths are created automatically when first needed:

- `cache.db`: created by `CacheDB` initialisation.
- `chromadb/` (or backend equivalent): created by vector `PersistentClient`.
- `consistency_graphs/`: created by consistency graph manager/configured outputs.
- academic databases/files: created by academic ingestion flows.

No manual pre-creation is required for normal operation.

## Configuration Keys

Primary path settings:

```env
RAG_DATA_PATH=/workspaces/governance-rag/rag_data
CONSISTENCY_GRAPH_SQLITE=/workspaces/governance-rag/rag_data/consistency_graphs/consistency_graph.sqlite
RAG_GRAPH_SQLITE_PATH=/workspaces/governance-rag/rag_data/consistency_graphs/consistency_graph.sqlite
```

Cache settings used by current code paths:

```env
RAG_CACHE_ENABLED=true
RAG_CACHE_PATH=/workspaces/governance-rag/rag_data/cache.db
LLM_CACHE_ENABLED=true
EMBEDDING_CACHE_ENABLED=true
```

## Operational Guidance

### Safe Cleanup Scope

If you need a full rebuild:

```bash
rm -rf rag_data/
```

If you only need to clear caches while preserving source stores:

```bash
rm -f rag_data/cache.db
```

### Backup Priority

For fast recovery, prioritise backup of:
1. `rag_data/chromadb/` and `rag_data/chroma.sqlite3`
2. `rag_data/cache.db`
3. `rag_data/consistency_graphs/consistency_graph.sqlite`
4. academic databases if academic workflows are used

## Troubleshooting

### Permission Errors
- Ensure write permissions on `RAG_DATA_PATH`.

### Missing Files After Clean Start
- Expected on first run; files are generated lazily when each subsystem is exercised.

### Stale or Large Cache
- Remove `rag_data/cache.db` to reset cache state.
- Re-run ingestion/query workloads to repopulate.

### Vector Store Issues
- If Chroma storage appears inconsistent, remove Chroma artefacts under `rag_data/` and re-ingest.
