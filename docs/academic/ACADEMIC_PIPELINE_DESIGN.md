# Academic Ingestion Pipeline Design

---

## Pipeline Architecture

### Primary Ingestion Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT: One or more academic documents (PDF/URL)                     │
└────────────────────┬────────────────────────────────────────────────┘
                     │
         ┌───────────▼──────────────┐
         │ 1. LOAD & PARSE          │
         │ - PDF extraction         │
         │ - Text normalisation     │
         │ - Bibliography detection │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────┐
         │ 2. EXTRACT CITATIONS     │
         │ - Raw citation text      │
         │ - Normalisation          │
         │ - Fuzzy deduplication    │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────────────────┐
         │ 3. RESOLVE REFERENCES (with cache)   │
         │ - Crossref → OpenAlex → etc.         │
         │ - Reuse cache hits                   │
         │ - Flag unresolved                    │
         └───────────┬──────────────────────────┘
                     │
         ┌───────────▼──────────────────────────┐
         │ 4. DISCOVER OA PDFs                  │
         │ - Unpaywall lookup                   │
         │ - arXiv direct links                 │
         │ - Size/type validation               │
         └───────────┬──────────────────────────┘
                     │
         ┌───────────▼──────────────────────────┐
         │ 5. DOWNLOAD & INGEST REFERENCES      │
         │ - Fetch PDFs (with dedup)            │
         │ - Extract text                       │
         │ - Chunk + embed                      │
         │ - Store in ChromaDB                  │
         └───────────┬──────────────────────────┘
                     │
         ┌───────────▼──────────────────────────┐
         │ 6. BUILD CITATION GRAPH              │
         │ - Create edges (doc → ref)           │
         │ - Transitive closure                 │
         │ - Impact scoring                     │
         └───────────┬──────────────────────────┘
                     │
         ┌───────────▼──────────────────────────┐
         │ 7. EXTRACT DOMAIN TERMINOLOGY        │
         │ - Cross-doc term extraction          │
         │ - Frequency analysis                 │
         │ - Domain relevance scoring           │
         └───────────┬──────────────────────────┘
                     │
         ┌───────────▼──────────────────────────┐
         │ OUTPUT: BatchIngestResult            │
         └──────────────────────────────────────┘
```

### Re-validation Flow (Standalone)

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT: Re-validation mode (stale/online/all/failed/ids)             │
└────────────────────┬────────────────────────────────────────────────┘
                     │
         ┌───────────▼──────────────────────────┐
         │ 1. SELECT REFERENCES                 │
         │ - Query DB by mode criteria          │
         │ - Apply staleness threshold          │
         │ - Filter by reference type           │
         └───────────┬──────────────────────────┘
                     │
         ┌───────────▼──────────────────────────┐
         │ 2. RE-RESOLVE METADATA               │
         │ - Bypass cache (force refresh)       │
         │ - Query providers again              │
         │ - Detect metadata changes            │
         └───────────┬──────────────────────────┘
                     │
         ┌───────────▼──────────────────────────┐
         │ 3. UPDATE DATABASE                   │
         │ - Preserve doc_ids (citing links)    │
         │ - Update metadata fields             │
         │ - Update accessed_at timestamp       │
         └───────────┬──────────────────────────┘
                     │
         ┌───────────▼──────────────────────────┐
         │ 4. RE-CHUNK IF CONTENT CHANGED       │
         │ - Delete old chunks                  │
         │ - Re-fetch online content            │
         │ - Generate new embeddings            │
         │ - Store updated chunks               │
         └───────────┬──────────────────────────┘
                     │
         ┌───────────▼──────────────────────────┐
         │ OUTPUT: RevalidationResult           │
         │ - Updated count                      │
         │ - Unchanged count                    │
         │ - Failed count                       │
         │ - Change details                     │
         └──────────────────────────────────────┘
```

---

## Stage Implementations

### Stage 1: Load & Parse
                     │
         ┌───────────▼──────────────────────────┐
         │ 6. BUILD CITATION GRAPH              │
         │ - Parent doc → Reference edges       │
         │ - Reference → Reference edges        │
         │ - Transitive closure for queries     │
         └───────────┬──────────────────────────┘
                     │
         ┌───────────▼──────────────────────────┐
         │ 7. EXTRACT DOMAIN TERMINOLOGY        │
         │ - Candidate terms from all text      │
         │ - Cross-document frequency analysis  │
         │ - Domain relevance scoring           │
         └───────────┬──────────────────────────┘
                     │
         ┌───────────▼────────────────┐
         │ OUTPUT: Indexed corpus     │
         │ - Vector embeddings        │
         │ - Citation edges           │
         │ - Domain terms             │
         │ - Audit trail              │
         └────────────────────────────┘
```

---
