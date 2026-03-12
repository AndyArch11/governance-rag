# BM25 Ingestion-Time Indexing

## Overview

BM25 keyword indexing is integrated into ingestion and built at **chunk-level granularity** during ingestion (not at query time).

Current implementation indexes:
- Regular chunks
- Child chunks (when parent-child chunking is enabled)
- Parent chunks (context chunks)

This provides strong lexical matching for exact technical terms and improves hybrid search recall.

BM25 index characteristics:
- Pre-computed IDF values for the indexed corpus
- Faster query-time retrieval (no index build at query time)
- Hybrid search combines semantic + keyword effectively
- Optional storage of chunk original text via `BM25_INDEX_ORIGINAL_TEXT`

## Configuration

### Enable BM25 Indexing

Add to `.env` file:

```bash
# Enable BM25 indexing during ingestion (default: true)
BM25_INDEXING_ENABLED=true

# Index original full text vs preprocessed chunks (default: true)
# Recommended: true for maximum keyword preservation
BM25_INDEX_ORIGINAL_TEXT=true
```

### Disable BM25 Indexing

To skip BM25 indexing (not recommended for hybrid search):

```bash
BM25_INDEXING_ENABLED=false
```

## How It Works

### Ingestion Pipeline

When a document is ingested, the pipeline now includes:

1. **Extract raw text** (HTML / PDF parsing)
2. **Preprocess text** (LLM metadata generation)
3. **Chunk text** (semantic chunking)
4. **Store chunks** (ChromaDB with embeddings)
5. **Build BM25 index (chunk-level)**
    - Tokenise regular/child/parent chunk text
    - Compute term frequencies per chunk
    - Store in SQLite cache database

### Post-Ingestion Corpus Stats

After all documents are indexed:

1. **Compute IDF values** for all terms
2. **Calculate average document length**
3. **Update corpus statistics** in cache database

### Storage

BM25 index data is stored in `rag_data/cache.db` SQLite database:

**Tables:**
- `bm25_index`: Term frequencies per document
- `bm25_corpus_stats`: IDF values per term
- `bm25_doc_metadata`: Document lengths and original text

**Size:** ~10-20% of original document size

## Usage

### During Ingestion

BM25 indexing happens automatically when enabled:

```bash
# HTML document ingestion
python3 scripts/ingest/ingest.py

# Git code ingestion
python3 scripts/ingest/ingest_git.py --provider bitbucket --host https://bitbucket.org --project PROJ --repo my-repo

# Academic ingestion
python3 scripts/ingest/ingest_academic.py --papers-dir data_raw/academic_papers

# Reset and rebuild all indexes
python3 scripts/ingest/ingest.py --reset

python3 scripts/ingest/ingest_git.py --provider bitbucket --host https://bitbucket.org --project PROJ --repo my-repo --reset

python3 scripts/ingest/ingest_academic.py --papers-dir data_raw/academic_papers --reset

# Academic ingestion with explicit BM25 controls
python3 scripts/ingest/ingest_academic.py --papers-dir data_raw/academic_papers --bm25-indexing
python3 scripts/ingest/ingest_academic.py --papers-dir data_raw/academic_papers --skip-bm25
```

### Academic Ingestion Notes (Parent + Child Indexing)

`ingest_academic.py` indexes BM25 at chunk-level and includes parent/child chunks when available.

Implications:
- **Reported BM25 corpus size can appear larger** because both child and parent segments are indexed.
- **Storage usage increases** versus child-only indexing due to additional parent chunk entries.
- **Matching quality improves** because parent chunks add broader context terms while child chunks retain precision.

Important nuance for word clouds:
- Word cloud frequencies in academic ingestion are currently derived from extracted document/reference text (`put_word_frequencies` path), not directly from BM25 index entries.
- Parent+child BM25 indexing therefore skews BM25 corpus/document counts, but does not directly double-count word-cloud tokens.

### During Retrieval

Use the pre-built BM25 index for fast keyword search:

```python
from scripts.search.bm25_retrieval import BM25Retriever

# Initialise retriever
retriever = BM25Retriever()

# Search with pre-built index
results = retriever.search("authentication security", top_k=10)
# Returns: [(doc_id, bm25_score), ...]

# Get index statistics
stats = retriever.get_stats()
print(f"Indexed documents: {stats['total_documents']}")
print(f"Avg doc length: {stats['avg_doc_length']} tokens")
```

### Hybrid Search Integration

The BM25 pre-built index is **automatically used** in the RAG query pipeline with intelligent fallback:

**Automatic Integration in `scripts/rag/retrieve.py`:**

```python
# When hybrid search is enabled (default: true)
# 1. Try BM25Retriever with pre-built index
# 2. Fall back to simple keyword search if index not available

from scripts.rag.query import main
main()  # Uses BM25 index automatically if available
```

**Query Pipeline:**
```
User Query
    ↓
Vector Search (semantic)  +  BM25 Search (keyword)
    ↓                               ↓
  5 chunks                   Try: Pre-built BM25 index
    ↓                            Fallback: On-the-fly TF search
    ↓                               ↓
    └────────→ Combine Results ←────┘
                    ↓
              RRF/Linear Fusion
                    ↓
           Return top-k results
```

**Benefits:**
- **Inbuilt default search logic** - Works automatically when index exists
- **Graceful degradation** - Falls back to simple search if no index
- **Performance boost** - 100-1000x faster keyword matching
- **Better results** - Uses original text, not summarised chunks

**To verify it's working:**
```bash
# Check logs for BM25 retrieval usage
tail -f logs/rag.log | grep "BM25 retrieval"

# Expected output when using pre-built index:
# BM25 retrieval (pre-built index) found 8 matches
```

**Audit events:**
- `bm25_retrieval_used`: Pre-built index was used
- `keyword_search_fallback_used`: Fell back to simple TF search

### Manual BM25 Retrieval

You can also use BM25Retriever directly for keyword-only searches:

```python
from scripts.search.bm25_retrieval import BM25Retriever

# Initialise retriever
retriever = BM25Retriever()

# Search with pre-built index
results = retriever.search("authentication security", top_k=10)
# Returns: [(doc_id, bm25_score), ...]

# Get index statistics
stats = retriever.get_stats()
print(f"Indexed documents: {stats['total_documents']}")
print(f"Avg doc length: {stats['avg_doc_length']} tokens")
```

### Advanced Hybrid Search

For custom fusion strategies or parameter tuning:

```python
from scripts.search.bm25_retrieval import BM25Retriever
from scripts.search.hybrid_search import HybridSearch, FusionStrategy
from scripts.rag.retrieve import _semantic_search

# Initialise searches
bm25 = BM25Retriever()
def keyword_fn(query, top_k):
    return bm25.search(query, top_k)

def semantic_fn(query, top_k):
    # Semantic search implementation
    return _semantic_search(query, collection, top_k)

# Create hybrid search
hybrid = HybridSearch(
    semantic_search_fn=semantic_fn,
    keyword_search_fn=keyword_fn,
    fusion_strategy=FusionStrategy.RRF,
    alpha=0.6,  # For LINEAR strategy
    rrf_k=60,   # For RRF strategy
)

# Search
results = hybrid.search("authentication MFA", top_k=10, parallel=True)
```

## Performance

### Ingestion Impact

- **Index build time:** ~0.1-0.5s per document (higher when parent+child chunks are indexed)
- **Overhead:** Minimal (concurrent with embedding generation)
- **Storage:** ~10-20% increase in cache.db size (can be higher with parent+child chunk indexing)

### Query Performance

- **Keyword search:** 1-10ms (pre-built index)
- **vs. On-the-fly indexing:** 100-1000x faster
- **Hybrid search:** 10-50ms (semantic + keyword + fusion)

## Monitoring

### Ingestion Logs

Check logs for BM25 indexing confirmation:

```bash
tail -f logs/ingest.log | grep bm25
```

Expected output:
```
BM25 indexed 42 chunks for doc_12345 (granularity=chunk-level)
Updating BM25 corpus stats for academic artifacts...
BM25 corpus stats updated: 234 documents, avg length 287.1 tokens
```

### Audit Events

BM25 indexing creates audit events:

- `bm25_indexed`: Document successfully indexed
- `bm25_index_failed`: Document indexing failed (non-fatal)
- `bm25_corpus_stats_updated`: Corpus statistics computed
- `bm25_corpus_stats_failed`: Corpus stats computation failed

### Database Inspection

Check BM25 index contents:

```bash
# Open cache database
sqlite3 rag_data/cache.db

# Count indexed documents
SELECT COUNT(*) FROM bm25_doc_metadata;

# Count unique terms
SELECT COUNT(DISTINCT term) FROM bm25_index;

# Top 10 most common terms
SELECT term, document_frequency, idf 
FROM bm25_corpus_stats 
ORDER BY document_frequency DESC 
LIMIT 10;

# Document with most unique terms
SELECT doc_id, COUNT(DISTINCT term) as unique_terms
FROM bm25_index
GROUP BY doc_id
ORDER BY unique_terms DESC
LIMIT 10;
```

## Troubleshooting

### BM25 Index is Empty

**Symptom:** `BM25 index is empty` warning when searching

**Causes:**
1. BM25 indexing disabled: `BM25_INDEXING_ENABLED=false`
2. No documents ingested yet
3. Ingestion ran in dry-run mode (`--dry-run`)

**Solution:**
```bash
# Enable BM25 indexing
echo "BM25_INDEXING_ENABLED=true" >> .env

# Re-run ingestion
python3 scripts/ingest/ingest.py --reset
```

### BM25 Indexing Failed

**Symptom:** `bm25_index_failed` audit events in logs

**Causes:**
1. Import error (missing BM25Search module)
2. Database connection failure
3. Tokenisation error (invalid text encoding)

**Solution:**
- Check logs for detailed error: `grep bm25_index_failed logs/ingest_audit.jsonl`
- Verify cache database is writable: `ls -la rag_data/cache.db`
- Re-run ingestion with `--verbose` flag

### Corpus Stats Not Updated

**Symptom:** IDF values all zero or None

**Causes:**
1. Ingestion did not complete (interrupted)
2. Cache database locked
3. No documents successfully indexed

**Solution:**
```bash
# Manually trigger corpus stats update
python3 -c "
from scripts.utils.db_factory import get_cache_client
db = get_cache_client(enable_cache=True)
total = db.get_bm25_corpus_size()
db.update_bm25_corpus_stats(total)
print(f'Updated stats for {total} documents')
"
```

## Best Practices

### For Best Keyword Matching

1. **Index original text:** Set `BM25_INDEX_ORIGINAL_TEXT=true`
2. **Preserve technical terms:** Add to `PRESERVE_DOMAIN_KEYWORDS` in `.env`
3. **Re-index after major document updates:** Run with `--reset` flag

### For Storage Efficiency

1. **Index preprocessed text only:** Set `BM25_INDEX_ORIGINAL_TEXT=false` (saves ~50% storage)
2. **Periodic cleanup:** Remove old document versions from index
3. **Selective indexing:** Only index documents with `BM25_INDEXING_ENABLED=true` for specific categories

### For Query Performance

1. **Pre-warm index:** Load BM25Retriever at application startup
2. **Cache frequent queries:** Use LRU cache for common search terms
3. **Limit top_k:** Request only needed results (default: 10)

## Technical Details

### BM25 Formula

$$
\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}
$$

Where:
- $D$ = document
- $Q$ = query
- $q_i$ = query term
- $f(q_i, D)$ = frequency of $q_i$ in $D$
- $|D|$ = length of $D$ (tokens)
- $\text{avgdl}$ = average document length
- $k_1$ = term frequency saturation (default: 1.5)
- $b$ = length normalisation (default: 0.75)

### IDF Calculation

$$
\text{IDF}(t) = \ln\left(\frac{N - df(t) + 0.5}{df(t) + 0.5} + 1\right)
$$

Where:
- $N$ = total documents in corpus
- $df(t)$ = document frequency of term $t$

### Tokenisation

Default tokenisation strategy:
1. Lowercase all text
2. Split on non-word characters: `\b\w+\b`
3. Filter tokens shorter than 3 characters
4. No stemming or lemmatization (preserves exact terms)

Custom tokenisation can be configured via `BM25Search.tokenize()` method.

## References

- [BM25 Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Robertson & Zaragoza (2009): BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
- [Hybrid Search Patterns](https://www.pinecone.io/learn/hybrid-search/)

---


