# LLM Cache Documentation

## Overview

The LLM caching system reduces redundant API calls during document ingestion by storing LLM outputs keyed by document content hash. This dramatically speeds up re-ingestion when document content hasn't changed.

**Storage:** SQLite database at `rag_data/cache.db`

## How It Works

### Cache Key Format
Cache keys are SHA256 hashes of the prompt text:
```
SHA256(prompt_text) -> content_hash
```

Example: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

### Document Hash Algorithm
Document content hashes use **SHA-256** for change detection:
- **File-based documents** (PDF, HTML, etc.): SHA-256 hash of file content (computed in 8KB chunks)
- **Chunk-level hashes**: SHA-256 hash of chunk text (UTF-8 encoded)
- **Purpose**: Fast change detection (not cryptographic security)

**Implementation:**
```python
# Document hash (ingest.py)
h = hashlib.sha256()
with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
        h.update(chunk)
return h.hexdigest()

# Cache key hash (cache_db.py)
def _compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

### Cached Operations

The following LLM operations are cached based on their full prompt text (which typically includes document/chunk content):

1. **Text Cleaning** (`clean_text_with_llm`)
   - Removes boilerplate and navigation
   - Sanitises company/people references
   - Cache key: SHA256(prompt containing document text)

2. **Metadata Extraction** (`extract_metadata_with_llm`)
   - Extracts doc_type, key_topics, summary
   - Cache key: SHA256(prompt containing document content)

3. **Summary Scoring** (`score_summary`)
   - Scores summary quality (relevance, coverage, clarity, conciseness)
   - Cache key: SHA256(prompt containing summary text)

4. **Summary Regeneration** (`regenerate_summary`)
   - Regenerates low-quality summaries
   - Cache key: SHA256(prompt containing document content)

5. **Semantic Drift Detection** (`detect_semantic_drift`)
   - Compares document versions for meaningful changes
   - Cache key: SHA256(prompt containing old and new versions)

6. **Chunk Validation** (`validate_chunk_semantics`)
   - Validates chunk coherence and usefulness
   - Cache key: SHA256(prompt containing chunk text)
   - Note: Each chunk generates a unique cache entry

7. **Chunk Repair** (`repair_chunk_with_llm`)
   - Repairs invalid/truncated chunks
   - Cache key: SHA256(prompt containing chunk text)
   - Note: Each chunk generates a unique cache entry

**How it works:** When a document's SHA-256 hash remains unchanged, LLM prompts (which embed that content) produce identical SHA-256 cache keys, resulting in cache hits and skipped LLM calls.

## Configuration

Add these settings to your [.env](../.env) file (see [.env.example](../.env.example) for all options):

```bash
# Enable/disable LLM caching
LLM_CACHE_ENABLED=true

# Cache database location (SQLite)
RAG_DATA_PATH=./rag_data  # Cache stored at rag_data/cache.db

# Cache expiry in days (0 = never expire)
LLM_CACHE_MAX_AGE_DAYS=30

# Enable semantic drift detection (only runs on updates when version > 1)
ENABLE_SEMANTIC_DRIFT_DETECTION=true
```

**N.B.:** The cache is automatically created at `rag_data/cache.db` on first use. No manual setup required.

## Cache Storage

- **Format**: SQLite database with Write-Ahead Logging (WAL) for concurrent access
- **Location**: `rag_data/cache.db` (auto-created on first use)
- **Schema**:
```sql
CREATE TABLE llm_cache (
    content_hash TEXT PRIMARY KEY,  -- SHA256 hash of prompt
    prompt TEXT NOT NULL,           -- Full prompt text
    response TEXT NOT NULL,         -- LLM response
    model TEXT,                     -- Model name (optional)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hits INTEGER DEFAULT 0          -- Cache hit counter
);
```

**Performance Features:**
- Thread-safe with connection pooling
- WAL mode for concurrent read/write
- Automatic indexing on `hits` column for analytics
- Hit counter tracks cache efficiency

## Cache Operations

### Automatic Cache Management
- Cache is initialised on first database operation
- Changes are committed automatically per operation
- No manual save/flush required (SQLite handles persistence)
- Thread-safe for concurrent ingestion workers

### Manual Cache Operations

**View Cache Stats:**
```python
from scripts.ingest.cache_db import CacheDB
from pathlib import Path

cache = CacheDB(rag_data_path=Path("rag_data"))
stats = cache.llm_stats()
print(f"Total entries: {stats['entries']}")
print(f"Total cache hits: {stats['total_hits']}")
print(f"Database path: {stats['db_path']}")
```

**Clear LLM Cache:**
```python
cache.clear_llm_cache()  # Removes all LLM cache entries
```

**Retrieve Cached Result:**
```python
response = cache.get_llm_result(prompt="Extract metadata from this text", model="llama3.2")
if response:
    print("Cache hit!")
```

**Store New Result:**
```python
cache.put_llm_result(
    prompt="Extract metadata from this text",
    response="{...}",
    model="llama3.2"
)
```

## Performance Impact

### Expected Speedup
- **First ingestion**: No speedup (cache is empty)
- **Re-ingestion (no changes)**: ~80-90% faster (skips all LLM calls)
- **Re-ingestion (minor changes)**: ~50-70% faster (skips unchanged chunks)

### Example Timing
**Without cache:**
- Text cleaning: 2-3s per document
- Metadata extraction: 3-4s per document
- Summary scoring: 2s per document
- Chunk validation: 1s per chunk
- **Total**: ~10-15s per document + chunks

**With cache (unchanged content):**
- All LLM operations: <0.1s (cache lookup)
- **Total**: ~1-2s per document + chunks

## Cache Invalidation

Cache entries are automatically invalidated when:
1. **Document content changes**: SHA-256 hash of file content changes, creating new cache key
2. **Cache expiry**: Entry exceeds `LLM_CACHE_MAX_AGE_DAYS` (if age-based cleanup implemented)
3. **Manual clearing**: Via `clear_llm_cache()` method

**Hash Change Scenarios:**
- File modification (even whitespace changes trigger new SHA-256)
- Re-extraction (same content = same hash = cache hit)
- Chunk splitting changes (different chunk text = different SHA-256)

## Best Practices

1. **Set appropriate expiry**: 30 days works well for stable documents
2. **Monitor cache size**: Use `llm_stats()` to track entry count and hit rate
3. **Back up cache database**: SQLite files can be backed up with simple file copy:
   ```bash
   cp rag_data/cache.db rag_data/cache.db.backup
   ```
4. **Clear stale entries**: Periodically clear cache for removed documents or reset workflows
5. **Use WAL mode**: Enabled by default for concurrent read/write safety
6. **Check hit rate**: High `total_hits / entries` ratio indicates good cache efficiency

## Troubleshooting

**Cache not working:**
- Check `LLM_CACHE_ENABLED=true` in [.env](../.env)
- Verify `rag_data/` directory is writable
- Check logs for cache hits/misses
- Verify SQLite is available: `python3 -c "import sqlite3; print(sqlite3.sqlite_version)"`

**Cache database too large:**
- Check database size: `du -h rag_data/cache.db`
- Run `VACUUM` to reclaim space: `sqlite3 rag_data/cache.db "VACUUM;"`
- Clear old entries: `cache.clear_llm_cache()`
- Consider archiving old cache: `mv cache.db cache.db.old && restart ingestion`

**Stale results:**
- Document SHA-256 hash should change when content changes
- Force re-processing by clearing cache: `cache.clear_llm_cache()`
- Verify hash calculation: check file content actually changed
- For debugging: Query cache directly:
  ```bash
  sqlite3 rag_data/cache.db "SELECT content_hash, created_at, hits FROM llm_cache LIMIT 10;"
  ```

**Database locked errors:**
- WAL mode should prevent most locks (auto-enabled)
- If issues persist, increase `busy_timeout` in `cache_db.py`
- Check for stale write locks: close all connections to database
