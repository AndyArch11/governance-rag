# RAG Enhancement - Quick Reference

## Feature Toggle Quick Reference

```python
from scripts.rag.retrieve import retrieve_with_filters
from pathlib import Path

chunks, metadata = retrieve_with_filters(
    query="your query here",
    collection=chroma_collection,
    k=5,
    
    # Always Recommended
    auto_detect_filters=True,    # Auto-detect metadata filters
    enable_reranking=True,       # Re-rank by relevance
    fetch_neighbours=False,      # Expand with prev/next chunks
    enable_caching=True,         # Cache hot entities
    enable_graph=True,           # Graph-based expansion
    cache_dir=Path("logs/"),     # Cache file location
    
    # Academic Features
    persona="supervisor"         # Persona-aware filtering (supervisor/assessor/researcher)
)
```

## Recommended Settings by Use Case

### Maximum Quality (Default)
```python
retrieve_with_filters(
    query=query,
    collection=collection,
    k=5,
    auto_detect_filters=True,
    enable_reranking=True,
    enable_caching=True,
    enable_graph=True,
    fetch_neighbours=True  # Full context expansion
)
```

### Maximum Performance
```python
retrieve_with_filters(
    query=query,
    collection=collection,
    k=5,
    auto_detect_filters=True,
    enable_reranking=False,  # Skip re-ranking
    enable_caching=True,     # Cache is fast
    enable_graph=False,      # Skip graph expansion
    fetch_neighbours=False
)
```

### Resource-Constrained
```python
retrieve_with_filters(
    query=query,
    collection=collection,
    k=3,                     # Fewer results
    auto_detect_filters=True,
    enable_reranking=True,   # Lightweight, worth it
    enable_caching=True,     # Reduces future queries
    enable_graph=False,      # Skip graph overhead
    fetch_neighbours=False
)
```

## Cache Management

```python
from scripts.rag.context_cache import get_context_cache
from pathlib import Path

# Create cache pointing to rag_data
cache = get_context_cache(cache_dir=Path("rag_data/"))

# Check statistics
stats = cache.get_stats()
print(f"Cached: {stats['total_entries']}, Hits: {stats['total_accesses']}")

# Manual operations
cache.invalidate("entity_name")  # Remove one
cache.clear()                    # Remove all
```

## Graph Operations

```python
from scripts.rag.graph_retrieval import get_graph_retriever

retriever = get_graph_retriever()

# Expand with neighbors
expanded = retriever.expand_with_neighbors(
    chunk_ids=["chunk_1", "chunk_2"],
    max_hops=1,       # 1 or 2 recommended
    max_neighbors=3   # Chunks per hop
)

# Find by entity
chunks = retriever.expand_with_entities(
    entities=["auth_service", "database"],
    max_chunks_per_entity=5
)

# Detect conflicts
conflicts = retriever.find_conflicting_chunks(["chunk_1"])
```

## Persona-Aware Retrieval (Academic Queries)

```python
from scripts.rag.retrieve import retrieve_with_filters

# For thesis supervisors - foundational understanding
chunks, metadata = retrieve_with_filters(
    query="What are the core compliance requirements?",
    collection=collection,
    k=5,
    persona="supervisor",  # Filters to academic/report/preprint, min quality 0.6
    enable_reranking=True
)

# For thesis assessors - verification focus
chunks, metadata = retrieve_with_filters(
    query="What evidence supports this claim?",
    collection=collection,
    k=5,
    persona="assessor",  # Requires verifiable sources, excludes stale links
    enable_reranking=True
)

# For researchers - novelty discovery
chunks, metadata = retrieve_with_filters(
    query="What are the latest preprints on this topic?",
    collection=collection,
    k=5,
    persona="researcher",  # Prefers recent preprints, min quality 0.4
    enable_reranking=True
)
```

### Persona Configurations

| Persona | Depth | Min Quality | Min Citations | Preferred Types | Stale Links | Recency Bias |
|---------|-------|-------------|---------------|-----------------|-------------|-------------|
| **Supervisor** | 2 | 0.6 | 10 | academic, report, preprint | ✓ (penalty 0.2) | 0.1 |
| **Assessor** | 3 | 0.7 | 5 | academic, report | ✗ | 0.0 |
| **Researcher** | 1 | 0.4 | 0 | preprint, academic, report | ✓ (penalty 0.3) | 0.4 |

### Persona Metadata

When `persona` is set, retrieved chunks include:
```python
{
    # Standard fields
    "text": "...",
    "reference_type": "academic",  # academic, preprint, report, news, blog, online
    "quality_score": 0.85,
    "citation_count": 42,
    "link_status": "available",    # available, stale_404, stale_timeout, stale_moved
    "year": 2024,
    
    # Persona enrichment
    "persona": "supervisor",
    "persona_score": 0.82          # Combined score (similarity + quality + citations + recency)
}
```

## Parent-Child Chunking

```python
from scripts.ingest.chunk import create_parent_child_chunks

# Create parent-child hierarchy
children, parents = create_parent_child_chunks(
    text=document_text,
    doc_type="technical",  # or "governance", "api_reference"
    parent_size=1200,      # Tokens in parent chunk
    child_size=400         # Tokens in child chunk
)

# children: Small chunks for embedding/search
# parents: Large chunks for LLM context
```

## Metadata Fields

When enhanced metadata is enabled, each chunk has:

```python
{
    # Core
    "chunk_id": "chunk_123",
    "text": "...",
    "source_file": "doc.html",
    
    # Hierarchy
    "heading_path": "Config > Database > Pool",
    "parent_section": "Database Configuration",
    "section_depth": 3,
    
    # Sequence
    "prev_chunk_id": "chunk_122",
    "next_chunk_id": "chunk_124",
    
    # Technical
    "technical_entities": ["auth_service", "db_config"],
    "code_language": "python",
    
    # Flags
    "contains_code": True,
    "contains_table": False,
    "contains_diagram": False,
    "is_api_reference": True,
    "is_configuration": True,
    
    # Classification
    "content_type": "api_documentation",
    "source_category": "governance"
}
```

## Performance Expectations

| Operation | Latency | Notes |
|-----------|---------|-------|
| Cache hit | <1ms | Instant |
| Basic retrieval | 30-50ms | Semantic search only |
| + Re-ranking | +10-15ms | Lightweight scoring |
| + Graph expansion | +20-30ms | Neighbor lookup |
| + Neighbor fetch | +10-20ms | Prev/next chunks |
| **Full pipeline** | **100-200ms** | All features enabled |

## Troubleshooting

### Cache not working
- Check `enable_caching=True`
- Verify `cache_dir` is writable
- Check TTL not expired (default 1 hour)

### Graph not expanding
- Verify `consistency_graph.sqlite` exists in `rag_data/consistency_graphs/`
- Check graph has `chunk_id` in metadata
- Look for "Graph enhancement enabled" in logs

### Low quality results
- Enable `auto_detect_filters=True`
- Enable `enable_reranking=True`
- Try `fetch_neighbors=True` for continuity
- Check metadata richness with enhanced chunking

## Quick Test

```python
# Verify everything works
from scripts.rag.context_cache import get_context_cache
from scripts.rag.graph_retrieval import get_graph_retriever
from scripts.ingest.chunk import create_parent_child_chunks

cache = get_context_cache(enabled=True)
retriever = get_graph_retriever()
children, parents = create_parent_child_chunks("Test " * 100, "technical")

print(f"✓ Cache: {cache._cache is not None}")
print(f"✓ Graph: {retriever.graph is not None}")
print(f"✓ Chunking: {len(children)} children, {len(parents)} parents")
```

## Key Files

```
scripts/rag/
  ├── retrieve.py            # Retrieval system
  ├── context_cache.py       # Caching system
  └── graph_retrieval.py     # Graph expansion

scripts/ingest/
  ├── chunk.py               # Enhanced chunking + parent-child
  ├── schemas.py             # Metadata schemas
  ├── ingest.py              # Pipeline integration
  └── vectors.py             # Enhanced storage

tests/
  ├── test_context_cache.py  
  └── test_graph_retrieval.py 
```

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review documentation in `docs/`
3. Run tests: `pytest tests/test_context_cache.py tests/test_graph_retrieval.py -v`
