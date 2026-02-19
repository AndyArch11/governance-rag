# Persona-Aware Retrieval Configuration Guide

## Overview

Persona-aware retrieval applies domain-specific filtering and reranking to academic queries based on user role (supervisor, assessor, or researcher). This feature is designed for thesis evaluation and academic research workflows.

## Quick Start

### Dashboard Usage

1. Navigate to the **Chat** tab in the dashboard
2. Select a persona from the dropdown (default: None)
3. Enter your query
4. Results are automatically filtered and reranked based on persona rules

### Programmatic Usage

```python
from scripts.rag.retrieve import retrieve_with_filters

chunks, metadata = retrieve_with_filters(
    query="What are the compliance requirements?",
    collection=chroma_collection,
    k=5,
    persona="supervisor"  # or "assessor", "researcher"
)
```

### Answer Generation

```python
from scripts.rag.generate import answer

response = answer(
    query="What are the compliance requirements?",
    llm=ollama_llm,
    persona="supervisor"
)
```

## Persona Configurations

### Supervisor

**Use Case:** Thesis supervisors seeking foundational understanding of research topics

**Configuration:**
```python
PersonaConfig(
    persona="supervisor",
    reference_depth=2,              # Follow citations 2 levels deep
    min_quality_score=0.6,          # Require moderate quality
    min_citation_count=10,          # Require established sources
    prefer_reference_types=("academic", "report", "preprint"),
    include_stale_links=True,       # Include but penalise stale links
    require_verifiable=False,       # Allow unverified sources
    stale_link_penalty=0.2,         # Light penalty for stale links
    recency_bias=0.1                # Slight preference for recent
)
```

**Filtering Rules:**
- ✅ Academic papers, technical reports, preprints
- ✅ Quality score ≥ 0.6
- ✅ Citation count ≥ 10
- ✅ Includes stale links (with -0.2 score penalty)
- ❌ News, blogs, online sources

**Best For:**
- Literature reviews
- Background research
- Foundational understanding
- Broad topic exploration

### Assessor

**Use Case:** Thesis assessors verifying claims and checking evidence

**Configuration:**
```python
PersonaConfig(
    persona="assessor",
    reference_depth=3,              # Deep citation verification
    min_quality_score=0.7,          # Require high quality
    min_citation_count=5,           # Require peer-reviewed work
    prefer_reference_types=("academic", "report"),
    include_stale_links=False,      # Exclude stale links entirely
    require_verifiable=True,        # Only verifiable sources
    stale_link_penalty=0.5,         # Heavy penalty if link fails
    recency_bias=0.0                # No recency preference
)
```

**Filtering Rules:**
- ✅ Academic papers and technical reports only
- ✅ Quality score ≥ 0.7
- ✅ Citation count ≥ 5
- ✅ Active, verifiable links only
- ❌ Preprints, news, blogs, stale links

**Best For:**
- Plagiarism detection preparation
- Evidence verification
- Claim substantiation
- Academic rigor assessment

### Researcher

**Use Case:** Researchers discovering novel approaches and recent developments

**Configuration:**
```python
PersonaConfig(
    persona="researcher",
    reference_depth=1,              # Focus on direct sources
    min_quality_score=0.4,          # Accept lower quality for novelty
    min_citation_count=0,           # Accept uncited work
    prefer_reference_types=("preprint", "academic", "report"),
    include_stale_links=True,       # Include stale links
    require_verifiable=False,       # Allow unverified sources
    stale_link_penalty=0.3,         # Moderate penalty for stale links
    recency_bias=0.4                # Strong preference for recent work
)
```

**Filtering Rules:**
- ✅ Preprints, academic papers, technical reports
- ✅ Quality score ≥ 0.4 (accepts lower quality)
- ✅ Citation count ≥ 0 (accepts uncited work)
- ✅ Includes stale links (with -0.3 score penalty)
- ✅ Strong recency bias (+0.4 for recent publications)
- ❌ News, blogs (unless academic context)

**Best For:**
- Cutting-edge research discovery
- Preprint scanning
- Emerging trends identification
- Novel methodology exploration

## Configuration Comparison

| Aspect | Supervisor | Assessor | Researcher |
|--------|-----------|----------|-----------|
| **Reference Depth** | 2 | 3 | 1 |
| **Min Quality** | 0.6 | 0.7 | 0.4 |
| **Min Citations** | 10 | 5 | 0 |
| **Allowed Types** | academic, report, preprint | academic, report | preprint, academic, report |
| **Stale Links** | Include (penalty -0.2) | Exclude | Include (penalty -0.3) |
| **Verifiable Only** | No | Yes | No |
| **Recency Bias** | +0.1 | 0.0 | +0.4 |

## Metadata Requirements

Persona filtering requires these metadata fields on ingested documents:

### Required Fields
```python
{
    "reference_type": str,      # academic, preprint, report, news, blog, online
    "quality_score": float,     # 0.0 - 1.0
    "citation_count": int,      # Number of citations
}
```

### Optional Fields
```python
{
    "link_status": str,         # available, stale_404, stale_timeout, stale_moved
    "year": int,                # Publication year (for recency scoring)
    "publication_year": int,    # Alternative field name
    "domain_relevance_score": float,  # 0.0 - 1.0
    "relevance_score": float,   # Alternative field name
}
```

## Scoring Formula

Each chunk receives a weighted persona score:

```python
score = (
    0.5 * similarity_score        # Semantic similarity
    + 0.2 * quality_score         # Source quality
    + 0.15 * domain_relevance     # Domain match
    + 0.1 * citation_score        # Normalised citations (min(1.0, count/100))
    + 0.05 * recency_score        # Recency (1.0 - years_old/10)
    + recency_bias * recency_score  # Persona-specific recency boost
    - stale_link_penalty          # Penalty for broken links
)
```

### Similarity Score Sources
1. **ChromaDB distance:** `similarity = max(0.0, 1.0 - distance)`
2. **BM25 TF score:** `similarity = min(1.0, tf_score / max_tf)`

### Recency Score
```python
years_old = current_year - publication_year
recency_score = max(0.0, 1.0 - (years_old / 10.0))
```

### Citation Score
```python
citation_score = min(1.0, citation_count / 100.0)
```

## Implementation Details

### Filtering Pipeline

1. **Reference Type Filter:** Exclude non-preferred types
2. **Verifiability Filter:** (Assessor only) Exclude non-available links
3. **Stale Link Filter:** (Assessor only) Exclude stale_* link statuses
4. **Quality Filter:** Exclude below min_quality_score
5. **Citation Filter:** Exclude below min_citation_count
6. **Scoring:** Apply weighted formula with persona adjustments
7. **Ranking:** Sort by persona_score descending
8. **Top-K:** Return top_k results

### Integration Points

**Retrieval Layer:**
```python
# scripts/rag/retrieve.py
from scripts.search.persona_retrieval import apply_persona_reranking

if persona:
    chunks, metadata = apply_persona_reranking(
        chunks=chunks,
        metadata=metadata,
        persona=persona,
        top_k=k
    )
```

**Answer Generation:**
```python
# scripts/rag/generate.py
def answer(query, llm, persona=None, ...):
    chunks, metadata = retrieve(
        query=query,
        collection=collection,
        persona=persona,
        k=k
    )
    # Build prompt and generate
```

**Dashboard:**
```python
# scripts/consistency_graph/dashboard.py
dcc.Dropdown(
    id="rag-query-persona",
    options=[
        {"label": "None", "value": "none"},
        {"label": "Supervisor", "value": "supervisor"},
        {"label": "Assessor", "value": "assessor"},
        {"label": "Researcher", "value": "researcher"},
    ]
)
```

## Testing

Run persona retrieval tests:
```bash
pytest tests/test_persona_retrieval.py -v
```
## Troubleshooting

### No results returned

**Problem:** Persona filtering is too strict for your corpus

**Solutions:**
1. Use `researcher` persona (most permissive)
2. Check metadata completeness:
   ```python
   # Verify metadata
   for meta in metadata:
       print(f"Type: {meta.get('reference_type')}, "
             f"Quality: {meta.get('quality_score')}, "
             f"Citations: {meta.get('citation_count')}")
   ```
3. Lower thresholds by modifying persona config in `scripts/search/persona_retrieval.py`

### Stale links included unexpectedly

**Problem:** Supervisor/Researcher include stale links by design

**Solution:** 
- Use `assessor` persona for verified-only results
- Or filter in post-processing:
  ```python
  verified = [(c, m) for c, m in zip(chunks, metadata) 
              if m.get("link_status") == "available"]
  ```

### Recency bias too strong/weak

**Problem:** Recent papers over-ranked or under-ranked

**Solution:**
- Adjust `recency_bias` in persona config (0.0 = no bias, 0.4 = strong bias)
- For custom bias, modify scoring in `apply_persona_reranking()`

### Persona not applied

**Problem:** Results identical regardless of persona selection

**Checklist:**
1. Verify persona parameter passed to `retrieve()`:
   ```python
   print(f"Using persona: {persona}")
   ```
2. Check metadata has required fields:
   ```python
   assert "reference_type" in metadata[0]
   assert "quality_score" in metadata[0]
   ```
3. Verify persona enrichment in results:
   ```python
   assert "persona" in metadata[0]
   assert "persona_score" in metadata[0]
   ```

## Advanced Configuration

### Custom Persona

Define custom persona in `scripts/search/persona_retrieval.py`:

```python
CUSTOM_CONFIG = PersonaConfig(
    persona="custom",
    reference_depth=2,
    min_quality_score=0.5,
    min_citation_count=3,
    prefer_reference_types=("academic", "preprint", "report", "news"),
    include_stale_links=True,
    require_verifiable=False,
    stale_link_penalty=0.25,
    recency_bias=0.2,
)

def get_persona_config(persona: str) -> PersonaConfig:
    persona_key = (persona or "").strip().lower()
    if persona_key == "custom":
        return CUSTOM_CONFIG
    # ... existing code
```

### Modify Scoring Weights

Edit scoring formula in `apply_persona_reranking()`:

```python
# Default: 50% similarity, 20% quality, 15% domain, 10% citations, 5% recency
score = (
    0.5 * similarity_score
    + 0.2 * quality_score
    + 0.15 * domain_relevance
    + 0.1 * citation_score
    + 0.05 * recency_score
)

# Custom: Emphasise quality and citations
score = (
    0.3 * similarity_score
    + 0.3 * quality_score    # Increased from 0.2
    + 0.1 * domain_relevance # Decreased from 0.15
    + 0.2 * citation_score   # Increased from 0.1
    + 0.1 * recency_score    # Increased from 0.05
)
```

### Link Status Thresholds

Customise stale link handling:

```python
# Default: All stale statuses penalised equally
if link_status != "available":
    score -= config.stale_link_penalty

# Custom: Different penalties by status type
penalties = {
    "stale_404": 0.5,      # Severe penalty
    "stale_timeout": 0.3,  # Moderate penalty
    "stale_moved": 0.1,    # Light penalty (redirect works)
}
score -= penalties.get(link_status, 0.0)
```

## Best Practices

1. **Match persona to task:**
   - Broad exploration → Supervisor
   - Verification/assessment → Assessor
   - Novel discovery → Researcher

2. **Check metadata coverage:**
   - Ensure ingestion populates required fields
   - Validate quality scores are calibrated (0.0-1.0 range)
   - Verify citation counts from metadata providers

3. **Monitor persona effectiveness:**
   ```python
   # Log persona application
   logger.info(f"Persona: {persona}, Results: {len(chunks)}, "
               f"Avg Score: {sum(m['persona_score'] for m in metadata) / len(metadata)}")
   ```

4. **Combine with other features:**
   - Use with `enable_reranking=True` for best results
   - Enable caching for repeated persona queries
   - Apply filters before persona reranking for efficiency

## Related Documentation

- [Academic Ingestion Policy](ACADEMIC_INGESTION_POLICY.md) - Governance and persona definitions
- [Academic Persona Retrieval](ACADEMIC_PERSONA_RETRIEVAL.md) - Design rationale
- [RAG Quick Reference](RAG_QUICK_REFERENCE.md) - General retrieval configuration
- [RAG Query Quick Start](RAG_QUERY_QUICK_START.md) - Dashboard usage guide

## API Reference

### get_persona_config()

```python
def get_persona_config(persona: str) -> PersonaConfig:
    """Return persona configuration by name.
    
    Args:
        persona: Persona name (supervisor/assessor/researcher)
        
    Returns:
        PersonaConfig instance
        
    Raises:
        ValueError: If persona is unknown
    """
```

### apply_persona_reranking()

```python
def apply_persona_reranking(
    chunks: List[str],
    metadata: List[Dict[str, Any]],
    persona: str,
    top_k: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Filter and rerank chunks based on persona rules.
    
    Args:
        chunks: Retrieved text chunks
        metadata: Metadata for each chunk
        persona: Persona name (supervisor/assessor/researcher)
        top_k: Result size after reranking
        
    Returns:
        Filtered and reranked (chunks, metadata) tuple
        
    Raises:
        ValueError: If persona is unknown
    """
```

---

**File:** `docs/academic/PERSONA_RETRIEVAL_CONFIG.md`  
**Tests:** `tests/test_persona_retrieval.py`  
**Dashboard:** Integrated with persona dropdown  
