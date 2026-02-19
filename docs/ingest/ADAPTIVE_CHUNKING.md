# Adaptive Chunking

## Overview

Adaptive chunking automatically adjusts chunk size and overlap based on document type and content structure. This optimisation ensures that different document types are chunked appropriately for optimal retrieval quality.

## Problem Statement

Static chunking (fixed 800 chars, 150 overlap) doesn't account for:
- **Policy documents**: Concise bullet points and rules → too much context per chunk
- **Technical guides**: Step-by-step procedures → too little context per chunk
- **Content structure**: Heading density and sentence length affect ideal chunk boundaries

## Solution

Dynamic chunk sizing based on:
1. **Document type keywords** → Base chunk size
2. **Content analysis** → Fine-tuning adjustments

## How It Works

### Base Sizes by Document Type

The system recognises keywords in `doc_type` to select initial chunk size:

| Document Type | Keywords | Chunk Size | Rationale |
|--------------|----------|------------|-----------|
| **Policy/Compliance** | policy, compliance, standard, requirement, regulation, control, guideline, principle, rule | 600 chars | Concise rules and lists benefit from smaller, focused chunks |
| **Guides/Procedures** | guide, procedure, tutorial, how-to, walkthrough, instruction, deployment, implementation, setup | 1000 chars | Step-by-step content needs larger chunks for complete context |
| **Architecture/Patterns** | architecture, pattern, design, framework, model, blueprint, structure, concept | 1000 chars | Complex concepts require more context |
| **Reference/API** | reference, api, specification, documentation, manual | 800 chars | Balanced documentation |
| **Default** | (no match) | 800 chars | Original static default |

### Content Structure Adjustments

After selecting base size, the system analyses content to fine-tune:

| Content Characteristic | Adjustment | Reason |
|----------------------|------------|---------|
| **High heading density** (>15 headings per 1000 chars) | -200 chars | Already well-structured, smaller chunks work well |
| **Long sentences** (avg >150 chars) | +200 chars | Complex prose needs more context |
| **Short sentences** (avg <60 chars) | -100 chars | Bullet points/lists are self-contained |

**Bounds**: Final chunk_size is clamped to 400-1200 chars.

### Overlap Calculation

Overlap is always **19% of chunk_size** to maintain consistent context preservation across chunk boundaries.

## Examples

### Policy Document

```python
doc_type = "compliance policy"
text = "Security policy: Users must authenticate. MFA required. Password min 12 chars..."

chunk_size, overlap = determine_chunk_params(doc_type, text)
# Result: chunk_size=500, overlap=95
# (600 base - 100 for short sentences)

chunks = chunk_text(text, doc_type=doc_type)
# Creates ~6 chunks for 2900 chars (smaller, focused chunks)
```

### Deployment Guide

```python
doc_type = "deployment guide"
text = "Step 1: Install prerequisites. Download package. Verify checksums. Extract..."

chunk_size, overlap = determine_chunk_params(doc_type, text)
# Result: chunk_size=900, overlap=171
# (1000 base - 100 for short sentences)

chunks = chunk_text(text, doc_type=doc_type)
# Creates ~4 chunks for 2900 chars (larger, comprehensive chunks)
```

### Architecture Document

```python
doc_type = "system architecture"  
text = "Microservices architecture with REST APIs. Each service manages database..."

chunk_size, overlap = determine_chunk_params(doc_type, text)
# Result: chunk_size=1000, overlap=190
# (1000 base, no adjustments)

chunks = chunk_text(text, doc_type=doc_type)
# Creates ~5 chunks for 2900 chars (large chunks for complex concepts)
```

## Configuration

### Enable/Disable Adaptive Chunking

```python
# Adaptive mode (default)
chunks = chunk_text(text, doc_type="policy", adaptive=True)

# Static mode (basic 800/150)
chunks = chunk_text(text, adaptive=False)
```

### Adding Custom Doc Type Keywords

Edit [scripts/ingest/chunk.py](../scripts/ingest/chunk.py) in `determine_chunk_params()`:

```python
# Add new document type category
elif any(keyword in doc_type_lower for keyword in [
    'report', 'analysis', 'assessment'  # Your keywords
]):
    base_size = 700  # Your preferred size
```

### Tuning Content Analysis Thresholds

Adjust in `determine_chunk_params()`:

```python
# Current thresholds
if heading_density > 0.15:  # Adjust this
    adjustment -= 200  # Or this

if avg_sentence_length > 150:  # Adjust this
    adjustment += 200  # Or this
```

## Integration with Ingestion Pipeline

Adaptive chunking is automatically enabled in [scripts/ingest/ingest.py](../scripts/ingest/ingest.py):

```python
# Extract doc_type from preprocessing
doc_type = preprocessed_text.get("doc_type")

# Chunk with adaptive sizing
chunks = chunk_text(
    preprocessed_text["cleaned_text"],
    doc_type=doc_type,
    adaptive=True
)
```

The LLM-generated `doc_type` from metadata extraction flows directly into chunk size determination.

## Performance Impact

### Chunk Count Comparison (2900 char document)

| Mode | Chunks | Avg Size |
|------|--------|----------|
| Policy (adaptive) | 7 | 449 chars |
| Static (800/150) | 5 | 580 chars |
| Guide (adaptive) | 4 | 753 chars |
| Architecture (adaptive) | 5 | 915 chars |

### Expected Benefits

1. **Better Retrieval Precision**
   - Policy chunks: More focused, less noise
   - Guide chunks: Complete procedures, better context
   
2. **Improved Context Quality**
   - Complex docs: Larger chunks preserve relationships
   - Simple docs: Smaller chunks reduce irrelevant context
   
3. **Consistent Behaviour**
   - 19% overlap maintains context across all sizes
   - Bounds (400-1200) prevent extreme values

## Testing

Run adaptive chunking tests:
```bash
pytest tests/test_adaptive_chunking.py -v
```

Test coverage includes:
- ✅ Doc type keyword recognition (19 tests)
- ✅ Content structure analysis (heading density, sentence length)
- ✅ Chunk size bounds enforcement
- ✅ Overlap calculation
- ✅ Realistic document examples

## Monitoring

### Audit Trail

Check chunking decisions in logs:
```bash
grep "Created.*chunks" logs/ingest.log
```

Example output:
```
Created 7 chunks (doc_type: compliance policy)
Created 4 chunks (doc_type: deployment guide)
```

### Retrieval Quality

Monitor RAG answer quality and adjust thresholds if:
- **Too much noise**: Chunks too large → reduce base_size
- **Insufficient context**: Chunks too small → increase base_size
- **Poor boundaries**: Adjust content analysis thresholds

## Tuning Recommendations

### For Policy-Heavy Collections
```python
# Decrease policy base size for even more focused chunks
if 'policy' in doc_type_lower:
    base_size = 500  # Down from 600
```

### For Technical Documentation
```python
# Increase guide base size for more comprehensive chunks
if 'guide' in doc_type_lower:
    base_size = 1200  # Up from 1000
```

### For Heading-Dense Documents
```python
# Make heading density adjustment more aggressive
if heading_density > 0.10:  # Lower threshold
    adjustment -= 300  # Larger reduction
```

## Default Behaviour

```python
# default (uses adaptive=True by default)
chunks = chunk_text(text)

# explicit
chunks = chunk_text(text, doc_type="policy", adaptive=True)

# disable adaptive
chunks = chunk_text(text, adaptive=False)
```

## Related Features

- **LLM Metadata Extraction**: Generates `doc_type` used for chunking
- **Chunk Heuristic Validation**: Works with adaptive chunks of any size
- **Semantic Drift Detection**: Operates on adaptive chunk collections
- **Vector Embeddings**: Adaptive chunks maintain similar semantic density

## Troubleshooting

### Chunks Too Large/Small

**Symptom**: Retrieval returning too much/little context  
**Solution**: Adjust base_size for the affected doc_type category

### Inconsistent Chunking

**Symptom**: Same doc_type producing very different chunk counts  
**Solution**: Check content analysis adjustments (heading density, sentence length)

### Unknown Doc Types Not Chunking Well

**Symptom**: Documents without matching keywords chunked poorly  
**Solution**: Add new keyword category or improve LLM doc_type prompts

### Debug Chunk Parameters

```python
from chunk import determine_chunk_params

chunk_size, overlap = determine_chunk_params("your doc_type", your_text)
print(f"Will use: {chunk_size} chars with {overlap} overlap")
```

## TODO: Future Enhancements

Potential improvements:
- [ ] ML-based chunk size prediction
- [ ] Per-collection chunking profiles
- [ ] Dynamic overlap based on content complexity
- [ ] Chunk quality scoring feedback loop
- [ ] A/B testing framework for chunk parameters
