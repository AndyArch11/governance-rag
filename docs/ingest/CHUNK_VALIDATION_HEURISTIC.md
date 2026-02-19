# Chunk Validation Heuristic

## Overview

The chunk validation heuristic is a performance optimisation that uses fast, lightweight checks to identify high-quality chunks that can skip expensive LLM-based semantic validation. This significantly reduces API calls and processing time during ingestion.

## How It Works

### Heuristic Checks

The system evaluates each chunk using three metrics:

1. **Length Check**
   - Minimum: 100 characters
   - Maximum: 2000 characters
   - Rationale: Too short indicates fragments, too long indicates concatenation errors

2. **Stopword Ratio**
   - Threshold: ≥ 15% of words must be common English stopwords
   - Common words: "the", "a", "is", "and", "of", "to", "in", etc.
   - Rationale: Natural language contains predictable patterns of function words

3. **Boilerplate Detection**
   - Checks for ≥ 3 boilerplate patterns in chunk
   - Patterns: copyright, privacy policy, terms of service, navigation text, etc.
   - Rationale: Boilerplate content often needs validation/repair

4. **Entropy**
   - Minimum: 3.0 (character-level Shannon entropy)
   - Rationale: Low entropy indicates repetitive or low-information content

### Decision Logic

```python
skip_validation = (
    100 <= length <= 2000
    AND stopword_ratio >= 0.15
    AND boilerplate_patterns < 3
    AND entropy >= 3.0
)
```

If all checks pass → **SKIP** semantic validation (high confidence chunk is clean)  
If any check fails → **VALIDATE** with LLM (chunk may need repair)

## Configuration

Enable/disable the heuristic in [.env](../.env) (or see [.env.example](../.env.example)):

```bash
# Enable chunk quality heuristic (default: true)
ENABLE_CHUNK_HEURISTIC_SKIP=true
```

When disabled, all chunks undergo semantic validation regardless of quality indicators.

## Performance Impact

### Expected Reduction in LLM Calls

Based on typical document quality:
- **Well-formatted documents** (wikis, documentation): 70-90% of chunks skip validation
- **Mixed-quality documents** (web scrapes, exports): 40-60% skip
- **Poor-quality documents** (OCR, raw HTML): 10-30% skip

### Time Savings

Assuming 1 second per LLM validation call:
- 100 chunks @ 70% skip rate: ~70 seconds saved
- 1000 chunks @ 50% skip rate: ~500 seconds (8+ minutes) saved

Heuristic overhead: < 1ms per chunk

## Metrics and Logging

The system logs heuristic decisions:

```json
{
  "event": "chunk_heuristic_skip",
  "doc_id": "azure_guardrails",
  "chunk_id": "azure_guardrails-chunk-5",
  "confidence": 0.78,
  "reason": "heuristic_pass"
}
```

Summary logged at end of validation:
```
INFO: Heuristic skipped semantic validation for 42/60 chunks
```

### Audit Trail

Check `logs/ingest_audit.jsonl` for:
- `chunk_heuristic_skip` events for skipped chunks
- `chunk_semantic_validation` events for validated chunks

## Examples

### ✓ SKIP - Natural Language
```
This is a well-formed chunk of text that contains natural language 
patterns. It has sufficient length and includes common English 
stopwords. The content provides meaningful information and demonstrates 
coherent sentence structure.

Confidence: 0.74, Reason: heuristic_pass
```

### ✓ SKIP - Technical Content
```
Azure Virtual Machines provide scalable compute resources in the cloud. 
Configure instance types, networking, and storage based on workload 
requirements. Implement monitoring and auto-scaling policies to optimise 
performance and cost.

Confidence: 0.74, Reason: heuristic_pass
```

### ✗ VALIDATE - Boilerplate
```
Copyright 2024. All rights reserved. Terms of Service | Privacy Policy | 
Cookie Settings | Contact Us | About | FAQ

Confidence: 0.30, Reason: boilerplate_detected_6_patterns
```

### ✗ VALIDATE - Too Short
```
Click here

Confidence: 0.20, Reason: too_short
```

### ✗ VALIDATE - Low Stopword Ratio
```
AZURE_STORAGE_CONNECTION_STRING DATA_PROCESSOR_ENABLED CHUNK_SIZE 
OVERLAP VALIDATION_THRESHOLD RETRY_COUNT TIMEOUT_SECONDS

Confidence: 0.30, Reason: low_stopword_ratio_0.08
```

## Implementation Details

### Code Location

- Heuristic function: [scripts/ingest/vectors.py](../scripts/ingest/vectors.py) - `compute_chunk_quality_heuristic()`
- Integration point: [scripts/ingest/vectors.py](../scripts/ingest/vectors.py) - `process_and_validate_chunks()`
- Tests: [tests/test_chunk_heuristic.py](../tests/test_chunk_heuristic.py)

### Return Values

```python
Tuple[bool, float, str]
  - skip_validation: True if chunk appears clean
  - confidence: 0.0-1.0 quality score
  - reason: Explanation (e.g., "heuristic_pass", "too_short", "low_stopword_ratio_0.12")
```

### Confidence Calculation

For chunks passing all checks:
```python
confidence = min(0.95, 0.5 + (stopword_ratio * 0.5) + ((entropy - 3.0) / 10.0))
```

Higher stopword ratio and entropy → higher confidence.

## Tuning

### Notebook-Based Assessment

Use the tuning notebook for a practical quality assessment and threshold recommendations:

- Notebook: [notebooks/chunk_quality_tuning_advisor.ipynb](../notebooks/chunk_quality_tuning_advisor.ipynb)
- Inputs:
   - Chroma chunk collection (default source)
   - Optional `logs/ingest_audit.jsonl` heuristic skip events
- Outputs:
   - Skip/validate rate summary
   - Failure reason distribution (`too_short`, `too_long`, `low_stopword_ratio_*`, `low_entropy_*`, `boilerplate_detected_*`)
   - Guide-aligned tuning suggestions based on observed reason patterns

Recommended workflow:
1. Run ingestion on a representative sample corpus.
2. Execute the notebook end-to-end to profile current chunk quality.
3. Apply threshold changes in [scripts/ingest/vectors.py](../scripts/ingest/vectors.py) if needed.
4. Re-ingest the same sample and compare results in the notebook.
5. Promote tuned values once skip rate and quality trade-offs are acceptable.

### Adjusting Thresholds

Edit [scripts/ingest/vectors.py](../scripts/ingest/vectors.py) in `compute_chunk_quality_heuristic()`:

```python
# Current values (conservative - fewer skips, higher validation)
MIN_LENGTH = 100
MAX_LENGTH = 2000
MIN_STOPWORD_RATIO = 0.15
MIN_ENTROPY = 3.0
BOILERPLATE_THRESHOLD = 3

# More aggressive (skip more chunks, faster but riskier)
MIN_STOPWORD_RATIO = 0.10  # Lower threshold
MIN_ENTROPY = 2.5          # Lower threshold
BOILERPLATE_THRESHOLD = 4  # More tolerant

# More conservative (validate more chunks, slower but safer)
MIN_STOPWORD_RATIO = 0.20  # Higher threshold
MIN_ENTROPY = 3.5          # Higher threshold
BOILERPLATE_THRESHOLD = 2  # Less tolerant
```

### Adding Custom Patterns

To detect domain-specific boilerplate:

```python
boilerplate_patterns = [
    # Default patterns
    'copyright', 'all rights reserved', 'terms of service', 'privacy policy',
    
    # Add your custom patterns
    'internal use only', 'confidential', 'draft version',
    'page X of Y', 'table of contents', 'references'
]
```

## Relationship to Other Optimisations

This feature complements existing optimisations:

1. **LLM Cache**: Skips repeated LLM calls for same content
2. **Drift Detection**: Skips re-validation when content unchanged
3. **Chunk Heuristic** (this feature): Skips validation for obviously clean chunks

Together, these reduce LLM calls by 80-95% on typical re-ingestion workflows.

## Testing

Run heuristic tests:
```bash
pytest tests/test_chunk_heuristic.py -v
```

Test with your own chunks:
```python
from scripts.ingest.vectors import compute_chunk_quality_heuristic

chunk = "Your chunk text here..."
skip, confidence, reason = compute_chunk_quality_heuristic(chunk)
print(f"Skip: {skip}, Confidence: {confidence:.2f}, Reason: {reason}")
```

## Troubleshooting

### Too Many Chunks Being Validated

**Symptom**: Low skip rate (< 30%) on good-quality documents  
**Solution**: Lower stopword/entropy thresholds or check for domain-specific patterns

### Too Many Chunks Being Skipped

**Symptom**: Invalid chunks making it through without validation  
**Solution**: Raise thresholds or disable heuristic temporarily with:
```bash
ENABLE_CHUNK_HEURISTIC_SKIP=false
```

### Unexpected Validation Reasons

Check audit log for patterns:
```bash
grep "chunk_heuristic_skip" logs/ingest_audit.jsonl | jq '.data.reason' | sort | uniq -c
```

Common reasons:
- `too_short`: Increase chunk size in chunking config
- `low_stopword_ratio`: Check for technical jargon or non-English content
- `boilerplate_detected`: Review boilerplate patterns, may need customisation
- `low_entropy`: Check for repetitive content or formatting issues

## Best Practices

1. **Enable by default** for production workloads (significant performance gain)
2. **Disable temporarily** when debugging chunk quality issues
3. **Monitor skip rates** in logs to ensure appropriate tuning
4. **Customise boilerplate patterns** for your document domain
5. **Adjust thresholds** based on your content quality and validation requirements

## Related Documentation

- [LLM Cache](./LLM_CACHE.md) - Persistent caching of LLM outputs
- [Adaptive Chunking](./ADAPTIVE_CHUNKING.md) - Dynamic overlap and semantic-aware chunking
- [Logger Consolidation](./LOGGER_CONSOLIDATION.md) - Audit logging and structured tracing
