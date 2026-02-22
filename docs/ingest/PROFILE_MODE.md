# Profile Mode - Quick Validation Run

## Overview

Profile mode provides a quick validation run that processes a small sample of documents with detailed timing and error analysis. This helps operators validate their configuration and identify potential issues before running expensive full ingestion jobs.

## Key Features

- **Auto-Configuration:** Automatically sets `--limit 10` and `--verbose`
- **Detailed Timing:** Tracks time spent in each processing stage
- **Error Analysis:** Collects all errors and warnings with context
- **Performance Metrics:** Measures throughput, cache hit rates, and bottlenecks
- **Optimisation Recommendations:** Provides actionable suggestions for improving performance
- **Production Readiness:** Clear assessment of whether pipeline is ready for full run

## Usage

### Basic Profile Run

```bash
python ingest.py --profile
```

This will:
- Process the first 10 documents
- Enable verbose output
- Display detailed timing and error analysis
- Provide optimisation recommendations

### Custom Sample Size

```bash
python ingest.py --profile --limit 5
```

Process only 5 documents (override default of 10).

### Profile with Reset

```bash
python ingest.py --profile --reset
```

Clear ChromaDB collections and profile with fresh data.

### Profile URL Seeds

```bash
python ingest.py --profile --include-url-seeds
```

Profile mode with URL seed ingestion included (preview).

## Report Format

The profile report includes the following sections:

### 1. Processing Summary
- Documents processed
- Total chunks created
- Total elapsed time
- Throughput (docs/sec)

### 2. Document Timing
- Average time per document
- Fastest document time
- Slowest document time

### 3. Stage Breakdown
- Parse: HTML parsing time
- Preprocess: LLM metadata generation time
- Chunk: Text chunking time
- Validate: Chunk validation/repair time
- Embed: Embedding generation time
- Store: ChromaDB storage time
- **Bottleneck:** Slowest stage with percentage

### 4. API Efficiency
- LLM calls (total and per document average)
- LLM cache hit rate
- Embedding calls
- Embedding cache hit rate

### 5. Quality Metrics
- Chunks per document average
- Errors encountered
- Warnings raised

### 6. Optimisation Recommendations

Actionable suggestions based on observed metrics:
- **High validation overhead:** Enable `ENABLE_CHUNK_HEURISTIC_SKIP`
- **Low cache hit rates:** Ensure caching is enabled
- **High chunk count:** Consider increasing chunk size
- **Slow throughput:** Consider increasing `MAX_WORKERS`
- **High error rate:** Review errors before full run

### 7. Production Readiness

Clear assessment:
- ✅ **READY FOR PRODUCTION RUN** - No errors detected
- ⚠️ **REVIEW REQUIRED** - Errors need attention

## Example Output

```
╔══════════════════════════════════════════════════════════════════╗
║                      PROFILE ANALYSIS REPORT                     ║
╠══════════════════════════════════════════════════════════════════╣
║ PROCESSING SUMMARY                                               ║
║   Documents processed:             10                            ║
║   Total chunks created:           245                            ║
║   Total elapsed time:            32.5s                           ║
║   Throughput:                   0.31 docs/sec                    ║
╠══════════════════════════════════════════════════════════════════╣
║ DOCUMENT TIMING (seconds)                                        ║
║   Average per document:          3.25s                           ║
║   Fastest document:              1.20s                           ║
║   Slowest document:              6.80s                           ║
╠══════════════════════════════════════════════════════════════════╣
║ STAGE BREAKDOWN                                                  ║
║   Parse:                         2.10s                           ║
║   Preprocess:                    8.50s                           ║
║   Chunk:                         1.30s                           ║
║   Validate:                     12.40s                           ║
║   Embed:                         6.20s                           ║
║   Store:                         2.00s                           ║
║   Bottleneck:                 Validate (38.0% of total)          ║
╠══════════════════════════════════════════════════════════════════╣
║ API EFFICIENCY                                                   ║
║   LLM calls:                       85 ( 8.5 per doc)             ║
║   LLM cache hit rate:            45.2%                           ║
║   Embedding calls:                255                            ║
║   Embedding cache rate:          62.4%                           ║
╠══════════════════════════════════════════════════════════════════╣
║ QUALITY METRICS                                                  ║
║   Chunks per document:           24.5                            ║
║   Errors encountered:               0                            ║
║   Warnings raised:                  2                            ║
╠══════════════════════════════════════════════════════════════════╣
║ OPTIMISATION RECOMMENDATIONS                                     ║
║ • Enable ENABLE_CHUNK_HEURISTIC_SKIP to reduce validation        ║
║   overhead                                                       ║
║ ✓ Pipeline configuration looks optimal                           ║
╚══════════════════════════════════════════════════════════════════╝

⚠️  WARNINGS:
  1. doc_123: Chunk required repair (validation failed)
  2. doc_456: Summary regenerated due to low score

════════════════════════════════════════════════════════════════════
✅ READY FOR PRODUCTION RUN
   No errors detected. Pipeline configured correctly.
   Remove --profile flag to process full dataset.
```

## Benefits

### For Operators

1. **Fast Validation:** < 1 minute to validate configuration
2. **Cost Awareness:** See API usage patterns before expensive runs
3. **Error Detection:** Catch configuration issues early
4. **Performance Tuning:** Identify bottlenecks and optimise before full run

### For Developers

1. **Debugging Aid:** Quick iteration during development
2. **Regression Testing:** Verify changes don't slow pipeline
3. **Baseline Metrics:** Establish performance baselines
4. **Configuration Guidance:** Data-driven tuning recommendations

## Integration with Other Modes

Profile mode is complementary to dry-run mode:

| Mode | Purpose | Writes to DB | Processing | Uses LLM Credits | Output |
|------|---------|-------------|------------|--------|
| **Normal** | Full ingestion | ✅ Yes | Complete | Yes | Progress bars |
| **--dry-run** | Cost estimation | ❌ No | Complete | ⚠️ Yes | Progress bars and Preview report |
| **--profile** | Quick validation | ✅ Yes | Sample (10 docs) | ⚠️ Yes | Timing analysis and Recommendations |

Recommended workflow:
2. **First run:** `--profile` to validate configuration and performance, incurs some LLm credits
1. **Second run:** `--dry-run` to validate corpus is able to be ingested without errors, incurs same LLM credits as a Production run
3. **Production:** Remove flags for full ingestion

**N.B.** Future plans are to provide `--dry-run` and `--profile` options that emulate LLM calls without consuming any LLM credits to gauge approximate LLM credit consumption. These future options will be useful for LLM consumption estimates, but will not be useful for optimisation recommendations.

## Implementation Details

### ProfileStats Class

`/workspaces/governance-rag/scripts/ingest/ingest.py`

Thread-safe statistics collector tracking:
- Document count and chunk count
- Stage-level timing (6 stages)
- Errors and warnings with context
- LLM and embedding call counts
- Cache hit rates (LLM and embedding)
- Individual document processing times

### Auto-Configuration Logic

When `--profile` is specified:
```python
if args.profile:
    args.verbose = True  # Enable detailed output
    if not args.limit:
        args.limit = 10  # Default sample size
```

### Metrics Collection

Statistics are collected throughout `process_file()`:
- **Document completion:** Record chunk count and total time
- **Stage timings:** Record time spent in each phase
- **Errors:** Capture all exceptions with document context
- **Cache events:** Track LLM and embedding cache hits

### Report Generation

The `ProfileStats.get_report()` method:
1. Calculates aggregate statistics (avg, min, max)
2. Identifies bottleneck stage (highest time percentage)
3. Computes cache hit rates
4. Generates optimisation recommendations
5. Formats Unicode box report
6. Appends error/warning details
7. Provides production readiness assessment

## Limitations

1. **Sample Size:** 10 documents may not represent full dataset characteristics
2. **No Semantic Drift:** Profile mode doesn't test drift detection on updates
3. **Cold Start:** First run may show lower cache hit rates than typical
4. **Statistical Variance:** Small sample size means high variance in timings

## TODO: Future Enhancements

Potential improvements for future versions:

1. **Stratified Sampling:** Sample across document types/sizes
2. **Performance Comparison:** Compare against previous profile runs
3. **Resource Monitoring:** Track GPU/CPU/memory utilisation
4. **Export Metrics:** JSON export for dashboards/monitoring
5. **Auto-Tuning:** Suggest optimal MAX_WORKERS based on observed parallelism
6. **Regression Detection:** Alert when performance degrades vs baseline

## Configuration Reference

Profile mode respects all standard configuration:

```bash
# In .env file
MAX_WORKERS=4              # Affects parallelism
LLM_CACHE_ENABLED=true     # Affects cache hit rates
ENABLE_CHUNK_HEURISTIC_SKIP=true  # Affects validation time
LLM_RATE_LIMIT=10.0        # Affects LLM call throttling
```

## Troubleshooting

### "No documents found"
- Check `RAG_BASE_PATH` points to directory with HTML files
- Verify file patterns match expected naming

### High error rate
- Review error messages in report
- Check Ollama service is running: `ollama list`
- Verify ChromaDB is accessible

### Low throughput
- Increase `MAX_WORKERS` (if GPU has capacity)
- Enable `ENABLE_CHUNK_HEURISTIC_SKIP=true`
- Check Ollama GPU utilisation: `nvidia-smi`

### Low cache hit rates
- Expected on first run (cold cache)
- Ensure `LLM_CACHE_ENABLED=true` and `EMBEDDING_CACHE_ENABLED=true`
- Run profile mode twice to see warm cache performance

## Related Documentation

- [DRY_RUN_PREVIEW.md](DRY_RUN_PREVIEW.md) - Cost estimation without processing
- [RETRY_IMPLEMENTATION.md](RETRY_IMPLEMENTATION.md) - Resilience and retry logic
- [.env.example](../.env.example) - Complete configuration reference
