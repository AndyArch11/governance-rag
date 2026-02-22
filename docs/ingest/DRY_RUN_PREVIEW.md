# Dry-Run Preview Report Implementation

## Overview

Reporting for the `--dry-run` CLI argument that shows detected documents, estimated chunks, LLM operations, embedding calls, cost estimates, and processing time before actual ingestion.

All three ingestion scripts support `--dry-run`:
- `scripts/ingest/ingest.py` - Generic documents (PDF, HTML, text)
- `scripts/ingest/ingest_git.py` - Code repositories (Bitbucket, GitHub, GitLab, Azure DevOps)
- `scripts/ingest/ingest_academic.py` - Academic papers and references

## Quick Reference: Dry-Run Operations

| Operation | ingest.py | ingest_git.py | ingest_academic.py |
|-----------|-----------|---------------|-------------------|
| **Document/file discovery** | ✅ Performed | ✅ Performed | ✅ Performed |
| **Text extraction** | ✅ Performed | ✅ Performed | ✅ Performed |
| **LLM preprocessing** | ✅ Performed | ✅ Performed | ✅ Performed |
| **Code parsing (AST)** | N/A | ✅ Performed | N/A |
| **Citation extraction** | N/A | N/A | ✅ Performed |
| **Provider API calls** | N/A | N/A | ❌ **Skipped** |
| **Reference downloads** | N/A | N/A | ❌ **Skipped** |
| **Chunking pipeline** | ✅ Performed | ✅ Performed | ✅ Performed |
| **Chunk validation** | ✅ Performed | ✅ Performed | ✅ Performed |
| **Embedding generation** | ✅ Performed | ✅ Performed | ✅ Performed |
| **ChromaDB writes** | ❌ **Skipped** | ❌ **Skipped** | ❌ **Skipped** |
| **BM25 indexing** | ❌ **Skipped** | ❌ **Skipped** | ❌ **Skipped** |
| **Cache persistence** | ❌ **Skipped** | ❌ **Skipped** | ❌ **Skipped** |
| **Citation graph write** | N/A | N/A | ❌ **Skipped** |
| **Terminology DB write** | N/A | N/A | ❌ **Skipped** |

**Key Benefits:**
- **Accurate estimation:** Processes documents fully for realistic time/cost projections
- **Rate limit preservation:** Academic script skips external API calls
- **Fast preview:** Skips all persistence operations (no database writes)

---

## Usage

### DryRunStats Class

`/workspaces/governance-rag/scripts/ingest/ingest.py`

**Purpose:** Thread-safe statistics collector for dry-run preview metrics.

**N.B.:** This class and formatted preview report are currently only implemented in `ingest.py`. The `ingest_git.py` and `ingest_academic.py` scripts support `--dry-run` mode and skip persistence operations, but emit log messages instead of a formatted report.

**Tracked Metrics:**
- **Documents:** New, updated, and skipped counts
- **Chunks:** Total chunks that would be created, average per document
- **LLM Operations:**
  - Metadata generation calls (1 per document)
  - Chunk validation calls (estimated 20% of chunks)
  - Chunk repair calls (estimated 5% of chunks)
- **Embedding Operations:**
  - Chunk embeddings (1 per chunk)
  - Document embeddings (1 per document)
- **Cost Estimates:** Cloud API pricing (for reference, Ollama is free)
- **Time Estimates:** Sequential and parallel processing time

### Associated Functions

**1. `process_file()`**
- `dry_run_stats` optional parameter
- Records document status (new/updated/skipped) in dry-run mode
- Tracks chunk count and processing time
- Thread-safe statistics collection via lock

**2. `main()`**
- Initialises `DryRunStats` when `--dry-run` flag is set
- Passes stats collector to all `process_file()` calls
- Displays formatted preview report after processing
- Emits `dry_run_preview` audit event with summary

### Preview Report Format (ingest.py only)

```
╔══════════════════════════════════════════════════════════════════╗
║                      DRY-RUN PREVIEW REPORT                      ║
╠══════════════════════════════════════════════════════════════════╣
║ DOCUMENT SUMMARY                                                 ║
║   Total documents discovered:      X                             ║
║   New documents:                   X                             ║
║   Updated documents:               X                             ║
║   Skipped (unchanged):             X                             ║
║   Documents to process:            X                             ║
╠══════════════════════════════════════════════════════════════════╣
║ CHUNK ANALYSIS                                                   ║
║   Total chunks to create:          X                             ║
║   Avg chunks per doc:            X.X                             ║
╠══════════════════════════════════════════════════════════════════╣
║ LLM OPERATIONS                                                   ║
║   Metadata generation:             X calls                       ║
║   Chunk validation:                X calls (est. 20%)            ║
║   Chunk repair:                    X calls (est. 5%)             ║
║   Total LLM calls:                 X                             ║
╠══════════════════════════════════════════════════════════════════╣
║ EMBEDDING OPERATIONS                                             ║
║   Chunk embeddings:                X calls                       ║
║   Document embeddings:             X calls                       ║
║   Total embeddings:                X                             ║
╠══════════════════════════════════════════════════════════════════╣
║ ESTIMATED COSTS (Cloud API Reference)                            ║
║   LLM calls:                  $   X.XX                           ║
║   Embeddings:                 $   X.XX                           ║
║   Total (if using cloud):     $   X.XX                           ║
║   Note: Ollama (local) is FREE                                   ║
╠══════════════════════════════════════════════════════════════════╣
║ ESTIMATED PROCESSING TIME                                        ║
║   Sequential processing:      X minutes/hours                    ║
║   With parallelisation:       X minutes/hours   (4 workers)      ║
╚══════════════════════════════════════════════════════════════════╝

To proceed with ingestion, run without --dry-run flag.
```

## Usage Examples

### ingest.py (Generic Documents)

#### Basic Dry-Run (Check All Documents)
```bash
python scripts/ingest/ingest.py --dry-run
```

#### Dry-Run with Limit (Quick Preview)
```bash
python scripts/ingest/ingest.py --dry-run --limit 10
```

#### Dry-Run with Reset (Preview Full Ingestion)
```bash
python scripts/ingest/ingest.py --dry-run --reset
```

#### Dry-Run with Custom Workers
```bash
python scripts/ingest/ingest.py --dry-run --workers 8
```

### ingest_git.py (Code Repositories)

#### Basic Dry-Run (Bitbucket)
```bash
python scripts/ingest/ingest_git.py --dry-run \
  --provider bitbucket \
  --host https://bitbucket.org/workspace/repo \
  --username user \
  --password token
```

#### Dry-Run with GitHub
```bash
python scripts/ingest/ingest_git.py --dry-run \
  --provider github \
  --host https://github.com/owner/repo \
  --token ghp_xxxx
```

#### Dry-Run with File Type Filter
```bash
python scripts/ingest/ingest_git.py --dry-run \
  --provider bitbucket \
  --file-types ".py,.js,.ts" \
  --username user --password token
```

#### Dry-Run with Repo Reset
```bash
python scripts/ingest/ingest_git.py --dry-run --reset-repo \
  --provider gitlab --host https://gitlab.com/group/repo
```

### ingest_academic.py (Academic References)

#### Basic Dry-Run (Single Paper)
```bash
python scripts/ingest/ingest_academic.py --dry-run \
  data_raw/academic_papers/paper.pdf
```

#### Dry-Run Batch Processing
```bash
python scripts/ingest/ingest_academic.py --dry-run \
  --batch data_raw/academic_references.txt
```

#### Dry-Run Papers Directory
```bash
python scripts/ingest/ingest_academic.py --dry-run \
  --papers-dir data_raw/academic_papers/
```

#### Dry-Run with Reset
```bash
python scripts/ingest/ingest_academic.py --dry-run --reset
```

#### Dry-Run Citation Extraction Only
```bash
python scripts/ingest/ingest_academic.py --dry-run \
  --title "Paper Title" \
  --authors "Author1, Author2" \
  paper.pdf
```

## Dry-Run Behaviour by Script

### ingest.py (Generic Documents)

**Purpose:** Preview ingestion of PDFs, HTML, and generic text documents.

**Operations Performed:**
- Document discovery and hash computation
- Full text extraction (PDF/HTML parsing)
- LLM-based preprocessing and metadata generation
- Semantic chunking and validation
- Chunk quality heuristics
- Embedding generation (via cache or Ollama)
- Cost and time estimation

**Operations Skipped:**
- ChromaDB writes (chunk/document storage)
- BM25 index writes
- Cache persistence (LLM/embedding results are computed but not saved)
- Drift detection updates

**Output:**
- Formatted preview report (document counts, chunk analysis, LLM/embedding operations, cost estimates, time estimates)
- Audit event: `dry_run_preview` with summary statistics
- Real processing time measured for accuracy

**N.B.:** Dry-run still consumes LLM/embedding resources for accurate estimation.

---

### ingest_git.py (Code Repositories)

**Purpose:** Preview ingestion of source code from Git repositories (Bitbucket, GitHub, GitLab, Azure DevOps).

**Operations Performed:**
- Repository cloning or pulling (unless `--no-refresh`)
- File discovery with extension filtering
- Encoding detection and file content reading
- Code parsing (TreeSitter AST analysis)
- Function/class extraction and documentation parsing
- Code summary generation (if enabled with `--generate-summaries`)
- Chunk creation (parent-child or regular)
- Document ID and hash computation

**Operations Skipped:**
- ChromaDB writes (chunks, documents)
- BM25 keyword indexing
- Cache updates (LLM, embedding)
- Drift detection hash storage

**Output:**
- Log messages: `[DRY-RUN] Would process <file_path> (doc_id=<id>)`
- Log messages: `[DRY-RUN] Would store <n> chunks for <doc_id>`
- Log messages: `[DRY-RUN] Would store <n> child chunks for <doc_id>`
- Log messages: `[DRY-RUN] Would store <n> parent chunks for <doc_id>`
- Audit events still emitted for file processing stages

**Configuration:**
- CLI flag: `--dry-run`
- Environment variable: `GIT_DRY_RUN=true`
- Precedence: CLI overrides environment

**Use Cases:**
- Preview file counts and language distribution before full ingestion
- Verify file type filtering (`--file-types`)
- Test repository connection and authentication
- Estimate chunk counts for capacity planning

---

### ingest_academic.py (Academic References)

**Purpose:** Preview ingestion of academic papers with citation extraction and reference resolution.

**Operations Performed:**
- PDF text extraction from thesis/paper documents
- Citation extraction (regex-based or LLM-based)
- Citation normalisation and deduplication
- Reference resolution metadata creation (placeholder `Reference` objects)
  - **N.B.:** Provider chain (Crossref, Semantic Scholar, etc.) is **not called** in dry-run mode to avoid rate limit consumption and external API calls
- Reference cache lookups (read-only)
- Full chunking pipeline (parent-child or regular)
- Citation graph construction (in-memory)
- Domain terminology extraction (if enabled)

**Operations Skipped:**
- Reference downloads (PDFs from providers)
- Provider API calls (Crossref, Semantic Scholar, Unpaywall, etc.)
- ChromaDB writes (chunks, documents)
- BM25 keyword indexing
- Citation graph database write (`academic_citation_graph.db`)
- Reference cache writes (`academic_references.db`)
- Domain terminology database writes (`academic_terminology.db`)
- Academic PDF cache writes

**Output:**
- Log message: `[DRY_RUN] Dry run enabled: skipping downloads`
- Log message: `[DRY_RUN] Would store <n> chunks for <doc_id> (title: <title>... doi: <doi> provider: <source>)`
- Log message: `[DRY_RUN] Would store <n> child chunks and <n> parent chunks for <doc_id>`
- Log message: `[DRY_RUN] Would write citation graph to <path>`
- Log message: `[DRY_RUN] Would reset collections and caches` (if `--reset` flag used)
- Audit events with `"dry_run": true` metadata

**Configuration:**
- CLI flag: `--dry-run`
- Environment override: `ACADEMIC_INGEST_DRY_RUN=True` (via config overrides)

**Use Cases:**
- Preview citation extraction quality without provider API consumption
- Verify reference parsing logic on thesis documents
- Test chunking strategy for academic content
- Estimate reference counts before full ingestion and download
- Validate citation graph structure before persistence

**Important Notes:**
- **Provider rate limits preserved:** No external academic provider APIs are called during dry-run
- **Reference resolution quality:** Dry-run creates placeholder references with partial metadata; actual resolution quality can only be assessed in live mode
- **Download cost preview:** Dry-run shows how many references would be downloaded but doesn't estimate download time or storage size

---

## Cost Estimation Details

**Cloud API Pricing (Reference Only):**
- **Mistral LLM:** ~$0.0002 per 1K tokens
  - Average 500 tokens/call = **$0.0001 per call**
- **Embedding API:** ~$0.0001 per 1K tokens
  - Average 200 tokens/embedding = **$0.00002 per call**

**N.B:** Ollama (local deployment) is **completely free** - costs shown are for cloud API comparison only.

## Statistics Collection Logic

### Document Counting
- **New:** Document doesn't exist in ChromaDB (version 1)
- **Updated:** Document exists with different hash (version > 1)
- **Skipped:** Document exists with identical hash (no changes)

### LLM Call Estimates
- **Metadata Generation:** 1 call per new/updated document (100% of docs)
- **Chunk Validation:** ~20% of chunks require semantic validation
- **Chunk Repair:** ~5% of chunks require LLM-based repair

### Embedding Call Estimates
- **Chunk Embeddings:** 1 per chunk (100% of chunks)
- **Document Embeddings:** 1 per document (100% of new/updated docs)

### Time Estimation
- **Sequential:** Sum of actual processing times from dry-run execution
- **Parallel:** Estimated time / number of workers (default: 4)
- Includes: parsing, preprocessing, LLM calls, chunking, validation, embeddings

## Implementation Notes

### Thread Safety
- `DryRunStats` uses `threading.Lock()` for concurrent updates
- Safe for multi-threaded `ThreadPoolExecutor` processing
- No race conditions in statistic collection

### Dry-Run Behaviour (ingest.py)
- **Still processes documents fully** (parse, preprocess, chunk, validate)
- **Only skips:** ChromaDB writes (add/update/delete operations)
- **Benefits:** Accurate time/cost estimates based on actual execution
- **Trade-off:** Dry-run still consumes resources (LLM calls, embeddings)

**N.B.** Future plans are to provide `--dry-run` and `--profile` options that emulate LLM calls without consuming any LLM credits to gauge approximate LLM credit consumption. These future options will be useful for LLM consumption estimates, but will not be useful for optimisation recommendations.

**N.B.** For script-specific dry-run behaviour, see [Dry-Run Behaviour by Script](#dry-run-behaviour-by-script) above.

### Audit Integration
- Emits `dry_run_preview` audit event with summary statistics
- Includes: document counts, chunk totals, operation counts, time estimates
- Queryable via `logs/ingest_audit.jsonl` for historical analysis

Check `logs/ingest_audit.jsonl` for dry-run preview events


## Benefits

### For Users
1. **Preview Before Execution:** See what will happen before committing
2. **Cost Estimation:** Understand cloud API costs if migrating from Ollama
3. **Time Planning:** Estimate job duration for scheduling
4. **Sanity Checking:** Verify document counts and detection logic

### For Operators
1. **Capacity Planning:** Estimate resource requirements for large batches
2. **Debugging:** Identify unexpected document counts or chunk distributions
3. **Optimisation:** Compare chunk counts across different document types
4. **Validation:** Confirm ingestion configuration before production runs

## Limitations

### Current Implementation
- **Estimates Only:** Validation/repair percentages are heuristic (20%/5%)
- **Actual May Vary:** Real LLM calls depend on chunk quality
- **Still Processes Docs:** Dry-run isn't "free" - uses LLM/embedding resources
- **No Cache Accounting:** Doesn't factor in LLM/embedding cache hit rates

### TODO: Future Enhancements
1. **Fast Dry-Run Mode:** Skip LLM calls, use heuristics only
2. **LLM/Embedding Cache Tracking:** Instrument cache hits in profile stats
3. **Cache-Aware Estimates:** Factor in cache hit rates for cost reduction
4. **Historical Analysis:** Compare estimates to actual runs
5. **Performance Comparison:** Compare against previous runs
6. **Stratified Sampling:** Sample across document types/sizes
7. **Per-Category Breakdown:** Show stats by document source category
8. **Resource Monitoring:** Track GPU/CPU/memory utilisation
9. **GPU Utilisation Estimates:** Predict VRAM usage for local deployment
10. **Auto-Tuning:** Suggest optimal MAX_WORKERS based on observed parallelism
11. **Regression Detection:** Alert when performance degrades
12. **Export Metrics:** JSON export for dashboards

## Compatibility

**Python Version:** 3.10+  
**Dependencies:** Ollama and ChromaDB  
**Performance Impact:** Negligible (statistics collection overhead < 1ms/doc)  
**Supported Scripts:**
- `scripts/ingest/ingest.py` - Generic document ingestion (PDF, HTML, text)
- `scripts/ingest/ingest_git.py` - Code repository ingestion (Bitbucket, GitHub, GitLab, Azure DevOps)
- `scripts/ingest/ingest_academic.py` - Academic paper and reference ingestion
