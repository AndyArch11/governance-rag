# Academic Document Ingestion: Scope & Policy

---

## Table of Contents

1. [Overview](#overview)
2. [Shared-Domain Assumptions](#shared-domain-assumptions)
3. [Reference Caching & Deduplication](#reference-caching--deduplication)
4. [Metadata Provider Strategy](#metadata-provider-strategy)
5. [Download & Storage Policy](#download--storage-policy)
6. [Persona-Aware Retrieval Parameters](#persona-aware-retrieval-parameters)
7. [Rate Limiting & Resource Constraints](#rate-limiting--resource-constraints)
8. [Environment Configuration](#environment-configuration)
9. [Error Recovery & Audit](#error-recovery--audit)

---

## Overview

The academic ingestion system processes one or more academic documents (theses, papers, publications) within a **shared domain context**. It:

- **Parses** PDF/text content and extracts citations
- **Resolves** references via metadata providers (Crossref, OpenAlex, Semantic Scholar, ORCID, Google Scholar, Unpaywall, direct URL fetch)
- **Classifies** reference types (academic papers, preprints, news articles, blog posts, online content)
- **Scores** reference quality (0.0-1.0 based on type, venue, citations, accessibility)
- **Downloads** open-access PDFs where available
- **Handles** paywalled content (metadata-only extraction)
- **Tracks** provenance and versioning (source snapshots, provider response IDs, access timestamps)
- **Ingests** referenced materials and creates citation graph edges
- **Assesses** domain-specific terminology across primary and reference materials
- **Deduplicates** references across multiple input documents to avoid redundant fetches/storage
- **Supports** three query personas (Supervisor, Assessor, Researcher) with distinct retrieval strategies

---

## Shared-Domain Assumptions

### Multi-Document Ingestion Model

When multiple academic documents are provided:

1. **Shared Domain Scope**
   - All documents are assumed to cover the same or closely related domains
   - Example: "Machine Learning in Healthcare" thesis references another "ML for Diagnosis" paper
   - Cross-document reference deduplication enables knowledge reuse

2. **Domain Boundary Recognition**
   - Individual documents may have tangential coverage of other domains (e.g., statistical methods in an ML thesis)
   - Flag such coverage but do not exclude references
   - Metadata enrichment helps categorise cross-domain references

3. **Terminology Extraction**
   - **Default behaviour**: Assess words across all ingested material (primary + references) for domain-specific terms
   - Build domain vocabulary incrementally as documents are processed
   - Use existing terminology for subsequent reference resolution (improve recall)

### Per-Document Domain Tagging

Each `AcademicDocument` must capture:
- **primary_domain**: e.g., "machine_learning", "healthcare_informatics"
- **secondary_domains**: e.g., ["statistics", "data_engineering"]
- **topic**: narrow focus (e.g., "federated learning for cardiac monitoring")
- **terminology_tags**: extracted domain-specific terms for keyword filtering

---

## Reference Caching & Deduplication

### Cache Scope

A **reference cache** stores metadata and download status across an ingestion batch:

```
Cache Key: (doi, title_hash, first_author_normalised)
Cache Value: {
  doi: str,
  title: str,
  authors: [str],
  year: int,
  abstract: str,
  oa_url: str (optional),
  pdf_local_path: str (optional),
  metadata_provider: str,
  resolved_at: timestamp,
  doc_ids: [str] → documents that cite this reference
}
```

### Deduplication Rules

When processing multiple documents:

1. **Citation Matching**
   - Normalise raw citation text: lowercase, strip punctuation, split by semicolon/comma
   - Match against cache using fuzzy string matching (>90% similarity) + year proximity (±1 year)

2. **Reuse Logic**
   - If reference found in cache with status `resolved`:
     - Skip metadata provider calls; use cached result
     - Append current doc_id to reference's doc_ids list
     - If PDF/web artifact already downloaded, reuse local path
   - If reference in cache but download failed:
     - Retry once per session (track retry_count)
     - Log retry outcome to audit

3. **Cache Eviction**
   - Persist cache to SQLite (reuses existing cache infra)
   - Scope: per ingestion batch (session)
   - TTL: 30 days (configurable); can be cleared via `--cache-reset`
   - Non-PDF artifacts (web pages, text files) stored alongside PDFs with content-type metadata

### Storage Integration

Uses shared `scripts.utils.db_factory` pattern:

```python
cache_db = get_cache_client(enable_cache=True)
cache_db.put_academic_reference(ref_id, ref_metadata)
cache_db.get_academic_reference(ref_id)
cache_db.get_academic_references_by_doc(doc_id)
```

---

## Metadata Provider Strategy

### Provider Priority Chain

Resolve metadata in strict order:

1. **Crossref** (DOI lookup)
   - Fast, authoritative for journal articles
   - Rate limit: 50 req/sec (per Crossref policy)
   - Timeout: 10 sec

2. **OpenAlex** (fallback if no DOI)
   - Broad coverage (papers, datasets, authors)
   - Rate limit: 100k req/day (permissive)
   - Returns abstract + OA status

3. **Semantic Scholar** (alternative if OpenAlex fails)
   - Strong NLP-indexed metadata
   - Rate limit: 100 req/sec
   - Includes citation context

4. **ORCID** (author disambiguation)
   - Retrieve author ORCID IDs for author linking
   - Rate limit: 24 req/sec (public API tier)
   - Used for author verification and publication lists

5. **Unpaywall** (OA discovery)
   - Called after Crossref/OpenAlex to find OA PDF URLs
   - Rate limit: 100k req/day
   - Authoritative on OA status + PDF hosting

6. **Google Scholar** (metadata-only, fallback for hard-to-find content)
   - Fallback for rare/old papers and non-academic content
   - Rate limit: 5 req/sec (self-imposed, respect robots.txt)
   - **Policy**: metadata-only; do NOT download PDFs from Scholar links
   - Useful for news articles, blog posts, online content metadata

7. **URL Fetch** (direct content retrieval for non-academic references)
   - For news articles, blog posts, online journal articles with public URLs
   - Rate limit: 5 req/sec (self-imposed)
   - Handles paywall detection (metadata-only if paywalled)
   - TODO: Use newspaper3k for content extraction

### Reference Type Classification

The system classifies references into types with associated quality scores:

| Reference Type | Quality Score | Examples | Providers |
|---------------|---------------|----------|-----------|
| Academic (peer-reviewed) | 1.0 | Journal articles, conference papers with DOI | Crossref, OpenAlex, Semantic Scholar |
| Preprint | 0.8 | arXiv, bioRxiv | OpenAlex, arXiv API |
| Technical Report | 0.7 | White papers, institutional reports | ORCID, Google Scholar |
| News Article | 0.5 | NYTimes, Guardian, Reuters | URL Fetch, Google Scholar |
| Blog Post | 0.4 | Medium, Substack | URL Fetch, Google Scholar |
| Online Content | 0.3 | Generic web content | URL Fetch |

Quality scores are adjusted based on:
- Presence of DOI (+0.1 to +0.2)
- Venue prestige (top journals +0.1)
- Citation count (>100: +0.05, >50: +0.03, >10: +0.01)
- OA availability (+0.05)
- Paywall detected (-0.1)

### Paywall Handling

When paywalled content is detected:
1. Extract metadata only (title, author, abstract from meta tags)
2. Set `paywall_detected=True` and `oa_available=False`
3. Store reference with metadata but no full text
4. Apply quality penalty (-0.1 to final score)
5. Flag in audit trail for user awareness

### Online Content Staleness Tracking

For non-academic references (news, blogs, online articles):
1. Record `accessed_at` timestamp when content is fetched
2. Cache staleness check: warn if content is >30 days old (configurable)
3. Audit trail logs stale references for potential re-validation
4. Users can trigger re-fetch for updated content
5. Rationale: News articles can be updated/corrected; blog posts edited; links go stale

**Staleness Thresholds (TODO):**
- News articles: 30 days
- Blog posts: 60 days
- Online content: 90 days
- Academic PDFs: No staleness check (static)

#### Implementation Status (as of February 2026)

- Current runtime behaviour uses a single threshold (default 30 days) via `--staleness-threshold`.
- Type-specific thresholds (news/blog/online) are policy intent and are not yet enforced in runtime selection logic.
- The notebook `notebooks/academic_reference_staleness_check.ipynb` currently uses a single `STALE_THRESHOLD_DAYS` value.

#### Implementation Guide: Type-Specific Staleness

1. Add a threshold mapping by reference type:
  - `news: 30`, `blog: 60`, `online: 90`, default fallback `30`
2. Apply mapping in stale selection logic for `--revalidate stale`:
  - File: `scripts/ingest/ingest_academic.py`
  - Scope: reference selection stage before revalidation calls
3. Preserve manual override semantics:
  - If `--staleness-threshold` is provided, apply it as a global override
  - Otherwise use type-specific mapping
4. Align notebook checks:
  - File: `notebooks/academic_reference_staleness_check.ipynb`
  - Replace single threshold age rule with per-type threshold evaluation
5. Add tests:
  - Verify stale selection for `news/blog/online`
  - Verify CLI override precedence over per-type defaults

#### Operational Check Notebook

Use the maintenance notebook at `notebooks/academic_reference_staleness_check.ipynb` to:
- Inspect stale references from `rag_data/academic_references.db`
- Classify stale records via `expires_at`, age threshold, `link_status`, and `status=failed`
- Print ready-to-run re-validation commands

Recommended re-validation commands:

```bash
python scripts/ingest/ingest_academic.py --revalidate stale --staleness-threshold 30
python scripts/ingest/ingest_academic.py --revalidate failed
python scripts/ingest/ingest_academic.py --revalidate online
python scripts/ingest/ingest_academic.py --revalidate all
python scripts/ingest/ingest_academic.py --revalidate ids --ref-ids <space separated ref IDs>
```

Operational sequence:
1. Run the notebook and review stale categories.
2. Run the matching re-validation command.
3. Re-run the notebook to confirm stale counts have decreased.

### Retry Strategy

For each provider call:
- Retry up to 2 times on network failure
- Exponential backoff: 1s, 3s
- Fail-open: if all providers exhausted, store reference as `unresolved` with raw citation text
- Log all failures to audit trail

### Metadata Validation

Require at least **one of**:
- Valid DOI
- Title + author match (>85% similarity)
- Abstract present

If no match found, mark reference as `unresolved`.

---

## Download & Storage Policy

### Allowed Sources

Only download PDFs from:
- **Unpaywall**: verified OA URLs
- **arXiv**: direct links
- **Publisher OA mirrors**: identified via OpenAlex/Crossref

**Forbidden**:
- ResearchGate, Academia.edu (TOS violations)
- Google Scholar PDFs (indirect/unreliable)
- Paywalled URLs

### Download Constraints

```
Max PDF size: 50 MB
Max concurrent downloads: 3 (thread pool)
Timeout per download: 60 sec
Retry on 429/503: exponential backoff, max 2 retries
```

### Storage & Deduplication

- **Hash-based reuse**: compute SHA256(pdf_bytes) → file_id
- **Reuse rule**: if file_id already in storage, link to existing doc instead of re-storing
- **Storage path**: `rag_data/academic_pdfs/{file_id[:4]}/{file_id}.pdf`
- **Metadata**: store file_id + original URL in reference record

---

## Persona-Aware Retrieval Parameters

### Supervisor Persona

**Intent**: Guide student, ensure methodological coherence, validate argument structure  
**Focus**: pedagogical clarity, foundational knowledge, connections

**Retrieval Configuration**:
```python
{
  "persona": "supervisor",
  "chunk_size": 512,           # medium context
  "citation_depth": 1,          # direct refs only
  "include_competing_work": True,
  "rank_weights": {
    "recency": 0.1,             # prefer foundational
    "citation_count": 0.4,
    "abstract_relevance": 0.3,
    "domain_match": 0.2
  },
  "filter_rules": [
    "exclude: too_recent (< 2 yrs old without high citation count)",
    "include: foundational (>100 citations, domain-specific)",
    "boost: pedagogical papers, reviews, surveys"
  ],
  "max_results": 5
}
```

**TODO: Query enhancement**: 
- Suggest related terminology
- Offer background reading recommendations
- Flag methodology gaps

---

### Assessor Persona

**Intent**: Validate rigor, assess coverage, check for gaps, verify claims  
**Focus**: citation validation, reproducibility, completeness

**Retrieval Configuration**:
```python
{
  "persona": "assessor",
  "chunk_size": 256,            # detailed excerpts
  "citation_depth": 2,           # primary + secondary refs
  "include_competing_work": True,
  "rank_weights": {
    "recency": 0.15,
    "citation_count": 0.5,       # high citations → rigor signal
    "abstract_relevance": 0.2,
    "domain_match": 0.15,
    "corroboration": 0.2        # multi-source confirmation
  },
  "filter_rules": [
    "include: peer-reviewed only",
    "boost: contradictory/competing approaches",
    "flag: uncited or low-citation references"
  ],
  "max_results": 10,
  "include_citation_graph": True
}
```

**TODO: Query enhancement**:
- Show citation disagreements
- Highlight coverage gaps in literature
- Provide claim verification checklist
- Identify where contributing to new knowledge
- Identify potential plagiarism

---

### Researcher Persona

**Intent**: Identify novelty, find technical depth, discover reusable techniques  
**Focus**: cutting-edge methods, datasets, implementation details

**Retrieval Configuration**:
```python
{
  "persona": "researcher",
  "chunk_size": 1024,           # deep context
  "citation_depth": 3,           # full chain
  "include_competing_work": True,
  "rank_weights": {
    "recency": 0.4,             # favor recent
    "citation_count": 0.2,
    "abstract_relevance": 0.3,
    "domain_match": 0.1,
    "novelty_score": 0.3        # novel combos/techniques
  },
  "filter_rules": [
    "include: arxiv preprints (early discovery)",
    "boost: papers citing few others (novel directions)",
    "include: datasets, benchmarks, code references"
  ],
  "max_results": 15,
  "include_full_citation_chain": True,
  "include_methodology_details": True
}
```

**TODO: Query enhancement**:
- Suggest novel combinations of techniques
- Highlight emerging sub-domains
- Show methodological variations

---

## Rate Limiting & Resource Constraints

### API Rate Limits

| Provider | Limit | Window | Action |
|----------|-------|--------|--------|
| Crossref | 50 req/sec | per provider instance | throttle via provider `_rate_limit()` |
| OpenAlex | 10 req/sec | per provider instance | throttle via provider `_rate_limit()` |
| Semantic Scholar | 1 req/sec | per provider instance | throttle via provider `_rate_limit()` |
| Unpaywall | 1 req/sec | per provider instance | throttle via provider `_rate_limit()` |
| ORCID | 5 req/sec | per provider instance | throttle via provider `_rate_limit()` |
| DataCite | 5 req/sec | per provider instance | throttle via provider `_rate_limit()` |
| arXiv | 3 req/sec | per provider instance | throttle via provider `_rate_limit()` |
| PubMed | 3 req/sec (no API key), 10 req/sec (with API key) | per provider instance | throttle via provider `_rate_limit()` |
| Google Scholar | 1 req/sec | per provider instance | throttle via provider `_rate_limit()` |
| URL Fetch | 2 req/sec | per provider instance | throttle via provider `_rate_limit()` |

**Implementation (current runtime behaviour):**
- Each provider defines its own `rate_limit` value in its provider class.
- Every outbound HTTP call goes through `BaseProvider._request_with_retry()`.
- `_request_with_retry()` calls `BaseProvider._rate_limit()` before making the request.
- `_rate_limit()` enforces a minimum interval of `1.0 / rate_limit` seconds using `last_request_time`.
- Rate limiting is per provider instance (not a single global limiter shared across all providers).

Code references:
- `scripts/ingest/academic/providers/base.py`
- `scripts/ingest/academic/providers/*.py`

### Resource Constraints

- **Memory**: Keep reference cache + active downloads in memory; stream large PDFs to disk
- **Disk**: Max 5 GB for academic PDFs (configurable); cleanup old docs if exceeded
- **Threads**: 3 concurrent downloads; 4 workers for PDF text extraction
- **Time**: Timeout per document: 10 min; per reference: 30 sec

### Fallback Behaviour

If rate limit hit:
- Queue remaining references
- Pause 60 sec, retry
- Log to audit: `rate_limit_exceeded`

---

## Environment Configuration

### API Keys & Credentials

Academic ingestion requires credentials for metadata providers. Set in `.env`:

```bash
# === Required ===
UNPAYWALL_EMAIL=your-email@institution.edu
# Used for Unpaywall API access (identifies polite user-agent)

# === Strongly Recommended ===
CROSSREF_EMAIL=your-email@institution.edu
# Enables Crossref "polite pool" (higher rate limits)

# === Optional (Increases Limits) ===
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here
# Optional Semantic Scholar API key (default: free tier, 100 req/sec)

ORCID_CLIENT_ID=xxxx-xxxx-xxxx-xxxx
ORCID_CLIENT_SECRET=xxxxxxxxxxxxxxxx
# Optional ORCID OAuth credentials (default: public API, 24 req/sec)

# === Optional (Development/Testing) ===
ACADEMIC_INGEST_DRY_RUN=false
# Set to 'true' to test without fetching PDFs

ACADEMIC_INGEST_LOG_LEVEL=info
# Debug level: debug, info, warning, error

ACADEMIC_INGEST_MAX_PDF_SIZE_MB=50
# Max PDF size to download (default: 50 MB)

ACADEMIC_INGEST_CONCURRENT_DOWNLOADS=3
# Parallel PDF download threads (default: 3)

ACADEMIC_INGEST_CACHE_DIR=rag_data/academic_pdfs
# Directory for PDF caching (default: rag_data/academic_pdfs)
```

### Provider-Specific Configuration

**Crossref (Mandatory email)**
- Why: "Polite pool" requires identification
- Default: Uses `mailto:api@crossref.org` (restrictive limits)
- Recommended: Use institutional email for 50 req/sec tier
- Source: https://github.com/CrossRef/rest-api-doc#etiquette

**Unpaywall (Mandatory email)**
- Why: Required for API access
- Default: None (API calls will fail)
- Recommended: Use institutional email
- Source: https://unpaywall.org/products/api

**Semantic Scholar (Optional API key)**
- Why: Free tier is sufficient for most use cases
- Default: Public API (100 req/sec)
- Optional key: S2 API partner (higher limits available upon request)
- Source: https://www.semanticscholar.org/product/api

**ORCID (Optional OAuth)**
- Why: Public API sufficient for most use cases
- Default: Public API (24 req/sec)
- Optional credentials: For institutional members
- Source: https://orcid.org/organizations/integrators/API

### Configuration Loading

Config is loaded in this priority order (highest to lowest):

1. CLI arguments (--crossref-email, --unpaywall-email, etc.)
2. System environment variables
3. Environment variables (.env file) [only for vars not already set in system env]
4. Hardcoded defaults (if applicable)
5. Raise error if required values missing

Priority: CLI overrides > system env > .env (non-overriding) > defaults

### Example .env Setup

```bash
# Minimal (will work, but lower rate limits)
UNPAYWALL_EMAIL=your-email@organization.edu

# Recommended
UNPAYWALL_EMAIL=your-email@organization.edu
CROSSREF_EMAIL=your-email@organization.edu

# Full (highest rate limits)
UNPAYWALL_EMAIL=your-email@organization.edu
CROSSREF_EMAIL=your-email@organization.edu
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here
ORCID_CLIENT_ID=xxxx-xxxx-xxxx-xxxx
ORCID_CLIENT_SECRET=xxxxxxxxxxxxxxxx

# Development
ACADEMIC_INGEST_DRY_RUN=true
ACADEMIC_INGEST_LOG_LEVEL=debug
```

### Verification

Verify credentials are loaded correctly:

```bash
# Check that credentials are accessible
python -c "
from scripts.ingest.academic.config import load_config
config = load_config()
print(f'Crossref: {config.crossref_email}')
print(f'Unpaywall: {config.unpaywall_email}')
print(f'Semantic Scholar: {config.semantic_scholar_key}')
"
```

---

## Error Recovery & Audit

### Error Classification

| Level | Example | Action |
|-------|---------|--------|
| **Recoverable** | Network timeout, 429 rate limit | Retry with backoff |
| **Degraded** | No OA PDF found (metadata OK) | Continue; store metadata only |
| **Fatal** | All metadata providers failed | Mark reference as unresolved; log |

### Audit Trail

Log all operations to `academic_ingest.jsonl`:

```json
{
  "timestamp": "2026-02-01T12:34:56Z",
  "event": "reference_resolved",
  "doc_id": "acad_thesis_2024_001",
  "ref_id": "ref_crossref_10_1234_xyz",
  "provider": "crossref",
  "status": "success",
  "metadata": {
    "title": "...",
    "authors": [...],
    "doi": "10.1234/xyz"
  },
  "cached": false,
  "duration_ms": 245
}
```

### Summary Report

After ingestion batch:

```
================================================================================
Academic ingestion complete.
================================================================================

Domain Terminology Extraction:
  Total unique terms extracted: 1234
  Top 20 terms for 'machine_learning':
    1. ...

BM25 Keyword Indexing:
  Total corpus size: 10472
  Average chunk length: 287 tokens

Resource Monitor Summary:
  (CPU/RAM/IO summary printed by ResourceMonitor)

Total time: 742.18s
```

Current implementation notes:
- Per-document progress is logged (processing, reference storage counts, completion time per document).
- BM25 corpus statistics are reported when BM25 indexing is enabled.
- Domain terminology totals/top terms are printed at completion.
- Resource utilisation summary is printed and exported to JSON.

### TODO: Summary Enhancements

Provide a richer end-of-run summary, e.g. implement a final aggregate block that includes the following items:

1. **Batch totals**
  - Documents processed
  - Total citations extracted
  - Total references resolved, unresolved, and failed
2. **Provider breakdown**
  - Counts by provider (`crossref`, `openalex`, `semantic_scholar`, etc.)
  - Optional confidence bands (high/medium/low)
3. **Download outcomes**
  - PDFs downloaded successfully
  - Web artefacts downloaded successfully
  - Skipped and failed downloads
4. **Cache effectiveness**
  - Cache hits vs misses
  - Hit rate percentage
5. **Timing**
  - Total duration
  - Average per-document duration

Modules that would need updating:
- `scripts/ingest/ingest_academic.py`: accumulate counters during `stage_resolve_metadata`, `stage_download_references`, and per-document processing.
- `scripts/utils/logger.py` + audit events: emit a structured `academic_ingest_summary` event at completion.
- Optional: write the summary as JSON to `logs/` for dashboard ingestion.

Implementation sequence:
1. Add in-memory summary accumulator dataclass.
2. Update counters at each stage boundary.
3. Print final human-readable summary block.
4. Emit structured JSON/audit summary for downstream reporting.

---

## Reference Re-validation

### Overview

Re-validation allows updating existing references without re-ingesting primary documents. This is essential for:
- **Stale Content**: News articles, blogs updated or deleted
- **Link Rot**: URLs becoming invalid (404s)
- **Metadata Updates**: Citation counts, new DOIs, OA status changes
- **Failed Resolution Retry**: Re-attempt previously failed references

### Re-validation Modes

| Mode | Target References | Use Case |
|------|------------------|----------|
| `stale` | Online content exceeding staleness threshold | Update news/blogs >30 days old |
| `online` | All news/blog/online references | Force refresh all web content |
| `all` | All resolved references | Complete metadata refresh |
| `failed` | Previously unresolved/failed | Retry failed resolutions |
| `ids` | Specific ref_ids | Targeted update |

### Document Input Specification

When invoking the ingestion pipeline, users specify academic documents via CLI arguments:

**Positional arguments:**
```bash
python scripts/ingest/ingest_academic.py /path/to/paper.pdf [/path/to/another.pdf ...]
```

**Named arguments:**
```bash
python scripts/ingest/ingest_academic.py --papers /path/to/paper.pdf --papers /path/to/another.pdf
```

**Directory input:**
```bash
python scripts/ingest/ingest_academic.py --papers-dir ./thesis_references/
# Recursively ingest all .pdf files from directory
```

**Document metadata (optional):**
```bash
python scripts/ingest/ingest_academic.py \
  --papers thesis.pdf \
  --domain "machine_learning" \
  --topic "federated learning for healthcare" \
  --authors "John Doe, Jane Smith" \
  --institution "MIT"
```

**Batch mode (JSON file):**
```bash
# docs_manifest.json
{
  "documents": [
    {
      "path": "thesis.pdf",
      "domain": "machine_learning",
      "topic": "federated learning",
      "authors": ["John Doe"]
    },
    {
      "path": "related_work.pdf",
      "domain": "healthcare_informatics"
    }
  ]
}

python scripts/ingest/ingest_academic.py --batch docs_manifest.json
```

### CLI Usage

```bash
# Basic ingestion
python scripts/ingest/ingest_academic.py thesis.pdf

# Multiple documents
python scripts/ingest/ingest_academic.py thesis.pdf related_work.pdf

# With domain context
python scripts/ingest/ingest_academic.py thesis.pdf --domain machine_learning

# Batch ingestion
python scripts/ingest/ingest_academic.py --batch manifest.json

# Dry run (preview without fetching PDFs)
python scripts/ingest/ingest_academic.py thesis.pdf --dry-run

# Clear cache and start fresh
python scripts/ingest/ingest_academic.py thesis.pdf --cache-reset

# Refresh (re-resolve all references, bypass cache)
python scripts/ingest/ingest_academic.py thesis.pdf --refresh

# Update stale references (>30 days)
python scripts/ingest/ingest_academic.py --revalidate stale --staleness-threshold 30

# Update all online content
python scripts/ingest/ingest_academic.py --revalidate online

# Retry failed references
python scripts/ingest/ingest_academic.py --revalidate failed

# Update specific references
python scripts/ingest/ingest_academic.py --revalidate ids --ref-ids ref_abc123 ref_def456

# Force refresh all references
python scripts/ingest/ingest_academic.py --revalidate all

# New ingest reset with additional metadata
python scripts/ingest/ingest_academic.py --institution "University Name" --authors "John Doe, Jane Smith" --papers-dir /workspaces/governance-rag/data_raw/academic_papers --domain "security" --topic "RBAC for AI" --title "Title of Thesis"  --reset --purge-logs
```

### Re-validation Process

1. **Select References**: Query database based on mode criteria
2. **Re-resolve Metadata**: Query providers with cache bypass (`force_refresh=True`)
3. **Detect Changes**: Compare old vs new metadata (title, authors, DOI, citation count, etc.)
4. **Update Database**: Preserve `doc_ids` (citing documents), update metadata, refresh `accessed_at`
5. **Re-chunk Content**: If online content changed significantly (>10% text difference), delete old chunks and re-embed
6. **Audit Trail**: Log all updates, changes, and failures

### Preserved vs Updated Fields

**Preserved** (maintain existing values):
- `ref_id` — Keep same identifier
- `doc_ids` — Maintain citing document links
- `created_at` — Original creation timestamp

**Updated** (refresh from providers):
- `title`, `authors`, `year`, `abstract`, `venue` — Core metadata
- `doi`, `arxiv_id` — Identifiers (if newly assigned)
- `oa_available`, `oa_url`, `oa_status` — Open access status
- `citation_count` — Impact metrics
- `paywall_detected` — Accessibility status
- `quality_score` — Recalculated based on new metadata
- `accessed_at` — Current timestamp


### Stale Link Handling

When reference links become unavailable (404, timeout, moved):

**PRESERVE (keep existing data):**
- All chunks in ChromaDB — Historical value, already cited in documents
- Metadata — Title, authors, abstract remain accessible
- Citation graph edges — Citation relationship is historical fact
- Reference record — Update status fields only

**UPDATE (mark unavailability):**
- `link_status` → `stale_404`, `stale_timeout`, or `stale_moved`
- `link_checked_at` → Current timestamp
- `quality_score` → Apply penalty (-0.2 for 404, -0.15 for timeout, -0.05 for redirect)

**DO NOT delete:**
- Chunks would break citation context in primary documents
- Metadata still valuable for understanding cited claims
- Users may have local copies or alternative access methods

**Query filtering options:**
- Default: Include stale links with ranking penalty
- Assessor persona: Filter out stale links (require verifiable sources)
- Visualisation: Mark stale nodes with dashed borders and colour coding

**Archival modes:**
- `KEEP_ACTIVE` (default): Keep in main DB, mark as stale
- `ARCHIVE`: Move to archive table, exclude from default queries
- `DELETE` (not recommended): Remove all data

---

## TODO: Future Considerations

- **Legal/ethics**: Copyright/fair use constraints, cache retention limits, takedown process
- **Robots/ToS compliance**: Provider-specific compliance flags (especially scraping sources)
- **Security**: URL fetching sandbox, SSRF protection, content-type/size caps, malware scanning for PDFs
- **Data quality**: Canonical URL resolution, DOI normalisation, author disambiguation edge cases
- **Performance**: Provider backpressure, queueing, circuit-breakers, batch-level rate limits
- **Observability**: Per-provider latency/error dashboards, metadata drift monitoring, stale-link trends
- **Bias/coverage**: Regional coverage gaps, language handling, non-English sources, transliteration
- **User controls**: Source allow/deny lists, persona-specific source policies
- **Archival strategy**: Optional Wayback/LOCKSS integration for unstable URLs

---

## References & Dependencies

- **Implemented ingestion orchestration**:
  - `scripts/ingest/ingest_academic.py`: end-to-end academic ingestion pipeline
  - `scripts/ingest/academic/config.py`: academic ingest configuration and overrides
  - `scripts/ingest/academic/parser.py`: citation extraction and parsing utilities
  - `scripts/ingest/academic/downloader.py`: PDF/web artefact download handling

- **Implemented metadata resolution providers**:
  - `scripts/ingest/academic/providers/chain.py`: provider chain orchestration and confidence handling
  - `scripts/ingest/academic/providers/base.py`: base provider interface, retry, and per-provider rate limiting
  - `scripts/ingest/academic/providers/crossref.py`
  - `scripts/ingest/academic/providers/openalex.py`
  - `scripts/ingest/academic/providers/semantic_scholar.py`
  - `scripts/ingest/academic/providers/orcid.py`
  - `scripts/ingest/academic/providers/unpaywall.py`
  - `scripts/ingest/academic/providers/google_scholar.py`
  - `scripts/ingest/academic/providers/datacite.py`
  - `scripts/ingest/academic/providers/pubmed.py`
  - `scripts/ingest/academic/providers/arxiv.py`
  - `scripts/ingest/academic/providers/url_fetch.py`

- **Implemented storage, indexing, and graph components**:
  - `scripts/ingest/academic/cache.py`: SQLite reference cache (`rag_data/academic_references.db`)
  - `scripts/ingest/academic/citation_graph_writer.py`: citation graph persistence
  - `scripts/ingest/academic/citation_graph_schema.py`: citation graph schema
  - `scripts/ingest/academic/graph.py`: graph construction utilities
  - `scripts/ingest/vectors.py`: ChromaDB chunk/document storage
  - `scripts/ingest/bm25_indexing.py`: BM25 chunk-level indexing and corpus stats updates
  - `scripts/search/bm25_search.py`: BM25 retrieval/query support

- **Implemented supporting utilities**:
  - `scripts/ingest/chunk.py`: semantic and parent-child chunking
  - `scripts/ingest/pdfparser.py`, `scripts/ingest/htmlparser.py`: content extraction
  - `scripts/ingest/word_frequency.py`: word frequency extraction for visualisation
  - `scripts/ingest/academic/terminology.py`: terminology extraction and SQLite storage
  - `scripts/utils/db_factory.py`: cache/database client access
  - `scripts/utils/logger.py`: logging and audit integration
  - `scripts/utils/resource_monitor.py`: resource monitoring summary and JSON export

- **Operational notebooks/tools**:
  - `notebooks/academic_reference_staleness_check.ipynb`: stale reference checks and revalidation guidance

- **External APIs**:
  - Crossref REST API: https://api.crossref.org
  - OpenAlex API: https://api.openalex.org
  - Semantic Scholar API: https://api.semanticscholar.org
  - Unpaywall API: https://api.unpaywall.org
  - ORCID API: https://pub.orcid.org/
  - NCBI PubMed E-Utilities: https://www.ncbi.nlm.nih.gov/books/NBK25501/
  - DataCite REST API: https://api.datacite.org/
  - arXiv API: https://info.arxiv.org/help/api/index.html

- **Python libraries**:
  - `pydantic`: data models
  - `requests`: HTTP calls
  - `pypdf`: PDF text extraction
  - `beautifulsoup4`: HTML parsing
  - `python-dotenv`: environment loading
  - `chromadb`: vector storage backend
