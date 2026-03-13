# Project Roadmap (Merged with TODO Backlog)

This roadmap merges the previous strategic roadmap with all actionable in-repo TODO items (code, tests, and docs) and re-groups them by topic.

## Scope and method

- Scope scanned: `scripts/**`, `tests/**`, `docs/**`, root `*.md`.
- Included: actionable TODO/FIXME/TBD items with implementation intent.
- Excluded: generated, vendored, minified, and downloaded/export artefacts (for example `rag_data/**`, `data_raw/**` exports, `htmlcov/**`, vendored `vis` assets), where TODO strings are not project-maintained backlog.
- Effort scale: `S` (small), `M`, `L`, `XL`.
- Impact scale: `Low`, `Medium`, `High`, `Very High`.

---

## Priority roadmap (updated)

## P0 — Immediate (high impact, low-medium effort)

### 1) Ingestion robustness and correctness
- **Move late imports to module scope** for maintainability and static analysis hygiene.
  - Effort: `M` | Impact: `High`
  - Refs: `scripts/ingest/ingest_git.py` (lines 684, 1212, 1672, 1721, 1728, 1738, 1750, 1782, 1855, 2191), `scripts/ingest/ingest.py` (2099), `scripts/ingest/ingest_academic.py` (725)

### 2) Embedding and chunking quality controls
- **Tune chunk sizing/truncation** and reduce debug-log noise in vector ingestion.
  - Effort: `M` | Impact: `High`
  - Refs: `scripts/ingest/vectors.py` (1395, 1603, 1644, 1645)
- **Add table size guardrails** to avoid oversized table chunks reaching LLM paths.
  - Effort: `S` | Impact: `High`
  - Refs: `scripts/ingest/vectors.py` (616)
- **Validate semantic drift schema compliance** and clarify doc embedding metadata handling.
  - Effort: `M` | Impact: `Medium`
  - Refs: `scripts/ingest/vectors.py` (1804, 2220)
- **Improve table-aware chunking strategy** (current approach acknowledged as weak).
  - Effort: `L` | Impact: `High`
  - Refs: `scripts/ingest/chunk.py` (283, 386)

### 3) Observability and monitoring baseline
- **Add OpenTelemetry/OTLP tracing/log enrichment** in ingestion and graph builder.
  - Effort: `M` | Impact: `Very High`
  - Refs: `scripts/ingest/ingest.py` (TODO list line 43 block), `scripts/consistency_graph/build_consistency_graph.py` (104)
- **Fix and extend resource monitor signals** (Chroma detection, trend recording, alert integration, GPU metrics, file logging, purge).
  - Effort: `M` | Impact: `High`
  - Refs: `scripts/utils/resource_monitor.py` (15-21)
- **Embed timestamps and richer perf diagnostics in dashboard workflows**.
  - Effort: `S` | Impact: `Medium`
  - Refs: `scripts/ui/dashboard.py` (TODO block at line 21)

### 4) Testing debt that blocks safe refactors
- **Add missing ingestion orchestration tests** (CLI entrypoint, concurrency, recovery paths).
  - Effort: `M` | Impact: `High`
  - Refs: `tests/test_ingest.py` (TODO block at line 6)
- **Retire/replace transitional tests and fixtures**.
  - Effort: `S` | Impact: `Medium`
  - Refs: `tests/test_rag_logger.py` (6), `tests/test_bitbucket_code_ingestion.py` (3), `tests/test_code_parser_camel.py` (3)

## P1 — Near-term (high impact, medium-large effort)

### 5) PDF/HTML parsing uplift (core ingestion quality)
- **Evaluate and potentially switch PDF extraction stack** (grobid/nougat/pdfplumber/texxtract/pdf2txt).
  - Effort: `L` | Impact: `Very High`
  - Refs: `scripts/ingest/pdfparser.py` (11)
- **Add structure-aware PDF extraction and metadata enrichment** (sections, citations, metadata, layouts, tables, OCR, multilingual, quality/access/versioning, malware pre-scan).
  - Effort: `XL` | Impact: `Very High`
  - Refs: `scripts/ingest/pdfparser.py` (27-40)
- **Remove sequencing workaround once chunk ordering is corrected**.
  - Effort: `M` | Impact: `Medium`
  - Refs: `scripts/ingest/pdfparser.py` (371)
- **Improve HTML table understanding** (purpose detection, multi-row headers, nested table recursion, richer rowspan handling).
  - Effort: `L` | Impact: `High`
  - Refs: `scripts/ingest/htmlparser.py` (91, 94, 182, 234)

### 6) Academic ingestion and citation intelligence
- **Improve reference extraction quality** (newspaper3k fallback strategy, staleness thresholds by reference type).
  - Effort: `M` | Impact: `High`
  - Refs: `scripts/ingest/academic/providers/url_fetch.py` (10), `scripts/ingest/academic/providers/base.py` (6), `docs/academic/ACADEMIC_INGESTION_POLICY.md` (167, 207)
- **Capture missing citation metadata** (venue rank, citation count, OA availability).
  - Effort: `M` | Impact: `High`
  - Refs: `scripts/ingest/academic/graph.py` (76, 90), `scripts/ui/academic/citation_graph_viz.py` (429)
- **Scale citation graph UI and add exports** (subgraph loading, pagination/lazy load, caching, default handling, visual polish, export CSV/JSON).
  - Effort: `L` | Impact: `High`
  - Refs: `scripts/ui/academic/citation_graph_viz.py` (24-26, 235-238)
- **Advance PhD assessor heuristics and placeholders** (benchmarking corpus, affiliation diversity, sentence splitting, section labelling).
  - Effort: `L` | Impact: `High`
  - Refs: `scripts/ingest/academic/phd_assessor.py` (1122, 1501, 2191, 2304, 2701)
- **Improve academic terminology extraction quality** (better than sliding-window n-grams).
  - Effort: `M` | Impact: `Medium`
  - Refs: `scripts/ingest/academic/terminology.py` (26), `scripts/ui/academic/academic_references.py` (85)
- **Stabilise academic references UI implementation quality** (remove debug prints, strengthen DB error handling, add query caching).
  - Effort: `M` | Impact: `Medium`
  - Refs: `scripts/ui/academic/academic_references.py` (14-15)

### 7) Retrieval and ranking improvements
- **Unify query expansion modules across search and RAG** and externalise term configuration.
  - Effort: `M` | Impact: `High`
  - Refs: `scripts/search/query_expansion.py` (9), `scripts/rag/query_expansion.py` (9-10), `scripts/rag/domain_terms.py` (41)
- **Improve token preprocessing quality** (remove repeated-word artefacts in n-grams).
  - Effort: `S` | Impact: `Medium`
  - Refs: `scripts/search/text_preprocessing.py` (120)
- **Improve reranker capability** (evaluate stronger model, support additional cross-encoder backends).
  - Effort: `M` | Impact: `High`
  - Refs: `scripts/search/reranker.py` (92-93)
- **Fix tokenisation and cache telemetry accuracy in generation path**.
  - Effort: `M` | Impact: `Medium`
  - Refs: `scripts/rag/generate.py` (521, 605)
- **Address fallback literal/model-name testing fragility in retrieval**.
  - Effort: `S` | Impact: `Medium`
  - Refs: `scripts/rag/retrieve.py` (88)
- **Advance prompt assembly strategy** (configurable prompts, prompt optimisation, reranking before assembly, better code-snippet detection).
  - Effort: `M` | Impact: `High`
  - Refs: `scripts/rag/assemble.py` (23, 537)

### 8) Dashboard UX and reliability fixes
- **Resolve known node-selection/document-detail mismatch in academic graph flows**.
  - Effort: `M` | Impact: `High`
  - Refs: `scripts/ui/dashboard.py` (TODO block at line 21)
- **Add practical UX controls** (tooltips, busy states, disabled empty tabs, ingest/config UI).
  - Effort: `M` | Impact: `Medium`
  - Refs: `scripts/ui/dashboard.py` (TODO block at line 21)
- **Explore optional performance improvements** (WebGL auto-enable and Redis caching).
  - Effort: `M` | Impact: `Medium`
  - Refs: `scripts/ui/dashboard.py` (TODO block at line 21), `scripts/ui/dashboard.py` (605, 9415)

## P2 — Strategic (large scope, architecture-level)

### 9) Code intelligence and architecture mining
- **Expand parser from regex-heavy extraction toward richer service graph semantics** (imports/calls resolution, endpoint detail, config semantics, comments/docs semantics, confidence scoring).
  - Effort: `XL` | Impact: `Very High`
  - Refs: `scripts/ingest/git/code_parser.py` (15-22, 411-412), `docs/git/CODE_PARSER.md` (218)
- **Support more ecosystems/frameworks for repository ingestion**.
  - Effort: `L` | Impact: `High`
  - Refs: `scripts/ingest/git/code_parser.py` (19)

### 10) Preprocessing architecture modernisation
- **Refactor preprocess module into cleaner components** and improve stage-level observability.
  - Effort: `L` | Impact: `High`
  - Refs: `scripts/ingest/preprocess.py` (16-17)
- **Enable async/multi-model/multi-provider preprocessing strategy**.
  - Effort: `L` | Impact: `High`
  - Refs: `scripts/ingest/preprocess.py` (18-20)
- **Improve cleaning quality policy** (chunked cleaning, better prompts, two-pass strategy, optional anonymisation policy).
  - Effort: `M` | Impact: `High`
  - Refs: `scripts/ingest/preprocess.py` (428-431, 641)

### 11) Data model and schema hardening
- **Tighten schema contracts** (enums, limits, validation rules).
  - Effort: `M` | Impact: `High`
  - Refs: `scripts/utils/schemas.py` (30, 58)
- **Clarify Chroma wrapper return contracts** (add/update/delete result semantics).
  - Effort: `M` | Impact: `Medium`
  - Refs: `scripts/ingest/chromadb_sqlite.py` (109, 344, 396)
- **Improve word-frequency/indexing strategy** (token policy, multilingual support, corpus aggregation, sharding/doc-id tracking).
  - Effort: `M` | Impact: `Medium`
  - Refs: `scripts/ingest/word_frequency.py` (80-83, 100), `scripts/ingest/cache_db.py` (271-272), `docs/academic/WORD_FREQUENCY_IMPLEMENTATION.md` (193), `docs/academic/WORD_FREQUENCY_QUICK_START.md` (175)

### 12) Security and governance maturity
- **Tune DLP masking for real ingestion workloads** to reduce false positives/negatives.
  - Effort: `M` | Impact: `High`
  - Refs: `scripts/security/dlp.py` (21)
- **Add consistency-graph embedding cache and safer default classification behaviour**.
  - Effort: `M` | Impact: `High`
  - Refs: `scripts/consistency_graph/build_consistency_graph.py` (105, 886)
- **Improve CLI ergonomics for graph builder options and flags**.
  - Effort: `S` | Impact: `Medium`
  - Refs: `scripts/consistency_graph/build_consistency_graph.py` (2540-2541)
- **Upgrade community detection approach** (Louvain support).
  - Effort: `S` | Impact: `Medium`
  - Refs: `scripts/consistency_graph/advanced_analytics.py` (100)

---

## Previous roadmap initiatives (merged and re-scored)

The following existing roadmap items remain valid and are now mapped to updated priorities above:

- **Multi-model support** → covered in Topic 10 (`P2`).
- **Semantic reranking** → covered in Topic 7 (`P1`).
- **Support more source targets (GitHub, Confluence, etc.)** → covered in Topics 1 and 9 (`P0/P2`).
- **Parse more document types (PDF, DOCX, ODT, MD, etc.)** → covered in Topics 5 and 1 (`P1/P0`).
- **Fix logging and observability quality** → covered in Topics 3 and 10 (`P0/P2`).
- **MCP server support, multi-tenant, SSO/RBAC, zero trust, federated search, personalisation** → still strategic epics (`P2`) pending concrete implementation TODOs in code.
- **Comprehensive testing expansion** → covered in Topic 4 (`P0`) and Topic 11 (`P2`).
- **Graph analytics, anomaly detection, agent-based search, pgvector migration** → retained as strategic platform initiatives (`P2`), with implementation sequencing after ingestion/retrieval quality stabilises.

---

## Documentation backlog grouped by topic

These TODO blocks are mostly design/operational enhancements and should be executed alongside the technical topics above:

- **Retrieval and assistant docs**: `docs/search/RAG_QUERY_ASSISTANT.md` (363)
- **Retry and resilience docs**: `docs/ingest/RETRY_IMPLEMENTATION.md` (221)
- **Dry-run and profiling docs**: `docs/ingest/DRY_RUN_PREVIEW.md` (407), `docs/ingest/PROFILE_MODE.md` (240)
- **Adaptive chunking and cleaning docs**: `docs/ingest/ADAPTIVE_CHUNKING.md` (279), `docs/ingest/HTML_PDF_CLEANING.md` (137)
- **Semantic drift docs**: `docs/ingest/SEMANTIC_DRIFT.md` (39)
- **Logger consolidation docs**: `docs/ingest/LOGGER_CONSOLIDATION.md` (206)
- **Code parser docs**: `docs/git/CODE_PARSER.md` (218)
- **Academic docs**: `docs/academic/ACADEMIC_PERSONA_RETRIEVAL.md` (59), `docs/academic/DOMAIN_TERMINOLOGY_IMPLEMENTATION.md` (145), `docs/academic/PHD_ASSESSMENT_README.md` (330), `docs/academic/ACADEMIC_INGESTION_POLICY.md` (167, 207, 338, 374, 413, 643, 850), `docs/academic/WORD_FREQUENCY_IMPLEMENTATION.md` (193), `docs/academic/WORD_FREQUENCY_QUICK_START.md` (175)

---

## Suggested execution order (next 3 iterations)

1. **Iteration 1 (stability first)**: Topics 1, 2, 3, 4
2. **Iteration 2 (quality uplift)**: Topics 5, 6, 7, 8
3. **Iteration 3 (platform maturity)**: Topics 9, 10, 11, 12
