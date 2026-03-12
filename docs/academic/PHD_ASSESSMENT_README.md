# PhD Quality Assessment 

---

## Overview

PhD Quality Assessment provides AI-driven analysis of PhD theses to identify potential issues and guide assessors to areas requiring attention. Initial release focuses on structural coherence, citation patterns, and basic red flag detection.

---

## Features Implemented

### ✅ PhD ingest

**File**: `scripts/ingest/ingest_academic.py`

- **Ingestion**
  - Set institution, authors, title, subject domain
  - Provide path to where PDF is located
  - Ingests document and validates citations
  - Following ingestion, run `scripts/consistency_graph/build_consistency_graph.py`
    - Builds semantic relationships between citations used by the PhD

### ✅ Core Assessment Engine

**File**: `scripts/ingest/academic/phd_assessor.py`

- **PhDQualityAssessor** class with embedding-based analysis
- **Structural Coherence Analysis**:
  - Chapter flow coherence using cosine similarity of embeddings
  - Missing required sections detection
  - Abrupt topic transitions (similarity < 0.3)
  - Average coherence scoring

- **Citation Pattern Analysis**:
  - Citation recency score (% from last 5 years)
  - Orphaned claims detection (sections with <30% citation density)
  - Citation clustering by topic
  - Geographic diversity placeholder (planned)

- **Red Flag Detection**:
  - Missing limitations section
  - Scope creep detection (conclusion > 1.5× results size)
  - Critical/Warning/Info severity levels

- **Persona-Based Scoring**:
  - Supervisor: Structure (60%), Citations (40%)
  - Assessor: Structure (50%), Citations (50%)
  - Researcher: Structure (40%), Citations (60%)

### ✅ Dashboard Integration

**File**: `scripts/ui/dashboard.py`

- **"Assessment" Tab** (🎓 icon)
- **Interactive Visualisation**:
  - Overall quality score with colour-coded display
  - Critical red flags with suggestions
  - Chapter flow coherence chart (Plotly line graph)
  - Structural and citation metrics
  - Recommended next steps

- **Tab Features**:
  - Auto-detection of PhD thesis documents
  - Colour-coded severity indicators:
    - 🟢 Green: Score ≥ 70%
    - 🟠 Orange: Score 50-69%
    - 🔴 Red: Score < 50%

### ✅ Persona Queries
**File**: `scripts/search/persona_retrieval.py`

Each persona applies different reranking strategies:

**Supervisor Persona** (Quality & Authority):
- Prioritise high-citation count documents
- Prefer established, peer-reviewed sources
- Emphasise foundational knowledge
- Smaller result set (k=8) focused on quality

**Researcher Persona** (Discovery & Breadth):
- Emphasise recent publications
- Prioritise novel methodologies
- Include diverse viewpoints
- Larger result set (k=15) for exploration

**Assessor Persona** (Balanced Coverage):
- Balance authority with breadth
- Include foundational + emerging works
- Comprehensive coverage
- Medium result set (k=10)

---

## Usage

### From Dashboard

1. Start dashboard: `python3 -m scripts.ui.dashboard`
2. Click "🎓 Assessment" tab
3. Assessment runs automatically on detected PhD thesis

### Programmatic Usage

```python
from scripts.ingest.academic.phd_assessor import PhDQualityAssessor
from scripts.utils.db_factory import get_vector_client

# Initialise
client = PersistentClient(path="rag_data/chromadb")
collection = client.get_collection("governance_docs_chunks")
assessor = PhDQualityAssessor(collection)

# Run assessment
report = assessor.assess_thesis(
    doc_id="Author_2026_PhD_Thesis",
    persona="supervisor"  # or 'assessor', 'researcher'
)

# Access results
print(f"Overall Score: {report.overall_score*100:.1f}%")
print(f"Summary: {report.summary}")

for flag in report.critical_red_flags:
    print(f"⚠️  {flag.title}: {flag.description}")

for step in report.next_steps:
    print(f"→ {step}")
```

---

## Data Structures

### AssessmentReport

```python
@dataclass
class AssessmentReport:
    doc_id: str
    assessed_at: datetime
    persona: str
    overall_score: float  # 0-1
    structure_analysis: StructureAnalysis
    citation_analysis: CitationPatternAnalysis
    critical_red_flags: List[RedFlag]
    summary: str
    next_steps: List[str]
```

### RedFlag

```python
@dataclass
class RedFlag:
    severity: str  # 'critical', 'warning', 'info'
    category: str  # 'structure', 'citations', 'scope'
    title: str
    description: str
    location: Optional[str]
    suggestion: Optional[str]
```
---

## How Each Persona Influences Results

### Example Query: "What is machine learning?"

**Supervisor Persona** (k=8, temp=0.2):
- Returns 8 high-authority references
- Emphasises foundational papers (Hinton, LeCun, etc.)
- Prioritises well-cited works
- Conservative, high-quality only
- Suitable for: Academic rigor, authority-based decisions

**Researcher Persona** (k=15, temp=0.5):
- Returns 15 diverse references
- Includes both classic and recent papers
- Prioritises novel approaches and emerging research
- Exploratory, discovery-focused
- Suitable for: Literature exploration, methodological innovation

**Assessor Persona** (k=10, temp=0.3):
- Returns 10 balanced references
- Mix of foundational and recent works
- Good coverage of the topic
- Moderate, well-rounded
- Suitable for: Comprehensive reviews, balanced assessment

Five academic templates include persona-specific parameters:

| Template | Persona | k | Temp | Focus |
|----------|---------|---|------|-------|
| Find high-quality references | supervisor | 8 | 0.2 | Quality & Authority |
| Find recent research | researcher | 15 | 0.5 | Discovery & Breadth |
| Literature review | assessor | 10 | 0.3 | Balance & Coverage |
| Find methodology | researcher | 15 | 0.5 | Methodological Innovation |
| Citation analysis | supervisor | 8 | 0.2 | Influential Works |

---

## Architecture Overview

```
┌─────────────────┐
│  Dashboard UI   │
│ [Persona ▼]     │
└────────┬────────┘
         │
    ┌────▼────────────────────┐
    │  Query Templates        │
    │  ├─ Template Name       │
    │  ├─ k_results: 8        │
    │  ├─ temperature: 0.2    │
    │  └─ persona: supervisor │
    └────┬────────────────────┘
         │ "apply_template()" callback
         │ returns all params including persona
         │
    ┌────▼───────────────────────┐
    │  run_rag_query()           │
    │  - reads rag-query-persona │
    │  - passes to rag_answer()  │
    └────┬───────────────────────┘
         │ persona parameter
         │
    ┌────▼──────────────────────┐
    │  generate.answer()        │
    │  - forwards to retrieve() │
    └────┬──────────────────────┘
         │ persona parameter
         │
    ┌────▼────────────────────────────┐
    │  retrieve()                     │
    │  - hybrid search (vector+BM25)  │
    │  - initial results: 2k chunks   │
    │  - if persona → rerank          │
    └────┬────────────────────────────┘
         │
    ┌────▼───────────────────────────┐
    │  apply_persona_reranking()     │
    │  - reorder by persona logic    │
    │  - supervisor: quality focus   │
    │  - researcher: diversity focus │
    │  - assessor: balanced focus    │
    └────┬───────────────────────────┘
         │
    ┌────▼──────────────┐
    │  Final Results    │
    │  (k chunks)       │
    │  Persona-tailored │
    └───────────────────┘
```

---

## Assessment Criteria

### Structure Analysis

| Metric | Threshold | Action |
|--------|-----------|--------|
| Chapter coherence | < 0.3 | Flag abrupt transition (warning) |
| Average coherence | < 0.5 | Flag low overall coherence (warning) |
| Missing sections | Any | Flag missing sections (critical) |

**Required Sections**:
- Introduction
- Literature Review
- Methodology
- Results
- Discussion
- Conclusion
- Limitations

### Citation Analysis

| Metric | Threshold | Action |
|--------|-----------|--------|
| Recency score | < 20% | Flag stale references (warning) |
| Citation density | < 30% | Flag orphaned claims (critical) |
| Topic concentration | > 50% | Flag echo chamber (info) |

### Red Flags

| Flag | Severity | Trigger |
|------|----------|---------|
| Missing limitations | Critical | No "limitation" in section titles |
| Scope creep | Warning | Conclusion > 1.5× Results size |
| Unsupported claims | Critical | Section citation density < 30% |

---

## Sample Output

```
📊 ASSESSMENT REPORT: Author_2026_PhD_Thesis
======================================================================

Overall Score: 75%
Persona: supervisor
Assessed: 2026-02-06 10:30:15

Summary:
  ✓ All required sections present | ✓ Strong chapter flow (0.82) | 
  ✓ Good citation recency (65% recent) | ✓ No critical issues

📚 STRUCTURE ANALYSIS
  Chapters: 8
  Average Coherence: 0.82
  Missing Sections: None
  Abrupt Transitions: 0
  Flow Scores: [0.85, 0.79, 0.88, 0.75, 0.81, 0.84, 0.80]

📖 CITATION ANALYSIS
  Total Citations: 287
  Unique Citations: 198
  Recency Score: 65%
  Orphaned Claims: 0

✅ RECOMMENDED NEXT STEPS
  1. [SUPERVISOR] Review chapter transitions for narrative coherence
  2. Consider diversifying citation sources (some topic concentration)
```

---

## Future

### TODO: Planned Features

- **LLM-Based Claim Extraction**: Identify assertions requiring citation support
  - LLM assessment of quotations against corresponding citation
- **Contradiction Detection**: Cross-reference claims for internal consistency
- **Methodological Rigor Checks**:
  - Sample size justification
  - Reproducibility indicators
  - Statistical reporting completeness
  - Duplicate text identification
- **Advanced Red Flags**:
  - Data-conclusion mismatch detection
  - Citation misrepresentation checking
- **Comparative Benchmarking**: Compare against successful PhD corpus by domain

---

## Technical Details

### Dependencies

- **numpy**: Embedding similarity computation
- **ChromaDB/SQLite**: Document chunk retrieval
- **Plotly**: Charts
- **Dash**: Dashboard integration

### Performance

- **Assessment Time**: ~2-5 seconds for 300-page thesis
- **Memory Usage**: ~100MB (embeddings loaded on-demand)
- **Thread-Safe**: Yes (for dashboard callbacks)

---

## Known Limitations

1. **Metadata Dependency**: Requires `chapter` or `section_title` metadata in chunks
2. **Citation Detection**: Heuristic-based (Future versions will use LLM extraction)
3. **Single Document**: Currently assesses one document at a time
4. **No Comparative Analysis**: Future feature

---

## Troubleshooting

### Issue: "No chunks found for doc_id"

**Cause**: Document not ingested or incorrect doc_id

**Verification**:
```bash
# List available documents
python3 -c "
from scripts.utils.db_factory import *
client = PersistentClient(path='rag_data/chromadb')
coll = client.get_collection('governance_docs_chunks')
docs = coll.get(include=['metadatas'])
doc_ids = set(m['doc_id'] for m in docs['metadatas'])
print('\n'.join(sorted(doc_ids)))
"
```

### Issue: "Low coherence score (0.00)"

**Cause**: Only 1 chapter detected or no embeddings available

**Solution**:
- Ensure document has proper chapter/section metadata
- Re-ingest with chapter detection enabled
- Check that embeddings are stored in ChromaDB

### Issue: "All sections reported as missing"

**Cause**: Section titles don't match required keywords

**Solution**:
- Update `required_sections` in phd_assessor.py
- Add fuzzy matching for section detection (future)
- Ensure metadata includes standardised section names

---

## Files

### Primary Files

- ✅ `scripts/ingest/ingest_academic.py` 
- ✅ `scripts/ingest/academic/phd_assessor.py` 
- ✅ `test_phd_assessment.py` 
- ✅ `PHD_ASSESSMENT_README.md` 

### Shared GUI

- ✅ `scripts/consistency_graph/dashboard.py`:
  - Assessment tab button 
  - Assessment tab content div 
  - switch_tabs callback (inputs/outputs)
  - populate_assessment_tab callback 

---

## Validation

✅ Test script runs successfully  
✅ PhD ingested and consistency graph built without errors
✅ Dashboard loads without errors  
✅ Assessment tab displays correctly  
✅ Sample report generated for PhD document  
✅ All red flags properly categorised  
✅ Colour-coded score display working  
✅ Chapter flow chart renders  

---

## Future Steps

1. **User Feedback**: Gather assessor input on red flag priority and overall usefulness of features
2. **Metadata Enhancement**: Improve chapter/section detection during ingestion
3. **LLM Extraction**: LLM-based claim extraction and consistency checking
4. **Dashboard UX**: Add persona selector, filter controls (partially implemented)
5. **Export Feature**: PDF report generation (partially implemented)

---

## Contact

For questions or feature requests, see project documentation or raise an issue.
