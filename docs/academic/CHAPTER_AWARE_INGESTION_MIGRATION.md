# Chapter-Aware Thesis Ingestion Migration

## Purpose

Make thesis structure authoritative throughout ingestion, retrieval, assessment,
and graph construction. A stored thesis chunk must retain its original source
span and structural identity; consumers must not reconstruct chapter membership
from chunk body text.

## Current State

The academic ingestion pipeline extracts PDF structure but stores the source
thesis as text-only parent and child chunks. Storage later tries to rediscover
each chunk's location with `full_text.find(...)`. This fails for repeated or
normalised text and leaves `chapter`, `section_title`, `heading_path`, and
`parent_section` empty. Parent and child chunks also have independent sequence
ranges, so they cannot be mixed as a single ordered structural view.

## Target Metadata Contract

All thesis child chunks must persist:

```python
{
    "thesis_id": "canonical thesis ID",
    "source_kind": "thesis_document",
    "chunk_type": "child",
    "sequence_number": 137,
    "source_start": 84231,
    "source_end": 86190,
    "chapter_id": "chapter-04",
    "chapter_number": 4,
    "chapter": "Chapter 4: Methodology",
    "section_id": "chapter-04.section-02",
    "section_title": "Data Collection",
    "heading_path": "Chapter 4: Methodology > Data Collection",
    "section_level": 2,
}
```

Parent chunks must retain their source span, `chapter_id`, and the first/last
child sequence numbers. A parent spanning multiple sections must persist the
section IDs it contains.

## Implementation Plan

1. Add source spans to parent and child chunk records in `scripts/ingest/chunk.py`.
2. Create chunks inside parsed structural spans for `source_kind=thesis_document`.
   Chunks must not cross a chapter boundary unless explicitly marked.
3. Map chunk spans to `extract_structure_from_text()` output before storage.
4. Persist supplied offsets and structure metadata unchanged in `vectors.py`.
   Remove `full_text.find(...)` and average-size position estimation from thesis
   storage paths.
5. Validate thesis ingestion before writing:
   - every child has valid source offsets;
   - chapter metadata coverage meets the configured threshold;
   - child source spans and sequence numbers are monotonic;
   - no unmarked child crosses a chapter boundary.
6. After metadata validation is implemented, regenerate Chroma and graph data from
   the source thesis PDFs using the normal ingestion/reset workflow. No parallel
   versioned collection, backward-compatibility path or historical chunk repair is
   required; preserve the source PDFs and user-authored profile/configuration files.
7. Make the assessment pipeline consume `chapter_id`, `chapter_number`, and
   source spans directly. Keep text inference only for legacy collections.
8. Add a separate thesis evidence graph with `thesis`, `chapter`, `section`,
   `chunk`, `research_question`, `claim`, `method`, `finding`, `conclusion`,
   `citation`, and `reference` nodes. Keep the citation and cross-document
   consistency graphs as separate products.
9. Use the thesis evidence graph for bounded query expansion after thesis-scoped
   retrieval, with external references included only by explicit query scope.

## Acceptance Criteria

- All newly ingested thesis child chunks have source spans.
- At least 95% of thesis child chunks have non-empty `chapter` and `heading_path`.
- Parent chunk metadata reflects its children rather than guessed text position.
- Chapter coherence and concept coverage follow ordered `chapter_id` values.
- Retrieval can filter by `thesis_id`, `chapter_id`, and `source_kind`.
- Integration tests assert persisted Chroma metadata for a representative thesis.
- Re-ingestion reports structural coverage and fails clearly when it is incomplete.

## Status

- [x] Root cause identified: source offsets are lost during parent/child chunking.
- [x] Legacy assessor fallback prevents invalid UI ordering.
- [x] Span-bearing chapter-aware chunk model.
- [x] Thesis-specific storage validation.
- [x] Persisted chapter-aware chunk metadata.
- [x] Re-ingested `O. Meyers - PhD thesis` after validation.
- [x] Standalone front and post matter scopes, including qualified appendices.
- [x] Persisted thesis evidence graph hierarchy (`thesis -> chapter -> section -> chunk`).
- [x] Bounded graph-guided query expansion from the highest-ranked thesis chunk.
- [x] Assessment entity graph (`research_question`, `claim`, `method`, `finding`, and `conclusion`).
- [x] Graph-backed Examiner Readiness Evidence with criterion-to-chunk provenance.
- [ ] Chapter extraction validated against the thesis Table of Contents. Partial: chapters
  are still derived from a heading rubric and ToC entries are discarded during structure
  extraction; the ToC is only used for ordering in the assessor. See
  `THESIS_GRAPH_PLAN.md` task F1.