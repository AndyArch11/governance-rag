# Examiner Readiness Assessment Implementation Plan

## Purpose

Track delivery of evidence-based PhD assessment assistance derived from the
local `Guide for Examiners - PhD.pdf` reference document.

The feature supports human academic assessment. It must not generate, prepare,
or recommend a formal examiner report or examination outcome.

## Assessment Boundary

- [ ] Display that findings are screening evidence for human review.
- [ ] Display that results are not a Pass, Minor Amendments, Major Amendments,
  Deferred, or Fail recommendation.
- [ ] Distinguish missing evidence from evidence of a thesis deficiency.
- [ ] Label LLM-derived findings separately from deterministic findings.
- [ ] Confirm institutional approval before supporting a formal examination
  workflow, including the suitability of local-only AI processing.

## Existing Coverage

The current assessor already examines:

- Structural coherence, required sections, research-question alignment, and
  concept progression.
- Citation recency, citation coverage, and unsupported claims.
- Methodology evidence, limitations, writing quality, and claim consistency.
- Alignment between stated contributions and findings.

## Delivery Plan

### Phase 1: Readiness Data Model (Complete)

- [x] Add `ExaminerReadinessAnalysis` in
  `scripts/ingest/academic/phd_assessor.py`.
- [x] Define criterion statuses: `evidence_present`, `evidence_incomplete`,
  `needs_human_review`, and `not_applicable`.
- [x] Store criterion confidence, evidence snippets, source sections, and
  related `RedFlag` items.
- [x] Add the analysis to `AssessmentReport` without mapping it to an
  examination outcome.
- [x] Add unit tests for report construction and empty/not-applicable states.

Acceptance criteria:

- Every readiness criterion reports its status and supporting evidence.
- No public data model or UI field represents a formal outcome recommendation.

### Phase 2: Core Deterministic Criteria

- [x] Research significance: identify a meaningful research question and its
  disciplinary, professional, social, cultural, or policy context.
- [x] Contribution articulation: identify explicit statements of original and
  significant contribution, including nature and extent.
- [x] Conceptual integration: identify links between literature, conceptual or
  theoretical framework, objectives, and hypotheses.
- [x] Methodological rigour: identify justification, procedural detail,
  reproducibility evidence, and methodological or technical competence.
- [x] Findings quality: identify direct links to research questions, logical
  presentation, consideration of limitations, and treatment of surprising
  findings.
- [x] Contribution contextualisation: identify how findings support, extend, or
  contradict prior research and expose future research opportunities.

Acceptance criteria:

- Checks are deterministic by default and provide source evidence.
- A criterion does not emit a defect when relevant sections cannot be located.
- New flags use `needs_human_review` for ambiguous or low-confidence evidence.

### Phase 3: Conditional Thesis Forms (Complete)

- [x] Detect thesis-by-publication indicators.
- [x] For a detected publication-based thesis, check for an original
  introduction/literature review, framing chapter, bridging narrative, and
  independent general discussion.
- [x] Detect practice-based, creative-work, and exegesis indicators.
- [x] For a detected practice-based thesis, check for evidence that the exegesis
  integrates the practical or creative component.
- [x] Detect declared generative-AI use and check for tool, purpose, extent, and
  prompt/disclosure evidence.

Acceptance criteria:

- Conditional criteria report `not_applicable` unless reliable indicators are
  present.
- The tool does not attempt to assess creative work or verify confidential
  examination-process requirements.

### Phase 4: Optional LLM Evidence Extraction (Complete)

- [x] Add opt-in `llm_flags` for ambiguous readiness criteria.
- [x] Require structured findings with criterion, evidence, section, reason,
  and confidence.
- [x] Handle malformed responses and low-confidence output without producing
  definitive flags.
- [x] Retain deterministic checks as the baseline when LLM support is disabled.
- [x] Record the source of each finding as deterministic or LLM-derived.

Acceptance criteria:

- No LLM response can create an examination outcome.
- Every LLM finding has reviewable textual evidence and a confidence value.

### Phase 5: Score and Recommendation Policy (Complete)

- [x] Keep `overall_score` labelled as a readiness signal, not a grade.
- [x] Avoid translating scores or flags into amendment categories or outcomes.
- [x] Group findings by evidence state and human-review priority.
- [x] Ensure missing metadata, parse failures, and unavailable citations do not
  reduce readiness scores as if they were thesis defects.

Acceptance criteria:

- The UI and exportable report contain an explicit human-review disclaimer.
- Scores cannot be misread as formal examination recommendations.

### Phase 6: Source-Order Integrity (Complete)

- [x] Define a single canonical chapter ordering from the source document using
  sequence metadata, with table-of-contents order as a validated fallback.
- [x] Preserve that canonical ordering through extraction, assessment report
  construction, and dashboard rendering.
- [x] Correct the Chapter Flow chart so its nodes, transition labels, and data
  series follow source order rather than alphabetical, database, or graph order.
- [x] Apply the same ordering to all chapter-based assessment sections,
  including chapter summaries, coherence findings, research-question alignment,
  concept progression, and evidence lists.
- [x] Flag missing, duplicate, conflicting, or non-monotonic chapter sequence
  metadata for human review instead of silently reordering content.
- [x] Add an explicit source-order field to assessment data where the current
  chart/report contract cannot retain ordering implicitly.

Acceptance criteria:

- A thesis with Chapters 1, 2, 3, and 10 always displays in that source order.
- Pre-matter and post-matter are consistently excluded or included according to
  the individual assessment view's documented purpose.
- All chapter-based views use the same order for a given assessment run.
- Tests cover numbered chapters, unnumbered labelled sections, table-of-contents
  fallback, duplicate labels, malformed sequence metadata, and full document
  order from Abstract and Acknowledgements through References and Appendix.

### Phase 7: Dashboard Integration (Complete)

- [x] Add an Examiner Readiness panel in `scripts/ui/dashboard.py`.
- [x] Show criterion status, confidence, source sections, and evidence snippets.
- [x] Surface `needs_human_review` items before lower-priority observations.
- [x] Mark conditional criteria as `not_applicable` when not triggered.
- [x] Add the human-assessment boundary in the panel and any report export.

Acceptance criteria:

- A reviewer can inspect the evidence for every displayed finding.
- The assessment interface never offers examiner-report drafting or outcome
  controls.

### Phase 8: Semantic Graph 3D Visualisation

- [x] Add a user-selectable 3D visualisation mode for the semantic consistency
  graph in `scripts/ui/dashboard.py`.
- [x] Reuse graph nodes, edges, filters, clustering, and selection state from
  the existing 2D graph rather than creating a separate graph data model.
- [x] Use a stable 3D layout with deterministic seed support so a reviewer can
  compare views across reloads and filters.
- [x] Encode node type, cluster, conflict severity, and selected state without
  relying on colour alone.
- [x] Provide interactive rotation, zoom, pan, hover details, node selection,
  and a control to reset the camera orientation.
- [x] Preserve all existing filters and make selection in the 3D view update
  the document and assessment detail panels.
- [x] Provide a 2D fallback when WebGL is unavailable, disabled, or graph size
  exceeds a configurable rendering threshold.
- [x] Add performance instrumentation for layout, serialisation, and render
  preparation, including the existing timestamped performance log schema.

Acceptance criteria:

- A reviewer can inspect 3D graph relationships without losing current 2D
  graph features or selected-node context.
- The visualisation is nonblank, interactive, and legible on supported desktop
  and mobile viewports.
- Large graphs degrade predictably to 2D or require explicit user opt-in rather
  than freezing the dashboard.
- Tests cover layout determinism, filter propagation, selection propagation,
  WebGL fallback, and performance-threshold behaviour.

### Phase 9: Validation and Calibration (In Progress)

- [x] Add unit tests for every deterministic criterion.
- [x] Add tests for publication-based and practice-based applicability detection.
- [x] Add LLM contract tests for valid, malformed, and low-confidence JSON.
- [x] Add `AssessmentReport` and dashboard rendering regression tests.
- [ ] Create an anonymised fixture set for false-positive and
  false-negative calibration.
- [ ] Obtain subject-matter review before enabling new criteria by default.

Human calibration and institutional sign-off are governed by
`docs/academic/EXAMINER_READINESS_CALIBRATION_PROTOCOL.md`.

Acceptance criteria:

- Tests cover positive, negative, ambiguous, and not-applicable inputs.
- Human review confirms the new findings are useful and not presented as grades.

## Proposed Implementation Order

1. Phase 1: data model and assessment boundary.
2. Phase 2: core deterministic checks.
3. Phase 6: source-order integrity for current assessment views.
4. Phase 7: dashboard evidence display for the completed checks.
5. Phase 8: semantic graph 3D visualisation.
6. Phase 3: conditional thesis-form checks.
7. Phase 4: optional LLM evidence extraction.
8. Phase 5 and Phase 9: final policy review, calibration, and default enablement.

## Source Guidance

The plan is based on the local examiner guide, especially its criteria for:

- Significant research questions and articulated objectives/hypotheses.
- Critical literature and conceptual-framework integration.
- Original, significant contribution to knowledge.
- Appropriate, justified, and clearly described methodology.
- Findings connected to research questions and interpreted with limitations.
- Contextualisation of findings against prior research and future opportunities.
- Cohesion requirements for publication-based theses.
- Integration requirements for creative/practice-based work and exegesis.
- Disclosure of generative-AI tool use in thesis preparation.
