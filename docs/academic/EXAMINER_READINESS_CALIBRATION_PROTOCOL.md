# Examiner Readiness Calibration Protocol

## Purpose

This protocol supports human calibration of the examiner-readiness assessment.
It does not authorise use of the application for preparing an examiner report or
making an examination outcome recommendation.

## Confidentiality

Use anonymised or institution-approved material only. Do not commit theses,
examiner reports, confidential correspondence, identifiable candidate data, or
unapproved extracted text to this repository.

## Calibration Corpus

The synthetic baseline is:

`tests/fixtures/examiner_readiness_calibration.json`

It covers conventional, publication-based, and practice-based thesis forms.
Synthetic scenarios protect regression behaviour but cannot establish academic
validity or acceptable false-positive rates.

## Human Review Procedure

1. Select an anonymised, institution-approved set of representative thesis
   extracts, including positive, negative, ambiguous, and not-applicable cases.
2. Have at least two qualified academic reviewers independently record the
   expected state for each readiness criterion:
   `evidence_present`, `evidence_incomplete`, `needs_human_review`, or
   `not_applicable`.
3. Compare reviewer expectations with deterministic assessment output.
4. Record false positives, false negatives, ambiguous outcomes, and extraction
   or metadata limitations separately from thesis-quality observations.
5. Review marker language and evidence snippets for discipline-specific bias or
   misleading inference.
6. Update synthetic fixtures and regression tests only after the reviewers agree
   on an expected state and rationale.

## Acceptance Record

For each reviewed scenario, retain an approved record outside this repository
containing:

- Scenario identifier and approved data-handling classification.
- Reviewer roles and review date.
- Expected and observed readiness state for each criterion.
- False-positive and false-negative findings.
- Agreed marker or threshold changes.
- Confirmation that the output is human-assessment support, not a grade or
  examination recommendation.

## Institutional Gate

Before enabling new criteria by default for any formal assessment workflow,
obtain approval from the relevant institution or Graduate Research School for:

- The intended use of local-only AI processing.
- Confidentiality and data-retention controls.
- The qualification and responsibility of human reviewers.
- The application's non-grading and non-report-authoring boundary.
