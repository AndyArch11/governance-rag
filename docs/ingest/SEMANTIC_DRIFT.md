# Version Semantic Drift

This short guide explains what semantic drift is, how we detect it in the governance intelligence pipeline, and how to act on it.

## What Is Semantic Drift?
Semantic drift is the change in meaning or intent of a document across versions. It focuses on meaning, not formatting or minor copy edits. Drift matters because downstream consumers (services, policies, humans) may behave incorrectly if semantics shift without notice.

## Signals Used
- **Embedding distance**: Cosine distance between adjacent versions’ embeddings (higher = more drift).
- **Topic/cluster changes**: Movement between risk/topic clusters across versions.
- **Summary delta**: Change in generated summaries (ROUGE-L / token-diff heuristics).
- **Conflict delta**: Change in conflict edges/scores in the consistency graph.
- **Metadata shifts**: Source category, tags, owners, and linked systems changing between versions.

## Drift Classification (suggested defaults)
- **Low**: Distance < 0.15 and no cluster change.
- **Moderate**: 0.15–0.30 or minor cluster change.
- **High**: > 0.30 or major cluster/edge churn.
Thresholds should be tuned per corpus; start with the above and adjust after a few ingestions.

## Workflow in this Project
1) **Ingestion** captures versioned documents and computes embeddings. A change in documents between ingestions creates a new version. Versions relate to the ingestion process, not the lifecycle versioning of the doucment.
2) **Summaries** are generated; deltas between summaries flag semantic changes.
3) **Graph build** recomputes conflicts; conflict delta contributes to drift.
4) **Dashboard** (Timeline tab) surfaces version histories, drift scores, and cluster badges.

## Acting on Drift
- **Validate intent**: Confirm the new version’s intent with the doc owner.
- **Update dependents**: Notify downstream services/teams tied to the prior semantics.
- **Re-run tests**: For code/policy changes, re-run impacted suites or contract checks.
- **Pin versions**: For high-risk areas, pin consumers to a known-good version until reviewed.

## Practical Tips
- Treat repeated moderate drift as a trend—investigate root causes (ownership changes, unclear templates).
- Use cluster changes as a high-signal alert even if distance is modest.
- Compare conflict edges before/after; new conflicts often indicate breaking semantic shifts.
- Store and review drift summaries alongside ingest logs for audits.

## TODO: Extending Detection
- **Sentence-level alignment**: Align sentences between versions and score semantic shifts locally.
- **Section weighting**: Weight critical sections (e.g., SLAs, auth) higher in drift scoring.
- **Anomaly detection**: Model drift scores over time to flag outliers automatically.
