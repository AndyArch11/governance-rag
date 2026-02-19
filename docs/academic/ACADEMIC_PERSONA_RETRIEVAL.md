# Academic Persona Retrieval Logic

---

## Overview

This document defines **persona-aware retrieval** for three user types:

- **Supervisor** (foundational, pedagogical)
- **Assessor** (verification, rigor, evidence)
- **Researcher** (novelty, technical depth)

Persona logic is applied at **query time** (not ingestion time) to avoid biasing storage. It uses:

- Reference type and quality score
- Stale link status
- Citation graph depth
- Domain relevance score
- Recency (publication year)

---

## Persona Profiles

### 1. Supervisor (Foundational)

**Goal**: Provide reliable, pedagogical context and core references.

- Emphasise **high-quality, well-cited** references
- Prefer **review articles, textbooks, survey papers**
- Allow older references if they are foundational
- Include stale links (historical value)

**Default Settings**:

```python
SupervisorConfig = {
    "persona": "supervisor",
    "reference_depth": 2,  # direct + first-hop citations
    "min_quality_score": 0.6,
    "min_citation_count": 10,
    "prefer_reference_types": ["academic", "report", "preprint"],
    "include_stale_links": True,
    "stale_link_penalty": 0.2,
    "recency_bias": 0.1,  # slight preference for newer work
}
```

---

### 2. Assessor (Verification)

**Goal**: Ensure claims are verifiable and evidence-based.

- Require **verifiable sources** (no stale links)
- Prefer peer‑reviewed, OA sources
- Penalise news/blog references heavily
- Emphasise citation graph depth (supporting evidence)
TODO:
- Identify **potential plagiarism** (text overlap + citation mismatch)
- Highlight **new knowledge contributions** (novelty vs. prior art)

**Default Settings**:

```python
AssessorConfig = {
    "persona": "assessor",
    "reference_depth": 3,  # include supporting chains
    "min_quality_score": 0.7,
    "min_citation_count": 5,
    "prefer_reference_types": ["academic", "report"],
    "include_stale_links": False,
    "require_verifiable": True,
    "stale_link_penalty": 0.5,
    "recency_bias": 0.0,  # neutral
    "enable_plagiarism_checks": True,
    "enable_novelty_analysis": True,
}
```

---

### 3. Researcher (Novelty)

**Goal**: Surface cutting‑edge work and emerging methods.

- Favor recent publications (last 3–5 years)
- Include preprints and arXiv sources
- Allow some lower‑quality references if novel
- Balance depth and novelty

**Default Settings**:

```python
ResearcherConfig = {
    "persona": "researcher",
    "reference_depth": 1,  # direct refs first
    "min_quality_score": 0.4,
    "min_citation_count": 0,
    "prefer_reference_types": ["preprint", "academic", "report"],
    "include_stale_links": True,
    "stale_link_penalty": 0.3,
    "recency_bias": 0.4,  # strong
}
```

---

## Retrieval Ranking Strategy

### Base Score Components

Each candidate chunk/reference is scored with:

| Component | Weight | Description |
|-----------|--------|-------------|
| Semantic similarity | 0.50 | Embedding similarity to query |
| Quality score | 0.20 | Reference quality (0.0–1.0) |
| Domain relevance | 0.15 | Domain match score |
| Citation count | 0.10 | Normalised citation impact |
| Recency | 0.05 | Publication year proximity |

### Persona Adjustments

- **Supervisor**: Increase quality score weight, allow older references
- **Assessor**: Increase domain relevance + citation count weight, filter stale links
- **Researcher**: Increase recency weight, allow preprints

### Example Scoring

$$\text{score} = 0.5 s + 0.2 q + 0.15 d + 0.1 c + 0.05 r$$

Where:
- $s$ = semantic similarity
- $q$ = quality score
- $d$ = domain relevance
- $c$ = citation impact
- $r$ = recency (scaled 0–1)

---

## Citation Depth Selection

Persona determines how far to traverse citation graph:

| Persona | Depth | Behaviour |
|---------|-------|----------|
| Supervisor | 2 | Include direct and first-hop references |
| Assessor | 3 | Include supporting evidence chains |
| Researcher | 1 | Focus on direct references first |

---
