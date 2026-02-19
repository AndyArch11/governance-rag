"""Persona-aware retrieval utilities for academic references.

Applies persona-specific filtering and reranking rules to retrieved chunks
based on metadata such as reference_type, quality_score, citation_count,
link_status, and publication year.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class PersonaConfig:
    """Configuration for persona-aware retrieval."""

    persona: str
    reference_depth: int
    min_quality_score: float
    min_citation_count: int
    prefer_reference_types: Tuple[str, ...]
    include_stale_links: bool
    require_verifiable: bool
    stale_link_penalty: float
    recency_bias: float


SUPERVISOR_CONFIG = PersonaConfig(
    persona="supervisor",
    reference_depth=2,
    min_quality_score=0.6,
    min_citation_count=10,
    prefer_reference_types=("academic", "report", "preprint"),
    include_stale_links=True,
    require_verifiable=False,
    stale_link_penalty=0.2,
    recency_bias=0.1,
)

ASSESSOR_CONFIG = PersonaConfig(
    persona="assessor",
    reference_depth=3,
    min_quality_score=0.7,
    min_citation_count=5,
    prefer_reference_types=("academic", "report"),
    include_stale_links=False,
    require_verifiable=True,
    stale_link_penalty=0.5,
    recency_bias=0.0,
)

RESEARCHER_CONFIG = PersonaConfig(
    persona="researcher",
    reference_depth=1,
    min_quality_score=0.4,
    min_citation_count=0,
    prefer_reference_types=("preprint", "academic", "report"),
    include_stale_links=True,
    require_verifiable=False,
    stale_link_penalty=0.3,
    recency_bias=0.4,
)


def get_persona_config(persona: str) -> PersonaConfig:
    """Return persona config by name."""
    persona_key = (persona or "").strip().lower()
    if persona_key == "supervisor":
        return SUPERVISOR_CONFIG
    if persona_key == "assessor":
        return ASSESSOR_CONFIG
    if persona_key == "researcher":
        return RESEARCHER_CONFIG
    raise ValueError(f"Unknown persona: {persona}")


def apply_persona_reranking(
    chunks: List[str],
    metadata: List[Dict[str, Any]],
    persona: str,
    top_k: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Filter and rerank chunks based on persona rules.

    Args:
        chunks: Retrieved text chunks
        metadata: Metadata for each chunk
        persona: Persona name (supervisor/assessor/researcher)
        top_k: Result size after reranking

    Returns:
        Filtered and reranked (chunks, metadata)
    """
    if not chunks or not metadata:
        return chunks, metadata

    config = get_persona_config(persona)
    current_year = datetime.now().year

    # Normalise TF scores if available
    tf_scores = [
        m.get("tf_score") for m in metadata if isinstance(m, dict) and m.get("tf_score") is not None
    ]
    tf_max = max(tf_scores) if tf_scores else 1.0

    scored_items: List[Tuple[float, str, Dict[str, Any]]] = []

    for chunk, meta in zip(chunks, metadata):
        if not isinstance(meta, dict):
            meta = {}

        # Allow academic_reference chunks even without persona metadata
        # These are ingested academic papers/thesis documents that don't have
        # reference_type/quality_score/citation_count fields from the reference
        # extraction pipeline. They should still be retrievable by all personas.
        source_category = (meta.get("source_category") or "").lower()
        is_academic_content = source_category == "academic_reference"

        reference_type = (meta.get("reference_type") or "").lower()
        link_status = (meta.get("link_status") or "available").lower()
        quality_score = float(meta.get("quality_score") or 0.0)
        citation_count = int(meta.get("citation_count") or 0)
        domain_relevance = float(
            meta.get("domain_relevance_score") or meta.get("relevance_score") or 0.0
        )

        # Filter: reference type
        # SKIP this filter for academic_reference content (full papers vs extracted citations)
        if config.prefer_reference_types and reference_type and not is_academic_content:
            if reference_type not in config.prefer_reference_types:
                continue

        # Filter: verifiability / stale links
        # SKIP for academic_reference (ingested papers without link metadata)
        if not is_academic_content:
            if config.require_verifiable and link_status != "available":
                continue
            if not config.include_stale_links and link_status != "available":
                continue

        # Filter: min thresholds
        # SKIP for academic_reference (papers without citation extraction metadata)
        if not is_academic_content:
            if quality_score < config.min_quality_score:
                continue
            if citation_count < config.min_citation_count:
                continue

        # Base similarity score
        similarity_score = 0.0
        if "distance" in meta:
            # Chroma distance: lower is better, map to similarity
            try:
                similarity_score = max(0.0, 1.0 - float(meta.get("distance")))
            except (TypeError, ValueError):
                similarity_score = 0.0
        elif "tf_score" in meta:
            try:
                similarity_score = min(1.0, float(meta.get("tf_score")) / tf_max)
            except (TypeError, ValueError):
                similarity_score = 0.0

        # Normalise citation count to [0,1]
        citation_score = min(1.0, citation_count / 100.0) if citation_count else 0.0

        # Recency score
        year = meta.get("year") or meta.get("publication_year")
        try:
            year = int(year) if year else None
        except (TypeError, ValueError):
            year = None
        if year:
            years_old = max(0, current_year - year)
            recency_score = max(0.0, 1.0 - (years_old / 10.0))
        else:
            recency_score = 0.0

        # Base weighted score
        score = (
            0.5 * similarity_score
            + 0.2 * quality_score
            + 0.15 * domain_relevance
            + 0.1 * citation_score
            + 0.05 * recency_score
        )

        # Persona recency adjustment
        if config.recency_bias:
            score += config.recency_bias * recency_score

        # Stale link penalty
        if link_status != "available":
            score -= config.stale_link_penalty

        meta = dict(meta)
        meta["persona"] = config.persona
        meta["persona_score"] = score

        scored_items.append((score, chunk, meta))

    scored_items.sort(key=lambda x: x[0], reverse=True)

    if top_k and top_k > 0:
        scored_items = scored_items[:top_k]

    return [c for _, c, _ in scored_items], [m for _, _, m in scored_items]
