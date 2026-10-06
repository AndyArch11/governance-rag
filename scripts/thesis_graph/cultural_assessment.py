"""Surface profile-indicator evidence for human cultural review."""

from __future__ import annotations

import re
from typing import Any


def _indicator_excerpt(document: str, match: re.Match[str], context_chars: int = 100) -> str:
    """Return a bounded excerpt around an indicator match.

    This function returns a substring with bounded context around a matched phrase.

    Args:
        document: The text of the document chunk.
        match: The matched phrase span in the document.
        context_chars: Number of characters to include before and after the match.

    Returns:
        A string excerpt surrounding the matched phrase.
    """
    start = max(0, match.start() - context_chars)
    end = min(len(document), match.end() + context_chars)
    excerpt = document[start:end].strip()
    if start:
        excerpt = f"...{excerpt}"
    if end < len(document):
        excerpt = f"{excerpt}..."
    return excerpt


def find_cultural_criteria_evidence(
    profile: dict[str, Any],
    chunk_ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find candidate indicator matches without scoring or judging criterion absence.
    Args:
        profile: The cultural lens profile containing assessment criteria.
        chunk_ids: List of chunk IDs corresponding to the text chunks.
        documents: List of text chunks from the thesis.
        metadatas: List of metadata dictionaries corresponding to the text chunks.

    Returns:
        A list of dictionaries, each representing a criterion with its matched indicators and review information.

    Raises:
        ValueError: If chunk IDs, documents and metadata have different lengths.
    """
    if len(chunk_ids) != len(documents) or len(documents) != len(metadatas):
        raise ValueError("chunk IDs, documents and metadata must have equal lengths")

    results = []
    for criterion in profile.get("assessment_criteria", []):
        indicators = [
            str(indicator).strip()
            for indicator in criterion.get("evidence_indicators", [])
            if str(indicator).strip()
        ]
        matches = []
        for chunk_id, document, metadata in zip(chunk_ids, documents, metadatas):
            indicator_matches = []
            for indicator in indicators:
                escaped_words = r"\s+".join(re.escape(word) for word in indicator.split())
                pattern = re.compile(rf"(?<!\w){escaped_words}(?!\w)", re.IGNORECASE)
                match = pattern.search(document)
                if match:
                    indicator_matches.append((indicator, match))
            if indicator_matches:
                matched_indicators = [indicator for indicator, _ in indicator_matches]
                chunk_source_start = metadata.get("source_start")
                indicator_evidence = [
                    {
                        "indicator": indicator,
                        "text": _indicator_excerpt(document, match),
                        "source_start": (
                            chunk_source_start + match.start()
                            if isinstance(chunk_source_start, int) and chunk_source_start >= 0
                            else None
                        ),
                        "source_end": (
                            chunk_source_start + match.end()
                            if isinstance(chunk_source_start, int) and chunk_source_start >= 0
                            else None
                        ),
                    }
                    for indicator, match in indicator_matches
                ]
                matches.append(
                    {
                        "chunk_id": chunk_id,
                        "section": metadata.get("heading_path")
                        or metadata.get("section_title")
                        or metadata.get("chapter")
                        or "Unclassified",
                        "matched_indicators": matched_indicators,
                        "indicator_evidence": indicator_evidence,
                        "text": indicator_evidence[0]["text"],
                    }
                )
        results.append(
            {
                "id": criterion.get("id", ""),
                "criterion": criterion.get("criterion", ""),
                "description": criterion.get("description", ""),
                "indicator_matches": matches,
                "review_required": True,
                "review_status": criterion.get("review_status", "draft"),
            }
        )
    return results
