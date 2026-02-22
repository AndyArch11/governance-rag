"""Heuristic validator for academic reference metadata.

Scans academic_reference chunks and flags common extraction issues:
- Cojoined citations (multiple years / multiple URLs)
- URLs with spaces
- URLs ending with hyphens (truncated)
- Very long tokens (likely missing whitespace)
- Suspicious punctuation/encoding issues

Usage:
    python examples/academic/validate_academic_references.py --limit 5000
    python examples/academic/validate_academic_references.py --only-reference-metadata
    python examples/academic/validate_academic_references.py --output-json rag_data/academic_ref_issues.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from scripts.ingest.academic.config import get_academic_config
from scripts.utils.db_factory import get_default_vector_path, get_vector_client

URL_REGEX = re.compile(r"https?://[^\s\)\]\}>\"']+", re.IGNORECASE)
YEAR_REGEX = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
LONG_TOKEN_REGEX = re.compile(r"[A-Za-z]{20,}")
NON_ASCII_HYPHENS = {"\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"}


def _extract_text(meta: Dict[str, Any]) -> str:
    return (
        meta.get("citation") or meta.get("title") or meta.get("doc_id") or meta.get("source") or ""
    )


def _find_urls(text: str) -> List[str]:
    return URL_REGEX.findall(text)


def _has_url_spaces(text: str) -> bool:
    # Detect URLs split by spaces (e.g., "https://.../file %20name.pdf")
    return bool(re.search(r"https?://\S*\s+\S*", text))


def _url_trailing_hyphen(url: str) -> bool:
    return url.endswith("-") or url.endswith("–") or url.endswith("—")


def _contains_non_ascii_hyphen(text: str) -> bool:
    return any(ch in text for ch in NON_ASCII_HYPHENS)


def _long_tokens(text: str) -> List[str]:
    return LONG_TOKEN_REGEX.findall(text)


def _detect_issues(text: str) -> Dict[str, Any]:
    issues: Dict[str, Any] = {}

    urls = _find_urls(text)
    if len(urls) > 1:
        issues["multiple_urls"] = urls

    years = YEAR_REGEX.findall(text)
    if len(years) > 1:
        issues["multiple_years"] = len(years)

    if _has_url_spaces(text):
        issues["url_has_spaces"] = True

    truncated_urls = [u for u in urls if _url_trailing_hyphen(u)]
    if truncated_urls:
        issues["url_trailing_hyphen"] = truncated_urls

    long_tokens = _long_tokens(text)
    if long_tokens:
        issues["long_tokens"] = long_tokens[:5]  # cap

    if _contains_non_ascii_hyphen(text):
        issues["non_ascii_hyphen"] = True

    return issues


def _iter_academic_chunks(
    limit: int,
    only_reference_metadata: bool,
) -> Iterable[Dict[str, Any]]:
    config = get_academic_config()
    PersistentClient, _using_sqlite = get_vector_client(prefer="chroma")
    vector_path = get_default_vector_path(Path(config.rag_data_path), _using_sqlite)
    client = PersistentClient(path=vector_path)
    collection = client.get_or_create_collection(config.chunk_collection_name)

    where = {"source_category": "academic_reference"}

    results = collection.get(
        where=where,
        limit=limit,
        include=["metadatas"],
    )

    for meta in results.get("metadatas", []) or []:
        if not isinstance(meta, dict):
            continue
        if only_reference_metadata:
            source_val = str(meta.get("source", ""))
            if "reference_metadata:" not in source_val:
                continue
        yield meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate academic reference metadata for common extraction issues"
    )
    parser.add_argument("--limit", type=int, default=5000, help="Max references to scan")
    parser.add_argument(
        "--only-reference-metadata",
        action="store_true",
        help="Only scan reference_metadata sources",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    issues_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total = 0

    for meta in _iter_academic_chunks(args.limit, args.only_reference_metadata):
        total += 1
        text = _extract_text(meta)
        if not text:
            continue

        issues = _detect_issues(text)
        if not issues:
            continue

        record = {
            "doc_id": meta.get("doc_id"),
            "source": meta.get("source"),
            "title": meta.get("title"),
            "citation": meta.get("citation"),
            "issues": issues,
        }

        for issue_type in issues.keys():
            issues_by_type[issue_type].append(record)

    # Print summary
    print(f"Scanned {total} academic references")
    print("Issue summary:")
    for issue_type, items in sorted(issues_by_type.items()):
        print(f"  - {issue_type}: {len(items)}")

    # Show samples
    print("\nSamples (up to 3 per issue):")
    for issue_type, items in sorted(issues_by_type.items()):
        print(f"\n[{issue_type}]")
        for item in items[:3]:
            snippet = (item.get("citation") or item.get("title") or "")[:180]
            print(f"- doc_id={item.get('doc_id')} | {snippet}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(issues_by_type, f, indent=2, ensure_ascii=False)
        print(f"\nWrote JSON report to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
