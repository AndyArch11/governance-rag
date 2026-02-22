#!/usr/bin/env python
"""Compare extracted citations against a reference list file.

Usage:
  python examples/academic/compare_reference_extraction.py \
    --pdf /path/to/document.pdf \
    --reference-file /path/to/academic_references.txt
"""

from __future__ import annotations

import argparse
import re
from typing import List, Tuple

from scripts.ingest.academic.parser import extract_citations
from scripts.ingest.pdfparser import extract_text_from_pdf

URL_PATTERN = re.compile(
    r"https?://(?:[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=]|%[0-9A-Fa-f]{2})+",
    re.IGNORECASE,
)

NON_ASCII_HYPHENS = {"\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"}


def _strip_urls(text: str) -> str:
    if not text:
        return ""
    return URL_PATTERN.sub("", text)


def _normalise_line(text: str, strip_urls: bool = False) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if strip_urls:
        cleaned = _strip_urls(cleaned)
    # Normalise unicode hyphens to ASCII
    for h in NON_ASCII_HYPHENS:
        cleaned = cleaned.replace(h, "-")
        # Remove leading punctuation artifacts (e.g., PDF line breaks: ".Julien" -> "Julien")
        cleaned = re.sub(r"^\s*\.", "", cleaned)
        # Normalise spaces around punctuation and ampersands for author lists
        # "M., Wright" vs "M.,Wright" vs "M. , Wright" should all normalise the same
        cleaned = re.sub(r",\s*", ",", cleaned)  # Remove spaces after commas
        cleaned = re.sub(r"\s*&\s*", "&", cleaned)  # Normalise spaces around &
        cleaned = re.sub(r"\.\s+", ".", cleaned)  # Remove space after periods
        # Collapse remaining whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)
        # Remove trailing punctuation noise
        cleaned = re.sub(r"[\s\.,;:]+$", "", cleaned)
    # Lowercase for matching
    cleaned = cleaned.lower()
    return cleaned


def _load_reference_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def _extract_from_pdf(pdf_path: str) -> List[str]:
    text = extract_text_from_pdf(pdf_path)
    citations = extract_citations(text)
    return [c.raw_text for c in citations]


def _count_urls(lines: List[str]) -> int:
    return sum(1 for ln in lines if URL_PATTERN.search(ln))


def _diff_lists(
    expected: List[str], actual: List[str], strip_urls: bool = False
) -> Tuple[List[str], List[str]]:
    expected_norm = {_normalise_line(l, strip_urls=strip_urls): l for l in expected}
    actual_norm = {_normalise_line(l, strip_urls=strip_urls): l for l in actual}

    missing_norm = [k for k in expected_norm.keys() if k not in actual_norm]
    extra_norm = [k for k in actual_norm.keys() if k not in expected_norm]

    missing = [expected_norm[k] for k in missing_norm]
    extra = [actual_norm[k] for k in extra_norm]
    return missing, extra


def _print_samples(title: str, items: List[str], limit: int = 10) -> None:
    print(f"\n{title} (showing {min(limit, len(items))} of {len(items)}):")
    for item in items[:limit]:
        print(f"- {item}")


def _analyse_reference_quality(lines: List[str]) -> None:
    url_with_spaces = []
    url_trailing_hyphen = []
    missing_url = []
    cojoined_tokens = []

    for ln in lines:
        urls = URL_PATTERN.findall(ln)
        if not urls:
            missing_url.append(ln)
        for url in urls:
            if re.search(r"\s", url):
                url_with_spaces.append(ln)
            if url.endswith("-"):
                url_trailing_hyphen.append(ln)

        if re.search(r"[a-z][A-Z]", ln):
            cojoined_tokens.append(ln)

    print("\nReference list quality scan:")
    print(f"  Lines without URL: {len(missing_url)}")
    print(f"  URLs with spaces: {len(url_with_spaces)}")
    print(f"  URLs ending with hyphen: {len(url_trailing_hyphen)}")
    print(f"  Cojoined tokens (lowercase+uppercase): {len(cojoined_tokens)}")

    if url_with_spaces:
        _print_samples("URL with spaces", url_with_spaces, limit=10)
    if url_trailing_hyphen:
        _print_samples("URL trailing hyphen", url_trailing_hyphen, limit=10)
    if cojoined_tokens:
        _print_samples("Cojoined tokens", cojoined_tokens, limit=10)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare extracted citations to reference list")
    parser.add_argument("--pdf", help="Path to the source PDF")
    parser.add_argument("--reference-file", required=True, help="Path to academic_references.txt")
    parser.add_argument(
        "--extracted-file",
        help="Optional path to a text file with extracted citations (one per line)",
    )
    parser.add_argument(
        "--strip-urls",
        action="store_true",
        help="Ignore URLs when comparing citations",
    )
    args = parser.parse_args()

    reference_lines = _load_reference_lines(args.reference_file)

    extracted_lines: List[str] = []
    if args.extracted_file:
        extracted_lines = _load_reference_lines(args.extracted_file)
    elif args.pdf:
        extracted_lines = _extract_from_pdf(args.pdf)

    print("=" * 80)
    print("Reference Extraction Comparison")
    print("=" * 80)
    print(f"Reference list lines: {len(reference_lines)}")

    if extracted_lines:
        missing, extra = _diff_lists(reference_lines, extracted_lines, strip_urls=args.strip_urls)
        print(f"Extracted citations: {len(extracted_lines)}")
        print(f"Missing from extraction: {len(missing)}")
        print(f"Extra in extraction: {len(extra)}")

        print("\nURL coverage:")
        print(f"  Reference list URLs: { _count_urls(reference_lines) }")
        print(f"  Extracted URLs:      { _count_urls(extracted_lines) }")

        if missing:
            _print_samples("Missing citations", missing, limit=15)
        if extra:
            _print_samples("Extra citations", extra, limit=15)
    else:
        print("No extracted citations provided. Skipping extraction comparison.")

    _analyse_reference_quality(reference_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
