#!/usr/bin/env python3
"""Compare extracted citations against a reference list file.

Usage:
    python3 examples/academic/compare_reference_extraction.py \
    --pdf /path/to/document.pdf \
    --reference-file /path/to/academic_references.txt
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Tuple

# Ensure project modules are importable when run as a script path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ingest.academic.parser import extract_citations
from scripts.ingest.pdfparser import extract_text_from_pdf

URL_PATTERN = re.compile(
    r"https?://(?:[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=]|%[0-9A-Fa-f]{2})+",
    re.IGNORECASE,
)
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
REFERENCE_YEAR_PATTERN = re.compile(r"\((?:19|20)\d{2}[a-z]?\)")

NON_ASCII_HYPHENS = {"\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"}


def _strip_urls(text: str) -> str:
    """Remove URLs from the given text.

    Args:
        text: Text potentially containing URLs.

    Returns:
        Text with URLs removed.
    """
    if not text:
        return ""
    return URL_PATTERN.sub("", text)


def _normalise_line(text: str, strip_urls: bool = False) -> str:
    """Normalise a line of reference text for comparison.

    Args:
        text: The reference line to normalise.
        strip_urls: Whether to remove URLs from the text.

    Returns:
        The normalised reference line.
    """
    if not text:
        return ""
    cleaned = text.strip()
    if strip_urls:
        cleaned = _strip_urls(cleaned)
    cleaned = html.unescape(cleaned)
    # Normalise unicode hyphens to ASCII
    for h in NON_ASCII_HYPHENS:
        cleaned = cleaned.replace(h, "-")
    # Remove leading punctuation artifacts (e.g., PDF line breaks: ".Julien" -> "Julien")
    cleaned = re.sub(r"^\s*\.", "", cleaned)
    # Normalise spaces around punctuation and ampersands for author lists.
    cleaned = re.sub(r"\s*,\s*", ",", cleaned)
    cleaned = re.sub(r"\s*&\s*", "&", cleaned)
    cleaned = re.sub(r"\.\s+", ".", cleaned)
    # Collapse remaining whitespace and remove trailing extraction punctuation.
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[\s\.,;:-]+$", "", cleaned)
    # Lowercase for matching
    cleaned = cleaned.lower()
    return cleaned


def _canonical_line(text: str, strip_urls: bool = False) -> str:
    """Create a formatting-tolerant key for reference comparison."""
    cleaned = _normalise_line(text, strip_urls=strip_urls)
    cleaned = unicodedata.normalize("NFKD", cleaned)
    return re.sub(r"[^a-z0-9]+", "", cleaned)


def _canonical_diff(
    expected: List[str], actual: List[str], strip_urls: bool = False
) -> Tuple[List[str], List[str], int]:
    """Compare references after removing PDF formatting-only differences."""
    expected_norm = {_canonical_line(line, strip_urls=strip_urls): line for line in expected}
    actual_norm = {_canonical_line(line, strip_urls=strip_urls): line for line in actual}
    matched = len(expected_norm.keys() & actual_norm.keys())
    missing = [expected_norm[key] for key in expected_norm.keys() - actual_norm.keys()]
    extra = [actual_norm[key] for key in actual_norm.keys() - expected_norm.keys()]
    return missing, extra, matched


def _load_reference_lines(path: str) -> List[str]:
    """Load reference lines from a text file, stripping empty lines and whitespace.

    Args:
        path: Path to the text file containing reference lines.

    Returns:
        A list of non-empty, stripped reference lines.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def _extract_from_pdf(pdf_path: str) -> List[str]:
    """Extract reference lines from a PDF file using the academic parser.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of extracted reference lines from the PDF.
    """
    text = extract_text_from_pdf(pdf_path)
    citations = extract_citations(text)
    return [c.raw_text for c in citations]


def _count_urls(lines: List[str]) -> int:
    """Count the number of lines containing URLs in a list of reference lines.

    Args:
        lines: A list of reference lines to check for URLs.

    Returns:
        The number of lines that contain at least one URL.
    """
    return sum(1 for ln in lines if URL_PATTERN.search(ln))


def _normalised_dois(lines: List[str]) -> set[str]:
    """Return canonical DOI values present in reference lines.
    
    Args:
        lines: A list of reference lines to extract DOIs from.

    Returns:
        A set of canonical DOI values found in the reference lines.
    """
    return {
        doi.lower().rstrip(".,;:-")
        for line in lines
        for doi in DOI_PATTERN.findall(line)
    }


def _find_merged_reference_candidates(lines: List[str]) -> List[str]:
    """Identify extracted lines that appear to contain multiple bibliography entries.

    Args:
        lines: A list of reference lines to check for merged entries.

    Returns:
        A list of lines that likely contain multiple bibliography entries.
    """
    return [line for line in lines if len(REFERENCE_YEAR_PATTERN.findall(line)) > 1]


def _diff_lists(
    expected: List[str], actual: List[str], strip_urls: bool = False
) -> Tuple[List[str], List[str]]:
    """Compare two lists of reference lines and return missing and extra items.

    Args:
        expected: The list of expected reference lines.
        actual: The list of actual reference lines extracted.
        strip_urls: Whether to normalise lines by removing URLs before comparison.

    Returns:
        A tuple containing two lists:
        - The first list contains reference lines missing from the actual list.
        - The second list contains extra reference lines present in the actual list.
    """
    expected_norm = {_normalise_line(l, strip_urls=strip_urls): l for l in expected}
    actual_norm = {_normalise_line(l, strip_urls=strip_urls): l for l in actual}

    missing_norm = [k for k in expected_norm.keys() if k not in actual_norm]
    extra_norm = [k for k in actual_norm.keys() if k not in expected_norm]

    missing = [expected_norm[k] for k in missing_norm]
    extra = [actual_norm[k] for k in extra_norm]
    return missing, extra


def _print_samples(title: str, items: List[str], limit: int = 10) -> None:
    """Print a sample of items with a title.

    Args:
        title: The title to display before the sample items.
        items: The list of items to sample from.
        limit: The maximum number of items to display.
    """
    print(f"\n{title} (showing {min(limit, len(items))} of {len(items)}):")
    for item in items[:limit]:
        print(f"- {item}")


def _analyse_reference_quality(lines: List[str]) -> None:
    """Analyse the quality of reference lines for common issues.

    Args:
        lines: A list of reference lines to analyse.

    Returns:
        None. Prints a summary of issues found in the reference lines.
    """
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
    """Main entry point for the reference extraction comparison script.

    Returns:
        An integer exit code (0 for success).
    """
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
        canonical_missing, canonical_extra, canonical_matches = _canonical_diff(
            reference_lines, extracted_lines, strip_urls=args.strip_urls
        )
        expected_dois = _normalised_dois(reference_lines)
        extracted_dois = _normalised_dois(extracted_lines)
        missing_dois = expected_dois - extracted_dois
        merged_candidates = _find_merged_reference_candidates(extracted_lines)
        print(f"Extracted citations: {len(extracted_lines)}")
        print(f"Missing from extraction: {len(missing)}")
        print(f"Extra in extraction: {len(extra)}")
        print(f"Reference count recall: {len(extracted_lines) / len(reference_lines):.1%}")
        print(f"Canonical reference matches: {canonical_matches}")
        print(f"Canonical unmatched expected: {len(canonical_missing)}")
        print(f"Canonical unmatched extracted: {len(canonical_extra)}")

        print("\nURL coverage:")
        print(f"  Reference list URLs: { _count_urls(reference_lines) }")
        print(f"  Extracted URLs:      { _count_urls(extracted_lines) }")

        print("\nDOI recovery:")
        print(f"  Reference list DOIs: {len(expected_dois)}")
        print(f"  Extracted DOIs:      {len(extracted_dois)}")
        print(f"  Matched DOIs:        {len(expected_dois & extracted_dois)}")
        print(f"  Missing DOIs:        {len(missing_dois)}")

        print(f"\nPossible merged citation blocks: {len(merged_candidates)}")

        if missing:
            _print_samples("Missing citations", missing, limit=15)
        if extra:
            _print_samples("Extra citations", extra, limit=15)
        if merged_candidates:
            _print_samples("Possible merged citation blocks", merged_candidates, limit=15)
    else:
        print("No extracted citations provided. Skipping extraction comparison.")

    _analyse_reference_quality(reference_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
