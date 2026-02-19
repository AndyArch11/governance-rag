"""Academic document parsing utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class ParsedCitation:
    raw_text: str
    doi: Optional[str] = None


REFERENCE_SECTION_PATTERN = re.compile(
    r"\b(references|bibliography|works\s+cited)\b", re.IGNORECASE
)
STOP_SECTION_PATTERN = re.compile(
    r"^\s*(appendix|acknowledg(e)?ments?|tables?|figures?|author\s+biograph(y|ies)|about\s+the\s+author|supplementary\s+material)\b",
    re.IGNORECASE,
)
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
# Key pattern: (YYYY). marks the boundary between authors and citation content
YEAR_DOT_PATTERN = re.compile(r"\((?:19|20)\d{2}[a-z]?\)\.")
NUMBERED_REF_PATTERN = re.compile(r"^(\[\d+\]|[1-9]\d{0,2}\.|\d+\))\s+")
AUTHOR_LEAD_PATTERN = re.compile(
    r"^(?:(?:[Vv]an|[Dd]e(?:r)?|[Dd]el|[Dd]a|[Dd]i|[Ll]e|[Ll]a)\s+)?"
    r"[^\W\d_][^\W\d_\-'’]*"
    r"(?:\s+[^\W\d_][^\W\d_\-'’]*)?"
    r"(?:-[^\W\d_][^\W\d_\-'’]*)?"
    r"(?:,\s*|\s+)"
    r"[A-Z](?:\.[A-Z])?\.?"
)
# Author pattern without comma before initials: "Lastname F." or "Lastname F.M."
NO_COMMA_AUTHOR_PATTERN = re.compile(
    r"^(?:(?:[Vv]an|[Dd]e(?:r)?|[Dd]el|[Dd]a|[Dd]i|[Ll]e|[Ll]a)\s+)?"
    r"[^\W\d_][^\W\d_\-'’]*"
    r"(?:\s+[^\W\d_][^\W\d_\-'’]*)?"
    r"(?:-[^\W\d_][^\W\d_\-'’]*)?"
    r"\s+"
    r"[A-Z](?:\.[A-Z])?\."
)

# Pattern for multi-author lists: "Lastname, F., Lastname, F., & Lastname, F."
# This catches cases where multiple authors with initials are listed
MULTI_AUTHOR_PATTERN = re.compile(r"^[A-Z][a-z]+,\s+[A-Z]\.,.*?(&|\&).*?[A-Z][a-z]+,\s+[A-Z]\.")

# Organisational/institutional references: "Organisation Name. (2023)" or "Organisation, (Acronym). (2023)"
ORG_REF_PATTERN = re.compile(
    r"^[^\W\d_][^\W\d_&''\-/\.,]*"
    r"(?:\s+[^\W\d_&''\-/\.,]+)*"
    r"(?:,?\s*\([A-Za-z&]{2,}\))?"
    r"\s*[\.,]?\s*\(\d{4}[a-z]?\)"
)

# Year variant pattern: (2019a), (2019b), etc. at start of line
# Catches continuation of multi-year citations
YEAR_VARIANT_PATTERN = re.compile(r"^\((?:19|20)\d{2}[a-z]\)")

# Potential reference starts for inline boundary detection
REF_START_PATTERN = re.compile(
    r"(?:"
    r"[^\W\d_][^\W\d_\-'’]*(?:-[^\W\d_][^\W\d_\-'’]*)?,\s*[A-Z]"
    r"|[^\W\d_][^\W\d_\-'’]*(?:-[^\W\d_][^\W\d_\-'’]*)?\s+[A-Z]\."
    r"|[^\W\d_][^\W\d_&'’\-/\.,]*(?:\s+[^\W\d_&'’\-/\.,]+)*(?:,?\s*\([A-Za-z&]{2,}\))?\.?\s*\(\d{4}[a-z]?\)"
    r"|\((?:19|20)\d{2}[a-z]\)"
    r")"
)
URL_REF_SPLIT_PATTERN = re.compile(
    r"(?:https?://[^\s]+?|doi:[^\s]+?|10\.\d{4,9}/[^\s]+?)(?=[A-Z][a-z\'&,]|"
    + REF_START_PATTERN.pattern
    + r")|"
    r"(?:https?://\S+|doi:\S+|10\.\d{4,9}/\S+)\s+(?=" + REF_START_PATTERN.pattern + r")"
)
# Period followed by reference start (with or without space)
# Also handles cases like "77-98.D'Cruz" where period is directly followed by apostrophe name
PERIOD_REF_SPLIT_PATTERN = re.compile(
    r"\.\s+(?=" + REF_START_PATTERN.pattern + r")|"
    r"\.(?=[A-Z]'[A-Z])"  # Period before capital-apostrophe-capital (e.g., ".D'Cruz")
)

TOC_DOT_LEADER_PATTERN = re.compile(r"\.{3,}\s*\d+\s*$")


def _is_heading(line: str) -> bool:
    if not line:
        return False
    if STOP_SECTION_PATTERN.search(line):
        return True
    return False


def _is_toc_line(line: str) -> bool:
    return bool(TOC_DOT_LEADER_PATTERN.search(line))


def _split_reference_lines(section_text: str) -> Iterable[str]:
    for line in section_text.splitlines():
        cleaned = " ".join(line.strip().split())
        if cleaned:
            yield cleaned


def _insert_spaces_in_cojoined_text(text: str) -> str:
    """
    Fix cojoined text where spaces were removed during PDF extraction.

    Only applies to words >= 20 characters long to avoid splitting legitimate
    multi-part names (AbuJbara, AlDhaen, McCalman are all <20 chars).
    Real PDF corruption creates very long cojoined strings like
    "AustralianBureauofStatistics" or "toolkitAustralianIndigenousHealthInfoNet".

    Primary fix: lowercase-to-uppercase transitions indicate missing spaces
    These are the most reliable indicators from PDF text extraction errors.

    Examples:
    - "analysisThis" -> "analysis This" (if 20+ chars)
    - "methodologyAnalysis" -> "methodology Analysis" (if 20+ chars)
    - "AbuJbara" -> "AbuJbara" (13 chars, not modified)

    Args:
        text: Text potentially with cojoined words

    Returns:
        Text with spaces inserted at likely word boundaries (20+ char words only)
    """
    if not text or len(text) < 3:
        return text

    # Process only words that are 18+ characters long
    # Split by whitespace, process each word, rejoin
    words = text.split()
    processed_words = []

    for word in words:
        if len(word) >= 18:
            # PRIMARY FIX: Insert space before capital letter that follows lowercase
            # "AustralianBureauofStatistics" -> "Australian Bureau of Statistics"
            word = re.sub(r"([a-z])([A-Z])", r"\1 \2", word)

            # SECONDARY FIX: Insert space after punctuation/closing brackets before capital letter
            # "word)Analysis" -> "word) Analysis"
            word = re.sub(r"([\)\],.:;])([A-Z][a-z]{2,})", r"\1 \2", word)

            # TERTIARY FIX: Multiple capitals followed by mixed case (acronym followed by word)
            # "XMLParser" -> "XML Parser" (if 20+ chars)
            word = re.sub(r"([A-Z]{2,})([A-Z][a-z])", r"\1 \2", word)

        processed_words.append(word)

    return " ".join(processed_words)


def _split_reference_blocks(section_text: str) -> List[str]:
    """
    Split references by looking for author name patterns that signal new references.

    After observing the PDF: references end with DOI/URL or page numbers,
    then a new reference starts with "LastName, F." pattern.
    """
    blocks: List[str] = []
    current_lines: List[str] = []

    for raw_line in section_text.splitlines():
        line = " ".join(raw_line.strip().split())

        if not line:
            continue

        if _is_heading(line):
            if current_lines:
                blocks.append(" ".join(current_lines).strip())
            break

        # Check if this line starts a NEW reference
        # Key heuristic: Line starts with "Author, I." pattern or numbered ref
        # AND we already have content in current
        starts_new_ref = False

        if current_lines:  # Only check if we have existing content
            # Check for multi-author list: "LastName, F., LastName, F., & LastName, F."
            # This is a strong indicator of a new reference
            if MULTI_AUTHOR_PATTERN.match(line):
                starts_new_ref = True
            # Check for year variant: (2019a), (2019b), etc.
            # This catches multi-year citations where org/author already in previous block
            elif YEAR_VARIANT_PATTERN.match(line):
                starts_new_ref = True
            # Check for author-led: "LastName, F." at start
            # Must have year pattern or be very long to avoid breaking mid-reference
            elif AUTHOR_LEAD_PATTERN.match(line):
                # Require the line also contains a year to confirm it's a new ref
                has_year_in_line = bool(YEAR_DOT_PATTERN.search(line))
                if has_year_in_line or len(line) > 100:
                    starts_new_ref = True
            # Check for author-led without comma: "LastName F."
            elif NO_COMMA_AUTHOR_PATTERN.match(line):
                has_year_in_line = bool(YEAR_DOT_PATTERN.search(line))
                if has_year_in_line or len(line) > 100:
                    starts_new_ref = True
            # Check for org-led references: "Organisation Name. (2023)"
            elif ORG_REF_PATTERN.match(line):
                starts_new_ref = True
            # Check for numbered: "[1]" or "1." at start
            # BUT reject if it's just a short line like "17. https://..." (likely a page number)
            elif NUMBERED_REF_PATTERN.match(line) and len(line) > 40:
                starts_new_ref = True

        if starts_new_ref:
            # Check if previous block ends with incomplete page range (e.g., "1–" or "1-")
            # If so, don't split - this is a continuation
            # BUT: exclude DOI URLs (10.xxxx/...-) which commonly end with hyphens
            if current_lines:
                last_line = current_lines[-1] if current_lines else ""
                # Pattern: ends with 1-4 digits followed by en-dash or hyphen (page range)
                # Must NOT be a DOI (would have "10." or "doi" somewhere before)
                is_page_range = bool(re.search(r"\d{1,4}[–-]\s*$", last_line))
                is_doi_or_url = bool(re.search(r"(?:doi\.org|10\.\d|https?://)", last_line))
                if is_page_range and not is_doi_or_url:
                    starts_new_ref = False

        if starts_new_ref:
            # Save current block and start new one
            blocks.append(" ".join(current_lines).strip())
            current_lines = [line]
        else:
            # Continue current reference
            current_lines.append(line)

    # Don't forget the last block
    if current_lines:
        blocks.append(" ".join(current_lines).strip())

    # Clean up trailing PDF page numbers from blocks
    cleaned_blocks = []
    for block in blocks:
        # Cojoining fix only applies to words >= 20 characters
        # This preserves legitimate multi-part names while fixing PDF corruption
        block = _insert_spaces_in_cojoined_text(block)

        # Remove standalone trailing page numbers (e.g., "... Management, 8(2), 81-89. 369")
        # Pattern: ends with period/URL/DOI followed by whitespace and 1-4 digit number
        cleaned = re.sub(r"(\.|https?://[^\s]+|10\.\d+/[^\s]+)\s+\d{1,4}$", r"\1", block)

        # Remove embedded page numbers in middle of text (e.g., "contract 370 violation")
        # Pattern: lowercase letter, space, 3-4 digit number, space, lowercase letter
        # This captures page numbers that appear mid-sentence
        cleaned = re.sub(r"([a-z])\s+(\d{3,4})\s+([a-z])", r"\1 \3", cleaned)

        # Fix broken URLs from PDF page breaks (e.g., "https://openresearch- repository" -> "https://openresearch-repository")
        # Remove spaces within URLs that were split across lines
        cleaned = re.sub(r"(https?://[^\s]*)-\s+([^\s]+)", r"\1-\2", cleaned)

        # Fix URLs with embedded spaces (common in PDF text extraction)
        # e.g., "https://example.com/path %20 more" or "...%E 2%80%93..."
        # This comprehensive regex finds URLs with internal spaces and consolidates them
        def fix_url_spaces(match):
            url = match.group(0)
            # Remove all internal spaces from URL
            url = url.replace(" ", "")
            # Handle percent-encoded characters that may have been split: "%E 2%80%93" -> "%E2%80%93"
            url = re.sub(r"%\s+([0-9A-Fa-f]{2})", r"%\1", url)
            # Remove trailing punctuation that's definitely not part of URL
            # Keep: / = & ? # : @ . - _ ~ [ ]
            # Remove: , ; ! ? ) } , " ' etc.
            url = re.sub(r'[,;:!?\)}\]"\']+$', "", url)
            return url

        # Find all URLs with embedded spaces and fix them
        # Pattern: http(s):// followed by various URL-valid characters and spaces
        cleaned = re.sub(
            r"https?://(?:[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=]|\s|%[0-9A-Fa-f]{2})+",
            fix_url_spaces,
            cleaned,
        )

        # Fix concatenated references: URLs/DOIs followed immediately by author name
        # Insert a space after URL/DOI when the next token looks like a new reference start
        # Updated with smarter URL matching that stops before author names (capital+lowercase)
        cleaned = re.sub(
            r"(https?://[^\s]+?|doi:[^\s]+?|10\.\d{4,9}/[^\s]+?)(?=[A-Z][a-z\'&,]|"
            + REF_START_PATTERN.pattern
            + r")",
            r"\1 ",
            cleaned,
        )

        # Insert missing space after URL/DOI if a new reference starts immediately after
        # This fixes cases like ".../toolkitAustralian Indigenous..." or "...doi...Braun, V."
        # Also handles concatenated org names: "...latest-releaseAustralianBureau..." -> "...latest-release Australian..."
        def insert_space_after_identifier(text: str) -> str:
            # Smarter URL pattern: stops before capital letter that starts author names
            # Pattern ends URL when followed by [A-Z][a-z] (author name) to prevent greedy matching
            # e.g., "...163604-13D'Cruz" -> URL ends at "13", not including "D'Cruz"
            pattern = re.compile(
                r"(https?://[^\s]+?(?=[A-Z][a-z\'&,]|(?<=[a-zA-Z0-9])\s|$)|doi:[^\s]+?(?=[A-Z][a-z]|(?<=[a-zA-Z0-9])\s|$)|10\.\d{4,9}/[^\s]+?(?=[A-Z][a-z]|(?<=[a-zA-Z0-9])\s|$))"
            )
            parts = []
            last = 0
            for match in pattern.finditer(text):
                end = match.end()
                parts.append(text[last:end])
                if end < len(text):
                    next_chunk = text[end : end + 200]

                    # Check if next character is a capital letter (start of new reference)
                    # This catches: URL followed by CapitalLetter with no space
                    if next_chunk and next_chunk[0].isupper() and not next_chunk[0].isdigit():
                        # Check if it looks like a reference start (not just a random word)
                        if (
                            AUTHOR_LEAD_PATTERN.match(next_chunk)
                            or NO_COMMA_AUTHOR_PATTERN.match(next_chunk)
                            or ORG_REF_PATTERN.match(next_chunk)
                            or YEAR_VARIANT_PATTERN.match(next_chunk)
                            or re.match(
                                r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\((?:19|20)\d{2}", next_chunk
                            )  # Org name with year
                        ):
                            parts.append(" ")
                    elif (
                        AUTHOR_LEAD_PATTERN.match(next_chunk)
                        or NO_COMMA_AUTHOR_PATTERN.match(next_chunk)
                        or ORG_REF_PATTERN.match(next_chunk)
                        or YEAR_VARIANT_PATTERN.match(next_chunk)
                    ):
                        parts.append(" ")
                last = end
            parts.append(text[last:])
            return "".join(parts)

        cleaned = insert_space_after_identifier(cleaned)

        # Remove broken URL fragments with hyphenated words (e.g., "#cite- window1" -> "")
        # Pattern: #word- space word (indicates PDF line break in fragment)
        # Only remove if it looks like a broken anchor (cite, window, etc. followed by dash-space-word)
        cleaned = re.sub(r"#(?:cite|window|section|part)[-\s]+\w+", "", cleaned)

        cleaned_blocks.append(cleaned)

    # SECOND PASS: Split any merged references within cleaned blocks
    # Look for blocks with multiple (YYYY). or (YYYY), patterns
    # Use broader pattern for second pass to catch book chapters and different citation styles
    YEAR_SPLIT_PATTERN = re.compile(r"\((?:19|20)\d{2}[a-z]?\)[.,]")
    # For org references without punctuation after year: "Org Name (YYYYx) NextWord"
    # This specifically targets organisation citations with year+letter followed by capitalised next word
    ORG_YEAR_BOUNDARY_PATTERN = re.compile(
        r"([A-Z][\w\s&]+?)\s*\((?:19|20)\d{2}[a-z]\)\s+(?=[A-Z][a-z])"
    )

    final_blocks = []
    for block in cleaned_blocks:
        url_splits = list(URL_REF_SPLIT_PATTERN.finditer(block))
        if url_splits:
            sub_blocks = []
            last_end = 0

            for match in url_splits:
                split_pos = match.end()
                prefix = block[last_end:split_pos].strip()
                candidate = block[split_pos:].strip()

                starts_like_ref = bool(
                    AUTHOR_LEAD_PATTERN.match(candidate)
                    or NO_COMMA_AUTHOR_PATTERN.match(candidate)
                    or ORG_REF_PATTERN.match(candidate)
                    or YEAR_VARIANT_PATTERN.match(candidate)
                )
                prefix_has_year = bool(YEAR_PATTERN.search(prefix))

                if prefix_has_year and starts_like_ref and len(prefix) > 40:
                    sub_blocks.append(prefix)
                    last_end = split_pos

            remainder = block[last_end:].strip()
            if sub_blocks and remainder:
                sub_blocks.append(remainder)
                final_blocks.extend(sub_blocks)
                continue

        period_splits = list(PERIOD_REF_SPLIT_PATTERN.finditer(block))
        if period_splits:
            sub_blocks = []
            last_end = 0

            for match in period_splits:
                split_pos = match.end()
                prefix = block[last_end : match.start() + 1].strip()
                candidate = block[split_pos:].strip()

                starts_like_ref = bool(
                    AUTHOR_LEAD_PATTERN.match(candidate)
                    or NO_COMMA_AUTHOR_PATTERN.match(candidate)
                    or ORG_REF_PATTERN.match(candidate)
                    or YEAR_VARIANT_PATTERN.match(candidate)
                )
                prefix_has_year = bool(YEAR_PATTERN.search(prefix))
                candidate_has_year = bool(YEAR_PATTERN.search(candidate[:200]))

                if prefix_has_year and candidate_has_year and starts_like_ref and len(prefix) > 40:
                    sub_blocks.append(prefix)
                    last_end = split_pos

            remainder = block[last_end:].strip()
            if sub_blocks and remainder:
                sub_blocks.append(remainder)
                final_blocks.extend(sub_blocks)
                continue

        year_positions = list(YEAR_SPLIT_PATTERN.finditer(block))
        org_year_positions = list(ORG_YEAR_BOUNDARY_PATTERN.finditer(block))

        # If we have org year patterns, mark them as potential split points
        has_org_split = len(org_year_positions) > 0

        if len(year_positions) <= 1 and not has_org_split:
            # Single reference or org ref - keep as is
            final_blocks.append(block)
        elif len(year_positions) <= 1 and has_org_split:
            # Has org year pattern but only one punctuated year - split on org pattern
            # For cases like "...ABC. (2019a). ... Organisation (2019b) Microdata..."
            # Split at the organisation pattern boundary
            sub_blocks = []
            last_end = 0

            for match in org_year_positions:
                # The match includes the org name and year, find where to split (before the org name)
                org_start = match.start()
                sub_ref = block[last_end:org_start].strip()
                if len(sub_ref) > 25:
                    sub_blocks.append(sub_ref)
                last_end = org_start

            # Add the last part
            final_sub = block[last_end:].strip()
            if len(final_sub) > 25:
                sub_blocks.append(final_sub)

            if sub_blocks:
                final_blocks.extend(sub_blocks)
            else:
                final_blocks.append(block)
        else:
            # Multiple (YYYY). or (YYYY), detected - likely merged references
            # Split at each year pattern, looking backwards for author names
            sub_blocks = []

            for i, match in enumerate(year_positions):
                if i == 0:
                    # First sub-reference goes from start to this year's end
                    start = 0
                else:
                    # Subsequent refs: find where previous one ended
                    start = prev_end

                year_end = match.end()

                # Find end of this reference
                if i + 1 < len(year_positions):
                    # Look for boundary before next year
                    next_year_start = year_positions[i + 1].start()
                    segment = block[year_end:next_year_start]

                    # Find last sentence/URL boundary in segment
                    boundary_matches = list(
                        re.finditer(
                            r"(https?://[^\s]+|10\.\d+/[^\s]+|doi:[^\s]+)\s+(?=[A-Z])", segment
                        )
                    )

                    if boundary_matches:
                        # End at last boundary
                        end = year_end + boundary_matches[-1].end()
                    else:
                        # Try looking for period + space before capital
                        period_matches = list(re.finditer(r"\.\s+(?=[A-Z])", segment))
                        if period_matches:
                            end = year_end + period_matches[-1].end()
                        else:
                            # No clear boundary - split at next year's author start
                            # Back up from next_year_start to find author pattern
                            author_search = block[:next_year_start]
                            author_matches = list(AUTHOR_LEAD_PATTERN.finditer(author_search))

                            if author_matches and author_matches[-1].start() > year_end:
                                end = author_matches[-1].start()
                            else:
                                # Can't find good split - take up to next year
                                end = next_year_start
                else:
                    # Last reference in this block
                    end = len(block)

                sub_ref = block[start:end].strip()
                # Validate: must start with author pattern or be long enough to be valid
                if len(sub_ref) > 25:
                    # Check if it starts like a real reference (not a fragment like "17. https://...")
                    starts_valid = bool(
                        AUTHOR_LEAD_PATTERN.match(sub_ref)
                        or NO_COMMA_AUTHOR_PATTERN.match(sub_ref)
                        or ORG_REF_PATTERN.match(sub_ref)
                        or NUMBERED_REF_PATTERN.match(sub_ref)
                    )
                    if starts_valid or i == 0:  # Always keep first sub-block
                        sub_blocks.append(sub_ref)

                prev_end = end

            final_blocks.extend(sub_blocks)

    return final_blocks


def _filter_reference_candidates(lines: Iterable[str]) -> List[str]:
    candidates: List[str] = []
    for line in lines:
        if len(line) < 15:
            continue
        lower = line.lower()
        has_identifier = (
            "doi" in lower
            or DOI_PATTERN.search(line)
            or "http://" in lower
            or "https://" in lower
            or "arxiv.org" in lower
            or "doi.org" in lower
        )
        has_year = bool(YEAR_PATTERN.search(line))
        if NUMBERED_REF_PATTERN.match(line) and (has_year or has_identifier):
            candidates.append(line)
        elif has_identifier or has_year:
            candidates.append(line)
    return candidates


def _is_reference_like(block: str) -> bool:
    if len(block) < 25:
        return False
    lower = block.lower()
    has_identifier = (
        "doi" in lower
        or DOI_PATTERN.search(block)
        or "http://" in lower
        or "https://" in lower
        or "arxiv.org" in lower
        or "doi.org" in lower
    )
    has_year = bool(YEAR_PATTERN.search(block))
    # Require either numbered start or author-like structure at beginning
    has_structure = bool(
        NUMBERED_REF_PATTERN.match(block)
        or AUTHOR_LEAD_PATTERN.match(block)
        or NO_COMMA_AUTHOR_PATTERN.match(block)
        or ORG_REF_PATTERN.match(block)
    )
    return (has_year or has_identifier) and has_structure


def _has_malformed_author(block: str) -> bool:
    """Detect references with incomplete or malformed author information."""
    # Strip numbered prefix if present ([1], 1., etc.)
    cleaned = NUMBERED_REF_PATTERN.sub("", block)

    # Check for single capital letter followed by parenthesis: "C. (2017)"
    # Indicates missing author surname
    if re.match(r"^[A-Z]\.\s*\((?:19|20)\d{2}", cleaned):
        return True

    # Check for author format with space between last name and initials without comma
    # Pattern: "Lastname I. I." (no comma), should be "Lastname, I. I."
    # But allow if it's part of a hyphenated name or already has proper comma
    if re.match(r"^[A-Z][-A-Za-z]+\s+[A-Z]\.\s+[A-Z]\.", cleaned):
        # Check if there's already a comma in the author section
        # If no comma in first 50 chars, it's likely malformed
        first_part = cleaned[:100]
        if "," not in first_part and re.search(r"\((?:19|20)\d{2}", first_part):
            # This looks like "Lastname I. I. (YEAR)" without comma
            return True

    return False


def _is_orphaned_journal_reference(block: str) -> bool:
    """Detect journal-only references without author information (incomplete extractions)."""
    # Strip numbered prefix if present ([1], 1., etc.)
    cleaned = NUMBERED_REF_PATTERN.sub("", block)

    # Pattern: Starts with journal name (title case, with commas), has volume/issue
    # But NO author information at the start
    # Example: "Race, Ethnicity and Education, 19(4), 784-807. https://..."

    # If it has a TRUE author pattern (Lastname, Initial. or Initial.), not orphaned
    # Match: "LastName, I." or "LastName, I.M." not "Word, AnotherWord"
    if re.match(r"^[A-Z][a-z]+,\s+[A-Z]\.", cleaned):
        return False

    # If it has organisation pattern, not orphaned
    if ORG_REF_PATTERN.match(cleaned):
        return False

    # Check for journal reference pattern: starts with capital words, has commas and numbers
    # "Title Case Journal, Vol(Issue)" pattern
    # This will catch: "Race, Ethnicity and Education, 19(4)..."
    # because after "Education, " comes a number directly
    if re.match(r"^[A-Z][A-Za-z,&\s-]+,\s*\d+\(", cleaned):
        return True

    return False


def _extract_candidates_from_section(section: str) -> List[str]:
    blocks = _split_reference_blocks(section)

    candidates: List[str] = []
    non_ref_streak = 0
    for block in blocks:
        # Skip malformed and orphaned references
        if _has_malformed_author(block) or _is_orphaned_journal_reference(block):
            continue
        if _is_reference_like(block):
            candidates.append(block)
            non_ref_streak = 0
        elif candidates:
            non_ref_streak += 1
            if non_ref_streak >= 40:
                break

    if candidates:
        return candidates

    lines = _split_reference_lines(section)
    return _filter_reference_candidates(lines)


def extract_citations(text: str) -> List[ParsedCitation]:
    """Extract citations from text using a heuristic reference-section parser.

    Args:
        text: Full text of the document to extract citations from.

    Returns:
        List of ParsedCitation objects containing raw text and optional DOI.
    """
    if not text:
        return []

    matches = list(REFERENCE_SECTION_PATTERN.finditer(text))
    if not matches:
        return []

    best_candidates: List[str] = []
    for match in matches:
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end == -1:
            line_end = len(text)
        line = text[line_start:line_end].strip()
        if _is_toc_line(line):
            continue

        section = text[match.end() :]
        candidates = _extract_candidates_from_section(section)
        if len(candidates) > len(best_candidates):
            best_candidates = candidates

    if not best_candidates:
        return []

    citations: List[ParsedCitation] = []
    for line in best_candidates:
        # Cojoining fix only applies to words >= 20 characters
        # This preserves legitimate multi-part names while fixing real PDF corruption
        cleaned_line = _insert_spaces_in_cojoined_text(line)

        doi_match = DOI_PATTERN.search(cleaned_line)
        citations.append(
            ParsedCitation(raw_text=cleaned_line, doi=doi_match.group(0) if doi_match else None)
        )

    return citations
