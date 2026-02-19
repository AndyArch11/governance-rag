"""PDF extraction and cleaning for document ingestion.

Removes headers, footers, page numbers, and other boilerplate from PDFs
before text extraction to reduce junk content.

Structure Extraction:
    - Identifies chapter headings and numbers
    - Detects section and subsection hierarchies
    - Builds heading paths for contextual metadata

TODO: Consider using a different PDF extraction tool:
- grobid: Excellent for scientific papers, extracts structured metadata and sections
- nougat: A newer library focused on clean text extraction from PDFs, with built-in boilerplate removal
- docling PDFParser: Designed for LLM ingestion, with features for cleaning and structuring PDF content
    - requires docling-heirarchical-pdf for extracting hierarchical structure including chapters and sections
- MarkItDown: A tool for converting PDFs to Markdown with structure and boilerplate removal, good for LLM ingestion
- unstructured: A powerful library for extracting structured data from PDFs, including tables and metadata, with good handling of complex layouts
- pdfplumber: More robust layout analysis, better for complex PDFs
- pdfminer.six: More control over text extraction, can handle some edge cases better
- Apache Tika: Can extract text from a wide variety of formats, including PDFs, with good metadata extraction
- PyMuPDF (fitz): Fast and can extract text with layout information, good for structured documents
- pypdfium2: Uses PDFium for rendering and text extraction, can handle complex PDFs with better accuracy
- pymupdf4llm: A wrapper around PyMuPDF optimised for LLM ingestion, with built-in cleaning and structure extraction features
- marker-pdf: A tool designed for extracting structured content from PDFs, with features for identifying and removing boilerplate, and extracting document structure for LLM ingestion
- texxtract: A library for extracting text and metadata from PDFs, with support for cleaning and structuring content for LLM ingestion
- pdf2txt: A command-line tool that can extract text from PDFs with options for cleaning and structuring the output, useful for preprocessing PDFs before ingestion into LLMs
TODO: Improve structure extraction into chunking process to ensure chunks are aware of their section/chapter context, which can improve retrieval relevance and allow for more targeted question answering (e.g., "What methodology was used?" can be directed to the Methods section)
TODO: Improve functionality to extract and store document structure (chapters, sections) as metadata in ChromaDB, which can be used for more advanced retrieval and question answering based on document hierarchy
TODO: Improve functionality to detect and handle document citations and references, which can be important for understanding the context and relationships between documents, especially in academic papers, and may require special handling to ensure that cited documents are also ingested and indexed in ChromaDB for comprehensive retrieval and question answering capabilities
TODO: Improve functionality to detect and handle document metadata (e.g., author, publication date, keywords), which can be important for understanding the context and relevance of documents, and can be used as additional metadata in ChromaDB for improved retrieval and question answering capabilities
TODO: Add functionality to detect and handle multi-column layouts, which can cause issues with text extraction and structure detection, especially in academic papers and reports
TODO: Add functionality to detect and handle tables and figures, which are common in PDFs and can contain important information that may not be captured well by standard text extraction methods
TODO: Add functionality to detect and handle scanned PDFs (images), which require OCR for text extraction, and may have different structure and boilerplate patterns compared to digitally generated PDFs
TODO: Add functionality to detect and handle embedded fonts and special characters, which can cause issues with text extraction and may require additional processing to ensure the extracted text is clean and usable for LLM ingestion
TODO: Add functionality to detect and handle different languages and character sets, which may require different cleaning and structure extraction approaches, especially for non-Latin scripts
TODO: Add functionality to detect and handle different document types (e.g., academic papers, business reports, legal documents), which may have different structure and boilerplate patterns, and may benefit from tailored extraction and cleaning approaches for optimal LLM ingestion
TODO: Add functionality to detect and handle document updates and versioning, which can be important for maintaining an up-to-date knowledge base, especially for documents that are frequently updated (e.g., living documents, online reports), and may require re-ingestion and updating of the ChromaDB index when changes are detected
TODO: Add functionality to detect and handle document access restrictions (e.g., paywalls, login requirements), which can affect the ability to ingest certain documents, and may require special handling (e.g., using APIs, web scraping with authentication) to access and ingest the content for LLM ingestion
TODO: Add functionality to detect and handle document quality issues (e.g., low-resolution scans, corrupted files), which can affect the ability to extract usable text and structure, and may require additional processing (e.g., image enhancement for scanned PDFs) or exclusion from ingestion if the quality is too poor for effective LLM ingestion
TODO: Add functionality to scan for viruses and malware in PDFs before ingestion, especially if ingesting from untrusted sources, to ensure the security of the system and prevent potential harm from malicious documents
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from pypdf import PdfReader

from scripts.utils.retry_utils import retry_with_backoff

# Common PDF header/footer patterns
PDF_BOILERPLATE_PATTERNS = [
    r"^Page \d+\s*(?:of|/)?\s*\d*",  # Page numbers
    r"^\d+\s*$",  # Standalone page numbers
    r"^Copyright\s+[©©]\s*\d{4}",  # Copyright
    r"^Version\s+\d+\.\d+(\.\d+)?",  # Version numbers
    r"^v\d+\.\d+",  # v-prefixed versions
    r"^Generated\s+on\s+\d{1,2}[-/]\d{1,2}[-/]\d{4}",  # Dates
    r"^Last\s+modified\s+on",
    r"^URL:|^URI:",  # URLs as headers
    r"^\.{3,}|^-{3,}|^_{3,}|^\*{3,}",  # Separator lines
    r"^Confidential|^Draft|^Internal",
    r"^Document\s+ID:",
    r"^Revision\s*\d+",
]

# Patterns for lines that are likely repeating footers/headers
REPEATING_FOOTER_PATTERNS = [
    r"^Related\s+Documents?",
    r"^See\s+Also",
    r"^Next\s+Steps?",
    r"^Appendix\s+[A-Z]",
    r"^For\s+more\s+information",
    r"^Contact\s+us",
    r"^Questions\?",
    r"^Feedback",
    r"^www\.|^http",
]


def _is_likely_header_footer(line: str) -> bool:
    """Check if a line is likely a header or footer.

    Header/footer lines are typically:
    - Very short
    - Page numbers or document metadata
    - Navigation/section markers
    - URLs or document references
    """
    stripped = line.strip()

    # Very short lines without punctuation (likely page numbers or headers)
    if len(stripped) < 5 and not any(c in stripped for c in ".!?:"):
        return True

    # Check against boilerplate patterns
    if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in PDF_BOILERPLATE_PATTERNS):
        return True

    # Check against repeating patterns
    if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in REPEATING_FOOTER_PATTERNS):
        return True

    return False


def _clean_pdf_text(text: str) -> str:
    """Clean extracted PDF text by removing boilerplate.

    - Removes page numbers and dates
    - Removes repeating headers/footers
    - Normalises whitespace
    - Removes excessive blank lines

    Args:
        text: Raw text extracted from PDF

    Returns:
        Cleaned text
    """
    lines = text.split("\n")
    cleaned_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Keep empty lines for structure (but will collapse later)
        if not stripped:
            cleaned_lines.append("")
            continue

        # Check if this line is boilerplate
        if _is_likely_header_footer(stripped):
            # Don't add this line
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Collapse multiple blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def extract_pdf_metadata(path: str) -> Dict[str, Any]:
    """Extract metadata from PDF file (title, author, year, keywords).

    Attempts to extract structured metadata from PDF properties.
    Falls back to extracting from first page content if metadata unavailable.

    Args:
        path: Path to PDF file

    Returns:
        Dict with keys: title, author, year, keywords, subject
        Values may be None if not extractable.

    Example:
        >>> meta = extract_pdf_metadata("thesis.pdf")
        >>> meta['title']
        'New Insight Contributing to Human Knowledge'
        >>> meta['author']
        'FirstName Surname'
        >>> meta['year']
        '2026'
    """
    import re
    from pathlib import Path

    metadata = {
        "title": None,
        "author": None,
        "year": None,
        "keywords": None,
        "subject": None,
    }

    try:
        reader = PdfReader(path)
        pdf_meta = reader.metadata

        if pdf_meta:
            # Extract title
            if pdf_meta.get("/Title"):
                metadata["title"] = str(pdf_meta["/Title"]).strip()

            # Extract author
            if pdf_meta.get("/Author"):
                metadata["author"] = str(pdf_meta["/Author"]).strip()

            # Extract subject/keywords
            if pdf_meta.get("/Subject"):
                metadata["subject"] = str(pdf_meta["/Subject"]).strip()
            if pdf_meta.get("/Keywords"):
                metadata["keywords"] = str(pdf_meta["/Keywords"]).strip()

            # Extract year from creation/modification date
            for date_field in ["/CreationDate", "/ModDate"]:
                if pdf_meta.get(date_field):
                    date_str = str(pdf_meta[date_field])
                    # PDF dates often in format: D:20190315... or 2019-03-15
                    year_match = re.search(r"(20\d{2})", date_str)
                    if year_match:
                        metadata["year"] = year_match.group(1)
                        break

        # Fallback: Extract from first page if metadata unavailable
        if not metadata["title"] and reader.pages:
            first_page_text = reader.pages[0].extract_text()
            if first_page_text:
                lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]
                # First significant line is often the title
                for line in lines[:5]:  # Check first 5 lines
                    if len(line) > 10 and not line.startswith(("Page ", "http")):
                        metadata["title"] = line[:200]  # Cap title length
                        break

                # Look for author patterns ("By ", "Author: ", etc.)
                for line in lines[:10]:
                    if re.match(r"^(By|Author|Authors?):\s*", line, re.IGNORECASE):
                        author = re.sub(r"^(By|Author|Authors?):\s*", "", line, flags=re.IGNORECASE)
                        metadata["author"] = author[:100]
                        break

        # Final fallback: use filename as title
        if not metadata["title"]:
            metadata["title"] = Path(path).stem

    except Exception as e:
        import logging

        logging.warning(f"Failed to extract PDF metadata from {path}: {e}")
        # Return partial metadata if available
        if not metadata["title"]:
            metadata["title"] = Path(path).stem

    return metadata


@retry_with_backoff(
    max_retries=3,
    initial_delay=1.0,
    transient_types=(IOError, MemoryError, TimeoutError),
    operation_name="parse_pdf",
)
def extract_text_from_pdf(path: str) -> str:
    """Extract and clean text from PDF documents.

    Attempts multiple extraction strategies:
    1. Standard extract_text() method
    2. Fallback to extract_text_with_layout() for problematic PDFs

    Removes:
    - Page numbers and page markers
    - Document metadata (version, date, copyright)
    - Repeating headers and footers
    - Excessive whitespace

    Args:
        path: Path to PDF file

    Returns:
        Cleaned text content

    Example:
        >>> text = extract_text_from_pdf("document.pdf")
        >>> # Text is cleaned, with headers/footers and page numbers removed
    """
    import logging

    reader = PdfReader(path)
    pages_text = []
    extraction_method = "standard"

    # Try standard extraction first
    for page_num, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages_text.append(page_text)
        except Exception as e:
            logging.warning(f"Failed to extract text from page {page_num} (standard): {e}")
            continue

    # If standard extraction yielded nothing, try alternative method
    if not pages_text or all(not p.strip() for p in pages_text):
        logging.info(f"Standard extraction failed for {path}, attempting fallback method...")
        pages_text = []
        extraction_method = "fallback"

        for page_num, page in enumerate(reader.pages):
            try:
                # Try extraction with layout mode
                page_text = page.extract_text(layout_mode="plain")
                if not page_text:
                    # Try with default layout
                    page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages_text.append(page_text)
            except Exception as e:
                logging.warning(f"Failed to extract text from page {page_num} (fallback): {e}")
                continue

    if not pages_text:
        logging.warning(f"No text could be extracted from {path} using any method")
        return ""

    # Join pages with page break marker for separation
    text = "\n\n".join(pages_text)

    # Clean the combined text
    text = _clean_pdf_text(text)

    if not text.strip():
        logging.warning(
            f"PDF {path} extracted {len(pages_text)} pages but produced no usable text after cleaning"
        )

    return text


def _is_table_of_contents_entry(line: str) -> bool:
    """Detect if a line is from a Table of Contents.

    TOC entries have characteristic patterns:
    - Text (chapter/section heading)
    - Multiple dots (leader dots): ....... or ......
    - Page number: digits, possibly preceded by 'page'
    - May span multiple spaces for alignment

    Examples:
        "Chapter 1: Introduction ........................... 5"
        "2.3. Methodology ............................... 45"
        "References ............................ 156"
        "Alignment to this Study ............................. page 120"

    Args:
        line: Text line to check

    Returns:
        True if line appears to be a TOC entry
    """
    stripped = line.strip()

    # TOC entries typically have 3+ dots and a page number
    # Pattern: text ... digits or text ... page digits
    toc_pattern = r"^.{3,}\.{3,}\.?\s*(?:page\s*)?\d+\s*(?:T)?$"
    if re.match(toc_pattern, stripped, re.IGNORECASE):
        return True

    # Also match if line ends with only dots and page (less strict)
    # Catches cases like "Chapter 1: Introduction ........................... 5"
    if re.search(r"\.{4,}\s*?\d{1,3}\s*$", stripped):
        return True

    return False


def _classify_document_section(section_title: str) -> str:
    """Classify whether a section is pre-matter, main matter, or post-matter.

    Pre-matter: Abstract, Acknowledgements, Dedications, Glossary, Foreword, Preface, etc.
    Main matter: Chapter 1-9, Introduction (when not alone), Methodology, Results, etc.
    Post-matter: References, Bibliography, Appendix, Index, etc.

    TODO: remove, should not be required if sequencing of chunks is correct.

    Args:
        section_title: Section title/heading text

    Returns:
        'pre-matter', 'main-matter', or 'post-matter'
    """
    title_lower = section_title.lower()

    # Pre-matter sections (come before main chapters)
    pre_matter_keywords = [
        "abstract",
        "acknowledgement",
        "dedication",
        "glossary",
        "foreword",
        "preface",
        "prologue",
        "table of contents",
        "list of figures",
        "list of tables",
    ]

    # Post-matter sections (come after main chapters)
    post_matter_keywords = [
        "reference",
        "bibliography",
        "appendix",
        "index",
        "epilogue",
        "colophon",
        "statement of contribution",
        "author note",
    ]

    for keyword in pre_matter_keywords:
        if keyword in title_lower:
            return "pre-matter"

    for keyword in post_matter_keywords:
        if keyword in title_lower:
            return "post-matter"

    return "main-matter"


def extract_structure_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract chapter and section structure from document text.

    Identifies hierarchical document structure by detecting:
    - Chapter headings (Chapter 1, Chapter One, CHAPTER I, etc.)
    - Major section headings (Introduction, Methodology, Results, etc.)
    - Subsection markers (numbered sections like 2.1, 2.1.1, etc.)
    - Heading paths for context
    - Filters out Table of Contents entries

    Args:
        text: Full document text

    Returns:
        List of structure entries, each containing:
            - start_pos: Character position where section starts
            - end_pos: Character position where section ends (or None for last section)
            - chapter: Chapter label (e.g., "Chapter 1", "Introduction")
            - section_title: Section heading text
            - heading_path: Full hierarchical path (e.g., "Chapter 1 > Methods > Data Collection")
            - level: Heading level (0=chapter, 1=major section, 2=subsection)

    Example:
        >>> structure = extract_structure_from_text(pdf_text)
        >>> for section in structure:
        ...     print(f"{section['chapter']}: {section['heading_path']}")
        Chapter 1: Chapter 1 > Introduction
        Chapter 2: Chapter 2 > Literature Review > Theoretical Framework
    """
    structure = []

    # Patterns for detecting chapter headings (case-insensitive)
    chapter_patterns = [
        # "Chapter 1", "Chapter One", "CHAPTER 1:"
        r"^(?:chapter|ch\.?)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)[\s:.\-—]*(.*?)$",
        # Roman numerals: "I. Introduction", "II - Methods"
        r"^([IVX]{1,5})[\s:.\-—]+(.*?)$",
        # Numbered chapters: "1. Introduction"
        r"^(\d{1,2})[\s:.\-—]+(.*?)$",
    ]

    # Standard academic section patterns
    standard_sections = [
        "abstract",
        "acknowledgements",
        "introduction",
        "background",
        "literature review",
        "theoretical framework",
        "conceptual framework",
        "methodology",
        "methods",
        "research design",
        "approach",
        "results",
        "findings",
        "analysis",
        "data analysis",
        "discussion",
        "implications",
        "conclusion",
        "conclusions",
        "recommendations",
        "future work",
        "limitations",
        "references",
        "bibliography",
        "appendix",
        "appendices",
    ]

    # Subsection patterns (e.g., "2.1 Data Collection", "2.1.1 Sampling")
    subsection_pattern = r"^(\d+(?:\.\d+)+)[\s:.\-—]+(.*?)$"

    lines = text.split("\n")
    current_chapter = None
    current_section = None
    hierarchy: List[Tuple[int, str]] = []  # (level, title)

    for line_no, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            continue

        # FILTER: Skip Table of Contents entries (have dots and page numbers)
        # Examples: "Chapter 1: Introduction ........................ 5"
        #           "2.3. Methodology ............................ page 45"
        if _is_table_of_contents_entry(stripped):
            continue

        # Calculate character position
        char_pos = sum(len(lines[i]) + 1 for i in range(line_no))  # +1 for newline

        # Check for chapter headings
        chapter_match = None
        for pattern in chapter_patterns:
            match = re.match(pattern, stripped, re.IGNORECASE | re.MULTILINE)
            if match:
                chapter_match = match
                break

        if chapter_match:
            # Extract chapter number/label and title
            chapter_num = chapter_match.group(1)
            chapter_title = (
                chapter_match.group(2).strip() if len(chapter_match.groups()) > 1 else ""
            )

            # Normalise chapter number (convert words to digits, roman to arabic)
            chapter_label = _normalise_chapter_number(chapter_num, chapter_title)

            # Close previous section if exists
            if structure:
                structure[-1]["end_pos"] = char_pos

            current_chapter = chapter_label
            full_title = f"{chapter_label}: {chapter_title}" if chapter_title else chapter_label

            structure.append(
                {
                    "start_pos": char_pos,
                    "end_pos": None,  # Will be set when next section starts
                    "chapter": chapter_label,
                    "section_title": full_title,
                    "heading_path": chapter_label,
                    "level": 0,  # Chapter level
                }
            )

            hierarchy = [(0, chapter_label)]
            current_section = full_title
            continue

        # Check for subsections (e.g., "2.1", "2.1.1")
        subsection_match = re.match(subsection_pattern, stripped, re.MULTILINE)
        if subsection_match:
            section_num = subsection_match.group(1)
            section_title = subsection_match.group(2).strip()
            level = section_num.count(".") + 1  # 2.1 = level 1, 2.1.1 = level 2

            # Close previous section
            if structure:
                structure[-1]["end_pos"] = char_pos

            # Update hierarchy (remove deeper levels)
            hierarchy = [(l, t) for l, t in hierarchy if l < level]
            hierarchy.append((level, section_title))

            heading_path = " > ".join(t for _, t in hierarchy)

            structure.append(
                {
                    "start_pos": char_pos,
                    "end_pos": None,
                    "chapter": current_chapter or "Unknown",
                    "section_title": section_title,
                    "heading_path": heading_path,
                    "level": level,
                }
            )
            continue

        # Check for standard academic sections (if they look like headings)
        # Heuristic: line is short, matches a standard section, and next line is blank or content
        if len(stripped) < 80:  # Headings are typically short
            section_lower = stripped.lower()
            matched_section = None

            for std_section in standard_sections:
                if std_section in section_lower:
                    matched_section = stripped
                    break

            if matched_section:
                # Verify it looks like a heading (check if next line is content or blank)
                if line_no + 1 < len(lines):
                    next_line = lines[line_no + 1].strip()
                    # If next line is blank or starts with lower-case (continuation), likely a heading
                    if not next_line or (next_line and next_line[0].islower()):
                        # Close previous section
                        if structure:
                            structure[-1]["end_pos"] = char_pos

                        # Update hierarchy
                        if current_chapter:
                            hierarchy = [(0, current_chapter)]
                        hierarchy.append((1, matched_section))
                        heading_path = " > ".join(t for _, t in hierarchy)

                        structure.append(
                            {
                                "start_pos": char_pos,
                                "end_pos": None,
                                "chapter": current_chapter or matched_section,
                                "section_title": matched_section,
                                "heading_path": heading_path,
                                "level": 1,
                            }
                        )
                        current_section = matched_section

    # Set end_pos for final section to end of text
    if structure:
        structure[-1]["end_pos"] = len(text)

    return structure


def _normalise_chapter_number(chapter_num: str, title: str = "") -> str:
    """Normalise chapter number to consistent format.

    Args:
        chapter_num: Raw chapter number/label (e.g., "1", "One", "I", "Introduction")
        title: Optional chapter title for context

    Returns:
        Normalised chapter label (e.g., "Chapter 1", "Introduction")
    """
    # Word to number mapping
    word_to_num = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }

    # Roman to arabic
    roman_to_num = {
        "I": "1",
        "II": "2",
        "III": "3",
        "IV": "4",
        "V": "5",
        "VI": "6",
        "VII": "7",
        "VIII": "8",
        "IX": "9",
        "X": "10",
    }

    chapter_lower = chapter_num.strip().lower()
    chapter_upper = chapter_num.strip().upper()

    # Check if it's a word number
    if chapter_lower in word_to_num:
        return f"Chapter {word_to_num[chapter_lower]}"

    # Check if it's a roman numeral
    if chapter_upper in roman_to_num:
        return f"Chapter {roman_to_num[chapter_upper]}"

    # Check if it's already a digit
    if chapter_num.strip().isdigit():
        return f"Chapter {chapter_num.strip()}"

    # Otherwise, use the title if it looks like a standard section
    if title:
        return title

    # Fallback
    return f"Chapter {chapter_num}"


def map_text_to_structure(
    text: str, structure: List[Dict[str, Any]], chunk_start: int, chunk_end: int
) -> Dict[str, Optional[str]]:
    """Map a text chunk to its structural metadata.

    Determines which chapter/section a chunk belongs to based on character positions.

    Args:
        text: Full document text
        structure: List of structure entries from extract_structure_from_text()
        chunk_start: Starting character position of chunk
        chunk_end: Ending character position of chunk

    Returns:
        Dict with chapter, section_title, and heading_path metadata
    """
    if not structure:
        return {"chapter": None, "section_title": None, "heading_path": None}

    # Find which section this chunk falls into
    # A chunk belongs to a section if it overlaps with the section's range
    for section in structure:
        section_start = section["start_pos"]
        section_end = section["end_pos"] or len(text)

        # Check if chunk overlaps with this section
        if not (chunk_end < section_start or chunk_start >= section_end):
            return {
                "chapter": section["chapter"],
                "section_title": section["section_title"],
                "heading_path": section["heading_path"],
            }

    # No matching section found
    return {"chapter": None, "section_title": None, "heading_path": None}
