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
    - alternative extenstion: docling-hierarchical-pdf (https://github.com/krrome/docling-hierarchical-pdf) for extracting hierarchical structure including chapters and sections
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

import os
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


def _looks_like_numbered_heading(title: str) -> bool:
    """Return whether text after a chapter number resembles a heading, not prose."""
    if not title:
        return True

    words = title.split()
    if len(title) > 120 or len(words) > 14 or title.endswith((".", "!", "?", ";")):
        return False

    return title[0].isalpha() and title[0].isupper()


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

    metadata: Dict[str, Any] = {
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


def _convert_pdf_with_docling(path: str) -> Any | None:
    """Convert a PDF to a Docling document when the optional dependency is installed.

    Docling preserves heading hierarchy, reading order, and table structure,
    allowing downstream academic assessment to map chunks to source sections.
    Native PDF text is preferred by default; OCR can be enabled for scans.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            EasyOcrOptions,
            OcrMode,
            PdfPipelineOptions,
            RapidOcrOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError:
        return None

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = _get_docling_bool_setting(
        "DOCLING_GENERATE_PICTURE_IMAGES", True
    )
    enable_ocr = _get_docling_bool_setting("DOCLING_ENABLE_OCR", False)
    pipeline_options.do_ocr = enable_ocr
    pipeline_options.force_backend_text = _get_docling_bool_setting(
        "DOCLING_FORCE_BACKEND_TEXT", True
    )
    pipeline_options.do_table_structure = _get_docling_bool_setting("DOCLING_TABLE_STRUCTURE", True)
    pipeline_options.heading_hierarchy_options.enabled = _get_docling_bool_setting(
        "DOCLING_HEADING_HIERARCHY", True
    )
    if enable_ocr:
        ocr_language = os.getenv("DOCLING_OCR_LANGUAGE", "en").strip() or "en"
        ocr_engine = os.getenv("DOCLING_OCR_ENGINE", "rapidocr").strip().lower()
        if ocr_engine == "easyocr":
            pipeline_options.ocr_options = EasyOcrOptions(
                lang=[ocr_language],
                mode=OcrMode.FULL_PAGE,
                use_gpu=None,
            )
        elif ocr_engine == "rapidocr":
            pipeline_options.ocr_options = RapidOcrOptions(
                lang=[ocr_language],
                mode=OcrMode.FULL_PAGE,
            )
        else:
            raise ValueError(
                "DOCLING_OCR_ENGINE must be 'rapidocr' or 'easyocr', " f"got {ocr_engine!r}"
            )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    result = converter.convert(path)
    return result.document


def _extract_text_with_docling(path: str) -> Optional[str]:
    """Convert a PDF to structured Markdown using Docling when installed."""
    document = _convert_pdf_with_docling(path)
    if document is None:
        return None
    markdown = document.export_to_markdown()
    return markdown.strip() or None


def extract_figures_from_docling_document(document: Any) -> List[Dict[str, Any]]:
    """Extract locally available figure images and source metadata from Docling output.

    Figure images remain in memory as PIL objects; this helper does not write them
    to disk or invoke a vision model.

    Args:
        document: A Docling document object from which to extract figures.

    Returns:
        A list of dictionaries, each containing the figure image (as a PIL object),
        bounding box information, caption, and alternative text.
    """
    figures: List[Dict[str, Any]] = []
    for item, _ in document.iterate_items(traverse_pictures=True):
        label = getattr(item, "label", None)
        label = getattr(label, "value", label)
        if label not in {"picture", "chart"}:
            continue

        provenance = getattr(item, "prov", []) or []
        primary_provenance = provenance[0] if provenance else None
        bbox = getattr(primary_provenance, "bbox", None)
        bbox_data = None
        if bbox is not None:
            bbox_data = {
                "left": getattr(bbox, "l", None),
                "top": getattr(bbox, "t", None),
                "right": getattr(bbox, "r", None),
                "bottom": getattr(bbox, "b", None),
                "coordinate_origin": str(getattr(bbox, "coord_origin", "")),
            }

        caption = ""
        caption_reader = getattr(item, "caption_text", None)
        if callable(caption_reader):
            try:
                caption = str(caption_reader(document) or "").strip()
            except (AttributeError, TypeError, ValueError):
                caption = ""

        metadata = getattr(item, "meta", None)
        alt_text = str(getattr(metadata, "description", "") or "").strip()
        if not alt_text:
            for annotation in getattr(item, "annotations", []) or []:
                if getattr(annotation, "kind", None) == "description":
                    alt_text = str(getattr(annotation, "text", "") or "").strip()
                    if alt_text:
                        break

        caption_number_match = re.match(
            r"^\s*(?:fig(?:ure)?s?)\.?\s+((?:[A-Z]\.)?\d+(?:\.\d+)*)\b",
            caption,
            re.IGNORECASE,
        )

        image = None
        image_reader = getattr(item, "get_image", None)
        if callable(image_reader):
            try:
                image = image_reader(document)
            except (AttributeError, TypeError, ValueError):
                image = None

        figures.append(
            {
                "figure_number": len(figures) + 1,
                "kind": str(label),
                "page_number": getattr(primary_provenance, "page_no", None),
                "bbox": bbox_data,
                "caption": caption,
                "caption_number": caption_number_match.group(1) if caption_number_match else None,
                "alt_text": alt_text,
                "image": image,
            }
        )
    return figures


def extract_pdf_text_and_figures(path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract structured text and figure assets from one local Docling conversion.

    Args:
        path: Path to the PDF file to be processed.

    Returns:
        A tuple containing:
        - The extracted and cleaned text as a string.
        - A list of dictionaries representing the figures found in the document.
    """
    import logging

    try:
        document = _convert_pdf_with_docling(path)
    except Exception as exc:
        logging.warning("Docling figure conversion failed for %s: %s", path, exc)
        document = None

    if document is None:
        return extract_text_from_pdf(path, _skip_docling=True), []

    markdown = str(document.export_to_markdown() or "").strip()
    figures = extract_figures_from_docling_document(document)
    structure = extract_structure_from_text(markdown)
    next_caption_search_start = 0
    for figure in figures:
        caption = str(figure.get("caption") or "").strip()
        if not caption:
            continue
        caption_start = markdown.casefold().find(caption.casefold(), next_caption_search_start)
        if caption_start < 0:
            continue
        caption_end = caption_start + len(caption)
        location = map_text_to_structure(markdown, structure, caption_start, caption_end)
        figure["chapter"] = location.get("chapter")
        figure["section_title"] = location.get("section_title")
        figure["heading_path"] = location.get("heading_path")
        figure["source_start"] = caption_start
        figure["source_end"] = caption_end
        next_caption_search_start = caption_end
    if not markdown:
        markdown = extract_text_from_pdf(path, _skip_docling=True)
    return markdown, figures


def _get_docling_bool_setting(name: str, default: bool) -> bool:
    """Read an optional Docling boolean setting from the environment.

    Args:
        name: Name of the environment variable to read
        default: Default boolean value if the environment variable is not set

    Returns:
        bool: The boolean value of the environment variable or the default
    """
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@retry_with_backoff(
    max_retries=3,
    initial_delay=1.0,
    transient_types=(IOError, MemoryError, TimeoutError),
    operation_name="parse_pdf",
)
def extract_text_from_pdf(path: str, *, _skip_docling: bool = False) -> str:
    """Extract and clean text from PDF documents.

    Attempts multiple extraction strategies:
    1. Docling structured Markdown with configurable OCR and layout analysis, when installed
    2. Standard pypdf extract_text() method
    3. pypdf layout-mode fallback for problematic PDFs

    Removes:
    - Page numbers and page markers
    - Document metadata (version, date, copyright)
    - Repeating headers and footers
    - Excessive whitespace

    Args:
        path: Path to PDF file
        _skip_docling: Whether to skip using the Docling extraction method (default: False)

    Returns:
        Cleaned text content

    Example:
        >>> text = extract_text_from_pdf("document.pdf")
        >>> # Text is cleaned, with headers/footers and page numbers removed
    """
    import logging

    if not _skip_docling:
        try:
            docling_text = _extract_text_with_docling(path)
            if docling_text:
                logging.info("Extracted PDF with Docling layout and OCR pipeline: %s", path)
                return docling_text
        except Exception as exc:
            logging.warning(
                "Docling conversion failed for %s; falling back to pypdf: %s", path, exc
            )

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


def _is_wrapped_table_of_contents_entry(line: str, following_line: str) -> bool:
    """Detect a ToC title whose leader dots and page number are on the next line.

    Args:
        line: The line containing the potential chapter title.
        following_line: The line containing the leader dots and page number.

    Returns:
        True if the line and following_line together form a wrapped TOC entry.
    """
    title_pattern = (
        r"^\s*(?:(?:chapter|ch\.?)\s+)?"
        r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|[IVX]{1,5})"
        r"(?:[.)]|[\s:.-]+)\s*\S.+$"
    )
    page_pattern = r"^\s*\.{2,}\s*(?:page\s*)?\d+\s*(?:T)?$"
    return bool(
        re.match(title_pattern, line, re.IGNORECASE)
        and re.match(page_pattern, following_line, re.IGNORECASE)
    )


def _extract_toc_chapter_entries(text: str) -> Dict[str, str]:
    """Return canonical chapter titles keyed by normalised chapter labels.

    Args:
        text: Full text of the document to extract ToC entries from.

    Returns:
        Dictionary mapping normalised chapter labels to their canonical titles.
    """
    chapter_entry_pattern = re.compile(
        r"^\s*(?:chapter|ch\.?)\s+"
        r"(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten|[IVX]{1,5})"
        r"[\s:.-]*(?P<title>.*?)\s*\.{3,}\s*(?:page\s*)?\d+\s*(?:T)?$",
        re.IGNORECASE,
    )
    numbered_entry_pattern = re.compile(
        r"^\s*(?P<number>\d{1,2})[.)]\s*(?P<title>.+?)\s*" r"\.{2,}\s*(?:page\s*)?\d+\s*(?:T)?$",
        re.IGNORECASE,
    )
    wrapped_heading_pattern = re.compile(
        r"^\s*(?:(?:chapter|ch\.?)\s+)?"
        r"(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten|[IVX]{1,5})"
        r"(?:[.)]|[\s:.-]+)\s*(?P<title>\S.*?)\s*$",
        re.IGNORECASE,
    )
    wrapped_page_pattern = re.compile(r"^\s*\.{2,}\s*(?:page\s*)?\d+\s*(?:T)?$", re.IGNORECASE)
    chapters: Dict[str, str] = {}
    lines = text.splitlines()
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index].strip()
        if line.startswith("|") and line.endswith("|"):
            line = line[1:-1].strip()
        match = chapter_entry_pattern.match(line) or numbered_entry_pattern.match(line)
        if (
            match is None
            and line_index + 1 < len(lines)
            and wrapped_page_pattern.match(lines[line_index + 1])
        ):
            match = wrapped_heading_pattern.match(line)
            if match:
                line_index += 1
        if not match:
            line_index += 1
            continue
        title = re.sub(r"\s+", " ", match.group("title")).strip(" .:-—")
        chapter_label = _normalise_chapter_number(match.group("number"), title)
        chapters[chapter_label] = title
        line_index += 1
    return chapters


def _extract_outline_toc_chapters(text: str) -> Tuple[Dict[str, str], set[int]]:
    """Extract numbered chapter titles from an explicit page-less ToC block.

    Args:
        text: The text content of the ToC block.

    Returns:
        A tuple containing:
        - A dictionary of chapter labels and titles.
        - A set of line numbers corresponding to the ToC entries.
    """
    chapter_pattern = re.compile(
        r"^\s*(?:chapter|ch\.?)\s+"
        r"(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten|[IVX]{1,5})"
        r"[\s:.-]+(?P<title>\S.*?)\s*$",
        re.IGNORECASE,
    )
    numbered_pattern = re.compile(
        r"^\s*(?P<number>\d{1,2})[.)]\s*(?P<title>\S.*?)\s*$",
        re.IGNORECASE,
    )
    lines = text.splitlines()
    chapters: Dict[str, str] = {}
    toc_line_numbers: set[int] = set()
    header_pattern = re.compile(
        r"^\s*#*\s*(?:table\s+of\s+contents|contents)\s*#*\s*$", re.IGNORECASE
    )

    for header_index, line in enumerate(lines):
        if not header_pattern.match(line):
            continue
        toc_line_numbers.add(header_index)
        found_entry = False
        seen_content = False
        for line_index in range(header_index + 1, len(lines)):
            candidate_line = lines[line_index].strip()
            if not candidate_line:
                if seen_content:
                    break
                continue
            seen_content = True
            if _is_table_of_contents_entry(candidate_line):
                continue
            match = chapter_pattern.match(candidate_line) or numbered_pattern.match(candidate_line)
            if not match:
                continue
            title = re.sub(r"\s+", " ", match.group("title")).strip(" .:-—")
            chapter_label = _normalise_chapter_number(match.group("number"), title)
            chapters[chapter_label] = title
            toc_line_numbers.add(line_index)
            found_entry = True

    return chapters, toc_line_numbers


def _extend_outline_toc_chapters(
    text: str, chapters: Dict[str, str], toc_line_numbers: set[int]
) -> Tuple[Dict[str, str], set[int]]:
    """Include unnumbered outline rows and wrapped chapter-title continuations.

    Args:
        text: The full document text.
        chapters: Chapter labels mapped to their ToC titles.
        toc_line_numbers: Source line numbers already identified as ToC entries.

    Returns:
        The updated chapter map and ToC line numbers.
    """
    lines = text.splitlines()
    header_pattern = re.compile(
        r"^\s*#*\s*(?:table\s+of\s+contents|contents)\s*#*\s*$", re.IGNORECASE
    )
    chapter_pattern = re.compile(
        r"^\s*(?:chapter|ch\.?)\s+"
        r"(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten|[IVX]{1,5})"
        r"[\s:.-]*(?P<title>.*?)\s*$",
        re.IGNORECASE,
    )
    numbered_pattern = re.compile(r"^\s*(?P<number>\d{1,2})[.)]\s*(?P<title>.+?)\s*$")
    page_suffix_pattern = re.compile(r"\s+\.{2,}\s*(?:page\s*)?\d+\s*(?:T)?$", re.IGNORECASE)
    page_row_pattern = re.compile(r"^\s*\.{2,}\s*(?:page\s*)?\d+\s*(?:T)?$", re.IGNORECASE)
    excluded_titles = {
        "abstract",
        "acknowledgement",
        "acknowledgements",
        "contents",
        "table of contents",
        "list of figures",
        "list of tables",
        "list of abbreviations",
        "references",
        "bibliography",
        "appendix",
        "appendices",
        "index",
        "glossary",
        "dedication",
        "foreword",
        "preface",
    }

    for header_index, header_line in enumerate(lines):
        if not header_pattern.match(header_line):
            continue
        seen_content = False
        line_index = header_index + 1
        while line_index < len(lines):
            raw_line = lines[line_index]
            candidate_line = raw_line.strip()
            if not candidate_line:
                if seen_content:
                    break
                line_index += 1
                continue
            seen_content = True
            toc_line_numbers.add(line_index)
            is_markdown_table_row = candidate_line.startswith("|") and candidate_line.endswith("|")
            if is_markdown_table_row:
                candidate_line = candidate_line[1:-1].strip()

            if page_row_pattern.match(candidate_line):
                line_index += 1
                continue

            has_page_suffix = bool(page_suffix_pattern.search(candidate_line))
            title_line = page_suffix_pattern.sub("", candidate_line).strip(" .:-—")
            match = chapter_pattern.match(title_line) or numbered_pattern.match(title_line)
            if match:
                title = re.sub(r"\s+", " ", match.group("title")).strip(" .:-—")
                continuation_parts = []
                consumed_indices = []
                page_row_found = has_page_suffix
                scan_index = line_index + 1
                while not page_row_found and scan_index < min(len(lines), line_index + 5):
                    continuation = lines[scan_index].strip()
                    if not continuation:
                        break
                    if page_row_pattern.match(continuation):
                        consumed_indices.append(scan_index)
                        page_row_found = True
                        break
                    continuation_has_page = bool(page_suffix_pattern.search(continuation))
                    continuation_title = page_suffix_pattern.sub("", continuation).strip(" .:-—")
                    if chapter_pattern.match(continuation_title) or numbered_pattern.match(
                        continuation_title
                    ):
                        break
                    if continuation_title:
                        continuation_parts.append(continuation_title)
                        consumed_indices.append(scan_index)
                    if continuation_has_page:
                        page_row_found = True
                        break
                    scan_index += 1
                if page_row_found and continuation_parts:
                    title = " ".join(part for part in (title, *continuation_parts) if part)
                    toc_line_numbers.update(consumed_indices)
                    line_index = scan_index + 1
                else:
                    line_index += 1
                chapter_label = _normalise_chapter_number(match.group("number"), title)
                if title:
                    chapters[chapter_label] = title
                continue

            if (
                is_markdown_table_row
                or raw_line[:1].isspace()
                or not title_line
                or (_is_table_of_contents_entry(candidate_line) and not has_page_suffix)
                or title_line.casefold() in excluded_titles
                or title_line.casefold().startswith(("appendix ", "appendices ", "list of "))
            ):
                line_index += 1
                continue

            synthetic_number = 1
            while f"Chapter {synthetic_number}" in chapters:
                synthetic_number += 1
            chapters[f"Chapter {synthetic_number}"] = re.sub(r"\s+", " ", title_line)
            line_index += 1

    return chapters, toc_line_numbers


def _match_toc_chapter_title(heading: str, toc_chapters: Dict[str, str]) -> Optional[str]:
    """Return the ToC chapter label for an exact body-heading title match.

    Args:
        heading: The body-heading title to match.
        toc_chapters: Dictionary of ToC chapter labels and titles.

    Returns:
        The matching ToC chapter label if found, otherwise None.
    """

    def _normalise(value: str) -> str:
        value = re.sub(r"^#+\s*", "", value.strip())
        value = re.sub(r"\s+", " ", value)
        return value.strip(" .:-—").casefold()

    normalised_heading = _normalise(heading)
    for chapter_label, title in toc_chapters.items():
        if normalised_heading in {
            _normalise(title),
            _normalise(chapter_label),
            _normalise(f"{chapter_label}: {title}"),
        }:
            return chapter_label
    return None


def _find_unlisted_chapter_candidates(text: str, toc_chapters: Dict[str, str]) -> List[str]:
    """Find explicit chapter headings that the ToC-authoritative parser rejects.

    Args:
        text: Full text of the document containing the ToC.
        toc_chapters: Dictionary of ToC chapter labels and titles.

    Returns:
        A list of candidate chapter headings not listed in the ToC.
    """
    chapter_pattern = re.compile(
        r"^\s*#*\s*(?:chapter|ch\.?)\s+"
        r"(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten|[IVX]{1,5})"
        r"[\s:.-]*(?P<title>.*?)\s*$",
        re.IGNORECASE,
    )
    candidates = []
    lines = text.splitlines()
    for line_index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _is_table_of_contents_entry(stripped):
            continue
        if line_index + 1 < len(lines) and _is_wrapped_table_of_contents_entry(
            stripped, lines[line_index + 1].strip()
        ):
            continue
        match = chapter_pattern.match(stripped)
        if not match:
            continue
        title = match.group("title").strip()
        if title and not _looks_like_numbered_heading(title):
            continue
        chapter_label = _normalise_chapter_number(match.group("number"), title)
        if chapter_label in toc_chapters:
            continue
        candidate = f"{chapter_label}: {title}" if title else chapter_label
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def validate_structure_against_toc(
    text: str,
    structure: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Report chapter coverage and order against explicit ToC chapter entries.

    Args:
        text: Full text of the document containing the ToC.
        structure: List of chapter entries with "chapter" and "level".

    Returns:
        Dictionary containing coverage metrics, missing chapters, unexpected chapters,
        and order validation against the ToC.
    """
    toc_chapters = _extract_toc_chapter_entries(text)
    outline_chapters, _ = _extract_outline_toc_chapters(text)
    outline_chapters, _ = _extend_outline_toc_chapters(text, outline_chapters, set())
    toc_chapters.update(outline_chapters)
    expected_chapters = list(toc_chapters)
    if not expected_chapters:
        return {
            "toc_present": False,
            "expected_chapters": 0,
            "matched_chapters": 0,
            "coverage": None,
            "missing_chapters": [],
            "unexpected_chapters": [],
            "order_matches": None,
        }

    detected_chapters = []
    unexpected_chapters = _find_unlisted_chapter_candidates(text, toc_chapters)
    for entry in structure:
        if entry.get("level") != 0:
            continue
        chapter = str(entry.get("chapter") or "")
        if chapter in toc_chapters:
            if chapter not in detected_chapters:
                detected_chapters.append(chapter)
        elif chapter.lower().startswith("chapter ") and chapter not in unexpected_chapters:
            unexpected_chapters.append(chapter)

    matched_chapters = len(set(detected_chapters) & set(expected_chapters))
    missing_chapters = [
        chapter for chapter in expected_chapters if chapter not in detected_chapters
    ]
    return {
        "toc_present": True,
        "expected_chapters": len(expected_chapters),
        "matched_chapters": matched_chapters,
        "coverage": matched_chapters / len(expected_chapters),
        "missing_chapters": missing_chapters,
        "unexpected_chapters": unexpected_chapters,
        "detected_order": detected_chapters,
        "expected_order": expected_chapters,
        "order_matches": detected_chapters == expected_chapters,
    }


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
    structure: List[Dict[str, Any]] = []
    toc_chapters = _extract_toc_chapter_entries(text)
    outline_chapters, toc_outline_line_numbers = _extract_outline_toc_chapters(text)
    outline_chapters, toc_outline_line_numbers = _extend_outline_toc_chapters(
        text, outline_chapters, toc_outline_line_numbers
    )
    toc_chapters.update(outline_chapters)

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
        if line_no in toc_outline_line_numbers:
            continue
        if not stripped or len(stripped) < 3:
            continue

        if line_no + 1 < len(lines) and _is_wrapped_table_of_contents_entry(
            stripped, lines[line_no + 1].strip()
        ):
            continue

        # FILTER: Skip Table of Contents entries (have dots and page numbers)
        # Examples: "Chapter 1: Introduction ........................ 5"
        #           "2.3. Methodology ............................ page 45"
        if _is_table_of_contents_entry(stripped):
            continue

        # Calculate character position
        char_pos = sum(len(lines[i]) + 1 for i in range(line_no))  # +1 for newline

        # Docling Markdown headings retain the document hierarchy and take
        # precedence over text-only chapter inference.
        markdown_heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", stripped)
        if markdown_heading:
            level = len(markdown_heading.group(1)) - 1
            section_title = markdown_heading.group(2).strip()
            markdown_chapter_label = None
            standalone_section_names = {
                "abstract",
                "acknowledgement",
                "acknowledgements",
                "declaration",
                "signed declaration",
                "copyright",
                "copyright notice",
                "copyright statement",
                "ethics statement",
                "statement of ethics",
                "statement of ethics approval",
                "ethics declaration",
                "ethics approval",
                "conflict of interest statement",
                "funding statement",
                "dedication",
                "foreword",
                "preface",
                "prelude",
                "prologue",
                "table of contents",
                "list of figures",
                "list of tables",
                "list of abbreviations",
                "references",
                "bibliography",
                "appendix",
                "appendices",
                "index",
            }
            toc_chapter_label = _match_toc_chapter_title(section_title, toc_chapters)
            chapter_heading = re.match(chapter_patterns[0], section_title, re.IGNORECASE)
            if toc_chapter_label:
                markdown_chapter_label = toc_chapter_label
                section_title = f"{toc_chapter_label}: {toc_chapters[toc_chapter_label]}"
                level = 0
            elif chapter_heading:
                chapter_num = chapter_heading.group(1)
                chapter_title = chapter_heading.group(2).strip()
                if _looks_like_numbered_heading(chapter_title):
                    chapter_label = _normalise_chapter_number(chapter_num, chapter_title)
                    toc_title = toc_chapters.get(chapter_label)
                    if toc_chapters and toc_title is None:
                        continue
                    if toc_title:
                        chapter_title = toc_title
                    markdown_chapter_label = chapter_label
                    section_title = (
                        f"{chapter_label}: {chapter_title}" if chapter_title else chapter_label
                    )
                    level = 0
            else:
                normalised_section_name = section_title.lower().rstrip(":.")
                is_appendix_heading = normalised_section_name.startswith(
                    ("appendix ", "appendices ")
                )
                if normalised_section_name in standalone_section_names or is_appendix_heading:
                    markdown_chapter_label = section_title
                    level = 0
            if structure:
                structure[-1]["end_pos"] = char_pos

            hierarchy = [
                (existing_level, title)
                for existing_level, title in hierarchy
                if existing_level < level
            ]
            hierarchy.append((level, section_title))
            if level == 0:
                current_chapter = markdown_chapter_label or section_title
            heading_path = " > ".join(title for _, title in hierarchy)
            structure_entry = {
                "start_pos": char_pos,
                "end_pos": None,
                "chapter": current_chapter or section_title,
                "section_title": section_title,
                "heading_path": heading_path,
                "level": level,
            }
            if level == 0:
                toc_title = toc_chapters.get(str(current_chapter or section_title))
                structure_entry["toc_verified"] = toc_title is not None
                if toc_title is not None:
                    structure_entry["toc_title"] = toc_title
            structure.append(structure_entry)
            current_section = section_title
            continue

        toc_chapter_label = _match_toc_chapter_title(stripped, toc_chapters)
        if toc_chapter_label:
            if structure:
                structure[-1]["end_pos"] = char_pos
            current_chapter = toc_chapter_label
            section_title = f"{toc_chapter_label}: {toc_chapters[toc_chapter_label]}"
            structure.append(
                {
                    "start_pos": char_pos,
                    "end_pos": None,
                    "chapter": toc_chapter_label,
                    "section_title": section_title,
                    "heading_path": toc_chapter_label,
                    "level": 0,
                    "toc_verified": True,
                    "toc_title": toc_chapters[toc_chapter_label],
                }
            )
            hierarchy = [(0, toc_chapter_label)]
            current_section = section_title
            continue

        # Check for chapter headings
        chapter_match = None
        for pattern_index, pattern in enumerate(chapter_patterns):
            match = re.match(pattern, stripped, re.IGNORECASE | re.MULTILINE)
            if match:
                # A bare "1. Title" form is also used for numbered prose and
                # cultural lists. Only treat it as a chapter when the title
                # clearly names an academic section; explicit "Chapter N"
                # headings remain unrestricted.
                if pattern_index == 2:
                    if re.match(r"^\d+\.\d+", stripped):
                        continue
                    numbered_title = match.group(2).strip()
                    if not re.search(
                        r"\b(?:introduction|background|literature|method(?:ology|s)?|result(?:s)?|"
                        r"finding(?:s)?|discussion|analysis|design|conclusion|recommendation|"
                        r"implication|theoretical|conceptual|data|endpoint|authentication)\b",
                        numbered_title,
                        re.IGNORECASE,
                    ):
                        continue
                chapter_match = match
                break

        if chapter_match:
            # Extract chapter number/label and title
            chapter_num = chapter_match.group(1)
            chapter_title = (
                chapter_match.group(2).strip() if len(chapter_match.groups()) > 1 else ""
            )
            if not _looks_like_numbered_heading(chapter_title):
                continue

            # Normalise chapter number (convert words to digits, roman to arabic)
            chapter_label = _normalise_chapter_number(chapter_num, chapter_title)
            toc_title = toc_chapters.get(chapter_label)
            if toc_chapters and toc_title is None:
                continue
            if toc_title:
                chapter_title = toc_title

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
                    "toc_verified": toc_title is not None,
                    **({"toc_title": toc_title} if toc_title is not None else {}),
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
                if re.fullmatch(rf"{re.escape(std_section)}[\s:.\-—]*", section_lower):
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
