"""HTML extraction and cleaning for document ingestion.

Removes navigation, headers, footers, table of contents, and boilerplate
elements from HTML documents before text extraction. Handles Confluence
exports and general web content.

Table Handling:
    Intelligently extracts and formats tables with special markers:
    - Preserves table structure (headers, rows, columns)
    - Converts markdown-like table format for semantic chunking
    - Marks table boundaries for better chunk creation
    - Identifies table context and purpose
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from bs4 import BeautifulSoup, NavigableString

from scripts.utils.retry_utils import retry_with_backoff

# Confluence-specific patterns to remove
CONFLUENCE_NAV_PATTERNS = [
    r"Home\s*>",  # Breadcrumb navigation
    r"Confluence\s*[>|]",
    r"Export\s+PDF",
    r"Watch\s+\|",
    r"Share\s+\|",
    r"Tools\s+\|",
]

# Common header/footer patterns
BOILERPLATE_PATTERNS = [
    r"^Page \d+\s*of\s*\d+",  # Page numbers
    r"^Generated\s+on\s+\d{1,2}/\d{1,2}/\d{4}",  # Generation dates
    r"^Last\s+modified\s+on",
    r"^Document\s+ID:|ID:\s*[A-Z0-9-]+",
    r"^Copyright\s+[©©]\s*\d{4}",
    r"^Version\s+\d+\.\d+",
    # Note: Table separators like |---|---| are preserved (meaningful within tables)
    r"^[\s-]{3,}$",  # Horizontal separator lines (no pipes, just dashes/spaces)
]

# Patterns for repeating headers/footers (appear at top/bottom of pages)
REPEATING_PATTERNS = [
    r"^Related\s+Articles?",
    r"^See\s+Also",
    r"^Next\s+Steps?",
    r"^Recommended\s+Reading",
    r"^Attachments?",
    r"^Labels?:",
    r"^Was\s+this\s+helpful\?",
    r"^Rate\s+this\s+page",
    r"^Comments?:",
]


def _extract_confluence_metadata(text: str) -> tuple[str, dict]:
    """Extract and remove Confluence metadata from text.

    Args:
        text: Raw text extracted from HTML

    Returns:
        Tuple of (cleaned_text, metadata_dict)
    """
    metadata = {}
    lines = text.split("\n")
    content_lines = []

    for line in lines:
        # Check for Confluence metadata
        if line.strip().startswith("Labels:"):
            try:
                metadata["labels"] = line.split(":", 1)[1].strip()
            except IndexError:
                pass
            continue
        if line.strip().startswith("Related Articles"):
            metadata["has_related"] = True
            continue

        content_lines.append(line)

    return "\n".join(content_lines), metadata


def _extract_table_metadata(table_elem) -> Dict[str, Any]:
    """Extract metadata from a table element.

    TODO: some intelligence around identifying table purpose (e.g., data table vs layout table)
    and context (e.g., is it a key results table, a list of items, etc)
    based on content and surrounding text. This could inform chunking and relevance scoring later on.
    TODO: extract more detailed structure info (e.g., multi-row headers, nested tables)
    for better understanding of table complexity and content.

    Args:
        table_elem: BeautifulSoup table element

    Returns:
        Dict with table metadata (rows, cols, has_headers, etc)
    """
    # Use recursive=False to avoid counting rows from nested tables
    rows = table_elem.find_all("tr", recursive=False)
    # Also check tbody/thead for rows if direct children aren't found
    if not rows:
        for section in table_elem.find_all(["tbody", "thead", "tfoot"], recursive=False):
            rows.extend(section.find_all("tr", recursive=False))

    row_count = len(rows)

    if row_count == 0:
        return {"rows": 0, "cols": 0, "has_headers": False, "header_rows": 0}

    # Count consecutive header rows from the start
    header_row_count = 0
    for row in rows:
        if row.find_all("th", recursive=False):
            header_row_count += 1
        else:
            break

    # Count columns accounting for colspan (use first row)
    cells = rows[0].find_all(["td", "th"], recursive=False)
    col_count = 0
    for cell in cells:
        colspan = int(cell.get("colspan", 1))
        col_count += colspan

    nested_table_count = max(0, len(table_elem.find_all("table")) - 1)
    table_type = _classify_table_type(
        table_elem=table_elem,
        rows=rows,
        row_count=row_count,
        col_count=col_count,
        header_row_count=header_row_count,
        nested_table_count=nested_table_count,
    )

    return {
        "rows": row_count,
        "cols": col_count,
        "has_headers": header_row_count > 0,
        "header_rows": header_row_count,
        "nested_tables": nested_table_count,
        "table_type": table_type,
    }


def _classify_table_type(
    table_elem,
    rows: List[Any],
    row_count: int,
    col_count: int,
    header_row_count: int,
    nested_table_count: int,
) -> str:
    """Classify table as 'content' or 'layout' using structural signals.

    The goal is to prevent cosmetic layout grids from being over-preserved as
    semantic data tables while keeping true data tables intact.
    """
    # Nested tables usually represent embedded data blocks rather than cosmetic layout.
    if nested_table_count > 0:
        return "content"

    # Captions, explicit headers, and semantic roles strongly suggest content tables.
    caption_elem = table_elem.find("caption")
    if caption_elem and caption_elem.get_text(strip=True):
        return "content"

    role = str(table_elem.get("role", "")).strip().lower()
    if role in {"grid", "treegrid"}:
        return "content"

    if header_row_count > 0 or table_elem.find("thead") is not None:
        return "content"

    # Gather direct cell content only (avoid nested table duplication).
    direct_cells = []
    for row in rows:
        direct_cells.extend(row.find_all(["td", "th"], recursive=False))

    if not direct_cells:
        return "layout"

    non_empty_cells = 0
    short_cell_count = 0
    long_cell_count = 0
    token_total = 0

    for cell in direct_cells:
        cell_text = re.sub(r"\s+", " ", cell.get_text(separator=" ", strip=True)).strip()
        if not cell_text:
            continue

        non_empty_cells += 1
        token_count = len(cell_text.split())
        token_total += token_count

        if token_count <= 3:
            short_cell_count += 1
        if token_count >= 8:
            long_cell_count += 1

    if non_empty_cells == 0:
        return "layout"

    short_ratio = short_cell_count / non_empty_cells
    avg_tokens = token_total / non_empty_cells

    # Compact 2-column (or single-column) grids with terse labels are usually layout scaffolding.
    if row_count <= 4 and col_count <= 2 and short_ratio >= 0.75 and long_cell_count == 0:
        return "layout"

    # Very sparse text in tiny tables is often presentational.
    if row_count <= 3 and col_count <= 3 and avg_tokens <= 2.0:
        return "layout"

    return "content"


def _convert_table_to_text(table_elem, table_index: int) -> str:
    """Convert HTML table to readable text format.

    Converts tables to pipe-delimited markdown format:
    | Header 1 | Header 2 |
    |----------|----------|
    | Row 1    | Data 1   |

    Handles:
    - Merged cells (colspan/rowspan) by repeating content
    - Nested tables (removes them to avoid double-processing)
    - Table captions and titles
    - Context preservation (paragraph before table)

    Args:
        table_elem: BeautifulSoup table element
        table_index: Index of table in document (for markers)

    Returns:
        Formatted table text with boundary markers and context
    """
    metadata = _extract_table_metadata(table_elem)
    table_type = metadata.get("table_type", "content")

    # Extract table caption if present
    caption = ""
    caption_elem = table_elem.find("caption")
    if caption_elem:
        caption = caption_elem.get_text(strip=True)

    # Try to find context from previous sibling (often explanatory paragraph)
    context = ""
    prev_sibling = table_elem.find_previous_sibling(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    if prev_sibling:
        context_text = prev_sibling.get_text(strip=True)
        # Only include if it's short enough to be a table title/description
        if len(context_text) < 200:
            context = context_text

    # Handle nested tables: extract their text content, then remove them to avoid
    # processing as separate tables (which would duplicate the content)
    # Clone the element to avoid modifying the original
    from copy import copy

    table_copy = copy(table_elem)

    # Replace nested tables with their text content (flattened)
    # TODO: Does not properly handle nested tables within nested tables.
    # Should ideally recursively flatten all nested tables.
    # As data tables are typically buried within layout tables in websites, current approach fails to capture these.
    for nested_table in table_copy.find_all("table"):
        if nested_table != table_copy:
            # Extract text from nested table before removing it
            nested_text = nested_table.get_text(separator=" ", strip=True)
            # Replace table with text node
            nested_table.replace_with(f"[nested: {nested_text}]")

    # Use recursive=False to only get direct tr children, or from tbody/thead/tfoot
    rows = table_copy.find_all("tr", recursive=False)
    if not rows:
        for section in table_copy.find_all(["tbody", "thead", "tfoot"], recursive=False):
            rows.extend(section.find_all("tr", recursive=False))

    if not rows:
        return ""

    # Detect all consecutive header rows (rows with <th> tags)
    # This handles multi-row headers common in complex tables
    header_row_count = 0
    for row in rows:
        if row.find_all("th", recursive=False):
            header_row_count += 1
        else:
            # Stop at first non-header row
            break

    # Extract table data with colspan/rowspan handling
    table_data: List[List[str]] = []
    for row in rows:
        cells = row.find_all(["td", "th"], recursive=False)
        row_data = []
        for cell in cells:
            # Get cell content (nested tables already removed)
            cell_text = cell.get_text(separator=" ", strip=True)
            # Remove extra whitespace but preserve structure
            cell_text = re.sub(r"\s+", " ", cell_text)

            # Preserve empty cells as empty strings (don't skip them)
            # This is important for table structure alignment
            if not cell_text:
                cell_text = ""

            # Handle colspan by repeating the cell content
            colspan = int(cell.get("colspan", 1))
            for _ in range(colspan):
                row_data.append(cell_text)

            # Note: rowspan is harder to handle in this simple format
            # We'll just capture the content in the first row it appears
            # TODO: More complex handling would require tracking cell positions

        if row_data:
            table_data.append(row_data)

    if not table_data:
        return ""

    # Determine column count
    col_count = max(len(row) for row in table_data) if table_data else 0

    # Pad rows to same length
    for row in table_data:
        while len(row) < col_count:
            row.append("")

    # Build markdown-like table with context
    lines = [f"[TABLE {table_index}]", f"Table-Type: {table_type}"]

    # Add context if available
    if context:
        lines.append(f"Context: {context}")

    # Add caption if available
    if caption:
        lines.append(f"Caption: {caption}")

    if context or caption:
        lines.append("")  # Blank line separator

    # Add all header rows, then separator, then data rows
    if header_row_count > 0:
        # Add all header rows
        for i in range(header_row_count):
            header_line = " | ".join(table_data[i])
            lines.append(f"| {header_line} |")
        # Add separator after all headers
        lines.append("|" + "|".join(["---"] * col_count) + "|")
        data_rows = table_data[header_row_count:]
    else:
        data_rows = table_data

    # Add data rows
    for row in data_rows:
        row_line = " | ".join(row)
        lines.append(f"| {row_line} |")

    lines.append(f"[/TABLE {table_index}]")

    return "\n".join(lines)


def _extract_tables_from_html(soup: BeautifulSoup) -> Tuple[str, List[Dict]]:
    """Extract tables from HTML with structure preservation.

    Only processes top-level tables to avoid double-processing nested tables.
    Nested tables are handled within their parent table's content.

    Args:
        soup: BeautifulSoup parsed HTML document

    Returns:
        Tuple of (tables_text, table_metadata_list)
    """
    from scripts.utils.logger import get_logger

    logger = get_logger("ingest")

    # Find all tables
    all_tables = soup.find_all("table")

    # Filter to only top-level tables (not nested inside other tables)
    top_level_tables = []
    for table in all_tables:
        # Check if this table is nested inside another table
        is_nested = False
        for parent in table.parents:
            if parent.name == "table" and parent in all_tables:
                is_nested = True
                break
        if not is_nested:
            top_level_tables.append(table)

    tables_text_parts = []
    tables_metadata = []
    skipped_count = 0
    nested_count = len(all_tables) - len(top_level_tables)

    for idx, table in enumerate(top_level_tables):
        # Extract metadata about the table
        metadata = _extract_table_metadata(table)
        metadata["index"] = idx
        tables_metadata.append(metadata)

        # Only skip completely empty tables (0 rows or 0 cols)
        # Single-row tables might be headers or meaningful content
        if metadata["rows"] == 0 or metadata["cols"] == 0:
            skipped_count += 1
            logger.debug(f"Skipping empty table {idx}: {metadata['rows']}x{metadata['cols']}")
            continue

        # Convert table to text format
        table_text = _convert_table_to_text(table, idx)
        if table_text:
            tables_text_parts.append(table_text)

    if all_tables:
        logger.debug(
            f"Extracted {len(tables_text_parts)}/{len(top_level_tables)} top-level tables "
            f"(skipped {skipped_count} empty, {nested_count} nested)"
        )

    return "\n\n".join(tables_text_parts), tables_metadata


def _strip_boilerplate(text: str) -> str:
    """Strip common boilerplate patterns from text.

    Removes page numbers, dates, separators, and other recurring elements.

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text with boilerplate removed
    """
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            cleaned_lines.append(line)
            continue

        # Skip lines matching boilerplate patterns
        is_boilerplate = any(
            re.match(pattern, stripped, re.IGNORECASE) for pattern in BOILERPLATE_PATTERNS
        )

        if is_boilerplate:
            continue

        # Skip lines matching repeating patterns (likely section headers/footers)
        is_repeating = any(
            re.match(pattern, stripped, re.IGNORECASE) for pattern in REPEATING_PATTERNS
        )

        if is_repeating:
            continue

        # Skip breadcrumb navigation
        if any(re.search(pattern, stripped, re.IGNORECASE) for pattern in CONFLUENCE_NAV_PATTERNS):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _clean_whitespace(text: str) -> str:
    """Normalise whitespace and remove excessive blank lines.

    - Removes leading/trailing whitespace
    - Collapses multiple blank lines into single blank line
    - Normalises tabs to spaces

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text with normalised whitespace
    """
    # Replace tabs with spaces
    text = text.replace("\t", "    ")

    # Collapse multiple blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


@retry_with_backoff(
    max_retries=2,
    initial_delay=0.5,
    transient_types=(IOError, MemoryError, TimeoutError),
    operation_name="parse_html",
)
def extract_text_from_html(path: str) -> str:
    """Extract and clean text from HTML documents.

    Removes:
    - Script and style tags
    - Navigation elements (nav, header, footer)
    - Confluence-specific navigation (breadcrumbs, buttons)
    - Page numbers and dates
    - Repeating headers/footers
    - Excessive whitespace

    Intelligently handles:
    - Tables: Preserves structure with markdown-like formatting
    - Table markers: [TABLE N] and [/TABLE N] for boundary detection
    - Table metadata: Identifies rows, columns, headers for chunking

    Args:
        path: Path to HTML file

    Returns:
        Cleaned text content with tables formatted and marked

    Example:
        >>> text = extract_text_from_html("confluence_export.html")
        >>> # Text includes [TABLE 0] markers around extracted tables
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style tags (content not needed)
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Extract and replace tables inline (preserve position/context)
    all_tables = soup.find_all("table")

    # Filter to only top-level tables
    top_level_tables = []
    for table in all_tables:
        is_nested = False
        for parent in table.parents:
            if parent.name == "table" and parent in all_tables:
                is_nested = True
                break
        if not is_nested:
            top_level_tables.append(table)

    # Replace each table with its text representation inline
    for idx, table in enumerate(top_level_tables):
        table_text = _convert_table_to_text(table, idx)
        if table_text:
            # Replace table element with formatted text
            table.replace_with(f"\n{table_text}\n")
        else:
            # Remove empty tables
            table.decompose()

    # Remove navigation elements
    for tag in soup(["nav", "header", "footer"]):
        tag.decompose()

    # Remove common navigation patterns in divs/sections
    # Look for elements with class/id containing 'nav', 'header', 'footer', 'breadcrumb', 'toc'
    # Collect tags to remove first to avoid modifying while iterating
    tags_to_remove = []
    for tag in soup.find_all(["div", "section", "aside"]):
        if tag is None or not hasattr(tag, "attrs") or tag.attrs is None:
            continue
        classes = tag.get("class", [])
        elem_id = tag.get("id", "")

        # Handle potential None values from BeautifulSoup
        if classes is None:
            classes = []
        if elem_id is None:
            elem_id = ""

        for nav_pattern in ["nav", "header", "footer", "breadcrumb", "toc", "menu", "sidebar"]:
            if (
                any(nav_pattern in str(c).lower() for c in classes)
                or nav_pattern in str(elem_id).lower()
            ):
                tags_to_remove.append(tag)
                break

    # Now remove collected tags
    for tag in tags_to_remove:
        tag.decompose()

    # Extract text with newline separator
    text = soup.get_text(separator="\n")

    # Remove Confluence metadata and get cleaned text
    text, _ = _extract_confluence_metadata(text)

    # Strip boilerplate patterns
    text = _strip_boilerplate(text)

    # Clean and normalise whitespace
    text = _clean_whitespace(text)

    return text
