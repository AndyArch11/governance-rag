"""Text chunking utilities for document segmentation.

Provides semantic-aware text chunking for RAG applications.
Chunks are sized to balance context retention and retrieval precision.
Supports adaptive chunking based on document type and content structure.

Table Handling:
    - Attempts to preserve table integrity (tables not split across chunks)
    - Detects [TABLE N] markers from HTML parser
    - Keeps tables with contextual content for semantic relevance
    - Prevents small table chunks from being filtered

Enhanced Metadata:
    - Extracts hierarchical heading paths
    - Identifies technical entities (APIs, components, configs)
    - Detects code blocks and language
    - Tracks sequential chunk relationships
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from scripts.utils.schemas import EnhancedChunkMetadata, ParentChunkSchema

# Import embedding limits to ensure chunks fit within model context
try:
    from scripts.ingest.vectors import EMBEDDING_CONTEXT_SAFETY_MARGIN, EMBEDDING_USABLE_TOKEN_LIMIT

    # Calculate maximum safe chunk size in characters
    # Use conservative estimate: 1 token per 3 chars
    # Reserve 12 chars for " [TRUNCATED]" suffix that may be added during embedding
    MAX_CHUNK_SIZE = (EMBEDDING_USABLE_TOKEN_LIMIT * 3) - EMBEDDING_CONTEXT_SAFETY_MARGIN - 12
except ImportError:
    # Fallback if vectors.py not available (shouldn't happen in production)
    MAX_CHUNK_SIZE = 700  # Conservative default


def create_parent_child_chunks(
    text: str, doc_type: Optional[str] = None, parent_size: int = 1200, child_size: int = 400
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Create parent-child chunk pairs for improved retrieval.

    Strategy:
    - Split text into large parent chunks (800-1200 tokens)
    - Further split each parent into smaller child chunks (200-400 tokens)
    - Child chunks are embedded and searched
    - Parent chunks provide context when children are retrieved

    Args:
        text: Full document text
        doc_type: Document type for adaptive sizing
        parent_size: Target size for parent chunks (default 1200)
        child_size: Target size for child chunks (default 400)

    Returns:
        Tuple of (child_chunks, parent_chunks) where:
            - child_chunks: List of dicts with {id, text, parent_id}
            - parent_chunks: List of dicts with {id, text, child_ids}

    Example:
        >>> children, parents = create_parent_child_chunks(doc_text)
        >>> # Store children for searching
        >>> # Store parents for context retrieval
    """
    # GUARD: Cap sizes to fit within embedding model's context limit
    # Parent chunks aren't embedded in the current implementation,
    # but child chunks are, so we must cap child_size
    if child_size > MAX_CHUNK_SIZE:
        child_size = MAX_CHUNK_SIZE

    # Parent size can be larger since parents aren't directly embedded,
    # but cap it reasonably to avoid memory issues
    if parent_size > MAX_CHUNK_SIZE * 3:
        parent_size = MAX_CHUNK_SIZE * 3

    # Create parent chunks (larger sections)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=int(parent_size * 0.15),  # 15% overlap
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
    )
    parent_texts = parent_splitter.split_text(text)

    # Create child chunks from each parent
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=int(child_size * 0.20),  # 20% overlap for children
        separators=["\n### ", "\n#### ", "\n\n", "\n", " ", ""],
    )

    child_chunks = []
    parent_chunks = []

    for parent_idx, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{parent_idx}"

        # Split parent into children
        child_texts = child_splitter.split_text(parent_text)
        child_ids = []

        for child_idx, child_text in enumerate(child_texts):
            child_id = f"child_{parent_idx}_{child_idx}"
            child_ids.append(child_id)

            child_chunks.append({"id": child_id, "text": child_text, "parent_id": parent_id})

        # Store parent with reference to its children
        parent_chunks.append({"id": parent_id, "text": parent_text, "child_ids": child_ids})

    return child_chunks, parent_chunks


def determine_chunk_params(doc_type: Optional[str], text: str) -> Tuple[int, int]:
    """Determine optimal chunk size and overlap based on document type and content.

    Analyses document type keywords and content structure to select appropriate
    chunk sizing. Adjusts for heading density, sentence length, and document
    purpose to optimise retrieval quality.

    Args:
        doc_type: Document type string (e.g., "governance policy", "technical guide").
        text: Full document text for structure analysis.

    Returns:
        Tuple of (chunk_size, chunk_overlap) in characters.

    Sizing Strategy:
        Base sizes by doc_type keywords:
        - policy/compliance/standard/requirement: 600 (concise rules/lists)
        - guide/procedure/tutorial/how-to: 1000 (step-by-step instructions)
        - architecture/pattern/design/framework: 1000 (complex concepts)
        - reference/api/specification: 800 (balanced documentation)
        - Default: 800

        Adjustments from content analysis:
        - High heading density (>15%): -200 (already well-structured)
        - Long avg sentence (>150 chars): +200 (complex prose needs more context)
        - Short avg sentence (<60 chars): -100 (bullet points/lists)

        Overlap: 18-20% of chunk_size (maintains context)

    Example:
        >>> determine_chunk_params("compliance policy", text)
        (600, 114)  # Short chunks for concise policies

        >>> determine_chunk_params("deployment guide", text)
        (1000, 190)  # Longer chunks for detailed procedures
    """
    # Base chunk size from doc_type
    base_size = 800  # Default

    if doc_type:
        doc_type_lower = doc_type.lower()

        # Policy/compliance documents - shorter chunks for concise rules
        if any(
            keyword in doc_type_lower
            for keyword in [
                "policy",
                "compliance",
                "standard",
                "requirement",
                "regulation",
                "control",
                "guideline",
                "principle",
                "rule",
            ]
        ):
            base_size = 600

        # Guides/procedures - longer chunks for step-by-step content
        elif any(
            keyword in doc_type_lower
            for keyword in [
                "guide",
                "procedure",
                "tutorial",
                "how-to",
                "walkthrough",
                "instruction",
                "deployment",
                "implementation",
                "setup",
            ]
        ):
            base_size = 1000

        # Architecture/patterns - longer chunks for complex concepts
        elif any(
            keyword in doc_type_lower
            for keyword in [
                "architecture",
                "pattern",
                "design",
                "framework",
                "model",
                "blueprint",
                "structure",
                "concept",
            ]
        ):
            base_size = 1000

        # Reference/API/spec - balanced chunks
        elif any(
            keyword in doc_type_lower
            for keyword in ["reference", "api", "specification", "documentation", "manual"]
        ):
            base_size = 800

    # Analyse content structure for fine-tuning
    adjustment = 0

    # Count headings (markdown format)
    heading_count = len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))
    text_length = len(text)

    if text_length > 0:
        heading_density = heading_count / (text_length / 100)  # Headings per 100 chars

        # High heading density means already well-structured - use smaller chunks
        if heading_density > 0.15:  # More than 15 headings per 1000 chars
            adjustment -= 200

        # Analyse sentence length
        sentences = re.split(r"[.!?]+\s+", text)
        if sentences:
            avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)

            # Long sentences (complex prose) - need more context
            if avg_sentence_length > 150:
                adjustment += 200
            # Short sentences (lists/bullets) - can use smaller chunks
            elif avg_sentence_length < 60:
                adjustment -= 100

    # Apply adjustment with bounds
    chunk_size = max(400, min(1200, base_size + adjustment))

    # Dynamic overlap based on content type (18-40% range)
    base_overlap_ratio = 0.19  # Default 19%

    # Increase overlap for technical content
    if doc_type:
        doc_type_lower = doc_type.lower()

        # API/reference docs need high overlap (context critical)
        if any(kw in doc_type_lower for kw in ["api", "reference", "specification"]):
            base_overlap_ratio = 0.35  # 35% overlap

        # Technical specs need increased overlap
        elif any(kw in doc_type_lower for kw in ["technical", "architecture", "design"]):
            base_overlap_ratio = 0.25  # 25% overlap

        # Code documentation needs high overlap
        elif any(kw in doc_type_lower for kw in ["code", "implementation", "development"]):
            base_overlap_ratio = 0.40  # 40% overlap

    # Check for code content in text sample
    if "```" in text or re.search(r"`\w+\(", text):
        base_overlap_ratio = max(base_overlap_ratio, 0.35)  # At least 35% for code

    chunk_overlap = int(chunk_size * base_overlap_ratio)

    return chunk_size, chunk_overlap


def chunk_text(text: str, doc_type: Optional[str] = None, adaptive: bool = True) -> List[str]:
    """Split text into semantic chunks for vector embedding.

    Uses recursive character splitting with adaptive chunk sizing based on
    document type and content structure. Prioritises semantic boundaries
    (headings, paragraphs) over arbitrary character limits.

    Table-Aware Chunking:
        - Detects [TABLE N] markers inserted by HTML parser
        - Preserves table structure and marker boundaries
        - Splits oversized tables by row groups while repeating headers/context
        - Keeps table order aligned with surrounding narrative context
        - Ensures tables with markers are not filtered as "small chunks"

    Args:
        text: Cleaned and preprocessed document text.
        doc_type: Optional document type for adaptive sizing (e.g., 'policy', 'guide').
        adaptive: Enable adaptive chunking based on content analysis (default: True).

    Returns:
        List of text chunks, each suitable for embedding.

    Adaptive Chunking Logic:
        When adaptive=True:
        1. Check doc_type for initial size estimate:
           - Policy/compliance: 600 chars (concise lists)
           - Guides/procedures: 1000 chars (detailed steps)
           - Architecture/patterns: 1000 chars (complex concepts)
           - Reference/standard: 800 chars (balanced)

        2. Analyse content structure:
           - High heading density (>15%): -200 chars
           - Long sentences (>150 chars): +200 chars
           - Short sentences (<60 chars): -100 chars

        3. Maintain overlap at 18-20% of chunk_size

    Static Chunking (adaptive=False):
        chunk_size: 800 tokens
        chunk_overlap: 150 tokens

    Configuration:
        separators: ["\\n## ", "\\n### ", "\\n", " ", ""]
            - Prioritises markdown headings
            - Falls back to paragraphs, sentences, words
            - Adjust if code blocks or lists break incorrectly

    Example:
        >>> # Policy document - shorter chunks
        >>> chunks = chunk_text(text, doc_type="compliance policy")
        >>> # Expected ~600 char chunks

        >>> # Technical guide - longer chunks
        >>> chunks = chunk_text(text, doc_type="deployment guide")
        >>> # Expected ~1000 char chunks

        >>> # Static chunking (legacy)
        >>> chunks = chunk_text(text, adaptive=False)
        >>> # Fixed 800 char chunks

    Note:
        Tune parameters via doc_type keywords in determine_chunk_params().
        Monitor chunk_heuristic_skip metrics to optimise retrieval performance.
    """
    # Check for tables and handle separately if present
    has_table_markers = "[TABLE" in text and "[/TABLE" in text

    if has_table_markers:
        # Use table-aware chunking strategy
        chunks = _chunk_text_with_tables(text, doc_type, adaptive)
    else:
        # Standard chunking for non-table content
        chunks = _chunk_text_standard(text, doc_type, adaptive)

    return chunks


def _chunk_text_standard(
    text: str, doc_type: Optional[str] = None, adaptive: bool = True
) -> List[str]:
    """Standard text chunking without table awareness.

    Internal helper for chunk_text().
    """
    # Determine chunk parameters
    if adaptive and text:
        chunk_size, chunk_overlap = determine_chunk_params(doc_type, text)
    else:
        # Static defaults for non-adaptive mode
        chunk_size = 800
        chunk_overlap = 150

    # GUARD: Cap chunk size to fit within embedding model's context limit
    # This prevents "context length exceeds" errors during embedding
    if chunk_size > MAX_CHUNK_SIZE:
        chunk_size = MAX_CHUNK_SIZE
        # Maintain ~20% overlap ratio
        chunk_overlap = max(50, int(chunk_size * 0.2))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n", " ", ""],
    )

    chunks = splitter.split_text(text)

    return chunks


def _chunk_text_with_tables(
    text: str, doc_type: Optional[str] = None, adaptive: bool = True
) -> List[str]:
    """Table-aware chunking that preserves table integrity.

    Strategy:
    1. Parse document into alternating non-table and table segments while
       preserving original order.
    2. Chunk non-table segments with standard chunking parameters.
    3. Chunk each table block by row boundaries when oversized, keeping
       headers/captions/context in each table part.
    4. Mark table chunks so downstream filters preserve them.

    Internal helper for chunk_text().
    """
    table_pattern = re.compile(r"\[TABLE (\d+)\](.*?)\[/TABLE \1\]", re.DOTALL)
    chunks: List[str] = []
    cursor = 0

    for match in table_pattern.finditer(text):
        non_table_segment = text[cursor : match.start()].strip()
        if non_table_segment:
            chunks.extend(_chunk_text_standard(non_table_segment, doc_type, adaptive))

        table_block = match.group(0)
        prefix_context = _extract_table_prefix_context(non_table_segment)
        chunks.extend(_chunk_single_table_block(table_block, prefix_context, doc_type, adaptive))
        cursor = match.end()

    trailing_non_table = text[cursor:].strip()
    if trailing_non_table:
        chunks.extend(_chunk_text_standard(trailing_non_table, doc_type, adaptive))

    return chunks


def _extract_table_prefix_context(non_table_segment: str, max_chars: int = 240) -> str:
    """Extract trailing context from non-table text for nearby table chunks."""
    if not non_table_segment:
        return ""

    lines = [line.strip() for line in non_table_segment.splitlines() if line.strip()]
    if not lines:
        return ""

    # Prefer recent heading/paragraph lines for compact context.
    selected = "\n".join(lines[-3:])
    if len(selected) <= max_chars:
        return selected
    return selected[-max_chars:]


def _chunk_single_table_block(
    table_block: str,
    prefix_context: str,
    doc_type: Optional[str] = None,
    adaptive: bool = True,
) -> List[str]:
    """Chunk one [TABLE N] block, splitting large tables by row boundaries.

    Distinguishes likely cosmetic/layout tables from content tables:
    - Layout table: no markdown separator row, small row/column footprint,
      and no nested-table signal. Converted to narrative bullets and chunked
      with standard text logic.
    - Content table: preserved with table markers and split by row groups.

    Nested tables (flattened upstream as "[nested: ...]") are treated as
    content-bearing and keep table markers.
    """
    lines = [line.rstrip() for line in table_block.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return [f"#### TABLE MARKER ####\n{table_block}\n#### /TABLE MARKER ####"]

    open_marker = lines[0]
    close_marker = lines[-1]
    body_lines = lines[1:-1]

    metadata_lines: List[str] = []
    table_rows: List[str] = []

    for line in body_lines:
        stripped = line.strip()
        is_table_row = stripped.startswith("|") and stripped.endswith("|")
        if is_table_row:
            table_rows.append(line)
        else:
            metadata_lines.append(line)

    explicit_table_type: Optional[str] = None
    cleaned_metadata_lines: List[str] = []
    for meta_line in metadata_lines:
        match = re.match(r"^Table-Type:\s*(layout|content)\s*$", meta_line.strip(), re.IGNORECASE)
        if match:
            explicit_table_type = match.group(1).lower()
            continue
        cleaned_metadata_lines.append(meta_line)

    metadata_lines = cleaned_metadata_lines

    separator_idx: Optional[int] = None
    for idx, row in enumerate(table_rows):
        if re.fullmatch(r"\|[\s:\-|]+\|", row.strip()):
            separator_idx = idx
            break

    if separator_idx is not None:
        header_lines = table_rows[: separator_idx + 1]
        data_lines = table_rows[separator_idx + 1 :]
    else:
        header_lines = []
        data_lines = table_rows

    is_nested_table = any("[nested:" in row.lower() for row in table_rows)
    col_count = max((row.count("|") - 1 for row in table_rows), default=0)
    row_count = len(table_rows)

    is_likely_layout_table = (
        separator_idx is None
        and not is_nested_table
        and row_count <= 4
        and col_count <= 2
        and bool(table_rows)
    )

    if explicit_table_type == "layout":
        is_likely_layout_table = True
    elif explicit_table_type == "content":
        is_likely_layout_table = False

    if is_likely_layout_table:
        layout_text = _format_layout_table_as_text(open_marker, prefix_context, metadata_lines, table_rows)
        return _chunk_text_standard(layout_text, doc_type, adaptive)

    if not data_lines:
        # Small/metadata-only tables: keep intact.
        return [
            _format_table_chunk(
                open_marker,
                close_marker,
                prefix_context,
                metadata_lines,
                header_lines,
                [],
                None,
            )
        ]

    expanded_rows: List[str] = []
    for row in data_lines:
        expanded_rows.extend(_split_oversized_table_row(row))

    row_groups: List[List[str]] = []
    current_group: List[str] = []

    for row in expanded_rows:
        candidate_group = current_group + [row]
        candidate_chunk = _format_table_chunk(
            open_marker,
            close_marker,
            prefix_context,
            metadata_lines,
            header_lines,
            candidate_group,
            None,
        )

        if current_group and len(candidate_chunk) > MAX_CHUNK_SIZE:
            row_groups.append(current_group)
            current_group = [row]
        else:
            current_group = candidate_group

    if current_group:
        row_groups.append(current_group)

    total_parts = len(row_groups)
    chunks: List[str] = []
    for idx, row_group in enumerate(row_groups):
        chunks.append(
            _format_table_chunk(
                open_marker,
                close_marker,
                prefix_context,
                metadata_lines,
                header_lines,
                row_group,
                (idx + 1, total_parts),
            )
        )

    return chunks


def _format_layout_table_as_text(
    open_marker: str,
    prefix_context: str,
    metadata_lines: List[str],
    table_rows: List[str],
) -> str:
    """Convert a likely cosmetic/layout table to narrative text.

    This avoids over-preserving non-semantic layout scaffolding as table chunks.
    """
    match = re.search(r"\[TABLE\s+(\d+)\]", open_marker)
    table_idx = match.group(1) if match else "?"

    lines: List[str] = [f"Layout table {table_idx} (cosmetic structure)"]
    if prefix_context:
        lines.append(f"Context: {prefix_context}")

    lines.extend(metadata_lines)

    for row in table_rows:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        cells = [cell for cell in cells if cell]
        if cells:
            lines.append(" - " + " | ".join(cells))

    return "\n".join(lines)


def _split_oversized_table_row(row: str) -> List[str]:
    """Split a very long markdown table row into multiple row fragments.

    Preserves cell boundaries by grouping consecutive cells into row-sized
    fragments that are easier to pack into chunk limits.
    """
    if len(row) <= int(MAX_CHUNK_SIZE * 0.7):
        return [row]

    stripped = row.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return [row]

    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if len(cells) <= 1:
        return [row]

    fragments: List[str] = []
    current_cells: List[str] = []

    for cell in cells:
        candidate_cells = current_cells + [cell]
        candidate_row = "| " + " | ".join(candidate_cells) + " |"

        if current_cells and len(candidate_row) > int(MAX_CHUNK_SIZE * 0.7):
            fragments.append("| " + " | ".join(current_cells) + " |")
            current_cells = [cell]
        else:
            current_cells = candidate_cells

    if current_cells:
        fragments.append("| " + " | ".join(current_cells) + " |")

    return fragments if fragments else [row]


def _format_table_chunk(
    open_marker: str,
    close_marker: str,
    prefix_context: str,
    metadata_lines: List[str],
    header_lines: List[str],
    data_lines: List[str],
    part_info: Optional[Tuple[int, int]],
) -> str:
    """Format a single table chunk with marker wrappers and optional part label."""
    parts: List[str] = ["#### TABLE MARKER ####"]
    parts.append(open_marker)

    if part_info is not None:
        part_no, total_parts = part_info
        parts.append(f"Table Part: {part_no}/{total_parts}")

    if prefix_context:
        parts.append(f"Context: {prefix_context}")

    parts.extend(metadata_lines)
    parts.extend(header_lines)
    parts.extend(data_lines)
    parts.append(close_marker)
    parts.append("#### /TABLE MARKER ####")
    return "\n".join(parts)


def extract_technical_entities(text: str) -> List[str]:
    """Extract technical entities from chunk text.

    Identifies:
    - API endpoints (/api/v1/users, GET /resource)
    - Function/method names (camelCase, snake_case, function())
    - Class names (PascalCase)
    - Constants (UPPER_SNAKE_CASE)
    - Configuration keys (config.database.host)
    - Package/module names (module.submodule)

    Args:
        text: Chunk text to analyse

    Returns:
        List of unique technical entity names
    """
    entities: Set[str] = set()

    # API endpoints
    api_endpoints = re.findall(r"/(?:api/)?[\w/]+", text)
    entities.update(e for e in api_endpoints if "/" in e and len(e) > 3)

    # Function/method names (snake_case or camelCase with optional parens)
    functions = re.findall(r"\b[a-z_][a-z0-9_]*\([^)]*\)", text)
    entities.update(f.split("(")[0] for f in functions)

    # Class names (PascalCase)
    classes = re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:[A-Z][a-z0-9]+)+\b", text)
    entities.update(classes)

    # Constants (UPPER_SNAKE_CASE)
    constants = re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text)
    entities.update(c for c in constants if "_" in c)

    # Configuration keys (dot notation)
    config_keys = re.findall(r"\b[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_.]*\b", text)
    entities.update(config_keys)

    # Package/module names (underscore or dot notation)
    modules = re.findall(r"\b[a-z][a-z0-9_]*(?:_[a-z0-9]+)+\b", text)
    entities.update(m for m in modules if len(m) > 5)  # Filter short names

    return sorted(entities)[:20]  # Limit to top 20 entities


def detect_code_language(text: str) -> Optional[str]:
    """Detect programming language in text.

    Looks for:
    - Code fence markers (```python, ```javascript, etc.)
    - Common keywords and patterns

    Args:
        text: Text possibly containing code

    Returns:
        Language name or None if no code detected
    """
    # Check for code fence markers
    fence_match = re.search(r"```(\w+)", text)
    if fence_match:
        return fence_match.group(1)

    # Check for inline code markers with language hints
    if "`" in text:
        # Python indicators
        if any(kw in text for kw in ["def ", "import ", "class ", "self.", "print("]):
            return "python"
        # JavaScript indicators
        if any(kw in text for kw in ["const ", "let ", "var ", "function(", "=>", "console.log"]):
            return "javascript"
        # SQL indicators
        if any(kw in text.upper() for kw in ["SELECT ", "FROM ", "WHERE ", "INSERT INTO"]):
            return "sql"
        # YAML indicators
        if re.search(r"^\w+:\s*$", text, re.MULTILINE):
            return "yaml"

    return None


def extract_heading_path(text: str, full_text: str) -> Optional[str]:
    """Extract hierarchical heading path for chunk.

    Finds the chunk's position in document structure and builds
    a path like "Configuration > Database > Connection Pool"

    Args:
        text: Chunk text
        full_text: Complete document text for context

    Returns:
        Heading path string or None if no headings found
    """
    # Find chunk position in full text
    chunk_pos = full_text.find(text[:100]) if len(text) >= 100 else full_text.find(text)
    if chunk_pos == -1:
        return None

    # Extract all headings before this chunk
    headings_before = re.findall(r"^(#{1,6})\s+(.+)$", full_text[:chunk_pos], re.MULTILINE)

    if not headings_before:
        return None

    # Build hierarchical path
    current_levels: Dict[int, str] = {}
    for level_markers, heading_text in headings_before:
        level = len(level_markers)
        current_levels[level] = heading_text.strip()
        # Clear deeper levels
        current_levels = {k: v for k, v in current_levels.items() if k <= level}

    # Build path from top to bottom
    path_parts = [current_levels[level] for level in sorted(current_levels.keys())]
    return " > ".join(path_parts) if path_parts else None


def create_enhanced_metadata(
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    doc_id: str,
    full_text: str = "",
    doc_type: Optional[str] = None,
    document_structure: Optional[List[Dict[str, Any]]] = None,
    chunk_char_start: Optional[int] = None,
    chunk_char_end: Optional[int] = None,
) -> EnhancedChunkMetadata:
    """Create enhanced metadata for a chunk.

    Extracts rich contextual information to improve retrieval quality.

    Args:
        chunk_text: The chunk text content
        chunk_index: Index of this chunk (0-based)
        total_chunks: Total number of chunks in document
        doc_id: Document identifier
        full_text: Complete document text for context
        doc_type: Document type classification
        document_structure: Optional pre-extracted structure from pdfparser
        chunk_char_start: Character position where chunk starts in full_text
        chunk_char_end: Character position where chunk ends in full_text

    Returns:
        EnhancedChunkMetadata instance with extracted information
    """
    # Detect content types
    contains_code = bool(re.search(r"```|`\w+\(", chunk_text))
    contains_table = "[TABLE" in chunk_text or "| --- |" in chunk_text
    contains_diagram = any(
        marker in chunk_text.lower() for marker in ["[diagram", "[image", "![", "figure ", "chart "]
    )

    # Extract technical entities
    entities = extract_technical_entities(chunk_text)

    # Detect code language
    code_lang = detect_code_language(chunk_text) if contains_code else None

    # Determine content classification
    is_api_ref = any(
        term in chunk_text.lower() for term in ["endpoint", "api", "method:", "returns:"]
    )
    is_config = any(
        term in chunk_text.lower()
        for term in ["configuration", "config", "setting", "parameter", "option"]
    )

    # Determine content type
    if contains_code and len(re.findall(r"```", chunk_text)) >= 2:
        content_type = "code"
    elif contains_table or contains_diagram:
        content_type = "structured"
    elif contains_code:
        content_type = "mixed"
    else:
        content_type = "text"

    # Extract heading path from structure if available, otherwise fall back to text parsing
    heading_path = None
    parent_section = None
    section_depth = None
    chapter = None
    section_title = None

    if document_structure and chunk_char_start is not None and chunk_char_end is not None:
        # Use pre-extracted structure for efficient chapter/section mapping
        from scripts.ingest.pdfparser import map_text_to_structure

        struct_meta = map_text_to_structure(
            full_text, document_structure, chunk_char_start, chunk_char_end
        )
        chapter = struct_meta.get("chapter")
        section_title = struct_meta.get("section_title")
        heading_path = struct_meta.get("heading_path")

        if heading_path:
            path_parts = heading_path.split(" > ")
            parent_section = path_parts[-1] if path_parts else None
            section_depth = len(path_parts)
    else:
        # Fallback to text-based extraction
        heading_path = extract_heading_path(chunk_text, full_text) if full_text else None

        if heading_path:
            path_parts = heading_path.split(" > ")
            parent_section = path_parts[-1] if path_parts else None
            section_depth = len(path_parts)

    # Sequential relationships
    prev_chunk_id = f"{doc_id}_chunk_{chunk_index - 1}" if chunk_index > 0 else None
    next_chunk_id = f"{doc_id}_chunk_{chunk_index + 1}" if chunk_index < total_chunks - 1 else None

    return EnhancedChunkMetadata(
        chapter=chapter,
        section_title=section_title,
        heading_path=heading_path,
        parent_section=parent_section,
        section_depth=section_depth,
        prev_chunk_id=prev_chunk_id,
        next_chunk_id=next_chunk_id,
        technical_entities=entities,
        code_language=code_lang,
        contains_table=contains_table,
        contains_code=contains_code,
        contains_diagram=contains_diagram,
        is_api_reference=is_api_ref,
        is_configuration=is_config,
        content_type=content_type,
    )
