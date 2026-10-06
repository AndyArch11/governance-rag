"""Prompt assembly for RAG generation.

Constructs prompts for the LLM by combining user queries with retrieved context.
Provides a configurable system prompt template emphasising grounding in provided
context to reduce hallucination and improve faithfulness.

Features:
  - Clear separation of system instructions, context, and question
  - Explicit instruction against inventing information
  - Numbered chunk labels for easy citation (e.g., "[Chunk 1]")
  - Token budgeting: truncates context if it exceeds max size
  - Encourages source citation and acknowledges ambiguity

Best practices applied:
  - Per-chunk source ID labeling for better attribution
  - Token-aware truncation to prevent oversized prompts
  - System prompt explicitly references chunk labels

TODO: Future improvements:
  - Configurable system prompts per use case (Q&A, summarisation, etc.)
  - Prompt optimisation (instruction tuning, few-shot examples)
  - Chunk reranking before assembly
  - More sophisticated code snippet detection and formatting
"""

from typing import Any, Dict, List, Optional

from scripts.utils.logger import create_module_logger

from .rag_config import RAGConfig

get_logger, _ = create_module_logger("rag")

# Get config for token budgeting
config = RAGConfig()
logger = get_logger()  # Initialise logger from module-level create_module_logger


def _format_truncation_summary(
    included_chunks: int,
    total_chunks: int,
    chunk_number: int,
    partial_chars: int,
) -> str:
    """Describe complete and partial context included before a budget cutoff."""
    if partial_chars:
        return (
            f"Included {included_chunks} complete chunks and a partial prefix of chunk "
            f"{chunk_number} ({partial_chars} chars), out of {total_chunks}."
        )
    return (
        f"Included {included_chunks} complete chunks; no partial text from chunk "
        f"{chunk_number} fit, out of {total_chunks}."
    )


def _format_chunk_label(chunk_number: int, metadata: Optional[Dict[str, Any]]) -> str:
    """Format a chunk label with available source and section provenance.

    Args:
        chunk_number (int): The number of the chunk.
        metadata (Optional[Dict[str, Any]]): Metadata containing source and section information.

    Returns:
        str: Formatted chunk label with provenance information if available.
    """
    label = f"[Chunk {chunk_number}]"
    if not metadata:
        return label

    provenance = []
    source_path = metadata.get("source_path") or metadata.get("file_path") or metadata.get("source")
    if source_path:
        provenance.append(f"source: {source_path}")

    section_path = metadata.get("heading_path") or metadata.get("section_title")
    if section_path:
        provenance.append(f"section: {section_path}")
    elif metadata.get("chapter"):
        provenance.append(f"chapter: {metadata['chapter']}")
    if metadata.get("graph_provenance"):
        provenance.append(f"graph path: {metadata['graph_provenance']}")

    return f"{label} ({'; '.join(provenance)})" if provenance else label


# System prompt template for RAG assistant.
# Emphasises grounding in provided context and explicit instructions to reduce hallucination.
# Note: The prompt explicitly instructs the LLM to cite chunk numbers and avoid inventing information.
# Spelling as per US English conventions (e.g., "specializing") to match common LLM training data.
SYSTEM_PROMPT = """
You are a technical assistant specializing in governance, security, and infrastructure policies.

IMPORTANT INSTRUCTIONS:
- Use ONLY the provided context to answer questions.
- Context is provided as numbered chunks (e.g., [Chunk 1], [Chunk 2]).
- When a source path, section location, or graph path is shown with a chunk, cite it alongside the chunk number.
- The chunks shown are ONLY the most relevant results retrieved for this query, NOT the entire document corpus.
- Cite the chunk number(s) when using information: "According to [Chunk 1]..."
- If the context doesn't contain relevant information, say "Based on the retrieved chunks, I don't see..."
- For questions about document statistics (e.g., "how many times", "all occurrences"), clarify that you can only see the retrieved chunks, not all documents.
- Do NOT invent, assume, or add information not in the context.
- If multiple chunks provide similar or conflicting information, acknowledge this.
- If multiple interpretations exist, acknowledge the ambiguity and cite each interpretation's source.
- Respond using UK English spelling and grammar conventions.
"""

# System prompt template for code-specific queries
# System prompt template for academic/research queries
ACADEMIC_SYSTEM_PROMPT = """
You are a research assistant specializing in academic papers, theses, and scholarly literature.

IMPORTANT INSTRUCTIONS:
- Use ONLY the provided context to answer questions about research papers and academic content.
- Context is provided as numbered chunks (e.g., [Chunk 1], [Chunk 2]).
- When a source path, section location, or graph path is shown with a chunk, cite it alongside the chunk number.
- The chunks shown are ONLY the most relevant results retrieved for this query, NOT the entire academic corpus.
- Cite the chunk number(s) when using information: "According to [Chunk 1]..."
- Include author names, publication years, and research institutions when mentioned in the context.
- Reference methodologies, findings, and conclusions from the papers.
- If the context doesn't contain relevant information, say "Based on the retrieved papers, I don't see..."
- For questions about research trends or citation counts, clarify that you can only see the retrieved papers, not all publications.
- Do NOT invent, assume, or add information not in the context.
- Acknowledge when papers present different approaches, methodologies, or conflicting findings.
- Cite specific papers when discussing research methods or results.
- Respond using UK English spelling and grammar conventions.
"""


def build_prompt(
    query: str,
    chunks: List[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
    custom_role: Optional[str] = None,
) -> str:
    """Build a RAG prompt from query and retrieved context chunks.

    Combines system instructions, context, and user query into a structured prompt
    that grounds the LLM response in provided documents and reduces hallucination.

    Features:
    - Labels each chunk with a number (e.g., "[Chunk 1]", "[Chunk 2]") for citation
    - Implements token budgeting: truncates context if total exceeds max_context_chars
    - Logs warning if context was truncated
    - Supports custom role/system prompt override
    - Auto-detects academic content and uses appropriate system prompt

    Args:
        query: User question or query string to be answered.
        chunks: List of retrieved context chunks (already ranked by relevance).
        metadata: Optional list of metadata dicts for auto-detecting query type
                 (e.g., academic vs general). Used to select appropriate system prompt.
        custom_role: Optional custom system prompt. If None, auto-detects from metadata
                    or uses default SYSTEM_PROMPT.

    Returns:
        Formatted prompt string with system prompt, labeled chunks, raw context,
        and question. Ready to pass directly to LLM.invoke(prompt).

    Raises:
        ValueError: If query is empty or chunks list is empty.

    Example:
        >>> chunks = ["MFA requires two or more authentication factors...", "MFA is used for security..."]
        >>> prompt = build_prompt("What is MFA?", chunks)
        >>> print(prompt)  # doctest: +SKIP
        You are a technical assistant...
        CONTEXT:
        [Chunk 1] MFA requires two or more...
        [Chunk 2] MFA is used for...
        QUESTION:
        What is MFA?
        ANSWER:

    Note: Implements token budgeting via max_context_chars from config.
    If context exceeds this limit, chunks are truncated with a warning logged.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if not chunks:
        raise ValueError("Context chunks cannot be empty")

    # Label and assemble chunks with token budgeting
    context_parts: List[str] = []
    used_chunks: List[str] = []
    total_chars = 0
    truncated = False

    for i, chunk in enumerate(chunks, 1):
        chunk_metadata = metadata[i - 1] if metadata and i <= len(metadata) else None
        chunk_label = _format_chunk_label(i, chunk_metadata)
        labeled_chunk = f"{chunk_label} {chunk}"
        chunk_size = len(labeled_chunk) + 2  # +2 for newlines

        # Check if adding this chunk would exceed budget (only when enabled)
        if (
            config.max_context_chars is not None
            and total_chars + chunk_size > config.max_context_chars
        ):
            remaining_chars = config.max_context_chars - total_chars - len(chunk_label) - 3
            truncated_chunk = ""
            if not context_parts and remaining_chars > 0:
                truncated_chunk = chunk[:remaining_chars].rsplit(" ", 1)[0].strip()
                if truncated_chunk:
                    context_parts.append(f"{chunk_label} {truncated_chunk}")
                    used_chunks.append(truncated_chunk)
            truncated = True
            logger.warning(
                f"Context truncated: {total_chars + chunk_size} chars would exceed "
                f"budget of {config.max_context_chars}. "
                f"{_format_truncation_summary(i - 1, len(chunks), i, len(truncated_chunk))}"
            )
            break

        context_parts.append(labeled_chunk)
        used_chunks.append(chunk)
        total_chars += chunk_size

    context = "\n\n".join(context_parts)
    # Auto-detect query type from metadata if not using custom role
    if custom_role:
        system_prompt = custom_role
    elif is_academic_query(metadata):
        system_prompt = ACADEMIC_SYSTEM_PROMPT
    else:
        system_prompt = SYSTEM_PROMPT

    return f"""{system_prompt}

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""


def is_academic_query(
    metadata: Optional[List[Dict[str, Any]]],
) -> bool:
    """Detect if retrieved chunks are primarily from academic sources.

    Checks if majority of chunks have source_category='academic_reference'.

    Args:
        metadata: List of metadata dicts with optional "source_category" field.

    Returns:
        True if majority of chunks are from academic sources, False otherwise.
    """
    if not metadata:
        return False

    academic_count = sum(
        1
        for meta in metadata
        if isinstance(meta, dict) and meta.get("source_category") == "academic_reference"
    )

    # Consider it academic if >50% of chunks are from academic sources
    return academic_count > len(metadata) / 2


def build_academic_aware_prompt(
    query: str,
    chunks: List[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
    custom_role: Optional[str] = None,
    additional_system_guidance: Optional[str] = None,
) -> str:
    """Build an academic-aware RAG prompt for research/thesis queries.

    Similar to build_prompt() but optimised for academic content.
    Includes paper titles, authors, institutions when available.
    Detects when all chunks are from the same thesis/dissertation and injects
    thesis-specific context to improve awareness.

    Args:
        query: User question about research papers or academic content.
        chunks: List of retrieved context chunks (paper excerpts).
        metadata: Optional list of metadata dicts for each chunk
                 (e.g., title, authors, institution, year).
        custom_role: Optional custom system prompt. If None, uses ACADEMIC_SYSTEM_PROMPT.
        additional_system_guidance: Optional instructions appended to the system prompt.

    Returns:
        Formatted prompt with academic-specific system prompt and metadata context.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if not chunks:
        raise ValueError("Context chunks cannot be empty")

    # Detect if all chunks are from the same thesis/academic work
    thesis_context = ""
    if metadata:
        # Extract unique doc_ids, titles, authors
        unique_docs = set()
        unique_titles = set()
        unique_authors = set()

        for meta in metadata:
            if meta:
                # Get doc_id (fallback to title if no doc_id)
                doc_id = meta.get("doc_id") or meta.get("title", "")
                if doc_id:
                    unique_docs.add(doc_id)

                # Collect display_name or title
                display_name = meta.get("display_name") or meta.get("title")
                if display_name:
                    unique_titles.add(display_name)

                # Collect author
                author = meta.get("author") or meta.get("authors")
                if author:
                    unique_authors.add(author)

        # If all chunks from single document, inject thesis context
        if len(unique_docs) == 1 and unique_titles:
            title = list(unique_titles)[0]
            author = list(unique_authors)[0] if unique_authors else "Unknown Author"

            # Extract year from metadata if available
            year = None
            for meta in metadata:
                if meta and meta.get("year"):
                    year = meta["year"]
                    break

            year_text = f" ({year})" if year else ""

            thesis_context = f"""
THESIS CONTEXT:
You are analysing excerpts from a single academic work:
Title: {title}
Author: {author}{year_text}

All retrieved chunks come from this thesis. When answering questions, you can reference
"this thesis", "the author", or "this research" rather than citing individual chunks
for general themes. However, still cite chunk numbers for specific claims or quotes.
"""

    # Build metadata context if provided
    metadata_context = ""
    if metadata:
        metadata_items = []
        for i, meta in enumerate(metadata, 1):
            if meta:
                # Extract academic-specific fields
                meta_parts = []

                # Use display_name if available, otherwise title
                display_name = meta.get("display_name")
                title = meta.get("title")
                if display_name:
                    meta_parts.append(f"source: {display_name}")
                elif title:
                    meta_parts.append(f"title: {title}")

                # Add other fields
                for field in ["authors", "author", "institution", "year", "doc_type"]:
                    if meta.get(field) and field not in [
                        "title"
                    ]:  # Skip title since we used display_name
                        meta_parts.append(f"{field}: {meta[field]}")

                if meta_parts:
                    metadata_items.append(f"[Chunk {i}] {', '.join(meta_parts)}")

        if metadata_items:
            metadata_context = "\nPAPER METADATA:\n" + "\n".join(metadata_items)

    # Label and assemble chunks with token budgeting
    context_parts: List[str] = []
    used_chunks: List[str] = []
    total_chars = 0

    for i, chunk in enumerate(chunks, 1):
        chunk_metadata = metadata[i - 1] if metadata and i <= len(metadata) else None
        chunk_label = _format_chunk_label(i, chunk_metadata)
        labeled_chunk = f"{chunk_label} {chunk}"
        chunk_size = len(labeled_chunk) + 2

        if (
            config.max_context_chars is not None
            and total_chars + chunk_size > config.max_context_chars
        ):
            remaining_chars = config.max_context_chars - total_chars - len(chunk_label) - 3
            truncated_chunk = ""
            if not context_parts and remaining_chars > 0:
                truncated_chunk = chunk[:remaining_chars].rsplit(" ", 1)[0].strip()
                if truncated_chunk:
                    context_parts.append(f"{chunk_label} {truncated_chunk}")
                    used_chunks.append(truncated_chunk)
            logger.warning(
                f"Academic context truncated: {total_chars + chunk_size} chars would exceed "
                f"budget of {config.max_context_chars}. "
                f"{_format_truncation_summary(i - 1, len(chunks), i, len(truncated_chunk))}"
            )
            break

        context_parts.append(labeled_chunk)
        used_chunks.append(chunk)
        total_chars += chunk_size

    context = "\n\n".join(context_parts)
    system_prompt = custom_role if custom_role else ACADEMIC_SYSTEM_PROMPT
    if additional_system_guidance:
        system_prompt = f"{system_prompt}\n\n{additional_system_guidance}"

    return f"""{system_prompt}{thesis_context}{metadata_context}

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
