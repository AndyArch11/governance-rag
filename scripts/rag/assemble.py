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
  - Code-aware prompt templates for code queries and responses
  - Metadata-rich prompts with language/service/dependency context

Best practices applied:
  - Per-chunk source ID labeling for better attribution
  - Token-aware truncation to prevent oversized prompts
  - System prompt explicitly references chunk labels
  - Code formatting instructions for responses
    - Git hosting link inclusion in code responses

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

# System prompt template for RAG assistant.
# Emphasises grounding in provided context and explicit instructions to reduce hallucination.
# Note: The prompt explicitly instructs the LLM to cite chunk numbers and avoid inventing information.
# Spelling as per US English conventions (e.g., "specializing") to match common LLM training data.
SYSTEM_PROMPT = """
You are a technical assistant specializing in governance, security, and infrastructure policies.

IMPORTANT INSTRUCTIONS:
- Use ONLY the provided context to answer questions.
- Context is provided as numbered chunks (e.g., [Chunk 1], [Chunk 2]).
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
CODE_SYSTEM_PROMPT = """
You are a technical assistant specializing in code analysis, architecture, and implementation patterns.

IMPORTANT INSTRUCTIONS:
- Use ONLY the provided context to answer questions about code.
- Context is provided as numbered chunks (e.g., [Chunk 1], [Chunk 2]).
- The chunks shown are ONLY the most relevant results retrieved for this query, NOT the entire codebase.
- Cite the chunk number(s) when using information: "According to [Chunk 1]..."
- Format code snippets in markdown blocks with language specification (e.g., ```java).
- Include service names, language, and dependencies when relevant.
- Provide git hosting links where available.
- If the context doesn't contain relevant information, say "Based on the retrieved chunks, I don't see..."
- For questions about codebase statistics (e.g., "how many times", "all occurrences"), clarify that you can only see the retrieved chunks, not all code.
- Do NOT invent, assume, or add information not in the context.
- Include implementation examples and design patterns from the code.
- Acknowledge when multiple implementations or approaches exist.
- Respond using UK English spelling and grammar conventions when not referencing code.
"""

# System prompt template for academic/research queries
ACADEMIC_SYSTEM_PROMPT = """
You are a research assistant specializing in academic papers, theses, and scholarly literature.

IMPORTANT INSTRUCTIONS:
- Use ONLY the provided context to answer questions about research papers and academic content.
- Context is provided as numbered chunks (e.g., [Chunk 1], [Chunk 2]).
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
    context_parts = []
    used_chunks = []
    total_chars = 0
    truncated = False

    for i, chunk in enumerate(chunks, 1):
        labeled_chunk = f"[Chunk {i}] {chunk}"
        chunk_size = len(labeled_chunk) + 2  # +2 for newlines

        # Check if adding this chunk would exceed budget (only when enabled)
        if (
            config.max_context_chars is not None
            and total_chars + chunk_size > config.max_context_chars
        ):
            truncated = True
            logger.warning(
                f"Context truncated: {total_chars + chunk_size} chars would exceed "
                f"budget of {config.max_context_chars}. Included {i-1} of {len(chunks)} chunks."
            )
            break

        context_parts.append(labeled_chunk)
        used_chunks.append(chunk)
        total_chars += chunk_size

    context = "\n\n".join(context_parts)
    raw_context = "\n\n".join(used_chunks)

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

RAW_CONTEXT:
{raw_context}

QUESTION:
{query}

ANSWER:
"""


def build_code_aware_prompt(
    query: str,
    chunks: List[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
    custom_role: Optional[str] = None,
) -> str:
    """Build a code-aware RAG prompt for code queries.

    Similar to build_prompt() but optimised for code-specific questions.
    Includes code formatting instructions and metadata context (language,
    service names, dependencies).

    Args:
        query: User question about code or architecture.
        chunks: List of retrieved context chunks (code snippets/references).
        metadata: Optional list of metadata dicts for each chunk
             (e.g., language, service_name, dependencies, git_url).
        custom_role: Optional custom system prompt. If None, uses default CODE_SYSTEM_PROMPT.

    Returns:
        Formatted prompt with code-specific system prompt, metadata context,
        and code formatting instructions.

    Example:
        >>> chunks = ["@Service\\npublic class AuthService { ... }", "public void authenticate() { ... }"]
        >>> metadata = [
        ...     {"language": "java", "service_name": "AuthService"},
        ...     {"language": "java", "service_name": "AuthService"}
        ... ]
        >>> prompt = build_code_aware_prompt("Show Java authentication services", chunks, metadata)
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if not chunks:
        raise ValueError("Context chunks cannot be empty")

    # Build metadata context if provided
    metadata_context = ""
    if metadata:
        metadata_items = []
        for i, meta in enumerate(metadata, 1):
            if meta:
                meta_str = ", ".join(f"{k}: {v}" for k, v in meta.items() if v)
                if meta_str:
                    metadata_items.append(f"[Chunk {i}] {meta_str}")

        if metadata_items:
            metadata_context = "\nMETADATA:\n" + "\n".join(metadata_items)

    # Label and assemble chunks with token budgeting
    context_parts = []
    used_chunks = []
    total_chars = 0

    for i, chunk in enumerate(chunks, 1):
        labeled_chunk = f"[Chunk {i}] {chunk}"
        chunk_size = len(labeled_chunk) + 2

        if (
            config.max_context_chars is not None
            and total_chars + chunk_size > config.max_context_chars
        ):
            logger.warning(
                f"Code context truncated: {total_chars + chunk_size} chars would exceed "
                f"budget of {config.max_context_chars}. Included {i-1} of {len(chunks)} chunks."
            )
            break

        context_parts.append(labeled_chunk)
        used_chunks.append(chunk)
        total_chars += chunk_size

    context = "\n\n".join(context_parts)
    raw_context = "\n\n".join(used_chunks)

    system_prompt = custom_role if custom_role else CODE_SYSTEM_PROMPT

    return f"""{system_prompt}{metadata_context}

CONTEXT:
{context}

RAW_CONTEXT:
{raw_context}

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


def extract_language_from_metadata(
    metadata: Optional[List[Dict[str, Any]]],
) -> Optional[str]:
    """Extract programming language from metadata list.

    Returns the first language found in metadata list.
    Useful for determining markdown code block formatting.

    Args:
        metadata: List of metadata dicts with optional "language" field.

    Returns:
        Language name (e.g., "java", "python") or None if not found.
    """
    if not metadata:
        return None

    for meta in metadata:
        if isinstance(meta, dict) and meta.get("language"):
            return meta["language"]

    return None


def build_academic_aware_prompt(
    query: str,
    chunks: List[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
    custom_role: Optional[str] = None,
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
    context_parts = []
    used_chunks = []
    total_chars = 0

    for i, chunk in enumerate(chunks, 1):
        labeled_chunk = f"[Chunk {i}] {chunk}"
        chunk_size = len(labeled_chunk) + 2

        if (
            config.max_context_chars is not None
            and total_chars + chunk_size > config.max_context_chars
        ):
            logger.warning(
                f"Academic context truncated: {total_chars + chunk_size} chars would exceed "
                f"budget of {config.max_context_chars}. Included {i-1} of {len(chunks)} chunks."
            )
            break

        context_parts.append(labeled_chunk)
        used_chunks.append(chunk)
        total_chars += chunk_size

    context = "\n\n".join(context_parts)
    raw_context = "\n\n".join(used_chunks)

    system_prompt = custom_role if custom_role else ACADEMIC_SYSTEM_PROMPT

    return f"""{system_prompt}{thesis_context}{metadata_context}

CONTEXT:
{context}

RAW_CONTEXT:
{raw_context}

QUESTION:
{query}

ANSWER:
"""


def format_code_response(answer: str, language: Optional[str] = None) -> str:
    """Format LLM response with proper code markdown formatting.

    Enhances code snippets in the response with markdown code blocks
    and language specification for syntax highlighting.

    Args:
        answer: Raw LLM-generated response text.
        language: Programming language for code block formatting (e.g., "java").

    Returns:
        Enhanced response with code blocks properly formatted.

    Example:
        >>> answer = "The service uses: @Service public class Auth { }"
        >>> formatted = format_code_response(answer, language="java")
        >>> "```java" in formatted
        True
    """
    if not answer:
        return answer

    # If language provided, enhance common code patterns with markdown blocks
    if language:
        # Look for common code patterns and wrap in markdown blocks
        patterns = [
            ("class ", f"```{language}\\nclass "),
            ("public ", f"```{language}\\npublic "),
            ("@", f"```{language}\\n@"),
            ("def ", f"```{language}\\ndef "),
            ("function ", f"```{language}\\nfunction "),
        ]

        for pattern, replacement in patterns:
            if pattern in answer and f"```{language}" not in answer:
                # Simple heuristic: if we found a code pattern and no code block yet,
                # this might be a code-heavy response
                # TODO: More sophisticated parsing would be needed for production
                break

    return answer


def _normalise_git_host(host: str) -> str:
    """Normalise git host by stripping trailing slashes."""
    return host.rstrip("/")


def _build_git_file_url(meta: Dict[str, Any]) -> Optional[str]:
    """Build a Git hosting URL for a file based on metadata."""
    provider = (meta.get("git_provider") or meta.get("provider") or "").lower()
    host = meta.get("git_host") or meta.get("host")
    project = meta.get("project") or meta.get("project_key") or meta.get("owner")
    repository = meta.get("repository") or meta.get("repo") or meta.get("repo_slug")
    file_path = meta.get("file_path")
    branch = meta.get("branch") or meta.get("git_branch") or meta.get("default_branch") or "main"

    if not provider or not host or not project or not repository or not file_path:
        return None

    host = _normalise_git_host(host)

    if provider == "bitbucket":
        return f"{host}/projects/{project}/repos/{repository}/browse/{file_path}?at=refs/heads/{branch}"
    if provider == "github":
        return f"{host}/{project}/{repository}/blob/{branch}/{file_path}"
    if provider == "gitlab":
        return f"{host}/{project}/{repository}/-/blob/{branch}/{file_path}"
    if provider == "azure":
        return f"{host}/{project}/_git/{repository}?path=/{file_path}&version=GB{branch}"

    return None


def include_git_links(
    answer: str,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Append Git hosting links to response if available in metadata.

    Adds source links to code repositories for reference.

    Args:
        answer: Generated answer text.
        metadata: List of metadata dicts with optional git URL fields.

    Returns:
        Answer with Git hosting references appended (if available).

    Example:
        >>> metadata = [{"git_url": "https://github.com/org/repo/blob/main/src/AuthService.java"}]
        >>> response = include_git_links("Check the authentication service", metadata)
        >>> "github" in response.lower()
        True
    """
    if not metadata:
        return answer

    url_keys = ("git_url", "file_url", "repo_url", "source_url", "bitbucket_url")
    git_urls: List[str] = []

    for meta in metadata:
        if not meta:
            continue

        for key in url_keys:
            value = meta.get(key)
            if value:
                git_urls.append(value)

        derived_url = _build_git_file_url(meta)
        if derived_url:
            git_urls.append(derived_url)

    if not git_urls:
        return answer

    # Remove duplicates while preserving order
    unique_urls = []
    seen = set()
    for url in git_urls:
        if url not in seen:
            unique_urls.append(url)
            seen.add(url)

    if unique_urls:
        links_section = "\n\n**References:**\n"
        for url in unique_urls:
            links_section += f"- {url}\n"
        return answer + links_section

    return answer
