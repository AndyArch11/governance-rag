"""Text preprocessing and metadata extraction using LLM.

Handles:
- Text cleaning and normalisation
- LLM-based metadata generation (doc_type, topics, summary)
- JSON extraction and repair from LLM outputs
- Validation using Pydantic schemas
- Summary quality scoring

JSON extraction/repair is delegated to shared utilities in
scripts.utils.json_utils, with thin wrappers here to preserve ingest
logging and retry semantics expected by tests.

Uses a configured Ollama LLM for all generation and validation tasks.

TODO: Consider breaking this into multiple modules (e.g., cleaning.py, metadata.py, validation.py) for better separation of concerns and testability.
TODO: Consider adding more granular logging and metrics for each stage of preprocessing for better observability and debugging.
TODO: Currently built synchronously for simplicity, but could be made asynchronous if needed for performance with large documents or high volume. LLM calls would need to be adapted to async and rate limiting would need to be compatible with async operations.
TODO: Currently focused on a single LLM model, but could be extended to support multiple models (e.g., different models for cleaning vs metadata extraction) with configuration and dynamic selection logic.
TODO: Currently implemented against Ollama, but could be abstracted to support multiple LLM providers with a common interface for easier switching and testing.
"""

import json
import logging
import re
import textwrap
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from langchain_ollama import OllamaLLM
from pydantic import ValidationError

from scripts.ingest.ingest_config import get_ingest_config
from scripts.security.dlp import DLPScanner
from scripts.utils.json_utils import extract_first_json_block as utils_extract_first_json_block
from scripts.utils.json_utils import repair_json as utils_repair_json
from scripts.utils.json_utils import (
    sanitise_for_json,
)
from scripts.utils.logger import create_module_logger
from scripts.utils.retry_utils import retry_ollama_call

get_logger, audit = create_module_logger("ingest")
from scripts.utils.schemas import MetadataSchema, SummarySchema

if TYPE_CHECKING:
    from scripts.utils.rate_limiter import RateLimiter

    from .llm_cache import LLMCache

# Centralised ingest configuration
CONFIG = get_ingest_config()

# Primary LLM for content generation
primary_llm = OllamaLLM(model=CONFIG.llm_model_name)

# Separate LLM instance for validation and repair
# Maintains isolation between generation and validation
# For greater separation, consider using a different model
validator_llm = OllamaLLM(model=CONFIG.validator_llm_model_name)

# Shared DLP scanner instance for preprocessing
_dlp_scanner = DLPScanner()

# Rate limiter for LLM calls (initialised on first use)
_rate_limiter: Optional["RateLimiter"] = None


def _get_rate_limiter() -> Optional["RateLimiter"]:
    """Get or initialise the rate limiter for LLM calls.

    Uses lazy initialisation to avoid import issues.
    Returns None if rate limiter module is not available.
    """
    global _rate_limiter
    if _rate_limiter is None:
        try:
            from scripts.utils.rate_limiter import get_rate_limiter

            _rate_limiter = get_rate_limiter()
        except (ImportError, RuntimeError):
            # Rate limiter not initialised yet or not available
            return None
    return _rate_limiter


def redact_sensitive_text(text: str, doc_hash: Optional[str] = None) -> Tuple[str, Dict[str, int]]:
    """Redact sensitive patterns before further processing.

    Applies default DLP rules and returns the redacted text plus counts for each pattern.
    """
    logger = get_logger()

    matches = _dlp_scanner.find(text)
    if not matches:
        return text, {}

    counts = {name: len(items) for name, items in matches.items()}
    redacted = _dlp_scanner.redact(text)

    logger.info(f"DLP redaction applied (doc_hash={doc_hash}): {counts}")
    audit("dlp_redaction_applied", {"doc_hash": doc_hash, "pattern_counts": counts})

    return redacted, counts


@retry_ollama_call(max_retries=3, initial_delay=1.0, operation_name="llm_invoke_with_rate_limit")
def llm_invoke_with_rate_limit(llm: OllamaLLM, prompt: str, timeout: float = 120.0) -> str:
    """Invoke LLM with rate limiting and retry logic.

    Automatically retries transient failures (connection errors, timeouts,
    rate limits) with exponential backoff. Hard failures (validation errors)
    fail immediately without retry.

    Args:
        llm: LLM instance to invoke.
        prompt: Prompt text to send to LLM.
        timeout: Timeout in seconds (for consistency with other async operations).

    Returns:
        LLM response.

    Raises:
        Exception: On hard failure or after all retries exhausted.
    """
    limiter = _get_rate_limiter()
    if limiter:
        # Acquire 1 token (represents 1 LLM call)
        limiter.acquire(tokens=1, blocking=True)

    return llm.invoke(prompt)


# Wrapper preserves previous logging + validation semantics
def repair_json(text: str, attempt: int = 1, max_attempts: int = 3) -> str:
    """Repair JSON with retries and log failures.

    Wraps utils_repair_json to keep ingest-specific logging/validation
    expected by tests and downstream code.

    Args:
        text: JSON string to repair.
        attempt: Current attempt number (for logging). Starts at 1.
        max_attempts: Maximum number of repair attempts before giving up.

    Returns:
        Repaired JSON string.

    Raises:
        json.JSONDecodeError: If JSON is still invalid after max attempts.
    """
    logger = get_logger()

    # Apply utility repair first
    repaired = utils_repair_json(text, attempt=attempt, max_attempts=max_attempts)

    # Validate repaired JSON; retry with simpler cleanup on failure
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError as e:
        if attempt < max_attempts:
            logger.debug(f"JSON repair attempt {attempt} failed, retrying (error: {str(e)[:100]})")
            simpler = re.sub(r",\s*([}\]])", r"\1", repaired)
            return repair_json(simpler, attempt + 1, max_attempts)
        else:
            logger.exception(f"JSON repair failed after {max_attempts} attempts")
            logger.debug(f"Final malformed JSON: {repaired[:500]}...")
            raise


def extract_first_json_block(text: str, max_repair_attempts: int = 3) -> Dict[str, Any]:
    """Wrapper to use shared util while keeping ingest audit/logging.

    Args:
        text: Text containing JSON block to extract.
        max_repair_attempts: Maximum number of repair attempts before giving up.

    Returns:
        Extracted JSON block as a dictionary.

    Raises:
        ValueError: If no JSON block is found.
        json.JSONDecodeError: If JSON repair fails after all attempts.
    """
    logger = get_logger()
    try:
        return utils_extract_first_json_block(text, max_repair_attempts=max_repair_attempts)
    except ValueError as e:
        # Preserve previous behaviour: propagate ValueError
        raise
    except json.JSONDecodeError as e:
        # Log failure once for visibility
        logger.exception("JSON repair failed in extract_first_json_block")
        raise


def validate_metadata(metadata: Dict[str, Any]) -> MetadataSchema:
    """Validate document metadata against Pydantic schema.

    Ensures metadata contains required fields with proper types:
    - doc_type: Non-empty string
    - key_topics: List of strings
    - summary: Non-empty string

    Args:
        metadata: Dictionary with doc_type, key_topics, and summary.

    Returns:
        Validated MetadataSchema instance.

    Raises:
        ValidationError: If metadata fails schema validation.
            Logs error details and audit trail before raising.
    """
    logger = get_logger()

    try:
        return MetadataSchema(**metadata)
    except ValidationError as e:
        logger.error(f"Metadata validation failed: {e}")
        audit("metadata_validation_error", {"errors": e.errors(), "metadata": metadata})
        raise


def validate_summary(summary: str) -> SummarySchema:
    """Validate document summary against quality requirements.

    Ensures summary is:
    - At least 30 characters long
    - Contains at least 5 words
    - Not trivial placeholder text

    Args:
        summary: Summary text to validate.

    Returns:
        Validated SummarySchema instance.

    Raises:
        ValidationError: If summary fails validation.
            Logs error details and audit trail before raising.
    """
    logger = get_logger()

    try:
        return SummarySchema(summary=summary)
    except ValidationError as e:
        logger.error(f"Summary validation failed: {e}")
        audit("summary_validation_error", {"errors": e.errors(), "summary": summary})
        raise


def get_LLM_validator() -> OllamaLLM:
    """Get LLM instance for validation and repair tasks.

    Returns separate validator LLM instance to maintain isolation
    between content generation and validation operations.

    Returns:
        Configured LLM instance for validation.
    """
    return validator_llm


def score_summary(
    summary: str,
    cleaned_text: str,
    doc_hash: Optional[str] = None,
    llm_cache: Optional["LLMCache"] = None,
) -> Dict[str, Any]:
    """Evaluate summary quality using LLM-based scoring.

    Asks LLM to assess summary across multiple dimensions:
    - Relevance: How well summary reflects actual content (0-10)
    - Coverage: How well it covers main points (0-10)
    - Clarity: How easy to understand (0-10)
    - Conciseness: Efficiency without losing meaning (0-10)
    - Overall: Combined quality score (0-10)

    Args:
        summary: Summary text to evaluate.
        cleaned_text: Original cleaned document text for comparison.
        doc_hash: Optional document hash for caching.
        llm_cache: Optional LLM cache instance.

    Returns:
        Dictionary with score breakdown:
            - relevance (int): Relevance score
            - coverage (int): Coverage score
            - clarity (int): Clarity score
            - conciseness (int): Conciseness score
            - overall (int): Overall quality score
            - comment (str): Explanation of scores

    Example:
        >>> scores = score_summary(
        ...     "Security policy requires MFA.",
        ...     "All users must enable multi-factor authentication..."
        ... )
        >>> scores['overall']
        8
    """
    # Check cache first
    if doc_hash and llm_cache:
        cached = llm_cache.get(doc_hash, "score_summary")
        if cached is not None:
            return cached

    prompt = textwrap.dedent(f"""
    You are evaluating the quality of a summary for a technical governance document.

    Criteria (0-10, integers):
    - relevance: how well the summary reflects the actual content
    - coverage: how well it covers the main points
    - clarity: how easy it is to understand
    - conciseness: how efficient it is without losing meaning

    Respond ONLY with JSON:
    {{
      "relevance": 0-10,
      "coverage": 0-10,
      "clarity": 0-10,
      "conciseness": 0-10,
      "overall": 0-10,
      "comment": "short explanation"
    }}

    SUMMARY:
    \"\"\"{summary}\"\"\"

    DOCUMENT EXCERPT (may be truncated):
    \"\"\"{cleaned_text[:4000]}\"\"\"
    """)

    raw = validator_llm.invoke(prompt)
    result = extract_first_json_block(raw)

    # Store in cache
    if doc_hash and llm_cache:
        llm_cache.put(doc_hash, "score_summary", result)

    return result


def regenerate_summary(
    cleaned_text: str, doc_hash: Optional[str] = None, llm_cache: Optional["LLMCache"] = None
) -> str:
    """Generate a new high-quality summary using LLM.

    Used when initial summary scores poorly. Instructs LLM to create
    factual, concise summary focusing on purpose, scope, and key points.

    Args:
        cleaned_text: Preprocessed document text.
        doc_hash: Optional document hash for caching.
        llm_cache: Optional LLM cache instance.

    Returns:
        Generated summary text (3-6 sentences).

    Note:
        Does not invent requirements or policies - stays factual.
    """
    # Check cache first
    if doc_hash and llm_cache:
        cached = llm_cache.get(doc_hash, "regenerate_summary")
        if cached is not None:
            return cached

    prompt = textwrap.dedent(f"""
    You are generating a high-quality summary for a technical governance document.

    RULES:
    - Capture the main purpose, scope, and key responsibilities.
    - Do NOT invent new requirements or policies.
    - Keep it factual and concise.
    - 3-6 sentences maximum.

    TEXT:
    {cleaned_text}

    Return ONLY the summary text.
    """)
    result = primary_llm.invoke(prompt).strip()

    # Store in cache
    if doc_hash and llm_cache:
        llm_cache.put(doc_hash, "regenerate_summary", result)

    return result


def clean_text_with_llm(
    raw_text: str, doc_hash: Optional[str] = None, llm_cache: Optional["LLMCache"] = None
) -> str:
    """Clean and normalise text using LLM while preserving meaning.

    Removes navigation, boilerplate, and UI elements without summarizing
    or rewriting. Makes text company- and people-agnostic by removing
    references to company entity names and individual names.

    Args:
        raw_text: Raw extracted text from HTML.
        doc_hash: Optional document hash for caching.
        llm_cache: Optional LLM cache instance.

    Returns:
        Cleaned text with original meaning preserved, sanitised for JSON.

    Cleaning Rules:
        - Removes: Navigation, headers/footers, boilerplate, duplicates
        - Removes: Company references
        - Removes: People references (@ mentions, names)
        - Preserves: All technical content, requirements, policies
        - Does NOT: Summarise, rewrite, reorder, or shorten

    Note:
        Output is sanitised for JSON embedding (escapes quotes/backslashes).
    """

    # Check cache first
    if doc_hash and llm_cache:
        cached = llm_cache.get(doc_hash, "clean_text")
        if cached is not None:
            return cached

    # Stage 1: clean / normalise text, no JSON.
    # TODO: For large documents, consider chunking and cleaning in parts to avoid LLM input limits.
    # TODO: Add examples of boilerplate vs technical content to prompt for better accuracy.
    # TODO: Consider a two-pass approach where we first identify boilerplate sections, then remove them in a second pass to preserve more technical content.
    # TODO: Have explicit instructions to remove company/people references to ensure output is anonymised as an option rather than hard-coded.
    clean_prompt = textwrap.dedent(f"""
    You are a text normalisation assistant.

    Your job is to CLEAN the text, not summarise it.

    RULES:
    - Do NOT summarise.
    - Do NOT rewrite.
    - Do NOT add new sentences.
    - Do NOT remove technical content.
    - Do NOT interpret meaning.
    - Do NOT reorder content.
    - Do NOT shorten content.
    - Do NOT include references to company names. Keep text output company agnostic.
    - Do NOT include references to people including @ references. Keep text output people agnostic.
    - ONLY remove:
      - navigation menus
      - headers/footers
      - boilerplate
      - duplicated text
      - irrelevant UI elements
      - company references
      - references to people including @ references

    Return ONLY the cleaned text, with all original meaning preserved.

    Text to clean:
    {raw_text}
    """)

    cleaned_text = primary_llm.invoke(clean_prompt)
    cleaned_text = sanitise_for_json(cleaned_text)

    # Cache result
    if doc_hash and llm_cache:
        llm_cache.put(doc_hash, "clean_text", cleaned_text)

    return cleaned_text


def extract_metadata_with_llm(
    cleaned_text: str,
    source_category: Optional[str] = None,
    doc_hash: Optional[str] = None,
    llm_cache: Optional["LLMCache"] = None,
) -> Dict[str, Any]:
    """Extract structured metadata from document using LLM.

    Asks LLM to analyse text and extract:
    - doc_type: Document category/type
    - key_topics: List of main topics
    - summary: Concise document summary

    Uses deterministic few-shot template with strict JSON skeleton to improve
    parsing reliability and reduce repair attempts.

    Args:
        cleaned_text: Cleaned and normalised document text.
        source_category: Optional category hint from folder structure (e.g., 'Governance', 'Patterns').
        doc_hash: Optional document hash for caching.
        llm_cache: Optional LLM cache instance.

    Returns:
        Dictionary with doc_type, key_topics, and summary.

    Raises:
        Exception: If JSON extraction fails after repair attempts.

    Note:
        Uses deterministic template with examples and strict schema.
        Raw output logged only once on final failure to avoid noise.
    """
    logger = get_logger()

    # Check cache first
    if doc_hash and llm_cache:
        cached = llm_cache.get(doc_hash, "metadata")
        if cached is not None:
            return cached

    # Build category context if provided
    category_context = ""
    if source_category:
        category_context = f"This document is from: {source_category}"

    # Deterministic few-shot template with strict JSON skeleton
    metadata_prompt = textwrap.dedent(
        f"""You are an information extraction system. Extract metadata from the given text.
    
{category_context if category_context else 'Extract key information from the document.'}

RETURN ONLY VALID JSON - no other text before or after.

JSON format (required fields only):
{{
  "doc_type": "<single category: guide/policy/architecture/other>",
  "key_topics": ["<topic1>", "<topic2>", "<topic3>"],
  "summary": "<2-3 sentence concise summary>"
}}

Examples:

Example 1:
{{
  "doc_type": "guide",
  "key_topics": ["onboarding", "best practices", "requirements"],
  "summary": "Guide for new team members covering onboarding procedures and best practices."
}}

Example 2:
{{
  "doc_type": "policy",
  "key_topics": ["security", "access control", "authentication"],
  "summary": "Security policy defining access control requirements and authentication procedures."
}}

Now extract metadata from this text:

TEXT:
{cleaned_text}

JSON (no markdown, no extra text):"""
    )

    metadata_json = primary_llm.invoke(metadata_prompt)
    try:
        # Use max 3 repair attempts for JSON parsing
        metadata = extract_first_json_block(metadata_json, max_repair_attempts=3)

        # Store in cache if available
        if doc_hash and llm_cache:
            llm_cache.put(doc_hash, "metadata", metadata)

        logger.debug(f"Metadata extracted: doc_type={metadata.get('doc_type')}")
        return metadata
    except Exception as e:
        logger.exception("Failed to extract metadata")
        logger.error(f"Failed to extract metadata: {str(e)[:100]}")
        logger.debug(
            f"LLM output: {metadata_json[:300]}..."
            if len(metadata_json) > 300
            else f"LLM output: {metadata_json}"
        )
        raise


def preprocess_text(
    raw_text: str,
    source_category: str = None,
    doc_hash: Optional[str] = None,
    llm_cache: Optional["LLMCache"] = None,
) -> Dict[str, Any]:
    """Complete preprocessing pipeline: clean, extract metadata, validate.

    Multi-stage process:
    0. Redact sensitive patterns (DLP)
    1. Clean text with LLM (remove boilerplate, normalise)
    2. Extract metadata (doc_type, topics, summary)
    3. Validate metadata against schema
    4. Score summary quality
    5. Regenerate summary if quality is too low

    Args:
        raw_text: Raw text extracted from HTML document.
        source_category: Optional category hint from folder structure (e.g., 'Governance', 'Patterns').
        doc_hash: Optional document hash for caching.
        llm_cache: Optional LLM cache instance.

    Returns:
        Dictionary containing:
            - cleaned_text (str): Normalised document text
            - doc_type (str): Document category
            - key_topics (List[str]): Main topics
            - summary (str): Document summary
            - summary_scores (Dict): Quality scores for summary
            - source_category (str): Category hint if provided

    Side Effects:
        - Logs validation and scoring events
        - Creates audit trail
        - May regenerate summary if quality is poor

    Quality Threshold:
        MIN_SUMMARY_SCORE = 5 (on 0-10 scale)
        Summaries below threshold trigger regeneration.
    """
    logger = get_logger()

    # Apply DLP redaction before any LLM calls to avoid leaking sensitive data
    redacted_text, dlp_counts = redact_sensitive_text(raw_text, doc_hash)
    cleaned_text = clean_text_with_llm(redacted_text, doc_hash, llm_cache)
    metadata_raw = extract_metadata_with_llm(cleaned_text, source_category, doc_hash, llm_cache)

    # Validate metadata is compliant to expected schema
    metadata = validate_metadata(metadata_raw)

    # Validate summary
    validate_summary(metadata.summary)

    # Summary quality scoring
    summary_scores = score_summary(metadata.summary, cleaned_text, doc_hash, llm_cache)
    overall = summary_scores.get("overall", 0)

    audit(
        "summary_scored",
        {"doc_type": metadata.doc_type, "overall": overall, "scores": summary_scores},
    )

    # Enforce a minimum quality threshold
    # TODO - consider regenerating summary, accept summary as is but flag, or fail ingestion for this doc
    MIN_SUMMARY_SCORE = 5
    if overall < MIN_SUMMARY_SCORE:
        logger.warning(f"Low summary quality for doc_type={metadata.doc_type}, overall={overall}")
        audit("summary_low_quality", {"doc_type": metadata.doc_type, "scores": summary_scores})

        regenerated = regenerate_summary(cleaned_text, doc_hash, llm_cache)

        # validate regenerated summary
        validate_summary(regenerated)

        # Re-score regenerated summary
        regenerated_scores = score_summary(regenerated, cleaned_text, doc_hash, llm_cache)
        regenerated_overall = regenerated_scores.get("overall", 0)

        audit(
            "summary_regenerated",
            {
                "doc_type": metadata.doc_type,
                "old_score": overall,
                "new_score": regenerated_overall,
                "new_scores": regenerated_scores,
            },
        )

        if regenerated_overall > overall:
            logger.info(f"Summary improved from {overall} → {regenerated_overall}")
            metadata.summary = regenerated
            summary_scores = regenerated_scores
            overall = regenerated_overall
        else:
            logger.info(
                f"Regenerated was worse than original {overall} → {regenerated_overall}, keeping original."
            )

    # Detect and analyse tables in content
    cleaned_text_with_analysis, table_metadata = detect_and_mark_tables(cleaned_text)

    if table_metadata["has_tables"]:
        logger.info(f"Detected {table_metadata['table_count']} tables in document")
        audit(
            "tables_detected",
            {
                "table_count": table_metadata["table_count"],
                "table_sizes": table_metadata["table_sizes"],
            },
        )

    return {
        "cleaned_text": cleaned_text,
        "doc_type": metadata.doc_type,
        "key_topics": metadata.key_topics,
        "summary": metadata.summary,
        "summary_scores": summary_scores,
        "source_category": source_category,
        "table_metadata": table_metadata,
        "dlp_pattern_counts": dlp_counts,
    }


def detect_and_mark_tables(text: str) -> Tuple[str, Dict[str, Any]]:
    """Detect tables in preprocessed text and enhance metadata.

    Identifies table markers ([TABLE N] ... [/TABLE N]) inserted by
    HTML parser and collects statistics for better chunking strategy.

    Args:
        text: Preprocessed text potentially containing table markers.

    Returns:
        Tuple of (text, table_metadata) where:
        - text: Original text (unchanged)
        - table_metadata: Dict with:
            - table_count: Number of tables detected
            - has_tables: Boolean indicating presence of tables
            - tables_start_pos: Character position of [TABLES START]
            - table_sizes: List of (table_index, approx_char_count)
    """
    table_metadata = {
        "table_count": 0,
        "has_tables": False,
        "tables_start_pos": -1,
        "table_sizes": [],
    }

    # Check for table markers
    if "[TABLES START]" not in text or "[TABLES END]" not in text:
        return text, table_metadata

    start_pos = text.find("[TABLES START]")
    end_pos = text.find("[TABLES END]")

    if start_pos < 0 or end_pos < 0:
        return text, table_metadata

    tables_section = text[start_pos : end_pos + len("[TABLES END]")]
    table_metadata["has_tables"] = True
    table_metadata["tables_start_pos"] = start_pos

    # Count and measure tables
    table_pattern = r"\[TABLE (\d+)\].*?\[/TABLE \1\]"
    matches = re.finditer(table_pattern, tables_section, re.DOTALL)

    for match in matches:
        table_index = int(match.group(1))
        table_text = match.group(0)
        table_size = len(table_text)
        table_metadata["table_sizes"].append((table_index, table_size))
        table_metadata["table_count"] += 1

    return text, table_metadata


def extract_table_context(text: str, table_index: int, context_lines: int = 5) -> Optional[str]:
    """Extract context around a specific table.

    Useful for understanding table purpose and relevance.

    Args:
        text: Full preprocessed text
        table_index: Index of table to extract context for
        context_lines: Number of lines before table to include

    Returns:
        Context string or None if table not found
    """
    pattern = rf"\[TABLE {table_index}\]"
    match = re.search(pattern, text)

    if not match:
        return None

    # Find context before table
    text_before = text[: match.start()]
    context_lines_list = text_before.split("\n")

    # Get last N lines before table
    relevant_context = "\n".join(context_lines_list[-context_lines:])

    return relevant_context.strip()
