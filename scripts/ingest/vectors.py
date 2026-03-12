"""Vector embedding and document health management for RAG ingestion.

This module handles:
- Document chunk validation and semantic quality checks
- LLM-based chunk repair for invalid or truncated content
- Semantic drift detection between document versions
- Document health scoring and metrics
- Vector embedding generation and storage in ChromaDB
- Version metadata management and retrieval

The module uses Ollama LLMs for semantic validation and embeddings
for document representation in vector space.
"""

import datetime
import hashlib
import json
import math
import os
import re
import textwrap
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from langchain_ollama import OllamaEmbeddings
from pydantic import ValidationError

from scripts.utils.retry_utils import retry_chromadb_call, retry_ollama_call

# Collection typing (best-effort)
try:
    from chromadb.api.models.Collection import (  # type: ignore  # noqa: WPS433,E402
        Collection as ChromaDBCollection,
    )
    from chromadb.errors import InvalidArgumentError
except Exception:
    ChromaDBCollection = Any  # type: ignore

    class InvalidArgumentError(Exception):  # Fallback exception
        pass


try:
    from .chromadb_sqlite import ChromaSQLiteCollection  # noqa: WPS433,E402
except Exception:
    ChromaSQLiteCollection = Any  # type: ignore

Collection = Union[ChromaDBCollection, ChromaSQLiteCollection, Any]
CollectionType = Collection  # Alias for backward compatibility

from scripts.utils.logger import create_module_logger
from scripts.utils.logger import get_logger as get_global_logger

from .embedding_cache import EmbeddingCache

# Shared logger for all ingest operations
get_logger, audit = create_module_logger("ingest")
from scripts.utils.adaptive_cache_tuning import (
    get_adaptive_cache_tuner,
    init_adaptive_cache_tuner,
)
from scripts.utils.adaptive_rate_limiter import (
    get_adaptive_rate_limiter,
    init_adaptive_rate_limiter,
)
from scripts.utils.json_utils import extract_first_json_block
from scripts.utils.metrics_export import get_metrics_collector
from scripts.utils.monitoring import get_perf_metrics, get_token_counter, init_monitoring
from scripts.utils.schemas import ChunkSchema

from .llm_cache import LLMCache
from .preprocess import get_LLM_validator

# Import enhanced metadata for RAG optimisation
try:
    from .chunk import create_enhanced_metadata
except ImportError:
    create_enhanced_metadata = None  # Gracefully handle if not available

if TYPE_CHECKING:
    pass

# Cosine similarity threshold for filtering low-similarity documents
# Tune based on your use case: higher = stricter matching
SIMILARITY_THRESHOLD: float = 0.4

# Embedding model identifier (used for metadata and cache namespace)
# mxbai-embed-large: 1024-dim, better semantic understanding than nomic-embed-text
EMBEDDING_MODEL_NAME: str = "mxbai-embed-large"

# Expected embedding dimension for EMBEDDING_MODEL_NAME
# Used for validation: mxbai-embed-large = 1024, all-minilm = 384
EXPECTED_EMBEDDING_DIMENSION: dict = {
    "mxbai-embed-large": 1024,
    "nomic-embed-text": 768,
    "all-minilm": 384,
}
EXPECTED_EMBEDDING_DIM: int = EXPECTED_EMBEDDING_DIMENSION.get(EMBEDDING_MODEL_NAME, 1024)

# Hard limit to avoid exceeding ChromaDB's max batch size (default ~5461)
CHROMADB_ADD_BATCH_LIMIT: int = 5000

# Embedding model context length (maximum tokens it can process)
# mxbai-embed-large has a 512-token context limit
# We use a conservative limit (300 tokens) to account for:
# - Token estimation variance (academic text has more tokens per char)
# - Batch processing overhead
# - Safety margin for edge cases
EMBEDDING_MODEL_MAX_TOKENS: dict = {
    "mxbai-embed-large": 512,
    "nomic-embed-text": 2048,
    "all-minilm": 256,
}
EMBEDDING_MODEL_MAX_TOKEN_LIMIT: int = EMBEDDING_MODEL_MAX_TOKENS.get(EMBEDDING_MODEL_NAME, 512)

# Conservative usable token limit (use only 60% of actual limit)
# This accounts for: special chars, punctuation, multilingual content
EMBEDDING_USABLE_TOKEN_LIMIT: int = max(100, int(EMBEDDING_MODEL_MAX_TOKEN_LIMIT * 0.6))

# Safety margin (in characters) to account for token estimation variance
# Approximate token count: ~1 token per 3 characters (more conservative than 4)
# This gives us: max_chars = (usable_tokens * 3) - margin
EMBEDDING_CONTEXT_SAFETY_MARGIN: int = 200

# Initialise monitoring
init_monitoring()


def _create_embed_model() -> OllamaEmbeddings:
    """Create an embedding model with conservative context settings.

    Forces Ollama context size to the model limit and disables keep-alive
    to avoid potential state accumulation across rapid embedding calls.
    """
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        num_ctx=EMBEDDING_MODEL_MAX_TOKEN_LIMIT,
        keep_alive=0,
    )


def _embed_documents_with_fallback(
    embed_model: OllamaEmbeddings, texts: List[str]
) -> List[List[float]]:
    """Embed documents with a fallback path that enables server-side truncation.

    Ollama occasionally returns context-length errors despite short inputs.
    If that happens, retry via the native client with truncate=True.
    """
    import time

    try:
        return embed_model.embed_documents(texts)
    except Exception as exc:
        exc_str = str(exc)
        # Check for model initialisation error - intermittent Ollama server issue
        if "failed to initialise model" in exc_str or "no such file" in exc_str:
            logger = get_logger()
            logger.warning(
                f"Ollama model initialisation failed (intermittent error). Retrying after 2s delay. "
                f"Error: {exc_str}"
            )
            # Brief delay then retry - often works on second attempt
            time.sleep(2)
            try:
                return embed_model.embed_documents(texts)
            except Exception as retry_exc:
                logger.error(
                    f"Ollama model initialisation failed on retry. Model '{EMBEDDING_MODEL_NAME}' may not be loaded. "
                    f"Error: {str(retry_exc)}"
                )
                raise RuntimeError(
                    f"Ollama model '{EMBEDDING_MODEL_NAME}' failed to initialise after retry. "
                    "Ensure the model is available on the Ollama server."
                ) from retry_exc
        if "context length" not in exc_str:
            raise

        logger = get_logger()
        logger.warning(
            "Ollama context-length error despite short input; retrying with truncate=True"
        )
        try:
            from ollama import Client

            client = Client()
            response = client.embed(
                model=EMBEDDING_MODEL_NAME,
                input=texts,
                truncate=True,
                options={"num_ctx": EMBEDDING_MODEL_MAX_TOKEN_LIMIT},
            )
            return response["embeddings"]
        except Exception as retry_exc:
            # Hard truncate and retry once more
            logger.warning("Retry with truncate=True failed; applying hard truncation and retrying")
            hard_truncated = [t[:256] for t in texts]
            try:
                response = client.embed(
                    model=EMBEDDING_MODEL_NAME,
                    input=hard_truncated,
                    truncate=True,
                    options={"num_ctx": EMBEDDING_MODEL_MAX_TOKEN_LIMIT},
                )
                return response["embeddings"]
            except Exception:
                logger.error(
                    f"Hard truncation failed; using synthetic embeddings. Error: {retry_exc}"
                )
                # Fallback to deterministic synthetic embeddings to keep ingestion moving
                synthetic = []
                for text in texts:
                    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
                    emb = [float(b) / 255.0 for b in digest]
                    while len(emb) < EXPECTED_EMBEDDING_DIM:
                        emb.extend(emb[: min(len(emb), EXPECTED_EMBEDDING_DIM - len(emb))])
                    synthetic.append(emb[:EXPECTED_EMBEDDING_DIM])
                return synthetic


def _normalise_for_embedding(text: str) -> str:
    """Normalise text to reduce token spikes from repeated punctuation.

    Collapses long dot leaders (e.g., table-of-contents lines) that can
    tokenise into many subword/punctuation tokens.
    """
    if not text:
        return text

    # Collapse runs of dots (with optional spaces) to a single ellipsis
    text = re.sub(r"(?:\.\s*){4,}", "… ", text)
    # Collapse any remaining long dot runs without spaces
    text = re.sub(r"\.{4,}", "…", text)
    return text


def _truncate_chunk_to_token_limit(
    text: str,
    max_tokens: int = EMBEDDING_USABLE_TOKEN_LIMIT,
    safety_margin: int = EMBEDDING_CONTEXT_SAFETY_MARGIN,
) -> tuple[str, bool]:
    """Truncate text to fit within embedding model's token limit.

    Uses conservative estimates to account for:
    - Special characters and punctuation (increase token count)
    - Multilingual/academic terminology
    - Batch processing overhead
    - Truncation marker suffix

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens to target (default: conservative limit of ~300)
        safety_margin: Safety margin in characters to account for estimation variance

    Returns:
        Tuple of (truncated_text, was_truncated)
        - truncated_text: Text that fits within the token limit
        - was_truncated: Boolean indicating if truncation occurred
    """
    # Normalise to reduce token spikes from punctuation-heavy lines
    normalised = _normalise_for_embedding(text)

    # Conservative estimate: ~1 token per 3 characters (accounts for special chars in academic text)
    # This is more conservative than typical 1 token per 4 chars
    # Reserve 12 chars for " [TRUNCATED]" suffix
    max_chars = (max_tokens * 3) - safety_margin - 12

    if len(normalised) <= max_chars:
        return normalised, False

    # Truncate and find last complete word boundary
    truncated = normalised[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:  # Only use word boundary if reasonable
        truncated = truncated[:last_space]

    truncated = truncated.rstrip() + " [TRUNCATED]"
    return truncated, True


def _init_truncation_stats(total_chunks: int) -> Dict[str, Any]:
    return {
        "total_chunks": total_chunks,
        "truncated_chunks": 0,
        "truncation_chars_lost": 0,
        "truncation_loss_ratio_sum": 0.0,
    }


def _update_truncation_stats(
    stats: Dict[str, Any],
    original_len: int,
    truncated_len: int,
    was_truncated: bool,
) -> None:
    if not was_truncated or original_len <= 0:
        return

    chars_lost = max(0, original_len - truncated_len)
    stats["truncated_chunks"] += 1
    stats["truncation_chars_lost"] += chars_lost
    stats["truncation_loss_ratio_sum"] += chars_lost / original_len


def _finalise_truncation_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    total_chunks = stats["total_chunks"]
    truncated_chunks = stats["truncated_chunks"]
    truncation_ratio = truncated_chunks / total_chunks if total_chunks else 0.0
    truncation_loss_avg_pct = (
        (stats["truncation_loss_ratio_sum"] / truncated_chunks) * 100.0 if truncated_chunks else 0.0
    )

    return {
        "total_chunks": total_chunks,
        "truncated_chunks": truncated_chunks,
        "truncation_ratio": truncation_ratio,
        "truncation_loss_avg_pct": truncation_loss_avg_pct,
        "truncation_chars_lost": stats["truncation_chars_lost"],
    }


def _calculate_truncation_stats(texts: List[str]) -> Dict[str, Any]:
    stats = _init_truncation_stats(len(texts))
    for text in texts:
        truncated_text, was_truncated = _truncate_chunk_to_token_limit(text)
        _update_truncation_stats(stats, len(text), len(truncated_text), was_truncated)

    return _finalise_truncation_stats(stats)


def _add_to_collection_in_batches(
    chunk_collection: Collection,
    ids: List[str],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: Optional[List[List[float]]] = None,
) -> None:
    """Add large batches to ChromaDB in manageable chunks."""

    if not ids:
        return

    batch_limit = CHROMADB_ADD_BATCH_LIMIT
    for start in range(0, len(ids), batch_limit):
        end = start + batch_limit
        collection_kwargs = {
            "ids": ids[start:end],
            "documents": documents[start:end],
            "metadatas": metadatas[start:end],
        }
        if embeddings is not None:
            collection_kwargs["embeddings"] = embeddings[start:end]
        chunk_collection.add(**collection_kwargs)


token_counter = get_token_counter()
perf_metrics = get_perf_metrics()
metrics_collector = get_metrics_collector()

# Initialise adaptive rate limiting for embedding API if enabled
_adaptive_rate_limiter_enabled = False
try:
    from scripts.rag.rag_config import RAGConfig

    config = RAGConfig()
    if config.enable_adaptive_rate_limiting and not _adaptive_rate_limiter_enabled:
        init_adaptive_rate_limiter(
            initial_rate=20.0,
            max_rate=100.0,
        )
        _adaptive_rate_limiter_enabled = True
        embedding_logger = get_global_logger("embedding")
        embedding_logger.info(
            "Adaptive rate limiting enabled for embedding API "
            f"(initial_rate=20.0, max_rate=100.0, model={EMBEDDING_MODEL_NAME})"
        )
except Exception as e:
    ingest_logger = get_logger()
    ingest_logger.debug(f"Could not initialise adaptive rate limiting: {e}")


def _parse_expected_dimension(error_msg: str) -> Optional[int]:
    """Extract the expected embedding dimension from a ChromaDB error message.

    Tries multiple patterns to find the expected dimension:
    - 'dimension of X, got Y'
    - 'expected (\\d+), got (\\d+)'
    - 'dimension (\\d+)'
    """
    # Try primary pattern: "dimension of X, got Y"
    match = re.search(r"dimension of (\d+), got (\d+)", error_msg)
    if match:
        return int(match.group(1))

    # Try alternative pattern: "expected X, got Y"
    match = re.search(r"expected (\d+), got (\d+)", error_msg)
    if match:
        return int(match.group(1))

    # Try to find any explicit dimension mention
    match = re.search(r"dimension[\s:]*of[\s:]?(\d+)", error_msg, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def _resize_embeddings(embeddings: List[List[float]], expected_dim: int) -> List[List[float]]:
    """Pad or truncate embedding vectors to match the expected dimension."""
    adjusted: List[List[float]] = []
    for emb in embeddings:
        if len(emb) == expected_dim:
            adjusted.append(emb)
        elif len(emb) > expected_dim:
            adjusted.append(emb[:expected_dim])
        else:
            adjusted.append(emb + [0.0] * (expected_dim - len(emb)))
    return adjusted


def _cache_key(text: str) -> str:
    """Namespaced cache key to avoid cross-model collisions."""
    return f"[{EMBEDDING_MODEL_NAME}] {text}"


def detect_semantic_drift(
    old_summary: str,
    new_summary: str,
    old_topics: List[str],
    new_topics: List[str],
    doc_hash: Optional[str] = None,
    llm_cache: Optional["LLMCache"] = None,
) -> Dict[str, Any]:
    """Detect semantic drift between document versions using LLM analysis.

    Compares summaries and topic lists from two versions to determine
    if the document's meaning or intent has meaningfully changed.
    Uses LLM to provide nuanced semantic comparison beyond simple
    text diff.

    Args:
        old_summary: Summary text from previous version.
        new_summary: Summary text from current version.
        old_topics: List of key topics from previous version.
        new_topics: List of key topics from current version.
        doc_hash: Optional document hash for caching.
        llm_cache: Optional LLM cache instance.

    Returns:
        Dictionary containing:
            - drift_detected (bool): Whether drift was found.
            - severity (str): 'none', 'low', 'medium', or 'high'.
            - explanation (str): Human-readable explanation.

    Example:
        >>> drift = detect_semantic_drift(
        ...     "Policy requires MFA",
        ...     "Policy optionally supports MFA",
        ...     ["security", "authentication"],
        ...     ["security", "optional"]
        ... )
        >>> drift["severity"]
        'high'
    """
    # Check cache first
    if doc_hash and llm_cache:
        cached = llm_cache.get(doc_hash, "detect_semantic_drift")
        if cached is not None:
            return cached

    prompt = f"""
    You are comparing two versions of a governance document.

    OLD SUMMARY:
    {old_summary}

    NEW SUMMARY:
    {new_summary}

    OLD TOPICS:
    {old_topics}

    NEW TOPICS:
    {new_topics}

    Determine if the meaning or intent of the document has changed.

    Respond ONLY with JSON:
    {{
      "drift_detected": true/false,
      "severity": "none" | "low" | "medium" | "high",
      "explanation": "short explanation"
    }}
    """
    validator_llm = get_LLM_validator()  # reuse LLM validator in preprocess module

    # Apply retry logic to LLM invocation
    @retry_ollama_call(max_retries=3, initial_delay=1.0, operation_name="detect_semantic_drift")
    def _invoke_llm():
        return validator_llm.invoke(prompt)

    raw = _invoke_llm()
    result = extract_first_json_block(raw)

    # Store in cache
    if doc_hash and llm_cache:
        llm_cache.put(doc_hash, "detect_semantic_drift", result)

    return result


def validate_chunk(chunk_id: str, text: str, doc_id: str) -> ChunkSchema:
    """Validate chunk structure using Pydantic schema.

    Performs structural validation to ensure chunk meets minimum
    requirements: non-empty text, minimum length, valid IDs.
    This is the first layer of validation before semantic checks.

    Args:
        chunk_id: Unique identifier for the chunk.
        text: Chunk text content.
        doc_id: Parent document identifier.

    Returns:
        Validated ChunkSchema instance.

    Raises:
        ValidationError: If chunk fails structural validation.
            Logs error details and audit trail before raising.

    See Also:
        validate_chunk_semantics: For content quality validation.
    """
    logger = get_logger()

    try:
        return ChunkSchema(chunk_id=chunk_id, text=text, doc_id=doc_id)
    except ValidationError as e:
        logger.error(f"Chunk validation failed: {e}")
        audit(
            "chunk_validation_error",
            {
                "errors": e.errors(),
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text_preview": text[:200],
            },
        )
        raise


def filter_and_merge_small_chunks(
    chunks: List[str], min_char_threshold: int = 100, doc_id: str = None
) -> Tuple[List[str], Dict[str, int]]:
    """Filter out and merge small chunks before LLM processing.

    Removes very short chunks (< min_char_threshold) that waste LLM calls:
    - Drop small chunks in middle (likely boilerplate/footers)
    - Merge last small chunk with previous (preserve document ending)
    - Keep very last chunk even if small (likely meaningful)
    - PRESERVE ALL TABLE CHUNKS (marked with #### TABLE MARKER ####)

    This early filter saves LLM calls on fragments while preserving content
    and ensuring tables are always kept intact.

    Args:
        chunks: List of text chunks from document.
        min_char_threshold: Minimum chunk size in characters (default 100).
        doc_id: Document ID for logging (optional).

    Returns:
        Tuple of (filtered_chunks, stats):
            - filtered_chunks: List of chunks after filtering/merging
            - stats: Dictionary with:
              - dropped_count: Number of small chunks dropped
              - merged_count: Number of chunks merged into previous
              - kept_small_count: Number of small chunks kept (last chunk)
              - preserved_table_count: Number of table chunks preserved
              - original_count: Original chunk count

    Statistics:
        Enables audit trail to track efficiency gains from filtering.
        Table preservation count helps track table processing.
    """
    logger = get_logger()

    if not chunks:
        return chunks, {
            "dropped_count": 0,
            "merged_count": 0,
            "kept_small_count": 0,
            "preserved_table_count": 0,
            "original_count": 0,
        }

    filtered = []
    stats = {
        "dropped_count": 0,
        "merged_count": 0,
        "kept_small_count": 0,
        "preserved_table_count": 0,
        "original_count": len(chunks),
    }

    for i, chunk in enumerate(chunks):
        chunk_size = len(chunk)
        is_last = i == len(chunks) - 1
        is_small = chunk_size < min_char_threshold

        # Check if chunk contains table markers (preserve all tables)
        is_table_chunk = "#### TABLE MARKER ####" in chunk or "[TABLE" in chunk

        if is_table_chunk:
            # Always preserve table chunks regardless of size
            # TODO: Consider adding a check to ensure table chunks are not excessively large (e.g., > 5000 chars) which could cause LLM issues.
            # For now, we preserve all tables and rely on later truncation if needed. Consider alternative approaches to addressing table chunks.
            filtered.append(chunk)
            stats["preserved_table_count"] += 1
            logger.debug(f"Preserved table chunk ({chunk_size} chars) with markers")
        elif is_small:
            if is_last:
                # Keep last chunk even if small (preserve document ending)
                if filtered:
                    # Merge into previous chunk to consolidate
                    filtered[-1] = filtered[-1] + "\n" + chunk
                    stats["merged_count"] += 1
                    logger.debug(
                        f"Merged last small chunk ({chunk_size} chars) into previous chunk"
                    )
                else:
                    # Only chunk and it's small - keep it
                    filtered.append(chunk)
                    stats["kept_small_count"] += 1
                    logger.debug(f"Keeping single small chunk ({chunk_size} chars)")
            elif filtered:
                # Middle small chunk - merge into previous
                filtered[-1] = filtered[-1] + "\n" + chunk
                stats["merged_count"] += 1
                logger.debug(f"Merged small middle chunk ({chunk_size} chars) into previous")
            else:
                # Very first chunk is small - keep it
                filtered.append(chunk)
                stats["kept_small_count"] += 1
                logger.debug(f"Keeping first small chunk ({chunk_size} chars)")
        else:
            # Normal size chunk - keep as-is
            filtered.append(chunk)

    if stats["dropped_count"] + stats["merged_count"] > 0 or stats["preserved_table_count"] > 0:
        msg = f"Small chunk filtering: {stats['dropped_count']} dropped, {stats['merged_count']} merged, {stats['preserved_table_count']} tables preserved"
        logger.info(msg)
        audit(
            "small_chunk_filter",
            {
                "doc_id": doc_id,
                "original_count": stats["original_count"],
                "filtered_count": len(filtered),
                "dropped_count": stats["dropped_count"],
                "merged_count": stats["merged_count"],
                "kept_small_count": stats["kept_small_count"],
                "preserved_table_count": stats["preserved_table_count"],
            },
        )

    return filtered, stats


def validate_chunk_semantics(
    text: str,
    doc_type: Optional[str] = None,
    chunk_hash: Optional[str] = None,
    llm_cache: Optional["LLMCache"] = None,
) -> Tuple[bool, str]:
    """Validate semantic quality of chunk content using LLM.

    Uses LLM to assess whether chunk is coherent, complete, and useful
    as context for RAG. Filters out navigation boilerplate, truncated
    sentences, and meaningless fragments.

    Args:
        text: Chunk text to validate.
        doc_type: Optional document type for context (e.g., 'governance policy').
        chunk_hash: Optional chunk hash for caching (use chunk text hash).
        llm_cache: Optional LLM cache instance.

    Returns:
        Tuple of (is_valid, reason):
            - is_valid: True if chunk passes semantic validation.
            - reason: Explanation of validation result.

    Example:
        >>> is_valid, reason = validate_chunk_semantics(
        ...     "Navigate: Home > Policies > Security",
        ...     "governance policy"
        ... )
        >>> is_valid
        False
        >>> reason
        'Navigation boilerplate, not meaningful content'
    """
    # Check cache first (using chunk hash)
    if chunk_hash and llm_cache:
        cached = llm_cache.get(chunk_hash, "validate_chunk_semantics")
        if cached is not None:
            return cached["is_valid"], cached["reason"]

    context_type = doc_type or "technical governance document"

    prompt = textwrap.dedent(f"""
    You are validating chunks from a {context_type}.

    Determine if the following text chunk is:
    - coherent
    - complete enough to be useful as context
    - not just navigation, boilerplate, or noise
    - not obviously truncated mid-sentence in a way that breaks meaning

    Respond ONLY in JSON with this schema:
    {{
      "is_valid": true or false,
      "reason": "short explanation"
    }}

    CHUNK:
    \"\"\"{text}\"\"\"
    """)

    validator_llm = get_LLM_validator()  # reuse LLM validator in preprocess module

    # Apply retry logic to LLM invocation
    @retry_ollama_call(max_retries=3, initial_delay=1.0, operation_name="validate_chunk_semantics")
    def _invoke_llm():
        return validator_llm.invoke(prompt)

    raw = _invoke_llm()
    result = extract_first_json_block(raw)  # reuse shared JSON extractor (scripts.utils.json_utils)

    is_valid = bool(result.get("is_valid", False))
    reason = result.get("reason", "")

    # Store in cache
    if chunk_hash and llm_cache:
        llm_cache.put(
            chunk_hash, "validate_chunk_semantics", {"is_valid": is_valid, "reason": reason}
        )

    return is_valid, reason


def repair_chunk_with_llm(
    text: str,
    doc_type: Optional[str] = None,
    chunk_hash: Optional[str] = None,
    llm_cache: Optional["LLMCache"] = None,
) -> str:
    """Attempt to repair invalid chunk using LLM while preserving meaning.

    Instructs LLM to fix common issues like truncated sentences,
    navigation boilerplate, or fragmentation while maintaining all
    technical facts and requirements from the original text.

    Args:
        text: Invalid or problematic chunk text.
        doc_type: Optional document type for context.
        chunk_hash: Optional chunk hash for caching (use chunk text hash).
        llm_cache: Optional LLM cache instance.

    Returns:
        Repaired chunk text, stripped of whitespace.

    Warning:
        LLM repairs are not guaranteed to succeed. Re-validate
        repaired chunks before using them.

    Note:
        The repair preserves technical meaning but may rephrase
        for clarity and coherence.
    """
    # Check cache first
    if chunk_hash and llm_cache:
        cached = llm_cache.get(chunk_hash, "repair_chunk_with_llm")
        if cached is not None:
            return cached

    context_type = doc_type or "technical governance document"

    prompt = textwrap.dedent(f"""
    You are repairing a text chunk from a {context_type}.

    RULES:
    - Preserve all technical meaning and facts.
    - Do NOT introduce new policies, requirements, or technologies.
    - You MAY:
      - fix truncated sentences
      - remove obvious navigation boilerplate
      - lightly rephrase to make the chunk self-contained
    - The result MUST be a coherent, self-contained paragraph or short section.

    Return ONLY the repaired text, no explanations.

    ORIGINAL CHUNK:
    \"\"\"{text}\"\"\"
    """)

    validator_llm = get_LLM_validator()  # reuse LLM validator in preprocess module

    # Apply retry logic to LLM invocation
    @retry_ollama_call(max_retries=3, initial_delay=1.0, operation_name="repair_chunk_with_llm")
    def _invoke_llm():
        return validator_llm.invoke(prompt)

    repaired = _invoke_llm()
    result = repaired.strip()

    # Store in cache
    if chunk_hash and llm_cache:
        llm_cache.put(chunk_hash, "repair_chunk_with_llm", result)

    return result


def compute_chunk_quality_heuristic(text: str) -> Tuple[bool, float, str]:
    """Compute fast heuristic to determine if chunk likely needs semantic validation.

    Uses simple text metrics to identify high-quality chunks that can skip
    expensive LLM validation:
    - Length: Substantial content (not too short/long)
    - Stopwords: Natural language patterns
    - Entropy: Information density

    Args:
        text: Chunk text to evaluate.

    Returns:
        Tuple of (skip_validation, confidence_score, reason):
            - skip_validation: True if chunk appears clean, False if needs validation
            - confidence_score: 0.0-1.0 confidence in quality
            - reason: Explanation of decision

    Heuristic Rules:
        - Skip if: length 100-2000 chars AND stopword ratio > 0.25 AND entropy > 3.0
        - Validate if: fails any metric (likely navigation/boilerplate/truncated)
    """
    # Length check (too short = fragment, too long = concatenation)
    length = len(text)
    if length < 100:
        return False, 0.2, "too_short"
    if length > 2000:
        return False, 0.3, "too_long"

    # Stopword ratio (natural language should have common words)
    # Common English stopwords
    # TODO: Consider domain-specific stopwords, eg for technical documents (e.g., "shall", "must", "requirement")
    # TODO: Switch to using NLTK stopword list in line with other components for consistency
    stopwords = {
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "us",
    }

    words = text.lower().split()
    if len(words) < 10:
        return False, 0.3, "too_few_words"

    stopword_count = sum(1 for word in words if word in stopwords)
    stopword_ratio = stopword_count / len(words)

    if stopword_ratio < 0.15:  # Lowered from 0.25 to be less strict
        return False, 0.4, f"low_stopword_ratio_{stopword_ratio:.2f}"

    # Check for boilerplate patterns (navigation, copyright, etc.)
    boilerplate_patterns = [
        "copyright",
        "all rights reserved",
        "terms of service",
        "privacy policy",
        "cookie settings",
        "contact us",
        "©",
        "trademark",
        "click here",
        "next page",
        "previous",
        "home page",
        "navigation",
        "breadcrumb",
        "sitemap",
        "skip to",
        "login",
        "sign up",
        "subscribe",
    ]

    text_lower = text.lower()
    boilerplate_count = sum(1 for pattern in boilerplate_patterns if pattern in text_lower)
    if boilerplate_count >= 3:  # Multiple boilerplate indicators
        return False, 0.3, f"boilerplate_detected_{boilerplate_count}_patterns"

    # Entropy check (information density)
    # Simple character-level entropy approximation
    char_counts = Counter(text.lower())
    total_chars = len(text)
    entropy = -sum(
        (count / total_chars) * math.log2(count / total_chars) for count in char_counts.values()
    )

    if entropy < 3.0:
        return False, 0.5, f"low_entropy_{entropy:.2f}"

    # All heuristics passed - chunk appears clean
    confidence = min(0.95, 0.5 + (stopword_ratio * 0.5) + ((entropy - 3.0) / 10.0))
    return True, confidence, "heuristic_pass"


def process_and_validate_chunks(
    chunks: List[str],
    doc_id: str,
    doc_type: str,
    llm_cache: Optional["LLMCache"] = None,
    enable_chunk_heuristic: bool = True,
    min_chunk_size: int = 100,
    preserve_domain_keywords: Optional[set] = None,
) -> Tuple[List[Tuple[str, str]], int, int, int]:
    """Apply multi-stage validation and repair pipeline to document chunks.

    Three-stage process for each chunk:
    1. Size filtering (drop very small chunks before LLM work)
    2. Structural validation (length, format)
    3. Semantic validation (coherence, usefulness)
    4. Automatic repair if validation fails

    Failed chunks are logged and skipped. Successful chunks are returned
    with generated IDs.

    Args:
        chunks: List of raw text chunks from document.
        doc_id: Parent document identifier.
        doc_type: Document type for context during validation.
        llm_cache: Optional LLM cache instance.
        enable_chunk_heuristic: Whether to skip semantic validation for high-confidence chunks.
        min_chunk_size: Minimum chunk size in characters (default 100, saves LLM calls).

    Returns:
        Tuple of (processed_chunks, valid_count, repaired_count, failed_count):
            - processed_chunks: List of (chunk_id, text) tuples for valid chunks.
            - valid_count: Number of chunks that passed initial validation.
            - repaired_count: Number of chunks successfully repaired.
            - failed_count: Number of chunks that failed after repair.

    Side Effects:
        - Logs validation and repair events
        - Creates audit trail for failed chunks
        - Filters out very small chunks (saves LLM calls)
    """
    logger = get_logger()

    # Stage 0: Pre-filter small chunks before any LLM work
    # This saves expensive LLM validation calls on fragments
    filtered_chunks, filter_stats = filter_and_merge_small_chunks(
        chunks, min_char_threshold=min_chunk_size, doc_id=doc_id
    )

    valid_chunks = 0
    repaired_chunks = 0
    failed_chunks = 0
    heuristic_skipped = 0

    processed = []
    for i, chunk in enumerate(filtered_chunks):
        chunk_id = f"{doc_id}-chunk-{i}"

        # Compute chunk hash for caching (using hashlib)
        import hashlib

        chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()

        # 1. Structural/schema validation (length, non-empty)
        try:
            validate_chunk(chunk_id, chunk, doc_id)  # Pydantic-based validator
        except Exception as e:
            logger.warning(f"Structural chunk validation failed for {chunk_id}: {e}")
            audit("chunk_structural_validation_error", {"doc_id": doc_id, "chunk_id": chunk_id})
            continue

        # 1.5. Quality heuristic check (skip semantic validation for clean chunks)
        # But ALWAYS preserve chunks containing domain keywords
        has_domain_keyword = False
        if preserve_domain_keywords:
            has_domain_keyword = any(keyword in chunk for keyword in preserve_domain_keywords)

        skip_semantic, confidence, reason = (
            compute_chunk_quality_heuristic(chunk)
            if enable_chunk_heuristic
            else (False, 0.0, "heuristic_disabled")
        )

        # Force-preserve if chunk contains domain keywords (override heuristic)
        if has_domain_keyword and not skip_semantic:
            skip_semantic = True
            confidence = 1.0
            reason = "domain_keyword_preserved"

        if skip_semantic:
            logger.debug(
                f"Skipping semantic validation for {chunk_id} (heuristic confidence: {confidence:.2f}, reason: {reason})"
            )
            # NOTE: Removed per-chunk audit logging to reduce noise
            # Summary is logged after all chunks are processed
            valid_chunks += 1
            heuristic_skipped += 1
            processed.append((chunk_id, chunk))
            continue

        # 2. Semantic validation (skip for academic reference metadata - no LLM needed)
        if doc_type == "academic_reference":
            # For reference metadata, skip LLM validation - metadata is structured and short
            is_valid = True
            reason = "reference_metadata_no_validation"
        else:
            is_valid, reason = validate_chunk_semantics(chunk, doc_id, chunk_hash, llm_cache)

        if not is_valid and doc_type != "academic_reference":
            logger.info(f"Semantic invalid chunk {chunk_id}: {reason}")
            audit(
                "chunk_semantic_invalid", {"doc_id": doc_id, "chunk_id": chunk_id, "reason": reason}
            )

            # 3. Attempt repair
            repaired = repair_chunk_with_llm(chunk, doc_id, chunk_hash, llm_cache)

            # Re-run structural validation
            try:
                validate_chunk(chunk_id, repaired, doc_id)
            except Exception as e:
                logger.warning(f"Repaired chunk still structurally invalid {chunk_id}: {e}")
                audit("chunk_repair_failed_structural", {"doc_id": doc_id, "chunk_id": chunk_id})
                failed_chunks += 1
                continue

            # Re-run semantic validation (with new hash for repaired chunk)
            repaired_hash = hashlib.sha256(repaired.encode("utf-8")).hexdigest()
            is_valid_repaired, reason_repaired = validate_chunk_semantics(
                repaired, doc_id, repaired_hash, llm_cache
            )
            if not is_valid_repaired:
                logger.warning(
                    f"Repaired chunk still semantically invalid {chunk_id}: {reason_repaired}"
                )
                audit(
                    "chunk_repair_failed_semantic",
                    {"doc_id": doc_id, "chunk_id": chunk_id, "reason": reason_repaired},
                )
                failed_chunks += 1
                continue

            logger.info(f"Chunk repaired successfully {chunk_id}")
            audit("chunk_repaired", {"doc_id": doc_id, "chunk_id": chunk_id})
            repaired_chunks += 1
            chunk = repaired  # use repaired version
        else:
            valid_chunks += 1

        processed.append((chunk_id, chunk))

    if heuristic_skipped > 0:
        logger.info(
            f"Heuristic skipped semantic validation for {heuristic_skipped}/{len(chunks)} chunks"
        )

    return processed, valid_chunks, repaired_chunks, failed_chunks


def compute_document_health(
    summary_score: int,
    total_chunks: int,
    valid_chunks: int,
    repaired_chunks: int,
    failed_chunks: int,
    truncated_chunks: int,
    truncation_loss_avg_pct: float,
    truncation_chars_lost: int,
    drift: Optional[Dict],
    preprocess_time: float,
    ingest_time: float,
) -> Dict[str, Any]:
    """Calculate comprehensive health metrics for a processed document.

    Computes weighted health score based on:
    - Summary quality (40% weight)
    - Chunk validity ratio (30% weight)
    - Chunk failure ratio (20% weight)
    - Chunk repair ratio (10% weight)

    Args:
        summary_score: LLM-generated summary quality score (0-10).
        total_chunks: Total number of chunks created.
        valid_chunks: Number passing initial validation.
        repaired_chunks: Number successfully repaired.
        failed_chunks: Number that failed completely.
        truncated_chunks: Number of chunks truncated for embedding.
        truncation_loss_avg_pct: Average truncation loss percentage across truncated chunks.
        truncation_chars_lost: Total characters lost to truncation.
        drift: Optional semantic drift detection result.
        preprocess_time: Time spent in preprocessing (seconds).
        ingest_time: Time spent in ingestion (seconds).

    Returns:
        Dictionary with health metrics:
            - summary_score: Summary quality (0-10)
            - total_chunks: Total chunk count
            - valid_chunks: Valid chunk count
            - repaired_chunks: Repaired chunk count
            - failed_chunks: Failed chunk count
            - truncated_chunks: Truncated chunk count
            - chunk_validity_ratio: Ratio of valid chunks (0-1)
            - chunk_repair_ratio: Ratio of repaired chunks (0-1)
            - chunk_failure_ratio: Ratio of failed chunks (0-1)
            - chunk_truncation_ratio: Ratio of truncated chunks (0-1)
            - chunk_truncation_loss_avg_pct: Average truncation loss percentage
            - chunk_truncation_chars_lost: Total characters lost to truncation
            - semantic_drift: Drift severity if detected
            - preprocess_time: Processing time (seconds)
            - ingest_time: Ingestion time (seconds)
            - health_score: Overall weighted score (0-100)

    Note:
        Weights can be tuned based on your quality priorities.
    """

    chunk_validity_ratio = valid_chunks / total_chunks if total_chunks else 0
    chunk_repair_ratio = repaired_chunks / total_chunks if total_chunks else 0
    chunk_failure_ratio = failed_chunks / total_chunks if total_chunks else 0
    chunk_truncation_ratio = truncated_chunks / total_chunks if total_chunks else 0

    truncation_impact = chunk_truncation_ratio * (truncation_loss_avg_pct / 100.0)
    truncation_penalty = truncation_impact * 10

    # Weighted health score (tune as needed)
    health_score = (
        (summary_score * 0.4)
        + (chunk_validity_ratio * 30)
        + ((1 - chunk_failure_ratio) * 20)
        + ((1 - chunk_repair_ratio) * 10)
        - truncation_penalty
    )
    health_score = max(0.0, health_score)

    if drift is not None:
        drift_severity = drift["severity"]
    else:
        drift_severity = None

    return {
        "summary_score": summary_score,
        "total_chunks": total_chunks,
        "valid_chunks": valid_chunks,
        "repaired_chunks": repaired_chunks,
        "failed_chunks": failed_chunks,
        "truncated_chunks": truncated_chunks,
        "chunk_validity_ratio": chunk_validity_ratio,
        "chunk_repair_ratio": chunk_repair_ratio,
        "chunk_failure_ratio": chunk_failure_ratio,
        "chunk_truncation_ratio": chunk_truncation_ratio,
        "chunk_truncation_loss_avg_pct": truncation_loss_avg_pct,
        "chunk_truncation_chars_lost": truncation_chars_lost,
        "semantic_drift": drift_severity,
        "preprocess_time": preprocess_time,
        "ingest_time": ingest_time,
        "health_score": round(health_score, 2),
    }


def get_previous_version_metadata(
    chunk_collection: Collection, doc_id: str
) -> Optional[Dict[str, Any]]:
    """Retrieve metadata from the previous version of a document.

    Searches ChromaDB for chunks from the most recent previous version
    and extracts shared metadata (summary, topics, version number).
    Used for semantic drift detection and version comparison.

    Args:
        chunk_collection: ChromaDB collection containing chunks.
        doc_id: Document identifier to query.

    Returns:
        Dictionary with previous version metadata:
            - version (int): Previous version number.
            - summary (str): Document summary.
            - key_topics (List[str]): List of key topics.

        Returns None if:
            - No previous versions exist
            - Only version 1 exists (no prior version)
            - Metadata retrieval fails

    Note:
        All chunks for a version share the same metadata, so only
        the first chunk's metadata is returned.
    """

    # Fetch all chunks for this doc_id (with retry, filtered by model)
    @retry_chromadb_call(
        max_retries=5, initial_delay=0.5, operation_name=f"get_previous_version_metadata({doc_id})"
    )
    def _get_chunks():
        return chunk_collection.get(
            where={"$and": [{"doc_id": doc_id}, {"embedding_model": EMBEDDING_MODEL_NAME}]},
            include=["metadatas"],
        )

    results = _get_chunks()

    if not results or not results["metadatas"]:
        return None

    # Extract all versions present
    versions = [m.get("version", 1) for m in results["metadatas"]]
    latest_version = max(versions)

    # If only version 1 exists → no previous version
    if latest_version <= 1:
        return None

    previous_version = latest_version - 1

    # Filter metadata for the previous version
    prev_meta = [m for m in results["metadatas"] if m.get("version") == previous_version]

    if not prev_meta:
        return None

    # All chunks for a version share the same summary/topics → take first
    meta = prev_meta[0]

    # key_topics stored as JSON string → decode
    key_topics_raw = meta.get("key_topics", "[]")
    try:
        key_topics = json.loads(key_topics_raw)
    except Exception:
        key_topics = []

    return {
        "version": previous_version,
        "summary": meta.get("summary", ""),
        "key_topics": key_topics,
    }


def generate_chunk_embeddings_batch(
    texts: List[str], embedding_cache: Optional["EmbeddingCache"] = None, batch_size: int = 1
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Generate embeddings for chunks with caching and batching.

    Optimised embedding generation that:
    1. Checks cache for existing embeddings (hash-based lookup)
    2. Batches uncached texts for efficient generation
    3. Stores new embeddings in cache for future reuse

    Args:
        texts: List of chunk texts to embed.
        embedding_cache: Optional cache for storing/retrieving embeddings.
        batch_size: Number of texts to embed in each batch (default 1 - Ollama concatenates batch into single context).

    Returns:
        Tuple of (embeddings, truncation_stats).
            - embeddings: List of embedding vectors, one per input text.
            - truncation_stats: Summary of truncation extent and loss magnitude.

    Performance:
        - Cache hit: ~1ms (hash lookup)
        - Cache miss (batch): ~50-100ms per batch of 32 chunks
        - Cache miss (individual): ~5-10ms per chunk
        - Typical hit rate: 10-50% (depends on content duplication)

    Example:
        >>> cache = EmbeddingCache("/path/to/cache.json")
        >>> embeddings, truncation_stats = generate_chunk_embeddings_batch(
        ...     ["chunk1", "chunk2", "chunk3"],
        ...     embedding_cache=cache,
        ...     batch_size=32
        ... )
        >>> len(embeddings) == 3  # One embedding per chunk
        True
    """
    logger = get_logger()
    embedding_logger = get_global_logger("embedding")

    # Log high-level embedding operation to dedicated embedding log
    avg_len = sum(len(t) for t in texts) // len(texts) if texts else 0
    embedding_logger.info(
        f"Embedding batch: {len(texts)} chunks, avg_length={avg_len} chars, "
        f"batch_size={batch_size}, model={EMBEDDING_MODEL_NAME}"
    )

    # DEBUG: Log entry to function
    # TODO: remove or reduce verbosity of this log after debugging embedding issues (e.g., oversized chunks)
    import sys

    logger.error(
        f"🔥 generate_chunk_embeddings_batch called with {len(texts)} texts, batch_size={batch_size}"
    )
    if texts:
        logger.error(
            f"🔥 First text length: {len(texts[0])} chars, Last text length: {len(texts[-1])} chars"
        )
    sys.stdout.flush()
    sys.stderr.flush()

    if not texts:
        return [], _finalise_truncation_stats(_init_truncation_stats(0))

    truncation_stats = _init_truncation_stats(len(texts))
    truncation_results: List[Tuple[str, bool]] = []
    for text in texts:
        truncated_text, was_truncated = _truncate_chunk_to_token_limit(text)
        truncation_results.append((truncated_text, was_truncated))
        _update_truncation_stats(truncation_stats, len(text), len(truncated_text), was_truncated)

    final_truncation_stats = _finalise_truncation_stats(truncation_stats)
    truncated_texts = [result[0] for result in truncation_results]
    truncation_flags = [result[1] for result in truncation_results]

    # Initialise embedding model
    embed_model = _create_embed_model()

    # Try to get cached embeddings
    if embedding_cache and embedding_cache.enabled:
        namespaced_texts = [_cache_key(t) for t in texts]
        embeddings, uncached_ns_texts = embedding_cache.get_batch(namespaced_texts)
        cache_hits = sum(1 for e in embeddings if e is not None)

        if cache_hits > 0:
            logger.debug(
                f"Embedding cache: {cache_hits}/{len(texts)} hits ({cache_hits/len(texts)*100:.1f}%)"
            )

        # If all cached, return immediately
        if not uncached_ns_texts:
            return embeddings, final_truncation_stats  # type: ignore

        # Generate embeddings for uncached texts in batches (with retry)
        uncached_embeddings = []
        # Map back to original texts
        ns_to_original = {_cache_key(t): t for t in texts}
        ns_to_truncated = {_cache_key(texts[i]): truncated_texts[i] for i in range(len(texts))}
        ns_to_truncation_flag = {
            _cache_key(texts[i]): truncation_flags[i] for i in range(len(texts))
        }
        uncached_texts = [ns_to_original[ns] for ns in uncached_ns_texts]
        uncached_truncated_texts = [ns_to_truncated[ns] for ns in uncached_ns_texts]
        uncached_truncation_flags = [ns_to_truncation_flag[ns] for ns in uncached_ns_texts]

        for i in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[i : i + batch_size]
            truncated_batch_texts = uncached_truncated_texts[i : i + batch_size]
            batch_truncation_flags = uncached_truncation_flags[i : i + batch_size]

            # Apply retry logic to embedding generation
            @retry_ollama_call(
                max_retries=3, initial_delay=1.0, operation_name=f"embed_batch_{i//batch_size}"
            )
            def _generate_embeddings():
                import time

                # Apply adaptive rate limiting for embedding API if enabled
                limiter = None
                if config.enable_adaptive_rate_limiting:
                    limiter = get_adaptive_rate_limiter()
                    if limiter:
                        limiter.acquire(blocking=True)

                # GUARD: Truncate batch to fit within embedding model's context length
                truncated_batch = []
                truncation_count = 0
                for original_text, truncated_text, was_truncated in zip(
                    batch, truncated_batch_texts, batch_truncation_flags
                ):
                    truncated_batch.append(truncated_text)
                    if was_truncated:
                        truncation_count += 1
                        original_len = len(original_text)
                        truncated_len = len(truncated_text)
                        logger.debug(
                            f"Truncated chunk for embedding: {original_len} chars → {truncated_len} chars"
                        )

                if truncation_count > 0:
                    logger.warning(
                        f"Truncated {truncation_count}/{len(batch)} chunks to fit within "
                        f"{EMBEDDING_MODEL_MAX_TOKEN_LIMIT}-token context limit for {EMBEDDING_MODEL_NAME}"
                    )

                # CRITICAL DEBUG: Log all chunk sizes before embedding
                chunk_lengths = [len(chunk) for chunk in truncated_batch]
                logger.warning(
                    f"🔍 Embedding batch of {len(truncated_batch)} chunks. Lengths: {chunk_lengths}"
                )
                max_chunk_len = max(chunk_lengths) if chunk_lengths else 0
                if max_chunk_len > EMBEDDING_USABLE_TOKEN_LIMIT * 3:
                    logger.error(
                        f"⚠️  OVERSIZED CHUNK DETECTED! Max chunk in batch: {max_chunk_len} chars "
                        f"(limit: {EMBEDDING_USABLE_TOKEN_LIMIT * 3} chars)"
                    )
                    # Log the problematic chunks
                    for idx, chunk in enumerate(truncated_batch):
                        if len(chunk) > EMBEDDING_USABLE_TOKEN_LIMIT * 3:
                            logger.error(
                                f"Chunk {idx}: {len(chunk)} chars, preview: {chunk[:200]}..."
                            )

                start_time = time.perf_counter()
                try:
                    embeddings = _embed_documents_with_fallback(embed_model, truncated_batch)

                    # GUARD: Validate embedding dimensions match expected
                    if embeddings and len(embeddings) > 0:
                        actual_dim = len(embeddings[0])
                        if actual_dim != EXPECTED_EMBEDDING_DIM:
                            raise ValueError(
                                f"Embedding dimension mismatch: expected {EXPECTED_EMBEDDING_DIM}D "
                                f"(for {EMBEDDING_MODEL_NAME}), got {actual_dim}D. "
                                f"Check EMBEDDING_MODEL_NAME in vectors.py"
                            )
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    # Record in adaptive rate limiter if enabled
                    if limiter:
                        limiter.record_request(latency_ms=latency_ms, success=True, status_code=200)
                except Exception as e:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    if limiter:
                        limiter.record_request(
                            latency_ms=latency_ms, success=False, error_type=type(e).__name__
                        )
                    # Dump failed batch for analysis
                    try:
                        debug_dir = "/tmp/embedding_debug"
                        os.makedirs(debug_dir, exist_ok=True)
                        batch_num = i // batch_size
                        debug_file = f"{debug_dir}/failed_batch_{batch_num:04d}.txt"
                        with open(debug_file, "w", encoding="utf-8") as f:
                            f.write(f"Batch #{batch_num} FAILED\n")
                            f.write(f"Number of chunks in batch: {len(truncated_batch)}\n")
                            f.write(f"Chunk lengths: {[len(c) for c in truncated_batch]}\n")
                            f.write(f"Total chars: {sum(len(c) for c in truncated_batch)}\n")
                            f.write("=" * 80 + "\n")
                            for idx, chunk in enumerate(truncated_batch):
                                f.write(f"\n--- Chunk {idx} (len={len(chunk)}) ---\n")
                                f.write(chunk)
                                f.write("\n")
                        logger.error(f"Dumped failed batch {batch_num} to {debug_file}")
                    except Exception as dump_exc:
                        logger.warning(f"Failed to dump failed batch: {dump_exc}")
                    raise

                # Record embedding metrics
                # Estimate tokens (embeddings are ~1 token per 4 chars on average)
                total_chars = sum(len(text) for text in batch)
                estimated_tokens = total_chars // 4

                perf_metrics.record_generation(
                    latency_ms=latency_ms,
                    model=EMBEDDING_MODEL_NAME,
                )
                metrics_collector.record_llm_call(
                    model=f"embedding:{EMBEDDING_MODEL_NAME}",
                    input_tokens=estimated_tokens,
                    output_tokens=len(batch) * 1024,  # Each embedding is 1024 dims
                    latency_ms=latency_ms,
                    success=True,
                )

                return embeddings

            batch_embs = _generate_embeddings()
            uncached_embeddings.extend(batch_embs)

        # Store newly generated embeddings in cache
        # Store using namespaced keys
        embedding_cache.put_batch(uncached_ns_texts, uncached_embeddings)

        # Merge cached and newly generated embeddings
        result = []
        uncached_idx = 0
        for emb in embeddings:
            if emb is not None:
                result.append(emb)
            else:
                result.append(uncached_embeddings[uncached_idx])
                uncached_idx += 1

        return result, final_truncation_stats

    else:
        # No cache - generate all embeddings in batches
        all_embeddings = []
        logger = get_logger()
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            truncated_batch_texts = truncated_texts[i : i + batch_size]
            batch_truncation_flags = truncation_flags[i : i + batch_size]

            # GUARD: Truncate batch to fit within embedding model's context length
            # TODO: Chunk size needs tuning, currently truncating just about every chunk
            truncated_batch = []
            truncation_count = 0
            for truncated_text, was_truncated in zip(truncated_batch_texts, batch_truncation_flags):
                truncated_batch.append(truncated_text)
                if was_truncated:
                    truncation_count += 1

            if truncation_count > 0:
                logger.warning(
                    f"Truncated {truncation_count}/{len(batch)} chunks to fit within "
                    f"{EMBEDDING_MODEL_MAX_TOKEN_LIMIT}-token context limit for {EMBEDDING_MODEL_NAME}"
                )

            # CRITICAL DEBUG: Log all chunk sizes before embedding
            chunk_lengths = [len(chunk) for chunk in truncated_batch]
            logger.warning(
                f"🔍 Embedding batch of {len(truncated_batch)} chunks. Lengths: {chunk_lengths}"
            )

            # Calculate total context size (Ollama concatenates all texts in batch)
            total_chars = sum(chunk_lengths)
            estimated_tokens = total_chars / 3
            logger.error(
                f"💥 BATCH TOTAL: {total_chars} chars = ~{estimated_tokens:.0f} tokens "
                f"(limit: {EMBEDDING_MODEL_MAX_TOKEN_LIMIT} tokens)"
            )

            # Check individual chunks
            max_chunk_len = max(chunk_lengths) if chunk_lengths else 0
            if max_chunk_len > EMBEDDING_USABLE_TOKEN_LIMIT * 3:
                logger.error(
                    f"⚠️  OVERSIZED CHUNK DETECTED! Max chunk in batch: {max_chunk_len} chars "
                    f"(limit: {EMBEDDING_USABLE_TOKEN_LIMIT * 3} chars)"
                )
                # Log the problematic chunks
                for idx, chunk in enumerate(truncated_batch):
                    if len(chunk) > EMBEDDING_USABLE_TOKEN_LIMIT * 3:
                        logger.error(f"Chunk {idx}: {len(chunk)} chars, preview: {chunk[:200]}...")

            # Log first chunk preview for debugging
            # TODO: Remove or reduce logging after debugging is complete to avoid log bloat
            # TODO: Add more structured logging around chunk sizes and truncation for better monitoring
            if truncated_batch:
                logger.error(f"📝 First chunk preview: {truncated_batch[0][:150]}...")
                logger.error(
                    f"🔍 truncated_batch type: {type(truncated_batch)}, len: {len(truncated_batch)}"
                )
                logger.error(
                    f"🔍 truncated_batch[0] type: {type(truncated_batch[0])}, len: {len(truncated_batch[0])}"
                )

            # DEBUG: Dump chunks to file for analysis
            import os

            debug_dir = "/tmp/embedding_debug"
            os.makedirs(debug_dir, exist_ok=True)
            batch_num = i // batch_size
            debug_file = f"{debug_dir}/batch_{batch_num:04d}_len_{len(truncated_batch[0]) if truncated_batch else 0}.txt"
            try:
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(f"Batch #{batch_num}\n")
                    f.write(f"Number of chunks in batch: {len(truncated_batch)}\n")
                    f.write(f"Chunk lengths: {[len(c) for c in truncated_batch]}\n")
                    f.write(f"Total chars: {sum(len(c) for c in truncated_batch)}\n")
                    f.write(f"Estimated tokens: {sum(len(c) for c in truncated_batch) / 3:.0f}\n")
                    f.write("=" * 80 + "\n")
                    for idx, chunk in enumerate(truncated_batch):
                        f.write(f"\n--- Chunk {idx} (len={len(chunk)}) ---\n")
                        f.write(chunk)
                        f.write("\n")
                logger.debug(f"Dumped batch {batch_num} to {debug_file}")
            except Exception as e:
                logger.warning(f"Failed to dump debug file: {e}")

            # WORKAROUND: Add small delay to prevent Ollama server state accumulation
            # Issue: After ~100 rapid embedding requests, Ollama fails even with valid inputs
            # This suggests internal buffer overflow or memory leak in Ollama server
            import time

            time.sleep(0.05)  # 50ms delay between batches

            try:
                batch_embs = _embed_documents_with_fallback(embed_model, truncated_batch)
            except Exception as e:
                # Dump failed batch for analysis
                try:
                    debug_dir = "/tmp/embedding_debug"
                    os.makedirs(debug_dir, exist_ok=True)
                    batch_num = i // batch_size
                    debug_file = f"{debug_dir}/failed_batch_{batch_num:04d}.txt"
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write(f"Batch #{batch_num} FAILED\n")
                        f.write(f"Number of chunks in batch: {len(truncated_batch)}\n")
                        f.write(f"Chunk lengths: {[len(c) for c in truncated_batch]}\n")
                        f.write(f"Total chars: {sum(len(c) for c in truncated_batch)}\n")
                        f.write("=" * 80 + "\n")
                        for idx, chunk in enumerate(truncated_batch):
                            f.write(f"\n--- Chunk {idx} (len={len(chunk)}) ---\n")
                            f.write(chunk)
                            f.write("\n")
                    logger.error(f"Dumped failed batch {batch_num} to {debug_file}")
                except Exception as dump_exc:
                    logger.warning(f"Failed to dump failed batch: {dump_exc}")
                raise e
            all_embeddings.extend(batch_embs)

        return all_embeddings, final_truncation_stats


def store_chunks_in_chroma(
    doc_id: str,
    file_hash: str,
    source_path: str,
    version: int,
    chunks: List[str],
    metadata: Dict[str, Any],
    chunk_collection: Collection,
    doc_collection: Collection,
    preprocess_duration: float,
    ingest_duration: float,
    dry_run: bool = False,
    llm_cache: Optional["LLMCache"] = None,
    enable_drift_detection: bool = False,
    enable_chunk_heuristic: bool = True,
    embedding_cache: Optional["EmbeddingCache"] = None,
    embedding_batch_size: int = 1,
    chunk_hashes: Optional[List[str]] = None,
    preserve_domain_keywords: Optional[set] = None,
    full_text: str = "",
    document_structure: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Store validated chunks and document embeddings in ChromaDB.

    Complete storage workflow:
    1. Validate and repair chunks using LLM
    2. Detect semantic drift from previous version
    3. Calculate document health metrics
    4. Generate embeddings for chunks (with caching and batching)
    5. Store chunks with metadata in chunk collection
    6. Store document-level embedding in document collection

    Args:
        doc_id: Unique document identifier.
        file_hash: SHA-256 hash for change detection.
        source_path: Original file path.
        version: Document version number.
        chunks: List of text chunks from document.
        metadata: Preprocessed metadata including summary, topics, scores.
        chunk_collection: ChromaDB collection for chunks.
        doc_collection: ChromaDB collection for document embeddings.
        preprocess_duration: Time spent preprocessing (seconds).
        ingest_duration: Time spent ingesting (seconds).
        dry_run: If True, simulate without writing to database.
        llm_cache: Optional LLM output cache.
        enable_drift_detection: Whether to detect semantic drift.
        enable_chunk_heuristic: Whether to use chunk quality heuristics.
        embedding_cache: Optional embedding cache for reusing vectors.
        embedding_batch_size: Batch size for embedding generation (default 1 - Ollama concatenates batch into single context).
        chunk_hashes: Optional list of SHA-256 hashes for chunks (for idempotency tracking).

    Side Effects:
        - Writes chunks to chunk_collection
        - Writes document embedding to doc_collection
        - Logs storage events and health metrics
        - Creates audit trail

    Performance:
        - Embedding cache: Reuses embeddings for identical text (10-50% hit rate)
        - Batch processing: Generates embeddings in batches (2-3x faster)
        - Chunk idempotency: Skips re-processing of unchanged chunks

    Note:
        key_topics are JSON-encoded for Chroma metadata storage.
        Document embedding uses summary text for semantic representation.
        chunk_hashes enables chunk-level deduplication for efficiency.
    """
    logger = get_logger()

    # Check health of chunks and attempt repair if possible
    doc_type = metadata.get("doc_type", "unknown")
    validated_chunks, valid_count, repaired_count, failed_count = process_and_validate_chunks(
        chunks,
        doc_id,
        doc_type,
        llm_cache=llm_cache,
        enable_chunk_heuristic=enable_chunk_heuristic,
        preserve_domain_keywords=preserve_domain_keywords,
    )

    # Ensure key_topics is JSON-encoded string for Chroma
    key_topics = metadata.get("key_topics", [])
    key_topics_str = json.dumps(key_topics)

    # Check for semantic drift from previous version
    # Only runs on updates (hash changed) when version > 1 and drift detection is enabled
    drift = None
    if enable_drift_detection and version > 1:
        previous_version = get_previous_version_metadata(chunk_collection, doc_id)
        if previous_version is not None:
            logger.info(f"Detecting semantic drift for {doc_id} (version {version})")
            # TODO validate semantic drift schema compliance
            drift = detect_semantic_drift(
                old_summary=previous_version["summary"],
                new_summary=metadata.get("summary", ""),
                old_topics=previous_version["key_topics"],
                new_topics=key_topics_str,
                doc_hash=file_hash,
                llm_cache=llm_cache,
            )

            audit(
                "semantic_drift",
                {"doc_id": doc_id, "previous_version": previous_version["version"], "drift": drift},
            )
        else:
            logger.warning(
                f"Previous version metadata not found for {doc_id}, skipping drift detection"
            )
            audit(
                "drift_detection_skipped",
                {"doc_id": doc_id, "reason": "previous_version_not_found"},
            )

    truncation_stats = _calculate_truncation_stats([text for _, text in validated_chunks])

    health = compute_document_health(
        summary_score=metadata["summary_scores"]["overall"],
        total_chunks=len(chunks),
        valid_chunks=valid_count,
        repaired_chunks=repaired_count,
        failed_chunks=failed_count,
        truncated_chunks=truncation_stats["truncated_chunks"],
        truncation_loss_avg_pct=truncation_stats["truncation_loss_avg_pct"],
        truncation_chars_lost=truncation_stats["truncation_chars_lost"],
        drift=drift,
        preprocess_time=preprocess_duration,
        ingest_time=ingest_duration,
    )

    audit("document_health", {"doc_id": doc_id, "health": health})

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    base_metadata = {
        "doc_id": doc_id,
        "source": source_path,
        "version": version,
        "hash": file_hash,
        "doc_type": doc_type,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "key_topics": key_topics_str,
        "summary": metadata.get("summary", ""),
        "summary_scores": json.dumps(metadata.get("summary_scores", {})),
        "health": json.dumps(health),
        "source_category": metadata.get("source_category") or "",
        "timestamp": timestamp,
    }

    # Add repository metadata for cross-repo clustering (code ingestion)
    if metadata.get("repository"):
        base_metadata["repository"] = metadata["repository"]
    if metadata.get("project_key"):
        base_metadata["project"] = metadata["project_key"]
    if metadata.get("branch"):
        base_metadata["branch"] = metadata["branch"]

    # Add code-specific metadata if present (for code documents)
    if metadata.get("source_category") == "code":
        # Code parser fields from ParseResult
        if "language" in metadata:
            base_metadata["language"] = metadata["language"]
        if "service_name" in metadata:
            base_metadata["service_name"] = metadata["service_name"] or metadata.get(
                "service_name", ""
            )
        if "service_type" in metadata:
            base_metadata["service_type"] = metadata["service_type"] or ""
        if "external_dependencies" in metadata:
            # Store as JSON string for ChromaDB compatibility
            deps = metadata["external_dependencies"]
            base_metadata["dependencies"] = json.dumps(deps) if isinstance(deps, list) else deps
        if "internal_calls" in metadata:
            calls = metadata["internal_calls"]
            base_metadata["internal_calls"] = (
                json.dumps(calls) if isinstance(calls, list) else calls
            )
        if "endpoints" in metadata:
            eps = metadata["endpoints"]
            base_metadata["endpoints"] = json.dumps(eps) if isinstance(eps, list) else eps
        if "database_refs" in metadata:
            dbs = metadata["database_refs"]
            base_metadata["db"] = json.dumps(dbs) if isinstance(dbs, list) else dbs
        if "message_queues" in metadata:
            queues = metadata["message_queues"]
            base_metadata["queue"] = json.dumps(queues) if isinstance(queues, list) else queues
        if "exports" in metadata:
            exports = metadata["exports"]
            base_metadata["exports"] = json.dumps(exports) if isinstance(exports, list) else exports

    ids = []
    docs = []
    metas = []

    for chunk_idx, (chunk_id, text) in enumerate(validated_chunks):
        ids.append(chunk_id)
        docs.append(text)
        chunk_meta = base_metadata.copy()

        # Add chunk-specific hash if provided (for idempotency tracking)
        if chunk_hashes and chunk_idx < len(chunk_hashes):
            chunk_meta["chunk_text_hash"] = chunk_hashes[chunk_idx]

        # Add enhanced metadata if available
        if create_enhanced_metadata and full_text:
            try:
                # Calculate chunk position in full text for structure mapping
                chunk_char_start = None
                chunk_char_end = None
                if document_structure:
                    # Try to find exact chunk position in full text
                    # Search for first 100 chars (or less if chunk is shorter)
                    search_len = min(100, len(text))
                    if search_len > 10:  # Only search if we have meaningful text
                        chunk_pos = full_text.find(text[:search_len])
                        if chunk_pos >= 0:
                            chunk_char_start = chunk_pos
                            chunk_char_end = chunk_pos + len(text)
                        else:
                            # Fallback: Estimate position based on chunk index and average chunk size
                            # This ensures we still get structure metadata even if exact match fails
                            avg_chunk_size = len(full_text) // max(1, len(validated_chunks))
                            chunk_char_start = chunk_idx * avg_chunk_size
                            chunk_char_end = min(chunk_char_start + len(text), len(full_text))
                            logger.debug(
                                f"Chunk {chunk_idx}: Exact match failed, using estimated position {chunk_char_start}-{chunk_char_end}"
                            )

                enhanced = create_enhanced_metadata(
                    chunk_text=text,
                    chunk_index=chunk_idx,
                    total_chunks=len(validated_chunks),
                    doc_id=doc_id,
                    full_text=full_text,
                    doc_type=doc_type,
                    document_structure=document_structure,
                    chunk_char_start=chunk_char_start,
                    chunk_char_end=chunk_char_end,
                )
                # Add enhanced fields to metadata (serialise as JSON for ChromaDB compatibility)
                chunk_meta["chapter"] = enhanced.chapter or ""
                chunk_meta["section_title"] = enhanced.section_title or ""
                chunk_meta["heading_path"] = enhanced.heading_path or ""
                chunk_meta["parent_section"] = enhanced.parent_section or ""
                chunk_meta["section_depth"] = enhanced.section_depth or 0
                chunk_meta["prev_chunk_id"] = enhanced.prev_chunk_id or ""
                chunk_meta["next_chunk_id"] = enhanced.next_chunk_id or ""
                chunk_meta["technical_entities"] = json.dumps(enhanced.technical_entities)
                chunk_meta["code_language"] = enhanced.code_language or ""
                chunk_meta["contains_table"] = enhanced.contains_table
                chunk_meta["contains_code"] = enhanced.contains_code
                chunk_meta["contains_diagram"] = enhanced.contains_diagram
                chunk_meta["is_api_reference"] = enhanced.is_api_reference
                chunk_meta["is_configuration"] = enhanced.is_configuration
                chunk_meta["content_type"] = enhanced.content_type
            except Exception as e:
                # Log but don't fail on enhanced metadata extraction errors
                logger.warning(f"Failed to extract enhanced metadata for chunk {chunk_idx}: {e}")

        # Add sequence number to preserve document order
        chunk_meta["sequence_number"] = chunk_idx

        metas.append(chunk_meta)

    if dry_run:
        logger.info(f"[DRY-RUN] Would store {len(chunks)} chunks for {doc_id}")
        audit("dry_run_store", {"doc_id": doc_id, "chunk_count": len(chunks), "hash": file_hash})
        return

    # Store chunks if any exist (when not using parent-child chunking)
    if ids:
        # Log chunk statistics BEFORE embedding
        chunk_lengths = [len(doc) for doc in docs]
        logger.info(
            f"Preparing to embed {len(docs)} chunks for {doc_id}. "
            f"Sizes: min={min(chunk_lengths)}, max={max(chunk_lengths)}, avg={sum(chunk_lengths)//len(chunk_lengths)}"
        )
        if max(chunk_lengths) > EMBEDDING_USABLE_TOKEN_LIMIT * 3:
            logger.error(
                f"⚠️  CRITICAL: Chunk exceeds embedding limit BEFORE truncation! "
                f"Max chunk: {max(chunk_lengths)} chars, limit: {EMBEDDING_USABLE_TOKEN_LIMIT * 3} chars"
            )
            # Find and log the oversized chunks
            for idx, doc in enumerate(docs):
                if len(doc) > EMBEDDING_USABLE_TOKEN_LIMIT * 3:
                    logger.error(f"Oversized chunk #{idx}: {len(doc)} chars")
                    logger.error(f"Preview: {doc[:300]}...")

        # Generate embeddings for chunks (with batching and caching)
        # All documents must use the same embedding model (EMBEDDING_MODEL_NAME)
        import sys

        logger.error(f"🚨 ABOUT TO EMBED {len(docs)} chunks for {doc_id}")
        sys.stdout.flush()
        sys.stderr.flush()
        embedding_result = generate_chunk_embeddings_batch(
            docs, embedding_cache=embedding_cache, batch_size=embedding_batch_size
        )
        if isinstance(embedding_result, tuple):
            chunk_embeddings, _ = embedding_result
        else:
            chunk_embeddings = embedding_result

        # GUARD: Validate embedding dimensions before storing
        if chunk_embeddings and len(chunk_embeddings) > 0:
            actual_dim = len(chunk_embeddings[0])
            if actual_dim != EXPECTED_EMBEDDING_DIM:
                raise ValueError(
                    f"❌ EMBEDDING DIMENSION MISMATCH: Expected {EXPECTED_EMBEDDING_DIM}D "
                    f"(for {EMBEDDING_MODEL_NAME}), got {actual_dim}D for {doc_id}. "
                    f"This will cause RAG queries to fail. "
                    f"Check: (1) EMBEDDING_MODEL_NAME in vectors.py, "
                    f"(2) ChromaDB collections not created with default embedding function. "
                    f"Run: client.delete_collection('governance_docs_chunks') to reset."
                )

        # Add chunks to ChromaDB with retry logic
        @retry_chromadb_call(
            max_retries=5, initial_delay=0.5, operation_name=f"chunk_collection.add({doc_id})"
        )
        def _add_chunks(embeddings: Optional[List[List[float]]]):
            # Filter and sanitise metadata for ChromaDB - remove None, convert to valid types
            sanitised_metas = []
            for m in metas:
                sanitised = {}
                for k, v in m.items():
                    if v is None:
                        continue  # Skip None values
                    # Only keep strings, numbers, bools, or JSON strings
                    if isinstance(v, (str, int, float, bool)):
                        sanitised[k] = v
                    elif isinstance(v, (list, dict)):
                        # Serialise complex types to JSON strings
                        sanitised[k] = json.dumps(v)
                    else:
                        # Convert other types to string
                        sanitised[k] = str(v)
                sanitised_metas.append(sanitised)

            chunk_collection.add(
                ids=ids,
                documents=docs,
                metadatas=sanitised_metas,
                embeddings=(embeddings if embeddings else None),  # Let Chroma generate if None
            )

        try:
            _add_chunks(chunk_embeddings)
        except (InvalidArgumentError, ValueError, TypeError, Exception) as e:
            # Automatically reconcile dimension mismatches by padding/truncating
            error_str = str(e)
            if "dimension" in error_str.lower() or "embedding" in error_str.lower():
                expected_dim = _parse_expected_dimension(error_str)
                if expected_dim and chunk_embeddings and len(chunk_embeddings) > 0:
                    try:
                        actual_dim = len(chunk_embeddings[0])
                        if actual_dim != expected_dim:
                            adjusted_embeddings = _resize_embeddings(chunk_embeddings, expected_dim)
                            logger.warning(
                                f"Adjusted chunk embeddings from dim {actual_dim} to expected {expected_dim}"
                            )
                            _add_chunks(adjusted_embeddings)
                        else:
                            # Dimensions match but still failed - re-raise original
                            raise
                    except (InvalidArgumentError, ValueError, TypeError) as resize_err:
                        logger.error(
                            f"Failed to adjust embeddings: {resize_err}. Original error: {error_str}"
                        )
                        raise
                else:
                    if not expected_dim:
                        logger.debug(f"Could not parse expected dimension from error: {error_str}")
                    raise
            else:
                # Not a dimension-related error, re-raise as-is
                raise

        logger.info(f"STORE {doc_id} - {len(docs)} chunks")
        audit(
            "store_chunks",
            {
                "doc_id": doc_id,
                "chunk_count": len(docs),
                "hash": file_hash,
                "embeddings_cached": sum(1 for e in (chunk_embeddings or []) if e is not None),
            },
        )
    else:
        logger.debug(f"No chunks to store for {doc_id} (using parent-child chunking)")
        audit("no_chunks_stored", {"doc_id": doc_id, "reason": "parent_child_chunking"})

    # Always store document embedding, even if no chunks (for parent-child chunking)

    # Calculate document embedding (with caching)
    # N.B. Chroma DB automatically calculates vector embedding on insertion as well, this overrides the Chroma embeddings
    # Could also use the cleaned_text rather than the summary. Using the summary is better for larger documents
    summary_text = metadata.get("summary", "")

    # GUARD: Truncate summary to fit within embedding model's context limit
    # This is critical - summaries can sometimes be very long
    if summary_text:
        original_len = len(summary_text)
        summary_text, was_truncated = _truncate_chunk_to_token_limit(summary_text)
        if was_truncated:
            logger.warning(
                f"Truncated document summary for {doc_id}: {original_len} chars → {len(summary_text)} chars"
            )
        logger.info(f"Document summary for {doc_id}: {len(summary_text)} chars")

    # Check cache for document embedding
    if embedding_cache and embedding_cache.enabled:
        doc_embedding = embedding_cache.get(_cache_key(summary_text))
        if doc_embedding is None:
            # Not in cache - generate and store (with retry)
            @retry_ollama_call(
                max_retries=3, initial_delay=1.0, operation_name=f"embed_document({doc_id})"
            )
            def _generate_doc_embedding():
                import time

                start_time = time.perf_counter()
                embed_model = _create_embed_model()
                embedding = _embed_documents_with_fallback(embed_model, [summary_text])[0]
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Record metrics
                estimated_tokens = len(summary_text) // 4
                metrics_collector.record_llm_call(
                    model=f"embedding:{EMBEDDING_MODEL_NAME}",
                    input_tokens=estimated_tokens,
                    output_tokens=1024,  # 1024-dim embedding
                    latency_ms=latency_ms,
                    success=True,
                )

                return embedding

            try:
                doc_embedding = _generate_doc_embedding()
                embedding_cache.put(_cache_key(summary_text), doc_embedding)
                logger.debug(f"Generated and cached document embedding for {doc_id}")
            except Exception as e:
                # For academic references and when Ollama is unavailable, use synthetic embedding
                if doc_type == "academic_reference":
                    logger.info(
                        f"Ollama unavailable for doc embedding, using synthetic for {doc_id}"
                    )
                    doc_embedding = _generate_synthetic_embedding()
                    embedding_cache.put(_cache_key(summary_text), doc_embedding)
                else:
                    raise
        else:
            logger.debug(f"Using cached document embedding for {doc_id}")
    else:
        # No cache - generate directly (with retry)
        @retry_ollama_call(
            max_retries=3, initial_delay=1.0, operation_name=f"embed_document_nocache({doc_id})"
        )
        def _generate_doc_embedding():
            import time

            start_time = time.perf_counter()
            embed_model = _create_embed_model()
            embedding = embed_model.embed_documents([summary_text])[0]
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Record metrics
            estimated_tokens = len(summary_text) // 4
            metrics_collector.record_llm_call(
                model=f"embedding:{EMBEDDING_MODEL_NAME}",
                input_tokens=estimated_tokens,
                output_tokens=1024,
                latency_ms=latency_ms,
                success=True,
            )

            return embedding

        # For academic references without embeddings service, generate synthetic embedding
        def _generate_synthetic_embedding():
            import hashlib

            # Generate deterministic embedding based on doc_id for academic references
            hash_bytes = hashlib.sha256(doc_id.encode()).digest()
            # Convert hash bytes to embedding vector (first 384 floats matching embedding model dimension)
            embedding = [float(b) / 255.0 for b in hash_bytes]
            # Pad to embedding dimension (1024 for mxbai-embed-large)
            embedding_dimension = 1024
            while len(embedding) < embedding_dimension:
                embedding.extend(
                    embedding[: min(len(embedding), embedding_dimension - len(embedding))]
                )
            return embedding[:embedding_dimension]

        try:
            doc_embedding = _generate_doc_embedding()
        except Exception as e:
            # For academic references and when Ollama is unavailable, use synthetic embedding
            if doc_type == "academic_reference":
                logger.info(
                    f"Ollama unavailable for doc embedding, using synthetic for {doc_id}: {str(e)[:50]}"
                )
                doc_embedding = _generate_synthetic_embedding()
            else:
                raise

    # TODO should doc_metadata also store the doc_embedding? Chroma does not handle it as a raw value.

    source_category = metadata.get("source_category") or ""
    if not source_category and metadata.get("language"):
        source_category = "code"

    doc_metadata = {
        "doc_id": doc_id,
        "source": source_path,
        "file_path": metadata.get("file_path") or source_path,
        "repository": metadata.get("repository"),
        "project": metadata.get("project"),
        "version": version,
        "doc_type": doc_type,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "key_topics": key_topics_str,
        "summary": metadata.get("summary", ""),
        "summary_scores": json.dumps(metadata.get("summary_scores", {})),
        "health": json.dumps(health),
        "source_category": source_category,
        "timestamp": timestamp,
    }

    # Add code-specific metadata to document-level metadata (same as base_metadata for chunks)
    if "language" in metadata:
        doc_metadata["language"] = metadata["language"]
    if "service_name" in metadata:
        doc_metadata["service_name"] = metadata["service_name"]
    if "service_type" in metadata:
        doc_metadata["service_type"] = metadata["service_type"]
    if "external_dependencies" in metadata:
        deps = metadata["external_dependencies"]
        doc_metadata["dependencies"] = deps if isinstance(deps, str) else json.dumps(deps)
    if "internal_calls" in metadata:
        calls = metadata["internal_calls"]
        doc_metadata["internal_calls"] = calls if isinstance(calls, str) else json.dumps(calls)
    if "endpoints" in metadata:
        eps = metadata["endpoints"]
        doc_metadata["endpoints"] = eps if isinstance(eps, str) else json.dumps(eps)
    if "database_refs" in metadata:
        dbs = metadata["database_refs"]
        doc_metadata["db"] = dbs if isinstance(dbs, str) else json.dumps(dbs)
    if "message_queues" in metadata:
        queues = metadata["message_queues"]
        doc_metadata["queue"] = queues if isinstance(queues, str) else json.dumps(queues)
    if "exports" in metadata:
        exports = metadata["exports"]
        doc_metadata["exports"] = exports if isinstance(exports, str) else json.dumps(exports)

    # Add document to ChromaDB with retry logic
    # Use versioned ID to allow multiple versions of the same document
    versioned_doc_id = f"{doc_id}_v{version}"

    # Filter and sanitise metadata for ChromaDB - remove None, convert to valid types
    doc_metadata_sanitised = {}
    for k, v in doc_metadata.items():
        if v is None:
            continue  # Skip None values
        # Only keep strings, numbers, bools, or JSON strings
        if isinstance(v, (str, int, float, bool)):
            doc_metadata_sanitised[k] = v
        elif isinstance(v, (list, dict)):
            # Serialise complex types to JSON strings
            doc_metadata_sanitised[k] = json.dumps(v)
        else:
            # Convert other types to string
            doc_metadata_sanitised[k] = str(v)

    @retry_chromadb_call(
        max_retries=5, initial_delay=0.5, operation_name=f"doc_collection.add({versioned_doc_id})"
    )
    def _add_document(embedding_vector: List[float]):
        doc_collection.add(
            ids=[versioned_doc_id],
            embeddings=[embedding_vector],
            metadatas=[doc_metadata_sanitised],
            documents=[summary_text],  # or cleaned_text
        )

    try:
        _add_document(doc_embedding)
    except (InvalidArgumentError, ValueError, TypeError, Exception) as e:
        error_str = str(e)
        if "dimension" in error_str.lower() or "embedding" in error_str.lower():
            expected_dim = _parse_expected_dimension(error_str)
            if expected_dim and len(doc_embedding) > 0:
                try:
                    actual_dim = len(doc_embedding)
                    if actual_dim != expected_dim:
                        adjusted = _resize_embeddings([doc_embedding], expected_dim)[0]
                        logger.warning(
                            f"Adjusted document embedding from dim {actual_dim} to expected {expected_dim}"
                        )
                        _add_document(adjusted)
                    else:
                        # Dimensions match but still failed - re-raise original
                        raise
                except (InvalidArgumentError, ValueError, TypeError) as resize_err:
                    logger.error(
                        f"Failed to adjust document embedding: {resize_err}. Original error: {error_str}"
                    )
                    raise
            else:
                if not expected_dim:
                    logger.debug(f"Could not parse expected dimension from error: {error_str}")
                raise
        else:
            # Not a dimension-related error, re-raise as-is
            raise

    logger.info(f"Indexed document-level embedding for {versioned_doc_id}")
    audit(
        "doc_index_added",
        {"doc_id": doc_id, "version": doc_metadata["version"], "versioned_id": versioned_doc_id},
    )


# =========================
#  INCREMENTAL LOGIC
# =========================


@retry_chromadb_call(max_retries=5, initial_delay=0.5, operation_name="get_existing_doc_hash")
def get_existing_doc_hash(doc_id: str, chunk_collection: Collection) -> Optional[str]:
    """Retrieve the stored hash for a document from ChromaDB.

    Used for incremental ingestion to detect file changes.
    If hash matches current file, document is skipped.
    Filters by current embedding model to avoid retrieving outdated hashes from prior models.

    Automatically retries transient ChromaDB failures (connection errors,
    timeouts) with exponential backoff.

    Args:
        doc_id: Document identifier to query.
        chunk_collection: ChromaDB collection containing chunks.

    Returns:
        SHA-256 hash string if document exists, None otherwise.

    Note:
        All chunks for a document share the same hash,
        so only the first chunk's hash is returned.
    """
    try:
        # Check if collection is empty first (avoids schema issues with fresh collections)
        count = chunk_collection.count()
        if count == 0:
            return None

        results = chunk_collection.get(
            where={"$and": [{"doc_id": doc_id}, {"embedding_model": EMBEDDING_MODEL_NAME}]},
            include=["metadatas"],
        )
        if not results or len(results["metadatas"]) == 0:
            return None

        # All chunks for a doc share the same hash, take the first
        return results["metadatas"][0].get("hash")
    except Exception as e:
        # If query fails (e.g., fresh collection with no data), treat as not found
        # This prevents "Missing field" errors on newly created collections
        from scripts.utils.logger import get_logger

        logger = get_logger("ingest")
        logger.debug(f"Error querying existing hash for {doc_id}: {e}. Treating as new document.")
        return None


def delete_document_chunks(
    doc_id: str, chunk_collection: Collection, dry_run: bool = False
) -> None:
    """Delete all chunks for a given document from ChromaDB.

    Used during document updates or manual cleanup.
    Respects dry-run mode for safe testing.

    Automatically retries transient ChromaDB failures (connection errors,
    timeouts) with exponential backoff.

    Args:
        doc_id: Document identifier whose chunks should be deleted.
        chunk_collection: ChromaDB collection containing chunks.
        dry_run: If True, log action without executing deletion.

    Side Effects:
        - Deletes all chunks matching doc_id (if not dry_run)
        - Logs deletion event
        - Creates audit trail

    Warning:
        This permanently removes chunks. Consider version pruning
        instead if you need to maintain history.
    """
    logger = get_logger()
    if dry_run:
        logger.info(f"[DRY-RUN] Would delete chunks for doc_id={doc_id}")
        audit("dry_run_delete", {"doc_id": doc_id})
        return

    # Apply retry logic to delete operation
    @retry_chromadb_call(
        max_retries=5, initial_delay=0.5, operation_name=f"delete_chunks({doc_id})"
    )
    def _delete_chunks():
        chunk_collection.delete(where={"doc_id": doc_id})

    _delete_chunks()


def store_parent_chunks(
    doc_id: str,
    parent_chunks: List[Dict[str, Any]],
    chunk_collection: Collection,
    base_metadata: Dict[str, Any],
    dry_run: bool = False,
    full_text: str = "",
    doc_type: str = "unknown",
) -> None:
    """Store parent chunks without embeddings for context retrieval.

    Parent chunks provide rich context when child chunks are retrieved.
    They are NOT embedded (saving compute/storage) but are stored with
    metadata for later retrieval.

    Args:
        doc_id: Unique document identifier.
        parent_chunks: List of parent chunk dicts with {id, text, child_ids}.
        chunk_collection: ChromaDB collection for chunks.
        base_metadata: Shared metadata for all chunks (doc_id, version, etc.).
        dry_run: If True, simulate without writing to database.
        full_text: Full document text for enhanced metadata extraction.
        doc_type: Document type for metadata extraction.

    Side Effects:
        - Writes parent chunks to chunk_collection (without embeddings)
        - Logs storage events
        - Creates audit trail

    Note:
        Parent chunks are stored with is_parent=True flag for identification.
        Child chunks reference their parent via parent_id in metadata.
    """
    logger = get_logger()

    if dry_run:
        logger.info(f"[DRY-RUN] Would store {len(parent_chunks)} parent chunks for {doc_id}")
        return

    if not parent_chunks:
        logger.debug(f"No parent chunks to store for {doc_id}")
        return

    ids = []
    docs = []
    metas = []

    for chunk_idx, parent in enumerate(parent_chunks):
        parent_id = f"{doc_id}-{parent['id']}"
        ids.append(parent_id)
        docs.append(parent["text"])

        parent_meta = base_metadata.copy()
        parent_meta["is_parent"] = True
        parent_meta["parent_id"] = parent_id
        parent_meta["child_ids"] = json.dumps(parent["child_ids"])
        parent_meta["chunk_type"] = "parent"

        # Extract enhanced metadata if available
        if create_enhanced_metadata and full_text:
            try:
                enhanced = create_enhanced_metadata(
                    chunk_text=parent["text"],
                    chunk_index=chunk_idx,
                    total_chunks=len(parent_chunks),
                    doc_id=doc_id,
                    full_text=full_text,
                    doc_type=doc_type,
                )
                # Add enhanced fields to metadata
                parent_meta["heading_path"] = enhanced.heading_path or ""
                parent_meta["parent_section"] = enhanced.parent_section or ""
                parent_meta["section_title"] = getattr(enhanced, "section_title", "") or ""
                parent_meta["chapter"] = getattr(enhanced, "chapter", "") or ""
                parent_meta["section_depth"] = enhanced.section_depth or 0
                parent_meta["content_type"] = enhanced.content_type
                parent_meta["contains_code"] = enhanced.contains_code
                parent_meta["contains_table"] = enhanced.contains_table
                parent_meta["contains_diagram"] = enhanced.contains_diagram
                parent_meta["technical_entities"] = json.dumps(enhanced.technical_entities)
                parent_meta["code_language"] = enhanced.code_language or ""
                parent_meta["is_api_reference"] = enhanced.is_api_reference
                parent_meta["is_configuration"] = enhanced.is_configuration
            except Exception as e:
                logger.warning(
                    f"Failed to extract enhanced metadata for parent chunk {chunk_idx}: {e}"
                )

        metas.append(parent_meta)

    try:
        # Store parent chunks WITHOUT embeddings
        _add_to_collection_in_batches(
            chunk_collection=chunk_collection,
            ids=ids,
            documents=docs,
            metadatas=metas,
        )
        source_path = next((m.get("source_path", "unknown") for m in metas if m), "unknown")
        doc_type = next((m.get("doc_type", "unknown") for m in metas if m), "unknown")
        provider = next((m.get("source", "unknown") for m in metas if m), "unknown")
        logger.info(
            f"Stored {len(parent_chunks)} parent chunks | "
            f"doc_id={doc_id} | "
            f"source={source_path[:50]}... | "
            f"type={doc_type} | "
            f"provider={provider} | "
            f"storage=metadata_only"
        )
        audit(
            "parent_chunks_stored",
            {
                "doc_id": doc_id,
                "parent_count": len(parent_chunks),
                "source_path": source_path,
                "doc_type": doc_type,
                "provider": provider,
                "storage_type": "metadata_only",
            },
        )
    except (InvalidArgumentError, ValueError, TypeError, Exception) as e:
        # If ChromaDB tries to auto-generate embeddings and fails with dimension mismatch,
        # provide zero-padded embeddings with the expected dimension
        error_str = str(e)
        if "dimension" in error_str.lower() or "embedding" in error_str.lower():
            expected_dim = _parse_expected_dimension(error_str)
            if expected_dim:
                logger.debug(
                    f"Collection requires embeddings - providing zero-padded embeddings "
                    f"with dimension {expected_dim} for parent chunks"
                )
                try:
                    # Provide dummy zero-padded embeddings with the correct dimension
                    dummy_embeddings = [[0.0] * expected_dim for _ in ids]
                    _add_to_collection_in_batches(
                        chunk_collection=chunk_collection,
                        ids=ids,
                        documents=docs,
                        metadatas=metas,
                        embeddings=dummy_embeddings,
                    )
                    logger.info(f"Stored {len(parent_chunks)} parent chunks with dummy embeddings")
                except Exception as retry_err:
                    logger.error(
                        f"Failed to store parent chunks with dummy embeddings: {retry_err}. "
                        f"Original error: {error_str}",
                        exc_info=True,
                    )
                    raise
            else:
                logger.error(
                    f"Could not parse expected dimension from error: {error_str}", exc_info=True
                )
                raise
        else:
            logger.error(f"Failed to store parent chunks for {doc_id}: {error_str}", exc_info=True)
            raise


def store_child_chunks(
    doc_id: str,
    child_chunks: List[Dict[str, str]],
    chunk_collection: Collection,
    base_metadata: Dict[str, Any],
    dry_run: bool = False,
    full_text: str = "",
    doc_type: str = "unknown",
) -> None:
    """Store child chunks with embeddings for semantic search.

    Child chunks are the searchable units that get embedded and indexed.
    They reference their parent chunk via parent_id in metadata for
    context retrieval during RAG queries.

    Args:
        doc_id: Unique document identifier.
        child_chunks: List of child chunk dicts with {id, text, parent_id}.
        chunk_collection: ChromaDB collection for chunks.
        base_metadata: Shared metadata for all chunks (doc_id, version, etc.).
        dry_run: If True, simulate without writing to database.
        full_text: Full document text for enhanced metadata extraction.
        doc_type: Document type for metadata extraction.

    Side Effects:
        - Writes child chunks to chunk_collection (WITH embeddings via ChromaDB)
        - Logs storage events
        - Creates audit trail

    Note:
        Child chunks are stored with is_parent=False and chunk_type="child".
        Embeddings are auto-generated by ChromaDB for these chunks.
    """
    logger = get_logger()

    if dry_run:
        logger.info(f"[DRY-RUN] Would store {len(child_chunks)} child chunks for {doc_id}")
        return

    if not child_chunks:
        logger.debug(f"No child chunks to store for {doc_id}")
        return

    ids = []
    docs = []
    metas = []

    for chunk_idx, child in enumerate(child_chunks):
        # Construct child ID with doc_id prefix: "{doc_id}-{child_id}"
        child_id = f"{doc_id}-{child['id']}"
        ids.append(child_id)
        docs.append(child["text"])

        child_meta = base_metadata.copy()
        child_meta["is_parent"] = False
        child_meta["chunk_type"] = "child"
        # Store parent reference as constructed parent ID: "{doc_id}-parent_{idx}"
        child_meta["parent_id"] = f"{doc_id}-{child['parent_id']}"

        # Extract enhanced metadata if available
        if create_enhanced_metadata and full_text:
            try:
                enhanced = create_enhanced_metadata(
                    chunk_text=child["text"],
                    chunk_index=chunk_idx,
                    total_chunks=len(child_chunks),
                    doc_id=doc_id,
                    full_text=full_text,
                    doc_type=doc_type,
                )
                # Add enhanced fields to metadata
                child_meta["heading_path"] = enhanced.heading_path or ""
                child_meta["parent_section"] = enhanced.parent_section or ""
                child_meta["section_title"] = getattr(enhanced, "section_title", "") or ""
                child_meta["chapter"] = getattr(enhanced, "chapter", "") or ""
                child_meta["section_depth"] = enhanced.section_depth or 0
                child_meta["content_type"] = enhanced.content_type
                child_meta["contains_code"] = enhanced.contains_code
                child_meta["contains_table"] = enhanced.contains_table
                child_meta["contains_diagram"] = enhanced.contains_diagram
                child_meta["technical_entities"] = json.dumps(enhanced.technical_entities)
                child_meta["code_language"] = enhanced.code_language or ""
                child_meta["is_api_reference"] = enhanced.is_api_reference
                child_meta["is_configuration"] = enhanced.is_configuration
            except Exception as e:
                logger.warning(
                    f"Failed to extract enhanced metadata for child chunk {chunk_idx}: {e}"
                )

        metas.append(child_meta)

    # Generate embeddings manually to have control over dimension handling
    from langchain_ollama import OllamaEmbeddings

    embed_model = _create_embed_model()

    try:
        # Generate embeddings for all child chunk texts
        embeddings = embed_model.embed_documents(docs)

        # Store child chunks with manually generated embeddings
        _add_to_collection_in_batches(
            chunk_collection=chunk_collection,
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings,
        )
        source_path = next((m.get("source_path", "unknown") for m in metas if m), "unknown")
        doc_type = next((m.get("doc_type", "unknown") for m in metas if m), "unknown")
        provider = next((m.get("source", "unknown") for m in metas if m), "unknown")
        logger.info(
            f"Stored {len(child_chunks)} child chunks | "
            f"doc_id={doc_id} | "
            f"source={source_path[:50]}... | "
            f"type={doc_type} | "
            f"provider={provider} | "
            f"storage=with_embeddings"
        )
        audit(
            "child_chunks_stored",
            {"doc_id": doc_id, "child_count": len(child_chunks), "storage_type": "embedded"},
        )
    except (InvalidArgumentError, ValueError, TypeError, Exception) as e:
        # If there's an embedding dimension mismatch, resize embeddings
        error_str = str(e)
        if "dimension" in error_str.lower() or "embedding" in error_str.lower():
            expected_dim = _parse_expected_dimension(error_str)
            if expected_dim and expected_dim != len(embeddings[0]):
                logger.warning(
                    f"Dimension mismatch storing child chunks for {doc_id}. "
                    f"Expected {expected_dim}, got {len(embeddings[0])}. Resizing embeddings..."
                )
                # Resize embeddings to match expected dimension
                resized_embeddings = _resize_embeddings(embeddings, expected_dim)
                try:
                    _add_to_collection_in_batches(
                        chunk_collection=chunk_collection,
                        ids=ids,
                        documents=docs,
                        metadatas=metas,
                        embeddings=resized_embeddings,
                    )
                    logger.info(
                        f"Successfully stored {len(child_chunks)} child chunks with resized embeddings"
                    )
                    audit(
                        "child_chunks_stored",
                        {
                            "doc_id": doc_id,
                            "child_count": len(child_chunks),
                            "storage_type": "embedded_resized",
                            "original_dim": len(embeddings[0]),
                            "resized_dim": expected_dim,
                        },
                    )
                except Exception as retry_err:
                    logger.error(
                        f"Failed to store child chunks with resized embeddings: {retry_err}. "
                        f"Original error: {error_str}",
                        exc_info=True,
                    )
                    raise
            else:
                logger.error(
                    f"Could not parse expected dimension from error: {error_str}", exc_info=True
                )
                raise
        else:
            logger.error(f"Failed to store child chunks for {doc_id}: {error_str}", exc_info=True)
            raise


def get_parent_for_child(
    child_chunk_id: str,
    chunk_collection: Collection,
) -> Optional[Dict[str, Any]]:
    """Retrieve parent chunk text for a given child chunk ID.

    Used during retrieval to provide rich context to the LLM when
    child chunks are matched in semantic search.

    Args:
        child_chunk_id: ID of the child chunk.
        chunk_collection: ChromaDB collection containing chunks.

    Returns:
        Dict with parent chunk data {id, text, metadata} or None if not found.

    Example:
        >>> parent = get_parent_for_child("doc123-child_0_2", collection)
        >>> if parent:
        ...     llm_context = parent["text"]  # Rich parent context
    """
    logger = get_logger()

    try:
        # Get child chunk to find parent_id
        child_result = chunk_collection.get(ids=[child_chunk_id], include=["metadatas"])

        if not child_result or not child_result["metadatas"]:
            logger.warning(f"Child chunk not found: {child_chunk_id}")
            return None

        child_meta = child_result["metadatas"][0]
        parent_id = child_meta.get("parent_id")

        if not parent_id:
            logger.debug(f"No parent_id for child chunk: {child_chunk_id}")
            return None

        # Retrieve parent chunk
        parent_result = chunk_collection.get(ids=[parent_id], include=["documents", "metadatas"])

        if not parent_result or not parent_result["documents"]:
            logger.warning(f"Parent chunk not found: {parent_id}")
            return None

        return {
            "id": parent_id,
            "text": parent_result["documents"][0],
            "metadata": parent_result["metadatas"][0],
        }

    except Exception as e:
        logger.error(f"Error retrieving parent for child {child_chunk_id}: {e}", exc_info=True)
        return None


def batch_get_parents_for_children(
    child_chunk_ids: List[str],
    chunk_collection: Collection,
    logger=None,
) -> Dict[str, Dict[str, Any]]:
    """Batch retrieve parent chunks for multiple child chunk IDs.

    Optimised for retrieving multiple parents in a single query.
    More efficient than calling get_parent_for_child() repeatedly.

    Args:
        child_chunk_ids: List of child chunk IDs.
        chunk_collection: ChromaDB collection containing chunks.
        logger: Optional logger instance. If not provided, uses ingest logger.

    Returns:
        Dict mapping child_id -> parent data {id, text, metadata}.
        Missing parents are excluded from results.

    Example:
        >>> parents = batch_get_parents_for_children(["child_0_1", "child_0_2"], coll)
        >>> for child_id, parent in parents.items():
        ...     print(f"{child_id} -> {parent['id']}")
    """
    if logger is None:
        logger = get_logger()

    if not child_chunk_ids:
        return {}

    # Filter out empty/None values and deduplicate chunk IDs (ChromaDB requires unique, non-empty IDs)
    # Preserve original list for mapping but query with unique IDs only
    filtered_ids = [cid for cid in child_chunk_ids if cid]  # Remove empty strings and None
    unique_child_ids = list(dict.fromkeys(filtered_ids))  # Preserves order, removes duplicates

    if not unique_child_ids:
        return {}

    try:
        # Get all child chunks to extract parent_ids
        children_result = chunk_collection.get(ids=unique_child_ids, include=["metadatas"])

        if not children_result or not children_result["metadatas"]:
            return {}

        # Build mapping of child_id -> parent_id
        child_to_parent = {}
        parent_ids = set()

        for idx, child_id in enumerate(children_result["ids"]):
            child_meta = children_result["metadatas"][idx]
            parent_id = child_meta.get("parent_id")
            if parent_id:
                child_to_parent[child_id] = parent_id
                parent_ids.add(parent_id)

        if not parent_ids:
            return {}

        # Batch retrieve all parent chunks
        parents_result = chunk_collection.get(
            ids=list(parent_ids), include=["documents", "metadatas"]
        )

        if not parents_result or not parents_result["documents"]:
            return {}

        # Build parent_id -> parent data mapping
        parents_map = {}
        for idx, parent_id in enumerate(parents_result["ids"]):
            parents_map[parent_id] = {
                "id": parent_id,
                "text": parents_result["documents"][idx],
                "metadata": parents_result["metadatas"][idx],
            }

        # Map children to their parents
        result = {}
        for child_id, parent_id in child_to_parent.items():
            if parent_id in parents_map:
                result[child_id] = parents_map[parent_id]

        logger.debug(f"Retrieved {len(result)} parents for {len(child_chunk_ids)} children")
        return result

    except Exception as e:
        logger.error(f"Error batch retrieving parents: {e}", exc_info=True)
        return {}
