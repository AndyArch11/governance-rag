"""LLM-based answer generation from retrieved context.

Orchestrates the complete RAG pipeline:
  1. Retrieve semantically similar chunks from the vector DB
  2. Assemble a prompt with context and user query
    3. Generate answer using a configured LLM (e.g., Ollama)
  4. Return structured response with chunks, sources, and timing

Provides end-to-end error handling, logging, and performance metrics.
Optimised for governance/security Q&A with grounded, faithful responses.
Includes code-aware prompt generation and response formatting for code queries.
"""

import time
from typing import Any, Dict, List, Optional, Union

# Collection typing (best-effort)
try:
    from chromadb.api.models.Collection import (  # type: ignore  # noqa: WPS433,E402
        Collection as ChromaDBCollection,
    )
except Exception:
    ChromaDBCollection = Any  # type: ignore

try:
    from scripts.ingest.chromadb_sqlite import ChromaSQLiteCollection  # noqa: WPS433,E402
except Exception:
    ChromaSQLiteCollection = Any  # type: ignore

Collection = Union[ChromaDBCollection, ChromaSQLiteCollection, Any]

from langchain_ollama import OllamaLLM

from scripts.utils.logger import create_module_logger

from .assemble import (
    ACADEMIC_SYSTEM_PROMPT,
    build_academic_aware_prompt,
    build_code_aware_prompt,
    build_prompt,
    extract_language_from_metadata,
    format_code_response,
    include_git_links,
)
from .assemble import is_academic_query as _is_academic_query
from .rag_config import RAGConfig

get_logger, audit = create_module_logger("rag")
from scripts.utils.adaptive_rate_limiter import (
    get_adaptive_rate_limiter,
    init_adaptive_rate_limiter,
)
from scripts.utils.db_factory import get_cache_client
from scripts.utils.metrics_export import get_metrics_collector
from scripts.utils.monitoring import get_perf_metrics, get_token_counter, init_monitoring
from scripts.utils.rate_limiter import get_rate_limiter
from scripts.utils.retry_utils import retry_ollama_call

from .retrieve import (
    EmbeddingDimensionMismatch,
    detect_filters_from_query,
    explain_retrieval,
    retrieve,
)

try:
    from scripts.rag.adaptive_weighting import get_adaptive_weight_learner
except ImportError:
    get_adaptive_weight_learner = None

# Global config and logger; these are singletons and thread-safe
config = RAGConfig()
logger = get_logger()  # Initialise logger from module-level create_module_logger

# Initialise monitoring infrastructure
init_monitoring()
token_counter = get_token_counter()
perf_metrics = get_perf_metrics()
metrics_collector = get_metrics_collector()

# Initialise adaptive rate limiting if enabled
if config.enable_adaptive_rate_limiting:
    init_adaptive_rate_limiter(
        initial_rate=10.0,
        max_rate=50.0,
    )
    logger.info("Adaptive rate limiting enabled for LLM calls")
else:
    logger.info("Adaptive rate limiting disabled")


@retry_ollama_call(max_retries=3, initial_delay=1.0, operation_name="generate_answer")
def _invoke_llm_with_retry(prompt: str, temperature: Optional[float] = None) -> str:
    """Invoke LLM with retry and optional rate limiting.

    Args:
        prompt: The prompt to send to the LLM.
        temperature: Optional custom temperature (0.0-1.0). If None, uses config default.

    Returns:
        Generated text from the LLM.
    """
    limiter = get_rate_limiter()
    if limiter:
        limiter.acquire()

    # Apply adaptive rate limiting if enabled
    adaptive_limiter = None
    if config.enable_adaptive_rate_limiting:
        adaptive_limiter = get_adaptive_rate_limiter()
        if adaptive_limiter:
            adaptive_limiter.acquire(blocking=True)

    llm = _get_llm(temperature=temperature)

    start_time = time.perf_counter()
    try:
        response = llm.invoke(prompt)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Record in adaptive rate limiter if enabled
        if adaptive_limiter:
            adaptive_limiter.record_request(latency_ms=latency_ms, success=True, status_code=200)

        return response
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        if adaptive_limiter:
            adaptive_limiter.record_request(
                latency_ms=latency_ms, success=False, error_type=type(e).__name__
            )
        raise


def _get_llm(temperature: Optional[float] = None) -> OllamaLLM:
    """Factory function to create LLM instance with current config settings.

    Creates a fresh OllamaLLM instance using the latest configuration values
    (model name and temperature). This allows environment variable overrides to
    take effect at runtime without requiring a process restart.

    Args:
        temperature: Optional custom temperature (0.0-1.0). If None, uses config default.

    Returns:
        OllamaLLM instance configured with current RAGConfig settings or custom temperature.

    Note: Called once per answer() invocation; temperature from config is passed
    but may not be used by all LLM backends (OllamaLLM support depends on version).
    """
    temp = temperature if temperature is not None else config.temperature
    return OllamaLLM(model=config.model_name, temperature=temp)


def _is_code_query(query: str) -> bool:
    """Detect if query is code-related.

    Uses detect_filters_from_query() to determine if this is a code query.
    Code queries get special prompt formatting and response enhancement.
    """
    try:
        filters = detect_filters_from_query(query)
        return filters.get("code_filter", False) or "code" in filters.get("categories", [])
    except Exception:
        return False


def record_query_sample_for_adaptive_learning(
    query: str,
    response: Dict[str, Any],
    user_relevancy_score: Optional[float] = None,
) -> None:
    """Record a query sample for adaptive weight learning.

    Captures query execution metrics for training the adaptive weight learner.
    This enables the system to learn optimal vector/keyword weights from actual
    query performance and user feedback over time.

    Args:
        query: The user query text
        response: The response dict from answer()
        user_relevancy_score: Optional user relevancy rating (0.0-1.0).
               If not provided, estimates from retrieval similarity scores.

    Note:
        Gracefully handles unavailable adaptive learner without raising exceptions.
    """
    if get_adaptive_weight_learner is None:
        return

    try:
        learner = get_adaptive_weight_learner()
        if learner is None:
            return

        # Extract metrics from response
        is_code = response.get("is_code_query", False)
        sources = response.get("sources", [])
        total_time = response.get("total_time", 0.0)
        generation_time = response.get("generation_time", 0.0)

        # Estimate similarity scores from sources
        similarities = [s.get("distance", 0.0) for s in sources if s.get("distance") is not None]
        embed_sim_avg = (1.0 - (sum(similarities) / len(similarities))) if similarities else 0.5

        # Use BM25 scores if available in metadata
        bm25_scores = [
            s.get("bm25_score", 0.5)
            for s in sources
            if s.get("retrieval_method") in ("keyword", "hybrid", None)
        ]
        bm25_score_avg = (sum(bm25_scores) / len(bm25_scores)) if bm25_scores else 0.5

        # Estimate query type
        query_type = "code" if is_code else "natural"
        query_length = len(query.split())
        result_count = response.get("retrieval_count", 0)

        # Estimate relevancy score from generation context
        if user_relevancy_score is None:
            # Heuristic: if we got results and non-empty answer, assume moderate relevance
            answer_length = len(response.get("answer", "").split())
            if result_count > 0 and answer_length > 10:
                user_relevancy_score = 0.7  # Moderate relevance (had results, generated answer)
            elif result_count > 0:
                user_relevancy_score = 0.5  # Had results but short answer
            else:
                user_relevancy_score = 0.3  # No results

        # Get current weights from config
        try:
            from scripts.rag.hybrid_search_weights import get_weight_manager

            weight_manager = get_weight_manager()
            if weight_manager:
                vector_weight = weight_manager.weights.vector_weight
                keyword_weight = weight_manager.weights.keyword_weight
            else:
                vector_weight = 0.6
                keyword_weight = 0.4
        except Exception:
            vector_weight = 0.6
            keyword_weight = 0.4

        # Record the sample
        learner.record_sample(
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            relevancy_score=user_relevancy_score,
            query_type=query_type,
            query_length=query_length,
            result_count=result_count,
            embedding_sim_avg=embed_sim_avg,
            bm25_score_avg=bm25_score_avg,
        )

        logger.debug(
            f"Recorded adaptive learning sample: "
            f"relevancy={user_relevancy_score:.2f}, "
            f"query_type={query_type}, "
            f"embedding_sim={embed_sim_avg:.2f}, "
            f"bm25_score={bm25_score_avg:.2f}"
        )

        # Periodically get recommendation and apply if available
        # Apply every 20 recorded samples to avoid too frequent weight changes
        if len(learner.samples) % 20 == 0:
            rec = learner.get_recommendation()
            if rec is not None:
                logger.info(
                    f"Adaptive weight learning ready: "
                    f"vector={rec.vector_weight:.2f}, "
                    f"keyword={rec.keyword_weight:.2f}, "
                    f"confidence={rec.confidence:.0%}, "
                    f"improvement={rec.expected_improvement:.0%}"
                )
                learner.apply_recommendation_to_weights()

    except Exception as e:
        logger.debug(f"Failed to record adaptive learning sample: {e}")
        # Don't raise - adaptive learning is optional enhancement


def _is_code_query(query: str) -> bool:
    """Detect if query is code-related.

    Uses detect_filters_from_query() to determine if this is a code query.
    Code queries get special prompt formatting and response enhancement.

    Args:
        query: User query string.

    Returns:
        True if query is detected as code-related, False otherwise.
    """
    try:
        filters = detect_filters_from_query(query)
        source_category = (filters.get("source_category") or "").lower()
        categories = {
            item.lower() for item in filters.get("categories", []) if isinstance(item, str)
        }
        is_code = (
            source_category == "code" or source_category.endswith("_code") or "code" in categories
        )
        if is_code:
            logger.debug(f"Code query detected with filters: {filters}")
        return is_code
    except Exception as e:
        logger.debug(f"Filter detection failed: {e}, assuming non-code query")
        return False


def _enhance_code_response(
    answer: str,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Enhance code response with formatting and links.

    Applies code-specific formatting and includes Git hosting references.

    Args:
        answer: Generated answer text.
        metadata: Optional list of metadata dicts for source chunks.

    Returns:
        Enhanced answer with code formatting and links.
    """
    # Extract language from metadata for proper syntax highlighting
    language = extract_language_from_metadata(metadata)

    # Format code snippets with markdown
    formatted_answer = format_code_response(answer, language)

    # Include Git hosting links if available
    enhanced_answer = include_git_links(formatted_answer, metadata)

    return enhanced_answer


def answer(
    query: str,
    collection: Collection,
    k: Optional[int] = None,
    temperature: Optional[float] = None,
    custom_role: Optional[str] = None,
    persona: Optional[str] = None,
    enable_code_detection: bool = True,
    allow_code_category: bool = True,
) -> Dict[str, Any]:
    """Generate an answer to a query using the full RAG pipeline.

    Orchestrates retrieval, prompt assembly, and LLM generation in sequence:
    1. Retrieve k most similar chunks from ChromaDB (semantic search)
    2. Detect if query is code-related for special handling
    3. Build a grounded prompt (code-aware or standard)
    4. Generate answer using configured LLM
    5. Enhance response with code formatting and Git hosting links if needed
    6. Aggregate results with timing and source metadata

    Args:
        query: User question or query to answer. Must be non-empty.
               Examples: "What is MFA?" (governance) or "Show Java services" (code)
        collection: ChromaDB collection to search against.
        k: Number of chunks to retrieve. If None, uses config.k_results (default: 5).
        temperature: LLM temperature (0.0-1.0). If None, uses config default (0.3).
        custom_role: Optional custom system prompt. If None, uses default prompts.
        persona: Optional persona name ("supervisor", "assessor", "researcher") used
            to apply persona-aware filtering and reranking during retrieval.
        enable_code_detection: Whether to auto-detect code queries (default: True).
            Set to False if Code-Aware Context is disabled in dashboard.
        allow_code_category: Whether to include code results (default: True).
            Set to False if "Code" is unchecked in Result Type filters.

    Returns:
        Dictionary with structured response:
            - answer (str): Generated response text from LLM.
            - chunks (List[str]): Retrieved context chunks passed to LLM.
            - sources (List[Dict]): Metadata for each chunk (doc ID, language, service, etc.).
            - generation_time (float): LLM inference time in seconds.
            - total_time (float): End-to-end time including retrieval and prompt assembly.
            - retrieval_count (int): Number of chunks retrieved.
            - model (str): Name of LLM model used.
            - is_code_query (bool): Whether query was detected as code-related.

    Raises:
        ValueError: If query is empty or invalid.
        Exception: If retrieval or generation fails (not caught; should be handled
                   by caller for production use).

    Example:
        >>> response = answer("What is MFA?", collection, k=3)
        >>> response["answer"]
        'Multi-factor authentication (MFA) requires two or more...'
        >>> response["retrieval_count"]
        3
        >>> response["is_code_query"]
        False

        >>> code_response = answer("Show Java authentication services", collection)
        >>> code_response["is_code_query"]
        True
        >>> "```java" in code_response["answer"] or "Java" in code_response["answer"]
        True

    Note:
        - If no chunks are retrieved, returns a graceful "no results" response.
                - Code queries (detected via language/pattern keywords) receive special
                    formatting with code blocks, Git hosting links, and metadata.
        - Standard queries receive governance/security-focused prompting.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if k is None:
        k = config.k_results

    # Academic personas should not use code detection/filtering
    # (academic queries shouldn't be classified as code queries)
    if persona in ("supervisor", "assessor", "researcher"):
        enable_code_detection = False
        allow_code_category = True  # Still allow code chunks if explicitly requested

    start_time = time.perf_counter()

    try:
        # Step 1: Retrieve relevant chunks via semantic similarity
        logger.info(f"Retrieving {k} chunks for query: {query[:80]}...")
        try:
            chunks, sources = retrieve(
                query,
                collection,
                k=k,
                persona=persona,
                enable_code_detection=enable_code_detection,
                allow_code_category=allow_code_category,
            )
        except EmbeddingDimensionMismatch as dim_err:
            elapsed = time.perf_counter() - start_time
            temp = temperature if temperature is not None else config.temperature
            logger.error(f"Embedding dimension mismatch during retrieval: {dim_err}")
            return {
                "answer": (
                    "Embedding dimension mismatch detected. "
                    "The query embedding size does not match the collection embeddings. "
                    "Please re-ingest after fixing the embedding configuration."
                ),
                "chunks": [],
                "sources": [],
                "generation_time": round(elapsed, 2),
                "total_time": round(elapsed, 2),
                "retrieval_count": 0,
                "model": config.model_name,
                "temperature": temp,
                "is_code_query": False,
                "error": "embedding_dimension_mismatch",
                "error_details": str(dim_err),
            }

        # Graceful fallback if no relevant chunks found (e.g., empty KB)
        if not chunks:
            logger.warning("No chunks retrieved")
            elapsed = time.perf_counter() - start_time
            temp = temperature if temperature is not None else config.temperature
            response = {
                "answer": "No relevant information found in the knowledge base.",
                "chunks": [],
                "sources": [],
                "generation_time": round(elapsed, 2),
                "total_time": round(elapsed, 2),
                "retrieval_count": 0,
                "model": config.model_name,
                "temperature": temp,
                "is_code_query": False,
            }
            # Minimal audit event to satisfy consumers expecting only the query field
            audit("answer_no_results", {"query": query})

            # Detailed audit for telemetry
            audit(
                "answer_no_results",
                {
                    "query": query,
                    "query_length": len(query),
                    "chunks_retrieved": 0,
                    "generation_time": response["generation_time"],
                    "total_time": response["total_time"],
                    "model": config.model_name,
                    "temperature": config.temperature,
                    "is_code_query": False,
                },
            )
            return response

        # Step 2: Detect query type (code, academic, or general) for appropriate prompt
        is_code = _is_code_query(query)
        is_academic = _is_academic_query(sources) if sources else False

        # Step 3: Build structured prompt (code-aware, academic-aware, or standard)
        query_type = "code-aware" if is_code else ("academic-aware" if is_academic else "")
        logger.info(f"Building {query_type} prompt with {len(chunks)} chunks")

        prompt_kwargs = {"custom_role": custom_role} if custom_role is not None else {}

        if is_code:
            prompt = build_code_aware_prompt(query, chunks, metadata=sources, **prompt_kwargs)
        elif is_academic:
            prompt = build_academic_aware_prompt(query, chunks, metadata=sources, **prompt_kwargs)
        else:
            prompt = build_prompt(query, chunks, metadata=sources, **prompt_kwargs)

        # Step 4: Generate answer using LLM (blocking call; consider async for scale)
        generation_start = time.perf_counter()
        temp = temperature if temperature is not None else config.temperature
        logger.info(f"Generating answer with LLM (model={config.model_name}, temperature={temp})")

        # Invoke LLM with monitoring
        generated_answer = _invoke_llm_with_retry(prompt, temperature=temperature)
        generation_time = time.perf_counter() - generation_start

        # Track tokens (estimate based on text length for now)
        # TODO: Use actual tokeniser for precise counts
        input_tokens = len(prompt) // 4  # ~4 chars per token estimate
        output_tokens = len(generated_answer) // 4

        # Record metrics
        token_counter.record_tokens(
            model=config.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=True,
        )
        perf_metrics.record_generation(
            latency_ms=generation_time * 1000,
            model=config.model_name,
        )
        metrics_collector.record_llm_call(
            model=config.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=generation_time * 1000,
            success=True,
        )

        # Step 5: Enhance response for code queries
        if is_code:
            generated_answer = _enhance_code_response(generated_answer, sources)

        total_time = time.perf_counter() - start_time

        # Step 6: Generate explainability data
        explainability = explain_retrieval(query, chunks, sources, k)

        temp = temperature if temperature is not None else config.temperature
        response = {
            "answer": generated_answer.strip(),
            "chunks": chunks,
            "sources": sources,
            "generation_time": round(generation_time, 2),
            "total_time": round(total_time, 2),
            "retrieval_count": len(chunks),
            "model": config.model_name,
            "temperature": temp,
            "is_code_query": is_code,
            "is_academic_query": is_academic,
            "explainability": explainability,
        }

        logger.info(
            f"Answer generated in {total_time:.2f}s (code_query={is_code}, academic_query={is_academic})"
        )
        audit(
            "answer_generated",
            {
                "query_length": len(query),
                "chunks_retrieved": len(chunks),
                "generation_time": round(generation_time, 2),
                "total_time": round(total_time, 2),
                "model": config.model_name,
                "temperature": config.temperature,
                "is_code_query": is_code,
            },
        )

        # Log query analytics for performance tracking
        try:
            cache = get_cache_client()
            if cache and hasattr(cache, "log_query_analytics"):
                # Calculate similarity scores from sources
                similarities = [
                    s.get("distance", 0.0) for s in sources if s.get("distance") is not None
                ]
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
                max_similarity = max(similarities) if similarities else 0.0

                retrieval_time_ms = (total_time - generation_time) * 1000

                cache.log_query_analytics(
                    query_text=query,
                    k_results=k,
                    retrieval_time_ms=retrieval_time_ms,
                    generation_time_ms=generation_time * 1000,
                    num_chunks_retrieved=len(chunks),
                    avg_similarity_score=avg_similarity,
                    max_similarity_score=max_similarity,
                    cache_hit=False,  # TODO: Track actual cache hits
                    is_code_query=is_code,
                    model_name=config.model_name,
                    temperature=temp,
                    metadata={
                        "query_length": len(query),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                )
        except Exception as e:
            logger.debug(f"Failed to log query analytics: {e}")

        return response

    except Exception as e:
        logger.error(f"Answer generation failed: {e}", exc_info=True)
        audit("answer_error", {"query": query[:100], "error": str(e)})

        # Record error in metrics
        metrics_collector.record_llm_call(
            model=config.model_name,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            success=False,
        )

        raise
