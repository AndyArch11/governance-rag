"""Document chunk retrieval from ChromaDB or SQLite backend.

Performs semantic similarity search to find relevant chunks for a query.
Returns retrieved text chunks along with their metadata (source, version, etc.)
for context and source attribution in downstream generation.

Enhanced Features:
  - Metadata-based filtering (pre-filter by category, type, content flags)
  - Auto-detection of filters from natural language queries
  - Context reconstruction (fetch neighbouring chunks)
  - Lightweight re-ranking without LLM overhead
  - Cross-encoder reranking for improved relevance (technical docs)
  - Context caching for frequently accessed entities
  - Graph-enhanced retrieval with relationship expansion

Note: Supports optional learned reranking using cross-encoders.
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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

from scripts.utils.logger import create_module_logger

from .context_cache import get_context_cache

get_logger, audit = create_module_logger("rag")
from scripts.utils.metrics_export import get_metrics_collector
from scripts.utils.monitoring import get_perf_metrics, init_monitoring
from scripts.utils.rate_limiter import get_rate_limiter
from scripts.utils.retry_utils import retry_chromadb_call, retry_ollama_call

try:
    from scripts.ingest.vectors import batch_get_parents_for_children
except ImportError:
    batch_get_parents_for_children = None

try:
    from scripts.search.bm25_retrieval import BM25Retriever
except ImportError:
    BM25Retriever = None

try:
    from scripts.search.reranker import RankerResult, RerankerConfig, rerank_results
except ImportError:
    RerankerConfig = None
    rerank_results = None
    RankerResult = None

try:
    from scripts.rag.hybrid_search_weights import HybridSearchWeights, get_weight_manager
except ImportError:
    get_weight_manager = None
    HybridSearchWeights = None

try:
    from scripts.rag.query_expansion import get_expansion_cache, get_query_expander
except ImportError:
    get_query_expander = None
    get_expansion_cache = None

try:
    from scripts.search.persona_retrieval import apply_persona_reranking
except ImportError:
    apply_persona_reranking = None

try:
    from scripts.ingest.vectors import EMBEDDING_MODEL_NAME
except Exception:
    # TODO: Address string literal usage below., reconsider testing approach requiring this.
    EMBEDDING_MODEL_NAME = "mxbai-embed-large"

# Initialise monitoring
init_monitoring()
perf_metrics = get_perf_metrics()
metrics_collector = get_metrics_collector()


class EmbeddingDimensionMismatch(ValueError):
    """Raised when query embedding dimension does not match collection embeddings."""

    def __init__(self, expected_dim: int, actual_dim: int, model_name: str) -> None:
        super().__init__(
            f"Embedding dimension mismatch: expected {expected_dim}D in collection, "
            f"got {actual_dim}D from model '{model_name}'."
        )
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim
        self.model_name = model_name


def _get_collection_embedding_dim(collection: Collection) -> Optional[int]:
    """Return embedding dimension for a collection, or None if empty/unavailable."""
    try:
        sample = collection.get(limit=1, include=["embeddings"])
        embeddings = sample.get("embeddings") or []
        if embeddings and len(embeddings) > 0:
            return len(embeddings[0])
    except Exception:
        return None
    return None


# ------------------------------
# Test-friendly helper functions
# ------------------------------


def _combine_results(
    vector_chunks: List[str],
    vector_metadata: List[Dict],
    keyword_chunks: List[str],
    keyword_metadata: List[Dict],
    counts_chunks: List[str],
    counts_metadata: List[Dict],
    k: int,
    logger,
    use_weights: bool = True,
) -> Tuple[List[str], List[Dict], int, int]:
    """Combine vector/keyword/counts results with optional weighted combination.

    Returns final chunks/metadata limited to k, and counts of vector/keyword items
    for logging/audit. This helper is pure with respect to inputs and is easy to unit test.

    Args:
        use_weights: If True, use HybridSearchWeightManager for intelligent combination

    Returns:
        Tuple of (final_chunks, final_metadata, vector_count, keyword_count)
    """
    combined_chunks: List[str] = []
    combined_metadata: List[Dict] = []
    seen_chunks: Set[str] = set()

    # Always prepend counts summary first (if available)
    for chunk, meta in zip(counts_chunks, counts_metadata):
        if chunk not in seen_chunks:
            seen_chunks.add(chunk)
            combined_chunks.append(chunk)
            combined_metadata.append(meta or {})

    # If weights available and enabled, use weighted combination
    if use_weights and get_weight_manager:
        try:
            weight_manager = get_weight_manager()

            # Normalise scores to 0-1 range
            vector_scores = [
                1.0 - (i * 0.1) for i in range(len(vector_chunks))
            ]  # Decreasing from 1.0
            keyword_scores = [1.0 - (i * 0.1) for i in range(len(keyword_chunks))]

            # Clamp to 0-1 range
            vector_scores = [max(0.0, min(1.0, s)) for s in vector_scores]
            keyword_scores = [max(0.0, min(1.0, s)) for s in keyword_scores]

            # Use weight manager to combine
            weighted_chunks, weighted_metadata, weighted_scores = weight_manager.combine_results(
                vector_chunks=vector_chunks,
                vector_metadata=vector_metadata,
                vector_scores=vector_scores,
                keyword_chunks=keyword_chunks,
                keyword_metadata=keyword_metadata,
                keyword_scores=keyword_scores,
                k=k,
            )

            combined_chunks.extend(weighted_chunks)
            combined_metadata.extend(weighted_metadata)

        except Exception as e:
            logger.debug(f"Weighted combination failed, falling back to default: {e}")
            use_weights = False

    # Fallback to default combination if weights not used
    if not use_weights or len(combined_chunks) == len(counts_chunks):
        # Prepend counts summary first
        for chunk, meta in zip(counts_chunks, counts_metadata):
            if chunk not in seen_chunks:
                seen_chunks.add(chunk)
                combined_chunks.append(chunk)
                combined_metadata.append(meta or {})

        # Vector results take priority
        for chunk, meta in zip(vector_chunks, vector_metadata):
            if chunk not in seen_chunks:
                seen_chunks.add(chunk)
                combined_chunks.append(chunk)
                meta_copy = dict(meta) if meta else {}
                meta_copy["retrieval_method"] = "vector"
                combined_metadata.append(meta_copy)

        # Keyword results fill remaining slots
        for chunk, meta in zip(keyword_chunks, keyword_metadata):
            if chunk not in seen_chunks and len(combined_chunks) < k * 2:
                seen_chunks.add(chunk)
                combined_chunks.append(chunk)
                meta_copy = dict(meta) if meta else {}
                meta_copy["retrieval_method"] = "keyword"
                combined_metadata.append(meta_copy)

    final_chunks = combined_chunks[:k]
    final_metadata = combined_metadata[:k]

    # Count items: hybrid items are counted as vector (they came from both sources)
    # This way k_count only reflects pure keyword items, not duplicates
    vector_count = sum(
        1 for m in final_metadata if m.get("retrieval_method") in ["vector", "hybrid"]
    )
    keyword_count = sum(1 for m in final_metadata if m.get("retrieval_method") == "keyword")

    if keyword_count > 0:
        strategy = "weighted " if use_weights and get_weight_manager else ""
        logger.info(
            f"Hybrid retrieval ({strategy}combination): {vector_count} vector + {keyword_count} keyword = {len(final_chunks)} total"
        )

    return final_chunks, final_metadata, vector_count, keyword_count


def _replace_children_with_parents(
    chunks: List[str],
    metadata: List[Dict],
    collection: Collection,
    enable_parent_child: bool,
    logger,
) -> Tuple[List[str], List[Dict], int]:
    """Replace child chunks with parent chunks when available.

    Returns updated chunks/metadata and the count of replacements performed.
    """
    parent_replacements = 0
    if not enable_parent_child or not batch_get_parents_for_children:
        return chunks, metadata, parent_replacements

    try:
        chunk_ids = [m.get("chunk_id", "") or m.get("id", "") for m in metadata]
        if chunk_ids:
            parents_map = batch_get_parents_for_children(chunk_ids, collection, logger=get_logger())
            for idx, m in enumerate(metadata):
                child_id = m.get("chunk_id", "") or m.get("id", "")
                if child_id and child_id in parents_map:
                    parent_data = parents_map[child_id]
                    chunks[idx] = parent_data["text"]
                    m["used_parent"] = True
                    m["parent_id"] = parent_data["id"]
                    m["original_child_id"] = child_id
                    parent_replacements += 1
            if parent_replacements > 0:
                logger.info(
                    f"Replaced {parent_replacements} child chunks with parent chunks for richer context"
                )
    except Exception as e:
        logger.debug(f"Parent chunk retrieval skipped: {e}")

    return chunks, metadata, parent_replacements


def _apply_learned_reranking(
    chunks: List[str],
    metadata: List[Dict],
    query: str,
    k: int,
    model_name: Optional[str],
    top_k: int,
    device: str,
    batch_size: Optional[int] = None,
    enable_cache: bool = True,
    strict_offline: bool = False,
    logger=None,
) -> Tuple[List[str], List[Dict]]:
    """Apply learned reranking when available.

    This helper encapsulates the reranking flow used in both retrieval functions.
    """
    if not (RerankerConfig and rerank_results and chunks):
        return chunks, metadata

    try:
        reranker_config = RerankerConfig(
            enable_reranking=True,
            model_name=model_name,
            rerank_top_k=min(top_k, len(chunks)),
            final_top_k=k,
            device=device,
            batch_size=batch_size if batch_size is not None else 16,
            enable_cache=enable_cache,
            strict_offline=strict_offline,
        )

        docs_for_reranking = [
            {
                "doc_id": m.get("id") or m.get("chunk_id"),
                "text": chunk,
                "hybrid_score": 1.0 - m.get("distance", 0.0),
            }
            for chunk, m in zip(chunks, metadata)
        ]

        reranked = rerank_results(query, docs_for_reranking, reranker_config)
        if reranked:
            reranked_ids = {r.doc_id: idx for idx, r in enumerate(reranked)}
            pairs = [
                (chunk, meta)
                for chunk, meta in zip(chunks, metadata)
                if (meta.get("id") or meta.get("chunk_id")) in reranked_ids
            ]
            pairs.sort(
                key=lambda x: reranked_ids.get(
                    x[1].get("id") or x[1].get("chunk_id"), len(reranked)
                )
            )
            if pairs:
                chunks, metadata = zip(*pairs)
                chunks, metadata = list(chunks), list(metadata)
            if logger:
                logger.info("Applied learned reranking: results reordered by relevance")
            audit(
                "retrieve_reranked", {"reranker_model": model_name, "reranked_count": len(reranked)}
            )
    except Exception as e:
        if logger:
            logger.debug(f"Learned reranking skipped: {e}")

    return chunks, metadata


@retry_ollama_call(max_retries=3, initial_delay=1.0, operation_name="embed_query")
def _embed_query(query: str, model_name: str) -> List[float]:
    """Embed a query string using Ollama with retry and rate limit."""
    from langchain_ollama import OllamaEmbeddings

    limiter = get_rate_limiter()
    if limiter:
        limiter.acquire()

    embed_model = OllamaEmbeddings(model=model_name)
    try:
        return embed_model.embed_query(query)
    except AttributeError:
        if hasattr(embed_model, "embed_documents"):
            return embed_model.embed_documents([query])[0]
        raise


@retry_chromadb_call(max_retries=3, initial_delay=0.5, operation_name="vector_similarity_search")
def _query_collection(
    collection: Collection,
    query_embedding: List[float],
    k: int,
    model_name: str,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute ChromaDB similarity search with retry.

    Args:
        collection: ChromaDB collection
        query_embedding: Query embedding vector
        k: Number of results
        model_name: Embedding model name
        filters: Optional metadata filters (merged with model filter)
    """
    # Build where clause with model filter and any additional filters
    where_clause: Dict[str, Any] = {"embedding_model": model_name}
    if filters:
        if any(str(key).startswith("$") for key in filters.keys()):
            where_clause = {"$and": [{"embedding_model": model_name}, filters]}
        else:
            conditions = [{"embedding_model": model_name}]
            for key, value in filters.items():
                conditions.append({key: value})
            where_clause = {"$and": conditions} if len(conditions) > 1 else conditions[0]

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_clause,
        include=["documents", "metadatas", "distances"],
    )


def _bm25_search_with_fallback(
    query: str, collection: Collection, k: int, filters: Optional[Dict[str, Any]] = None
) -> Tuple[List[str], List[Dict], List[str]]:
    """Perform BM25 keyword search using pre-built index, with fallback to on-the-fly search.

    Attempts to use BM25Retriever with pre-built index from ingestion.
    Falls back to simple term frequency search if index is not available.

    Args:
        query: Search query
        collection: ChromaDB collection
        k: Number of results to return
        filters: Optional metadata filters (e.g., {"language": "java", "source_category": "code"})

    Returns:
        Tuple of (chunks, metadata, chunk_ids) sorted by relevance
    """
    logger = get_logger()

    # Try using BM25Retriever with pre-built index first
    if BM25Retriever:
        retriever = None
        try:
            from .rag_config import RAGConfig

            config = RAGConfig()

            # Initialise BM25Retriever
            retriever = BM25Retriever(rag_data_path=Path(config.rag_data_path))

            # Check if index has documents
            if retriever.total_docs > 0:
                # Search using pre-built index
                results = retriever.search(query, top_k=k)

                if results:
                    # Fetch chunks and metadata from collection using doc_ids
                    chunks = []
                    metadatas = []
                    chunk_ids = []

                    for doc_id, score in results:
                        try:
                            # Get chunk from collection
                            chunk_data = collection.get(
                                ids=[doc_id], include=["documents", "metadatas"]
                            )

                            if chunk_data["ids"] and chunk_data["documents"]:
                                chunks.append(chunk_data["documents"][0])
                                meta = chunk_data["metadatas"][0] if chunk_data["metadatas"] else {}
                                meta["bm25_score"] = score
                                # Add synthetic distance for explainability (BM25 scores are positive, normalise to 0-1 range as distance)
                                # Higher BM25 score = lower distance (better match)
                                # Assume typical BM25 scores range 0-10, map to distance 1.0-0.0
                                meta["distance"] = max(0.0, min(1.0, 1.0 - (score / 10.0)))
                                metadatas.append(meta)
                                chunk_ids.append(doc_id)
                        except Exception as e:
                            logger.debug(f"Failed to fetch chunk {doc_id}: {e}")
                            continue

                    if chunks:
                        logger.info(f"BM25 retrieval (pre-built index) found {len(chunks)} matches")
                        audit(
                            "bm25_retrieval_used",
                            {
                                "query": query[:100],
                                "matches_found": len(chunks),
                                "source": "pre_built_index",
                                "corpus_size": retriever.total_docs,
                            },
                        )
                        # Cleanup and return
                        if retriever:
                            retriever.close()
                        return chunks, metadatas, chunk_ids
        except Exception as e:
            logger.debug(f"BM25Retriever failed, falling back to simple keyword search: {e}")
        finally:
            # Ensure cleanup
            if retriever:
                retriever.close()

    # Fallback: Simple keyword search (on-the-fly term frequency)
    return _keyword_search_fallback(query, collection, k, filters)


def _keyword_search_fallback(
    query: str, collection: Collection, k: int, filters: Optional[Dict[str, Any]] = None
) -> Tuple[List[str], List[Dict], List[str]]:
    """Fallback keyword search using simple term frequency (on-the-fly indexing).

    Retrieves all child chunks and scores them using simple term frequency.
    This is used when BM25 pre-built index is not available.

    Args:
        query: Search query
        collection: ChromaDB collection
        k: Number of results to return
        filters: Optional metadata filters (e.g., {"language": "java", "source_category": "code"})

    Returns:
        Tuple of (chunks, metadata, chunk_ids) sorted by relevance
    """
    logger = get_logger()

    # Extract keywords from query (simple tokenisation)
    keywords = set(query.lower().split())
    keywords = {w for w in keywords if len(w) > 2}  # Filter short words

    if not keywords:
        return [], [], []

    # Build where clause with filters
    # Keyword search should target child chunks to avoid parent summaries.
    where_clause = {"chunk_type": "child"}
    if filters:
        where_clause.update(filters)

    # Get all child chunks (they're the searchable ones)
    # We limit to a reasonable batch size to avoid memory issues
    try:
        all_chunks = collection.get(
            where=where_clause,  # Always include chunk_type filter
            limit=10000,  # Reasonable limit for keyword search
            include=["documents", "metadatas"],
        )
        logger.debug(
            f"Keyword fallback: fetched {len(all_chunks.get('ids', []))} chunks (where={where_clause})"
        )
    except Exception as e:
        logger.debug(f"Keyword search failed to fetch chunks: {e}")
        return [], [], []

    if not all_chunks["ids"]:
        return [], [], []

    # Score each chunk by keyword matches
    scored_results = []
    for i, (chunk_id, doc, meta) in enumerate(
        zip(all_chunks["ids"], all_chunks["documents"], all_chunks["metadatas"])
    ):
        if not doc:
            continue

        doc_lower = doc.lower()
        # Count keyword matches (simple TF scoring)
        score = sum(doc_lower.count(keyword) for keyword in keywords)

        if score > 0:
            scored_results.append((score, chunk_id, doc, meta))

    # Sort by score descending and take top k
    scored_results.sort(key=lambda x: x[0], reverse=True)
    top_results = scored_results[:k]

    if top_results:
        scores, chunk_ids, chunks, metadatas_raw = zip(*top_results)
        # Add distance to metadata for explainability
        # Normalise scores to distance (higher TF score = lower distance)
        max_score = max(scores) if scores else 1.0
        metadatas = []
        for score, meta in zip(scores, metadatas_raw):
            meta_copy = dict(meta) if meta else {}
            meta_copy["tf_score"] = score
            # Map score to distance: highest score gets distance 0.3, lowest gets 0.9
            meta_copy["distance"] = 0.9 - (0.6 * (score / max_score))
            metadatas.append(meta_copy)

        logger.info(
            f"Keyword search (on-the-fly TF) found {len(top_results)} matches (top score: {scores[0]})"
        )
        audit(
            "keyword_search_fallback_used",
            {"query": query[:100], "matches_found": len(top_results), "source": "on_the_fly_tf"},
        )
        return list(chunks), metadatas, list(chunk_ids)

    return [], [], []


# Backward compatibility alias for existing code/tests
_keyword_search = _bm25_search_with_fallback


def _run_counts_branch(
    query: str,
    k: int,
    logger,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Detect count/list intent and return synthetic corpus summary chunks."""
    lower_q = query.lower()
    is_count_intent = any(
        p in lower_q for p in ["how many", "count", "number of", "total", "totals"]
    )
    is_list_intent = any(p in lower_q for p in ["list all", "show all", "give me all", "enumerate"])

    is_total_corpus_query = any(
        p in lower_q
        for p in [
            "make up the corpus",
            "in the corpus",
            "corpus size",
            "total documents",
            "how many documents",
            "size of corpus",
        ]
    ) and not any(p in lower_q for p in ["contain", "reference", "mention", "use", "about"])

    if not (is_count_intent or is_list_intent):
        return [], []

    summary = None

    if is_total_corpus_query:
        try:
            from .counts_service import CountsService

            with CountsService() as svc:
                total = svc.total_documents()
                if total > 0:
                    summary = {
                        "term": "ALL_DOCUMENTS",
                        "total_docs": total,
                        "total_occurrences": 0,
                        "category_breakdown": [],
                        "sample_docs": [],
                    }
        except Exception:
            summary = None
    else:
        try:
            from scripts.search.bm25_search import BM25Search

            tok = BM25Search().tokenise
            tokens = [t for t in tok(query) if len(t) > 2]
        except Exception:
            tokens = [t for t in query.split() if len(t) > 2]

        question_words = {
            "how",
            "many",
            "what",
            "which",
            "where",
            "when",
            "who",
            "why",
            "are",
            "there",
            "list",
            "show",
            "give",
            "count",
            "number",
            "total",
            "all",
            "the",
            "contain",
            "contains",
            "reference",
            "references",
            "document",
            "documents",
            "file",
            "files",
            "corpus",
            "make",
            "size",
            "appear",
            "appears",
            "mentioned",
            "mentions",
            "times",
            "does",
            "did",
            "has",
            "have",
        }
        subject_tokens = [t for t in tokens if t.lower() not in question_words]

        # Prefer longer tokens (likely more specific terms) over generic words
        # Sort by length descending, then take the first (longest)
        if subject_tokens:
            subject_tokens.sort(key=len, reverse=True)
            term = subject_tokens[0]
        else:
            term = tokens[-1] if tokens else None
        if term:
            try:
                from .counts_service import CountsService

                with CountsService() as svc:
                    summary = svc.summarise_term(term, limit=min(10, k))
            except Exception:
                summary = None

    if not summary:
        return [], []

    if summary["term"] == "ALL_DOCUMENTS":
        synth_chunk = (
            "Corpus statistics:\n"
            f"• Total documents in corpus: {summary['total_docs']}\n"
            "• This represents the complete document collection indexed in the system."
        )
        log_term = "TOTAL_CORPUS"
    else:
        breakdown_str = ", ".join(
            [f"{cat}: {cnt}" for cat, cnt in summary.get("category_breakdown", [])]
        )
        synth_chunk = (
            f"Corpus summary for '{summary['term']}':\n"
            f"• The term appears in {summary['total_docs']} different documents\n"
            f"• The term is mentioned {summary['total_occurrences']} times total across all documents\n"
            f"• Top categories: {breakdown_str or 'n/a'}\n"
            f"• Sample docs: {', '.join(summary.get('sample_docs', [])) or 'n/a'}"
        )
        log_term = summary["term"]

    counts_chunks = [synth_chunk]
    counts_metadata = [
        {
            "retrieval_method": "counts",
            "counts_term": summary["term"],
            "counts_total_docs": summary["total_docs"],
            "counts_total_occurrences": summary["total_occurrences"],
            "counts_category_breakdown": summary.get("category_breakdown", []),
        }
    ]

    logger.info(
        f"Agentic SQL counts branch activated (term='{log_term}') → docs={summary['total_docs']}"
    )
    audit(
        "counts_branch_used",
        {
            "query": query[:100],
            "term": log_term,
            "total_docs": summary["total_docs"],
            "total_occurrences": summary["total_occurrences"],
        },
    )

    return counts_chunks, counts_metadata


def _run_vector_search(
    query: str,
    collection: Collection,
    k: int,
    code_filters: Dict[str, Any],
    embedding_model_name: str,
    logger,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Execute vector similarity search and return chunks/metadata."""
    try:
        query_embedding = _embed_query(query, embedding_model_name)
    except Exception as embed_err:
        logger.warning(f"Embedding generation failed: {embed_err}")
        return [], []

    # GUARD: Detect embedding dimension mismatch vs. stored collection
    try:
        collection_dim = _get_collection_embedding_dim(collection)
        if collection_dim:
            query_dim = len(query_embedding)
            if collection_dim != query_dim:
                raise EmbeddingDimensionMismatch(
                    expected_dim=collection_dim,
                    actual_dim=query_dim,
                    model_name=embedding_model_name,
                )
    except EmbeddingDimensionMismatch:
        raise
    except Exception as dim_err:
        logger.debug(f"Embedding dimension check skipped: {dim_err}")

    try:
        results = _query_collection(
            collection, query_embedding, k, embedding_model_name, filters=code_filters
        )
    except (TypeError, ValueError, EmbeddingDimensionMismatch) as e:
        logger.warning(f"Vector search failed, will rely on keyword search: {e}")
        return [], []
    except Exception:
        return [], []

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    distances = results.get("distances") or []

    vector_chunks = documents[0] if documents else []
    vector_metadata = metadatas[0] if metadatas else []
    vector_distances = distances[0] if distances else []

    # Merge distances into metadata for explainability
    for i, meta in enumerate(vector_metadata):
        if i < len(vector_distances):
            meta["distance"] = vector_distances[i]

    if vector_chunks:
        filter_info = f" (filters: {code_filters})" if code_filters else ""
        logger.info(f"Vector search retrieved {len(vector_chunks)} chunks{filter_info}")

    return vector_chunks, vector_metadata


def _run_keyword_search(
    query: str,
    collection: Collection,
    k: int,
    code_filters: Dict[str, Any],
    use_hybrid: bool,
    has_vector_results: bool,
    logger,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Execute keyword/BM25 search when hybrid is enabled or vector search is empty."""
    if not use_hybrid and has_vector_results:
        return [], []

    keyword_chunks, keyword_metadata, _ = _bm25_search_with_fallback(
        query, collection, k, filters=code_filters
    )

    if keyword_chunks:
        filter_info = f" (filters: {code_filters})" if code_filters else ""
        logger.info(
            f"✓ Keyword search activated: found {len(keyword_chunks)} term matches{filter_info}"
        )
        audit(
            "keyword_search_used",
            {
                "query": query[:100],
                "matches_found": len(keyword_chunks),
                "reason": "hybrid_search" if has_vector_results else "vector_search_failed",
                "filters_applied": bool(code_filters),
            },
        )

    return keyword_chunks, keyword_metadata


def retrieve(
    query: str,
    collection: Collection,
    k: int = 5,
    language_filter: Optional[str] = None,
    source_category_filter: Optional[str] = None,
    persona: Optional[str] = None,
    domain: Optional[str] = None,
    enable_code_detection: bool = True,
    allow_code_category: bool = True,
) -> Tuple[List[str], List[Dict]]:
    """Retrieve semantically similar chunks for a query using hybrid search.

    Simplified interface for backward compatibility. Delegates to retrieve_with_filters()
    with hybrid search enabled by default.

    Combines vector similarity search with keyword matching for better retrieval.
    Vector search provides semantic understanding while keyword search catches
    exact term matches that embeddings might miss.

    Supports explicit code-specific filtering via language_filter and
    source_category_filter parameters. Also respects dashboard context signals
    that control whether automatic code detection should be applied.

    Args:
        query: User question or query text.
        collection: ChromaDB collection to search.
        k: Number of results to retrieve (default: 5).
        language_filter: Filter to specific programming language (e.g., "java", "groovy").
        source_category_filter: Filter to specific source category (e.g., "code", "governance_doc").
        persona: Optional persona name ("supervisor", "assessor", "researcher") to apply
            persona-aware filtering and reranking when metadata is available.
        domain: Optional domain for domain-specific term expansion (e.g., 'aboriginal_torres_strait_islander').
        enable_code_detection: Whether to auto-detect code queries from query text.
            If False, code category/language will not be auto-suggested (dashboard signal).
        allow_code_category: Whether code results are allowed in returned chunks.
            If False, code chunks will be filtered out (dashboard signal).

    Returns:
        Tuple of (chunks, metadata):
            - chunks: List of retrieved text chunks.
            - metadata: List of metadata dictionaries for each chunk.

    Raises:
        ValueError: If query is empty or k is invalid.
        Exception: If ChromaDB query fails.

    Example:
        >>> chunks, meta = retrieve("What is MFA?", collection, k=5)
        >>> len(chunks)
        5

        >>> # Code-specific retrieval
        >>> chunks, meta = retrieve("Show Java services", collection, language_filter="java")

        >>> # Academic persona with code detection disabled
        >>> chunks, meta = retrieve("Rate the methodology", collection, enable_code_detection=False)
    """
    # Get config for defaults
    try:
        from .rag_config import RAGConfig

        config = RAGConfig()
        enable_parent_child = getattr(config, "enable_parent_child", False)
        enable_learned_reranking = getattr(config, "enable_learned_reranking", False)
        reranker_model = getattr(config, "reranker_model", "BAAI/bge-reranker-base")
    except Exception:
        enable_parent_child = False
        enable_learned_reranking = False
        reranker_model = "BAAI/bge-reranker-base"

    # Build filters dict from language and category filters
    filters = {}
    if language_filter:
        filters["language"] = language_filter.lower()
    if source_category_filter:
        filters["source_category"] = source_category_filter.lower()

    # Delegate to comprehensive retrieve_with_filters
    return retrieve_with_filters(
        query=query,
        collection=collection,
        k=k,
        filters=filters if filters else None,
        language_filter=None,  # Already in filters dict
        source_category_filter=None,  # Already in filters dict
        persona=persona,
        domain=domain,
        auto_detect_filters=enable_code_detection,  # Pass dashboard signal
        enable_code_detection=enable_code_detection,  # Additional context signal
        allow_code_category=allow_code_category,  # Additional context signal
        enable_hybrid_search=True,  # retrieve() always uses hybrid search
        enable_reranking=False,  # Lightweight reranking disabled by default
        enable_learned_reranking=enable_learned_reranking,
        reranker_model=reranker_model,
        fetch_neighbours=False,  # Don't fetch neighbours by default
        enable_caching=False,  # Caching disabled for simple retrieve()
        enable_graph=False,  # Graph expansion disabled for simple retrieve()
        enable_parent_child=enable_parent_child,
        cache_dir=None,
    )


def detect_filters_from_query(
    query: str,
    enable_code_detection: bool = True,
    allow_code_category: bool = True,
    persona: Optional[str] = None,
) -> Dict[str, Any]:
    """Auto-detect metadata filters from natural language query.

    Analyses query text for keywords that suggest specific content types,
    categories, technical domains, and programming languages. Constructs
    ChromaDB filter conditions to narrow search space.

    Respects dashboard context signals:
    - If enable_code_detection is False (Code-Aware Context disabled), skips
      automatic code query detection
    - If allow_code_category is False (Code unchecked in Result Type), does not
      suggest code category filters
    - If persona is academic mode (supervisor/assessor/researcher), disables
      code detection as academic queries should not be misclassified as code

    Supports code-specific queries like:
    - "Show me Java services" → filters to Java code
    - "Find payment APIs" → filters to code with endpoints
    - "Which services use Okta?" → filters to code with dependencies

    Args:
        query: Natural language query text
        enable_code_detection: Whether to allow automatic code detection.
            Dashboard signal: False when Code-Aware Context is disabled.
        allow_code_category: Whether code category is allowed in results.
            Dashboard signal: False when "Code" is unchecked in Result Type filter.
        persona: Optional persona name ("supervisor", "assessor", "researcher").
            Academic personas should not enable code detection.

    Returns:
        Dictionary of ChromaDB where conditions

    Example:
        >>> detect_filters_from_query("Show me API security policies")
        {'source_category': 'governance', 'is_api_reference': True}

        >>> detect_filters_from_query("Find Java microservices")
        {'source_category': 'code', 'language': 'java'}

        >>> # Academic persona disables code detection
        >>> detect_filters_from_query(
        ...     "Rate the methodology",
        ...     enable_code_detection=False,
        ...     persona="assessor"
        ... )
        {}  # No code filters applied
    """
    filters = {}
    query_lower = query.lower()

    # Academic corpus indicators (thesis, papers, dissertations)
    academic_terms = [
        "thesis",
        "phd",
        "dissertation",
        "methodology",
        "methodologies",
        "method",
        "methods",
        "literature review",
        "research question",
        "research questions",
        "findings",
        "discussion",
        "conclusion",
        "chapter",
        "abstract",
        "supervisor",
        "assessor",
        "academic",
        "citation",
        "citations",
        "reference",
        "references",
        "bibliography",
        "document",
        "feedback",
        "quality",
        "analysis",
    ]

    # Only match explicit programming/API patterns
    explicit_code_terms = [
        "microservice",
        "/api/",
        "rest api",
        "soap api",
        "graphql",
        "endpoint",
        "controller class",
        "java class",
        "python class",
        "function definition",
        "code snippet",
        "implementation code",
        "service class",
    ]

    academic_like = any(term in query_lower for term in academic_terms)
    code_like = any(term in query_lower for term in explicit_code_terms)

    # ==========================================
    # CODE-SPECIFIC FILTER DETECTION
    # ==========================================
    # Only apply code detection if:
    # 1. Code detection is enabled by dashboard (enable_code_detection=True)
    # 2. Code category is allowed in results (allow_code_category=True)
    # 3. Not an academic query (unless explicitly matching code terms)
    if enable_code_detection and allow_code_category and (not academic_like or code_like):
        # Language detection - check more specific languages first
        language_keywords = {
            "groovy": ["groovy", "spock"],  # Check groovy before gradle
            "java": ["java", "spring", "maven", "junit", "gradle"],
            "kotlin": ["kotlin", "kotlinx"],
            "python": ["python", "django", "flask", "pytest"],
            "go": ["go", "golang", "goroutine"],
            "rust": ["rust", "cargo", "tokio"],
            "javascript": ["javascript", "nodejs", "npm", "ts", "typescript"],
            "sql": ["sql", "plsql", "tsql", "hive"],
            "xml": ["xml", "xpath", "xsd"],
            "yaml": ["yaml", "yml", "helm"],
            "json": ["json", "jsonpath"],
        }

        for lang, keywords in language_keywords.items():
            if any(kw in query_lower for kw in keywords):
                filters["language"] = lang
                filters["source_category"] = "code"
                break

        # Code-related queries - be specific
        if any(
            term in query_lower
            for term in [
                "source code",
                "code snippet",
                "code example",
                "implementation code",
                "microservice",
                "api endpoint",
            ]
        ):
            if "source_category" not in filters:
                filters["source_category"] = "code"

        # Service detection (implies code) - require technical qualifiers
        if any(
            term in query_lower
            for term in ["microservice", "web service", "rest service", "api service"]
        ):
            filters["source_category"] = "code"

        # Dependency detection - require technical qualifiers
        if any(
            term in query_lower
            for term in [
                "maven dependency",
                "npm package",
                "pip install",
                "gradle",
                "library import",
                "package import",
            ]
        ):
            filters["source_category"] = "code"

        # API/Endpoint detection - require explicit patterns
        if any(
            term in query_lower
            for term in [
                "/api/",
                "rest api",
                "soap endpoint",
                "graphql",
                "http request",
                "http response",
                "api controller",
            ]
        ):
            filters["source_category"] = "code"

        # Internal calls/references detection - require code-specific context
        # (Removed generic "call", "calls", "uses" to avoid academic false positives)
        if any(
            term in query_lower
            for term in ["function call", "method invocation", "api call", "service invocation"]
        ):
            if "source_category" not in filters:
                filters["source_category"] = "code"

        # Fallback: if query matched explicit_code_terms but no specific pattern,
        # still mark as code category
        if code_like and "source_category" not in filters:
            filters["source_category"] = "code"

    # ==========================================
    # GOVERNANCE/DOCUMENTATION FILTERS
    # ==========================================

    # Category detection
    if any(
        term in query_lower
        for term in ["security", "mfa", "authentication", "authorization", "authorisation"]
    ):
        if "source_category" not in filters:
            filters["source_category"] = "governance"

    if any(term in query_lower for term in ["pattern", "architecture", "design"]):
        if "source_category" not in filters:
            filters["source_category"] = "patterns"

    # Technical vs general content (for documentation)
    # Only flag as API reference if explicitly technical, not academic
    if any(
        term in query_lower
        for term in [
            "/api/",
            "rest api",
            "soap api",
            "api endpoint",
            "http request",
            "http response",
        ]
    ):
        if not academic_like or code_like:
            filters["is_api_reference"] = True

    # Configuration queries
    if any(
        term in query_lower
        for term in ["config", "setting", "parameter", "option", "configuration"]
    ):
        if "source_category" not in filters:
            filters["is_configuration"] = True

    # Table/structured data
    if any(term in query_lower for term in ["table", "list", "matrix", "comparison"]):
        filters["contains_table"] = True

    return filters


def build_code_filters(
    language: Optional[str] = None,
    include_dependencies: bool = False,
    include_endpoints: bool = False,
    include_services: bool = False,
) -> Dict[str, Any]:
    """Build explicit code-specific metadata filters.

    Creates ChromaDB where conditions for code document filtering.
    Useful for targeted code searches like "Show me all Java services".

    Args:
        language: Programming language filter (e.g., "java", "groovy")
        include_dependencies: Only return code with external dependencies
        include_endpoints: Only return code with REST/API endpoints
        include_services: Only return code that's a service

    Returns:
        Dictionary of ChromaDB where conditions

    Example:
        >>> filters = build_code_filters(language="java", include_endpoints=True)
        >>> chunks, meta = retrieve_with_filters("auth", collection, filters=filters)
    """
    filters = {"source_category": "code"}

    if language:
        filters["language"] = language.lower()

    if include_dependencies:
        # Could add a has_dependencies boolean field when ingesting code
        filters["has_dependencies"] = True

    if include_endpoints:
        # Could add an has_endpoints boolean field when ingesting code
        filters["has_endpoints"] = True

    if include_services:
        # Services typically have certain naming patterns or markers
        filters["is_service"] = True

    return filters


def calculate_rerank_score(chunk: str, metadata: Dict, query: str, distance: float) -> float:
    """Calculate lightweight re-ranking score without LLM.

    Combines semantic similarity (distance) with heuristic signals:
    - Keyword overlap (BM25-style)
    - Metadata match bonuses
    - Section depth preference
    - Content type relevance
    - Length optimisation

    Args:
        chunk: Chunk text
        metadata: Chunk metadata dict
        query: Query text
        distance: Semantic similarity distance (lower = better)

    Returns:
        Combined score (higher = better)
    """
    # Base score from semantic similarity (invert distance)
    # Typical cosine distances are 0.0-2.0, so normalise
    base_score = max(0, 1.0 - (distance / 2.0))

    # Keyword overlap bonus
    query_terms = set(query.lower().split())
    chunk_terms = set(chunk.lower().split())
    overlap_ratio = len(query_terms & chunk_terms) / len(query_terms) if query_terms else 0
    base_score += overlap_ratio * 0.2

    # Metadata bonuses
    if metadata.get("section_depth") == 1:  # Top-level sections are often overviews
        base_score += 0.1

    # Content type match bonuses
    query_lower = query.lower()
    if "api" in query_lower and metadata.get("is_api_reference"):
        base_score += 0.15
    if "config" in query_lower and metadata.get("is_configuration"):
        base_score += 0.15
    if "code" in query_lower and metadata.get("contains_code"):
        base_score += 0.1

    # Length preference (moderate-length chunks are often more useful)
    chunk_tokens = len(chunk.split())
    if 300 <= chunk_tokens <= 600:
        base_score += 0.05
    elif chunk_tokens < 100:
        base_score -= 0.05  # Very short chunks may lack context

    return base_score


def retrieve_with_filters(
    query: str,
    collection: Collection,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    language_filter: Optional[str] = None,
    source_category_filter: Optional[str] = None,
    persona: Optional[str] = None,
    domain: Optional[str] = None,
    auto_detect_filters: bool = True,
    enable_code_detection: bool = True,
    allow_code_category: bool = True,
    enable_hybrid_search: bool = True,
    enable_reranking: bool = False,
    enable_learned_reranking: bool = True,
    reranker_model: str = "BAAI/bge-reranker-base",
    fetch_neighbours: bool = False,
    enable_caching: bool = True,
    enable_graph: bool = True,
    enable_parent_child: bool = True,
    cache_dir: Optional[Path] = None,
) -> Tuple[List[str], List[Dict]]:
    """Enhanced retrieval with hybrid search, filtering, caching, and graph expansion.

    Comprehensive retrieval with all advanced capabilities:
    1. Check context cache for frequently accessed entities
    2. Build filters from explicit params and/or auto-detect from query
    3. Run counts branch for count/list queries
    4. Execute hybrid search (vector + keyword/BM25)
    5. Combine results with intelligent weighting
    6. Optionally expand with graph-connected chunks
    7. Replace child chunks with parent chunks for richer context
    8. Apply learned reranking with cross-encoder models
    9. Apply persona-aware reranking
    10. Optionally fetch neighbouring chunks for context
    11. Cache results for hot entities

    Args:
        query: Natural language query
        collection: ChromaDB collection
        k: Number of final results to return
        filters: Explicit metadata filters (ChromaDB where conditions)
        language_filter: Filter to specific programming language (e.g., "java", "python")
        source_category_filter: Filter to source category (e.g., "code", "governance")
        persona: Optional persona name ("supervisor", "assessor", "researcher")
        domain: Optional domain for domain-specific term expansion (e.g., 'aboriginal_torres_strait_islander')
        auto_detect_filters: Automatically detect filters from query
        enable_code_detection: Whether to allow auto-detection of code queries (dashboard signal).
            If False, code category/language will not be auto-suggested.
        allow_code_category: Whether code results are allowed in returned chunks (dashboard signal).
            If False, code chunks will be filtered out regardless.
        enable_hybrid_search: Use hybrid vector + keyword search
        enable_reranking: Apply lightweight re-ranking
        enable_learned_reranking: Use cross-encoder reranking
        reranker_model: Cross-encoder model name
        fetch_neighbours: Fetch prev/next chunks for expanded context
        enable_caching: Use context cache for hot entities
        enable_graph: Expand with graph-connected chunks
        enable_parent_child: Replace matched children with parent chunks
        cache_dir: Directory for cache file (default: rag_data/)

    Returns:
        Tuple of (chunks, metadata) with enhanced retrieval

    Example:
        >>> chunks, meta = retrieve_with_filters(
        ...     "API authentication config",
        ...     collection,
        ...     k=5,
        ...     language_filter="java",
        ...     enable_graph=True,
        ...     persona="assessor"
        ... )
    """
    logger = get_logger()

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    # Apply domain-specific query expansion if domain provided
    expanded_query = query
    if domain and get_expansion_cache:
        try:
            cache = get_expansion_cache(domain=domain)
            expanded_terms = cache.get_expanded(query)
            if expanded_terms and len(expanded_terms) > len(query.split()):
                expanded_query = " ".join(expanded_terms)
                logger.debug(f"Domain-expanded query ({domain}): {expanded_query[:100]}")
                audit(
                    "domain_query_expansion",
                    {
                        "domain": domain,
                        "original_terms": len(query.split()),
                        "expanded_terms": len(expanded_terms),
                    },
                )
        except Exception as e:
            logger.debug(f"Domain query expansion failed: {e}")

    # Use expanded query for remainder of retrieval
    query = expanded_query

    # Track retrieval start time
    retrieval_start = time.perf_counter()
    cache_hit = False

    # Try cache first if enabled
    if enable_caching:
        from .context_cache import get_context_cache

        # Use rag_data as default cache directory for consistency with other caches
        if cache_dir is None:
            from .rag_config import RAGConfig

            config = RAGConfig()
            cache_dir = Path(config.rag_data_path)

        # Simple entity extraction for cache key
        cache_key = query.lower().strip()[:100]  # Use query as cache key
        cache = get_context_cache(cache_dir=cache_dir, enabled=True)

        cached_context = cache.get(cache_key)
        if cached_context:
            logger.info(f"Cache hit for query: {query[:50]}")
            audit("cache_hit", {"query": query[:100]})
            cache_hit = True

            # Parse cached context back into chunks/metadata
            try:
                cached_data = json.loads(cached_context)
                chunks = cached_data.get("chunks", [])
                metadata = cached_data.get("metadata", [])

                # Record cache hit metrics
                retrieval_time = time.perf_counter() - retrieval_start
                perf_metrics.record_retrieval(
                    latency_ms=retrieval_time * 1000,
                    result_count=len(chunks),
                    cache_hit=True,
                )
                metrics_collector.record_retrieval(
                    latency_ms=retrieval_time * 1000,
                    result_count=len(chunks),
                    cache_hit=True,
                )

                return chunks, metadata
            except (json.JSONDecodeError, AttributeError):
                # Fallback if cache format is wrong
                logger.warning("Invalid cache format, proceeding with retrieval")

    # Initialise graph retriever if enabled
    graph_retriever = None
    if enable_graph:
        try:
            from .graph_retrieval import get_graph_retriever

            graph_retriever = get_graph_retriever()
            if graph_retriever.graph:
                logger.info("Graph enhancement enabled")
        except Exception as e:
            logger.warning(f"Graph retrieval unavailable: {e}")

    # Build combined filters from all sources
    combined_filters = filters.copy() if filters else {}

    # Add explicit language/category filters if provided
    if language_filter:
        combined_filters["language"] = language_filter.lower()
    if source_category_filter:
        combined_filters["source_category"] = source_category_filter.lower()

    # Auto-detect additional filters from query if enabled
    if auto_detect_filters:
        auto_filters = detect_filters_from_query(
            query,
            enable_code_detection=enable_code_detection,
            allow_code_category=allow_code_category,
            persona=persona,
        )
        # Auto-detected filters don't override explicit ones
        for key, value in auto_filters.items():
            if key not in combined_filters:
                combined_filters[key] = value

    # Log filters for debugging
    if combined_filters:
        logger.info(f"Applying filters: {combined_filters}")
        audit("retrieve_filters", {"query": query[:100], "filters": combined_filters})

    # Get embedding model name
    try:
        from scripts.ingest.vectors import EMBEDDING_MODEL_NAME
    except Exception:
        EMBEDDING_MODEL_NAME = "mxbai-embed-large"

    # 0. Agentic SQL counts branch
    counts_chunks, counts_metadata = [], []
    try:
        counts_chunks, counts_metadata = _run_counts_branch(query, k, logger)
        if counts_chunks:
            logger.info(f"Counts branch returned {len(counts_chunks)} synthetic chunk(s)")
    except Exception as e:
        logger.warning(f"Counts branch failed: {e}", exc_info=True)

    # 1. Vector similarity search
    vector_chunks, vector_metadata = _run_vector_search(
        query,
        collection,
        k,
        combined_filters,
        EMBEDDING_MODEL_NAME,
        logger,
    )

    # 2. Keyword/BM25 search (if hybrid enabled or vector search failed)
    keyword_chunks, keyword_metadata = [], []
    if enable_hybrid_search:
        keyword_chunks, keyword_metadata = _run_keyword_search(
            query,
            collection,
            k,
            combined_filters,
            enable_hybrid_search,
            bool(vector_chunks),
            logger,
        )

    # 3. Combine results with deduplication and optional weighting
    try:
        use_hybrid_weights = enable_hybrid_search and get_weight_manager is not None
    except Exception:
        use_hybrid_weights = False

    documents, metadatas, vector_count, keyword_count = _combine_results(
        vector_chunks,
        vector_metadata,
        keyword_chunks,
        keyword_metadata,
        counts_chunks,
        counts_metadata,
        k,
        logger,
        use_weights=use_hybrid_weights,
    )

    if not documents:
        logger.info("Retrieved 0 chunks for query")
        # Record audit and metrics even for empty results
        retrieval_time = time.perf_counter() - retrieval_start
        perf_metrics.record_retrieval(
            latency_ms=retrieval_time * 1000,
            result_count=0,
            cache_hit=cache_hit,
        )
        metrics_collector.record_retrieval(
            latency_ms=retrieval_time * 1000,
            result_count=0,
            cache_hit=cache_hit,
        )
        audit(
            "retrieve",
            {
                "query_length": len(query),
                "filters_applied": len(combined_filters),
                "retrieved_count": 0,
                "requested_count": k,
                "vector_count": vector_count,
                "keyword_count": keyword_count,
                "hybrid_enabled": enable_hybrid_search,
                "retrieval_time_ms": round(retrieval_time * 1000, 2),
                "cache_hit": cache_hit,
            },
        )
        return [], []

    # Graph expansion: Add related chunks from graph connections
    if enable_graph and graph_retriever and graph_retriever.graph:
        # Extract chunk IDs from metadata
        chunk_ids = [meta.get("chunk_id", "") for meta in metadatas if meta.get("chunk_id")]

        if chunk_ids:
            # Expand with 1-hop neighbours
            expanded_ids = graph_retriever.expand_with_neighbours(
                chunk_ids, max_hops=1, max_neighbours=3
            )

            # Fetch additional chunks from graph
            new_ids = [cid for cid in expanded_ids if cid not in chunk_ids]
            if new_ids:
                logger.info(f"Graph expansion added {len(new_ids)} related chunks")

                # Fetch graph-expanded chunks from ChromaDB
                for new_id in new_ids[:k]:  # Limit expansion
                    try:
                        # Query by chunk_id metadata
                        graph_results = collection.get(
                            where={"chunk_id": new_id}, include=["documents", "metadatas"]
                        )
                        if graph_results.get("documents"):
                            documents.append(graph_results["documents"][0])
                            metadatas.append(graph_results["metadatas"][0])
                    except Exception as e:
                        logger.debug(f"Failed to fetch graph chunk {new_id}: {e}")

    # Re-rank if enabled
    if enable_reranking:
        # Calculate distances if not present (graph-expanded chunks get 0.5)
        distances = [meta.get("distance", 0.5) for meta in metadatas]

        scored_results = [
            (
                doc,
                meta,
                calculate_rerank_score(doc, meta, query, dist),
            )
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]
        # Sort by score (higher is better)
        scored_results.sort(key=lambda x: x[2], reverse=True)
        documents = [x[0] for x in scored_results[:k]]
        metadatas = [x[1] for x in scored_results[:k]]

        logger.info(f"Re-ranked {len(scored_results)} candidates to {len(documents)} results")
    else:
        documents = documents[:k]
        metadatas = metadatas[:k]

    # Replace child chunks with parent chunks if enabled
    parent_replacements = 0
    try:
        documents, metadatas, parent_replacements = _replace_children_with_parents(
            documents,
            metadatas,
            collection,
            enable_parent_child,
            logger,
        )
    except Exception as e:
        logger.warning(f"Failed to fetch parent chunks: {e}")

    logger.info(f"Retrieved {len(documents)} chunks for query")

    # Apply learned reranking if enabled and available
    if enable_learned_reranking:
        try:
            try:
                from .rag_config import RAGConfig

                config = RAGConfig()
                reranker_top_k = getattr(config, "rerank_top_k", 50)
                reranker_device = getattr(config, "reranker_device", "cpu")
                reranker_batch_size = getattr(config, "reranker_batch_size", 32)
                reranker_enable_cache = getattr(config, "enable_reranker_cache", True)
                reranker_strict_offline = getattr(config, "reranker_strict_offline", False)
            except Exception:
                reranker_top_k = 50
                reranker_device = "cpu"
                reranker_batch_size = 32
                reranker_enable_cache = True
                reranker_strict_offline = False

            documents, metadatas = _apply_learned_reranking(
                documents,
                metadatas,
                query,
                k,
                reranker_model,
                top_k=reranker_top_k,
                device=reranker_device,
                batch_size=reranker_batch_size,
                enable_cache=reranker_enable_cache,
                strict_offline=reranker_strict_offline,
                logger=logger,
            )
        except Exception as e:
            logger.warning(f"Learned reranking failed: {e}")

    # Apply persona-aware reranking if requested
    if persona and apply_persona_reranking:
        try:
            documents, metadatas = apply_persona_reranking(
                documents,
                metadatas,
                persona,
                k,
            )
        except Exception as persona_err:
            logger.debug(f"Persona reranking skipped: {persona_err}")

    # Cache results if enabled and this appears to be a hot query
    if enable_caching and not cache_hit:
        try:
            cache_data = {
                "chunks": documents,
                "metadata": metadatas,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            chunk_ids = [m.get("chunk_id", "") for m in metadatas if m.get("chunk_id")]
            cache.put(
                entity=cache_key,
                context=json.dumps(cache_data),
                chunk_ids=chunk_ids,
                ttl=3600,  # 1 hour
                metadata={"query": query[:100], "result_count": len(documents)},
            )
            logger.debug(f"Cached results for query: {query[:50]}")
        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")

    # Fetch neighbouring chunks for context if requested
    if fetch_neighbours:
        expanded_chunks, expanded_meta = fetch_chunk_neighbours(documents, metadatas, collection)

        # Record metrics for enhanced retrieval
        retrieval_time = time.perf_counter() - retrieval_start
        perf_metrics.record_retrieval(
            latency_ms=retrieval_time * 1000,
            result_count=len(expanded_chunks),
            cache_hit=cache_hit,
        )
        metrics_collector.record_retrieval(
            latency_ms=retrieval_time * 1000,
            result_count=len(expanded_chunks),
            cache_hit=cache_hit,
        )

        audit(
            "retrieve",
            {
                "query_length": len(query),
                "filters_applied": len(combined_filters),
                "retrieved_count": len(expanded_chunks),
                "vector_count": vector_count,
                "keyword_count": keyword_count,
                "hybrid_enabled": enable_hybrid_search,
                "reranked": enable_reranking,
                "parent_replacements": parent_replacements,
                "persona": persona,
                "retrieval_time_ms": round(retrieval_time * 1000, 2),
                "cache_hit": cache_hit,
            },
        )

        return expanded_chunks, expanded_meta

    # Record metrics for standard enhanced retrieval
    retrieval_time = time.perf_counter() - retrieval_start
    perf_metrics.record_retrieval(
        latency_ms=retrieval_time * 1000,
        result_count=len(documents),
        cache_hit=cache_hit,
    )
    metrics_collector.record_retrieval(
        latency_ms=retrieval_time * 1000,
        result_count=len(documents),
        cache_hit=cache_hit,
    )

    audit(
        "retrieve",
        {
            "query_length": len(query),
            "filters_applied": len(combined_filters),
            "retrieved_count": len(documents),
            "requested_count": k,
            "vector_count": vector_count,
            "keyword_count": keyword_count,
            "hybrid_enabled": enable_hybrid_search,
            "reranked": enable_reranking,
            "parent_replacements": parent_replacements,
            "persona": persona,
            "retrieval_time_ms": round(retrieval_time * 1000, 2),
            "cache_hit": cache_hit,
        },
    )

    return documents, metadatas


def fetch_chunk_neighbours(
    chunks: List[str], metadatas: List[Dict], collection: Collection
) -> Tuple[List[str], List[Dict]]:
    """Fetch neighbouring chunks to provide expanded context.

    For each retrieved chunk, optionally fetch its prev/next chunks
    based on the chunk metadata relationships.

    Args:
        chunks: Retrieved chunks
        metadatas: Chunk metadata with prev_chunk_id, next_chunk_id
        collection: ChromaDB collection

    Returns:
        Expanded chunks and metadata including neighbours
    """
    logger = get_logger()
    expanded_chunks = []
    expanded_meta = []

    neighbour_ids: Set[str] = set()
    for meta in metadatas:
        if meta.get("prev_chunk_id"):
            neighbour_ids.add(meta["prev_chunk_id"])
        if meta.get("next_chunk_id"):
            neighbour_ids.add(meta["next_chunk_id"])

    # Fetch neighbours in batch
    neighbour_data = {}
    if neighbour_ids:
        try:
            results = collection.get(ids=list(neighbour_ids), include=["documents", "metadatas"])
            for idx, chunk_id in enumerate(results.get("ids", [])):
                neighbour_data[chunk_id] = {
                    "text": results["documents"][idx],
                    "metadata": results["metadatas"][idx],
                }
        except Exception as e:
            logger.warning(f"Failed to fetch neighbours: {e}")

    # Reconstruct with neighbours
    for chunk, meta in zip(chunks, metadatas):
        # Add previous chunk if available
        prev_id = meta.get("prev_chunk_id")
        if prev_id and prev_id in neighbour_data:
            expanded_chunks.append(neighbour_data[prev_id]["text"])
            expanded_meta.append({**neighbour_data[prev_id]["metadata"], "is_neighbour": True})

        # Add main chunk
        expanded_chunks.append(chunk)
        expanded_meta.append(meta)

        # Add next chunk if available
        next_id = meta.get("next_chunk_id")
        if next_id and next_id in neighbour_data:
            expanded_chunks.append(neighbour_data[next_id]["text"])
            expanded_meta.append({**neighbour_data[next_id]["metadata"], "is_neighbour": True})

    logger.info(f"Expanded {len(chunks)} chunks to {len(expanded_chunks)} with neighbours")
    return expanded_chunks, expanded_meta


def explain_retrieval(
    query: str,
    chunks: List[str],
    metadatas: List[Dict],
    k: int,
) -> Dict[str, Any]:
    """Generate explainability data for retrieval results.

    Provides transparency into why specific chunks were retrieved,
    including similarity scores, ranking factors, and metadata insights.

    Args:
        query: The user query
        chunks: Retrieved chunks
        metadatas: Chunk metadata with distance/scores
        k: Number of results requested

    Returns:
        Dict with explainability data:
            - retrieval_method: How chunks were retrieved
            - ranking_explanation: Why chunks are ordered this way
            - similarity_scores: List of similarity values
            - confidence_level: Overall confidence (high/medium/low)
            - metadata_insights: Key metadata patterns
    """
    logger = get_logger()

    # Extract similarity scores (lower distance = higher similarity)
    similarities = []
    for meta in metadatas:
        safe_meta = meta if isinstance(meta, dict) else {}
        if safe_meta.get("distance") is not None:
            # Convert distance to similarity (0-1 scale)
            similarity = 1.0 - min(safe_meta["distance"], 1.0)
            similarities.append(round(similarity, 3))
        else:
            similarities.append(None)

    # Determine retrieval methods used
    retrieval_methods = set()
    for meta in metadatas:
        safe_meta = meta if isinstance(meta, dict) else {}
        method = safe_meta.get("retrieval_method", "vector")
        retrieval_methods.add(method)

    # Calculate confidence level
    valid_sims = [s for s in similarities if s is not None]
    if valid_sims:
        avg_similarity = sum(valid_sims) / len(valid_sims)
        if avg_similarity >= 0.7:
            confidence = "high"
        elif avg_similarity >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"
    else:
        confidence = "unknown"

    # Extract metadata insights
    source_categories = set()
    languages = set()
    services = set()

    for meta in metadatas:
        safe_meta = meta if isinstance(meta, dict) else {}
        if safe_meta.get("source_category"):
            source_categories.add(safe_meta["source_category"])
        if safe_meta.get("language"):
            languages.add(safe_meta["language"])
        if safe_meta.get("service_name"):
            services.add(safe_meta["service_name"])

    # Build ranking explanation
    ranking_parts = []
    if "vector" in retrieval_methods:
        ranking_parts.append("semantic similarity")
    if "keyword" in retrieval_methods:
        ranking_parts.append("keyword matching")
    if "graph" in retrieval_methods:
        ranking_parts.append("graph relationships")

    ranking_explanation = (
        f"Ranked by {' + '.join(ranking_parts)}" if ranking_parts else "Ranked by relevance"
    )

    if valid_sims:
        top_score = max(valid_sims)
        ranking_explanation += f". Top match has {top_score:.1%} similarity."

    return {
        "retrieval_method": list(retrieval_methods),
        "ranking_explanation": ranking_explanation,
        "similarity_scores": similarities,
        "confidence_level": confidence,
        "avg_similarity": round(sum(valid_sims) / len(valid_sims), 3) if valid_sims else 0.0,
        "metadata_insights": {
            "source_categories": list(source_categories),
            "languages": list(languages),
            "services": list(services),
            "total_chunks": len(chunks),
            "k_requested": k,
        },
    }
