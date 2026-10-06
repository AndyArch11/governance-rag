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
    from chromadb.api.models.Collection import Collection as ChromaDBCollection  # noqa: WPS433,E402
except Exception:
    ChromaDBCollection = Any  # type: ignore

try:
    from scripts.ingest.chromadb_sqlite import ChromaSQLiteCollection  # noqa: WPS433,E402
except Exception:
    ChromaSQLiteCollection = Any  # type: ignore

Collection = Union[ChromaDBCollection, ChromaSQLiteCollection, Any]


def _collection_get(collection: Collection, **kwargs: Any) -> Dict[str, Any]:
    """Read records through the common ChromaDB and SQLite collection contract."""
    collection_backend: Any = collection
    return collection_backend.get(**kwargs)


def _collection_query(collection: Collection, **kwargs: Any) -> Dict[str, Any]:
    """Query records through the common ChromaDB and SQLite collection contract."""
    collection_backend: Any = collection
    return collection_backend.query(**kwargs)


from scripts.utils.logger import create_module_logger

from .context_cache import get_context_cache

get_logger, audit = create_module_logger("rag")
from scripts.utils.llm_instrumentation import record_embedding_usage
from scripts.utils.metrics_export import get_metrics_collector
from scripts.utils.monitoring import get_perf_metrics, init_monitoring
from scripts.utils.rate_limiter import get_rate_limiter
from scripts.utils.retry_utils import retry_chromadb_call, retry_ollama_call

batch_get_parents_for_children: Any | None = None
try:
    from scripts.ingest.vectors import batch_get_parents_for_children as _batch_get_parents
except ImportError:
    pass
else:
    batch_get_parents_for_children = _batch_get_parents

BM25Retriever: Any | None = None
try:
    from scripts.search.bm25_retrieval import BM25Retriever as _BM25Retriever
except ImportError:
    pass
else:
    BM25Retriever = _BM25Retriever

RankerResult: Any | None = None
RerankerConfig: Any | None = None
rerank_results: Any | None = None
try:
    from scripts.search.reranker import RankerResult as _RankerResult
    from scripts.search.reranker import RerankerConfig as _RerankerConfig
    from scripts.search.reranker import rerank_results as _rerank_results
except ImportError:
    pass
else:
    RankerResult = _RankerResult
    RerankerConfig = _RerankerConfig
    rerank_results = _rerank_results

HybridSearchWeights: Any | None = None
get_weight_manager: Any | None = None
try:
    from scripts.rag.hybrid_search_weights import HybridSearchWeights as _HybridSearchWeights
    from scripts.rag.hybrid_search_weights import get_weight_manager as _get_weight_manager
except ImportError:
    pass
else:
    HybridSearchWeights = _HybridSearchWeights
    get_weight_manager = _get_weight_manager

get_query_expander: Any | None = None
get_expansion_cache: Any | None = None
try:
    from scripts.rag.query_expansion import get_expansion_cache as _get_expansion_cache
    from scripts.rag.query_expansion import get_query_expander as _get_query_expander
except ImportError:
    pass
else:
    get_query_expander = _get_query_expander
    get_expansion_cache = _get_expansion_cache

apply_persona_reranking: Any | None = None
try:
    from scripts.search.persona_retrieval import apply_persona_reranking as _apply_persona_reranking
except ImportError:
    pass
else:
    apply_persona_reranking = _apply_persona_reranking

from scripts.utils.embedding_model_config import EMBEDDING_MODEL_NAME

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
        sample = _collection_get(collection, limit=1, include=["embeddings"])
        embeddings = sample.get("embeddings") or []
        if embeddings and len(embeddings) > 0:
            return len(embeddings[0])
    except Exception:
        return None
    return None


# ------------------------------
# Test-friendly helper functions
# ------------------------------


def _expand_thesis_graph_candidates(
    collection: Collection,
    seed_metadata: List[Dict],
    logger: Any,
) -> Tuple[List[str], List[Dict]]:
    """Fetch thesis-graph neighbours for inclusion in hybrid ranking.

    Args:
        collection (Collection): The collection to query for thesis graph neighbours.
        seed_metadata (List[Dict]): Metadata of the seed chunks.
        logger (Any): Logger instance for logging.

    Returns:
        Tuple[List[str], List[Dict]]: A tuple containing a list of expanded document texts and their corresponding metadata.
    """
    thesis_ids = {
        str(meta.get("thesis_id") or meta.get("doc_id"))
        for meta in seed_metadata
        if isinstance(meta, dict) and meta.get("source_kind") == "thesis_document"
    }
    if len(thesis_ids) != 1:
        return [], []

    thesis_id = next(iter(thesis_ids))
    seed_chunk_ids = list(
        dict.fromkeys(
            str(meta["chroma_chunk_id"])
            for meta in seed_metadata
            if isinstance(meta, dict)
            and meta.get("source_kind") == "thesis_document"
            and str(meta.get("thesis_id") or meta.get("doc_id")) == thesis_id
            and meta.get("chroma_chunk_id")
        )
    )
    if not seed_chunk_ids:
        return [], []

    try:
        from scripts.rag.rag_config import RAGConfig
        from scripts.thesis_graph.thesis_evidence_graph import (
            expand_evidence_chunk_ids,
            get_thesis_graph_path,
        )

        config = RAGConfig()
        expanded_ids = expand_evidence_chunk_ids(
            get_thesis_graph_path(Path(config.thesis_graphs_dir), thesis_id),
            thesis_id,
            seed_chunk_ids,
            max_chunks=config.thesis_graph_max_chunks,
        )
        if not expanded_ids:
            return [], []

        graph_results = _collection_get(
            collection,
            ids=expanded_ids,
            include=["documents", "metadatas"],
        )
        graph_candidates: Dict[str, Tuple[str, Dict]] = {}
        for chunk_id, document, metadata in zip(
            graph_results.get("ids", []),
            graph_results.get("documents", []),
            graph_results.get("metadatas", []),
        ):
            graph_metadata = dict(metadata)
            graph_metadata["chroma_chunk_id"] = chunk_id
            graph_metadata["retrieval_method"] = "thesis_graph"
            graph_candidates[chunk_id] = (document, graph_metadata)

        ordered_candidates = [
            graph_candidates[chunk_id] for chunk_id in expanded_ids if chunk_id in graph_candidates
        ]
        return (
            [document for document, _ in ordered_candidates],
            [metadata for _, metadata in ordered_candidates],
        )
    except Exception as thesis_graph_error:
        logger.debug(f"Thesis graph expansion skipped: {thesis_graph_error}")
        return [], []


def _attach_thesis_graph_provenance(metadatas: List[Dict], logger: Any) -> List[Dict]:
    """Attach section-to-citation graph paths to retrieved thesis chunks.
    Args:
        metadatas (List[Dict]): List of metadata dictionaries for retrieved chunks.
        logger (Any): Logger instance for logging.

    Returns:
        List[Dict]: Updated list of metadata dictionaries with attached graph provenance if available.
    """
    thesis_ids = {
        str(meta.get("thesis_id") or meta.get("doc_id"))
        for meta in metadatas
        if isinstance(meta, dict) and meta.get("source_kind") == "thesis_document"
    }
    if len(thesis_ids) != 1:
        return metadatas

    thesis_id = next(iter(thesis_ids))
    chunk_ids = list(
        dict.fromkeys(
            str(meta.get("chroma_chunk_id") or meta.get("chunk_id"))
            for meta in metadatas
            if isinstance(meta, dict)
            and meta.get("source_kind") == "thesis_document"
            and (meta.get("chroma_chunk_id") or meta.get("chunk_id"))
        )
    )
    if not chunk_ids:
        return metadatas

    try:
        from scripts.rag.rag_config import RAGConfig
        from scripts.thesis_graph.thesis_evidence_graph import (
            get_chunk_provenance_paths,
            get_thesis_graph_path,
        )

        config = RAGConfig()
        graph_path = get_thesis_graph_path(Path(config.thesis_graphs_dir), thesis_id)
        paths_by_chunk = get_chunk_provenance_paths(graph_path, thesis_id, chunk_ids)
        for metadata in metadatas:
            if not isinstance(metadata, dict):
                continue
            chunk_id = str(metadata.get("chroma_chunk_id") or metadata.get("chunk_id") or "")
            paths = paths_by_chunk.get(chunk_id)
            if paths:
                metadata["graph_provenance"] = "; ".join(paths)
    except Exception as provenance_error:
        logger.debug(f"Thesis graph provenance skipped: {provenance_error}")
    return metadatas


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
    graph_chunks: Optional[List[str]] = None,
    graph_metadata: Optional[List[Dict]] = None,
) -> Tuple[List[str], List[Dict], int, int]:
    """Combine vector/keyword/counts results with optional weighted combination.

    Returns final chunks/metadata limited to k, and counts of vector/keyword items
    for logging/audit. This helper is pure with respect to inputs and is easy to unit test.

    Args:
        vector_chunks (List[str]): List of chunks retrieved via vector search.
        vector_metadata (List[Dict]): Corresponding metadata for vector chunks.
        keyword_chunks (List[str]): List of chunks retrieved via keyword search.
        keyword_metadata (List[Dict]): Corresponding metadata for keyword chunks.
        counts_chunks (List[str]): List of chunks retrieved via counts-based search.
        counts_metadata (List[Dict]): Corresponding metadata for counts chunks.
        k (int): Maximum number of chunks to return.
        logger (Any): Logger instance for logging.
        use_weights (bool, optional): Whether to use weighted combination. Defaults to True.
        graph_chunks (Optional[List[str]], optional): List of chunks retrieved from the graph. Defaults to None.
        graph_metadata (Optional[List[Dict]], optional): Corresponding metadata for graph chunks. Defaults to None.

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
                graph_chunks=graph_chunks or [],
                graph_metadata=graph_metadata or [],
                graph_scores=[1.0 / (i + 1) for i in range(len(graph_chunks or []))],
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

        for chunk, meta in zip(graph_chunks or [], graph_metadata or []):
            if chunk not in seen_chunks and len(combined_chunks) < k * 2:
                seen_chunks.add(chunk)
                combined_chunks.append(chunk)
                meta_copy = dict(meta) if meta else {}
                meta_copy["retrieval_method"] = "thesis_graph"
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


def _embed_query(query: str, model_name: str) -> List[float]:
    """Embed a query string using Ollama with retry and rate limit."""
    from langchain_ollama import OllamaEmbeddings

    attempt = 0

    @retry_ollama_call(max_retries=3, initial_delay=1.0, operation_name="embed_query")
    def embed_once() -> List[float]:
        nonlocal attempt
        attempt += 1
        limiter = get_rate_limiter()
        if limiter:
            limiter.acquire()

        embed_model = OllamaEmbeddings(model=model_name)
        started_at = time.perf_counter()
        try:
            try:
                embedding = embed_model.embed_query(query)
            except AttributeError:
                if not hasattr(embed_model, "embed_documents"):
                    raise
                embedding = embed_model.embed_documents([query])[0]
        except Exception as exc:
            record_embedding_usage(
                "retrieve.query_embedding",
                "rag_retrieval",
                model_name,
                [query],
                success=False,
                attempt=attempt,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                failure_reason=type(exc).__name__,
            )
            raise

        record_embedding_usage(
            "retrieve.query_embedding",
            "rag_retrieval",
            model_name,
            [query],
            success=True,
            attempt=attempt,
            latency_ms=(time.perf_counter() - started_at) * 1000,
        )
        return embedding

    return embed_once()


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

    return _collection_query(
        collection,
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
        filters: Optional metadata filters (e.g., {"source_kind": "thesis_document"})

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
                            chunk_data = _collection_get(
                                collection, ids=[doc_id], include=["documents", "metadatas"]
                            )

                            if chunk_data["ids"] and chunk_data["documents"]:
                                meta = chunk_data["metadatas"][0] if chunk_data["metadatas"] else {}
                                if filters and not all(
                                    meta.get(key) == value for key, value in filters.items()
                                ):
                                    continue
                                chunks.append(chunk_data["documents"][0])
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
        filters: Optional metadata filters (e.g., {"source_kind": "thesis_document"})

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
        all_chunks = _collection_get(
            collection,
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
        term: Optional[str]
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
    filters: Dict[str, Any],
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
            collection, query_embedding, k, embedding_model_name, filters=filters
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
    vector_ids = (results.get("ids") or [[]])[0]

    # Merge distances into metadata for explainability
    for i, meta in enumerate(vector_metadata):
        if not isinstance(meta, dict):
            continue
        if i < len(vector_ids):
            meta["chroma_chunk_id"] = vector_ids[i]
        if i < len(vector_distances):
            meta["distance"] = vector_distances[i]

    if vector_chunks:
        filter_info = f" (filters: {filters})" if filters else ""
        logger.info(f"Vector search retrieved {len(vector_chunks)} chunks{filter_info}")

    return vector_chunks, vector_metadata


def _run_keyword_search(
    query: str,
    collection: Collection,
    k: int,
    filters: Dict[str, Any],
    use_hybrid: bool,
    has_vector_results: bool,
    logger,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Execute keyword/BM25 search when hybrid is enabled or vector search is empty."""
    if not use_hybrid and has_vector_results:
        return [], []

    keyword_chunks, keyword_metadata, _ = _bm25_search_with_fallback(
        query, collection, k, filters=filters
    )

    if keyword_chunks:
        filter_info = f" (filters: {filters})" if filters else ""
        logger.info(
            f"✓ Keyword search activated: found {len(keyword_chunks)} term matches{filter_info}"
        )
        audit(
            "keyword_search_used",
            {
                "query": query[:100],
                "matches_found": len(keyword_chunks),
                "reason": "hybrid_search" if has_vector_results else "vector_search_failed",
                "filters_applied": bool(filters),
            },
        )

    return keyword_chunks, keyword_metadata


def retrieve(
    query: str,
    collection: Collection,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    persona: Optional[str] = None,
    domain: Optional[str] = None,
    enable_thesis_graph: bool = False,
) -> Tuple[List[str], List[Dict]]:
    """Retrieve semantically similar chunks for a query using hybrid search.

    Simplified interface for backward compatibility. Delegates to retrieve_with_filters()
    with hybrid search enabled by default.

    Combines vector similarity search with keyword matching for better retrieval.
    Vector search provides semantic understanding while keyword search catches
    exact term matches that embeddings might miss.

    Passes caller-supplied metadata filters through to vector and keyword search.

    Args:
        query: User question or query text.
        collection: ChromaDB collection to search.
        k: Number of results to retrieve (default: 5).
        filters: Optional explicit metadata filters (ChromaDB where conditions).
        persona: Optional persona name ("supervisor", "assessor", "researcher") to apply
            persona-aware filtering and reranking when metadata is available.
        domain: Optional domain for domain-specific term expansion (e.g., 'aboriginal_torres_strait_islander').
        enable_thesis_graph: Expand thesis-scoped results with graph-linked evidence.

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

        >>> chunks, meta = retrieve(
        ...     "Rate the methodology", collection,
        ...     filters={"source_kind": "thesis_document"}
        ... )
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

    # Preserve explicit filters from callers such as the dashboard, then layer
    # convenience filters on top.
    filters = filters.copy() if filters else {}

    # Delegate to comprehensive retrieve_with_filters
    return retrieve_with_filters(
        query=query,
        collection=collection,
        k=k,
        filters=filters if filters else None,
        persona=persona,
        domain=domain,
        enable_hybrid_search=True,  # retrieve() always uses hybrid search
        enable_reranking=False,  # Lightweight reranking disabled by default
        enable_learned_reranking=enable_learned_reranking,
        reranker_model=reranker_model,
        fetch_neighbours=False,  # Don't fetch neighbours by default
        enable_caching=False,  # Caching disabled for simple retrieve()
        enable_graph=False,  # Graph expansion disabled for simple retrieve()
        enable_thesis_graph=enable_thesis_graph,
        enable_parent_child=enable_parent_child,
        cache_dir=None,
    )


def detect_filters_from_query(
    query: str,
) -> Dict[str, Any]:
    """Auto-detect metadata filters from natural language query.

        Analyses query text for keywords that suggest documentation categories,
        technical domains, and content types. Constructs
    ChromaDB filter conditions to narrow search space.

    Args:
        query: Natural language query text

    Returns:
        Dictionary of ChromaDB where conditions

    Example:
        >>> detect_filters_from_query("Show me API security policies")
        {'source_category': 'governance', 'is_api_reference': True}

    """
    filters: Dict[str, Any] = {}
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

    academic_like = any(term in query_lower for term in academic_terms)
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
        if not academic_like:
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
    persona: Optional[str] = None,
    domain: Optional[str] = None,
    auto_detect_filters: bool = True,
    enable_hybrid_search: bool = True,
    enable_reranking: bool = False,
    enable_learned_reranking: bool = True,
    reranker_model: str = "BAAI/bge-reranker-base",
    fetch_neighbours: bool = False,
    enable_caching: bool = True,
    enable_graph: bool = True,
    enable_thesis_graph: bool = False,
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
        persona: Optional persona name ("supervisor", "assessor", "researcher")
        domain: Optional domain for domain-specific term expansion (e.g., 'aboriginal_torres_strait_islander')
        auto_detect_filters: Automatically detect filters from query
        enable_hybrid_search: Use hybrid vector + keyword search
        enable_reranking: Apply lightweight re-ranking
        enable_learned_reranking: Use cross-encoder reranking
        reranker_model: Cross-encoder model name
        fetch_neighbours: Fetch prev/next chunks for expanded context
        enable_caching: Use context cache for hot entities
        enable_graph: Expand with graph-connected chunks
        enable_thesis_graph: Expand thesis-scoped results through the thesis evidence graph
        enable_parent_child: Replace matched children with parent chunks
        cache_dir: Directory for cache file (default: rag_data/)

    Returns:
        Tuple of (chunks, metadata) with enhanced retrieval

    Example:
        >>> chunks, meta = retrieve_with_filters(
        ...     "API authentication config",
        ...     collection,
        ...     k=5,
        ...     filters={"source_kind": "thesis_document"},
        ...     enable_graph=True,
        ...     persona="assessor"
        ... )
    """
    logger = get_logger()

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    candidate_k = min(k * 3, 50) if persona else k
    if candidate_k > k:
        logger.info(
            "Retrieving %d candidates for persona '%s' before selecting %d final chunks",
            candidate_k,
            persona,
            k,
        )

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

    # Build the complete retrieval scope before consulting the context cache.
    # A query-only cache key can otherwise return unfiltered results to a
    # thesis-only request, bypassing the caller's source constraints.
    combined_filters = filters.copy() if filters else {}
    if auto_detect_filters:
        auto_filters = detect_filters_from_query(query)
        for key, value in auto_filters.items():
            if key not in combined_filters:
                combined_filters[key] = value

    retrieval_start = time.perf_counter()
    cache_hit = False

    if enable_caching:
        from .context_cache import get_context_cache

        if cache_dir is None:
            from .rag_config import RAGConfig

            config = RAGConfig()
            cache_dir = Path(config.rag_data_path)

        cache_scope = {
            "query": query.lower().strip(),
            "k": k,
            "filters": combined_filters,
            "persona": persona,
            "enable_hybrid_search": enable_hybrid_search,
            "enable_reranking": enable_reranking,
            "enable_learned_reranking": enable_learned_reranking,
            "enable_graph": enable_graph,
            "enable_thesis_graph": enable_thesis_graph,
            "enable_parent_child": enable_parent_child,
            "fetch_neighbours": fetch_neighbours,
        }
        cache_key = json.dumps(cache_scope, sort_keys=True, default=str)
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

    # Log filters for debugging
    if combined_filters:
        logger.info(f"Applying filters: {combined_filters}")
        audit("retrieve_filters", {"query": query[:100], "filters": combined_filters})

    # 0. Agentic SQL counts branch
    counts_chunks: List[str] = []
    counts_metadata: List[Dict[str, Any]] = []
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
        candidate_k,
        combined_filters,
        EMBEDDING_MODEL_NAME,
        logger,
    )

    # 2. Keyword/BM25 search (if hybrid enabled or vector search failed)
    keyword_chunks: List[str] = []
    keyword_metadata: List[Dict[str, Any]] = []
    if enable_hybrid_search:
        keyword_chunks, keyword_metadata = _run_keyword_search(
            query,
            collection,
            candidate_k,
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

    graph_chunks: List[str] = []
    graph_metadata: List[Dict[str, Any]] = []
    if enable_thesis_graph:
        graph_chunks, graph_metadata = _expand_thesis_graph_candidates(
            collection,
            vector_metadata + keyword_metadata,
            logger,
        )

    documents, metadatas, vector_count, keyword_count = _combine_results(
        vector_chunks,
        vector_metadata,
        keyword_chunks,
        keyword_metadata,
        counts_chunks,
        counts_metadata,
        candidate_k,
        logger,
        use_weights=use_hybrid_weights,
        graph_chunks=graph_chunks,
        graph_metadata=graph_metadata,
    )
    if enable_thesis_graph:
        metadatas = _attach_thesis_graph_provenance(metadatas, logger)

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
                        graph_results = _collection_get(
                            collection,
                            where={"chunk_id": new_id},
                            include=["documents", "metadatas"],
                        )
                        if graph_results.get("documents"):
                            documents.append(graph_results["documents"][0])
                            metadatas.append(graph_results["metadatas"][0])
                    except Exception as e:
                        logger.debug(f"Failed to fetch graph chunk {new_id}: {e}")

    # Persona rules must operate on a wider candidate pool, otherwise they can
    # only reorder the already selected k chunks and cannot affect membership.
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
            results = _collection_get(
                collection,
                ids=list(neighbour_ids),
                include=["documents", "metadatas"],
            )
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
    """Summarise retrieval methods, similarity and source-kind coverage."""
    del query
    similarities: List[Optional[float]] = []
    for metadata in metadatas:
        safe_metadata = metadata if isinstance(metadata, dict) else {}
        distance = safe_metadata.get("distance")
        similarity = 1.0 - min(distance, 1.0) if distance is not None else None
        similarities.append(round(similarity, 3) if similarity is not None else None)

    retrieval_methods = {
        (metadata if isinstance(metadata, dict) else {}).get("retrieval_method", "vector")
        for metadata in metadatas
    }
    valid_similarities = [score for score in similarities if score is not None]
    if valid_similarities:
        average_similarity = sum(valid_similarities) / len(valid_similarities)
        confidence = (
            "high"
            if average_similarity >= 0.7
            else "medium" if average_similarity >= 0.5 else "low"
        )
    else:
        average_similarity = 0.0
        confidence = "unknown"

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
    if valid_similarities:
        ranking_explanation += f". Top match has {max(valid_similarities):.1%} similarity."

    source_kinds = {
        metadata["source_kind"]
        for metadata in metadatas
        if isinstance(metadata, dict) and isinstance(metadata.get("source_kind"), str)
    }
    return {
        "retrieval_method": sorted(retrieval_methods),
        "ranking_explanation": ranking_explanation,
        "similarity_scores": similarities,
        "confidence_level": confidence,
        "avg_similarity": round(average_similarity, 3),
        "metadata_insights": {
            "source_kinds": sorted(source_kinds),
            "total_chunks": len(chunks),
            "k_requested": k,
        },
    }
