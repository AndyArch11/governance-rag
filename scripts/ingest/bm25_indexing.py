"""Common BM25 indexing utilities for chunk-level ingestion.

Provides centralised BM25 indexing logic at chunk granularity,
ensuring consistency across ingest.py, ingest_git.py, and ingest_academic.py.

This module handles:
- Indexing regular chunks with IDs: doc_id-chunk-{i}
- Indexing child chunks with IDs: doc_id-{child['id']}
- Indexing parent chunks with IDs: doc_id-{parent['id']}
- Consistent error handling and logging
- Audit trail creation

Usage:
    from scripts.ingest.bm25_indexing import index_chunks_in_bm25
    
    total_indexed = index_chunks_in_bm25(
        doc_id="my_doc",
        chunks=chunks,
        child_chunks=child_chunks,
        parent_chunks=parent_chunks,
        config=config,
        cache_db=cache_db,
        logger=logger,
    )
"""

from collections import Counter
from typing import Any, Dict, List, Optional

from scripts.search.bm25_search import BM25Search
from scripts.utils.db_factory import get_cache_client
from scripts.utils.logger import get_logger


def index_chunks_in_bm25(
    doc_id: str,
    chunks: List[str],
    child_chunks: Optional[List[Dict[str, str]]] = None,
    parent_chunks: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Any] = None,
    cache_db: Optional[Any] = None,
    logger: Optional[Any] = None,
) -> int:
    """Index chunks at chunk-level granularity for BM25 search.

    Indexes all chunk types (regular, child, parent) with consistent
    chunk-level IDs matching ChromaDB schema. This ensures hybrid search
    can correctly locate exact chunks when BM25 results are returned.

    Args:
        doc_id: Unique document identifier
        chunks: List of chunk texts (regular chunks)
        child_chunks: Optional list of child chunk dicts with {id, text, parent_id}
        parent_chunks: Optional list of parent chunk dicts with {id, text, child_ids}
        config: Ingest config object with bm25_index_original_text setting
        cache_db: Cache database client for storing BM25 documents
        logger: Logger instance for debug/info logging

    Returns:
        Total number of chunks indexed

    Raises:
        ValueError: If required parameters are missing
        Exception: Propagates exceptions from BM25 tokenisation or cache_db storage

    Example:
        >>> from scripts.utils.db_factory import get_cache_client
        >>> cache_db = get_cache_client(enable_cache=True)
        >>> total = index_chunks_in_bm25(
        ...     doc_id="my_doc",
        ...     chunks=["chunk1 text", "chunk2 text"],
        ...     config=config,
        ...     cache_db=cache_db,
        ...     logger=logger,
        ... )
        >>> print(f"Indexed {total} chunks")
    """
    if not doc_id or (not chunks and not child_chunks and not parent_chunks):
        raise ValueError("doc_id and chunks are required")

    if cache_db is None:
        cache_db = get_cache_client(enable_cache=True)

    if logger is None:
        logger = get_logger()

    # Initialise BM25 tokeniser
    bm25 = BM25Search()

    total_indexed = 0
    total_tokens = 0

    try:
        # Index regular chunks at chunk-level granularity
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}-chunk-{chunk_idx}"
            tokens = bm25.tokenise(chunk_text)
            term_freq = Counter(tokens)

            cache_db.put_bm25_document(
                doc_id=chunk_id,
                term_frequencies=dict(term_freq),
                doc_length=len(tokens),
                original_text=(
                    chunk_text
                    if (config and getattr(config, "bm25_index_original_text", False))
                    else None
                ),
            )
            total_indexed += 1
            total_tokens += len(tokens)

        # Index child chunks if present
        if child_chunks:
            for child_chunk in child_chunks:
                child_id = f"{doc_id}-{child_chunk['id']}"
                tokens = bm25.tokenise(child_chunk["text"])
                term_freq = Counter(tokens)

                cache_db.put_bm25_document(
                    doc_id=child_id,
                    term_frequencies=dict(term_freq),
                    doc_length=len(tokens),
                    original_text=child_chunk["text"]
                    if (config and getattr(config, "bm25_index_original_text", False))
                    else None,
                )
                total_indexed += 1
                total_tokens += len(tokens)

        # Index parent chunks if present
        if parent_chunks:
            for parent_chunk in parent_chunks:
                parent_id = f"{doc_id}-{parent_chunk['id']}"
                tokens = bm25.tokenise(parent_chunk["text"])
                term_freq = Counter(tokens)

                cache_db.put_bm25_document(
                    doc_id=parent_id,
                    term_frequencies=dict(term_freq),
                    doc_length=len(tokens),
                    original_text=parent_chunk["text"]
                    if (config and getattr(config, "bm25_index_original_text", False))
                    else None,
                )
                total_indexed += 1
                total_tokens += len(tokens)

        logger.debug(
            f"BM25 indexed {total_indexed} chunks for {doc_id}: "
            f"{total_tokens} tokens total, granularity=chunk-level"
        )

        return total_indexed

    except Exception as e:
        logger.error(
            f"BM25 indexing failed for {doc_id} after {total_indexed} chunks: {e}"
        )
        raise
