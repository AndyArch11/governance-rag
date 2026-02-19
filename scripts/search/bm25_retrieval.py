"""BM25 retrieval using pre-built index from cache database.

Uses the BM25 index built during document ingestion for fast keyword-based retrieval.
The index is stored in SQLite (rag_data/cache.db) with term frequencies, IDF values,
and document lengths pre-computed.

This is more efficient than building the BM25 index on-the-fly at query time, especially
for large document corpora, and preserves the original full document text for accurate
keyword matching (rather than using summarised chunks which may lose keywords).

Usage:
    from scripts.search.bm25_retrieval import BM25Retriever

    retriever = BM25Retriever()
    results = retriever.search("query terms", top_k=10)
    # Returns: [(doc_id, score), ...]
"""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.search.text_preprocessing import PreprocessingStrategy, TextPreprocessor

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 retrieval using pre-built index from cache database."""

    def __init__(
        self,
        rag_data_path: Optional[Path] = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """Initialise BM25 retriever with pre-built index.

        Args:
            rag_data_path: Path to rag_data directory (defaults to repo rag_data/)
            k1: Term frequency saturation parameter (1.2-2.0 typical)
            b: Length normalisation parameter (0.0-1.0, 0.75 typical)
            epsilon: Floor value for IDF to avoid division issues
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Use shared text preprocessor for consistent tokenisation and stopword filtering
        # This ensures BM25 retrieval uses the same preprocessing as BM25 indexing
        self.preprocessor = TextPreprocessor(
            strategy=PreprocessingStrategy.LOWERCASE, remove_stopwords=True, min_token_length=2
        )

        # Initialise cache database connection
        from scripts.utils.db_factory import get_cache_client

        self.cache_db = get_cache_client(rag_data_path=rag_data_path, enable_cache=True)

        # Cache corpus statistics
        self.avg_doc_length = self.cache_db.get_bm25_avg_doc_length()
        self.total_docs = self.cache_db.get_bm25_corpus_size()

        if self.total_docs == 0:
            logger.warning("BM25 index is empty. Run ingestion with BM25_INDEXING_ENABLED=true")

    def __del__(self):
        """Cleanup: close database connection on object destruction."""
        if hasattr(self, "cache_db") and self.cache_db is not None:
            try:
                self.cache_db.close()
            except Exception:
                pass  # Ignore errors during cleanup

    def close(self):
        """Explicitly close the database connection."""
        if self.cache_db is not None:
            self.cache_db.close()
            self.cache_db = None

    def tokenise(self, text: str) -> List[str]:
        """Tokenise text into terms using shared preprocessor.

        Uses TextPreprocessor for consistent tokenisation and stopword filtering.
        Ensures retrieval uses the same preprocessing as indexing.

        Args:
            text: Input text

        Returns:
            List of preprocessed tokens
        """
        return self.preprocessor.preprocess(text)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for documents matching the query using pre-built BM25 index.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        if self.total_docs == 0:
            logger.warning("BM25 index is empty. No documents to search.")
            return []

        # Tokenise query
        query_terms = self.tokenise(query)

        if not query_terms:
            logger.warning(f"Query tokenisation resulted in no terms: '{query}'")
            return []

        # Collect all candidate documents (documents containing any query term)
        candidate_docs: Dict[str, float] = {}

        for term in query_terms:
            # Get term stats (document frequency and IDF)
            term_stats = self.cache_db.get_bm25_term_stats(term)

            if term_stats is None:
                # Term not in index
                continue

            df, idf = term_stats

            # Get all documents containing this term
            docs_with_term = self.cache_db.get_bm25_docs_with_term(term)

            for doc_id, term_freq, doc_length in docs_with_term:
                # Compute BM25 score contribution for this term
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )
                term_score = idf * (numerator / denominator)

                # Accumulate score for this document
                if doc_id in candidate_docs:
                    candidate_docs[doc_id] += term_score
                else:
                    candidate_docs[doc_id] = term_score

        # Sort by score descending
        sorted_results = sorted(candidate_docs.items(), key=lambda x: x[1], reverse=True)

        return sorted_results[:top_k]

    def get_stats(self) -> Dict[str, any]:
        """Get BM25 index statistics.

        Returns:
            Dict with corpus statistics
        """
        return {
            "total_documents": self.total_docs,
            "avg_doc_length": self.avg_doc_length,
            "config": {
                "k1": self.k1,
                "b": self.b,
                "epsilon": self.epsilon,
            },
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    retriever = BM25Retriever()

    print(f"\nBM25 Index Stats: {retriever.get_stats()}")

    # Example queries
    queries = [
        "authentication security",
        "data privacy policy",
        "kubernetes deployment",
    ]

    print("\n" + "=" * 60)
    print("BM25 Retrieval Demo (using pre-built index)")
    print("=" * 60)

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = retriever.search(query, top_k=5)

        if results:
            for rank, (doc_id, score) in enumerate(results, 1):
                print(f"  {rank}. {doc_id}: {score:.4f}")
        else:
            print("  No results found")
