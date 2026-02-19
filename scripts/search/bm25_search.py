"""BM25 keyword search implementation for document retrieval.

BM25 (Best Matching 25) is a probabilistic ranking function used for keyword-based
document retrieval. It improves upon TF-IDF by:
- Using term saturation (diminishing returns for repeated terms)
- Normalising for document length
- Tunable parameters (k1, b) for different collections

Reference:
    Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework:
    BM25 and Beyond. Foundations and Trends in Information Retrieval.
"""

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from scripts.search.text_preprocessing import PreprocessingStrategy, TextPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class BM25Config:
    """Configuration for BM25 algorithm."""

    k1: float = 1.5  # Term frequency saturation parameter (1.2-2.0 typical)
    b: float = 0.75  # Length normalisation parameter (0.0-1.0, 0.75 typical)
    epsilon: float = 0.25  # Floor value for IDF to avoid division issues


@dataclass
class BM25Index:
    """Inverted index for BM25 search."""

    # Document ID -> tokenised document
    documents: Dict[str, List[str]] = field(default_factory=dict)

    # Term -> set of document IDs containing term
    inverted_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    # Document ID -> document length (token count)
    doc_lengths: Dict[str, int] = field(default_factory=dict)

    # Term -> inverse document frequency
    idf_cache: Dict[str, float] = field(default_factory=dict)

    # Average document length across corpus
    avg_doc_length: float = 0.0

    # Total number of documents
    num_docs: int = 0


class BM25Search:
    """BM25 ranking function for keyword-based document retrieval."""

    def __init__(self, config: Optional[BM25Config] = None):
        """Initialise BM25 search engine.

        Args:
            config: BM25 configuration parameters
        """
        self.config = config or BM25Config()
        self.index = BM25Index()

        # Use shared text preprocessor for consistent tokenisation and stopword filtering
        # This ensures BM25 uses the same preprocessing as the rest of the search pipeline
        self.preprocessor = TextPreprocessor(
            strategy=PreprocessingStrategy.LOWERCASE, remove_stopwords=True, min_token_length=2
        )

    def tokenise(self, text: str) -> List[str]:
        """Tokenise text into terms using shared preprocessor.

        Uses TextPreprocessor for consistent tokenisation across search pipeline.
        Applies stopword filtering and minimum token length constraints.

        Args:
            text: Input text

        Returns:
            List of preprocessed tokens
        """
        return self.preprocessor.preprocess(text)

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the BM25 index.

        Args:
            doc_id: Unique document identifier
            text: Document text content
        """
        tokens = self.tokenise(text)

        # Store tokenised document
        self.index.documents[doc_id] = tokens
        self.index.doc_lengths[doc_id] = len(tokens)

        # Update inverted index
        for token in set(tokens):
            self.index.inverted_index[token].add(doc_id)

        # Update corpus statistics
        self.index.num_docs = len(self.index.documents)
        total_length = sum(self.index.doc_lengths.values())
        self.index.avg_doc_length = (
            total_length / self.index.num_docs if self.index.num_docs > 0 else 0
        )

        # Invalidate IDF cache (will be recomputed on next search)
        self.index.idf_cache.clear()

    def add_documents_batch(self, documents: Dict[str, str]) -> None:
        """Add multiple documents to the index.

        Args:
            documents: Dictionary mapping doc_id -> text
        """
        for doc_id, text in documents.items():
            tokens = self.tokenise(text)
            self.index.documents[doc_id] = tokens
            self.index.doc_lengths[doc_id] = len(tokens)

            for token in set(tokens):
                self.index.inverted_index[token].add(doc_id)

        # Update corpus statistics once at the end
        self.index.num_docs = len(self.index.documents)
        total_length = sum(self.index.doc_lengths.values())
        self.index.avg_doc_length = (
            total_length / self.index.num_docs if self.index.num_docs > 0 else 0
        )
        self.index.idf_cache.clear()

        logger.info(
            f"Indexed {len(documents)} documents. Total corpus: {self.index.num_docs} docs."
        )

    def _compute_idf(self, term: str) -> float:
        """Compute Inverse Document Frequency for a term.

        IDF = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
        where:
            N = total number of documents
            n(t) = number of documents containing term t

        Args:
            term: Query term

        Returns:
            IDF score
        """
        if term in self.index.idf_cache:
            return self.index.idf_cache[term]

        N = self.index.num_docs
        n_t = len(self.index.inverted_index.get(term, set()))

        # BM25 IDF formula with smoothing
        idf = math.log((N - n_t + 0.5) / (n_t + 0.5) + 1)

        # Apply epsilon floor to avoid negative IDF
        idf = max(self.config.epsilon, idf)

        self.index.idf_cache[term] = idf
        return idf

    def _compute_bm25_score(self, doc_id: str, query_terms: List[str]) -> float:
        """Compute BM25 score for a document given query terms.

        BM25(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
        where:
            D = document
            Q = query
            qi = query term
            f(qi, D) = frequency of qi in D
            |D| = length of D
            avgdl = average document length
            k1, b = tuning parameters

        Args:
            doc_id: Document identifier
            query_terms: List of query terms

        Returns:
            BM25 score
        """
        score = 0.0
        doc_tokens = self.index.documents[doc_id]
        doc_length = self.index.doc_lengths[doc_id]

        # Count term frequencies in document
        term_freqs = Counter(doc_tokens)

        k1 = self.config.k1
        b = self.config.b
        avgdl = self.index.avg_doc_length

        for term in query_terms:
            if term not in self.index.inverted_index:
                continue

            if doc_id not in self.index.inverted_index[term]:
                continue

            # IDF component
            idf = self._compute_idf(term)

            # Term frequency in document
            tf = term_freqs.get(term, 0)

            # BM25 term score with length normalisation
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avgdl))

            term_score = idf * (numerator / denominator)
            score += term_score

        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for documents matching the query.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        if self.index.num_docs == 0:
            logger.warning("BM25 index is empty. No documents to search.")
            return []

        # Tokenise query
        query_terms = self.tokenise(query)

        if not query_terms:
            logger.warning(f"Query tokenisation resulted in no terms: '{query}'")
            return []

        # Find candidate documents (documents containing at least one query term)
        candidate_docs = set()
        for term in query_terms:
            if term in self.index.inverted_index:
                candidate_docs.update(self.index.inverted_index[term])

        if not candidate_docs:
            logger.info(f"No documents found matching query: '{query}'")
            return []

        # Score each candidate document
        scores: List[Tuple[str, float]] = []
        for doc_id in candidate_docs:
            score = self._compute_bm25_score(doc_id, query_terms)
            scores.append((doc_id, score))

        # Sort by score descending and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"BM25 search for '{query}': {len(scores)} candidates, returning top {top_k}")
        return scores[:top_k]

    def get_stats(self) -> Dict[str, any]:
        """Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        return {
            "num_documents": self.index.num_docs,
            "num_unique_terms": len(self.index.inverted_index),
            "avg_doc_length": self.index.avg_doc_length,
            "total_tokens": sum(self.index.doc_lengths.values()),
            "config": {
                "k1": self.config.k1,
                "b": self.config.b,
                "epsilon": self.config.epsilon,
            },
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create BM25 search engine
    bm25 = BM25Search()

    # Sample documents
    docs = {
        "doc1": "The quick brown fox jumps over the lazy dog",
        "doc2": "Never jump over the lazy dog quickly",
        "doc3": "The dog is lazy but the fox is quick",
        "doc4": "Python is a programming language for data science",
        "doc5": "Machine learning and data science use Python frequently",
    }

    # Index documents
    bm25.add_documents_batch(docs)

    # Search queries
    queries = [
        "lazy dog",
        "quick fox",
        "python data science",
        "machine learning",
    ]

    print("\n" + "=" * 60)
    print("BM25 Search Demo")
    print("=" * 60)

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = bm25.search(query, top_k=3)
        for rank, (doc_id, score) in enumerate(results, 1):
            print(f"  {rank}. {doc_id}: {score:.4f} - {docs[doc_id][:60]}...")

    print("\n" + "=" * 60)
    print("Index Statistics:")
    print("=" * 60)
    stats = bm25.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
