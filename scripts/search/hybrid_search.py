"""Hybrid search combining semantic (vector) and keyword (BM25) search.

Provides multiple fusion strategies:
- Weighted linear combination: alpha * semantic + (1-alpha) * keyword
- Reciprocal Rank Fusion (RRF): Combines ranked lists without score normalisation
- Custom reranking strategies

Searches are executed in parallel using ThreadPoolExecutor for optimal performance.

References:
    - Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
      Reciprocal rank fusion outperforms condorcet and individual rank learning methods.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Strategy for combining semantic and keyword search results."""

    LINEAR = "linear"  # Weighted linear combination
    RRF = "rrf"  # Reciprocal Rank Fusion
    MAX = "max"  # Take maximum score
    MIN = "min"  # Take minimum score


@dataclass
class SearchResult:
    """A single search result with metadata."""

    doc_id: str
    score: float
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    rank: Optional[int] = None


class HybridSearch:
    """Hybrid search combining semantic and keyword approaches."""

    def __init__(
        self,
        semantic_search_fn=None,
        keyword_search_fn=None,
        fusion_strategy: FusionStrategy = FusionStrategy.RRF,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ):
        """Initialise hybrid search.

        Args:
            semantic_search_fn: Function(query, top_k) -> List[(doc_id, score)]
            keyword_search_fn: Function(query, top_k) -> List[(doc_id, score)]
            fusion_strategy: How to combine results
            alpha: Weight for linear combination (0.0 = keyword only, 1.0 = semantic only)
            rrf_k: Constant for RRF formula (typically 60)
        """
        self.semantic_search_fn = semantic_search_fn
        self.keyword_search_fn = keyword_search_fn
        self.fusion_strategy = fusion_strategy
        self.alpha = alpha
        self.rrf_k = rrf_k

    def _normalise_scores(self, results: List[Tuple[str, float]]) -> Dict[str, float]:
        """Normalise scores to [0, 1] range using min-max normalisation.

        Args:
            results: List of (doc_id, score) tuples

        Returns:
            Dictionary mapping doc_id -> normalised_score
        """
        if not results:
            return {}

        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)

        # Handle case where all scores are the same
        if max_score == min_score:
            return {doc_id: 1.0 for doc_id, _ in results}

        # Min-max normalisation
        normalised = {}
        for doc_id, score in results:
            norm_score = (score - min_score) / (max_score - min_score)
            normalised[doc_id] = norm_score

        return normalised

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[str, float]],
    ) -> List[SearchResult]:
        """Combine results using Reciprocal Rank Fusion.

        RRF score = Σ 1 / (k + rank_i)
        where k is a constant (typically 60) and rank_i is the rank in list i.

        Args:
            semantic_results: Semantic search results
            keyword_results: Keyword search results

        Returns:
            Combined results sorted by RRF score
        """
        # Build rank mappings
        semantic_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(semantic_results)}
        keyword_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(keyword_results)}

        # Get all unique documents
        all_docs = set(semantic_ranks.keys()) | set(keyword_ranks.keys())

        # Compute RRF scores
        rrf_scores = {}
        for doc_id in all_docs:
            score = 0.0

            if doc_id in semantic_ranks:
                score += 1.0 / (self.rrf_k + semantic_ranks[doc_id])

            if doc_id in keyword_ranks:
                score += 1.0 / (self.rrf_k + keyword_ranks[doc_id])

            rrf_scores[doc_id] = score

        # Build results with metadata
        results = []
        for doc_id, score in rrf_scores.items():
            result = SearchResult(
                doc_id=doc_id,
                score=score,
                semantic_score=semantic_ranks.get(doc_id),
                keyword_score=keyword_ranks.get(doc_id),
            )
            results.append(result)

        # Sort by RRF score descending
        results.sort(key=lambda x: x.score, reverse=True)

        # Add final ranks
        for rank, result in enumerate(results, 1):
            result.rank = rank

        return results

    def _linear_combination(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[str, float]],
    ) -> List[SearchResult]:
        """Combine results using weighted linear combination.

        Combined score = alpha * semantic_score + (1 - alpha) * keyword_score

        Args:
            semantic_results: Semantic search results
            keyword_results: Keyword search results

        Returns:
            Combined results sorted by hybrid score
        """
        # Normalise scores
        semantic_norm = self._normalise_scores(semantic_results)
        keyword_norm = self._normalise_scores(keyword_results)

        # Get all unique documents
        all_docs = set(semantic_norm.keys()) | set(keyword_norm.keys())

        # Compute hybrid scores
        hybrid_scores = {}
        for doc_id in all_docs:
            sem_score = semantic_norm.get(doc_id, 0.0)
            key_score = keyword_norm.get(doc_id, 0.0)

            # Weighted combination
            hybrid_score = self.alpha * sem_score + (1 - self.alpha) * key_score
            hybrid_scores[doc_id] = hybrid_score

        # Build results with metadata
        results = []
        for doc_id, score in hybrid_scores.items():
            result = SearchResult(
                doc_id=doc_id,
                score=score,
                semantic_score=semantic_norm.get(doc_id),
                keyword_score=keyword_norm.get(doc_id),
            )
            results.append(result)

        # Sort by hybrid score descending
        results.sort(key=lambda x: x.score, reverse=True)

        # Add final ranks
        for rank, result in enumerate(results, 1):
            result.rank = rank

        return results

    def search(
        self,
        query: str,
        top_k: int = 10,
        semantic_top_k: Optional[int] = None,
        keyword_top_k: Optional[int] = None,
        parallel: bool = True,
    ) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword approaches.

        Args:
            query: Search query
            top_k: Number of final results to return
            semantic_top_k: Number of semantic results to retrieve (default: top_k * 2)
            keyword_top_k: Number of keyword results to retrieve (default: top_k * 2)
            parallel: Execute searches in parallel (default: True)

        Returns:
            List of SearchResult objects sorted by hybrid score
        """
        # Default to retrieving more candidates than final top_k
        semantic_top_k = semantic_top_k or (top_k * 2)
        keyword_top_k = keyword_top_k or (top_k * 2)

        semantic_results = []
        keyword_results = []

        if parallel and self.semantic_search_fn and self.keyword_search_fn:
            # Execute searches in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both searches concurrently
                semantic_future = executor.submit(self.semantic_search_fn, query, semantic_top_k)
                keyword_future = executor.submit(self.keyword_search_fn, query, keyword_top_k)

                # Retrieve semantic results
                try:
                    semantic_results = semantic_future.result()
                    logger.info(f"Retrieved {len(semantic_results)} semantic results (parallel)")
                except Exception as e:
                    logger.error(f"Semantic search failed: {e}")

                # Retrieve keyword results
                try:
                    keyword_results = keyword_future.result()
                    logger.info(f"Retrieved {len(keyword_results)} keyword results (parallel)")
                except Exception as e:
                    logger.error(f"Keyword search failed: {e}")
        else:
            # Sequential execution (fallback or when only one search is available)
            # Retrieve semantic results
            if self.semantic_search_fn:
                try:
                    semantic_results = self.semantic_search_fn(query, semantic_top_k)
                    logger.info(f"Retrieved {len(semantic_results)} semantic results (sequential)")
                except Exception as e:
                    logger.error(f"Semantic search failed: {e}")

            # Retrieve keyword results
            if self.keyword_search_fn:
                try:
                    keyword_results = self.keyword_search_fn(query, keyword_top_k)
                    logger.info(f"Retrieved {len(keyword_results)} keyword results (sequential)")
                except Exception as e:
                    logger.error(f"Keyword search failed: {e}")

        # Handle case where one or both searches failed
        if not semantic_results and not keyword_results:
            logger.warning("Both semantic and keyword search returned no results")
            return []

        if not semantic_results:
            logger.info("Falling back to keyword-only results")
            return [
                SearchResult(doc_id=doc_id, score=score, keyword_score=score)
                for doc_id, score in keyword_results[:top_k]
            ]

        if not keyword_results:
            logger.info("Falling back to semantic-only results")
            return [
                SearchResult(doc_id=doc_id, score=score, semantic_score=score)
                for doc_id, score in semantic_results[:top_k]
            ]

        # Combine results using selected strategy
        if self.fusion_strategy == FusionStrategy.RRF:
            combined = self._reciprocal_rank_fusion(semantic_results, keyword_results)
        elif self.fusion_strategy == FusionStrategy.LINEAR:
            combined = self._linear_combination(semantic_results, keyword_results)
        else:
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}")

        logger.info(
            f"Combined {len(semantic_results)} semantic + {len(keyword_results)} keyword -> {len(combined)} results"
        )

        return combined[:top_k]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Mock search functions for demonstration
    def mock_semantic_search(query: str, top_k: int) -> List[Tuple[str, float]]:
        """Mock semantic search returning similarity scores."""
        # Simulated semantic similarity scores
        results = [
            ("doc1", 0.95),
            ("doc2", 0.87),
            ("doc3", 0.82),
            ("doc4", 0.75),
            ("doc5", 0.68),
        ]
        return results[:top_k]

    def mock_keyword_search(query: str, top_k: int) -> List[Tuple[str, float]]:
        """Mock BM25 search returning BM25 scores."""
        # Simulated BM25 scores (different ranking)
        results = [
            ("doc3", 12.5),
            ("doc1", 10.2),
            ("doc6", 8.7),
            ("doc2", 7.5),
            ("doc7", 6.2),
        ]
        return results[:top_k]

    # Test both fusion strategies
    print("\n" + "=" * 70)
    print("Hybrid Search Demo")
    print("=" * 70)

    # Reciprocal Rank Fusion
    print("\n1. Reciprocal Rank Fusion (RRF)")
    print("-" * 70)
    hybrid_rrf = HybridSearch(
        semantic_search_fn=mock_semantic_search,
        keyword_search_fn=mock_keyword_search,
        fusion_strategy=FusionStrategy.RRF,
    )

    results_rrf = hybrid_rrf.search("sample query", top_k=5)
    for result in results_rrf:
        print(f"  Rank {result.rank}: {result.doc_id} (score={result.score:.4f})")

    # Linear Combination
    print("\n2. Linear Combination (alpha=0.6)")
    print("-" * 70)
    hybrid_linear = HybridSearch(
        semantic_search_fn=mock_semantic_search,
        keyword_search_fn=mock_keyword_search,
        fusion_strategy=FusionStrategy.LINEAR,
        alpha=0.6,
    )

    results_linear = hybrid_linear.search("sample query", top_k=5)
    for result in results_linear:
        sem_str = f"{result.semantic_score:.2f}" if result.semantic_score is not None else "N/A"
        key_str = f"{result.keyword_score:.2f}" if result.keyword_score is not None else "N/A"
        print(
            f"  Rank {result.rank}: {result.doc_id} (score={result.score:.4f}, "
            f"sem={sem_str}, key={key_str})"
        )

    print("\n" + "=" * 70)
