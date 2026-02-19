"""Configurable weighting system for hybrid search results.

Manages vector and keyword search result combination with learnable weights.
Supports both static configuration and adaptive learning from relevancy ratings.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.utils.logger import create_module_logger

get_logger, audit = create_module_logger("rag")


@dataclass
class HybridSearchWeights:
    """Configuration for hybrid search result weighting."""

    vector_weight: float = 0.6  # Weight for vector search results (0.0-1.0)
    keyword_weight: float = 0.4  # Weight for keyword search results (0.0-1.0)

    # Reranking strategy: "sum" (weighted sum), "rank_fusion" (RRF), "top_k" (take top from each)
    combination_strategy: str = "sum"

    # Whether to normalise weights to sum to 1.0
    normalise_weights: bool = True

    def __post_init__(self):
        """Validate and normalise weights after initialisation."""
        if self.vector_weight < 0 or self.keyword_weight < 0:
            raise ValueError("Weights must be non-negative")

        if self.normalise_weights:
            total = self.vector_weight + self.keyword_weight
            if total > 0:
                self.vector_weight /= total
                self.keyword_weight /= total

        if self.combination_strategy not in ["sum", "rank_fusion", "top_k"]:
            raise ValueError(f"Unknown combination strategy: {self.combination_strategy}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialisation."""
        return {
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight,
            "combination_strategy": self.combination_strategy,
            "normalise_weights": self.normalise_weights,
            "timestamp": datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridSearchWeights":
        """Create from dictionary."""
        return cls(
            vector_weight=data.get("vector_weight", 0.6),
            keyword_weight=data.get("keyword_weight", 0.4),
            combination_strategy=data.get("combination_strategy", "sum"),
            normalise_weights=data.get("normalise_weights", True),
        )


class HybridSearchWeightManager:
    """Manages hybrid search weights with persistence and adaptation."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialise weight manager.

        Args:
            config_path: Path to weights config file (default: rag_data/hybrid_search_weights.json)
        """
        if config_path is None:
            from scripts.rag.rag_config import RAGConfig

            config = RAGConfig()
            config_path = Path(config.rag_data_path) / "hybrid_search_weights.json"

        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

        # Load or create default weights
        self.weights = self._load_weights()

    def _load_weights(self) -> HybridSearchWeights:
        """Load weights from file or create defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)
                    weights = HybridSearchWeights.from_dict(data)
                    self.logger.debug(
                        f"Loaded hybrid search weights: vector={weights.vector_weight:.2f}, keyword={weights.keyword_weight:.2f}"
                    )
                    return weights
            except Exception as e:
                self.logger.warning(f"Failed to load weights: {e}, using defaults")

        # Return defaults
        return HybridSearchWeights()

    def save_weights(self, weights: Optional[HybridSearchWeights] = None) -> bool:
        """Save weights to file.

        Args:
            weights: Weights to save (uses current if None)

        Returns:
            True if successful, False otherwise
        """
        try:
            weights_to_save = weights or self.weights
            with open(self.config_path, "w") as f:
                json.dump(weights_to_save.to_dict(), f, indent=2)
            self.logger.debug(f"Saved hybrid search weights to {self.config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save weights: {e}")
            return False

    def update_weights(
        self,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
        strategy: Optional[str] = None,
    ) -> bool:
        """Update and persist weights.

        Args:
            vector_weight: New vector search weight
            keyword_weight: New keyword search weight
            strategy: New combination strategy

        Returns:
            True if successful
        """
        try:
            if vector_weight is not None:
                self.weights.vector_weight = vector_weight
            if keyword_weight is not None:
                self.weights.keyword_weight = keyword_weight
            if strategy is not None:
                self.weights.combination_strategy = strategy

            # Reinitialise to apply normalisation
            updated = HybridSearchWeights(
                vector_weight=self.weights.vector_weight,
                keyword_weight=self.weights.keyword_weight,
                combination_strategy=self.weights.combination_strategy,
                normalise_weights=self.weights.normalise_weights,
            )

            self.weights = updated
            return self.save_weights()
        except Exception as e:
            self.logger.error(f"Failed to update weights: {e}")
            return False

    def combine_results(
        self,
        vector_chunks: List[str],
        vector_metadata: List[Dict],
        vector_scores: List[float],
        keyword_chunks: List[str],
        keyword_metadata: List[Dict],
        keyword_scores: List[float],
        k: int = 5,
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """Combine vector and keyword results using configured weights.

        Args:
            vector_chunks: List of chunks from vector search
            vector_metadata: Metadata for vector chunks
            vector_scores: Normalised scores (0-1) for vector chunks
            keyword_chunks: List of chunks from keyword search
            keyword_metadata: Metadata for keyword chunks
            keyword_scores: Normalised scores (0-1) for keyword chunks
            k: Number of final results to return

        Returns:
            Tuple of (combined_chunks, combined_metadata, combined_scores)
        """
        if self.weights.combination_strategy == "sum":
            return self._combine_weighted_sum(
                vector_chunks,
                vector_metadata,
                vector_scores,
                keyword_chunks,
                keyword_metadata,
                keyword_scores,
                k,
            )
        elif self.weights.combination_strategy == "rank_fusion":
            return self._combine_rank_fusion(
                vector_chunks,
                vector_metadata,
                vector_scores,
                keyword_chunks,
                keyword_metadata,
                keyword_scores,
                k,
            )
        else:  # top_k
            return self._combine_top_k(
                vector_chunks,
                vector_metadata,
                vector_scores,
                keyword_chunks,
                keyword_metadata,
                keyword_scores,
                k,
            )

    def _combine_weighted_sum(
        self,
        vector_chunks: List[str],
        vector_metadata: List[Dict],
        vector_scores: List[float],
        keyword_chunks: List[str],
        keyword_metadata: List[Dict],
        keyword_scores: List[float],
        k: int,
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """Combine using weighted sum strategy."""
        combined: Dict[str, Tuple[str, Dict, float]] = {}

        # Add vector results
        for chunk, meta, score in zip(vector_chunks, vector_metadata, vector_scores):
            weighted_score = score * self.weights.vector_weight
            combined[chunk] = (chunk, {**meta, "retrieval_method": "vector"}, weighted_score)

        # Add/merge keyword results
        for chunk, meta, score in zip(keyword_chunks, keyword_metadata, keyword_scores):
            weighted_score = score * self.weights.keyword_weight

            if chunk in combined:
                # Chunk found in both: sum scores
                _, existing_meta, existing_score = combined[chunk]
                combined[chunk] = (
                    chunk,
                    {**existing_meta, "retrieval_method": "hybrid"},
                    existing_score + weighted_score,
                )
            else:
                # New chunk from keyword search
                combined[chunk] = (chunk, {**meta, "retrieval_method": "keyword"}, weighted_score)

        # Sort by combined score descending
        sorted_results = sorted(combined.values(), key=lambda x: x[2], reverse=True)

        # Extract top k
        if sorted_results:
            chunks, metadata, scores = zip(*sorted_results[:k])
            return list(chunks), list(metadata), list(scores)

        return [], [], []

    def _combine_rank_fusion(
        self,
        vector_chunks: List[str],
        vector_metadata: List[Dict],
        vector_scores: List[float],
        keyword_chunks: List[str],
        keyword_metadata: List[Dict],
        keyword_scores: List[float],
        k: int,
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """Combine using Reciprocal Rank Fusion (RRF).

        RRF formula: score = 1 / (60 + rank)
        Gives equal importance to ranking position vs absolute scores.
        """
        combined: Dict[str, Tuple[str, Dict, float]] = {}

        # Add vector results with RRF scores
        for rank, (chunk, meta, _) in enumerate(zip(vector_chunks, vector_metadata, vector_scores)):
            rrf_score = (1.0 / (60 + rank + 1)) * self.weights.vector_weight
            combined[chunk] = (chunk, {**meta, "retrieval_method": "vector"}, rrf_score)

        # Add/merge keyword results with RRF scores
        for rank, (chunk, meta, _) in enumerate(
            zip(keyword_chunks, keyword_metadata, keyword_scores)
        ):
            rrf_score = (1.0 / (60 + rank + 1)) * self.weights.keyword_weight

            if chunk in combined:
                _, existing_meta, existing_score = combined[chunk]
                combined[chunk] = (
                    chunk,
                    {**existing_meta, "retrieval_method": "hybrid"},
                    existing_score + rrf_score,
                )
            else:
                combined[chunk] = (chunk, {**meta, "retrieval_method": "keyword"}, rrf_score)

        # Sort by RRF score descending
        sorted_results = sorted(combined.values(), key=lambda x: x[2], reverse=True)

        if sorted_results:
            chunks, metadata, scores = zip(*sorted_results[:k])
            return list(chunks), list(metadata), list(scores)

        return [], [], []

    def _combine_top_k(
        self,
        vector_chunks: List[str],
        vector_metadata: List[Dict],
        vector_scores: List[float],
        keyword_chunks: List[str],
        keyword_metadata: List[Dict],
        keyword_scores: List[float],
        k: int,
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """Combine using top-k from each method.

        Takes proportional top results from vector and keyword based on weights.
        Example: vector_weight=0.6, keyword_weight=0.4 with k=5 takes 3 from vector, 2 from keyword.
        """
        combined_chunks = []
        combined_metadata = []
        combined_scores = []
        seen = set()

        # Calculate how many to take from each method
        vector_k = max(1, int(k * self.weights.vector_weight))
        keyword_k = max(1, int(k * self.weights.keyword_weight))

        # Add vector results
        for chunk, meta, score in zip(
            vector_chunks[:vector_k], vector_metadata[:vector_k], vector_scores[:vector_k]
        ):
            combined_chunks.append(chunk)
            combined_metadata.append({**meta, "retrieval_method": "vector"})
            combined_scores.append(score * self.weights.vector_weight)
            seen.add(chunk)

        # Add keyword results (avoiding duplicates)
        for chunk, meta, score in zip(keyword_chunks, keyword_metadata, keyword_scores):
            if chunk not in seen and len(combined_chunks) < k:
                combined_chunks.append(chunk)
                combined_metadata.append({**meta, "retrieval_method": "keyword"})
                combined_scores.append(score * self.weights.keyword_weight)
                seen.add(chunk)

        return combined_chunks, combined_metadata, combined_scores


# Global instance
_weight_manager: Optional[HybridSearchWeightManager] = None


def get_weight_manager(config_path: Optional[Path] = None) -> HybridSearchWeightManager:
    """Get or create global weight manager instance."""
    global _weight_manager
    if _weight_manager is None:
        _weight_manager = HybridSearchWeightManager(config_path)
    return _weight_manager
