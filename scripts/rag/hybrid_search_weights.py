"""Configurable weighting system for hybrid search results.

Manages vector and keyword search result combination with learnable weights.
Supports both static configuration and adaptive learning from relevancy ratings.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import floor
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
    graph_weight: float = 0.15  # Weight for graph-proximity results (0.0-1.0)

    def __post_init__(self):
        """Validate and normalise weights after initialisation."""
        if self.vector_weight < 0 or self.keyword_weight < 0 or self.graph_weight < 0:
            raise ValueError("Weights must be non-negative")

        if self.normalise_weights:
            total = self.vector_weight + self.keyword_weight + self.graph_weight
            if total > 0:
                self.vector_weight /= total
                self.keyword_weight /= total
                self.graph_weight /= total

        if self.combination_strategy not in ["sum", "rank_fusion", "top_k"]:
            raise ValueError(f"Unknown combination strategy: {self.combination_strategy}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialisation."""
        return {
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight,
            "graph_weight": self.graph_weight,
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
            graph_weight=data.get("graph_weight", 0.15),
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
        graph_weight: Optional[float] = None,
    ) -> bool:
        """Update and persist weights.

        Args:
            vector_weight: New vector search weight
            keyword_weight: New keyword search weight
            strategy: New combination strategy
            graph_weight: New graph search weight

        Returns:
            True if successful
        """
        try:
            if vector_weight is not None:
                self.weights.vector_weight = vector_weight
            if keyword_weight is not None:
                self.weights.keyword_weight = keyword_weight
            if graph_weight is not None:
                self.weights.graph_weight = graph_weight
            if strategy is not None:
                self.weights.combination_strategy = strategy

            # Reinitialise to apply normalisation
            updated = HybridSearchWeights(
                vector_weight=self.weights.vector_weight,
                keyword_weight=self.weights.keyword_weight,
                graph_weight=self.weights.graph_weight,
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
        graph_chunks: Optional[List[str]] = None,
        graph_metadata: Optional[List[Dict]] = None,
        graph_scores: Optional[List[float]] = None,
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
            graph_chunks: List of chunks from graph search
            graph_metadata: Metadata for graph chunks
            graph_scores: Normalised scores (0-1) for graph chunks

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
                graph_chunks or [],
                graph_metadata or [],
                graph_scores or [],
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
                graph_chunks or [],
                graph_metadata or [],
                graph_scores or [],
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
                graph_chunks or [],
                graph_metadata or [],
                graph_scores or [],
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
        graph_chunks: List[str],
        graph_metadata: List[Dict],
        graph_scores: List[float],
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """Combine using weighted sum strategy.
        Args:
            vector_chunks (List[str]): List of chunks retrieved via vector search.
            vector_metadata (List[Dict]): Corresponding metadata for vector chunks.
            vector_scores (List[float]): Normalised scores (0-1) for vector chunks.
            keyword_chunks (List[str]): List of chunks retrieved via keyword search.
            keyword_metadata (List[Dict]): Corresponding metadata for keyword chunks.
            keyword_scores (List[float]): Normalised scores (0-1) for keyword chunks.
            k (int): Maximum number of chunks to return.
            graph_chunks (List[str]): List of chunks retrieved from the graph.
            graph_metadata (List[Dict]): Corresponding metadata for graph chunks.
            graph_scores (List[float]): Normalised scores (0-1) for graph chunks.

        Returns:
            Tuple[List[str], List[Dict], List[float]]: Combined chunks, their metadata, and combined scores.
        """
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

        for chunk, meta, score in zip(graph_chunks, graph_metadata, graph_scores):
            weighted_score = score * self.weights.graph_weight
            graph_meta = {**meta, "graph_proximity_score": score}
            if chunk in combined:
                _, existing_meta, existing_score = combined[chunk]
                combined[chunk] = (
                    chunk,
                    {**existing_meta, **graph_meta, "retrieval_method": "hybrid"},
                    existing_score + weighted_score,
                )
            else:
                combined[chunk] = (
                    chunk,
                    {**graph_meta, "retrieval_method": "thesis_graph"},
                    weighted_score,
                )

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
        graph_chunks: List[str],
        graph_metadata: List[Dict],
        graph_scores: List[float],
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """Combine using Reciprocal Rank Fusion (RRF).

        RRF formula: score = 1 / (60 + rank)
        Gives equal importance to ranking position vs absolute scores.

        Args:
            vector_chunks (List[str]): List of chunks retrieved via vector search.
            vector_metadata (List[Dict]): Corresponding metadata for vector chunks.
            vector_scores (List[float]): Normalised scores (0-1) for vector chunks.
            keyword_chunks (List[str]): List of chunks retrieved via keyword search.
            keyword_metadata (List[Dict]): Corresponding metadata for keyword chunks.
            keyword_scores (List[float]): Normalised scores (0-1) for keyword chunks.
            k (int): Maximum number of chunks to return.
            graph_chunks (List[str]): List of chunks retrieved from the graph.
            graph_metadata (List[Dict]): Corresponding metadata for graph chunks.
            graph_scores (List[float]): Normalised scores (0-1) for graph chunks.

        Returns:
            Tuple[List[str], List[Dict], List[float]]: Combined chunks, their metadata, and combined scores.
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

        for rank, (chunk, meta, _) in enumerate(zip(graph_chunks, graph_metadata, graph_scores)):
            rrf_score = (1.0 / (60 + rank + 1)) * self.weights.graph_weight
            graph_meta = {**meta, "graph_proximity_score": 1.0 / (rank + 1)}
            if chunk in combined:
                _, existing_meta, existing_score = combined[chunk]
                combined[chunk] = (
                    chunk,
                    {**existing_meta, **graph_meta, "retrieval_method": "hybrid"},
                    existing_score + rrf_score,
                )
            else:
                combined[chunk] = (
                    chunk,
                    {**graph_meta, "retrieval_method": "thesis_graph"},
                    rrf_score,
                )

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
        graph_chunks: List[str],
        graph_metadata: List[Dict],
        graph_scores: List[float],
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """Combine using top-k from each method.

        Takes proportional top results from each available source based on weights.

        Args:
            vector_chunks (List[str]): List of chunks retrieved via vector search.
            vector_metadata (List[Dict]): Corresponding metadata for vector chunks.
            vector_scores (List[float]): Normalised scores (0-1) for vector chunks.
            keyword_chunks (List[str]): List of chunks retrieved via keyword search.
            keyword_metadata (List[Dict]): Corresponding metadata for keyword chunks.
            keyword_scores (List[float]): Normalised scores (0-1) for keyword chunks.
            k (int): Maximum number of chunks to return.
            graph_chunks (List[str]): List of chunks retrieved from the graph.
            graph_metadata (List[Dict]): Corresponding metadata for graph chunks.
            graph_scores (List[float]): Normalised scores (0-1) for graph chunks.

        Returns:
            Tuple[List[str], List[Dict], List[float]]: Combined chunks, their metadata, and combined scores.
        """
        sources = [
            (vector_chunks, vector_metadata, vector_scores, self.weights.vector_weight, "vector"),
            (
                keyword_chunks,
                keyword_metadata,
                keyword_scores,
                self.weights.keyword_weight,
                "keyword",
            ),
            (graph_chunks, graph_metadata, graph_scores, self.weights.graph_weight, "thesis_graph"),
        ]
        available = [
            index
            for index, (chunks, _, _, weight, _) in enumerate(sources)
            if chunks and weight > 0
        ]
        combined_chunks: List[str] = []
        combined_metadata: List[Dict] = []
        combined_scores: List[float] = []
        selected_counts = [0] * len(sources)

        if available:
            total_weight = sum(sources[index][3] for index in available)
            targets = [
                (k * sources[index][3] / total_weight) if index in available else 0.0
                for index in range(len(sources))
            ]
            for index in available:
                selected_counts[index] = min(len(sources[index][0]), floor(targets[index]))

            slots_remaining = min(k, sum(len(sources[index][0]) for index in available)) - sum(
                selected_counts
            )
            while slots_remaining > 0:
                eligible = [
                    index for index in available if selected_counts[index] < len(sources[index][0])
                ]
                if not eligible:
                    break
                selected = max(
                    eligible,
                    key=lambda index: (targets[index] - selected_counts[index], -index),
                )
                selected_counts[selected] += 1
                slots_remaining -= 1

        selected_by_chunk: Dict[str, int] = {}
        for index, (chunks, metadata, scores, weight, method) in enumerate(sources):
            for chunk, meta, score in zip(
                chunks[: selected_counts[index]],
                metadata[: selected_counts[index]],
                scores[: selected_counts[index]],
            ):
                graph_score = score if method == "thesis_graph" else None
                if chunk in selected_by_chunk:
                    existing_index = selected_by_chunk[chunk]
                    combined_metadata[existing_index]["retrieval_method"] = "hybrid"
                    if graph_score is not None:
                        combined_metadata[existing_index]["graph_proximity_score"] = graph_score
                        combined_scores[existing_index] += graph_score * weight
                    continue

                result_metadata = {**meta, "retrieval_method": method}
                if graph_score is not None:
                    result_metadata["graph_proximity_score"] = graph_score
                selected_by_chunk[chunk] = len(combined_chunks)
                combined_chunks.append(chunk)
                combined_metadata.append(result_metadata)
                combined_scores.append(score * weight)

        if len(combined_chunks) < k:
            remaining_candidates = []
            for chunks, metadata, scores, weight, method in sources:
                for chunk, meta, score in zip(chunks, metadata, scores):
                    if chunk not in selected_by_chunk:
                        remaining_candidates.append((score * weight, chunk, meta, score, method))
            remaining_candidates.sort(key=lambda item: item[0], reverse=True)
            for weighted_score, chunk, meta, score, method in remaining_candidates:
                if len(combined_chunks) >= k:
                    break
                if chunk in selected_by_chunk:
                    continue
                result_metadata = {**meta, "retrieval_method": method}
                if method == "thesis_graph":
                    result_metadata["graph_proximity_score"] = score
                selected_by_chunk[chunk] = len(combined_chunks)
                combined_chunks.append(chunk)
                combined_metadata.append(result_metadata)
                combined_scores.append(weighted_score)

        return combined_chunks, combined_metadata, combined_scores


# Global instance
_weight_manager: Optional[HybridSearchWeightManager] = None


def get_weight_manager(config_path: Optional[Path] = None) -> HybridSearchWeightManager:
    """Get or create global weight manager instance."""
    global _weight_manager
    if _weight_manager is None:
        _weight_manager = HybridSearchWeightManager(config_path)
    return _weight_manager
