"""Adaptive weight learning for hybrid search based on benchmarking metrics.

Learns optimal vector/keyword balance by analysing user relevancy ratings
from benchmarking. Uses linear regression to predict optimal weights from
user feedback patterns.

Features:
- Read relevancy ratings from benchmarks
- Linear regression weight optimisation
- Automatic weight tuning suggestions
- Performance tracking over time
- Confidence scoring for recommendations
- A/B testing support for weight variants

Weight learning process:
1. Collect user relevancy ratings from benchmarks
2. Extract features (query type, term distribution, etc.)
3. Fit linear regression model: relevancy = w_vector * f_vector + w_keyword * f_keyword
4. Recommend optimal weights
5. Track performance of recommendations
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from scripts.utils.logger import create_module_logger

get_logger, audit = create_module_logger("rag")
logger = get_logger()


@dataclass
class WeightSample:
    """Training sample for weight learning."""

    vector_weight: float
    keyword_weight: float
    relevancy_score: float  # 0.0-1.0, user rating
    query_type: str  # 'technical', 'natural', 'mixed'
    query_length: int
    result_count: int
    embedding_sim_avg: float  # Average embedding similarity
    bm25_score_avg: float  # Average BM25 score
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    data_version: str = ""  # Version/timestamp of ChromaDB data when sample was created


@dataclass
class WeightRecommendation:
    """Recommended weight configuration."""

    vector_weight: float
    keyword_weight: float
    confidence: float  # 0.0-1.0
    reasoning: str
    expected_improvement: float  # Expected improvement over baseline
    samples_used: int
    model_r2: float  # Model fit quality
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class AdaptiveWeightLearner:
    """Learns optimal vector/keyword weights from user feedback.

    Attributes:
        learning_path (Path): Path to store samples and model
        min_samples (int): Minimum samples to fit model
        confidence_threshold (float): Minimum confidence for recommendations
        sample_decay_days (int): Days after which samples start decaying in weight
        stale_sample_threshold_days (int): Days after which samples are considered stale
    """

    def __init__(
        self,
        learning_path: Optional[Path] = None,
        min_samples: int = 20,
        confidence_threshold: float = 0.6,
        sample_decay_days: int = 30,
        stale_sample_threshold_days: int = 90,
    ):
        """Initialise adaptive weight learner.

        Args:
            learning_path: Path to store learning data
            min_samples: Minimum samples before making recommendations
            confidence_threshold: Minimum confidence threshold (0.0-1.0)
            sample_decay_days: Days after which sample weights start decaying (default: 30)
            stale_sample_threshold_days: Days after data update to discard samples (default: 90)
        """
        if learning_path is None:
            from scripts.rag.rag_config import RAGConfig

            config = RAGConfig()
            learning_path = Path(config.rag_data_path) / "learning"

        self.learning_path = Path(learning_path)
        self.learning_path.mkdir(parents=True, exist_ok=True)

        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
        self.sample_decay_days = sample_decay_days
        self.stale_sample_threshold_days = stale_sample_threshold_days

        self.samples: List[WeightSample] = []
        self.model: Optional[LinearRegression] = None
        self.last_recommendation: Optional[WeightRecommendation] = None
        self.chromadb_last_modified: Optional[str] = None

        self._load_samples()
        self._detect_chromadb_updates()
        self._fit_model()

    def record_sample(
        self,
        vector_weight: float,
        keyword_weight: float,
        relevancy_score: float,
        query_type: str,
        query_length: int,
        result_count: int,
        embedding_sim_avg: float,
        bm25_score_avg: float,
        data_version: Optional[str] = None,
    ) -> None:
        """Record a weight sample from user feedback.

        Args:
            vector_weight: Weight used for vector search (0.0-1.0)
            keyword_weight: Weight used for keyword search (0.0-1.0)
            relevancy_score: User relevancy rating (0.0-1.0)
            query_type: Query type classification
            query_length: Query length in tokens
            result_count: Number of results returned
            embedding_sim_avg: Average embedding similarity score
            bm25_score_avg: Average BM25 score
            data_version: Optional data version/timestamp (auto-detected if None)
        """
        # Get current data version if not provided
        if data_version is None:
            data_version = self._get_current_data_version()

        sample = WeightSample(
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            relevancy_score=relevancy_score,
            query_type=query_type,
            query_length=query_length,
            result_count=result_count,
            embedding_sim_avg=embedding_sim_avg,
            bm25_score_avg=bm25_score_avg,
            data_version=data_version,
        )

        self.samples.append(sample)
        self._save_samples()

        # Re-fit model if we have enough samples
        if len(self.samples) >= self.min_samples:
            self._fit_model()

    def get_recommendation(
        self,
        query_type: str = "mixed",
        current_vector_weight: float = 0.6,
        current_keyword_weight: float = 0.4,
    ) -> Optional[WeightRecommendation]:
        """Get recommended weights based on learned model.

        Args:
            query_type: Type of query ('technical', 'natural', 'mixed')
            current_vector_weight: Current vector weight (for comparison)
            current_keyword_weight: Current keyword weight (for comparison)

        Returns:
            WeightRecommendation if model is fit and confident, else None
        """
        if self.model is None or len(self.samples) < self.min_samples:
            logger.info(
                f"Not enough samples ({len(self.samples)}/{self.min_samples}) for recommendation"
            )
            return None

        # Test different weight combinations
        best_score = -np.inf
        best_weights = (current_vector_weight, current_keyword_weight)

        for vector_w in np.arange(0.1, 1.0, 0.1):
            keyword_w = 1.0 - vector_w

            # Predict average relevancy for this weight combination
            # Average across recent samples
            predicted_scores = []
            for sample in self.samples[-50:]:  # Use recent 50 samples
                features = self._extract_features(
                    vector_w,
                    keyword_w,
                    sample.query_type,
                    sample.query_length,
                    sample.embedding_sim_avg,
                    sample.bm25_score_avg,
                )
                pred = self.model.predict([features])[0]
                predicted_scores.append(pred)

            avg_predicted = np.mean(predicted_scores) if predicted_scores else 0.0

            if avg_predicted > best_score:
                best_score = avg_predicted
                best_weights = (vector_w, keyword_w)

        # Calculate confidence (model R²) and expected improvement
        model_r2 = self.model.score(self._get_X(), self._get_y())
        confidence = min(1.0, max(0.0, model_r2))

        if confidence < self.confidence_threshold:
            logger.debug(
                f"Model confidence {confidence:.2f} below threshold {self.confidence_threshold}"
            )
            return None

        # Expected improvement over current weights
        current_features = self._extract_features(
            current_vector_weight,
            current_keyword_weight,
            query_type,
            len(query_type) * 3,
            0.6,
            0.5,  # Default scores
        )
        current_pred = self.model.predict([current_features])[0]

        new_features = self._extract_features(
            best_weights[0],
            best_weights[1],
            query_type,
            len(query_type) * 3,
            0.6,
            0.5,
        )
        new_pred = self.model.predict([new_features])[0]

        expected_improvement = max(0.0, new_pred - current_pred)

        reasoning = self._generate_reasoning(
            best_weights[0], best_weights[1], query_type, confidence, expected_improvement
        )

        recommendation = WeightRecommendation(
            vector_weight=best_weights[0],
            keyword_weight=best_weights[1],
            confidence=confidence,
            reasoning=reasoning,
            expected_improvement=expected_improvement,
            samples_used=len(self.samples),
            model_r2=model_r2,
        )

        self.last_recommendation = recommendation
        self._save_recommendation()

        audit(
            "weight_recommendation",
            {
                "vector_weight": best_weights[0],
                "keyword_weight": best_weights[1],
                "confidence": confidence,
                "improvement": expected_improvement,
            },
        )

        return recommendation

    def _extract_features(
        self,
        vector_weight: float,
        keyword_weight: float,
        query_type: str,
        query_length: int,
        embedding_sim: float,
        bm25_score: float,
    ) -> List[float]:
        """Extract features for model prediction."""
        query_type_encoded = {
            "technical": 1.0,
            "natural": 0.5,
            "mixed": 0.0,
        }.get(query_type, 0.0)

        return [
            vector_weight,
            keyword_weight,
            query_type_encoded,
            query_length / 100.0,  # Normalise
            embedding_sim,
            bm25_score,
        ]

    def _get_X(self) -> np.ndarray:
        """Get feature matrix from samples."""
        X = []
        for sample in self.samples:
            features = self._extract_features(
                sample.vector_weight,
                sample.keyword_weight,
                sample.query_type,
                sample.query_length,
                sample.embedding_sim_avg,
                sample.bm25_score_avg,
            )
            X.append(features)
        return np.array(X)

    def _get_y(self) -> np.ndarray:
        """Get target vector from samples."""
        return np.array([s.relevancy_score for s in self.samples])

    def _get_sample_weights(self) -> np.ndarray:
        """Calculate sample weights based on data freshness.

        Samples collected before ChromaDB updates receive reduced weight.
        Applies exponential decay based on time since data update.

        Returns:
            Array of sample weights (0.0-1.0) aligned with samples.
        """
        weights = []
        current_time = datetime.now()

        for sample in self.samples:
            weight = 1.0

            # Parse sample creation time
            try:
                sample_time = datetime.fromisoformat(sample.created_at.replace("Z", "+00:00"))
            except Exception:
                # If parsing fails, use full weight
                weights.append(weight)
                continue

            # Check if sample predates ChromaDB update
            if self.chromadb_last_modified:
                try:
                    chromadb_time = datetime.fromisoformat(
                        self.chromadb_last_modified.replace("Z", "+00:00")
                    )

                    # If sample is from before last data update, apply decay
                    if sample_time < chromadb_time:
                        days_since_update = (current_time - chromadb_time).days

                        # Discard samples if data was updated too long ago
                        if days_since_update > self.stale_sample_threshold_days:
                            weight = 0.0
                        # Apply exponential decay for samples before update
                        elif days_since_update > 0:
                            # Decay factor: e^(-days/decay_period)
                            decay_factor = np.exp(-days_since_update / self.sample_decay_days)
                            weight = max(0.1, decay_factor)  # Minimum 10% weight

                except Exception as e:
                    logger.debug(f"Failed to parse chromadb_last_modified: {e}")

            weights.append(weight)

        weights_array = np.array(weights)

        # Log if significant sample filtering occurred
        active_samples = np.sum(weights_array > 0.1)
        if active_samples < len(weights_array):
            logger.info(
                f"Sample weighting: {active_samples}/{len(weights_array)} samples have >10% weight "
                f"(mean weight: {np.mean(weights_array):.2f})"
            )

        return weights_array

    def _get_current_data_version(self) -> str:
        """Get current ChromaDB data version/timestamp.

        Returns timestamp of most recent document in the collection,
        or current time if detection fails.

        Returns:
            ISO format timestamp string
        """
        try:
            from scripts.rag.rag_config import RAGConfig
            from scripts.utils.db_factory import get_default_vector_path, get_vector_client

            config = RAGConfig()
            PersistentClient, USING_SQLITE = get_vector_client(prefer="chroma")
            chroma_path = get_default_vector_path(Path(config.rag_data_path), USING_SQLITE)

            # Check if collection exists
            client = PersistentClient(path=chroma_path)
            try:
                collection = client.get_collection(config.chunk_collection_name)

                # Get one document to check metadata
                result = collection.get(limit=1, include=["metadatas"])
                if result and result.get("metadatas"):
                    # Use ingestion timestamp if available
                    metadata = result["metadatas"][0]
                    if "ingestion_timestamp" in metadata:
                        return metadata["ingestion_timestamp"]
                    if "created_at" in metadata:
                        return metadata["created_at"]

            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Could not detect data version: {e}")

        # Fallback to current time
        return datetime.now().isoformat()

    def _detect_chromadb_updates(self) -> None:
        """Detect when ChromaDB was last updated.

        Stores the most recent modification timestamp to compare against
        sample creation times for data freshness assessment.
        """
        try:
            # Try to get last modified time from ChromaDB metadata
            version = self._get_current_data_version()

            # Load previously stored version
            version_file = self.learning_path / "chromadb_version.txt"
            if version_file.exists():
                with open(version_file, "r") as f:
                    stored_version = f.read().strip()

                # If version changed, we have an update
                if stored_version != version:
                    self.chromadb_last_modified = version
                    logger.info(f"ChromaDB data update detected: {stored_version} → {version}")

                    # Save new version
                    with open(version_file, "w") as f:
                        f.write(version)
                else:
                    self.chromadb_last_modified = stored_version
            else:
                # First time - store current version
                self.chromadb_last_modified = version
                with open(version_file, "w") as f:
                    f.write(version)

        except Exception as e:
            logger.debug(f"Could not detect ChromaDB updates: {e}")
            self.chromadb_last_modified = None

    def _fit_model(self) -> None:
        """Fit linear regression model with sample weighting."""
        if len(self.samples) < self.min_samples:
            return

        try:
            X = self._get_X()
            y = self._get_y()
            sample_weights = self._get_sample_weights()

            # Filter out zero-weight samples
            non_zero_mask = sample_weights > 0
            if np.sum(non_zero_mask) < self.min_samples:
                logger.warning(
                    f"Insufficient non-stale samples ({np.sum(non_zero_mask)}/{self.min_samples}) "
                    f"after data freshness filtering"
                )
                return

            X_filtered = X[non_zero_mask]
            y_filtered = y[non_zero_mask]
            weights_filtered = sample_weights[non_zero_mask]

            self.model = LinearRegression()
            self.model.fit(X_filtered, y_filtered, sample_weight=weights_filtered)

            r2 = self.model.score(X_filtered, y_filtered, sample_weight=weights_filtered)
            active_samples = np.sum(non_zero_mask)
            logger.info(
                f"Weight learning model fitted (R²={r2:.3f}, "
                f"active_samples={active_samples}/{len(self.samples)}, "
                f"mean_weight={np.mean(weights_filtered):.2f})"
            )

        except Exception as e:
            logger.error(f"Failed to fit weight learning model: {e}")
            self.model = None

    def _generate_reasoning(
        self,
        vector_weight: float,
        keyword_weight: float,
        query_type: str,
        confidence: float,
        improvement: float,
    ) -> str:
        """Generate human-readable reasoning."""
        reasons = []

        if vector_weight > keyword_weight:
            reasons.append(f"Vector search dominates ({vector_weight:.1%} weight)")
        else:
            reasons.append(f"Keyword search dominates ({keyword_weight:.1%} weight)")

        if query_type == "technical":
            reasons.append("optimised for technical queries")
        elif query_type == "natural":
            reasons.append("optimised for natural language")
        else:
            reasons.append("balanced for mixed query types")

        reasons.append(f"confidence: {confidence:.0%}")

        if improvement > 0.05:
            reasons.append(f"expected {improvement:.0%} improvement")

        return ", ".join(reasons)

    def _save_samples(self) -> None:
        """Save samples to disk."""
        try:
            path = self.learning_path / "weight_samples.json"
            data = [asdict(s) for s in self.samples]
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save samples: {e}")

    def _load_samples(self) -> None:
        """Load samples from disk."""
        try:
            path = self.learning_path / "weight_samples.json"
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                    self.samples = [WeightSample(**item) for item in data]
        except Exception as e:
            logger.error(f"Failed to load samples: {e}")

    def _save_recommendation(self) -> None:
        """Save last recommendation."""
        try:
            path = self.learning_path / "last_recommendation.json"
            if self.last_recommendation:
                with open(path, "w") as f:
                    json.dump(asdict(self.last_recommendation), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save recommendation: {e}")

    def get_statistics(self) -> Dict:
        """Get learning statistics including sample freshness metrics."""
        stats = {
            "samples_collected": len(self.samples),
            "model_fitted": self.model is not None,
            "last_recommendation": (
                asdict(self.last_recommendation) if self.last_recommendation else None
            ),
            "confidence_threshold": self.confidence_threshold,
            "chromadb_last_modified": self.chromadb_last_modified,
        }

        # Add model quality metrics if fitted
        if self.model is not None:
            try:
                sample_weights = self._get_sample_weights()
                non_zero_mask = sample_weights > 0
                X = self._get_X()[non_zero_mask]
                y = self._get_y()[non_zero_mask]
                weights = sample_weights[non_zero_mask]

                if len(X) > 0:
                    stats["model_r2"] = self.model.score(X, y, sample_weight=weights)
                    stats["active_samples"] = int(np.sum(non_zero_mask))
                    stats["stale_samples"] = len(self.samples) - int(np.sum(non_zero_mask))
                    stats["mean_sample_weight"] = float(np.mean(sample_weights))
                else:
                    stats["model_r2"] = None
                    stats["active_samples"] = 0
                    stats["stale_samples"] = len(self.samples)
            except Exception as e:
                logger.debug(f"Failed to compute statistics: {e}")
                stats["model_r2"] = None
        else:
            stats["model_r2"] = None

        return stats

    def apply_recommendation_to_weights(self) -> bool:
        """Apply last recommendation to HybridSearchWeightManager.

        Uses the latest recommendation to update the hybrid search weights
        if recommendation exists and is confident enough.

        Returns:
            True if recommendation was applied, False otherwise.
        """
        if self.last_recommendation is None:
            logger.debug("No recommendation to apply")
            return False

        try:
            from scripts.rag.hybrid_search_weights import get_weight_manager

            weight_manager = get_weight_manager()
            if weight_manager is None:
                logger.warning("Weight manager not available, cannot apply recommendation")
                return False

            updated = weight_manager.update_weights(
                vector_weight=self.last_recommendation.vector_weight,
                keyword_weight=self.last_recommendation.keyword_weight,
            )

            if updated:
                logger.info(
                    f"Applied adaptive recommendation: vector={self.last_recommendation.vector_weight:.2f}, "
                    f"keyword={self.last_recommendation.keyword_weight:.2f} "
                    f"(confidence={self.last_recommendation.confidence:.0%})"
                )
                audit(
                    "adaptive_weights_applied",
                    {
                        "vector_weight": self.last_recommendation.vector_weight,
                        "keyword_weight": self.last_recommendation.keyword_weight,
                        "confidence": self.last_recommendation.confidence,
                        "expected_improvement": self.last_recommendation.expected_improvement,
                    },
                )
            return updated

        except Exception as e:
            logger.error(f"Failed to apply recommendation: {e}")
            return False


# Global instance
_weight_learner: Optional[AdaptiveWeightLearner] = None


def get_adaptive_weight_learner() -> AdaptiveWeightLearner:
    """Get or create global adaptive weight learner instance."""
    global _weight_learner
    if _weight_learner is None:
        _weight_learner = AdaptiveWeightLearner()
    return _weight_learner
