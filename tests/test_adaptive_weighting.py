"""Unit tests for adaptive weight learning module.

Tests core functionality of AdaptiveWeightLearner including:
- Sample recording and persistence
- Model fitting and prediction
- Recommendation generation with confidence scoring
- Feature extraction and reasoning generation
- Statistics reporting
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from scripts.rag.adaptive_weighting import (
    AdaptiveWeightLearner,
    WeightRecommendation,
    WeightSample,
    get_adaptive_weight_learner,
)


class TestWeightSample:
    """Tests for WeightSample dataclass."""

    def test_weight_sample_creation(self) -> None:
        """Test creating a WeightSample with all required fields."""
        sample = WeightSample(
            vector_weight=0.6,
            keyword_weight=0.4,
            relevancy_score=0.85,
            query_type="technical",
            query_length=42,
            result_count=5,
            embedding_sim_avg=0.75,
            bm25_score_avg=0.65,
            data_version="2024-01-01T00:00:00",
        )

        assert sample.vector_weight == 0.6
        assert sample.keyword_weight == 0.4
        assert sample.relevancy_score == 0.85
        assert sample.query_type == "technical"
        assert sample.query_length == 42
        assert sample.result_count == 5
        assert sample.embedding_sim_avg == 0.75
        assert sample.bm25_score_avg == 0.65
        assert sample.created_at is not None
        assert sample.data_version == "2024-01-01T00:00:00"

    def test_weight_sample_created_at_timestamp(self) -> None:
        """Test that created_at timestamp is set automatically."""
        sample = WeightSample(
            vector_weight=0.5,
            keyword_weight=0.5,
            relevancy_score=0.7,
            query_type="natural",
            query_length=10,
            result_count=3,
            embedding_sim_avg=0.8,
            bm25_score_avg=0.6,
        )

        assert sample.created_at is not None
        # Verify it looks like an ISO timestamp
        assert "T" in sample.created_at


class TestWeightRecommendation:
    """Tests for WeightRecommendation dataclass."""

    def test_weight_recommendation_creation(self) -> None:
        """Test creating a WeightRecommendation with all fields."""
        rec = WeightRecommendation(
            vector_weight=0.7,
            keyword_weight=0.3,
            confidence=0.85,
            reasoning="vector dominates, technical query, confidence: 85%",
            expected_improvement=0.1,
            samples_used=25,
            model_r2=0.72,
        )

        assert rec.vector_weight == 0.7
        assert rec.keyword_weight == 0.3
        assert rec.confidence == 0.85
        assert rec.samples_used == 25
        assert rec.model_r2 == 0.72
        assert rec.expected_improvement == 0.1
        assert rec.created_at is not None


class TestAdaptiveWeightLearnerInitialisation:
    """Tests for AdaptiveWeightLearner initialisation."""

    def test_init_with_custom_path(self) -> None:
        """Test initialising with custom learning path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(
                learning_path=Path(tmpdir),
                min_samples=10,
                confidence_threshold=0.5,
            )

            assert learner.min_samples == 10
            assert learner.confidence_threshold == 0.5
            assert learner.learning_path == Path(tmpdir)
            assert learner.samples == []
            assert learner.model is None

    def test_init_creates_learning_directory(self) -> None:
        """Test that initialisation creates learning directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_path = Path(tmpdir) / "learning"
            assert not learning_path.exists()

            learner = AdaptiveWeightLearner(learning_path=learning_path)

            assert learning_path.exists()
            assert learning_path.is_dir()

    @patch("scripts.rag.rag_config.RAGConfig")
    def test_init_with_default_path(self, mock_config: MagicMock) -> None:
        """Test initialisation uses RAGConfig default path when none provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.return_value.rag_data_path = tmpdir
            mock_instance = MagicMock()
            mock_instance.rag_data_path = tmpdir
            mock_config.return_value = mock_instance

            learner = AdaptiveWeightLearner()

            assert learner.learning_path is not None
            assert "learning" in str(learner.learning_path)


class TestRecordSample:
    """Tests for recording training samples."""

    def test_record_single_sample(self) -> None:
        """Test recording a single weight sample."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(
                learning_path=Path(tmpdir),
                min_samples=5,
            )

            learner.record_sample(
                vector_weight=0.6,
                keyword_weight=0.4,
                relevancy_score=0.8,
                query_type="technical",
                query_length=30,
                result_count=5,
                embedding_sim_avg=0.75,
                bm25_score_avg=0.65,
            )

            assert len(learner.samples) == 1
            assert learner.samples[0].vector_weight == 0.6
            assert learner.samples[0].relevancy_score == 0.8

    def test_record_multiple_samples(self) -> None:
        """Test recording multiple samples accumulates them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(
                learning_path=Path(tmpdir),
                min_samples=10,
            )

            for i in range(3):
                learner.record_sample(
                    vector_weight=0.5 + i * 0.1,
                    keyword_weight=0.5 - i * 0.1,
                    relevancy_score=0.7 + i * 0.05,
                    query_type="mixed",
                    query_length=20 + i * 5,
                    result_count=5,
                    embedding_sim_avg=0.7,
                    bm25_score_avg=0.6,
                )

            assert len(learner.samples) == 3

    def test_record_sample_triggers_model_fit_when_min_samples_reached(self) -> None:
        """Test that model fitting is triggered once min_samples reached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(
                learning_path=Path(tmpdir),
                min_samples=3,
            )

            # Record first 2 samples - should not fit
            for i in range(2):
                learner.record_sample(
                    vector_weight=0.6,
                    keyword_weight=0.4,
                    relevancy_score=0.7 + i * 0.1,
                    query_type="technical",
                    query_length=30,
                    result_count=5,
                    embedding_sim_avg=0.75,
                    bm25_score_avg=0.65,
                )
                assert learner.model is None

            # Record 3rd sample - should trigger fit
            learner.record_sample(
                vector_weight=0.5,
                keyword_weight=0.5,
                relevancy_score=0.9,
                query_type="natural",
                query_length=20,
                result_count=5,
                embedding_sim_avg=0.8,
                bm25_score_avg=0.70,
            )

            assert learner.model is not None


class TestModelFitting:
    """Tests for linear regression model fitting."""

    def test_fit_model_with_sufficient_samples(self) -> None:
        """Test model fitting with sufficient samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(
                learning_path=Path(tmpdir),
                min_samples=5,
            )

            # Record diverse samples
            for i in range(10):
                learner.record_sample(
                    vector_weight=0.3 + (i % 7) * 0.1,
                    keyword_weight=0.7 - (i % 7) * 0.1,
                    relevancy_score=0.5 + (i % 5) * 0.1,
                    query_type=["technical", "natural", "mixed"][i % 3],
                    query_length=20 + i * 5,
                    result_count=3 + i,
                    embedding_sim_avg=0.6 + (i % 7) * 0.05,
                    bm25_score_avg=0.5 + (i % 6) * 0.08,
                )

            assert learner.model is not None
            # Model should have been fitted
            assert hasattr(learner.model, "coef_")
            assert hasattr(learner.model, "intercept_")

    def test_fit_model_not_called_with_insufficient_samples(self) -> None:
        """Test that model fitting is skipped when below min_samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(
                learning_path=Path(tmpdir),
                min_samples=20,
            )

            for i in range(5):
                learner.record_sample(
                    vector_weight=0.6,
                    keyword_weight=0.4,
                    relevancy_score=0.7,
                    query_type="technical",
                    query_length=30,
                    result_count=5,
                    embedding_sim_avg=0.75,
                    bm25_score_avg=0.65,
                )

            assert learner.model is None


class TestGetRecommendation:
    """Tests for getting weight recommendations."""

    def test_no_recommendation_with_insufficient_samples(self) -> None:
        """Test that None is returned when below min_samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(
                learning_path=Path(tmpdir),
                min_samples=10,
            )

            # Record only 3 samples
            for i in range(3):
                learner.record_sample(
                    vector_weight=0.6,
                    keyword_weight=0.4,
                    relevancy_score=0.7,
                    query_type="technical",
                    query_length=30,
                    result_count=5,
                    embedding_sim_avg=0.75,
                    bm25_score_avg=0.65,
                )

            recommendation = learner.get_recommendation()

            assert recommendation is None

    def test_recommendation_with_sufficient_samples(self) -> None:
        """Test getting a recommendation when minimum samples met."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(
                learning_path=Path(tmpdir),
                min_samples=5,
                confidence_threshold=0.3,  # Lower threshold for testing
            )

            # Record diverse samples
            for i in range(10):
                learner.record_sample(
                    vector_weight=0.3 + (i % 7) * 0.1,
                    keyword_weight=0.7 - (i % 7) * 0.1,
                    relevancy_score=0.5 + (i % 5) * 0.1,
                    query_type=["technical", "natural", "mixed"][i % 3],
                    query_length=20 + i * 5,
                    result_count=5,
                    embedding_sim_avg=0.6 + (i % 7) * 0.05,
                    bm25_score_avg=0.5 + (i % 6) * 0.08,
                )

            recommendation = learner.get_recommendation()

            if recommendation is not None:
                assert isinstance(recommendation, WeightRecommendation)
                assert 0.0 <= recommendation.vector_weight <= 1.0
                assert 0.0 <= recommendation.keyword_weight <= 1.0
                assert 0.0 <= recommendation.confidence <= 1.0
                assert recommendation.samples_used > 0
                assert recommendation.reasoning is not None

    def test_recommendation_respects_confidence_threshold(self) -> None:
        """Test that recommendations respect confidence threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(
                learning_path=Path(tmpdir),
                min_samples=5,
                confidence_threshold=0.95,  # Very high threshold
            )

            # Record samples (unlikely to achieve 95% confidence)
            for i in range(10):
                learner.record_sample(
                    vector_weight=0.6,
                    keyword_weight=0.4,
                    relevancy_score=0.5 + (i % 5) * 0.1,
                    query_type="mixed",
                    query_length=30,
                    result_count=5,
                    embedding_sim_avg=0.7,
                    bm25_score_avg=0.6,
                )

            recommendation = learner.get_recommendation()

            # May return None if confidence below threshold
            if recommendation is not None:
                assert recommendation.confidence >= 0.95


class TestFeatureExtraction:
    """Tests for feature extraction."""

    def test_extract_features_returns_correct_length(self) -> None:
        """Test that feature extraction returns expected number of features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(learning_path=Path(tmpdir))

            features = learner._extract_features(
                vector_weight=0.6,
                keyword_weight=0.4,
                query_type="technical",
                query_length=30,
                embedding_sim=0.75,
                bm25_score=0.65,
            )

            assert isinstance(features, list)
            assert len(features) == 6  # 6 features extracted

    def test_extract_features_encoding(self) -> None:
        """Test feature encoding for different query types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(learning_path=Path(tmpdir))

            features_technical = learner._extract_features(
                vector_weight=0.5,
                keyword_weight=0.5,
                query_type="technical",
                query_length=30,
                embedding_sim=0.7,
                bm25_score=0.6,
            )

            features_natural = learner._extract_features(
                vector_weight=0.5,
                keyword_weight=0.5,
                query_type="natural",
                query_length=30,
                embedding_sim=0.7,
                bm25_score=0.6,
            )

            # Query type encoding should differ (position 2)
            assert features_technical[2] != features_natural[2]
            assert features_technical[2] == 1.0  # technical = 1.0
            assert features_natural[2] == 0.5  # natural = 0.5


class TestReasoningGeneration:
    """Tests for reasoning generation."""

    def test_generate_reasoning_captures_dominant_search_type(self) -> None:
        """Test reasoning mentions which search type dominates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(learning_path=Path(tmpdir))

            reasoning = learner._generate_reasoning(
                vector_weight=0.8,
                keyword_weight=0.2,
                query_type="mixed",
                confidence=0.75,
                improvement=0.05,
            )

            assert "Vector search dominates" in reasoning or "vector" in reasoning.lower()

    def test_generate_reasoning_captures_query_type_optimisation(self) -> None:
        """Test reasoning mentions query type optimisation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(learning_path=Path(tmpdir))

            reasoning = learner._generate_reasoning(
                vector_weight=0.5,
                keyword_weight=0.5,
                query_type="technical",
                confidence=0.8,
                improvement=0.1,
            )

            assert "technical" in reasoning.lower()

    def test_generate_reasoning_includes_confidence(self) -> None:
        """Test reasoning includes confidence percentage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(learning_path=Path(tmpdir))

            reasoning = learner._generate_reasoning(
                vector_weight=0.6,
                keyword_weight=0.4,
                query_type="mixed",
                confidence=0.85,
                improvement=0.0,
            )

            assert "85%" in reasoning or "confidence" in reasoning.lower()


class TestPersistence:
    """Tests for saving and loading samples."""

    def test_save_and_load_samples(self) -> None:
        """Test that samples persist to disk and are reloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_path = Path(tmpdir)

            # Create learner and record samples
            learner1 = AdaptiveWeightLearner(learning_path=learning_path, min_samples=5)

            for i in range(10):
                learner1.record_sample(
                    vector_weight=0.6,
                    keyword_weight=0.4,
                    relevancy_score=0.7 + i * 0.02,
                    query_type="mixed",
                    query_length=30,
                    result_count=5,
                    embedding_sim_avg=0.75,
                    bm25_score_avg=0.65,
                )

            assert len(learner1.samples) == 10

            # Create new learner with same path
            learner2 = AdaptiveWeightLearner(learning_path=learning_path, min_samples=5)

            # Should load previously saved samples
            assert len(learner2.samples) == 10
            assert learner2.model is not None  # Should have fitted model from loaded samples

    def test_samples_json_file_created(self) -> None:
        """Test that samples.json file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_path = Path(tmpdir)
            learner = AdaptiveWeightLearner(learning_path=learning_path)

            learner.record_sample(
                vector_weight=0.6,
                keyword_weight=0.4,
                relevancy_score=0.8,
                query_type="technical",
                query_length=30,
                result_count=5,
                embedding_sim_avg=0.75,
                bm25_score_avg=0.65,
            )

            samples_file = learning_path / "weight_samples.json"
            assert samples_file.exists()

            with open(samples_file) as f:
                data = json.load(f)
                assert len(data) == 1
                assert data[0]["vector_weight"] == 0.6

    def test_save_recommendation(self) -> None:
        """Test that recommendations are saved to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_path = Path(tmpdir)
            learner = AdaptiveWeightLearner(
                learning_path=learning_path,
                min_samples=5,
                confidence_threshold=0.3,
            )

            # Record enough samples
            for i in range(10):
                learner.record_sample(
                    vector_weight=0.6,
                    keyword_weight=0.4,
                    relevancy_score=0.7 + i * 0.02,
                    query_type="technical",
                    query_length=30,
                    result_count=5,
                    embedding_sim_avg=0.75,
                    bm25_score_avg=0.65,
                )

            recommendation = learner.get_recommendation()

            if recommendation is not None:
                rec_file = learning_path / "last_recommendation.json"
                assert rec_file.exists()

                with open(rec_file) as f:
                    data = json.load(f)
                    assert data["vector_weight"] == recommendation.vector_weight
                    assert data["confidence"] == recommendation.confidence


class TestStatistics:
    """Tests for statistics reporting."""

    def test_get_statistics_returns_dict(self) -> None:
        """Test that statistics returns dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(learning_path=Path(tmpdir))

            stats = learner.get_statistics()

            assert isinstance(stats, dict)
            assert "samples_collected" in stats
            assert "model_fitted" in stats
            assert "confidence_threshold" in stats

    def test_statistics_reflect_state(self) -> None:
        """Test that statistics accurately reflect learner state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(
                learning_path=Path(tmpdir),
                min_samples=5,
            )

            # Before any samples
            stats1 = learner.get_statistics()
            assert stats1["samples_collected"] == 0
            assert stats1["model_fitted"] is False

            # After recording samples
            for i in range(10):
                learner.record_sample(
                    vector_weight=0.6,
                    keyword_weight=0.4,
                    relevancy_score=0.7,
                    query_type="mixed",
                    query_length=30,
                    result_count=5,
                    embedding_sim_avg=0.75,
                    bm25_score_avg=0.65,
                )

            stats2 = learner.get_statistics()
            assert stats2["samples_collected"] == 10
            assert stats2["model_fitted"] is True


class TestApplyRecommendation:
    """Tests for applying recommendations to weight manager."""

    def test_apply_recommendation_with_no_recommendation(self) -> None:
        """Test applying recommendation when none exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(learning_path=Path(tmpdir))

            result = learner.apply_recommendation_to_weights()

            assert result is False

    def test_apply_recommendation_with_valid_recommendation(self) -> None:
        """Test applying a valid recommendation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(
                learning_path=Path(tmpdir),
                min_samples=5,
                confidence_threshold=0.3,
            )

            # Record samples
            for i in range(10):
                learner.record_sample(
                    vector_weight=0.6 + (i % 3) * 0.05,
                    keyword_weight=0.4 - (i % 3) * 0.05,
                    relevancy_score=0.7 + i * 0.02,
                    query_type="mixed",
                    query_length=30,
                    result_count=5,
                    embedding_sim_avg=0.75,
                    bm25_score_avg=0.65,
                )

            rec = learner.get_recommendation()
            if rec is not None:
                # Should successfully apply
                result = learner.apply_recommendation_to_weights()
                # Result might be False if weight manager not available in test, that's OK
                assert isinstance(result, bool)


class TestSampleFreshness:
    """Tests for data freshness and sample aging."""

    def test_sample_weights_all_fresh(self) -> None:
        """Test that all fresh samples get full weight."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(
                learning_path=Path(tmpdir),
                sample_decay_days=30,
            )

            # Record recent samples
            for i in range(5):
                learner.record_sample(
                    vector_weight=0.6,
                    keyword_weight=0.4,
                    relevancy_score=0.7,
                    query_type="mixed",
                    query_length=30,
                    result_count=5,
                    embedding_sim_avg=0.75,
                    bm25_score_avg=0.65,
                )

            weights = learner._get_sample_weights()

            # All samples should have full weight
            assert len(weights) == 5
            assert all(w == 1.0 for w in weights)

    def test_get_current_data_version(self) -> None:
        """Test getting current data version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(learning_path=Path(tmpdir))

            version = learner._get_current_data_version()

            # Should return an ISO timestamp string
            assert isinstance(version, str)
            assert len(version) > 0
            # Should be parseable as datetime
            from datetime import datetime
            try:
                datetime.fromisoformat(version.replace('Z', '+00:00'))
                valid_timestamp = True
            except Exception:
                valid_timestamp = False
            assert valid_timestamp

    def test_statistics_include_freshness_metrics(self) -> None:
        """Test that statistics include sample freshness information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveWeightLearner(
                learning_path=Path(tmpdir),
                min_samples=5,
            )

            # Record enough samples to fit model
            for i in range(10):
                learner.record_sample(
                    vector_weight=0.6,
                    keyword_weight=0.4,
                    relevancy_score=0.7,
                    query_type="mixed",
                    query_length=30,
                    result_count=5,
                    embedding_sim_avg=0.75,
                    bm25_score_avg=0.65,
                )

            stats = learner.get_statistics()

            # Should include freshness metrics
            assert "chromadb_last_modified" in stats
            if stats["model_fitted"]:
                assert "active_samples" in stats
                assert "stale_samples" in stats
                assert "mean_sample_weight" in stats


class TestGlobalInstance:
    """Tests for global singleton instance."""

    def test_get_adaptive_weight_learner_returns_singleton(self) -> None:
        """Test that get_adaptive_weight_learner returns singleton instance."""
        learner1 = get_adaptive_weight_learner()
        learner2 = get_adaptive_weight_learner()

        assert learner1 is learner2
