"""Tests for persona-aware retrieval functionality.

Tests persona configuration, filtering logic, reranking scores,
and edge cases for academic query personalisation.
"""

import pytest
from datetime import datetime

from scripts.search.persona_retrieval import (
    PersonaConfig,
    get_persona_config,
    apply_persona_reranking,
    SUPERVISOR_CONFIG,
    ASSESSOR_CONFIG,
    RESEARCHER_CONFIG,
)


class TestPersonaConfig:
    """Test persona configuration retrieval."""

    def test_get_supervisor_config(self):
        """Retrieve supervisor configuration."""
        config = get_persona_config("supervisor")
        assert config.persona == "supervisor"
        assert config.reference_depth == 2
        assert config.min_quality_score == 0.6
        assert config.min_citation_count == 10
        assert config.prefer_reference_types == ("academic", "report", "preprint")
        assert config.include_stale_links is True
        assert config.require_verifiable is False
        assert config.stale_link_penalty == 0.2
        assert config.recency_bias == 0.1

    def test_get_assessor_config(self):
        """Retrieve assessor configuration."""
        config = get_persona_config("assessor")
        assert config.persona == "assessor"
        assert config.reference_depth == 3
        assert config.min_quality_score == 0.7
        assert config.min_citation_count == 5
        assert config.prefer_reference_types == ("academic", "report")
        assert config.include_stale_links is False
        assert config.require_verifiable is True
        assert config.stale_link_penalty == 0.5
        assert config.recency_bias == 0.0

    def test_get_researcher_config(self):
        """Retrieve researcher configuration."""
        config = get_persona_config("researcher")
        assert config.persona == "researcher"
        assert config.reference_depth == 1
        assert config.min_quality_score == 0.4
        assert config.min_citation_count == 0
        assert config.prefer_reference_types == ("preprint", "academic", "report")
        assert config.include_stale_links is True
        assert config.require_verifiable is False
        assert config.stale_link_penalty == 0.3
        assert config.recency_bias == 0.4

    def test_get_config_case_insensitive(self):
        """Config retrieval is case-insensitive."""
        assert get_persona_config("SUPERVISOR").persona == "supervisor"
        assert get_persona_config("Assessor").persona == "assessor"
        assert get_persona_config("  researcher  ").persona == "researcher"

    def test_get_config_unknown_persona(self):
        """Raise ValueError for unknown persona."""
        with pytest.raises(ValueError, match="Unknown persona"):
            get_persona_config("invalid")
        with pytest.raises(ValueError, match="Unknown persona"):
            get_persona_config("")
        with pytest.raises(ValueError, match="Unknown persona"):
            get_persona_config("student")


class TestApplyPersonaReranking:
    """Test persona-based filtering and reranking."""

    def test_empty_input(self):
        """Handle empty chunks gracefully."""
        chunks, metadata = apply_persona_reranking([], [], "supervisor", top_k=5)
        assert chunks == []
        assert metadata == []

    def test_filter_by_reference_type_supervisor(self):
        """Supervisor filters to academic/report/preprint only."""
        chunks = ["chunk1", "chunk2", "chunk3", "chunk4"]
        metadata = [
            {"reference_type": "academic", "quality_score": 0.8, "citation_count": 15, "distance": 0.1},
            {"reference_type": "blog", "quality_score": 0.8, "citation_count": 15, "distance": 0.1},
            {"reference_type": "preprint", "quality_score": 0.8, "citation_count": 15, "distance": 0.15},
            {"reference_type": "news", "quality_score": 0.8, "citation_count": 15, "distance": 0.1},
        ]
        result_chunks, result_meta = apply_persona_reranking(
            chunks, metadata, "supervisor", top_k=10
        )
        # Should only have academic and preprint
        assert len(result_chunks) == 2
        types = {m["reference_type"] for m in result_meta}
        assert types == {"academic", "preprint"}

    def test_filter_by_reference_type_assessor(self):
        """Assessor filters to academic/report only."""
        chunks = ["chunk1", "chunk2", "chunk3"]
        metadata = [
            {"reference_type": "academic", "quality_score": 0.8, "citation_count": 10, "distance": 0.1},
            {"reference_type": "report", "quality_score": 0.8, "citation_count": 10, "distance": 0.1},
            {"reference_type": "preprint", "quality_score": 0.8, "citation_count": 10, "distance": 0.1},
        ]
        result_chunks, result_meta = apply_persona_reranking(
            chunks, metadata, "assessor", top_k=10
        )
        # Should only have academic and report
        assert len(result_chunks) == 2
        types = {m["reference_type"] for m in result_meta}
        assert types == {"academic", "report"}

    def test_filter_by_quality_score(self):
        """Filter by minimum quality score threshold."""
        chunks = ["chunk1", "chunk2", "chunk3"]
        metadata = [
            {"reference_type": "academic", "quality_score": 0.8, "citation_count": 10, "distance": 0.1},
            {"reference_type": "academic", "quality_score": 0.5, "citation_count": 10, "distance": 0.1},
            {"reference_type": "academic", "quality_score": 0.3, "citation_count": 10, "distance": 0.1},
        ]
        # Assessor requires min_quality >= 0.7
        result_chunks, result_meta = apply_persona_reranking(
            chunks, metadata, "assessor", top_k=10
        )
        assert len(result_chunks) == 1
        assert result_meta[0]["quality_score"] == 0.8

    def test_filter_by_citation_count(self):
        """Filter by minimum citation count threshold."""
        chunks = ["chunk1", "chunk2", "chunk3"]
        metadata = [
            {"reference_type": "academic", "quality_score": 0.8, "citation_count": 50, "distance": 0.1},
            {"reference_type": "academic", "quality_score": 0.8, "citation_count": 5, "distance": 0.1},
            {"reference_type": "academic", "quality_score": 0.8, "citation_count": 0, "distance": 0.1},
        ]
        # Supervisor requires min_citation >= 10
        result_chunks, result_meta = apply_persona_reranking(
            chunks, metadata, "supervisor", top_k=10
        )
        assert len(result_chunks) == 1
        assert result_meta[0]["citation_count"] == 50

    def test_filter_stale_links_assessor(self):
        """Assessor excludes stale links."""
        chunks = ["chunk1", "chunk2", "chunk3"]
        metadata = [
            {
                "reference_type": "academic",
                "quality_score": 0.8,
                "citation_count": 10,
                "link_status": "available",
                "distance": 0.1,
            },
            {
                "reference_type": "academic",
                "quality_score": 0.8,
                "citation_count": 10,
                "link_status": "stale_404",
                "distance": 0.1,
            },
            {
                "reference_type": "academic",
                "quality_score": 0.8,
                "citation_count": 10,
                "link_status": "stale_timeout",
                "distance": 0.1,
            },
        ]
        result_chunks, result_meta = apply_persona_reranking(
            chunks, metadata, "assessor", top_k=10
        )
        # Assessor require_verifiable=True and include_stale_links=False
        assert len(result_chunks) == 1
        assert result_meta[0]["link_status"] == "available"

    def test_include_stale_links_supervisor(self):
        """Supervisor includes stale links but penalises them."""
        chunks = ["chunk1", "chunk2"]
        metadata = [
            {
                "reference_type": "academic",
                "quality_score": 0.8,
                "citation_count": 15,
                "link_status": "available",
                "distance": 0.2,
            },
            {
                "reference_type": "academic",
                "quality_score": 0.8,
                "citation_count": 15,
                "link_status": "stale_404",
                "distance": 0.2,
            },
        ]
        result_chunks, result_meta = apply_persona_reranking(
            chunks, metadata, "supervisor", top_k=10
        )
        # Supervisor include_stale_links=True, both should appear
        assert len(result_chunks) == 2

        # Stale link should be ranked lower due to penalty
        assert result_meta[0]["link_status"] == "available"
        assert result_meta[1]["link_status"] == "stale_404"
        # Stale link gets -0.2 penalty for supervisor
        assert result_meta[0]["persona_score"] > result_meta[1]["persona_score"]

    def test_recency_bias_researcher(self):
        """Researcher applies recency bias to recent publications."""
        current_year = datetime.now().year
        chunks = ["chunk1", "chunk2"]
        metadata = [
            {
                "reference_type": "preprint",
                "quality_score": 0.6,
                "citation_count": 5,
                "year": current_year - 1,  # Recent
                "distance": 0.2,
            },
            {
                "reference_type": "preprint",
                "quality_score": 0.6,
                "citation_count": 5,
                "year": current_year - 10,  # Old
                "distance": 0.2,
            },
        ]
        result_chunks, result_meta = apply_persona_reranking(
            chunks, metadata, "researcher", top_k=10
        )
        # Researcher has recency_bias=0.4, recent paper should rank higher
        assert len(result_chunks) == 2
        assert result_meta[0]["year"] == current_year - 1
        assert result_meta[1]["year"] == current_year - 10
        assert result_meta[0]["persona_score"] > result_meta[1]["persona_score"]

    def test_top_k_limit(self):
        """Respect top_k result limit."""
        chunks = [f"chunk{i}" for i in range(20)]
        metadata = [
            {
                "reference_type": "academic",
                "quality_score": 0.8,
                "citation_count": 15,
                "distance": 0.1 + (i * 0.01),
            }
            for i in range(20)
        ]
        result_chunks, result_meta = apply_persona_reranking(
            chunks, metadata, "supervisor", top_k=5
        )
        assert len(result_chunks) == 5
        assert len(result_meta) == 5

    def test_scoring_with_tf_scores(self):
        """Use TF scores for similarity when available."""
        chunks = ["chunk1", "chunk2"]
        metadata = [
            {
                "reference_type": "academic",
                "quality_score": 0.8,
                "citation_count": 15,
                "tf_score": 10.0,  # High TF score
            },
            {
                "reference_type": "academic",
                "quality_score": 0.8,
                "citation_count": 15,
                "tf_score": 2.0,  # Low TF score
            },
        ]
        result_chunks, result_meta = apply_persona_reranking(
            chunks, metadata, "supervisor", top_k=10
        )
        # Higher TF score should rank first
        assert result_meta[0]["tf_score"] == 10.0
        assert result_meta[1]["tf_score"] == 2.0

    def test_metadata_enrichment(self):
        """Add persona and persona_score to metadata."""
        chunks = ["chunk1"]
        metadata = [
            {
                "reference_type": "academic",
                "quality_score": 0.8,
                "citation_count": 15,
                "distance": 0.1,
            }
        ]
        result_chunks, result_meta = apply_persona_reranking(
            chunks, metadata, "supervisor", top_k=10
        )
        assert result_meta[0]["persona"] == "supervisor"
        assert "persona_score" in result_meta[0]
        assert isinstance(result_meta[0]["persona_score"], float)

    def test_missing_metadata_fields(self):
        """Handle missing optional metadata fields gracefully."""
        chunks = ["chunk1", "chunk2"]
        metadata = [
            # Minimal metadata - missing required fields for filtering
            {"distance": 0.1},
            # Has quality score that passes researcher threshold (0.4)
            {"tf_score": 5.0, "quality_score": 0.5},
        ]
        # Should not crash, researcher has no filters
        result_chunks, result_meta = apply_persona_reranking(
            chunks, metadata, "researcher", top_k=10
        )
        # Only chunk2 should pass (has quality_score >= 0.4)
        # chunk1 missing quality_score defaults to 0.0 < 0.4 threshold
        assert len(result_chunks) == 1
        assert "tf_score" in result_meta[0]

    def test_domain_relevance_scoring(self):
        """Include domain_relevance_score in scoring calculation."""
        chunks = ["chunk1", "chunk2"]
        metadata = [
            {
                "reference_type": "academic",
                "quality_score": 0.8,
                "citation_count": 15,
                "distance": 0.2,
                "domain_relevance_score": 0.9,  # High domain relevance
            },
            {
                "reference_type": "academic",
                "quality_score": 0.8,
                "citation_count": 15,
                "distance": 0.2,
                "domain_relevance_score": 0.1,  # Low domain relevance
            },
        ]
        result_chunks, result_meta = apply_persona_reranking(
            chunks, metadata, "supervisor", top_k=10
        )
        # Higher domain relevance should rank first (15% weight in formula)
        assert result_meta[0]["domain_relevance_score"] == 0.9
        assert result_meta[1]["domain_relevance_score"] == 0.1

    def test_stale_link_penalty_applied(self):
        """Verify stale link penalty is correctly applied."""
        chunks = ["chunk1", "chunk2"]
        metadata = [
            {
                "reference_type": "academic",
                "quality_score": 0.8,
                "citation_count": 15,
                "link_status": "available",
                "distance": 0.2,
            },
            {
                "reference_type": "academic",
                "quality_score": 0.8,
                "citation_count": 15,
                "link_status": "stale_404",
                "distance": 0.2,
            },
        ]
        result_chunks, result_meta = apply_persona_reranking(
            chunks, metadata, "assessor", top_k=10  # Only returns available
        )
        # Assessor filters stale links, should only return available
        assert len(result_chunks) == 1

        # Try with supervisor (includes stale)
        result_chunks, result_meta = apply_persona_reranking(
            chunks, metadata, "supervisor", top_k=10
        )
        assert len(result_chunks) == 2
        # Penalty = 0.2 for supervisor
        penalty = result_meta[0]["persona_score"] - result_meta[1]["persona_score"]
        assert penalty > 0.15  # Should be close to 0.2

    def test_citation_score_normalisation(self):
        """Normalise citation counts to [0,1] range."""
        chunks = ["chunk1", "chunk2", "chunk3"]
        metadata = [
            {
                "reference_type": "academic",
                "quality_score": 0.8,
                "citation_count": 0,  # No citations
                "distance": 0.2,
            },
            {
                "reference_type": "academic",
                "quality_score": 0.8,
                "citation_count": 50,  # Medium citations
                "distance": 0.2,
            },
            {
                "reference_type": "academic",
                "quality_score": 0.8,
                "citation_count": 200,  # High citations (capped at 100)
                "distance": 0.2,
            },
        ]
        result_chunks, result_meta = apply_persona_reranking(
            chunks, metadata, "researcher", top_k=10
        )
        # Higher citations should rank higher (10% weight)
        assert result_meta[0]["citation_count"] >= result_meta[1]["citation_count"]
        assert result_meta[1]["citation_count"] >= result_meta[2]["citation_count"]
