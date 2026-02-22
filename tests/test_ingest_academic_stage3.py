"""Tests for academic ingestion stage 3 (metadata resolution)."""

from unittest.mock import Mock, patch

from scripts.ingest.academic.cache import ReferenceCache
from scripts.ingest.academic.config import AcademicIngestConfig
from scripts.ingest.academic.providers import resolve_reference
from scripts.ingest.academic.providers.base import Reference, ReferenceStatus
from scripts.ingest.ingest_academic import stage_resolve_metadata


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, msg):
        self.infos.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)

    def error(self, msg, exc_info=False):
        self.errors.append(msg)


def test_resolve_reference_returns_reference_object():
    """Test that resolve_reference returns a Reference object and confidence tuple."""
    # Mock the provider chain to avoid actual API calls
    mock_ref = Reference(
        ref_id="test_123",
        raw_citation="Smith, J. (2020). Test Paper.",
        doi="10.1000/182",
        resolved=True,
        title="Test Paper",
        year=2020,
        status=ReferenceStatus.RESOLVED,
    )

    # Mock the default chain's resolve method
    with patch("scripts.ingest.academic.providers.create_default_chain") as mock_chain:
        # Create a mock ResolutionResult (use Mock to avoid dataclass instantiation)
        mock_result = Mock()
        mock_result.reference = mock_ref
        mock_result.confidence = 0.95
        mock_chain.return_value.resolve.return_value = mock_result

        resolved_ref, confidence = resolve_reference(
            "Smith, J. (2020). Test Paper. doi:10.1000/182"
        )
        assert resolved_ref is not None
        assert hasattr(resolved_ref, "doi")
        assert resolved_ref.doi == "10.1000/182"
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0


def test_stage_resolve_metadata_returns_list():
    """Test that stage_resolve_metadata returns a list of dicts."""
    cache = ReferenceCache()
    logger = DummyLogger()
    config = AcademicIngestConfig()
    config.dry_run = True  # Skip API calls in dry-run mode
    citations = ["Example citation 1", "Example citation 2"]

    resolved = stage_resolve_metadata(citations, cache, config, logger)

    assert isinstance(resolved, list)
    assert len(resolved) == 2
    for ref in resolved:
        assert isinstance(ref, dict)
        assert "citation" in ref


def test_stage_resolve_metadata_dry_run_skips_resolution():
    """Test that dry-run mode skips expensive provider resolution."""
    cache = ReferenceCache()
    logger = DummyLogger()
    config = AcademicIngestConfig()
    config.dry_run = True
    citations = ["Example citation"]

    resolved = stage_resolve_metadata(citations, cache, config, logger)

    # In dry-run mode, should create unresolved references quickly
    assert len(resolved) == 1
    assert resolved[0]["citation"] == "Example citation"


def test_stage_resolve_metadata_cache_integration():
    """Test that cache integration works in stage_resolve_metadata."""
    cache = ReferenceCache()
    logger = DummyLogger()
    config = AcademicIngestConfig()
    config.dry_run = True

    # Run resolution twice with same citations
    citations = ["Test citation 1", "Test citation 2"]
    first_run = stage_resolve_metadata(citations, cache, config, logger)
    second_run = stage_resolve_metadata(citations, cache, config, logger)

    # Both runs should produce same number of results
    assert len(first_run) == len(second_run) == 2

    # Results should have citation text
    assert all("citation" in ref for ref in first_run)
    assert all("citation" in ref for ref in second_run)
