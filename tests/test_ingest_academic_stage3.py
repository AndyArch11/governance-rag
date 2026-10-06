"""Tests for academic ingestion stage 3 (metadata resolution)."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.ingest import ingest_academic as ingest_academic_module
from scripts.ingest.academic.cache import ReferenceCache
from scripts.ingest.academic.config import AcademicIngestConfig
from scripts.ingest.academic.providers import resolve_reference
from scripts.ingest.academic.providers.base import Reference, ReferenceStatus
from scripts.ingest.ingest_academic import record_document_citations, stage_resolve_metadata


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


def test_record_document_citations_links_resolved_refs_to_source_document():
    class CitationCache:
        def __init__(self):
            self.links = []

        def add_citation(self, doc_id, ref_id, raw_citation):
            self.links.append((doc_id, ref_id, raw_citation))

    cache = CitationCache()
    count = record_document_citations(
        cache,
        "thesis-1",
        [
            {"ref_id": "ref-1", "citation": "First citation"},
            {"ref_id": "ref-2", "citation": "Second citation"},
            {"citation": "Missing ID"},
            {"ref_id": "ref-3"},
        ],
    )

    assert count == 2
    assert cache.links == [
        ("thesis-1", "ref-1", "First citation"),
        ("thesis-1", "ref-2", "Second citation"),
    ]


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
    assert resolved[0]["ref_id"].startswith("placeholder_")


def test_stage_resolve_metadata_reports_structured_progress(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(
        ingest_academic_module,
        "audit",
        lambda event, data: events.append((event, data)),
    )
    cache = ReferenceCache(str(tmp_path / "references.sqlite"))
    config = SimpleNamespace(dry_run=True)

    stage_resolve_metadata(["First citation", "Second citation"], cache, config, DummyLogger())

    checkpoints = [data for event, data in events if event == "progress_checkpoint"]
    assert checkpoints[0]["stage"] == "reference_resolution"
    assert checkpoints[0]["items_done"] == 0
    assert checkpoints[-1]["items_done"] == 2
    assert checkpoints[-1]["items_total"] == 2
    assert checkpoints[-1]["skipped"] == 2


def test_stage_resolve_metadata_forwards_doi_and_year_to_provider(monkeypatch):
    """Provider resolution uses parsed identifiers before fuzzy title matching."""
    from scripts.ingest import ingest_academic

    cache = Mock()
    cache.get.return_value = None
    logger = DummyLogger()
    config = Mock(dry_run=False)
    resolved_reference = Reference(
        ref_id="doi:10.1000/example",
        raw_citation="Smith, J. (2024). Citation title.",
        doi="10.1000/example",
        title="Citation title",
        year=2024,
        resolved=True,
    )
    calls = []

    def fake_resolve_reference(citation, year=None, doi=None, logger=None):
        calls.append({"citation": citation, "year": year, "doi": doi})
        return resolved_reference, 0.95

    monkeypatch.setattr(ingest_academic, "resolve_reference", fake_resolve_reference)

    resolved = stage_resolve_metadata(
        ["Smith, J. (2024). Citation title. https://doi.org/10.1000/example-"],
        cache,
        config,
        logger,
    )

    assert calls == [
        {
            "citation": "Smith, J. (2024). Citation title. https://doi.org/10.1000/example-",
            "year": 2024,
            "doi": "10.1000/example",
        }
    ]
    assert resolved[0]["title"] == "Citation title"


def test_stage_resolve_metadata_converts_provider_reference_for_cache(monkeypatch):
    """Provider results cross into the cache through its explicit persistence model."""
    cache = Mock()
    cache.get.return_value = None
    logger = DummyLogger()
    config = Mock(dry_run=False)
    provider_reference = Reference(
        ref_id="ref-1",
        raw_citation="Smith (2024). Test.",
        title="Test",
        status=ReferenceStatus.RESOLVED,
        resolved=True,
    )
    monkeypatch.setattr(
        "scripts.ingest.ingest_academic.resolve_reference",
        lambda *args, **kwargs: (provider_reference, 0.9),
    )

    resolved = stage_resolve_metadata([provider_reference.raw_citation], cache, config, logger)

    cached_reference = cache.put.call_args.args[1]
    assert type(cached_reference).__module__ == "scripts.ingest.academic.cache"
    assert cached_reference.status == "resolved"
    assert cached_reference.ref_id == "ref-1"
    assert resolved[0]["ref_id"] == "ref-1"


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
    assert [ref["ref_id"] for ref in first_run] == [ref["ref_id"] for ref in second_run]
