"""
Fixtures for testing academic models.

Provides pytest fixtures for creating valid/invalid test instances
of all model classes with various edge cases and scenarios.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest

from scripts.ingest.academic.models import (
    # Enums
    DocumentStatus,
    ReferenceStatus,
    EdgeType,
    OAStatus,
    LinkStatus,
    # Core Models
    AcademicDocument,
    Reference,
    CitationEdge,
    RawCitation,
    ParsedCitation,
    NormalisedCitation,
    DomainTerm,
    BatchIngestResult,
    RevalidationResult,
    PersonaContext,
    # Factory Functions
    create_document,
    create_reference,
)


# ============================================================================
# Document Fixtures
# ============================================================================

@pytest.fixture
def valid_document_data() -> Dict[str, Any]:
    """Valid academic document data."""
    return {
        "doc_id": "acad_ml_2024_abc12345",
        "source_file": "/data/thesis.pdf",
        "title": "Machine Learning for Healthcare: A Comprehensive Survey",
        "authors": ["John Smith", "Jane Doe"],
        "author_orcids": ["0000-0001-2345-6789", "0000-0002-3456-7890"],
        "year": 2024,
        "abstract": "This paper surveys recent advances in ML for healthcare applications.",
        "keywords": ["machine learning", "healthcare", "survey"],
        "primary_domain": "machine_learning",
        "secondary_domains": ["healthcare", "artificial_intelligence"],
        "topic": "deep learning for medical imaging",
        "full_text": "Full text of the document..." * 100,
        "file_hash": "abc123def456",
        "status": DocumentStatus.LOADED,
        "word_count": 15000,
        "citation_count": 150,
        "reference_count": 120,
    }


@pytest.fixture
def valid_document(valid_document_data) -> AcademicDocument:
    """Create valid academic document."""
    return AcademicDocument(**valid_document_data)


@pytest.fixture
def minimal_document() -> AcademicDocument:
    """Minimal valid document (only required fields)."""
    return AcademicDocument(
        doc_id="acad_test_001",
        full_text="Some content",
        title="Test",
    )


@pytest.fixture
def document_with_sections() -> AcademicDocument:
    """Document with sections."""
    return AcademicDocument(
        doc_id="acad_test_sections",
        full_text="Full text",
        title="Sectioned Document",
        sections={
            "introduction": "Introduction text",
            "methods": "Methods text",
            "results": "Results text",
            "conclusion": "Conclusion text",
        }
    )


@pytest.fixture
def document_ingested() -> AcademicDocument:
    """Fully processed document."""
    doc = AcademicDocument(
        doc_id="acad_ingested_001",
        full_text="Complete document",
        title="Ingested Document",
        status=DocumentStatus.INGESTED,
        word_count=5000,
        citation_count=50,
        reference_count=45,
    )
    doc.reference_ids = ["ref_001", "ref_002", "ref_003"]
    return doc


# ============================================================================
# Reference Fixtures
# ============================================================================

@pytest.fixture
def valid_reference_data() -> Dict[str, Any]:
    """Valid reference data."""
    return {
        "ref_id": "ref_10.1234/example",
        "doi": "10.1234/example",
        "title": "Example Paper",
        "authors": ["Author One", "Author Two"],
        "author_orcids": ["0000-0001-1111-1111", None],
        "year": 2023,
        "abstract": "This is an example paper about something.",
        "venue": "Journal of Examples",
        "volume": "42",
        "issue": "3",
        "pages": "123-145",
        "raw_citation": "Author One, Author Two (2023). Example Paper. Journal of Examples, 42(3), 123-145.",
        "citation_format": "harvard",
        "resolved": True,
        "status": ReferenceStatus.RESOLVED,
        "metadata_provider": "crossref",
        "oa_available": True,
        "oa_status": OAStatus.GREEN,
        "oa_url": "https://example.org/paper.pdf",
        "reference_type": "academic",
        "quality_score": 0.95,
    }


@pytest.fixture
def valid_reference(valid_reference_data) -> Reference:
    """Create valid reference."""
    return Reference(**valid_reference_data)


@pytest.fixture
def unresolved_reference() -> Reference:
    """Unresolved reference (only raw citation)."""
    return Reference(
        ref_id="ref_unresolved_001",
        raw_citation="Smith et al. (2020). Some title. Unknown Journal.",
        status=ReferenceStatus.RAW,
        resolved=False,
    )


@pytest.fixture
def preprint_reference() -> Reference:
    """arXiv preprint reference."""
    return Reference(
        ref_id="ref_arxiv_2401.01234",
        arxiv_id="2401.01234",
        title="A Preprint About Something",
        authors=["Alice Author"],
        year=2024,
        venue="arXiv",
        reference_type="preprint",
        metadata_provider="arxiv",
        quality_score=0.75,
        resolved=True,
        status=ReferenceStatus.RESOLVED,
    )


@pytest.fixture
def oa_reference() -> Reference:
    """Open access reference with multiple OA statuses."""
    return Reference(
        ref_id="ref_oa_gold",
        doi="10.1234/oa_gold",
        title="Gold OA Paper",
        authors=["Gold Author"],
        year=2024,
        venue="Open Access Journal",
        oa_available=True,
        oa_status=OAStatus.GOLD,
        oa_url="https://publisher.org/open/paper.pdf",
        reference_type="academic",
        quality_score=1.0,
        resolved=True,
    )


@pytest.fixture
def reference_with_pdf() -> Reference:
    """Reference with downloaded PDF."""
    return Reference(
        ref_id="ref_pdf_001",
        doi="10.1234/pdf_test",
        title="PDF Downloaded Paper",
        authors=["PDF Author"],
        year=2023,
        pdf_downloaded=True,
        pdf_local_path="/data/pdfs/paper_abc123.pdf",
        pdf_file_hash="sha256_hash_here",
        accessed_at=datetime.now(timezone.utc),
        resolved=True,
    )


@pytest.fixture
def stale_reference() -> Reference:
    """Reference with stale link."""
    return Reference(
        ref_id="ref_stale_001",
        title="Stale Paper",
        authors=["Stale Author"],
        link_status=LinkStatus.STALE_404,
        link_checked_at=datetime.now(timezone.utc),
        resolved=True,
    )


@pytest.fixture
def cited_reference(valid_reference) -> Reference:
    """Reference cited by multiple documents."""
    ref = valid_reference
    ref.doc_ids = ["doc_001", "doc_002", "doc_003"]
    ref.citation_count = 25
    return ref


# ============================================================================
# Citation Edge Fixtures
# ============================================================================

@pytest.fixture
def direct_citation_edge() -> CitationEdge:
    """Direct citation edge."""
    return CitationEdge(
        from_id="doc_001",
        to_id="ref_10.1234/example",
        edge_type=EdgeType.DIRECT,
        is_inline=True,
        mention_text="As shown in Smith et al. (2023), this approach works well.",
        strength_score=0.9,
    )


@pytest.fixture
def indirect_citation_edge() -> CitationEdge:
    """Indirect citation edge (A→B→C)."""
    return CitationEdge(
        from_id="ref_001",
        to_id="ref_002",
        edge_type=EdgeType.INDIRECT,
        depth=2,
        strength_score=0.5,
    )


@pytest.fixture
def contradicting_edge() -> CitationEdge:
    """Contradicting reference edge."""
    return CitationEdge(
        from_id="doc_001",
        to_id="ref_002",
        edge_type=EdgeType.CONTRADICTS,
        mention_text="However, recent work contradicts this claim.",
        cited_as="critique",
        strength_score=0.8,
    )


@pytest.fixture
def extending_edge() -> CitationEdge:
    """Extending methodology edge."""
    return CitationEdge(
        from_id="doc_001",
        to_id="ref_003",
        edge_type=EdgeType.EXTENDS,
        mention_text="We extend the methodology of...",
        cited_as="extension",
        strength_score=0.85,
    )


# ============================================================================
# Citation Parsing Fixtures
# ============================================================================

@pytest.fixture
def raw_citation_bibliography() -> RawCitation:
    """Raw bibliography citation."""
    return RawCitation(
        raw_text="Smith, J., Doe, J. (2023). Example Paper. Journal of Examples, 42(3), 123-145.",
        doc_id="doc_001",
        position="bibliography",
    )


@pytest.fixture
def raw_citation_inline() -> RawCitation:
    """Raw inline citation."""
    return RawCitation(
        raw_text="Smith et al. (2023)",
        doc_id="doc_001",
        position="inline",
        position_offset=1234,
    )


@pytest.fixture
def parsed_citation_harvard() -> ParsedCitation:
    """Parsed Harvard format citation."""
    return ParsedCitation(
        title="Example Paper",
        authors=["Smith, J.", "Doe, J."],
        year=2023,
        venue="Journal of Examples",
        volume="42",
        issue="3",
        pages="123-145",
        format="harvard",
        confidence=0.95,
    )


@pytest.fixture
def parsed_citation_doi() -> ParsedCitation:
    """Parsed citation with DOI."""
    return ParsedCitation(
        title="DOI Paper",
        authors=["Author, A."],
        year=2024,
        doi="10.1234/example",
        format="unknown",
        confidence=0.80,
    )


@pytest.fixture
def parsed_citation_arxiv() -> ParsedCitation:
    """Parsed arXiv preprint."""
    return ParsedCitation(
        title="ML Preprint",
        authors=["ML Researcher"],
        year=2024,
        arxiv_id="2401.12345",
        venue="arXiv",
        format="unknown",
        confidence=0.85,
    )


# ============================================================================
# Normalisation Fixtures
# ============================================================================

@pytest.fixture
def normalised_citation() -> NormalisedCitation:
    """Normalised citation for matching."""
    return NormalisedCitation(
        raw_text="Smith, J., Doe, J. (2023). Example Paper.",
        authors_normalised=["smith, j.", "doe, j."],
        title_normalised="example paper",
        year=2023,
        match_key="smith_doe_example_paper_2023",
        doi="10.1234/example",
    )


# ============================================================================
# Domain Term Fixtures
# ============================================================================

@pytest.fixture
def domain_term_concept() -> DomainTerm:
    """Domain term for a concept."""
    return DomainTerm(
        term_id="term_federated_learning",
        term="federated learning",
        domain="machine_learning",
        frequency=45,
        doc_ids=["doc_001", "doc_002", "doc_003"],
        domain_relevance_score=0.95,
        tf_idf_score=8.3,
        term_type="concept",
    )


@pytest.fixture
def domain_term_method() -> DomainTerm:
    """Domain term for a method."""
    return DomainTerm(
        term_id="term_gradient_descent",
        term="gradient descent",
        domain="machine_learning",
        frequency=120,
        doc_ids=["doc_001", "doc_002", "doc_004", "doc_005"],
        domain_relevance_score=0.98,
        tf_idf_score=12.5,
        related_terms=["SGD", "optimization"],
        broader_terms=["optimization algorithm"],
        narrower_terms=["stochastic gradient descent"],
        term_type="method",
    )


@pytest.fixture
def domain_term_tool() -> DomainTerm:
    """Domain term for a tool."""
    return DomainTerm(
        term_id="term_pytorch",
        term="PyTorch",
        domain="machine_learning",
        frequency=65,
        doc_ids=["doc_002", "doc_003", "doc_006"],
        domain_relevance_score=0.85,
        tf_idf_score=9.1,
        term_type="tool",
    )


# ============================================================================
# Batch & Aggregation Fixtures
# ============================================================================

@pytest.fixture
def batch_result_successful(valid_document, valid_reference) -> BatchIngestResult:
    """Successful batch ingestion."""
    return BatchIngestResult(
        primary_docs=[valid_document],
        references={valid_document.doc_id: [valid_reference]},
        domain_terms=[],
        total_citations=150,
        resolved=145,
        unresolved=5,
        downloaded=89,
        pdf_reused=12,
        duration_sec=4.5,
    )


@pytest.fixture
def batch_result_with_errors() -> BatchIngestResult:
    """Batch with errors and warnings."""
    return BatchIngestResult(
        primary_docs=[],
        references={},
        domain_terms=[],
        total_citations=100,
        resolved=85,
        unresolved=15,
        duration_sec=2.3,
        errors=["Failed to resolve 10 citations", "PDF download timeout"],
        warnings=["3 PDFs behind paywall", "2 links stale"],
    )


@pytest.fixture
def revalidation_result_all_updated() -> RevalidationResult:
    """Revalidation with all references updated."""
    return RevalidationResult(
        total=100,
        updated=75,
        unchanged=20,
        failed=5,
        duration_sec=12.5,
        details={
            "metadata_changed": [
                {"ref_id": "ref_001", "fields": ["title", "year"]},
                {"ref_id": "ref_002", "fields": ["authors"]},
            ],
            "failed": [
                {"ref_id": "ref_999", "reason": "API timeout"},
            ],
        }
    )


# ============================================================================
# Persona Context Fixtures
# ============================================================================

@pytest.fixture
def persona_supervisor() -> PersonaContext:
    """Supervisor persona context."""
    return PersonaContext(
        persona="supervisor",
        reference_depth=2,
        include_citations=True,
        include_methodology=True,
        include_datasets=True,
        include_stale_links=True,
        require_verifiable=False,
        stale_link_penalty=0.2,
    )


@pytest.fixture
def persona_assessor() -> PersonaContext:
    """Assessor persona context."""
    return PersonaContext(
        persona="assessor",
        reference_depth=1,
        include_citations=True,
        include_methodology=False,
        include_datasets=False,
        include_stale_links=False,
        require_verifiable=True,
        stale_link_penalty=0.5,
    )


@pytest.fixture
def persona_researcher() -> PersonaContext:
    """Researcher persona context."""
    return PersonaContext(
        persona="researcher",
        reference_depth=3,
        include_citations=True,
        include_methodology=True,
        include_datasets=True,
        include_stale_links=True,
        require_verifiable=False,
        stale_link_penalty=0.3,
    )


# ============================================================================
# Invalid Data Fixtures (for error testing)
# ============================================================================

@pytest.fixture
def invalid_year_too_old() -> Dict[str, Any]:
    """Invalid: year too old."""
    return {
        "doc_id": "test",
        "full_text": "content",
        "year": 1850,  # Too old
    }


@pytest.fixture
def invalid_year_future() -> Dict[str, Any]:
    """Invalid: year in future."""
    return {
        "doc_id": "test",
        "full_text": "content",
        "year": 2050,  # Too far in future
    }


@pytest.fixture
def invalid_score_bounds() -> Dict[str, Any]:
    """Invalid: score out of bounds."""
    return {
        "ref_id": "test",
        "raw_citation": "test",
        "quality_score": 1.5,  # > 1.0
    }


@pytest.fixture
def invalid_negative_count() -> Dict[str, Any]:
    """Invalid: negative count."""
    return {
        "doc_id": "test",
        "full_text": "content",
        "word_count": -5,  # Negative
    }


# ============================================================================
# Factory Function Fixtures
# ============================================================================

@pytest.fixture
def document_from_factory() -> AcademicDocument:
    """Document created via factory."""
    return create_document(
        doc_id="factory_doc_001",
        full_text="Generated content",
        title="Factory Generated",
        authors=["Generated Author"],
        year=2024,
    )


@pytest.fixture
def reference_from_factory() -> Reference:
    """Reference created via factory."""
    return create_reference(
        ref_id="factory_ref_001",
        raw_citation="Factory citation",
        title="Factory Reference",
        authors=["Factory Author"],
        year=2023,
        doi="10.1234/factory",
    )


__all__ = [
    # Document fixtures
    "valid_document_data",
    "valid_document",
    "minimal_document",
    "document_with_sections",
    "document_ingested",
    # Reference fixtures
    "valid_reference_data",
    "valid_reference",
    "unresolved_reference",
    "preprint_reference",
    "oa_reference",
    "reference_with_pdf",
    "stale_reference",
    "cited_reference",
    # Citation edge fixtures
    "direct_citation_edge",
    "indirect_citation_edge",
    "contradicting_edge",
    "extending_edge",
    # Citation parsing fixtures
    "raw_citation_bibliography",
    "raw_citation_inline",
    "parsed_citation_harvard",
    "parsed_citation_doi",
    "parsed_citation_arxiv",
    # Normalisation fixtures
    "normalised_citation",
    # Domain term fixtures
    "domain_term_concept",
    "domain_term_method",
    "domain_term_tool",
    # Batch fixtures
    "batch_result_successful",
    "batch_result_with_errors",
    "revalidation_result_all_updated",
    # Persona fixtures
    "persona_supervisor",
    "persona_assessor",
    "persona_researcher",
    # Invalid data fixtures
    "invalid_year_too_old",
    "invalid_year_future",
    "invalid_score_bounds",
    "invalid_negative_count",
    # Factory fixtures
    "document_from_factory",
    "reference_from_factory",
]
