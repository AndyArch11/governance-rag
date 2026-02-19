"""
Tests for academic data models.

Tests validation, serialisation, and round-trip conversion for all model types.
"""

import json
from datetime import datetime, timezone

import pytest

from scripts.ingest.academic.models import (
    DocumentStatus,
    ReferenceStatus,
    EdgeType,
    OAStatus,
    LinkStatus,
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
    create_document,
    create_reference,
)


# ============================================================================
# AcademicDocument Tests
# ============================================================================

class TestAcademicDocument:
    """Tests for AcademicDocument model."""
    
    def test_create_minimal_document(self, minimal_document):
        """Test creation with minimal required fields."""
        assert minimal_document.doc_id == "acad_test_001"
        assert minimal_document.full_text == "Some content"
        assert minimal_document.status == DocumentStatus.LOADED
        assert minimal_document.word_count == 0
    
    def test_create_full_document(self, valid_document):
        """Test creation with all fields."""
        assert valid_document.title is not None
        assert len(valid_document.authors) == 2
        assert valid_document.year == 2024
        assert valid_document.citation_count == 150
    
    def test_document_serialisation(self, valid_document):
        """Test to_dict() serialisation."""
        doc_dict = valid_document.to_dict()
        assert doc_dict["doc_id"] == valid_document.doc_id
        assert doc_dict["title"] == valid_document.title
        assert doc_dict["status"] == "loaded"
        assert isinstance(doc_dict["ingest_timestamp"], str)
    
    def test_document_json_serialisation(self, valid_document):
        """Test JSON serialisation."""
        json_str = valid_document.to_json()
        json_obj = json.loads(json_str)
        assert json_obj["doc_id"] == valid_document.doc_id
        assert json_obj["title"] == valid_document.title
    
    def test_document_chromadb_metadata(self, valid_document):
        """Test ChromaDB metadata conversion."""
        metadata = valid_document.to_chromadb_metadata()
        assert metadata["doc_id"] == valid_document.doc_id
        assert metadata["doc_type"] == "academic_primary"
        assert metadata["title"] == valid_document.title
        assert "primary_domain" in metadata
        assert "ingest_timestamp" in metadata
    
    def test_document_from_dict(self, valid_document_data):
        """Test reconstruction from dict."""
        doc = AcademicDocument.from_dict(valid_document_data)
        assert doc.doc_id == valid_document_data["doc_id"]
        assert doc.title == valid_document_data["title"]
    
    def test_document_round_trip(self, valid_document):
        """Test serialisation → deserialisation round-trip."""
        doc_dict = valid_document.to_dict()
        recreated = AcademicDocument.from_dict(doc_dict)
        assert recreated.doc_id == valid_document.doc_id
        assert recreated.title == valid_document.title
        assert recreated.year == valid_document.year
    
    def test_year_validation_too_old(self, invalid_year_too_old):
        """Test year validation rejects too-old dates."""
        with pytest.raises(ValueError):
            AcademicDocument(**invalid_year_too_old)
    
    def test_year_validation_future(self, invalid_year_future):
        """Test year validation rejects future dates."""
        with pytest.raises(ValueError):
            AcademicDocument(**invalid_year_future)
    
    def test_reference_count_validation(self):
        """Test reference count cannot exceed citation count."""
        with pytest.raises(ValueError, match="Reference count cannot exceed citation count"):
            AcademicDocument(
                doc_id="test",
                full_text="content",
                citation_count=50,
                reference_count=60,  # > citation_count
            )
    
    def test_document_sections(self, document_with_sections):
        """Test document with sections."""
        assert len(document_with_sections.sections) == 4
        assert "introduction" in document_with_sections.sections
        assert "methods" in document_with_sections.sections
    
    def test_document_status_transitions(self):
        """Test document status transitions."""
        doc = AcademicDocument(
            doc_id="status_test",
            full_text="content",
            status=DocumentStatus.LOADED,
        )
        assert doc.status == DocumentStatus.LOADED
        
        # Manually update status
        doc.status = DocumentStatus.PARSED
        assert doc.status == DocumentStatus.PARSED


# ============================================================================
# Reference Tests
# ============================================================================

class TestReference:
    """Tests for Reference model."""
    
    def test_create_resolved_reference(self, valid_reference):
        """Test creation of resolved reference."""
        assert valid_reference.resolved is True
        assert valid_reference.status == ReferenceStatus.RESOLVED
        assert valid_reference.resolved_at is not None
    
    def test_create_unresolved_reference(self, unresolved_reference):
        """Test creation of unresolved reference."""
        assert unresolved_reference.resolved is False
        assert unresolved_reference.status == ReferenceStatus.RAW
    
    def test_reference_doi_identifier(self, valid_reference):
        """Test DOI as identifier."""
        assert valid_reference.doi == "10.1234/example"
        assert valid_reference.ref_id == "ref_10.1234/example"
    
    def test_reference_arxiv_identifier(self, preprint_reference):
        """Test arXiv identifier."""
        assert preprint_reference.arxiv_id == "2401.01234"
        assert preprint_reference.reference_type == "preprint"
    
    def test_reference_open_access(self, oa_reference):
        """Test OA reference fields."""
        assert oa_reference.oa_available is True
        assert oa_reference.oa_status == OAStatus.GOLD
        assert oa_reference.oa_url is not None
        assert oa_reference.quality_score == 1.0
    
    def test_reference_pdf_tracking(self, reference_with_pdf):
        """Test PDF tracking fields."""
        assert reference_with_pdf.pdf_downloaded is True
        assert reference_with_pdf.pdf_local_path is not None
        assert reference_with_pdf.pdf_file_hash is not None
        assert reference_with_pdf.accessed_at is not None
    
    def test_reference_link_tracking(self, stale_reference):
        """Test link availability tracking."""
        assert stale_reference.link_status == LinkStatus.STALE_404
        assert stale_reference.link_checked_at is not None
    
    def test_reference_citation_count(self, cited_reference):
        """Test citation tracking."""
        assert len(cited_reference.doc_ids) == 3
        assert cited_reference.citation_count == 25
    
    def test_reference_serialisation(self, valid_reference):
        """Test to_dict() serialisation."""
        ref_dict = valid_reference.to_dict()
        assert ref_dict["ref_id"] == valid_reference.ref_id
        assert ref_dict["doi"] == valid_reference.doi
        assert ref_dict["resolved"] is True
        assert ref_dict["status"] == "resolved"
    
    def test_reference_json_serialisation(self, valid_reference):
        """Test JSON serialisation."""
        json_str = valid_reference.to_json()
        json_obj = json.loads(json_str)
        assert json_obj["ref_id"] == valid_reference.ref_id
    
    def test_reference_chromadb_metadata(self, valid_reference):
        """Test ChromaDB metadata conversion."""
        metadata = valid_reference.to_chromadb_metadata()
        assert metadata["doc_type"] == "academic_reference"
        assert metadata["ref_id"] == valid_reference.ref_id
        assert metadata["resolved"] is True
        assert "oa_available" in metadata
    
    def test_reference_round_trip(self, valid_reference):
        """Test serialisation → deserialisation round-trip."""
        ref_dict = valid_reference.to_dict()
        recreated = Reference.from_dict(ref_dict)
        assert recreated.ref_id == valid_reference.ref_id
        assert recreated.doi == valid_reference.doi
    
    def test_year_validation(self):
        """Test reference year validation."""
        with pytest.raises(ValueError):
            Reference(
                ref_id="test",
                raw_citation="test",
                year=1800,  # Too old
            )
    
    def test_quality_score_bounds(self, invalid_score_bounds):
        """Test quality score validation."""
        with pytest.raises(ValueError):
            Reference(**invalid_score_bounds)
    
    def test_resolved_status_auto_update(self):
        """Test resolved status auto-updates fields."""
        ref = Reference(
            ref_id="test",
            raw_citation="test",
            resolved=True,
        )
        # Should auto-update resolved_at and status
        assert ref.resolved_at is not None
        assert ref.status in [ReferenceStatus.RESOLVED, ReferenceStatus.RAW]


# ============================================================================
# Citation Edge Tests
# ============================================================================

class TestCitationEdge:
    """Tests for CitationEdge model."""
    
    def test_direct_edge(self, direct_citation_edge):
        """Test direct citation edge."""
        assert direct_citation_edge.edge_type == EdgeType.DIRECT
        assert direct_citation_edge.depth == 1
        assert direct_citation_edge.is_inline is True
    
    def test_indirect_edge(self, indirect_citation_edge):
        """Test indirect citation edge."""
        assert indirect_citation_edge.edge_type == EdgeType.INDIRECT
        assert indirect_citation_edge.depth == 2
    
    def test_contradicting_edge(self, contradicting_edge):
        """Test contradicting relationship."""
        assert contradicting_edge.edge_type == EdgeType.CONTRADICTS
        assert contradicting_edge.cited_as == "critique"
    
    def test_extending_edge(self, extending_edge):
        """Test extending methodology relationship."""
        assert extending_edge.edge_type == EdgeType.EXTENDS
    
    def test_edge_serialisation(self, direct_citation_edge):
        """Test edge serialisation."""
        edge_dict = direct_citation_edge.to_dict()
        assert edge_dict["from_id"] == direct_citation_edge.from_id
        assert edge_dict["edge_type"] == "direct"
        assert isinstance(edge_dict["created_at"], str)
    
    def test_edge_strength_score(self):
        """Test strength score validation."""
        edge = CitationEdge(
            from_id="a",
            to_id="b",
            strength_score=0.85,
        )
        assert edge.strength_score == 0.85


# ============================================================================
# Citation Parsing Tests
# ============================================================================

class TestCitationParsing:
    """Tests for citation parsing models."""
    
    def test_raw_citation_bibliography(self, raw_citation_bibliography):
        """Test raw bibliography citation."""
        assert raw_citation_bibliography.position == "bibliography"
        assert len(raw_citation_bibliography.raw_text) > 0
    
    def test_raw_citation_inline(self, raw_citation_inline):
        """Test raw inline citation."""
        assert raw_citation_inline.position == "inline"
        assert raw_citation_inline.position_offset > 0
    
    def test_parsed_citation_harvard(self, parsed_citation_harvard):
        """Test Harvard parsed citation."""
        assert parsed_citation_harvard.format == "harvard"
        assert len(parsed_citation_harvard.authors) > 0
        assert parsed_citation_harvard.year == 2023
    
    def test_parsed_citation_doi(self, parsed_citation_doi):
        """Test parsed citation with DOI."""
        assert parsed_citation_doi.doi == "10.1234/example"
        assert parsed_citation_doi.confidence == 0.80
    
    def test_parsed_citation_arxiv(self, parsed_citation_arxiv):
        """Test parsed arXiv citation."""
        assert parsed_citation_arxiv.arxiv_id == "2401.12345"
        assert parsed_citation_arxiv.venue == "arXiv"
    
    def test_normalised_citation(self, normalised_citation):
        """Test normalised citation."""
        assert "smith" in normalised_citation.authors_normalised[0]
        assert normalised_citation.title_normalised == "example paper"
        assert normalised_citation.match_key == "smith_doe_example_paper_2023"


# ============================================================================
# Domain Term Tests
# ============================================================================

class TestDomainTerm:
    """Tests for DomainTerm model."""
    
    def test_domain_term_concept(self, domain_term_concept):
        """Test domain term concept."""
        assert domain_term_concept.term_type == "concept"
        assert domain_term_concept.domain_relevance_score == 0.95
        assert domain_term_concept.frequency == 45
    
    def test_domain_term_method(self, domain_term_method):
        """Test domain term method."""
        assert domain_term_method.term_type == "method"
        assert len(domain_term_method.related_terms) > 0
        assert len(domain_term_method.broader_terms) > 0
        assert len(domain_term_method.narrower_terms) > 0
    
    def test_domain_term_tool(self, domain_term_tool):
        """Test domain term tool."""
        assert domain_term_tool.term_type == "tool"
        assert domain_term_tool.term == "PyTorch"
    
    def test_domain_term_serialisation(self, domain_term_concept):
        """Test domain term serialisation."""
        term_dict = domain_term_concept.to_dict()
        assert term_dict["term_id"] == domain_term_concept.term_id
        assert term_dict["frequency"] == 45


# ============================================================================
# Batch & Aggregation Tests
# ============================================================================

class TestBatchResults:
    """Tests for batch ingestion results."""
    
    def test_batch_successful(self, batch_result_successful):
        """Test successful batch result."""
        assert len(batch_result_successful.primary_docs) == 1
        assert batch_result_successful.resolved == 145
        assert batch_result_successful.unresolved == 5
    
    def test_batch_with_errors(self, batch_result_with_errors):
        """Test batch with errors."""
        assert len(batch_result_with_errors.errors) > 0
        assert len(batch_result_with_errors.warnings) > 0
    
    def test_batch_summary(self, batch_result_successful):
        """Test batch summary output."""
        summary = batch_result_successful.summary()
        assert "Batch Ingest Summary" in summary
        assert "150" in summary  # total citations
    
    def test_batch_validation(self):
        """Test batch result validation."""
        with pytest.raises(ValueError, match="Resolved count cannot exceed"):
            BatchIngestResult(
                primary_docs=[],
                references={},
                total_citations=100,
                resolved=150,  # > total
            )


class TestRevalidationResult:
    """Tests for revalidation results."""
    
    def test_revalidation_all_updated(self, revalidation_result_all_updated):
        """Test revalidation with updates."""
        assert revalidation_result_all_updated.total == 100
        assert revalidation_result_all_updated.updated == 75
        assert revalidation_result_all_updated.unchanged == 20
        assert revalidation_result_all_updated.failed == 5
    
    def test_revalidation_summary(self, revalidation_result_all_updated):
        """Test revalidation summary."""
        summary = revalidation_result_all_updated.summary()
        assert "Revalidation Complete" in summary
        assert "100" in summary  # total
    
    def test_revalidation_validation(self):
        """Test revalidation count validation."""
        with pytest.raises(ValueError, match="must equal Total"):
            RevalidationResult(
                total=100,
                updated=50,
                unchanged=30,
                failed=15,  # 50+30+15 = 95 ≠ 100
            )


# ============================================================================
# Persona Context Tests
# ============================================================================

class TestPersonaContext:
    """Tests for PersonaContext model."""
    
    def test_persona_supervisor(self, persona_supervisor):
        """Test supervisor persona."""
        assert persona_supervisor.persona == "supervisor"
        assert persona_supervisor.reference_depth == 2
        assert persona_supervisor.include_stale_links is True
    
    def test_persona_assessor(self, persona_assessor):
        """Test assessor persona."""
        assert persona_assessor.persona == "assessor"
        assert persona_assessor.require_verifiable is True
        assert persona_assessor.include_stale_links is False
    
    def test_persona_researcher(self, persona_researcher):
        """Test researcher persona."""
        assert persona_researcher.persona == "researcher"
        assert persona_researcher.reference_depth == 3
    
    def test_applies_to_ref_supervisor(self, persona_supervisor, cited_reference):
        """Test supervisor filtering (foundational refs)."""
        cited_reference.citation_count = 25  # > 10
        assert persona_supervisor.applies_to_ref(cited_reference) is True
        
        cited_reference.citation_count = 5  # < 10
        assert persona_supervisor.applies_to_ref(cited_reference) is False
    
    def test_applies_to_ref_assessor_stale(self, persona_assessor, stale_reference):
        """Test assessor filters stale links."""
        assert persona_assessor.applies_to_ref(stale_reference) is False
    
    def test_applies_to_ref_assessor_available(self, persona_assessor, valid_reference):
        """Test assessor accepts available links."""
        valid_reference.link_status = LinkStatus.AVAILABLE
        assert persona_assessor.applies_to_ref(valid_reference) is True
    
    def test_compute_ranking_penalty_available(self, persona_supervisor, valid_reference):
        """Test no penalty for available links."""
        valid_reference.link_status = LinkStatus.AVAILABLE
        penalty = persona_supervisor.compute_ranking_penalty(valid_reference)
        assert penalty == 0.0
    
    def test_compute_ranking_penalty_404(self, persona_supervisor, stale_reference):
        """Test full penalty for 404."""
        penalty = persona_supervisor.compute_ranking_penalty(stale_reference)
        assert penalty == persona_supervisor.stale_link_penalty * 1.0
    
    def test_compute_ranking_penalty_timeout(self, persona_supervisor):
        """Test reduced penalty for timeout."""
        ref = Reference(
            ref_id="test",
            raw_citation="test",
            link_status=LinkStatus.STALE_TIMEOUT,
        )
        penalty = persona_supervisor.compute_ranking_penalty(ref)
        assert penalty == persona_supervisor.stale_link_penalty * 0.75
    
    def test_compute_ranking_penalty_moved(self, persona_supervisor):
        """Test minimal penalty for moved."""
        ref = Reference(
            ref_id="test",
            raw_citation="test",
            link_status=LinkStatus.STALE_MOVED,
        )
        penalty = persona_supervisor.compute_ranking_penalty(ref)
        assert penalty == persona_supervisor.stale_link_penalty * 0.25


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_document_factory(self, document_from_factory):
        """Test document factory function."""
        assert document_from_factory.doc_id == "factory_doc_001"
        assert document_from_factory.title == "Factory Generated"
        assert document_from_factory.year == 2024
    
    def test_create_reference_factory(self, reference_from_factory):
        """Test reference factory function."""
        assert reference_from_factory.ref_id == "factory_ref_001"
        assert reference_from_factory.doi == "10.1234/factory"
        assert reference_from_factory.year == 2023


# ============================================================================
# Type Safety Tests
# ============================================================================

class TestTypeSafety:
    """Tests for type validation and safety."""
    
    def test_enum_value_serialisation(self, valid_document):
        """Test enum values are serialised as strings."""
        doc_dict = valid_document.to_dict()
        assert isinstance(doc_dict["status"], str)
        assert doc_dict["status"] == "loaded"
    
    def test_datetime_serialisation(self, valid_reference):
        """Test datetime serialisation to ISO format."""
        ref_dict = valid_reference.to_dict()
        if ref_dict["resolved_at"]:
            assert isinstance(ref_dict["resolved_at"], str)
            assert "T" in ref_dict["resolved_at"]  # ISO format
    
    def test_json_array_fields(self, valid_document):
        """Test JSON array fields are properly handled."""
        doc_dict = valid_document.to_dict()
        assert isinstance(doc_dict["authors"], list)
        assert isinstance(doc_dict["keywords"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
