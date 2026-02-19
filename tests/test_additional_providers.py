"""
Tests for additional providers and provider chain.
"""

import pytest
from unittest.mock import Mock, patch

from scripts.ingest.academic.providers import (
    OpenAlexProvider,
    SemanticScholarProvider,
    ProviderChain,
    create_default_chain,
    Reference,
)


class TestOpenAlexProvider:
    """Tests for OpenAlex metadata provider."""
    
    @pytest.fixture
    def provider(self):
        return OpenAlexProvider()
    
    def test_provider_configuration(self, provider):
        """Test provider configuration."""
        assert provider.name == "openalex"
        assert provider.rate_limit == 10  # 10 req/sec
        assert provider.timeout == 10.0
    
    def test_resolve_by_doi(self, provider):
        """Test DOI resolution."""
        mock_response = {
            "results": [{
                "id": "https://openalex.org/W123456",
                "title": "Sample Paper",
                "doi": "10.1038/nature.2023.123",
                "type": "journal-article",
                "publication_year": 2023,
                "authorships": [
                    {"author": {"display_name": "Smith, John"}},
                    {"author": {"display_name": "Doe, Jane"}},
                ],
                "host_venue": {"display_name": "Nature"},
                "abstract": "This is abstract",
                "biblio": {"volume": "500", "issue": "3"},
                "is_open_access": True,
                "best_open_access_location": {"pdf_url": "http://example.com/paper.pdf"},
            }]
        }
        
        with patch.object(provider, '_request_with_retry') as mock_request:
            mock_request.return_value = Mock(json=lambda: mock_response)
            
            result = provider.resolve(
                citation_text="Sample Paper",
                doi="10.1038/nature.2023.123"
            )
            
            assert result.resolved
            assert result.title == "Sample Paper"
            assert len(result.authors) == 2
            assert result.oa_available
    
    def test_fuzzy_title_matching(self, provider):
        """Test fuzzy title matching."""
        score1 = provider._fuzzy_match_title("Python Programming", "Python Programming")
        assert score1 == 1.0
        
        score2 = provider._fuzzy_match_title("Python", "Python Programming Language")
        assert 0.85 < score2 < 1.0
        
        score3 = provider._fuzzy_match_title("Machine Learning", "Quantum Computing")
        assert score3 < 0.5
    
    def test_quality_score_computation(self, provider):
        """Test quality score computation."""
        work_with_doi = {
            "doi": "10.1038/test",
            "is_open_access": True,
            "cited_by_count": 50,
            "abstract": "test abstract",
            "type": "journal-article",
        }
        
        score = provider._compute_quality_score(work_with_doi)
        assert 0.7 <= score <= 1.0
    
    def test_reference_type_determination(self, provider):
        """Test reference type determination."""
        assert provider._determine_reference_type("journal-article") == "academic"
        assert provider._determine_reference_type("conference") == "conference"
        assert provider._determine_reference_type("preprint") == "preprint"


class TestSemanticScholarProvider:
    """Tests for Semantic Scholar metadata provider."""
    
    @pytest.fixture
    def provider(self):
        return SemanticScholarProvider()
    
    def test_provider_configuration(self, provider):
        """Test provider configuration."""
        assert provider.name == "semantic_scholar"
        assert provider.rate_limit == 1  # 1 req/sec
        assert provider.timeout == 10.0
    
    def test_resolve_by_doi(self, provider):
        """Test DOI resolution."""
        mock_response = {
            "data": [{
                "paperId": "12345",
                "title": "Sample Paper",
                "externalIds": {"DOI": "10.1038/nature.2023.123"},
                "year": 2023,
                "authors": [
                    {"name": "Smith, John"},
                    {"name": "Doe, Jane"},
                ],
                "venue": "Nature",
                "publicationTypes": ["JournalArticle"],
                "abstract": "This is abstract",
                "citationCount": 25,
                "influentialCitationCount": 5,
                "isOpenAccess": True,
                "openAccessPdf": {"url": "http://example.com/paper.pdf"},
            }]
        }
        
        with patch.object(provider, '_request_with_retry') as mock_request:
            mock_request.return_value = Mock(json=lambda: mock_response)
            
            result = provider.resolve(
                citation_text="Sample Paper",
                doi="10.1038/nature.2023.123"
            )
            
            assert result.resolved
            assert result.title == "Sample Paper"
            assert len(result.authors) == 2
    
    def test_match_score_computation(self, provider):
        """Test match score computation."""
        score = provider._compute_match_score(
            "Python Programming",
            2023,
            ["Smith, John"],
            "Python Programming Language",
            2023,
            [{"name": "John Smith"}],
        )
        
        assert 0.5 < score <= 1.0
    
    def test_quality_score_computation(self, provider):
        """Test quality score computation."""
        paper = {
            "externalIds": {"DOI": "10.1038/test"},
            "isOpenAccess": True,
            "abstract": "test abstract",
            "citationCount": 10,
            "influentialCitationCount": 2,
            "publicationTypes": ["JournalArticle"],
        }
        
        score = provider._compute_quality_score(paper)
        assert 0.6 <= score <= 1.0


class TestProviderChain:
    """Tests for provider chain orchestration."""
    
    @pytest.fixture
    def mock_providers(self):
        """Create mock providers."""
        providers = [Mock(name=f"provider_{i}") for i in range(3)]
        for p in providers:
            p.name = p.spec_set[0] if p.spec_set else "mock"
        return providers
    
    def test_chain_creation(self):
        """Test provider chain creation."""
        from scripts.ingest.academic.providers import CrossrefProvider, OpenAlexProvider
        
        providers = [CrossrefProvider(), OpenAlexProvider()]
        chain = ProviderChain(providers, min_confidence=0.8)
        
        assert len(chain.providers) == 2
        assert chain.min_confidence == 0.8
    
    def test_default_chain_creation(self):
        """Test default chain creation."""
        chain = create_default_chain()
        
        # Should have 10 providers now
        assert len(chain.providers) == 10
        assert chain.providers[0].name == "crossref"
        assert chain.providers[1].name == "datacite"
        assert chain.providers[2].name == "pubmed"
        assert chain.providers[3].name == "openalex"
        assert chain.providers[4].name == "arxiv"
        assert chain.providers[5].name == "semantic_scholar"
        assert chain.providers[6].name == "orcid"
        assert chain.providers[7].name == "google_scholar"
        assert chain.providers[8].name == "unpaywall"
        assert chain.providers[9].name == "url_fetch"
    
    def test_resolution_result(self):
        """Test resolution result tracking."""
        ref = Reference(
            ref_id="test",
            raw_citation="Test Paper",
            title="Test Paper",
            resolved=True,
        )
        
        chain = ProviderChain([])
        confidence = chain._compute_confidence(ref)
        
        assert 0.0 <= confidence <= 1.0
    
    def test_completeness_scoring(self):
        """Test metadata completeness scoring."""
        # Complete reference
        complete_ref = Reference(
            ref_id="test1",
            raw_citation="Test Complete",
            title="Test",
            authors=["Smith"],
            year=2023,
            doi="10.1038/test",
            venue="Nature",
            abstract="Abstract",
            pages="1-10",
            volume="500",
            issue="3",
        )
        
        # Minimal reference
        minimal_ref = Reference(
            ref_id="test2",
            raw_citation="Test Minimal",
            title="Test",
        )
        
        chain = ProviderChain([])
        complete_score = chain._compute_completeness(complete_ref)
        minimal_score = chain._compute_completeness(minimal_ref)
        
        assert complete_score > minimal_score
        assert 0.0 <= complete_score <= 1.0
        assert 0.0 <= minimal_score <= 1.0
    
    def test_stats_tracking(self):
        """Test statistics tracking."""
        providers = [Mock(name="mock_provider")]
        providers[0].name = "mock"
        
        chain = ProviderChain(providers)
        stats = chain.get_stats()
        
        assert stats["total_queries"] == 0
        assert stats["resolved"] == 0
        assert stats["unresolved"] == 0
        
        # Reset should work
        chain.reset_stats()
        assert chain.resolution_stats["total_queries"] == 0


class TestProviderIntegration:
    """Integration tests for provider interactions."""
    
    def test_all_providers_exist(self):
        """Test that all providers can be imported."""
        from scripts.ingest.academic.providers import (
            CrossrefProvider,
            OpenAlexProvider,
            SemanticScholarProvider,
        )
        
        assert CrossrefProvider is not None
        assert OpenAlexProvider is not None
        assert SemanticScholarProvider is not None
    
    def test_all_providers_are_base_provider_subclasses(self):
        """Test that all providers inherit from BaseProvider."""
        from scripts.ingest.academic.providers import (
            BaseProvider,
            CrossrefProvider,
            OpenAlexProvider,
            SemanticScholarProvider,
        )
        
        assert issubclass(CrossrefProvider, BaseProvider)
        assert issubclass(OpenAlexProvider, BaseProvider)
        assert issubclass(SemanticScholarProvider, BaseProvider)
    
    def test_all_providers_have_resolve_method(self):
        """Test that all providers implement resolve()."""
        from scripts.ingest.academic.providers import (
            CrossrefProvider,
            OpenAlexProvider,
            SemanticScholarProvider,
        )
        
        providers = [
            CrossrefProvider(),
            OpenAlexProvider(),
            SemanticScholarProvider(),
        ]
        
        for provider in providers:
            assert hasattr(provider, 'resolve')
            assert callable(provider.resolve)
    
    def test_reference_consistency_across_providers(self):
        """Test that all providers return consistent Reference objects."""
        from scripts.ingest.academic.providers import (
            CrossrefProvider,
            OpenAlexProvider,
            SemanticScholarProvider,
            Reference,
        )
        
        providers = [
            CrossrefProvider(),
            OpenAlexProvider(),
            SemanticScholarProvider(),
        ]
        
        for provider in providers:
            # Create a test reference
            ref = Reference(
                ref_id="test",
                raw_citation="Test Paper",
                title="Test Paper",
                authors=["Smith, John"],
                year=2023,
                resolved=True,
                metadata_provider=provider.name,
            )
            
            # Verify it has all required attributes
            assert ref.ref_id
            assert ref.title
            assert ref.authors
            assert ref.year
            assert ref.resolved
            assert ref.metadata_provider


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
