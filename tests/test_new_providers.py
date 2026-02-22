"""
Tests for ArxivProvider, PubMedProvider, and DataCiteProvider.
"""

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest

from scripts.ingest.academic.providers import (
    ArxivProvider,
    DataCiteProvider,
    PubMedProvider,
    ReferenceStatus,
)


class TestArxivProvider:
    """Tests for arXiv provider."""

    def test_initialisation(self):
        """Test provider initialisation."""
        provider = ArxivProvider()

        assert provider.name == "arxiv"
        assert provider.rate_limit == 3
        assert provider.timeout == 10.0

    def test_extract_arxiv_id(self):
        """Test arXiv ID extraction from citation."""
        provider = ArxivProvider()

        # Standard format
        assert provider._extract_arxiv_id("arXiv:1234.5678") == "1234.5678"

        # With version
        assert provider._extract_arxiv_id("arXiv:1234.5678v2") == "1234.5678v2"

        # URL format
        assert provider._extract_arxiv_id("https://arxiv.org/abs/2103.12345") == "2103.12345"

        # No arXiv ID
        assert provider._extract_arxiv_id("Some random text") is None

    @patch("scripts.ingest.academic.providers.arxiv.ArxivProvider._request_with_retry")
    def test_resolve_by_arxiv_id(self, mock_request):
        """Test resolution by arXiv ID."""
        provider = ArxivProvider()

        # Mock XML response
        mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/2103.12345v1</id>
                <title>Test Paper Title</title>
                <author><name>John Doe</name></author>
                <author><name>Jane Smith</name></author>
                <summary>This is an abstract.</summary>
                <published>2021-03-23T00:00:00Z</published>
                <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.AI"/>
            </entry>
        </feed>
        """

        mock_response = Mock()
        mock_response.text = mock_xml
        mock_request.return_value = mock_response

        ref = provider._resolve_by_arxiv_id("2103.12345")

        assert ref.resolved is True
        assert ref.title == "Test Paper Title"
        assert ref.authors == ["John Doe", "Jane Smith"]
        assert ref.year == 2021
        assert ref.abstract == "This is an abstract."
        assert ref.venue == "arXiv:cs.AI"
        assert ref.venue_type == "preprint"
        assert ref.reference_type == "preprint"
        assert ref.oa_available is True
        assert "arxiv.org/pdf" in ref.oa_url

    @patch("scripts.ingest.academic.providers.arxiv.ArxivProvider._request_with_retry")
    def test_resolve_by_search(self, mock_request):
        """Test resolution by title search."""
        provider = ArxivProvider()

        # Mock XML response with multiple entries
        mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/1234.5678</id>
                <title>Deep Learning Tutorial</title>
                <author><name>Author One</name></author>
                <summary>Great tutorial</summary>
                <published>2020-01-01T00:00:00Z</published>
                <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.LG"/>
            </entry>
        </feed>
        """

        mock_response = Mock()
        mock_response.text = mock_xml
        mock_request.return_value = mock_response

        ref = provider._resolve_by_search(
            "Deep Learning Tutorial", year=2020, authors=["Author One"]
        )

        assert ref.resolved is True
        assert ref.title == "Deep Learning Tutorial"
        assert ref.year == 2020

    def test_fuzzy_match_title(self):
        """Test fuzzy title matching."""
        provider = ArxivProvider()

        # Exact match
        assert provider._fuzzy_match_title("Test Title", "Test Title") == 1.0

        # Substring match
        score = provider._fuzzy_match_title("Test", "Test Title")
        assert score == 0.9

        # Similar match
        score = provider._fuzzy_match_title("Deep Learning", "Deep Learning Tutorial")
        assert score == 0.9

        # Different
        score = provider._fuzzy_match_title("Quantum Computing", "Machine Learning")
        assert score < 0.5

    def test_quality_score_computation(self):
        """Test quality score computation."""
        provider = ArxivProvider()

        # Create mock XML entry
        xml = """<entry xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
            <summary>Abstract text</summary>
            <author><name>A1</name></author>
            <author><name>A2</name></author>
            <author><name>A3</name></author>
            <arxiv:doi>10.1234/test</arxiv:doi>
        </entry>
        """

        entry = ET.fromstring(xml)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

        score = provider._compute_quality_score(entry, ns)

        # 0.75 base + 0.1 DOI + 0.05 abstract + 0.05 authors + 0.05 OA = 1.0
        assert score == 1.0


class TestPubMedProvider:
    """Tests for PubMed provider."""

    def test_initialisation(self):
        """Test provider initialisation."""
        # Without API key
        provider = PubMedProvider()
        assert provider.name == "pubmed"
        assert provider.rate_limit == 3
        assert provider.api_key is None

        # With API key
        provider_with_key = PubMedProvider(api_key="test_key")
        assert provider_with_key.rate_limit == 10
        assert provider_with_key.api_key == "test_key"

    def test_extract_pmid(self):
        """Test PMID extraction from citation."""
        provider = PubMedProvider()

        # Standard format
        assert provider._extract_pmid("PMID: 12345678") == "12345678"

        # No space
        assert provider._extract_pmid("PMID:12345678") == "12345678"

        # URL format
        assert provider._extract_pmid("pubmed/12345678") == "12345678"

        # No PMID
        assert provider._extract_pmid("Some random text") is None

    @patch("scripts.ingest.academic.providers.pubmed.PubMedProvider._request_with_retry")
    def test_resolve_by_pmid(self, mock_request):
        """Test resolution by PMID."""
        provider = PubMedProvider()

        # Mock XML response
        mock_xml = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <ArticleTitle>Test Medical Paper</ArticleTitle>
                        <AuthorList>
                            <Author><LastName>Doe</LastName><ForeName>John</ForeName></Author>
                        </AuthorList>
                        <Abstract><AbstractText>Medical abstract</AbstractText></Abstract>
                        <Journal><Title>Nature Medicine</Title></Journal>
                    </Article>
                    <ArticleIdList>
                        <ArticleId IdType="doi">10.1234/test</ArticleId>
                        <ArticleId IdType="pmc">PMC12345</ArticleId>
                    </ArticleIdList>
                </MedlineCitation>
                <PubmedData>
                    <ArticleIdList/>
                </PubmedData>
            </PubmedArticle>
        </PubmedArticleSet>
        """

        mock_response = Mock()
        mock_response.text = mock_xml
        mock_request.return_value = mock_response

        ref = provider._resolve_by_pmid("12345678")

        assert ref.resolved is True
        assert ref.title == "Test Medical Paper"
        assert ref.authors == ["John Doe"]
        assert ref.abstract == "Medical abstract"
        assert ref.venue == "Nature Medicine"
        assert ref.doi == "10.1234/test"
        assert ref.oa_available is True
        assert "pmc/articles/PMC12345" in ref.oa_url

    @patch("scripts.ingest.academic.providers.pubmed.PubMedProvider._request_with_retry")
    def test_resolve_by_doi(self, mock_request):
        """Test resolution by DOI."""
        provider = PubMedProvider()

        # Mock search response
        mock_search_response = Mock()
        mock_search_response.json.return_value = {"esearchresult": {"idlist": ["12345678"]}}

        # Mock fetch response
        mock_xml = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <ArticleTitle>Test Paper</ArticleTitle>
                        <AuthorList></AuthorList>
                        <Journal><Title>Test Journal</Title></Journal>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """

        mock_fetch_response = Mock()
        mock_fetch_response.text = mock_xml

        # Configure mock to return different responses
        mock_request.side_effect = [mock_search_response, mock_fetch_response]

        ref = provider._resolve_by_doi("10.1234/test")

        assert ref.resolved is True
        assert ref.title == "Test Paper"

    def test_fuzzy_match_title(self):
        """Test fuzzy title matching."""
        provider = PubMedProvider()

        # Exact match
        assert provider._fuzzy_match_title("Cancer Research", "Cancer Research") == 1.0

        # Close match
        score = provider._fuzzy_match_title("Cancer Research", "Cancer Research Methods")
        assert score == 0.9

    def test_quality_score_computation(self):
        """Test quality score computation."""
        provider = PubMedProvider()

        # Create mock article
        xml = """<PubmedArticle>
            <MedlineCitation>
                <ArticleIdList>
                    <ArticleId IdType="doi">10.1234/test</ArticleId>
                </ArticleIdList>
            </MedlineCitation>
        </PubmedArticle>
        """

        article = ET.fromstring(xml)

        # With PMC (OA)
        score = provider._compute_quality_score(article, "PMC12345")
        assert score == 1.0  # 0.9 base + 0.05 DOI + 0.05 PMC

        # Without PMC
        score = provider._compute_quality_score(article, None)
        assert abs(score - 0.95) < 0.01  # 0.9 base + 0.05 DOI


class TestDataCiteProvider:
    """Tests for DataCite provider."""

    def test_initialisation(self):
        """Test provider initialisation."""
        provider = DataCiteProvider()

        assert provider.name == "datacite"
        assert provider.rate_limit == 5
        assert provider.timeout == 10.0

    @patch("scripts.ingest.academic.providers.datacite.DataCiteProvider._request_with_retry")
    def test_resolve_by_doi(self, mock_request):
        """Test resolution by DOI."""
        provider = DataCiteProvider()

        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "attributes": {
                    "doi": "10.5061/dryad.12345",
                    "titles": [{"title": "Test Dataset"}],
                    "creators": [{"givenName": "John", "familyName": "Doe"}],
                    "publicationYear": 2022,
                    "descriptions": [
                        {"descriptionType": "Abstract", "description": "Dataset description"}
                    ],
                    "publisher": "Dryad",
                    "types": {"resourceTypeGeneral": "Dataset", "resourceType": "Research Data"},
                    "url": "https://datadryad.org/12345",
                    "rightsList": [{"rights": "CC0 1.0", "rightsIdentifier": "cc0-1.0"}],
                }
            }
        }
        mock_request.return_value = mock_response

        ref = provider._resolve_by_doi("10.5061/dryad.12345")

        assert ref.resolved is True
        assert ref.title == "Test Dataset"
        assert ref.authors == ["John Doe"]
        assert ref.year == 2022
        assert ref.abstract == "Dataset description"
        assert "Research Data" in ref.venue
        assert ref.reference_type == "dataset"
        assert ref.venue_type == "repository"
        assert ref.doi == "10.5061/dryad.12345"
        assert ref.oa_available is True

    @patch("scripts.ingest.academic.providers.datacite.DataCiteProvider._request_with_retry")
    def test_resolve_by_search(self, mock_request):
        """Test resolution by title search."""
        provider = DataCiteProvider()

        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "attributes": {
                        "doi": "10.5061/dryad.test",
                        "titles": [{"title": "Climate Data 2020"}],
                        "creators": [{"name": "Research Team"}],
                        "publicationYear": 2020,
                        "descriptions": [],
                        "publisher": "Zenodo",
                        "types": {"resourceTypeGeneral": "Dataset"},
                        "url": "https://zenodo.org/test",
                        "rightsList": [],
                    }
                }
            ]
        }
        mock_request.return_value = mock_response

        ref = provider._resolve_by_search("Climate Data 2020", year=2020)

        assert ref.resolved is True
        assert ref.title == "Climate Data 2020"
        assert ref.year == 2020

    def test_determine_reference_type(self):
        """Test reference type determination."""
        provider = DataCiteProvider()

        assert provider._determine_reference_type("Dataset") == "dataset"
        assert provider._determine_reference_type("Software") == "software"
        assert provider._determine_reference_type("DataPaper") == "academic"
        assert provider._determine_reference_type("ConferencePaper") == "conference"
        assert provider._determine_reference_type("Preprint") == "preprint"
        assert provider._determine_reference_type("Unknown") == "other"

    def test_determine_venue_type(self):
        """Test venue type determination."""
        provider = DataCiteProvider()

        assert provider._determine_venue_type("Dataset") == "repository"
        assert provider._determine_venue_type("Software") == "repository"
        assert provider._determine_venue_type("DataPaper") == "journal"
        assert provider._determine_venue_type("ConferencePaper") == "conference"

    def test_quality_score_computation(self):
        """Test quality score computation."""
        provider = DataCiteProvider()

        attributes = {
            "descriptions": [{"description": "test"}],
            "creators": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
            "fundingReferences": [{"funderName": "NSF"}],
        }

        score = provider._compute_quality_score(attributes, "Dataset")
        # 0.85 base + 0.05 desc + 0.05 creators + 0.05 funding = 1.0
        assert score == 1.0

        # Minimal attributes
        minimal = {}
        score = provider._compute_quality_score(minimal, "Text")
        assert score == 0.8  # Base score only


class TestProviderIntegration:
    """Integration tests for all new providers."""

    def test_all_providers_importable(self):
        """Test that all providers can be imported."""
        from scripts.ingest.academic.providers import (
            ArxivProvider,
            DataCiteProvider,
            PubMedProvider,
        )

        assert ArxivProvider is not None
        assert PubMedProvider is not None
        assert DataCiteProvider is not None

    def test_all_providers_inherit_base(self):
        """Test that all providers inherit from BaseProvider."""
        from scripts.ingest.academic.providers import (
            ArxivProvider,
            BaseProvider,
            DataCiteProvider,
            PubMedProvider,
        )

        assert issubclass(ArxivProvider, BaseProvider)
        assert issubclass(PubMedProvider, BaseProvider)
        assert issubclass(DataCiteProvider, BaseProvider)

    def test_all_providers_have_resolve_method(self):
        """Test that all providers implement resolve method."""
        providers = [
            ArxivProvider(),
            PubMedProvider(),
            DataCiteProvider(),
        ]

        for provider in providers:
            assert hasattr(provider, "resolve")
            assert callable(provider.resolve)

    def test_provider_chain_includes_new_providers(self):
        """Test that create_default_chain includes all providers."""
        from scripts.ingest.academic.providers import create_default_chain

        chain = create_default_chain()

        provider_names = [p.name for p in chain.providers]

        assert "arxiv" in provider_names
        assert "pubmed" in provider_names
        assert "datacite" in provider_names
        assert "crossref" in provider_names
        assert "openalex" in provider_names
        assert "semantic_scholar" in provider_names

        # Should have 10 providers total now
        assert len(chain.providers) == 10
