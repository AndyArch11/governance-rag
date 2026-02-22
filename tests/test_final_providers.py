"""
Tests for ORCID, Google Scholar, Unpaywall, and URL Fetch providers.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest

from scripts.ingest.academic.providers import (
    GoogleScholarProvider,
    ORCIDProvider,
    ReferenceStatus,
    UnpaywallProvider,
    URLFetchProvider,
)


class TestORCIDProvider:
    """Tests for ORCID provider."""

    def test_initialisation(self):
        """Test provider initialisation."""
        provider = ORCIDProvider()

        assert provider.name == "orcid"
        assert provider.rate_limit == 5
        assert provider.timeout == 10.0

    def test_extract_orcid_id(self):
        """Test ORCID ID extraction from citation."""
        provider = ORCIDProvider()

        # Standard format
        assert provider._extract_orcid_id("0000-0002-1825-0097") == "0000-0002-1825-0097"

        # URL format
        assert (
            provider._extract_orcid_id("https://orcid.org/0000-0002-1825-0097")
            == "0000-0002-1825-0097"
        )

        # With X check digit
        assert provider._extract_orcid_id("0000-0002-1825-009X") == "0000-0002-1825-009X"

        # No ORCID ID
        assert provider._extract_orcid_id("Some random text") is None

    @patch("scripts.ingest.academic.providers.orcid.ORCIDProvider._request_with_retry")
    def test_resolve_by_orcid_works(self, mock_request):
        """Test resolution by ORCID works search."""
        provider = ORCIDProvider()

        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "group": [
                {
                    "work-summary": [
                        {
                            "title": {"title": {"value": "Test Paper Title"}},
                            "publication-date": {"year": {"value": "2020"}},
                            "external-ids": {
                                "external-id": [
                                    {"external-id-type": "doi", "external-id-value": "10.1234/test"}
                                ]
                            },
                            "type": "journal-article",
                            "journal-title": {"value": "Nature"},
                        }
                    ]
                }
            ]
        }
        mock_request.return_value = mock_response

        ref = provider._resolve_by_orcid_works(
            "0000-0002-1825-0097", "Test Paper Title", year=2020, doi="10.1234/test"
        )

        assert ref.resolved is True
        assert ref.title == "Test Paper Title"
        assert ref.year == 2020
        assert ref.doi == "10.1234/test"
        assert ref.venue == "Nature"
        assert ref.reference_type == "academic"

    def test_determine_reference_type(self):
        """Test reference type determination."""
        provider = ORCIDProvider()

        assert provider._determine_reference_type("journal-article") == "academic"
        assert provider._determine_reference_type("conference-paper") == "conference"
        assert provider._determine_reference_type("preprint") == "preprint"
        assert provider._determine_reference_type("data-set") == "dataset"
        assert provider._determine_reference_type("software") == "software"

    def test_quality_score_computation(self):
        """Test quality score computation."""
        provider = ORCIDProvider()

        work = {
            "external-ids": {"external-id": [{"external-id-type": "doi"}]},
            "journal-title": {"value": "Test Journal"},
            "publication-date": {"year": {"value": "2020"}},
        }

        score = provider._compute_quality_score(work)
        # 0.80 base + 0.10 DOI + 0.05 journal + 0.05 date = 1.00
        assert score == 1.00


class TestGoogleScholarProvider:
    """Tests for Google Scholar provider."""

    def test_initialisation(self):
        """Test provider initialisation."""
        provider = GoogleScholarProvider()

        assert provider.name == "google_scholar"
        assert provider.rate_limit == 1  # Very conservative
        assert provider.timeout == 15.0

    @patch(
        "scripts.ingest.academic.providers.google_scholar.GoogleScholarProvider._request_with_retry"
    )
    @patch(
        "scripts.ingest.academic.providers.google_scholar.GoogleScholarProvider._enforce_extra_delay"
    )
    def test_resolve_by_search(self, mock_delay, mock_request):
        """Test resolution by search."""
        provider = GoogleScholarProvider()

        # Mock HTML response
        mock_html = """
        <html>
            <h3 class="gs_rt"><a href="#">Deep Learning for Computer Vision</a></h3>
            <div class="gs_a">John Doe, Jane Smith - arXiv preprint - 2020 - arxiv.org</div>
        </html>
        """

        mock_response = Mock()
        mock_response.text = mock_html
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        ref = provider._resolve_by_search("Deep Learning for Computer Vision", year=2020)

        assert ref.resolved is True
        assert "Deep Learning" in ref.title
        assert ref.year == 2020

    def test_fuzzy_match_title(self):
        """Test fuzzy title matching."""
        provider = GoogleScholarProvider()

        # Exact match
        assert provider._fuzzy_match_title("Test Title", "Test Title") == 1.0

        # Close match
        score = provider._fuzzy_match_title("Deep Learning", "Deep Learning Tutorial")
        assert score == 0.9

    def test_quality_score_computation(self):
        """Test quality score computation."""
        provider = GoogleScholarProvider()

        # With DOI and venue
        score = provider._compute_quality_score("10.1234/test", "Nature", ["Author1", "Author2"])
        # 0.50 + 0.20 DOI + 0.10 venue + 0.10 authors + 0.05 multiple = 0.95
        assert score == 0.95

        # Minimal
        score = provider._compute_quality_score(None, "", [])
        assert score == 0.50


class TestUnpaywallProvider:
    """Tests for Unpaywall provider."""

    def test_initialisation(self):
        """Test provider initialisation."""
        # Without email
        provider = UnpaywallProvider()
        assert provider.name == "unpaywall"
        assert provider.email == "research@example.com"

        # With email
        provider_with_email = UnpaywallProvider(email="test@test.com")
        assert provider_with_email.email == "test@test.com"

    @patch("scripts.ingest.academic.providers.unpaywall.UnpaywallProvider._request_with_retry")
    def test_resolve_by_doi(self, mock_request):
        """Test resolution by DOI."""
        provider = UnpaywallProvider()

        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "doi": "10.1234/test",
            "title": "Test Paper",
            "z_authors": [{"given": "John", "family": "Doe"}, {"given": "Jane", "family": "Smith"}],
            "published_date": "2020-01-01",
            "journal_name": "Nature",
            "is_oa": True,
            "oa_status": "gold",
            "best_oa_location": {"url_for_pdf": "https://example.com/paper.pdf"},
            "genre": "journal-article",
        }
        mock_request.return_value = mock_response

        ref = provider._resolve_by_doi("10.1234/test")

        assert ref.resolved is True
        assert ref.title == "Test Paper"
        assert ref.authors == ["John Doe", "Jane Smith"]
        assert ref.year == 2020
        assert ref.venue == "Nature"
        assert ref.oa_available is True
        assert ref.oa_url == "https://example.com/paper.pdf"

    def test_resolve_without_doi(self):
        """Test resolution without DOI returns unresolved."""
        provider = UnpaywallProvider()

        ref = provider.resolve("Some citation without DOI", doi=None)

        assert ref.resolved is False
        assert ref.status == ReferenceStatus.UNRESOLVED

    def test_determine_reference_type(self):
        """Test reference type determination."""
        provider = UnpaywallProvider()

        assert provider._determine_reference_type("journal-article", "Nature") == "academic"
        assert provider._determine_reference_type("proceedings-article", "") == "conference"
        assert provider._determine_reference_type("posted-content", "") == "preprint"

    def test_quality_score_computation(self):
        """Test quality score computation."""
        provider = UnpaywallProvider()

        record = {"abstract": "Test abstract"}

        # Gold OA
        score = provider._compute_quality_score(record, "gold")
        assert score == 1.00  # 0.85 + 0.05 abstract + 0.10 gold

        # Green OA
        score = provider._compute_quality_score(record, "green")
        assert abs(score - 0.95) < 0.01  # 0.85 + 0.05 abstract + 0.05 green


class TestURLFetchProvider:
    """Tests for URL Fetch provider."""

    def test_initialisation(self):
        """Test provider initialisation."""
        provider = URLFetchProvider()

        assert provider.name == "url_fetch"
        assert provider.rate_limit == 2
        assert provider.timeout == 15.0

    def test_extract_url(self):
        """Test URL extraction from citation."""
        provider = URLFetchProvider()

        # Simple URL
        assert (
            provider._extract_url("https://example.com/paper.pdf")
            == "https://example.com/paper.pdf"
        )

        # URL in text
        url = provider._extract_url("See the paper at https://arxiv.org/abs/1234.5678 for details")
        assert url == "https://arxiv.org/abs/1234.5678"

        # No URL
        assert provider._extract_url("No URL here") is None

    @patch("scripts.ingest.academic.providers.url_fetch.URLFetchProvider._request_with_retry")
    def test_parse_html_metadata(self, mock_request):
        """Test HTML metadata parsing."""
        provider = URLFetchProvider()

        # Mock HTML with meta tags
        mock_html = """
        <html>
            <head>
                <meta name="citation_title" content="Test Paper Title">
                <meta name="citation_author" content="John Doe">
                <meta name="citation_author" content="Jane Smith">
                <meta name="citation_publication_date" content="2020-01-01">
                <meta name="citation_doi" content="10.1234/test">
                <meta name="description" content="This is an abstract">
            </head>
        </html>
        """

        ref = provider._parse_html_metadata("https://example.com/paper", mock_html, "Test")

        assert ref.resolved is True
        assert ref.title == "Test Paper Title"
        assert "John Doe" in ref.authors
        assert "Jane Smith" in ref.authors
        assert ref.year == 2020
        assert ref.doi == "10.1234/test"
        assert ref.abstract == "This is an abstract"

    @patch("scripts.ingest.academic.providers.url_fetch.URLFetchProvider._request_with_retry")
    def test_parse_pdf_metadata(self, mock_request):
        """Test PDF metadata parsing."""
        provider = URLFetchProvider()

        mock_response = Mock()
        mock_response.headers = {"Content-Type": "application/pdf"}

        ref = provider._parse_pdf_metadata("https://example.com/paper_2020.pdf", mock_response)

        assert ref.resolved is True
        assert "paper 2020" in ref.title.lower()
        assert ref.year == 2020
        assert ref.oa_available is True
        assert ref.quality_score == 0.40  # Low quality for PDF

    def test_determine_type_from_url(self):
        """Test reference type determination from URL."""
        provider = URLFetchProvider()

        # Government
        ref_type, venue_type = provider._determine_type_from_url("https://nih.gov/report", {})
        assert ref_type == "government"

        # Academic
        ref_type, venue_type = provider._determine_type_from_url("https://mit.edu/paper", {})
        assert ref_type == "technical"

        # arXiv
        ref_type, venue_type = provider._determine_type_from_url("https://arxiv.org/abs/1234", {})
        assert ref_type == "preprint"

        # GitHub
        ref_type, venue_type = provider._determine_type_from_url("https://github.com/repo", {})
        assert ref_type == "software"

        # Blog
        ref_type, venue_type = provider._determine_type_from_url("https://medium.com/post", {})
        assert ref_type == "blog"

    def test_quality_score_computation(self):
        """Test quality score computation."""
        provider = URLFetchProvider()

        # Maximum metadata
        metadata = {
            "doi": "10.1234/test",
            "authors": ["Author1", "Author2"],
            "year": 2020,
            "abstract": "Test abstract",
        }
        score = provider._compute_quality_score(metadata)
        # 0.30 + 0.30 DOI + 0.15 authors + 0.10 year + 0.10 abstract + 0.05 multi-author = 1.00
        assert score == 1.00

        # Minimal metadata
        metadata_min = {}
        score = provider._compute_quality_score(metadata_min)
        assert score == 0.30  # Base score only


class TestProviderIntegration:
    """Integration tests for all new providers."""

    def test_all_providers_importable(self):
        """Test that all providers can be imported."""
        from scripts.ingest.academic.providers import (
            GoogleScholarProvider,
            ORCIDProvider,
            UnpaywallProvider,
            URLFetchProvider,
        )

        assert ORCIDProvider is not None
        assert GoogleScholarProvider is not None
        assert UnpaywallProvider is not None
        assert URLFetchProvider is not None

    def test_all_providers_inherit_base(self):
        """Test that all providers inherit from BaseProvider."""
        from scripts.ingest.academic.providers import (
            BaseProvider,
            GoogleScholarProvider,
            ORCIDProvider,
            UnpaywallProvider,
            URLFetchProvider,
        )

        assert issubclass(ORCIDProvider, BaseProvider)
        assert issubclass(GoogleScholarProvider, BaseProvider)
        assert issubclass(UnpaywallProvider, BaseProvider)
        assert issubclass(URLFetchProvider, BaseProvider)

    def test_all_providers_have_resolve_method(self):
        """Test that all providers implement resolve method."""
        providers = [
            ORCIDProvider(),
            GoogleScholarProvider(),
            UnpaywallProvider(),
            URLFetchProvider(),
        ]

        for provider in providers:
            assert hasattr(provider, "resolve")
            assert callable(provider.resolve)

    def test_provider_chain_includes_all_providers(self):
        """Test that create_default_chain includes all 10 providers."""
        from scripts.ingest.academic.providers import create_default_chain

        chain = create_default_chain()

        provider_names = [p.name for p in chain.providers]

        # High-value providers
        assert "crossref" in provider_names
        assert "datacite" in provider_names
        assert "pubmed" in provider_names
        assert "openalex" in provider_names
        assert "arxiv" in provider_names
        assert "semantic_scholar" in provider_names

        # Lower-value providers
        assert "orcid" in provider_names
        assert "google_scholar" in provider_names
        assert "unpaywall" in provider_names
        assert "url_fetch" in provider_names

        # Should have 10 providers total
        assert len(chain.providers) == 10

    def test_provider_chain_order(self):
        """Test that providers are in correct priority order."""
        from scripts.ingest.academic.providers import create_default_chain

        chain = create_default_chain()

        expected_order = [
            "crossref",
            "datacite",
            "pubmed",
            "openalex",
            "arxiv",
            "semantic_scholar",
            "orcid",
            "google_scholar",
            "unpaywall",
            "url_fetch",
        ]

        actual_order = [p.name for p in chain.providers]
        assert actual_order == expected_order
