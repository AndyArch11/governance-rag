"""
Unit tests for Crossref provider and cache.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from scripts.ingest.academic.cache import Reference, ReferenceCache, ReferenceStatus
from scripts.ingest.academic.providers import CrossrefProvider, FatalError, RecoverableError


class TestCrossrefProvider:
    """Tests for Crossref metadata provider."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return CrossrefProvider()

    def test_resolve_by_doi_success(self, provider):
        """Test successful DOI resolution."""
        mock_response = {
            "message": {
                "DOI": "10.1038/nature12373",
                "title": ["Sample Paper Title"],
                "author": [
                    {
                        "family": "Smith",
                        "given": "John",
                        "ORCID": "https://orcid.org/0000-0001-2345-6789",
                    },
                    {"family": "Doe", "given": "Jane"},
                ],
                "published": {"date-parts": [[2023, 1, 15]]},
                "container-title": ["Nature"],
                "volume": "500",
                "issue": "3",
                "page": "123-145",
                "abstract": "This is a sample abstract.",
            }
        }

        with patch.object(provider, "_request_with_retry") as mock_request:
            mock_request.return_value = Mock(json=lambda: mock_response)

            result = provider.resolve("Sample Paper Title", year=2023, doi="10.1038/nature12373")

            assert result.resolved
            assert result.doi == "10.1038/nature12373"
            assert result.title == "Sample Paper Title"
            assert len(result.authors) == 2
            assert result.year == 2023
            assert result.venue == "Nature"
            assert result.reference_type == "academic"

    def test_resolve_by_title_success(self, provider):
        """Test successful title-based resolution."""
        mock_response = {
            "message": {
                "items": [
                    {
                        "DOI": "10.1038/nature12373",
                        "title": ["Sample Paper Title"],
                        "author": [{"family": "Smith", "given": "John"}],
                        "published": {"date-parts": [[2023, 1, 15]]},
                        "container-title": ["Nature"],
                    }
                ]
            }
        }

        with patch.object(provider, "_request_with_retry") as mock_request:
            mock_request.return_value = Mock(json=lambda: mock_response)

            result = provider.resolve("Sample Paper Title", year=2023)

            assert result.resolved
            assert result.title == "Sample Paper Title"
            assert result.doi == "10.1038/nature12373"

    def test_resolve_doi_not_found(self, provider):
        """Test handling of non-existent DOI."""
        with patch.object(provider, "_request_with_retry") as mock_request:
            mock_request.side_effect = FatalError("DOI not found")

            with pytest.raises(FatalError):
                provider._resolve_by_doi("10.1234/nonexistent")

    def test_resolve_fallback_to_title(self, provider):
        """Test fallback from DOI to title when DOI fails."""
        title_response = {
            "message": {
                "items": [
                    {
                        "title": ["Sample Paper"],
                        "author": [{"family": "Smith", "given": "John"}],
                        "published": {"date-parts": [[2023]]},
                    }
                ]
            }
        }

        with patch.object(provider, "_request_with_retry") as mock_request:
            # First call (DOI) fails, second call (title) succeeds
            mock_request.side_effect = [
                Mock(side_effect=FatalError("Not found")),
                Mock(json=lambda: title_response),
            ]

            # We need to patch the method calls differently
            with patch.object(provider, "_resolve_by_doi", side_effect=FatalError("Not found")):
                with patch.object(provider, "_resolve_by_title") as mock_title:
                    mock_title.return_value = Reference(
                        ref_id="test",
                        title="Sample Paper",
                        resolved=True,
                    )

                    result = provider.resolve("Sample Paper", year=2023, doi="10.1234/bad")

                    assert result.resolved

    def test_provider_initialisation(self, provider):
        """Test provider is properly initialised."""
        # Ensure provider is properly initialised
        assert provider.rate_limit == 50  # 50 req/sec for Crossref
        assert hasattr(provider, "resolve")
        assert hasattr(provider, "_rate_limit")

    def test_rate_limiting(self, provider):
        """Test that rate limiting works."""
        provider.rate_limit = 10  # 10 req/sec = 0.1s min interval

        import time

        start = time.time()

        # Make two rate-limited calls
        provider._rate_limit()
        provider._rate_limit()

        elapsed = time.time() - start

        # Should take at least one interval (~0.1s)
        assert elapsed >= 0.05  # Allow some tolerance


class TestReferenceCache:
    """Tests for reference cache."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create cache instance with temporary database."""
        db_path = str(tmp_path / "test_cache.db")
        return ReferenceCache(db_path)

    def test_cache_initialisation(self, cache):
        """Test that cache database is created."""
        assert cache.db_path.exists()

    def test_cache_put_and_get(self, cache):
        """Test storing and retrieving references."""
        ref = Reference(
            ref_id="test_ref_1",
            title="Test Paper",
            authors=["Smith, John", "Doe, Jane"],
            year=2023,
            doi="10.1038/test",
            resolved=True,
            metadata_provider="crossref",
            quality_score=0.95,
        )

        cache_key = cache.compute_cache_key(doi="10.1038/test")
        cache.put(cache_key, ref)

        # Retrieve
        retrieved = cache.get(cache_key)

        assert retrieved is not None
        assert retrieved.ref_id == "test_ref_1"
        assert retrieved.title == "Test Paper"
        assert retrieved.year == 2023
        assert retrieved.quality_score == 0.95

    def test_cache_key_uniqueness(self, cache):
        """Test that different citations get different keys."""
        key1 = cache.compute_cache_key(doi="10.1038/test1")
        key2 = cache.compute_cache_key(doi="10.1038/test2")

        assert key1 != key2

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get("nonexistent_key")
        assert result is None

    def test_add_citation_tracking(self, cache):
        """Test tracking document citations."""
        ref = Reference(
            ref_id="ref_1",
            title="Test Paper",
            resolved=True,
        )

        cache_key = cache.compute_cache_key(title="Test Paper")
        cache.put(cache_key, ref)
        cache.add_citation("doc_1", "ref_1", "Smith et al. 2023")

        # Verify citation was tracked
        import sqlite3
        from contextlib import closing

        with closing(sqlite3.connect(str(cache.db_path))) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM document_citations WHERE doc_id = ? AND ref_id = ?",
                ("doc_1", "ref_1"),
            )
            row = cursor.fetchone()

            assert row is not None

    def test_cache_expiration(self, cache):
        """Test that cache expiration is computed correctly."""
        ref = Reference(
            ref_id="test_ref",
            title="Test Paper",
            resolved=True,
        )

        cache_key = "test_key"

        # Put with long expiration (365 days default)
        cache.put(cache_key, ref, expires_in_days=365)

        # Should retrieve the item with future expiration
        retrieved = cache.get(cache_key)
        assert retrieved is not None
        assert retrieved.ref_id == "test_ref"


class TestReferenceObject:
    """Tests for Reference data model."""

    def test_reference_creation(self):
        """Test Reference object creation."""
        ref = Reference(
            ref_id="test",
            title="Test",
            authors=["Smith, John"],
            year=2023,
            doi="10.1038/test",
            resolved=True,
            reference_type="academic",
            quality_score=0.95,
        )

        assert ref.ref_id == "test"
        assert ref.resolved
        assert ref.quality_score == 0.95
        assert ref.reference_type == "academic"

    def test_government_reference_type(self):
        """Test government reference type support."""
        ref = Reference(
            ref_id="abs_1",
            title="ABS Census Data",
            reference_type="government",
            quality_score=0.85,
            resolved=True,
        )

        assert ref.reference_type == "government"
        assert ref.quality_score == 0.85

    def test_reference_default_values(self):
        """Test Reference default values."""
        ref = Reference(ref_id="test", raw_citation="Test citation")

        assert ref.authors == []
        assert ref.doc_ids == []
        assert not ref.resolved
        assert ref.quality_score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
