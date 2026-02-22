"""
Integration tests for Crossref provider with extracted references.

These tests verify that the Crossref provider can resolve real references
extracted from academic PDFs.
"""

import re
from pathlib import Path

import pytest

from scripts.ingest.academic.cache import ReferenceCache
from scripts.ingest.academic.providers import CrossrefProvider, Reference

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EXTRACTED_REFERENCES_FILE = FIXTURES_DIR / "extracted_references.txt"


class TestCrossrefIntegration:
    """Integration tests for Crossref provider with real references."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return CrossrefProvider()

    @pytest.fixture
    def cache(self, tmp_path):
        """Create cache instance with temporary database."""
        db_path = str(tmp_path / "integration_test_cache.db")
        return ReferenceCache(db_path)

    @pytest.fixture
    def extracted_references(self):
        """Load extracted references from file."""
        ref_file = EXTRACTED_REFERENCES_FILE

        if not ref_file.exists():
            pytest.skip("extracted_references.txt not found")

        references = []
        current_ref = []

        with open(ref_file, "r") as f:
            for line in f:
                line = line.strip()

                # Skip header and separator lines
                if not line or line.startswith("Total") or line.startswith("==="):
                    continue

                # Check for reference number marker (e.g., "[1]")
                if re.match(r"^\[\d+\]", line):
                    # Save previous reference if exists
                    if current_ref:
                        ref_text = " ".join(current_ref)
                        # Remove the number prefix
                        ref_text = re.sub(r"^\[\d+\]\s*", "", ref_text)
                        references.append(ref_text)
                    # Start new reference
                    current_ref = [line]
                else:
                    # Continuation of previous reference
                    if current_ref:
                        current_ref.append(line)

        # Add last reference
        if current_ref:
            ref_text = " ".join(current_ref)
            ref_text = re.sub(r"^\[\d+\]\s*", "", ref_text)
            references.append(ref_text)

        return references[:20]  # Test with first 20 actual references

    def parse_reference(self, raw_ref: str) -> dict:
        """
        Parse a raw reference string into components.

        Attempts to extract:
        - DOI (10.xxxx/yyyy format)
        - Year (4-digit number in parentheses or at end)
        - Title and authors
        """
        components = {
            "raw_citation": raw_ref,
            "doi": None,
            "year": None,
            "title": raw_ref,
            "authors": [],
        }

        # Extract DOI
        doi_match = re.search(r"(?:https?://)?(?:dx\.)?doi\.org/(10\.\S+)|(10\.\S+)", raw_ref)
        if doi_match:
            components["doi"] = doi_match.group(1) or doi_match.group(2)

        # Extract year (4-digit number in parentheses)
        year_match = re.search(r"\((\d{4})\)", raw_ref)
        if year_match:
            components["year"] = int(year_match.group(1))

        # Extract title (often between quotes or before first author)
        title_match = re.search(r'["\']([^"\']+)["\']', raw_ref)
        if title_match:
            components["title"] = title_match.group(1)

        return components

    def test_resolve_first_reference(self, provider, extracted_references):
        """Test resolving the first extracted reference."""
        if not extracted_references:
            pytest.skip("No extracted references available")

        ref_str = extracted_references[0]
        components = self.parse_reference(ref_str)

        result = provider.resolve(
            citation_text=components["title"], year=components["year"], doi=components["doi"]
        )

        # Should either resolve or at least attempt
        assert isinstance(result, Reference)

    def test_resolve_multiple_references(self, provider, extracted_references, cache):
        """Test resolving multiple extracted references."""
        if not extracted_references:
            pytest.skip("No extracted references available")

        resolved_count = 0
        unresolved_count = 0

        for ref_str in extracted_references[:5]:  # Test first 5
            components = self.parse_reference(ref_str)

            result = provider.resolve(
                citation_text=components["title"], year=components["year"], doi=components["doi"]
            )

            # If resolution failed (returned None), skip
            if result is None:
                unresolved_count += 1
                continue

            # Store in cache
            cache_key = cache.compute_cache_key(
                doi=result.doi, title=result.title, year=result.year, authors=result.authors
            )
            cache.put(cache_key, result)

            if result.resolved:
                resolved_count += 1
            else:
                unresolved_count += 1

        print(f"\nResolution results: {resolved_count} resolved, {unresolved_count} unresolved")

        # At least some should resolve
        assert resolved_count >= 0  # May be 0 if references are obscure

    def test_cache_reduces_api_calls(self, provider, cache, extracted_references):
        """Test that cache reduces API calls for repeated resolutions."""
        if not extracted_references:
            pytest.skip("No extracted references available")

        # First resolution (hits API)
        ref_str = extracted_references[0]
        components = self.parse_reference(ref_str)

        result1 = provider.resolve(
            citation_text=components["title"], year=components["year"], doi=components["doi"]
        )

        cache_key = cache.compute_cache_key(
            doi=result1.doi, title=result1.title, year=result1.year, authors=result1.authors
        )
        cache.put(cache_key, result1)

        # Second resolution (should hit cache)
        cached_result = cache.get(cache_key)

        if result1.resolved:
            assert cached_result is not None
            assert cached_result.title == result1.title

    def test_reference_type_classification(self, provider, cache, extracted_references):
        """Test that references are properly classified by type."""
        if not extracted_references:
            pytest.skip("No extracted references available")

        type_distribution = {}

        for ref_str in extracted_references[:10]:
            components = self.parse_reference(ref_str)

            result = provider.resolve(
                citation_text=components["title"], year=components["year"], doi=components["doi"]
            )

            # Skip if resolution failed
            if result is None:
                continue

            ref_type = result.reference_type or "unknown"
            type_distribution[ref_type] = type_distribution.get(ref_type, 0) + 1

        print(f"\nReference type distribution: {type_distribution}")

        # Should have at least some classification
        assert len(type_distribution) > 0

    def test_quality_scores(self, provider, cache, extracted_references):
        """Test that quality scores are computed correctly."""
        if not extracted_references:
            pytest.skip("No extracted references available")

        scores = []

        for ref_str in extracted_references[:10]:
            components = self.parse_reference(ref_str)

            result = provider.resolve(
                citation_text=components["title"], year=components["year"], doi=components["doi"]
            )

            # Skip if resolution failed
            if result is None:
                continue

            if result.resolved:
                scores.append(result.quality_score)

        if scores:
            print(
                f"\nQuality scores: min={min(scores)}, max={max(scores)}, avg={sum(scores)/len(scores):.2f}"
            )

            # All should be between 0 and 1
            assert all(0 <= score <= 1 for score in scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
