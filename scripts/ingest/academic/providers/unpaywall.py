"""
Unpaywall metadata provider for open access discovery.

Unpaywall finds legal, open access versions of scholarly articles.

API: https://api.unpaywall.org/v2/
Rate limit: 100,000 requests/day (~1 req/sec sustained)
Documentation: https://unpaywall.org/products/api
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from .base import BaseProvider, FatalError, RecoverableError, Reference, ReferenceStatus

logger = logging.getLogger(__name__)


class UnpaywallProvider(BaseProvider):
    """
    Unpaywall metadata provider for open access discovery.

    Unpaywall provides:
    - Open access availability checking
    - Legal OA PDF links
    - Publisher and repository versions
    - OA status classification (gold, green, hybrid, bronze)

    Note: Requires DOI - best used as a complement to other providers
    Email parameter recommended for polite pool access
    """

    BASE_URL = "https://api.unpaywall.org/v2"

    def __init__(self, email: Optional[str] = None):
        """
        Initialise Unpaywall provider.

        Args:
            email: Email for polite pool access (recommended)
        """
        self.name = "unpaywall"
        self.email = email or "research@example.com"  # Default fallback
        self.rate_limit = 1  # 1 req/sec - conservative
        self.timeout = 10.0
        super().__init__()

    def resolve(
        self,
        citation_text: str,
        year: Optional[int] = None,
        doi: Optional[str] = None,
        authors: Optional[List[str]] = None,
        logger=None,
    ) -> Reference:
        """
        Resolve citation via Unpaywall.

        Strategy:
        1. DOI lookup (required for Unpaywall)
        2. Find OA locations
        3. Return metadata with OA links

        Note: Unpaywall REQUIRES a DOI - returns unresolved without one

        Args:
            citation_text: Raw citation or title
            year: Publication year (optional, not used)
            doi: DOI (REQUIRED for Unpaywall)
            authors: Author list (optional, not used)
            logger: Optional logger instance (optional)

        Returns:
            Reference with OA metadata
        """
        # Set up logger
        use_logger = logger or globals().get("logger")
        if use_logger is None:
            use_logger = logging.getLogger(__name__)

        # Unpaywall requires DOI
        if not doi:
            return Reference(
                ref_id=f"unpaywall_unresolved_{hash(citation_text)}",
                raw_citation=citation_text,
                resolved=False,
                status=ReferenceStatus.UNRESOLVED,
                metadata_provider=self.name,
            )

        try:
            return self._resolve_by_doi(doi)
        except FatalError:
            # Return unresolved reference
            return Reference(
                ref_id=f"unpaywall_unresolved_{doi}",
                raw_citation=citation_text,
                resolved=False,
                status=ReferenceStatus.UNRESOLVED,
                metadata_provider=self.name,
            )

    def _resolve_by_doi(self, doi: str) -> Reference:
        """Resolve by DOI via Unpaywall."""
        # Clean DOI
        doi = doi.strip().lower()
        if doi.startswith("http"):
            doi = doi.split("doi.org/")[-1]

        url = f"{self.BASE_URL}/{doi}"
        params = {"email": self.email}

        try:
            response = self._request_with_retry("GET", url, params=params)
            data = response.json()

            # Check if we got valid data
            if not data or "title" not in data:
                raise FatalError(f"No Unpaywall data for DOI: {doi}")

            return self._parse_unpaywall_record(data)

        except Exception as e:
            if isinstance(e, (RecoverableError, FatalError)):
                raise
            raise RecoverableError(f"Unpaywall lookup failed: {e}")

    def _parse_unpaywall_record(self, record: dict) -> Reference:
        """Parse Unpaywall record."""
        # Extract basic metadata
        doi = record.get("doi", "")
        title = record.get("title", "")

        # Extract authors
        authors = []
        z_authors = record.get("z_authors", [])
        if z_authors:
            for author in z_authors:
                if isinstance(author, dict):
                    given = author.get("given", "")
                    family = author.get("family", "")
                    if family:
                        full_name = f"{given} {family}".strip()
                        authors.append(full_name)

        # Extract year
        year = None
        published_date = record.get("published_date")
        if published_date:
            try:
                year = int(published_date.split("-")[0])
            except (ValueError, IndexError, AttributeError):
                pass

        # If no published_date, try year field
        if not year:
            year_value = record.get("year")
            if year_value:
                try:
                    year = int(year_value)
                except (ValueError, TypeError):
                    pass

        # Extract journal/venue
        journal = record.get("journal_name", "")
        publisher = record.get("publisher", "")
        venue = journal or publisher

        # Extract abstract (if available)
        abstract = record.get("abstract", "")

        # Determine if open access
        is_oa = record.get("is_oa", False)
        oa_status = record.get("oa_status", "")  # gold, green, hybrid, bronze, closed

        # Find best OA location
        oa_url = None
        best_oa_location = record.get("best_oa_location")
        if best_oa_location:
            oa_url = best_oa_location.get("url_for_pdf") or best_oa_location.get("url")

        # If no best location, check all OA locations
        if not oa_url:
            oa_locations = record.get("oa_locations", [])
            for location in oa_locations:
                if location.get("url_for_pdf"):
                    oa_url = location["url_for_pdf"]
                    break
                elif location.get("url"):
                    oa_url = location["url"]
                    break

        # Determine reference type
        genre = record.get("genre", "")
        reference_type = self._determine_reference_type(genre, journal)
        venue_type = self._determine_venue_type(genre, journal)

        # Compute quality score
        quality_score = self._compute_quality_score(record, oa_status)

        return Reference(
            ref_id=doi,
            raw_citation="",
            title=title,
            authors=authors,
            year=year,
            abstract=abstract[:500] if abstract else "",  # Limit abstract length
            venue=venue,
            venue_type=venue_type,
            doi=doi,
            reference_type=reference_type,
            resolved=True,
            status=ReferenceStatus.RESOLVED,
            quality_score=quality_score,
            metadata_provider=self.name,
            oa_available=is_oa,
            oa_url=oa_url,
            resolved_at=datetime.now(timezone.utc),
        )

    def _determine_reference_type(self, genre: str, journal: str) -> str:
        """Determine reference type from Unpaywall data."""
        genre_lower = genre.lower()

        if "journal" in genre_lower:
            return "academic"
        elif "conference" in genre_lower or "proceedings" in genre_lower:
            return "conference"
        elif "book" in genre_lower:
            return "academic"
        elif "preprint" in genre_lower or "posted-content" in genre_lower:
            return "preprint"
        elif "report" in genre_lower:
            return "technical"

        # Default to academic if in a journal
        if journal:
            return "academic"

        return "other"

    def _determine_venue_type(self, genre: str, journal: str) -> str:
        """Determine venue type from Unpaywall data."""
        genre_lower = genre.lower()

        if "journal" in genre_lower:
            return "journal"
        elif "conference" in genre_lower or "proceedings" in genre_lower:
            return "conference"
        elif "book" in genre_lower:
            return "book"
        elif "preprint" in genre_lower:
            return "preprint"

        # Default to journal if in a journal
        if journal:
            return "journal"

        return "other"

    def _compute_quality_score(self, record: dict, oa_status: str) -> float:
        """Compute quality score for Unpaywall record."""
        # Base score
        score = 0.85

        # Has abstract: +0.05
        if record.get("abstract"):
            score += 0.05

        # Open access bonus based on type
        if oa_status == "gold":
            score += 0.10  # Publisher OA
        elif oa_status == "green":
            score += 0.05  # Repository OA
        elif oa_status == "hybrid":
            score += 0.08  # Hybrid OA
        elif oa_status == "bronze":
            score += 0.03  # Free to read

        return min(1.0, score)
