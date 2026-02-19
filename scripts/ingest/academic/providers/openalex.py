"""
OpenAlex metadata provider for comprehensive academic reference resolution.

OpenAlex is a free, open, comprehensive catalog of scholarly works and their
relationships. It covers journals, conferences, institutions, and authors.

API: https://api.openalex.org/
Rate limit: 10 requests/second (100,000 requests/day)
Documentation: https://docs.openalex.org/
"""

import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import List, Optional
from urllib.parse import quote

from .base import BaseProvider, FatalError, RecoverableError, Reference, ReferenceStatus

logger = logging.getLogger(__name__)


class OpenAlexProvider(BaseProvider):
    """
    OpenAlex metadata provider.

    OpenAlex provides comprehensive metadata including:
    - Full bibliographic information
    - Open access status and URLs
    - Author affiliations and profiles
    - Citation counts
    - Related works
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(self):
        self.name = "openalex"
        self.rate_limit = 10  # 10 req/sec
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
        Resolve citation via OpenAlex.

        Priority:
        1. DOI lookup (if provided)
        2. Title + year search
        3. Title + author search
        4. Title-only search

        Args:
            citation_text: Raw citation or title
            year: Publication year (optional)
            doi: DOI if available (optional)
            authors: Author list (optional)
            logger: Optional logger instance (optional)

        Returns:
            Reference with resolved metadata
        """
        # Set up logger
        use_logger = logger or globals().get("logger")
        if use_logger is None:
            use_logger = logging.getLogger(__name__)

        # Try DOI first if provided
        if doi:
            try:
                return self._resolve_by_doi(doi)
            except (RecoverableError, FatalError) as e:
                use_logger.debug(f"DOI lookup failed: {e}, trying title search")

        # Try title + year search
        if year:
            try:
                return self._resolve_by_title_year(citation_text, year)
            except (RecoverableError, FatalError):
                use_logger.debug(f"Title+year search failed, trying title-only search")

        # Try title + author search
        if authors:
            try:
                return self._resolve_by_title_authors(citation_text, authors)
            except (RecoverableError, FatalError):
                use_logger.debug(f"Title+author search failed, trying title-only search")

        # Final fallback: title-only search
        try:
            return self._resolve_by_title(citation_text)
        except FatalError:
            # Return unresolved reference
            return Reference(
                ref_id=f"openalex_unresolved_{hash(citation_text)}",
                raw_citation=citation_text,
                resolved=False,
                status=ReferenceStatus.UNRESOLVED,
                metadata_provider=self.name,
            )

    def _resolve_by_doi(self, doi: str) -> Reference:
        """Resolve by DOI via OpenAlex."""
        # Normalise DOI
        doi_clean = doi.lower().replace("https://doi.org/", "").replace("http://dx.doi.org/", "")

        # Query OpenAlex works endpoint with DOI filter
        url = f"{self.BASE_URL}/works?filter=doi:{quote(doi_clean)}"

        try:
            response = self._request_with_retry("GET", url)
            data = response.json()

            if not data.get("results") or len(data["results"]) == 0:
                raise FatalError(f"DOI not found in OpenAlex: {doi_clean}")

            work = data["results"][0]
            return self._parse_openalex_work(work)

        except Exception as e:
            if isinstance(e, (RecoverableError, FatalError)):
                raise
            raise RecoverableError(f"OpenAlex DOI lookup failed: {e}")

    def _resolve_by_title_year(self, title: str, year: int) -> Reference:
        """Resolve by title + year via OpenAlex."""
        # Query OpenAlex with title and year filters
        url = f"{self.BASE_URL}/works?filter=title.search:{quote(title)},publication_year:{year}&per_page=50"

        try:
            response = self._request_with_retry("GET", url)
            data = response.json()

            results = data.get("results", [])
            if not results:
                raise FatalError(f"No results for title+year: {title} ({year})")

            # Find best match by title similarity
            best_match = None
            best_score = 0.75  # Minimum threshold

            for work in results:
                work_title = work.get("title", "")
                if not work_title:
                    continue

                score = self._fuzzy_match_title(title, work_title)
                if score > best_score:
                    best_score = score
                    best_match = work

            if not best_match:
                raise FatalError(f"Low confidence match for: {title} ({year})")

            return self._parse_openalex_work(best_match)

        except Exception as e:
            if isinstance(e, (RecoverableError, FatalError)):
                raise
            raise RecoverableError(f"OpenAlex title+year lookup failed: {e}")

    def _resolve_by_title_authors(self, title: str, authors: List[str]) -> Reference:
        """Resolve by title + authors via OpenAlex."""
        if not authors:
            raise FatalError("No authors provided")

        # Use first author for search
        first_author = authors[0].split(",")[0].strip()

        # Query with title and author search
        url = f"{self.BASE_URL}/works?filter=title.search:{quote(title)},authorships.author.display_name.search:{quote(first_author)}&per_page=50"

        try:
            response = self._request_with_retry("GET", url)
            data = response.json()

            results = data.get("results", [])
            if not results:
                raise FatalError(f"No results for title+author: {title} by {first_author}")

            # Take first result (already filtered by author)
            best_match = results[0]
            return self._parse_openalex_work(best_match)

        except Exception as e:
            if isinstance(e, (RecoverableError, FatalError)):
                raise
            raise RecoverableError(f"OpenAlex title+author lookup failed: {e}")

    def _resolve_by_title(self, title: str) -> Reference:
        """Resolve by title only via OpenAlex."""
        url = f"{self.BASE_URL}/works?filter=title.search:{quote(title)}&per_page=50"

        try:
            response = self._request_with_retry("GET", url)
            data = response.json()

            results = data.get("results", [])
            if not results:
                raise FatalError(f"No results found for: {title}")

            # Find best match by title similarity
            best_match = None
            best_score = 0.75

            for work in results:
                work_title = work.get("title", "")
                if not work_title:
                    continue

                score = self._fuzzy_match_title(title, work_title)
                if score > best_score:
                    best_score = score
                    best_match = work

            if not best_match:
                raise FatalError(f"Low confidence match for: {title}")

            return self._parse_openalex_work(best_match)

        except Exception as e:
            if isinstance(e, (RecoverableError, FatalError)):
                raise
            raise RecoverableError(f"OpenAlex title lookup failed: {e}")

    def _fuzzy_match_title(self, query: str, candidate: str) -> float:
        """
        Compute fuzzy title match score.

        Returns score between 0 and 1.
        """
        # Normalise both strings
        q = query.lower().strip()
        c = candidate.lower().strip()

        # Exact match gets 1.0
        if q == c:
            return 1.0

        # Check containment
        if q in c or c in q:
            return 0.9

        # Use SequenceMatcher for similarity
        matcher = SequenceMatcher(None, q, c)
        return matcher.ratio()

    def _parse_openalex_work(self, work: dict) -> Reference:
        """Parse OpenAlex work response into Reference."""
        # Extract authors
        authors = []
        for authorship in work.get("authorships", []):
            author_info = authorship.get("author", {})
            display_name = author_info.get("display_name", "")
            if display_name:
                authors.append(display_name)

        # Extract publication date
        pub_date = work.get("publication_date", "")
        year = None
        if pub_date:
            try:
                year = int(pub_date.split("-")[0])
            except (ValueError, IndexError):
                pass

        # Extract venue
        venue = None
        host_venue = work.get("host_venue")
        if host_venue:
            venue = host_venue.get("display_name") or host_venue.get("abbreviated_name")

        # Determine venue type
        venue_type = work.get("type", "").lower() if work.get("type") else "article"
        if "journal-article" in venue_type:
            venue_type = "journal"
        elif "conference" in venue_type:
            venue_type = "conference"

        # Check open access
        is_oa = work.get("is_open_access", False)
        oa_url = None
        if is_oa:
            best_oa = work.get("best_open_access_location")
            if best_oa:
                oa_url = best_oa.get("pdf_url") or best_oa.get("landing_page_url")

        # Compute quality score
        quality_score = self._compute_quality_score(work)

        # Determine reference type
        reference_type = self._determine_reference_type(work)

        return Reference(
            ref_id=work.get("id", "").replace("https://openalex.org/", ""),
            raw_citation="",  # Set by caller if needed
            title=work.get("title", ""),
            authors=authors,
            year=year,
            abstract=work.get("abstract", ""),
            venue=venue,
            venue_type=venue_type,
            volume=(
                str(work.get("biblio", {}).get("volume", ""))
                if work.get("biblio", {}).get("volume")
                else None
            ),
            issue=(
                str(work.get("biblio", {}).get("issue", ""))
                if work.get("biblio", {}).get("issue")
                else None
            ),
            pages=(
                work.get("biblio", {}).get("first_page")
                + "-"
                + work.get("biblio", {}).get("last_page")
                if work.get("biblio", {}).get("first_page")
                and work.get("biblio", {}).get("last_page")
                else None
            ),
            doi=work.get("doi", "").replace("https://doi.org/", "") if work.get("doi") else None,
            reference_type=reference_type,
            resolved=True,
            status=ReferenceStatus.RESOLVED,
            quality_score=quality_score,
            metadata_provider=self.name,
            oa_available=is_oa,
            oa_url=oa_url,
            resolved_at=datetime.now(timezone.utc),
        )

    def _compute_quality_score(self, work: dict) -> float:
        """Compute quality score for OpenAlex work."""
        score = 0.5  # Base score

        # Has DOI: +0.2
        if work.get("doi"):
            score += 0.2

        # Is open access: +0.1
        if work.get("is_open_access"):
            score += 0.1

        # Has citation count: +0.1
        if work.get("cited_by_count", 0) > 0:
            score += 0.1

        # Has abstract: +0.05
        if work.get("abstract"):
            score += 0.05

        # Boost by type
        work_type = work.get("type", "").lower()
        if "peer-reviewed" in work_type or "journal-article" in work_type:
            score += 0.15
        elif "conference" in work_type:
            score += 0.1
        elif "preprint" in work_type:
            score -= 0.05

        return min(1.0, score)

    def _determine_reference_type(self, work) -> str:
        """Determine reference type from OpenAlex work."""
        # Handle both dict (from API) and string (for testing)
        if isinstance(work, dict):
            work_type = work.get("type", "").lower()
        else:
            work_type = str(work).lower()

        if "journal-article" in work_type:
            return "academic"
        elif "conference" in work_type:
            return "conference"
        elif "preprint" in work_type:
            return "preprint"
        elif "report" in work_type:
            return "technical_report"
        elif "book" in work_type:
            return "book"
        else:
            return "online"
