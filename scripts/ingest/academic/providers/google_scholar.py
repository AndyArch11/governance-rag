"""
Google Scholar metadata provider (unofficial).

Google Scholar is a widely used academic search engine with broad coverage.
This provider uses web scraping as there's no official API.

WARNING: This provider uses web scraping and may be subject to rate limiting
or blocking. Use sparingly and respectfully.

Rate limit: 1 req/sec (very conservative to avoid blocking)
Documentation: No official API
"""

import logging
import re
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import List, Optional
from urllib.parse import quote, urlencode

from .base import BaseProvider, FatalError, RecoverableError, Reference, ReferenceStatus

logger = logging.getLogger(__name__)


class GoogleScholarProvider(BaseProvider):
    """
    Google Scholar metadata provider (unofficial, web scraping).

    IMPORTANT NOTES:
    - No official API - uses web scraping
    - Subject to rate limiting and blocking
    - Should be used as last resort before URL fetch
    - Very conservative rate limiting (1 req/sec)
    - May require user agent rotation in production

    Best practices:
    - Use only when other providers fail
    - Implement delays between requests
    - Consider using residential proxies in production
    - Monitor for blocking/CAPTCHA
    """

    BASE_URL = "https://scholar.google.com/scholar"

    def __init__(self):
        self.name = "google_scholar"
        self.rate_limit = 1  # 1 req/sec - very conservative
        self.timeout = 15.0
        self.last_request_time = 0
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
        Resolve citation via Google Scholar search.

        Strategy:
        1. Search by title (and year if available)
        2. Parse HTML results (top result)
        3. Extract basic metadata

        Note: This is a fallback provider due to lack of official API

        Args:
            citation_text: Raw citation or title
            year: Publication year (optional)
            doi: DOI if available (optional, not used)
            authors: Author list (optional)
            logger: Optional logger instance (optional)

        Returns:
            Reference with basic metadata
        """
        # Set up logger
        use_logger = logger or globals().get("logger")
        if use_logger is None:
            use_logger = logging.getLogger(__name__)

        try:
            return self._resolve_by_search(citation_text, year, authors)
        except FatalError:
            # Return unresolved reference
            return Reference(
                ref_id=f"scholar_unresolved_{hash(citation_text)}",
                raw_citation=citation_text,
                resolved=False,
                status=ReferenceStatus.UNRESOLVED,
                metadata_provider=self.name,
            )

    def _resolve_by_search(
        self,
        title: str,
        year: Optional[int] = None,
        authors: Optional[List[str]] = None,
    ) -> Reference:
        """Resolve via Google Scholar search."""
        # Build search query
        query = title

        # Add year to query if available
        if year:
            query = f"{title} {year}"

        # Add first author if available
        if authors and len(authors) > 0:
            author = authors[0].split(",")[0].strip()
            query = f'{query} author:"{author}"'

        # Build URL
        params = {
            "q": query,
            "hl": "en",
            "as_sdt": "0,5",  # Include patents
        }
        url = f"{self.BASE_URL}?{urlencode(params)}"

        # Extra delay for Google Scholar
        self._enforce_extra_delay()

        # Set user agent to avoid blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        try:
            response = self._request_with_retry("GET", url, headers=headers)

            # Check for CAPTCHA or blocking
            if "captcha" in response.text.lower() or response.status_code == 503:
                raise RecoverableError("Google Scholar CAPTCHA/blocking detected")

            # Parse HTML response
            return self._parse_scholar_html(response.text, title)

        except Exception as e:
            if isinstance(e, (RecoverableError, FatalError)):
                raise
            raise RecoverableError(f"Google Scholar search failed: {e}")

    def _enforce_extra_delay(self):
        """Enforce extra delay between requests to avoid blocking."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        # Enforce minimum 2 second delay for Google Scholar
        min_delay = 2.0
        if time_since_last < min_delay:
            sleep_time = min_delay - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _parse_scholar_html(self, html: str, query_title: str) -> Reference:
        """
        Parse Google Scholar HTML response.

        Note: This is fragile and may break if Google changes their HTML structure.
        """
        # Look for result blocks
        # This is a simplified parser - production would use BeautifulSoup

        # Find first result title (in h3 tags)
        title_match = re.search(r'<h3[^>]*class="gs_rt"[^>]*>(.*?)</h3>', html, re.DOTALL)
        if not title_match:
            raise FatalError("No results found in Google Scholar")

        title_html = title_match.group(1)

        # Remove HTML tags
        title = re.sub(r"<[^>]+>", "", title_html).strip()

        # Clean up common artifacts
        title = title.replace("[HTML]", "").replace("[PDF]", "").strip()

        # Verify title match
        if self._fuzzy_match_title(query_title, title) < 0.70:
            raise FatalError("Low confidence match in Google Scholar results")

        # Extract authors and year from citation line
        # Pattern: Authors - Source, Year - ...
        citation_match = re.search(
            r'<div class="gs_a">(.*?)</div>',
            html[title_match.end() : title_match.end() + 500],
            re.DOTALL,
        )

        authors = []
        year = None
        venue = ""

        if citation_match:
            citation_text = re.sub(r"<[^>]+>", "", citation_match.group(1))

            # Try to extract year (4 digits)
            year_match = re.search(r"\b(19|20)\d{2}\b", citation_text)
            if year_match:
                try:
                    year = int(year_match.group(0))
                except ValueError:
                    pass

            # Try to extract authors (before first dash or comma-separated list)
            author_part = citation_text.split("-")[0].strip()
            if author_part:
                # Split by comma, take up to 5 authors
                author_list = [a.strip() for a in author_part.split(",")[:5]]
                authors = [a for a in author_list if a and len(a) < 50]

            # Try to extract venue (between dashes)
            parts = citation_text.split("-")
            if len(parts) > 1:
                venue = parts[1].strip()
                # Remove year if present
                venue = re.sub(r",?\s*(19|20)\d{2}", "", venue).strip()

        # Try to find DOI in the snippet
        doi = None
        doi_match = re.search(
            r'doi\.org/(10\.\d{4,}/[^\s<>"]+)', html[title_match.end() : title_match.end() + 1000]
        )
        if doi_match:
            doi = doi_match.group(1).rstrip(".,;")

        # Try to find PDF link
        oa_url = None
        pdf_match = re.search(
            r'href="([^"]+\.pdf[^"]*)"', html[title_match.start() : title_match.end() + 500]
        )
        if pdf_match:
            oa_url = pdf_match.group(1)
            if not oa_url.startswith("http"):
                oa_url = None

        # Build reference ID
        ref_id = doi if doi else f"scholar_{hash(title)}"

        # Determine reference type (default to academic)
        reference_type = "academic"
        venue_type = "journal"

        # Check for common patterns
        if venue:
            venue_lower = venue.lower()
            if "arxiv" in venue_lower:
                reference_type = "preprint"
                venue_type = "preprint"
            elif "conference" in venue_lower or "proceedings" in venue_lower:
                reference_type = "conference"
                venue_type = "conference"

        # Compute quality score (lower for Scholar due to uncertainty)
        quality_score = self._compute_quality_score(doi, venue, authors)

        return Reference(
            ref_id=ref_id,
            raw_citation="",
            title=title,
            authors=authors,
            year=year,
            abstract="",  # Not available from search results
            venue=venue,
            venue_type=venue_type,
            doi=doi,
            reference_type=reference_type,
            resolved=True,
            status=ReferenceStatus.RESOLVED,
            quality_score=quality_score,
            metadata_provider=self.name,
            oa_available=oa_url is not None,
            oa_url=oa_url,
            resolved_at=datetime.now(timezone.utc),
        )

    def _fuzzy_match_title(self, query: str, candidate: str) -> float:
        """Compute fuzzy title match score."""
        q = query.lower().strip()
        c = candidate.lower().strip()

        if q == c:
            return 1.0

        if q in c or c in q:
            return 0.9

        matcher = SequenceMatcher(None, q, c)
        return matcher.ratio()

    def _compute_quality_score(self, doi: Optional[str], venue: str, authors: List[str]) -> float:
        """Compute quality score for Google Scholar result."""
        # Base score lower due to parsing uncertainty
        score = 0.50

        # Has DOI: +0.20
        if doi:
            score += 0.20

        # Has venue: +0.10
        if venue:
            score += 0.10

        # Has authors: +0.10
        if authors:
            score += 0.10

        # Multiple authors: +0.05
        if len(authors) >= 2:
            score += 0.05

        return min(1.0, score)
