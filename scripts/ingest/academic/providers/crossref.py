"""
Crossref metadata provider for DOI-based citation resolution.

Crossref is the official DOI registration agency and provides high-quality
bibliographic metadata for peer-reviewed publications.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import quote

from .base import BaseProvider, FatalError, RecoverableError, Reference, ReferenceStatus

logger = logging.getLogger(__name__)


class CrossrefProvider(BaseProvider):
    """Crossref metadata provider."""

    name = "crossref"
    base_url = "https://api.crossref.org"
    rate_limit = 50  # Requests per second (polite pool)
    timeout = 10  # seconds

    def resolve(
        self, citation_text: str, year: Optional[int] = None, doi: Optional[str] = None, logger=None
    ) -> Reference:
        """
        Resolve citation via Crossref.

        Priority:
        1. Direct DOI lookup (if provided)
        2. Title + year search (if no DOI)
        3. Fall back to title-only search

        Args:
            citation_text: Raw citation or title
            year: Publication year (optional)
            doi: DOI if available (optional)
            logger: Optional logger instance for status messages

        Returns:
            Reference with resolved metadata

        Raises:
            RecoverableError: Transient failure
            FatalError: Permanent failure
        """
        # Use provided logger or fall back to module logger
        use_logger = logger or globals().get("logger")
        if use_logger is None:
            use_logger = logging.getLogger(__name__)

        # Try DOI lookup first
        if doi:
            try:
                return self._resolve_by_doi(doi, use_logger)
            except FatalError as e:
                use_logger.debug(f"DOI lookup failed for {doi}: {e}, trying title search")
            except RecoverableError:
                raise  # Don't continue on recoverable errors

        # Fall back to title-based search
        return self._resolve_by_title(citation_text, year, use_logger)

    def _resolve_by_doi(self, doi: str, logger=None) -> Reference:
        """
        Resolve by DOI (most reliable method).

        Args:
            doi: DOI identifier
            logger: Optional logger instance

        Returns:
            Resolved Reference

        Raises:
            FatalError: DOI not found or invalid
            RecoverableError: API failure
        """
        use_logger = logger or globals().get("logger") or logging.getLogger(__name__)

        # Normalise DOI (remove https://doi.org/ prefix if present)
        doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

        url = f"{self.base_url}/works/{quote(doi_clean)}"
        use_logger.debug(f"Looking up DOI: {doi_clean}")

        response = self._request_with_retry("GET", url)
        data = response.json()

        if "message" not in data:
            raise FatalError(f"Invalid Crossref response for DOI {doi_clean}")

        return self._parse_crossref_work(data["message"], doi_clean)

    def _resolve_by_title(self, title: str, year: Optional[int] = None, logger=None) -> Reference:
        """
        Resolve by title and year (fallback).

        Args:
            title: Publication title
            year: Publication year (optional, improves accuracy)
            logger: Optional logger instance

        Returns:
            Resolved Reference

        Raises:
            FatalError: No matches found
            RecoverableError: API failure
        """
        use_logger = logger or globals().get("logger") or logging.getLogger(__name__)

        url = f"{self.base_url}/works"
        params = {
            "query.title": title,
            "rows": 5,
            "sort": "score",
        }

        # Add year filter if provided
        if year:
            params["filter"] = f"from-pub-date:{year},until-pub-date:{year}"

        use_logger.debug(f"Searching Crossref for: {title} ({year or 'no year'})")

        response = self._request_with_retry("GET", url, params=params)
        data = response.json()

        if "message" not in data or "items" not in data["message"]:
            # No results found - not an error, just can't resolve
            return None

        items = data["message"]["items"]
        if not items:
            # No matches - return None to indicate unresolved
            return None

        # Fuzzy match title and select best match
        best_match = self._fuzzy_match_title(title, items)

        if best_match["score"] < 0.75:
            use_logger.warning(
                f"Low confidence match ({best_match['score']:.2f}) for title: {title}"
            )
            # Return unresolved reference instead of raising error
            return None

        use_logger.debug(
            f"Matched with score {best_match['score']:.2f}: {best_match['item'].get('title', [None])[0]}"
        )

        return self._parse_crossref_work(best_match["item"])

    def _fuzzy_match_title(self, query_title: str, items: List[dict]) -> dict:
        """
        Find best title match using simple fuzzy matching.

        Args:
            query_title: Title to search for
            items: List of Crossref work items

        Returns:
            Dict with 'item' and 'score' keys
        """
        from difflib import SequenceMatcher

        query_lower = query_title.lower()
        best_score = 0.0
        best_item = items[0]

        for item in items:
            item_title = item.get("title", [None])[0] or ""
            item_title_lower = item_title.lower()

            # Calculate string similarity
            matcher = SequenceMatcher(None, query_lower, item_title_lower)
            score = matcher.ratio()

            if score > best_score:
                best_score = score
                best_item = item

        return {"item": best_item, "score": best_score}

    def _parse_crossref_work(self, work: dict, doi: Optional[str] = None) -> Reference:
        """
        Parse Crossref work JSON into Reference object.

        Args:
            work: Crossref work data
            doi: DOI (if already known)

        Returns:
            Reference object
        """
        # Extract DOI
        if not doi:
            doi = work.get("DOI", "").replace("https://doi.org/", "")

        # Extract authors
        authors = []
        author_orcids = []
        for author in work.get("author", []):
            name_parts = []
            if author.get("family"):
                name_parts.append(author["family"])
            if author.get("given"):
                name_parts.append(author["given"])

            if name_parts:
                authors.append(", ".join(reversed(name_parts)))

            if author.get("ORCID"):
                orcid = author["ORCID"].replace("https://orcid.org/", "")
                author_orcids.append(orcid)

        # Extract publication year
        year = None
        if work.get("published"):
            # published can be {"date-parts": [[2020, 1, 15]], ...}
            date_parts = work["published"].get("date-parts", [[None]])[0]
            year = date_parts[0] if date_parts else None

        # Extract venue (container-title = journal/conference name)
        venue = work.get("container-title", [""])[0] if work.get("container-title") else None

        # Determine venue type
        venue_type = "journal"
        if work.get("type") == "conference-paper":
            venue_type = "conference"
        elif work.get("type") in ["preprint", "posted-content"]:
            venue_type = "preprint"

        # Generate reference ID
        ref_id = f"crossref_{doi}" if doi else f"crossref_{id(work)}"

        # Determine reference type (academic by default for Crossref)
        reference_type = "academic"

        return Reference(
            ref_id=ref_id,
            doi=doi,
            title=work.get("title", [""])[0] if work.get("title") else None,
            authors=authors,
            year=year,
            abstract=work.get("abstract"),
            venue=venue,
            venue_type=venue_type,
            volume=work.get("volume"),
            issue=work.get("issue"),
            pages=work.get("page"),
            reference_type=reference_type,
            resolved=True,
            status=ReferenceStatus.RESOLVED,
            metadata_provider=self.name,
            raw_citation="",  # Would be set by caller
            resolved_at=datetime.now(timezone.utc),
        )
