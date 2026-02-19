"""
ORCID metadata provider for author-linked publications.

ORCID (Open Researcher and Contributor ID) provides a persistent identifier
for researchers and links to their research outputs.

API: https://pub.orcid.org/
Rate limit: No strict limit, but be respectful (5 req/sec recommended)
Documentation: https://info.orcid.org/documentation/api-tutorials/
"""

import logging
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import List, Optional

from .base import BaseProvider, FatalError, RecoverableError, Reference, ReferenceStatus

logger = logging.getLogger(__name__)


class ORCIDProvider(BaseProvider):
    """
    ORCID metadata provider for author-linked publications.

    ORCID provides:
    - Author-verified publication lists
    - High-confidence author attribution
    - Links to other metadata sources
    - DOI resolution for author works

    Note: Best used when ORCID ID is known or can be inferred from author name
    """

    BASE_URL = "https://pub.orcid.org/v3.0"
    SEARCH_URL = "https://pub.orcid.org/v3.0/search"

    def __init__(self):
        self.name = "orcid"
        self.rate_limit = 5  # 5 req/sec - conservative
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
        Resolve citation via ORCID.

        Strategy:
        1. Extract ORCID ID if present in citation
        2. Search by author name to find ORCID
        3. Search author's works for matching publication

        Note: ORCID is most effective when author information is available

        Args:
            citation_text: Raw citation or title
            year: Publication year (optional)
            doi: DOI if available (optional)
            authors: Author list (optional, highly recommended)
            logger: Optional logger instance (optional)

        Returns:
            Reference with resolved metadata
        """
        # Set up logger
        use_logger = logger or globals().get("logger")
        if use_logger is None:
            use_logger = logging.getLogger(__name__)

        # Try to extract ORCID ID from citation
        orcid_id = self._extract_orcid_id(citation_text)

        if orcid_id:
            try:
                return self._resolve_by_orcid_works(orcid_id, citation_text, year, doi)
            except (RecoverableError, FatalError) as e:
                use_logger.debug(f"ORCID ID lookup failed: {e}")

        # If we have author names, try to find their ORCID and search works
        if authors:
            try:
                return self._resolve_by_author_search(authors[0], citation_text, year, doi)
            except (RecoverableError, FatalError) as e:
                use_logger.debug(f"Author search failed: {e}")

        # Return unresolved - ORCID requires author context
        return Reference(
            ref_id=f"orcid_unresolved_{hash(citation_text)}",
            raw_citation=citation_text,
            resolved=False,
            status=ReferenceStatus.UNRESOLVED,
            metadata_provider=self.name,
        )

    def _extract_orcid_id(self, text: str) -> Optional[str]:
        """Extract ORCID ID from citation text."""
        # Match patterns like: 0000-0002-1825-0097, orcid.org/0000-0002-1825-0097
        patterns = [
            r"orcid\.org/(\d{4}-\d{4}-\d{4}-\d{3}[0-9X])",
            r"\b(\d{4}-\d{4}-\d{4}-\d{3}[0-9X])\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _resolve_by_orcid_works(
        self,
        orcid_id: str,
        title: str,
        year: Optional[int] = None,
        doi: Optional[str] = None,
    ) -> Reference:
        """Resolve by searching ORCID researcher's works."""
        url = f"{self.BASE_URL}/{orcid_id}/works"
        headers = {"Accept": "application/json"}

        try:
            response = self._request_with_retry("GET", url, headers=headers)
            data = response.json()

            # Search through works for matching title or DOI
            works = data.get("group", [])

            for work_group in works:
                work_summary = work_group.get("work-summary", [])
                if not work_summary:
                    continue

                work = work_summary[0]  # Take first summary

                # Check DOI match
                if doi:
                    external_ids = work.get("external-ids", {}).get("external-id", [])
                    for ext_id in external_ids:
                        if ext_id.get("external-id-type") == "doi":
                            work_doi = ext_id.get("external-id-value", "")
                            if work_doi.lower() == doi.lower():
                                return self._parse_orcid_work(work, orcid_id)

                # Check title match
                work_title = work.get("title", {}).get("title", {}).get("value", "")
                if work_title:
                    score = self._fuzzy_match_title(title, work_title)
                    if score > 0.85:
                        # Additional year check if provided
                        if year:
                            pub_date = work.get("publication-date")
                            if pub_date:
                                work_year = pub_date.get("year", {}).get("value")
                                if work_year and int(work_year) != year:
                                    continue

                        return self._parse_orcid_work(work, orcid_id)

            raise FatalError(f"No matching work found in ORCID {orcid_id}")

        except Exception as e:
            if isinstance(e, (RecoverableError, FatalError)):
                raise
            raise RecoverableError(f"ORCID works lookup failed: {e}")

    def _resolve_by_author_search(
        self,
        author: str,
        title: str,
        year: Optional[int] = None,
        doi: Optional[str] = None,
    ) -> Reference:
        """Resolve by searching for author's ORCID first."""
        # Search for ORCID by author name
        # Note: This is simplified - in production, author name parsing would be more robust
        author_parts = author.replace(",", "").split()
        if len(author_parts) < 2:
            raise FatalError("Insufficient author information for ORCID search")

        # Try to find ORCID for author
        # For now, return unresolved as ORCID search API requires more complex implementation
        # In production, you would:
        # 1. Search ORCID for author
        # 2. Get top matching ORCID IDs
        # 3. Search each ORCID's works

        raise FatalError("ORCID author search not implemented - requires ORCID ID")

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

    def _parse_orcid_work(self, work: dict, orcid_id: str) -> Reference:
        """Parse ORCID work summary."""
        # Extract title
        title = work.get("title", {}).get("title", {}).get("value", "")

        # Extract year
        year = None
        pub_date = work.get("publication-date")
        if pub_date:
            year_value = pub_date.get("year", {}).get("value")
            if year_value:
                try:
                    year = int(year_value)
                except (ValueError, TypeError):
                    pass

        # Extract DOI and other external IDs
        doi = None
        external_ids = work.get("external-ids", {}).get("external-id", [])
        for ext_id in external_ids:
            if ext_id.get("external-id-type") == "doi":
                doi = ext_id.get("external-id-value", "")
                break

        # Extract work type
        work_type = work.get("type", "")
        reference_type = self._determine_reference_type(work_type)
        venue_type = self._determine_venue_type(work_type)

        # Extract journal/venue
        journal = work.get("journal-title", {}).get("value", "")

        # Build reference ID
        ref_id = doi if doi else f"orcid_{orcid_id}_{hash(title)}"

        # Compute quality score
        quality_score = self._compute_quality_score(work)

        return Reference(
            ref_id=ref_id,
            raw_citation="",
            title=title,
            authors=[],  # ORCID work summary doesn't include full author list
            year=year,
            abstract="",  # Not available in work summary
            venue=journal,
            venue_type=venue_type,
            doi=doi,
            reference_type=reference_type,
            resolved=True,
            status=ReferenceStatus.RESOLVED,
            quality_score=quality_score,
            metadata_provider=self.name,
            oa_available=False,  # Would need to check external links
            oa_url=None,
            resolved_at=datetime.now(timezone.utc),
        )

    def _determine_reference_type(self, work_type: str) -> str:
        """Determine reference type from ORCID work type."""
        type_mapping = {
            "journal-article": "academic",
            "conference-paper": "conference",
            "book": "academic",
            "book-chapter": "academic",
            "dissertation": "academic",
            "preprint": "preprint",
            "report": "technical",
            "working-paper": "preprint",
            "data-set": "dataset",
            "software": "software",
        }

        return type_mapping.get(work_type.lower(), "other")

    def _determine_venue_type(self, work_type: str) -> str:
        """Determine venue type from ORCID work type."""
        type_mapping = {
            "journal-article": "journal",
            "conference-paper": "conference",
            "book": "book",
            "book-chapter": "book",
            "dissertation": "academic",
            "preprint": "preprint",
            "report": "technical",
            "working-paper": "preprint",
            "data-set": "repository",
            "software": "repository",
        }

        return type_mapping.get(work_type.lower(), "other")

    def _compute_quality_score(self, work: dict) -> float:
        """Compute quality score for ORCID work."""
        # Base score for ORCID-verified work
        score = 0.80

        # Has DOI: +0.10
        external_ids = work.get("external-ids", {}).get("external-id", [])
        has_doi = any(ext.get("external-id-type") == "doi" for ext in external_ids)
        if has_doi:
            score += 0.10

        # Has journal title: +0.05
        if work.get("journal-title", {}).get("value"):
            score += 0.05

        # Has publication date: +0.05
        if work.get("publication-date"):
            score += 0.05

        return min(1.0, score)
