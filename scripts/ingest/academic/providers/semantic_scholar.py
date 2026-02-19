"""
Semantic Scholar metadata provider for AI-powered reference resolution.

Semantic Scholar uses machine learning and NLP to extract and organise
academic papers and their relationships.

API: https://api.semanticscholar.org/
Rate limit: 1 request/second per IP (100 requests/minute recommended)
Documentation: https://api.semanticscholar.org/oa/api/v1/papers/search
"""

import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import List, Optional
from urllib.parse import quote

from .base import BaseProvider, FatalError, RecoverableError, Reference, ReferenceStatus

logger = logging.getLogger(__name__)


class SemanticScholarProvider(BaseProvider):
    """
    Semantic Scholar metadata provider.

    Semantic Scholar provides:
    - Full-text search with AI understanding
    - Citation context and influence
    - Author profiles and collaboration networks
    - Paper similarity and recommendations
    - High-quality metadata extraction via ML
    """

    BASE_URL = "https://api.semanticscholar.org/oa"

    def __init__(self):
        self.name = "semantic_scholar"
        self.rate_limit = 1  # 1 req/sec - conservative to stay under limits
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
        Resolve citation via Semantic Scholar.

        Priority:
        1. DOI lookup (if provided)
        2. Full-text search with all available metadata

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
                use_logger.debug(f"DOI lookup failed: {e}, trying search")

        # Try full-text search with all available info
        try:
            return self._resolve_by_search(citation_text, year, authors)
        except FatalError:
            # Return unresolved reference
            return Reference(
                ref_id=f"semantic_scholar_unresolved_{hash(citation_text)}",
                raw_citation=citation_text,
                resolved=False,
                status=ReferenceStatus.UNRESOLVED,
                metadata_provider=self.name,
            )

    def _resolve_by_doi(self, doi: str) -> Reference:
        """Resolve by DOI via Semantic Scholar."""
        # Normalise DOI
        doi_clean = doi.lower().replace("https://doi.org/", "").replace("http://dx.doi.org/", "")

        url = f"{self.BASE_URL}/papers/search?query={quote(doi_clean)}"

        try:
            response = self._request_with_retry("GET", url)
            data = response.json()

            papers = data.get("data", [])
            if not papers:
                raise FatalError(f"DOI not found: {doi_clean}")

            # Find exact DOI match
            for paper in papers:
                if paper.get("externalIds", {}).get("DOI", "").lower() == doi_clean:
                    return self._parse_semantic_scholar_paper(paper)

            # Fallback to first result if no exact match
            return self._parse_semantic_scholar_paper(papers[0])

        except Exception as e:
            if isinstance(e, (RecoverableError, FatalError)):
                raise
            raise RecoverableError(f"Semantic Scholar DOI lookup failed: {e}")

    def _resolve_by_search(
        self,
        title: str,
        year: Optional[int] = None,
        authors: Optional[List[str]] = None,
    ) -> Reference:
        """Resolve via full-text search."""
        # Build query with all available metadata
        query_parts = [title]
        if year:
            query_parts.append(str(year))
        if authors:
            query_parts.append(" ".join(authors[:2]))  # First 2 authors

        query = " ".join(query_parts)

        url = f"{self.BASE_URL}/papers/search?query={quote(query)}&limit=50"

        try:
            response = self._request_with_retry("GET", url)
            data = response.json()

            papers = data.get("data", [])
            if not papers:
                raise FatalError(f"No results for: {title}")

            # Find best match considering title, year, and authors
            best_match = None
            best_score = 0.0

            for paper in papers:
                paper_title = paper.get("title", "")
                if not paper_title:
                    continue

                # Compute match score
                score = self._compute_match_score(
                    title, year, authors, paper_title, paper.get("year"), paper.get("authors", [])
                )

                if score > best_score:
                    best_score = score
                    best_match = paper

            if not best_match or best_score < 0.75:
                raise FatalError(f"Low confidence match for: {title}")

            return self._parse_semantic_scholar_paper(best_match)

        except Exception as e:
            if isinstance(e, (RecoverableError, FatalError)):
                raise
            raise RecoverableError(f"Semantic Scholar search failed: {e}")

    def _compute_match_score(
        self,
        query_title: str,
        query_year: Optional[int],
        query_authors: Optional[List[str]],
        paper_title: str,
        paper_year: Optional[int],
        paper_authors: List[dict],
    ) -> float:
        """Compute match score between query and paper."""
        score = 0.0

        # Title similarity (0.0 - 0.7)
        title_sim = self._fuzzy_match_title(query_title, paper_title)
        score += title_sim * 0.7

        # Year match (0.0 - 0.15)
        if query_year and paper_year:
            year_diff = abs(query_year - paper_year)
            if year_diff == 0:
                score += 0.15
            elif year_diff <= 1:
                score += 0.1
            elif year_diff <= 2:
                score += 0.05

        # Author match (0.0 - 0.15)
        if query_authors:
            paper_author_names = [a.get("name", "").split()[-1].lower() for a in paper_authors]
            matched_authors = 0
            for qa in query_authors[:2]:  # Check first 2 query authors
                qa_last = qa.split(",")[-1].lower().strip()
                if any(qa_last in pa or pa in qa_last for pa in paper_author_names):
                    matched_authors += 1

            score += (matched_authors / max(len(query_authors), 1)) * 0.15

        return min(1.0, score)

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

    def _parse_semantic_scholar_paper(self, paper: dict) -> Reference:
        """Parse Semantic Scholar paper into Reference."""
        # Extract authors
        authors = []
        for author in paper.get("authors", []):
            name = author.get("name", "")
            if name:
                authors.append(name)

        # Extract URLs
        doi = None
        if paper.get("externalIds", {}).get("DOI"):
            doi = paper["externalIds"]["DOI"]

        # Determine venue type
        venue_type = paper.get("venue_type", "").lower()
        if not venue_type or venue_type == "unknown":
            venue_type = "journal" if paper.get("publicationTypes") else "article"

        # Publication types for reference type
        pub_types = paper.get("publicationTypes", [])
        reference_type = self._determine_reference_type(pub_types)

        # Compute quality score
        quality_score = self._compute_quality_score(paper)

        # Extract citations and influence
        citation_count = paper.get("citationCount", 0)
        influence_score = paper.get("influentialCitationCount", 0)

        return Reference(
            ref_id=paper.get("paperId", ""),
            raw_citation="",  # Set by caller if needed
            title=paper.get("title", ""),
            authors=authors,
            year=paper.get("year"),
            abstract=paper.get("abstract", ""),
            venue=paper.get("venue", ""),
            venue_type=venue_type,
            doi=doi,
            reference_type=reference_type,
            resolved=True,
            status=ReferenceStatus.RESOLVED,
            quality_score=quality_score,
            metadata_provider=self.name,
            oa_available=paper.get("isOpenAccess", False),
            oa_url=(
                paper.get("openAccessPdf", {}).get("url") if paper.get("openAccessPdf") else None
            ),
            resolved_at=datetime.now(timezone.utc),
        )

    def _compute_quality_score(self, paper: dict) -> float:
        """Compute quality score for Semantic Scholar paper."""
        score = 0.5  # Base score

        # Has DOI: +0.2
        if paper.get("externalIds", {}).get("DOI"):
            score += 0.2

        # Is open access: +0.1
        if paper.get("isOpenAccess"):
            score += 0.1

        # Has abstract: +0.05
        if paper.get("abstract"):
            score += 0.05

        # Citation impact: +0.1 (normalised by citation count)
        citation_count = paper.get("citationCount", 0)
        if citation_count > 0:
            score += min(0.1, citation_count / 1000)  # Cap at 0.1

        # Influential citations: +0.05
        influence_count = paper.get("influentialCitationCount", 0)
        if influence_count > 0:
            score += min(0.05, influence_count / 100)  # Cap at 0.05

        # Publication type boost
        pub_types = paper.get("publicationTypes", [])
        if "JournalArticle" in pub_types:
            score += 0.1
        elif "Conference" in pub_types:
            score += 0.08
        elif "Preprint" in pub_types:
            score -= 0.05

        return min(1.0, score)

    def _determine_reference_type(self, pub_types: List[str]) -> str:
        """Determine reference type from publication types."""
        if "JournalArticle" in pub_types:
            return "academic"
        elif "Conference" in pub_types:
            return "conference"
        elif "Preprint" in pub_types:
            return "preprint"
        elif "Report" in pub_types or "Book" in pub_types:
            return "technical_report"
        else:
            return "online"
