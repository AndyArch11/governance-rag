"""
arXiv metadata provider for preprint resolution.

arXiv is a free distribution service and open-access archive for scholarly
articles in physics, mathematics, computer science, and related disciplines.

API: https://arxiv.org/help/api/
Rate limit: No strict limit, but respect their servers (3 req/sec recommended)
Documentation: https://arxiv.org/help/api/user-manual
"""

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import List, Optional
from urllib.parse import quote

from .base import BaseProvider, FatalError, RecoverableError, Reference, ReferenceStatus

logger = logging.getLogger(__name__)


class ArxivProvider(BaseProvider):
    """
    arXiv metadata provider.

    arXiv provides:
    - Free access to preprints
    - High-quality metadata for STEM fields
    - Full-text PDF links
    - Author information
    - Category/subject classification
    """

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self):
        self.name = "arxiv"
        self.rate_limit = 3  # 3 req/sec - conservative
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
        Resolve citation via arXiv.

        Priority:
        1. arXiv ID lookup (if detected in citation)
        2. Title search
        3. Author + title search

        Args:
            citation_text: Raw citation or title
            year: Publication year (optional)
            doi: DOI if available (optional, not used by arXiv)
            authors: Author list (optional)
            logger: Optional logger instance (optional)

        Returns:
            Reference with resolved metadata
        """
        # Set up logger
        use_logger = logger or globals().get("logger")
        if use_logger is None:
            use_logger = logging.getLogger(__name__)

        # Try to extract arXiv ID from citation
        arxiv_id = self._extract_arxiv_id(citation_text)

        if arxiv_id:
            try:
                return self._resolve_by_arxiv_id(arxiv_id)
            except (RecoverableError, FatalError) as e:
                use_logger.debug(f"arXiv ID lookup failed: {e}, trying search")

        # Try title search
        try:
            return self._resolve_by_search(citation_text, year, authors)
        except FatalError:
            # Return unresolved reference
            return Reference(
                ref_id=f"arxiv_unresolved_{hash(citation_text)}",
                raw_citation=citation_text,
                resolved=False,
                status=ReferenceStatus.UNRESOLVED,
                metadata_provider=self.name,
            )

    def _extract_arxiv_id(self, text: str) -> Optional[str]:
        """Extract arXiv ID from citation text."""
        # Match patterns like: arXiv:1234.5678, arXiv:1234.5678v2, 1234.5678
        patterns = [
            r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)",
            r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",
            r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _resolve_by_arxiv_id(self, arxiv_id: str) -> Reference:
        """Resolve by arXiv ID."""
        url = f"{self.BASE_URL}?id_list={quote(arxiv_id)}"

        try:
            response = self._request_with_retry("GET", url)
            return self._parse_arxiv_response(response.text)

        except Exception as e:
            if isinstance(e, (RecoverableError, FatalError)):
                raise
            raise RecoverableError(f"arXiv ID lookup failed: {e}")

    def _resolve_by_search(
        self,
        title: str,
        year: Optional[int] = None,
        authors: Optional[List[str]] = None,
    ) -> Reference:
        """Resolve via arXiv search."""
        clean_title = self._extract_title_from_citation(title)
        if not clean_title:
            raise FatalError("Empty title after sanitisation")

        # Build search query
        query_parts = [f'ti:"{clean_title}"']

        if authors:
            # Use first author
            author = authors[0].split(",")[0].strip()
            query_parts.append(f'au:"{author}"')

        query = " AND ".join(query_parts)
        url = f"{self.BASE_URL}?search_query={quote(query)}&max_results=10"

        try:
            response = self._request_with_retry("GET", url)

            # Parse XML response
            root = ET.fromstring(response.text)

            # Find entries
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")

            if not entries:
                raise FatalError(f"No arXiv results for: {title}")

            # Find best match
            best_match = None
            best_score = 0.75

            for entry in entries:
                entry_title = entry.find("{http://www.w3.org/2005/Atom}title")
                if entry_title is not None and entry_title.text:
                    score = self._fuzzy_match_title(clean_title, entry_title.text)
                    if score > best_score:
                        best_score = score
                        best_match = entry

            if best_match is None:
                raise FatalError(f"Low confidence match for: {title}")

            return self._parse_arxiv_entry(best_match)

        except Exception as e:
            if isinstance(e, (RecoverableError, FatalError)):
                raise
            raise RecoverableError(f"arXiv search failed: {e}")

    def _extract_title_from_citation(self, citation_text: str) -> str:
        """Extract a clean title from citation text for arXiv queries."""
        if not citation_text:
            return ""

        text = citation_text

        # Remove URLs (including DOI URLs)
        text = re.sub(r"https?://\S+", " ", text)

        # Remove DOI tokens (non-URL DOIs)
        text = re.sub(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", " ", text, flags=re.IGNORECASE)

        # Try to extract title after year pattern: "(2015). Title."
        title_match = re.search(r"\(\d{4}\)\.\s*([^\.]+)", text)
        if title_match:
            title = title_match.group(1).strip()
        else:
            title = text

        # Normalise whitespace and trim
        title = re.sub(r"\s+", " ", title).strip()

        # Limit length for arXiv title queries
        return title[:200]

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

    def _parse_arxiv_response(self, xml_text: str) -> Reference:
        """Parse arXiv API XML response."""
        root = ET.fromstring(xml_text)
        entry = root.find("{http://www.w3.org/2005/Atom}entry")

        if entry is None:
            raise FatalError("No entry found in arXiv response")

        return self._parse_arxiv_entry(entry)

    def _parse_arxiv_entry(self, entry: ET.Element) -> Reference:
        """Parse arXiv entry XML element."""
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

        # Extract arXiv ID
        arxiv_id_elem = entry.find("atom:id", ns)
        arxiv_id = arxiv_id_elem.text.split("/")[-1] if arxiv_id_elem is not None else ""

        # Extract title
        title_elem = entry.find("atom:title", ns)
        title = title_elem.text.strip() if title_elem is not None else ""

        # Extract authors
        authors = []
        for author_elem in entry.findall("atom:author", ns):
            name_elem = author_elem.find("atom:name", ns)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text)

        # Extract abstract
        summary_elem = entry.find("atom:summary", ns)
        abstract = summary_elem.text.strip() if summary_elem is not None else ""

        # Extract publication date
        published_elem = entry.find("atom:published", ns)
        year = None
        if published_elem is not None and published_elem.text:
            try:
                year = int(published_elem.text[:4])
            except (ValueError, IndexError):
                pass

        # Extract categories
        categories = []
        for category_elem in entry.findall("atom:category", ns):
            term = category_elem.get("term")
            if term:
                categories.append(term)

        # Primary category
        primary_category_elem = entry.find("arxiv:primary_category", ns)
        venue = primary_category_elem.get("term") if primary_category_elem is not None else ""

        # Extract DOI if present
        doi = None
        doi_elem = entry.find("arxiv:doi", ns)
        if doi_elem is not None and doi_elem.text:
            doi = doi_elem.text

        # PDF URL
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # Compute quality score
        quality_score = self._compute_quality_score(entry, ns)

        return Reference(
            ref_id=arxiv_id,
            raw_citation="",
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            venue=f"arXiv:{venue}" if venue else "arXiv",
            venue_type="preprint",
            doi=doi,
            reference_type="preprint",
            resolved=True,
            status=ReferenceStatus.RESOLVED,
            quality_score=quality_score,
            metadata_provider=self.name,
            oa_available=True,  # arXiv is always open access
            oa_url=pdf_url,
            resolved_at=datetime.now(timezone.utc),
        )

    def _compute_quality_score(self, entry: ET.Element, ns: dict) -> float:
        """Compute quality score for arXiv entry."""
        score = 0.75  # Base score for preprints

        # Has DOI (published version exists): +0.1
        doi_elem = entry.find("arxiv:doi", ns)
        if doi_elem is not None and doi_elem.text:
            score += 0.1

        # Has abstract: +0.05
        summary_elem = entry.find("atom:summary", ns)
        if summary_elem is not None and summary_elem.text:
            score += 0.05

        # Multiple authors (collaborative work): +0.05
        authors = entry.findall("atom:author", ns)
        if len(authors) >= 3:
            score += 0.05

        # Open access: +0.05
        score += 0.05  # Always OA

        return min(1.0, score)
