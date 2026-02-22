"""
URL Fetch metadata provider - last resort metadata extraction from URLs.

This provider attempts to extract metadata directly from web pages and PDFs
when all other providers have failed.

WARNING: This is a best-effort provider with lower reliability.
Should only be used as absolute last resort.

TODO: Use newspaper3k or similar library for more robust HTML parsing and metadata extraction
"""

import logging
import re
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlparse

import requests

from .base import BaseProvider, FatalError, RecoverableError, Reference, ReferenceStatus

logger = logging.getLogger(__name__)


class URLFetchProvider(BaseProvider):
    """
    URL fetch provider for direct metadata extraction.

    This provider:
    - Extracts metadata from HTML meta tags
    - Parses common metadata formats (Dublin Core, OpenGraph, etc.)
    - Attempts to extract title, authors, year from page content
    - Falls back to basic HTML parsing

    Use cases:
    - Technical reports from institutional websites
    - Government publications
    - White papers
    - Blog posts and online articles
    - Any citable web resource

    Note: This is a last resort provider with lower confidence
    """

    def __init__(self):
        self.name = "url_fetch"
        self.rate_limit = 2  # 2 req/sec
        self.timeout = 15.0
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
        Resolve citation by fetching URL and extracting metadata.

        Strategy:
        1. Extract URL from citation text
        2. Fetch URL content
        3. Parse HTML for metadata (meta tags, JSON-LD, etc.)
        4. Extract title, authors, date from content

        Args:
            citation_text: Raw citation or URL
            year: Publication year (optional)
            doi: DOI if available (optional)
            authors: Author list (optional)
            logger: Optional logger instance (optional)

        Returns:
            Reference with extracted metadata
        """
        # Set up logger
        use_logger = logger or globals().get("logger")
        if use_logger is None:
            use_logger = logging.getLogger(__name__)

        # Try to extract URL from citation
        url = self._extract_url(citation_text)

        if not url:
            return Reference(
                ref_id=f"url_unresolved_{hash(citation_text)}",
                raw_citation=citation_text,
                resolved=False,
                status=ReferenceStatus.UNRESOLVED,
                metadata_provider=self.name,
            )

        try:
            return self._resolve_by_url(url, citation_text)
        except (RecoverableError, FatalError):
            # Return unresolved reference
            return Reference(
                ref_id=f"url_unresolved_{hash(url)}",
                raw_citation=citation_text,
                resolved=False,
                status=ReferenceStatus.UNRESOLVED,
                metadata_provider=self.name,
            )

    def _extract_url(self, text: str) -> Optional[str]:
        """Extract URL from citation text, cleaning up spaces and line breaks."""
        # Match http/https URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;:!?\)\]]'
        match = re.search(url_pattern, text)

        if match:
            url = match.group(0)
            # Strip leading/trailing whitespace
            url = url.strip()
            # Remove spaces/line breaks within the URL (common in PDF extractions)
            # e.g., "https://example.com/path- \nmore" -> "https://example.com/path-more"
            url = re.sub(r"\s+", "", url)
            # Remove common trailing punctuation that's not part of URL
            url = re.sub(r"[.,;:!?\)\]]*$", "", url)
            return url if url else None

        return None

    def _resolve_by_url(self, url: str, original_citation: str) -> Reference:
        """Resolve by fetching URL and parsing metadata."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        try:
            response = self._request_with_retry("GET", url, headers=headers)

            # Check content type
            content_type = response.headers.get("Content-Type", "").lower()

            if "pdf" in content_type:
                # Handle PDF - extract from headers/URL only
                return self._parse_pdf_metadata(url, response)
            elif "html" in content_type or not content_type:
                # Parse HTML
                return self._parse_html_metadata(url, response.text, original_citation)
            else:
                raise FatalError(f"Unsupported content type: {content_type}")

        except Exception as e:
            if isinstance(e, (RecoverableError, FatalError)):
                raise
            raise RecoverableError(f"URL fetch failed: {e}")

    def _parse_html_metadata(self, url: str, html: str, original_citation: str) -> Reference:
        """Parse metadata from HTML page."""
        metadata = {
            "title": None,
            "authors": [],
            "year": None,
            "abstract": None,
            "doi": None,
        }

        # 1. Try meta tags (Dublin Core, OpenGraph, Twitter Card, etc.)

        # Title
        title_patterns = [
            r'<meta\s+name=["\'](?:dc\.title|citation_title|og:title|twitter:title)["\'].*?content=["\']([^"\']+)["\']',
            r'<meta\s+property=["\'](?:og:title|article:title)["\'].*?content=["\']([^"\']+)["\']',
            r"<title[^>]*>(.*?)</title>",
        ]

        for pattern in title_patterns:
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                metadata["title"] = match.group(1).strip()
                break

        # Authors
        author_patterns = [
            r'<meta\s+name=["\'](?:dc\.creator|citation_author|author)["\'].*?content=["\']([^"\']+)["\']',
            r'<meta\s+property=["\'](?:article:author)["\'].*?content=["\']([^"\']+)["\']',
        ]

        for pattern in author_patterns:
            matches = re.finditer(pattern, html, re.IGNORECASE)
            for match in matches:
                author = match.group(1).strip()
                if author and author not in metadata["authors"]:
                    metadata["authors"].append(author)

        # Year/Date
        date_patterns = [
            r'<meta\s+name=["\'](?:dc\.date|citation_publication_date|article:published_time)["\'].*?content=["\']([^"\']+)["\']',
            r'<meta\s+property=["\'](?:article:published_time)["\'].*?content=["\']([^"\']+)["\']',
        ]

        for pattern in date_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                # Extract year (first 4 digits)
                year_match = re.search(r"(19|20)\d{2}", date_str)
                if year_match:
                    try:
                        metadata["year"] = int(year_match.group(0))
                        break
                    except ValueError:
                        pass

        # Abstract/Description
        abstract_patterns = [
            r'<meta\s+name=["\'](?:dc\.description|citation_abstract|description)["\'].*?content=["\']([^"\']+)["\']',
            r'<meta\s+property=["\'](?:og:description)["\'].*?content=["\']([^"\']+)["\']',
        ]

        for pattern in abstract_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                metadata["abstract"] = match.group(1).strip()
                break

        # DOI
        doi_patterns = [
            r'<meta\s+name=["\'](?:dc\.identifier|citation_doi)["\'].*?content=["\'](?:doi:)?([^"\']+)["\']',
            r'doi\.org/(10\.\d{4,}/[^\s<>"]+)',
        ]

        for pattern in doi_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                doi = match.group(1).strip()
                # Clean DOI
                doi = doi.rstrip(".,;")
                if doi.startswith("10."):
                    metadata["doi"] = doi
                    break

        # 2. Fallback: Extract from page title if no meta tags
        if not metadata["title"]:
            title_match = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
            if title_match:
                metadata["title"] = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()

        # 3. Use original citation as title if nothing found
        if not metadata["title"]:
            metadata["title"] = original_citation[:200]

        # Determine reference type from URL and metadata
        reference_type, venue_type = self._determine_type_from_url(url, metadata)

        # Build venue from URL domain
        parsed_url = urlparse(url)
        venue = parsed_url.netloc.replace("www.", "")

        # Build reference ID
        ref_id = metadata["doi"] if metadata["doi"] else f"url_{hash(url)}"

        # Compute quality score (lower for URL fetch)
        quality_score = self._compute_quality_score(metadata)

        return Reference(
            ref_id=ref_id,
            raw_citation=original_citation,
            title=metadata["title"],
            authors=metadata["authors"],
            year=metadata["year"],
            abstract=metadata["abstract"][:500] if metadata["abstract"] else "",
            venue=venue,
            venue_type=venue_type,
            doi=metadata["doi"],
            reference_type=reference_type,
            resolved=True,
            status=ReferenceStatus.RESOLVED,
            quality_score=quality_score,
            metadata_provider=self.name,
            oa_available=True,  # URL is accessible
            oa_url=url,
            resolved_at=datetime.now(timezone.utc),
        )

    def _parse_pdf_metadata(self, url: str, response) -> Reference:
        """Parse metadata from PDF (headers/URL only, no content extraction)."""
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = parsed_url.path.split("/")[-1]

        # Use filename as title (remove .pdf)
        title = filename.replace(".pdf", "").replace("_", " ").replace("-", " ").strip()

        # Clean up title
        title = re.sub(r"\s+", " ", title)

        # Try to extract year from filename
        year = None
        year_match = re.search(r"(19|20)\d{2}", filename)
        if year_match:
            try:
                year = int(year_match.group(0))
            except ValueError:
                pass

        # Build venue from domain
        venue = parsed_url.netloc.replace("www.", "")

        return Reference(
            ref_id=f"pdf_{hash(url)}",
            raw_citation=url,
            title=title,
            authors=[],
            year=year,
            abstract="",
            venue=venue,
            venue_type="online",
            doi=None,
            reference_type="technical",
            resolved=True,
            status=ReferenceStatus.RESOLVED,
            quality_score=0.40,  # Low quality for PDF without metadata
            metadata_provider=self.name,
            oa_available=True,
            oa_url=url,
            resolved_at=datetime.now(timezone.utc),
        )

    def _determine_type_from_url(self, url: str, metadata: dict) -> tuple:
        """Determine reference and venue type from URL and metadata."""
        url_lower = url.lower()

        # Government sites
        if ".gov" in url_lower or "government" in url_lower:
            return ("government", "government")

        # Academic institutions
        if ".edu" in url_lower or ".ac." in url_lower:
            return ("technical", "academic")

        # arXiv
        if "arxiv" in url_lower:
            return ("preprint", "preprint")

        # Known repositories
        if any(site in url_lower for site in ["github", "gitlab", "bitbucket"]):
            return ("software", "repository")

        # Blog platforms
        if any(site in url_lower for site in ["medium.com", "wordpress", "blogger", "substack"]):
            return ("blog", "blog")

        # News sites
        if any(site in url_lower for site in ["news", "times", "post", "guardian", "bbc"]):
            return ("news", "news")

        # Technical documentation
        if any(word in url_lower for word in ["docs", "documentation", "manual", "guide"]):
            return ("technical", "documentation")

        # Default to online resource
        return ("online", "online")

    def _compute_quality_score(self, metadata: dict) -> float:
        """Compute quality score for URL-fetched metadata."""
        # Base score very low for URL fetch
        score = 0.30

        # Has DOI: +0.30 (major boost)
        if metadata.get("doi"):
            score += 0.30

        # Has authors: +0.15
        if metadata.get("authors"):
            score += 0.15

        # Has year: +0.10
        if metadata.get("year"):
            score += 0.10

        # Has abstract: +0.10
        if metadata.get("abstract"):
            score += 0.10

        # Multiple authors: +0.05
        if len(metadata.get("authors", [])) >= 2:
            score += 0.05

        return min(1.0, score)
