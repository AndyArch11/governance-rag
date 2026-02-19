"""
PubMed/PubMed Central metadata provider for biomedical literature.

PubMed is a database of biomedical and life sciences literature maintained
by the National Library of Medicine (NLM).

API: https://www.ncbi.nlm.nih.gov/books/NBK25501/
Rate limit: 3 req/sec without API key, 10 req/sec with key
Documentation: https://www.ncbi.nlm.nih.gov/home/develop/api/
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


class PubMedProvider(BaseProvider):
    """
    PubMed metadata provider for biomedical literature.

    PubMed provides:
    - Comprehensive biomedical literature coverage
    - High-quality metadata
    - PMC open access links
    - MeSH terms and keywords
    - Author affiliations
    """

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise PubMed provider.

        Args:
            api_key: NCBI API key (optional, increases rate limit to 10 req/sec)
        """
        self.name = "pubmed"
        self.api_key = api_key
        self.rate_limit = 10 if api_key else 3  # Higher rate with API key
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
        Resolve citation via PubMed.

        Priority:
        1. DOI lookup
        2. PMID lookup (if detected in citation)
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

        # Try DOI first
        if doi:
            try:
                return self._resolve_by_doi(doi)
            except (RecoverableError, FatalError) as e:
                use_logger.debug(f"DOI lookup failed: {e}, trying other methods")

        # Try to extract PMID
        pmid = self._extract_pmid(citation_text)
        if pmid:
            try:
                return self._resolve_by_pmid(pmid)
            except (RecoverableError, FatalError) as e:
                use_logger.debug(f"PMID lookup failed: {e}, trying search")

        # Try title + author search
        if authors:
            try:
                return self._resolve_by_search(citation_text, year, authors)
            except (RecoverableError, FatalError) as e:
                use_logger.debug(f"Title + author search failed: {e}, trying title-only")

        # Try title-only search
        try:
            return self._resolve_by_search(citation_text, year, None)
        except FatalError:
            # Return unresolved reference
            return Reference(
                ref_id=f"pubmed_unresolved_{hash(citation_text)}",
                raw_citation=citation_text,
                resolved=False,
                status=ReferenceStatus.UNRESOLVED,
                metadata_provider=self.name,
            )

    def _extract_pmid(self, text: str) -> Optional[str]:
        """Extract PMID from citation text."""
        # Match patterns like: PMID: 12345678, PMID:12345678, PMID 12345678
        patterns = [
            r"PMID:?\s*(\d{7,8})",
            r"pubmed/(\d{7,8})",
            r"\bPM(\d{7,8})\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _resolve_by_doi(self, doi: str) -> Reference:
        """Resolve by DOI."""
        # Search for PMID via DOI
        params = {
            "db": "pubmed",
            "term": f"{doi}[DOI]",
            "retmode": "json",
        }

        if self.api_key:
            params["api_key"] = self.api_key

        try:
            response = self._request_with_retry("GET", self.ESEARCH_URL, params=params)
            data = response.json()

            pmid_list = data.get("esearchresult", {}).get("idlist", [])
            if not pmid_list:
                raise FatalError(f"No PubMed entry for DOI: {doi}")

            pmid = pmid_list[0]
            return self._resolve_by_pmid(pmid)

        except Exception as e:
            if isinstance(e, (RecoverableError, FatalError)):
                raise
            raise RecoverableError(f"PubMed DOI lookup failed: {e}")

    def _resolve_by_pmid(self, pmid: str) -> Reference:
        """Resolve by PMID."""
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
        }

        if self.api_key:
            params["api_key"] = self.api_key

        try:
            response = self._request_with_retry("GET", self.EFETCH_URL, params=params)
            return self._parse_pubmed_xml(response.text, pmid)

        except Exception as e:
            if isinstance(e, (RecoverableError, FatalError)):
                raise
            raise RecoverableError(f"PubMed PMID lookup failed: {e}")

    def _resolve_by_search(
        self,
        title: str,
        year: Optional[int] = None,
        authors: Optional[List[str]] = None,
    ) -> Reference:
        """Resolve via PubMed search."""
        # Build search query
        query_parts = [f'"{title}"[Title]']

        if year:
            query_parts.append(f"{year}[Publication Date]")

        if authors:
            # Use first author
            author = authors[0].split(",")[0].strip()
            query_parts.append(f'"{author}"[Author]')

        query = " AND ".join(query_parts)

        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": 10,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        try:
            response = self._request_with_retry("GET", self.ESEARCH_URL, params=params)
            data = response.json()

            pmid_list = data.get("esearchresult", {}).get("idlist", [])
            if not pmid_list:
                raise FatalError(f"No PubMed results for: {title}")

            # Fetch details for top results
            for pmid in pmid_list[:5]:
                try:
                    ref = self._resolve_by_pmid(pmid)

                    # Check title match
                    if ref.title and self._fuzzy_match_title(title, ref.title) > 0.75:
                        return ref

                except Exception as e:
                    logger.debug(f"Failed to fetch PMID {pmid}: {e}")
                    continue

            raise FatalError(f"Low confidence match for: {title}")

        except Exception as e:
            if isinstance(e, (RecoverableError, FatalError)):
                raise
            raise RecoverableError(f"PubMed search failed: {e}")

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

    def _parse_pubmed_xml(self, xml_text: str, pmid: str) -> Reference:
        """Parse PubMed XML response."""
        root = ET.fromstring(xml_text)
        article = root.find(".//PubmedArticle")

        if article is None:
            raise FatalError(f"No article found for PMID: {pmid}")

        # Extract title
        title_elem = article.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None and title_elem.text else ""

        # Extract authors
        authors = []
        author_list = article.find(".//AuthorList")
        if author_list is not None:
            for author_elem in author_list.findall("Author"):
                last_name = author_elem.find("LastName")
                fore_name = author_elem.find("ForeName")

                if last_name is not None and last_name.text:
                    if fore_name is not None and fore_name.text:
                        authors.append(f"{fore_name.text} {last_name.text}")
                    else:
                        authors.append(last_name.text)

        # Extract abstract
        abstract_elem = article.find(".//Abstract/AbstractText")
        abstract = abstract_elem.text if abstract_elem is not None and abstract_elem.text else ""

        # Extract publication date
        year = None
        pub_date = article.find(".//PubDate")
        if pub_date is not None:
            year_elem = pub_date.find("Year")
            if year_elem is not None and year_elem.text:
                try:
                    year = int(year_elem.text)
                except ValueError:
                    pass

        # Extract journal info
        journal = article.find(".//Journal/Title")
        venue = journal.text if journal is not None and journal.text else ""

        # Extract DOI
        doi = None
        article_id_list = article.find(".//ArticleIdList")
        if article_id_list is not None:
            for article_id in article_id_list.findall("ArticleId"):
                if article_id.get("IdType") == "doi" and article_id.text:
                    doi = article_id.text
                    break

        # Extract PMCID (PubMed Central ID for OA)
        pmcid = None
        oa_url = None
        if article_id_list is not None:
            for article_id in article_id_list.findall("ArticleId"):
                if article_id.get("IdType") == "pmc" and article_id.text:
                    pmcid = article_id.text
                    oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                    break

        # Determine reference type and venue type
        pub_type_list = article.find(".//PublicationTypeList")
        reference_type = "academic"
        venue_type = "journal"

        if pub_type_list is not None:
            for pub_type in pub_type_list.findall("PublicationType"):
                if pub_type.text:
                    type_text = pub_type.text.lower()
                    if "review" in type_text:
                        reference_type = "academic"
                    elif "case report" in type_text:
                        reference_type = "clinical"
                    elif "clinical trial" in type_text:
                        reference_type = "clinical"

        # Compute quality score
        quality_score = self._compute_quality_score(article, pmcid)

        return Reference(
            ref_id=pmid,
            raw_citation="",
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            venue=venue,
            venue_type=venue_type,
            doi=doi,
            reference_type=reference_type,
            resolved=True,
            status=ReferenceStatus.RESOLVED,
            quality_score=quality_score,
            metadata_provider=self.name,
            oa_available=pmcid is not None,
            oa_url=oa_url,
            resolved_at=datetime.now(timezone.utc),
        )

    def _compute_quality_score(self, article: ET.Element, pmcid: Optional[str]) -> float:
        """Compute quality score for PubMed entry."""
        score = 0.9  # Base score for peer-reviewed biomedical literature

        # Has DOI: +0.05
        doi_found = False
        article_id_list = article.find(".//ArticleIdList")
        if article_id_list is not None:
            for article_id in article_id_list.findall("ArticleId"):
                if article_id.get("IdType") == "doi" and article_id.text:
                    doi_found = True
                    break
        if doi_found:
            score += 0.05

        # Open access via PMC: +0.05
        if pmcid:
            score += 0.05

        # Has abstract: +0.0 (expected for PubMed)
        # Most articles have abstracts, so no bonus

        return min(1.0, score)
