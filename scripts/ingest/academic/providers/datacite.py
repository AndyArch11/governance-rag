"""
DataCite metadata provider for research data and scholarly objects.

DataCite provides DOIs for research datasets, software, and other scholarly
outputs beyond traditional publications.

API: https://api.datacite.org/
Rate limit: No strict limit, but be respectful
Documentation: https://support.datacite.org/docs/api
"""

import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import List, Optional

from .base import BaseProvider, FatalError, RecoverableError, Reference, ReferenceStatus

logger = logging.getLogger(__name__)


class DataCiteProvider(BaseProvider):
    """
    DataCite metadata provider for research data and scholarly objects.

    DataCite provides:
    - DOIs for datasets, software, and other research outputs
    - Rich metadata including contributors and funders
    - Resource type classification
    - Version information
    """

    BASE_URL = "https://api.datacite.org"

    def __init__(self):
        self.name = "datacite"
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
        Resolve citation via DataCite.

        Priority:
        1. DOI lookup
        2. Title + creator search

        Args:
            citation_text: Raw citation or title
            year: Publication year (optional)
            doi: DOI if available (required for DataCite)
            authors: Creator list (optional)
            logger: Optional logger instance (optional)

        Returns:
            Reference with resolved metadata
        """
        # Set up logger
        use_logger = logger or globals().get("logger")
        if use_logger is None:
            use_logger = logging.getLogger(__name__)

        # DataCite primarily works via DOI
        if doi:
            try:
                return self._resolve_by_doi(doi)
            except (RecoverableError, FatalError) as e:
                use_logger.debug(f"DOI lookup failed: {e}")

        # Fallback to title search
        try:
            return self._resolve_by_search(citation_text, year, authors)
        except FatalError:
            # Return unresolved reference
            return Reference(
                ref_id=f"datacite_unresolved_{hash(citation_text)}",
                raw_citation=citation_text,
                resolved=False,
                status=ReferenceStatus.UNRESOLVED,
                metadata_provider=self.name,
            )

    def _resolve_by_doi(self, doi: str) -> Reference:
        """Resolve by DOI."""
        url = f"{self.BASE_URL}/dois/{doi}"

        try:
            response = self._request_with_retry("GET", url)
            data = response.json()

            if "data" not in data:
                raise FatalError(f"No DataCite record for DOI: {doi}")

            return self._parse_datacite_record(data["data"])

        except Exception as e:
            if isinstance(e, (RecoverableError, FatalError)):
                raise
            raise RecoverableError(f"DataCite DOI lookup failed: {e}")

    def _resolve_by_search(
        self,
        title: str,
        year: Optional[int] = None,
        creators: Optional[List[str]] = None,
    ) -> Reference:
        """Resolve via DataCite search."""
        params = {
            "query": f'titles.title:"{title}"',
            "page[size]": 10,
        }

        if year:
            params["query"] += f" AND publicationYear:{year}"

        url = f"{self.BASE_URL}/dois"

        try:
            response = self._request_with_retry("GET", url, params=params)
            data = response.json()

            results = data.get("data", [])
            if not results:
                raise FatalError(f"No DataCite results for: {title}")

            # Find best match
            best_match = None
            best_score = 0.75

            for item in results:
                attributes = item.get("attributes", {})
                titles = attributes.get("titles", [])

                if titles:
                    item_title = titles[0].get("title", "")
                    score = self._fuzzy_match_title(title, item_title)

                    if score > best_score:
                        best_score = score
                        best_match = item

            if not best_match:
                raise FatalError(f"Low confidence match for: {title}")

            return self._parse_datacite_record(best_match)

        except Exception as e:
            if isinstance(e, (RecoverableError, FatalError)):
                raise
            raise RecoverableError(f"DataCite search failed: {e}")

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

    def _parse_datacite_record(self, record: dict) -> Reference:
        """Parse DataCite record."""
        attributes = record.get("attributes", {})

        # Extract DOI
        doi = attributes.get("doi", "")

        # Extract titles
        titles = attributes.get("titles", [])
        title = titles[0].get("title", "") if titles else ""

        # Extract creators
        creators_data = attributes.get("creators", [])
        authors = []
        for creator in creators_data:
            name = creator.get("name") or creator.get("givenName", "") + " " + creator.get(
                "familyName", ""
            )
            if name.strip():
                authors.append(name.strip())

        # Extract publication year
        pub_year = attributes.get("publicationYear")
        year = int(pub_year) if pub_year else None

        # Extract descriptions (abstract)
        descriptions = attributes.get("descriptions", [])
        abstract = ""
        for desc in descriptions:
            if desc.get("descriptionType") == "Abstract":
                abstract = desc.get("description", "")
                break

        # If no abstract, use first description
        if not abstract and descriptions:
            abstract = descriptions[0].get("description", "")

        # Extract publisher/venue
        publisher = attributes.get("publisher", "")

        # Extract resource type
        resource_types = attributes.get("types", {})
        resource_type_general = resource_types.get("resourceTypeGeneral", "Dataset")
        resource_type_specific = resource_types.get("resourceType", "")

        # Determine reference type and venue type
        reference_type = self._determine_reference_type(resource_type_general)
        venue_type = self._determine_venue_type(resource_type_general)

        # Construct venue
        venue = resource_type_specific or resource_type_general
        if publisher:
            venue = f"{venue} ({publisher})"

        # Extract URL
        url = attributes.get("url", "")

        # Check for open access
        rights_list = attributes.get("rightsList", [])
        oa_available = any(
            "open" in str(right.get("rights", "")).lower()
            or "cc" in str(right.get("rightsIdentifier", "")).lower()
            for right in rights_list
        )

        # Compute quality score
        quality_score = self._compute_quality_score(attributes, resource_type_general)

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
            oa_available=oa_available,
            oa_url=url if oa_available else None,
            resolved_at=datetime.now(timezone.utc),
        )

    def _determine_reference_type(self, resource_type: str) -> str:
        """Determine reference type from DataCite resource type."""
        type_mapping = {
            "Dataset": "dataset",
            "Software": "software",
            "Model": "software",
            "Workflow": "software",
            "Image": "dataset",
            "PhysicalObject": "dataset",
            "Collection": "dataset",
            "DataPaper": "academic",
            "Text": "academic",
            "Report": "technical",
            "ConferencePaper": "conference",
            "Journal Article": "academic",
            "Preprint": "preprint",
            "Book": "academic",
            "BookChapter": "academic",
        }

        return type_mapping.get(resource_type, "other")

    def _determine_venue_type(self, resource_type: str) -> str:
        """Determine venue type from DataCite resource type."""
        type_mapping = {
            "Dataset": "repository",
            "Software": "repository",
            "Model": "repository",
            "Workflow": "repository",
            "Image": "repository",
            "PhysicalObject": "repository",
            "Collection": "repository",
            "DataPaper": "journal",
            "Text": "journal",
            "Report": "technical",
            "ConferencePaper": "conference",
            "Journal Article": "journal",
            "Preprint": "preprint",
            "Book": "book",
            "BookChapter": "book",
        }

        return type_mapping.get(resource_type, "other")

    def _compute_quality_score(self, attributes: dict, resource_type: str) -> float:
        """Compute quality score for DataCite record."""
        # Base score varies by resource type
        base_scores = {
            "Dataset": 0.85,
            "Software": 0.85,
            "DataPaper": 0.9,
            "ConferencePaper": 0.9,
            "Journal Article": 0.95,
            "Text": 0.8,
        }

        score = base_scores.get(resource_type, 0.75)

        # Has DOI (always true for DataCite): +0.0

        # Has description/abstract: +0.05
        descriptions = attributes.get("descriptions", [])
        if descriptions:
            score += 0.05

        # Has multiple creators (collaborative): +0.05
        creators = attributes.get("creators", [])
        if len(creators) >= 3:
            score += 0.05

        # Has funding information: +0.05
        funding_refs = attributes.get("fundingReferences", [])
        if funding_refs:
            score += 0.05

        return min(1.0, score)
