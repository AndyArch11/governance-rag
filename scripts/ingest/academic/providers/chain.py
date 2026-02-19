"""
Provider chain for orchestrating metadata resolution across multiple providers.

Implements a smart provider chain that queries providers in priority order
until a confident match is found.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from .base import BaseProvider, FatalError, RecoverableError, Reference, ReferenceStatus

logger = logging.getLogger(__name__)


@dataclass
class ResolutionResult:
    """Result of a resolution attempt."""

    reference: Reference
    provider: str
    confidence: float  # 0.0 - 1.0
    attempt_count: int


class ProviderChain:
    """
    Orchestrate metadata resolution across multiple providers.

    Queries providers in priority order, balancing speed and comprehensiveness.
    Configurable confidence thresholds and retry strategies.
    """

    def __init__(self, providers: List[BaseProvider], min_confidence: float = 0.85):
        """
        Initialise provider chain.

        Args:
            providers: Ordered list of providers to query
            min_confidence: Minimum confidence threshold (0.0 - 1.0)
        """
        self.providers = providers
        self.min_confidence = min_confidence
        self.resolution_stats = {
            "total_queries": 0,
            "resolved": 0,
            "unresolved": 0,
            "by_provider": {p.name: {"success": 0, "failure": 0} for p in providers},
        }

    def resolve(
        self,
        citation_text: str,
        year: Optional[int] = None,
        doi: Optional[str] = None,
        authors: Optional[List[str]] = None,
        logger=None,
    ) -> ResolutionResult:
        """
        Resolve citation using provider chain.

        Tries each provider in order until confidence threshold is met
        or all providers are exhausted.

        Args:
            citation_text: Raw citation or title
            year: Publication year (optional)
            doi: DOI if available (optional)
            authors: Author list (optional)
            logger: Optional logger instance for status messages

        Returns:
            ResolutionResult with best match and confidence
        """
        # Use provided logger or fall back to module logger
        use_logger = logger or globals().get("logger")
        if use_logger is None:
            use_logger = logging.getLogger(__name__)

        self.resolution_stats["total_queries"] += 1

        best_result = None
        best_confidence = 0.0

        for attempt, provider in enumerate(self.providers, 1):
            try:
                use_logger.debug(f"Attempting resolution with {provider.name} (attempt {attempt})")

                # Call provider.resolve() with only parameters it accepts
                # All providers accept: citation_text, year, doi
                # Only some accept: authors (so we don't pass it here)
                reference = provider.resolve(
                    citation_text=citation_text,
                    year=year,
                    doi=doi,
                    logger=use_logger,
                )

                # Skip if provider couldn't resolve (returns None)
                if reference is None:
                    use_logger.debug(f"{provider.name} returned None (unresolved)")
                    self.resolution_stats["by_provider"][provider.name]["failure"] += 1
                    continue

                # Compute confidence for this reference
                confidence = self._compute_confidence(reference)

                use_logger.debug(
                    f"{provider.name} resolved: {reference.title} "
                    f"(confidence: {confidence:.2f})"
                )

                # Track success
                self.resolution_stats["by_provider"][provider.name]["success"] += 1

                # Update best match
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = ResolutionResult(
                        reference=reference,
                        provider=provider.name,
                        confidence=confidence,
                        attempt_count=attempt,
                    )

                # Return if confident enough
                if confidence >= self.min_confidence and reference.resolved:
                    use_logger.info(
                        f"Resolved {citation_text} with {provider.name} "
                        f"(confidence: {confidence:.2f})"
                    )
                    self.resolution_stats["resolved"] += 1
                    return best_result

            except RecoverableError as e:
                use_logger.debug(f"{provider.name} recoverable error: {e}", exc_info=True)
                self.resolution_stats["by_provider"][provider.name]["failure"] += 1
                continue

            except FatalError as e:
                use_logger.debug(f"{provider.name} fatal error: {e}", exc_info=True)
                self.resolution_stats["by_provider"][provider.name]["failure"] += 1
                continue

            except Exception as e:
                use_logger.warning(f"{provider.name} unexpected error: {e}", exc_info=True)
                self.resolution_stats["by_provider"][provider.name]["failure"] += 1
                continue

        # Return best result found, even if below threshold
        if best_result:
            use_logger.warning(
                f"Resolved {citation_text} with {best_result.provider} "
                f"(low confidence: {best_confidence:.2f})"
            )
            self.resolution_stats["resolved"] += 1
            return best_result

        # Completely unresolved
        use_logger.warning(f"Could not resolve: {citation_text}")
        self.resolution_stats["unresolved"] += 1

        return ResolutionResult(
            reference=Reference(
                ref_id=f"unresolved_{hash(citation_text)}",
                raw_citation=citation_text,
                resolved=False,
                status=ReferenceStatus.UNRESOLVED,
                metadata_provider="unresolved",
            ),
            provider="none",
            confidence=0.0,
            attempt_count=len(self.providers),
        )

    def resolve_batch(
        self,
        citations: List[str],
        batch_size: int = 10,
        progress_callback=None,
    ) -> List[ResolutionResult]:
        """
        Resolve a batch of citations.

        Args:
            citations: List of citation texts
            batch_size: Number of citations per log message
            progress_callback: Optional callback for progress updates

        Returns:
            List of ResolutionResult objects
        """
        results = []

        for i, citation in enumerate(citations):
            result = self.resolve(citation)
            results.append(result)

            if progress_callback and (i + 1) % batch_size == 0:
                progress_callback(i + 1, len(citations))

        return results

    def _compute_confidence(self, reference: Reference) -> float:
        """
        Compute confidence score for a resolved reference.

        Considers:
        - Resolution status
        - Quality score
        - Metadata completeness
        """
        if not reference.resolved:
            return 0.0

        # Base confidence from quality score
        confidence = reference.quality_score

        # Boost for metadata completeness
        completeness = self._compute_completeness(reference)
        confidence = confidence * 0.7 + completeness * 0.3

        return confidence

    def _compute_completeness(self, reference: Reference) -> float:
        """Compute metadata completeness score."""
        score = 0.0
        max_fields = 10
        fields_present = 0

        # Check key fields
        if reference.title:
            fields_present += 1
        if reference.authors:
            fields_present += 1
        if reference.year:
            fields_present += 1
        if reference.doi:
            fields_present += 1
        if reference.venue:
            fields_present += 1
        if reference.abstract:
            fields_present += 1
        if reference.pages:
            fields_present += 1
        if reference.volume:
            fields_present += 1
        if reference.issue:
            fields_present += 1
        if reference.oa_url:
            fields_present += 1

        return fields_present / max_fields

    def get_stats(self) -> dict:
        """Get resolution statistics."""
        total = self.resolution_stats["total_queries"]
        return {
            **self.resolution_stats,
            "resolution_rate": self.resolution_stats["resolved"] / total if total > 0 else 0.0,
            "average_attempts": (
                (self.resolution_stats["resolved"] + self.resolution_stats["unresolved"]) / total
                if total > 0
                else 0.0
            ),
        }

    def reset_stats(self):
        """Reset statistics."""
        self.resolution_stats = {
            "total_queries": 0,
            "resolved": 0,
            "unresolved": 0,
            "by_provider": {p.name: {"success": 0, "failure": 0} for p in self.providers},
        }


def create_default_chain(
    cache=None, api_key: Optional[str] = None, email: Optional[str] = None
) -> ProviderChain:
    """
    Create a default provider chain with all 10 available providers.

    Priority order (high to low value):
    1. Crossref (fastest, most reliable for DOI)
    2. DataCite (for datasets and research objects)
    3. PubMed (biomedical literature)
    4. OpenAlex (comprehensive, good for title search)
    5. arXiv (preprints in CS/physics/math)
    6. Semantic Scholar (AI-powered, best fuzzy matching)
    7. ORCID (author-verified publications)
    8. Google Scholar (broad coverage, web scraping)
    9. Unpaywall (OA discovery, requires DOI)
    10. URL Fetch (last resort, metadata from web pages)

    Args:
        cache: Optional ReferenceCache for caching results
        api_key: Optional API key for PubMed (increases rate limit)
        email: Optional email for Unpaywall polite pool

    Returns:
        Configured ProviderChain with all 10 providers
    """
    from .arxiv import ArxivProvider
    from .crossref import CrossrefProvider
    from .datacite import DataCiteProvider
    from .google_scholar import GoogleScholarProvider
    from .openalex import OpenAlexProvider
    from .orcid import ORCIDProvider
    from .pubmed import PubMedProvider
    from .semantic_scholar import SemanticScholarProvider
    from .unpaywall import UnpaywallProvider
    from .url_fetch import URLFetchProvider

    providers = [
        CrossrefProvider(),
        DataCiteProvider(),
        PubMedProvider(api_key=api_key),
        OpenAlexProvider(),
        ArxivProvider(),
        SemanticScholarProvider(),
        ORCIDProvider(),
        GoogleScholarProvider(),
        UnpaywallProvider(email=email),
        URLFetchProvider(),
    ]

    return ProviderChain(providers, min_confidence=0.85)
