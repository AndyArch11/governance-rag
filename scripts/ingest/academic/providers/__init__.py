"""
Academic metadata providers package.

Provides interfaces and implementations for resolving academic references
through various metadata APIs.
"""

from .arxiv import ArxivProvider
from .base import BaseProvider, FatalError, RecoverableError, Reference, ReferenceStatus
from .chain import ProviderChain, ResolutionResult, create_default_chain
from .crossref import CrossrefProvider
from .datacite import DataCiteProvider
from .google_scholar import GoogleScholarProvider
from .openalex import OpenAlexProvider
from .orcid import ORCIDProvider
from .pubmed import PubMedProvider
from .semantic_scholar import SemanticScholarProvider
from .unpaywall import UnpaywallProvider
from .url_fetch import URLFetchProvider

# Module-level provider chain (lazily initialised)
_default_chain = None


def resolve_reference(citation_text: str, year=None, doi=None, logger=None):
    """
    Resolve a citation to a Reference using the default provider chain.

    Args:
        citation_text: The citation text to resolve
        year: Optional year from parsed citation
        doi: Optional DOI from parsed citation
        logger: Optional logger instance for status messages

    Returns:
        tuple: (Reference object, confidence float) - Reference with resolved metadata
               and the confidence score from the resolution (0.0-1.0)
    """
    global _default_chain
    if _default_chain is None:
        _default_chain = create_default_chain()

    result = _default_chain.resolve(citation_text=citation_text, year=year, doi=doi, logger=logger)

    # Return both Reference and confidence from ResolutionResult
    if result:
        return result.reference, result.confidence
    else:
        return (
            Reference(
                ref_id="",
                raw_citation=citation_text,
                resolved=False,
                metadata_provider="unresolved",
            ),
            0.0,
        )


__all__ = [
    "BaseProvider",
    "Reference",
    "ReferenceStatus",
    "RecoverableError",
    "FatalError",
    "CrossrefProvider",
    "OpenAlexProvider",
    "SemanticScholarProvider",
    "ArxivProvider",
    "PubMedProvider",
    "DataCiteProvider",
    "ORCIDProvider",
    "GoogleScholarProvider",
    "UnpaywallProvider",
    "URLFetchProvider",
    "ProviderChain",
    "ResolutionResult",
    "create_default_chain",
    "resolve_reference",
]
