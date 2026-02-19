"""Academic ingestion module.

Provides document ingestion, citation parsing, reference resolution, and graph building
for academic documents with comprehensive data models and metadata providers.
"""

from .models import (  # Enums; Core Models; Factory Functions
    AcademicDocument,
    BatchIngestResult,
    CitationEdge,
    DocumentStatus,
    DomainTerm,
    EdgeType,
    LinkStatus,
    NormalisedCitation,
    OAStatus,
    ParsedCitation,
    PersonaContext,
    RawCitation,
    Reference,
    ReferenceStatus,
    RevalidationResult,
    create_document,
    create_reference,
)
from .providers import (
    BaseProvider,
    ProviderChain,
    ResolutionResult,
    create_default_chain,
)

__all__ = [
    # Enums
    "DocumentStatus",
    "ReferenceStatus",
    "EdgeType",
    "OAStatus",
    "LinkStatus",
    # Core Models
    "AcademicDocument",
    "Reference",
    "CitationEdge",
    "RawCitation",
    "ParsedCitation",
    "NormalisedCitation",
    "DomainTerm",
    "BatchIngestResult",
    "RevalidationResult",
    "PersonaContext",
    # Factory Functions
    "create_document",
    "create_reference",
    # Providers
    "BaseProvider",
    "ProviderChain",
    "ResolutionResult",
    "create_default_chain",
]
