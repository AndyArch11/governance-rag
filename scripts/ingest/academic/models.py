"""
Academic ingestion data models.

Comprehensive dataclasses for documents, references, citations, and domain terms.
Uses Pydantic v2 for validation and serialisation.

All models support:
- JSON serialisation/deserialisation
- SQLite database mapping
- ChromaDB metadata conversion
- Field validation and constraints
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ============================================================================
# Enums
# ============================================================================


class DocumentStatus(str, Enum):
    """Document processing status."""

    LOADED = "loaded"
    PARSED = "parsed"
    CITATIONS_EXTRACTED = "citations_extracted"
    REFERENCES_RESOLVED = "references_resolved"
    INGESTED = "ingested"
    FAILED = "failed"


class ReferenceStatus(str, Enum):
    """Reference resolution status."""

    RAW = "raw"  # Extracted, not yet resolved
    RESOLVING = "resolving"  # Currently querying providers
    RESOLVED = "resolved"  # Metadata found
    UNRESOLVED = "unresolved"  # No metadata found
    AMBIGUOUS = "ambiguous"  # Multiple matches
    ERROR = "error"  # Resolution error


class EdgeType(str, Enum):
    """Citation edge relationship type."""

    DIRECT = "direct"  # A directly cites B
    INDIRECT = "indirect"  # A cites C which cites B
    CONTRADICTS = "contradicts"  # A contradicts B
    EXTENDS = "extends"  # A extends methodology from B
    REFUTES = "refutes"  # A refutes claim in B


class OAStatus(str, Enum):
    """Open access status."""

    GOLD = "gold"  # Published open access
    GREEN = "green"  # Author/repository version
    HYBRID = "hybrid"  # Open access option in subscription journal
    BRONZE = "bronze"  # Free version from publisher
    CLOSED = "closed"  # No open access


class LinkStatus(str, Enum):
    """Link availability status."""

    AVAILABLE = "available"  # Link works
    STALE_404 = "stale_404"  # 404 not found
    STALE_TIMEOUT = "stale_timeout"  # Request timeout
    STALE_MOVED = "stale_moved"  # Moved/redirect


# ============================================================================
# Core Models
# ============================================================================


class AcademicDocument(BaseModel):
    """
    Primary academic document (thesis, paper, publication).

    Represents a document being ingested into the system with full-text content,
    metadata, and processing status tracking.
    """

    # Identifiers
    doc_id: str = Field(..., description="Generated: acad_{domain}_{year}_{hash[:8]}")
    source_url: Optional[str] = Field(None, description="Source URL if loaded from web")
    source_file: Optional[str] = Field(None, description="Local file path if available")

    # Content
    full_text: str = Field(default="", description="Complete extracted text")
    file_hash: str = Field(default="", description="MD5 hash of full text for change detection")

    # Metadata
    title: Optional[str] = Field(None, description="Document title")
    authors: List[str] = Field(default_factory=list, description="Author list")
    author_orcids: List[Optional[str]] = Field(
        default_factory=list, description="ORCID iDs parallel to authors"
    )
    year: Optional[int] = Field(None, description="Publication year", ge=1900, le=2100)
    abstract: Optional[str] = Field(None, description="Document abstract")
    keywords: List[str] = Field(default_factory=list, description="Associated keywords")

    # Domain classification
    primary_domain: str = Field(default="", description="Primary domain (e.g., 'machine_learning')")
    secondary_domains: List[str] = Field(default_factory=list, description="Secondary domains")
    topic: str = Field(default="", description="Narrow focus topic")

    # Document structure
    sections: Dict[str, str] = Field(
        default_factory=dict, description="section_name → text mapping"
    )
    bibliography_raw: str = Field(default="", description="Raw bibliography text before parsing")

    # Processing status
    status: DocumentStatus = Field(default=DocumentStatus.LOADED, description="Processing status")
    ingest_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Statistics
    word_count: int = Field(default=0, ge=0, description="Total word count")
    citation_count: int = Field(
        default=0, ge=0, description="Count of citations extracted from document"
    )
    reference_count: int = Field(default=0, ge=0, description="Count of resolved references")

    # Relationships
    reference_ids: List[str] = Field(
        default_factory=list, description="References cited by this doc"
    )
    domain_term_ids: List[str] = Field(default_factory=list, description="Domain terms extracted")

    # Audit
    version: int = Field(default=1, ge=1, description="Record version")
    processing_duration_sec: float = Field(
        default=0.0, ge=0, description="Processing time in seconds"
    )

    model_config = ConfigDict(
        ser_json_timedelta="float",
    )

    @field_validator("year")
    @classmethod
    def validate_year(cls, v: Optional[int]) -> Optional[int]:
        """Validate publication year is reasonable."""
        if v is not None and (v < 1900 or v > datetime.now().year + 1):
            raise ValueError(f"Year must be between 1900 and {datetime.now().year + 1}")
        return v

    @field_validator("reference_count")
    @classmethod
    def validate_reference_count(cls, v: int, info) -> int:
        """Reference count should not exceed citation count."""
        if info.data.get("citation_count") and v > info.data["citation_count"]:
            raise ValueError("Reference count cannot exceed citation count")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dict for database storage."""
        data = self.model_dump()
        # Convert enums to strings
        if "status" in data:
            data["status"] = (
                data["status"].value if isinstance(data["status"], Enum) else data["status"]
            )
        # Convert datetimes to ISO format
        for key in ["ingest_timestamp", "last_updated"]:
            if key in data and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        return data

    def to_chromadb_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB metadata dict."""
        return {
            "doc_id": self.doc_id,
            "doc_type": "academic_primary",
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "primary_domain": self.primary_domain,
            "secondary_domains": self.secondary_domains,
            "topic": self.topic,
            "status": self.status.value,
            "citation_count": self.citation_count,
            "reference_count": self.reference_count,
            "ingest_timestamp": self.ingest_timestamp.isoformat(),
        }

    def to_json(self) -> str:
        """Serialise to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AcademicDocument:
        """Create instance from dict."""
        return cls(**data)


class Reference(BaseModel):
    """
    Academic reference cited by one or more documents.

    Represents a single citation with full resolution metadata, open access status,
    and link tracking.
    """

    # Identifiers
    ref_id: str = Field(..., description="Generated: ref_{cache_key[:12]}")
    doi: Optional[str] = Field(None, description="DOI identifier")
    arxiv_id: Optional[str] = Field(None, description="arXiv identifier")

    # Metadata (from resolution)
    title: Optional[str] = Field(None, description="Reference title")
    authors: List[str] = Field(default_factory=list, description="Author list")
    author_orcids: List[Optional[str]] = Field(
        default_factory=list, description="ORCID iDs parallel to authors"
    )
    year: Optional[int] = Field(None, description="Publication year", ge=1900, le=2100)
    abstract: Optional[str] = Field(None, description="Reference abstract")
    venue: Optional[str] = Field(None, description="Journal/conference/publisher name")
    volume: Optional[str] = Field(None, description="Volume number")
    issue: Optional[str] = Field(None, description="Issue number")
    pages: Optional[str] = Field(None, description="Page range")

    # Venue metadata (for visualisation)
    venue_name: Optional[str] = Field(None, description="Structured venue name")
    venue_type: Optional[str] = Field(
        None, description="Venue type: journal, conference, preprint, web"
    )
    venue_rank: Optional[str] = Field(None, description="Venue rank: Q1-Q4 or A*, A, B, C")
    impact_factor: Optional[float] = Field(None, description="Journal impact factor or h5-index")

    # Citation parsing artifacts
    raw_citation: str = Field(default="", description="Original citation text")
    citation_format: str = Field(
        default="", description="Detected format: harvard, ieee, apa, etc."
    )

    # Resolution
    resolved: bool = Field(default=False, description="Is resolved flag")
    status: ReferenceStatus = Field(default=ReferenceStatus.RAW, description="Resolution status")
    metadata_provider: Optional[str] = Field(
        None, description="Provider that resolved this (e.g., 'crossref')"
    )
    resolved_at: Optional[datetime] = Field(None, description="When resolved")

    # OA Status
    oa_available: bool = Field(default=False, description="Is open access available")
    oa_url: Optional[str] = Field(None, description="Open access URL")
    oa_status: Optional[OAStatus] = Field(None, description="OA classification")

    # Download status
    pdf_downloaded: bool = Field(default=False, description="PDF downloaded flag")
    pdf_local_path: Optional[str] = Field(None, description="Local PDF path if downloaded")
    pdf_file_hash: Optional[str] = Field(None, description="SHA256 hash of PDF")
    pdf_reused: bool = Field(default=False, description="PDF reused from cache")
    download_failed: bool = Field(default=False, description="PDF download failed")
    accessed_at: Optional[datetime] = Field(None, description="When online content was fetched")

    # Citing documents
    doc_ids: List[str] = Field(default_factory=list, description="Documents citing this reference")

    # Domain classification
    primary_domain: Optional[str] = Field(
        None, description="Primary domain inferred from citing docs"
    )
    relevance_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Relevance to citing doc domain"
    )

    # Reference classification
    reference_type: str = Field(
        default="academic", description="Type: academic, preprint, news, blog, online, report"
    )
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Quality rating 0.0-1.0")
    paywall_detected: bool = Field(default=False, description="Behind paywall flag")

    # Link availability tracking
    link_status: LinkStatus = Field(
        default=LinkStatus.AVAILABLE, description="Link availability status"
    )
    link_checked_at: Optional[datetime] = Field(None, description="Last link validation check")
    check_frequency_days: int = Field(
        default=30, ge=1, description="How often to re-check link health"
    )
    consecutive_failures: int = Field(
        default=0, ge=0, description="Consecutive failed health checks"
    )
    last_success_at: Optional[datetime] = Field(None, description="Last successful link access")

    # Alternative/archived URLs
    alternative_urls: List[str] = Field(
        default_factory=list, description="Backup URLs (DOI resolver, archives)"
    )
    archived_url: Optional[str] = Field(None, description="Wayback Machine or archive URL")
    archive_timestamp: Optional[datetime] = Field(None, description="When content was archived")

    # Content change detection (for online references)
    content_hash: Optional[str] = Field(
        None, description="SHA256 of full text for change detection"
    )
    last_content_check: Optional[datetime] = Field(
        None, description="When content was last verified"
    )
    content_changed: bool = Field(default=False, description="Content differs from initial version")

    # Statistics
    citation_count: int = Field(
        default=0, ge=0, description="How many times this reference is cited"
    )

    # Relationships
    nested_references: List[str] = Field(
        default_factory=list, description="References to other references"
    )

    # Audit
    version: int = Field(default=1, ge=1, description="Record version")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(
        ser_json_timedelta="float",
    )

    @field_validator("year")
    @classmethod
    def validate_year(cls, v: Optional[int]) -> Optional[int]:
        """Validate publication year."""
        if v is not None and (v < 1900 or v > datetime.now().year + 1):
            raise ValueError(f"Year must be between 1900 and {datetime.now().year + 1}")
        return v

    @model_validator(mode="after")
    def validate_resolved_status(self) -> Reference:
        """Validate resolved status is consistent with metadata."""
        if self.resolved and not self.resolved_at:
            self.resolved_at = datetime.now(timezone.utc)
        if self.resolved and self.status == ReferenceStatus.RAW:
            self.status = ReferenceStatus.RESOLVED
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dict for database storage."""
        data = self.model_dump()
        # Convert enums to strings manually
        if isinstance(data.get("status"), DocumentStatus):
            data["status"] = data["status"].value
        if isinstance(data.get("status"), ReferenceStatus):
            data["status"] = data["status"].value
        if isinstance(data.get("oa_status"), OAStatus):
            data["oa_status"] = data["oa_status"].value if data["oa_status"] else None
        if isinstance(data.get("link_status"), LinkStatus):
            data["link_status"] = data["link_status"].value
        # Convert datetimes
        for key in ["resolved_at", "accessed_at", "link_checked_at", "created_at", "last_updated"]:
            if key in data and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        return data

    def to_chromadb_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB metadata dict."""
        return {
            "ref_id": self.ref_id,
            "doc_type": "academic_reference",
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "venue": self.venue,
            "resolved": self.resolved,
            "metadata_provider": self.metadata_provider,
            "oa_available": self.oa_available,
            "oa_status": self.oa_status.value if self.oa_status else None,
            "cited_by_docs": self.doc_ids,
            "primary_domain": self.primary_domain,
            "relevance_score": self.relevance_score,
            "citation_count": self.citation_count,
            "reference_type": self.reference_type,
            "quality_score": self.quality_score,
            "paywall_detected": self.paywall_detected,
            "link_status": self.link_status.value,
        }

    def to_json(self) -> str:
        """Serialise to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Reference:
        """Create instance from dict."""
        return cls(**data)


class CitationEdge(BaseModel):
    """
    Citation relationship between documents/references.

    Represents directed edges in the citation graph with relationship metadata.
    """

    # Identifiers
    from_id: str = Field(..., description="Source doc_id or ref_id")
    to_id: str = Field(..., description="Target doc_id or ref_id")

    # Type
    edge_type: EdgeType = Field(default=EdgeType.DIRECT, description="Relationship type")

    # Metadata
    relationship_type: str = Field(
        default="cites", description="Relationship: cites, extends, contradicts, etc."
    )
    depth: int = Field(default=1, ge=1, description="Citation depth (1=direct, 2+=indirect)")

    # Context
    mention_text: Optional[str] = Field(None, description="Sentence mentioning the citation")
    mention_position: int = Field(default=0, ge=0, description="Character position in text")

    # Classification
    is_inline: bool = Field(default=False, description="Inline vs. bibliography citation")
    cited_as: Optional[str] = Field(
        None, description="How cited: support, critique, alternative, etc."
    )

    # Strength
    strength_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Importance of citation 0.0-1.0"
    )

    # Audit
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dict."""
        edge_type_val = (
            self.edge_type.value if isinstance(self.edge_type, EdgeType) else self.edge_type
        )
        return {
            "from_id": self.from_id,
            "to_id": self.to_id,
            "edge_type": edge_type_val,
            "relationship_type": self.relationship_type,
            "depth": self.depth,
            "mention_text": self.mention_text,
            "mention_position": self.mention_position,
            "is_inline": self.is_inline,
            "cited_as": self.cited_as,
            "strength_score": self.strength_score,
            "created_at": self.created_at.isoformat(),
        }

    def to_json(self) -> str:
        """Serialise to JSON string."""
        return self.model_dump_json()


class RawCitation(BaseModel):
    """
    Raw citation text before parsing/resolution.

    Represents extracted but not yet processed citation text.
    """

    raw_text: str = Field(..., description="Original text from bibliography")
    doc_id: str = Field(..., description="Source document ID")
    position: str = Field(default="bibliography", description="Position: bibliography or inline")
    position_offset: int = Field(default=0, ge=0, description="Character offset in document")
    parsed: Optional[ParsedCitation] = Field(None, description="Parsed citation (if available)")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict()


class ParsedCitation(BaseModel):
    """
    Structured citation fields extracted via parsing.

    Represents fields extracted from raw citation text by parsing.
    """

    title: Optional[str] = Field(None, description="Parsed title")
    authors: List[str] = Field(default_factory=list, description="Parsed author list")
    author_orcids: List[Optional[str]] = Field(
        default_factory=list, description="ORCID iDs if available"
    )
    year: Optional[int] = Field(None, description="Parsed year", ge=1900, le=2100)
    venue: Optional[str] = Field(None, description="Journal/conference name")
    doi: Optional[str] = Field(None, description="DOI if found")
    arxiv_id: Optional[str] = Field(None, description="arXiv ID if found")
    url: Optional[str] = Field(None, description="URL if found")
    volume: Optional[str] = Field(None, description="Volume number")
    issue: Optional[str] = Field(None, description="Issue number")
    pages: Optional[str] = Field(None, description="Page range")

    # Parsing metadata
    format: str = Field(default="unknown", description="Format: harvard, ieee, apa, etc.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Parsing confidence 0.0-1.0")

    model_config = ConfigDict()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dict."""
        return self.model_dump()


class NormalisedCitation(BaseModel):
    """
    Citation after normalisation for matching.

    Normalised fields used for deduplication and citation matching.
    """

    raw_text: str = Field(..., description="Original raw citation text")

    # Normalised fields
    authors_normalised: List[str] = Field(
        default_factory=list, description="Normalised authors (lowercase, last-name-first)"
    )
    title_normalised: str = Field(
        default="", description="Normalised title (lowercase, no punctuation)"
    )
    year: Optional[int] = Field(None, description="Publication year", ge=1900, le=2100)

    # Matching key
    match_key: str = Field(..., description="Used for in-doc deduplication")

    # Identifiers if found
    doi: Optional[str] = Field(None, description="DOI if available (highest priority match)")
    arxiv_id: Optional[str] = Field(None, description="arXiv ID if available")

    model_config = ConfigDict()


class DomainTerm(BaseModel):
    """
    Domain-specific terminology extracted from documents.

    Represents vocabulary terms specific to a domain with frequency and relationship data.
    """

    term_id: str = Field(..., description="Generated: term_{hash}")
    term: str = Field(..., description="The vocabulary word/phrase")
    domain: str = Field(..., description="Associated domain")

    # Frequency analysis
    frequency: int = Field(default=0, ge=0, description="Total occurrences across all docs")
    doc_ids: List[str] = Field(default_factory=list, description="Documents containing this term")

    # Scoring
    domain_relevance_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Relevance to domain"
    )
    tf_idf_score: float = Field(default=0.0, ge=0.0, description="TF-IDF score")

    # Relationships
    related_terms: List[str] = Field(default_factory=list, description="Synonym/related terms")
    broader_terms: List[str] = Field(default_factory=list, description="More general terms")
    narrower_terms: List[str] = Field(default_factory=list, description="More specific terms")

    # Classification
    term_type: str = Field(
        default="concept", description="Type: concept, method, tool, dataset, etc."
    )

    # Audit
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dict."""
        return {
            "term_id": self.term_id,
            "term": self.term,
            "domain": self.domain,
            "frequency": self.frequency,
            "doc_ids": self.doc_ids,
            "domain_relevance_score": self.domain_relevance_score,
            "tf_idf_score": self.tf_idf_score,
            "related_terms": self.related_terms,
            "broader_terms": self.broader_terms,
            "narrower_terms": self.narrower_terms,
            "term_type": self.term_type,
            "created_at": self.created_at.isoformat(),
        }


class BatchIngestResult(BaseModel):
    """
    Summary of batch ingestion results.

    Aggregates results from ingesting a batch of academic documents.
    """

    # Documents
    primary_docs: List[AcademicDocument] = Field(..., description="Ingested documents")

    # References (mapped by source doc_id)
    references: Dict[str, List[Reference]] = Field(
        default_factory=dict, description="References by doc_id"
    )

    # Domain analysis
    domain_terms: List[DomainTerm] = Field(
        default_factory=list, description="Domain terms extracted"
    )

    # Statistics
    total_citations: int = Field(default=0, ge=0, description="Total citations found")
    resolved: int = Field(default=0, ge=0, description="Resolved citations")
    unresolved: int = Field(default=0, ge=0, description="Unresolved citations")
    downloaded: int = Field(default=0, ge=0, description="PDFs downloaded")
    pdf_reused: int = Field(default=0, ge=0, description="PDFs reused from cache")

    # Timing
    duration_sec: float = Field(default=0.0, ge=0.0, description="Ingestion duration in seconds")

    # Audit
    errors: List[str] = Field(default_factory=list, description="Error messages")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_resolved_count(self) -> BatchIngestResult:
        """Validate resolved count doesn't exceed total."""
        if self.resolved > self.total_citations:
            raise ValueError("Resolved count cannot exceed total citations")
        if self.unresolved > self.total_citations:
            raise ValueError("Unresolved count cannot exceed total citations")
        return self

    def summary(self) -> str:
        """Human-readable summary."""
        resolution_rate = (
            (self.resolved / self.total_citations * 100) if self.total_citations > 0 else 0
        )
        return f"""
Batch Ingest Summary:
  Documents: {len(self.primary_docs)}
  Total Citations: {self.total_citations}
  Resolved: {self.resolved} ({resolution_rate:.1f}%)
  Unresolved: {self.unresolved}
  PDFs Downloaded: {self.downloaded}
  PDFs Reused: {self.pdf_reused}
  Duration: {self.duration_sec:.2f}s
  Errors: {len(self.errors)}
  Warnings: {len(self.warnings)}
"""


class RevalidationResult(BaseModel):
    """
    Summary of reference re-validation operation.

    Results from re-checking and updating existing reference metadata.
    """

    total: int = Field(..., ge=0, description="Total references processed")
    updated: int = Field(default=0, ge=0, description="References with changed metadata")
    unchanged: int = Field(default=0, ge=0, description="References with no changes")
    failed: int = Field(default=0, ge=0, description="Re-validation failures")

    # Detailed results
    details: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict, description="Detailed results by type"
    )

    # Timing
    duration_sec: float = Field(default=0.0, ge=0.0, description="Revalidation duration in seconds")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_counts(self) -> RevalidationResult:
        """Validate counts sum correctly."""
        sum_counts = self.updated + self.unchanged + self.failed
        if sum_counts != self.total:
            raise ValueError(
                f"Updated ({self.updated}) + Unchanged ({self.unchanged}) + Failed ({self.failed}) must equal Total ({self.total})"
            )
        return self

    def summary(self) -> str:
        """Human-readable summary."""
        return f"""
Revalidation Complete:
  Total: {self.total}
  Updated: {self.updated}
  Unchanged: {self.unchanged}
  Failed: {self.failed}
  Duration: {self.duration_sec:.2f}s
"""


class PersonaContext(BaseModel):
    """
    Context for persona-aware retrieval and filtering.

    Defines how references should be returned for different personas
    (supervisor, assessor, researcher) with customisable filters.
    """

    persona: str = Field(..., description="Persona: supervisor, assessor, researcher")
    reference_depth: int = Field(default=1, ge=1, description="Depth: 1=direct, 2+=indirect")
    include_citations: bool = Field(default=True, description="Include citation context")
    include_methodology: bool = Field(default=True, description="Include method details")
    include_datasets: bool = Field(default=True, description="Include dataset references")

    # Stale link handling
    include_stale_links: bool = Field(
        default=True, description="Include refs with unavailable links"
    )
    require_verifiable: bool = Field(default=False, description="Only return available links")
    stale_link_penalty: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Ranking penalty for stale links"
    )

    model_config = ConfigDict()

    def applies_to_ref(self, ref: Reference) -> bool:
        """
        Check if reference should be included for this persona.

        Args:
            ref: Reference to check

        Returns:
            True if reference matches persona criteria
        """
        # Filter out stale links if verifiability required
        if self.require_verifiable and ref.link_status != LinkStatus.AVAILABLE:
            return False

        # Filter out stale links if disabled
        if not self.include_stale_links and ref.link_status != LinkStatus.AVAILABLE:
            return False

        # Persona-specific filters
        if self.persona == "supervisor":
            return ref.citation_count > 10  # Foundational
        elif self.persona == "assessor":
            return ref.resolved and (ref.oa_available or ref.link_status == LinkStatus.AVAILABLE)
        elif self.persona == "researcher":
            return ref.year is None or ref.year >= 2020  # Recent

        return True

    def compute_ranking_penalty(self, ref: Reference) -> float:
        """
        Compute ranking penalty based on link status.

        Args:
            ref: Reference to evaluate

        Returns:
            Penalty multiplier (0.0 = no penalty, 1.0 = maximum penalty)
        """
        if ref.link_status == LinkStatus.AVAILABLE:
            return 0.0
        elif ref.link_status == LinkStatus.STALE_404:
            return self.stale_link_penalty * 1.0  # Full penalty
        elif ref.link_status == LinkStatus.STALE_TIMEOUT:
            return self.stale_link_penalty * 0.75  # 75% penalty
        elif ref.link_status == LinkStatus.STALE_MOVED:
            return self.stale_link_penalty * 0.25  # 25% penalty
        return 0.0


# ============================================================================
# Factory Functions
# ============================================================================


def create_document(
    doc_id: str,
    full_text: str,
    title: Optional[str] = None,
    authors: Optional[List[str]] = None,
    year: Optional[int] = None,
    **kwargs,
) -> AcademicDocument:
    """
    Factory function to create AcademicDocument with defaults.

    Args:
        doc_id: Document ID
        full_text: Complete text content
        title: Document title
        authors: Author list
        year: Publication year
        **kwargs: Additional fields

    Returns:
        AcademicDocument instance
    """
    return AcademicDocument(
        doc_id=doc_id, full_text=full_text, title=title, authors=authors or [], year=year, **kwargs
    )


def create_reference(
    ref_id: str,
    raw_citation: str,
    title: Optional[str] = None,
    authors: Optional[List[str]] = None,
    year: Optional[int] = None,
    doi: Optional[str] = None,
    **kwargs,
) -> Reference:
    """
    Factory function to create Reference with defaults.

    Args:
        ref_id: Reference ID
        raw_citation: Raw citation text
        title: Reference title
        authors: Author list
        year: Publication year
        doi: DOI identifier
        **kwargs: Additional fields

    Returns:
        Reference instance
    """
    return Reference(
        ref_id=ref_id,
        raw_citation=raw_citation,
        title=title,
        authors=authors or [],
        year=year,
        doi=doi,
        **kwargs,
    )


# ============================================================================
# Export List
# ============================================================================

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
]
