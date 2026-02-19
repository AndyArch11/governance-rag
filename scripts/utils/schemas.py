"""Pydantic validation schemas for document ingestion.

Provides type-safe validation for:
- Document metadata (doc_type, topics, summary)
- Text chunks (content, IDs, hierarchical context)
- Enhanced chunk metadata (neighbours, entities, structure)
- Summary quality requirements

All schemas enforce minimum quality standards and data integrity.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MetadataSchema(BaseModel):
    """Validation schema for document metadata.

    Ensures extracted metadata meets quality requirements:
    - Document type is specified and non-empty
    - Topics are provided as list of strings
    - Summary is meaningful and non-empty

    Attributes:
        doc_type: Document category (e.g., 'governance policy', 'technical guide').
        key_topics: List of main topics covered in document.
        summary: Concise document summary.

    TODO:
        - Add enums for common doc_types
        - Add length limits for summary and topics
    """

    model_config = ConfigDict(
        extra="forbid",  # Disallow unexpected fields
        str_strip_whitespace=True,  # Trim whitespace from strings
    )

    doc_type: str = Field(..., min_length=1, description="Document category or type")
    key_topics: List[str] = Field(default_factory=list, description="List of key topics covered")
    summary: str = Field(..., min_length=1, description="Document summary")


class ChunkSchema(BaseModel):
    """Validation schema for document text chunks.

    Ensures chunks meet minimum quality standards:
    - Valid non-empty IDs
    - Minimum text length (20 characters)
    - No whitespace-only content

    Attributes:
        chunk_id: Unique chunk identifier (format: {doc_id}-chunk-{index}).
        text: Chunk text content.
        doc_id: Parent document identifier.

    TODO:
        - Add maximum length limit
        - Add character ratio validation (avoid gibberish)
        -- Character Error Rate (CER) and Word Error Rate (WER) checks
    """

    model_config = ConfigDict(
        extra="forbid",  # Disallow unexpected fields
        str_strip_whitespace=True,  # Trim whitespace from strings
    )

    chunk_id: str = Field(..., min_length=1, description="Unique chunk identifier")
    text: str = Field(..., min_length=20, description="Chunk text content (minimum 20 characters)")
    doc_id: str = Field(..., min_length=1, description="Parent document identifier")


class SummarySchema(BaseModel):
    """Validation schema for document summaries.

    Ensures summaries are meaningful and substantial:
    - Minimum 30 characters
    - At least 5 words
    - Not trivial placeholders like "N/A" or "Summary unavailable"

    Attributes:
        summary: Document summary text.
    """

    summary: str = Field(..., min_length=30, description="Document summary (minimum 30 characters)")

    @field_validator("summary")
    @classmethod
    def ensure_meaningful_summary(cls, v: str) -> str:
        """Ensure summary contains sufficient content.

        Args:
            v: Summary text to validate.

        Returns:
            Validated summary.

        Raises:
            ValueError: If summary has fewer than 5 words.
        """
        if len(v.split()) < 5:
            raise ValueError("Summary too short - must contain at least 5 words")
        return v


class EnhancedChunkMetadata(BaseModel):
    """Enhanced metadata for RAG optimisation.

    Provides rich context for improved retrieval and graph-based relationships.

    Attributes:
        chapter: Chapter label (e.g., "Chapter 1", "Introduction")
        section_title: Section heading text
        heading_path: Full hierarchical path (e.g., "Config > Database > Pool")
        parent_section: Immediate parent section title
        section_depth: Nesting level (1 = top-level, 2 = subsection, etc.)
        prev_chunk_id: ID of previous chunk in document sequence
        next_chunk_id: ID of next chunk in document sequence
        technical_entities: Extracted technical terms, components, APIs
        code_language: Programming language if chunk contains code
        contains_table: Whether chunk contains table data
        contains_code: Whether chunk contains code blocks
        contains_diagram: Whether chunk references diagrams/images
        is_api_reference: Whether chunk is API documentation
        is_configuration: Whether chunk describes configuration
        content_type: General classification (text, code, mixed, structured)
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    # Document structure (for academic documents)
    chapter: Optional[str] = Field(None, description="Chapter label (e.g., 'Chapter 1')")
    section_title: Optional[str] = Field(None, description="Section heading text")

    # Hierarchical context
    heading_path: Optional[str] = Field(
        None, description="Full section path, e.g. 'Config > Database > Connection Pool'"
    )
    parent_section: Optional[str] = Field(None, description="Immediate parent section")
    section_depth: Optional[int] = Field(None, ge=0, description="Section nesting level")

    # Sequential relationships
    prev_chunk_id: Optional[str] = Field(None, description="Previous chunk ID in sequence")
    next_chunk_id: Optional[str] = Field(None, description="Next chunk ID in sequence")
    sequence_number: Optional[int] = Field(
        None, ge=0, description="Sequential position in document (0-based)"
    )

    # Technical entities and markers
    technical_entities: List[str] = Field(
        default_factory=list, description="Technical terms, components, APIs mentioned"
    )
    code_language: Optional[str] = Field(None, description="Programming language if code present")

    # Content type flags
    contains_table: bool = Field(False, description="Contains table data")
    contains_code: bool = Field(False, description="Contains code blocks")
    contains_diagram: bool = Field(False, description="References diagrams or images")
    is_api_reference: bool = Field(False, description="Is API documentation")
    is_configuration: bool = Field(False, description="Describes configuration")

    # General classification
    content_type: str = Field(
        "text", description="Content classification: text, code, mixed, structured"
    )


class ParentChunkSchema(BaseModel):
    """Schema for parent chunks in parent-child chunking strategy.

    Parent chunks are larger sections that provide context for smaller
    searchable child chunks. Parent chunks are NOT embedded or searched
    directly, but retrieved when their children match a query.

    Attributes:
        parent_id: Unique identifier for the parent chunk
        text: Full text of the parent chunk (larger context)
        child_ids: List of child chunk IDs contained in this parent
        doc_id: Document identifier
        section_title: Section heading for this parent chunk
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    parent_id: str = Field(..., min_length=1, description="Unique parent chunk identifier")
    text: str = Field(..., min_length=100, description="Parent chunk text (full context)")
    child_ids: List[str] = Field(default_factory=list, description="Child chunk IDs")
    doc_id: str = Field(..., min_length=1, description="Document identifier")
    section_title: Optional[str] = Field(None, description="Section heading")
