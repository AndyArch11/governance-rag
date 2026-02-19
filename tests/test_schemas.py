"""Tests for document ingestion schemas.

Tests Pydantic validation schemas for:
- MetadataSchema: Document metadata validation
- ChunkSchema: Text chunk validation
- SummarySchema: Summary quality validation
"""

import pytest
from pydantic import ValidationError

from scripts.utils.schemas import (
    ChunkSchema,
    MetadataSchema,
    SummarySchema,
)


class TestMetadataSchema:
    """Tests for MetadataSchema validation."""

    def test_valid_metadata_with_all_fields(self):
        """Test valid metadata with all fields provided."""
        metadata = MetadataSchema(
            doc_type="governance policy",
            key_topics=["Azure", "Security", "Compliance"],
            summary="This document outlines Azure security governance policies.",
        )
        assert metadata.doc_type == "governance policy"
        assert metadata.key_topics == ["Azure", "Security", "Compliance"]
        assert metadata.summary == "This document outlines Azure security governance policies."

    def test_valid_metadata_with_empty_topics(self):
        """Test valid metadata with empty topics list (default)."""
        metadata = MetadataSchema(
            doc_type="technical guide", summary="Technical documentation for Azure VMs."
        )
        assert metadata.doc_type == "technical guide"
        assert metadata.key_topics == []
        assert metadata.summary == "Technical documentation for Azure VMs."

    def test_invalid_empty_doc_type(self):
        """Test that empty doc_type is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MetadataSchema(doc_type="", summary="Summary text")
        assert "at least 1 character" in str(exc_info.value).lower()

    def test_invalid_empty_summary(self):
        """Test that empty summary is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MetadataSchema(doc_type="policy", summary="")
        assert "at least 1 character" in str(exc_info.value).lower()

    def test_invalid_missing_doc_type(self):
        """Test that missing doc_type raises validation error."""
        with pytest.raises(ValidationError):
            MetadataSchema(summary="Summary text")

    def test_invalid_missing_summary(self):
        """Test that missing summary raises validation error."""
        with pytest.raises(ValidationError):
            MetadataSchema(doc_type="policy")

    def test_metadata_with_single_topic(self):
        """Test metadata with a single topic."""
        metadata = MetadataSchema(
            doc_type="reference", key_topics=["Azure"], summary="Azure reference documentation."
        )
        assert len(metadata.key_topics) == 1
        assert metadata.key_topics[0] == "Azure"

    def test_metadata_doc_type_whitespace_only(self):
        """Test that whitespace-only doc_type fails validation."""
        with pytest.raises(ValidationError):
            MetadataSchema(doc_type="   ", summary="Summary")

    def test_metadata_doc_type_with_different_whitespace(self):
        """Test doc_type with tabs and newlines."""
        metadata = MetadataSchema(
            doc_type=" \r\n\tAzure Policy \n", summary="Document about Azure policies."
        )
        assert metadata.doc_type == "Azure Policy"

    def test_metadata_summary_whitespace_only(self):
        """Test that whitespace-only summary fails validation."""
        with pytest.raises(ValidationError):
            MetadataSchema(doc_type="policy", summary="     ")

    def test_metadata_with_special_characters(self):
        """Test metadata fields with special characters."""
        metadata = MetadataSchema(
            doc_type="technical-guide@2024!",
            key_topics=["Cloud#Computing", "DevOps&CI/CD"],
            summary="This summary includes special characters: @#$%^&*()!",
        )
        assert "@" in metadata.doc_type
        assert "#" in metadata.key_topics[0]
        assert "&" in metadata.key_topics[1]
        assert "!" in metadata.summary

    def test_metadata_with_extra_fields(self):
        """Test that extra unexpected fields raise validation error."""
        with pytest.raises(ValidationError):
            MetadataSchema(
                doc_type="policy", summary="Valid summary", unexpected_field="should fail"
            )


class TestChunkSchema:
    """Tests for ChunkSchema validation."""

    def test_valid_chunk(self):
        """Test valid chunk with all required fields."""
        chunk = ChunkSchema(
            chunk_id="doc123-chunk-1",
            text="This is a valid chunk of text with sufficient length for testing.",
            doc_id="doc 123",
        )
        assert chunk.chunk_id == "doc123-chunk-1"
        assert chunk.text == "This is a valid chunk of text with sufficient length for testing."
        assert chunk.doc_id == "doc 123"

    def test_chunk_at_minimum_text_length(self):
        """Test chunk with exactly 20 characters (minimum)."""
        chunk = ChunkSchema(
            chunk_id="doc1-chunk-1",
            text="123456789 1234567890",  # Exactly 20 characters
            doc_id="doc1",
        )
        assert len(chunk.text) == 20

    def test_invalid_chunk_text_too_short(self):
        """Test that text shorter than 20 characters is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkSchema(
                chunk_id="doc1-chunk-1", text="short text", doc_id=".doc"  # Only 10 characters
            )
        assert "at least 20 characters" in str(exc_info.value).lower()

    def test_invalid_chunk_text_too_short_after_spaces(self):
        """Test that text shorter than 20 characters is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkSchema(
                chunk_id="doc1-chunk-1",
                text="  3456789012345678  ",  # Only 16 characters after trimming spaces
                doc_id=".doc",
            )
        assert "at least 20 characters" in str(exc_info.value).lower()

    def test_invalid_chunk_empty_text(self):
        """Test that empty text is rejected."""
        with pytest.raises(ValidationError):
            ChunkSchema(chunk_id="doc1-chunk-1", text="", doc_id="doc1")

    def test_invalid_chunk_whitespace_only_text(self):
        """Test that whitespace-only text is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkSchema(
                chunk_id="doc1-chunk-1", text="                    ", doc_id="doc1"  # 20 spaces
            )
        assert "at least 20 characters" in str(exc_info.value).lower()

    def test_invalid_chunk_empty_id(self):
        """Test that empty chunk_id is rejected."""
        with pytest.raises(ValidationError):
            ChunkSchema(chunk_id="", text="This is valid text content for a chunk.", doc_id="doc1")

    def test_invalid_chunk_whitespace_only_id(self):
        """Test that whitespace-only chunk_id is rejected."""
        with pytest.raises(ValidationError):
            ChunkSchema(
                chunk_id="     ", text="This is valid text content for a chunk.", doc_id="doc1"
            )

    def test_invalid_chunk_empty_doc_id(self):
        """Test that empty doc_id is rejected."""
        with pytest.raises(ValidationError):
            ChunkSchema(
                chunk_id="doc1-chunk-1", text="This is valid text content for a chunk.", doc_id=""
            )

    def test_invalid_doc_id_whitespace_only(self):
        """Test that whitespace-only doc_id is rejected."""
        with pytest.raises(ValidationError):
            ChunkSchema(
                chunk_id="doc1-chunk-1",
                text="This is valid text content for a chunk.",
                doc_id="     ",
            )

    def test_invalid_chunk_missing_required_fields(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            ChunkSchema(
                chunk_id="doc1-chunk-1",
                text="This is valid text content for a chunk.",
                # Missing doc_id
            )

    def test_chunk_with_long_text(self):
        """Test chunk with very long text."""
        long_text = "word " * 1000  # Very long text
        chunk = ChunkSchema(chunk_id="doc1-chunk-1", text=long_text, doc_id="doc1")
        assert len(chunk.text) > 20

    def test_chunk_with_special_characters(self):
        """Test chunk with special characters and formatting."""
        chunk = ChunkSchema(
            chunk_id="doc1-chunk-1",
            text="Text with special chars: @#$%^&*() and emojis 🚀 and newlines\nand\ttabs",
            doc_id="doc1",
        )
        assert "@#$%^&*()" in chunk.text
        assert "🚀" in chunk.text

    def test_chunk_with_newlines_and_tabs(self):
        """Test that newlines and tabs are preserved as valid content."""
        chunk = ChunkSchema(
            chunk_id="doc1-chunk-1",
            text="Line 1\n\tTabbed line 2\n\t\tDouble tab line 3\r\nCarriage return and line feed (CRLF) and \\ backslash and / forward slash.",
            doc_id="doc1",
        )
        assert "\n" in chunk.text
        assert "\t" in chunk.text
        assert "\r\n" in chunk.text
        assert "\\" in chunk.text
        assert "/" in chunk.text


class TestSummarySchema:
    """Tests for SummarySchema validation."""

    def test_valid_summary(self):
        """Test valid summary with sufficient length and words."""
        summary = SummarySchema(
            summary="This is a comprehensive summary that explains the main concepts and provides valuable insights."
        )
        assert len(summary.summary) >= 30
        assert len(summary.summary.split()) >= 5

    def test_summary_at_minimum_length(self):
        """Test summary with exactly 30 characters (minimum) and 5 words."""
        summary = SummarySchema(
            summary="one two three four recognition"  # Exactly 30 characters, 5 words
        )
        assert len(summary.summary) == 30
        assert len(summary.summary.split()) == 5

    def test_summary_with_exactly_five_words(self):
        """Test summary with exactly 5 words (minimum word count)."""
        summary = SummarySchema(summary="Very extensive characteristic recognition algorithms.")
        assert len(summary.summary.split()) == 5

    def test_invalid_summary_too_short_characters(self):
        """Test that summary shorter than 30 characters is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SummarySchema(summary="A very short summary text too")  # 29 characters, > 5 words
        assert "at least 30 characters" in str(exc_info.value).lower()

    def test_invalid_summary_too_few_words(self):
        """Test that summary with fewer than 5 words is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SummarySchema(
                summary="Very extensive characteristic recognition"
            )  # Only 4 words, 41 characters
        assert "at least 5 words" in str(exc_info.value).lower()

    def test_invalid_summary_empty(self):
        """Test that empty summary is rejected."""
        with pytest.raises(ValidationError):
            SummarySchema(summary="")

    def test_invalid_summary_whitespace_only(self):
        """Test that whitespace-only summary is rejected."""
        with pytest.raises(ValidationError):
            SummarySchema(summary="                              ")

    def test_summary_placeholder_rejection(self):
        """Test that trivial placeholders are rejected."""
        # "N/A" is only 3 characters and 1 word
        with pytest.raises(ValidationError):
            SummarySchema(summary="N/A")

        # "Summary unavailable" is 19 characters and 2 words
        with pytest.raises(ValidationError):
            SummarySchema(summary="Summary unavailable")

    def test_valid_summary_with_punctuation(self):
        """Test summary with punctuation and special characters."""
        summary = SummarySchema(
            summary="This summary contains punctuation! It has commas, periods, and question marks? Yes it does."
        )
        assert "!" in summary.summary
        assert "?" in summary.summary
        assert "," in summary.summary
        assert "." in summary.summary

    def test_valid_summary_with_numbers(self):
        """Test summary with numbers and numerical data."""
        summary = SummarySchema(
            summary="Azure has 200+ services across 60 regions with 99.99% uptime SLA for critical workloads worldwide."
        )
        assert "200+" in summary.summary
        assert "60" in summary.summary
        assert "99.99%" in summary.summary


class TestSchemaIntegration:
    """Integration tests combining multiple schemas."""

    def test_document_with_metadata_and_chunks(self):
        """Test creating a document with metadata and multiple chunks."""
        metadata = MetadataSchema(
            doc_type="technical guide",
            key_topics=["Azure", "Kubernetes", "Containers"],
            summary="Guide to deploying Kubernetes on Azure infrastructure with best practices.",
        )

        chunks = [
            ChunkSchema(
                chunk_id="doc1-chunk-1",
                text="First chunk content with sufficient length for vector embedding and retrieval.",
                doc_id="doc1",
            ),
            ChunkSchema(
                chunk_id="doc1-chunk-2",
                text="Second chunk content with more information about Kubernetes deployment strategies.",
                doc_id="doc1",
            ),
        ]

        assert metadata.doc_type == "technical guide"
        assert len(chunks) == 2
        assert all(c.doc_id == "doc1" for c in chunks)

    def test_summary_schema_with_metadata_summary(self):
        """Test that metadata summary also validates as SummarySchema."""
        metadata = MetadataSchema(
            doc_type="policy",
            summary="This comprehensive policy document provides guidelines for Azure security governance, compliance requirements, and best practices for enterprise organisations.",
        )

        # The same summary should also validate with stricter SummarySchema
        summary = SummarySchema(summary=metadata.summary)
        assert summary.summary == metadata.summary
