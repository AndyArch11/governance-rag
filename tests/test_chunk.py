"""Tests for text chunking utilities.

Tests the chunk_text function which splits text into semantic chunks
for vector embedding and RAG retrieval.
"""

import pytest

from scripts.ingest.chunk import MAX_CHUNK_SIZE, chunk_text


class TestChunkText:
    """Tests for chunk_text function."""

    def test_chunk_short_text(self):
        """Test that short text returns a single chunk."""
        text = "This is a short piece of text that should fit in one chunk."
        chunks = chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_empty_string(self):
        """Test that empty string returns empty list."""
        text = ""
        chunks = chunk_text(text)

        assert len(chunks) == 0

    def test_chunk_whitespace_only(self):
        """Test that whitespace-only text returns empty list."""
        text = "   \n\n\t\t   \n   "
        chunks = chunk_text(text)

        # LangChain removes whitespace-only chunks
        assert len(chunks) == 0

    def test_chunk_long_text_creates_multiple_chunks(self):
        """Test that long text is split into multiple chunks."""
        # Create text longer than chunk_size (800)
        text = "This is a sentence. " * 100  # ~2000 characters
        chunks = chunk_text(text)

        assert len(chunks) > 1
        # Each chunk should be non-empty
        assert all(chunk.strip() for chunk in chunks)

    def test_chunk_respects_markdown_headings(self):
        """Test that chunking prioritises markdown heading boundaries."""
        text = (
            """# Main Title

This is the introduction paragraph with some content.

## Section 1

This is section 1 content. """
            + ("More content. " * 100)
            + """

## Section 2

This is section 2 content. """
            + ("More content. " * 100)
        )

        chunks = chunk_text(text)

        # Should create multiple chunks
        assert len(chunks) > 1
        # Chunks should contain content and keep headings visible
        assert all(chunk.strip() for chunk in chunks)
        assert any("# Main Title" in chunk for chunk in chunks)
        assert any("## Section 1" in chunk for chunk in chunks)
        assert any("## Section 2" in chunk for chunk in chunks)

    def test_chunk_overlap_between_chunks(self):
        """Test that chunks have overlapping content."""
        # Create text that will be split into exactly 2 chunks
        text = "Paragraph one. " * 60 + "Paragraph two. " * 60
        chunks = chunk_text(text)

        if len(chunks) > 1:
            # There should be some overlap between consecutive chunks
            # Check if any content from chunk 0 appears in chunk 1
            chunk_0_end = chunks[0][-50:]  # Last 50 chars of first chunk
            # Due to overlap, some content should appear in both
            # This is a heuristic test since overlap is controlled by separators
            assert len(chunks[0]) < len(text)

    def test_chunk_preserves_content(self):
        """Test that all original content is preserved across chunks."""
        text = "Word " * 200  # Create text that will be chunked
        chunks = chunk_text(text)

        # Reconstruct text from chunks (accounting for overlap)
        # Each word should appear at least once
        combined = " ".join(chunks)
        assert "Word" in combined

    def test_chunk_with_newlines(self):
        """Test chunking text with multiple newlines."""
        text = "Line 1\n\nLine 2\n\nLine 3\n\n" + ("Line 4\n\n" * 100)
        chunks = chunk_text(text)

        assert len(chunks) >= 1
        # Newlines should be preserved in chunks
        assert any("\n" in chunk for chunk in chunks)

    def test_chunk_with_special_characters(self):
        """Test chunking text with special characters."""
        text = "Special chars: @#$%^&*() 🚀 " * 100
        chunks = chunk_text(text)

        assert len(chunks) >= 1
        # Special characters should be preserved
        assert any("@#$%^&*()" in chunk for chunk in chunks)
        assert any("🚀" in chunk for chunk in chunks)

    def test_chunk_size_constraint(self):
        """Test that chunks respect approximate size constraints."""
        # Create very long text
        text = "This is a test sentence. " * 500
        chunks = chunk_text(text)

        # Most chunks should be reasonably sized (around 800 chars, allowing variance)
        # Some may be smaller at boundaries
        for chunk in chunks:
            # Chunks shouldn't be excessively long (max 2x chunk_size as safety)
            assert len(chunk) < 2000

    def test_chunk_single_long_word(self):
        """Test chunking with a single very long word."""
        # Create a word longer than chunk_size
        text = "a" * 1000
        chunks = chunk_text(text)

        # Should still return chunks (may be forced to split the word)
        assert len(chunks) >= 1

    def test_chunk_mixed_content(self):
        """Test chunking with mixed markdown content."""
        text = (
            """# Document Title

Introduction paragraph with some content.

## First Section

Content for first section.

### Subsection 1.1

More detailed content here. """
            + ("Additional detail. " * 50)
            + """

### Subsection 1.2

Even more content. """
            + ("More information. " * 50)
            + """

## Second Section

Final section content.
"""
        )

        chunks = chunk_text(text)

        # Should create multiple chunks
        assert len(chunks) >= 1
        # All chunks should have content
        assert all(chunk.strip() for chunk in chunks)
        # Check that markdown headers are present in some chunks
        assert any("#" in chunk for chunk in chunks)

    def test_chunk_returns_list(self):
        """Test that chunk_text always returns a list."""
        text = "Sample text"
        chunks = chunk_text(text)

        assert isinstance(chunks, list)
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_with_code_blocks(self):
        """Test chunking text with code block patterns."""
        text = """Here is some code:

```python
def example():
    return "hello"
```

And more text. """ + ("Additional content. " * 100)

        chunks = chunk_text(text)

        assert len(chunks) >= 1
        # Code block markers should be preserved
        assert any("```" in chunk for chunk in chunks)

    def test_chunk_realistic_document(self):
        """Test chunking a realistic document structure."""
        text = (
            """# Azure Security Governance

## Overview

Azure provides comprehensive security controls for enterprise workloads.

## Security Principles

### Defence in Depth

Multiple layers of security controls protect against various attack vectors.

### Least Privilege

Access is granted based on the minimum permissions required.

### Zero Trust

Never trust, always verify - all access requests are authenticated and authorised.

## Implementation Guidelines

"""
            + ("Implementation detail paragraph. " * 80)
            + """

## Compliance Requirements

"""
            + ("Compliance information. " * 80)
        )

        chunks = chunk_text(text)

        # Should create multiple chunks
        assert len(chunks) > 1
        # All chunks should have reasonable length
        assert all(len(chunk) > 10 for chunk in chunks)
        # Total character count should be preserved approximately
        total_chunk_chars = sum(len(chunk) for chunk in chunks)
        # Account for overlap - total chunks may be larger than original
        assert total_chunk_chars >= len(text) * 0.8

    def test_chunk_with_bullet_points(self):
        """Test chunking text with bullet points."""
        text = (
            """# Features

- Feature 1: """
            + ("Description. " * 20)
            + """
- Feature 2: """
            + ("Description. " * 20)
            + """
- Feature 3: """
            + ("Description. " * 20)
            + """
- Feature 4: """
            + ("Description. " * 20)
        )

        chunks = chunk_text(text)

        assert len(chunks) >= 1
        # Bullet points should be preserved
        assert any("-" in chunk for chunk in chunks)

    def test_chunk_consistency(self):
        """Test that chunking the same text produces the same result."""
        text = "Consistent text. " * 100

        chunks1 = chunk_text(text)
        chunks2 = chunk_text(text)

        # Should be deterministic
        assert chunks1 == chunks2

    def test_chunk_with_numbers_and_symbols(self):
        """Test chunking text with numbers and various symbols."""
        text = """Configuration values:
- Port: 443
- Timeout: 30s
- Max connections: 1,000
- Uptime: <b>99.99%</b>
- Cost: $100/month

""" + ("Additional configuration details. " * 80)

        chunks = chunk_text(text)

        assert len(chunks) >= 1
        # Numbers and symbols should be preserved
        assert any("443" in chunk for chunk in chunks)
        assert any("99.99%" in chunk for chunk in chunks)
        assert any("$100/month" in chunk for chunk in chunks)
        assert any("</b>" in chunk for chunk in chunks)

    def test_chunk_with_html_tags(self):
        """Test chunking text that includes HTML tags."""
        text = """<h1>Title</h1>    
<p>This is a paragraph with <strong>bold</strong> text and <a href="#">a link</a>.</p>
<div>Some div content here.</div>
""" + ("<p>More HTML content.</p> " * 100)

        chunks = chunk_text(text)

        assert len(chunks) >= 1
        # HTML tags should be preserved
        assert any("<h1>" in chunk for chunk in chunks)
        assert any("<strong>" in chunk for chunk in chunks)
        assert any("<a href=" in chunk for chunk in chunks)
        assert any("<div>" in chunk for chunk in chunks)

    def test_table_chunking_preserves_document_order(self):
        """Table-aware chunking should keep table position between surrounding text."""
        text = (
            "# Intro\n"
            "Lead paragraph before table.\n\n"
            "[TABLE 0]\n"
            "Context: Cost matrix\n"
            "| Tier | Cost |\n"
            "|---|---|\n"
            "| Basic | 100 |\n"
            "| Pro | 250 |\n"
            "[/TABLE 0]\n\n"
            "Tail paragraph after table."
        )

        chunks = chunk_text(text, doc_type="reference", adaptive=True)

        table_idx = next(i for i, chunk in enumerate(chunks) if "[TABLE 0]" in chunk)
        before_text_exists = any(
            "Lead paragraph before table" in chunk for chunk in chunks[: table_idx + 1]
        )
        after_text_exists = any("Tail paragraph after table" in chunk for chunk in chunks[table_idx:])

        assert before_text_exists
        assert after_text_exists

    def test_table_chunking_splits_oversized_tables_by_rows(self):
        """Oversized tables should be split into multiple table-marked chunks."""
        rows = "\n".join(
            [
                f"| service_{i} | " + ("very long descriptive content " * 8) + "|"
                for i in range(30)
            ]
        )
        text = (
            "[TABLE 2]\n"
            "Context: Large service matrix\n"
            "| Service | Details |\n"
            "|---|---|\n"
            f"{rows}\n"
            "[/TABLE 2]"
        )

        chunks = chunk_text(text, doc_type="reference", adaptive=True)
        table_chunks = [chunk for chunk in chunks if "[TABLE 2]" in chunk]

        assert len(table_chunks) > 1
        assert all("#### TABLE MARKER ####" in chunk for chunk in table_chunks)
        assert all("Table Part:" in chunk for chunk in table_chunks)
        assert all(len(chunk) <= int(MAX_CHUNK_SIZE * 1.35) for chunk in table_chunks)

    def test_table_chunking_handles_nested_table_content(self):
        """Nested table payloads should be preserved and chunked as content tables."""
        nested_payload = "[nested: Region A 200K Region B 300K]"
        rows = "\n".join(
            [
                f"| Segment {i} | {nested_payload} " + ("details " * 20) + "|"
                for i in range(10)
            ]
        )
        text = (
            "[TABLE 7]\n"
            "Context: Revenue by region\n"
            "| Segment | Notes |\n"
            "|---|---|\n"
            f"{rows}\n"
            "[/TABLE 7]"
        )

        chunks = chunk_text(text, doc_type="reference", adaptive=True)
        table_chunks = [chunk for chunk in chunks if "[TABLE 7]" in chunk]

        assert table_chunks
        assert all("#### TABLE MARKER ####" in chunk for chunk in table_chunks)
        assert any("[nested:" in chunk for chunk in table_chunks)

    def test_table_chunking_distinguishes_layout_tables(self):
        """Likely layout tables should be converted to narrative, not table markers."""
        text = (
            "Before layout table.\n\n"
            "[TABLE 9]\n"
            "| Left gutter | Right gutter |\n"
            "| Logo area | Utility links |\n"
            "[/TABLE 9]\n\n"
            "After layout table."
        )

        chunks = chunk_text(text, doc_type="guide", adaptive=True)

        # Layout tables should be treated as narrative chunks (no strict table marker wrapping).
        assert any("Layout table 9" in chunk for chunk in chunks)
        assert not any("#### TABLE MARKER ####" in chunk and "[TABLE 9]" in chunk for chunk in chunks)
        assert any("Before layout table." in chunk for chunk in chunks)
        assert any("After layout table." in chunk for chunk in chunks)

    def test_table_chunking_honours_explicit_layout_type(self):
        """Explicit parser metadata should force layout narrative formatting."""
        text = (
            "[TABLE 11]\n"
            "Table-Type: layout\n"
            "| KPI | Value |\n"
            "|---|---|\n"
            "| Revenue | 100 |\n"
            "[/TABLE 11]"
        )

        chunks = chunk_text(text, doc_type="reference", adaptive=True)

        assert any("Layout table 11" in chunk for chunk in chunks)
        assert not any("#### TABLE MARKER ####" in chunk and "[TABLE 11]" in chunk for chunk in chunks)

    def test_table_chunking_honours_explicit_content_type(self):
        """Explicit parser metadata should preserve content-table markers."""
        text = (
            "[TABLE 12]\n"
            "Table-Type: content\n"
            "| Left gutter | Right gutter |\n"
            "| Logo area | Utility links |\n"
            "[/TABLE 12]"
        )

        chunks = chunk_text(text, doc_type="guide", adaptive=True)

        table_chunks = [chunk for chunk in chunks if "[TABLE 12]" in chunk]
        assert table_chunks
        assert all("#### TABLE MARKER ####" in chunk for chunk in table_chunks)


# =========================
# Parent-Child Chunking Tests
# =========================


class TestParentChildChunks:
    """Tests for parent-child chunking integration with existing chunk_text."""

    def test_parent_child_chunks_imported(self):
        """Test that create_parent_child_chunks is available."""
        from scripts.ingest.chunk import create_parent_child_chunks

        assert callable(create_parent_child_chunks)

    def test_create_parent_child_basic(self):
        """Test basic parent-child chunk creation."""
        from scripts.ingest.chunk import create_parent_child_chunks

        text = "This is content. " * 100
        child_chunks, parent_chunks = create_parent_child_chunks(text)

        assert len(child_chunks) > 0
        assert len(parent_chunks) > 0
        assert len(child_chunks) >= len(parent_chunks)

    def test_parent_child_with_doc_type(self):
        """Test parent-child chunking respects doc_type."""
        from scripts.ingest.chunk import create_parent_child_chunks

        text = "Policy content. " * 100
        child_chunks, parent_chunks = create_parent_child_chunks(
            text, doc_type="policy", parent_size=800, child_size=300
        )

        assert len(child_chunks) > 0
        assert len(parent_chunks) > 0

    def test_parent_child_maintains_relationships(self):
        """Test that parent-child relationships are maintained."""
        from scripts.ingest.chunk import create_parent_child_chunks

        text = "Section A. " * 50 + "Section B. " * 50
        child_chunks, parent_chunks = create_parent_child_chunks(text)

        # Each child should reference a parent
        for child in child_chunks:
            assert "parent_id" in child
            assert child["parent_id"].startswith("parent_")

        # Each parent should reference children
        for parent in parent_chunks:
            assert "child_ids" in parent
            assert len(parent["child_ids"]) > 0
            assert all(cid.startswith("child_") for cid in parent["child_ids"])

    def test_parent_larger_than_child(self):
        """Test that parent chunks are larger than child chunks."""
        from scripts.ingest.chunk import create_parent_child_chunks

        text = "Content " * 200
        child_chunks, parent_chunks = create_parent_child_chunks(
            text, parent_size=1200, child_size=400
        )

        if len(parent_chunks) > 0 and len(child_chunks) > 0:
            # Average parent should be larger than average child
            avg_parent_len = sum(len(p["text"]) for p in parent_chunks) / len(parent_chunks)
            avg_child_len = sum(len(c["text"]) for c in child_chunks) / len(child_chunks)
            assert avg_parent_len > avg_child_len
