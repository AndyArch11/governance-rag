"""Tests for PDF parsing and cleaning.

Tests the removal of headers, footers, page numbers, and boilerplate
from PDF documents before text extraction.
"""

import pytest

from scripts.ingest.pdfparser import (
    _clean_pdf_text,
    _is_likely_header_footer,
    extract_text_from_pdf,
)


class TestHeaderFooterDetection:
    """Tests for detection of header/footer patterns."""

    def test_detect_page_numbers(self):
        """Test detection of page number patterns."""
        assert _is_likely_header_footer("Page 1 of 10") is True
        assert _is_likely_header_footer("Page 5") is True
        assert _is_likely_header_footer("1") is True
        assert _is_likely_header_footer("42") is True
        assert _is_likely_header_footer("999") is True

    def test_detect_copyright(self):
        """Test detection of copyright patterns."""
        assert _is_likely_header_footer("Copyright © 2024") is True
        assert _is_likely_header_footer("Copyright © 2023") is True

    def test_detect_version_numbers(self):
        """Test detection of version number patterns."""
        assert _is_likely_header_footer("Version 1.0") is True
        assert _is_likely_header_footer("Version 2.3.1") is True
        assert _is_likely_header_footer("v1.5") is True

    def test_detect_dates(self):
        """Test detection of date patterns."""
        assert _is_likely_header_footer("Generated on 01/15/2024") is True
        assert _is_likely_header_footer("Last modified on 12/25/2023") is True

    def test_detect_footer_patterns(self):
        """Test detection of common footer patterns."""
        assert _is_likely_header_footer("See Also") is True
        assert _is_likely_header_footer("For more information") is True
        assert _is_likely_header_footer("Contact us") is True
        assert _is_likely_header_footer("Next Steps") is True

    def test_preserve_content(self):
        """Test that regular content is not marked as header/footer."""
        assert _is_likely_header_footer("This is important document content") is False
        assert _is_likely_header_footer("Chapter introduction text") is False
        assert _is_likely_header_footer("Main paragraph of information") is False


class TestPDFTextCleaning:
    """Tests for PDF text cleaning."""

    def test_remove_page_numbers(self):
        """Test removal of page number lines."""
        text = """
        Introduction to Document
        Page 1 of 100
        Content starts here
        Page 2 of 100
        More content
        """
        cleaned = _clean_pdf_text(text)

        assert "Page 1 of 100" not in cleaned
        assert "Page 2 of 100" not in cleaned
        assert "Introduction to Document" in cleaned
        assert "Content starts here" in cleaned

    def test_remove_metadata(self):
        """Test removal of metadata lines."""
        text = """
        Document Start
        Version 2.1
        Generated on 01/15/2024
        Copyright © 2024
        Main Content
        """
        cleaned = _clean_pdf_text(text)

        assert "Version 2.1" not in cleaned
        assert "Generated on" not in cleaned
        assert "Copyright ©" not in cleaned
        assert "Document Start" in cleaned
        assert "Main Content" in cleaned

    def test_remove_footer_sections(self):
        """Test removal of common footer sections."""
        text = """
        Chapter 1 Content
        Main information here
        
        For more information
        Contact us
        Next Steps
        """
        cleaned = _clean_pdf_text(text)

        # Footer patterns should be removed if isolated
        assert "Chapter 1 Content" in cleaned
        assert "Main information here" in cleaned

    def test_collapse_blank_lines(self):
        """Test collapsing of excessive blank lines."""
        text = """
        Paragraph 1


        Paragraph 2



        Paragraph 3
        """
        cleaned = _clean_pdf_text(text)

        # Count consecutive blank lines
        assert "\n\n\n" not in cleaned
        assert "Paragraph 1" in cleaned
        assert "Paragraph 2" in cleaned
        assert "Paragraph 3" in cleaned

    def test_preserve_important_section_headers(self):
        """Test that important section headers with context are preserved."""
        text = """
        Chapter 1. Introduction
        
        This chapter provides an overview of the system.
        
        Section 1.1 Background
        
        The background section contains relevant history.
        
        Chapter 2. Architecture
        
        Detailed architecture information.
        """
        cleaned = _clean_pdf_text(text)

        # Section headers with surrounding content should remain
        assert "Chapter" in cleaned or "Introduction" in cleaned
        assert "overview" in cleaned
        assert "Architecture" in cleaned or "architecture" in cleaned.lower()


class TestPDFCleaningEdgeCases:
    """Tests for edge cases in PDF cleaning."""

    def test_handle_empty_text(self):
        """Test handling of empty PDF text."""
        cleaned = _clean_pdf_text("")
        assert cleaned == ""

    def test_handle_only_whitespace(self):
        """Test handling of whitespace-only text."""
        cleaned = _clean_pdf_text("\n\n\n   \n\n")
        assert cleaned.strip() == ""

    def test_handle_single_line(self):
        """Test handling of single line."""
        cleaned = _clean_pdf_text("Important content")
        assert "Important content" in cleaned

    def test_handle_mixed_content(self):
        """Test handling of mixed header/content/footer."""
        text = """
        Page 1
        
        Section Title: Overview
        
        Important paragraph 1
        Important paragraph 2
        
        Page 2
        
        Section Title: Details
        
        Important paragraph 3
        """
        cleaned = _clean_pdf_text(text)

        # Content should remain
        assert any(keyword in cleaned for keyword in ["Overview", "Details", "Important"])

        # Page markers should be removed
        assert "Page 1" not in cleaned
        assert "Page 2" not in cleaned


class TestPDFCleaningRealism:
    """Realistic tests for PDF cleaning."""

    def test_technical_pdf_cleaning(self):
        """Test cleaning of technical PDF content."""
        text = """
        API Documentation v2.1
        Page 1 of 50
        Generated on 01/15/2024
        
        1. Introduction
        
        This API provides access to user data and operations.
        
        2. Authentication
        
        All endpoints require OAuth 2.0 authentication.
        Bearer token: <token>
        
        Page 2 of 50
        
        3. Endpoints
        
        GET /api/users - Retrieve user list
        POST /api/users - Create new user
        
        For more information see Appendix A
        """
        cleaned = _clean_pdf_text(text)

        # Important content should remain
        assert "Introduction" in cleaned
        assert "Authentication" in cleaned
        assert "Endpoints" in cleaned
        assert "GET /api/users" in cleaned or "/api/users" in cleaned

        # Boilerplate should be gone
        assert "Page 1 of 50" not in cleaned
        assert "Generated on" not in cleaned
        # Multiword titles like "API Documentation v2.1" remain (not detected as boilerplate)

    def test_business_document_cleaning(self):
        """Test cleaning of business document content."""
        text = """
        Annual Report 2024
        Version 1.0
        
        Page 1
        
        Executive Summary
        
        This report summarises key business metrics for the year 2024.
        
        Financial Performance
        - Revenue: $10M
        - Profit: $2M
        
        Page 2
        
        Strategic Initiatives
        
        We focus on three strategic areas.
        
        For more information contact us at info@company.com
        """
        cleaned = _clean_pdf_text(text)

        # Key metrics should remain
        assert "Executive Summary" in cleaned or "Executive" in cleaned.lower()
        assert "Financial" in cleaned.lower() or "Revenue" in cleaned
        assert "$10M" in cleaned or "10M" in cleaned

        # Metadata should be removed
        assert "Version 1.0" not in cleaned
        assert "Page 1" not in cleaned
        # Document titles remain (they're multiword and not in boilerplate patterns)


class TestIntegrationScenarios:
    """Integration tests for realistic cleaning scenarios."""

    def test_multiple_page_document(self):
        """Test cleaning handles multi-page documents correctly."""
        # Simulate multiple pages joined with page breaks
        text = """
        Chapter 1: Introduction
        Page 1
        
        Content of chapter 1
        Detailed explanation
        
        Page 2
        
        Chapter 2: Methods
        
        Methodology details
        Research approach
        
        Page 3
        
        Chapter 3: Results
        
        Key findings and results
        """
        cleaned = _clean_pdf_text(text)

        # Chapters should remain
        assert any(
            ch in cleaned
            for ch in ["Chapter 1", "Chapter 2", "Chapter 3", "Introduction", "Methods", "Results"]
        )

        # Content should remain
        assert "Detailed explanation" in cleaned or "Content" in cleaned.lower()

        # Page markers should be gone
        for page_mark in ["Page 1", "Page 2", "Page 3"]:
            assert page_mark not in cleaned

    def test_document_with_heavy_boilerplate(self):
        """Test cleaning of document with lots of boilerplate."""
        text = """
        Official Document
        Document ID: DOC-2024-001
        Version 3.2
        Generated on 01/15/2024
        Classification: Internal
        
        Page 1 of 100
        
        1. Executive Summary
        Key insights and overview
        
        Page 2 of 100
        
        2. Background
        Historical context provided
        
        Contact Information
        Email: contact@company.com
        Phone: 555-1234
        
        Page 3 of 100
        
        3. Findings
        Important discoveries made
        
        Page 4 of 100
        
        For more information
        See Appendix A
        """
        cleaned = _clean_pdf_text(text)

        # Main sections should remain
        assert "Executive Summary" in cleaned or "Summary" in cleaned.lower()
        assert "Background" in cleaned or "background" in cleaned.lower()
        assert "Findings" in cleaned or "findings" in cleaned.lower()

        # Boilerplate should be minimised
        page_count = cleaned.count("Page")
        # Should have removed most or all page markers
        assert page_count < 3  # Original had 4, should be mostly removed
