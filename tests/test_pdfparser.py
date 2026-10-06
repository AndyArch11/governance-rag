"""Tests for PDF parsing and cleaning.

Tests the removal of headers, footers, page numbers, and boilerplate
from PDF documents before text extraction.
"""

from types import SimpleNamespace

import pytest

from scripts.ingest.pdfparser import (
    _clean_pdf_text,
    _convert_pdf_with_docling,
    _get_docling_bool_setting,
    _is_likely_header_footer,
    extract_figures_from_docling_document,
    extract_pdf_text_and_figures,
    extract_structure_from_text,
    extract_text_from_pdf,
    validate_structure_against_toc,
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


class TestPDFStructureExtraction:
    """Tests for heading detection used by academic assessment metadata."""

    def test_rejects_numbered_prose_and_section_word_mentions(self):
        """Narrative text must not become a structural chapter or Introduction section."""
        text = """Chapter 1: Literature Review
This chapter establishes the conceptual context.
Chapter 8: the experience of participants was complex and situated.
The introduction to this discussion explains the approach.
Chapter 9: Findings
Findings

The findings are presented here.
"""

        structure = extract_structure_from_text(text)
        titles = [entry["section_title"] for entry in structure]

        assert "Chapter 1: Literature Review" in titles
        assert "Chapter 9: Findings" in titles
        assert "Findings" in titles
        assert not any("the experience of participants" in title for title in titles)
        assert not any("introduction to this discussion" in title.lower() for title in titles)

    def test_preserves_docling_markdown_heading_hierarchy(self):
        """Layout-aware Docling headings map chunks to their documented sections."""
        text = """# Abstract
Summary of the thesis.
# Chapter 1: Literature Review
## Introduction
Literature review context.
# Chapter 2: Methodology
Method details.
"""

        structure = extract_structure_from_text(text)

        assert [entry["section_title"] for entry in structure] == [
            "Abstract",
            "Chapter 1: Literature Review",
            "Introduction",
            "Chapter 2: Methodology",
        ]
        assert structure[2]["heading_path"] == "Chapter 1: Literature Review > Introduction"

    def test_markdown_chapter_heading_resets_current_chapter(self):
        """Each markdown chapter heading owns following sections in source order."""
        text = """# Chapter 1: Introduction
## Context
Chapter 1 content.
# Chapter 2: Methods
## Design
Chapter 2 content.
"""

        structure = extract_structure_from_text(text)

        assert structure[0]["chapter"] == "Chapter 1"
        assert structure[1]["chapter"] == "Chapter 1"
        assert structure[2]["chapter"] == "Chapter 2"
        assert structure[3]["chapter"] == "Chapter 2"
        assert [entry["level"] for entry in structure] == [0, 1, 0, 1]

    def test_numbered_prose_list_is_not_misclassified_as_chapters(self):
        """Numbered prose lists must not create false chapter boundaries."""
        text = """## Cultural positioning
1. First Lore principle
This is explanatory prose.
2. Second Lore principle
More explanatory prose.
## Chapter 1: Introduction
Chapter content.
"""

        structure = extract_structure_from_text(text)

        assert not any(entry["chapter"] == "Chapter 2" for entry in structure)
        assert any(entry["section_title"] == "Chapter 1: Introduction" for entry in structure)

    def test_table_of_contents_is_authoritative_for_chapter_membership(self):
        """Use ToC titles and ignore rubric-detected headings absent from the ToC."""
        text = """TABLE OF CONTENTS
Chapter 1: Introduction ........ 1
Chapter 2: Methodology ........ 20
Chapter 3: Findings ........ 50

Chapter 1: Introduction
Opening text.
Chapter 2: Methodology
Methods text.
Chapter 9: Country relationships
This is a numbered cultural framework, not a chapter in the ToC.
Chapter 3: Findings
Findings text.
"""

        structure = extract_structure_from_text(text)
        chapter_entries = [entry for entry in structure if entry["level"] == 0]

        assert [entry["chapter"] for entry in chapter_entries] == [
            "Chapter 1",
            "Chapter 2",
            "Chapter 3",
        ]
        assert chapter_entries[1]["section_title"] == "Chapter 2: Methodology"
        report = validate_structure_against_toc(text, structure)
        assert report["unexpected_chapters"] == ["Chapter 9: Country relationships"]

    def test_table_of_contents_report_surfaces_missing_chapters_and_order(self):
        text = """CONTENTS
Chapter 1: Introduction ........ 1
Chapter 2: Methodology ........ 20
Chapter 3: Findings ........ 50

Chapter 1: Introduction
Opening text.
Chapter 3: Findings
Findings text.
"""

        report = validate_structure_against_toc(text, extract_structure_from_text(text))

        assert report["toc_present"] is True
        assert report["expected_chapters"] == 3
        assert report["matched_chapters"] == 2
        assert report["coverage"] == 2 / 3
        assert report["missing_chapters"] == ["Chapter 2"]
        assert report["order_matches"] is False

    def test_markdown_table_toc_counts_only_explicit_chapters(self):
        text = """## Table of Contents
| Abstract ........................................................ ii |
| Chapter 1: Introduction ......................................... 1 |
| Background ...................................................... 2 |
| Chapter 2: Methodology ......................................... 20 |

Introduction
Opening text.
Methodology
Methods text.
"""

        structure = extract_structure_from_text(text)
        report = validate_structure_against_toc(text, structure)

        assert report["expected_chapters"] == 2
        assert report["matched_chapters"] == 2
        assert report["coverage"] == 1.0
        assert report["order_matches"] is True

    WRAPPED_MARKDOWN_TABLE_TOC = """## Table of Contents
| Abstract ........................................................ ii |
| Chapter 1: Introduction ......................................... 1 |
| Chapter 2: Context ............................................. 20 |
| Chapter 3: Narrative
    Design ......................................................... 50 |
| Chapter 4: Methodology ......................................... 80 |
| Chapter 5: Findings ............................................. |
| 140 |
| Chapter 6: Discussion ......................................... 170 |
| Chapter 7: Implications ........................................ 200 |
| Chapter 8: Synthesis
    ............................................................... 230 |
| Chapter 9: Limitations ......................................... 260 |
| Chapter
| 10: Conclusion ................................................ 300 |
| Background ..................................................... 310 |

Chapter 1: Introduction
Chapter 2: Context
Chapter 3: Narrative Design
Chapter 4: Methodology
Chapter 5: Findings
Chapter 6: Discussion
Chapter 7: Implications
Chapter 8: Synthesis
Chapter 9: Limitations
Chapter 10: Conclusion
"""

    @pytest.mark.parametrize("text", [WRAPPED_MARKDOWN_TABLE_TOC])
    def test_markdown_table_toc_supports_wrapped_chapter_rows(self, text):
        structure = extract_structure_from_text(text)
        report = validate_structure_against_toc(text, structure)

        assert report["expected_chapters"] == 10
        assert report["matched_chapters"] == 10
        assert report["coverage"] == 1.0

    def test_table_of_contents_supports_numbered_and_wrapped_entries(self):
        text = """CONTENTS
1. Introduction ........ 1
2. Methodology
................ 20
3. Findings ........ 50

1. Introduction
Opening text.
2. Methodology
Methods text.
3. Findings
Findings text.
"""

        structure = extract_structure_from_text(text)
        report = validate_structure_against_toc(text, structure)

        assert report["toc_present"] is True
        assert report["expected_chapters"] == 3
        assert report["matched_chapters"] == 3
        assert report["order_matches"] is True

    def test_unnumbered_body_headings_match_numbered_toc_chapters(self):
        text = """CONTENTS
Chapter 1: Introduction ........ 1
Chapter 2: Country and Kinship ........ 20

Introduction
This thesis begins here.
Country and Kinship
This chapter discusses relational concepts.
"""

        structure = extract_structure_from_text(text)
        chapter_entries = [entry for entry in structure if entry["level"] == 0]
        report = validate_structure_against_toc(text, structure)

        assert [entry["chapter"] for entry in chapter_entries] == ["Chapter 1", "Chapter 2"]
        assert report["matched_chapters"] == 2
        assert report["order_matches"] is True

    def test_outline_only_toc_controls_chapter_extraction(self):
        text = """CONTENTS

Chapter 1: Introduction
Chapter 2: Country and Kinship

Introduction
This thesis begins here.
Country and Kinship
The chapter considers relational healing.
"""

        structure = extract_structure_from_text(text)
        chapter_entries = [entry for entry in structure if entry["level"] == 0]
        report = validate_structure_against_toc(text, structure)

        assert report["toc_present"] is True
        assert [entry["chapter"] for entry in chapter_entries] == ["Chapter 1", "Chapter 2"]
        assert report["matched_chapters"] == 2
        assert report["order_matches"] is True

    def test_unnumbered_outline_toc_entries_are_matched_to_body_headings(self):
        text = """CONTENTS
Introduction
Literature Review
Methodology
Findings and Discussion
Conclusion

Introduction
Introductory text.
Literature Review
Prior research.
Methodology
Research methods.
Findings and Discussion
Study results.
Conclusion
Closing summary.
"""

        structure = extract_structure_from_text(text)
        chapter_entries = [entry for entry in structure if entry["level"] == 0]
        report = validate_structure_against_toc(text, structure)

        assert [entry["chapter"] for entry in chapter_entries] == [
            "Chapter 1",
            "Chapter 2",
            "Chapter 3",
            "Chapter 4",
            "Chapter 5",
        ]
        assert report["matched_chapters"] == 5
        assert report["order_matches"] is True

    def test_unnumbered_outline_entries_with_page_leaders_are_matched(self):
        text = """CONTENTS
Introduction ........ 1
Methodology ........ 10

Introduction
Introductory text.
Methodology
Research methods.
"""

        structure = extract_structure_from_text(text)
        chapter_entries = [entry for entry in structure if entry["level"] == 0]
        report = validate_structure_against_toc(text, structure)

        assert [entry["chapter"] for entry in chapter_entries] == ["Chapter 1", "Chapter 2"]
        assert report["matched_chapters"] == 2
        assert report["order_matches"] is True

    def test_toc_joins_title_continuation_before_leader_dots(self):
        text = """CONTENTS
Chapter 1: Indigenous research methods
and community health ........ 1

Chapter 1: Indigenous research methods and community health
This chapter describes the approach.
"""

        structure = extract_structure_from_text(text)
        chapter_entries = [entry for entry in structure if entry["level"] == 0]

        assert chapter_entries[0]["toc_title"] == (
            "Indigenous research methods and community health"
        )

    def test_toc_joins_unlabelled_continuation_and_separate_page_row(self):
        text = """CONTENTS
Chapter 1: Indigenous research
methods in community health
................ 1

Chapter 1: Indigenous research methods in community health
This chapter describes the approach.
"""

        structure = extract_structure_from_text(text)
        chapter_entries = [entry for entry in structure if entry["level"] == 0]

        assert len(chapter_entries) == 1
        assert chapter_entries[0]["toc_title"] == (
            "Indigenous research methods in community health"
        )


class TestDoclingFallback:
    """Tests for optional Docling conversion before pypdf fallback."""

    def test_prefers_docling_markdown_when_available(self, monkeypatch):
        """A successful Docling conversion bypasses the text-only parser."""
        monkeypatch.setattr(
            "scripts.ingest.pdfparser._extract_text_with_docling",
            lambda path: "# Abstract\nStructured thesis text",
        )

        assert extract_text_from_pdf("thesis.pdf") == "# Abstract\nStructured thesis text"

    def test_docling_boolean_settings_support_scan_override(self, monkeypatch):
        """OCR stays off by default but can be enabled for scanned PDFs."""
        monkeypatch.delenv("DOCLING_ENABLE_OCR", raising=False)
        assert _get_docling_bool_setting("DOCLING_ENABLE_OCR", False) is False

        monkeypatch.setenv("DOCLING_ENABLE_OCR", "true")
        assert _get_docling_bool_setting("DOCLING_ENABLE_OCR", False) is True

    def test_docling_enables_picture_image_generation(self, monkeypatch):
        from docling.datamodel import pipeline_options

        captured = {}

        class FakePipelineOptions:
            def __init__(self):
                self.generate_picture_images = False
                self.heading_hierarchy_options = SimpleNamespace(enabled=True)

        class FakeConverter:
            def __init__(self, *, format_options):
                captured["options"] = next(iter(format_options.values())).pipeline_options

            def convert(self, path):
                return SimpleNamespace(document="docling-document")

        class CapturingPdfFormatOption:
            def __init__(self, *, pipeline_options):
                self.pipeline_options = pipeline_options

        monkeypatch.delenv("DOCLING_GENERATE_PICTURE_IMAGES", raising=False)
        monkeypatch.setattr(pipeline_options, "PdfPipelineOptions", FakePipelineOptions)
        monkeypatch.setattr("docling.document_converter.PdfFormatOption", CapturingPdfFormatOption)
        monkeypatch.setattr("docling.document_converter.DocumentConverter", FakeConverter)

        document = _convert_pdf_with_docling("thesis.pdf")

        assert document == "docling-document"
        assert captured["options"].generate_picture_images is True

    def test_extracts_figure_caption_alt_text_page_bounds_and_image(self):
        image = object()
        picture = SimpleNamespace(
            label=SimpleNamespace(value="picture"),
            captions=[],
            prov=[
                SimpleNamespace(
                    page_no=7,
                    bbox=SimpleNamespace(l=10.0, t=20.0, r=110.0, b=120.0, coord_origin="TOPLEFT"),
                )
            ],
            meta=SimpleNamespace(description="Diagram showing community governance."),
            annotations=[],
            caption_text=lambda document: "Figure 3. Community governance model",
            get_image=lambda document: image,
        )

        class FakeDoclingDocument:
            def iterate_items(self, traverse_pictures=False):
                assert traverse_pictures is True
                return [(picture, 0)]

        figures = extract_figures_from_docling_document(FakeDoclingDocument())

        assert len(figures) == 1
        assert figures[0]["figure_number"] == 1
        assert figures[0]["caption_number"] == "3"
        assert figures[0]["page_number"] == 7
        assert figures[0]["caption"] == "Figure 3. Community governance model"
        assert figures[0]["alt_text"] == "Diagram showing community governance."
        assert figures[0]["bbox"] == {
            "left": 10.0,
            "top": 20.0,
            "right": 110.0,
            "bottom": 120.0,
            "coordinate_origin": "TOPLEFT",
        }
        assert figures[0]["image"] is image

    def test_combined_extraction_maps_figure_caption_to_heading_path(self, monkeypatch):
        caption = "Figure 1. Community governance model"
        markdown = (
            "# Chapter 1: Introduction\n"
            "## Conceptual Framework\n"
            f"{caption}\n"
            "Supporting discussion."
        )
        picture = SimpleNamespace(
            label=SimpleNamespace(value="picture"),
            captions=[],
            prov=[],
            meta=SimpleNamespace(description=""),
            annotations=[],
            caption_text=lambda document: caption,
            get_image=lambda document: None,
        )
        document = SimpleNamespace(
            export_to_markdown=lambda: markdown,
            iterate_items=lambda traverse_pictures: [(picture, 0)],
        )
        monkeypatch.setattr(
            "scripts.ingest.pdfparser._convert_pdf_with_docling", lambda path: document
        )

        extracted_text, figures = extract_pdf_text_and_figures("thesis.pdf")

        assert extracted_text == markdown
        assert figures[0]["chapter"] == "Chapter 1"
        assert figures[0]["heading_path"] == "Chapter 1: Introduction > Conceptual Framework"


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
