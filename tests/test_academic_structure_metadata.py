"""Tests for academic document structure metadata extraction and chunk enrichment.

Verifies that chapter/section structure is correctly extracted from PDFs
and propagated to chunk metadata during ingestion.
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest

from scripts.ingest.chunk import create_enhanced_metadata
from scripts.ingest.pdfparser import (
    _normalise_chapter_number,
    extract_structure_from_text,
    map_text_to_structure,
)
from scripts.utils.schemas import EnhancedChunkMetadata


class TestStructureExtraction:
    """Test structure extraction from document text."""

    def test_extract_chapters_with_numbers(self):
        """Test extraction of numbered chapters."""
        text = """
        Chapter 1: Introduction
        
        This is the introduction content.
        
        Chapter 2: Literature Review
        
        This is the literature review.
        
        Chapter 3 Methods
        
        This describes the methods.
        """

        structure = extract_structure_from_text(text)

        assert len(structure) >= 3
        assert any(s["chapter"] == "Chapter 1" for s in structure)
        assert any(s["chapter"] == "Chapter 2" for s in structure)
        assert any(s["chapter"] == "Chapter 3" for s in structure)

    def test_extract_chapters_with_words(self):
        """Test extraction of chapters with word numbers."""
        text = """
        Chapter One: Introduction
        
        Content here.
        
        Chapter Two: Methods
        
        More content.
        """

        structure = extract_structure_from_text(text)

        assert len(structure) >= 2
        assert any(s["chapter"] == "Chapter 1" for s in structure)
        assert any(s["chapter"] == "Chapter 2" for s in structure)

    def test_extract_chapters_with_roman_numerals(self):
        """Test extraction of chapters with roman numerals."""
        text = """
        I. Introduction
        
        Content for chapter 1.
        
        II. Background
        
        Content for chapter 2.
        
        III - Methods
        
        Content for chapter 3.
        """

        structure = extract_structure_from_text(text)

        assert len(structure) >= 3
        assert any(s["chapter"] == "Chapter 1" for s in structure)
        assert any(s["chapter"] == "Chapter 2" for s in structure)
        assert any(s["chapter"] == "Chapter 3" for s in structure)

    def test_extract_standard_sections(self):
        """Test extraction of standard academic sections without chapter numbers."""
        text = """
        Abstract
        
        This study examines...
        
        Introduction
        
        The purpose of this research...
        
        Methodology
        
        We used a mixed methods approach...
        
        Results
        
        Our findings show...
        
        Discussion
        
        These results indicate...
        
        Conclusion
        
        In summary...
        """

        structure = extract_structure_from_text(text)

        assert len(structure) >= 4
        # Should detect Introduction, Methodology, Results, Discussion as sections
        section_names = [s["section_title"] for s in structure]
        assert any("Introduction" in name for name in section_names)
        assert any("Methodology" in name or "Methods" in name for name in section_names)
        assert any("Results" in name for name in section_names)
        assert any("Discussion" in name for name in section_names)

    def test_extract_subsections(self):
        """Test extraction of numbered subsections."""
        text = """
        Chapter 2: Methods
        
        This chapter describes the methodology.
        
        2.1 Research Design
        
        The research used a qualitative approach.
        
        2.1.1 Sampling
        
        Participants were recruited through...
        
        2.2 Data Collection
        
        Data was collected via interviews.
        """

        structure = extract_structure_from_text(text)

        # Should detect chapter and subsections
        assert len(structure) >= 3  # At least chapter and some subsections
        assert any(s["level"] == 0 for s in structure)  # Chapter level
        assert any(s["level"] >= 1 for s in structure)  # At least one subsection level

    def test_heading_path_construction(self):
        """Test that heading paths are correctly constructed."""
        text = """
        Chapter 1: Introduction
        
        Content.
        
        1.1 Background
        
        More content.
        
        1.1.1 Historical Context
        
        Even more content.
        """

        structure = extract_structure_from_text(text)

        # Find the deepest subsection
        deepest = [s for s in structure if s.get("level", 0) == 2]
        if deepest:
            assert " > " in deepest[0]["heading_path"]
            # Should be something like "Chapter 1 > Background > Historical Context"


class TestChapterNormalisation:
    """Test chapter number normalisation."""

    def test_normalise_digit(self):
        """Test normalising digit chapter numbers."""
        assert _normalise_chapter_number("1") == "Chapter 1"
        assert _normalise_chapter_number("10") == "Chapter 10"

    def test_normalise_word(self):
        """Test normalising word chapter numbers."""
        assert _normalise_chapter_number("one") == "Chapter 1"
        assert _normalise_chapter_number("two") == "Chapter 2"
        assert _normalise_chapter_number("ten") == "Chapter 10"

    def test_normalise_roman(self):
        """Test normalising roman numeral chapter numbers."""
        assert _normalise_chapter_number("I") == "Chapter 1"
        assert _normalise_chapter_number("II") == "Chapter 2"
        assert _normalise_chapter_number("V") == "Chapter 5"
        assert _normalise_chapter_number("X") == "Chapter 10"


class TestStructureMapping:
    """Test mapping chunks to structure."""

    def test_map_chunk_to_chapter(self):
        """Test mapping a chunk to its chapter."""
        text = """Chapter 1: Introduction

This is the introduction with some content that will be chunked.

Chapter 2: Methods

This is the methods section with different content."""

        structure = extract_structure_from_text(text)

        # Find position of text in Chapter 1
        chunk_start = text.find("introduction with some")
        chunk_end = chunk_start + 50

        mapping = map_text_to_structure(text, structure, chunk_start, chunk_end)

        assert mapping["chapter"] is not None
        assert "Chapter 1" in mapping["chapter"] or "Introduction" in str(mapping["chapter"])

    def test_map_chunk_to_subsection(self):
        """Test mapping a chunk to a subsection."""
        text = """Chapter 2: Methods

2.1 Data Collection

Interviews were conducted with participants.

2.2 Data Analysis

Thematic analysis was performed."""

        structure = extract_structure_from_text(text)

        # Find position in Data Analysis section
        chunk_start = text.find("Thematic analysis")
        chunk_end = chunk_start + 30

        mapping = map_text_to_structure(text, structure, chunk_start, chunk_end)

        assert mapping["chapter"] is not None
        assert mapping["section_title"] is not None or mapping["heading_path"] is not None


class TestEnhancedMetadataWithStructure:
    """Test that create_enhanced_metadata uses structure correctly."""

    def test_metadata_includes_chapter(self):
        """Test that enhanced metadata includes chapter when structure provided."""
        text = """Chapter 1: Introduction

This is a sample chunk of text from the introduction chapter."""

        chunk_text = "This is a sample chunk"
        structure = extract_structure_from_text(text)

        # Find chunk position
        chunk_start = text.find(chunk_text)
        chunk_end = chunk_start + len(chunk_text)

        metadata = create_enhanced_metadata(
            chunk_text=chunk_text,
            chunk_index=0,
            total_chunks=1,
            doc_id="test_doc",
            full_text=text,
            doc_type="academic_reference",
            document_structure=structure,
            chunk_char_start=chunk_start,
            chunk_char_end=chunk_end,
        )

        assert isinstance(metadata, EnhancedChunkMetadata)
        assert metadata.chapter is not None
        # Chapter should be "Chapter 1" or "Introduction"
        assert "Chapter 1" in metadata.chapter or metadata.chapter == "Introduction"

    def test_metadata_without_structure_still_works(self):
        """Test that metadata extraction works without structure (fallback mode)."""
        chunk_text = "This is a sample chunk"
        full_text = "Some full text\n\n" + chunk_text

        metadata = create_enhanced_metadata(
            chunk_text=chunk_text,
            chunk_index=0,
            total_chunks=1,
            doc_id="test_doc",
            full_text=full_text,
            doc_type="academic_reference",
            document_structure=None,  # No structure
            chunk_char_start=None,
            chunk_char_end=None,
        )

        assert isinstance(metadata, EnhancedChunkMetadata)
        # Should still return valid metadata even without structure


class TestIntegration:
    """Integration tests for end-to-end structure extraction."""

    def test_realistic_academic_document(self):
        """Test with a realistic academic document structure."""
        text = """Finding Your Cultural Leadership Lens: A Mixed Methods Study

Abstract

This study examines cultural leadership in Australian organisations.

Chapter 1: Introduction

1.1 Background

Indigenous leadership has been underrepresented in research.

1.2 Research Questions

This study asks three main research questions.

Chapter 2: Literature Review

2.1 Theoretical Framework

We draw on critical race theory and Indigenous standpoint theory.

2.2 Previous Research

Prior studies have shown...

Chapter 3: Methodology

3.1 Research Design

This study used a convergent mixed methods design.

3.2 Participants

Participants (N=45) were recruited from...

3.3 Data Collection

3.3.1 Interviews

Semi-structured interviews were conducted.

3.3.2 Surveys

Quantitative surveys measured...

Chapter 4: Results

4.1 Quantitative Findings

Statistical analysis revealed...

4.2 Qualitative Findings

Thematic analysis identified three major themes.

Chapter 5: Discussion

These findings contribute to understanding...

Chapter 6: Conclusion

In conclusion, this research demonstrates...

References

Aarons, G. A., Ehrhart, M. G., Torres, E. M., et al. (2017)..."""

        structure = extract_structure_from_text(text)

        # Should extract multiple chapters
        assert len(structure) >= 6

        # Should identify chapters
        chapters = [s for s in structure if s.get("level") == 0]
        assert len(chapters) >= 6

        # Should identify subsections
        subsections = [s for s in structure if s.get("level", 0) > 0]
        assert len(subsections) >= 5

        # Test mapping different chunks
        intro_pos = text.find("Indigenous leadership")
        intro_mapping = map_text_to_structure(text, structure, intro_pos, intro_pos + 50)
        assert intro_mapping["chapter"] is not None
        assert "Chapter 1" in intro_mapping["chapter"] or "Introduction" in str(
            intro_mapping["chapter"]
        )

        methods_pos = text.find("Semi-structured interviews")
        methods_mapping = map_text_to_structure(text, structure, methods_pos, methods_pos + 50)
        assert methods_mapping["chapter"] is not None
        assert "Chapter 3" in methods_mapping["chapter"] or "Methodology" in str(
            methods_mapping.get("heading_path", "")
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
