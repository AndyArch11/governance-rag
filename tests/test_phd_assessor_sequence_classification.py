"""Tests for sequence-based chapter classification and ToC parsing in PhDQualityAssessor."""

from unittest.mock import Mock

import numpy as np
import pytest

from scripts.ingest.academic.phd_assessor import PhDQualityAssessor


class TestSequenceBasedClassification:
    """Tests for sequence number-based section classification."""

    @pytest.fixture
    def assessor(self):
        """Create PhDQualityAssessor instance."""
        mock_collection = Mock()
        return PhDQualityAssessor(chunk_collection=mock_collection)

    def test_classify_section_type_with_sequence_pre_matter(self, assessor):
        """Sections before first numbered chapter are pre-matter."""
        # Abstract at sequence 10, Chapter 1 at 100, Chapter 5 at 500
        result = assessor._classify_section_type(
            label="Abstract", sequence_num=10, first_chapter_seq=100, last_chapter_seq=500
        )
        assert result == "pre-matter"

    def test_classify_section_type_with_sequence_main_matter(self, assessor):
        """Sections between first and last numbered chapters are main-matter."""
        # Chapter 3 at sequence 300, between Chapter 1 (100) and Chapter 5 (500)
        result = assessor._classify_section_type(
            label="Chapter 3", sequence_num=300, first_chapter_seq=100, last_chapter_seq=500
        )
        assert result == "main-matter"

    def test_classify_section_type_with_sequence_post_matter(self, assessor):
        """Sections after last numbered chapter are post-matter."""
        # References at sequence 600, after Chapter 5 at 500
        result = assessor._classify_section_type(
            label="References", sequence_num=600, first_chapter_seq=100, last_chapter_seq=500
        )
        assert result == "post-matter"

    def test_classify_section_type_without_sequence_falls_back(self, assessor):
        """Without valid sequence bounds, falls back to keyword matching."""
        # No numbered chapters found (first_chapter_seq = inf)
        result = assessor._classify_section_type(
            label="Introduction",
            sequence_num=100,
            first_chapter_seq=float("inf"),
            last_chapter_seq=float("-inf"),
        )
        # Should use keyword fallback → Introduction is main-matter
        assert result == "main-matter"

    def test_classify_section_type_keyword_fallback_pre_matter(self, assessor):
        """Keyword fallback correctly identifies pre-matter."""
        result = assessor._classify_section_type_by_keyword("Abstract")
        assert result == "pre-matter"

    def test_classify_section_type_keyword_fallback_post_matter(self, assessor):
        """Keyword fallback correctly identifies post-matter."""
        result = assessor._classify_section_type_by_keyword("References")
        assert result == "post-matter"

    def test_classify_section_type_keyword_fallback_main_matter(self, assessor):
        """Keyword fallback correctly identifies main-matter."""
        result = assessor._classify_section_type_by_keyword("Chapter 3")
        assert result == "main-matter"

    def test_classify_section_type_keyword_fallback_unknown(self, assessor):
        """Keyword fallback returns unknown for ambiguous labels."""
        result = assessor._classify_section_type_by_keyword("Random Section")
        assert result == "unknown"


class TestToCParsing:
    """Tests for Table of Contents parsing."""

    @pytest.fixture
    def assessor(self):
        """Create PhDQualityAssessor instance."""
        mock_collection = Mock()
        return PhDQualityAssessor(chunk_collection=mock_collection)

    def test_parse_toc_structure_basic(self, assessor):
        """Parse basic ToC with chapter entries."""
        toc_text = """
        Table of Contents
        
        Chapter 1: Introduction ........................... 1
        Chapter 2: Literature Review ..................... 15
        Chapter 3: Methodology ........................... 35
        Chapter 4: Results ............................... 52
        Chapter 5: Discussion ............................ 78
        """

        chunks_data = {
            "documents": [toc_text],
            "metadatas": [{"section_title": "Table of Contents"}],
        }

        toc_structure = assessor._parse_toc_structure(chunks_data)

        # Should extract chapter names and page numbers
        assert len(toc_structure) > 0
        # Check for some expected chapters (exact matching depends on regex)
        chapter_names = list(toc_structure.keys())
        assert any("Chapter 1" in name for name in chapter_names)

    def test_parse_toc_structure_no_toc(self, assessor):
        """Returns empty dict when no ToC chunks present."""
        chunks_data = {
            "documents": ["Regular content without ToC patterns"],
            "metadatas": [{"section_title": "Introduction"}],
        }

        toc_structure = assessor._parse_toc_structure(chunks_data)
        assert toc_structure == {}

    def test_parse_toc_structure_with_section_headings(self, assessor):
        """Parse ToC with non-numbered section headings."""
        toc_text = """
        Contents
        
        Abstract .......................................... 1
        Acknowledgements .................................. 3
        Chapter 1: Introduction ........................... 5
        References ....................................... 95
        Appendix A ....................................... 100
        """

        chunks_data = {"documents": [toc_text], "metadatas": [{"section_title": "Contents"}]}

        toc_structure = assessor._parse_toc_structure(chunks_data)

        # Should extract at least the chapter entry
        assert len(toc_structure) >= 1

    def test_parse_toc_structure_with_roman_numerals_and_dots(self, assessor):
        """Parse ToC lines with roman numerals and chapter number dots."""
        toc_text = "\n".join(
            [
                "Table of Contents",
                "",
                "i. Statement of Authentication ............................. 2",
                "ii. Acknowledgements ....................................... 3",
                "iii. Abstract .............................................. 5",
                "iv. Abbreviations .......................................... 8",
                "v. Glossary ................................................ 10",
                "vi. Table of Contents ...................................... 13",
                "vii. List of Tables ......................................... 26",
                "viii. List of Figures ....................................... 28",
                "Chapter 1. Perceived Inclusion and Exclusion ............... 29",
                "Chapter 2. Developing Inclusive Work Environments .......... 59",
                "Statement of Contributions ................................ 366",
                "References ................................................ 368",
                "Appendix A: Project Outputs ............................... 490",
            ]
        )

        chunks_data = {
            "documents": [toc_text],
            "metadatas": [{"section_title": "Table of Contents"}],
        }

        toc_structure = assessor._parse_toc_structure(chunks_data)

        assert "Statement of Authentication" in toc_structure
        assert "Acknowledgements" in toc_structure
        assert "Abstract" in toc_structure
        assert "Abbreviations" in toc_structure
        assert "Glossary" in toc_structure
        assert "List of Tables" in toc_structure
        assert "List of Figures" in toc_structure
        assert any("Chapter 1" in name for name in toc_structure)
        assert any("Chapter 2" in name for name in toc_structure)
        assert "Statement of Contributions" in toc_structure
        assert "References" in toc_structure
        assert "Appendix A: Project Outputs" in toc_structure


class TestExtractChaptersSequenceBased:
    """Tests for _extract_chapters with sequence-based classification."""

    @pytest.fixture
    def assessor(self):
        """Create PhDQualityAssessor instance."""
        mock_collection = Mock()
        return PhDQualityAssessor(chunk_collection=mock_collection)

    def test_extract_chapters_sequence_classification(self, assessor):
        """Chapters are classified using sequence number positions."""
        # Create chunks with sequence numbers
        chunks_data = {
            "ids": ["1", "2", "3", "4", "5"],
            "documents": [
                "Abstract content",
                "Chapter 1: Introduction content",
                "Chapter 2: Methods content",
                "Chapter 3: Results content",
                "References section",
            ],
            "metadatas": [
                {"chunk_type": "parent", "chapter": "Abstract", "sequence_number": 10},
                {"chunk_type": "parent", "chapter": "Chapter 1", "sequence_number": 100},
                {"chunk_type": "parent", "chapter": "Chapter 2", "sequence_number": 200},
                {"chunk_type": "parent", "chapter": "Chapter 3", "sequence_number": 300},
                {"chunk_type": "parent", "chapter": "References", "sequence_number": 400},
            ],
            "embeddings": [
                np.random.rand(1024),
                np.random.rand(1024),
                np.random.rand(1024),
                np.random.rand(1024),
                np.random.rand(1024),
            ],
        }

        chapters = assessor._extract_chapters(chunks_data)

        # Should have all chapters
        assert len(chapters) == 5

        # Check classification
        chapter_dict = {ch["name"]: ch["section_type"] for ch in chapters}

        # Abstract (seq 10) should be pre-matter (before Chapter 1 at seq 100)
        assert chapter_dict.get("Abstract") == "pre-matter"

        # Chapters 1-3 should be main-matter
        assert chapter_dict.get("Chapter 1") == "main-matter"
        assert chapter_dict.get("Chapter 2") == "main-matter"
        assert chapter_dict.get("Chapter 3") == "main-matter"

        # References (seq 400, after Chapter 3 at seq 300) should be post-matter
        assert chapter_dict.get("References") == "post-matter"

    def test_extract_chapters_preserves_document_order(self, assessor):
        """Chapters are returned in document order (sequence number order)."""
        chunks_data = {
            "ids": ["1", "2", "3"],
            "documents": [
                "Chapter 3 content",
                "Chapter 1 content",
                "Chapter 2 content",
            ],
            "metadatas": [
                {"chunk_type": "parent", "chapter": "Chapter 3", "sequence_number": 300},
                {"chunk_type": "parent", "chapter": "Chapter 1", "sequence_number": 100},
                {"chunk_type": "parent", "chapter": "Chapter 2", "sequence_number": 200},
            ],
            "embeddings": [
                np.random.rand(1024),
                np.random.rand(1024),
                np.random.rand(1024),
            ],
        }

        chapters = assessor._extract_chapters(chunks_data)

        # Should be ordered by sequence number
        chapter_names = [ch["name"] for ch in chapters]
        assert chapter_names == ["Chapter 1", "Chapter 2", "Chapter 3"]

    def test_extract_chapters_filters_unknown_sections(self, assessor):
        """Sections classified as 'unknown' are filtered out."""
        chunks_data = {
            "ids": ["1", "2", "3"],
            "documents": [
                "Chapter 1 content",
                "Invalid Header N Mean SD",  # Should be filtered as invalid
                "Chapter 2 content",
            ],
            "metadatas": [
                {"chunk_type": "parent", "chapter": "Chapter 1", "sequence_number": 100},
                {"chunk_type": "parent", "chapter": "N Mean SD", "sequence_number": 150},
                {"chunk_type": "parent", "chapter": "Chapter 2", "sequence_number": 200},
            ],
            "embeddings": [
                np.random.rand(1024),
                np.random.rand(1024),
                np.random.rand(1024),
            ],
        }

        chapters = assessor._extract_chapters(chunks_data)

        # Should only have valid chapters (invalid label filtered out)
        chapter_names = [ch["name"] for ch in chapters]
        assert "N Mean SD" not in chapter_names
        assert len(chapters) == 2

    def test_extract_chapters_without_numbered_chapters_uses_keywords(self, assessor):
        """Without numbered chapters, falls back to keyword classification."""
        chunks_data = {
            "ids": ["1", "2", "3"],
            "documents": [
                "Abstract content",
                "Introduction content",
                "References content",
            ],
            "metadatas": [
                {"chunk_type": "parent", "chapter": "Abstract", "sequence_number": 10},
                {"chunk_type": "parent", "chapter": "Introduction", "sequence_number": 100},
                {"chunk_type": "parent", "chapter": "References", "sequence_number": 200},
            ],
            "embeddings": [
                np.random.rand(1024),
                np.random.rand(1024),
                np.random.rand(1024),
            ],
        }

        chapters = assessor._extract_chapters(chunks_data)

        # Should fall back to keyword classification (no numbered chapters)
        chapter_dict = {ch["name"]: ch["section_type"] for ch in chapters}

        # Abstract should be pre-matter (keyword)
        assert chapter_dict.get("Abstract") == "pre-matter"

        # Introduction should be main-matter (keyword)
        assert chapter_dict.get("Introduction") == "main-matter"

        # References should be post-matter (keyword)
        assert chapter_dict.get("References") == "post-matter"

    def test_extract_chapters_uses_toc_order_when_sequence_missing(self, assessor):
        """Uses ToC order when sequence numbers are missing or invalid."""
        toc_text = """
        Table of Contents

        Chapter 1: Introduction ........................... 1
        Chapter 2: Literature Review ..................... 12
        Chapter 6: Methodology ........................... 48
        Chapter 7: Results ................................ 72
        Appendix .......................................... 120
        """

        chunks_data = {
            "documents": [
                toc_text,
                "Chapter 7 content",
                "Appendix content",
                "Chapter 1 content",
                "Chapter 6 content",
                "Chapter 2 content",
            ],
            "metadatas": [
                {"section_title": "Table of Contents"},
                {"chunk_type": "parent", "chapter": "Chapter 7"},
                {"chunk_type": "parent", "chapter": "Appendix"},
                {"chunk_type": "parent", "chapter": "Chapter 1"},
                {"chunk_type": "parent", "chapter": "Chapter 6"},
                {"chunk_type": "parent", "chapter": "Chapter 2"},
            ],
            "embeddings": [
                np.random.rand(1024),
                np.random.rand(1024),
                np.random.rand(1024),
                np.random.rand(1024),
                np.random.rand(1024),
                np.random.rand(1024),
            ],
        }

        chapters = assessor._extract_chapters(chunks_data)
        chapter_names = [ch["name"] for ch in chapters]

        assert chapter_names == [
            "Chapter 1: Introduction",
            "Chapter 2: Literature Review",
            "Chapter 6: Methodology",
            "Chapter 7: Results",
            "Appendix",
        ]

    def test_extract_chapters_toc_fallback_classification(self, assessor):
        """ToC-based ordering drives pre/main/post classification when sequence is missing."""
        toc_text = "\n".join(
            [
                "Contents",
                "",
                "Abstract .......................................... 1",
                "Chapter 1: Introduction ........................... 5",
                "Chapter 2: Methods ................................ 30",
                "References ........................................ 90",
            ]
        )

        chunks_data = {
            "documents": [
                toc_text,
                "Abstract content",
                "Chapter 2 content",
                "References content",
                "Chapter 1 content",
            ],
            "metadatas": [
                {"section_title": "Contents"},
                {"chunk_type": "parent", "chapter": "Abstract"},
                {"chunk_type": "parent", "chapter": "Chapter 2"},
                {"chunk_type": "parent", "chapter": "References"},
                {"chunk_type": "parent", "chapter": "Chapter 1"},
            ],
            "embeddings": [
                np.random.rand(1024),
                np.random.rand(1024),
                np.random.rand(1024),
                np.random.rand(1024),
                np.random.rand(1024),
            ],
        }

        chapters = assessor._extract_chapters(chunks_data)
        chapter_dict = {ch["name"]: ch["section_type"] for ch in chapters}

        assert chapter_dict.get("Abstract") == "pre-matter"
        assert chapter_dict.get("Chapter 1: Introduction") == "main-matter"
        assert chapter_dict.get("Chapter 2: Methods") == "main-matter"
        assert chapter_dict.get("References") == "post-matter"

    def test_extract_chapters_toc_fuzzy_match(self, assessor):
        """ToC fallback matches chapter labels with differing titles."""
        toc_text = "\n".join(
            [
                "Contents",
                "",
                "Chapter 1: Introduction ........................... 4",
                "Chapter 2: Methods ................................ 18",
                "Chapter 3: Results ................................ 52",
            ]
        )

        chunks_data = {
            "documents": [
                toc_text,
                "Chapter 1 content",
                "Chapter 2 content",
                "Chapter 3 content",
            ],
            "metadatas": [
                {"section_title": "Contents"},
                {"chunk_type": "parent", "chapter": "Chapter 1"},
                {"chunk_type": "parent", "chapter": "Chapter 2"},
                {"chunk_type": "parent", "chapter": "Chapter 3"},
            ],
            "embeddings": [
                np.random.rand(1024),
                np.random.rand(1024),
                np.random.rand(1024),
                np.random.rand(1024),
            ],
        }

        chapters = assessor._extract_chapters(chunks_data)
        chapter_names = [ch["name"] for ch in chapters]

        assert chapter_names == [
            "Chapter 1: Introduction",
            "Chapter 2: Methods",
            "Chapter 3: Results",
        ]
