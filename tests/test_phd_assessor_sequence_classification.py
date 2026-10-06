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

    def test_extract_chapters_prefers_source_sequence_over_conflicting_toc(self, assessor):
        """Source sequence metadata remains canonical when the ToC is inconsistent."""
        chunks_data = {
            "ids": ["toc", "1", "2"],
            "documents": [
                (
                    "Table of Contents\n\nChapter 2: Methods ................... 1\n"
                    "Chapter 1: Introduction ............... 20"
                ),
                "Chapter 1: Introduction content",
                "Chapter 2: Methods content",
            ],
            "metadatas": [
                {
                    "chunk_type": "parent",
                    "section_title": "Table of Contents",
                    "sequence_number": 0,
                },
                {"chunk_type": "parent", "chapter": "Chapter 1", "sequence_number": 100},
                {"chunk_type": "parent", "chapter": "Chapter 2", "sequence_number": 200},
            ],
            "embeddings": [np.random.rand(1024) for _ in range(3)],
        }

        chapters = assessor._extract_chapters(chunks_data)

        assert [chapter["name"] for chapter in chapters] == [
            "Chapter 1: Introduction",
            "Chapter 2: Methods",
        ]

    def test_section_embeddings_follow_source_sequence(self, assessor):
        """Argument-flow sections follow source sequence rather than collection order."""
        chunks_data = {
            "ids": ["1", "2", "3"],
            "documents": ["Results", "Introduction", "Methods"],
            "metadatas": [
                {"chunk_type": "parent", "section_title": "Results", "sequence_number": 300},
                {"chunk_type": "parent", "section_title": "Introduction", "sequence_number": 100},
                {"chunk_type": "parent", "section_title": "Methods", "sequence_number": 200},
            ],
            "embeddings": [np.random.rand(1024) for _ in range(3)],
        }

        sections = assessor._extract_section_embeddings(chunks_data)

        assert [section["name"] for section in sections] == ["Introduction", "Methods", "Results"]

    def test_extract_chapters_orders_double_digit_chapters_by_source_sequence(self, assessor):
        """Chapters 1, 2, 3, and 10 retain numeric source order rather than lexical order."""
        chapters = ["Chapter 10", "Chapter 3", "Chapter 1", "Chapter 2"]
        chunks_data = {
            "ids": [str(index) for index in range(4)],
            "documents": [f"{chapter} content" for chapter in chapters],
            "metadatas": [
                {"chunk_type": "parent", "chapter": chapter, "sequence_number": sequence}
                for chapter, sequence in zip(chapters, [1000, 300, 100, 200])
            ],
            "embeddings": [np.random.rand(1024) for _ in chapters],
        }

        extracted_chapters = assessor._extract_chapters(chunks_data)

        assert [chapter["name"] for chapter in extracted_chapters] == [
            "Chapter 1",
            "Chapter 2",
            "Chapter 3",
            "Chapter 10",
        ]

    def test_extract_chapters_preserves_full_document_source_order(self, assessor):
        """Front matter, chapters, and post matter retain the source document sequence."""
        source_items = [
            ("Abstract", 10),
            ("Acknowledgements", 20),
            ("Chapter 1", 100),
            ("Chapter 2", 200),
            ("Chapter 3", 300),
            ("Chapter 10", 1000),
            ("References", 1100),
            ("Appendix A", 1200),
        ]
        collection_items = [source_items[index] for index in [6, 3, 0, 7, 4, 1, 5, 2]]
        chunks_data = {
            "ids": [str(index) for index in range(len(collection_items))],
            "documents": [f"{label} content" for label, _ in collection_items],
            "metadatas": [
                {"chunk_type": "parent", "chapter": label, "sequence_number": sequence}
                for label, sequence in collection_items
            ],
            "embeddings": [np.random.rand(1024) for _ in collection_items],
        }

        extracted_chapters = assessor._extract_chapters(chunks_data)
        chapter_text = assessor._group_text_by_chapter(chunks_data)

        expected_order = [label for label, _ in source_items]
        assert [chapter["name"] for chapter in extracted_chapters] == expected_order
        assert list(chapter_text) == expected_order
        assert [chapter["section_type"] for chapter in extracted_chapters] == [
            "pre-matter",
            "pre-matter",
            "main-matter",
            "main-matter",
            "main-matter",
            "main-matter",
            "post-matter",
            "post-matter",
        ]

    def test_extract_chapters_uses_leading_headings_when_metadata_is_missing(self, assessor):
        """Heading boundaries prevent body references from being treated as chapters."""
        chunks_data = {
            "ids": ["1", "2", "3", "4", "5", "6"],
            "documents": [
                "## Abstract",
                "This thesis refers to Chapter 4 when outlining its later methodology.",
                "## Acknowledgements",
                "Thanks to the supervisors and participants who supported this thesis.",
                "## Chapter 1: Introduction",
                "This chapter establishes the research context.",
            ],
            "metadatas": [{"chunk_type": "child", "sequence_number": index} for index in range(6)],
            "embeddings": [np.random.rand(1024) for _ in range(6)],
        }

        chapters = assessor._extract_chapters(chunks_data)

        assert [chapter["name"] for chapter in chapters] == [
            "Abstract",
            "Acknowledgements",
            "Chapter 1",
        ]

    def test_extract_chapters_uses_children_when_parent_metadata_is_missing(self, assessor):
        """Unlabelled parent chunks do not interleave with child source sequence."""
        chunks_data = {
            "ids": ["parent-0", "parent-1", "child-0", "child-1", "child-2", "child-3"],
            "documents": [
                "Parent text mentioning Chapter 2.",
                "Parent text mentioning Chapter 1.",
                "## Abstract",
                "Abstract text.",
                "## Chapter 1: Introduction",
                "Introduction text.",
            ],
            "metadatas": [
                {"chunk_type": "parent", "sequence_number": 0},
                {"chunk_type": "parent", "sequence_number": 1},
                {"chunk_type": "child", "sequence_number": 0},
                {"chunk_type": "child", "sequence_number": 1},
                {"chunk_type": "child", "sequence_number": 2},
                {"chunk_type": "child", "sequence_number": 3},
            ],
            "embeddings": [np.random.rand(1024) for _ in range(6)],
        }

        chapters = assessor._extract_chapters(chunks_data)

        assert [chapter["name"] for chapter in chapters] == ["Abstract", "Chapter 1"]

    def test_structure_reports_conflicting_chapter_source_order(self, assessor):
        """Numbered chapters with contradictory source sequence are visible for review."""
        chunks_data = {
            "ids": ["1", "2"],
            "documents": ["Chapter 2: Methods content", "Chapter 1: Introduction content"],
            "metadatas": [
                {"chunk_type": "parent", "chapter": "Chapter 2", "sequence_number": 100},
                {"chunk_type": "parent", "chapter": "Chapter 1", "sequence_number": 200},
            ],
            "embeddings": [np.random.rand(1024), np.random.rand(1024)],
        }

        structure = assessor.analyse_structure(chunks_data)

        assert structure.chapter_order == ["Chapter 2", "Chapter 1"]
        assert structure.chapter_order_issues == [
            "Source sequence orders numbered chapters as Chapter 2 before Chapter 1."
        ]
        assert any(
            flag.title == "Chapter Source Order Requires Review" for flag in structure.red_flags
        )

    def test_structure_reports_malformed_and_duplicate_sequence_metadata(self, assessor):
        """Malformed and duplicate chapter sequence values are retained as review issues."""
        chunks_data = {
            "ids": ["1", "2", "3"],
            "documents": [
                "Chapter 1: Introduction content",
                "Chapter 2: Methods content",
                "Chapter 3: Results content",
            ],
            "metadatas": [
                {"chunk_type": "parent", "chapter": "Chapter 1", "sequence_number": "first"},
                {"chunk_type": "parent", "chapter": "Chapter 2", "sequence_number": 100},
                {"chunk_type": "parent", "chapter": "Chapter 3", "sequence_number": 100},
            ],
            "embeddings": [np.random.rand(1024) for _ in range(3)],
        }

        structure = assessor.analyse_structure(chunks_data)

        assert any(
            "Malformed sequence metadata 'first'" in issue
            for issue in structure.chapter_order_issues
        )
        assert any("Duplicate sequence 100" in issue for issue in structure.chapter_order_issues)

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
