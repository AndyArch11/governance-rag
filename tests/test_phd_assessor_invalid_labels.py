"""Tests for invalid label filtering and stable sorting fixes."""

from unittest.mock import Mock

import numpy as np
import pytest

from scripts.ingest.academic.phd_assessor import PhDQualityAssessor


class TestInvalidLabelFiltering:
    """Tests that invalid labels like 'N Mean SD SE 95%' are properly filtered."""

    @pytest.fixture
    def assessor(self):
        """Create PhDQualityAssessor instance."""
        mock_collection = Mock()
        return PhDQualityAssessor(chunk_collection=mock_collection)

    def test_table_header_label_filtered(self, assessor):
        """Statistical table headers should be classified as unknown and filtered."""
        # "N Mean SD SE 95%" is a table header, not a chapter
        result = assessor._classify_section_type(
            label="N Mean SD SE 95%", sequence_num=250, first_chapter_seq=100, last_chapter_seq=500
        )
        assert result == "unknown"

    def test_statistical_notation_label_filtered(self, assessor):
        """Statistical notation should be rejected as invalid chapter label."""
        invalid_labels = [
            "N Mean SD SE 95%",
            "Mean SD SE",
            "N   Mean   SD   SE",
            "% CI 95%",
        ]

        for label in invalid_labels:
            result = assessor._classify_section_type(
                label=label, sequence_num=200, first_chapter_seq=100, last_chapter_seq=500
            )
            assert result == "unknown", f"Label '{label}' should be filtered as unknown"

    def test_extract_chapters_filters_invalid_labels(self, assessor):
        """_extract_chapters should filter out invalid labels like table headers."""
        chunks_data = {
            "ids": ["1", "2", "3", "4"],
            "documents": [
                "Chapter 1 content",
                "N Mean SD SE 95% table data",  # Invalid label
                "Chapter 2 content",
                "Chapter 3 content",
            ],
            "metadatas": [
                {"chunk_type": "parent", "chapter": "Chapter 1", "sequence_number": 100},
                {"chunk_type": "parent", "chapter": "N Mean SD SE 95%", "sequence_number": 150},
                {"chunk_type": "parent", "chapter": "Chapter 2", "sequence_number": 200},
                {"chunk_type": "parent", "chapter": "Chapter 3", "sequence_number": 300},
            ],
            "embeddings": [
                np.random.rand(1024),
                np.random.rand(1024),
                np.random.rand(1024),
                np.random.rand(1024),
            ],
        }

        chapters = assessor._extract_chapters(chunks_data)

        # Should have 3 valid chapters (invalid one filtered)
        chapter_names = [ch["name"] for ch in chapters]
        assert "N Mean SD SE 95%" not in chapter_names
        assert len(chapters) == 3
        assert "Chapter 1" in chapter_names
        assert "Chapter 2" in chapter_names
        assert "Chapter 3" in chapter_names


class TestStableSorting:
    """Tests that sections with identical sequence numbers maintain insertion order."""

    @pytest.fixture
    def assessor(self):
        """Create PhDQualityAssessor instance."""
        mock_collection = Mock()
        return PhDQualityAssessor(chunk_collection=mock_collection)

    def test_sections_with_same_sequence_preserve_insertion_order(self, assessor):
        """Sections with identical sequence numbers should maintain insertion order."""
        # All sections have same sequence number (inf), should preserve order
        chunks_data = {
            "ids": ["1", "2", "3", "4", "5"],
            "documents": [
                "Abstract content",
                "Acknowledgements content",
                "Introduction content",
                "Conclusion content",
                "References content",
            ],
            "metadatas": [
                {"chunk_type": "parent", "chapter": "Abstract", "sequence_number": float("inf")},
                {
                    "chunk_type": "parent",
                    "chapter": "Acknowledgements",
                    "sequence_number": float("inf"),
                },
                {
                    "chunk_type": "parent",
                    "chapter": "Introduction",
                    "sequence_number": float("inf"),
                },
                {"chunk_type": "parent", "chapter": "Conclusion", "sequence_number": float("inf")},
                {"chunk_type": "parent", "chapter": "References", "sequence_number": float("inf")},
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
        chapter_names = [ch["name"] for ch in chapters]

        # Should maintain insertion order, NOT alphabetical
        expected_order = [
            "Abstract",
            "Acknowledgements",
            "Introduction",
            "Conclusion",
            "References",
        ]
        assert (
            chapter_names == expected_order
        ), f"Expected insertion order {expected_order}, got {chapter_names}"

    def test_mixed_sequence_numbers_with_stable_secondary_sort(self, assessor):
        """Mix of valid and inf sequence numbers should sort correctly."""
        chunks_data = {
            "ids": ["1", "2", "3", "4", "5", "6"],
            "documents": [
                "Abstract content",
                "Chapter 1 content",
                "Chapter 2 content",
                "Acknowledgements content",  # Same seq as Abstract
                "Chapter 3 content",
                "References content",
            ],
            "metadatas": [
                {"chunk_type": "parent", "chapter": "Abstract", "sequence_number": 10},
                {"chunk_type": "parent", "chapter": "Chapter 1", "sequence_number": 100},
                {"chunk_type": "parent", "chapter": "Chapter 2", "sequence_number": 200},
                {
                    "chunk_type": "parent",
                    "chapter": "Acknowledgements",
                    "sequence_number": 10,
                },  # Same as Abstract
                {"chunk_type": "parent", "chapter": "Chapter 3", "sequence_number": 300},
                {"chunk_type": "parent", "chapter": "References", "sequence_number": 400},
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

        # Abstract and Acknowledgements both have seq 10, should maintain insertion order
        # Abstract appears first in chunks, so should come before Acknowledgements
        abstract_idx = chapter_names.index("Abstract")
        ack_idx = chapter_names.index("Acknowledgements")
        assert (
            abstract_idx < ack_idx
        ), f"Abstract (seq 10, inserted first) should come before Acknowledgements (seq 10, inserted later)"

        # Verify overall order respects sequence numbers
        chapter1_idx = chapter_names.index("Chapter 1")
        assert (
            abstract_idx < chapter1_idx
        ), "Abstract (seq 10) should come before Chapter 1 (seq 100)"


class TestChapterSkipping:
    """Tests that chapter gaps are detected correctly."""

    @pytest.fixture
    def assessor(self):
        """Create PhDQualityAssessor instance."""
        mock_collection = Mock()
        return PhDQualityAssessor(chunk_collection=mock_collection)

    def test_chapter_ordering_with_gap(self, assessor):
        """Chapters should be in sequence number order even if numbered chapters have gaps."""
        chunks_data = {
            "ids": ["1", "2", "3", "4", "5"],
            "documents": [
                "Chapter 1 content",
                "Chapter 2 content",
                "Chapter 5 content",  # Gap: Chapter 3, 4 missing
                "Chapter 6 content",  # Out of sequence in data
                "Chapter 7 content",
            ],
            "metadatas": [
                {"chunk_type": "parent", "chapter": "Chapter 1", "sequence_number": 100},
                {"chunk_type": "parent", "chapter": "Chapter 2", "sequence_number": 200},
                {"chunk_type": "parent", "chapter": "Chapter 5", "sequence_number": 500},
                {
                    "chunk_type": "parent",
                    "chapter": "Chapter 6",
                    "sequence_number": 600,
                },  # Correct seq
                {"chunk_type": "parent", "chapter": "Chapter 7", "sequence_number": 700},
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
        chapter_names = [ch["name"] for ch in chapters]

        # Should be in sequence number order
        expected_order = ["Chapter 1", "Chapter 2", "Chapter 5", "Chapter 6", "Chapter 7"]
        assert (
            chapter_names == expected_order
        ), f"Chapters should be in sequence order {expected_order}, got {chapter_names}"
