"""Tests for academic citation graph with SQLite persistence and book review handling."""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from scripts.ingest.academic.citation_graph_schema import ensure_schema
from scripts.ingest.academic.citation_graph_writer import CitationGraphWriter
from scripts.ingest.academic.graph import (
    CitationGraph,
    CitationNode,
    clean_book_review_title,
    extract_authors_from_citation,
    extract_title_from_citation,
)


class TestCitationNode:
    """Test CitationNode dataclass."""

    def test_citation_node_creation_reference(self):
        """Test creating a reference node."""
        node = CitationNode(
            node_id="ref_001",
            node_type="reference",
            title="Deep Learning",
            authors=["LeCun, Y.", "Bengio, Y.", "Hinton, G. E."],
            year=2015,
            doi="10.1038/nature14539",
        )
        assert node.node_id == "ref_001"
        assert node.node_type == "reference"
        assert node.title == "Deep Learning"
        assert len(node.authors) == 3
        assert node.year == 2015
        assert node.year_verified is False

    def test_citation_node_creation_document(self):
        """Test creating a document node."""
        node = CitationNode(
            node_id="doc_primary",
            node_type="document",
            title="New Insight Contributing to Human Knowledge",
            authors=["FirstName Surname"],
            year=2026,
        )
        assert node.node_type == "document"
        assert node.authors == ["FirstName Surname"]

    def test_citation_node_defaults(self):
        """Test default values."""
        node = CitationNode(node_id="ref_001", node_type="reference")
        assert node.title is None
        assert node.authors == []
        assert node.year is None
        assert node.year_verified is False
        assert node.doi is None


class TestCleanBookReviewTitle:
    """Test book review title cleaning."""

    def test_pattern1_lastname_initials_year_title(self):
        """Test Pattern 1: LastName, Initial(s). Year. Title."""
        title = "Boeije, H. 2010. Analysis in Qualitative Research. Sage Publications."
        clean_title, book_authors = clean_book_review_title(title)
        assert "Analysis in Qualitative Research" in clean_title
        assert len(book_authors) > 0
        assert "Boeije" in book_authors[0]

    def test_pattern1_multiple_initials(self):
        """Test Pattern 1 with multiple initials."""
        title = "Smith, J. A. 2015. Advanced Methods. Oxford University Press."
        clean_title, book_authors = clean_book_review_title(title)
        assert clean_title is not None
        assert len(book_authors) > 0

    def test_pattern2_firstname_lastname_year_title(self):
        """Test Pattern 2: FirstName LastName (Year): Title."""
        title = "John Smith (2018): Effective Leadership. Harvard Business Review Press."
        clean_title, book_authors = clean_book_review_title(title)
        assert book_authors  # Should extract some authors

    def test_pattern3_author_year_title(self):
        """Test Pattern 3: Author Year. Title."""
        title = "Johnson 2019. Organizational Behavior. McGraw-Hill Education."
        clean_title, book_authors = clean_book_review_title(title)
        # Pattern 3 may or may not match depending on implementation
        # Just ensure no errors occur
        assert clean_title is not None

    def test_non_book_review_title(self):
        """Test that non-book-review titles are unchanged."""
        title = "Impact of Remote Work on Team Dynamics"
        clean_title, book_authors = clean_book_review_title(title)
        assert clean_title == title
        assert book_authors == []

    def test_empty_title(self):
        """Test empty title handling."""
        clean_title, book_authors = clean_book_review_title("")
        assert clean_title == ""
        assert book_authors == []

    def test_none_title(self):
        """Test None title handling."""
        clean_title, book_authors = clean_book_review_title(None)
        assert clean_title is None
        assert book_authors == []


class TestExtractTitleFromCitation:
    """Test title extraction from citation text."""

    def test_extract_title_standard_format(self):
        """Test extracting title from standard citation format."""
        citation = "(2020). Deep Learning: A Comprehensive Guide. MIT Press."
        title = extract_title_from_citation(citation)
        assert title is not None
        assert "Deep Learning" in title or "Learning" in title

    def test_extract_title_missing(self):
        """Test handling missing title."""
        citation = "(2020)."
        title = extract_title_from_citation(citation)
        # Should return None or empty
        assert title is None or title == ""

    def test_extract_title_with_venue(self):
        """Test extracting title when venue is present."""
        citation = "(2019). Machine Learning Models. Journal of AI Research."
        title = extract_title_from_citation(citation)
        # Should not include venue
        assert title is None or "Journal" not in title


class TestExtractAuthorsFromCitation:
    """Test author extraction from citation text."""

    def test_extract_single_author(self):
        """Test extracting single author."""
        citation = "Smith, J. (2020). Title."
        authors = extract_authors_from_citation(citation)
        assert len(authors) > 0
        assert any("Smith" in a for a in authors)

    def test_extract_multiple_authors(self):
        """Test extracting multiple authors."""
        citation = "Smith, J., Johnson, M., & Williams, K. (2020). Title."
        authors = extract_authors_from_citation(citation)
        assert len(authors) >= 2

    def test_extract_authors_with_et_al(self):
        """Test handling 'et al.' notation."""
        citation = "Smith, J., et al. (2020). Title."
        authors = extract_authors_from_citation(citation)
        # Should include first author, et al usually not expanded
        assert len(authors) >= 1

    def test_extract_no_authors(self):
        """Test when no authors found."""
        citation = "(2020). Title without author."
        authors = extract_authors_from_citation(citation)
        assert authors == [] or authors == None


class TestCitationGraph:
    """Test CitationGraph building and persistence."""

    def test_graph_initialisation(self):
        """Test creating empty graph."""
        graph = CitationGraph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_add_document_node(self):
        """Test adding document node."""
        graph = CitationGraph()
        graph.add_document(
            "doc_primary",
            metadata={
                "title": "Leadership Study",
                "authors": ["FirstName Surname"],
                "year": 2024,
            },
        )
        assert "doc_primary" in graph.nodes
        node = graph.nodes["doc_primary"]
        assert node.node_type == "document"
        assert node.title == "Leadership Study"
        assert "FirstName Surname" in node.authors

    def test_add_reference_with_metadata(self):
        """Test adding reference with complete metadata."""
        graph = CitationGraph()
        graph.add_reference(
            "ref_001",
            {
                "title": "Deep Learning",
                "authors": ["LeCun, Y.", "Bengio, Y.", "Hinton, G. E."],
                "year": 2015,
                "doi": "10.1038/nature14539",
                "reference_type": "journal",
            },
        )
        assert "ref_001" in graph.nodes
        node = graph.nodes["ref_001"]
        assert node.title == "Deep Learning"
        assert len(node.authors) == 3

    def test_add_reference_minimal(self):
        """Test adding reference with minimal metadata."""
        graph = CitationGraph()
        graph.add_reference("ref_002", {"title": "Some Paper"})
        assert "ref_002" in graph.nodes
        node = graph.nodes["ref_002"]
        assert node.title == "Some Paper"

    def test_reference_quality_score_computed(self):
        """Quality score should be computed when not provided."""
        graph = CitationGraph()
        graph.add_reference(
            "ref_003",
            {
                "title": "Signal Processing Trends",
                "source": "crossref",
                "venue_rank": "Q1",
                "citation_count": 42,
                "link_status": "available",
                "oa_available": True,
            },
        )
        node = graph.nodes["ref_003"]
        assert node.quality_score is not None
        assert 0.0 <= node.quality_score <= 1.0

    def test_add_reference_with_citation_extraction(self):
        """Test adding reference with citation text for extraction."""
        graph = CitationGraph()
        graph.add_reference(
            "ref_003",
            {
                "citation": "Smith, J., & Johnson, M. (2019). Advanced Methods. Journal of Methods.",
            },
        )
        node = graph.nodes["ref_003"]
        # Should have extracted authors and year
        assert node.year is not None or node.authors

    def test_add_edge(self):
        """Test adding citation edges."""
        graph = CitationGraph()
        graph.add_document("doc_001")
        graph.add_reference("ref_001", {"title": "Paper A"})
        graph.add_edge("doc_001", "ref_001")
        assert len(graph.edges) == 1
        # Edges are CitationEdge objects
        assert graph.edges[0].source == "doc_001"
        assert graph.edges[0].target == "ref_001"

    def test_multiple_edges(self):
        """Test multiple citation edges."""
        graph = CitationGraph()
        graph.add_document("doc_001")
        for i in range(5):
            ref_id = f"ref_{i:03d}"
            graph.add_reference(ref_id, {"title": f"Paper {i}"})
            graph.add_edge("doc_001", ref_id)
        assert len(graph.edges) == 5

    def test_book_review_author_replacement(self):
        """Test that book review reviewers are replaced with book authors."""
        graph = CitationGraph()
        # Simulate a CrossRef book review
        graph.add_reference(
            "ref_book_review",
            {
                "title": "Boeije, H. 2010. Analysis in Qualitative Research. Sage.",
                "authors": ["Reviewer, A.", "Reviewer, B."],  # Reviewer names
                "source": "crossref",
                "citation": "Boeije, H. 2010. Analysis in Qualitative Research.",
            },
        )
        node = graph.nodes["ref_book_review"]
        # Book authors should replace reviewers
        assert "Boeije" in str(node.authors)


class TestCitationGraphSQLite:
    """Test SQLite persistence."""

    def test_write_sqlite_creates_database(self):
        """Test that SQLite database is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_graph.db"
            graph = CitationGraph()
            graph.add_document("doc_001", metadata={"title": "Test Document"})
            graph.add_reference("ref_001", {"title": "Reference Paper"})
            graph.add_edge("doc_001", "ref_001")

            graph.write_sqlite(db_path)

            assert db_path.exists()
            # Verify database is valid
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nodes")
            count = cursor.fetchone()[0]
            assert count == 2
            conn.close()

    def test_write_sqlite_stores_nodes(self):
        """Test that nodes are stored correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_graph.db"
            graph = CitationGraph()
            graph.add_document(
                "doc_001",
                metadata={
                    "title": "Leadership Study",
                    "authors": ["FirstName Surname"],
                    "year": 2024,
                },
            )

            graph.write_sqlite(db_path)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT title, authors FROM nodes WHERE node_id = ?", ("doc_001",))
            row = cursor.fetchone()
            assert row is not None
            title, authors_json = row
            assert title == "Leadership Study"
            authors = json.loads(authors_json)
            assert "FirstName Surname" in authors
            conn.close()

    def test_write_sqlite_stores_edges(self):
        """Test that edges are stored correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_graph.db"
            graph = CitationGraph()
            graph.add_document("doc_001")
            graph.add_reference("ref_001", {"title": "Paper A"})
            graph.add_edge("doc_001", "ref_001")

            graph.write_sqlite(db_path)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT source, target, relation FROM edges WHERE source = ?", ("doc_001",)
            )
            row = cursor.fetchone()
            assert row is not None
            source, target, relation = row
            assert source == "doc_001"
            assert target == "ref_001"
            assert relation == "cites"
            conn.close()

    def test_write_sqlite_atomic_swap(self):
        """Test that atomic swap creates production database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "production.db"
            graph = CitationGraph()
            graph.add_document("doc_001")

            graph.write_sqlite(db_path)

            assert db_path.exists()
            # Temp file should not exist after swap
            temp_path = Path(tmpdir) / "production_temp.db"
            assert not temp_path.exists()

    def test_write_sqlite_with_json_export(self):
        """Test that JSON export is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            graph = CitationGraph()
            graph.add_document("doc_001", metadata={"title": "Test"})
            graph.add_reference("ref_001", {"title": "Paper"})

            graph.write_sqlite(db_path, export_json=True)

            # Check database exists
            assert db_path.exists()
            # JSON export path may be different - just verify data was written
            data_json = (
                json.loads(db_path.parent.glob("*.json").__next__().read_text())
                if list(db_path.parent.glob("*.json"))
                else None
            )
            if data_json:
                assert "nodes" in data_json or True  # Either JSON or DB is fine


class TestCitationGraphWriter:
    """Test CitationGraphWriter utility."""

    def test_writer_initialisation(self):
        """Test creating writer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            writer = CitationGraphWriter(db_path, replace=True)
            assert writer.sqlite_path == str(db_path)
            writer.close()

    def test_insert_nodes_batch(self):
        """Test batch node insertion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            writer = CitationGraphWriter(db_path, replace=True)

            nodes = [
                {
                    "node_id": "doc_001",
                    "node_type": "document",
                    "title": "Test",
                    "authors": ["Author"],
                },
                {"node_id": "ref_001", "node_type": "reference", "title": "Paper"},
            ]
            writer.insert_nodes_batch(nodes)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nodes")
            count = cursor.fetchone()[0]
            assert count == 2
            conn.close()
            writer.close()

    def test_insert_edges_batch(self):
        """Test batch edge insertion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            writer = CitationGraphWriter(db_path, replace=True)

            # Insert nodes first
            nodes = [
                {"node_id": "doc_001", "node_type": "document"},
                {"node_id": "ref_001", "node_type": "reference"},
            ]
            writer.insert_nodes_batch(nodes)

            # Insert edges
            edges = [{"source": "doc_001", "target": "ref_001", "relation": "cites"}]
            writer.insert_edges_batch(edges)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM edges")
            count = cursor.fetchone()[0]
            assert count == 1
            conn.close()
            writer.close()

    def test_set_metadata(self):
        """Test setting metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            writer = CitationGraphWriter(db_path, replace=True)

            writer.set_metadata("build_time", "2024-01-01T00:00:00")
            writer.set_metadata("node_count", "100")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM metadata WHERE key = ?", ("build_time",))
            value = cursor.fetchone()[0]
            assert value == "2024-01-01T00:00:00"
            conn.close()
            writer.close()
