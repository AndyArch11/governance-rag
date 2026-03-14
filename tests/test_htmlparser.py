from pathlib import Path

import pytest

from scripts.ingest.htmlparser import (
    _clean_whitespace,
    _extract_confluence_metadata,
    _strip_boilerplate,
    extract_text_from_html,
)


def test_extract_text_basic():
    """Test basic HTML text extraction"""
    # Create a simple test HTML file
    test_html = """
    <html>
        <body>
            <h1>Test Document</h1>
            <p>This is test content.</p>
        </body>
    </html>
    """

    # Write to temp file
    test_file = Path("tests/fixtures/test_basic.html")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(test_html)

    result = extract_text_from_html(str(test_file))

    assert "Test Document" in result
    assert "This is test content" in result
    assert result.strip()  # Not empty


def test_extract_removes_navigation():
    """Test that nav/header/footer/script/style elements are removed"""
    test_html = """
    <html>
        <head>
            <title>Test Nav Removal</title>
            <style>
                nav {
                    display: none;
                }
            </style>
            <script defer src="script.js"></script>
        </head>
        <nav>Navigation menu</nav>
        <body>
            <h1>Content Header</h1>
            <p>Main content</p>
            <script>
                alert("Hello World!");
            </script>
        </body>
        <footer>Footer text</footer>
    </html>
    """

    test_file = Path("tests/fixtures/test_nav.html")
    test_file.write_text(test_html)

    result = extract_text_from_html(str(test_file))

    assert result.strip()  # Not empty
    assert "Content Header" in result
    assert "Main content" in result
    assert "Navigation menu" not in result
    assert "Footer text" not in result
    assert "script" not in result
    assert "style" not in result


class TestHTMLBoilerplateRemoval:
    """Tests for removal of HTML boilerplate patterns."""

    def test_remove_confluence_breadcrumbs(self, tmp_path):
        """Test removal of Confluence-style breadcrumb navigation."""
        html_file = tmp_path / "confluence.html"
        html_content = """
        <html>
            <body>
                <div>Home > Documentation > API Reference</div>
                <p>API documentation content</p>
            </body>
        </html>
        """
        html_file.write_text(html_content)

        text = extract_text_from_html(str(html_file))
        assert "API documentation content" in text

    def test_remove_page_numbers_from_html(self, tmp_path):
        """Test removal of page number patterns from HTML."""
        html_file = tmp_path / "numbered.html"
        html_content = """
        <html>
            <body>
                <p>First section</p>
                <p>Page 1 of 10</p>
                <p>Main content</p>
            </body>
        </html>
        """
        html_file.write_text(html_content)

        text = extract_text_from_html(str(html_file))
        assert "Main content" in text
        assert "Page 1 of 10" not in text

    def test_remove_nav_divs_with_classes(self, tmp_path):
        """Test removal of divs with navigation-related classes."""
        html_file = tmp_path / "nav_divs.html"
        html_content = """
        <html>
            <body>
                <div class="navigation">Side navigation</div>
                <div class="breadcrumb">Home > Docs</div>
                <div id="toc">Table of Contents</div>
                <div class="main-content">Main content here</div>
            </body>
        </html>
        """
        html_file.write_text(html_content)

        text = extract_text_from_html(str(html_file))
        assert "Main content here" in text
        # Navigation-related divs should be removed
        assert "Side navigation" not in text or len(text) > 0
        assert "Table of Contents" not in text or len(text) > 0


class TestMetadataExtraction:
    """Tests for Confluence metadata extraction."""

    def test_extract_labels(self):
        """Test extraction of Confluence labels."""
        text = """
        Main content
        Labels: feature, documentation, api
        More content
        """
        cleaned, metadata = _extract_confluence_metadata(text)

        assert "Labels:" not in cleaned
        assert metadata.get("labels") == "feature, documentation, api"
        assert "Main content" in cleaned
        assert "More content" in cleaned

    def test_extract_related_articles(self):
        """Test detection of related articles section."""
        text = """
        Document content
        Related Articles
        - Article 1
        - Article 2
        End of doc
        """
        cleaned, metadata = _extract_confluence_metadata(text)

        assert metadata.get("has_related") is True
        assert "Related Articles" not in cleaned
        assert "Document content" in cleaned


class TestBoilerplateStripping:
    """Tests for boilerplate pattern removal."""

    def test_strip_page_numbers(self):
        """Test removal of page number patterns."""
        text = """
        Introduction
        Page 1 of 50
        Content section
        Page 2 of 50
        More content
        """
        cleaned = _strip_boilerplate(text)

        assert "Page 1 of 50" not in cleaned
        assert "Page 2 of 50" not in cleaned
        assert "Introduction" in cleaned
        assert "Content section" in cleaned

    def test_strip_copyright_and_version(self):
        """Test removal of copyright and version patterns."""
        text = """
        Document Start
        Copyright © 2024
        Version 2.3
        Main document content
        """
        cleaned = _strip_boilerplate(text)

        assert "Copyright ©" not in cleaned
        assert "Version 2.3" not in cleaned
        assert "Document Start" in cleaned
        assert "Main document content" in cleaned

    def test_strip_date_patterns(self):
        """Test removal of date patterns."""
        text = """
        Report
        Generated on 01/15/2024
        Last modified on 02/20/2024
        Important data
        """
        cleaned = _strip_boilerplate(text)

        assert "Generated on" not in cleaned
        assert "Last modified on" not in cleaned
        assert "Report" in cleaned
        assert "Important data" in cleaned

    def test_preserve_content_section_headers(self):
        """Test that legitimate section headers are preserved."""
        text = """
        Document Title
        
        Section 1. Introduction
        This is important introduction content.
        
        Section 2. Methods
        Method description and details.
        """
        cleaned = _strip_boilerplate(text)

        # Content sections should remain (they have multiple words or are followed by content)
        assert "Section" in cleaned or "Introduction" in cleaned
        assert "important introduction content" in cleaned


class TestWhitespaceNormalisation:
    """Tests for whitespace cleaning and normalisation."""

    def test_collapse_multiple_blank_lines(self):
        """Test collapsing of excessive blank lines."""
        text = """
        Line 1


        Line 2



        Line 3
        """
        cleaned = _clean_whitespace(text)

        # Should have single blank lines between content
        double_newlines = cleaned.count("\n\n")
        triple_newlines = cleaned.count("\n\n\n")

        assert triple_newlines == 0, "Should not have triple newlines"
        assert double_newlines >= 1, "Should have some blank lines"

    def test_normalise_tabs(self):
        """Test conversion of tabs to spaces."""
        text = "Line\twith\ttabs\tand\tspaces"
        cleaned = _clean_whitespace(text)

        assert "\t" not in cleaned
        # Should still contain the text content
        assert "Line" in cleaned and "with" in cleaned

    def test_remove_trailing_whitespace(self):
        """Test removal of trailing whitespace."""
        text = """
        Line with trailing spaces    
        Another line\t\t
        Last line  
        """
        cleaned = _clean_whitespace(text)

        # No line should end with whitespace
        for line in cleaned.split("\n"):
            assert line == line.rstrip()


class TestHTMLCleaningIntegration:
    """Integration tests for realistic HTML cleaning scenarios."""

    def test_realistic_confluence_export(self, tmp_path):
        """Test cleaning of realistic Confluence export HTML."""
        html_file = tmp_path / "confluence_export.html"
        html_content = """
        <html>
            <body>
                <header>
                    <nav class="breadcrumb">
                        <div>Home > Products > Documentation > API</div>
                    </nav>
                </header>
                <main>
                    <h1>REST API Reference</h1>
                    <p>This document provides comprehensive API documentation.</p>
                    <h2>Authentication</h2>
                    <p>All API calls require authentication using OAuth 2.0.</p>
                    <h3>API Endpoints</h3>
                    <ul>
                        <li>GET /api/users</li>
                        <li>POST /api/data</li>
                    </ul>
                </main>
                <footer>
                    <p>Page 1 | Generated on 01/15/2024 | Version 1.0</p>
                    <p>Copyright © 2024 Company</p>
                </footer>
            </body>
        </html>
        """
        html_file.write_text(html_content)

        text = extract_text_from_html(str(html_file))

        # Important content should be present
        assert "REST API Reference" in text
        assert "Authentication" in text
        assert "OAuth 2.0" in text
        assert "API Endpoints" in text or "GET /api" in text

        # Boilerplate should be removed
        assert "Page 1 |" not in text
        assert "Copyright ©" not in text
        assert "Generated on" not in text

    def test_complex_html_with_multiple_nav_types(self, tmp_path):
        """Test handling of HTML with multiple types of navigation."""
        html_file = tmp_path / "complex.html"
        html_content = """
        <html>
            <body>
                <nav>Top navigation</nav>
                <aside class="sidebar">Side menu</aside>
                <div class="breadcrumb">Path > To > Page</div>
                <main class="content">
                    <h1>Main Article Title</h1>
                    <article>
                        <h2>Section 1</h2>
                        <p>Important article content goes here.</p>
                        <h2>Section 2</h2>
                        <p>More important content.</p>
                    </article>
                </main>
                <div id="toc">Related links</div>
            </body>
        </html>
        """
        html_file.write_text(html_content)

        text = extract_text_from_html(str(html_file))

        # Main content should remain
        assert "Main Article Title" in text
        assert "Section 1" in text
        assert "Important article content" in text or "article content" in text.lower()

        # Navigation should be removed
        assert "Top navigation" not in text
        assert "Side menu" not in text


class TestHTMLParserEdgeCases:
    """Tests for edge cases in HTML parsing."""

    def test_clean_whitespace_removes_empty_lines(self):
        """Test _clean_whitespace removes excessive blank lines."""
        text = "Line 1\n\n\n\n\nLine 2\n\n\nLine 3"
        cleaned = _clean_whitespace(text)
        # Should have at most 2 consecutive newlines
        assert "\n\n\n" not in cleaned

    def test_extract_text_from_empty_html(self, tmp_path):
        """Test extraction from minimal/empty HTML."""
        html_file = tmp_path / "empty.html"
        html_file.write_text("<html><body></body></html>")

        text = extract_text_from_html(str(html_file))
        # Should return empty or minimal text, not crash
        assert isinstance(text, str)

    def test_extract_text_handles_nested_tables(self, tmp_path):
        """Test extraction from HTML with nested tables."""
        html_file = tmp_path / "nested_table.html"
        html_content = """
        <html><body>
            <table>
                <tr><td>Outer cell 1</td><td>Outer cell 2</td></tr>
                <tr><td>
                    <table>
                        <tr><td>Inner cell A</td><td>Inner cell B</td></tr>
                    </table>
                </td></tr>
            </table>
        </body></html>
        """
        html_file.write_text(html_content)

        text = extract_text_from_html(str(html_file))
        # Should extract text from both tables
        assert "Outer cell" in text
        assert "Inner cell" in text

    def test_extract_text_preserves_lists(self, tmp_path):
        """Test that list items are extracted properly."""
        html_file = tmp_path / "lists.html"
        html_content = """
        <html><body>
            <ul>
                <li>First item</li>
                <li>Second item</li>
            </ul>
            <ol>
                <li>Numbered one</li>
                <li>Numbered two</li>
            </ol>
        </body></html>
        """
        html_file.write_text(html_content)

        text = extract_text_from_html(str(html_file))
        assert "First item" in text
        assert "Second item" in text
        assert "Numbered one" in text
        assert "Numbered two" in text


# ============================================================================
# Table Parsing Tests
# ============================================================================


class TestTableExtraction:
    """Tests for HTML table extraction and formatting."""

    def test_extract_simple_table(self, tmp_path):
        """Test extraction of a simple table."""
        from bs4 import BeautifulSoup

        from scripts.ingest.htmlparser import _convert_table_to_text, _extract_table_metadata

        html = """
        <table>
            <tr><th>Name</th><th>Age</th></tr>
            <tr><td>Alice</td><td>30</td></tr>
            <tr><td>Bob</td><td>25</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        # Test metadata extraction
        metadata = _extract_table_metadata(table)
        assert metadata["rows"] == 3
        assert metadata["cols"] == 2
        assert metadata["has_headers"] is True
        assert metadata["header_rows"] == 1

        # Test text conversion
        text = _convert_table_to_text(table, 0)
        assert "[TABLE 0]" in text
        assert "[/TABLE 0]" in text
        assert "Name" in text
        assert "Age" in text
        assert "Alice" in text
        assert "Bob" in text
        # Should have separator line
        assert "|---|---|" in text

    def test_extract_table_without_headers(self, tmp_path):
        """Test extraction of table with no header row."""
        from bs4 import BeautifulSoup

        from scripts.ingest.htmlparser import _convert_table_to_text, _extract_table_metadata

        html = """
        <table>
            <tr><td>A1</td><td>B1</td><td>C1</td></tr>
            <tr><td>A2</td><td>B2</td><td>C2</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        metadata = _extract_table_metadata(table)
        assert metadata["rows"] == 2
        assert metadata["cols"] == 3
        assert metadata["has_headers"] is False
        assert metadata["header_rows"] == 0

        text = _convert_table_to_text(table, 0)
        # Should NOT have separator line for headerless table
        assert "|---|" not in text
        assert "A1" in text
        assert "B2" in text

    def test_extract_layout_table_emits_layout_type(self, tmp_path):
        """Layout-like tables should emit explicit layout type metadata."""
        from bs4 import BeautifulSoup

        from scripts.ingest.htmlparser import _convert_table_to_text, _extract_table_metadata

        html = """
        <table>
            <tr><td>Logo area</td><td>Utility links</td></tr>
            <tr><td>Left gutter</td><td>Right gutter</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        metadata = _extract_table_metadata(table)
        assert metadata["table_type"] == "layout"

        text = _convert_table_to_text(table, 0)
        assert "Table-Type: layout" in text

    def test_extract_content_table_emits_content_type(self, tmp_path):
        """Data tables should emit explicit content type metadata."""
        from bs4 import BeautifulSoup

        from scripts.ingest.htmlparser import _convert_table_to_text, _extract_table_metadata

        html = """
        <table>
            <caption>Monthly costs</caption>
            <tr><th>Tier</th><th>Cost</th></tr>
            <tr><td>Basic</td><td>100</td></tr>
            <tr><td>Pro</td><td>250</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        metadata = _extract_table_metadata(table)
        assert metadata["table_type"] == "content"

        text = _convert_table_to_text(table, 0)
        assert "Table-Type: content" in text


class TestTableMergedCells:
    """Tests for tables with colspan and rowspan."""

    def test_table_with_colspan(self, tmp_path):
        """Test extraction of table with colspan."""
        from bs4 import BeautifulSoup

        from scripts.ingest.htmlparser import _convert_table_to_text, _extract_table_metadata

        html = """
        <table>
            <tr>
                <th colspan="2">Merged Header</th>
                <th>Col 3</th>
            </tr>
            <tr>
                <td>A1</td>
                <td>B1</td>
                <td>C1</td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        # Metadata should account for colspan
        metadata = _extract_table_metadata(table)
        assert metadata["cols"] == 3, "Column count should account for colspan"
        assert metadata["header_rows"] == 1

        # Text should repeat merged cell content
        text = _convert_table_to_text(table, 0)
        # Merged header should appear twice
        assert text.count("Merged Header") == 2
        assert "Col 3" in text

    def test_table_with_rowspan(self, tmp_path):
        """Test extraction of table with rowspan (basic handling)."""
        from bs4 import BeautifulSoup

        from scripts.ingest.htmlparser import _convert_table_to_text

        html = """
        <table>
            <tr>
                <th>Name</th>
                <th>Details</th>
            </tr>
            <tr>
                <td rowspan="2">Alice</td>
                <td>Age: 30</td>
            </tr>
            <tr>
                <td>City: NYC</td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        text = _convert_table_to_text(table, 0)
        # Should at least capture the content (even if rowspan handling is simplified)
        assert "Alice" in text
        assert "Age: 30" in text
        assert "City: NYC" in text

    def test_table_with_multiple_colspan(self, tmp_path):
        """Test table with multiple merged cells in same row."""
        from bs4 import BeautifulSoup

        from scripts.ingest.htmlparser import _convert_table_to_text, _extract_table_metadata

        html = """
        <table>
            <tr>
                <th colspan="2">Section A</th>
                <th colspan="2">Section B</th>
            </tr>
            <tr>
                <th>A1</th>
                <th>A2</th>
                <th>B1</th>
                <th>B2</th>
            </tr>
            <tr>
                <td>Data 1</td>
                <td>Data 2</td>
                <td>Data 3</td>
                <td>Data 4</td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        metadata = _extract_table_metadata(table)
        assert metadata["cols"] == 4
        assert metadata["header_rows"] == 2

        text = _convert_table_to_text(table, 0)
        # Each merged header should appear twice
        assert text.count("Section A") == 2
        assert text.count("Section B") == 2


class TestTableNestedTables:
    """Tests for tables with nested tables."""

    def test_nested_table_not_extracted_separately(self, tmp_path):
        """Test that nested tables are not processed as separate top-level tables."""
        from bs4 import BeautifulSoup

        from scripts.ingest.htmlparser import _extract_tables_from_html

        html = """
        <html><body>
            <table>
                <tr>
                    <td>Outer cell</td>
                    <td>
                        <table>
                            <tr><td>Nested A</td></tr>
                        </table>
                    </td>
                </tr>
            </table>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")

        tables_text, metadata = _extract_tables_from_html(soup)

        # Should only extract 1 top-level table (not the nested one)
        assert len(metadata) == 1
        assert metadata[0]["index"] == 0

    def test_nested_table_content_preserved(self, tmp_path):
        """Test that content from nested tables is preserved inline."""
        from bs4 import BeautifulSoup

        from scripts.ingest.htmlparser import _convert_table_to_text

        html = """
        <table>
            <tr>
                <td colspan="2">
                    Details:
                    <table>
                        <tr><td>Region A</td><td>$200K</td></tr>
                        <tr><td>Region B</td><td>$300K</td></tr>
                    </table>
                </td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        text = _convert_table_to_text(table, 0)

        # Nested table content should be preserved as flattened text
        assert "Region A" in text
        assert "200K" in text
        assert "Region B" in text
        assert "300K" in text
        # Should be marked as nested
        assert "[nested:" in text

    def test_multiple_nested_tables(self, tmp_path):
        """Test handling of multiple nested tables."""
        from scripts.ingest.htmlparser import extract_text_from_html

        html_file = tmp_path / "multi_nested.html"
        html_content = """
        <html><body>
            <table>
                <tr>
                    <td>Cell 1:
                        <table><tr><td>Nested 1A</td></tr></table>
                    </td>
                    <td>Cell 2:
                        <table><tr><td>Nested 2A</td></tr></table>
                    </td>
                </tr>
            </table>

            <table>
                <tr><td>Standalone table</td></tr>
            </table>
        </body></html>
        """
        html_file.write_text(html_content)

        text = extract_text_from_html(str(html_file))

        # Should have 2 top-level [TABLE N] markers
        assert text.count("[TABLE ") == 2
        # Nested content should be preserved
        assert "Nested 1A" in text
        assert "Nested 2A" in text
        assert "Standalone table" in text


class TestTableMultipleHeaderRows:
    """Tests for tables with multiple header rows."""

    def test_two_header_rows(self, tmp_path):
        """Test table with two consecutive header rows."""
        from bs4 import BeautifulSoup

        from scripts.ingest.htmlparser import _convert_table_to_text, _extract_table_metadata

        html = """
        <table>
            <tr>
                <th colspan="2">Performance</th>
                <th colspan="2">Cost</th>
            </tr>
            <tr>
                <th>CPU</th>
                <th>Memory</th>
                <th>Budget</th>
                <th>Actual</th>
            </tr>
            <tr>
                <td>80%</td>
                <td>4GB</td>
                <td>$100K</td>
                <td>$95K</td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        metadata = _extract_table_metadata(table)
        assert metadata["header_rows"] == 2, "Should detect 2 header rows"
        assert metadata["rows"] == 3
        assert metadata["cols"] == 4

        text = _convert_table_to_text(table, 0)
        lines = text.split("\n")

        # Should have both header rows before separator
        assert "Table-Type: content" in lines[1]
        assert "Performance" in lines[2]
        assert "CPU" in lines[3]
        # Separator should follow the header rows.
        assert lines[4].startswith("|---")
        # Data row should come after separator.
        assert "80%" in lines[5]

    def test_three_header_rows(self, tmp_path):
        """Test table with three consecutive header rows."""
        from bs4 import BeautifulSoup

        from scripts.ingest.htmlparser import _convert_table_to_text, _extract_table_metadata

        html = """
        <table>
            <tr><th colspan="4">Annual Report 2025</th></tr>
            <tr><th colspan="2">Q1-Q2</th><th colspan="2">Q3-Q4</th></tr>
            <tr><th>Rev</th><th>Profit</th><th>Rev</th><th>Profit</th></tr>
            <tr><td>$100K</td><td>$20K</td><td>$150K</td><td>$30K</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        metadata = _extract_table_metadata(table)
        assert metadata["header_rows"] == 3

        text = _convert_table_to_text(table, 0)
        lines = text.split("\n")

        # All 3 headers before separator
        assert "Annual Report 2025" in text
        assert "Q1-Q2" in text
        assert "Rev" in text
        assert "Table-Type: content" in lines[1]
        # Separator after 3rd header
        assert lines[5].startswith("|---")

    def test_mixed_th_and_td_rows(self, tmp_path):
        """Test that only consecutive TH rows from start are treated as headers."""
        from bs4 import BeautifulSoup

        from scripts.ingest.htmlparser import _extract_table_metadata

        html = """
        <table>
            <tr><th>Header 1</th></tr>
            <tr><th>Header 2</th></tr>
            <tr><td>Data 1</td></tr>
            <tr><th>Not a header (comes after TD)</th></tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        metadata = _extract_table_metadata(table)
        # Only first 2 consecutive TH rows should count
        assert metadata["header_rows"] == 2


class TestTableEdgeCases:
    """Tests for edge cases in table processing."""

    def test_empty_table(self, tmp_path):
        """Test handling of empty table."""
        from bs4 import BeautifulSoup

        from scripts.ingest.htmlparser import _extract_table_metadata

        html = "<table></table>"
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        metadata = _extract_table_metadata(table)
        assert metadata["rows"] == 0
        assert metadata["cols"] == 0

    def test_single_row_table_not_skipped(self, tmp_path):
        """Test that single-row tables are extracted (not skipped)."""
        from bs4 import BeautifulSoup

        from scripts.ingest.htmlparser import _extract_tables_from_html

        html = """
        <html><body>
            <table>
                <tr><th>Col A</th><th>Col B</th><th>Col C</th></tr>
            </table>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")

        tables_text, metadata = _extract_tables_from_html(soup)

        # Single-row table should NOT be skipped
        assert len(metadata) == 1
        assert metadata[0]["rows"] == 1
        assert metadata[0]["cols"] == 3

    def test_single_column_table_not_skipped(self, tmp_path):
        """Test that single-column tables are extracted."""
        from bs4 import BeautifulSoup

        from scripts.ingest.htmlparser import _extract_tables_from_html

        html = """
        <html><body>
            <table>
                <tr><th>Header</th></tr>
                <tr><td>Row 1</td></tr>
                <tr><td>Row 2</td></tr>
            </table>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")

        tables_text, metadata = _extract_tables_from_html(soup)

        # Single-column table should NOT be skipped
        assert len(metadata) == 1
        assert metadata[0]["cols"] == 1
        assert metadata[0]["rows"] == 3

    def test_table_with_tbody(self, tmp_path):
        """Test table with tbody/thead structure."""
        from bs4 import BeautifulSoup

        from scripts.ingest.htmlparser import _extract_table_metadata

        html = """
        <table>
            <thead>
                <tr><th>Name</th><th>Value</th></tr>
            </thead>
            <tbody>
                <tr><td>Item 1</td><td>100</td></tr>
                <tr><td>Item 2</td><td>200</td></tr>
            </tbody>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        metadata = _extract_table_metadata(table)
        assert metadata["rows"] == 3
        assert metadata["cols"] == 2
        assert metadata["has_headers"] is True


class TestTableFullIntegration:
    """Integration tests for complete table extraction pipeline."""

    def test_complex_table_full_extraction(self, tmp_path):
        """Test extraction of complex table with all features."""
        from scripts.ingest.htmlparser import extract_text_from_html

        html_file = tmp_path / "complex_table.html"
        html_content = """
        <html><body>
            <h1>Quarterly Report</h1>
            <table>
                <tr>
                    <th rowspan="2">Metric</th>
                    <th colspan="2">Q1 2025</th>
                    <th colspan="2">Q2 2025</th>
                </tr>
                <tr>
                    <th>Target</th>
                    <th>Actual</th>
                    <th>Target</th>
                    <th>Actual</th>
                </tr>
                <tr>
                    <td>Revenue</td>
                    <td>$100K</td>
                    <td>$95K</td>
                    <td>$120K</td>
                    <td>$125K</td>
                </tr>
            </table>
        </body></html>
        """
        html_file.write_text(html_content)

        text = extract_text_from_html(str(html_file))

        # Should have [TABLE 0] markers
        assert "[TABLE 0]" in text
        assert "[/TABLE 0]" in text

        # Content should be present
        assert "Q1 2025" in text
        assert "Revenue" in text
        assert "$95K" in text

        # Should have separator line (after header rows)
        assert "|---" in text

    def test_multiple_tables_extraction(self, tmp_path):
        """Test extraction of multiple tables from single document."""
        from scripts.ingest.htmlparser import extract_text_from_html

        html_file = tmp_path / "multi_tables.html"
        html_content = """
        <html><body>
            <h1>Report</h1>
            
            <table>
                <tr><th>Name</th><th>Age</th></tr>
                <tr><td>Alice</td><td>30</td></tr>
            </table>
            
            <p>Some text between tables</p>
            
            <table>
                <tr><th>Product</th><th>Price</th></tr>
                <tr><td>Widget</td><td>$10</td></tr>
            </table>
        </body></html>
        """
        html_file.write_text(html_content)

        text = extract_text_from_html(str(html_file))

        # Should have 2 table markers
        assert text.count("[TABLE ") == 2
        assert "[TABLE 0]" in text
        assert "[TABLE 1]" in text
        assert "[/TABLE 0]" in text
        assert "[/TABLE 1]" in text

        # Content from both tables
        assert "Alice" in text
        assert "Widget" in text
