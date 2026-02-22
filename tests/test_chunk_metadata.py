"""Tests for chunk metadata extraction functions.

Tests cover the functions in chunk.py lines 400+ that extract technical entities,
detect code languages, build heading paths, and create enhanced metadata.
"""

import pytest

from scripts.ingest.chunk import (
    create_enhanced_metadata,
    detect_code_language,
    extract_heading_path,
    extract_technical_entities,
)

# =========================
# Test extract_technical_entities
# =========================


class TestExtractTechnicalEntities:
    """Tests for extract_technical_entities function."""

    def test_extract_api_endpoints(self):
        """Test extraction of API endpoints."""
        text = "Call /api/v1/users and /api/v2/resources/items endpoints"
        entities = extract_technical_entities(text)

        assert "/api/v1/users" in entities
        assert "/api/v2/resources/items" in entities

    def test_extract_function_names(self):
        """Test extraction of function names with parentheses."""
        text = "Use get_user_data() and process_record() functions"
        entities = extract_technical_entities(text)

        assert "get_user_data" in entities
        assert "process_record" in entities

    def test_extract_class_names(self):
        """Test extraction of PascalCase class names."""
        text = "UserManager and DataProcessor classes handle the logic"
        entities = extract_technical_entities(text)

        assert "UserManager" in entities
        assert "DataProcessor" in entities

    def test_extract_constants(self):
        """Test extraction of UPPER_SNAKE_CASE constants."""
        text = "Set MAX_CONNECTIONS and DEFAULT_TIMEOUT values"
        entities = extract_technical_entities(text)

        assert "MAX_CONNECTIONS" in entities
        assert "DEFAULT_TIMEOUT" in entities

    def test_extract_config_keys(self):
        """Test extraction of configuration keys with dot notation."""
        text = "Configure database.host and server.port settings"
        entities = extract_technical_entities(text)

        assert "database.host" in entities
        assert "server.port" in entities

    def test_extract_module_names(self):
        """Test extraction of module names with underscores."""
        text = "Import user_authentication and data_processor modules"
        entities = extract_technical_entities(text)

        assert "user_authentication" in entities
        assert "data_processor" in entities

    def test_extract_mixed_entities(self):
        """Test extraction of multiple entity types."""
        text = """
        API endpoint: /api/v1/users
        Function: get_user_by_id()
        Class: UserService
        Constant: MAX_RETRY_COUNT
        Config: app.database.connection_string
        Module: authentication_helper
        """
        entities = extract_technical_entities(text)

        assert "/api/v1/users" in entities
        assert "get_user_by_id" in entities
        assert "UserService" in entities
        assert "MAX_RETRY_COUNT" in entities
        assert "app.database.connection_string" in entities
        assert "authentication_helper" in entities

    def test_extract_empty_text(self):
        """Test extraction from empty text."""
        entities = extract_technical_entities("")

        assert entities == []

    def test_extract_no_entities(self):
        """Test extraction from text with no technical entities."""
        text = "This is just plain text without any code elements"
        entities = extract_technical_entities(text)

        assert len(entities) == 0

    def test_extract_limits_to_20_entities(self):
        """Test that extraction limits to top 20 entities."""
        # Create text with many entities
        text = " ".join([f"func_{i}()" for i in range(50)])
        entities = extract_technical_entities(text)

        assert len(entities) <= 20

    def test_extract_filters_short_module_names(self):
        """Test that short module names are filtered."""
        text = "Use abc and very_long_module_name modules"
        entities = extract_technical_entities(text)

        # Short names (< 6 chars) should be filtered
        assert "abc" not in entities
        # Long names should be kept
        assert "very_long_module_name" in entities

    def test_extract_deduplicates_entities(self):
        """Test that duplicate entities are removed."""
        text = "Call get_data() again, use get_data() function"
        entities = extract_technical_entities(text)

        # Should only appear once
        assert entities.count("get_data") == 1

    def test_extract_sorts_entities(self):
        """Test that entities are sorted alphabetically."""
        text = "zulu() yankee() alpha() bravo()"
        entities = extract_technical_entities(text)

        # Should be sorted
        assert entities == sorted(entities)

    def test_extract_api_with_methods(self):
        """Test extraction of API endpoints with HTTP methods."""
        text = "GET /api/users POST /api/users/create"
        entities = extract_technical_entities(text)

        assert "/api/users" in entities
        assert "/api/users/create" in entities

    def test_extract_nested_config_keys(self):
        """Test extraction of deeply nested config keys."""
        text = "Set app.database.connection.pool.max_size value"
        entities = extract_technical_entities(text)

        assert "app.database.connection.pool.max_size" in entities

    def test_extract_with_special_characters(self):
        """Test extraction handles special characters gracefully."""
        text = "Use get_user!() and process@data() somehow"
        entities = extract_technical_entities(text)

        # Functions with special chars may not be extracted, but shouldn't crash
        assert isinstance(entities, list)


# =========================
# Test detect_code_language
# =========================


class TestDetectCodeLanguage:
    """Tests for detect_code_language function."""

    def test_detect_python_fence(self):
        """Test detection of Python from code fence."""
        text = "```python\ndef hello():\n    print('world')\n```"
        lang = detect_code_language(text)

        assert lang == "python"

    def test_detect_javascript_fence(self):
        """Test detection of JavaScript from code fence."""
        text = "```javascript\nconst x = 5;\n```"
        lang = detect_code_language(text)

        assert lang == "javascript"

    def test_detect_sql_fence(self):
        """Test detection of SQL from code fence."""
        text = "```sql\nSELECT * FROM users;\n```"
        lang = detect_code_language(text)

        assert lang == "sql"

    def test_detect_python_keywords(self):
        """Test detection of Python from keywords."""
        text = "Here is code: `def my_function():` and `import pandas`"
        lang = detect_code_language(text)

        assert lang == "python"

    def test_detect_python_class_keyword(self):
        """Test detection of Python from class keyword."""
        text = "Define a `class MyClass:` here"
        lang = detect_code_language(text)

        assert lang == "python"

    def test_detect_python_self_keyword(self):
        """Test detection of Python from self usage."""
        text = "Use `self.attribute` in methods"
        lang = detect_code_language(text)

        assert lang == "python"

    def test_detect_python_print(self):
        """Test detection of Python from print function."""
        text = "Output with `print('hello')`"
        lang = detect_code_language(text)

        assert lang == "python"

    def test_detect_javascript_const(self):
        """Test detection of JavaScript from const keyword."""
        text = "Declare `const myVar = 10;`"
        lang = detect_code_language(text)

        assert lang == "javascript"

    def test_detect_javascript_let(self):
        """Test detection of JavaScript from let keyword."""
        text = "Use `let counter = 0;`"
        lang = detect_code_language(text)

        assert lang == "javascript"

    def test_detect_javascript_var(self):
        """Test detection of JavaScript from var keyword."""
        text = "Old style `var x = 5;`"
        lang = detect_code_language(text)

        assert lang == "javascript"

    def test_detect_javascript_arrow_function(self):
        """Test detection of JavaScript from arrow function."""
        text = "Arrow function `const fn = () => { };`"
        lang = detect_code_language(text)

        assert lang == "javascript"

    def test_detect_javascript_console_log(self):
        """Test detection of JavaScript from console.log."""
        text = "Debug with `console.log('value')`"
        lang = detect_code_language(text)

        assert lang == "javascript"

    def test_detect_sql_select(self):
        """Test detection of SQL from SELECT keyword."""
        text = "Query: `SELECT id, name FROM users`"
        lang = detect_code_language(text)

        assert lang == "sql"

    def test_detect_sql_insert(self):
        """Test detection of SQL from INSERT keyword."""
        text = "Add data: `INSERT INTO users VALUES (1, 'john')`"
        lang = detect_code_language(text)

        assert lang == "sql"

    def test_detect_yaml_from_structure(self):
        """Test detection of YAML from structure."""
        text = """Config:
```
version: 1.0
database:
  host: localhost
```"""
        lang = detect_code_language(text)

        assert lang == "yaml"

    def test_detect_no_code(self):
        """Test detection returns None when no code present."""
        text = "This is just plain text without any code"
        lang = detect_code_language(text)

        assert lang is None

    def test_detect_empty_text(self):
        """Test detection with empty text."""
        lang = detect_code_language("")

        assert lang is None

    def test_detect_prefers_fence_over_keywords(self):
        """Test that code fence takes precedence over keywords."""
        # Text has JavaScript fence but also Python keywords
        text = "```javascript\nconst x = 5;\n```\nAlso has print() in description"
        lang = detect_code_language(text)

        # Should detect from fence first
        assert lang == "javascript"

    def test_detect_case_insensitive_sql(self):
        """Test SQL detection is case-insensitive."""
        text = "Run query: `select * from users`"
        lang = detect_code_language(text)

        assert lang == "sql"


# =========================
# Test extract_heading_path
# =========================


class TestExtractHeadingPath:
    """Tests for extract_heading_path function."""

    def test_extract_single_heading(self):
        """Test extraction with single heading."""
        full_text = """# Main Title

This is content.
More content here."""
        chunk = "More content here."

        path = extract_heading_path(chunk, full_text)

        assert path == "Main Title"

    def test_extract_nested_headings(self):
        """Test extraction with nested heading hierarchy."""
        full_text = """# Document

## Section 1

### Subsection 1.1

Content for subsection 1.1"""
        chunk = "Content for subsection 1.1"

        path = extract_heading_path(chunk, full_text)

        assert path == "Document > Section 1 > Subsection 1.1"

    def test_extract_heading_multiple_levels(self):
        """Test extraction across multiple heading levels."""
        full_text = """# Title

## Chapter 1

### Part A

#### Detail 1

This is the detailed content."""
        chunk = "This is the detailed content."

        path = extract_heading_path(chunk, full_text)

        assert "Title" in path
        assert "Chapter 1" in path
        assert "Part A" in path
        assert "Detail 1" in path

    def test_extract_heading_mid_document(self):
        """Test extraction from middle of document."""
        full_text = """# Title

## First Section

Content here.

## Second Section

### Subsection

Target content."""
        chunk = "Target content."

        path = extract_heading_path(chunk, full_text)

        assert path == "Title > Second Section > Subsection"

    def test_extract_no_headings(self):
        """Test extraction when no headings present."""
        full_text = "Just plain text without any headings."
        chunk = "plain text"

        path = extract_heading_path(chunk, full_text)

        assert path is None

    def test_extract_chunk_not_found(self):
        """Test extraction when chunk not found in full text."""
        full_text = "This is the full text."
        chunk = "This is different text."

        path = extract_heading_path(chunk, full_text)

        assert path is None

    def test_extract_chunk_at_start(self):
        """Test extraction when chunk is at document start."""
        full_text = """# Title

First paragraph of content.

## Section

More content."""
        chunk = "First paragraph of content."

        path = extract_heading_path(chunk, full_text)

        assert path == "Title"

    def test_extract_heading_clears_deeper_levels(self):
        """Test that deeper heading levels are cleared when encountering shallower level."""
        full_text = """# Title

## Section A

### Subsection A.1

Content A.1

## Section B

Content B"""
        chunk = "Content B"

        path = extract_heading_path(chunk, full_text)

        # Should be Title > Section B, not include Subsection A.1
        assert path == "Title > Section B"

    def test_extract_short_chunk(self):
        """Test extraction with very short chunk."""
        full_text = """# Heading

Some longer content paragraph here."""
        chunk = "Some"

        path = extract_heading_path(chunk, full_text)

        assert path == "Heading"

    def test_extract_empty_chunk(self):
        """Test extraction with empty chunk."""
        full_text = "# Heading\n\nContent"
        chunk = ""

        path = extract_heading_path(chunk, full_text)

        # Empty chunk won't be found
        assert path is None

    def test_extract_with_special_characters_in_headings(self):
        """Test extraction with special characters in heading text."""
        full_text = """# Configuration & Setup

## Database (MySQL)

### Connection: localhost:3306

Content here"""
        chunk = "Content here"

        path = extract_heading_path(chunk, full_text)

        assert "Configuration & Setup" in path
        assert "Database (MySQL)" in path
        assert "Connection: localhost:3306" in path


# =========================
# Test create_enhanced_metadata
# =========================


class TestCreateEnhancedMetadata:
    """Tests for create_enhanced_metadata function."""

    def test_create_metadata_basic(self):
        """Test basic metadata creation."""
        chunk = "This is a simple text chunk."
        metadata = create_enhanced_metadata(
            chunk_text=chunk, chunk_index=0, total_chunks=3, doc_id="doc_123"
        )

        assert metadata.content_type == "text"
        assert metadata.contains_code is False
        assert metadata.contains_table is False
        assert metadata.contains_diagram is False

    def test_create_metadata_with_code(self):
        """Test metadata creation with code content."""
        chunk = "```python\ndef hello():\n    print('world')\n```"
        metadata = create_enhanced_metadata(
            chunk_text=chunk, chunk_index=0, total_chunks=1, doc_id="doc_123"
        )

        assert metadata.contains_code is True
        assert metadata.content_type == "code"
        assert metadata.code_language == "python"

    def test_create_metadata_with_inline_code(self):
        """Test metadata with inline code markers."""
        chunk = "Use the `get_user()` function to retrieve data."
        metadata = create_enhanced_metadata(
            chunk_text=chunk, chunk_index=0, total_chunks=1, doc_id="doc_123"
        )

        assert metadata.contains_code is True
        assert metadata.content_type == "mixed"

    def test_create_metadata_with_table_marker(self):
        """Test metadata with table marker."""
        chunk = "[TABLE 1]\n| Col1 | Col2 |\n| ---- | ---- |\n[/TABLE 1]"
        metadata = create_enhanced_metadata(
            chunk_text=chunk, chunk_index=0, total_chunks=1, doc_id="doc_123"
        )

        assert metadata.contains_table is True
        assert metadata.content_type == "structured"

    def test_create_metadata_with_markdown_table(self):
        """Test metadata with markdown table."""
        chunk = "| Name | Age |\n| --- | --- |\n| John | 30 |"
        metadata = create_enhanced_metadata(
            chunk_text=chunk, chunk_index=0, total_chunks=1, doc_id="doc_123"
        )

        assert metadata.contains_table is True

    def test_create_metadata_with_diagram(self):
        """Test metadata with diagram reference."""
        chunk = "See the diagram below:\n![Architecture Diagram](image.png)"
        metadata = create_enhanced_metadata(
            chunk_text=chunk, chunk_index=0, total_chunks=1, doc_id="doc_123"
        )

        assert metadata.contains_diagram is True

    def test_create_metadata_api_reference(self):
        """Test metadata identifies API reference content."""
        chunk = "API endpoint: GET /api/users\nReturns: List of users"
        metadata = create_enhanced_metadata(
            chunk_text=chunk, chunk_index=0, total_chunks=1, doc_id="doc_123"
        )

        assert metadata.is_api_reference is True

    def test_create_metadata_configuration(self):
        """Test metadata identifies configuration content."""
        chunk = "Configuration settings:\n- database.host: localhost\n- server.port: 8080"
        metadata = create_enhanced_metadata(
            chunk_text=chunk, chunk_index=0, total_chunks=1, doc_id="doc_123"
        )

        assert metadata.is_configuration is True

    def test_create_metadata_technical_entities(self):
        """Test metadata extracts technical entities."""
        chunk = "Use UserService class and get_user() function"
        metadata = create_enhanced_metadata(
            chunk_text=chunk, chunk_index=0, total_chunks=1, doc_id="doc_123"
        )

        assert "UserService" in metadata.technical_entities
        assert "get_user" in metadata.technical_entities

    def test_create_metadata_with_heading_path(self):
        """Test metadata includes heading path."""
        full_text = """# Guide

## Setup

### Installation

Run the installer"""
        chunk = "Run the installer"
        metadata = create_enhanced_metadata(
            chunk_text=chunk, chunk_index=0, total_chunks=1, doc_id="doc_123", full_text=full_text
        )

        assert metadata.heading_path == "Guide > Setup > Installation"
        assert metadata.parent_section == "Installation"
        assert metadata.section_depth == 3

    def test_create_metadata_sequential_chunks(self):
        """Test metadata includes sequential chunk references."""
        chunk = "Middle chunk content"
        metadata = create_enhanced_metadata(
            chunk_text=chunk, chunk_index=5, total_chunks=10, doc_id="doc_123"
        )

        assert metadata.prev_chunk_id == "doc_123_chunk_4"
        assert metadata.next_chunk_id == "doc_123_chunk_6"

    def test_create_metadata_first_chunk(self):
        """Test metadata for first chunk has no previous."""
        metadata = create_enhanced_metadata(
            chunk_text="First chunk", chunk_index=0, total_chunks=5, doc_id="doc_123"
        )

        assert metadata.prev_chunk_id is None
        assert metadata.next_chunk_id == "doc_123_chunk_1"

    def test_create_metadata_last_chunk(self):
        """Test metadata for last chunk has no next."""
        metadata = create_enhanced_metadata(
            chunk_text="Last chunk", chunk_index=4, total_chunks=5, doc_id="doc_123"
        )

        assert metadata.prev_chunk_id == "doc_123_chunk_3"
        assert metadata.next_chunk_id is None

    def test_create_metadata_single_chunk(self):
        """Test metadata for single chunk document."""
        metadata = create_enhanced_metadata(
            chunk_text="Only chunk", chunk_index=0, total_chunks=1, doc_id="doc_123"
        )

        assert metadata.prev_chunk_id is None
        assert metadata.next_chunk_id is None

    def test_create_metadata_mixed_content_type(self):
        """Test metadata with mixed content (code + text)."""
        chunk = "Here is some code `function()` and regular text"
        metadata = create_enhanced_metadata(
            chunk_text=chunk, chunk_index=0, total_chunks=1, doc_id="doc_123"
        )

        assert metadata.content_type == "mixed"
        assert metadata.contains_code is True

    def test_create_metadata_without_full_text(self):
        """Test metadata creation without full_text parameter."""
        chunk = "Content without context"
        metadata = create_enhanced_metadata(
            chunk_text=chunk, chunk_index=0, total_chunks=1, doc_id="doc_123"
        )

        assert metadata.heading_path is None
        assert metadata.parent_section is None
        assert metadata.section_depth is None

    def test_create_metadata_with_doc_type(self):
        """Test metadata creation with doc_type."""
        chunk = "Configuration content"
        metadata = create_enhanced_metadata(
            chunk_text=chunk,
            chunk_index=0,
            total_chunks=1,
            doc_id="doc_123",
            doc_type="technical_guide",
        )

        # doc_type is passed but not stored in metadata (used for chunking)
        assert metadata is not None

    def test_create_metadata_javascript_code(self):
        """Test metadata detects JavaScript code."""
        chunk = "```javascript\nconst x = 5;\nconsole.log(x);\n```"
        metadata = create_enhanced_metadata(
            chunk_text=chunk, chunk_index=0, total_chunks=1, doc_id="doc_123"
        )

        assert metadata.code_language == "javascript"
        assert metadata.contains_code is True

    def test_create_metadata_sql_code(self):
        """Test metadata detects SQL code in fence."""
        chunk = "```sql\nSELECT * FROM users WHERE active = 1\n```"
        metadata = create_enhanced_metadata(
            chunk_text=chunk, chunk_index=0, total_chunks=1, doc_id="doc_123"
        )

        # SQL in code fence should be detected
        assert metadata.code_language == "sql"
        assert metadata.contains_code is True
        assert metadata.content_type == "code"

    def test_create_metadata_multiple_features(self):
        """Test metadata with multiple content features."""
        chunk = """# API Reference

## User Endpoint

API: GET /api/users
Configuration: api.base_url

```python
def get_users():
    return client.get('/api/users')
```

| Status | Description |
| --- | --- |
| 200 | Success |
"""
        full_text = f"# API Reference\n\n{chunk}"
        metadata = create_enhanced_metadata(
            chunk_text=chunk, chunk_index=0, total_chunks=1, doc_id="doc_123", full_text=full_text
        )

        assert metadata.contains_code is True
        assert metadata.contains_table is True
        assert metadata.is_api_reference is True
        assert metadata.is_configuration is True
        assert metadata.code_language == "python"
        assert "get_users" in metadata.technical_entities
        assert "/api/users" in metadata.technical_entities

    def test_create_metadata_parent_section_extraction(self):
        """Test parent section extraction from heading path."""
        full_text = """# Book

## Chapter 1

### Section A

Content here"""
        chunk = "Content here"
        metadata = create_enhanced_metadata(
            chunk_text=chunk, chunk_index=0, total_chunks=1, doc_id="doc_123", full_text=full_text
        )

        # Parent section should be the deepest heading
        assert metadata.parent_section == "Section A"
        assert metadata.section_depth == 3
