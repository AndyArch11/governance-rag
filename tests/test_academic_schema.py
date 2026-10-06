"""Tests for the declarative academic ingestion SQLite schema."""

import sqlite3

from scripts.ingest.academic import schema


def test_schema_creation_sql_initialises_all_declared_tables() -> None:
    """The combined migration SQL creates every table described by schema info."""
    connection = sqlite3.connect(":memory:")
    connection.executescript(schema.get_schema_creation_sql())

    table_names = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert set(schema.get_table_creation_sqls()).issubset(table_names)
    assert "schema_migrations" in table_names
    assert connection.execute("SELECT version FROM schema_migrations").fetchone() == (
        schema.SCHEMA_VERSION,
    )


def test_schema_info_matches_creation_and_query_contracts() -> None:
    """Schema metadata and public SQL templates remain consistent."""
    table_sql = schema.get_table_creation_sqls()
    info = schema.get_schema_info()

    assert info["version"] == schema.SCHEMA_VERSION
    assert set(info["tables"]) == set(table_sql)
    assert "academic_documents" in schema.INSERT_DOCUMENT
    assert "academic_references" in schema.INSERT_REFERENCE
    assert "academic_citation_edges" in schema.INSERT_CITATION_EDGE
    assert "academic_domain_terms" in schema.INSERT_DOMAIN_TERM
    assert "WHERE doc_id = ?" in schema.SELECT_DOCUMENT_BY_ID
    assert "WHERE ref_id = ?" in schema.SELECT_REFERENCE_BY_ID
