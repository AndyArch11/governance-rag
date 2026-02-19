"""
Academic ingestion database schema.

SQLite tables for storing academic documents, references, citation edges,
and domain terms with proper indexing and constraints.

**Tables**:
- academic_documents: Primary ingested documents
- academic_references: Cited references with resolution metadata
- academic_citation_edges: Citation relationships (graph edges)
- academic_domain_terms: Extracted domain-specific terminology

**Indexes**: Applied to commonly queried fields (domain, year, resolution status, etc.)

**Schema Version**: 1.0
"""

from typing import Any, Dict

# ============================================================================
# Database Schema (SQL)
# ============================================================================

SCHEMA_VERSION = "1.0"

# Academic Documents Table
CREATE_ACADEMIC_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS academic_documents (
    doc_id TEXT PRIMARY KEY,
    source_url TEXT,
    source_file TEXT,
    title TEXT,
    authors TEXT,                      -- JSON array
    author_orcids TEXT,               -- JSON array of ORCID iDs (nullable)
    year INTEGER CHECK(year >= 1900 AND year <= 2100),
    abstract TEXT,
    keywords TEXT,                     -- JSON array
    primary_domain TEXT,
    secondary_domains TEXT,            -- JSON array
    topic TEXT,
    file_hash TEXT UNIQUE,             -- For change detection
    full_text TEXT,
    status TEXT DEFAULT 'loaded',      -- loaded, parsed, citations_extracted, references_resolved, ingested, failed
    ingest_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    word_count INTEGER DEFAULT 0 CHECK(word_count >= 0),
    citation_count INTEGER DEFAULT 0 CHECK(citation_count >= 0),
    reference_count INTEGER DEFAULT 0 CHECK(reference_count >= 0),
    version INTEGER DEFAULT 1 CHECK(version >= 1)
);

CREATE INDEX IF NOT EXISTS idx_academic_documents_primary_domain ON academic_documents(primary_domain);
CREATE INDEX IF NOT EXISTS idx_academic_documents_year ON academic_documents(year);
CREATE INDEX IF NOT EXISTS idx_academic_documents_file_hash ON academic_documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_academic_documents_status ON academic_documents(status);
CREATE INDEX IF NOT EXISTS idx_academic_documents_ingest_timestamp ON academic_documents(ingest_timestamp);
"""

# Academic References Table
CREATE_ACADEMIC_REFERENCES = """
CREATE TABLE IF NOT EXISTS academic_references (
    ref_id TEXT PRIMARY KEY,
    doi TEXT UNIQUE,
    arxiv_id TEXT UNIQUE,
    title TEXT,
    authors TEXT,                     -- JSON array
    author_orcids TEXT,              -- JSON array of ORCID iDs (nullable)
    year INTEGER CHECK(year >= 1900 AND year <= 2100),
    abstract TEXT,
    venue TEXT,
    volume TEXT,
    issue TEXT,
    pages TEXT,
    raw_citation TEXT,
    citation_format TEXT,             -- harvard, ieee, apa, etc.
    resolved BOOLEAN DEFAULT 0,
    status TEXT DEFAULT 'raw',        -- raw, resolving, resolved, unresolved, ambiguous, error
    metadata_provider TEXT,
    resolved_at TIMESTAMP,
    oa_available BOOLEAN DEFAULT 0,
    oa_url TEXT,
    oa_status TEXT,                   -- gold, hybrid, green, bronze, closed
    pdf_downloaded BOOLEAN DEFAULT 0,
    pdf_local_path TEXT,
    pdf_file_hash TEXT,
    pdf_reused BOOLEAN DEFAULT 0,
    download_failed BOOLEAN DEFAULT 0,
    accessed_at TIMESTAMP,            -- When online content was fetched
    doc_ids TEXT DEFAULT '[]',        -- JSON array of citing doc_ids
    primary_domain TEXT,
    relevance_score REAL DEFAULT 0.0 CHECK(relevance_score >= 0.0 AND relevance_score <= 1.0),
    reference_type TEXT DEFAULT 'academic',  -- academic, preprint, news, blog, online, report
    quality_score REAL DEFAULT 0.0 CHECK(quality_score >= 0.0 AND quality_score <= 1.0),
    paywall_detected BOOLEAN DEFAULT 0,
    link_status TEXT DEFAULT 'available',    -- available, stale_404, stale_timeout, stale_moved
    link_checked_at TIMESTAMP,
    check_frequency_days INTEGER DEFAULT 30 CHECK(check_frequency_days >= 1),
    consecutive_failures INTEGER DEFAULT 0 CHECK(consecutive_failures >= 0),
    last_success_at TIMESTAMP,
    alternative_urls TEXT DEFAULT '[]',      -- JSON array of backup URLs
    archived_url TEXT,
    archive_timestamp TIMESTAMP,
    content_hash TEXT,                       -- SHA256 for content change detection
    last_content_check TIMESTAMP,
    content_changed BOOLEAN DEFAULT 0,
    venue_name TEXT,                         -- Structured venue name for visualisation
    venue_type TEXT,                         -- journal, conference, preprint, web
    venue_rank TEXT,                         -- Q1-Q4 or A*, A, B, C
    impact_factor REAL,
    citation_count INTEGER DEFAULT 0 CHECK(citation_count >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1 CHECK(version >= 1)
);

CREATE INDEX IF NOT EXISTS idx_academic_references_doi ON academic_references(doi);
CREATE INDEX IF NOT EXISTS idx_academic_references_arxiv_id ON academic_references(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_academic_references_resolved ON academic_references(resolved);
CREATE INDEX IF NOT EXISTS idx_academic_references_status ON academic_references(status);
CREATE INDEX IF NOT EXISTS idx_academic_references_primary_domain ON academic_references(primary_domain);
CREATE INDEX IF NOT EXISTS idx_academic_references_reference_type ON academic_references(reference_type);
CREATE INDEX IF NOT EXISTS idx_academic_references_quality_score ON academic_references(quality_score);
CREATE INDEX IF NOT EXISTS idx_academic_references_link_status ON academic_references(link_status);
CREATE INDEX IF NOT EXISTS idx_academic_references_year ON academic_references(year);
CREATE INDEX IF NOT EXISTS idx_academic_references_metadata_provider ON academic_references(metadata_provider);
CREATE INDEX IF NOT EXISTS idx_academic_references_created_at ON academic_references(created_at);
CREATE INDEX IF NOT EXISTS idx_academic_references_venue_type ON academic_references(venue_type);
CREATE INDEX IF NOT EXISTS idx_academic_references_venue_rank ON academic_references(venue_rank);
CREATE INDEX IF NOT EXISTS idx_academic_references_content_changed ON academic_references(content_changed);
CREATE INDEX IF NOT EXISTS idx_academic_references_last_content_check ON academic_references(last_content_check);
"""

# Citation Edges Table (for graph representation)
CREATE_ACADEMIC_CITATION_EDGES = """
CREATE TABLE IF NOT EXISTS academic_citation_edges (
    from_id TEXT,                      -- doc_id or ref_id
    to_id TEXT,                        -- doc_id or ref_id
    edge_type TEXT DEFAULT 'direct',   -- direct, indirect, extends, contradicts, refutes
    relationship_type TEXT DEFAULT 'cites',
    depth INTEGER DEFAULT 1 CHECK(depth >= 1),
    mention_text TEXT,
    mention_position INTEGER DEFAULT 0 CHECK(mention_position >= 0),
    is_inline BOOLEAN DEFAULT 0,
    cited_as TEXT,
    strength_score REAL DEFAULT 1.0 CHECK(strength_score >= 0.0 AND strength_score <= 1.0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (from_id, to_id)
);

CREATE INDEX IF NOT EXISTS idx_academic_citation_edges_from_id ON academic_citation_edges(from_id);
CREATE INDEX IF NOT EXISTS idx_academic_citation_edges_to_id ON academic_citation_edges(to_id);
CREATE INDEX IF NOT EXISTS idx_academic_citation_edges_edge_type ON academic_citation_edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_academic_citation_edges_depth ON academic_citation_edges(depth);
"""

# Domain Terms Table
CREATE_ACADEMIC_DOMAIN_TERMS = """
CREATE TABLE IF NOT EXISTS academic_domain_terms (
    term_id TEXT PRIMARY KEY,
    term TEXT,
    domain TEXT,
    frequency INTEGER DEFAULT 0 CHECK(frequency >= 0),
    doc_ids TEXT DEFAULT '[]',        -- JSON array
    domain_relevance_score REAL DEFAULT 0.0 CHECK(domain_relevance_score >= 0.0 AND domain_relevance_score <= 1.0),
    tf_idf_score REAL DEFAULT 0.0 CHECK(tf_idf_score >= 0.0),
    related_terms TEXT DEFAULT '[]',  -- JSON array
    broader_terms TEXT DEFAULT '[]',  -- JSON array
    narrower_terms TEXT DEFAULT '[]', -- JSON array
    term_type TEXT DEFAULT 'concept', -- concept, method, tool, dataset, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (term, domain)
);

CREATE INDEX IF NOT EXISTS idx_academic_domain_terms_domain ON academic_domain_terms(domain);
CREATE INDEX IF NOT EXISTS idx_academic_domain_terms_term ON academic_domain_terms(term);
CREATE INDEX IF NOT EXISTS idx_academic_domain_terms_term_type ON academic_domain_terms(term_type);
CREATE INDEX IF NOT EXISTS idx_academic_domain_terms_domain_relevance ON academic_domain_terms(domain_relevance_score);
"""

# Visualisation State Table (for persistent graph layouts)
CREATE_ACADEMIC_VISUALISATION_STATE = """
CREATE TABLE IF NOT EXISTS academic_visualisation_state (
    ref_id TEXT PRIMARY KEY,
    node_x REAL,
    node_y REAL,
    is_collapsed BOOLEAN DEFAULT 0,
    user_notes TEXT,
    trust_override REAL CHECK(trust_override IS NULL OR (trust_override >= 0.0 AND trust_override <= 1.0)),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ref_id) REFERENCES academic_references(ref_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_academic_visualisation_state_updated_at ON academic_visualisation_state(updated_at);
"""

# Migration tracking table
CREATE_MIGRATIONS = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

INSERT OR IGNORE INTO schema_migrations (version, description)
VALUES ('1.0', 'Initial schema with documents, references, edges, and domain terms');
"""

# ============================================================================
# Migration Functions
# ============================================================================

ALL_MIGRATIONS = [
    CREATE_ACADEMIC_DOCUMENTS,
    CREATE_ACADEMIC_REFERENCES,
    CREATE_ACADEMIC_CITATION_EDGES,
    CREATE_ACADEMIC_DOMAIN_TERMS,
    CREATE_ACADEMIC_VISUALISATION_STATE,
    CREATE_MIGRATIONS,
]


def get_schema_creation_sql() -> str:
    """
    Get complete SQL for schema creation.

    Returns:
        SQL string containing all table creation statements
    """
    return "\n\n".join(ALL_MIGRATIONS)


def get_table_creation_sqls() -> Dict[str, str]:
    """
    Get individual table creation SQL statements.

    Returns:
        Dict mapping table name to SQL statement
    """
    return {
        "academic_documents": CREATE_ACADEMIC_DOCUMENTS,
        "academic_references": CREATE_ACADEMIC_REFERENCES,
        "academic_citation_edges": CREATE_ACADEMIC_CITATION_EDGES,
        "academic_domain_terms": CREATE_ACADEMIC_DOMAIN_TERMS,
        "academic_visualisation_state": CREATE_ACADEMIC_VISUALISATION_STATE,
    }


# ============================================================================
# Schema Validation & Info
# ============================================================================

SCHEMA_INFO = {
    "version": SCHEMA_VERSION,
    "tables": {
        "academic_documents": {
            "description": "Primary ingested academic documents",
            "primary_key": "doc_id",
            "unique_fields": ["file_hash"],
            "indexed_fields": ["primary_domain", "year", "file_hash", "status", "ingest_timestamp"],
        },
        "academic_references": {
            "description": "Citations with resolution metadata and OA status",
            "primary_key": "ref_id",
            "unique_fields": ["doi", "arxiv_id"],
            "indexed_fields": [
                "doi",
                "arxiv_id",
                "resolved",
                "status",
                "primary_domain",
                "reference_type",
                "quality_score",
                "link_status",
                "year",
                "metadata_provider",
                "created_at",
            ],
        },
        "academic_citation_edges": {
            "description": "Citation relationships forming directed graph",
            "primary_key": ["from_id", "to_id"],
            "unique_fields": [],
            "indexed_fields": ["from_id", "to_id", "edge_type", "depth"],
        },
        "academic_domain_terms": {
            "description": "Domain-specific terminology with frequency analysis",
            "primary_key": "term_id",
            "unique_fields": ["(term, domain)"],
            "indexed_fields": ["domain", "term", "term_type", "domain_relevance_score"],
        },
        "academic_visualisation_state": {
            "description": "User interaction state for citation graph visualisation",
            "primary_key": "ref_id",
            "unique_fields": [],
            "indexed_fields": ["updated_at"],
        },
    },
    "constraints": {
        "year_bounds": "1900 <= year <= 2100",
        "score_bounds": "0.0 <= score <= 1.0",
        "positive_integers": "count >= 0",
        "depth_minimum": "depth >= 1",
    },
}


def get_schema_info() -> Dict[str, Any]:
    """
    Get comprehensive schema information.

    Returns:
        Dict containing version, tables, indexed fields, and constraints
    """
    return SCHEMA_INFO


# ============================================================================
# SQL Query Templates
# ============================================================================

INSERT_DOCUMENT = """
INSERT INTO academic_documents (
    doc_id, source_url, source_file, title, authors, author_orcids, year,
    abstract, keywords, primary_domain, secondary_domains, topic, file_hash,
    full_text, status, word_count, citation_count, reference_count
) VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
)
"""

INSERT_REFERENCE = """
INSERT INTO academic_references (
    ref_id, doi, arxiv_id, title, authors, author_orcids, year, abstract, venue,
    volume, issue, pages, raw_citation, citation_format, resolved, status,
    metadata_provider, resolved_at, oa_available, oa_url, oa_status,
    pdf_downloaded, pdf_local_path, pdf_file_hash, pdf_reused, doc_ids,
    primary_domain, relevance_score, reference_type, quality_score,
    paywall_detected, link_status, venue_name, venue_type, venue_rank, impact_factor,
    alternative_urls, archived_url, content_hash
) VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
)
"""

INSERT_CITATION_EDGE = """
INSERT INTO academic_citation_edges (
    from_id, to_id, edge_type, relationship_type, depth, mention_text,
    mention_position, is_inline, cited_as, strength_score
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_DOMAIN_TERM = """
INSERT INTO academic_domain_terms (
    term_id, term, domain, frequency, doc_ids, domain_relevance_score,
    tf_idf_score, related_terms, broader_terms, narrower_terms, term_type
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Query templates
SELECT_DOCUMENT_BY_ID = "SELECT * FROM academic_documents WHERE doc_id = ?"
SELECT_REFERENCE_BY_ID = "SELECT * FROM academic_references WHERE ref_id = ?"
SELECT_REFERENCE_BY_DOI = "SELECT * FROM academic_references WHERE doi = ?"
SELECT_REFERENCES_BY_DOMAIN = (
    "SELECT * FROM academic_references WHERE primary_domain = ? ORDER BY quality_score DESC"
)
SELECT_UNRESOLVED_REFERENCES = """
SELECT * FROM academic_references
WHERE resolved = 0 AND status IN ('raw', 'unresolved', 'error')
ORDER BY created_at ASC
LIMIT ?
"""
SELECT_CITATION_EDGES_FROM = "SELECT * FROM academic_citation_edges WHERE from_id = ?"
SELECT_CITATION_EDGES_TO = "SELECT * FROM academic_citation_edges WHERE to_id = ?"
SELECT_DOMAIN_TERMS_BY_DOMAIN = (
    "SELECT * FROM academic_domain_terms WHERE domain = ? ORDER BY domain_relevance_score DESC"
)


__all__ = [
    "SCHEMA_VERSION",
    "CREATE_ACADEMIC_DOCUMENTS",
    "CREATE_ACADEMIC_REFERENCES",
    "CREATE_ACADEMIC_CITATION_EDGES",
    "CREATE_ACADEMIC_DOMAIN_TERMS",
    "CREATE_ACADEMIC_VISUALISATION_STATE",
    "CREATE_MIGRATIONS",
    "ALL_MIGRATIONS",
    "get_schema_creation_sql",
    "get_table_creation_sqls",
    "get_schema_info",
    "SCHEMA_INFO",
    "INSERT_DOCUMENT",
    "INSERT_REFERENCE",
    "INSERT_CITATION_EDGE",
    "INSERT_DOMAIN_TERM",
    "SELECT_DOCUMENT_BY_ID",
    "SELECT_REFERENCE_BY_ID",
    "SELECT_REFERENCE_BY_DOI",
    "SELECT_REFERENCES_BY_DOMAIN",
    "SELECT_UNRESOLVED_REFERENCES",
    "SELECT_CITATION_EDGES_FROM",
    "SELECT_CITATION_EDGES_TO",
    "SELECT_DOMAIN_TERMS_BY_DOMAIN",
]
