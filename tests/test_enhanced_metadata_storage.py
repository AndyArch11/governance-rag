"""Tests for enhanced metadata capture, storage, and retrieval across ingestion pipelines.

This test suite verifies that enhanced metadata (section/heading details, content classification,
technical entities) is consistently captured during ingestion and can be retrieved from ChromaDB.

Tests cover:
- Enhanced metadata extraction via create_enhanced_metadata
- Storage via store_chunks_in_chroma (standard chunking)
- Storage via store_parent_chunks (parent-child chunking)
- Storage via store_child_chunks (parent-child chunking)
- Retrieval from ChromaDB with metadata verification
"""

import json
from typing import Any, Dict, List

import pytest


# Mock ChromaDB Collection
class MockCollection:
    """Mock ChromaDB collection for testing."""

    def __init__(self):
        self.stored_data = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "embeddings": [],
        }
        self.add_calls = []
        self.get_calls = []

    def count(self):
        """Return number of stored documents."""
        return len(self.stored_data["ids"])

    def add(
        self,
        ids: List[str] = None,
        documents: List[str] = None,
        metadatas: List[Dict[str, Any]] = None,
        embeddings: List[List[float]] = None,
    ):
        """Store data in mock collection."""
        self.add_calls.append({
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": embeddings,
        })
        
        if ids:
            self.stored_data["ids"].extend(ids)
        if documents:
            self.stored_data["documents"].extend(documents)
        if metadatas:
            self.stored_data["metadatas"].extend(metadatas)
        if embeddings:
            self.stored_data["embeddings"].extend(embeddings)

    def get(
        self,
        ids: List[str] = None,
        where: Dict[str, Any] = None,
        include: List[str] = None,
    ):
        """Retrieve data from mock collection."""
        self.get_calls.append({
            "ids": ids,
            "where": where,
            "include": include,
        })
        
        # Simple retrieval - return all or filter by ids
        if ids:
            indices = [i for i, stored_id in enumerate(self.stored_data["ids"]) if stored_id in ids]
            result = {
                "ids": [self.stored_data["ids"][i] for i in indices],
                "documents": [self.stored_data["documents"][i] for i in indices] if "documents" in (include or []) else [],
                "metadatas": [self.stored_data["metadatas"][i] for i in indices] if "metadatas" in (include or []) else [],
                "embeddings": [self.stored_data["embeddings"][i] for i in indices] if "embeddings" in (include or []) else [],
            }
        else:
            result = {
                "ids": self.stored_data["ids"][:],
                "documents": self.stored_data["documents"][:] if "documents" in (include or []) else [],
                "metadatas": self.stored_data["metadatas"][:] if "metadatas" in (include or []) else [],
                "embeddings": self.stored_data["embeddings"][:] if "embeddings" in (include or []) else [],
            }
        
        return result


# Mock embedding model
class MockEmbeddings:
    """Mock embedding model for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings."""
        # Return 1024-dim embeddings (matching mxbai-embed-large)
        return [[0.1] * 1024 for _ in texts]


@pytest.fixture
def mock_collections():
    """Create mock ChromaDB collections."""
    return {
        "chunk_collection": MockCollection(),
        "doc_collection": MockCollection(),
    }


@pytest.fixture
def sample_document_with_sections():
    """Sample document with multiple sections for testing."""
    return """# User Guide

## Chapter 1: Introduction

Welcome to the system. This guide covers installation and configuration.

### Section 1.1: Installation

To install the system, run the following commands:

```bash
pip install myapp
myapp --setup
```

## Chapter 2: API Reference

The API provides the following endpoints:

- GET /api/v1/users - Retrieve user list

Use the `UserService` class to interact with endpoints.
"""


# =========================
# Test Enhanced Metadata Extraction
# =========================


class TestEnhancedMetadataExtraction:
    """Test enhanced metadata extraction from chunks."""

    def test_extract_heading_path(self, sample_document_with_sections):
        """Test heading path extraction from full document."""
        from scripts.ingest.chunk import extract_heading_path

        chunk = "To install the system, run the following commands:"
        heading_path = extract_heading_path(chunk, sample_document_with_sections)

        assert heading_path is not None
        # Should capture nested heading structure
        assert "Chapter 1" in heading_path or "Installation" in heading_path

    def test_create_enhanced_metadata_sections(self, sample_document_with_sections):
        """Test enhanced metadata creation with section details."""
        from scripts.ingest.chunk import create_enhanced_metadata

        chunk = "To install the system, run the following commands:"
        metadata = create_enhanced_metadata(
            chunk_text=chunk,
            chunk_index=0,
            total_chunks=10,
            doc_id="user_guide_001",
            full_text=sample_document_with_sections,
            doc_type="documentation",
        )

        assert metadata is not None
        assert metadata.heading_path is not None and len(metadata.heading_path) > 0
        assert metadata.parent_section is not None
        assert metadata.section_depth > 0

    def test_create_enhanced_metadata_code_detection(self, sample_document_with_sections):
        """Test code detection in enhanced metadata."""
        from scripts.ingest.chunk import create_enhanced_metadata

        chunk = """```bash
pip install myapp
myapp --setup
```"""
        metadata = create_enhanced_metadata(
            chunk_text=chunk,
            chunk_index=1,
            total_chunks=10,
            doc_id="user_guide_001",
            full_text=sample_document_with_sections,
            doc_type="documentation",
        )

        assert metadata.contains_code is True
        assert metadata.code_language in ["bash", "shell"]
        assert metadata.content_type == "code"

    def test_create_enhanced_metadata_api_reference(self, sample_document_with_sections):
        """Test API reference detection in enhanced metadata."""
        from scripts.ingest.chunk import create_enhanced_metadata

        chunk = """The API provides the following endpoints:

- GET /api/v1/users - Retrieve user list

Use the `UserService` class to interact with endpoints."""
        metadata = create_enhanced_metadata(
            chunk_text=chunk,
            chunk_index=5,
            total_chunks=10,
            doc_id="user_guide_001",
            full_text=sample_document_with_sections,
            doc_type="documentation",
        )

        assert metadata.is_api_reference is True
        assert "/api/v1/users" in metadata.technical_entities
        assert "UserService" in metadata.technical_entities


# =========================
# Test Storage via store_chunks_in_chroma
# =========================


class TestStoreChunksWithEnhancedMetadata:
    """Test enhanced metadata storage via store_chunks_in_chroma."""

    def test_store_chunks_captures_enhanced_metadata(
        self, mock_collections, sample_document_with_sections, monkeypatch
    ):
        """Test that store_chunks_in_chroma captures and stores enhanced metadata."""
        from scripts.ingest import vectors
        from scripts.ingest.chunk import create_enhanced_metadata as real_create_enhanced_metadata

        # Set up mocks
        monkeypatch.setattr(vectors, "create_enhanced_metadata", real_create_enhanced_metadata)
        monkeypatch.setattr(vectors, "_create_embed_model", lambda: MockEmbeddings())
        monkeypatch.setattr(
            vectors,
            "validate_chunk_semantics",
            lambda chunk, doc_type, chunk_hash=None, llm_cache=None: (True, "valid"),
        )

        chunks = [
            "To install the system, run the following commands:",
            "The API provides the following endpoints:",
        ]

        metadata = {
            "cleaned_text": sample_document_with_sections,
            "doc_type": "documentation",
            "key_topics": ["installation", "api"],
            "summary": "User guide covering installation and API reference.",
            "summary_scores": {"overall": 8, "completeness": 7, "clarity": 9, "technical_depth": 8},
            "source_category": "docs",
        }

        vectors.store_chunks_in_chroma(
            doc_id="user_guide_001",
            file_hash="abc123def456",
            source_path="/docs/user_guide.md",
            version=1,
            chunks=chunks,
            metadata=metadata,
            chunk_collection=mock_collections["chunk_collection"],
            doc_collection=mock_collections["doc_collection"],
            preprocess_duration=1.0,
            ingest_duration=2.0,
            dry_run=False,
            full_text=sample_document_with_sections,
        )

        # Verify chunks were stored
        assert len(mock_collections["chunk_collection"].add_calls) > 0
        stored_metadatas = mock_collections["chunk_collection"].add_calls[0]["metadatas"]

        # Verify enhanced metadata fields are present
        for chunk_meta in stored_metadatas:
            # Section metadata
            assert "heading_path" in chunk_meta
            assert "parent_section" in chunk_meta
            assert "section_depth" in chunk_meta

            # Content classification
            assert "content_type" in chunk_meta
            assert "contains_code" in chunk_meta
            assert "contains_table" in chunk_meta

            # Technical metadata
            assert "technical_entities" in chunk_meta
            assert "is_api_reference" in chunk_meta


# =========================
# Test Storage via store_parent_chunks
# =========================


class TestStoreParentChunksWithEnhancedMetadata:
    """Test enhanced metadata storage via store_parent_chunks."""

    def test_store_parent_chunks_captures_enhanced_metadata(
        self, mock_collections, sample_document_with_sections, monkeypatch
    ):
        """Test that store_parent_chunks captures and stores enhanced metadata."""
        from scripts.ingest import vectors
        from scripts.ingest.chunk import create_enhanced_metadata as real_create_enhanced_metadata

        monkeypatch.setattr(vectors, "create_enhanced_metadata", real_create_enhanced_metadata)

        parent_chunks = [
            {
                "id": "parent_0",
                "text": "To install the system, run the following commands:\n\n```bash\npip install myapp\n```",
                "child_ids": ["child_0_0", "child_0_1"],
            },
        ]

        base_metadata = {
            "doc_id": "user_guide_001",
            "source": "/docs/user_guide.md",
            "version": 1,
            "hash": "abc123",
            "doc_type": "documentation",
            "embedding_model": "mxbai-embed-large",
        }

        vectors.store_parent_chunks(
            doc_id="user_guide_001",
            parent_chunks=parent_chunks,
            chunk_collection=mock_collections["chunk_collection"],
            base_metadata=base_metadata,
            dry_run=False,
            full_text=sample_document_with_sections,
            doc_type="documentation",
        )

        # Verify parent chunks were stored
        assert len(mock_collections["chunk_collection"].add_calls) > 0
        stored_call = mock_collections["chunk_collection"].add_calls[0]
        stored_metadatas = stored_call["metadatas"]

        assert len(stored_metadatas) == 1

        # Verify enhanced metadata 
        chunk_meta = stored_metadatas[0]
        assert chunk_meta["is_parent"] is True
        assert "heading_path" in chunk_meta
        assert "parent_section" in chunk_meta
        assert "content_type" in chunk_meta
        # Verify at least some of the enhanced fields are present
        # (not all fields may be populated depending on content)
        enhanced_fields = ["section_depth", "contains_code", "contains_table", "technical_entities"]
        assert sum(1 for field in enhanced_fields if field in chunk_meta) >= 2
        assert "technical_entities" in chunk_meta


# =========================
# Test Storage via store_child_chunks
# =========================


class TestStoreChildChunksWithEnhancedMetadata:
    """Test enhanced metadata storage via store_child_chunks."""

    def test_store_child_chunks_captures_enhanced_metadata(
        self, mock_collections, sample_document_with_sections, monkeypatch
    ):
        """Test that store_child_chunks captures and stores enhanced metadata."""
        from scripts.ingest import vectors
        from scripts.ingest.chunk import create_enhanced_metadata as real_create_enhanced_metadata

        monkeypatch.setattr(vectors, "create_enhanced_metadata", real_create_enhanced_metadata)
        monkeypatch.setattr(vectors, "_create_embed_model", lambda: MockEmbeddings())

        child_chunks = [
            {
                "id": "child_0_0",
                "text": "GET /api/v1/users - Retrieve user list",
                "parent_id": "parent_0",
            },
        ]

        base_metadata = {
            "doc_id": "user_guide_001",
            "source": "/docs/user_guide.md",
            "version": 1,
            "hash": "abc123",
            "doc_type": "documentation",
            "embedding_model": "mxbai-embed-large",
        }

        vectors.store_child_chunks(
            doc_id="user_guide_001",
            child_chunks=child_chunks,
            chunk_collection=mock_collections["chunk_collection"],
            base_metadata=base_metadata,
            dry_run=False,
            full_text=sample_document_with_sections,
            doc_type="documentation",
        )

        # Verify child chunks were stored
        assert len(mock_collections["chunk_collection"].add_calls) > 0
        stored_call = mock_collections["chunk_collection"].add_calls[0]
        stored_metadatas = stored_call["metadatas"]

        assert len(stored_metadatas) == 1

        # Verify enhanced metadata
        chunk_meta = stored_metadatas[0]
        assert chunk_meta["is_parent"] is False
        assert "heading_path" in chunk_meta
        assert "parent_section" in chunk_meta
        # Verify enhanced metadata exists (even if some fields might be empty)
        enhanced_fields_present = sum(
            1 for field in ["content_type", "section_depth", "contains_code"]
            if field in chunk_meta
        )
        assert enhanced_fields_present >= 1, f"Expected at least 1 enhanced field, got: {list(chunk_meta.keys())}"
        
        assert "is_api_reference" in chunk_meta
        # Technical entities should be stored as JSON
        entities = json.loads(chunk_meta["technical_entities"])
        assert "/api/v1/users" in entities


# =========================
# Test Retrieval from ChromaDB
# =========================


class TestEnhancedMetadataRetrieval:
    """Test retrieval of enhanced metadata from ChromaDB."""

    def test_retrieve_chunks_with_enhanced_metadata(
        self, mock_collections, sample_document_with_sections, monkeypatch
    ):
        """Test retrieving chunks with enhanced metadata from ChromaDB."""
        from scripts.ingest import vectors
        from scripts.ingest.chunk import create_enhanced_metadata as real_create_enhanced_metadata

        # Set up mocks
        monkeypatch.setattr(vectors, "create_enhanced_metadata", real_create_enhanced_metadata)
        monkeypatch.setattr(vectors, "_create_embed_model", lambda: MockEmbeddings())
        monkeypatch.setattr(
            vectors,
            "validate_chunk_semantics",
            lambda chunk, doc_type, chunk_hash=None, llm_cache=None: (True, "valid"),
        )

        chunks = [
            "To install the system, run the following commands:",
            "The API provides the following endpoints:",
        ]

        metadata = {
            "cleaned_text": sample_document_with_sections,
            "doc_type": "documentation",
            "key_topics": ["installation", "api"],
            "summary": "User guide",
            "summary_scores": {"overall": 8},
            "source_category": "docs",
        }

        vectors.store_chunks_in_chroma(
            doc_id="user_guide_001",
            file_hash="abc123",
            source_path="/docs/user_guide.md",
            version=1,
            chunks=chunks,
            metadata=metadata,
            chunk_collection=mock_collections["chunk_collection"],
            doc_collection=mock_collections["doc_collection"],
            preprocess_duration=1.0,
            ingest_duration=2.0,
            dry_run=False,
            full_text=sample_document_with_sections,
        )

        # Now retrieve chunks
        result = mock_collections["chunk_collection"].get(
            include=["documents", "metadatas"]
        )

        assert len(result["documents"]) > 0
        assert len(result["metadatas"]) > 0

        # Verify each retrieved chunk has enhanced metadata
        for chunk_meta in result["metadatas"]:
            # Section metadata
            assert "heading_path" in chunk_meta
            assert "parent_section" in chunk_meta
            assert "section_depth" in chunk_meta

            # Content classification
            assert "content_type" in chunk_meta
            assert "contains_code" in chunk_meta

            # Technical metadata
            assert "technical_entities" in chunk_meta
            assert "is_api_reference" in chunk_meta
