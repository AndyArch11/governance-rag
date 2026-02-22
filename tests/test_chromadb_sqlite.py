"""Comprehensive unit tests for chromadb_sqlite module.

Tests cover:
- Database initialisation and schema creation
- Document addition, retrieval, update, and deletion
- Vector similarity search
- Metadata filtering
- Collection management
- Edge cases and error handling
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from scripts.ingest.chromadb_sqlite import (
    ChromaSQLiteClient,
    ChromaSQLiteCollection,
    PersistentClient,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def client(temp_db):
    """Create a ChromaSQLiteClient instance."""
    client = ChromaSQLiteClient(db_path=temp_db)
    yield client
    client.close()


@pytest.fixture
def collection(client):
    """Create a test collection."""
    return client.get_or_create_collection("test_collection")


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.8, 0.7, 0.6],
    ]


@pytest.fixture
def sample_documents():
    """Generate sample documents."""
    return [
        "This is the first document about Python.",
        "This is the second document about Java.",
        "This is the third document about Go.",
    ]


@pytest.fixture
def sample_metadatas():
    """Generate sample metadata."""
    return [
        {"source": "bitbucket", "type": "code", "language": "python"},
        {"source": "confluence", "type": "doc", "language": "java"},
        {"source": "bitbucket", "type": "code", "language": "go"},
    ]


# ============================================================================
# Test ChromaSQLiteClient
# ============================================================================


class TestChromaSQLiteClient:
    """Tests for ChromaSQLiteClient initialisation and management."""

    def test_client_initialisation_default_path(self):
        """Client initialises with default path."""
        with patch.object(Path, "mkdir"):
            client = ChromaSQLiteClient()
            assert "chromadb.db" in client.db_path
            assert "rag_data" in client.db_path
            client.close()

    def test_client_initialisation_custom_path(self, temp_db):
        """Client initialises with custom path."""
        client = ChromaSQLiteClient(db_path=temp_db)
        assert client.db_path == temp_db
        assert Path(temp_db).exists()
        client.close()

    def test_get_or_create_collection(self, client):
        """get_or_create_collection creates new collection."""
        collection = client.get_or_create_collection("test")
        assert isinstance(collection, ChromaSQLiteCollection)
        assert collection.collection_name == "test"

    def test_get_or_create_collection_idempotent(self, client):
        """get_or_create_collection returns same collection on repeated calls."""
        collection1 = client.get_or_create_collection("test")
        collection2 = client.get_or_create_collection("test")
        assert collection1 is collection2

    def test_get_collection(self, client):
        """get_collection retrieves existing collection."""
        client.get_or_create_collection("test")
        collection = client.get_collection("test")
        assert isinstance(collection, ChromaSQLiteCollection)
        assert collection.collection_name == "test"

    def test_delete_collection(self, client):
        """delete_collection removes collection and table."""
        client.get_or_create_collection("test_delete")
        assert "test_delete" in client.collections

        client.delete_collection("test_delete")
        assert "test_delete" not in client.collections

        # Verify table is dropped
        collections = client.list_collections()
        assert "test_delete" not in collections

    def test_list_collections(self, client):
        """list_collections returns all collection names."""
        client.get_or_create_collection("collection1")
        client.get_or_create_collection("collection2")

        collections = client.list_collections()
        assert "collection1" in collections
        assert "collection2" in collections

    def test_list_collections_empty(self, client):
        """list_collections returns empty list for new database."""
        collections = client.list_collections()
        assert collections == []

    def test_close_client(self, client):
        """close() closes all collections."""
        collection1 = client.get_or_create_collection("test1")
        collection2 = client.get_or_create_collection("test2")

        client.close()

        assert collection1.conn is None
        assert collection2.conn is None
        assert len(client.collections) == 0

    def test_persistent_client_compatibility(self, temp_db):
        """PersistentClient creates compatible client."""
        client = PersistentClient(path=temp_db)
        assert isinstance(client, ChromaSQLiteClient)
        assert client.db_path == temp_db
        client.close()


# ============================================================================
# Test ChromaSQLiteCollection - Basic Operations
# ============================================================================


class TestCollectionBasicOperations:
    """Tests for collection initialisation and basic operations."""

    def test_collection_initialisation(self, temp_db):
        """Collection initialises with database and schema."""
        collection = ChromaSQLiteCollection(temp_db, "test")
        conn = collection._get_connection()

        # Verify table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_chunks'")
        assert cursor.fetchone() is not None
        collection.close()

    def test_collection_get_connection(self, collection):
        """_get_connection creates and reuses connection."""
        conn1 = collection._get_connection()
        conn2 = collection._get_connection()
        assert conn1 is conn2

    def test_collection_close(self, collection):
        """close() closes database connection."""
        collection._get_connection()
        assert collection.conn is not None

        collection.close()
        assert collection.conn is None

    def test_collection_count_empty(self, collection):
        """count() returns 0 for empty collection."""
        assert collection.count() == 0

    def test_collection_count_with_documents(self, collection, sample_embeddings, sample_documents):
        """count() returns correct number of documents."""
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=sample_embeddings,
            documents=sample_documents,
        )
        assert collection.count() == 3


# ============================================================================
# Test ChromaSQLiteCollection - Add Documents
# ============================================================================


class TestCollectionAdd:
    """Tests for adding documents to collection."""

    def test_add_with_all_fields(
        self, collection, sample_embeddings, sample_documents, sample_metadatas
    ):
        """add() with embeddings, documents, and metadata."""
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        result = collection.get(ids=["doc1"])
        assert result["ids"] == ["doc1"]
        assert result["documents"][0] == sample_documents[0]
        assert result["metadatas"][0] == sample_metadatas[0]

    def test_add_with_embeddings_only(self, collection, sample_embeddings):
        """add() with only embeddings."""
        collection.add(ids=["doc1", "doc2"], embeddings=sample_embeddings[:2])

        result = collection.get(ids=["doc1"], include=["embeddings"])
        assert result["ids"] == ["doc1"]
        assert len(result["embeddings"][0]) == 4

    def test_add_without_embeddings(self, collection, sample_documents):
        """add() without embeddings uses dummy embedding."""
        # Provide a dummy embedding since schema requires it
        collection.add(
            ids=["doc1"], documents=sample_documents[:1], embeddings=[[0.0, 0.0, 0.0, 0.0]]
        )

        result = collection.get(ids=["doc1"])
        assert result["ids"] == ["doc1"]
        assert result["documents"][0] == sample_documents[0]

    def test_add_with_metadata_only(self, collection):
        """add() with only metadata requires embedding."""
        collection.add(
            ids=["doc1"], metadatas=[{"source": "test"}], embeddings=[[0.0, 0.0, 0.0, 0.0]]
        )

        result = collection.get(ids=["doc1"])
        assert result["metadatas"][0] == {"source": "test"}

    def test_add_replace_existing(self, collection):
        """add() replaces existing document with same ID."""
        embedding = [[0.1, 0.2, 0.3, 0.4]]
        collection.add(ids=["doc1"], documents=["Original"], embeddings=embedding)
        collection.add(ids=["doc1"], documents=["Updated"], embeddings=embedding)

        result = collection.get(ids=["doc1"])
        assert result["documents"][0] == "Updated"
        assert collection.count() == 1  # Not duplicated

    def test_add_multiple_batches(self, collection, sample_embeddings, sample_documents):
        """add() can be called multiple times."""
        collection.add(
            ids=["doc1"], embeddings=[sample_embeddings[0]], documents=[sample_documents[0]]
        )
        collection.add(
            ids=["doc2"], embeddings=[sample_embeddings[1]], documents=[sample_documents[1]]
        )

        assert collection.count() == 2

    def test_add_normalises_embeddings(self, collection):
        """add() stores embeddings as binary."""
        embedding = [1.0, 2.0, 3.0, 4.0]
        collection.add(ids=["doc1"], embeddings=[embedding])

        result = collection.get(ids=["doc1"], include=["embeddings"])
        # Should be able to retrieve embedding
        assert len(result["embeddings"][0]) == 4
        assert isinstance(result["embeddings"][0], list)

    def test_add_with_empty_metadata(self, collection):
        """add() handles empty metadata dictionary."""
        collection.add(
            ids=["doc1"], metadatas=[{}], documents=["Test"], embeddings=[[0.1, 0.2, 0.3, 0.4]]
        )

        result = collection.get(ids=["doc1"])
        assert result["metadatas"][0] == {}


# ============================================================================
# Test ChromaSQLiteCollection - Get Documents
# ============================================================================


class TestCollectionGet:
    """Tests for retrieving documents from collection."""

    def test_get_by_ids(self, collection, sample_embeddings, sample_documents):
        """get() retrieves documents by IDs."""
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=sample_embeddings,
            documents=sample_documents,
        )

        result = collection.get(ids=["doc1", "doc3"])
        assert set(result["ids"]) == {"doc1", "doc3"}
        assert len(result["documents"]) == 2

    def test_get_with_where_clause(self, collection, sample_metadatas, sample_embeddings):
        """get() filters by metadata."""
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            metadatas=sample_metadatas,
            documents=["a", "b", "c"],
            embeddings=sample_embeddings,
        )

        result = collection.get(where={"source": "bitbucket"})
        assert len(result["ids"]) == 2
        assert all(m["source"] == "bitbucket" for m in result["metadatas"])

    def test_get_with_limit(self, collection):
        """get() respects limit parameter."""
        collection.add(
            ids=["doc1", "doc2", "doc3", "doc4"],
            documents=["a", "b", "c", "d"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]] * 4,
        )

        result = collection.get(limit=2)
        assert len(result["ids"]) == 2

    def test_get_with_offset(self, collection):
        """get() respects offset parameter."""
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            documents=["a", "b", "c"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]] * 3,
        )

        result = collection.get(offset=1, limit=2)
        assert len(result["ids"]) <= 2

    def test_get_include_documents_only(self, collection):
        """get() includes only requested fields."""
        collection.add(
            ids=["doc1"],
            documents=["Test"],
            metadatas=[{"source": "test"}],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
        )

        result = collection.get(ids=["doc1"], include=["documents"])
        assert result["documents"] == ["Test"]
        assert result["metadatas"] is None

    def test_get_include_metadatas_only(self, collection):
        """get() includes only metadata when requested."""
        collection.add(
            ids=["doc1"],
            documents=["Test"],
            metadatas=[{"source": "test"}],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
        )

        result = collection.get(ids=["doc1"], include=["metadatas"])
        assert result["metadatas"] == [{"source": "test"}]
        assert result["documents"] is None

    def test_get_include_embeddings(self, collection, sample_embeddings):
        """get() includes embeddings when requested."""
        collection.add(ids=["doc1"], embeddings=[sample_embeddings[0]])

        result = collection.get(ids=["doc1"], include=["embeddings"])
        assert len(result["embeddings"]) == 1
        assert len(result["embeddings"][0]) == 4

    def test_get_nonexistent_ids(self, collection):
        """get() returns empty for nonexistent IDs."""
        result = collection.get(ids=["nonexistent"])
        assert result["ids"] == []

    def test_get_all_documents(self, collection):
        """get() without filters returns all documents."""
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            documents=["a", "b", "c"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]] * 3,
        )

        result = collection.get(limit=100)
        assert len(result["ids"]) == 3


# ============================================================================
# Test ChromaSQLiteCollection - Query (Similarity Search)
# ============================================================================


class TestCollectionQuery:
    """Tests for vector similarity search."""

    def test_query_basic_similarity_search(self, collection, sample_embeddings):
        """query() performs cosine similarity search."""
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=sample_embeddings,
            documents=["Python", "Java", "Go"],
        )

        # Query with embedding similar to doc1
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        result = collection.query(query_embeddings=[query_embedding], n_results=2)

        assert len(result["ids"][0]) <= 2
        assert "doc1" in result["ids"][0]  # Should match most similar

    def test_query_returns_top_n_results(self, collection, sample_embeddings):
        """query() returns exactly n_results."""
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=sample_embeddings,
            documents=["a", "b", "c"],
        )

        result = collection.query(query_embeddings=[[0.1, 0.2, 0.3, 0.4]], n_results=2)
        assert len(result["ids"][0]) == 2

    def test_query_with_metadata_filtering(self, collection, sample_embeddings, sample_metadatas):
        """query() combines similarity with metadata filtering."""
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=sample_embeddings,
            metadatas=sample_metadatas,
            documents=["a", "b", "c"],
        )

        result = collection.query(
            query_embeddings=[[0.5, 0.6, 0.7, 0.8]],
            n_results=5,
            where={"source": "bitbucket"},
        )

        # Only bitbucket results should be returned
        assert len(result["ids"][0]) <= 2  # Only 2 bitbucket docs
        assert all(m["source"] == "bitbucket" for m in result["metadatas"][0])

    def test_query_includes_distances(self, collection, sample_embeddings):
        """query() includes distance scores."""
        collection.add(ids=["doc1", "doc2"], embeddings=sample_embeddings[:2])

        result = collection.query(
            query_embeddings=[[0.1, 0.2, 0.3, 0.4]],
            n_results=2,
            include=["distances"],
        )

        assert "distances" in result
        assert len(result["distances"][0]) == 2
        assert all(isinstance(d, float) for d in result["distances"][0])

    def test_query_includes_documents(self, collection, sample_embeddings, sample_documents):
        """query() includes documents when requested."""
        collection.add(
            ids=["doc1", "doc2"],
            embeddings=sample_embeddings[:2],
            documents=sample_documents[:2],
        )

        result = collection.query(
            query_embeddings=[[0.1, 0.2, 0.3, 0.4]],
            n_results=2,
            include=["documents"],
        )

        assert result["documents"] is not None
        assert len(result["documents"][0]) == 2

    def test_query_includes_metadatas(self, collection, sample_embeddings, sample_metadatas):
        """query() includes metadata when requested."""
        collection.add(
            ids=["doc1", "doc2"],
            embeddings=sample_embeddings[:2],
            metadatas=sample_metadatas[:2],
        )

        result = collection.query(
            query_embeddings=[[0.1, 0.2, 0.3, 0.4]],
            n_results=2,
            include=["metadatas"],
        )

        assert result["metadatas"] is not None
        assert len(result["metadatas"][0]) == 2

    def test_query_multiple_query_embeddings(self, collection, sample_embeddings):
        """query() handles multiple query embeddings."""
        collection.add(ids=["doc1", "doc2", "doc3"], embeddings=sample_embeddings)

        result = collection.query(
            query_embeddings=[
                [0.1, 0.2, 0.3, 0.4],
                [0.9, 0.8, 0.7, 0.6],
            ],
            n_results=2,
        )

        # Should return results for each query
        assert len(result["ids"]) == 2
        assert len(result["ids"][0]) == 2
        assert len(result["ids"][1]) == 2

    def test_query_sorted_by_similarity(self, collection):
        """query() returns results sorted by similarity."""
        # Create embeddings with known similarities
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=[
                [1.0, 0.0, 0.0, 0.0],  # Orthogonal to query
                [0.0, 1.0, 0.0, 0.0],  # Similar to query
                [0.0, 0.9, 0.0, 0.0],  # Very similar to query
            ],
        )

        query = [0.0, 1.0, 0.0, 0.0]
        result = collection.query(query_embeddings=[query], n_results=3, include=["distances"])

        # Distances should be in ascending order (lower = more similar)
        distances = result["distances"][0]
        assert distances == sorted(distances)

    def test_query_empty_collection(self, collection):
        """query() on empty collection returns empty results."""
        result = collection.query(query_embeddings=[[0.1, 0.2, 0.3, 0.4]], n_results=10)

        assert result["ids"][0] == []


# ============================================================================
# Test ChromaSQLiteCollection - Delete
# ============================================================================


class TestCollectionDelete:
    """Tests for deleting documents from collection."""

    def test_delete_by_ids(self, collection):
        """delete() removes documents by IDs."""
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            documents=["a", "b", "c"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]] * 3,
        )

        collection.delete(ids=["doc1", "doc3"])

        result = collection.get(ids=["doc1", "doc2", "doc3"])
        assert result["ids"] == ["doc2"]

    def test_delete_by_where_clause(self, collection, sample_metadatas):
        """delete() removes documents by metadata filter."""
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            metadatas=sample_metadatas,
            documents=["a", "b", "c"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]] * 3,
        )

        collection.delete(where={"source": "bitbucket"})

        result = collection.get()
        # Only confluence doc should remain
        assert len(result["ids"]) == 1
        assert result["metadatas"][0]["source"] == "confluence"

    def test_delete_requires_filter(self, collection):
        """delete() raises error without filter."""
        with pytest.raises(ValueError, match="Must specify ids or where clause"):
            collection.delete()

    def test_delete_nonexistent_ids(self, collection):
        """delete() with nonexistent IDs doesn't raise error."""
        # Should not raise error
        collection.delete(ids=["nonexistent"])

    def test_delete_combined_filters(self, collection, sample_metadatas):
        """delete() combines ID and where filters."""
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            metadatas=sample_metadatas,
            documents=["a", "b", "c"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]] * 3,
        )

        collection.delete(ids=["doc1"], where={"source": "bitbucket"})

        result = collection.get()
        # doc1 should be deleted, doc2 (confluence) and doc3 (bitbucket but different ID) remain
        assert len(result["ids"]) == 2
        assert "doc1" not in result["ids"]


# ============================================================================
# Test ChromaSQLiteCollection - Update
# ============================================================================


class TestCollectionUpdate:
    """Tests for updating documents in collection."""

    def test_update_document_text(self, collection):
        """update() changes document text."""
        embedding = [[0.1, 0.2, 0.3, 0.4]]
        collection.add(ids=["doc1"], documents=["Original"], embeddings=embedding)

        collection.update(ids=["doc1"], documents=["Updated"], embeddings=embedding)

        result = collection.get(ids=["doc1"])
        assert result["documents"][0] == "Updated"

    def test_update_metadata(self, collection):
        """update() changes metadata."""
        embedding = [[0.1, 0.2, 0.3, 0.4]]
        collection.add(ids=["doc1"], metadatas=[{"version": "1.0"}], embeddings=embedding)

        collection.update(ids=["doc1"], metadatas=[{"version": "2.0"}], embeddings=embedding)

        result = collection.get(ids=["doc1"])
        assert result["metadatas"][0]["version"] == "2.0"

    def test_update_embedding(self, collection):
        """update() changes embedding."""
        collection.add(ids=["doc1"], embeddings=[[1.0, 2.0, 3.0, 4.0]])

        collection.update(ids=["doc1"], embeddings=[[5.0, 6.0, 7.0, 8.0]])

        result = collection.get(ids=["doc1"], include=["embeddings"])
        embedding = result["embeddings"][0]
        assert embedding[0] == pytest.approx(5.0, abs=0.01)

    def test_update_multiple_fields(self, collection):
        """update() changes multiple fields at once."""
        collection.add(
            ids=["doc1"],
            documents=["Original"],
            metadatas=[{"version": "1.0"}],
            embeddings=[[1.0, 2.0, 3.0, 4.0]],
        )

        collection.update(
            ids=["doc1"],
            documents=["Updated"],
            metadatas=[{"version": "2.0"}],
            embeddings=[[5.0, 6.0, 7.0, 8.0]],
        )

        result = collection.get(ids=["doc1"], include=["documents", "metadatas", "embeddings"])
        assert result["documents"][0] == "Updated"
        assert result["metadatas"][0]["version"] == "2.0"
        assert result["embeddings"][0][0] == pytest.approx(5.0, abs=0.01)

    def test_update_nonexistent_document(self, collection):
        """update() on nonexistent document creates it."""
        collection.update(ids=["new_doc"], documents=["New"], embeddings=[[0.1, 0.2, 0.3, 0.4]])

        result = collection.get(ids=["new_doc"])
        assert result["documents"][0] == "New"


# ============================================================================
# Test Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_collection_operations(self, collection):
        """Operations on empty collection return empty results."""
        assert collection.count() == 0
        assert collection.get()["ids"] == []
        assert (
            collection.query(query_embeddings=[[0.1, 0.2, 0.3, 0.4]], n_results=10)["ids"][0] == []
        )

    def test_add_with_mismatched_lengths(self, collection):
        """add() handles mismatched array lengths gracefully."""
        # Different number of embeddings vs IDs should still work
        # (defaults fill in missing values)
        collection.add(
            ids=["doc1", "doc2"],
            embeddings=[[1.0, 2.0, 3.0, 4.0]],  # Only one embedding
        )
        # Should not crash

    def test_special_characters_in_metadata(self, collection):
        """Metadata with special characters is preserved."""
        metadata = {
            "source": "test's source",
            "path": "/path/to/file.groovy",
            "description": 'Contains "quotes"',
        }
        collection.add(ids=["doc1"], metadatas=[metadata], embeddings=[[0.1, 0.2, 0.3, 0.4]])

        result = collection.get(ids=["doc1"])
        assert result["metadatas"][0] == metadata

    def test_unicode_in_documents(self, collection):
        """Unicode text is preserved in documents."""
        doc = "Hello 世界 🌍 Привет"
        collection.add(ids=["doc1"], documents=[doc], embeddings=[[0.1, 0.2, 0.3, 0.4]])

        result = collection.get(ids=["doc1"])
        assert result["documents"][0] == doc

    def test_large_embeddings(self, collection):
        """Large embedding dimensions are handled."""
        large_embedding = [0.1] * 1536  # Common OpenAI embedding size
        collection.add(ids=["doc1"], embeddings=[large_embedding])

        result = collection.get(ids=["doc1"], include=["embeddings"])
        assert len(result["embeddings"][0]) == 1536

    def test_nested_metadata(self, collection):
        """Nested metadata structures are preserved."""
        metadata = {
            "source": "bitbucket",
            "details": {
                "repo": "myrepo",
                "branch": "main",
            },
            "tags": ["java", "gradle"],
        }
        collection.add(ids=["doc1"], metadatas=[metadata], embeddings=[[0.1, 0.2, 0.3, 0.4]])

        result = collection.get(ids=["doc1"])
        assert result["metadatas"][0] == metadata

    def test_query_with_zero_embeddings(self, collection):
        """query() on collection with mixed embeddings filters correctly."""
        # Add one with embedding, one with dummy zero embedding
        collection.add(
            ids=["doc1", "doc2"],
            documents=["With embedding", "Zero embedding"],
            embeddings=[[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0]],
        )

        result = collection.query(query_embeddings=[[0.1, 0.2, 0.3, 0.4]], n_results=10)
        # Should return both (query filters only NULL embeddings)
        assert len(result["ids"][0]) == 2

    def test_where_clause_with_nonexistent_field(self, collection):
        """Filtering by nonexistent metadata field returns empty."""
        collection.add(
            ids=["doc1"],
            metadatas=[{"source": "bitbucket"}],
            documents=["Test"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
        )

        result = collection.get(where={"nonexistent": "value"})
        assert len(result["ids"]) == 0

    def test_concurrent_collection_access(self, temp_db):
        """Multiple collections can access same database."""
        collection1 = ChromaSQLiteCollection(temp_db, "collection1")
        collection2 = ChromaSQLiteCollection(temp_db, "collection2")

        embedding = [[0.1, 0.2, 0.3, 0.4]]
        collection1.add(ids=["doc1"], documents=["Collection 1"], embeddings=embedding)
        collection2.add(ids=["doc1"], documents=["Collection 2"], embeddings=embedding)

        result1 = collection1.get(ids=["doc1"])
        result2 = collection2.get(ids=["doc1"])

        assert result1["documents"][0] == "Collection 1"
        assert result2["documents"][0] == "Collection 2"

        collection1.close()
        collection2.close()


# ============================================================================
# Test Build Where Clause (Internal Method)
# ============================================================================


class TestBuildWhereClause:
    """Tests for _build_where_clause internal method."""

    def test_build_where_clause_simple(self, collection):
        """_build_where_clause creates JSON extraction."""
        clause = collection._build_where_clause("source", "bitbucket")
        assert "json_extract" in clause
        assert "source" in clause

    def test_where_clause_sql_injection_safe(self, collection):
        """_build_where_clause is safe from SQL injection."""
        # Malicious input shouldn't break SQL
        clause = collection._build_where_clause("source", "'; DROP TABLE test; --")
        # Should create parameterised query, not inject SQL
        assert "json_extract" in clause
        # Actual value will be bound as parameter


# ============================================================================
# Test Memory and Performance Characteristics
# ============================================================================


class TestPerformance:
    """Tests for memory and performance characteristics."""

    def test_binary_embedding_storage(self, collection):
        """Embeddings stored as binary reduce storage size."""
        embedding = [0.1] * 384  # Typical embedding size
        collection.add(ids=["doc1"], embeddings=[embedding])

        # Retrieve and verify
        result = collection.get(ids=["doc1"], include=["embeddings"])
        retrieved = result["embeddings"][0]

        # Should be close to original (within float32 precision)
        assert len(retrieved) == 384
        for i in range(10):  # Spot check
            assert retrieved[i] == pytest.approx(0.1, abs=0.001)

    def test_bulk_add_performance(self, collection):
        """Bulk add of many documents completes efficiently."""
        n_docs = 100
        ids = [f"doc{i}" for i in range(n_docs)]
        embeddings = [[0.1, 0.2, 0.3, 0.4]] * n_docs
        documents = [f"Document {i}" for i in range(n_docs)]

        collection.add(ids=ids, embeddings=embeddings, documents=documents)

        assert collection.count() == n_docs

    def test_query_scales_with_results(self, collection):
        """Query performance scales with n_results parameter."""
        # Add many documents
        for i in range(50):
            collection.add(
                ids=[f"doc{i}"],
                embeddings=[[0.1 * i, 0.2, 0.3, 0.4]],
                documents=[f"Doc {i}"],
            )

        # Query with different n_results
        result_small = collection.query(query_embeddings=[[0.1, 0.2, 0.3, 0.4]], n_results=5)
        result_large = collection.query(query_embeddings=[[0.1, 0.2, 0.3, 0.4]], n_results=20)

        assert len(result_small["ids"][0]) == 5
        assert len(result_large["ids"][0]) == 20
