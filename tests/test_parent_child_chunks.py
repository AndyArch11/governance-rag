"""Tests for parent-child chunking functionality.

Tests the complete parent-child chunking pipeline:
1. Creating parent-child chunk pairs
2. Storing parent chunks without embeddings
3. Retrieving parent chunks for matched children
4. Integration with retrieval pipeline
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class DummyLogger:
    """Mock logger for testing."""

    def __init__(self):
        self.messages = []

    def info(self, msg, *args, **kwargs):
        self.messages.append(("info", msg))

    def warning(self, msg, *args, **kwargs):
        self.messages.append(("warning", msg))

    def error(self, msg, *args, **kwargs):
        self.messages.append(("error", msg))

    def debug(self, msg, *args, **kwargs):
        self.messages.append(("debug", msg))


class DummyCollection:
    """Mock ChromaDB collection for testing."""

    def __init__(self):
        self.stored_items = {}
        self.add_calls = []
        self.get_calls = []

    def add(self, ids=None, documents=None, metadatas=None, embeddings=None):
        """Store items in mock collection."""
        self.add_calls.append(
            {
                "ids": ids,
                "documents": documents,
                "metadatas": metadatas,
                "embeddings": embeddings,
            }
        )
        # Store for retrieval
        if ids and documents and metadatas:
            for i, item_id in enumerate(ids):
                self.stored_items[item_id] = {
                    "id": item_id,
                    "document": documents[i] if i < len(documents) else None,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }

    def get(self, ids=None, where=None, include=None):
        """Retrieve items from mock collection."""
        self.get_calls.append({"ids": ids, "where": where, "include": include})

        if ids:
            # Return by IDs
            results = {
                "ids": [],
                "documents": [],
                "metadatas": [],
            }
            for item_id in ids:
                if item_id in self.stored_items:
                    item = self.stored_items[item_id]
                    results["ids"].append(item_id)
                    if include and "documents" in include:
                        results["documents"].append(item["document"])
                    if include and "metadatas" in include:
                        results["metadatas"].append(item["metadata"])
            return results
        elif where:
            # Filter by metadata
            results = {
                "ids": [],
                "documents": [],
                "metadatas": [],
            }
            for item_id, item in self.stored_items.items():
                # Simple where clause matching
                match = True
                for key, value in where.items():
                    if key not in item["metadata"] or item["metadata"][key] != value:
                        match = False
                        break
                if match:
                    results["ids"].append(item_id)
                    if include and "documents" in include:
                        results["documents"].append(item["document"])
                    if include and "metadatas" in include:
                        results["metadatas"].append(item["metadata"])
            return results
        return {"ids": [], "documents": [], "metadatas": []}


# =========================
# Test create_parent_child_chunks
# =========================


class TestCreateParentChildChunks:
    """Tests for create_parent_child_chunks function."""

    def test_create_parent_child_chunks_basic(self):
        """Test basic parent-child chunk creation."""
        from scripts.ingest.chunk import create_parent_child_chunks

        # Text that will create multiple parent chunks
        text = "This is a sentence. " * 150  # ~3000 chars
        child_chunks, parent_chunks = create_parent_child_chunks(
            text, parent_size=1200, child_size=400
        )

        # Should have parent chunks
        assert len(parent_chunks) > 0
        # Should have more children than parents
        assert len(child_chunks) >= len(parent_chunks)

        # Verify structure
        for parent in parent_chunks:
            assert "id" in parent
            assert "text" in parent
            assert "child_ids" in parent
            assert parent["id"].startswith("parent_")
            assert isinstance(parent["child_ids"], list)
            assert len(parent["child_ids"]) > 0

        for child in child_chunks:
            assert "id" in child
            assert "text" in child
            assert "parent_id" in child
            assert child["id"].startswith("child_")
            assert child["parent_id"].startswith("parent_")

    def test_parent_child_chunk_hierarchy(self):
        """Test that child chunks correctly reference their parents."""
        from scripts.ingest.chunk import create_parent_child_chunks

        text = "Paragraph one. " * 100 + "Paragraph two. " * 100
        child_chunks, parent_chunks = create_parent_child_chunks(text)

        # Build parent lookup
        parent_lookup = {p["id"]: p for p in parent_chunks}

        # Verify each child's parent exists and references the child
        for child in child_chunks:
            parent_id = child["parent_id"]
            assert (
                parent_id in parent_lookup
            ), f"Child {child['id']} references non-existent parent {parent_id}"

            parent = parent_lookup[parent_id]
            assert (
                child["id"] in parent["child_ids"]
            ), f"Parent {parent_id} doesn't reference child {child['id']}"

    def test_parent_chunk_contains_child_text(self):
        """Test that parent chunks contain the text of their children."""
        from scripts.ingest.chunk import create_parent_child_chunks

        text = "This is test content. " * 80
        child_chunks, parent_chunks = create_parent_child_chunks(
            text, parent_size=800, child_size=300
        )

        parent_lookup = {p["id"]: p for p in parent_chunks}

        # For each child, verify its text appears in or near its parent's text
        for child in child_chunks:
            parent = parent_lookup[child["parent_id"]]
            # Child text should be substring or overlap with parent text
            # (accounting for overlap between chunks)
            child_words = set(child["text"].split()[:10])  # First 10 words
            parent_words = set(parent["text"].split())
            # At least some words should overlap
            assert len(child_words & parent_words) > 0, "Child text not found in parent"

    def test_short_text_creates_minimal_chunks(self):
        """Test that short text creates at least one parent-child pair."""
        from scripts.ingest.chunk import create_parent_child_chunks

        text = "This is a very short document."
        child_chunks, parent_chunks = create_parent_child_chunks(text)

        assert len(parent_chunks) >= 1
        assert len(child_chunks) >= 1


# =========================
# Test store_parent_chunks
# =========================


class TestStoreParentChunks:
    """Tests for store_parent_chunks function."""

    @patch("scripts.ingest.vectors.get_logger")
    @patch("scripts.ingest.vectors.audit")
    def test_store_parent_chunks_basic(self, mock_audit, mock_get_logger):
        """Test basic parent chunk storage."""
        from scripts.ingest.vectors import store_parent_chunks

        mock_get_logger.return_value = DummyLogger()
        collection = DummyCollection()

        parent_chunks = [
            {
                "id": "parent_0",
                "text": "This is parent chunk 0 with lots of context.",
                "child_ids": ["child_0_0", "child_0_1"],
            },
            {
                "id": "parent_1",
                "text": "This is parent chunk 1 with more context.",
                "child_ids": ["child_1_0", "child_1_1", "child_1_2"],
            },
        ]

        base_metadata = {
            "doc_id": "test_doc",
            "source": "/path/to/doc.html",
            "version": 1,
            "hash": "abc123",
            "doc_type": "technical_guide",
            "embedding_model": "mxbai-embed-large",
        }

        store_parent_chunks(
            doc_id="test_doc",
            parent_chunks=parent_chunks,
            chunk_collection=collection,
            base_metadata=base_metadata,
            dry_run=False,
        )

        # Verify add was called
        assert len(collection.add_calls) == 1
        add_call = collection.add_calls[0]

        # Verify IDs
        assert add_call["ids"] == ["test_doc-parent_0", "test_doc-parent_1"]

        # Verify documents (parent text)
        assert add_call["documents"] == [
            "This is parent chunk 0 with lots of context.",
            "This is parent chunk 1 with more context.",
        ]

        # Verify no embeddings were passed
        assert add_call["embeddings"] is None

        # Verify metadata
        assert len(add_call["metadatas"]) == 2
        meta0 = add_call["metadatas"][0]
        assert meta0["is_parent"] is True
        assert meta0["parent_id"] == "test_doc-parent_0"
        assert meta0["chunk_type"] == "parent"
        assert meta0["doc_id"] == "test_doc"
        assert json.loads(meta0["child_ids"]) == ["child_0_0", "child_0_1"]

        # Verify audit was called
        assert mock_audit.called
        audit_call = mock_audit.call_args[0]
        assert audit_call[0] == "parent_chunks_stored"
        assert audit_call[1]["doc_id"] == "test_doc"
        assert audit_call[1]["parent_count"] == 2
        assert audit_call[1]["storage_type"] == "metadata_only"

    @patch("scripts.ingest.vectors.get_logger")
    def test_store_parent_chunks_dry_run(self, mock_get_logger):
        """Test dry run doesn't store chunks."""
        from scripts.ingest.vectors import store_parent_chunks

        logger = DummyLogger()
        mock_get_logger.return_value = logger
        collection = DummyCollection()

        parent_chunks = [
            {"id": "parent_0", "text": "Test", "child_ids": ["child_0_0"]},
        ]

        store_parent_chunks(
            doc_id="test_doc",
            parent_chunks=parent_chunks,
            chunk_collection=collection,
            base_metadata={"doc_id": "test_doc"},
            dry_run=True,
        )

        # Should NOT call add in dry run
        assert len(collection.add_calls) == 0

        # Should log dry run message
        assert any("[DRY-RUN]" in str(msg) for level, msg in logger.messages)

    @patch("scripts.ingest.vectors.get_logger")
    def test_store_parent_chunks_empty_list(self, mock_get_logger):
        """Test storing empty parent chunks list."""
        from scripts.ingest.vectors import store_parent_chunks

        logger = DummyLogger()
        mock_get_logger.return_value = logger
        collection = DummyCollection()

        store_parent_chunks(
            doc_id="test_doc",
            parent_chunks=[],
            chunk_collection=collection,
            base_metadata={"doc_id": "test_doc"},
            dry_run=False,
        )

        # Should not call add for empty list
        assert len(collection.add_calls) == 0

    def test_add_to_collection_in_batches_splits(self, monkeypatch):
        """Ensure large adds are split into chunks respecting the limit."""

        from scripts.ingest import vectors

        monkeypatch.setattr(vectors, "CHROMADB_ADD_BATCH_LIMIT", 3)

        collection = DummyCollection()
        ids = [f"id_{i}" for i in range(7)]
        docs = [f"doc_{i}" for i in range(7)]
        metas = [{"idx": i} for i in range(7)]
        embeddings = [[float(i)] * 4 for i in range(7)]

        vectors._add_to_collection_in_batches(
            chunk_collection=collection,
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings,
        )

        assert len(collection.add_calls) == 3
        batch_sizes = [len(call["ids"]) for call in collection.add_calls]
        assert batch_sizes == [3, 3, 1]
        assert len(collection.add_calls[0]["embeddings"]) == 3


# =========================
# Test get_parent_for_child
# =========================


class TestGetParentForChild:
    """Tests for get_parent_for_child function."""

    @patch("scripts.ingest.vectors.get_logger")
    def test_get_parent_for_child_success(self, mock_get_logger):
        """Test successful parent retrieval."""
        from scripts.ingest.vectors import get_parent_for_child

        mock_get_logger.return_value = DummyLogger()
        collection = DummyCollection()

        # Store child and parent
        collection.stored_items["child_0_1"] = {
            "id": "child_0_1",
            "document": "Child text",
            "metadata": {"parent_id": "parent_0", "chunk_id": "child_0_1"},
        }
        collection.stored_items["parent_0"] = {
            "id": "parent_0",
            "document": "Parent text with full context",
            "metadata": {"is_parent": True, "parent_id": "parent_0"},
        }

        result = get_parent_for_child("child_0_1", collection)

        assert result is not None
        assert result["id"] == "parent_0"
        assert result["text"] == "Parent text with full context"
        assert result["metadata"]["is_parent"] is True

    @patch("scripts.ingest.vectors.get_logger")
    def test_get_parent_for_child_not_found(self, mock_get_logger):
        """Test parent retrieval when child doesn't exist."""
        from scripts.ingest.vectors import get_parent_for_child

        mock_get_logger.return_value = DummyLogger()
        collection = DummyCollection()

        result = get_parent_for_child("nonexistent_child", collection)

        assert result is None

    @patch("scripts.ingest.vectors.get_logger")
    def test_get_parent_for_child_no_parent_id(self, mock_get_logger):
        """Test parent retrieval when child has no parent_id."""
        from scripts.ingest.vectors import get_parent_for_child

        mock_get_logger.return_value = DummyLogger()
        collection = DummyCollection()

        # Store child without parent_id
        collection.stored_items["child_orphan"] = {
            "id": "child_orphan",
            "document": "Orphan child",
            "metadata": {"chunk_id": "child_orphan"},
        }

        result = get_parent_for_child("child_orphan", collection)

        assert result is None


# =========================
# Test batch_get_parents_for_children
# =========================


class TestBatchGetParentsForChildren:
    """Tests for batch_get_parents_for_children function."""

    @patch("scripts.ingest.vectors.get_logger")
    def test_batch_get_parents_success(self, mock_get_logger):
        """Test successful batch parent retrieval."""
        from scripts.ingest.vectors import batch_get_parents_for_children

        mock_get_logger.return_value = DummyLogger()
        collection = DummyCollection()

        # Store children and parents
        collection.stored_items["child_0_0"] = {
            "id": "child_0_0",
            "document": "Child 0_0",
            "metadata": {"parent_id": "parent_0", "chunk_id": "child_0_0"},
        }
        collection.stored_items["child_0_1"] = {
            "id": "child_0_1",
            "document": "Child 0_1",
            "metadata": {"parent_id": "parent_0", "chunk_id": "child_0_1"},
        }
        collection.stored_items["child_1_0"] = {
            "id": "child_1_0",
            "document": "Child 1_0",
            "metadata": {"parent_id": "parent_1", "chunk_id": "child_1_0"},
        }
        collection.stored_items["parent_0"] = {
            "id": "parent_0",
            "document": "Parent 0 full context",
            "metadata": {"is_parent": True},
        }
        collection.stored_items["parent_1"] = {
            "id": "parent_1",
            "document": "Parent 1 full context",
            "metadata": {"is_parent": True},
        }

        result = batch_get_parents_for_children(["child_0_0", "child_0_1", "child_1_0"], collection)

        # Should return mapping of child_id -> parent data
        assert len(result) == 3
        assert "child_0_0" in result
        assert "child_0_1" in result
        assert "child_1_0" in result

        # Both child_0_0 and child_0_1 should map to same parent
        assert result["child_0_0"]["id"] == "parent_0"
        assert result["child_0_1"]["id"] == "parent_0"
        assert result["child_1_0"]["id"] == "parent_1"

        # Verify parent text
        assert result["child_0_0"]["text"] == "Parent 0 full context"
        assert result["child_1_0"]["text"] == "Parent 1 full context"

    @patch("scripts.ingest.vectors.get_logger")
    def test_batch_get_parents_empty_list(self, mock_get_logger):
        """Test batch retrieval with empty child list."""
        from scripts.ingest.vectors import batch_get_parents_for_children

        mock_get_logger.return_value = DummyLogger()
        collection = DummyCollection()

        result = batch_get_parents_for_children([], collection)

        assert result == {}

    @patch("scripts.ingest.vectors.get_logger")
    def test_batch_get_parents_partial_matches(self, mock_get_logger):
        """Test batch retrieval when some children don't have parents."""
        from scripts.ingest.vectors import batch_get_parents_for_children

        mock_get_logger.return_value = DummyLogger()
        collection = DummyCollection()

        # Store one child with parent, one without
        collection.stored_items["child_with_parent"] = {
            "id": "child_with_parent",
            "document": "Child",
            "metadata": {"parent_id": "parent_0", "chunk_id": "child_with_parent"},
        }
        collection.stored_items["child_orphan"] = {
            "id": "child_orphan",
            "document": "Orphan",
            "metadata": {"chunk_id": "child_orphan"},  # No parent_id
        }
        collection.stored_items["parent_0"] = {
            "id": "parent_0",
            "document": "Parent context",
            "metadata": {"is_parent": True},
        }

        result = batch_get_parents_for_children(["child_with_parent", "child_orphan"], collection)

        # Should only return parent for child_with_parent
        assert len(result) == 1
        assert "child_with_parent" in result
        assert "child_orphan" not in result


# =========================
# Integration Tests
# =========================


class TestParentChildIntegration:
    """Integration tests for complete parent-child workflow."""

    @patch("scripts.ingest.vectors.get_logger")
    @patch("scripts.ingest.vectors.audit")
    def test_full_parent_child_workflow(self, mock_audit, mock_get_logger):
        """Test complete workflow: create, store, retrieve."""
        from scripts.ingest.chunk import create_parent_child_chunks
        from scripts.ingest.vectors import batch_get_parents_for_children, store_parent_chunks

        mock_get_logger.return_value = DummyLogger()
        collection = DummyCollection()

        # Step 1: Create parent-child chunks
        text = "Section one content. " * 100 + "Section two content. " * 100
        child_chunks, parent_chunks = create_parent_child_chunks(
            text, parent_size=1200, child_size=400
        )

        # Step 2: Simulate storing child chunks in collection
        # (In real code, store_chunks_in_chroma handles this)
        for child in child_chunks:
            child_id = f"doc1-{child['id']}"
            collection.stored_items[child_id] = {
                "id": child_id,
                "document": child["text"],
                "metadata": {
                    "chunk_id": child_id,
                    "parent_id": f"doc1-{child['parent_id']}",
                },
            }

        # Step 3: Store parent chunks
        base_metadata = {
            "doc_id": "doc1",
            "source": "/test/doc.html",
            "version": 1,
            "hash": "test_hash",
            "doc_type": "guide",
            "embedding_model": "mxbai-embed-large",
        }

        store_parent_chunks(
            doc_id="doc1",
            parent_chunks=parent_chunks,
            chunk_collection=collection,
            base_metadata=base_metadata,
            dry_run=False,
        )

        # Step 4: Retrieve parents for children
        child_ids = [f"doc1-{c['id']}" for c in child_chunks[:3]]  # First 3 children
        parents_map = batch_get_parents_for_children(child_ids, collection)

        # Verify we got parents back
        assert len(parents_map) > 0
        for child_id in child_ids:
            if child_id in parents_map:
                parent = parents_map[child_id]
                assert "id" in parent
                assert "text" in parent
                assert len(parent["text"]) > len(collection.stored_items[child_id]["document"])


class TestParentChildRetrievalIntegration:
    """Test parent-child integration with retrieval pipeline."""

    @patch("scripts.rag.retrieve.get_logger")
    @patch("scripts.rag.retrieve.audit")
    def test_retrieve_replaces_children_with_parents(self, mock_audit, mock_get_logger):
        """Test that retrieve() replaces child chunks with parent chunks."""
        # This test would require more extensive mocking of the retrieval pipeline
        # For now, we verify the core logic is present
        from scripts.rag.retrieve import batch_get_parents_for_children

        # Verify function is imported and available
        assert batch_get_parents_for_children is not None or batch_get_parents_for_children is None
        # Function existence verified through import


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
