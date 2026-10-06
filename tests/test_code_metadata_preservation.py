"""
Tests for code metadata preservation in ChromaDB and consistency graph.

Verifies that code-specific metadata fields (service, language, dependencies,
internal_calls, etc.) are correctly:
1. Stored in ChromaDB during ingestion
2. Loaded from ChromaDB into graph nodes
3. Serialised/deserialised as JSON for list fields
"""

import json

# Mock imports before importing the modules
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from chromadb import Collection

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_chroma_collection():
    """Create a mock ChromaDB collection."""
    collection = Mock(spec=Collection)
    return collection


@pytest.fixture
def sample_code_metadata():
    """Sample code metadata from ParseResult."""
    return {
        "doc_id": "PaymentService.java",
        "source": "/path/to/PaymentService.java",
        "version": 1,
        "hash": "abc123",
        "doc_type": "java",
        "summary": "Payment processing service",
        "summary_scores": {"overall": 0.9},
        "key_topics": ["payment", "transaction"],
        "source_category": "code",
        # Code-specific fields
        "language": "java",
        "service_name": "PaymentService",
        "service_type": "controller",
        "external_dependencies": [
            "com.stripe:stripe-java:20.0.0",
            "org.springframework.boot:spring-boot-starter-web:2.5.0",
        ],
        "internal_calls": ["AuthService", "TransactionRepository"],
        "endpoints": ["/api/payment/process", "/api/payment/refund"],
        "database_refs": ["payments_db", "transactions"],
        "message_queues": ["payment.queue", "notification.queue"],
        "exports": ["PaymentController", "PaymentService"],
    }


@pytest.fixture
def sample_versioned_doc():
    """Sample versioned doc record from ChromaDB with code metadata."""
    return {
        "doc_id": "PaymentService.java",
        "version": 1,
        "timestamp": "2024-01-01T00:00:00Z",
        "embedding": [0.1] * 1024,
        "summary": "Payment processing service",
        "doc_type": "java",
        "source_category": "code",
        "health": {"overall": 0.9},
        # Code-specific fields (as stored in ChromaDB - JSON strings for lists)
        "language": "java",
        "service_name": "PaymentService",
        "service_type": "controller",
        "dependencies": json.dumps(
            [
                "com.stripe:stripe-java:20.0.0",
                "org.springframework.boot:spring-boot-starter-web:2.5.0",
            ]
        ),
        "internal_calls": json.dumps(["AuthService", "TransactionRepository"]),
        "endpoints": json.dumps(["/api/payment/process", "/api/payment/refund"]),
        "db": json.dumps(["payments_db", "transactions"]),
        "queue": json.dumps(["payment.queue", "notification.queue"]),
        "exports": json.dumps(["PaymentController", "PaymentService"]),
    }


class TestCodeMetadataStorage:
    """Test code metadata storage in ChromaDB."""

    @patch("scripts.ingest.vectors.get_logger")
    @patch("scripts.ingest.vectors.audit")
    @patch("scripts.ingest.vectors.generate_chunk_embeddings_batch")
    @patch("scripts.ingest.vectors.process_and_validate_chunks")
    @patch("scripts.ingest.vectors.compute_document_health")
    def test_code_parser_fields_are_not_stored(
        self,
        mock_compute_health,
        mock_process_chunks,
        mock_generate_embeddings,
        mock_audit,
        mock_get_logger,
        mock_chroma_collection,
        sample_code_metadata,
    ):
        """Code parser metadata is ignored by the generic vector storage path."""
        from scripts.ingest.vectors import store_chunks_in_chroma

        # Setup mocks
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_process_chunks.return_value = ([("chunk1", "test text")], 1, 0, 0)
        mock_compute_health.return_value = {"overall": 0.9}
        mock_generate_embeddings.return_value = (
            [[0.1] * 1024],
            {
                "total_chunks": 1,
                "truncated_chunks": 0,
                "truncation_ratio": 0.0,
                "truncation_loss_avg_pct": 0.0,
                "truncation_chars_lost": 0,
            },
        )

        chunk_collection = Mock(spec=Collection)
        doc_collection = Mock(spec=Collection)

        # Call function
        store_chunks_in_chroma(
            doc_id=sample_code_metadata["doc_id"],
            file_hash=sample_code_metadata["hash"],
            source_path=sample_code_metadata["source"],
            version=sample_code_metadata["version"],
            chunks=["test chunk"],
            metadata=sample_code_metadata,
            chunk_collection=chunk_collection,
            doc_collection=doc_collection,
            preprocess_duration=1.0,
            ingest_duration=1.0,
            dry_run=False,
        )

        # Verify chunk_collection.add was called
        assert chunk_collection.add.called
        call_args = chunk_collection.add.call_args

        # Get the metadata that was stored
        stored_metadata = call_args.kwargs["metadatas"][0]

        code_fields = {
            "language",
            "service_name",
            "service_type",
            "dependencies",
            "internal_calls",
            "endpoints",
            "db",
            "queue",
            "exports",
            "repository",
            "project",
            "branch",
        }
        assert code_fields.isdisjoint(stored_metadata)

    @patch("scripts.ingest.vectors.get_logger")
    @patch("scripts.ingest.vectors.audit")
    @patch("scripts.ingest.vectors.generate_chunk_embeddings_batch")
    @patch("scripts.ingest.vectors.process_and_validate_chunks")
    @patch("scripts.ingest.vectors.compute_document_health")
    def test_non_code_metadata_unchanged(
        self,
        mock_compute_health,
        mock_process_chunks,
        mock_generate_embeddings,
        mock_audit,
        mock_get_logger,
        mock_chroma_collection,
    ):
        """Test that non-code documents don't get code-specific fields."""
        from scripts.ingest.vectors import store_chunks_in_chroma

        # Setup mocks
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_process_chunks.return_value = ([("chunk1", "test text")], 1, 0, 0)
        mock_compute_health.return_value = {"overall": 0.9}
        mock_generate_embeddings.return_value = (
            [[0.1] * 1024],
            {
                "total_chunks": 1,
                "truncated_chunks": 0,
                "truncation_ratio": 0.0,
                "truncation_loss_avg_pct": 0.0,
                "truncation_chars_lost": 0,
            },
        )

        chunk_collection = Mock(spec=Collection)
        doc_collection = Mock(spec=Collection)

        # Non-code metadata
        metadata = {
            "doc_id": "policy.html",
            "source": "/path/to/policy.html",
            "version": 1,
            "hash": "xyz789",
            "doc_type": "html",
            "summary": "Security policy document",
            "summary_scores": {"overall": 0.8},
            "key_topics": ["security", "policy"],
            "source_category": "html",  # Not code
        }

        # Call function
        store_chunks_in_chroma(
            doc_id=metadata["doc_id"],
            file_hash=metadata["hash"],
            source_path=metadata["source"],
            version=metadata["version"],
            chunks=["test chunk"],
            metadata=metadata,
            chunk_collection=chunk_collection,
            doc_collection=doc_collection,
            preprocess_duration=1.0,
            ingest_duration=1.0,
            dry_run=False,
        )

        # Verify chunk_collection.add was called
        assert chunk_collection.add.called
        call_args = chunk_collection.add.call_args

        # Get the metadata that was stored
        stored_metadata = call_args.kwargs["metadatas"][0]

        # Verify code-specific fields are NOT present
        assert "language" not in stored_metadata
        assert "service_name" not in stored_metadata
        assert "dependencies" not in stored_metadata
        assert "internal_calls" not in stored_metadata


class TestCodeMetadataInGraph:
    """Test code metadata preservation in consistency graph nodes."""

    def test_code_metadata_copied_to_graph_nodes(self, sample_versioned_doc):
        """Test that code metadata is copied from ChromaDB to graph nodes."""
        from scripts.consistency_graph.build_consistency_graph import (
            build_consistency_graph_parallel,
        )

        # Mock collection with query method that returns empty results
        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        # Create versioned docs list
        versioned_docs = [sample_versioned_doc]

        # Build graph (without actual LLM calls - just node creation)
        graph = build_consistency_graph_parallel(
            versioned_docs=versioned_docs,
            doc_collection=mock_collection,
            max_neighbours=5,
            sim_threshold=0.5,
            workers=1,
            progress_callback=None,
        )

        # Verify node was created
        assert len(graph["nodes"]) == 1
        node_id = "PaymentService.java_v1"
        assert node_id in graph["nodes"]

        node = graph["nodes"][node_id]

        # Verify core fields
        assert node["doc_id"] == "PaymentService.java"
        assert node["version"] == 1
        assert node["source_category"] == "code"

        code_fields = {
            "language",
            "service_name",
            "service_type",
            "dependencies",
            "internal_calls",
            "endpoints",
            "db",
            "queue",
            "exports",
        }
        assert code_fields.isdisjoint(node)

    def test_multiple_code_nodes_with_metadata(self):
        """Test multiple code nodes preserve their individual metadata."""
        from scripts.consistency_graph.build_consistency_graph import (
            build_consistency_graph_parallel,
        )

        # Mock collection
        mock_collection = Mock(spec=Collection)

        # Create multiple versioned docs
        versioned_docs = [
            {
                "doc_id": "AuthService.java",
                "version": 1,
                "timestamp": "2024-01-01T00:00:00Z",
                "embedding": [0.1] * 768,
                "summary": "Authentication service",
                "doc_type": "java",
                "source_category": "code",
                "health": {},
                "language": "java",
                "service": "AuthService",
                "internal_calls": json.dumps(["UserRepository", "TokenService"]),
                "dependencies": json.dumps(
                    ["org.springframework.security:spring-security-core:5.5.0"]
                ),
            },
            {
                "doc_id": "PaymentService.java",
                "version": 1,
                "timestamp": "2024-01-01T00:00:00Z",
                "embedding": [0.2] * 768,
                "summary": "Payment service",
                "doc_type": "java",
                "source_category": "code",
                "health": {},
                "language": "java",
                "service": "PaymentService",
                "internal_calls": json.dumps(["AuthService", "TransactionRepository"]),
                "dependencies": json.dumps(["com.stripe:stripe-java:20.0.0"]),
            },
        ]

        # Build graph
        graph = build_consistency_graph_parallel(
            versioned_docs=versioned_docs,
            doc_collection=mock_collection,
            max_neighbours=5,
            sim_threshold=0.5,
            workers=1,
            progress_callback=None,
        )

        # Verify both nodes exist
        assert len(graph["nodes"]) == 2

        # Code parser fields are not copied into consistency graph nodes.
        auth_node = graph["nodes"]["AuthService.java_v1"]
        payment_node = graph["nodes"]["PaymentService.java_v1"]
        assert "service" not in auth_node
        assert "dependencies" not in auth_node
        assert "internal_calls" not in auth_node
        assert "service" not in payment_node
        assert "dependencies" not in payment_node
        assert "internal_calls" not in payment_node

    def test_code_json_metadata_is_ignored(self):
        """Malformed code parser metadata does not enter the graph node."""
        from scripts.consistency_graph.build_consistency_graph import (
            build_consistency_graph_parallel,
        )

        # Mock collection
        mock_collection = Mock(spec=Collection)

        # Create doc with malformed JSON
        versioned_docs = [
            {
                "doc_id": "BrokenService.java",
                "version": 1,
                "timestamp": "2024-01-01T00:00:00Z",
                "embedding": [0.1] * 768,
                "summary": "Service with broken metadata",
                "doc_type": "java",
                "source_category": "code",
                "health": {},
                "language": "java",
                "service": "BrokenService",
                "internal_calls": "not-valid-json-{[}",  # Malformed JSON
                "dependencies": json.dumps(["valid.dependency:1.0.0"]),  # Valid JSON
            }
        ]

        # Build graph - should not crash
        graph = build_consistency_graph_parallel(
            versioned_docs=versioned_docs,
            doc_collection=mock_collection,
            max_neighbours=5,
            sim_threshold=0.5,
            workers=1,
            progress_callback=None,
        )

        # Verify node was created
        assert len(graph["nodes"]) == 1
        node = graph["nodes"]["BrokenService.java_v1"]

        assert "internal_calls" not in node
        assert "dependencies" not in node


class TestNoCodeDerivedGraphEdges:
    """Test that code metadata no longer creates graph relationships."""

    def test_shared_dependencies_do_not_create_relationships(self):
        """Shared code dependencies do not synthesise consistency graph edges."""
        from scripts.consistency_graph.build_consistency_graph import (
            build_consistency_graph_parallel,
        )

        # Mock collection
        mock_collection = Mock(spec=Collection)

        # Create versioned docs with shared dependencies
        versioned_docs = [
            {
                "doc_id": "ServiceA.java",
                "version": 1,
                "timestamp": "2024-01-01T00:00:00Z",
                "embedding": [0.1] * 768,
                "summary": "Service A",
                "doc_type": "java",
                "source_category": "code",
                "health": {},
                "service": "ServiceA",
                "dependencies": json.dumps(
                    ["com.shared:library:1.0.0", "org.springframework:spring-core:5.0.0"]
                ),
            },
            {
                "doc_id": "ServiceB.java",
                "version": 1,
                "timestamp": "2024-01-01T00:00:00Z",
                "embedding": [0.2] * 768,
                "summary": "Service B",
                "doc_type": "java",
                "source_category": "code",
                "health": {},
                "service": "ServiceB",
                "dependencies": json.dumps(["com.shared:library:1.0.0", "com.other:lib:2.0.0"]),
            },
        ]

        # Build graph; code metadata must not add non-classified edges.
        graph = build_consistency_graph_parallel(
            versioned_docs=versioned_docs,
            doc_collection=mock_collection,
            max_neighbours=5,
            sim_threshold=0.5,
            workers=1,
            progress_callback=None,
        )

        assert "dependencies" not in graph["nodes"]["ServiceA.java_v1"]
        assert "dependencies" not in graph["nodes"]["ServiceB.java_v1"]

        assert not any(
            edge.get("relationship") in {"dependency", "cross_repo_service"}
            for edge in graph["edges"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
