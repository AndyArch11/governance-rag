"""Tests for rag_logger module - module-specific behaviours only.

Core logging and audit behaviours are tested in test_loggers.py.
This file contains only RAG-specific event types and data structures.

TODO Remove after confirming coverage of testing, replaced by test_utils_logger.py
"""

import importlib
import json

import pytest

from scripts.rag import rag_logger


@pytest.fixture
def temp_logs_dir(tmp_path, monkeypatch):
    """Create temporary logs directory for testing."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    # Mock LOGS_DIR to use temp directory
    monkeypatch.setattr(rag_logger, "LOGS_DIR", logs_dir)
    audit_log_path = logs_dir / "rag_audit.jsonl"
    monkeypatch.setattr(rag_logger, "AUDIT_LOG", audit_log_path)

    # Clear existing handlers to avoid pollution
    rag_logger.logger.handlers.clear()
    rag_logger.logger.propagate = False

    # Recreate rotating file handler with temp path
    from logging.handlers import RotatingFileHandler

    rfh = RotatingFileHandler(logs_dir / "rag.log", maxBytes=5 * 1024 * 1024, backupCount=5)
    rfh.setLevel(rag_logger.LOGGING_LEVEL)
    rfh.setFormatter(rag_logger.formatter)
    monkeypatch.setattr(rag_logger, "rfh", rfh)

    return logs_dir


class TestRAGSpecificBehaviours:
    """Tests for RAG-specific behaviours not covered by common logger tests."""

    def test_audit_event_types(self, temp_logs_dir):
        """Test different RAG-specific event types in audit log."""
        rag_logger.audit("query_start", {"query": "test"})  # type: ignore
        rag_logger.audit("retrieval_complete", {"chunks": 5})  # type: ignore
        rag_logger.audit("generation_start", {})  # type: ignore
        rag_logger.audit("generation_complete", {"tokens": 150})  # type: ignore
        rag_logger.audit("error", {"error_type": "timeout"})  # type: ignore

        audit_file = temp_logs_dir / "rag_audit.jsonl"
        lines = audit_file.read_text().strip().split("\n")

        assert len(lines) == 5
        events = [json.loads(line)["event"] for line in lines]
        assert events == [
            "query_start",
            "retrieval_complete",
            "generation_start",
            "generation_complete",
            "error",
        ]

    def test_audit_with_complex_data(self, temp_logs_dir):
        """Test audit with RAG-specific nested data structures."""
        complex_data = {
            "query": "test query",
            "results": [{"doc_id": "doc1", "score": 0.95}, {"doc_id": "doc2", "score": 0.87}],
            "metadata": {"source_category": "Governance", "doc_type": "policy"},
        }

        rag_logger.audit("retrieval", complex_data)  # type: ignore

        audit_file = temp_logs_dir / "rag_audit.jsonl"
        entry = json.loads(audit_file.read_text().strip())

        assert entry["event"] == "retrieval"
        assert entry["query"] == "test query"
        assert len(entry["results"]) == 2
        assert entry["results"][0]["doc_id"] == "doc1"
        assert entry["metadata"]["source_category"] == "Governance"
