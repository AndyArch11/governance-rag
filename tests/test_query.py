"""Tests for query CLI module.

Covers argument parsing, collection loading, validation, and main flow with
mocked dependencies to avoid real Chroma or LLM calls.
"""

import sys
from pathlib import Path

import pytest


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.errors = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg)

    def error(self, msg, *args, **kwargs):
        self.errors.append(msg)


class DummyCollection:
    def __init__(self):
        self.name = "dummy"


@pytest.fixture()
def query_module(monkeypatch):
    """Load query module with patched logger and audit."""
    import importlib

    sys.modules.pop("scripts.rag.query", None)
    query = importlib.import_module("scripts.rag.query")

    dummy_logger = DummyLogger()
    audit_events = []

    monkeypatch.setattr(query, "logger", dummy_logger)
    monkeypatch.setattr(query, "audit", lambda evt, data: audit_events.append((evt, data)))

    # Patch RAGConfig to disable resource monitoring in tests
    original_ragconfig = query.RAGConfig

    class TestRAGConfig(original_ragconfig):
        def __init__(self):
            super().__init__()
            self.enable_resource_monitoring = False

    monkeypatch.setattr(query, "RAGConfig", TestRAGConfig)

    return query, dummy_logger, audit_events


class TestParseArgs:
    """Tests for parse_args."""

    def test_parse_args_defaults(self, monkeypatch, query_module):
        query, _, _ = query_module
        monkeypatch.setattr(sys, "argv", ["query.py", "What is MFA?"])
        args = query.parse_args()
        assert args.query == "What is MFA?"
        assert args.k is None
        assert args.verbose is False
        assert args.show_sources is False

    def test_parse_args_with_flags(self, monkeypatch, query_module):
        query, _, _ = query_module
        monkeypatch.setattr(
            sys, "argv", ["query.py", "--k", "7", "--verbose", "--show-sources", "Test?"]
        )
        args = query.parse_args()
        assert args.query == "Test?"
        assert args.k == 7
        assert args.verbose is True
        assert args.show_sources is True


class TestGetCollection:
    """Tests for get_collection."""

    def test_get_collection_success(self, monkeypatch, query_module):
        query, logger, audit_events = query_module

        class DummyClient:
            def __init__(self, path=None):
                self.path = path
                self.collections = {}

            def get_collection(self, name):
                self.collections[name] = DummyCollection()
                return self.collections[name]

        # Patch PersistentClient
        monkeypatch.setattr(query, "PersistentClient", DummyClient)

        config = query.RAGConfig()
        config.chunk_collection_name = "test_collection"

        collection = query.get_collection(config)
        assert isinstance(collection, DummyCollection)
        assert any("Loaded collection" in msg for msg in logger.infos)
        assert audit_events == []  # No audit on success

    def test_get_collection_failure(self, monkeypatch, query_module):
        query, logger, audit_events = query_module

        class FailingClient:
            def __init__(self, path=None):
                pass

            def get_collection(self, name):
                raise RuntimeError("boom")

        monkeypatch.setattr(query, "PersistentClient", FailingClient)

        config = query.RAGConfig()
        with pytest.raises(RuntimeError):
            query.get_collection(config)

        assert any("Failed to load collection" in msg for msg in logger.errors)
        assert any(evt == "collection_load_error" for evt, _ in audit_events)


class TestMainFlow:
    """Tests for main function behaviour."""

    def test_main_empty_query_exits(self, monkeypatch, query_module):
        query, logger, audit_events = query_module
        monkeypatch.setattr(sys, "argv", ["query.py", "   "])

        with pytest.raises(SystemExit) as exc:
            query.main()
        assert exc.value.code == 1
        assert any("Query cannot be empty" in msg for msg in logger.errors)
        assert audit_events == []

    def test_main_invalid_k_exits(self, monkeypatch, query_module):
        query, logger, audit_events = query_module
        monkeypatch.setattr(sys, "argv", ["query.py", "--k", "0", "Test"])

        with pytest.raises(SystemExit) as exc:
            query.main()
        assert exc.value.code == 1
        assert any("k must be >= 1" in msg for msg in logger.errors)
        assert audit_events == []

    def test_main_success_flow(self, monkeypatch, query_module, capsys):
        query, logger, audit_events = query_module

        # Patch args
        monkeypatch.setattr(sys, "argv", ["query.py", "--k", "2", "What is MFA?"])

        # Patch get_collection
        dummy_collection = DummyCollection()
        monkeypatch.setattr(query, "get_collection", lambda cfg: dummy_collection)

        # Patch answer
        def fake_answer(q, collection, k=None, temperature=None):
            return {
                "answer": "MFA requires two factors.",
                "chunks": ["chunk1", "chunk2"],
                "sources": [{"source": "doc1"}, {"source": "doc2"}],
                "generation_time": 1.2,
                "total_time": 1.5,
                "retrieval_count": 2,
                "model": "test-model",
            }

        monkeypatch.setattr(query, "answer", fake_answer)

        # Run main (should not exit on success)
        query.main()

        # Output should include answer
        captured = capsys.readouterr()
        assert "ANSWER:" in captured.out
        assert "MFA requires two factors." in captured.out

        # Audits for start and complete
        events = [evt for evt, _ in audit_events]
        assert "query_start" in events
        assert "query_complete" in events

    def test_main_handles_answer_exception(self, monkeypatch, query_module):
        query, logger, audit_events = query_module

        monkeypatch.setattr(sys, "argv", ["query.py", "Test"])
        monkeypatch.setattr(query, "get_collection", lambda cfg: DummyCollection())

        def failing_answer(*args, **kwargs):
            raise RuntimeError("fail")

        monkeypatch.setattr(query, "answer", failing_answer)

        with pytest.raises(SystemExit) as exc:
            query.main()
        assert exc.value.code == 1
        assert any("Query failed" in msg for msg in logger.errors)
        assert any(evt == "query_failed" for evt, _ in audit_events)

    def test_main_keyboard_interrupt(self, monkeypatch, query_module):
        query, logger, audit_events = query_module

        monkeypatch.setattr(sys, "argv", ["query.py", "Test"])
        monkeypatch.setattr(query, "get_collection", lambda cfg: DummyCollection())

        def interrupt_answer(*args, **kwargs):
            raise KeyboardInterrupt()

        monkeypatch.setattr(query, "answer", interrupt_answer)

        with pytest.raises(SystemExit) as exc:
            query.main()
        assert exc.value.code == 0
        assert any("interrupted" in evt for evt, _ in audit_events)
