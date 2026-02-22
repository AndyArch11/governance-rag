"""Tests for counts/list intent routing in scripts.rag.retrieve.retrieve.

Monkeypatches external dependencies to avoid network calls and asserts that
the synthetic counts summary chunk is prepended when count/list intent is detected.
"""

import sys
import types


class DummyCollection:
    def query(self, *a, **k):
        # Return empty results for vector search
        return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

    def get(self, *a, **k):
        # Used by keyword/BM25 fallback paths; keep simple
        return {"ids": [], "documents": [], "metadatas": []}


def test_retrieve_prepends_counts_summary(monkeypatch):
    from scripts.rag import retrieve as retr

    # Fake CountsService.summarise_term
    class FakeCountsService:
        def __init__(self):
            pass

        def summarise_term(self, term, limit=10):
            return {
                "term": term,
                "total_docs": 42,
                "total_occurrences": 100,
                "sample_docs": ["docA", "docB"],
                "category_breakdown": [("governance", 30), ("code", 12)],
            }

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()
            return False

    # Ensure dynamic import inside retrieve() resolves to our fake
    fake_module = types.SimpleNamespace(CountsService=FakeCountsService)
    monkeypatch.setitem(sys.modules, "scripts.rag.counts_service", fake_module)

    # Avoid Ollama embeddings or Chroma calls
    monkeypatch.setattr(
        "scripts.rag.retrieve._embed_query", lambda q, m: [0.0, 0.0, 0.0], raising=False
    )
    monkeypatch.setattr(
        "scripts.rag.retrieve._query_collection",
        lambda collection, embedding, k, model, filters=None: {
            "documents": [[]],
            "metadatas": [[]],
            "ids": [[]],
        },
        raising=False,
    )
    monkeypatch.setattr(
        "scripts.rag.retrieve._bm25_search_with_fallback",
        lambda q, c, k, filters=None: ([], [], []),
        raising=False,
    )

    chunks, meta = retr.retrieve("How many auth docs do we have?", DummyCollection(), k=3)

    assert len(chunks) >= 1
    assert "Corpus summary for" in chunks[0]
    assert meta[0].get("retrieval_method") == "counts"
    assert meta[0].get("counts_total_docs") == 42


def test_retrieve_list_intent(monkeypatch):
    from scripts.rag import retrieve as retr

    class FakeCountsService:
        def __init__(self):
            pass

        def summarise_term(self, term, limit=10):
            return {
                "term": term,
                "total_docs": 5,
                "total_occurrences": 7,
                "sample_docs": ["doc1"],
                "category_breakdown": [("patterns", 3), ("governance", 2)],
            }

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()
            return False

    fake_module = types.SimpleNamespace(CountsService=FakeCountsService)
    monkeypatch.setitem(sys.modules, "scripts.rag.counts_service", fake_module)
    monkeypatch.setattr("scripts.rag.retrieve._embed_query", lambda q, m: [0.0], raising=False)
    monkeypatch.setattr(
        "scripts.rag.retrieve._query_collection",
        lambda collection, embedding, k, model, filters=None: {
            "documents": [[]],
            "metadatas": [[]],
            "ids": [[]],
        },
        raising=False,
    )
    monkeypatch.setattr(
        "scripts.rag.retrieve._bm25_search_with_fallback",
        lambda q, c, k, filters=None: ([], [], []),
        raising=False,
    )

    chunks, meta = retr.retrieve("List all auth documents", DummyCollection(), k=2)

    assert len(chunks) >= 1
    assert "Corpus summary for" in chunks[0]
    assert meta[0].get("counts_total_docs") == 5


def test_retrieve_non_count_query_no_counts_chunk(monkeypatch):
    from scripts.rag import retrieve as retr

    # No counts branch; ensure vector/keyword path continues without counts
    monkeypatch.setattr("scripts.rag.retrieve._embed_query", lambda q, m: [0.0], raising=False)
    monkeypatch.setattr(
        "scripts.rag.retrieve._query_collection",
        lambda collection, embedding, k, model, filters=None: {
            "documents": [["v1"]],
            "metadatas": [[{}]],
            "ids": [["id1"]],
        },
        raising=False,
    )
    monkeypatch.setattr(
        "scripts.rag.retrieve._bm25_search_with_fallback",
        lambda q, c, k, filters=None: ([], [], []),
        raising=False,
    )

    chunks, meta = retr.retrieve("What is MFA?", DummyCollection(), k=1)

    assert chunks == ["v1"]
    # Ensure no counts metadata present
    assert meta[0].get("retrieval_method") == "vector"
