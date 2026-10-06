"""CLI-level tests for academic ingestion."""

import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.ingest import ingest_academic
from scripts.ingest.academic import config as academic_config
from scripts.utils import logger as utils_logger
from scripts.utils.config import BaseConfig


def test_purge_logs_disabled_in_prod(monkeypatch, tmp_path):
    def fake_ensure_dir(self, path: Path) -> Path:
        tmp_path.mkdir(parents=True, exist_ok=True)
        return tmp_path

    # Route logs to temp directory
    monkeypatch.setattr(BaseConfig, "ensure_dir", fake_ensure_dir, raising=True)

    # Set Prod environment
    monkeypatch.setenv("ENVIRONMENT", "Prod")

    # Seed logs
    log_file = tmp_path / "academic_ingest.log"
    audit_file = tmp_path / "academic_ingest_audit.jsonl"
    log_file.write_text("old-log")
    audit_file.write_text("old-audit")

    # Prepare CLI args
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_academic.py",
            "--purge-logs",
            "--revalidate",
            "all",
        ],
    )

    academic_config._ACADEMIC_CONFIG = None
    BaseConfig.clear_overrides()

    result = ingest_academic.main()

    # Should abort in Prod without deleting logs
    assert result == 1
    assert log_file.read_text() == "old-log"
    assert audit_file.read_text() == "old-audit"

    academic_config._ACADEMIC_CONFIG = None
    BaseConfig.clear_overrides()


def test_academic_ingestion_accepts_cultural_lens_option(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["ingest_academic.py", "thesis.pdf", "--cultural-lens", "lens_draft"],
    )

    args = ingest_academic.parse_args()

    assert args.cultural_lens == "lens_draft"


def test_purge_logs_replaces_ingest_log_records_in_dev(monkeypatch, tmp_path):
    """Academic log purging removes prior shared ingest and audit records in Dev."""
    monkeypatch.setattr(utils_logger, "LOGS_DIR", tmp_path)
    monkeypatch.setenv("ENVIRONMENT", "Dev")

    ingest_log = tmp_path / "ingest.log"
    audit_log = tmp_path / "ingest_audit.jsonl"
    ingest_log.write_text("old ingest log record\n")
    audit_log.write_text('{"event": "old_audit_record"}\n')

    cached_logger = utils_logger._loggers.pop("ingest", None)
    if cached_logger:
        for handler in cached_logger.handlers:
            handler.close()

    monkeypatch.setattr(
        sys,
        "argv",
        ["ingest_academic.py", "--purge-logs"],
    )
    academic_config._ACADEMIC_CONFIG = None
    BaseConfig.clear_overrides()

    try:
        assert ingest_academic.main() == 1
        assert "old ingest log record" not in ingest_log.read_text()
        assert "No documents provided" in ingest_log.read_text()
        assert "old_audit_record" not in audit_log.read_text()
        assert "purge_logs" in audit_log.read_text()
    finally:
        logger = utils_logger._loggers.pop("ingest", None)
        if logger:
            for handler in logger.handlers:
                handler.close()
        academic_config._ACADEMIC_CONFIG = None
        BaseConfig.clear_overrides()


def test_empty_single_pdf_does_not_write_an_empty_citation_graph(monkeypatch, tmp_path):
    pdf_path = tmp_path / "unreadable-thesis.pdf"
    pdf_path.write_bytes(b"not extractable by fixture")
    args = Namespace(
        papers_positional=[],
        papers=[str(pdf_path)],
        papers_dir=None,
        batch=None,
        title=None,
        domain=None,
        topic=None,
        authors=None,
        institution=None,
        skip_citations=False,
        skip_terminology=True,
        cultural_lens=None,
        dry_run=False,
        cache_reset=False,
        reset=False,
        refresh=False,
        purge_logs=False,
        bm25_indexing=None,
        skip_bm25=False,
        revalidate=None,
        staleness_threshold=30,
        ref_ids=[],
        crossref_email=None,
        unpaywall_email=None,
        semantic_scholar_key=None,
        orcid_client_id=None,
        orcid_client_secret=None,
    )
    config = SimpleNamespace(
        environment="Dev",
        dry_run=False,
        rag_data_path=tmp_path / "rag_data",
        chunk_collection_name="thesis_chunks",
        doc_collection_name="thesis_documents",
        bm25_indexing_enabled=False,
    )
    graph_writes = []

    class FakeVectorClient:
        def __init__(self, path):
            self.path = path

        def get_or_create_collection(self, *, name, embedding_function):
            return object()

    class FakeResourceMonitor:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def print_summary(self):
            pass

        def export_json(self):
            return tmp_path / "resource-stats.json"

    monkeypatch.setattr(ingest_academic, "parse_args", lambda: args)
    monkeypatch.setattr(ingest_academic, "build_overrides", lambda parsed_args: {})
    monkeypatch.setattr(ingest_academic, "get_academic_config", lambda reset: config)
    monkeypatch.setattr(ingest_academic.BaseConfig, "set_overrides", lambda overrides: None)
    monkeypatch.setattr(ingest_academic, "get_logger", lambda: Mock())
    monkeypatch.setattr(ingest_academic, "audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingest_academic, "configure_child_logger_propagation", lambda *args: None)
    monkeypatch.setattr(ingest_academic, "collect_documents", lambda parsed_args: [pdf_path])
    monkeypatch.setattr(ingest_academic, "stage_load_document", lambda *args, **kwargs: "")
    monkeypatch.setattr(ingest_academic, "ResourceMonitor", FakeResourceMonitor)
    monkeypatch.setattr(ingest_academic, "ReferenceCache", lambda: object())
    monkeypatch.setattr(
        ingest_academic, "get_vector_client", lambda prefer: (FakeVectorClient, False)
    )
    monkeypatch.setattr(
        ingest_academic,
        "get_default_vector_path",
        lambda rag_data_path, using_sqlite: tmp_path / "vectors",
    )
    monkeypatch.setattr(
        ingest_academic.CitationGraph,
        "write_sqlite",
        lambda self, *args, **kwargs: graph_writes.append((args, kwargs)),
    )

    assert ingest_academic.main() == 1
    assert graph_writes == []


def test_revalidation_cli_dispatches_without_collecting_documents(monkeypatch, tmp_path):
    cache_path = tmp_path / "academic_references.db"
    cache_path.touch()
    config = SimpleNamespace(
        environment="Dev",
        dry_run=False,
        rag_data_path=tmp_path,
        cache_dir=tmp_path / "web_cache",
        chunk_collection_name="thesis_chunks",
        doc_collection_name="thesis_documents",
    )
    revalidation_calls = []
    chunk_collection = object()
    doc_collection = object()

    class FakeVectorClient:
        def __init__(self, path):
            self.path = path

        def get_collection(self, *, name):
            return {
                "thesis_chunks": chunk_collection,
                "thesis_documents": doc_collection,
            }[name]

    def fake_revalidation(
        cache,
        citation_graph_path,
        mode,
        threshold,
        ref_ids,
        logger,
        web_content_dir,
        thesis_id,
        chunk_collection,
        doc_collection,
    ):
        revalidation_calls.append(
            (
                cache.db_path,
                citation_graph_path,
                mode,
                threshold,
                ref_ids,
                web_content_dir,
                thesis_id,
                chunk_collection,
                doc_collection,
            )
        )
        return SimpleNamespace(total=0, failed=0)

    def reject_document_collection(args):
        raise AssertionError("revalidation must not enter document ingestion")

    monkeypatch.setattr(
        sys,
        "argv",
        ["ingest_academic.py", "--revalidate", "all", "--thesis-id", "thesis-1"],
    )
    monkeypatch.setattr(ingest_academic.BaseConfig, "set_overrides", lambda overrides: None)
    monkeypatch.setattr(ingest_academic, "get_academic_config", lambda reset: config)
    monkeypatch.setattr(ingest_academic, "get_logger", lambda: Mock())
    monkeypatch.setattr(ingest_academic, "audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingest_academic, "configure_child_logger_propagation", lambda *args: None)
    monkeypatch.setattr(ingest_academic, "collect_documents", reject_document_collection)
    monkeypatch.setattr(
        ingest_academic, "get_vector_client", lambda prefer: (FakeVectorClient, False)
    )
    monkeypatch.setattr(
        ingest_academic,
        "get_default_vector_path",
        lambda rag_data_path, using_sqlite: tmp_path / "vectors",
    )
    monkeypatch.setattr(
        ingest_academic,
        "revalidate_cached_references",
        fake_revalidation,
        raising=False,
    )

    assert ingest_academic.main() == 0
    assert revalidation_calls == [
        (
            cache_path,
            tmp_path / "academic_citation_graph.db",
            "all",
            30,
            [],
            tmp_path / "web_cache",
            "thesis-1",
            chunk_collection,
            doc_collection,
        )
    ]
