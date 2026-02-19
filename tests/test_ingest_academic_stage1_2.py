"""Tests for academic ingestion stages 1/2 (load + citation extraction)."""

from pathlib import Path

import pytest

from scripts.ingest.academic.parser import extract_citations
from scripts.ingest.ingest_academic import stage_extract_citations, stage_load_document


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, msg):
        self.infos.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)

    def error(self, msg):
        self.errors.append(msg)


class DummyConfig:
    def __init__(self, max_pdf_size_mb=50):
        self.max_pdf_size_mb = max_pdf_size_mb


def test_extract_citations_from_references_section():
    text = """
    Introduction text.

    References
    [1] Smith, J. (2020). Example Paper. https://doi.org/10.1000/182
    [2] Doe, A. Another Paper. doi:10.5555/12345678
    """
    citations = extract_citations(text)
    assert len(citations) == 2
    assert citations[0].doi == "10.1000/182"
    assert citations[1].doi == "10.5555/12345678"


def test_extract_citations_without_reference_section():
    text = "No references here."
    citations = extract_citations(text)
    assert citations == []


def test_stage_extract_citations_returns_raw_texts():
    text = """
    References
    1. Example Ref. https://doi.org/10.1234/abcd
    """
    logger = DummyLogger()
    raw = stage_extract_citations(text, logger)
    assert raw == ["1. Example Ref. https://doi.org/10.1234/abcd"]
    assert any("Extracted" in msg for msg in logger.infos)


def test_extract_citations_handles_multiline_blocks():
    text = """
    References
    Smith, J. (2020). Example Paper.
    Journal of Testing, 12(3), 45-67.

    Doe, A. (2019). Another Paper.
    Proceedings of Example Conf, 101-110.
    """
    citations = extract_citations(text)
    assert len(citations) >= 2


def test_extract_citations_ignores_toc_heading_and_uses_later_section():
    text = """
    Table of Contents
    References.................................................... 12
    Chapter 1 Introduction........................................ 1

    References
    Smith, J. (2020). Example Paper. Journal of Testing, 12(3), 45-67.
    Doe, A. (2019). Another Paper. Proceedings of Example Conf, 101-110.
    """
    citations = extract_citations(text)
    assert len(citations) == 2


def test_stage_load_document_skips_missing(tmp_path):
    logger = DummyLogger()
    config = DummyConfig()
    missing_path = tmp_path / "missing.pdf"
    result = stage_load_document(missing_path, config, logger)
    assert result is None
    assert any("Document not found" in msg for msg in logger.warnings)


def test_stage_load_document_skips_non_pdf(tmp_path):
    logger = DummyLogger()
    config = DummyConfig()
    non_pdf = tmp_path / "doc.txt"
    non_pdf.write_text("hello")
    result = stage_load_document(non_pdf, config, logger)
    assert result is None
    assert any("Skipping non-PDF" in msg for msg in logger.warnings)


def test_stage_load_document_skips_oversize(tmp_path, monkeypatch):
    logger = DummyLogger()
    config = DummyConfig(max_pdf_size_mb=1)
    pdf_path = tmp_path / "big.pdf"
    pdf_path.write_bytes(b"x" * 2 * 1024 * 1024)  # 2MB
    result = stage_load_document(pdf_path, config, logger)
    assert result is None
    assert any("exceeds limit" in msg for msg in logger.warnings)


def test_stage_load_document_extracts_text(tmp_path, monkeypatch):
    logger = DummyLogger()
    config = DummyConfig()
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    def fake_extract_text(path):
        assert Path(path) == pdf_path
        return "PDF text"

    monkeypatch.setattr("scripts.ingest.ingest_academic.extract_text_from_pdf", fake_extract_text)

    result = stage_load_document(pdf_path, config, logger)
    assert result == "PDF text"
