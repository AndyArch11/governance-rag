"""Tests for academic ingestion stage 4 (downloads)."""

from types import SimpleNamespace

from scripts.ingest.ingest_academic import stage_download_references


class DummyLogger:
    def __init__(self):
        self.infos = []

    def info(self, msg):
        self.infos.append(msg)


def test_stage_download_references_skips_on_dry_run():
    logger = DummyLogger()
    config = SimpleNamespace(dry_run=True, cache_dir="/tmp", max_pdf_size_mb=50)
    refs = [{"reference_type": "online", "url": "https://example.com"}]

    updated = stage_download_references(refs, config, logger)
    assert updated[0]["download_status"] == "skipped"
    assert any("Dry run" in msg for msg in logger.infos)


def test_stage_download_references_marks_skipped():
    logger = DummyLogger()
    config = SimpleNamespace(dry_run=False, cache_dir="/tmp", max_pdf_size_mb=50)
    refs = [{"reference_type": "academic"}]

    updated = stage_download_references(refs, config, logger)
    assert updated[0]["download_status"] == "skipped"


def test_stage_download_references_calls_pdf(monkeypatch):
    logger = DummyLogger()
    config = SimpleNamespace(dry_run=False, cache_dir="/tmp", max_pdf_size_mb=50)
    refs = [{"reference_type": "academic", "pdf_url": "https://example.com/file.pdf"}]

    def fake_download(url, dest_dir, max_size_mb):
        assert url.endswith(".pdf")
        return SimpleNamespace(success=True, path="/tmp/x.pdf")

    monkeypatch.setattr("scripts.ingest.ingest_academic.download_reference_pdf", fake_download)

    updated = stage_download_references(refs, config, logger)
    assert updated[0]["download_status"] == "success"
    assert updated[0]["artifact_path"] == "/tmp/x.pdf"


def test_stage_download_references_calls_web(monkeypatch):
    logger = DummyLogger()
    config = SimpleNamespace(dry_run=False, cache_dir="/tmp", max_pdf_size_mb=50)
    refs = [{"reference_type": "online", "url": "https://example.com"}]

    def fake_download(url, dest_dir):
        assert url.startswith("https://")
        return SimpleNamespace(success=True, path="/tmp/x.html")

    monkeypatch.setattr("scripts.ingest.ingest_academic.download_web_content", fake_download)

    updated = stage_download_references(refs, config, logger)
    assert updated[0]["download_status"] == "success"
    assert updated[0]["artifact_path"] == "/tmp/x.html"
