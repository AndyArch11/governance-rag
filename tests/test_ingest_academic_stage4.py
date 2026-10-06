"""Tests for academic ingestion stage 4 (downloads)."""

from types import SimpleNamespace

from scripts.ingest import ingest_academic as ingest_academic_module
from scripts.ingest.academic.downloader import DownloadResult
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


def test_stage_download_references_reports_structured_progress(monkeypatch):
    events = []
    results = iter(
        [
            DownloadResult(success=True, path="/tmp/one.html"),
            DownloadResult(success=False, error="http_404"),
        ]
    )
    monkeypatch.setattr(
        ingest_academic_module,
        "audit",
        lambda event, data: events.append((event, data)),
    )
    monkeypatch.setattr(
        ingest_academic_module,
        "download_web_content",
        lambda url, dest_dir: next(results),
    )
    config = SimpleNamespace(dry_run=False, cache_dir="/tmp", max_pdf_size_mb=50)
    refs = [
        {"reference_type": "online", "url": "https://example.com/one"},
        {"reference_type": "online", "url": "https://example.com/two"},
        {"reference_type": "academic"},
    ]

    updated = stage_download_references(refs, config, DummyLogger())

    assert [reference["download_status"] for reference in updated] == [
        "success",
        "http_404",
        "skipped",
    ]
    checkpoints = [data for event, data in events if event == "progress_checkpoint"]
    assert checkpoints[0]["stage"] == "reference_download"
    assert checkpoints[-1]["items_done"] == 3
    assert checkpoints[-1]["items_total"] == 3
    assert checkpoints[-1]["succeeded"] == 1
    assert checkpoints[-1]["failed"] == 1
    assert checkpoints[-1]["skipped"] == 1
