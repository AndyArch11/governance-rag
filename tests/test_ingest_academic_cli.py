"""CLI-level tests for academic ingestion."""

import sys
from pathlib import Path

from scripts.ingest import ingest_academic
from scripts.ingest.academic import config as academic_config
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
