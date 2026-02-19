"""Unified, parameterised tests for project loggers.

Covers common behaviour across rag_logger without duplicating test code.
Note: ingest_logger and consistency_logger have been removed in favour of
scripts.utils.logger directly.
"""

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import pytest

# Parameter set for each logger module
# Note: ingest_logger and consistency_logger have been removed in favour of scripts.utils.logger directly
LOGGER_CASES = [
    {
        "scripts_subdir": "rag",
        "module_name": "rag_logger",
        "logger_name": "rag",
        "log_filename": "rag.log",
        "audit_filename": "rag_audit.jsonl",
    },
]


def _import_logger_module(case: Dict):
    """Import a logger module given its scripts subdir and module name."""
    # Build package-qualified module name
    package_name = f"scripts.{case['scripts_subdir']}.{case['module_name']}"
    # Reload to ensure fresh state per parametrised run
    sys.modules.pop(package_name, None)
    return importlib.import_module(package_name)


@pytest.fixture()
def temp_logs_dir(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    return logs_dir


@pytest.mark.parametrize("case", LOGGER_CASES)
def test_get_logger_and_write(case, temp_logs_dir, monkeypatch, capsys):
    mod = _import_logger_module(case)

    # Patch log paths & reset handlers
    if hasattr(mod, "LOGS_DIR"):
        monkeypatch.setattr(mod, "LOGS_DIR", temp_logs_dir)
    if hasattr(mod, "AUDIT_LOG"):
        monkeypatch.setattr(mod, "AUDIT_LOG", temp_logs_dir / case["audit_filename"])  # type: ignore

    # Rewire rotating file handler or log file constants as needed
    # Clear existing handlers to avoid pollution
    if hasattr(mod, "logger"):
        mod.logger.handlers.clear()
        mod.logger.propagate = False

    target_log_file = temp_logs_dir / case["log_filename"]

    # Some modules expose a RotatingFileHandler instance in `rfh`; replace it
    try:
        from logging.handlers import RotatingFileHandler

        if hasattr(mod, "rfh"):
            rfh = RotatingFileHandler(target_log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
            monkeypatch.setattr(mod, "rfh", rfh)
        elif hasattr(mod, "LOG_FILE"):
            monkeypatch.setattr(mod, "LOG_FILE", target_log_file)
    except Exception:
        pass

    logger = mod.get_logger()  # type: ignore
    assert isinstance(logger, logging.Logger)
    assert logger.name == case["logger_name"]

    # File-only
    logger = mod.get_logger(log_to_console=False)  # type: ignore
    logger.info("File-only test message")
    assert target_log_file.exists()
    content = target_log_file.read_text()
    assert "File-only test message" in content

    # Console-enabled
    logger = mod.get_logger(log_to_console=True)  # type: ignore
    logger.info("Console test message")
    captured = capsys.readouterr()
    # Allow stdout or stderr depending on handler config
    assert "Console test message" in captured.err or "Console test message" in captured.out

    # Level & formatting
    assert mod.LOGGING_LEVEL == logging.INFO  # type: ignore
    assert logger.level == mod.LOGGING_LEVEL  # type: ignore
    logger.info("Formatted message")
    content = target_log_file.read_text()
    assert "[INFO]" in content
    import re

    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", content)

    # Warning/error presence
    logger.warning("Warn msg")
    logger.error("Err msg")
    content = target_log_file.read_text()
    assert "[WARNING]" in content
    assert "[ERROR]" in content


@pytest.mark.parametrize("case", LOGGER_CASES)
def test_audit_basic(case, temp_logs_dir, monkeypatch):
    mod = _import_logger_module(case)

    # Patch audit/log paths
    if hasattr(mod, "LOGS_DIR"):
        monkeypatch.setattr(mod, "LOGS_DIR", temp_logs_dir)
    audit_path = temp_logs_dir / case["audit_filename"]
    if hasattr(mod, "AUDIT_LOG"):
        monkeypatch.setattr(mod, "AUDIT_LOG", audit_path)

    # Ensure clean handlers
    if hasattr(mod, "logger"):
        mod.logger.handlers.clear()
        mod.logger.propagate = False

    # Write an audit entry
    mod.audit("test_event", {"key": "value", "number": 123})  # type: ignore
    assert audit_path.exists()

    entry = json.loads(audit_path.read_text().strip())
    assert entry["event"] == "test_event"
    assert entry["key"] == "value"
    assert entry["number"] == 123
    assert "timestamp" in entry


@pytest.mark.parametrize("case", LOGGER_CASES)
def test_audit_multiple_entries(case, temp_logs_dir, monkeypatch):
    mod = _import_logger_module(case)

    # Patch audit/log paths
    if hasattr(mod, "LOGS_DIR"):
        monkeypatch.setattr(mod, "LOGS_DIR", temp_logs_dir)
    audit_path = temp_logs_dir / case["audit_filename"]
    if hasattr(mod, "AUDIT_LOG"):
        monkeypatch.setattr(mod, "AUDIT_LOG", audit_path)

    # Write multiple entries
    mod.audit("event1", {"id": 1})  # type: ignore
    mod.audit("event2", {"id": 2})  # type: ignore
    mod.audit("event3", {"id": 3})  # type: ignore

    lines = audit_path.read_text().strip().split("\n")
    assert len(lines) == 3

    e1 = json.loads(lines[0])
    e2 = json.loads(lines[1])
    e3 = json.loads(lines[2])

    assert e1["event"] == "event1"
    assert e1["id"] == 1
    assert e2["event"] == "event2"
    assert e2["id"] == 2
    assert e3["event"] == "event3"
    assert e3["id"] == 3


@pytest.mark.parametrize("case", LOGGER_CASES)
def test_logger_does_not_propagate(case, temp_logs_dir, monkeypatch):
    """Test that logger propagate is set to False for all loggers."""
    mod = _import_logger_module(case)

    if hasattr(mod, "LOGS_DIR"):
        monkeypatch.setattr(mod, "LOGS_DIR", temp_logs_dir)
    if hasattr(mod, "logger"):
        mod.logger.handlers.clear()

    assert mod.logger.propagate is False


@pytest.mark.parametrize("case", LOGGER_CASES)
def test_no_duplicate_file_handlers(case, temp_logs_dir, monkeypatch):
    """Test that calling get_logger multiple times doesn't add duplicate file handlers."""
    from logging.handlers import RotatingFileHandler

    mod = _import_logger_module(case)

    if hasattr(mod, "LOGS_DIR"):
        monkeypatch.setattr(mod, "LOGS_DIR", temp_logs_dir)
    if hasattr(mod, "LOG_FILE"):
        monkeypatch.setattr(mod, "LOG_FILE", temp_logs_dir / case["log_filename"])
    if hasattr(mod, "AUDIT_LOG"):
        monkeypatch.setattr(mod, "AUDIT_LOG", temp_logs_dir / case["audit_filename"])

    # Clear handlers
    if hasattr(mod, "logger"):
        mod.logger.handlers.clear()

    # Call get_logger multiple times
    logger1 = mod.get_logger()
    logger2 = mod.get_logger()
    logger3 = mod.get_logger()

    # Count RotatingFileHandler instances
    file_handlers = [h for h in logger1.handlers if isinstance(h, RotatingFileHandler)]

    # Should only have one file handler
    assert len(file_handlers) == 1


@pytest.mark.parametrize("case", LOGGER_CASES)
def test_no_duplicate_console_handlers(case, temp_logs_dir, monkeypatch):
    """Test that calling get_logger with console=True doesn't add duplicate console handlers."""
    from logging.handlers import RotatingFileHandler

    mod = _import_logger_module(case)

    if hasattr(mod, "LOGS_DIR"):
        monkeypatch.setattr(mod, "LOGS_DIR", temp_logs_dir)
    if hasattr(mod, "LOG_FILE"):
        monkeypatch.setattr(mod, "LOG_FILE", temp_logs_dir / case["log_filename"])
    if hasattr(mod, "AUDIT_LOG"):
        monkeypatch.setattr(mod, "AUDIT_LOG", temp_logs_dir / case["audit_filename"])

    # Clear handlers
    if hasattr(mod, "logger"):
        mod.logger.handlers.clear()

    # Call get_logger with console multiple times
    logger1 = mod.get_logger(log_to_console=True)
    logger2 = mod.get_logger(log_to_console=True)
    logger3 = mod.get_logger(log_to_console=True)

    # Count StreamHandler instances (console handlers)
    console_handlers = [
        h
        for h in logger1.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
    ]

    # Should only have one console handler
    assert len(console_handlers) == 1


@pytest.mark.parametrize("case", LOGGER_CASES)
def test_rotating_file_handler_configuration(case, temp_logs_dir, monkeypatch):
    """Test that RotatingFileHandler is configured correctly."""
    from logging.handlers import RotatingFileHandler

    mod = _import_logger_module(case)

    if hasattr(mod, "LOGS_DIR"):
        monkeypatch.setattr(mod, "LOGS_DIR", temp_logs_dir)
    if hasattr(mod, "LOG_FILE"):
        monkeypatch.setattr(mod, "LOG_FILE", temp_logs_dir / case["log_filename"])
    if hasattr(mod, "AUDIT_LOG"):
        monkeypatch.setattr(mod, "AUDIT_LOG", temp_logs_dir / case["audit_filename"])

    # Clear handlers and recreate
    if hasattr(mod, "logger"):
        mod.logger.handlers.clear()

    logger = mod.get_logger()

    # Find the RotatingFileHandler
    rfh = None
    for h in logger.handlers:
        if isinstance(h, RotatingFileHandler):
            rfh = h
            break

    assert rfh is not None
    assert rfh.maxBytes == 5 * 1024 * 1024  # 5 MB
    assert rfh.backupCount == 5


@pytest.mark.parametrize("case", LOGGER_CASES)
def test_logger_debug_not_logged_by_default(case, temp_logs_dir, monkeypatch):
    """Test that DEBUG messages are not logged at INFO level."""
    mod = _import_logger_module(case)

    if hasattr(mod, "LOGS_DIR"):
        monkeypatch.setattr(mod, "LOGS_DIR", temp_logs_dir)
    if hasattr(mod, "LOG_FILE"):
        monkeypatch.setattr(mod, "LOG_FILE", temp_logs_dir / case["log_filename"])
    if hasattr(mod, "AUDIT_LOG"):
        monkeypatch.setattr(mod, "AUDIT_LOG", temp_logs_dir / case["audit_filename"])

    # Clear handlers
    if hasattr(mod, "logger"):
        mod.logger.handlers.clear()

    # Recreate file handler for this test
    from logging.handlers import RotatingFileHandler

    log_file = temp_logs_dir / case["log_filename"]
    rfh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
    if hasattr(mod, "rfh"):
        monkeypatch.setattr(mod, "rfh", rfh)

    logger = mod.get_logger()
    logger.debug("Debug message")

    # DEBUG should not appear since logger level is INFO
    content = log_file.read_text()
    assert "Debug message" not in content


@pytest.mark.parametrize("case", LOGGER_CASES)
def test_logger_different_levels(case, temp_logs_dir, monkeypatch):
    """Test logging at different levels in a single test."""
    mod = _import_logger_module(case)

    if hasattr(mod, "LOGS_DIR"):
        monkeypatch.setattr(mod, "LOGS_DIR", temp_logs_dir)
    if hasattr(mod, "LOG_FILE"):
        monkeypatch.setattr(mod, "LOG_FILE", temp_logs_dir / case["log_filename"])
    if hasattr(mod, "AUDIT_LOG"):
        monkeypatch.setattr(mod, "AUDIT_LOG", temp_logs_dir / case["audit_filename"])

    # Clear handlers
    if hasattr(mod, "logger"):
        mod.logger.handlers.clear()

    # Recreate file handler for this test
    from logging.handlers import RotatingFileHandler

    log_file = temp_logs_dir / case["log_filename"]
    rfh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
    if hasattr(mod, "rfh"):
        monkeypatch.setattr(mod, "rfh", rfh)

    logger = mod.get_logger()
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    content = log_file.read_text()

    assert "[INFO]" in content
    assert "[WARNING]" in content
    assert "[ERROR]" in content
    assert "Info message" in content
    assert "Warning message" in content
    assert "Error message" in content


@pytest.mark.parametrize("case", LOGGER_CASES)
def test_logs_directory_exists(case, temp_logs_dir, monkeypatch):
    """Test that LOGS_DIR exists and is accessible."""
    mod = _import_logger_module(case)

    if hasattr(mod, "LOGS_DIR"):
        monkeypatch.setattr(mod, "LOGS_DIR", temp_logs_dir)

    assert temp_logs_dir.exists()
    assert temp_logs_dir.is_dir()

    # Verify we can write to the directory
    test_file = temp_logs_dir / "test_write.txt"
    test_file.write_text("test")
    assert test_file.exists()


@pytest.mark.parametrize("case", LOGGER_CASES)
def test_audit_appends_not_overwrites(case, temp_logs_dir, monkeypatch):
    """Test that audit appends to file rather than overwriting."""
    mod = _import_logger_module(case)

    if hasattr(mod, "LOGS_DIR"):
        monkeypatch.setattr(mod, "LOGS_DIR", temp_logs_dir)
    audit_path = temp_logs_dir / case["audit_filename"]
    if hasattr(mod, "AUDIT_LOG"):
        monkeypatch.setattr(mod, "AUDIT_LOG", audit_path)

    # Clear handlers
    if hasattr(mod, "logger"):
        mod.logger.handlers.clear()

    mod.audit("first_event", {"order": 1})
    mod.audit("second_event", {"order": 2})

    lines = audit_path.read_text().strip().split("\n")

    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert first["event"] == "first_event"
    assert second["event"] == "second_event"
