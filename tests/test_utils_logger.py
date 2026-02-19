"""Tests for unified logger in scripts.utils.logger.

Tests the centralised logging module that provides module-based logging
with rotating file handlers and JSONL audit trails.
"""

import json
import sys
from pathlib import Path

import pytest

# Add scripts to path for imports
scripts_path = Path(__file__).parent.parent / "scripts"
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))

from scripts.utils import logger as utils_logger  # noqa: E402


@pytest.fixture
def temp_logs_dir(tmp_path, monkeypatch):
    """Create temporary logs directory for testing."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    # Mock LOGS_DIR to use temp directory
    monkeypatch.setattr(utils_logger, "LOGS_DIR", logs_dir)

    # Clear logger cache to avoid cross-test pollution
    utils_logger._loggers.clear()

    return logs_dir


class TestUnifiedLogger:
    """Tests for the unified logging module."""

    def test_get_logger_creates_module_logger(self, temp_logs_dir):
        """Test that get_logger creates a logger for the specified module."""
        logger = utils_logger.get_logger("test_module")
        assert logger.name == "test_module"
        assert logger.level == utils_logger.LOGGING_LEVEL

    def test_get_logger_caches_loggers(self, temp_logs_dir):
        """Test that get_logger returns the same logger instance for the same module."""
        logger1 = utils_logger.get_logger("test_module")
        logger2 = utils_logger.get_logger("test_module")
        assert logger1 is logger2

    def test_get_logger_creates_log_file(self, temp_logs_dir):
        """Test that get_logger creates a log file for the module."""
        logger = utils_logger.get_logger("test_module")
        logger.info("Test message")

        log_file = temp_logs_dir / "test_module.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    def test_get_logger_with_console(self, temp_logs_dir, caplog):
        """Test that log_to_console=True adds console handler."""
        logger = utils_logger.get_logger("test_module", log_to_console=True)

        # Check that console handler was added
        import logging

        has_console = any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
            for h in logger.handlers
        )
        assert has_console

    def test_audit_creates_jsonl_entry(self, temp_logs_dir):
        """Test that audit creates a JSONL entry with timestamp and data."""
        test_data = {"key1": "value1", "key2": 42}
        utils_logger.audit("test_module", "test_event", test_data)

        audit_file = temp_logs_dir / "test_module_audit.jsonl"
        assert audit_file.exists()

        with open(audit_file) as f:
            line = f.readline()
            entry = json.loads(line)

        assert entry["event"] == "test_event"
        assert entry["key1"] == "value1"
        assert entry["key2"] == 42
        assert "timestamp" in entry

    def test_audit_appends_entries(self, temp_logs_dir):
        """Test that multiple audit calls append to the same file."""
        utils_logger.audit("test_module", "event1", {"id": 1})
        utils_logger.audit("test_module", "event2", {"id": 2})

        audit_file = temp_logs_dir / "test_module_audit.jsonl"
        lines = audit_file.read_text().strip().split("\n")
        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        entry2 = json.loads(lines[1])
        assert entry1["event"] == "event1"
        assert entry2["event"] == "event2"

    def test_audit_handles_special_characters(self, temp_logs_dir):
        """Test that audit handles special characters correctly."""
        special_data = {
            "path": "/home/user/file with spaces.txt",
            "quote": 'She said "hello"',
            "backslash": "C:\\Windows\\System32",
            "unicode": "café ☕ 🚀",
        }
        utils_logger.audit("test_module", "special_chars", special_data)

        audit_file = temp_logs_dir / "test_module_audit.jsonl"
        with open(audit_file) as f:
            entry = json.loads(f.readline())

        assert entry["path"] == "/home/user/file with spaces.txt"
        assert entry["quote"] == 'She said "hello"'
        assert entry["backslash"] == "C:\\Windows\\System32"
        assert entry["unicode"] == "café ☕ 🚀"

    def test_create_module_logger_backward_compatibility(self, temp_logs_dir):
        """Test that create_module_logger provides backward-compatible API."""
        get_logger_func, audit_func = utils_logger.create_module_logger("test_module")

        # Test get_logger function
        logger = get_logger_func()
        assert logger.name == "test_module"

        # Test audit function
        audit_func("test_event", {"data": "value"})

        audit_file = temp_logs_dir / "test_module_audit.jsonl"
        assert audit_file.exists()
        with open(audit_file) as f:
            entry = json.loads(f.readline())
        assert entry["event"] == "test_event"
        assert entry["data"] == "value"

    def test_multiple_modules_separate_logs(self, temp_logs_dir):
        """Test that different modules get separate log files."""
        logger1 = utils_logger.get_logger("module1")
        logger2 = utils_logger.get_logger("module2")

        logger1.info("Module 1 message")
        logger2.info("Module 2 message")

        log1 = temp_logs_dir / "module1.log"
        log2 = temp_logs_dir / "module2.log"

        assert log1.exists()
        assert log2.exists()
        assert "Module 1 message" in log1.read_text()
        assert "Module 2 message" in log2.read_text()
        assert "Module 2 message" not in log1.read_text()
        assert "Module 1 message" not in log2.read_text()


class TestBackwardCompatibilityWrappers:
    """Test that old logger modules still work via backward compatibility wrappers."""

    def test_rag_logger_wrapper(self, temp_logs_dir, monkeypatch):
        """Test that rag_logger wrapper works correctly."""
        # Mock the unified logger's LOGS_DIR
        monkeypatch.setattr(utils_logger, "LOGS_DIR", temp_logs_dir)
        utils_logger._loggers.clear()

        from scripts.rag import rag_logger

        # Also need to monkeypatch the wrapper's AUDIT_LOG since it's set at import time
        monkeypatch.setattr(rag_logger, "AUDIT_LOG", temp_logs_dir / "rag_audit.jsonl")

        # Test get_logger
        logger = rag_logger.get_logger()
        assert logger.name == "rag"

        # Test audit
        rag_logger.audit("test_event", {"data": "test"})
        audit_file = temp_logs_dir / "rag_audit.jsonl"
        assert audit_file.exists()
