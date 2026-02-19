"""RAG logging module (backward compatibility wrapper).

This module now wraps the unified logger in scripts.utils.logger.
All functionality is preserved for backward compatibility.
"""

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict

from scripts.utils import logger as utils_logger

# Expose attributes for backward compatibility with tests
LOGS_DIR = utils_logger.LOGS_DIR
AUDIT_LOG = LOGS_DIR / "rag_audit.jsonl"
LOGGING_LEVEL = utils_logger.LOGGING_LEVEL

# Get the logger instance
logger = utils_logger.get_logger("rag")

# Create formatter for backward compatibility
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Create rotating file handler for backward compatibility
rfh = RotatingFileHandler(LOGS_DIR / "rag.log", maxBytes=5 * 1024 * 1024, backupCount=5)
rfh.setLevel(LOGGING_LEVEL)
rfh.setFormatter(formatter)


def get_logger(log_to_console: bool = False) -> logging.Logger:
    """Get configured logger instance.

    This wrapper allows tests to monkeypatch LOGS_DIR and other attributes.
    """
    # Use the module-level logger which can be monkeypatched
    global logger, rfh, formatter

    # Ensure rfh has formatter set (in case it was monkeypatched)
    rfh.setLevel(LOGGING_LEVEL)
    rfh.setFormatter(formatter)

    # Add file handler if not present
    has_file_handler = any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
    if not has_file_handler:
        # Use module-level rfh which can be monkeypatched
        if rfh not in logger.handlers:
            logger.addHandler(rfh)

    # Add console handler if requested
    if log_to_console:
        has_console_handler = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
            for h in logger.handlers
        )
        if not has_console_handler:
            ch = logging.StreamHandler()
            ch.setLevel(LOGGING_LEVEL)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

    return logger


def audit(event_type: str, data: Dict[str, Any]) -> None:
    """Write a structured JSONL audit entry.

    This wrapper allows tests to monkeypatch AUDIT_LOG.
    """
    # Use module-level AUDIT_LOG which can be monkeypatched
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event_type, **data}
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


__all__ = [
    "get_logger",
    "audit",
    "LOGS_DIR",
    "AUDIT_LOG",
    "logger",
    "formatter",
    "rfh",
    "LOGGING_LEVEL",
]
