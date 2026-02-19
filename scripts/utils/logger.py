"""Unified logging module for all RAG project components.

Provides centralised logging configuration with support for:
- Rotating file handlers to manage disk space
- Structured audit logging in JSONL format
- UTC timestamps for consistency across systems
- Optional console output for debugging
- Multiple logger instances per module (ingest, rag, consistency)

Configuration:
- Log files: logs/{module}.log (max 5 MB per file, 5 backups retained)
- Audit logs: logs/{module}_audit.jsonl (structured audit events)
- Level: INFO by default
- Formatter: ISO 8601 timestamps with severity and message

Usage:
    from scripts.utils.logger import get_logger, audit

    logger = get_logger("ingest", log_to_console=True)
    logger.info("Processing document")
    audit("ingest", "document_processed", {"doc_id": "123", "status": "ok"})
"""

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

# ============================================================================
# Configuration
# ============================================================================

LOGGING_LEVEL = logging.INFO

# Central logs directory (prefer shared configuration)
try:
    from scripts.utils.config import BaseConfig

    LOGS_DIR = BaseConfig().logs_dir
except Exception:
    LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Cache for logger instances
_loggers: Dict[str, logging.Logger] = {}

# Formatter for all loggers
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


# ============================================================================
# Logging Functions
# ============================================================================


def get_logger(module_name: str = "app", log_to_console: bool = False) -> logging.Logger:
    """Get a configured logger instance for a specific module.

    Returns a singleton logger per module that writes to rotating file log and
    optionally to console. Guards against adding duplicate handlers on repeated calls.

    Args:
        module_name: Name of the module (e.g., 'ingest', 'rag', 'consistency')
        log_to_console: If True, also stream log messages to stdout/stderr

    Returns:
        Configured logger instance ready for use

    Example:
        >>> logger = get_logger("ingest", log_to_console=True)
        >>> logger.info("System initialised")
    """
    # Return cached logger if it exists
    if module_name in _loggers:
        logger = _loggers[module_name]

        # Add console handler if requested and not already present
        if log_to_console:
            has_console = any(
                isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
                for h in logger.handlers
            )
            if not has_console:
                ch = logging.StreamHandler()
                ch.setLevel(LOGGING_LEVEL)
                ch.setFormatter(formatter)
                logger.addHandler(ch)

        return logger

    # Create new logger
    logger = logging.getLogger(module_name)
    logger.setLevel(LOGGING_LEVEL)
    logger.propagate = False

    # Add rotating file handler
    log_file = LOGS_DIR / f"{module_name}.log"
    rfh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)  # 5 MB
    rfh.setLevel(LOGGING_LEVEL)
    rfh.setFormatter(formatter)
    logger.addHandler(rfh)

    # Add console handler if requested
    if log_to_console:
        ch = logging.StreamHandler()
        ch.setLevel(LOGGING_LEVEL)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Cache and return
    _loggers[module_name] = logger
    return logger


# ============================================================================
# Audit Functions
# ============================================================================


def audit(module_name: str, event_type: str, data: Dict[str, Any]) -> None:
    """Write a structured JSONL audit entry for analytics and dashboards.

    Appends a JSON-serialised event to the module's audit log with ISO timestamp.
    Each line is a complete JSON object (JSONL format) for easy parsing.

    Args:
        module_name: Name of the module (e.g., 'ingest', 'rag', 'consistency')
        event_type: Type of event (e.g., 'query_start', 'document_processed')
        data: Event metadata dictionary (arbitrary key-value pairs)

    Example:
        >>> audit("rag", "query_start", {"query": "What is MFA?", "k": 5})
        # Writes to logs/rag_audit.jsonl:
        # {"timestamp": "2026-01-16T14:30:45.123456+00:00", "event": "query_start", "query": "What is MFA?", "k": 5}

    Note: Appends to file without locking; in high-concurrency scenarios,
    consider using a queue or dedicated logging service.
    """
    audit_log = LOGS_DIR / f"{module_name}_audit.jsonl"
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event_type, **data}
    with open(audit_log, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ============================================================================
# Backward Compatibility Helpers
# ============================================================================


def create_module_logger(module_name: str):
    """Create backward-compatible logger functions for a specific module.

    Returns a tuple of (get_logger_func, audit_func) that can be used
    to replace module-specific logger imports.

    Args:
        module_name: Name of the module (e.g., 'ingest', 'rag', 'consistency')

    Returns:
        Tuple of (get_logger function, audit function)
    """

    def module_get_logger(log_to_console: bool = False) -> logging.Logger:
        return get_logger(module_name, log_to_console)

    def module_audit(event_type: str, data: Dict[str, Any]) -> None:
        audit(module_name, event_type, data)

    return module_get_logger, module_audit


def flush_all_handlers(logger: logging.Logger) -> None:
    """Flush all handlers attached to a logger to ensure buffered messages are written.

    This is important when exiting early or before a critical point where log
    messages must be persisted before the process potentially crashes.
    Handles edge cases like None logger or mock/dummy loggers in tests.

    Args:
        logger: Logger instance to flush
    """
    if logger is None:
        return
    if not hasattr(logger, "handlers"):
        return
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass  # Best-effort flush


def configure_child_logger_propagation(parent_module_name: str, child_prefix: str) -> None:
    """Configure child loggers to propagate to parent module logger.

    Useful for ensuring that loggers like "academic.providers.*" messages
    propagate to and are captured by the "academic_ingest" logger.

    Args:
        parent_module_name: Name of parent logger module (e.g., 'academic_ingest')
        child_prefix: Prefix for child loggers (e.g., 'academic.providers')
    """
    parent_logger = logging.getLogger(parent_module_name)

    # Create a handler that captures messages from child loggers
    child_root = logging.getLogger(child_prefix.split(".")[0])  # e.g., 'academic'
    child_root.propagate = False

    # Add parent's handlers to child root logger so messages propagate
    for handler in parent_logger.handlers:
        if handler not in child_root.handlers:
            child_root.addHandler(handler)
