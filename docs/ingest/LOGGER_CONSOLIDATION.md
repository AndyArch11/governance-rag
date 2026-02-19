# Logger Consolidation

This document describes the unified logging architecture in the RAG project.

## Overview

All logging functionality is consolidated in a single module: `scripts.utils.logger`. 

## Architecture

### Unified Logger Module

**Location**: `scripts/utils/logger.py`

The unified logger provides:
- **Module-based logging**: Each component gets its own logger instance and log file
- **Rotating file handlers**: Automatic log rotation at 5MB with 5 backup files
- **JSONL audit trails**: Structured event logging for analytics
- **Console output**: Optional console logging for debugging
- **Logger caching**: Reuses logger instances to prevent duplicate handlers

### Key Functions

#### `get_logger(module_name: str, log_to_console: bool = False) -> logging.Logger`

Creates or retrieves a cached logger for the specified module.

**Parameters**:
- `module_name`: Name of the module (e.g., "ingest", "rag", "consistency")
- `log_to_console`: If True, also logs to console/stdout

**Returns**: Configured `logging.Logger` instance

**Example**:
```python
from scripts.utils.logger import get_logger

logger = get_logger("ingest", log_to_console=True)
logger.debug("Function x: var y = 10")
logger.info("Processing document...")
logger.warning("Rate limit approaching")
logger.error("Failed to connect", exc_info=True)
```

**Log File**: `logs/{module_name}.log`

#### `audit(module_name: str, event_type: str, data: Dict[str, Any]) -> None`

Writes a structured JSONL audit entry for analytics and dashboards.

**Parameters**:
- `module_name`: Name of the module
- `event_type`: Type of event (e.g., "document_processed", "query_start")
- `data`: Dictionary of event metadata

**Example**:
```python
from scripts.utils.logger import audit

audit("ingest", "document_processed", {
    "doc_id": "123",
    "chunks": 45,
    "source": "confluence",
    "processing_time_ms": 1250
})
```

**Audit File**: `logs/{module_name}_audit.jsonl`

#### `create_module_logger(module_name: str) -> Tuple[Callable, Callable]`

Creates backward-compatible logger functions for a specific module.

**Returns**: Tuple of `(get_logger_func, audit_func)` that match the old API

**Example**:
```python
from scripts.utils.logger import create_module_logger

get_logger, audit = create_module_logger("ingest")

logger = get_logger(log_to_console=True)
audit("event_type", {"data": "value"})
```

Many modules use this approach for backward-compatible API:
- `scripts/ingest/preprocess.py`
- `scripts/ingest/vectors.py`
- Other modules using module-level `get_logger` and `audit` functions

## Log Files

All logs are written to the project root `logs/` directory:

| Module | Log File | Audit File |
|--------|----------|------------|
| ingest | `logs/ingest.log` | `logs/ingest_audit.jsonl` |
| rag | `logs/rag.log` | `logs/rag_audit.jsonl` |
| consistency_graph | `logs/consistency.log` | `logs/consistency_audit.jsonl` |

### Log Rotation

- **Max size**: 5 MB per file
- **Backups**: 5 files retained
- **Naming**: `{module}.log`, `{module}.log.1`, `{module}.log.2`, etc.

### Audit Format

Each audit entry is a single-line JSON object:

```json
{"timestamp": "2025-01-10T14:30:45.123456+00:00", "event": "document_processed", "doc_id": "123", "chunks": 45}
```

## Development Guide

### For New Code

Use the unified logger directly:

```python
from scripts.utils.logger import get_logger, audit

logger = get_logger("my_module")
logger.info("Starting process")

audit("my_module", "process_complete", {"status": "success"})
```

Or use `create_module_logger` for module-level logger functions:

```python
from scripts.utils.logger import create_module_logger

get_logger, audit = create_module_logger("my_module")

logger = get_logger(log_to_console=True)
audit("process_complete", {"status": "success"})
```

## Testing

All logger functionality is tested in `tests/test_utils_logger.py`:

Run tests:
```bash
pytest tests/test_utils_logger.py -v
```

## Benefits

1. **Single Source of Truth**: One module for all logging configuration
2. **Consistency**: All modules use the same logging format and behaviour
3. **Maintainability**: Changes to logging only need to be made in one place
4. **Flexibility**: Module-based approach allows per-component configuration
5. **Backward Compatible**: No breaking changes to existing code
6. **Better Testing**: Centralised testing of logging functionality

## Configuration

The logger uses these defaults from `scripts.utils.logger`:

```python
LOGGING_LEVEL = logging.INFO
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 5
```

To change the logging level for a specific module:

```python
logger = get_logger("my_module")
logger.setLevel(logging.DEBUG)
```

## Advanced Usage

### Custom Formatters

```python
from scripts.utils.logger import get_logger

logger = get_logger("custom")

# Add custom handler
import logging
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(name)s - %(message)s'))
logger.addHandler(handler)
```

### Filtering Audit Logs

```bash
# Get all document_processed events
grep '"event": "document_processed"' logs/ingest_audit.jsonl

# Count events by type
jq -r '.event' logs/ingest_audit.jsonl | sort | uniq -c

# Filter by timestamp
jq 'select(.timestamp > "2025-01-10")' logs/ingest_audit.jsonl
```

## TODO: Future Enhancements

Potential improvements:
- Structured logging with Python `logging.dictConfig`
- Log aggregation to centralised service (e.g., Elasticsearch)
- Metrics collection from audit logs
- Configurable log levels per module via config file
- Async logging for high-throughput scenarios
