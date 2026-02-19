# Quick Reference: Applying Rate Limiting & Retry Logic

## Fast Implementation Guide

### Step 1: Import Required Decorators

```python
from scripts.utils.retry_utils import (
    retry_chromadb_call,
    retry_ollama_call,
    retry_with_backoff,
)
from scripts.utils.rate_limiter import get_rate_limiter
```

---

## Decorator Reference

### For ChromaDB Operations

```python
@retry_chromadb_call(
    max_retries=3,           # Default for queries
    initial_delay=0.5,       # Start with 0.5s delay
    operation_name="custom_operation"  # For logging
)
def your_chromadb_function():
    # Your ChromaDB operation
    return collection.get(...)
```

**When to use:**
- Collection creation/loading
- Vector similarity search
- Metadata filtering
- Document insertion/deletion

**Recommended Settings:**
- Query operations: max_retries=3, initial_delay=0.5
- Write operations: max_retries=5, initial_delay=1.0

---

### For Ollama LLM Calls

```python
@retry_ollama_call(
    max_retries=3,
    initial_delay=1.0,
    operation_name="llm_generate"
)
def your_ollama_function(prompt: str) -> str:
    limiter = get_rate_limiter()
    if limiter:
        limiter.acquire(tokens=1, blocking=True)
    
    return llm.invoke(prompt)
```

**When to use:**
- LLM prompting
- Embedding generation
- Text processing with LLM

**Recommended Settings:**
- max_retries=3, initial_delay=1.0
- Always add rate limiting for LLM calls

---

### For File I/O & Parsing

```python
@retry_with_backoff(
    max_retries=2,
    initial_delay=0.5,
    transient_types=(IOError, MemoryError, TimeoutError),
    operation_name="parse_pdf"
)
def your_io_function():
    # Your file parsing operation
    return parse_file(...)
```

**When to use:**
- PDF/HTML parsing
- File reading
- Network requests (non-API)

**Recommended Settings:**
- max_retries=2-3, initial_delay=0.5-1.0
- Specify transient_types for your use case

---

## Example Implementations

### RAG Query Module - Collection Loading

**File:** `scripts/rag/query.py`

**Without Retry Protection:**
```python
def load_collection(collection_name: str) -> Collection:
    """Load ChromaDB collection."""
    client = PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(name=collection_name)
```

**With Retry Protection:**
```python
from scripts.utils.retry_utils import retry_chromadb_call

@retry_chromadb_call(
    max_retries=3,
    initial_delay=1.0,
    operation_name="load_collection"
)
def load_collection(collection_name: str) -> Collection:
    """Load ChromaDB collection with retry protection."""
    client = PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(name=collection_name)
```

---

### RAG Retrieve Module - Vector Search

**File:** `scripts/rag/retrieve.py`

**Without Retry Protection:**
```python
def retrieve(query: str, collection: Collection, k: int = 5) -> Tuple[List[str], List[Dict]]:
    """Retrieve chunks by similarity."""
    # ... create query_embedding ...
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
    )
```

**With Retry Protection:**
```python
@retry_chromadb_call(
    max_retries=3,
    initial_delay=0.5,
    operation_name="vector_similarity_search"
)
def retrieve(query: str, collection: Collection, k: int = 5) -> Tuple[List[str], List[Dict]]:
    """Retrieve chunks by similarity with retry protection."""
    # ... create query_embedding ...
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
    )
    # ... process results ...
    return ...
```

---

### RAG Generate Module - LLM Invocation

**File:** `scripts/rag/generate.py`

**Without Retry Protection:**
```python
def generate(prompt: str, max_tokens: int = 512) -> str:
    """Generate answer using LLM."""
    llm = _get_llm()
    return llm.invoke(prompt)
```

**With Retry & Rate Limiting:**
```python
from scripts.utils.retry_utils import retry_ollama_call
from scripts.utils.rate_limiter import get_rate_limiter

@retry_ollama_call(
    max_retries=3,
    initial_delay=1.0,
    operation_name="generate_answer"
)
def generate(prompt: str, max_tokens: int = 512) -> str:
    """Generate answer using LLM with retry and rate limiting."""
    # Apply rate limiting
    limiter = get_rate_limiter()
    if limiter:
        limiter.acquire(tokens=1, blocking=True)
    
    # Invoke LLM (retry decorator handles failures)
    llm = _get_llm()
    return llm.invoke(prompt)
```

---

### PDFParser Module - File Parsing

**File:** `scripts/ingest/pdfparser.py`

**Without Retry Protection:**
```python
def parse_pdf(pdf_path: str) -> Dict[str, Any]:
    """Parse PDF and extract content."""
    pdf = PdfReader(pdf_path)
    # ... parsing logic ...
```

**With Retry Protection:**
```python
from scripts.utils.retry_utils import retry_with_backoff

@retry_with_backoff(
    max_retries=3,
    initial_delay=1.0,
    transient_types=(IOError, MemoryError),
    operation_name="parse_pdf"
)
def parse_pdf(pdf_path: str) -> Dict[str, Any]:
    """Parse PDF and extract content with retry protection."""
    pdf = PdfReader(pdf_path)
    # ... parsing logic ...
```

---

## Testing

### Unit Test Template

```python
import pytest
from unittest.mock import patch, MagicMock

def test_function_with_retry_on_transient_error():
    """Test that function retries on transient failures."""
    with patch("module.external_service") as mock_service:
        # First call fails (transient), second succeeds
        mock_service.side_effect = [
            ConnectionError("Network timeout"),
            {"status": "success"}
        ]
        
        result = your_function()
        assert result == {"status": "success"}
        assert mock_service.call_count == 2  # Verify retry happened

def test_function_fails_fast_on_hard_error():
    """Test that function fails immediately on hard errors."""
    with patch("module.external_service") as mock_service:
        mock_service.side_effect = ValueError("Invalid input")
        
        with pytest.raises(ValueError):
            your_function()
        
        assert mock_service.call_count == 1  # No retries on hard error
```

---

## Debugging Retry Issues

### Enable Audit Logging

```python
# In config or __init__
import logging
from scripts.utils.logger import get_logger

logger = get_logger("my_module")
logger.info("Retry and rate limiting enabled")
```

### Monitor Retry Attempts

```bash
# Watch retry events across all modules in real-time
tail -f logs/*_audit.jsonl | grep "retry"

# Count by operation (ingest module)
grep "retry_attempt_failed" logs/ingest_audit.jsonl | \
    jq -r '.data.operation' | sort | uniq -c

# Find retry exhaustion (all modules)
grep "retry_exhausted" logs/*_audit.jsonl | \
    jq '.data | {operation, exception, attempts}'

# Check RAG-specific retries
grep "retry" logs/rag_audit.jsonl | jq '.data'
```

---

## Backoff Strategy

All retry decorators use exponential backoff with jitter:

```
Retry 1: initial_delay (default 0.5-1.0s)
Retry 2: initial_delay * 2 + jitter
Retry 3: initial_delay * 4 + jitter
...
```

This prevents "thundering herd" problem when multiple processes fail simultaneously.

---

## Rate Limiting Configuration

### Initialise Global Rate Limiter

```python
from scripts.utils.rate_limiter import init_rate_limiter

# In your main() function
rate_limiter = init_rate_limiter(rate=10.0)  # 10 LLM calls/second
```

### Verify Rate Limiter is Active

```python
from scripts.utils.rate_limiter import get_rate_limiter

limiter = get_rate_limiter()
if limiter:
    stats = limiter.get_stats()
    print(f"Rate: {stats['rate']} calls/sec")
    print(f"Available tokens: {stats['available_tokens']}")
else:
    print("Rate limiter not initialised!")
```

---

## Common Issues & Solutions

### Issue: Decorator not working
**Solution:** Ensure import is from `scripts.utils` 
```python
# ✅ CORRECT
from scripts.utils.retry_utils import retry_chromadb_call

# ❌ WRONG
from scripts.ingest.retry_utils import retry_chromadb_call
```

### Issue: Rate limiter not blocking
**Solution:** Make sure you call `limiter.acquire()`
```python
limiter = get_rate_limiter()
if limiter:
    limiter.acquire(tokens=1, blocking=True)  # This blocks if no tokens
```

### Issue: Too many retries causing slowdown
**Solution:** Adjust `max_retries` and `initial_delay`
```python
# More aggressive for transient services
@retry_chromadb_call(max_retries=2, initial_delay=0.3)

# More tolerant for flaky networks
@retry_chromadb_call(max_retries=5, initial_delay=1.0)
```

### Issue: Hard errors being retried
**Solution:** Verify transient_types are correct
```python
# Only retry on these specific transient errors
@retry_with_backoff(
    transient_types=(IOError, MemoryError, TimeoutError),
    # NOT: ValueError, TypeError (these are hard errors)
)
```

---

## Recommended Settings by Service

### ChromaDB (Local Service)
```python
@retry_chromadb_call(max_retries=3, initial_delay=0.5)
# Use fewer retries (local service, fast recovery)
```

### Ollama (GPU Service)
```python
@retry_ollama_call(max_retries=3, initial_delay=1.0)
# Include rate limiting for concurrent calls
```

### External APIs (Bitbucket, etc.)
```python
@retry_with_backoff(max_retries=5, initial_delay=2.0)
# More retries for unreliable networks
```

### File I/O
```python
@retry_with_backoff(max_retries=2, initial_delay=0.5, transient_types=(IOError,))
# Fewer retries (usually indicates real problem)
```

---

## Verification Checklist

Before deploying:

- [ ] Decorator imports from `scripts.utils`
- [ ] Decorator placed above function definition
- [ ] Rate limiter initialised in main()
- [ ] Tests verify retry behaviour on transient errors
- [ ] Tests verify fail-fast on hard errors
- [ ] Audit logging enabled in config
- [ ] Performance baseline measured
- [ ] Monitoring dashboard updated

