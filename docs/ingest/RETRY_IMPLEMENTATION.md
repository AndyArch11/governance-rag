# Retry and Failure Handling Implementation

## Overview

Retry/backoff logic for Ollama LLM operations and ChromaDB database operations with intelligent classification of transient vs hard failures.


### Module: `ingest/retry_utils.py`

**Purpose:** Centralised retry logic with exponential backoff for handling transient failures in external service calls.

**Key Features:**
- Exception classification (transient vs hard failures)
- Exponential backoff with jitter to prevent thundering herd
- Configurable retry limits and delays
- Audit logging for all retry attempts
- Specialised wrappers for Ollama and ChromaDB operations

### Exception Classification

**Transient Errors (Will Retry):**
- Network errors: `ConnectionError`, `TimeoutError`, `ConnectionRefusedError`
- HTTP rate limits: `429 Too Many Requests`
- HTTP server errors: `502 Bad Gateway`, `503 Service Unavailable`, `504 Gateway Timeout`
- ChromaDB connection/timeout issues
- Ollama service errors: model loading, CUDA out of memory, GPU errors
- Any error with "timeout", "timed out", "deadline exceeded" in message
- Any error with "rate limit", "too many requests", "throttle" in message

**Hard Failures (Fail Immediately):**
- Validation errors: `pydantic.ValidationError`, `ValueError`, `TypeError`
- HTTP client errors: `400 Bad Request`, `401 Unauthorised`, `403 Forbidden`, `404 Not Found`
- Programming errors: `KeyError`, `AttributeError`, `NotImplementedError`
- ChromaDB NotFoundError
- Unknown/unexpected errors (fail fast principle)

### Retry Configuration

**Ollama Operations:**
- Max retries: 3
- Initial delay: 1.0s
- Backoff factor: 2.0x
- Max delay cap: 30s
- Jitter: enabled

**ChromaDB Operations:**
- Max retries: 5
- Initial delay: 0.5s
- Backoff factor: 2.0x
- Max delay cap: 20s
- Jitter: enabled

### Used By:

#### 1. `preprocess.py`
- Imported `retry_ollama_call` decorator
- Wrapped `llm_invoke_with_rate_limit()` with retry logic

**Operations Protected:**
- LLM metadata generation
- Summary scoring
- Summary regeneration

#### 2. `vectors.py`
- Imported `retry_ollama_call` and `retry_chromadb_call` decorators
- Wrapped `validator_llm.invoke()` calls with retry logic
- Wrapped `chunk_collection.add()` and `doc_collection.add()` operations
- Wrapped `collection.get()` queries
- Wrapped `collection.delete()` operations
- Added retry to embedding generation (both batch and single-doc)

**Operations Protected:**
- Semantic drift detection LLM calls
- Chunk validation LLM calls
- Chunk repair LLM calls
- Chunk storage in ChromaDB
- Document storage in ChromaDB
- Previous version metadata retrieval
- Document hash retrieval
- Document chunk deletion
- Embedding generation (Ollama)

#### 3. `ingest.py`
- Imported `retry_chromadb_call` decorator
- Wrapped version assignment `collection.get()`
- Wrapped prune operations `collection.get()` and `collection.delete()`
- Wrapped chunk idempotency `collection.get()`

**Operations Protected:**
- Version metadata queries
- Old version pruning (count queries and deletions)
- Chunk idempotency checks

### Audit Events

**Audit Events:**
- `retry_attempt_failed`: Logged on each failed attempt (includes attempt number, error type, failure classification)
- `retry_success`: Logged when retry succeeds after initial failures
- `hard_failure`: Logged when hard failure detected (no retry)
- `retry_exhausted`: Logged when all retries exhausted

**Audit Metadata:**
- `operation`: Name of the operation being retried
- `attempt`: Current attempt number
- `max_retries`: Maximum retry limit
- `error_type`: Exception class name
- `error_message`: First 200 chars of error message
- `failure_type`: "transient" or "hard"
- `will_retry`: Boolean indicating if retry will be attempted

### Usage Examples

**Direct Decorator Usage:**
```python
from retry_utils import retry_ollama_call, retry_chromadb_call

@retry_ollama_call(max_retries=3, initial_delay=1.0)
def call_llm(prompt):
    return llm.invoke(prompt)

@retry_chromadb_call(max_retries=5, initial_delay=0.5)
def query_db(collection, filter):
    return collection.get(where=filter)
```

**Inline Usage (for dynamic operations):**
```python
from retry_utils import retry_chromadb_call

# Wrap operation-specific calls
@retry_chromadb_call(max_retries=5, operation_name=f"delete_chunks({doc_id})")
def _delete_chunks():
    chunk_collection.delete(where={"doc_id": doc_id})

_delete_chunks()
```

### Testing

**Test Coverage:**
- Exception classification logic
- Retry decorator behaviour
- Exponential backoff timing
- Max delay capping
- Audit event generation
- Ollama-specific retry behaviour
- ChromaDB-specific retry behaviour
- Integration scenarios (intermittent failures, rate limiting, permanent failures)

## Behaviour Changes

### Without
- Network timeouts cause immediate job failure
- Rate limiting errors terminates processing
- Transient ChromaDB connection issues fails insertion/retrieval documents
- Ollama model loading delays results in errors
- No distinction between transitory and permanent failures

### With
- Network timeouts automatically retry up to 3-5 times
- Rate limiting triggers backoff and retry
- ChromaDB connection issues resolved through retry
- Ollama service errors handled gracefully with delays
- Hard failures (validation, programming errors) fail fast without wasted retries
- All retry attempts logged for debugging and monitoring

## Performance Impact

**Positive:**
- Reduced failed document count due to transient errors
- Better resilience to network/service instability
- Improved overall throughput in unstable environments

**Considerations:**
- Transient failures now take longer to surface (due to retries)
- Additional log volume from retry audit events
- Slight increase in processing time for documents with transient errors

**Mitigation:**
- Conservative retry limits prevent excessive delays
- Exponential backoff avoids hammering failing services
- Hard failures detected early without retry overhead
- Jitter prevents synchronised retry storms

## Monitoring

**Key Metrics to Monitor:**
1. **Retry rate:** Count of `retry_attempt_failed` events
2. **Success after retry:** Count of `retry_success` events
3. **Exhausted retries:** Count of `retry_exhausted` events
4. **Hard failures:** Count of `hard_failure` events
5. **Retry by operation:** Group audit events by `operation` field

See retry_metrics notebook for example audit python queries

**Example Audit Query (using jq):**
```bash
# Count retries by operation
grep "retry_attempt_failed" logs/[module]_audit.jsonl | jq -r '.metadata.operation' | sort | uniq -c

# Find operations exhausting retries
grep "retry_exhausted" logs/[module]_audit.jsonl | jq '.metadata.operation'

# Calculate retry success rate
grep "retry_" logs/[module]_audit.jsonl | jq -r '.event' | sort | uniq -c
```

## Recommendations

### For Development/Testing
1. Monitor audit logs during testing to tune retry parameters
2. Add operation-specific retry configs for known problematic operations
3. Consider adding circuit breaker pattern if persistent failures detected

### For Production
1. Set up alerts for high retry rates (indicates service issues)
2. Monitor `retry_exhausted` events (may indicate configuration or infrastructure problems)
3. Track hard failure rates to identify data quality issues
4. Consider adjusting retry limits based on observed failure patterns

### TODO: Future Enhancements
1. **Circuit Breaker:** Temporarily disable retries for consistently failing operations
2. **Adaptive Backoff:** Adjust delays based on service health metrics
3. **Retry Budget:** Limit total retry time per document to prevent runaway processing
4. **Prometheus Metrics:** Export retry metrics for dashboard visualisation
5. **Service-Specific Tuning:** Different retry configs for different Ollama models

## Compatibility

**Python Version:** 3.13+  
**Dependencies:**
- `pydantic` (for ValidationError detection)
- `requests` (for HTTP error handling)
- Existing: `chromadb`, `langchain_ollama`

## Rollback Plan

If issues arise:
1. Remove import statements from `preprocess.py`, `vectors.py`, `ingest.py`, `ingest_git.py`
2. Remove @retry decorator wrappers (operations will execute without retry)
3. System returns to immediate-failure behaviour
4. All tests should still pass with retry logic disabled

## References

- Implementation: `scripts/utils/retry_utils.py`
- Tests: `tests/test_retry_utils.py`
- Utilised in modules: `ingest.py`, `pdfparser.py`, `htmlparser.py`, `bitbuck_connector.py`, `github_connector.py`, `preprocess.py`, `vectors.py`, `query.py`, `retrieve.py`, `generate.py`
- Audit events: `logs/[module]_audit.jsonl` (when running)
