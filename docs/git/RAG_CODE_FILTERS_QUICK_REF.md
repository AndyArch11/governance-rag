# Quick Reference: RAG Code Filters

## Quick Start

### Use Auto-Detected Filters
```python
from scripts.rag.retrieve import retrieve

# Query is automatically parsed for language and code patterns
chunks, meta = retrieve("Show Java authentication", collection)
```

### Use Explicit Language Filter
```python
chunks, meta = retrieve(
    "authentication patterns",
    collection,
    language_filter="java"  # Restrict to Java code
)
```

### Use Explicit Source Filter
```python
chunks, meta = retrieve(
    "API design",
    collection,
    source_category_filter="code"  # Only code, not docs
)
```

### Use Programmatic Filters
```python
from scripts.rag.retrieve import build_code_filters, retrieve_with_filters

filters = build_code_filters(
    language="python",
    include_dependencies=True
)
chunks, meta = retrieve_with_filters("services", collection, filters=filters)
```

## Supported Languages

| Language | Keywords |
|----------|----------|
| java | java, spring, maven, junit, gradle |
| groovy | groovy, spock |
| kotlin | kotlin, kotlinx |
| python | python, django, flask, pytest |
| go | go, golang, goroutine |
| rust | rust, cargo, tokio |
| javascript | javascript, nodejs, npm, ts, typescript |
| sql | sql, plsql, tsql, hive |
| xml | xml, xpath, xsd |
| yaml | yaml, yml, helm |
| json | json, jsonpath |

## Code-Specific Query Keywords

| Pattern | Result |
|---------|--------|
| "Java services" | language=java, source_category=code |
| "REST endpoints" | source_category=code |
| "library dependencies" | source_category=code |
| "payment API" | source_category=code |
| "microservice" | source_category=code |

## Function Signatures

### retrieve()
```python
def retrieve(
    query: str,
    collection: Collection,
    k: int = 5,
    language_filter: Optional[str] = None,  # NEW
    source_category_filter: Optional[str] = None,  # NEW
) -> Tuple[List[str], List[Dict]]:
```

### build_code_filters()
```python
def build_code_filters(
    language: Optional[str] = None,
    include_dependencies: bool = False,
    include_endpoints: bool = False,
    include_services: bool = False,
) -> Dict[str, Any]:
```

### detect_filters_from_query()
```python
def detect_filters_from_query(query: str) -> Dict[str, Any]:
```

## Test Examples

### Language Detection
```python
from scripts.rag.retrieve import detect_filters_from_query

# Detects language from keywords
filters = detect_filters_from_query("Show me Java services")
assert filters["language"] == "java"
assert filters["source_category"] == "code"
```

### Filter Integration
```python
from scripts.rag.retrieve import retrieve

# Filters applied to ChromaDB query
chunks, meta = retrieve(
    "authentication",
    collection,
    language_filter="java",
    k=5
)
# Results limited to Java code files
```

### Hybrid Search with Filters
```python
# Both vector search AND keyword search use filters
chunks, meta = retrieve(
    "error handling",
    collection,
    language_filter="python"
)
# Returns semantic + keyword matches for Python code
```

## Common Patterns

### Get language from user input
```python
user_query = "Show Java services"
filters = detect_filters_from_query(user_query)
language = filters.get("language")
if language:
    chunks, meta = retrieve(user_query, collection, language_filter=language)
```

### Filter to code only
```python
chunks, meta = retrieve(
    query,
    collection,
    source_category_filter="code"
)
```

### Combine multiple code criteria
```python
filters = build_code_filters(
    language="groovy",
    include_endpoints=True,  # Only REST controllers
    include_dependencies=True  # Only files with deps
)
chunks, meta = retrieve_with_filters(query, collection, filters=filters)
```

### Check what retrieval method was used
```python
chunks, meta = retrieve("auth", collection, language_filter="java")
for chunk, m in zip(chunks, meta):
    method = m.get("retrieval_method")  # "vector" or "keyword"
    print(f"Retrieved via {method}: {chunk[:50]}")
```

## Testing

Run all code filter tests:
```bash
pytest tests/test_retrieve.py::TestCodeFilters -v
pytest tests/test_retrieve.py::TestDetectFiltersFromQuery -v
pytest tests/test_retrieve.py::TestBuildCodeFilters -v
```

Run all retrieve tests (24 total):
```bash
pytest tests/test_retrieve.py -v
```

## Troubleshooting

**Issue:** Language not detected correctly
- **Solution:** Check language_keywords in detect_filters_from_query() - may need custom keywords

**Issue:** Filters not reducing results
- **Solution:** Verify chunks have metadata fields set correctly during ingestion

**Issue:** Keyword search not receiving filters
- **Solution:** Ensure collection.get() call is passing where clause correctly
