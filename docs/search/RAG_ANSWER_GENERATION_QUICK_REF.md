# Quick Reference: Code-Aware Answer Generation

## Quick Start

### Code Query Detection
```python
from scripts.rag.generate import answer

# Automatically detects and formats code responses
response = answer("Show Java authentication services", collection)

# Check if code query was detected
if response["is_code_query"]:
    print("Code formatting applied!")
    print("Git hosting links included!")
```

### Code-Aware Prompt Building
```python
from scripts.rag.assemble import build_code_aware_prompt

chunks = ["@Service public class Auth { }"]
metadata = [{"language": "java", "service_name": "AuthService"}]

prompt = build_code_aware_prompt("Show auth service", chunks, metadata)
```

### Format Code Responses
```python
from scripts.rag.assemble import format_code_response, include_git_links

answer = "The service uses: public class Auth { }"
formatted = format_code_response(answer, language="java")
enhanced = include_git_links(formatted, metadata)
```

## Functions Reference

| Function | Purpose | File |
|----------|---------|------|
| `build_code_aware_prompt()` | Build code-specific prompts | assemble.py |
| `extract_language_from_metadata()` | Get language from metadata | assemble.py |
| `format_code_response()` | Add markdown code blocks | assemble.py |
| `include_git_links()` | Append Git hosting links | assemble.py |
| `_is_code_query()` | Detect code queries | generate.py |
| `_enhance_code_response()` | Orchestrate enhancements | generate.py |
| `answer()` | Full pipeline (updated) | generate.py |

## Response Structure

```python
response = {
    "answer": str,              # Enhanced with formatting/links if code query
    "chunks": List[str],        # Retrieved context
    "sources": List[Dict],      # Metadata for each chunk
    "generation_time": float,   # LLM inference time (seconds)
    "total_time": float,        # End-to-end time (seconds)
    "retrieval_count": int,     # Number of chunks
    "model": str,               # LLM model name
    "temperature": float,       # Temperature setting
    "is_code_query": bool,      # NEW: Code query detection flag
}
```

## Query Type Detection

### Code Queries (is_code_query=True)
- "Show Java services"
- "Find REST API endpoints"
- "Which services use JDBC?"
- "Groovy REST controller examples"
- "Python payment service implementation"

### Governance Queries (is_code_query=False)
- "What is MFA?"
- "Describe security policies"
- "What are our compliance requirements?"
- "Explain our authentication strategy"

## System Prompts

### Standard Prompt (Governance)
```
You are a technical assistant specialising in governance, security, and infrastructure policies.
- Use ONLY the provided context
- Cite chunk numbers when using information
- Do NOT invent information
```

### Code-Aware Prompt
```
You are a technical assistant specialising in code analysis, architecture, and implementation patterns.
- Format code snippets in markdown blocks with language specification
- Include service names, language, and dependencies when relevant
- Provide BitBucket links where available
- Acknowledge when multiple implementations or approaches exist
```

## Metadata Fields Supported

| Field | Example | Used For |
|-------|---------|----------|
| `language` | "java" | Code formatting, prompt context |
| `service_name` | "AuthService" | Prompt metadata, response context |
| `dependencies` | "Spring Framework" | Prompt context, architecture info |
| `bitbucket_url` | "https://bitbucket.com/..." | Source links in response |
| `retrieval_method` | "vector" or "keyword" | Response attribution |

## Audit Logging

All response generation logged with code query flag:
```python
# Audit event for code query
audit("answer_generated", {
    "query_length": 40,
    "chunks_retrieved": 5,
    "generation_time": 1.23,
    "total_time": 2.45,
    "model": "mistral",
    "temperature": 0.3,
    "is_code_query": True,  
})

# Audit event for governance query
audit("answer_generated", {
    "query_length": 35,
    "chunks_retrieved": 3,
    "generation_time": 0.87,
    "total_time": 1.65,
    "model": "mistral",
    "temperature": 0.3,
    "is_code_query": False,  
})
```

## Common Patterns

### Detect and Handle Code Responses
```python
response = answer(user_query, collection)

if response["is_code_query"]:
    # Response has formatted code and BitBucket links
    display_code_response(response["answer"])
else:
    # Standard governance response
    display_text_response(response["answer"])
```

### Extract All Code Snippets from Response
```python
import re

def extract_code_blocks(response_text):
    """Extract all code blocks from response."""
    pattern = r'```(\w+)?\n(.*?)\n```'
    matches = re.findall(pattern, response_text, re.DOTALL)
    return [(lang, code) for lang, code in matches]

if response["is_code_query"]:
    code_blocks = extract_code_blocks(response["answer"])
    for lang, code in code_blocks:
        print(f"Language: {lang}")
        print(f"Code:\n{code}")
```

### Track Code vs. Governance Query Performance
```python
metrics = {
    "code_queries": 0,
    "gov_queries": 0,
    "avg_code_time": 0.0,
    "avg_gov_time": 0.0,
}

response = answer(query, collection)

if response["is_code_query"]:
    metrics["code_queries"] += 1
    metrics["avg_code_time"] = (
        (metrics["avg_code_time"] * (metrics["code_queries"] - 1) + response["total_time"]) 
        / metrics["code_queries"]
    )
else:
    metrics["gov_queries"] += 1
    metrics["avg_gov_time"] = (
        (metrics["avg_gov_time"] * (metrics["gov_queries"] - 1) + response["total_time"]) 
        / metrics["gov_queries"]
    )
```

## Testing

Run code-aware answer generation tests:
```bash
pytest tests/test_generate.py::TestCodeAwareAnswerGeneration -v
```

Run all generate tests (15 total):
```bash
pytest tests/test_generate.py -v
```

Run full rag test suite:
```bash
pytest tests/test_retrieve.py tests/test_generate.py -q
```

## Integration with Code Aware Filters

Code-aware answer generation automatically uses filters:

1. **Query Detection:** Uses `detect_filters_from_query()` to identify code queries
2. **Metadata Context:** Uses enriched metadata from retrieval
3. **Language Detection:** Uses language detected by code aware filters
4. **Source Filtering:** Works with language_filter and source_category_filter

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Code not being detected as code query | Check query keywords match language/pattern detection |
| No BitBucket links in response | Verify metadata contains `bitbucket_url` field |
| Code formatting not applied | Check language is specified in metadata |
| Standard prompt used for code query | Verify filter detection working |
| Metadata not in prompt | Check `build_code_aware_prompt()` is called |

## Performance Tips

1. **Cache Detection:** Filters cached for identical queries
2. **Metadata Optimisation:** Include only relevant metadata fields
3. **Language Hints:** Always provide language in metadata when possible
4. **URL Deduplication:** BitBucket link deduplication handles duplicates
5. **Prompt Size:** Code-aware prompts similar size to standard prompts


