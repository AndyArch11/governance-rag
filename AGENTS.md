# AI Assistant Instructions

## Language and Spelling Conventions

**IMPORTANT: Use British/Australian English throughout the codebase.**

### Spelling Standards
- Use British/Australian English spelling in all:
  - Code comments
  - Documentation
  - Docstrings
  - Variable/function names where applicable
  - Commit messages
  - Markdown files
  - Error messages
  - Log messages

### Common Differences to Apply
- `-ise` not `-ize` (visualise, organise, optimise, initialise, normalise, parallelise, serialise, authorise)
- `-our` not `-or` (colour, behaviour, favour, neighbour)
- `-re` not `-er` (centre, metre, litre, fibre)
- `-ogue` not `-og` (catalogue, dialogue, analogue)
- `-ence` not `-ense` (defence, licence [noun], offence)
- `-yse` not `-yze` (analyse, paralyse)
- Double `l` in British forms (cancelled, modelling, travelled)
- `-ogue` endings (catalogue not catalog)

### Technical Terms
When writing technical documentation or comments:
- "authorisation" not "authorization"
- "customisation" not "customization"
- "initialisation" not "initialization"
- "normalisation" not "normalization"
- "optimisation" not "optimization"
- "organisation" not "organization"
- "sanitisation" not "sanitization"
- "serialisable" not "serializable"
- "serialisation" not "serialization"
- "stabilisation" not "stabilization"
- "tokenisation" not "tokenization"
- "utilisation" not "utilization"
- "virtualisation" not "virtualization"
- "visualisation" not "visualization"
- "behaviour" not "behavior"
- "colour" not "color"
- "grey", not "gray"

### Code Style
- Function/variable names: Use British spelling where it makes sense (e.g., `normalise_data`, `optimisation_params`)
- But maintain compatibility: Don't break existing APIs or external library conventions (e.g., `color` in matplotlib parameters)

### Documentation
- README files: British English
- Inline comments: British English
- Docstrings: British English
- Error messages: British English

## Exceptions
- Keep US English only when:
  - Required by external APIs or libraries
  - Part of standard technical terminology universally spelled with US conventions
  - In JSON/config keys that interface with US-centric systems

---

## Code Quality Standards

### Type Safety
- **Always use type hints**: All function signatures must include parameter and return types
- Use proper typing imports: `from typing import List, Dict, Optional, Tuple, Any, Callable`
- Prefer specific types over `Any` where possible
- Use `T | None` for nullable values (Python 3.10+ syntax preferred)
- Add type hints to class attributes in `__init__`

### Error Handling
- Use the shared `retry_utils.py` module for all retry logic:
  - `@retry_ollama_call()` for LLM/embedding operations
  - `@retry_chromadb_call()` for vector database operations
  - `@retry_with_backoff()` for custom retry scenarios
- Classify exceptions as transient vs hard failures
- Don't write custom retry loops—use decorators
- Always handle exceptions at appropriate levels
- Use specific exception types, not bare `except:`

### Logging
- Import from centralised logger: `from scripts.utils.logger import get_logger`
- Don't create module-specific logger files
- Use consistent log levels: DEBUG, INFO, WARNING, ERROR
- Include context in log messages (operation name, key parameters)
- Use audit logging for critical operations via `audit()` function

### Standard Patterns
- **Schemas**: Import from `scripts.utils.schemas` (centralised dataclasses)
- **Rate limiting**: Use `scripts.utils.rate_limiter`:
  - `RateLimiter` for fixed-rate limiting (simple, fixed calls/sec)
  - `AdaptiveRateLimiter` for self-tuning (monitors latency, errors, auto-adjusts)
  - Use `get_rate_limiter()` or `get_adaptive_rate_limiter()` for global instances
- **Configuration**: Use dataclasses with `from dataclasses import dataclass`
- **Paths**: Use `pathlib.Path` for file operations, not string concatenation
- Keep related functionality in utils modules (don't duplicate)

---

## Communication Style

### Response Guidelines
- **Be concise**: Short, factual responses. No fluff or unnecessary explanations
- **No emojis**: Unless explicitly requested
- **Direct implementation**: Implement changes directly rather than suggesting them
- **No marketing speak**: Avoid phrases like "Let me help you", "Great question", etc.
- **State facts**: "Changed X to Y" not "I've successfully updated X to Y for you"
- **Minimal narration**: Don't announce which tools you're using

### Response Length
- Simple queries: 1-3 sentences maximum
- Code-only responses: Just deliver the code, minimal explanation
- Complex tasks: Brief progress updates, detailed only when needed
- Don't create summary markdown files unless explicitly requested

### When Uncertain
- Don't guess—use tools to discover the answer
- Read files to understand context before editing
- Use grep/semantic search to find patterns
- If still unsure, stop to ask for clarification

---

## Implementation Patterns

### Tool Usage Efficiency
- **Parallel operations**: Use `multi_replace_string_in_file` for multiple independent edits
- **Batch reads**: Read multiple files in parallel when gathering context
- **Targeted searches**: Use specific grep patterns with `includePattern` to narrow results
- Avoid sequential tool calls when parallel execution is possible
- Use `read_file` with large ranges over multiple small reads

### File Operations
- Use `replace_string_in_file` with 3-5 lines of context before/after the change
- Never use `sed`, `awk`, or terminal commands to edit code files
- Use `create_file` only for new files, not to overwrite existing ones
- Check current file contents before editing (files may have changed)

### Testing
- Run targeted tests after changes: `pytest path/to/test_file.py::TestClass::test_method -xvs`
- Don't run full test suite unless requested
- Fix failing tests before moving to next task
- Check for type errors with `get_errors()` after file modifications

---

## Project-Specific Conventions

### Project Structure
```
~/rag-project/
├── data_raw/          # Raw data for ingestion (PDFs, HTML)
├── docs/              # Documentation
├── examples/          # Example scripts and troubleshooting tools
├── logs/              # Application logs
├── models/            # Placeholder for AI models
├── notebooks/         # Jupyter notebooks for verification
├── rag_data/          # All database and cache files
│   ├── chromadb/      # Vector store database
│   ├── consistency_graphs/         # Consistency Graph database
│   ├── cache.db       # SQLite cache (embeddings, LLM, graphs)
│   ├── academic_references.db      # Academic reference cache
│   ├── academic_citation_graph.db  # Citation graph database
│   └── academic_terminology.db     # Domain terminology database
├── repos/             # default location for ingest_git.py to clone repos to
├── scripts/           # Main application code (see Module Structure below)
└── tests/             # pytest unit and integration tests
```

### Module Structure
```
scripts/
├── consistency_graph/  # Graph-based consistency analysis
├── ingest/            # Data ingestion pipeline
│   ├── chunk.py       # Text chunking with adaptive strategies
│   ├── vectors.py     # Embedding generation and ChromaDB ops
│   ├── preprocess.py  # LLM-based preprocessing
│   └── ingest.py      # Main ingestion orchestration, also ingest_git.py and ingest_academic.py
├── rag/               # RAG query pipeline
│   ├── query.py       # Query interface
│   ├── retrieve.py    # Vector retrieval
│   └── generate.py    # Answer generation
├── search/            # Search utilities
│   └── hybrid_seach.py       # Supports BM25 key word hybrid search
├── security/          # Security utilities
│   └── dlp.py         # Masking of sensitive data
├── ui/          # Security utilities
│   └── dashboard.py   # Visualisation of databases
└── utils/             # Shared utilities
    ├── logger.py      # Centralised logging
    ├── retry_utils.py # Retry/backoff decorators
    ├── rate_limiter.py # Token bucket rate limiter
    └── schemas.py     # Shared dataclasses
```

### Import Organisation
1. Standard library imports
2. Third-party imports
3. Local imports (scripts.*)
4. Blank line between groups
5. Use absolute imports: `from scripts.utils.logger import get_logger`
6. Don't use relative imports across packages

### Naming Conventions
- Functions: `snake_case` (e.g., `extract_text_from_pdf`)
- Classes: `PascalCase` (e.g., `BitbucketConnector`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`)
- Private methods: `_leading_underscore` (e.g., `_make_request`)
- Use descriptive names: `embedding_cache` not `cache`, `retry_count` not `count`

### Docstring Style
```python
def function_name(param: str, optional_param: int = 5) -> bool:
    """Brief one-line description.

    More detailed explanation if needed. Describe what the function does,
    not how it does it. Focus on behaviour and contracts.

    Args:
        param: Description of parameter
        optional_param: Description with default noted

    Returns:
        Description of return value

    Raises:
        SpecificException: When this exceptional case occurs

    Example:
        >>> result = function_name("test", optional_param=10)
        >>> # Expected behaviour demonstrated
    """
```

### Anti-Patterns to Avoid
- ❌ Creating duplicate retry logic (use `retry_utils`)
- ❌ Module-specific logger files (use centralised logger)
- ❌ Hardcoded paths without `Path` objects
- ❌ Bare `except:` clauses
- ❌ Missing type hints on function signatures
- ❌ Sequential file edits when parallel is possible
- ❌ Creating markdown summary files after each change
- ❌ Suggesting changes instead of implementing them
- ❌ Adding requested functionality, but not connecting/wiring up the new functionality, leaving it orphaned/unused
- ❌ Using `sed`/`awk` to edit Python files

---

## Workflow Patterns

### Multi-Step Tasks
1. Plan the work (use `manage_todo_list` for complex tasks)
2. Verify names of endpoints, database tables, etc, don't guess what they should be
3. Gather context efficiently (parallel reads/searches)
4. Implement changes (use multi-edit when possible)
5. Use the local .venv when needed, don't attempt to install new Python libraries because not available outside of the project environment.
6. Validate (run tests, check errors)
7. Wire up new functionality and test integration
8. Mark complete and move to next item
9. Don't stop mid-task to ask questions unless clarification or permission is required

### Context Gathering
- Use `semantic_search` for conceptual queries
- Use `grep_search` for exact patterns
- Use `file_search` for glob patterns
- Read surrounding code before editing
- Check imports and dependencies
- If a code improvement is identified such as adding a missing type definition and fix is easy to apply, include in change without being asked
- Create pytest unit test cases before implementing to validate thinking before implementing

### Error Resolution
1. Read the error message carefully
2. Locate the file and line number
3. Read surrounding context
4. Understand the root cause
5. Don't guess what a broken interface call should look like, verify first
6. Fix comprehensively (not just the symptom)
7. Verify with tests
8. If libraries/modules are missing, make sure .venv is loaded, don't automatically attempt to download missing libraries/modules

---

## Technology Stack

### Core Dependencies
- **Python 3.10+**: Use modern type hints (`T | None`, `list[str]`, etc.)
- **ChromaDB**: Vector database for embeddings
- **Ollama**: Local LLM and embedding models
- **LangChain**: LLM framework abstractions
- **pytest**: Testing framework
- **mypy**: Static type checking

### Key Libraries
- `pypdf`: PDF text extraction
- `beautifulsoup4`: HTML parsing
- `requests`: HTTP client for APIs
- `pydantic`: Data validation (where used)
- `dataclasses`: Preferred for simple data structures

### Development Tools
- Use `pytest -xvs` for verbose test output
- Use `mypy` for type checking
- Follow existing patterns in the codebase
- Check `get_errors()` after edits

### Tool Chain
Black (Formatting)
    ↓
isort (Import Organisation)
    ↓
pylint (Linting) → --fail-under=8
    ↓
mypy (Type Checking) → --ignore-missing-imports
    ↓
pytest (Testing) 
