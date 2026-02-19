# Project Roadmap: Governance Graph Visualisation

```
General
├── Auto-tuning & Adaptive Algorithms 📋- Partially Implemented
├── Future Features
│   ├── 1. Multi-Model Support (with specialised LLMs for different query types)
│   ├── 2. Semantic Reranking (cross-encoder post-retrieval reranking)
│   ├── 3. Multi-tenant (different teams, Business Units), SSO auth, RBAC, limit query types by roles
│   ├── 4. Query Personalisation (user preference tracking)
│   ├── 5. Federated Search (multi-repository querying)
│   ├── 6. Zero Trust - encryption, DB auth, module auth
│   ├── 7. Support for different source targets for ingestion such as GitHub, Confluence, etc
│   │   ├── Semantic parsing of source code comments and md files
│   │   └── Source code dependency graphs
│   ├── 8. Parse multiple document types - PDF, DOCX, ODT, MD, etc
│   ├── 9. Fix logging, too many log entries are made without context
│   ├── 10. Safeguards against top 10 AI OWASP - https://genai.owasp.org/llm-top-10/
│   ├── 11. More robust front end than the plotly dashboard. Multi-tenancy may require this as a prereq
│   └── 12. MCP Server support
├── Enterprise Scalability
├── UX Optimisation
└── Comprehensive Testing (currently at ~50-60% coverage)

Analytics & Intelligence:
├── Graph analysis metrics
├── Anomaly detection
├── Agent based searches to complement the RAG embeddings
└── PostgreSQL with pgvector extension for production-scale embeddings
    ├── Migration from ChromaDB/SQLite to pgvector
    ├── Support for millions of document embeddings
    ├── ACID compliance and concurrent access
    ├── Native vector similarity search (HNSW, IVFFlat indexes)
    └── Integration with existing PostgreSQL infrastructure

Governance Agents:
├── Ingest grounding data
│   └──Due to context window size limitations, multiple agents/grounding data DBs?
├── Assess documentation and code against grounding data, generate compliance reports
├── Create seeded documents pre-filled from grounding data (template hybrids)
└── Generate relationship diagrams / STRIDE data flow diagrams from discovery - mermaid?


## 📈 Potential Roadmap Items under consideration

- Cluster drift over time
- Semantic caching
- Document lineage graph
- Governance rule extraction
- Automated remediation suggestions
- Multi‑tenant dashboard mode, supporting multiple teams / business units with SSO and RBAC authentication
- Zero trust - encrypted connections, authentication to DB, API auth
- Embedding‑only topic clustering
- Cluster‑level risk scoring
- MCP server support
- Document source category filtering
- Additional document types to parse, not only HTML
- URL parsing 
- Image parsing - design diagrams, etc
- Additional ingestion sources, currently just Git providers and local disk
- Add distributed tracing for retry attempts
- Machine learning-based optimal retry strategy
- Add timeout configuration to all retry decorators
- Circuit breaker pattern for failing services
- Instrumentation - OpenTelemetry / Prometheus and maybe an optional Jaeger backend?

## Quality Roadmap

### Short-term
- 📋 Add pre-commit hooks for automatic checks

### Medium-term
- [ ] Add type hints to RAG module parameters
- [ ] Fix default parameter type mismatches (preprocess.py)
- [ ] Reduce mypy override scope
- [ ] Consider py.typed marker for public API
- [ ] Fix open DB connection warnings in test runs

### Long-term
- [ ] Gradual increase in mypy strictness
- [ ] Increase test coverage to 70%+
- [ ] Document typing patterns
- [ ] Monitor type coverage metrics