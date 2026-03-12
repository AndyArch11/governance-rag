# RAG Query Assistant - Quick Start

## Access the Chat Interface

**Dashboard URL:** http://localhost:8050

**Navigation:** Open the **Conversations** view and use the query controls

## Basic Usage

### 1. Select Query Persona (Optional)

For academic queries, choose a persona to filter results:
- **None** - Default retrieval (all sources)
- **Supervisor** - Foundational understanding (academic/report/preprint, quality ≥ 0.6)
- **Assessor** - Verification focus (academic/report only, requires verifiable sources)
- **Researcher** - Novelty discovery (preprints, recent publications, quality ≥ 0.4)

### 2. Type Your Question
```
Input box: "What would you like to know about governance documents?"
```

Examples:
- "What are the key compliance requirements?"
- "How is data security handled?"
- "Compare the access control policies"
- "What is the risk management framework?"

**Academic Examples (with persona):**
- **Supervisor:** "What are the foundational theories in risk management?"
- **Assessor:** "What evidence supports this compliance framework?"
- **Researcher:** "What are the latest approaches to data governance?"

### 3. Review the Answer
The system returns:
- **Answer Text**: The generated response
- **Retrieval Time**: How long to find documents
- **Generation Time**: How long to create answer
- **Chunks Retrieved**: Number of document sections used
- **Persona Applied**: If academic filtering was used (when persona selected)

### 4. Check Sources
Click **"📚 Sources"** to see which documents were referenced

**Persona Filtering:** When a persona is selected, sources are filtered by:
- Reference type (academic, preprint, report, etc.)
- Quality score and citation count thresholds
- Link verification status (for assessor persona)
- Recency bias (for researcher persona)

## Requirements

✓ **Ollama Running:**
```bash
ollama serve
```

✓ **Documents Indexed:**
- Run the ingestion pipeline 
- ChromaDB collection: `governance_docs_chunks`

✓ **Dashboard Running:**
```bash
cd /workspaces/governance-rag
source .venv/bin/activate
python3 -m scripts.ui.dashboard
```

## Performance Baseline

| Operation | Typical Time |
|-----------|-------------|
| Retrieval | 0.1 - 0.5s |
| Generation | 1.0 - 3.0s |
| Total | 1.1 - 3.5s |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "RAG components failed to load" | Check Ollama is running: `ollama serve` |
| Query takes >10s | Reduce k value in the dashboard query controls (e.g., from 5 to 3) |
| No sources shown | Re-run ingestion to create chunks |
| Empty answer | Try different phrasing or broader question |

## Tips for Better Results

✅ **DO:**
- Ask clear, specific questions
- Use relevant keywords
- Ask one question at a time
- Check sources for accuracy

❌ **DON'T:**
- Ask questions outside document scope
- Assume answers without checking sources
- Ask overly complex multi-part questions
- Trust hallucinated information

## For More Information

See [RAG_QUERY_ASSISTANT.md](RAG_QUERY_ASSISTANT.md) for:
- Detailed architecture
- Advanced configuration
- API reference
- Performance tuning
- Troubleshooting guide

---

**Dashboard:** http://localhost:8050  
**Primary View:** Conversations + query controls  
