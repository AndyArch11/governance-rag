# RAG Query Assistant in Dashboard

## Overview

The Dashboard includes a fully integrated **RAG (Retrieval-Augmented Generation) Query Assistant** that allows users to ask natural language questions about governance documents while viewing the consistency graph.

The assistant is available in the **Chat** tab of the dashboard and provides:
- Natural language question answering
- Semantic document retrieval
- Multi-turn conversation history
- Performance metrics and source attribution
- Integration with ChromaDB and Ollama LLM

## Accessing the Assistant

1. **Start the dashboard:**
   ```bash
   cd /workspaces/governance-rag
   source .venv/bin/activate
   python3 -m scripts.ui.dashboard
   ```

2. **Navigate to the assistant view:**
   - Open http://localhost:8050 in your browser
   - Use the **Conversations** and query controls in the dashboard

## Features

### 1. Ask Questions
- Type questions in the **"Ask a Question"** input box
- Questions are processed against indexed ingested documents
- System retrieves relevant content and generates answers using Ollama

**Example questions:**
```
- What are the key governance policies?
- How is data security handled?
- What are the compliance requirements?
- Summarise the risk management framework
- Compare versions of the security policy
```

### 2. Real-Time Processing
The assistant shows processing status with:
- **Retrieval phase:** Document chunks being searched
- **Generation phase:** LLM generating answer
- Completion status with timing breakdown

### 3. Conversation History
- All questions and answers are maintained in conversation history
- Newest messages appear first (standard chat UX)
- Persistent within the session
- Each message shows the question and detailed response

### 4. Performance Metrics
For each response, the dashboard displays:
- **⏱️ Retrieval Time:** Time to find relevant documents
- **💬 Generation Time:** Time for LLM to generate answer
- **📄 Chunks Retrieved:** Number of document chunks used

### 5. Source Attribution
- Click **"📚 Sources"** expander for each answer
- Shows which documents were used
- Includes:
  - Document name/ID
  - Version information (document version in the collection)
  - Metadata snippet from the chunk

### 6. Conversation Management
- **Clear History button:** Removes all messages and starts fresh
- Conversation persists across tab switches (within same session)

## Architecture

### Components

```
User Query
    ↓
[RAG Query Assistant]
    ├─ Embedding Model (mxbai-embed-large via Ollama)
    ├─ Document Retrieval (ChromaDB)
    │  └─ Collection: governance_docs_chunks
    ├─ Answer Generation (Ollama)
    │  └─ Model: mistral (configurable)
    └─ Response Formatting
        ├─ Answer text
        ├─ Metadata (timing, sources)
        └─ Source references
```

### Data Flow

1. **Query Embedding:** User query is embedded using mxbai-embed-large
2. **Semantic Retrieval:** Top-k similar chunks retrieved from ChromaDB (default k=5)
3. **Context Preparation:** Retrieved chunks formatted with metadata
4. **Answer Generation:** Ollama generates contextual answer
5. **Response Display:** Answer shown with metadata and source attribution

### Configuration

**Dashboard (scripts/ui/dashboard.py):**
- RAG Query Assistant implementation
- Chat input at line
- RAG component loading
- Response generation

**RAG Module (scripts/rag/):**
- `query.py`: Document retrieval logic
- `generate.py`: Answer generation
- `rag_config.py`: RAG configuration

**Ingestion Module (scripts/ingest/):**
- Creates embeddings and stores in ChromaDB
- Collections: `governance_docs_chunks` (for querying), `governance_docs_documents` (for display)

## Usage Examples

### Example 1: Policy Summary
```
Q: What is the main governance policy?
A: [System retrieves and summarises governance policy]
   - Shows 5 relevant chunks
   - Time breakdown (retrieval: 0.45s, generation: 1.2s)
   - Sources from policy documents
```

### Example 2: Cross-Document Search
```
Q: How do different versions handle security?
A: [System compares security approaches across versions]
   - Multiple document versions compared
   - Key differences highlighted
   - Sources shown with version numbers (v1, v2, etc.)
```

### Example 3: Compliance Verification
```
Q: Are we compliant with ISO 27001?
A: [System checks compliance requirements]
   - Relevant sections retrieved
   - Compliance status assessed
   - Source documents referenced
```

## Performance Considerations

### Retrieval Performance
- Retrieval time depends on ChromaDB collection size
- Default k=5 (5 chunks retrieved per query)
- Typical retrieval: 0.1-0.5 seconds for 50+ documents

### Generation Performance
- Generation time depends on Ollama model (mistral)
- Ollama must be running: `ollama serve` or system service
- Typical generation: 1-3 seconds for short answers
- Longer answers may take 5-10 seconds

### Optimisation Tips
1. **Reduce k value** in dashboard query controls:
   ```python
   response = rag_answer(query=query_text.strip(), collection=collection, k=3)
   ```

2. **Use faster model** if available:
   - Check `scripts/rag/rag_config.py` for LLM configuration
   - Use `neural-chat` for faster responses (less quality)
   - Use `mistral` for balance
   - Use `neural-chat:7b-q8_0` for highest quality

3. **Pre-warm embeddings:**
   - First query is slower due to model loading
   - Subsequent queries are faster

## Troubleshooting

### Issue: Reranker load report shows `UNEXPECTED` for `roberta.embeddings.position_ids`
**Status:** Usually not an issue.

When using `BAAI/bge-reranker-base` (or related reranker checkpoints), you may see a Hugging Face load report entry like:

```text
roberta.embeddings.position_ids | UNEXPECTED
```

This is generally a checkpoint/buffer compatibility detail and can be ignored **if model loading succeeds and reranking still returns results**.

Treat it as a real issue only if you also see one of these symptoms:
1. Model load fails with an exception
2. Multiple critical parameter mismatches (not just `position_ids`)
3. Reranker returns errors, NaN scores, or clearly degraded ranking quality

If needed, test reranker health directly by running a query and confirming logs show successful hybrid retrieval + reranking output.

### Issue: "RAG components failed to load"
**Solution:** Check these components are running:
1. **ChromaDB:** Documents ingested and available
   ```bash
   python3 -c "from chromadb import PersistentClient; c = PersistentClient(path='/workspaces/governance-rag/rag_data/chromadb'); print(c.list_collections())"
   ```

2. **Ollama:** Running with embedding and LLM models
   ```bash
   ollama serve  # In another terminal
   ollama pull mxbai-embed-large
   ollama pull mistral
   ```

3. **Python paths:** RAG module imports working
   ```bash
   python3 -c "from scripts.rag.generate import answer; print('OK')"
   ```

### Issue: Query takes too long
**Solutions:**
1. Check Ollama is responsive: `curl http://localhost:11434/api/tags`
2. Reduce k value (fewer chunks to retrieve)
3. Monitor system resources: `top` or `htop`
4. Check ChromaDB collection size: `ls -lh /workspaces/governance-rag/rag_data/`

### Issue: "No answer generated"
**Possible causes:**
1. ChromaDB collection is empty (no documents ingested)
   - Run ingestion
2. Ollama model not loaded
   - Restart Ollama: `pkill ollama; ollama serve`
3. Query too specific (no matching documents)
   - Try broader question
   - Use related keywords

### Issue: Sources not showing
**Solution:** Check chunk metadata is preserved during ingestion
- Review ingestion configuration in "Ingestion" tab
- Check `governance_docs_chunks` collection has metadata

## Integration with Graph

The RAG Query Assistant complements the consistency graph by:

1. **Context for Graph Nodes:**
   - Ask about specific document versions shown in graph
   - Understand relationships between documents

2. **Conflict Resolution:**
   - Query for details about conflicts shown in graph edges
   - Get explanations for high-conflict nodes

3. **Version Comparison:**
   - Ask about differences between document versions (v1, v2, etc.)
   - Understand semantic drift from user perspective

4. **Policy Enforcement:**
   - Query policy details shown in graph
   - Get compliance-related information

## Advanced Configuration

### Custom Retrieval Strategy
Edit `scripts/rag/query.py` to:
- Change similarity threshold
- Adjust retrieval strategy
- Add filters to query

### Custom Answer Generation
Edit `scripts/rag/generate.py` to:
- Change prompt engineering
- Adjust answer formatting
- Add post-processing

### Collection Configuration
Edit `scripts/ingest/ingest_config.py` to:
- Change collection names
- Adjust chunking strategy
- Modify metadata extraction

## API Reference

### RAG Answer Function Signature
```python
def answer(query: str, collection: Collection, k: int = 5) -> Dict[str, Any]:
    """
    Generate answer for query using RAG.
    
    Args:
        query: Natural language question
        collection: ChromaDB collection with document chunks
        k: Number of chunks to retrieve (default 5)
    
    Returns:
        Dict with:
        - answer: Generated answer text
        - retrieval_count: Number of chunks retrieved
        - generation_time: Time to generate answer
        - total_time: Total processing time
        - sources: List of source documents
        - model: LLM model used
    """
```

### Response Structure
```python
{
    "answer": "Generated answer text...",
    "retrieval_count": 5,
    "retrieval_time": 0.45,
    "generation_time": 1.23,
    "total_time": 1.68,
    "sources": [
        {
            "source": "governance_policy_v1",
            "version": "1",
            "metadata": "Policy excerpt..."
        },
        # ...more sources
    ],
    "model": "mistral"
}
```

## Best Practices

1. **Ask Clear Questions:**
   - ✅ "What are the data retention policies?"
   - ❌ "Tell me about stuff"

2. **Use Specific Keywords:**
   - ✅ "What are the GDPR compliance requirements?"
   - ❌ "What are the rules?"

3. **One Question at a Time:**
   - ✅ "How is encryption handled?"
   - ❌ "How is encryption and access control handled and what about..." (break into separate queries)

4. **Check Sources:**
   - Always review sources for accuracy
   - Verify answers against original documents
   - Report hallucinations

5. **Monitor Performance:**
   - Watch retrieval and generation times
   - Excessive times indicate configuration issues
   - Report performance problems

## Session State

The dashboard maintains Dash state for query workflows, including client-side `dcc.Store`
entries (for example `conversation-store`) and persisted conversation records in
`rag_data/conversations.db`.

Session persists until:
- Browser window closed
- Dashboard server restarted
- Session data is cleared or replaced

## Limitations

1. **Knowledge Cutoff:** Based on ingested documents only
2. **Context Window:** Retrieval limited to a maximum of k=15 chunks per query
3. **Response Length:** Depends on LLM model and timeout
4. **Real-Time Updates:** Requires re-ingestion for document updates
5. **Hallucination:** LLM may generate plausible-sounding incorrect answers

## TODO: Future Enhancements

Potential improvements:
- [ ] Multi-turn context awareness
- [ ] Multi-language support
- [ ] Answer caching
- [ ] Custom system prompts per query
- [ ] Export conversation as PDF
- [ ] Integration with external knowledge bases

## See Also

- [LLM Cache Documentation](LLM_CACHE.md)
- [Query Module Reference](../scripts/rag/query.py)
- [Retrieve Module Reference](../scripts/rag/retrieve.py)
- [Generate Module Reference](../scripts/rag/generate.py)

---

**Status:** Production Ready  
**Last Updated:** January 2026  
**Maintained By:** RAG System
