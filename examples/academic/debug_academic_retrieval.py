#!/usr/bin/env python
"""Diagnostic script to debug academic document retrieval."""

from pathlib import Path

from scripts.rag.rag_config import RAGConfig
from scripts.utils.db_factory import get_default_vector_path, get_vector_client

config = RAGConfig()
PersistentClient, using_sqlite = get_vector_client(prefer="chroma")
chroma_path = get_default_vector_path(Path(config.rag_data_path), using_sqlite)
client = PersistentClient(path=chroma_path)
col = client.get_collection(config.chunk_collection_name)

print("=" * 80)
print("ACADEMIC DOCUMENT RETRIEVAL DIAGNOSTICS")
print("=" * 80)

# 1. Check total counts
print("\n1. COLLECTION STATISTICS:")
total = col.count()
print(f"   Total chunks: {total}")

# 2. Check academic chunks
print("\n2. ACADEMIC CHUNKS:")
academic_results = col.get(
    where={"source_category": "academic_reference"}, limit=5, include=["metadatas", "documents"]
)
academic_count = len(academic_results["ids"])
print(f"   Academic chunks found: {academic_count}")

if academic_count > 0:
    print("\n   Sample academic chunks:")
    for i, (chunk_id, meta, doc) in enumerate(
        zip(
            academic_results["ids"][:3],
            academic_results["metadatas"][:3],
            academic_results["documents"][:3],
        ),
        1,
    ):
        print(f"\n   [{i}] ID: {chunk_id}")
        print(f"       Title: {meta.get('title', 'N/A')[:60]}")
        print(f"       Authors: {meta.get('authors', 'N/A')[:60]}")
        print(f"       Doc ID: {meta.get('doc_id', 'N/A')[:60]}")
        print(f"       Content preview: {doc[:150]}...")
        print(f"       Embedding model: {meta.get('embedding_model', 'N/A')}")

# 3. Try vector search on academic topics
print("\n3. VECTOR SEARCH TEST (academic topics):")
from scripts.ingest.vectors import EMBEDDING_MODEL_NAME

test_queries = ["leadership", "research methodology", "thesis findings", "academic research"]

for query in test_queries:
    from langchain_ollama import OllamaEmbeddings

    embed_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    query_emb = embed_model.embed_query(query)

    results = col.query(
        query_embeddings=[query_emb],
        n_results=3,
        where={"embedding_model": EMBEDDING_MODEL_NAME},
        include=["metadatas", "distances"],
    )

    if results["ids"] and results["ids"][0]:
        print(f"\n   Query: '{query}'")
        print(f"   Results: {len(results['ids'][0])}")
        for j, (chunk_id, meta, distance) in enumerate(
            zip(results["ids"][0][:2], results["metadatas"][0][:2], results["distances"][0][:2]), 1
        ):
            print(
                f"   [{j}] source_category: {meta.get('source_category', 'N/A')}, distance: {distance:.4f}"
            )
    else:
        print(f"\n   Query: '{query}' → NO RESULTS")

# 4. Try keyword search (BM25)
print("\n4. KEYWORD SEARCH TEST:")
from scripts.rag.retrieve import _keyword_search_fallback

for query in ["leadership", "thesis", "research"]:
    chunks, meta, ids = _keyword_search_fallback(query, col, k=3, filters=None)
    print(f"\n   Query: '{query}'")
    print(f"   Results: {len(chunks)}")
    if chunks:
        for j, m in enumerate(meta[:2], 1):
            print(f"   [{j}] source_category: {m.get('source_category', 'N/A')}")

# 5. Check if any chunks have keywords
print("\n5. KEYWORD PRESENCE CHECK:")
keywords = ["leadership", "thesis", "research", "methodology"]
for keyword in keywords:
    # Get all chunks and count keyword occurrences
    all_chunks = col.get(limit=100, include=["documents", "metadatas"])
    if all_chunks["documents"]:
        count = sum(1 for doc in all_chunks["documents"] if keyword.lower() in doc.lower())
        print(
            f"   '{keyword}' appears in {count}/{len(all_chunks['documents'])} chunks (sampled 100)"
        )

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
if academic_count == 0:
    print("❌ No academic chunks found!")
    print("   → Re-run academic ingestion with proper source_category metadata")
elif academic_count > 0:
    print(f"✅ Found {academic_count} academic chunks")
    print("   → Check if chunk content matches your query topics")
    print("   → Try running: python scripts/rag/query.py 'What research papers discuss [TOPIC]?'")
print("=" * 80)
