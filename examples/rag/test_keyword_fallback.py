#!/usr/bin/env python
"""Test keyword search fallback directly."""

from pathlib import Path
from scripts.rag.rag_config import RAGConfig
from scripts.utils.db_factory import get_default_vector_path, get_vector_client

config = RAGConfig()
PersistentClient, using_sqlite = get_vector_client(prefer="chroma")
chroma_path = get_default_vector_path(Path(config.rag_data_path), using_sqlite)
client = PersistentClient(path=chroma_path)
col = client.get_collection(config.chunk_collection_name)

print("Testing keyword fallback search directly...")
print("=" * 80)

# Import the fallback function
from scripts.rag.retrieve import _keyword_search_fallback

# Test with "leadership"
print("\nTest 1: Query='leadership', k=5, no filters")
chunks, meta, ids = _keyword_search_fallback("leadership", col, k=5, filters=None)
print(f"Results: {len(chunks)} chunks")
if chunks:
    for i, (chunk, m) in enumerate(zip(chunks[:2], meta[:2]), 1):
        print(f"\n[{i}] source_category: {m.get('source_category', 'N/A')}")
        print(f"    tf_score: {m.get('tf_score', 'N/A')}")
        print(f"    Preview: {chunk[:100]}...")
else:
    print("❌ No results!")

# Test with filters
print("\n" + "=" * 80)
print("\nTest 2: Query='leadership', k=5, filter={}")
chunks, meta, ids = _keyword_search_fallback("leadership", col, k=5, filters={})
print(f"Results: {len(chunks)} chunks")

# Check if chunks exist at all
print("\n" + "=" * 80)
print("\nDirect ChromaDB check:")
all_chunks = col.get(limit=10, include=["documents"])
print(f"Total chunks fetched: {len(all_chunks['ids'])}")
if all_chunks["documents"]:
    doc_sample = all_chunks['documents'][0]
    print(f"Sample doc preview: {doc_sample[:100]}...")
    print(f"Contains 'leadership': {'leadership' in doc_sample.lower()}")

print("=" * 80)
