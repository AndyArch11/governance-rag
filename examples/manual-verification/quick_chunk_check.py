#!/usr/bin/env python3
"""Quick check of chunk text content."""

from pathlib import Path

import chromadb

client = chromadb.PersistentClient(
    Path("/workspaces/governance-rag/rag_data/chromadb").expanduser()
)
collection = client.get_collection("governance_docs_chunks")

# Get first 5 chunks
results = collection.get(limit=5, include=["documents", "metadatas"])

for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
    print(f"\n{'='*80}")
    print(f"CHUNK {i}:")
    print(f"{'='*80}")
    print(f"Source: {meta.get('source', 'N/A')}")
    print(f"\nText preview (first 500 chars):")
    print(doc[:500])
    print("...")
