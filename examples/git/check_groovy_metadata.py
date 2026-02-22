#!/usr/bin/env python
"""Check specific Groovy file metadata in ChromaDB."""

import json
from pathlib import Path

from chromadb import PersistentClient

# Connect to ChromaDB
rag_data_path = Path.home() / "rag-project" / "rag_data"
client = PersistentClient(path=str(rag_data_path))

# Get the document collection
doc_collection = client.get_or_create_collection("governance_docs_documents")

# Get the specific Groovy document from the log
doc_id = "Name of the Groovy file to check.groovy"

print("=" * 80)
print(f"Checking ChromaDB for: {doc_id}")
print("=" * 80)

try:
    results = doc_collection.get(ids=[doc_id], include=["metadatas"])

    if not results["ids"]:
        print("\n❌ Document not found in ChromaDB!")
    else:
        print("\n✅ Document found!")
        metadata = results["metadatas"][0]

        print("\n--- All metadata fields ---")
        for key, value in sorted(metadata.items()):
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")

        print("\n--- Code-specific fields ---")
        code_fields = [
            "language",
            "service_name",
            "service",
            "service_type",
            "dependencies",
            "internal_calls",
            "endpoints",
            "db",
            "queue",
            "exports",
        ]

        found = []
        missing = []
        for field in code_fields:
            if field in metadata:
                found.append(field)
                print(f"  ✅ {field}: {metadata[field]}")
            else:
                missing.append(field)
                print(f"  ❌ {field}: MISSING")

        print(f"\n  Found: {len(found)}/{len(code_fields)}")
        print(f"  Missing: {missing}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
