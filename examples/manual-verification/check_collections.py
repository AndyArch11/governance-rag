#!/usr/bin/env python3
"""Check what collections exist in ChromaDB."""

from pathlib import Path
import chromadb

rag_data_dir = Path(__file__).parent.parent / "rag_data"
chroma_db_path = rag_data_dir / "chromadb"

print(f"Checking ChromaDB at: {chroma_db_path}")
print(f"Path exists: {chroma_db_path.exists()}")

if chroma_db_path.exists():
    client = chromadb.PersistentClient(path=str(chroma_db_path))
    collections = client.list_collections()
    print(f"\nCollections found: {len(collections)}")
    for coll in collections:
        print(f"  - {coll.name}")
