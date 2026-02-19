#!/usr/bin/env python
"""Diagnostic: Check what metadata is actually stored in ChromaDB for code documents."""

import argparse
from pathlib import Path

from chromadb import PersistentClient

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ChromaDB metadata for code documents (Bitbucket/GitHub)."
    )
    parser.add_argument(
        "--provider",
        choices=["all", "bitbucket", "github"],
        default="all",
        help="Filter by git provider (default: all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max documents to show (default: 5).",
    )
    return parser.parse_args()


def _get_results(doc_collection, where: dict, limit: int) -> dict:
    return doc_collection.get(
        where=where,
        limit=limit,
        include=["metadatas", "documents"],
    )


args = _parse_args()

# Connect to ChromaDB
rag_data_path = Path.home() / "rag-project" / "rag_data"
client = PersistentClient(path=str(rag_data_path))

# Get the document collection
doc_collection = client.get_or_create_collection("governance_docs_documents")

# Primary filter: current schema uses source_category="code" with git_provider
where_filter = {"source_category": "code"}
if args.provider != "all":
    where_filter["git_provider"] = args.provider

results = _get_results(doc_collection, where_filter, args.limit)

# Legacy fallback: earlier ingests sometimes used provider-specific source_category
if not results.get("ids"):
    legacy_categories = []
    if args.provider in ("all", "bitbucket"):
        legacy_categories.append("bitbucket_code")
    if args.provider in ("all", "github"):
        legacy_categories.append("github_code")

    for legacy_category in legacy_categories:
        legacy_filter = {"source_category": legacy_category}
        results = _get_results(doc_collection, legacy_filter, args.limit)
        if results.get("ids"):
            break

print("=" * 80)
print("ChromaDB Diagnostic: Code Document Metadata")
print(f"Provider filter: {args.provider}")
print("=" * 80)

if not results.get("ids"):
    print("\n❌ No code documents found in ChromaDB with the current filter")
else:
    print(f"\n✅ Found {len(results['ids'])} code documents")
    
    for i, (doc_id, metadata) in enumerate(zip(results["ids"], results["metadatas"]), 1):
        print(f"\n--- Document {i} ---")
        print(f"ID: {doc_id}")
        print("\nMetadata fields present:")
        
        if not metadata:
            print("  ❌ NO METADATA")
            continue
            
        # Check for code-specific fields
        code_fields = ["language", "service_name", "service", "service_type", 
                       "dependencies", "internal_calls", "endpoints", 
                       "db", "queue", "exports"]
        
        found_count = 0
        for field in code_fields:
            if field in metadata:
                value = metadata[field]
                # Truncate long values
                if isinstance(value, str) and len(value) > 100:
                    display_value = value[:100] + "..."
                else:
                    display_value = value
                print(f"  ✅ {field}: {display_value}")
                found_count += 1
            else:
                print(f"  ❌ {field}: MISSING")
        
        print(f"\n  Total code fields found: {found_count}/{len(code_fields)}")
        
        print("\n  All available fields:")
        for key, value in sorted(metadata.items()):
            if key not in ["text", "document"]:  # Skip large fields
                if isinstance(value, str) and len(value) > 50:
                    print(f"    {key}: {value[:50]}...")
                else:
                    print(f"    {key}: {value}")

print("\n" + "="*80)
