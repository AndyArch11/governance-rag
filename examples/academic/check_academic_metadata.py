#!/usr/bin/env python3
"""Check metadata fields on academic chunks to diagnose persona filtering."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import chromadb

def main():
    client = chromadb.PersistentClient(path="rag_data/chromadb")
    collection = client.get_collection("governance_docs_chunks")
    
    # Query for academic chunks
    results = collection.get(
        where={"source_category": "academic_reference"},
        limit=5,
        include=["metadatas", "documents"]
    )
    
    print(f"\n{'='*80}")
    print("ACADEMIC CHUNK METADATA FIELDS")
    print(f"{'='*80}\n")
    
    if not results["ids"]:
        print("❌ No academic chunks found!")
        return
    
    print(f"Found {len(results['ids'])} academic chunks\n")
    
    for i, (chunk_id, meta, doc) in enumerate(zip(results["ids"], results["metadatas"], results["documents"]), 1):
        print(f"[{i}] Chunk ID: {chunk_id[:60]}...")
        print(f"    Document preview: {doc[:100]}...")
        print(f"    Metadata fields present:")
        
        # Key fields for persona filtering
        key_fields = ["reference_type", "link_status", "quality_score", 
                     "citation_count", "domain_relevance_score", "source_category"]
        
        for field in key_fields:
            if field in meta:
                print(f"      ✓ {field}: {meta[field]}")
            else:
                print(f"      ✗ {field}: MISSING")
        
        print(f"    All fields: {list(meta.keys())}\n")
    
    # Check ASSESSOR requirements
    print(f"{'='*80}")
    print("ASSESSOR PERSONA REQUIREMENTS")
    print(f"{'='*80}\n")
    print("From ASSESSOR_CONFIG:")
    print("  prefer_reference_types: ('academic', 'report')")
    print("  min_quality_score: 0.7")
    print("  min_citation_count: 5")
    print("  require_verifiable: True")
    print("  include_stale_links: False")
    print()
    print("Filter logic:")
    print("  1. If reference_type exists AND NOT in ('academic', 'report') → FILTERED OUT")
    print("  2. If require_verifiable=True AND link_status != 'available' → FILTERED OUT")
    print("  3. If quality_score < 0.7 → FILTERED OUT")
    print("  4. If citation_count < 5 → FILTERED OUT")
    print()
    
    # Analyse which filters would trigger
    print(f"{'='*80}")
    print("FILTER ANALYSIS")
    print(f"{'='*80}\n")
    
    for i, meta in enumerate(results["metadatas"], 1):
        print(f"[{i}] Chunk analysis:")
        
        filters_triggered = []
        
        # Check reference_type filter
        reference_type = (meta.get("reference_type") or "").lower()
        if reference_type and reference_type not in ("academic", "report"):
            filters_triggered.append(f"reference_type '{reference_type}' not in allowed types")
        elif not reference_type:
            filters_triggered.append("reference_type is MISSING/empty")
        
        # Check link_status filter
        link_status = (meta.get("link_status") or "available").lower()
        if link_status != "available":
            filters_triggered.append(f"link_status '{link_status}' (require_verifiable=True needs 'available')")
        
        # Check quality_score filter
        quality_score = float(meta.get("quality_score") or 0.0)
        if quality_score < 0.7:
            filters_triggered.append(f"quality_score {quality_score} < 0.7")
        
        # Check citation_count filter
        citation_count = int(meta.get("citation_count") or 0)
        if citation_count < 5:
            filters_triggered.append(f"citation_count {citation_count} < 5")
        
        if filters_triggered:
            print(f"    ❌ WOULD BE FILTERED OUT:")
            for reason in filters_triggered:
                print(f"       - {reason}")
        else:
            print(f"    ✓ WOULD PASS ALL FILTERS")
        print()

if __name__ == "__main__":
    main()
