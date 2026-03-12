#!/usr/bin/env python
"""Check BM25 index status."""

from pathlib import Path

from scripts.rag.rag_config import RAGConfig
from scripts.search.bm25_retrieval import BM25Retriever

config = RAGConfig()

print("=" * 80)
print("BM25 INDEX STATUS")
print("=" * 80)

retriever = BM25Retriever(rag_data_path=Path(config.rag_data_path))

print(f"\nRAG Data Path: {config.rag_data_path}")
print(f"Total docs in index: {retriever.total_docs}")
print(f"Average doc length: {retriever.avg_doc_length:.2f}")

if retriever.total_docs > 0:
    print("\n✅ BM25 index is available and has documents")

    # Test a search
    print("\nTest search for 'leadership':")
    results = retriever.search("leadership", top_k=5)
    print(f"Results found: {len(results)}")

    if results:
        for i, (doc_id, score) in enumerate(results[:3], 1):
            print(f"  [{i}] doc_id: {doc_id[:60]}... (score: {score:.4f})")
    else:
        print("  ❌ No results returned!")

    # Check if ChromaDB chunk IDs exist in BM25 index
    print("\nChecking if ChromaDB chunk-level IDs are in BM25 index...")
    from scripts.utils.db_factory import get_default_vector_path, get_vector_client

    PersistentClient, using_sqlite = get_vector_client(prefer="chroma")
    chroma_path = get_default_vector_path(Path(config.rag_data_path), using_sqlite)
    client = PersistentClient(path=chroma_path)
    col = client.get_collection(config.chunk_collection_name)

    # Get all ChromaDB chunk IDs
    all_chunks = col.get(limit=None, include=["metadatas"])
    chromadb_ids = set(all_chunks["ids"])

    print(f"Total ChromaDB chunks: {len(chromadb_ids)}")

    # Sample a few for display
    sample_chromadb_ids = list(chromadb_ids)[:3]
    print(f"Sample ChromaDB chunk IDs:")
    for chroma_id in sample_chromadb_ids:
        print(f"  - {chroma_id[:80]}...")

    # Get all BM25 index doc_ids
    import sqlite3

    cache_db_path = Path(config.rag_data_path) / "cache.db"

    if not cache_db_path.exists():
        print(f"\n❌ Cache database not found at {cache_db_path}")
        print("   Run ingestion first to create BM25 index")
    else:
        conn = sqlite3.connect(str(cache_db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT doc_id FROM bm25_index")
        bm25_ids = set(row[0] for row in cursor.fetchall())
        conn.close()

        print(f"\nTotal BM25 index doc_ids: {len(bm25_ids)}")

        # Sample a few for display
        sample_bm25_ids = list(bm25_ids)[:3]
        print(f"Sample BM25 index doc_ids:")
        for bm25_id in sample_bm25_ids:
            print(f"  - {bm25_id[:80]}...")

        # Check for complete overlap
        overlap = chromadb_ids & bm25_ids
        overlap_pct = (len(overlap) / len(chromadb_ids) * 100) if chromadb_ids else 0
        missing = len(chromadb_ids) - len(overlap)
        print(f"\nChunk-level ID overlap: {len(overlap)}/{len(chromadb_ids)} ({overlap_pct:.1f}%)")

        if missing == 0:
            print("\n✅ Perfect alignment: all ChromaDB chunks are indexed in BM25")
        elif missing <= 10:
            print(f"\n✅ Excellent alignment: {missing} chunk(s) missing (likely edge cases)")
            if missing > 0:
                # Only show details if there are actual missing chunks
                print(
                    f"   This is negligible ({100-overlap_pct:.2f}% of {len(chromadb_ids)} chunks)"
                )
        else:
            print(
                f"\n⚠️  {missing} ChromaDB chunks are missing from BM25 index ({100-overlap_pct:.1f}%)"
            )
            print("   This may indicate:")
            print("   - Chunks from citations are not being indexed in BM25")
            print("   - Main document chunks and citation chunks are treated differently")
            print("\n✅ Next steps:")
            print("   1. Check if citations are being indexed: grep for 'citation' in ingest logs")
            print("   2. Verify BM25 indexing logic handles all chunk types consistently")
else:
    print("\n❌ BM25 index is empty!")
    print("   → Build index: python3 scripts/rag/bm25_indexer.py")

retriever.close()
print("=" * 80)
