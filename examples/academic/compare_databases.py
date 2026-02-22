#!/usr/bin/env python3
"""
Compare ChromaDB, Consistency Graph, and Citation Graph to understand the relationship.
"""
from pathlib import Path
import sqlite3
from collections import Counter

import chromadb

print("=" * 80)
print("DATABASE COMPARISON ANALYSIS")
print("=" * 80)

# 1. ChromaDB Analysis
print("\n[1] ChromaDB (governance_docs_chunks)")
print("-" * 80)
chroma_path = Path("/workspaces/governance-rag/rag_data/chromadb").expanduser()
chroma_client = chromadb.PersistentClient(path=str(chroma_path))

doc_ids = set()
source_categories = Counter()
doc_types = Counter()
academic_docs = []
reference_docs = []
total_chunks = 0

try:
    collection = chroma_client.get_collection(name="governance_docs_chunks")

    # Get all chunks
    all_chunks = collection.get(include=["metadatas"])
    total_chunks = len(all_chunks["ids"])

    # Extract unique doc_ids and source_categories
    for meta in all_chunks["metadatas"]:
        if meta.get("doc_id"):
            doc_ids.add(meta["doc_id"])
        if meta.get("source_category"):
            source_categories[meta["source_category"]] += 1
        if meta.get("doc_type"):
            doc_types[meta["doc_type"]] += 1

    print(f"Total chunks: {total_chunks}")
    print(f"Unique documents: {len(doc_ids)}")
    print("\nTop source categories:")
    for cat, count in source_categories.most_common(10):
        print(f"  {cat}: {count} chunks")

    print("\nAcademic-related documents:")
    academic_docs = [
        d
        for d in doc_ids
        if "academic" in d.lower() or "2025" in d
    ]
    print(f"  Found {len(academic_docs)} academic doc_ids:")
    for doc in academic_docs[:5]:
        print(f"    - {doc}")

    # Check for reference nodes
    reference_docs = [d for d in doc_ids if "doi:" in d or "ref:" in d]
    print(f"\n  Citation/Reference doc_ids: {len(reference_docs)}")
    if reference_docs:
        print(f"    Examples: {reference_docs[:3]}")
except chromadb.errors.NotFoundError:
    print("ChromaDB collection 'governance_docs_chunks' not found.")
    print("Run ingestion or update the collection name before comparing databases.")

# 2. Consistency Graph Analysis
print("[2] Consistency Graph (consistency_graph.sqlite)")
print("-" * 80)
consistency_db = Path(
    "/workspaces/governance-rag/rag_data/consistency_graphs/consistency_graph.sqlite"
).expanduser()

if not consistency_db.exists():
    print(f"Consistency graph database not found: {consistency_db}")
    total_nodes = 0
    total_edges = 0
    cursor = None
else:
    conn = sqlite3.connect(str(consistency_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()


def table_exists(db_cursor: sqlite3.Cursor, table_name: str) -> bool:
    db_cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return db_cursor.fetchone() is not None

if cursor and table_exists(cursor, "nodes") and table_exists(cursor, "edges"):
    # Get node count and types
    cursor.execute("SELECT COUNT(*) as count FROM nodes")
    total_nodes = cursor.fetchone()["count"]

    cursor.execute("SELECT COUNT(*) as count FROM edges")
    total_edges = cursor.fetchone()["count"]

    print(f"Total nodes: {total_nodes}")
    print(f"Total edges: {total_edges}")

    # Check for academic nodes
    cursor.execute(
        "SELECT node_id FROM nodes WHERE node_id LIKE '%academic%' "
        "OR node_id LIKE '%2025%' LIMIT 10"
    )
    academic_graph_nodes = [row["node_id"] for row in cursor.fetchall()]
    print(f"\nAcademic nodes in consistency graph: {len(academic_graph_nodes)}")
    if academic_graph_nodes:
        for node in academic_graph_nodes[:3]:
            print(f"  - {node}")

    # Check for reference/citation nodes
    cursor.execute(
        "SELECT node_id FROM nodes WHERE node_id LIKE 'doi:%' "
        "OR node_id LIKE 'ref:%' LIMIT 5"
    )
    ref_graph_nodes = [row["node_id"] for row in cursor.fetchall()]
    print(f"\nReference nodes in consistency graph: {len(ref_graph_nodes)}")
    if ref_graph_nodes:
        print(f"  Examples: {ref_graph_nodes}")
elif cursor:
    print("Consistency graph schema not found (missing nodes/edges tables).")
    total_nodes = 0
    total_edges = 0

if cursor:
    conn.close()

# 3. Citation Graph Analysis
print("[3] Citation Graph (academic_citation_graph.db)")
print("-" * 80)
conn2 = sqlite3.connect(Path("/workspaces/governance-rag/rag_data/academic_citation_graph.db").expanduser())
conn2.row_factory = sqlite3.Row
cursor2 = conn2.cursor()

citation_node_counts = {}
citation_total_nodes = 0
citation_doc_count = 0
citation_ref_count = 0

if table_exists(cursor2, "nodes") and table_exists(cursor2, "edges"):
    cursor2.execute("SELECT node_type, COUNT(*) as count FROM nodes GROUP BY node_type")
    print("Node types:")
    for row in cursor2.fetchall():
        citation_node_counts[row["node_type"]] = row["count"]
        citation_total_nodes += row["count"]
        print(f"  {row['node_type']}: {row['count']}")

    citation_doc_count = citation_node_counts.get("document", 0)
    citation_ref_count = citation_node_counts.get("reference", 0)

    cursor2.execute("SELECT COUNT(*) as count FROM edges")
    citation_edges = cursor2.fetchone()["count"]
    print(f"\nTotal edges: {citation_edges}")

    # Get document node
    cursor2.execute("SELECT node_id, title FROM nodes WHERE node_type = 'document'")
    doc_node = cursor2.fetchone()
    if doc_node:
        print(f"\nDocument node: {doc_node['node_id']}")
        print(f"  Title: {doc_node['title'][:80]}...")
    else:
        print("\nDocument node: not found")
else:
    print("Citation graph schema not found (missing nodes/edges tables).")
    citation_edges = 0

conn2.close()

# 4. Cross-reference Analysis
print("\n[4] CROSS-REFERENCE ANALYSIS")
print("-" * 80)

print(f"\nChromaDB unique docs: {len(doc_ids)}")
print(f"Consistency Graph nodes: {total_nodes}")
print(
    f"Citation Graph nodes: {citation_doc_count} document + "
    f"{citation_ref_count} references = {citation_total_nodes}"
)

print(f"\nDISCREPANCY ANALYSIS:")

print(f"\n  Actual ChromaDB docs: {len(doc_ids)}")
print(f"  Actual Consistency nodes: {total_nodes}")


print(f"\nN.B. Citation references are not in ChromaDB.")
print(f"  → Nor are they in Consistency Graph.")
print(f"  → They only exist in academic_citation_graph.db")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
if len(reference_docs) == 0:
    print(
        f"The citation references ({citation_ref_count} refs) are stored in "
        "academic_citation_graph.db."
    )
    print(f"  • Consistency Graph (from ChromaDB) has {total_nodes} docs")
    print(
        f"  • Citation Graph has {citation_total_nodes} nodes "
        f"({citation_doc_count} thesis + {citation_ref_count} refs)"
    )

    print("\nThe databases serve different purposes:")
    print("  • ChromaDB: Full-text chunks for RAG queries")
    print("  • Consistency Graph: Document conflict analysis")
    print("  • Citation Graph: Academic reference tracking")
