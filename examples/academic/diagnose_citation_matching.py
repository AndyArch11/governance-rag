#!/usr/bin/env python3
"""
Diagnose citation graph vs ChromaDB doc_id matching.
"""
import sqlite3
import chromadb

# Check ChromaDB doc_ids
print("=== ChromaDB Document IDs ===")
client = chromadb.PersistentClient(path="/workspaces/governance-rag/rag_data/chromadb")
collection = client.get_collection(name="governance_docs_chunks")

# Get sample metadata
results = collection.get(limit=10, include=['metadatas'])
doc_ids = set()
for meta in results['metadatas']:
    if meta.get('doc_id'):
        doc_ids.add(meta['doc_id'])
        
print(f"Found {len(doc_ids)} unique doc_ids (first 10 chunks)")
for doc_id in sorted(doc_ids)[:5]:
    print(f"  - {doc_id}")

# Check Citation Graph
print("\n=== Citation Graph Nodes ===")
db_path = "/workspaces/governance-rag/rag_data/academic_citation_graph.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get sample nodes
cursor.execute("SELECT node_id, node_type, title FROM nodes LIMIT 10")
print("Sample nodes:")
for row in cursor:
    print(f"  [{row['node_type']}] {row['node_id']}: {row['title'][:60] if row['title'] else 'N/A'}...")

# Count by type
cursor.execute("SELECT node_type, COUNT(*) as count FROM nodes GROUP BY node_type")
print("\nNode counts by type:")
for row in cursor:
    print(f"  {row['node_type']}: {row['count']}")

# Check edges
cursor.execute("SELECT COUNT(*) as count FROM edges")
print(f"\nTotal edges: {cursor.fetchone()['count']}")

# Sample edges
cursor.execute("SELECT source, target FROM edges LIMIT 5")
print("\nSample edges:")
for row in cursor:
    print(f"  {row['source']} -> {row['target']}")

# Try to find nodes with '2025' in ID or title (looking for academic references)
print("\n=== Searching for academic nodes ===")
cursor.execute("SELECT node_id, node_type, title FROM nodes WHERE node_id LIKE '%academic%' OR node_id LIKE '%2025%' OR title LIKE '%2025%' LIMIT 10")
matches = cursor.fetchall()
if matches:
    print(f"Found {len(matches)} matches:")
    for row in matches:
        print(f"  [{row['node_type']}] {row['node_id']}: {row['title'][:80] if row['title'] else 'N/A'}")
else:
    print("No matches found for '2025'")

# Check for document type nodes
cursor.execute("SELECT node_id, title FROM nodes WHERE node_type = 'document' LIMIT 10")
print("\nDocument-type nodes:")
for row in cursor:
    print(f"  {row['node_id']}: {row['title'][:80] if row['title'] else 'N/A'}")

conn.close()
