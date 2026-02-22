#!/usr/bin/env python3
"""
Smoke test for SQLite consistency graph migration.

Tests:
1. Schema creation
2. Graph writing (nodes, edges, clusters)
3. Graph reading via SQLiteGraphStore
4. Edge canonicalisation and undirected semantics
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.consistency_graph.sqlite_schema import ensure_schema
from scripts.consistency_graph.sqlite_store import SQLiteGraphStore
from scripts.consistency_graph.sqlite_writer import SQLiteGraphWriter


def test_schema_creation():
    """Test schema creation."""
    print("TEST 1: Schema creation")
    db_path = "/tmp/test_consistency_graph.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        ensure_schema(conn, drop_existing=False)

        # Verify tables exist
        cursor = conn.cursor()
        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t[0] for t in tables]

        expected = ["clusters", "edges", "metadata", "node_clusters", "nodes"]
        assert table_names == expected, f"Expected {expected}, got {table_names}"

        # Verify view exists
        views = cursor.execute("SELECT name FROM sqlite_master WHERE type='view'").fetchall()
        assert len(views) == 1 and views[0][0] == "edges_directed", "edges_directed view missing"

        print("✓ Schema creation completed\n")
    except Exception as e:
        raise e
    finally:
        if conn:
            conn.close()

    try:
        os.remove(db_path)
    except FileNotFoundError:
        # Ignore if already removed
        print("No DB file to remove")

    print("✓ Schema creation completed\n")


def test_graph_writing():
    """Test writing a simple graph."""
    print("TEST 2: Graph writing")
    db_path = "/tmp/test_consistency_graph.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)

    # Create test graph
    graph = {
        "nodes": {
            "doc1_v1": {
                "doc_id": "doc1",
                "version": 1,
                "doc_type": "policy",
                "summary": "Test document 1",
                "conflict_score": 0.5,
            },
            "doc2_v1": {
                "doc_id": "doc2",
                "version": 1,
                "doc_type": "code",
                "language": "python",
                "summary": "Test document 2",
                "conflict_score": 0.3,
            },
        },
        "edges": [
            {
                "source": "doc1_v1",
                "target": "doc2_v1",
                "similarity": 0.8,
                "confidence": 0.9,
                "severity": 0.7,
                "relationship": "conflict",
                "explanation": "Test conflict",
            },
        ],
        "clusters": {
            "risk": [
                {
                    "id": 0,
                    "type": "risk",
                    "label": "Test Risk Cluster",
                    "summary": "Test summary",
                    "size": 2,
                    "members": ["doc1_v1", "doc2_v1"],
                }
            ],
            "topic": [],
        },
    }

    # Write graph
    with SQLiteGraphWriter(db_path, replace=True) as writer:
        writer.insert_nodes_batch(graph["nodes"])
        writer.insert_edges_batch(graph["edges"])
        writer.insert_clusters(graph["clusters"])
        writer.set_build_metadata("test_key", "test_value")

    # Verify data
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        node_count = cursor.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        assert node_count == 2, f"Expected 2 nodes, got {node_count}"

        edge_count = cursor.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        assert edge_count == 1, f"Expected 1 edge, got {edge_count}"

        # Verify edge canonicalisation (doc1_v1 < doc2_v1)
        edge = cursor.execute("SELECT node_a, node_b FROM edges").fetchone()
        assert edge == ("doc1_v1", "doc2_v1"), f"Edge not canonicalised: {edge}"

        # Verify edges_directed view returns both directions
        directed_count = cursor.execute("SELECT COUNT(*) FROM edges_directed").fetchone()[0]
        assert directed_count == 2, f"Expected 2 directed edges, got {directed_count}"

        cluster_count = cursor.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        assert cluster_count == 1, f"Expected 1 cluster, got {cluster_count}"

        meta_value = cursor.execute("SELECT value FROM metadata WHERE key='test_key'").fetchone()[0]
        assert meta_value == "test_value", f"Metadata mismatch: {meta_value}"

    except Exception as e:
        raise e
    finally:
        if conn:
            conn.close()

    try:
        os.remove(db_path)
    except FileNotFoundError:
        # Ignore if already removed
        print("No DB file to remove")

    print("✓ Graph writing completed\n")


def test_graph_reading():
    """Test reading graph via SQLiteGraphStore."""
    print("TEST 3: Graph reading via SQLiteGraphStore")
    db_path = "/tmp/test_consistency_graph.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)

    # Create test graph
    graph = {
        "nodes": {
            "doc1_v1": {"doc_id": "doc1", "version": 1, "doc_type": "policy"},
            "doc2_v1": {"doc_id": "doc2", "version": 1, "doc_type": "code"},
            "doc3_v1": {"doc_id": "doc3", "version": 1, "doc_type": "standard"},
        },
        "edges": [
            {
                "source": "doc1_v1",
                "target": "doc2_v1",
                "similarity": 0.8,
                "confidence": 0.9,
                "severity": 0.7,
                "relationship": "conflict",
            },
        ],
        "clusters": {"risk": [], "topic": []},
    }

    # Write graph
    with SQLiteGraphWriter(db_path, replace=True) as writer:
        writer.insert_nodes_batch(graph["nodes"])
        writer.insert_edges_batch(graph["edges"])
        writer.insert_clusters(graph["clusters"])
        writer.set_build_metadata("total_nodes", "3")

    # Read via store
    store = None
    try:
        store = SQLiteGraphStore(db_path)
        assert store.load_metadata(), "Failed to load metadata"

        metadata = store.get_metadata()
        assert metadata.get("total_nodes") == "3", f"Metadata mismatch: {metadata}"

        node_ids = store.get_node_ids()
        assert len(node_ids) == 3, f"Expected 3 node IDs, got {len(node_ids)}"
        assert set(node_ids) == {"doc1_v1", "doc2_v1", "doc3_v1"}, f"Node IDs mismatch: {node_ids}"

        # Test pagination
        nodes, total_pages = store.get_nodes_paginated(page=0, page_size=2)
        assert len(nodes) == 2, f"Expected 2 nodes in page, got {len(nodes)}"
        assert total_pages == 2, f"Expected 2 pages, got {total_pages}"

        # Test get_node
        node = store.get_node("doc1_v1")
        assert node is not None, "Failed to get node"
        assert node["doc_type"] == "policy", f"Node doc_type mismatch: {node}"

        # Test get_edges (should return 2 directed edges from 1 undirected)
        edges = store.get_edges()
        assert len(edges) == 2, f"Expected 2 directed edges, got {len(edges)}"

        # Verify both directions present
        sources = {e["source"] for e in edges}
        targets = {e["target"] for e in edges}
        assert sources == {"doc1_v1", "doc2_v1"}, f"Source mismatch: {sources}"
        assert targets == {"doc1_v1", "doc2_v1"}, f"Target mismatch: {targets}"
    finally:
        if store:
            store.close()

    os.remove(db_path)
    print("✓ Graph reading completed\n")


def test_edge_canonicalisation():
    """Test edge canonicalisation enforces node_a < node_b."""
    print("TEST 4: Edge canonicalisation")
    db_path = "/tmp/test_consistency_graph.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)

    # Create edges in non-canonical order
    graph = {
        "nodes": {
            "zebra_v1": {"doc_id": "zebra", "version": 1},
            "apple_v1": {"doc_id": "apple", "version": 1},
        },
        "edges": [
            {
                "source": "zebra_v1",  # Should be swapped to node_b
                "target": "apple_v1",  # Should become node_a
                "similarity": 0.5,
                "confidence": 0.5,
                "severity": 0.5,
                "relationship": "consistent",
            },
        ],
        "clusters": {"risk": [], "topic": []},
    }

    with SQLiteGraphWriter(db_path, replace=True) as writer:
        writer.insert_nodes_batch(graph["nodes"])
        writer.insert_edges_batch(graph["edges"])

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        edge = cursor.execute("SELECT node_a, node_b FROM edges").fetchone()
    finally:
        if conn:
            conn.close()

    assert edge == ("apple_v1", "zebra_v1"), f"Edge not canonicalised: {edge}"

    try:
        os.remove(db_path)
    except FileNotFoundError:
        # Ignore if already removed
        print("No DB file to remove")

    print("✓ Edge canonicalisation completed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("SQLite Consistency Graph Migration - Smoke Tests")
    print("=" * 60 + "\n")

    try:
        test_schema_creation()
        test_graph_writing()
        test_graph_reading()
        test_edge_canonicalisation()

        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
