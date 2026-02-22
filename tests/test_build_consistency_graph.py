"""Tests for build_consistency_graph module.

Covers normalisation helpers, JSON repair/extraction, severity computation,
NetworkX conversion/metrics, cluster assignment (including overlapping), and
parallel graph building with mocked edge generation.
"""

import json
from pathlib import Path

import networkx as nx
import pytest

from scripts.consistency_graph import build_consistency_graph as bcg


class TestNormalisation:
    def test_normalise_relationship_exact_and_fuzzy(self):
        assert bcg.normalise_relationship("consistent") == "consistent"
        assert bcg.normalise_relationship("Fully aligned") == "consistent"
        assert bcg.normalise_relationship("partially conflicting") == "partial_conflict"
        assert bcg.normalise_relationship("contradiction") == "conflict"
        assert bcg.normalise_relationship("near duplicate") == "duplicate"
        # Unknown defaults to consistent
        assert bcg.normalise_relationship("something else") == "consistent"


class TestJsonHelpers:
    def test_repair_and_extract_first_json_block(self):
        raw_text = 'Extra text before {"a": 1,\n "b": 2,}\nTrailing'
        repaired = bcg.repair_json(raw_text)
        obj = bcg.extract_first_json_block(repaired)
        assert obj == {"a": 1, "b": 2}


class TestSeverityAndMetrics:
    def test_compute_edge_severity_scales(self):
        # conflict should have the highest base weight
        sev_conflict = bcg.compute_edge_severity("conflict", confidence=1.0, similarity=1.0)
        sev_partial = bcg.compute_edge_severity("partial_conflict", confidence=1.0, similarity=1.0)
        sev_duplicate = bcg.compute_edge_severity("duplicate", confidence=1.0, similarity=1.0)
        assert sev_conflict > sev_partial > sev_duplicate
        # confidence and similarity scale linearly
        sev_scaled = bcg.compute_edge_severity("conflict", confidence=0.5, similarity=0.5)
        assert sev_scaled == pytest.approx(sev_conflict * 0.25)

    def test_distance_to_similarity_clamps_bounds(self):
        assert bcg.distance_to_similarity(0.0) == pytest.approx(1.0)
        assert bcg.distance_to_similarity(0.4) == pytest.approx(0.6)
        assert bcg.distance_to_similarity(1.2) == pytest.approx(0.0)
        assert bcg.distance_to_similarity(-0.5) == pytest.approx(1.0)

    def test_to_networkx_merges_edges(self):
        graph = {
            "nodes": {
                "a_v1": {"doc_id": "a", "version": 1},
                "b_v1": {"doc_id": "b", "version": 1},
            },
            "edges": [
                {"source": "a_v1", "target": "b_v1", "severity": 0.2, "similarity": 0.3},
                {"source": "a_v1", "target": "b_v1", "severity": 0.7, "similarity": 0.6},
            ],
        }
        G = bcg.to_networkx(graph)
        assert G.has_edge("a_v1", "b_v1")
        # merged edge keeps max severity/similarity
        assert G["a_v1"]["b_v1"]["severity"] == 0.7
        assert G["a_v1"]["b_v1"]["similarity"] == 0.6

    def test_compute_version_node_conflict(self):
        G = nx.Graph()
        G.add_node("a_v1")
        G.add_node("b_v1")
        G.add_node("c_v1")
        G.add_edge("a_v1", "b_v1", severity=0.5)
        G.add_edge("a_v1", "c_v1", severity=0.3)
        bcg.compute_version_node_conflict(G)
        assert G.nodes["a_v1"]["conflict_score"] == pytest.approx(0.8)
        assert G.nodes["b_v1"]["conflict_score"] == pytest.approx(0.5)
        assert G.nodes["c_v1"]["conflict_score"] == pytest.approx(0.3)


class TestLoader:
    def test_load_versioned_docs_excludes_documents_when_requested(self):
        calls = []

        class FakeCollection:
            def get(self, include, limit, offset, where=None):
                calls.append(include)
                return {
                    "metadatas": [
                        {
                            "doc_id": "x",
                            "version": 1,
                            "summary": "s",
                            "doc_type": "d",
                        }
                    ],
                    "embeddings": [[0.1, 0.2]],
                    "documents": ["text-body"],
                }

        collection = FakeCollection()
        records = bcg.load_versioned_docs(collection, batch_size=10, include_documents=False)

        # Should not request documents when include_documents is False
        assert calls[0] == ["metadatas", "embeddings"]
        # Returned record should have empty text placeholder
        assert records[0]["text"] == ""


class TestClusters:
    def test_compute_version_node_clusters_overlapping(self, monkeypatch):
        # Build simple graph
        G = nx.Graph()
        for n in ["a_v1", "b_v1", "c_v1"]:
            G.add_node(n, doc_id=n.split("_v")[0], version=1)
        G.add_edge("a_v1", "b_v1", severity=1.0, similarity=0.9)
        G.add_edge("b_v1", "c_v1", severity=0.6, similarity=0.6)

        # Stub community detection to control clusters
        def fake_greedy(graph, weight=None):
            if weight == "risk_weight":
                return [set(["a_v1", "b_v1"]), set(["c_v1"])]
            return [set(["a_v1"]), set(["b_v1", "c_v1"])]

        monkeypatch.setattr(bcg, "greedy_modularity_communities", fake_greedy)

        risk, topic = bcg.compute_version_node_clusters(G)

        # Primary assignments present
        assert set(G.nodes["a_v1"]["risk_clusters"]) == {0}
        # c_v1 should include cluster 1; overlapping may add more
        assert 1 in set(G.nodes["c_v1"].get("risk_clusters", []))
        # Overlapping expectation: b_v1 should include the secondary cluster (1)
        b_risk = set(G.nodes["b_v1"].get("risk_clusters", []))
        assert 1 in b_risk

        # Topic assignments from fake_greedy
        assert 0 in set(G.nodes["a_v1"].get("topic_clusters", []))
        assert 1 in set(G.nodes["b_v1"].get("topic_clusters", []))
        assert 1 in set(G.nodes["c_v1"].get("topic_clusters", []))

    def test_generate_cluster_metadata_invokes_llm_label_cluster(self, monkeypatch):
        G = nx.Graph()
        G.add_node("a_v1", doc_id="a", version=1, summary="alpha")
        G.add_node("b_v1", doc_id="b", version=1, summary="beta")
        clusters = [set(["a_v1", "b_v1"])]

        called = {}

        def fake_llm_label_cluster(
            docs,
            severities,
            topics,
            cluster_type,
            is_code_cluster=False,
            code_languages=None,
            code_services=None,
        ):
            called["docs"] = docs
            called["severities"] = severities
            called["topics"] = topics
            called["cluster_type"] = cluster_type
            called["is_code_cluster"] = is_code_cluster
            called["code_languages"] = code_languages
            called["code_services"] = code_services
            return ("L", "D", "S")

        monkeypatch.setattr(bcg, "llm_label_cluster", fake_llm_label_cluster)

        meta = bcg.generate_cluster_metadata(G, clusters, "risk")
        assert meta[0]["label"] == "L"
        assert meta[0]["description"] == "D"
        assert meta[0]["summary"] == "S"
        assert meta[0]["size"] == 2
        # Should not be code cluster
        assert called["is_code_cluster"] is False

    def test_generate_cluster_metadata_code_cluster(self, monkeypatch):
        G = nx.Graph()
        G.add_node(
            "c1",
            doc_id="svc1",
            version=1,
            summary="auth service",
            source_category="code",
            language="java",
            service="auth",
        )
        G.add_node(
            "c2",
            doc_id="svc2",
            version=1,
            summary="payment service",
            source_category="code",
            language="java",
            service="payment",
        )
        clusters = [set(["c1", "c2"])]

        called = {}

        def fake_llm_label_cluster(
            docs,
            severities,
            topics,
            cluster_type,
            is_code_cluster=False,
            code_languages=None,
            code_services=None,
        ):
            called["docs"] = docs
            called["severities"] = severities
            called["topics"] = topics
            called["cluster_type"] = cluster_type
            called["is_code_cluster"] = is_code_cluster
            called["code_languages"] = code_languages
            called["code_services"] = code_services
            return ("CodeLabel", "CodeDesc", "CodeSummary")

        monkeypatch.setattr(bcg, "llm_label_cluster", fake_llm_label_cluster)

        meta = bcg.generate_cluster_metadata(G, clusters, "topic")
        assert meta[0]["label"] == "CodeLabel"
        assert meta[0]["description"] == "CodeDesc"
        assert meta[0]["summary"] == "CodeSummary"
        assert meta[0]["size"] == 2
        # Should be detected as code cluster
        assert called["is_code_cluster"] is True
        assert set(called["code_languages"]) == {"java"}
        assert set(called["code_services"]) == {"auth", "payment"}


class TestParallelGraphBuild:
    def test_build_consistency_graph_parallel_uses_stubbed_edges(self, monkeypatch):
        # Prepare versioned docs
        versioned_docs = [
            {
                "doc_id": "a",
                "version": 1,
                "timestamp": None,
                "doc_type": "typeA",
                "summary": "s1",
                "source_category": "cat",
                "health": {},
                "embedding": [0.0],
                "text": "t1",
            },
            {
                "doc_id": "b",
                "version": 1,
                "timestamp": None,
                "doc_type": "typeB",
                "summary": "s2",
                "source_category": "cat",
                "health": {},
                "embedding": [0.0],
                "text": "t2",
            },
        ]

        def stub_process(
            record,
            doc_collection,
            max_neighbours,
            sim_threshold,
            enable_heuristic=True,
            embedding_cache=None,
            llm_batcher=None,
            enable_dynamic_expansion=True,
            expansion_quality_threshold=0.7,
            max_expansion_multiplier=1.5,
        ):
            if record["doc_id"] == "a":
                return [
                    {
                        "source": "a_v1",
                        "target": "b_v1",
                        "relationship": "conflict",
                        "confidence": 0.9,
                        "explanation": "stub",
                        "similarity": 0.8,
                        "severity": 0.72,
                        "version_source": 1,
                        "version_target": 1,
                    }
                ], {
                    "llm_calls": 1,
                    "filtered": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "expansion_applied": False,
                    "expanded_neighbours_found": 0,
                }
            return [], {
                "llm_calls": 0,
                "filtered": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "expansion_applied": False,
                "expanded_neighbours_found": 0,
            }

        monkeypatch.setattr(bcg, "process_document_for_graph", stub_process)

        graph = bcg.build_consistency_graph_parallel(
            versioned_docs=versioned_docs,
            doc_collection=None,
            max_neighbours=10,
            sim_threshold=0.5,
            workers=2,
            progress_callback=None,
        )

        # Nodes were created
        assert set(graph["nodes"].keys()) == {"a_v1", "b_v1"}
        # Edge from stub present
        assert len(graph["edges"]) == 1
        edge = graph["edges"][0]
        assert edge["source"] == "a_v1"
        assert edge["target"] == "b_v1"
        assert edge["severity"] == pytest.approx(0.72)

    def test_dedupe_edges_normalises_and_keeps_highest(self):
        edges = [
            {"source": "b_v1", "target": "a_v1", "severity": 0.3},
            {"source": "a_v1", "target": "b_v1", "severity": 0.8},
            {"source": "a_v1", "target": "c_v1", "severity": 0.1},
        ]

        deduped = bcg.dedupe_edges(edges)

        # Should collapse to two undirected edges with normalised ordering
        assert len(deduped) == 2
        edge_map = {(e["source"], e["target"]): e for e in deduped}
        assert ("a_v1", "b_v1") in edge_map
        assert edge_map[("a_v1", "b_v1")]["severity"] == pytest.approx(0.8)
        assert ("a_v1", "c_v1") in edge_map


class TestErrorHandling:
    """Tests for error handling and logging in main() and build functions."""

    def test_main_handles_collection_load_failure(self, monkeypatch, capsys):
        """Test that main() handles ChromaDB collection load failures gracefully."""

        def failing_get_collection():
            raise Exception("ChromaDB connection failed")

        monkeypatch.setattr(bcg, "get_collection", failing_get_collection)
        monkeypatch.setattr(
            bcg,
            "parse_args",
            lambda: type(
                "Args",
                (),
                {
                    "output": "tests/test.json",
                    "sqlite_output": None,
                    "max_neighbours": 10,
                    "similarity_threshold": 0.4,
                    "workers": 2,
                    "progress_interval": 10,
                    "include_documents": True,
                    "filter_doc_type": "all",
                    "purge_logs": False,
                    "dashboard_mode": False,
                    "enable_llm_batching": False,
                    "enable_embedding_cache": False,
                    "enable_graph_sampling": False,
                    "sampling_rate": 0.1,
                },
            )(),
        )

        with pytest.raises(SystemExit) as exc_info:
            bcg.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Failed to load ChromaDB collection" in captured.out

    def test_main_handles_graph_build_failure(self, monkeypatch, capsys):
        """Test that main() handles graph construction failures gracefully."""
        monkeypatch.setattr(bcg, "get_collection", lambda: "fake_collection")
        monkeypatch.setattr(
            bcg,
            "parse_args",
            lambda: type(
                "Args",
                (),
                {
                    "output": "tests/test.json",
                    "sqlite_output": None,
                    "max_neighbours": 10,
                    "similarity_threshold": 0.4,
                    "workers": 2,
                    "progress_interval": 10,
                    "include_documents": True,
                    "filter_doc_type": "all",
                    "purge_logs": False,
                    "dashboard_mode": False,
                    "enable_llm_batching": False,
                    "enable_embedding_cache": False,
                    "enable_graph_sampling": False,
                    "sampling_rate": 0.1,
                    "include_advanced_analytics": False,
                },
            )(),
        )

        def failing_build(*args, **kwargs):
            raise ValueError("Invalid graph parameters")

        monkeypatch.setattr(bcg, "build_consistency_graph", failing_build)

        with pytest.raises(SystemExit) as exc_info:
            bcg.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Failed to build consistency graph" in captured.out

    def test_main_continues_on_visualisation_failure(self, monkeypatch, capsys, tmp_path):
        """Test that main() completes successfully with SQLite output."""
        output_sqlite = tmp_path / "test_graph.db"

        monkeypatch.setattr(bcg, "get_collection", lambda: "fake_collection")
        monkeypatch.setattr(
            bcg,
            "parse_args",
            lambda: type(
                "Args",
                (),
                {
                    "output": "tests/test.json",
                    "sqlite_output": str(output_sqlite),
                    "max_neighbours": 10,
                    "similarity_threshold": 0.4,
                    "workers": 2,
                    "progress_interval": 10,
                    "include_documents": True,
                    "filter_doc_type": "all",
                    "purge_logs": False,
                    "dashboard_mode": False,
                    "enable_llm_batching": False,
                    "enable_embedding_cache": False,
                    "enable_graph_sampling": False,
                    "sampling_rate": 0.1,
                    "include_advanced_analytics": False,
                },
            )(),
        )

        # Successful graph build
        test_graph = {
            "nodes": {"a_v1": {"doc_id": "a", "version": 1}},
            "edges": [],
            "clusters": {"risk": {}, "topic": {}},
        }
        monkeypatch.setattr(bcg, "build_consistency_graph", lambda *a, **k: test_graph)

        # Mock SQLiteGraphWriter
        class MockSQLiteWriter:
            def __init__(self, path, replace=False):
                self.path = path

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def insert_nodes_batch(self, nodes):
                pass

            def insert_edges_batch(self, edges):
                pass

            def insert_clusters(self, clusters):
                pass

            def set_build_metadata(self, key, value):
                pass

            def atomic_swap(self, target):
                # Create the output file to simulate success
                Path(target).touch()

        monkeypatch.setattr(bcg, "SQLiteGraphWriter", MockSQLiteWriter)

        # Should complete successfully
        bcg.main()

        captured = capsys.readouterr()
        assert "Consistency graph built successfully" in captured.out
        assert output_sqlite.exists()

    def test_main_handles_json_write_failure(self, monkeypatch, capsys, tmp_path):
        """Test that SQLite write failures are handled properly."""
        output_sqlite = tmp_path / "test_graph.db"

        monkeypatch.setattr(bcg, "get_collection", lambda: "fake_collection")
        monkeypatch.setattr(
            bcg,
            "parse_args",
            lambda: type(
                "Args",
                (),
                {
                    "output": "tests/test.json",
                    "sqlite_output": str(output_sqlite),
                    "max_neighbours": 10,
                    "similarity_threshold": 0.4,
                    "workers": 2,
                    "progress_interval": 10,
                    "include_documents": True,
                    "filter_doc_type": "all",
                    "purge_logs": False,
                    "dashboard_mode": False,
                    "enable_llm_batching": False,
                    "enable_embedding_cache": False,
                    "enable_graph_sampling": False,
                    "sampling_rate": 0.1,
                    "include_advanced_analytics": False,
                },
            )(),
        )

        test_graph = {
            "nodes": {"a_v1": {"doc_id": "a", "version": 1}},
            "edges": [],
            "clusters": {"risk": {}, "topic": {}},
        }
        monkeypatch.setattr(bcg, "build_consistency_graph", lambda *a, **k: test_graph)

        # Mock SQLiteGraphWriter to fail
        class FailingSQLiteWriter:
            def __init__(self, path, replace=False):
                raise IOError("Failed to write SQLite database")

        monkeypatch.setattr(bcg, "SQLiteGraphWriter", FailingSQLiteWriter)

        with pytest.raises(SystemExit) as exc_info:
            bcg.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Failed to write SQLite graph" in captured.out

    def test_main_handles_keyboard_interrupt(self, monkeypatch, capsys):
        """Test that keyboard interrupts are handled gracefully."""
        monkeypatch.setattr(bcg, "get_collection", lambda: "fake_collection")
        monkeypatch.setattr(
            bcg,
            "parse_args",
            lambda: type(
                "Args",
                (),
                {
                    "output": "tests/test.json",
                    "sqlite_output": None,
                    "max_neighbours": 10,
                    "similarity_threshold": 0.4,
                    "workers": 2,
                    "progress_interval": 10,
                    "include_documents": True,
                    "filter_doc_type": "all",
                    "purge_logs": False,
                    "dashboard_mode": False,
                    "enable_llm_batching": False,
                    "enable_embedding_cache": False,
                    "enable_graph_sampling": False,
                    "sampling_rate": 0.1,
                    "include_advanced_analytics": False,
                },
            )(),
        )

        def interrupted_build(*args, **kwargs):
            raise KeyboardInterrupt()

        monkeypatch.setattr(bcg, "build_consistency_graph", interrupted_build)

        with pytest.raises(SystemExit) as exc_info:
            bcg.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "interrupted" in captured.out.lower()

    def test_main_success_path_with_all_outputs(self, monkeypatch, capsys, tmp_path):
        """Test successful execution with SQLite output."""
        output_sqlite = tmp_path / "test_graph.db"

        # Create logs directory
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        monkeypatch.setattr(bcg, "get_collection", lambda: "fake_collection")
        monkeypatch.setattr(
            bcg,
            "parse_args",
            lambda: type(
                "Args",
                (),
                {
                    "output": "tests/test.json",
                    "sqlite_output": str(output_sqlite),
                    "max_neighbours": 10,
                    "similarity_threshold": 0.4,
                    "workers": 2,
                    "progress_interval": 10,
                    "include_documents": True,
                    "filter_doc_type": "all",
                    "purge_logs": False,
                    "dashboard_mode": False,
                    "enable_llm_batching": False,
                    "enable_embedding_cache": False,
                    "enable_graph_sampling": False,
                    "sampling_rate": 0.1,
                    "include_advanced_analytics": False,
                },
            )(),
        )

        test_graph = {
            "nodes": {"a_v1": {"doc_id": "a", "version": 1}, "b_v1": {"doc_id": "b", "version": 1}},
            "edges": [{"source": "a_v1", "target": "b_v1", "severity": 0.5}],
            "clusters": {
                "risk": {"0": {"label": "Risk Cluster", "size": 2}},
                "topic": {"0": {"label": "Topic Cluster", "size": 2}},
            },
        }
        monkeypatch.setattr(bcg, "build_consistency_graph", lambda *a, **k: test_graph)

        # Mock SQLiteGraphWriter
        class MockSQLiteWriter:
            def __init__(self, path, replace=False):
                self.path = path

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def insert_nodes_batch(self, nodes):
                pass

            def insert_edges_batch(self, edges):
                pass

            def insert_clusters(self, clusters):
                pass

            def set_build_metadata(self, key, value):
                pass

            def atomic_swap(self, target):
                # Create the output file to simulate success
                Path(target).touch()

        monkeypatch.setattr(bcg, "SQLiteGraphWriter", MockSQLiteWriter)

        bcg.main()

        captured = capsys.readouterr()
        assert "Consistency graph built successfully" in captured.out
        assert "Nodes: 2" in captured.out
        assert "Edges: 1" in captured.out
        assert "Risk Clusters: 1" in captured.out
        assert "Topic Clusters: 1" in captured.out
        assert output_sqlite.exists()


class TestDocTypeFilter:
    def test_filter_doc_type_passes_correct_where(self, monkeypatch):
        """Test that --filter-doc-type sets the correct ChromaDB filter."""
        called = {}

        # Patch load_versioned_docs to capture 'where' argument
        def fake_load_versioned_docs(
            collection, batch_size=500, where=None, include_documents=True
        ):
            called["where"] = where
            # Return a minimal doc list
            return [
                {
                    "doc_id": "x",
                    "version": 1,
                    "timestamp": None,
                    "doc_type": "java",
                    "summary": "s",
                    "source_category": "code",
                    "health": {},
                    "embedding": [0.0],
                    "text": "code...",
                }
            ]

        monkeypatch.setattr(bcg, "load_versioned_docs", fake_load_versioned_docs)
        # Patch build_consistency_graph_parallel to return a valid graph structure
        monkeypatch.setattr(
            bcg,
            "build_consistency_graph_parallel",
            lambda *a, **k: {"nodes": {"x_v1": {"doc_id": "x", "version": 1}}, "edges": []},
        )

        # Patch to_networkx to return a valid NetworkX Graph with the expected node
        def fake_to_networkx(graph):
            G = nx.Graph()
            G.add_node(
                "x_v1",
                doc_id="x",
                version=1,
                conflict_score=0.0,
                risk_clusters=[0],
                topic_clusters=[0],
            )
            return G

        monkeypatch.setattr(bcg, "to_networkx", fake_to_networkx)
        # Patch generate_cluster_metadata to avoid downstream errors
        monkeypatch.setattr(
            bcg,
            "generate_cluster_metadata",
            lambda G, clusters, cluster_type: {
                0: {"label": "L", "description": "D", "summary": "S", "size": 1}
            },
        )
        # Patch parse_args to simulate CLI
        monkeypatch.setattr(
            bcg,
            "parse_args",
            lambda: type(
                "Args",
                (),
                {
                    "output": "tests/test.json",
                    "sqlite_output": None,
                    "max_neighbours": 10,
                    "similarity_threshold": 0.4,
                    "workers": 2,
                    "progress_interval": 10,
                    "include_documents": True,
                    "purge_logs": False,
                    "filter_doc_type": "java",
                    "dashboard_mode": False,
                    "enable_llm_batching": False,
                    "enable_embedding_cache": False,
                    "enable_graph_sampling": False,
                    "sampling_rate": 0.1,
                    "include_advanced_analytics": False,
                },
            )(),
        )
        monkeypatch.setattr(bcg, "get_collection", lambda: "fake_collection")
        monkeypatch.setattr(bcg, "audit", lambda *a, **k: None)
        monkeypatch.setattr(
            bcg,
            "logger",
            type(
                "Logger",
                (),
                {
                    "info": lambda *a, **k: None,
                    "error": lambda *a, **k: None,
                    "warning": lambda *a, **k: None,
                },
            )(),
        )

        # Mock SQLiteGraphWriter
        class MockSQLiteWriter:
            def __init__(self, path, replace=False):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def insert_nodes_batch(self, nodes):
                pass

            def insert_edges_batch(self, edges):
                pass

            def insert_clusters(self, clusters):
                pass

            def set_build_metadata(self, key, value):
                pass

            def atomic_swap(self, target):
                pass

        monkeypatch.setattr(bcg, "SQLiteGraphWriter", MockSQLiteWriter)

        # Run main
        bcg.main()
        # Should filter for code+java
        assert called["where"] == {"source_category": "code", "doc_type": "java"}


class TestOptimisations:
    """Tests for opt-in optimisation features (batching, caching, sampling)."""

    def test_llm_batcher_caches_results(self):
        """Test that LLMBatcher caches results and avoids duplicate LLM calls."""
        batcher = bcg.LLMBatcher(batch_size=2)

        # Add first pair
        result1 = batcher.validate_pair(
            "doc_a_text", "doc_b_text", {"doc_type": "doc_a"}, {"doc_type": "doc_b"}
        )

        # First call should be queued
        assert len(batcher.pending_pairs) == 1
        cache_key_1 = batcher.pending_pairs[0]["cache_key"]

        # Same validation should return from cache (same cache_key)
        result2 = batcher.validate_pair(
            "doc_a_text", "doc_b_text", {"doc_type": "doc_a"}, {"doc_type": "doc_b"}
        )

        # After batch processing (triggered by 2nd call), pending should be empty
        # but cache should have the key
        assert len(batcher.pending_pairs) == 0
        # Both results should reference the same cache operation
        assert len(batcher.results_cache) >= 1

    def test_llm_batcher_processes_at_batch_size(self):
        """Test that LLMBatcher triggers processing at batch size limit."""
        batcher = bcg.LLMBatcher(batch_size=2)

        # Add first pair - should queue
        batcher.validate_pair("doc1", "doc2", {}, {})
        assert len(batcher.pending_pairs) == 1

        # Add second pair - should trigger batch processing
        batcher.validate_pair("doc3", "doc4", {}, {})
        # After batch processing, pending should be empty
        assert len(batcher.pending_pairs) == 0

    def test_embedding_cache_hit_and_miss(self):
        """Test EmbeddingCacheManager hit/miss tracking."""
        cache = bcg.EmbeddingCacheManager()

        # Cache miss
        result1 = cache.get("doc1", "mxbai-embed-large")
        assert result1 is None
        assert cache.misses == 1

        # Put in cache
        embedding = [0.1, 0.2, 0.3]
        cache.put("doc1", "mxbai-embed-large", embedding)

        # Cache hit
        result2 = cache.get("doc1", "mxbai-embed-large")
        assert result2 == embedding
        assert cache.hits == 1

        # Different model is a miss
        result3 = cache.get("doc1", "other-model")
        assert result3 is None
        assert cache.misses == 2

    def test_embedding_cache_stats(self):
        """Test EmbeddingCacheManager statistics reporting."""
        cache = bcg.EmbeddingCacheManager()

        # Add some data
        cache.put("doc1", "model1", [0.1])
        cache.put("doc2", "model1", [0.2])
        cache.get("doc1", "model1")  # hit
        cache.get("doc1", "model1")  # hit
        cache.get("doc3", "model1")  # miss

        stats = cache.stats()
        assert stats["cache_size"] == 2
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["total"] == 3
        assert stats["hit_rate"] == pytest.approx(66.67, abs=1)

    def test_sample_and_interpolate_graph_samples_correctly(self):
        """Test that graph sampling creates sample and remaining subsets."""
        versioned_docs = [
            {
                "doc_id": f"doc{i}",
                "version": 1,
                "embedding": [float(i)],
                "summary": f"Summary {i}",
                "doc_type": "test",
                "source_category": "test",
                "health": {},
                "text": f"Text {i}",
            }
            for i in range(10)
        ]

        # Mock collection to return predictable neighbors
        class MockCollection:
            def query(self, query_embeddings, n_results, include=None):
                return {
                    "metadatas": [[{"doc_id": "doc0", "version": 1}]],
                    "distances": [[0.1]],
                }

        # Mock process_document_for_graph to return empty edges
        def mock_build(*args, **kwargs):
            return {
                "nodes": {f"doc{i}_v1": {"doc_id": f"doc{i}", "version": 1} for i in range(10)},
                "edges": [],
                "clusters": {"risk": {}, "topic": {}},
            }

        # Use mock to test structure
        import unittest.mock as mock

        with mock.patch(
            "scripts.consistency_graph.build_consistency_graph.build_consistency_graph_parallel",
            mock_build,
        ):
            result = bcg.sample_and_interpolate_graph(
                versioned_docs=versioned_docs,
                doc_collection=MockCollection(),
                max_neighbours=5,
                sim_threshold=0.4,
                sampling_rate=0.3,  # 3 out of 10
                workers=2,
            )

        # Should have all nodes (sampled + remaining)
        assert len(result["nodes"]) == 10
        # All nodes should be present
        for i in range(10):
            assert f"doc{i}_v1" in result["nodes"]

    def test_build_consistency_graph_parallel_with_batching(self, monkeypatch):
        """Test that enable_llm_batching flag is passed through correctly."""
        versioned_docs = [
            {
                "doc_id": "a",
                "version": 1,
                "timestamp": None,
                "doc_type": "typeA",
                "summary": "s1",
                "source_category": "cat",
                "health": {},
                "embedding": [0.0],
                "text": "t1",
            },
        ]

        batcher_created = {}

        def stub_process(
            record,
            doc_collection,
            max_neighbours,
            sim_threshold,
            enable_heuristic=True,
            embedding_cache=None,
            llm_batcher=None,
            enable_dynamic_expansion=True,
            expansion_quality_threshold=0.7,
            max_expansion_multiplier=1.5,
        ):
            batcher_created["llm_batcher"] = llm_batcher is not None
            return [], {
                "llm_calls": 0,
                "filtered": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "expansion_applied": False,
                "expanded_neighbours_found": 0,
            }

        monkeypatch.setattr(bcg, "process_document_for_graph", stub_process)

        # Build with batching disabled
        graph1 = bcg.build_consistency_graph_parallel(
            versioned_docs=versioned_docs,
            doc_collection=None,
            max_neighbours=10,
            sim_threshold=0.5,
            enable_llm_batching=False,
        )
        assert batcher_created.get("llm_batcher") is False

        # Build with batching enabled
        graph2 = bcg.build_consistency_graph_parallel(
            versioned_docs=versioned_docs,
            doc_collection=None,
            max_neighbours=10,
            sim_threshold=0.5,
            enable_llm_batching=True,
        )
        assert batcher_created.get("llm_batcher") is True

    def test_build_consistency_graph_parallel_with_embedding_cache(self, monkeypatch):
        """Test that enable_embedding_cache flag is passed through correctly."""
        versioned_docs = [
            {
                "doc_id": "a",
                "version": 1,
                "timestamp": None,
                "doc_type": "typeA",
                "summary": "s1",
                "source_category": "cat",
                "health": {},
                "embedding": [0.0],
                "text": "t1",
            },
        ]

        cache_created = {}

        def stub_process(
            record,
            doc_collection,
            max_neighbours,
            sim_threshold,
            enable_heuristic=True,
            embedding_cache=None,
            llm_batcher=None,
            enable_dynamic_expansion=True,
            expansion_quality_threshold=0.7,
            max_expansion_multiplier=1.5,
        ):
            cache_created["embedding_cache"] = embedding_cache is not None
            return [], {
                "llm_calls": 0,
                "filtered": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "expansion_applied": False,
                "expanded_neighbours_found": 0,
            }

        monkeypatch.setattr(bcg, "process_document_for_graph", stub_process)

        # Build with cache disabled
        graph1 = bcg.build_consistency_graph_parallel(
            versioned_docs=versioned_docs,
            doc_collection=None,
            max_neighbours=10,
            sim_threshold=0.5,
            enable_embedding_cache=False,
        )
        assert cache_created.get("embedding_cache") is False

        # Build with cache enabled
        graph2 = bcg.build_consistency_graph_parallel(
            versioned_docs=versioned_docs,
            doc_collection=None,
            max_neighbours=10,
            sim_threshold=0.5,
            enable_embedding_cache=True,
        )
        assert cache_created.get("embedding_cache") is True
