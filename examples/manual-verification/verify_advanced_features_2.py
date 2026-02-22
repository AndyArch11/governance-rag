#!/usr/bin/env python3
"""Verification script for Advanced Features.

This is a manual, runnable example that exercises:
1. Enhanced Query Analytics
2. Smart Caching Optimisation
3. Advanced Graph Analytics
4. Explainability Features

Run it directly:
  python examples/manual_verification/verify_advanced_features_2.py
"""

import sys
from pathlib import Path

# Add project root to path (examples/ → project root)
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def test_query_analytics():
    print("\n=== Testing Query Analytics ===")
    try:
        from scripts.utils.db_factory import get_cache_client

        cache = get_cache_client(enable_cache=True)
        if not cache or not hasattr(cache, "log_query_analytics"):
            print("(i) Query analytics not supported in current cache backend — skipping")
            return True

        cache.log_query_analytics(
            query_text="What is MFA?",
            k_results=5,
            retrieval_time_ms=123.4,
            generation_time_ms=2341.5,
            num_chunks_retrieved=5,
            avg_similarity_score=0.734,
            max_similarity_score=0.873,
            cache_hit=False,
            is_code_query=False,
            model_name="mistral",
            temperature=0.3,
            metadata={"test": True},
        )

        if hasattr(cache, "query_analytics_stats"):
            stats = cache.query_analytics_stats()
            print(f"✓ Query analytics stats: {stats}")

        if hasattr(cache, "get_recent_queries"):
            recent = cache.get_recent_queries(limit=5)
            print(f"✓ Recent queries: {len(recent)} records")

        return True
    except Exception as e:
        print(f"✗ Query analytics failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_smart_caching():
    print("\n=== Testing Smart Caching ===")
    try:
        from scripts.utils.db_factory import get_cache_client

        cache = get_cache_client(enable_cache=True)
        if not cache:
            print("(i) Cache backend not available — skipping")
            return True

        if hasattr(cache, "track_cache_access"):
            cache.track_cache_access("embedding", "test_hash_001")
            cache.track_cache_access("embedding", "test_hash_001")  # Second access
            cache.track_cache_access("embedding", "test_hash_002")
            print("✓ Cache access tracking completed")
        else:
            print("(i) track_cache_access not supported — skipping")

        if hasattr(cache, "get_prefetch_candidates"):
            candidates = cache.get_prefetch_candidates("embedding", limit=5)
            print(f"✓ Prefetch candidates: {candidates}")
        else:
            print("(i) get_prefetch_candidates not supported — skipping")

        if hasattr(cache, "evict_lru_cache_entries"):
            evicted = cache.evict_lru_cache_entries("embedding", keep_count=10000)
            print(f"✓ LRU eviction: {evicted} entries evicted")
        else:
            print("(i) evict_lru_cache_entries not supported — skipping")

        return True
    except Exception as e:
        print(f"✗ Smart caching failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_graph_analytics():
    print("\n=== Testing Graph Analytics ===")
    try:
        import networkx as nx

        from scripts.consistency_graph.advanced_analytics import (
            compute_advanced_analytics,
            compute_betweenness_centrality,
            compute_network_topology_metrics,
            compute_pagerank_influence,
            detect_communities_louvain,
            get_node_influence_rank,
        )

        G = nx.karate_club_graph()
        print(f"✓ Test graph loaded: {len(G.nodes())} nodes, {len(G.edges())} edges")

        pagerank = compute_pagerank_influence(G)
        print(f"✓ PageRank computed for {len(pagerank)} nodes")

        betweenness = compute_betweenness_centrality(G)
        print(f"✓ Betweenness centrality computed for {len(betweenness)} nodes")

        communities = detect_communities_louvain(G)
        print(f"✓ Community detection: {len(communities)} communities")

        topology = compute_network_topology_metrics(G)
        print(
            f"✓ Topology metrics: density={topology['density']:.4f}, diameter={topology['diameter']}"
        )

        analytics = compute_advanced_analytics(G)
        print("✓ Comprehensive analytics computed")
        print(f"  - Influence scores: {len(analytics['influence_scores']['pagerank'])} nodes")
        print(f"  - Communities (Louvain): {len(analytics['communities']['louvain'])}")
        print(f"  - Communities (LP): {len(analytics['communities']['label_propagation'])}")
        print(
            f"  - Top influencers: {len(analytics.get('top_influencers', {}).get('by_pagerank', []))} listed"
        )

        node_id = 0
        rank = get_node_influence_rank(analytics, node_id)
        print(
            f"✓ Node influence rank for node {node_id}: PageRank rank #{rank['pagerank']['rank']}"
        )

        return True
    except Exception as e:
        print(f"✗ Graph analytics failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_explainability():
    print("\n=== Testing Explainability ===")
    try:
        from scripts.rag.retrieve import explain_retrieval

        query = "What is multi-factor authentication?"
        chunks = [
            "MFA requires two or more verification factors...",
            "Multi-factor authentication adds security layers...",
            "Authentication factors include something you know...",
        ]
        metadata = [
            {
                "distance": 0.12,
                "retrieval_method": "vector",
                "source_category": "governance_doc",
                "doc_id": "security_policy_v2",
            },
            {
                "distance": 0.23,
                "retrieval_method": "keyword",
                "source_category": "governance_doc",
                "doc_id": "authentication_guide_v1",
            },
            {
                "distance": 0.35,
                "retrieval_method": "vector",
                "source_category": "code",
                "language": "java",
                "doc_id": "AuthService.java",
            },
        ]

        explain = explain_retrieval(query, chunks, metadata, k=3)

        print("✓ Explainability data generated:")
        print(f"  - Confidence level: {explain['confidence_level']}")
        print(f"  - Avg similarity: {explain['avg_similarity']:.1%}")
        print(f"  - Ranking explanation: {explain['ranking_explanation']}")
        print(f"  - Retrieval methods: {explain['retrieval_method']}")
        print(f"  - Similarity scores: {explain['similarity_scores']}")
        print(f"  - Source categories: {explain['metadata_insights']['source_categories']}")

        assert explain["confidence_level"] in ["high", "medium", "low", "unknown"]
        assert 0.0 <= explain["avg_similarity"] <= 1.0
        assert len(explain["similarity_scores"]) == 3

        print("✓ All assertions passed")

        return True
    except Exception as e:
        print(f"✗ Explainability failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Phase 6 Step 3: Advanced Features Verification (Manual)")
    print("=" * 60)

    results = {
        "Query Analytics": test_query_analytics(),
        "Smart Caching": test_smart_caching(),
        "Graph Analytics": test_graph_analytics(),
        "Explainability": test_explainability(),
    }

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for feature, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{feature:25} {status}")

    all_passed = all(results.values())
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
