import networkx as nx
import pytest

from scripts.consistency_graph.advanced_analytics import (
    compute_advanced_analytics,
    compute_betweenness_centrality,
    compute_network_topology_metrics,
    compute_pagerank_influence,
    detect_communities_louvain,
    get_node_influence_rank,
)
from scripts.rag.retrieve import explain_retrieval


def test_graph_analytics_end_to_end():
    G = nx.karate_club_graph()

    # Individual analytics
    pr = compute_pagerank_influence(G)
    bc = compute_betweenness_centrality(G)
    comms = detect_communities_louvain(G)
    topo = compute_network_topology_metrics(G)

    assert len(pr) == len(G.nodes())
    assert len(bc) == len(G.nodes())
    assert len(comms) >= 1
    assert isinstance(topo["density"], float)

    # Comprehensive analytics and influencer rank
    analytics = compute_advanced_analytics(G)
    assert "influence_scores" in analytics
    assert "communities" in analytics

    rank0 = get_node_influence_rank(analytics, 0)
    assert rank0["pagerank"]["total_nodes"] == len(G.nodes())


def test_explainability_verification():
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
    assert explain["confidence_level"] in {"high", "medium", "low", "unknown"}
    assert 0.0 <= explain["avg_similarity"] <= 1.0
    assert len(explain["similarity_scores"]) == 3


def test_query_analytics_if_available():
    try:
        from scripts.utils.db_factory import get_cache_client
    except Exception:
        pytest.skip("cache client unavailable")

    cache = get_cache_client(enable_cache=True)
    if not cache or not hasattr(cache, "log_query_analytics"):
        pytest.skip("query analytics not supported in this environment")

    cache.log_query_analytics(
        query_text="What is MFA?",
        k_results=5,
        retrieval_time_ms=10.0,
        generation_time_ms=100.0,
        num_chunks_retrieved=5,
        avg_similarity_score=0.7,
        max_similarity_score=0.9,
        cache_hit=False,
        is_code_query=False,
        model_name="mistral",
        temperature=0.3,
        metadata={"test": True},
    )

    # These methods may not exist in all cache backends; guard accordingly
    if hasattr(cache, "query_analytics_stats"):
        stats = cache.query_analytics_stats()
        assert isinstance(stats, dict)

    if hasattr(cache, "get_recent_queries"):
        recent = cache.get_recent_queries(limit=5)
        assert isinstance(recent, list)


def test_smart_caching_if_available():
    try:
        from scripts.utils.db_factory import get_cache_client
    except Exception:
        pytest.skip("cache client unavailable")

    cache = get_cache_client(enable_cache=True)
    if not cache:
        pytest.skip("cache backend not available")

    # Access tracking/eviction APIs may not be present in all implementations
    if hasattr(cache, "track_cache_access"):
        cache.track_cache_access("embedding", "test_hash_001")
        cache.track_cache_access("embedding", "test_hash_001")
        cache.track_cache_access("embedding", "test_hash_002")

    if hasattr(cache, "get_prefetch_candidates"):
        candidates = cache.get_prefetch_candidates("embedding", limit=5)
        assert isinstance(candidates, list)

    if hasattr(cache, "evict_lru_cache_entries"):
        evicted = cache.evict_lru_cache_entries("embedding", keep_count=10000)
        assert isinstance(evicted, int)
