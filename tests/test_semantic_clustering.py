"""Unit tests for semantic keyword clustering module.

Tests SQLite-backed caching, term clustering, synonym detection,
and bidirectional synonym relationships.
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from scripts.rag.semantic_clustering import (
    Cluster,
    SemanticClusterer,
    Term,
    get_semantic_clusterer,
)


@pytest.fixture
def temp_cache_db(tmp_path):
    """Create temporary SQLite cache database."""
    cache_path = tmp_path / "semantic_clustering.db"
    return cache_path


@pytest.fixture
def mock_embedder():
    """Mock Ollama embeddings API."""
    with patch("langchain_ollama.OllamaEmbeddings") as mock:
        # Create a mock embedder instance
        mock_instance = MagicMock()

        # Predefined embeddings for test terms
        embeddings_map = {
            "authentication": [0.9, 0.1, 0.0, 0.0],
            "auth": [0.85, 0.15, 0.0, 0.0],
            "login": [0.80, 0.20, 0.0, 0.0],
            "security": [0.70, 0.30, 0.0, 0.0],
            "database": [0.1, 0.9, 0.0, 0.0],
            "storage": [0.15, 0.85, 0.0, 0.0],
            "sql": [0.2, 0.8, 0.0, 0.0],
            "testing": [0.0, 0.0, 0.9, 0.1],
            "pytest": [0.0, 0.0, 0.85, 0.15],
        }

        def embed_query(text):
            # Normalise text for lookup
            normalised = text.lower().strip()
            if normalised in embeddings_map:
                return embeddings_map[normalised]
            # Random embedding for unknown terms
            return list(np.random.rand(4))

        mock_instance.embed_query.side_effect = embed_query
        mock.return_value = mock_instance
        yield mock


# ============================================================================
# Initialisation and Cache Tests
# ============================================================================


class TestSemanticClustererInit:
    """Test SemanticClusterer initialisation and database setup."""

    def test_init_creates_database(self, temp_cache_db):
        """Test that initialisation creates SQLite database."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        assert temp_cache_db.exists()
        assert clusterer.cache_path == temp_cache_db
        assert clusterer.embedding_model == "mxbai-embed-large"
        assert clusterer.similarity_threshold == 0.75

    def test_init_creates_schema(self, temp_cache_db):
        """Test that database schema is created correctly."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        conn = sqlite3.connect(str(temp_cache_db))
        cursor = conn.cursor()

        # Check table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='semantic_embeddings'
        """)
        assert cursor.fetchone() is not None

        # Check columns
        cursor.execute("PRAGMA table_info(semantic_embeddings)")
        columns = {row[1] for row in cursor.fetchall()}
        expected_columns = {
            "cache_key",
            "model",
            "term",
            "embedding",
            "created_at",
            "accessed_at",
            "access_count",
        }
        assert expected_columns.issubset(columns)

        conn.close()

    def test_init_with_custom_threshold(self, temp_cache_db):
        """Test initialisation with custom similarity threshold."""
        clusterer = SemanticClusterer(similarity_threshold=0.85, cache_path=temp_cache_db)

        assert clusterer.similarity_threshold == 0.85

    def test_default_cache_path(self):
        """Test default cache path uses rag_data."""
        with patch("scripts.rag.rag_config.RAGConfig") as mock_config:
            mock_config.return_value.rag_data_path = "/tmp/rag_data"

            clusterer = SemanticClusterer()

            assert str(clusterer.cache_path).endswith("cache/semantic_clustering.db")
            assert "/tmp/rag_data/" in str(clusterer.cache_path)


# ============================================================================
# Cache Operations Tests
# ============================================================================


class TestCacheOperations:
    """Test SQLite cache get/put operations."""

    def test_cache_miss_generates_embedding(self, temp_cache_db, mock_embedder):
        """Test that cache miss generates and stores embedding."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        embedding = clusterer._get_embedding("authentication")

        assert embedding is not None
        assert len(embedding) == 4
        assert clusterer.cache_misses == 1
        assert clusterer.cache_hits == 0

    def test_cache_hit_returns_stored_embedding(self, temp_cache_db, mock_embedder):
        """Test that second access hits cache."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        # First access - cache miss
        emb1 = clusterer._get_embedding("authentication")

        # Reset mock to ensure it's not called again
        mock_embedder.return_value.embed_query.reset_mock()

        # Second access - should hit cache
        emb2 = clusterer._get_embedding("authentication")

        assert emb1 == emb2
        assert clusterer.cache_hits == 1
        assert clusterer.cache_misses == 1
        assert not mock_embedder.return_value.embed_query.called

    def test_cache_stores_in_database(self, temp_cache_db, mock_embedder):
        """Test that embeddings are persisted to SQLite."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        embedding = clusterer._get_embedding("authentication")

        # Query database directly
        conn = sqlite3.connect(str(temp_cache_db))
        cursor = conn.cursor()
        cursor.execute("SELECT term, embedding FROM semantic_embeddings")
        result = cursor.fetchone()
        conn.close()

        assert result is not None
        assert result[0] == "authentication"
        assert json.loads(result[1]) == embedding

    def test_cache_updates_access_count(self, temp_cache_db, mock_embedder):
        """Test that cache tracks access count."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        # Access multiple times
        for _ in range(3):
            clusterer._get_embedding("authentication")

        # Check access count in DB
        conn = sqlite3.connect(str(temp_cache_db))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT access_count FROM semantic_embeddings 
            WHERE term = 'authentication'
        """)
        access_count = cursor.fetchone()[0]
        conn.close()

        assert access_count == 3

    def test_cache_key_case_insensitive(self, temp_cache_db, mock_embedder):
        """Test that cache keys are case-insensitive."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        emb1 = clusterer._get_embedding("Authentication")
        emb2 = clusterer._get_embedding("AUTHENTICATION")
        emb3 = clusterer._get_embedding("authentication")

        assert emb1 == emb2 == emb3
        # First access is miss, next two are hits
        assert clusterer.cache_misses == 1
        assert clusterer.cache_hits == 2


# ============================================================================
# Term Clustering Tests
# ============================================================================


class TestTermClustering:
    """Test semantic term clustering functionality."""

    def test_cluster_empty_list(self, temp_cache_db):
        """Test clustering empty term list."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        clusters = clusterer.cluster_terms([])

        assert clusters == []

    def test_cluster_single_term(self, temp_cache_db, mock_embedder):
        """Test clustering single term returns one cluster."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        clusters = clusterer.cluster_terms(["authentication"])

        assert len(clusters) == 1
        assert clusters[0].terms == ["authentication"]
        assert clusters[0].confidence == 1.0

    def test_cluster_similar_terms(self, temp_cache_db, mock_embedder):
        """Test that similar terms cluster together."""
        clusterer = SemanticClusterer(similarity_threshold=0.75, cache_path=temp_cache_db)

        # auth, authentication, login should cluster together (similar embeddings)
        clusters = clusterer.cluster_terms(
            [
                "authentication",
                "auth",
                "login",
                "database",  # Different cluster
            ]
        )

        # Should have 2 clusters: auth-related and database-related
        assert len(clusters) >= 1  # At least one cluster

        # Find auth cluster
        auth_cluster = None
        for cluster in clusters:
            if "authentication" in cluster.terms:
                auth_cluster = cluster
                break

        assert auth_cluster is not None
        # Auth, authentication, login should be together
        assert len(set(auth_cluster.terms) & {"authentication", "auth", "login"}) >= 2

    def test_cluster_dissimilar_terms(self, temp_cache_db, mock_embedder):
        """Test that dissimilar terms form separate clusters."""
        clusterer = SemanticClusterer(similarity_threshold=0.75, cache_path=temp_cache_db)

        # Very different terms
        clusters = clusterer.cluster_terms(
            [
                "authentication",
                "database",
                "testing",
            ]
        )

        # Should form multiple clusters
        assert len(clusters) >= 2

    def test_cluster_has_valid_structure(self, temp_cache_db, mock_embedder):
        """Test that clusters have valid structure."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        clusters = clusterer.cluster_terms(
            [
                "authentication",
                "auth",
                "database",
            ]
        )

        for cluster in clusters:
            assert isinstance(cluster, Cluster)
            assert cluster.cluster_id >= 0
            assert len(cluster.terms) > 0
            assert cluster.centroid is not None
            assert len(cluster.centroid) == 4  # 4D test embeddings
            assert 0.0 <= cluster.confidence <= 1.0

    def test_cluster_linkage_methods(self, temp_cache_db, mock_embedder):
        """Test different linkage methods."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        terms = ["authentication", "auth", "login", "database"]

        for method in ["single", "complete", "average", "ward"]:
            clusters = clusterer.cluster_terms(terms, linkage_method=method)
            assert len(clusters) >= 1

    def test_cluster_preserves_all_terms(self, temp_cache_db, mock_embedder):
        """Test that all input terms appear in clusters."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        terms = ["authentication", "auth", "database", "storage", "testing"]
        clusters = clusterer.cluster_terms(terms)

        clustered_terms = set()
        for cluster in clusters:
            clustered_terms.update(cluster.terms)

        assert clustered_terms == set(terms)


# ============================================================================
# Synonym Detection Tests
# ============================================================================


class TestSynonymDetection:
    """Test synonym finding functionality."""

    def test_find_synonyms_empty_candidates(self, temp_cache_db, mock_embedder):
        """Test finding synonyms with empty candidate list."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        synonyms = clusterer.find_synonyms("authentication", [])

        assert synonyms == []

    def test_find_synonyms_returns_similar_terms(self, temp_cache_db, mock_embedder):
        """Test that find_synonyms returns similar terms."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        synonyms = clusterer.find_synonyms(
            "authentication", ["auth", "login", "database", "storage"], top_k=3
        )

        assert len(synonyms) <= 3
        assert all(isinstance(syn, tuple) for syn in synonyms)
        assert all(len(syn) == 2 for syn in synonyms)

        # Check similarity scores
        for term, score in synonyms:
            assert isinstance(term, str)
            assert 0.0 <= score <= 1.0

    def test_find_synonyms_sorted_by_similarity(self, temp_cache_db, mock_embedder):
        """Test that synonyms are sorted by similarity descending."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        synonyms = clusterer.find_synonyms("authentication", ["auth", "login", "database"], top_k=3)

        # Scores should be descending
        scores = [score for _, score in synonyms]
        assert scores == sorted(scores, reverse=True)

    def test_find_synonyms_respects_top_k(self, temp_cache_db, mock_embedder):
        """Test that top_k parameter is respected."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        synonyms = clusterer.find_synonyms(
            "authentication", ["auth", "login", "database", "storage", "sql"], top_k=2
        )

        assert len(synonyms) == 2

    def test_detect_synonyms_bidirectional(self, temp_cache_db, mock_embedder):
        """Test bidirectional synonym detection."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        synonyms = clusterer.detect_synonyms_bidirectional(
            ["authentication", "auth", "login", "database"], similarity_threshold=0.75
        )

        # Should be dict mapping term -> set of synonyms
        assert isinstance(synonyms, dict)
        assert "authentication" in synonyms

        # Check bidirectionality: if A is synonym of B, B is synonym of A
        for term_a, syn_set in synonyms.items():
            for term_b in syn_set:
                assert term_a in synonyms[term_b]

    def test_detect_synonyms_threshold(self, temp_cache_db, mock_embedder):
        """Test that similarity threshold filters results."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        # High threshold - fewer synonyms
        strict_synonyms = clusterer.detect_synonyms_bidirectional(
            ["authentication", "auth", "login", "database"], similarity_threshold=0.95
        )

        # Low threshold - more synonyms
        loose_synonyms = clusterer.detect_synonyms_bidirectional(
            ["authentication", "auth", "login", "database"], similarity_threshold=0.50
        )

        # Lower threshold should find more or equal synonyms
        strict_count = sum(len(s) for s in strict_synonyms.values())
        loose_count = sum(len(s) for s in loose_synonyms.values())
        assert loose_count >= strict_count


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Test statistics and monitoring."""

    def test_get_statistics_initial_state(self, temp_cache_db, mock_embedder):
        """Test statistics in initial state."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        stats = clusterer.get_statistics()

        assert stats["cache_size"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["embedding_model"] == "mxbai-embed-large"

    def test_get_statistics_after_operations(self, temp_cache_db, mock_embedder):
        """Test statistics after cache operations."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        # Generate some embeddings
        clusterer._get_embedding("authentication")
        clusterer._get_embedding("auth")
        clusterer._get_embedding("authentication")  # Cache hit

        stats = clusterer.get_statistics()

        assert stats["cache_size"] == 2  # Two unique terms
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2
        assert stats["hit_rate"] == 1.0 / 3.0  # 1 hit out of 3 requests

    def test_get_statistics_shows_access_count(self, temp_cache_db, mock_embedder):
        """Test that total_accesses reflects cumulative access."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        # Access same term multiple times
        for _ in range(5):
            clusterer._get_embedding("authentication")

        stats = clusterer.get_statistics()

        assert stats["total_accesses"] >= 5


# ============================================================================
# Global Instance Tests
# ============================================================================


class TestGlobalInstance:
    """Test global clusterer instance management."""

    def test_get_semantic_clusterer_returns_instance(self):
        """Test that get_semantic_clusterer returns SemanticClusterer."""
        with patch("scripts.rag.rag_config.RAGConfig"):
            clusterer = get_semantic_clusterer()

            assert isinstance(clusterer, SemanticClusterer)

    def test_get_semantic_clusterer_singleton(self):
        """Test that get_semantic_clusterer returns same instance."""
        with patch("scripts.rag.rag_config.RAGConfig"):
            # Reset global instance
            import scripts.rag.semantic_clustering as sc_module

            sc_module._semantic_clusterer = None

            clusterer1 = get_semantic_clusterer()
            clusterer2 = get_semantic_clusterer()

            assert clusterer1 is clusterer2

    def test_get_semantic_clusterer_custom_params(self):
        """Test get_semantic_clusterer with custom parameters."""
        with patch("scripts.rag.rag_config.RAGConfig"):
            # Reset global instance
            import scripts.rag.semantic_clustering as sc_module

            sc_module._semantic_clusterer = None

            clusterer = get_semantic_clusterer(
                embedding_model="test-model", similarity_threshold=0.85
            )

            assert clusterer.embedding_model == "test-model"
            assert clusterer.similarity_threshold == 0.85


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_embedding_generation_failure(self, temp_cache_db):
        """Test handling of embedding generation failure."""
        with patch("langchain_ollama.OllamaEmbeddings") as mock:
            mock.return_value.embed_query.side_effect = Exception("API error")

            clusterer = SemanticClusterer(cache_path=temp_cache_db)
            embedding = clusterer._get_embedding("test")

            assert embedding is None

    def test_cluster_with_failed_embeddings(self, temp_cache_db):
        """Test clustering when some embeddings fail."""
        with patch("langchain_ollama.OllamaEmbeddings") as mock:

            def embed_with_failure(text):
                if text == "fail":
                    raise Exception("Embedding failed")
                return [0.1, 0.2, 0.3, 0.4]

            mock.return_value.embed_query.side_effect = embed_with_failure

            clusterer = SemanticClusterer(cache_path=temp_cache_db)
            clusters = clusterer.cluster_terms(["success1", "fail", "success2"])

            # Should cluster the two successful terms
            assert len(clusters) >= 1
            all_terms = set()
            for cluster in clusters:
                all_terms.update(cluster.terms)
            assert "fail" not in all_terms

    def test_thread_safe_caching(self, temp_cache_db, mock_embedder):
        """Test that caching is thread-safe."""
        import threading

        clusterer = SemanticClusterer(cache_path=temp_cache_db)
        errors = []

        def access_cache():
            try:
                for i in range(10):
                    clusterer._get_embedding(f"term{i % 3}")
            except Exception as e:
                errors.append(e)

        # Run multiple threads accessing cache
        threads = [threading.Thread(target=access_cache) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0  # No threading errors
        assert clusterer.cache_hits + clusterer.cache_misses > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_cluster_and_find_synonyms(self, temp_cache_db, mock_embedder):
        """Test clustering followed by synonym detection."""
        clusterer = SemanticClusterer(cache_path=temp_cache_db)

        terms = ["authentication", "auth", "login", "database", "storage"]

        # First cluster the terms
        clusters = clusterer.cluster_terms(terms)
        assert len(clusters) >= 1

        # Then find synonyms for a term
        synonyms = clusterer.find_synonyms("authentication", terms, top_k=3)
        assert len(synonyms) > 0

        # Both operations should use cache efficiently
        stats = clusterer.get_statistics()
        assert stats["cache_hits"] > 0  # Second operation used cache

    def test_persistent_cache_across_instances(self, temp_cache_db, mock_embedder):
        """Test that cache persists across SemanticClusterer instances."""
        # First instance generates embeddings
        clusterer1 = SemanticClusterer(cache_path=temp_cache_db)
        clusterer1._get_embedding("authentication")

        # Create second instance with same cache
        clusterer2 = SemanticClusterer(cache_path=temp_cache_db)

        # Reset mock to ensure it's not called
        mock_embedder.return_value.embed_query.reset_mock()

        # Should hit cache from first instance
        embedding = clusterer2._get_embedding("authentication")

        assert embedding is not None
        assert not mock_embedder.return_value.embed_query.called
        assert clusterer2.cache_hits == 1
