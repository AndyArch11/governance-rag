"""Comprehensive test suite for Phase 4: Hybrid Search & BM25.

Tests:
- BM25 keyword search
- Text preprocessing (stemming, lemmatization)
- Hybrid search (RRF, linear combination)
- Query expansion (WordNet, custom synonyms)
- Integration tests
"""

import pytest

from scripts.search.bm25_search import BM25Config, BM25Search
from scripts.search.hybrid_search import FusionStrategy, HybridSearch
from scripts.search.query_expansion import ExpansionConfig, QueryExpander, load_domain_synonyms
from scripts.search.text_preprocessing import PreprocessingStrategy, TextPreprocessor


class TestBM25Search:
    """Test BM25 keyword search implementation."""

    def test_bm25_basic_search(self):
        """Test basic BM25 search functionality."""
        bm25 = BM25Search()

        docs = {
            "doc1": "The quick brown fox jumps over the lazy dog",
            "doc2": "The lazy dog sleeps all day",
            "doc3": "The quick fox is very fast",
        }

        bm25.add_documents_batch(docs)

        results = bm25.search("lazy dog", top_k=2)

        assert len(results) <= 2
        assert results[0][0] in ["doc1", "doc2"]  # Either doc should rank high
        assert results[0][1] > 0  # Positive score

    def test_bm25_empty_index(self):
        """Test BM25 with empty index."""
        bm25 = BM25Search()
        results = bm25.search("test query")
        assert results == []

    def test_bm25_no_matches(self):
        """Test BM25 when query has no matches."""
        bm25 = BM25Search()
        bm25.add_documents_batch({"doc1": "python programming"})

        results = bm25.search("java script")
        assert results == []

    def test_bm25_scoring_order(self):
        """Test that BM25 scores are in descending order."""
        bm25 = BM25Search()

        docs = {
            "doc1": "python",
            "doc2": "python python",
            "doc3": "python python python",
        }

        bm25.add_documents_batch(docs)
        results = bm25.search("python", top_k=3)

        # More occurrences should rank higher (with saturation)
        assert results[0][0] == "doc3"
        assert results[1][0] == "doc2"
        assert results[2][0] == "doc1"

    def test_bm25_config(self):
        """Test BM25 with custom configuration."""
        config = BM25Config(k1=2.0, b=0.5, epsilon=0.1)
        bm25 = BM25Search(config)

        assert bm25.config.k1 == 2.0
        assert bm25.config.b == 0.5
        assert bm25.config.epsilon == 0.1

    def test_bm25_stats(self):
        """Test BM25 index statistics."""
        bm25 = BM25Search()
        bm25.add_documents_batch(
            {
                "doc1": "hello world",
                "doc2": "hello python world",
            }
        )

        stats = bm25.get_stats()

        assert stats["num_documents"] == 2
        assert stats["num_unique_terms"] > 0
        assert stats["avg_doc_length"] > 0


class TestTextPreprocessing:
    """Test text preprocessing utilities."""

    def test_lowercase_strategy(self):
        """Test lowercase-only preprocessing."""
        preprocessor = TextPreprocessor(
            strategy=PreprocessingStrategy.LOWERCASE,
            remove_stopwords=False,
        )

        tokens = preprocessor.preprocess("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_stopword_removal(self):
        """Test stopword filtering."""
        preprocessor = TextPreprocessor(
            strategy=PreprocessingStrategy.LOWERCASE,
            remove_stopwords=True,
        )

        tokens = preprocessor.preprocess("the quick brown fox")
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "the" not in tokens  # Stopword removed

    def test_min_token_length(self):
        """Test minimum token length filtering."""
        preprocessor = TextPreprocessor(
            strategy=PreprocessingStrategy.LOWERCASE,
            remove_stopwords=False,
            min_token_length=4,
        )

        tokens = preprocessor.preprocess("a big elephant")
        assert "elephant" in tokens
        assert "a" not in tokens  # Too short
        assert "big" not in tokens  # Too short

    def test_porter_stemming(self):
        """Test Porter stemmer."""
        try:
            preprocessor = TextPreprocessor(strategy=PreprocessingStrategy.STEM_PORTER)
            tokens = preprocessor.preprocess("running runs ran")

            # All should stem to "run"
            assert "run" in tokens
        except RuntimeError:
            pytest.skip("NLTK not available")

    def test_lemmatization(self):
        """Test WordNet lemmatization."""
        try:
            preprocessor = TextPreprocessor(strategy=PreprocessingStrategy.LEMMATIZE)
            tokens = preprocessor.preprocess("running runs")

            # Should lemmatize to base forms
            assert len(tokens) > 0
        except RuntimeError:
            pytest.skip("NLTK not available")


class TestHybridSearch:
    """Test hybrid search functionality."""

    def test_rrf_fusion(self):
        """Test Reciprocal Rank Fusion."""

        def semantic_fn(q, k):
            return [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]

        def keyword_fn(q, k):
            return [("doc2", 10.0), ("doc3", 8.0), ("doc4", 6.0)]

        hybrid = HybridSearch(
            semantic_search_fn=semantic_fn,
            keyword_search_fn=keyword_fn,
            fusion_strategy=FusionStrategy.RRF,
        )

        results = hybrid.search("test", top_k=3)

        assert len(results) > 0
        assert all(hasattr(r, "doc_id") for r in results)
        assert all(hasattr(r, "score") for r in results)
        # RRF should boost doc2 and doc3 (appear in both lists)
        assert results[0].doc_id in ["doc2", "doc3"]

    def test_linear_fusion(self):
        """Test linear combination fusion."""

        def semantic_fn(q, k):
            return [("doc1", 0.9), ("doc2", 0.5)]

        def keyword_fn(q, k):
            return [("doc2", 10.0), ("doc3", 5.0)]

        hybrid = HybridSearch(
            semantic_search_fn=semantic_fn,
            keyword_search_fn=keyword_fn,
            fusion_strategy=FusionStrategy.LINEAR,
            alpha=0.5,  # Equal weight
        )

        results = hybrid.search("test", top_k=3)

        assert len(results) > 0
        # doc2 should rank high (appears in both)
        doc_ids = [r.doc_id for r in results]
        assert "doc2" in doc_ids

    def test_semantic_only_fallback(self):
        """Test fallback when keyword search fails."""

        def semantic_fn(q, k):
            return [("doc1", 0.9), ("doc2", 0.8)]

        def keyword_fn(q, k):
            return []  # No results

        hybrid = HybridSearch(
            semantic_search_fn=semantic_fn,
            keyword_search_fn=keyword_fn,
            fusion_strategy=FusionStrategy.RRF,
        )

        results = hybrid.search("test", top_k=2)

        assert len(results) == 2
        assert results[0].doc_id == "doc1"

    def test_keyword_only_fallback(self):
        """Test fallback when semantic search fails."""

        def semantic_fn(q, k):
            return []

        def keyword_fn(q, k):
            return [("doc1", 10.0)]

        hybrid = HybridSearch(
            semantic_search_fn=semantic_fn,
            keyword_search_fn=keyword_fn,
            fusion_strategy=FusionStrategy.RRF,
        )

        results = hybrid.search("test", top_k=1)

        assert len(results) == 1
        assert results[0].doc_id == "doc1"


class TestQueryExpansion:
    """Test query expansion functionality."""

    def test_custom_synonym_expansion(self):
        """Test expansion with custom synonyms."""
        custom_syns = {
            "fast": ["quick", "speedy"],
            "code": ["program", "script"],
        }

        config = ExpansionConfig(
            use_wordnet=False,
            custom_synonyms=custom_syns,
        )
        expander = QueryExpander(config)

        expanded = expander.expand_term("fast")

        assert "fast" in expanded
        assert "quick" in expanded
        assert "speedy" in expanded

    def test_query_expansion_preserves_original(self):
        """Test that original terms are preserved."""
        config = ExpansionConfig(use_wordnet=False, custom_synonyms={})
        expander = QueryExpander(config)

        original = "algorithm performance"
        expanded = expander.expand_query(original)

        # Original terms should be in expanded query
        assert "algorithm" in expanded
        assert "performance" in expanded

    def test_weighted_expansion(self):
        """Test weighted term expansion."""
        custom_syns = {"data": ["information", "records"]}
        config = ExpansionConfig(
            use_wordnet=False,
            custom_synonyms=custom_syns,
            original_term_weight=1.0,
            synonym_weight=0.5,
        )
        expander = QueryExpander(config)

        weighted = expander.expand_query_weighted("data")

        # Find original term weight
        term_weights = {term: weight for term, weight in weighted}
        assert term_weights["data"] == 1.0
        assert all(term_weights[t] == 0.5 for t in ["information", "records"])

    def test_domain_synonyms_loading(self):
        """Test loading domain-specific synonyms."""
        synonyms = load_domain_synonyms("technical")

        assert len(synonyms) > 0
        assert "algorithm" in synonyms
        assert "database" in synonyms

    def test_wordnet_expansion(self):
        """Test WordNet synonym expansion."""
        try:
            config = ExpansionConfig(use_wordnet=True, max_synonyms_per_term=2)
            expander = QueryExpander(config)

            expanded = expander.expand_term("fast")

            # Should include original + some synonyms
            assert len(expanded) >= 1
            assert "fast" in expanded
        except:
            pytest.skip("WordNet not available")


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_bm25_with_preprocessing(self):
        """Test BM25 with text preprocessing."""
        try:
            # Create preprocessor
            preprocessor = TextPreprocessor(
                strategy=PreprocessingStrategy.STEM_PORTER,
                remove_stopwords=True,
            )

            # Create BM25 with custom tokeniser
            bm25 = BM25Search()
            bm25.tokenise = preprocessor.preprocess

            docs = {
                "doc1": "running fast algorithms",
                "doc2": "algorithm runs quickly",
            }

            bm25.add_documents_batch(docs)
            results = bm25.search("running algorithm", top_k=2)

            # Both docs should match due to stemming
            assert len(results) == 2
        except RuntimeError:
            pytest.skip("NLTK not available")

    def test_hybrid_with_expansion(self):
        """Test hybrid search with query expansion."""

        # Simple mock searches
        def semantic_fn(q, k):
            return [("doc1", 0.9)]

        def keyword_fn(q, k):
            return [("doc2", 10.0)]

        # Create expander
        config = ExpansionConfig(
            use_wordnet=False,
            custom_synonyms={"test": ["exam", "trial"]},
        )
        expander = QueryExpander(config)

        # Expand query
        expanded_query = expander.expand_query("test")

        # Search with expanded query
        hybrid = HybridSearch(
            semantic_search_fn=semantic_fn,
            keyword_search_fn=keyword_fn,
            fusion_strategy=FusionStrategy.RRF,
        )

        results = hybrid.search(expanded_query, top_k=2)

        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
