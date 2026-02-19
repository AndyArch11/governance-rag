"""Tests for hybrid search enhancements (Phase 7).

Tests configurable weights, query expansion with spelling variants,
and weight management functionality.
"""

import pytest
from pathlib import Path
from typing import Dict, List

from scripts.rag.hybrid_search_weights import (
    HybridSearchWeights,
    HybridSearchWeightManager,
)
from scripts.rag.query_expansion import (
    QueryExpander,
    QueryExpansionCache,
    SPELLING_VARIANTS,
    DOMAIN_SYNONYMS,
)


class TestHybridSearchWeights:
    """Test weight configuration and management."""
    
    def test_weights_normalisation(self):
        """Test that weights are normalised to sum to 1.0."""
        weights = HybridSearchWeights(
            vector_weight=2.0,
            keyword_weight=1.0,
            normalise_weights=True
        )
        assert abs(weights.vector_weight - 2/3) < 0.001
        assert abs(weights.keyword_weight - 1/3) < 0.001
        assert abs(weights.vector_weight + weights.keyword_weight - 1.0) < 0.001
    
    def test_weights_no_normalisation(self):
        """Test that weights can be stored without normalisation."""
        weights = HybridSearchWeights(
            vector_weight=0.7,
            keyword_weight=0.5,
            normalise_weights=False
        )
        assert weights.vector_weight == 0.7
        assert weights.keyword_weight == 0.5
    
    def test_invalid_strategy(self):
        """Test that invalid combination strategy raises error."""
        with pytest.raises(ValueError):
            HybridSearchWeights(combination_strategy="invalid")
    
    def test_negative_weights_error(self):
        """Test that negative weights raise error."""
        with pytest.raises(ValueError):
            HybridSearchWeights(vector_weight=-0.1)
    
    def test_weights_serialisation(self):
        """Test weights can be serialised and deserialised."""
        original = HybridSearchWeights(
            vector_weight=0.6,
            keyword_weight=0.4,
            combination_strategy="rank_fusion"
        )
        
        data = original.to_dict()
        restored = HybridSearchWeights.from_dict(data)
        
        assert abs(restored.vector_weight - original.vector_weight) < 0.001
        assert abs(restored.keyword_weight - original.keyword_weight) < 0.001
        assert restored.combination_strategy == original.combination_strategy


class TestWeightedCombination:
    """Test different result combination strategies."""
    
    def test_weighted_sum_combination(self):
        """Test weighted sum strategy combines results correctly."""
        manager = HybridSearchWeightManager(config_path=Path("/tmp/test_weights.json"))
        manager.weights.combination_strategy = "sum"
        manager.weights.vector_weight = 0.6
        manager.weights.keyword_weight = 0.4
        
        vector_chunks = ["chunk1", "chunk2"]
        vector_metadata = [{"id": "1"}, {"id": "2"}]
        vector_scores = [0.9, 0.7]
        
        keyword_chunks = ["chunk2", "chunk3"]
        keyword_metadata = [{"id": "2"}, {"id": "3"}]
        keyword_scores = [0.8, 0.6]
        
        combined_chunks, combined_meta, combined_scores = manager.combine_results(
            vector_chunks, vector_metadata, vector_scores,
            keyword_chunks, keyword_metadata, keyword_scores,
            k=3
        )
        
        assert len(combined_chunks) <= 3
        assert "chunk1" in combined_chunks
        assert "chunk3" in combined_chunks
        
        # chunk2 should have combined score from both methods
        chunk2_score = [s for c, s in zip(combined_chunks, combined_scores) if c == "chunk2"]
        assert len(chunk2_score) > 0
        assert chunk2_score[0] > 0.5 * 0.6  # At least vector weight
    
    def test_rank_fusion_combination(self):
        """Test RRF strategy for combining results."""
        manager = HybridSearchWeightManager(config_path=Path("/tmp/test_weights_rrf.json"))
        manager.weights.combination_strategy = "rank_fusion"
        manager.weights.vector_weight = 0.5
        manager.weights.keyword_weight = 0.5
        
        vector_chunks = ["chunk1", "chunk2", "chunk3"]
        vector_metadata = [{"id": str(i)} for i in range(1, 4)]
        vector_scores = [1.0, 0.5, 0.3]
        
        keyword_chunks = ["chunk4", "chunk5"]
        keyword_metadata = [{"id": "4"}, {"id": "5"}]
        keyword_scores = [0.9, 0.7]
        
        combined_chunks, _, _ = manager.combine_results(
            vector_chunks, vector_metadata, vector_scores,
            keyword_chunks, keyword_metadata, keyword_scores,
            k=4
        )
        
        assert len(combined_chunks) <= 4
        # First items should have better RRF scores
        assert combined_chunks[0] in ["chunk1", "chunk4"]
    
    def test_top_k_combination(self):
        """Test top-k strategy."""
        manager = HybridSearchWeightManager(config_path=Path("/tmp/test_weights_topk.json"))
        manager.weights.combination_strategy = "top_k"
        manager.weights.vector_weight = 0.7
        manager.weights.keyword_weight = 0.3
        
        vector_chunks = ["v1", "v2", "v3"]
        vector_metadata = [{"id": i} for i in range(3)]
        vector_scores = [0.9, 0.8, 0.7]
        
        keyword_chunks = ["k1", "k2"]
        keyword_metadata = [{"id": i} for i in range(2)]
        keyword_scores = [0.95, 0.85]
        
        combined_chunks, _, _ = manager.combine_results(
            vector_chunks, vector_metadata, vector_scores,
            keyword_chunks, keyword_metadata, keyword_scores,
            k=5
        )
        
        # Should take ~70% from vector (2-3 chunks) and ~30% from keyword (1 chunk)
        vector_in_result = sum(1 for c in combined_chunks if c.startswith("v"))
        keyword_in_result = sum(1 for c in combined_chunks if c.startswith("k"))
        
        assert vector_in_result >= 2
        assert keyword_in_result >= 1


class TestQueryExpansion:
    """Test query expansion with spelling variants and synonyms."""
    
    def test_us_british_spelling_variants(self):
        """Test US/British spelling variant expansion."""
        expander = QueryExpander()
        
        # US to British
        variants = expander.expand_query("optimize", include_variants=True, include_synonyms=False)
        assert "optimise" in variants
        
        # British to US
        variants = expander.expand_query("optimise", include_variants=True, include_synonyms=False)
        assert "optimize" in variants
    
    def test_color_colour_variant(self):
        """Test color/colour variant."""
        expander = QueryExpander()
        
        variants = expander.expand_query("color", include_variants=True, include_synonyms=False)
        assert "colour" in variants
    
    def test_domain_synonyms_expansion(self):
        """Test domain synonym expansion."""
        expander = QueryExpander()
        
        expanded = expander.expand_query("auth", include_variants=False, include_synonyms=True)
        assert "authentication" in expanded
        assert "authorize" in expanded
        assert "credential" in expanded
    
    def test_abbreviation_expansion(self):
        """Test abbreviation expansion."""
        expander = QueryExpander()
        
        expanded = expander.expand_query("mfa", include_variants=False, include_synonyms=True)
        assert "multi" in expanded
        assert "factor" in expanded
        assert "authentication" in expanded
    
    def test_query_expansion_combined(self):
        """Test combined expansion (variants + synonyms + abbreviations)."""
        expander = QueryExpander()
        
        expanded = expander.expand_query("authentication", include_variants=True, include_synonyms=True)
        
        # Should include original
        assert "authentication" in expanded
        # Should include synonyms
        assert "login" in expanded
        # May include variants
        assert len(expanded) > 1
    
    def test_query_expansion_string(self):
        """Test expansion to combined search string."""
        expander = QueryExpander()
        
        expanded_str = expander.expand_query_string("auth")
        assert "auth" in expanded_str
        assert "authentication" in expanded_str
        # Should be space-separated
        assert " " in expanded_str
    
    def test_spelling_variant_getter(self):
        """Test getting specific spelling variant."""
        expander = QueryExpander()
        
        assert expander.get_spelling_variants("color") == "colour"
        assert expander.get_spelling_variants("colour") == "color"
        assert expander.get_spelling_variants("unknown") is None
    
    def test_synonyms_getter(self):
        """Test getting synonyms for a term."""
        expander = QueryExpander()
        
        synonyms = expander.get_synonyms("auth")
        assert "authentication" in synonyms
        assert "login" in synonyms
        
        assert expander.get_synonyms("unknown") == []
    
    def test_abbreviation_expansion_getter(self):
        """Test getting abbreviation expansion."""
        expander = QueryExpander()
        
        assert "multi" in expander.get_abbreviation_expansion("mfa")
        assert "factor" in expander.get_abbreviation_expansion("mfa")
        assert expander.get_abbreviation_expansion("unknown") is None


class TestQueryExpansionCache:
    """Test query expansion caching."""
    
    def test_expansion_cache_hit(self):
        """Test cache hit for repeated queries."""
        cache = QueryExpansionCache(max_entries=10)
        
        # First call
        expanded1 = cache.get_expanded("authentication")
        # Second call (should be cached)
        expanded2 = cache.get_expanded("authentication")
        
        assert expanded1 == expanded2
        assert "authentication" in expanded1
    
    def test_expansion_cache_capacity(self):
        """Test cache respects max_entries limit."""
        cache = QueryExpansionCache(max_entries=3)
        
        for i in range(5):
            cache.get_expanded(f"query{i}")
        
        assert len(cache.cache) <= 3
    
    def test_expansion_cache_clear(self):
        """Test cache can be cleared."""
        cache = QueryExpansionCache()
        
        cache.get_expanded("test")
        assert len(cache.cache) > 0
        
        cache.clear()
        assert len(cache.cache) == 0


class TestSpellingVariantsCompleteness:
    """Test that spelling variants are bidirectional and complete."""
    
    def test_us_british_symmetry(self):
        """Test that US/British variants are symmetric."""
        for us, variant_data in SPELLING_VARIANTS.items():
            # Extract the variant string (handle both string and dict formats)
            if isinstance(variant_data, dict):
                british = variant_data.get("variant")
            else:
                british = variant_data
            
            # If A->B exists, then B->A should exist
            if us == british:
                continue
            
            # Check reverse mapping exists
            assert british in SPELLING_VARIANTS, f"Missing reverse mapping for {british}"
    
    def test_common_spellings_covered(self):
        """Test that common US/British spelling differences are covered."""
        expected_pairs = [
            ("optimize", "optimise"),
            ("color", "colour"),
            ("authorize", "authorise"),
            ("organize", "organise"),
        ]
        
        for us, british in expected_pairs:
            # Extract variant string from dict if needed
            us_variant = SPELLING_VARIANTS.get(us)
            if isinstance(us_variant, dict):
                us_variant = us_variant.get("variant")
            
            british_variant = SPELLING_VARIANTS.get(british)
            if isinstance(british_variant, dict):
                british_variant = british_variant.get("variant")
            
            assert us_variant == british, f"Missing {us} -> {british}"
            assert british_variant == us, f"Missing reverse {british} -> {us}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
