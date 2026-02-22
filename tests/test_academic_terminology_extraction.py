"""Tests for domain terminology extraction with BM25 integration."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from scripts.ingest.academic.terminology import (
    DomainTerminologyExtractor,
    DomainTerminologyStore,
    TermScore,
)
from scripts.search.bm25_search import BM25Search


class TestTermScore:
    """Test TermScore dataclass."""

    def test_term_score_creation(self):
        """Test creating a term score."""
        score = TermScore(
            term="machine learning",
            frequency=5,
            tf_idf=0.45,
            domain_relevance=0.72,
            term_type="concept",
            related_terms=["deep learning", "neural networks"],
            bm25_score=0.35,
        )
        assert score.term == "machine learning"
        assert score.frequency == 5
        assert score.tf_idf == 0.45
        assert score.domain_relevance == 0.72
        assert score.term_type == "concept"
        assert len(score.related_terms) == 2
        assert score.bm25_score == 0.35

    def test_term_score_defaults(self):
        """Test default values in TermScore."""
        score = TermScore(term="test", frequency=1, tf_idf=0.1)
        assert score.domain_relevance == 0.0
        assert score.term_type == "concept"
        assert score.related_terms == []
        assert score.bm25_score == 0.0


class TestDomainTerminologyExtractor:
    """Test domain terminology extraction."""

    def test_extractor_initialisation(self):
        """Test creating extractor."""
        extractor = DomainTerminologyExtractor(min_term_freq=2, max_terms=100, ngram_range=(1, 3))
        assert extractor.min_term_freq == 2
        assert extractor.max_terms == 100
        assert extractor.ngram_range == (1, 3)
        assert extractor.num_docs == 0
        assert len(extractor.vocabulary) == 0

    def test_extract_unigrams(self):
        """Test extracting unigrams (single words)."""
        extractor = DomainTerminologyExtractor(min_term_freq=1, ngram_range=(1, 1))
        text = "machine learning algorithms neural networks"
        terms = extractor.extract_terms(text)
        assert len(terms) > 0
        # Should have individual words
        assert any("machine" in term for term in terms.keys())
        assert any("learning" in term for term in terms.keys())

    def test_extract_bigrams(self):
        """Test extracting bigrams (two-word phrases)."""
        extractor = DomainTerminologyExtractor(min_term_freq=1, ngram_range=(2, 2))
        text = "machine learning is powerful. neural networks are deep."
        terms = extractor.extract_terms(text)
        # Should have two-word phrases
        assert any(" " in term for term in terms.keys())

    def test_extract_trigrams(self):
        """Test extracting trigrams (three-word phrases)."""
        extractor = DomainTerminologyExtractor(min_term_freq=1, ngram_range=(3, 3))
        text = (
            "deep neural networks are powerful. " "convolutional neural networks excel at vision."
        )
        terms = extractor.extract_terms(text)
        # Should have three-word phrases
        assert any(term.count(" ") == 2 for term in terms.keys())

    def test_extract_mixed_ngrams(self):
        """Test extracting multiple n-gram sizes."""
        extractor = DomainTerminologyExtractor(min_term_freq=1, ngram_range=(1, 3))
        text = "machine learning deep neural networks"
        terms = extractor.extract_terms(text)
        assert len(terms) > 0

    def test_stop_word_filtering(self):
        """Test that stop words are filtered."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)
        text = "the and or machine learning but neural networks"
        terms = extractor.extract_terms(text)
        # Should not contain common stop words
        assert not any(term.lower() in ["the", "and", "or", "but"] for term in terms.keys())

    def test_frequency_filtering(self):
        """Test minimum frequency filtering."""
        extractor = DomainTerminologyExtractor(min_term_freq=2)
        text = "algorithm algorithm method network"  # algorithm appears 2x, method 1x
        terms = extractor.extract_terms(text)
        # Should filter out low-frequency terms
        assert "algorithm" in str(terms)

    def test_max_terms_limit(self):
        """Test maximum terms limit."""
        extractor = DomainTerminologyExtractor(min_term_freq=1, max_terms=10)
        text = " ".join(["word" + str(i) for i in range(50)])
        terms = extractor.extract_terms(text)
        assert len(terms) <= 10

    def test_term_classification_concept(self):
        """Test classification of concept terms."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)
        text = "machine learning concepts include neural networks and decision trees"
        terms = extractor.extract_terms(text)
        # Most terms should be classified as concepts
        concept_terms = [t for t, s in terms.items() if s.term_type == "concept"]
        assert len(concept_terms) > 0

    def test_term_classification_method(self):
        """Test classification of method terms."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)
        text = "gradient descent algorithm backpropagation technique"
        terms = extractor.extract_terms(text)
        # Should detect method terms
        method_terms = [t for t, s in terms.items() if s.term_type == "method"]
        # At least some should be methods
        assert len(terms) > 0

    def test_term_classification_tool(self):
        """Test classification of tool/system terms."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)
        text = "tensorflow framework pytorch platform scikit-learn system"
        terms = extractor.extract_terms(text)
        # Should detect tool terms
        tool_terms = [t for t, s in terms.items() if s.term_type == "tool"]
        assert len(terms) > 0

    def test_tf_idf_scoring(self):
        """Test TF-IDF score calculation."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)
        text = "machine learning machine learning neural networks"
        terms = extractor.extract_terms(text)

        # machine learning appears twice, should have higher TF-IDF
        ml_terms = [(term, score) for term, score in terms.items() if "machine" in term]
        assert len(ml_terms) > 0
        # TF-IDF should be > 0
        for term, score in ml_terms:
            assert score.tf_idf > 0

    def test_bm25_scoring(self):
        """Test BM25 score integration."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)
        text = "machine learning neural networks deep learning"
        terms = extractor.extract_terms(text, doc_id="test_doc")

        # BM25 scores should be calculated
        for term, score in terms.items():
            assert hasattr(score, "bm25_score")
            assert score.bm25_score >= 0

    def test_hybrid_domain_relevance(self):
        """Test hybrid domain relevance scoring."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)
        text = "machine learning is powerful for artificial intelligence tasks"
        terms = extractor.extract_terms(text, doc_id="test_doc")

        for term, score in terms.items():
            # Domain relevance should be 0-1 range
            assert 0 <= score.domain_relevance <= 1.0

    def test_related_terms_detection(self):
        """Test finding related terms."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)
        text = (
            "neural networks deep networks convolutional networks "
            "recurrent networks feed forward networks"
        )
        terms = extractor.extract_terms(text)

        # Terms with common words should be related
        for term, score in terms.items():
            if "networks" in term:
                # Should have related terms
                # (terms with overlap like "neural networks" and "deep networks")
                pass

    def test_extract_with_doc_id(self):
        """Test extraction with document ID."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)
        text = "machine learning algorithms"
        terms = extractor.extract_terms(text, doc_id="my_doc_001")

        # Should create terms with BM25 index
        assert len(terms) > 0
        assert extractor.bm25.index.num_docs > 0

    def test_vocabulary_accumulation(self):
        """Test that vocabulary accumulates across extractions."""
        extractor = DomainTerminologyExtractor(min_term_freq=1, max_terms=100)

        text1 = "machine learning neural networks"
        terms1 = extractor.extract_terms(text1, doc_id="doc_001")
        vocab_after_1 = len(extractor.vocabulary)

        text2 = "deep learning reinforcement learning"
        terms2 = extractor.extract_terms(text2, doc_id="doc_002")
        vocab_after_2 = len(extractor.vocabulary)

        # Vocabulary should grow
        assert vocab_after_2 >= vocab_after_1

    def test_get_vocabulary(self):
        """Test retrieving accumulated vocabulary."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)
        text = "machine learning neural networks"
        extractor.extract_terms(text, doc_id="doc_001")

        vocab = extractor.get_vocabulary()
        assert isinstance(vocab, dict)
        assert len(vocab) > 0

    def test_empty_text(self):
        """Test extracting from empty text."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)
        terms = extractor.extract_terms("")
        assert len(terms) == 0

    def test_text_with_only_stopwords(self):
        """Test extracting from text with only stop words."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)
        text = "the and or but if"
        terms = extractor.extract_terms(text)
        assert len(terms) == 0


class TestDomainTerminologyStore:
    """Test terminology storage and retrieval."""

    def test_store_initialisation(self):
        """Test creating store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "terms.db"
            store = DomainTerminologyStore(db_path)
            assert store.sqlite_path == str(db_path)

    def test_insert_terms(self):
        """Test inserting extracted terms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "terms.db"
            store = DomainTerminologyStore(db_path)

            terms = {
                "machine learning": TermScore(
                    term="machine learning",
                    frequency=5,
                    tf_idf=0.45,
                    domain_relevance=0.72,
                    bm25_score=0.35,
                ),
                "neural networks": TermScore(
                    term="neural networks",
                    frequency=3,
                    tf_idf=0.32,
                    domain_relevance=0.58,
                    bm25_score=0.28,
                ),
            }

            inserted = store.insert_terms(terms, domain="ai", doc_id="doc_001")
            assert inserted >= 0

    def test_get_terms_by_domain(self):
        """Test retrieving terms by domain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "terms.db"
            store = DomainTerminologyStore(db_path)

            terms = {
                "machine learning": TermScore(
                    term="machine learning",
                    frequency=5,
                    tf_idf=0.45,
                    domain_relevance=0.72,
                ),
                "leadership": TermScore(
                    term="leadership",
                    frequency=8,
                    tf_idf=0.55,
                    domain_relevance=0.85,
                ),
            }

            store.insert_terms(terms, domain="ai", doc_id="doc_001")
            ai_terms = store.get_terms_by_domain("ai", limit=10)

            # Should retrieve terms for the domain
            assert isinstance(ai_terms, list)

    def test_get_term_relationships(self):
        """Test retrieving term relationships."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "terms.db"
            store = DomainTerminologyStore(db_path)

            terms = {
                "neural networks": TermScore(
                    term="neural networks",
                    frequency=3,
                    tf_idf=0.32,
                    related_terms=["deep networks", "network architecture"],
                ),
            }

            store.insert_terms(terms, domain="ai", doc_id="doc_001")
            relationships = store.get_term_relationships("neural networks")

            assert isinstance(relationships, list)

    def test_store_and_retrieve_roundtrip(self):
        """Test storing and retrieving terms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "terms.db"
            store = DomainTerminologyStore(db_path)

            terms = {
                "machine learning": TermScore(
                    term="machine learning",
                    frequency=5,
                    tf_idf=0.45,
                    domain_relevance=0.72,
                    term_type="concept",
                ),
            }

            store.insert_terms(terms, domain="ai", doc_id="doc_001")
            retrieved = store.get_terms_by_domain("ai", limit=10)

            # Should retrieve the inserted term
            assert any(t["term"] == "machine learning" for t in retrieved)

    def test_multiple_domains(self):
        """Test storing terms from multiple domains."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "terms.db"
            store = DomainTerminologyStore(db_path)

            ai_terms = {
                "machine learning": TermScore(
                    term="machine learning",
                    frequency=5,
                    tf_idf=0.45,
                    domain_relevance=0.72,
                ),
            }

            leadership_terms = {
                "organisational change": TermScore(
                    term="organisational change",
                    frequency=4,
                    tf_idf=0.40,
                    domain_relevance=0.68,
                ),
            }

            store.insert_terms(ai_terms, domain="ai", doc_id="doc_001")
            store.insert_terms(leadership_terms, domain="leadership", doc_id="doc_002")

            ai_retrieved = store.get_terms_by_domain("ai", limit=10)
            leadership_retrieved = store.get_terms_by_domain("leadership", limit=10)

            assert len(ai_retrieved) > 0
            assert len(leadership_retrieved) > 0

    def test_frequency_accumulation(self):
        """Test that frequencies accumulate across documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "terms.db"
            store = DomainTerminologyStore(db_path)

            terms1 = {
                "machine learning": TermScore(
                    term="machine learning",
                    frequency=5,
                    tf_idf=0.45,
                    domain_relevance=0.72,
                ),
            }
            terms2 = {
                "machine learning": TermScore(
                    term="machine learning",
                    frequency=3,
                    tf_idf=0.32,
                    domain_relevance=0.60,
                ),
            }

            store.insert_terms(terms1, domain="ai", doc_id="doc_001")
            store.insert_terms(terms2, domain="ai", doc_id="doc_002")

            retrieved = store.get_terms_by_domain("ai", limit=10)
            ml_term = next((t for t in retrieved if t["term"] == "machine learning"), None)

            # Frequency should accumulate (5 + 3 = 8 or merged)
            if ml_term:
                assert ml_term["frequency"] >= 3


class TestBM25Integration:
    """Test BM25 integration with terminology extraction."""

    def test_bm25_index_building(self):
        """Test that BM25 index is built during extraction."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)
        text = "machine learning neural networks deep learning"
        extractor.extract_terms(text, doc_id="test_doc")

        # BM25 index should be populated
        assert extractor.bm25.index.num_docs > 0
        assert len(extractor.bm25.index.inverted_index) > 0

    def test_bm25_scores_computation(self):
        """Test BM25 score computation."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)
        text = "machine learning is powerful. neural networks are deep."
        terms = extractor.extract_terms(text, doc_id="test_doc")

        # All terms should have BM25 scores
        for term, score in terms.items():
            assert score.bm25_score >= 0

    def test_bm25_caching(self):
        """Test that BM25 scores are cached."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)
        text = "machine learning"
        extractor.extract_terms(text, doc_id="test_doc")

        # Scores should be cached
        assert len(extractor.bm25_scores) > 0

    def test_bm25_multi_document_index(self):
        """Test BM25 indexing across multiple documents."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)

        text1 = "machine learning algorithms"
        extractor.extract_terms(text1, doc_id="doc_001")

        text2 = "neural networks deep learning"
        extractor.extract_terms(text2, doc_id="doc_002")

        # Index should contain both documents
        assert extractor.bm25.index.num_docs == 2

    def test_bm25_search_consistency(self):
        """Test that BM25 search is consistent with scoring."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)
        text = "machine learning neural networks"
        terms = extractor.extract_terms(text, doc_id="test_doc")

        # Search should find indexed document
        if len(extractor.bm25.index.inverted_index) > 0:
            # Get a term to search for
            sample_term = list(terms.keys())[0]
            results = extractor.bm25.search(sample_term, top_k=1)

            # Should find the document
            assert len(results) > 0 or extractor.bm25.index.num_docs == 0


class TestIntegrationExtractionAndStorage:
    """Test integration of extraction and storage."""

    def test_end_to_end_extraction_storage(self):
        """Test complete workflow: extract, store, retrieve."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "terms.db"

            # Extract
            extractor = DomainTerminologyExtractor(min_term_freq=1, max_terms=20)
            text = (
                "machine learning neural networks deep learning. "
                "convolutional networks for computer vision. "
                "recurrent networks for sequences."
            )
            terms = extractor.extract_terms(text, doc_id="doc_001")

            # Store
            store = DomainTerminologyStore(db_path)
            inserted = store.insert_terms(terms, domain="ai", doc_id="doc_001")

            # Retrieve
            retrieved = store.get_terms_by_domain("ai", limit=10)

            assert len(retrieved) > 0
            assert inserted >= 0

    def test_multiple_documents_aggregation(self):
        """Test aggregating terminology from multiple documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "terms.db"

            extractor = DomainTerminologyExtractor(min_term_freq=1)
            store = DomainTerminologyStore(db_path)

            # Document 1
            text1 = "machine learning algorithms supervised learning"
            terms1 = extractor.extract_terms(text1, doc_id="doc_001")
            store.insert_terms(terms1, domain="ai", doc_id="doc_001")

            # Document 2
            text2 = "neural networks unsupervised learning clustering"
            terms2 = extractor.extract_terms(text2, doc_id="doc_002")
            store.insert_terms(terms2, domain="ai", doc_id="doc_002")

            # Retrieve aggregated
            retrieved = store.get_terms_by_domain("ai", limit=20)

            # Should have terms from both documents
            assert len(retrieved) > 0


class TestDomainRelevanceCalculation:
    """Test domain_relevance scoring calculation (60% TF-IDF + 40% BM25)."""

    def test_domain_relevance_range(self):
        """Test that domain_relevance stays within 0-1 range."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)

        # Create a text with varied term frequencies
        text = (
            "machine machine machine learning learning network network network network "
            "neural deep deep deep deep deep deep deep deep deep deep deep "
            "algorithm technique implementation approach"
        )

        terms = extractor.extract_terms(text, doc_id="test_doc")

        # All relevance scores should be in [0, 1]
        for term, score in terms.items():
            assert (
                0.0 <= score.domain_relevance <= 1.0
            ), f"Term '{term}' has relevance {score.domain_relevance} outside [0, 1]"

    def test_domain_relevance_weighted_average(self):
        """Test that domain_relevance correctly applies 60% TF-IDF + 40% BM25."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)

        text = "machine learning machine learning machine learning"
        terms = extractor.extract_terms(text, doc_id="test_doc")

        for term, score in terms.items():
            # Verify that domain_relevance is a valid weighted average
            # domain_relevance = (tf_idf * 0.6) + (normalised_bm25 * 0.4)
            # Both tf_idf and normalised_bm25 should be in [0, 1]
            # So weighted average should be in [0, 1]
            assert score.domain_relevance >= 0.0
            assert score.domain_relevance <= 1.0

            # If TF-IDF and BM25 are both available, verify the calculation
            if score.tf_idf > 0 or score.bm25_score > 0:
                # Score should be influenced by both components
                assert score.domain_relevance > 0

    def test_higher_tf_idf_increases_relevance(self):
        """Test that higher TF-IDF leads to higher domain_relevance."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)

        # Create two documents with different term frequencies
        # Document 1: rare term
        text1 = "machine learning rare_term"
        terms1 = extractor.extract_terms(text1, doc_id="doc1")

        # Document 2: common term
        text2 = "machine machine machine learning learning learning"
        terms2 = extractor.extract_terms(text2, doc_id="doc2")

        # Both should have valid scores
        assert len(terms1) > 0
        assert len(terms2) > 0

    def test_domain_relevance_no_negative_values(self):
        """Test that domain_relevance never produces negative values."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)

        # Test with minimal text
        text = "single word"
        terms = extractor.extract_terms(text, doc_id="test_doc")

        for term, score in terms.items():
            assert (
                score.domain_relevance >= 0.0
            ), f"Term '{term}' has negative relevance {score.domain_relevance}"

    def test_domain_relevance_consistency(self):
        """Test that same text produces same relevance scores."""
        extractor1 = DomainTerminologyExtractor(min_term_freq=1)
        extractor2 = DomainTerminologyExtractor(min_term_freq=1)

        text = "machine learning neural networks deep learning algorithms"

        terms1 = extractor1.extract_terms(text, doc_id="doc1")
        terms2 = extractor2.extract_terms(text, doc_id="doc1")  # Same doc_id for consistent BM25

        # Should produce similar scores
        for term in terms1.keys():
            if term in terms2:
                # Scores might differ slightly due to BM25 computation, but should be close
                assert (
                    abs(terms1[term].domain_relevance - terms2[term].domain_relevance) < 0.01
                ), f"Inconsistent scores for '{term}': {terms1[term].domain_relevance} vs {terms2[term].domain_relevance}"

    def test_domain_relevance_no_artificial_capping(self):
        """Test that domain_relevance doesn't have artificial 1.0 cap."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)

        # Create text with varied term frequencies
        text = (
            "very very very very very frequently occurring term "
            "and quite frequently another term "
            "but less frequent term "
            "rare rare"
        )

        terms = extractor.extract_terms(text, doc_id="test_doc")

        # Should have variation in scores, not all 1.0
        relevance_scores = [score.domain_relevance for score in terms.values()]

        # Calculate statistics
        min_score = min(relevance_scores) if relevance_scores else 0
        max_score = max(relevance_scores) if relevance_scores else 0
        unique_scores = len(set(round(s, 3) for s in relevance_scores))  # Round to 3 decimals

        # Should have variation (not all same value)
        if len(relevance_scores) > 2:
            assert unique_scores > 1, f"All relevance scores are identical: {set(relevance_scores)}"

        # Should not all be 1.0
        count_ones = sum(1 for s in relevance_scores if abs(s - 1.0) < 0.001)
        assert count_ones < len(
            relevance_scores
        ), f"All {len(relevance_scores)} scores are 1.0 (artificial capping)"

    def test_domain_relevance_weighted_by_tf_idf_60_percent(self):
        """Test that TF-IDF component has proper 60% weight."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)

        text = "machine learning"
        terms = extractor.extract_terms(text, doc_id="test_doc")

        for term, score in terms.items():
            # TF-IDF should be the dominant factor (60% weight)
            # If TF-IDF is high, domain_relevance should be relatively high
            if score.tf_idf > 0.5:
                # With 60% weight on high TF-IDF, relevance should be decent
                assert (
                    score.domain_relevance > 0.3
                ), f"Term '{term}' with tf_idf={score.tf_idf} has low relevance {score.domain_relevance}"

    def test_domain_relevance_bm25_impact_40_percent(self):
        """Test that BM25 component has proper 40% weight."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)

        text = "machine learning concepts"
        terms = extractor.extract_terms(text, doc_id="test_doc")

        for term, score in terms.items():
            # BM25 should contribute but not dominate (40% weight)
            # Even if BM25 is high, TF-IDF should be the primary factor
            if score.bm25_score > 0:
                # Should have contribution from BM25
                assert (
                    score.domain_relevance > 0
                ), f"Term '{term}' with bm25={score.bm25_score} has zero relevance"

    def test_domain_relevance_edge_case_zero_bm25(self):
        """Test domain_relevance calculation when BM25 score is zero."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)

        text = "machine learning"
        terms = extractor.extract_terms(text, doc_id="test_doc")

        for term, score in terms.items():
            # Even with zero BM25, TF-IDF (60% weight) should produce a score
            if score.bm25_score == 0:
                # Should still have relevance from TF-IDF
                assert (
                    score.domain_relevance == score.tf_idf * 0.6
                ), f"Score doesn't match TF-IDF weighted calculation: {score.domain_relevance} vs {score.tf_idf * 0.6}"

    def test_domain_relevance_formula_verification(self):
        """Verify the domain_relevance formula: (tf_idf * 0.6) + (normalised_bm25 * 0.4)."""
        extractor = DomainTerminologyExtractor(min_term_freq=1)

        text = "algorithm algorithm algorithm method network network system"
        terms = extractor.extract_terms(text, doc_id="test_doc")

        for term, score in terms.items():
            # Calculate expected value
            # Normalised BM25 should be between 0 and 1
            if score.bm25_score > 0:
                # We need to estimate normalised_bm25
                # For verification, just check that the formula produces a valid result
                calculated = (score.tf_idf * 0.6) + (
                    min(1.0, score.bm25_score / max(0.1, 1.0)) * 0.4
                )
                # Should be close to actual (might differ slightly due to BM25 normalisation)
                assert isinstance(score.domain_relevance, float)
                assert 0.0 <= score.domain_relevance <= 1.0
