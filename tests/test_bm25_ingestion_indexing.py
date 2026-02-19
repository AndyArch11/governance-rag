"""Tests for BM25 ingestion-time indexing and retrieval.

Tests the complete BM25 indexing pipeline:
1. Database schema and operations (cache_db.py)
2. Ingestion-time indexing (ingest.py integration)
3. Retrieval using pre-built index (bm25_retrieval.py)
4. Parallel hybrid search (hybrid_search.py)
"""

import pytest
import tempfile
from pathlib import Path
from collections import Counter
from unittest.mock import MagicMock, patch

from scripts.ingest.cache_db import CacheDB
from scripts.search.bm25_retrieval import BM25Retriever
from scripts.search.bm25_search import BM25Search
from scripts.search.hybrid_search import HybridSearch, FusionStrategy


class TestCacheDBBM25Operations:
    """Test BM25 index storage in cache database."""
    
    def test_bm25_table_creation(self):
        """Test that BM25 tables are created on initialisation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_db = CacheDB(rag_data_path=Path(tmpdir), enable_cache=True)
            
            # Verify tables exist by attempting to query them
            with cache_db._get_cursor() as cursor:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'bm25%'")
                tables = [row['name'] for row in cursor.fetchall()]
            
            assert 'bm25_index' in tables
            assert 'bm25_corpus_stats' in tables
            assert 'bm25_doc_metadata' in tables
            
            # Cleanup
            cache_db.close()
    
    def test_put_bm25_document(self):
        """Test storing BM25 document with term frequencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_db = CacheDB(rag_data_path=Path(tmpdir), enable_cache=True)
            
            # Store a document
            term_freq = {"hello": 3, "world": 2, "python": 1}
            doc_length = 6
            
            cache_db.put_bm25_document(
                doc_id="doc1",
                term_frequencies=term_freq,
                doc_length=doc_length,
                original_text="hello world hello python hello world"
            )
            
            # Verify document metadata
            with cache_db._get_cursor() as cursor:
                cursor.execute("SELECT * FROM bm25_doc_metadata WHERE doc_id = ?", ("doc1",))
                result = cursor.fetchone()
            
            assert result is not None
            assert result['doc_length'] == 6
            assert result['original_text'] == "hello world hello python hello world"
            
            # Verify term frequencies
            doc_terms = cache_db.get_bm25_doc_terms("doc1")
            assert doc_terms == term_freq
            
            # Cleanup
            cache_db.close()
    
    def test_update_bm25_corpus_stats(self):
        """Test computation of IDF values for corpus."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_db = CacheDB(rag_data_path=Path(tmpdir), enable_cache=True)
            
            # Add multiple documents
            cache_db.put_bm25_document("doc1", {"python": 2, "code": 1}, 3)
            cache_db.put_bm25_document("doc2", {"python": 1, "java": 2}, 3)
            cache_db.put_bm25_document("doc3", {"java": 1, "rust": 3}, 4)
            
            # Update corpus stats
            cache_db.update_bm25_corpus_stats(total_docs=3)
            
            # Verify IDF values are computed
            python_stats = cache_db.get_bm25_term_stats("python")
            assert python_stats is not None
            df, idf = python_stats
            assert df == 2  # python appears in 2 documents
            assert idf > 0  # IDF should be positive
            
            java_stats = cache_db.get_bm25_term_stats("java")
            assert java_stats is not None
            df_java, idf_java = java_stats
            assert df_java == 2  # java appears in 2 documents
            
            rust_stats = cache_db.get_bm25_term_stats("rust")
            assert rust_stats is not None
            df_rust, idf_rust = rust_stats
            assert df_rust == 1  # rust appears in 1 document
            assert idf_rust > idf  # Rarer terms should have higher IDF
            
            # Cleanup
            cache_db.close()
    
    def test_get_bm25_docs_with_term(self):
        """Test retrieving documents containing a specific term."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_db = CacheDB(rag_data_path=Path(tmpdir), enable_cache=True)
            
            cache_db.put_bm25_document("doc1", {"python": 3}, 5)
            cache_db.put_bm25_document("doc2", {"python": 1, "java": 2}, 6)
            cache_db.put_bm25_document("doc3", {"java": 4}, 4)
            
            # Get documents containing "python"
            docs = cache_db.get_bm25_docs_with_term("python")
            doc_ids = [doc_id for doc_id, _, _ in docs]
            
            assert len(docs) == 2
            assert "doc1" in doc_ids
            assert "doc2" in doc_ids
            assert "doc3" not in doc_ids
            
            # Cleanup
            cache_db.close()
    
    def test_get_bm25_avg_doc_length(self):
        """Test computation of average document length."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_db = CacheDB(rag_data_path=Path(tmpdir), enable_cache=True)
            
            cache_db.put_bm25_document("doc1", {"a": 1}, 10)
            cache_db.put_bm25_document("doc2", {"b": 1}, 20)
            cache_db.put_bm25_document("doc3", {"c": 1}, 30)
            
            avg_len = cache_db.get_bm25_avg_doc_length()
            assert avg_len == 20.0  # (10 + 20 + 30) / 3
            
            # Cleanup
            cache_db.close()
    
    def test_delete_bm25_document(self):
        """Test removing a document from BM25 index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_db = CacheDB(rag_data_path=Path(tmpdir), enable_cache=True)
            
            cache_db.put_bm25_document("doc1", {"test": 1}, 5)
            cache_db.put_bm25_document("doc2", {"test": 2}, 6)
            
            assert cache_db.get_bm25_corpus_size() == 2
            
            # Delete doc1
            cache_db.delete_bm25_document("doc1")
            
            assert cache_db.get_bm25_corpus_size() == 1
            assert cache_db.get_bm25_doc_terms("doc1") == {}
            assert cache_db.get_bm25_doc_terms("doc2") != {}
            
            # Cleanup
            cache_db.close()


class TestBM25Retriever:
    """Test BM25 retrieval using pre-built index."""
    
    def test_retriever_initialisation(self):
        """Test BM25Retriever initialises correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create index
            cache_db = CacheDB(rag_data_path=Path(tmpdir), enable_cache=True)
            cache_db.put_bm25_document("doc1", {"test": 1}, 5)
            cache_db.update_bm25_corpus_stats(1)
            
            # Initialise retriever
            retriever = BM25Retriever(rag_data_path=Path(tmpdir))
            
            assert retriever.total_docs == 1
            assert retriever.avg_doc_length == 5.0
            
            # Cleanup
            cache_db.close()
            retriever.cache_db.close()
    
    def test_retriever_search_basic(self):
        """Test basic BM25 search with pre-built index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_db = CacheDB(rag_data_path=Path(tmpdir), enable_cache=True)
            
            # Build index
            cache_db.put_bm25_document("doc1", {"python": 3, "programming": 2}, 5)
            cache_db.put_bm25_document("doc2", {"python": 1, "data": 2}, 3)
            cache_db.put_bm25_document("doc3", {"java": 2, "programming": 1}, 3)
            cache_db.update_bm25_corpus_stats(3)
            
            # Search
            retriever = BM25Retriever(rag_data_path=Path(tmpdir))
            results = retriever.search("python programming", top_k=2)
            
            assert len(results) <= 2
            assert results[0][0] in ["doc1", "doc2", "doc3"]
            assert results[0][1] > 0  # Score should be positive
            
            # Cleanup
            cache_db.close()
            retriever.cache_db.close()
    
    def test_retriever_search_no_results(self):
        """Test search when query terms not in index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_db = CacheDB(rag_data_path=Path(tmpdir), enable_cache=True)
            cache_db.put_bm25_document("doc1", {"python": 1}, 1)
            cache_db.update_bm25_corpus_stats(1)
            
            retriever = BM25Retriever(rag_data_path=Path(tmpdir))
            results = retriever.search("javascript", top_k=10)
            
            assert results == []
            
            # Cleanup
            cache_db.close()
            retriever.cache_db.close()
    
    def test_retriever_search_ranking(self):
        """Test that results are ranked by BM25 score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_db = CacheDB(rag_data_path=Path(tmpdir), enable_cache=True)
            
            # doc1 has more occurrences of "python"
            cache_db.put_bm25_document("doc1", {"python": 5}, 5)
            cache_db.put_bm25_document("doc2", {"python": 2}, 5)
            cache_db.put_bm25_document("doc3", {"python": 1}, 5)
            cache_db.update_bm25_corpus_stats(3)
            
            retriever = BM25Retriever(rag_data_path=Path(tmpdir))
            results = retriever.search("python", top_k=3)
            
            # doc1 should rank highest
            assert results[0][0] == "doc1"
            
            # Scores should be descending
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True)
            
            # Cleanup
            cache_db.close()
            retriever.cache_db.close()
    
    def test_retriever_empty_index(self):
        """Test retriever with empty index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = BM25Retriever(rag_data_path=Path(tmpdir))
            
            assert retriever.total_docs == 0
            results = retriever.search("test query", top_k=10)
            assert results == []
            
            # Cleanup
            retriever.cache_db.close()


class TestHybridSearchParallel:
    """Test parallel execution in hybrid search."""
    
    def test_parallel_execution(self):
        """Test that searches execute in parallel."""
        call_times = []
        
        def slow_semantic_search(query, top_k):
            import time
            call_times.append(('semantic_start', time.time()))
            time.sleep(0.1)  # Simulate slow search
            call_times.append(('semantic_end', time.time()))
            return [("doc1", 0.9), ("doc2", 0.8)]
        
        def slow_keyword_search(query, top_k):
            import time
            call_times.append(('keyword_start', time.time()))
            time.sleep(0.1)  # Simulate slow search
            call_times.append(('keyword_end', time.time()))
            return [("doc2", 10.0), ("doc3", 8.0)]
        
        hybrid = HybridSearch(
            semantic_search_fn=slow_semantic_search,
            keyword_search_fn=slow_keyword_search,
            fusion_strategy=FusionStrategy.RRF,
        )
        
        import time
        start_time = time.time()
        results = hybrid.search("test query", top_k=3, parallel=True)
        total_time = time.time() - start_time
        
        # Parallel execution should take ~0.1s (max of both), not ~0.2s (sum)
        assert total_time < 0.15  # Allow some overhead
        
        # Both searches should have started before either ended
        semantic_start = next(t for evt, t in call_times if evt == 'semantic_start')
        keyword_start = next(t for evt, t in call_times if evt == 'keyword_start')
        semantic_end = next(t for evt, t in call_times if evt == 'semantic_end')
        keyword_end = next(t for evt, t in call_times if evt == 'keyword_end')
        
        # Verify overlap (one started before the other finished)
        assert keyword_start < semantic_end or semantic_start < keyword_end
        
        # Verify results were combined
        assert len(results) > 0
    
    def test_sequential_fallback(self):
        """Test that sequential mode still works."""
        def semantic_search(query, top_k):
            return [("doc1", 0.9)]
        
        def keyword_search(query, top_k):
            return [("doc2", 10.0)]
        
        hybrid = HybridSearch(
            semantic_search_fn=semantic_search,
            keyword_search_fn=keyword_search,
            fusion_strategy=FusionStrategy.RRF,
        )
        
        results = hybrid.search("test", top_k=5, parallel=False)
        assert len(results) > 0
    
    def test_parallel_with_error_handling(self):
        """Test that errors in one search don't block the other (parallel)."""
        def failing_semantic_search(query, top_k):
            raise Exception("Semantic search failed")
        
        def working_keyword_search(query, top_k):
            return [("doc1", 10.0), ("doc2", 8.0)]
        
        hybrid = HybridSearch(
            semantic_search_fn=failing_semantic_search,
            keyword_search_fn=working_keyword_search,
            fusion_strategy=FusionStrategy.RRF,
        )
        
        results = hybrid.search("test", top_k=5, parallel=True)
        
        # Should fall back to keyword-only results
        assert len(results) == 2
        assert results[0].doc_id in ["doc1", "doc2"]
    
    def test_parallel_both_searches_fail(self):
        """Test handling when both searches fail in parallel."""
        def failing_search(query, top_k):
            raise Exception("Search failed")
        
        hybrid = HybridSearch(
            semantic_search_fn=failing_search,
            keyword_search_fn=failing_search,
            fusion_strategy=FusionStrategy.RRF,
        )
        
        results = hybrid.search("test", top_k=5, parallel=True)
        assert results == []


class TestBM25IngestionIntegration:
    """Integration tests for BM25 indexing during ingestion."""
    
    def test_tokenisation_consistency(self):
        """Test that tokenisation is consistent between indexing and retrieval."""
        text = "Hello World! Python programming with TESTING-123"
        
        # Tokenise during indexing (BM25Search)
        bm25_search = BM25Search()
        tokens_index = bm25_search.tokenise(text)
        
        # Tokenise during retrieval (BM25Retriever)
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = BM25Retriever(rag_data_path=Path(tmpdir))
            tokens_retrieve = retriever.tokenise(text)
        
        # Should produce identical tokens
        assert tokens_index == tokens_retrieve
    
    def test_full_pipeline_simulation(self):
        """Simulate complete ingestion -> retrieval pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_db = CacheDB(rag_data_path=Path(tmpdir), enable_cache=True)
            
            # Simulate ingestion of 3 documents
            docs = [
                ("doc1", "Python programming language for data science and machine learning"),
                ("doc2", "Java programming for enterprise applications and web services"),
                ("doc3", "Python and Java comparison for backend development"),
            ]
            
            bm25 = BM25Search()
            for doc_id, text in docs:
                tokens = bm25.tokenise(text)
                term_freq = Counter(tokens)
                cache_db.put_bm25_document(doc_id, dict(term_freq), len(tokens))
            
            # Update corpus statistics
            cache_db.update_bm25_corpus_stats(len(docs))
            
            # Simulate retrieval
            retriever = BM25Retriever(rag_data_path=Path(tmpdir))
            
            # Query for Python
            results = retriever.search("python programming", top_k=2)
            result_ids = [doc_id for doc_id, _ in results]
            
            assert len(results) > 0
            assert "doc1" in result_ids or "doc3" in result_ids  # Both contain "python"
            
            # Cleanup
            cache_db.close()
            retriever.cache_db.close()
    
    def test_incremental_indexing(self):
        """Test adding documents incrementally and updating stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_db = CacheDB(rag_data_path=Path(tmpdir), enable_cache=True)
            bm25 = BM25Search()
            
            # Index first batch
            cache_db.put_bm25_document("doc1", {"test": 1}, 1)
            cache_db.update_bm25_corpus_stats(1)
            
            # Index second batch
            cache_db.put_bm25_document("doc2", {"test": 2}, 2)
            cache_db.put_bm25_document("doc3", {"test": 1}, 1)
            cache_db.update_bm25_corpus_stats(3)  # Update with new total
            
            # Verify corpus size
            assert cache_db.get_bm25_corpus_size() == 3
            
            # Verify IDF updated for new corpus size
            stats = cache_db.get_bm25_term_stats("test")
            assert stats is not None
            df, idf = stats
            assert df == 3  # "test" appears in all 3 documents
            
            # Cleanup
            cache_db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
