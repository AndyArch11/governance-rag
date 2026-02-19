"""Verify that cache.clear() methods don't affect BM25 tables."""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ingest.cache_db import CacheDB


def test_embedding_cache_clear_preserves_bm25():
    """Verify clear_embedding_cache() doesn't wipe BM25 tables."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create cache DB with both BM25 and embedding data
        cache_db = CacheDB(rag_data_path=tmpdir, enable_cache=True)
        
        # Add BM25 document
        cache_db.put_bm25_document(
            doc_id="test_doc",
            term_frequencies={"python": 3, "test": 2},
            doc_length=5,
            original_text="test document"
        )
        
        # Add embedding cache entry
        cache_db.put_embedding("test_text", [0.1, 0.2, 0.3])
        
        # Verify both exist
        assert cache_db.get_bm25_corpus_size() == 1, "BM25 document not indexed"
        embedding = cache_db.get_embedding("test_text")
        assert embedding is not None, "Embedding not stored"
        
        # Clear only embedding cache (not BM25)
        cache_db.clear_embedding_cache()
        
        # Verify BM25 data is still there
        assert cache_db.get_bm25_corpus_size() == 1, "❌ BM25 data was cleared by clear_embedding_cache()!"
        
        # Verify embedding cache is actually cleared
        embedding_after = cache_db.get_embedding("test_text")
        assert embedding_after is None, "Embedding cache wasn't cleared"
        
        print("✓ test_embedding_cache_clear_preserves_bm25 PASSED")
        cache_db.close()


def test_llm_cache_clear_preserves_bm25():
    """Verify clear_llm_cache() doesn't wipe BM25 tables."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create cache DB with both BM25 and LLM data
        cache_db = CacheDB(rag_data_path=tmpdir, enable_cache=True)
        
        # Add BM25 document
        cache_db.put_bm25_document(
            doc_id="test_doc",
            term_frequencies={"java": 2, "test": 1},
            doc_length=3,
            original_text="java test"
        )
        
        # Add LLM cache entry
        cache_db.put_llm_result("test_prompt", "test response")
        
        # Verify both exist
        assert cache_db.get_bm25_corpus_size() == 1, "BM25 document not indexed"
        llm_result = cache_db.get_llm_result("test_prompt")
        assert llm_result == "test response", "LLM result not stored"
        
        # Clear only LLM cache (not BM25)
        cache_db.clear_llm_cache()
        
        # Verify BM25 data is still there
        assert cache_db.get_bm25_corpus_size() == 1, "❌ BM25 data was cleared by clear_llm_cache()!"
        
        # Verify LLM cache is actually cleared
        llm_result_after = cache_db.get_llm_result("test_prompt")
        assert llm_result_after is None, "LLM cache wasn't cleared"
        
        print("✓ test_llm_cache_clear_preserves_bm25 PASSED")
        cache_db.close()


if __name__ == "__main__":
    try:
        test_embedding_cache_clear_preserves_bm25()
        test_llm_cache_clear_preserves_bm25()
        print("\n✓✓ All tests PASSED - BM25 data is protected from cache clearing")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
