#!/usr/bin/env python3
"""Rebuild BM25 index from existing ChromaDB documents.

This script repopulates the BM25 index tables from documents already in ChromaDB.
Use this if the BM25 index was accidentally cleared or is out of sync.
"""

import sys
import logging
from pathlib import Path
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.db_factory import get_vector_client, get_cache_client
from scripts.search.bm25_search import BM25Search

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/bm25_rebuild.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Rebuild BM25 index from ChromaDB documents."""
    try:
        logger.info("Starting BM25 index rebuild from ChromaDB...")
        
        # Get clients
        logger.info("Connecting to ChromaDB...")
        client_class, using_sqlite = get_vector_client(prefer="chroma")
        vector_client = client_class(path=str(project_root / "rag_data" / "chromadb"))
        collection = vector_client.get_collection("governance_docs_documents")
        
        logger.info("Connecting to cache database...")
        cache_db = get_cache_client(enable_cache=True)
        
        # Get current corpus size
        existing_docs = cache_db.get_bm25_corpus_size()
        logger.info(f"Current BM25 corpus size: {existing_docs} documents")
        
        # Initialise BM25 tokeniser
        bm25 = BM25Search()
        
        # Get all documents from Chroma
        logger.info("Fetching documents from ChromaDB...")
        total_docs = collection.count()
        logger.info(f"Found {total_docs} documents in ChromaDB")
        
        if total_docs == 0:
            logger.error("No documents found in ChromaDB!")
            return
        
        # Process documents in batches
        batch_size = 500
        indexed_count = 0
        failed_count = 0
        
        logger.info(f"Processing documents in batches of {batch_size}...")
        
        offset = 0
        while offset < total_docs:
            try:
                # Fetch batch
                batch = collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"]
                )
                
                if not batch or not batch.get("documents"):
                    break
                
                # Index each document
                for i, (doc_id, text, metadata) in enumerate(zip(
                    batch["ids"],
                    batch["documents"],
                    batch["metadatas"]
                )):
                    try:
                        if not text:
                            logger.debug(f"Skipping empty document: {doc_id}")
                            continue
                        
                        # Tokenise
                        tokens = bm25.tokenise(text)
                        term_freq = Counter(tokens)
                        doc_length = len(tokens)
                        
                        # Index
                        cache_db.put_bm25_document(
                            doc_id=doc_id,
                            term_frequencies=dict(term_freq),
                            doc_length=doc_length,
                            original_text=text[:1000]  # Store first 1000 chars
                        )
                        
                        indexed_count += 1
                        
                        if indexed_count % 100 == 0:
                            logger.info(f"Indexed {indexed_count}/{total_docs} documents...")
                    
                    except Exception as e:
                        logger.warning(f"Failed to index document {doc_id}: {e}")
                        failed_count += 1
                        continue
                
                offset += batch_size
            
            except Exception as e:
                logger.error(f"Failed to process batch at offset {offset}: {e}")
                offset += batch_size
                continue
        
        logger.info(f"Document indexing complete: {indexed_count} succeeded, {failed_count} failed")
        
        # Update corpus statistics
        logger.info("Computing corpus statistics (IDF values)...")
        corpus_size = cache_db.get_bm25_corpus_size()
        
        if corpus_size > 0:
            cache_db.update_bm25_corpus_stats(corpus_size)
            avg_doc_len = cache_db.get_bm25_avg_doc_length()
            logger.info(f"✓ BM25 corpus stats updated: {corpus_size} documents, avg length {avg_doc_len:.1f} tokens")
        else:
            logger.error("No documents were indexed!")
        
        logger.info("BM25 index rebuild complete!")
    
    except Exception as e:
        logger.error(f"Fatal error during BM25 rebuild: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
