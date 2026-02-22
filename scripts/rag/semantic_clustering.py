"""Semantic keyword clustering for intelligent term grouping.

Groups semantically related terms into clusters using embeddings-based similarity.
Enables synonym detection, term disambiguation, and improved query understanding.

Features:
- Token embedding generation (using Ollama)
- Agglomerative clustering with configurable linkage
- Bidirectional synonym detection
- Cluster visualisation/inspection
- Caching for performance

Use cases:
- Reduce redundancy in query expansion (group similar terms)
- Suggest synonyms for user queries
- Improve BM25 result organisation (group by semantic similarity)
- Quality assurance for domain vocabularies
"""

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

try:
    from scripts.ingest.vectors import EMBEDDING_MODEL_NAME
except ImportError:
    EMBEDDING_MODEL_NAME = "mxbai-embed-large"

from scripts.utils.logger import create_module_logger

get_logger, audit = create_module_logger("rag")
logger = get_logger()


@dataclass
class Term:
    """Represents a term in the vocabulary."""

    text: str
    domain: str
    frequency: int = 1
    weight: float = 1.0


@dataclass
class Cluster:
    """Represents a semantic cluster of related terms."""

    cluster_id: int
    terms: List[str]
    centroid: Optional[List[float]] = None
    label: Optional[str] = None  # Human-readable label
    confidence: float = 0.0  # Cluster cohesion score


class SemanticClusterer:
    """Groups related terms using embedding-based semantic similarity.

    Uses SQLite database for persistent caching of embeddings to avoid
    regenerating them for frequently analysed terms.

    Attributes:
        embedding_model (str): Model name for embeddings
        similarity_threshold (float): Minimum similarity for same cluster (0.0-1.0)
        cache_path (Path): Path to SQLite cache database
        cache_hits (int): Number of cache hits
        cache_misses (int): Number of cache misses
    """

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL_NAME,
        similarity_threshold: float = 0.75,
        cache_path: Optional[Path] = None,
    ):
        """Initialise semantic clusterer.

        Args:
            embedding_model: Model for generating embeddings
            similarity_threshold: Minimum cosine similarity (0.0-1.0)
            cache_path: Path to SQLite database (default: rag_data/cache/semantic_clustering.db)
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

        if cache_path is None:
            from scripts.rag.rag_config import RAGConfig

            config = RAGConfig()
            cache_path = Path(config.rag_data_path) / "cache" / "semantic_clustering.db"

        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.cache_hits = 0
        self.cache_misses = 0
        self.lock = threading.Lock()

        # Initialise database schema
        self._init_db()

    def _init_db(self) -> None:
        """Initialise SQLite database schema for embedding cache."""
        conn = sqlite3.connect(str(self.cache_path))
        cursor = conn.cursor()

        # Create embeddings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                term TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 1
            )
        """)

        # Create index on model and term for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_term 
            ON semantic_embeddings(model, term)
        """)

        conn.commit()
        conn.close()

    def _cache_key(self, text: str) -> str:
        """Generate cache key for a term."""
        return f"{self.embedding_model}:{text.lower().strip()}"

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text, using cache or generating new.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if generation fails
        """
        key = self._cache_key(text)

        # Try cache first
        with self.lock:
            conn = sqlite3.connect(str(self.cache_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT embedding FROM semantic_embeddings 
                WHERE cache_key = ?
            """,
                (key,),
            )

            result = cursor.fetchone()

            if result:
                # Update access statistics
                cursor.execute(
                    """
                    UPDATE semantic_embeddings 
                    SET accessed_at = ?, access_count = access_count + 1
                    WHERE cache_key = ?
                """,
                    (datetime.now().isoformat(), key),
                )
                conn.commit()
                conn.close()

                self.cache_hits += 1
                return json.loads(result[0])

            conn.close()
            self.cache_misses += 1

        # Generate new embedding
        try:
            from langchain_ollama import OllamaEmbeddings

            embedder = OllamaEmbeddings(model=self.embedding_model)
            embedding = embedder.embed_query(text)

            # Store in cache
            with self.lock:
                conn = sqlite3.connect(str(self.cache_path))
                cursor = conn.cursor()

                now = datetime.now().isoformat()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO semantic_embeddings 
                    (cache_key, model, term, embedding, created_at, accessed_at, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                    (key, self.embedding_model, text, json.dumps(embedding), now, now),
                )

                conn.commit()
                conn.close()

            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding for '{text}': {e}")
            return None

    def cluster_terms(
        self,
        terms: List[str],
        linkage_method: str = "average",
    ) -> List[Cluster]:
        """Cluster terms by semantic similarity.

        Args:
            terms: List of terms to cluster
            linkage_method: Linkage method for clustering
                - 'single': minimum distance (loose clusters)
                - 'complete': maximum distance (tight clusters)
                - 'average': mean distance (balanced, default)
                - 'ward': variance minimisation

        Returns:
            List of Cluster objects
        """
        if not terms:
            return []

        if len(terms) == 1:
            return [Cluster(cluster_id=0, terms=terms[:1], confidence=1.0)]

        # Generate embeddings
        embeddings = []
        valid_terms = []

        for term in terms:
            emb = self._get_embedding(term)
            if emb is not None:
                embeddings.append(emb)
                valid_terms.append(term)

        if len(embeddings) < 2:
            return [Cluster(cluster_id=0, terms=valid_terms, confidence=1.0)]

        # Compute linkage
        embeddings_array = np.array(embeddings)

        # Use cosine distance
        distances = pdist(embeddings_array, metric="cosine")
        linkage_matrix = linkage(distances, method=linkage_method)

        # Form clusters at threshold
        cluster_labels = fcluster(
            linkage_matrix, 1.0 - self.similarity_threshold, criterion="distance"
        )

        # Group terms by cluster
        clusters_dict: Dict[int, List[str]] = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(valid_terms[i])

        # Create Cluster objects
        clusters = []
        for cluster_id, cluster_terms in sorted(clusters_dict.items()):
            # Calculate centroid
            cluster_indices = [valid_terms.index(t) for t in cluster_terms]
            centroid = np.mean([embeddings_array[i] for i in cluster_indices], axis=0).tolist()

            # Calculate confidence (avg similarity within cluster)
            if len(cluster_terms) > 1:
                similarities = []
                for i, idx_i in enumerate(cluster_indices):
                    for idx_j in cluster_indices[i + 1 :]:
                        sim = 1.0 - (
                            distances[
                                int(
                                    idx_i * len(embeddings_array)
                                    - idx_i * (idx_i + 1) / 2
                                    + idx_j
                                    - idx_i
                                    - 1
                                )
                            ]
                            if idx_i < idx_j
                            else distances[
                                int(
                                    idx_j * len(embeddings_array)
                                    - idx_j * (idx_j + 1) / 2
                                    + idx_i
                                    - idx_j
                                    - 1
                                )
                            ]
                        )
                        similarities.append(sim)
                confidence = np.mean(similarities) if similarities else 0.0
            else:
                confidence = 1.0

            clusters.append(
                Cluster(
                    cluster_id=len(clusters),
                    terms=cluster_terms,
                    centroid=centroid,
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                )
            )

        return clusters

    def find_synonyms(
        self,
        term: str,
        candidate_terms: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Find synonyms for a term from candidates.

        Args:
            term: Query term
            candidate_terms: Terms to search for synonyms
            top_k: Maximum results

        Returns:
            List of (term, similarity) tuples, sorted by similarity descending
        """
        query_emb = self._get_embedding(term)
        if query_emb is None:
            return []

        query_emb = np.array(query_emb)
        similarities = []

        for candidate in candidate_terms:
            cand_emb = self._get_embedding(candidate)
            if cand_emb is None:
                continue

            cand_emb = np.array(cand_emb)
            sim = 1.0 - np.linalg.norm(query_emb - cand_emb) / np.linalg.norm(query_emb + cand_emb)
            similarities.append((candidate, float(sim)))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    def detect_synonyms_bidirectional(
        self,
        terms: List[str],
        similarity_threshold: float = 0.80,
    ) -> Dict[str, Set[str]]:
        """Find bidirectional synonyms in term list.

        A->B and B->A are both true for synonym pair.

        Args:
            terms: Terms to analyse
            similarity_threshold: Minimum similarity for synonyms

        Returns:
            Dict mapping term to set of synonym terms
        """
        synonyms: Dict[str, Set[str]] = {term: set() for term in terms}

        embeddings = {}
        for term in terms:
            emb = self._get_embedding(term)
            if emb is not None:
                embeddings[term] = np.array(emb)

        # Check bidirectional similarity
        for i, term_a in enumerate(terms):
            if term_a not in embeddings:
                continue

            emb_a = embeddings[term_a]

            for j, term_b in enumerate(terms):
                if i >= j or term_b not in embeddings:
                    continue

                emb_b = embeddings[term_b]

                # Cosine similarity
                sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))

                if sim >= similarity_threshold:
                    synonyms[term_a].add(term_b)
                    synonyms[term_b].add(term_a)

        return synonyms

    def get_statistics(self) -> Dict:
        """Get clustering statistics from SQLite cache."""
        conn = sqlite3.connect(str(self.cache_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM semantic_embeddings")
        cache_size = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*), SUM(access_count) 
            FROM semantic_embeddings 
            WHERE model = ?
        """,
            (self.embedding_model,),
        )
        model_entries, total_accesses = cursor.fetchone()

        conn.close()

        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_size": cache_size,
            "model_entries": model_entries,
            "cache_path": str(self.cache_path),
            "embedding_model": self.embedding_model,
            "similarity_threshold": self.similarity_threshold,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_accesses": total_accesses or 0,
        }


# Global instance
_semantic_clusterer: Optional[SemanticClusterer] = None


def get_semantic_clusterer(
    embedding_model: str = EMBEDDING_MODEL_NAME,
    similarity_threshold: float = 0.75,
) -> SemanticClusterer:
    """Get or create global semantic clusterer instance."""
    global _semantic_clusterer
    if _semantic_clusterer is None:
        _semantic_clusterer = SemanticClusterer(
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
        )
    return _semantic_clusterer
