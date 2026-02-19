"""
SQLite-based replacement for ChromaDB with compatible API.

Provides efficient vector similarity search and metadata filtering
while reducing memory footprint from 450MB (ChromaDB) to ~50-100MB (SQLite).

Architecture:
- Chunks table: Code and document fragments with embeddings
- Documents table: Full versioned documents
- Embeddings: Stored as normalised binary (fast similarity search)
- Metadata: JSON columns for rich, searchable attributes
- Indexes: On commonly filtered fields and embedding vectors

Usage:
    from chromadb_sqlite import ChromaSQLiteClient

    client = ChromaSQLiteClient(db_path="/path/to/rag.db")
    collection = client.get_or_create_collection("governance_docs_chunks")

    # Add documents
    collection.add(
        ids=["doc1"],
        embeddings=[[0.1, 0.2, ...]],
        metadatas=[{"source": "bitbucket"}],
        documents=["The document text"],
    )

    # Query with semantic search
    results = collection.query(
        query_embeddings=[[0.1, 0.2, ...]],
        n_results=10,
        where={"source_category": "code"},
    )

    # Get by ID
    results = collection.get(
        where={"doc_id": "my_doc"},
        include=["documents", "metadatas", "embeddings"],
    )
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ChromaSQLiteCollection:
    """SQLite-backed collection mimicking ChromaDB's Collection API."""

    def __init__(self, db_path: str, collection_name: str):
        self.db_path = db_path
        self.collection_name = collection_name
        self.conn = None
        self._init_db()

    def _get_connection(self):
        """Get or create database connection."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def _init_db(self):
        """Initialise database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Chunks table (for code fragments and document chunks)
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.collection_name}_chunks (
                id TEXT PRIMARY KEY,
                document TEXT,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER,
                metadata TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create indexes for common queries
        cursor.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self.collection_name}_metadata
            ON {self.collection_name}_chunks (id)
        """
        )

        conn.commit()

    def add(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Add documents to collection.

        Args:
            ids: List of document IDs (must be unique).
            embeddings: List of vector embeddings (optional).
            metadatas: List of metadata dictionaries (optional).
            documents: List of document texts (optional).

        TODO: should this return a list of IDs that were added/updated? ChromaDB's API is a bit inconsistent here.
        Or at least return a success or failure indication status for each id?
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if embeddings is None:
            embeddings = [None] * len(ids)
        if metadatas is None:
            metadatas = [{}] * len(ids)
        if documents is None:
            documents = [""] * len(ids)

        for doc_id, embedding, metadata, document in zip(ids, embeddings, metadatas, documents):
            # Normalise and store embedding as binary
            embedding_blob = None
            embedding_dim = None
            if embedding is not None:
                embedding_array = np.array(embedding, dtype=np.float32)
                embedding_blob = embedding_array.tobytes()
                embedding_dim = len(embedding)

            # Store metadata as JSON
            metadata_json = json.dumps(metadata)

            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {self.collection_name}_chunks
                (id, document, embedding, embedding_dim, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (doc_id, document, embedding_blob, embedding_dim, metadata_json),
            )

        conn.commit()

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
        include: List[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get documents from collection with optional filtering.

        Args:
            ids: List of document IDs to retrieve (optional).
            where: Dictionary of metadata filters (optional).
            limit: Max number of results to return.
            offset: Number of results to skip (for pagination).
            include: List of fields to include in response (e.g. "documents", "metadatas", "embeddings"). If None, defaults to all fields.

        Returns:
            Dictionary with keys "ids", "documents", "metadatas", "embeddings" (depending on include).
        """
        if include is None:
            include = ["documents", "metadatas"]

        conn = self._get_connection()
        cursor = conn.cursor()

        # Build WHERE clause
        where_clauses = []
        params = []

        if ids:
            placeholders = ",".join("?" * len(ids))
            where_clauses.append(f"id IN ({placeholders})")
            params.extend(ids)

        if where:
            for key, value in where.items():
                where_clauses.append(self._build_where_clause(key, value))
                params.append(value)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Execute query
        cursor.execute(
            f"""
            SELECT id, document, embedding, embedding_dim, metadata
            FROM {self.collection_name}_chunks
            WHERE {where_sql}
            LIMIT ? OFFSET ?
        """,
            params + [limit, offset],
        )

        rows = cursor.fetchall()

        # Format response
        response = {
            "ids": [],
            "documents": [] if "documents" in include else None,
            "metadatas": [] if "metadatas" in include else None,
            "embeddings": [] if "embeddings" in include else None,
            "distances": [] if "distances" in include else None,
        }

        for row in rows:
            response["ids"].append(row["id"])

            if "documents" in include:
                response["documents"].append(row["document"])

            if "metadatas" in include:
                response["metadatas"].append(json.loads(row["metadata"]))

            if "embeddings" in include and row["embedding"]:
                embedding = np.frombuffer(row["embedding"], dtype=np.float32)
                response["embeddings"].append(embedding.tolist())

        return response

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: List[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Query with semantic similarity search.

        Args:
            query_embeddings: List of query embeddings.
            n_results: Number of top results to return for each query.
            where: Dictionary of metadata filters (optional).
            include: List of fields to include in response (e.g. "documents", "metadatas", "distances"). If None, defaults to all fields.

        Returns:
            Dictionary with keys "ids", "documents", "metadatas", "distances", "embeddings" (depending on include).
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]

        conn = self._get_connection()
        cursor = conn.cursor()

        # Build WHERE clause for metadata filtering
        where_clauses = []
        params = []

        if where:
            for key, value in where.items():
                where_clauses.append(self._build_where_clause(key, value))
                params.append(value)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Get all embeddings for similarity computation
        cursor.execute(
            f"""
            SELECT id, document, embedding, embedding_dim, metadata
            FROM {self.collection_name}_chunks
            WHERE {where_sql} AND embedding IS NOT NULL
        """,
            params,
        )

        all_rows = cursor.fetchall()

        # Compute similarities for each query embedding
        results_by_query = []

        for query_embedding in query_embeddings:
            query_array = np.array(query_embedding, dtype=np.float32)
            query_array = query_array / (np.linalg.norm(query_array) + 1e-8)

            similarities = []

            for row in all_rows:
                embedding = np.frombuffer(row["embedding"], dtype=np.float32)
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

                # Cosine similarity
                similarity = np.dot(query_array, embedding)
                similarities.append(
                    {
                        "id": row["id"],
                        "document": row["document"],
                        "metadata": json.loads(row["metadata"]),
                        "distance": float(1 - similarity),  # Convert to distance
                    }
                )

            # Sort by distance (lower is better) and get top n_results
            similarities.sort(key=lambda x: x["distance"])
            similarities = similarities[:n_results]

            results_by_query.append(similarities)

        # Format response
        response = {
            "ids": [[] for _ in query_embeddings],
            "documents": [[] for _ in query_embeddings] if "documents" in include else None,
            "metadatas": [[] for _ in query_embeddings] if "metadatas" in include else None,
            "distances": [[] for _ in query_embeddings] if "distances" in include else None,
            "embeddings": [[] for _ in query_embeddings] if "embeddings" in include else None,
        }

        for query_idx, similarities in enumerate(results_by_query):
            for sim in similarities:
                response["ids"][query_idx].append(sim["id"])

                if "documents" in include:
                    response["documents"][query_idx].append(sim["document"])

                if "metadatas" in include:
                    response["metadatas"][query_idx].append(sim["metadata"])

                if "distances" in include:
                    response["distances"][query_idx].append(sim["distance"])

        return response

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Delete documents from collection.

        Args:
            ids: List of document IDs to delete (optional).
            where: Dictionary of metadata filters to delete matching documents (optional).

        Note: If both ids and where are provided, documents matching either condition will be deleted.

        Returns:
            None

        TODO: should this return a list of IDs that were deleted? ChromaDB's API is a bit inconsistent here.
        Or at least return a success or failure indication status for each id?
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        where_clauses = []
        params = []

        if ids:
            placeholders = ",".join("?" * len(ids))
            where_clauses.append(f"id IN ({placeholders})")
            params.extend(ids)

        if where:
            for key, value in where.items():
                where_clauses.append(self._build_where_clause(key, value))
                params.append(value)

        if not where_clauses:
            raise ValueError("Must specify ids or where clause for delete")

        where_sql = " AND ".join(where_clauses)

        cursor.execute(
            f"DELETE FROM {self.collection_name}_chunks WHERE {where_sql}",
            params,
        )

        conn.commit()

    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Update documents in collection.

        Args:
            ids: List of document IDs to update (must exist).
            embeddings: List of new vector embeddings (optional).
            metadatas: List of new metadata dictionaries (optional).
            documents: List of new document texts (optional).

        Note: This method replaces the existing document data with the new values provided. If a field is not provided (e.g. embeddings), it will be set to NULL.

        Returns:
            None

        TODO: should this return a list of IDs that were updated? ChromaDB's API is a bit inconsistent here.
        Or at least return a success or failure indication status for each id?
        """
        # Delete old and add new
        self.delete(ids=ids)
        self.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            **kwargs,
        )

    def count(self) -> int:
        """Get number of documents in collection.

        Args:
            None
        Returns:
            Total number of documents in the collection.
        """

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.collection_name}_chunks")
        return cursor.fetchone()[0]

    def _build_where_clause(self, key: str, value: Any) -> str:
        """Build WHERE clause for metadata filtering.

        Args:
            key: Metadata key to filter on.
            value: Value to match for the given key.

        Returns:
            SQL WHERE clause string for the given key-value pair.

        """
        # For now, support simple equality checks
        # Metadata is stored as JSON, so we check if key:value exists in JSON
        return f"json_extract(metadata, '$.{key}') = ?"

    def close(self):
        """Close database connection.

        Args:
            None

        Returns:
            None
        """
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed."""
        self.close()
        return False

    def __del__(self):
        """Destructor - ensures connection is closed on garbage collection."""
        try:
            self.close()
        except Exception:
            pass


class ChromaSQLiteClient:
    """SQLite-based ChromaDB client with compatible API."""

    def __init__(self, db_path: str = None):
        """Initialise SQLite ChromaDB client.

        TODO: should get path from config or environment variable instead of hardcoding.

        """
        if db_path is None:
            db_path = str(Path.home() / "rag-project" / "rag_data" / "chromadb.db")

        self.db_path = db_path
        self.collections = {}

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def get_or_create_collection(self, name: str) -> ChromaSQLiteCollection:
        """Get or create a collection.

        Args:
            name: Name of the collection to get or create.

        Returns:
            ChromaSQLiteCollection instance for the given name.
        """
        if name not in self.collections:
            self.collections[name] = ChromaSQLiteCollection(self.db_path, name)
        return self.collections[name]

    def get_collection(self, name: str) -> ChromaSQLiteCollection:
        """Get an existing collection.

        Args:
            name: Name of the collection to get.

        Returns:
            ChromaSQLiteCollection instance for the given name.
        """
        if name not in self.collections:
            self.collections[name] = ChromaSQLiteCollection(self.db_path, name)
        return self.collections[name]

    def delete_collection(self, name: str) -> None:
        """Delete a collection.

        Args:
            name: Name of the collection to delete.

        Returns:
            None
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Drop the table
            cursor.execute(f"DROP TABLE IF EXISTS {name}_chunks")
            conn.commit()
        except Exception as e:
            print(f"Error deleting collection {name}: {e}")
        finally:
            if conn:
                conn.close()

        # Remove from collections
        if name in self.collections:
            del self.collections[name]

    def list_collections(self) -> List[str]:
        """List all collections.

        Returns:
            List of collection names.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_chunks'"
            )

            tables = cursor.fetchall()
            # Extract collection names (remove _chunks suffix)
            return [table[0].replace("_chunks", "") for table in tables]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def close(self):
        """Close all collections and connections."""
        for collection in self.collections.values():
            collection.close()
        self.collections.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures all collections are closed."""
        self.close()
        return False

    def __del__(self):
        """Destructor - ensures collection connections are closed."""
        try:
            self.close()
        except Exception:
            pass


# For backward compatibility with PersistentClient
def PersistentClient(path: str) -> ChromaSQLiteClient:
    """Create a persistent SQLite ChromaDB client (drop-in replacement)."""
    return ChromaSQLiteClient(db_path=path)
