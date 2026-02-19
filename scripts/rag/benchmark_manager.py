"""
Benchmark Manager for RAG Dashboard

Handles recording and analysing performance metrics for RAG queries.
Stores benchmarks in SQLite database and provides analytics.
Includes system metrics (CPU, GPU, RAM, VRAM) and user relevancy ratings.
"""

import json
import os
import socket
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil


class BenchmarkManager:
    """Manager for RAG query performance benchmarking."""

    def __init__(self, db_path: str = "rag_data/benchmarks.db"):
        """Initialise benchmark manager with database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Create benchmarks table if it doesn't exist."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS query_benchmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    collection_name TEXT,
                    k_value INTEGER,
                    temperature REAL,
                    model_name TEXT,
                    is_code_query BOOLEAN,
                    
                    -- Performance metrics
                    retrieval_time REAL,
                    generation_time REAL,
                    total_time REAL,
                    
                    -- Result metrics
                    retrieval_count INTEGER,
                    response_length INTEGER,
                    source_count INTEGER,
                    
                    -- Token metrics (if available)
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    
                    -- Cache metrics
                    cache_hit BOOLEAN,
                    cache_key TEXT,
                    
                    -- System metrics (CPU, GPU, RAM, VRAM)
                    cpu_start_percent REAL,
                    cpu_max_percent REAL,
                    ram_start_mb REAL,
                    ram_max_mb REAL,
                    gpu_start_percent REAL,
                    gpu_max_percent REAL,
                    vram_start_mb REAL,
                    vram_max_mb REAL,
                    
                    -- Network latency (ms)
                    network_latency_ms REAL,
                    
                    -- RAG Strategy & Confidence
                    retrieval_methods TEXT,
                    ranking_method TEXT,
                    confidence_level TEXT,
                    avg_similarity REAL,
                    search_strategy TEXT,
                    
                    -- Success/Error
                    success BOOLEAN,
                    error_message TEXT,
                    
                    -- User relevancy rating (1-5 scale)
                    relevancy_rating INTEGER,
                    relevancy_feedback TEXT,
                    
                    -- Additional metadata
                    metadata TEXT
                )
            """
            )

            # Create indices for faster queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON query_benchmarks(timestamp)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_model 
                ON query_benchmarks(model_name)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_success 
                ON query_benchmarks(success)
            """
            )

            conn.commit()

            # Run migrations for existing databases
            self._migrate_database(conn)

        except sqlite3.Error as e:
            print(f"Database Error: {e}")
        finally:
            if conn:
                conn.close()

    def _migrate_database(self, conn: sqlite3.Connection):
        """Add missing columns to existing database tables.

        Args:
            conn: Active database connection
        """
        cursor = conn.cursor()

        # Check existing columns
        cursor.execute("PRAGMA table_info(query_benchmarks)")
        columns = [row[1] for row in cursor.fetchall()]

        # Define columns to add if missing
        columns_to_add = [
            ("relevancy_rating", "INTEGER"),
            ("relevancy_feedback", "TEXT"),
            ("cpu_start_percent", "REAL"),
            ("cpu_max_percent", "REAL"),
            ("ram_start_mb", "REAL"),
            ("ram_max_mb", "REAL"),
            ("gpu_start_percent", "REAL"),
            ("gpu_max_percent", "REAL"),
            ("vram_start_mb", "REAL"),
            ("vram_max_mb", "REAL"),
            ("network_latency_ms", "REAL"),
            ("retrieval_methods", "TEXT"),
            ("ranking_method", "TEXT"),
            ("confidence_level", "TEXT"),
            ("avg_similarity", "REAL"),
            ("search_strategy", "TEXT"),
        ]

        # Add missing columns
        for col_name, col_type in columns_to_add:
            if col_name not in columns:
                try:
                    cursor.execute(
                        f"""
                        ALTER TABLE query_benchmarks 
                        ADD COLUMN {col_name} {col_type}
                    """
                    )
                    print(f"✓ Added {col_name} column to query_benchmarks table")
                except sqlite3.Error as e:
                    print(f"Note: Could not add {col_name} column: {e}")

        conn.commit()

    @staticmethod
    def get_system_metrics() -> Dict[str, Any]:
        """Capture current system metrics.

        Returns:
            Dict with CPU %, RAM MB, GPU %, VRAM MB
        """
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_mb": psutil.virtual_memory().used / 1024 / 1024,
            "gpu_percent": None,
            "vram_mb": None,
        }

        # Try to get GPU metrics using nvidia-smi if available
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 2:
                    metrics["gpu_percent"] = float(parts[0].strip())
                    metrics["vram_mb"] = float(parts[1].strip())
        except Exception:
            pass  # GPU metrics not available

        return metrics

    @staticmethod
    def measure_network_latency(host: str = "8.8.8.8", port: int = 53) -> Optional[float]:
        """Measure network latency in milliseconds.

        Args:
            host: Host to ping (default Google DNS)
            port: Port to use for latency test

        Returns:
            Latency in milliseconds, or None if unavailable
        """
        try:
            start = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(2)
            sock.sendto(b"ping", (host, port))
            latency_ms = (time.time() - start) * 1000
            sock.close()
            return latency_ms
        except Exception:
            return None

    @contextmanager
    def track_query_metrics(self):
        """Context manager to track system metrics during query execution.

        Yields:
            Dict with start metrics and methods to update max values
        """
        start_metrics = self.get_system_metrics()
        max_metrics = start_metrics.copy()

        def update_max():
            """Update max values with current metrics."""
            current = self.get_system_metrics()
            if current["cpu_percent"] is not None:
                max_metrics["cpu_percent"] = max(
                    max_metrics.get("cpu_percent", 0), current["cpu_percent"]
                )
            if current["ram_mb"] is not None:
                max_metrics["ram_mb"] = max(max_metrics.get("ram_mb", 0), current["ram_mb"])
            if current["gpu_percent"] is not None:
                max_metrics["gpu_percent"] = max(
                    max_metrics.get("gpu_percent", 0), current["gpu_percent"]
                )
            if current["vram_mb"] is not None:
                max_metrics["vram_mb"] = max(max_metrics.get("vram_mb", 0), current["vram_mb"])

        try:
            yield {
                "start": start_metrics,
                "max": max_metrics,
                "update_max": update_max,
            }
        finally:
            update_max()  # Final update

    def record_query(
        self,
        query: str,
        response: Dict[str, Any],
        query_params: Dict[str, Any],
        system_metrics: Optional[Dict[str, Any]] = None,
        network_latency_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> int:
        """Record a query benchmark with extended metrics.

        Args:
            query: Query text
            response: RAG response dict with timing/metrics
            query_params: Query parameters (k, temperature, etc.)
            system_metrics: Dict with start/max CPU, GPU, RAM, VRAM metrics
            network_latency_ms: Network latency in milliseconds
            error: Optional error message if query failed

        Returns:
            ID of inserted record
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Extract metrics from response
            generation_time = response.get("generation_time")
            total_time = response.get("total_time")
            retrieval_count = response.get("retrieval_count", 0)
            sources = response.get("sources", [])
            answer = response.get("answer", "")
            is_code_query = response.get("is_code_query", False)
            model_name = response.get("model", "unknown")

            # Calculate retrieval time
            retrieval_time = None
            if total_time is not None and generation_time is not None:
                retrieval_time = total_time - generation_time

            # Token metrics (if available)
            prompt_tokens = response.get("prompt_tokens")
            completion_tokens = response.get("completion_tokens")
            total_tokens = response.get("total_tokens")

            # Cache metrics
            cache_hit = response.get("cache_hit", False)
            cache_key = response.get("cache_key")

            # RAG Strategy & Confidence metrics
            explainability = response.get("explainability", {})
            retrieval_methods = explainability.get("retrieval_method", [])
            retrieval_methods_str = ",".join(retrieval_methods) if retrieval_methods else None
            ranking_method = explainability.get("ranking_explanation")
            confidence_level = explainability.get("confidence_level")
            avg_similarity = explainability.get("avg_similarity")

            # Determine search strategy
            search_strategy = None
            if retrieval_methods:
                if len(retrieval_methods) > 1:
                    search_strategy = "hybrid"
                elif "vector" in retrieval_methods:
                    search_strategy = "vector"
                elif "keyword" in retrieval_methods:
                    search_strategy = "keyword"
                elif "graph" in retrieval_methods:
                    search_strategy = "graph"

            # System metrics (start and max values)
            cpu_start = None
            cpu_max = None
            ram_start = None
            ram_max = None
            gpu_start = None
            gpu_max = None
            vram_start = None
            vram_max = None

            if system_metrics:
                if "start" in system_metrics:
                    cpu_start = system_metrics["start"].get("cpu_percent")
                    ram_start = system_metrics["start"].get("ram_mb")
                    gpu_start = system_metrics["start"].get("gpu_percent")
                    vram_start = system_metrics["start"].get("vram_mb")
                if "max" in system_metrics:
                    cpu_max = system_metrics["max"].get("cpu_percent")
                    ram_max = system_metrics["max"].get("ram_mb")
                    gpu_max = system_metrics["max"].get("gpu_percent")
                    vram_max = system_metrics["max"].get("vram_mb")

            # Additional metadata
            metadata = {
                "custom_role": query_params.get("custom_role"),
                "filters": query_params.get("filters", {}),
            }

            cursor.execute(
                """
                INSERT INTO query_benchmarks (
                    timestamp, query, collection_name, k_value, temperature,
                    model_name, is_code_query,
                    retrieval_time, generation_time, total_time,
                    retrieval_count, response_length, source_count,
                    prompt_tokens, completion_tokens, total_tokens,
                    cache_hit, cache_key,
                    cpu_start_percent, cpu_max_percent,
                    ram_start_mb, ram_max_mb,
                    gpu_start_percent, gpu_max_percent,
                    vram_start_mb, vram_max_mb,
                    network_latency_ms,
                    retrieval_methods, ranking_method, confidence_level, 
                    avg_similarity, search_strategy,
                    success, error_message, relevancy_rating, relevancy_feedback, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    query,
                    query_params.get("collection_name", "unknown"),
                    query_params.get("k"),
                    query_params.get("temperature"),
                    model_name,
                    is_code_query,
                    retrieval_time,
                    generation_time,
                    total_time,
                    retrieval_count,
                    len(answer),
                    len(sources),
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    cache_hit,
                    cache_key,
                    cpu_start,
                    cpu_max,
                    ram_start,
                    ram_max,
                    gpu_start,
                    gpu_max,
                    vram_start,
                    vram_max,
                    network_latency_ms,
                    retrieval_methods_str,
                    ranking_method,
                    confidence_level,
                    avg_similarity,
                    search_strategy,
                    error is None,
                    error,
                    None,  # relevancy_rating - not provided during initial record
                    None,  # relevancy_feedback - not provided during initial record
                    json.dumps(metadata),
                ),
            )

            record_id = cursor.lastrowid
            conn.commit()
        except Exception as e:
            print(f"Database Error: {e}")
            record_id = -1
        finally:
            if conn:
                conn.close()

        return record_id

    def update_relevancy_rating(
        self,
        record_id: int,
        rating: int,
        feedback: Optional[str] = None,
    ) -> bool:
        """Update relevancy rating for a query benchmark.

        Args:
            record_id: ID of benchmark record
            rating: Relevancy rating (1-5 scale)
            feedback: Optional user feedback text

        Returns:
            True if update succeeded, False otherwise
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")

        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE query_benchmarks
                SET relevancy_rating = ?, relevancy_feedback = ?
                WHERE id = ?
            """,
                (rating, feedback, record_id),
            )

            conn.commit()
            return True
        except Exception as e:
            print(f"Error updating relevancy rating: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def get_relevancy_stats(
        self,
        time_range_hours: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get relevancy rating statistics.

        Args:
            time_range_hours: Optional time range in hours

        Returns:
            Dict with relevancy statistics
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            where_clauses = ["relevancy_rating IS NOT NULL"]
            params = []

            if time_range_hours:
                cutoff = datetime.now() - timedelta(hours=time_range_hours)
                where_clauses.append("timestamp >= ?")
                params.append(cutoff.isoformat())

            where_sql = " AND ".join(where_clauses)

            cursor.execute(
                f"""
                SELECT
                    COUNT(*) as total_rated,
                    AVG(relevancy_rating) as avg_rating,
                    MIN(relevancy_rating) as min_rating,
                    MAX(relevancy_rating) as max_rating
                FROM query_benchmarks
                WHERE {where_sql}
            """,
                params,
            )

            result = cursor.fetchone()

            # Get rating distribution
            cursor.execute(
                f"""
                SELECT relevancy_rating, COUNT(*) as count
                FROM query_benchmarks
                WHERE {where_sql}
                GROUP BY relevancy_rating
                ORDER BY relevancy_rating
            """,
                params,
            )

            rating_distribution = {str(row[0]): row[1] for row in cursor.fetchall()}

            return {
                "total_rated": result[0] or 0,
                "avg_rating": result[1] or 0,
                "min_rating": result[2] or 0,
                "max_rating": result[3] or 0,
                "distribution": rating_distribution,
            }
        except Exception as e:
            print(f"Error retrieving relevancy stats: {e}")
            return {}
        finally:
            if conn:
                conn.close()

    def get_queries_by_relevancy(
        self,
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        time_range_hours: Optional[int] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get queries filtered by relevancy rating.

        Args:
            min_rating: Minimum rating (1-5)
            max_rating: Maximum rating (1-5)
            time_range_hours: Optional time range
            limit: Maximum number of results

        Returns:
            List of query records with ratings
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            where_clauses = ["relevancy_rating IS NOT NULL"]
            params = []

            if min_rating is not None:
                where_clauses.append("relevancy_rating >= ?")
                params.append(min_rating)

            if max_rating is not None:
                where_clauses.append("relevancy_rating <= ?")
                params.append(max_rating)

            if time_range_hours:
                cutoff = datetime.now() - timedelta(hours=time_range_hours)
                where_clauses.append("timestamp >= ?")
                params.append(cutoff.isoformat())

            where_sql = " AND ".join(where_clauses)

            cursor.execute(
                f"""
                SELECT
                    id, timestamp, query, model_name, total_time,
                    relevancy_rating, relevancy_feedback, cache_hit,
                    cpu_max_percent, ram_max_mb, gpu_max_percent, vram_max_mb
                FROM query_benchmarks
                WHERE {where_sql}
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                params + [limit],
            )

            columns = [
                "id",
                "timestamp",
                "query",
                "model",
                "total_time",
                "rating",
                "feedback",
                "cache_hit",
                "cpu_max",
                "ram_max",
                "gpu_max",
                "vram_max",
            ]

            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

            return results
        except Exception as e:
            print(f"Error retrieving queries by relevancy: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_statistics(
        self,
        time_range_hours: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get aggregate statistics for benchmarks.

        Args:
            time_range_hours: Optional time range in hours (e.g., 24 for last day)
            model_name: Optional filter by model name

        Returns:
            Dict with aggregate statistics
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Build WHERE clause
            where_clauses = ["success = 1"]  # Only successful queries
            params = []

            if time_range_hours:
                cutoff = datetime.now() - timedelta(hours=time_range_hours)
                where_clauses.append("timestamp >= ?")
                params.append(cutoff.isoformat())

            if model_name:
                where_clauses.append("model_name = ?")
                params.append(model_name)

            where_sql = " AND ".join(where_clauses)

            # Get aggregate metrics
            cursor.execute(
                f"""
                SELECT
                    COUNT(*) as total_queries,
                    AVG(total_time) as avg_total_time,
                    AVG(generation_time) as avg_generation_time,
                    AVG(retrieval_time) as avg_retrieval_time,
                    MIN(total_time) as min_time,
                    MAX(total_time) as max_time,
                    AVG(retrieval_count) as avg_retrieval_count,
                    AVG(response_length) as avg_response_length,
                    AVG(total_tokens) as avg_total_tokens,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits,
                    SUM(CASE WHEN is_code_query = 1 THEN 1 ELSE 0 END) as code_queries
                FROM query_benchmarks
                WHERE {where_sql}
            """,
                params,
            )

            row = cursor.fetchone()

            if not row or row[0] == 0:
                conn.close()
                return {
                    "total_queries": 0,
                    "avg_total_time": 0,
                    "avg_generation_time": 0,
                    "avg_retrieval_time": 0,
                    "cache_hit_rate": 0,
                    "code_query_rate": 0,
                    "avg_cpu_percent": 0,
                    "avg_ram_mb": 0,
                    "avg_gpu_percent": 0,
                    "avg_vram_mb": 0,
                }

            total_queries = row[0]
            cache_hits = row[9] or 0
            code_queries = row[10] or 0

            stats = {
                "total_queries": total_queries,
                "avg_total_time": row[1] or 0,
                "avg_generation_time": row[2] or 0,
                "avg_retrieval_time": row[3] or 0,
                "min_time": row[4] or 0,
                "max_time": row[5] or 0,
                "avg_retrieval_count": row[6] or 0,
                "avg_response_length": row[7] or 0,
                "avg_total_tokens": row[8] or 0,
                "cache_hit_rate": (cache_hits / total_queries * 100) if total_queries > 0 else 0,
                "code_query_rate": (code_queries / total_queries * 100) if total_queries > 0 else 0,
            }

            # Get system metrics averages
            cursor.execute(
                f"""
                SELECT
                    AVG(cpu_max_percent) as avg_cpu,
                    AVG(ram_max_mb) as avg_ram,
                    AVG(gpu_max_percent) as avg_gpu,
                    AVG(vram_max_mb) as avg_vram,
                    AVG(network_latency_ms) as avg_latency
                FROM query_benchmarks
                WHERE {where_sql}
            """,
                params,
            )

            sys_row = cursor.fetchone()
            if sys_row:
                stats["avg_cpu_percent"] = sys_row[0] or 0
                stats["avg_ram_mb"] = sys_row[1] or 0
                stats["avg_gpu_percent"] = sys_row[2] or 0
                stats["avg_vram_mb"] = sys_row[3] or 0
                stats["avg_network_latency_ms"] = sys_row[4] or 0

            return stats
        except Exception as e:
            print(f"Error retrieving statistics: {e}")
            return {}
        finally:
            if conn:
                conn.close()

    def get_time_series(
        self,
        metric: str = "total_time",
        time_range_hours: int = 24,
        bucket_minutes: int = 60,
    ) -> List[Tuple[str, float]]:
        """Get time series data for a specific metric.

        Args:
            metric: Metric name (total_time, generation_time, etc.)
            time_range_hours: Time range in hours
            bucket_minutes: Bucket size in minutes for aggregation

        Returns:
            List of (timestamp, value) tuples
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff = datetime.now() - timedelta(hours=time_range_hours)

            # SQLite doesn't have great time bucketing, so we'll fetch all and bucket in Python
            cursor.execute(
                f"""
                SELECT timestamp, {metric}
                FROM query_benchmarks
                WHERE timestamp >= ? AND success = 1 AND {metric} IS NOT NULL
                ORDER BY timestamp
            """,
                (cutoff.isoformat(),),
            )

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return []

            # Bucket by time
            buckets: Dict[str, List[float]] = {}
            bucket_size = timedelta(minutes=bucket_minutes)

            for ts_str, value in rows:
                ts = datetime.fromisoformat(ts_str)
                # Round down to bucket
                bucket_start = ts.replace(second=0, microsecond=0)
                bucket_start = bucket_start - timedelta(
                    minutes=bucket_start.minute % bucket_minutes
                )
                bucket_key = bucket_start.isoformat()

                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                buckets[bucket_key].append(value)

            # Average each bucket
            result = []
            for bucket_key in sorted(buckets.keys()):
                values = buckets[bucket_key]
                avg_value = sum(values) / len(values)
                result.append((bucket_key, avg_value))

            return result
        except Exception as e:
            print(f"Error retrieving time series: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_slowest_queries(
        self,
        limit: int = 10,
        time_range_hours: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get slowest queries.

        Args:
            limit: Number of queries to return
            time_range_hours: Optional time range filter

        Returns:
            List of query records sorted by total_time (descending)
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            where_clause = "success = 1"
            params = []

            if time_range_hours:
                cutoff = datetime.now() - timedelta(hours=time_range_hours)
                where_clause += " AND timestamp >= ?"
                params.append(cutoff.isoformat())

            cursor.execute(
                f"""
                SELECT
                    timestamp, query, total_time, generation_time,
                    retrieval_count, model_name, is_code_query
                FROM query_benchmarks
                WHERE {where_clause}
                ORDER BY total_time DESC
                LIMIT ?
            """,
                params + [limit],
            )

            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append(
                    {
                        "timestamp": row[0],
                        "query": row[1],
                        "total_time": row[2],
                        "generation_time": row[3],
                        "retrieval_count": row[4],
                        "model_name": row[5],
                        "is_code_query": bool(row[6]),
                    }
                )

            return results
        except Exception as e:
            print(f"Error retrieving slowest queries: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def export_report(
        self,
        output_path: str,
        time_range_hours: Optional[int] = None,
    ) -> str:
        """Export benchmark report to Markdown.

        Args:
            output_path: Path to save report
            time_range_hours: Optional time range filter

        Returns:
            Report content as string
        """
        stats = self.get_statistics(time_range_hours=time_range_hours)
        slowest = self.get_slowest_queries(limit=10, time_range_hours=time_range_hours)

        lines = []
        lines.append("# RAG Query Benchmark Report\n")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if time_range_hours:
            lines.append(f"**Time Range:** Last {time_range_hours} hours\n")

        lines.append("---\n")
        lines.append("## Summary Statistics\n")
        lines.append(f"- **Total Queries:** {stats['total_queries']}")
        lines.append(f"- **Average Total Time:** {stats['avg_total_time']:.2f}s")
        lines.append(f"- **Average Generation Time:** {stats['avg_generation_time']:.2f}s")
        lines.append(f"- **Average Retrieval Time:** {stats['avg_retrieval_time']:.2f}s")
        lines.append(f"- **Min Time:** {stats['min_time']:.2f}s")
        lines.append(f"- **Max Time:** {stats['max_time']:.2f}s")
        lines.append(f"- **Cache Hit Rate:** {stats['cache_hit_rate']:.1f}%")
        lines.append(f"- **Code Query Rate:** {stats['code_query_rate']:.1f}%")
        lines.append(f"- **Avg Response Length:** {stats['avg_response_length']:.0f} chars")
        lines.append("")

        lines.append("## Slowest Queries\n")
        lines.append("| Timestamp | Query | Total Time | Generation Time | Model |")
        lines.append("|-----------|-------|------------|-----------------|-------|")

        for query_rec in slowest:
            timestamp = query_rec["timestamp"][:19]  # Truncate to seconds
            query_text = (
                query_rec["query"][:50] + "..."
                if len(query_rec["query"]) > 50
                else query_rec["query"]
            )
            total_time = query_rec["total_time"]
            gen_time = query_rec["generation_time"]
            model = query_rec["model_name"]

            lines.append(
                f"| {timestamp} | {query_text} | {total_time:.2f}s | {gen_time:.2f}s | {model} |"
            )

        lines.append("")

        report = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(report)

        return report

    def clear_old_benchmarks(self, days: int = 30):
        """Delete benchmarks older than specified days.

        Args:
            days: Number of days to keep
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff = datetime.now() - timedelta(days=days)

            cursor.execute(
                """
                DELETE FROM query_benchmarks
                WHERE timestamp < ?
            """,
                (cutoff.isoformat(),),
            )

            deleted = cursor.rowcount
            conn.commit()
        except Exception as e:
            print(f"Error clearing old benchmarks: {e}")
            deleted = 0
        finally:
            if conn:
                conn.close()

        return deleted
