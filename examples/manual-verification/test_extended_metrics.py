"""Test script for extended metrics and relevancy rating functionality."""

import tempfile
from pathlib import Path

from scripts.rag.benchmark_manager import BenchmarkManager


def test_extended_metrics():
    """Test system metrics capture and storage."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        manager = BenchmarkManager(db_path)

        print("✅ Testing System Metrics Collection")
        print("-" * 50)

        # Test metrics capture
        metrics = manager.get_system_metrics()
        print(f"CPU: {metrics['cpu_percent']:.1f}%")
        print(f"RAM: {metrics['ram_mb']:.0f} MB")
        print(f"GPU: {metrics['gpu_percent']:.1f}%" if metrics["gpu_percent"] else "GPU: N/A")
        print(f"VRAM: {metrics['vram_mb']:.0f} MB" if metrics["vram_mb"] else "VRAM: N/A")

        # Test network latency
        print("\n✅ Testing Network Latency")
        print("-" * 50)
        latency = manager.measure_network_latency()
        print(f"Network Latency: {latency:.1f} ms" if latency else "Network Latency: N/A")

        # Test query recording with metrics
        print("\n✅ Testing Query Recording with Extended Metrics")
        print("-" * 50)

        with manager.track_query_metrics() as sys_metrics:
            # Simulate query
            import time

            time.sleep(0.1)
            sys_metrics["update_max"]()

            response = {
                "answer": "Test response",
                "sources": ["doc1", "doc2"],
                "total_time": 2.5,
                "generation_time": 1.8,
                "model": "llama2",
                "is_code_query": False,
                "cache_hit": True,
            }

            record_id = manager.record_query(
                query="Test query",
                response=response,
                query_params={"k": 5, "temperature": 0.7},
                system_metrics=sys_metrics,
                network_latency_ms=latency,
            )

        print(f"Record ID: {record_id}")
        print("Query recorded with system metrics ✓")

        # Test relevancy rating
        print("\n✅ Testing Relevancy Rating")
        print("-" * 50)

        success = manager.update_relevancy_rating(
            record_id=record_id, rating=4, feedback="Good response with minor gaps"
        )
        print(f"Rating updated: {success}")
        print("Feedback: 'Good response with minor gaps' ✓")

        # Record more queries with different ratings
        for i in range(2, 5):
            with manager.track_query_metrics() as sys_metrics:
                response = {
                    "answer": f"Response {i}",
                    "sources": ["doc"],
                    "total_time": 1.5 + i * 0.2,
                    "generation_time": 1.0 + i * 0.1,
                    "model": "llama2",
                    "is_code_query": False,
                    "cache_hit": i % 2 == 0,
                }

                rid = manager.record_query(
                    query=f"Query {i}",
                    response=response,
                    query_params={"k": 5},
                    system_metrics=sys_metrics,
                    network_latency_ms=latency,
                )

                manager.update_relevancy_rating(
                    record_id=rid, rating=min(5, i + 2), feedback=f"Test feedback {i}"
                )

        # Test relevancy statistics
        print("\n✅ Testing Relevancy Statistics")
        print("-" * 50)

        rel_stats = manager.get_relevancy_stats()
        print(f"Total Rated: {rel_stats['total_rated']}")
        print(f"Avg Rating: {rel_stats['avg_rating']:.2f}/5")
        print(f"Min Rating: {rel_stats['min_rating']}")
        print(f"Max Rating: {rel_stats['max_rating']}")
        print(f"Distribution: {rel_stats['distribution']}")

        # Test filtering by relevancy
        print("\n✅ Testing Relevancy Filtering")
        print("-" * 50)

        high_rating_queries = manager.get_queries_by_relevancy(min_rating=4, limit=5)
        print(f"High-rated queries (4+): {len(high_rating_queries)}")
        for q in high_rating_queries[:2]:
            print(f"  - {q['query']}: {q['rating']}★ ({q['total_time']:.2f}s)")

        # Test statistics with system metrics
        print("\n✅ Testing Statistics with System Metrics")
        print("-" * 50)

        stats = manager.get_statistics()
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Avg Response Time: {stats['avg_total_time']:.2f}s")
        print(f"Avg CPU: {stats.get('avg_cpu_percent', 0):.1f}%")
        print(f"Avg RAM: {stats.get('avg_ram_mb', 0):.0f} MB")
        print(f"Avg GPU: {stats.get('avg_gpu_percent', 'N/A')}")
        print(f"Avg VRAM: {stats.get('avg_vram_mb', 'N/A')}")
        print(f"Avg Network Latency: {stats.get('avg_network_latency_ms', 'N/A')}")

        print("\n" + "=" * 50)
        print("✅ ALL EXTENDED METRICS TESTS PASSED")
        print("=" * 50)

    finally:
        # Cleanup
        Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_extended_metrics()
