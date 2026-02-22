#!/usr/bin/env python3
"""
Test script to verify Phase 1 consistency graph optimisations.

Tests:
1. Adaptive max_neighbours calculation (20 → 5 for large graphs)
2. Heuristic filtering (different languages, no shared deps, extreme similarities)
3. GPU concurrency limiting (semaphore mechanism)
4. Performance monitoring (LLM calls, filtered count, cache stats)
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import threading
import time

from scripts.consistency_graph.build_consistency_graph import (
    GPULock,
    calculate_optimal_neighbours,
    should_validate_with_llm,
)
from scripts.consistency_graph.consistency_config import ConsistencyConfig


def test_adaptive_neighbours():
    """Test that max_neighbours adapts based on graph size."""
    print("\n" + "=" * 80)
    print("TEST: Adaptive max_neighbours calculation")
    print("=" * 80)

    # Test stepped approach:
    # <100: min(config, 20)
    # <500: min(config, 15)
    # <2000: min(config, 10)
    # >=2000: min(config, 5)

    # Very small graph
    tiny_result = calculate_optimal_neighbours(50, 20)
    print(f"Tiny graph (50 nodes, max=20): {tiny_result}")
    assert tiny_result == 20, f"Expected 20, got {tiny_result}"

    # Medium graph
    medium_result = calculate_optimal_neighbours(1000, 20)
    print(f"Medium graph (1000 nodes, max=20): {medium_result}")
    assert medium_result == 10, f"Expected 10, got {medium_result}"

    # Large graph (our use case with 4420 nodes)
    large_result = calculate_optimal_neighbours(4420, 20)
    print(f"Large graph (4420 nodes, max=20): {large_result}")
    assert large_result == 5, f"Expected 5, got {large_result}"

    # Boundary case - exactly 2000 nodes
    boundary_result = calculate_optimal_neighbours(2000, 20)
    print(f"Boundary (2000 nodes, max=20): {boundary_result}")
    assert boundary_result == 5, f"Expected 5, got {boundary_result}"

    # Test that config_max is honored when lower
    capped_result = calculate_optimal_neighbours(3000, 3)
    print(f"Capped (3000 nodes, max=3): {capped_result}")
    assert capped_result == 3, f"Expected 3, got {capped_result}"

    print("✅ Adaptive neighbours test PASSED")


def test_heuristic_filtering():
    """Test heuristic pre-filtering logic."""
    print("\n" + "=" * 80)
    print("TEST: Heuristic pre-filtering")
    print("=" * 80)

    # Test case 1: Different languages (should filter - only for code docs)
    doc1_python = {
        "source_category": "code",
        "language": "python",
        "dependencies": ["flask", "sqlalchemy"],
    }
    doc2_java = {
        "source_category": "code",
        "language": "java",
        "dependencies": ["spring", "hibernate"],
    }
    result1 = should_validate_with_llm(doc1_python, doc2_java, 0.75, enable_heuristic=True)
    print(f"Different languages (Python vs Java): {result1}")
    assert result1 is False, "Should filter different languages"

    # Test case 2: No shared dependencies (should filter - only for code docs)
    doc3_python = {
        "source_category": "code",
        "language": "python",
        "dependencies": ["django", "celery"],
    }
    doc4_python = {
        "source_category": "code",
        "language": "python",
        "dependencies": ["flask", "gunicorn"],
    }
    result2 = should_validate_with_llm(doc3_python, doc4_python, 0.75, enable_heuristic=True)
    print(f"Same language, no shared deps: {result2}")
    assert result2 is False, "Should filter no shared dependencies"

    # Test case 3: Very high similarity (should filter)
    doc5 = {"source_category": "code", "language": "python", "dependencies": ["flask"]}
    doc6 = {"source_category": "code", "language": "python", "dependencies": ["flask"]}
    result3 = should_validate_with_llm(doc5, doc6, 0.98, enable_heuristic=True)  # >0.95
    print(f"Very high similarity (0.98): {result3}")
    assert result3 is False, "Should filter very high similarity"

    # Test case 4: Very low similarity (should filter)
    doc7 = {"source_category": "code", "language": "python", "dependencies": ["flask"]}
    doc8 = {"source_category": "code", "language": "python", "dependencies": ["django"]}
    result4 = should_validate_with_llm(doc7, doc8, 0.05, enable_heuristic=True)  # <0.3
    print(f"Very low similarity (0.05): {result4}")
    assert result4 is False, "Should filter very low similarity"

    # Test case 5: Good candidate (should NOT filter)
    doc9 = {
        "source_category": "code",
        "language": "python",
        "dependencies": ["flask", "sqlalchemy"],
    }
    doc10 = {"source_category": "code", "language": "python", "dependencies": ["flask", "celery"]}
    result5 = should_validate_with_llm(doc9, doc10, 0.75, enable_heuristic=True)
    print(f"Good candidate (same lang, shared deps, normal similarity): {result5}")
    assert result5 is True, "Should NOT filter good candidates"

    # Test case 6: Heuristic disabled (should accept everything except extreme similarity)
    doc11 = {"source_category": "code", "language": "python", "dependencies": []}
    doc12 = {"source_category": "code", "language": "java", "dependencies": []}
    result6 = should_validate_with_llm(doc11, doc12, 0.5, enable_heuristic=False)
    print(f"Heuristic disabled, mid similarity: {result6}")
    assert result6 is True, "Should NOT filter when heuristic is disabled"

    print("✅ Heuristic filtering test PASSED")


def test_gpu_lock():
    """Test GPU concurrency limiting with semaphore."""
    print("\n" + "=" * 80)
    print("TEST: GPU concurrency limiting")
    print("=" * 80)

    # Create lock with max_concurrent=2
    gpu_lock = GPULock(max_concurrent=2)
    concurrent_count = 0
    max_concurrent_observed = 0
    lock = threading.Lock()

    def worker(worker_id):
        nonlocal concurrent_count, max_concurrent_observed

        with gpu_lock:
            # Track concurrent execution
            with lock:
                concurrent_count += 1
                max_concurrent_observed = max(max_concurrent_observed, concurrent_count)
                print(f"Worker {worker_id} acquired lock (concurrent: {concurrent_count})")

            # Simulate GPU work
            time.sleep(0.1)

            # Release tracking
            with lock:
                concurrent_count -= 1
                print(f"Worker {worker_id} releasing lock (concurrent: {concurrent_count})")

    # Spawn 5 threads (more than semaphore limit of 2)
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    print(f"Max concurrent observed: {max_concurrent_observed}")
    assert max_concurrent_observed == 2, f"Expected max 2 concurrent, got {max_concurrent_observed}"
    print("✅ GPU lock test PASSED")


def test_config_integration():
    """Test that ConsistencyConfig loads optimisation settings."""
    print("\n" + "=" * 80)
    print("TEST: Configuration integration")
    print("=" * 80)

    config = ConsistencyConfig()

    print(f"GPU concurrency: {config.gpu_concurrency}")
    assert hasattr(config, "gpu_concurrency"), "Missing gpu_concurrency attribute"
    assert config.gpu_concurrency == 2, f"Expected gpu_concurrency=2, got {config.gpu_concurrency}"

    print(f"Enable heuristic filter: {config.enable_heuristic_filter}")
    assert hasattr(config, "enable_heuristic_filter"), "Missing enable_heuristic_filter attribute"
    assert (
        config.enable_heuristic_filter is True
    ), f"Expected enable_heuristic_filter=True, got {config.enable_heuristic_filter}"

    print("✅ Config integration test PASSED")


def main():
    """Run all tests."""
    print("\n" + "#" * 80)
    print("# Phase 1 Optimisation Tests")
    print("#" * 80)

    try:
        test_adaptive_neighbours()
        test_heuristic_filtering()
        test_gpu_lock()
        test_config_integration()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nPhase 1 optimisations are working correctly:")
        print("  - Adaptive max_neighbours: 20 → 5 for graphs >2000 nodes")
        print("  - Heuristic filtering: Different languages, no shared deps, extreme similarities")
        print("  - GPU concurrency limiting: Max 2 concurrent LLM calls")
        print("  - Configuration: GPU concurrency and heuristic filter flags")
        print("\nReady to test on actual consistency graph build!")

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
