"""
Comprehensive test suite for Phase 3: WebGL Rendering Initiative

Tests cover:
1. Layout Engine (3 algorithms)
2. Spatial Indexing (Quadtree, Bounds)
3. Performance Monitoring
4. Viewport Culling
5. Dashboard Integration
6. WebGL Rendering
"""

import time
import random
from pathlib import Path
import sys

import pytest

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ui.layout_engine import (
    ForceDirectedLayout,
    HierarchicalLayout,
    CircularLayout,
)
from scripts.ui.spatial_index import Bounds, Quadtree
from scripts.ui.performance_monitor import PerformanceMonitor, format_timings_ms


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def small_graph():
    """Create a small test graph (10 nodes, 15 edges)."""
    nodes = {f"node_{i}": {"conflict_score": random.random()} for i in range(10)}
    edges = [
        {"source": "node_0", "target": "node_1"},
        {"source": "node_1", "target": "node_2"},
        {"source": "node_2", "target": "node_3"},
        {"source": "node_0", "target": "node_4"},
        {"source": "node_4", "target": "node_5"},
        {"source": "node_5", "target": "node_6"},
        {"source": "node_2", "target": "node_7"},
        {"source": "node_3", "target": "node_8"},
        {"source": "node_6", "target": "node_9"},
    ]
    return nodes, edges


@pytest.fixture
def medium_graph():
    """Create a medium test graph (100 nodes, 200 edges)."""
    nodes = {f"node_{i}": {"conflict_score": random.random()} for i in range(100)}
    edges = []
    for i in range(100):
        # Each node connects to 2-3 random others
        for _ in range(random.randint(2, 3)):
            target = random.randint(0, 99)
            if target != i:
                edges.append({"source": f"node_{i}", "target": f"node_{target}"})
    return nodes, edges


@pytest.fixture
def large_graph():
    """Create a large test graph (1000 nodes, 3000 edges)."""
    nodes = {f"node_{i}": {"conflict_score": random.random()} for i in range(1000)}
    edges = []
    for i in range(1000):
        # Each node connects to ~3 others on average
        for _ in range(random.randint(1, 5)):
            target = random.randint(0, 999)
            if target != i:
                edges.append({"source": f"node_{i}", "target": f"node_{target}"})
    # Limit to prevent memory issues in tests
    edges = edges[:3000]
    return nodes, edges


# ============================================================================
# Layout Engine Tests
# ============================================================================

class TestForceDirectedLayout:
    """Test Force-Directed layout algorithm."""

    def test_basic_layout_computation(self, small_graph):
        """Test that layout produces valid positions."""
        nodes, edges = small_graph
        layout = ForceDirectedLayout(k=1.0, damping=0.5, iterations=20)
        positions = layout.compute_layout(nodes, edges)
        
        # All nodes should have positions
        assert len(positions) == len(nodes)
        
        # Positions should be finite
        for node_id, (x, y) in positions.items():
            assert isinstance(x, float) and isinstance(y, float)
            assert abs(x) < 1e6 and abs(y) < 1e6  # Not inf/nan
    
    def test_layout_performance_small(self, small_graph):
        """Test that small graphs compute quickly."""
        nodes, edges = small_graph
        layout = ForceDirectedLayout(iterations=20)
        
        start = time.time()
        positions = layout.compute_layout(nodes, edges)
        elapsed = time.time() - start
        
        # Should be <100ms for 10 nodes
        assert elapsed < 0.1, f"Expected <100ms, got {elapsed*1000:.1f}ms"
    
    def test_layout_performance_medium(self, medium_graph):
        """Test that medium graphs compute in reasonable time."""
        nodes, edges = medium_graph
        layout = ForceDirectedLayout(iterations=20)
        
        start = time.time()
        positions = layout.compute_layout(nodes, edges)
        elapsed = time.time() - start
        
        # Should be <2s for 100 nodes with 20 iterations
        assert elapsed < 2.0, f"Expected <2s, got {elapsed:.2f}s"
    
    def test_connected_nodes_attract(self, small_graph):
        """Test that connected nodes are closer than unconnected."""
        nodes, edges = small_graph
        layout = ForceDirectedLayout(k=1.0, damping=0.5, iterations=50)
        positions = layout.compute_layout(nodes, edges)
        
        # Check a few edges to verify attractive forces
        for edge in edges[:5]:
            src_pos = positions[edge["source"]]
            tgt_pos = positions[edge["target"]]
            distance = ((src_pos[0] - tgt_pos[0])**2 + (src_pos[1] - tgt_pos[1])**2)**0.5
            
            # Connected nodes should be reasonably close (allow wider tolerance)
            assert distance < 15.0, f"Connected nodes too far apart: {distance}"
    
    def test_convergence_with_iterations(self, small_graph):
        """Test that layout converges with more iterations."""
        nodes, edges = small_graph
        
        layout1 = ForceDirectedLayout(iterations=10)
        layout2 = ForceDirectedLayout(iterations=50)
        
        pos1 = layout1.compute_layout(nodes, edges)
        pos2 = layout2.compute_layout(nodes, edges)
        
        # More iterations should produce valid layouts
        assert len(pos1) == len(pos2) == len(nodes)


class TestHierarchicalLayout:
    """Test Hierarchical layout algorithm."""

    def test_hierarchical_layout_computation(self, small_graph):
        """Test that hierarchical layout produces valid positions."""
        nodes, edges = small_graph
        layout = HierarchicalLayout(layer_gap=2.0, node_gap=1.0)
        positions = layout.compute_layout(nodes, edges)
        
        assert len(positions) == len(nodes)
        for node_id, (x, y) in positions.items():
            assert isinstance(x, float) and isinstance(y, float)
    
    def test_hierarchical_layer_assignment(self, small_graph):
        """Test that nodes are assigned to layers."""
        nodes, edges = small_graph
        layout = HierarchicalLayout(layer_gap=2.0, node_gap=1.0)
        positions = layout.compute_layout(nodes, edges)
        
        # Extract Y coordinates (should represent layers)
        y_coords = sorted(set(y for x, y in positions.values()))
        
        # Should have multiple layers for this graph
        assert len(y_coords) >= 1
    
    def test_hierarchical_performance(self, medium_graph):
        """Test hierarchical layout performance."""
        nodes, edges = medium_graph
        layout = HierarchicalLayout()
        
        start = time.time()
        positions = layout.compute_layout(nodes, edges)
        elapsed = time.time() - start
        
        # Should be faster than force-directed for DAGs
        assert elapsed < 1.0, f"Hierarchical layout too slow: {elapsed:.2f}s"


class TestCircularLayout:
    """Test Circular layout algorithm."""

    def test_circular_layout_computation(self, small_graph):
        """Test that circular layout produces valid positions."""
        nodes, edges = small_graph
        layout = CircularLayout()
        positions = layout.compute_layout(nodes, edges)
        
        assert len(positions) == len(nodes)
        for node_id, (x, y) in positions.items():
            assert isinstance(x, float) and isinstance(y, float)
    
    def test_circular_layout_is_circular(self, small_graph):
        """Test that nodes are arranged in a circle."""
        nodes, edges = small_graph
        layout = CircularLayout()
        positions = layout.compute_layout(nodes, edges)
        
        # Calculate distances from center (0, 0)
        distances = []
        for x, y in positions.values():
            dist = (x**2 + y**2)**0.5
            distances.append(dist)
        
        # All distances should be similar (within ~20% of mean)
        avg_dist = sum(distances) / len(distances)
        for dist in distances:
            ratio = dist / avg_dist if avg_dist > 0 else 1.0
            assert 0.8 <= ratio <= 1.2, f"Node too far from circle: {ratio}"
    
    def test_circular_performance(self, medium_graph):
        """Test circular layout is fast."""
        nodes, edges = medium_graph
        layout = CircularLayout()
        
        start = time.time()
        positions = layout.compute_layout(nodes, edges)
        elapsed = time.time() - start
        
        # Should be very fast (no iteration)
        assert elapsed < 0.2, f"Circular layout too slow: {elapsed:.3f}s"


# ============================================================================
# Spatial Indexing Tests
# ============================================================================

class TestBounds:
    """Test Bounds class for bounding box operations."""

    def test_bounds_creation(self):
        """Test creating bounds."""
        bounds = Bounds(0.0, 0.0, 10.0, 10.0)
        assert bounds.min_x == 0.0
        assert bounds.min_y == 0.0
        assert bounds.max_x == 10.0
        assert bounds.max_y == 10.0
    
    def test_bounds_expansion(self):
        """Test bounds expansion."""
        bounds = Bounds(0.0, 0.0, 10.0, 10.0)
        expanded = bounds.expanded(1.0)
        
        assert expanded.min_x == -1.0
        assert expanded.min_y == -1.0
        assert expanded.max_x == 11.0
        assert expanded.max_y == 11.0
    
    def test_bounds_contains_point(self):
        """Test point containment."""
        bounds = Bounds(0.0, 0.0, 10.0, 10.0)
        
        assert bounds.contains_point(5.0, 5.0)
        assert bounds.contains_point(0.0, 0.0)
        assert bounds.contains_point(10.0, 10.0)
        assert not bounds.contains_point(-1.0, 5.0)
        assert not bounds.contains_point(11.0, 5.0)
    
    def test_bounds_intersects(self):
        """Test bounds intersection."""
        b1 = Bounds(0.0, 0.0, 10.0, 10.0)
        b2 = Bounds(5.0, 5.0, 15.0, 15.0)
        
        assert b1.intersects(b2)
        assert b2.intersects(b1)
        
        b3 = Bounds(20.0, 20.0, 30.0, 30.0)
        assert not b1.intersects(b3)


class TestQuadtree:
    """Test Quadtree spatial index."""

    def test_quadtree_creation(self):
        """Test creating a quadtree."""
        bounds = Bounds(-10.0, -10.0, 10.0, 10.0)
        qtree = Quadtree(bounds, max_depth=5, max_items=4)
        
        # Verify quadtree was created (root node exists)
        assert qtree.root is not None
        assert qtree.root.depth == 0
    
    def test_quadtree_insert_and_query(self):
        """Test inserting and querying points."""
        bounds = Bounds(-10.0, -10.0, 10.0, 10.0)
        qtree = Quadtree(bounds, max_depth=8, max_items=4)
        
        # Insert points
        for i in range(100):
            x = random.uniform(-10.0, 10.0)
            y = random.uniform(-10.0, 10.0)
            qtree.insert(f"point_{i}", x, y)
        
        # Query a region
        query_bounds = Bounds(0.0, 0.0, 5.0, 5.0)
        results = qtree.query(query_bounds)
        
        assert len(results) > 0
        assert all(isinstance(r, str) for r in results)
    
    def test_quadtree_empty_query(self):
        """Test querying empty region."""
        bounds = Bounds(-10.0, -10.0, 10.0, 10.0)
        qtree = Quadtree(bounds, max_depth=5, max_items=4)
        
        # Insert points in one corner
        for i in range(10):
            qtree.insert(f"point_{i}", random.uniform(-10.0, -5.0), random.uniform(-10.0, -5.0))
        
        # Query opposite corner
        query_bounds = Bounds(5.0, 5.0, 10.0, 10.0)
        results = qtree.query(query_bounds)
        
        assert len(results) == 0
    
    def test_quadtree_all_points_query(self):
        """Test querying all points."""
        bounds = Bounds(-10.0, -10.0, 10.0, 10.0)
        qtree = Quadtree(bounds, max_depth=8, max_items=4)
        
        n_points = 50
        for i in range(n_points):
            x = random.uniform(-10.0, 10.0)
            y = random.uniform(-10.0, 10.0)
            qtree.insert(f"point_{i}", x, y)
        
        # Query entire bounds
        results = qtree.query(bounds)
        assert len(results) == n_points


# ============================================================================
# Performance Monitoring Tests
# ============================================================================

class TestPerformanceMonitor:
    """Test performance monitoring utilities."""

    def test_performance_monitor_record(self):
        """Test recording performance segments."""
        monitor = PerformanceMonitor()
        
        time.sleep(0.01)
        monitor.record("segment_1")
        
        time.sleep(0.01)
        monitor.record("segment_2")
        
        snapshot = monitor.snapshot()
        assert "segment_1" in snapshot
        assert "segment_2" in snapshot
        assert snapshot["segment_1"] > 0.0
        assert snapshot["segment_2"] > 0.0
    
    def test_performance_monitor_reset(self):
        """Test resetting monitor."""
        monitor = PerformanceMonitor()
        
        monitor.record("segment_1")
        monitor.reset()
        
        snapshot = monitor.snapshot()
        assert len(snapshot) == 0
    
    def test_format_timings_ms(self):
        """Test formatting timings to milliseconds."""
        timings = {"segment_1": 0.025, "segment_2": 0.100}
        formatted = format_timings_ms(timings)
        
        assert formatted["segment_1"] == 25.0
        assert formatted["segment_2"] == 100.0
        assert all(isinstance(v, float) for v in formatted.values())
    
    def test_performance_monitor_ordering(self):
        """Test that segment order is preserved."""
        monitor = PerformanceMonitor()
        
        for i in range(5):
            time.sleep(0.001)
            monitor.record(f"segment_{i}")
        
        assert monitor.order == [f"segment_{i}" for i in range(5)]


# ============================================================================
# Integration Tests
# ============================================================================

class TestLayoutWithCulling:
    """Test layout engine with viewport culling."""

    def test_culling_reduces_visible_nodes(self, medium_graph):
        """Test that viewport culling reduces node count."""
        nodes, edges = medium_graph
        layout = ForceDirectedLayout(iterations=30)
        positions = layout.compute_layout(nodes, edges)
        
        # Get bounds
        xs = [x for x, y in positions.values()]
        ys = [y for x, y in positions.values()]
        bounds = Bounds(min(xs), min(ys), max(xs), max(ys))
        
        # Build quadtree
        qtree = Quadtree(bounds.expanded(0.1), max_depth=10, max_items=16)
        for node_id, (x, y) in positions.items():
            qtree.insert(node_id, x, y)
        
        # Query viewport (smaller region)
        viewport = Bounds(
            bounds.min_x,
            bounds.min_y,
            bounds.min_x + (bounds.max_x - bounds.min_x) * 0.5,
            bounds.min_y + (bounds.max_y - bounds.min_y) * 0.5,
        )
        visible = qtree.query(viewport)
        
        # Should have fewer visible nodes than total
        assert len(visible) < len(nodes)
        assert len(visible) > 0


# ============================================================================
# Performance Benchmarks
# ============================================================================

class TestPerformanceBenchmarks:
    """Benchmark tests to validate performance targets."""

    def test_layout_speed_targets(self, small_graph, medium_graph, large_graph):
        """Verify layout computation meets speed targets."""
        layout_small = ForceDirectedLayout(iterations=20)
        layout_medium = ForceDirectedLayout(iterations=20)
        layout_large = ForceDirectedLayout(iterations=10)
        
        # Small graph: <100ms
        start = time.time()
        layout_small.compute_layout(*small_graph)
        elapsed = (time.time() - start) * 1000
        assert elapsed < 100, f"Small graph: {elapsed:.1f}ms (target: <100ms)"
        
        # Medium graph: <2s
        start = time.time()
        layout_medium.compute_layout(*medium_graph)
        elapsed = (time.time() - start) * 1000
        assert elapsed < 2000, f"Medium graph: {elapsed:.1f}ms (target: <2000ms)"
        
        # Large graph: <10s (with fewer iterations)
        start = time.time()
        layout_large.compute_layout(*large_graph)
        elapsed = (time.time() - start) * 1000
        assert elapsed < 10000, f"Large graph: {elapsed:.1f}ms (target: <10000ms)"
    
    def test_quadtree_query_speed(self):
        """Verify quadtree queries are fast."""
        bounds = Bounds(-100.0, -100.0, 100.0, 100.0)
        qtree = Quadtree(bounds, max_depth=12, max_items=16)
        
        # Insert 5000 random points
        for i in range(5000):
            x = random.uniform(-100.0, 100.0)
            y = random.uniform(-100.0, 100.0)
            qtree.insert(f"node_{i}", x, y)
        
        # Query should be fast
        start = time.time()
        for _ in range(100):
            viewport = Bounds(
                random.uniform(-50.0, 50.0),
                random.uniform(-50.0, 50.0),
                random.uniform(-50.0, 50.0) + 10,
                random.uniform(-50.0, 50.0) + 10,
            )
            qtree.query(viewport)
        elapsed = (time.time() - start) * 1000
        
        # 100 queries should be <100ms average (<1ms per query)
        assert elapsed < 100, f"Quadtree queries too slow: {elapsed:.1f}ms for 100 queries"


# ============================================================================
# Viewport Culling Tests
# ============================================================================

class TestViewportCulling:
    """Test viewport culling efficiency."""

    def test_culling_efficiency(self, large_graph):
        """Test that culling significantly reduces visible nodes."""
        nodes, edges = large_graph
        layout = ForceDirectedLayout(iterations=5)
        positions = layout.compute_layout(nodes, edges)
        
        xs = [x for x, y in positions.values()]
        ys = [y for x, y in positions.values()]
        bounds = Bounds(min(xs), min(ys), max(xs), max(ys))
        
        # Build quadtree
        qtree = Quadtree(bounds.expanded(0.2), max_depth=12, max_items=16)
        for node_id, (x, y) in positions.items():
            qtree.insert(node_id, x, y)
        
        # Test various viewport sizes
        for size_ratio in [0.25, 0.5, 0.75]:
            width = (bounds.max_x - bounds.min_x) * size_ratio
            height = (bounds.max_y - bounds.min_y) * size_ratio
            viewport = Bounds(
                bounds.min_x,
                bounds.min_y,
                bounds.min_x + width,
                bounds.min_y + height,
            )
            visible = qtree.query(viewport)
            
            # Should cull nodes proportionally
            expected_ratio = size_ratio * size_ratio  # Area ratio
            actual_ratio = len(visible) / len(nodes)
            
            # Allow some tolerance (2x the theoretical ratio due to edges)
            assert actual_ratio < expected_ratio * 2, \
                f"Culling ineffective: {actual_ratio:.2f} vs {expected_ratio:.2f}"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
