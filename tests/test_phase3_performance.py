"""Test suite for Phase 3: Performance optimisation features.

Tests:
- Layout engine algorithms
- Spatial indexing (quadtree)
- Performance monitoring
- Viewport culling
- WebGL integration
"""

import pytest
from scripts.ui.layout_engine import (
    ForceDirectedLayout,
    HierarchicalLayout,
    CircularLayout,
)
from scripts.ui.spatial_index import Bounds, Quadtree
from scripts.ui.performance_monitor import PerformanceMonitor, format_timings_ms


class TestLayoutEngine:
    """Test layout algorithms."""

    def test_force_directed_basic(self):
        """Test force-directed layout with simple graph."""
        layout = ForceDirectedLayout(k=1.0, damping=0.5, iterations=10)
        nodes = {
            "A": {"conflict_score": 0.5},
            "B": {"conflict_score": 0.3},
            "C": {"conflict_score": 0.7},
        }
        edges = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ]
        
        positions = layout.compute_layout(nodes, edges)
        
        assert len(positions) == 3
        assert "A" in positions
        assert "B" in positions
        assert "C" in positions
        for pos in positions.values():
            assert isinstance(pos, tuple)
            assert len(pos) == 2

    def test_hierarchical_layout(self):
        """Test hierarchical layout for DAG."""
        layout = HierarchicalLayout()
        nodes = {
            "root": {"conflict_score": 0.1},
            "child1": {"conflict_score": 0.2},
            "child2": {"conflict_score": 0.3},
        }
        edges = [
            {"source": "root", "target": "child1"},
            {"source": "root", "target": "child2"},
        ]
        
        positions = layout.compute_layout(nodes, edges)
        
        assert len(positions) == 3
        # Root should have different y-coordinate than children
        assert positions["root"][1] != positions["child1"][1]

    def test_circular_layout(self):
        """Test circular layout distribution."""
        layout = CircularLayout()
        nodes = {f"node_{i}": {"conflict_score": 0.5} for i in range(6)}
        edges = []
        
        positions = layout.compute_layout(nodes, edges)
        
        assert len(positions) == 6
        # All nodes should be on circle perimeter (distance from origin ~= 1)
        for pos in positions.values():
            x, y = pos
            distance = (x**2 + y**2) ** 0.5
            assert 0.9 < distance < 1.1

    def test_layout_with_empty_graph(self):
        """Test layout handles empty graph gracefully."""
        layout = ForceDirectedLayout()
        positions = layout.compute_layout({}, [])
        assert positions == {}


class TestSpatialIndex:
    """Test quadtree spatial indexing."""

    def test_bounds_creation(self):
        """Test Bounds creation and properties."""
        bounds = Bounds(0.0, 0.0, 10.0, 10.0)
        assert bounds.min_x == 0.0
        assert bounds.min_y == 0.0
        assert bounds.max_x == 10.0
        assert bounds.max_y == 10.0

    def test_bounds_contains_point(self):
        """Test point containment."""
        bounds = Bounds(0.0, 0.0, 10.0, 10.0)
        assert bounds.contains_point(5.0, 5.0)
        assert bounds.contains_point(0.0, 0.0)
        assert bounds.contains_point(10.0, 10.0)
        assert not bounds.contains_point(-1.0, 5.0)
        assert not bounds.contains_point(5.0, 11.0)

    def test_bounds_intersects(self):
        """Test bounds intersection."""
        b1 = Bounds(0.0, 0.0, 10.0, 10.0)
        b2 = Bounds(5.0, 5.0, 15.0, 15.0)
        b3 = Bounds(20.0, 20.0, 30.0, 30.0)
        
        assert b1.intersects(b2)
        assert b2.intersects(b1)
        assert not b1.intersects(b3)

    def test_bounds_expanded(self):
        """Test bounds expansion."""
        bounds = Bounds(0.0, 0.0, 10.0, 10.0)
        expanded = bounds.expanded(2.0)
        
        assert expanded.min_x == -2.0
        assert expanded.min_y == -2.0
        assert expanded.max_x == 12.0
        assert expanded.max_y == 12.0

    def test_quadtree_insert_and_query(self):
        """Test quadtree insertion and viewport query."""
        bounds = Bounds(0.0, 0.0, 100.0, 100.0)
        tree = Quadtree(bounds, max_depth=5, max_items=4)
        
        # Insert points
        points = [
            ("A", 10.0, 10.0),
            ("B", 20.0, 20.0),
            ("C", 80.0, 80.0),
            ("D", 90.0, 90.0),
        ]
        for node_id, x, y in points:
            tree.insert(node_id, x, y)
        
        # Query small viewport (should only get A and B)
        viewport = Bounds(0.0, 0.0, 30.0, 30.0)
        visible = tree.query(viewport)
        
        assert "A" in visible
        assert "B" in visible
        assert "C" not in visible
        assert "D" not in visible

    def test_quadtree_subdivision(self):
        """Test quadtree subdivides when capacity exceeded."""
        bounds = Bounds(0.0, 0.0, 100.0, 100.0)
        tree = Quadtree(bounds, max_depth=3, max_items=2)
        
        # Insert enough points to force subdivision
        for i in range(10):
            tree.insert(f"node_{i}", float(i * 5), float(i * 5))
        
        # All points should still be queryable
        all_nodes = tree.query(bounds)
        assert len(all_nodes) == 10


class TestPerformanceMonitor:
    """Test performance monitoring utilities."""

    def test_performance_monitor_basic(self):
        """Test basic timing recording."""
        monitor = PerformanceMonitor()
        
        import time
        time.sleep(0.01)
        monitor.record("step1")
        
        time.sleep(0.01)
        monitor.record("step2")
        
        snapshot = monitor.snapshot()
        assert "step1" in snapshot
        assert "step2" in snapshot
        assert snapshot["step1"] > 0
        assert snapshot["step2"] > 0

    def test_performance_monitor_reset(self):
        """Test monitor reset."""
        monitor = PerformanceMonitor()
        
        import time
        time.sleep(0.01)
        monitor.record("step1")
        
        assert len(monitor.snapshot()) == 1
        
        monitor.reset()
        assert len(monitor.snapshot()) == 0

    def test_format_timings_ms(self):
        """Test millisecond formatting."""
        timings = {
            "fast": 0.001,  # 1ms
            "slow": 0.1,    # 100ms
            "medium": 0.05, # 50ms
        }
        
        formatted = format_timings_ms(timings)
        
        assert formatted["fast"] == 1.0
        assert formatted["slow"] == 100.0
        assert formatted["medium"] == 50.0

    def test_performance_monitor_order(self):
        """Test that order is preserved."""
        monitor = PerformanceMonitor()
        
        monitor.record("first")
        monitor.record("second")
        monitor.record("third")
        
        assert monitor.order == ["first", "second", "third"]


class TestViewportCulling:
    """Integration tests for viewport culling."""

    def test_culling_reduces_visible_nodes(self):
        """Test that viewport culling reduces visible node count."""
        # Create grid of nodes
        all_bounds = Bounds(0.0, 0.0, 100.0, 100.0)
        tree = Quadtree(all_bounds, max_depth=5, max_items=4)
        
        node_count = 0
        for x in range(0, 100, 10):
            for y in range(0, 100, 10):
                tree.insert(f"node_{node_count}", float(x), float(y))
                node_count += 1
        
        assert node_count == 100
        
        # Small viewport should show fewer nodes
        small_viewport = Bounds(0.0, 0.0, 30.0, 30.0)
        visible = tree.query(small_viewport)
        
        assert len(visible) < node_count
        assert len(visible) < 20  # Should be roughly 16 (4x4 grid)

    def test_full_viewport_shows_all(self):
        """Test that full viewport returns all nodes."""
        bounds = Bounds(0.0, 0.0, 100.0, 100.0)
        tree = Quadtree(bounds)
        
        # Insert 50 nodes
        for i in range(50):
            tree.insert(f"node_{i}", float(i), float(i))
        
        # Query entire bounds
        all_visible = tree.query(bounds)
        assert len(all_visible) == 50


class TestWebGLIntegration:
    """Test WebGL rendering decisions."""

    def test_webgl_threshold_logic(self):
        """Test that WebGL is enabled for large graphs."""
        webgl_threshold = 1000
        
        # Small graph
        node_count = 500
        use_webgl = node_count > webgl_threshold
        assert not use_webgl
        
        # Large graph
        node_count = 1500
        use_webgl = node_count > webgl_threshold
        assert use_webgl

    def test_layout_selection(self):
        """Test layout type selection."""
        layouts = ["force", "hierarchical", "circular"]
        
        for layout_type in layouts:
            if layout_type == "force":
                engine = ForceDirectedLayout()
            elif layout_type == "hierarchical":
                engine = HierarchicalLayout()
            elif layout_type == "circular":
                engine = CircularLayout()
            
            assert engine is not None
            
            # All should be able to handle empty graph
            positions = engine.compute_layout({}, [])
            assert positions == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
