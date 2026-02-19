"""Simple Quadtree spatial index for viewport culling.

Provides a minimal 2D spatial index to quickly determine which nodes
fall inside the current viewport. Designed for use within the Plotly
Dash dashboard to cull nodes/edges that are off-screen when rendering
large graphs.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Bounds:
    """Axis-aligned bounding box."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def contains_point(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def intersects(self, other: "Bounds") -> bool:
        return not (
            other.max_x < self.min_x
            or other.min_x > self.max_x
            or other.max_y < self.min_y
            or other.min_y > self.max_y
        )

    def expanded(self, padding: float) -> "Bounds":
        return Bounds(
            self.min_x - padding,
            self.min_y - padding,
            self.max_x + padding,
            self.max_y + padding,
        )

    def width(self) -> float:
        return self.max_x - self.min_x

    def height(self) -> float:
        return self.max_y - self.min_y


class QuadtreeNode:
    """Internal quadtree node."""

    def __init__(
        self,
        bounds: Bounds,
        depth: int,
        max_depth: int,
        max_items: int,
    ) -> None:
        self.bounds = bounds
        self.depth = depth
        self.max_depth = max_depth
        self.max_items = max_items
        self.points: List[Tuple[str, float, float]] = []
        self.children: List[QuadtreeNode] = []

    def insert(self, node_id: str, x: float, y: float) -> bool:
        if not self.bounds.contains_point(x, y):
            return False

        if self.children:
            return any(child.insert(node_id, x, y) for child in self.children)

        self.points.append((node_id, x, y))

        if len(self.points) > self.max_items and self.depth < self.max_depth:
            self._subdivide()
        return True

    def query(self, bounds: Bounds, results: List[str]) -> None:
        if not self.bounds.intersects(bounds):
            return

        for node_id, x, y in self.points:
            if bounds.contains_point(x, y):
                results.append(node_id)

        for child in self.children:
            child.query(bounds, results)

    def _subdivide(self) -> None:
        mid_x = (self.bounds.min_x + self.bounds.max_x) / 2.0
        mid_y = (self.bounds.min_y + self.bounds.max_y) / 2.0

        self.children = [
            QuadtreeNode(
                Bounds(self.bounds.min_x, self.bounds.min_y, mid_x, mid_y),
                self.depth + 1,
                self.max_depth,
                self.max_items,
            ),
            QuadtreeNode(
                Bounds(mid_x, self.bounds.min_y, self.bounds.max_x, mid_y),
                self.depth + 1,
                self.max_depth,
                self.max_items,
            ),
            QuadtreeNode(
                Bounds(self.bounds.min_x, mid_y, mid_x, self.bounds.max_y),
                self.depth + 1,
                self.max_depth,
                self.max_items,
            ),
            QuadtreeNode(
                Bounds(mid_x, mid_y, self.bounds.max_x, self.bounds.max_y),
                self.depth + 1,
                self.max_depth,
                self.max_items,
            ),
        ]

        existing_points = self.points
        self.points = []
        for node_id, x, y in existing_points:
            for child in self.children:
                if child.insert(node_id, x, y):
                    break


class Quadtree:
    """Public quadtree wrapper for easy insertion and querying."""

    def __init__(
        self,
        bounds: Bounds,
        max_depth: int = 8,
        max_items: int = 32,
    ) -> None:
        self.root = QuadtreeNode(bounds, depth=0, max_depth=max_depth, max_items=max_items)

    def insert(self, node_id: str, x: float, y: float) -> bool:
        return self.root.insert(node_id, x, y)

    def query(self, bounds: Bounds) -> List[str]:
        results: List[str] = []
        self.root.query(bounds, results)
        return results

    def __len__(self) -> int:
        counter: List[int] = []
        self._count(self.root, counter)
        return sum(counter)

    def _count(self, node: QuadtreeNode, counter: List[int]) -> None:
        counter.append(len(node.points))
        for child in node.children:
            self._count(child, counter)
