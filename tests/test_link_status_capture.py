#!/usr/bin/env python3
"""
Test that link_status is captured from provider resolution and used in quality scoring.
"""

import pytest

from scripts.ingest.academic.graph import CitationGraph
from scripts.ingest.academic.providers.base import Reference, ReferenceStatus


def test_link_status_captured_in_metadata():
    """Test that link_status from Reference gets into metadata dict passed to graph."""
    graph = CitationGraph()

    # Create a reference with different link statuses
    test_cases = [
        ("available", 1.0),
        ("stale_404", 0.3),
        ("stale_timeout", 0.6),
        ("stale_moved", 0.8),
        ("unresolved", 0.4),
    ]

    for link_status, expected_link_score in test_cases:
        metadata = {
            "citation": "Test citation",
            "title": "Test Title",
            "source": "crossref",
            "link_status": link_status,
            "confidence": 0.9,
        }

        # Add reference - this should compute quality_score using link_status
        graph.add_reference("ref_001", metadata)

        # Check that link_status was captured in the node
        node = graph.nodes["ref_001"]
        assert (
            node.link_status == link_status
        ), f"Expected link_status={link_status}, got {node.link_status}"

        # Quality score should include link_status contribution
        # With crossref (0.9) and link_score, confidence (0.9):
        # score = 0.9*0.3 + 0.5*0.2 + 0*0.2 + link_score*0.2 + 0.9*0.1
        #       = 0.27 + 0.1 + 0 + link_score*0.2 + 0.09
        #       = 0.46 + link_score*0.2
        expected_score = 0.46 + expected_link_score * 0.2

        assert node.quality_score == pytest.approx(
            expected_score, abs=0.01
        ), f"link_status={link_status}: expected quality_score≈{expected_score}, got {node.quality_score}"

        # Clean up for next iteration
        graph.nodes.clear()
        graph.edges.clear()


def test_link_status_affects_quality_score():
    """Test that different link statuses produce different quality scores."""
    graph = CitationGraph()

    # Base metadata (same for all)
    base_metadata = {
        "citation": "Test citation",
        "title": "Test Title",
        "source": "crossref",
        "confidence": 0.9,
    }

    scores = {}

    # Test each link status
    for i, link_status in enumerate(
        ["available", "stale_404", "stale_timeout", "stale_moved", "unresolved"]
    ):
        metadata = base_metadata.copy()
        metadata["link_status"] = link_status

        ref_id = f"ref_{i:03d}"
        graph.add_reference(ref_id, metadata)
        scores[link_status] = graph.nodes[ref_id].quality_score

    # Scores should reflect link_status differences
    assert scores["available"] > scores["stale_404"], "available should score higher than stale_404"
    assert (
        scores["stale_moved"] > scores["stale_timeout"]
    ), "stale_moved should score higher than stale_timeout"
    assert (
        scores["available"] > scores["unresolved"]
    ), "available should score higher than unresolved"

    print(f"\nQuality scores by link_status:\n{scores}")


def test_missing_link_status_defaults_to_available():
    """Test that missing link_status field defaults to 'available' (score 1.0)."""
    graph = CitationGraph()

    metadata = {
        "citation": "Test citation",
        "title": "Test Title",
        "source": "crossref",
        "confidence": 0.9,
        # No link_status field
    }

    graph.add_reference("ref_001", metadata)

    node = graph.nodes["ref_001"]
    # Should default to "available" when stored on node
    assert node.link_status == "available"

    # But quality score uses 0.7 default from _compute_quality_score since metadata dict didn't have it
    # score = 0.9*0.3 + 0.5*0.2 + 0*0.2 + 0.7*0.2 + 0.9*0.1 = 0.27 + 0.1 + 0 + 0.14 + 0.09 = 0.6
    expected_score = 0.6
    assert node.quality_score == pytest.approx(expected_score, abs=0.01)


if __name__ == "__main__":
    test_link_status_captured_in_metadata()
    test_link_status_affects_quality_score()
    test_missing_link_status_defaults_to_available()
    print("✓ All tests passed!")
