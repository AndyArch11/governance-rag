"""Tests for citation graph Dash callback behaviour without a running server."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest


class FakeCitationVisualiser:
    """In-memory citation visualiser used to exercise callback orchestration."""

    PERSONA_FILTERS = {
        "supervisor": SimpleNamespace(venue_types=["journal"]),
        "examiner": SimpleNamespace(venue_types=["journal", "conference"]),
    }
    VENUE_RANK_COLOURS = {"Q1": "#123456"}
    SOURCE_TRUSTWORTHINESS_COLOURS = {"crossref": "#654321"}

    def __init__(self) -> None:
        self.db_path = Path("/tmp/academic_citation_graph.db")
        self.load_graph_calls: list[dict] = []
        self.export_calls: list[dict] = []
        self._graph = nx.Graph()
        self._primary_docs = set()
        self._graph.add_node(
            "reference-1",
            title="Research Methods",
            authors=["Ada Example", "Ben Example"],
            year=2024,
            venue_name="Journal of Testing",
            venue_rank="Q1",
            impact_factor=3.2,
            quality_score=0.9,
            citation_count=12,
            source="crossref",
            confidence=0.95,
            reference_type="academic",
            oa_available=True,
            link_status="available",
            doi="10.1000/example",
        )

    def load_graph(self, **kwargs):
        self.load_graph_calls.append(kwargs)
        return self._graph

    def create_plotly_figure(self, **kwargs):
        return {"figure": kwargs}

    def export_citations(self, **kwargs):
        self.export_calls.append(kwargs)
        return "title,year\nResearch Methods,2024\n"


@pytest.fixture()
def callback_module(monkeypatch):
    """Load callback functions with an identity registration decorator."""
    monkeypatch.setattr(
        sys.modules["dash"],
        "callback",
        lambda *args, **kwargs: lambda callback_function: callback_function,
    )
    module = importlib.import_module("scripts.ui.academic.citation_graph_callbacks")
    return importlib.reload(module)


@pytest.fixture()
def visualiser(callback_module, monkeypatch):
    """Inject one stable visualiser instance into every callback test."""
    instance = FakeCitationVisualiser()
    monkeypatch.setattr(callback_module, "get_citation_viz", lambda: instance)
    return instance


def test_persona_filters_use_configured_or_default_venue_types(callback_module, visualiser):
    """Persona selection returns configured venue types and safe defaults."""
    assert callback_module.update_filters_from_persona("examiner") == [["journal", "conference"]]
    assert callback_module.update_filters_from_persona("unknown") == [["journal"]]


def test_graph_callback_forwards_active_filters(callback_module, visualiser):
    """Graph refresh forwards selected filters to the visualiser unchanged."""
    result = callback_module.update_citation_graph(
        1,
        "examiner",
        "circular",
        ["available"],
        ["academic"],
        ["journal"],
        ["Q1"],
        ["crossref"],
    )

    assert result == {"figure": {"layout": "circular", "width": 1200, "height": 800}}
    assert visualiser.load_graph_calls == [
        {
            "persona": "examiner",
            "link_statuses": ["available"],
            "reference_types": ["academic"],
            "venue_types": ["journal"],
            "venue_ranks": ["Q1"],
            "sources": ["crossref"],
        }
    ]


def test_graph_callback_forwards_three_dimensional_layout(callback_module, visualiser):
    result = callback_module.update_citation_graph(
        1,
        "supervisor",
        "three_d",
        [],
        [],
        [],
        [],
        [],
    )

    assert result == {"figure": {"layout": "three_d", "width": 1200, "height": 800}}


def test_citation_graph_layout_selector_includes_three_dimensional_option(monkeypatch, tmp_path):
    import dash
    from scripts.ui.academic.citation_graph_viz import CitationGraphViz

    class ComponentFactory:
        def __getattr__(self, component_name):
            def build_component(*children, **properties):
                component_children = (
                    children[0]
                    if len(children) == 1 and isinstance(children[0], list)
                    else list(children)
                )
                return {
                    "component": component_name,
                    "children": component_children,
                    "props": properties,
                }

            return build_component

    factory = ComponentFactory()
    monkeypatch.setattr(dash, "html", factory)
    monkeypatch.setattr(dash, "dcc", factory)
    layout = CitationGraphViz(tmp_path / "citation.sqlite").create_dash_layout()

    def find_component(component, component_id):
        if isinstance(component, dict):
            if component.get("props", {}).get("id") == component_id:
                return component
            return find_component(component.get("children", []), component_id)
        if isinstance(component, list):
            for child in component:
                found = find_component(child, component_id)
                if found is not None:
                    return found
        return None

    layout_dropdown = find_component(layout, "citation-layout-dropdown")
    assert {
        "label": "3D Force-Directed",
        "value": "three_d",
    } in layout_dropdown["props"]["options"]


def test_graph_callback_loads_selected_combined_thesis_graph(
    callback_module, visualiser, monkeypatch
):
    graph = nx.DiGraph()
    graph.add_node("thesis-root", node_type="thesis", layer="thesis")
    graph.add_node("reference-1", node_type="reference", layer="citation")
    graph.add_edge("thesis-root", "reference-1", relation="cites", layer="bridge")
    calls = []

    def fake_load_academic_graph(*args, **kwargs):
        calls.append((args, kwargs))
        return graph

    monkeypatch.setattr(callback_module, "load_academic_graph", fake_load_academic_graph)

    result = callback_module.update_citation_graph(
        1,
        "supervisor",
        "hierarchical",
        ["available"],
        ["academic"],
        ["journal"],
        ["Q1"],
        ["crossref"],
        "thesis_and_references",
        "thesis-one",
    )

    assert result == {"figure": {"layout": "hierarchical", "width": 1200, "height": 800}}
    assert calls[0][0][0] == visualiser.db_path.parent / "thesis_graphs" / "registry.sqlite"
    assert calls[0][0][2] == "thesis-one"
    assert calls[0][1]["mode"] == "thesis_and_references"
    assert calls[0][1]["reference_types"] == ["academic"]
    assert visualiser._graph is graph
    assert visualiser._primary_docs == {"thesis-root"}


def test_node_details_render_for_selected_reference(callback_module, visualiser):
    """A selected graph node renders its citation details without a server."""
    details = callback_module.display_node_details({"points": [{"customdata": "reference-1"}]})

    assert details is not None


def test_citation_graph_three_dimensional_layout_renders_node_ids(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import scripts.ui.academic.citation_graph_viz as citation_graph_viz
    from scripts.ui.academic.citation_graph_viz import CitationGraphViz

    class FakeFigure:
        def __init__(self):
            self.data = []
            self.layout = SimpleNamespace()
            self.annotations = []

        def add_trace(self, trace):
            self.data.append(trace)

        def add_annotation(self, **kwargs):
            self.annotations.append(kwargs)

        def update_layout(self, **kwargs):
            self.layout = SimpleNamespace(**kwargs)

    def fake_scatter3d(**kwargs):
        return SimpleNamespace(type="scatter3d", **kwargs)

    monkeypatch.setattr(
        citation_graph_viz,
        "go",
        SimpleNamespace(Figure=FakeFigure, Scatter3d=fake_scatter3d),
    )

    visualiser = CitationGraphViz(tmp_path / "citation_graph.sqlite")
    visualiser._graph = nx.DiGraph(mode="references")
    visualiser._graph.add_node(
        "document-1",
        node_type="document",
        title="Primary thesis",
        source="document",
    )
    visualiser._graph.add_node(
        "reference-1",
        node_type="reference",
        title="Research Methods",
        source="crossref",
        link_status="available",
        quality_score=0.9,
    )
    visualiser._graph.add_edge("document-1", "reference-1", relation="cites")
    visualiser._primary_docs = {"document-1"}

    figure = visualiser.create_plotly_figure(layout="three_d", width=640, height=480)
    node_traces = [
        trace for trace in figure.data if trace.type == "scatter3d" and trace.mode == "markers"
    ]

    assert node_traces
    assert {node_id for trace in node_traces for node_id in trace.customdata} == {
        "document-1",
        "reference-1",
    }
    assert figure.layout.scene["xaxis"]["visible"] is False
    assert all(len(trace.x) == len(trace.y) == len(trace.z) for trace in node_traces)


def test_citation_graph_three_dimensional_layout_limits_large_graphs(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import scripts.ui.academic.citation_graph_viz as citation_graph_viz
    from scripts.ui.academic.citation_graph_viz import CitationGraphViz

    class FakeFigure:
        def __init__(self):
            self.annotations = []

        def add_annotation(self, **kwargs):
            self.annotations.append(kwargs)

        def update_layout(self, **kwargs):
            self.layout = SimpleNamespace(**kwargs)

    class FailingLayout:
        def __init__(self, *args, **kwargs):
            raise AssertionError("3D force layout must not run for oversized graphs")

    monkeypatch.setattr(citation_graph_viz, "go", SimpleNamespace(Figure=FakeFigure))
    monkeypatch.setattr(citation_graph_viz, "ForceDirected3DLayout", FailingLayout)
    visualiser = CitationGraphViz(tmp_path / "citation_graph.sqlite")
    visualiser._graph = nx.DiGraph()
    visualiser._graph.add_nodes_from(
        f"node-{index}" for index in range(citation_graph_viz.THREE_D_MAX_NODES + 1)
    )

    figure = visualiser.create_plotly_figure(layout="three_d")

    assert "limited to 750 nodes" in figure.annotations[0]["text"]


def test_export_callback_returns_csv_download_payload(callback_module, visualiser):
    """CSV export forwards filters and returns a timestamped download payload."""
    result = callback_module.export_citations_csv(
        1,
        "examiner",
        ["available"],
        ["academic"],
        ["journal"],
        ["Q1"],
        ["crossref"],
    )

    assert result["content"] == "title,year\nResearch Methods,2024\n"
    assert result["filename"].startswith("citations_export_")
    assert result["filename"].endswith(".csv")
    assert visualiser.export_calls == [
        {
            "persona": "examiner",
            "link_statuses": ["available"],
            "reference_types": ["academic"],
            "venue_types": ["journal"],
            "venue_ranks": ["Q1"],
            "sources": ["crossref"],
        }
    ]
