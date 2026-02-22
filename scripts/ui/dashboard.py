"""
Memory-Efficient Plotly Dash Dashboard for Consistency Graph Visualisation.

Provides a web-based interface for exploring consistency graphs of governance documents,
with lazy loading and pagination to minimise memory footprint.

Features:
- Interactive graph visualisation with Plotly (GPU-accelerated rendering)
- Lazy-loaded document details (not all data in memory at startup)
- Risk matrix heatmaps with pagination
- Semantic drift analysis with on-demand computation
- Dynamic cluster exploration
- Threshold adjustment with graph rebuilding
- Document search with semantic embeddings (cached locally)

Architecture:
- Loads graph metadata first; full graph data on-demand
- Chunked batch loading for large result sets
- Plotly for rendering with optional WebGL for large graphs

TODO:
  - Mouse tooltips
  - Mouse busy indicators
  - Disabled or hidden tabs when no relevant content is available
  - Draggable nodes with dynamic layout updates (future)
  - Identify if GPU is available and automatically enable WebGL for large graphs
  - Redis caching (future consideration)
  - UI for ingest and consistency_graph configuration
  - With academic references, clicking a consistency graph node does not select the right document, and details tab of Select Document view shows "Document not found"
  - Select Document - Details tab does not show connected nodes & relationships or document text.
  - Include timestamps with performance metrics and resource usage monitoring for better debugging and optimisation insights.

Usage:
    python -m scripts.ui.dashboard

    Access at http://localhost:8050

Configuration:
    CHROMA_PATH: <project_root>/rag_data/chromadb
    DOC_COLLECTION_NAME: governance_docs_documents
    GRAPH_SQLITE: <project_root>/rag_data/consistency_graphs/consistency_graph.sqlite
"""

import asyncio
import cProfile
import io
import json
import logging
import math
import os
import pstats
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import dash
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from dash import (
    ALL,
    Dash,
    Input,
    Output,
    State,
    callback,
    dcc,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

# Ensure project root is in sys.path for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.rag.benchmark_manager import BenchmarkManager
from scripts.rag.generate import answer as rag_answer
from scripts.rag.rag_config import RAGConfig
from scripts.ui.academic.citation_graph_callbacks import register_citation_graph_callbacks
from scripts.ui.academic.citation_graph_viz import get_citation_viz
from scripts.ui.export_manager import ExportManager
from scripts.ui.word_cloud_provider import get_word_cloud_data, get_word_cloud_stats
from scripts.utils.db_factory import get_default_vector_path, get_vector_client, get_cache_client

# Logger setup
logger = logging.getLogger(__name__)

# Centralised backend selection (Chroma preferred)
PersistentClient, USING_SQLITE = get_vector_client(prefer="chroma")

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None

from scripts.consistency_graph.consistency_config import get_consistency_config

# Import graph builder and utilities
from scripts.consistency_graph.graph_cache_manager import GraphCacheManager
from scripts.consistency_graph.graph_filter import GraphFilter
from scripts.ingest.ingest_config import get_ingest_config
from scripts.ui.academic.academic_references import (
    AcademicReferences,
    _get_global_module,
    _set_global_module,
)
from scripts.ui.layout_engine import (
    CircularLayout,
    ForceDirectedLayout,
    HierarchicalLayout,
    compute_layout,
)
from scripts.ui.performance_monitor import PerformanceMonitor, format_timings_ms
from scripts.ui.spatial_index import Bounds, Quadtree
from scripts.utils.resource_monitor import ResourceMonitor

# Configuration
CONFIG = get_consistency_config()
INGEST_CONFIG = get_ingest_config()
LOGS_DIR = CONFIG.logs_dir
CHROMA_PATH = get_default_vector_path(Path(CONFIG.rag_data_path), USING_SQLITE)
DOC_COLLECTION_NAME = CONFIG.doc_collection_name
GRAPH_SQLITE = CONFIG.output_sqlite

# RAG query configuration
RAG_CONFIG = RAGConfig()
_QUERY_COLLECTION = None


def _get_query_collection():
    """Lazy-load and cache the RAG chunks collection for queries."""
    global _QUERY_COLLECTION
    if _QUERY_COLLECTION is None:
        chroma_path = get_default_vector_path(Path(RAG_CONFIG.rag_data_path), USING_SQLITE)
        client = PersistentClient(path=chroma_path)
        _QUERY_COLLECTION = client.get_collection(RAG_CONFIG.chunk_collection_name)
    return _QUERY_COLLECTION


# Pagination & Lazy Loading
NODES_PER_PAGE = 50
EDGES_PER_BATCH = 100
MAX_GRAPH_VISUALISATION_NODES = 300


def human_size(n: int) -> str:
    """Convert bytes to human-readable format."""
    units = ["B", "KB", "MB", "GB"]
    s = float(n)
    for u in units:
        if s < 1024.0:
            return f"{s:.1f} {u}"
        s /= 1024.0
    return f"{s:.1f} TB"


def get_display_name(node_data: Dict, node_id: str) -> str:
    """
    Extract display name for a node.
    For academic nodes, use summary (title); otherwise use node_id.
    """
    if not isinstance(node_data, dict):
        return node_id

    source_category = node_data.get("source_category", "")
    if source_category == "academic_reference":
        summary = node_data.get("summary", "")
        if summary:
            # Extract title from summary, clean up whitespace
            title_text = " ".join(summary.split())
            # Truncate if too long
            if len(title_text) > 100:
                return title_text[:97] + "..."
            else:
                return title_text

    # Fallback to node_id
    return node_id


from scripts.consistency_graph.sqlite_store import SQLiteGraphStore

# Initialise Dash app
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="Governance Consistency Graph Dashboard",
)

# Custom CSS for better styling
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: #f5f5f5;
                margin: 0;
                padding: 0;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 24px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .header h1 {
                margin: 0;
                font-size: 24px;
                font-weight: 600;
            }
            .header p {
                margin: 4px 0 0 0;
                font-size: 14px;
                opacity: 0.9;
            }
            .container {
                max-width: 1600px;
                margin: 0 auto;
                padding: 24px;
            }
            .card {
                background: white;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                padding: 20px;
                margin-bottom: 16px;
            }
            .tabs {
                display: flex;
                gap: 8px;
                margin-bottom: 16px;
                border-bottom: 2px solid #eee;
            }
            .tab-button {
                padding: 12px 20px;
                border: none;
                background: transparent;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                color: #666;
                border-bottom: 3px solid transparent;
                transition: all 0.2s;
            }
            .tab-button.active {
                color: #667eea;
                border-bottom-color: #667eea;
            }
            .tab-button:hover {
                color: #667eea;
            }
            .metric {
                display: inline-block;
                margin-right: 24px;
                margin-bottom: 8px;
            }
            .metric-value {
                font-size: 24px;
                font-weight: 600;
                color: #667eea;
            }
            .metric-label {
                font-size: 12px;
                color: #999;
                margin-top: 4px;
            }
            .filter-group {
                display: flex;
                gap: 16px;
                flex-wrap: wrap;
                margin-bottom: 16px;
            }
            .filter-group label {
                display: flex;
                flex-direction: column;
                gap: 4px;
                font-size: 12px;
                font-weight: 500;
                color: #666;
            }
            .filter-group input,
            .filter-group select {
                padding: 8px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }
            .filter-group input:focus,
            .filter-group select:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            .spinner {
                display: inline-block;
                width: 16px;
                height: 16px;
                border: 2px solid #f3f3f3;
                border-top: 2px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                vertical-align: middle;
                margin-left: 8px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .pagination {
                display: flex;
                gap: 8px;
                align-items: center;
                justify-content: center;
                margin: 16px 0;
            }
            .pagination button {
                padding: 8px 12px;
                border: 1px solid #ddd;
                background: white;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                transition: all 0.2s;
            }
            .pagination button:hover:not(:disabled) {
                background: #667eea;
                color: white;
                border-color: #667eea;
            }
            .pagination button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            button:disabled {
                opacity: 0.4;
                cursor: not-allowed !important;
                background: #f5f5f5 !important;
                color: #999 !important;
                border-color: #ddd !important;
            }
            button:not(:disabled):hover {
                background: #667eea !important;
                color: white !important;
                transition: all 0.2s;
            }
            .info-box {
                background: #e8f4f8;
                border-left: 4px solid #0288d1;
                padding: 12px 16px;
                border-radius: 4px;
                margin-bottom: 16px;
                font-size: 14px;
                color: #01579b;
            }
            .warning-box {
                background: #fff3e0;
                border-left: 4px solid #f57c00;
                padding: 12px 16px;
                border-radius: 4px;
                margin-bottom: 16px;
                font-size: 14px;
                color: #e65100;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

# Store for lazy-loaded data
graph_store = SQLiteGraphStore(GRAPH_SQLITE)

# Initialise graph filter (will be populated after graph loads)
graph_filter = None
filter_data = {
    "available_doc_types": [],
    "available_languages": [],
    "available_repositories": [],
    "filtered_nodes": set(),
}

# WebGL/Layout engine configuration
current_layout_type = "force"  # Default layout
layout_positions_cache = {}  # Cache computed positions
use_webgl_global = False  # Track WebGL usage
webgl_threshold = 1000  # Use WebGL for >1k nodes
# Viewport culling tuning
QUADTREE_MAX_DEPTH = 10
QUADTREE_MAX_ITEMS = 16
SPATIAL_PADDING = 0.1  # padding around computed bounds
VIEWPORT_EXPAND = 0.05  # slight viewport grow to avoid pop-in

# Performance monitoring defaults
PERF_ENABLED = True


def _compute_bounds(
    positions: Dict[str, Tuple[float, float]], node_ids: List[str], padding: float = SPATIAL_PADDING
) -> Bounds:
    """Compute bounding box for given node positions."""
    if not positions or not node_ids:
        return Bounds(-1.0, -1.0, 1.0, 1.0).expanded(padding)

    xs = []
    ys = []
    for nid in node_ids:
        x, y = positions.get(nid, (0.0, 0.0))
        xs.append(x)
        ys.append(y)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return Bounds(min_x, min_y, max_x, max_y).expanded(padding)


def _extract_viewport_bounds(relayout_data: Dict[str, Any], default_bounds: Bounds) -> Bounds:
    """Derive current viewport bounds from Plotly relayout data."""
    if not relayout_data:
        return default_bounds

    x0 = relayout_data.get("xaxis.range[0]") or relayout_data.get("xaxis.range", [None, None])[0]
    x1 = relayout_data.get("xaxis.range[1]") or relayout_data.get("xaxis.range", [None, None])[-1]
    y0 = relayout_data.get("yaxis.range[0]") or relayout_data.get("yaxis.range", [None, None])[0]
    y1 = relayout_data.get("yaxis.range[1]") or relayout_data.get("yaxis.range", [None, None])[-1]

    if None in (x0, x1, y0, y1):
        return default_bounds

    return Bounds(float(x0), float(y0), float(x1), float(y1))


def _load_domain_terminology_capitalisations() -> Dict[str, str]:
    """Load domain terminology for capitalisation lookup.

    Builds a mapping of lowercase -> properly capitalised domain terms
    (e.g., 'indigenous' -> 'Indigenous', 'aboriginal' -> 'Aboriginal').

    Returns:
        Dictionary mapping lowercase terms to their proper capitalisation
    """
    capitalisations: Dict[str, str] = {}

    try:
        from pathlib import Path

        from scripts.ingest.academic.phd_assessor import CAPITALISATION_OVERRIDES
        from scripts.ingest.ingest_config import get_ingest_config

        config = get_ingest_config()
        terminology_db_path = Path(config.rag_data_path) / "academic_terminology.db"

        if terminology_db_path.exists():
            import sqlite3

            conn = sqlite3.connect(str(terminology_db_path), check_same_thread=False)
            cursor = conn.cursor()

            # Query all domain terms from the database
            cursor.execute("SELECT DISTINCT term FROM domain_terms LIMIT 1000")
            rows = cursor.fetchall()

            # Build lowercase -> proper case mapping
            for (term,) in rows:
                if term:
                    capitalisations[term.lower()] = term

            conn.close()

        for term, override in CAPITALISATION_OVERRIDES.items():
            if term and override:
                capitalisations[term.lower()] = override
    except Exception as e:
        # Silently fail - fall back to generic capitalisation
        pass

    return capitalisations


# Cache domain terminology capitalisations at module level
_DOMAIN_TERM_CAPITALISATIONS = _load_domain_terminology_capitalisations()


def _capitalise_word(word: str) -> str:
    """Apply appropriate capitalisation to a word for display.

    Checks domain terminology database first, then applies fallback rules
    for words not in the terminology database.

    Args:
        word: The word (typically lowercase from database)

    Returns:
        Word with appropriate capitalisation applied
    """
    word_lower = word.lower()

    # First, check if this word exists in domain terminology with proper capitalisation
    if word_lower in _DOMAIN_TERM_CAPITALISATIONS:
        return _DOMAIN_TERM_CAPITALISATIONS[word_lower]

    # Fallback capitalisation rules for common terms
    CAPITALISE_TERMS = {
        "ai",
        "ml",
        "gpt",
        "nlp",
        "nlg",
        "rag",
        "llm",
        "llms",
        "gdpr",
        "okr",
        "kpi",
        "kpis",
        "ceo",
        "cfo",
        "cto",
        "coo",
        "ciso",
        "chro",
    }

    if word_lower in CAPITALISE_TERMS:
        return word.upper() if len(word) <= 4 else word.title()

    # For multi-word terms, title case
    if " " in word:
        return word.title()

    # Single words - return as-is (lowercase from database)
    return word


def _calculate_text_bbox(x: float, y: float, text: str, font_size: int) -> Dict[str, float]:
    """Calculate approximate bounding box for text element.

    Args:
        x: Horizontal position (centre)
        y: Vertical position (centre)
        text: The text string
        font_size: Font size in pixels

    Returns:
        Dictionary with bounds: {x_min, x_max, y_min, y_max}
    """
    # Estimate text width: ~0.6 character width in font units
    char_width = font_size * 0.6
    text_width = len(text) * char_width

    # Estimate text height: approximately font size
    text_height = font_size * 1.0

    # Add padding around text
    padding = font_size * 0.2

    return {
        "x_min": x - (text_width / 2) - padding,
        "x_max": x + (text_width / 2) + padding,
        "y_min": y - (text_height / 2) - padding,
        "y_max": y + (text_height / 2) + padding,
    }


def _bboxes_collide(bbox1: Dict[str, float], bbox2: Dict[str, float]) -> bool:
    """Check if two bounding boxes overlap.

    Args:
        bbox1: First bounding box {x_min, x_max, y_min, y_max}
        bbox2: Second bounding box {x_min, x_max, y_min, y_max}

    Returns:
        True if bounding boxes overlap, False otherwise
    """
    return not (
        bbox1["x_max"] < bbox2["x_min"]
        or bbox1["x_min"] > bbox2["x_max"]
        or bbox1["y_max"] < bbox2["y_min"]
        or bbox1["y_min"] > bbox2["y_max"]
    )


def _build_word_cloud_figure(word_data: List[Dict[str, Any]]) -> go.Figure:
    """Build a word cloud scatter plot with collision detection using golden-angle spiral.

    Places words on a golden-angle spiral, skipping positions until no collision
    is detected with previously placed words.

    TODO: Consider using wordcloud library to generate layout and then render with Plotly for better aesthetics and performance, especially for larger word clouds.
    """
    if not word_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No word frequency data available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 14, "color": "#999"},
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        return fig

    words = [entry.get("word", "") for entry in word_data]
    sizes = [entry.get("size", 0.0) for entry in word_data]
    freqs = [entry.get("frequency", 0) for entry in word_data]
    doc_counts = [entry.get("doc_count", 0) for entry in word_data]

    text_sizes = [max(10, int(10 + (size * 28))) for size in sizes]

    x_vals = []
    y_vals = []
    placed_bboxes = []  # Track bounding boxes of placed words

    golden_angle = 2.399963229728653
    min_radius = 0.5
    radius_step = 2

    for word_idx, (size, font_size) in enumerate(zip(sizes, text_sizes)):
        # Each word tries positions along an independent spiral
        # Higher-frequency words (earlier in list) get closer-in positions
        spiral_idx = 0
        max_attempts = 200
        placed = False

        while spiral_idx < max_attempts and not placed:
            # Position on this word's spiral
            angle = spiral_idx * golden_angle
            radius = min_radius + (spiral_idx * radius_step)

            x = radius * math.cos(angle)
            y = radius * math.sin(angle)

            # Check for collisions with previously placed words
            current_bbox = _calculate_text_bbox(x, y, words[word_idx], font_size)
            has_collision = any(
                _bboxes_collide(current_bbox, placed_bbox) for placed_bbox in placed_bboxes
            )

            if not has_collision:
                x_vals.append(x)
                y_vals.append(y)
                placed_bboxes.append(current_bbox)
                placed = True
            else:
                spiral_idx += 1

        # If we couldn't find a good position after max attempts, force place it
        if not placed:
            angle = spiral_idx * golden_angle
            radius = min_radius + (spiral_idx * radius_step)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            x_vals.append(x)
            y_vals.append(y)

    hover_text = [
        f"{word}<br>Frequency: {freq}<br>Doc count: {doc_count}"
        for word, freq, doc_count in zip(words, freqs, doc_counts)
    ]

    # Apply capitalisation rules for display
    display_words = [_capitalise_word(word) for word in words]

    fig = go.Figure(
        data=go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="text",
            text=display_words,
            textfont={"size": text_sizes, "color": "#2f2f2f"},
            hovertext=hover_text,
            hoverinfo="text",
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ============================================================================
# Callbacks
# ============================================================================


@callback(
    Output("layout-seed-store", "data"),
    Input("regenerate-layout-btn", "n_clicks"),
    prevent_initial_call=True,
)
def regenerate_layout_seed(n_clicks):
    """Generate a new random seed when regenerate button is clicked."""
    import random

    return random.randint(1, 10000)


@callback(
    Output("graph-container", "children"),
    Output("performance-store", "data"),
    Input("tab-graph", "n_clicks"),
    Input("graph-page-slider", "value"),
    Input("filter-state-store", "data"),
    Input("layout-selector", "value"),
    Input("webgl-toggle", "value"),
    Input("show-node-names-toggle", "value"),
    Input("show-unlinked-nodes-toggle", "value"),
    Input("nodes-per-page-selector", "value"),
    Input("selected-node-store", "data"),
    Input("layout-seed-store", "data"),
    State("min-conflict-slider", "value"),
    State("sim-threshold-slider", "value"),
)
def update_graph_tab(
    _,
    page,
    filter_state,
    layout_type,
    use_webgl_list,
    show_names_list,
    show_unlinked_list,
    nodes_per_page,
    selected_node,
    layout_seed,
    min_conflict,
    sim_threshold,
):
    """Update graph visualisation with layout engine, WebGL, and filtering."""
    # Note: relayout_data removed as it creates circular dependency (graph-figure is output of this callback)
    relayout_data = None

    # Check if nodes are loaded, if not try to load metadata (handles hot reload after graph rebuild)
    if graph_store.get_node_ids() == []:
        graph_store.load_metadata()
        if graph_store.get_node_ids() == []:
            return (
                html.Div(
                    [html.P("Loading graph...", style={"textAlign": "center", "color": "#999"})]
                ),
                None,
            )

    perf = PerformanceMonitor() if PERF_ENABLED else None

    try:
        # Get filtered nodes if filters are active
        all_nodes = graph_store.get_node_ids()
        if filter_state and filter_state.get("filtered_nodes"):
            all_nodes = filter_state.get("filtered_nodes", all_nodes)
        if perf:
            perf.record("filter_load")

        # Get paginated nodes from filtered list
        if not all_nodes:
            return (
                html.Div(
                    [
                        html.P(
                            "No nodes match filters.",
                            style={"textAlign": "center", "color": "#999"},
                        )
                    ]
                ),
                None,
            )

        # Use dynamic nodes per page (default to NODES_PER_PAGE if not set)
        effective_nodes_per_page = nodes_per_page if nodes_per_page else NODES_PER_PAGE

        # Calculate pagination for filtered nodes
        total_filtered = len(all_nodes)
        total_pages = max(0, (total_filtered - 1) // effective_nodes_per_page)

        # Get page slice (pagination is always consistent, regardless of show_unlinked toggle)
        start_idx = page * effective_nodes_per_page
        end_idx = start_idx + effective_nodes_per_page
        page_node_ids = all_nodes[start_idx:end_idx]

        if not page_node_ids:
            return (
                html.Div(
                    [
                        html.P(
                            "No nodes on this page.", style={"textAlign": "center", "color": "#999"}
                        )
                    ]
                ),
                None,
            )

        # Get node data for this page
        nodes_data = [
            (nid, graph_store.get_node(nid)) for nid in page_node_ids if graph_store.get_node(nid)
        ]

        # Get all edges to determine globally linked nodes
        edges_data = graph_store.get_edges()

        # Filter edges by relationship type if specified in filter state
        if filter_state and filter_state.get("relationship_types"):
            allowed_types = filter_state.get("relationship_types", [])
            if allowed_types:
                edges_data = [
                    edge
                    for edge in edges_data
                    if edge.get("relationship", "consistent") in allowed_types
                ]

        # Build set of linked nodes globally (from ALL edges, not just current page)
        globally_linked_node_ids = set()
        for edge in edges_data:
            globally_linked_node_ids.add(edge.get("source"))
            globally_linked_node_ids.add(edge.get("target"))

        if perf:
            perf.record("page_fetch")

        if not nodes_data:
            return (
                html.Div(
                    [html.P("No nodes loaded.", style={"textAlign": "center", "color": "#999"})]
                ),
                None,
            )

        # Extract node IDs for this page
        node_ids_set = {nid for nid, _ in nodes_data}
        page_node_ids = list(node_ids_set)  # Update to only valid node IDs
        original_page_size = len(page_node_ids)  # Save for stats display

        # Filter edges to only those connecting nodes on this page (for rendering)
        filtered_edges = [
            edge
            for edge in edges_data
            if edge.get("source") in node_ids_set and edge.get("target") in node_ids_set
        ]

        # Check show_unlinked toggle (this only affects rendering, not pagination)
        show_unlinked = show_unlinked_list and "show-unlinked" in show_unlinked_list

        # =====================================================
        # Layout Computation using Adaptive Layout Engine
        # =====================================================

        # Determine layout type (default to force-directed)
        layout_type = layout_type or "force"

        # Prepare node and edge data for layout engine
        nodes_dict = {nid: data for nid, data in nodes_data}
        edges_list = filtered_edges  # Already in dict format with 'source' and 'target'

        # Use provided seed or default
        layout_seed = layout_seed or 42

        # Compute layout positions using adaptive parameters
        try:
            if layout_type == "hierarchical":
                layout_engine = HierarchicalLayout(layer_gap=2.0, node_gap=1.0)
                positions = layout_engine.compute_layout(nodes_dict, edges_list)
            elif layout_type == "circular":
                layout_engine = CircularLayout()
                positions = layout_engine.compute_layout(nodes_dict, edges_list)
            else:
                # Force-directed with adaptive parameters based on graph size
                positions = compute_layout(
                    nodes_dict, edges_list, layout_type="force", adaptive=True, seed=layout_seed
                )
        except Exception as layout_error:
            print(f"Layout computation error: {layout_error}. Using fallback positions.")
            # Fallback to simple positions if layout fails
            positions = {nid: (i * 0.1, i * 0.1 % 1.0) for i, nid in enumerate(node_ids_set)}
        if perf:
            perf.record("layout_compute")

        # Build spatial index for viewport culling
        graph_bounds = _compute_bounds(positions, page_node_ids, padding=SPATIAL_PADDING)
        viewport_bounds = _extract_viewport_bounds(relayout_data or {}, graph_bounds).expanded(
            VIEWPORT_EXPAND
        )

        qtree = Quadtree(graph_bounds, max_depth=QUADTREE_MAX_DEPTH, max_items=QUADTREE_MAX_ITEMS)
        for nid in page_node_ids:
            x, y = positions.get(nid, (0.0, 0.0))
            qtree.insert(nid, x, y)

        visible_ids = set(qtree.query(viewport_bounds))
        if not visible_ids:
            # If no nodes intersect, fall back to full page to avoid empty renders
            visible_ids = set(page_node_ids)
        if perf:
            perf.record("cull")

        # Filter visible nodes based on show_unlinked toggle
        # If toggle is off, only show nodes that have at least one edge globally
        if not show_unlinked:
            visible_ids = visible_ids & globally_linked_node_ids

        visible_nodes = [(nid, nodes_dict[nid]) for nid in page_node_ids if nid in visible_ids]
        if not visible_nodes:
            if not show_unlinked:
                # No linked nodes on this page
                return (
                    html.Div(
                        [
                            html.P(
                                "No linked nodes on this page. Enable 'Show Unlinked Nodes' to see all nodes.",
                                style={"textAlign": "center", "color": "#999"},
                            )
                        ]
                    ),
                    None,
                )
            else:
                return (
                    html.Div(
                        [
                            html.P(
                                "No nodes in current viewport.",
                                style={"textAlign": "center", "color": "#999"},
                            )
                        ]
                    ),
                    None,
                )

        # Extract positions for visible nodes only
        node_x = [positions.get(nid, (0.0, 0.0))[0] for nid, _ in visible_nodes]
        node_y = [positions.get(nid, (0.0, 0.0))[1] for nid, _ in visible_nodes]

        # Build edge data with relationship status colouring and dynamic weights
        visible_ids_set = {nid for nid, _ in visible_nodes}

        # Status to colour mapping
        status_colours = {
            "consistent": "rgba(52, 211, 153, 0.7)",  # Green
            "duplicate": "rgba(59, 130, 246, 0.7)",  # Blue
            "partial_conflict": "rgba(251, 191, 36, 0.7)",  # Amber
            "conflict": "rgba(239, 68, 68, 0.7)",  # Red
        }

        # Emphasised colours for selected node edges
        status_colours_emphasised = {
            "consistent": "rgba(52, 211, 153, 1.0)",
            "duplicate": "rgba(59, 130, 246, 1.0)",
            "partial_conflict": "rgba(251, 191, 36, 1.0)",
            "conflict": "rgba(239, 68, 68, 1.0)",
        }

        # Group edges by relationship status and selection state
        edges_by_status = {
            status: {"x": [], "y": [], "widths": []} for status in status_colours.keys()
        }
        edges_by_status_selected = {
            status: {"x": [], "y": [], "widths": []} for status in status_colours.keys()
        }
        visible_edges_count = 0

        for edge in filtered_edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src in visible_ids_set and tgt in visible_ids_set:
                x0, y0 = positions[src]
                x1, y1 = positions[tgt]

                # Get relationship status (default to 'consistent' if missing)
                status = edge.get("relationship", "consistent")
                if status not in edges_by_status:
                    status = "consistent"

                # Check if this edge is connected to selected node
                is_selected_edge = selected_node and (src == selected_node or tgt == selected_node)

                # Compute dynamic line width based on max(similarity, severity)
                similarity = float(edge.get("similarity", 0.5))
                severity = float(edge.get("severity", 0.0))
                edge_weight = max(similarity, severity)
                # Scale to 0.5-4.0 px range, or 2-6 px for selected edges
                if is_selected_edge:
                    line_width = 2.0 + (edge_weight * 4.0)
                    edges_by_status_selected[status]["x"].extend([x0, x1, None])
                    edges_by_status_selected[status]["y"].extend([y0, y1, None])
                    edges_by_status_selected[status]["widths"].append(line_width)
                else:
                    line_width = 0.5 + (edge_weight * 3.5)
                    # Add to status group with width info in coordinates
                    edges_by_status[status]["x"].extend([x0, x1, None])
                    edges_by_status[status]["y"].extend([y0, y1, None])
                    edges_by_status[status]["widths"].append(line_width)

            visible_edges_count += 1

        # Create separate traces for each edge status (for colouring and legend)
        edge_traces = []

        # Add normal edges (potentially faded if selection is active)
        for status, colour in status_colours.items():
            if edges_by_status[status]["x"]:
                # Fade edges if there's a selection
                edge_opacity = 0.2 if selected_node else 0.7
                trace_colour = colour.replace("0.7)", f"{edge_opacity})")
                trace = go.Scatter(
                    x=edges_by_status[status]["x"],
                    y=edges_by_status[status]["y"],
                    mode="lines",
                    line=dict(width=1.0 if selected_node else 2.0, color=trace_colour),
                    hoverinfo="skip",
                    showlegend=True,
                    name=status.replace("_", " ").title(),
                    visible=True,
                )
                edge_traces.append(trace)

        # Add selected edges on top (full opacity, thicker)
        if selected_node:
            for status, colour in status_colours_emphasised.items():
                if edges_by_status_selected[status]["x"]:
                    trace = go.Scatter(
                        x=edges_by_status_selected[status]["x"],
                        y=edges_by_status_selected[status]["y"],
                        mode="lines",
                        line=dict(width=4.0, color=colour),
                        hoverinfo="skip",
                        showlegend=False,
                        name=f"{status} (selected)",
                        visible=True,
                    )
                    edge_traces.append(trace)

        # Prepare node display data with enhanced tooltips
        def format_node_tooltip(nid: str, node_data: Dict) -> str:
            """Format comprehensive node tooltip with all available metrics."""
            conflict = node_data.get("conflict_score", 0)

            # Extract health metrics if available
            health = node_data.get("health", {})
            health_score = health.get("health_score", None) if isinstance(health, dict) else None
            summary_score = health.get("summary_score", None) if isinstance(health, dict) else None

            # For academic nodes, use summary field which contains the title
            source_category = node_data.get("source_category", "")
            if source_category == "academic_reference":
                summary = node_data.get("summary", "")
                if summary:
                    # Extract title from summary (usually first line or first ~100 chars)
                    # Clean up newlines and excessive whitespace
                    title_text = " ".join(summary.split())
                    # Truncate if too long
                    if len(title_text) > 100:
                        display_title = title_text[:97] + "..."
                    else:
                        display_title = title_text
                    tooltip = f"<b>{display_title}</b><br>"
                else:
                    # Fallback to node_id if no summary
                    tooltip = f"<b>{nid}</b><br>"
            else:
                # Non-academic nodes use node_id
                tooltip = f"<b>{nid}</b><br>"

            tooltip += f"Conflict: {conflict:.3f}<br>"

            if health_score is not None:
                tooltip += f"Health: {health_score:.2f}<br>"
            if summary_score is not None:
                tooltip += f"Summary: {summary_score:.2f}<br>"

            # Add doc type if available
            doc_type = node_data.get("doc_type", "")
            if doc_type:
                tooltip += f"Type: {doc_type}<br>"

            # Add source category if available
            source = node_data.get("source_category", "")
            if source:
                tooltip += f"Source: {source}"

            return tooltip

        node_text = [format_node_tooltip(nid, nodes_dict[nid]) for nid, _ in visible_nodes]
        node_colours = [nodes_dict[nid].get("conflict_score", 0) for nid, _ in visible_nodes]

        # Calculate node degrees (number of connections) for sizing
        node_degrees = {}
        for edge in filtered_edges:
            src = edge.get("source")
            tgt = edge.get("target")
            node_degrees[src] = node_degrees.get(src, 0) + 1
            node_degrees[tgt] = node_degrees.get(tgt, 0) + 1

        # Scale node sizes based on degree: base_size + (degree * scale_factor)
        # For WebGL: base=8, max_additional=12 (sizes 8-20)
        # For SVG: base=10, max_additional=15 (sizes 10-25)
        max_degree = max(node_degrees.values()) if node_degrees else 1

        node_sizes_webgl = [
            8 + (node_degrees.get(nid, 0) / max(max_degree, 1)) * 12 for nid, _ in visible_nodes
        ]
        node_sizes_svg = [
            10 + (node_degrees.get(nid, 0) / max(max_degree, 1)) * 15 for nid, _ in visible_nodes
        ]

        # Handle node selection emphasis
        selected_connected_nodes = set()
        if selected_node:
            # Find all nodes connected to the selected node
            for edge in filtered_edges:
                if edge.get("source") == selected_node:
                    selected_connected_nodes.add(edge.get("target"))
                elif edge.get("target") == selected_node:
                    selected_connected_nodes.add(edge.get("source"))
            selected_connected_nodes.add(selected_node)

        # Adjust node appearance based on selection
        node_opacities = []
        node_line_widths = []
        node_line_colours = []
        for nid, _ in visible_nodes:
            if selected_node:
                if nid == selected_node:
                    # Emphasize selected node
                    node_opacities.append(1.0)
                    node_line_widths.append(4)
                    node_line_colours.append("#FF6B00")  # Orange border
                elif nid in selected_connected_nodes:
                    # Connected nodes - highlight
                    node_opacities.append(1.0)
                    node_line_widths.append(3)
                    node_line_colours.append("#667eea")  # Purple border
                else:
                    # Fade unrelated nodes
                    node_opacities.append(0.2)
                    node_line_widths.append(1)
                    node_line_colours.append("#ccc")
            else:
                # Normal state - no selection
                node_opacities.append(1.0)
                node_line_widths.append(1)
                node_line_colours.append("#fff")

        # =====================================================
        # WebGL Rendering Decision
        # =====================================================

        # Determine if we should use WebGL (scattergl); user toggle wins, fallback to auto only before the control initialises
        use_webgl_selected = "webgl" in (use_webgl_list or [])
        use_webgl = (
            total_filtered > webgl_threshold if use_webgl_list is None else use_webgl_selected
        )
        show_node_names = show_names_list and "show-names" in show_names_list
        trace_mode = "markers+text" if show_node_names else "markers"

        fig = go.Figure()

        # Add edge traces (one per relationship status)
        for edge_trace in edge_traces:
            fig.add_trace(edge_trace)

        # Create node trace based on rendering mode
        if use_webgl:
            # Use Scattergl for WebGL rendering (high performance for large graphs)
            node_trace = go.Scattergl(
                x=node_x,
                y=node_y,
                mode=trace_mode,
                marker=dict(
                    size=node_sizes_webgl,
                    color=node_colours,
                    colorscale="RdYlGn_r",
                    showscale=True,
                    opacity=node_opacities,
                    line=dict(width=node_line_widths, color=node_line_colours),
                    colorbar=dict(title="Conflict<br>Score", len=0.7),
                ),
                text=[get_display_name(nodes_dict.get(nid, {}), nid) for nid in page_node_ids],
                textposition="top center",
                hovertext=node_text,
                hoverinfo="text",
                showlegend=False,
                name="Nodes",
            )
        else:
            # Use regular Scatter for small graphs
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode=trace_mode,
                marker=dict(
                    size=node_sizes_svg,
                    color=node_colours,
                    colorscale="RdYlGn_r",
                    showscale=True,
                    opacity=node_opacities,
                    line=dict(width=node_line_widths, color=node_line_colours),
                    colorbar=dict(title="Conflict<br>Score"),
                ),
                text=[get_display_name(nodes_dict.get(nid, {}), nid) for nid in page_node_ids],
                textposition="top center",
                hovertext=node_text,
                hoverinfo="text",
                showlegend=False,
                name="Nodes",
            )

        # Add nodes on top so they're clickable and hovers over nodes show
        fig.add_trace(node_trace)
        if perf:
            perf.record("figure_build")

        # Determine layout algorithm info for display
        num_nodes = len(nodes_dict)
        if layout_type == "force":
            if num_nodes < 50:
                algo_info = "Fruchterman-Reingold"
            elif num_nodes < 200:
                algo_info = "Force-Directed"
            else:
                algo_info = "Force Atlas 2"
        else:
            algo_info = layout_type.title()

        # Build figure title with rendering and algorithm info
        render_method = "WebGL (GPU)" if use_webgl else "SVG (CPU)"
        fig.update_layout(
            title=(
                f"Consistency Graph - {algo_info} Layout ({render_method})<br>"
                f"<sub>Page {page + 1} of {total_pages}, "
                f"{len(visible_nodes)}/{original_page_size} nodes visible, "
                f"{visible_edges_count}/{len(filtered_edges)} edges visible</sub>"
            ),
            showlegend=False,
            hovermode="closest",  # Hover over the closest element (edges or nodes)
            dragmode="pan",  # Enable panning and dragging (use shift+click to select nodes)
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                autorange=True,  # Auto-scale to fit all nodes
                scaleanchor=None,  # Allow independent scaling
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                autorange=True,  # Auto-scale to fit all nodes
                scaleanchor="x",  # Maintain aspect ratio with x-axis
                scaleratio=1,  # Equal scaling for x and y
            ),
            height=600,
            plot_bgcolor="#fafafa",
            font=dict(size=12),
        )

        perf_data = None
        if perf:
            timings = format_timings_ms(perf.snapshot())
            perf_data = {
                "timings_ms": timings,
                "page_size": original_page_size,
                "visible_nodes": len(visible_nodes),
                "total_edges": len(filtered_edges),
                "visible_edges": visible_edges_count,
                "use_webgl": use_webgl,
                "layout": layout_type,
            }

        return (
            dcc.Graph(
                id="graph-figure", figure=fig, config={"responsive": True, "displayModeBar": True}
            ),
            perf_data,
        )

    except Exception as e:
        import traceback

        return (
            html.Div(
                [
                    html.P(
                        f"Error rendering graph: {str(e)}",
                        style={"color": "red", "textAlign": "center"},
                    ),
                    html.Pre(
                        traceback.format_exc(),
                        style={
                            "fontSize": "10px",
                            "color": "#999",
                            "maxHeight": "200px",
                            "overflow": "auto",
                        },
                    ),
                ]
            ),
            None,
        )


@callback(
    Output("render-stats", "children"),
    Output("perf-panel", "children"),
    Output("fps-display", "children"),
    Input("filter-state-store", "data"),
    Input("layout-selector", "value"),
    Input("webgl-toggle", "value"),
    Input("performance-store", "data"),
    Input("fps-store", "data"),
)
def update_render_stats(filter_state, layout_type, use_webgl_list, perf_data, fps_data):
    """Update rendering statistics display and performance panel."""
    try:
        all_nodes = graph_store.get_node_ids()
        if filter_state and filter_state.get("filtered_nodes"):
            all_nodes = filter_state.get("filtered_nodes", all_nodes)

        total_filtered = len(all_nodes)
        use_webgl_selected = "webgl" in (use_webgl_list or [])
        use_webgl = (
            total_filtered > webgl_threshold if use_webgl_list is None else use_webgl_selected
        )
        layout_type = layout_type or "force"

        render_method = "🚀 WebGL (GPU)" if use_webgl else "📊 SVG (CPU)"

        # Show adaptive algorithm info for force-directed layouts
        if layout_type == "force":
            if total_filtered < 50:
                layout_name = "Fruchterman-Reingold"
            elif total_filtered < 200:
                layout_name = "Force-Directed"
            else:
                layout_name = "Force Atlas 2"
        else:
            layout_name = {"hierarchical": "Hierarchical", "circular": "Circular"}.get(
                layout_type, "Unknown"
            )

        stats_text = f"{render_method} • {layout_name} Layout • {total_filtered} nodes"

        perf_children = []
        if perf_data and perf_data.get("timings_ms"):
            t = perf_data["timings_ms"]

            def fmt(label: str) -> str:
                return f"{label}: {t.get(label, 0)} ms"

            perf_children = [
                html.Div("Performance", style={"fontWeight": 600, "marginBottom": "4px"}),
                html.Div(
                    [
                        html.Span(fmt("filter_load")),
                        html.Br(),
                        html.Span(fmt("page_fetch")),
                        html.Br(),
                        html.Span(fmt("layout_compute")),
                        html.Br(),
                        html.Span(fmt("cull")),
                        html.Br(),
                        html.Span(fmt("figure_build")),
                    ],
                    style={"fontSize": "11px", "color": "#444"},
                ),
                html.Div(
                    f"Visible: {perf_data.get('visible_nodes', 0)}/{perf_data.get('page_size', 0)} nodes, "
                    f"{perf_data.get('visible_edges', 0)}/{perf_data.get('total_edges', 0)} edges",
                    style={"fontSize": "11px", "color": "#666", "marginTop": "6px"},
                ),
            ]

        fps_child = None
        if fps_data:
            fps_child = html.Span(f"FPS: {fps_data}", style={"fontWeight": 500, "color": "#2b7a0b"})

        return (
            html.Span(stats_text, style={"fontWeight": "500", "color": "#667eea"}),
            perf_children,
            fps_child,
        )

    except Exception as e:
        return html.Span(f"Error: {str(e)}", style={"color": "#999", "fontSize": "11px"}), [], None


# ============================================================================
# Query Template Callbacks
# ============================================================================


@callback(
    Output("templates-panel", "style"),
    Input("toggle-templates-btn", "n_clicks"),
    State("templates-panel", "style"),
    prevent_initial_call=False,
)
def toggle_templates_panel(n_clicks, current_style):
    """Toggle templates panel visibility."""
    if not n_clicks:
        return current_style

    if current_style.get("display") == "none":
        # Show panel
        style = current_style.copy()
        style["display"] = "block"
    else:
        # Hide panel
        style = current_style.copy()
        style["display"] = "none"
    return style


@callback(
    Output("filters-panel", "style"),
    Input("toggle-filters-btn", "n_clicks"),
    State("filters-panel", "style"),
    prevent_initial_call=False,
)
def toggle_filters_panel(n_clicks, current_style):
    """Toggle filters panel visibility."""
    if not n_clicks:
        return current_style

    if current_style.get("display") == "none":
        # Show panel
        style = current_style.copy()
        style["display"] = "block"
    else:
        # Hide panel
        style = current_style.copy()
        style["display"] = "none"
    return style


@callback(
    Output("filters-main-panel", "style"),
    Output("toggle-main-filters-btn", "children"),
    Input("toggle-main-filters-btn", "n_clicks"),
    State("filters-main-panel", "style"),
    prevent_initial_call=False,
)
def toggle_main_filters(n_clicks, current_style):
    """Toggle main Filters & Controls expander visibility and button label."""
    if current_style is None:
        current_style = {"display": "none"}
    if not n_clicks:
        # Initial state
        label = "▸ Show Filters" if current_style.get("display") == "none" else "▾ Hide Filters"
        return current_style, label
    if current_style.get("display") == "none":
        style = {**current_style, "display": "block"}
        return style, "▾ Hide Filters"
    else:
        style = {**current_style, "display": "none"}
        return style, "▸ Show Filters"


@callback(
    Output("custom-role-input", "value", allow_duplicate=True),
    Output("date-range-filter", "start_date", allow_duplicate=True),
    Output("date-range-filter", "end_date", allow_duplicate=True),
    Output("confidence-filter", "value", allow_duplicate=True),
    Output("result-type-filter", "value", allow_duplicate=True),
    Output("tags-filter-input", "value", allow_duplicate=True),
    Input("reset-advanced-filters-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_advanced_filters(n_clicks):
    """Reset all advanced filters to default values."""
    if not n_clicks:
        raise PreventUpdate
    return "", None, None, 0, ["documents", "code"], ""


@callback(
    Output("template-select", "options"),
    Output("template-select", "value"),
    Input("template-category-select", "value"),
)
def update_template_options(category):
    """Update template dropdown based on selected category."""
    from scripts.ui.query_templates import QueryTemplateManager

    try:
        # Get templates from database
        template_db_path = Path(RAG_CONFIG.rag_data_path) / "query_templates.db"
        template_manager = QueryTemplateManager(template_db_path)

        if category == "all":
            # Get all categories and their templates
            all_categories = template_manager.get_all_categories()
            options = []
            for cat in all_categories:
                templates = template_manager.get_templates_by_category(cat)
                for template in templates:
                    options.append(
                        {"label": f"[{cat.upper()}] {template['name']}", "value": template["name"]}
                    )
        else:
            # Get templates for specific category
            templates = template_manager.get_templates_by_category(category)
            options = [
                {"label": template["name"], "value": template["name"]} for template in templates
            ]

        return options, ""
    except Exception as e:
        print(f"Error loading templates: {e}")
        return [{"label": "Error loading templates", "value": ""}], ""


@callback(
    Output("rag-query-input", "value", allow_duplicate=True),
    Output("template-params-input", "value"),
    Output("rag-query-k", "value", allow_duplicate=True),
    Output("rag-query-temp", "value", allow_duplicate=True),
    Output("rag-query-code-aware", "value", allow_duplicate=True),
    Output("rag-query-persona", "value"),
    Input("apply-template-btn", "n_clicks"),
    State("template-select", "value"),
    State("template-params-input", "value"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def apply_template(n_clicks, template_name, params_text):
    """Apply selected template to query input and update chunk settings, including persona."""
    if not template_name:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    from scripts.ui.query_templates import QueryTemplateManager

    try:
        template_db_path = Path(RAG_CONFIG.rag_data_path) / "query_templates.db"
        template_manager = QueryTemplateManager(template_db_path)
        template = template_manager.get_template(template_name)

        if not template:
            return (
                dash.no_update,
                "Template not found",
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        # Substitute parameters if template has {}
        query_text = template["template_text"]
        if "{}" in query_text and params_text:
            # Split parameters by comma
            params = [p.strip() for p in params_text.split(",")]
            for param in params:
                query_text = query_text.replace("{}", param, 1)

        # Extract chunk settings from template
        k_results = template.get("k_results", 5)
        temperature = template.get("temperature", 0.3)
        code_aware = template.get("code_aware", True)
        persona = template.get("persona")
        persona_value = persona if persona else "none"

        return (
            query_text,
            params_text or "",
            k_results,
            temperature,
            ([code_aware] if code_aware else []),
            persona_value,
        )
    except Exception as e:
        print(f"Error applying template: {e}")
        return (
            dash.no_update,
            f"Error: {str(e)}",
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )


# ============================================================================
# ChromaDB Status Check Callback
# ============================================================================


@callback(
    Output("chromadb-status-alert", "children"),
    Output("chromadb-status-alert", "style"),
    Output("rag-query-run", "disabled"),
    Input("rag-query-run", "n_clicks"),  # Trigger on page load and query attempts
    prevent_initial_call=False,
)
def check_chromadb_status(n_clicks):
    """Check if ChromaDB is available and show alert if not."""
    try:
        collection = _get_query_collection()
        if collection and collection.count() > 0:
            # ChromaDB has data, queries available
            return "", {"display": "none"}, False
        else:
            # ChromaDB is empty
            alert_content = html.Div(
                [
                    html.Strong("⚠️ Query Feature Unavailable"),
                    html.P(
                        "The document database is empty. Queries will not return results until data is re-ingested. "
                        "Please run the ingestion process (ingest.py or ingest_git.py) to populate the database.",
                        style={"margin": "8px 0"},
                    ),
                ],
                style={
                    "backgroundColor": "#fff3cd",
                    "borderColor": "#ffc107",
                    "color": "#856404",
                    "border": "1px solid #ffc107",
                },
            )
            return (
                alert_content,
                {
                    "display": "block",
                    "backgroundColor": "#fff3cd",
                    "borderColor": "#ffc107",
                    "color": "#856404",
                    "border": "1px solid #ffc107",
                },
                True,
            )
    except Exception as e:
        # Error accessing ChromaDB
        alert_content = html.Div(
            [
                html.Strong("⚠️ Cannot Access Query Database"),
                html.P(
                    f"Error: {str(e)}. Please check the logs for details.",
                    style={"margin": "8px 0"},
                ),
            ],
            style={
                "backgroundColor": "#f8d7da",
                "borderColor": "#f5c6cb",
                "color": "#721c24",
                "border": "1px solid #f5c6cb",
            },
        )
        return (
            alert_content,
            {
                "display": "block",
                "backgroundColor": "#f8d7da",
                "borderColor": "#f5c6cb",
                "color": "#721c24",
                "border": "1px solid #f5c6cb",
            },
            True,
        )


# ============================================================================
# RAG Query Callback
# ============================================================================


@callback(
    Output("rag-query-answer", "children"),
    Output("rag-query-sources", "children"),
    Output("rag-query-code-preview", "children"),
    Output("code-preview-container", "style"),
    Output("rag-query-metadata", "children"),
    Output("rag-query-explainability", "children"),
    Output("explainability-container", "style"),
    Output("rag-query-status", "children"),
    Output("current-query-record-id", "data"),
    Input("rag-query-run", "n_clicks"),
    State("rag-query-input", "value"),
    State("rag-query-k", "value"),
    State("rag-query-temp", "value"),
    State("rag-query-code-aware", "value"),
    State("rag-query-persona", "value"),
    State("custom-role-input", "value"),
    State("date-range-filter", "start_date"),
    State("date-range-filter", "end_date"),
    State("confidence-filter", "value"),
    State("result-type-filter", "value"),
    State("tags-filter-input", "value"),
    prevent_initial_call=True,
)
def run_rag_query(
    n_clicks,
    query_text,
    k_value,
    temp_value,
    code_aware_list,
    persona_value,
    custom_role,
    start_date,
    end_date,
    min_confidence,
    result_types,
    tags_filter,
):
    """Execute RAG query with code-aware context, custom role, and filters."""
    if not n_clicks:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    if not query_text or not query_text.strip():
        return (
            "",
            "",
            "",
            {"display": "none"},
            "",
            "",
            {"display": "none"},
            "Enter a question to run the query.",
        )

    try:
        import re
        from datetime import datetime

        collection = _get_query_collection()
        code_aware_enabled = code_aware_list and "enable" in code_aware_list

        # Prepare custom role (use only if provided and not empty)
        role_to_use = custom_role.strip() if custom_role and custom_role.strip() else None
        persona_to_use = None
        if persona_value and persona_value != "none":
            persona_to_use = persona_value

        # Extract context signals from dashboard UI
        enable_code_detection = code_aware_list and "enable" in code_aware_list
        allow_code_category = result_types and "code" in result_types

        # Track system metrics during query execution
        manager = get_benchmark_manager()
        if manager:
            with manager.track_query_metrics() as metrics:
                response = rag_answer(
                    query=query_text.strip(),
                    collection=collection,
                    k=k_value,
                    temperature=temp_value,
                    custom_role=role_to_use,
                    persona=persona_to_use,
                    enable_code_detection=enable_code_detection,
                    allow_code_category=allow_code_category,
                )
                # Store metrics for benchmark recording later
                system_metrics_data = {"start": metrics["start"], "max": metrics["max"]}
        else:
            # No benchmark manager, just execute query
            response = rag_answer(
                query=query_text.strip(),
                collection=collection,
                k=k_value,
                temperature=temp_value,
                custom_role=role_to_use,
                persona=persona_to_use,
                enable_code_detection=enable_code_detection,
                allow_code_category=allow_code_category,
            )
            system_metrics_data = None

        answer_md = response.get("answer", "(no answer returned)")
        if not isinstance(answer_md, str):
            answer_md = str(answer_md)
        sources = response.get("sources", []) or []
        gen_time = response.get("generation_time")
        total_time = response.get("total_time")
        is_code_query = response.get("is_code_query", False)
        retrieval_count = response.get("retrieval_count", 0)

        # Apply filters to sources
        filtered_sources = sources

        # Date range filter
        if start_date or end_date:

            def _filter_by_date(source):
                # Try to get creation date from source metadata
                created_date = source.get("created_date") or source.get("modified_date")
                if not created_date:
                    return True  # Keep if no date info

                try:
                    src_date = datetime.fromisoformat(created_date.replace("Z", "+00:00"))
                    if start_date and src_date < datetime.fromisoformat(start_date):
                        return False
                    if end_date and src_date > datetime.fromisoformat(end_date):
                        return False
                    return True
                except:
                    return True  # Keep if date parsing fails

            filtered_sources = [s for s in filtered_sources if _filter_by_date(s)]

        # Confidence filter (based on distance/similarity score)
        if min_confidence > 0:
            # Assuming distance is stored; lower distance = higher confidence
            # Convert confidence (0-100) to max distance threshold
            # This is a heuristic: confidence 100% = distance ~0, confidence 0% = distance ~2
            max_distance = 2.0 * (1.0 - min_confidence / 100.0)
            filtered_sources = [s for s in filtered_sources if s.get("distance", 0) <= max_distance]

        # Result type filter
        if result_types and len(result_types) < 2:  # If not both selected
            if "code" in result_types:
                # Keep only code sources
                filtered_sources = [
                    s
                    for s in filtered_sources
                    if s.get("language") or "code" in s.get("doc_id", "").lower()
                ]
            elif "documents" in result_types:
                # Keep only document sources (non-code)
                filtered_sources = [
                    s
                    for s in filtered_sources
                    if not s.get("language") and "code" not in s.get("doc_id", "").lower()
                ]

        # Tags filter
        if tags_filter and tags_filter.strip():
            tags = [t.strip().lower() for t in tags_filter.split(",") if t.strip()]
            if tags:

                def _has_matching_tag(source):
                    # Check if any source tag matches filter tags
                    source_tags = source.get("tags", [])
                    if not source_tags:
                        return False
                    source_tags_lower = [t.lower() for t in source_tags]
                    return any(tag in source_tags_lower for tag in tags)

                filtered_sources = [s for s in filtered_sources if _has_matching_tag(s)]

        # Use filtered sources for display
        sources = filtered_sources

        # Extract code blocks for preview if code-aware and code query detected
        code_preview_content = ""
        code_preview_style = {"display": "none"}

        if code_aware_enabled and is_code_query:
            # Extract code blocks using regex
            code_pattern = r"```(\w*)\n(.*?)\n```"
            code_blocks = re.findall(code_pattern, answer_md, re.DOTALL)

            if code_blocks:
                code_preview_style = {"display": "block"}
                code_items = []
                for i, (lang, code_text) in enumerate(code_blocks):
                    lang_display = lang if lang else "code"

                    # Add line numbers
                    lines = code_text.split("\n")
                    max_line_num = len(lines)
                    line_num_width = len(str(max_line_num))

                    # Build line numbers and highlighted code
                    line_items = []
                    for line_idx, line in enumerate(lines, 1):
                        line_items.append(
                            html.Div(
                                [
                                    html.Span(
                                        f"{line_idx:{line_num_width}}",
                                        style={
                                            "color": "#999",
                                            "marginRight": "12px",
                                            "fontFamily": "monospace",
                                            "fontSize": "10px",
                                            "userSelect": "none",
                                        },
                                    ),
                                    html.Span(
                                        line, style={"fontFamily": "monospace", "fontSize": "11px"}
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "whiteSpace": "pre-wrap",
                                    "wordBreak": "break-all",
                                },
                            )
                        )

                    code_items.append(
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            lang_display.upper(),
                                            style={"fontWeight": 600, "fontSize": "10px"},
                                        ),
                                        html.Button(
                                            "📋 Copy",
                                            id={"type": "copy-code-btn", "index": i},
                                            n_clicks=0,
                                            title="Copy code to clipboard",
                                            style={
                                                "float": "right",
                                                "padding": "4px 8px",
                                                "fontSize": "10px",
                                                "backgroundColor": "#e9ecef",
                                                "border": "1px solid #dee2e6",
                                                "borderRadius": "3px",
                                                "cursor": "pointer",
                                            },
                                        ),
                                    ],
                                    style={
                                        "fontWeight": 600,
                                        "fontSize": "10px",
                                        "marginBottom": "8px",
                                        "display": "flex",
                                        "justifyContent": "space-between",
                                    },
                                ),
                                html.Div(
                                    line_items,
                                    style={
                                        "backgroundColor": "#f5f5f5",
                                        "padding": "8px",
                                        "borderRadius": "4px",
                                        "overflow": "auto",
                                        "maxHeight": "250px",
                                        "border": "1px solid #e0e0e0",
                                    },
                                ),
                            ],
                            style={"marginBottom": "12px"},
                        )
                    )
                code_preview_content = code_items

        # Build source list with enhanced metadata (numbered to match chunk citations)
        source_items = []
        for src in sources:
            label_parts = []

            # Document identifier - prefer display_name for academic sources
            if src.get("source_category") == "academic_reference":
                # For academic sources, try to look up the node data and get display name
                doc_id = src.get("doc_id")
                if doc_id:
                    # Find the corresponding node in the consistency graph
                    node_data = graph_store.get_node(doc_id)
                    if node_data:
                        display_name = get_display_name(node_data, doc_id)
                        label_parts.append(display_name)
                    else:
                        # Try variants of the doc_id (with version suffix)
                        # Consistency graph nodes have _vN suffix, but sources might not
                        for i in range(1, 6):
                            variant_id = f"{doc_id}_v{i}"
                            node_data = graph_store.get_node(variant_id)
                            if node_data:
                                display_name = get_display_name(node_data, variant_id)
                                label_parts.append(display_name)
                                break
                        else:
                            # No match found, use doc_id as fallback
                            label_parts.append(doc_id)
                else:
                    label_parts.append("(unknown academic source)")
            else:
                # For non-academic sources, use doc_id as before
                for key in ("doc_id", "document_id", "source", "file"):
                    if src.get(key):
                        label_parts.append(str(src.get(key)))
                        break

            # Code-specific metadata
            if code_aware_enabled and is_code_query:
                for key in ("language", "service", "doc_type"):
                    if src.get(key):
                        label_parts.append(str(src.get(key)))
            else:
                for key in ("language", "service", "doc_type"):
                    if src.get(key):
                        label_parts.append(str(src.get(key)))

            # Similarity score
            if src.get("score") is not None:
                label_parts.append(f"score={src.get('score'):.3f}")

            # Only add source item if there are label parts
            if label_parts:
                source_items.append(html.Li(" • ".join(label_parts)))

        if not source_items:
            source_content = html.Div("No sources returned", style={"color": "#777"})
        else:
            source_content = html.Ol(source_items, style={"paddingLeft": "20px"})

        # Build metadata display
        metadata_items = [
            html.Li(f"Query: {query_text[:80]}{'...' if len(query_text) > 80 else ''}"),
            html.Li(f"Type: {'Code Query' if is_code_query else 'Governance Query'}"),
            html.Li(f"Chunks Retrieved: {retrieval_count}"),
            html.Li(f"Generation Time: {gen_time:.2f}s"),
            html.Li(f"Total Time: {total_time:.2f}s"),
        ]
        if code_aware_enabled:
            metadata_items.append(html.Li("Code-Aware Context: Enabled"))

        # Build explainability display
        explainability = response.get("explainability", {})
        explainability_content = ""
        explainability_style = {"display": "none"}

        if explainability:
            explainability_style = {"display": "block"}
            confidence_level = explainability.get("confidence_level", "unknown")
            confidence_colour = {
                "high": "#28a745",
                "medium": "#ffc107",
                "low": "#dc3545",
                "unknown": "#6c757d",
            }.get(confidence_level, "#999")

            avg_similarity = explainability.get("avg_similarity", 0.0)
            ranking_explanation = explainability.get("ranking_explanation", "")
            retrieval_methods = explainability.get("retrieval_method", [])

            explainability_content = html.Div(
                [
                    html.Div(
                        [
                            html.Strong("Confidence Level: "),
                            html.Span(
                                confidence_level.upper(),
                                style={"color": confidence_colour, "fontWeight": "bold"},
                            ),
                        ],
                        style={"marginBottom": "8px"},
                    ),
                    html.Div(
                        [
                            html.Strong("Average Similarity: "),
                            html.Span(f"{avg_similarity:.1%}"),
                        ],
                        style={"marginBottom": "8px"},
                    ),
                    html.Div(
                        [
                            html.Strong("Ranking Method: "),
                            html.Span(ranking_explanation),
                        ],
                        style={"marginBottom": "8px"},
                    ),
                    html.Div(
                        [
                            html.Strong("Retrieval Methods: "),
                            html.Span(
                                ", ".join(str(m) for m in retrieval_methods)
                                if retrieval_methods
                                else "vector"
                            ),
                        ],
                        style={"marginBottom": "8px"},
                    ),
                ]
            )

        status_parts = ["Done"]
        if total_time is not None:
            status_parts.append(f"total {total_time:.2f}s")
        if gen_time is not None:
            status_parts.append(f"llm {gen_time:.2f}s")
        status_text = " • ".join(status_parts)

        # Record benchmark
        record_id = None
        try:
            manager = get_benchmark_manager()
            if manager:
                # Measure network latency
                network_latency = manager.measure_network_latency()

                record_id = manager.record_query(
                    query=query_text.strip(),
                    response=response,
                    query_params={
                        "k": k_value,
                        "temperature": temp_value,
                        "custom_role": role_to_use,
                        "collection_name": "default",
                    },
                    system_metrics=system_metrics_data,
                    network_latency_ms=network_latency,
                )
        except Exception as benchmark_error:
            print(f"Warning: Could not record benchmark: {benchmark_error}")

        return (
            answer_md,
            source_content,
            code_preview_content,
            code_preview_style,
            metadata_items,
            explainability_content,
            explainability_style,
            status_text,
            record_id,
        )

    except Exception as e:
        error_div = html.Div(f"Error: {str(e)}", style={"color": "red"})
        return (
            "",
            error_div,
            "",
            {"display": "none"},
            "",
            "",
            {"display": "none"},
            "Query failed",
            None,
        )


@callback(
    Output("cache-stats-display", "children"),
    Input("cache-stats-interval", "n_intervals"),
)
def update_cache_stats(n_intervals):
    """Update cache statistics from CacheDB."""
    try:
        from scripts.ingest.ingest_config import get_ingest_config

        config = get_ingest_config()
        cache = get_cache_client(
            rag_data_path=Path(config.rag_data_path), enable_cache=config.cache_enabled
        )

        if not config.cache_enabled:
            return html.Div("Cache disabled", style={"color": "#999"})

        stats = cache.get_all_stats()

        # Build cache statistics display
        cache_items = []

        # Embeddings stats
        if "embeddings" in stats:
            emb_stats = stats["embeddings"]
            cache_items.append(
                html.Div(
                    [
                        html.Strong("🧠 Embeddings: "),
                        (
                            html.Span(f"{emb_stats.get('entries', 0)} cached")
                            if emb_stats.get("entries", 0) > 0
                            else "Empty"
                        ),
                    ]
                )
            )

        # LLM cache stats
        if "llm" in stats:
            llm_stats = stats["llm"]
            cache_items.append(
                html.Div(
                    [
                        html.Strong("💬 LLM Cache: "),
                        (
                            html.Span(f"{llm_stats.get('entries', 0)} cached")
                            if llm_stats.get("entries", 0) > 0
                            else "Empty"
                        ),
                    ]
                )
            )

        # File info
        graph_file_size = os.path.getsize(GRAPH_SQLITE) if os.path.exists(GRAPH_SQLITE) else 0
        cache_items.append(html.Hr(style={"margin": "8px 0"}))
        cache_items.append(
            html.Div(
                [
                    html.Strong("📁 Graph File: "),
                    html.Span(human_size(graph_file_size)),
                ]
            )
        )
        cache_items.append(
            html.Div(
                [
                    html.Strong("Loaded Nodes: "),
                    html.Span(str(len(graph_store.get_node_ids()))),
                ]
            )
        )

        return cache_items

    except Exception as e:
        return html.Div(f"Error: {str(e)}", style={"color": "red"})


# Relevancy feedback callbacks
@callback(
    Output("relevancy-feedback-section", "style"),
    Input("current-query-record-id", "data"),
    prevent_initial_call=False,
)
def show_relevancy_feedback(record_id):
    """Show relevancy feedback section when a query has been recorded."""
    if record_id and record_id > 0:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("rating-1", "style"),
    Output("rating-2", "style"),
    Output("rating-3", "style"),
    Output("rating-4", "style"),
    Output("rating-5", "style"),
    Output("relevancy-feedback-text", "style"),
    Output("relevancy-feedback-status", "children"),
    Input("rating-1", "n_clicks"),
    Input("rating-2", "n_clicks"),
    Input("rating-3", "n_clicks"),
    Input("rating-4", "n_clicks"),
    Input("rating-5", "n_clicks"),
    State("relevancy-feedback-text", "value"),
    State("current-query-record-id", "data"),
    prevent_initial_call=True,
)
def handle_relevancy_rating(r1, r2, r3, r4, r5, feedback_text, record_id):
    """Handle star rating clicks and save to database."""
    base_style = {
        "fontSize": "24px",
        "background": "none",
        "border": "none",
        "cursor": "pointer",
        "padding": "0 4px",
    }
    filled_style = {**base_style, "color": "#FFB800"}
    empty_style = {**base_style, "color": "#ddd"}
    textarea_hidden = {
        "width": "100%",
        "height": "60px",
        "padding": "8px",
        "fontSize": "13px",
        "border": "1px solid #ddd",
        "borderRadius": "4px",
        "resize": "vertical",
        "display": "none",
    }
    textarea_visible = {
        "width": "100%",
        "height": "60px",
        "padding": "8px",
        "fontSize": "13px",
        "border": "1px solid #ddd",
        "borderRadius": "4px",
        "resize": "vertical",
        "display": "block",
    }

    if not record_id or record_id <= 0:
        return base_style, base_style, base_style, base_style, base_style, textarea_hidden, ""

    ctx = dash.callback_context
    if not ctx.triggered:
        return base_style, base_style, base_style, base_style, base_style, textarea_hidden, ""

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Determine which star was clicked
    rating = None
    if trigger_id == "rating-1":
        rating = 1
    elif trigger_id == "rating-2":
        rating = 2
    elif trigger_id == "rating-3":
        rating = 3
    elif trigger_id == "rating-4":
        rating = 4
    elif trigger_id == "rating-5":
        rating = 5

    # If a star was clicked, save the rating
    if rating:
        try:
            manager = get_benchmark_manager()
            if manager:
                success = manager.update_relevancy_rating(
                    record_id=record_id,
                    rating=rating,
                    feedback=feedback_text if feedback_text else None,
                )

                if success:
                    status_msg = f"✓ Rated {rating}/5 stars"
                    if feedback_text:
                        status_msg += " with feedback"
                else:
                    status_msg = "✗ Failed to save rating"
            else:
                status_msg = "✗ Benchmark manager unavailable"
        except Exception as e:
            status_msg = f"✗ Error: {str(e)}"

        # Style stars based on rating
        styles = [filled_style if i < rating else empty_style for i in range(5)]

        return styles[0], styles[1], styles[2], styles[3], styles[4], textarea_visible, status_msg

    # If just feedback text changed (no new rating), don't change anything
    return base_style, base_style, base_style, base_style, base_style, textarea_visible, ""


@callback(
    Output("relevancy-feedback-status", "children", allow_duplicate=True),
    Input("relevancy-feedback-text", "value"),
    State("current-query-record-id", "data"),
    State("rating-1", "n_clicks"),
    State("rating-2", "n_clicks"),
    State("rating-3", "n_clicks"),
    State("rating-4", "n_clicks"),
    State("rating-5", "n_clicks"),
    prevent_initial_call=True,
)
def save_feedback_text(feedback_text, record_id, r1, r2, r3, r4, r5):
    """Auto-save feedback text when it changes (if rating already given)."""
    if not record_id or record_id <= 0:
        return ""

    # Check if any rating has been given (any star clicked)
    total_clicks = (r1 or 0) + (r2 or 0) + (r3 or 0) + (r4 or 0) + (r5 or 0)
    if total_clicks == 0:
        return ""

    # Determine current rating from which star has been clicked
    # The last clicked star determines the rating (1-5)
    if r5 and r5 > 0:
        current_rating = 5
    elif r4 and r4 > 0:
        current_rating = 4
    elif r3 and r3 > 0:
        current_rating = 3
    elif r2 and r2 > 0:
        current_rating = 2
    elif r1 and r1 > 0:
        current_rating = 1
    else:
        return ""

    if feedback_text and feedback_text.strip():
        try:
            manager = get_benchmark_manager()
            if manager:
                success = manager.update_relevancy_rating(
                    record_id=record_id, rating=current_rating, feedback=feedback_text.strip()
                )
                if success:
                    return f"✓ Feedback saved"
                else:
                    return "✗ Failed to save feedback"
            else:
                return ""
        except Exception as e:
            return f"✗ Error: {str(e)}"

    return ""


def _build_connected_nodes_list(selected_doc, all_edges):
    """Build list of connected nodes and relationships for details tab."""
    connected_edges = [
        edge
        for edge in all_edges
        if edge.get("source") == selected_doc or edge.get("target") == selected_doc
    ]

    if not connected_edges:
        return [html.Div("No connections found", style={"color": "#999", "fontSize": "12px"})]

    edge_items = []
    seen = set()
    for edge in connected_edges:
        # Determine the connected node
        connected_node = (
            edge.get("target") if edge.get("source") == selected_doc else edge.get("source")
        )
        relationship = edge.get("relationship", "consistent")
        field = edge.get("field")
        value = edge.get("value")
        key = (
            "|".join(sorted([str(selected_doc), str(connected_node)])),
            relationship,
            field,
            value,
        )
        if key in seen:
            continue
        seen.add(key)

        # Colour schemes for relationship types
        bg_colours = {
            "consistent": "#dcfce7",
            "duplicate": "#dbeafe",
            "partial_conflict": "#fef3c7",
            "conflict": "#fee2e2",
        }
        text_colours = {
            "consistent": "#166534",
            "duplicate": "#1e40af",
            "partial_conflict": "#92400e",
            "conflict": "#991b1b",
        }

        edge_items.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong("📄 " + connected_node),
                            html.Div(
                                [
                                    html.Span(
                                        relationship.replace("_", " ").title(),
                                        style={
                                            "display": "inline-block",
                                            "padding": "2px 8px",
                                            "marginLeft": "8px",
                                            "marginRight": "8px",
                                            "backgroundColor": bg_colours.get(
                                                relationship, "#f3f4f6"
                                            ),
                                            "color": text_colours.get(relationship, "#374151"),
                                            "borderRadius": "8px",
                                            "fontSize": "11px",
                                            "fontWeight": "600",
                                        },
                                    ),
                                    html.Span(
                                        f"Similarity: {edge.get('similarity', 0):.2f}",
                                        style={
                                            "fontSize": "11px",
                                            "color": "#666",
                                            "marginRight": "12px",
                                        },
                                    ),
                                    html.Span(
                                        f"Severity: {edge.get('severity', 0):.2f}",
                                        style={"fontSize": "11px", "color": "#666"},
                                    ),
                                ],
                                style={"marginTop": "4px"},
                            ),
                        ],
                        style={
                            "padding": "8px 12px",
                            "backgroundColor": "#fafafa",
                            "borderRadius": "6px",
                            "marginBottom": "8px",
                            "border": "1px solid #e5e7eb",
                        },
                    )
                ]
            )
        )

    return edge_items


@callback(
    Output("document-details-container", "children"),
    Input("tab-details", "n_clicks"),
    Input("document-selector", "value"),
    Input("selected-node-store", "data"),
)
def update_document_details(_, dropdown_value, selected_node):
    """Load document details on demand."""
    # Prioritise dropdown selection, then selected node from graph
    selected_doc = dropdown_value or selected_node

    if not selected_doc:
        return html.Div(
            [
                html.P(
                    "Select a document from the dropdown or click a node in the graph",
                    style={"color": "#999"},
                )
            ]
        )

    try:
        node_data = graph_store.get_node(selected_doc)
        if not node_data:
            return html.Div([html.P("Document not found", style={"color": "red"})])

        doc_text, doc_meta = graph_store.get_doc(selected_doc)

        # Get cluster names instead of IDs
        risk_cluster_labels = []
        for cluster_id in node_data.get("risk_clusters", []):
            label = (
                graph_filter.get_risk_cluster_label(cluster_id)
                if graph_filter
                else f"Risk Cluster {cluster_id}"
            )
            risk_cluster_labels.append(label)

        topic_cluster_labels = []
        for cluster_id in node_data.get("topic_clusters", []):
            label = (
                graph_filter.get_topic_cluster_label(cluster_id)
                if graph_filter
                else f"Topic Cluster {cluster_id}"
            )
            topic_cluster_labels.append(label)

        # Extract health metrics
        health_data = node_data.get("health", {})
        health_score = health_data.get("health_score") if isinstance(health_data, dict) else None
        summary_score = health_data.get("summary_score") if isinstance(health_data, dict) else None

        display_name = get_display_name(node_data, selected_doc)
        details = html.Div(
            [
                html.H3(display_name, style={"color": "#667eea", "marginBottom": "16px"}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Strong("Document Type:"),
                                        " ",
                                        node_data.get("doc_type", "N/A"),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Strong("Language:"),
                                        " ",
                                        node_data.get("language", "N/A"),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Strong("Source Category:"),
                                        " ",
                                        node_data.get("source_category", "N/A"),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Strong("Version:"),
                                        " ",
                                        str(node_data.get("version", "N/A")),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                            ],
                            style={"flex": "1", "marginRight": "24px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Strong("Conflict Score:"),
                                        " ",
                                        html.Span(
                                            f"{node_data.get('conflict_score', 0):.3f}",
                                            style={
                                                "color": (
                                                    "#e74c3c"
                                                    if node_data.get("conflict_score", 0) > 0.5
                                                    else "#27ae60"
                                                )
                                            },
                                        ),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Strong("Health Score:"),
                                        " ",
                                        html.Span(
                                            (
                                                f"{health_score:.2f}"
                                                if health_score is not None
                                                else "N/A"
                                            ),
                                            style={
                                                "color": (
                                                    "#27ae60"
                                                    if (health_score or 0) > 0.7
                                                    else "#e74c3c"
                                                )
                                            },
                                        ),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Strong("Summary Score:"),
                                        " ",
                                        html.Span(
                                            f"{summary_score:.2f}"
                                            if summary_score is not None
                                            else "N/A"
                                        ),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                            ],
                            style={"flex": "1"},
                        ),
                    ],
                    style={"display": "flex", "marginBottom": "16px"},
                ),
                html.Hr(),
                html.Div(
                    [
                        html.Strong("Risk Clusters:"),
                        html.Div(
                            [
                                html.Span(
                                    label,
                                    style={
                                        "display": "inline-block",
                                        "padding": "4px 12px",
                                        "margin": "4px 4px 4px 0",
                                        "backgroundColor": "#fee2e2",
                                        "color": "#991b1b",
                                        "borderRadius": "12px",
                                        "fontSize": "12px",
                                        "fontWeight": "500",
                                    },
                                )
                                for label in risk_cluster_labels
                            ]
                            if risk_cluster_labels
                            else [html.Span("None", style={"color": "#999", "fontSize": "12px"})]
                        ),
                    ],
                    style={"marginBottom": "12px"},
                ),
                html.Div(
                    [
                        html.Strong("Topic Clusters:"),
                        html.Div(
                            [
                                html.Span(
                                    label,
                                    style={
                                        "display": "inline-block",
                                        "padding": "4px 12px",
                                        "margin": "4px 4px 4px 0",
                                        "backgroundColor": "#dbeafe",
                                        "color": "#1e40af",
                                        "borderRadius": "12px",
                                        "fontSize": "12px",
                                        "fontWeight": "500",
                                    },
                                )
                                for label in topic_cluster_labels
                            ]
                            if topic_cluster_labels
                            else [html.Span("None", style={"color": "#999", "fontSize": "12px"})]
                        ),
                    ],
                    style={"marginBottom": "16px"},
                ),
                html.Hr(),
                # Connected Nodes and Relationships Section
                html.H4(
                    "Connected Nodes & Relationships",
                    style={"marginTop": "16px", "marginBottom": "12px"},
                ),
                html.Div(
                    _build_connected_nodes_list(selected_doc, graph_store.get_edges()),
                    style={"marginBottom": "16px"},
                ),
                html.Hr(),
                html.H4("Document Text", style={"marginTop": "16px"}),
                html.Div(
                    [
                        (
                            html.Pre(
                                doc_text[:2000] + "..." if len(doc_text) > 2000 else doc_text,
                                style={
                                    "backgroundColor": "#f5f5f5",
                                    "padding": "12px",
                                    "borderRadius": "4px",
                                    "overflowX": "auto",
                                    "fontSize": "12px",
                                    "lineHeight": "1.6",
                                    "whiteSpace": "pre-wrap",
                                },
                            )
                            if doc_text
                            else html.P(
                                "Document text not available from ChromaDB.",
                                style={"color": "#999", "fontStyle": "italic"},
                            )
                        )
                    ]
                ),
                html.Hr(),
                html.H4("Document Metadata", style={"marginTop": "16px"}),
                html.Div(
                    [
                        (
                            html.Pre(
                                (
                                    json.dumps(doc_meta, indent=2)
                                    if doc_meta
                                    else "No metadata available"
                                ),
                                style={
                                    "backgroundColor": "#f5f5f5",
                                    "padding": "12px",
                                    "borderRadius": "4px",
                                    "fontSize": "11px",
                                    "color": "#666",
                                    "overflowX": "auto",
                                    "wordBreak": "break-word",
                                    "whiteSpace": "pre-wrap",
                                    "maxHeight": "400px",
                                    "overflowY": "auto",
                                },
                            )
                            if doc_meta
                            else html.P("No metadata available", style={"color": "#999"})
                        )
                    ]
                ),
            ]
        )

        return details

    except Exception as e:
        import traceback

        return html.Div(
            [
                html.P(f"Error loading details: {e}", style={"color": "red"}),
                html.Pre(traceback.format_exc(), style={"fontSize": "10px", "color": "#999"}),
            ]
        )


@callback(
    Output("heatmap-container", "children"),
    Output("heatmap-node-ids-store", "data"),
    Input("tab-heatmap", "n_clicks"),
    Input("heatmap-node-count", "value"),
    Input("selected-node-store", "data"),
)
def update_heatmap(_, node_count, selected_node):
    """Generate risk heatmap (lazy loaded) with top-risk nodes sorted by relationship severity."""
    try:
        # Get all nodes
        all_node_ids = graph_store.get_node_ids()

        if not all_node_ids:
            return html.P("No nodes to display", style={"color": "#999"}), []

        # Get edges for all nodes to compute relationship severity scores
        all_edges = graph_store.get_edges()

        # Build temporary graph of all nodes to compute relationship scores
        G_full = graph_store.to_networkx_subgraph(set(all_node_ids))

        # Score nodes by max relationship severity (how conflicted they are with other nodes)
        nodes_with_relationship_risk = []
        for node_id in all_node_ids:
            max_severity = 0
            if node_id in G_full:
                for neighbor in G_full.neighbors(node_id):
                    edge_data = G_full.get_edge_data(node_id, neighbor)
                    if edge_data:
                        severity = edge_data.get("severity", 0)
                        max_severity = max(max_severity, severity)
            nodes_with_relationship_risk.append((node_id, max_severity))

        # Sort by relationship severity (descending) - highest conflict severity first
        nodes_with_relationship_risk.sort(key=lambda x: x[1], reverse=True)

        # Take only the top node_count nodes
        effective_count = min(node_count or 50, len(nodes_with_relationship_risk))
        top_nodes = [nid for nid, _ in nodes_with_relationship_risk[:effective_count]]

        if not top_nodes:
            return html.P("No nodes to display", style={"color": "#999"}), []

        # Build severity matrix for visible nodes
        node_ids = top_nodes
        G_sub = graph_store.to_networkx_subgraph(set(node_ids))

        matrix = []
        for u in node_ids:
            row = []
            for v in node_ids:
                if G_sub.has_edge(u, v):
                    row.append(G_sub[u][v].get("severity", 0))
                else:
                    row.append(0)
            matrix.append(row)

        # Create heatmap with colourscale: white for 0 (no relationship), red for high severity
        # Build display labels for heatmap axes
        heatmap_labels = []
        for nid in node_ids:
            node_data = graph_store.get_node(nid)
            display_name = get_display_name(node_data or {}, nid) if node_data else nid
            # Truncate for readability on heatmap
            if len(display_name) > 30:
                heatmap_labels.append(display_name[:27] + "...")
            else:
                heatmap_labels.append(display_name)

        hovertemplate = "%{x} ↔ %{y}<br>Relationship Severity: %{z:.3f}<extra></extra>"
        heatmap_trace = go.Heatmap(
            z=matrix,
            x=heatmap_labels,
            y=heatmap_labels,
            colorscale=[
                [0, "white"],
                [0.001, "lightblue"],
                [1, "red"],
            ],  # White for 0, red for high
            hovertemplate=hovertemplate,
            name="Relationship Severity",
        )

        fig = go.Figure(data=[heatmap_trace])

        # Add diagonal line to show mirror boundary using shapes (not traces)
        n = len(node_ids) - 1
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=-0.5,
            x1=n + 0.5,
            y1=n + 0.5,
            line=dict(color="rgba(0, 0, 0, 0.6)", width=2, dash="dot"),
            xref="x",
            yref="y",
        )

        fig.update_layout(
            title=f"Document Relationship Severity Matrix (Top {len(node_ids)} nodes by relationship severity)",
            xaxis_title="Document",
            yaxis_title="Document",
            height=700,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
        )

        return dcc.Graph(figure=fig, id="risk-heatmap-graph"), node_ids

    except Exception as e:
        return html.Div([html.P(f"Error rendering heatmap: {e}", style={"color": "red"})]), []


@callback(
    Output("heatmap-node-count", "value"),
    Output("heatmap-node-prev", "style"),
    Output("heatmap-node-next", "style"),
    Output("heatmap-node-count-input", "value"),
    Input("heatmap-node-prev", "n_clicks"),
    Input("heatmap-node-next", "n_clicks"),
    Input("heatmap-node-count-input", "value"),
    State("heatmap-node-count", "value"),
    prevent_initial_call=True,
)
def update_heatmap_node_count(prev_clicks, next_clicks, input_value, current_value):
    """Update heatmap node count via Previous/Next buttons or direct input."""
    from dash import callback_context

    if not callback_context.triggered:
        raise PreventUpdate

    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    new_value = current_value

    if trigger_id == "heatmap-node-prev" and current_value > 10:
        new_value = max(10, current_value - 10)
    elif trigger_id == "heatmap-node-next" and current_value < 200:
        new_value = min(200, current_value + 10)
    elif trigger_id == "heatmap-node-count-input" and input_value is not None:
        # Validate and constrain input value
        new_value = max(10, min(200, int(input_value)))
        # Round to nearest 10
        new_value = round(new_value / 10) * 10

    # Style button disabled states
    prev_style = {
        "padding": "8px 16px",
        "marginRight": "8px",
        "border": "1px solid #ddd",
        "background": "#f5f5f5",
        "color": "#999",
        "borderRadius": "4px",
        "cursor": "not-allowed",
        "fontSize": "14px",
        "fontWeight": "500",
        "opacity": "0.6",
    }
    next_style = {
        "padding": "8px 16px",
        "border": "1px solid #ddd",
        "background": "#f5f5f5",
        "color": "#999",
        "borderRadius": "4px",
        "cursor": "not-allowed",
        "fontSize": "14px",
        "fontWeight": "500",
        "opacity": "0.6",
    }

    enabled_style = {
        "padding": "8px 16px",
        "border": "1px solid #667eea",
        "background": "white",
        "color": "#667eea",
        "borderRadius": "4px",
        "cursor": "pointer",
        "fontSize": "14px",
        "fontWeight": "500",
    }

    # Apply enabled/disabled states
    if new_value > 10:
        prev_style = enabled_style.copy()
        prev_style["marginRight"] = "8px"

    if new_value < 200:
        next_style = enabled_style.copy()

    return new_value, prev_style, next_style, new_value


@callback(
    Output("document-selector", "value", allow_duplicate=True),
    Output("selected-node-store", "data", allow_duplicate=True),
    Input("risk-heatmap-graph", "clickData"),
    State("heatmap-node-ids-store", "data"),
    prevent_initial_call=True,
)
def handle_heatmap_click(click_data, node_ids):
    """Handle clicks on heatmap cells and select the document."""
    if not click_data or "points" not in click_data or len(click_data["points"]) == 0:
        raise PreventUpdate

    try:
        point_data = click_data["points"][0]
        raw_x = point_data.get("x")
        raw_y = point_data.get("y")

        if not node_ids:
            raise PreventUpdate

        clicked_doc = None

        # Prefer row (y) document as the reference; fall back to column (x)
        if isinstance(raw_y, str) and raw_y in node_ids:
            clicked_doc = raw_y
        elif isinstance(raw_x, str) and raw_x in node_ids:
            clicked_doc = raw_x
        # If numeric indices, map to node_ids list (use y first)
        elif isinstance(raw_y, (int, float)):
            idx = int(raw_y)
            if 0 <= idx < len(node_ids):
                clicked_doc = node_ids[idx]
        elif isinstance(raw_x, (int, float)):
            idx = int(raw_x)
            if 0 <= idx < len(node_ids):
                clicked_doc = node_ids[idx]
        else:
            # Fallback: try pointIndex/pointIndices (Plotly heatmap provides [row, col])
            point_idx = point_data.get("pointIndex") or point_data.get("pointIndices")
            if isinstance(point_idx, (list, tuple)) and len(point_idx) >= 1:
                row_idx_val = point_idx[0]
                if isinstance(row_idx_val, (int, float)):
                    row_idx_val = int(row_idx_val)
                    if 0 <= row_idx_val < len(node_ids):
                        clicked_doc = node_ids[row_idx_val]

        if not clicked_doc:
            raise PreventUpdate

        # Validate that the clicked document exists in node store
        if not graph_store.get_node(clicked_doc):
            print(f"Warning: Clicked document not found in store: {clicked_doc}")
            raise PreventUpdate

        # Update both dropdown and selected node store
        return clicked_doc, clicked_doc
    except Exception as e:
        print(f"Error in handle_heatmap_click: {e}")
        raise PreventUpdate


@callback(
    Output("document-selector", "value", allow_duplicate=True),
    Output("selected-node-store", "data", allow_duplicate=True),
    Input("dep-network-graph", "clickData"),
    prevent_initial_call=True,
)
def handle_dep_network_click(click_data):
    """Handle clicks on dependency network nodes and select the document."""
    if not click_data or "points" not in click_data or len(click_data["points"]) == 0:
        raise PreventUpdate

    try:
        point_data = click_data["points"][0]
        # Plotly scatter node data stores the node ID in the customdata or label
        clicked_node = point_data.get("customdata") or point_data.get("text")

        if not clicked_node:
            raise PreventUpdate

        # Validate that the clicked node exists in graph store
        if not graph_store.get_node(clicked_node):
            print(f"Warning: Clicked node not found in store: {clicked_node}")
            raise PreventUpdate

        # Update both dropdown and selected node store
        return clicked_node, clicked_node
    except Exception as e:
        print(f"Error in handle_dep_network_click: {e}")
        raise PreventUpdate


@callback(
    Output("document-selector", "options", allow_duplicate=True),
    Input("selected-node-store", "data"),
    State("document-selector", "options"),
    prevent_initial_call=True,
)
def ensure_dropdown_has_selected(selected_doc, current_options):
    """Ensure the dropdown options contain the currently selected document for visibility."""
    if not selected_doc:
        raise PreventUpdate

    options = current_options or []
    if any(opt.get("value") == selected_doc for opt in options):
        return options

    # Prepend selected document to keep it visible without exploding option list
    new_options = [{"label": selected_doc, "value": selected_doc}] + options
    return new_options


@callback(
    Output("perf-log-sink", "children"),
    Input("performance-store", "data"),
    prevent_initial_call=True,
)
def log_performance(perf_data):
    """Persist performance snapshots to log for later benchmarking."""
    if not perf_data:
        raise PreventUpdate

    try:
        log_path = Path(LOGS_DIR) / "perf_metrics.log"
        log_entry = json.dumps(perf_data)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
    except Exception as e:
        print(f"Perf log write failed: {e}")

    return ""


# ============================================================================
# App Layout
# ============================================================================


def create_layout():
    """Create the main application layout."""

    graph_store.load_metadata()
    metadata = graph_store.get_metadata()
    node_ids = graph_store.get_node_ids()

    # Set up ChromaDB collection for document text retrieval
    # Use chunks collection since parent chunks contain the full document text
    try:
        client = PersistentClient(path=CHROMA_PATH)
        # Use the chunks collection which has parent chunks with full text
        collection = client.get_collection(RAG_CONFIG.chunk_collection_name)
        graph_store.set_collection(collection)
    except Exception as e:
        print(f"Warning: Could not load ChromaDB collection for document text: {e}")

    global graph_filter

    # Build graph_data from SQLiteGraphStore for GraphFilter
    node_ids = graph_store.get_node_ids()
    nodes_dict = {}
    for node_id in node_ids:
        node_data = graph_store.get_node(node_id)
        if node_data:
            nodes_dict[node_id] = node_data

    edges_list = graph_store.get_edges()
    clusters_dict = graph_store.get_clusters()

    graph_data = {
        "nodes": nodes_dict,
        "edges": edges_list,
        "clusters": clusters_dict,
    }

    graph_filter = GraphFilter(graph_data)

    doc_types = graph_filter.get_available_doc_types()
    languages = graph_filter.get_available_languages()
    repositories = graph_filter.get_available_repositories()

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1(
                                "🏛️ Governance Consistency Graph Dashboard",
                                style={"display": "inline-block", "marginRight": "20px"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
                    ),
                    html.P(
                        f"{len(node_ids)} documents • {len(graph_store.get_edges())} relationships • Memory-optimised with Plotly Dash"
                    ),
                ],
                className="header",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(f"{len(node_ids)}", className="metric-value"),
                                            html.Div("Documents", className="metric-label"),
                                        ],
                                        className="metric",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                f"{len(graph_store.get_edges())}",
                                                className="metric-value",
                                            ),
                                            html.Div("Relationships", className="metric-label"),
                                        ],
                                        className="metric",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                f"{metadata.get('build_duration_seconds', 'N/A')}s",
                                                className="metric-value",
                                            ),
                                            html.Div("Build Time", className="metric-label"),
                                        ],
                                        className="metric",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                f"{metadata.get('max_neighbours_used', 'N/A')}",
                                                className="metric-value",
                                            ),
                                            html.Div("Max Neighbours", className="metric-label"),
                                        ],
                                        className="metric",
                                    ),
                                ],
                                style={"marginBottom": "16px"},
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "⏱️ Built: " + metadata.get("built_at", "N/A"),
                                        style={"fontSize": "12px", "color": "#999"},
                                    ),
                                ],
                                style={"marginBottom": "16px"},
                            ),
                        ],
                        className="card",
                    ),
                    html.Div(
                        [
                            html.H3("Filters & Controls", style={"marginTop": "0"}),
                            html.Div(
                                [
                                    html.Button(
                                        "▸ Show Filters",
                                        id="toggle-main-filters-btn",
                                        n_clicks=0,
                                        style={
                                            "padding": "6px 12px",
                                            "backgroundColor": "#6c757d",
                                            "color": "white",
                                            "border": "none",
                                            "borderRadius": "4px",
                                            "cursor": "pointer",
                                            "marginBottom": "12px",
                                            "fontSize": "12px",
                                        },
                                    ),
                                    html.Div(
                                        id="filters-summary-caption",
                                        children="",
                                        style={
                                            "fontSize": "12px",
                                            "color": "#666",
                                            "marginTop": "4px",
                                            "marginBottom": "8px",
                                        },
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        [
                                            html.Span(
                                                "Min Conflict Score:",
                                                title="Filter out nodes below a minimum conflict score (0-1)",
                                                style={"cursor": "help"},
                                            ),
                                            html.Span(
                                                "\u2139\ufe0f",
                                                title="Conflict score is computed per node; increase to keep only higher-severity items.",
                                                style={
                                                    "marginLeft": "6px",
                                                    "cursor": "help",
                                                    "color": "#667eea",
                                                },
                                            ),
                                            dcc.Slider(
                                                id="min-conflict-slider",
                                                min=0,
                                                max=1,
                                                step=0.1,
                                                value=0,
                                                marks={i / 10: f"{i / 10:.1f}" for i in range(11)},
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": False,
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Label(
                                        [
                                            html.Span(
                                                "Similarity Threshold:",
                                                title="Increase to reduce edges by requiring stronger similarity",
                                                style={"cursor": "help"},
                                            ),
                                            html.Span(
                                                "\u2139\ufe0f",
                                                title="Raising the threshold reduces edge clutter and focuses on stronger relationships.",
                                                style={
                                                    "marginLeft": "6px",
                                                    "cursor": "help",
                                                    "color": "#667eea",
                                                },
                                            ),
                                            dcc.Slider(
                                                id="sim-threshold-slider",
                                                min=0.1,
                                                max=1.0,
                                                step=0.05,
                                                value=0.4,
                                                marks={
                                                    i / 20: f"{i / 20:.2f}" for i in range(2, 21, 2)
                                                },
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": False,
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                [
                                                    html.Span(
                                                        "Document Types:",
                                                        title="Filter by document types (e.g., policy, standard, code)",
                                                        style={"cursor": "help"},
                                                    ),
                                                    html.Span(
                                                        "\u2139\ufe0f",
                                                        title="Select one or more types to narrow the graph.",
                                                        style={
                                                            "marginLeft": "6px",
                                                            "cursor": "help",
                                                            "color": "#667eea",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            dcc.Dropdown(
                                                id="doc-type-filter",
                                                options=[
                                                    {"label": dt, "value": dt}
                                                    for dt in sorted(doc_types)
                                                ],
                                                value=[],
                                                multi=True,
                                                placeholder="Select document types...",
                                            ),
                                        ],
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                [
                                                    html.Span(
                                                        "Programming Languages:",
                                                        title="Filter by programming language for code nodes",
                                                        style={"cursor": "help"},
                                                    ),
                                                    html.Span(
                                                        "\u2139\ufe0f",
                                                        title="Useful when focusing code-only views.",
                                                        style={
                                                            "marginLeft": "6px",
                                                            "cursor": "help",
                                                            "color": "#667eea",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            dcc.Dropdown(
                                                id="language-filter",
                                                options=[
                                                    {"label": lang, "value": lang}
                                                    for lang in sorted(languages)
                                                ],
                                                value=[],
                                                multi=True,
                                                placeholder="Select languages...",
                                            ),
                                        ],
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                [
                                                    html.Span(
                                                        "Repositories:",
                                                        title="Filter by repository source (Git repo)",
                                                        style={"cursor": "help"},
                                                    ),
                                                    html.Span(
                                                        "\u2139\ufe0f",
                                                        title="Repository options come from node metadata or inferred from IDs.",
                                                        style={
                                                            "marginLeft": "6px",
                                                            "cursor": "help",
                                                            "color": "#667eea",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            dcc.Dropdown(
                                                id="repository-filter",
                                                options=[
                                                    {"label": repo, "value": repo}
                                                    for repo in sorted(repositories)
                                                ],
                                                value=[],
                                                multi=True,
                                                placeholder="Select repositories...",
                                            ),
                                        ],
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                [
                                                    html.Span(
                                                        "Topic Clusters:",
                                                        title="Filter by topic clusters (unsupervised groupings)",
                                                        style={"cursor": "help"},
                                                    ),
                                                    html.Span(
                                                        "\u2139\ufe0f",
                                                        title="Select cluster IDs to focus on certain topics.",
                                                        style={
                                                            "marginLeft": "6px",
                                                            "cursor": "help",
                                                            "color": "#667eea",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            dcc.Dropdown(
                                                id="topic-cluster-filter",
                                                options=[],
                                                value=[],
                                                multi=True,
                                                placeholder="Select topic clusters...",
                                            ),
                                        ],
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                [
                                                    html.Span(
                                                        "Risk Clusters:",
                                                        title="Filter by risk clusters (model-derived risk buckets)",
                                                        style={"cursor": "help"},
                                                    ),
                                                    html.Span(
                                                        "\u2139\ufe0f",
                                                        title="Select cluster IDs to focus on risk areas.",
                                                        style={
                                                            "marginLeft": "6px",
                                                            "cursor": "help",
                                                            "color": "#667eea",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            dcc.Dropdown(
                                                id="risk-cluster-filter",
                                                options=[],
                                                value=[],
                                                multi=True,
                                                placeholder="Select risk clusters...",
                                            ),
                                        ],
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                [
                                                    html.Span(
                                                        "Cluster Dropdown Options:",
                                                        style={"fontWeight": 500},
                                                    ),
                                                    html.Span(
                                                        "\u2139\ufe0f",
                                                        title="Toggle whether size-1 clusters are included in the Topic/Risk dropdowns.",
                                                        style={
                                                            "marginLeft": "6px",
                                                            "cursor": "help",
                                                            "color": "#667eea",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            dcc.Checklist(
                                                id="show-singleton-clusters-toggle",
                                                options=[
                                                    {
                                                        "label": " Show size 1 clusters",
                                                        "value": "show_singleton",
                                                    }
                                                ],
                                                value=[],
                                                inline=True,
                                                style={"marginTop": "8px", "fontSize": "14px"},
                                            ),
                                        ],
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                [
                                                    html.Span(
                                                        "Relationship Types:",
                                                        title="Filter edges by relationship status",
                                                        style={"cursor": "help"},
                                                    ),
                                                    html.Span(
                                                        "\u2139\ufe0f",
                                                        title="Consistent, Duplicate, Partial Conflict, or Conflict relationships.",
                                                        style={
                                                            "marginLeft": "6px",
                                                            "cursor": "help",
                                                            "color": "#667eea",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            dcc.Dropdown(
                                                id="relationship-type-filter",
                                                options=[
                                                    {
                                                        "label": "Consistent (green)",
                                                        "value": "consistent",
                                                    },
                                                    {
                                                        "label": "Duplicate (blue)",
                                                        "value": "duplicate",
                                                    },
                                                    {
                                                        "label": "Partial Conflict (amber)",
                                                        "value": "partial_conflict",
                                                    },
                                                    {
                                                        "label": "Conflict (red)",
                                                        "value": "conflict",
                                                    },
                                                ],
                                                value=[],
                                                multi=True,
                                                placeholder="Select relationship types...",
                                            ),
                                        ],
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "Apply Filters",
                                                id="apply-filters-btn",
                                                n_clicks=0,
                                                style={
                                                    "marginRight": "8px",
                                                    "padding": "8px 16px",
                                                    "backgroundColor": "#667eea",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                },
                                                title="Apply selected filters to the graph",
                                            ),
                                            html.Button(
                                                "Reset All",
                                                id="reset-filters-btn",
                                                n_clicks=0,
                                                style={
                                                    "padding": "8px 16px",
                                                    "backgroundColor": "#999",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                },
                                                title="Reset all filters to defaults",
                                            ),
                                        ],
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                id="filter-stats",
                                                style={
                                                    "fontSize": "12px",
                                                    "color": "#666",
                                                    "marginTop": "8px",
                                                },
                                            ),
                                        ]
                                    ),
                                ],
                                id="filters-main-panel",
                                style={"display": "none"},
                            ),
                            html.Div(
                                [
                                    html.Label("Select Document:"),
                                    dcc.Dropdown(
                                        id="document-selector",
                                        options=[
                                            {
                                                "label": get_display_name(
                                                    graph_store.get_node(nid) or {}, nid
                                                ),
                                                "value": nid,
                                            }
                                            for nid in node_ids[:100]
                                        ],
                                        placeholder="Choose a document...",
                                        searchable=True,
                                    ),
                                ],
                                style={
                                    "marginBottom": "16px",
                                    "marginTop": "16px",
                                    "paddingTop": "16px",
                                    "borderTop": "1px solid #eee",
                                },
                            ),
                        ],
                        className="card",
                    ),
                    dcc.Store(id="filter-state-store", data={"filtered_nodes": []}),
                    dcc.Store(id="selected-node-store", storage_type="session"),
                    dcc.Store(id="heatmap-node-ids-store", data=[]),
                    dcc.Store(id="heatmap-clicked-cell-store", data=None),
                    dcc.Store(id="performance-store", storage_type="session"),
                    dcc.Store(id="fps-store", storage_type="session"),
                    html.Div(id="perf-log-sink", style={"display": "none"}),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Button(
                                        "📊 Graph",
                                        id="tab-graph",
                                        n_clicks=0,
                                        className="tab-button active",
                                    ),
                                    html.Button(
                                        "🔥 Heatmap",
                                        id="tab-heatmap",
                                        n_clicks=0,
                                        className="tab-button",
                                    ),
                                    html.Button(
                                        "🔗 Dependencies",
                                        id="tab-dependencies",
                                        n_clicks=0,
                                        className="tab-button",
                                    ),
                                    html.Button(
                                        "📄 Details",
                                        id="tab-details",
                                        n_clicks=0,
                                        className="tab-button",
                                    ),
                                    html.Button(
                                        "🔎 Query",
                                        id="tab-query",
                                        n_clicks=0,
                                        className="tab-button",
                                    ),
                                    html.Button(
                                        "💬 Conversations",
                                        id="tab-conversations",
                                        n_clicks=0,
                                        className="tab-button",
                                    ),
                                    html.Button(
                                        "📚 References",
                                        id="tab-academic-references",
                                        n_clicks=0,
                                        className="tab-button",
                                    ),
                                    html.Button(
                                        "🕸️ Citation Graph",
                                        id="tab-citation-graph",
                                        n_clicks=0,
                                        className="tab-button",
                                    ),
                                    html.Button(
                                        "🎓 Assessment",
                                        id="tab-assessment",
                                        n_clicks=0,
                                        className="tab-button",
                                    ),
                                    html.Button(
                                        "� Graph Analytics",
                                        id="tab-graph-analytics",
                                        n_clicks=0,
                                        className="tab-button",
                                    ),
                                    html.Button(
                                        "🚀 Metrics",
                                        id="tab-metrics",
                                        n_clicks=0,
                                        className="tab-button",
                                    ),
                                    html.Button(
                                        "⚡ Benchmarks",
                                        id="tab-benchmarks",
                                        n_clicks=0,
                                        className="tab-button",
                                    ),
                                ],
                                className="tabs",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Page:"),
                                            html.Div(
                                                [
                                                    html.Button(
                                                        "◀ Previous",
                                                        id="graph-page-prev",
                                                        n_clicks=0,
                                                        style={
                                                            "padding": "8px 16px",
                                                            "marginRight": "8px",
                                                            "border": "1px solid #667eea",
                                                            "background": "white",
                                                            "color": "#667eea",
                                                            "borderRadius": "4px",
                                                            "cursor": "pointer",
                                                            "fontSize": "14px",
                                                            "fontWeight": "500",
                                                        },
                                                    ),
                                                    html.Button(
                                                        "Next ▶",
                                                        id="graph-page-next",
                                                        n_clicks=0,
                                                        style={
                                                            "padding": "8px 16px",
                                                            "border": "1px solid #667eea",
                                                            "background": "white",
                                                            "color": "#667eea",
                                                            "borderRadius": "4px",
                                                            "cursor": "pointer",
                                                            "fontSize": "14px",
                                                            "fontWeight": "500",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="graph-page-input",
                                                        type="number",
                                                        placeholder="Go to page...",
                                                        min=1,
                                                        style={
                                                            "padding": "8px 12px",
                                                            "border": "1px solid #ccc",
                                                            "borderRadius": "4px",
                                                            "fontSize": "14px",
                                                            "width": "120px",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "marginBottom": "8px",
                                                    "display": "flex",
                                                    "gap": "8px",
                                                    "alignItems": "center",
                                                },
                                            ),
                                            dcc.Slider(
                                                id="graph-page-slider",
                                                min=0,
                                                max=max(0, (len(node_ids) - 1) // NODES_PER_PAGE),
                                                step=1,
                                                value=0,
                                                marks={
                                                    i: f"Page {i + 1}"
                                                    for i in range(
                                                        0,
                                                        max(
                                                            0, (len(node_ids) - 1) // NODES_PER_PAGE
                                                        )
                                                        + 1,
                                                        max(
                                                            1,
                                                            (len(node_ids) // NODES_PER_PAGE) // 5,
                                                        ),
                                                    )
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                "Nodes per Page:",
                                                style={"fontWeight": "500", "marginBottom": "4px"},
                                            ),
                                            dcc.Dropdown(
                                                id="nodes-per-page-selector",
                                                options=[
                                                    {"label": "10 nodes", "value": 10},
                                                    {"label": "25 nodes", "value": 25},
                                                    {"label": "50 nodes (default)", "value": 50},
                                                    {"label": "100 nodes", "value": 100},
                                                    {"label": "250 nodes", "value": 250},
                                                    {"label": "500 nodes", "value": 500},
                                                    {
                                                        "label": "All nodes (single page)",
                                                        "value": 999999,
                                                    },
                                                ],
                                                value=50,
                                                style={"width": "200px"},
                                            ),
                                        ],
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Layout Algorithm:",
                                                        style={
                                                            "fontWeight": "500",
                                                            "marginBottom": "4px",
                                                        },
                                                    ),
                                                    dcc.RadioItems(
                                                        id="layout-selector",
                                                        options=[
                                                            {
                                                                "label": " Force-Directed (Default)",
                                                                "value": "force",
                                                            },
                                                            {
                                                                "label": " Hierarchical (DAGs)",
                                                                "value": "hierarchical",
                                                            },
                                                            {
                                                                "label": " Circular (Clusters)",
                                                                "value": "circular",
                                                            },
                                                        ],
                                                        value="force",
                                                        style={
                                                            "display": "flex",
                                                            "gap": "16px",
                                                            "marginBottom": "8px",
                                                        },
                                                    ),
                                                    html.Button(
                                                        "🔄 Regenerate Layout",
                                                        id="regenerate-layout-btn",
                                                        n_clicks=0,
                                                        style={
                                                            "padding": "6px 12px",
                                                            "fontSize": "13px",
                                                            "backgroundColor": "#8b9dc3",
                                                            "color": "white",
                                                            "border": "none",
                                                            "borderRadius": "4px",
                                                            "cursor": "pointer",
                                                        },
                                                    ),
                                                    dcc.Store(id="layout-seed-store", data=42),
                                                ],
                                                style={
                                                    "display": "inline-block",
                                                    "marginRight": "32px",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Rendering Options:",
                                                        style={
                                                            "fontWeight": "500",
                                                            "marginBottom": "4px",
                                                        },
                                                    ),
                                                    dcc.Checklist(
                                                        id="webgl-toggle",
                                                        options=[
                                                            {
                                                                "label": " Enable WebGL (GPU acceleration)",
                                                                "value": "webgl",
                                                            }
                                                        ],
                                                        value=[],
                                                        style={"marginBottom": "8px"},
                                                    ),
                                                    dcc.Checklist(
                                                        id="show-node-names-toggle",
                                                        options=[
                                                            {
                                                                "label": " Show Node Names",
                                                                "value": "show-names",
                                                            }
                                                        ],
                                                        value=["show-names"],
                                                        style={"marginBottom": "8px"},
                                                    ),
                                                    dcc.Checklist(
                                                        id="show-unlinked-nodes-toggle",
                                                        options=[
                                                            {
                                                                "label": " Show Unlinked Nodes",
                                                                "value": "show-unlinked",
                                                            }
                                                        ],
                                                        value=[],
                                                        style={"marginBottom": "12px"},
                                                    ),
                                                    dcc.Interval(
                                                        id="fps-interval",
                                                        interval=500,
                                                        n_intervals=0,
                                                    ),
                                                    html.Div(
                                                        id="render-stats",
                                                        style={
                                                            "fontSize": "12px",
                                                            "color": "#999",
                                                            "marginTop": "4px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        id="perf-panel",
                                                        style={
                                                            "marginTop": "8px",
                                                            "padding": "8px",
                                                            "backgroundColor": "#f4f6fb",
                                                            "borderRadius": "4px",
                                                            "border": "1px solid #e3e7f3",
                                                        },
                                                    ),
                                                    html.Div(
                                                        id="fps-display",
                                                        style={
                                                            "fontSize": "11px",
                                                            "color": "#555",
                                                            "marginTop": "6px",
                                                        },
                                                    ),
                                                ],
                                                style={"display": "inline-block"},
                                            ),
                                        ],
                                        style={
                                            "marginBottom": "16px",
                                            "padding": "12px",
                                            "backgroundColor": "#f9f9f9",
                                            "borderRadius": "4px",
                                        },
                                    ),
                                    html.Div(id="graph-container", style={"marginTop": "16px"}),
                                ],
                                id="graph-tab",
                                style={"display": "block"},
                                className="card",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        id="document-details-container", style={"marginTop": "16px"}
                                    ),
                                ],
                                id="details-tab",
                                style={"display": "none"},
                                className="card",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                "Nodes to Display:",
                                                style={"fontWeight": "500", "marginBottom": "8px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Button(
                                                        "◀ Less",
                                                        id="heatmap-node-prev",
                                                        n_clicks=0,
                                                        style={
                                                            "padding": "8px 16px",
                                                            "marginRight": "8px",
                                                            "border": "1px solid #667eea",
                                                            "background": "white",
                                                            "color": "#667eea",
                                                            "borderRadius": "4px",
                                                            "cursor": "pointer",
                                                            "fontSize": "14px",
                                                            "fontWeight": "500",
                                                        },
                                                    ),
                                                    html.Button(
                                                        "More ▶",
                                                        id="heatmap-node-next",
                                                        n_clicks=0,
                                                        style={
                                                            "padding": "8px 16px",
                                                            "border": "1px solid #667eea",
                                                            "background": "white",
                                                            "color": "#667eea",
                                                            "borderRadius": "4px",
                                                            "cursor": "pointer",
                                                            "fontSize": "14px",
                                                            "fontWeight": "500",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="heatmap-node-count-input",
                                                        type="number",
                                                        placeholder="Go to count...",
                                                        min=10,
                                                        max=200,
                                                        step=10,
                                                        style={
                                                            "padding": "8px 12px",
                                                            "border": "1px solid #ccc",
                                                            "borderRadius": "4px",
                                                            "fontSize": "14px",
                                                            "width": "140px",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "marginBottom": "8px",
                                                    "display": "flex",
                                                    "gap": "8px",
                                                    "alignItems": "center",
                                                },
                                            ),
                                            dcc.Slider(
                                                id="heatmap-node-count",
                                                min=10,
                                                max=200,
                                                step=10,
                                                value=50,
                                                marks={i: str(i) for i in range(10, 210, 20)},
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": True,
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(id="heatmap-container", style={"marginTop": "16px"}),
                                ],
                                id="heatmap-tab",
                                style={"display": "none"},
                                className="card",
                            ),
                            html.Div(
                                [
                                    html.H3("RAG Query Assistant", style={"marginTop": 0}),
                                    html.Div(
                                        id="chromadb-status-alert",
                                        style={
                                            "padding": "12px",
                                            "marginBottom": "12px",
                                            "borderRadius": "4px",
                                            "display": "none",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "🆕 New Conversation",
                                                id="new-conversation-btn",
                                                n_clicks=0,
                                                style={
                                                    "padding": "8px 16px",
                                                    "backgroundColor": "#6c757d",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "marginBottom": "12px",
                                                    "marginRight": "8px",
                                                },
                                            ),
                                            html.Button(
                                                "📋 Templates",
                                                id="toggle-templates-btn",
                                                n_clicks=0,
                                                style={
                                                    "padding": "8px 16px",
                                                    "backgroundColor": "#17a2b8",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "marginBottom": "12px",
                                                },
                                            ),
                                            html.Span(
                                                id="conversation-info",
                                                style={
                                                    "fontSize": "12px",
                                                    "color": "#666",
                                                    "marginLeft": "16px",
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "12px"},
                                    ),
                                    html.Div(
                                        [
                                            html.H5(
                                                "📋 Query Templates",
                                                style={"marginTop": 0, "marginBottom": "10px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Category",
                                                        style={
                                                            "fontWeight": 500,
                                                            "fontSize": "12px",
                                                        },
                                                    ),
                                                    dcc.Dropdown(
                                                        id="template-category-select",
                                                        options=[
                                                            {"label": "All", "value": "all"},
                                                            {
                                                                "label": "Governance",
                                                                "value": "governance",
                                                            },
                                                            {
                                                                "label": "Architecture",
                                                                "value": "architecture",
                                                            },
                                                            {
                                                                "label": "Security",
                                                                "value": "security",
                                                            },
                                                            {"label": "Code", "value": "code"},
                                                            {
                                                                "label": "Operations",
                                                                "value": "operations",
                                                            },
                                                            {
                                                                "label": "Academic",
                                                                "value": "academic",
                                                            },
                                                        ],
                                                        value="all",
                                                        style={"marginBottom": "8px"},
                                                    ),
                                                ],
                                                style={"marginBottom": "10px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Template",
                                                        style={
                                                            "fontWeight": 500,
                                                            "fontSize": "12px",
                                                        },
                                                    ),
                                                    dcc.Dropdown(
                                                        id="template-select",
                                                        options=[
                                                            {
                                                                "label": "Select a template...",
                                                                "value": "",
                                                            }
                                                        ],
                                                        value="",
                                                        style={"marginBottom": "8px"},
                                                    ),
                                                ],
                                                style={"marginBottom": "10px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Template Parameters",
                                                        style={
                                                            "fontWeight": 500,
                                                            "fontSize": "12px",
                                                        },
                                                    ),
                                                    dcc.Textarea(
                                                        id="template-params-input",
                                                        placeholder="Enter parameters separated by commas (e.g., ServiceA, ServiceB)",
                                                        style={
                                                            "width": "100%",
                                                            "height": "50px",
                                                            "borderRadius": "4px",
                                                            "fontSize": "12px",
                                                        },
                                                    ),
                                                ],
                                                style={"marginBottom": "10px"},
                                            ),
                                            html.Button(
                                                "✓ Use Template",
                                                id="apply-template-btn",
                                                n_clicks=0,
                                                style={
                                                    "padding": "6px 12px",
                                                    "backgroundColor": "#28a745",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "fontSize": "12px",
                                                },
                                            ),
                                        ],
                                        id="templates-panel",
                                        style={
                                            "display": "none",
                                            "backgroundColor": "#f0f8ff",
                                            "padding": "12px",
                                            "borderRadius": "4px",
                                            "marginBottom": "12px",
                                            "border": "1px solid #b3d9ff",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "🎯 Advanced Filters",
                                                id="toggle-filters-btn",
                                                n_clicks=0,
                                                style={
                                                    "padding": "8px 16px",
                                                    "backgroundColor": "#6c757d",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "marginBottom": "12px",
                                                },
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.H5(
                                                "🎯 Advanced Filters",
                                                style={"marginTop": 0, "marginBottom": "10px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Custom System Role (Optional)",
                                                        style={
                                                            "fontWeight": 500,
                                                            "fontSize": "12px",
                                                        },
                                                    ),
                                                    dcc.Textarea(
                                                        id="custom-role-input",
                                                        placeholder="Enter a custom system role/prompt, or leave blank to use default...",
                                                        style={
                                                            "width": "100%",
                                                            "height": "60px",
                                                            "borderRadius": "4px",
                                                            "fontSize": "12px",
                                                        },
                                                    ),
                                                    html.Small(
                                                        "Default: 'You are a technical assistant specializing in governance, security, and infrastructure policies.' (or code-specific variant)",
                                                        style={"color": "#666", "fontSize": "11px"},
                                                    ),
                                                ],
                                                style={"marginBottom": "12px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Date Range Filter",
                                                        style={
                                                            "fontWeight": 500,
                                                            "fontSize": "12px",
                                                        },
                                                    ),
                                                    dcc.DatePickerRange(
                                                        id="date-range-filter",
                                                        start_date_placeholder_text="Start Date",
                                                        end_date_placeholder_text="End Date",
                                                        style={"marginBottom": "8px"},
                                                    ),
                                                ],
                                                style={"marginBottom": "12px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Minimum Confidence Score (%)",
                                                        style={
                                                            "fontWeight": 500,
                                                            "fontSize": "12px",
                                                        },
                                                    ),
                                                    dcc.Slider(
                                                        id="confidence-filter",
                                                        min=0,
                                                        max=100,
                                                        step=5,
                                                        value=0,
                                                        marks={0: "0%", 50: "50%", 100: "100%"},
                                                        tooltip={
                                                            "placement": "bottom",
                                                            "always_visible": False,
                                                        },
                                                    ),
                                                ],
                                                style={"marginBottom": "12px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Result Type",
                                                        style={
                                                            "fontWeight": 500,
                                                            "fontSize": "12px",
                                                        },
                                                    ),
                                                    dcc.Checklist(
                                                        id="result-type-filter",
                                                        options=[
                                                            {
                                                                "label": " Documents",
                                                                "value": "documents",
                                                            },
                                                            {"label": " Code", "value": "code"},
                                                        ],
                                                        value=["documents", "code"],
                                                        style={"marginBottom": "8px"},
                                                    ),
                                                ],
                                                style={"marginBottom": "12px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Tags Filter (comma-separated)",
                                                        style={
                                                            "fontWeight": 500,
                                                            "fontSize": "12px",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="tags-filter-input",
                                                        type="text",
                                                        placeholder="e.g., security, java, api",
                                                        style={
                                                            "width": "100%",
                                                            "borderRadius": "4px",
                                                            "fontSize": "12px",
                                                        },
                                                    ),
                                                ],
                                                style={"marginBottom": "12px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Button(
                                                        "Reset Filters",
                                                        id="reset-advanced-filters-btn",
                                                        n_clicks=0,
                                                        style={
                                                            "padding": "6px 12px",
                                                            "backgroundColor": "#dc3545",
                                                            "color": "white",
                                                            "border": "none",
                                                            "borderRadius": "4px",
                                                            "cursor": "pointer",
                                                            "fontSize": "12px",
                                                        },
                                                    ),
                                                ]
                                            ),
                                        ],
                                        id="filters-panel",
                                        style={
                                            "display": "none",
                                            "backgroundColor": "#fff3cd",
                                            "padding": "12px",
                                            "borderRadius": "4px",
                                            "marginBottom": "12px",
                                            "border": "1px solid #ffd966",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Question"),
                                            dcc.Textarea(
                                                id="rag-query-input",
                                                placeholder="Ask about governance, security, or code...",
                                                style={
                                                    "width": "100%",
                                                    "height": "80px",
                                                    "borderRadius": "6px",
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "12px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Chunks (k)", style={"fontWeight": 500}
                                                    ),
                                                    dcc.Slider(
                                                        id="rag-query-k",
                                                        min=1,
                                                        max=15,
                                                        step=1,
                                                        value=RAG_CONFIG.k_results,
                                                        marks={1: "1", 5: "5", 10: "10", 15: "15"},
                                                    ),
                                                ],
                                                style={"marginBottom": "12px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Temperature", style={"fontWeight": 500}
                                                    ),
                                                    dcc.Slider(
                                                        id="rag-query-temp",
                                                        min=0.0,
                                                        max=1.0,
                                                        step=0.05,
                                                        value=RAG_CONFIG.temperature,
                                                        marks={
                                                            0.0: "0.0",
                                                            0.3: "0.3",
                                                            0.6: "0.6",
                                                            1.0: "1.0",
                                                        },
                                                    ),
                                                ],
                                                style={"marginBottom": "12px"},
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Checklist(
                                                        id="rag-query-code-aware",
                                                        options=[
                                                            {
                                                                "label": " Code-Aware Context (detect & format code responses)",
                                                                "value": "enable",
                                                            }
                                                        ],
                                                        value=["enable"],
                                                        style={"marginBottom": "12px"},
                                                    ),
                                                ],
                                                style={"marginBottom": "12px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Persona (Academic Mode)",
                                                        style={
                                                            "fontWeight": 500,
                                                            "marginBottom": "6px",
                                                        },
                                                    ),
                                                    dcc.Dropdown(
                                                        id="rag-query-persona",
                                                        options=[
                                                            {"label": "None", "value": "none"},
                                                            {
                                                                "label": "👨‍🎓 Supervisor - Strict quality filtering",
                                                                "value": "supervisor",
                                                            },
                                                            {
                                                                "label": "🔍 Researcher - Discovery-focused",
                                                                "value": "researcher",
                                                            },
                                                            {
                                                                "label": "📋 Assessor - Balanced approach",
                                                                "value": "assessor",
                                                            },
                                                        ],
                                                        value="none",
                                                        style={"width": "100%"},
                                                        clearable=False,
                                                        persistence=True,
                                                        persistence_type="session",
                                                    ),
                                                ],
                                                style={"marginBottom": "12px"},
                                            ),
                                            html.Button(
                                                "Send Query",
                                                id="rag-query-run",
                                                n_clicks=0,
                                                style={
                                                    "padding": "10px 18px",
                                                    "backgroundColor": "#667eea",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                },
                                            ),
                                            html.Span(
                                                id="rag-query-status",
                                                style={"marginLeft": "10px", "color": "#666"},
                                            ),
                                        ],
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.H4(
                                                "Conversation History",
                                                style={
                                                    "marginTop": "0",
                                                    "borderBottom": "1px solid #eee",
                                                    "paddingBottom": "8px",
                                                },
                                            ),
                                            html.Div(
                                                id="conversation-history",
                                                style={
                                                    "maxHeight": "400px",
                                                    "overflowY": "auto",
                                                    "fontSize": "13px",
                                                    "lineHeight": "1.6",
                                                    "marginBottom": "16px",
                                                    "backgroundColor": "#f9f9f9",
                                                    "borderRadius": "4px",
                                                    "padding": "12px",
                                                },
                                            ),
                                        ],
                                        className="card",
                                        style={"marginBottom": "12px"},
                                    ),
                                    html.Div(
                                        [
                                            html.H4("Current Response", style={"marginTop": "0"}),
                                            dcc.Markdown(
                                                id="rag-query-answer",
                                                style={
                                                    "whiteSpace": "pre-wrap",
                                                    "lineHeight": "1.5",
                                                },
                                            ),
                                            # Relevancy Feedback Section
                                            html.Div(
                                                [
                                                    html.Hr(style={"margin": "16px 0"}),
                                                    html.Div(
                                                        [
                                                            html.Strong(
                                                                "Rate this response: ",
                                                                style={
                                                                    "marginRight": "12px",
                                                                    "fontSize": "14px",
                                                                },
                                                            ),
                                                            html.Button(
                                                                "★",
                                                                id="rating-1",
                                                                n_clicks=0,
                                                                style={
                                                                    "fontSize": "24px",
                                                                    "background": "none",
                                                                    "border": "none",
                                                                    "cursor": "pointer",
                                                                    "color": "#ddd",
                                                                    "padding": "0 4px",
                                                                },
                                                            ),
                                                            html.Button(
                                                                "★",
                                                                id="rating-2",
                                                                n_clicks=0,
                                                                style={
                                                                    "fontSize": "24px",
                                                                    "background": "none",
                                                                    "border": "none",
                                                                    "cursor": "pointer",
                                                                    "color": "#ddd",
                                                                    "padding": "0 4px",
                                                                },
                                                            ),
                                                            html.Button(
                                                                "★",
                                                                id="rating-3",
                                                                n_clicks=0,
                                                                style={
                                                                    "fontSize": "24px",
                                                                    "background": "none",
                                                                    "border": "none",
                                                                    "cursor": "pointer",
                                                                    "color": "#ddd",
                                                                    "padding": "0 4px",
                                                                },
                                                            ),
                                                            html.Button(
                                                                "★",
                                                                id="rating-4",
                                                                n_clicks=0,
                                                                style={
                                                                    "fontSize": "24px",
                                                                    "background": "none",
                                                                    "border": "none",
                                                                    "cursor": "pointer",
                                                                    "color": "#ddd",
                                                                    "padding": "0 4px",
                                                                },
                                                            ),
                                                            html.Button(
                                                                "★",
                                                                id="rating-5",
                                                                n_clicks=0,
                                                                style={
                                                                    "fontSize": "24px",
                                                                    "background": "none",
                                                                    "border": "none",
                                                                    "cursor": "pointer",
                                                                    "color": "#ddd",
                                                                    "padding": "0 4px",
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "display": "flex",
                                                            "alignItems": "center",
                                                            "marginBottom": "12px",
                                                        },
                                                    ),
                                                    dcc.Textarea(
                                                        id="relevancy-feedback-text",
                                                        placeholder="Optional: Provide feedback on this response...",
                                                        style={
                                                            "width": "100%",
                                                            "height": "60px",
                                                            "padding": "8px",
                                                            "fontSize": "13px",
                                                            "border": "1px solid #ddd",
                                                            "borderRadius": "4px",
                                                            "resize": "vertical",
                                                            "display": "none",  # Hidden until rating is given
                                                        },
                                                    ),
                                                    html.Div(
                                                        id="relevancy-feedback-status",
                                                        style={
                                                            "marginTop": "8px",
                                                            "fontSize": "12px",
                                                            "color": "#28a745",
                                                            "fontStyle": "italic",
                                                        },
                                                    ),
                                                ],
                                                id="relevancy-feedback-section",
                                                style={"display": "none"},
                                            ),
                                        ],
                                        className="card",
                                        style={"marginBottom": "12px"},
                                    ),
                                    html.Div(
                                        [
                                            html.H4(
                                                "Explainability & Confidence",
                                                style={"marginTop": "0"},
                                            ),
                                            html.Div(
                                                id="rag-query-explainability",
                                                style={"fontSize": "13px"},
                                            ),
                                        ],
                                        className="card",
                                        id="explainability-container",
                                        style={"marginBottom": "12px", "display": "none"},
                                    ),
                                    html.Div(
                                        [
                                            html.H4("Code Preview", style={"marginTop": "0"}),
                                            html.Div(
                                                id="rag-query-code-preview",
                                                style={"display": "none"},
                                            ),
                                        ],
                                        className="card",
                                        id="code-preview-container",
                                        style={"marginBottom": "12px", "display": "none"},
                                    ),
                                    html.Div(
                                        [
                                            html.H4(
                                                "Metadata & Query Info", style={"marginTop": "0"}
                                            ),
                                            html.Ul(
                                                id="rag-query-metadata",
                                                style={"fontSize": "11px", "paddingLeft": "18px"},
                                            ),
                                        ],
                                        className="card",
                                        style={"marginBottom": "12px"},
                                    ),
                                    html.Div(
                                        [
                                            html.H4("Sources", style={"marginTop": "0"}),
                                            html.Div(
                                                id="rag-query-sources",
                                                style={"fontSize": "12px", "color": "#555"},
                                            ),
                                        ],
                                        className="card",
                                    ),
                                    dcc.Store(
                                        id="conversation-store", data={"id": "", "turns": []}
                                    ),
                                    dcc.Store(id="current-query-record-id", data=None),
                                    html.Div(id="copy-feedback", style={"display": "none"}),
                                    dcc.Interval(
                                        id="copy-feedback-timer", interval=2000, disabled=True
                                    ),
                                ],
                                id="query-tab",
                                style={"display": "none"},
                                className="card",
                            ),
                            html.Div(
                                [
                                    html.H3("🔗 Service Dependencies", style={"marginTop": 0}),
                                    html.Div(
                                        [
                                            html.Label("Top Services to Display:"),
                                            html.Div(
                                                [
                                                    html.Button(
                                                        "◀ Less",
                                                        id="dep-node-prev",
                                                        n_clicks=0,
                                                        style={
                                                            "padding": "8px 16px",
                                                            "marginRight": "8px",
                                                            "border": "1px solid #667eea",
                                                            "background": "white",
                                                            "color": "#667eea",
                                                            "borderRadius": "4px",
                                                            "cursor": "pointer",
                                                            "fontSize": "14px",
                                                            "fontWeight": "500",
                                                        },
                                                    ),
                                                    html.Button(
                                                        "More ▶",
                                                        id="dep-node-next",
                                                        n_clicks=0,
                                                        style={
                                                            "padding": "8px 16px",
                                                            "border": "1px solid #667eea",
                                                            "background": "white",
                                                            "color": "#667eea",
                                                            "borderRadius": "4px",
                                                            "cursor": "pointer",
                                                            "fontSize": "14px",
                                                            "fontWeight": "500",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="dep-node-count-input",
                                                        type="number",
                                                        placeholder="Go to count...",
                                                        min=10,
                                                        max=500,
                                                        step=10,
                                                        style={
                                                            "padding": "8px 12px",
                                                            "border": "1px solid #ccc",
                                                            "borderRadius": "4px",
                                                            "fontSize": "14px",
                                                            "width": "140px",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "marginBottom": "8px",
                                                    "display": "flex",
                                                    "gap": "8px",
                                                    "alignItems": "center",
                                                },
                                            ),
                                            dcc.Slider(
                                                id="dep-node-count",
                                                min=10,
                                                max=500,
                                                step=10,
                                                value=50,
                                                marks={i: str(i) for i in range(50, 550, 50)},
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": True,
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Label(
                                                                "Metrics",
                                                                style={
                                                                    "fontWeight": 500,
                                                                    "marginBottom": "8px",
                                                                    "display": "block",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Div(
                                                                        id="dep-metric-services",
                                                                        style={
                                                                            "padding": "8px",
                                                                            "backgroundColor": "#f0f0f0",
                                                                            "borderRadius": "4px",
                                                                            "marginBottom": "8px",
                                                                        },
                                                                    ),
                                                                    html.Div(
                                                                        id="dep-metric-internal-calls",
                                                                        style={
                                                                            "padding": "8px",
                                                                            "backgroundColor": "#f0f0f0",
                                                                            "borderRadius": "4px",
                                                                            "marginBottom": "8px",
                                                                        },
                                                                    ),
                                                                    html.Div(
                                                                        id="dep-metric-shared-deps",
                                                                        style={
                                                                            "padding": "8px",
                                                                            "backgroundColor": "#f0f0f0",
                                                                            "borderRadius": "4px",
                                                                            "marginBottom": "8px",
                                                                        },
                                                                    ),
                                                                    html.Div(
                                                                        id="dep-metric-circular",
                                                                        style={
                                                                            "padding": "8px",
                                                                            "backgroundColor": "#f0f0f0",
                                                                            "borderRadius": "4px",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={"fontSize": "13px"},
                                                            ),
                                                        ],
                                                        style={"marginBottom": "12px"},
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label(
                                                                "Info",
                                                                style={
                                                                    "fontWeight": 500,
                                                                    "marginBottom": "8px",
                                                                    "display": "block",
                                                                },
                                                            ),
                                                            html.Div(
                                                                id="dep-info-message",
                                                                style={
                                                                    "fontSize": "12px",
                                                                    "color": "#666",
                                                                    "paddingLeft": "8px",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                style={"width": "100%", "maxWidth": "200px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Dependency Network",
                                                        style={
                                                            "marginTop": 0,
                                                            "marginBottom": "12px",
                                                        },
                                                    ),
                                                    dcc.Graph(
                                                        id="dep-network-graph",
                                                        style={"height": "600px"},
                                                    ),
                                                ],
                                                style={"flex": "1"},
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "gap": "16px",
                                            "marginBottom": "16px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.H4("Service Call Matrix", style={"marginTop": 0}),
                                            dcc.Graph(
                                                id="dep-service-matrix", style={"height": "500px"}
                                            ),
                                        ],
                                        className="card",
                                        style={"marginBottom": "12px"},
                                    ),
                                ],
                                id="dependencies-tab",
                                style={"display": "none"},
                                className="card",
                            ),
                            html.Div(
                                [
                                    html.H3("💬 Conversation Browser", style={"marginTop": 0}),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Search or filter conversations:",
                                                        style={"fontWeight": 500},
                                                    ),
                                                    dcc.Input(
                                                        id="conversation-search-input",
                                                        type="text",
                                                        placeholder="Search by title or description...",
                                                        style={
                                                            "width": "100%",
                                                            "padding": "8px",
                                                            "borderRadius": "4px",
                                                            "marginBottom": "8px",
                                                        },
                                                    ),
                                                ],
                                                style={"marginBottom": "12px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Conversations:", style={"fontWeight": 500}
                                                    ),
                                                    dcc.Dropdown(
                                                        id="conversation-list-dropdown",
                                                        options=[
                                                            {"label": "Loading...", "value": ""}
                                                        ],
                                                        value="",
                                                        style={"marginBottom": "8px"},
                                                    ),
                                                ],
                                                style={"marginBottom": "12px"},
                                            ),
                                            html.Button(
                                                "📂 Load Conversation",
                                                id="load-conversation-btn",
                                                n_clicks=0,
                                                style={
                                                    "padding": "10px 18px",
                                                    "backgroundColor": "#8b9dc3",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "marginRight": "8px",
                                                },
                                            ),
                                            html.Button(
                                                "🗑️ Delete",
                                                id="delete-conversation-btn",
                                                n_clicks=0,
                                                style={
                                                    "padding": "10px 18px",
                                                    "backgroundColor": "#b85c69",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "marginRight": "8px",
                                                },
                                            ),
                                            html.Button(
                                                " Export Markdown",
                                                id="export-markdown-btn",
                                                n_clicks=0,
                                                style={
                                                    "padding": "10px 18px",
                                                    "backgroundColor": "#7a8288",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "marginRight": "8px",
                                                },
                                            ),
                                            html.Button(
                                                "📕 Export PDF",
                                                id="export-pdf-btn",
                                                n_clicks=0,
                                                style={
                                                    "padding": "10px 18px",
                                                    "backgroundColor": "#c4996b",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                },
                                            ),
                                            dcc.Download(id="download-export"),
                                        ],
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.H4("Conversation Details", style={"marginTop": 0}),
                                            html.Div(
                                                id="conversation-details-display",
                                                style={"fontSize": "13px"},
                                            ),
                                        ],
                                        className="card",
                                        style={"marginBottom": "12px"},
                                    ),
                                    html.Div(
                                        [
                                            html.H4(
                                                "Conversation History Preview",
                                                style={"marginTop": 0},
                                            ),
                                            html.Div(
                                                id="conversation-preview-display",
                                                style={
                                                    "maxHeight": "500px",
                                                    "overflowY": "auto",
                                                    "backgroundColor": "#f9f9f9",
                                                    "padding": "12px",
                                                    "borderRadius": "4px",
                                                    "fontSize": "12px",
                                                },
                                            ),
                                        ],
                                        className="card",
                                    ),
                                ],
                                id="conversation-browser-tab",
                                style={"display": "none"},
                                className="card",
                            ),
                            html.Div(
                                [
                                    html.H3(
                                        "🕸️ Consistency Graph Analytics", style={"marginTop": 0}
                                    ),
                                    html.Div(
                                        [
                                            html.P(
                                                "Network analysis metrics computed from the consistency graph (relationships between governance documents). "
                                                "Separate from academic references analytics — this tab focuses on document consistency patterns.",
                                                style={
                                                    "fontSize": "13px",
                                                    "color": "#666",
                                                    "fontStyle": "italic",
                                                    "marginBottom": "12px",
                                                },
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Button(
                                                        "🔄 Compute Analytics",
                                                        id="compute-analytics-btn",
                                                        n_clicks=0,
                                                        style={
                                                            "padding": "10px 18px",
                                                            "backgroundColor": "#667eea",
                                                            "color": "white",
                                                            "border": "none",
                                                            "borderRadius": "4px",
                                                            "cursor": "pointer",
                                                            "marginBottom": "16px",
                                                        },
                                                    ),
                                                    html.Span(
                                                        id="analytics-status",
                                                        style={
                                                            "marginLeft": "10px",
                                                            "color": "#666",
                                                            "fontSize": "13px",
                                                            "fontWeight": "500",
                                                        },
                                                    ),
                                                ],
                                                style={"marginBottom": "12px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Network Topology", style={"marginTop": "0"}
                                                    ),
                                                    html.Div(
                                                        id="topology-metrics",
                                                        style={"fontSize": "14px"},
                                                    ),
                                                ],
                                                className="card",
                                                style={"marginBottom": "12px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Top Influencers (PageRank)",
                                                        style={"marginTop": "0"},
                                                    ),
                                                    html.Div(
                                                        id="pagerank-table",
                                                        style={"fontSize": "13px"},
                                                    ),
                                                ],
                                                className="card",
                                                style={"marginBottom": "12px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Bridge Nodes (Betweenness Centrality)",
                                                        style={"marginTop": "0"},
                                                    ),
                                                    html.Div(
                                                        id="betweenness-table",
                                                        style={"fontSize": "13px"},
                                                    ),
                                                ],
                                                className="card",
                                                style={"marginBottom": "12px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Community Detection",
                                                        style={"marginTop": "0"},
                                                    ),
                                                    html.Div(
                                                        id="communities-display",
                                                        style={"fontSize": "13px"},
                                                    ),
                                                ],
                                                className="card",
                                                style={"marginBottom": "12px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Relationship Strength Distribution",
                                                        style={"marginTop": "0"},
                                                    ),
                                                    dcc.Graph(id="relationship-strength-chart"),
                                                ],
                                                className="card",
                                            ),
                                        ]
                                    ),
                                    dcc.Store(id="analytics-data-store", storage_type="session"),
                                ],
                                id="graph-analytics-tab",
                                style={"display": "none"},
                                className="card",
                            ),
                            html.Div(
                                [
                                    html.H3("� System Metrics Dashboard", style={"marginTop": 0}),
                                    html.Div(
                                        [
                                            html.Button(
                                                "🔄 Refresh Metrics",
                                                id="refresh-metrics-btn",
                                                n_clicks=0,
                                                style={
                                                    "padding": "10px 18px",
                                                    "backgroundColor": "#667eea",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "marginRight": "8px",
                                                    "marginBottom": "12px",
                                                },
                                            ),
                                            dcc.Interval(
                                                id="metrics-refresh-interval",
                                                interval=5000,
                                                n_intervals=0,
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.H4("Live Metrics Summary", style={"marginTop": 0}),
                                            html.Div(
                                                id="metrics-summary-display",
                                                style={"fontSize": "13px"},
                                            ),
                                        ],
                                        className="card",
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.H4(
                                                "Performance Visualisation", style={"marginTop": 0}
                                            ),
                                            dcc.Graph(
                                                id="metrics-figure", style={"height": "600px"}
                                            ),
                                        ],
                                        className="card",
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Model Performance", style={"marginTop": 0}
                                                    ),
                                                    html.Div(
                                                        id="metrics-model-stats",
                                                        style={"fontSize": "12px"},
                                                    ),
                                                ],
                                                className="card",
                                                style={"marginBottom": "16px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "System Health", style={"marginTop": 0}
                                                    ),
                                                    html.Div(
                                                        id="metrics-health-status",
                                                        style={"fontSize": "12px"},
                                                    ),
                                                    html.Hr(style={"margin": "16px 0"}),
                                                    html.H5(
                                                        "Cache Statistics",
                                                        style={"marginTop": 0, "fontSize": "14px"},
                                                    ),
                                                    html.Div(
                                                        id="cache-stats-display",
                                                        style={"fontSize": "12px"},
                                                    ),
                                                    dcc.Interval(
                                                        id="cache-stats-interval",
                                                        interval=5000,
                                                        n_intervals=0,
                                                    ),
                                                ],
                                                className="card",
                                                style={"marginBottom": "16px"},
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.H4("Cost Estimation", style={"marginTop": 0}),
                                            html.Div(
                                                id="metrics-cost-display",
                                                style={"fontSize": "13px", "fontWeight": "bold"},
                                            ),
                                        ],
                                        className="card",
                                        style={"marginBottom": "16px"},
                                    ),
                                ],
                                id="metrics-tab",
                                style={"display": "none"},
                                className="card",
                            ),
                            html.Div(
                                [
                                    html.H3("�📊 Performance Benchmarks", style={"marginTop": 0}),
                                    html.Div(
                                        [
                                            html.Button(
                                                "🔄 Refresh Metrics",
                                                id="refresh-benchmarks-btn",
                                                n_clicks=0,
                                                style={
                                                    "padding": "10px 18px",
                                                    "backgroundColor": "#667eea",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "marginRight": "8px",
                                                    "marginBottom": "12px",
                                                },
                                            ),
                                            html.Button(
                                                "📄 Export Report",
                                                id="export-benchmark-report-btn",
                                                n_clicks=0,
                                                style={
                                                    "padding": "10px 18px",
                                                    "backgroundColor": "#28a745",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "marginBottom": "12px",
                                                },
                                            ),
                                            dcc.Download(id="download-benchmark-report"),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                "Time Range:",
                                                style={"fontWeight": 500, "marginRight": "8px"},
                                            ),
                                            dcc.Dropdown(
                                                id="benchmark-time-range",
                                                options=[
                                                    {"label": "Last Hour", "value": 1},
                                                    {"label": "Last 6 Hours", "value": 6},
                                                    {"label": "Last 24 Hours", "value": 24},
                                                    {"label": "Last 7 Days", "value": 168},
                                                    {"label": "All Time", "value": -1},
                                                ],
                                                value=24,
                                                style={
                                                    "width": "200px",
                                                    "display": "inline-block",
                                                    "marginBottom": "16px",
                                                },
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.H4("Summary Statistics", style={"marginTop": 0}),
                                            html.Div(
                                                id="benchmark-summary", style={"fontSize": "13px"}
                                            ),
                                        ],
                                        className="card",
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Response Time Trend",
                                                        style={"marginTop": 0},
                                                    ),
                                                    dcc.Graph(id="response-time-chart"),
                                                ],
                                                className="card",
                                                style={"marginBottom": "16px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Cache Hit Rate", style={"marginTop": 0}
                                                    ),
                                                    dcc.Graph(id="cache-rate-chart"),
                                                ],
                                                className="card",
                                                style={"marginBottom": "16px"},
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.H4("Slowest Queries", style={"marginTop": 0}),
                                            html.Div(
                                                id="slowest-queries-table",
                                                style={"fontSize": "12px"},
                                            ),
                                        ],
                                        className="card",
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.H4("System Metrics", style={"marginTop": 0}),
                                            html.Div(
                                                id="system-metrics-display",
                                                style={"fontSize": "13px"},
                                            ),
                                        ],
                                        className="card",
                                        style={"marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            html.H4(
                                                "Response Relevancy Ratings", style={"marginTop": 0}
                                            ),
                                            html.Div(
                                                id="relevancy-stats-display",
                                                style={"fontSize": "13px", "marginBottom": "16px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Filter by Rating:",
                                                        style={"fontWeight": 500},
                                                    ),
                                                    dcc.Slider(
                                                        id="relevancy-rating-filter",
                                                        min=0,
                                                        max=5,
                                                        value=0,
                                                        marks={i: str(i) for i in range(6)},
                                                        tooltip={
                                                            "placement": "bottom",
                                                            "always_visible": True,
                                                        },
                                                    ),
                                                    html.Div(
                                                        id="filtered-relevancy-queries",
                                                        style={
                                                            "fontSize": "12px",
                                                            "marginTop": "12px",
                                                        },
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="card",
                                    ),
                                ],
                                id="benchmarks-tab",
                                style={"display": "none"},
                                className="card",
                            ),
                            html.Div(
                                [
                                    html.H3("Academic References"),
                                    html.P("Loading...", style={"color": "#999"}),
                                ],
                                id="academic-references-tab",
                                style={"display": "none"},
                                className="card",
                            ),
                            html.Div(
                                [
                                    html.H3("Citation Graph"),
                                    html.P("Loading...", style={"color": "#999"}),
                                ],
                                id="citation-graph-tab",
                                style={"display": "none"},
                                className="card",
                            ),
                            html.Div(
                                [
                                    html.H3("🎓 PhD Quality Assessment"),
                                    html.Div(
                                        [
                                            html.Label(
                                                "Persona",
                                                style={"fontWeight": "bold", "marginRight": "12px"},
                                            ),
                                            dcc.Dropdown(
                                                id="assessment-persona-dropdown",
                                                options=[
                                                    {"label": "Supervisor", "value": "supervisor"},
                                                    {"label": "Assessor", "value": "assessor"},
                                                    {"label": "Researcher", "value": "researcher"},
                                                ],
                                                value="supervisor",
                                                clearable=False,
                                                style={"width": "240px"},
                                            ),
                                            html.Label(
                                                "Document",
                                                style={
                                                    "fontWeight": "bold",
                                                    "marginLeft": "16px",
                                                    "marginRight": "12px",
                                                },
                                            ),
                                            dcc.Dropdown(
                                                id="assessment-doc-dropdown",
                                                options=[],
                                                value=None,
                                                placeholder="Select a document...",
                                                style={"minWidth": "420px", "flex": "1"},
                                            ),
                                            dcc.Checklist(
                                                id="assessment-llm-toggle",
                                                options=[
                                                    {
                                                        "label": html.Span(
                                                            "LLM: Claims",
                                                            title="Use LLM to extract claims and detect contradictions.",
                                                        ),
                                                        "value": "claims",
                                                    },
                                                    {
                                                        "label": html.Span(
                                                            "LLM: Data/Conclusion",
                                                            title="Use LLM to compare results vs conclusions for mismatches.",
                                                        ),
                                                        "value": "data_mismatch",
                                                    },
                                                    {
                                                        "label": html.Span(
                                                            "LLM: Citation Checks",
                                                            title="Use LLM to check if strong claims align with cited sources.",
                                                        ),
                                                        "value": "citation_misrep",
                                                    },
                                                ],
                                                value=[],
                                                style={"marginLeft": "16px"},
                                            ),
                                            dcc.RadioItems(
                                                id="assessment-issue-limit",
                                                options=[
                                                    {"label": "Show first 10", "value": "first_10"},
                                                    {"label": "Show all", "value": "all"},
                                                ],
                                                value="first_10",
                                                labelStyle={"marginRight": "12px"},
                                                style={"marginLeft": "16px"},
                                            ),
                                            html.Span(
                                                "ⓘ",
                                                title="Enable LLM checks for deeper analysis (slower).",
                                                style={
                                                    "marginLeft": "6px",
                                                    "color": "#667eea",
                                                    "cursor": "help",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "gap": "8px",
                                            "flexWrap": "wrap",
                                            "marginBottom": "16px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "Export Checklist CSV",
                                                id="export-methodology-csv-btn",
                                                n_clicks=0,
                                                style={
                                                    "padding": "6px 12px",
                                                    "backgroundColor": "#1f7a8c",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "fontSize": "12px",
                                                },
                                            ),
                                            html.Button(
                                                "Export Checklist PDF",
                                                id="export-methodology-pdf-btn",
                                                n_clicks=0,
                                                style={
                                                    "padding": "6px 12px",
                                                    "backgroundColor": "#4b5563",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "fontSize": "12px",
                                                },
                                            ),
                                            dcc.Download(id="download-methodology-checklist"),
                                        ],
                                        style={
                                            "display": "flex",
                                            "gap": "8px",
                                            "marginBottom": "12px",
                                        },
                                    ),
                                    html.Div(
                                        id="assessment-report",
                                        children=html.P(
                                            "Select a document to run assessment.",
                                            style={"color": "#999"},
                                        ),
                                    ),
                                ],
                                id="assessment-tab",
                                style={"display": "none"},
                                className="card",
                            ),
                        ],
                        style={"marginTop": "16px"},
                    ),
                ],
                className="container",
            ),
            html.Hr(style={"marginTop": "32px"}),
            html.Div(
                [
                    html.P(
                        f"Built with Plotly Dash • Graph last updated: {metadata.get('built_at', 'N/A')} • Free tier safe ✓",
                        style={
                            "fontSize": "12px",
                            "color": "#999",
                            "textAlign": "center",
                            "marginBottom": "8px",
                        },
                    ),
                ],
                className="container",
            ),
        ]
    )


app.layout = create_layout


# ============================================================================
# Filter Callbacks
# ============================================================================


@callback(
    [
        Output("doc-type-filter", "options"),
        Output("language-filter", "options"),
        Output("repository-filter", "options"),
        Output("topic-cluster-filter", "options"),
        Output("risk-cluster-filter", "options"),
    ],
    [
        Input("graph-tab", "n_clicks"),
        Input("dependencies-tab", "style"),
        Input("show-singleton-clusters-toggle", "value"),
    ],
    prevent_initial_call=False,
)
def populate_filter_options(_graph_clicks, _dep_style, show_singletons):
    """Populate filter dropdown options after graph loads."""
    try:
        if graph_filter is None:
            return [], [], [], [], []

        doc_types = graph_filter.get_available_doc_types()
        languages = graph_filter.get_available_languages()
        repositories = graph_filter.get_available_repositories()

        # Convert to Dash dropdown format
        doc_type_opts = [{"label": dt, "value": dt} for dt in sorted(doc_types)]
        lang_opts = [{"label": lang, "value": lang} for lang in sorted(languages)]
        repo_opts = [{"label": repo, "value": repo} for repo in sorted(repositories)]

        # Cluster options with membership counts, sorted by size (descending) then label (ascending)
        topic_clusters = graph_filter.get_available_topic_clusters()
        risk_clusters = graph_filter.get_available_risk_clusters()

        # Get all topic cluster metadata and sort
        topic_cluster_data = []
        for cid in topic_clusters:
            label = graph_filter.get_topic_cluster_filter_label(cid)
            # Extract size from label format "Label (N nodes)"
            try:
                size = int(label.split("(")[1].split(" ")[0])
            except:
                size = 0
            topic_cluster_data.append({"value": cid, "label": label, "size": size})

        # Sort by size (descending), then by label (ascending)
        topic_cluster_data.sort(key=lambda x: (-x["size"], x["label"]))
        include_singletons = bool(show_singletons and "show_singleton" in show_singletons)
        topic_opts = [
            {"label": item["label"], "value": item["value"]}
            for item in topic_cluster_data
            if include_singletons or item["size"] > 1
        ]

        # Get all risk cluster metadata and sort
        risk_cluster_data = []
        for cid in risk_clusters:
            label = graph_filter.get_risk_cluster_filter_label(cid)
            # Extract size from label format "Label (N nodes)"
            try:
                size = int(label.split("(")[1].split(" ")[0])
            except:
                size = 0
            risk_cluster_data.append({"value": cid, "label": label, "size": size})

        # Sort by size (descending), then by label (ascending)
        risk_cluster_data.sort(key=lambda x: (-x["size"], x["label"]))
        risk_opts = [
            {"label": item["label"], "value": item["value"]}
            for item in risk_cluster_data
            if include_singletons or item["size"] > 1
        ]

        return doc_type_opts, lang_opts, repo_opts, topic_opts, risk_opts
    except Exception as e:
        print(f"Error populating filter options: {e}")
        return [], [], [], [], []


@callback(
    Output("filters-summary-caption", "children"),
    [Input("graph-tab", "n_clicks"), Input("dependencies-tab", "style")],
    prevent_initial_call=False,
)
def update_filters_summary(_graph_clicks, _dep_style):
    """Update summary caption showing counts of available types/langs/repos."""
    try:
        if graph_filter is None:
            return ""
        types_count = len(graph_filter.get_available_doc_types() or [])
        langs_count = len(graph_filter.get_available_languages() or [])
        repos_count = len(graph_filter.get_available_repositories() or [])
        topics_count = len(graph_filter.get_available_topic_clusters() or [])
        risks_count = len(graph_filter.get_available_risk_clusters() or [])
        return f"Types: {types_count} • Languages: {langs_count} • Repositories: {repos_count} • Topic Clusters: {topics_count} • Risk Clusters: {risks_count}"
    except Exception as e:
        print(f"Error updating filters summary: {e}")
        return ""


@callback(
    Output("filter-state-store", "data"),
    Input("apply-filters-btn", "n_clicks"),
    [
        State("doc-type-filter", "value"),
        State("language-filter", "value"),
        State("repository-filter", "value"),
        State("topic-cluster-filter", "value"),
        State("risk-cluster-filter", "value"),
        State("relationship-type-filter", "value"),
        State("min-conflict-slider", "value"),
    ],
    prevent_initial_call=True,
)
def apply_filters(
    n_clicks,
    doc_types,
    languages,
    repositories,
    topic_clusters,
    risk_clusters,
    relationship_types,
    min_conflict,
):
    """Apply selected filters and store filtered node list."""
    if not graph_store.get_node_ids():
        return {"filtered_nodes": [], "active_filters": {}, "relationship_types": []}

    try:
        # Build filter dictionary
        filters = {}
        if doc_types:
            filters["doc_types"] = doc_types if isinstance(doc_types, list) else [doc_types]
        if languages:
            filters["languages"] = languages if isinstance(languages, list) else [languages]
        if repositories:
            filters["repositories"] = (
                repositories if isinstance(repositories, list) else [repositories]
            )
        if topic_clusters:
            filters["topic_clusters"] = (
                topic_clusters if isinstance(topic_clusters, list) else [topic_clusters]
            )
        if risk_clusters:
            filters["risk_clusters"] = (
                risk_clusters if isinstance(risk_clusters, list) else [risk_clusters]
            )
        if min_conflict:
            filters["min_conflict"] = min_conflict

        # Get all nodes
        all_nodes = graph_store.get_node_ids()

        # Apply filters using GraphFilter
        if graph_filter and all_nodes:
            filtered_nodes = graph_filter.filter_nodes(all_nodes, filters)
        else:
            filtered_nodes = all_nodes

        return {
            "filtered_nodes": filtered_nodes,
            "active_filters": filters,
            "relationship_types": (
                relationship_types
                if isinstance(relationship_types, list)
                else ([] if not relationship_types else [relationship_types])
            ),
            "total_filtered": len(filtered_nodes),
            "total_nodes": len(all_nodes),
        }
    except Exception as e:
        print(f"Error applying filters: {e}")
        return {
            "filtered_nodes": graph_store.get_node_ids(),
            "active_filters": {},
            "relationship_types": [],
        }


@callback(
    [
        Output("doc-type-filter", "value"),
        Output("language-filter", "value"),
        Output("repository-filter", "value"),
        Output("topic-cluster-filter", "value"),
        Output("risk-cluster-filter", "value"),
        Output("relationship-type-filter", "value"),
        Output("min-conflict-slider", "value"),
        Output("show-singleton-clusters-toggle", "value"),
    ],
    Input("reset-filters-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(n_clicks):
    """Reset all filters to default values."""
    return [], [], [], [], [], [], 0.0, []


@callback(
    Output("filter-stats", "children"),
    Input("filter-state-store", "data"),
)
def update_filter_stats(filter_state):
    """Update filter statistics display."""
    if not filter_state:
        return "No filters applied"

    try:
        total_filtered = filter_state.get("total_filtered", 0)
        total_nodes = filter_state.get("total_nodes", 0)
        active_filters = filter_state.get("active_filters", {})

        stats_text = f"Showing {total_filtered} of {total_nodes} nodes"

        if active_filters:
            filter_desc = []
            if "doc_types" in active_filters:
                filter_desc.append(f"Types: {', '.join(active_filters['doc_types'])}")
            if "languages" in active_filters:
                filter_desc.append(f"Languages: {', '.join(active_filters['languages'])}")
            if "repositories" in active_filters:
                filter_desc.append(f"Repos: {', '.join(active_filters['repositories'])}")
            if "topic_clusters" in active_filters:
                filter_desc.append(
                    f"Topics: {', '.join(str(x) for x in active_filters['topic_clusters'])}"
                )
            if "risk_clusters" in active_filters:
                filter_desc.append(
                    f"Risks: {', '.join(str(x) for x in active_filters['risk_clusters'])}"
                )
            if "min_conflict" in active_filters:
                filter_desc.append(f"Min Conflict: {active_filters['min_conflict']:.2f}")

            if filter_desc:
                stats_text += " | " + " | ".join(filter_desc)

        return stats_text
    except Exception as e:
        return f"Error: {e}"


@callback(
    Output("graph-page-slider", "max"),
    Input("filter-state-store", "data"),
    Input("nodes-per-page-selector", "value"),
)
def update_slider_max(filter_state, nodes_per_page):
    """Update page slider max value based on filtered nodes and page size."""
    if not filter_state or not filter_state.get("filtered_nodes"):
        total_nodes = len(graph_store.get_node_ids())
    else:
        total_nodes = filter_state.get("total_filtered", len(graph_store.get_node_ids()))

    # Use dynamic nodes per page
    effective_nodes_per_page = nodes_per_page if nodes_per_page else NODES_PER_PAGE

    # Calculate max pages (ceiling division)
    max_pages = max(0, (total_nodes - 1) // effective_nodes_per_page)
    return max_pages


@callback(
    Output("graph-page-slider", "value", allow_duplicate=True),
    Output("graph-page-prev", "disabled"),
    Output("graph-page-next", "disabled"),
    Output("graph-page-input", "value", allow_duplicate=True),
    Input("graph-page-prev", "n_clicks"),
    Input("graph-page-next", "n_clicks"),
    Input("graph-page-input", "value"),
    State("graph-page-slider", "value"),
    State("graph-page-slider", "max"),
    prevent_initial_call=True,
)
def navigate_pages(prev_clicks, next_clicks, input_value, current_page, max_page):
    """Handle previous/next button navigation and direct page input for accessibility."""
    from dash import ctx

    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    new_page = current_page
    if trigger_id == "graph-page-prev" and current_page > 0:
        new_page = current_page - 1
    elif trigger_id == "graph-page-next" and current_page < max_page:
        new_page = current_page + 1
    elif trigger_id == "graph-page-input" and input_value is not None:
        # Convert from 1-based (user input) to 0-based (internal)
        new_page = max(0, min(max_page, int(input_value) - 1))

    # Determine button states
    prev_disabled = new_page <= 0
    next_disabled = new_page >= max_page

    # Return 1-based page number for display in input field
    return new_page, prev_disabled, next_disabled, new_page + 1


@callback(
    Output("graph-page-prev", "disabled", allow_duplicate=True),
    Output("graph-page-next", "disabled", allow_duplicate=True),
    Input("graph-page-slider", "value"),
    State("graph-page-slider", "max"),
    prevent_initial_call=True,
)
def update_button_states(current_page, max_page):
    """Update button disabled states when slider value changes."""
    prev_disabled = current_page <= 0
    next_disabled = current_page >= max_page
    return prev_disabled, next_disabled


@callback(
    [
        Output("graph-container", "children", allow_duplicate=True),
        Output("graph-page-slider", "value"),
    ],
    Input("filter-state-store", "data"),
    prevent_initial_call=True,
)
def apply_filters_to_graph(filter_state):
    """Apply filters and reset page to 0."""
    if not filter_state:
        return dash.no_update, dash.no_update

    # Reset to page 0 when filters change
    return dash.no_update, 0


@callback(
    Output("selected-node-store", "data"),
    Output("document-selector", "value"),
    Input("graph-figure", "clickData"),
    Input("document-selector", "value"),
    prevent_initial_call=True,
)
def update_selected_node(click_data, dropdown_value):
    """Update selected node from graph click or dropdown selection."""
    from dash import ctx

    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "graph-figure" and click_data:
        # Extract node ID from click data
        point = click_data.get("points", [{}])[0]
        # The text field contains the node ID
        selected_node = point.get("text", None)
        # Update both store and dropdown
        return selected_node, selected_node
    elif trigger_id == "document-selector" and dropdown_value:
        # Update only store, dropdown already has the value
        return dropdown_value, dash.no_update

    return None, dash.no_update


@callback(
    Output("graph-tab", "style"),
    Output("details-tab", "style"),
    Output("heatmap-tab", "style"),
    Output("query-tab", "style"),
    Output("dependencies-tab", "style"),
    Output("conversation-browser-tab", "style"),
    Output("citation-graph-tab", "style"),
    Output("assessment-tab", "style"),
    Output("graph-analytics-tab", "style"),
    Output("metrics-tab", "style"),
    Output("benchmarks-tab", "style"),
    Output("academic-references-tab", "style"),
    Output("tab-graph", "className"),
    Output("tab-details", "className"),
    Output("tab-heatmap", "className"),
    Output("tab-query", "className"),
    Output("tab-dependencies", "className"),
    Output("tab-conversations", "className"),
    Output("tab-citation-graph", "className"),
    Output("tab-assessment", "className"),
    Output("tab-graph-analytics", "className"),
    Output("tab-metrics", "className"),
    Output("tab-benchmarks", "className"),
    Output("tab-academic-references", "className"),
    Input("tab-graph", "n_clicks"),
    Input("tab-details", "n_clicks"),
    Input("tab-heatmap", "n_clicks"),
    Input("tab-query", "n_clicks"),
    Input("tab-dependencies", "n_clicks"),
    Input("tab-conversations", "n_clicks"),
    Input("tab-citation-graph", "n_clicks"),
    Input("tab-assessment", "n_clicks"),
    Input("tab-graph-analytics", "n_clicks"),
    Input("tab-metrics", "n_clicks"),
    Input("tab-benchmarks", "n_clicks"),
    Input("tab-academic-references", "n_clicks"),
)
def switch_tabs(
    graph_clicks,
    details_clicks,
    heatmap_clicks,
    query_clicks,
    dependencies_clicks,
    conversations_clicks,
    citation_graph_clicks,
    assessment_clicks,
    analytics_clicks,
    metrics_clicks,
    benchmarks_clicks,
    references_clicks,
):
    """Toggle tab visibility and active styles."""
    ctx = dash.callback_context
    active = "tab-graph"
    if ctx.triggered:
        active = ctx.triggered[0]["prop_id"].split(".")[0]
        print(f"DEBUG: Tab clicked: {active}")  # Debug output

    # Map button IDs to content div IDs
    button_to_content = {
        "tab-graph": "graph-tab",
        "tab-details": "details-tab",
        "tab-heatmap": "heatmap-tab",
        "tab-query": "query-tab",
        "tab-dependencies": "dependencies-tab",
        "tab-conversations": "conversation-browser-tab",
        "tab-citation-graph": "citation-graph-tab",
        "tab-assessment": "assessment-tab",
        "tab-graph-analytics": "graph-analytics-tab",
        "tab-metrics": "metrics-tab",
        "tab-benchmarks": "benchmarks-tab",
        "tab-academic-references": "academic-references-tab",
    }

    active_content = button_to_content.get(active, "graph-tab")
    print(f"DEBUG: Active content div: {active_content}")  # Debug output

    def style_for(content_id: str) -> Dict[str, str]:
        return {"display": "block"} if content_id == active_content else {"display": "none"}

    def class_for(button_id: str) -> str:
        base = "tab-button"
        return f"{base} active" if button_id == active else base

    return (
        style_for("graph-tab"),
        style_for("details-tab"),
        style_for("heatmap-tab"),
        style_for("query-tab"),
        style_for("dependencies-tab"),
        style_for("conversation-browser-tab"),
        style_for("citation-graph-tab"),
        style_for("assessment-tab"),
        style_for("graph-analytics-tab"),
        style_for("metrics-tab"),
        style_for("benchmarks-tab"),
        style_for("academic-references-tab"),
        class_for("tab-graph"),
        class_for("tab-details"),
        class_for("tab-heatmap"),
        class_for("tab-query"),
        class_for("tab-dependencies"),
        class_for("tab-conversations"),
        class_for("tab-citation-graph"),
        class_for("tab-assessment"),
        class_for("tab-graph-analytics"),
        class_for("tab-metrics"),
        class_for("tab-benchmarks"),
        class_for("tab-academic-references"),
    )


@callback(
    Output("academic-references-tab", "children"),
    Input("academic-references-tab", "style"),
)
def populate_academic_references_tab(style):
    """Populate the academic references tab with visualisation when it becomes visible."""
    print(f"DEBUG: populate_academic_references_tab() called with style={style}")

    # Only populate when tab is visible (display: block)
    if not style or style.get("display") != "block":
        print(
            f"DEBUG: Tab not visible (display={style.get('display') if style else None}), raising PreventUpdate"
        )
        raise PreventUpdate

    print("DEBUG: Tab is visible, populating content...")

    try:
        terminology_db_path = Path(INGEST_CONFIG.rag_data_path) / "academic_terminology.db"
        print(f"DEBUG: Checking for terminology DB at: {terminology_db_path}")

        if not terminology_db_path.exists():
            print("DEBUG: Terminology DB not found")
            return html.Div(
                [
                    html.H3("Academic References"),
                    html.P(
                        "No academic terminology database found. Run academic ingest first.",
                        style={"color": "#999"},
                    ),
                ]
            )

        print("DEBUG: Initialising AcademicReferences...")
        # Initialise module and get layout (or reuse if already initialised at startup)
        module = _get_global_module()
        if not module:
            print("DEBUG: Module not initialised at startup, creating now...")
            module = AcademicReferences(str(terminology_db_path))
            _set_global_module(module)
        else:
            print("DEBUG: Using module initialised at startup")

        layout = module.create_layout()

        word_cloud_section = None
        try:
            word_cloud_data = get_word_cloud_data(limit=80, min_frequency=2, format="weighted")
            if isinstance(word_cloud_data, list) and word_cloud_data:
                fig = _build_word_cloud_figure(word_cloud_data)
                stats = get_word_cloud_stats()
                word_cloud_section = html.Div(
                    [
                        html.H4("Word Cloud (Academic Ingest)", style={"marginTop": 0}),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            str(stats.get("total_unique_words", 0)),
                                            className="metric-value",
                                        ),
                                        html.Div("Unique words", className="metric-label"),
                                    ],
                                    className="metric",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            str(stats.get("total_frequency", 0)),
                                            className="metric-value",
                                        ),
                                        html.Div("Total frequency", className="metric-label"),
                                    ],
                                    className="metric",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            str(stats.get("max_frequency", 0)),
                                            className="metric-value",
                                        ),
                                        html.Div("Max frequency", className="metric-label"),
                                    ],
                                    className="metric",
                                ),
                            ],
                            style={"marginBottom": "12px"},
                        ),
                        dcc.Graph(
                            figure=fig,
                            config={"displayModeBar": False},
                            style={"height": "360px"},
                        ),
                    ],
                    className="card",
                )
            else:
                word_cloud_section = html.Div(
                    [
                        html.H4("Word Cloud (Academic Ingest)", style={"marginTop": 0}),
                        html.P(
                            "No word cloud data available. Run academic ingest to generate word frequencies.",
                            style={"color": "#999"},
                        ),
                    ],
                    className="card",
                )
        except Exception as e:
            logger.warning("Failed to load word cloud data: %s", e)
            word_cloud_section = html.Div(
                [
                    html.H4("Word Cloud (Academic Ingest)", style={"marginTop": 0}),
                    html.P("Error loading word cloud data.", style={"color": "#d32f2f"}),
                ],
                className="card",
            )

        print("DEBUG: Layout created successfully")
        return html.Div(
            [
                word_cloud_section,
                layout,
            ],
            style={"display": "flex", "flexDirection": "column", "gap": "16px"},
        )
    except Exception as e:
        print(f"DEBUG: Error loading academic references: {e}")
        import traceback

        traceback.print_exc()
        return html.Div(
            [
                html.H3("Academic References"),
                html.P(f"Error loading academic references: {str(e)}", style={"color": "#d32f2f"}),
            ]
        )


@callback(
    Output("citation-graph-tab", "children"),
    Input("citation-graph-tab", "style"),
)
def populate_citation_graph_tab(style):
    """Populate the citation graph tab when it becomes visible."""
    if not style or style.get("display") != "block":
        raise PreventUpdate

    try:
        viz = get_citation_viz()
        return viz.create_dash_layout()
    except Exception as e:
        return html.Div(
            [
                html.H3("Citation Graph"),
                html.P(f"Error loading citation graph: {str(e)}", style={"color": "#d32f2f"}),
            ]
        )


@callback(
    Output("assessment-doc-dropdown", "options"),
    Output("assessment-doc-dropdown", "value"),
    Input("assessment-tab", "style"),
    State("assessment-doc-dropdown", "value"),
)
def populate_assessment_doc_options(style, current_selection):
    """Populate assessment document dropdown when visible."""
    if not style or style.get("display") != "block":
        raise PreventUpdate

    collection = _get_query_collection()
    all_docs = collection.get(include=["metadatas"])
    doc_ids = sorted(
        set(meta.get("doc_id", "") for meta in all_docs.get("metadatas", []) if meta.get("doc_id"))
    )

    options = [{"label": doc_id, "value": doc_id} for doc_id in doc_ids]

    if current_selection and current_selection in doc_ids:
        selected = current_selection
    else:
        selected = doc_ids[0] if doc_ids else None

    return options, selected


@callback(
    Output("academic-doc-dropdown", "options"),
    Output("academic-doc-dropdown", "value"),
    Input("academic-references-tab", "style"),
    Input("academic-domain-dropdown", "value"),
    State("assessment-doc-dropdown", "value"),
)
def populate_academic_doc_options(style, domain, assessment_doc_id):
    """Populate academic references document dropdown when visible."""
    if not style or style.get("display") != "block":
        raise PreventUpdate

    module = _get_global_module()
    doc_ids = []
    if module and domain:
        doc_ids = module.get_doc_ids_for_domain(domain)

    options = [{"label": "All documents", "value": "__all__"}]
    options.extend({"label": doc_id, "value": doc_id} for doc_id in doc_ids)

    selected = "__all__"
    if assessment_doc_id and assessment_doc_id in doc_ids:
        selected = assessment_doc_id
    elif doc_ids:
        selected = doc_ids[0]

    return options, selected


@callback(
    Output("assessment-report", "children"),
    Input("assessment-tab", "style"),
    Input("assessment-doc-dropdown", "value"),
    Input("assessment-persona-dropdown", "value"),
    Input("assessment-llm-toggle", "value"),
    Input("assessment-issue-limit", "value"),
)
def update_assessment_report(style, selected_doc_id, persona, llm_toggle, issue_limit):
    """Render assessment report for selected document and persona."""
    if not style or style.get("display") != "block":
        raise PreventUpdate
    if not selected_doc_id:
        return html.P("Select a document to run assessment.", style={"color": "#999"})

    try:
        from scripts.ingest.academic.phd_assessor import PhDQualityAssessor
        from scripts.rag.generate import _get_llm

        collection = _get_query_collection()
        llm_flags = {
            "claims": bool(llm_toggle and "claims" in llm_toggle),
            "data_mismatch": bool(llm_toggle and "data_mismatch" in llm_toggle),
            "citation_misrep": bool(llm_toggle and "citation_misrep" in llm_toggle),
        }
        use_any_llm = any(llm_flags.values())
        llm_client = _get_llm(temperature=0.0) if use_any_llm else None

        # Path to citation graph database
        from pathlib import Path

        citation_db_path = Path(RAG_CONFIG.rag_data_path) / "academic_citation_graph.db"

        assessor = PhDQualityAssessor(
            collection,
            llm_client=llm_client,
            llm_flags=llm_flags,
            citation_db_path=str(citation_db_path) if citation_db_path.exists() else None,
        )
        report = assessor.assess_thesis(selected_doc_id, persona=persona or "supervisor")

        def _build_concept_progression_figure(structure_analysis):
            import re

            concepts = structure_analysis.key_concepts or []
            if not concepts:
                return None

            chapter_labels = set()
            for concept in concepts:
                intro = concept.get("intro_section")
                concluded = concept.get("concluded_section")
                developed = concept.get("developed_sections") or []
                if intro:
                    chapter_labels.add(intro)
                if concluded:
                    chapter_labels.add(concluded)
                for section in developed:
                    chapter_labels.add(section)

            if not chapter_labels:
                return None

            def _chapter_sort_key(label: str) -> tuple:
                match = re.search(r"(\d+)", str(label))
                if match:
                    return (0, int(match.group(1)), str(label))
                return (1, str(label).lower())

            chapter_order = getattr(structure_analysis, "chapter_order", None)
            if chapter_order:
                ordered_chapters = [label for label in chapter_order if label in chapter_labels]
            else:
                ordered_chapters = []

            if not ordered_chapters:
                ordered_chapters = sorted(chapter_labels, key=_chapter_sort_key)

            def _shorten_chapter_label(label: str) -> str:
                match = re.search(r"chapter\s+(\d+)", str(label), re.IGNORECASE)
                if match:
                    return f"Ch {match.group(1)}"
                return str(label)

            short_chapter_labels = [_shorten_chapter_label(label) for label in ordered_chapters]
            concepts_to_plot = concepts[:20]
            concept_labels = [c.get("concept", "") for c in concepts_to_plot]

            matrix = []
            for concept in concepts_to_plot:
                intro = concept.get("intro_section")
                concluded = concept.get("concluded_section")
                developed = concept.get("developed_sections") or []
                present = {section for section in developed}
                if intro:
                    present.add(intro)
                if concluded:
                    present.add(concluded)
                row = [1 if chapter in present else 0 for chapter in ordered_chapters]
                matrix.append(row)

            height = min(520, 240 + (len(concept_labels) * 12))
            full_chapter_matrix = [ordered_chapters for _ in concept_labels]
            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    x=short_chapter_labels,
                    y=concept_labels,
                    customdata=full_chapter_matrix,
                    colorscale=[[0, "#f3f4f6"], [1, "#2563eb"]],
                    showscale=False,
                    hovertemplate="Concept: %{y}<br>Chapter: %{customdata}<extra></extra>",
                )
            )
            fig.update_layout(
                title="Concept Coverage by Chapter",
                height=height,
                margin=dict(l=40, r=20, t=40, b=120),
                xaxis_title="Chapter",
                yaxis_title="Concept",
                yaxis=dict(autorange="reversed"),
            )
            fig.update_xaxes(tickangle=-35, automargin=True, tickfont=dict(size=10))
            return fig

        concept_progression_fig = _build_concept_progression_figure(report.structure_analysis)

        def _shorten_transition_labels(labels: List[str]) -> Tuple[List[str], List[str]]:
            import re

            def _shorten(label: str) -> str:
                match = re.search(r"chapter\s+(\d+)", label, re.IGNORECASE)
                if match:
                    return f"Ch {match.group(1)}"
                return label

            short_labels = []
            full_labels = []
            for label in labels:
                parts = [part.strip() for part in str(label).split("→")]
                if len(parts) == 2:
                    short_labels.append(f"{_shorten(parts[0])} → {_shorten(parts[1])}")
                else:
                    short_labels.append(_shorten(label))
                full_labels.append(str(label))
            return short_labels, full_labels

        short_transitions, full_transitions = _shorten_transition_labels(
            report.structure_analysis.chapter_transition_labels
        )

        return html.Div(
            [
                html.Div(
                    [
                        html.H3(
                            f"🎓 PhD Quality Assessment: {selected_doc_id}", style={"margin": 0}
                        ),
                        html.Span(
                            (
                                "LLM: "
                                + ", ".join(
                                    [k.replace("_", " ") for k, v in llm_flags.items() if v]
                                )
                                if use_any_llm
                                else "LLM: OFF"
                            ),
                            style={
                                "marginLeft": "12px",
                                "padding": "4px 8px",
                                "borderRadius": "12px",
                                "fontSize": "12px",
                                "color": "white" if use_any_llm else "#666",
                                "backgroundColor": "#28a745" if use_any_llm else "#e0e0e0",
                            },
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
                ),
                # Overall Score Card
                html.Div(
                    [
                        html.H4("Overall Quality Score"),
                        html.Div(
                            [
                                html.Div(
                                    f"{report.overall_score*100:.0f}%",
                                    style={
                                        "fontSize": "48px",
                                        "fontWeight": "bold",
                                        "color": (
                                            "#28a745"
                                            if report.overall_score >= 0.7
                                            else (
                                                "#FF6B00"
                                                if report.overall_score >= 0.5
                                                else "#d32f2f"
                                            )
                                        ),
                                    },
                                ),
                                html.P(
                                    report.summary,
                                    style={
                                        "fontSize": "14px",
                                        "color": "#666",
                                        "maxWidth": "800px",
                                        "margin": "12px auto",
                                    },
                                ),
                            ],
                            style={"textAlign": "center", "padding": "20px"},
                        ),
                    ],
                    className="card",
                    style={"marginBottom": "16px"},
                ),
                # Critical Red Flags
                html.Div(
                    [
                        html.H4("🚨 Critical Issues"),
                        html.Div(
                            [
                                (
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Strong(
                                                        flag.title, style={"color": "#d32f2f"}
                                                    ),
                                                    html.P(
                                                        flag.description,
                                                        style={
                                                            "margin": "4px 0",
                                                            "fontSize": "14px",
                                                        },
                                                    ),
                                                    (
                                                        html.P(
                                                            f"💡 {flag.suggestion}",
                                                            style={
                                                                "color": "#667eea",
                                                                "fontSize": "12px",
                                                                "fontStyle": "italic",
                                                            },
                                                        )
                                                        if flag.suggestion
                                                        else html.Div()
                                                    ),
                                                ],
                                                style={
                                                    "padding": "12px",
                                                    "border": "1px solid #fee",
                                                    "borderRadius": "4px",
                                                    "marginBottom": "8px",
                                                    "backgroundColor": "#fff5f5",
                                                },
                                            )
                                            for flag in report.critical_red_flags[:5]
                                        ]
                                    )
                                    if report.critical_red_flags
                                    else html.P(
                                        "✅ No critical issues found",
                                        style={"color": "#28a745", "fontSize": "14px"},
                                    )
                                )
                            ]
                        ),
                    ],
                    className="card",
                    style={"marginBottom": "16px"},
                ),
                # Structure Analysis
                html.Div(
                    [
                        html.H4("📚 Structural Coherence"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span("Chapters: ", style={"fontWeight": "bold"}),
                                        html.Span(str(report.structure_analysis.chapter_count)),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            "Average Coherence: ", style={"fontWeight": "bold"}
                                        ),
                                        html.Span(f"{report.structure_analysis.avg_coherence:.2f}"),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            "Missing Sections: ", style={"fontWeight": "bold"}
                                        ),
                                        html.Span(
                                            ", ".join(report.structure_analysis.missing_sections)
                                            if report.structure_analysis.missing_sections
                                            else "None ✓"
                                        ),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            "Abrupt Transitions: ", style={"fontWeight": "bold"}
                                        ),
                                        html.Span(
                                            str(len(report.structure_analysis.abrupt_transitions))
                                        ),
                                    ]
                                ),
                            ],
                            style={"fontSize": "14px", "marginBottom": "16px"},
                        ),
                        # Chapter flow chart
                        (
                            dcc.Graph(
                                figure=go.Figure(
                                    data=[
                                        go.Scatter(
                                            x=short_transitions,
                                            y=report.structure_analysis.chapter_flow_scores,
                                            mode="lines+markers",
                                            name="Flow Score",
                                            line=dict(color="#667eea", width=2),
                                            marker=dict(size=8),
                                            customdata=full_transitions,
                                            hovertemplate="Transition: %{customdata}<br>Score: %{y:.2f}<extra></extra>",
                                        ),
                                        go.Scatter(
                                            x=short_transitions,
                                            y=[0.3]
                                            * len(report.structure_analysis.chapter_flow_scores),
                                            mode="lines",
                                            name="Threshold",
                                            line=dict(color="#FF6B00", width=1, dash="dash"),
                                            hoverinfo="skip",
                                        ),
                                    ],
                                    layout=go.Layout(
                                        title="Chapter-to-Chapter Coherence",
                                        xaxis_title="Chapter Transition",
                                        xaxis=dict(
                                            tickangle=-35, automargin=True, tickfont=dict(size=10)
                                        ),
                                        yaxis_title="Similarity Score",
                                        height=360,
                                        margin=dict(l=40, r=20, t=40, b=140),
                                    ),
                                ),
                                config={"displayModeBar": False},
                            )
                            if report.structure_analysis.chapter_flow_scores
                            else html.P(
                                "Not enough chapters for flow analysis",
                                style={"color": "#999", "fontSize": "12px"},
                            )
                        ),
                        # Research Question Alignment
                        html.Div(
                            [
                                html.H5(
                                    "🎯 Research Question Alignment",
                                    style={"marginTop": "16px", "marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span(
                                                    "RQ Alignment Score: ",
                                                    style={"fontWeight": "bold"},
                                                ),
                                                html.Span(
                                                    f"{report.structure_analysis.rq_alignment_score*100:.0f}%",
                                                    style={
                                                        "color": (
                                                            "#28a745"
                                                            if report.structure_analysis.rq_alignment_score
                                                            >= 0.7
                                                            else "#FF6B00"
                                                        )
                                                    },
                                                ),
                                            ],
                                            style={"marginBottom": "8px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Span(
                                                    "Research Questions Found: ",
                                                    style={"fontWeight": "bold"},
                                                ),
                                                html.Span(
                                                    str(
                                                        len(
                                                            report.structure_analysis.research_questions
                                                        )
                                                    )
                                                ),
                                            ],
                                            style={"marginBottom": "8px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Span(
                                                    "Unaddressed: ", style={"fontWeight": "bold"}
                                                ),
                                                html.Span(
                                                    str(
                                                        len(
                                                            report.structure_analysis.unaddressed_rqs
                                                        )
                                                    )
                                                ),
                                            ]
                                        ),
                                    ],
                                    style={"fontSize": "14px", "marginBottom": "8px"},
                                ),
                                (
                                    html.Details(
                                        [
                                            html.Summary(
                                                "View research questions",
                                                style={
                                                    "fontWeight": "bold",
                                                    "cursor": "pointer",
                                                    "fontSize": "12px",
                                                },
                                            ),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        rq[:200] + "..." if len(rq) > 200 else rq,
                                                        style={
                                                            "fontSize": "12px",
                                                            "marginBottom": "4px",
                                                            "color": (
                                                                "#d32f2f"
                                                                if rq
                                                                in report.structure_analysis.unaddressed_rqs
                                                                else "#333"
                                                            ),
                                                        },
                                                    )
                                                    for rq in report.structure_analysis.research_questions[
                                                        :10
                                                    ]
                                                ],
                                                style={"marginTop": "6px"},
                                            ),
                                        ],
                                        style={"marginBottom": "12px"},
                                    )
                                    if report.structure_analysis.research_questions
                                    else html.P(
                                        "No explicit research questions detected.",
                                        style={"color": "#999", "fontSize": "12px"},
                                    )
                                ),
                                (
                                    html.Div(
                                        [
                                            html.P(
                                                "Unaddressed RQs:",
                                                style={
                                                    "fontWeight": "bold",
                                                    "fontSize": "12px",
                                                    "marginBottom": "4px",
                                                },
                                            ),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        rq,
                                                        style={
                                                            "fontSize": "12px",
                                                            "color": "#d32f2f",
                                                        },
                                                    )
                                                    for rq in report.structure_analysis.unaddressed_rqs[
                                                        :5
                                                    ]
                                                ]
                                            ),
                                        ],
                                        style={"marginTop": "8px"},
                                    )
                                    if report.structure_analysis.unaddressed_rqs
                                    else html.Div()
                                ),
                            ],
                            style={
                                "padding": "12px",
                                "backgroundColor": "#f9f9f9",
                                "borderRadius": "4px",
                                "marginTop": "12px",
                            },
                        ),
                        # Concept Progression
                        html.Div(
                            [
                                html.H5(
                                    "🔑 Key Concept Progression",
                                    style={"marginTop": "16px", "marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span(
                                                    "Tracked Concepts: ",
                                                    style={"fontWeight": "bold"},
                                                ),
                                                html.Span(
                                                    str(len(report.structure_analysis.key_concepts))
                                                ),
                                            ],
                                            style={"marginBottom": "8px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Span(
                                                    "Orphaned (single section): ",
                                                    style={"fontWeight": "bold"},
                                                ),
                                                html.Span(
                                                    str(
                                                        len(
                                                            report.structure_analysis.orphaned_concepts
                                                        )
                                                    )
                                                ),
                                            ]
                                        ),
                                    ],
                                    style={"fontSize": "14px", "marginBottom": "8px"},
                                ),
                                (
                                    dcc.Graph(
                                        figure=concept_progression_fig,
                                        config={"displayModeBar": False},
                                    )
                                    if concept_progression_fig
                                    else html.P(
                                        "Not enough concepts tracked.",
                                        style={"color": "#999", "fontSize": "12px"},
                                    )
                                ),
                                (
                                    html.Details(
                                        [
                                            html.Summary(
                                                "View concept progression",
                                                style={
                                                    "fontWeight": "bold",
                                                    "cursor": "pointer",
                                                    "fontSize": "12px",
                                                },
                                            ),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        [
                                                            html.Strong(f"{concept['concept']}: "),
                                                            html.Span(
                                                                f"Intro: {concept.get('intro_section', 'N/A')} → "
                                                                f"Developed: {len(concept.get('developed_sections', []))} sections → "
                                                                f"Concluded: {concept.get('concluded_section', 'N/A')}",
                                                                style={"color": "#666"},
                                                            ),
                                                        ],
                                                        style={
                                                            "fontSize": "12px",
                                                            "marginBottom": "6px",
                                                        },
                                                    )
                                                    for concept in report.structure_analysis.key_concepts[
                                                        :10
                                                    ]
                                                ],
                                                style={"marginTop": "6px"},
                                            ),
                                        ],
                                        style={"marginBottom": "12px"},
                                    )
                                    if report.structure_analysis.key_concepts
                                    else html.Div()
                                ),
                            ],
                            style={
                                "padding": "12px",
                                "backgroundColor": "#f9f9f9",
                                "borderRadius": "4px",
                                "marginTop": "12px",
                            },
                        ),
                    ],
                    className="card",
                    style={"marginBottom": "16px"},
                ),
                # Citation Analysis
                html.Div(
                    [
                        html.H4("📖 Citation Patterns"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            "Total Citations: ", style={"fontWeight": "bold"}
                                        ),
                                        html.Span(str(report.citation_analysis.total_citations)),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            "Unique Citations: ", style={"fontWeight": "bold"}
                                        ),
                                        html.Span(str(report.citation_analysis.unique_citations)),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Span("Recency Score: ", style={"fontWeight": "bold"}),
                                        html.Span(
                                            f"{report.citation_analysis.citation_recency_score*100:.0f}%"
                                        ),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.P(
                                    "Counts are best-effort and may not capture 100% of citations used.",
                                    style={
                                        "fontSize": "12px",
                                        "color": "#666",
                                        "fontStyle": "italic",
                                    },
                                ),
                            ],
                            style={"fontSize": "14px"},
                        ),
                    ],
                    className="card",
                    style={"marginBottom": "16px"},
                ),
                # Claim Analysis
                html.Div(
                    [
                        html.H4("🧩 Claims & Contradictions"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span("Total Claims: ", style={"fontWeight": "bold"}),
                                        html.Span(str(report.claim_analysis.total_claims)),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Span("Contradictions: ", style={"fontWeight": "bold"}),
                                        html.Span(str(len(report.claim_analysis.contradictions))),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            "Unsupported Claims: ", style={"fontWeight": "bold"}
                                        ),
                                        html.Span(str(len(report.claim_analysis.orphaned_claims))),
                                    ]
                                ),
                            ],
                            style={"fontSize": "14px", "marginBottom": "12px"},
                        ),
                        html.Div(
                            [
                                (
                                    html.Div(
                                        [
                                            html.P(
                                                "Potential contradiction examples:",
                                                style={"fontWeight": "bold", "marginBottom": "6px"},
                                            ),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        f"A: {item['claim_a'][:140]}... | B: {item['claim_b'][:140]}..."
                                                        f" [{item.get('source', 'heuristic').upper()}]",
                                                        style={
                                                            "fontSize": "12px",
                                                            "marginBottom": "6px",
                                                        },
                                                    )
                                                    for item in report.claim_analysis.contradictions[
                                                        :3
                                                    ]
                                                ]
                                            ),
                                        ]
                                    )
                                    if report.claim_analysis.contradictions
                                    else html.P(
                                        "No contradictions detected.",
                                        style={"color": "#28a745", "fontSize": "12px"},
                                    )
                                )
                            ]
                        ),
                        html.Div(
                            [
                                (
                                    html.Div(
                                        [
                                            html.P(
                                                (
                                                    "Extracted claims (showing all):"
                                                    if issue_limit == "all"
                                                    else "Extracted claims (showing first 10):"
                                                ),
                                                style={
                                                    "fontWeight": "bold",
                                                    "marginBottom": "6px",
                                                    "marginTop": "12px",
                                                },
                                            ),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        (
                                                            claim[:200] + "..."
                                                            if len(claim) > 200
                                                            else claim
                                                        ),
                                                        style={
                                                            "fontSize": "12px",
                                                            "marginBottom": "6px",
                                                            "color": "#333",
                                                        },
                                                    )
                                                    for claim in (
                                                        report.claim_analysis.claims
                                                        if issue_limit == "all"
                                                        else report.claim_analysis.claims[:10]
                                                    )
                                                ]
                                            ),
                                        ]
                                    )
                                    if report.claim_analysis.claims
                                    else html.P(
                                        "No claims extracted.",
                                        style={"color": "#999", "fontSize": "12px"},
                                    )
                                )
                            ]
                        ),
                        html.Div(
                            [
                                (
                                    html.Div(
                                        [
                                            html.P(
                                                (
                                                    "Unsupported claim examples (showing all):"
                                                    if issue_limit == "all"
                                                    else "Unsupported claim examples (showing first 10):"
                                                ),
                                                style={
                                                    "fontWeight": "bold",
                                                    "marginBottom": "6px",
                                                    "marginTop": "16px",
                                                },
                                            ),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        (
                                                            claim[:200] + "..."
                                                            if len(claim) > 200
                                                            else claim
                                                        ),
                                                        style={
                                                            "fontSize": "12px",
                                                            "marginBottom": "6px",
                                                            "color": "#666",
                                                        },
                                                    )
                                                    for claim in (
                                                        report.claim_analysis.orphaned_claims
                                                        if issue_limit == "all"
                                                        else report.claim_analysis.orphaned_claims[
                                                            :10
                                                        ]
                                                    )
                                                ]
                                            ),
                                        ]
                                    )
                                    if report.claim_analysis.orphaned_claims
                                    else html.P(
                                        "All strong claims appear to have citations.",
                                        style={
                                            "color": "#28a745",
                                            "fontSize": "12px",
                                            "marginTop": "12px",
                                        },
                                    )
                                )
                            ]
                        ),
                    ],
                    className="card",
                    style={"marginBottom": "16px"},
                ),
                # Data/Conclusion Checks
                html.Div(
                    [
                        html.H4("📊 Data vs Conclusions"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            "Potential Mismatches: ", style={"fontWeight": "bold"}
                                        ),
                                        html.Span(str(len(report.data_conclusion_mismatch.issues))),
                                    ]
                                ),
                            ],
                            style={"fontSize": "14px", "marginBottom": "12px"},
                        ),
                        html.Div(
                            [
                                (
                                    html.Details(
                                        [
                                            html.Summary(
                                                (
                                                    "Examples (showing all)"
                                                    if issue_limit == "all"
                                                    else "Examples (showing first 10)"
                                                ),
                                                style={"fontWeight": "bold", "cursor": "pointer"},
                                            ),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        [
                                                            html.Span(
                                                                (
                                                                    issue.get("claim", "")[:200]
                                                                    + "..."
                                                                    if len(issue.get("claim", ""))
                                                                    > 200
                                                                    else issue.get("claim", "")
                                                                ),
                                                            ),
                                                            html.Span(
                                                                f" ({issue.get('reason', '')})",
                                                                style={
                                                                    "color": "#666",
                                                                    "marginLeft": "6px",
                                                                },
                                                            ),
                                                            html.Span(
                                                                f" [{issue.get('source', 'heuristic').upper()}]",
                                                                style={
                                                                    "color": "#555",
                                                                    "marginLeft": "6px",
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "fontSize": "12px",
                                                            "marginBottom": "6px",
                                                        },
                                                    )
                                                    for issue in (
                                                        report.data_conclusion_mismatch.issues
                                                        if issue_limit == "all"
                                                        else report.data_conclusion_mismatch.issues[
                                                            :10
                                                        ]
                                                    )
                                                ],
                                                style={"marginTop": "8px"},
                                            ),
                                        ]
                                    )
                                    if report.data_conclusion_mismatch.issues
                                    else html.P(
                                        "No data/conclusion mismatches detected.",
                                        style={"color": "#28a745", "fontSize": "12px"},
                                    )
                                )
                            ]
                        ),
                    ],
                    className="card",
                    style={"marginBottom": "16px"},
                ),
                # Citation Checks
                html.Div(
                    [
                        html.H4("📚 Citation Checks"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            "Potential Issues: ", style={"fontWeight": "bold"}
                                        ),
                                        html.Span(
                                            str(len(report.citation_misrepresentation.issues))
                                        ),
                                    ]
                                ),
                            ],
                            style={"fontSize": "14px", "marginBottom": "12px"},
                        ),
                        html.Div(
                            [
                                (
                                    html.Details(
                                        [
                                            html.Summary(
                                                (
                                                    "Examples (showing all)"
                                                    if issue_limit == "all"
                                                    else "Examples (showing first 10)"
                                                ),
                                                style={"fontWeight": "bold", "cursor": "pointer"},
                                            ),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        [
                                                            html.Span(
                                                                (
                                                                    issue.get("claim", "")[:200]
                                                                    + "..."
                                                                    if len(issue.get("claim", ""))
                                                                    > 200
                                                                    else issue.get("claim", "")
                                                                ),
                                                            ),
                                                            html.Span(
                                                                f" ({issue.get('reason', '')})",
                                                                style={
                                                                    "color": "#666",
                                                                    "marginLeft": "6px",
                                                                },
                                                            ),
                                                            html.Span(
                                                                f" [{issue.get('source', 'heuristic').upper()}]",
                                                                style={
                                                                    "color": "#555",
                                                                    "marginLeft": "6px",
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "fontSize": "12px",
                                                            "marginBottom": "6px",
                                                        },
                                                    )
                                                    for issue in (
                                                        report.citation_misrepresentation.issues
                                                        if issue_limit == "all"
                                                        else report.citation_misrepresentation.issues[
                                                            :10
                                                        ]
                                                    )
                                                ],
                                                style={"marginTop": "8px"},
                                            ),
                                        ]
                                    )
                                    if report.citation_misrepresentation.issues
                                    else html.P(
                                        "No citation misrepresentation issues detected.",
                                        style={"color": "#28a745", "fontSize": "12px"},
                                    )
                                )
                            ]
                        ),
                    ],
                    className="card",
                    style={"marginBottom": "16px"},
                ),
                # Methodology Checklist
                html.Div(
                    [
                        html.H4("🧪 Methodology Checklist"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            "Checklist Score: ", style={"fontWeight": "bold"}
                                        ),
                                        html.Span(f"{report.methodology_checklist.score*100:.0f}%"),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Span("Missing Items: ", style={"fontWeight": "bold"}),
                                        html.Span(
                                            ", ".join(report.methodology_checklist.missing_items)
                                            if report.methodology_checklist.missing_items
                                            else "None ✓"
                                        ),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                (
                                    html.P(
                                        report.methodology_checklist.confidence_note,
                                        style={
                                            "fontSize": "12px",
                                            "color": "#666",
                                            "fontStyle": "italic",
                                        },
                                    )
                                    if report.methodology_checklist.confidence_note
                                    else html.Div()
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            "Evidence by checklist item:",
                                            style={"fontWeight": "bold", "marginTop": "8px"},
                                        ),
                                        html.Ul(
                                            [
                                                html.Li(
                                                    [
                                                        html.Span(
                                                            f"{item.replace('_', ' ').title()}: ",
                                                            style={"fontWeight": "bold"},
                                                        ),
                                                        html.Span(
                                                            (
                                                                "Present"
                                                                if data.get("count", 0) > 0
                                                                else "Missing"
                                                            ),
                                                            style={
                                                                "color": (
                                                                    "#28a745"
                                                                    if data.get("count", 0) > 0
                                                                    else "#d32f2f"
                                                                )
                                                            },
                                                        ),
                                                        html.Span(
                                                            f" (hits: {data.get('count', 0)}, density: {data.get('strength_per_1k', 0):.2f}/1k)",
                                                            style={
                                                                "color": "#666",
                                                                "marginLeft": "6px",
                                                            },
                                                        ),
                                                        (
                                                            html.Div(
                                                                data.get("summary", ""),
                                                                style={
                                                                    "fontSize": "12px",
                                                                    "color": "#555",
                                                                    "marginTop": "4px",
                                                                    "fontStyle": "italic",
                                                                },
                                                            )
                                                            if data.get("summary")
                                                            else html.Div()
                                                        ),
                                                        (
                                                            html.Div(
                                                                [
                                                                    html.Ul(
                                                                        [
                                                                            html.Li(
                                                                                [
                                                                                    html.Span(
                                                                                        f"{snippet.get('location', 'Unknown')}: ",
                                                                                        style={
                                                                                            "fontWeight": "bold"
                                                                                        },
                                                                                    ),
                                                                                    html.Span(
                                                                                        snippet.get(
                                                                                            "snippet",
                                                                                            "",
                                                                                        )
                                                                                    ),
                                                                                    (
                                                                                        html.Span(
                                                                                            " ("
                                                                                            + ", ".join(
                                                                                                snippet.get(
                                                                                                    "tags",
                                                                                                    [],
                                                                                                )
                                                                                            )
                                                                                            + ")",
                                                                                            style={
                                                                                                "color": "#888",
                                                                                                "marginLeft": "6px",
                                                                                            },
                                                                                        )
                                                                                        if snippet.get(
                                                                                            "tags"
                                                                                        )
                                                                                        else html.Div()
                                                                                    ),
                                                                                ],
                                                                                style={
                                                                                    "marginBottom": "6px"
                                                                                },
                                                                            )
                                                                            for snippet in data.get(
                                                                                "snippets", []
                                                                            )
                                                                        ],
                                                                        style={
                                                                            "paddingLeft": "18px",
                                                                            "marginTop": "6px",
                                                                        },
                                                                    )
                                                                ],
                                                                style={
                                                                    "fontSize": "12px",
                                                                    "color": "#666",
                                                                },
                                                            )
                                                            if data.get("snippets")
                                                            else html.Div(
                                                                "No direct evidence snippet found.",
                                                                style={
                                                                    "fontSize": "12px",
                                                                    "color": "#999",
                                                                    "marginTop": "4px",
                                                                },
                                                            )
                                                        ),
                                                    ],
                                                    style={"marginBottom": "8px"},
                                                )
                                                for item, data in report.methodology_checklist.evidence.items()
                                            ],
                                            style={"paddingLeft": "18px"},
                                        ),
                                    ],
                                    style={"marginTop": "8px"},
                                ),
                            ],
                            style={"fontSize": "14px"},
                        ),
                    ],
                    className="card",
                    style={"marginBottom": "16px"},
                ),
                # Writing Quality
                html.Div(
                    [
                        html.H4("✍️ Writing Quality"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            "Readability (Flesch): ", style={"fontWeight": "bold"}
                                        ),
                                        html.Span(
                                            f"{report.writing_quality.readability_score:.1f} "
                                            f"({report.writing_quality.education_level})",
                                        ),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            "Avg Sentence Length: ", style={"fontWeight": "bold"}
                                        ),
                                        html.Span(
                                            f"{report.writing_quality.avg_sentence_length:.1f} words"
                                        ),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            "Passive Voice Ratio: ", style={"fontWeight": "bold"}
                                        ),
                                        html.Span(
                                            f"{report.writing_quality.passive_voice_ratio*100:.0f}%"
                                        ),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Span("Jargon Density: ", style={"fontWeight": "bold"}),
                                        html.Span(
                                            f"{report.writing_quality.jargon_density*100:.0f}%"
                                        ),
                                    ]
                                ),
                            ],
                            style={"fontSize": "14px"},
                        ),
                    ],
                    className="card",
                    style={"marginBottom": "16px"},
                ),
                # Contribution Alignment
                html.Div(
                    [
                        html.H4("🎯 Contribution-Finding Alignment"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span("Overlap Score: ", style={"fontWeight": "bold"}),
                                        html.Span(
                                            f"{report.contribution_alignment.overlap_score:.2f}"
                                        ),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            "Unmatched Contributions: ",
                                            style={"fontWeight": "bold"},
                                        ),
                                        html.Span(
                                            ", ".join(
                                                report.contribution_alignment.unmatched_contributions[
                                                    :6
                                                ]
                                            )
                                            or "None ✓"
                                        ),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            "Unmatched Findings: ", style={"fontWeight": "bold"}
                                        ),
                                        html.Span(
                                            ", ".join(
                                                report.contribution_alignment.unmatched_findings[:6]
                                            )
                                            or "None ✓"
                                        ),
                                    ]
                                ),
                            ],
                            style={"fontSize": "14px"},
                        ),
                    ],
                    className="card",
                    style={"marginBottom": "16px"},
                ),
                # Next Steps
                html.Div(
                    [
                        html.H4("✅ Recommended Next Steps"),
                        html.Ol(
                            [
                                html.Li(step, style={"marginBottom": "8px", "fontSize": "14px"})
                                for step in report.next_steps
                            ]
                        ),
                    ],
                    className="card",
                ),
            ]
        )

    except Exception as e:
        import traceback

        return html.Div(
            [
                html.P(f"Error running assessment: {str(e)}", style={"color": "#d32f2f"}),
                html.Pre(
                    traceback.format_exc(),
                    style={
                        "fontSize": "10px",
                        "color": "#999",
                        "maxHeight": "300px",
                        "overflow": "auto",
                    },
                ),
            ]
        )


@callback(
    Output("download-methodology-checklist", "data"),
    Input("export-methodology-csv-btn", "n_clicks"),
    Input("export-methodology-pdf-btn", "n_clicks"),
    State("assessment-doc-dropdown", "value"),
    State("assessment-persona-dropdown", "value"),
    State("assessment-llm-toggle", "value"),
    prevent_initial_call=True,
)
def export_methodology_checklist(csv_clicks, pdf_clicks, selected_doc_id, persona, llm_toggle):
    """Export methodology checklist report to CSV or PDF."""
    if not selected_doc_id:
        raise PreventUpdate

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    if button_id not in {"export-methodology-csv-btn", "export-methodology-pdf-btn"}:
        raise PreventUpdate

    from pathlib import Path

    from scripts.ingest.academic.phd_assessor import PhDQualityAssessor
    from scripts.rag.generate import _get_llm

    collection = _get_query_collection()
    llm_flags = {
        "claims": bool(llm_toggle and "claims" in llm_toggle),
        "data_mismatch": bool(llm_toggle and "data_mismatch" in llm_toggle),
        "citation_misrep": bool(llm_toggle and "citation_misrep" in llm_toggle),
    }
    use_any_llm = any(llm_flags.values())
    llm_client = _get_llm(temperature=0.0) if use_any_llm else None

    citation_db_path = Path(RAG_CONFIG.rag_data_path) / "academic_citation_graph.db"
    assessor = PhDQualityAssessor(
        collection,
        llm_client=llm_client,
        llm_flags=llm_flags,
        citation_db_path=str(citation_db_path) if citation_db_path.exists() else None,
    )
    report = assessor.assess_thesis(selected_doc_id, persona=persona or "supervisor")
    checklist = report.methodology_checklist

    def _build_csv_payload():
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "item",
                "status",
                "count",
                "density_per_1k",
                "summary",
                "keywords",
                "snippet_1",
                "location_1",
                "tags_1",
                "snippet_2",
                "location_2",
                "tags_2",
                "snippet_3",
                "location_3",
                "tags_3",
            ]
        )

        for item, data in checklist.evidence.items():
            snippets = data.get("snippets", [])

            def _get_snippet(idx):
                if idx < len(snippets):
                    return (
                        snippets[idx].get("snippet", ""),
                        snippets[idx].get("location", ""),
                        ", ".join(snippets[idx].get("tags", [])),
                    )
                return ("", "", "")

            s1, l1, t1 = _get_snippet(0)
            s2, l2, t2 = _get_snippet(1)
            s3, l3, t3 = _get_snippet(2)

            writer.writerow(
                [
                    item,
                    "present" if data.get("count", 0) > 0 else "missing",
                    data.get("count", 0),
                    f"{data.get('strength_per_1k', 0):.2f}",
                    data.get("summary", ""),
                    "; ".join(data.get("keywords", [])),
                    s1,
                    l1,
                    t1,
                    s2,
                    l2,
                    t2,
                    s3,
                    l3,
                    t3,
                ]
            )

        filename = f"methodology_checklist_{selected_doc_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return dict(content=output.getvalue(), filename=filename)

    if button_id == "export-methodology-csv-btn":
        return _build_csv_payload()

    if button_id == "export-methodology-pdf-btn":
        if not REPORTLAB_AVAILABLE:
            return _build_csv_payload()

        import base64
        import io

        from reportlab.lib.pagesizes import letter  # type: ignore[import]
        from reportlab.pdfgen import canvas  # type: ignore[import]

        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        def write_lines(lines, start_y):
            y = start_y
            for line in lines:
                pdf.drawString(40, y, line)
                y -= 12
                if y < 60:
                    pdf.showPage()
                    y = height - 60
            return y

        lines = [
            "Methodology Checklist Report",
            f"Document: {selected_doc_id}",
            f"Persona: {persona or 'supervisor'}",
            f"Score: {checklist.score * 100:.0f}%",
            f"Missing Items: {', '.join(checklist.missing_items) if checklist.missing_items else 'None'}",
            f"Scope Note: {checklist.confidence_note}",
            "",
        ]

        for item, data in checklist.evidence.items():
            lines.append(f"Item: {item.replace('_', ' ').title()}")
            lines.append(f"Status: {'Present' if data.get('count', 0) > 0 else 'Missing'}")
            lines.append(
                f"Hits: {data.get('count', 0)} | Density: {data.get('strength_per_1k', 0):.2f}/1k"
            )
            lines.append(f"Summary: {data.get('summary', '')}")
            lines.append(f"Keywords: {', '.join(data.get('keywords', []))}")
            for snippet in data.get("snippets", []):
                tags = ", ".join(snippet.get("tags", []))
                lines.append(
                    f" - {snippet.get('location', 'Unknown')}: {snippet.get('snippet', '')} ({tags})"
                )
            lines.append("")

        wrapped_lines = []
        max_len = 100
        for line in lines:
            while len(line) > max_len:
                wrapped_lines.append(line[:max_len])
                line = line[max_len:]
            wrapped_lines.append(line)

        write_lines(wrapped_lines, height - 60)
        pdf.save()

        pdf_bytes = buffer.getvalue()
        pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        filename = f"methodology_checklist_{selected_doc_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return dict(content=pdf_b64, filename=filename, base64=True)

    raise PreventUpdate


def self_build_argument_flow_figure(flow):
    """Build a Plotly figure for argument flow graph."""
    if not flow or not flow.nodes:
        return go.Figure()

    node_x = []
    node_y = []
    labels = []
    node_index = {}

    for idx, node in enumerate(flow.nodes):
        node_x.append(idx)
        node_y.append(0)
        labels.append(node.get("label", node.get("id", "")))
        node_index[node.get("id")] = idx

    edge_traces = []
    for edge in flow.edges:
        src = node_index.get(edge.get("source"))
        tgt = node_index.get(edge.get("target"))
        if src is None or tgt is None:
            continue
        weight = edge.get("weight", 0.0)
        edge_traces.append(
            go.Scatter(
                x=[src, tgt],
                y=[0, 0],
                mode="lines",
                line=dict(width=max(1.0, 4.0 * weight), color="#667eea"),
                hoverinfo="text",
                text=f"Similarity: {weight:.2f}",
                showlegend=False,
            )
        )

    fig = go.Figure(edge_traces)
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=12, color="#ffb703", line=dict(width=1, color="#333")),
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


# Academic References Callbacks are registered in academic_references.py


@callback(
    Output("analytics-data-store", "data", allow_duplicate=True),
    Output("analytics-status", "children", allow_duplicate=True),
    Input("graph-analytics-tab", "style"),
    prevent_initial_call="initial_duplicate",
    running=[
        (Output("compute-analytics-btn", "disabled"), True, False),
    ],
)
def load_precomputed_analytics(tab_style):
    """Auto-load pre-computed analytics on startup if available."""
    try:
        # Try to load pre-computed analytics from graph metadata
        analytics = graph_store.get_analytics()

        if analytics:
            return analytics, "✓ Pre-computed analytics loaded"
        else:
            # No pre-computed analytics available
            return {}, "Analytics not pre-computed. Click button to compute."
    except Exception as e:
        import traceback

        logger.error(f"Error loading pre-computed analytics: {e}", exc_info=True)
        return {}, "Ready to compute analytics"


@callback(
    Output("analytics-data-store", "data", allow_duplicate=True),
    Output("analytics-status", "children", allow_duplicate=True),
    Input("compute-analytics-btn", "n_clicks"),
    prevent_initial_call=True,
)
def compute_analytics(n_clicks):
    """Compute advanced analytics for the graph on-demand."""
    if not n_clicks:
        return dash.no_update, dash.no_update

    try:
        from scripts.consistency_graph.advanced_analytics import compute_advanced_analytics

        # Load graph data from SQLiteGraphStore
        node_ids = graph_store.get_node_ids()
        if not node_ids:
            return {}, "Error: No nodes found in graph"

        # Build graph data dict
        nodes_dict = {}
        for node_id in node_ids:
            node_data = graph_store.get_node(node_id)
            if node_data:
                nodes_dict[node_id] = node_data

        edges_list = graph_store.get_edges()

        if not nodes_dict:
            return {}, "Error: Graph data not available"

        # Build NetworkX graph
        G = nx.Graph()
        for node_id, node_data in nodes_dict.items():
            G.add_node(node_id, **node_data)

        for edge in edges_list:
            G.add_edge(edge["source"], edge["target"], **edge)

        # Compute analytics
        analytics = compute_advanced_analytics(G)

        # Serialise for storage (convert sets to lists)
        serialised_analytics = {
            "influence_scores": analytics["influence_scores"],
            "communities": {
                "louvain": [list(comm) for comm in analytics["communities"]["louvain"]],
                "label_propagation": [
                    list(comm) for comm in analytics["communities"]["label_propagation"]
                ],
            },
            "topology": analytics["topology"],
            "node_clustering": analytics["node_clustering"],
            "top_influencers": analytics.get("top_influencers", {}),
        }

        return (
            serialised_analytics,
            f"✓ On-demand computation complete ({len(G.nodes())} nodes, {len(G.edges())} edges)",
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {}, f"Error: {str(e)}"


@callback(
    Output("topology-metrics", "children"),
    Output("pagerank-table", "children"),
    Output("betweenness-table", "children"),
    Output("communities-display", "children"),
    Output("relationship-strength-chart", "figure"),
    Input("analytics-data-store", "data"),
)
def update_analytics_displays(analytics):
    """Update analytics display components from computed analytics."""
    if not analytics:
        empty_msg = html.Div(
            "Click 'Compute Analytics' to generate metrics",
            style={"color": "#999", "fontStyle": "italic"},
        )
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data", height=300)
        return empty_msg, empty_msg, empty_msg, empty_msg, empty_fig

    try:
        # Topology metrics
        topology = analytics.get("topology", {})
        topology_content = html.Div(
            [
                html.Div(
                    [html.Strong("Nodes: "), html.Span(f"{topology.get('num_nodes', 0)}")],
                    style={"marginBottom": "4px"},
                ),
                html.Div(
                    [html.Strong("Edges: "), html.Span(f"{topology.get('num_edges', 0)}")],
                    style={"marginBottom": "4px"},
                ),
                html.Div(
                    [html.Strong("Density: "), html.Span(f"{topology.get('density', 0.0):.4f}")],
                    style={"marginBottom": "4px"},
                ),
                html.Div(
                    [
                        html.Strong("Avg Clustering: "),
                        html.Span(f"{topology.get('avg_clustering', 0.0):.4f}"),
                    ],
                    style={"marginBottom": "4px"},
                ),
                html.Div(
                    [
                        html.Strong("Components: "),
                        html.Span(f"{topology.get('num_components', 0)}"),
                    ],
                    style={"marginBottom": "4px"},
                ),
                html.Div(
                    [html.Strong("Diameter: "), html.Span(f"{topology.get('diameter', 'N/A')}")],
                    style={"marginBottom": "4px"},
                ),
                html.Div(
                    [
                        html.Strong("Avg Path Length: "),
                        html.Span(f"{topology.get('avg_path_length', 'N/A')}"),
                    ]
                ),
            ]
        )

        # PageRank table
        top_pagerank = analytics.get("top_influencers", {}).get("by_pagerank", [])
        if top_pagerank:
            pagerank_rows = [
                html.Tr(
                    [
                        html.Td(f"#{i+1}", style={"width": "40px"}),
                        html.Td(node, style={"fontSize": "11px"}),
                        html.Td(
                            f"{score:.4f}", style={"textAlign": "right", "fontFamily": "monospace"}
                        ),
                    ]
                )
                for i, (node, score) in enumerate(top_pagerank[:10])
            ]
            pagerank_table = html.Table(
                [
                    html.Thead(html.Tr([html.Th("Rank"), html.Th("Node"), html.Th("Score")])),
                    html.Tbody(pagerank_rows),
                ],
                style={"width": "100%", "borderCollapse": "collapse"},
            )
        else:
            pagerank_table = html.Div("No data", style={"color": "#999"})

        # Betweenness table
        top_betweenness = analytics.get("top_influencers", {}).get("by_betweenness", [])
        if top_betweenness:
            betweenness_rows = [
                html.Tr(
                    [
                        html.Td(f"#{i+1}", style={"width": "40px"}),
                        html.Td(node, style={"fontSize": "11px"}),
                        html.Td(
                            f"{score:.4f}", style={"textAlign": "right", "fontFamily": "monospace"}
                        ),
                    ]
                )
                for i, (node, score) in enumerate(top_betweenness[:10])
            ]
            betweenness_table = html.Table(
                [
                    html.Thead(html.Tr([html.Th("Rank"), html.Th("Node"), html.Th("Score")])),
                    html.Tbody(betweenness_rows),
                ],
                style={"width": "100%", "borderCollapse": "collapse"},
            )
        else:
            betweenness_table = html.Div("No data", style={"color": "#999"})

        # Communities display
        communities_louvain = analytics.get("communities", {}).get("louvain", [])
        communities_lp = analytics.get("communities", {}).get("label_propagation", [])

        communities_content = html.Div(
            [
                html.Div(
                    [
                        html.Strong("Louvain Method: "),
                        html.Span(f"{len(communities_louvain)} communities"),
                    ],
                    style={"marginBottom": "8px"},
                ),
                html.Div(
                    [
                        html.Strong("Label Propagation: "),
                        html.Span(f"{len(communities_lp)} communities"),
                    ],
                    style={"marginBottom": "8px"},
                ),
                html.Div(
                    [
                        html.Strong("Largest Community: "),
                        html.Span(f"{max([len(c) for c in communities_louvain], default=0)} nodes"),
                    ]
                ),
            ]
        )

        # Relationship strength chart
        node_clustering = analytics.get("node_clustering", {})
        if node_clustering:
            # Create histogram of clustering coefficients
            clustering_values = list(node_clustering.values())
            fig = go.Figure(
                data=[
                    go.Histogram(
                        x=clustering_values,
                        nbinsx=20,
                        marker=dict(color="#667eea"),
                    )
                ]
            )
            fig.update_layout(
                title="Clustering Coefficient Distribution",
                xaxis_title="Clustering Coefficient",
                yaxis_title="Number of Nodes",
                height=350,
                margin=dict(l=40, r=20, t=40, b=40),
            )
        else:
            fig = go.Figure()
            fig.update_layout(title="No data", height=350)

        return topology_content, pagerank_table, betweenness_table, communities_content, fig

    except Exception as e:
        import traceback

        traceback.print_exc()
        error_msg = html.Div(f"Error: {str(e)}", style={"color": "red"})
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Error", height=350)
        return error_msg, error_msg, error_msg, error_msg, empty_fig


@callback(
    Output("conversation-store", "data"),
    Output("conversation-info", "children"),
    Input("new-conversation-btn", "n_clicks"),
    prevent_initial_call=False,
)
def start_new_conversation(n_clicks):
    """Start a new conversation."""
    import uuid
    from datetime import datetime

    conv_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%H:%M:%S")

    return (
        {"id": conv_id, "turns": [], "created_at": timestamp},
        f"Conversation ID: {conv_id} | Started: {timestamp}",
    )


@callback(
    Output("conversation-store", "data", allow_duplicate=True),
    Output("conversation-history", "children"),
    Input("rag-query-answer", "children"),
    State("conversation-store", "data"),
    State("rag-query-input", "value"),
    State("rag-query-status", "children"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def update_conversation_history_on_answer(answer_text, conv_data, query_text, status_text):
    """Update conversation history when answer is ready (not when button is clicked)."""
    if not answer_text or not query_text or not status_text:
        return dash.no_update, dash.no_update

    # Only update if we got a real answer (not an error or empty)
    if isinstance(answer_text, str) and answer_text.strip() and "Error" not in str(answer_text):
        if not conv_data:
            conv_data = {"id": "", "turns": []}

        from datetime import datetime

        from scripts.rag.conversation_manager import ConversationManager

        # Add turn to conversation
        turn = {
            "query": query_text,
            "answer": answer_text[:500],  # Store truncated answer
            "timestamp": datetime.now().isoformat(),
        }
        conv_data["turns"].append(turn)
        turn_number = len(conv_data["turns"])

        # Save to database
        try:
            conv_db_path = Path(RAG_CONFIG.rag_data_path) / "conversations.db"
            conv_manager = ConversationManager(conv_db_path)

            # Ensure conversation is created
            if not conv_data.get("id"):
                import uuid

                conv_data["id"] = str(uuid.uuid4())[:8]

            conv_manager.save_conversation(conv_data["id"])

            # Add this turn to database
            answer_str = str(answer_text)
            conv_manager.add_turn(
                conv_id=conv_data["id"],
                turn_number=turn_number,
                query=query_text,
                answer=answer_str[:1000],  # Truncate for DB
                sources=None,
                tokens_used=0,
                cache_hit=False,
            )
        except Exception as e:
            print(f"Warning: Could not save conversation to DB: {e}")

        # Build history display
        history_items = []
        for i, turn in enumerate(conv_data["turns"], 1):
            query_time = turn.get("timestamp", "")
            if query_time:
                query_time = datetime.fromisoformat(query_time).strftime("%H:%M:%S")

            history_items.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Strong(f"Turn {i} ({query_time})", style={"color": "#667eea"}),
                            ],
                            style={"marginTop": "8px", "marginBottom": "4px"},
                        ),
                        html.Div(
                            [
                                html.Strong("Q: "),
                                html.Span(
                                    turn["query"][:100]
                                    + ("..." if len(turn["query"]) > 100 else "")
                                ),
                            ],
                            style={"fontSize": "12px", "marginBottom": "4px"},
                        ),
                        html.Div(
                            [
                                html.Strong("A: "),
                                html.Span(
                                    turn["answer"][:100]
                                    + ("..." if len(turn["answer"]) > 100 else "")
                                ),
                            ],
                            style={"fontSize": "12px", "color": "#555"},
                        ),
                        html.Hr(style={"margin": "8px 0"}),
                    ]
                )
            )

        if not history_items:
            history_items = [
                html.Div(
                    "No conversation history yet", style={"color": "#999", "fontStyle": "italic"}
                )
            ]

        return conv_data, html.Div(history_items)

    return dash.no_update, dash.no_update


@callback(
    Output("copy-feedback", "children"),
    Output("copy-feedback", "style"),
    Input({"type": "copy-code-btn", "index": ALL}, "n_clicks"),
    State({"type": "copy-code-block", "index": ALL}, "children"),
    prevent_initial_call=True,
)
def handle_copy_code(n_clicks_list, code_blocks):
    """Handle copy button click and provide feedback.

    Args:
        n_clicks_list: List of click counts for each copy button
        code_blocks: List of code block contents

    Returns:
        Feedback message and styling
    """
    # Determine which button was clicked
    if not n_clicks_list or not code_blocks:
        return dash.no_update, dash.no_update

    try:
        # Find the button that was just clicked (highest n_clicks that changed)
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update

        # Extract index from the triggered component
        triggered_id = ctx.triggered[0]["prop_id"]
        if "copy-code-btn" not in triggered_id:
            return dash.no_update, dash.no_update

        # Parse the index from {"type": "copy-code-btn", "index": N}
        import json

        id_obj = json.loads(triggered_id.split(".")[0])
        idx = id_obj.get("index", 0)

        if 0 <= idx < len(code_blocks):
            # Extract text content from code block
            code_text = ""
            if isinstance(code_blocks[idx], dict):
                code_text = code_blocks[idx].get("props", {}).get("children", "")
            elif isinstance(code_blocks[idx], str):
                code_text = code_blocks[idx]

            # In a real app, JavaScript would handle this
            # For now, we show feedback that copy was triggered
            feedback_msg = f"✓ Code block copied to clipboard! ({len(code_text)} chars)"
            style = {
                "position": "fixed",
                "top": "20px",
                "right": "20px",
                "backgroundColor": "#4caf50",
                "color": "white",
                "padding": "12px 16px",
                "borderRadius": "4px",
                "zIndex": 1000,
                "boxShadow": "0 2px 8px rgba(0,0,0,0.2)",
                "fontSize": "13px",
                "transition": "opacity 0.3s",
            }
            return feedback_msg, style
    except Exception as e:
        print(f"Error in copy handler: {e}")

    return dash.no_update, dash.no_update


@callback(
    Output("copy-feedback", "style", allow_duplicate=True),
    Input("copy-feedback-timer", "n_intervals"),
    State("copy-feedback", "style"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def hide_copy_feedback(n_intervals, current_style):
    """Hide copy feedback message after 2 seconds."""
    if n_intervals > 0 and current_style:
        # Fade out
        style = current_style.copy()
        style["opacity"] = "0"
        style["pointerEvents"] = "none"
        return style
    return dash.no_update


# ============================================================================
# Conversation Browser Callbacks
# ============================================================================


@callback(
    Output("conversation-list-dropdown", "options"),
    Output("conversation-list-dropdown", "value"),
    Input("conversation-search-input", "value"),
    prevent_initial_call=False,
)
def update_conversation_list(search_query):
    """Update conversation list based on search."""
    from scripts.rag.conversation_manager import ConversationManager

    try:
        conv_db_path = Path(RAG_CONFIG.rag_data_path) / "conversations.db"
        conv_manager = ConversationManager(conv_db_path)

        if search_query:
            conversations = conv_manager.search_conversations(search_query, limit=50)
        else:
            conversations = conv_manager.load_conversations(limit=50, offset=0)

        options = []
        for c in conversations:
            title = c.get("title") or f"Unnamed ({c['created_at'][:10]})"
            label = f"{title} - {c['turn_count']} turns"
            options.append({"label": label, "value": c["id"]})

        return options, ""
    except Exception as e:
        print(f"Error loading conversations: {e}")
        return [{"label": "Error loading conversations", "value": ""}], ""


@callback(
    Output("conversation-details-display", "children"),
    Output("conversation-preview-display", "children"),
    Input("load-conversation-btn", "n_clicks"),
    State("conversation-list-dropdown", "value"),
    prevent_initial_call=True,
)
def load_conversation_details(n_clicks, conv_id):
    """Load and display conversation details."""
    if not conv_id:
        return "Please select a conversation", ""

    from scripts.rag.conversation_manager import ConversationManager

    try:
        conv_db_path = Path(RAG_CONFIG.rag_data_path) / "conversations.db"
        conv_manager = ConversationManager(conv_db_path)
        conv = conv_manager.load_conversation(conv_id)

        if not conv:
            return "Conversation not found", ""

        # Build details
        details = html.Div(
            [
                html.P(f"**ID:** {conv['id']}", style={"fontSize": "12px", "marginBottom": "4px"}),
                html.P(
                    f"**Title:** {conv.get('title', 'Untitled')}",
                    style={"fontSize": "12px", "marginBottom": "4px"},
                ),
                html.P(
                    f"**Created:** {conv.get('created_at', 'N/A')}",
                    style={"fontSize": "12px", "marginBottom": "4px"},
                ),
                html.P(
                    f"**Updated:** {conv.get('updated_at', 'N/A')}",
                    style={"fontSize": "12px", "marginBottom": "4px"},
                ),
                html.P(
                    f"**Turns:** {conv.get('turn_count', 0)}",
                    style={"fontSize": "12px", "marginBottom": "4px"},
                ),
                html.P(
                    f"**Total Tokens:** {conv.get('total_tokens', 0)}",
                    style={"fontSize": "12px", "marginBottom": "4px"},
                ),
                html.P(
                    f"**Cached Results:** {conv.get('cached_results', 0)}",
                    style={"fontSize": "12px"},
                ),
            ]
        )

        # Build preview
        preview_items = []
        for turn in conv.get("turns", [])[:5]:  # Show first 5 turns
            preview_items.append(
                html.Div(
                    [
                        html.Strong(
                            f"Turn {turn['turn_number']} - {turn.get('timestamp', 'N/A')[:19]}",
                            style={"color": "#667eea"},
                        ),
                        html.Div(
                            f"Q: {turn['query'][:100]}...",
                            style={"fontSize": "11px", "marginTop": "4px"},
                        ),
                        html.Div(
                            f"A: {turn['answer'][:100]}...",
                            style={"fontSize": "11px", "color": "#555", "marginBottom": "8px"},
                        ),
                    ],
                    style={
                        "marginBottom": "8px",
                        "borderBottom": "1px solid #ddd",
                        "paddingBottom": "8px",
                    },
                )
            )

        if not preview_items:
            preview_items = [html.Div("No turns in this conversation", style={"color": "#999"})]

        return details, html.Div(preview_items)
    except Exception as e:
        print(f"Error loading conversation: {e}")
        return f"Error: {str(e)}", ""


@callback(
    Output("conversation-search-input", "value", allow_duplicate=True),
    Output("conversation-details-display", "children", allow_duplicate=True),
    Output("conversation-preview-display", "children", allow_duplicate=True),
    Input("delete-conversation-btn", "n_clicks"),
    State("conversation-list-dropdown", "value"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def delete_conversation(n_clicks, conv_id):
    """Delete a conversation."""
    if not conv_id:
        return dash.no_update, dash.no_update, dash.no_update

    from scripts.rag.conversation_manager import ConversationManager

    try:
        conv_db_path = Path(RAG_CONFIG.rag_data_path) / "conversations.db"
        conv_manager = ConversationManager(conv_db_path)
        success = conv_manager.delete_conversation(conv_id)

        if success:
            return "", "", ""  # Clear search to refresh list, clear details and preview
        else:
            return dash.no_update, dash.no_update, dash.no_update
    except Exception as e:
        print(f"Error deleting conversation: {e}")
        return dash.no_update, dash.no_update, dash.no_update


# ============================================================================
# Export Callbacks (Feature 6)
# ============================================================================


@callback(
    Output("download-export", "data"),
    Input("export-markdown-btn", "n_clicks"),
    Input("export-pdf-btn", "n_clicks"),
    State("conversation-list-dropdown", "value"),
    prevent_initial_call=True,
)
def export_conversation(md_clicks, pdf_clicks, conv_id):
    """Export conversation to Markdown or PDF."""
    if not conv_id:
        raise PreventUpdate

    from scripts.rag.conversation_manager import ConversationManager

    try:
        conv_db_path = Path(RAG_CONFIG.rag_data_path) / "conversations.db"
        conv_manager = ConversationManager(conv_db_path)
        conv_data = conv_manager.load_conversation(conv_id)

        if not conv_data:
            raise PreventUpdate

        # Determine which button was clicked
        ctx = dash.callback_context
        button_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        if button_id == "export-markdown-btn":
            # Export to Markdown
            md_content = ExportManager.export_conversation_markdown(
                conv_id,
                conv_data.get("turns", []),
                metadata={
                    "created_at": conv_data.get("created_at"),
                    "title": conv_data.get("title"),
                    "total_turns": len(conv_data.get("turns", [])),
                },
            )
            return dict(content=md_content, filename=f"conversation_{conv_id}.md")

        elif button_id == "export-pdf-btn":
            # Export to PDF
            if not REPORTLAB_AVAILABLE:
                # Fallback to markdown if PDF not available
                md_content = ExportManager.export_conversation_markdown(
                    conv_id,
                    conv_data.get("turns", []),
                )
                return dict(content=md_content, filename=f"conversation_{conv_id}.md")

            pdf_bytes = ExportManager.export_conversation_pdf(
                conv_id,
                conv_data.get("turns", []),
                metadata={
                    "created_at": conv_data.get("created_at"),
                    "title": conv_data.get("title"),
                },
            )
            # Encode bytes to base64 string for Dash download
            import base64

            pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
            return dict(content=pdf_b64, filename=f"conversation_{conv_id}.pdf", base64=True)

        raise PreventUpdate

    except Exception as e:
        print(f"Error exporting conversation: {e}")
        raise PreventUpdate


# ============================================================================
# Benchmark Callbacks (Feature 7)
# ============================================================================

benchmark_manager = None


def get_benchmark_manager():
    """Get or create benchmark manager singleton."""
    global benchmark_manager
    if benchmark_manager is None:
        try:
            benchmark_manager = BenchmarkManager(Path(RAG_CONFIG.rag_data_path) / "benchmarks.db")
        except Exception as e:
            print(f"Error initializing benchmark manager: {e}")
    return benchmark_manager


@callback(
    Output("benchmark-summary", "children"),
    Output("response-time-chart", "figure"),
    Output("cache-rate-chart", "figure"),
    Output("slowest-queries-table", "children"),
    Output("system-metrics-display", "children"),
    Output("relevancy-stats-display", "children"),
    Input("refresh-benchmarks-btn", "n_clicks"),
    Input("benchmark-time-range", "value"),
    prevent_initial_call=False,
)
def update_benchmarks(refresh_clicks, time_range):
    """Update benchmark displays including system metrics and relevancy stats."""
    manager = get_benchmark_manager()
    if not manager:
        return "Benchmark manager unavailable", {}, {}, "No data", "N/A", "N/A"

    try:
        # Get statistics
        hours = time_range if time_range > 0 else None
        stats = manager.get_statistics(time_range_hours=hours)

        # Build summary HTML
        summary_div = html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Strong(f"{stats['total_queries']}"),
                                html.Div(
                                    "Total Queries", style={"fontSize": "12px", "color": "#666"}
                                ),
                            ],
                            style={
                                "padding": "12px",
                                "backgroundColor": "#f8f9fa",
                                "borderRadius": "4px",
                                "marginRight": "8px",
                            },
                        ),
                        html.Div(
                            [
                                html.Strong(f"{stats['avg_total_time']:.2f}s"),
                                html.Div("Avg Time", style={"fontSize": "12px", "color": "#666"}),
                            ],
                            style={
                                "padding": "12px",
                                "backgroundColor": "#f8f9fa",
                                "borderRadius": "4px",
                                "marginRight": "8px",
                            },
                        ),
                        html.Div(
                            [
                                html.Strong(f"{stats['cache_hit_rate']:.1f}%"),
                                html.Div(
                                    "Cache Hit Rate", style={"fontSize": "12px", "color": "#666"}
                                ),
                            ],
                            style={
                                "padding": "12px",
                                "backgroundColor": "#f8f9fa",
                                "borderRadius": "4px",
                                "marginRight": "8px",
                            },
                        ),
                        html.Div(
                            [
                                html.Strong(f"{stats['code_query_rate']:.1f}%"),
                                html.Div(
                                    "Code Queries", style={"fontSize": "12px", "color": "#666"}
                                ),
                            ],
                            style={
                                "padding": "12px",
                                "backgroundColor": "#f8f9fa",
                                "borderRadius": "4px",
                            },
                        ),
                    ],
                    style={"display": "flex", "gap": "8px"},
                ),
            ]
        )

        # Get time series data
        time_series = manager.get_time_series(metric="total_time", time_range_hours=hours or 24)

        # Build response time chart
        if time_series:
            times, values = zip(*time_series)
            response_fig = {
                "data": [
                    {
                        "x": list(times),
                        "y": list(values),
                        "type": "scatter",
                        "mode": "lines+markers",
                        "name": "Response Time",
                        "line": {"color": "#667eea"},
                    }
                ],
                "layout": {
                    "title": "Response Time Trend",
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Time (seconds)"},
                    "hovermode": "x unified",
                    "height": 300,
                },
            }
        else:
            response_fig = {"data": [], "layout": {"title": "No data available"}}

        # Build cache rate chart
        cache_fig = {
            "data": [
                {
                    "values": [stats["cache_hit_rate"], 100 - stats["cache_hit_rate"]],
                    "labels": ["Cache Hits", "Cache Misses"],
                    "type": "pie",
                    "marker": {"colors": ["#28a745", "#dc3545"]},
                }
            ],
            "layout": {
                "title": "Cache Hit Rate",
                "height": 300,
            },
        }

        # Get slowest queries
        slowest = manager.get_slowest_queries(limit=5, time_range_hours=hours)

        slowest_div = html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Query", style={"textAlign": "left", "padding": "8px"}),
                            html.Th("Time", style={"textAlign": "right", "padding": "8px"}),
                        ]
                    )
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(
                                    q["query"][:60] + "..." if len(q["query"]) > 60 else q["query"],
                                    style={"padding": "8px"},
                                ),
                                html.Td(
                                    f"{q['total_time']:.2f}s",
                                    style={"textAlign": "right", "padding": "8px"},
                                ),
                            ]
                        )
                        for q in slowest
                    ]
                ),
            ],
            style={"width": "100%", "borderCollapse": "collapse", "fontSize": "12px"},
        )

        # System metrics display
        system_metrics_div = html.Div(
            [
                html.Table(
                    [
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(
                                            "Avg CPU Usage:",
                                            style={"padding": "6px", "fontWeight": 500},
                                        ),
                                        html.Td(
                                            f"{stats.get('avg_cpu_percent', 0):.1f}%",
                                            style={"padding": "6px"},
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Avg RAM Usage:",
                                            style={"padding": "6px", "fontWeight": 500},
                                        ),
                                        html.Td(
                                            f"{stats.get('avg_ram_mb', 0):.0f} MB",
                                            style={"padding": "6px"},
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Avg GPU Usage:",
                                            style={"padding": "6px", "fontWeight": 500},
                                        ),
                                        html.Td(
                                            (
                                                f"{stats.get('avg_gpu_percent', 0):.1f}%"
                                                if stats.get("avg_gpu_percent")
                                                else "N/A"
                                            ),
                                            style={"padding": "6px"},
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Avg VRAM Usage:",
                                            style={"padding": "6px", "fontWeight": 500},
                                        ),
                                        html.Td(
                                            (
                                                f"{stats.get('avg_vram_mb', 0):.0f} MB"
                                                if stats.get("avg_vram_mb")
                                                else "N/A"
                                            ),
                                            style={"padding": "6px"},
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Avg Network Latency:",
                                            style={"padding": "6px", "fontWeight": 500},
                                        ),
                                        html.Td(
                                            (
                                                f"{stats.get('avg_network_latency_ms', 0):.1f} ms"
                                                if stats.get("avg_network_latency_ms")
                                                else "N/A"
                                            ),
                                            style={"padding": "6px"},
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                    style={"width": "100%", "borderCollapse": "collapse"},
                )
            ]
        )

        # Relevancy stats display
        rel_stats = manager.get_relevancy_stats(time_range_hours=hours)

        # Handle case where rel_stats is empty (database error)
        if rel_stats:
            relevancy_div = html.Div(
                [
                    html.Table(
                        [
                            html.Tbody(
                                [
                                    html.Tr(
                                        [
                                            html.Td(
                                                "Total Rated:",
                                                style={"padding": "6px", "fontWeight": 500},
                                            ),
                                            html.Td(
                                                f"{rel_stats.get('total_rated', 0)}",
                                                style={"padding": "6px"},
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td(
                                                "Avg Rating:",
                                                style={"padding": "6px", "fontWeight": 500},
                                            ),
                                            html.Td(
                                                f"{rel_stats.get('avg_rating', 0):.2f}/5",
                                                style={
                                                    "padding": "6px",
                                                    "color": "#667eea",
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td(
                                                "Rating Distribution:",
                                                style={"padding": "6px", "fontWeight": 500},
                                            ),
                                            html.Td(
                                                " | ".join(
                                                    [
                                                        f"{rating}★: {count}"
                                                        for rating, count in sorted(
                                                            rel_stats.get(
                                                                "distribution", {}
                                                            ).items()
                                                        )
                                                    ]
                                                ),
                                                style={"padding": "6px"},
                                            ),
                                        ]
                                    ),
                                ]
                            )
                        ],
                        style={"width": "100%", "borderCollapse": "collapse"},
                    )
                ]
            )
        else:
            relevancy_div = html.Div(
                "No relevancy data available", style={"padding": "10px", "color": "#888"}
            )

        return summary_div, response_fig, cache_fig, slowest_div, system_metrics_div, relevancy_div

    except Exception as e:
        print(f"Error updating benchmarks: {e}")
        return f"Error: {str(e)}", {}, {}, "Error loading data", "Error", "Error"


@callback(
    Output("filtered-relevancy-queries", "children"),
    Input("relevancy-rating-filter", "value"),
    Input("benchmark-time-range", "value"),
    prevent_initial_call=False,
)
def update_relevancy_filter(min_rating, time_range):
    """Update relevancy filtered queries display."""
    manager = get_benchmark_manager()
    if not manager:
        return "Manager unavailable"

    try:
        hours = time_range if time_range > 0 else None

        if min_rating == 0:
            # Show all rated queries
            queries = manager.get_queries_by_relevancy(time_range_hours=hours, limit=10)
        else:
            # Show only queries with specific minimum rating
            queries = manager.get_queries_by_relevancy(
                min_rating=min_rating, time_range_hours=hours, limit=10
            )

        if not queries:
            return html.Div(
                "No queries with that rating", style={"color": "#999", "fontStyle": "italic"}
            )

        query_rows = []
        for q in queries:
            rating_stars = "★" * int(q["rating"]) + "☆" * (5 - int(q["rating"]))
            query_rows.append(
                html.Tr(
                    [
                        html.Td(
                            q["query"][:40] + "..." if len(q["query"]) > 40 else q["query"],
                            style={"padding": "6px", "fontSize": "11px"},
                        ),
                        html.Td(
                            rating_stars,
                            style={"padding": "6px", "color": "#FFB800", "fontWeight": "bold"},
                        ),
                        html.Td(
                            f"{q['total_time']:.2f}s",
                            style={"padding": "6px", "textAlign": "right"},
                        ),
                    ]
                )
            )

        return html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th(
                                "Query",
                                style={
                                    "textAlign": "left",
                                    "padding": "6px",
                                    "borderBottom": "1px solid #ddd",
                                },
                            ),
                            html.Th(
                                "Rating",
                                style={
                                    "textAlign": "center",
                                    "padding": "6px",
                                    "borderBottom": "1px solid #ddd",
                                },
                            ),
                            html.Th(
                                "Time",
                                style={
                                    "textAlign": "right",
                                    "padding": "6px",
                                    "borderBottom": "1px solid #ddd",
                                },
                            ),
                        ]
                    )
                ),
                html.Tbody(query_rows),
            ],
            style={"width": "100%", "borderCollapse": "collapse", "fontSize": "12px"},
        )

    except Exception as e:
        print(f"Error updating relevancy filter: {e}")
        return f"Error: {str(e)}"


@callback(
    Output("download-benchmark-report", "data"),
    Input("export-benchmark-report-btn", "n_clicks"),
    State("benchmark-time-range", "value"),
    prevent_initial_call=True,
)
def export_benchmark_report(n_clicks, time_range):
    """Export benchmark report to Markdown."""
    if not n_clicks:
        raise PreventUpdate

    manager = get_benchmark_manager()
    if not manager:
        raise PreventUpdate

    try:
        hours = time_range if time_range > 0 else None
        report = manager.export_report("", time_range_hours=hours)

        return dict(
            content=report,
            filename=f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        )

    except Exception as e:
        print(f"Error exporting benchmark report: {e}")
        raise PreventUpdate


# Check if reportlab is available
try:
    from reportlab.lib.pagesizes import letter  # type: ignore[import]

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ============================================================================
# Dependencies Tab Callbacks
# ============================================================================


@callback(
    Output("dep-network-graph", "figure"),
    Output("dep-service-matrix", "figure"),
    Output("dep-metric-services", "children"),
    Output("dep-metric-internal-calls", "children"),
    Output("dep-metric-shared-deps", "children"),
    Output("dep-metric-circular", "children"),
    Output("dep-info-message", "children"),
    Input("dependencies-tab", "style"),
    Input("dep-node-count", "value"),
    State("show-node-names-toggle", "value"),
    prevent_initial_call=True,
)
def build_dependency_visualisation(tab_style, node_count, show_names_list):
    """Build dependency graph visualisation from code nodes."""
    import networkx as nx
    import numpy as np
    import plotly.graph_objects as go

    print(f"\n{'='*80}")
    print(f"DEBUG: build_dependency_visualisation() called with tab_style={tab_style}")
    print(f"DEBUG: tab_style type: {type(tab_style)}")

    try:
        # Check if tab is visible - properly handle different input types
        is_visible = False
        if tab_style is not None:
            print(f"DEBUG: tab_style is not None, checking visibility...")
            if isinstance(tab_style, dict):
                print(f"DEBUG: tab_style is dict with keys: {tab_style.keys()}")
                is_visible = tab_style.get("display") == "block"
                print(f"DEBUG: is_visible from dict check: {is_visible}")
            elif tab_style == {"display": "block"}:
                is_visible = True
                print(f"DEBUG: is_visible from equality check: {is_visible}")
        else:
            print(f"DEBUG: tab_style is None")

        if not is_visible:
            print(f"DEBUG: Tab not visible, raising PreventUpdate")
            raise PreventUpdate

        print(f"DEBUG: Tab is visible, proceeding with visualisation build")

        # Get node IDs to check if graph is loaded
        print(f"DEBUG: Checking if graph_store has nodes...")
        node_ids = graph_store.get_node_ids()
        if not node_ids:
            print("DEBUG: No nodes in graph_store, attempting to load metadata")
            graph_store.load_metadata()
            node_ids = graph_store.get_node_ids()

        if not node_ids:
            print("DEBUG: Still no nodes available - graph not loaded yet")
            empty_fig = go.Figure().add_annotation(
                text="Graph not loaded - navigate to Graph tab first"
            )
            return (
                empty_fig,
                empty_fig,
                "Services: 0",
                "Internal Calls: 0",
                "Shared Deps: 0",
                "Circular Paths: 0",
                "Load a graph first",
            )

        print(f"DEBUG: Found {len(node_ids)} nodes in graph_store")

        # Get full node data from graph_store
        all_nodes_data = {}
        for node_id in node_ids:
            node_data = graph_store.get_node(node_id)
            if node_data:
                all_nodes_data[node_id] = node_data

        print(f"DEBUG: Processing {len(all_nodes_data)} nodes for dependency visualisation")

        # Extract code nodes (filter for nodes with source_category)
        code_nodes_with_metadata = {}
        categories_found = set()
        processed_count = 0
        code_categories = {"code", "source"}
        for node_id, data in all_nodes_data.items():
            processed_count += 1
            if isinstance(data, dict):
                category = data.get("source_category", "")
                categories_found.add(category)
                if category in code_categories:
                    code_nodes_with_metadata[node_id] = data
            else:
                print(f"DEBUG: node_id={node_id}, data is not dict, type={type(data)}")

        print(
            f"DEBUG: Processed {processed_count} nodes, found {len(code_nodes_with_metadata)} code nodes"
        )
        print(f"DEBUG: Categories in graph: {categories_found}")

        if not code_nodes_with_metadata:
            print(f"DEBUG: No code nodes found. Available categories: {categories_found}")
            empty_fig = go.Figure().add_annotation(text="No code nodes found in graph")
            return (
                empty_fig,
                empty_fig,
                "Services: 0",
                "Internal Calls: 0",
                "Shared Deps: 0",
                "Circular Paths: 0",
                "This tab displays dependency analysis for code repositories. Your current data contains documentation only.",
            )

        # Limit nodes if requested (default to 50)
        effective_node_count = min(node_count or 50, len(code_nodes_with_metadata))

        # First pass: build full graph to compute node importance scores
        temp_graph = nx.DiGraph()
        for node_id in code_nodes_with_metadata.keys():
            temp_graph.add_node(node_id)

        # Add edges to compute connectivity
        for node_id, data in code_nodes_with_metadata.items():
            internal_calls = data.get("internal_calls", [])
            if internal_calls:
                calls_list = (
                    internal_calls if isinstance(internal_calls, list) else [internal_calls]
                )
                for called_service in calls_list:
                    for target_id in code_nodes_with_metadata.keys():
                        target_data = code_nodes_with_metadata[target_id]
                        target_service = target_data.get("service_name") or target_data.get(
                            "service"
                        )
                        if target_service == called_service or called_service in str(target_id):
                            if node_id != target_id:
                                temp_graph.add_edge(node_id, target_id)
                            break

            # Also add shared dependency edges
            deps_a = set(
                data.get("dependencies", []) if isinstance(data.get("dependencies"), list) else []
            )
            if deps_a:
                for other_id, other_data in code_nodes_with_metadata.items():
                    if other_id != node_id:
                        deps_b = set(
                            other_data.get("dependencies", [])
                            if isinstance(other_data.get("dependencies"), list)
                            else []
                        )
                        if deps_a & deps_b:  # Shared dependencies exist
                            temp_graph.add_edge(node_id, other_id)

        # Score nodes by total degree (in + out)
        node_scores = [
            (node_id, temp_graph.degree(node_id)) for node_id in code_nodes_with_metadata.keys()
        ]
        node_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top N nodes
        top_node_ids = [node_id for node_id, _ in node_scores[:effective_node_count]]
        limited_code_nodes = {nid: code_nodes_with_metadata[nid] for nid in top_node_ids}

        print(
            f"DEBUG: Limited to top {len(limited_code_nodes)} nodes (from {len(code_nodes_with_metadata)} total)"
        )

        # Build dependency graph with limited nodes
        dep_graph = nx.DiGraph()

        # Add nodes with service metadata
        for node_id, data in limited_code_nodes.items():
            service = data.get("service_name") or data.get("service", node_id)
            language = data.get("language", "")
            dep_graph.add_node(node_id, service=service, language=language)

        # Add edges for internal calls
        internal_calls_edges = 0
        for node_id, data in limited_code_nodes.items():
            internal_calls = data.get("internal_calls", [])
            if internal_calls:
                calls_list = (
                    internal_calls if isinstance(internal_calls, list) else [internal_calls]
                )
                for called_service in calls_list:
                    for target_id, target_data in limited_code_nodes.items():
                        target_service = target_data.get("service_name") or target_data.get(
                            "service"
                        )
                        if target_service == called_service or called_service in str(target_id):
                            if node_id != target_id and not dep_graph.has_edge(node_id, target_id):
                                dep_graph.add_edge(
                                    node_id, target_id, type="internal_call", weight=1.0
                                )
                                internal_calls_edges += 1
                            break

        # Add edges for shared dependencies
        dependency_edges = 0
        for i, (node_id_a, data_a) in enumerate(limited_code_nodes.items()):
            deps_a = set(
                data_a.get("dependencies", [])
                if isinstance(data_a.get("dependencies"), list)
                else []
            )
            for node_id_b, data_b in list(limited_code_nodes.items())[i + 1 :]:
                deps_b = set(
                    data_b.get("dependencies", [])
                    if isinstance(data_b.get("dependencies"), list)
                    else []
                )
                shared = deps_a & deps_b
                if shared:
                    shared_list = sorted(shared)
                    if not dep_graph.has_edge(node_id_a, node_id_b):
                        weight = (
                            len(shared) / max(len(deps_a), len(deps_b))
                            if max(len(deps_a), len(deps_b)) > 0
                            else 0
                        )
                        dep_graph.add_edge(
                            node_id_a,
                            node_id_b,
                            type="shared_deps",
                            shared=shared_list,
                            weight=weight,
                        )
                        dependency_edges += 1

        # Detect circular dependencies
        circular_deps = list(nx.simple_cycles(dep_graph))
        circular_services = set()
        for cycle in circular_deps:
            circular_services.update(cycle)

        # Build network graph visualisation using spring layout
        pos = nx.spring_layout(dep_graph, k=2, iterations=50, seed=42)

        # Language colours
        language_colours = {
            "java": "#0066cc",
            "groovy": "#228B22",
            "kotlin": "#FF9933",
            "gradle": "#9966cc",
            "python": "#3776ab",
            "javascript": "#f1e05a",
            "typescript": "#2b7489",
        }

        # Create network graph
        edge_x = []
        edge_y = []
        edge_colours = []
        edge_widths = []

        for source, target, data in dep_graph.edges(data=True):
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            # Colour edges based on type
            # N.B. Plotly does not support hover text on line segments, applies line hover to attached node
            # Online forums suggest intrapolating with invisible nodes for hover
            # TODO: Explore cost of adding invisible midpoints for hover info
            if data.get("type") == "internal_call":
                edge_colours.extend(["#ff9933", "#ff9933", "#ff9933"])
                edge_widths.extend([2, 2, 2])
            else:
                edge_colours.extend(["#66ccff", "#66ccff", "#66ccff"])
                edge_widths.extend([1, 1, 1])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            showlegend=False,
        )

        # Add nodes
        node_x = []
        node_y = []
        node_colours = []
        node_sizes = []
        node_labels = []
        node_hover = []
        node_ids = []

        for node_id, data in dep_graph.nodes(data=True):
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            node_ids.append(node_id)

            service = data.get("service", node_id)
            language = data.get("language", "")

            # Colour by language, or red if in circular dependency
            if node_id in circular_services:
                colour = "#ff4d4d"
            else:
                colour = language_colours.get(language.lower() if language else "", "#4da6ff")

            node_colours.append(colour)

            # For display label, prefer display name for academic nodes
            node_data = limited_code_nodes.get(node_id, {})
            if node_data.get("source_category") == "academic_reference":
                display_name = get_display_name(node_data, node_id)
                label_text = display_name[:20]  # Truncate to 20 chars for display
            else:
                label_text = service[:20]  # Truncate service name

            node_labels.append(label_text)

            # Node size based on degree
            degree = dep_graph.degree(node_id)
            node_sizes.append(max(15, 15 + degree * 2))

            hover_text = f"<b>{service}</b><br>Language: {language}<br>Node: {node_id}"
            if node_id in circular_services:
                hover_text += "<br><b style='color: red'>⚠️ In circular dependency</b>"
            node_hover.append(hover_text)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode=(
                "markers+text"
                if (show_names_list and "show-names" in show_names_list)
                else "markers"
            ),
            text=node_labels,
            customdata=node_ids,
            textposition="top center",
            hoverinfo="text",
            hovertext=node_hover,
            marker=dict(
                size=node_sizes, color=node_colours, line=dict(width=2, color="white"), opacity=0.9
            ),
            showlegend=False,
        )

        # Create network figure
        network_fig = go.Figure(data=[edge_trace, node_trace])
        network_fig.update_layout(
            title="Service Dependency Network (Click a node to select)",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#f9f9f9",
            height=600,
        )

        # Build service call matrix
        services = sorted(list(limited_code_nodes.keys()))
        matrix = [[0.0 for _ in services] for _ in services]

        for i, node_a in enumerate(services):
            for j, node_b in enumerate(services):
                if dep_graph.has_edge(node_a, node_b):
                    edges_data = dep_graph.get_edge_data(node_a, node_b)
                    matrix[i][j] = edges_data.get("weight", 1.0)

        # Create heatmap
        service_labels = [
            limited_code_nodes[nid].get("service_name")
            or limited_code_nodes[nid].get("service", nid)
            for nid in services
        ]
        service_labels = [label[:15] for label in service_labels]  # Truncate

        heatmap_fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=service_labels,
                y=service_labels,
                colorscale="YlOrRd",
                hovertemplate="From: %{y}<br>To: %{x}<br>Weight: %{z:.2f}<extra></extra>",
                colorbar=dict(title="Dependency<br>Strength"),
            )
        )

        heatmap_fig.update_layout(
            title="Service Call Matrix (Who Calls Whom)",
            xaxis_title="Called Service",
            yaxis_title="Calling Service",
            height=500,
        )

        # Build metrics
        metrics_services = f"Services: {len(dep_graph.nodes())}"
        metrics_internal = f"Internal Calls: {internal_calls_edges}"
        metrics_shared = f"Shared Dependencies: {dependency_edges}"
        metrics_circular = f"Circular Paths: {len(circular_deps)}"

        if circular_deps:
            warning_msg = f"⚠️ Found {len(circular_deps)} circular dependency path(s)"
        else:
            warning_msg = "✅ No circular dependencies detected"

        return (
            network_fig,
            heatmap_fig,
            metrics_services,
            metrics_internal,
            metrics_shared,
            metrics_circular,
            warning_msg,
        )

    except PreventUpdate:
        raise

    except Exception as e:
        print(f"Error building dependency visualisation: {e}")
        import traceback

        traceback.print_exc()
        empty_fig = go.Figure().add_annotation(text=f"Error: {str(e)}")
        return (
            empty_fig,
            empty_fig,
            "Services: Error",
            "Internal Calls: Error",
            "Shared Deps: Error",
            "Circular Paths: Error",
            f"Error generating visualisation: {str(e)}",
        )


@callback(
    Output("dep-node-count", "value"),
    Output("dep-node-prev", "style"),
    Output("dep-node-next", "style"),
    Output("dep-node-count-input", "value"),
    Input("dep-node-prev", "n_clicks"),
    Input("dep-node-next", "n_clicks"),
    Input("dep-node-count-input", "value"),
    State("dep-node-count", "value"),
    prevent_initial_call=True,
)
def update_dep_node_count(prev_clicks, next_clicks, input_value, current_value):
    """Update dependencies node count via Previous/Next buttons or direct input."""
    from dash import callback_context

    if not callback_context.triggered:
        raise PreventUpdate

    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    new_value = current_value

    if trigger_id == "dep-node-prev" and current_value > 10:
        new_value = max(10, current_value - 10)
    elif trigger_id == "dep-node-next" and current_value < 500:
        new_value = min(500, current_value + 10)
    elif trigger_id == "dep-node-count-input" and input_value is not None:
        # Validate and constrain input value
        new_value = max(10, min(500, int(input_value)))
        # Round to nearest 10
        new_value = round(new_value / 10) * 10

    # Style button disabled states
    prev_style = {
        "padding": "8px 16px",
        "marginRight": "8px",
        "border": "1px solid #ddd",
        "background": "#f5f5f5",
        "color": "#999",
        "borderRadius": "4px",
        "cursor": "not-allowed",
        "fontSize": "14px",
        "fontWeight": "500",
        "opacity": "0.6",
    }
    next_style = {
        "padding": "8px 16px",
        "border": "1px solid #ddd",
        "background": "#f5f5f5",
        "color": "#999",
        "borderRadius": "4px",
        "cursor": "not-allowed",
        "fontSize": "14px",
        "fontWeight": "500",
        "opacity": "0.6",
    }

    enabled_style = {
        "padding": "8px 16px",
        "border": "1px solid #667eea",
        "background": "white",
        "color": "#667eea",
        "borderRadius": "4px",
        "cursor": "pointer",
        "fontSize": "14px",
        "fontWeight": "500",
    }

    # Apply enabled/disabled states
    if new_value > 10:
        prev_style = enabled_style.copy()
        prev_style["marginRight"] = "8px"

    if new_value < 500:
        next_style = enabled_style.copy()

    return new_value, prev_style, next_style, new_value


# ============================================================================
# Metrics Dashboard Callbacks
# ============================================================================


@callback(
    Output("metrics-summary-display", "children"),
    Output("metrics-figure", "figure"),
    Output("metrics-model-stats", "children"),
    Output("metrics-health-status", "children"),
    Output("metrics-cost-display", "children"),
    Input("refresh-metrics-btn", "n_clicks"),
    Input("metrics-refresh-interval", "n_intervals"),
    prevent_initial_call=False,
)
def update_metrics_display(refresh_clicks, intervals):
    """Update metrics display with current system metrics."""
    try:
        from scripts.utils.metrics_export import get_metrics_collector

        collector = get_metrics_collector()
        stats = collector.get_stats()
        health = collector.get_health_status()
        model_stats = collector.get_model_stats()
        cost = collector.estimate_cost()

        # Generate visualisation
        figure = stats.to_plotly_figure()

        # Generate HTML summary
        summary_html = html.Div(
            [
                dcc.Markdown(
                    collector.to_html_summary(),
                    dangerously_allow_html=True,
                )
            ]
        )

        # Model stats display
        model_stats_content = []
        if model_stats:
            for model, stats_data in sorted(model_stats.items()):
                model_stats_content.append(
                    html.Div(
                        [
                            html.Strong(
                                f"Model: {model}", style={"fontSize": "13px", "color": "#667eea"}
                            ),
                            html.Div(
                                [
                                    f"Calls: {stats_data['calls']} | "
                                    f"Tokens: {stats_data['tokens']:,} "
                                    f"(Input: {stats_data['input_tokens']:,}, Output: {stats_data['output_tokens']:,}) | "
                                    f"Avg Latency: {stats_data['avg_latency_ms']:.2f}ms | "
                                    f"Errors: {stats_data['errors']}"
                                ],
                                style={
                                    "fontSize": "12px",
                                    "color": "#555",
                                    "marginLeft": "8px",
                                    "marginTop": "4px",
                                },
                            ),
                        ],
                        style={"marginBottom": "8px"},
                    )
                )
        else:
            model_stats_content = [html.P("No LLM calls recorded yet.", style={"color": "#999"})]

        # Health status display
        health_colour_map = {
            "healthy": "#2ecc71",
            "warning": "#f39c12",
            "degraded": "#e74c3c",
        }
        health_colour = health_colour_map.get(health["status"], "#95a5a6")

        health_content = html.Div(
            [
                html.Div(
                    [
                        html.Strong("Status: ", style={"fontSize": "13px"}),
                        html.Span(
                            health["status"].upper(),
                            style={
                                "color": health_colour,
                                "fontWeight": "bold",
                                "fontSize": "14px",
                            },
                        ),
                    ],
                    style={"marginBottom": "8px"},
                ),
                html.Div(
                    f"Error Rate: {health['error_rate']:.2f}%",
                    style={"fontSize": "12px", "marginBottom": "4px"},
                ),
                html.Div(
                    f"Uptime: {health['uptime_hours']:.2f} hours",
                    style={"fontSize": "12px", "marginBottom": "4px"},
                ),
                html.Div(
                    f"Total Operations: {health['total_operations']}",
                    style={"fontSize": "12px", "marginBottom": "4px"},
                ),
                html.Div(
                    f"Cache Hit Rate: {health['cache_hit_rate']:.1f}%", style={"fontSize": "12px"}
                ),
            ]
        )

        # Cost display
        cost_content = html.Div(
            [
                html.H4(f"${cost:.4f}", style={"margin": "0", "color": "#667eea"}),
                html.P(
                    f"Estimated cost for {stats.total_tokens_used:,} tokens used",
                    style={"margin": "8px 0 0 0", "fontSize": "12px", "color": "#999"},
                ),
            ]
        )

        return summary_html, figure, model_stats_content, health_content, cost_content

    except Exception as e:
        logger.exception("Error updating metrics display")
        error_msg = html.Div(
            [
                html.P(f"Error loading metrics: {str(e)}", style={"color": "#e74c3c"}),
            ]
        )
        return error_msg, {}, error_msg, error_msg, error_msg


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    print(f"Starting Plotly Dash dashboard at http://localhost:8050")
    print(f"Graph: {GRAPH_SQLITE}")
    print(f"ChromaDB: {CHROMA_PATH}")
    print(f"Nodes per page: {NODES_PER_PAGE}")
    print("")
    print("Memory-efficient features:")
    print("  ✓ Lazy loading with pagination (50 nodes/page by default)")
    print("  ✓ Plotly rendering (GPU-accelerated, efficient)")
    print("  ✓ On-demand document loading (not all in memory)")
    print("  ✓ SQLite backend ready (future migration)")
    print("")

    # Initialise academic references module at startup
    try:
        terminology_db_path = Path(INGEST_CONFIG.rag_data_path) / "academic_terminology.db"
        if terminology_db_path.exists():
            print(f"Initialising AcademicReferences at startup...")
            module = AcademicReferences(str(terminology_db_path))
            _set_global_module(module)
            print("AcademicReferences initialised successfully")
        else:
            print("Academic terminology database not found, module will initialise on demand")
    except Exception as e:
        print(f"Warning: Could not initialise AcademicReferences: {e}")

    app.run(debug=True, host="0.0.0.0", port=8050, dev_tools_ui=True)
