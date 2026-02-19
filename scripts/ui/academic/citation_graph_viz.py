"""
Citation Graph Visualisation Component for Dash Dashboard.

Provides interactive visualisation of academic reference citation networks with:
- Node encoding: colour (link status), size (quality), border (type), shape (OA status)
- Persona-specific filtering (Supervisor, Assessor, Researcher)
- Link health indicators and venue prestige badges
- Hierarchical layout with depth-based positioning
- Interactive tooltips with metadata
- Persistent layout state (save user positions)

Usage:
    from scripts.ui.academic.citation_graph_viz import CitationGraphViz

    viz = CitationGraphViz(db_path="rag_data/academic_citations.db")
    layout = viz.create_dash_layout()

Integration with existing dashboard:
    # In dashboard.py, add a new tab:
    dcc.Tab(label="Citation Graph", value="citation-graph", children=[
        citation_graph_viz.create_dash_layout()
    ])

TODO: revisit legends and colour schemes of nodes.
TODO: add export functionality (CSV/JSON) for filtered citations with metadata and filter settings included in export.
TODO: Venue data not currently captured on ingestion.
"""

import json
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.utils.logger import create_module_logger

get_logger, audit = create_module_logger("consistency_graph.academic.citation_graph_viz")


@dataclass
class NodeAttributes:
    """Visual attributes for a citation graph node."""

    color: str
    size: float
    border_color: str
    border_width: float
    shape: str  # 'circle', 'square', 'diamond'
    opacity: float
    label: str
    tooltip: str


@dataclass
class PersonaFilter:
    """Filtering configuration for a specific persona."""

    quality_threshold: float
    include_stale: bool
    venue_types: List[str]
    require_doi: bool = False
    max_depth: int = 3


class CitationGraphViz:
    """
    Citation graph visualisation with persona-specific filtering and encoding.
    """

    # Persona filter configurations
    PERSONA_FILTERS = {
        "supervisor": PersonaFilter(
            quality_threshold=0.6,
            include_stale=True,  # Historical context
            venue_types=["journal", "conference", "preprint"],
            max_depth=1,
        ),
        "assessor": PersonaFilter(
            quality_threshold=0.7,
            include_stale=False,  # Require verifiable sources
            venue_types=["journal", "conference"],
            require_doi=True,
            max_depth=2,
        ),
        "researcher": PersonaFilter(
            quality_threshold=0.5,
            include_stale=True,
            venue_types=["journal", "conference", "preprint", "report"],
            max_depth=3,
        ),
    }

    # Link status colours
    LINK_STATUS_COLOURS = {
        "available": "#10b981",  # green
        "stale_404": "#ef4444",  # red
        "stale_timeout": "#f59e0b",  # orange
        "stale_moved": "#eab308",  # gold
        "unresolved": "#6b7280",  # gray
    }

    # Reference type border styles
    REFERENCE_TYPE_STYLES = {
        "academic": {"width": 2, "dash": "solid"},
        "preprint": {"width": 2, "dash": "dash"},
        "news": {"width": 1, "dash": "dot"},
        "blog": {"width": 1, "dash": "dot"},
        "online": {"width": 1, "dash": "dashdot"},
        "report": {"width": 2, "dash": "dash"},
    }

    # Verification source trustworthiness colours (most to least trustworthy)
    SOURCE_TRUSTWORTHINESS_COLOURS = {
        "crossref": "#10b981",  # green - most trustworthy (metadata APIs)
        "arxiv": "#3b82f6",  # blue - trustworthy (indexed preprints)
        "url_fetch": "#f59e0b",  # amber - moderate (fetched from web)
        "document": "#ef4444",  # red - low confidence (manually extracted)
        "unresolved": "#6b7280",  # gray - unresolved/unknown
    }

    def __init__(self, db_path: Path | str = None):
        """Initialise citation graph visualiser.

        Args:
            db_path: Optional path to SQLite database containing citation graph data.
                     If None, defaults to 'rag_data/academic_citation_graph.db' in project root.
        """
        if db_path is None:
            # Use absolute path to project root
            # __file__ = scripts/consistency_graph/academic/citation_graph_viz.py
            # Need to go up 4 levels to reach project root
            project_root = Path(__file__).parent.parent.parent.parent
            db_path = project_root / "rag_data" / "academic_citation_graph.db"
        self.db_path = Path(db_path)
        self.logger = get_logger()
        self._local = threading.local()  # Thread-local storage
        self._connections: set[sqlite3.Connection] = set()
        self._conn_lock = threading.Lock()
        self._graph = None
        self._primary_docs = set()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create thread-local database connection."""
        # Use thread-local storage to avoid sharing connections across threads
        if not hasattr(self._local, "conn") or self._local.conn is None:
            if not self.db_path.exists():
                self.logger.warning(f"Database not found: {self.db_path}")
                self.logger.warning(f"Looking for: {self.db_path.absolute()}")
                return None
            self.logger.info(f"Connecting to: {self.db_path} (thread {threading.get_ident()})")
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            with self._conn_lock:
                self._connections.add(conn)
            self._local.conn = conn
        return self._local.conn

    def load_graph(
        self,
        persona: str = "supervisor",
        doc_ids: Optional[List[str]] = None,
        link_statuses: Optional[List[str]] = None,
        reference_types: Optional[List[str]] = None,
        venue_types: Optional[List[str]] = None,
        venue_ranks: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ) -> nx.DiGraph:
        """
        Load citation graph from database with persona-specific filtering.

        Args:
            persona: Query persona (supervisor, assessor, researcher)
            doc_ids: Optional list of document IDs to focus on (None = all)
            link_statuses: Filter by link status (available, stale_404, etc.)
            reference_types: Filter by reference type (academic, preprint, etc.)
            venue_types: Filter by venue type (journal, conference, etc.)
            venue_ranks: Filter by venue rank (Q1-Q4, A*-C)
            sources: Filter by verification source (arxiv, crossref, url_fetch, unresolved, document)

        Returns:
            NetworkX directed graph
        """
        conn = self._get_connection()
        if not conn:
            self.logger.error(f"Failed to connect to database: {self.db_path}")
            return nx.DiGraph()

        filter_config = self.PERSONA_FILTERS.get(persona, self.PERSONA_FILTERS["supervisor"])

        # Build SQL WHERE clause for filters
        # Handle NULL values gracefully - if a column is NULL, include it in results
        where_clauses = []
        params = []

        if link_statuses:
            placeholders = ",".join("?" * len(link_statuses))
            where_clauses.append(f"(link_status IS NULL OR link_status IN ({placeholders}))")
            params.extend(link_statuses)

        if reference_types:
            placeholders = ",".join("?" * len(reference_types))
            where_clauses.append(f"(reference_type IS NULL OR reference_type IN ({placeholders}))")
            params.extend(reference_types)

        if venue_types:
            placeholders = ",".join("?" * len(venue_types))
            where_clauses.append(f"(venue_type IS NULL OR venue_type IN ({placeholders}))")
            params.extend(venue_types)

        if venue_ranks:
            placeholders = ",".join("?" * len(venue_ranks))
            where_clauses.append(f"(venue_rank IS NULL OR venue_rank IN ({placeholders}))")
            params.extend(venue_ranks)

        if sources:
            placeholders = ",".join("?" * len(sources))
            where_clauses.append(f"(source IS NULL OR source IN ({placeholders}))")
            params.extend(sources)

        where_sql = ""
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)

        # Build graph
        G = nx.DiGraph()

        # Load all nodes from the nodes table with filters
        # TODO: Consider loading only relevant subgraph based on doc_ids and max_depth for performance on large datasets
        # TODO: Add pagination or lazy loading for very large graphs to avoid memory issues
        # TODO: consider caching the graph in memory and only re-querying for updates or specific subgraphs to improve performance on repeated access.
        # TODO: consider defaults if values are not found
        try:
            query = f"SELECT * FROM nodes{where_sql}"
            cursor = conn.execute(query, params)
            node_count = 0
            error_count = 0

            for row in cursor:
                try:
                    node_count += 1
                    node_id = row["node_id"]

                    try:
                        node_type = row["node_type"] if row["node_type"] else "reference"
                    except (IndexError, TypeError):
                        node_type = "reference"

                    # Track primary documents (node_type = 'document')
                    if node_type == "document":
                        self._primary_docs.add(node_id)

                    try:
                        title = row["title"] if row["title"] else ""
                    except (IndexError, TypeError):
                        title = ""

                    try:
                        authors_raw = row["authors"] if row["authors"] else ""
                        # Parse JSON authors field
                        if authors_raw:
                            try:
                                authors = json.loads(authors_raw)
                            except (json.JSONDecodeError, TypeError):
                                # Fallback to raw string if not valid JSON
                                authors = authors_raw
                        else:
                            authors = ""
                    except (IndexError, TypeError):
                        authors = ""

                    try:
                        year = row["year"]
                    except (IndexError, TypeError):
                        year = None

                    try:
                        doi = row["doi"]
                    except (IndexError, TypeError):
                        doi = None

                    try:
                        reference_type = (
                            row["reference_type"] if row["reference_type"] else "academic"
                        )
                    except (IndexError, TypeError):
                        reference_type = "academic"

                    try:
                        source = row["source"] if row["source"] else ""
                    except (IndexError, TypeError):
                        source = ""

                    try:
                        confidence = row["confidence"]
                    except (IndexError, TypeError):
                        confidence = None

                    try:
                        quality_score = (
                            row["quality_score"] if row["quality_score"] is not None else None
                        )
                    except (IndexError, TypeError, KeyError):
                        quality_score = None

                    try:
                        link_status = row["link_status"] if row["link_status"] else "available"
                    except (IndexError, TypeError, KeyError):
                        link_status = "available"

                    G.add_node(
                        node_id,
                        title=title,
                        authors=authors,
                        year=year,
                        doi=doi,
                        node_type=node_type,
                        quality_score=quality_score,
                        link_status=link_status,
                        reference_type=reference_type,
                        source=source,
                        confidence=confidence,
                    )

                except Exception as node_err:
                    error_count += 1
                    if error_count <= 5:  # Log first 5 errors
                        self.logger.debug(f"Error loading node {node_count}: {node_err}")

        except Exception as e:
            self.logger.error(f"Error loading nodes: {e}", exc_info=True)
            return nx.DiGraph()

        self.logger.info(f"Loaded {node_count} nodes ({error_count} errors)")

        # Load edges
        cursor = conn.execute("SELECT source, target FROM edges")

        for row in cursor:
            source_id = row["source"]
            target_id = row["target"]

            # Only add edge if both nodes exist
            if G.has_node(source_id) and G.has_node(target_id):
                G.add_edge(
                    source_id,
                    target_id,
                    relationship_type="cites",
                    strength_score=1.0,
                )

        self._graph = G
        self.logger.info(
            f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
            f"(persona={persona})"
        )

        return G

    def compute_node_attributes(self, node_id: str) -> NodeAttributes:
        """
        Compute visual attributes for a node based on metadata.

        Args:
            node_id: Node identifier

        Returns:
            NodeAttributes with colour, size, border, shape, etc.
        """
        if not self._graph or not self._graph.has_node(node_id):
            # Default attributes for missing nodes (updated based on citation attributes below)
            return NodeAttributes(
                color="#gray",
                size=10,
                border_color="#gray",
                border_width=1,
                shape="circle",
                opacity=0.5,
                label="Unknown",
                tooltip="No data",
            )

        node_data = self._graph.nodes[node_id]

        # Primary documents get special treatment
        if node_id in self._primary_docs:
            return NodeAttributes(
                color="#9333ea",  # purple
                size=30,
                border_color="#6b21a8",
                border_width=3,
                shape="diamond",
                opacity=1.0,
                label=node_data.get("title", "Primary Document")[:50],
                tooltip=self._build_tooltip(node_id, node_data),
            )

        # colour by verification source trustworthiness
        source = node_data.get("source", "unresolved")
        colour = self.SOURCE_TRUSTWORTHINESS_COLOURS.get(source, "#6b7280")

        # Size by quality score (10-40px)
        quality_score = node_data.get("quality_score")
        if quality_score is None:
            node_type = node_data.get("node_type", "reference")
            quality_score = 1.0 if node_type == "document" else 0.5
        size = 10 + (quality_score * 30)

        # Border by reference type
        ref_type = node_data.get("reference_type", "academic")
        border_style = self.REFERENCE_TYPE_STYLES.get(ref_type, {"width": 1, "dash": "solid"})

        # Border colour: darken node colour for visibility
        # Convert hex to RGB, darken by 30%, convert back to hex
        rgb = tuple(int(colour.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        darkened_rgb = tuple(int(c * 0.7) for c in rgb)
        border_colour = f"#{darkened_rgb[0]:02x}{darkened_rgb[1]:02x}{darkened_rgb[2]:02x}"

        # Link status for opacity
        link_status = node_data.get("link_status", "available")

        # Shape by OA status
        # TODO: Add OA tracking logic during ingestion to populate oa_available field
        oa_available = node_data.get("oa_available", False)
        shape = "circle" if oa_available else "square"

        # Opacity by staleness (reduce if link status not checked recently)
        opacity = 1.0 if link_status == "available" else 0.7

        # Label
        title = node_data.get("title", "Untitled")
        authors = node_data.get("authors", "")
        if authors:
            if isinstance(authors, str):
                first_author = authors.split(",")[0].strip()
            else:
                first_author = str(authors[0]).strip() if authors else ""
            label = f"{first_author} ({node_data.get('year', '?')})"
        else:
            label = title[:30]

        return NodeAttributes(
            color=colour,
            size=size,
            border_color=border_colour,
            border_width=border_style["width"],
            shape=shape,
            opacity=opacity,
            label=label,
            tooltip=self._build_tooltip(node_id, node_data),
        )

    def _build_tooltip(self, node_id: str, node_data: Dict[str, Any]) -> str:
        """Build rich HTML tooltip for node.

        Args:
            node_id: Node identifier
            node_data: Node metadata dictionary from graph

        Returns:
            HTML string for tooltip content

        """
        title = node_data.get("title", "Untitled")
        authors_raw = node_data.get("authors", "Unknown")

        # Handle authors as list or string
        if isinstance(authors_raw, list):
            authors = ", ".join(str(a).strip() for a in authors_raw) if authors_raw else "Unknown"
        else:
            authors = str(authors_raw) if authors_raw else "Unknown"

        year = node_data.get("year", "?")

        # Primary document badge
        badge_html = ""
        if node_id in self._primary_docs:
            badge_html = "<span style='background:#9333ea;color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>PRIMARY</span>"

        # Venue rank badge (if available - currently mostly unused in dataset)
        venue_rank = node_data.get("venue_rank")
        if venue_rank:
            # Fallback colour mapping for venue rank if present
            rank_colours = {
                "Q1": "#10b981",
                "A*": "#10b981",
                "Q2": "#3b82f6",
                "A": "#3b82f6",
                "Q3": "#f59e0b",
                "B": "#f59e0b",
                "Q4": "#ef4444",
                "C": "#ef4444",
            }
            rank_colour = rank_colours.get(venue_rank, "#gray")
            badge_html += f" <span style='background:{rank_colour};color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>{venue_rank}</span>"

        # Verification source badge
        source = node_data.get("source")
        if source:
            source_colour = self.SOURCE_TRUSTWORTHINESS_COLOURS.get(source, "#6b7280")
            source_labels = {
                "crossref": "CrossRef",
                "arxiv": "ArXiv",
                "url_fetch": "URL Fetch",
                "document": "Document",
                "unresolved": "Unresolved",
            }
            source_label = source_labels.get(source, source.title())
            badge_html += f" <span style='background:{source_colour};color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>{source_label}</span>"

        # Link status warning
        link_status = node_data.get("link_status", "available")
        status_icon = ""
        if link_status != "available":
            status_icon = f"<br>⚠️ <em>Link {link_status.replace('_', ' ')}</em>"

        # Content changed warning
        if node_data.get("content_changed"):
            status_icon += "<br>🔄 <em>Content updated</em>"

        # Build tooltip
        confidence = node_data.get("confidence")
        confidence_line = ""
        if confidence is not None:
            confidence_line = f"Confidence: {confidence:.2f}<br>"

        quality_score = node_data.get("quality_score")
        if quality_score is None:
            node_type = node_data.get("node_type", "reference")
            quality_score = 1.0 if node_type == "document" else 0.5

        tooltip = f"""
        <b>{title}</b><br>
        {authors}<br>
        <em>{year}</em><br>
        {badge_html}<br>
        Quality: {quality_score:.2f}<br>
        {confidence_line}Citations: {node_data.get('citation_count', 0)}<br>
        Type: {node_data.get('reference_type', 'academic')}<br>
        {f"DOI: {node_data['doi']}<br>" if node_data.get('doi') else ''}
        {status_icon}
        """

        return tooltip.strip()

    def export_citations(
        self,
        persona: str = "supervisor",
        link_statuses: Optional[List[str]] = None,
        reference_types: Optional[List[str]] = None,
        venue_types: Optional[List[str]] = None,
        venue_ranks: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ) -> str:
        """
        Export filtered citations to CSV format with metadata.

        Args:
            persona: Selected persona (supervisor, assessor, researcher)
            link_statuses: Filter by link status
            reference_types: Filter by reference type
            venue_types: Filter by venue type
            venue_ranks: Filter by venue rank
            sources: Filter by verification source (arxiv, crossref, url_fetch, unresolved, document)

        Returns:
            CSV string with citations and filter metadata
        """
        import csv
        import io

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)

        # Write metadata header
        writer.writerow(["# CITATION EXPORT METADATA"])
        writer.writerow(["Export Date", datetime.now().isoformat()])
        writer.writerow(["Persona", persona])
        writer.writerow(
            ["Link Status Filter", ", ".join(link_statuses) if link_statuses else "All"]
        )
        writer.writerow(
            ["Reference Type Filter", ", ".join(reference_types) if reference_types else "All"]
        )
        writer.writerow(["Venue Type Filter", ", ".join(venue_types) if venue_types else "All"])
        writer.writerow(["Venue Rank Filter", ", ".join(venue_ranks) if venue_ranks else "All"])
        writer.writerow(["Verification Source Filter", ", ".join(sources) if sources else "All"])
        writer.writerow([])  # Blank row separator

        # Write data header
        writer.writerow(
            [
                "node_id",
                "title",
                "authors",
                "year",
                "doi",
                "reference_type",
                "venue_type",
                "venue_rank",
                "link_status",
                "source",
                "confidence",
                "node_type",
            ]
        )

        # Query database directly for filtered nodes
        try:
            with closing(sqlite3.connect(str(self.db_path), check_same_thread=False)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Build filter query - only include filters with actual data
                where_clauses = []
                params = []

                # Always filter by reference type if specified
                if reference_types:
                    placeholders = ",".join("?" * len(reference_types))
                    where_clauses.append(f"reference_type IN ({placeholders})")
                    params.extend(reference_types)

                # Check if venue_type has non-NULL data before filtering
                if venue_types:
                    # Only apply venue_type filter if it has non-NULL data in matching records
                    check_venue = cursor.execute(
                        "SELECT COUNT(*) FROM nodes WHERE venue_type IS NOT NULL"
                    ).fetchone()[0]
                    if check_venue > 0:
                        placeholders = ",".join("?" * len(venue_types))
                        where_clauses.append(f"venue_type IN ({placeholders})")
                        params.extend(venue_types)

                # Check if venue_rank has non-NULL data before filtering
                if venue_ranks:
                    check_rank = cursor.execute(
                        "SELECT COUNT(*) FROM nodes WHERE venue_rank IS NOT NULL"
                    ).fetchone()[0]
                    if check_rank > 0:
                        placeholders = ",".join("?" * len(venue_ranks))
                        where_clauses.append(f"venue_rank IN ({placeholders})")
                        params.extend(venue_ranks)

                # Check if link_status has non-NULL data before filtering
                if link_statuses:
                    check_link = cursor.execute(
                        "SELECT COUNT(*) FROM nodes WHERE link_status IS NOT NULL"
                    ).fetchone()[0]
                    if check_link > 0:
                        placeholders = ",".join("?" * len(link_statuses))
                        where_clauses.append(f"link_status IN ({placeholders})")
                        params.extend(link_statuses)

                # Check if sources has non-NULL data before filtering
                if sources:
                    check_sources = cursor.execute(
                        "SELECT COUNT(*) FROM nodes WHERE source IS NOT NULL"
                    ).fetchone()[0]
                    if check_sources > 0:
                        placeholders = ",".join("?" * len(sources))
                        where_clauses.append(f"source IN ({placeholders})")
                        params.extend(sources)

                where_sql = ""
                if where_clauses:
                    where_sql = " WHERE " + " AND ".join(where_clauses)

                # Fetch and write nodes
                query = f"SELECT * FROM nodes{where_sql} ORDER BY year DESC, title ASC"
                cursor.execute(query, params)

                row_count = 0
                for row in cursor:
                    writer.writerow(
                        [
                            row["node_id"],
                            row["title"] or "",
                            row["authors"] or "",
                            row["year"] or "",
                            row["doi"] or "",
                            row["reference_type"] or "",
                            row["venue_type"] or "",
                            row["venue_rank"] or "",
                            row["link_status"] or "",
                            row["source"] or "",
                            f"{row['confidence']:.2f}" if row["confidence"] is not None else "",
                            row["node_type"] or "reference",
                        ]
                    )
                    row_count += 1

            # Add summary
            csv_buffer.write(f"\n# Total citations exported: {row_count}")

            self.logger.info(
                f"Exported {row_count} citations with filters: persona={persona}, sources={sources}"
            )

        except Exception as e:
            self.logger.error(f"Error exporting citations: {e}", exc_info=True)
            writer.writerow(["ERROR", str(e)])

        return csv_buffer.getvalue()

    def create_plotly_figure(
        self,
        layout: str = "hierarchical",
        width: int = 1200,
        height: int = 800,
    ) -> go.Figure:
        """
        Create Plotly figure for citation graph.

        Args:
            layout: Layout algorithm (hierarchical, force, circular)
            width: Figure width in pixels
            height: Figure height in pixels

        Returns:
            Plotly Figure object
        """
        if not self._graph or self._graph.number_of_nodes() == 0:
            # Empty state
            fig = go.Figure()
            fig.add_annotation(
                text="No citations to display. Run ingestion first.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray"),
            )
            fig.update_layout(
                width=width,
                height=height,
                template="plotly_white",
            )
            return fig

        # Compute layout positions
        if layout == "hierarchical":
            pos = nx.spring_layout(self._graph, k=1.5, iterations=50, seed=42)
            # Adjust Y based on depth for hierarchical view
            for node in self._graph.nodes():
                # Primary docs at top
                if node in self._primary_docs:
                    pos[node] = (pos[node][0], 1.0)
                else:
                    # Use incoming edge depth if available
                    depths = [
                        self._graph.edges[u, node].get("depth", 1)
                        for u in self._graph.predecessors(node)
                    ]
                    avg_depth = sum(depths) / len(depths) if depths else 1
                    pos[node] = (pos[node][0], 1.0 - (avg_depth * 0.3))
        elif layout == "circular":
            pos = nx.circular_layout(self._graph)
        else:  # force-directed
            pos = nx.spring_layout(self._graph, k=1.0, iterations=50, seed=42)

        # Build edge traces grouped by link status
        edge_traces = {}

        for edge in self._graph.edges():
            source_node, target_node = edge
            # Use target node's link status to colour the edge
            target_data = self._graph.nodes[target_node]
            link_status = target_data.get("link_status", "available")

            # Initialise trace for this link status if not exists
            if link_status not in edge_traces:
                edge_traces[link_status] = {
                    "x": [],
                    "y": [],
                }

            x0, y0 = pos[source_node]
            x1, y1 = pos[target_node]
            edge_traces[link_status]["x"].extend([x0, x1, None])
            edge_traces[link_status]["y"].extend([y0, y1, None])

        # Build node traces (separate by source for legend)
        node_traces = {}

        # Human-readable labels for legend
        source_labels = {
            "crossref": "CrossRef (verified)",
            "arxiv": "ArXiv (preprint)",
            "url_fetch": "Web fetched",
            "document": "Manual extraction",
            "unresolved": "Unresolved",
        }

        for node in self._graph.nodes():
            attrs = self.compute_node_attributes(node)

            # Group by source for legend (primary docs get special treatment)
            if node in self._primary_docs:
                legend_label = "Primary Document"
            else:
                node_data = self._graph.nodes[node]
                source = node_data.get("source", "unresolved")
                legend_label = source_labels.get(source, source.title())

            if legend_label not in node_traces:
                node_traces[legend_label] = {
                    "x": [],
                    "y": [],
                    "sizes": [],
                    "colours": [],
                    "symbols": [],
                    "customdata": [],
                }

            x, y = pos[node]
            node_traces[legend_label]["x"].append(x)
            node_traces[legend_label]["y"].append(y)
            node_traces[legend_label]["sizes"].append(attrs.size)
            node_traces[legend_label]["colours"].append(attrs.color)
            node_traces[legend_label]["symbols"].append(attrs.shape)
            node_traces[legend_label]["customdata"].append(node)

        # Create figure
        fig = go.Figure()

        # Add edge traces coloured by link status
        for link_status, trace_data in edge_traces.items():
            edge_colour = self.LINK_STATUS_COLOURS.get(link_status, "#cbd5e1")
            fig.add_trace(
                go.Scatter(
                    x=trace_data["x"],
                    y=trace_data["y"],
                    mode="lines",
                    line=dict(width=0.5, color=edge_colour),
                    hoverinfo="none",
                    showlegend=False,
                    name=f"Link: {link_status}",
                )
            )

        # Add node traces with rich hover text
        for legend_label, trace_data in node_traces.items():
            # Create rich hover text
            hover_texts = []
            for node_id in trace_data["customdata"]:
                node_data = self._graph.nodes[node_id]
                hover_texts.append(self._build_tooltip(node_id, node_data))

            # Ensure hover_texts length matches data length
            if len(hover_texts) != len(trace_data["x"]):
                self.logger.warning(
                    f"Hover text mismatch for {legend_label}: {len(hover_texts)} vs {len(trace_data['x'])}"
                )
                hover_texts = [f"Node {i}" for i in range(len(trace_data["x"]))]

            # Set marker symbol (use first symbol since all nodes in same trace should have same symbol)
            marker_symbol = trace_data["symbols"][0] if trace_data["symbols"] else "circle"

            fig.add_trace(
                go.Scatter(
                    x=trace_data["x"],
                    y=trace_data["y"],
                    mode="markers",
                    name=legend_label,
                    hovertext=hover_texts,
                    hoverinfo="text",
                    marker=dict(
                        size=trace_data["sizes"],
                        color=trace_data["colours"],
                        symbol=marker_symbol,
                    ),
                    customdata=trace_data["customdata"],
                    showlegend=True,
                )
            )

        # Update layout
        fig.update_layout(
            title=dict(text="Citation Graph", font=dict(size=16)),
            showlegend=True,
            hovermode="closest",
            width=width,
            height=height,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
            plot_bgcolor="#f8fafc",
        )

        return fig

    def create_dash_layout(self):
        """
        Create Dash layout components for citation graph tab.

        Returns:
            Dash HTML layout
        """
        from dash import dcc, html

        return html.Div(
            [
                html.Div(
                    [
                        html.H3("Citation Graph", style={"margin": "0 0 8px 0"}),
                        html.P(
                            "Interactive visualisation of academic reference networks",
                            style={"color": "#64748b", "margin": 0},
                        ),
                    ],
                    className="card",
                ),
                # Filters
                html.Div(
                    [
                        html.H4("Filters", style={"margin": "0 0 12px 0"}),
                        # Row 1: Persona and Layout
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Persona:"),
                                        dcc.Dropdown(
                                            id="citation-persona-dropdown",
                                            options=[
                                                {"label": "Supervisor", "value": "supervisor"},
                                                {"label": "Assessor", "value": "assessor"},
                                                {"label": "Researcher", "value": "researcher"},
                                            ],
                                            value="supervisor",
                                            clearable=False,
                                            style={"width": "200px"},
                                        ),
                                    ],
                                    style={"display": "inline-block", "marginRight": "20px"},
                                ),
                                html.Div(
                                    [
                                        html.Label("Layout:"),
                                        dcc.Dropdown(
                                            id="citation-layout-dropdown",
                                            options=[
                                                {"label": "Hierarchical", "value": "hierarchical"},
                                                {"label": "Force-Directed", "value": "force"},
                                                {"label": "Circular", "value": "circular"},
                                            ],
                                            value="hierarchical",
                                            clearable=False,
                                            style={"width": "200px"},
                                        ),
                                    ],
                                    style={"display": "inline-block", "marginRight": "20px"},
                                ),
                            ],
                            style={"marginBottom": "16px"},
                        ),
                        # Row 2: Data filters
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Link Status:"),
                                        dcc.Dropdown(
                                            id="citation-link-status-dropdown",
                                            options=[
                                                {"label": "Available", "value": "available"},
                                                {"label": "Stale (404)", "value": "stale_404"},
                                                {
                                                    "label": "Stale (Timeout)",
                                                    "value": "stale_timeout",
                                                },
                                                {"label": "Stale (Moved)", "value": "stale_moved"},
                                            ],
                                            value=[],
                                            multi=True,
                                            placeholder="All statuses",
                                            style={"width": "250px"},
                                        ),
                                    ],
                                    style={"display": "inline-block", "marginRight": "20px"},
                                ),
                                html.Div(
                                    [
                                        html.Label("Reference Type:"),
                                        dcc.Dropdown(
                                            id="citation-reference-type-dropdown",
                                            options=[
                                                {"label": "Academic", "value": "academic"},
                                                {"label": "Preprint", "value": "preprint"},
                                                {"label": "News", "value": "news"},
                                                {"label": "Blog", "value": "blog"},
                                                {"label": "Online", "value": "online"},
                                                {"label": "Report", "value": "report"},
                                            ],
                                            value=["academic", "preprint"],
                                            multi=True,
                                            placeholder="All types",
                                            style={"width": "250px"},
                                        ),
                                    ],
                                    style={"display": "inline-block", "marginRight": "20px"},
                                ),
                                html.Div(
                                    [
                                        html.Label("Verification Source:"),
                                        dcc.Dropdown(
                                            id="citation-source-dropdown",
                                            options=[
                                                {"label": "ArXiv", "value": "arxiv"},
                                                {"label": "CrossRef", "value": "crossref"},
                                                {"label": "URL Fetch", "value": "url_fetch"},
                                                {"label": "Unresolved", "value": "unresolved"},
                                                {"label": "Document", "value": "document"},
                                            ],
                                            value=[],
                                            multi=True,
                                            placeholder="All sources",
                                            style={"width": "250px"},
                                        ),
                                    ],
                                    style={"display": "inline-block", "marginRight": "20px"},
                                ),
                            ],
                            style={"marginBottom": "16px"},
                        ),
                        # Row 3: Venue filters
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Venue Type:"),
                                        dcc.Dropdown(
                                            id="citation-venue-type-dropdown",
                                            options=[
                                                {"label": "Journal", "value": "journal"},
                                                {"label": "Conference", "value": "conference"},
                                                {"label": "Preprint", "value": "preprint"},
                                                {"label": "Web", "value": "web"},
                                            ],
                                            value=[],
                                            multi=True,
                                            placeholder="All venues",
                                            style={"width": "250px"},
                                        ),
                                    ],
                                    style={"display": "inline-block", "marginRight": "20px"},
                                ),
                                html.Div(
                                    [
                                        html.Label("Venue Rank:"),
                                        dcc.Dropdown(
                                            id="citation-venue-rank-dropdown",
                                            options=[
                                                {"label": "Q1", "value": "Q1"},
                                                {"label": "Q2", "value": "Q2"},
                                                {"label": "Q3", "value": "Q3"},
                                                {"label": "Q4", "value": "Q4"},
                                                {"label": "A*", "value": "A*"},
                                                {"label": "A", "value": "A"},
                                                {"label": "B", "value": "B"},
                                                {"label": "C", "value": "C"},
                                            ],
                                            value=[],
                                            multi=True,
                                            placeholder="All ranks",
                                            style={"width": "250px"},
                                        ),
                                    ],
                                    style={"display": "inline-block", "marginRight": "20px"},
                                ),
                                html.Div(
                                    [
                                        html.Button(
                                            "Apply Filters",
                                            id="citation-reload-button",
                                            n_clicks=0,
                                            style={
                                                "padding": "8px 16px",
                                                "background": "#667eea",
                                                "color": "white",
                                                "border": "none",
                                                "borderRadius": "4px",
                                                "cursor": "pointer",
                                                "marginTop": "20px",
                                            },
                                        ),
                                    ],
                                    style={"display": "inline-block", "marginRight": "10px"},
                                ),
                                html.Div(
                                    [
                                        html.Button(
                                            "📥 Export Citations",
                                            id="citation-export-button",
                                            n_clicks=0,
                                            style={
                                                "padding": "8px 16px",
                                                "background": "#059669",
                                                "color": "white",
                                                "border": "none",
                                                "borderRadius": "4px",
                                                "cursor": "pointer",
                                                "marginTop": "20px",
                                            },
                                        ),
                                        dcc.Download(id="citation-export-download"),
                                    ],
                                    style={"display": "inline-block"},
                                ),
                            ],
                            style={"marginBottom": "16px"},
                        ),
                    ],
                    className="card",
                ),
                # Graph
                html.Div(
                    [
                        dcc.Loading(
                            id="citation-graph-loading",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id="citation-graph",
                                    config={"displayModeBar": True, "displaylogo": False},
                                )
                            ],
                        ),
                    ],
                    className="card",
                ),
                # Selected node details
                html.Div(
                    [
                        html.H4("Reference Details", style={"margin": "0 0 12px 0"}),
                        html.Div(
                            id="citation-node-details",
                            children=[
                                html.P("Click a node to see details", style={"color": "#64748b"})
                            ],
                        ),
                    ],
                    className="card",
                ),
            ],
            style={"padding": "0"},
        )

    def close(self):
        """Close database connection."""
        with self._conn_lock:
            for conn in list(self._connections):
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()

        if hasattr(self._local, "conn"):
            self._local.conn = None

    def __del__(self):
        """Best-effort cleanup to avoid leaked SQLite connections."""
        try:
            self.close()
        except Exception:
            pass


# Singleton instance for dashboard integration
_viz_instance = None


def get_citation_viz() -> CitationGraphViz:
    """Get or create singleton citation graph visualiser."""
    global _viz_instance
    if _viz_instance is None:
        _viz_instance = CitationGraphViz()
    return _viz_instance
