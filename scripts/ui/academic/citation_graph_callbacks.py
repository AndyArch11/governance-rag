"""
Dash Callbacks for Citation Graph Visualisation.

Defines callback functions for the citation graph tab in the dashboard.
These should be imported and registered in the main dashboard.py file.
"""

from dash import Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate

from scripts.ui.academic.citation_graph_viz import get_citation_viz
from scripts.utils.logger import create_module_logger

get_logger, audit = create_module_logger("consistency_graph.academic.citation_graph_callbacks")


@callback(
    [
        Output("citation-venue-type-dropdown", "value"),
    ],
    Input("citation-persona-dropdown", "value"),
    prevent_initial_call=False,
)
def update_filters_from_persona(persona):
    """
    Update filter dropdowns based on selected persona.

    Args:
        persona: Selected persona (supervisor, assessor, researcher)

    Returns:
        Updated filter values from persona config
    """
    logger = get_logger()

    try:
        viz = get_citation_viz()

        # Get persona filter config
        filter_config = viz.PERSONA_FILTERS.get(persona, viz.PERSONA_FILTERS["supervisor"])

        # Update venue_types filter to match persona
        venue_types = (
            filter_config.venue_types
            if filter_config.venue_types
            else ["journal", "conference", "preprint", "web"]
        )

        logger.info(f"Persona '{persona}' selected: venue_types={venue_types}")

        return [venue_types]

    except Exception as e:
        logger.error(f"Error updating filters from persona: {e}")
        return [no_update]


@callback(
    Output("citation-graph", "figure"),
    [
        Input("citation-reload-button", "n_clicks"),
        Input("citation-persona-dropdown", "value"),
        Input("citation-layout-dropdown", "value"),
        Input("citation-link-status-dropdown", "value"),
        Input("citation-reference-type-dropdown", "value"),
        Input("citation-venue-type-dropdown", "value"),
        Input("citation-venue-rank-dropdown", "value"),
        Input("citation-source-dropdown", "value"),
    ],
    prevent_initial_call=False,
)
def update_citation_graph(
    n_clicks,
    persona,
    layout,
    link_statuses,
    reference_types,
    venue_types,
    venue_ranks,
    sources,
):
    """
    Update citation graph based on filters.

    Args:
        n_clicks: Reload button clicks
        persona: Selected persona (supervisor, assessor, researcher)
        layout: Layout algorithm (hierarchical, force, circular)
        link_statuses: Filter by link status (available, stale_404, etc.)
        reference_types: Filter by reference type (academic, preprint, etc.)
        venue_types: Filter by venue type (journal, conference, etc.)
        venue_ranks: Filter by venue rank (Q1-Q4, A*-C)
        sources: Filter by verification source (arxiv, crossref, url_fetch, unresolved, document)

    Returns:
        Plotly figure for citation graph
    """
    logger = get_logger()

    try:
        viz = get_citation_viz()

        # Load graph with all filters
        graph = viz.load_graph(
            persona=persona or "supervisor",
            link_statuses=link_statuses if link_statuses else None,
            reference_types=reference_types if reference_types else None,
            venue_types=venue_types if venue_types else None,
            venue_ranks=venue_ranks if venue_ranks else None,
            sources=sources if sources else None,
        )

        # Create visualisation
        fig = viz.create_plotly_figure(
            layout=layout or "hierarchical",
            width=1200,
            height=800,
        )

        logger.info(
            f"Citation graph updated: persona={persona}, layout={layout}, "
            f"nodes={graph.number_of_nodes()}, "
            f"filters=(link={link_statuses}, ref={reference_types}, "
            f"venue={venue_types}, rank={venue_ranks}, source={sources})"
        )

        return fig

    except Exception as e:
        logger.error(f"Error updating citation graph: {e}")

        # Return error figure
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading citation graph: {str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        fig.update_layout(template="plotly_white")
        return fig


@callback(
    Output("citation-node-details", "children"),
    [Input("citation-graph", "clickData")],
    prevent_initial_call=True,
)
def display_node_details(clickData):
    """
    Display detailed information for clicked node.

    Args:
        clickData: Plotly click event data

    Returns:
        Dash HTML components with node details
    """
    from dash import html

    if not clickData or "points" not in clickData:
        raise PreventUpdate

    try:
        # Extract node ID from customdata
        point = clickData["points"][0]
        node_id = point.get("customdata")

        if not node_id:
            raise PreventUpdate

        viz = get_citation_viz()

        if not viz._graph or not viz._graph.has_node(node_id):
            return html.P("Node data not available", style={"color": "#ef4444"})

        node_data = viz._graph.nodes[node_id]

        # Build details panel
        details = [
            html.H5(node_data.get("title", "Untitled"), style={"marginTop": 0}),
        ]

        # Authors
        authors = node_data.get("authors", "")
        if authors:
            if isinstance(authors, str):
                author_list_full = authors.split(",")
                author_list = [a.strip() for a in author_list_full[:5]]  # First 5 authors, stripped
                has_more = len(author_list_full) > 5
            else:
                author_list = [str(a).strip() for a in authors[:5]]  # Convert to string and strip
                has_more = len(authors) > 5
            details.append(
                html.P(
                    [
                        html.Strong("Authors: "),
                        ", ".join(author_list),
                        ", et al." if has_more else "",
                    ]
                )
            )

        # Year and venue
        year = node_data.get("year", "?")
        venue_name = node_data.get("venue_name", "")
        if venue_name:
            details.append(
                html.P(
                    [
                        html.Strong("Published: "),
                        f"{venue_name} ({year})",
                    ]
                )
            )
        else:
            details.append(html.P([html.Strong("Year: "), str(year)]))

        # Venue rank badge
        venue_rank = node_data.get("venue_rank")
        if venue_rank:
            rank_colour = viz.VENUE_RANK_COLOURS.get(venue_rank, "#gray")
            details.append(
                html.Div(
                    [
                        html.Span(
                            venue_rank,
                            style={
                                "background": rank_colour,
                                "color": "white",
                                "padding": "4px 8px",
                                "borderRadius": "4px",
                                "fontSize": "12px",
                                "fontWeight": "bold",
                                "marginRight": "8px",
                            },
                        ),
                        html.Span(
                            f"Impact Factor: {node_data.get('impact_factor', 'N/A')}",
                            style={"fontSize": "14px"},
                        ),
                    ]
                )
            )

        # Quality metrics
        quality_score = node_data.get("quality_score")
        if quality_score is None:
            node_type = node_data.get("node_type", "reference")
            quality_score = 1.0 if node_type == "document" else 0.5
        citation_count = node_data.get("citation_count", 0)
        details.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong("Quality Score: "),
                            html.Span(
                                f"{quality_score:.2f}",
                                style={
                                    "color": (
                                        "#10b981"
                                        if quality_score > 0.7
                                        else "#f59e0b" if quality_score > 0.5 else "#ef4444"
                                    )
                                },
                            ),
                        ],
                        style={"display": "inline-block", "marginRight": "20px"},
                    ),
                    html.Div(
                        [
                            html.Strong("Citations: "),
                            str(citation_count),
                        ],
                        style={"display": "inline-block"},
                    ),
                ],
                style={"margin": "8px 0"},
            )
        )

        # Verification source and confidence
        source = node_data.get("source", "unresolved")
        confidence = node_data.get("confidence")

        source_labels = {
            "crossref": "CrossRef",
            "arxiv": "ArXiv",
            "url_fetch": "URL Fetch",
            "document": "Document",
            "unresolved": "Unresolved",
        }
        source_label = source_labels.get(source, source.title() if source else "Unknown")

        source_colours = viz.SOURCE_TRUSTWORTHINESS_COLOURS
        source_colour = source_colours.get(source, "#6b7280")

        details.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong("Verification Source: "),
                            html.Span(
                                source_label,
                                style={
                                    "background": source_colour,
                                    "color": "white",
                                    "padding": "2px 8px",
                                    "borderRadius": "4px",
                                    "fontSize": "12px",
                                    "fontWeight": "bold",
                                },
                            ),
                        ],
                        style={"display": "inline-block", "marginRight": "20px"},
                    ),
                    html.Div(
                        [
                            html.Strong("Confidence: "),
                            html.Span(
                                f"{confidence:.2f}" if confidence is not None else "N/A",
                                style={
                                    "color": (
                                        "#10b981"
                                        if (confidence and confidence > 0.8)
                                        else (
                                            "#f59e0b"
                                            if (confidence and confidence > 0.5)
                                            else "#ef4444" if confidence else "#6b7280"
                                        )
                                    )
                                },
                            ),
                        ],
                        style={"display": "inline-block"},
                    ),
                ],
                style={"margin": "8px 0"},
            )
        )

        # Reference type and OA status
        ref_type = node_data.get("reference_type", "academic")
        oa_available = node_data.get("oa_available", False)
        details.append(
            html.P(
                [
                    html.Strong("Type: "),
                    ref_type,
                    html.Span(
                        " • Open Access" if oa_available else " • Paywalled",
                        style={"color": "#10b981" if oa_available else "#ef4444"},
                    ),
                ]
            )
        )

        # Link status warning
        link_status = node_data.get("link_status", "available")
        if link_status != "available":
            details.append(
                html.Div(
                    [
                        html.Span("⚠️ ", style={"fontSize": "18px"}),
                        html.Span(
                            f"Link {link_status.replace('_', ' ')}",
                            style={"color": "#ef4444", "fontWeight": "bold"},
                        ),
                    ],
                    style={
                        "padding": "8px",
                        "background": "#fef2f2",
                        "borderRadius": "4px",
                        "margin": "8px 0",
                    },
                )
            )

        # Content changed warning
        if node_data.get("content_changed"):
            details.append(
                html.Div(
                    [
                        html.Span("🔄 ", style={"fontSize": "18px"}),
                        html.Span(
                            "Content has been updated since initial ingestion",
                            style={"color": "#f59e0b", "fontWeight": "bold"},
                        ),
                    ],
                    style={
                        "padding": "8px",
                        "background": "#fffbeb",
                        "borderRadius": "4px",
                        "margin": "8px 0",
                    },
                )
            )

        # DOI link
        doi = node_data.get("doi")
        if doi:
            details.append(
                html.P(
                    [
                        html.Strong("DOI: "),
                        html.A(
                            doi,
                            href=f"https://doi.org/{doi}",
                            target="_blank",
                            style={"color": "#3b82f6"},
                        ),
                    ]
                )
            )

        return html.Div(details)

    except Exception as e:
        logger = get_logger()
        logger.error(f"Error displaying node details: {e}")
        return html.P(f"Error loading details: {str(e)}", style={"color": "#ef4444"})


@callback(
    Output("citation-export-download", "data"),
    [
        Input("citation-export-button", "n_clicks"),
    ],
    [
        State("citation-persona-dropdown", "value"),
        State("citation-link-status-dropdown", "value"),
        State("citation-reference-type-dropdown", "value"),
        State("citation-venue-type-dropdown", "value"),
        State("citation-venue-rank-dropdown", "value"),
        State("citation-source-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def export_citations_csv(
    n_clicks,
    persona,
    link_statuses,
    reference_types,
    venue_types,
    venue_ranks,
    sources,
):
    """
    Export filtered citations to CSV file.

    Args:
        n_clicks: Export button clicks
        persona: Selected persona
        link_statuses: Filter by link status
        reference_types: Filter by reference type
        venue_types: Filter by venue type
        venue_ranks: Filter by venue rank
        sources: Filter by verification source

    Returns:
        Dict with CSV data for download
    """
    logger = get_logger()

    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate

    try:
        viz = get_citation_viz()

        # Generate CSV content
        csv_content = viz.export_citations(
            persona=persona or "supervisor",
            link_statuses=link_statuses if link_statuses else None,
            reference_types=reference_types if reference_types else None,
            venue_types=venue_types if venue_types else None,
            venue_ranks=venue_ranks if venue_ranks else None,
            sources=sources if sources else None,
        )

        # Return as file download
        from datetime import datetime

        filename = f"citations_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        logger.info(f"Exporting citations to {filename}")

        return dict(content=csv_content, filename=filename)

    except Exception as e:
        logger.error(f"Error exporting citations: {e}", exc_info=True)
        raise


def register_citation_graph_callbacks():
    """
    Register all citation graph callbacks with the Dash app.

    Usage in dashboard.py:
        from scripts.ui.academic.citation_graph_callbacks import register_citation_graph_callbacks
        register_citation_graph_callbacks()"""
    pass
