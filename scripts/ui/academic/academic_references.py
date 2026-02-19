"""Academic References Dashboard Module.

Provides visualisation and analysis tools for academic references extracted from papers.
Shows merged references, orphaned journals, domain terminology, and citation relationships.

Features:
- Merged reference detection highlighting
- Orphaned journal reference display
- Domain terminology analysis (top terms, relationships)
- Reference metadata table with search/filter
- Citation count statistics
- Terminology co-occurrence visualisation

TODO: Clean up debug print statements and ensure all database interactions are robust with error handling.
TODO: Consider adding caching for expensive queries.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dash
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html


class AcademicReferences:
    """Module for academic references visualisation in dashboard."""

    def __init__(self, terminology_db: Optional[Path] = None):
        """Initialise module with optional terminology database path.

        Args:
            terminology_db: Optional path to SQLite database containing terminology data.
        """
        self.terminology_db = terminology_db
        self.conn: Optional[sqlite3.Connection] = None

        print(f"DEBUG AcademicReferences.__init__: terminology_db={terminology_db}")
        if terminology_db and Path(terminology_db).exists():
            # Use check_same_thread=False to allow usage across Dash callback threads
            self.conn = sqlite3.connect(str(terminology_db), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            print(f"DEBUG: Database connection established to {terminology_db}")
        else:
            print(
                f"DEBUG: Database not connected. terminology_db={terminology_db}, exists={Path(terminology_db).exists() if terminology_db else 'N/A'}"
            )

    def close(self) -> None:
        """Close the terminology database connection if open."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "AcademicReferences":
        """Enter context manager scope."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Ensure connection is closed on context manager exit."""
        self.close()
        return False

    def __del__(self) -> None:
        """Best-effort cleanup to prevent unclosed SQLite connections."""
        try:
            self.close()
        except Exception:
            pass

    def get_top_terms(
        self, domain: str, limit: int = 20, doc_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get top domain terms by relevance.

        Filters out overlapping sub-ngrams to reduce noise. For example, if we have
        'Aboriginal Torres Strait Islander', we don't show 'Aboriginal Torres Strait',
        'Torres Strait Islander', 'Torres Strait', etc.

        TODO: not all sub-ngrams are noise - some may be relevant in their own right (e.g. 'Torres Strait' is a valid term even if it's part of 'Aboriginal Torres Strait Islander').
        Consider allowing users to toggle this filtering on/off or to show related sub-terms in a hierarchical way. For now, we filter to keep the most relevant parent n-grams to reduce noise in the top results, but this is an area for future improvement.

        Args:
            domain: Domain name to filter terms.
            limit: Maximum number of terms to return.
            doc_filter: Optional document ID filter.

        Returns:
            List of dictionaries containing term information.
        """
        if not self.conn:
            return []

        try:
            cursor = self.conn.cursor()
            if doc_filter:
                cursor.execute(
                    """
                    SELECT term, frequency, domain_relevance_score, term_type
                    FROM domain_terms
                    WHERE domain = ? AND doc_ids LIKE ?
                    ORDER BY domain_relevance_score DESC, frequency DESC
                """,
                    (domain, f"%{doc_filter}%"),
                )
            else:
                cursor.execute(
                    """
                    SELECT term, frequency, domain_relevance_score, term_type
                    FROM domain_terms
                    WHERE domain = ?
                    ORDER BY domain_relevance_score DESC, frequency DESC
                """,
                    (domain,),
                )

            all_terms = [dict(row) for row in cursor.fetchall()]

            # Filter to remove overlapping sub-ngrams
            # If a term is a substring of another term in a higher-ranked result,
            # don't include it (reduces noise from ngram decomposition)
            filtered_terms = []
            term_strings = [t["term"] for t in all_terms]

            for i, term_data in enumerate(all_terms):
                term = term_data["term"]
                is_subterm = False

                # Check if this term appears within any higher-ranked (more relevant) term
                for j in range(i):
                    other_term = term_strings[j]
                    # Check if current term is a substring of a higher-ranked term
                    # (same terms or as a component)
                    if term != other_term and self._is_subterm(term, other_term):
                        is_subterm = True
                        break

                if not is_subterm:
                    filtered_terms.append(term_data)
                    if len(filtered_terms) >= limit:
                        break

            return filtered_terms
        except Exception as e:
            print(f"Error fetching top terms: {e}")
            return []

    def _is_subterm(self, term: str, other_term: str) -> bool:
        """Check if term is a sub-ngram of other_term.

        Examples:
        - 'torres strait' is a sub-ngram of 'aboriginal torres strait islander'
        - 'strait' is a sub-ngram of 'torres strait'
        - 'aboriginal torres' is a sub-ngram of 'aboriginal torres strait islander'

        Args:
            term: Potential sub-ngram
            other_term: Potential parent ngram

        Returns:
            True if term is a component of other_term's words
        """
        term_words = term.split()
        other_words = other_term.split()

        # Empty terms or when parent is empty cannot be subterms
        if not term_words or not other_words:
            return False

        if len(term_words) >= len(other_words):
            return False

        # Check if term's words appear consecutively in other_term's words
        for i in range(len(other_words) - len(term_words) + 1):
            if other_words[i : i + len(term_words)] == term_words:
                return True

        return False

    def get_doc_ids_for_domain(self, domain: str) -> List[str]:
        """Get distinct doc_ids associated with a domain.

        Args:
            domain: Domain name to filter doc_ids.

        Returns:
            List of distinct document IDs.
        """
        if not self.conn:
            return []

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT doc_ids
                FROM domain_terms
                WHERE domain = ?
            """,
                (domain,),
            )

            doc_ids: set[str] = set()
            for row in cursor.fetchall():
                doc_id_str = row[0] or ""
                for doc_id in doc_id_str.split(","):
                    cleaned = doc_id.strip()
                    if cleaned:
                        doc_ids.add(cleaned)

            return sorted(doc_ids)
        except Exception as e:
            print(f"Error fetching doc_ids: {e}")
            return []

    def get_term_relationships(self, term: str, limit: int = 10) -> List[Tuple[str, str]]:
        """Get related terms for a given term.

        Args:
            term: Term to find relationships for.
            limit: Maximum number of related terms to return.

        Returns:
            List of tuples containing related term and relationship type.
        """
        if not self.conn:
            return []

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT target_term, relationship_type
                FROM term_relationships
                WHERE source_term = ?
                ORDER BY target_term
                LIMIT ?
            """,
                (term, limit),
            )

            return [(row[0], row[1]) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error fetching term relationships: {e}")
            return []

    def get_reference_statistics(self) -> Dict[str, Any]:
        """Get overall reference statistics.

        Returns:
            Dictionary containing total terms, domains, total frequency, and domain count.
        """
        print(f"DEBUG get_reference_statistics: self.conn={self.conn}")
        if not self.conn:
            return {}

        try:
            cursor = self.conn.cursor()

            # Total unique terms
            cursor.execute("SELECT COUNT(*) FROM domain_terms")
            total_terms = cursor.fetchone()[0]

            # Domains
            cursor.execute("SELECT DISTINCT domain FROM domain_terms")
            domains = [row[0] for row in cursor.fetchall()]
            print(f"DEBUG get_reference_statistics: found {len(domains)} domains: {domains}")

            # Total frequency across all terms
            cursor.execute("SELECT SUM(frequency) FROM domain_terms")
            total_frequency = cursor.fetchone()[0] or 0

            return {
                "total_terms": total_terms,
                "domains": domains,
                "total_frequency": total_frequency,
                "domain_count": len(domains),
            }
        except Exception as e:
            print(f"Error fetching reference statistics: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def create_layout(self) -> html.Div:
        """Create the academic references dashboard layout.

        Returns:
            A Dash HTML Div containing the layout for the academic references tab.
        """
        stats = self.get_reference_statistics()
        print(f"DEBUG create_layout: stats={stats}")

        return html.Div(
            [
                html.H3("📚 Academic References Analysis", style={"marginTop": 0}),
                # Summary statistics
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4(str(stats.get("total_terms", 0))),
                                html.P(
                                    "Total Terms",
                                    style={"margin": 0, "fontSize": "12px", "color": "#666"},
                                ),
                            ],
                            className="stat-card",
                            style={
                                "backgroundColor": "#e3f2fd",
                                "borderLeft": "4px solid #2196f3",
                                "padding": "16px",
                                "borderRadius": "4px",
                                "textAlign": "center",
                            },
                        ),
                        html.Div(
                            [
                                html.H4(str(stats.get("domain_count", 0))),
                                html.P(
                                    "Domains",
                                    style={"margin": 0, "fontSize": "12px", "color": "#666"},
                                ),
                            ],
                            className="stat-card",
                            style={
                                "backgroundColor": "#f3e5f5",
                                "borderLeft": "4px solid #9c27b0",
                                "padding": "16px",
                                "borderRadius": "4px",
                                "textAlign": "center",
                            },
                        ),
                        html.Div(
                            [
                                html.H4(str(stats.get("total_frequency", 0))),
                                html.P(
                                    "Total References",
                                    style={"margin": 0, "fontSize": "12px", "color": "#666"},
                                ),
                            ],
                            className="stat-card",
                            style={
                                "backgroundColor": "#e8f5e9",
                                "borderLeft": "4px solid #4caf50",
                                "padding": "16px",
                                "borderRadius": "4px",
                                "textAlign": "center",
                            },
                        ),
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(3, 1fr)",
                        "gap": "12px",
                        "marginBottom": "24px",
                    },
                ),
                # Domain selection
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    "Select Domain:",
                                    style={"fontWeight": 500, "marginRight": "8px"},
                                ),
                                dcc.Dropdown(
                                    id="academic-domain-dropdown",
                                    options=[
                                        {"label": d, "value": d} for d in stats.get("domains", [])
                                    ],
                                    value=(
                                        stats.get("domains", [None])[0]
                                        if stats.get("domains")
                                        else None
                                    ),
                                    style={"width": "300px"},
                                ),
                            ],
                            style={"marginRight": "16px"},
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Select Document:",
                                    style={"fontWeight": 500, "marginRight": "8px"},
                                ),
                                dcc.Dropdown(
                                    id="academic-doc-dropdown",
                                    options=[{"label": "All documents", "value": "__all__"}],
                                    value="__all__",
                                    style={"width": "360px"},
                                ),
                            ]
                        ),
                    ],
                    style={
                        "marginBottom": "24px",
                        "display": "flex",
                        "flexWrap": "wrap",
                        "alignItems": "center",
                    },
                ),
                # Top terms visualisation
                html.Div(
                    [
                        html.H4("Top Domain Terms", style={"marginTop": 0}),
                        dcc.Graph(id="academic-top-terms-chart", style={"height": "400px"}),
                    ],
                    className="card",
                    style={"marginBottom": "24px"},
                ),
                # Terms table
                html.Div(
                    [
                        html.H4("Detailed Terms Analysis", style={"marginTop": 0}),
                        html.Div(
                            id="academic-terms-table",
                            style={
                                "fontSize": "12px",
                                "maxHeight": "500px",
                                "overflowY": "auto",
                            },
                        ),
                    ],
                    className="card",
                    style={"marginBottom": "24px"},
                ),
                # Term relationships
                html.Div(
                    [
                        html.H4("Term Relationships", style={"marginTop": 0}),
                        html.Div(
                            [
                                html.Label(
                                    "Select Term:", style={"fontWeight": 500, "marginRight": "8px"}
                                ),
                                dcc.Dropdown(id="academic-term-selector", style={"width": "300px"}),
                            ],
                            style={"marginBottom": "12px"},
                        ),
                        html.Div(
                            id="academic-term-relationships",
                            style={
                                "fontSize": "12px",
                                "padding": "12px",
                                "backgroundColor": "#f5f5f5",
                                "borderRadius": "4px",
                            },
                        ),
                    ],
                    className="card",
                ),
            ]
        )


# Global module instance for callback access
_global_module: Optional[AcademicReferences] = None


def _get_global_module() -> Optional[AcademicReferences]:
    """Get the global module instance.

    Returns:
        The global AcademicReferences instance if set, otherwise None.
    """
    return _global_module


def _set_global_module(module: AcademicReferences):
    """Set the global module instance.

    Args:
        module: The AcademicReferences instance to set as global.
    """
    global _global_module
    _global_module = module


def _normalise_doc_filter(doc_filter: Optional[str]) -> Optional[str]:
    """Normalise the document filter value.

    Args:
        doc_filter: The document filter value.

    Returns:
        The normalised document filter value, or None if no filter is applied.
    """
    if not doc_filter or doc_filter == "__all__":
        return None
    return re.sub(r"_v\d+$", "", doc_filter)


# Module-level callback registrations
@callback(
    Output("academic-top-terms-chart", "figure"),
    Input("academic-domain-dropdown", "value"),
    Input("academic-doc-dropdown", "value"),
)
def _update_top_terms_chart(domain, doc_filter):
    """Update top terms chart callback.

    Args:
        domain: Selected domain from dropdown.
        doc_filter: Selected document filter from dropdown.

    Returns:
        A Plotly figure object representing the top terms bar chart.
    """
    module = _get_global_module()
    print(
        f"DEBUG _update_top_terms_chart: domain={domain}, doc_filter={doc_filter}, module={module}"
    )
    if not module:
        return go.Figure().add_annotation(text="Module not initialised")
    if not domain:
        return go.Figure().add_annotation(text="No domain selected")

    effective_doc_filter = _normalise_doc_filter(doc_filter)

    terms = module.get_top_terms(domain, limit=20, doc_filter=effective_doc_filter)
    print(f"DEBUG _update_top_terms_chart: got {len(terms)} terms")
    if not terms:
        return go.Figure().add_annotation(text="No terms found for this domain")

    # Reverse list so highest relevance (first in query result) renders at TOP of horizontal bar
    # Plotly renders horizontal bars bottom-to-top in array order
    terms_for_chart = list(reversed(terms))

    fig = go.Figure(
        data=[
            go.Bar(
                x=[t["frequency"] for t in terms_for_chart],
                y=[t["term"] for t in terms_for_chart],
                orientation="h",
                marker=dict(
                    color=[t["domain_relevance_score"] for t in terms_for_chart],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Relevance"),
                ),
                text=[f"{t['frequency']}" for t in terms_for_chart],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>Frequency: %{x}<br>Relevance: %{marker.color:.2f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=f"Top Terms in {domain}",
        xaxis_title="Frequency",
        yaxis_title="Term",
        height=400,
        margin=dict(l=200, r=20, t=40, b=20),
        showlegend=False,
    )

    return fig


@callback(
    Output("academic-terms-table", "children"),
    Input("academic-domain-dropdown", "value"),
    Input("academic-doc-dropdown", "value"),
)
def _update_terms_table(domain, doc_filter):
    """Update terms table callback.

    Args:
        domain: Selected domain from dropdown.
        doc_filter: Selected document filter from dropdown.

    Returns:
        An HTML table element representing the terms table.
    """
    module = _get_global_module()
    print(f"DEBUG _update_terms_table: domain={domain}, doc_filter={doc_filter}, module={module}")
    if not module:
        return html.P("Module not initialised", style={"color": "#d32f2f"})
    if not domain:
        return html.P("Select a domain to view terms", style={"color": "#999"})

    effective_doc_filter = _normalise_doc_filter(doc_filter)

    terms = module.get_top_terms(domain, limit=50, doc_filter=effective_doc_filter)
    print(f"DEBUG _update_terms_table: got {len(terms)} terms")
    if not terms:
        return html.P("No terms found", style={"color": "#999"})

    # Create table
    rows = []
    for i, term in enumerate(terms, 1):
        rows.append(
            html.Tr(
                [
                    html.Td(
                        str(i), style={"textAlign": "center", "width": "40px", "fontWeight": 500}
                    ),
                    html.Td(term["term"], style={"fontWeight": 500}),
                    html.Td(f"{term['term_type']}", style={"color": "#666", "fontSize": "11px"}),
                    html.Td(f"{term['frequency']}", style={"textAlign": "right", "width": "60px"}),
                    html.Td(
                        f"{term['domain_relevance_score']:.2f}",
                        style={"textAlign": "right", "width": "80px"},
                    ),
                ],
                style={
                    "borderBottom": "1px solid #eee",
                    "padding": "8px",
                },
            )
        )

    return html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th(
                            "#", style={"textAlign": "center", "width": "40px", "fontWeight": 600}
                        ),
                        html.Th("Term", style={"fontWeight": 600}),
                        html.Th("Type", style={"fontWeight": 600, "width": "100px"}),
                        html.Th(
                            "Freq.",
                            style={"textAlign": "right", "fontWeight": 600, "width": "60px"},
                        ),
                        html.Th(
                            "Relevance",
                            style={"textAlign": "right", "fontWeight": 600, "width": "80px"},
                        ),
                    ],
                    style={"borderBottom": "2px solid #333", "padding": "8px"},
                )
            ),
            html.Tbody(rows),
        ],
        style={"width": "100%", "borderCollapse": "collapse"},
    )


@callback(
    Output("academic-term-selector", "options"),
    Input("academic-domain-dropdown", "value"),
    Input("academic-doc-dropdown", "value"),
)
def _update_term_selector(domain, doc_filter):
    """Update term selector callback.

    Args:
        domain: Selected domain from dropdown.
        doc_filter: Selected document filter from dropdown.

    Returns:
        A list of options for the term selector dropdown.
    """
    module = _get_global_module()
    print(f"DEBUG _update_term_selector: domain={domain}, doc_filter={doc_filter}, module={module}")
    if not module:
        return []
    if not domain:
        return []

    effective_doc_filter = _normalise_doc_filter(doc_filter)
    terms = module.get_top_terms(domain, limit=50, doc_filter=effective_doc_filter)
    print(f"DEBUG _update_term_selector: got {len(terms)} terms")
    return [{"label": t["term"], "value": t["term"]} for t in terms]


@callback(
    Output("academic-term-relationships", "children"),
    Input("academic-term-selector", "value"),
)
def _update_term_relationships(term):
    """Update term relationships callback.

    Args:
        term: Selected term from dropdown.

    Returns:
        An HTML Div containing the related terms and their relationship types.
    """
    module = _get_global_module()
    print(f"DEBUG _update_term_relationships: term={term}, module={module}")
    if not module:
        return html.P("Module not initialised", style={"color": "#d32f2f"})
    if not term:
        return html.P("Select a term to view relationships", style={"color": "#999"})

    relationships = module.get_term_relationships(term, limit=15)
    print(f"DEBUG _update_term_relationships: got {len(relationships)} relationships")
    if not relationships:
        return html.P(f"No related terms found for '{term}'", style={"color": "#999"})

    items = []
    for i, (related_term, rel_type) in enumerate(relationships, 1):
        items.append(
            html.Div(
                [
                    html.Span(f"{i}. ", style={"fontWeight": 500, "color": "#999"}),
                    html.Span(related_term, style={"fontWeight": 500}),
                    html.Span(
                        f" ({rel_type})",
                        style={"color": "#666", "fontSize": "11px", "marginLeft": "8px"},
                    ),
                ],
                style={"marginBottom": "6px"},
            )
        )

    return html.Div(items)


def create_academic_references_layout(
    terminology_db: Optional[Path] = None,
) -> Tuple[html.Div, Any]:
    """Create academic references tab layout and return (layout, module) tuple.

    Args:
        terminology_db: Optional path to SQLite database containing terminology data.

    Returns:
        A tuple containing the Dash HTML Div for the layout and the AcademicReferences instance.
    """
    module = AcademicReferences(terminology_db)
    _set_global_module(module)  # Set global instance for callbacks
    layout = module.create_layout()
    return layout, module
