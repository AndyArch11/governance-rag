"""Academic-specific dashboard components and callbacks."""

from scripts.ui.academic.academic_references import (
    AcademicReferences,
    create_academic_references_layout,
)
from scripts.ui.academic.citation_graph_callbacks import (
    register_citation_graph_callbacks,
)
from scripts.ui.academic.citation_graph_viz import (
    CitationGraphViz,
    get_citation_viz,
)

__all__ = [
    "AcademicReferences",
    "create_academic_references_layout",
    "CitationGraphViz",
    "get_citation_viz",
    "register_citation_graph_callbacks",
]
