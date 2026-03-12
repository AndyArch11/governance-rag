"""Citation graph builder for academic ingestion."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class CitationNode:
    node_id: str
    node_type: str  # "document" or "reference"
    title: Optional[str] = None
    authors: List[str] = field(
        default_factory=list
    )  # Author list for academic/author-year citations
    doi: Optional[str] = None
    year: Optional[int] = None
    year_verified: bool = False  # True if year from CrossRef/OpenAlex resolution
    reference_type: Optional[str] = None
    quality_score: Optional[float] = None  # Quality rating (0.0-1.0) if available
    link_status: Optional[str] = None  # available, stale_404, stale_timeout, stale_moved
    venue_type: Optional[str] = None  # journal, conference, preprint, web
    venue_rank: Optional[str] = None  # Q1-Q4 or A*, A, B, C
    source: Optional[str] = None
    confidence: Optional[float] = None  # Confidence score from provider resolution (0.0-1.0)


@dataclass
class CitationEdge:
    source: str
    target: str
    relation: str = "cites"


@dataclass
class CitationGraph:
    nodes: Dict[str, CitationNode] = field(default_factory=dict)
    edges: List[CitationEdge] = field(default_factory=list)

    def _compute_quality_score(self, metadata: dict, node_type: str) -> float:
        """Compute a heuristic quality score (0.0-1.0)."""
        source = (metadata.get("source") or "").lower()
        venue_rank = metadata.get("venue_rank")
        citation_count = metadata.get("citation_count")
        link_status = metadata.get("link_status")
        oa_available = metadata.get("oa_available")
        paywall_detected = metadata.get("paywall_detected")
        year_verified = metadata.get("year_verified")
        confidence = metadata.get("confidence")

        # Source trustworthiness
        source_scores = {
            "crossref": 0.9,
            "openalex": 0.85,
            "pubmed": 0.85,
            "semantic_scholar": 0.8,
            "arxiv": 0.75,
            "datacite": 0.8,
            "orcid": 0.75,
            "unpaywall": 0.7,
            "google_scholar": 0.55,
            "url_fetch": 0.45,
            "document": 0.6,
            "heuristic": 0.35,
            "unresolved": 0.2,
        }
        source_score = source_scores.get(source, 0.5)

        # Venue rank signal
        # TODO: not capturing venue rank in current metadata - need to add this in provider resolution step
        venue_scores = {
            "Q1": 0.95,
            "A*": 0.95,
            "Q2": 0.85,
            "A": 0.85,
            "Q3": 0.75,
            "B": 0.75,
            "Q4": 0.65,
            "C": 0.65,
        }
        venue_score = venue_scores.get(venue_rank, 0.5)

        # Citation count (log scaled)
        # TODO: not currently capturing citation count in metadata - need to add this in provider resolution step
        citation_score = 0.0
        if isinstance(citation_count, int) and citation_count > 0:
            citation_score = min(1.0, math.log1p(citation_count) / math.log1p(200))

        # Link status
        #
        link_scores = {
            "available": 1.0,
            "stale_moved": 0.8,
            "stale_timeout": 0.6,
            "stale_404": 0.3,
            "unresolved": 0.4,
        }
        link_score = link_scores.get(link_status, 0.7)

        # Confidence (provider score)
        confidence_score = 0.5
        if isinstance(confidence, (int, float)):
            confidence_score = max(0.0, min(1.0, float(confidence)))

        # Weighted blend
        weights = {
            "source": 0.3,
            "venue": 0.2,
            "citations": 0.2,
            "link": 0.2,
            "confidence": 0.1,
        }
        score = (
            source_score * weights["source"]
            + venue_score * weights["venue"]
            + citation_score * weights["citations"]
            + link_score * weights["link"]
            + confidence_score * weights["confidence"]
        )

        # Small adjustments
        if oa_available is True:
            score += 0.05
        if paywall_detected is True:
            score -= 0.05
        if year_verified is True:
            score += 0.03

        # Slight bias for primary documents
        if node_type == "document":
            score += 0.05

        return max(0.0, min(1.0, score))

    def add_document(self, doc_id: str, metadata: Optional[dict] = None) -> None:
        """Add document node with optional metadata (title, authors, year, etc.)."""
        if doc_id not in self.nodes:
            if metadata is None:
                metadata = {}

            # Parse authors - handle various formats
            authors = metadata.get("authors", [])
            if authors:
                if isinstance(authors, str):
                    # Try JSON parsing first
                    try:
                        import json

                        parsed = json.loads(authors)
                        if isinstance(parsed, list):
                            authors = parsed
                        else:
                            # If not JSON, treat as comma-separated string
                            authors = [a.strip() for a in authors.split(",") if a.strip()]
                    except (json.JSONDecodeError, ValueError):
                        # Not JSON, treat as comma-separated string
                        authors = [a.strip() for a in authors.split(",") if a.strip()]
                elif not isinstance(authors, list):
                    authors = []

            quality_score = metadata.get("quality_score")
            if quality_score is None:
                quality_score = self._compute_quality_score(metadata, "document")

            self.nodes[doc_id] = CitationNode(
                node_id=doc_id,
                node_type="document",
                title=metadata.get("title"),
                authors=authors,
                year=metadata.get("year"),
                quality_score=quality_score,
                source=metadata.get("source", "document"),
            )

    def add_reference(self, ref_id: str, metadata: dict) -> None:
        if ref_id not in self.nodes:
            # Year is verified if from CrossRef, OpenAlex, or other authoritative source
            source = metadata.get("source", "")
            year = metadata.get("year")
            year_verified = source in (
                "crossref",
                "datacite",
                "openAlex",
                "arxiv",
                "pubmed",
                "semantic_scholar",
                "orcid",
            )

            # If no year from provider, try to extract from citation text
            if not year and not year_verified:
                citation_text = metadata.get("citation")
                extracted_year = extract_year_from_citation(citation_text)
                if extracted_year:
                    year = extracted_year
                    # Mark as extracted from citation (unverified but better than NULL)
                    year_verified = False

            # Get title - prefer provided, fallback to extraction from citation
            title = metadata.get("title")

            # Clean book review titles (common in CrossRef data for book reviews)
            # Also extract the book's actual authors (not the reviewer)
            book_authors = []
            if title and source == "crossref":
                title, book_authors = clean_book_review_title(title)

            if not title:
                citation_text = metadata.get("citation")
                extracted_title = extract_title_from_citation(citation_text)
                if extracted_title:
                    title = extracted_title

            # Get authors - prefer provided list, fallback to extraction from citation
            # For book reviews, use the book's authors (not the reviewer)
            authors = metadata.get("authors", [])
            if book_authors:
                # Use book authors extracted from title instead of reviewer
                authors = book_authors
            elif not authors:
                citation_text = metadata.get("citation")
                extracted_authors = extract_authors_from_citation(citation_text)
                if extracted_authors:
                    authors = extracted_authors

            quality_score = metadata.get("quality_score")
            if quality_score is None:
                quality_score = self._compute_quality_score(metadata, "reference")

            self.nodes[ref_id] = CitationNode(
                node_id=ref_id,
                node_type="reference",
                title=title,
                authors=authors,
                doi=metadata.get("doi"),
                year=year,
                year_verified=year_verified,
                reference_type=metadata.get("reference_type"),
                quality_score=quality_score,
                link_status=metadata.get("link_status", "available"),
                venue_type=metadata.get("venue_type"),
                venue_rank=metadata.get("venue_rank"),
                source=source,
                confidence=metadata.get("confidence"),
            )

    def add_edge(self, doc_id: str, ref_id: str) -> None:
        self.edges.append(CitationEdge(source=doc_id, target=ref_id))

    def to_dict(self) -> dict:
        return {
            "nodes": [node.__dict__ for node in self.nodes.values()],
            "edges": [edge.__dict__ for edge in self.edges],
        }

    def write_sqlite(
        self,
        sqlite_path: Path,
        doc_id: Optional[str] = None,
        temp_suffix: str = "_temp",
        export_json: bool = True,
    ) -> None:
        """
        Write graph to SQLite database with atomic swap.

        Args:
            sqlite_path: Target SQLite database path
            doc_id: Document ID for metadata tracking
            temp_suffix: Suffix for temporary database during build
            export_json: If True, also export to JSON for backward compatibility
        """
        from datetime import datetime

        from .citation_graph_writer import CitationGraphWriter

        # Build to temporary database
        temp_path = sqlite_path.parent / f"{sqlite_path.stem}{temp_suffix}.db"

        with CitationGraphWriter(temp_path, replace=True) as writer:
            # Insert all nodes
            nodes_data = []
            for node in self.nodes.values():
                nodes_data.append(
                    {
                        "node_id": node.node_id,
                        "node_type": node.node_type,
                        "title": node.title,
                        "authors": node.authors,
                        "doi": node.doi,
                        "year": node.year,
                        "year_verified": node.year_verified,
                        "reference_type": node.reference_type,
                        "quality_score": node.quality_score,
                        "link_status": node.link_status,
                        "venue_type": node.venue_type,
                        "venue_rank": node.venue_rank,
                        "source": node.source,
                        "confidence": node.confidence,
                    }
                )

            writer.insert_nodes_batch(nodes_data)

            # Insert all edges
            edges_data = []
            for edge in self.edges:
                edges_data.append(
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "relation": edge.relation,
                    }
                )

            writer.insert_edges_batch(edges_data)

            # Set metadata
            if doc_id:
                writer.set_metadata("document_id", doc_id)
            writer.set_metadata("build_time", datetime.now().isoformat())
            writer.set_metadata("node_count", str(len(self.nodes)))
            writer.set_metadata("edge_count", str(len(self.edges)))

            # Finalise
            writer.finalise()

            # Export to JSON if requested
            if export_json:
                json_path = sqlite_path.parent / f"{sqlite_path.stem}.json"
                writer.export_to_json(json_path)

            # Atomic swap to production path
            writer.atomic_swap(sqlite_path)


def reference_id_from_metadata(metadata: dict) -> str:
    doi = metadata.get("doi")
    if doi:
        return f"doi:{doi.lower()}"
    title = metadata.get("title") or metadata.get("citation") or ""
    digest = hashlib.sha256(title.encode("utf-8")).hexdigest()
    return f"ref:{digest}"


def extract_year_from_citation(citation_text: str) -> Optional[int]:
    """
    Extract publication year from citation text.
    Searches for 4-digit years between 1900-2100.
    Returns the first year found or None.
    """
    if not citation_text:
        return None

    # Look for 4-digit years in common patterns
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", citation_text)

    if years:
        # Convert to int and filter for reasonable years
        for year_str in years:
            year = int(year_str)
            if 1900 <= year <= 2100:
                return year

    return None


def extract_authors_from_citation(citation_text: str) -> List[str]:
    """
    Extract author names from citation text (usually at start).
    Properly groups surname with initials.
    Handles patterns like:
      - "Aarons, G. A., Ehrhart, M. G., & Torres, E. M. (2020). Title..."
      - "Smith et al. (2020). Title..."
    Returns list of author names (properly grouped) or empty list if not found.
    """
    if not citation_text:
        return []

    # Find the year pattern to know where authors should end
    year_match = re.search(r"\(20\d{2}|1[89]\d{2}\)", citation_text)
    if not year_match:
        # No year found, likely not a proper citation
        return []

    # Get text before the year
    pre_year = citation_text[: year_match.start()].strip()

    # Handle "et al." - extract first author only
    if "et al" in pre_year.lower():
        # Extract text before "et al"
        et_al_match = re.search(r"(.+?)\s+et\s+al\.?", pre_year, re.IGNORECASE)
        if et_al_match:
            first_author = et_al_match.group(1).strip()
            # Clean up trailing comma or period
            first_author = first_author.rstrip(".,;:")
            if first_author and len(first_author) > 1:
                return [first_author]
        return []

    # Split by '&' and 'and' to get individual author groups
    # Pattern: "LastName, Initials, LastName, Initials, & LastName, Initials"
    # We want to split on & but keep the structure

    # Split on & first to handle the last author
    parts = re.split(r"\s*&\s*", pre_year)

    authors = []

    for part in parts:
        # Each part may contain multiple authors separated by commas
        # In Harvard style: "Name1, Init1., Name2, Init2., Name3, Init3."
        # We need to group "Name, Init." together

        # Strategy: Split by comma, then pair surname with following initials
        segments = [s.strip() for s in part.split(",")]
        segments = [s for s in segments if s]  # Remove empty strings

        i = 0
        while i < len(segments):
            segment = segments[i]

            # Check if next segment is initials (uppercase letters with dots/periods)
            if i + 1 < len(segments):
                next_seg = segments[i + 1]
                # Check if next looks like initials (e.g., "G. A." or "G.A")
                if re.match(r"^[A-Z]\.?\s*[A-Z]\.?|^[A-Z]\.$", next_seg):
                    # Group surname + initials
                    full_name = f"{segment}, {next_seg}"
                    authors.append(full_name)
                    i += 2
                    continue

            # If segment is a surname (capitalised word), add it alone
            if segment and len(segment) > 1 and segment[0].isupper():
                authors.append(segment)

            i += 1

    return authors[:5]  # Return top 5 authors


def clean_book_review_title(title: str) -> Tuple[str, List[str]]:
    """
    Clean titles that are actually book reviews or citations.
    Pattern: "Author, Initial. Year. Book Title. Publisher"
    Returns tuple of (cleaned_title, book_authors).
    """
    if not title:
        return title, []

    # Define publishers once to avoid repetition
    publishers = [
        "Sage",
        "Springer",
        "Wiley",
        "Oxford",
        "Cambridge",
        "Routledge",
        "Pearson",
        "McGraw",
        "Elsevier",
        "Academic Press",
        "MIT Press",
        "Londres",
    ]
    publishers_pattern = "|".join(publishers)
    publisher_regex = rf"^(.+?)\.\s+(?:{publishers_pattern}).*$"

    # Check if title starts with author pattern: "LastName, Initial(s). Year."
    # Pattern 1: Single or multiple authors with standard format
    # "LastName, Initial(s)., LastName, Initial(s). Year."
    match = re.match(
        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+([A-Z]\.?\s*)+\s*\.?\s*(\d{4})\s*[.:]\s+(.+)", title
    )

    if match:
        # Extract book author (group 1 + initials in group 2)
        book_author_surname = match.group(1).strip()
        book_author_initials = match.group(2).strip()
        book_authors = [f"{book_author_surname}, {book_author_initials}"]

        # Extract just the book title (group 4)
        book_title = match.group(4).strip()

        # Remove publisher at the end if present
        publisher_match = re.match(publisher_regex, book_title, re.IGNORECASE)

        if publisher_match:
            book_title = publisher_match.group(1).strip()

        return book_title, book_authors

    # Pattern 2: Multiple authors "FirstName LastName, FirstName LastName (Year): Title"
    match2 = re.match(
        r"^([A-Z][a-z]+\s+[A-Z][a-z]+(?:,\s+[A-Z][a-z]+\s+[A-Z][a-z]+)*)\s+\((\d{4})\)\s*[.:]\s+(.+)",
        title,
    )

    if match2:
        # Parse book authors (comma-separated full names)
        authors_str = match2.group(1).strip()
        book_authors = [a.strip() for a in authors_str.split(",")]

        book_title = match2.group(3).strip()

        # Remove publisher and location
        publisher_match = re.match(publisher_regex, book_title, re.IGNORECASE)

        if publisher_match:
            book_title = publisher_match.group(1).strip()

        return book_title, book_authors

    # Pattern 3: "Author Year. Title" (no comma)
    match3 = re.match(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(\d{4})\s*[.:]\s+(.+)", title)

    if match3:
        book_authors = [match3.group(1).strip()]
        book_title = match3.group(3).strip()

        # Remove publisher
        publisher_match = re.match(publisher_regex, book_title, re.IGNORECASE)

        if publisher_match:
            book_title = publisher_match.group(1).strip()

        return book_title, book_authors

    return title, []


def extract_title_from_citation(citation_text: str) -> Optional[str]:
    """
    Extract publication title from citation text.
    In Harvard/APA format, title typically appears after year: "(YYYY). Title. Journal..."
    Returns title or None if not found.
    """
    if not citation_text:
        return None

    # Find the year pattern to locate where title should start
    year_match = re.search(r"\(\s*(20\d{2}|1[89]\d{2})\s*\)", citation_text)
    if not year_match:
        return None

    # Get text after the year pattern
    post_year = citation_text[year_match.end() :].strip()

    # Title typically starts with a dot/period or directly after parenthesis
    # Remove leading punctuation
    post_year = post_year.lstrip("., ")

    if not post_year:
        return None

    # Find where title ends - usually at venue/journal indicator
    # Common patterns:
    # - Title. Journal/Venue, ...
    # - Title in Sentence Case. Publisher
    # - Title (Edition). Publisher
    #
    # Look for sentence ending followed by likely venue indicator or volume/issue info
    title_match = re.match(
        r"^([^.]+(?:\.[^.])*?)\.\s+(?:In\s+|Journal|Review|Harvard|Oxford|Springer|Wiley|Proceedings|American|British|International|\d+\()",
        post_year,
        re.IGNORECASE,
    )

    if title_match:
        title = title_match.group(1).strip()
        # Clean up common artifacts
        title = re.sub(r"\s+", " ", title)  # Normalise whitespace
        return title if len(title) > 3 else None

    # Fallback: take first sentence (period) if text is reasonably short
    if len(post_year) < 200:
        period_match = re.match(r"^([^.]+)\.", post_year)
        if period_match:
            title = period_match.group(1).strip()
            # Filter out likely non-titles (numbers, too short, etc.)
            if len(title) > 3 and not re.match(r"^[\d\s]+$", title):
                return title

    return None


def add_references_to_graph(
    graph: CitationGraph,
    doc_id: str,
    references: List[dict],
    doc_metadata: Optional[dict] = None,
) -> None:
    """Add document and its references to the citation graph.

    Args:
        graph: CitationGraph to add nodes/edges to
        doc_id: Document identifier
        references: List of reference metadata dicts
        doc_metadata: Optional document metadata (title, authors, year, etc.)
    """
    graph.add_document(doc_id, metadata=doc_metadata)
    for ref in references:
        ref_id = reference_id_from_metadata(ref)
        graph.add_reference(ref_id, ref)
        graph.add_edge(doc_id, ref_id)
