"""Build a deterministic evidence graph from chapter-aware thesis chunks."""

from __future__ import annotations

import hashlib
import io
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Protocol, Tuple

from scripts.ingest.academic.phd_assessor import PhDQualityAssessor
from scripts.utils.logger import audit

_ADAPTED_FROM_PATTERN = re.compile(
    r"\b(?:(?:adapted|modified|reproduced|redrawn|derived)\s+from|based\s+on)\b",
    re.IGNORECASE,
)
_AUTHOR_YEAR_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z'-]+)(?:\s+et\s+al\.?)?(?:\s*\(\s*|,\s*|\s+)" r"((?:19|20)\d{2})[a-z]?\s*\)?",
    re.IGNORECASE,
)
_NUMBERED_CITATION_PATTERN = re.compile(r"\[(\d+(?:(?:\s*[-\u2013]\s*\d+)|(?:\s*[,;]\s*\d+))*)\]")


class ChunkCollection(Protocol):
    """Minimal collection contract required to build a thesis evidence graph.

    Attributes:
        get: Method to retrieve ChromaDB-style records.
    """

    def get(self, **kwargs: Any) -> Dict[str, Any]:
        """Return ChromaDB-style records."""


@dataclass(frozen=True)
class ThesisEvidenceGraph:
    """Persisted deterministic thesis structure graph.

    Attributes:
        thesis_id: Canonical thesis ID.
        output_path: Path to the persisted SQLite graph.
        node_count: Number of nodes in the graph.
        edge_count: Number of edges in the graph.
    """

    thesis_id: str
    output_path: Path
    node_count: int
    edge_count: int


def _report_progress_checkpoint(stage: str, items_done: int, items_total: int) -> None:
    """Write a run-correlated progress checkpoint for dashboard polling.

    Args:
        stage: Current stage of the processing.
        items_done: Number of items processed so far.
        items_total: Total number of items to process.
    """
    total = max(0, items_total)
    completed = max(0, min(items_done, total))
    percent = 100.0 * completed / total if total else 100.0
    audit(
        "thesis_graph",
        "progress_checkpoint",
        {
            "stage": stage,
            "items_done": completed,
            "items_total": total,
            "percent": percent,
            "succeeded": completed,
            "failed": 0,
        },
    )


def get_thesis_graph_path(graphs_dir: Path, thesis_id: str) -> Path:
    """Return the stable SQLite path for a thesis evidence graph.

    Args:
        graphs_dir: Directory to store the SQLite graph.
        thesis_id: Canonical thesis ID.

    Returns:
        Path: Stable SQLite path for the thesis evidence graph.
    """
    filename = re.sub(r"[^a-z0-9]+", "_", thesis_id.lower()).strip("_")
    return graphs_dir / f"{filename or 'thesis'}.sqlite"


def _node_id(node_type: str, thesis_id: str, value: str) -> str:
    """Create a stable graph node ID from a thesis-scoped value.

    Args:
        node_type: Type of the graph node.
        thesis_id: Canonical thesis ID.
        value: Value to be hashed for the node ID.

    Returns:
        str: Stable graph node ID.
    """
    digest = hashlib.sha256(f"{thesis_id}\0{node_type}\0{value}".encode()).hexdigest()[:16]
    return f"{node_type}:{digest}"


def _normalise_heading_path(metadata: Dict[str, Any]) -> str:
    """Return the most specific persisted structure path for a chunk.

    Args:
        metadata: Chunk metadata containing structural information.

    Returns:
        str: Most specific persisted structure path for the chunk.
    """
    return str(
        metadata.get("heading_path")
        or metadata.get("section_title")
        or metadata.get("chapter")
        or "Unclassified"
    ).strip()


def _normalise_text(value: str) -> str:
    """Normalise text for conservative entity-to-chunk matching."""
    return " ".join(value.lower().split())


def _find_entity_chunk_id(
    entity_text: str,
    chunks: List[Tuple[str, str, Dict[str, Any]]],
) -> str | None:
    """Return the first source chunk containing a deterministic entity string."""
    entity_text = _normalise_text(entity_text.removeprefix("[Heuristic] ").removeprefix("[LLM] "))
    for chunk_id, document, _ in chunks:
        if entity_text and entity_text in _normalise_text(document):
            return chunk_id
    return None


def _find_figure_reference_chunk_ids(
    figure: Dict[str, Any],
    chunks: List[Tuple[str, str, Dict[str, Any]]],
) -> List[str]:
    """Return body chunks with numbered references to a figure, excluding its caption.
    Args:
        figure: Figure metadata containing the figure number and source positions.
        chunks: List of tuples containing chunk ID, document text, and metadata.

    Returns:
        List[str]: Chunk IDs containing references to the figure, excluding its caption.
    """
    caption_number = str(figure.get("caption_number") or "").strip()
    if not re.fullmatch(r"(?:[A-Z]\.)?\d+(?:\.\d+)*", caption_number, re.IGNORECASE):
        return []

    pattern = re.compile(
        rf"\b(?:fig(?:ure)?s?)\.?\s*{re.escape(caption_number)}(?![\d.])",
        re.IGNORECASE,
    )
    caption_start = figure.get("source_start")
    caption_end = figure.get("source_end")
    reference_chunk_ids = []
    for chunk_id, document, metadata in chunks:
        if "list of figures" in _normalise_heading_path(metadata).lower():
            continue
        chunk_start = metadata.get("source_start")
        if not isinstance(chunk_start, int) or chunk_start < 0:
            continue
        for match in pattern.finditer(document):
            reference_start = chunk_start + match.start()
            reference_end = chunk_start + match.end()
            if (
                isinstance(caption_start, int)
                and isinstance(caption_end, int)
                and reference_start < caption_end
                and reference_end > caption_start
            ):
                continue
            reference_chunk_ids.append(chunk_id)
            break
    return reference_chunk_ids


def _figure_reference_status(
    figure: Dict[str, Any],
    reference_chunk_ids: List[str],
    chunks: List[Tuple[str, str, Dict[str, Any]]],
) -> str:
    """Describe explicit-reference scan coverage without asserting semantic absence.

    Args:
        figure: Figure metadata containing the figure number and source positions.
        reference_chunk_ids: List of chunk IDs containing references to the figure.
        chunks: List of tuples containing chunk ID, document text, and metadata.

    Returns:
        str: Status describing the explicit-reference scan coverage.
    """
    caption_number = str(figure.get("caption_number") or "").strip()
    if not re.fullmatch(r"(?:[A-Z]\.)?\d+(?:\.\d+)*", caption_number, re.IGNORECASE):
        return "caption_number_unavailable"
    if reference_chunk_ids:
        return "referenced"

    chunks_with_offsets = sum(
        isinstance(metadata.get("source_start"), int) and metadata["source_start"] >= 0
        for _, _, metadata in chunks
    )
    if not chunks_with_offsets:
        return "source_offsets_unavailable"
    if chunks_with_offsets < len(chunks):
        return "partial_source_offsets"
    return "no_explicit_reference_found"


def _normalise_figure_title(title: str) -> str:
    """Normalise figure titles for punctuation-insensitive comparison."""
    return re.sub(r"[\W_]+", " ", title.casefold()).strip()


def _reference_author_surnames(authors_json: str | None) -> set[str]:
    """Return normalised surnames from a citation-graph authors value.

    Args:
        authors_json (str | None): JSON-encoded list of authors from the citation graph.

    Returns:
        set[str]: Normalised surnames extracted from the authors list.
    """
    try:
        authors = json.loads(authors_json or "[]")
    except json.JSONDecodeError:
        authors = [authors_json] if authors_json else []
    if isinstance(authors, str):
        authors = [authors]
    if not isinstance(authors, list):
        return set()

    surnames = set()
    for author in authors:
        author_text = str(author).strip()
        if not author_text:
            continue
        surname = author_text.split(",", 1)[0] if "," in author_text else author_text.split()[-1]
        normalised_surname = re.sub(r"[^a-z]", "", surname.casefold())
        if normalised_surname:
            surnames.add(normalised_surname)
    return surnames


def _resolve_adapted_figure_references(
    figure: Dict[str, Any],
    thesis_id: str,
    citation_graph_path: Path | None,
    chunks: List[Tuple[str, str, Dict[str, Any]]] | None = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Resolve explicit author-year sources in adapted-figure captions conservatively.

    Args:
        figure (Dict[str, Any]): The figure dictionary containing metadata and caption.
        thesis_id (str): The node ID of the thesis in the citation graph.
        citation_graph_path (Path | None): Path to the SQLite citation graph database.

    Returns:
        Tuple[str, List[Dict[str, Any]]]: A tuple containing the resolution status and a list of resolved references.
    """
    caption = str(figure.get("caption") or "")
    adaptation_marker = _ADAPTED_FROM_PATTERN.search(caption)
    if adaptation_marker is None:
        return "not_adapted", []
    if citation_graph_path is None or not citation_graph_path.exists():
        return "citation_graph_unavailable", []

    caption_citations = caption[adaptation_marker.end() :]
    footnote_match = re.search(
        r"\bfootnotes?\s+(?:no\.?\s*)?(\d+)\b", caption_citations, re.IGNORECASE
    )
    if footnote_match:
        footnote_number = footnote_match.group(1)
        footnote_pattern = re.compile(
            rf"^\s*footnote\s+{re.escape(footnote_number)}\s*[:.)-]\s*(.+?)\s*$",
            re.IGNORECASE,
        )
        footnote_texts = []
        for _, document, metadata in chunks or []:
            if "list of figures" in _normalise_heading_path(metadata).casefold():
                continue
            for line in document.splitlines():
                match = footnote_pattern.match(line)
                if match and match.group(1).strip() not in footnote_texts:
                    footnote_texts.append(match.group(1).strip())
        if not footnote_texts:
            return "footnote_unresolved", []
        if len(footnote_texts) > 1:
            return "ambiguous", []
        caption_citations = footnote_texts[0]
    return _resolve_citation_references(caption_citations, thesis_id, citation_graph_path)


def _resolve_citation_references(
    citation_text: str,
    thesis_id: str,
    citation_graph_path: Path | None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Resolve explicit author-year and numbered markers to thesis references.

    Args:
        citation_text: Text containing explicit citation markers.
        thesis_id: Canonical thesis document ID in the citation graph.
        citation_graph_path: Path to the shared citation graph database.

    Returns:
        Resolution status and unique resolved reference records.
    """
    if citation_graph_path is None or not citation_graph_path.exists():
        return "citation_graph_unavailable", []

    citations = list(_AUTHOR_YEAR_PATTERN.finditer(citation_text))
    numbered_citations = list(_NUMBERED_CITATION_PATTERN.finditer(citation_text))
    if not citations and not numbered_citations:
        return "citation_format_unsupported", []

    try:
        with sqlite3.connect(citation_graph_path) as connection:
            rows = connection.execute(
                """
                SELECT reference.node_id, reference.title, reference.authors,
                       reference.year, reference.doi, reference.source
                FROM nodes AS reference
                JOIN edges ON edges.target = reference.node_id
                WHERE edges.source = ? AND reference.node_type = 'reference'
                  ORDER BY edges.rowid
                """,
                (thesis_id,),
            ).fetchall()
    except sqlite3.Error:
        return "citation_graph_unavailable", []

    resolved: Dict[str, Dict[str, Any]] = {}
    unresolved = False
    ambiguous = False

    def record_reference(row: tuple[Any, ...]) -> None:
        """Record a resolved reference in the resolved dictionary."""
        resolved[row[0]] = {
            "node_id": row[0],
            "title": row[1],
            "authors": json.loads(row[2] or "[]"),
            "year": row[3],
            "doi": row[4],
            "source": row[5],
        }

    for citation in citations:
        citation_surname = re.sub(r"[^a-z]", "", citation.group(1).casefold())
        citation_year = int(citation.group(2))
        matches = [
            row
            for row in rows
            if row[3] == citation_year and citation_surname in _reference_author_surnames(row[2])
        ]
        if len(matches) == 1:
            record_reference(matches[0])
        elif len(matches) > 1:
            ambiguous = True
        else:
            unresolved = True

    for citation in numbered_citations:
        reference_numbers: List[int] = []
        marker_resolved = True
        for part in re.split(r"[,;]", citation.group(1)):
            range_match = re.fullmatch(r"\s*(\d+)\s*[-\u2013]\s*(\d+)\s*", part)
            if range_match:
                range_start, range_end = map(int, range_match.groups())
                if range_start < 1 or range_end < range_start or range_end > len(rows):
                    marker_resolved = False
                    continue
                reference_numbers.extend(range(range_start, range_end + 1))
                continue

            number_match = re.fullmatch(r"\s*(\d+)\s*", part)
            if not number_match:
                marker_resolved = False
                continue
            reference_number = int(number_match.group(1))
            if reference_number < 1 or reference_number > len(rows):
                marker_resolved = False
                continue
            reference_numbers.append(reference_number)

        for reference_number in dict.fromkeys(reference_numbers):
            record_reference(rows[reference_number - 1])
        if not marker_resolved:
            unresolved = True

    if resolved and (unresolved or ambiguous):
        status = "partially_linked"
    elif resolved:
        status = "linked"
    elif ambiguous:
        status = "ambiguous"
    else:
        status = "unresolved"
    return status, list(resolved.values())


def _reconcile_figure_with_list(
    figure: Dict[str, Any],
    chunks: List[Tuple[str, str, Dict[str, Any]]],
) -> Tuple[str, str | None]:
    """Compare an explicit figure caption label and title with extracted list entries.

    Args:
        figure (Dict[str, Any]): The figure dictionary containing metadata and caption.
        chunks (List[Tuple[str, str, Dict[str, Any]]]): A list of document chunks, each represented as a tuple containing the document ID, the document text, and the associated metadata.

    Returns:
        Tuple[str, str | None]: A tuple containing the comparison status and the matching list entry title if available.
    """
    list_chunks = [
        document
        for _, document, metadata in chunks
        if "list of figures" in _normalise_heading_path(metadata).lower()
    ]
    if not list_chunks:
        return "list_not_detected", None

    caption_number = str(figure.get("caption_number") or "").strip()
    if not re.fullmatch(r"(?:[A-Z]\.)?\d+(?:\.\d+)*", caption_number, re.IGNORECASE):
        return "caption_number_unavailable", None

    entry_pattern = re.compile(
        r"^\s*(?:fig(?:ure)?s?)\.?\s*((?:[A-Z]\.)?\d+(?:\.\d+)*)" r"(?:\s*[:.)-]\s*|\s+)(.+?)\s*$",
        re.IGNORECASE,
    )
    entries: Dict[str, List[str]] = {}
    current_entry_number = None
    for document in list_chunks:
        for source_line in document.splitlines():
            line = source_line
            stripped_line = source_line.strip()
            if stripped_line.startswith("|") and stripped_line.endswith("|"):
                cells = [cell.strip() for cell in stripped_line.strip("|").split("|")]
                if not cells or all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells if cell):
                    continue
                first_cell = cells[0] if cells else ""
                if re.match(r"^fig(?:ure)?s?\.?\s+", first_cell, re.IGNORECASE):
                    title_cells = cells[1:]
                    if len(cells) >= 3 and title_cells and re.fullmatch(r"\d+", title_cells[-1]):
                        title_cells = title_cells[:-1]
                    line = f"{first_cell} {' '.join(title_cells)}"
                else:
                    line = " ".join(cell for cell in cells if cell)
            match = entry_pattern.match(line)
            if match:
                current_entry_number = match.group(1).casefold()
                title = re.sub(r"\s*\.{2,}\s*\d+\s*$", "", match.group(2)).strip(" .:-")
                entries.setdefault(current_entry_number, []).append(title)
                continue
            if current_entry_number is None:
                continue
            continuation = re.sub(r"\s*\.{2,}\s*\d+\s*$", "", line).strip(" .:-")
            if not continuation or re.fullmatch(r"\d+", continuation):
                continue
            entries[current_entry_number][
                -1
            ] = f"{entries[current_entry_number][-1]} {continuation}".strip()

    if not entries:
        return "entries_not_detected", None

    matching_titles = entries.get(caption_number.casefold(), [])
    if not matching_titles:
        return "not_listed", None
    if len(matching_titles) > 1:
        return "ambiguous", matching_titles[0]

    caption = str(figure.get("caption") or "")
    adaptation_marker = _ADAPTED_FROM_PATTERN.search(caption)
    if adaptation_marker is not None:
        caption = caption[: adaptation_marker.start()].rstrip(" (:;-.")
    caption_title = re.sub(
        rf"^\s*(?:fig(?:ure)?s?)\.?\s*{re.escape(caption_number)}(?:\s*[:.)-]\s*|\s+)",
        "",
        caption,
        flags=re.IGNORECASE,
    )
    status = (
        "matched"
        if _normalise_figure_title(caption_title) == _normalise_figure_title(matching_titles[0])
        else "title_mismatch"
    )
    return status, matching_titles[0]


def _assessment_section_type(metadata: Dict[str, Any]) -> str | None:
    """Classify persisted section metadata for assessment evidence nodes."""
    heading = _normalise_heading_path(metadata).lower()
    if any(term in heading for term in ("results", "findings", "discussion")):
        return "finding"
    if "conclusion" in heading:
        return "conclusion"
    return None


def _find_research_question_evidence(
    question: str,
    chunks: List[Tuple[str, str, Dict[str, Any]]],
    assessor: PhDQualityAssessor,
) -> Dict[str, float]:
    """Return finding/conclusion chunks meeting the assessor's RQ overlap threshold.
    Args:
        question: Research question text.
        chunks: List of document chunks with their IDs and metadata.
        assessor: PhDQualityAssessor instance for tokenisation and assessment.

    Returns:
        Dict[str, float]: Mapping of chunk IDs to their match ratios with the research question.
    """
    key_terms = [
        word.lower()
        for word in assessor._tokenise_words(question)
        if len(word) > 4
        and word.lower() not in {"research", "question", "hypothesis", "study", "thesis"}
    ]
    if not key_terms:
        return {}

    evidence: Dict[str, float] = {}
    for chunk_id, document, metadata in chunks:
        if _assessment_section_type(metadata) not in {"finding", "conclusion"}:
            continue
        document_lower = document.lower()
        match_ratio = sum(1 for term in key_terms if term in document_lower) / len(key_terms)
        if match_ratio >= 0.4:
            evidence[chunk_id] = match_ratio
    return evidence


def _find_readiness_source_chunk_ids(
    evidence: List[str],
    source_sections: List[str],
    chunks: List[Tuple[str, str, Dict[str, Any]]],
) -> List[str]:
    """Return source chunk IDs for deterministic readiness evidence.

    Args:
        evidence: List of evidence strings to match against document chunks.
        source_sections: List of source section headings to match against document chunks.
        chunks: List of document chunks with their IDs and metadata.

    Returns:
        List[str]: List of source chunk IDs that match the evidence or source sections.
    """
    source_chunk_ids: List[str] = []
    normalised_evidence = [_normalise_text(item) for item in evidence if item]
    normalised_sections = {_normalise_text(item) for item in source_sections if item}
    for chunk_id, document, metadata in chunks:
        normalised_document = _normalise_text(document)
        heading_path = _normalise_text(_normalise_heading_path(metadata))
        if (
            any(item in normalised_document for item in normalised_evidence)
            or heading_path in normalised_sections
        ):
            source_chunk_ids.append(chunk_id)
    return source_chunk_ids


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create the thesis evidence graph schema.

    Args:
        connection: SQLite database connection.

    Returns:
        None
    """
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            thesis_id TEXT NOT NULL,
            label TEXT NOT NULL,
            source_start INTEGER,
            source_end INTEGER,
            sequence_number INTEGER,
            attributes_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS edges (
            source_node_id TEXT NOT NULL,
            target_node_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            PRIMARY KEY (source_node_id, target_node_id, relation),
            FOREIGN KEY (source_node_id) REFERENCES nodes(node_id),
            FOREIGN KEY (target_node_id) REFERENCES nodes(node_id)
        );

        CREATE TABLE IF NOT EXISTS figure_assets (
            figure_node_id TEXT PRIMARY KEY,
            media_type TEXT NOT NULL,
            image_bytes BLOB NOT NULL,
            FOREIGN KEY (figure_node_id) REFERENCES nodes(node_id)
        );

        CREATE INDEX IF NOT EXISTS idx_nodes_thesis_type ON nodes(thesis_id, node_type);
        CREATE INDEX IF NOT EXISTS idx_nodes_sequence ON nodes(thesis_id, sequence_number);
        CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_id);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_id);
        """)


def _load_existing_figure_assets(graph_path: Path) -> List[Dict[str, Any]]:
    """Read figure nodes and image blobs before atomically rebuilding a thesis graph.

    Args:
        graph_path: Path to the SQLite database containing the thesis graph.

    Returns:
        A list of dictionaries representing figure assets, each containing
        the figure ID, media type, and image bytes.
    """
    if not graph_path.exists():
        return []
    try:
        with sqlite3.connect(graph_path) as connection:
            rows = connection.execute("""
                SELECT nodes.node_id, nodes.attributes_json,
                       figure_assets.media_type, figure_assets.image_bytes
                FROM nodes
                JOIN figure_assets ON figure_assets.figure_node_id = nodes.node_id
                WHERE nodes.node_type = 'figure'
                ORDER BY nodes.sequence_number, nodes.node_id
                """).fetchall()
    except sqlite3.Error:
        return []

    figures = []
    for node_id, attributes_json, media_type, image_bytes in rows:
        attributes = json.loads(attributes_json or "{}")
        attributes.update(
            figure_id=attributes.get("figure_id") or node_id,
            image_bytes=image_bytes,
            media_type=media_type,
        )
        figures.append(attributes)
    return figures


def build_thesis_evidence_graph(
    collection: ChunkCollection,
    thesis_id: str,
    output_path: Path,
    readiness: Any | None = None,
    structure_analysis: Any | None = None,
    cultural_lens_profile: Dict[str, Any] | None = None,
    figures: List[Dict[str, Any]] | None = None,
    citation_graph_path: Path | None = None,
) -> ThesisEvidenceGraph:
    """Build and persist a thesis hierarchy from chapter-aware child chunks.

    Args:
        collection: ChromaDB-compatible chunk collection.
        thesis_id: Canonical thesis ID used during ingestion.
        output_path: SQLite database path to replace atomically.
        readiness: Optional pre-computed examiner readiness analysis. When not
            supplied, the deterministic assessor computes it from the chunks.
        structure_analysis: Optional pre-computed structure/RQ analysis. When supplied,
            its reconciled inquiry aliases and source mappings are persisted.
        cultural_lens_profile: Optional validated profile assigned to this thesis.
        figures: Optional Docling figure assets extracted from the thesis PDF.
        citation_graph_path: Optional shared citation graph used to resolve adapted figures.

    Returns:
        Statistics and output location for the completed graph.

    Raises:
        ValueError: When no chapter-aware child chunks are found.
    """
    records = collection.get(
        where={"$and": [{"thesis_id": thesis_id}, {"chunk_type": "child"}]},
        include=["documents", "metadatas"],
        limit=50000,
    )
    record_ids = records.get("ids", [])
    documents = records.get("documents", [])
    metadatas = records.get("metadatas", [])
    chunks: List[Tuple[str, str, Dict[str, Any]]] = []

    for record_id, document, metadata in zip(record_ids, documents, metadatas):
        if not isinstance(metadata, dict) or not metadata.get("chapter"):
            continue
        chunks.append((str(record_id), str(document), metadata))

    if not chunks:
        raise ValueError(f"No chapter-aware child chunks found for thesis_id: {thesis_id}")

    figure_records = collection.get(
        where={"$and": [{"thesis_id": thesis_id}, {"chunk_type": "figure"}]},
        include=["documents", "metadatas"],
        limit=50000,
    )
    figure_chunks = [
        (str(record_id), str(document), metadata)
        for record_id, document, metadata in zip(
            figure_records.get("ids", []),
            figure_records.get("documents", []),
            figure_records.get("metadatas", []),
        )
        if isinstance(metadata, dict)
        and metadata.get("thesis_id") == thesis_id
        and metadata.get("chunk_type") == "figure"
    ]

    chunks.sort(key=lambda item: (int(item[2].get("sequence_number", 0)), item[0]))
    figures = figures if figures is not None else _load_existing_figure_assets(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    node_count = 0
    edge_count = 0
    thesis_node_id = _node_id("thesis", thesis_id, thesis_id)
    chapters: Dict[str, str] = {}
    sections: Dict[str, str] = {}
    section_by_chunk_id: Dict[str, str] = {}
    evidence_node_by_chunk_id: Dict[str, str] = {}
    figure_node_by_id: Dict[str, str] = {}
    figure_node_by_number: Dict[str, str] = {}
    persisted_reference_node_ids = set()
    progress_interval = max(1, (len(chunks) + 99) // 100)
    _report_progress_checkpoint("thesis_graph_chunks", 0, len(chunks))

    with sqlite3.connect(temporary_path) as connection:
        _ensure_schema(connection)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute(
            "INSERT INTO nodes (node_id, node_type, thesis_id, label, attributes_json) VALUES (?, ?, ?, ?, ?)",
            (thesis_node_id, "thesis", thesis_id, thesis_id, json.dumps({"thesis_id": thesis_id})),
        )
        node_count += 1

        for chunk_index, (chunk_id, document, metadata) in enumerate(chunks, start=1):
            chapter_label = str(metadata["chapter"]).strip()
            chapter_node_id = chapters.get(chapter_label)
            if chapter_node_id is None:
                chapter_node_id = _node_id("chapter", thesis_id, chapter_label)
                chapters[chapter_label] = chapter_node_id
                connection.execute(
                    "INSERT OR IGNORE INTO nodes (node_id, node_type, thesis_id, label, attributes_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        chapter_node_id,
                        "chapter",
                        thesis_id,
                        chapter_label,
                        json.dumps({"chapter": chapter_label}),
                    ),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO edges VALUES (?, ?, ?)",
                    (thesis_node_id, chapter_node_id, "contains"),
                )
                node_count += 1
                edge_count += 1

            heading_path = _normalise_heading_path(metadata)
            section_key = f"{chapter_label}\0{heading_path}"
            section_node_id = sections.get(section_key)
            if section_node_id is None:
                section_node_id = _node_id("section", thesis_id, section_key)
                sections[section_key] = section_node_id
                connection.execute(
                    "INSERT OR IGNORE INTO nodes (node_id, node_type, thesis_id, label, attributes_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        section_node_id,
                        "section",
                        thesis_id,
                        heading_path,
                        json.dumps({"chapter": chapter_label, "heading_path": heading_path}),
                    ),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO edges VALUES (?, ?, ?)",
                    (chapter_node_id, section_node_id, "contains"),
                )
                node_count += 1
                edge_count += 1

            evidence_node_id = f"chunk:{chunk_id}"
            section_by_chunk_id[chunk_id] = section_node_id
            evidence_node_by_chunk_id[chunk_id] = evidence_node_id
            connection.execute(
                "INSERT OR REPLACE INTO nodes (node_id, node_type, thesis_id, label, source_start, source_end, sequence_number, attributes_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    evidence_node_id,
                    "chunk",
                    thesis_id,
                    document[:160],
                    metadata.get("source_start"),
                    metadata.get("source_end"),
                    metadata.get("sequence_number"),
                    json.dumps(
                        {
                            "chapter": chapter_label,
                            "heading_path": heading_path,
                            "chroma_chunk_id": chunk_id,
                        }
                    ),
                ),
            )
            connection.execute(
                "INSERT OR IGNORE INTO edges VALUES (?, ?, ?)",
                (section_node_id, evidence_node_id, "contains"),
            )
            node_count += 1
            edge_count += 1
            if chunk_index % progress_interval == 0 or chunk_index == len(chunks):
                _report_progress_checkpoint("thesis_graph_chunks", chunk_index, len(chunks))

        citation_reference_ids_by_chunk_id: Dict[str, set[str]] = {}
        for chunk_id, document, _ in chunks:
            _, cited_references = _resolve_citation_references(
                document, thesis_id, citation_graph_path
            )
            cited_reference_ids = set()
            for reference in cited_references:
                reference_node_id = reference["node_id"]
                cited_reference_ids.add(reference_node_id)
                if reference_node_id not in persisted_reference_node_ids:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO nodes
                            (node_id, node_type, thesis_id, label, attributes_json)
                        VALUES (?, 'reference', ?, ?, ?)
                        """,
                        (
                            reference_node_id,
                            thesis_id,
                            reference.get("title") or reference_node_id,
                            json.dumps(reference),
                        ),
                    )
                    if connection.execute("SELECT changes()").fetchone()[0]:
                        node_count += 1
                    persisted_reference_node_ids.add(reference_node_id)
                connection.execute(
                    "INSERT OR IGNORE INTO edges VALUES (?, ?, ?)",
                    (evidence_node_by_chunk_id[chunk_id], reference_node_id, "cites"),
                )
                if connection.execute("SELECT changes()").fetchone()[0]:
                    edge_count += 1
            citation_reference_ids_by_chunk_id[chunk_id] = cited_reference_ids

        for figure in figures:
            figure_number = figure.get("figure_number")
            page_number = figure.get("page_number")
            figure_id = str(
                figure.get("figure_id")
                or (
                    f"figure_{figure_number}"
                    if figure_number is not None
                    else f"page-{page_number or 'unknown'}-figure-{len(figures)}"
                )
            )
            figure_node_id = _node_id("figure", thesis_id, figure_id)
            figure_node_by_id[figure_id] = figure_node_id
            if figure_number is not None:
                figure_node_by_number[str(figure_number)] = figure_node_id
            body_reference_chunk_ids = _find_figure_reference_chunk_ids(figure, chunks)
            body_reference_status = _figure_reference_status(
                figure, body_reference_chunk_ids, chunks
            )
            list_status, list_title = _reconcile_figure_with_list(figure, chunks)
            adapted_reference_status, adapted_references = _resolve_adapted_figure_references(
                figure, thesis_id, citation_graph_path, chunks=chunks
            )
            caption = str(figure.get("caption") or "").strip()
            alt_text = str(figure.get("alt_text") or "").strip()
            label = caption or alt_text or f"Figure {figure_number or '?'}"
            figure_attributes = {
                "figure_id": figure_id,
                "figure_number": figure_number,
                "kind": figure.get("kind", "picture"),
                "chapter": figure.get("chapter"),
                "section_title": figure.get("section_title"),
                "heading_path": figure.get("heading_path"),
                "page_number": page_number,
                "bbox": figure.get("bbox"),
                "source_start": figure.get("source_start"),
                "source_end": figure.get("source_end"),
                "caption": caption,
                "caption_number": figure.get("caption_number"),
                "alt_text": alt_text,
                "vision_status": figure.get("vision_status", "pending"),
                "description": figure.get("description", ""),
                "vision_assessment": figure.get("vision_assessment"),
                "body_reference_chunk_ids": body_reference_chunk_ids,
                "body_reference_status": body_reference_status,
                "list_of_figures_status": list_status,
                "list_of_figures_title": list_title,
                "adapted_reference_status": adapted_reference_status,
                "adapted_reference_ids": [reference["node_id"] for reference in adapted_references],
            }
            connection.execute(
                """
                INSERT OR REPLACE INTO nodes
                    (node_id, node_type, thesis_id, label, sequence_number, attributes_json)
                VALUES (?, 'figure', ?, ?, ?, ?)
                """,
                (
                    figure_node_id,
                    thesis_id,
                    label,
                    int(figure_number) if figure_number is not None else None,
                    json.dumps(figure_attributes),
                ),
            )
            figure_section_key = f"{figure.get('chapter', '')}\0{figure.get('heading_path', '')}"
            container_node_id = sections.get(figure_section_key, thesis_node_id)
            connection.execute(
                "INSERT OR IGNORE INTO edges VALUES (?, ?, ?)",
                (container_node_id, figure_node_id, "contains"),
            )
            for chunk_id in body_reference_chunk_ids:
                figure_evidence_node_id = evidence_node_by_chunk_id.get(chunk_id)
                if figure_evidence_node_id is None:
                    continue
                connection.execute(
                    "INSERT OR IGNORE INTO edges VALUES (?, ?, ?)",
                    (figure_evidence_node_id, figure_node_id, "references"),
                )
                edge_count += 1
            for reference in adapted_references:
                reference_node_id = reference["node_id"]
                if reference_node_id not in persisted_reference_node_ids:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO nodes
                            (node_id, node_type, thesis_id, label, attributes_json)
                        VALUES (?, 'reference', ?, ?, ?)
                        """,
                        (
                            reference_node_id,
                            thesis_id,
                            reference.get("title") or reference_node_id,
                            json.dumps(reference),
                        ),
                    )
                    if connection.execute("SELECT changes()").fetchone()[0]:
                        node_count += 1
                    persisted_reference_node_ids.add(reference_node_id)
                connection.execute(
                    "INSERT OR IGNORE INTO edges VALUES (?, ?, ?)",
                    (figure_node_id, reference_node_id, "adapted_from"),
                )
                if connection.execute("SELECT changes()").fetchone()[0]:
                    edge_count += 1
            image_bytes = figure.get("image_bytes")
            image = figure.get("image")
            if image_bytes is None and image is not None:
                image_buffer = io.BytesIO()
                image.save(image_buffer, format="PNG")
                image_bytes = image_buffer.getvalue()
            if image_bytes:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO figure_assets
                        (figure_node_id, media_type, image_bytes)
                    VALUES (?, ?, ?)
                    """,
                    (figure_node_id, figure.get("media_type", "image/png"), image_bytes),
                )
            node_count += 1
            edge_count += 1

        for chunk_id, document, metadata in figure_chunks:
            figure_chunk_figure_node_id = figure_node_by_id.get(
                str(metadata.get("figure_id") or "")
            )
            if figure_chunk_figure_node_id is None and metadata.get("figure_number") is not None:
                figure_chunk_figure_node_id = figure_node_by_number.get(
                    str(metadata["figure_number"])
                )
            if figure_chunk_figure_node_id is None:
                continue

            figure_chunk_node_id = f"figure_chunk:{chunk_id}"
            connection.execute(
                """
                INSERT OR REPLACE INTO nodes
                    (node_id, node_type, thesis_id, label, sequence_number, attributes_json)
                VALUES (?, 'figure_chunk', ?, ?, ?, ?)
                """,
                (
                    figure_chunk_node_id,
                    thesis_id,
                    document[:160],
                    metadata.get("sequence_number"),
                    json.dumps(
                        {
                            "chroma_chunk_id": chunk_id,
                            "figure_id": metadata.get("figure_id"),
                            "figure_number": metadata.get("figure_number"),
                            "page_number": metadata.get("page_number"),
                        }
                    ),
                ),
            )
            connection.execute(
                "INSERT OR IGNORE INTO edges VALUES (?, ?, ?)",
                (figure_chunk_figure_node_id, figure_chunk_node_id, "described_by"),
            )
            node_count += 1
            edge_count += 1

        chunks_data = {
            "ids": [chunk_id for chunk_id, _, _ in chunks],
            "documents": [document for _, document, _ in chunks],
            "metadatas": [metadata for _, _, metadata in chunks],
        }
        assessor = PhDQualityAssessor(
            chunk_collection=collection,
            cultural_lens_profile=cultural_lens_profile,
        )
        structure = structure_analysis or assessor.analyse_structure(chunks_data)
        claims = assessor.analyse_claims_and_contradictions(chunks_data)
        methodology = assessor.validate_methodology_checklist(chunks_data)
        readiness = readiness or assessor.analyse_examiner_readiness(chunks_data)
        research_question_evidence = {
            question: _find_research_question_evidence(question, chunks, assessor)
            for question in structure.research_questions
        }

        assessment_entities: List[Tuple[str, str, str, str, Dict[str, Any]]] = []
        for question in structure.research_questions:
            matching_chunks = research_question_evidence[question]
            assessment_entities.append(
                (
                    "research_question",
                    question,
                    "states",
                    "research_question",
                    {
                        "text": question,
                        "inquiry_id": structure.research_inquiry_ids.get(question),
                        "parent_inquiry_id": structure.research_inquiry_parent_ids.get(question),
                        "inquiry_type": structure.research_inquiry_types.get(
                            question, "research_question"
                        ),
                        "restatements": structure.research_inquiry_aliases.get(question, []),
                        "evidence_sources": structure.research_inquiry_sources.get(question, []),
                        "alignment_status": "addressed" if matching_chunks else "unaddressed",
                        "alignment_score": max(matching_chunks.values(), default=0.0),
                        "evidence_chunk_ids": list(matching_chunks),
                    },
                )
            )
        for claim in claims.claims:
            assessment_entities.append(("claim", claim, "contains_claim", "claim", {"text": claim}))
        for item, evidence in methodology.evidence.items():
            if not methodology.items.get(item):
                continue
            assessment_entities.append(
                (
                    "method",
                    item,
                    "evidences",
                    "method",
                    {"item": item, "evidence": evidence},
                )
            )

        entity_nodes: Dict[Tuple[str, str], str] = {}
        for node_type, label, relation, _, attributes in assessment_entities:
            key = (node_type, label)
            entity_node_id = entity_nodes.get(key)
            if entity_node_id is None:
                entity_node_id = _node_id(node_type, thesis_id, label)
                entity_nodes[key] = entity_node_id
                connection.execute(
                    "INSERT OR IGNORE INTO nodes (node_id, node_type, thesis_id, label, attributes_json) VALUES (?, ?, ?, ?, ?)",
                    (entity_node_id, node_type, thesis_id, label, json.dumps(attributes)),
                )
                node_count += 1

            if node_type == "research_question":
                source_chunk_ids = [
                    str(source["chunk_id"])
                    for source in structure.research_inquiry_sources.get(label, [])
                    if source.get("chunk_id")
                ]
            else:
                source_chunk_id = _find_entity_chunk_id(label, chunks)
                source_chunk_ids = [source_chunk_id] if source_chunk_id else []
            for source_chunk_id in source_chunk_ids:
                if source_chunk_id not in evidence_node_by_chunk_id:
                    continue
                connection.execute(
                    "INSERT OR IGNORE INTO edges VALUES (?, ?, ?)",
                    (evidence_node_by_chunk_id[source_chunk_id], entity_node_id, relation),
                )
                edge_count += 1
            if node_type == "claim":
                _, claim_references = _resolve_citation_references(
                    label, thesis_id, citation_graph_path
                )
                claim_reference_ids = {reference["node_id"] for reference in claim_references}
                for source_chunk_id in source_chunk_ids:
                    for (
                        reference_node_id
                    ) in claim_reference_ids & citation_reference_ids_by_chunk_id.get(
                        source_chunk_id, set()
                    ):
                        connection.execute(
                            "INSERT OR IGNORE INTO edges VALUES (?, ?, ?)",
                            (entity_node_id, reference_node_id, "supported_by_citation"),
                        )
                        if connection.execute("SELECT changes()").fetchone()[0]:
                            edge_count += 1

        for chunk_id, document, metadata in chunks:
            entity_type = _assessment_section_type(metadata)
            if entity_type is None:
                continue
            label = _normalise_heading_path(metadata)
            entity_node_id = _node_id(entity_type, thesis_id, label)
            if (entity_type, label) not in entity_nodes:
                entity_nodes[(entity_type, label)] = entity_node_id
                connection.execute(
                    "INSERT OR IGNORE INTO nodes (node_id, node_type, thesis_id, label, attributes_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        entity_node_id,
                        entity_type,
                        thesis_id,
                        label,
                        json.dumps({"heading_path": label}),
                    ),
                )
                node_count += 1
            connection.execute(
                "INSERT OR IGNORE INTO edges VALUES (?, ?, ?)",
                (evidence_node_by_chunk_id[chunk_id], entity_node_id, "evidences"),
            )
            edge_count += 1

        chunk_metadata_by_id = {chunk_id: metadata for chunk_id, _, metadata in chunks}
        for question, matching_chunks in research_question_evidence.items():
            question_node_id = entity_nodes.get(("research_question", question))
            if question_node_id is None or not matching_chunks:
                continue
            for chunk_id in matching_chunks:
                metadata = chunk_metadata_by_id[chunk_id]
                section_type = _assessment_section_type(metadata)
                if section_type is None:
                    continue
                section_label = _normalise_heading_path(metadata)
                target_node_id = entity_nodes.get((section_type, section_label))
                if target_node_id is None:
                    continue
                connection.execute(
                    "INSERT OR IGNORE INTO edges VALUES (?, ?, ?)",
                    (question_node_id, target_node_id, "addressed_by"),
                )
                edge_count += 1

        inquiry_node_ids = {
            structure.research_inquiry_ids.get(question): entity_nodes[
                ("research_question", question)
            ]
            for question in structure.research_questions
            if structure.research_inquiry_ids.get(question)
            and ("research_question", question) in entity_nodes
        }
        for question, parent_id in structure.research_inquiry_parent_ids.items():
            question_node_id = entity_nodes.get(("research_question", question))
            parent_node_id = inquiry_node_ids.get(parent_id)
            if question_node_id and parent_node_id and question_node_id != parent_node_id:
                connection.execute(
                    "INSERT OR IGNORE INTO edges VALUES (?, ?, ?)",
                    (question_node_id, parent_node_id, "sub_question_of"),
                )
                edge_count += 1

        for criterion in readiness.criteria:
            criterion_node_id = _node_id("readiness_criterion", thesis_id, criterion.criterion)
            connection.execute(
                "INSERT OR IGNORE INTO nodes (node_id, node_type, thesis_id, label, attributes_json) VALUES (?, ?, ?, ?, ?)",
                (
                    criterion_node_id,
                    "readiness_criterion",
                    thesis_id,
                    criterion.criterion,
                    json.dumps(
                        {
                            "status": criterion.status,
                            "confidence": criterion.confidence,
                            "reason": criterion.reason,
                            "source": criterion.source,
                            "source_sections": criterion.source_sections,
                        }
                    ),
                ),
            )
            node_count += 1
            for source_chunk_id in _find_readiness_source_chunk_ids(
                criterion.evidence,
                criterion.source_sections,
                chunks,
            ):
                connection.execute(
                    "INSERT OR IGNORE INTO edges VALUES (?, ?, ?)",
                    (criterion_node_id, evidence_node_by_chunk_id[source_chunk_id], "supported_by"),
                )
                edge_count += 1

        connection.executemany(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            [
                ("thesis_id", thesis_id),
                ("built_at", datetime.now(timezone.utc).isoformat()),
                ("node_count", str(node_count)),
                ("edge_count", str(edge_count)),
                ("graph_type", "thesis_evidence_graph"),
            ],
        )
        connection.commit()

    temporary_path.replace(output_path)
    _report_progress_checkpoint("thesis_graph_complete", 1, 1)
    return ThesisEvidenceGraph(thesis_id, output_path, node_count, edge_count)


def get_readiness_criterion_sources(graph_path: Path, thesis_id: str) -> Dict[str, List[str]]:
    """Return graph-linked evidence chunks for each examiner-readiness criterion."""
    if not graph_path.exists():
        raise FileNotFoundError(f"Thesis evidence graph not found: {graph_path}")

    with sqlite3.connect(graph_path) as connection:
        rows = connection.execute(
            """
            SELECT criterion.label, evidence.attributes_json
            FROM nodes AS criterion
            LEFT JOIN edges
              ON edges.source_node_id = criterion.node_id
             AND edges.relation = 'supported_by'
            LEFT JOIN nodes AS evidence ON evidence.node_id = edges.target_node_id
            WHERE criterion.thesis_id = ?
              AND criterion.node_type = 'readiness_criterion'
            ORDER BY criterion.label, evidence.sequence_number
            """,
            (thesis_id,),
        ).fetchall()

    sources: Dict[str, List[str]] = {}
    for criterion_name, attributes_json in rows:
        if attributes_json is None:
            sources.setdefault(criterion_name, [])
            continue
        attributes = json.loads(attributes_json)
        chunk_id = attributes.get("chroma_chunk_id")
        if chunk_id:
            sources.setdefault(criterion_name, []).append(chunk_id)
    return sources


def get_chunk_provenance_paths(
    graph_path: Path,
    thesis_id: str,
    chunk_ids: List[str],
) -> Dict[str, List[str]]:
    """Return section-to-claim-to-reference paths for thesis evidence chunks.
    Args:
        graph_path (Path): Path to the SQLite graph database.
        thesis_id (str): ID of the thesis to query.
        chunk_ids (List[str]): List of chunk IDs to retrieve provenance paths for.

    Returns:
        Dict[str, List[str]]: Mapping from chunk IDs to their section-to-claim-to-reference paths.
    """
    if not graph_path.exists() or not chunk_ids:
        return {}

    chunk_node_ids = [f"chunk:{chunk_id}" for chunk_id in dict.fromkeys(chunk_ids)]
    placeholders = ", ".join("?" for _ in chunk_node_ids)
    with sqlite3.connect(graph_path) as connection:
        rows = connection.execute(
            f"""
            SELECT evidence.node_id, chapter.label, section.label, claim.label,
                   reference.label, reference.attributes_json
            FROM nodes AS evidence
            LEFT JOIN edges AS section_edge
              ON section_edge.target_node_id = evidence.node_id
             AND section_edge.relation = 'contains'
            LEFT JOIN nodes AS section ON section.node_id = section_edge.source_node_id
            LEFT JOIN edges AS chapter_edge
              ON chapter_edge.target_node_id = section.node_id
             AND chapter_edge.relation = 'contains'
            LEFT JOIN nodes AS chapter ON chapter.node_id = chapter_edge.source_node_id
            LEFT JOIN edges AS claim_edge
              ON claim_edge.source_node_id = evidence.node_id
             AND claim_edge.relation = 'contains_claim'
            LEFT JOIN nodes AS claim ON claim.node_id = claim_edge.target_node_id
            LEFT JOIN edges AS citation_edge
              ON citation_edge.source_node_id = claim.node_id
             AND citation_edge.relation = 'supported_by_citation'
            LEFT JOIN nodes AS reference ON reference.node_id = citation_edge.target_node_id
            WHERE evidence.thesis_id = ? AND evidence.node_id IN ({placeholders})
            ORDER BY evidence.sequence_number, claim.label, reference.label
            """,
            [thesis_id, *chunk_node_ids],
        ).fetchall()

    paths_by_chunk: Dict[str, List[str]] = {}
    for (
        node_id,
        chapter_label,
        section_label,
        claim_label,
        reference_label,
        attributes_json,
    ) in rows:
        metadata = json.loads(attributes_json or "{}") if reference_label else {}
        if section_label and chapter_label and section_label.startswith(chapter_label):
            location_parts = [section_label]
        else:
            location_parts = [label for label in (chapter_label, section_label) if label]

        path_parts = location_parts
        if claim_label:
            claim_path = f'claim "{claim_label}"'
            if reference_label:
                reference_details = [
                    str(metadata[field])
                    for field in ("year", "venue_rank", "link_status")
                    if metadata.get(field) is not None
                ]
                reference = str(reference_label)
                if reference_details:
                    reference += f" ({', '.join(reference_details)})"
                claim_path += f" -> cites {reference}"
            path_parts.append(claim_path)

        if path_parts:
            chunk_id = node_id.removeprefix("chunk:")
            paths_by_chunk.setdefault(chunk_id, [])
            path = " > ".join(path_parts)
            if path not in paths_by_chunk[chunk_id]:
                paths_by_chunk[chunk_id].append(path)
    return paths_by_chunk


def expand_evidence_chunk_ids(
    graph_path: Path,
    thesis_id: str,
    seed_chunk_ids: List[str],
    max_chunks: int = 3,
) -> List[str]:
    """Return thesis-local evidence chunks sharing a seed's section or chapter.

    Results prioritise chunks in the same section, then the containing chapter,
    then chunks connected through shared assessment entities. Seed chunks are
    excluded because retrieval already includes them.

    Args:
        graph_path: Path to the persisted thesis evidence graph.
        thesis_id: Canonical thesis ID.
        seed_chunk_ids: List of seed chunk IDs to expand from.
        max_chunks: Maximum number of additional chunks to return.

    Returns:
        List[str]: List of expanded evidence chunk IDs, prioritising section-level matches first.
    """
    if max_chunks < 1 or not graph_path.exists() or not seed_chunk_ids:
        return []

    seed_node_ids = [f"chunk:{chunk_id}" for chunk_id in seed_chunk_ids]
    placeholders = ", ".join("?" for _ in seed_node_ids)

    candidates: Dict[str, Tuple[int, int | None]] = {}
    with sqlite3.connect(graph_path) as connection:
        structural_rows = connection.execute(
            f"""
            WITH seed_sections AS (
                SELECT edges.source_node_id AS section_node_id
                FROM edges
                JOIN nodes ON nodes.node_id = edges.target_node_id
                WHERE edges.relation = 'contains'
                  AND nodes.thesis_id = ?
                  AND edges.target_node_id IN ({placeholders})
            ), candidates AS (
                SELECT chunk_nodes.node_id, chunk_nodes.sequence_number,
                       CASE WHEN section_edges.source_node_id IN (SELECT section_node_id FROM seed_sections)
                            THEN 0 ELSE 1 END AS priority
                FROM nodes AS chunk_nodes
                JOIN edges AS section_edges
                  ON section_edges.target_node_id = chunk_nodes.node_id
                 AND section_edges.relation = 'contains'
                JOIN edges AS chapter_edges
                  ON chapter_edges.target_node_id = section_edges.source_node_id
                 AND chapter_edges.relation = 'contains'
                WHERE chunk_nodes.thesis_id = ?
                  AND chunk_nodes.node_type = 'chunk'
                  AND chunk_nodes.node_id NOT IN ({placeholders})
                  AND chapter_edges.source_node_id IN (
                      SELECT chapter_edges.source_node_id
                      FROM edges AS chapter_edges
                      WHERE chapter_edges.target_node_id IN (SELECT section_node_id FROM seed_sections)
                        AND chapter_edges.relation = 'contains'
                  )
            )
            SELECT node_id, sequence_number, priority
            FROM candidates
            ORDER BY priority, sequence_number
            LIMIT ?
            """,
            [thesis_id, *seed_node_ids, thesis_id, *seed_node_ids, max_chunks],
        ).fetchall()

        for node_id, sequence_number, priority in structural_rows:
            candidates[node_id] = (priority, sequence_number)

        frontier = set(seed_node_ids)
        visited = set(seed_node_ids)
        reachable_entities = set()
        structural_node_types = {"thesis", "chapter", "section", "chunk"}

        for _ in range(2):
            if not frontier:
                break
            frontier_placeholders = ", ".join("?" for _ in frontier)
            adjacent_edges = connection.execute(
                f"""
                SELECT source_node_id, target_node_id
                FROM edges
                WHERE source_node_id IN ({frontier_placeholders})
                   OR target_node_id IN ({frontier_placeholders})
                """,
                [*frontier, *frontier],
            ).fetchall()
            neighbours = {
                target_node_id if source_node_id in frontier else source_node_id
                for source_node_id, target_node_id in adjacent_edges
            } - visited
            if not neighbours:
                break

            neighbour_placeholders = ", ".join("?" for _ in neighbours)
            neighbour_nodes = connection.execute(
                f"""
                SELECT node_id, node_type, sequence_number
                FROM nodes
                WHERE thesis_id = ? AND node_id IN ({neighbour_placeholders})
                """,
                [thesis_id, *neighbours],
            ).fetchall()
            next_frontier = set()
            for node_id, node_type, sequence_number in neighbour_nodes:
                visited.add(node_id)
                if node_type == "chunk":
                    current = candidates.get(node_id)
                    if current is None or current[0] > 2:
                        candidates[node_id] = (2, sequence_number)
                elif node_type not in structural_node_types:
                    reachable_entities.add(node_id)
                    next_frontier.add(node_id)
            frontier = next_frontier

        if reachable_entities:
            entity_placeholders = ", ".join("?" for _ in reachable_entities)
            evidence_edges = connection.execute(
                f"""
                SELECT nodes.node_id, nodes.sequence_number
                FROM edges
                JOIN nodes ON nodes.node_id = CASE
                    WHEN edges.source_node_id IN ({entity_placeholders})
                    THEN edges.target_node_id ELSE edges.source_node_id END
                WHERE (
                    edges.source_node_id IN ({entity_placeholders})
                    OR edges.target_node_id IN ({entity_placeholders})
                ) AND nodes.thesis_id = ? AND nodes.node_type = 'chunk'
                """,
                [*reachable_entities, *reachable_entities, *reachable_entities, thesis_id],
            ).fetchall()
            for node_id, sequence_number in evidence_edges:
                if node_id not in seed_node_ids and node_id not in candidates:
                    candidates[node_id] = (2, sequence_number)

    ordered_candidates = sorted(
        candidates.items(),
        key=lambda item: (
            item[1][0],
            item[1][1] if item[1][1] is not None else float("inf"),
            item[0],
        ),
    )
    return [node_id.removeprefix("chunk:") for node_id, _ in ordered_candidates[:max_chunks]]
