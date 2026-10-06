"""Tests for deterministic thesis evidence graph construction."""

import json
import sqlite3
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts.ingest.academic.citation_graph_schema import ensure_schema
from scripts.ingest.academic.phd_assessor import ExaminerReadinessAnalysis, ReadinessCriterion
from scripts.thesis_graph import thesis_evidence_graph as thesis_graph_module
from scripts.thesis_graph.thesis_evidence_graph import (
    _figure_reference_status,
    _reconcile_figure_with_list,
    _resolve_adapted_figure_references,
    build_thesis_evidence_graph,
    expand_evidence_chunk_ids,
    get_chunk_provenance_paths,
    get_readiness_criterion_sources,
)


class FakeCollection:
    """Minimal Chroma-compatible collection fixture."""

    def get(self, **kwargs):
        return {
            "ids": ["chunk-1", "chunk-2", "chunk-3"],
            "documents": [
                "Introduction evidence",
                "More introduction evidence",
                "Methods evidence",
            ],
            "metadatas": [
                {
                    "thesis_id": "thesis-001",
                    "chunk_type": "child",
                    "sequence_number": 0,
                    "source_start": 0,
                    "source_end": 21,
                    "chapter": "Chapter 1",
                    "section_title": "Introduction",
                    "heading_path": "Chapter 1 > Introduction",
                },
                {
                    "thesis_id": "thesis-001",
                    "chunk_type": "child",
                    "sequence_number": 1,
                    "source_start": 22,
                    "source_end": 47,
                    "chapter": "Chapter 1",
                    "section_title": "Introduction",
                    "heading_path": "Chapter 1 > Introduction",
                },
                {
                    "thesis_id": "thesis-001",
                    "chunk_type": "child",
                    "sequence_number": 2,
                    "source_start": 48,
                    "source_end": 64,
                    "chapter": "Chapter 2",
                    "section_title": "Methods",
                    "heading_path": "Chapter 2 > Methods",
                },
            ],
        }


def test_build_thesis_evidence_graph_persists_hierarchy(tmp_path):
    """Graph materialises thesis, chapter, section, and chunk hierarchy."""
    output_path = tmp_path / "thesis-001.sqlite"

    graph = build_thesis_evidence_graph(FakeCollection(), "thesis-001", output_path)

    assert graph.output_path == output_path
    assert graph.node_count >= 8
    assert graph.edge_count >= 7

    with sqlite3.connect(output_path) as connection:
        node_counts = dict(
            connection.execute("SELECT node_type, COUNT(*) FROM nodes GROUP BY node_type")
        )
        edge_counts = dict(
            connection.execute("SELECT relation, COUNT(*) FROM edges GROUP BY relation")
        )
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))

    assert node_counts["chapter"] == 2
    assert node_counts["chunk"] == 3
    assert node_counts["section"] == 2
    assert node_counts["thesis"] == 1
    assert node_counts["readiness_criterion"] == 12
    assert edge_counts["contains"] == 7
    assert metadata["thesis_id"] == "thesis-001"
    assert metadata["graph_type"] == "thesis_evidence_graph"


def test_build_thesis_evidence_graph_links_chunk_and_claim_citations(tmp_path, monkeypatch):
    citation_graph_path = tmp_path / "academic_citation_graph.db"
    with sqlite3.connect(citation_graph_path) as connection:
        ensure_schema(connection)
        connection.executemany(
            "INSERT INTO nodes (node_id, node_type, title, authors, year) VALUES (?, ?, ?, ?, ?)",
            [
                ("thesis-001", "document", "Thesis", "[]", 2026),
                ("ref:smith2020", "reference", "Community research", '["Smith, Jane"]', 2020),
                ("ref:doe2021", "reference", "Comparative frameworks", '["Doe, John"]', 2021),
            ],
        )
        connection.executemany(
            "INSERT INTO edges (source, target, relation) VALUES (?, ?, ?)",
            [
                ("thesis-001", "ref:smith2020", "cites"),
                ("thesis-001", "ref:doe2021", "cites"),
            ],
        )

    claim_text = "Community-led interviews improved access (Smith, 2020)."
    chunk_text = f"{claim_text} Comparative frameworks include [2]."

    class CitationCollection(FakeCollection):
        def get(self, **kwargs):
            records = super().get(**kwargs)
            if "figure" in str(kwargs.get("where", "")):
                return {"ids": [], "documents": [], "metadatas": []}
            records["documents"][0] = chunk_text
            return records

    monkeypatch.setattr(
        thesis_graph_module.PhDQualityAssessor,
        "analyse_claims_and_contradictions",
        lambda self, chunks_data: SimpleNamespace(
            claims=[claim_text], contradictions=[], orphaned_claims=[], red_flags=[]
        ),
    )
    output_path = tmp_path / "thesis-001.sqlite"
    build_thesis_evidence_graph(
        CitationCollection(),
        "thesis-001",
        output_path,
        citation_graph_path=citation_graph_path,
    )

    with sqlite3.connect(output_path) as connection:
        bridge_edges = connection.execute("""
            SELECT source.node_type, target.node_id, edges.relation
            FROM edges
            JOIN nodes AS source ON source.node_id = edges.source_node_id
            JOIN nodes AS target ON target.node_id = edges.target_node_id
            WHERE edges.relation IN ('cites', 'supported_by_citation')
            """).fetchall()

    assert set(bridge_edges) == {
        ("chunk", "ref:smith2020", "cites"),
        ("chunk", "ref:doe2021", "cites"),
        ("claim", "ref:smith2020", "supported_by_citation"),
    }


def test_build_thesis_evidence_graph_reports_chunk_progress(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(
        thesis_graph_module,
        "audit",
        lambda module, event, data: events.append((module, event, data)),
        raising=False,
    )

    build_thesis_evidence_graph(FakeCollection(), "thesis-001", tmp_path / "thesis-001.sqlite")

    checkpoints = [data for _, event, data in events if event == "progress_checkpoint"]
    assert checkpoints
    chunk_checkpoints = [
        checkpoint for checkpoint in checkpoints if checkpoint["stage"] == "thesis_graph_chunks"
    ]
    assert chunk_checkpoints[0]["items_done"] == 0
    chunk_checkpoint = chunk_checkpoints[-1]
    assert chunk_checkpoint["items_done"] == 3
    assert chunk_checkpoint["items_total"] == 3
    assert chunk_checkpoint["percent"] == 100.0
    assert checkpoints[-1] == {
        "stage": "thesis_graph_complete",
        "items_done": 1,
        "items_total": 1,
        "percent": 100.0,
        "succeeded": 1,
        "failed": 0,
    }


def test_build_thesis_evidence_graph_rejects_missing_chapter_metadata(tmp_path):
    """Graph refuses non-chapter-aware collections instead of inferring hierarchy."""

    class UnstructuredCollection:
        def get(self, **kwargs):
            return {
                "ids": ["chunk-1"],
                "documents": ["Text"],
                "metadatas": [{"chunk_type": "child"}],
            }

    with pytest.raises(ValueError, match="No chapter-aware"):
        build_thesis_evidence_graph(
            UnstructuredCollection(), "thesis-001", tmp_path / "graph.sqlite"
        )


def test_thesis_graph_persists_figure_assets_across_rebuilds(tmp_path):
    class FigureChunkCollection(FakeCollection):
        def get(self, **kwargs):
            if {"chunk_type": "figure"} in kwargs.get("where", {}).get("$and", []):
                return {
                    "ids": ["thesis-001-figure_1"],
                    "documents": ["Figure 3: Community governance model"],
                    "metadatas": [
                        {
                            "thesis_id": "thesis-001",
                            "chunk_type": "figure",
                            "figure_id": "figure_1",
                            "figure_number": 1,
                            "page_number": 7,
                        }
                    ],
                }
            records = super().get(**kwargs)
            records["documents"][0] = "As shown in Fig. 3, the model links community members."
            records["metadatas"][0]["source_end"] = len(records["documents"][0])
            records["ids"].append("chunk-list-of-figures")
            records["documents"].append("Figure 3: Community governance model")
            records["metadatas"].append(
                {
                    "thesis_id": "thesis-001",
                    "chunk_type": "child",
                    "sequence_number": 3,
                    "source_start": 100,
                    "source_end": 136,
                    "chapter": "Front matter",
                    "section_title": "List of Figures",
                    "heading_path": "List of Figures",
                }
            )
            records["ids"].append("chunk-figure-caption")
            records["documents"].append("Figure 3: Community governance model")
            records["metadatas"].append(
                {
                    "thesis_id": "thesis-001",
                    "chunk_type": "child",
                    "sequence_number": 4,
                    "source_start": 400,
                    "source_end": 436,
                    "chapter": "Chapter 1",
                    "section_title": "Introduction",
                    "heading_path": "Chapter 1 > Introduction",
                }
            )
            return records

    output_path = tmp_path / "thesis-001.sqlite"
    citation_graph_path = tmp_path / "academic_citation_graph.db"
    with sqlite3.connect(citation_graph_path) as connection:
        ensure_schema(connection)
        connection.executemany(
            "INSERT INTO nodes (node_id, node_type, title, authors, year) VALUES (?, ?, ?, ?, ?)",
            [
                ("thesis-001", "document", "Thesis", "[]", 2026),
                ("ref:smith2020", "reference", "Community research", '["Smith, Jane"]', 2020),
            ],
        )
        connection.execute(
            "INSERT INTO edges (source, target, relation) VALUES (?, ?, ?)",
            ("thesis-001", "ref:smith2020", "cites"),
        )
    figure = {
        "figure_number": 1,
        "caption_number": "3",
        "kind": "picture",
        "page_number": 7,
        "source_start": 400,
        "source_end": 430,
        "bbox": {"left": 1.0, "top": 2.0, "right": 30.0, "bottom": 40.0},
        "chapter": "Chapter 1",
        "heading_path": "Chapter 1 > Introduction",
        "caption": "Figure 3: Community governance model (adapted from Smith et al. (2020))",
        "alt_text": "Diagram of community governance relationships.",
        "vision_status": "human_review_required",
        "description": "A diagram linking community members and governance groups.",
        "vision_assessment": {"figure_type": "conceptual diagram", "human_review_required": True},
        "image": Image.new("RGB", (2, 2), color="red"),
    }

    collection = FigureChunkCollection()
    build_thesis_evidence_graph(
        collection,
        "thesis-001",
        output_path,
        figures=[figure],
        citation_graph_path=citation_graph_path,
    )
    build_thesis_evidence_graph(
        collection, "thesis-001", output_path, citation_graph_path=citation_graph_path
    )

    with sqlite3.connect(output_path) as connection:
        figure_nodes = connection.execute(
            "SELECT node_id, label, attributes_json FROM nodes WHERE node_type = 'figure'"
        ).fetchall()
        assets = connection.execute("SELECT image_bytes, media_type FROM figure_assets").fetchall()
        section_links = connection.execute("""
            SELECT source.label
            FROM edges
            JOIN nodes AS source ON source.node_id = edges.source_node_id
            JOIN nodes AS target ON target.node_id = edges.target_node_id
            WHERE source.node_type = 'section' AND target.node_type = 'figure'
            """).fetchall()
        figure_chunk_links = connection.execute("""
            SELECT figure.node_type, figure_chunk.node_type,
                   json_extract(figure_chunk.attributes_json, '$.chroma_chunk_id')
            FROM edges
            JOIN nodes AS figure ON figure.node_id = edges.source_node_id
            JOIN nodes AS figure_chunk ON figure_chunk.node_id = edges.target_node_id
            WHERE edges.relation = 'described_by'
            """).fetchall()
        figure_reference_links = connection.execute("""
            SELECT json_extract(source.attributes_json, '$.chroma_chunk_id')
            FROM edges
            JOIN nodes AS source ON source.node_id = edges.source_node_id
            JOIN nodes AS target ON target.node_id = edges.target_node_id
            WHERE edges.relation = 'references' AND target.node_type = 'figure'
            """).fetchall()
        adapted_reference_links = connection.execute("""
            SELECT figure.node_type, reference.node_type, edges.relation, reference.node_id
            FROM edges
            JOIN nodes AS figure ON figure.node_id = edges.source_node_id
            JOIN nodes AS reference ON reference.node_id = edges.target_node_id
            WHERE edges.relation = 'adapted_from'
            """).fetchall()

    assert len(figure_nodes) == 1
    assert figure_nodes[0][1] == (
        "Figure 3: Community governance model (adapted from Smith et al. (2020))"
    )
    figure_attributes = json.loads(figure_nodes[0][2])
    assert figure_attributes["page_number"] == 7
    assert figure_attributes["caption_number"] == "3"
    assert figure_attributes["list_of_figures_status"] == "matched"
    assert figure_attributes["list_of_figures_title"] == "Community governance model"
    assert figure_attributes["adapted_reference_status"] == "linked"
    assert figure_attributes["adapted_reference_ids"] == ["ref:smith2020"]
    assert figure_attributes["alt_text"] == "Diagram of community governance relationships."
    assert figure_attributes["vision_status"] == "human_review_required"
    assert figure_attributes["vision_assessment"]["figure_type"] == "conceptual diagram"
    assert figure_attributes["body_reference_chunk_ids"] == ["chunk-1"]
    assert figure_attributes["body_reference_status"] == "referenced"
    assert section_links == [("Chapter 1 > Introduction",)]
    assert figure_chunk_links == [("figure", "figure_chunk", "thesis-001-figure_1")]
    assert figure_reference_links == [("chunk-1",)]
    assert adapted_reference_links == [("figure", "reference", "adapted_from", "ref:smith2020")]
    assert len(assets) == 1
    assert assets[0][0]
    assert assets[0][1] == "image/png"


def test_figure_list_reconciliation_distinguishes_missing_and_mismatched_entries():
    figure = {"caption_number": "2", "caption": "Figure 2: Expected title"}
    list_metadata = {"heading_path": "List of Figures"}

    assert _reconcile_figure_with_list(figure, []) == ("list_not_detected", None)
    assert _reconcile_figure_with_list(
        figure,
        [("list", "Figure 1: Other title", list_metadata)],
    ) == ("not_listed", None)
    assert _reconcile_figure_with_list(
        figure,
        [("list", "Figure 2: Different title", list_metadata)],
    ) == ("title_mismatch", "Different title")
    assert _reconcile_figure_with_list(
        figure,
        [("list", "Figure 2: Expected\n    title ........ 14", list_metadata)],
    ) == ("matched", "Expected title")
    assert _reconcile_figure_with_list(
        figure,
        [("list", "Figure 2: Expected\ntitle ........ 14", list_metadata)],
    ) == ("matched", "Expected title")
    assert _reconcile_figure_with_list(
        figure,
        [
            ("list-1", "Figure 2: Expected", list_metadata),
            ("list-2", "title ........ 14", list_metadata),
        ],
    ) == ("matched", "Expected title")
    assert _reconcile_figure_with_list(
        figure,
        [
            (
                "list-table",
                "| Figure 2 | Expected title | 14 |\n| --- | --- | --- |",
                list_metadata,
            )
        ],
    ) == ("matched", "Expected title")
    assert _reconcile_figure_with_list(
        {"caption_number": "2", "caption": "Figure 2: Expected title 2020"},
        [("list-table", "| Figure 2 | Expected title 2020 |", list_metadata)],
    ) == ("matched", "Expected title 2020")


def test_figure_reference_status_reports_scan_limits_without_claiming_absence():
    figure = {"caption_number": "2"}
    complete_chunks = [("chunk-1", "Body text", {"source_start": 100})]
    partial_chunks = [
        ("chunk-1", "Body text", {"source_start": 100}),
        ("chunk-2", "More body text", {"source_start": -1}),
    ]

    assert _figure_reference_status(figure, [], complete_chunks) == "no_explicit_reference_found"
    assert _figure_reference_status(figure, ["chunk-1"], complete_chunks) == "referenced"
    assert _figure_reference_status(figure, [], []) == "source_offsets_unavailable"
    assert _figure_reference_status(figure, [], partial_chunks) == "partial_source_offsets"
    assert _figure_reference_status({}, [], complete_chunks) == "caption_number_unavailable"


def test_adapted_figure_reference_requires_a_unique_author_year_match(tmp_path):
    """Test that adapted figure references require a unique author-year match."""
    citation_graph_path = tmp_path / "academic_citation_graph.db"
    with sqlite3.connect(citation_graph_path) as connection:
        ensure_schema(connection)
        connection.executemany(
            "INSERT INTO nodes (node_id, node_type, title, authors, year) VALUES (?, ?, ?, ?, ?)",
            [
                ("thesis-001", "document", "Thesis", "[]", 2026),
                ("ref:smith-a", "reference", "Source A", '["Smith, Jane"]', 2020),
                ("ref:smith-b", "reference", "Source B", '["Smith, John"]', 2020),
                ("ref:jones2021", "reference", "Source C", '["Jones, Alex"]', 2021),
            ],
        )
        connection.executemany(
            "INSERT INTO edges (source, target, relation) VALUES (?, ?, ?)",
            [
                ("thesis-001", "ref:smith-a", "cites"),
                ("thesis-001", "ref:smith-b", "cites"),
                ("thesis-001", "ref:jones2021", "cites"),
            ],
        )

    status, references = _resolve_adapted_figure_references(
        {"caption": "Adapted from Smith et al. (2020)"},
        "thesis-001",
        citation_graph_path,
    )
    assert status == "ambiguous"
    assert references == []

    status, references = _resolve_adapted_figure_references(
        {"caption": "Adapted from Jones et al. 2021"},
        "thesis-001",
        citation_graph_path,
    )
    assert status == "linked"
    assert [reference["node_id"] for reference in references] == ["ref:jones2021"]

    status, references = _resolve_adapted_figure_references(
        {"caption": "Adapted from footnote 3"},
        "thesis-001",
        citation_graph_path,
        chunks=[],
    )
    assert status == "footnote_unresolved"
    assert references == []

    status, references = _resolve_adapted_figure_references(
        {"caption": "Adapted from footnote 2"},
        "thesis-001",
        citation_graph_path,
        chunks=[
            ("chunk-1", "Footnote 2: Jones et al. (2021)", {}),
            ("chunk-2", "Footnote 2: Jones et al. (2021); see also Smith (2020)", {}),
        ],
    )
    assert status == "ambiguous"
    assert references == []

    status, references = _resolve_adapted_figure_references(
        {"caption": "Adapted from [1]"},
        "thesis-001",
        citation_graph_path,
    )
    assert status == "linked"
    assert [reference["node_id"] for reference in references] == ["ref:smith-a"]

    status, references = _resolve_adapted_figure_references(
        {"caption": "Adapted from [4]"},
        "thesis-001",
        citation_graph_path,
    )
    assert status == "unresolved"
    assert references == []

    status, references = _resolve_adapted_figure_references(
        {"caption": "Adapted from [1, 4]"},
        "thesis-001",
        citation_graph_path,
    )
    assert status == "partially_linked"
    assert [reference["node_id"] for reference in references] == ["ref:smith-a"]

    status, references = _resolve_adapted_figure_references(
        {"caption": "Adapted from [1-2]"},
        "thesis-001",
        citation_graph_path,
    )
    assert status == "linked"
    assert [reference["node_id"] for reference in references] == [
        "ref:smith-a",
        "ref:smith-b",
    ]

    status, references = _resolve_adapted_figure_references(
        {"caption": "Adapted from [1, 2]"},
        "thesis-001",
        citation_graph_path,
    )
    assert status == "linked"
    assert [reference["node_id"] for reference in references] == [
        "ref:smith-a",
        "ref:smith-b",
    ]

    status, references = _resolve_adapted_figure_references(
        {"caption": "Adapted from [4-5]"},
        "thesis-001",
        citation_graph_path,
    )
    assert status == "unresolved"
    assert references == []


def test_adapted_figure_reference_resolves_explicit_footnote_text(tmp_path):
    citation_graph_path = tmp_path / "academic_citation_graph.db"
    with sqlite3.connect(citation_graph_path) as connection:
        ensure_schema(connection)
        connection.execute(
            "INSERT INTO nodes (node_id, node_type, title, authors, year) VALUES (?, ?, ?, ?, ?)",
            ("thesis-001", "document", "Thesis", "[]", 2026),
        )
        connection.execute(
            "INSERT INTO nodes (node_id, node_type, title, authors, year) VALUES (?, ?, ?, ?, ?)",
            ("ref:jones2021", "reference", "Source", '["Jones, Alex"]', 2021),
        )
        connection.execute(
            "INSERT INTO edges (source, target, relation) VALUES (?, ?, ?)",
            ("thesis-001", "ref:jones2021", "cites"),
        )

    status, references = _resolve_adapted_figure_references(
        {"caption": "Adapted from footnote 2"},
        "thesis-001",
        citation_graph_path,
        chunks=[("chunk-1", "Footnote 2: Jones et al. (2021)", {})],
    )

    assert status == "linked"
    assert [reference["node_id"] for reference in references] == ["ref:jones2021"]


def test_expand_evidence_chunk_ids_stays_within_seed_chapter(tmp_path):
    """Expansion adds section and chapter siblings but excludes other chapters."""
    output_path = tmp_path / "thesis-001.sqlite"
    build_thesis_evidence_graph(FakeCollection(), "thesis-001", output_path)

    expanded = expand_evidence_chunk_ids(output_path, "thesis-001", ["chunk-1"], max_chunks=3)

    assert expanded == ["chunk-2"]


def test_expand_evidence_chunk_ids_follows_shared_claim_across_chapters(tmp_path):
    """Graph-linked evidence can expand beyond structural chapter siblings."""
    output_path = tmp_path / "thesis-001.sqlite"
    build_thesis_evidence_graph(FakeCollection(), "thesis-001", output_path)

    with sqlite3.connect(output_path) as connection:
        connection.execute(
            """
            INSERT INTO nodes (node_id, node_type, thesis_id, label, attributes_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("claim:shared", "claim", "thesis-001", "Shared claim", "{}"),
        )
        connection.executemany(
            "INSERT INTO edges VALUES (?, ?, ?)",
            [
                ("chunk:chunk-1", "claim:shared", "contains_claim"),
                ("chunk:chunk-3", "claim:shared", "contains_claim"),
            ],
        )

    expanded = expand_evidence_chunk_ids(output_path, "thesis-001", ["chunk-1"], max_chunks=3)

    assert expanded == ["chunk-2", "chunk-3"]


def test_get_chunk_provenance_paths_links_section_claim_and_reference(tmp_path):
    graph_path = tmp_path / "thesis-001.sqlite"
    build_thesis_evidence_graph(FakeCollection(), "thesis-001", graph_path)

    with sqlite3.connect(graph_path) as connection:
        connection.execute(
            """
            INSERT INTO nodes (node_id, node_type, thesis_id, label, attributes_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "claim:supported",
                "claim",
                "thesis-001",
                "A supported finding",
                "{}",
            ),
        )
        connection.execute(
            """
            INSERT INTO nodes (node_id, node_type, thesis_id, label, attributes_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "ref:smith2020",
                "reference",
                "thesis-001",
                "Smith",
                json.dumps({"year": 2020, "venue_rank": "Q1", "link_status": "available"}),
            ),
        )
        connection.executemany(
            "INSERT INTO edges VALUES (?, ?, ?)",
            [
                ("chunk:chunk-1", "claim:supported", "contains_claim"),
                ("claim:supported", "ref:smith2020", "supported_by_citation"),
            ],
        )

    paths = get_chunk_provenance_paths(graph_path, "thesis-001", ["chunk-1"])

    assert paths["chunk-1"] == [
        'Chapter 1 > Introduction > claim "A supported finding" '
        "-> cites Smith (2020, Q1, available)"
    ]


def test_build_thesis_evidence_graph_persists_assessment_entities(tmp_path):
    """Assessment entities are attached to their source thesis chunks."""

    class AssessmentCollection:
        def get(self, **kwargs):
            return {
                "ids": ["chunk-1", "chunk-2", "chunk-3", "chunk-4"],
                "documents": [
                    "## Chapter 1: Introduction\n\nResearch Question 1: How do interviews support recovery?",
                    "## Chapter 2: Methodology\n\nThe research design uses purposive sampling and interviews.",
                    "## Chapter 3: Findings\n\nThe findings demonstrate that participants valued peer support.",
                    "## Chapter 4: Conclusion\n\nThis study demonstrates that peer support improves recovery.",
                ],
                "metadatas": [
                    {
                        "thesis_id": "thesis-001",
                        "chunk_type": "child",
                        "sequence_number": index,
                        "source_start": index * 100,
                        "source_end": (index + 1) * 100,
                        "chapter": f"Chapter {index + 1}",
                        "section_title": section,
                        "heading_path": f"Chapter {index + 1} > {section}",
                    }
                    for index, section in enumerate(
                        ["Introduction", "Methodology", "Findings", "Conclusion"]
                    )
                ],
            }

    output_path = tmp_path / "thesis-001.sqlite"
    canonical_question = "How do interviews support recovery?"
    sub_question = "Which support strategies do participants describe?"
    conclusion_restatement = "To what extent do interviews contribute to recovery outcomes?"
    structure_analysis = SimpleNamespace(
        research_questions=[canonical_question, sub_question],
        research_inquiry_types={
            canonical_question: "research_question",
            sub_question: "sub_question",
        },
        research_inquiry_ids={canonical_question: "RQ1", sub_question: "RQ1a"},
        research_inquiry_parent_ids={sub_question: "RQ1"},
        research_inquiry_aliases={canonical_question: [conclusion_restatement]},
        research_inquiry_sources={
            canonical_question: [
                {"chunk_id": "chunk-1", "section": "Chapter 1 > Introduction"},
                {
                    "chunk_id": "chunk-4",
                    "section": "Chapter 4 > Conclusion",
                    "restatement": conclusion_restatement,
                },
            ],
            sub_question: [{"chunk_id": "chunk-1", "section": "Chapter 1 > Introduction"}],
        },
    )
    build_thesis_evidence_graph(
        AssessmentCollection(),
        "thesis-001",
        output_path,
        structure_analysis=structure_analysis,
    )

    with sqlite3.connect(output_path) as connection:
        node_types = {row[0] for row in connection.execute("SELECT node_type FROM nodes")}
        relationships = {row[0] for row in connection.execute("SELECT relation FROM edges")}
        addressed_rows = connection.execute("""
                SELECT DISTINCT target.label
                FROM edges
                JOIN nodes AS source ON source.node_id = edges.source_node_id
                JOIN nodes AS target ON target.node_id = edges.target_node_id
                WHERE edges.relation = 'addressed_by'
                  AND source.node_type = 'research_question'
            """).fetchall()
        addressed_sections = [row[0] for row in addressed_rows]
        inquiry_attributes_json = connection.execute(
            "SELECT attributes_json FROM nodes WHERE node_type = 'research_question' AND label = ?",
            (canonical_question,),
        ).fetchone()[0]
        subquestion_edge = connection.execute("""
            SELECT source.attributes_json, target.attributes_json
            FROM edges
            JOIN nodes AS source ON source.node_id = edges.source_node_id
            JOIN nodes AS target ON target.node_id = edges.target_node_id
            WHERE edges.relation = 'sub_question_of'
            """).fetchone()

    assert {"research_question", "method", "finding", "conclusion"} <= node_types
    assert {"states", "evidences", "addressed_by", "sub_question_of"} <= relationships
    assert set(addressed_sections) == {"Chapter 3 > Findings", "Chapter 4 > Conclusion"}
    inquiry_attributes = json.loads(inquiry_attributes_json)
    assert inquiry_attributes["text"] == canonical_question
    assert inquiry_attributes["inquiry_id"] == "RQ1"
    assert inquiry_attributes["parent_inquiry_id"] is None
    assert inquiry_attributes["restatements"] == [conclusion_restatement]
    assert [source["chunk_id"] for source in inquiry_attributes["evidence_sources"]] == [
        "chunk-1",
        "chunk-4",
    ]
    assert subquestion_edge is not None
    assert json.loads(subquestion_edge[0])["inquiry_id"] == "RQ1a"
    assert json.loads(subquestion_edge[0])["parent_inquiry_id"] == "RQ1"
    assert json.loads(subquestion_edge[1])["inquiry_id"] == "RQ1"


def test_profile_framed_inquiry_is_persisted_in_thesis_graph(tmp_path):
    class CulturalInquiryCollection:
        def get(self, **kwargs):
            return {
                "ids": ["chunk-1"],
                "documents": [
                    "The guiding yarning topics explored healing, Country and kinship relations."
                ],
                "metadatas": [
                    {
                        "thesis_id": "thesis-001",
                        "chunk_type": "child",
                        "sequence_number": 0,
                        "source_start": 0,
                        "source_end": 83,
                        "chapter": "Chapter 1",
                        "section_title": "Methodology",
                        "heading_path": "Chapter 1 > Methodology",
                    }
                ],
            }

    profile = {
        "research_question_framings": [
            {
                "framing": "Guiding yarning topics",
                "indicators": ["guiding yarning topics"],
                "classify_as": "guiding_question",
            }
        ]
    }
    output_path = tmp_path / "thesis-001.sqlite"

    build_thesis_evidence_graph(
        CulturalInquiryCollection(),
        "thesis-001",
        output_path,
        cultural_lens_profile=profile,
    )

    with sqlite3.connect(output_path) as connection:
        inquiry_labels = [
            (row[0], json.loads(row[1]))
            for row in connection.execute(
                "SELECT label, attributes_json FROM nodes WHERE node_type = 'research_question'"
            )
        ]

    assert len(inquiry_labels) == 1
    assert inquiry_labels[0][0] == (
        "The guiding yarning topics explored healing, Country and kinship relations."
    )
    assert inquiry_labels[0][1]["inquiry_type"] == "guiding_question"
    assert inquiry_labels[0][1]["evidence_sources"][0]["chunk_id"] == "chunk-1"


def test_build_thesis_evidence_graph_links_readiness_criteria_to_evidence(tmp_path):
    """Readiness criteria persist graph links to their supporting thesis chunks."""
    output_path = tmp_path / "thesis-001.sqlite"
    readiness = ExaminerReadinessAnalysis(
        notice="Human review required.",
        criteria=[
            ReadinessCriterion(
                criterion="research_significance",
                status="evidence_present",
                confidence=0.8,
                evidence=["Introduction evidence"],
                source_sections=["Introduction"],
                reason="Evidence located.",
                source="deterministic",
            )
        ],
        criteria_by_status={"evidence_present": ["research_significance"]},
        human_review_priorities=[],
        red_flags=[],
    )

    build_thesis_evidence_graph(FakeCollection(), "thesis-001", output_path, readiness=readiness)

    with sqlite3.connect(output_path) as connection:
        node_types = {row[0] for row in connection.execute("SELECT node_type FROM nodes")}
        relationships = {row[0] for row in connection.execute("SELECT relation FROM edges")}

    assert "readiness_criterion" in node_types
    assert "supported_by" in relationships
    assert get_readiness_criterion_sources(output_path, "thesis-001") == {
        "research_significance": ["chunk-1", "chunk-2"]
    }
