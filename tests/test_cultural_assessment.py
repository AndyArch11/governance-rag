"""Tests for non-grading cultural lens evidence matching."""

import pytest

from scripts.thesis_graph.cultural_assessment import find_cultural_criteria_evidence


def test_cultural_criteria_return_candidate_evidence_for_human_review() -> None:
    profile = {
        "assessment_criteria": [
            {
                "id": "community_governance",
                "criterion": "Community governance",
                "description": "Look for community-led research governance.",
                "evidence_indicators": ["community-led", "advisory group"],
                "review_status": "draft",
            }
        ]
    }

    results = find_cultural_criteria_evidence(
        profile,
        ["chunk-1", "chunk-2"],
        ["A community-led advisory group shaped the project.", "Methods are described."],
        [
            {"heading_path": "Chapter 1 > Governance"},
            {"heading_path": "Chapter 3 > Methods"},
        ],
    )

    assert results[0]["review_required"] is True
    assert results[0]["review_status"] == "draft"
    assert results[0]["indicator_matches"] == [
        {
            "chunk_id": "chunk-1",
            "section": "Chapter 1 > Governance",
            "matched_indicators": ["community-led", "advisory group"],
            "indicator_evidence": [
                {
                    "indicator": "community-led",
                    "text": "A community-led advisory group shaped the project.",
                    "source_start": None,
                    "source_end": None,
                },
                {
                    "indicator": "advisory group",
                    "text": "A community-led advisory group shaped the project.",
                    "source_start": None,
                    "source_end": None,
                },
            ],
            "text": "A community-led advisory group shaped the project.",
        }
    ]


def test_no_indicator_match_is_not_reported_as_a_failed_criterion() -> None:
    results = find_cultural_criteria_evidence(
        {
            "assessment_criteria": [
                {
                    "id": "reciprocity",
                    "criterion": "Reciprocity",
                    "description": "Human review required.",
                    "evidence_indicators": ["reciprocity", "returned to community"],
                }
            ]
        },
        ["chunk-1"],
        ["The findings are presented in this chapter."],
        [{"section_title": "Results"}],
    )

    assert results[0]["indicator_matches"] == []
    assert results[0]["review_required"] is True
    assert "status" not in results[0]


def test_indicator_evidence_excerpt_centres_a_match_late_in_the_chunk() -> None:
    leading_text = "General background without the indicator. " * 20
    document = leading_text + "The project recognises data sovereignty in its governance."
    results = find_cultural_criteria_evidence(
        {
            "assessment_criteria": [
                {
                    "id": "data_sovereignty",
                    "criterion": "Data sovereignty",
                    "evidence_indicators": ["data sovereignty"],
                }
            ]
        },
        ["chunk-1"],
        [document],
        [{"heading_path": "Chapter 4 > Governance"}],
    )

    excerpt = results[0]["indicator_matches"][0]["text"]
    assert "data sovereignty" in excerpt
    assert not excerpt.startswith(leading_text[:40])


def test_indicator_matching_does_not_match_inside_larger_words() -> None:
    results = find_cultural_criteria_evidence(
        {
            "assessment_criteria": [
                {
                    "id": "art",
                    "criterion": "Art",
                    "evidence_indicators": ["art"],
                }
            ]
        },
        ["chunk-1", "chunk-2"],
        ["The article discusses artifacts.", "Art informs the research design."],
        [{"heading_path": "Chapter 1"}, {"heading_path": "Chapter 2"}],
    )

    assert [match["chunk_id"] for match in results[0]["indicator_matches"]] == ["chunk-2"]


def test_each_matched_indicator_has_its_own_supporting_excerpt() -> None:
    document = (
        "The project was community-led. "
        + "General methods context. " * 20
        + "An advisory group reviewed the findings."
    )
    results = find_cultural_criteria_evidence(
        {
            "assessment_criteria": [
                {
                    "id": "community_governance",
                    "criterion": "Community governance",
                    "evidence_indicators": ["community-led", "advisory group"],
                }
            ]
        },
        ["chunk-1"],
        [document],
        [{"heading_path": "Chapter 2 > Governance"}],
    )

    indicator_evidence = results[0]["indicator_matches"][0]["indicator_evidence"]
    assert [item["indicator"] for item in indicator_evidence] == [
        "community-led",
        "advisory group",
    ]
    assert "community-led" in indicator_evidence[0]["text"]
    assert "advisory group" in indicator_evidence[1]["text"]


def test_indicator_evidence_reports_absolute_source_offsets() -> None:
    document = "Background paragraph. Data sovereignty is discussed here."
    source_start = 1200
    results = find_cultural_criteria_evidence(
        {
            "assessment_criteria": [
                {
                    "id": "data_sovereignty",
                    "criterion": "Data sovereignty",
                    "evidence_indicators": ["data sovereignty"],
                }
            ]
        },
        ["chunk-7"],
        [document],
        [{"heading_path": "Chapter 4 > Governance", "source_start": source_start}],
    )

    evidence = results[0]["indicator_matches"][0]["indicator_evidence"][0]
    expected_start = source_start + document.index("Data sovereignty")
    assert evidence["source_start"] == expected_start
    assert evidence["source_end"] == expected_start + len("Data sovereignty")


def test_cultural_evidence_rejects_misaligned_chunk_records() -> None:
    with pytest.raises(
        ValueError, match="chunk IDs, documents and metadata must have equal lengths"
    ):
        find_cultural_criteria_evidence(
            {"assessment_criteria": []},
            ["chunk-1", "chunk-2"],
            ["Only one document"],
            [{"heading_path": "Chapter 1"}, {"heading_path": "Chapter 2"}],
        )
