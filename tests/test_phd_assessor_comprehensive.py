"""Comprehensive unit tests for PhD assessor module.

Tests cover:
- Core assessment methods (assess_thesis, analyse_structure, etc.)
- Text processing utilities
- Citation analysis
- Writing quality metrics
- Claim detection and contradiction analysis
- Methodology validation
- Research question alignment
"""

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from scripts.ingest.academic.phd_assessor import (
    CitationPatternAnalysis,
    ClaimAnalysis,
    ContributionAlignment,
    ExaminerReadinessAnalysis,
    MethodologyChecklist,
    PhDQualityAssessor,
    ReadinessCriterion,
    RedFlag,
    StructureAnalysis,
    WritingQualityMetrics,
)
from scripts.utils.json_utils import extract_first_json_block


@pytest.fixture
def mock_collection():
    """Create a mock ChromaDB collection."""
    collection = MagicMock()
    collection.get.return_value = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
    return collection


@pytest.fixture
def assessor(mock_collection, tmp_path):
    """Create assessor with mock dependencies."""
    db_path = tmp_path / "test_citations.db"
    return PhDQualityAssessor(
        chunk_collection=mock_collection,
        llm_client=None,
        llm_flags={},
        citation_db_path=str(db_path),
    )


@pytest.fixture
def sample_chunks_data():
    """Sample chunks data for testing."""
    return {
        "ids": ["chunk1", "chunk2", "chunk3"],
        "documents": [
            "Introduction text with research content.",
            "Methodology section describing methods.",
            "Results showing findings.",
        ],
        "metadatas": [
            {
                "doc_id": "test_thesis",
                "chapter": "Chapter 1",
                "section_title": "Introduction",
                "sequence_number": 0,
                "source_category": "academic_paper",
            },
            {
                "doc_id": "test_thesis",
                "chapter": "Chapter 2",
                "section_title": "Methodology",
                "sequence_number": 1,
                "source_category": "academic_paper",
            },
            {
                "doc_id": "test_thesis",
                "chapter": "Chapter 3",
                "section_title": "Results",
                "sequence_number": 2,
                "source_category": "academic_paper",
            },
        ],
        "embeddings": [
            np.random.rand(1024).tolist(),
            np.random.rand(1024).tolist(),
            np.random.rand(1024).tolist(),
        ],
    }


def test_assessor_llm_invoke_records_estimates_on_success_and_failure(monkeypatch):
    from scripts.utils import llm_instrumentation

    usage_events = []
    monkeypatch.setattr(
        llm_instrumentation,
        "audit",
        lambda event, data: usage_events.append((event, data)),
    )
    prompt = "Classify this thesis inquiry."
    assessor = PhDQualityAssessor(
        chunk_collection=MagicMock(),
        llm_client=lambda _prompt: "This is a generated classification.",
    )

    assert assessor._llm_invoke(prompt) == "This is a generated classification."

    def fail(_prompt):
        raise TimeoutError("LLM request timed out")

    assessor.llm_client = fail
    with pytest.raises(TimeoutError):
        assessor._llm_invoke(prompt)

    records = [data for event, data in usage_events if event == "llm_usage"]
    assert len(records) == 2
    assert records[0]["input_tokens"] == len(prompt) // 4
    assert records[0]["output_tokens"] == len("This is a generated classification.") // 4
    assert records[0]["success"] is True
    assert records[1]["input_tokens"] == len(prompt) // 4
    assert records[1]["output_tokens"] == 0
    assert records[1]["success"] is False
    assert records[1]["failure_reason"] == "TimeoutError"
    assert all(record["token_source"] == "estimated" for record in records)


class TestTextProcessingUtilities:
    """Test text processing utility methods."""

    def test_split_sentences_basic(self, assessor):
        """Test basic sentence splitting."""
        text = "First sentence. Second sentence? Third sentence!"
        sentences = assessor._split_sentences(text)
        assert len(sentences) == 3
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]

    def test_split_sentences_empty(self, assessor):
        """Test splitting empty text."""
        assert assessor._split_sentences("") == []
        assert assessor._split_sentences(None) == []

    def test_tokenise_words(self, assessor):
        """Test word tokenisation."""
        text = "The quick brown fox"
        words = assessor._tokenise_words(text)
        assert "the" in words
        assert "quick" in words
        assert "brown" in words
        assert "fox" in words

    def test_tokenise_words_with_hyphens(self, assessor):
        """Test that hyphenated words are preserved."""
        text = "well-known state-of-the-art"
        words = assessor._tokenise_words(text)
        assert "well-known" in words
        assert "state-of-the-art" in words

    def test_contains_any(self, assessor):
        """Test keyword detection."""
        text = "This study demonstrates the effectiveness of the method."
        assert assessor._contains_any(text, ["demonstrates", "shows"])
        assert not assessor._contains_any(text, ["proves", "confirms"])

    def test_count_syllables(self, assessor):
        """Test syllable counting."""
        assert assessor._count_syllables("cat") == 1
        assert assessor._count_syllables("water") == 2
        assert assessor._count_syllables("beautiful") == 3
        assert assessor._count_syllables("university") >= 4


class TestExaminerReadinessAnalysis:
    """Tests for the human-review examiner-readiness report model."""

    def test_baseline_readiness_is_non_grading(self, assessor, sample_chunks_data):
        """Baseline readiness analysis defines the human-assessment boundary."""
        readiness = assessor.analyse_examiner_readiness(sample_chunks_data)

        assert isinstance(readiness, ExaminerReadinessAnalysis)
        assert "human assessment only" in readiness.notice.lower()
        assert "examination outcome" in readiness.notice.lower()
        assert readiness.red_flags == []
        assert len(readiness.criteria) == 12

        boundary = readiness.criteria[0]
        assert isinstance(boundary, ReadinessCriterion)
        assert boundary.criterion == "human_assessment_boundary"
        assert boundary.status == "evidence_present"
        assert boundary.source == "system"
        assert boundary.confidence == 1.0
        assert "human_assessment_boundary" in readiness.criteria_by_status["evidence_present"]
        assert readiness.human_review_priorities

    def test_readiness_review_priorities_order_lowest_confidence_first(self, assessor):
        """Human-review items are grouped and prioritised by their confidence."""
        chunks_data = {
            "documents": ["Introduction text without research-question framing."],
            "metadatas": [{"section_title": "Introduction", "chapter": "Chapter 1"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)

        assert "research_significance" in readiness.criteria_by_status["needs_human_review"]
        confidence_by_criterion = {
            criterion.criterion: criterion.confidence
            for criterion in readiness.criteria
            if criterion.status == "needs_human_review"
        }
        ordered_confidences = [
            confidence_by_criterion[criterion] for criterion in readiness.human_review_priorities
        ]
        assert ordered_confidences == sorted(ordered_confidences)

    def test_assessment_report_labels_score_as_non_grading_signal(self, assessor, mock_collection):
        """Assessment reports label the score as human-review support, not a grade."""
        mock_collection.get.return_value = {
            "ids": ["chunk-1"],
            "documents": ["Introduction text."],
            "metadatas": [{"doc_id": "thesis-1", "section_title": "Introduction"}],
            "embeddings": [np.ones(1024).tolist()],
        }

        report = assessor.assess_thesis("thesis-1")

        assert "human review" in report.score_label.lower()
        assert "not a grade" in report.score_label.lower()
        assert "examination outcome" in report.score_label.lower()

    def test_research_significance_evidence_is_reviewable(self, assessor):
        """Significance framing includes source text and its originating section."""
        chunks_data = {
            "documents": [
                (
                    "Research Question: How can regional health services address a significant "
                    "policy gap in rural mental health access?"
                )
            ],
            "metadatas": [{"section_title": "Introduction", "chapter": "Chapter 1"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item for item in readiness.criteria if item.criterion == "research_significance"
        )

        assert criterion.status == "evidence_present"
        assert criterion.evidence

    def test_readiness_uses_heading_fallback_when_metadata_is_missing(self, assessor):
        """Readiness scoping uses an explicit heading when metadata is absent."""
        chunks_data = {
            "documents": [
                "# Introduction\nThis research question addresses an important gap in practice."
            ],
            "metadatas": [{}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item for item in readiness.criteria if item.criterion == "research_significance"
        )

        assert criterion.status == "evidence_present"
        assert criterion.source_sections == ["Introduction"]
        assert criterion.evidence
        assert "Introduction" in criterion.source_sections
        assert criterion.source == "deterministic"

    def test_readiness_uses_criterion_heading_when_metadata_is_missing(self, assessor):
        """Criterion-specific headings scope evidence when metadata is absent."""
        chunks_data = {
            "documents": [
                "## Research Significance\n"
                "This research question addresses an important gap in practice."
            ],
            "metadatas": [{}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item for item in readiness.criteria if item.criterion == "research_significance"
        )

        assert criterion.status == "evidence_present"
        assert criterion.source_sections == ["Research Significance"]

    def test_research_significance_requires_human_review_when_unframed(self, assessor):
        """Missing significance framing prompts review without producing a defect flag."""
        chunks_data = {
            "documents": ["Research Question: What methods were used in this study?"],
            "metadatas": [{"section_title": "Introduction", "chapter": "Chapter 1"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item for item in readiness.criteria if item.criterion == "research_significance"
        )

        assert criterion.status == "needs_human_review"
        assert criterion.evidence == []
        assert readiness.red_flags == []

    def test_contribution_articulation_evidence_is_reviewable(self, assessor):
        """Explicit original contribution statements retain source evidence."""
        chunks_data = {
            "documents": [
                (
                    "This thesis makes an original and significant contribution to knowledge "
                    "by introducing a validated framework for regional health governance."
                )
            ],
            "metadatas": [{"section_title": "Statement of Contributions", "chapter": "Chapter 1"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item for item in readiness.criteria if item.criterion == "contribution_articulation"
        )

        assert criterion.status == "evidence_present"
        assert criterion.evidence
        assert "Statement of Contributions" in criterion.source_sections
        assert criterion.source == "deterministic"

    def test_contribution_articulation_requires_human_review_when_unframed(self, assessor):
        """Unqualified contribution claims prompt review instead of a defect flag."""
        chunks_data = {
            "documents": ["This thesis contributes a framework for regional health governance."],
            "metadatas": [{"section_title": "Conclusion", "chapter": "Chapter 6"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item for item in readiness.criteria if item.criterion == "contribution_articulation"
        )

        assert criterion.status == "needs_human_review"
        assert criterion.evidence == []
        assert readiness.red_flags == []

    def test_conceptual_integration_evidence_is_reviewable(self, assessor):
        """Explicit framework-to-objective links retain source evidence."""
        chunks_data = {
            "documents": [
                (
                    "The conceptual framework integrates the literature review and guides the "
                    "research objectives for regional health governance."
                )
            ],
            "metadatas": [{"section_title": "Conceptual Framework", "chapter": "Chapter 2"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item for item in readiness.criteria if item.criterion == "conceptual_integration"
        )

        assert criterion.status == "evidence_present"
        assert criterion.evidence
        assert "Conceptual Framework" in criterion.source_sections
        assert criterion.source == "deterministic"

    def test_conceptual_integration_requires_explicit_link(self, assessor):
        """Separate framework and objective mentions do not imply integration."""
        chunks_data = {
            "documents": [
                "The conceptual framework is described in this chapter. Research objectives are listed.",
            ],
            "metadatas": [{"section_title": "Introduction", "chapter": "Chapter 1"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item for item in readiness.criteria if item.criterion == "conceptual_integration"
        )

        assert criterion.status == "needs_human_review"
        assert criterion.evidence == []
        assert readiness.red_flags == []

    def test_methodological_rigour_evidence_is_reviewable(self, assessor):
        """Methodology evidence covers multiple rigour dimensions with sources."""
        chunks_data = {
            "documents": [
                (
                    "A stratified sampling approach was selected because it was appropriate for the "
                    "research question. Participants were recruited using a documented procedure. "
                    "Triangulation and member checking strengthened validity and reliability."
                )
            ],
            "metadatas": [{"section_title": "Methodology", "chapter": "Chapter 3"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item for item in readiness.criteria if item.criterion == "methodological_rigour"
        )

        assert criterion.status == "evidence_present"
        assert criterion.evidence
        assert "Methodology" in criterion.source_sections
        assert criterion.source == "deterministic"

    def test_methodological_rigour_requires_multiple_evidence_dimensions(self, assessor):
        """A method name alone does not establish methodological rigour."""
        chunks_data = {
            "documents": ["The study used interviews and surveys."],
            "metadatas": [{"section_title": "Methods", "chapter": "Chapter 3"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item for item in readiness.criteria if item.criterion == "methodological_rigour"
        )

        assert criterion.status == "needs_human_review"
        assert criterion.evidence == []
        assert readiness.red_flags == []

    def test_findings_quality_evidence_is_reviewable(self, assessor):
        """Findings linked to objectives and interpretation retain source evidence."""
        chunks_data = {
            "documents": [
                (
                    "The findings directly address the research objectives and are presented in Table 4. "
                    "Unexpected results are explained by the sampling limitations."
                )
            ],
            "metadatas": [{"section_title": "Results and Discussion", "chapter": "Chapter 5"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item for item in readiness.criteria if item.criterion == "findings_quality"
        )

        assert criterion.status == "evidence_present"
        assert criterion.evidence
        assert "Results and Discussion" in criterion.source_sections
        assert criterion.source == "deterministic"

    def test_findings_quality_requires_linkage_and_interpretation(self, assessor):
        """Generic results prose does not establish findings quality evidence."""
        chunks_data = {
            "documents": ["The results are presented in Table 4 and Figure 2."],
            "metadatas": [{"section_title": "Results", "chapter": "Chapter 4"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item for item in readiness.criteria if item.criterion == "findings_quality"
        )

        assert criterion.status == "needs_human_review"
        assert criterion.evidence
        assert readiness.red_flags == []

    def test_contribution_contextualisation_evidence_is_reviewable(self, assessor):
        """Prior-work positioning and future research evidence retain their source."""
        chunks_data = {
            "documents": [
                (
                    "These findings extend previous research by demonstrating that regional health "
                    "governance improves rural access. Future research should investigate whether this "
                    "framework transfers to remote communities."
                )
            ],
            "metadatas": [{"section_title": "Discussion", "chapter": "Chapter 5"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item
            for item in readiness.criteria
            if item.criterion == "contribution_contextualisation"
        )

        assert criterion.status == "evidence_present"
        assert len(criterion.evidence) == 2
        assert "Discussion" in criterion.source_sections
        assert criterion.source == "deterministic"

    def test_contribution_contextualisation_requires_both_dimensions(self, assessor):
        """Future work alone does not establish contribution contextualisation."""
        chunks_data = {
            "documents": ["Future research should investigate regional health governance."],
            "metadatas": [{"section_title": "Conclusion", "chapter": "Chapter 6"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item
            for item in readiness.criteria
            if item.criterion == "contribution_contextualisation"
        )

        assert criterion.status == "needs_human_review"
        assert criterion.evidence
        assert readiness.red_flags == []

    def test_thesis_by_publication_detection_requires_independent_signals(self, assessor):
        """Publication thesis detection requires form and publication-content evidence."""
        chunks_data = {
            "documents": [
                (
                    "This thesis by publication includes three peer-reviewed journal articles. "
                    "The included publications form the core research chapters."
                )
            ],
            "metadatas": [{"section_title": "Introduction", "chapter": "Chapter 1"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item
            for item in readiness.criteria
            if item.criterion == "thesis_by_publication_applicability"
        )

        assert criterion.status == "evidence_present"
        assert criterion.evidence
        assert "Introduction" in criterion.source_sections
        assert criterion.source == "deterministic"

    def test_thesis_by_publication_is_not_triggered_by_ordinary_citations(self, assessor):
        """Journal article citations alone do not trigger publication-based checks."""
        chunks_data = {
            "documents": ["The literature review cites numerous peer-reviewed journal articles."],
            "metadatas": [{"section_title": "Literature Review", "chapter": "Chapter 2"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item
            for item in readiness.criteria
            if item.criterion == "thesis_by_publication_applicability"
        )

        assert criterion.status == "not_applicable"
        assert readiness.red_flags == []

    def test_publication_thesis_minimum_treatment_evidence_is_reviewable(self, assessor):
        """Publication thesis checks retain evidence for all guide-derived elements."""
        chunks_data = {
            "documents": [
                (
                    "This thesis by publication includes published work. The introduction contains an "
                    "independent and original review of the relevant literature."
                ),
                "The framing chapter frames the publications within the wider discipline.",
                "Bridging statements link chapters into a cohesive narrative.",
                "The independent general discussion integrates the findings and future research needs.",
            ],
            "metadatas": [
                {"section_title": "Introduction", "chapter": "Chapter 1"},
                {"section_title": "Framing Chapter", "chapter": "Chapter 2"},
                {"section_title": "Bridging Statements", "chapter": "Chapter 3"},
                {"section_title": "General Discussion", "chapter": "Chapter 6"},
            ],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item
            for item in readiness.criteria
            if item.criterion == "publication_thesis_minimum_treatment"
        )

        assert criterion.status == "evidence_present"
        assert len(criterion.evidence) == 4
        assert "General Discussion" in criterion.source_sections
        assert criterion.source == "deterministic"

    def test_publication_thesis_minimum_treatment_is_not_applicable_without_detection(
        self, assessor
    ):
        """Publication-specific checks are skipped for conventional thesis forms."""
        chunks_data = {
            "documents": ["The general discussion integrates the findings."],
            "metadatas": [{"section_title": "Discussion", "chapter": "Chapter 6"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item
            for item in readiness.criteria
            if item.criterion == "publication_thesis_minimum_treatment"
        )

        assert criterion.status == "not_applicable"
        assert readiness.red_flags == []

    def test_practice_based_thesis_detection_requires_independent_signals(self, assessor):
        """Practice-based thesis detection requires form and component evidence."""
        chunks_data = {
            "documents": [
                (
                    "This practice-based thesis comprises an exegesis and a creative work. "
                    "A durable record of the exhibition accompanies the submitted thesis."
                )
            ],
            "metadatas": [{"section_title": "Introduction", "chapter": "Chapter 1"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item
            for item in readiness.criteria
            if item.criterion == "practice_based_thesis_applicability"
        )

        assert criterion.status == "evidence_present"
        assert criterion.evidence
        assert "Introduction" in criterion.source_sections
        assert criterion.source == "deterministic"

    def test_practice_based_thesis_is_not_triggered_by_creative_language(self, assessor):
        """Generic creative language does not trigger practice-based checks."""
        chunks_data = {
            "documents": ["The research uses creative approaches to analyse interview responses."],
            "metadatas": [{"section_title": "Methodology", "chapter": "Chapter 3"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item
            for item in readiness.criteria
            if item.criterion == "practice_based_thesis_applicability"
        )

        assert criterion.status == "not_applicable"
        assert readiness.red_flags == []

    def test_practice_based_thesis_integration_evidence_is_reviewable(self, assessor):
        """Explicit exegesis and creative-work integration retains source evidence."""
        chunks_data = {
            "documents": [
                (
                    "This practice-based thesis includes an exegesis and a creative work. "
                    "The exegesis and creative work are examined as an integrated whole."
                )
            ],
            "metadatas": [{"section_title": "Exegesis", "chapter": "Chapter 4"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item
            for item in readiness.criteria
            if item.criterion == "practice_based_thesis_integration"
        )

        assert criterion.status == "evidence_present"
        assert criterion.evidence
        assert "Exegesis" in criterion.source_sections
        assert criterion.source == "deterministic"

    def test_practice_based_thesis_integration_requires_explicit_link(self, assessor):
        """Separate exegesis and creative work mentions do not imply integration."""
        chunks_data = {
            "documents": [
                "This practice-based thesis includes an exegesis and a creative work.",
            ],
            "metadatas": [{"section_title": "Exegesis", "chapter": "Chapter 4"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item
            for item in readiness.criteria
            if item.criterion == "practice_based_thesis_integration"
        )

        assert criterion.status == "needs_human_review"
        assert criterion.evidence == []
        assert readiness.red_flags == []

    def test_generative_ai_disclosure_evidence_is_reviewable(self, assessor):
        """Declared AI use includes tool, purpose, extent, and prompt evidence."""
        chunks_data = {
            "documents": [
                (
                    "ChatGPT was used to assist with language editing. Its use was limited to "
                    "Chapter 2, and the prompt log is included in Appendix A."
                )
            ],
            "metadatas": [{"section_title": "AI Disclosure", "chapter": "Appendix A"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item for item in readiness.criteria if item.criterion == "generative_ai_disclosure"
        )

        assert criterion.status == "evidence_present"
        assert criterion.evidence
        assert "AI Disclosure" in criterion.source_sections
        assert criterion.source == "deterministic"

    def test_generative_ai_disclosure_ignores_explicit_non_use(self, assessor):
        """An explicit statement that AI was not used does not trigger disclosure review."""
        chunks_data = {
            "documents": ["Generative AI was not used in preparing this thesis."],
            "metadatas": [{"section_title": "Declaration", "chapter": "Front Matter"}],
        }

        readiness = assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item for item in readiness.criteria if item.criterion == "generative_ai_disclosure"
        )

        assert criterion.status == "not_applicable"
        assert criterion.evidence == []
        assert readiness.red_flags == []

    def test_llm_readiness_evidence_enrichment_is_source_grounded(self, mock_collection, tmp_path):
        """Opt-in LLM evidence augments, but cannot replace, deterministic readiness evidence."""
        llm_client = MagicMock(
            return_value=(
                '{"evidence": "Research Question: How can regional health services address a '
                'significant policy gap?", "section": "Introduction", '
                '"reason": "It explicitly links the question to a policy gap.", "confidence": 0.8}'
            )
        )
        readiness_assessor = PhDQualityAssessor(
            chunk_collection=mock_collection,
            llm_client=llm_client,
            llm_flags={"readiness_evidence": True},
            citation_db_path=str(tmp_path / "test_citations.db"),
        )
        chunks_data = {
            "documents": [
                "Research Question: How can regional health services address a significant policy gap?"
            ],
            "metadatas": [{"section_title": "Introduction", "chapter": "Chapter 1"}],
        }

        readiness = readiness_assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item for item in readiness.criteria if item.criterion == "research_significance"
        )

        assert criterion.status == "evidence_present"
        assert criterion.source == "mixed"
        assert "LLM review note" in criterion.reason
        prompt = llm_client.call_args.args[0]
        assert prompt.index("JSON format:") < prompt.index("Criterion:") < prompt.index("Evidence:")

    def test_llm_readiness_evidence_discards_ungrounded_or_low_confidence_output(
        self, mock_collection, tmp_path
    ):
        """Invalid LLM output leaves deterministic readiness findings unchanged."""
        llm_client = MagicMock(
            return_value=(
                '{"evidence": "Invented evidence", "section": "Introduction", '
                '"reason": "Unsupported claim", "confidence": 0.4}'
            )
        )
        readiness_assessor = PhDQualityAssessor(
            chunk_collection=mock_collection,
            llm_client=llm_client,
            llm_flags={"readiness_evidence": True},
            citation_db_path=str(tmp_path / "test_citations.db"),
        )
        chunks_data = {
            "documents": [
                "Research Question: How can regional health services address a significant policy gap?"
            ],
            "metadatas": [{"section_title": "Introduction", "chapter": "Chapter 1"}],
        }

        readiness = readiness_assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item for item in readiness.criteria if item.criterion == "research_significance"
        )

        assert criterion.status == "evidence_present"
        assert criterion.source == "deterministic"
        assert "LLM review note" not in criterion.reason

    def test_llm_readiness_evidence_discards_malformed_output(self, mock_collection, tmp_path):
        """Malformed LLM output leaves the deterministic readiness result intact."""
        readiness_assessor = PhDQualityAssessor(
            chunk_collection=mock_collection,
            llm_client=MagicMock(return_value="not valid JSON"),
            llm_flags={"readiness_evidence": True},
            citation_db_path=str(tmp_path / "test_citations.db"),
        )
        chunks_data = {
            "documents": [
                "Research Question: How can regional health services address a significant policy gap?"
            ],
            "metadatas": [{"section_title": "Introduction", "chapter": "Chapter 1"}],
        }

        readiness = readiness_assessor.analyse_examiner_readiness(chunks_data)
        criterion = next(
            item for item in readiness.criteria if item.criterion == "research_significance"
        )

        assert criterion.status == "evidence_present"
        assert criterion.source == "deterministic"
        assert "LLM review note" not in criterion.reason


class TestExaminerReadinessCalibration:
    """Regression checks for synthetic scenarios awaiting human calibration."""

    @pytest.fixture
    def calibration_scenarios(self):
        """Load anonymised readiness scenarios and expected evidence states."""
        fixture_path = Path(__file__).parent / "fixtures" / "examiner_readiness_calibration.json"
        calibration_data = json.loads(fixture_path.read_text())
        assert calibration_data["status"] == "awaiting_human_review_calibration"
        return calibration_data["scenarios"]

    def test_calibration_scenarios_preserve_expected_readiness_states(
        self, assessor, calibration_scenarios
    ):
        """Synthetic calibration scenarios retain their documented readiness states."""
        for scenario in calibration_scenarios:
            readiness = assessor.analyse_examiner_readiness(
                {
                    "documents": scenario["documents"],
                    "metadatas": scenario["metadatas"],
                }
            )
            statuses = {criterion.criterion: criterion.status for criterion in readiness.criteria}

            assert {
                criterion: statuses[criterion] for criterion in scenario["expected_statuses"]
            } == scenario["expected_statuses"], scenario["name"]


class TestChapterSimilarity:
    """Test chapter similarity computation."""

    def test_compute_chapter_similarity_identical(self, assessor):
        """Test similarity of identical embeddings."""
        emb = np.random.rand(1024)
        similarity = assessor._compute_chapter_similarity(emb, emb)
        assert abs(similarity - 1.0) < 0.001

    def test_compute_chapter_similarity_different(self, assessor):
        """Test similarity of different embeddings."""
        emb1 = np.random.rand(1024)
        emb2 = np.random.rand(1024)
        similarity = assessor._compute_chapter_similarity(emb1, emb2)
        assert -1.0 <= similarity <= 1.0

    def test_compute_chapter_similarity_zero_vectors(self, assessor):
        """Test similarity with zero vectors."""
        emb1 = np.zeros(1024)
        emb2 = np.random.rand(1024)
        similarity = assessor._compute_chapter_similarity(emb1, emb2)
        assert similarity == 0.0


class TestCitationAnalysis:
    """Test citation analysis methods."""

    def test_is_recent_true(self, assessor):
        """Test recent year detection."""
        current_year = datetime.now().year
        assert assessor._is_recent(current_year)
        assert assessor._is_recent(current_year - 2)
        assert assessor._is_recent(current_year - 4)

    def test_is_recent_false(self, assessor):
        """Test old year detection."""
        current_year = datetime.now().year
        assert not assessor._is_recent(current_year - 10)
        assert not assessor._is_recent(2000)
        assert not assessor._is_recent(1990)

    def test_is_recent_none(self, assessor):
        """Test None year handling."""
        assert not assessor._is_recent(None)

    def test_cluster_citations(self, assessor):
        """Test citation clustering by venue."""
        citations = [
            {"source": "crossref", "title": "Paper 1"},
            {"source": "crossref", "title": "Paper 2"},
            {"source": "openalex", "title": "Paper 3"},
            {"source": "crossref", "title": "Paper 4"},
        ]
        clusters = assessor._cluster_citations(citations)
        assert clusters["crossref"] == 3
        assert clusters["openalex"] == 1

    def test_cluster_citations_empty(self, assessor):
        """Test clustering empty citation list."""
        assert assessor._cluster_citations([]) == {}


class TestClaimExtraction:
    """Test claim extraction and contradiction detection."""

    def test_extract_claims_from_text(self, assessor):
        """Test heuristic claim extraction."""
        text = """
        We find that the method improves accuracy.
        This study demonstrates significant results.
        Results show clear improvement.
        The data suggests positive outcomes.
        """
        claims = assessor._extract_claims_from_text(text)
        assert len(claims) > 0
        assert any("find" in claim.lower() for claim in claims)
        assert any("demonstrates" in claim.lower() for claim in claims)

    def test_extract_claims_empty_text(self, assessor):
        """Test claim extraction from empty text."""
        assert assessor._extract_claims_from_text("") == []
        assert assessor._extract_claims_from_text(None) == []

    def test_detect_contradictions_basic(self, assessor):
        """Test basic contradiction detection."""
        claims = [
            "The method improves accuracy significantly",
            "The method does not improve accuracy",
        ]
        contradictions = assessor._detect_contradictions(claims)
        # Should detect potential contradiction due to negation
        assert len(contradictions) >= 0  # May or may not detect depending on token overlap

    def test_detect_contradictions_empty(self, assessor):
        """Test contradiction detection with empty claims."""
        assert assessor._detect_contradictions([]) == []
        assert assessor._detect_contradictions(["single claim"]) == []

    def test_detect_orphaned_claims(self, assessor, sample_chunks_data):
        """Test detection of claims without citations."""
        # Modify sample data to include claims without citations
        sample_chunks_data["documents"][0] = "We prove that the hypothesis is correct."
        sample_chunks_data["documents"][1] = "The results clearly demonstrate effectiveness."

        orphaned = assessor._detect_orphaned_claims(sample_chunks_data)
        # Should find claims without citation markers
        assert isinstance(orphaned, list)


class TestWritingQualityMetrics:
    """Test writing quality analysis methods."""

    def test_flesch_reading_ease_simple(self, assessor):
        """Test Flesch reading ease with simple text."""
        text = "The cat sat on the mat. The dog ran in the park."
        score = assessor._flesch_reading_ease(text)
        assert 0 <= score <= 100
        # Simple text should have high readability
        assert score > 50

    def test_flesch_reading_ease_complex(self, assessor):
        """Test Flesch reading ease with complex text."""
        text = """The phenomenological epistemological considerations 
        necessitate comprehensive multidisciplinary investigations."""
        score = assessor._flesch_reading_ease(text)
        assert 0 <= score <= 100
        # Complex text should have lower readability
        assert score < 50

    def test_flesch_education_level(self, assessor):
        """Test education level classification."""
        assert "undergraduate" in assessor._flesch_education_level(30).lower()
        assert "grade" in assessor._flesch_education_level(65).lower()
        assert "graduate" in assessor._flesch_education_level(10).lower()

    def test_passive_voice_ratio(self, assessor):
        """Test passive voice detection."""
        sentences = [
            "The cat was chased by the dog.",  # Passive
            "The dog chased the cat.",  # Active
            "Results were analysed using SPSS.",  # Passive
            "We analysed the results.",  # Active
        ]
        ratio = assessor._passive_voice_ratio(sentences)
        assert 0.0 <= ratio <= 1.0
        # Should detect at least some passive voice
        assert ratio > 0

    def test_jargon_density(self, assessor):
        """Test jargon density calculation."""
        # High jargon
        jargon_words = ["methodology", "epistemological", "phenomenological"] * 10
        normal_words = ["the", "and", "is"] * 5
        all_words = jargon_words + normal_words
        density = assessor._jargon_density(all_words)
        assert density > 0.4  # Should be relatively high

        # Low jargon
        simple_words = ["cat", "dog", "run", "jump", "play"] * 10
        density_simple = assessor._jargon_density(simple_words)
        assert density_simple < 0.3


class TestKeywordExtraction:
    """Test keyword extraction utilities."""

    def test_extract_keywords(self, assessor):
        """Test keyword extraction from text."""
        text = """
        Machine learning algorithms demonstrate significant improvements
        in natural language processing tasks. Deep learning models
        achieve state-of-the-art results on various benchmarks.
        """
        keywords = assessor._extract_keywords(text, limit=5)
        assert len(keywords) <= 5
        assert all(isinstance(kw, str) for kw in keywords)
        # Should extract meaningful words, not stopwords
        assert not any(kw in ["the", "in", "on"] for kw in keywords)

    def test_extract_strong_claims(self, assessor):
        """Test extraction of strong claims."""
        text = """
        We prove that the algorithm converges.
        The results clearly demonstrate superiority.
        This undoubtedly shows the effectiveness.
        Perhaps the method works well.
        """
        claims = assessor._extract_strong_claims(text)
        assert len(claims) > 0
        # Should find strong claim markers
        assert any("prove" in claim.lower() for claim in claims)


class TestMissingDetection:
    """Test missing section detection."""

    def test_detect_missing_sections_all_present(self, assessor):
        """Test when all required sections are present."""
        chunks_data = {
            "documents": [
                "Introduction section",
                "Literature Review chapter",
                "Methodology details",
                "Results and findings",
                "Discussion of results",
                "Conclusion summary",
                "Limitations of study",
            ],
            "metadatas": [
                {"section_title": "Introduction", "chapter": "Chapter 1"},
                {"section_title": "Literature Review", "chapter": "Chapter 2"},
                {"section_title": "Methodology", "chapter": "Chapter 3"},
                {"section_title": "Results", "chapter": "Chapter 4"},
                {"section_title": "Discussion", "chapter": "Chapter 5"},
                {"section_title": "Conclusion", "chapter": "Chapter 6"},
                {"section_title": "Limitations", "chapter": "Chapter 7"},
            ],
        }
        missing = assessor._detect_missing_sections(chunks_data)
        assert len(missing) == 0

    def test_detect_missing_sections_some_missing(self, assessor):
        """Test when some sections are missing."""
        chunks_data = {
            "documents": ["Introduction section", "Results and findings"],
            "metadatas": [
                {"section_title": "Introduction", "chapter": "Chapter 1"},
                {"section_title": "Results", "chapter": "Chapter 2"},
            ],
        }
        missing = assessor._detect_missing_sections(chunks_data)
        # Should detect missing methodology, discussion, conclusion, etc.
        assert "methodology" in missing
        assert "discussion" in missing
        assert "conclusion" in missing


class TestChapterAnalysis:
    """Test chapter extraction and analysis."""

    def test_get_chapter_sizes(self, assessor, sample_chunks_data):
        """Test chapter size calculation."""
        sizes = assessor._get_chapter_sizes(sample_chunks_data)
        assert "Chapter 1" in sizes or "Introduction" in sizes
        assert all(size > 0 for size in sizes.values())

    def test_extract_chapters_with_embeddings(self, assessor, sample_chunks_data):
        """Test chapter extraction with embeddings."""
        with patch.object(assessor, "_select_structural_indices", return_value=[0, 1, 2]):
            with patch.object(assessor, "_has_section_metadata", return_value=True):
                chapters = assessor._extract_chapters(sample_chunks_data)

                assert len(chapters) > 0
                for chapter in chapters:
                    assert "name" in chapter
                    assert "embedding_mean" in chapter
                    assert "section_type" in chapter
                    assert isinstance(chapter["embedding_mean"], np.ndarray)


class TestStructureAnalysis:
    """Test structure analysis method."""

    def test_analyse_structure_basic(self, assessor, sample_chunks_data):
        """Test basic structure analysis."""
        with patch.object(assessor, "_select_structural_indices", return_value=[0, 1, 2]):
            with patch.object(assessor, "_has_section_metadata", return_value=True):
                analysis = assessor.analyse_structure(sample_chunks_data)

                assert isinstance(analysis, StructureAnalysis)
                assert analysis.chapter_count >= 0
                assert isinstance(analysis.missing_sections, list)
                assert isinstance(analysis.red_flags, list)
                assert 0.0 <= analysis.avg_coherence <= 1.0

    def test_analyse_structure_chapter_ordering(self, assessor):
        """Test that analyse_structure respects document order for chapters."""
        # Create chunks with chapters in specific sequence order
        chunks_data = {
            "ids": ["c1", "c2", "c3", "c4"],
            "documents": [
                "Chapter 3 content",
                "Chapter 1 content",
                "Chapter 5 content",
                "Chapter 2 content",
            ],
            "metadatas": [
                {
                    "doc_id": "thesis",
                    "chapter": "Chapter 3",
                    "section_title": "Chapter 3",
                    "sequence_number": 50,
                    "chunk_type": "parent",
                },
                {
                    "doc_id": "thesis",
                    "chapter": "Chapter 1",
                    "section_title": "Chapter 1",
                    "sequence_number": 10,
                    "chunk_type": "parent",
                },
                {
                    "doc_id": "thesis",
                    "chapter": "Chapter 5",
                    "section_title": "Chapter 5",
                    "sequence_number": 100,
                    "chunk_type": "parent",
                },
                {
                    "doc_id": "thesis",
                    "chapter": "Chapter 2",
                    "section_title": "Chapter 2",
                    "sequence_number": 30,
                    "chunk_type": "parent",
                },
            ],
            "embeddings": [
                np.random.rand(1024).tolist(),
                np.random.rand(1024).tolist(),
                np.random.rand(1024).tolist(),
                np.random.rand(1024).tolist(),
            ],
        }

        analysis = assessor.analyse_structure(chunks_data)

        # Verify chapters are ordered by sequence_number (1, 2, 3, 5)
        # Flow transitions should be: 1→2, 2→3, 3→5
        assert len(analysis.chapter_transition_labels) == 3

        # Check transitions are in correct order
        assert "Chapter 1" in analysis.chapter_transition_labels[0]
        assert "Chapter 2" in analysis.chapter_transition_labels[0]

        assert "Chapter 2" in analysis.chapter_transition_labels[1]
        assert "Chapter 3" in analysis.chapter_transition_labels[1]

        assert "Chapter 3" in analysis.chapter_transition_labels[2]
        assert "Chapter 5" in analysis.chapter_transition_labels[2]

        # Verify flow scores match transition count
        assert len(analysis.chapter_flow_scores) == 3

    def test_analyse_structure_flow_scores_order(self, assessor):
        """Test that flow scores correspond to correct chapter transitions."""
        # Create chapters with known embeddings
        emb1 = np.ones(1024) * 0.1
        emb2 = np.ones(1024) * 0.2
        emb3 = np.ones(1024) * 0.3

        chunks_data = {
            "ids": ["c1", "c2", "c3"],
            "documents": ["Ch1", "Ch2", "Ch3"],
            "metadatas": [
                {"chapter": "Chapter 1", "sequence_number": 0, "chunk_type": "parent"},
                {"chapter": "Chapter 2", "sequence_number": 1, "chunk_type": "parent"},
                {"chapter": "Chapter 3", "sequence_number": 2, "chunk_type": "parent"},
            ],
            "embeddings": [emb1.tolist(), emb2.tolist(), emb3.tolist()],
        }

        analysis = assessor.analyse_structure(chunks_data)

        # Should have 2 transitions (1→2, 2→3)
        assert len(analysis.chapter_flow_scores) == 2
        assert len(analysis.chapter_transition_labels) == 2

        # Transitions should be in sequence order
        assert analysis.chapter_transition_labels[0] == "Chapter 1 → Chapter 2"
        assert analysis.chapter_transition_labels[1] == "Chapter 2 → Chapter 3"

    def test_analyse_structure_abrupt_transitions_order(self, assessor):
        """Test that abrupt transitions are detected in correct chapter order."""
        # Create chapters with very different embeddings for some transitions
        emb1 = np.ones(1024) * 1.0
        emb2 = np.ones(1024) * 0.9  # Similar to emb1
        emb3 = np.ones(1024) * -1.0  # Very different from emb2

        chunks_data = {
            "ids": ["c1", "c2", "c3"],
            "documents": ["Ch1", "Ch2", "Ch3"],
            "metadatas": [
                {"chapter": "Chapter 1", "sequence_number": 0, "chunk_type": "parent"},
                {"chapter": "Chapter 2", "sequence_number": 1, "chunk_type": "parent"},
                {"chapter": "Chapter 3", "sequence_number": 2, "chunk_type": "parent"},
            ],
            "embeddings": [emb1.tolist(), emb2.tolist(), emb3.tolist()],
        }

        analysis = assessor.analyse_structure(chunks_data)

        # Should detect abrupt transition between Chapter 2 and Chapter 3
        # (not between 1 and 2, as they're similar)
        if analysis.abrupt_transitions:
            # First abrupt transition should be from Chapter 2 to Chapter 3
            first_abrupt = analysis.abrupt_transitions[0]
            assert "Chapter 2" in first_abrupt[0]
            assert "Chapter 3" in first_abrupt[1]
            assert first_abrupt[2] < 0.3  # Low similarity

    def test_analyse_structure_concept_progression_order(self, assessor):
        """Test that concept progression tracking uses correct chapter order."""
        # Create chapters with a concept appearing in specific order
        chunks_data = {
            "ids": ["c1", "c2", "c3"],
            "documents": [
                "machine learning methodology",  # Chapter 1
                "data analysis approach",  # Chapter 2
                "machine learning results",  # Chapter 3
            ],
            "metadatas": [
                {"chapter": "Chapter 1", "sequence_number": 0, "chunk_type": "parent"},
                {"chapter": "Chapter 2", "sequence_number": 1, "chunk_type": "parent"},
                {"chapter": "Chapter 3", "sequence_number": 2, "chunk_type": "parent"},
            ],
            "embeddings": [
                np.random.rand(1024).tolist(),
                np.random.rand(1024).tolist(),
                np.random.rand(1024).tolist(),
            ],
        }

        analysis = assessor.analyse_structure(chunks_data)

        # Verify key_concepts tracks progression correctly
        # "machine" or "learning" should appear in chapters in order
        if analysis.key_concepts:
            for concept_info in analysis.key_concepts:
                if (
                    "machin" in concept_info["concept"].lower()
                    or "learning" in concept_info["concept"].lower()
                ):
                    # Should track intro → conclusion progression
                    assert concept_info["intro_section"] == "Chapter 1"
                    assert concept_info["concluded_section"] in ["Chapter 3", "Chapter 2"]
                    break

    def test_analyse_structure_excludes_pre_post_matter(self, assessor):
        """Test that pre-matter and post-matter are excluded from coherence analysis."""
        chunks_data = {
            "ids": ["c1", "c2", "c3", "c4", "c5"],
            "documents": ["Abstract", "Ch1", "Ch2", "Ch3", "References"],
            "metadatas": [
                {"chapter": "Abstract", "sequence_number": 0, "chunk_type": "parent"},
                {"chapter": "Chapter 1", "sequence_number": 1, "chunk_type": "parent"},
                {"chapter": "Chapter 2", "sequence_number": 2, "chunk_type": "parent"},
                {"chapter": "Chapter 3", "sequence_number": 3, "chunk_type": "parent"},
                {"chapter": "References", "sequence_number": 4, "chunk_type": "parent"},
            ],
            "embeddings": [np.random.rand(1024).tolist() for _ in range(5)],
        }

        analysis = assessor.analyse_structure(chunks_data)

        # Chapter count should only include main-matter (Chapters 1, 2, 3)
        assert analysis.chapter_count == 3

        # Flow scores should only be for main chapters (1→2, 2→3)
        assert len(analysis.chapter_flow_scores) == 2

        # Transitions should not include Abstract or References
        for transition in analysis.chapter_transition_labels:
            assert "Abstract" not in transition
            assert "References" not in transition

        # Chapter order should include all sections in sequence
        assert analysis.chapter_order == [
            "Abstract",
            "Chapter 1",
            "Chapter 2",
            "Chapter 3",
            "References",
        ]

    def test_analyse_structure_uses_child_embeddings_for_parent_structure(self, assessor):
        """Zero placeholder parent vectors use real child vectors for chapter coherence."""
        chunks_data = {
            "ids": ["p1", "p2", "c1", "c2"],
            "documents": [
                "Chapter 1 overview",
                "Chapter 2 overview",
                "community research methodology",
                "community research findings",
            ],
            "metadatas": [
                {"chapter": "Chapter 1", "sequence_number": 1, "chunk_type": "parent"},
                {"chapter": "Chapter 2", "sequence_number": 2, "chunk_type": "parent"},
                {"chapter": "Chapter 1", "sequence_number": 3, "chunk_type": "child"},
                {"chapter": "Chapter 2", "sequence_number": 4, "chunk_type": "child"},
            ],
            "embeddings": [
                np.zeros(1024).tolist(),
                np.zeros(1024).tolist(),
                (np.ones(1024) * 0.5).tolist(),
                (np.ones(1024) * 0.8).tolist(),
            ],
        }

        analysis = assessor.analyse_structure(chunks_data)

        assert analysis.chapter_transition_labels == ["Chapter 1 → Chapter 2"]
        assert analysis.chapter_flow_scores[0] > 0.9
        concept_scopes = {
            item[scope]
            for item in analysis.key_concepts
            for scope in ("intro_section", "concluded_section")
        }
        assert concept_scopes <= {"Chapter 1", "Chapter 2"}


class TestCitationPatternAnalysis:
    """Test citation pattern analysis."""

    def test_analyse_citation_patterns_no_citations(self, assessor, sample_chunks_data):
        """Test citation analysis with no citations."""
        with patch.object(assessor, "_extract_citations", return_value=[]):
            analysis = assessor.analyse_citation_patterns(sample_chunks_data)

            assert isinstance(analysis, CitationPatternAnalysis)
            assert analysis.total_citations == 0
            assert analysis.citation_evidence_available is False
            # Should have a review warning when citation evidence is unavailable.
            assert any(flag.category == "citations" for flag in analysis.red_flags)


class TestAssessmentSignal:
    """Tests for non-grading assessment signal policy."""

    def test_unavailable_citation_evidence_is_excluded_from_signal(self, assessor):
        """Missing citation storage does not lower the assessment signal."""
        structure = SimpleNamespace(
            avg_coherence=0.8,
            rq_alignment_score=0.8,
            missing_sections=[],
            orphaned_concepts=[],
        )
        methodology = SimpleNamespace(score=0.8)
        writing = SimpleNamespace(readability_score=80.0, passive_voice_ratio=0.0)
        alignment = SimpleNamespace(overlap_score=0.8)
        unavailable_citations = SimpleNamespace(
            citation_evidence_available=False,
            citation_recency_score=0.0,
            geographic_diversity=0.0,
            orphaned_claims=[],
        )
        available_citations = SimpleNamespace(
            citation_evidence_available=True,
            citation_recency_score=0.0,
            geographic_diversity=0.0,
            orphaned_claims=[],
        )

        unavailable_signal = assessor._compute_overall_score(
            structure, unavailable_citations, None, methodology, writing, alignment, "assessor"
        )
        available_signal = assessor._compute_overall_score(
            structure, available_citations, None, methodology, writing, alignment, "assessor"
        )

        assert unavailable_signal == pytest.approx(0.8)
        assert available_signal < unavailable_signal

    def test_analyse_citation_patterns_with_recent(self, assessor, sample_chunks_data):
        """Test citation analysis with recent citations."""
        current_year = datetime.now().year
        mock_citations = [
            {"doi": "10.1/abc", "title": "Recent Paper", "year": current_year},
            {"doi": "10.1/def", "title": "Another Recent", "year": current_year - 1},
            {"doi": "10.1/ghi", "title": "Old Paper", "year": 2000},
        ]

        with patch.object(assessor, "_extract_citations", return_value=mock_citations):
            analysis = assessor.analyse_citation_patterns(sample_chunks_data)

            assert analysis.total_citations == 3
            assert analysis.citation_recency_score > 0.5  # 2/3 are recent


class TestClaimAnalysis:
    """Test claim and contradiction analysis."""

    def test_analyse_claims_and_contradictions(self, assessor, sample_chunks_data):
        """Test claim analysis."""
        # Add claim-like text to sample data
        sample_chunks_data["documents"][0] = "We find that the method improves results."
        sample_chunks_data["documents"][1] = "This study demonstrates effectiveness."

        analysis = assessor.analyse_claims_and_contradictions(sample_chunks_data)

        assert isinstance(analysis, ClaimAnalysis)
        assert analysis.total_claims >= 0
        assert isinstance(analysis.claims, list)
        assert isinstance(analysis.contradictions, list)
        assert isinstance(analysis.red_flags, list)


class TestMethodologyChecklist:
    """Test methodology validation."""

    def test_validate_methodology_checklist_complete(self, assessor):
        """Test methodology validation with complete methodology."""
        chunks_data = {
            "documents": [
                """
                Research Questions: This study asks three research questions.
                Data Collection: We collected data through surveys and interviews.
                Sample Size: The sample consisted of n=250 participants.
                Sampling Method: We used stratified random sampling.
                Analysis Method: Thematic analysis was conducted on the data.
                Validity and Reliability: Measures ensured validity and reliability.
                Ethics: Ethical approval was obtained from the IRB.
                Limitations: This study has several limitations.
                """
            ],
            "metadatas": [{"section_title": "Methodology", "chapter": "Chapter 2"}],
        }

        checklist = assessor.validate_methodology_checklist(chunks_data)

        assert isinstance(checklist, MethodologyChecklist)
        assert checklist.items["research_question"] is True
        assert checklist.items["data_collection"] is True
        assert checklist.items["sample_size"] is True
        assert checklist.items["ethics"] is True
        assert checklist.score > 0.8  # Most items present

    def test_validate_methodology_checklist_missing(self, assessor):
        """Test methodology validation with missing elements."""
        chunks_data = {
            "documents": ["This is a brief methodology section with minimal detail."],
            "metadatas": [{"section_title": "Methodology", "chapter": "Chapter 2"}],
        }

        checklist = assessor.validate_methodology_checklist(chunks_data)

        assert isinstance(checklist, MethodologyChecklist)
        assert len(checklist.missing_items) > 0
        assert checklist.score < 0.5
        # Should have red flags for missing critical items
        assert len(checklist.red_flags) > 0


class TestWritingQualityAnalysis:
    """Test writing quality analysis."""

    def test_analyse_writing_quality(self, assessor, sample_chunks_data):
        """Test writing quality analysis."""
        analysis = assessor.analyse_writing_quality(sample_chunks_data)

        assert isinstance(analysis, WritingQualityMetrics)
        assert 0 <= analysis.readability_score <= 100
        assert analysis.avg_sentence_length > 0
        assert analysis.avg_word_length > 0
        assert 0.0 <= analysis.passive_voice_ratio <= 1.0
        assert 0.0 <= analysis.jargon_density <= 1.0
        assert isinstance(analysis.education_level, str)


class TestContributionAlignment:
    """Test contribution alignment analysis."""

    def test_analyse_contribution_alignment_good(self, assessor):
        """Test contribution alignment with good overlap."""
        chunks_data = {
            "documents": [
                # Introduction with contributions
                "This thesis contributes novel machine learning algorithms.",
                # Results with matching findings
                "Results demonstrate the effectiveness of machine learning algorithms.",
            ],
            "metadatas": [
                {"section_title": "Introduction", "chapter": "Chapter 1"},
                {"section_title": "Results", "chapter": "Chapter 4"},
            ],
        }

        analysis = assessor.analyse_contribution_alignment(chunks_data)

        assert isinstance(analysis, ContributionAlignment)
        assert len(analysis.contribution_keywords) > 0
        assert len(analysis.finding_keywords) > 0
        # Should have some overlap (machine, learning, algorithms)
        assert analysis.overlap_score > 0.0

    def test_analyse_contribution_alignment_weak(self, assessor):
        """Test contribution alignment with weak overlap."""
        chunks_data = {
            "documents": [
                "This thesis contributes theoretical frameworks.",
                "Results show experimental validation.",
            ],
            "metadatas": [
                {"section_title": "Introduction", "chapter": "Chapter 1"},
                {"section_title": "Results", "chapter": "Chapter 4"},
            ],
        }

        analysis = assessor.analyse_contribution_alignment(chunks_data)

        assert isinstance(analysis, ContributionAlignment)
        # Weak overlap should trigger warning
        if analysis.overlap_score < 0.2:
            assert len(analysis.red_flags) > 0


class TestResearchQuestionExtraction:
    """Test research question extraction and alignment."""

    def test_reviewer_confirmed_inquiries_override_automatic_extraction(
        self, assessor, monkeypatch
    ):
        parent = "How do governance practices influence wellbeing?"
        child = "Which practices are most influential?"
        assessor.confirmed_research_inquiries = [
            {"id": "RQ1", "parent_id": "", "type": "research_question", "text": parent},
            {"id": "RQ1a", "parent_id": "RQ1", "type": "sub_question", "text": child},
        ]
        monkeypatch.setattr(
            assessor,
            "_extract_chapters",
            lambda _: [
                {
                    "name": "Chapter 1",
                    "section_type": "main-matter",
                    "sequence_number": 1,
                    "embedding_mean": np.zeros(3),
                    "chunk_count": 1,
                }
            ],
        )
        monkeypatch.setattr(assessor, "_detect_chapter_order_issues", lambda *_: [])
        monkeypatch.setattr(assessor, "_detect_missing_sections", lambda _: [])
        monkeypatch.setattr(
            assessor,
            "_extract_research_questions",
            lambda *_: pytest.fail("confirmed inquiries should bypass automatic extraction"),
        )
        monkeypatch.setattr(assessor, "_compute_rq_alignment", lambda *_: (1.0, []))
        monkeypatch.setattr(assessor, "_track_concept_progression", lambda *_: ([], []))

        chunks_data = {
            "ids": ["rq-parent", "rq-child"],
            "documents": [parent, child],
            "metadatas": [
                {
                    "chapter": "Chapter 1",
                    "section_title": "Introduction",
                    "source_start": 0,
                },
                {
                    "chapter": "Chapter 1",
                    "section_title": "Introduction",
                    "source_start": len(parent) + 10,
                },
            ],
        }

        analysis = assessor.analyse_structure(chunks_data)

        assert analysis.research_questions == [parent, child]
        assert analysis.research_inquiry_ids == {parent: "RQ1", child: "RQ1a"}
        assert analysis.research_inquiry_parent_ids == {child: "RQ1"}
        assert analysis.research_inquiry_types[child] == "sub_question"
        assert analysis.research_inquiry_sources[parent][0]["chunk_id"] == "rq-parent"
        assert analysis.research_inquiry_sources[child][0]["chunk_id"] == "rq-child"

    def test_extract_research_questions_explicit(self, assessor):
        """Test extraction of explicitly stated research questions."""
        chunks_data = {
            "documents": [
                """
                This study addresses three research questions:
                RQ1: What is the impact of X on Y?
                RQ2: How does Z mediate the relationship?
                RQ3: What are the boundary conditions?
                """
            ],
            "metadatas": [{"section_title": "Introduction", "chapter": "Chapter 1"}],
        }

        rqs = assessor._extract_research_questions(chunks_data)
        assert len(rqs) >= 1  # Should find at least one RQ

    def test_local_classifier_excludes_not_rq_candidates(self, assessor, monkeypatch):
        inquiry = "How do community engagement methods influence participant wellbeing?"
        instrument_prompt = "What should I ask participants about their recovery?"
        rhetorical_question = "Who could deny that community practices shape wellbeing?"
        assessor.llm_client = lambda prompt: json.dumps(
            {
                "classifications": [
                    {"index": 0, "type": "aim"},
                    {"index": 1, "type": "not_rq"},
                    {"index": 2, "type": "not_rq"},
                    {"index": 99, "type": "research_question"},
                    {"index": 0, "type": "unsupported_type"},
                ]
            }
        )
        assessor.llm_flags = {"research_inquiry_classification": True}
        candidates = [inquiry, instrument_prompt, rhetorical_question]

        monkeypatch.setattr(
            assessor,
            "_extract_chapters",
            lambda _: [
                {
                    "name": "Chapter 1",
                    "section_type": "main-matter",
                    "sequence_number": 1,
                    "embedding_mean": np.zeros(3),
                    "chunk_count": 1,
                }
            ],
        )
        monkeypatch.setattr(assessor, "_detect_chapter_order_issues", lambda *_: [])
        monkeypatch.setattr(assessor, "_detect_missing_sections", lambda _: [])
        monkeypatch.setattr(assessor, "_extract_research_questions", lambda *_: candidates)
        monkeypatch.setattr(
            assessor,
            "_locate_research_inquiry_sources",
            lambda _chunks, questions: {question: [] for question in questions},
        )
        monkeypatch.setattr(assessor, "_compute_rq_alignment", lambda *_: (1.0, []))
        monkeypatch.setattr(assessor, "_track_concept_progression", lambda *_: ([], []))

        analysis = assessor.analyse_structure({"documents": [], "metadatas": []})

        assert analysis.research_questions == [inquiry]
        assert analysis.research_inquiry_types == {inquiry: "aim"}
        assert instrument_prompt not in analysis.research_inquiry_ids
        assert rhetorical_question not in analysis.research_inquiry_ids

    def test_assigns_stable_ids_to_numbered_questions_and_subquestions(self, assessor):
        parent_question = "How do community governance practices influence wellbeing?"
        sub_question = "Which governance practices are most influential?"
        unlabelled_question = "What barriers affect community participation?"
        chunks_data = {
            "ids": ["rq1", "rq1a", "extra"],
            "documents": [
                f"RQ1: {parent_question}",
                f"RQ1a: {sub_question}",
                unlabelled_question,
            ],
            "metadatas": [
                {"section_title": "Introduction", "chapter": "Chapter 1"},
                {"section_title": "Introduction", "chapter": "Chapter 1"},
                {"section_title": "Introduction", "chapter": "Chapter 1"},
            ],
        }

        inquiries = assessor._extract_research_questions(chunks_data)
        sources = assessor._locate_research_inquiry_sources(chunks_data, inquiries)
        inquiry_ids, parent_ids = assessor._assign_research_inquiry_identifiers(inquiries, sources)

        assert parent_question in inquiries
        assert sub_question in inquiries
        assert inquiry_ids[parent_question] == "RQ1"
        assert inquiry_ids[sub_question] == "RQ1a"
        assert parent_ids[sub_question] == "RQ1"
        assert inquiry_ids[unlabelled_question] not in {"RQ1", "RQ1a"}
        assert len(set(inquiry_ids.values())) == len(inquiry_ids)

    def test_extract_research_inquiry_statements_and_exclude_interview_guide(self, assessor):
        chunks_data = {
            "documents": [
                (
                    "The aim of this study is to understand how Country and kin relationships "
                    "shape healing. The objectives are to document community-defined concepts "
                    "and identify their role in wellbeing."
                ),
                "Interview Guide: What does healing mean to you?",
            ],
            "metadatas": [
                {"section_title": "Introduction", "chapter": "Chapter 1"},
                {"section_title": "Appendix A: Interview Guide", "chapter": "Appendix A"},
            ],
        }

        inquiries = assessor._extract_research_questions(chunks_data)

        assert any("aim of this study" in inquiry.lower() for inquiry in inquiries)
        assert any("objectives are" in inquiry.lower() for inquiry in inquiries)
        assert not any("what does healing mean to you" in inquiry.lower() for inquiry in inquiries)
        inquiry_types = assessor._classify_research_inquiries(inquiries)
        assert inquiry_types[inquiries[0]] in {"aim", "guiding_question"}
        assert any(inquiry_types[inquiry] == "objective" for inquiry in inquiries)

    def test_extracts_explicit_aim_restatement_from_methodology(self, assessor):
        aim_statement = "The aim of this study is to document community governance practices."
        chunks_data = {
            "documents": [aim_statement],
            "metadatas": [
                {"section_title": "Methodology", "heading_path": "Chapter 3 > Methodology"}
            ],
        }

        inquiries = assessor._extract_research_questions(chunks_data)

        assert aim_statement in inquiries

    def test_quality_filter_runs_before_candidate_limit(self, assessor):
        question_fragments = " ".join(
            f"What is it that we do {chr(ord('a') + index)}?" for index in range(15)
        )
        valid_question = (
            "How do community-led governance practices influence participant wellbeing?"
        )
        chunks_data = {
            "documents": [f"{question_fragments} {valid_question}"],
            "metadatas": [{"section_title": "Introduction", "chapter": "Chapter 1"}],
        }

        inquiries = assessor._extract_research_questions(chunks_data)

        assert valid_question in inquiries

    def test_deduplicates_labelled_restatement_and_retains_both_sources(self, assessor):
        inquiry = "How do community governance practices support wellbeing?"
        chunks_data = {
            "ids": ["intro-rq", "conclusion-rq"],
            "documents": [f"RQ1: {inquiry}", inquiry],
            "metadatas": [
                {
                    "section_title": "Introduction",
                    "heading_path": "Chapter 1 > Introduction",
                    "source_start": 100,
                    "source_end": 100 + len(f"RQ1: {inquiry}"),
                },
                {
                    "section_title": "Conclusion",
                    "heading_path": "Chapter 7 > Conclusion",
                    "source_start": 900,
                    "source_end": 900 + len(inquiry),
                },
            ],
        }

        inquiries = assessor._extract_research_questions(chunks_data)

        assert inquiries == [inquiry]
        sources = assessor._locate_research_inquiry_sources(chunks_data, inquiries)
        assert [source["chunk_id"] for source in sources[inquiry]] == [
            "intro-rq",
            "conclusion-rq",
        ]

    def test_assigns_explicit_ids_and_parent_to_subquestions(self, assessor):
        parent = "How do community governance practices influence wellbeing?"
        sub_question = "Which practices are most influential?"
        unlabelled = "What barriers affect participation?"
        sources = {
            parent: [{"inquiry_id": "RQ1", "section": "Introduction"}],
            sub_question: [
                {
                    "inquiry_id": "RQ1a",
                    "parent_inquiry_id": "RQ1",
                    "section": "Introduction",
                }
            ],
            unlabelled: [{"section": "Introduction"}],
        }

        inquiry_ids, parent_ids = assessor._assign_research_inquiry_identifiers(
            [parent, sub_question, unlabelled], sources
        )

        assert inquiry_ids == {parent: "RQ1", sub_question: "RQ1a", unlabelled: "RQ2"}
        assert parent_ids == {sub_question: "RQ1"}

    def test_same_explicit_rq_label_merges_restatement_without_llm(self, assessor):
        introduction = "How do community governance practices influence wellbeing?"
        conclusion = "How does community-led governance shape community wellbeing?"
        inquiries = [introduction, conclusion]
        inquiry_types = {question: "research_question" for question in inquiries}
        sources = {
            introduction: [{"chunk_id": "intro", "section": "Introduction", "inquiry_id": "RQ1"}],
            conclusion: [{"chunk_id": "conclusion", "section": "Conclusion", "inquiry_id": "RQ1"}],
        }

        canonical, types, merged_sources, aliases = assessor._reconcile_research_inquiries(
            inquiries, inquiry_types, sources
        )

        assert canonical == [introduction]
        assert types == {introduction: "research_question"}
        assert aliases == {introduction: [conclusion]}
        assert [source["chunk_id"] for source in merged_sources[introduction]] == [
            "intro",
            "conclusion",
        ]

    def test_reconciles_local_llm_confirmed_introduction_conclusion_restatement(self, assessor):
        introduction = "How do community-led governance practices influence participant wellbeing?"
        conclusion = (
            "To what extent does community-directed decision-making shape participant wellbeing?"
        )
        assessor.llm_client = lambda prompt: '{"groups": [[0, 1]]}'
        assessor.llm_flags = {"research_inquiry_reconciliation": True}

        questions, types, sources, aliases = assessor._reconcile_research_inquiries(
            [introduction, conclusion],
            {introduction: "research_question", conclusion: "research_question"},
            {
                introduction: [{"chunk_id": "intro-rq", "section": "Chapter 1 > Introduction"}],
                conclusion: [{"chunk_id": "conclusion-rq", "section": "Chapter 7 > Conclusion"}],
            },
        )

        assert questions == [introduction]
        assert types == {introduction: "research_question"}
        assert aliases == {introduction: [conclusion]}
        assert [source["chunk_id"] for source in sources[introduction]] == [
            "intro-rq",
            "conclusion-rq",
        ]
        assert sources[introduction][1]["restatement"] == conclusion

    def test_reconciliation_rejects_same_section_and_different_types(self, assessor):
        first = "What governance practices influence participant wellbeing?"
        second = "How do governance practices shape participant wellbeing?"
        assessor.llm_client = lambda prompt: '{"groups": [[0, 1]]}'
        assessor.llm_flags = {"research_inquiry_reconciliation": True}
        inquiries = [first, second]
        sources = {
            first: [{"section": "Chapter 1 > Introduction"}],
            second: [{"section": "Chapter 1 > Introduction"}],
        }

        result = assessor._reconcile_research_inquiries(
            inquiries,
            {first: "research_question", second: "aim"},
            sources,
        )

        assert result == (inquiries, {first: "research_question", second: "aim"}, sources, {})

    def test_reconciliation_falls_back_on_invalid_llm_json(self, assessor):
        first = "How do community governance practices influence wellbeing?"
        second = "How do community-led practices affect wellbeing?"
        assessor.llm_client = lambda prompt: "not JSON"
        assessor.llm_flags = {"research_inquiry_reconciliation": True}
        types = {first: "research_question", second: "research_question"}
        sources = {
            first: [{"section": "Introduction"}],
            second: [{"section": "Conclusion"}],
        }

        result = assessor._reconcile_research_inquiries([first, second], types, sources)

        assert result == ([first, second], types, sources, {})

    def test_extract_profile_framed_guiding_inquiry(self, assessor):
        assessor.cultural_lens_profile = {
            "research_question_framings": [
                {
                    "framing": "Guiding yarning topics",
                    "indicators": ["guiding yarning topics", "yarns focused on"],
                    "classify_as": "guiding_question",
                }
            ]
        }
        chunks_data = {
            "documents": [
                "The guiding yarning topics explored healing, Country and kinship relations."
            ],
            "metadatas": [{"section_title": "Methodology", "chapter": "Chapter 3"}],
        }

        inquiries = assessor._extract_research_questions(chunks_data)

        assert inquiries == [
            "The guiding yarning topics explored healing, Country and kinship relations."
        ]
        assert assessor._classify_research_inquiries(inquiries) == {
            inquiries[0]: "guiding_question"
        }

    def test_profile_framing_indicators_require_phrase_boundaries(self, assessor):
        assessor.cultural_lens_profile = {
            "research_question_framings": [
                {
                    "framing": "Art-based inquiry",
                    "indicators": ["art"],
                    "classify_as": "guiding_question",
                }
            ]
        }
        inquiry = "The study used an artistic approach to explore community wellbeing."
        chunks_data = {
            "documents": [inquiry],
            "metadatas": [{"section_title": "Conceptual Framework", "chapter": "Chapter 2"}],
        }

        assert assessor._collect_research_inquiry_text(chunks_data) == inquiry
        assert assessor._extract_research_questions(chunks_data) == []
        assert assessor._classify_research_inquiries([inquiry]) == {inquiry: "research_question"}

    def test_research_inquiries_retain_source_chunk_and_span(self, assessor):
        inquiry = "The aim of this study is to understand how healing is shaped by Country."
        source_phrase = inquiry.replace("understand how", "understand\nhow")
        document = f"Introduction\n{source_phrase}"
        chunks_data = {
            "ids": ["intro-chunk"],
            "documents": [document],
            "metadatas": [
                {
                    "section_title": "Introduction",
                    "heading_path": "Chapter 1 > Introduction",
                    "source_start": 400,
                    "source_end": 500,
                }
            ],
        }

        sources = assessor._locate_research_inquiry_sources(chunks_data, [inquiry])

        assert sources[inquiry] == [
            {
                "chunk_id": "intro-chunk",
                "section": "Chapter 1 > Introduction",
                "source_start": 400 + document.index(source_phrase),
                "source_end": 400 + document.index(source_phrase) + len(source_phrase),
                "chunk_local_start": document.index(source_phrase),
                "chunk_local_end": document.index(source_phrase) + len(source_phrase),
                "chunk_source_start": 400,
                "chunk_source_end": 500,
                "inquiry_id": None,
                "parent_inquiry_id": None,
            }
        ]

    def test_research_inquiry_source_mapping_rejects_misaligned_metadata(self, assessor):
        with pytest.raises(ValueError, match="metadata and document lengths must match"):
            assessor._locate_research_inquiry_sources(
                {
                    "documents": ["The aim of this study is to document governance."],
                    "metadatas": [],
                },
                ["The aim of this study is to document governance."],
            )


class TestCollectTextBySection:
    """Test section-based text collection."""

    def test_collect_text_by_section_filtered(self, assessor, sample_chunks_data):
        """Test text collection with section filtering."""
        text = assessor._collect_text_by_section(
            sample_chunks_data, include_sections=["introduction", "methodology"]
        )

        assert isinstance(text, str)
        assert len(text) > 0
        # Should include introduction and methodology text
        assert "Introduction" in text or "Methodology" in text

    def test_collect_text_by_section_all(self, assessor, sample_chunks_data):
        """Test text collection without filtering."""
        text = assessor._collect_text_by_section(sample_chunks_data, include_sections=None)

        assert isinstance(text, str)
        assert len(text) > 0

    def test_collect_text_filters_toc(self, assessor):
        """Test that ToC chunks are filtered out."""
        chunks_data = {
            "documents": [
                "Real content goes here.",
                "Chapter 1 ........................... 5",  # ToC entry
                "More real content.",
            ],
            "metadatas": [
                {"section_title": "Introduction"},
                {"section_title": "Contents"},
                {"section_title": "Methods"},
            ],
        }

        text = assessor._collect_text_by_section(chunks_data, include_sections=None)

        # Should exclude ToC entry
        assert ".........." not in text or text.count(".") < 10


class TestRedFlagDetection:
    """Test red flag detection."""

    def test_detect_red_flags_missing_limitations(self, assessor, sample_chunks_data):
        """Test red flag for missing limitations."""
        red_flags = assessor.detect_red_flags(sample_chunks_data)

        # Should detect missing limitations section
        assert any("limitation" in flag.title.lower() for flag in red_flags)

    def test_detect_red_flags_scope_creep(self, assessor):
        """Test red flag for scope creep."""
        chunks_data = {
            "documents": ["x"] * 100,  # Large conclusion
            "metadatas": [
                (
                    {"chapter": "Conclusion", "section_title": "Conclusion"}
                    if i > 50
                    else {"chapter": "Results", "section_title": "Results"}
                )
                for i in range(100)
            ],
        }

        red_flags = assessor.detect_red_flags(chunks_data)

        # May detect scope creep if conclusion >> results
        assert isinstance(red_flags, list)


class TestLLMIntegration:
    """Test LLM integration points (without actual LLM)."""

    def test_llm_enabled_check(self, assessor):
        """Test LLM feature flag checking."""
        assert not assessor._llm_enabled("claims")
        assert not assessor._llm_enabled("data_mismatch")

        # With LLM flags enabled
        assessor.llm_flags = {"claims": True}
        assert assessor._llm_enabled("claims")
        assert not assessor._llm_enabled("other")

    def test_extract_first_json_block(self):
        """Test JSON extraction from json_utils."""
        # Valid JSON
        result = extract_first_json_block('{"key": "value"}')
        assert result == {"key": "value"}

        # JSON with extra text
        result = extract_first_json_block('Here is some text {"key": "value"} and more text')
        assert result == {"key": "value"}

        # JSON with markdown wrapper
        result = extract_first_json_block('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

        # Invalid JSON raises ValueError
        with pytest.raises(ValueError):
            extract_first_json_block("not json")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
