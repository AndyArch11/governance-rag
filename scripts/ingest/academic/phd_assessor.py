"""
PhD Quality Assessment Module

Provides AI-driven assessment of PhD thesis to identify potential issues:
- Chapter flow coherence (semantic continuity)
- Missing required sections
- Citation pattern analysis
- Basic red flags (scope creep, missing limitations)
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from scripts.ingest.academic.terminology import get_all_stopwords
from scripts.search.text_preprocessing import PreprocessingStrategy, TextPreprocessor
from scripts.utils.embedding_model_config import EXPECTED_EMBEDDING_DIM
from scripts.utils.json_utils import extract_first_json_block
from scripts.utils.llm_instrumentation import invoke_with_usage

# Cache stopwords at module level for efficiency
_STOPWORDS = get_all_stopwords()

# Cache stemmer for concept normalisation
_STEMMER = TextPreprocessor(
    strategy=PreprocessingStrategy.STEM_PORTER,
    remove_stopwords=False,  # We handle stopwords separately
    min_token_length=1,
)


def _contains_whole_phrase(text: str, phrase: str) -> bool:
    """Return whether text contains a case-insensitive, whole-phrase match."""
    escaped_words = r"\s+".join(re.escape(word) for word in phrase.split())
    return bool(escaped_words and re.search(rf"(?<!\w){escaped_words}(?!\w)", text, re.IGNORECASE))


# Caveat: these capitalisation overrides follow Australian context and
# may not apply to uses of aboriginal/indigenous in other regions.
CAPITALISATION_OVERRIDES = {
    "aboriginal": "Aboriginal",
    "aboriginality": "Aboriginality",
    "indigenous": "Indigenous",
    "indigeneity": "Indigeneity",
    "first-nations": "First Nations",
    "first_nations": "First Nations",
    "firstnations": "First Nations",
    "aboriginal and torres strait islander": "Aboriginal and Torres Strait Islander",
    "aboriginal and torres strait islanders": "Aboriginal and Torres Strait Islanders",
    "torres strait islander": "Torres Strait Islander",
    "torres-strait-islander": "Torres Strait Islander",
    "torres_strait_islander": "Torres Strait Islander",
    "torresstraitislander": "Torres Strait Islander",
    "australia": "Australia",
    "australian": "Australian",
    "australians": "Australians",
}


@dataclass
class RedFlag:
    """Red flag detected in thesis."""

    severity: str  # 'critical', 'warning', 'info'
    category: str  # 'structure', 'methodology', 'citations', 'scope', 'consistency', 'writing', 'contribution'
    title: str
    description: str
    location: Optional[str] = None  # Chapter/section reference
    suggestion: Optional[str] = None


@dataclass
class StructureAnalysis:
    """Results from structural coherence analysis."""

    chapter_count: int
    chapter_flow_scores: List[float]  # Cosine similarity between consecutive chapters
    chapter_transition_labels: List[
        str
    ]  # Labels for chapter transitions, e.g., "Chapter 1 → Chapter 2"
    abrupt_transitions: List[Tuple[str, str, float]]  # (chapter1, chapter2, similarity)
    missing_sections: List[str]
    avg_coherence: float
    chapter_order: List[str]  # Ordered list of chapter/section labels
    chapter_order_issues: List[str]  # Source-order metadata requiring human review
    research_questions: List[str]  # Extracted research questions
    rq_alignment_score: float  # 0-1, how well findings address RQs
    unaddressed_rqs: List[str]  # RQs not mentioned in findings/conclusion
    key_concepts: List[
        Dict[str, Any]
    ]  # Concept tracking: {concept, intro_section, developed_sections, concluded_section}
    orphaned_concepts: List[str]  # Concepts introduced but not developed/concluded
    red_flags: List[RedFlag]
    research_inquiry_types: Dict[str, str] = field(default_factory=dict)
    research_inquiry_sources: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    research_inquiry_aliases: Dict[str, List[str]] = field(default_factory=dict)
    research_inquiry_ids: Dict[str, str] = field(default_factory=dict)
    research_inquiry_parent_ids: Dict[str, str] = field(default_factory=dict)


@dataclass
class CitationPatternAnalysis:
    """Results from citation pattern analysis."""

    citation_evidence_available: bool
    total_citations: int
    unique_citations: int
    citation_recency_score: float  # 0-1, based on presence of recent sources
    orphaned_claims: List[str]  # Sections with claims but no citations
    citation_clusters: Dict[str, int]  # Topic -> citation count
    geographic_diversity: float  # 0-1, based on author affiliation diversity
    red_flags: List[RedFlag]


@dataclass
class ClaimAnalysis:
    """Results from claim extraction and contradiction detection."""

    total_claims: int
    claims: List[str]
    contradictions: List[Dict[str, Any]]  # {"claim_a": str, "claim_b": str, "overlap": float}
    orphaned_claims: List[str]  # Claims without supporting citations
    red_flags: List[RedFlag]


@dataclass
class MethodologyChecklist:
    """Results from methodology checklist validation."""

    items: Dict[str, bool]
    missing_items: List[str]
    score: float  # 0-1
    confidence_note: str
    evidence: Dict[str, Dict[str, Any]]
    red_flags: List[RedFlag]


@dataclass
class WritingQualityMetrics:
    """Results from writing quality analysis."""

    readability_score: float  # Flesch Reading Ease
    education_level: str
    avg_sentence_length: float
    avg_word_length: float
    passive_voice_ratio: float
    jargon_density: float
    red_flags: List[RedFlag]


@dataclass
class ContributionAlignment:
    """Results from contribution-finding alignment analysis."""

    contribution_keywords: List[str]
    finding_keywords: List[str]
    overlap_score: float  # 0-1
    unmatched_contributions: List[str]
    unmatched_findings: List[str]
    red_flags: List[RedFlag]


@dataclass
class DataConclusionMismatch:
    """Results from data-to-conclusion mismatch detection."""

    issues: List[Dict[str, Any]]  # {"claim": str, "reason": str}
    red_flags: List[RedFlag]


@dataclass
class CitationMisrepresentation:
    """Results from citation misrepresentation checks."""

    issues: List[Dict[str, Any]]  # {"claim": str, "reason": str}
    red_flags: List[RedFlag]


@dataclass
class BenchmarkingResult:
    """Results from comparative benchmarking."""

    status: str  # 'not_configured' | 'ok'
    notes: str
    metrics: Dict[str, Any]


@dataclass
class ArgumentFlowGraph:
    """Argument flow graph data for visualisation."""

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


ReadinessStatus = Literal[
    "evidence_present",
    "evidence_incomplete",
    "needs_human_review",
    "not_applicable",
]


@dataclass
class ReadinessCriterion:
    """Reviewable evidence for one non-grading examiner-readiness criterion."""

    criterion: str
    status: ReadinessStatus
    confidence: float
    evidence: List[str]
    source_sections: List[str]
    reason: str
    source: Literal["deterministic", "llm", "mixed", "system"]


@dataclass
class ExaminerReadinessAnalysis:
    """Evidence-based readiness assistance for a human thesis assessment."""

    notice: str
    criteria: List[ReadinessCriterion]
    criteria_by_status: Dict[ReadinessStatus, List[str]]
    human_review_priorities: List[str]
    red_flags: List[RedFlag]


@dataclass
class AssessmentReport:
    """Complete PhD quality assessment report."""

    doc_id: str
    assessed_at: datetime
    persona: str  # 'supervisor', 'assessor', 'researcher'
    overall_score: float  # 0-1 assessment signal, not an examination outcome
    score_label: str
    structure_analysis: StructureAnalysis
    citation_analysis: CitationPatternAnalysis
    claim_analysis: ClaimAnalysis
    methodology_checklist: MethodologyChecklist
    writing_quality: WritingQualityMetrics
    contribution_alignment: ContributionAlignment
    data_conclusion_mismatch: DataConclusionMismatch
    citation_misrepresentation: CitationMisrepresentation
    benchmarking: BenchmarkingResult
    argument_flow: ArgumentFlowGraph
    examiner_readiness: ExaminerReadinessAnalysis
    critical_red_flags: List[RedFlag]
    summary: str
    next_steps: List[str]


class PhDQualityAssessor:
    """
    PhD quality assessor using ChromaDB embeddings and LLM analysis.

    - Chapter flow coherence (embedding-based)
    - Missing sections detection (structure-based)
    - Citation recency and clustering (metadata-based)
    - Basic red flags (rule-based)
    - Claim extraction and contradiction detection
    - Methodology checklist validation
    - Writing quality metrics
    - Contribution-finding alignment
    """

    def __init__(
        self,
        chunk_collection,
        llm_client=None,
        llm_flags: Optional[Dict[str, bool]] = None,
        citation_db_path: Optional[str] = None,
        cultural_lens_profile: Optional[Dict[str, Any]] = None,
        confirmed_research_inquiries: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Initialise assessor.

        Args:
            chunk_collection: ChromaDB collection with thesis chunks
            llm_client: Optional LLM client for claim extraction (Phase 2)
            llm_flags: Dict to enable/disable specific LLM features
            citation_db_path: Path to citation graph SQLite database
            cultural_lens_profile: Optional validated profile assigned to this thesis.
            confirmed_research_inquiries: Optional reviewer-confirmed inquiry records.
        """
        self.chunk_collection = chunk_collection
        self.llm_client = llm_client
        self.llm_flags = llm_flags or {}
        self.cultural_lens_profile = cultural_lens_profile
        self.confirmed_research_inquiries = confirmed_research_inquiries

        # Citation database path
        if citation_db_path is None:
            from pathlib import Path

            project_root = Path(__file__).resolve().parent.parent.parent.parent
            citation_db_path = str(project_root / "rag_data" / "academic_citation_graph.db")
        self.citation_db_path = citation_db_path

        # Required sections for PhD thesis
        self.required_sections = {
            "introduction",
            "literature review",
            "methodology",
            "results",
            "discussion",
            "conclusion",
            "limitations",
        }

    def assess_thesis(self, doc_id: str, persona: str = "supervisor") -> AssessmentReport:
        """
        Run complete assessment on PhD thesis.

        Args:
            doc_id: Document ID in ChromaDB
            persona: Assessment perspective ('supervisor', 'assessor', 'researcher')

        Returns:
            AssessmentReport with all findings
        """
        # Get all chunks for this document
        chunks_data = self.chunk_collection.get(
            where={"doc_id": doc_id}, include=["documents", "metadatas", "embeddings"]
        )

        if not chunks_data or not chunks_data["ids"]:
            raise ValueError(f"No chunks found for doc_id: {doc_id}")

        # Run analyses
        structure_analysis = self.analyse_structure(chunks_data)
        citation_analysis = self.analyse_citation_patterns(chunks_data)
        claim_analysis = self.analyse_claims_and_contradictions(chunks_data)
        methodology_checklist = self.validate_methodology_checklist(chunks_data)
        writing_quality = self.analyse_writing_quality(chunks_data)
        contribution_alignment = self.analyse_contribution_alignment(chunks_data)
        data_conclusion_mismatch = self.analyse_data_conclusion_mismatch(chunks_data)
        citation_misrepresentation = self.analyse_citation_misrepresentation(chunks_data)
        benchmarking = self.analyse_benchmarking(chunks_data)
        argument_flow = self.analyse_argument_flow_graph(chunks_data)
        examiner_readiness = self.analyse_examiner_readiness(chunks_data)

        # Collect all red flags
        all_red_flags = (
            structure_analysis.red_flags
            + citation_analysis.red_flags
            + claim_analysis.red_flags
            + methodology_checklist.red_flags
            + writing_quality.red_flags
            + contribution_alignment.red_flags
            + data_conclusion_mismatch.red_flags
            + citation_misrepresentation.red_flags
        )

        # Filter critical flags
        critical_flags = [f for f in all_red_flags if f.severity == "critical"]

        # Compute overall score (weighted)
        overall_score = self._compute_overall_score(
            structure_analysis,
            citation_analysis,
            claim_analysis,
            methodology_checklist,
            writing_quality,
            contribution_alignment,
            persona,
        )

        # Generate summary
        summary = self._generate_summary(
            structure_analysis,
            citation_analysis,
            claim_analysis,
            methodology_checklist,
            writing_quality,
            contribution_alignment,
            critical_flags,
            persona,
        )

        # Generate next steps
        next_steps = self._generate_next_steps(all_red_flags, persona)

        return AssessmentReport(
            doc_id=doc_id,
            assessed_at=datetime.now(timezone.utc),
            persona=persona,
            overall_score=overall_score,
            score_label="Assessment signal for human review; not a grade or examination outcome.",
            structure_analysis=structure_analysis,
            citation_analysis=citation_analysis,
            claim_analysis=claim_analysis,
            methodology_checklist=methodology_checklist,
            writing_quality=writing_quality,
            contribution_alignment=contribution_alignment,
            data_conclusion_mismatch=data_conclusion_mismatch,
            citation_misrepresentation=citation_misrepresentation,
            benchmarking=benchmarking,
            argument_flow=argument_flow,
            examiner_readiness=examiner_readiness,
            critical_red_flags=critical_flags,
            summary=summary,
            next_steps=next_steps,
        )

    def analyse_examiner_readiness(self, chunks_data: Dict[str, Any]) -> ExaminerReadinessAnalysis:
        """Create a non-grading readiness assessment for human review.

        The analysis records evidence for human review and deliberately does not
        infer an examination outcome. Criterion-specific evidence extraction is
        added incrementally by readiness assessment phases.

        Args:
            chunks_data: ChromaDB query result containing thesis chunks.

        Returns:
            Baseline readiness analysis with its human-review boundary.
        """
        publication_applicability = self._detect_thesis_by_publication(chunks_data)
        practice_based_applicability = self._detect_practice_based_thesis(chunks_data)
        criteria = [
            ReadinessCriterion(
                criterion="human_assessment_boundary",
                status="evidence_present",
                confidence=1.0,
                evidence=[],
                source_sections=[],
                reason="Assessment results are screening evidence requiring human review.",
                source="system",
            ),
            self._analyse_research_significance(chunks_data),
            self._analyse_contribution_articulation(chunks_data),
            self._analyse_conceptual_integration(chunks_data),
            self._analyse_methodological_rigour(chunks_data),
            self._analyse_findings_quality(chunks_data),
            self._analyse_contribution_contextualisation(chunks_data),
            publication_applicability,
            self._analyse_publication_thesis_minimum_treatment(
                chunks_data,
                publication_applicability,
            ),
            practice_based_applicability,
            self._analyse_practice_based_thesis_integration(
                chunks_data,
                practice_based_applicability,
            ),
            self._analyse_generative_ai_disclosure(chunks_data),
        ]
        if self.llm_client and self._llm_enabled("readiness_evidence"):
            criteria = self._enrich_readiness_criteria_with_llm(criteria)

        criteria_by_status = self._group_readiness_criteria_by_status(criteria)
        human_review_priorities = sorted(
            (
                criterion.criterion
                for criterion in criteria
                if criterion.status == "needs_human_review"
            ),
            key=lambda criterion_name: next(
                criterion.confidence
                for criterion in criteria
                if criterion.criterion == criterion_name
            ),
        )

        return ExaminerReadinessAnalysis(
            notice=(
                "This readiness analysis supports human assessment only. It does not "
                "prepare an examiner report or determine an examination outcome."
            ),
            criteria=criteria,
            criteria_by_status=criteria_by_status,
            human_review_priorities=human_review_priorities,
            red_flags=[],
        )

    def _group_readiness_criteria_by_status(
        self,
        criteria: List[ReadinessCriterion],
    ) -> Dict[ReadinessStatus, List[str]]:
        """Group readiness criteria by evidence state for human-review presentation.

        Args:
            criteria: List of readiness criteria to group by status.

        Returns:
            Dictionary mapping readiness status to lists of criterion names.
        """
        grouped: Dict[ReadinessStatus, List[str]] = {
            "evidence_present": [],
            "evidence_incomplete": [],
            "needs_human_review": [],
            "not_applicable": [],
        }
        for criterion in criteria:
            grouped[criterion.status].append(criterion.criterion)
        return grouped

    def _enrich_readiness_criteria_with_llm(
        self,
        criteria: List[ReadinessCriterion],
    ) -> List[ReadinessCriterion]:
        """Add source-grounded LLM evidence without changing readiness outcomes.

        The deterministic status remains authoritative. LLM output may add a
        reviewable excerpt and rationale only when the excerpt is verbatim from
        the deterministic evidence supplied to the model.

        Args:
            criteria: List of deterministic readiness criteria.

        Returns:
            List of readiness criteria enriched with LLM evidence.
        """
        enriched_criteria: List[ReadinessCriterion] = []
        for criterion in criteria:
            if criterion.source == "system" or criterion.status == "not_applicable":
                enriched_criteria.append(criterion)
                continue

            llm_evidence = self._extract_readiness_evidence_llm(criterion)
            if llm_evidence is None:
                enriched_criteria.append(criterion)
                continue

            evidence, source_section, reason, confidence = llm_evidence
            criterion.evidence = list(dict.fromkeys(criterion.evidence + [evidence]))[:4]
            criterion.source_sections = list(
                dict.fromkeys(criterion.source_sections + [source_section])
            )[:4]
            criterion.reason = f"{criterion.reason} LLM review note: {reason}"
            criterion.confidence = min(criterion.confidence, confidence)
            criterion.source = "mixed"
            enriched_criteria.append(criterion)

        return enriched_criteria

    def _extract_readiness_evidence_llm(
        self,
        criterion: ReadinessCriterion,
    ) -> Optional[Tuple[str, str, str, float]]:
        """Return one validated LLM evidence item for a readiness criterion.

        Invalid JSON, ungrounded text, or low-confidence output is discarded so
        it cannot change the deterministic screening result.

        Args:
            criterion: ReadinessCriterion to extract LLM evidence for.

        Returns:
            Tuple of (evidence, source_section, reason, confidence) if valid,
            otherwise None.
        """
        if not criterion.evidence or not criterion.source_sections:
            return None

        evidence_text = "\n".join(
            f"[{section}] {evidence}"
            for section, evidence in zip(criterion.source_sections, criterion.evidence)
        )
        prompt = (
            "You are assisting human academic assessment. Do not assign a grade, recommend an "
            "examination outcome, or infer missing evidence. Return JSON only.\n"
            "Select one verbatim supporting excerpt from the supplied evidence and explain why it "
            "is relevant.\n"
            'JSON format: {"evidence": "verbatim excerpt", "section": "source section", '
            '"reason": "brief explanation", "confidence": 0.0}\n\n'
            f"Criterion: {criterion.criterion}\n"
            f"Evidence:\n{evidence_text}"
        )
        try:
            response = self._llm_invoke(prompt)
            data = extract_first_json_block(response) if response else None
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

        if not isinstance(data, dict):
            return None
        evidence = data.get("evidence")
        source_section = data.get("section")
        reason = data.get("reason")
        confidence = data.get("confidence")
        if not isinstance(evidence, str) or not isinstance(source_section, str):
            return None
        if not isinstance(reason, str) or not isinstance(confidence, (int, float)):
            return None

        evidence = " ".join(evidence.split()).strip()
        source_section = source_section.strip()
        reason = " ".join(reason.split()).strip()
        if (
            not evidence
            or not source_section
            or not reason
            or not 0.6 <= float(confidence) <= 1.0
            or source_section not in criterion.source_sections
        ):
            return None

        normalised_source_text = " ".join(" ".join(criterion.evidence).split()).lower()
        if evidence.lower() not in normalised_source_text:
            return None

        return evidence, source_section, reason, float(confidence)

    def _analyse_research_significance(self, chunks_data: Dict[str, Any]) -> ReadinessCriterion:
        """Identify evidence that research questions are framed as significant.

        This is a deterministic screening check. It identifies explicit framing
        evidence for a human reviewer; it does not judge the academic merit of
        the research question.

        Args:
            chunks_data: ChromaDB query result containing thesis chunks.
        Returns:
            Criterion-specific readiness analysis for research significance.
        """
        scoped_chunks = self._get_readiness_section_chunks(
            chunks_data,
            [
                "abstract",
                "introduction",
                "research significance",
                "research question",
                "research aim",
                "research objective",
            ],
        )
        if not scoped_chunks:
            return ReadinessCriterion(
                criterion="research_significance",
                status="needs_human_review",
                confidence=0.0,
                evidence=[],
                source_sections=[],
                reason=(
                    "No abstract, introduction, or research-question section could be located. "
                    "A human reviewer should assess research-question significance directly."
                ),
                source="deterministic",
            )

        question_markers = [
            "research question",
            "research questions",
            "research aim",
            "research aims",
            "research objective",
            "research objectives",
            "this study investigates",
            "this research investigates",
        ]
        significance_markers = [
            "significant",
            "importance",
            "important",
            "research gap",
            "gap in",
            "challenge",
            "need for",
            "addresses",
            "policy",
            "professional practice",
            "discipline",
            "wider field",
        ]

        evidence: List[str] = []
        source_sections: List[str] = []
        has_question_framing = False
        for section_label, text in scoped_chunks:
            sentences = self._split_sentences(text)
            has_question_framing = has_question_framing or self._contains_any(
                text, question_markers
            )
            for sentence in sentences:
                if self._contains_any(sentence, question_markers) and self._contains_any(
                    sentence, significance_markers
                ):
                    evidence.append(sentence.strip()[:500])
                    source_sections.append(section_label)

        unique_evidence = list(dict.fromkeys(evidence))[:3]
        unique_sections = list(dict.fromkeys(source_sections))[:3]
        if unique_evidence:
            return ReadinessCriterion(
                criterion="research_significance",
                status="evidence_present",
                confidence=0.8,
                evidence=unique_evidence,
                source_sections=unique_sections,
                reason=(
                    "Located explicit research-question framing linked to significance, a gap, "
                    "or wider context. Human review is required to evaluate its academic merit."
                ),
                source="deterministic",
            )

        if has_question_framing:
            reason = (
                "Located research-question framing but no explicit significance or wider-context "
                "evidence in the scoped sections."
            )
        else:
            reason = "No explicit research-question framing was located in the scoped sections."

        return ReadinessCriterion(
            criterion="research_significance",
            status="needs_human_review",
            confidence=0.5 if has_question_framing else 0.3,
            evidence=[],
            source_sections=[],
            reason=f"{reason} A human reviewer should assess this criterion directly.",
            source="deterministic",
        )

    def _analyse_contribution_articulation(self, chunks_data: Dict[str, Any]) -> ReadinessCriterion:
        """Identify explicit statements of original, significant contribution.

        This check identifies thesis statements for a human reviewer to assess.
        It does not determine whether the claimed contribution is original or
        significant in the relevant discipline.

        Args:
            chunks_data: ChromaDB query result containing thesis chunks.
        Returns:
            Criterion-specific readiness analysis for contribution articulation.
        """
        scoped_chunks = self._get_readiness_section_chunks(
            chunks_data,
            [
                "abstract",
                "introduction",
                "contribution",
                "contributions",
                "discussion",
                "conclusion",
            ],
        )
        if not scoped_chunks:
            return ReadinessCriterion(
                criterion="contribution_articulation",
                status="needs_human_review",
                confidence=0.0,
                evidence=[],
                source_sections=[],
                reason=(
                    "No contribution-relevant sections could be located. A human reviewer should "
                    "identify and assess the claimed contribution directly."
                ),
                source="deterministic",
            )

        contribution_markers = [
            "contribution",
            "contributions",
            "this thesis contributes",
            "this study contributes",
            "this research contributes",
            "the contribution of this thesis",
        ]
        originality_markers = [
            "original",
            "novel",
            "new knowledge",
            "advances knowledge",
            "extends knowledge",
            "significant",
            "substantial",
        ]

        evidence: List[str] = []
        source_sections: List[str] = []
        has_contribution_statement = False
        for section_label, text in scoped_chunks:
            sentences = self._split_sentences(text)
            has_contribution_statement = has_contribution_statement or self._contains_any(
                text, contribution_markers
            )
            for sentence in sentences:
                if self._contains_any(sentence, contribution_markers) and self._contains_any(
                    sentence, originality_markers
                ):
                    evidence.append(sentence.strip()[:500])
                    source_sections.append(section_label)

        unique_evidence = list(dict.fromkeys(evidence))[:3]
        unique_sections = list(dict.fromkeys(source_sections))[:3]
        if unique_evidence:
            return ReadinessCriterion(
                criterion="contribution_articulation",
                status="evidence_present",
                confidence=0.8,
                evidence=unique_evidence,
                source_sections=unique_sections,
                reason=(
                    "Located explicit contribution statements with originality or significance framing. "
                    "A human reviewer must assess the claimed contribution's nature and extent."
                ),
                source="deterministic",
            )

        if has_contribution_statement:
            reason = (
                "Located contribution statements but no explicit originality or significance framing "
                "in the scoped sections."
            )
        else:
            reason = "No explicit contribution statement was located in the scoped sections."

        return ReadinessCriterion(
            criterion="contribution_articulation",
            status="needs_human_review",
            confidence=0.5 if has_contribution_statement else 0.3,
            evidence=[],
            source_sections=[],
            reason=f"{reason} A human reviewer should assess this criterion directly.",
            source="deterministic",
        )

    def _analyse_conceptual_integration(self, chunks_data: Dict[str, Any]) -> ReadinessCriterion:
        """Identify explicit links from framework or literature to research design.

        This screening check locates statements that connect a conceptual or
        theoretical framework, or literature analysis, to objectives or
        hypotheses. It does not judge whether the selected framework is sound.

        Args:
            chunks_data: ChromaDB query result containing thesis chunks.
        Returns:
            Criterion-specific readiness analysis for conceptual integration.
        """
        scoped_chunks = self._get_readiness_section_chunks(
            chunks_data,
            [
                "abstract",
                "introduction",
                "literature",
                "conceptual framework",
                "theoretical framework",
                "research question",
                "research objective",
                "hypothesis",
            ],
        )
        if not scoped_chunks:
            return ReadinessCriterion(
                criterion="conceptual_integration",
                status="needs_human_review",
                confidence=0.0,
                evidence=[],
                source_sections=[],
                reason=(
                    "No literature, framework, or research-design sections could be located. "
                    "A human reviewer should assess conceptual integration directly."
                ),
                source="deterministic",
            )

        framework_markers = [
            "conceptual framework",
            "theoretical framework",
            "theoretical construct",
            "framework",
            "theory",
            "model",
        ]
        literature_markers = [
            "literature review",
            "prior literature",
            "existing literature",
            "scholarship",
        ]
        research_design_markers = [
            "research objective",
            "research objectives",
            "research question",
            "research questions",
            "hypothesis",
            "hypotheses",
            "research aim",
        ]
        integration_markers = [
            "guides",
            "guided",
            "informs",
            "underpins",
            "frames",
            "situates",
            "aligns",
            "linked",
            "connects",
            "based on",
        ]

        evidence: List[str] = []
        source_sections: List[str] = []
        has_framework_or_literature = False
        has_research_design = False
        for section_label, text in scoped_chunks:
            has_framework_or_literature = has_framework_or_literature or self._contains_any(
                text, framework_markers + literature_markers
            )
            has_research_design = has_research_design or self._contains_any(
                text, research_design_markers
            )
            for sentence in self._split_sentences(text):
                has_framework = self._contains_any(sentence, framework_markers)
                has_literature = self._contains_any(sentence, literature_markers)
                has_design = self._contains_any(sentence, research_design_markers)
                has_integration = self._contains_any(sentence, integration_markers)
                if has_design and (
                    (has_framework and has_integration) or (has_framework and has_literature)
                ):
                    evidence.append(sentence.strip()[:500])
                    source_sections.append(section_label)

        unique_evidence = list(dict.fromkeys(evidence))[:3]
        unique_sections = list(dict.fromkeys(source_sections))[:3]
        if unique_evidence:
            return ReadinessCriterion(
                criterion="conceptual_integration",
                status="evidence_present",
                confidence=0.8,
                evidence=unique_evidence,
                source_sections=unique_sections,
                reason=(
                    "Located an explicit link between the conceptual or theoretical framing and "
                    "research objectives or hypotheses. Human review is required to assess its adequacy."
                ),
                source="deterministic",
            )

        if has_framework_or_literature and has_research_design:
            reason = (
                "Located framework or literature discussion and research-design terms, but no explicit "
                "integration statement in the scoped sections."
            )
            confidence = 0.5
        else:
            reason = "Insufficient framework, literature, or research-design evidence was located."
            confidence = 0.3

        return ReadinessCriterion(
            criterion="conceptual_integration",
            status="needs_human_review",
            confidence=confidence,
            evidence=[],
            source_sections=[],
            reason=f"{reason} A human reviewer should assess this criterion directly.",
            source="deterministic",
        )

    def _analyse_methodological_rigour(self, chunks_data: Dict[str, Any]) -> ReadinessCriterion:
        """Identify reviewable methodology justification, detail, and rigour evidence.

        This check screens for description of why an approach was used, how it
        was applied, and how its quality or reproducibility was addressed. It
        does not determine whether the methodology is appropriate or sufficient.

        Args:
            chunks_data: ChromaDB query result containing thesis chunks.
        Returns:
            Criterion-specific readiness analysis for methodological rigour.
        """
        scoped_chunks = self._get_readiness_section_chunks(
            chunks_data,
            [
                "method",
                "methodology",
                "research design",
                "methods",
                "data collection",
                "sampling",
                "analysis",
            ],
        )
        if not scoped_chunks:
            return ReadinessCriterion(
                criterion="methodological_rigour",
                status="needs_human_review",
                confidence=0.0,
                evidence=[],
                source_sections=[],
                reason=(
                    "No methodology-relevant sections could be located. A human reviewer should "
                    "assess methodological rigour directly."
                ),
                source="deterministic",
            )

        signal_groups = {
            "justification": [
                "justified",
                "rationale",
                "appropriate",
                "selected because",
                "chosen because",
                "suitable for",
                "to address the research question",
            ],
            "procedural_detail": [
                "procedure",
                "protocol",
                "step",
                "data were collected",
                "data was collected",
                "participants were recruited",
                "interviews were conducted",
                "survey was administered",
                "analysed using",
                "analyzed using",
                "analysis was conducted",
            ],
            "rigour_or_reproducibility": [
                "validity",
                "reliability",
                "triangulation",
                "audit trail",
                "replicable",
                "reproducible",
                "inter-rater",
                "member checking",
                "robustness",
                "sensitivity analysis",
            ],
        }

        evidence_by_group: Dict[str, List[str]] = {group: [] for group in signal_groups}
        sections_by_group: Dict[str, List[str]] = {group: [] for group in signal_groups}
        for section_label, text in scoped_chunks:
            for sentence in self._split_sentences(text):
                for group, markers in signal_groups.items():
                    if self._contains_any(sentence, markers):
                        evidence_by_group[group].append(sentence.strip()[:500])
                        sections_by_group[group].append(section_label)

        evidenced_groups = [group for group, snippets in evidence_by_group.items() if snippets]
        evidence = list(
            dict.fromkeys(
                snippet for group in evidenced_groups for snippet in evidence_by_group[group]
            )
        )[:3]
        source_sections = list(
            dict.fromkeys(
                section for group in evidenced_groups for section in sections_by_group[group]
            )
        )[:3]

        if len(evidenced_groups) >= 2:
            displayed_groups = ", ".join(group.replace("_", " ") for group in evidenced_groups)
            return ReadinessCriterion(
                criterion="methodological_rigour",
                status="evidence_present",
                confidence=min(0.9, 0.55 + (0.15 * len(evidenced_groups))),
                evidence=evidence,
                source_sections=source_sections,
                reason=(
                    f"Located methodology evidence for {displayed_groups}. Human review is required "
                    "to assess methodological appropriateness, sufficiency, and technical mastery."
                ),
                source="deterministic",
            )

        if evidenced_groups:
            reason = (
                "Located limited methodology evidence for "
                f"{evidenced_groups[0].replace('_', ' ')}, but not enough evidence across other "
                "rigour dimensions."
            )
        else:
            reason = (
                "No explicit methodology justification, detail, or rigour evidence was located."
            )

        return ReadinessCriterion(
            criterion="methodological_rigour",
            status="needs_human_review",
            confidence=0.5 if evidenced_groups else 0.3,
            evidence=evidence,
            source_sections=source_sections,
            reason=f"{reason} A human reviewer should assess this criterion directly.",
            source="deterministic",
        )

    def _analyse_findings_quality(self, chunks_data: Dict[str, Any]) -> ReadinessCriterion:
        """Identify evidence that findings are linked, presented, and interpreted.

        The check looks for explicit links between findings and research
        questions, alongside evidence of logical presentation or interpretation
        through limitations or unexpected results. It does not judge the quality
        or correctness of the findings.

        Args:
            chunks_data: ChromaDB query result containing thesis chunks.
        Returns:
            Criterion-specific readiness analysis for findings quality.
        """
        scoped_chunks = self._get_readiness_section_chunks(
            chunks_data,
            ["results", "findings", "analysis", "discussion", "conclusion"],
        )
        if not scoped_chunks:
            return ReadinessCriterion(
                criterion="findings_quality",
                status="needs_human_review",
                confidence=0.0,
                evidence=[],
                source_sections=[],
                reason=(
                    "No results, findings, analysis, discussion, or conclusion sections could be "
                    "located. A human reviewer should assess findings quality directly."
                ),
                source="deterministic",
            )

        signal_groups = {
            "research_question_linkage": [
                "research question",
                "research questions",
                "research objective",
                "research objectives",
                "hypothesis",
                "hypotheses",
                "address",
                "answer",
                "respond",
            ],
            "logical_presentation": [
                "results are presented",
                "findings are presented",
                "as shown in",
                "table",
                "figure",
                "illustrat",
                "summaris",
            ],
            "limitations_interpretation": [
                "limitation",
                "constraint",
                "threat to validity",
                "interpret",
                "cautio",
            ],
            "unexpected_results": [
                "unexpected",
                "surprising",
                "unanticipated",
                "contrary to",
                "did not support",
                "explained by",
            ],
        }
        linkage_action_markers = ["address", "answer", "respond", "test", "support", "refute"]

        evidence_by_group: Dict[str, List[str]] = {group: [] for group in signal_groups}
        sections_by_group: Dict[str, List[str]] = {group: [] for group in signal_groups}
        for section_label, text in scoped_chunks:
            for sentence in self._split_sentences(text):
                for group, markers in signal_groups.items():
                    has_signal = self._contains_any(sentence, markers)
                    if group == "research_question_linkage":
                        has_signal = has_signal and self._contains_any(
                            sentence, linkage_action_markers
                        )
                    if has_signal:
                        evidence_by_group[group].append(sentence.strip()[:500])
                        sections_by_group[group].append(section_label)

        evidenced_groups = [group for group, snippets in evidence_by_group.items() if snippets]
        evidence = list(
            dict.fromkeys(
                snippet for group in evidenced_groups for snippet in evidence_by_group[group]
            )
        )[:3]
        source_sections = list(
            dict.fromkeys(
                section for group in evidenced_groups for section in sections_by_group[group]
            )
        )[:3]
        has_linkage = "research_question_linkage" in evidenced_groups
        has_interpretation_or_presentation = any(
            group in evidenced_groups
            for group in (
                "logical_presentation",
                "limitations_interpretation",
                "unexpected_results",
            )
        )

        if has_linkage and has_interpretation_or_presentation:
            displayed_groups = ", ".join(group.replace("_", " ") for group in evidenced_groups)
            return ReadinessCriterion(
                criterion="findings_quality",
                status="evidence_present",
                confidence=min(0.9, 0.55 + (0.1 * len(evidenced_groups))),
                evidence=evidence,
                source_sections=source_sections,
                reason=(
                    f"Located findings evidence for {displayed_groups}. Human review is required "
                    "to assess the findings' logic, sufficiency, and interpretation."
                ),
                source="deterministic",
            )

        if evidenced_groups:
            reason = (
                "Located limited findings evidence for "
                f"{', '.join(group.replace('_', ' ') for group in evidenced_groups)}, but no "
                "explicit combination of research-question linkage and presentation or interpretation."
            )
        else:
            reason = "No explicit findings linkage, presentation, or interpretation evidence was located."

        return ReadinessCriterion(
            criterion="findings_quality",
            status="needs_human_review",
            confidence=0.5 if evidenced_groups else 0.3,
            evidence=evidence,
            source_sections=source_sections,
            reason=f"{reason} A human reviewer should assess this criterion directly.",
            source="deterministic",
        )

    def _analyse_contribution_contextualisation(
        self, chunks_data: Dict[str, Any]
    ) -> ReadinessCriterion:
        """Identify positioning of findings against prior work and future opportunities.

        This check identifies statements a human reviewer can use to assess how
        the thesis contextualises its contribution. It does not judge whether
        the interpretation of prior work or proposed future research is valid.

        Args:
            chunks_data: ChromaDB query result containing thesis chunks.
        Returns:
            Criterion-specific readiness analysis for contribution contextualisation.
        """
        scoped_chunks = self._get_readiness_section_chunks(
            chunks_data,
            ["findings", "results", "discussion", "conclusion", "contribution"],
        )
        if not scoped_chunks:
            return ReadinessCriterion(
                criterion="contribution_contextualisation",
                status="needs_human_review",
                confidence=0.0,
                evidence=[],
                source_sections=[],
                reason=(
                    "No findings, discussion, conclusion, or contribution sections could be located. "
                    "A human reviewer should assess contribution contextualisation directly."
                ),
                source="deterministic",
            )

        prior_work_markers = [
            "prior research",
            "previous research",
            "existing literature",
            "prior literature",
            "earlier studies",
            "previous studies",
            "literature",
        ]
        positioning_markers = [
            "support",
            "extend",
            "contradict",
            "challenge",
            "consistent with",
            "in contrast to",
            "builds on",
            "differs from",
        ]
        future_research_markers = [
            "future research",
            "future work",
            "further research",
            "further work",
            "avenue for research",
            "opportunity for research",
            "should investigate",
        ]

        evidence_by_group: Dict[str, List[str]] = {
            "prior_work_positioning": [],
            "future_research": [],
        }
        sections_by_group: Dict[str, List[str]] = {
            "prior_work_positioning": [],
            "future_research": [],
        }
        for section_label, text in scoped_chunks:
            for sentence in self._split_sentences(text):
                if self._contains_any(sentence, prior_work_markers) and self._contains_any(
                    sentence, positioning_markers
                ):
                    evidence_by_group["prior_work_positioning"].append(sentence.strip()[:500])
                    sections_by_group["prior_work_positioning"].append(section_label)
                if self._contains_any(sentence, future_research_markers):
                    evidence_by_group["future_research"].append(sentence.strip()[:500])
                    sections_by_group["future_research"].append(section_label)

        evidenced_groups = [group for group, snippets in evidence_by_group.items() if snippets]
        evidence = list(
            dict.fromkeys(
                snippet for group in evidenced_groups for snippet in evidence_by_group[group]
            )
        )[:3]
        source_sections = list(
            dict.fromkeys(
                section for group in evidenced_groups for section in sections_by_group[group]
            )
        )[:3]

        if len(evidenced_groups) == len(evidence_by_group):
            return ReadinessCriterion(
                criterion="contribution_contextualisation",
                status="evidence_present",
                confidence=0.8,
                evidence=evidence,
                source_sections=source_sections,
                reason=(
                    "Located explicit positioning of findings against prior work and future-research "
                    "opportunities. Human review is required to assess their academic validity."
                ),
                source="deterministic",
            )

        if evidenced_groups:
            reason = (
                "Located evidence for "
                f"{evidenced_groups[0].replace('_', ' ')}, but not for the complementary "
                "contribution-contextualisation dimension."
            )
        else:
            reason = "No explicit prior-work positioning or future-research evidence was located."

        return ReadinessCriterion(
            criterion="contribution_contextualisation",
            status="needs_human_review",
            confidence=0.5 if evidenced_groups else 0.3,
            evidence=evidence,
            source_sections=source_sections,
            reason=f"{reason} A human reviewer should assess this criterion directly.",
            source="deterministic",
        )

    def _detect_thesis_by_publication(self, chunks_data: Dict[str, Any]) -> ReadinessCriterion:
        """Detect a thesis-by-publication form using multiple independent signals.

        Ordinary references to journal articles are not enough to trigger this
        conditional assessment path. The thesis must identify its publication
        form and include evidence of publication-derived content.

        Args:
            chunks_data: ChromaDB query result containing thesis chunks.
        Returns:
            Criterion-specific readiness analysis for thesis-by-publication applicability.
        """
        thesis_form_markers = [
            "thesis by publication",
            "thesis-by-publication",
            "thesis comprises publications",
            "thesis comprises published",
            "thesis includes published",
            "thesis consists of published",
        ]
        publication_content_markers = [
            "included publication",
            "included publications",
            "published work",
            "work in progress for publication",
            "manuscript submitted",
            "manuscript accepted",
            "peer-reviewed publication",
            "journal article",
        ]

        evidence_by_group: Dict[str, List[str]] = {
            "thesis_form": [],
            "publication_content": [],
        }
        sections_by_group: Dict[str, List[str]] = {
            "thesis_form": [],
            "publication_content": [],
        }
        documents = chunks_data.get("documents", [])
        metadatas = chunks_data.get("metadatas", [])
        for index, document in enumerate(documents):
            if self._is_toc_chunk(document):
                continue

            metadata = metadatas[index] if index < len(metadatas) else {}
            section_label = self._get_section_label(metadata, document)
            for sentence in self._split_sentences(document):
                if self._contains_any(sentence, thesis_form_markers):
                    evidence_by_group["thesis_form"].append(sentence.strip()[:500])
                    sections_by_group["thesis_form"].append(section_label)
                if self._contains_any(sentence, publication_content_markers):
                    evidence_by_group["publication_content"].append(sentence.strip()[:500])
                    sections_by_group["publication_content"].append(section_label)

        evidence = list(
            dict.fromkeys(
                snippet
                for group in ("thesis_form", "publication_content")
                for snippet in evidence_by_group[group]
            )
        )[:3]
        source_sections = list(
            dict.fromkeys(
                section
                for group in ("thesis_form", "publication_content")
                for section in sections_by_group[group]
            )
        )[:3]

        if all(evidence_by_group.values()):
            return ReadinessCriterion(
                criterion="thesis_by_publication_applicability",
                status="evidence_present",
                confidence=0.85,
                evidence=evidence,
                source_sections=source_sections,
                reason=(
                    "Detected explicit thesis-by-publication and publication-content evidence. "
                    "Publication-based readiness criteria should be assessed by a human reviewer."
                ),
                source="deterministic",
            )

        return ReadinessCriterion(
            criterion="thesis_by_publication_applicability",
            status="not_applicable",
            confidence=0.8,
            evidence=evidence,
            source_sections=source_sections,
            reason=(
                "No reliable combined evidence of a thesis-by-publication form was located. "
                "Publication-based readiness criteria are not applied."
            ),
            source="deterministic",
        )

    def _analyse_publication_thesis_minimum_treatment(
        self,
        chunks_data: Dict[str, Any],
        applicability: ReadinessCriterion,
    ) -> ReadinessCriterion:
        """Identify guide-aligned minimum-treatment evidence for publication theses.

        The conditional check applies only when the thesis explicitly identifies
        itself as publication-based. It records evidence for human assessment and
        does not determine whether publication-derived material is compliant.

        Args:
            chunks_data: ChromaDB query result containing thesis chunks.
            applicability: ReadinessCriterion indicating whether publication-based
                minimum-treatment criteria should be applied.
        Returns:
            Criterion-specific readiness analysis for publication-thesis minimum treatment.
        """
        if applicability.status != "evidence_present":
            return ReadinessCriterion(
                criterion="publication_thesis_minimum_treatment",
                status="not_applicable",
                confidence=applicability.confidence,
                evidence=[],
                source_sections=[],
                reason=(
                    "Publication-based minimum-treatment criteria are not applied because the "
                    "thesis-by-publication form was not reliably detected."
                ),
                source="deterministic",
            )

        required_elements = {
            "original_introduction_or_literature_review": [
                "independent and original review",
                "original literature review",
                "independent literature review",
            ],
            "framing_chapter": ["framing chapter", "framing framework", "frames the publications"],
            "bridging_narrative": [
                "bridging statement",
                "bridging statements",
                "link chapters",
                "links the chapters",
                "connect chapters",
                "cohesive narrative",
            ],
            "independent_general_discussion": [
                "independent general discussion",
                "general discussion integrates",
                "integrates the findings",
                "integrate the findings",
            ],
        }
        evidence_by_element: Dict[str, List[str]] = {element: [] for element in required_elements}
        sections_by_element: Dict[str, List[str]] = {element: [] for element in required_elements}
        documents = chunks_data.get("documents", [])
        metadatas = chunks_data.get("metadatas", [])
        for index, document in enumerate(documents):
            if self._is_toc_chunk(document):
                continue

            metadata = metadatas[index] if index < len(metadatas) else {}
            section_label = self._get_section_label(metadata, document)
            section_text = " ".join(
                str(metadata.get(field, ""))
                for field in ("section_title", "parent_section", "heading_path", "chapter")
            ).lower()
            for element, markers in required_elements.items():
                if self._contains_any(section_text, markers):
                    evidence_by_element[element].append(section_label)
                    sections_by_element[element].append(section_label)
                for sentence in self._split_sentences(document):
                    if self._contains_any(sentence, markers):
                        evidence_by_element[element].append(sentence.strip()[:500])
                        sections_by_element[element].append(section_label)

        evidenced_elements = [
            element for element, snippets in evidence_by_element.items() if snippets
        ]
        missing_elements = [
            element.replace("_", " ")
            for element in required_elements
            if element not in evidenced_elements
        ]
        evidence = list(
            dict.fromkeys(
                snippet
                for element in evidenced_elements
                for snippet in evidence_by_element[element]
            )
        )[:4]
        source_sections = list(
            dict.fromkeys(
                section
                for element in evidenced_elements
                for section in sections_by_element[element]
            )
        )[:4]

        if len(evidenced_elements) == len(required_elements):
            return ReadinessCriterion(
                criterion="publication_thesis_minimum_treatment",
                status="evidence_present",
                confidence=0.85,
                evidence=evidence,
                source_sections=source_sections,
                reason=(
                    "Located evidence for the publication-based thesis minimum-treatment elements. "
                    "A human reviewer must verify their adequacy and authorship."
                ),
                source="deterministic",
            )

        return ReadinessCriterion(
            criterion="publication_thesis_minimum_treatment",
            status="needs_human_review",
            confidence=0.4 + (0.1 * len(evidenced_elements)),
            evidence=evidence,
            source_sections=source_sections,
            reason=(
                "Located incomplete publication-based minimum-treatment evidence. Missing evidence for: "
                f"{', '.join(missing_elements)}. A human reviewer should assess this criterion directly."
            ),
            source="deterministic",
        )

    def _detect_practice_based_thesis(self, chunks_data: Dict[str, Any]) -> ReadinessCriterion:
        """Detect a practice-based thesis or exegesis using independent signals.

        The detector prevents conditional creative-work checks from applying to
        conventional theses that only use general creative language.

        Args:
            chunks_data: ChromaDB query result containing thesis chunks.
        Returns:
            Criterion-specific readiness analysis for practice-based thesis applicability.
        """
        thesis_form_markers = [
            "practice-based thesis",
            "practice based thesis",
            "practice-based research",
            "practice based research",
            "creative practice research",
            "creative work and exegesis",
            "exegesis",
        ]
        component_markers = [
            "creative work",
            "creative component",
            "practice-based component",
            "practice based component",
            "performance",
            "exhibition",
            "portfolio",
            "durable record",
            "artefact",
            "artifact",
        ]

        evidence_by_group: Dict[str, List[str]] = {
            "thesis_form": [],
            "practical_or_creative_component": [],
        }
        sections_by_group: Dict[str, List[str]] = {
            "thesis_form": [],
            "practical_or_creative_component": [],
        }
        documents = chunks_data.get("documents", [])
        metadatas = chunks_data.get("metadatas", [])
        for index, document in enumerate(documents):
            if self._is_toc_chunk(document):
                continue

            metadata = metadatas[index] if index < len(metadatas) else {}
            section_label = self._get_section_label(metadata, document)
            for sentence in self._split_sentences(document):
                if self._contains_any(sentence, thesis_form_markers):
                    evidence_by_group["thesis_form"].append(sentence.strip()[:500])
                    sections_by_group["thesis_form"].append(section_label)
                if self._contains_any(sentence, component_markers):
                    evidence_by_group["practical_or_creative_component"].append(
                        sentence.strip()[:500]
                    )
                    sections_by_group["practical_or_creative_component"].append(section_label)

        evidence = list(
            dict.fromkeys(
                snippet
                for group in ("thesis_form", "practical_or_creative_component")
                for snippet in evidence_by_group[group]
            )
        )[:3]
        source_sections = list(
            dict.fromkeys(
                section
                for group in ("thesis_form", "practical_or_creative_component")
                for section in sections_by_group[group]
            )
        )[:3]

        if all(evidence_by_group.values()):
            return ReadinessCriterion(
                criterion="practice_based_thesis_applicability",
                status="evidence_present",
                confidence=0.85,
                evidence=evidence,
                source_sections=source_sections,
                reason=(
                    "Detected practice-based or exegesis thesis form and a practical or creative "
                    "component. Practice-based readiness criteria should be reviewed by a human."
                ),
                source="deterministic",
            )

        return ReadinessCriterion(
            criterion="practice_based_thesis_applicability",
            status="not_applicable",
            confidence=0.8,
            evidence=evidence,
            source_sections=source_sections,
            reason=(
                "No reliable combined evidence of a practice-based thesis or exegesis was located. "
                "Practice-based readiness criteria are not applied."
            ),
            source="deterministic",
        )

    def _analyse_practice_based_thesis_integration(
        self,
        chunks_data: Dict[str, Any],
        applicability: ReadinessCriterion,
    ) -> ReadinessCriterion:
        """Identify explicit integration of an exegesis and practical work.

        This conditional check records text that links an exegesis with the
        practical or creative component. It does not evaluate the quality of
        either component or attempt to assess creative work itself.

        Args:
            chunks_data: ChromaDB query result containing thesis chunks.
            applicability: ReadinessCriterion indicating whether practice-based
                integration criteria should be applied.
        Returns:
            Criterion-specific readiness analysis for practice-based thesis integration.
        """
        if applicability.status != "evidence_present":
            return ReadinessCriterion(
                criterion="practice_based_thesis_integration",
                status="not_applicable",
                confidence=applicability.confidence,
                evidence=[],
                source_sections=[],
                reason=(
                    "Practice-based integration criteria are not applied because a practice-based "
                    "thesis or exegesis was not reliably detected."
                ),
                source="deterministic",
            )

        exegesis_markers = ["exegesis", "exegetical"]
        component_markers = [
            "creative work",
            "creative component",
            "practice-based component",
            "practice based component",
            "performance",
            "exhibition",
            "portfolio",
            "artefact",
            "artifact",
        ]
        integration_markers = [
            "integrated",
            "integration",
            "integrated whole",
            "considered together",
            "examined together",
            "relationship between",
            "connects",
            "links",
        ]

        evidence: List[str] = []
        source_sections: List[str] = []
        has_exegesis = False
        has_component = False
        documents = chunks_data.get("documents", [])
        metadatas = chunks_data.get("metadatas", [])
        for index, document in enumerate(documents):
            if self._is_toc_chunk(document):
                continue

            metadata = metadatas[index] if index < len(metadatas) else {}
            section_label = self._get_section_label(metadata, document)
            has_exegesis = has_exegesis or self._contains_any(document, exegesis_markers)
            has_component = has_component or self._contains_any(document, component_markers)
            for sentence in self._split_sentences(document):
                if (
                    self._contains_any(sentence, exegesis_markers)
                    and self._contains_any(sentence, component_markers)
                    and self._contains_any(sentence, integration_markers)
                ):
                    evidence.append(sentence.strip()[:500])
                    source_sections.append(section_label)

        unique_evidence = list(dict.fromkeys(evidence))[:3]
        unique_sections = list(dict.fromkeys(source_sections))[:3]
        if unique_evidence:
            return ReadinessCriterion(
                criterion="practice_based_thesis_integration",
                status="evidence_present",
                confidence=0.8,
                evidence=unique_evidence,
                source_sections=unique_sections,
                reason=(
                    "Located explicit evidence that the exegesis and practical or creative component "
                    "are integrated. Human review is required to assess that integration."
                ),
                source="deterministic",
            )

        if has_exegesis and has_component:
            reason = (
                "Located exegesis and practical or creative component evidence, but no explicit "
                "integration statement."
            )
            confidence = 0.5
        else:
            reason = "Insufficient exegesis or practical-component evidence was located."
            confidence = 0.3

        return ReadinessCriterion(
            criterion="practice_based_thesis_integration",
            status="needs_human_review",
            confidence=confidence,
            evidence=[],
            source_sections=[],
            reason=f"{reason} A human reviewer should assess this criterion directly.",
            source="deterministic",
        )

    def _analyse_generative_ai_disclosure(self, chunks_data: Dict[str, Any]) -> ReadinessCriterion:
        """Identify declared generative-AI use and supporting disclosure evidence.

        The check records disclosures for human review only. It does not infer
        whether AI use was appropriate, permitted, or academically compliant.

        Args:
            chunks_data: ChromaDB query result containing thesis chunks.
        Returns:
            Criterion-specific readiness analysis for generative-AI disclosure.
        """
        ai_tool_markers = [
            "generative ai",
            "artificial intelligence",
            "large language model",
            "chatgpt",
            "claude",
            "copilot",
            "gemini",
            "grok",
            "llm",
            "openai",
            "notebooklm",
            "paperpal",
            "jenni ai",
            "julius ai",
            "quillbot",
        ]
        use_markers = [
            "used",
            "use of",
            "utilised",
            "utilized",
            "employed",
            "assisted by",
            "generated with",
        ]
        non_use_markers = [
            "not used",
            "no generative ai",
            "no artificial intelligence",
            "without using",
            "did not use",
        ]
        disclosure_groups = {
            "tool": ai_tool_markers,
            "purpose": [
                "purpose",
                "used to",
                "utilised to",
                "utilized to",
                "assisted with",
                "for language editing",
                "for proofreading",
                "for coding",
                "for translation",
            ],
            "extent": [
                "extent of use",
                "limited to",
                "only used",
                "used only",
                "exclusively",
                "throughout the thesis",
                "in chapters",
            ],
            "prompts_or_disclosure": [
                "prompt",
                "prompt log",
                "prompt history",
                "ai disclosure",
                "disclosure statement",
                "acknowledgement of ai",
                "declared ai use",
            ],
        }

        evidence_by_group: Dict[str, List[str]] = {group: [] for group in disclosure_groups}
        sections_by_group: Dict[str, List[str]] = {group: [] for group in disclosure_groups}
        declaration_evidence: List[str] = []
        declaration_sections: List[str] = []
        documents = chunks_data.get("documents", [])
        metadatas = chunks_data.get("metadatas", [])
        for index, document in enumerate(documents):
            if self._is_toc_chunk(document):
                continue

            metadata = metadatas[index] if index < len(metadatas) else {}
            section_label = self._get_section_label(metadata, document)
            for sentence in self._split_sentences(document):
                sentence_lower = sentence.lower()
                declares_use = self._contains_any(sentence, ai_tool_markers) and self._contains_any(
                    sentence, use_markers
                )
                if declares_use and not self._contains_any(sentence_lower, non_use_markers):
                    declaration_evidence.append(sentence.strip()[:500])
                    declaration_sections.append(section_label)
                for group, markers in disclosure_groups.items():
                    if self._contains_any(sentence, markers):
                        evidence_by_group[group].append(sentence.strip()[:500])
                        sections_by_group[group].append(section_label)

        if not declaration_evidence:
            return ReadinessCriterion(
                criterion="generative_ai_disclosure",
                status="not_applicable",
                confidence=0.8,
                evidence=[],
                source_sections=[],
                reason=(
                    "No declared generative-AI use was located. AI-use disclosure criteria are not applied."
                ),
                source="deterministic",
            )

        evidenced_groups = [group for group, snippets in evidence_by_group.items() if snippets]
        evidence = list(
            dict.fromkeys(
                declaration_evidence
                + [snippet for group in evidenced_groups for snippet in evidence_by_group[group]]
            )
        )[:4]
        source_sections = list(
            dict.fromkeys(
                declaration_sections
                + [section for group in evidenced_groups for section in sections_by_group[group]]
            )
        )[:4]
        missing_groups = [
            group.replace("_", " ") for group in disclosure_groups if group not in evidenced_groups
        ]

        if len(evidenced_groups) == len(disclosure_groups):
            return ReadinessCriterion(
                criterion="generative_ai_disclosure",
                status="evidence_present",
                confidence=0.8,
                evidence=evidence,
                source_sections=source_sections,
                reason=(
                    "Located a declared generative-AI use with tool, purpose, extent, and prompt or "
                    "disclosure evidence. A human reviewer must verify the disclosure's completeness."
                ),
                source="deterministic",
            )

        return ReadinessCriterion(
            criterion="generative_ai_disclosure",
            status="needs_human_review",
            confidence=0.4 + (0.1 * len(evidenced_groups)),
            evidence=evidence,
            source_sections=source_sections,
            reason=(
                "Located declared generative-AI use with incomplete disclosure evidence. Missing evidence for: "
                f"{', '.join(missing_groups)}. A human reviewer should assess this criterion directly."
            ),
            source="deterministic",
        )

    def _get_readiness_section_chunks(
        self,
        chunks_data: Dict[str, Any],
        section_keywords: List[str],
    ) -> List[Tuple[str, str]]:
        """Return non-ToC chunks from sections relevant to a readiness criterion.

        Args:
            chunks_data: ChromaDB query result with chunks, metadata, embeddings
            section_keywords: List of keywords to identify relevant sections
        Returns:
            List of tuples (section_label, chunk_text) for relevant chunks
        """
        matched_chunks: List[Tuple[str, str]] = []
        documents = chunks_data.get("documents", [])
        metadatas = chunks_data.get("metadatas", [])

        for index, document in enumerate(documents):
            if self._is_toc_chunk(document):
                continue

            metadata = metadatas[index] if index < len(metadatas) else {}
            section_label = self._get_section_label(metadata, document)
            section_text = " ".join(
                str(metadata.get(field, ""))
                for field in ("section_title", "parent_section", "heading_path", "chapter")
            ).lower()
            # Older or child-only records may not carry structural metadata.
            # The heading-derived fallback still lets readiness checks locate
            # the relevant source section without treating the heading itself
            # as substantive evidence.
            if not section_text.strip() and section_label != "Unclassified":
                section_text = section_label.lower()
            if any(keyword in section_text for keyword in section_keywords):
                matched_chunks.append((section_label, document))

        return matched_chunks

    def analyse_structure(self, chunks_data: Dict) -> StructureAnalysis:
        """
        Analyse structural coherence of thesis.

        Checks:
        - Chapter flow (embedding similarity between consecutive chapters)
        - Missing required sections
        - Abrupt topic transitions

        Only main-matter chapters (not pre/post matter) are counted for structural
        coherence metrics to give accurate chapter counts.

        Args:
            chunks_data: ChromaDB query result with chunks, metadata, embeddings

        Returns:
            StructureAnalysis with findings
        """
        # Extract chapter information
        all_chapters = self._extract_chapters(chunks_data)
        chapter_order_issues = self._detect_chapter_order_issues(chunks_data, all_chapters)

        # Filter to only main-matter chapters for coherence analysis
        main_chapters = [ch for ch in all_chapters if ch.get("section_type") == "main-matter"]

        # Check for missing sections
        missing_sections = self._detect_missing_sections(chunks_data)

        # Analyse chapter flow coherence (only for main chapters)
        flow_scores = []
        chapter_transition_labels = []
        abrupt_transitions = []

        for i in range(len(main_chapters) - 1):
            curr_chapter = main_chapters[i]
            next_chapter = main_chapters[i + 1]

            # Compute similarity between chapter embeddings
            similarity = self._compute_chapter_similarity(
                curr_chapter["embedding_mean"], next_chapter["embedding_mean"]
            )

            flow_scores.append(similarity)
            chapter_transition_labels.append(f"{curr_chapter['name']} → {next_chapter['name']}")

            # Flag abrupt transitions (similarity < 0.3)
            if similarity < 0.3:
                abrupt_transitions.append((curr_chapter["name"], next_chapter["name"], similarity))

        # Compute average coherence
        avg_coherence = np.mean(flow_scores) if flow_scores else 0.0

        # Generate red flags
        red_flags = []

        # Missing sections
        if missing_sections:
            red_flags.append(
                RedFlag(
                    severity="critical",
                    category="structure",
                    title="Missing Required Sections",
                    description=f"The following required sections are missing or not detected: {', '.join(missing_sections)}",
                    suggestion="Ensure all standard PhD sections are present and clearly labeled.",
                )
            )

        if chapter_order_issues:
            red_flags.append(
                RedFlag(
                    severity="warning",
                    category="structure",
                    title="Chapter Source Order Requires Review",
                    description="; ".join(chapter_order_issues),
                    suggestion=(
                        "Verify extracted chapter sequence metadata against the source document and "
                        "correct any conflicting chapter order."
                    ),
                )
            )

        # Abrupt transitions
        for chapter1, chapter2, sim in abrupt_transitions:
            red_flags.append(
                RedFlag(
                    severity="warning",
                    category="structure",
                    title=f"Abrupt Transition: {chapter1} → {chapter2}",
                    description=f"Low semantic similarity ({sim:.2f}) between consecutive chapters suggests abrupt topic change.",
                    location=f"{chapter1}/{chapter2}",
                    suggestion="Consider adding transitional text or restructuring to improve flow.",
                )
            )

        # Low overall coherence
        if avg_coherence < 0.5 and len(main_chapters) > 1:
            red_flags.append(
                RedFlag(
                    severity="warning",
                    category="structure",
                    title="Low Overall Coherence",
                    description=f"Average chapter coherence is {avg_coherence:.2f}, indicating potential structural issues.",
                    suggestion="Review chapter organisation and ensure logical progression of ideas.",
                )
            )

        # Extract questions unless a reviewer has confirmed a thesis-specific set.
        if self.confirmed_research_inquiries is None:
            section_types = {chapter["name"]: chapter["section_type"] for chapter in all_chapters}
            research_questions = self._extract_research_questions(chunks_data, section_types)
            research_inquiry_sources = self._locate_research_inquiry_sources(
                chunks_data, research_questions
            )
            research_inquiry_types = self._classify_research_inquiries(
                research_questions, research_inquiry_sources
            )
            rejected_inquiries = {
                inquiry
                for inquiry, inquiry_type in research_inquiry_types.items()
                if inquiry_type == "not_rq"
            }
            if rejected_inquiries:
                research_questions = [
                    inquiry for inquiry in research_questions if inquiry not in rejected_inquiries
                ]
                research_inquiry_types = {
                    inquiry: inquiry_type
                    for inquiry, inquiry_type in research_inquiry_types.items()
                    if inquiry not in rejected_inquiries
                }
                research_inquiry_sources = {
                    inquiry: sources
                    for inquiry, sources in research_inquiry_sources.items()
                    if inquiry not in rejected_inquiries
                }
            (
                research_questions,
                research_inquiry_types,
                research_inquiry_sources,
                research_inquiry_aliases,
            ) = self._reconcile_research_inquiries(
                research_questions,
                research_inquiry_types,
                research_inquiry_sources,
            )
            research_inquiry_ids, research_inquiry_parent_ids = (
                self._assign_research_inquiry_identifiers(
                    research_questions, research_inquiry_sources
                )
            )
        else:
            confirmed = self.confirmed_research_inquiries
            research_questions = [str(item["text"]) for item in confirmed]
            research_inquiry_types = {
                str(item["text"]): str(item.get("type") or "research_question")
                for item in confirmed
            }
            research_inquiry_sources = self._locate_research_inquiry_sources(
                chunks_data, research_questions
            )
            research_inquiry_aliases = {}
            research_inquiry_ids = {str(item["text"]): str(item["id"]) for item in confirmed}
            research_inquiry_parent_ids = {
                str(item["text"]): str(item["parent_id"])
                for item in confirmed
                if item.get("parent_id")
            }

        for question, parent_id in research_inquiry_parent_ids.items():
            if parent_id and research_inquiry_types.get(question) == "research_question":
                research_inquiry_types[question] = "sub_question"
        rq_alignment_score, unaddressed_rqs = self._compute_rq_alignment(
            research_questions, chunks_data, research_inquiry_aliases
        )

        if unaddressed_rqs:
            red_flags.append(
                RedFlag(
                    severity="warning",
                    category="structure",
                    title="Unaddressed Research Questions",
                    description=f"{len(unaddressed_rqs)} research questions may not be fully addressed in findings/conclusion.",
                    suggestion="Ensure each research question is explicitly answered in results and discussion sections.",
                )
            )

        # Concept coverage is meaningful only across the assessed thesis chapters,
        # not front matter, title pages, references, or appendices.
        key_concepts, orphaned_concepts = self._track_concept_progression(
            chunks_data, main_chapters
        )

        if orphaned_concepts:
            red_flags.append(
                RedFlag(
                    severity="info",
                    category="structure",
                    title="Orphaned Concepts",
                    description=f"{len(orphaned_concepts)} key concepts appear in only one section (not developed across thesis).",
                    suggestion="Consider developing key concepts across multiple sections for stronger argumentation.",
                )
            )

        return StructureAnalysis(
            chapter_count=len(main_chapters),  # Only count main-matter chapters
            chapter_flow_scores=flow_scores,
            chapter_transition_labels=chapter_transition_labels,
            abrupt_transitions=abrupt_transitions,
            missing_sections=missing_sections,
            avg_coherence=float(avg_coherence),
            chapter_order=[chapter["name"] for chapter in all_chapters],
            chapter_order_issues=chapter_order_issues,
            research_questions=research_questions,
            rq_alignment_score=rq_alignment_score,
            unaddressed_rqs=unaddressed_rqs,
            key_concepts=key_concepts,
            orphaned_concepts=orphaned_concepts,
            red_flags=red_flags,
            research_inquiry_types=research_inquiry_types,
            research_inquiry_sources=research_inquiry_sources,
            research_inquiry_aliases=research_inquiry_aliases,
            research_inquiry_ids=research_inquiry_ids,
            research_inquiry_parent_ids=research_inquiry_parent_ids,
        )

    def analyse_citation_patterns(self, chunks_data: Dict) -> CitationPatternAnalysis:
        """
        Analyse citation patterns in thesis.

        Checks:
        - Citation recency (presence of recent sources)
        - Citation diversity (range of venues/topics)
        - Geographic diversity

        Args:
            chunks_data: ChromaDB query result (used to get doc_id)

        Returns:
            CitationPatternAnalysis with findings
        """
        # An unavailable citation store cannot be treated as evidence that the
        # thesis has no citations.
        citation_evidence_available = Path(self.citation_db_path).is_file()

        # Extract citation metadata from SQLite database
        citations = self._extract_citations(chunks_data)

        # Count unique citations by DOI or title
        unique_citations = len(
            set(c.get("doi") if c.get("doi") else c.get("title", "") for c in citations)
        )

        # Check recency (% of sources from last 5 years)
        recent_count = sum(1 for c in citations if self._is_recent(c.get("year")))
        recency_score = (recent_count / len(citations)) if citations else 0.0

        # Detect citation clusters by venue/topic
        citation_clusters = self._cluster_citations(citations)

        # Compute geographic diversity (placeholder)
        geographic_diversity = self._compute_geographic_diversity(citations)

        # Generate red flags
        red_flags = []

        # Citation evidence unavailable
        if not citation_evidence_available:
            red_flags.append(
                RedFlag(
                    severity="warning",
                    category="citations",
                    title="Citation Evidence Unavailable",
                    description=(
                        "Citation evidence could not be loaded, so citation quality is excluded "
                        "from the assessment signal."
                    ),
                    suggestion="Verify citation extraction and the citation graph database before review.",
                )
            )

        # No citations found in an available citation store
        elif len(citations) == 0:
            red_flags.append(
                RedFlag(
                    severity="warning",
                    category="citations",
                    title="No Citation Evidence Found",
                    description="The available citation graph contains no citations for this document.",
                    suggestion="A human reviewer should verify citations in the thesis and ingestion data.",
                )
            )

        # Low recency
        elif recency_score < 0.2 and len(citations) > 10:
            red_flags.append(
                RedFlag(
                    severity="warning",
                    category="citations",
                    title="Stale References",
                    description=f"Only {recency_score*100:.0f}% of citations are from the last 5 years.",
                    suggestion="Include more recent publications to demonstrate current knowledge.",
                )
            )

        # Citation clustering (echo chamber)
        if citation_clusters and len(citations) > 0:
            max_cluster_ratio = max(citation_clusters.values()) / sum(citation_clusters.values())
            if max_cluster_ratio > 0.5:
                red_flags.append(
                    RedFlag(
                        severity="info",
                        category="citations",
                        title="Citation Concentration",
                        description=f"Over 50% of citations are from a single venue/topic area.",
                        suggestion="Diversify citation sources to demonstrate broad literature engagement.",
                    )
                )

        return CitationPatternAnalysis(
            citation_evidence_available=citation_evidence_available,
            total_citations=len(citations),
            unique_citations=unique_citations,
            citation_recency_score=recency_score,
            orphaned_claims=[],  # Moved to claim analysis
            citation_clusters=citation_clusters,
            geographic_diversity=geographic_diversity,
            red_flags=red_flags,
        )

    def analyse_claims_and_contradictions(self, chunks_data: Dict) -> ClaimAnalysis:
        """
        Extract claims and detect potential contradictions.

        Uses heuristic extraction by default. If llm_client is provided, uses it
        for higher-quality claim extraction and contradiction detection.

        Args:
            chunks_data: ChromaDB query result with chunks, metadata, embeddings

        Returns:
            ClaimAnalysis with extracted claims and detected contradictions
        """
        text = self._collect_text_by_section(
            chunks_data,
            include_sections=["abstract", "introduction", "results", "discussion", "conclusion"],
        )

        llm_used = False
        claims = self._extract_claims_from_text(text)
        contradictions = self._detect_contradictions(claims)

        # Detect orphaned claims (moved from citation patterns)
        orphaned_claims = self._detect_orphaned_claims(chunks_data)

        if self.llm_client and self._llm_enabled("claims"):
            llm_claims = self._extract_claims_from_text_llm(text)
            if llm_claims:
                llm_used = True
                claims = llm_claims
                contradictions = self._detect_contradictions_llm(claims)

        if llm_used:
            claims = [f"[LLM] {c}" for c in claims]
        else:
            claims = [f"[Heuristic] {c}" for c in claims]

        red_flags = []
        if contradictions:
            red_flags.append(
                RedFlag(
                    severity="warning",
                    category="consistency",
                    title="Potential Contradictions Detected",
                    description=f"Detected {len(contradictions)} potential contradictions across claims.",
                    suggestion="Review highlighted claims for consistency or clarify scope/conditions.",
                )
            )

        # Add red flag for orphaned claims
        if orphaned_claims:
            red_flags.append(
                RedFlag(
                    severity="warning",
                    category="claims",
                    title="Unsupported Claims",
                    description=f"{len(orphaned_claims)} claims lack visible citations.",
                    location=orphaned_claims[0][:100] + "..." if orphaned_claims else "",
                    suggestion="Ensure all strong claims are supported by citations to authoritative sources.",
                )
            )

        return ClaimAnalysis(
            total_claims=len(claims),
            claims=claims,
            contradictions=contradictions,
            orphaned_claims=orphaned_claims,
            red_flags=red_flags,
        )

    def validate_methodology_checklist(self, chunks_data: Dict) -> MethodologyChecklist:
        """Validate methodology checklist based on section content.

        Args:
            chunks_data: ChromaDB query result with chunks, metadata, embeddings

        Returns:
            MethodologyChecklist with checklist results and evidence
        """
        import re

        include_sections = [
            "method",
            "methodology",
            "methods",
            "research design",
            "sampling",
            "analysis",
        ]
        metadatas = chunks_data.get("metadatas", [])
        documents = chunks_data.get("documents", [])

        scope_indices = []
        scope_section_labels = []
        for i, meta in enumerate(metadatas):
            # CRITICAL: Skip ToC chunks to prevent extracting ToC entries as methodology evidence
            if i < len(documents) and self._is_toc_chunk(documents[i]):
                continue

            section = (meta.get("section_title") or meta.get("chapter") or "").lower()
            if any(key in section for key in include_sections):
                scope_indices.append(i)
                scope_section_labels.append(section or "unknown")

        if not scope_indices:
            scope_indices = [i for i, doc in enumerate(documents) if not self._is_toc_chunk(doc)]
            scope_note = (
                f"Scanned full document ({len(scope_indices)} chunks) for methodology evidence "
                "(Table of Contents excluded)."
            )
        else:
            unique_sections = sorted(set(s for s in scope_section_labels if s))
            section_preview = ", ".join(unique_sections[:5])
            if len(unique_sections) > 5:
                section_preview += ", ..."
            scope_note = f"Evidence scoped to {len(scope_indices)} chunks from sections like: {section_preview}."

        text = "\n".join(
            documents[i]
            for i in scope_indices
            if i < len(documents) and not self._is_toc_chunk(documents[i])
        )

        def _classify_section_tags(label: str) -> List[str]:
            label_lower = label.lower()
            tags = []
            if "appendix" in label_lower:
                tags.append("appendix")
            if any(k in label_lower for k in ["method", "methodology", "methods"]):
                tags.append("methods")
            if "design" in label_lower:
                tags.append("design")
            if "sampling" in label_lower:
                tags.append("sampling")
            if "analysis" in label_lower:
                tags.append("analysis")
            if "results" in label_lower or "findings" in label_lower:
                tags.append("results")
            if "discussion" in label_lower:
                tags.append("discussion")
            if "introduction" in label_lower:
                tags.append("introduction")
            if "literature" in label_lower or "review" in label_lower:
                tags.append("literature")
            if "conclusion" in label_lower:
                tags.append("conclusion")
            if "ethic" in label_lower:
                tags.append("ethics")
            return tags or ["other"]

        keyword_map = {
            "research_question": [
                "research question",
                "research questions",
                "hypothesis",
                "aim",
                "objective",
            ],
            "data_collection": [
                "data collection",
                "research design",
                "survey",
                "interview",
                "dataset",
            ],
            "sample_size": ["sample size", "sample", "n=", "participants", "respondents"],
            "sampling_method": [
                "sampling",
                "sampling frame",
                "sampling technique",
                "sampling methodology",
                "random",
                "stratified",
                "purposive",
            ],
            "analysis_method": [
                "analysis",
                "analysis method",
                "quantitative analysis",
                "qualitative analysis",
                "thematic",
                "coding",
                "regression",
                "findings",
            ],
            "validity_reliability": ["validity", "reliability", "robustness"],
            "ethics": [
                "ethics",
                "ethical",
                "ethically",
                "consent",
                "irb",
                "approval",
                "ethics committee",
                "human research ethics",
            ],
            "limitations": ["limitation", "limitations", "constraint", "threats to validity"],
        }

        def _make_snippet(doc: str, start: int, end: int) -> str:
            snippet_start = max(0, start)
            snippet_end = min(len(doc), end)
            while snippet_start > 0 and doc[snippet_start - 1].isalnum():
                snippet_start -= 1
            while snippet_end < len(doc) and doc[snippet_end : snippet_end + 1].isalnum():
                snippet_end += 1
            snippet = doc[snippet_start:snippet_end].replace("\n", " ").strip()
            return " ".join(snippet.split())

        checklist = {}
        evidence = {}
        scope_word_count = sum(
            len(documents[i].split()) for i in scope_indices if i < len(documents)
        )

        section_labels_by_index: Dict[int, str] = {}
        unknown_counter = 0
        for i in scope_indices:
            if i >= len(documents):
                continue
            doc = documents[i]
            if self._is_toc_chunk(doc):
                continue
            meta = metadatas[i] if i < len(metadatas) else {}
            section_label = self._get_section_label(meta, doc)
            if section_label in {"Unknown", "Unclassified"}:
                section_label = self._get_chapter_label(meta, doc)
            if section_label in {"Unknown", "Unclassified"}:
                unknown_counter += 1
                section_label = f"Section {unknown_counter}"
            section_labels_by_index[i] = section_label

        for item, keywords in keyword_map.items():
            pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
            count = 0
            snippets: List[Dict[str, Any]] = []
            location_counts: Dict[str, int] = {}
            doc_snippet_added: set[int] = set()

            for i in scope_indices:
                if i >= len(documents):
                    continue
                doc = documents[i]
                if self._is_toc_chunk(doc):
                    continue
                matched_section_label = section_labels_by_index.get(i)
                if not matched_section_label:
                    continue
                location_counts[matched_section_label] = (
                    location_counts.get(matched_section_label, 0) + 0
                )

                for match in pattern.finditer(doc):
                    count += 1
                    location_counts[matched_section_label] = (
                        location_counts.get(matched_section_label, 0) + 1
                    )
                    if len(snippets) < 3 and i not in doc_snippet_added:
                        start = match.start() - 140
                        end = match.end() + 140
                        snippet = _make_snippet(doc, start, end)
                        snippets.append(
                            {
                                "location": section_label,
                                "snippet": snippet,
                                "tags": _classify_section_tags(section_label),
                            }
                        )
                        doc_snippet_added.add(i)

            checklist[item] = count > 0
            strength_per_1k = (count / max(1, scope_word_count)) * 1000.0
            top_location = ""
            top_hits = 0
            if location_counts:
                top_location, top_hits = max(location_counts.items(), key=lambda kv: kv[1])
            if count > 0:
                summary = (
                    f"Evidence appears in {top_location} (top section, {top_hits} hits); "
                    f"density {strength_per_1k:.2f}/1k words."
                )
            else:
                summary = "No evidence found in the scoped text."
            evidence[item] = {
                "count": count,
                "keywords": keywords,
                "snippets": snippets,
                "strength_per_1k": strength_per_1k,
                "summary": summary,
            }

        missing_items = [name for name, present in checklist.items() if not present]
        score = 1.0 - (len(missing_items) / max(1, len(checklist)))

        red_flags = []
        critical_missing = {"data_collection", "analysis_method", "sample_size"}
        if any(item in missing_items for item in critical_missing):
            red_flags.append(
                RedFlag(
                    severity="critical",
                    category="methodology",
                    title="Missing Core Methodology Elements",
                    description="Core methodology elements appear to be missing or unclear.",
                    suggestion="Ensure data collection, sample size, and analysis method are clearly described.",
                )
            )
        elif len(missing_items) >= 3:
            red_flags.append(
                RedFlag(
                    severity="warning",
                    category="methodology",
                    title="Methodology Checklist Gaps",
                    description=f"{len(missing_items)} methodology elements are not detected.",
                    suggestion="Clarify methodology details to improve reproducibility.",
                )
            )

        confidence_note = scope_note

        return MethodologyChecklist(
            items=checklist,
            missing_items=missing_items,
            score=score,
            confidence_note=confidence_note,
            evidence=evidence,
            red_flags=red_flags,
        )

    def analyse_writing_quality(self, chunks_data: Dict) -> WritingQualityMetrics:
        """Analyse writing quality metrics for clarity and readability.

        Args:
            chunks_data: ChromaDB query result with chunks, metadata, embeddings

        Returns:
            WritingQualityMetrics with findings
        """
        text = self._collect_text_by_section(chunks_data, include_sections=None)
        sentences = self._split_sentences(text)
        words = self._tokenise_words(text)

        avg_sentence_length = len(words) / max(1, len(sentences))
        avg_word_length = sum(len(w) for w in words) / max(1, len(words))

        readability = self._flesch_reading_ease(text)
        education_level = self._flesch_education_level(readability)
        passive_ratio = self._passive_voice_ratio(sentences)
        jargon_density = self._jargon_density(words)

        red_flags = []
        if readability < 15:
            red_flags.append(
                RedFlag(
                    severity="warning",
                    category="writing",
                    title="Low Readability",
                    description=f"Flesch Reading Ease score is {readability:.1f} (very complex).",
                    suggestion="Shorten sentences and reduce complex phrasing where possible.",
                )
            )
        if passive_ratio > 0.35:
            red_flags.append(
                RedFlag(
                    severity="info",
                    category="writing",
                    title="High Passive Voice Usage",
                    description=f"Passive voice detected in {passive_ratio*100:.0f}% of sentences.",
                    suggestion="Use active voice to improve clarity and directness.",
                )
            )
        if jargon_density > 0.12:
            red_flags.append(
                RedFlag(
                    severity="info",
                    category="writing",
                    title="High Jargon Density",
                    description=f"Estimated jargon density is {jargon_density*100:.0f}%.",
                    suggestion="Define specialised terms and reduce excessive jargon where possible.",
                )
            )

        return WritingQualityMetrics(
            readability_score=readability,
            education_level=education_level,
            avg_sentence_length=avg_sentence_length,
            avg_word_length=avg_word_length,
            passive_voice_ratio=passive_ratio,
            jargon_density=jargon_density,
            red_flags=red_flags,
        )

    def analyse_contribution_alignment(self, chunks_data: Dict) -> ContributionAlignment:
        """Check alignment between stated contributions and reported findings.

        Args:
            chunks_data: ChromaDB query result with chunks, metadata, embeddings

        Returns:
            ContributionAlignment with findings
        """
        contributions_text = self._collect_text_by_section(
            chunks_data,
            include_sections=["introduction", "contribution", "contributions", "conclusion"],
        )
        findings_text = self._collect_text_by_section(
            chunks_data,
            include_sections=["results", "discussion", "findings"],
        )

        contribution_keywords = self._extract_keywords(contributions_text, limit=20)
        finding_keywords = self._extract_keywords(findings_text, limit=20)

        overlap = set(contribution_keywords) & set(finding_keywords)
        union = set(contribution_keywords) | set(finding_keywords)
        overlap_score = len(overlap) / max(1, len(union))

        unmatched_contributions = [k for k in contribution_keywords if k not in overlap]
        unmatched_findings = [k for k in finding_keywords if k not in overlap]

        red_flags = []
        if overlap_score < 0.2 and contribution_keywords and finding_keywords:
            red_flags.append(
                RedFlag(
                    severity="warning",
                    category="contribution",
                    title="Weak Contribution-Finding Alignment",
                    description=f"Low overlap between contribution and finding keywords (score {overlap_score:.2f}).",
                    suggestion="Ensure stated contributions are directly supported by results and discussion.",
                )
            )

        return ContributionAlignment(
            contribution_keywords=contribution_keywords,
            finding_keywords=finding_keywords,
            overlap_score=overlap_score,
            unmatched_contributions=unmatched_contributions,
            unmatched_findings=unmatched_findings,
            red_flags=red_flags,
        )

    def analyse_data_conclusion_mismatch(self, chunks_data: Dict) -> DataConclusionMismatch:
        """Detect potential mismatch between results and conclusions.

        Args:
            chunks_data: ChromaDB query result with chunks, metadata, embeddings

        Returns:
            DataConclusionMismatch with findings

        """
        results_text = self._collect_text_by_section(
            chunks_data,
            include_sections=["results", "findings"],
        )
        conclusion_text = self._collect_text_by_section(
            chunks_data,
            include_sections=["conclusion", "discussion"],
        )

        issues = []
        red_flags = []

        strong_claims = self._extract_strong_claims(conclusion_text)
        has_quant_evidence = self._contains_any(
            results_text, ["%", "p <", "p=", "table", "figure", "r=", "+/-", "mean"]
        )

        if strong_claims and not has_quant_evidence:
            issues.append(
                {
                    "claim": strong_claims[0],
                    "reason": "Strong conclusion claim without clear quantitative evidence in results section.",
                    "source": "heuristic",
                }
            )

        if self.llm_client and self._llm_enabled("data_mismatch"):
            llm_issues = self._detect_data_conclusion_mismatch_llm(results_text, conclusion_text)
            if llm_issues:
                issues = [{**issue, "source": "llm"} for issue in llm_issues]

        if issues:
            red_flags.append(
                RedFlag(
                    severity="warning",
                    category="consistency",
                    title="Data-Conclusion Mismatch",
                    description=f"Detected {len(issues)} potential mismatches between results and conclusions.",
                    suggestion="Verify that conclusions are supported by results and clarify evidence.",
                )
            )

        return DataConclusionMismatch(issues=issues, red_flags=red_flags)

    def analyse_citation_misrepresentation(self, chunks_data: Dict) -> CitationMisrepresentation:
        """Detect potential citation misrepresentation in conclusions/discussion.

        Args:
            chunks_data: ChromaDB query result with chunks, metadata, embeddings

        Returns:
            CitationMisrepresentation with findings
        """
        conclusion_text = self._collect_text_by_section(
            chunks_data,
            include_sections=["discussion", "conclusion"],
        )
        references = self._extract_citations(chunks_data)
        reference_titles = [
            r.get("title", "") for r in references if isinstance(r, dict) and r.get("title")
        ]

        issues = []
        red_flags = []

        claim_sentences = self._extract_strong_claims(conclusion_text)
        if claim_sentences and not references:
            issues.append(
                {
                    "claim": claim_sentences[0],
                    "reason": "Strong claims present but no references detected in conclusions/discussion.",
                    "source": "heuristic",
                }
            )

        if self.llm_client and reference_titles and self._llm_enabled("citation_misrep"):
            llm_issues = self._detect_citation_misrepresentation_llm(
                claim_sentences, reference_titles
            )
            if llm_issues:
                issues = [{**issue, "source": "llm"} for issue in llm_issues]

        if issues:
            red_flags.append(
                RedFlag(
                    severity="warning",
                    category="citations",
                    title="Potential Citation Misrepresentation",
                    description=f"Detected {len(issues)} claims that may not be supported by cited sources.",
                    suggestion="Verify that each strong claim is supported by the cited references.",
                )
            )

        return CitationMisrepresentation(issues=issues, red_flags=red_flags)

    def analyse_benchmarking(self, chunks_data: Dict) -> BenchmarkingResult:
        """Stub comparative benchmarking until corpus is configured.

        TODO: Implement actual benchmarking against a corpus of PhD theses with known outcomes.

        Args:
            chunks_data: ChromaDB query result with chunks, metadata, embeddings

        Returns:
            BenchmarkingResult with placeholder status and notes

        """
        return BenchmarkingResult(
            status="not_configured",
            notes="Benchmarking corpus not configured yet.",
            metrics={},
        )

    def analyse_argument_flow_graph(self, chunks_data: Dict) -> ArgumentFlowGraph:
        """Build a simple argument flow graph between sections.

        Args:
            chunks_data: ChromaDB query result with chunks, metadata, embeddings

        Returns:
            ArgumentFlowGraph with nodes and edges representing section flow
        """
        sections = self._extract_section_embeddings(chunks_data)
        nodes = []
        edges = []

        for idx, section in enumerate(sections):
            nodes.append(
                {
                    "id": section["name"],
                    "label": section["name"],
                    "index": idx,
                }
            )

        for idx in range(len(sections) - 1):
            weight = self._compute_chapter_similarity(
                sections[idx]["embedding_mean"],
                sections[idx + 1]["embedding_mean"],
            )
            edges.append(
                {
                    "source": sections[idx]["name"],
                    "target": sections[idx + 1]["name"],
                    "weight": weight,
                }
            )

        return ArgumentFlowGraph(nodes=nodes, edges=edges)

    def detect_red_flags(self, chunks_data: Dict) -> List[RedFlag]:
        """
        Detect specific red flags in thesis.

        Phase 1 red flags:
        - Missing limitations section
        - Scope creep in conclusion
        - Methodology changes between chapters

        Args:
            chunks_data: ChromaDB query result

        Returns:
            List of RedFlag objects
        """
        red_flags = []

        # Check for limitations section
        has_limitations = any(
            "limitation" in meta.get("section_title", "").lower()
            for meta in chunks_data["metadatas"]
        )

        if not has_limitations:
            red_flags.append(
                RedFlag(
                    severity="critical",
                    category="structure",
                    title="Missing Limitations Section",
                    description="No dedicated limitations section found in thesis.",
                    suggestion="Add a section discussing study limitations and future work.",
                )
            )

        # Detect scope creep (conclusion much larger than results)
        chapter_sizes = self._get_chapter_sizes(chunks_data)
        conclusion_size = chapter_sizes.get("conclusion", 0)
        results_size = chapter_sizes.get("results", 0)

        if results_size > 0 and conclusion_size / results_size > 1.5:
            red_flags.append(
                RedFlag(
                    severity="warning",
                    category="scope",
                    title="Potential Scope Creep",
                    description="Conclusion section is significantly larger than results, suggesting scope expansion.",
                    location="Conclusion",
                    suggestion="Ensure conclusion focuses on summarising findings without introducing new material.",
                )
            )

        return red_flags

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _has_section_metadata(self, meta: Dict) -> bool:
        """Check whether metadata contains usable section labels."""
        return any(
            (meta.get(key) or "").strip() and (meta.get(key) or "") != "Unknown"
            for key in ("section_title", "parent_section", "heading_path", "chapter")
        )

    def _is_toc_chunk(self, text: str) -> bool:
        """Check if chunk text contains Table of Contents patterns.

        ToC entries have characteristic patterns:
        - Multiple dots (leader dots): ....... or ......
        - Followed by page numbers
        - Text like "Chapter 1: Introduction ........................... 5"

        Args:
            text: Chunk text to check

        Returns:
            True if chunk appears to contain ToC entries
        """
        import re

        # Check first 300 chars for ToC patterns (ToC lines are usually near start)
        sample = text[:300]

        # ToC pattern: 3+ dots followed by page numbers
        if re.search(r"\.{3,}\s*\d+\s*$", sample, re.MULTILINE):
            return True
        if re.search(r"\.{3,}\.?\s*(?:page\s*)?\d+", sample):
            return True

        # Alternative pattern: "Chapter N .... page N" or similar
        if re.search(r"chapter\s+\d+.*?\.{3,}", sample, re.IGNORECASE):
            return True

        # List of Tables/Figures entries often mimic ToC leader-dot formatting
        if re.search(
            r"^(table|figure)\s+\d+(?:\.\d+)*.*?\.{3,}\s*\d+", sample, re.IGNORECASE | re.MULTILINE
        ):
            return True

        # Some list-of-tables entries appear later in a chunk, so scan a larger window
        extended = text[:2000]
        if re.search(
            r"^(table|figure)\s+\d+(?:\.\d+)*.*?\.{3,}\s*\d+",
            extended,
            re.IGNORECASE | re.MULTILINE,
        ):
            return True

        return False

    def _select_structural_indices(self, chunks_data: Dict) -> List[int]:
        """Select chunk indices that best represent section structure.

        Prefers parent chunks (if available) to avoid over-weighting repeated
        child chunk content in structure metrics. Also filters out Table of
        Contents chunks to prevent incorrect chapter/section mapping.
        """
        metadatas = chunks_data.get("metadatas", [])
        documents = chunks_data.get("documents", [])
        if not metadatas:
            return []

        # Filter out ToC chunks first
        non_toc_indices = [
            idx
            for idx in range(len(metadatas))
            if idx < len(documents) and not self._is_toc_chunk(documents[idx])
        ]

        # Then prefer parent chunks from non-ToC indices
        parent_indices = [
            idx for idx in non_toc_indices if metadatas[idx].get("chunk_type") == "parent"
        ]

        if parent_indices:
            if any(self._has_section_metadata(metadatas[idx]) for idx in parent_indices):
                return parent_indices
            # Parent and child chunks have independent sequence ranges. When
            # parent structure metadata is absent, mixing the two levels
            # corrupts source order, so derive structure from child chunks.
            child_indices = [
                idx for idx in non_toc_indices if metadatas[idx].get("chunk_type") == "child"
            ]
            if child_indices:
                return child_indices
            return parent_indices

        return non_toc_indices

    def _source_ordered_structural_indices(self, chunks_data: Dict[str, Any]) -> List[int]:
        """Return structural chunk indices ordered by source sequence metadata.

        Chunks without numeric sequence metadata retain their original relative
        order after all source-sequenced chunks. This is used by all section and
        chapter views that need a consistent source-order contract.
        """
        metadatas = chunks_data.get("metadatas", [])

        def _order_key(index: int) -> Tuple[int, float, int]:
            sequence = metadatas[index].get("sequence_number")
            if isinstance(sequence, (int, float)):
                return (0, float(sequence), index)
            return (1, float("inf"), index)

        return sorted(self._select_structural_indices(chunks_data), key=_order_key)

    def _get_chapter_label(self, meta: Dict, fallback_text: str) -> str:
        """Derive a chapter label using metadata, then fallback text.

        Properly handles chapter numbering (1-based) and distinguishes between
        pre-matter (Abstract, Acknowledgements), main chapters (1-9), and
        post-matter (References, Appendix) sections.
        """
        # Priority 1: Use chapter metadata if present and well-formed
        chapter = (meta.get("chapter") or "").strip()
        if chapter and chapter != "Unknown":
            # Ensure chapters are properly formatted (e.g., "Chapter 1", not "Chapter 0")
            if chapter.lower().startswith("chapter"):
                return chapter
            # Allow pre-matter and post-matter sections through
            if any(
                kw in chapter.lower()
                for kw in [
                    "abstract",
                    "acknowledgement",
                    "introduction",
                    "methodology",
                    "results",
                    "discussion",
                    "conclusion",
                    "references",
                    "appendix",
                ]
            ):
                return chapter

        # Priority 2: Use parent_section (from hierarchy, usually most reliable)
        parent_section = (meta.get("parent_section") or "").strip()
        if parent_section and parent_section != "Unknown" and len(parent_section) < 150:
            # Avoid using text snippets - expect parent_section to be structured labels
            if self._is_valid_chapter_label(parent_section):
                return parent_section

        # Priority 3: Use heading path (first level is usually chapter)
        heading_path = (meta.get("heading_path") or "").strip()
        if heading_path and heading_path != "Unknown":
            first_level = heading_path.split(" > ")[0].strip()
            if first_level and len(first_level) < 150:
                return first_level

        # Priority 4: Use section title
        section_title = (meta.get("section_title") or "").strip()
        if section_title and section_title != "Unknown":
            # Only use if it looks like a chapter reference
            if "chapter" in section_title.lower() or any(
                kw in section_title.lower()
                for kw in [
                    "introduction",
                    "methodology",
                    "results",
                    "discussion",
                    "conclusion",
                    "references",
                    "appendix",
                    "abstract",
                    "acknowledgement",
                ]
            ):
                return section_title

        # Fallback to text parsing if metadata is missing or insufficient
        import re

        # Try multiple patterns to extract chapter number
        patterns = [
            r"\bchapter\s+(\d+)[:\s—\-]",  # "Chapter 2:" or "Chapter 2 "
            r"^\s*chapter\s+(\d+)",  # Chapter at start of text
            r"\n\s*chapter\s+(\d+)",  # Chapter after newline
        ]

        for pattern in patterns:
            match = re.search(pattern, fallback_text, re.IGNORECASE | re.MULTILINE)
            if match:
                chapter_num = match.group(1)
                # Ensure it's 1-based (reject 0)
                if chapter_num != "0":
                    return f"Chapter {chapter_num}"

        # Try to identify generic sections (but keep them as is, don't rename)
        for section_name in [
            "Abstract",
            "Acknowledgements",
            "Introduction",
            "Methodology",
            "Results",
            "Discussion",
            "Conclusion",
            "References",
            "Appendix",
        ]:
            if re.search(rf"\b{section_name}\b", fallback_text, re.IGNORECASE):
                return section_name

        # Final fallback: only use clear heading patterns (ALL CAPS or very short multi-word titles)
        lines = fallback_text.split("\n")
        for line in lines[:5]:  # Check first 5 lines only
            line_stripped = line.strip()
            # Strict criteria: all caps (like "INTRODUCTION") or very short title-case heading
            if 5 <= len(line_stripped) <= 80:
                # Check if it's ALL CAPS (clear heading)
                if line_stripped.isupper() and not line_stripped.endswith("."):
                    return line_stripped
                # Or check if it's title case without common sentence patterns
                word_count = len(line_stripped.split())
                if 1 <= word_count <= 5:  # Short phrase (likely heading)
                    upper_ratio = sum(1 for c in line_stripped if c.isupper()) / len(line_stripped)
                    if upper_ratio > 0.3 and not line_stripped.endswith("."):
                        # Additional check: doesn't contain common prepositions/verbs
                        if not any(
                            word.lower() in line_stripped.lower()
                            for word in [
                                "throughout",
                                "research",
                                "study",
                                "to",
                                "from",
                                "with",
                                "and",
                            ]
                        ):
                            return line_stripped[:60]

        return "Unknown"

    def _get_section_label(self, meta: Dict, fallback_text: Optional[str] = None) -> str:
        """Derive a section label using section metadata.

        Returns the most specific section identifier available from metadata,
        with intelligent fallbacks to prevent "Unknown" labels.
        """
        # Priority 1: section_title (most specific)
        section_title = (meta.get("section_title") or "").strip()
        if section_title and section_title != "Unknown":
            return section_title

        # Priority 2: heading_path (may contain hierarchy)
        heading_path = (meta.get("heading_path") or "").strip()
        if heading_path and heading_path != "Unknown":
            # Get the most specific part (last component)
            last_component = heading_path.split(" > ")[-1].strip()
            if last_component:
                return last_component

        # Priority 3: parent_section
        parent_section = (meta.get("parent_section") or "").strip()
        if parent_section and parent_section != "Unknown":
            return parent_section

        # Priority 4: chapter (use as fallback if no section available)
        chapter = (meta.get("chapter") or "").strip()
        if chapter and chapter != "Unknown":
            return chapter

        # Priority 5: Text parsing if metadata insufficient
        if fallback_text:
            # Try to find a section heading in the text
            import re

            markdown_heading = re.search(r"^#{1,6}\s+(.+?)\s*#*\s*$", fallback_text, re.MULTILINE)
            if markdown_heading:
                heading = markdown_heading.group(1).strip()
                if heading:
                    return heading[:100]

            for section_name in [
                "Introduction",
                "Acknowledgements",
                "Methodology",
                "Results",
                "Findings",
                "Analysis",
                "Discussion",
                "Conclusion",
                "References",
                "Appendix",
                "Abstract",
                "Research Significance",
                "Original Contribution",
                "Conceptual Integration",
                "Methodological Rigour",
                "Findings Quality",
                "Contribution Contextualisation",
                "Literature Review",
            ]:
                if re.search(rf"\b{section_name}\b", fallback_text, re.IGNORECASE):
                    return section_name

        return "Unclassified"  # Better than "Unknown"

        return "Unknown"

    def _is_valid_chapter_label(self, label: str) -> bool:
        """Check if a label looks like a valid chapter or section label.

        Rejects obvious non-chapters like table headers, statistical notation,
        or content snippets.

        TODO: Heuristic filter is hit and miss - consider ways to improve such as using a small LLM classifier if needed or a more capable PDF parser that can identify structural elements more reliably.
        """
        if not label or len(label) > 200:
            return False

        # Reject obvious table headers and statistical notation
        bad_patterns = [
            r"^\s*[A-Z]\s+",  # Single letter followed by space (like "N Mean SD")
            r"\bSD\b.*\bSE\b",  # Statistical abbreviations together
            r"Mean.*SD.*SE",  # Statistical sequence
            r"\d+%.*\d+%",  # Multiple percentages
            r"^\s*%.*%\s*$",  # Only percentages
            r"^\s*\(n=\d+\)",  # Just sample size
            r"\*\*\*|\*\*",  # Significance markers
        ]

        for pattern in bad_patterns:
            if re.search(pattern, label, re.IGNORECASE):
                return False

        # Reject if it's mostly abbreviations/acronyms with no vowels
        word_chars = len([c for c in label if c.isalpha()])
        if word_chars > 0:
            vowels = len([c for c in label.lower() if c in "aeiou"])
            vowel_ratio = vowels / word_chars
            if vowel_ratio < 0.15:  # Too few vowels = likely all acronyms
                return False

        # Reject if too short (less than 2 chars) or obviously content
        if len(label) < 2:
            return False

        return True

    def _parse_toc_structure(self, chunks_data: Dict) -> Dict[str, int]:
        """Parse Table of Contents to extract chapter structure.

        Args:
            chunks_data: ChromaDB query result with documents and metadatas

        Returns:
            Dict mapping chapter names to their ToC order/page numbers
        """
        toc_structure = {}
        documents = chunks_data.get("documents", [])

        for doc in documents:
            if not self._is_toc_chunk(doc):
                continue

            lines = [line.strip() for line in doc.splitlines()]
            combined_lines: List[str] = []
            line_idx = 0
            while line_idx < len(lines):
                line = lines[line_idx]
                if not line:
                    line_idx += 1
                    continue
                next_line = lines[line_idx + 1] if line_idx + 1 < len(lines) else ""
                if (
                    re.search(r"^(chapter\s+\d+|appendix\s+[a-z])", line, re.IGNORECASE)
                    and not re.search(r"\d+\s*$", line)
                    and re.search(r"\.{3,}\s*\d+\s*$", next_line)
                ):
                    combined_lines.append(f"{line} {next_line}")
                    line_idx += 2
                    continue
                combined_lines.append(line)
                line_idx += 1

            doc_to_parse = "\n".join(combined_lines)

            # Extract chapter entries from ToC
            # Pattern: "Chapter N: Title ......... PageNum" or similar
            chapter_patterns = [
                r"(chapter\s+\d+(?:\.\s*)?[^.\n]*?)[\.\s]{3,}(\d+)",  # Chapter N. Title ... PageNum
                r"(chapter\s+\d+(?:\.\s*)?[^\n]{3,}?)\s+(\d{1,4})$",  # Chapter N. Title 123
                r"^(?:[ivxlcdm]+\.)\s*([A-Z][^.\n]{3,80})[\.\s]{3,}(\d+)",  # i. Front matter ... PageNum
                r"^([A-Z][^.\n]{5,80})[\.\s]{3,}(\d+)",  # Title case heading ... PageNum
                r"^(appendix\s+[a-z][^\n]{0,80})[\.\s]{3,}(\d+)",  # Appendix A: ... PageNum
                r"^(appendix\s+[a-z][^\n]{0,80})\s+(\d{1,4})$",  # Appendix A: ... 123
            ]

            for pattern in chapter_patterns:
                matches = re.finditer(pattern, doc_to_parse, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    chapter_name = match.group(1).strip()
                    page_num = int(match.group(2))

                    # Clean up chapter name
                    chapter_name = re.sub(r"\s+", " ", chapter_name).strip()
                    chapter_name = re.sub(r"\s*[:\.]\s*$", "", chapter_name)

                    if self._is_valid_chapter_label(chapter_name):
                        toc_structure[chapter_name] = page_num

        return toc_structure

    def _build_toc_order(self, chunks_data: Dict) -> Dict[str, int]:
        """Build a ToC order map from chapter labels to ordered index."""
        toc_structure = self._parse_toc_structure(chunks_data)
        return {
            chapter_name: idx
            for idx, (chapter_name, _) in enumerate(
                sorted(toc_structure.items(), key=lambda item: item[1])
            )
        }

    def _match_toc_label(self, label: str, toc_map: Dict[str, int]) -> Optional[str]:
        if not toc_map:
            return None
        if label in toc_map:
            return label
        label_lower = label.lower()
        chapter_num_match = re.match(r"^chapter\s+(\d+)\b", label_lower)
        if chapter_num_match:
            chapter_num = chapter_num_match.group(1)
            for toc_name in toc_map:
                if re.match(rf"^chapter\s+{re.escape(chapter_num)}\b", toc_name.lower()):
                    return toc_name
        if label_lower == "appendix":
            appendix_matches = [
                toc_name for toc_name in toc_map if toc_name.lower().startswith("appendix")
            ]
            if appendix_matches:
                return min(appendix_matches, key=lambda name: toc_map.get(name, float("inf")))
        label_words = set(re.findall(r"[a-z0-9]+", label_lower))
        for toc_name in toc_map:
            toc_lower = toc_name.lower()
            if label_lower in toc_lower or toc_lower in label_lower:
                toc_words = set(re.findall(r"[a-z0-9]+", toc_lower))
                if label_words & toc_words:
                    return toc_name
        return None

    def _classify_section_type_by_keyword(self, label: str) -> str:
        """Classify a section label using keyword matching (fallback method).

        Args:
            label: Section or chapter label

        Returns:
            'pre-matter', 'main-matter', 'post-matter', or 'unknown'
        """
        if not self._is_valid_chapter_label(label):
            return "unknown"

        label_lower = label.lower()

        # Pre-matter keywords
        pre_matter_keywords = [
            "abstract",
            "acknowledgement",
            "acknowledgements",
            "dedication",
            "glossary",
            "foreword",
            "preface",
            "prologue",
            "table of contents",
            "list of figures",
            "list of tables",
            "abbreviations",
            "list of abbreviations",
        ]

        # Post-matter keywords
        post_matter_keywords = [
            "reference",
            "references",
            "bibliography",
            "appendix",
            "index",
            "epilogue",
            "colophon",
            "statement of contribution",
            "statement of contributions",
            "author note",
        ]

        for keyword in pre_matter_keywords:
            if keyword in label_lower:
                return "pre-matter"

        for keyword in post_matter_keywords:
            if keyword in label_lower:
                return "post-matter"

        # Numbered chapters are main matter
        match = re.match(r"^chapter\s+(\d+)", label_lower)
        if match:
            chapter_num = int(match.group(1))
            if 1 <= chapter_num <= 12:
                return "main-matter"

        # Standard main matter sections
        main_keywords = [
            "introduction",
            "methodology",
            "methods",
            "results",
            "findings",
            "analysis",
            "discussion",
            "conclusion",
            "conclusions",
        ]
        for keyword in main_keywords:
            if keyword in label_lower:
                return "main-matter"

        return "unknown"

    def _classify_section_type(
        self, label: str, sequence_num: float, first_chapter_seq: float, last_chapter_seq: float
    ) -> str:
        """Classify section using sequence number position relative to main chapters.

        This is more robust than keyword matching. Uses sequence numbers to determine
        if a section appears before, during, or after the main numbered chapters.

        Args:
            label: Section or chapter label
            sequence_num: Sequence number for this section
            first_chapter_seq: Sequence number of first numbered chapter (Chapter 1)
            last_chapter_seq: Sequence number of last numbered chapter

        Returns:
            'pre-matter', 'main-matter', 'post-matter', or 'unknown'
        """
        # Validate label first (reject invalid labels like "N Mean SD SE 95%")
        if not self._is_valid_chapter_label(label):
            return "unknown"

        # Reject "Unknown" labels early (fallback from invalid labels)
        if label == "Unknown":
            return "unknown"

        if re.match(r"^chapter\s+\d+", label.lower()) and sequence_num == float("inf"):
            return "main-matter"

        # If we have valid sequence bounds, use sequence-based classification
        if first_chapter_seq < float("inf") and last_chapter_seq > float("-inf"):
            # Before first numbered chapter = pre-matter
            if sequence_num < first_chapter_seq:
                return "pre-matter"
            # After last numbered chapter = post-matter
            elif sequence_num > last_chapter_seq:
                return "post-matter"
            # Between first and last = main-matter
            else:
                return "main-matter"

        # Fallback to keyword-based classification if sequence numbers unavailable
        return self._classify_section_type_by_keyword(label)

    def _extract_chapters(self, chunks_data: Dict) -> List[Dict]:
        """Extract chapter information from chunk metadata.

        Returns chapters in document order (using sequence_number metadata)
        with proper classification (pre-matter, main-matter, post-matter).

        Uses sequence number-based classification: sections before the first numbered
        chapter are pre-matter, sections after the last numbered chapter are post-matter,
        and everything in between is main-matter. Falls back to keyword matching if
        numbered chapters cannot be identified.

        Also parses ToC structure for validation where available.

        For structural coherence analysis, only main-matter chapters should be counted.

        Args:
            chunks_data: ChromaDB query result with documents, metadatas, embeddings

        Returns:
            List of chapters with name, chunk count, embedding mean, section type, and sequence number
        """
        chapters_dict: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"chunks": [], "embeddings": [], "sequence_number": float("inf")}
        )
        ordered_labels: List[str] = []

        indices = self._source_ordered_structural_indices(chunks_data)
        documents = chunks_data.get("documents", [])
        metadatas = chunks_data.get("metadatas", [])

        toc_order = self._build_toc_order(chunks_data)

        # First pass: extract all chapters and identify numbered chapter boundaries
        # Track original insertion order for stable sorting when sequence numbers are identical
        insertion_order: Dict[str, int] = {}
        active_chapter_name: Optional[str] = None
        for i in indices:
            if i >= len(metadatas) or i >= len(documents):
                continue
            meta = metadatas[i]
            doc = documents[i]

            if self._has_section_metadata(meta):
                chapter_name = self._get_chapter_label(meta, doc)
                active_chapter_name = chapter_name
            else:
                heading_match = re.match(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$", doc)
                if heading_match:
                    chapter_name = self._get_chapter_label({}, heading_match.group(1))
                    if chapter_name != "Unknown":
                        active_chapter_name = chapter_name
                else:
                    chapter_name = active_chapter_name or "Unknown"
            toc_match = self._match_toc_label(chapter_name, toc_order)
            canonical_name = toc_match or chapter_name
            if (
                toc_order
                and not toc_match
                and canonical_name.lower()
                in {
                    "introduction",
                    "methodology",
                    "results",
                    "findings",
                    "discussion",
                    "conclusion",
                }
            ):
                continue
            if canonical_name not in chapters_dict:
                ordered_labels.append(canonical_name)
                insertion_order[canonical_name] = len(ordered_labels)  # Track insertion order

            # Track lowest sequence number for ordering
            seq_num = meta.get("sequence_number", float("inf"))
            if isinstance(seq_num, (int, float)):
                chapters_dict[canonical_name]["sequence_number"] = min(
                    chapters_dict[canonical_name]["sequence_number"], seq_num
                )

            chapters_dict[canonical_name]["chunks"].append(doc)
            if chunks_data.get("embeddings") is not None and i < len(chunks_data["embeddings"]):
                embedding = chunks_data["embeddings"][i]
                if embedding is not None and np.linalg.norm(embedding) > 0:
                    chapters_dict[canonical_name]["embeddings"].append(embedding)

        # Parent chunks define the structural outline but are intentionally stored
        # with zero placeholder embeddings. Backfill chapter coherence from child
        # chunks, which retain the real semantic vectors used for retrieval.
        if chunks_data.get("embeddings") is not None:
            for i, meta in enumerate(metadatas):
                if (
                    i >= len(documents)
                    or i >= len(chunks_data["embeddings"])
                    or meta.get("chunk_type") != "child"
                    or self._is_toc_chunk(documents[i])
                ):
                    continue
                embedding = chunks_data["embeddings"][i]
                if embedding is None or np.linalg.norm(embedding) == 0:
                    continue
                chapter_name = self._get_chapter_label(meta, documents[i])
                canonical_name = self._match_toc_label(chapter_name, toc_order) or chapter_name
                if canonical_name in chapters_dict:
                    chapters_dict[canonical_name]["embeddings"].append(embedding)

        # Assign effective sequence numbers. Source extraction order is canonical;
        # table-of-contents order is a fallback only when sequence metadata is absent.
        for label in ordered_labels:
            seq_num = chapters_dict[label]["sequence_number"]
            if seq_num != float("inf"):
                chapters_dict[label]["effective_sequence"] = seq_num
            elif toc_order and label in toc_order:
                chapters_dict[label]["effective_sequence"] = toc_order[label]
            else:
                chapters_dict[label]["effective_sequence"] = float("inf")

        # Find sequence number range for numbered chapters (Chapter 1, Chapter 2, etc.)
        first_chapter_seq = float("inf")
        last_chapter_seq = float("-inf")

        for label in ordered_labels:
            label_lower = label.lower()
            match = re.match(r"^chapter\s+(\d+)", label_lower)
            if match:
                chapter_num = int(match.group(1))
                if (
                    1 <= chapter_num <= 12
                ):  # Valid chapter number, assumes max 12 chapters for PhD thesis
                    # Use effective sequence (which may be ToC-based)
                    eff_seq = chapters_dict[label]["effective_sequence"]
                    if eff_seq < float("inf"):
                        first_chapter_seq = min(first_chapter_seq, eff_seq)
                        last_chapter_seq = max(last_chapter_seq, eff_seq)

        # Sequence metadata is the source-order contract. Chapter numbers are
        # useful for validation, but must not reorder unnumbered sections.
        def _chapter_sort_key(label: str) -> Tuple[int, float, int]:
            seq_value = chapters_dict[label]["sequence_number"]
            if isinstance(seq_value, (int, float)) and seq_value != float("inf"):
                return (0, float(seq_value), insertion_order.get(label, 0))
            if toc_order and label in toc_order:
                return (1, float(toc_order[label]), insertion_order.get(label, 0))
            return (2, float("inf"), insertion_order.get(label, 0))

        ordered_labels_sorted = sorted(ordered_labels, key=_chapter_sort_key)

        # Second pass: classify and build final chapter list
        chapters: List[Dict[str, Any]] = []
        for name in ordered_labels_sorted:
            data = chapters_dict[name]
            # Use effective sequence for classification (may be ToC-based)
            seq_num = data["effective_sequence"]

            # Classify using sequence-based approach
            section_type = self._classify_section_type(
                name, seq_num, first_chapter_seq, last_chapter_seq
            )

            # Skip unknown sections (invalid labels)
            if section_type == "unknown":
                continue

            if data["embeddings"]:
                embedding_mean = np.mean(data["embeddings"], axis=0)
            else:
                embedding_mean = np.zeros(EXPECTED_EMBEDDING_DIM)

            chapters.append(
                {
                    "name": name,
                    "chunk_count": len(data["chunks"]),
                    "embedding_mean": embedding_mean,
                    "section_type": section_type,
                    "sequence_number": seq_num,
                }
            )

        return chapters

    def _detect_chapter_order_issues(
        self,
        chunks_data: Dict[str, Any],
        chapters: List[Dict[str, Any]],
    ) -> List[str]:
        """Identify source-order metadata that should be verified by a human.

        The method does not reorder chapters. It makes malformed, duplicate, or
        numerically inconsistent sequence metadata visible to the assessment
        reviewer while preserving the source-derived canonical order.

        Args:
            chunks_data: ChromaDB query result with documents and metadatas
            chapters: List of chapter dicts extracted from chunks_data
        Returns:
            List of issue strings describing potential chapter order problems.
        """
        issues: List[str] = []
        labels_by_sequence: Dict[float, set[str]] = defaultdict(set)
        documents = chunks_data.get("documents", [])
        metadatas = chunks_data.get("metadatas", [])

        for index in self._select_structural_indices(chunks_data):
            if index >= len(metadatas) or index >= len(documents):
                continue
            metadata = metadatas[index]
            raw_sequence = metadata.get("sequence_number")
            if raw_sequence is None:
                continue
            if not isinstance(raw_sequence, (int, float)):
                issues.append(
                    f"Malformed sequence metadata '{raw_sequence}' for "
                    f"{self._get_chapter_label(metadata, documents[index])}."
                )
                continue
            labels_by_sequence[float(raw_sequence)].add(
                self._get_chapter_label(metadata, documents[index])
            )

        for sequence, labels in labels_by_sequence.items():
            if len(labels) > 1:
                issues.append(
                    f"Duplicate sequence {sequence:g} is assigned to: {', '.join(sorted(labels))}."
                )

        previous_number: Optional[int] = None
        for chapter in chapters:
            match = re.match(r"^chapter\s+(\d+)\b", str(chapter["name"]), re.IGNORECASE)
            if not match:
                continue
            chapter_number = int(match.group(1))
            if previous_number is not None and chapter_number < previous_number:
                issues.append(
                    "Source sequence orders numbered chapters as "
                    f"Chapter {previous_number} before Chapter {chapter_number}."
                )
            previous_number = chapter_number

        return list(dict.fromkeys(issues))

    def _detect_missing_sections(self, chunks_data: Dict) -> List[str]:
        """Detect missing required sections by parsing chunk text content.

        Args:
            chunks_data: ChromaDB query result with documents and metadatas

        Returns:
            List of missing section names from required_sections
        """
        import re

        found_sections = set()

        for i, doc in enumerate(chunks_data["documents"]):
            # CRITICAL: Skip ToC chunks to prevent false positives from ToC entries
            if self._is_toc_chunk(doc):
                continue

            # First check metadata
            meta = chunks_data["metadatas"][i]
            section = meta.get("section_title", "").lower()
            chapter = meta.get("chapter", "").lower()

            # Then check the actual text content
            doc_lower = doc.lower()

            for required in self.required_sections:
                # Check metadata first
                if required in section or required in chapter:
                    found_sections.add(required)
                    continue

                # Check for section heading patterns in text
                # Pattern 1: "Introduction" or "1.1 Introduction" or "1.1. Introduction"
                patterns = [
                    rf"\b{required}\b",  # Word boundary match
                    rf"^\s*{required}\s*$",  # Standalone line
                    rf"\n\s*{required}\s*\n",  # Between newlines
                    rf"\n\s*\d+\.?\d*\.?\s*{required}\s*\n",  # Numbered section
                ]

                for pattern in patterns:
                    if re.search(pattern, doc_lower, re.MULTILINE):
                        found_sections.add(required)
                        break

        return sorted(list(self.required_sections - found_sections))

    def _compute_chapter_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def _extract_citations(self, chunks_data: Dict) -> List[Dict]:
        """Extract citation metadata from SQLite citation graph database.

        Args:
            chunks_data: ChromaDB query result with documents and metadatas

        Returns:
            List of citations with metadata (node_id, title, authors, year, doi, reference_type, source)
        """
        import sqlite3
        from pathlib import Path

        citations: List[Dict[str, Any]] = []

        # Get doc_id and source_category from chunks_data
        if not chunks_data.get("metadatas"):
            return citations

        doc_id = chunks_data["metadatas"][0].get("doc_id", "") if chunks_data["metadatas"] else ""
        source_category = (
            chunks_data["metadatas"][0].get("source_category", "")
            if chunks_data["metadatas"]
            else ""
        )

        # Check if citation database exists
        db_path = Path(self.citation_db_path)
        if not db_path.exists():
            return citations

        conn = None
        try:
            # Connect to citation graph database
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Strategy 1: Try exact doc_id match
            query = """
                SELECT DISTINCT n.node_id, n.title, n.authors, n.year, n.doi, 
                       n.reference_type, n.source
                FROM nodes n
                INNER JOIN edges e ON n.node_id = e.target
                WHERE e.source = ? AND n.node_type = 'reference'
            """
            cursor.execute(query, (doc_id,))
            rows = cursor.fetchall()

            # Strategy 2: If no exact match, try partial matches on title/filename
            if not rows and doc_id:
                # Extract key parts from doc_id (e.g., author name, year)
                parts = doc_id.replace("_", " ").split()

                # Try matching document nodes by title similarity
                cursor.execute("SELECT node_id FROM nodes WHERE node_type = 'document'")
                doc_nodes = cursor.fetchall()

                for doc_node in doc_nodes:
                    node_id = doc_node["node_id"]
                    # Check if any significant part matches
                    node_parts = node_id.replace("_", " ").lower().split()
                    doc_parts = [p.lower() for p in parts if len(p) > 3]  # Skip short words

                    # If at least 2 significant words match, consider it a match
                    matches = sum(1 for dp in doc_parts if any(dp in np for np in node_parts))
                    if matches >= 2 or len(doc_nodes) == 1:  # Or if only one document exists
                        cursor.execute(query.replace("e.source = ?", "e.source = ?"), (node_id,))
                        rows = cursor.fetchall()
                        if rows:
                            break

            # Strategy 3: If academic paper and only one document in graph, use it
            if not rows and source_category == "academic_paper":
                cursor.execute("SELECT COUNT(*) as count FROM nodes WHERE node_type = 'document'")
                doc_count = cursor.fetchone()["count"]

                if doc_count == 1:
                    # Get the single document's citations
                    cursor.execute("""
                        SELECT DISTINCT n.node_id, n.title, n.authors, n.year, n.doi, 
                               n.reference_type, n.source
                        FROM nodes n
                        INNER JOIN edges e ON n.node_id = e.target
                        WHERE e.source = (SELECT node_id FROM nodes WHERE node_type = 'document' LIMIT 1)
                          AND n.node_type = 'reference'
                    """)
                    rows = cursor.fetchall()

            # Convert rows to citation dicts
            for row in rows:
                citations.append(
                    {
                        "node_id": row["node_id"],
                        "title": row["title"],
                        "authors": row["authors"],
                        "year": row["year"],
                        "doi": row["doi"],
                        "reference_type": row["reference_type"],
                        "source": row["source"],
                    }
                )

        except Exception as e:
            # Silently fail - citation analysis is optional
            pass
        finally:
            if conn:
                conn.close()

        return citations

    def _is_recent(self, year: Optional[int], threshold_years: int = 5) -> bool:
        """Check if year is within threshold_years of current year.

        Args:
            year: Publication year to check
            threshold_years: Number of years to consider as "recent"

        Returns:
            True if year is recent, False otherwise
        """
        if not year:
            return False
        current_year = datetime.now().year
        return (current_year - year) <= threshold_years

    def _cluster_citations(self, citations: List[Dict]) -> Dict[str, int]:
        """Cluster citations by venue/source.

        Args:
            citations: List of citation dicts with metadata

        Returns:
            Dict mapping venue/source to count of citations from that venue
        """
        from collections import defaultdict

        venue_counts: Dict[str, int] = defaultdict(int)

        for citation in citations:
            # Group by source or reference_type
            venue = citation.get("source", citation.get("reference_type", "Unknown"))
            if venue:
                venue_counts[venue] += 1

        return dict(venue_counts)

    def _detect_orphaned_claims(self, chunks_data: Dict) -> List[str]:
        """Detect claims without supporting citations in text.

        This should be called from claim analysis, not citation patterns.
        Returns list of text snippets with unsupported claims.

        Args:
            chunks_data: ChromaDB query result with documents and metadatas

        Returns:
            List of text snippets that contain claims without citations
        """
        import re

        orphaned: List[str] = []
        orphaned_norm: List[str] = []

        # Claim indicators
        claim_patterns = [
            r"\b(prove[sd]?|demonstrate[sd]?|establish(?:ed)?|confirm[se]d?|show[sn]?)\b",
            r"\b(clearly|obviously|undoubtedly|certainly|undeniably)\b",
            r"\b(all|every|always|never|no one|everyone)\b",
        ]

        # Citation indicators
        citation_patterns = [
            r"\([A-Z][a-z]+(?:\s+et al\.)?,\s+\d{4}\)",  # (Author, 2020)
            r"\[[0-9]+\]",  # [1]
            r"\b[A-Z][a-z]+\s+\(\d{4}\)",  # Author (2020)
        ]

        claim_re = re.compile("|".join(claim_patterns), re.IGNORECASE)
        citation_re = re.compile("|".join(citation_patterns))

        def _make_snippet(doc: str, start: int, end: int) -> str:
            snippet_start = max(0, start)
            snippet_end = min(len(doc), end)
            while snippet_start > 0 and doc[snippet_start - 1].isalnum():
                snippet_start -= 1
            while snippet_end < len(doc) and doc[snippet_end : snippet_end + 1].isalnum():
                snippet_end += 1
            snippet = doc[snippet_start:snippet_end].replace("\n", " ").strip()
            return " ".join(snippet.split())

        def _normalise_snippet(text: str) -> str:
            return " ".join(text.lower().split())

        for i, doc in enumerate(chunks_data["documents"]):
            # CRITICAL: Skip ToC chunks to prevent extracting list entries as claims
            if self._is_toc_chunk(doc):
                continue

            # Check if chunk contains claim language
            claim_match = claim_re.search(doc)
            if claim_match:
                # Check if it also contains citations
                if not citation_re.search(doc):
                    # Extract snippet around claim
                    start = claim_match.start() - 140
                    end = claim_match.end() + 140
                    snippet = _make_snippet(doc, start, end)
                    if not snippet:
                        continue
                    norm = _normalise_snippet(snippet)
                    replaced = False
                    for idx, existing_norm in enumerate(orphaned_norm):
                        if norm.startswith(existing_norm) or existing_norm.startswith(norm):
                            if len(norm) > len(existing_norm):
                                orphaned[idx] = snippet
                                orphaned_norm[idx] = norm
                            replaced = True
                            break
                    if replaced:
                        continue
                    if len(orphaned) < 10:  # Limit to 10 examples
                        orphaned.append(snippet)
                        orphaned_norm.append(norm)

        return orphaned

    def _compute_geographic_diversity(self, citations: List[Dict]) -> float:
        """Compute geographic diversity of author affiliations.

        TODO: This is a placeholder. Requires author affiliation data from the citation graph database.
        The idea is to analyse the affiliations of cited authors to determine if the thesis draws on a geographically diverse set of sources,
        which can be an indicator of breadth and inclusivity in research.

        Args:
            citations: List of citation dicts with metadata (including author affiliations if available)

        Returns:
            Geographic diversity score (0 to 1), where higher means more diverse
        """
        # Placeholder (requires author affiliation data)
        return 0.5

    def _get_chapter_sizes(self, chunks_data: Dict) -> Dict[str, int]:
        """Get word count per chapter.

        Args:
            chunks_data: ChromaDB query result with documents and metadatas

        Returns:
            Dict mapping chapter names to total word count in that chapter
        """
        chapter_sizes: Dict[str, int] = defaultdict(int)

        for i, meta in enumerate(chunks_data["metadatas"]):
            chapter = meta.get("chapter", meta.get("section_title", "Unknown"))
            word_count = len(chunks_data["documents"][i].split())
            chapter_sizes[chapter] += word_count

        return dict(chapter_sizes)

    def _collect_text_by_section(
        self,
        chunks_data: Dict,
        include_sections: Optional[List[str]] = None,
        max_chars: int = 200000,
    ) -> str:
        """Collect concatenated text filtered by section keywords.

        Args:
            chunks_data: ChromaDB query result with metadatas and documents
            include_sections: Section names/keywords to filter by (case-insensitive)
            max_chars: Maximum characters to collect

        Returns:
            Concatenated text from matching sections (excluding ToC chunks)

        Note:
            Searches in section_title, parent_section, heading_path, and chapter fields.
            Filters out Table of Contents chunks to prevent extracting list entries as claims.
            If no matches found with include_sections, falls back to all non-ToC text.
        """
        collected = []
        matched = False
        include_sections_lc = (
            [key.lower() for key in include_sections] if include_sections else None
        )

        for i, meta in enumerate(chunks_data.get("metadatas", [])):
            doc = (
                chunks_data.get("documents", [])[i]
                if i < len(chunks_data.get("documents", []))
                else ""
            )

            # CRITICAL: Filter out Table of Contents chunks
            if self._is_toc_chunk(doc):
                continue

            # Check multiple metadata fields for section information
            section_title = (meta.get("section_title") or "").lower()
            parent_section = (meta.get("parent_section") or "").lower()
            heading_path = (meta.get("heading_path") or "").lower()
            chapter = (meta.get("chapter") or "").lower()

            # Combine all section-related fields for matching
            section_text = f"{section_title} {parent_section} {heading_path} {chapter}"

            if include_sections_lc:
                if not any(key in section_text for key in include_sections_lc):
                    continue
                matched = True

            collected.append(doc)
            if sum(len(c) for c in collected) >= max_chars:
                break

        # If no sections matched, fall back to all non-ToC text
        if include_sections_lc and not matched:
            return self._collect_text_by_section(
                chunks_data, include_sections=None, max_chars=max_chars
            )

        return "\n".join(collected)

    def _contains_any(self, text: str, keywords: List[str]) -> bool:
        """Check if any keyword appears in text.

        Args:
            text: Text to search within
            keywords: List of keywords to check for (case-insensitive)

        Returns:
            True if any keyword is found in text, False otherwise
        """
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in keywords)

    def _split_sentences(self, text: str) -> List[str]:
        """Naive sentence splitter.

        N.B. This is a heuristic splitter and does not handle edge cases (e.g., abbreviations, decimal points, or PDF newlines breaking sentences).

        TODO: This is a very basic splitter.
        Consider using a more robust sentence tokeniser (like NLTK's sent_tokenize)
        but be mindful of dependencies and performance and whether needed for this application.

        Args:
            text: Text to split into sentences

        Returns:
            List of sentences (split on ., ?, !, and newlines)
        """
        if not text:
            return []
        separators = [". ", "? ", "! ", "\n"]
        for sep in separators:
            text = text.replace(sep, sep.strip() + "|")
        return [s.strip() for s in text.split("|") if s.strip()]

    def _tokenise_words(self, text: str) -> List[str]:
        """Tokenise text into lowercase words.

        Args:
            text: Text to tokenise
        Returns:
            List of word tokens (alphanumeric, apostrophes, and hyphens)
        """
        tokens = []
        current = []
        for ch in text:
            if ch.isalnum() or ch in ("'", "-"):
                current.append(ch.lower())
            else:
                if current:
                    tokens.append("".join(current))
                    current = []
        if current:
            tokens.append("".join(current))
        return tokens

    def _extract_claims_from_text(self, text: str) -> List[str]:
        """Extract likely claims using heuristic patterns.

        Args:
            text: Text to extract claims from

        Returns:
            List of extracted claim sentences (up to 200)

        """
        if not text:
            return []
        claim_markers = (
            "we find",
            "we show",
            "this thesis",
            "this study",
            "results show",
            "results indicate",
            "demonstrates",
            "suggests",
            "indicates",
            "confirms",
        )

        def _looks_truncated(sentence: str) -> bool:
            stripped = sentence.strip()
            if not stripped:
                return True
            if stripped.endswith("-"):
                return True
            first_word = stripped.split()[0]
            if stripped[0].islower() and len(first_word) <= 3 and first_word not in {"i"}:
                return True
            return False

        claims: List[str] = []
        seen: set[str] = set()
        for sentence in self._split_sentences(text):
            sentence_clean = " ".join(sentence.split()).strip()
            if not sentence_clean:
                continue
            sentence_lower = sentence_clean.lower()
            if any(marker in sentence_lower for marker in claim_markers):
                if _looks_truncated(sentence_clean):
                    continue
                if sentence_lower in seen:
                    continue
                seen.add(sentence_lower)
                claims.append(sentence_clean)
            if len(claims) >= 200:
                break
        return claims

    def _extract_claims_from_text_llm(self, text: str) -> List[str]:
        """Extract claims using an optional LLM client.

        Args:
            text: Text to extract claims from

        Returns:
            List of extracted claim sentences (up to 30)

        """
        prompt = (
            "You are extracting thesis claims. Return JSON only.\n"
            "Extract up to 30 concise, declarative claims from the text.\n"
            'JSON format: {"claims": ["...", "..."]}\n\n'
            f"Text:\n{text}\n"
        )
        response = self._llm_invoke(prompt)
        try:
            data = extract_first_json_block(response) if response else None
        except (ValueError, Exception):
            data = None
        if isinstance(data, dict) and isinstance(data.get("claims"), list):
            claims: List[str] = []
            seen: set[str] = set()
            for claim in data["claims"]:
                if not isinstance(claim, str):
                    continue
                claim_clean = " ".join(claim.split()).strip()
                if not claim_clean:
                    continue
                claim_lower = claim_clean.lower()
                if claim_lower in seen:
                    continue
                seen.add(claim_lower)
                claims.append(claim_clean)
            return claims[:30]
        return []

    def _detect_contradictions(self, claims: List[str]) -> List[Dict[str, Any]]:
        """Detect potential contradictions using polarity and token overlap.

        Args:
            claims: List of claim sentences to analyse for contradictions

        Returns:
            List of detected contradictions with claim pairs and overlap score (up to 20)

        """
        contradictions: List[Dict[str, Any]] = []
        if len(claims) < 2:
            return contradictions
        negations = {"not", "no", "never", "none", "cannot", "failed", "fails"}
        for i in range(len(claims)):
            tokens_a = set(self._tokenise_words(claims[i]))
            if not tokens_a:
                continue
            polarity_a = any(tok in negations for tok in tokens_a)
            for j in range(i + 1, len(claims)):
                tokens_b = set(self._tokenise_words(claims[j]))
                if not tokens_b:
                    continue
                overlap = len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))
                if overlap < 0.35:
                    continue
                polarity_b = any(tok in negations for tok in tokens_b)
                if polarity_a != polarity_b:
                    contradictions.append(
                        {
                            "claim_a": claims[i],
                            "claim_b": claims[j],
                            "overlap": overlap,
                            "source": "heuristic",
                        }
                    )
                if len(contradictions) >= 20:
                    return contradictions
        return contradictions

    def _detect_contradictions_llm(self, claims: List[str]) -> List[Dict[str, Any]]:
        """Detect contradictions using an optional LLM client.

        Args:
            claims: List of claim sentences to analyse for contradictions

        Returns:
            List of detected contradictions with claim pairs and reasons (up to 20)

        """
        if not claims:
            return []
        claims_text = "\n".join(f"- {c}" for c in claims[:30])
        prompt = (
            "You are detecting contradictions between thesis claims. Return JSON only.\n"
            "Find contradictory claim pairs if any.\n"
            'JSON format: {"contradictions": [{"claim_a": "...", "claim_b": "...", "reason": "..."}]}\n\n'
            f"Claims:\n{claims_text}\n"
        )
        response = self._llm_invoke(prompt)
        try:
            data = extract_first_json_block(response) if response else None
        except (ValueError, Exception):
            data = None
        if isinstance(data, dict) and isinstance(data.get("contradictions"), list):
            cleaned = []
            for item in data["contradictions"]:
                if not isinstance(item, dict):
                    continue
                claim_a = item.get("claim_a")
                claim_b = item.get("claim_b")
                if isinstance(claim_a, str) and isinstance(claim_b, str):
                    cleaned.append(
                        {
                            "claim_a": claim_a.strip(),
                            "claim_b": claim_b.strip(),
                            "reason": str(item.get("reason", "")).strip(),
                            "source": "llm",
                        }
                    )
            return cleaned[:20]
        return []

    def _llm_invoke(self, prompt: str) -> str:
        """Invoke llm_client, supporting callable or .invoke().

        Args:
            prompt: Prompt string to send to the LLM client

        Returns:
            Response from the LLM client as a string, or empty string on failure

        """
        if callable(self.llm_client):
            client = self.llm_client
        elif hasattr(self.llm_client, "invoke"):
            client = self.llm_client
        else:
            return ""

        return invoke_with_usage(
            client,
            prompt,
            operation="phd_assessor.llm_invoke",
            component="phd_assessment",
        )

    def _llm_enabled(self, key: str) -> bool:
        """Check if a given LLM feature is enabled via flags.

        Args:
            key: Feature key to check (e.g., 'claim_extraction', 'contradiction_detection')

        Returns:
            True if the feature is enabled, False otherwise
        """
        return bool(self.llm_flags.get(key, False))

    def _flesch_reading_ease(self, text: str) -> float:
        """Compute Flesch Reading Ease score.

        Args:
            text: Text to compute readability for

        Returns:
            Flesch Reading Ease score (0 to 100, higher is easier to read)
        """
        sentences = self._split_sentences(text)
        words = self._tokenise_words(text)
        syllables = sum(self._count_syllables(word) for word in words)

        if not sentences or not words:
            return 0.0

        words_per_sentence = len(words) / max(1, len(sentences))
        syllables_per_word = syllables / max(1, len(words))
        score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        return max(0.0, min(100.0, score))

    def _flesch_education_level(self, score: float) -> str:
        """Map Flesch Reading Ease to an education level description."""
        if score >= 90:
            return "5th grade"
        if score >= 80:
            return "6th grade"
        if score >= 70:
            return "7th grade"
        if score >= 60:
            return "8th-9th grade"
        if score >= 50:
            return "10th-12th grade"
        if score >= 30:
            return "undergraduate"
        return "postgraduate"

    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for a word.

        Args:
            word: Word to count syllables in

        Returns:
            Estimated number of syllables in the word
        """
        word = word.lower().strip()
        if not word:
            return 0
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    def _passive_voice_ratio(self, sentences: List[str]) -> float:
        """Estimate passive voice ratio based on heuristic patterns.

        Args:
            sentences: List of sentences to analyse for passive voice

        Returns:
            Ratio of sentences likely in passive voice (0 to 1)
        """
        if not sentences:
            return 0.0
        be_verbs = {"is", "are", "was", "were", "be", "been", "being"}
        passive_count = 0
        for sentence in sentences:
            tokens = self._tokenise_words(sentence)
            for i in range(len(tokens) - 1):
                if tokens[i] in be_verbs and (
                    tokens[i + 1].endswith("ed") or tokens[i + 1] in {"known", "shown", "seen"}
                ):
                    passive_count += 1
                    break
        return passive_count / max(1, len(sentences))

    def _jargon_density(self, words: List[str]) -> float:
        """Estimate jargon density using long words and acronyms.

        Args:
            words: List of word tokens to analyse for jargon

        Returns:
            Jargon density score (0 to 1), where higher means more jargon
        """
        if not words:
            return 0.0
        jargon_count = 0
        for word in words:
            if len(word) >= 12:
                jargon_count += 1
            elif word.isupper() and len(word) >= 3:
                jargon_count += 1
        return jargon_count / max(1, len(words))

    def _extract_keywords(self, text: str, limit: int = 15) -> List[str]:
        """Extract top keywords from text using frequency, ignoring stopwords.

        Uses NLTK stopwords + domain-specific academic stopwords from terminology module.

        Args:
            text: Text to extract keywords from
            limit: Maximum number of keywords to return

        Returns:
            List of top keywords sorted by frequency (up to limit)
        """
        tokens = [t for t in self._tokenise_words(text) if t not in _STOPWORDS and len(t) > 3]
        if not tokens:
            return []
        freqs: Dict[str, int] = defaultdict(int)
        for tok in tokens:
            freqs[tok] += 1
        sorted_tokens = sorted(freqs.items(), key=lambda x: (-x[1], x[0]))
        return [tok for tok, _ in sorted_tokens[:limit]]

    def _extract_strong_claims(self, text: str) -> List[str]:
        """Extract strong claims from text using marker verbs.

        Args:
            text: Text to extract strong claims from

        Returns:
            List of strong claim sentences (up to 50)
        """
        if not text:
            return []
        strong_markers = (
            "prove",
            "demonstrate",
            "establish",
            "confirm",
            "show",
            "evidence",
            "causal",
            "significant",
        )
        claims = []
        for sentence in self._split_sentences(text):
            sentence_lower = sentence.lower()
            if any(marker in sentence_lower for marker in strong_markers):
                claims.append(sentence.strip())
            if len(claims) >= 50:
                break
        return claims

    def _extract_section_embeddings(self, chunks_data: Dict) -> List[Dict[str, Any]]:
        """Extract ordered section embeddings using metadata section labels.

        TODO: Should not be stating section number which is largely meaningless to the user in relation to the document.
        Should be using section labels from metadata which are more meaningful and human-readable, showing the actual chapter/section titles where possible.

        Args:
            chunks_data: ChromaDB query result with metadatas and embeddings

        Returns:
            List of sections with their names and mean embeddings, ordered by first-seen section label in metadata. Sections with "Unknown" label are renamed to "Section N" based on first-seen order to ensure consistent grouping.
        """
        section_order: List[str] = []
        section_map: Dict[str, Dict[str, Any]] = {}
        unknown_counter = 0

        indices = self._source_ordered_structural_indices(chunks_data)
        metadatas = chunks_data.get("metadatas", [])
        documents = chunks_data.get("documents", [])

        for i in indices:
            if i >= len(metadatas):
                continue
            meta = metadatas[i]
            doc = documents[i] if i < len(documents) else ""
            section = self._get_section_label(meta, doc)
            if section == "Unknown":
                unknown_counter += 1
                section = f"Section {unknown_counter}"
            if section not in section_map:
                section_order.append(section)
                section_map[section] = {"embeddings": []}
            if chunks_data.get("embeddings") is not None and i < len(chunks_data["embeddings"]):
                section_map[section]["embeddings"].append(chunks_data["embeddings"][i])

        sections = []
        for section in section_order:
            embeddings = section_map[section]["embeddings"]
            if embeddings:
                embedding_mean = np.mean(embeddings, axis=0)
            else:
                embedding_mean = np.zeros(EXPECTED_EMBEDDING_DIM)
            sections.append({"name": section, "embedding_mean": embedding_mean})

        return sections

    def _group_text_by_section(self, chunks_data: Dict) -> Dict[str, str]:
        """Group chunk text by section label, preserving first-seen order.

        Args:
            chunks_data: ChromaDB query result with metadatas and documents

        Returns:
            Dict mapping section labels to concatenated text from chunks in that section, preserving the order of first occurrence of each section label in the metadata. Sections with "Unknown" label are renamed to "Section N" based on first-seen order to ensure consistent grouping.
        """
        section_text: Dict[str, List[str]] = defaultdict(list)
        indices = self._source_ordered_structural_indices(chunks_data)
        documents = chunks_data.get("documents", [])
        metadatas = chunks_data.get("metadatas", [])
        unknown_counter = 0

        for i in indices:
            if i >= len(metadatas) or i >= len(documents):
                continue
            meta = metadatas[i]
            doc = documents[i] if i < len(documents) else ""
            label = self._get_section_label(meta, doc)
            if label == "Unknown":
                unknown_counter += 1
                label = f"Section {unknown_counter}"
            section_text[label].append(documents[i])

        return {label: "\n".join(texts) for label, texts in section_text.items()}

    def _group_text_by_chapter(self, chunks_data: Dict) -> Dict[str, str]:
        """Group chunk text by chapter label, preserving document order.

        CRITICAL: Must NOT transform Unknown labels to "Chapter N" because
        the chapters list already has specific labels from _extract_chapters().
        Mismatched labels will break concept progression tracking.

        Args:
            chunks_data: ChromaDB query result with metadatas and documents

        Returns:
            Dict mapping chapter labels to concatenated text from chunks in that chapter, preserving the order of first occurrence of each chapter label in the metadata. Unknown labels are kept as "Unknown" to ensure consistency with chapter labels extracted in _extract_chapters().
        """
        chapter_text: Dict[str, List[str]] = defaultdict(list)
        indices = self._select_structural_indices(chunks_data)
        documents = chunks_data.get("documents", [])
        metadatas = chunks_data.get("metadatas", [])
        toc_order = self._build_toc_order(chunks_data)

        for i in indices:
            if i >= len(metadatas) or i >= len(documents):
                continue
            meta = metadatas[i]
            doc = documents[i] if i < len(documents) else ""
            # Use the same label extraction as _extract_chapters
            label = self._get_chapter_label(meta, doc)
            toc_match = self._match_toc_label(label, toc_order)
            canonical_label = toc_match or label
            if (
                toc_order
                and not toc_match
                and canonical_label.lower()
                in {
                    "introduction",
                    "methodology",
                    "results",
                    "findings",
                    "discussion",
                    "conclusion",
                }
            ):
                continue
            # NOTE: Keep "Unknown" as-is, do NOT transform to "Chapter N"
            # This ensures text_map keys match chapters list names
            chapter_text[canonical_label].append(documents[i])

        canonical_order = [chapter["name"] for chapter in self._extract_chapters(chunks_data)]
        ordered_labels = [label for label in canonical_order if label in chapter_text]
        ordered_labels.extend(label for label in chapter_text if label not in canonical_order)
        return {label: "\n".join(chapter_text[label]) for label in ordered_labels}

    def _detect_data_conclusion_mismatch_llm(
        self, results_text: str, conclusion_text: str
    ) -> List[Dict[str, Any]]:
        """Use LLM to detect mismatches between results and conclusions.

        Args:
            results_text: Concatenated text from results/findings sections
            conclusion_text: Concatenated text from conclusion/discussion sections

        Returns:
            List of detected issues with claims and reasons (up to 20)
        """
        prompt = (
            "You compare results and conclusions for mismatches. Return JSON only.\n"
            'JSON format: {"issues": [{"claim": "...", "reason": "..."}]}\n\n'
            f"Results:\n{results_text}\n\nConclusions:\n{conclusion_text}\n"
        )
        response = self._llm_invoke(prompt)
        try:
            data = extract_first_json_block(response) if response else None
        except (ValueError, Exception):
            data = None
        if isinstance(data, dict) and isinstance(data.get("issues"), list):
            cleaned = []
            for item in data["issues"]:
                if isinstance(item, dict) and isinstance(item.get("claim"), str):
                    cleaned.append(
                        {
                            "claim": item.get("claim", "").strip(),
                            "reason": str(item.get("reason", "")).strip(),
                        }
                    )
            return cleaned[:20]
        return []

    def _detect_citation_misrepresentation_llm(
        self,
        claims: List[str],
        reference_titles: List[str],
    ) -> List[Dict[str, Any]]:
        """Use LLM to flag claims not supported by reference titles.

        Args:
            claims: List of claim sentences extracted from the thesis
            reference_titles: List of titles from the citation graph database

        Returns:
            List of issues where claims may not be supported by references, with claim and reason (up to 20)
        """
        if not claims or not reference_titles:
            return []
        claims_text = "\n".join(f"- {c}" for c in claims[:20])
        refs_text = "\n".join(f"- {t}" for t in reference_titles[:30])
        prompt = (
            "You are checking whether claims align with reference titles. Return JSON only.\n"
            'JSON format: {"issues": [{"claim": "...", "reason": "..."}]}\n\n'
            f"Claims:\n{claims_text}\n\nReference Titles:\n{refs_text}\n"
        )
        response = self._llm_invoke(prompt)
        try:
            data = extract_first_json_block(response) if response else None
        except (ValueError, Exception):
            data = None
        if isinstance(data, dict) and isinstance(data.get("issues"), list):
            cleaned = []
            for item in data["issues"]:
                if isinstance(item, dict) and isinstance(item.get("claim"), str):
                    cleaned.append(
                        {
                            "claim": item.get("claim", "").strip(),
                            "reason": str(item.get("reason", "")).strip(),
                        }
                    )
            return cleaned[:20]
        return []

    def _compute_overall_score(
        self,
        structure: StructureAnalysis,
        citations: CitationPatternAnalysis,
        claims: ClaimAnalysis,
        methodology: MethodologyChecklist,
        writing: WritingQualityMetrics,
        alignment: ContributionAlignment,
        persona: str,
    ) -> float:
        """
        Compute a weighted assessment signal for human review.

        Persona weights:
        - supervisor: structure (0.6), citations (0.4)
        - assessor: structure (0.5), citations (0.5)
        - researcher: structure (0.4), citations (0.6)

        Args:
            structure: StructureAnalysis results
            citations: CitationPatternAnalysis results
            claims: ClaimAnalysis results
            methodology: MethodologyChecklist results
            writing: WritingQualityMetrics results
            alignment: ContributionAlignment results
            persona: Persona type to determine weighting (e.g., 'supervisor', 'assessor', 'researcher')

        Returns:
            Assessment signal (0 to 1), not an examination outcome.
        """
        weights = {
            "supervisor": {
                "structure": 0.4,
                "citations": 0.25,
                "methodology": 0.15,
                "writing": 0.1,
                "alignment": 0.1,
            },
            "assessor": {
                "structure": 0.3,
                "citations": 0.25,
                "methodology": 0.25,
                "writing": 0.1,
                "alignment": 0.1,
            },
            "researcher": {
                "structure": 0.3,
                "citations": 0.3,
                "methodology": 0.15,
                "writing": 0.1,
                "alignment": 0.15,
            },
        }

        w = weights.get(persona, weights["supervisor"])

        # Structure score (includes coherence and RQ alignment)
        structure_score = (structure.avg_coherence + structure.rq_alignment_score) / 2
        if structure.missing_sections:
            structure_score *= 0.7  # Penalty for missing sections
        if structure.orphaned_concepts:
            structure_score *= 0.95  # Minor penalty for undeveloped concepts

        # Citation score. Unavailable citation evidence is excluded rather than
        # treated as a thesis defect, and remaining weights are re-normalised.
        citation_score = (citations.citation_recency_score + citations.geographic_diversity) / 2
        if citations.orphaned_claims:
            citation_score *= 0.8  # Penalty for unsupported claims

        # Methodology score
        methodology_score = methodology.score

        # Writing score (normalise readability to 0-1)
        writing_score = max(0.0, min(1.0, writing.readability_score / 100.0))
        writing_score *= max(0.0, 1.0 - writing.passive_voice_ratio)

        # Alignment score
        alignment_score = alignment.overlap_score

        scored_components = {
            "structure": structure_score,
            "methodology": methodology_score,
            "writing": writing_score,
            "alignment": alignment_score,
        }
        if citations.citation_evidence_available:
            scored_components["citations"] = citation_score

        active_weight = sum(w[name] for name in scored_components)
        return sum(w[name] * score for name, score in scored_components.items()) / active_weight

    def _collect_research_inquiry_text(
        self,
        chunks_data: Dict,
        section_types: Optional[Dict[str, str]] = None,
    ) -> str:
        """Collect pre-matter and main-matter text while excluding instruments.

        Args:
            chunks_data: ChromaDB query result with metadatas and documents
            section_types: Optional F1 section identity map keyed by canonical section name.

        Returns:
            A string containing the concatenated text of included sections.
        """
        documents = chunks_data.get("documents", [])
        metadatas = chunks_data.get("metadatas", [])
        if not metadatas:
            return ""

        included = []
        excluded_keywords = (
            "appendix",
            "interview guide",
            "interview schedule",
            "interview protocol",
            "survey instrument",
            "questionnaire",
        )
        if section_types is None:
            section_types = {
                chapter["name"]: chapter["section_type"]
                for chapter in self._extract_chapters(chunks_data)
            }
        toc_order = self._build_toc_order(chunks_data)

        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            section_text = " ".join(
                str(metadata.get(field) or "")
                for field in ("section_title", "parent_section", "heading_path", "chapter")
            ).casefold()
            document_start = str(document[:200]).casefold()
            if any(
                keyword in section_text or keyword in document_start
                for keyword in excluded_keywords
            ):
                continue
            section_label = self._get_chapter_label(metadata, str(document))
            section_label = self._match_toc_label(section_label, toc_order) or section_label
            section_type = section_types.get(section_label)
            if section_type in {"pre-matter", "main-matter"}:
                included.append(document)
        return "\n".join(included)

    def _extract_research_questions(
        self,
        chunks_data: Dict,
        section_types: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Extract explicit questions, aims and objectives from thesis narrative sections.

        Args:
            chunks_data: ChromaDB query result with metadatas and documents

        Returns:
            List of extracted research questions (up to 10)
        """
        import re

        text = self._collect_research_inquiry_text(chunks_data, section_types)

        research_questions = []

        statement_patterns = [
            r"\b(?:the\s+)?(?:primary\s+|main\s+)?aim\s+of\s+(?:this|the)\s+"
            r"(?:study|research|thesis)\s+(?:is|was)\s+[^.!?\n]+[.!?]",
            r"\b(?:the\s+)?(?:primary\s+|specific\s+)?research\s+aims?\s+"
            r"(?:are|is|were|was|include|includes)\s+[^.!?\n]+[.!?]",
            r"\b(?:the\s+)?(?:research\s+)?objectives?\s+"
            r"(?:are|is|were|was|include|includes)\s+[^.!?\n]+[.!?]",
            r"\bthe\s+purpose\s+of\s+(?:this|the)\s+(?:study|research|thesis)\s+"
            r"(?:is|was)\s+[^.!?\n]+[.!?]",
        ]
        for pattern in statement_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                candidate = " ".join(match.group(0).split()).strip()
                if len(candidate) >= 25:
                    research_questions.append(candidate)

        allowed_inquiry_types = {
            "research_question",
            "sub_question",
            "aim",
            "objective",
            "hypothesis",
            "guiding_question",
        }
        profile_framings = (self.cultural_lens_profile or {}).get("research_question_framings", [])
        for framing in profile_framings:
            if framing.get("classify_as") not in allowed_inquiry_types:
                continue
            indicators = [
                str(indicator).casefold()
                for indicator in framing.get("indicators", [])
                if indicator
            ]
            if not indicators:
                continue
            for sentence in self._split_sentences(text):
                if any(_contains_whole_phrase(sentence, indicator) for indicator in indicators):
                    candidate = " ".join(sentence.split()).strip()
                    if 20 <= len(candidate) <= 500:
                        research_questions.append(candidate)

        # Pattern 1: "RQ1:", "RQ 1:", "Research Question(s) 1:", "Question(s) 1:" followed by actual question text
        # Handles singular/plural and optional numbering
        pattern1 = (
            r"(?:RQ\s*\d+(?:[a-z]|\.\d+)*|Research\s+Questions?\s*\d*(?:[a-z]|\.\d+)*|"
            r"Questions?\s*\d+(?:[a-z]|\.\d+)*)\s*[:.)-]?\s*([^?.!\n]+[?.!])"
        )
        matches = re.findall(pattern1, text, re.IGNORECASE)
        for match in matches:
            cleaned = match.strip()
            # Filter out short fragments, meta-references, and table headers
            if len(cleaned) < 15:  # Too short
                continue
            if any(
                word in cleaned.lower()
                for word in ["alignment to", "variable", "construct", "table", "figure", "appendix"]
            ):
                continue
            if cleaned.lower().startswith("and "):  # Fragment
                continue
            research_questions.append(cleaned)

        # Pattern 2: Look for any complete question (text ending with ?) in context after "research questions"
        # Split text by the phrase to get context after it
        # More specific pattern to avoid matching ToC - require "were/are as follows" or similar close phrasing
        rq_intro_pattern = r"(?:major\s+)?research questions?\s+(?:were|are|is)\s+as follows[:\s]+"
        split_pos = 0
        for match in re.finditer(rq_intro_pattern, text, re.IGNORECASE):
            split_pos = match.end()
            break

        if split_pos > 0:
            # Extract text after "research questions were/are/is..."
            rq_section = text[split_pos : split_pos + 2000]  # Get next 2000 chars

            # Find all segments ending with ? by splitting on ?
            segments = rq_section.split("?")
            for segment in segments:
                if not segment.strip():
                    continue

                # Reconstruct with ?
                question_text = segment + "?"
                # Replace newlines with spaces
                question_text = question_text.replace("\n", " ").strip()

                # Remove leading markers (numbers, bullets, Q labels)
                question_text = re.sub(r"^[\s\-\*•]+", "", question_text)
                question_text = re.sub(r"^Q\d+[\.\):\s]+", "", question_text, flags=re.IGNORECASE)
                question_text = re.sub(r"^\d+[\.\):\s]+", "", question_text)
                question_text = question_text.strip()

                # Remove trailing list artifacts ("; and")
                question_text = re.sub(
                    r"[;\s]*and\s*$", "", question_text, flags=re.IGNORECASE
                ).strip()

                # Must have question words and be substantial (35+ chars filters survey questions)
                if any(
                    marker in question_text.lower()
                    for marker in ["how", "what", "why", "when", "where", "which", "who"]
                ):
                    if len(question_text) >= 35 and len(question_text) <= 500:
                        research_questions.append(question_text)

        # Pattern 3: Explicit question sentences in RQ section
        sentences = self._split_sentences(text)
        for sentence in sentences:
            if "?" not in sentence:
                continue

            sentence = sentence.strip()

            # Must contain question words
            if not any(
                marker in sentence.lower()
                for marker in ["how", "what", "why", "when", "where", "which", "who"]
            ):
                continue

            # Reasonable length
            if len(sentence) < 20 or len(sentence) > 400:
                continue

            # Filter out fragments and meta-references
            if sentence.lower().startswith("and "):
                continue
            if any(
                word in sentence.lower()
                for word in [
                    "alignment to research question",
                    "refers to",
                    "see table",
                    "see figure",
                ]
            ):
                continue

            # Must have at least 3 words
            if len(sentence.split()) < 3:
                continue

            research_questions.append(sentence)

        # Deduplicate by phrase while ignoring only leading question/list labels.
        unique_rqs = []
        seen_inquiry_keys = set()
        for inquiry in research_questions:
            deduplication_text = re.sub(
                r"^\s*(?:(?:rq|research\s+questions?|questions?)\s*\d+|\d+|"
                r"(?:research\s+questions?|questions?))\s*[:.)-]\s*",
                "",
                inquiry,
                flags=re.IGNORECASE,
            )
            deduplication_key = " ".join(deduplication_text.casefold().split()).strip(
                " \t\r\n.,;:!?"
            )
            if deduplication_key and deduplication_key not in seen_inquiry_keys:
                unique_rqs.append(inquiry)
                seen_inquiry_keys.add(deduplication_key)

        # Final cleanup: remove any that are just fragments
        cleaned_rqs = []
        for rq in unique_rqs:
            # Must contain actual content words
            words = [w for w in self._tokenise_words(rq) if len(w) > 3]
            if len(words) >= 3:  # At least 3 substantial words
                cleaned_rqs.append(rq)

        return cleaned_rqs[:10]  # Return top 10 after cleanup

    def _classify_research_inquiries(
        self,
        inquiries: List[str],
        sources_by_inquiry: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> Dict[str, str]:
        """Assign conservative inquiry types, optionally using the local LLM.

        Args:
            inquiries: List of research inquiries to classify

        Returns:
            Dictionary mapping each inquiry to its classified type
        """
        allowed_types = {
            "research_question",
            "sub_question",
            "aim",
            "objective",
            "hypothesis",
            "guiding_question",
            "not_rq",
        }
        profile_framings = (self.cultural_lens_profile or {}).get("research_question_framings", [])
        classification: Dict[str, str] = {}
        for inquiry in inquiries:
            inquiry_lower = inquiry.casefold()
            profile_type = next(
                (
                    framing.get("classify_as")
                    for framing in profile_framings
                    if framing.get("classify_as")
                    and any(
                        _contains_whole_phrase(inquiry, str(indicator))
                        for indicator in framing.get("indicators", [])
                        if indicator
                    )
                ),
                None,
            )
            if profile_type:
                classification[inquiry] = profile_type
            elif re.search(r"\b(objective|objectives)\b", inquiry_lower):
                classification[inquiry] = "objective"
            elif re.search(r"\b(aim|aims|purpose)\b", inquiry_lower):
                classification[inquiry] = "aim"
            elif re.search(r"\bhypothesis|hypotheses\b", inquiry_lower):
                classification[inquiry] = "hypothesis"
            else:
                classification[inquiry] = "research_question"

        if (
            not inquiries
            or not self.llm_client
            or not self._llm_enabled("research_inquiry_classification")
        ):
            return classification

        candidates = [
            {
                "index": index,
                "text": inquiry,
                "sections": [
                    source.get("section", "")
                    for source in (sources_by_inquiry or {}).get(inquiry, [])
                ],
                "heuristic_type": classification[inquiry],
            }
            for index, inquiry in enumerate(inquiries)
        ]
        prompt = (
            "Classify each thesis inquiry candidate. Allowed types: "
            "research_question, sub_question, aim, objective, hypothesis, guiding_question, not_rq. "
            "Classify the candidate's role in the thesis, not only its sentence form. Use not_rq "
            "for rhetorical questions, interview/survey prompts, quoted questions, or other text "
            "that does not express the thesis's own research purpose. Preserve the candidate text; return only JSON in the "
            'shape {"classifications": [{"index": 0, "type": "research_question"}]}. '
            "Include one entry per candidate and do not invent indexes.\n\n"
            f"Candidates:\n{json.dumps(candidates, ensure_ascii=False)}"
        )
        try:
            response = self._llm_invoke(prompt)
            payload = extract_first_json_block(response) if response else None
        except Exception:
            payload = None
        if not isinstance(payload, dict) or not isinstance(payload.get("classifications"), list):
            return classification

        for item in payload["classifications"]:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            inquiry_type = item.get("type")
            if type(index) is int and 0 <= index < len(inquiries) and inquiry_type in allowed_types:
                classification[inquiries[index]] = inquiry_type
        return classification

    def _reconcile_research_inquiries(
        self,
        inquiries: List[str],
        inquiry_types: Dict[str, str],
        sources_by_inquiry: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[
        List[str],
        Dict[str, str],
        Dict[str, List[Dict[str, Any]]],
        Dict[str, List[str]],
    ]:
        """Conservatively merge local-LLM-confirmed introduction/conclusion restatements.

        Args:
            inquiries: List of research inquiries
            inquiry_types: Dictionary mapping each inquiry to its classified type
            sources_by_inquiry: Dictionary mapping each inquiry to its source sections

        Returns:
            A tuple containing:
                - List of reconciled inquiries
                - Updated dictionary of inquiry types
                - Updated dictionary of sources by inquiry
                - Dictionary of aliases for merged inquiries
        """
        aliases: Dict[str, List[str]] = {}
        explicit_groups: Dict[Tuple[str, str], List[int]] = {}
        for index, inquiry in enumerate(inquiries):
            inquiry_type = inquiry_types.get(inquiry, "research_question")
            for source in sources_by_inquiry.get(inquiry, []):
                explicit_id = str(source.get("inquiry_id") or "").strip().upper()
                if explicit_id:
                    explicit_groups.setdefault((inquiry_type, explicit_id), []).append(index)
                    break

        explicit_canonical_by_member: Dict[int, int] = {}
        for members in explicit_groups.values():
            members = sorted(set(members))
            if len(members) < 2:
                continue
            introduction_members = [
                index
                for index in members
                if any(
                    "introduction" in str(source.get("section", "")).casefold()
                    for source in sources_by_inquiry.get(inquiries[index], [])
                )
            ]
            canonical_index = min(introduction_members or members)
            explicit_canonical_by_member.update({index: canonical_index for index in members})

        if explicit_canonical_by_member:
            consolidated_inquiries: List[str] = []
            consolidated_types: Dict[str, str] = {}
            consolidated_sources: Dict[str, List[Dict[str, Any]]] = {}
            for index, inquiry in enumerate(inquiries):
                canonical_index = explicit_canonical_by_member.get(index, index)
                if canonical_index != index:
                    continue
                canonical = inquiries[canonical_index]
                consolidated_inquiries.append(canonical)
                consolidated_types[canonical] = inquiry_types.get(canonical, "research_question")
                members = [
                    member_index
                    for member_index, member_canonical in explicit_canonical_by_member.items()
                    if member_canonical == canonical_index
                ] or [canonical_index]
                consolidated_sources[canonical] = []
                for member_index in members:
                    member_text = inquiries[member_index]
                    if member_index != canonical_index:
                        aliases.setdefault(canonical, []).append(member_text)
                    for source in sources_by_inquiry.get(member_text, []):
                        combined_source = dict(source)
                        if member_index != canonical_index:
                            combined_source["restatement"] = member_text
                        consolidated_sources[canonical].append(combined_source)
            inquiries = consolidated_inquiries
            inquiry_types = consolidated_types
            sources_by_inquiry = consolidated_sources

        if (
            len(inquiries) < 2
            or not self.llm_client
            or not self._llm_enabled("research_inquiry_reconciliation")
        ):
            return inquiries, inquiry_types, sources_by_inquiry, aliases

        prompt_items = [
            {
                "index": index,
                "type": inquiry_types.get(inquiry, "research_question"),
                "sections": [
                    source.get("section", "") for source in sources_by_inquiry.get(inquiry, [])
                ],
                "text": inquiry,
            }
            for index, inquiry in enumerate(inquiries)
        ]
        prompt = (
            "Identify only clearly equivalent restatements of the same research inquiry. "
            "Do not merge inquiries merely because they discuss the same topic. Do not merge "
            "different inquiry types or sub-questions. A merge is valid only when the group "
            "has evidence from both an Introduction and a Conclusion section. Return JSON only "
            'with shape {"groups": [[0, 2]]}; list zero-based indices for groups of equivalent inquiries. '
            "Omit singleton groups and return an empty groups list when uncertain.\n\n"
            f"Inquiries:\n{json.dumps(prompt_items, ensure_ascii=False)}"
        )

        try:
            response = self._llm_invoke(prompt)
            payload = extract_first_json_block(response) if response else None
        except Exception:
            payload = None
        if not isinstance(payload, dict) or not isinstance(payload.get("groups"), list):
            return inquiries, inquiry_types, sources_by_inquiry, aliases

        accepted_groups: List[List[int]] = []
        assigned_indices: set[int] = set()
        for raw_group in payload["groups"]:
            if not isinstance(raw_group, list) or len(raw_group) < 2:
                continue
            if any(
                type(index) is not int or not 0 <= index < len(inquiries) for index in raw_group
            ):
                continue
            group = sorted(set(raw_group))
            if len(group) < 2 or assigned_indices.intersection(group):
                continue
            if (
                len({inquiry_types.get(inquiries[index], "research_question") for index in group})
                != 1
            ):
                continue

            sections = [
                str(source.get("section", "")).casefold()
                for index in group
                for source in sources_by_inquiry.get(inquiries[index], [])
            ]
            has_introduction = any("introduction" in section for section in sections)
            has_conclusion = any("conclusion" in section for section in sections)
            if not (has_introduction and has_conclusion):
                continue

            accepted_groups.append(group)
            assigned_indices.update(group)

        if not accepted_groups:
            return inquiries, inquiry_types, sources_by_inquiry, aliases

        canonical_index_by_member: Dict[int, int] = {}
        for group in accepted_groups:
            if any(
                source.get("parent_inquiry_id")
                for index in group
                for source in sources_by_inquiry.get(inquiries[index], [])
            ):
                continue
            introduction_indices = [
                index
                for index in group
                if any(
                    "introduction" in str(source.get("section", "")).casefold()
                    for source in sources_by_inquiry.get(inquiries[index], [])
                )
            ]
            canonical_index = min(introduction_indices or group)
            canonical_index_by_member.update({index: canonical_index for index in group})

        canonical_inquiries: List[str] = []
        canonical_types: Dict[str, str] = {}
        canonical_sources: Dict[str, List[Dict[str, Any]]] = {}
        for index, inquiry in enumerate(inquiries):
            canonical_index = canonical_index_by_member.get(index, index)
            if canonical_index != index:
                continue

            canonical = inquiries[canonical_index]
            canonical_inquiries.append(canonical)
            canonical_types[canonical] = inquiry_types.get(canonical, "research_question")
            group = next(
                (group for group in accepted_groups if canonical_index in group), [canonical_index]
            )
            canonical_sources[canonical] = []
            for member_index in group:
                member_text = inquiries[member_index]
                if member_index != canonical_index:
                    aliases.setdefault(canonical, []).append(member_text)
                for source in sources_by_inquiry.get(member_text, []):
                    combined_source = dict(source)
                    if member_index != canonical_index:
                        combined_source["restatement"] = member_text
                    canonical_sources[canonical].append(combined_source)

        return canonical_inquiries, canonical_types, canonical_sources, aliases

    def _assign_research_inquiry_identifiers(
        self,
        inquiries: List[str],
        sources_by_inquiry: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Assign explicit RQ labels or stable source-order IDs to inquiries.
        Args:
            inquiries (List[str]): The list of inquiry phrases to assign identifiers to.
            sources_by_inquiry (Dict[str, List[Dict[str, Any]]]): A mapping from each inquiry phrase to its associated sources.

        Returns:
            Tuple[Dict[str, str], Dict[str, str]]: A tuple containing two dictionaries:
                - The first dictionary maps each inquiry to its assigned explicit or generated ID.
                - The second dictionary maps each inquiry to its parent inquiry ID, if available.
        """
        explicit_ids: Dict[str, str] = {}
        parent_ids: Dict[str, str] = {}
        label_pattern = re.compile(
            r"(?:\bRQ\s*|\bresearch\s+questions?\s*|\bquestions?\s*)"
            r"(\d+(?:[a-z]|\.\d+)*)\s*[:.)-]?\s*$",
            re.IGNORECASE,
        )

        for inquiry in inquiries:
            for source in sources_by_inquiry.get(inquiry, []):
                inquiry_id = source.get("inquiry_id")
                if inquiry_id:
                    explicit_ids[inquiry] = str(inquiry_id)
                    parent_id = source.get("parent_inquiry_id")
                    if parent_id:
                        parent_ids[inquiry] = str(parent_id)
                    break

        reserved_ids = set(explicit_ids.values())
        used_ids: set[str] = set()
        inquiry_ids: Dict[str, str] = {}
        next_number = 1
        for inquiry in inquiries:
            explicit_id = explicit_ids.get(inquiry)
            if explicit_id and explicit_id not in used_ids:
                inquiry_ids[inquiry] = explicit_id
                used_ids.add(explicit_id)
                continue

            while f"RQ{next_number}" in reserved_ids or f"RQ{next_number}" in used_ids:
                next_number += 1
            generated_id = f"RQ{next_number}"
            inquiry_ids[inquiry] = generated_id
            used_ids.add(generated_id)
            next_number += 1

        return inquiry_ids, parent_ids

    def _locate_research_inquiry_sources(
        self,
        chunks_data: Dict[str, Any],
        inquiries: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Map extracted inquiry phrases to exact local and absolute source spans.

        Args:
            chunks_data (Dict[str, Any]): The chunked document data containing texts, metadatas, and ids.
            inquiries (List[str]): The list of inquiry phrases to locate within the documents.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A mapping from each inquiry phrase to a list of source spans,
            each containing local and absolute start and end positions along with any relevant metadata.
        """
        sources_by_inquiry: Dict[str, List[Dict[str, Any]]] = {}
        documents = chunks_data.get("documents", [])
        metadatas = chunks_data.get("metadatas", [])
        chunk_ids = chunks_data.get("ids", [])
        if "metadatas" in chunks_data and len(metadatas) != len(documents):
            raise ValueError("metadata and document lengths must match for inquiry source mapping")
        if "ids" in chunks_data and len(chunk_ids) != len(documents):
            raise ValueError("chunk ID and document lengths must match for inquiry source mapping")
        for inquiry in inquiries:
            normalised_inquiry = " ".join(inquiry.casefold().split())
            matches = []
            for index, document in enumerate(documents):
                document_text = str(document)
                normalised_characters: List[str] = []
                source_indices: List[int] = []
                for source_index, character in enumerate(document_text):
                    if character.isspace():
                        if normalised_characters and normalised_characters[-1] != " ":
                            normalised_characters.append(" ")
                            source_indices.append(source_index)
                    else:
                        for folded_character in character.casefold():
                            normalised_characters.append(folded_character)
                            source_indices.append(source_index)
                if normalised_characters and normalised_characters[-1] == " ":
                    normalised_characters.pop()
                    source_indices.pop()
                normalised_document = "".join(normalised_characters)
                match_start = normalised_document.find(normalised_inquiry)
                if not normalised_inquiry or match_start < 0:
                    continue
                match_end = match_start + len(normalised_inquiry)
                local_start = source_indices[match_start]
                local_end = source_indices[match_end - 1] + 1
                metadata = metadatas[index] if index < len(metadatas) else {}
                chunk_source_start = metadata.get("source_start")
                chunk_source_end = metadata.get("source_end")
                absolute_start = (
                    chunk_source_start + local_start
                    if isinstance(chunk_source_start, int)
                    else None
                )
                absolute_end = (
                    chunk_source_start + local_end if isinstance(chunk_source_start, int) else None
                )
                prefix = document_text[max(0, local_start - 80) : local_start]
                label_match = re.search(
                    r"(?:\bRQ\s*|\bresearch\s+questions?\s*|\bquestions?\s*)"
                    r"(\d+(?:[a-z]|\.\d+)*)\s*[:.)-]?\s*$",
                    prefix,
                    re.IGNORECASE,
                )
                inquiry_id = None
                parent_inquiry_id = None
                if label_match:
                    number = label_match.group(1).casefold()
                    inquiry_id = f"RQ{number}"
                    subquestion_match = re.fullmatch(r"(\d+)(?:[a-z]|\.\d+)", number)
                    if subquestion_match:
                        parent_inquiry_id = f"RQ{subquestion_match.group(1)}"
                matches.append(
                    {
                        "chunk_id": chunk_ids[index] if index < len(chunk_ids) else None,
                        "section": metadata.get("heading_path")
                        or metadata.get("section_title")
                        or metadata.get("chapter")
                        or "Unclassified",
                        "source_start": absolute_start,
                        "source_end": absolute_end,
                        "chunk_local_start": local_start,
                        "chunk_local_end": local_end,
                        "chunk_source_start": chunk_source_start,
                        "chunk_source_end": chunk_source_end,
                        "inquiry_id": inquiry_id,
                        "parent_inquiry_id": parent_inquiry_id,
                    }
                )
            sources_by_inquiry[inquiry] = matches
        return sources_by_inquiry

    def _compute_rq_alignment(
        self,
        research_questions: List[str],
        chunks_data: Dict,
        aliases: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[float, List[str]]:
        """Check if research questions are addressed in findings/conclusion.

        Args:
            research_questions: List of extracted research questions
            chunks_data: ChromaDB query result with metadatas and documents to check for alignment
            aliases: Optional dictionary mapping research questions to lists of synonymous terms

        Returns:
            Tuple of (alignment_score, unaddressed_rqs) where alignment_score is the proportion of RQs that are addressed in the findings/conclusion sections, and unaddressed_rqs is a list of RQs that were not sufficiently addressed (truncated for display)
        """
        if not research_questions:
            return 1.0, []  # No RQs to check

        findings_text = self._collect_text_by_section(
            chunks_data,
            include_sections=["results", "findings", "discussion", "conclusion"],
        )

        findings_text_lower = findings_text.lower()
        addressed_count = 0
        unaddressed = []

        for rq in research_questions:
            # Extract key terms from RQ (nouns, verbs)
            related_inquiries = [rq, *((aliases or {}).get(rq, []))]
            key_terms = list(
                dict.fromkeys(
                    word.lower()
                    for related_inquiry in related_inquiries
                    for word in self._tokenise_words(related_inquiry)
                    if len(word) > 4
                    and word.lower()
                    not in {"research", "question", "hypothesis", "study", "thesis"}
                )
            )

            # Check if at least 40% of key terms appear in findings
            if key_terms:
                matches = sum(1 for term in key_terms if term in findings_text_lower)
                if matches / len(key_terms) >= 0.4:
                    addressed_count += 1
                else:
                    unaddressed.append(rq[:150])  # Truncate for display

        alignment_score = addressed_count / len(research_questions) if research_questions else 1.0
        return alignment_score, unaddressed

    def _track_concept_progression(
        self, chunks_data: Dict, chapters: List[Dict]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Track key concepts across chapter progression using metadata.

        Args:
            chunks_data: ChromaDB query result
            chapters: Ordered list of chapter dicts (from _extract_chapters)

        Returns:
            Tuple of (concept_tracking, orphaned_concepts)
        """
        import re
        from collections import Counter

        # Use provided chapters (already in document order)
        if len(chapters) < 2:
            return [], []  # Not enough sections

        # Identify key concepts using a simple frequency approach
        all_text = " ".join(chunks_data.get("documents", []))
        words = self._tokenise_words(all_text)
        word_freq = Counter(
            w.lower() for w in words if len(w) > 4 and any(ch.isalpha() for ch in w)
        )

        # Get top concepts (using NLTK + domain stopwords from terminology module)
        stem_freq: Counter[str] = Counter()
        variant_counts: Dict[str, Counter] = defaultdict(Counter)
        for word, count in word_freq.items():
            # Use NLTK Porter stemmer for robust concept normalisation
            stem = _STEMMER.preprocess_token(word)
            if not stem or stem in _STOPWORDS or word in _STOPWORDS:
                continue
            stem_freq[stem] += count
            variant_counts[stem][word] += count

        top_concepts = [stem for stem, count in stem_freq.most_common(50) if count >= 2][:20]

        # Group text by chapter
        chapter_text_map = self._group_text_by_chapter(chunks_data)
        # Use chapter order from provided chapters list
        ordered_chapter_names = [chapter["name"] for chapter in chapters]

        # Track where each concept appears
        concept_tracking = []
        orphaned_concepts = []

        for concept in top_concepts:
            variants = list(variant_counts[concept].keys())
            display_concept = variant_counts[concept].most_common(1)[0][0]
            override = CAPITALISATION_OVERRIDES.get(display_concept.lower())
            if override:
                display_concept = override
            pattern = re.compile(r"\b(" + "|".join(re.escape(v) for v in variants) + r")\b")
            appearances = []
            for chapter_name in ordered_chapter_names:
                chapter_text = chapter_text_map.get(chapter_name, "")
                if chapter_text and pattern.search(chapter_text.lower()):
                    appearances.append(chapter_name)

            if len(appearances) == 0:
                continue
            elif len(appearances) == 1:
                orphaned_concepts.append(display_concept)
            else:
                # Classify progression
                intro_section = appearances[0]
                developed_sections = appearances[1:-1] if len(appearances) > 2 else []
                concluded_section = appearances[-1]

                concept_tracking.append(
                    {
                        "concept": display_concept,
                        "intro_section": intro_section,
                        "developed_sections": developed_sections,
                        "concluded_section": concluded_section,
                        "total_appearances": len(appearances),
                    }
                )

        return concept_tracking, orphaned_concepts

    def _generate_summary(
        self,
        structure: StructureAnalysis,
        citations: CitationPatternAnalysis,
        claims: ClaimAnalysis,
        methodology: MethodologyChecklist,
        writing: WritingQualityMetrics,
        alignment: ContributionAlignment,
        critical_flags: List[RedFlag],
        persona: str,
    ) -> str:
        """Generate human-readable summary.

        Args:
            structure: StructureAnalysis results
            citations: CitationPatternAnalysis results
            claims: ClaimAnalysis results
            methodology: MethodologyChecklist results
            writing: WritingQualityMetrics results
            alignment: ContributionAlignment results
            critical_flags: List of critical RedFlags to highlight
            persona: Persona type to tailor summary (e.g., 'supervisor', 'assessor', 'researcher')

        Returns:
            Concise summary string with emojis and key findings tailored to the persona
        """
        summary_parts = []

        # Structure summary
        if structure.missing_sections:
            summary_parts.append(f"❌ Missing {len(structure.missing_sections)} required sections")
        else:
            summary_parts.append("✓ All required sections present")

        # Coherence summary
        if structure.avg_coherence >= 0.7:
            summary_parts.append(
                f"✓ Strong chapter flow (coherence: {structure.avg_coherence:.2f})"
            )
        elif structure.avg_coherence >= 0.5:
            summary_parts.append(
                f"⚠ Moderate chapter flow (coherence: {structure.avg_coherence:.2f})"
            )
        else:
            summary_parts.append(f"❌ Weak chapter flow (coherence: {structure.avg_coherence:.2f})")

        # RQ alignment summary
        if structure.research_questions:
            if structure.rq_alignment_score >= 0.8:
                summary_parts.append(
                    f"✓ Research questions well addressed ({structure.rq_alignment_score*100:.0f}%)"
                )
            elif structure.rq_alignment_score >= 0.5:
                summary_parts.append(
                    f"⚠ Some RQs may need more attention ({structure.rq_alignment_score*100:.0f}%)"
                )
            else:
                summary_parts.append(
                    f"❌ {len(structure.unaddressed_rqs)} research questions not adequately addressed"
                )

        # Concept progression summary
        if structure.orphaned_concepts:
            summary_parts.append(
                f"⚠ {len(structure.orphaned_concepts)} key concepts lack development across chapters"
            )

        # Citation summary
        if citations.citation_recency_score >= 0.5:
            summary_parts.append(
                f"✓ Good citation recency ({citations.citation_recency_score*100:.0f}% recent)"
            )
        else:
            summary_parts.append(
                f"⚠ Stale citations ({citations.citation_recency_score*100:.0f}% recent)"
            )

        if citations.orphaned_claims:
            summary_parts.append(
                f"❌ {len(citations.orphaned_claims)} sections with unsupported claims"
            )

        # Methodology summary
        if methodology.missing_items:
            summary_parts.append(
                f"⚠ Methodology gaps: {len(methodology.missing_items)} items missing"
            )

        # Claims summary
        if claims.contradictions:
            summary_parts.append(f"⚠ {len(claims.contradictions)} potential contradictions")

        # Alignment summary
        if (
            alignment.overlap_score < 0.2
            and alignment.contribution_keywords
            and alignment.finding_keywords
        ):
            summary_parts.append("⚠ Weak contribution-finding alignment")

        # Critical flags
        if critical_flags:
            summary_parts.append(f"🚨 {len(critical_flags)} CRITICAL issues require attention")

        return " | ".join(summary_parts)

    def _generate_next_steps(self, red_flags: List[RedFlag], persona: str) -> List[str]:
        """Generate actionable next steps based on findings.

        Args:
            red_flags: List of RedFlags identified in the analysis
            persona: Persona type to tailor recommendations (e.g., 'supervisor', 'assessor', 'researcher')

        Returns:
            List of recommended next steps, prioritised by severity and tailored to the persona
        """
        steps = []

        # Group flags by severity
        critical = [f for f in red_flags if f.severity == "critical"]
        warnings = [f for f in red_flags if f.severity == "warning"]

        if critical:
            steps.append(f"[CRITICAL] Address {len(critical)} critical issues immediately")
            for flag in critical[:3]:  # Top 3
                if flag.suggestion:
                    steps.append(f"  → {flag.suggestion}")

        if warnings:
            steps.append(f"[WARNING] Review {len(warnings)} warning-level items")
            for flag in warnings[:2]:  # Top 2
                if flag.suggestion:
                    steps.append(f"  → {flag.suggestion}")

        # Persona-specific recommendations
        if persona == "supervisor":
            steps.append("[SUPERVISOR] Review chapter transitions for narrative coherence")
        elif persona == "assessor":
            steps.append("[ASSESSOR] Verify all claims are adequately supported by citations")
        elif persona == "researcher":
            steps.append("[RESEARCHER] Check citation diversity and recent literature coverage")

        return steps
