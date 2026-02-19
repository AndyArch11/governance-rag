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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scripts.ingest.academic.terminology import get_all_stopwords
from scripts.ingest.vectors import EXPECTED_EMBEDDING_DIM
from scripts.search.text_preprocessing import PreprocessingStrategy, TextPreprocessor
from scripts.utils.json_utils import extract_first_json_block

# Cache stopwords at module level for efficiency
_STOPWORDS = get_all_stopwords()

# Cache stemmer for concept normalisation
_STEMMER = TextPreprocessor(
    strategy=PreprocessingStrategy.STEM_PORTER,
    remove_stopwords=False,  # We handle stopwords separately
    min_token_length=1,
)


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
    research_questions: List[str]  # Extracted research questions
    rq_alignment_score: float  # 0-1, how well findings address RQs
    unaddressed_rqs: List[str]  # RQs not mentioned in findings/conclusion
    key_concepts: List[
        Dict[str, Any]
    ]  # Concept tracking: {concept, intro_section, developed_sections, concluded_section}
    orphaned_concepts: List[str]  # Concepts introduced but not developed/concluded
    red_flags: List[RedFlag]


@dataclass
class CitationPatternAnalysis:
    """Results from citation pattern analysis."""

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


@dataclass
class AssessmentReport:
    """Complete PhD quality assessment report."""

    doc_id: str
    assessed_at: datetime
    persona: str  # 'supervisor', 'assessor', 'researcher'
    overall_score: float  # 0-1
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
    ):
        """
        Initialise assessor.

        Args:
            chunk_collection: ChromaDB collection with thesis chunks
            llm_client: Optional LLM client for claim extraction (Phase 2)
            llm_flags: Dict to enable/disable specific LLM features
            citation_db_path: Path to citation graph SQLite database
        """
        self.chunk_collection = chunk_collection
        self.llm_client = llm_client
        self.llm_flags = llm_flags or {}

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
            assessed_at=datetime.utcnow(),
            persona=persona,
            overall_score=overall_score,
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
            critical_red_flags=critical_flags,
            summary=summary,
            next_steps=next_steps,
        )

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

        # Extract research questions and check alignment
        research_questions = self._extract_research_questions(chunks_data)
        rq_alignment_score, unaddressed_rqs = self._compute_rq_alignment(
            research_questions, chunks_data
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

        # Track concept progression (using all chapters for concept tracking)
        key_concepts, orphaned_concepts = self._track_concept_progression(chunks_data, all_chapters)

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
            avg_coherence=avg_coherence,
            chapter_order=[chapter["name"] for chapter in all_chapters],
            research_questions=research_questions,
            rq_alignment_score=rq_alignment_score,
            unaddressed_rqs=unaddressed_rqs,
            key_concepts=key_concepts,
            orphaned_concepts=orphaned_concepts,
            red_flags=red_flags,
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

        # No citations found
        if len(citations) == 0:
            red_flags.append(
                RedFlag(
                    severity="critical",
                    category="citations",
                    title="No Citations Found",
                    description="Citation graph database returned no citations for this document.",
                    suggestion="Ensure citations have been extracted during ingestion.",
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
            snippets = []
            location_counts = {}
            doc_snippet_added: set[int] = set()

            for i in scope_indices:
                if i >= len(documents):
                    continue
                doc = documents[i]
                if self._is_toc_chunk(doc):
                    continue
                section_label = section_labels_by_index.get(i)
                if not section_label:
                    continue
                location_counts[section_label] = location_counts.get(section_label, 0) + 0

                for match in pattern.finditer(doc):
                    count += 1
                    location_counts[section_label] = location_counts.get(section_label, 0) + 1
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
            # Fall back to non-ToC chunks if parent metadata is missing section labels
            return non_toc_indices

        return non_toc_indices

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

            for section_name in [
                "Introduction",
                "Methodology",
                "Results",
                "Discussion",
                "Conclusion",
                "References",
                "Appendix",
                "Abstract",
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
        chapters_dict = defaultdict(
            lambda: {"chunks": [], "embeddings": [], "sequence_number": float("inf")}
        )
        ordered_labels: List[str] = []

        indices = self._select_structural_indices(chunks_data)
        documents = chunks_data.get("documents", [])
        metadatas = chunks_data.get("metadatas", [])

        toc_order = self._build_toc_order(chunks_data)

        # First pass: extract all chapters and identify numbered chapter boundaries
        # Track original insertion order for stable sorting when sequence numbers are identical
        insertion_order = {}
        for i in indices:
            if i >= len(metadatas) or i >= len(documents):
                continue
            meta = metadatas[i]
            doc = documents[i]

            chapter_name = self._get_chapter_label(meta, doc)
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
                chapters_dict[canonical_name]["embeddings"].append(chunks_data["embeddings"][i])

        # Assign effective sequence numbers (use ToC order as fallback for missing sequence numbers)
        for label in ordered_labels:
            seq_num = chapters_dict[label]["sequence_number"]
            if toc_order and label in toc_order:
                chapters_dict[label]["effective_sequence"] = toc_order[label]
            elif seq_num != float("inf"):
                chapters_dict[label]["effective_sequence"] = seq_num
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

        # Sort chapters by effective sequence number (with ToC fallback)
        # Use insertion order as secondary key for stable sorting when sequence numbers are identical
        def _chapter_sort_key(label: str) -> Tuple[int, float, int]:
            if toc_order and label in toc_order:
                return (0, float(toc_order[label]), insertion_order.get(label, 0))
            seq_value = chapters_dict[label]["sequence_number"]
            if isinstance(seq_value, (int, float)) and seq_value != float("inf"):
                return (1, float(seq_value), insertion_order.get(label, 0))
            return (2, float("inf"), insertion_order.get(label, 0))

        ordered_labels_sorted = sorted(ordered_labels, key=_chapter_sort_key)

        # Second pass: classify and build final chapter list
        chapters = []
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

        citations = []

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
                    cursor.execute(
                        """
                        SELECT DISTINCT n.node_id, n.title, n.authors, n.year, n.doi, 
                               n.reference_type, n.source
                        FROM nodes n
                        INNER JOIN edges e ON n.node_id = e.target
                        WHERE e.source = (SELECT node_id FROM nodes WHERE node_type = 'document' LIMIT 1)
                          AND n.node_type = 'reference'
                    """
                    )
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

        venue_counts = defaultdict(int)

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
            if claim_re.search(doc):
                # Check if it also contains citations
                if not citation_re.search(doc):
                    # Extract snippet around claim
                    match = claim_re.search(doc)
                    start = match.start() - 140
                    end = match.end() + 140
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
        chapter_sizes = defaultdict(int)

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
        contradictions = []
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
            return str(self.llm_client(prompt))
        if hasattr(self.llm_client, "invoke"):
            return str(self.llm_client.invoke(prompt))
        return ""

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
        freqs = defaultdict(int)
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

        indices = self._select_structural_indices(chunks_data)
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
        indices = self._select_structural_indices(chunks_data)
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

        return {label: "\n".join(texts) for label, texts in chapter_text.items()}

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
        Compute weighted overall quality score.

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
            Overall quality score (0 to 1) based on weighted combination of components
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

        # Citation score
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

        overall = (
            w["structure"] * structure_score
            + w["citations"] * citation_score
            + w["methodology"] * methodology_score
            + w["writing"] * writing_score
            + w["alignment"] * alignment_score
        )

        return overall

    def _extract_research_questions(self, chunks_data: Dict) -> List[str]:
        """Extract research questions from introduction/abstract sections.

        Args:
            chunks_data: ChromaDB query result with metadatas and documents

        Returns:
            List of extracted research questions (up to 10)
        """
        import re

        text = self._collect_text_by_section(
            chunks_data,
            include_sections=["abstract", "introduction", "research questions"],
        )

        research_questions = []

        # Pattern 1: "RQ1:", "RQ 1:", "Research Question(s) 1:", "Question(s) 1:" followed by actual question text
        # Handles singular/plural and optional numbering
        pattern1 = (
            r"(?:RQ\s*\d+|Research\s+Questions?\s*\d*|Questions?\s*\d+)\s*[:.]?\s*([^?.!\n]+[?.!])"
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

        # Deduplicate and limit
        unique_rqs = list(dict.fromkeys(research_questions))[:15]  # Increased to 15 to capture more

        # Final cleanup: remove any that are just fragments
        cleaned_rqs = []
        for rq in unique_rqs:
            # Must contain actual content words
            words = [w for w in self._tokenise_words(rq) if len(w) > 3]
            if len(words) >= 3:  # At least 3 substantial words
                cleaned_rqs.append(rq)

        return cleaned_rqs[:10]  # Return top 10 after cleanup

    def _compute_rq_alignment(
        self, research_questions: List[str], chunks_data: Dict
    ) -> Tuple[float, List[str]]:
        """Check if research questions are addressed in findings/conclusion.

        Args:
            research_questions: List of extracted research questions
            chunks_data: ChromaDB query result with metadatas and documents to check for alignment

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
            key_terms = [
                word.lower()
                for word in self._tokenise_words(rq)
                if len(word) > 4
                and word.lower() not in {"research", "question", "hypothesis", "study", "thesis"}
            ]

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
        stem_freq = Counter()
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
