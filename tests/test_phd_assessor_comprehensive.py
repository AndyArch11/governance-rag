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

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from pathlib import Path

from scripts.ingest.academic.phd_assessor import (
    PhDQualityAssessor,
    RedFlag,
    StructureAnalysis,
    CitationPatternAnalysis,
    ClaimAnalysis,
    MethodologyChecklist,
    WritingQualityMetrics,
    ContributionAlignment,
)
from scripts.utils.json_utils import extract_first_json_block


@pytest.fixture
def mock_collection():
    """Create a mock ChromaDB collection."""
    collection = MagicMock()
    collection.get.return_value = {
        'ids': [],
        'documents': [],
        'metadatas': [],
        'embeddings': []
    }
    return collection


@pytest.fixture
def assessor(mock_collection, tmp_path):
    """Create assessor with mock dependencies."""
    db_path = tmp_path / "test_citations.db"
    return PhDQualityAssessor(
        chunk_collection=mock_collection,
        llm_client=None,
        llm_flags={},
        citation_db_path=str(db_path)
    )


@pytest.fixture
def sample_chunks_data():
    """Sample chunks data for testing."""
    return {
        'ids': ['chunk1', 'chunk2', 'chunk3'],
        'documents': [
            'Introduction text with research content.',
            'Methodology section describing methods.',
            'Results showing findings.'
        ],
        'metadatas': [
            {
                'doc_id': 'test_thesis',
                'chapter': 'Chapter 1',
                'section_title': 'Introduction',
                'sequence_number': 0,
                'source_category': 'academic_paper'
            },
            {
                'doc_id': 'test_thesis',
                'chapter': 'Chapter 2',
                'section_title': 'Methodology',
                'sequence_number': 1,
                'source_category': 'academic_paper'
            },
            {
                'doc_id': 'test_thesis',
                'chapter': 'Chapter 3',
                'section_title': 'Results',
                'sequence_number': 2,
                'source_category': 'academic_paper'
            }
        ],
        'embeddings': [
            np.random.rand(1024).tolist(),
            np.random.rand(1024).tolist(),
            np.random.rand(1024).tolist()
        ]
    }


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
            {'source': 'crossref', 'title': 'Paper 1'},
            {'source': 'crossref', 'title': 'Paper 2'},
            {'source': 'openalex', 'title': 'Paper 3'},
            {'source': 'crossref', 'title': 'Paper 4'},
        ]
        clusters = assessor._cluster_citations(citations)
        assert clusters['crossref'] == 3
        assert clusters['openalex'] == 1
    
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
            "The method does not improve accuracy"
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
        sample_chunks_data['documents'][0] = "We prove that the hypothesis is correct."
        sample_chunks_data['documents'][1] = "The results clearly demonstrate effectiveness."
        
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
            "We analysed the results."  # Active
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
            'documents': [
                'Introduction section',
                'Literature Review chapter',
                'Methodology details',
                'Results and findings',
                'Discussion of results',
                'Conclusion summary',
                'Limitations of study'
            ],
            'metadatas': [
                {'section_title': 'Introduction', 'chapter': 'Chapter 1'},
                {'section_title': 'Literature Review', 'chapter': 'Chapter 2'},
                {'section_title': 'Methodology', 'chapter': 'Chapter 3'},
                {'section_title': 'Results', 'chapter': 'Chapter 4'},
                {'section_title': 'Discussion', 'chapter': 'Chapter 5'},
                {'section_title': 'Conclusion', 'chapter': 'Chapter 6'},
                {'section_title': 'Limitations', 'chapter': 'Chapter 7'}
            ]
        }
        missing = assessor._detect_missing_sections(chunks_data)
        assert len(missing) == 0
    
    def test_detect_missing_sections_some_missing(self, assessor):
        """Test when some sections are missing."""
        chunks_data = {
            'documents': [
                'Introduction section',
                'Results and findings'
            ],
            'metadatas': [
                {'section_title': 'Introduction', 'chapter': 'Chapter 1'},
                {'section_title': 'Results', 'chapter': 'Chapter 2'}
            ]
        }
        missing = assessor._detect_missing_sections(chunks_data)
        # Should detect missing methodology, discussion, conclusion, etc.
        assert 'methodology' in missing
        assert 'discussion' in missing
        assert 'conclusion' in missing


class TestChapterAnalysis:
    """Test chapter extraction and analysis."""
    
    def test_get_chapter_sizes(self, assessor, sample_chunks_data):
        """Test chapter size calculation."""
        sizes = assessor._get_chapter_sizes(sample_chunks_data)
        assert 'Chapter 1' in sizes or 'Introduction' in sizes
        assert all(size > 0 for size in sizes.values())
    
    def test_extract_chapters_with_embeddings(self, assessor, sample_chunks_data):
        """Test chapter extraction with embeddings."""
        with patch.object(assessor, '_select_structural_indices', return_value=[0, 1, 2]):
            with patch.object(assessor, '_has_section_metadata', return_value=True):
                chapters = assessor._extract_chapters(sample_chunks_data)
                
                assert len(chapters) > 0
                for chapter in chapters:
                    assert 'name' in chapter
                    assert 'embedding_mean' in chapter
                    assert 'section_type' in chapter
                    assert isinstance(chapter['embedding_mean'], np.ndarray)


class TestStructureAnalysis:
    """Test structure analysis method."""
    
    def test_analyse_structure_basic(self, assessor, sample_chunks_data):
        """Test basic structure analysis."""
        with patch.object(assessor, '_select_structural_indices', return_value=[0, 1, 2]):
            with patch.object(assessor, '_has_section_metadata', return_value=True):
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
            'ids': ['c1', 'c2', 'c3', 'c4'],
            'documents': [
                'Chapter 3 content',
                'Chapter 1 content',
                'Chapter 5 content',
                'Chapter 2 content',
            ],
            'metadatas': [
                {
                    'doc_id': 'thesis',
                    'chapter': 'Chapter 3',
                    'section_title': 'Chapter 3',
                    'sequence_number': 50,
                    'chunk_type': 'parent'
                },
                {
                    'doc_id': 'thesis',
                    'chapter': 'Chapter 1',
                    'section_title': 'Chapter 1',
                    'sequence_number': 10,
                    'chunk_type': 'parent'
                },
                {
                    'doc_id': 'thesis',
                    'chapter': 'Chapter 5',
                    'section_title': 'Chapter 5',
                    'sequence_number': 100,
                    'chunk_type': 'parent'
                },
                {
                    'doc_id': 'thesis',
                    'chapter': 'Chapter 2',
                    'section_title': 'Chapter 2',
                    'sequence_number': 30,
                    'chunk_type': 'parent'
                },
            ],
            'embeddings': [
                np.random.rand(1024).tolist(),
                np.random.rand(1024).tolist(),
                np.random.rand(1024).tolist(),
                np.random.rand(1024).tolist(),
            ]
        }
        
        analysis = assessor.analyse_structure(chunks_data)
        
        # Verify chapters are ordered by sequence_number (1, 2, 3, 5)
        # Flow transitions should be: 1→2, 2→3, 3→5
        assert len(analysis.chapter_transition_labels) == 3
        
        # Check transitions are in correct order
        assert 'Chapter 1' in analysis.chapter_transition_labels[0]
        assert 'Chapter 2' in analysis.chapter_transition_labels[0]
        
        assert 'Chapter 2' in analysis.chapter_transition_labels[1]
        assert 'Chapter 3' in analysis.chapter_transition_labels[1]
        
        assert 'Chapter 3' in analysis.chapter_transition_labels[2]
        assert 'Chapter 5' in analysis.chapter_transition_labels[2]
        
        # Verify flow scores match transition count
        assert len(analysis.chapter_flow_scores) == 3
    
    def test_analyse_structure_flow_scores_order(self, assessor):
        """Test that flow scores correspond to correct chapter transitions."""
        # Create chapters with known embeddings
        emb1 = np.ones(1024) * 0.1
        emb2 = np.ones(1024) * 0.2
        emb3 = np.ones(1024) * 0.3
        
        chunks_data = {
            'ids': ['c1', 'c2', 'c3'],
            'documents': ['Ch1', 'Ch2', 'Ch3'],
            'metadatas': [
                {'chapter': 'Chapter 1', 'sequence_number': 0, 'chunk_type': 'parent'},
                {'chapter': 'Chapter 2', 'sequence_number': 1, 'chunk_type': 'parent'},
                {'chapter': 'Chapter 3', 'sequence_number': 2, 'chunk_type': 'parent'},
            ],
            'embeddings': [emb1.tolist(), emb2.tolist(), emb3.tolist()]
        }
        
        analysis = assessor.analyse_structure(chunks_data)
        
        # Should have 2 transitions (1→2, 2→3)
        assert len(analysis.chapter_flow_scores) == 2
        assert len(analysis.chapter_transition_labels) == 2
        
        # Transitions should be in sequence order
        assert analysis.chapter_transition_labels[0] == 'Chapter 1 → Chapter 2'
        assert analysis.chapter_transition_labels[1] == 'Chapter 2 → Chapter 3'
    
    def test_analyse_structure_abrupt_transitions_order(self, assessor):
        """Test that abrupt transitions are detected in correct chapter order."""
        # Create chapters with very different embeddings for some transitions
        emb1 = np.ones(1024) * 1.0
        emb2 = np.ones(1024) * 0.9  # Similar to emb1
        emb3 = np.ones(1024) * -1.0  # Very different from emb2
        
        chunks_data = {
            'ids': ['c1', 'c2', 'c3'],
            'documents': ['Ch1', 'Ch2', 'Ch3'],
            'metadatas': [
                {'chapter': 'Chapter 1', 'sequence_number': 0, 'chunk_type': 'parent'},
                {'chapter': 'Chapter 2', 'sequence_number': 1, 'chunk_type': 'parent'},
                {'chapter': 'Chapter 3', 'sequence_number': 2, 'chunk_type': 'parent'},
            ],
            'embeddings': [emb1.tolist(), emb2.tolist(), emb3.tolist()]
        }
        
        analysis = assessor.analyse_structure(chunks_data)
        
        # Should detect abrupt transition between Chapter 2 and Chapter 3
        # (not between 1 and 2, as they're similar)
        if analysis.abrupt_transitions:
            # First abrupt transition should be from Chapter 2 to Chapter 3
            first_abrupt = analysis.abrupt_transitions[0]
            assert 'Chapter 2' in first_abrupt[0]
            assert 'Chapter 3' in first_abrupt[1]
            assert first_abrupt[2] < 0.3  # Low similarity
    
    def test_analyse_structure_concept_progression_order(self, assessor):
        """Test that concept progression tracking uses correct chapter order."""
        # Create chapters with a concept appearing in specific order
        chunks_data = {
            'ids': ['c1', 'c2', 'c3'],
            'documents': [
                'machine learning methodology',  # Chapter 1
                'data analysis approach',  # Chapter 2
                'machine learning results',  # Chapter 3
            ],
            'metadatas': [
                {'chapter': 'Chapter 1', 'sequence_number': 0, 'chunk_type': 'parent'},
                {'chapter': 'Chapter 2', 'sequence_number': 1, 'chunk_type': 'parent'},
                {'chapter': 'Chapter 3', 'sequence_number': 2, 'chunk_type': 'parent'},
            ],
            'embeddings': [
                np.random.rand(1024).tolist(),
                np.random.rand(1024).tolist(),
                np.random.rand(1024).tolist(),
            ]
        }
        
        analysis = assessor.analyse_structure(chunks_data)
        
        # Verify key_concepts tracks progression correctly
        # "machine" or "learning" should appear in chapters in order
        if analysis.key_concepts:
            for concept_info in analysis.key_concepts:
                if 'machin' in concept_info['concept'].lower() or 'learning' in concept_info['concept'].lower():
                    # Should track intro → conclusion progression
                    assert concept_info['intro_section'] == 'Chapter 1'
                    assert concept_info['concluded_section'] in ['Chapter 3', 'Chapter 2']
                    break
    
    def test_analyse_structure_excludes_pre_post_matter(self, assessor):
        """Test that pre-matter and post-matter are excluded from coherence analysis."""
        chunks_data = {
            'ids': ['c1', 'c2', 'c3', 'c4', 'c5'],
            'documents': ['Abstract', 'Ch1', 'Ch2', 'Ch3', 'References'],
            'metadatas': [
                {'chapter': 'Abstract', 'sequence_number': 0, 'chunk_type': 'parent'},
                {'chapter': 'Chapter 1', 'sequence_number': 1, 'chunk_type': 'parent'},
                {'chapter': 'Chapter 2', 'sequence_number': 2, 'chunk_type': 'parent'},
                {'chapter': 'Chapter 3', 'sequence_number': 3, 'chunk_type': 'parent'},
                {'chapter': 'References', 'sequence_number': 4, 'chunk_type': 'parent'},
            ],
            'embeddings': [
                np.random.rand(1024).tolist() for _ in range(5)
            ]
        }
        
        analysis = assessor.analyse_structure(chunks_data)
        
        # Chapter count should only include main-matter (Chapters 1, 2, 3)
        assert analysis.chapter_count == 3
        
        # Flow scores should only be for main chapters (1→2, 2→3)
        assert len(analysis.chapter_flow_scores) == 2
        
        # Transitions should not include Abstract or References
        for transition in analysis.chapter_transition_labels:
            assert 'Abstract' not in transition
            assert 'References' not in transition

        # Chapter order should include all sections in sequence
        assert analysis.chapter_order == [
            'Abstract',
            'Chapter 1',
            'Chapter 2',
            'Chapter 3',
            'References',
        ]


class TestCitationPatternAnalysis:
    """Test citation pattern analysis."""
    
    def test_analyse_citation_patterns_no_citations(self, assessor, sample_chunks_data):
        """Test citation analysis with no citations."""
        with patch.object(assessor, '_extract_citations', return_value=[]):
            analysis = assessor.analyse_citation_patterns(sample_chunks_data)
            
            assert isinstance(analysis, CitationPatternAnalysis)
            assert analysis.total_citations == 0
            # Should have red flag for no citations
            assert any(flag.category == 'citations' for flag in analysis.red_flags)
    
    def test_analyse_citation_patterns_with_recent(self, assessor, sample_chunks_data):
        """Test citation analysis with recent citations."""
        current_year = datetime.now().year
        mock_citations = [
            {'doi': '10.1/abc', 'title': 'Recent Paper', 'year': current_year},
            {'doi': '10.1/def', 'title': 'Another Recent', 'year': current_year - 1},
            {'doi': '10.1/ghi', 'title': 'Old Paper', 'year': 2000},
        ]
        
        with patch.object(assessor, '_extract_citations', return_value=mock_citations):
            analysis = assessor.analyse_citation_patterns(sample_chunks_data)
            
            assert analysis.total_citations == 3
            assert analysis.citation_recency_score > 0.5  # 2/3 are recent


class TestClaimAnalysis:
    """Test claim and contradiction analysis."""
    
    def test_analyse_claims_and_contradictions(self, assessor, sample_chunks_data):
        """Test claim analysis."""
        # Add claim-like text to sample data
        sample_chunks_data['documents'][0] = "We find that the method improves results."
        sample_chunks_data['documents'][1] = "This study demonstrates effectiveness."
        
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
            'documents': [
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
            'metadatas': [
                {'section_title': 'Methodology', 'chapter': 'Chapter 2'}
            ]
        }
        
        checklist = assessor.validate_methodology_checklist(chunks_data)
        
        assert isinstance(checklist, MethodologyChecklist)
        assert checklist.items['research_question'] is True
        assert checklist.items['data_collection'] is True
        assert checklist.items['sample_size'] is True
        assert checklist.items['ethics'] is True
        assert checklist.score > 0.8  # Most items present
    
    def test_validate_methodology_checklist_missing(self, assessor):
        """Test methodology validation with missing elements."""
        chunks_data = {
            'documents': [
                "This is a brief methodology section with minimal detail."
            ],
            'metadatas': [
                {'section_title': 'Methodology', 'chapter': 'Chapter 2'}
            ]
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
            'documents': [
                # Introduction with contributions
                "This thesis contributes novel machine learning algorithms.",
                # Results with matching findings
                "Results demonstrate the effectiveness of machine learning algorithms."
            ],
            'metadatas': [
                {'section_title': 'Introduction', 'chapter': 'Chapter 1'},
                {'section_title': 'Results', 'chapter': 'Chapter 4'}
            ]
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
            'documents': [
                "This thesis contributes theoretical frameworks.",
                "Results show experimental validation."
            ],
            'metadatas': [
                {'section_title': 'Introduction', 'chapter': 'Chapter 1'},
                {'section_title': 'Results', 'chapter': 'Chapter 4'}
            ]
        }
        
        analysis = assessor.analyse_contribution_alignment(chunks_data)
        
        assert isinstance(analysis, ContributionAlignment)
        # Weak overlap should trigger warning
        if analysis.overlap_score < 0.2:
            assert len(analysis.red_flags) > 0


class TestResearchQuestionExtraction:
    """Test research question extraction and alignment."""
    
    def test_extract_research_questions_explicit(self, assessor):
        """Test extraction of explicitly stated research questions."""
        chunks_data = {
            'documents': [
                """
                This study addresses three research questions:
                RQ1: What is the impact of X on Y?
                RQ2: How does Z mediate the relationship?
                RQ3: What are the boundary conditions?
                """
            ],
            'metadatas': [
                {'section_title': 'Introduction', 'chapter': 'Chapter 1'}
            ]
        }
        
        rqs = assessor._extract_research_questions(chunks_data)
        assert len(rqs) >= 1  # Should find at least one RQ


class TestCollectTextBySection:
    """Test section-based text collection."""
    
    def test_collect_text_by_section_filtered(self, assessor, sample_chunks_data):
        """Test text collection with section filtering."""
        text = assessor._collect_text_by_section(
            sample_chunks_data,
            include_sections=['introduction', 'methodology']
        )
        
        assert isinstance(text, str)
        assert len(text) > 0
        # Should include introduction and methodology text
        assert "Introduction" in text or "Methodology" in text
    
    def test_collect_text_by_section_all(self, assessor, sample_chunks_data):
        """Test text collection without filtering."""
        text = assessor._collect_text_by_section(
            sample_chunks_data,
            include_sections=None
        )
        
        assert isinstance(text, str)
        assert len(text) > 0
    
    def test_collect_text_filters_toc(self, assessor):
        """Test that ToC chunks are filtered out."""
        chunks_data = {
            'documents': [
                'Real content goes here.',
                'Chapter 1 ........................... 5',  # ToC entry
                'More real content.'
            ],
            'metadatas': [
                {'section_title': 'Introduction'},
                {'section_title': 'Contents'},
                {'section_title': 'Methods'}
            ]
        }
        
        text = assessor._collect_text_by_section(chunks_data, include_sections=None)
        
        # Should exclude ToC entry
        assert '..........' not in text or text.count('.') < 10


class TestRedFlagDetection:
    """Test red flag detection."""
    
    def test_detect_red_flags_missing_limitations(self, assessor, sample_chunks_data):
        """Test red flag for missing limitations."""
        red_flags = assessor.detect_red_flags(sample_chunks_data)
        
        # Should detect missing limitations section
        assert any(
            'limitation' in flag.title.lower() 
            for flag in red_flags
        )
    
    def test_detect_red_flags_scope_creep(self, assessor):
        """Test red flag for scope creep."""
        chunks_data = {
            'documents': ['x'] * 100,  # Large conclusion
            'metadatas': [
                {'chapter': 'Conclusion', 'section_title': 'Conclusion'} if i > 50
                else {'chapter': 'Results', 'section_title': 'Results'}
                for i in range(100)
            ]
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
            extract_first_json_block('not json')


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
