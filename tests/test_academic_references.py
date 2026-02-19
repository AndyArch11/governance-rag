"""Unit tests for AcademicReferences.

Tests the academic references dashboard module functionality including:
- Sub-ngram filtering logic
- Term display filtering (removes overlapping n-grams)
- Database connectivity
- Error handling
"""

import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

import pytest

# Note: dash and plotly are mocked in conftest.py before any imports
from scripts.ui.academic.academic_references import AcademicReferences


class TestIsSubterm:
    """Tests for the _is_subterm method."""
    
    def test_single_word_subterm(self):
        """Test identifying single-word sub-ngrams."""
        module = AcademicReferences()
        
        # Single word as part of a multi-word phrase
        assert module._is_subterm('strait', 'torres strait'), \
            "Single word should be identified as subterm"
        
        assert module._is_subterm('torres', 'torres strait'), \
            "First word should be identified as subterm"
    
    def test_multi_word_subterm(self):
        """Test identifying multi-word sub-ngrams."""
        module = AcademicReferences()
        
        # Two-word subterm
        assert module._is_subterm('torres strait', 'aboriginal torres strait'), \
            "Two-word term should be identified as subterm"
        
        assert module._is_subterm('strait islander', 'torres strait islander'), \
            "Ending subterm should be identified"
        
        assert module._is_subterm('aboriginal torres', 'aboriginal torres strait'), \
            "Starting subterm should be identified"
    
    def test_non_contiguous_words_not_subterm(self):
        """Test that non-contiguous word sequences are not identified as subterms."""
        module = AcademicReferences()
        
        # Words don't appear consecutively
        assert not module._is_subterm('aboriginal strait', 'aboriginal torres strait'), \
            "Non-contiguous words should not be identified as subterm"
        
        assert not module._is_subterm('torres islander', 'torres strait islander'), \
            "Non-contiguous words should not be identified as subterm"
    
    def test_equal_terms_not_subterm(self):
        """Test that equal terms are not considered subterms."""
        module = AcademicReferences()
        
        assert not module._is_subterm('torres strait', 'torres strait'), \
            "Equal terms should not be subterm of each other"
    
    def test_longer_term_not_subterm(self):
        """Test that longer terms are not subterms of shorter ones."""
        module = AcademicReferences()
        
        assert not module._is_subterm('aboriginal torres strait', 'torres strait'), \
            "Longer term should not be subterm of shorter one"
        
        assert not module._is_subterm('torres strait islander', 'torres strait'), \
            "Longer phrase should not be subterm of shorter one"
    
    def test_case_sensitive_matching(self):
        """Test that matching is case-sensitive (after lowercasing during extraction)."""
        module = AcademicReferences()
        
        # Both should be lowercase after extraction
        assert module._is_subterm('torres', 'torres strait'), \
            "Case-sensitive matching should work"
    
    def test_empty_terms(self):
        """Test handling of empty terms."""
        module = AcademicReferences()
        
        assert not module._is_subterm('', 'torres strait'), \
            "Empty term should not be subterm"
        
        assert not module._is_subterm('torres', ''), \
            "No parent term to check against"
    
    def test_single_word_parent_with_multi_word_child(self):
        """Test that multi-word terms cannot be subterms of single-word terms."""
        module = AcademicReferences()
        
        assert not module._is_subterm('torres strait', 'strait'), \
            "Multi-word term cannot be subterm of single word"


class TestAcademicReferencesModule:
    """Tests for AcademicReferences initialisation and database handling."""
    
    def test_init_without_database(self):
        """Test module initialisation without database."""
        module = AcademicReferences()
        
        assert module.terminology_db is None
        assert module.conn is None
    
    def test_init_with_nonexistent_database(self):
        """Test module initialisation with nonexistent database path."""
        nonexistent_path = Path("/nonexistent/path/terminology.db")
        
        module = AcademicReferences(terminology_db=nonexistent_path)
        
        assert module.terminology_db == nonexistent_path
        assert module.conn is None, "Should not connect to nonexistent database"
    
    def test_init_with_existing_database(self):
        """Test module initialisation with existing database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_terminology.db"
            
            # Create a minimal valid database
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE domain_terms (
                    id INTEGER PRIMARY KEY,
                    term TEXT,
                    domain TEXT,
                    frequency INTEGER,
                    domain_relevance_score REAL,
                    term_type TEXT,
                    doc_ids TEXT
                )
            """)
            conn.commit()
            conn.close()
            
            # Now test module connection
            module = AcademicReferences(terminology_db=db_path)
            
            assert module.terminology_db == db_path
            assert module.conn is not None, "Should connect to existing database"
    
    def test_get_top_terms_no_database(self):
        """Test get_top_terms returns empty list when no database connected."""
        module = AcademicReferences()
        
        result = module.get_top_terms('test_domain')
        
        assert result == [], "Should return empty list when no database"
    
    def test_get_top_terms_with_real_database(self):
        """Test get_top_terms with a real database containing terms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_terminology.db"
            
            # Create database with test data
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE domain_terms (
                    id INTEGER PRIMARY KEY,
                    term TEXT,
                    domain TEXT,
                    frequency INTEGER,
                    domain_relevance_score REAL,
                    term_type TEXT,
                    doc_ids TEXT
                )
            """)
            
            # Insert test terms
            test_terms = [
                ('aboriginal torres strait islander', 'indigenous', 75, 0.95, 'concept', 'doc1,doc2'),
                ('torres strait', 'indigenous', 74, 0.85, 'concept', 'doc1'),
                ('aboriginal torres', 'indigenous', 71, 0.80, 'concept', 'doc2'),
                ('strait', 'indigenous', 74, 0.60, 'concept', 'doc1'),
            ]
            
            cursor.executemany("""
                INSERT INTO domain_terms (term, domain, frequency, domain_relevance_score, term_type, doc_ids)
                VALUES (?, ?, ?, ?, ?, ?)
            """, test_terms)
            conn.commit()
            conn.close()
            
            # Test module retrieval
            module = AcademicReferences(terminology_db=db_path)
            result = module.get_top_terms('indigenous', limit=10)
            
            # Should get at least the top term
            assert len(result) > 0, "Should retrieve terms from database"
            
            # Top term should be the full phrase
            assert result[0]['term'] == 'aboriginal torres strait islander', \
                "Top term should be the full phrase"
    
    def test_get_top_terms_with_subterm_filtering(self):
        """Test that get_top_terms filters out sub-ngrams correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_terminology.db"
            
            # Create database with overlapping terms
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE domain_terms (
                    id INTEGER PRIMARY KEY,
                    term TEXT,
                    domain TEXT,
                    frequency INTEGER,
                    domain_relevance_score REAL,
                    term_type TEXT,
                    doc_ids TEXT
                )
            """)
            
            # Insert overlapping terms in order of relevance
            test_terms = [
                ('aboriginal torres strait islander', 'indigenous', 71, 0.95, 'concept', 'doc1'),
                ('torres strait islander', 'indigenous', 74, 0.90, 'concept', 'doc1'),
                ('torres strait', 'indigenous', 74, 0.85, 'concept', 'doc1'),
                ('strait', 'indigenous', 74, 0.75, 'concept', 'doc1'),
                ('aboriginal torres', 'indigenous', 71, 0.70, 'concept', 'doc1'),
            ]
            
            cursor.executemany("""
                INSERT INTO domain_terms (term, domain, frequency, domain_relevance_score, term_type, doc_ids)
                VALUES (?, ?, ?, ?, ?, ?)
            """, test_terms)
            conn.commit()
            conn.close()
            
            # Test module retrieval with filtering
            module = AcademicReferences(terminology_db=db_path)
            result = module.get_top_terms('indigenous', limit=10)
            
            # Should only get the top-level term, filtering out all subterms
            assert len(result) == 1, \
                f"Should filter out all subterms, got {len(result)} terms: {[t['term'] for t in result]}"
            
            assert result[0]['term'] == 'aboriginal torres strait islander', \
                "Should only return the top-level term"
    
    def test_get_top_terms_respects_limit(self):
        """Test that get_top_terms respects the limit parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_terminology.db"
            
            # Create database with multiple non-overlapping terms
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE domain_terms (
                    id INTEGER PRIMARY KEY,
                    term TEXT,
                    domain TEXT,
                    frequency INTEGER,
                    domain_relevance_score REAL,
                    term_type TEXT,
                    doc_ids TEXT
                )
            """)
            
            # Insert non-overlapping terms
            test_terms = []
            for i in range(10):
                test_terms.append((f'term{i}', 'domain', 10-i, 0.9-i*0.05, 'concept', 'doc1'))
            
            cursor.executemany("""
                INSERT INTO domain_terms (term, domain, frequency, domain_relevance_score, term_type, doc_ids)
                VALUES (?, ?, ?, ?, ?, ?)
            """, test_terms)
            conn.commit()
            conn.close()
            
            # Test with different limits
            module = AcademicReferences(terminology_db=db_path)
            
            result_5 = module.get_top_terms('domain', limit=5)
            assert len(result_5) <= 5, "Should respect limit of 5"
            
            result_3 = module.get_top_terms('domain', limit=3)
            assert len(result_3) <= 3, "Should respect limit of 3"
    
    def test_get_doc_ids_for_domain_no_database(self):
        """Test get_doc_ids_for_domain returns empty list when no database."""
        module = AcademicReferences()
        
        result = module.get_doc_ids_for_domain('test_domain')
        
        assert result == [], "Should return empty list when no database"


class TestSubtermFilteringScenarios:
    """Integration tests for realistic sub-ngram filtering scenarios."""
    
    def test_indigenous_australian_scenario(self):
        """Test filtering for the Indigenous Australian terminology scenario from bug report."""
        module = AcademicReferences()
        
        # Simulate the scenario from the bug report
        # Full phrase should be kept, components should be filtered
        full_term = 'aboriginal torres strait islander'
        components = [
            'torres strait islander',
            'aboriginal torres strait',
            'torres strait',
            'aboriginal torres',
            'strait',
            'islander',
            'torre',  # This shouldn't exist with our fix, but included for completeness
        ]
        
        # Verify filtering logic
        for component in components[:-1]:  # Skip 'torre' since it shouldn't exist
            assert module._is_subterm(component, full_term), \
                f"{component} should be filtered when {full_term} exists"
    
    def test_non_overlapping_terms_preserved(self):
        """Test that non-overlapping terms are not filtered."""
        module = AcademicReferences()
        
        # These terms don't overlap
        term1 = 'indigenous health'
        term2 = 'cultural practice'
        
        assert not module._is_subterm(term1, term2), \
            "Non-overlapping terms should not be filtered"
        
        assert not module._is_subterm(term2, term1), \
            "Non-overlapping terms should not be filtered"
    
    def test_partial_overlap_not_subterm(self):
        """Test that partial overlaps (non-contiguous) are not considered subterms."""
        module = AcademicReferences()
        
        # These terms overlap but not contiguously
        term1_partial = 'aboriginal indigenous'
        term1_full = 'aboriginal torres strait indigenous'
        
        # 'aboriginal' and 'indigenous' both appear but not together
        assert not module._is_subterm(term1_partial, term1_full), \
            "Non-contiguous overlapping words should not be subterm"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
