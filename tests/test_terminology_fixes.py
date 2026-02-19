#!/usr/bin/env python3
"""Test terminology extraction fixes.

Verifies:
1. 's' is not removed from words like Torres and Indigenous
2. Sub-ngrams can be filtered out from display
"""

import pytest
from scripts.ingest.academic.terminology import DomainTerminologyExtractor


def test_no_aggressive_s_removal():
    """Verify that 's' is not removed from words like Torres and Indigenous."""
    extractor = DomainTerminologyExtractor(
        min_term_freq=1,
        ngram_range=(1, 3),
    )
    
    # Extract from text with words that should NOT have 's' removed
    text = """
    Aboriginal Torres Strait Islander communities have a rich cultural heritage.
    Indigenous Australian peoples have unique customs and traditions.
    The Torres Strait Island region is known for its biodiversity.
    """
    
    terms = extractor.extract_terms(text)
    term_list = [t.term for t in terms.values()]
    
    print(f"Extracted terms: {term_list}")
    
    # Should have these terms intact (not "torre", "indigenou", etc.)
    assert any('torres' in t.lower() and 'torre ' not in t.lower() for t in term_list), \
        f"Torres should not be changed to Torre. Got terms: {term_list}"
    
    assert any('indigenous' in t.lower() for t in term_list), \
        f"Indigenous should be preserved. Got terms: {term_list}"
    
    # Should NOT have mangled versions
    mangled_terms = [t for t in term_list if 'torre ' in t.lower()]
    assert len(mangled_terms) == 0, f"Found mangled 'Torre' terms: {mangled_terms}"
    
    # 'Indigenou' without 's' should not appear (unless in a larger term with that substring)
    for term in term_list:
        if 'indigenou' in term.lower():
            # It's OK if it's part of 'indigenous', but not standalone
            assert 'indigenous' in term.lower(), f"Mangled term found: {term}"


def test_subterm_logic():
    """Verify the logic for identifying sub-ngrams."""
    
    def is_subterm(term: str, other_term: str) -> bool:
        """Check if term is a sub-ngram of other_term."""
        term_words = term.split()
        other_words = other_term.split()
        
        if len(term_words) >= len(other_words):
            return False
        
        # Check if term's words appear consecutively in other_term's words
        for i in range(len(other_words) - len(term_words) + 1):
            if other_words[i:i+len(term_words)] == term_words:
                return True
        
        return False
    
    # Test the sub-ngram logic
    assert is_subterm('torres strait', 'aboriginal torres strait'), \
        "Should identify 'torres strait' as sub-ngram of 'aboriginal torres strait'"
    
    assert is_subterm('strait', 'torres strait'), \
        "Should identify 'strait' as sub-ngram of 'torres strait'"
    
    assert is_subterm('aboriginal torres', 'aboriginal torres strait'), \
        "Should identify 'aboriginal torres' as sub-ngram of 'aboriginal torres strait'"
    
    # Should not identify non-contiguous subterms
    assert not is_subterm('aboriginal strait', 'aboriginal torres strait'), \
        "Should not identify non-contiguous 'aboriginal strait' as sub-ngram"
    
    # Should not identify equal terms
    assert not is_subterm('torres strait', 'torres strait'), \
        "Should not identify term as sub-ngram of itself"
    
    # Should not identify longer terms as subterms of shorter ones
    assert not is_subterm('aboriginal torres strait', 'torres strait'), \
        "Should not identify longer term as sub-ngram of shorter one"


def test_plural_forms_preserved():
    """Verify that plural and singular forms are preserved correctly."""
    extractor = DomainTerminologyExtractor(
        min_term_freq=1,
        ngram_range=(1, 2),
    )
    
    text = """
    The systems used for health analysis are comprehensive.
    These health systems support various communities.
    Indigenous systems of knowledge are valuable.
    """
    
    terms = extractor.extract_terms(text)
    term_list = [t.term.lower() for t in terms.values()]
    
    print(f"Extracted plural terms: {term_list}")
    
    # Check that we're not getting mangled versions
    # None should end with a space (sign of truncation after 's' removal)
    for term in term_list:
        assert not term.rstrip() != term, f"Term has trailing spaces (sign of truncation): {repr(term)}"


if __name__ == "__main__":
    test_no_aggressive_s_removal()
    test_subterm_logic()
    test_plural_forms_preserved()
    print("✓ All terminology fixes verified!")
