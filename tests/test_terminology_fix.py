#!/usr/bin/env python3
"""
Quick test of the terminology extraction scoring fixes.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.ingest.academic.terminology import DomainTerminologyExtractor


def _run_scoring_check() -> bool:
    """Run terminology scoring check and return pass/fail status."""
    extractor = DomainTerminologyExtractor(min_term_freq=2, max_terms=50)

    # Create test text with real domain terms and noise
    text = """
    Leadership is a core concept in organisational psychology. Ethical leadership and 
    servant leadership models emphasise diversity and inclusivity in the workplace.
    Indigenous perspectives on leadership bring unique ethical considerations.
    
    Some random noise: spcrd interchange benjamins aral preemption tvgn toor.
    More noise repeated twice: spcrd spcrd interchange interchange.
    
    Leadership appears many times: leadership leadership leadership leadership
    leadership leadership leadership leadership leadership leadership.
    Ethical also appears frequently: ethical ethical ethical ethical ethical.
    Diversity: diversity diversity diversity diversity diversity.
    """

    scores = extractor.extract_terms(text, doc_id="test_doc")

    # Find top terms
    sorted_terms = sorted(scores.items(), key=lambda x: x[1].domain_relevance, reverse=True)

    print("Top 30 terms by domain relevance:")
    print("-" * 80)
    for i, (term, score) in enumerate(sorted_terms[:30], 1):
        print(
            f"{i:2}. {term:30} freq={score.frequency:3} rel={score.domain_relevance:.3f} type={score.term_type}"
        )

    # Check that real domain terms rank higher than noise
    top_5_terms = {term for term, _ in sorted_terms[:5]}
    top_10_terms = {term for term, _ in sorted_terms[:10]}
    noise_terms = {"spcrd", "interchange", "benjamins", "aral", "preemption", "tvgn", "toor"}

    print("\n" + "=" * 80)
    print(f"Top 5 terms: {top_5_terms}")
    print(f"Noise terms in top 5: {noise_terms & top_5_terms}")

    if (
        "leadership" in top_5_terms
        and not (noise_terms & top_5_terms)
        and "ethical" in top_10_terms
    ):
        print("✓ Domain terms ranking correctly!")
        return True
    print("✗ Domain terms NOT ranking high enough")
    return False


def test_scoring():
    """Test that scoring penalises noise terms correctly."""
    assert _run_scoring_check(), "Domain terms did not rank high enough"


if __name__ == "__main__":
    success = _run_scoring_check()
    sys.exit(0 if success else 1)
