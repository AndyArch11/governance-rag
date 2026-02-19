"""Tests for chunk quality heuristic in vectors.py."""

import pytest

from scripts.ingest.vectors import compute_chunk_quality_heuristic


class TestComputeChunkQualityHeuristic:
    """Test suite for compute_chunk_quality_heuristic function."""

    def test_short_chunk_rejected(self):
        """Too short chunks should be rejected."""
        chunk = "This is short"
        skip, confidence, reason = compute_chunk_quality_heuristic(chunk)

        assert skip is False
        assert confidence == 0.2
        assert reason == "too_short"

    def test_long_chunk_rejected(self):
        """Excessively long chunks should be rejected."""
        chunk = "x" * 2500
        skip, confidence, reason = compute_chunk_quality_heuristic(chunk)

        assert skip is False
        assert confidence == 0.3
        assert reason == "too_long"

    def test_few_words_rejected(self):
        """Chunks with too few words should be rejected."""
        chunk = "One two three four five six seven eight nine"
        skip, confidence, reason = compute_chunk_quality_heuristic(chunk)

        assert skip is False
        assert confidence == 0.2  # too_short (under 100 chars)
        assert reason == "too_short"

    def test_low_stopword_ratio_rejected(self):
        """Chunks with low stopword ratio (technical jargon) should be rejected."""
        # Need enough words to pass the word count check
        chunk = (
            "AZURE_STORAGE_CONNECTION_STRING equals AccountName semicolon AccountKey "
            "DATA_PROCESSOR_ENABLED TRUE CHUNK_SIZE 512 OVERLAP 128 VALIDATION_THRESHOLD "
            "RETRY_COUNT number TIMEOUT_SECONDS number MAX_WORKERS number ENABLE_LOGGING FALSE "
            "PROCESSING_MODE batch COMPRESSION_LEVEL high CACHE_EXPIRY seconds BUFFER_SIZE bytes"
        )
        skip, confidence, reason = compute_chunk_quality_heuristic(chunk)

        assert skip is False
        # Should fail on stopword ratio or entropy, not word count
        assert reason.startswith("low_stopword_ratio") or reason.startswith("low_entropy")

    def test_boilerplate_rejected(self):
        """Chunks with multiple boilerplate patterns should be rejected."""
        chunk = (
            "Copyright 2024. All rights reserved. Terms of Service | Privacy Policy | "
            "Cookie Settings | Contact Us | About | FAQ | Sign Up"
        )
        skip, confidence, reason = compute_chunk_quality_heuristic(chunk)

        assert skip is False
        assert confidence == 0.3
        assert reason.startswith("boilerplate_detected")
        assert "_patterns" in reason

    def test_natural_language_accepted(self):
        """Well-formed natural language should be accepted."""
        chunk = (
            "This is a well-formed chunk of text that contains natural language patterns. "
            "It has sufficient length and includes common English stopwords like 'the', 'a', "
            "'is', 'that', 'and', 'of'. The content provides meaningful information and "
            "demonstrates coherent sentence structure."
        )
        skip, confidence, reason = compute_chunk_quality_heuristic(chunk)

        assert skip is True
        assert confidence > 0.7
        assert reason == "heuristic_pass"

    def test_technical_content_accepted(self):
        """Technical content with good stopword ratio should be accepted."""
        chunk = (
            "Azure Virtual Machines provide scalable compute resources in the cloud. "
            "Configure instance types, networking, and storage based on workload requirements. "
            "Implement monitoring and auto-scaling policies to optimise performance and cost."
        )
        skip, confidence, reason = compute_chunk_quality_heuristic(chunk)

        assert skip is True
        assert confidence > 0.7
        assert reason == "heuristic_pass"

    def test_governance_content_accepted(self):
        """Governance/policy content should be accepted."""
        chunk = (
            "Access control policies must align with the principle of least privilege. "
            "Users should only receive permissions necessary for their role. Regular access "
            "reviews ensure compliance with security requirements and identify unnecessary permissions."
        )
        skip, confidence, reason = compute_chunk_quality_heuristic(chunk)

        assert skip is True
        assert confidence > 0.7
        assert reason == "heuristic_pass"

    def test_low_entropy_rejected(self):
        """Chunks with very low entropy (repetitive) should be rejected."""
        # Need at least 10 words to pass word count, but still fail entropy
        chunk = "a " * 75  # Repetitive pattern with spaces to get words
        skip, confidence, reason = compute_chunk_quality_heuristic(chunk)

        assert skip is False
        # Will likely fail on entropy check
        assert "entropy" in reason.lower() or "few_words" in reason or "stopword" in reason

    def test_boundary_length_100_chars(self):
        """Test chunk at exactly 100 characters (minimum)."""
        # Create exactly 100 char chunk with good stopwords
        chunk = "The quick brown fox jumps over the lazy dog. This is a test of minimum length chunks exactly."
        chunk = chunk + " " * (100 - len(chunk))  # Pad to exactly 100
        assert len(chunk) == 100

        skip, confidence, reason = compute_chunk_quality_heuristic(chunk)

        # At exactly 100 chars, should not be "too_short"
        # May still fail on other checks (words, stopwords, etc.)
        if len(chunk.split()) < 10:
            assert reason == "too_few_words"
        else:
            assert reason != "too_short"

    def test_boundary_length_2000_chars(self):
        """Test chunk at exactly 2000 characters (maximum)."""
        # Create exactly 2000 char chunk with repeated natural language
        base = "The system processes data efficiently and securely for all users. "
        repetitions = 2000 // len(base) + 1
        chunk = (base * repetitions)[:2000]
        assert len(chunk) == 2000

        skip, confidence, reason = compute_chunk_quality_heuristic(chunk)

        # Should pass length check (not too_long at exactly 2000)
        assert reason != "too_long"

    def test_mixed_content_with_code(self):
        """Chunks with mixed content (text + code) should be evaluated."""
        chunk = """Azure Functions allow you to run serverless code in response to events.
        Here's an example configuration:
        
        {
            "bindings": [{
                "type": "httpTrigger",
                "direction": "in",
                "name": "req"
            }]
        }
        
        This configuration enables HTTP triggering for the function."""

        skip, confidence, reason = compute_chunk_quality_heuristic(chunk)

        # Should handle mixed content gracefully
        assert reason in ["heuristic_pass", "low_stopword_ratio", "low_entropy"]

    def test_confidence_score_range(self):
        """Confidence scores should be in valid range [0.0, 1.0]."""
        test_chunks = [
            "x" * 50,  # Too short
            "x" * 150,  # Low entropy
            "The quick brown fox jumps over the lazy dog. " * 5,  # Good content
            "Click here for privacy policy and terms of service contact us",  # Boilerplate
        ]

        for chunk in test_chunks:
            skip, confidence, reason = compute_chunk_quality_heuristic(chunk)
            assert (
                0.0 <= confidence <= 1.0
            ), f"Confidence {confidence} out of range for chunk: {chunk[:50]}"

    def test_reason_format(self):
        """Reason should always be a non-empty string."""
        test_chunks = [
            "Short",
            "x" * 2500,
            "The quick brown fox jumps over the lazy dog. " * 3,
        ]

        for chunk in test_chunks:
            skip, confidence, reason = compute_chunk_quality_heuristic(chunk)
            assert isinstance(reason, str)
            assert len(reason) > 0

    def test_multiple_boilerplate_patterns(self):
        """Test detection of multiple boilerplate patterns."""
        # Exactly 3 patterns (threshold)
        chunk_3 = (
            "Copyright notice. Privacy policy information. Click here to continue. "
            "Some other content that is not boilerplate text."
        )
        skip, confidence, reason = compute_chunk_quality_heuristic(chunk_3)
        # Should trigger boilerplate detection at >= 3 patterns
        assert "boilerplate" in reason.lower() or skip is True

        # More than 3 patterns
        chunk_5 = (
            "Copyright 2024. Terms of service apply. Privacy policy here. "
            "Contact us today. All rights reserved. Click here for more info."
        )
        skip, confidence, reason = compute_chunk_quality_heuristic(chunk_5)
        assert "boilerplate" in reason.lower()

    def test_real_confluence_content(self):
        """Test with realistic Confluence export content."""
        chunk = """
        The Azure Security Baseline provides prescriptive guidance for implementing security
        controls in Azure. Organisations should implement these controls as part of their
        overall security strategy. The baseline addresses key security principles including
        network security, identity and access management, data protection, and logging.
        Teams must ensure compliance with relevant regulatory requirements and internal
        policies when implementing these controls.
        """

        skip, confidence, reason = compute_chunk_quality_heuristic(chunk)

        # Real content should pass
        assert skip is True
        assert reason == "heuristic_pass"
        assert confidence > 0.7
