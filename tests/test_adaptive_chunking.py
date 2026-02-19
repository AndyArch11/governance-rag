"""Tests for adaptive chunking functionality in chunk.py."""

import pytest

from scripts.ingest.chunk import chunk_text, determine_chunk_params


class TestDetermineChunkParams:
    """Test suite for determine_chunk_params function."""

    def test_policy_doc_type_shorter_chunks(self):
        """Policy documents should get shorter chunks (600)."""
        text = "Standard text for testing. " * 100
        chunk_size, overlap = determine_chunk_params("compliance policy", text)

        # Should be based on 600 base, possibly adjusted
        assert 400 <= chunk_size <= 800  # Policy range
        assert overlap == int(chunk_size * 0.19)

    def test_guide_doc_type_longer_chunks(self):
        """Guide documents should get longer chunks (1000)."""
        text = "Detailed procedure step. " * 100
        chunk_size, overlap = determine_chunk_params("deployment guide", text)

        # Should be based on 1000 base, possibly adjusted
        assert 800 <= chunk_size <= 1200  # Guide range
        assert overlap == int(chunk_size * 0.19)

    def test_architecture_doc_type_longer_chunks(self):
        """Architecture documents should get longer chunks (1000)."""
        text = "Complex architectural concept. " * 100
        chunk_size, overlap = determine_chunk_params("system architecture", text)

        assert 800 <= chunk_size <= 1200
        # Architecture docs use 25% overlap, not 19%
        expected_overlap = int(chunk_size * 0.25)
        assert overlap == expected_overlap

    def test_default_doc_type_balanced_chunks(self):
        """Unknown doc types should get default size (800)."""
        text = "Generic document content. " * 100
        chunk_size, overlap = determine_chunk_params("unknown document", text)

        assert 600 <= chunk_size <= 1000  # Default range
        assert overlap == int(chunk_size * 0.19)

    def test_none_doc_type_uses_default(self):
        """None doc_type should use default base (800)."""
        text = "Text without type. " * 100
        chunk_size, overlap = determine_chunk_params(None, text)

        assert 600 <= chunk_size <= 1000
        assert overlap == int(chunk_size * 0.19)

    def test_high_heading_density_reduces_chunk_size(self):
        """Documents with many headings should get smaller chunks."""
        # Create text with high heading density
        text = "\n".join([f"## Heading {i}\nContent here." for i in range(50)])

        chunk_size, overlap = determine_chunk_params("guide", text)

        # Base is 1000 for guide, but high heading density should reduce
        assert chunk_size < 1000  # Should be reduced from base

    def test_long_sentences_increase_chunk_size(self):
        """Documents with long complex sentences should get larger chunks."""
        # Create text with very long sentences
        long_sentence = (
            "This is a very long sentence that contains many clauses and continues for a significant length because it is discussing complex technical concepts that require detailed explanation and comprehensive coverage. "
            * 5
        )
        text = long_sentence * 10

        chunk_size, overlap = determine_chunk_params("policy", text)

        # Base is 600 for policy, but long sentences should increase
        assert chunk_size > 600  # Should be increased from base

    def test_short_sentences_decrease_chunk_size(self):
        """Documents with short sentences (bullet points) should get smaller chunks."""
        # Create text with very short sentences
        text = "Item one. Item two. Item three. Item four. " * 50

        chunk_size, overlap = determine_chunk_params("guide", text)

        # Base is 1000 for guide, but short sentences should decrease
        assert chunk_size < 1000  # Should be decreased from base

    def test_chunk_size_bounds_enforced(self):
        """Chunk size should stay within 400-1200 range."""
        # Try to trigger extremes
        very_long_sentences = ("A" * 200 + ". ") * 100
        very_short_sentences = "X. " * 100

        size_long, _ = determine_chunk_params("guide", very_long_sentences)
        size_short, _ = determine_chunk_params("policy", very_short_sentences)

        assert 400 <= size_long <= 1200
        assert 400 <= size_short <= 1200

    def test_overlap_is_appropriate_for_doc_type(self):
        """Overlap should vary by document type (19-40%)."""
        test_cases = [
            ("policy", "Short. " * 100, 0.19),  # Default 19%
            ("guide", "Medium length content. " * 100, 0.19),  # Default 19%
            (
                "architecture",
                "Complex architectural description. " * 100,
                0.25,
            ),  # 25% for technical
        ]

        for doc_type, text, expected_ratio in test_cases:
            chunk_size, overlap = determine_chunk_params(doc_type, text)
            expected_overlap = int(chunk_size * expected_ratio)
            assert (
                overlap == expected_overlap
            ), f"Failed for {doc_type}: expected {expected_overlap}, got {overlap}"


class TestAdaptiveChunking:
    """Test suite for adaptive chunking in chunk_text function."""

    def test_adaptive_enabled_by_default(self):
        """Adaptive chunking should be enabled by default."""
        text = "Test content. " * 100

        # With doc_type, should adapt
        chunks_policy = chunk_text(text, doc_type="compliance policy")
        chunks_guide = chunk_text(text, doc_type="deployment guide")

        # Different doc types should produce different chunk counts
        # (Policy uses smaller chunks, guide uses larger chunks)
        assert len(chunks_policy) != len(chunks_guide)

    def test_static_mode_uses_defaults(self):
        """Static mode (adaptive=False) should use fixed 800/150."""
        text = "Test content. " * 200

        chunks_static = chunk_text(text, doc_type="policy", adaptive=False)
        chunks_adaptive = chunk_text(text, doc_type="policy", adaptive=True)

        # Static should ignore doc_type and use defaults
        # Adaptive should use smaller chunks for policy
        assert len(chunks_static) != len(chunks_adaptive)

    def test_policy_creates_more_chunks_than_guide(self):
        """Policy (smaller chunks) should create more chunks than guide (larger chunks)."""
        text = "Content for testing chunking behaviour. " * 200

        chunks_policy = chunk_text(text, doc_type="security policy")
        chunks_guide = chunk_text(text, doc_type="implementation guide")

        # Policy uses smaller chunks (600) so more chunks
        # Guide uses larger chunks (1000) so fewer chunks
        assert len(chunks_policy) > len(chunks_guide)

    def test_empty_text_returns_empty_list(self):
        """Empty text should return empty list regardless of mode."""
        assert chunk_text("", adaptive=True) == []
        assert chunk_text("", adaptive=False) == []
        assert chunk_text("", doc_type="policy", adaptive=True) == []

    def test_whitespace_only_returns_empty_list(self):
        """Whitespace-only text should return empty list."""
        assert chunk_text("   \n\n  \t  ", adaptive=True) == []
        assert chunk_text("   \n\n  \t  ", doc_type="guide") == []

    def test_doc_type_keywords_recognised(self):
        """Various doc_type keyword patterns should be recognised."""
        text = "Test content. " * 200

        # Policy keywords
        policy_types = ["governance policy", "compliance standard", "security requirement"]
        policy_chunk_counts = [len(chunk_text(text, doc_type=dt)) for dt in policy_types]

        # Guide keywords
        guide_types = ["deployment guide", "setup procedure", "installation tutorial"]
        guide_chunk_counts = [len(chunk_text(text, doc_type=dt)) for dt in guide_types]

        # Policy types should produce more chunks (smaller size)
        # Guide types should produce fewer chunks (larger size)
        avg_policy_chunks = sum(policy_chunk_counts) / len(policy_chunk_counts)
        avg_guide_chunks = sum(guide_chunk_counts) / len(guide_chunk_counts)

        assert avg_policy_chunks > avg_guide_chunks

    def test_adaptive_with_none_doc_type(self):
        """Adaptive mode with None doc_type should use default base."""
        text = "Test content. " * 150

        chunks_none = chunk_text(text, doc_type=None, adaptive=True)
        chunks_default = chunk_text(text, doc_type="unknown type", adaptive=True)

        # Both should use similar chunking (default base of 800)
        assert abs(len(chunks_none) - len(chunks_default)) <= 1

    def test_realistic_policy_document(self):
        """Test with realistic policy document structure."""
        policy_text = """
        # Security Policy
        
        ## Access Control
        - All users must authenticate
        - Multi-factor authentication required
        - Password complexity: minimum 12 characters
        
        ## Data Classification
        - Confidential data must be encrypted
        - Public data follows standard controls
        
        ## Compliance
        - Annual security reviews required
        - Incident reporting within 24 hours
        """

        chunks = chunk_text(policy_text, doc_type="security policy", adaptive=True)

        # Should create reasonable number of chunks
        assert len(chunks) >= 1
        # Chunks should not be too large for a policy doc
        assert all(len(chunk) <= 1000 for chunk in chunks)

    def test_realistic_guide_document(self):
        """Test with realistic guide document structure."""
        guide_text = (
            """
        # Deployment Guide
        
        ## Prerequisites
        Before beginning the deployment, ensure you have completed the following steps.
        First, verify that your environment meets all system requirements.
        This includes checking CPU, memory, disk space, and network connectivity.
        
        ## Installation Steps
        
        ### Step 1: Download Package
        Navigate to the releases page and download the latest stable version.
        Verify the checksum matches the published value to ensure integrity.
        Extract the archive to your desired installation directory.
        
        ### Step 2: Configure Environment
        Copy the example configuration file to your environment-specific location.
        Edit the configuration file with your specific settings and credentials.
        Test the configuration using the validation utility.
        
        ### Step 3: Run Deployment
        Execute the deployment script with appropriate permissions.
        Monitor the logs for any errors or warnings during installation.
        Verify all services start correctly after deployment completes.
        """
            * 3
        )

        chunks = chunk_text(guide_text, doc_type="deployment guide", adaptive=True)

        # Guide should create reasonable chunks
        assert len(chunks) >= 1
        # Chunks can be larger for guides
        assert all(len(chunk) <= 1400 for chunk in chunks)
