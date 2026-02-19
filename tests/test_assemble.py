"""Tests for assemble module.

Tests prompt building, context formatting, and validation logic
for RAG generation.
"""

import pytest

from scripts.rag.assemble import SYSTEM_PROMPT, build_prompt


class TestSystemPrompt:
    """Tests for SYSTEM_PROMPT constant."""

    def test_system_prompt_exists(self):
        """Test that SYSTEM_PROMPT is defined and non-empty."""
        assert SYSTEM_PROMPT
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0

    def test_system_prompt_contains_key_instructions(self):
        """Test that SYSTEM_PROMPT contains essential instructions."""
        assert "technical assistant" in SYSTEM_PROMPT.lower()
        assert "context" in SYSTEM_PROMPT.lower()
        assert "do not" in SYSTEM_PROMPT.lower() or "don't" in SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_governance(self):
        """Test that system prompt mentions governance domain."""
        assert "governance" in SYSTEM_PROMPT.lower()

    def test_system_prompt_instructs_against_invention(self):
        """Test that system prompt warns against inventing information."""
        prompt_lower = SYSTEM_PROMPT.lower()
        assert "do not invent" in prompt_lower or "don't invent" in prompt_lower


class TestBuildPrompt:
    """Tests for build_prompt function."""

    def test_build_prompt_basic(self):
        """Test basic prompt building with single chunk."""
        query = "What is MFA?"
        chunks = ["Multi-factor authentication (MFA) requires two or more verification factors."]

        prompt = build_prompt(query, chunks)

        assert isinstance(prompt, str)
        assert query in prompt
        assert chunks[0] in prompt
        assert "CONTEXT:" in prompt
        assert "QUESTION:" in prompt
        assert "ANSWER:" in prompt

    def test_build_prompt_multiple_chunks(self):
        """Test prompt building with multiple context chunks."""
        query = "What are the security requirements?"
        chunks = [
            "Security requirement 1: All users must use MFA.",
            "Security requirement 2: Passwords must be at least 12 characters.",
            "Security requirement 3: Sessions expire after 30 minutes.",
        ]

        prompt = build_prompt(query, chunks)

        # All chunks should be in the prompt
        for chunk in chunks:
            assert chunk in prompt

        # Query should be in the prompt
        assert query in prompt

    def test_build_prompt_chunks_separated(self):
        """Test that chunks are separated in the prompt."""
        query = "Test query"
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]

        prompt = build_prompt(query, chunks)

        # Check that chunks are separated by double newlines
        assert "Chunk 1\n\nChunk 2\n\nChunk 3" in prompt

    def test_build_prompt_includes_system_prompt(self):
        """Test that built prompt includes system instructions."""
        query = "Test query"
        chunks = ["Test context"]

        prompt = build_prompt(query, chunks)

        # Should contain parts of system prompt
        assert "technical assistant" in prompt.lower()
        assert "context" in prompt.lower()

    def test_build_prompt_structure(self):
        """Test that prompt follows expected structure."""
        query = "What is this?"
        chunks = ["This is a test context."]

        prompt = build_prompt(query, chunks)

        # Check structure: System prompt, then CONTEXT, then QUESTION, then ANSWER
        system_idx = prompt.find("technical assistant")
        context_idx = prompt.find("CONTEXT:")
        question_idx = prompt.find("QUESTION:")
        answer_idx = prompt.find("ANSWER:")

        assert system_idx < context_idx < question_idx < answer_idx

    def test_build_prompt_empty_query_raises_error(self):
        """Test that empty query raises ValueError."""
        chunks = ["Some context"]

        with pytest.raises(ValueError, match="Query cannot be empty"):
            build_prompt("", chunks)

    def test_build_prompt_whitespace_query_raises_error(self):
        """Test that whitespace-only query raises ValueError."""
        chunks = ["Some context"]

        with pytest.raises(ValueError, match="Query cannot be empty"):
            build_prompt("   ", chunks)

    def test_build_prompt_empty_chunks_raises_error(self):
        """Test that empty chunks list raises ValueError."""
        query = "What is this?"

        with pytest.raises(ValueError, match="Context chunks cannot be empty"):
            build_prompt(query, [])

    def test_build_prompt_with_special_characters(self):
        """Test prompt building with special characters in query and chunks."""
        query = "What is MFA? (Multi-Factor Auth)"
        chunks = ['MFA requires "two or more" verification factors.', "Symbols: @#$%^&*()"]

        prompt = build_prompt(query, chunks)

        assert query in prompt
        assert chunks[0] in prompt
        assert chunks[1] in prompt

    def test_build_prompt_with_long_query(self):
        """Test prompt building with long query."""
        query = " ".join(["word"] * 100)  # 100-word query
        chunks = ["Context chunk"]

        prompt = build_prompt(query, chunks)

        assert query in prompt
        assert chunks[0] in prompt

    def test_build_prompt_with_many_chunks(self):
        """Test prompt building with many context chunks."""
        query = "Test query"
        chunks = [f"Chunk {i}" for i in range(50)]

        prompt = build_prompt(query, chunks)

        # All chunks should be present
        for chunk in chunks:
            assert chunk in prompt

    def test_build_prompt_with_multiline_chunks(self):
        """Test prompt building with multiline context chunks."""
        query = "What are the requirements?"
        chunks = ["Line 1\nLine 2\nLine 3", "Another chunk\nWith multiple lines"]

        prompt = build_prompt(query, chunks)

        assert chunks[0] in prompt
        assert chunks[1] in prompt

    def test_build_prompt_with_unicode(self):
        """Test prompt building with unicode characters."""
        query = "What is été?"
        chunks = ["Été means summer in French. Ñoño is a Spanish word."]

        prompt = build_prompt(query, chunks)

        assert query in prompt
        assert chunks[0] in prompt

    def test_build_prompt_preserves_chunk_order(self):
        """Test that chunk order is preserved in the prompt."""
        query = "Test"
        chunks = ["First", "Second", "Third"]

        prompt = build_prompt(query, chunks)

        first_idx = prompt.find("First")
        second_idx = prompt.find("Second")
        third_idx = prompt.find("Third")

        assert first_idx < second_idx < third_idx

    def test_build_prompt_with_governance_context(self):
        """Test prompt building with governance-related content."""
        query = "What are the compliance requirements?"
        chunks = [
            "AESCSF requires annual security assessments.",
            "CIS Controls mandate multi-factor authentication.",
            "All systems must undergo vulnerability scanning quarterly.",
        ]

        prompt = build_prompt(query, chunks)

        assert "compliance requirements" in prompt
        assert "AESCSF" in prompt
        assert "CIS Controls" in prompt
        assert "vulnerability scanning" in prompt

    def test_build_prompt_with_patterns_context(self):
        """Test prompt building with architectural pattern content."""
        query = "How should I structure microservices?"
        chunks = [
            "Microservices should be independently deployable.",
            "Use API gateways for external communication.",
            "Implement circuit breakers for resilience.",
        ]

        prompt = build_prompt(query, chunks)

        assert query in prompt
        assert all(chunk in prompt for chunk in chunks)

    def test_build_prompt_result_is_string(self):
        """Test that build_prompt always returns a string."""
        query = "Test"
        chunks = ["Context"]

        result = build_prompt(query, chunks)

        assert isinstance(result, str)

    def test_build_prompt_with_single_word_query(self):
        """Test prompt building with single word query."""
        query = "MFA"
        chunks = ["Multi-factor authentication explanation"]

        prompt = build_prompt(query, chunks)

        assert query in prompt
        assert chunks[0] in prompt

    def test_build_prompt_context_formatting(self):
        """Test that context section is properly formatted."""
        query = "Test"
        chunks = ["A", "B", "C"]

        prompt = build_prompt(query, chunks)

        # Find the CONTEXT section
        context_start = prompt.find("CONTEXT:")
        question_start = prompt.find("QUESTION:")

        context_section = prompt[context_start:question_start]

        # Should contain all chunks with proper separation
        assert "A\n\nB\n\nC" in context_section


class TestBuildPromptEdgeCases:
    """Edge case tests for build_prompt function."""

    def test_build_prompt_with_empty_string_in_chunks(self):
        """Test that chunks containing empty strings are handled."""
        query = "Test"
        chunks = ["Valid chunk", "", "Another chunk"]

        # Should not raise an error
        prompt = build_prompt(query, chunks)

        assert "Valid chunk" in prompt
        assert "Another chunk" in prompt

    def test_build_prompt_query_with_newlines(self):
        """Test query containing newlines."""
        query = "What is this?\nAnd what is that?"
        chunks = ["Context"]

        prompt = build_prompt(query, chunks)

        assert query in prompt

    def test_build_prompt_with_very_long_chunk(self):
        """Test with a very long context chunk."""
        query = "Test"
        long_chunk = "word " * 10000  # Very long chunk
        chunks = [long_chunk]

        prompt = build_prompt(query, chunks)

        assert long_chunk.strip() in prompt

    def test_build_prompt_idempotency(self):
        """Test that calling build_prompt multiple times with same inputs gives same result."""
        query = "What is MFA?"
        chunks = ["MFA is multi-factor authentication."]

        prompt1 = build_prompt(query, chunks)
        prompt2 = build_prompt(query, chunks)

        assert prompt1 == prompt2
