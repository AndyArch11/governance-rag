"""Tests for generate module.

Covers answer orchestration: retrieval, prompt building, LLM invocation,
and fallbacks when no context is available.
"""

import sys
from pathlib import Path

import pytest


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []
        self.debugs = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg)

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg)

    def error(self, msg, *args, **kwargs):
        self.errors.append(msg)

    def debug(self, msg, *args, **kwargs):
        self.debugs.append(msg)


class DummyLLM:
    def __init__(self, model: str = "dummy", temperature: float = 0.0, **kwargs):
        """Accept model and temperature like OllamaLLM for compatibility."""
        self.model = model
        self.temperature = temperature
        self.prompts = []
        self.responses = []

    def set_responses(self, *responses):
        self.responses = list(responses)

    def invoke(self, prompt: str):  # noqa: D401
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return ""


@pytest.fixture()
def generate_module(monkeypatch):
    """Load generate with mocked dependencies."""
    # Import lazily after patching path
    import importlib

    # Fresh import to apply monkeypatches cleanly
    sys.modules.pop("scripts.rag.generate", None)
    generate = importlib.import_module("scripts.rag.generate")

    # Replace logger and audit
    dummy_logger = DummyLogger()
    audit_events = []

    def dummy_audit(event_type, data):
        audit_events.append((event_type, data))

    monkeypatch.setattr(generate, "logger", dummy_logger)
    monkeypatch.setattr(generate, "audit", dummy_audit)

    # Replace _get_llm factory to return our dummy LLM with proper temperature support
    # Track the last created LLM instance and pending responses
    created_llm = [None]  # Use list to allow mutation in nested function
    pending_responses = [None]  # Responses to set on next LLM creation

    def mock_get_llm(temperature=None):
        temp = temperature if temperature is not None else generate.config.temperature
        llm = DummyLLM(temperature=temp)
        # Apply any pending responses set before LLM was created
        if pending_responses[0]:
            llm.set_responses(*pending_responses[0])
            pending_responses[0] = None
        created_llm[0] = llm
        return llm

    monkeypatch.setattr(generate, "_get_llm", mock_get_llm)

    # Create tracker that manages pending responses
    class LLMTracker:
        def __init__(self):
            self.created = created_llm
            self.pending = pending_responses
            self.responses = []

        def set_responses(self, *responses):
            self.responses = list(responses)
            # If LLM exists, set directly; otherwise store for next creation
            if self.created[0]:
                self.created[0].set_responses(*responses)
            else:
                self.pending[0] = responses

        @property
        def temperature(self):
            return self.created[0].temperature if self.created[0] else None

        @property
        def prompts(self):
            """Delegate to created LLM's prompts."""
            return self.created[0].prompts if self.created[0] else []

    dummy_llm = LLMTracker()

    return generate, dummy_logger, dummy_llm, audit_events


class TestAnswer:
    """Tests for answer function."""

    def test_generation_retry_attempts_audit_estimated_tokens(self, monkeypatch, generate_module):
        from scripts.utils import llm_instrumentation

        generate, _, _, audit_events = generate_module
        attempts = []

        class FlakyLLM:
            model = "test-model"

            def invoke(self, _prompt):
                attempts.append(None)
                if len(attempts) < 3:
                    raise TimeoutError("request timed out")
                return "a generated answer"

        monkeypatch.setattr(generate, "_get_llm", lambda **_: FlakyLLM())
        monkeypatch.setattr("scripts.utils.retry_utils.time.sleep", lambda _: None)
        monkeypatch.setattr(
            llm_instrumentation,
            "audit",
            lambda event, data: audit_events.append((event, data)),
        )

        prompt = "A compact prompt for retry accounting."
        result = generate._invoke_llm_with_retry(prompt)

        usage_events = [data for event, data in audit_events if event == "llm_usage"]
        assert result == "a generated answer"
        assert len(usage_events) == 3
        assert [event["success"] for event in usage_events] == [False, False, True]
        assert all(event["input_tokens"] == len(prompt) // 4 for event in usage_events)
        assert [event["output_tokens"] for event in usage_events] == [0, 0, 4]
        assert [event["attempt"] for event in usage_events] == [1, 2, 3]
        assert all(event["token_source"] == "estimated" for event in usage_events)

    def test_empty_query_raises(self, generate_module):
        generate, _, _, _ = generate_module
        with pytest.raises(ValueError, match="Query cannot be empty"):
            generate.answer("", collection=None)

    def test_no_chunks_returns_fallback(self, monkeypatch, generate_module):
        generate, logger, _, audit_events = generate_module

        # Mock retrieve to return no chunks
        monkeypatch.setattr(
            generate, "retrieve", lambda q, c, k=None, persona=None, domain=None, **kwargs: ([], [])
        )

        response = generate.answer("test query", collection=None, k=3)

        assert response["answer"].startswith("No relevant information")
        assert response["chunks"] == []
        assert response["sources"] == []
        assert response["retrieval_count"] == 0
        assert response["model"]

        # Audit should record no results
        assert ("answer_no_results", {"query": "test query"[:100]}) in audit_events
        # Logger should have a warning
        assert any("No chunks" in msg for msg in logger.warnings)

    def test_answer_success_flow(self, monkeypatch, generate_module):
        generate, logger, dummy_llm, audit_events = generate_module

        # Mock retrieve to return chunks and sources
        chunks = ["chunk one", "chunk two"]
        sources = [{"id": "1"}, {"id": "2"}]
        monkeypatch.setattr(
            generate,
            "retrieve",
            lambda q, c, k=None, persona=None, domain=None, **kwargs: (chunks, sources),
        )

        # Mock build_prompt to check inputs
        prompts_built = []
        monkeypatch.setattr(
            generate,
            "build_prompt",
            lambda query, ctx, **kwargs: prompts_built.append((query, ctx)) or "PROMPT",
        )

        # LLM response
        dummy_llm.set_responses("ANSWER text ")

        response = generate.answer("What is MFA?", collection=None, k=2)

        # Validate response structure
        assert response["answer"] == "ANSWER text"
        assert response["chunks"] == chunks
        assert response["sources"] == sources
        assert response["retrieval_count"] == len(chunks)
        assert response["model"] == generate.config.model_name
        assert "generation_time" in response and "total_time" in response

        # Prompt built correctly
        assert prompts_built == [("What is MFA?", chunks)]
        # LLM invoked with built prompt
        assert dummy_llm.prompts and dummy_llm.prompts[0] == "PROMPT"

        # Audit event for success
        events = [evt for evt, _ in audit_events]
        assert "answer_generated" in events

        # Logger info messages present
        assert any("Retrieving" in msg for msg in logger.infos)
        assert any("Generating answer" in msg for msg in logger.infos)

    def test_academic_answer_passes_assigned_lens_guidance_to_prompt(
        self, monkeypatch, generate_module
    ):
        generate, _, dummy_llm, _ = generate_module
        chunks = ["The thesis discusses healing through relationships with Country and kin."]
        sources = [
            {
                "source_category": "academic_reference",
                "source_kind": "thesis_document",
                "thesis_id": "thesis-one",
            }
        ]
        monkeypatch.setattr(
            generate,
            "retrieve",
            lambda *args, **kwargs: (chunks, sources),
        )
        guidance_calls = []
        monkeypatch.setattr(
            generate,
            "get_cultural_lens_guidance_for_sources",
            lambda retrieved_sources, **kwargs: guidance_calls.append((retrieved_sources, kwargs))
            or "Use the assigned cultural lens.",
        )
        prompts_built = []
        monkeypatch.setattr(
            generate,
            "build_academic_aware_prompt",
            lambda query, context, **kwargs: prompts_built.append(kwargs) or "ACADEMIC PROMPT",
        )
        dummy_llm.set_responses("Grounded answer.")

        result = generate.answer(
            "How does the thesis describe healing?",
            collection=None,
            persona="assessor",
            custom_role="Custom academic role.",
        )

        assert result["answer"] == "Grounded answer."
        assert guidance_calls[0][0] == sources
        assert prompts_built[0]["custom_role"] == "Custom academic role."
        assert prompts_built[0]["additional_system_guidance"] == "Use the assigned cultural lens."

    def test_answer_respects_k_override(self, monkeypatch, generate_module):
        generate, _, dummy_llm, _ = generate_module

        captured_k = []

        def fake_retrieve(q, c, k=None, persona=None, domain=None, **kwargs):
            captured_k.append(k)
            return ["c"], ["s"]

        monkeypatch.setattr(generate, "retrieve", fake_retrieve)
        monkeypatch.setattr(generate, "build_prompt", lambda q, ctx, **kwargs: "PROMPT")
        dummy_llm.set_responses("ans")

        generate.answer("q", collection=None, k=7)

        assert captured_k == [7]

    def test_answer_uses_default_k_from_config(self, monkeypatch, generate_module):
        generate, _, dummy_llm, _ = generate_module

        captured_k = []

        def fake_retrieve(q, c, k=None, persona=None, domain=None, **kwargs):
            captured_k.append(k)
            return ["c"], ["s"]

        monkeypatch.setattr(generate, "retrieve", fake_retrieve)
        monkeypatch.setattr(generate, "build_prompt", lambda q, ctx, **kwargs: "PROMPT")
        dummy_llm.set_responses("ans")

        generate.answer("q", collection=None)

        assert captured_k == [generate.config.k_results]


class TestCodeQueryCompatibility:
    """Tests that code-specific answer generation is no longer active."""

    def test_code_like_query_uses_standard_prompt(self, monkeypatch, generate_module):
        generate, logger, dummy_llm, audit_events = generate_module

        chunks = ["public class AuthService { }"]
        sources = [{"language": "java", "service_name": "AuthService"}]
        monkeypatch.setattr(
            generate,
            "retrieve",
            lambda q, c, k=None, persona=None, domain=None, **kwargs: (chunks, sources),
        )
        prompts_built = []
        monkeypatch.setattr(
            generate,
            "build_prompt",
            lambda query, context, **kwargs: prompts_built.append((query, context))
            or "STANDARD_PROMPT",
        )
        dummy_llm.set_responses("Generated answer")

        response = generate.answer("Show Java services", collection=None)

        assert response["is_code_query"] is False
        assert prompts_built == [("Show Java services", chunks)]
        assert response["answer"] == "Generated answer"

    def test_non_code_query_uses_standard_prompt(self, monkeypatch, generate_module):
        generate, logger, dummy_llm, audit_events = generate_module

        chunks = ["MFA requires..."]
        sources = [{"id": "1"}]
        monkeypatch.setattr(
            generate,
            "retrieve",
            lambda q, c, k=None, persona=None, domain=None, **kwargs: (chunks, sources),
        )
        prompts_built = []
        monkeypatch.setattr(
            generate,
            "build_prompt",
            lambda query, ctx, **kwargs: prompts_built.append((query, ctx)) or "STD_PROMPT",
        )
        dummy_llm.set_responses("Security answer")

        response = generate.answer("What is MFA?", collection=None)

        assert response["is_code_query"] is False
        # Standard prompt should have been called
        assert len(prompts_built) > 0

    def test_code_like_response_is_not_postprocessed(self, monkeypatch, generate_module):
        generate, logger, dummy_llm, audit_events = generate_module

        chunks = ["@Service public class PaymentService { }"]
        sources = [
            {"language": "java", "bitbucket_url": "https://bitbucket.com/repo/PaymentService.java"}
        ]
        monkeypatch.setattr(
            generate,
            "retrieve",
            lambda q, c, k=None, persona=None, domain=None, **kwargs: (chunks, sources),
        )
        monkeypatch.setattr(generate, "build_prompt", lambda *args, **kwargs: "STANDARD_PROMPT")
        dummy_llm.set_responses("Payment service implementation")

        response = generate.answer("Show payment service class", collection=None)

        assert response["is_code_query"] is False
        assert response["answer"] == "Payment service implementation"

    def test_response_contains_is_code_query_flag(self, monkeypatch, generate_module):
        generate, _, dummy_llm, _ = generate_module

        chunks = ["test chunk"]
        sources = [{"id": "1"}]
        monkeypatch.setattr(
            generate,
            "retrieve",
            lambda q, c, k=None, persona=None, domain=None, **kwargs: (chunks, sources),
        )
        monkeypatch.setattr(generate, "build_prompt", lambda q, ctx, **kwargs: "PROMPT")
        dummy_llm.set_responses("answer")

        response = generate.answer("What is MFA?", collection=None)

        assert "is_code_query" in response
        assert isinstance(response["is_code_query"], bool)

    def test_audit_includes_code_query_flag(self, monkeypatch, generate_module):
        generate, _, dummy_llm, audit_events = generate_module

        chunks = ["chunk"]
        sources = [{"id": "1"}]
        monkeypatch.setattr(
            generate,
            "retrieve",
            lambda q, c, k=None, persona=None, domain=None, **kwargs: (chunks, sources),
        )
        monkeypatch.setattr(generate, "build_prompt", lambda q, ctx, **kwargs: "PROMPT")
        dummy_llm.set_responses("answer")

        generate.answer("test query", collection=None)

        # Find the answer_generated event
        answer_events = [(evt, data) for evt, data in audit_events if evt == "answer_generated"]
        assert len(answer_events) > 0
        assert "is_code_query" in answer_events[0][1]

    def test_answer_with_custom_temperature(self, monkeypatch, generate_module):
        generate, _, dummy_llm, _ = generate_module

        chunks = ["test chunk"]
        sources = [{"id": "1"}]
        monkeypatch.setattr(
            generate,
            "retrieve",
            lambda q, c, k=None, persona=None, domain=None, **kwargs: (chunks, sources),
        )
        monkeypatch.setattr(generate, "build_prompt", lambda q, ctx, **kwargs: "PROMPT")
        dummy_llm.set_responses("answer")

        # Call with custom temperature
        response = generate.answer("test query", collection=None, temperature=0.8)

        # Verify temperature in response
        assert response["temperature"] == 0.8
        # Verify LLM was created with custom temperature
        assert dummy_llm.temperature == 0.8

    def test_answer_temperature_defaults_to_config(self, monkeypatch, generate_module):
        generate, _, dummy_llm, _ = generate_module

        chunks = ["test chunk"]
        sources = [{"id": "1"}]
        monkeypatch.setattr(
            generate,
            "retrieve",
            lambda q, c, k=None, persona=None, domain=None, **kwargs: (chunks, sources),
        )
        monkeypatch.setattr(generate, "build_prompt", lambda q, ctx, **kwargs: "PROMPT")
        dummy_llm.set_responses("answer")

        # Call without temperature parameter - should use config default
        response = generate.answer("test query", collection=None)

        # Verify temperature from config is used
        assert response["temperature"] == generate.config.temperature
        assert dummy_llm.temperature == generate.config.temperature

    def test_answer_respects_custom_k_and_temperature(self, monkeypatch, generate_module):
        generate, _, dummy_llm, _ = generate_module

        chunks_10 = [f"chunk {i}" for i in range(10)]
        sources_10 = [{"id": str(i)} for i in range(10)]

        call_count = {"count": 0}
        original_retrieve = generate.retrieve

        def mock_retrieve(q, c, k=None, persona=None, domain=None, **kwargs):
            call_count["count"] += 1
            # Verify k parameter was passed correctly
            if k is not None:
                assert k == 10
            return chunks_10[:k] if k else chunks_10, sources_10[:k] if k else sources_10

        monkeypatch.setattr(generate, "retrieve", mock_retrieve)
        monkeypatch.setattr(generate, "build_prompt", lambda q, ctx, **kwargs: "PROMPT")
        dummy_llm.set_responses("answer")

        # Call with both custom k and temperature
        response = generate.answer("test query", collection=None, k=10, temperature=0.7)

        assert response["temperature"] == 0.7
        assert response["retrieval_count"] == 10
        assert call_count["count"] > 0
