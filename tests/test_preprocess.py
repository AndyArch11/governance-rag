"""Tests for preprocess module with mocked LLMs and logger.

Mocks external dependencies (Ollama LLM and centralised logger) to keep tests
fast and deterministic.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError


class DummyLogger:
    """Minimal logger stub for tests."""

    def __init__(self):
        self.messages = []

    def info(self, *args, **kwargs):
        self.messages.append(("info", args, kwargs))

    def warning(self, *args, **kwargs):
        self.messages.append(("warning", args, kwargs))

    def error(self, *args, **kwargs):
        self.messages.append(("error", args, kwargs))

    def debug(self, *args, **kwargs):
        self.messages.append(("debug", args, kwargs))

    def exception(self, *args, **kwargs):
        """Log exception (stores as 'error' for test compatibility)."""
        self.messages.append(("error", args, kwargs))


class DummyLLM:
    """Simple LLM stub that returns queued responses."""

    def __init__(self):
        self.responses = []

    def set_responses(self, *responses):
        self.responses = list(responses)

    def invoke(self, prompt: str):  # noqa: D401
        if self.responses:
            return self.responses.pop(0)
        return ""


@pytest.fixture()
def preprocess_module(monkeypatch):
    """Load preprocess with mocked dependencies."""

    # Ensure ingest package modules resolve
    scripts_ingest_path = Path(__file__).parent.parent / "scripts" / "ingest"
    sys.path.insert(0, str(scripts_ingest_path))

    # Stub langchain_ollama before importing preprocess
    dummy_langchain = types.ModuleType("langchain_ollama")

    class DummyOllamaLLM:  # noqa: D401
        def __init__(self, model=None):
            self.model = model

        def invoke(self, prompt: str):
            return ""

    dummy_langchain.OllamaLLM = DummyOllamaLLM
    monkeypatch.setitem(sys.modules, "langchain_ollama", dummy_langchain)

    # Ensure a clean import of preprocess with stubs
    sys.modules.pop("scripts.ingest.preprocess", None)
    preprocess = importlib.import_module("scripts.ingest.preprocess")

    # Override LLM instances and logger with controllable stubs
    dummy_logger = DummyLogger()
    preprocess.primary_llm = DummyLLM()
    preprocess.validator_llm = DummyLLM()
    monkeypatch.setattr(preprocess, "get_logger", lambda: dummy_logger)
    monkeypatch.setattr(preprocess, "audit", lambda *args, **kwargs: None)

    return preprocess, dummy_logger


def test_sanitise_for_json(preprocess_module):
    preprocess, _ = preprocess_module
    text = 'Quote: " and backslash \\'
    result = preprocess.sanitise_for_json(text)
    assert result == 'Quote: \\" and backslash \\\\'


def test_repair_json_missing_commas(preprocess_module):
    preprocess, _ = preprocess_module
    raw = '{\n  "a": "1"\n  "b": 2\n  "list": ["x"\n  "y"]\n}'
    repaired = preprocess.repair_json(raw)
    data = preprocess.json.loads(repaired)
    assert data == {"a": "1", "b": 2, "list": ["x", "y"]}


def test_extract_first_json_block_with_text(preprocess_module):
    preprocess, _ = preprocess_module
    text = 'Prefix text {"status": "ok", "value": 1} suffix'
    parsed = preprocess.extract_first_json_block(text)
    assert parsed == {"status": "ok", "value": 1}


def test_extract_first_json_block_repairs(preprocess_module):
    preprocess, _ = preprocess_module
    broken = 'Here {"a": 1\n "b": 2} there'  # missing comma after 1
    parsed = preprocess.extract_first_json_block(broken)
    assert parsed == {"a": 1, "b": 2}


def test_validate_metadata_and_summary(preprocess_module):
    preprocess, logger = preprocess_module
    metadata = {
        "doc_type": "policy",
        "key_topics": ["azure"],
        "summary": "A valid summary with enough words.",
    }
    validated = preprocess.validate_metadata(metadata)
    assert validated.doc_type == "policy"

    with pytest.raises(ValidationError):
        preprocess.validate_summary("too short")
    assert any(level == "error" for level, *_ in logger.messages)


def test_score_summary(preprocess_module):
    preprocess, _ = preprocess_module
    preprocess.validator_llm.set_responses(
        '{"relevance":8,"coverage":7,"clarity":9,"conciseness":8,"overall":8,"comment":"good"}'
    )
    scores = preprocess.score_summary("summary text", "cleaned text")
    assert scores["overall"] == 8
    assert scores["comment"] == "good"


def test_regenerate_summary(preprocess_module):
    preprocess, _ = preprocess_module
    preprocess.primary_llm.set_responses("New regenerated summary.")
    result = preprocess.regenerate_summary("cleaned text")
    assert result == "New regenerated summary."


def test_clean_text_with_llm(preprocess_module):
    preprocess, _ = preprocess_module
    preprocess.primary_llm.set_responses('He said: "hello" \\ test')
    cleaned = preprocess.clean_text_with_llm("raw")
    assert '\\"' in cleaned  # quotes escaped
    assert "\\\\" in cleaned  # backslashes escaped


def test_extract_metadata_with_llm(preprocess_module):
    preprocess, _ = preprocess_module
    preprocess.primary_llm.set_responses(
        '{"doc_type":"guide","key_topics":["k8s"],"summary":"Valid summary with enough words here."}'
    )
    metadata = preprocess.extract_metadata_with_llm("cleaned")
    assert metadata["doc_type"] == "guide"
    assert metadata["key_topics"] == ["k8s"]


def test_extract_metadata_with_source_category_governance(preprocess_module):
    """Test that source_category hint is included in prompt for Governance."""
    preprocess, _ = preprocess_module
    preprocess.primary_llm.set_responses(
        '{"doc_type":"governance","key_topics":["security"],"summary":"Valid summary with enough words here."}'
    )
    metadata = preprocess.extract_metadata_with_llm("cleaned", source_category="Governance")
    assert metadata["doc_type"] == "governance"
    # Verify LLM was called (source_category gets passed through)
    assert metadata["key_topics"] == ["security"]


def test_extract_metadata_with_source_category_patterns(preprocess_module):
    """Test that source_category hint is included in prompt for Patterns."""
    preprocess, _ = preprocess_module
    preprocess.primary_llm.set_responses(
        '{"doc_type":"architectural pattern","key_topics":["architecture"],"summary":"Valid summary with enough words here."}'
    )
    metadata = preprocess.extract_metadata_with_llm("cleaned", source_category="Patterns")
    assert metadata["doc_type"] == "architectural pattern"
    assert metadata["key_topics"] == ["architecture"]


def test_preprocess_text_end_to_end(monkeypatch, preprocess_module):
    preprocess, logger = preprocess_module

    # Prepare deterministic responses for each LLM call in pipeline
    preprocess.primary_llm.set_responses(
        "cleaned text output",  # clean_text_with_llm
        '{"doc_type":"policy","key_topics":["security"],"summary":"This is a valid summary containing many words."}',  # extract_metadata_with_llm
    )
    preprocess.validator_llm.set_responses(
        '{"relevance":7,"coverage":7,"clarity":7,"conciseness":7,"overall":7,"comment":"ok"}'  # score_summary
    )

    result = preprocess.preprocess_text("raw input text")

    assert result["cleaned_text"] == "cleaned text output"
    assert result["doc_type"] == "policy"
    assert result["key_topics"] == ["security"]
    assert result["summary"] == "This is a valid summary containing many words."
    assert result["summary_scores"]["overall"] == 7
    assert result["source_category"] is None
    # Ensure no low-quality regeneration path was triggered
    assert not any(level == "warning" for level, *_ in logger.messages)


def test_preprocess_text_with_governance_category(monkeypatch, preprocess_module):
    """Test preprocess_text with Governance source_category."""
    preprocess, logger = preprocess_module

    preprocess.primary_llm.set_responses(
        "cleaned governance text",
        '{"doc_type":"governance","key_topics":["compliance"],"summary":"This is a governance summary with many words here."}',
    )
    preprocess.validator_llm.set_responses(
        '{"relevance":8,"coverage":8,"clarity":8,"conciseness":8,"overall":8,"comment":"excellent"}'
    )

    result = preprocess.preprocess_text("raw governance text", source_category="Governance")

    assert result["cleaned_text"] == "cleaned governance text"
    assert result["doc_type"] == "governance"
    assert result["key_topics"] == ["compliance"]
    assert result["source_category"] == "Governance"
    assert result["summary_scores"]["overall"] == 8


def test_preprocess_text_with_patterns_category(monkeypatch, preprocess_module):
    """Test preprocess_text with Patterns source_category."""
    preprocess, logger = preprocess_module

    preprocess.primary_llm.set_responses(
        "cleaned pattern text",
        '{"doc_type":"architectural pattern","key_topics":["architecture"],"summary":"This is an architectural pattern summary with many words."}',
    )
    preprocess.validator_llm.set_responses(
        '{"relevance":8,"coverage":8,"clarity":8,"conciseness":8,"overall":8,"comment":"excellent"}'
    )

    result = preprocess.preprocess_text("raw pattern text", source_category="Patterns")

    assert result["cleaned_text"] == "cleaned pattern text"
    assert result["doc_type"] == "architectural pattern"
    assert result["source_category"] == "Patterns"
    assert result["summary_scores"]["overall"] == 8


def test_validate_summary_valid(preprocess_module):
    """Test validate_summary with valid summary."""
    preprocess, _ = preprocess_module

    valid_summary = "This is a valid summary with at least five words."
    validated = preprocess.validate_summary(valid_summary)

    assert validated.summary == valid_summary


def test_validate_summary_too_short(preprocess_module):
    """Test validate_summary rejects short summary."""
    preprocess, _ = preprocess_module

    with pytest.raises(ValidationError):
        preprocess.validate_summary("Too short")


def test_validate_summary_too_few_words(preprocess_module):
    """Test validate_summary rejects summary with too few words."""
    preprocess, _ = preprocess_module

    with pytest.raises(ValidationError):
        preprocess.validate_summary("Just three words")


def test_llm_invoke_with_rate_limit(preprocess_module):
    """Test llm_invoke_with_rate_limit calls LLM correctly."""
    preprocess, _ = preprocess_module

    # Mock LLM to return specific response
    preprocess.primary_llm.set_responses("LLM response text")

    result = preprocess.llm_invoke_with_rate_limit(preprocess.primary_llm, "test prompt")

    assert result == "LLM response text"


def test_extract_first_json_block_unbalanced_braces(preprocess_module):
    """Test extract_first_json_block with unbalanced braces."""
    preprocess, _ = preprocess_module

    unbalanced = 'Text {"key": "value"'  # Missing closing brace

    with pytest.raises(ValueError, match="Unbalanced JSON braces"):
        preprocess.extract_first_json_block(unbalanced)


def test_extract_first_json_block_no_json(preprocess_module):
    """Test extract_first_json_block with no JSON."""
    preprocess, _ = preprocess_module

    no_json = "This text has no JSON object at all"

    with pytest.raises(ValueError, match="No JSON object found"):
        preprocess.extract_first_json_block(no_json)


def test_repair_json_trailing_commas(preprocess_module):
    """Test repair_json removes trailing commas."""
    preprocess, _ = preprocess_module

    json_with_trailing = '{"a": 1, "b": 2,}'
    repaired = preprocess.repair_json(json_with_trailing)
    parsed = preprocess.json.loads(repaired)

    assert parsed == {"a": 1, "b": 2}


def test_repair_json_max_attempts_exceeded(preprocess_module):
    """Test repair_json fails after max attempts."""
    preprocess, logger = preprocess_module

    # Completely broken JSON that can't be repaired
    broken_json = '{"totally": broken json [[[[[['

    with pytest.raises(preprocess.json.JSONDecodeError):
        preprocess.repair_json(broken_json, max_attempts=2)

    # Verify error was logged
    assert any(level == "error" for level, *_ in logger.messages)
