"""Tests for local-only multimodal thesis figure assessment."""

import json as json_module

import pytest
from PIL import Image

from scripts.thesis_graph.figure_assessment import assess_figure_locally


class FakeResponse:
    def __init__(self, result):
        self.result = result

    def raise_for_status(self):
        return None

    def json(self):
        return self.result


def test_figure_assessment_uses_local_vision_model_and_reports_usage(monkeypatch):
    calls = []
    assessment = {
        "description": "A diagram of governance relationships.",
        "figure_type": "conceptual diagram",
        "caption_alignment": "consistent",
        "alt_text_quality": "adequate",
        "caption_description_alignment": "consistent",
        "alt_text_description_alignment": "inconsistent",
        "body_text_discussion": "interprets",
        "body_text_evidence": "This figure demonstrates how the framework connects governance roles.",
        "visible_text_in_image": "SACRED CEREMONY shown on the diagram",
        "visible_text_coverage": "complete",
        "non_text_visual_review_cues": [],
        "findings_or_claims": "The figure shows links between groups.",
        "limitations": "Text is too small to read.",
        "human_review_required": False,
    }

    def fake_post(url, *, json, timeout):
        calls.append((url, json, timeout))
        if url.endswith("/api/show"):
            return FakeResponse({"capabilities": ["completion", "vision"]})
        return FakeResponse(
            {
                "message": {"content": json_module.dumps(assessment)},
                "prompt_eval_count": 320,
                "eval_count": 55,
                "total_duration": 1200000,
            }
        )

    monkeypatch.setattr("scripts.thesis_graph.figure_assessment.requests.post", fake_post)
    image = Image.new("RGB", (2, 2), color="red")

    result = assess_figure_locally(
        {"image": image, "caption": "Figure 1", "alt_text": "A red square"},
        model="qwen3.6:27b",
        ollama_host="http://host.docker.internal:11434",
        sensitivity_terms=["sacred ceremony"],
        nearby_context="This figure demonstrates how the framework connects governance roles.",
    )
    assert calls[0][0] == "http://host.docker.internal:11434/api/show"
    assert calls[1][0] == "http://host.docker.internal:11434/api/chat"
    assert calls[1][1]["messages"][0]["images"]
    assert "caption_description_alignment" in calls[1][1]["messages"][0]["content"]
    assert "alt_text_description_alignment" in calls[1][1]["messages"][0]["content"]
    assert "body_text_discussion" in calls[1][1]["messages"][0]["content"]
    assert "body_text_evidence" in calls[1][1]["messages"][0]["content"]
    assert "visible_text_in_image" in calls[1][1]["messages"][0]["content"]
    assert "visible_text_coverage" in calls[1][1]["messages"][0]["content"]
    assert "non_text_visual_review_cues" in calls[1][1]["messages"][0]["content"]
    assert "This cue list is not a sensitivity judgement" in calls[1][1]["messages"][0]["content"]
    assert (
        "If a supplied term is legible, do not create a figure description"
        in calls[1][1]["messages"][0]["content"]
    )
    assert (
        "do not count the caption as body-text discussion" in calls[1][1]["messages"][0]["content"]
    )
    assert result["human_review_required"] is True
    assert result["caption_description_alignment"] == "consistent"
    assert result["alt_text_description_alignment"] == "inconsistent"
    assert result["body_text_discussion"] == "interprets"
    assert result["body_text_evidence"] == assessment["body_text_evidence"]
    assert result["sensitivity_term_matches"] == ["sacred ceremony"]
    assert result["sensitivity_screen_status"] == "matched"
    assert result["description"] == ""
    assert result["findings_or_claims"] == ""
    assert result["token_usage"] == {
        "input_tokens": 320,
        "output_tokens": 55,
        "total_duration_ns": 1200000,
        "token_source": "reported",
        "model": "qwen3.6:27b",
    }
    absent_text_result = assess_figure_locally(
        {"image": image},
        model="qwen3.6:27b",
        ollama_host="http://host.docker.internal:11434",
    )
    assert absent_text_result["caption_description_alignment"] == "not_provided"
    assert absent_text_result["alt_text_description_alignment"] == "not_provided"
    assert absent_text_result["caption_alignment"] == "caption_missing"
    assert absent_text_result["alt_text_quality"] == "missing"
    assert absent_text_result["body_text_discussion"] == "context_unavailable"
    assert absent_text_result["body_text_evidence"] == ""
    assert absent_text_result["sensitivity_term_matches"] == []
    assert absent_text_result["sensitivity_screen_status"] == "not_requested"

    assessment.pop("visible_text_in_image")
    assessment.pop("visible_text_coverage")
    assessment.pop("body_text_discussion")
    assessment.pop("body_text_evidence")
    assessment["description"] = "Description withheld for an inconclusive sensitivity screen."
    assessment["findings_or_claims"] = "Potential finding withheld."
    incomplete_contract_result = assess_figure_locally(
        {"image": image, "caption": "Figure 1"},
        model="qwen3.6:27b",
        ollama_host="http://host.docker.internal:11434",
        sensitivity_terms=["sacred ceremony"],
    )
    assert incomplete_contract_result["visible_text_coverage"] == "unclear"
    assert incomplete_contract_result["sensitivity_screen_status"] == "incomplete"
    assert incomplete_contract_result["assessment_status"] == "incomplete_response"
    assert incomplete_contract_result["body_text_discussion"] == "context_unavailable"
    assert incomplete_contract_result["description"] == ""
    assert incomplete_contract_result["findings_or_claims"] == ""

    assessment["caption_description_alignment"] = "consistent"
    assessment["visible_text_in_image"] = "An artisan craft label"
    unrelated_term_result = assess_figure_locally(
        {"image": image, "caption": "Figure 1"},
        model="qwen3.6:27b",
        ollama_host="http://host.docker.internal:11434",
        sensitivity_terms=["art"],
    )
    assert unrelated_term_result["sensitivity_term_matches"] == []

    assessment["non_text_visual_review_cues"] = ["person_or_face"]
    assessment["description"] = "A person appears beside the diagram."
    visual_review_result = assess_figure_locally(
        {"image": image, "caption": "Figure 1"},
        model="qwen3.6:27b",
        ollama_host="http://host.docker.internal:11434",
    )
    assert visual_review_result["non_text_visual_review_cues"] == ["person_or_face"]
    assert visual_review_result["non_text_visual_review_required"] is True
    assert visual_review_result["description"] == ""
    assert visual_review_result["findings_or_claims"] == ""

    assessment["non_text_visual_review_cues"] = ["sacred ceremonial object"]
    assessment["description"] = "A sacred ceremonial object."
    unrecognised_visual_cue_result = assess_figure_locally(
        {"image": image, "caption": "Figure 1"},
        model="qwen3.6:27b",
        ollama_host="http://host.docker.internal:11434",
    )
    assert unrecognised_visual_cue_result["non_text_visual_review_cues"] == [
        "unclear_non_text_content"
    ]
    assert unrecognised_visual_cue_result["non_text_visual_review_required"] is True
    assert unrecognised_visual_cue_result["description"] == ""

    assessment["visible_text_in_image"] = "Partly legible text"
    assessment["visible_text_coverage"] = "partial"
    incomplete_screen_result = assess_figure_locally(
        {"image": image, "caption": "Figure 1"},
        model="qwen3.6:27b",
        ollama_host="http://host.docker.internal:11434",
        sensitivity_terms=["sacred ceremony"],
    )
    assert incomplete_screen_result["sensitivity_term_matches"] == []
    assert incomplete_screen_result["sensitivity_screen_status"] == "incomplete"
    assert incomplete_screen_result["description"] == ""
    assert incomplete_screen_result["findings_or_claims"] == ""

    assessment["caption_description_alignment"] = "unsupported"
    invalid_category_result = assess_figure_locally(
        {"image": image, "caption": "Figure 1", "alt_text": "A red square"},
        model="qwen3.6:27b",
        ollama_host="http://host.docker.internal:11434",
    )
    assert invalid_category_result["caption_description_alignment"] == "unclear"
    assert invalid_category_result["assessment_status"] == "incomplete_response"


def test_figure_assessment_rejects_non_local_endpoint(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("A non-local endpoint must be rejected before a request")

    monkeypatch.setattr("scripts.thesis_graph.figure_assessment.requests.post", fail_if_called)
    with pytest.raises(ValueError, match="local Ollama host"):
        assess_figure_locally(
            {"image": Image.new("RGB", (1, 1))},
            model="qwen3.6:27b",
            ollama_host="https://example.invalid",
        )


def test_figure_assessment_rejects_model_without_vision(monkeypatch):
    monkeypatch.setattr(
        "scripts.thesis_graph.figure_assessment.requests.post",
        lambda *args, **kwargs: FakeResponse({"capabilities": ["completion"]}),
    )
    with pytest.raises(ValueError, match="does not report vision"):
        assess_figure_locally(
            {"image": Image.new("RGB", (1, 1))},
            model="text-only",
            ollama_host="http://localhost:11434",
        )
