"""Local-only multimodal assessment for thesis figures."""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any
from urllib.parse import urlparse

import requests

_ALLOWED_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}
_ALLOWED_NON_TEXT_VISUAL_CUES = {
    "person_or_face",
    "visible_injury_or_medical_content",
    "document_or_identifier",
    "unclear_non_text_content",
}


def assess_figure_locally(
    figure: dict[str, Any],
    *,
    model: str,
    ollama_host: str,
    nearby_context: str = "",
    sensitivity_terms: list[str] | None = None,
    timeout: int = 240,
) -> dict[str, Any]:
    """Assess one figure using only a vision-capable local Ollama endpoint.

    The response remains evidence for human review, not an authoritative
    assessment. Token counts and Ollama duration fields are returned to callers
    for observability.

    Args:
        figure: Dictionary containing figure data, including image bytes, caption, and alt text
        model: Name of the local Ollama model to use for assessment
        ollama_host: URL of the local Ollama host
        nearby_context: Optional nearby thesis text to provide context for the assessment
        sensitivity_terms: Profile terms checked only against text legible inside the image
        timeout: Request timeout in seconds (default: 240)

    Returns:
        Dictionary containing the assessment results, including description, figure_type,
        caption_alignment, alt_text_quality, findings_or_claims, limitations, human_review_required,
        and token usage information.
    """
    parsed_url = urlparse(ollama_host)
    if parsed_url.hostname not in _ALLOWED_LOCAL_HOSTS:
        raise ValueError("Figure assessment only supports a local Ollama host.")
    base_url = ollama_host.rstrip("/")

    show_response = requests.post(
        f"{base_url}/api/show",
        json={"model": model},
        timeout=timeout,
    )
    show_response.raise_for_status()
    model_info = show_response.json()
    if "vision" not in model_info.get("capabilities", []):
        raise ValueError(f"Configured figure model {model!r} does not report vision capability.")

    image_bytes = figure.get("image_bytes")
    image = figure.get("image")
    if image_bytes is None and image is not None:
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        image_bytes = image_buffer.getvalue()
    if not image_bytes:
        raise ValueError("Figure has no locally available image data.")

    prompt = (
        "First describe the image using only visible evidence, without relying on the caption "
        "or alt text. Then compare your description with each supplied text separately. "
        "Do not infer cultural meaning, identity, or restricted knowledge. Return JSON with "
        "string fields description, figure_type, caption_alignment, alt_text_quality, "
        "caption_description_alignment, alt_text_description_alignment, findings_or_claims, "
        "limitations, body_text_discussion, body_text_evidence, visible_text_in_image, "
        "visible_text_coverage, non_text_visual_review_cues as an array, and boolean "
        "human_review_required set to true. The nearby thesis excerpt excludes the caption; "
        "do not count the caption as body-text discussion. Assess only that excerpt: "
        "classify it as interprets, describes, mentions_only, no_discussion_in_context, "
        "unclear, or context_unavailable. Include a short verbatim quote in "
        "body_text_evidence for positive classifications, or an empty string if no quote "
        "supports one. Do not infer that the whole thesis lacks discussion from this excerpt. "
        "Transcribe legible text physically inside the image in visible_text_in_image only; "
        "exclude supplied caption, alt text and nearby thesis text. Do not infer sensitivity "
        "from visual concepts. Separately, list only directly visible non-text features that may "
        "warrant human review using these exact cue values: person_or_face, "
        "visible_injury_or_medical_content, document_or_identifier, or "
        "unclear_non_text_content. Do not infer identity, sacredness, ceremony, restricted "
        "knowledge, or any cultural meaning; use an empty array when no listed cue is clearly "
        "visible. This cue list is not a sensitivity judgement. The supplied terms are matched "
        "exactly against this transcription. "
        "Set visible_text_coverage to complete if all visible text is legible or no text exists, "
        "partial if some visible text is unreadable, or unclear if coverage cannot be judged. "
        "If a supplied term is legible, do not create a figure description or findings; leave "
        "those fields empty for human review. If sensitivity terms are supplied and coverage is "
        "partial or unclear, also leave description and findings empty for human review. "
        "Use caption_alignment "
        "values consistent, inconsistent, unclear, or caption_missing; use alt_text_quality "
        "values adequate, inadequate, missing, or unclear. Use caption_description_alignment "
        "and alt_text_description_alignment values consistent, inconsistent, unclear, or "
        "not_provided; use not_provided when the corresponding text is absent.\n\n"
        f"Caption: {figure.get('caption') or '[none]'}\n"
        f"Alt text: {figure.get('alt_text') or '[none]'}\n"
        f"Sensitivity terms: {json.dumps(sensitivity_terms or [])}\n"
        f"Nearby thesis text: {nearby_context[:2000] or '[none]'}"
    )
    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                }
            ],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    content = result.get("message", {}).get("content", "")
    try:
        assessment = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Local vision model returned invalid JSON for figure assessment.") from exc

    if not isinstance(assessment, dict):
        raise ValueError("Local vision model response must be a JSON object for figure assessment.")

    response_incomplete = False
    optional_text_defaults = {
        "description": "",
        "figure_type": "unclear",
        "caption_alignment": "unclear",
        "alt_text_quality": "unclear",
        "caption_description_alignment": "unclear",
        "alt_text_description_alignment": "unclear",
        "findings_or_claims": "",
        "limitations": "",
        "body_text_discussion": "context_unavailable" if not nearby_context.strip() else "unclear",
        "body_text_evidence": "",
        "visible_text_in_image": "",
        "visible_text_coverage": "unclear",
    }
    for field, default in optional_text_defaults.items():
        if not isinstance(assessment.get(field), str):
            assessment[field] = default
            response_incomplete = True

    if not assessment["description"].strip() or not assessment["figure_type"].strip():
        response_incomplete = True
    visual_cues = assessment.get("non_text_visual_review_cues")
    if not isinstance(visual_cues, list):
        visual_cues = []
        response_incomplete = True
    valid_visual_cues = []
    for cue in visual_cues:
        if isinstance(cue, str) and cue in _ALLOWED_NON_TEXT_VISUAL_CUES:
            valid_visual_cues.append(cue)
        else:
            response_incomplete = True
            valid_visual_cues.append("unclear_non_text_content")
    assessment["non_text_visual_review_cues"] = list(dict.fromkeys(valid_visual_cues))
    assessment["non_text_visual_review_required"] = bool(valid_visual_cues)
    if valid_visual_cues:
        assessment["description"] = ""
        assessment["findings_or_claims"] = ""
    allowed_values = {
        "caption_alignment": {"consistent", "inconsistent", "unclear", "caption_missing"},
        "alt_text_quality": {"adequate", "inadequate", "missing", "unclear"},
        "caption_description_alignment": {
            "consistent",
            "inconsistent",
            "unclear",
            "not_provided",
        },
        "alt_text_description_alignment": {
            "consistent",
            "inconsistent",
            "unclear",
            "not_provided",
        },
        "body_text_discussion": {
            "interprets",
            "describes",
            "mentions_only",
            "no_discussion_in_context",
            "unclear",
            "context_unavailable",
        },
        "visible_text_coverage": {"complete", "partial", "unclear"},
    }
    for field, values in allowed_values.items():
        if assessment[field] not in values:
            assessment[field] = "unclear"
            response_incomplete = True
    assessment["assessment_status"] = "incomplete_response" if response_incomplete else "complete"
    visible_text = " ".join(re.findall(r"\w+", assessment.pop("visible_text_in_image").casefold()))
    sensitivity_term_matches = []
    for term in sensitivity_terms or []:
        normalised_term = " ".join(re.findall(r"\w+", term.casefold()))
        if normalised_term and f" {normalised_term} " in f" {visible_text} ":
            sensitivity_term_matches.append(term)
    assessment["sensitivity_term_matches"] = list(dict.fromkeys(sensitivity_term_matches))
    if not sensitivity_terms:
        sensitivity_screen_status = "not_requested"
    elif assessment["sensitivity_term_matches"]:
        sensitivity_screen_status = "matched"
    elif assessment["visible_text_coverage"] == "complete":
        sensitivity_screen_status = "clear"
    else:
        sensitivity_screen_status = "incomplete"
    assessment["sensitivity_screen_status"] = sensitivity_screen_status
    if assessment["sensitivity_term_matches"] or sensitivity_screen_status == "incomplete":
        assessment["description"] = ""
        assessment["findings_or_claims"] = ""
    if not figure.get("caption"):
        assessment["caption_alignment"] = "caption_missing"
        assessment["caption_description_alignment"] = "not_provided"
    if not figure.get("alt_text"):
        assessment["alt_text_quality"] = "missing"
        assessment["alt_text_description_alignment"] = "not_provided"
    body_text_evidence = " ".join(assessment["body_text_evidence"].split())
    normalised_context = " ".join(nearby_context.casefold().split())
    normalised_evidence = body_text_evidence.casefold()
    if not normalised_context:
        assessment["body_text_discussion"] = "context_unavailable"
        assessment["body_text_evidence"] = ""
    elif assessment["body_text_discussion"] == "context_unavailable":
        assessment["body_text_discussion"] = "unclear"
    elif normalised_evidence and normalised_evidence not in normalised_context:
        assessment["body_text_discussion"] = "unclear"
        assessment["body_text_evidence"] = ""
    elif (
        assessment["body_text_discussion"] in {"interprets", "describes", "mentions_only"}
        and not body_text_evidence
    ):
        assessment["body_text_discussion"] = "unclear"
    assessment["human_review_required"] = True
    assessment["token_usage"] = {
        "input_tokens": int(result.get("prompt_eval_count") or 0),
        "output_tokens": int(result.get("eval_count") or 0),
        "total_duration_ns": int(result.get("total_duration") or 0),
        "token_source": (
            "reported"
            if result.get("prompt_eval_count") is not None or result.get("eval_count") is not None
            else "unavailable"
        ),
        "model": model,
    }
    return assessment
