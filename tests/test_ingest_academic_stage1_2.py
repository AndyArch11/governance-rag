"""Tests for academic ingestion stages 1/2 (load + citation extraction)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ingest.academic.parser import (
    _repair_citation_punctuation_spacing,
    _repair_cojoined_title_words,
    extract_citations,
)
from scripts.ingest.ingest_academic import (
    _assess_thesis_figures,
    stage_extract_citations,
    stage_load_document,
)


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, msg):
        self.infos.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)

    def error(self, msg):
        self.errors.append(msg)


class DummyConfig:
    def __init__(self, max_pdf_size_mb=50):
        self.max_pdf_size_mb = max_pdf_size_mb


def test_extract_citations_from_references_section():
    text = """
    Introduction text.

    References
    [1] Smith, J. (2020). Example Paper. https://doi.org/10.1000/182
    [2] Doe, A. Another Paper. doi:10.5555/12345678
    """
    citations = extract_citations(text)
    assert len(citations) == 2
    assert citations[0].doi == "10.1000/182"
    assert citations[1].doi == "10.5555/12345678"


def test_extract_citations_without_reference_section():
    text = "No references here."
    citations = extract_citations(text)
    assert citations == []


def test_stage_extract_citations_returns_raw_texts():
    text = """
    References
    1. Example Ref. https://doi.org/10.1234/abcd
    """
    logger = DummyLogger()
    raw = stage_extract_citations(text, logger)
    assert raw == ["1. Example Ref. https://doi.org/10.1234/abcd"]
    assert any("Extracted" in msg for msg in logger.infos)


def test_extract_citations_handles_multiline_blocks():
    text = """
    References
    Smith, J. (2020). Example Paper.
    Journal of Testing, 12(3), 45-67.

    Doe, A. (2019). Another Paper.
    Proceedings of Example Conf, 101-110.
    """
    citations = extract_citations(text)
    assert len(citations) >= 2


def test_extract_citations_splits_docling_hyphen_boundaries_and_repairs_titles():
    """Compressed bibliography text produces separate, provider-searchable references."""
    text = """
    References
    Adam,B.(1994). Timeandsocialtheory. Polity Press.-Adams,K.(2018).
    Challengingthecolonisationofbirth. Women and Birth,31(2),81-88.
    https://doi.org/10.1016/j.wombi.2017.07.014-
    """

    citations = extract_citations(text)

    assert len(citations) == 2
    assert "Time and social theory" in citations[0].raw_text
    assert citations[0].doi is None
    assert "Challenging the colonisation of birth" in citations[1].raw_text
    assert citations[1].doi == "10.1016/j.wombi.2017.07.014"


def test_extract_citations_splits_reference_after_trailing_doi_hyphen():
    """Docling's URL trailing hyphen cannot merge the following author reference."""
    text = """
    References
    Bell,A.(2014a). First title. Journal,24(4),506-516.https://doi.org/10.1177/1049732314524637-
    Bell,A.(2014b). Second title. Rutgers University Press.
    """

    citations = extract_citations(text)

    assert len(citations) == 2
    assert citations[0].doi == "10.1177/1049732314524637"
    assert "Second title" in citations[1].raw_text


def test_extract_citations_splits_dash_bullets_and_legal_references():
    """Docling can flatten bibliography bullets into dash-separated references."""
    text = """
    References
    - Aboriginals Protection and Restriction of the Sale of Opium Act 1897 (Qld). https://nla.gov.au/nla.obj-54468125/view?partId=nla.obj-54470690#page/n0/mode/1up-Aborigines Protection Act1909(NSW).https://www.austlii.edu.au/au/legis/nsw/num_act/apa1909n25262.pdf - Aboulghar, M. (2009). Prevention of OHSS. Reproductive BioMedicine Online, 19(1), 33-42. https://doi.org/10.1016/S1472-6483(10)60043-0-
    """

    citations = extract_citations(text)

    assert len(citations) == 3
    assert "Opium Act 1897" in citations[0].raw_text
    assert not citations[0].raw_text.startswith("-")
    assert "partId=" in citations[0].raw_text
    assert "Aborigines Protection Act1909" in citations[1].raw_text
    assert "num_act" in citations[1].raw_text
    assert "Prevention of OHSS" in citations[2].raw_text
    assert citations[2].doi == "10.1016/S1472-6483(10)60043-0"


def test_extract_citations_splits_unicode_initial_and_apostrophe_author_boundaries():
    """Reference boundaries may start with accented initials or apostrophe surnames."""
    text = """
    References
    DurkheimÉ. (1995). The elementary forms of religious life. Free Press. (Original work published 1912) - Durkheim, É. (2008). The division of labor in society. Free Press. (Original work published 1893)
    Nyinawagaba, B. (2024). Reproductive rights as fundamental human rights. Journal, 5(2), 9-14. - O'Donnell, E., Jackson, S., Langton, M., & Godden, L. (2022). Racialized water governance. Australasian Journal of Water Resources, 26(1), 59-71. https://doi.org/10.1080/13241583.2022.2049053-O'Donoghue,L.(1993). Aboriginal families and ATSIC. Family Matters, (35), 14-15.
    """

    citations = extract_citations(text)

    assert len(citations) == 5
    assert "The elementary forms" in citations[0].raw_text
    assert "The division of labor" in citations[1].raw_text
    assert "Reproductive rights" in citations[2].raw_text
    assert "Racialized water governance" in citations[3].raw_text
    assert "Aboriginal families and ATSIC" in citations[4].raw_text


def test_extract_citations_splits_url_before_title_style_reference():
    """A URL ending in a hyphen must not absorb a following title-style reference."""
    text = """
    References
    The Australian Women's Weekly. (1980). Australia's first test tube baby: At the birth of a miracle. The Australian Women's Weekly. https://trove.nla.gov.au/newspaper/article/55473228-TheHolyBible:NewInternationalVersion.(2011).Zondervan.https://www.biblegateway.com/passage/?search=Genesis%203:16&version=NIV
    """

    citations = extract_citations(text)

    assert len(citations) == 2
    assert "The Australian Women's Weekly" in citations[0].raw_text
    assert "The Holy Bible" in citations[1].raw_text


def test_extract_citations_keeps_legal_resource_without_author_pattern():
    """Identifier-bearing legislation remains a citation without an author/year prefix."""
    text = """
    References
    Aborigines Protection Act 1909 (NSW). https://www.austlii.edu.au/au/legis/nsw/num_act/apa1909n25262.pdf
    """

    citations = extract_citations(text)

    assert len(citations) == 1
    assert "Aborigines Protection Act 1909" in citations[0].raw_text


def test_extract_citations_repairs_short_title_words_joined_after_punctuation():
    """Variable kerning can join short title words across comma and question-mark boundaries."""
    text = """
    References
    Cannold,L.(2005). What,nobaby?Whywomenarelosingthefreedomtomother,andhowtheycangetitback.
    Curtin University Books.
    """

    citations = extract_citations(text)

    assert len(citations) == 1
    assert (
        "What, no baby?Why women are losing the freedom to mother, and how they can get it back"
        in citations[0].raw_text
    )


def test_repairs_long_cojoined_title_with_single_letter_word():
    """A valid one-letter word must not invalidate an otherwise useful title split."""
    title = (
        "Understandingpsychologicalsymptomsofendometriosisfromaresearchdomaincriteriaperspective"
    )

    repaired = _repair_cojoined_title_words(title)

    assert repaired == (
        "Understanding psychological symptoms of endometriosis from a research domain criteria perspective"
    )


def test_repairs_long_cojoined_title_with_possessive_apostrophe():
    """A possessive apostrophe does not prevent title-word segmentation."""
    title = "Women'slivedexperienceofinfertilityafterunsuccessfulmedicalintervention."

    repaired = _repair_cojoined_title_words(title)

    assert (
        repaired
        == "Women's lived experience of infertility after unsuccessful medical intervention."
    )


def test_repairs_long_cojoined_title_with_embedded_numerals():
    """Embedded title numerals do not prevent word segmentation."""
    title = "Life20yearsafterunsuccessfulinfertilitytreatment"

    repaired = _repair_cojoined_title_words(title)

    assert repaired == "Life 20 years after unsuccessful infertility treatment"


def test_repairs_cojoined_parenthetical_title_without_losing_parentheses():
    """Parenthetical title text is segmented without dropping citation punctuation."""
    title = "wanting(ornotwanting)tohavechildren"

    repaired = _repair_cojoined_title_words(title)

    assert repaired == "wanting (or not wanting) to have children"


def test_repairs_citation_punctuation_spacing_without_mutating_urls():
    """Prose punctuation is spaced while names and URLs retain their valid forms."""
    citation = (
        "O’Sullivan,J.&Brown,A.(2024). Women's health: Harper &amp; Row."
        "https://doi.org/10.1000/example:123?source=one&format=pdf"
    )

    repaired = _repair_citation_punctuation_spacing(citation)

    assert "O’Sullivan, J." in repaired
    assert "J. & Brown, A." in repaired
    assert "Women's health: Harper &amp; Row." in repaired
    assert "https://doi.org/10.1000/example:123?source=one&format=pdf" in repaired


def test_extract_citations_ignores_toc_heading_and_uses_later_section():
    text = """
    Table of Contents
    References.................................................... 12
    Chapter 1 Introduction........................................ 1

    References
    Smith, J. (2020). Example Paper. Journal of Testing, 12(3), 45-67.
    Doe, A. (2019). Another Paper. Proceedings of Example Conf, 101-110.
    """
    citations = extract_citations(text)
    assert len(citations) == 2


def test_stage_load_document_skips_missing(tmp_path):
    logger = DummyLogger()
    config = DummyConfig()
    missing_path = tmp_path / "missing.pdf"
    result = stage_load_document(missing_path, config, logger)
    assert result is None
    assert any("Document not found" in msg for msg in logger.warnings)


def test_stage_load_document_skips_non_pdf(tmp_path):
    logger = DummyLogger()
    config = DummyConfig()
    non_pdf = tmp_path / "doc.txt"
    non_pdf.write_text("hello")
    result = stage_load_document(non_pdf, config, logger)
    assert result is None
    assert any("Skipping non-PDF" in msg for msg in logger.warnings)


def test_stage_load_document_skips_oversize(tmp_path, monkeypatch):
    logger = DummyLogger()
    config = DummyConfig(max_pdf_size_mb=1)
    pdf_path = tmp_path / "big.pdf"
    pdf_path.write_bytes(b"x" * 2 * 1024 * 1024)  # 2MB
    result = stage_load_document(pdf_path, config, logger)
    assert result is None
    assert any("exceeds limit" in msg for msg in logger.warnings)


def test_stage_load_document_extracts_text(tmp_path, monkeypatch):
    logger = DummyLogger()
    config = DummyConfig()
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    def fake_extract_text(path):
        assert Path(path) == pdf_path
        return "PDF text"

    monkeypatch.setattr("scripts.ingest.ingest_academic.extract_text_from_pdf", fake_extract_text)

    result = stage_load_document(pdf_path, config, logger)
    assert result == "PDF text"


def test_stage_load_document_collects_figure_assets_without_changing_text_result(
    tmp_path, monkeypatch
):
    logger = DummyLogger()
    config = DummyConfig()
    pdf_path = tmp_path / "thesis.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    figure_assets = []
    extracted_figures = [{"figure_number": 1, "caption": "Figure caption", "image": object()}]
    monkeypatch.setattr(
        "scripts.ingest.ingest_academic.extract_pdf_text_and_figures",
        lambda path: ("PDF text", extracted_figures),
    )

    result = stage_load_document(pdf_path, config, logger, figures_out=figure_assets)

    assert result == "PDF text"
    assert figure_assets == extracted_figures


def test_figure_assessment_uses_local_model_and_flags_sensitive_captions(
    monkeypatch,
):
    logger = DummyLogger()
    config = SimpleNamespace(
        figure_assessment_enabled=True,
        vision_model_name="qwen3.6:27b",
        ollama_host="http://host.docker.internal:11434",
        vision_assessment_timeout=240,
    )
    figures = [
        {"figure_number": 1, "caption": "Community wellbeing model", "image": object()},
        {"figure_number": 2, "caption": "Sacred ceremony", "image": object()},
    ]
    assessments = []
    audit_events = []

    def fake_assess(figure, **kwargs):
        assessments.append(kwargs)
        return {
            "description": "A locally assessed figure.",
            "human_review_required": True,
            "token_usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "token_source": "reported",
                "model": kwargs["model"],
            },
        }

    monkeypatch.setattr("scripts.ingest.ingest_academic.assess_figure_locally", fake_assess)
    monkeypatch.setattr(
        "scripts.ingest.ingest_academic.audit",
        lambda event, data: audit_events.append((event, data)),
    )
    cultural_profile = {
        "sensitivities": [
            {
                "id": "ceremony",
                "applies_to": ["images", "captions"],
                "detection_terms": ["sacred ceremony"],
                "handling": "flag_for_review",
            }
        ]
    }

    _assess_thesis_figures(
        figures,
        "Before prose. Community wellbeing model After prose.",
        cultural_profile,
        config,
        logger,
    )

    assert len(assessments) == 1
    assert assessments[0]["model"] == "qwen3.6:27b"
    assert assessments[0]["ollama_host"] == "http://host.docker.internal:11434"
    assert assessments[0]["nearby_context"] == "Before prose.\nAfter prose."
    assert "Community wellbeing model" not in assessments[0]["nearby_context"]
    assert figures[0]["vision_status"] == "human_review_required"
    assert figures[0]["description"] == "A locally assessed figure."
    assert figures[1]["vision_status"] == "flagged_for_human_review"
    usage_events = [data for event, data in audit_events if event == "llm_usage"]
    assert usage_events[0]["operation"] == "vision.figure_assessment"
    assert usage_events[0]["input_tokens"] == 100
    assert usage_events[0]["output_tokens"] == 20


def test_figure_assessment_includes_distant_explicit_discussion(monkeypatch):
    logger = DummyLogger()
    config = SimpleNamespace(
        figure_assessment_enabled=True,
        vision_model_name="qwen3.6:27b",
        ollama_host="http://host.docker.internal:11434",
        vision_assessment_timeout=240,
    )
    figure = {
        "figure_number": 3,
        "caption_number": "3",
        "caption": "Figure 3: Community framework",
        "image": object(),
    }
    captured_contexts = []

    def fake_assess(_figure, **kwargs):
        captured_contexts.append(kwargs["nearby_context"])
        return {
            "description": "A community framework diagram.",
            "human_review_required": True,
            "token_usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "token_source": "reported",
                "model": kwargs["model"],
            },
        }

    monkeypatch.setattr("scripts.ingest.ingest_academic.assess_figure_locally", fake_assess)
    thesis_text = (
        "Figure 3: Community framework\n\n"
        + ("Unrelated background. " * 100)
        + "\n\nList of Figures\nFigure 3: Community framework ........ 14"
        + "\n\nFigure 3 demonstrates how community governance shapes the study's findings."
        + "\n\nFig. 3 also illustrates how those relationships influence the discussion."
    )

    _assess_thesis_figures([figure], thesis_text, None, config, logger)

    assert "Figure 3 demonstrates how community governance" in captured_contexts[0]
    assert "Fig. 3 also illustrates how those relationships" in captured_contexts[0]
    assert "Figure 3: Community framework" not in captured_contexts[0]
    assert "........ 14" not in captured_contexts[0]


def test_image_text_sensitivity_flags_and_redacts_visual_assessment(monkeypatch):
    logger = DummyLogger()
    config = SimpleNamespace(
        figure_assessment_enabled=True,
        vision_model_name="qwen3.6:27b",
        ollama_host="http://host.docker.internal:11434",
        vision_assessment_timeout=240,
    )
    figure = {
        "figure_number": 1,
        "caption": "Community governance framework",
        "alt_text": "A diagram of governance roles",
        "image": object(),
    }
    assessment_calls = []
    audit_events = []

    def fake_assess(figure, **kwargs):
        assessment_calls.append(kwargs)
        return {
            "description": "A sensitive image description that must not be persisted.",
            "sensitivity_term_matches": ["sacred ceremony"],
            "token_usage": {
                "input_tokens": 120,
                "output_tokens": 25,
                "token_source": "reported",
                "model": kwargs["model"],
            },
        }

    monkeypatch.setattr("scripts.ingest.ingest_academic.assess_figure_locally", fake_assess)
    monkeypatch.setattr(
        "scripts.ingest.ingest_academic.audit",
        lambda event, data: audit_events.append((event, data)),
    )
    profile = {
        "sensitivities": [
            {
                "id": "ceremony",
                "applies_to": ["images"],
                "detection_terms": ["sacred ceremony"],
            }
        ]
    }

    _assess_thesis_figures([figure], "Community governance framework", profile, config, logger)

    assert assessment_calls[0]["sensitivity_terms"] == ["sacred ceremony"]
    assert figure["vision_status"] == "flagged_for_human_review"
    assert figure["sensitivity_ids"] == ["ceremony"]
    assert "description" not in figure
    assert "sensitive image description" not in repr(figure).casefold()
    assert figure["vision_assessment"] == {
        "human_review_required": True,
        "sensitivity_screen": "legible_image_text",
        "sensitivity_ids": ["ceremony"],
        "token_usage": {
            "input_tokens": 120,
            "output_tokens": 25,
            "token_source": "reported",
            "model": "qwen3.6:27b",
        },
    }
    assert any(event == "vision_figure_flagged" for event, _ in audit_events)
    assert any(event == "llm_usage" for event, _ in audit_events)


def test_incomplete_image_text_sensitivity_screen_requires_review(monkeypatch):
    logger = DummyLogger()
    config = SimpleNamespace(
        figure_assessment_enabled=True,
        vision_model_name="qwen3.6:27b",
        ollama_host="http://host.docker.internal:11434",
        vision_assessment_timeout=240,
    )
    figure = {"figure_number": 2, "caption": "Figure caption", "image": object()}
    audit_events = []

    monkeypatch.setattr(
        "scripts.ingest.ingest_academic.assess_figure_locally",
        lambda figure, **kwargs: {
            "description": "Description withheld because OCR coverage is incomplete.",
            "sensitivity_term_matches": [],
            "sensitivity_screen_status": "incomplete",
            "token_usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "token_source": "reported",
                "model": kwargs["model"],
            },
        },
    )
    monkeypatch.setattr(
        "scripts.ingest.ingest_academic.audit",
        lambda event, data: audit_events.append((event, data)),
    )
    profile = {
        "sensitivities": [
            {
                "id": "ceremony",
                "applies_to": ["images"],
                "detection_terms": ["sacred ceremony"],
            }
        ]
    }

    _assess_thesis_figures([figure], "Figure caption", profile, config, logger)

    assert figure["vision_status"] == "flagged_for_human_review"
    assert figure["sensitivity_ids"] == ["ceremony"]
    assert "description" not in figure
    assert figure["vision_assessment"]["sensitivity_screen"] == "incomplete_legible_image_text"
    assert any(
        event == "vision_figure_flagged"
        and data["sensitivity_source"] == "incomplete_legible_image_text"
        for event, data in audit_events
    )


def test_non_text_visual_cue_flags_generic_human_review(monkeypatch):
    logger = DummyLogger()
    config = SimpleNamespace(
        figure_assessment_enabled=True,
        vision_model_name="qwen3.6:27b",
        ollama_host="http://host.docker.internal:11434",
        vision_assessment_timeout=240,
    )
    figure = {"figure_number": 4, "caption": "Figure caption", "image": object()}
    audit_events = []

    monkeypatch.setattr(
        "scripts.ingest.ingest_academic.assess_figure_locally",
        lambda _figure, **kwargs: {
            "description": "A person is shown in a sensitive context.",
            "findings_or_claims": "A potentially sensitive finding.",
            "non_text_visual_review_cues": ["person_or_face"],
            "sensitivity_term_matches": [],
            "sensitivity_screen_status": "not_requested",
            "token_usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "token_source": "reported",
                "model": kwargs["model"],
            },
        },
    )
    monkeypatch.setattr(
        "scripts.ingest.ingest_academic.audit",
        lambda event, data: audit_events.append((event, data)),
    )

    _assess_thesis_figures([figure], "Figure caption", None, config, logger)

    assert figure["vision_status"] == "flagged_for_human_review"
    assert "description" not in figure
    assert "sensitivity_ids" not in figure
    assert figure["vision_assessment"]["visual_content_review"] == "possible_non_text_cue"
    assert figure["vision_assessment"]["non_text_visual_review_cues"] == ["person_or_face"]
    assert "sensitive context" not in repr(figure)
    assert any(event == "vision_figure_visual_review" for event, _ in audit_events)


def test_missing_image_skips_vision_and_flags_image_sensitivities(monkeypatch):
    logger = DummyLogger()
    config = SimpleNamespace(
        figure_assessment_enabled=True,
        vision_model_name="qwen3.6:27b",
        ollama_host="http://host.docker.internal:11434",
        vision_assessment_timeout=240,
    )
    figure = {"figure_number": 3, "caption": "Figure caption", "image": None}
    audit_events = []

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Vision assessment must be skipped when the image is unavailable")

    monkeypatch.setattr("scripts.ingest.ingest_academic.assess_figure_locally", fail_if_called)
    monkeypatch.setattr(
        "scripts.ingest.ingest_academic.audit",
        lambda event, data: audit_events.append((event, data)),
    )
    profile = {
        "sensitivities": [
            {
                "id": "sacred_knowledge",
                "applies_to": ["images"],
                "detection_terms": ["sacred knowledge"],
            }
        ]
    }

    _assess_thesis_figures([figure], "Figure caption", profile, config, logger)

    assert figure["vision_status"] == "flagged_for_human_review"
    assert figure["sensitivity_ids"] == ["sacred_knowledge"]
    assert figure["vision_assessment"]["assessment_status"] == "image_unavailable"
    assert "description" not in figure
    assert any(event == "vision_figure_image_unavailable" for event, _ in audit_events)
