"""Tests for the dashboard pipeline controls."""

import base64
import json
from pathlib import Path

import pytest

from scripts.ui import pipelines
from scripts.ui.pipelines import (
    _build_pipeline_request,
    _pdf_options,
    _pipeline_control_styles,
    _run_progress,
    pipeline_control_enabled,
    save_uploaded_pdf,
)


def test_pipeline_controls_default_to_dev_only() -> None:
    assert pipeline_control_enabled("Dev") is True
    assert pipeline_control_enabled("Test") is False
    assert pipeline_control_enabled("Prod") is False


def test_pipeline_feature_flag_overrides_environment_default() -> None:
    assert pipeline_control_enabled("Prod", "true") is True
    assert pipeline_control_enabled("Dev", "false") is False


def test_reset_pipeline_request_is_marked_for_confirmation() -> None:
    request, requires_confirmation = _build_pipeline_request(
        "ingest_thesis",
        "/project/data_raw/academic_papers/thesis.pdf",
        None,
        "profile_1",
        ["dry_run"],
        "stale",
        ["reset"],
    )

    assert requires_confirmation is True
    assert request == {
        "job_type": "ingest_thesis",
        "options": {
            "paper_path": "/project/data_raw/academic_papers/thesis.pdf",
            "dry_run": True,
            "cultural_lens": "profile_1",
            "reset": True,
        },
        "thesis_id": None,
    }


def test_reset_values_are_scoped_to_their_pipeline_action() -> None:
    thesis_request, thesis_confirmation = _build_pipeline_request(
        "build_thesis_graph", None, "thesis-1", None, [], "stale", ["reset"]
    )
    graph_request, graph_confirmation = _build_pipeline_request(
        "build_consistency_graph",
        None,
        None,
        None,
        [],
        "stale",
        [],
        ["reset"],
    )

    assert thesis_confirmation is False
    assert thesis_request["options"] == {"thesis_id": "thesis-1"}
    assert graph_confirmation is True
    assert graph_request["options"] == {"reset": True}


def test_reference_revalidation_requires_and_carries_registered_thesis() -> None:
    with pytest.raises(ValueError, match="registered thesis"):
        _build_pipeline_request("revalidate_references", None, None, None, [], "stale", [])

    request, requires_confirmation = _build_pipeline_request(
        "revalidate_references", None, "thesis-1", None, [], "stale", []
    )

    assert requires_confirmation is False
    assert request == {
        "job_type": "revalidate_references",
        "options": {"mode": "stale", "thesis_id": "thesis-1"},
        "thesis_id": "thesis-1",
    }

    request, requires_confirmation = _build_pipeline_request(
        "revalidate_references",
        None,
        "thesis-1",
        None,
        [],
        "stale",
        ["reset"],
    )
    assert requires_confirmation is False
    assert request["options"] == {"mode": "stale", "thesis_id": "thesis-1"}


def test_pipeline_control_sections_follow_selected_action() -> None:
    new_thesis, registered_thesis, revalidation, consistency = _pipeline_control_styles(
        "revalidate_references"
    )

    assert new_thesis["display"] == "none"
    assert registered_thesis["display"] == "grid"
    assert revalidation["display"] == "grid"
    assert consistency["display"] == "none"


def test_pipeline_layout_exposes_non_destructive_actions(monkeypatch) -> None:
    class ComponentFactory:
        def __getattr__(self, component_name):
            def build_component(*children, **properties):
                component_children = (
                    children[0]
                    if len(children) == 1 and isinstance(children[0], list)
                    else list(children)
                )
                return {
                    "component": component_name,
                    "children": component_children,
                    "props": properties,
                }

            return build_component

    factory = ComponentFactory()
    monkeypatch.setattr(pipelines, "html", factory)
    monkeypatch.setattr(pipelines, "dcc", factory)
    button, panel = pipelines.build_pipeline_components("Dev")

    def find_component(component, component_id):
        if isinstance(component, dict):
            if component.get("props", {}).get("id") == component_id:
                return component
            return find_component(component.get("children", []), component_id)
        if isinstance(component, list):
            for child in component:
                found = find_component(child, component_id)
                if found is not None:
                    return found
        return None

    assert button["children"] == ["⚙️ Pipelines"]
    assert button["props"]["id"] == "tab-pipelines"
    assert panel["props"]["id"] == "pipelines-tab"
    new_thesis = find_component(panel, "pipeline-new-thesis-controls")
    registered_thesis = find_component(panel, "pipeline-registered-thesis-controls")
    consistency = find_component(panel, "pipeline-consistency-controls")
    upload = find_component(panel, "pipeline-upload")
    upload_help = find_component(panel, "pipeline-upload-help")
    submit = find_component(panel, "pipeline-submit")
    queue_help = find_component(panel, "pipeline-queue-help")
    assert new_thesis["props"]["style"]["display"] == "grid"
    assert find_component(new_thesis, "pipeline-reset") is not None
    assert find_component(registered_thesis, "pipeline-thesis") is not None
    assert find_component(registered_thesis, "pipeline-revalidation-mode") is not None
    assert find_component(consistency, "pipeline-consistency-reset") is not None
    assert upload["props"]["children"]["children"] == ["Choose PDF to upload"]
    assert "Upload stores the PDF only" in upload_help["children"][0]
    assert submit["children"] == ["Queue selected action"]
    assert "uploading a PDF alone does not run it" in queue_help["children"][0]
    assert find_component(panel, "pipeline-reset-confirm") is not None
    assert find_component(panel, "pipeline-job-poll") is not None


def test_uploaded_pdf_is_stored_under_academic_papers(tmp_path: Path) -> None:
    pdf_bytes = b"%PDF-1.7\nThesis contents"
    contents = "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode("ascii")

    saved_path = save_uploaded_pdf(contents, "thesis draft.pdf", tmp_path, 1)

    assert saved_path.is_file()
    assert saved_path.parent == tmp_path / "data_raw" / "academic_papers"
    assert saved_path.name.endswith("_thesis_draft.pdf")
    assert saved_path.read_bytes() == pdf_bytes


def test_pdf_selector_ignores_symlinks_outside_papers_directory(tmp_path: Path) -> None:
    papers_dir = tmp_path / "data_raw" / "academic_papers"
    papers_dir.mkdir(parents=True)
    (papers_dir / "thesis.pdf").write_bytes(b"%PDF-1.7")
    outside_pdf = tmp_path / "outside.pdf"
    outside_pdf.write_bytes(b"%PDF-1.7")
    (papers_dir / "external.pdf").symlink_to(outside_pdf)

    assert _pdf_options(tmp_path) == [
        {"label": "thesis.pdf", "value": str((papers_dir / "thesis.pdf").resolve())}
    ]


def test_progress_reader_returns_latest_checkpoint_for_job(tmp_path: Path) -> None:
    audit_path = tmp_path / "ingest_audit.jsonl"
    events = [
        {
            "event": "progress_checkpoint",
            "run_id": "another-run",
            "items_done": 9,
            "items_total": 10,
        },
        {
            "event": "progress_checkpoint",
            "run_id": "job-run",
            "stage": "document_ingestion",
            "items_done": 2,
            "items_total": 4,
            "percent": 50.0,
            "succeeded": 2,
            "failed": 0,
        },
        {
            "event": "progress_checkpoint",
            "run_id": "job-run",
            "stage": "document_ingestion",
            "items_done": 3,
            "items_total": 4,
            "percent": 75.0,
            "succeeded": 2,
            "failed": 1,
            "skipped": 0,
        },
    ]
    audit_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    assert _run_progress(audit_path.parent, "job-run") == {
        "stage": "document_ingestion",
        "items_done": 3,
        "items_total": 4,
        "percent": 75.0,
        "succeeded": 2,
        "failed": 1,
        "skipped": 0,
    }


@pytest.mark.parametrize(
    ("filename", "content", "limit_mb", "message"),
    [
        ("../outside.pdf", b"%PDF-1.7", 1, "filename"),
        ("thesis.txt", b"%PDF-1.7", 1, "PDF filename"),
        ("thesis.pdf", b"not a PDF", 1, "PDF signature"),
        ("thesis.pdf", b"%PDF-1.7" + b"x" * 32, 0, "size limit"),
    ],
)
def test_uploaded_pdf_rejects_unsafe_or_invalid_input(
    tmp_path: Path, filename: str, content: bytes, limit_mb: int, message: str
) -> None:
    encoded = "data:application/pdf;base64," + base64.b64encode(content).decode("ascii")

    with pytest.raises(ValueError, match=message):
        save_uploaded_pdf(encoded, filename, tmp_path, limit_mb)
