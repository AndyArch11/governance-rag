"""Dashboard controls for queued academic-ingestion and graph jobs."""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import uuid
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from scripts.utils.job_runner import JobRunner

_RUNNER: JobRunner | None = None
_RUNNER_LOCK = Lock()


def pipeline_control_enabled(environment: str, configured: str | None = None) -> bool:
    """Return whether dashboard pipeline actions are enabled for this environment.

    Args:
        environment: The current environment name (e.g., "dev", "prod").
        configured: Optional string indicating if pipeline control is explicitly enabled.

    Returns:
        True if pipeline control is enabled, False otherwise.
    """
    if configured is None:
        configured = os.getenv("UI_PIPELINE_CONTROL_ENABLED")
    if configured is None:
        return environment.casefold() == "dev"
    return configured.strip().casefold() in {"true", "1", "yes", "on"}


def _pdf_options(project_root: Path) -> list[dict[str, str]]:
    """Return a list of available PDF files in the academic papers directory.

    Args:
        project_root: The root path of the project.

    Returns:
        A list of dictionaries with 'label' and 'value' keys for each PDF file.
    """
    papers_dir = project_root / "data_raw" / "academic_papers"
    if not papers_dir.is_dir():
        return []
    papers_root = papers_dir.resolve()
    options = []
    for path in sorted(papers_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.casefold() != ".pdf":
            continue
        try:
            resolved_path = path.resolve(strict=True)
        except OSError:
            continue
        if resolved_path.is_relative_to(papers_root):
            options.append({"label": path.name, "value": str(resolved_path)})
    return options


def save_uploaded_pdf(contents: str, filename: str, project_root: Path, max_size_mb: int) -> Path:
    """Decode and safely store one bounded PDF upload under the academic papers directory.

    Args:
        contents: The base64-encoded content of the PDF file.
        filename: The original filename of the uploaded PDF.
        project_root: The root path of the project.
        max_size_mb: The maximum allowed size of the PDF in megabytes.

    Returns:
        The path to the stored PDF file.

    """
    if not isinstance(filename, str) or not filename.strip() or "/" in filename or "\\" in filename:
        raise ValueError("filename must be a simple PDF filename")
    if Path(filename).suffix.casefold() != ".pdf":
        raise ValueError("upload must have a PDF filename")
    if isinstance(max_size_mb, bool) or not isinstance(max_size_mb, int) or max_size_mb < 1:
        raise ValueError("size limit must be a positive number of megabytes")
    if not isinstance(contents, str):
        raise ValueError("upload content must be base64 data")
    header, separator, encoded = contents.partition(",")
    if not separator or not header.startswith("data:") or not header.endswith(";base64"):
        raise ValueError("upload content must be a base64 data URL")
    maximum_bytes = max_size_mb * 1024 * 1024
    maximum_encoded_length = 4 * ((maximum_bytes + 2) // 3)
    if len(encoded) > maximum_encoded_length:
        raise ValueError("upload exceeds the configured size limit")
    try:
        pdf_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("upload content is not valid base64") from exc
    if len(pdf_bytes) > maximum_bytes:
        raise ValueError("upload exceeds the configured size limit")
    if b"%PDF-" not in pdf_bytes[:1024]:
        raise ValueError("upload does not contain a PDF signature")

    safe_filename = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name).strip("._")
    papers_dir = (project_root / "data_raw" / "academic_papers").resolve()
    papers_dir.mkdir(parents=True, exist_ok=True)
    destination = papers_dir / f"{uuid.uuid4().hex}_{safe_filename}"
    if destination.parent != papers_dir:
        raise ValueError("upload destination is outside academic_papers")
    with destination.open("xb") as uploaded_file:
        uploaded_file.write(pdf_bytes)
    return destination


def _build_pipeline_request(
    action: str,
    paper_path: str | None,
    thesis_id: str | None,
    cultural_lens: str | None,
    dry_run_values: list[str] | None,
    revalidation_mode: str,
    reset_values: list[str] | None,
    consistency_reset_values: list[str] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Build an allowlisted pipeline request and identify destructive resets.

    Args:
        action: The pipeline action to perform.
        paper_path: Path to the thesis PDF, if applicable.
        thesis_id: ID of the registered thesis, if applicable.
        cultural_lens: The cultural lens profile to apply, if any.
        dry_run_values: List of dry run flags.
        revalidation_mode: Mode for revalidating references.
        reset_values: Ingestion reset flags.
        consistency_reset_values: Consistency-graph reset flags.

    Returns:
        A tuple containing the pipeline request dictionary and a boolean indicating if a destructive reset was requested.
    """
    reset_requested = action == "ingest_thesis" and "reset" in (reset_values or [])
    consistency_reset_requested = action == "build_consistency_graph" and "reset" in (
        consistency_reset_values or []
    )

    options: dict[str, Any]
    selected_thesis_id = thesis_id
    if action == "ingest_thesis":
        if not paper_path:
            raise ValueError("Select a thesis PDF first.")
        options = {
            "paper_path": paper_path,
            "dry_run": "dry_run" in (dry_run_values or []),
        }
        if cultural_lens and cultural_lens.strip():
            options["cultural_lens"] = cultural_lens.strip()
        if reset_requested:
            options["reset"] = True
    elif action == "build_thesis_graph":
        if not thesis_id:
            raise ValueError("Select a registered thesis first.")
        options = {"thesis_id": thesis_id}
    elif action == "build_consistency_graph":
        options = {"reset": True} if consistency_reset_requested else {}
        selected_thesis_id = None
    elif action == "revalidate_references":
        if not thesis_id:
            raise ValueError("Select a registered thesis first.")
        options = {"mode": revalidation_mode or "stale", "thesis_id": thesis_id}
        selected_thesis_id = thesis_id
    else:
        raise ValueError("Unsupported pipeline action.")
    return (
        {"job_type": action, "options": options, "thesis_id": selected_thesis_id},
        reset_requested or consistency_reset_requested,
    )


def _pipeline_control_styles(
    action: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """Return visibility styles for action-specific control groups."""

    def section_style(visible: bool) -> dict[str, str]:
        if not visible:
            return {"display": "none"}
        return {
            "display": "grid",
            "gridTemplateColumns": "minmax(180px, 1fr)",
            "gap": "8px",
        }

    return (
        section_style(action == "ingest_thesis"),
        section_style(action in {"build_thesis_graph", "revalidate_references"}),
        section_style(action == "revalidate_references"),
        section_style(action == "build_consistency_graph"),
    )


def build_pipeline_components(environment: str, configured: str | None = None) -> tuple[Any, Any]:
    """Build the dashboard navigation button and pipeline control panel.

    Args:
        environment: The current environment name (e.g., "dev", "prod").
        configured: Optional string indicating if pipeline control is explicitly enabled.

    Returns:
        A tuple containing the navigation button and the pipeline control panel.
    """
    enabled = pipeline_control_enabled(environment, configured)
    button = html.Button(
        "⚙️ Pipelines",
        id="tab-pipelines",
        n_clicks=0,
        className="tab-button",
        disabled=not enabled,
        style={"display": "inline-block" if enabled else "none"},
    )
    new_thesis_style, registered_thesis_style, revalidation_style, consistency_style = (
        _pipeline_control_styles("ingest_thesis")
    )
    panel = html.Div(
        [
            html.H3("Academic Pipelines"),
            html.P(
                "Jobs run one at a time in isolated CLI processes. Output and token totals are local.",
                style={"color": "#555"},
            ),
            html.Div(
                [
                    html.Label("Pipeline action"),
                    dcc.Dropdown(
                        id="pipeline-action",
                        options=[
                            {"label": "Ingest new thesis", "value": "ingest_thesis"},
                            {
                                "label": "Build thesis evidence graph",
                                "value": "build_thesis_graph",
                            },
                            {
                                "label": "Build consistency graph",
                                "value": "build_consistency_graph",
                            },
                            {
                                "label": "Revalidate references for thesis",
                                "value": "revalidate_references",
                            },
                        ],
                        value="ingest_thesis",
                        clearable=False,
                    ),
                    html.Div(
                        [
                            html.H4("New thesis"),
                            html.Label("Thesis PDF"),
                            dcc.Dropdown(
                                id="pipeline-paper",
                                options=[],
                                placeholder="Select a PDF from the academic papers library",
                                searchable=True,
                            ),
                            dcc.Upload(
                                id="pipeline-upload",
                                children=html.Button(
                                    "Choose PDF to upload",
                                    type="button",
                                    title="Stores the PDF in the library; it does not start ingestion.",
                                ),
                                accept=".pdf,application/pdf",
                                multiple=False,
                            ),
                            html.P(
                                "Upload stores the PDF only. Select it above, then queue the ingestion action.",
                                id="pipeline-upload-help",
                                style={"color": "#555", "margin": "0"},
                            ),
                            html.Div(id="pipeline-upload-status", role="status"),
                            dcc.Store(id="pipeline-uploaded-path", storage_type="session"),
                            html.Label("Cultural lens profile ID"),
                            dcc.Input(
                                id="pipeline-cultural-lens",
                                type="text",
                                placeholder="Optional profile ID",
                            ),
                            dcc.Checklist(
                                id="pipeline-dry-run",
                                options=[{"label": "Dry run", "value": "dry_run"}],
                                value=[],
                            ),
                            dcc.Checklist(
                                id="pipeline-reset",
                                options=[{"label": "Reset existing data", "value": "reset"}],
                                value=[],
                            ),
                            html.P(
                                "Reset clears all academic chunks, references and indexes in this workspace.",
                                style={"color": "#8a3b12", "margin": "0"},
                            ),
                        ],
                        id="pipeline-new-thesis-controls",
                        style=new_thesis_style,
                    ),
                    html.Div(
                        [
                            html.H4("Registered thesis"),
                            dcc.Dropdown(
                                id="pipeline-thesis",
                                options=[],
                                placeholder="Select a registered thesis",
                                searchable=True,
                            ),
                            html.Div(
                                [
                                    html.Label("Reference revalidation mode"),
                                    dcc.Dropdown(
                                        id="pipeline-revalidation-mode",
                                        options=[
                                            {"label": mode.title(), "value": mode}
                                            for mode in ("stale", "online", "all", "failed")
                                        ],
                                        value="stale",
                                        clearable=False,
                                    ),
                                ],
                                id="pipeline-revalidation-controls",
                                style=revalidation_style,
                            ),
                        ],
                        id="pipeline-registered-thesis-controls",
                        style=registered_thesis_style,
                    ),
                    html.Div(
                        [
                            html.H4("Consistency graph"),
                            dcc.Checklist(
                                id="pipeline-consistency-reset",
                                options=[
                                    {
                                        "label": "Reset consistency graph before building",
                                        "value": "reset",
                                    }
                                ],
                                value=[],
                            ),
                        ],
                        id="pipeline-consistency-controls",
                        style=consistency_style,
                    ),
                    html.Button(
                        "Queue selected action",
                        id="pipeline-submit",
                        n_clicks=0,
                        disabled=not enabled,
                        title="Starts the selected pipeline action.",
                    ),
                    html.P(
                        "Queueing starts the selected action; uploading a PDF alone does not run it.",
                        id="pipeline-queue-help",
                        style={"color": "#555", "margin": "0"},
                    ),
                    html.Div(id="pipeline-submit-message", role="status"),
                    dcc.ConfirmDialog(
                        id="pipeline-reset-confirm",
                        message="This reset deletes or replaces stored data. Continue?",
                        displayed=False,
                    ),
                    dcc.Store(id="pipeline-pending-job"),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(180px, 1fr)",
                    "gap": "8px",
                },
            ),
            dcc.Store(id="pipeline-last-job"),
            dcc.Store(id="pipeline-refreshed-job"),
            dcc.Interval(
                id="pipeline-job-poll",
                interval=2000,
                n_intervals=0,
                disabled=not enabled,
            ),
            html.Div(id="pipeline-job-status", role="status", style={"marginTop": "16px"}),
            html.Pre(
                id="pipeline-job-log",
                style={
                    "maxHeight": "360px",
                    "overflowY": "auto",
                    "whiteSpace": "pre-wrap",
                    "backgroundColor": "#f4f4f4",
                    "padding": "12px",
                },
            ),
        ],
        id="pipelines-tab",
        style={"display": "none"},
        className="card",
    )
    return button, panel


def _get_runner(project_root: Path, rag_data_path: Path) -> JobRunner:
    """Get or create a singleton JobRunner instance.

    Args:
        project_root: The root path of the project.
        rag_data_path: The path to the RAG data directory.

    Returns:
        A JobRunner instance.
    """
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            _RUNNER = JobRunner(
                project_root=project_root,
                database_path=rag_data_path / "jobs.db",
                logs_dir=project_root / "logs" / "jobs",
            )
        return _RUNNER


def _registered_thesis_options(rag_data_path: Path) -> list[dict[str, str]]:
    """Retrieve a list of registered thesis options from the thesis registry.

    Args:
        rag_data_path: The path to the RAG data directory.

    Returns:
        A list of dictionaries with 'label' and 'value' keys for each registered thesis.
    """
    registry_path = rag_data_path / "thesis_graphs" / "registry.sqlite"
    if not registry_path.is_file():
        return []
    from scripts.thesis_graph.thesis_registry import ThesisRegistry

    registry = ThesisRegistry(registry_path)
    return [
        {
            "label": f"{thesis['title']} ({thesis['thesis_id']})",
            "value": thesis["thesis_id"],
        }
        for thesis in registry.list_theses()
    ]


def _read_log_tail(path: Path, max_lines: int = 60) -> str:
    """Read the last few lines of a log file.

    Args:
        path: The path to the log file.
        max_lines: The maximum number of lines to read from the end of the file.

    Returns:
        A string containing the last `max_lines` lines of the log file.
    """
    if not path.is_file():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as log_file:
        return "".join(deque(log_file, maxlen=max_lines))


def _run_usage(logs_dir: Path, run_id: str) -> tuple[int, int]:
    """Calculate the total input and output tokens used for a specific run.

    Args:
        logs_dir: The directory containing audit log files.
        run_id: The unique identifier of the run to calculate usage for.

    Returns:
        A tuple containing the total input tokens and total output tokens for the run.
    """
    input_tokens = 0
    output_tokens = 0
    for audit_path in logs_dir.glob("*_audit.jsonl"):
        try:
            with audit_path.open("r", encoding="utf-8") as audit_file:
                for line in audit_file:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event") == "llm_usage" and event.get("run_id") == run_id:
                        input_tokens += int(event.get("input_tokens", 0) or 0)
                        output_tokens += int(event.get("output_tokens", 0) or 0)
        except OSError:
            continue
    return input_tokens, output_tokens


def _run_progress(logs_dir: Path, run_id: str) -> dict[str, Any] | None:
    """Return the latest structured progress checkpoint for a pipeline run.
    Args:
        logs_dir: The directory containing audit log files.
        run_id: The unique identifier of the pipeline run.

    Returns:
        A dictionary containing the latest progress checkpoint, or None if no checkpoint is found.
    """
    latest_event: dict[str, Any] | None = None
    latest_timestamp = ""
    for audit_path in logs_dir.glob("*_audit.jsonl"):
        try:
            with audit_path.open("r", encoding="utf-8") as audit_file:
                for line in audit_file:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event") != "progress_checkpoint" or event.get("run_id") != run_id:
                        continue
                    timestamp = str(event.get("timestamp", ""))
                    if latest_event is None or timestamp >= latest_timestamp:
                        latest_event = event
                        latest_timestamp = timestamp
        except OSError:
            continue
    if latest_event is None:
        return None
    try:
        items_done_value = latest_event.get("items_done")
        items_total_value = latest_event.get("items_total")
        if items_done_value is None:
            items_done_value = latest_event["completed"]
        if items_total_value is None:
            items_total_value = latest_event["total"]
        items_done = max(0, int(items_done_value))
        items_total = max(0, int(items_total_value))
    except (KeyError, TypeError, ValueError):
        return None
    try:
        percent = float(
            latest_event.get("percent", 100 * items_done / items_total if items_total else 0)
        )
    except (TypeError, ValueError):
        percent = 100 * items_done / items_total if items_total else 0
    return {
        "stage": str(latest_event.get("stage") or "document_ingestion"),
        "items_done": items_done,
        "items_total": items_total,
        "percent": percent,
        "succeeded": int(latest_event.get("succeeded", 0) or 0),
        "failed": int(latest_event.get("failed", 0) or 0),
        "skipped": int(latest_event.get("skipped", 0) or 0),
    }


def register_pipeline_callbacks(
    app: Dash,
    project_root: Path,
    rag_data_path: Path,
    environment: str,
    on_job_succeeded: Callable[[dict[str, Any]], list[dict[str, str]]] | None = None,
) -> None:
    """Register guarded submission and status-polling callbacks.

    Args:
        app: The Dash application instance.
        project_root: The root path of the project.
        rag_data_path: The path to the RAG data directory.
        environment: The current environment name (e.g., "dev", "prod").
        on_job_succeeded: Optional callback to refresh application data after a successful job.
    """
    enabled = pipeline_control_enabled(environment)

    @app.callback(
        Output("pipeline-new-thesis-controls", "style"),
        Output("pipeline-registered-thesis-controls", "style"),
        Output("pipeline-revalidation-controls", "style"),
        Output("pipeline-consistency-controls", "style"),
        Input("pipeline-action", "value"),
    )
    def update_pipeline_control_sections(
        action: str,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
        """Show controls relevant to the selected pipeline action."""
        return _pipeline_control_styles(action)

    @app.callback(
        Output("pipeline-last-job", "data"),
        Output("pipeline-submit-message", "children"),
        Output("pipeline-reset-confirm", "displayed"),
        Output("pipeline-pending-job", "data"),
        Input("pipeline-submit", "n_clicks"),
        Input("pipeline-reset-confirm", "submit_n_clicks"),
        Input("pipeline-reset-confirm", "cancel_n_clicks"),
        State("pipeline-action", "value"),
        State("pipeline-paper", "value"),
        State("pipeline-thesis", "value"),
        State("pipeline-cultural-lens", "value"),
        State("pipeline-dry-run", "value"),
        State("pipeline-revalidation-mode", "value"),
        State("pipeline-reset", "value"),
        State("pipeline-consistency-reset", "value"),
        State("pipeline-pending-job", "data"),
        prevent_initial_call=True,
    )
    def submit_pipeline_job(
        clicks: int,
        confirmation_clicks: int,
        cancellation_clicks: int,
        action: str,
        paper_path: str | None,
        thesis_id: str | None,
        cultural_lens: str | None,
        dry_run_values: list[str] | None,
        revalidation_mode: str,
        reset_values: list[str] | None,
        consistency_reset_values: list[str] | None,
        pending_request: dict[str, Any] | None,
    ) -> tuple[dict[str, str] | Any, str, bool, dict[str, Any] | None]:
        """Submit a pipeline job based on the selected action and options.

        Args:
            clicks: The number of times the submit button has been clicked.
            confirmation_clicks: The number of accepted reset confirmations.
            cancellation_clicks: The number of cancelled reset confirmations.
            action: The selected pipeline action.
            paper_path: The path to the thesis PDF file.
            thesis_id: The ID of the registered thesis.
            cultural_lens: The selected cultural lens.
            dry_run_values: The list of selected dry-run options.
            revalidation_mode: The selected revalidation mode.
            reset_values: The selected destructive reset option.
            consistency_reset_values: The selected consistency graph reset option.
            pending_request: A request awaiting destructive-action confirmation.

        Returns:
            A tuple containing the last job data, status message, dialog visibility and pending request.
        """
        if not enabled:
            return (
                no_update,
                "Dashboard pipeline controls are disabled in this environment.",
                False,
                None,
            )
        if ctx.triggered_id == "pipeline-reset-confirm":
            if any(prop.endswith(".cancel_n_clicks") for prop in ctx.triggered_prop_ids):
                return no_update, "Reset cancelled.", False, None
            if not confirmation_clicks or not pending_request:
                raise PreventUpdate
            request = pending_request
        else:
            if ctx.triggered_id != "pipeline-submit" or not clicks:
                raise PreventUpdate
            try:
                request, requires_confirmation = _build_pipeline_request(
                    action,
                    paper_path,
                    thesis_id,
                    cultural_lens,
                    dry_run_values,
                    revalidation_mode,
                    reset_values,
                    consistency_reset_values,
                )
            except ValueError as exc:
                return no_update, str(exc), False, None
            if requires_confirmation:
                return no_update, "Confirm the destructive reset to continue.", True, request
        try:
            job_id = _get_runner(project_root, rag_data_path).submit(
                request["job_type"],
                request["options"],
                thesis_id=request.get("thesis_id"),
            )
        except (OSError, ValueError) as exc:
            return no_update, str(exc), False, None
        return {"job_id": job_id}, f"Queued job {job_id}.", False, None

    @app.callback(
        Output("pipeline-uploaded-path", "data"),
        Output("pipeline-upload-status", "children"),
        Input("pipeline-upload", "contents"),
        State("pipeline-upload", "filename"),
        prevent_initial_call=True,
    )
    def store_uploaded_pdf(contents: str | None, filename: str | None) -> tuple[Any, str]:
        """Store the uploaded PDF file and return its path along with a status message.

        Args:
            contents: The contents of the uploaded PDF file.
            filename: The name of the uploaded PDF file.

        Returns:
            A tuple containing the uploaded PDF path data and a status message.
        """
        if not enabled:
            return no_update, "Dashboard pipeline controls are disabled in this environment."
        if not contents or not filename:
            raise PreventUpdate
        try:
            max_size_mb = int(os.getenv("MAX_PDF_SIZE_MB", "50"))
            uploaded_path = save_uploaded_pdf(contents, filename, project_root, max_size_mb)
        except (OSError, ValueError) as exc:
            return no_update, str(exc)
        return {"path": str(uploaded_path)}, "Upload stored. Select it above to queue ingestion."

    @app.callback(
        Output("pipeline-job-status", "children"),
        Output("pipeline-job-log", "children"),
        Output("pipeline-paper", "options"),
        Output("pipeline-thesis", "options"),
        Output("document-selector", "options", allow_duplicate=True),
        Output("pipeline-refreshed-job", "data"),
        Input("pipeline-job-poll", "n_intervals"),
        State("pipeline-last-job", "data"),
        State("pipeline-uploaded-path", "data"),
        State("pipeline-refreshed-job", "data"),
        prevent_initial_call=True,
    )
    def poll_pipeline_jobs(
        _interval: int,
        last_job: dict[str, str] | None,
        uploaded_pdf: dict[str, str] | None,
        refreshed_job_id: str | None,
    ) -> tuple[Any, str, list[dict[str, str]], list[dict[str, str]], Any, str | None]:
        """Poll the status of pipeline jobs and update the UI accordingly.

        Args:
            _interval: The current interval count from the polling component.
            last_job: The data of the last known pipeline job.
            uploaded_pdf: The data of the uploaded PDF file.
            refreshed_job_id: The ID of the last successful job whose caches were refreshed.

        Returns:
            A tuple containing status, log tail, paper and thesis options, document options and
            the last refreshed job ID.
        """
        paper_options = _pdf_options(project_root)
        if uploaded_pdf and uploaded_pdf.get("path"):
            try:
                uploaded_path = Path(uploaded_pdf["path"]).resolve(strict=True)
                papers_root = (project_root / "data_raw" / "academic_papers").resolve()
                if (
                    uploaded_path.is_file()
                    and uploaded_path.suffix.casefold() == ".pdf"
                    and uploaded_path.is_relative_to(papers_root)
                    and str(uploaded_path) not in {item["value"] for item in paper_options}
                ):
                    paper_options.append({"label": uploaded_path.name, "value": str(uploaded_path)})
            except (OSError, RuntimeError):
                pass
        thesis_options = _registered_thesis_options(rag_data_path)
        if not enabled:
            return (
                "Pipeline controls are disabled.",
                "",
                paper_options,
                thesis_options,
                no_update,
                refreshed_job_id,
            )
        runner = _get_runner(project_root, rag_data_path)
        job = runner.get_job(last_job.get("job_id", "")) if last_job else None
        if job is None:
            jobs = runner.list_jobs(limit=1)
            job = jobs[0] if jobs else None
        if job is None:
            return (
                "No pipeline jobs yet.",
                "",
                paper_options,
                thesis_options,
                no_update,
                refreshed_job_id,
            )
        input_tokens, output_tokens = _run_usage(project_root / "logs", job["run_id"])
        progress = _run_progress(project_root / "logs", job["run_id"])
        document_options: Any = no_update
        new_refreshed_job_id = refreshed_job_id
        status_children = [
            html.Strong(f"{job['job_type']}: {job['status']}"),
            html.Span(f" | Job {job['job_id']}"),
        ]
        if progress:
            status_children.extend(
                [
                    html.Span(
                        f" | {progress['stage']}: {progress['items_done']}/"
                        f"{progress['items_total']} ({progress['percent']:.1f}%)"
                    ),
                    html.Progress(
                        value=progress["items_done"],
                        max=max(progress["items_total"], 1),
                    ),
                    html.Span(
                        f" | Succeeded: {progress['succeeded']} | Failed: {progress['failed']}"
                        f" | Skipped: {progress['skipped']}"
                    ),
                ]
            )
        else:
            status_children.append(html.Span(" | No structured progress reported"))
        if (
            job["status"] == "succeeded"
            and job["job_id"] != refreshed_job_id
            and on_job_succeeded is not None
        ):
            try:
                document_options = on_job_succeeded(job)
            except Exception:
                status_children.append(html.Span(" | Graph refresh failed; reload the page."))
            new_refreshed_job_id = job["job_id"]
        status_children.append(
            html.Span(
                f" | Model tokens: {input_tokens} in, {output_tokens} out "
                "(embedding input estimates included)"
            )
        )
        status = html.Div(status_children)
        return (
            status,
            _read_log_tail(Path(job["log_path"])),
            paper_options,
            thesis_options,
            document_options,
            new_refreshed_job_id,
        )
