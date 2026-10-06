"""Tests for safe dashboard pipeline job scheduling."""

import json
import threading
from pathlib import Path

import pytest

from scripts.utils import job_runner
from scripts.utils.job_runner import JobRunner, build_job_command


def test_ingestion_command_uses_allowlisted_arguments(tmp_path: Path) -> None:
    papers_dir = tmp_path / "data_raw" / "academic_papers"
    papers_dir.mkdir(parents=True)
    thesis_pdf = papers_dir / "thesis.pdf"
    thesis_pdf.write_bytes(b"%PDF-test")

    command = build_job_command(
        "ingest_thesis",
        {
            "paper_path": str(thesis_pdf),
            "dry_run": True,
            "cultural_lens": "profile_1",
            "skip_citations": True,
        },
        tmp_path,
    )

    assert command[:3] == [command[0], "-m", "scripts.ingest.ingest_academic"]
    assert command[3:] == [
        "--papers",
        str(thesis_pdf),
        "--dry-run",
        "--skip-citations",
        "--cultural-lens",
        "profile_1",
    ]


def test_ingestion_command_rejects_paths_outside_papers_directory(tmp_path: Path) -> None:
    external_pdf = tmp_path / "private.pdf"
    external_pdf.write_bytes(b"%PDF-test")

    with pytest.raises(ValueError, match="academic_papers"):
        build_job_command("ingest_thesis", {"paper_path": str(external_pdf)}, tmp_path)


def test_command_builder_rejects_unknown_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported options"):
        build_job_command(
            "build_thesis_graph",
            {"thesis_id": "thesis-1", "shell": "touch /tmp/unwanted"},
            tmp_path,
        )


def test_command_builder_covers_graph_and_revalidation_jobs(tmp_path: Path) -> None:
    assert build_job_command("build_thesis_graph", {"thesis_id": "thesis-1"}, tmp_path) == [
        build_job_command("build_thesis_graph", {"thesis_id": "thesis-1"}, tmp_path)[0],
        "-m",
        "scripts.thesis_graph.build_thesis_evidence_graph",
        "thesis-1",
    ]
    assert build_job_command(
        "revalidate_references", {"mode": "stale", "staleness_threshold": 45}, tmp_path
    )[3:] == ["--revalidate", "stale", "--staleness-threshold", "45"]
    assert build_job_command(
        "revalidate_references", {"mode": "all", "thesis_id": "thesis-1"}, tmp_path
    )[3:] == ["--revalidate", "all", "--thesis-id", "thesis-1"]
    consistency_command = build_job_command(
        "build_consistency_graph", {"workers": 2, "reset": True}, tmp_path
    )
    assert consistency_command[2] == "scripts.consistency_graph.build_consistency_graph"
    assert consistency_command[3:] == ["--workers", "2", "--reset"]


def test_thesis_job_commands_accept_filename_stem_ids_with_spaces(tmp_path: Path) -> None:
    thesis_id = "O. Meyers - PhD thesis"

    graph_command = build_job_command("build_thesis_graph", {"thesis_id": thesis_id}, tmp_path)
    revalidation_command = build_job_command(
        "revalidate_references",
        {"mode": "all", "thesis_id": thesis_id},
        tmp_path,
    )

    assert graph_command[2:] == ["scripts.thesis_graph.build_thesis_evidence_graph", thesis_id]
    assert revalidation_command[3:] == ["--revalidate", "all", "--thesis-id", thesis_id]


@pytest.mark.parametrize("thesis_id", ["", "   ", ".", "..", "../thesis", "folder\\thesis", "id\n2"])
def test_thesis_job_commands_reject_unsafe_ids(tmp_path: Path, thesis_id: str) -> None:
    with pytest.raises(ValueError, match="thesis_id must be a valid identifier"):
        build_job_command("build_thesis_graph", {"thesis_id": thesis_id}, tmp_path)


def test_runner_persists_and_cancels_queued_job(tmp_path: Path) -> None:
    runner = JobRunner(
        project_root=tmp_path,
        database_path=tmp_path / "jobs.db",
        logs_dir=tmp_path / "jobs",
        start_worker=False,
    )
    job_id = runner.submit("build_thesis_graph", {"thesis_id": "thesis-1"})

    queued = runner.get_job(job_id)
    assert queued is not None
    assert queued["status"] == "queued"
    assert queued["run_id"]
    assert Path(queued["log_path"]).parent == tmp_path / "jobs"

    assert runner.cancel(job_id) is True
    assert runner.get_job(job_id)["status"] == "cancelled"
    runner.shutdown()


def test_runner_recovers_completed_ingestion_from_audit(tmp_path: Path, monkeypatch) -> None:
    papers_dir = tmp_path / "data_raw" / "academic_papers"
    papers_dir.mkdir(parents=True)
    thesis_pdf = papers_dir / "thesis.pdf"
    thesis_pdf.write_bytes(b"%PDF-test")
    database_path = tmp_path / "jobs.db"
    logs_dir = tmp_path / "jobs"
    runner = JobRunner(
        project_root=tmp_path,
        database_path=database_path,
        logs_dir=logs_dir,
        start_worker=False,
    )
    job_id = runner.submit("ingest_thesis", {"paper_path": str(thesis_pdf)})
    job = runner.get_job(job_id)
    assert job is not None

    with runner._connection() as connection:
        connection.execute(
            "UPDATE jobs SET status = 'running', worker_pid = 999999999 WHERE job_id = ?",
            (job_id,),
        )
    audit_dir = tmp_path / "logs"
    audit_dir.mkdir(exist_ok=True)
    (audit_dir / "ingest_audit.jsonl").write_text(
        json.dumps({"event": "complete", "run_id": job["run_id"]}) + "\n",
        encoding="utf-8",
    )
    runner.shutdown()

    monkeypatch.setattr(JobRunner, "_process_exists", staticmethod(lambda process_id: False))
    recovered_runner = JobRunner(
        project_root=tmp_path,
        database_path=database_path,
        logs_dir=logs_dir,
        start_worker=False,
    )

    recovered_job = recovered_runner.get_job(job_id)
    assert recovered_job is not None
    assert recovered_job["status"] == "succeeded"
    assert recovered_job["exit_code"] == 0
    assert recovered_job["finished_at"] is not None
    recovered_runner.shutdown()


def test_runner_streams_output_and_finishes_job(tmp_path: Path, monkeypatch) -> None:
    captured_environment: dict[str, str] = {}

    class CompletedProcess:
        pid = 12345
        stdout = iter(["pipeline started\n", "pipeline complete\n"])

        def __init__(self, *args, **kwargs):
            self.returncode = None
            captured_environment.update(kwargs["env"])

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    monkeypatch.setattr(job_runner.subprocess, "Popen", CompletedProcess)
    runner = JobRunner(
        project_root=tmp_path,
        database_path=tmp_path / "jobs.db",
        logs_dir=tmp_path / "jobs",
    )
    job_id = runner.submit("build_thesis_graph", {"thesis_id": "thesis-1"})
    finished = threading.Event()
    job = None
    for _ in range(100):
        job = runner.get_job(job_id)
        if job and job["status"] in {"succeeded", "failed", "cancelled"}:
            finished.set()
            break
        threading.Event().wait(0.01)
    runner.shutdown()

    assert finished.is_set()
    assert job is not None and job["status"] == "succeeded"
    assert job["exit_code"] == 0
    assert captured_environment["LLM_RUN_ID"] == job["run_id"]
    assert "pipeline complete" in Path(job["log_path"]).read_text()


def test_child_audit_events_record_run_id(tmp_path: Path, monkeypatch) -> None:
    from scripts.utils import logger

    monkeypatch.setattr(logger, "LOGS_DIR", tmp_path)
    monkeypatch.setenv("LLM_RUN_ID", "run-123")

    logger.audit("ingest", "llm_usage", {"input_tokens": 12})
    logger.audit("ingest", "pipeline_started", {})

    events = [
        json.loads(line) for line in (tmp_path / "ingest_audit.jsonl").read_text().splitlines()
    ]
    assert events[0]["run_id"] == "run-123"
    assert events[1]["run_id"] == "run-123"


def test_runner_cancels_active_process_group(tmp_path: Path, monkeypatch) -> None:
    terminated = threading.Event()
    active_process = None

    class BlockingOutput:
        def __iter__(self):
            return self

        def __next__(self):
            terminated.wait()
            raise StopIteration

    class RunningProcess:
        pid = 54321
        stdout = BlockingOutput()

        def __init__(self, *args, **kwargs):
            nonlocal active_process
            self.returncode = None
            active_process = self

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if not terminated.wait(timeout):
                raise job_runner.subprocess.TimeoutExpired("command", timeout)
            return self.returncode

    def terminate_group(_process_id, _signal_number):
        active_process.returncode = -15
        terminated.set()

    monkeypatch.setattr(job_runner.subprocess, "Popen", RunningProcess)
    monkeypatch.setattr(job_runner.os, "killpg", terminate_group)
    runner = JobRunner(
        project_root=tmp_path,
        database_path=tmp_path / "jobs.db",
        logs_dir=tmp_path / "jobs",
    )
    job_id = runner.submit("build_thesis_graph", {"thesis_id": "thesis-1"})
    running = threading.Event()
    for _ in range(100):
        job = runner.get_job(job_id)
        if job and job["status"] == "running":
            running.set()
            break
        threading.Event().wait(0.01)

    assert running.is_set()
    assert runner.cancel(job_id) is True
    for _ in range(100):
        job = runner.get_job(job_id)
        if job and job["status"] == "cancelled":
            break
        threading.Event().wait(0.01)
    runner.shutdown()

    assert job is not None and job["status"] == "cancelled"
