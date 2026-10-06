"""Queue dashboard pipeline jobs and run allowlisted CLI commands safely."""

from __future__ import annotations

import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from scripts.utils.config import BaseConfig

JOB_TYPES = {
    "ingest_thesis",
    "build_thesis_graph",
    "build_consistency_graph",
    "revalidate_references",
}
_THESIS_ID_PATTERN = re.compile(r"^[^\x00-\x1f\x7f/\\]{1,255}$")
_PROFILE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_REFERENCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")


def _reject_unknown_options(options: dict[str, Any], allowed: set[str]) -> None:
    """Raise an error if any keys in options are not in the allowed set.
    Args:
        options: The dictionary of options to check.
        allowed: The set of allowed option keys.

    Raises:
        ValueError: If any keys in options are not in the allowed set.
    """
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unsupported options: {', '.join(sorted(unknown))}")


def _is_valid_thesis_id(thesis_id: Any) -> bool:
    """Validate a thesis ID derived from a filename stem without allowing paths.

    Args:
        thesis_id: The thesis ID to validate.

    Returns:
        True if the thesis ID is valid, False otherwise.
    """
    return (
        isinstance(thesis_id, str)
        and bool(thesis_id.strip())
        and thesis_id not in {".", ".."}
        and _THESIS_ID_PATTERN.fullmatch(thesis_id) is not None
    )


def _boolean_option(options: dict[str, Any], name: str) -> bool:
    """Retrieve a boolean option from the options dictionary.

    Args:
        options: The dictionary of options to check.
        name: The name of the boolean option.

    Returns:
        The boolean value of the option, or False if not present.

    Raises:
        ValueError: If the option is present but not a boolean.
    """
    value = options.get(name, False)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _bounded_number(
    options: dict[str, Any], name: str, minimum: int | float, maximum: int | float
) -> int | float | None:
    """Retrieve a numeric option from the options dictionary, ensuring it is within bounds.

    Args:
        options: The dictionary of options to check.
        name: The name of the numeric option.
        minimum: The minimum allowed value.
        maximum: The maximum allowed value.

    Returns:
        The numeric value of the option, or None if not present.

    Raises:
        ValueError: If the option is present but not numeric or out of bounds.
    """
    value = options.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def build_job_command(job_type: str, options: dict[str, Any], project_root: Path) -> list[str]:
    """Build a subprocess argument list from a job type and whitelisted options.

    Args:
        job_type: The type of job to build the command for.
        options: The dictionary of whitelisted options for the job.
        project_root: The root path of the project.

    Returns:
        A list of strings representing the subprocess command.

    Raises:
        ValueError: If the job type is unsupported or any options are invalid.
    """
    if job_type not in JOB_TYPES:
        raise ValueError(f"Unsupported job type: {job_type}")

    command = [sys.executable, "-m"]
    if job_type == "ingest_thesis":
        _reject_unknown_options(
            options,
            {
                "paper_path",
                "dry_run",
                "reset",
                "cultural_lens",
                "skip_citations",
                "skip_terminology",
            },
        )
        paper_path_value = options.get("paper_path")
        if not isinstance(paper_path_value, str) or not paper_path_value.strip():
            raise ValueError("paper_path is required")
        paper_path = Path(paper_path_value).expanduser().resolve(strict=True)
        papers_root = (project_root / "data_raw" / "academic_papers").resolve()
        if paper_path.suffix.casefold() != ".pdf" or not paper_path.is_file():
            raise ValueError("paper_path must identify a PDF file")
        if not paper_path.is_relative_to(papers_root):
            raise ValueError("paper_path must be inside data_raw/academic_papers")
        command.extend(["scripts.ingest.ingest_academic", "--papers", str(paper_path)])
        for option_name, flag in (
            ("dry_run", "--dry-run"),
            ("reset", "--reset"),
            ("skip_citations", "--skip-citations"),
            ("skip_terminology", "--skip-terminology"),
        ):
            if _boolean_option(options, option_name):
                command.append(flag)
        cultural_lens = options.get("cultural_lens")
        if cultural_lens is not None:
            if not isinstance(cultural_lens, str) or not _PROFILE_ID_PATTERN.fullmatch(
                cultural_lens
            ):
                raise ValueError("cultural_lens must be a valid profile ID")
            command.extend(["--cultural-lens", cultural_lens])
        return command

    if job_type == "build_thesis_graph":
        _reject_unknown_options(options, {"thesis_id"})
        thesis_id = options.get("thesis_id")
        if not isinstance(thesis_id, str) or not _is_valid_thesis_id(thesis_id):
            raise ValueError("thesis_id must be a valid identifier")
        command.extend(["scripts.thesis_graph.build_thesis_evidence_graph", thesis_id])
        return command

    if job_type == "revalidate_references":
        _reject_unknown_options(options, {"mode", "staleness_threshold", "ref_ids", "thesis_id"})
        mode = options.get("mode")
        if mode not in {"stale", "online", "all", "failed", "ids"}:
            raise ValueError("mode must be stale, online, all, failed or ids")
        command.extend(["scripts.ingest.ingest_academic", "--revalidate", mode])
        thesis_id = options.get("thesis_id")
        if thesis_id is not None:
            if not isinstance(thesis_id, str) or not _is_valid_thesis_id(thesis_id):
                raise ValueError("thesis_id must be a valid identifier")
            command.extend(["--thesis-id", thesis_id])
        threshold = _bounded_number(options, "staleness_threshold", 1, 3650)
        if threshold is not None:
            command.extend(["--staleness-threshold", str(int(threshold))])
        reference_ids = options.get("ref_ids", [])
        if not isinstance(reference_ids, list) or any(
            not isinstance(reference_id, str) or not _REFERENCE_ID_PATTERN.fullmatch(reference_id)
            for reference_id in reference_ids
        ):
            raise ValueError("ref_ids must be a list of valid reference IDs")
        if mode == "ids" and not reference_ids:
            raise ValueError("ref_ids are required when mode is ids")
        if reference_ids:
            command.append("--ref-ids")
            command.extend(reference_ids)
        return command

    _reject_unknown_options(
        options,
        {
            "reset",
            "max_neighbours",
            "similarity_threshold",
            "workers",
            "include_advanced_analytics",
        },
    )
    command.append("scripts.consistency_graph.build_consistency_graph")
    for option_name, flag, minimum, maximum, integer in (
        ("max_neighbours", "--max-neighbours", 1, 200, True),
        ("similarity_threshold", "--similarity-threshold", 0, 2, False),
        ("workers", "--workers", 1, 64, True),
    ):
        value = _bounded_number(options, option_name, minimum, maximum)
        if value is not None:
            if integer and not isinstance(value, int):
                raise ValueError(f"{option_name} must be an integer")
            command.extend([flag, str(value)])
    if _boolean_option(options, "reset"):
        command.append("--reset")
    if _boolean_option(options, "include_advanced_analytics"):
        command.append("--include-advanced-analytics")
    return command


class JobRunner:
    """Persist and serially execute dashboard jobs as isolated CLI processes.

    This class manages the lifecycle of jobs, including submission, execution,
    cancellation, and recovery of orphaned jobs. Jobs are executed as separate
    CLI processes, and their status and metadata are stored in a SQLite database.

    Attributes:
        project_root: The root path of the project.
        database_path: The path to the SQLite database for job metadata.
        logs_dir: The directory where job logs are stored.
    """

    def __init__(
        self,
        project_root: Path | None = None,
        database_path: Path | None = None,
        logs_dir: Path | None = None,
        start_worker: bool = True,
    ) -> None:
        """Initialise the JobRunner instance.

        Args:
            project_root: The root path of the project.
            database_path: The path to the SQLite database for job metadata.
            logs_dir: The directory where job logs are stored.
            start_worker: Whether to start the worker thread immediately.
        """
        config = BaseConfig()
        self.project_root = Path(project_root or config.project_root).resolve()
        self.database_path = Path(database_path or Path(config.rag_data_path) / "jobs.db").resolve()
        self.logs_dir = Path(logs_dir or self.project_root / "logs" / "jobs").resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._condition = threading.Condition()
        self._stopping = False
        self._active_job_id: str | None = None
        self._active_process: subprocess.Popen[str] | None = None
        self._initialise_database()
        self._recover_orphaned_jobs()
        self._worker: threading.Thread | None = None
        if start_worker:
            self.start()

    def _connect(self) -> sqlite3.Connection:
        """Connect to the SQLite database and return the connection.

        Returns:
            A sqlite3.Connection object connected to the job database.
        """
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for a SQLite database connection.

        Yields:
            A sqlite3.Connection object connected to the job database.
        """
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialise_database(self) -> None:
        """Initialise the jobs database schema if it does not already exist."""
        with self._connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    thesis_id TEXT,
                    arguments_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    exit_code INTEGER,
                    log_path TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    worker_pid INTEGER,
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                )
                """)

    def _recover_orphaned_jobs(self) -> None:
        """Reconcile running jobs whose dashboard worker process has disappeared."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT job_id, job_type, run_id, worker_pid FROM jobs WHERE status = 'running'"
            ).fetchall()
            for row in rows:
                worker_pid = row["worker_pid"]
                if worker_pid and self._process_exists(worker_pid):
                    continue
                succeeded = self._has_successful_completion_event(row["job_type"], row["run_id"])
                connection.execute(
                    """UPDATE jobs SET status = ?, finished_at = ?,
                       exit_code = ?, worker_pid = NULL
                       WHERE job_id = ? AND status = 'running'""",
                    (
                        "succeeded" if succeeded else "failed",
                        self._now(),
                        0 if succeeded else -1,
                        row["job_id"],
                    ),
                )

    def _has_successful_completion_event(self, job_type: str, run_id: str) -> bool:
        """Check for a successful completion event for supported job types.

        Args:
            job_type: The type of job to check for completion.
            run_id: The run ID associated with the job.

        Returns:
            True if a successful completion event is found, False otherwise.
        """
        expected_event = {"ingest_thesis": "complete"}.get(job_type)
        if expected_event is None:
            return False

        for audit_path in (self.project_root / "logs").glob("*_audit.jsonl"):
            try:
                with audit_path.open("r", encoding="utf-8") as audit_file:
                    for line in audit_file:
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if event.get("run_id") == run_id and event.get("event") == expected_event:
                            return True
            except OSError:
                continue
        return False

    @staticmethod
    def _process_exists(process_id: int) -> bool:
        """Check if a process with the given PID exists.

        Args:
            process_id: The PID of the process to check.

        Returns:
            True if the process exists, False otherwise.
        """
        try:
            os.kill(process_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _now() -> str:
        """Get the current UTC time as an ISO 8601 formatted string.

        Returns:
            The current UTC time in ISO 8601 format.
        """
        return datetime.now(timezone.utc).isoformat()

    def start(self) -> None:
        """Start the queue worker if it is not already running."""
        with self._condition:
            if self._worker and self._worker.is_alive():
                return
            self._stopping = False
            self._worker = threading.Thread(
                target=self._work,
                name="dashboard-job-runner",
                daemon=True,
            )
            self._worker.start()

    def submit(self, job_type: str, options: dict[str, Any], thesis_id: str | None = None) -> str:
        """Validate and enqueue a job, returning its stable job ID.

        Args:
            job_type: The type of job to enqueue.
            options: A dictionary of job-specific options.
            thesis_id: An optional thesis identifier associated with the job.

        Returns:
            The stable job ID assigned to the enqueued job.
        """
        build_job_command(job_type, options, self.project_root)
        if thesis_id is not None and not _is_valid_thesis_id(thesis_id):
            raise ValueError("thesis_id must be a valid identifier")
        job_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        log_path = self.logs_dir / f"{job_id}.log"
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO jobs (
                    job_id, job_type, thesis_id, arguments_json, status, created_at,
                    log_path, run_id
                ) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)""",
                (
                    job_id,
                    job_type,
                    thesis_id,
                    json.dumps(options, ensure_ascii=False),
                    self._now(),
                    str(log_path),
                    run_id,
                ),
            )
        with self._condition:
            self._condition.notify_all()
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return a decoded job record, or ``None`` when it does not exist.

        Args:
            job_id: The stable job ID of the job to retrieve.

        Returns:
            A dictionary representing the job record, or ``None`` if the job does not exist.
        """
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._decode_job(row) if row else None

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the newest job records, bounded to a safe page size.

        Args:
            limit: The maximum number of job records to return. Must be between 1 and 500.

        Returns:
            A list of dictionaries representing the job records.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode_job(row) for row in rows]

    @staticmethod
    def _decode_job(row: sqlite3.Row) -> dict[str, Any]:
        job = dict(row)
        job["arguments"] = json.loads(job.pop("arguments_json"))
        job["cancel_requested"] = bool(job["cancel_requested"])
        return job

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued job or terminate the active job process group.

        Args:
            job_id: The stable job ID of the job to cancel.

        Returns:
            True if the job was successfully cancelled, False otherwise.
        """
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET status = 'cancelled', finished_at = ?
                   WHERE job_id = ? AND status = 'queued'""",
                (self._now(), job_id),
            )
            if cursor.rowcount:
                return True
            cursor = connection.execute(
                "UPDATE jobs SET cancel_requested = 1 WHERE job_id = ? AND status = 'running'",
                (job_id,),
            )
            if not cursor.rowcount:
                return False
        with self._condition:
            process = self._active_process if self._active_job_id == job_id else None
        if process is not None:
            self._terminate_process(process)
        return True

    def _work(self) -> None:
        """Worker loop that continuously claims and executes jobs until stopping.

        This method runs in a separate thread and handles job execution, including
        waiting for new jobs when none are available.
        """
        while True:
            with self._condition:
                if self._stopping:
                    return
            job = self._claim_next_job()
            if job is None:
                with self._condition:
                    if self._stopping:
                        return
                    self._condition.wait(timeout=0.5)
                continue
            self._execute(job)

    def _claim_next_job(self) -> dict[str, Any] | None:
        """Claim the next queued job and mark it as running.

        Returns:
            A dictionary representing the claimed job, or ``None`` if no job could be claimed.
        """
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            running = connection.execute(
                "SELECT 1 FROM jobs WHERE status = 'running' LIMIT 1"
            ).fetchone()
            if running:
                connection.commit()
                return None
            row = connection.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at, job_id LIMIT 1"
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            connection.execute(
                """UPDATE jobs SET status = 'running', started_at = ?, worker_pid = ?
                   WHERE job_id = ? AND status = 'queued'""",
                (self._now(), os.getpid(), row["job_id"]),
            )
            connection.commit()
            job = dict(row)
            job["status"] = "running"
            return job
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _execute(self, job: dict[str, Any]) -> None:
        """Execute a claimed job and handle its lifecycle, including logging and status updates.

        Args:
            job: A dictionary representing the job to execute.
        """
        job_id = str(job["job_id"])
        environment = os.environ.copy()
        environment["LLM_RUN_ID"] = str(job["run_id"])
        process: subprocess.Popen[str] | None = None
        return_code = -1
        failure: str | None = None
        try:
            command = build_job_command(
                str(job["job_type"]), json.loads(job["arguments_json"]), self.project_root
            )
            with open(job["log_path"], "a", encoding="utf-8") as log_file:
                log_file.write(f"[{self._now()}] Starting {job['job_type']}\n")
                log_file.flush()
                process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=(os.name != "nt"),
                )
                with self._condition:
                    self._active_job_id = job_id
                    self._active_process = process
                if self._cancel_was_requested(job_id):
                    self._terminate_process(process)
                if process.stdout is not None:
                    for output_line in process.stdout:
                        log_file.write(output_line)
                        log_file.flush()
                return_code = process.wait()
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            with open(job["log_path"], "a", encoding="utf-8") as log_file:
                log_file.write(f"\nJob failed: {failure}\n")
        finally:
            with self._condition:
                self._active_job_id = None
                self._active_process = None
            cancelled = self._cancel_was_requested(job_id)
            status = (
                "cancelled"
                if cancelled
                else "succeeded" if return_code == 0 and failure is None else "failed"
            )
            with self._connection() as connection:
                connection.execute(
                    """UPDATE jobs SET status = ?, finished_at = ?, exit_code = ?,
                       worker_pid = NULL WHERE job_id = ?""",
                    (status, self._now(), return_code, job_id),
                )

    def _cancel_was_requested(self, job_id: str) -> bool:
        """Check if a cancellation was requested for the given job.

        Args:
            job_id: The stable job ID of the job to check.

        Returns:
            True if a cancellation was requested, False otherwise.
        """
        with self._connection() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return bool(row and row[0])

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        """Terminate the given subprocess, attempting a graceful shutdown first and
        forcefully killing it if necessary.

        Args:
            process: The subprocess to terminate.
        """
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def shutdown(self, wait: bool = True) -> None:
        """Stop accepting work in this worker and optionally wait for completion.

        Args:
            wait: If True, block until the worker thread has finished executing.
        """
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
            worker = self._worker
        if wait and worker and worker is not threading.current_thread():
            worker.join()
