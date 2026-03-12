"""Unified Git ingestion entry point supporting multiple providers.

Supports ingesting code from:
- Bitbucket Cloud / Server
- GitHub Cloud / Enterprise
- GitLab Cloud / Self-hosted (future)
- Azure DevOps (future)

Provider selection via GIT_PROVIDER environment variable or CLI argument.
Shares common configuration and code parsing logic across all providers.

Usage:
    python -m scripts.ingest.ingest_git --provider bitbucket --host ... --branch main
    python -m scripts.ingest.ingest_git --provider github --token ... --owner myorg
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is importable when running as a script path, e.g.:
# python scripts/ingest/ingest_git.py --help
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chardet
import chromadb
import psutil

from scripts.ingest.bm25_indexing import index_chunks_in_bm25
from scripts.ingest.chunk import chunk_text, create_parent_child_chunks
from scripts.ingest.embedding_cache import EmbeddingCache
from scripts.ingest.git.bitbucket_git_connector import BitbucketGitConnector
from scripts.ingest.git.code_parser import CodeParser
from scripts.ingest.git.git_connector import GitConnector, GitProject
from scripts.ingest.git.git_ingest_config import GitIngestConfig
from scripts.ingest.git.github_connector import GitHubGitConnector
from scripts.ingest.llm_cache import LLMCache
from scripts.ingest.vectors import (
    EMBEDDING_MODEL_NAME,
    get_existing_doc_hash,
    store_child_chunks,
    store_chunks_in_chroma,
    store_parent_chunks,
)
from scripts.utils.clear_databases import clear_for_ingestion
from scripts.utils.db_factory import get_cache_client, get_default_vector_path, get_vector_client
from scripts.utils.logger import create_module_logger
import scripts.utils.logger as logger_module

try:
    from scripts.security.dlp import DLPScanner
except ImportError:
    DLPScanner = None  # type: ignore[assignment]

get_logger, audit = create_module_logger("ingest")
logger = get_logger(log_to_console=True)  # Enable console output by default


# =========================
# RESOURCE MONITORING & METRICS
# =========================


@dataclass
class ResourceMetrics:
    """Resource usage metrics for ingestion process."""

    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    thread_count: int = 0

    def __str__(self) -> str:
        return f"CPU: {self.cpu_percent:.1f}% | Memory: {self.memory_percent:.1f}% ({self.memory_mb:.0f}MB) | Threads: {self.thread_count}"


@dataclass
class ProcessingStats:
    """Statistics for file processing in ingestion."""

    total_files: int = 0
    files_processed: int = 0
    files_failed: int = 0
    files_skipped: int = 0
    chunks_created: int = 0
    total_time: float = 0.0
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    start_time: float = field(default_factory=time.time)
    metrics_history: List[ResourceMetrics] = field(default_factory=list)

    def record_metric(self, metric: ResourceMetrics) -> None:
        """Record a resource metric snapshot."""
        self.metrics_history.append(metric)
        self.peak_cpu_percent = max(self.peak_cpu_percent, metric.cpu_percent)
        self.peak_memory_mb = max(self.peak_memory_mb, metric.memory_mb)

    def finalise(self) -> None:
        """Finalise stats after processing completes."""
        self.total_time = time.time() - self.start_time

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        avg_memory = (
            sum(m.memory_mb for m in self.metrics_history) / len(self.metrics_history)
            if self.metrics_history
            else 0.0
        )
        return {
            "total_files": self.total_files,
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "files_skipped": self.files_skipped,
            "chunks_created": self.chunks_created,
            "total_time_seconds": self.total_time,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_memory_mb": avg_memory,
            "peak_cpu_percent": self.peak_cpu_percent,
            "files_per_second": (
                self.files_processed / self.total_time if self.total_time > 0 else 0
            ),
        }


class ResourceMonitor(threading.Thread):
    """Background thread that monitors system resource usage."""

    def __init__(self, interval: float = 5.0, stats: Optional[ProcessingStats] = None):
        """Initialise resource monitor.

        Args:
            interval: Sampling interval in seconds
            stats: ProcessingStats to record metrics into
        """
        super().__init__(daemon=True)
        self.interval = interval
        self.stats = stats
        self._running = True
        self._process = psutil.Process()

    def run(self) -> None:
        """Monitor resources until stopped."""
        try:
            while self._running:
                try:
                    memory_info = self._process.memory_info()
                    total_memory = psutil.virtual_memory().total
                    memory_mb = memory_info.rss / 1024 / 1024
                    memory_percent = (memory_mb * 1024 * 1024 / total_memory) * 100

                    metric = ResourceMetrics(
                        timestamp=time.time(),
                        cpu_percent=self._process.cpu_percent(interval=0.1),
                        memory_percent=memory_percent,
                        memory_mb=memory_mb,
                        thread_count=threading.active_count(),
                    )

                    if self.stats:
                        self.stats.record_metric(metric)

                    time.sleep(self.interval)
                except Exception as e:
                    logger.debug(f"Resource monitor error: {e}")
        except Exception as e:
            logger.warning(f"Resource monitor failed: {e}")

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False


class ErrorRecoveryStrategy:
    """Handles error recovery and aggregation during processing."""

    def __init__(self):
        """Initialise error recovery tracker."""
        self.errors = []
        self.warnings = []
        self.recovered_count = 0
        self.failed_count = 0

    def record_error(
        self,
        file_path: str,
        error: Exception,
        stage: str = "processing",
        recoverable: bool = True,
    ) -> None:
        """Record an error for later aggregation.

        Args:
            file_path: File where error occurred
            error: The exception
            stage: Processing stage (parsing, chunking, storage, etc)
            recoverable: Whether this error was recovered from
        """
        error_info = {
            "file_path": file_path,
            "error_type": type(error).__name__,
            "error_message": str(error)[:200],
            "stage": stage,
            "recoverable": recoverable,
            "timestamp": time.time(),
        }

        self.errors.append(error_info)

        if recoverable:
            self.recovered_count += 1
            logger.warning(f"Recovered from error in {file_path}: {error_info['error_type']}")
        else:
            self.failed_count += 1
            logger.error(f"Failed to process {file_path}: {error_info['error_type']}")

    def record_warning(self, file_path: str, warning: str, stage: str = "processing") -> None:
        """Record a warning for later review.

        Args:
            file_path: File associated with warning
            warning: Warning message
            stage: Processing stage
        """
        warning_info = {
            "file_path": file_path,
            "warning": warning[:200],
            "stage": stage,
            "timestamp": time.time(),
        }
        self.warnings.append(warning_info)

    def summary(self) -> Dict[str, Any]:
        """Get summary of all errors and warnings.

        Returns:
            Dictionary with error/warning counts and details
        """
        return {
            "total_errors": len(self.errors),
            "recovered_errors": self.recovered_count,
            "failed_errors": self.failed_count,
            "total_warnings": len(self.warnings),
            "errors": self.errors[:10],  # Top 10 errors
            "warnings": self.warnings[:10],  # Top 10 warnings
        }


class ConcurrentFileProcessor:
    """Processes files concurrently using thread pool."""

    def __init__(
        self,
        max_workers: int = 4,
        chunk_collection: Optional[Any] = None,
        doc_collection: Optional[Any] = None,
        parser: Optional[Any] = None,
        llm_cache: Optional[Any] = None,
        embedding_cache: Optional[Any] = None,
        config: Optional[Any] = None,
        dry_run: bool = False,
        error_recovery: Optional[ErrorRecoveryStrategy] = None,
    ):
        """Initialise concurrent processor.

        Args:
            max_workers: Maximum number of worker threads
            chunk_collection: ChromaDB chunk collection
            doc_collection: ChromaDB doc collection
            parser: CodeParser instance
            llm_cache: Optional LLM cache
            embedding_cache: Optional embedding cache
            config: GitIngestConfig
            dry_run: If True, simulate without persisting
            error_recovery: Optional ErrorRecoveryStrategy for error tracking
        """
        self.max_workers = max_workers
        self.chunk_collection = chunk_collection
        self.doc_collection = doc_collection
        self.parser = parser
        self.llm_cache = llm_cache
        self.embedding_cache = embedding_cache
        self.config = config
        self.dry_run = dry_run
        self.error_recovery = error_recovery or ErrorRecoveryStrategy()
        self.stats = ProcessingStats()
        self._lock = threading.Lock()

    def process_files(
        self,
        files: List[Tuple[str, str, float]],  # [(file_path, content, mtime), ...]
        project_key: str,
        repo_slug: str,
        branch: str,
    ) -> ProcessingStats:
        """Process multiple files concurrently.

        Args:
            files: List of (file_path, content, mtime) tuples
            project_key: Project key for grouping
            repo_slug: Repository slug
            branch: Branch name for URL generation

        Returns:
            ProcessingStats with results
        """
        self.stats.total_files = len(files)

        # Start resource monitor
        monitor = ResourceMonitor(interval=2.0, stats=self.stats)
        monitor.start()

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(
                        self._process_file_wrapper,
                        file_path,
                        content,
                        mtime,
                        project_key,
                        repo_slug,
                        branch,
                    ): file_path
                    for file_path, content, mtime in files
                }

                # Process completions
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        success = future.result()
                        with self._lock:
                            if success:
                                self.stats.files_processed += 1
                            else:
                                self.stats.files_failed += 1
                    except Exception as e:
                        logger.error(f"Task failed for {file_path}: {e}")
                        with self._lock:
                            self.stats.files_failed += 1
        finally:
            monitor.stop()
            monitor.join(timeout=2.0)
            self.stats.finalise()

        return self.stats

    def _process_file_wrapper(
        self,
        file_path: str,
        content: str,
        mtime: float,
        project_key: str,
        repo_slug: str,
        branch: str,
    ) -> bool:
        """Wrapper for process_code_file to handle thread-safety, encoding, and dry-run."""
        try:
            # Detect and handle encoding if enabled
            if self.config and getattr(self.config, "detect_encoding", True):
                try:
                    content_bytes = content.encode("utf-8") if isinstance(content, str) else content
                    detected_encoding, confidence = EncodingDetector.detect_encoding(
                        file_path, content_bytes
                    )
                    if confidence < 0.5:
                        # Low confidence, try fallback decoding
                        logger.debug(
                            f"Low confidence encoding detection for {file_path}: {detected_encoding} ({confidence:.2%})"
                        )
                        if isinstance(content, bytes):
                            content = EncodingDetector.decode_with_fallback(
                                content, file_path, detected_encoding
                            )
                except Exception as e:
                    logger.warning(
                        f"Encoding detection failed for {file_path}: {e}, continuing with current content"
                    )
                    self.error_recovery.record_warning(
                        file_path, f"Encoding detection failed: {e}", stage="encoding_detection"
                    )

            return process_code_file(
                file_path=file_path,
                file_content=content,
                project_key=project_key,
                repo_slug=repo_slug,
                branch=branch,
                chunk_collection=self.chunk_collection,
                doc_collection=self.doc_collection,
                parser=self.parser,
                llm_cache=self.llm_cache,
                embedding_cache=self.embedding_cache,
                config=self.config,
                dry_run=self.dry_run,
                error_recovery=self.error_recovery,
            )
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.error_recovery.record_error(
                file_path, e, stage="file_processing", recoverable=True
            )
            return False


# =========================
# ENCODING & ERROR RECOVERY
# =========================


class EncodingDetector:
    """Detects and handles file encoding with fallback strategies."""

    ENCODING_PRIORITIES = [
        "utf-8",
        "utf-16",
        "latin-1",
        "cp1252",
        "iso-8859-1",
        "gbk",
        "shift_jis",
    ]

    @staticmethod
    def detect_encoding(file_path: str, content_bytes: Optional[bytes] = None) -> Tuple[str, float]:
        """Detect file encoding with confidence score.

        Args:
            file_path: Path to file for logging
            content_bytes: Optional bytes to analyse (otherwise read from file)

        Returns:
            Tuple of (encoding, confidence_score)
        """
        try:
            if content_bytes is None:
                with open(file_path, "rb") as f:
                    content_bytes = f.read(100000)  # Read first 100KB for detection

            if not content_bytes:
                return "utf-8", 1.0

            # Try chardet detection
            detected = chardet.detect(content_bytes)
            if detected and detected.get("encoding"):
                encoding = detected["encoding"]
                confidence = detected.get("confidence", 0.0)

                # Normalise encoding names
                encoding = encoding.lower().replace("-", "_").replace("_", "-")
                logger.debug(f"Detected encoding for {file_path}: {encoding} ({confidence:.2%})")
                return encoding, confidence

            return "utf-8", 0.0
        except Exception as e:
            logger.debug(f"Encoding detection failed for {file_path}: {e}")
            return "utf-8", 0.0

    @staticmethod
    def decode_with_fallback(
        content_bytes: bytes,
        file_path: str,
        preferred_encoding: Optional[str] = None,
    ) -> str:
        """Decode bytes to string with fallback encodings.

        Args:
            content_bytes: Raw bytes to decode
            file_path: File path for logging
            preferred_encoding: Preferred encoding to try first

        Returns:
            Decoded string

        Raises:
            ValueError: If all encodings fail
        """
        encodings_to_try = []

        # Add preferred encoding first if provided
        if preferred_encoding:
            encodings_to_try.append(preferred_encoding)

        # Add standard priorities
        encodings_to_try.extend(EncodingDetector.ENCODING_PRIORITIES)

        # Remove duplicates while preserving order
        encodings_to_try = list(dict.fromkeys(encodings_to_try))

        last_error = None
        for encoding in encodings_to_try:
            try:
                decoded = content_bytes.decode(encoding, errors="strict")
                logger.debug(f"Successfully decoded {file_path} with {encoding}")
                return decoded
            except (UnicodeDecodeError, LookupError) as e:
                last_error = e
                logger.debug(f"Failed to decode {file_path} with {encoding}")
                continue

        # Final fallback: decode with errors='replace'
        try:
            logger.warning(f"Using lossy decode for {file_path} (unknown encoding)")
            return content_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            raise ValueError(
                f"Failed to decode {file_path}: {last_error}. "
                f"File may be binary or have incompatible encoding."
            )


# =========================
# SECURITY & UTILITY FUNCTIONS
# =========================


def obfuscate_password(value: str) -> str:
    """Obfuscate password for logging.

    Args:
        value: Password or token string

    Returns:
        Obfuscated string showing only first 2 and last 2 characters
    """
    if not value or len(value) <= 4:
        return "****"
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


def sanitise_args_for_logging(args: argparse.Namespace) -> Dict[str, Any]:
    """Sanitise command-line arguments for safe logging.

    Removes or obfuscates sensitive data like passwords and tokens.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dictionary with sanitised values
    """
    args_dict = vars(args).copy()

    # Obfuscate sensitive credentials
    if "password" in args_dict and args_dict["password"]:
        args_dict["password"] = obfuscate_password(args_dict["password"])
    if "token" in args_dict and args_dict["token"]:
        args_dict["token"] = obfuscate_password(args_dict["token"])
    if "api_username" in args_dict and args_dict.get("api_username"):
        # Don't log email addresses
        args_dict["api_username"] = "***REDACTED***"

    return args_dict


def is_test_file(path: str) -> bool:
    """Heuristic to detect test files by path or filename.

    Rules:
    - Any path segment named: test, tests, __tests__, src/test
    - Filenames ending with: Test.java, Tests.java, IT.java, Spec.java, etc.
    - Filenames containing: .test., .spec.
    """
    try:
        p_norm = path.replace("\\", "/")
        p_lower = p_norm.lower()
        parts = [part for part in p_lower.split("/") if part]

        # Directory-based checks (segment-aware)
        if ("src" in parts and "test" in parts) or (
            "test" in parts or "tests" in parts or "__tests__" in parts
        ):
            return True

        filename = os.path.basename(p_norm)
        name_lower = filename.lower()

        # Dot-style qualifiers
        if ".test." in name_lower or ".spec." in name_lower:
            return True

        # Filename suffixes for various languages
        test_suffixes = (
            "test.java",
            "tests.java",
            "it.java",
            "spec.java",
            "test.groovy",
            "tests.groovy",
            "it.groovy",
            "spec.groovy",
            "test.ts",
            "test.tsx",
            "test.js",
            "test.jsx",
            "spec.ts",
            "spec.js",
            "test.py",
            "spec.rb",
        )
        return any(name_lower.endswith(suffix) for suffix in test_suffixes)
    except Exception:
        return False


def infer_language_from_extension(file_path: str) -> str:
    """Infer programming language from file extension.

    Args:
        file_path: Path to the file

    Returns:
        Lowercase language name or "unknown" if not inferred
    """
    ext = Path(file_path).suffix.lower()
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".groovy": "groovy",
        ".gvy": "groovy",
        ".gsp": "groovy",
        ".gradle": "gradle",
        ".xml": "xml",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".json": "json",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".sql": "sql",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".ps1": "powershell",
        ".psm1": "powershell",
        ".tf": "terraform",
        ".sh": "shell",
        ".md": "markdown",
    }
    return mapping.get(ext, "unknown")


def redact_code_content(
    text: str,
    artifact_path: str,
    doc_id: str,
    repository: str,
    enable_dlp: bool = True,
) -> Tuple[str, Dict[str, int]]:
    """Apply DLP redaction to code content if enabled.

    Args:
        text: Code content to redact
        artifact_path: File path for logging
        doc_id: Document ID for audit trail
        repository: Repository name for context
        enable_dlp: Whether DLP is enabled

    Returns:
        Tuple of (redacted_text, match_counts)
    """
    if not enable_dlp:
        return text, {}

    try:
        if DLPScanner is None:
            logger.debug("DLPScanner not available, skipping redaction")
            return text, {}

        dlp_scanner = DLPScanner()
        matches = dlp_scanner.find(text)

        if not matches:
            return text, {}

        counts = {name: len(items) for name, items in matches.items()}
        redacted = dlp_scanner.redact(text)

        logger.info(f"DLP redaction applied to {artifact_path} (doc_id={doc_id}): {counts}")
        audit(
            "dlp_redaction_applied",
            {
                "doc_id": doc_id,
                "path": artifact_path,
                "repository": repository,
                "match_types": list(counts.keys()),
                "match_counts": counts,
            },
        )

        return redacted, counts
    except Exception as e:
        logger.warning(f"DLP redaction failed for {artifact_path}: {e}")
        return text, {}


def generate_code_summary(
    parse_result: Dict[str, Any],
    file_path: str,
    language: str,
    use_llm: bool = False,
    llm_cache: Optional[LLMCache] = None,
) -> Dict[str, Any]:
    """Generate summary and metadata from code parse results.

    Args:
        parse_result: Parsed code metadata from CodeParser
        file_path: File path for context
        language: Programming language
        use_llm: Whether to use LLM for summary generation
        llm_cache: Optional LLM cache

    Returns:
        Dictionary with summary, key_topics, and scores
    """
    try:
        # Extract key information from parse result
        service_name = parse_result.get("service_name", "")
        service_type = parse_result.get("service_type", "")
        exports = parse_result.get("exports", [])
        endpoints = parse_result.get("endpoints", [])
        external_deps = parse_result.get("external_dependencies", [])

        # Generate template-based summary
        parts = []
        if service_name:
            parts.append(f"Service: {service_name}")
        if service_type:
            parts.append(f"Type: {service_type}")
        if exports:
            parts.append(f"Exports: {len(exports)} items")
        if endpoints:
            parts.append(f"Endpoints: {len(endpoints)} items")
        if external_deps:
            parts.append(f"Dependencies: {len(external_deps)} external")

        summary = " | ".join(parts) if parts else f"Code file: {file_path}"

        # Extract key topics
        key_topics = []
        if exports:
            key_topics.extend([str(e)[:30] for e in exports[:5]])
        if endpoints:
            key_topics.extend([str(e)[:30] for e in endpoints[:5]])
        if external_deps:
            key_topics.extend([str(d)[:30] for d in external_deps[:5]])

        return {
            "summary": summary,
            "key_topics": list(set(key_topics[:10])),  # Top 10 unique topics
            "scores": {
                "completeness": 0.7 if (exports or endpoints) else 0.3,
                "clarity": 0.8 if summary else 0.4,
                "technical_depth": 0.6 if external_deps else 0.3,
            },
        }
    except Exception as e:
        logger.debug(f"Summary generation failed: {e}")
        return {
            "summary": f"Code file: {file_path}",
            "key_topics": [language] if language else [],
            "scores": {"completeness": 0.0, "clarity": 0.0, "technical_depth": 0.0},
        }


class GitConnectorFactory:
    """Factory for creating Git connectors based on provider type."""

    _connectors: Dict[str, type] = {
        "bitbucket": BitbucketGitConnector,
        "github": GitHubGitConnector,
        "gitlab": None,  # Future implementation
        "azure": None,  # Future implementation
    }

    @classmethod
    def create(cls, provider: str, config: GitIngestConfig) -> GitConnector:
        """Create Git connector for specified provider.

        Args:
            provider: Provider name (bitbucket, github, gitlab, azure)
            config: GitIngestConfig with provider-specific settings

        Returns:
            GitConnector instance for the provider

        Raises:
            ValueError: If provider not supported or not implemented
        """
        provider = provider.lower()

        if provider not in cls._connectors:
            raise ValueError(f"Unknown Git provider: {provider}")

        connector_class = cls._connectors[provider]
        if connector_class is None:
            raise ValueError(f"Git provider '{provider}' not yet implemented")

        logger.info(f"Creating {provider} connector")

        if provider == "bitbucket":
            return BitbucketGitConnector(
                host=config.git_host,
                username=config.bitbucket_username,
                password=config.bitbucket_password,
                is_cloud=config.bitbucket_is_cloud,
                verify_ssl=config.bitbucket_verify_ssl,
                api_username=getattr(config, "bitbucket_api_username", config.bitbucket_username),
            )

        elif provider == "github":
            return GitHubGitConnector(
                host=config.github_host,
                token=config.github_token,
                api_url=config.github_api_url or "https://api.github.com",
                verify_ssl=config.github_verify_ssl,
            )

        else:
            raise ValueError(f"Provider '{provider}' not yet implemented")


def compute_code_doc_id(file_path: str, repository: str, project_key: str) -> str:
    """Generate a stable document identifier for code files.

    Creates a unique ID that includes repository context to avoid
    collisions when ingesting multiple repositories.

    Args:
        file_path: Relative path within repository (e.g., "src/main/java/MyService.java")
        repository: Repository slug
        project_key: Project key/slug

    Returns:
        Document identifier (e.g., "PROJ_my-service_src_main_java_MyService.java")
    """
    # Normalise path separators
    normalised_path = file_path.replace("\\", "/")

    # Create namespaced ID: project_repo_path
    doc_id = f"{project_key}_{repository}_{normalised_path}"

    # Replace special characters with underscores
    doc_id = doc_id.replace("/", "_").replace(" ", "_").replace(".", "_").replace("-", "_")

    return doc_id


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of file contents for change detection.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except Exception as e:
        # For files that can't be read, use path as fallback
        h.update(file_path.encode("utf-8"))
    return h.hexdigest()


def build_git_file_url(
    provider: str,
    host: str,
    project_key: str,
    repo_slug: str,
    branch: str,
    file_path: str,
) -> Optional[str]:
    """Build a Git hosting URL for a file.

    Args:
        provider: Git provider name (bitbucket, github, gitlab, azure)
        host: Base host URL for the provider
        project_key: Project key or organisation name
        repo_slug: Repository slug
        branch: Branch name
        file_path: File path within the repository

    Returns:
        URL string or None if provider unsupported or inputs missing.
    """
    if not provider or not host or not project_key or not repo_slug or not file_path:
        return None

    normalised_host = host.rstrip("/")
    provider = provider.lower()

    if provider == "bitbucket":
        return (
            f"{normalised_host}/projects/{project_key}/repos/{repo_slug}/browse/{file_path}"
            f"?at=refs/heads/{branch}"
        )
    if provider == "github":
        return f"{normalised_host}/{project_key}/{repo_slug}/blob/{branch}/{file_path}"
    if provider == "gitlab":
        return f"{normalised_host}/{project_key}/{repo_slug}/-/blob/{branch}/{file_path}"
    if provider == "azure":
        return (
            f"{normalised_host}/{project_key}/_git/{repo_slug}"
            f"?path=/{file_path}&version=GB{branch}"
        )

    return None


def get_git_host_for_metadata(config: GitIngestConfig) -> str:
    """Resolve Git host for metadata based on provider settings."""
    provider = (config.git_provider or "").lower()

    if provider == "bitbucket":
        return config.bitbucket_host or config.git_host
    if provider == "github":
        return config.github_host
    if provider == "gitlab":
        return getattr(config, "gitlab_host", config.git_host)
    if provider == "azure":
        return getattr(config, "azure_host", config.git_host)

    return config.git_host


def process_code_file(
    file_path: str,
    file_content: str,
    project_key: str,
    repo_slug: str,
    branch: str,
    chunk_collection,
    doc_collection,
    parser: CodeParser,
    llm_cache: Optional[LLMCache],
    embedding_cache: Optional[EmbeddingCache],
    config: GitIngestConfig,
    dry_run: bool = False,
    error_recovery: Optional[ErrorRecoveryStrategy] = None,
) -> bool:
    """Process a single code file through the ingestion pipeline.

    Args:
        file_path: Path to file
        file_content: File content (should already be decoded)
        project_key: Project key
        repo_slug: Repository slug
        branch: Branch name for URL generation
        chunk_collection: ChromaDB chunks collection
        doc_collection: ChromaDB docs collection
        parser: CodeParser instance
        llm_cache: Optional LLM cache
        embedding_cache: Optional embedding cache
        config: GitIngestConfig
        dry_run: If True, simulate without persisting (default: False)
        error_recovery: Optional ErrorRecoveryStrategy for error tracking

    Returns:
        True if successful, False otherwise
    """
    recovery = error_recovery or ErrorRecoveryStrategy()

    try:
        start_time = time.time()

        # Compute doc ID and hash
        doc_id = compute_code_doc_id(file_path, repo_slug, project_key)
        file_hash = hashlib.sha256(file_content.encode("utf-8")).hexdigest()

        if dry_run:
            logger.debug(f"[DRY-RUN] Would process {file_path} (doc_id={doc_id})")

        # Check if file unchanged
        if not dry_run:
            existing_hash = get_existing_doc_hash(doc_id, chunk_collection)
            if existing_hash == file_hash:
                logger.debug(f"SKIP {file_path} (unchanged)")
                return True

        # Parse code file
        temp_file = Path(f"/tmp/{repo_slug}_{file_path.replace('/', '_')}")
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file.write_text(file_content)

        parse_result = parser.parse_file(str(temp_file))
        temp_file.unlink()

        # Fallback language inference for unsupported parsers (e.g., Python)
        language = (parse_result.language or "").strip().lower()
        if not language or language == "unknown":
            language = infer_language_from_extension(file_path)
            parse_result.language = language

        preprocess_time = time.time()
        preprocess_duration = preprocess_time - start_time

        # DLP redaction (avoid storing/processing secrets)
        file_content_original = file_content
        file_content, dlp_counts = redact_code_content(
            text=file_content,
            artifact_path=file_path,
            doc_id=doc_id,
            repository=repo_slug,
            enable_dlp=config.enable_dlp,
        )

        # Generate summary
        summary_data = generate_code_summary(
            parse_result=(
                parse_result.to_dict() if hasattr(parse_result, "to_dict") else parse_result
            ),
            file_path=file_path,
            language=parse_result.language,
            use_llm=config.generate_summaries,
            llm_cache=llm_cache,
        )

        # Chunk code
        chunks = chunk_text(file_content, doc_type=parse_result.language, adaptive=True)

        if not chunks:
            logger.warning(f"No chunks created for {file_path}")
            return False

        # Store in ChromaDB
        branch_name = branch or config.git_branch
        git_host = get_git_host_for_metadata(config)
        git_url = build_git_file_url(
            provider=config.git_provider,
            host=git_host,
            project_key=project_key,
            repo_slug=repo_slug,
            branch=branch_name,
            file_path=file_path,
        )

        metadata = {
            "doc_id": doc_id,
            "file_path": file_path,
            "repository": repo_slug,
            "project": project_key,
            "project_key": project_key,
            "branch": branch_name,
            "git_provider": config.git_provider,
            "git_host": git_host,
            "file_type": parse_result.file_type,
            "language": parse_result.language,
            "doc_type": "code",
            "source_category": "code",
            "doc_hash": file_hash,
            "version": 1,
            "summary": summary_data.get("summary", ""),
            "key_topics": summary_data.get("key_topics", []),
            "summary_scores": {
                "overall": sum(summary_data.get("scores", {}).values()) / 3,
                "completeness": summary_data.get("scores", {}).get("completeness", 0.0),
                "clarity": summary_data.get("scores", {}).get("clarity", 0.0),
                "technical_depth": summary_data.get("scores", {}).get("technical_depth", 0.0),
            },
            "dlp_applied": config.enable_dlp,
            "dlp_match_count": sum(dlp_counts.values()),
        }

        if git_url:
            metadata["git_url"] = git_url

        ingest_start = time.time()

        # Parent-child chunking (optional, matching ingest_bitbucket.py pattern)
        parent_chunks = None
        child_chunks = None
        using_parent_child = bool(getattr(config, "enable_parent_child_chunking", False))

        if using_parent_child:
            try:
                # create_parent_child_chunks returns (child_chunks, parent_chunks)
                child_chunks, parent_chunks = create_parent_child_chunks(
                    text=file_content, doc_type=parse_result.language
                )
                logger.debug(
                    f"Created {len(child_chunks)} child chunks and {len(parent_chunks)} parent chunks for {file_path}"
                )
            except Exception as e:
                logger.warning(f"Parent-child chunking failed for {file_path}: {e}")
                parent_chunks = None
                child_chunks = None

        # If using parent-child chunking, avoid storing duplicate child texts via generic store
        chunks_to_store = [] if parent_chunks else chunks

        if not dry_run:
            store_chunks_in_chroma(
                doc_id=doc_id,
                file_hash=file_hash,
                source_path=file_path,
                version=1,
                chunks=chunks_to_store,
                metadata=metadata,
                chunk_collection=chunk_collection,
                doc_collection=doc_collection,
                preprocess_duration=preprocess_duration,
                ingest_duration=0.0,  # Will be updated inside the function
                llm_cache=llm_cache,
                embedding_cache=embedding_cache,
                full_text=file_content,
            )
        else:
            logger.debug(f"[DRY-RUN] Would store {len(chunks_to_store)} chunks for {doc_id}")

        # If parent/child created, store child chunks (with embeddings) then parents
        # Following ingest_bitbucket.py pattern: child-first storage
        if parent_chunks:
            base_metadata = {
                "doc_id": doc_id,
                "source": file_path,
                "version": 1,
                "hash": file_hash,
                "doc_type": parse_result.language,
                "embedding_model": EMBEDDING_MODEL_NAME,
            }

            # Store child chunks (searchable, with real embeddings) FIRST
            if child_chunks:
                try:
                    if not dry_run:
                        store_child_chunks(
                            doc_id=doc_id,
                            child_chunks=child_chunks,
                            chunk_collection=chunk_collection,
                            base_metadata=base_metadata,
                            dry_run=False,
                            full_text=file_content,
                            doc_type=parse_result.language,
                        )
                        logger.debug(f"Stored {len(child_chunks)} child chunks for {doc_id}")
                    else:
                        logger.debug(
                            f"[DRY-RUN] Would store {len(child_chunks)} child chunks for {doc_id}"
                        )
                except Exception as child_err:
                    logger.error(f"Failed to store child chunks for {doc_id}: {child_err}")
                    recovery.record_error(
                        file_path, child_err, stage="child_chunks_storage", recoverable=False
                    )
                    audit(
                        "child_chunks_storage_failed",
                        {
                            "doc_id": doc_id,
                            "error": str(child_err)[:200],
                            "error_type": type(child_err).__name__,
                        },
                    )
                    raise

            # Then store parent chunks (metadata/context only; non-fatal)
            try:
                if not dry_run:
                    store_parent_chunks(
                        doc_id=doc_id,
                        parent_chunks=parent_chunks,
                        chunk_collection=chunk_collection,
                        base_metadata=base_metadata,
                        dry_run=False,
                        full_text=file_content,
                        doc_type=parse_result.language,
                    )
                    logger.debug(f"Stored {len(parent_chunks)} parent chunks for {doc_id}")
                else:
                    logger.debug(
                        f"[DRY-RUN] Would store {len(parent_chunks)} parent chunks for {doc_id}"
                    )
            except Exception as parent_err:
                logger.error(f"Failed to store parent chunks for {doc_id}: {parent_err}")
                recovery.record_error(
                    file_path, parent_err, stage="parent_chunks_storage", recoverable=True
                )
                audit(
                    "parent_chunks_storage_failed",
                    {
                        "doc_id": doc_id,
                        "error": str(parent_err)[:200],
                        "error_type": type(parent_err).__name__,
                    },
                )

        # BM25 Keyword Indexing (optional)
        # Uses common indexing utility for chunk-level granularity
        start_bm25_time = time.time()
        if config.bm25_indexing_enabled and not dry_run:
            try:
                cache_db = get_cache_client(enable_cache=True)
                total_indexed = index_chunks_in_bm25(
                    doc_id=doc_id,
                    chunks=chunks,
                    child_chunks=child_chunks,
                    parent_chunks=parent_chunks,
                    config=config,
                    cache_db=cache_db,
                    logger=logger,
                )

                bm25_duration = time.time() - start_bm25_time
                audit(
                    "bm25_indexed",
                    {
                        "doc_id": doc_id,
                        "version": 1,
                        "chunks_indexed": total_indexed,
                        "granularity": "chunk",
                        "language": parse_result.language,
                        "repository": repo_slug,
                        "duration_seconds": bm25_duration,
                    },
                )
            except Exception as e:
                logger.warning(f"BM25 indexing failed for {file_path}: {e}")
                audit(
                    "bm25_index_failed",
                    {
                        "doc_id": doc_id,
                        "repository": repo_slug,
                        "error": str(e)[:200],
                        "error_type": type(e).__name__,
                    },
                )

        logger.info(
            f"Ingested {file_path}: {len(chunks)} chunks, summary: {summary_data['summary'][:50]}"
        )
        return True

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}", exc_info=True)
        return False


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Ingest code from Git repositories (Bitbucket, GitHub, GitLab, Azure)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest from Bitbucket
  ingest_git --provider bitbucket --host https://bitbucket.org \\
             --username user --password token --branch main
  
  # Ingest from GitHub
  ingest_git --provider github --host https://github.com \\
             --token ghp_xxxx --owner myorg --branch main
             
  # Provider can also be set via GIT_PROVIDER environment variable
  GIT_PROVIDER=bitbucket ingest_git --host https://bitbucket.org ...
        """,
    )

    # Provider selection
    parser.add_argument(
        "--provider",
        type=str,
        default=os.getenv("GIT_PROVIDER", "bitbucket"),
        choices=["bitbucket", "github", "gitlab", "azure"],
        help="Git provider (default: $GIT_PROVIDER or 'bitbucket')",
    )

    # Generic Git settings
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Git host URL (default: provider-specific URL)",
    )

    parser.add_argument(
        "--branch",
        type=str,
        default=os.getenv("GIT_BRANCH", "main"),
        help="Default branch to clone (default: $GIT_BRANCH or 'main')",
    )

    parser.add_argument(
        "--clone-dir",
        type=str,
        default=os.getenv("GIT_CLONE_DIR", "./repos"),
        help="Directory for cloned repositories (default: $GIT_CLONE_DIR)",
    )

    parser.add_argument(
        "--reset-repo",
        action="store_true",
        default=os.getenv("GIT_RESET_REPO", "false").lower() == "true",
        help="Reset/re-clone repositories if they exist",
    )

    # Bitbucket-specific
    parser.add_argument(
        "--username",
        type=str,
        default=os.getenv("GIT_USERNAME"),
        help="Bitbucket username for git operations (default: $GIT_USERNAME)",
    )

    parser.add_argument(
        "--api-username",
        type=str,
        default=os.getenv("GIT_API_USERNAME"),
        help="Bitbucket ATLASSIAN EMAIL for API calls (default: $GIT_API_USERNAME, required for Cloud)",
    )

    parser.add_argument(
        "--password",
        type=str,
        default=os.getenv("GIT_PASSWORD"),
        help="Bitbucket password/token (default: $GIT_PASSWORD)",
    )

    parser.add_argument(
        "--is-cloud",
        action="store_true",
        default=os.getenv("GIT_IS_CLOUD", "true").lower() == "true",
        help="Bitbucket Cloud (vs Server) (default: $GIT_IS_CLOUD)",
    )

    # GitHub-specific
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("GIT_TOKEN"),
        help="GitHub Personal Access Token (default: $GIT_TOKEN)",
    )

    parser.add_argument(
        "--owner",
        type=str,
        default=os.getenv("GIT_OWNER"),
        help="GitHub organisation/owner (default: $GIT_OWNER)",
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default=os.getenv("GIT_API_URL"),
        help="GitHub API URL (default: auto-detected or $GIT_API_URL)",
    )

    # Repository selection
    parser.add_argument(
        "--project",
        type=str,
        default=os.getenv("GIT_PROJECT"),
        help="Specific project/workspace to ingest (default: all)",
    )

    parser.add_argument(
        "--repo",
        type=str,
        default=os.getenv("GIT_REPO"),
        help="Specific repository to ingest (default: all)",
    )

    parser.add_argument(
        "--repo-pattern",
        type=str,
        default=os.getenv("GIT_REPO_PATTERN"),
        help="Repository name pattern to match (substring, case-insensitive)",
    )

    parser.add_argument(
        "--repos-file",
        type=str,
        default=os.getenv("GIT_REPOS_FILE"),
        help='File containing repositories to ingest (text: one per line, or JSON: [{"repo": "name", "project": "key", "branch": "main"}])',
    )

    parser.add_argument(
        "--max-repos",
        type=int,
        default=int(os.getenv("GIT_MAX_REPOS", "0")),
        help="Maximum number of repositories to process (0=unlimited)",
    )

    # Connection options
    parser.add_argument(
        "--use-ssh",
        action="store_true",
        default=os.getenv("GIT_USE_SSH", "false").lower() == "true",
        help="Use SSH for cloning instead of HTTPS (default: $GIT_USE_SSH)",
    )

    # Ingestion options
    parser.add_argument(
        "--file-types",
        type=str,
        default=os.getenv(
            "GIT_FILE_TYPES",
            ".java,.groovy,.gvy,.gsp,.gradle,.xml,.properties,.yaml,.yml,.js,.jsx,.ts,.tsx,.cs,.ps1,.psm1,.tf,.sql,.html,.htm,.json,.py,.go,.rs,.rb,.php,.c,.cpp,.h,.hpp",
        ),
        help="File types to ingest (comma-separated, default: $GIT_FILE_TYPES)",
    )

    parser.add_argument(
        "--exclude-tests",
        action="store_true",
        default=os.getenv("GIT_EXCLUDE_TESTS", "true").lower() == "true",
        help="Exclude test files (default: $GIT_EXCLUDE_TESTS)",
    )

    parser.add_argument(
        "--generate-summaries",
        action="store_true",
        default=os.getenv("GIT_GENERATE_SUMMARIES", "false").lower() == "true",
        help="Generate summaries for code (default: $GIT_GENERATE_SUMMARIES)",
    )

    parser.add_argument(
        "--use-llm-summaries",
        action="store_true",
        default=os.getenv("GIT_USE_LLM_SUMMARIES", "false").lower() == "true",
        help="Use LLM to generate summaries (default: $GIT_USE_LLM_SUMMARIES)",
    )

    parser.add_argument(
        "--enable-dlp",
        action="store_true",
        default=os.getenv("GIT_ENABLE_DLP", "false").lower() == "true",
        help="Enable DLP filtering (default: $GIT_ENABLE_DLP)",
    )

    parser.add_argument(
        "--disable-dlp",
        dest="enable_dlp",
        action="store_false",
        help="Disable DLP redaction of code content before processing",
    )

    parser.add_argument(
        "--no-refresh",
        action="store_true",
        default=os.getenv("GIT_NO_REFRESH", "false").lower() == "true",
        help="Don't refresh existing ingests (default: $GIT_NO_REFRESH)",
    )

    # Utility options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=os.getenv("GIT_VERBOSE", "false").lower() == "true",
        help="Enable verbose logging (default: $GIT_VERBOSE)",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Alias for --reset-repo (delete and re-clone repositories)",
    )

    parser.add_argument(
        "--purge-logs",
        action="store_true",
        default=os.getenv("GIT_PURGE_LOGS", "false").lower() == "true",
        help="Purge ingest log files before starting (disabled in Production environment, default: $GIT_PURGE_LOGS)",
    )

    # BM25 Keyword Indexing options
    parser.add_argument(
        "--bm25-indexing",
        action="store_true",
        default=os.getenv("BM25_INDEXING_ENABLED", "true").lower() == "true",
        help="Enable BM25 keyword indexing during ingestion (default: from BM25_INDEXING_ENABLED env var)",
    )

    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        default=os.getenv("GIT_SKIP_BM25", "false").lower() == "true",
        help="Disable BM25 keyword indexing during ingestion (overrides --bm25-indexing and env var)",
    )

    # Phase 3: Operations features
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.getenv("GIT_DRY_RUN", "false").lower() == "true",
        help="Simulate processing without persisting to ChromaDB (default: $GIT_DRY_RUN)",
    )

    parser.add_argument(
        "--detect-encoding",
        action="store_true",
        default=os.getenv("GIT_DETECT_ENCODING", "true").lower() == "true",
        help="Auto-detect file encoding with fallback handling (default: $GIT_DETECT_ENCODING)",
    )

    parser.add_argument(
        "--error-recovery",
        type=str,
        choices=["skip", "continue", "strict"],
        default=os.getenv("GIT_ERROR_RECOVERY", "continue"),
        help="Error recovery strategy: skip (skip errors), continue (continue on errors), strict (fail on first error) (default: $GIT_ERROR_RECOVERY)",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> GitIngestConfig:
    """Build GitIngestConfig from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        GitIngestConfig instance
    """
    # Set environment variables from CLI arguments so GitIngestConfig can read them
    os.environ["GIT_PROVIDER"] = args.provider.lower()

    if args.host:
        os.environ["GIT_HOST"] = args.host

    if args.branch:
        os.environ["GIT_BRANCH"] = args.branch

    if args.clone_dir:
        os.environ["GIT_CLONE_DIR"] = args.clone_dir

    if args.reset_repo or getattr(args, "reset", False):
        os.environ["GIT_RESET_REPO"] = "true"

    if args.file_types:
        os.environ["GIT_FILE_TYPES"] = args.file_types

    os.environ["GIT_GENERATE_SUMMARIES"] = "true" if args.generate_summaries else "false"
    os.environ["GIT_USE_LLM_SUMMARIES"] = "true" if args.use_llm_summaries else "false"
    os.environ["GIT_ENABLE_DLP"] = "true" if args.enable_dlp else "false"
    os.environ["GIT_EXCLUDE_TESTS"] = "true" if args.exclude_tests else "false"
    os.environ["GIT_NO_REFRESH"] = "true" if args.no_refresh else "false"
    os.environ["GIT_DRY_RUN"] = "true" if getattr(args, "dry_run", False) else "false"
    os.environ["GIT_DETECT_ENCODING"] = (
        "true" if getattr(args, "detect_encoding", True) else "false"
    )
    os.environ["GIT_ERROR_RECOVERY"] = getattr(args, "error_recovery", "continue")

    # Set provider-specific environment variables
    if args.provider == "bitbucket":
        # CLI args or fail - no environment variable fallback
        username = args.username
        api_username = getattr(args, "api_username", None)
        password = args.password

        if not username or not password:
            raise ValueError(
                "Bitbucket requires --username and --password\n"
                "  CLI: --username USER --password TOKEN --api-username EMAIL\n"
                "  Env: GIT_USERNAME=USER GIT_PASSWORD=TOKEN GIT_API_USERNAME=EMAIL\n"
            )

        # For Bitbucket Cloud, API calls require Atlassian account email
        if args.is_cloud and not api_username:
            raise ValueError(
                "Bitbucket Cloud API requires --api-username (your Atlassian account email)\n"
                "\n"
                "Bitbucket uses:\n"
                "  --username USER       for git clone/push operations (your Bitbucket username)\n"
                "  --api-username EMAIL  for API calls (your Atlassian account email)\n"
                "  --password TOKEN      your Bitbucket app password\n"
                "\n"
                "Set via: --username USER --api-username you@example.com --password TOKEN\n"
                "Or env:  GIT_USERNAME=USER GIT_API_USERNAME=you@example.com GIT_PASSWORD=TOKEN"
            )

        api_username = api_username or username  # Fall back for Server
        os.environ["BITBUCKET_USERNAME"] = username
        os.environ["BITBUCKET_API_USERNAME"] = api_username
        os.environ["BITBUCKET_PASSWORD"] = password
        os.environ["BITBUCKET_IS_CLOUD"] = "true" if args.is_cloud else "false"

    elif args.provider == "github":
        # CLI args or fail - no environment variable fallback
        token = args.token
        owner = args.owner

        if not token:
            raise ValueError(
                "GitHub requires --token (Personal Access Token)\n"
                "  CLI: --token ghp_xxxx\n"
                "  Env: GIT_TOKEN=ghp_xxxx"
            )
        if not owner:
            raise ValueError(
                "GitHub requires --owner (organisation or username)\n"
                "  CLI: --owner myorg\n"
                "  Env: GIT_OWNER=myorg"
            )

        os.environ["GITHUB_TOKEN"] = token
        os.environ["GITHUB_OWNER"] = owner
        if args.api_url:
            os.environ["GITHUB_API_URL"] = args.api_url

    # Now create config from environment variables
    config = GitIngestConfig()

    # Repository selection options (not in env vars)
    config.git_project = getattr(args, "project", None)
    config.git_repo = getattr(args, "repo", None)
    config.git_repo_pattern = getattr(args, "repo_pattern", None)
    config.git_repos_file = getattr(args, "repos_file", None)
    config.git_max_repos = getattr(args, "max_repos", 0)
    config.git_use_ssh = getattr(args, "use_ssh", False)
    config.git_verbose = getattr(args, "verbose", False)
    config.git_purge_logs = getattr(args, "purge_logs", False)

    # Phase 3: Operations features
    config.dry_run = getattr(args, "dry_run", False)
    config.detect_encoding = getattr(args, "detect_encoding", True)
    config.error_recovery_strategy = getattr(args, "error_recovery", "continue")

    # BM25 indexing configuration (priority: --skip-bm25 > --bm25-indexing > env var)
    if getattr(args, "skip_bm25", False):
        config.bm25_indexing_enabled = False
    elif getattr(args, "bm25_indexing", False):
        config.bm25_indexing_enabled = True

    # Validate configuration
    if not config.validate():
        raise ValueError(f"Invalid configuration for provider {args.provider}")

    return config


def main():
    """Main entry point for Git ingestion."""
    global get_logger, logger, audit

    try:
        # Parse arguments
        args = parse_arguments()

        # Set verbose logging if requested
        if getattr(args, "verbose", False):
            logging.getLogger().setLevel(logging.DEBUG)
            # Ensure all handlers have DEBUG level
            for handler in logger.handlers:
                handler.setLevel(logging.DEBUG)
            logger.info("Verbose logging enabled (DEBUG level)")
            print("[DEBUG MODE ENABLED]\n")

        print(f"\n{'='*60}")
        print(f"Git Ingestion: {args.provider.upper()}")
        print(f"{'='*60}\n")
        logger.info(f"Starting Git ingestion from {args.provider}")
        logger.debug(f"Command-line arguments: {sanitise_args_for_logging(args)}")

        # Build configuration
        config = build_config(args)

        # Purge logs if requested (disabled in Production) - MUST happen after config is built
        if getattr(args, "purge_logs", False):
            if config.environment == "Prod":
                print("\n[ERROR] Log purging is disabled in Production environment for safety.")
                print("        Current environment: Prod")
                print("        To purge logs, set ENVIRONMENT=Dev or ENVIRONMENT=Test\n")
                sys.exit(1)

            log_dir = Path("logs")
            if log_dir.exists():
                purged_count = 0
                # Only delete ingest-specific log files
                for log_file in log_dir.glob("ingest.log*"):
                    log_file.unlink()
                    purged_count += 1
                # Delete ingest audit log
                audit_log = log_dir / "ingest_audit.jsonl"
                if audit_log.exists():
                    audit_log.unlink()
                    purged_count += 1
                print(f"Purged {purged_count} ingest log files")

                # Recreate logger to avoid writing to deleted file descriptors
                # Close and remove all handlers
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)

                # Clear logger from cache so get_logger creates a fresh instance with file handler
                if "ingest" in logger_module._loggers:
                    del logger_module._loggers["ingest"]

                # Recreate logger with fresh handlers
                get_logger, audit = create_module_logger("ingest")
                logger = get_logger(log_to_console=True)
                logger.info("Log files purged, logger recreated")
                print("Logger recreated\n")

        # Handle database reset at the start (before ChromaDB init)
        if config.git_reset_repo:
            logger.info("Performing full database reset (--reset flag)")
            print("\n[RESET] Clearing all databases and caches for fresh ingestion...")
            success = clear_for_ingestion(verbose=True, dry_run=False)
            if not success:
                logger.error("Database reset failed")
                sys.exit(1)
            print()

            # Also clear the Git clone directory (only for current provider)
            clone_dir = Path(config.provider_clone_dir)
            if clone_dir.exists():
                logger.info(
                    f"Resetting clone directory for provider '{config.git_provider}': {config.provider_clone_dir}"
                )
                shutil.rmtree(clone_dir)
                print(
                    f"[RESET] Cleared Git clone directory for '{config.git_provider}': {config.provider_clone_dir}"
                )
            clone_dir.mkdir(parents=True, exist_ok=True)

            # Check for clones from other providers and preserve them
            base_clone_dir = Path(config.git_clone_dir)
            other_clones_found = []
            for other_provider in ("bitbucket", "github", "gitlab", "azure"):
                if other_provider == config.git_provider:
                    continue
                other_dir = base_clone_dir / other_provider
                if other_dir.exists():
                    other_clones_found.append((other_provider, str(other_dir)))

            if other_clones_found:
                print("[RESET] Preserving clones from other Git providers:")
                for other_provider, other_dir in other_clones_found:
                    logger.info(f"Preserving existing clones for '{other_provider}' at {other_dir}")
                    print(f"  • {other_provider}: {other_dir}")
                print()

        # Initialise ChromaDB using factory pattern (consistent with build_consistency_graph)
        print("Initialising ChromaDB...")
        PersistentClient_class, using_sqlite = get_vector_client(prefer="chroma")
        chroma_path = get_default_vector_path(Path(config.rag_data_path), using_sqlite)
        chroma_client = PersistentClient_class(path=chroma_path)

        logger.info(f"ChromaDB backend: {'SQLite' if using_sqlite else 'Chroma'}")
        logger.info(f"ChromaDB path: {chroma_path}")

        # Get or create collections
        chunk_collection = chroma_client.get_or_create_collection(
            name=config.chunk_collection_name, metadata={"hnsw:space": "cosine"}
        )
        doc_collection = chroma_client.get_or_create_collection(
            name=config.doc_collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Using chunk collection: {config.chunk_collection_name}")
        logger.info(f"Using doc collection: {config.doc_collection_name}")
        print(f"✓ ChromaDB initialised: {chroma_path}\n")

        # Initialise caches
        llm_cache_path = Path(config.rag_data_path) / "llm_cache"
        llm_cache_path.mkdir(parents=True, exist_ok=True)
        llm_cache = LLMCache(cache_path=str(llm_cache_path)) if config.llm_cache_enabled else None

        embedding_cache_path = Path(config.rag_data_path) / "embedding_cache"
        embedding_cache_path.mkdir(parents=True, exist_ok=True)
        embedding_cache = (
            EmbeddingCache(cache_path=str(embedding_cache_path))
            if config.embedding_cache_enabled
            else None
        )

        # Initialise code parser
        parser = CodeParser()
        print(f"Provider:  {args.provider}")
        print(f"Host:      {config.git_host}")
        if config.git_project:
            print(f"Project:   {config.git_project}")
        if config.git_repo:
            print(f"Repo:      {config.git_repo}")
        if config.git_repo_pattern:
            print(f"Pattern:   {config.git_repo_pattern}")
        if config.git_max_repos > 0:
            print(f"Max repos: {config.git_max_repos}")
        print(f"Clone directory base: {config.git_clone_dir}")
        print(f"Provider-scoped clones: {config.provider_clone_dir}")

        # Show Phase 3 features
        if getattr(config, "dry_run", False):
            print(f"Dry-run:   ENABLED [NO DATA WILL BE PERSISTED]")
        if getattr(config, "detect_encoding", True):
            print(f"Encoding:  AUTO-DETECT with fallback")
        error_recovery = getattr(config, "error_recovery_strategy", "continue")
        print(f"Recovery:  {error_recovery.upper()}")
        allowed_exts = sorted(
            {ext.strip().lower() for ext in config.file_types if ext and ext.strip()}
        )
        logger.info(f"Configured file type filters: {allowed_exts}")
        print(f"File types: {', '.join(allowed_exts)}")
        print()

        # Create connector
        connector = GitConnectorFactory.create(args.provider, config)

        # Connect to Git provider
        print("Connecting to Git provider...")
        with connector:
            logger.info("Connected to Git provider")

            # Load repository list from file if specified
            repos_from_file = set()
            repos_config = {}  # Maps repo_slug -> {"project": str, "branch": str}
            if config.git_repos_file and Path(config.git_repos_file).exists():
                repos_file_path = Path(config.git_repos_file)

                # Try loading as JSON first (regardless of first character)
                is_json = False
                try:
                    with open(repos_file_path) as f:
                        content = f.read().strip()
                        repos_json = json.loads(content)

                        # If it parsed as JSON, process it
                        if isinstance(repos_json, list):
                            for repo_entry in repos_json:
                                repo_name = repo_entry.get("repo")
                                if repo_name:
                                    repos_from_file.add(repo_name)
                                    repos_config[repo_name] = {
                                        "project": repo_entry.get("project"),
                                        "branch": repo_entry.get("branch", config.git_branch),
                                    }
                            is_json = True
                            logger.info(
                                f"Loaded {len(repos_from_file)} repositories from JSON file"
                            )
                            print(f"Loaded {len(repos_from_file)} repositories from JSON file\n")
                except (json.JSONDecodeError, ValueError, AttributeError):
                    # Not JSON, treat as text format (one repo per line)
                    pass

                # If JSON parsing failed, fall back to text format
                if not is_json:
                    with open(repos_file_path) as f:
                        repos_from_file = {line.strip() for line in f if line.strip()}
                    logger.info(f"Loaded {len(repos_from_file)} repositories from text file")
                    print(f"Loaded {len(repos_from_file)} repositories from text file\n")

            # If repos file provided, use it directly without listing projects
            if repos_from_file:
                # Extract unique projects from JSON config, or use command-line project if specified
                if repos_config:
                    # JSON format with per-repo project specs
                    unique_projects = set()
                    for repo_name, cfg in repos_config.items():
                        if cfg.get("project"):
                            unique_projects.add(cfg["project"])

                    if not unique_projects:
                        # No project specs in JSON, use command-line or fail
                        if not config.git_project:
                            raise ValueError(
                                "When using --repos-file without project specs, you must specify --project\n"
                                "  Either: add 'project' field to each repo in JSON\n"
                                "  Or:     use --project argument"
                            )
                        unique_projects = {config.git_project}

                    projects = [GitProject(key=p, name=p) for p in sorted(unique_projects)]
                    logger.info(
                        f"Using {len(projects)} projects from JSON file: {', '.join(unique_projects)}"
                    )
                    print(
                        f"Using {len(projects)} projects from JSON file: {', '.join(sorted(unique_projects))}\n"
                    )
                else:
                    # Text format - use command-line project or fail
                    if not config.git_project:
                        raise ValueError(
                            "When using --repos-file (text format), you must specify --project\n"
                            "  --repos-file lists repos, --project specifies which project they belong to"
                        )
                    projects = [GitProject(key=config.git_project, name=config.git_project)]
                    logger.info(f"Using project from command-line: {config.git_project}")
                    print(f"Using project: {config.git_project}\n")
            else:
                if config.git_provider == "github" and config.git_repo:
                    owner_key = config.github_owner
                    if not owner_key:
                        logger.error("GitHub requires --owner when specifying --repo")
                        print("✗ GitHub requires --owner when specifying --repo")
                        return
                    projects = [GitProject(key=owner_key, name=owner_key)]
                    logger.info(f"Using specified GitHub owner: {owner_key}")
                    print(f"✓ Using GitHub owner: {owner_key}\n")
                else:
                    # List projects/organisations from API
                    all_projects = connector.list_projects()
                    logger.info(f"Found {len(all_projects)} projects")
                    print(f"✓ Found {len(all_projects)} projects\n")

                    if not all_projects:
                        logger.warning("No projects found. Nothing to ingest.")
                        print("✗ No projects found. Nothing to ingest.")
                        return

                    # Filter to specific project if requested
                    if config.git_project:
                        projects = [
                            p
                            for p in all_projects
                            if p.key == config.git_project or p.name == config.git_project
                        ]
                        if not projects:
                            logger.error(f"Project '{config.git_project}' not found")
                            print(f"✗ Project '{config.git_project}' not found")
                            return
                        logger.info(f"Filtering to project: {projects[0].name}")
                        print(f"Filtered to 1 project: {projects[0].name}\n")
                    else:
                        projects = all_projects
                        print(f"Processing {len(projects)} projects\n")

            # For each project, list and process repositories
            total_repos_processed = 0
            total_repos_failed = 0
            for proj_idx, project in enumerate(projects, 1):
                print(f"[{proj_idx}/{len(projects)}] {project.name}")
                logger.info(f"Processing project: {project.name}")
                all_repos = connector.list_repositories(project.key)
                logger.info(f"Found {len(all_repos)} repositories in {project.name}")

                # Apply repository filters
                repos = all_repos

                # Filter by specific repo name
                if config.git_repo:
                    repos = [
                        r for r in repos if r.slug == config.git_repo or r.name == config.git_repo
                    ]
                    if repos:
                        logger.info(f"Filtering to repository: {repos[0].name}")
                        print(f"  → Filtered to 1 repository: {repos[0].name}")
                    else:
                        print(f"  → No repositories found matching '{config.git_repo}'")

                # Filter by pattern
                if config.git_repo_pattern:
                    pattern = config.git_repo_pattern.lower()
                    repos = [
                        r for r in repos if pattern in r.slug.lower() or pattern in r.name.lower()
                    ]
                    logger.info(
                        f"Filtered to {len(repos)} repositories matching pattern '{config.git_repo_pattern}'"
                    )
                    print(
                        f"  → Filtered to {len(repos)} matching pattern '{config.git_repo_pattern}'"
                    )

                # Filter by repos file
                if repos_from_file:
                    if repos_config:
                        # JSON format: filter repos that belong to this project
                        matching_repos = [
                            r
                            for r in repos
                            if r.slug in repos_from_file or r.name in repos_from_file
                        ]
                        # Further filter to only those in this project's config
                        project_repos = {
                            repo_name
                            for repo_name, cfg in repos_config.items()
                            if cfg.get("project") == project.key
                        }
                        repos = [
                            r
                            for r in matching_repos
                            if r.slug in project_repos or r.name in project_repos
                        ]
                        logger.info(
                            f"Filtered to {len(repos)} repositories from file for project {project.key}"
                        )
                        print(f"  → Filtered to {len(repos)} from file for this project")
                    else:
                        # Text format: filter by repo name/slug only
                        repos = [
                            r
                            for r in repos
                            if r.slug in repos_from_file or r.name in repos_from_file
                        ]
                        logger.info(f"Filtered to {len(repos)} repositories from file")
                        print(f"  → Filtered to {len(repos)} from file")

                # Apply max repos limit
                if config.git_max_repos > 0:
                    remaining = config.git_max_repos - total_repos_processed
                    if remaining <= 0:
                        logger.info(f"Reached maximum repository limit ({config.git_max_repos})")
                        print(f"  ✓ Reached max repos limit ({config.git_max_repos})")
                        break
                    repos = repos[:remaining]
                    logger.info(f"Limited to {len(repos)} repositories (remaining: {remaining})")

                if repos:
                    print(f"  ⚙ Processing {len(repos)} repositories...")
                else:
                    print(f"  ⊘ No repositories to process")
                    continue

                logger.info(f"Processing {len(repos)} repositories in {project.name}")

                for repo_idx, repo in enumerate(repos, 1):
                    if config.git_max_repos > 0 and total_repos_processed >= config.git_max_repos:
                        logger.info(f"Reached maximum repository limit ({config.git_max_repos})")
                        break

                    logger.info(f"Ingesting repository: {repo.name}")

                    try:
                        # Get repo-specific config if available (from JSON)
                        repo_cfg = repos_config.get(repo.slug, {})
                        repo_project = repo_cfg.get("project", project.key)
                        repo_branch = repo_cfg.get("branch", config.git_branch)

                        # Clone repository
                        print(f"    [{repo_idx}/{len(repos)}] {repo.name}...", end="", flush=True)
                        repo_path = connector.clone_repository(
                            project_key=repo_project,
                            repo_slug=repo.slug,
                            target_dir=config.provider_clone_dir,
                            branch=repo_branch,
                        )

                        # Get file walker
                        walker = connector.get_repository_walker(repo_path)

                        # Collect files and filter tests
                        files_to_process = []
                        files_skipped = 0

                        for file_path, content, mtime in walker.walk_files(
                            extensions=list(config.file_types)
                        ):
                            # Skip tests if requested
                            if config.exclude_tests and is_test_file(file_path):
                                files_skipped += 1
                                continue

                            files_to_process.append((file_path, content, mtime))

                        if files_to_process:
                            extension_summary = Counter(
                                Path(f).suffix.lower() or "<no_ext>" for f, _, _ in files_to_process
                            )
                            summary_parts = [
                                f"{ext}:{count}" for ext, count in sorted(extension_summary.items())
                            ]
                            logger.info(
                                "Files selected for ingestion (by extension): %s",
                                ", ".join(summary_parts),
                            )

                        # Process files concurrently with resource monitoring
                        if files_to_process:
                            # Initialise error recovery tracker
                            error_recovery = ErrorRecoveryStrategy()

                            processor = ConcurrentFileProcessor(
                                max_workers=4,  # Configurable, default 4
                                chunk_collection=chunk_collection,
                                doc_collection=doc_collection,
                                parser=parser,
                                llm_cache=llm_cache,
                                embedding_cache=embedding_cache,
                                config=config,
                                dry_run=getattr(args, "dry_run", False),
                                error_recovery=error_recovery,
                            )

                            stats = processor.process_files(
                                files=files_to_process,
                                project_key=project.key,
                                repo_slug=repo.slug,
                                branch=repo_branch,
                            )

                            files_ingested = stats.files_processed
                            files_failed = stats.files_failed

                            # Log processing stats
                            stats_summary = stats.summary()
                            dry_run_status = " [DRY-RUN]" if getattr(args, "dry_run", False) else ""
                            logger.info(
                                f"Repository stats{dry_run_status} for {repo.name}: "
                                f"processed={files_ingested} failed={files_failed} skipped={files_skipped} "
                                f"time={stats_summary['total_time_seconds']:.2f}s "
                                f"rate={stats_summary['files_per_second']:.2f}f/s "
                                f"peak_memory={stats_summary['peak_memory_mb']:.1f}MB "
                                f"peak_cpu={stats_summary['peak_cpu_percent']:.1f}%"
                            )

                            # Log error recovery summary
                            recovery_summary = error_recovery.summary()
                            if recovery_summary["total_errors"] > 0:
                                logger.warning(
                                    f"Error recovery summary: "
                                    f"recovered={recovery_summary['recovered_errors']} "
                                    f"failed={recovery_summary['failed_errors']} "
                                    f"warnings={recovery_summary['total_warnings']}"
                                )
                        else:
                            files_ingested = 0
                            files_failed = 0

                        logger.info(
                            f"Ingested {files_ingested} files from {repo.name} ({files_failed} failed, {files_skipped} skipped)"
                        )
                        print(
                            f" ✓ ({files_ingested} ingested, {files_failed} failed, {files_skipped} skipped)"
                        )
                        total_repos_processed += 1

                    except Exception as e:
                        logger.error(f"Error ingesting {repo.name}: {e}", exc_info=True)
                        print(f" ✗ Error: {str(e)[:50]}")
                        total_repos_failed += 1
                        continue

                # Check if we hit the limit
                if config.git_max_repos > 0 and total_repos_processed >= config.git_max_repos:
                    break

        print(f"\n{'='*60}")
        if total_repos_failed > 0:
            print(f"⚠ Ingestion completed with {total_repos_failed} error(s)")
        else:
            print(f"✓ Ingestion completed successfully")
        print(f"{'='*60}")
        print(f"Repositories processed: {total_repos_processed}")
        if total_repos_failed > 0:
            print(f"Repositories failed:   {total_repos_failed}")

        # Update BM25 corpus statistics (IDF values) after all documents indexed
        if config.bm25_indexing_enabled and not getattr(args, "dry_run", False):
            try:
                logger.info("Computing BM25 corpus statistics (IDF values)...")
                cache_db = get_cache_client(enable_cache=True)
                total_docs = cache_db.get_bm25_corpus_size()

                if total_docs > 0:
                    cache_db.update_bm25_corpus_stats(total_docs)
                    avg_doc_len = cache_db.get_bm25_avg_doc_length()
                    logger.info(
                        f"BM25 corpus stats updated: {total_docs} documents, avg length {avg_doc_len:.1f} tokens"
                    )
                    audit(
                        "bm25_corpus_stats_updated",
                        {"total_documents": total_docs, "avg_doc_length": avg_doc_len},
                    )
            except Exception as e:
                logger.warning(f"Failed to update BM25 corpus statistics: {e}")
                audit(
                    "bm25_corpus_stats_failed",
                    {"error": str(e)[:200], "error_type": type(e).__name__},
                )

        # Summary report
        logger.info(
            f"Git ingestion summary: "
            f"repos={total_repos_processed} "
            f"repo_failures={total_repos_failed}"
        )

        if total_repos_failed > 0:
            logger.warning(
                f"Git ingestion completed - processed {total_repos_processed} repositories, {total_repos_failed} failed"
            )
            sys.exit(1)
        else:
            logger.info(
                f"Git ingestion completed successfully - processed {total_repos_processed} repositories"
            )
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Ingestion cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Git ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
