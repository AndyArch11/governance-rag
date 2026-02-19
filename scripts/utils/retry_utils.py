"""Retry logic with exponential backoff for transient failures.

Provides retry decorators and utilities for handling transient failures
in Ollama LLM calls and ChromaDB operations. Distinguishes between
transient errors (network issues, timeouts, rate limits) that should
be retried and hard errors (validation failures, not found errors)
that should fail immediately.

Features:
- Exponential backoff with jitter
- Configurable max retries and timeout
- Exception classification (transient vs hard)
- Comprehensive audit logging
- Thread-safe operation

Usage:
    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def call_ollama_api():
        return ollama.invoke(prompt)

    @retry_with_backoff(max_retries=5, backoff_factor=2.0)
    def query_chromadb(collection):
        return collection.get(where=filter)
"""

import functools
import random
import time
from typing import Any, Callable, Optional, Tuple, Type

import requests

from .logger import get_logger

# Import for type checking Pydantic validation errors
try:
    from pydantic import ValidationError as PydanticValidationError
except ImportError:
    # Fallback if Pydantic not available
    PydanticValidationError = None

logger = get_logger("retry")


def audit(event_type: str, data: dict) -> None:
    """Wrapper for audit logging that uses the retry module name."""
    from .logger import audit as base_audit

    base_audit("retry", event_type, data)


# =========================
# Exception Classification
# =========================


def is_transient_error(exception: Exception) -> bool:
    """Determine if an exception represents a transient failure.

    Transient failures are temporary conditions that may succeed
    on retry, such as network timeouts, connection errors, or
    rate limiting. Hard failures are permanent errors like
    validation failures or missing resources.

    Args:
        exception: The exception to classify.

    Returns:
        True if the error is transient and retry-worthy, False otherwise.

    Transient Error Types:
        - Network errors (ConnectionError, Timeout)
        - HTTP 429 (rate limit), 503 (service unavailable), 502/504 (gateway errors)
        - ChromaDB connection/timeout errors
        - Ollama service errors (model loading, CUDA errors)

    Hard Error Types:
        - Validation errors (Pydantic, schema mismatches)
        - 404 Not Found errors
        - 400 Bad Request (malformed queries)
        - Authentication/permission errors (401, 403)
        - Programming errors (KeyError, AttributeError)
    """
    exception_type = type(exception).__name__
    exception_msg = str(exception).lower()

    # Network-level transient errors
    if isinstance(
        exception,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            TimeoutError,
            ConnectionError,
            ConnectionRefusedError,
            ConnectionResetError,
        ),
    ):
        return True

    # HTTP status code transient errors
    if isinstance(exception, requests.exceptions.HTTPError):
        if hasattr(exception, "response") and exception.response is not None:
            status = exception.response.status_code
            # 429 = rate limit, 502/503/504 = gateway/service errors
            if status in (429, 502, 503, 504):
                return True

    # Generic database transient errors (readonly, locked, I/O)
    if any(
        keyword in exception_msg
        for keyword in [
            "attempt to write",
            "readonly",
            "database is locked",
            "disk i/o error",
        ]
    ):
        return True

    # ChromaDB-specific transient errors
    # Connection issues, server unavailable, database locks
    if "chromadb" in exception_type.lower() or "chroma" in exception_msg:
        # Connection errors, locks, and I/O issues are transient
        if any(
            keyword in exception_msg
            for keyword in [
                "connection",
                "timeout",
                "unavailable",
                "refused",
            ]
        ):
            return True
        # Not found errors are hard failures
        if "not found" in exception_msg or "notfound" in exception_type.lower():
            return False

    # Ollama/LangChain-specific transient errors
    if "ollama" in exception_type.lower() or "ollama" in exception_msg:
        # Model loading, CUDA errors can be transient
        if any(
            keyword in exception_msg
            for keyword in [
                "loading model",
                "cuda",
                "out of memory",
                "connection",
                "timeout",
                "unavailable",
            ]
        ):
            return True

    # CUDA/GPU errors (can occur without "ollama" in message)
    if any(keyword in exception_msg for keyword in ["cuda", "out of memory", "gpu"]):
        return True

    # Generic timeout indicators
    if any(keyword in exception_msg for keyword in ["timeout", "timed out", "deadline exceeded"]):
        return True

    # Rate limiting indicators
    if any(keyword in exception_msg for keyword in ["rate limit", "too many requests", "throttle"]):
        return True

    # Hard failures (validation, programming errors)
    hard_failure_types = [
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        NotImplementedError,
    ]

    # Add Pydantic ValidationError if available
    if PydanticValidationError is not None:
        hard_failure_types.append(PydanticValidationError)

    if isinstance(exception, tuple(hard_failure_types)):
        return False

    # HTTP hard failures
    if isinstance(exception, requests.exceptions.HTTPError):
        if hasattr(exception, "response") and exception.response is not None:
            status = exception.response.status_code
            # 400 = bad request, 401/403 = auth, 404 = not found
            if status in (400, 401, 403, 404):
                return False

    # Default: treat unknown errors as non-transient for safety
    # Better to fail fast on unexpected errors than retry indefinitely
    return False


def classify_failure(
    exception: Exception,
    operation: str,
    attempt: int,
    max_retries: int,
    transient_types: Optional[Tuple[Type[Exception], ...]] = None,
    hard_types: Optional[Tuple[Type[Exception], ...]] = None,
) -> Tuple[str, bool]:
    """Classify a failure and determine if retry should continue.

    Args:
        exception: The exception that occurred.
        operation: Name of the operation that failed.
        attempt: Current attempt number (1-indexed).
        max_retries: Maximum retry attempts allowed.

    Returns:
        Tuple of (failure_type, should_retry):
            - failure_type: 'transient' or 'hard'
            - should_retry: True if retry should be attempted
    """
    # Allow callers to override classification via explicit exception types
    if transient_types and isinstance(exception, transient_types):
        is_transient = True
    elif hard_types and isinstance(exception, hard_types):
        is_transient = False
    else:
        is_transient = is_transient_error(exception)
    failure_type = "transient" if is_transient else "hard"

    # Retry transient errors if attempts remain
    should_retry = is_transient and attempt < max_retries

    # Log classification
    logger.debug(
        f"Failure classified: {failure_type} | "
        f"Operation: {operation} | "
        f"Attempt: {attempt}/{max_retries} | "
        f"Will retry: {should_retry}"
    )

    return failure_type, should_retry


# =========================
# Retry Decorator
# =========================


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    operation_name: Optional[str] = None,
    transient_types: Optional[Tuple[Type[Exception], ...]] = None,
    hard_types: Optional[Tuple[Type[Exception], ...]] = None,
):
    """Decorator to retry a function with exponential backoff.

    Automatically retries transient failures with exponentially
    increasing delays between attempts. Hard failures fail immediately
    without retries.

    Args:
        max_retries: Maximum number of retry attempts (default 3).
        initial_delay: Initial delay in seconds (default 1.0).
        backoff_factor: Multiplier for delay on each retry (default 2.0).
        max_delay: Maximum delay between retries in seconds (default 60.0).
        jitter: Add random jitter to delays to prevent thundering herd (default True).
        operation_name: Name for logging (defaults to function name).

    Returns:
        Decorated function with retry logic.

    Classification Overrides:
        - transient_types: Treat exceptions matching these types as transient (always retried if attempts remain).
        - hard_types: Treat exceptions matching these types as hard failures (never retried).

    Example:
        @retry_with_backoff(max_retries=5, initial_delay=0.5)
        def fetch_embeddings(text):
            return ollama_embed.embed_query(text)

    Delay Calculation:
        delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
        if jitter:
            delay *= random.uniform(0.5, 1.5)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            attempt = 0
            last_exception = None

            while attempt < max_retries:
                attempt += 1

                try:
                    result = func(*args, **kwargs)

                    # Log successful retry if not first attempt
                    if attempt > 1:
                        logger.info(f"Retry succeeded: {op_name} on attempt {attempt}")
                        audit(
                            "retry_success",
                            {
                                "operation": op_name,
                                "attempt": attempt,
                                "total_attempts": max_retries,
                            },
                        )

                    return result

                except Exception as e:
                    last_exception = e
                    failure_type, should_retry = classify_failure(
                        e,
                        op_name,
                        attempt,
                        max_retries,
                        transient_types=transient_types,
                        hard_types=hard_types,
                    )

                    # Hard failure - don't retry
                    if not should_retry:
                        logger.error(
                            f"Hard failure (no retry): {op_name} | "
                            f"Error: {type(e).__name__}: {str(e)[:200]}"
                        )
                        audit(
                            "hard_failure",
                            {
                                "operation": op_name,
                                "attempt": attempt,
                                "error_type": type(e).__name__,
                                "error_message": str(e)[:200],
                            },
                        )
                        raise

                    # Log failure details
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} failed: {op_name} | "
                        f"Error: {type(e).__name__}: {str(e)[:100]} | "
                        f"Type: {failure_type}"
                    )

                    audit(
                        "retry_attempt_failed",
                        {
                            "operation": op_name,
                            "attempt": attempt,
                            "max_retries": max_retries,
                            "error_type": type(e).__name__,
                            "error_message": str(e)[:200],
                            "failure_type": failure_type,
                            "will_retry": should_retry,
                        },
                    )

                    # Calculate delay with exponential backoff
                    delay = min(initial_delay * (backoff_factor ** (attempt - 1)), max_delay)

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay *= random.uniform(0.5, 1.5)

                    logger.info(
                        f"Retrying {op_name} in {delay:.2f}s " f"(attempt {attempt}/{max_retries})"
                    )

                    time.sleep(delay)

            # All retries exhausted
            logger.error(
                f"All retries exhausted: {op_name} | "
                f"Attempts: {max_retries} | "
                f"Last error: {type(last_exception).__name__}: {str(last_exception)[:200]}"
            )
            audit(
                "retry_exhausted",
                {
                    "operation": op_name,
                    "max_retries": max_retries,
                    "final_error_type": type(last_exception).__name__,
                    "final_error_message": str(last_exception)[:200],
                },
            )

            raise last_exception

        return wrapper

    return decorator


# =========================
# Convenience Wrappers
# =========================


def retry_ollama_call(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    operation_name: Optional[str] = None,
    transient_types: Optional[Tuple[Type[Exception], ...]] = None,
    hard_types: Optional[Tuple[Type[Exception], ...]] = None,
):
    """Retry decorator optimised for Ollama LLM calls.

    Uses conservative defaults suitable for LLM operations:
    - Moderate retries (3 attempts)
    - Initial delay to allow model loading
    - Capped backoff to prevent long waits

    Args:
        max_retries: Maximum retry attempts (default 3).
        initial_delay: Initial delay in seconds (default 1.0).
        operation_name: Name for logging.

    Returns:
        Retry decorator configured for Ollama.
    """
    return retry_with_backoff(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=2.0,
        max_delay=30.0,  # Cap at 30s for LLM calls
        jitter=True,
        operation_name=operation_name or "ollama_call",
        transient_types=transient_types,
        hard_types=hard_types,
    )


def retry_chromadb_call(
    max_retries: int = 5,
    initial_delay: float = 0.5,
    operation_name: Optional[str] = None,
    transient_types: Optional[Tuple[Type[Exception], ...]] = None,
    hard_types: Optional[Tuple[Type[Exception], ...]] = None,
):
    """Retry decorator optimised for ChromaDB operations.

    Uses aggressive retries suitable for database operations:
    - More retries (5 attempts) for transient DB issues
    - Shorter initial delay for quick recovery
    - Moderate backoff for connection issues

    Args:
        max_retries: Maximum retry attempts (default 5).
        initial_delay: Initial delay in seconds (default 0.5).
        operation_name: Name for logging.

    Returns:
        Retry decorator configured for ChromaDB.
    """
    return retry_with_backoff(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=2.0,
        max_delay=20.0,  # Cap at 20s for DB operations
        jitter=True,
        operation_name=operation_name or "chromadb_call",
        transient_types=transient_types,
        hard_types=hard_types,
    )
