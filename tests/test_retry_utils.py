"""Tests for retry utility functions and decorators.

Tests cover:
- Exception classification (transient vs hard failures)
- Retry decorator behaviour with exponential backoff
- Failure tracking and audit events
- Ollama and ChromaDB-specific retry wrappers
"""

import time
from unittest.mock import Mock, call, patch

import pytest
import requests
from pydantic import ValidationError

# Import retry_utils from scripts.utils (conftest.py adds project root to sys.path)
from scripts.utils.retry_utils import (
    classify_failure,
    is_transient_error,
    retry_chromadb_call,
    retry_ollama_call,
    retry_with_backoff,
)

# =========================
# Exception Classification Tests
# =========================


class TestExceptionClassification:
    """Test transient vs hard error classification."""

    def test_network_errors_are_transient(self):
        """Network errors should be classified as transient."""
        assert is_transient_error(ConnectionError("Connection refused"))
        assert is_transient_error(TimeoutError("Operation timed out"))
        assert is_transient_error(requests.exceptions.ConnectionError())
        assert is_transient_error(requests.exceptions.Timeout())

    def test_http_rate_limit_is_transient(self):
        """HTTP 429 rate limit errors should be transient."""
        response = Mock()
        response.status_code = 429
        error = requests.exceptions.HTTPError()
        error.response = response
        assert is_transient_error(error)

    def test_http_server_errors_are_transient(self):
        """HTTP 502/503/504 errors should be transient."""
        for status in [502, 503, 504]:
            response = Mock()
            response.status_code = status
            error = requests.exceptions.HTTPError()
            error.response = response
            assert is_transient_error(error)

    def test_http_client_errors_are_hard(self):
        """HTTP 400/401/403/404 errors should be hard failures."""
        for status in [400, 401, 403, 404]:
            response = Mock()
            response.status_code = status
            error = requests.exceptions.HTTPError()
            error.response = response
            assert not is_transient_error(error)

    def test_validation_errors_are_hard(self):
        """Validation errors should be hard failures."""
        assert not is_transient_error(ValidationError.from_exception_data("test", []))
        assert not is_transient_error(ValueError("invalid value"))
        assert not is_transient_error(TypeError("wrong type"))
        assert not is_transient_error(KeyError("missing key"))

    def test_timeout_keyword_detection(self):
        """Errors with 'timeout' in message should be transient."""
        assert is_transient_error(Exception("Request timed out"))
        assert is_transient_error(Exception("Connection timeout"))
        assert is_transient_error(Exception("deadline exceeded"))

    def test_ollama_loading_errors_are_transient(self):
        """Ollama model loading errors should be transient."""
        assert is_transient_error(Exception("ollama: loading model"))
        assert is_transient_error(Exception("ollama connection refused"))
        assert is_transient_error(Exception("CUDA out of memory"))

    def test_chromadb_connection_errors_are_transient(self):
        """ChromaDB connection errors should be transient."""
        assert is_transient_error(Exception("chromadb connection failed"))
        assert is_transient_error(Exception("chroma server unavailable"))

    def test_chromadb_notfound_is_hard(self):
        """ChromaDB NotFoundError should be hard failure."""
        assert not is_transient_error(Exception("chromadb.errors.NotFoundError"))

    def test_unknown_errors_default_to_hard(self):
        """Unknown errors should default to hard (fail fast)."""
        assert not is_transient_error(Exception("mysterious error"))


class TestFailureClassification:
    """Test classify_failure function behaviour."""

    def test_classify_transient_with_retries_remaining(self):
        """Transient error with retries left should retry."""
        error = ConnectionError("connection failed")
        failure_type, should_retry = classify_failure(error, "test_op", 1, 3)
        assert failure_type == "transient"
        assert should_retry is True

    def test_classify_transient_at_max_retries(self):
        """Transient error at max retries should not retry."""
        error = TimeoutError("timeout")
        failure_type, should_retry = classify_failure(error, "test_op", 3, 3)
        assert failure_type == "transient"
        assert should_retry is False

    def test_classify_hard_failure(self):
        """Hard error should not retry regardless of attempt count."""
        error = ValueError("invalid input")
        failure_type, should_retry = classify_failure(error, "test_op", 1, 5)
        assert failure_type == "hard"
        assert should_retry is False

    def test_classify_transient_override(self):
        """Explicit transient_types should force transient classification."""
        error = ValueError("invalid input")
        failure_type, should_retry = classify_failure(
            error, "test_op", 1, 3, transient_types=(ValueError,), hard_types=None
        )
        assert failure_type == "transient"
        assert should_retry is True

    def test_classify_hard_override(self):
        """Explicit hard_types should force hard classification."""
        error = ConnectionError("connection failed")
        failure_type, should_retry = classify_failure(
            error, "test_op", 1, 3, transient_types=None, hard_types=(ConnectionError,)
        )
        assert failure_type == "hard"
        assert should_retry is False


# =========================
# Retry Decorator Tests
# =========================


class TestRetryDecorator:
    """Test retry_with_backoff decorator behaviour."""

    def test_successful_call_no_retry(self):
        """Successful call should not trigger retries."""
        mock_func = Mock(return_value="success")

        @retry_with_backoff(max_retries=3, initial_delay=0.1)
        def test_func():
            return mock_func()

        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 1

    def test_transient_failure_retries(self):
        """Transient failure should trigger retries."""
        mock_func = Mock(
            side_effect=[ConnectionError("failed"), ConnectionError("failed"), "success"]
        )

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def test_func():
            return mock_func()

        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 3

    def test_hard_failure_no_retry(self):
        """Hard failure should not retry."""
        mock_func = Mock(side_effect=ValueError("invalid"))

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def test_func():
            return mock_func()

        with pytest.raises(ValueError):
            test_func()

        assert mock_func.call_count == 1  # Only called once, no retries

    def test_retry_exhaustion(self):
        """Exhausting all retries should raise last exception."""
        mock_func = Mock(side_effect=ConnectionError("always fails"))

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def test_func():
            return mock_func()

        with pytest.raises(ConnectionError):
            test_func()

        assert mock_func.call_count == 3  # Max retries reached

    def test_exponential_backoff_timing(self):
        """Verify exponential backoff delay increases."""
        call_times = []

        def track_time():
            call_times.append(time.perf_counter())
            raise ConnectionError("fail")

        @retry_with_backoff(
            max_retries=3,
            initial_delay=0.05,
            backoff_factor=2.0,
            jitter=False,  # Disable jitter for predictable timing
        )
        def test_func():
            return track_time()

        with pytest.raises(ConnectionError):
            test_func()

        # Check delays between calls (allow some tolerance for timing)
        assert len(call_times) == 3
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        # First delay ~0.05s, second delay ~0.1s (2x backoff)
        assert 0.04 < delay1 < 0.07
        assert 0.08 < delay2 < 0.14

    def test_max_delay_cap(self):
        """Verify delay is capped at max_delay."""
        call_times = []

        def track_time():
            call_times.append(time.perf_counter())
            raise ConnectionError("fail")

        @retry_with_backoff(
            max_retries=5,
            initial_delay=1.0,
            backoff_factor=10.0,  # Would exceed max without cap
            max_delay=0.2,  # Cap at 0.2s
            jitter=False,
        )
        def test_func():
            return track_time()

        with pytest.raises(ConnectionError):
            test_func()

        # All delays should be capped at max_delay
        for i in range(1, len(call_times)):
            delay = call_times[i] - call_times[i - 1]
            assert delay <= 0.25  # Allow small tolerance

    def test_transient_override_retries_value_error(self):
        """Override should retry ValueError as transient."""
        mock_func = Mock(side_effect=[ValueError("invalid"), ValueError("invalid again"), "ok"])

        @retry_with_backoff(max_retries=3, initial_delay=0.01, transient_types=(ValueError,))
        def test_func():
            return mock_func()

        result = test_func()
        assert result == "ok"
        assert mock_func.call_count == 3

    def test_hard_override_no_retry_connection_error(self):
        """Override should not retry ConnectionError (hard)."""
        mock_func = Mock(side_effect=ConnectionError("conn fail"))

        @retry_with_backoff(max_retries=3, initial_delay=0.01, hard_types=(ConnectionError,))
        def test_func():
            return mock_func()

        with pytest.raises(ConnectionError):
            test_func()
        assert mock_func.call_count == 1

    @patch("scripts.utils.retry_utils.audit")
    def test_retry_audit_events(self, mock_audit):
        """Verify retry events are audited."""
        mock_func = Mock(side_effect=[ConnectionError("fail"), "success"])

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def test_func():
            return mock_func()

        result = test_func()

        # Should have retry_attempt_failed and retry_success events
        audit_calls = [call[0][0] for call in mock_audit.call_args_list]
        assert "retry_attempt_failed" in audit_calls
        assert "retry_success" in audit_calls

    @patch("scripts.utils.retry_utils.audit")
    def test_hard_failure_audit(self, mock_audit):
        """Verify hard failures are audited."""

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def test_func():
            raise ValueError("hard failure")

        with pytest.raises(ValueError):
            test_func()

        # Should have hard_failure audit event
        audit_calls = [call[0][0] for call in mock_audit.call_args_list]
        assert "hard_failure" in audit_calls


# =========================
# Specific Retry Wrapper Tests
# =========================


class TestOllamaRetryWrapper:
    """Test Ollama-specific retry decorator."""

    def test_ollama_retry_defaults(self):
        """Verify Ollama retry uses appropriate defaults."""
        mock_func = Mock(side_effect=[Exception("ollama connection timeout"), "success"])

        @retry_ollama_call(operation_name="test_ollama")
        def test_func():
            return mock_func()

        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 2


class TestChromaDBRetryWrapper:
    """Test ChromaDB-specific retry decorator."""

    def test_chromadb_retry_defaults(self):
        """Verify ChromaDB retry uses appropriate defaults."""
        mock_func = Mock(
            side_effect=[
                Exception("chroma connection refused"),
                Exception("chroma timeout"),
                "success",
            ]
        )

        @retry_chromadb_call(operation_name="test_chroma")
        def test_func():
            return mock_func()

        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 3


# =========================
# Integration Tests
# =========================


class TestRetryIntegration:
    """Integration tests with realistic scenarios."""

    def test_intermittent_network_issues(self):
        """Simulate intermittent network failures."""
        attempts = [0]

        @retry_with_backoff(max_retries=5, initial_delay=0.01)
        def flaky_network_call():
            attempts[0] += 1
            if attempts[0] < 4:
                raise ConnectionError("network unstable")
            return "data"

        result = flaky_network_call()
        assert result == "data"
        assert attempts[0] == 4

    def test_rate_limiting_scenario(self):
        """Simulate rate limit then success."""
        attempts = [0]

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def rate_limited_api():
            attempts[0] += 1
            if attempts[0] == 1:
                response = Mock()
                response.status_code = 429
                error = requests.exceptions.HTTPError()
                error.response = response
                raise error
            return "success"

        result = rate_limited_api()
        assert result == "success"
        assert attempts[0] == 2

    def test_permanent_failure_fast_fail(self):
        """Permanent errors should fail fast without retries."""
        start = time.perf_counter()

        @retry_with_backoff(max_retries=5, initial_delay=1.0)
        def permanent_failure():
            raise ValueError("invalid input")

        with pytest.raises(ValueError):
            permanent_failure()

        elapsed = time.perf_counter() - start
        # Should fail fast (< 0.1s), not wait for retries
        assert elapsed < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
