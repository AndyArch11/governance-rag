"""LLM call instrumentation for token tracking and monitoring.

Instruments Ollama and external LLM calls to track:
- Token consumption (input/output)
- Generation latency
- Error rates
- Cost per call (if applicable)

Usage:
    from scripts.utils.llm_instrumentation import instrument_ollama_call, instrument_claude_call

    # Track Ollama calls
    with instrument_ollama_call("model_name", {"question": "What is MFA?"}):
        response = ollama_client.generate(...)

    # Or manually record
    from scripts.utils.monitoring import get_token_counter
    counter = get_token_counter()
    counter.record_tokens("model_name", input_tokens=50, output_tokens=120)
"""

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, Generator, Optional

from scripts.utils.monitoring import (
    get_perf_metrics,
    get_token_counter,
    get_tracer,
    trace_operation,
)

logger = logging.getLogger(__name__)


@contextmanager
def instrument_ollama_call(
    model: str,
    context: Optional[Dict[str, Any]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Context manager for instrumenting Ollama LLM calls.

    Tracks token usage, latency, and errors for Ollama generation calls.

    Usage:
        with instrument_ollama_call("model_name", {"query": "How does MFA work?"}) as metrics:
            response = ollama_client.generate(...)
            metrics["input_tokens"] = response.get("prompt_eval_count", 0)
            metrics["output_tokens"] = response.get("eval_count", 0)

    Args:
        model: Model name (e.g., "model_name", "neural-chat")
        context: Optional context dict with operation details

    Yields:
        Metrics dict to populate with token counts
    """
    tracer = get_tracer(__name__)
    token_counter = get_token_counter()
    perf_metrics = get_perf_metrics()

    attributes = {"model": model}
    if context:
        for key, value in context.items():
            try:
                attributes[f"context.{key}"] = str(value)[:100]  # Limit string length
            except Exception:
                pass

    metrics = {
        "input_tokens": 0,
        "output_tokens": 0,
        "error": False,
        "error_message": None,
    }

    start_time = time.time()

    with tracer.start_as_current_span(f"llm.generate.{model}") as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)

        try:
            yield metrics

            # Record successful call
            input_tokens = metrics.get("input_tokens", 0)
            output_tokens = metrics.get("output_tokens", 0)
            token_counter.record_tokens(model, input_tokens, output_tokens, success=True)

            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            perf_metrics.record_generation(latency_ms, model)

            span.set_attribute("tokens.input", input_tokens)
            span.set_attribute("tokens.output", output_tokens)
            span.set_attribute("latency_ms", latency_ms)

            logger.debug(
                f"✓ Ollama call completed: {model}, "
                f"input={input_tokens}, output={output_tokens}, "
                f"latency={latency_ms:.1f}ms"
            )

        except Exception as e:
            # Record failed call
            token_counter.record_tokens(
                model,
                metrics.get("input_tokens", 0),
                metrics.get("output_tokens", 0),
                success=False,
            )

            span.set_attribute("error", True)
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e)[:200])

            logger.error(f"✗ Ollama call failed: {model}, error: {e}")
            raise


@contextmanager
def instrument_claude_call(
    model: str,
    context: Optional[Dict[str, Any]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Context manager for instrumenting Claude API calls.

    Tracks token usage from Claude responses via usage field.

    Usage:
        with instrument_claude_call("claude-3-sonnet") as metrics:
            response = client.messages.create(...)
            if response.usage:
                metrics["input_tokens"] = response.usage.input_tokens
                metrics["output_tokens"] = response.usage.output_tokens

    Args:
        model: Model name (e.g., "claude-3-sonnet", "claude-3-opus")
        context: Optional context dict with operation details

    Yields:
        Metrics dict to populate with token counts
    """
    tracer = get_tracer(__name__)
    token_counter = get_token_counter()
    perf_metrics = get_perf_metrics()

    attributes = {"model": model, "provider": "anthropic"}
    if context:
        for key, value in context.items():
            try:
                attributes[f"context.{key}"] = str(value)[:100]
            except Exception:
                pass

    metrics = {
        "input_tokens": 0,
        "output_tokens": 0,
        "error": False,
        "error_message": None,
    }

    start_time = time.time()

    with tracer.start_as_current_span(f"llm.generate.claude") as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)

        try:
            yield metrics

            # Record successful call
            input_tokens = metrics.get("input_tokens", 0)
            output_tokens = metrics.get("output_tokens", 0)
            token_counter.record_tokens(model, input_tokens, output_tokens, success=True)

            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            perf_metrics.record_generation(latency_ms, model)

            span.set_attribute("tokens.input", input_tokens)
            span.set_attribute("tokens.output", output_tokens)
            span.set_attribute("latency_ms", latency_ms)

            logger.debug(
                f"✓ Claude call completed: {model}, "
                f"input={input_tokens}, output={output_tokens}, "
                f"latency={latency_ms:.1f}ms"
            )

        except Exception as e:
            # Record failed call
            token_counter.record_tokens(
                model,
                metrics.get("input_tokens", 0),
                metrics.get("output_tokens", 0),
                success=False,
            )

            span.set_attribute("error", True)
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e)[:200])

            logger.error(f"✗ Claude call failed: {model}, error: {e}")
            raise


def instrument_retrieval(operation_name: str = "retrieval"):
    """Decorator to instrument retrieval operations.

    Usage:
        @instrument_retrieval("semantic_search")
        def retrieve_chunks(query, collection, k=5):
            # retrieval code
            return chunks, metadata

    Args:
        operation_name: Name of retrieval operation

    Returns:
        Decorated function with instrumentation
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            perf_metrics = get_perf_metrics()
            start_time = time.time()

            with trace_operation(operation_name, {"function": func.__name__}):
                try:
                    result = func(*args, **kwargs)

                    # Calculate metrics
                    latency_ms = (time.time() - start_time) * 1000
                    result_count = 0

                    # Try to extract result count
                    if isinstance(result, tuple):
                        # Many retrieval functions return (chunks, metadata)
                        result_count = len(result[0]) if result else 0
                    elif isinstance(result, list):
                        result_count = len(result)
                    elif isinstance(result, dict):
                        result_count = len(result.get("results", []))

                    perf_metrics.record_retrieval(latency_ms, result_count)
                    logger.debug(
                        f"✓ {operation_name}: {result_count} results in {latency_ms:.1f}ms"
                    )

                    return result

                except Exception as e:
                    latency_ms = (time.time() - start_time) * 1000
                    logger.error(f"✗ {operation_name} failed after {latency_ms:.1f}ms: {e}")
                    raise

        return wrapper

    return decorator


class TokenBudget:
    """Manages token usage budgets for cost control.

    Example:
        budget = TokenBudget(
            total_tokens=100_000,
            max_per_call=2000,
            warn_at_percent=80
        )

        if budget.can_use(estimated_tokens=150):
            # proceed with LLM call
            budget.use(actual_tokens)
        else:
            # token limit reached
            raise TokenLimitError()
    """

    def __init__(
        self,
        total_tokens: int,
        max_per_call: int = 4096,
        warn_at_percent: int = 80,
    ):
        """Initialise token budget.

        Args:
            total_tokens: Total token budget
            max_per_call: Maximum tokens per single call
            warn_at_percent: Warning threshold (0-100)
        """
        self.total_tokens = total_tokens
        self.max_per_call = max_per_call
        self.warn_at_percent = warn_at_percent
        self.used_tokens = 0

    def can_use(self, estimated_tokens: int) -> bool:
        """Check if tokens are available.

        Args:
            estimated_tokens: Estimated tokens to use

        Returns:
            True if tokens available
        """
        if estimated_tokens > self.max_per_call:
            logger.warning(
                f"Estimated tokens {estimated_tokens} exceeds max per call {self.max_per_call}"
            )
            return False

        if self.used_tokens + estimated_tokens > self.total_tokens:
            logger.warning(f"Token budget exceeded: {self.used_tokens}/{self.total_tokens}")
            return False

        # Check warning threshold
        percent_used = (self.used_tokens / self.total_tokens) * 100
        if percent_used >= self.warn_at_percent:
            logger.warning(f"Token usage at {percent_used:.1f}% of budget")

        return True

    def use(self, actual_tokens: int) -> None:
        """Record token usage.

        Args:
            actual_tokens: Actual tokens consumed
        """
        self.used_tokens += actual_tokens
        logger.debug(f"Tokens used: {actual_tokens}, Total: {self.used_tokens}/{self.total_tokens}")

    def remaining(self) -> int:
        """Get remaining tokens.

        Returns:
            Remaining tokens
        """
        return max(0, self.total_tokens - self.used_tokens)

    def percent_used(self) -> float:
        """Get percentage of budget used.

        Returns:
            Percentage (0-100)
        """
        return (self.used_tokens / self.total_tokens) * 100

    def exceeded(self) -> bool:
        """Check if budget exceeded.

        Returns:
            True if budget exceeded
        """
        return self.used_tokens > self.total_tokens

    def should_warn(self) -> bool:
        """Check if warning threshold reached.

        Returns:
            True if at or above warning threshold
        """
        return self.percent_used() >= self.warn_at_percent
