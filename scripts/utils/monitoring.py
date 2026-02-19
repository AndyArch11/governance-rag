"""Production monitoring and observability with OpenTelemetry.

Provides:
- Distributed tracing (traces with spans)
- Metrics collection (counters, histograms, gauges)
- Structured logging with context propagation
- Token usage tracking for LLM calls
- Performance metrics collection

Usage:
    from scripts.utils.monitoring import get_tracer, get_meter, init_monitoring

    # Initialise at startup
    init_monitoring(service_name="rag-system")

    # Use tracer in code
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("operation_name") as span:
        span.set_attribute("key", "value")
        # operation code

    # Track metrics
    meter = get_meter(__name__)
    counter = meter.create_counter("llm_tokens_used", description="Total LLM tokens")
    counter.add(100)
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional

try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.metrics.otlp.proto.grpc.metric_exporter import (  # type: ignore[import-not-found]
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.trace.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-not-found]
        OTLPSpanExporter,
    )
    from opentelemetry.instrumentation.requests import (  # type: ignore[import-not-found]
        RequestsInstrumentor,
    )
    from opentelemetry.instrumentation.urllib3 import (  # type: ignore[import-not-found]
        URLLib3Instrumentor,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


logger = logging.getLogger(__name__)

# Global tracer and meter
_tracer: Optional[trace.Tracer] = None
_meter: Optional[metrics.Meter] = None
_initialised = False


def init_monitoring(
    service_name: str = "rag-system",
    environment: str = "dev",
    version: str = "1.0.0",
    otlp_endpoint: Optional[str] = None,
    enable_console_exporter: bool = True,
) -> None:
    """Initialise OpenTelemetry monitoring.

    Args:
        service_name: Name of the service for identification
        environment: Environment name (dev, staging, prod)
        version: Application version
        otlp_endpoint: OTLP gRPC endpoint (e.g., "http://localhost:4317")
                      If None, uses environment variable OTEL_EXPORTER_OTLP_ENDPOINT
        enable_console_exporter: Enable console exporter for local debugging
    """
    global _tracer, _meter, _initialised

    if _initialised:
        logger.debug("Monitoring already initialised")
        return

    if not OTEL_AVAILABLE:
        logger.warning(
            "OpenTelemetry not installed. Monitoring disabled. Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
        )
        return

    try:
        # Create resource
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": version,
                "deployment.environment": environment,
            }
        )

        # Setup tracing
        trace_provider = TracerProvider(resource=resource)

        # Add OTLP exporter if endpoint provided or environment variable set
        otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otlp_endpoint:
            trace_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
            logger.info(f"OTLP trace exporter configured: {otlp_endpoint}")

        trace.set_tracer_provider(trace_provider)

        # Setup metrics
        metric_readers = []
        if otlp_endpoint:
            metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint)
            metric_readers.append(PeriodicExportingMetricReader(metric_exporter))
            logger.info(f"OTLP metric exporter configured: {otlp_endpoint}")

        meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
        metrics.set_meter_provider(meter_provider)

        # Instrument popular libraries
        try:
            RequestsInstrumentor().instrument()
            URLLib3Instrumentor().instrument()
            logger.debug("Library instrumentation enabled")
        except Exception as e:
            logger.debug(f"Could not instrument libraries: {e}")

        # Get tracer and meter
        _tracer = trace.get_tracer(__name__)
        _meter = metrics.get_meter(__name__)
        _initialised = True

        logger.info(f"✓ OpenTelemetry monitoring initialised ({service_name}, {environment})")

    except Exception as e:
        logger.error(f"Failed to initialise monitoring: {e}", exc_info=True)


def get_tracer(name: str = __name__) -> trace.Tracer:
    """Get tracer for creating spans.

    Args:
        name: Module name for tracer identification

    Returns:
        Tracer instance
    """
    if not _initialised:
        init_monitoring()

    if _tracer is None:
        # Fallback to noop tracer if OpenTelemetry not available
        return trace.get_tracer(__name__)

    return trace.get_tracer(name)


def get_meter(name: str = __name__) -> metrics.Meter:
    """Get meter for recording metrics.

    Args:
        name: Module name for meter identification

    Returns:
        Meter instance
    """
    if not _initialised:
        init_monitoring()

    if _meter is None:
        # Fallback to noop meter if OpenTelemetry not available
        return metrics.get_meter(__name__)

    return metrics.get_meter(name)


@contextmanager
def trace_operation(operation_name: str, attributes: Optional[Dict[str, Any]] = None):
    """Context manager for tracing an operation.

    Usage:
        with trace_operation("database_query", {"query_type": "semantic_search"}):
            results = search_collection(query)

    Args:
        operation_name: Name of the operation
        attributes: Optional attributes to add to span

    Yields:
        The current span
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(operation_name) as span:
        if attributes:
            for key, value in attributes.items():
                try:
                    span.set_attribute(key, value)
                except Exception:
                    # Skip invalid attributes
                    pass

        yield span


class LLMTokenCounter:
    """Track token usage across LLM calls."""

    def __init__(self):
        """Initialise token counter."""
        self.meter = get_meter(__name__)
        self.total_tokens = self.meter.create_counter(
            "llm.tokens.total",
            description="Total tokens used in LLM calls",
            unit="1",
        )
        self.input_tokens = self.meter.create_counter(
            "llm.tokens.input",
            description="Input tokens sent to LLM",
            unit="1",
        )
        self.output_tokens = self.meter.create_counter(
            "llm.tokens.output",
            description="Output tokens generated by LLM",
            unit="1",
        )
        self.calls = self.meter.create_counter(
            "llm.calls.total",
            description="Total number of LLM calls",
            unit="1",
        )
        self.errors = self.meter.create_counter(
            "llm.calls.errors",
            description="Number of LLM call errors",
            unit="1",
        )

    def record_tokens(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        success: bool = True,
    ) -> None:
        """Record token usage for an LLM call.

        Args:
            model: Model name/identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            success: Whether the call succeeded
        """
        attributes = {"model": model}

        total = input_tokens + output_tokens
        self.total_tokens.add(total, attributes)
        self.input_tokens.add(input_tokens, attributes)
        self.output_tokens.add(output_tokens, attributes)
        self.calls.add(1, attributes)

        if not success:
            self.errors.add(1, attributes)

        logger.debug(
            f"LLM tokens recorded: model={model}, input={input_tokens}, output={output_tokens}"
        )


class PerformanceMetrics:
    """Track performance metrics for core operations."""

    def __init__(self):
        """Initialise performance metrics."""
        self.meter = get_meter(__name__)

        # Retrieval metrics
        self.retrieval_latency = self.meter.create_histogram(
            "retrieval.latency_ms",
            description="Retrieval operation latency",
            unit="ms",
        )
        self.retrieval_count = self.meter.create_counter(
            "retrieval.count",
            description="Number of retrieval operations",
            unit="1",
        )

        # Generation metrics
        self.generation_latency = self.meter.create_histogram(
            "generation.latency_ms",
            description="LLM generation latency",
            unit="ms",
        )
        self.generation_count = self.meter.create_counter(
            "generation.count",
            description="Number of generation operations",
            unit="1",
        )

        # Cache metrics
        self.cache_hits = self.meter.create_counter(
            "cache.hits",
            description="Cache hit count",
            unit="1",
        )
        self.cache_misses = self.meter.create_counter(
            "cache.misses",
            description="Cache miss count",
            unit="1",
        )

    def record_retrieval(
        self, latency_ms: float, result_count: int, cache_hit: bool = False
    ) -> None:
        """Record retrieval operation metrics.

        Args:
            latency_ms: Operation latency in milliseconds
            result_count: Number of results retrieved
            cache_hit: Whether result came from cache
        """
        attributes = {"cache_hit": str(cache_hit), "result_count": result_count}
        self.retrieval_latency.record(latency_ms, attributes)
        self.retrieval_count.add(1, attributes)

        if cache_hit:
            self.cache_hits.add(1)
        else:
            self.cache_misses.add(1)

    def record_generation(self, latency_ms: float, model: str) -> None:
        """Record generation operation metrics.

        Args:
            latency_ms: Generation latency in milliseconds
            model: Model identifier
        """
        attributes = {"model": model}
        self.generation_latency.record(latency_ms, attributes)
        self.generation_count.add(1, attributes)


# Global metric instances
_token_counter: Optional[LLMTokenCounter] = None
_perf_metrics: Optional[PerformanceMetrics] = None


def get_token_counter() -> LLMTokenCounter:
    """Get global token counter instance."""
    global _token_counter
    if _token_counter is None:
        _token_counter = LLMTokenCounter()
    return _token_counter


def get_perf_metrics() -> PerformanceMetrics:
    """Get global performance metrics instance."""
    global _perf_metrics
    if _perf_metrics is None:
        _perf_metrics = PerformanceMetrics()
    return _perf_metrics


# Structured logging helpers
def create_context_logger(module_name: str) -> logging.LoggerAdapter:
    """Create a logger with context support.

    Args:
        module_name: Module name for logger identification

    Returns:
        LoggerAdapter with context support
    """
    logger = logging.getLogger(module_name)
    return logging.LoggerAdapter(logger, {})


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **context,
) -> None:
    """Log with structured context.

    Args:
        logger: Logger instance
        level: Log level (logging.INFO, etc.)
        message: Log message
        **context: Context key-value pairs to include in log
    """
    extra = {"context": context, "timestamp": datetime.utcnow().isoformat()}
    logger.log(level, message, extra=extra)
