"""Metrics collection and export for monitoring dashboards.

Provides:
- Metrics aggregation and reporting
- Health checks and system status
- Cost estimation based on token usage
- Real-time metrics endpoints

Usage:
    from scripts.utils.metrics_export import MetricsCollector

    collector = MetricsCollector()
    stats = collector.get_stats()

    # In Dash callback:
    @callback(
        Output("metrics-display", "children"),
        Input("metrics-interval", "n_intervals"),
    )
    def update_metrics(_):
        stats = collector.get_stats()
        return dcc.Graph(figure=stats.to_plotly_figure())
"""

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Snapshot of system metrics at a point in time."""

    timestamp: str
    total_llm_calls: int
    total_tokens_used: int
    input_tokens: int
    output_tokens: int
    llm_errors: int
    retrieval_operations: int
    retrieval_latency_avg_ms: float
    retrieval_latency_p99_ms: float
    generation_latency_avg_ms: float
    generation_latency_p99_ms: float
    cache_hit_rate: float
    uptime_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_plotly_figure(self) -> Optional[Any]:
        """Convert to Plotly figure with metrics visualisation.

        Returns:
            Plotly figure object or None if Plotly not available
        """
        if not HAS_PLOTLY:
            logger.warning("Plotly not installed; cannot generate visualisation")
            return None

        # Create subplots: 2x2 grid
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "LLM Operations & Tokens",
                "Retrieval Latency Distribution",
                "Generation Latency Distribution",
                "Cache Hit Rate & Errors",
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "indicator"}],
            ],
        )

        # Top-left: LLM calls and token usage
        fig.add_trace(
            go.Bar(
                x=["LLM Calls", "Errors"],
                y=[self.total_llm_calls, self.llm_errors],
                name="LLM Operations",
                marker_color=["#667eea", "#e74c3c"],
                text=[self.total_llm_calls, self.llm_errors],
                textposition="auto",
            ),
            row=1,
            col=1,
        )

        # Top-right: Token usage breakdown
        fig.add_trace(
            go.Bar(
                x=["Input Tokens", "Output Tokens"],
                y=[self.input_tokens, self.output_tokens],
                name="Tokens",
                marker_color=["#3498db", "#2ecc71"],
                text=[self.input_tokens, self.output_tokens],
                textposition="auto",
            ),
            row=1,
            col=2,
        )

        # Bottom-left: Retrieval latency (avg and p99)
        fig.add_trace(
            go.Bar(
                x=["Avg Latency (ms)", "P99 Latency (ms)"],
                y=[self.retrieval_latency_avg_ms, self.retrieval_latency_p99_ms],
                name="Retrieval",
                marker_color=["#f39c12", "#e67e22"],
                text=[
                    f"{self.retrieval_latency_avg_ms:.2f}",
                    f"{self.retrieval_latency_p99_ms:.2f}",
                ],
                textposition="auto",
            ),
            row=2,
            col=1,
        )

        # Bottom-right: Gauge for cache hit rate
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=self.cache_hit_rate,
                title={"text": "Cache Hit Rate (%)"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "#2ecc71"},
                    "steps": [
                        {"range": [0, 25], "color": "#e74c3c"},
                        {"range": [25, 50], "color": "#f39c12"},
                        {"range": [50, 75], "color": "#3498db"},
                        {"range": [75, 100], "color": "#2ecc71"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title_text=f"System Metrics Dashboard ({self.timestamp})",
            height=600,
            font=dict(size=12),
            hovermode="closest",
            showlegend=True,
            template="plotly_white",
        )

        # Update y-axes labels
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Tokens", row=1, col=2)
        fig.update_yaxes(title_text="Latency (ms)", row=2, col=1)

        return fig


class MetricsCollector:
    """Collects and aggregates metrics from the system."""

    def __init__(self, window_size: int = 3600):
        """Initialise metrics collector.

        Args:
            window_size: Time window in seconds for metrics (default: 1 hour)
        """
        self.window_size = window_size
        self.start_time = datetime.now(timezone.utc)

        # Metrics storage (keep last N samples)
        self.llm_calls: List[Dict[str, Any]] = []
        self.retrieval_calls: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []

    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """Record an LLM call.

        Args:
            model: Model identifier
            input_tokens: Input token count
            output_tokens: Output token count
            latency_ms: Call latency in milliseconds
            success: Whether call succeeded
        """
        self.llm_calls.append(
            {
                "timestamp": datetime.now(timezone.utc),
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "success": success,
            }
        )

        # Trim old entries
        self._trim_old_entries()

        if not success:
            self.errors.append(
                {
                    "timestamp": datetime.now(timezone.utc),
                    "type": "llm_call",
                    "model": model,
                }
            )

    def record_retrieval(
        self,
        latency_ms: float,
        result_count: int,
        cache_hit: bool = False,
    ) -> None:
        """Record a retrieval operation.

        Args:
            latency_ms: Operation latency in milliseconds
            result_count: Number of results
            cache_hit: Whether result came from cache
        """
        self.retrieval_calls.append(
            {
                "timestamp": datetime.now(timezone.utc),
                "latency_ms": latency_ms,
                "result_count": result_count,
                "cache_hit": cache_hit,
            }
        )

        self._trim_old_entries()

    def _trim_old_entries(self) -> None:
        """Remove entries older than window size."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.window_size)

        self.llm_calls = [c for c in self.llm_calls if c["timestamp"] > cutoff_time]
        self.retrieval_calls = [c for c in self.retrieval_calls if c["timestamp"] > cutoff_time]
        self.errors = [e for e in self.errors if e["timestamp"] > cutoff_time]

    def get_stats(self) -> MetricsSnapshot:
        """Get current metrics snapshot.

        Returns:
            MetricsSnapshot with aggregated metrics
        """
        self._trim_old_entries()

        # Calculate LLM metrics
        total_llm_calls = len(self.llm_calls)
        successful_calls = sum(1 for c in self.llm_calls if c["success"])
        total_tokens = sum(c["input_tokens"] + c["output_tokens"] for c in self.llm_calls)
        input_tokens = sum(c["input_tokens"] for c in self.llm_calls)
        output_tokens = sum(c["output_tokens"] for c in self.llm_calls)

        # Calculate retrieval metrics
        retrieval_latencies = [c["latency_ms"] for c in self.retrieval_calls]
        retrieval_latency_avg = (
            sum(retrieval_latencies) / len(retrieval_latencies) if retrieval_latencies else 0.0
        )
        retrieval_latency_p99 = (
            sorted(retrieval_latencies)[int(len(retrieval_latencies) * 0.99)]
            if retrieval_latencies
            else 0.0
        )

        # Calculate generation metrics
        generation_latencies = [c["latency_ms"] for c in self.llm_calls]
        generation_latency_avg = (
            sum(generation_latencies) / len(generation_latencies) if generation_latencies else 0.0
        )
        generation_latency_p99 = (
            sorted(generation_latencies)[int(len(generation_latencies) * 0.99)]
            if generation_latencies
            else 0.0
        )

        # Calculate cache hit rate
        cache_hits = sum(1 for c in self.retrieval_calls if c["cache_hit"])
        total_retrievals = len(self.retrieval_calls)
        cache_hit_rate = (cache_hits / total_retrievals * 100) if total_retrievals > 0 else 0.0

        # Calculate uptime
        uptime_seconds = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        return MetricsSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_llm_calls=total_llm_calls,
            total_tokens_used=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            llm_errors=total_llm_calls - successful_calls,
            retrieval_operations=len(self.retrieval_calls),
            retrieval_latency_avg_ms=retrieval_latency_avg,
            retrieval_latency_p99_ms=retrieval_latency_p99,
            generation_latency_avg_ms=generation_latency_avg,
            generation_latency_p99_ms=generation_latency_p99,
            cache_hit_rate=cache_hit_rate,
            uptime_seconds=uptime_seconds,
        )

    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics by model.

        Returns:
            Dictionary of model -> metrics
        """
        self._trim_old_entries()

        model_stats = defaultdict(
            lambda: {
                "calls": 0,
                "tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "avg_latency_ms": 0.0,
                "errors": 0,
            }
        )

        for call in self.llm_calls:
            model = call["model"]
            model_stats[model]["calls"] += 1
            model_stats[model]["tokens"] += call["input_tokens"] + call["output_tokens"]
            model_stats[model]["input_tokens"] += call["input_tokens"]
            model_stats[model]["output_tokens"] += call["output_tokens"]
            if not call["success"]:
                model_stats[model]["errors"] += 1

        # Calculate average latencies
        for call in self.llm_calls:
            model = call["model"]
            if model_stats[model]["calls"] > 0:
                latencies = [c["latency_ms"] for c in self.llm_calls if c["model"] == model]
                model_stats[model]["avg_latency_ms"] = sum(latencies) / len(latencies)

        return dict(model_stats)

    def estimate_cost(
        self,
        input_token_price: float = 0.001,  # $ per 1K tokens
        output_token_price: float = 0.002,  # $ per 1K tokens
    ) -> float:
        """Estimate cost based on token usage.

        Args:
            input_token_price: Price per 1K input tokens
            output_token_price: Price per 1K output tokens

        Returns:
            Estimated cost in dollars
        """
        self._trim_old_entries()

        input_tokens = sum(c["input_tokens"] for c in self.llm_calls)
        output_tokens = sum(c["output_tokens"] for c in self.llm_calls)

        input_cost = (input_tokens / 1000) * input_token_price
        output_cost = (output_tokens / 1000) * output_token_price

        return input_cost + output_cost

    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status.

        Returns:
            Health status dict
        """
        stats = self.get_stats()

        # Calculate error rate
        error_rate = 0.0
        if stats.total_llm_calls > 0:
            error_rate = (stats.llm_errors / stats.total_llm_calls) * 100

        # Determine health status
        if error_rate > 10:
            status = "degraded"
        elif error_rate > 5:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "error_rate": error_rate,
            "uptime_hours": stats.uptime_seconds / 3600,
            "total_operations": stats.total_llm_calls + stats.retrieval_operations,
            "cache_hit_rate": stats.cache_hit_rate,
        }

    def to_html_summary(self) -> str:
        """Generate HTML summary of metrics for dashboard display.

        Returns:
            HTML string with formatted metrics
        """
        stats = self.get_stats()
        health = self.get_health_status()

        # Determine health colour
        health_colour_map = {
            "healthy": "#2ecc71",
            "warning": "#f39c12",
            "degraded": "#e74c3c",
        }
        health_colour = health_colour_map.get(health["status"], "#95a5a6")

        html_content = f"""
        <div style="font-family: monospace; line-height: 1.8; padding: 12px; background: #f9f9f9; border-radius: 4px; border-left: 4px solid {health_colour};">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div>
                    <strong>🔍 LLM Operations</strong><br/>
                    Total Calls: <span style="color: #667eea;">{stats.total_llm_calls}</span><br/>
                    Input Tokens: <span style="color: #3498db;">{stats.input_tokens:,}</span><br/>
                    Output Tokens: <span style="color: #2ecc71;">{stats.output_tokens:,}</span><br/>
                    Total Tokens: <span style="color: #9b59b6;">{stats.total_tokens_used:,}</span><br/>
                    Errors: <span style="color: {'#e74c3c' if stats.llm_errors > 0 else '#95a5a6'};">{stats.llm_errors}</span>
                </div>
                <div>
                    <strong>📊 Retrieval Operations</strong><br/>
                    Count: <span style="color: #667eea;">{stats.retrieval_operations}</span><br/>
                    Avg Latency: <span style="color: #f39c12;">{stats.retrieval_latency_avg_ms:.2f}ms</span><br/>
                    P99 Latency: <span style="color: #e67e22;">{stats.retrieval_latency_p99_ms:.2f}ms</span><br/>
                    Cache Hit Rate: <span style="color: {'#2ecc71' if stats.cache_hit_rate > 50 else '#f39c12'};">{stats.cache_hit_rate:.1f}%</span>
                </div>
                <div>
                    <strong>⚡ Generation Latency</strong><br/>
                    Avg: <span style="color: #f39c12;">{stats.generation_latency_avg_ms:.2f}ms</span><br/>
                    P99: <span style="color: #e67e22;">{stats.generation_latency_p99_ms:.2f}ms</span><br/>
                    <br/>
                    <strong>🏥 System Health</strong><br/>
                    Status: <span style="color: {health_colour}; font-weight: bold;">{health['status'].upper()}</span><br/>
                    Error Rate: <span style="color: {'#e74c3c' if health['error_rate'] > 5 else '#2ecc71'};">{health['error_rate']:.2f}%</span>
                </div>
                <div>
                    <strong>📈 Summary</strong><br/>
                    Uptime: <span style="color: #3498db;">{health['uptime_hours']:.2f}h</span><br/>
                    Total Operations: <span style="color: #667eea;">{health['total_operations']}</span><br/>
                    Recorded At: <span style="color: #95a5a6; font-size: 12px;">{stats.timestamp}</span>
                </div>
            </div>
        </div>
        """
        return html_content


# Global metrics collector
_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
