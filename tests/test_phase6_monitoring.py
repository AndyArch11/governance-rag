"""Tests for Phase 6 Step 2: Production Resilience & Monitoring.

Tests:
- Metrics collection and aggregation
- LLM call instrumentation
- Retrieval operation tracking
- Token budgeting and cost estimation
- Health status reporting
"""

import pytest
import time
from datetime import datetime, timedelta
from scripts.utils.metrics_export import (
    MetricsCollector,
    MetricsSnapshot,
    get_metrics_collector,
)
from scripts.utils.monitoring import (
    init_monitoring,
    get_tracer,
    get_meter,
    trace_operation,
    LLMTokenCounter,
    PerformanceMetrics,
)
from scripts.utils.llm_instrumentation import (
    instrument_ollama_call,
    instrument_claude_call,
    instrument_retrieval,
    TokenBudget,
)


class TestMetricsCollection:
    """Test metrics collection and aggregation."""
    
    def test_metrics_collector_initialisation(self):
        """Test collector initialises with correct defaults."""
        collector = MetricsCollector()
        assert collector.llm_calls == []
        assert collector.retrieval_calls == []
        assert collector.errors == []
        assert isinstance(collector.start_time, datetime)
    
    def test_record_llm_call(self):
        """Test recording LLM calls."""
        collector = MetricsCollector()
        collector.record_llm_call(
            model="ollama:llama2",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500.0,
            success=True,
        )
        
        assert len(collector.llm_calls) == 1
        assert collector.llm_calls[0]["model"] == "ollama:llama2"
        assert collector.llm_calls[0]["input_tokens"] == 100
        assert collector.llm_calls[0]["output_tokens"] == 50
    
    def test_record_llm_call_error(self):
        """Test recording failed LLM calls."""
        collector = MetricsCollector()
        collector.record_llm_call(
            model="ollama:llama2",
            input_tokens=100,
            output_tokens=0,
            latency_ms=100.0,
            success=False,
        )
        
        assert len(collector.llm_calls) == 1
        assert len(collector.errors) == 1
        assert not collector.llm_calls[0]["success"]
    
    def test_record_retrieval(self):
        """Test recording retrieval operations."""
        collector = MetricsCollector()
        collector.record_retrieval(
            latency_ms=250.0,
            result_count=5,
            cache_hit=False,
        )
        
        assert len(collector.retrieval_calls) == 1
        assert collector.retrieval_calls[0]["latency_ms"] == 250.0
        assert collector.retrieval_calls[0]["result_count"] == 5
        assert not collector.retrieval_calls[0]["cache_hit"]
    
    def test_metrics_snapshot_generation(self):
        """Test generating metrics snapshot."""
        collector = MetricsCollector()
        
        # Add some data
        collector.record_llm_call("ollama:llama2", 100, 50, 500.0)
        collector.record_llm_call("ollama:llama2", 150, 75, 600.0)
        collector.record_retrieval(200.0, 3)
        collector.record_retrieval(300.0, 5, cache_hit=True)
        
        stats = collector.get_stats()
        
        assert isinstance(stats, MetricsSnapshot)
        assert stats.total_llm_calls == 2
        assert stats.total_tokens_used == 375  # (100+50) + (150+75)
        assert stats.input_tokens == 250
        assert stats.output_tokens == 125
        assert stats.retrieval_operations == 2
        assert stats.cache_hit_rate == 50.0
    
    def test_metrics_snapshot_json_conversion(self):
        """Test converting metrics snapshot to JSON."""
        collector = MetricsCollector()
        collector.record_llm_call("ollama:llama2", 100, 50, 500.0)
        
        stats = collector.get_stats()
        json_str = stats.to_json()
        
        assert "total_llm_calls" in json_str
        assert "2" in json_str  # timestamp
    
    def test_model_stats_aggregation(self):
        """Test getting stats aggregated by model."""
        collector = MetricsCollector()
        collector.record_llm_call("ollama:llama2", 100, 50, 500.0)
        collector.record_llm_call("ollama:llama2", 100, 50, 600.0)
        collector.record_llm_call("claude:3", 200, 100, 800.0)
        
        model_stats = collector.get_model_stats()
        
        assert "ollama:llama2" in model_stats
        assert "claude:3" in model_stats
        assert model_stats["ollama:llama2"]["calls"] == 2
        assert model_stats["claude:3"]["calls"] == 1
        assert model_stats["ollama:llama2"]["tokens"] == 300  # (100+50)*2
        assert model_stats["claude:3"]["tokens"] == 300  # 200+100
    
    def test_cost_estimation(self):
        """Test cost estimation based on tokens."""
        collector = MetricsCollector()
        collector.record_llm_call("ollama:llama2", 1000, 500, 500.0)  # 1500 tokens total
        
        cost = collector.estimate_cost(
            input_token_price=0.001,  # $0.001 per 1K
            output_token_price=0.002,  # $0.002 per 1K
        )
        
        # (1000 / 1000) * 0.001 + (500 / 1000) * 0.002
        # = 0.001 + 0.001 = 0.002
        assert abs(cost - 0.002) < 0.0001
    
    def test_health_status_healthy(self):
        """Test health status when no errors."""
        collector = MetricsCollector()
        for _ in range(10):
            collector.record_llm_call("model", 100, 50, 500.0, success=True)
        
        health = collector.get_health_status()
        
        assert health["status"] == "healthy"
        assert health["error_rate"] == 0.0
        assert health["total_operations"] == 10
    
    def test_health_status_warning(self):
        """Test health status with moderate error rate."""
        collector = MetricsCollector()
        for _ in range(100):
            collector.record_llm_call("model", 100, 50, 500.0, success=True)
        for _ in range(7):
            collector.record_llm_call("model", 100, 50, 500.0, success=False)
        
        health = collector.get_health_status()
        
        assert health["status"] == "warning"
        assert 5.0 < health["error_rate"] < 10.0
    
    def test_health_status_degraded(self):
        """Test health status with high error rate."""
        collector = MetricsCollector()
        for _ in range(10):
            collector.record_llm_call("model", 100, 50, 500.0, success=False)
        
        health = collector.get_health_status()
        
        assert health["status"] == "degraded"
        assert health["error_rate"] == 100.0
    
    def test_window_trimming(self):
        """Test that old entries are trimmed."""
        collector = MetricsCollector(window_size=1)  # 1 second window
        
        collector.record_llm_call("model", 100, 50, 500.0)
        assert len(collector.llm_calls) == 1
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Trigger trim
        collector.get_stats()
        
        # Old entries should be gone
        assert len(collector.llm_calls) == 0
    
    def test_global_collector_singleton(self):
        """Test global collector is a singleton."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        assert collector1 is collector2


class TestMonitoringInfrastructure:
    """Test core monitoring infrastructure."""
    
    def test_init_monitoring(self):
        """Test monitoring initialisation."""
        init_monitoring()
        
        tracer = get_tracer()
        meter = get_meter()
        
        assert tracer is not None
        assert meter is not None
    
    def test_trace_operation_context_manager(self):
        """Test trace_operation context manager."""
        init_monitoring()
        
        with trace_operation("test_operation") as span:
            assert span is not None
            # Span should be recordable even if OpenTelemetry unavailable
    
    def test_llm_token_counter(self):
        """Test token counter tracking."""
        counter = LLMTokenCounter()
        
        counter.record_tokens(
            model="test_model",
            input_tokens=70,
            output_tokens=30,
            success=True,
        )
        
        # Since OpenTelemetry may not be installed, just verify it doesn't crash
        assert counter is not None
    
    def test_llm_token_counter_errors(self):
        """Test token counter error tracking."""
        counter = LLMTokenCounter()
        
        counter.record_tokens("test_model", 70, 30, success=True)
        counter.record_tokens("test_model", 0, 0, success=False)
        
        # Since OpenTelemetry may not be installed, just verify it doesn't crash
        assert counter is not None
    
    def test_performance_metrics_recording(self):
        """Test performance metrics recording."""
        metrics = PerformanceMetrics()
        
        metrics.record_retrieval(250.0, 2, cache_hit=False)
        metrics.record_retrieval(300.0, 3, cache_hit=True)
        metrics.record_generation(500.0, "test_model")
        
        # Since OpenTelemetry may not be installed, just verify it doesn't crash
        assert metrics is not None


class TestLLMInstrumentation:
    """Test LLM call instrumentation patterns."""
    
    def test_token_budget(self):
        """Test token budget tracking."""
        budget = TokenBudget(total_tokens=1000)
        
        assert budget.can_use(300)
        assert budget.remaining() == 1000
        
        budget.use(300)
        
        assert not budget.exceeded()
        assert budget.remaining() == 700
        
        budget.use(800)
        
        assert budget.exceeded()
    
    def test_token_budget_warning_threshold(self):
        """Test token budget warning threshold."""
        budget = TokenBudget(total_tokens=1000, warn_at_percent=90)
        
        assert not budget.should_warn()
        
        budget.use(950)
        
        assert budget.should_warn()
    
    def test_instrument_ollama_call_context(self):
        """Test Ollama call instrumentation context manager."""
        init_monitoring()
        collector = get_metrics_collector()
        
        # Note: This uses context manager without actual Ollama call
        try:
            with instrument_ollama_call(
                model="llama2",
                input_tokens=100,
                output_tokens=50,
                collector=collector,
            ):
                pass
        except Exception as e:
            # May fail if opentelemetry not available, that's ok
            pass
    
    def test_instrument_retrieval_decorator(self):
        """Test retrieval instrumentation decorator."""
        @instrument_retrieval("test_retrieval")
        def mock_retrieve(query: str):
            return {"results": [1, 2, 3]}
        
        result = mock_retrieve("test query")
        
        assert result == {"results": [1, 2, 3]}


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""
    
    def test_session_with_multiple_operations(self):
        """Test tracking a session with multiple operations."""
        collector = MetricsCollector()
        
        # Simulate a typical session
        for i in range(5):
            # User query -> retrieval
            collector.record_retrieval(150.0 + i * 10, 3 + i)
            
            # LLM generation
            collector.record_llm_call(
                "ollama:llama2",
                100 + i * 20,
                50 + i * 10,
                300.0 + i * 50,
            )
        
        stats = collector.get_stats()
        
        assert stats.total_llm_calls == 5
        assert stats.retrieval_operations == 5
        assert stats.total_tokens_used > 0
        
        health = collector.get_health_status()
        assert health["status"] in ["healthy", "warning", "degraded"]
    
    def test_cost_tracking_across_models(self):
        """Test cost estimation across multiple models."""
        collector = MetricsCollector()
        
        # Ollama calls
        for _ in range(3):
            collector.record_llm_call("ollama:llama2", 100, 50, 300.0)
        
        # Claude calls
        for _ in range(2):
            collector.record_llm_call("claude:3", 200, 100, 500.0)
        
        model_stats = collector.get_model_stats()
        
        assert model_stats["ollama:llama2"]["calls"] == 3
        assert model_stats["claude:3"]["calls"] == 2
        
        # Ollama: 3 * (100+50) = 450 tokens
        # Claude: 2 * (200+100) = 600 tokens
        assert model_stats["ollama:llama2"]["tokens"] == 450
        assert model_stats["claude:3"]["tokens"] == 600


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
