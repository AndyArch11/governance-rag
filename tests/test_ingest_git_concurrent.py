"""Tests for concurrent file processing and resource monitoring in ingest_git.py

Tests the ConcurrentFileProcessor and ResourceMonitor classes for:
- Parallel file processing with thread pool
- Resource usage tracking (CPU, memory)
- Performance metrics collection and reporting
"""

import threading
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call
from typing import List, Tuple

from scripts.ingest.git.git_ingest_config import GitIngestConfig
from scripts.ingest.ingest_git import (
    ConcurrentFileProcessor,
    ResourceMonitor,
    ProcessingStats,
    ResourceMetrics,
)


@pytest.fixture
def mock_config():
    """Create mock GitIngestConfig."""
    config = MagicMock(spec=GitIngestConfig)
    config.enable_parent_child_chunking = True
    config.enable_dlp = False
    config.generate_summaries = False
    config.bm25_indexing_enabled = False
    config.git_provider = "github"
    config.git_host = "https://github.com"
    config.github_host = "https://github.com"
    config.bitbucket_host = ""
    return config


@pytest.fixture
def mock_collections():
    """Create mock ChromaDB collections."""
    chunk_collection = MagicMock()
    doc_collection = MagicMock()
    chunk_collection.get.return_value = {"ids": [], "metadatas": []}
    return chunk_collection, doc_collection


@pytest.fixture
def mock_parser():
    """Create mock CodeParser."""
    parser = MagicMock()
    mock_result = MagicMock()
    mock_result.language = "python"
    mock_result.file_type = "source"
    mock_result.to_dict.return_value = {
        "language": "python",
        "file_type": "source",
        "service_name": "test",
        "exports": [],
        "endpoints": [],
        "external_dependencies": []
    }
    parser.parse_file.return_value = mock_result
    return parser


class TestResourceMetrics:
    """Tests for ResourceMetrics data class."""
    
    def test_resource_metrics_initialisation(self):
        """Test ResourceMetrics can be initialised."""
        metric = ResourceMetrics(
            cpu_percent=50.5,
            memory_percent=25.3,
            memory_mb=1024.5,
            thread_count=8
        )
        
        assert metric.cpu_percent == 50.5
        assert metric.memory_percent == 25.3
        assert metric.memory_mb == 1024.5
        assert metric.thread_count == 8
    
    def test_resource_metrics_string_representation(self):
        """Test ResourceMetrics string representation."""
        metric = ResourceMetrics(
            cpu_percent=50.5,
            memory_percent=25.3,
            memory_mb=1024.5,
            thread_count=8
        )
        
        repr_str = str(metric)
        assert "50.5" in repr_str
        assert "25.3" in repr_str
        assert "1024" in repr_str
        assert "8" in repr_str


class TestProcessingStats:
    """Tests for ProcessingStats collection."""
    
    def test_processing_stats_initialisation(self):
        """Test ProcessingStats initialisation."""
        stats = ProcessingStats(total_files=100)
        
        assert stats.total_files == 100
        assert stats.files_processed == 0
        assert stats.files_failed == 0
        assert stats.files_skipped == 0
        assert stats.chunks_created == 0
    
    def test_record_metric(self):
        """Test recording metrics into stats."""
        stats = ProcessingStats()
        metric1 = ResourceMetrics(cpu_percent=30.0, memory_mb=512.0)
        metric2 = ResourceMetrics(cpu_percent=50.0, memory_mb=1024.0)
        
        stats.record_metric(metric1)
        stats.record_metric(metric2)
        
        assert len(stats.metrics_history) == 2
        assert stats.peak_cpu_percent == 50.0
        assert stats.peak_memory_mb == 1024.0
    
    def test_finalise_calculates_time(self):
        """Test finalise updates total_time."""
        stats = ProcessingStats()
        start_time = stats.start_time
        
        # Sleep briefly
        time.sleep(0.1)
        stats.finalise()
        
        assert stats.total_time >= 0.1
        assert stats.total_time < 1.0  # Should be quick
    
    def test_summary_generation(self):
        """Test summary() method returns correct structure."""
        stats = ProcessingStats(
            total_files=100,
            files_processed=95,
            files_failed=3,
            files_skipped=2,
        )
        metric = ResourceMetrics(cpu_percent=40.0, memory_mb=512.0)
        stats.record_metric(metric)
        stats.finalise()
        
        summary = stats.summary()
        
        assert "total_files" in summary
        assert "files_processed" in summary
        assert "files_failed" in summary
        assert "files_skipped" in summary
        assert "peak_memory_mb" in summary
        assert "peak_cpu_percent" in summary
        assert "avg_memory_mb" in summary
        assert "files_per_second" in summary
        
        assert summary["total_files"] == 100
        assert summary["files_processed"] == 95
        assert summary["peak_memory_mb"] == 512.0


class TestResourceMonitor:
    """Tests for ResourceMonitor thread."""
    
    def test_resource_monitor_initialisation(self):
        """Test ResourceMonitor can be created."""
        stats = ProcessingStats()
        monitor = ResourceMonitor(interval=1.0, stats=stats)
        
        assert monitor.interval == 1.0
        assert monitor.stats == stats
        assert monitor.daemon is True
    
    def test_resource_monitor_collects_metrics(self):
        """Test ResourceMonitor collects resource metrics."""
        stats = ProcessingStats()
        monitor = ResourceMonitor(interval=0.05, stats=stats)
        
        monitor.start()
        time.sleep(0.25)  # Allow a few samples
        monitor.stop()
        monitor.join(timeout=2.0)
        
        # Should have collected at least 1-2 samples
        assert len(stats.metrics_history) >= 1
        assert stats.peak_memory_mb > 0.0
    
    def test_resource_monitor_stop_halts_collection(self):
        """Test that stop() stops metric collection."""
        stats = ProcessingStats()
        monitor = ResourceMonitor(interval=0.05, stats=stats)
        
        monitor.start()
        time.sleep(0.15)
        count_before = len(stats.metrics_history)
        
        monitor.stop()
        monitor.join(timeout=1.0)
        time.sleep(0.1)  # Should not collect more
        
        # Count should not increase significantly after stop
        assert len(stats.metrics_history) == count_before


class TestConcurrentFileProcessor:
    """Tests for ConcurrentFileProcessor."""
    
    def test_processor_initialisation(self, mock_collections, mock_parser, mock_config):
        """Test ConcurrentFileProcessor initialisation."""
        chunk_collection, doc_collection = mock_collections
        
        processor = ConcurrentFileProcessor(
            max_workers=4,
            chunk_collection=chunk_collection,
            doc_collection=doc_collection,
            parser=mock_parser,
            config=mock_config,
        )
        
        assert processor.max_workers == 4
        assert processor.chunk_collection == chunk_collection
        assert processor.doc_collection == doc_collection
        assert processor.parser == mock_parser
        assert processor.config == mock_config
        assert processor.stats.total_files == 0
    
    def test_process_files_with_single_file(self, mock_collections, mock_parser, mock_config):
        """Test processing a single file."""
        chunk_collection, doc_collection = mock_collections
        
        processor = ConcurrentFileProcessor(
            max_workers=1,
            chunk_collection=chunk_collection,
            doc_collection=doc_collection,
            parser=mock_parser,
            config=mock_config,
        )
        
        files = [("test.py", "print('hello')", time.time())]
        
        with patch("scripts.ingest.ingest_git.process_code_file") as mock_process, \
             patch("scripts.ingest.ingest_git.ResourceMonitor"):
            mock_process.return_value = True
            
            stats = processor.process_files(
                files=files,
                project_key="TEST",
                repo_slug="test-repo",
                branch="main",
            )
            assert stats.files_processed == 1
            assert stats.files_failed == 0
            mock_process.assert_called_once()
    
    def test_process_files_with_multiple_files(self, mock_collections, mock_parser, mock_config):
        """Test processing multiple files concurrently."""
        chunk_collection, doc_collection = mock_collections
        
        processor = ConcurrentFileProcessor(
            max_workers=2,
            chunk_collection=chunk_collection,
            doc_collection=doc_collection,
            parser=mock_parser,
            config=mock_config,
        )
        
        files = [
            ("file1.py", "code1", time.time()),
            ("file2.py", "code2", time.time()),
            ("file3.py", "code3", time.time()),
        ]
        
        with patch("scripts.ingest.ingest_git.process_code_file") as mock_process, \
             patch("scripts.ingest.ingest_git.ResourceMonitor"):
            mock_process.return_value = True
            
            stats = processor.process_files(
                files=files,
                project_key="TEST",
                repo_slug="test-repo",
                branch="main",
            )
            
            assert stats.total_files == 3
            assert stats.files_processed == 3
            assert stats.files_failed == 0
            assert mock_process.call_count == 3
    
    def test_process_files_handles_failures(self, mock_collections, mock_parser, mock_config):
        """Test that processor counts failed files."""
        chunk_collection, doc_collection = mock_collections
        
        processor = ConcurrentFileProcessor(
            max_workers=2,
            chunk_collection=chunk_collection,
            doc_collection=doc_collection,
            parser=mock_parser,
            config=mock_config,
        )
        
        files = [
            ("good.py", "code", time.time()),
            ("bad.py", "bad code", time.time()),
        ]
        
        with patch("scripts.ingest.ingest_git.process_code_file") as mock_process, \
             patch("scripts.ingest.ingest_git.ResourceMonitor"):
            # First succeeds, second fails
            mock_process.side_effect = [True, False]
            
            stats = processor.process_files(
                files=files,
                project_key="TEST",
                repo_slug="test-repo",
                branch="main",
            )
            
            assert stats.total_files == 2
            assert stats.files_processed == 1
            assert stats.files_failed == 1
    
    def test_process_files_starts_resource_monitor(self, mock_collections, mock_parser, mock_config):
        """Test that process_files starts ResourceMonitor."""
        chunk_collection, doc_collection = mock_collections
        
        processor = ConcurrentFileProcessor(
            max_workers=1,
            chunk_collection=chunk_collection,
            doc_collection=doc_collection,
            parser=mock_parser,
            config=mock_config,
        )
        
        files = [("test.py", "code", time.time())]
        
        with patch("scripts.ingest.ingest_git.process_code_file") as mock_process, \
             patch("scripts.ingest.ingest_git.ResourceMonitor") as mock_monitor:
            
            mock_process.return_value = True
            mock_monitor_instance = MagicMock()
            mock_monitor.return_value = mock_monitor_instance
            
            processor.process_files(
                files=files,
                project_key="TEST",
                repo_slug="test-repo",
                branch="main",
            )
            
            # Monitor should be created and started
            mock_monitor.assert_called_once()
            mock_monitor_instance.start.assert_called_once()
            mock_monitor_instance.stop.assert_called_once()
    
    def test_process_files_finalises_stats(self, mock_collections, mock_parser, mock_config):
        """Test that stats are finalised after processing."""
        chunk_collection, doc_collection = mock_collections
        
        processor = ConcurrentFileProcessor(
            max_workers=1,
            chunk_collection=chunk_collection,
            doc_collection=doc_collection,
            parser=mock_parser,
            config=mock_config,
        )
        
        files = [("test.py", "code", time.time())]
        
        with patch("scripts.ingest.ingest_git.process_code_file") as mock_process, \
             patch("scripts.ingest.ingest_git.ResourceMonitor"):
            mock_process.return_value = True
            
            stats = processor.process_files(
                files=files,
                project_key="TEST",
                repo_slug="test-repo",
                branch="main",
            )
            
            # Stats should be finalised (total_time should be set)
            assert stats.total_time > 0
    
    def test_processor_uses_configured_max_workers(self, mock_collections, mock_parser, mock_config):
        """Test that processor respects max_workers setting."""
        chunk_collection, doc_collection = mock_collections
        
        processor = ConcurrentFileProcessor(
            max_workers=3,
            chunk_collection=chunk_collection,
            doc_collection=doc_collection,
            parser=mock_parser,
            config=mock_config,
        )
        
        # Verify max_workers is set
        assert processor.max_workers == 3


class TestConcurrentIntegration:
    """Integration tests for concurrent processing."""
    
    def test_processor_workflow_summary(self, mock_collections, mock_parser, mock_config):
        """Test that processor correctly assembles workflow stats."""
        chunk_collection, doc_collection = mock_collections
        
        processor = ConcurrentFileProcessor(
            max_workers=2,
            chunk_collection=chunk_collection,
            doc_collection=doc_collection,
            parser=mock_parser,
            config=mock_config,
        )
        
        # Verify summary method provides all required metrics
        summary = processor.stats.summary()
        
        required_keys = [
            "total_files", "files_processed", "files_failed", "files_skipped",
            "total_time_seconds", "peak_memory_mb", "peak_cpu_percent", 
            "avg_memory_mb", "files_per_second"
        ]
        
        for key in required_keys:
            assert key in summary
