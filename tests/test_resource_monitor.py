from types import SimpleNamespace

import pytest

from scripts.utils import resource_monitor as rm
from scripts.utils.resource_monitor import ResourceMonitor


def test_get_summary_includes_chromadb_when_missing(tmp_path):
    monitor = ResourceMonitor(
        operation_name="test",
        output_dir=tmp_path,
        enabled=False,
        monitor_chromadb=True,
        monitor_ollama=False,
    )

    summary = monitor.get_summary()
    chroma = summary["processes"]["chromadb"]

    assert chroma["detected"] is False

    zero_fields = [
        "cpu_percent_max",
        "cpu_percent_avg",
        "memory_mb_max",
        "memory_mb_avg",
        "memory_percent_max",
        "gpu_percent_max",
        "gpu_percent_avg",
        "vram_mb_max",
        "vram_mb_avg",
        "disk_read_mb_total",
        "disk_write_mb_total",
        "disk_read_mbps_max",
        "disk_write_mbps_max",
        "network_sent_mb_total",
        "network_recv_mb_total",
        "network_sent_mbps_max",
        "network_recv_mbps_max",
    ]
    assert all(chroma[field] == 0.0 for field in zero_fields)
    assert chroma["threads_max"] == 0
    assert chroma["file_descriptors_max"] == 0


def test_gpu_memory_prefers_per_process_then_python_fallback(monkeypatch, tmp_path):
    monitor = ResourceMonitor(output_dir=tmp_path, enabled=False)

    outputs = iter(
        [
            "1234, 256\n9999, 512",  # per-process match
            "9999, 500\n8888, 600",  # fallback target (pid missing)
            "9999, 500\n8888, 600",  # fallback target without permission
        ]
    )

    def fake_run(cmd, capture_output, text, timeout):
        return SimpleNamespace(returncode=0, stdout=next(outputs))

    monkeypatch.setattr(rm.subprocess, "run", fake_run)

    assert monitor._get_gpu_memory(pid=1234) == 256.0
    assert monitor._get_gpu_memory(pid=4321, allow_total_fallback=True) == 500.0
    assert monitor._get_gpu_memory(pid=4321, allow_total_fallback=False) == 0.0


def test_stop_halts_collection(tmp_path):
    """Test that stop() immediately halts metric collection with no extra samples recorded."""
    monitor = ResourceMonitor(
        operation_name="test_stop",
        interval=0.05,
        output_dir=tmp_path,
        monitor_ollama=False,
        monitor_chromadb=False,
    )
    monitor.start()
    import time

    time.sleep(0.15)
    monitor.stop()

    samples_after_stop = len(monitor.stats["python"]["cpu_percent"]["samples"])
    # Allow a tiny settling period; confirm no further samples accumulate
    time.sleep(0.15)
    assert len(monitor.stats["python"]["cpu_percent"]["samples"]) == samples_after_stop, (
        "Samples continued to accumulate after stop()"
    )
