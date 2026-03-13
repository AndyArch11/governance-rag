"""Resource monitoring for Python, Ollama, and ChromaDB processes.

Captures CPU, RAM, GPU utilisation, VRAM, IOPS, and network utilisation for
capacity planning and resource alerts. Records peak values during operation.

Features:
- Multi-process monitoring (Python, Ollama, ChromaDB)
- GPU utilisation tracking (% via nvidia-smi or ROCm)
- GPU memory tracking (VRAM via nvidia-smi or ROCm)
- Disk I/O operations (IOPS)
- Network I/O
- Peak value tracking
- JSON export for analysis

TODO: Include environment details (OS, hardware specs, Python version, package versions)
TODO: ChromaDB is not being detected.
TODO: Integrate with alerting system for threshold breaches
TODO: Record these metrics over time for trend analysis
TODO: Add option to log metrics to file in real-time (e.g., CSV) for long-running operations
TODO: CPU core-level metrics, per-process GPU%, VRAM per process, have more robust GPU detection as currently not detecting VRAM on some AMD systems
TODO: Add purge function to clear old stats files and manage disk space

Usage:
    from scripts.utils.resource_monitor import ResourceMonitor

    with ResourceMonitor(operation_name="ingestion") as monitor:
        # Your code here
        ingest_documents()

    # Access results
    print(monitor.get_summary())
    monitor.export_json("resource_stats.json")
"""

import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from scripts.utils.config import BaseConfig
from scripts.utils.logger import get_logger


class ResourceMonitor:
    """Monitor resource utilisation for Python, Ollama, and ChromaDB processes.

    Tracks CPU, RAM, GPU utilisation %, VRAM, IOPS, and network metrics.
    Records peak values for capacity planning and alerting.

    Attributes:
        operation_name: Name of operation being monitored
        interval: Sampling interval in seconds
        enabled: Whether monitoring is active
        stats: Collected statistics by process
    """

    def __init__(
        self,
        operation_name: str = "operation",
        interval: float = 1.0,
        enabled: bool = True,
        monitor_ollama: bool = True,
        monitor_chromadb: bool = True,
        output_dir: Optional[Path] = None,
    ):
        """Initialise resource monitor.

        Args:
            operation_name: Name of operation being monitored
            interval: Sampling interval in seconds (default: 1.0)
            enabled: Whether to enable monitoring (default: True)
            monitor_ollama: Track Ollama process (default: True)
            monitor_chromadb: Track ChromaDB processes (default: True)
            output_dir: Directory for output files (default: logs/)
        """
        self.operation_name = operation_name
        self.interval = interval
        self.enabled = enabled
        self.monitor_ollama = monitor_ollama
        self.monitor_chromadb = monitor_chromadb

        # Output directory (prefer central config for consistency)
        if output_dir is None:
            try:
                base_logs_dir = BaseConfig().logs_dir
            except Exception:
                project_root = Path(__file__).parent.parent.parent
                base_logs_dir = project_root / "logs"
            # Use subdirectory for resource stats to keep logs organised
            output_dir = base_logs_dir / "resource_stats"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Current process
        self.python_process = psutil.Process(os.getpid())
        # Prime CPU counter so subsequent interval=None reads are meaningful.
        self.python_process.cpu_percent(interval=None)

        # Tracking
        self._monitoring = False
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

        # Statistics storage
        self.stats: Dict[str, Dict[str, Any]] = {
            "python": self._init_stats(),
            "ollama": self._init_stats(),
            "chromadb": self._init_stats(),
        }

        # Initial network/disk counters
        self._initial_net_io = psutil.net_io_counters()
        self._initial_disk_io = psutil.disk_io_counters()

        self.logger = get_logger()

    def _init_stats(self) -> Dict[str, Any]:
        """Initialise statistics dictionary for a process."""
        return {
            "cpu_percent": {"current": 0.0, "max": 0.0, "avg": 0.0, "samples": []},
            "memory_mb": {"current": 0.0, "max": 0.0, "avg": 0.0, "samples": []},
            "memory_percent": {"current": 0.0, "max": 0.0, "avg": 0.0, "samples": []},
            "gpu_percent": {"current": 0.0, "max": 0.0, "avg": 0.0, "samples": []},
            "vram_mb": {"current": 0.0, "max": 0.0, "avg": 0.0, "samples": []},
            "disk_read_mb": {"total": 0.0, "rate_mbps": 0.0, "max_rate": 0.0},
            "disk_write_mb": {"total": 0.0, "rate_mbps": 0.0, "max_rate": 0.0},
            "net_sent_mb": {"total": 0.0, "rate_mbps": 0.0, "max_rate": 0.0},
            "net_recv_mb": {"total": 0.0, "rate_mbps": 0.0, "max_rate": 0.0},
            "num_threads": {"current": 0, "max": 0},
            "num_fds": {"current": 0, "max": 0},  # File descriptors
        }

    def _find_ollama_process(self) -> Optional[psutil.Process]:
        """Find Ollama process if running."""
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                name = proc.info["name"]
                if name and "ollama" in name.lower():
                    return psutil.Process(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None

    def _find_chromadb_processes(self) -> List[psutil.Process]:
        """Find ChromaDB-related processes."""
        processes = []
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline")
                if cmdline and any("chroma" in str(arg).lower() for arg in cmdline):
                    processes.append(psutil.Process(proc.pid))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes

    def _get_gpu_utilisation(self) -> float:
        """Get GPU utilisation percentage.

        Returns:
            GPU utilisation as percentage (0-100)
        """
        try:
            # Try nvidia-smi first
            cmd = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    # Get first GPU's utilisation
                    return float(output.split("\n")[0])
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
            ValueError,
        ):
            pass

        # Fallback: try AMD ROCm
        try:
            result = subprocess.run(
                ["rocm-smi", "--showuse"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                # Parse ROCm output for GPU usage
                for line in result.stdout.split("\n"):
                    if "GPU use" in line or "GPU%" in line:
                        # Try to extract percentage
                        parts = line.split()
                        for part in parts:
                            if "%" in part:
                                try:
                                    return float(part.replace("%", ""))
                                except ValueError:
                                    continue
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass

        return 0.0

    def _get_gpu_memory(
        self, pid: Optional[int] = None, allow_total_fallback: bool = False
    ) -> float:
        """Get GPU memory usage in MB.

        Args:
            pid: Process ID to check (None for all processes)
            allow_total_fallback: If True, when per-process VRAM is unavailable, fall back to total GPU memory.
                Use this for a single process (e.g., python) to avoid double-counting across processes.
        Returns:
            VRAM usage in MB
        """
        try:
            # Try nvidia-smi first
            if pid:
                cmd = [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ]
            else:
                cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                output = result.stdout.strip()
                if not output:
                    return 0.0

                if pid:
                    # Parse per-process output
                    for line in output.split("\n"):
                        parts = line.split(",")
                        if len(parts) == 2:
                            try:
                                if int(parts[0].strip()) == pid:
                                    return float(parts[1].strip())
                            except ValueError:
                                continue
                    if allow_total_fallback:
                        # Fallback: return total GPU memory if per-process is unavailable
                        try:
                            return float(output.split("\n")[0].split(",")[-1].strip())
                        except Exception:
                            return 0.0
                    return 0.0
                else:
                    # Total GPU memory
                    return float(output.split("\n")[0])
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
            ValueError,
        ):
            pass

        # Fallback: try AMD ROCm
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                # Parse ROCm output (format varies)
                # This is a simplified parser
                for line in result.stdout.split("\n"):
                    if "MB" in line.upper():
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "MB" in part.upper() and i > 0:
                                try:
                                    return float(parts[i - 1])
                                except ValueError:
                                    continue
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass

        return 0.0

    def _update_process_stats(self, process_name: str, proc: psutil.Process):
        """Update statistics for a specific process.

        Args:
            process_name: Name of process category (python, ollama, chromadb)
            proc: psutil.Process instance
        """
        stats = self.stats[process_name]

        try:
            # CPU (non-blocking; caller must prime the process before first call)
            cpu = proc.cpu_percent(interval=None)
            stats["cpu_percent"]["current"] = cpu
            stats["cpu_percent"]["max"] = max(stats["cpu_percent"]["max"], cpu)
            stats["cpu_percent"]["samples"].append(cpu)

            # Memory
            mem_info = proc.memory_info()
            mem_mb = mem_info.rss / (1024 * 1024)  # Convert to MB
            stats["memory_mb"]["current"] = mem_mb
            stats["memory_mb"]["max"] = max(stats["memory_mb"]["max"], mem_mb)
            stats["memory_mb"]["samples"].append(mem_mb)

            mem_percent = proc.memory_percent()
            stats["memory_percent"]["current"] = mem_percent
            stats["memory_percent"]["max"] = max(stats["memory_percent"]["max"], mem_percent)
            stats["memory_percent"]["samples"].append(mem_percent)

            # GPU utilisation (system-wide for now, as per-process GPU% is complex)
            gpu_util = self._get_gpu_utilisation()
            stats["gpu_percent"]["current"] = gpu_util
            stats["gpu_percent"]["max"] = max(stats["gpu_percent"]["max"], gpu_util)
            stats["gpu_percent"]["samples"].append(gpu_util)

            # VRAM (GPU memory)
            # Avoid double-counting VRAM: only Python process uses total fallback
            vram = self._get_gpu_memory(
                pid=proc.pid,
                allow_total_fallback=(process_name == "python"),
            )
            stats["vram_mb"]["current"] = vram
            stats["vram_mb"]["max"] = max(stats["vram_mb"]["max"], vram)
            stats["vram_mb"]["samples"].append(vram)

            # I/O (per-process, cumulative since process start)
            try:
                io_counters = proc.io_counters()
                read_mb = io_counters.read_bytes / (1024 * 1024)
                write_mb = io_counters.write_bytes / (1024 * 1024)

                # Store cumulative values (not deltas, unlike network I/O)
                # Note: These accumulate across multiple ResourceMonitor instances in same process
                stats["disk_read_mb"]["total"] = read_mb
                stats["disk_write_mb"]["total"] = write_mb

                # Calculate rates (MB/s)
                if stats["cpu_percent"]["samples"]:
                    elapsed = len(stats["cpu_percent"]["samples"]) * self.interval
                    if elapsed > 0:
                        read_rate = read_mb / elapsed
                        write_rate = write_mb / elapsed
                        stats["disk_read_mb"]["rate_mbps"] = read_rate
                        stats["disk_write_mb"]["rate_mbps"] = write_rate
                        stats["disk_read_mb"]["max_rate"] = max(
                            stats["disk_read_mb"]["max_rate"], read_rate
                        )
                        stats["disk_write_mb"]["max_rate"] = max(
                            stats["disk_write_mb"]["max_rate"], write_rate
                        )
            except (psutil.AccessDenied, AttributeError):
                pass

            # Threads
            num_threads = proc.num_threads()
            stats["num_threads"]["current"] = num_threads
            stats["num_threads"]["max"] = max(stats["num_threads"]["max"], num_threads)

            # File descriptors (Unix only)
            try:
                num_fds = proc.num_fds()
                stats["num_fds"]["current"] = num_fds
                stats["num_fds"]["max"] = max(stats["num_fds"]["max"], num_fds)
            except (AttributeError, psutil.AccessDenied):
                pass

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.debug(f"Cannot access {process_name} process: {e}")

    def _monitor_loop(self):
        """Main monitoring loop running in separate thread."""
        chroma_warned = False
        while not self._stop_event.is_set():
            try:
                # Monitor Python process
                self._update_process_stats("python", self.python_process)

                # Monitor Ollama
                if self.monitor_ollama:
                    ollama_proc = self._find_ollama_process()
                    if ollama_proc:
                        self._update_process_stats("ollama", ollama_proc)

                # Monitor ChromaDB
                if self.monitor_chromadb:
                    chroma_procs = self._find_chromadb_processes()
                    if chroma_procs:
                        chroma_warned = False
                        # Aggregate stats for all ChromaDB processes
                        for proc in chroma_procs:
                            self._update_process_stats("chromadb", proc)
                    else:
                        if not chroma_warned:
                            self.logger.info(
                                "ChromaDB process not detected; monitoring will show zeros for ChromaDB"
                            )
                            chroma_warned = True

                # Network I/O (system-wide, attribute to Python process)
                net_io = psutil.net_io_counters()
                if self._initial_net_io:
                    sent_mb = (net_io.bytes_sent - self._initial_net_io.bytes_sent) / (1024 * 1024)
                    recv_mb = (net_io.bytes_recv - self._initial_net_io.bytes_recv) / (1024 * 1024)

                    elapsed = len(self.stats["python"]["cpu_percent"]["samples"]) * self.interval
                    if elapsed > 0:
                        sent_rate = sent_mb / elapsed
                        recv_rate = recv_mb / elapsed

                        self.stats["python"]["net_sent_mb"]["total"] = sent_mb
                        self.stats["python"]["net_recv_mb"]["total"] = recv_mb
                        self.stats["python"]["net_sent_mb"]["rate_mbps"] = sent_rate
                        self.stats["python"]["net_recv_mb"]["rate_mbps"] = recv_rate
                        self.stats["python"]["net_sent_mb"]["max_rate"] = max(
                            self.stats["python"]["net_sent_mb"]["max_rate"], sent_rate
                        )
                        self.stats["python"]["net_recv_mb"]["max_rate"] = max(
                            self.stats["python"]["net_recv_mb"]["max_rate"], recv_rate
                        )

                if self._stop_event.wait(self.interval):
                    break

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                if self._stop_event.wait(self.interval):
                    break

    def _calculate_averages(self):
        """Calculate average values from samples."""
        for process_stats in self.stats.values():
            for metric in ["cpu_percent", "memory_mb", "memory_percent", "gpu_percent", "vram_mb"]:
                samples = process_stats[metric]["samples"]
                if samples:
                    process_stats[metric]["avg"] = sum(samples) / len(samples)

    def start(self):
        """Start monitoring."""
        if not self.enabled:
            self.logger.info("Resource monitoring is disabled")
            return

        if self._monitoring:
            self.logger.warning("Monitoring already started")
            return

        self._start_time = datetime.now()
        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        self.logger.info(f"Resource monitoring started for '{self.operation_name}'")

    def stop(self):
        """Stop monitoring and calculate final statistics."""
        if not self._monitoring:
            return

        self._monitoring = False
        self._stop_event.set()
        self._end_time = datetime.now()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

        self._calculate_averages()

        self.logger.info(f"Resource monitoring stopped for '{self.operation_name}'")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage.

        Returns:
            Dictionary with operation metadata and peak values
        """
        duration = None
        if self._start_time and self._end_time:
            duration = (self._end_time - self._start_time).total_seconds()

        summary = {
            "operation": self.operation_name,
            "timestamp": self._start_time.isoformat() if self._start_time else None,
            "duration_seconds": duration,
            "processes": {},
        }

        def _include_process(name: str) -> bool:
            if name == "chromadb":
                return self.monitor_chromadb
            if name == "ollama":
                return self.monitor_ollama
            return True  # always include python

        for process_name, stats in self.stats.items():
            if not _include_process(process_name):
                continue

            has_samples = bool(stats["cpu_percent"]["samples"])

            summary["processes"][process_name] = {
                "detected": has_samples,
                "cpu_percent_max": round(stats["cpu_percent"]["max"], 2) if has_samples else 0.0,
                "cpu_percent_avg": round(stats["cpu_percent"]["avg"], 2) if has_samples else 0.0,
                "memory_mb_max": round(stats["memory_mb"]["max"], 2) if has_samples else 0.0,
                "memory_mb_avg": round(stats["memory_mb"]["avg"], 2) if has_samples else 0.0,
                "memory_percent_max": (
                    round(stats["memory_percent"]["max"], 2) if has_samples else 0.0
                ),
                "gpu_percent_max": round(stats["gpu_percent"]["max"], 2) if has_samples else 0.0,
                "gpu_percent_avg": round(stats["gpu_percent"]["avg"], 2) if has_samples else 0.0,
                "vram_mb_max": round(stats["vram_mb"]["max"], 2) if has_samples else 0.0,
                "vram_mb_avg": round(stats["vram_mb"]["avg"], 2) if has_samples else 0.0,
                "disk_read_mb_total": (
                    round(stats["disk_read_mb"]["total"], 2) if has_samples else 0.0
                ),
                "disk_write_mb_total": (
                    round(stats["disk_write_mb"]["total"], 2) if has_samples else 0.0
                ),
                "disk_read_mbps_max": (
                    round(stats["disk_read_mb"]["max_rate"], 2) if has_samples else 0.0
                ),
                "disk_write_mbps_max": (
                    round(stats["disk_write_mb"]["max_rate"], 2) if has_samples else 0.0
                ),
                "network_sent_mb_total": (
                    round(stats["net_sent_mb"]["total"], 2) if has_samples else 0.0
                ),
                "network_recv_mb_total": (
                    round(stats["net_recv_mb"]["total"], 2) if has_samples else 0.0
                ),
                "network_sent_mbps_max": (
                    round(stats["net_sent_mb"]["max_rate"], 2) if has_samples else 0.0
                ),
                "network_recv_mbps_max": (
                    round(stats["net_recv_mb"]["max_rate"], 2) if has_samples else 0.0
                ),
                "threads_max": stats["num_threads"]["max"] if has_samples else 0,
                "file_descriptors_max": stats["num_fds"]["max"] if has_samples else 0,
            }

        return summary

    def export_json(self, filename: Optional[str] = None) -> Path:
        """Export statistics to JSON file.

        Args:
            filename: Output filename (default: resource_stats_{operation}_{timestamp}.json)

        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resource_stats_{self.operation_name}_{timestamp}.json"

        output_path = self.output_dir / filename

        summary = self.get_summary()

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Resource statistics exported to {output_path}")
        return output_path

    def print_summary(self):
        """Print formatted summary to console."""
        summary = self.get_summary()

        print(f"\n{'='*70}")
        print(f"Resource Usage Summary: {summary['operation']}")
        print(f"{'='*70}")

        if summary.get("duration_seconds"):
            print(f"Duration: {summary['duration_seconds']:.2f}s")
        print()

        for process_name, stats in summary.get("processes", {}).items():
            print(f"{process_name.upper()}:")
            print(
                f"  CPU:        {stats['cpu_percent_max']:.1f}% max, {stats['cpu_percent_avg']:.1f}% avg"
            )
            print(
                f"  Memory:     {stats['memory_mb_max']:.1f} MB max ({stats['memory_percent_max']:.1f}% of system)"
            )
            if stats["gpu_percent_max"] > 0:
                print(
                    f"  GPU:        {stats['gpu_percent_max']:.1f}% max, {stats['gpu_percent_avg']:.1f}% avg"
                )
            if stats["vram_mb_max"] > 0:
                print(f"  VRAM:       {stats['vram_mb_max']:.1f} MB max")
            print(
                f"  Disk I/O:   Read {stats['disk_read_mb_total']:.1f} MB (max {stats['disk_read_mbps_max']:.1f} MB/s)"
            )
            print(
                f"              Write {stats['disk_write_mb_total']:.1f} MB (max {stats['disk_write_mbps_max']:.1f} MB/s)"
            )
            if stats["network_sent_mb_total"] > 0 or stats["network_recv_mb_total"] > 0:
                print(
                    f"  Network:    Sent {stats['network_sent_mb_total']:.1f} MB (max {stats['network_sent_mbps_max']:.1f} MB/s)"
                )
                print(
                    f"              Recv {stats['network_recv_mb_total']:.1f} MB (max {stats['network_recv_mbps_max']:.1f} MB/s)"
                )
            print(f"  Threads:    {stats['threads_max']} max")
            if stats["file_descriptors_max"] > 0:
                print(f"  FDs:        {stats['file_descriptors_max']} max")
            print()

        print(f"{'='*70}\n")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def monitor_operation(operation_name: str, enabled: bool = True):
    """Decorator to monitor resource usage of a function.

    Args:
        operation_name: Name of operation for reporting
        enabled: Whether monitoring is enabled

    Example:
        @monitor_operation("document_ingestion")
        def ingest_documents():
            # Your code here
            pass
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with ResourceMonitor(operation_name=operation_name, enabled=enabled) as monitor:
                result = func(*args, **kwargs)

            monitor.print_summary()
            monitor.export_json()

            return result

        return wrapper

    return decorator
