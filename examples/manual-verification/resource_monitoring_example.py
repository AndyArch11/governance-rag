#!/usr/bin/env python
"""Example: Resource monitoring for ingestion, query, and graph building.

Demonstrates how to use ResourceMonitor to capture CPU, RAM, VRAM, IOPS,
and network utilisation for capacity planning.

Usage:
    python examples/resource_monitoring_example.py

Output:
    - Console summary with peak values
    - JSON file in logs/ with detailed statistics
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.resource_monitor import ResourceMonitor, monitor_operation


def example_context_manager():
    """Example 1: Using context manager for manual monitoring."""
    print("\n" + "=" * 70)
    print("Example 1: Context Manager")
    print("=" * 70)

    with ResourceMonitor(
        operation_name="test_operation",
        interval=0.5,  # Sample every 0.5 seconds
        enabled=True,
        monitor_ollama=True,
        monitor_chromadb=True,
    ) as monitor:
        # Simulate some work
        import time

        print("Simulating CPU-intensive work for 5 seconds...")

        # CPU work
        for i in range(5):
            _ = [x**2 for x in range(100000)]
            time.sleep(1)
            print(f"  Step {i+1}/5 complete")

    # Monitor automatically stops and calculates stats
    monitor.print_summary()

    # Export to JSON
    json_path = monitor.export_json()
    print(f"\nStatistics exported to: {json_path}")


@monitor_operation("decorated_function_test")
def example_decorator():
    """Example 2: Using decorator for automatic monitoring."""
    import time

    print("\nSimulating I/O-intensive work for 3 seconds...")

    # Simulate some file I/O
    temp_file = Path("/tmp/test_resource_monitor.txt")
    for i in range(3):
        with open(temp_file, "w") as f:
            f.write("x" * 1000000)  # Write 1MB
        time.sleep(1)
        print(f"  Step {i+1}/3 complete")

    temp_file.unlink(missing_ok=True)


def example_manual_control():
    """Example 3: Manual start/stop control."""
    print("\n" + "=" * 70)
    print("Example 3: Manual Control")
    print("=" * 70)

    monitor = ResourceMonitor(operation_name="manual_test", interval=1.0, enabled=True)

    # Start monitoring
    monitor.start()

    print("Monitoring started...")
    import time

    time.sleep(3)
    print("Work complete")

    # Stop monitoring
    monitor.stop()

    # Get summary
    summary = monitor.get_summary()
    print(f"\nOperation: {summary['operation']}")
    print(f"Duration: {summary['duration_seconds']:.2f}s")

    for process_name, stats in summary["processes"].items():
        print(f"\n{process_name.upper()}:")
        print(f"  Peak CPU: {stats['cpu_percent_max']}%")
        print(f"  Peak Memory: {stats['memory_mb_max']} MB")
        if stats["vram_mb_max"] > 0:
            print(f"  Peak VRAM: {stats['vram_mb_max']} MB")


def example_integration_points():
    """Example 4: Integration with existing modules."""
    print("\n" + "=" * 70)
    print("Example 4: Integration with Existing Modules")
    print("=" * 70)
    print("""
To integrate resource monitoring into existing modules:

1. Add to Module:
   
   from scripts.utils.resource_monitor import ResourceMonitor
   from scripts.xxxx.config import get_config
   
   config = get_config()
   
   if config.enable_resource_monitoring:
       with ResourceMonitor(
           operation_name="module_operation",
           interval=config.resource_monitoring_interval,
           monitor_ollama=config.monitor_ollama,
           monitor_chromadb=config.monitor_chromadb
       ) as monitor:
           # Run module operation
           module_operation()
       
       monitor.print_summary()
       monitor.export_json() # Optional JSON export

2. ENABLE VIA ENVIRONMENT:
   
   # In .env file:
   ENABLE_RESOURCE_MONITORING=true
   RESOURCE_MONITORING_INTERVAL=1.0
   MONITOR_OLLAMA=true
   MONITOR_CHROMADB=true

3. INTERPRET RESULTS:
   
   The JSON output includes:
   - cpu_percent_max: Peak CPU usage (%)
   - memory_mb_max: Peak RAM usage (MB)
   - vram_mb_max: Peak GPU memory (MB, if GPU available)
   - disk_read_mbps_max: Peak disk read rate (MB/s)
   - disk_write_mbps_max: Peak disk write rate (MB/s)
   - network_sent_mbps_max: Peak network send rate (MB/s)
   - network_recv_mbps_max: Peak network receive rate (MB/s)
   
   Use these values for:
   - Capacity planning (server sizing)
   - Setting resource limits (containers, k8s)
   - Configuring alerts (monitoring systems)
   - Performance optimisation targets
""")


if __name__ == "__main__":
    print("\nResource Monitoring Examples")
    print("=" * 70)

    # Run examples
    example_context_manager()

    print("\n" + "=" * 70)
    print("Example 2: Decorator")
    print("=" * 70)
    example_decorator()

    example_manual_control()
    example_integration_points()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nCheck logs/ directory for exported JSON files")
    print("Use these metrics for capacity planning and resource alerts")
