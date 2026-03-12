# Resource Monitoring Quick Start

## Enable Monitoring

Set environment variable before running any module:

```bash
export ENABLE_RESOURCE_MONITORING=true
```

Or pass inline:

```bash
ENABLE_RESOURCE_MONITORING=true python3 scripts/ingest/ingest.py
```

## Configure Monitoring

Add to `.env`:

```bash
ENABLE_RESOURCE_MONITORING=true
RESOURCE_MONITORING_INTERVAL=1.0    # Sample every 1 second
MONITOR_OLLAMA=true                 # Track LLM inference
MONITOR_CHROMADB=true               # Track database operations
```

## Run With Monitoring

### Document Ingestion
```bash
ENABLE_RESOURCE_MONITORING=true python3 scripts/ingest/ingest.py --reset
# Output: logs/resource_stats_document_ingestion_*.json
```

### Code Ingestion
```bash
ENABLE_RESOURCE_MONITORING=true python3 scripts/ingest/ingest_git.py --provider bitbucket --host https://bitbucket.org --reset
# Output: logs/resource_stats_git_ingestion_*.json
```

### RAG Query
```bash
ENABLE_RESOURCE_MONITORING=true python3 scripts/rag/query.py "What is MFA?"
# Output: logs/resource_stats_rag_query_What_is_*.json
```

### Consistency Graph Building
```bash
ENABLE_RESOURCE_MONITORING=true python3 scripts/consistency_graph/build_consistency_graph.py
# Output: logs/resource_stats_consistency_graph_build_*.json
```

## Interpret Results

Console output shows peak/average for each process:

```
PYTHON:
  CPU:        45.2% max, 32.1% avg      # Process CPU utilisation
  Memory:     1024.5 MB max (12.8% system)
  VRAM:       2048.0 MB max             # GPU memory if available
  Disk I/O:   Read 512.3 MB (max 25.6 MB/s)
              Write 1024.7 MB (max 50.2 MB/s)
  Network:    Sent 128.4 MB (max 5.2 MB/s)
              Recv 256.8 MB (max 10.1 MB/s)
  Threads:    8 max
  FDs:        45 max

OLLAMA:
  CPU:        89.5% max, 67.3% avg      # LLM inference process
  Memory:     4096.2 MB max
  VRAM:       8192.0 MB max
  ...
```

## Use Cases

### Capacity Planning
```bash
# Collect data
ENABLE_RESOURCE_MONITORING=true python3 scripts/ingest/ingest.py --limit 100

# Check peak values in logs/*.json
# Example: 45% CPU max → need 1 vCPU × 1.5 headroom = 1.5 vCPUs
```

### Performance Tuning
```bash
# Before optimisation
ENABLE_RESOURCE_MONITORING=true python3 scripts/ingest/ingest.py
ls -lt logs/resource_stats_*.json | head -1

# After optimisation
ENABLE_RESOURCE_MONITORING=true python3 scripts/ingest/ingest.py
diff logs/resource_stats_document_ingestion_*.json
```

### Check for Bottlenecks
- **High CPU, low disk I/O** → CPU-bound (LLM preprocessing)
- **Low CPU, high disk I/O** → Disk I/O bound
- **High network, moderate CPU** → Network latency or small batch size

## Troubleshooting

### GPU/VRAM shows 0
- Verify GPU available: `nvidia-smi` or `rocm-smi`
- May indicate GPU not accessible to Python process
- See [RESOURCE MONITORING IMPLEMENTATION](RESOURCE_MONITORING_IMPLEMENTATION.md) for GPU setup

### No output files created
- Check logs directory: `ls -la logs/`
- Verify free disk space: `df -h`
- Check permissions: `touch logs/test.json`

### Monitoring adds significant overhead
- Increase interval: `RESOURCE_MONITORING_INTERVAL=2.0` or higher
- Or disable for production: `ENABLE_RESOURCE_MONITORING=false`

## What Gets Monitored

| Metric | Description |
|--------|-------------|
| **CPU %** | Process CPU utilisation (0-100% per cpu, > 100% indicates mulitiple processors used) |
| **Memory MB** | Process memory in megabytes |
| **Memory %** | Percentage of system RAM |
| **GPU %** | GPU utilisation (0-100%) via nvidia-smi/rocm-smi |
| **VRAM MB** | GPU memory in megabytes |
| **Disk Read** | Total bytes read + peak rate MB/s |
| **Disk Write** | Total bytes written + peak rate MB/s |
| **Network Sent** | Total MB sent + peak rate MB/s |
| **Network Recv** | Total MB received + peak rate MB/s |
| **Threads** | Peak thread count |
| **FDs** | Peak file descriptor count |

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_RESOURCE_MONITORING` | `false` | Master switch |
| `RESOURCE_MONITORING_INTERVAL` | `1.0` | Sampling interval in seconds |
| `MONITOR_OLLAMA` | `true` | Track Ollama LLM process |
| `MONITOR_CHROMADB` | `true` | Track ChromaDB processes |

## JSON Output Format

Each run produces a JSON file in `logs/resource_stats/`:

```json
{
  "operation": "document_ingestion",
  "timestamp": "2026-01-18T14:30:00.123456",
  "duration_seconds": 125.43,
  "processes": {
    "python": {
      "cpu_percent_max": 45.2,
      "cpu_percent_avg": 32.1,
      "memory_mb_max": 1024.5,
      "memory_mb_avg": 856.3,
      "memory_percent_max": 12.8,
      "vram_mb_max": 2048.0,
      "disk_read_mb_total": 512.3,
      "disk_write_mb_total": 1024.7,
      "disk_read_mbps_max": 25.6,
      "disk_write_mbps_max": 50.2,
      "network_sent_mb_total": 128.4,
      "network_recv_mb_total": 256.8,
      "network_sent_mbps_max": 5.2,
      "network_recv_mbps_max": 10.1,
      "threads_max": 8,
      "file_descriptors_max": 45
    },
    "ollama": { ... },
    "chromadb": { ... }
  }
}
```

## Pro Tips

1. **Collect Multiple Baselines**: Run monitoring 3+ times to account for variability
2. **Compare with `diff`**: Use `diff` on JSON files to see changes from optimisations
3. **Set Alerts at 80%**: Alert threshold = observed_peak × 0.8
4. **Longer Intervals for Long Runs**: `RESOURCE_MONITORING_INTERVAL=5.0` for >1 hour operations
5. **Disable in Production**: Monitoring has ~1-2% CPU overhead, use dedicated tools (Prometheus) for prod

---

**For detailed information, see [RESOURCE_MONITORING_IMPLEMENTATION.md](RESOURCE_MONITORING_IMPLEMENTATION.md)**
