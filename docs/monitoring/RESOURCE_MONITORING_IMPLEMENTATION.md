# Resource Monitoring Implementation Summary

## Overview

Resource monitoring enables real-time tracking of CPU, memory, disk I/O, network, and GPU utilisation during ingestion, query processing, and consistency graph building.

## Implementation Details

### 1. Document Ingestion Module (`scripts/ingest/ingest.py`)

**What's Monitored:**
- Multi-threaded document ingestion pipeline
- ThreadPoolExecutor with configurable worker threads
- LLM preprocessing calls via Ollama
- ChromaDB vector storage operations

**How It Works:**
- ResourceMonitor initialised with `enable_resource_monitoring` check
- Monitoring starts before ThreadPoolExecutor begins processing documents
- Captures all worker threads simultaneously
- Monitoring stops after all documents processed (guaranteed by try/finally block)

**Configuration:**
- `ENABLE_RESOURCE_MONITORING=true` - Master switch
- `RESOURCE_MONITORING_INTERVAL=1.0` - Sample every second (adjust for long operations)
- `MONITOR_OLLAMA=true` - Track LLM inference
- `MONITOR_CHROMADB=true` - Track database operations

**Output:**
```
logs/resource_stats/resource_stats_document_ingestion_YYYYMMDD_HHMMSS.json
```

**Example Usage:**
```bash
ENABLE_RESOURCE_MONITORING=true python scripts/ingest/ingest.py
```

---

### 2. RAG Query Module (`scripts/rag/query.py`)

**What's Monitored:**
- Single query execution end-to-end
- Retrieval phase (ChromaDB vector search)
- Generation phase (Ollama LLM inference)
- Network I/O between components

**How It Works:**
- ResourceMonitor wraps the entire answer generation process
- Query string included in operation name (first 30 chars)
- Captures both retrieval and generation phases in one monitoring session
- Results include query-specific resource footprint

**Configuration:**
- `ENABLE_RESOURCE_MONITORING=true` - Master switch
- `RESOURCE_MONITORING_INTERVAL=1.0` - Sample interval
- `MONITOR_OLLAMA=true` - Track inference
- `MONITOR_CHROMADB=true` - Track retrieval

**Output:**
```
logs/resource_stats/resource_stats_rag_query_What_is_multi_factor_YYYYMMDD_HHMMSS.json
```

**Example Usage:**
```bash
ENABLE_RESOURCE_MONITORING=true python scripts/rag/query.py "What is multi-factor authentication?"
```

---

### 3. Consistency Graph Builder (`scripts/consistency_graph/build_consistency_graph.py`)

**What's Monitored:**
- LLM-based consistency checking between documents
- Parallel relationship classification via ThreadPoolExecutor
- Network I/O for embedding queries
- ChromaDB operations for document retrieval
- Community detection algorithms

**How It Works:**
- ResourceMonitor initialised just before graph building begins
- Captures all LLM inference during relationship classification
- Includes community detection and cluster labeling phases
- Monitoring stops when graph build completes or fails (guaranteed by finally block)

**Configuration:**
- `ENABLE_RESOURCE_MONITORING=true` - Master switch
- `RESOURCE_MONITORING_INTERVAL=1.0` - Sample interval (consider 2.0+ for long builds)
- `MONITOR_OLLAMA=true` - Critical: tracks LLM processing
- `MONITOR_CHROMADB=true` - Tracks retrieval phase

**Output:**
```
logs/resource_stats/resource_stats_consistency_graph_build_YYYYMMDD_HHMMSS.json
```

**Example Usage:**
```bash
ENABLE_RESOURCE_MONITORING=true python scripts/consistency_graph/build_consistency_graph.py
```

---

## Key Features

### 1. Master Switch Pattern
All modules respect `ENABLE_RESOURCE_MONITORING`:
- If `false` (default): Zero overhead, no monitoring
- If `true`: Monitoring enabled with configured intervals

### 2. Guaranteed Cleanup
Using try/finally blocks ensures resource monitoring always stops and exports data:
```python
try:
    # Execute operation
    ...
finally:
    # Always executes, even on errors
    resource_monitor.stop()
    resource_monitor.print_summary()
    resource_monitor.export_json()
```

### 3. Operator-Friendly Output
Each module prints a summary to console AND exports JSON:
```
======================================================================
Resource Usage Summary: document_ingestion
======================================================================
Duration: 125.43s

PYTHON:
  CPU:        45.2% max, 32.1% avg
  Memory:     1024.5 MB max (12.8% of system)
  VRAM:       2048.0 MB max
  Disk I/O:   Read 512.3 MB (max 25.6 MB/s)
              Write 1024.7 MB (max 50.2 MB/s)
  Network:    Sent 128.4 MB (max 5.2 MB/s)
              Recv 256.8 MB (max 10.1 MB/s)
  Threads:    8 max
  FDs:        45 max

OLLAMA:
  CPU:        89.5% max, 67.3% avg
  Memory:     4096.2 MB max (51.2% of system)
  VRAM:       8192.0 MB max
  ...
```

### 4. Per-Module Customisation
Each module can customise:
- Operation name (appears in JSON filename and reports)
- Monitoring interval (trade-off between accuracy and overhead)
- Which processes to monitor (Ollama, ChromaDB, or both)

---

## Configuration via Environment Variables

**Example `.env` file:**
```bash
# Resource Monitoring
ENABLE_RESOURCE_MONITORING=true
RESOURCE_MONITORING_INTERVAL=1.0
MONITOR_OLLAMA=true
MONITOR_CHROMADB=true
```

**Command-line Override:**
```bash
ENABLE_RESOURCE_MONITORING=true RESOURCE_MONITORING_INTERVAL=2.0 python scripts/ingest/ingest.py
```

---

## Use Cases

### 1. Capacity Planning
```bash
# Collect baseline data
ENABLE_RESOURCE_MONITORING=true python scripts/ingest/ingest.py --limit 50

# Extract peak values from JSON
python -c "
import json
data = json.load(open('logs/resource_stats/resource_stats_document_ingestion_*.json'))
print(f'Max CPU: {data[\"processes\"][\"python\"][\"cpu_percent_max\"]}%')
print(f'Max Memory: {data[\"processes\"][\"python\"][\"memory_mb_max\"]} MB')
"

# Size infrastructure: peak_value × overhead_factor × headroom
# e.g., 45% CPU × 1.4 (VM overhead) × 1.5 (headroom) ≈ 95% one vCPU reserved
```

### 2. Performance Optimisation
```bash
# Baseline run
ENABLE_RESOURCE_MONITORING=true python scripts/ingest/ingest.py

# After optimisation
ENABLE_RESOURCE_MONITORING=true python scripts/ingest/ingest.py

# Compare JSON files to measure improvement
```

### 3. Resource Alerts
```bash
# Set alerts at 80% of observed peak
observed_max_cpu = 45  # %
alert_threshold = observed_max_cpu * 0.8  # 36%
```

### 4. Bottleneck Identification
```bash
# High CPU, low disk I/O → CPU-bound (LLM preprocessing)
# Low CPU, high disk I/O, low network → Disk-bound (storage)
# High network, high CPU → Network-bound or communication overhead
```

---

## Integration Checklist

- [x] Import ResourceMonitor in module
- [x] Check configuration has resource_monitoring_* settings (all configs already have these)
- [x] Wrap main operations with ResourceMonitor context
- [x] Initialise with config values
- [x] Ensure cleanup via try/finally
- [x] Print summary to console
- [x] Export JSON for analysis

---

## Testing

### Quick Syntax Check
```bash
python -m py_compile scripts/ingest/ingest.py
python -m py_compile scripts/ingest/ingest_git.py
python -m py_compile scripts/ingest/ingest_academic.py
python -m py_compile scripts/rag/query.py
python -m py_compile scripts/consistency_graph/build_consistency_graph.py
```

### Test Monitoring Disabled (Default Behaviour)
```bash
# Default: ENABLE_RESOURCE_MONITORING=false
python scripts/ingest/ingest.py --limit 5
# Should run normally with no monitoring output
```

### Test Monitoring Enabled
```bash
ENABLE_RESOURCE_MONITORING=true python scripts/ingest/ingest.py --limit 5
# Should print resource summary and create JSON file in logs/
```

---

## Troubleshooting

### GPU/VRAM Not Detected
- Verify `nvidia-smi` is installed: `which nvidia-smi`
- Test manually: `nvidia-smi`
- See RESOURCE_MONITORING.md GPU Troubleshooting section

### Monitoring Creates Too Much Overhead
- Increase sampling interval: `RESOURCE_MONITORING_INTERVAL=2.0` or higher
- Or disable for production: `ENABLE_RESOURCE_MONITORING=false`

### JSON Files Not Created
- Check logs directory exists: `ls -la logs/resource_stats/`
- Check disk space: `df -h`
- Verify permissions: `touch logs/resource_stats/test.txt`

### Process Not Detected
- For Ollama: verify `ps aux | grep ollama`
- For ChromaDB: verify `ps aux | grep chroma`
- Check cross-boundary issues: both must run on same host

N.B. ChromsDB wont show independent stats if running within the same Python process as the Python module.

---

## Next Steps

1. **Test End-to-End:** Run each module with monitoring enabled
2. **Collect Baselines:** Run representative workloads and save JSON outputs
3. **Set Alerts:** Use observed peaks to configure resource alerts
4. **Monitor Production:** Consider periodic monitoring runs vs continuous (CPU overhead)
5. **Analyse Trends:** Track resource usage over time as data grows


