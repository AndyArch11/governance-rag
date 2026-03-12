# Resource Monitoring Guide

Comprehensive resource utilisation tracking for Python, Ollama, and ChromaDB processes to support capacity planning and resource alerting.

## Overview

The resource monitoring system captures:
- **CPU**: Process CPU utilisation (% and absolute)
- **RAM**: Process memory usage (MB and %)
- **VRAM**: GPU memory usage (MB) via nvidia-smi or ROCm
- **Disk I/O**: Read/write operations and rates (MB/s, IOPS)
- **Network I/O**: Send/receive rates (MB/s)
- **Threads**: Thread count per process
- **File Descriptors**: Open file descriptor count

**Peak values** are tracked for capacity planning and alert threshold configuration.

All tracked processes (Python, Ollama, ChromaDB) appear in summaries even if they were not detected; a `detected` flag denotes whether samples were captured, and missing processes report zeros (with a single log line when ChromaDB is absent). To avoid VRAM double counting, only the Python process may fall back to total GPU memory when per-process VRAM is unavailable.

## Quick Start

### 1. Enable Monitoring

In `.env`:
```bash
ENABLE_RESOURCE_MONITORING=true
RESOURCE_MONITORING_INTERVAL=1.0
MONITOR_OLLAMA=true
MONITOR_CHROMADB=true
```

### 2. Context Manager Usage

```python
from scripts.utils.resource_monitor import ResourceMonitor

with ResourceMonitor(operation_name="my_operation") as monitor:
    # Your code here
    perform_operation()

# Print summary
monitor.print_summary()

# Export to JSON
monitor.export_json()
```

### 3. Decorator Usage

```python
from scripts.utils.resource_monitor import monitor_operation

@monitor_operation("document_ingestion")
def ingest_documents():
    # Your ingestion code
    pass

# Automatically monitors and reports
ingest_documents()
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_RESOURCE_MONITORING` | `false` | Enable/disable monitoring |
| `RESOURCE_MONITORING_INTERVAL` | `1.0` | Sampling interval (seconds) |
| `MONITOR_OLLAMA` | `true` | Track Ollama process |
| `MONITOR_CHROMADB` | `true` | Track ChromaDB processes |

### VM and WSL Considerations

When running in virtualised environments (VMware, VirtualBox, Hyper-V, WSL), resource monitoring requires special consideration:

#### **WSL (Windows Subsystem for Linux)**

**Disk I/O:**
- WSL2 uses a virtual disk (ext4.vhdx) which can add overhead
- File operations crossing Windows/WSL boundary are slower
- Recommend storing rag_data/ within the Linux workspace filesystem (`/workspaces/governance-rag/rag_data`) not Windows mount (`/mnt/c/...`)
- IOPS measurements include virtualisation overhead
- For accurate disk performance: add 20-30% overhead to observed values

**GPU Access:**
- **WSL2 + NVIDIA**: Supports GPU passthrough via WDDM driver
  - Requires Windows 11 or Windows 10 21H2+
  - Install nvidia-utils in WSL: `sudo apt install nvidia-utils-535`
  - Verify: `nvidia-smi` should show GPU
  - GPU utilisation and VRAM tracking work normally
- **WSL2 + AMD**: Limited ROCm support
  - May require custom kernel compilation
  - GPU metrics may not be available

**Network:**
- WSL2 uses NAT networking by default
- Network I/O metrics reflect virtualised network stack
- Localhost traffic between WSL and Windows is virtualised
- External network measurements are accurate

**Memory:**
- WSL2 can use up to 50% of total system RAM by default
- Memory measurements are from WSL perspective (not Windows host)
- Configure .wslconfig to adjust memory limits:
  ```
  [wsl2]
  memory=8GB
  processors=4
  ```

**Process Detection:**
- Python, ChromaDB processes detected normally
- Ollama running on Windows host won't be detected from WSL
  - Run Ollama inside WSL for accurate monitoring
  - Or monitor separately on Windows side

**Recommendations for WSL:**
```bash
# Store data in WSL filesystem
RAG_DATA_PATH=/workspaces/governance-rag/rag_data  # ✅ Good
RAG_DATA_PATH=/mnt/c/Users/.../rag_data  # ❌ Slow

# Increase monitoring interval to reduce overhead
RESOURCE_MONITORING_INTERVAL=2.0

# Verify GPU access before enabling monitoring
nvidia-smi  # Should show your GPU

# Adjust capacity planning for virtualisation overhead:
# - Disk IOPS: Multiply observed max × 1.3
# - Network: Add 10% overhead for NAT
# - Memory: Ensure WSL has 2× your observed peak
```

#### **Traditional VMs (VMware, VirtualBox, Hyper-V)**

**CPU:**
- Virtual CPUs (vCPUs) may not map 1:1 to physical cores
- CPU % measurements are from VM perspective
- Host CPU contention not visible to guest
- Capacity planning: Add 25-40% overhead for shared hosts

**Memory:**
- Memory measurements are from guest OS perspective
- Hypervisor memory balloon drivers may affect readings
- Memory overcommitment on host not visible
- Capacity planning: Request 1.5× observed peak from host

**Disk I/O:**
- Virtual disk adds significant overhead
- Thin provisioned disks may show variable performance
- Shared storage (SAN/NAS) adds latency
- IOPS measurements include all virtualisation layers
- Capacity planning: Multiply observed IOPS × 2-3 for physical disk

**GPU:**
- **GPU Passthrough (vGPU, SR-IOV):**
  - Full GPU access if configured
  - nvidia-smi works normally
  - Measurements accurate for allocated GPU resources
  
- **No GPU Passthrough:**
  - GPU metrics will show 0
  - LLM operations fall back to CPU
  - Expect 10-50× slower inference

- **Partial vGPU (NVIDIA GRID, vSGA):**
  - Limited GPU memory available
  - GPU% may be capped at allocated slice
  - VRAM shows virtual allocation not physical

**Network:**
- Virtual network adapters add small overhead
- Bridged networking performs better than NAT
- Network metrics accurate for VM-to-VM or VM-to-external
- Internal VM traffic may be optimised by hypervisor

**File Descriptors:**
- Guest OS limits apply, not host
- Check VM's ulimit settings independently
- Default limits often lower in VMs

**Recommendations for VMs:**
```bash
# CPU capacity planning
observed_cpu_max=45  # % from monitoring
physical_cores_needed = ceil(observed_cpu_max / 100 * 1.4)

# Memory capacity planning
observed_ram_max=4096  # MB
vm_ram_needed = observed_ram_max * 1.5

# Disk IOPS planning
observed_write_max=50  # MB/s
physical_disk_iops = observed_write_max * 2.5

# GPU requirements
# If using vGPU: Request 2× observed VRAM for headroom
# If no GPU: Ensure 4× CPU capacity for CPU inference

# Network
# Prefer bridged over NAT for performance
# Monitor both VM and host network stats if possible
```

#### **Docker Containers**

**General:**
- Containers share host kernel, less overhead than VMs
- Resource measurements are accurate within container limits
- cgroups enforce resource limits transparently

**CPU:**
- `--cpus` limit affects observed CPU%
- 100% CPU in container = 100% of allocated cores
- Host CPU% may differ from container CPU%

**Memory:**
- `-m` or `--memory` sets hard limit
- Container OOM if exceeding limit
- Monitor both container and host memory

**Disk:**
- Storage driver affects I/O performance (overlay2, aufs, etc.)
- Bind mounts to host filesystem faster than container layers
- Volume mounts perform best for databases

**GPU:**
- Requires docker runtime with GPU support (nvidia-docker)
- Full GPU access or fraction via `--gpus` flag
- Measurements work normally with proper passthrough

**Network:**0202020202
- Bridge network adds minimal overhead
- Host network eliminates overhead but reduces isolation

**Recommendations for Containers:**
```yaml
# docker-compose.yml based on monitoring
services:
  rag-app:
    deploy:
      resources:
        limits:
          cpus: '2.0'  # observed_max(45%) × 2 cores × 1.3 overhead
          memory: 8G    # observed_max(4GB) × 1.5 + 1GB buffer
        reservations:
          cpus: '1.0'
          memory: 4G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      # Use named volumes for databases (best performance)
      - rag-data:/app/rag_data
      # Bind mounts for code (acceptable for read-mostly)
      - ./scripts:/app/scripts:ro
```

#### **Key Differences Summary**

| Metric | Bare Metal | WSL2 | VM | Docker | Notes |
|--------|-----------|------|-----|--------|-------|
| **CPU %** | Accurate | Accurate | Guest view | cgroup limit | VMs: add 25-40% overhead |
| **Memory** | Accurate | WSL limit | Guest view | cgroup limit | Plan for 1.5× observed |
| **GPU %** | Accurate | Passthrough | vGPU/Passthrough | Passthrough | Requires proper setup |
| **VRAM** | Accurate | Passthrough | vGPU/Passthrough | Passthrough | Virtual allocation in vGPU |
| **Disk IOPS** | Accurate | +20-30% | +100-200% | +10-20% | Virtual disks add overhead |
| **Network** | Accurate | NAT overhead | Minimal | Minimal | Bridged > NAT |
| **Process Detection** | Easy | Easy | Easy | Requires host PID | Cross-boundary detection hard |

#### **Best Practices for Virtualised Environments**

1. **Baseline on Target Platform:**
   - Run monitoring in actual deployment environment
   - Don't extrapolate bare metal measurements to VMs
   
2. **Account for Overhead:**
   - WSL/Docker: 10-20% overhead
   - VMs: 25-50% overhead
   - Add to capacity planning calculations

3. **Storage Location Matters:**
   - Keep frequently accessed data in native filesystem
   - Avoid crossing VM/container boundaries for hot paths
   - WSL: Use `~/` not `/mnt/c/`
   - Containers: Use volumes not bind mounts for databases

4. **GPU Verification:**
   ```bash
   # Test GPU is accessible
   nvidia-smi  # Should show GPU
   
   # Test GPU compute works
   python3 -c "import torch; print(torch.cuda.is_available())"
   
   # Monitor GPU during operation
   watch -n 1 nvidia-smi
   ```

5. **Network Optimisation:**
   - WSL: Use localhost for intra-WSL communication
   - VMs: Prefer bridged networking
   - Containers: Use host network for maximum performance

6. **Resource Limits:**
   - Set limits based on monitored peaks × overhead factor
   - Leave 20-30% headroom for spikes
   - Monitor host resources alongside guest
#### **Network Configuration and Security Considerations**

Resource monitoring in virtualised environments can be affected by network configuration and security policies:

**Firewall Rules (Host and Guest):**

- **Ollama Communication:**
  - Default port: 11434 (HTTP API)
  - Required access: VM → Ollama server
  - If Ollama runs on host: Guest firewall must allow outbound to host IP
  - If Ollama runs in another VM: vNIC must allow inter-VM traffic
  - Network isolation may prevent monitoring if Ollama not detected
  
  ```bash
  # Test Ollama connectivity from VM
  curl http://localhost:11434/api/tags
  curl http://<host_ip>:11434/api/tags
  
  # Check firewall (Ubuntu/Debian)
  sudo ufw status
  sudo ufw allow out to <ollama_host> port 11434
  
  # Check firewall (RHEL/CentOS)
  sudo firewall-cmd --list-all
  sudo firewall-cmd --add-rich-rule='rule family="ipv4" destination address="<ollama_host>" port port="11434" protocol="tcp" accept'
  ```

- **ChromaDB Communication:**
  - Default port: 8000 (HTTP API)
  - Required if ChromaDB runs as separate service
  - Embedded ChromaDB (default): No network required
  - Client/server mode: VM → ChromaDB server port 8000
  
  ```bash
  # Test ChromaDB connectivity
  curl http://localhost:8000/api/v1/heartbeat
  
  # Allow ChromaDB port
  sudo ufw allow out to <chromadb_host> port 8000
  ```

**Virtual Network Adapter (vNIC) Permissions:**

- **VMware:**
  - vSwitch security policies affect monitoring
  - Promiscuous mode: Not required for resource monitoring
  - MAC address changes: Not required
  - Forged transmits: Not required
  - **Network I/O metrics unaffected by security policies**
  
- **Hyper-V:**
  - Virtual switch extensions may filter traffic
  - Port mirroring not required
  - MAC spoofing not required
  - **Network monitoring works with default security**
  
- **Cloud VMs (AWS, Azure, GCP):**
  - **AWS EC2:**
    - Security groups control ingress/egress
    - Default: All outbound allowed, no inbound
    - Resource monitoring works by default
    - Required for Ollama/ChromaDB: Add inbound rule for source VMs
    
    ```bash
    # AWS Security Group rule for Ollama
    aws ec2 authorise-security-group-ingress \
      --group-id sg-xxxxx \
      --protocol tcp \
      --port 11434 \
      --source-group sg-yyyyy  # Other VMs in this SG
    ```
  
  - **Azure VMs:**
    - Network Security Groups (NSG) control traffic
    - Default: Outbound to internet allowed, inbound denied
    - Application Security Groups for service-based rules
    
    ```bash
    # Azure NSG rule for ChromaDB
    az network nsg rule create \
      --resource-group myRG \
      --nsg-name myNSG \
      --name AllowChromaDB \
      --priority 100 \
      --destination-port-ranges 8000 \
      --direction Inbound \
      --access Allow
    ```
  
  - **GCP Compute Engine:**
    - Firewall rules applied to network or instance tags
    - Default: Outbound allowed, inbound from same VPC allowed
    - External access requires explicit firewall rules

**External Access to VM (Inbound Security):**

If you need to access monitoring results or dashboards from outside:

- **SSH Access:**
  - Required for viewing logs: `logs/resource_stats/resource_stats_*.json`
  - Restrict source IPs in firewall/security groups
  - Use SSH keys, disable password authentication
  
  ```bash
  # Restrict SSH to specific IP
  sudo ufw allow from <your_ip> to any port 22
  sudo ufw deny 22
  ```

- **Dash Dashboard (Port 8050):**
  - If running dashboard in VM, needs inbound rule
  - **Security risk**: Dashboard has no built-in authentication
  - **Recommended**: SSH tunnel instead of direct access

  ```bash
  # SSH tunnel (secure method)
  ssh -L 8050:localhost:8050 user@vm-ip
  # Then access http://localhost:8050 on your machine

  # Direct access (insecure, not recommended)
  sudo ufw allow from <your_ip> to any port 8050
  ```

- **API Access for Monitoring:**
  - If exposing resource stats via API, implement authentication
  - Use reverse proxy with TLS (nginx, traefik)
  - Restrict source IPs or require VPN access

**Egress Filtering (Outbound from VM):**

Some environments restrict outbound traffic:

- **Ollama Model Downloads:**
  - Requires: HTTPS (443) to ollama.ai, huggingface.co
  - Corporate proxies may block or require authentication
  - Workaround: Pre-download models on allowed system, copy to VM
  
  ```bash
  # Test model download
  curl -I https://ollama.ai
  
  # If blocked, pre-download and copy
  # On allowed system:
  ollama pull mistral
  ollama pull mxbai-embed-large  
  ollama list  # Note model location
  
  # Copy ~/.ollama/models to VM
  scp -r ~/.ollama/models vm-ip:~/.ollama/
  ```

- **Embedding API Calls:**
  - If using external embedding service (OpenAI, Cohere)
  - Requires: HTTPS (443) to api.openai.com, etc.
  - Check egress firewall allows API endpoints

- **Package Installation:**
  - pip/apt require internet access
  - May need proxy configuration
  
  ```bash
  # Configure proxy for pip
  export HTTP_PROXY=http://proxy:8080
  export HTTPS_PROXY=http://proxy:8080
  pip install --proxy http://proxy:8080 package-name
  
  # Configure proxy for apt
  echo 'Acquire::http::Proxy "http://proxy:8080";' | sudo tee /etc/apt/apt.conf.d/proxy.conf
  ```

**Process Detection Across Network Boundaries:**

- **Ollama on Different Host:**
  - Resource monitor cannot track remote process
  - Can only monitor network I/O to that host
  - Solution: Run resource monitor on Ollama host, aggregate results
  
  ```python
  # Monitor locally only (Ollama remote)
  monitor = ResourceMonitor(
      operation_name="query",
      monitor_ollama=False,  # Can't monitor remote
      monitor_chromadb=True  # If local
  )
  ```

- **ChromaDB on Different Host:**
  - Same limitation, cannot track remote process
  - Network I/O to ChromaDB will be captured
  - Disk I/O won't include ChromaDB's database writes

- **Cross-Host Monitoring Solution:**
  ```bash
  # Run monitor on each host
  
  # On RAG VM:
  python3 scripts/rag/query.py  # Monitors Python process
  
  # On Ollama VM:
  python3 -c "
  from scripts.utils.resource_monitor import ResourceMonitor
  import time
  with ResourceMonitor('ollama_server', monitor_ollama=True) as m:
      time.sleep(3600)  # Monitor for 1 hour
  "
  
  # Aggregate JSON results
  python3 scripts/utils/aggregate_resource_stats.py \
    logs/resource_stats_query_*.json \
    ollama_vm:logs/resource_stats_ollama_*.json
  ```

**Network Performance and Monitoring:**

- **Localhost vs Remote:**
  - Ollama on localhost: ~10-50 MB/s traffic (model context)
  - Ollama remote: Same bandwidth + network latency
  - Network I/O metrics include API traffic to Ollama/ChromaDB
  
- **NAT/Firewall Impact on Metrics:**
  - Stateful firewalls add minimal latency (<1ms)
  - NAT translation adds ~0.1-0.5ms per packet
  - Network metrics reflect end-to-end including overhead
  - Cannot separate application traffic from network overhead

- **VPN Considerations:**
  - VPN adds encryption overhead
  - Network I/O metrics include VPN traffic
  - Bandwidth may be limited by VPN endpoint
  - Latency increased by encryption and routing

**Troubleshooting Network-Related Monitoring Issues:**

```bash
# 1. Verify Ollama accessible
curl -v http://localhost:11434/api/tags
# Expected: 200 OK with model list

# 2. Check process listening
sudo netstat -tulpn | grep ollama
# Expected: ollama listening on 0.0.0.0:11434 or 127.0.0.1:11434

# 3. Test from monitoring VM (if different host)
telnet <ollama_host> 11434
# Expected: Connected

# 4. Check firewall blocking
sudo iptables -L -n -v | grep 11434
sudo ufw status numbered
# Look for DENY rules

# 5. Verify security group (cloud)
# AWS:
aws ec2 describe-security-groups --group-ids sg-xxxxx
# Azure:
az network nsg show --name myNSG --resource-group myRG
# GCP:
gcloud compute firewall-rules list --filter="targetTags:my-vm-tag"

# 6. Test network I/O monitoring
python3 -c "
from scripts.utils.resource_monitor import ResourceMonitor
import requests
with ResourceMonitor('net_test', interval=0.5, enabled=True) as m:
    # Generate network traffic
    for _ in range(10):
        requests.get('http://localhost:11434/api/tags')
    summary = m.get_summary()
    net = summary['processes']['python']
    print(f'Network sent: {net.get(\"network_sent_mb\", 0):.2f} MB')
    print(f'Network recv: {net.get(\"network_recv_mb\", 0):.2f} MB')
"
```

**Security Best Practices Summary:**

| Component | Port | Direction | Security Recommendation |
|-----------|------|-----------|------------------------|
| **Ollama API** | 11434 | VM → Ollama | Restrict to known source IPs/VMs |
| **ChromaDB API** | 8000 | VM → ChromaDB | Use authentication if exposed |
| **Dash Dashboard** | 8050 | You → VM | SSH tunnel only, never direct |
| **SSH Access** | 22 | You → VM | Key-based auth, IP whitelist |
| **Model Downloads** | 443 | VM → Internet | Allow ollama.ai, huggingface.co |
| **Embedding APIs** | 443 | VM → API | Allow api.openai.com, etc. |

**Minimal Firewall Configuration for Resource Monitoring:**

```bash
# Ubuntu/Debian with ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow from <your_ip> to any port 22  # SSH access
# If Ollama on different host:
sudo ufw allow out to <ollama_ip> port 11434
# If ChromaDB on different host:
sudo ufw allow out to <chromadb_ip> port 8000
sudo ufw enable

# This allows:
# - Resource monitoring of local processes ✅
# - Outbound API calls to Ollama/ChromaDB ✅
# - SSH access from your IP ✅
# - No inbound access from internet ✅ (secure)
```


### Programmatic Configuration

```python
monitor = ResourceMonitor(
    operation_name="custom_operation",
    interval=0.5,           # Sample every 0.5 seconds
    enabled=True,
    monitor_ollama=True,
    monitor_chromadb=True,
    output_dir=Path("logs/")
)
```

## Integration Examples

### Ingestion Module

Add to `scripts/ingest/ingest.py`:

```python
from scripts.utils.resource_monitor import ResourceMonitor
from scripts.ingest.ingest_config import get_ingest_config

def main():
    config = get_ingest_config()
    
    # Optional resource monitoring
    if config.enable_resource_monitoring:
        with ResourceMonitor(
            operation_name="document_ingestion",
            interval=config.resource_monitoring_interval,
            monitor_ollama=config.monitor_ollama,
            monitor_chromadb=config.monitor_chromadb
        ) as monitor:
            # Run ingestion
            ingest_documents(args)
        
        monitor.print_summary()
        stats_file = monitor.export_json()
        logger.info(f"Resource stats saved to {stats_file}")
    else:
        # Normal execution without monitoring
        ingest_documents(args)
```

### RAG Query Module

Add to `scripts/rag/query.py`:

```python
from scripts.utils.resource_monitor import ResourceMonitor
from scripts.rag.rag_config import RAGConfig

def main():
    config = RAGConfig()
    
    if config.enable_resource_monitoring:
        with ResourceMonitor(
            operation_name=f"rag_query_{query[:30]}",
            interval=config.resource_monitoring_interval,
            monitor_ollama=config.monitor_ollama,
            monitor_chromadb=config.monitor_chromadb
        ) as monitor:
            response = answer(query, collection, k=args.k)
        
        if args.verbose:
            monitor.print_summary()
    else:
        response = answer(query, collection, k=args.k)
```

### Graph Building Module

Add to `scripts/consistency_graph/build_consistency_graph.py`:

```python
from scripts.utils.resource_monitor import ResourceMonitor
from scripts.consistency_graph.consistency_config import ConsistencyConfig

def main():
    config = ConsistencyConfig()
    
    if config.enable_resource_monitoring:
        with ResourceMonitor(
            operation_name="consistency_graph_build",
            interval=config.resource_monitoring_interval,
            monitor_ollama=config.monitor_ollama,
            monitor_chromadb=config.monitor_chromadb
        ) as monitor:
            graph = build_consistency_graph(args)
        
        monitor.print_summary()
        monitor.export_json()
    else:
        graph = build_consistency_graph(args)
```

## Output Format

- Each process includes a `detected` flag. When `false`, all metrics remain `0.0` but the process still appears in the summary/export so you can see what was expected to run.
- If ChromaDB is enabled but not running, a single log line notes the absence and the ChromaDB entry remains with zeroed metrics.
- To avoid VRAM double counting, only the Python process falls back to total GPU memory when per-process VRAM is unavailable; Ollama and ChromaDB remain at `0.0` if per-process VRAM cannot be read.
- **Disk I/O totals are cumulative per-process** (not per-operation). If you run multiple operations sequentially in the same Python process (e.g., looping through repos), disk I/O values accumulate. Network I/O resets per ResourceMonitor instance. For per-operation disk metrics, run operations in separate processes.

### Console Summary

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
  Disk I/O:   Read 2048.5 MB (max 75.3 MB/s)
              Write 512.1 MB (max 20.5 MB/s)
  Threads:    16 max
  FDs:        128 max

CHROMADB:
  CPU:        23.4% max, 15.2% avg
  Memory:     512.8 MB max (6.4% of system)
  Disk I/O:   Read 1024.2 MB (max 35.6 MB/s)
              Write 2048.9 MB (max 68.4 MB/s)
  Threads:    4 max
  FDs:        64 max

======================================================================
```

### JSON Export

Saved to `logs/resource_stats_{operation}_{timestamp}.json`:

```json
{
  "operation": "document_ingestion",
  "timestamp": "2026-01-18T14:30:00.123456",
  "duration_seconds": 125.43,
  "processes": {
    "python": {
      "detected": true,
      "cpu_percent_max": 45.2,
      "cpu_percent_avg": 32.1,
      "memory_mb_max": 1024.5,
      "memory_mb_avg": 856.3,
      "memory_percent_max": 12.8,
      "gpu_percent_max": 35.0,
      "gpu_percent_avg": 22.4,
      "vram_mb_max": 2048.0,
      "vram_mb_avg": 1536.0,
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
    "ollama": {
      "detected": true,
      "cpu_percent_max": 89.5,
      "cpu_percent_avg": 67.3,
      "memory_mb_max": 4096.2,
      "memory_mb_avg": 3584.6,
      "memory_percent_max": 51.2,
      "gpu_percent_max": 72.5,
      "gpu_percent_avg": 54.8,
      "vram_mb_max": 8192.0,
      "vram_mb_avg": 6144.0,
      "disk_read_mb_total": 2048.5,
      "disk_write_mb_total": 512.1,
      "disk_read_mbps_max": 75.3,
      "disk_write_mbps_max": 20.5,
      "network_sent_mb_total": 0.0,
      "network_recv_mb_total": 0.0,
      "network_sent_mbps_max": 0.0,
      "network_recv_mbps_max": 0.0,
      "threads_max": 16,
      "file_descriptors_max": 128
    },
    "chromadb": {
      "detected": true,
      "cpu_percent_max": 23.4,
      "cpu_percent_avg": 15.2,
      "memory_mb_max": 512.8,
      "memory_mb_avg": 428.3,
      "memory_percent_max": 6.4,
      "gpu_percent_max": 0.0,
      "gpu_percent_avg": 0.0,
      "vram_mb_max": 0.0,
      "vram_mb_avg": 0.0,
      "disk_read_mb_total": 1024.2,
      "disk_write_mb_total": 2048.9,
      "disk_read_mbps_max": 35.6,
      "disk_write_mbps_max": 68.4,
      "network_sent_mb_total": 0.0,
      "network_recv_mb_total": 0.0,
      "network_sent_mbps_max": 0.0,
      "network_recv_mbps_max": 0.0,
      "threads_max": 4,
      "file_descriptors_max": 64
    }
  }
}
```

## Use Cases

### 1. Capacity Planning

Use peak values to size infrastructure:

```python
# After running with monitoring enabled
stats = json.load(open("logs/resource_stats_ingestion_20260118.json"))

# Calculate required resources
total_cpu = sum(p["cpu_percent_max"] for p in stats["processes"].values())
total_ram_mb = sum(p["memory_mb_max"] for p in stats["processes"].values())
total_vram_mb = sum(p["vram_mb_max"] for p in stats["processes"].values())

print(f"Required vCPUs: {int(total_cpu / 100) + 1}")
print(f"Required RAM: {int(total_ram_mb / 1024) + 1} GB")
if total_vram_mb > 0:
    print(f"Required VRAM: {int(total_vram_mb / 1024) + 1} GB")
```

**Recommended sizing with 50% headroom:**
- CPU: Peak × 1.5 vCPUs
- RAM: Peak × 1.5 GB
- VRAM: Peak × 1.2 GB (GPU operations less variable)
- Disk: Peak write rate × 1.5 for sustained IOPS

### 2. Resource Alerts

Set alerts based on observed maximums:

```python
# Load historical stats
import glob
import json

stats_files = glob.glob("logs/resource_stats/resource_stats_*.json")
max_values = {
    "cpu": 0,
    "memory_mb": 0,
    "vram_mb": 0,
    "disk_write_mbps": 0,
}

for file in stats_files:
    with open(file) as f:
        data = json.load(f)
        for process_stats in data["processes"].values():
            max_values["cpu"] = max(max_values["cpu"], process_stats["cpu_percent_max"])
            max_values["memory_mb"] = max(max_values["memory_mb"], process_stats["memory_mb_max"])
            max_values["vram_mb"] = max(max_values["vram_mb"], process_stats["vram_mb_max"])
            max_values["disk_write_mbps"] = max(max_values["disk_write_mbps"], 
                                                  process_stats["disk_write_mbps_max"])

# Set alert thresholds at 80% of observed max
print("Recommended alert thresholds (80% of observed max):")
print(f"  CPU: {max_values['cpu'] * 0.8:.1f}%")
print(f"  RAM: {max_values['memory_mb'] * 0.8:.0f} MB")
print(f"  VRAM: {max_values['vram_mb'] * 0.8:.0f} MB")
print(f"  Disk Write: {max_values['disk_write_mbps'] * 0.8:.1f} MB/s")
```

### 3. Performance Optimisation

Compare resource usage across code changes:

```bash
# Baseline
ENABLE_RESOURCE_MONITORING=true python3 scripts/ingest/ingest.py

# After optimisation
ENABLE_RESOURCE_MONITORING=true python3 scripts/ingest/ingest.py

# Compare
python3 -c "
import json
baseline = json.load(open('logs/resource_stats_ingestion_baseline.json'))
optimised = json.load(open('logs/resource_stats_ingestion_optimised.json'))

print('Performance Improvement:')
for proc in ['python', 'ollama', 'chromadb']:
    if proc in baseline['processes'] and proc in optimised['processes']:
        cpu_reduction = baseline['processes'][proc]['cpu_percent_max'] - \
                       optimised['processes'][proc]['cpu_percent_max']
        mem_reduction = baseline['processes'][proc]['memory_mb_max'] - \
                       optimised['processes'][proc]['memory_mb_max']
        print(f'{proc.upper()}:')
        print(f'  CPU: {cpu_reduction:+.1f}%')
        print(f'  RAM: {mem_reduction:+.0f} MB')
"
```

### 4. Container Resource Limits

Docker/Kubernetes configuration based on monitoring:

```yaml
# docker-compose.yml
services:
  rag-app:
    image: rag-application:latest
    deploy:
      resources:
        limits:
          # Based on monitoring: python_max(1024MB) + ollama_max(4096MB) + 50% headroom
          memory: 8G
          # Based on monitoring: 90% CPU max observed
          cpus: '2.0'
        reservations:
          # Minimum guaranteed resources
          memory: 4G
          cpus: '1.0'
          devices:
            # GPU reservation if VRAM used
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Troubleshooting

### GPU Memory Not Detected

**Issue**: VRAM shows 0.0 MB even with GPU available

**Solutions**:
1. Install `nvidia-smi` (NVIDIA GPUs):
   ```bash
   # Check if installed
   which nvidia-smi
   
   # Install if needed (Ubuntu/Debian)
   sudo apt-get install nvidia-utils-xxx
   ```

2. Install `rocm-smi` (AMD GPUs):
   ```bash
   # Check if installed
   which rocm-smi
   
   # Install ROCm tools
   # See: https://rocm.docs.amd.com/
   ```

3. Verify GPU access:
   ```bash
   nvidia-smi  # Should show GPU info
   ```

### Process Not Found

**Issue**: Ollama or ChromaDB processes show no stats

**Solutions**:
1. Verify processes are running:
   ```bash
   ps aux | grep ollama
   ps aux | grep chroma
   ```

2. Check process name matching:
   ```python
   import psutil
   for proc in psutil.process_iter(['name', 'cmdline']):
       print(proc.info['name'], proc.info['cmdline'])
   ```

3. Disable specific monitoring:
   ```bash
   MONITOR_OLLAMA=false
   MONITOR_CHROMADB=false
   ```

### Permission Denied

**Issue**: Cannot access process information

**Solutions**:
1. Run with appropriate permissions
2. Check that processes are owned by current user
3. On Linux, may need to adjust `/proc` permissions

### High Monitoring Overhead

**Issue**: Resource monitoring itself uses significant resources

**Solutions**:
1. Increase sampling interval:
   ```bash
   RESOURCE_MONITORING_INTERVAL=5.0  # Sample every 5 seconds
   ```

2. Disable specific metrics:
   ```python
   monitor = ResourceMonitor(
       operation_name="low_overhead",
       interval=2.0,
       monitor_ollama=False,  # Skip if not needed
       monitor_chromadb=False
   )
   ```

## Best Practices

1. **Sample Rate**: Use 1-2 second intervals for most operations. Decrease for short operations (<10s), increase for long operations (>10 minutes).

2. **Baseline First**: Run monitoring on representative workloads to establish baselines before setting alerts.

3. **Multiple Runs**: Collect stats from multiple runs to account for variability (cold start, cache effects).

4. **Export Data**: Always export to JSON for historical analysis and trending.

5. **Production Monitoring**: Disable in production (overhead ~1-2% CPU). Use dedicated monitoring tools (Prometheus, Datadog, Dynatrace, etc) instead.

6. **Development/Testing**: Enable during development to understand resource patterns and optimise code.

## API Reference

### ResourceMonitor Class

```python
class ResourceMonitor:
    def __init__(
        operation_name: str = "operation",
        interval: float = 1.0,
        enabled: bool = True,
        monitor_ollama: bool = True,
        monitor_chromadb: bool = True,
        output_dir: Optional[Path] = None,
    )
    
    def start() -> None
    def stop() -> None
    def get_summary() -> Dict[str, Any]
    def export_json(filename: Optional[str] = None) -> Path
    def print_summary() -> None
```

### Decorator

```python
def monitor_operation(
    operation_name: str,
    enabled: bool = True
) -> Callable
```

### Context Manager

```python
with ResourceMonitor(...) as monitor:
    # Code to monitor
    pass

# Access results
monitor.get_summary()
monitor.export_json()
```

## See Also

- [examples/manual-verification/resource_monitoring_example.py](../examples/manual-verification/resource_monitoring_example.py) - Complete examples
- `.env.example` - Configuration reference
- `scripts/utils/resource_monitor.py` - Implementation
