# eray ⚡

[![PyPI version](https://img.shields.io/pypi/v/eray?logo=pypi&color=3776ab)](https://pypi.org/project/eray/)
[![Python](https://img.shields.io/badge/python-3.11--3.13-blue)](https://www.python.org/)
[![Ray](https://img.shields.io/badge/Ray-2.54%2B-informational)](https://github.com/ray-project/ray)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)](#)

> Ray-based distributed execution, scaling, and resource management for ML workloads on CPUs, GPUs, and TPUs.

## What is eray?

**eray** is a standalone extraction of the Ray executor stack from [eFormer](https://github.com/erfanzar/eFormer) / [EasyDeL](https://github.com/erfanzar/EasyDeL). It provides everything you need to run distributed machine learning workloads on Ray-managed clusters:

- **Fault-tolerant execution** — automatic retry and resumption for preempted/failed jobs
- **Multi-slice TPU support** — coordinate workloads across TPU pod slices with MegaScale
- **GPU & CPU execution** — run on NVIDIA GPUs, Intel/AMD accelerators, or plain CPU
- **Docker integration** — build, push, and run containers on remote nodes
- **Resource pooling** — actor pool management with health checks and auto-scaling
- **Rich type system** — job status tracking, exception serialization across processes

## Installation

```bash
pip install eray
```

For TPU environments, you may also want `jax` and `jaxlib` installed separately.

## Quickstart

### Single-pod execution

```python
import eray

# Run a function on a GPU pod with automatic retries
@eray.execute(eray.GpuAcceleratorConfig(count=1, gpu_model="A100"))
def train_step(batch):
    ...
    return loss
```

### Resumable execution with preemption recovery

```python
result = eray.execute_resumable(
    remote_fn=train_fn,
    accelerator_config=eray.TpuAcceleratorConfig(tpu_version="v4-8", pod_count=1),
    max_preemption_retries=1_000_000,
)
```

### Multi-slice TPU autoscaling

```python
results = eray.autoscale_execute_resumable(
    remote_fn=distributed_train,
    accelerator_config=eray.TpuAcceleratorConfig(
        tpu_version="v4-32",
        pod_count=4,
    ),
)
```

### Docker on distributed nodes

```python
from eray import DockerConfig, run_docker_multislice

docker_config = DockerConfig(
    image="my-ml-image:latest",
    command="python train.py",
    volumes={"/data": "/data"},
)

outputs = run_docker_multislice(
    docker_config,
    accelerator_config=eray.TpuAcceleratorConfig(tpu_version="v4-32", pod_count=4),
)
```

### `device_remote` decorator

```python
@eray.device_remote(accelerator_config=eray.TpuAcceleratorConfig(tpu_version="v4-8"))
class Trainer:
    def train(self, data):
        ...

trainer = Trainer()
trainer.train.remote(data)
```

## Module Map

| Module | Description |
|--------|-------------|
| `eray.types` | Job status types, exception serialization, sentinels, coordination primitives |
| `eray.resource_manager` | `ComputeResourceConfig`, `RayResources`, hardware constants |
| `eray.pool_manager` | Actor pool management, `SlicePoolManager`, `DeviceHostActor` |
| `eray.executor` | `RayExecutor`, `execute`, `autoscale_execute`, `device_remote` |
| `eray.docker_executor` | Docker config, build/push, container execution |

## Key Types

```python
from eray import (
    # Execution
    execute, execute_resumable,
    autoscale_execute, autoscale_execute_resumable,
    device_remote, RayExecutor,

    # Resource configs
    CpuAcceleratorConfig, GpuAcceleratorConfig, TpuAcceleratorConfig,
    ComputeResourceConfig, RayResources, AcceleratorConfigType,

    # Job status
    JobSucceeded, JobFailed, JobPreempted, JobError, JobStatus, JobInfo,

    # Docker
    DockerConfig, run_docker_on_pod, run_docker_multislice,

    # Pool management
    ResourcePoolManager, SlicePoolManager, DeviceHostActor,

    # Utilities
    handle_ray_error, ExceptionInfo, StopwatchActor, RefBox, DONE,
)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ERAY_SCALE_RETRY_SLEEP` | `60` | Sleep between retry attempts when slices are unavailable |
| `ERAY_SUBPROCESS_TIMEOUT` | — | Timeout for forked subprocess execution |
| `ERAY_SAFE_GATHER` | — | Use safe gather (prune dead actors) in slice preparation |
| `ERAY_MODERATE` | — | Adjust num_hosts/num_devices discovery |
| `ERAY_KILL_VFIO` | — | Kill VFIO holders on TPU hosts |
| `ERAY_INSTALL_LSOF` | — | Attempt quiet lsof install for VFIO holder detection |
| `ERAY_HOST_HEALTH_WAIT` | `60` | Timeout for waiting on all hosts to become healthy |
| `ERAY_SCALE_POLL` | `30` | Poll interval for actor pool scaling |
| `ERAY_SCALE_ADD_TIMEOUT` | `604800` | Timeout for adding actors to pool (7 days) |

## CLI — TPU Cluster Management

`eray` ships with a CLI for connecting TPU hosts into a Ray cluster. It ports the Ray connection logic from `scripts/tpu_setup.sh` into a clean Python tool.

### Prerequisites

- `gcloud` CLI installed and authenticated
- TPU VMs in `READY` state
- `ray` installed on each TPU host

### Commands

```bash
# Connect a TPU slice into a Ray cluster
eray tpu connect -n my-tpu -p my-project -z us-central2-b

# Check cluster status (nodes, TPU resources)
eray tpu status -a 10.0.0.1:6379

# Run health check (JAX devices per host)
eray tpu health -a 10.0.0.1:6379

# Stop Ray on all hosts
eray tpu disconnect -n my-tpu -p my-project -z us-central2-b

# List TPUs in a zone
eray tpu list -p my-project -z us-central2-b
```

### How `connect` works

1. **Discovers TPU** via `gcloud compute tpus tpu-vm describe` → gets internal IPs, accelerator type, worker count
2. **Cleans existing Ray** on all hosts (kills stale processes, frees ports)
3. **Starts Ray head** on worker 0 with TPU-specific custom resources
4. **Starts Ray workers** on all other hosts, pointing at the head
5. **Waits for readiness** — polls `ray.nodes()` until all hosts register

### Resource allocation

The cluster is configured with custom Ray resources matching the TPU topology:

| Resource | Head | Workers |
|----------|------|---------|
| `TPU` | `chips_per_host` | `chips_per_host` |
| `TPU-{version}` | `chips_per_host` | `chips_per_host` |
| `TPU-{version}-{slice}-head` | `1` | — |
| `accelerator_type:TPU-{VERSION}` | `1` | `1` |
| `head-node` | `1` | — |

### JSON output

All commands support `--json` for machine-readable output:

```bash
eray tpu connect -n my-tpu -p proj -z zone --json | jq .ray_address
```

## Relationship to eFormer

`eray` was extracted from [`eformer.executor.ray`](https://github.com/erfanzar/eFormer) to provide a focused, standalone Ray execution toolkit. It has **zero dependencies on eFormer or JAX** — you can use it for any Ray-based distributed workload, ML or otherwise.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

```
Copyright 2026 The EasyDeL/eray Author Erfan Zare Chavoshi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
