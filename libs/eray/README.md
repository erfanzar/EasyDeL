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

### Host chip splitting

A TPU host is normally driven by one process owning every chip. eray can carve
a host into multiple processes with disjoint chip ownership (the same libtpu
environment mechanism torch_xla uses):

```python
import eray

# Plan only (pure, no Ray): inspect the per-split environment.
plan = eray.plan_host_split(num_chips=4, num_splits=4, mode="isolated")

# On a slice: run fn once per split on every host.
# SliceActor.run_split_remote_fn / DeviceHostActor.run_split_remote_fn
refs = ray.get(host_actor.run_split_remote_fn.remote(fn, 4, mode="isolated"))
results = ray.get(refs)  # one result per split; ERAY_SPLIT_ID identifies each
```

- `isolated` (default): each split is an independent runtime with 1 or 2 chips
  assigned disjointly by Ray — e.g. four single-chip serving replicas on a
  v4/v5p host.
- `cooperative`: one process per chip forming a single multi-process runtime
  (communicating over ICI via `jax.distributed`); chip-dependent variables
  (`TPU_VISIBLE_CHIPS`, `CLOUD_TPU_TASK_ID`, `TPU_PROCESS_PORT`) are bound
  inside each task from Ray's chip assignment.

### Swarms — heterogeneous runs with a chip plan

`eray.swarm` packs several independent workloads onto the same TPU hosts.
`SwarmConfig` mirrors `TpuAcceleratorConfig` (it takes the chip type) plus the
number of runs and an optional plan for how to split each host's chips:

```python
import eray

# Even split: four replicas, one chip each per host.
status = eray.swarm_execute(serve_replica, eray.SwarmConfig(tpu_version="v5p-8", num_runs=4))

# A plan: three different workloads, chips split (2, 1, 1).
status = eray.swarm_execute(
    [
        eray.SwarmRun(train_probe, name="train"),
        eray.SwarmRun(serve_small, name="serve"),
        eray.SwarmRun(eval_loop, name="eval"),
    ],
    eray.SwarmConfig(tpu_version="v5p-8", chip_split=(2, 1, 1)),
)
```

**GPU swarms** use `GpuSwarmConfig` — same layout knobs, but no slice pool:
Ray natively assigns disjoint GPUs (`CUDA_VISIBLE_DEVICES`) per run, any GPU
count is allowed (fractions of one GPU too), and `colocate=True` packs all
runs onto one node via a placement group:

```python
status = eray.swarm_execute(
    [
        eray.SwarmRun(train_probe, name="train", num_cores=8),   # 2 GPUs, 8 CPU cores
        eray.SwarmRun(serve_small, name="serve"),                # 1 GPU
        eray.SwarmRun(eval_loop,  name="eval"),                  # 1 GPU
    ],
    eray.GpuSwarmConfig(gpu_model="A100", gpu_split=(2, 1, 1), colocate=True),
)
```

`SwarmRun.num_cores` reserves CPU cores per run on both TPU and GPU swarms
(default: the config's `cpu_count` on GPU, the launcher default on TPU).
GPU runs see `ERAY_RUN_GPUS` instead of `ERAY_RUN_CHIPS`.

There is also a decorator form — bare `SwarmRun(...)` specs (no `fn`) bind to
the decorated function, and calling it launches the swarm:

```python
@eray.swarmed(
    eray.SwarmConfig(tpu_version="v5p-8"),
    runs=[
        eray.SwarmRun(name="lr3e4", f_kwargs={"lr": 3e-4}),
        eray.SwarmRun(name="lr1e4", f_kwargs={"lr": 1e-4}),
        eray.SwarmRun(name="lr5e5", f_kwargs={"lr": 5e-5}),
        eray.SwarmRun(name="lr1e5", f_kwargs={"lr": 1e-5}),
    ],
)
def train(lr, warmup):
    ...

status = train(warmup=100)   # call-time kwargs broadcast to every run
```

A run may also be a plain **class**: it is instantiated as a long-lived,
detached named Ray actor on its chip share, addressable from any driver:

```python
runs = [
    eray.SwarmRun(ServeReplica, name="serve-a"),   # class -> named actor
    eray.SwarmRun(ServeReplica, name="serve-b"),
    eray.SwarmRun(train_probe,  name="train"),     # function -> task
]
status = eray.swarm_execute(
    runs,
    eray.SwarmConfig(tpu_version="v5p-8", chip_split=(1, 1, 2), namespace="my-swarm"),
)

# later, from any driver:
serve = ray.get_actor("serve-a", namespace="my-swarm")
ray.get(serve.generate.remote(prompt))

eray.shutdown_swarm("my-swarm")   # kill all of the swarm's named actors
```

Each run is an isolated runtime with disjoint chips (assigned by Ray) and sees
`ERAY_RUN_ID` / `ERAY_NUM_RUNS` / `ERAY_RUN_CHIPS` / `ERAY_RUN_NAME` in its
environment. Chip counts per run must be 1, 2, or the whole host, and the plan
must not oversubscribe the host. Hardware note (validated on v5p-8): 2-chip
runs only form a sub-slice on aligned chip pairs ({0,1}, {2,3}); a misaligned
Ray assignment raises a clear error in-task instead of crashing libtpu. When
launching a Ray driver on a TPU VM, invoke the venv python directly — Ray's
`uv run` hook repackages the working dir and workers lose libtpu. With `pod_count > 1` or multi-host slices the
same layout repeats on every host (actor names get a `-s{slice}h{host}`
suffix). Class-run actors are detached: they survive the swarm call and hold
their chips until `shutdown_swarm(namespace)` or `ray.kill`.

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

    # Host chip splitting
    HostSplitPlan, plan_host_split,

    # Swarms
    SwarmConfig, GpuSwarmConfig, SwarmRun, swarm_execute, swarmed, shutdown_swarm, HostPartitionPlan, plan_host_partition,

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

Run on the TPU VM itself, no flags are needed at all — the TPU's name, type,
zone, and every worker's IP are auto-detected from the instance metadata
server, and commands targeting this machine execute locally (no SSH):

```bash
eray tpu connect      # auto-detects the current TPU
eray tpu status       # auto-detects the head address
eray tpu health
eray resources        # cluster resources: used / total / free / util
eray tpu disconnect
```

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

# Show resource usage (CPU, TPU, memory, custom resources); --per-node for a host table
eray resources -a 10.0.0.1:6379 --per-node

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

## Spot fleet management

eray manages spot TPU capacity end to end: **queued resources** are the
provisioning primitive, a **fleet registry** remembers every cluster, and a
**watcher** re-queues preempted capacity, reconnects Ray, and resubmits jobs.

```bash
# capacity (stateless): spot by default, waits in queue instead of failing
eray qr create trainer1 --type v5p-64 --wait
eray qr list

# fleet (stateful): registry + reconciliation
eray fleet init --state gs://my-bucket/eray/clusters.json   # shared registry (optional)
eray fleet add trainer1 --type v5p-64 --setup-easydel        # EasyDeL bootstrap (--branch vnext pins a ref)
eray fleet add gpubox --type v5p-8 --bootstrap-cmd '...'     # or any custom per-host bootstrap command
eray fleet add n_server_spot_m           # adopt an existing QR by name
eray fleet ensure trainer1 --wait 3600   # request capacity + connect Ray, idempotent
eray fleet status

# jobs against a registered cluster; --restartable opts into auto-resubmission
eray run -c trainer1 --restartable -- python train.py

# the watcher: detect preemption -> re-queue -> reconnect -> resubmit
eray fleet watch --resubmit              # foreground; --once for cron; --dry-run to preview
eray fleet pause / eray fleet resume     # kill switch; resume clears HALTED_* park states
```

How recovery works: the watcher (run it OFF the TPU — the head dies with the
slice) fuses three signals per tick (queued-resource state, node state, head
TCP reachability). Definitive preemption (node `PREEMPTED`, QR `SUSPENDED`)
triggers: mark degraded → delete the owned QR → create `{name}-r{gen+1}` (the
node id never changes, so `eray tpu connect -n NAME` always works) → wait
ACTIVE → bootstrap once per generation → reconnect → resubmit `--restartable`
jobs as `{id}-p{N}` with `ERAY_RESTART_COUNT`/`ERAY_PREEMPTED_FROM` env (your
trainer resumes from its own checkpoints). Budgets (4 recreates/hour, 12/day)
and quota-flavored failures park the cluster in `HALTED_*` instead of looping.
Jobs are snapshotted while healthy, and resubmission re-packages the recorded
cwd, so run the watcher on the machine you submit from.

Systemd unit example:

```ini
[Unit]
Description=eray fleet watcher
After=network-online.target

[Service]
ExecStart=/path/to/venv/bin/eray fleet watch --resubmit
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

### From a laptop (macOS)

Ray heads listen on internal VPC IPs, so a laptop outside the VPC splits the
work in two: the laptop is the control plane (everything gcloud-backed), and
one always-on box inside the VPC — any existing VM works — runs the watcher
and submits restartable jobs (resubmission re-packages that box's cwd).

Laptop, once (Python 3.11–3.13):

```bash
brew install google-cloud-sdk uv && gcloud auth login
uv tool install "eray @ git+https://github.com/erfanzar/eray"
eray fleet init --state gs://my-bucket/eray/clusters.json    # shared with the watcher box
```

Watcher box, once (same install + `fleet init`, then keep alive via systemd
or tmux):

```bash
eray fleet watch --resubmit
```

Day to day, from the laptop:

```bash
eray fleet add trainer1 --type v5p-64 --setup-easydel && eray fleet up trainer1
eray fleet status --no-probe                    # registry view; probes need VPC access
gcloud compute ssh <watcher-box> -- 'cd repo && eray run -c trainer1 --restartable -- python train.py'
eray fleet tunnel trainer1 &                    # forward the head's dashboard to 127.0.0.1:8265
eray logs <job-id> -a http://127.0.0.1:8265 -f  # live logs / browser dashboard through the tunnel
```

Preemption needs nothing from you: the watcher re-queues, reconnects, and
resubmits; `eray fleet status` shows the new generation. Non-restartable
one-off jobs can be submitted straight from the laptop through the tunnel
(`eray run -a http://127.0.0.1:8265 -- ...`).

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
