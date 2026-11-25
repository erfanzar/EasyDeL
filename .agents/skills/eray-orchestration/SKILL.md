---
name: eray-orchestration
description: Work on eRay distributed Ray orchestration, TPU/GPU actor pools, chip-split swarms, Docker multislice execution, GCP queued-resource/fleet provisioning, spot preemption watcher, SSH tunnels, and the eray CLI under libs/eray. Use for RayExecutor, SlicePoolManager, SwarmConfig, plan_host_split, provision/{fleet,qr,launcher,tunnel,watcher}, or eray CLI/test work.
---

# Skill: Work On eRay Orchestration And CLI

This is a specialization of `.agents/skills/run-research/SKILL.md`. Load and follow `run-research` first.

Use this skill for work inside `libs/eray/`: Ray execution (`RayExecutor`), multi-slice TPU/GPU actor pools, host
chip-split swarms, Docker multislice runners, GCP Queued Resources / fleet reconciliation / preemption watcher /
autoscaler YAML generation / tracked SSH tunnels, and the `eray` CLI.

## Package Boundary

`libs/eray` is an independent foundation package (`pyproject.toml` `[tool.importlinter]`). It MUST NOT import
`easydel`, `spectrax`, `ejkernel`, `eformer`, or `surgecore`. Keep GCP operations on the `gcloud` CLI subprocess wrapper
(`libs/eray/eray/gcp.py`) — do not add `google-api-python-client` dependencies. Verify with `uv run lint-imports`.

## First Reads

- Package & exports: `libs/eray/pyproject.toml`, `libs/eray/eray/__init__.py`
- Execution & pools: `libs/eray/eray/execution/{executor.py,decorators.py,remote.py}`, `libs/eray/eray/pool/{base.py,slice.py,device_host.py}`, `libs/eray/eray/core/{status.py,exceptions.py,monitoring.py,cluster.py}`
- Swarms, topology & resources: `libs/eray/eray/swarm.py`, `libs/eray/eray/resources/{configs.py,topology.py,ray_resources.py,hardware.py}`
- Provisioning & fleet lifecycle: `libs/eray/eray/provision/{fleet.py,qr.py,registry.py,launcher.py,tunnel.py,watcher.py}`
- Docker & CLI: `libs/eray/eray/docker/{config.py,runner.py}`, `libs/eray/eray/cli/{main.py,tpu.py,jobs.py,hostjob.py,dashboard.py,fleet.py,qr.py,autoscale.py,utils.py}`
- Test fixtures: `libs/eray/tests/conftest.py`

## Architecture & Key Surfaces

1. **Execution (`eray/execution/`, `eray/core/`):**
   - `RayExecutor` static methods: `execute` / `execute_resumable` (single pod/slice, `pod_count == 1`),
     `autoscale_execute` / `autoscale_execute_resumable` (multi-slice, `pod_count > 1` — `execute` raises `ValueError`
     if `pod_count != 1`). Also `@device_remote`, `resolve_maybe_refs`, and `start_raylet_log_guard` / `sweep_raylet_logs`
     (guards head-node disk against raylet spam).
   - Every execution returns a typed `JobStatus`: `JobSucceeded`, `JobFailed`, `JobPreempted`, or `JobError`.
2. **Actor pools & canonical Ray resource labels (`eray/pool/`):**
   - `SlicePoolManager` schedules one `SliceActor` per TPU slice; each `SliceActor` creates a `STRICT_SPREAD` placement
     group across slice hosts and spawns a `DeviceHostActor` per host.
   - Custom Ray resource labels MUST keep exact casing across `eray tpu connect` and
     `libs/eray/eray/provision/launcher.py`: `TPU` (physical chips per host), `TPU-{fam}`, `TPU-{fam}-{size}-head`
     (exact key `SlicePoolManager` matches), `accelerator_type:TPU-{FAM}`, and `head-node`.
3. **Swarms & host chip splitting (`libs/eray/eray/swarm.py`, `libs/eray/eray/resources/topology.py`):**
   - `SwarmConfig` (TPU) / `GpuSwarmConfig` (GPU, supports fractional GPUs) + `SwarmRun` + `swarm_execute` / `@swarmed`
     / `shutdown_swarm(namespace)`.
   - `SwarmRun(fn=...)` runs a plain function as a task, or instantiates a plain class as a detached named Ray actor in
     `SwarmConfig.namespace` (survives until `shutdown_swarm` or `ray.kill`).
   - `plan_host_split` (uniform) and `plan_host_partition` (heterogeneous, e.g. `(2, 1, 1)`) are pure CPU-testable
     planners supporting two modes:
     - `isolated`: disjoint 1- or 2-chip independent runtimes (Ray assigns `TPU_VISIBLE_CHIPS` / host bounds).
     - `cooperative`: one process per chip forming one multi-process ICI runtime via `jax.distributed`; sets
       `RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS` and binds `TPU_VISIBLE_CHIPS`, `CLOUD_TPU_TASK_ID`, `TPU_PROCESS_PORT`
       in-task from Ray's chip assignment (never pin statically or Ray's chip lookup breaks).
   - **Hardware constraint (validated on v5p-8):** 2-chip sub-slices ONLY form on aligned chip pairs (`{0,1}`, `{2,3}`);
     misaligned pairs crash libtpu's `SliceBuilder`, so split runners validate and raise clearly instead.
4. **Provisioning, fleet registry, watcher & tunnels (`libs/eray/eray/provision/`):**
   - `qr.py`: `QrSpec` + `gcloud compute tpus queued-resources` lifecycle (`ACCEPTED`/`CREATING` ->
     `WAITING_FOR_RESOURCES` -> `PROVISIONING` -> `ACTIVE`, terminal `SUSPENDED`/`FAILED`). Pre-existing poll env vars:
     `ERAY_QR_POLL`, `ERAY_QR_WAIT_TIMEOUT`.
   - `registry.py`: `ClusterRegistry` / `ClusterRecord` with optimistic concurrency on GCS
     (`gcloud storage cp --if-generation-match` -> `ConflictError` retry) or local `~/.eray/clusters.json`
     (`fcntl.flock`), plus a stealable watcher lease (`LEASE_TTL_S = 120`).
   - `fleet.py`: idempotent `ensure_tpu` reconciliation; canonical ports `RAY_HEAD_PORT = 6379` and
     `RAY_DASHBOARD_PORT = 8265`.
   - `watcher.py`: off-TPU spot preemption watcher built strictly as `observe() -> plan() -> execute()`. Keep `plan()`
     pure (zero I/O, exhaustive unit-tested truth table); persist write-ahead `intent` before mutating `gcloud` calls;
     respect `WatchPolicy` recreate/restart budgets (halt into `HALTED_*` rather than looping forever).
   - `launcher.py`: generates Ray cluster-launcher YAMLs (`~/.eray/autoscale/`) with eray-canonical resource labels and
     `spot=True` flag (never hardcoded).
   - `tunnel.py`: tracks background SSH port-forwards in `~/.eray/tunnels.json` (dashboard `8265` + independent `-gcs`
     tunnel on `6379`) to reuse existing forwards instead of colliding on ports.
5. **Docker & full `eray` CLI (`libs/eray/eray/docker/`, `libs/eray/eray/cli/`):**
   - Docker: `DockerConfig`, `run_docker_multislice`, `run_docker_on_pod`, `run_docker_async`,
     `build_and_push_docker_image`.
   - CLI entry (`eray = eray.cli.main:main`):
     - `eray tpu {connect,disconnect,bounce,status,health,list,job,sync}`: `connect` supports 3 modes (local TPU-VM
       metadata auto-detect, `--tpu-name` gcloud, `--ips` direct SSH). `eray tpu bounce --yes-kill-jobs` resets wedged
       raylets. `eray tpu job` manages detached non-Ray per-host jobs under `~/.eray/hostjobs/<name>/` with `/dev/vfio/*`
       holder guards.
     - `eray qr ...`, `eray fleet ...`, `eray autoscale ...`, `eray dashboard ...`, `eray resources`.
     - `eray {run,status,logs,stop}` (`libs/eray/eray/cli/jobs.py`): `eray status` re-derives verdicts from driver logs
       so remote exceptions logged via `print_remote_raise` are not falsely reported as `SUCCEEDED`; `ENV_DENY_EXACT` /
       `ENV_DENY_PREFIXES` prevent leaking host env vars into remote jobs.

## Critical Operational & Testing Pitfalls

1. **NEVER launch TPU Ray drivers with `uv run` without disabling Ray's uv hook:**
   - Ray's automatic `uv run` runtime-env hook (`RAY_ENABLE_UV_RUN_RUNTIME_ENV`) repackages the working directory
     without `.venv`, causing Ray workers to fail importing `ray` or lose `libtpu`. On TPU VMs, invoke the venv Python
     directly (e.g., `.venv/bin/python ...`) or set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=False` (as `local_ray` does).
   - Non-interactive SSH shells (`eray tpu connect`) do not load interactive shell rc files; `_ray_bin_preamble`
     (`libs/eray/eray/cli/tpu.py`) auto-resolves the driver venv's `ray` binary first, then `PATH`, then
     `~/.local/bin/ray`.
2. **Test suite isolation (`libs/eray/tests/conftest.py`):**
   - `_isolate_tunnel_store` (autouse) redirects `tunnel.STORE_PATH` / `LOG_DIR` to `tmp_path` so tests never touch
     `~/.eray/tunnels.json`.
   - `local_ray` (module-scoped fixture) starts a fake-accelerator local Ray cluster
     (`num_cpus=8, num_gpus=4, resources={"TPU": 4}`), patches `ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False`,
     bounds `ray.get` to 30s, and subclasses `DeviceHostActor` (`HostOnlyActor`) to no-op `_kill_vfio_holders` and
     `_hacky_remove_tpu_lockfile`. Always use `local_ray` for local Ray orchestration tests instead of calling bare
     `ray.init()`.

## Routing

- TPU VM bootstrap, `scripts/tpu_setup.sh`, bad-node / libtpu lock recovery: `.agents/skills/debug-tpu-setup/SKILL.md`
- TPU-specific pod/hardware issues: `tpu-expert` agent

## Verification

`libs/eray` is pure Python / Ray orchestration with **no JAX array computation**, so all 19 test modules under
`libs/eray/tests/` are CPU-safe (Golden Rule #4):

```bash
uv run lint-imports

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/eray/tests/
```
