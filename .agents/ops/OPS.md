# EasyDeL Operations

Read this before treating infrastructure symptoms as code bugs.

## Accelerator Execution Rules

TPU hosts hold a single `libtpu` process lock: run one accelerator test process at a time. While a TPU job runs, only
non-computation probes (pure-Python/Rust tests, config/parser checks, or the device-listing probe below) may run on CPU;
computation tests wait for the accelerator. A CPU run of a JAX computation test is never validation — without
accelerator hardware, report the result as **unverified on hardware**.

```bash
# TPU computation test environment
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest <path>

# GPU computation test environment
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
  uv run pytest <path>

# Non-computation CPU probe only (never for JAX numerics validation)
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
  XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run python -c "import jax; print(jax.devices())"
```

## TPU Setup and Bad-Node Recovery

See also `.agents/skills/debug-tpu-setup/SKILL.md`.

Primary setup entry point:

```bash
scripts/tpu_setup.sh --branch vnext  # or any branch/commit SHA
```

Verified script facts (`scripts/tpu_setup.sh`):

- Clones or updates EasyDeL on every TPU host from `EASYDEL_REPO_URL` into `EASYDEL_SRC_DIR` (log:
  `/tmp/easydel-clone.log`).
- Installs a minimal stable Rust toolchain via `rustup` when `cargo` is missing so `libs/surgecore` builds through
  maturin (log: `/tmp/easydel-rustup.log`).
- Runs `uv sync --all-packages --active --inexact --extra tpu --extra torch --extra lm-eval --extra profile --group dev`
  installing all 6 workspace packages (`easydel`, `eformer`, `ejkernel`, `spectrax-lib`, `eray`, `surgecore`) and
  verifies profile/dev tools (`xprof`, `tb-nightly`, `xprof-nightly`, `ruff`, `pytest`, `pre-commit`, `basedpyright`,
  `import-linter`).
- Configures Ray on TPU hosts using `${REMOTE_VENV_PATH}/bin/ray` (log: `/tmp/easydel-ray-start.log`).
- Its health check sets `ENABLE_DISTRIBUTED_INIT=0` before importing JAX.

Triage sequence:

```bash
gcloud compute tpus tpu-vm list --project="$PROJECT_ID" --zone="$ZONE" \
  --format="table(name.basename(),state,health,acceleratorType)"
fuser /dev/vfio/0 || true
ls -l /tmp/libtpu_lockfile || true
```

If setup ends with an import error, scroll up to the first remote clone, `rustup`, or `uv sync` failure (check
`/tmp/easydel-clone.log`, `/tmp/easydel-rustup.log`, and `/tmp/easydel-ray-start.log`). The final import failure is
usually fallout from an earlier install or Rust build failure.

Bad-node indicators include TPU runtime errors such as `FAILED_PRECONDITION`, `Device or resource busy`, or an
otherwise healthy code path failing to initialize the accelerator on exactly one host. Confirm the target node before
any destructive action:

```bash
gcloud compute tpus tpu-vm describe "$TPU_NAME" --project="$PROJECT_ID" --zone="$ZONE"
gcloud compute tpus tpu-vm list --project="$PROJECT_ID" --zone="$ZONE" \
  --flatten="networkEndpoints[]" \
  --format="table(name.basename(),networkEndpoints.ipAddress,state,health)"
```

Only delete a specific bad TPU VM after explicit user approval:

```bash
gcloud compute tpus tpu-vm delete "$TPU_NAME" --project="$PROJECT_ID" --zone="$ZONE" --quiet
```

Do not delete a whole slice or restart broad infrastructure as a debugging shortcut.

## `eray` Cluster Triage

See also `.agents/skills/eray-orchestration/SKILL.md` and `libs/eray/README.md`.

```bash
eray tpu list -p "$PROJECT_ID" -z "$ZONE"     # list TPU VMs via gcloud
eray tpu status                                # auto-detects head or active dashboard/GCS tunnel
eray tpu health                                # per-worker device probe (run inside cluster/VPC, not over laptop tunnel)
eray qr list                                   # Queued Resources status
eray fleet status                              # add --no-probe outside the VPC
eray fleet watch --resubmit --dry-run          # preview preemption recovery actions without mutating
eray dashboard ls                              # active background dashboard/GCS port-forwards
```

## GPU Triage

```bash
nvidia-smi                                     # verify visible GPUs, free VRAM, and competing processes
uv sync --extra cuda                           # ensure CUDA JAX/ejkernel extras are installed
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
  uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
```

Keep `ENABLE_DISTRIBUTED_INIT=0` on single-host GPU probes to avoid joining stale coordinator env vars. If CUDA/Triton
kernel performance or numerics look inconsistent across runs, clear stale ejkernel autotune configs (next section)
before debugging kernel code.

## Stale eJKernel Persistent Cache

See also `.agents/skills/debug-ejkernel-ops/SKILL.md`.

eJKernel persists autotuned kernel configs per device + sharding fingerprint under `~/ejkernel-presistent-cache/`
(historical spelling in `libs/ejkernel/ejkernel/ops/config/persistent.py`; override with
`EJKERNEL_PERSISTENT_CACHE_DIR`, fallback `./.ejkernel-presistent-cache/` or `$TMPDIR/ejkernel-presistent-cache/`). A
stale cache can mask a regression or carry an outdated config across code or compiler changes. Clear it — or point
`EJKERNEL_PERSISTENT_CACHE_DIR` at a fresh temp directory — before trusting a suspicious fix or benchmark result. Note
that `EJKERNEL_PERSISTENT_CACHE_DIR` relocates autotuned configs only, not compiled-executable caches.

## eSurge Debugging

See also `.agents/skills/debug-esurge/SKILL.md`, `.agents/skills/debug-easydel-cache/SKILL.md`, and
`.agents/skills/develop-surgecore/SKILL.md`.

Start from these files:

- `libs/easydel/easydel/inference/esurge/`
- `libs/easydel/easydel/inference/esurge/runners/execution_manager.py`
- `libs/easydel/easydel/inference/esurge/runners/executors/model_executor.py`
- `libs/easydel/easydel/inference/esurge/runners/executors/batch_preparer.py`
- `libs/easydel/easydel/inference/esurge/engine/core_driver.py` (drives `surgecore`'s Rust `EngineCore` in
  `libs/surgecore/src/engine/` and `KvCore` in `libs/surgecore/src/kv/`)
- `libs/easydel/easydel/operations/kernels/ragged_page_attention.py`
- `libs/easydel/docs/esurge.rst`

Focused tests:

```bash
# Accelerator computation suite (TPU shown; substitute GPU env on CUDA)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/inference/esurge

# Pure-NumPy prepare-signature check (no JAX compute; safe on CPU)
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
  uv run pytest libs/easydel/tests/inference/esurge/runners/test_model_executor_prepare_signature.py

# surgecore Rust + host-path Python tests (CPU-safe, no JAX compute)
(cd libs/surgecore && PYO3_PYTHON="$(which python)" cargo test --locked)
(cd libs/surgecore && cargo test --locked --no-default-features)
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu uv run pytest libs/surgecore/tests
```

Benchmarking notes:

- The old root benchmark scripts were removed. Write a focused throwaway harness on the target accelerator reusing the
  pure helpers in `libs/easydel/easydel/inference/esurge/benchmarking.py` (`make_prompts`, `make_real_prompts`,
  `all_finished`, `zero_schedule_needs_update`). For single-request Qwen3.5 MTP speculative-decoding timing, see
  `libs/easydel/tests/inference/esurge/bench_specdecode.py`.
- Pass `sharding_axis_dims` in `(pp, dp, fsdp, ep, tp, sp)` order.
- Keep speculative/MTP disabled (`num_speculative_tokens=0`) for text-only serving baselines unless explicitly testing
  speculative decoding.
- Compare per-bucket step profiles grouped by compiled `(num_tokens, padded_num_reqs)` / total tokens rather than only
  aggregate tokens/sec.

Symptom routes:

- **Cache-shape or PP prepare-cache failures**: inspect `ModelStepExecutor._kv_prepare_signature`, `prepare_cache_key`
  call sites, and `ExecutionManager._init_operations_cache_with_retry`.
- **Page-table / device-sync corruption**: inspect `PageTable.cpu_version`,
  `BatchMetadataPreparer._cached_page_table_version` / `invalidate_page_table_cache()`, and the
  `EngineCoreDriver` ↔ `surgecore` (`EngineCore`/`KvCore`) boundary.
- **DP/KV-page sharding failures**: inspect `libs/easydel/easydel/axis.py`,
  `libs/easydel/easydel/inference/esurge/core/dp_sharding.py`, and
  `libs/easydel/tests/inference/esurge/core/test_dp_sharding_pages.py`.
- **Throughput regressions / recompilation storms**: compare per-bucket step profiles grouped by total tokens, pin
  compile buckets, and standardize multimodal input resolutions.
- Set `EASYDEL_SYNC_INPUTS_FOR_TIMING=1` only when measuring host prep-time accuracy; it adds a device sync round trip
  and hurts throughput.

## Disk-Pressure Cascade

Start with filesystem and cache ownership, then fix the source that filled the disk.

```bash
df -h
du -xh --max-depth=1 . | sort -h
du -xh --max-depth=1 /tmp | sort -h
uv cache prune
```

**TPU-fatal slice dumps (critical):** each TPU runtime fatal writes an ~11 GB `/tmp/slice_*.dump` file. Repeated fatal
probes quickly exhaust `/` and trigger `raylet` `file_system_monitor` / `ENOSPC` crashes that kill Ray on the host
(see `.agents/projects/moe-training-stack.md`). After any TPU fatal crash, inspect and remove stale dumps before the
next launch:

```bash
ls -lh /tmp/slice_*.dump 2>/dev/null || true
rm -f /tmp/slice_*.dump
```

Repo-specific staging/download sources:

- `scripts/download_hf_repo_chunked_to_gcs.py` accepts `--staging-dir`, `--chunk-gb`, and `--keep-staging`, and warns
  when `/mnt/gcs` is not mounted.
- `scripts/download_hf_large_weights_to_gcs.py` accepts `--cache-dir`; point it to a non-root disk or mount.
- `scripts/mount_gcsfuse.sh` mounts a GCS bucket before writing under `/mnt/gcs`.

Safe cleanup candidates in this repo are generated caches (and confirmed-stale `/tmp/slice_*.dump` files):

```bash
find . -type d \( -name .pytest_cache -o -name .ruff_cache -o -name __pycache__ \) -prune -print
```

Do not delete checkpoints, downloaded model weights, GCS-mounted outputs, or user scratch directories without explicit
confirmation.
