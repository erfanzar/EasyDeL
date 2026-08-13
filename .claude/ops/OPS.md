# EasyDeL Operations

Read this before treating infrastructure symptoms as code bugs. On a TPU host, pin only unrelated host-side probes to
CPU. CPU checks are not a substitute for TPU kernel, eSurge runtime, Mosaic lowering, or performance validation:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run python -c "import jax; print(jax.devices())"
```

## TPU Setup And Bad-Node Recovery

Primary setup entry point:

```bash
scripts/tpu_setup.sh --branch vnext # or any current branch
```

Verified script facts:

- `scripts/tpu_setup.sh` clones EasyDeL on every TPU host from
  `EASYDEL_REPO_URL` into `EASYDEL_SRC_DIR`.
- It installs editable workspace packages from `libs/eformer`,
  `libs/spectrax`, `libs/ejkernel`, and `libs/easydel`.
- It configures Ray with `RAY_EXECUTABLE_PATH`.
- Its health check sets `ENABLE_DISTRIBUTED_INIT=0` before importing JAX.

Triage sequence:

```bash
gcloud compute tpus tpu-vm list --project="$PROJECT_ID" --zone="$ZONE" \
  --format="table(name.basename(),state,health,acceleratorType)"
fuser /dev/vfio/0 || true
ls -l /tmp/libtpu_lockfile || true
```

If a setup failure ends with an import error, scroll up to the first remote install or clone failure. The later import
failure is often only fallout from a failed Git clone, submodule, or editable install.

Bad-node indicators include TPU runtime failures such as
`FAILED_PRECONDITION`, `Device or resource busy`, or an otherwise healthy code path failing to initialize the
accelerator on exactly one host. Confirm the target node before any destructive action:

```bash
gcloud compute tpus tpu-vm describe "$TPU_NAME" --project="$PROJECT_ID" --zone="$ZONE"
gcloud compute tpus tpu-vm list --project="$PROJECT_ID" --zone="$ZONE" \
  --flatten="networkEndpoints[]" \
  --format="table(name.basename(),networkEndpoints.ipAddress,state,health)"
```

Only delete a specific bad TPU VM when the user has approved node recovery:

```bash
gcloud compute tpus tpu-vm delete "$TPU_NAME" --project="$PROJECT_ID" --zone="$ZONE" --quiet
```

Do not delete a whole slice or restart broad infrastructure as a debugging shortcut.

## eSurge Debugging

Start from these files:

- `libs/easydel/easydel/inference/esurge/`
- `libs/easydel/easydel/inference/esurge/runners/execution_manager.py`
- `libs/easydel/easydel/inference/esurge/runners/executors/model_executor.py`
- `libs/easydel/easydel/inference/esurge/scheduler/`
- `libs/easydel/easydel/operations/kernels/ragged_page_attention.py`
- `libs/easydel/docs/esurge.rst`
- an eSurge benchmark harness

Focused tests:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/inference/esurge

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/inference/esurge/runners/test_model_executor_prepare_signature.py
```

Runtime benchmark harness:

```bash
python an eSurge benchmark harness --help
python an eSurge benchmark harness \
  --num-prompts 32 --prompt-len 1024 --output-len 256 \
  --warmups 1 --trials 1 \
  --json-out /tmp/easydel_esurge_bench.json
```

Use `--sharding-axis-dims` in `pp,dp,fsdp,ep,tp,sp` order. The benchmark constructs a no-MTP workload with
`num_speculative_tokens=0`; do not interpret it as a speculative-decoding test.

Symptom routes:

- Cache-shape or PP prepare-cache failures: inspect
  `ModelStepExecutor._kv_prepare_signature`, `prepare_cache_key` use sites, and
  `ExecutionManager._init_operations_cache_with_retry`.
- DP/KV-page sharding failures: inspect `easydel/axis.py`,
  `easydel/inference/esurge/core/dp_sharding.py`, and
  `tests/inference/esurge/core/test_dp_sharding_pages.py`.
- Throughput regressions: compare an eSurge benchmark harness JSON
  `profile_by_total_tokens` buckets, not only aggregate tokens/sec.
- Text-only serving: keep speculative/MTP disabled unless the user explicitly asks for speculative decoding.

Set `EASURGE_SYNC_INPUTS_FOR_TIMING=1` only when measuring prep-time accuracy; it adds a device round trip and can hurt
throughput.

## Disk-Pressure Cascade

Start with filesystem and cache ownership, then fix the source that filled the disk.

```bash
df -h
du -xh --max-depth=1 . | sort -h
du -xh --max-depth=1 /tmp | sort -h
uv cache prune
```

Repo-specific sources:

- `scripts/download_hf_repo_chunked_to_gcs.py` has `--staging-dir`,
  `--chunk-gb`, `--keep-staging`, and warns when `/mnt/gcs` is not mounted.
- `scripts/download_hf_large_weights_to_gcs.py` has `--cache-dir`; set it to a non-root disk or mount.
- `scripts/mount_gcsfuse.sh` is the intended helper for mounting a GCS bucket before writing under `/mnt/gcs`.

Safe cleanup candidates in this repo are generated caches:

```bash
find . -type d \( -name .pytest_cache -o -name .ruff_cache -o -name __pycache__ \) -prune -print
```

Do not delete checkpoints, downloaded model weights, GCS-mounted outputs, or user scratch directories without explicit
confirmation.
