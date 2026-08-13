---
name: convert-checkpoint
description: Convert, verify, download, or publish EasyDeL/Hugging Face checkpoints. Use for scripts/convert_hf_to_easydel.py, batch conversion, checkpoint verification, GCS staging, tensorstore layout, fused TP portability, entropy checks, or model-card updates.
---

# Skill: Convert Or Verify Checkpoints

This is a specialization of `.claude/skills/run-research/SKILL.md`.

Load and follow `run-research` first. For disk and staging failures, read the
`Disk-Pressure Cascade` section in `.claude/ops/OPS.md`.

## First Reads

Read these before running or editing conversion flows:

- `WORKSPACE.md`
- `.claude/ops/OPS.md`
- `scripts/convert_hf_to_easydel.py`
- `scripts/convert_hf_to_easydel_batch.py`
- a checkpoint verification harness
- `scripts/download_hf_repo_chunked_to_gcs.py`
- `scripts/download_hf_large_weights_to_gcs.py`
- `scripts/update_hf_model_readmes.py`
- `libs/easydel/tests/modules/test_conversion_roundtrip.py`

If the checkpoint uses fused projections, quantized linears, or TP-portable layout, also load
`.claude/skills/quantization-layout/SKILL.md`.

## Converter Selection

Use `scripts/convert_hf_to_easydel.py` for one source. Important flags:

- `--source`
- `--out`
- `--repo-id`
- `--push-to-hub` / `--no-push-to-hub`
- `--task`
- `--convert-mode` (`sequential` or `from_pretrained`)
- `--torch-streaming-cache` (`hf_cache` or `temp`)
- `--torch-streaming-tmp-dir`
- `--tensorstore-chunk-bytes`
- `--dtype`
- `--param-dtype`
- `--sharding-axis-dims` in `dp,fsdp,ep,tp,sp` order
- `--sharding-axis-names`
- `--auto-shard-model` / `--no-auto-shard-model`
- `--cache-dir`
- `--revision`
- `--token`
- `--local-files-only`
- `--force-download`
- `--trust-remote-code`
- `--enable-hf-transfer`

Use `scripts/convert_hf_to_easydel_batch.py` for many sources. It accepts repeatable `--source`, `--models-file`,
`--out-root`, `--repo-owner`,
`--python`, `--convert-script`, `--dry-run`, `--continue-on-error`, and
`--skip-existing`.

## GCS-Source, CPU-Only Conversion (gs:// -> gs://)

Use this when the HF-format checkpoint lives on GCS, the local disk cannot hold it, and TPU chips must stay untouched (a
training job may be running). Sequential mode streams gsutil shard-by-shard: only one safetensors shard (a few GB)
touches local disk at a time, and TensorStore writes the output directly to `gs://` via ePath. Verified 2026-07-02 on
qwen3_5 27B fp32 (110 GB) with 19 GB free local disk.

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
PYTHONPATH=libs/easydel:libs/ejkernel:libs/eformer:libs/spectrax:libs/eray \
  ~/easy-venv/bin/python scripts/convert_hf_to_easydel.py \
    --source gs://<bucket>/<hf-format-checkpoint> \
    --out gs://<bucket>/checkpoints_easydel/<name> \
    --convert-mode sequential \
    --dtype fp32 --param-dtype fp32 \
    --torch-streaming-cache temp \
    --torch-streaming-tmp-dir /dev/shm/hf-shards \
    --sharding-axis-dims 1,1,1,1,1
```

Stage shards on `/dev/shm` (tmpfs) when the host has RAM to spare: shard downloads then survive root-disk pressure (e.g.
the raylet.out log-spam flare-ups on pod heads, which killed a disk-staged conversion mid-`gsutil cp`
on 2026-07-02). Only one shard lives there at a time.

Rules that make this work:

- `--convert-mode sequential` is mandatory for `gs://` sources; the
  `from_pretrained` path cannot read them (the CLI enforces this).
- Run with the venv that has torch (`~/easy-venv`), not `uv run` — the workspace `.venv` has no torch and the streaming
  reader needs it.
- `JAX_PLATFORMS=cpu` with NO fake-device `XLA_FLAGS`: one CPU device means every mesh axis resolves to 1, so
  `--sharding-axis-dims 1,1,1,1,1` gives a tp=1 layout. The on-disk TensorStore arrays are full (unsharded) either way;
  the mesh only shapes in-RAM conversion.
- Match `--dtype`/`--param-dtype` to the source safetensors dtype (read a shard header) unless a cast is explicitly
  wanted.
- The CLI stages the small metadata files (config/tokenizer/processor) from
  `gs://` to a temp dir for the HF `Auto*` loaders and rsyncs tokenizer assets to the `gs://` output; weight shards
  never fully land on disk.
- On a TPU host, keep the process CPU-pinned so it cannot grab the libtpu lock from a running pod job.

## Download And Staging

For chunked HF repo download to GCS, use
`scripts/download_hf_repo_chunked_to_gcs.py` with `--repo-id`, `--repos-file`,
`--repo-type`, `--revision`, `--token`, `--staging-dir`, `--chunk-gb`,
`--download-workers`, `--path-in-repo`, `--only-zarr`, `--include`,
`--exclude`, `--skip-existing`, `--force-download`, `--local-files-only`,
`--dry-run`, `--continue-on-error`, `--keep-staging`,
`--gsutil-parallel` / `--no-gsutil-parallel`, and `--enable-hf-transfer`.

For large weight snapshots, use `scripts/download_hf_large_weights_to_gcs.py`
with `--out-root`, `--repo-id`, `--repos-file`, `--collection`, `--revision`,
`--token`, `--cache-dir`, `--min-size-mb`, `--include`, `--exclude`,
`--include-pytorch`, `--match-repo`, `--dry-run`, `--continue-on-error`, and
`--enable-hf-transfer`.

Use `scripts/mount_gcsfuse.sh` before writing to a GCS mount path.

## Verification

Every conversion needs an artifact check, not only a completed script:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run python a checkpoint verification harness <checkpoint> \
    --tokenizer <tokenizer-or-source> --tp 1 --seq 64
```

a checkpoint verification harness also accepts `--max-real-entropy` and
`--min-repeat-acc`. It builds an `eLargeModel` state, disables MTP in config, and exits nonzero on failed checks.

For conversion code changes, run the focused roundtrip test:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/modules/test_conversion_roundtrip.py
```

## Failure Routes

- High entropy, repeated-token failure, or scrambled logits: inspect tensor names, transposes, fused layout metadata,
  `fused_param_tp`, and TP ordering.
- Missing tensorstore metadata: inspect eFormer serialization through
  `.claude/skills/eformer-checkpoint-sharding/SKILL.md`.
- Disk full or slow staging: route to `.claude/ops/OPS.md`.
- Model-card work: use `scripts/update_hf_model_readmes.py` and dry-run first.
