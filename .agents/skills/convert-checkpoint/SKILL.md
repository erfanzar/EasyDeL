---
name: convert-checkpoint
description: Convert, verify, download, or publish EasyDeL/Hugging Face checkpoints. Use for scripts/convert_hf_to_easydel.py, batch conversion, checkpoint verification, GCS staging, tensorstore layout, fused TP portability, entropy checks, or model-card updates.
---

# Skill: Convert Or Verify Checkpoints

This is a specialization of `.agents/skills/run-research/SKILL.md`.

Load and follow `run-research` first. For disk and staging failures, read the
`Disk-Pressure Cascade` section in `.agents/ops/OPS.md`.

## First Reads

Read these before running or editing conversion flows:

- `WORKSPACE.md`
- `.agents/ops/OPS.md`
- `scripts/convert_hf_to_easydel.py`
- `scripts/convert_hf_to_easydel_batch.py`
- `scripts/verify_checkpoint.py`
- `scripts/download_hf_repo_chunked_to_gcs.py`
- `scripts/download_hf_large_weights_to_gcs.py`
- `scripts/update_hf_model_readmes.py`
- `libs/easydel/tests/modules/test_conversion_roundtrip.py`

If the checkpoint uses fused projections, quantized linears, or TP-portable
layout, also load `.agents/skills/quantization-layout/SKILL.md`.

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

Use `scripts/convert_hf_to_easydel_batch.py` for many sources. It accepts
repeatable `--source`, `--models-file`, `--out-root`, `--repo-owner`,
`--python`, `--convert-script`, `--dry-run`, `--continue-on-error`, and
`--skip-existing`.

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
  uv run python scripts/verify_checkpoint.py <checkpoint> \
    --tokenizer <tokenizer-or-source> --tp 1 --seq 64
```

`scripts/verify_checkpoint.py` also accepts `--max-real-entropy` and
`--min-repeat-acc`. It builds an `eLargeModel` state, disables MTP in config,
and exits nonzero on failed checks.

For conversion code changes, run the focused roundtrip test:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/modules/test_conversion_roundtrip.py
```

## Failure Routes

- High entropy, repeated-token failure, or scrambled logits: inspect tensor
  names, transposes, fused layout metadata, `fused_param_tp`, and TP ordering.
- Missing tensorstore metadata: inspect eFormer serialization through
  `.agents/skills/eformer-checkpoint-sharding/SKILL.md`.
- Disk full or slow staging: route to `.agents/ops/OPS.md`.
- Model-card work: use `scripts/update_hf_model_readmes.py` and dry-run first.
