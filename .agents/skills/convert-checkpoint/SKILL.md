---
name: convert-checkpoint
description: Convert, verify, download, or publish EasyDeL/Hugging Face checkpoints. Use for scripts/convert_hf_to_easydel.py, batch conversion, GCS staging, TensorStore/OCDBT checkpoint layout, fused TP portability, forward-pass sanity checks, or model-card updates.
---

# Skill: Convert Or Verify Checkpoints

This is a specialization of `.agents/skills/run-research/SKILL.md`.

Load and follow `run-research` first. For disk and staging failures, read the `Disk-Pressure Cascade` section in
`.agents/ops/OPS.md`.

## First Reads

Read these before running or editing conversion flows:

- `WORKSPACE.md`, `.agents/ops/OPS.md`
- `libs/easydel/easydel/scripts/convert_hf_to_easydel.py` (authoritative CLI implementation;
  `scripts/convert_hf_to_easydel.py` is a thin wrapper delegating to it)
- `scripts/convert_hf_to_easydel.py`, `scripts/convert_hf_to_easydel_batch.py`
- `libs/easydel/easydel/infra/mixins/bridge.py` (`stage_remote_metadata`, `_from_torch_pretrained`,
  `_copy_tokenizer_artifacts`, GCS shard staging)
- `libs/spectrax/spectrax/serialization/_fs.py` (remote URI `joinpath` dot normalization)
- `scripts/download_hf_repo_chunked_to_gcs.py`, `scripts/download_hf_large_weights_to_gcs.py`
- `scripts/update_hf_model_readmes.py`
- `libs/easydel/tests/modules/test_conversion_roundtrip.py`

For fused projections, quantized linears, or pre-quantized checkpoints, also load
`.agents/skills/quantization-layout/SKILL.md`. For EasyDeL checkpoint serialization/OCDBT metadata, load
`.agents/skills/spectrax-serialization/SKILL.md`.

## Environment & Converter Selection

Conversion needs PyTorch in the workspace environment. Install the workspace extra via `uv` (never bare `pip` or
machine-local venv paths):

```bash
uv sync --extra torch
```

Use `scripts/convert_hf_to_easydel.py` for one source. Key flags (`ConvertArgs` in
`libs/easydel/easydel/scripts/convert_hf_to_easydel.py`):

- `--source`, `--out`, `--repo-id`, `--push-to-hub` / `--no-push-to-hub`, `--task`
- `--convert-mode` (`sequential` or `from_pretrained`)
- `--torch-streaming-cache` (`hf_cache` or `temp`), `--torch-streaming-tmp-dir`
- `--tensorstore-chunk-bytes` (default 2 GiB)
- `--resume` (reuses parameters already recorded in `tensorstore_index.json` under `--out` and skips fetching their
  source tensors)
- `--dtype`, `--param-dtype` (`bf16|fp16|fp32`)
- `--sharding-axis-dims` in **6-axis `pp,dp,fsdp,ep,tp,sp` order** (default `1,1,-1,1,1,1`; `_parse_int_list` enforces
  `expected_len=6` and rejects 5-axis strings)
- `--sharding-axis-names` (default `pp,dp,fsdp,ep,tp,sp`), `--auto-shard-model` / `--no-auto-shard-model`
- `--cache-dir`, `--revision`, `--token`, `--local-files-only`, `--force-download`, `--trust-remote-code`,
  `--enable-hf-transfer`

For multiple sources, use `scripts/convert_hf_to_easydel_batch.py` (`--source`, `--models-file`, `--out-root`,
`--repo-owner`, `--python`, `--convert-script`, `--dry-run`, `--continue-on-error`, `--skip-existing`).

## GCS-Source, CPU-Only Conversion (`gs://` -> `gs://`)

Use CPU-pinned sequential conversion when the HF checkpoint lives on GCS, local disk cannot hold the full model, or
accelerator chips must stay free for a running job:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
  uv run python scripts/convert_hf_to_easydel.py \
    --source gs://<bucket>/<hf-format-checkpoint> \
    --out gs://<bucket>/checkpoints_easydel/<name> \
    --convert-mode sequential \
    --dtype bf16 --param-dtype bf16 \
    --torch-streaming-cache temp \
    --torch-streaming-tmp-dir /dev/shm/hf-shards \
    --sharding-axis-dims 1,1,1,1,1,1 \
    --resume
```

Key mechanics grounded in `convert_hf_to_easydel.py` and `bridge.py`:

- **Remote URI safety:** `convert_hf_to_easydel.py` wraps any `"://"` `--out` path in `ePath` (avoiding `pathlib.Path`
  mangling `gs://bucket/x` into a local `gs:/...` directory). `spectrax/serialization/_fs.py` normalizes `.` segments
  in remote `joinpath` calls.
- **Bounded GCS staging:** `bridge.py` downloads remote shards with `gcloud storage cp` (explicitly avoiding `gsutil`,
  falling back to `ePath`) and stages up to **two** shards at a time while enforcing `_STAGING_MIN_FREE_BYTES`. Stage
  temp shards on `/dev/shm` when RAM permits to stay resilient to root-disk pressure.
- **`sequential` vs `from_pretrained` on `gs://`:** `sequential` streams shards into TensorStore without holding the
  full model in RAM and supports `--resume` via `tensorstore_index.json`. `from_pretrained` also supports `gs://`
  sources (auto-switching to `torch_load_mode="streaming"` and staging remote metadata), but `sequential` is preferred
  when RAM or disk space is constrained.
- **Metadata & tokenizer artifacts:** `stage_remote_metadata` stages small JSON/tokenizer files locally for HF `Auto*`
  loaders; `convert_hf_to_easydel.py` writes processor/tokenizer assets via `save_pretrained(str(out_dir))` and
  `bridge.py` (`_copy_tokenizer_artifacts`) writes remote files via `ePath.write_bytes`.
- **6-axis CPU-only layout:** `JAX_PLATFORMS=cpu` with no fake-device `XLA_FLAGS` and `--sharding-axis-dims 1,1,1,1,1,1`
  yields a single-device conversion without taking the libtpu lock. Match `--dtype`/`--param-dtype` to the source
  weights unless casting intentionally. CPU-pinning is for conversion only — never report it as numerical validation.

## Download And Staging

- Chunked HF repo download to GCS (`scripts/download_hf_repo_chunked_to_gcs.py`): requires `--out-root`, plus
  `--repo-id`/`--repos-file`, `--repo-type`, `--revision`, `--token`, `--staging-dir`, `--chunk-gb`,
  `--download-workers`, `--path-in-repo`, `--only-zarr`, `--include`, `--exclude`, `--skip-existing`,
  `--force-download`, `--local-files-only`, `--dry-run`, `--continue-on-error`, `--keep-staging`,
  `--gsutil-parallel` / `--no-gsutil-parallel`, and `--enable-hf-transfer`.
- Large weight snapshots (`scripts/download_hf_large_weights_to_gcs.py`): `--out-root`, `--repo-id`, `--repos-file`,
  `--collection`, `--revision`, `--token`, `--cache-dir`, `--min-size-mb`, `--include`, `--exclude`,
  `--include-pytorch`, `--match-repo`, `--dry-run`, `--continue-on-error`, `--enable-hf-transfer`.
- Run `scripts/mount_gcsfuse.sh` before writing to a GCSFuse mount path.

## Verification

Every converted checkpoint requires an accelerator forward-pass sanity check (TPU shown; GPU:
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Wait for the TPU if a job holds the libtpu lock; without an
accelerator, report the checkpoint as **unverified on hardware**.

Run a short throwaway forward-pass probe on the accelerator that loads `--out` via
`AutoEasyDeLModelForCausalLM.from_pretrained` (or `eLargeModel`), runs a prompt forward pass (with MTP disabled unless
testing MTP), and checks finite logits, sensible next-token entropy, and coherent top-1/greedy continuation:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run python -c "
import jax, jax.numpy as jnp
from transformers import AutoTokenizer
import easydel as ed

ckpt = '<checkpoint>'
tok = AutoTokenizer.from_pretrained(ckpt)
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(ckpt, sharding_axis_dims=(1, 1, 1, 1, 1, 1))
ids = tok('The capital of France is', return_tensors='np')['input_ids']
logits = model(input_ids=jnp.asarray(ids)).logits[0, -1].astype(jnp.float32)
assert bool(jnp.all(jnp.isfinite(logits))), 'Non-finite logits'
probs = jax.nn.softmax(logits)
entropy = float(-jnp.sum(probs * jnp.log(jnp.clip(probs, 1e-12, 1.0))))
print('top1:', repr(tok.decode([int(jnp.argmax(logits))])), 'entropy:', round(entropy, 3))
"
```

For converter code changes, run the roundtrip and TP-portability tests on the accelerator:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/modules/test_conversion_roundtrip.py -k <family>

env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/modules/test_fused_layout_tp_portability.py
```

## Failure Routes

- High entropy, repeated-token collapse, or scrambled logits: inspect tensor names, transposes, `reform_param`, fused
  layout metadata (`fused_param_tp`), and TP ordering (`.agents/skills/quantization-layout/SKILL.md`).
- Missing TensorStore/OCDBT metadata or remote commit failures: `.agents/skills/spectrax-serialization/SKILL.md`
  (standalone legacy eFormer checkpoints: `.agents/skills/eformer-serialization/SKILL.md`).
- Disk full or slow staging: `.agents/ops/OPS.md`.
- Model-card updates: `scripts/update_hf_model_readmes.py` (dry-run first).
