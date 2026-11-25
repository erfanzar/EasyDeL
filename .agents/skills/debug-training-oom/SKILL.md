---
name: debug-training-oom
description: Diagnose EasyDeL training, compile-time HBM OOM, remat/checkpointing, gradient-accumulation, chunked loss, or XLA allocator failures. Use for TPU/JAX OOMs where the first failure, HLO dump, allocator label, named scope, model body, LM head, loss, optimizer, or trainer config must be attributed before changing code.
---

# Skill: Debug Training OOM

Specialization of `.agents/skills/run-research/SKILL.md`. Load `run-research` first; for TPU availability or bad-node
symptoms read `.agents/ops/OPS.md`.

## First Reads

Read the relevant surfaces before changing batch size, remat, loss chunking, or model code:

- `WORKSPACE.md` and `.agents/ops/OPS.md`
- `libs/easydel/docs/infra/tpu_training_optimizations.md` (measured TPU remat, MoE, and memory trade-offs)
- `.agents/projects/mpmd-training-memory.md` (MPMD PJRT enqueue run-ahead OOMs and grad-tree dtype promotion)
- Accumulation, placement, and data lanes:
  - `libs/easydel/easydel/trainers/execution_accumulation.py` (`gradient_accumulation_mode="execution"`)
  - `libs/easydel/easydel/trainers/batch_placement.py` (`BatchPlacedStep`, `batch_shard_group_rows`)
  - `libs/easydel/easydel/trainers/data_lanes.py` (`TrainingArguments.data_lane_local_batch`)
- Loss, remat, scan, and linear temps:
  - `libs/easydel/easydel/infra/loss_utils.py` (`LossConfig.chunk_vocab_size`, `chunk_token_size`, `chunk_block_size`)
  - `libs/easydel/easydel/infra/etils.py` (`EasyDeLGradientCheckPointers`)
  - `libs/easydel/easydel/infra/layer_scan.py` (`layer_scan_options`)
  - `libs/easydel/easydel/layers/linears/{_linear.py,_linear_quantized.py}` (`ColumnParallelLinear`)
- Focused regression tests under `libs/easydel/tests/trainers/` (`test_chunked_lm_head_trace_safety.py`,
  `test_distillation_teacher_microbatching.py`, `test_distillation_loss_math.py`,
  `test_training_utils_gradient_accumulation.py`, `test_execution_accumulation.py`, `test_batch_placement.py`).

If driven by eLarge, also load `.agents/skills/train-elarge/SKILL.md`. For fused/quantized projections load
`.agents/skills/quantization-layout/SKILL.md`; for MPMD schedules load `.agents/skills/spectrax-pipeline-runtime/SKILL.md`.

## First-Failure Rule

Capture the first real error before touching any knob:

- exact command, hardware topology, mesh (`sharding_axis_dims`), and model/checkpoint
- `total_batch_size`, `gradient_accumulation_steps`, `gradient_accumulation_mode`, sequence length, and loss chunking
- whether the failure is a **compile-time XLA allocator OOM** (buffer assignment fails before step 0) or a **runtime
  PJRT enqueue run-ahead OOM** (compiles cleanly, then hits `RESOURCE_EXHAUSTED` during execution as host async dispatch
  queues multiple weight-shaped grad trees ahead of device donate-merge — see `.agents/projects/mpmd-training-memory.md`)

## HLO And Allocator Attribution

Collect HLO dumps on target hardware (GPU: `JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). CPU checks flag syntax
only and never proves TPU/GPU HBM behavior:

```bash
env -u XLA_FLAGS \
  XLA_FLAGS="--xla_dump_to=/tmp/easydel_hlo --xla_dump_hlo_as_text" \
  ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  <training command>
```

Inspect the dumped HLO and logs for `jax.named_scope(...)` labels, `dot_general`, `ColumnParallelLinear`
(`libs/easydel/easydel/layers/linears/_linear.py`, `_linear_quantized.py`), LM-head/loss scopes, optimizer update
scopes, and the first oversized allocation.

## Diagnosis Routes

- **Scan + optimizer temp stacking**: switch `TrainingArguments.gradient_accumulation_mode="execution"`
  (`execution_accumulation.py`) to split grad accumulation and optimizer apply into separate executables (note:
  incompatible with `mpmd_scheduler`).
- **Global-batch replication on multi-host Grain lanes**: enable `TrainingArguments.data_lane_local_batch=True` with
  SPMD `BatchPlacedStep` (`data_lanes.py`, `batch_placement.py`) so hosts place local slices instead of replicating the
  global batch.
- **Model-body matmul / activation temps**: tune `EasyDeLGradientCheckPointers` (`infra/etils.py`), selective remat
  (`gradient_checkpointing="save_only_these_names"` + `gradient_checkpointing_targets`, e.g.
  `["indexer_topk", "mhc_logits", "mhc_coefficients"]`), `scan_layers`, `use_scan_mlp` / `scan_mlp_chunk_size`,
  `layer_scan_options` (`infra/layer_scan.py`), and fused projection sharding.
- **LM-head or loss temps**: set `LossConfig.chunk_vocab_size`, `chunk_token_size`, or `chunk_block_size`
  (`infra/loss_utils.py`) and verify via `test_chunked_lm_head_trace_safety.py`.
- **Teacher/student distillation temps**: enable teacher microbatching and chunked distillation loss.
- **MPMD runtime OOM (batch-insensitive, weight-shaped failing alloc)**: inspect host enqueue throttling and gradient
  dtype preservation (no silent bf16->f32 promotion on terminal scale) via `.agents/projects/mpmd-training-memory.md`
  and `.agents/skills/spectrax-pipeline-runtime/SKILL.md`.
- **Optimizer / checkpoint restore temps**: inspect optimizer state shape/sharding; route checkpointing/restore layout
  to `.agents/skills/spectrax-serialization/SKILL.md` and sharding to `.agents/skills/spectrax-sharding/SKILL.md`.
- **Dataset or packing explosion**: load `.agents/skills/build-dataset-pipeline/SKILL.md`.

## Knob Discipline

- Change one lever at a time and record before/after compile memory, peak HBM, and step time.
- **No new `os.environ` knobs**: wire mitigations through config dataclasses, ContextVars, or explicit setters.
- Do not claim a fix until the failing compile/step passes on the target accelerator; if hardware is unavailable,
  report the change as **unverified on hardware**.

## Focused Tests

Run on the accelerator (GPU: `JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`):

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest \
    libs/easydel/tests/trainers/test_chunked_lm_head_trace_safety.py \
    libs/easydel/tests/trainers/test_distillation_teacher_microbatching.py \
    libs/easydel/tests/trainers/test_distillation_loss_math.py \
    libs/easydel/tests/trainers/test_training_utils_gradient_accumulation.py \
    libs/easydel/tests/trainers/test_execution_accumulation.py \
    libs/easydel/tests/trainers/test_batch_placement.py \
    libs/easydel/tests/infra/test_gradient_checkpointing_targets.py
```
