---
name: debug-training-oom
description: Diagnose EasyDeL training, compile-time HBM OOM, remat/checkpointing, gradient-accumulation, chunked loss, or XLA allocator failures. Use for TPU/JAX OOMs where the first failure, HLO dump, allocator label, named scope, model body, LM head, loss, optimizer, or trainer config must be attributed before changing code.
---

# Skill: Debug Training OOM

This is a specialization of `.agents/skills/run-research/SKILL.md`.

Load and follow `run-research` first. For TPU availability or bad-node
symptoms, read `.agents/ops/OPS.md`.

## First Reads

Read these before changing batch size, remat, loss chunking, or model code:

- `WORKSPACE.md`
- `.agents/ops/OPS.md`
- `libs/easydel/easydel/trainers/`
- `libs/easydel/easydel/layers/`
- `libs/easydel/tests/trainers/test_chunked_lm_head_trace_safety.py`
- `libs/easydel/tests/trainers/test_distillation_teacher_microbatching.py`
- `libs/easydel/tests/trainers/test_distillation_loss_math.py`
- `libs/easydel/tests/trainers/test_training_utils_gradient_accumulation.py`
- `libs/easydel/tests/trainers/test_model_loading.py`
- `libs/easydel/tests/trainers/_common.py`

If the run is driven by eLarge, also load `.agents/skills/train-elarge/SKILL.md`.
If the issue involves fused or quantized projections, load
`.agents/skills/quantization-layout/SKILL.md`.

## First-Failure Rule

Do not patch the first suspicious knob. Capture the first real error and the
allocation source:

- command and environment
- model/checkpoint
- mesh and batch settings
- sequence length
- `total_batch_size`
- `gradient_accumulation_steps`
- loss or distillation chunk settings
- first OOM or compile error, not the later import or cleanup fallout

## HLO And Allocator Attribution

When a compile-time HBM OOM depends on TPU compilation, collect dumps on the
target hardware if possible. CPU can check flag spelling, but it does not prove
TPU memory behavior.

```bash
XLA_FLAGS="--xla_dump_to=/tmp/easydel_hlo --xla_dump_hlo_as_text" \
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu \
  <training command>
```

Search the dumped HLO and logs for named scopes, `dot_general`,
`ColumnParallelLinear`, LM-head/loss scopes, optimizer update scopes, and the
first large allocation label.

Useful repo breadcrumbs:

- trainers use `jax.named_scope(...)` in several loss paths
- `ColumnParallelLinear` lives under `libs/easydel/easydel/layers/`
- gradient accumulation behavior is covered by
  `libs/easydel/tests/trainers/test_training_utils_gradient_accumulation.py`

## Diagnosis Routes

- Model-body matmul temps: inspect model layers, sharding, remat/checkpointing,
  and fused projection layout before changing loss code.
- LM-head or loss temps: inspect chunked LM-head and loss math tests.
- Teacher/student distillation temps: inspect teacher microbatching and
  distillation loss tests.
- Optimizer temps: inspect optimizer state shape and sharding, then load
  `.agents/skills/eformer-checkpoint-sharding/SKILL.md` if serialization or
  restore layout is involved.
- Dataset or packing explosion: load
  `.agents/skills/build-dataset-pipeline/SKILL.md`.

## Knob Discipline

Change one class of knob at a time and keep the before/after compile evidence:

- `total_batch_size`
- `gradient_accumulation_steps`
- sequence length
- loss or LM-head chunk sizes
- remat/checkpointing policy
- teacher microbatching
- sharding axis dims

Do not report a fix until the affected compile or step has passed. If the TPU
is busy, say the TPU check was not run.

## Focused Tests

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/trainers/test_chunked_lm_head_trace_safety.py

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/trainers/test_distillation_teacher_microbatching.py libs/easydel/tests/trainers/test_distillation_loss_math.py

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/trainers/test_training_utils_gradient_accumulation.py
```
