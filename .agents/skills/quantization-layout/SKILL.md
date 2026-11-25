---
name: quantization-layout
description: Work on EasyDeL fused projection layout, TP-portable checkpoint layout, quantized linear integration, pre-quantized checkpoint adapters, QAT rule wiring, or ejkernel quantized matmul wiring. Use for fused_param_tp, FusedColumnLayout, FusedExpertLayout, canonical/runtime fused state conversion, sanitized partition specs, or EasyDeL quantization integration.
---

# Skill: Work On Quantization And Projection Layout

This is a specialization of `.agents/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Pair with:

- `.agents/skills/ejkernel-quantization/SKILL.md` for eJKernel weight-compression formats, TurboQuant kernels, and
  `quantized_matmul` tuning (and `.agents/skills/add-ejkernel-kernel/SKILL.md` for new backend kernels).
- `.agents/skills/spectrax-quantization/SKILL.md` for SpectraX `QuantProvider` / `QuantRule` / `quantize_model`
  primitives.

## First Reads

- `WORKSPACE.md`
- Fused projection & MoE layouts:
  `libs/easydel/easydel/layers/layouts/{_canonical.py,_dense.py,_moe.py,_builders.py,_runtime.py,_torch_packing.py,_reform.py,_types.py}`
- Quantization, QAT, and TurboQuant config:
  `libs/easydel/easydel/layers/quantization/{_configs.py,_quants.py,_straight_through.py,_quantized_training.py,_turboquant.py}`
- Pre-quantized checkpoint adapters:
  `libs/easydel/easydel/layers/quantization/checkpoint/{__init__.py,_adapter.py,_canonical.py,_codecs.py,_decode.py,_formats.py,_reform.py,_scheme.py}`
- Quantized linears & sharding sanitization:
  `libs/easydel/easydel/layers/linears/_linear_quantized.py`, `libs/easydel/easydel/infra/sharding.py`
- eJKernel op wrapper: `libs/ejkernel/ejkernel/modules/operations/quantized_matmul.py`

## Package Boundary

EasyDeL owns model integration, pre-quantized checkpoint adapters, fused projection layouts, partition-spec
sanitization, QAT rule wiring, eLarge builders, and quantized linears. eJKernel owns compression formats, TurboQuant
kernels, `quantized_matmul` backends, registry, autotune, and kernel benchmarks. SpectraX owns low-level module-path
QAT/PTQ rules (`spectrax.quantization`).

Only `libs/easydel` may import `spectrax`, `ejkernel`, `eformer`, `eray`, or `surgecore` — foundation packages never
import EasyDeL or each other (`uv run lint-imports`).

## Fused Layout Rules

Use the existing layout APIs — never split fused tensors with ad hoc `.reshape()` or axis assumptions (under TP the
fused axis is rank-interleaved):

- Canonical/runtime state & TP portability (`_canonical.py`): `FUSED_TP_FIELD`, `canonicalize_fused_state`,
  `runtimeize_fused_state`, `canonicalize_fused_optimizer_state`, `runtimeize_fused_optimizer_state`,
  `retp_fused_state`, `retp_fused_optimizer_state`, `read_fused_checkpoint_tp`, `write_fused_layout_marker`,
  `stamp_fused_tp_config`, `strip_fused_tp_config`, `fused_layout_param_specs`
- Dense & MoE layouts (`_dense.py`, `_moe.py`, `_builders.py`, `_types.py`): `FusedSegment`, `FusedColumnLayout`,
  `FusedExpertLayout`, `dense_gate_up_layout`, `dense_qkv_layout`, `build_fused_gate_up_projection`,
  `split_fused_gate_up_projection`, `build_fused_qkv_projection`, `split_fused_qkv_projection`

Preserve layout metadata and `fused_param_tp` so checkpoints remain portable across TP sizes.

## Quantized Linears, Pre-Quantized Checkpoints & QAT

- **Quantized linears (`_linear_quantized.py`):** inspect `ParallelLinearQuantized`, `RowParallelLinearQuantized`,
  `ColumnParallelLinearQuantized`, row-parallel quantized sharding guards, W4A4 packed-companion gating
  (`should_pack_int4_companion`, `W4A4_FUSED_PAIR_MIN_PACKED_BYTES`), `ej_quantized_matmul`, and
  `_distributed_quantized_matmul`.
- **Partition-spec sanitization:** `sanitize_partition_spec_for_shape` is defined in
  `libs/easydel/easydel/infra/sharding.py` and consumed by `_linear_quantized.py`.
- **Pre-quantized checkpoint adapters (`layers/quantization/checkpoint/`):** externally quantized checkpoints (FP8,
  compressed-tensors, AWQ/GPTQ, MXFP4, NVFP4, DeepSeek-V4 FP4) decode through the `CanonicalQuantizedWeight` /
  `QuantSpec` seam via `CheckpointQuantAdapter`, `CheckpointQuantScheme`, and `checkpoint_quant_reform_param`. Add a
  new format by registering a `CheckpointQuantAdapter` rather than creating a parallel loading path.
- **QAT wiring (`_quantized_training.py`):** `apply_quantization_rules` and `build_quantization_provider` map EasyDeL
  configs onto SpectraX's quantization engine (`.agents/skills/spectrax-quantization/SKILL.md`).
- **TurboQuant boundary:** EasyDeL's `_turboquant.py` owns `TurboQuantConfig`, `TurboQuantConstants`, and
  `_solve_lloyd_max_cached`; KV cache views live under `.agents/skills/debug-easydel-cache/SKILL.md`, while TurboQuant
  bitpacking and kernels live under `.agents/skills/ejkernel-quantization/SKILL.md`.

## Failure Routes

- `to_torch` or HF conversion mismatch: inspect fused layout metadata / `checkpoint_quant_reform_param` and load
  `.agents/skills/convert-checkpoint/SKILL.md`.
- Wrong gate/up, QKV, or MoE expert split: inspect `_dense.py`, `_moe.py`, `_builders.py`, and `_runtime.py` before
  touching model files.
- Quantized sharding crash: inspect `sanitize_partition_spec_for_shape` in `libs/easydel/easydel/infra/sharding.py`
  and the row-parallel sharding guards in `_linear_quantized.py`.
- Quantized matmul / TurboQuant kernel or performance claims: load `.agents/skills/ejkernel-quantization/SKILL.md` and
  `.agents/skills/optimize-ejkernel-kernel/SKILL.md` (new backend kernels: `.agents/skills/add-ejkernel-kernel/SKILL.md`).

## Verification

All quantization and layout computation tests must run on an accelerator (TPU shown; GPU:
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without an accelerator, report them as unverified on hardware:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/modules/test_fused_layout_tp_portability.py \
                libs/easydel/tests/modules/test_projection_layout.py \
                libs/easydel/tests/layers/test_fused_quantized_retp.py \
                libs/easydel/tests/layers/test_fused_layout_sharding_preservation.py

env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/layers/linears/test_quantized_sanitize_spec_for_shape.py \
                libs/easydel/tests/layers/linears/test_quantized_sharding.py \
                libs/easydel/tests/layers/quantization/test_base_module_rebuild.py \
                libs/easydel/tests/layers/quantization/test_checkpoint_quant_formats.py \
                libs/easydel/tests/layers/quantization/test_quantized_checkpoint_roundtrip.py \
                libs/easydel/tests/layers/quantization/test_qat_rules.py \
                libs/easydel/tests/trainers/test_quantization_group_size_compat.py

env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/ejkernel/test/modules/operations/test_quantized_matmul.py

uv run lint-imports
```
