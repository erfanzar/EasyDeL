---
name: quantization-layout
description: Work on EasyDeL fused projection layout, TP-portable checkpoint layout, quantized linear integration, or ejkernel quantized matmul wiring. Use for fused_param_tp, canonical/runtime fused state conversion, QKV or gate-up projection packing, quantization group-size behavior, sanitized partition specs, or EasyDeL-ejkernel quantization integration.
---

# Skill: Work On Quantization And Projection Layout

This is a specialization of `.agents/skills/run-research/SKILL.md`.

Load and follow `run-research` first. If the work is a new or changed ejkernel
kernel, also load `.agents/skills/add-ejkernel-kernel/SKILL.md`.

## First Reads

- `WORKSPACE.md`
- `libs/easydel/easydel/layers/layouts/_canonical.py`
- `libs/easydel/easydel/layers/layouts/_dense.py`
- `libs/easydel/easydel/layers/layouts/_builders.py`
- `libs/easydel/easydel/layers/layouts/_runtime.py`
- `libs/easydel/easydel/layers/layouts/_torch_packing.py`
- `libs/easydel/easydel/layers/layouts/_reform.py`
- `libs/easydel/easydel/layers/layouts/_types.py`
- `libs/easydel/easydel/layers/quantization/_configs.py`
- `libs/easydel/easydel/layers/quantization/_quants.py`
- `libs/easydel/easydel/layers/quantization/_straight_through.py`
- `libs/easydel/easydel/layers/quantization/_turboquant.py`
- `libs/easydel/easydel/layers/linears/_linear_quantized.py`
- `libs/ejkernel/ejkernel/modules/operations/quantized_matmul.py`

## Boundary

EasyDeL owns model integration, checkpoint layout, projection packing,
partition specs, eLarge builder wiring, and linears. eJKernel owns backend
kernels, operation wrappers, registry, autotune, and kernel benchmarks.

Only `libs/easydel` may import `ejkernel`. Do not make foundation packages
import EasyDeL or each other.

## Fused Layout Rules

Use the existing layout APIs:

- `FUSED_TP_FIELD`
- `canonicalize_fused_state`
- `runtimeize_fused_state`
- `canonicalize_fused_optimizer_state`
- `runtimeize_fused_optimizer_state`
- `retp_fused_state`
- `retp_fused_optimizer_state`
- `read_fused_checkpoint_tp`
- `FusedSegment`
- `FusedColumnLayout`
- `dense_gate_up_layout`
- `dense_qkv_layout`
- `build_fused_gate_up_projection`
- `split_fused_gate_up_projection`
- `build_fused_qkv_projection`
- `split_fused_qkv_projection`

Do not split fused tensors by ad hoc string or axis assumptions. Preserve
layout metadata and `fused_param_tp` so checkpoints can move between TP shapes.

## Quantized Linear Rules

Inspect these symbols before editing behavior:

- `RowParallelLinearQuantized`
- `ColumnParallelLinearQuantized`
- `sanitize_partition_spec_for_shape`
- `ej_quantized_matmul`
- `_distributed_quantized_matmul`

When changing quantized matmul behavior, verify both EasyDeL integration tests
and ejkernel quantized operation tests. Route backend-kernel changes through
`.agents/skills/add-ejkernel-kernel/SKILL.md`.

## Failure Routes

- `to_torch` or conversion mismatch: inspect fused layout metadata and load
  `.agents/skills/convert-checkpoint/SKILL.md`.
- Wrong gate/up or qkv split: inspect `_dense.py`, `_builders.py`, and
  `_runtime.py` before changing model code.
- Quantized sharding crash: inspect partition-spec sanitization in
  `_linear_quantized.py`.
- Performance claim for quantized matmul: use the ejkernel benchmark and
  profiling routes in `.agents/skills/add-ejkernel-kernel/SKILL.md`.

## Verification

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/modules/test_fused_layout_tp_portability.py libs/easydel/tests/modules/test_projection_layout.py

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/layers/linears/test_quantized_sanitize_spec_for_shape.py libs/easydel/tests/layers/linears/test_quantized_sharding.py

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/layers/quantization/test_base_module_rebuild.py libs/easydel/tests/trainers/test_quantization_group_size_compat.py

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/ejkernel/test/modules/operations/test_quantized_matmul.py
```

TPU Pallas quantized-kernel checks require TPU ownership and the
`add-ejkernel-kernel` skill.
