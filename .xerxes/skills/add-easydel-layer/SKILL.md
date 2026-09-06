---
name: add-easydel-layer
description: Add or update reusable EasyDeL neural-network layers under libs/easydel/easydel/layers. Use for attention variants, ParallelLinear, norms, RoPE, MoE routing, embeddings, quantization-aware layers, or fused projection builders.
---

# Skill: Add Or Update An EasyDeL Layer

This is a specialization of `.xerxes/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the change lives in
`libs/easydel/easydel/layers` or when a model needs a new reusable primitive.

## First Reads

Read these before editing:

- `WORKSPACE.md`
- `libs/easydel/pyproject.toml`
- `libs/easydel/easydel/layers/__init__.py`
- `libs/easydel/easydel/layers/attention/_unified.py`
- `libs/easydel/easydel/layers/attention/_flexible.py`
- `libs/easydel/easydel/layers/linears/_linear.py`
- `libs/easydel/easydel/layers/linears/_linear_quantized.py`
- `libs/easydel/easydel/layers/norms/_norms.py`
- `libs/easydel/easydel/layers/rotary/_rotary.py`
- `libs/easydel/easydel/layers/moe/_moe_module.py`
- `libs/easydel/easydel/layers/quantization/_configs.py`
- `libs/easydel/easydel/layers/quantization/_quants.py`
- `libs/easydel/easydel/layers/layouts/_builders.py`
- `libs/easydel/easydel/layers/layouts/_dense.py`

## Required Surfaces

A new layer usually needs:

- implementation under the right `layers/<area>/` subpackage
- export from `libs/easydel/easydel/layers/__init__.py`
- preservation of existing TP/FSDP sharding contracts (axis names, column/row parallel behavior, `PartitionAxis`)
- tests under `libs/easydel/tests/layers/`

Prefer extending existing classes (`UnifiedAttention`, `ParallelLinear`,
`BaseMoeModule`) over duplicating sharding or RoPE logic inside a model.

## Routing

- Quantized linears or fused projection packing: load
  `.xerxes/skills/quantization-layout/SKILL.md`.
- A new kernel backend for attention: load
  `.xerxes/skills/port-ejkernel-to-easydel-operation/SKILL.md`.
- Model-level integration of the new layer: load
  `.xerxes/skills/add-easydel-model/SKILL.md`.

## Verification

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/layers/<area>/
```

For attention or linears, also run sharding/layout tests:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/layers/linears libs/easydel/tests/layers/attention
```
