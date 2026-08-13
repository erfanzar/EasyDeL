---
name: add-easydel-operation
description: Add or update an EasyDeL native operation or attention kernel under libs/easydel/easydel/operations. Use for OperationImpl, OperationRegistry, executor, requirements, FlashAttn, RaggedPageAttn, RingAttn, or backend-specific forward paths.
---

# Skill: Add Or Update An EasyDeL Operation

This is a specialization of `.claude/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the change is inside
`libs/easydel/easydel/operations` and does not involve creating a new ejkernel kernel (use the ejkernel skills for
that).

## First Reads

Read these before editing:

- `WORKSPACE.md`
- `libs/easydel/pyproject.toml`
- `libs/easydel/easydel/operations/_base_operation.py`
- `libs/easydel/easydel/operations/_operation_impl.py`
- `libs/easydel/easydel/operations/_operation_meta.py`
- `libs/easydel/easydel/operations/executor.py`
- `libs/easydel/easydel/operations/requirements/types.py`
- `libs/easydel/easydel/operations/requirements/requirements.py`
- `libs/easydel/easydel/operations/kernels/flash_attention.py`
- `libs/easydel/easydel/operations/kernels/ragged_page_attention.py`
- `libs/easydel/easydel/operations/kernels/ring_attention.py`

## Required Surfaces

A new operation usually needs:

- a class extending `OperationImpl` or `BaseOperation`
- `get_impl_name()` and registration via `@OperationRegistry.register`
- `get_requirements()` declaring required `MetadataField`s and supported
  `CacheType`s
- `forward_native` and optional `forward_tpu` / `forward_gpu` / `forward_cuda`
- tests under `libs/easydel/tests/operations/`

## Routing

- The operation wraps a new ejkernel kernel: load
  `.claude/skills/port-ejkernel-to-easydel-operation/SKILL.md`.
- The operation needs a new ejkernel backend implementation: load
  `.claude/skills/add-ejkernel-kernel/SKILL.md`.
- Cache integration: load `.claude/skills/debug-easydel-cache/SKILL.md`.
- eSurge runtime failure: load `.claude/skills/debug-esurge/SKILL.md`.

## Verification

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/operations/
```

Backend-specific paths (TPU/GPU) need hardware ownership and focused benchmarks; do not claim them from CPU tests alone.
