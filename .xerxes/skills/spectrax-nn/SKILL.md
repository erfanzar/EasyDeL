---
name: spectrax-nn
description: Add or update SpectraX neural-network layers and functional primitives under libs/spectrax/spectrax/nn and libs/spectrax/spectrax/functional. Use for Linear, Conv, Attention, Norm, Embed, containers, mixed-precision casts, or new layer modules.
---

# Skill: Add Or Update A SpectraX Layer

This is a specialization of `.xerxes/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the work is inside
`libs/spectrax/spectrax/nn`, `libs/spectrax/spectrax/functional`, or when a new layer primitive is needed.

## First Reads

Read these before editing:

- `WORKSPACE.md`
- `libs/spectrax/pyproject.toml`
- `libs/spectrax/spectrax/nn/__init__.py`
- `libs/spectrax/spectrax/nn/linear.py`
- `libs/spectrax/spectrax/nn/dense.py`
- `libs/spectrax/spectrax/nn/attention.py`
- `libs/spectrax/spectrax/nn/norm.py`
- `libs/spectrax/spectrax/functional/linear.py`
- `libs/spectrax/spectrax/functional/attention.py`
- `libs/spectrax/spectrax/functional/norm.py`

## Required Surfaces

A new layer usually needs:

- a `Module` subclass in `nn/` following `Parameter` / `Buffer` + axis-names + policy-cast conventions
- a pure JAX primitive in `functional/` if one does not already exist
- export from `libs/spectrax/spectrax/nn/__init__.py`
- tests under `libs/spectrax/tests/nn/` and `libs/spectrax/tests/functional/`

Preserve channels-last conv / pool layout `(N, *spatial, C)` and sequence-second attention `(N, T, ...)` conventions.

## Routing

- Module/state semantics or graph round-trips: load
  `.xerxes/skills/spectrax-core/SKILL.md`.
- Mixed-precision / dtype policy issues: inspect `core/policy.py` and load
  `.xerxes/skills/spectrax-core/SKILL.md`.
- Sharding propagation: load `.xerxes/skills/spectrax-sharding/SKILL.md`.
- Transform interaction (jit/vmap/scan): load
  `.xerxes/skills/spectrax-transforms/SKILL.md`.

## Verification

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/spectrax/tests/nn/ libs/spectrax/tests/functional/
```
