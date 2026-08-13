---
name: spectrax-transforms
description: Work on SpectraX module-aware JAX transforms under libs/spectrax/spectrax/transforms. Use for spx.jit, grad, vmap, scan, remat, rng_axes, split_merge mutation handling, or mutable Buffer propagation.
---

# Skill: Work On SpectraX Transforms

This is a specialization of `.claude/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the work is inside
`libs/spectrax/spectrax/transforms` or when `spx.jit`, `spx.grad`, `spx.vmap`,
`spx.scan`, or mutable `Buffer` propagation behaves incorrectly.

## First Reads

Read these before editing:

- `WORKSPACE.md`
- `libs/spectrax/pyproject.toml`
- `libs/spectrax/docs/design.md`
- `libs/spectrax/spectrax/transforms/split_merge.py`
- `libs/spectrax/spectrax/transforms/jit.py`
- `libs/spectrax/spectrax/transforms/grad.py`
- `libs/spectrax/spectrax/transforms/vmap.py`
- `libs/spectrax/spectrax/transforms/scan.py`
- `libs/spectrax/spectrax/transforms/remat.py`
- `libs/spectrax/spectrax/transforms/rng_axes.py`

## Typical Tasks

1. Add support for a new JAX transform or extend `mutable=` semantics across transforms.
2. Debug mutation propagation: why a `Buffer` update under `spx.jit` / `spx.scan`
   is lost or raises `IllegalMutationError`.
3. Optimize the transform dispatch hot path (identity cache, flattened `State`
   ABI, scope-aware compilation).
4. Implement or fix RNG splitting behavior under `vmap` / `scan` using
   `StateAxes` and `split_rngs`.

## Routing

- Core module/state semantics: load `.claude/skills/spectrax-core/SKILL.md`.
- Pipeline runtime / MPMD dispatch: load
  `.claude/skills/spectrax-pipeline-runtime/SKILL.md`.
- Sharding / mesh issues inside transforms: load
  `.claude/skills/spectrax-sharding/SKILL.md`.

## Verification

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/spectrax/tests/transforms/
```

Run mutation and RNG tests first if those areas changed.
