---
name: spectrax-core
description: Work on the SpectraX core object model under libs/spectrax/spectrax/core. Use for Module, Variable, Parameter, Buffer, GraphDef, State, Selector, policy, lazy_init, stage assignment, or graph/state round-trips.
---

# Skill: Work On SpectraX Core

This is a specialization of `.xerxes/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the work is inside
`libs/spectrax/spectrax/core` or when module/state semantics, graph export/bind, selectors, or variable policies change.

## First Reads

Read these before editing:

- `WORKSPACE.md`
- `libs/spectrax/pyproject.toml`
- `libs/spectrax/README.md`
- `libs/spectrax/docs/design.md`
- `libs/spectrax/spectrax/core/module.py`
- `libs/spectrax/spectrax/core/variable.py`
- `libs/spectrax/spectrax/core/graph.py`
- `libs/spectrax/spectrax/core/state.py`
- `libs/spectrax/spectrax/core/selector.py`
- `libs/spectrax/spectrax/core/policy.py`
- `libs/spectrax/spectrax/core/lazy_init.py`
- `libs/spectrax/spectrax/core/stage_assignment.py`

## Typical Tasks

1. Add a new variable collection or specialized `Variable` subclass and wire it through `export` / `bind`.
2. Fix graph round-trip bugs: `GraphDef` mismatch after `clone`, shared-variable aliasing, or container reconstruction
   in `bind`.
3. Extend `Selector` behavior or fix `partition_state` interaction with tied weights / MPMD stage metadata.
4. Implement module-level hooks/contexts or debug mutations dropped inside
   `jax.jit` vs `spx.jit`.

## Routing

- Layer / model development: load `.xerxes/skills/spectrax-nn/SKILL.md`.
- Module-aware transforms: load `.xerxes/skills/spectrax-transforms/SKILL.md`.
- Pipeline runtime / MPMD stage issues: load
  `.xerxes/skills/spectrax-pipeline-runtime/SKILL.md`.
- Sharding / mesh issues: load `.xerxes/skills/spectrax-sharding/SKILL.md`.

## Verification

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/spectrax/tests/core/
```

Also run graph/state round-trip tests and selector tests if those areas changed.
