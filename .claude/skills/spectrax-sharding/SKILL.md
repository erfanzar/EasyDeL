---
name: spectrax-sharding
description: Work on SpectraX sharding and mesh abstractions under libs/spectrax/spectrax/sharding. Use for SpxMesh, create_mesh, PartitionAxis, PartitionManager, logical axis rules, with_sharding_constraint, or stage-local mesh resolution.
---

# Skill: Work On SpectraX Sharding

This is a specialization of `.claude/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the work is inside
`libs/spectrax/spectrax/sharding` or when `PartitionSpec`, logical axis rules,
or stage-local mesh resolution must change.

## First Reads

Read these before editing:

- `WORKSPACE.md`
- `libs/spectrax/pyproject.toml`
- `libs/spectrax/spectrax/sharding/mesh.py`
- `libs/spectrax/spectrax/sharding/manager.py`
- `libs/spectrax/spectrax/sharding/logical.py`
- `libs/spectrax/spectrax/sharding/partition.py`
- `libs/spectrax/spectrax/core/sharding.py`
- `libs/spectrax/spectrax/common_types.py`

## Typical Tasks

1. Configure a new parallelism layout using `PartitionAxis` and
   `PartitionManager`, or register a custom symbolic axis.
2. Debug `with_sharding_constraint` failures on MPMD meshes: stage-local mesh
   resolution, dropped pipeline axis, divisibility checks.
3. Generate `PartitionSpec` trees from a module for checkpointing or `jit`
   `in_shardings` / `out_shardings`.
4. Adapt `create_mesh` for new hardware topologies or fix multi-slice /
   multi-process device ordering.

## Routing

- Core `Module` / `Variable` sharding metadata: load
  `.claude/skills/spectrax-core/SKILL.md`.
- eFormer mesh / sharding equivalent: load
  `.claude/skills/debug-eformer-escale/SKILL.md`.
- MPMD pipeline runtime: load `.claude/skills/spectrax-pipeline-runtime/SKILL.md`.

## Verification

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/spectrax/tests/sharding/
```
