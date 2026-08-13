---
name: debug-eformer-escale
description: Debug or extend mesh and sharding orchestration in libs/eformer/eformer/escale. Use for create_mesh, PartitionAxis, PartitionManager, auto_partition_spec, with_sharding_constraint, or sharding-rule failures across DP/FSDP/TP/EP/SP.
---

# Skill: Debug eFormer eScale Sharding

This is a specialization of `.claude/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the failure involves mesh creation, logical-axis mapping,
automatic sharding, or sharding-constraint errors coming from `libs/eformer/eformer/escale`.

## First Reads

Read these before editing:

- `WORKSPACE.md`
- `libs/eformer/pyproject.toml`
- `libs/eformer/eformer/escale/__init__.py`
- `libs/eformer/eformer/escale/mesh/creation.py`
- `libs/eformer/eformer/escale/partition/manager.py`
- `libs/eformer/eformer/escale/partition/auto_spec.py`
- `libs/eformer/eformer/escale/partition/constraints.py`
- `libs/eformer/eformer/escale/helpers/base.py`
- `libs/eformer/eformer/common_types.py`

## Typical Tasks

1. Create or modify a device mesh for a new parallelism strategy.
2. Register a new logical/semantic axis and its mapping in `PartitionAxis`.
3. Change automatic sharding behavior in `auto_partition_spec` or add a new
   `ShardingRule`.
4. Trace sharding-constraint mismatches (wrong axis, non-divisible dims, missing mesh) through
   `with_sharding_constraint` and `PartitionManager`.

## Routing

- TPU setup / Ray / bad-node issues: load
  `.claude/skills/debug-tpu-setup/SKILL.md`.
- SpectraX sharding abstraction issues: load
  `.claude/skills/spectrax-sharding/SKILL.md`.
- Training OOM that looks sharding-related: load
  `.claude/skills/debug-training-oom/SKILL.md`.
- Checkpoint sharding layout: load
  `.claude/skills/eformer-checkpoint-sharding/SKILL.md`.

## Verification

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/eformer/tests/escale/
```

Mesh-creation tests should pass with the target parallelism string (e.g.,
`dp=2,fsdp=2,tp=2`).
