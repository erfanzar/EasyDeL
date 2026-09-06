---
name: eformer-checkpoint-sharding
description: Work on eFormer checkpointing, serialization, fsspec, async checkpoint management, mesh creation, partition constraints, or sharding utilities. Use for Checkpointer, tensorstore index metadata, process-safe remote writes, pytree serialization, escale mesh/partition APIs, or checkpoint restore layout bugs.
---

# Skill: Work On eFormer Checkpointing And Sharding

This is a specialization of `.xerxes/skills/run-research/SKILL.md`.

Load and follow `run-research` first. This skill adds eFormer package routing.

## First Reads

- `WORKSPACE.md`
- `libs/eformer/pyproject.toml`
- `libs/eformer/docs/api_docs/serialization/checkpointer.rst`
- `libs/eformer/docs/api_docs/serialization/serialization.rst`
- `libs/eformer/docs/api_docs/serialization/async_manager.rst`
- `libs/eformer/docs/api_docs/serialization/fsspec_utils.rst`
- `libs/eformer/docs/api_docs/serialization/sharding_utils.rst`
- `libs/eformer/docs/api_docs/escale/mesh/creation.rst`
- `libs/eformer/docs/api_docs/escale/partition/manager.rst`
- `libs/eformer/docs/api_docs/escale/partition/constraints.rst`
- `libs/eformer/docs/api_docs/escale/partition/auto_spec.rst`

Implementation paths:

- `libs/eformer/eformer/serialization/checkpointer.py`
- `libs/eformer/eformer/serialization/serialization.py`
- `libs/eformer/eformer/serialization/async_manager.py`
- `libs/eformer/eformer/serialization/fsspec_utils.py`
- `libs/eformer/eformer/serialization/sharding_utils.py`
- `libs/eformer/eformer/escale/mesh/creation.py`
- `libs/eformer/eformer/escale/partition/manager.py`
- `libs/eformer/eformer/escale/partition/constraints.py`
- `libs/eformer/eformer/escale/partition/auto_spec.py`

## Package Boundary

`libs/eformer` is a foundation package. It must not import `easydel`,
`spectrax`, or `ejkernel`.

## Checkpoint Rules

Use the existing APIs:

- `Checkpointer.save_checkpoint`
- `Checkpointer.load_checkpoint`
- `Checkpointer.save_pytree`
- `Checkpointer.load_pytree`
- `tree_serialize_leaves`
- `tree_deserialize_leaves`
- `create_sharding_tree_from_index`
- `apply_sharding_tree`

Preserve sharding metadata instead of loading everything unsharded. Relevant artifact names include
`tensorstore_index.json`, `metadata.json`, and
`checkpoint_metadata.json`.

For distributed remote writes, inspect
`should_write_shared_checkpoint_files` in
`libs/eformer/eformer/serialization/fsspec_utils.py`; nonzero processes should not race on shared metadata files.

## Mesh And Partition Rules

Use existing mesh and partition helpers:

- `create_mesh`
- `parse_mesh_from_string`
- `cpu_context`
- `dcn_mesh_dims`
- partition manager and constraint APIs under
  `libs/eformer/eformer/escale/partition/`

Do not add EasyDeL-specific axis policy to eFormer. Put integration code in EasyDeL if it depends on EasyDeL config
objects.

## Verification

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/eformer/tests/serialization/test_remote_checkpoint_writes.py

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/eformer/tests/serialization/test_serialization_utils.py

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/eformer/tests/escale/test_mesh_creation.py libs/eformer/tests/escale/test_partition_manager_api.py
```

If the failure comes from EasyDeL checkpoint conversion, also load
`.xerxes/skills/convert-checkpoint/SKILL.md`.
