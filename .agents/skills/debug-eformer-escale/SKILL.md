---
name: debug-eformer-escale
description: Debug or extend standalone eFormer mesh and sharding utilities under libs/eformer/eformer/escale. Use for eFormer create_mesh, parse_mesh_from_string, MeshPartitionHelper, PartitionAxis, PartitionManager, auto_partition_spec, with_sharding_constraint, or ShardingRule subclasses. For EasyDeL live sharding issues, use spectrax-sharding instead.
---

# Skill: Debug eFormer eScale Sharding

This is a specialization of `.agents/skills/run-research/SKILL.md`. Load and follow `run-research` first.

> **IMPORTANT — Ownership Notice:** EasyDeL has migrated its primary live sharding stack to `libs/spectrax/spectrax/sharding/`
> (see `.agents/skills/spectrax-sharding/SKILL.md`). Across `libs/easydel/easydel/`, only **one** compatibility hook
> still imports `eformer.escale` (`libs/easydel/easydel/trainers/dp_sharding.py`, patching
> `eformer.escale.partition.manager.PartitionAxis.resolve_axis` alongside SpectraX's). Use this skill ONLY for
> standalone `libs/eformer/eformer/escale/` work or that `dp_sharding.py` compat hook — route all other EasyDeL sharding
> bugs to `.agents/skills/spectrax-sharding/SKILL.md`.

## Package Boundary

`libs/eformer` is an independent foundation package (`pyproject.toml` `[tool.importlinter]`). It MUST NOT import
`easydel`, `spectrax`, `ejkernel`, `eray`, or `surgecore`. Do not add EasyDeL-specific axis policies to eFormer; keep
EasyDeL integration inside `libs/easydel`. Verify with `uv run lint-imports`.

## First Reads

- `libs/eformer/eformer/escale/__init__.py`
- `libs/eformer/eformer/escale/mesh/creation.py` (`create_mesh`, `parse_mesh_from_string`, `create_cpu_mesh`, `cpu_context`, `force_cpu`)
- `libs/eformer/eformer/escale/mesh/mesh_helpers.py` (`MeshPartitionHelper`)
- `libs/eformer/eformer/escale/partition/manager.py` (`PartitionAxis`, `PartitionManager`, `register`, `decode_*` rules)
- `libs/eformer/eformer/escale/partition/auto_spec.py` (`auto_partition_spec`, `vrn_auto_partition_spec`, `auto_namedsharding`, `auto_shard_array`)
- `libs/eformer/eformer/escale/partition/constraints.py` (`with_sharding_constraint`, `match_partition_rules`, `make_shard_and_gather_fns`, `get_incontext_mesh`)
- `libs/eformer/eformer/escale/helpers/base.py` (`ShardingRule` hierarchy, `ShardingAnalyzer`, `barrier_sync`)
- `libs/eformer/eformer/common_types.py`

## Key Semantics & Invariants

1. **5-axis default (no `pp` axis):**
   - `DEFAULT_NAMED_SHARDING_STG = ("dp", "fsdp", "ep", "tp", "sp")`
   - `DEFAULT_SHARDING_STG = (1, -1, 1, 1, 1)`
   - Contrast with SpectraX/EasyDeL's 6-axis convention `("pp", "dp", "fsdp", "ep", "tp", "sp")`.
2. **`parse_mesh_from_string` uses COLON syntax (`libs/eformer/eformer/escale/mesh/creation.py`):**
   - Named format MUST use colons: `"dp:2,fsdp:2,tp:2"` (all names in `names` must be present).
   - Positional format uses commas only: `"2,2,2"`.
   - Passing `"dp=2,fsdp=2,tp=2"` hits the positional `int(x)` branch and crashes with `ValueError`.
3. **`create_mesh` options & caching (`libs/eformer/eformer/escale/mesh/creation.py`):**
   - `dcn_mesh_dims` is an optional keyword **parameter** of `create_mesh(..., dcn_mesh_dims=...)` (not a standalone
     helper function) used for hybrid multi-slice/multi-process meshes via `create_hybrid_device_mesh`.
   - `axis_types` accepts `AxisType` or `"auto"` / `"explicit"` / `"manual"` (single value or per-axis sequence).
   - `use_jax=True` uses `jax.make_mesh` ONLY for single-slice, single-process calls with `dcn_mesh_dims=None`;
     otherwise falls back to `_cached_mesh` (`mesh_utils`).
   - Pre-existing multi-slice fallback env var: `MEGASCALE_NUM_SLICES` (checked when devices lack `slice_index`).
   - CPU debugging helpers: `create_cpu_mesh(...)`, `cpu_context(...)`, `force_cpu(...)`.
4. **Constraints & pre-existing env vars (`libs/eformer/eformer/escale/partition/constraints.py`):**
   - `with_sharding_constraint` (aliased as `lax_reshard`) auto-corrects incompatible specs against mesh/shape.
   - Pre-existing env vars read at import time: `MIN_SHARDING_SIZE` (default `16384`; arrays with fewer elements stay
     unsharded) and `LOG_SHARDING_MOVE` (`"true"`/`"1"` logs auto-adjustments). Do NOT add new `os.environ` knobs.
5. **Partition manager & `ShardingRule` hierarchy:**
   - `PartitionAxis` (`libs/eformer/eformer/escale/partition/manager.py`) supports custom axis registration
     (`PartitionAxis.register`) and generation-mode `decode_*` rules via `PartitionManager` context managers.
   - `libs/eformer/eformer/escale/helpers/base.py` provides the Strategy hierarchy: `ShardingRule` (ABC),
     `AutoShardingRule`, `ShapeBasedShardingRule`, `MemoryConstrainedShardingRule`, `CompositeShardingRule`, plus
     `ShardingAnalyzer` and `barrier_sync`.
   - `MeshPartitionHelper` (`libs/eformer/eformer/escale/mesh/mesh_helpers.py`) analyzes pytree shapes and suggests
     multi-axis sharding methods (e.g., `("fsdp", "sp")`, `("tp",)`).

## Routing

- EasyDeL live sharding & MPMD stage-local sharding: `.agents/skills/spectrax-sharding/SKILL.md`
- Standalone eFormer serialization/checkpointing: `.agents/skills/eformer-serialization/SKILL.md`
- Sharding PR review: `.agents/skills/distributed-review/SKILL.md` and `sharding-expert` agent
- TPU pod/setup issues: `.agents/skills/debug-tpu-setup/SKILL.md`

## Verification

Computation tests MUST run on an accelerator (TPU shown; for GPU use
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without an accelerator, report computation checks as **unverified on
hardware**.

```bash
uv run lint-imports

# Full eFormer escale suite (accelerator required for test_mesh_creation, test_partition_axis, test_partition_manager_api)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/eformer/tests/escale/
```

Pure-Python `@pytest.mark.cpu_ok` file (runnable on CPU via `JAX_PLATFORMS=cpu`):
- `libs/eformer/tests/escale/test_mesh_topology.py`

Verify `parse_mesh_from_string` with colon-separated strings (e.g. `parse_mesh_from_string("dp:2,fsdp:2,tp:2", ["dp", "fsdp", "tp"])`).
