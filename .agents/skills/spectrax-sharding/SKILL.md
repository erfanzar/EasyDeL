---
name: spectrax-sharding
description: Work on SpectraX sharding and mesh abstractions under libs/spectrax/spectrax/sharding — EasyDeL's primary live sharding stack. Use for SpxMesh, create_mesh, PartitionAxis, PartitionManager, logical_axis_rules, with_sharding_constraint, strict_sharding, audit_placement, setup placement helpers, or stage-local MPMD mesh resolution.
---

# Skill: Work On SpectraX Sharding

This is a specialization of `.agents/skills/run-research/SKILL.md`. Load and follow `run-research` first.

**Ownership:** `libs/spectrax/spectrax/sharding/` is **EasyDeL's primary live sharding stack**. EasyDeL layers,
operations, caches, configs, trainers, and `libs/easydel/easydel/axis.py` (`ATTN_DP` registered via
`PartitionAxis.register`) import `SpxMesh`, `create_mesh`, `PartitionAxis`, `PartitionManager`, `logical_axis_rules`,
and `with_sharding_constraint` from `spectrax.sharding`. Only one compat hook remains in EasyDeL touching
`eformer.escale` (`libs/easydel/easydel/trainers/dp_sharding.py`). Route to `.agents/skills/debug-eformer-escale/SKILL.md`
only for standalone `libs/eformer` work or that single compat hook.

## Package Boundary

`libs/spectrax` is an independent foundation package (`pyproject.toml` `[tool.importlinter]`). It MUST NOT import
`easydel`, `ejkernel`, `eformer`, `eray`, or `surgecore`. Do not put EasyDeL-specific axis policies (like `ATTN_DP`)
inside `libs/spectrax` — downstream packages extend via `PartitionAxis.register(...)`. Verify with `uv run lint-imports`.

## First Reads

- `libs/spectrax/spectrax/sharding/__init__.py`
- `libs/spectrax/spectrax/sharding/mesh.py` (`SpxMesh`, `create_mesh`, `parse_mesh_from_string`, 6-axis defaults)
- `libs/spectrax/spectrax/sharding/manager.py` (`PartitionAxis`, `PartitionManager`, `register`/`unregister`, decode modes)
- `libs/spectrax/spectrax/sharding/partition.py` (`_resolve_constraint_target`, `with_sharding_constraint`, sanitization)
- `libs/spectrax/spectrax/sharding/logical.py` (`logical_axis_rules`, `current_axis_rules`)
- `libs/spectrax/spectrax/sharding/strict.py` (`strict_sharding`, `set_strict_sharding`, `is_strict_sharding`)
- `libs/spectrax/spectrax/sharding/audit.py` (`audit_placement`, `PlacementReport`, `PlacementIssue`)
- `libs/spectrax/spectrax/sharding/placement.py` (`place_setup_leaf_with_sharding`, `place_setup_tree_with_shardings`,
  `reshape_with_named_shardings`, `transpose_with_named_shardings`, `spec_shape_mismatches`)
- `libs/spectrax/spectrax/core/sharding.py`, `libs/spectrax/spectrax/common_types.py`
- EasyDeL integration (when debugging EasyDeL): `libs/easydel/easydel/axis.py`, `libs/easydel/easydel/infra/sharding.py`

## Core Architectural Invariants

1. **6-axis default convention (`libs/spectrax/spectrax/sharding/mesh.py`):**
   - `DEFAULT_MESH_AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")`
   - `DEFAULT_MESH_AXIS_DIMS = (1, 1, -1, 1, 1, 1)` (`-1` fills remaining devices; contrast with eFormer's 5-axis default
     which omits `pp`).
   - `parse_mesh_from_string` uses **colon** syntax (`"dp:2,tp:4"`) or positional (`"2,4"`). Equals syntax (`dp=2`)
     raises `ValueError`.
2. **`SpxMesh.is_mpmd` requires `mpmd_axis` dim > 1 (`libs/spectrax/spectrax/sharding/mesh.py`):**
   - Setting `mpmd_axis="pp"` with `pp=1` intentionally makes `mesh.is_mpmd == False`. A single-stage pipeline is
     semantically pure SPMD; flipping MPMD on for `pp=1` would route `spx.jit` through the MPMD pipeline runtime and
     force XLA layout-reformat copies at JIT boundaries.
3. **5-step MPMD constraint resolution (`libs/spectrax/spectrax/sharding/partition.py`, `_resolve_constraint_target`):**
   On an `MpMdMesh`, `with_sharding_constraint` / `lax_reshard` must target a **stage-local sub-mesh**, never the full
   pipeline mesh. Target resolution priority is:
   1. Explicit `stage` / `stage_mesh` argument
   2. Variable's `pipeline_stage` metadata (`metadata` arg)
   3. Active `spx.assign_stage(...)` context
   4. Array's existing sharding
   5. Current process's MPMD rank
   - **Default fallback:** if no stage resolves, the constraint becomes a **silent no-op** rather than miscompiling, and
     `sanitize_partition_spec_for_mesh_and_shape` / `get_corrected_named_sharding` silently drops missing or
     non-divisible axes.
4. **Debugging silent replication with `strict_sharding` & `audit_placement` (`strict.py`, `audit.py`):**
   - Wrap suspect code in `with spx.sharding.strict_sharding():` (or `set_strict_sharding(True)`) to turn silent
     unresolvable-stage no-ops, dropped axes during sanitization, and missing-mesh constraints into `ShardingError` at
     trace time.
   - Run `report = spx.sharding.audit_placement(model, mesh); print(report.format()); report.raise_for_errors()` to
     audit actual variable placement vs metadata. Issue codes:
     - Errors: `misplaced` (wrong stage sub-mesh or spec), `unresolvable`
     - Warnings: `dropped_axis` (sanitization dropped an axis), `replicated_across_stages` (untagged variable >=
       `replicated_warn_bytes` on multi-stage MPMD), `unmaterialized` (deferred variable not initialized yet).
5. **Semantic axes & decode overrides (`libs/spectrax/spectrax/sharding/manager.py`, `common_types.py`):**
   - `PartitionAxis` resolves symbolic tokens (`BATCH`, `EMBED`, `HEAD`, `DP`, `FSDP`, ...) into physical mesh axes via
     `with PartitionManager(paxis):` (backed by a `ContextVar`).
   - `MODE_DECODE` and `MODE_INSERT` consult the `decode_*` overrides on `PartitionAxis`.
   - Downstream libraries register custom tokens via `PartitionAxis.register(token, rule, generation_axis_rule=...)`
     (e.g., EasyDeL's `ATTN_DP` in `libs/easydel/easydel/axis.py`).
6. **Setup-time placement (`libs/spectrax/spectrax/sharding/placement.py`):**
   - Multi-controller JAX rejects direct `jax.device_put` between differing physical device sets (common when moving
     replicated leaves onto stage sub-meshes). Use `place_setup_leaf_with_sharding` / `place_setup_tree_with_shardings`
     for init/load paths; use `reshape_with_named_shardings` / `transpose_with_named_shardings` to avoid dropping layout
     metadata across shape transforms; use `spec_shape_mismatches` for explicit divisibility diagnostics.

## Routing

- Core `Module` / `Variable` sharding metadata & `assign_stage`: `.agents/skills/spectrax-modules/SKILL.md`
- MPMD stage execution & per-rank sub-meshes: `.agents/skills/spectrax-pipeline-runtime/SKILL.md`
- Sharding review across PR diffs: `.agents/skills/distributed-review/SKILL.md` and `sharding-expert` agent
- Standalone `eformer.escale` (or `dp_sharding.py` compat hook): `.agents/skills/debug-eformer-escale/SKILL.md`

## Verification

Computation tests MUST run on an accelerator (TPU shown; for GPU use
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without an accelerator, report computation checks as **unverified on
hardware**.

```bash
uv run lint-imports

# Full SpectraX sharding suite (accelerator required)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/spectrax/tests/sharding/
```

Pure-Python `@pytest.mark.cpu_ok` tests (runnable on CPU via `JAX_PLATFORMS=cpu`):
- `libs/spectrax/tests/sharding/test_mesh_topology.py`
- `libs/spectrax/tests/sharding/test_semantic_sentinels.py`
- `cpu_ok`-marked cases in `libs/spectrax/tests/sharding/test_placement_audit.py`
