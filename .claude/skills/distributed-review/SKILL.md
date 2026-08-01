---
name: distributed-review
description: Focused review of EasyDeL workspace changes that touch Mesh, NamedSharding, PartitionSpec, shard_map, axis policies, FSDP/TP/SP/EP, or pipeline parallelism (SPMD or MPMD). Use as a review pass whenever a diff mentions sharding, partitioning, mesh axes, or stage assignment.
---

# Skill: Distributed Review

Specialization of `.claude/skills/review-pr/SKILL.md` for
distributed-execution changes. Findings must be grounded contract or
correctness violations, not preferences.

## First Reads

- `WORKSPACE.md` (layering + who owns which sharding layer)
- the diff, then the sharding layer it touches:
  - eformer escale: `libs/eformer/eformer/escale/` (`create_mesh`,
    `PartitionAxis`, `PartitionManager`, `auto_partition_spec`) — eformer's
    standalone surface; easydel currently gets `PartitionAxis` and mesh
    creation from spectrax, not escale
  - spectrax: `libs/spectrax/spectrax/sharding/` (logical rules,
    `get_named_sharding`, `SpxMesh`, `PartitionAxis`) and `runtime/`
    (MPMD/SPMD paths, schedules, `sxstage_iter` markers)
  - easydel infra: `libs/easydel/easydel/infra/base_config.py` (6-axis
    convention `pp,dp,fsdp,ep,tp,sp`, lazy `mesh`, `expert_mesh`),
    `infra/sharding.py` (`AxisPolicy`, `RuntimeShardingResolver`),
    `infra/base_module.py` (init sharding context, stage assignment)

## Review Checklist

1. **Resolution**: every new/changed spec resolves on a 1-device mesh and
   the fake 8-device CPU mesh. Regression surface:
   `libs/easydel/tests/infra/` (state sharding, pipeline stage regions,
   scan stage config).
2. **Divisibility**: sharded dims divide the mesh axis size for realistic
   topologies, not just the test mesh; `-1` auto-fill assumptions stated.
3. **Axis-name hygiene**: no hardcoded mesh axis strings in model code —
   resolve through `PartitionAxis`/`AxisPolicy`; generation-mode overrides
   respected for decode paths.
4. **Fused layouts under TP**: only the TP-aware splitters and
   `reform_param` rules touch fused axes
   (`libs/easydel/easydel/layers/layouts/`); portability surface:
   `tests/modules/test_fused_layout_tp_portability.py`.
5. **Communication**: identify each collective the change adds or moves
   (all-reduce from RowParallel, EP all-to-all, PP transport). An
   unexplained new all-gather is a finding.
6. **shard_map**: in_specs/out_specs consistent with the surrounding mesh;
   sequence axis name comes from metadata (default `"sp"`), not a literal.
7. **PP/MPMD**: stage assignments cover all layers (terminal reserve
   respected); markers only under MPMD trace; `scan_layers` and PP not
   combined; microbatching constraints (single positional input) hold.
8. **KV/cache sharding**: eSurge DP page locality
   (`inference/esurge/core/dp_sharding.py`,
   `tests/inference/esurge/core/test_dp_sharding_pages.py`);
   `use_sharded_kv_caching` interactions (mutually exclusive with KV-cache
   quantization).
9. **Checkpoint compatibility**: sharding-metadata changes still load
   existing checkpoints (TensorStore save/load preserves shardings without
   all-gather — a layout change can silently break restore).

## Verification

```bash
uv run lint-imports
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/infra <plus the touched surface>
```

Efficiency on real topologies is a hardware claim — report it as unverified
unless measured.
