---
name: distributed-review
description: Focused review of EasyDeL workspace changes that touch Mesh, NamedSharding, PartitionSpec, shard_map, axis policies, FSDP/TP/SP/EP, or pipeline parallelism (SPMD or MPMD). Use as a review pass whenever a diff mentions sharding, partitioning, mesh axes, or stage assignment.
---

# Skill: Distributed Review

Specialization of `.agents/skills/review-pr/SKILL.md` for distributed-execution changes. Findings must be grounded
contract or correctness violations, not preferences.

## First Reads

- `WORKSPACE.md` (layering + ownership of each sharding layer)
- The diff, then the sharding surface it touches:
  - eFormer escale: `libs/eformer/eformer/escale/` (`create_mesh`, `PartitionAxis`, `PartitionManager`,
    `auto_partition_spec`) — eFormer's standalone surface; `easydel/infra/` gets `PartitionAxis` and meshes from
    `spectrax`, though `libs/easydel/easydel/trainers/dp_sharding.py` still hooks both `spectrax.sharding.manager` and
    `eformer.escale.partition.manager`
  - SpectraX: `libs/spectrax/spectrax/sharding/` (`get_named_sharding`, `SpxMesh`, `PartitionAxis`), `runtime/`
    (MPMD/SPMD paths, schedules, `sxstage_iter`), and `serialization/` (TensorStore save/load, `_fs.py`)
  - EasyDeL infra: `libs/easydel/easydel/infra/base_config.py` (6-axis convention `pp,dp,fsdp,ep,tp,sp`, lazy `mesh`,
    `expert_mesh`, `sharding_dcn_axis_dims`, `fsdp_is_ep_bound`, `sp_is_ep_bound`), `infra/sharding.py` (`AxisPolicy`,
    `RuntimeShardingResolver`), `infra/base_module.py` (init sharding context, stage assignment), `infra/layer_scan.py`
  - EasyDeL trainers: `libs/easydel/easydel/trainers/{batch_placement.py,data_lanes.py,dp_sharding.py}`
  - eSurge DP placement: `libs/easydel/easydel/inference/esurge/core/dp_sharding.py`, `libs/surgecore/src/engine/dp.rs`,
    `libs/surgecore/src/payload.rs`

## Review Checklist

1. **Resolution**: every new/changed spec resolves on a 1-device mesh and the accelerator's multi-device mesh
   (`libs/easydel/tests/infra/`).
2. **Divisibility & multi-slice**: sharded dims divide mesh axis sizes for realistic topologies; `-1` auto-fill
   assumptions and multi-slice `sharding_dcn_axis_dims` interactions are stated and sound.
3. **Axis-name & config hygiene**: no hardcoded mesh axis strings in model code — resolve via
   `PartitionAxis`/`AxisPolicy`; decode overrides respected; **no new `os.environ` sharding/layout knobs** (see
   `test_pipeline_stage_layout_is_explicit_not_env` in `libs/easydel/tests/infra/test_base_config_scan_stage_config.py`).
4. **Fused layouts under TP**: only TP-aware splitters and `reform_param` rules touch fused axes
   (`libs/easydel/easydel/layers/layouts/`); portability surface: `tests/modules/test_fused_layout_tp_portability.py`.
5. **Folded expert mesh & layer scan**: when `fsdp_is_ep_bound` / `sp_is_ep_bound` fold axes into `expert_mesh`,
   per-shard expert counts follow the folded `(dp, ep, tp)` mesh, not physical `ep`; check `layer_scan_options`
   (`infra/layer_scan.py`, default `prefetch_axes=("fsdp",)` on training text scans).
6. **Trainer batch placement & DP sharding**: verify `BatchPlacedStep` / `batch_shard_group_rows` non-contiguous
   local-to-global lifting and rank-3 mRoPE `(3, B, S)` handling (`trainers/batch_placement.py`), the SPMD-only
   constraint on `data_lane_local_batch=True` (`trainers/data_lanes.py`), and `dp_spmd_vmap_scope` /
   `DP_REDUCE_COMPILE_OPTIONS` (`trainers/dp_sharding.py`, patching both `spectrax` and `eformer` `PartitionAxis`).
7. **Communication & `shard_map`**: account for every added/moved collective (RowParallel all-reduce, EP all-to-all, PP
   transport — an unexplained all-gather is a finding); `in_specs`/`out_specs` match the surrounding mesh; sequence axis
   comes from metadata (default `"sp"`), not a literal.
8. **PP/MPMD**: stage assignments cover all layers (terminal reserve respected); `sxstage_iter` markers only under MPMD
   trace; `scan_layers` disabled when `PP > 1`; single positional microbatch input constraint holds.
9. **KV/cache & eSurge DP sharding**: DP page locality and rank-major DP packing across
   `inference/esurge/core/dp_sharding.py`, `libs/surgecore/src/engine/dp.rs`, and `libs/surgecore/src/payload.rs`
   (`tests/inference/esurge/core/test_dp_sharding_pages.py`); `use_sharded_kv_caching` remains mutually exclusive with
   `kv_cache_quantization_config`.
10. **Checkpoint compatibility**: EasyDeL checkpoints use `libs/spectrax/spectrax/serialization/` (TensorStore save/load
    without all-gather + GCS `_fs.py`); sharding-metadata changes must keep
    `libs/easydel/tests/infra/test_checkpoint_{save,load}_contract.py` passing.

## Verification

```bash
uv run lint-imports
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/infra <plus touched surface>
```

If `libs/surgecore` was touched, also run its Rust and CPU tests (see `.agents/skills/test-workspace/SKILL.md`).
Efficiency on real topologies is a hardware claim — report it as unverified unless measured.
