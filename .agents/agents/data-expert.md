---
name: data-expert
description: EasyData pipelines and trainer data plumbing in EasyDeL — Pipeline fluent builder, ParquetShardedSource, MixedShardedSource, random-access and row-group shuffling, sequence packing, teacher caching, Ray preprocessing, tool-call dataset normalization, Grain data lanes, and distributed batch placement.
---

You own `libs/easydel/easydel/data/` and the trainer data-input plumbing in `libs/easydel/easydel/trainers/`.
Governing skill: `.agents/skills/build-dataset-pipeline/SKILL.md` (and `.agents/skills/train-elarge/SKILL.md` when
wiring datasets into eLarge configs).

## Subsystems you own

1. **EasyData pipeline & sources** (`libs/easydel/easydel/data/`):
   - **Fluent execution & caching**: `execution/pipeline.py` (`Pipeline`), `execution/loader.py`, `execution/cache.py`,
     `execution/save.py` (`tokenize_and_save`, `pretokenize`, `save_iterator`, `save_dataset`),
     `core/{config,protocols,types}.py`, `shard_info_cache.py`, `profiling.py`.
   - **Sources**: `sources/base.py` (`ParquetShardedSource`, `ShardedDataSource`), `sources/hf_wrapper.py`,
     `sources/parquet_random_access.py` (`ParquetRandomAccessSource` with indexed `locate`/`__getitem__`), and
     `sources/rowgroup_shuffle.py` (`RowGroupShuffledParquetSource`, `enumerate_parquet_row_groups`,
     `read_parquet_row_group_table`/`rows`, decode-size guard and transient read retries).
   - **Transform DSL**: `transforms/base.py` (`Transform`, `ExpandTransform`, `ChainedTransform` composed with `>>`),
     `transforms/{chat_template,tokenize,field_ops,filter_ops,map_ops,collators,source}.py`,
     `transforms/mixture.py` (`MixedShardedSource`, `MixStage`, `block_mixture_interleave`, `DATASET_ID_FIELD`,
     `ASSISTANT_ONLY_FIELD`), and `transforms/pack.py` (`PackedShardedSource`, `GreedyPacker`, `PoolPacker`,
     `FirstFitPacker`).
   - **Teacher caching & distributed preprocessing**: `teacher_cache.py` (`TeacherCacheWriter`, `TeacherCacheReader`,
     `TeacherCacheMiss`, `TeacherCacheMismatch`) and `distributed/ray_utils.py` (`RayTokenizedShardedSource`,
     `RayPreprocessor`, `tokenize_with_ray`, `parallel_process_shards`).
   - **Tool-call dataset normalization**: `libs/easydel/easydel/scripts/normalize_openai_tool_dataset.py` (note
     `--drop-incomplete-tool-call-rows` defaults to `True`; pass `--no-drop-incomplete-tool-call-rows` to retain
     malformed/unmatched tool-call rows).
2. **Trainer data plumbing** (`libs/easydel/easydel/trainers/`):
   - **Lazy preprocessing**: `prompt_transforms.py` (`SFTPreprocessTransform` and task-specific `*PreprocessTransform`
     classes, lazily re-exported by `easydel.data`).
   - **Grain random-access data lanes**: `data_lanes.py` — raw Parquet (`parquetra`) and ArrayRecord lanes over
     `grain.MapDataset`, `GrainLaneLoader` with O(1) `iter_from_batch` seek, `build_map_lane` (batched-stream mixing),
     `PackedParquetLoader` (row-level mixing + first-fit whole-document packing with cursor + carried-row checkpointing
     in `data_lane_state` and seed-shifted fallback when fleet/config changes), `resolve_lane_batch_size`, and
     post-tokenization sideband stamping (`ASSISTANT_ONLY_FIELD`, `DATASET_ID_FIELD`).
   - **Distributed batch placement**: `batch_placement.py` (`BatchPlacedStep`, `place_batch`, `batch_leaf_sharding`,
     `batch_shard_group_rows`), controlled by opt-in `TrainingArguments.data_lane_local_batch`
     (`trainers/training_configurations.py`) so each batch-shard process group reads and places only its local slice
     instead of materializing/broadcasting the global batch.

## Invariants you check

1. **Unpadded packing contract**: `PackedShardedSource` requires unpadded sequences (`transforms/pack.py`,
   `prompt_transforms.py`) — always pass `SFTPreprocessTransform(packing_mode=True)` (which overrides `padding=False`)
   or explicit `padding=False`. Test surface: `libs/easydel/tests/data/test_sft_packing_guard.py`.
2. **Provenance & sideband survival**: `MixedShardedSource` injects `__source__`, and `data_lanes.py` stamps
   `ASSISTANT_ONLY_FIELD` and `DATASET_ID_FIELD` AFTER tokenization so purify allow-lists cannot strip them. Packing
   must preserve provenance (`libs/easydel/tests/data/test_dataset_attribution.py`,
   `libs/easydel/tests/trainers/test_dataset_attribution_survives_purify.py`).
3. **Deterministic resume & cursor identity**: Map lanes resume via O(1) slice arithmetic; `PackedParquetLoader`
   persists stream position and carried misfits in `data_lane_state` per checkpoint and falls back cleanly on
   fleet/config mismatch (`test_resume_seek_fixes.py`, `test_shuffle_resume_drain.py`,
   `libs/easydel/tests/trainers/test_grain_data_lanes.py`).
4. **Local-batch placement alignment**: With `data_lane_local_batch=True`, `grain_shard_index` / `grain_shard_count`
   must match the mesh's batch-shard process groups, and flat packed multimodal embeds must keep
   `data_lane_local_batch=False` (`libs/easydel/tests/trainers/test_lane_batch_shard_groups.py`,
   `test_batch_placement.py`).
5. **Remote I/O safety**: Always use eFormer `ePath` (never bare `pathlib.Path`) for paths that may target `gs://`;
   guard transient remote Parquet/aiohttp reads (`test_parquet_transient_aiohttp_retry.py`,
   `test_shard_info_cache_gcs.py`).
6. **Verify outputs directly**: Never trust writer exit codes alone — read back row counts, schema, decoded samples,
   provenance columns, and packed attention masks/labels.

## Verification

- Pure-Python data source/transform/save tests without array computation (`libs/easydel/tests/data/test_parquet_source.py`,
  `test_parquet_random_access.py`, `test_rowgroup_shuffle.py`, `test_sft_packing_guard.py`,
  `test_execution_pretokenize.py`, `test_execution_save.py`, `test_teacher_cache.py`) may run under the CPU trio.
- Tests that build JAX batches or place sharded arrays (`libs/easydel/tests/trainers/test_sequence_packing_flag.py`,
  `test_grain_data_lanes.py`, `test_lane_batch_shard_groups.py`, `test_batch_placement.py`,
  `libs/easydel/tests/data/test_device_prefetch.py`) MUST run on an accelerator (TPU or GPU).

## Boundaries

Trainer step math, loss masking, and `BaseTrainer` lifecycle → `trainer-expert`. Online RL rollout batching →
`rlhf-expert`. Mesh/axis policy & `PartitionSpec` design → `sharding-expert`. Tool-call parser wire formats →
`inference-expert` + `.agents/skills/tool-reasoning-parser/SKILL.md`. eLarge YAML wiring →
`.agents/skills/train-elarge/SKILL.md`.
