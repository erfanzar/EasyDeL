---
name: build-dataset-pipeline
description: Build, normalize, pretokenize, pack, mix, save, or validate EasyData datasets and Grain trainer data lanes for EasyDeL training. Use for libs/easydel/easydel/data, ParquetShardedSource, MixedShardedSource, PackedShardedSource, Grain data lanes, batch placement, tool-call dataset normalization, GCS/local dataset output, or trainer input issues.
---

# Skill: Build Or Debug An EasyData Pipeline

This is a specialization of `.agents/skills/run-research/SKILL.md`.

Load and follow `run-research` first. For disk pressure or staging failures, read `.agents/ops/OPS.md`.

## First Reads

Read the EasyData docs before designing a pipeline:

- `WORKSPACE.md`
- `libs/easydel/docs/easydata/{index,quickstart,sources,transforms,pipeline,mixing,pretokenization,streaming,caching,trainer_integration}.md`

Then inspect the relevant implementation files:

- EasyData core & sources:
  - `libs/easydel/easydel/data/core/types.py`
  - `libs/easydel/easydel/data/sources/{base.py,hf_wrapper.py,parquet_random_access.py,rowgroup_shuffle.py}`
  - `libs/easydel/easydel/data/shard_info_cache.py`
- Transforms & packing:
  - `libs/easydel/easydel/data/transforms/{pack.py,tokenize.py,chat_template.py,collators.py,mixture.py}`
- Execution, distributed preprocessing, caching, and profiling:
  - `libs/easydel/easydel/data/execution/{pipeline.py,loader.py,save.py,cache.py}`
  - `libs/easydel/easydel/data/distributed/ray_utils.py`
  - `libs/easydel/easydel/data/{teacher_cache.py,profiling.py}`
- Trainer input, preprocessing & Grain data lanes:
  - `libs/easydel/easydel/trainers/prompt_transforms.py` (`SFTPreprocessTransform`)
  - `libs/easydel/easydel/trainers/data_lanes.py`
  - `libs/easydel/easydel/trainers/batch_placement.py`

## Existing Primitives & Packing Rules

Prefer the existing EasyData APIs:

- `ParquetShardedSource`, `MixedShardedSource`, `PackedShardedSource`
- `GreedyPacker`, `PoolPacker`, `FirstFitPacker`
- `tokenize_and_save`, `pretokenize`, `save_iterator`, `save_dataset`
- `SFTPreprocessTransform` (implemented in `libs/easydel/easydel/trainers/prompt_transforms.py` and lazily re-exported
  by `easydel.data`)

Key invariants:

- **Unpadded input for packing (`pack.py`, `prompt_transforms.py`):** `PackedShardedSource` requires unpadded
  sequences — set `SFTPreprocessTransform(packing_mode=True)` (or `padding=False`) before packing.
- **Source provenance:** `MixedShardedSource` injects `__source__`; `transforms/pack.py` forwards provenance through
  packed batches. Keep provenance intact when debugging mixed packed datasets.

## Grain Data Lanes & Trainer Input (`data_lanes.py` + `batch_placement.py`)

When debugging or extending trainer data loading:

- **Map / random-access lanes:** `parquetra` and ArrayRecord sources run through `build_map_lane` and
  `PackedParquetLoader` (`trainers/data_lanes.py`), checkpointing resumable cursors under `data_lane_state`
  (`DATA_LANE_STATE_DIR`, written by `BaseTrainer._save_data_lane_state`).
- **Opt-in local group batching (`TrainingArguments.data_lane_local_batch`):** `resolve_lane_batch_size`
  (`trainers/data_lanes.py`), `batch_shard_group_rows`, and `place_batch` (`trainers/batch_placement.py`) size each
  host group's loader to its local batch-shard rows and lift per-group host arrays into the global sharded batch rather
  than materializing full global batches on every host.
- **Sideband carriers:** preserve `ASSISTANT_ONLY_FIELD` and `DATASET_ID_FIELD` (`trainers/data_lanes.py`) across
  collators and lane transforms.

## Tool-Calling Dataset Normalization

This skill is the canonical owner of `libs/easydel/easydel/scripts/normalize_openai_tool_dataset.py` (cross-linked from
`.agents/skills/tool-reasoning-parser/SKILL.md`).

CLI flags:

- `--source-dataset`, `--config-name`, `--split`, `--out`, `--token`, `--max-rows`
- `--streaming` / `--no-streaming`
- `--drop-incomplete-tool-call-rows` / `--no-drop-incomplete-tool-call-rows` (**defaults to `True`** — silently drops
  rows with malformed tool calls or calls lacking matching tool results unless negated)
- `--repo-id`, `--push-to-hub` / `--no-push-to-hub`, `--private` / `--no-private`

Always run a small `--max-rows` validation pass first, then inspect `{split}.jsonl`, `{split}.parquet`, and
`metadata.json`.

## Output Contract

Before writing pipeline code, pin down:

- source dataset names, configs, and splits
- target schema and provenance fields that must survive packing
- whether the user wants one merged config label or source-preserving outputs (for "all in one", produce one merged
  output/config label while keeping row-level provenance where supported)
- local path, GCS path (`ePath`), or Hub repo
- sample count, shard target, tokenizer, and sequence length

## Verification

Validate produced datasets by reading them back — never trust exit code alone: check row counts, schema, one decoded
sample, `__source__` provenance when mixing, and packed sequence lengths / labels / attention masks.

Pure-host / `cpu_ok` data and lane unit tests (CPU trio is allowed):

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/data/test_parquet_source.py \
                libs/easydel/tests/data/test_execution_pretokenize.py \
                libs/easydel/tests/data/test_execution_save.py \
                libs/easydel/tests/data/test_sft_packing_guard.py \
                libs/easydel/tests/data/test_parquet_random_access.py \
                libs/easydel/tests/trainers/test_grain_data_lanes.py \
                libs/easydel/tests/trainers/test_lane_batch_shard_groups.py
```

Trainer batch placement and JAX packing computation tests must run on an accelerator (TPU shown; GPU:
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without an accelerator, report them as unverified on hardware:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/trainers/test_sequence_packing_flag.py \
                libs/easydel/tests/trainers/test_batch_placement.py
```

For eLarge YAML wiring or trainer integration, also load `.agents/skills/train-elarge/SKILL.md` or
`.agents/skills/add-easydel-trainer/SKILL.md`.
