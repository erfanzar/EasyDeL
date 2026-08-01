---
name: build-dataset-pipeline
description: Build, normalize, pretokenize, pack, mix, save, or validate EasyData datasets for EasyDeL training. Use for libs/easydel/easydel/data, ParquetShardedSource, MixedShardedSource, sequence packing, tool-call dataset normalization, GCS/local dataset output, or trainer input issues.
---

# Skill: Build Or Debug An EasyData Pipeline

This is a specialization of `.claude/skills/run-research/SKILL.md`.

Load and follow `run-research` first. For disk pressure or staging failures,
read `.claude/ops/OPS.md`.

## First Reads

Read the EasyData docs before designing a pipeline:

- `WORKSPACE.md`
- `libs/easydel/docs/easydata/index.md`
- `libs/easydel/docs/easydata/quickstart.md`
- `libs/easydel/docs/easydata/sources.md`
- `libs/easydel/docs/easydata/transforms.md`
- `libs/easydel/docs/easydata/pipeline.md`
- `libs/easydel/docs/easydata/mixing.md`
- `libs/easydel/docs/easydata/pretokenization.md`
- `libs/easydel/docs/easydata/streaming.md`
- `libs/easydel/docs/easydata/caching.md`
- `libs/easydel/docs/easydata/trainer_integration.md`

Then inspect the code path you are about to use:

- `libs/easydel/easydel/data/core/types.py`
- `libs/easydel/easydel/data/sources/base.py`
- `libs/easydel/easydel/data/sources/hf_wrapper.py`
- `libs/easydel/easydel/data/transforms/pack.py`
- `libs/easydel/easydel/data/transforms/tokenize.py`
- `libs/easydel/easydel/data/transforms/collators.py`
- `libs/easydel/easydel/data/transforms/mixture.py`
- `libs/easydel/easydel/data/execution/pipeline.py`
- `libs/easydel/easydel/data/execution/save.py`

## Existing Primitives

Prefer the existing EasyData APIs:

- `ParquetShardedSource`
- `MixedShardedSource`
- `PackedShardedSource`
- `GreedyPacker`
- `PoolPacker`
- `FirstFitPacker`
- `SFTPreprocessTransform`
- `tokenize_and_save`
- `pretokenize`
- `save_iterator`
- `save_dataset`

`MixedShardedSource` injects `__source__`; the packing path in
`libs/easydel/easydel/data/transforms/pack.py` forwards source provenance.
Keep that provenance visible when debugging mixed packed batches.

## Tool-Calling Dataset Normalization

For OpenAI-style tool-call records, start with:

- `libs/easydel/easydel/scripts/normalize_openai_tool_dataset.py`

Useful flags:

- `--source-dataset`
- `--config-name`
- `--split`
- `--out`
- `--token`
- `--max-rows`
- `--streaming` / `--no-streaming`
- `--repo-id`
- `--push-to-hub` / `--no-push-to-hub`
- `--private` / `--no-private`

Do a small `--max-rows` run first, then inspect the produced JSONL, Parquet, or
metadata artifact.

## Output Contract

Before writing code, pin down:

- source dataset names, configs, and splits
- target schema
- whether the user wants one merged config label or source-preserving outputs
- local path, GCS path, or Hub repo
- sample count and shard target
- tokenizer and sequence length
- provenance fields that must survive packing

If the user says "all in one", produce one merged output/config label while
keeping row-level provenance fields where the pipeline supports it.

## Verification

Validate by reading the output, not by trusting the writer exit code:

- count rows or examples
- inspect schema and one decoded sample
- verify `__source__` or equivalent provenance if mixing
- verify packed lengths and labels/attention masks when sequence packing is on

Focused tests:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/data/test_parquet_source.py

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/data/test_execution_pretokenize.py libs/easydel/tests/data/test_execution_save.py

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/trainers/test_sequence_packing_flag.py
```

If a dataset is meant for eLarge, also load `.claude/skills/train-elarge/SKILL.md`.
