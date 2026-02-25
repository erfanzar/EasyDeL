# EasyData: Data Management for EasyDeL

EasyData is EasyDeL's comprehensive data management module providing tools for loading, transforming, mixing, and efficiently streaming datasets for training and evaluation.

## Overview

EasyData provides:

- **Multiple Data Sources**: Load from local files (JSON, Parquet, Arrow, CSV), cloud storage (GCS, S3), or HuggingFace Hub
- **Transform DSL**: Chain transforms for tokenization, chat templates, field operations
- **Dataset Mixing**: Combine multiple datasets with static or dynamic weights
- **Token Packing**: Pack sequences efficiently with greedy, pool, or first-fit strategies
- **Streaming Support**: Process datasets too large to fit in memory
- **Multi-layer Caching**: Memory + disk caching for processed data
- **Async Data Loading**: Thread-based prefetching with JAX sharding support

## Architecture

```md
easydel/data/
├── core/           # Foundation layer (protocols, config, types)
│   ├── protocols.py    # ShardedDataSource, BaseStage interfaces
│   ├── config.py       # All configuration dataclasses
│   └── types.py        # Legacy DatasetMixture, TextDatasetInform
├── sources/        # Data source implementations
│   ├── base.py         # Parquet, JSON, Arrow, CSV, HF sources
│   └── hf_wrapper.py   # HFDatasetShardedSource wrapper
├── transforms/     # Transform DSL and processing
│   ├── base.py         # Transform interface
│   ├── tokenize.py     # TokenizeStage
│   ├── chat_template.py# ChatTemplateTransform
│   ├── field_ops.py    # Field manipulation transforms
│   ├── mixture.py      # MixedShardedSource, MixStage
│   └── pack.py         # GreedyPacker, PoolPacker, PackStage
├── execution/      # Data loading and pipeline
│   ├── pipeline.py     # Pipeline fluent API
│   ├── loader.py       # AsyncDataLoader, prefetching
│   ├── cache.py        # TreeCacheManager, DiskCache
│   └── save.py         # Dataset saving/export
└── distributed/    # Ray integration
    └── ray_utils.py
```

## Quick Links

| Guide                                         | Description                             |
| --------------------------------------------- | --------------------------------------- |
| [Quickstart](quickstart.md)                   | Get started in 5 minutes                |
| [Data Sources](sources.md)                    | Loading from files, cloud, HuggingFace  |
| [Transforms](transforms.md)                   | Tokenization, chat templates, field ops |
| [Dataset Mixing](mixing.md)                   | Combining multiple datasets             |
| [Pre-tokenization](pretokenization.md)        | Offline tokenization and saving         |
| [Streaming](streaming.md)                     | Streaming from HF and GCS               |
| [Pipeline API](pipeline.md)                   | Fluent pipeline DSL reference           |
| [Caching](caching.md)                         | Multi-layer caching system              |
| [Trainer Integration](trainer_integration.md) | Using with EasyDeL trainers             |

## Two APIs

EasyData provides two APIs for different use cases:

### 1. Simple API (HuggingFace Datasets)

For quick prototyping with HuggingFace datasets:

```python
from datasets import load_dataset
from easydel.data import block_mixture_interleave

# Load datasets
ds1 = load_dataset("dataset1", split="train")
ds2 = load_dataset("dataset2", split="train")

# Mix with weights (dict for explicit mapping)
mixed = block_mixture_interleave(
    datasets={"ds1": ds1, "ds2": ds2},
    weights={"ds1": 0.7, "ds2": 0.3},
    block_size=1000,
    seed=42,
    stop="restart",
)

# Use directly with trainer
trainer = ed.SFTTrainer(train_dataset=mixed, ...)
```

### 2. Pipeline API (ShardedDataSource)

For production workloads with full control:

```python
from easydel.data import (
    Pipeline,
    PipelineConfig,
    DatasetConfig,
    PackStageConfig,
)

config = PipelineConfig(
    datasets=[
        DatasetConfig(
            data_files="gs://bucket/data/*.parquet",
            tokenizer="meta-llama/Llama-2-7b",
            save_path="/output/tokenized",
        ),
        DatasetConfig(
            data_files="trl-lib/ultrafeedback_binarized",
            type="huggingface",
            split="train",
        ),
    ],
    pack=PackStageConfig(enabled=True, seq_length=2048),
)

pipeline = Pipeline.from_config(config)
for batch in pipeline.source().tokenize().mix().pack().load().build():
    train_step(batch)
```

## Key Concepts

### ShardedDataSource

The core abstraction for data sources. Every data format implements this protocol:

```python
class ShardedDataSource(Protocol[T]):
    @property
    def shard_names(self) -> Sequence[str]: ...
    def num_shards(self) -> int: ...
    def open_shard(self, shard_name: str) -> Iterator[T]: ...
    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[T]: ...
```

This enables:

- Resumable iteration (checkpoint at shard + row)
- Distributed training (assign shards to workers)
- Efficient cloud streaming (parallel shard access)

### Transform

Transforms modify examples one at a time:

```python
class Transform(Protocol):
    def __call__(self, example: dict) -> dict: ...
```

Transforms can be chained:

```python
from easydel.data import ChainedTransform, ChatTemplateTransform, TokenizeTransform

pipeline = ChainedTransform([
    ChatTemplateTransform(tokenizer),
    TokenizeTransform(tokenizer, max_length=2048),
])
```

### Pipeline Stages

The pipeline DSL chains stages:

```python
pipeline = Pipeline.from_config(config)
result = (
    pipeline
    .source()      # Load ShardedDataSource per dataset
    .tokenize()    # Apply tokenization
    .mix()         # Combine multiple datasets
    .pack()        # Pack into fixed-length sequences
    .load()        # Create AsyncDataLoader with prefetching
    .build()       # Return final iterator
)
```

## Installation

EasyData is included with EasyDeL:

```bash
pip install easydel
```

For cloud storage support:

```bash
pip install easydel[gcs]  # Google Cloud Storage
pip install easydel[s3]   # Amazon S3
```

## Next Steps

- Start with the [Quickstart Guide](quickstart.md)
- See [Streaming](streaming.md) for large-scale training
- Check [Trainer Integration](trainer_integration.md) for using with SFT/DPO/GRPO
