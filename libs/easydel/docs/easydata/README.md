# EasyData: Data Management for EasyDeL

EasyData is EasyDeL's comprehensive data management module, providing flexible and efficient tools for loading, transforming, mixing, and streaming datasets for large-scale model training.

## Why EasyData?

Training modern LLMs requires handling datasets that are:

- **Too large to fit in memory** - EasyData provides streaming from local files, GCS, S3, and HuggingFace Hub
- **From multiple sources** - Mix datasets with static or dynamic weights
- **In various formats** - Parquet, JSON, Arrow, CSV, or HuggingFace datasets
- **Require preprocessing** - Tokenization, chat templates, field transformations

EasyData solves these challenges with a unified, JAX-optimized data pipeline.

## Core Features

### Multi-Source Data Loading

```python
from easydel.data import ParquetShardedSource, HuggingFaceShardedSource

# Local or cloud files
source = ParquetShardedSource("gs://bucket/data/*.parquet")

# HuggingFace Hub (streaming)
source = HuggingFaceShardedSource("HuggingFaceFW/fineweb", streaming=True)
```

### Dataset Mixing

```python
from easydel.data import block_mixture_interleave

# Dict format for explicit mapping (recommended)
mixed = block_mixture_interleave(
    datasets={"code": code_ds, "text": text_ds, "math": math_ds},
    weights={"code": 0.4, "text": 0.5, "math": 0.1},
    block_size=1000,
    seed=42,
)
```

### Dynamic Weight Scheduling

```python
from easydel.data import WeightScheduler, WeightSchedulePoint

scheduler = WeightScheduler([
    WeightSchedulePoint(step=0, weights={"easy": 0.9, "hard": 0.1}),
    WeightSchedulePoint(step=50000, weights={"easy": 0.1, "hard": 0.9}),
], interpolation="linear")
```

### Pipeline API

```python
from easydel.data import Pipeline, PipelineConfig, DatasetConfig

config = PipelineConfig(
    datasets=[DatasetConfig(data_files="data/*.parquet", tokenizer="llama")],
    pack=PackStageConfig(enabled=True, seq_length=2048),
)

for batch in Pipeline.from_config(config).source().tokenize().pack().load().build():
    train_step(batch)
```

### Seamless Trainer Integration

```python
import easydel as ed

# All trainers accept EasyData sources directly
trainer = ed.DPOTrainer(
    model=model,
    train_dataset=mixed_source,  # ShardedDataSource or HF Dataset
    processing_class=tokenizer,
    arguments=ed.DPOConfig(...),
)
```

## Supported Data Formats

| Format          | Source Class               | Cloud Support |
| --------------- | -------------------------- | ------------- |
| Parquet         | `ParquetShardedSource`     | GCS, S3, HTTP |
| JSON/JSONL      | `JsonShardedSource`        | GCS, S3, HTTP |
| Arrow IPC       | `ArrowShardedSource`       | GCS, S3, HTTP |
| CSV             | `CsvShardedSource`         | GCS, S3, HTTP |
| Plain Text      | `TextShardedSource`        | GCS, S3, HTTP |
| HuggingFace Hub | `HuggingFaceShardedSource` | Native        |

## Key Concepts

### ShardedDataSource

The core abstraction enabling:

- **Resumable iteration** - Checkpoint at shard + row level
- **Distributed training** - Assign shards to workers
- **Efficient cloud streaming** - Parallel shard access

### Transforms

Composable preprocessing:

- `ChatTemplateTransform` - Convert messages to formatted text
- `TokenizedShardedSource` - On-the-fly tokenization
- Field operations - Select, rename, filter, combine

### Token Packing

Pack sequences efficiently for training:

- **Greedy** - Simple concatenation
- **Pool** - Multiple packers for better fit
- **First-fit** - Bin-packing algorithm

### Multi-Layer Caching

TreeCache-style caching with:

- Memory (LRU) + Disk layers
- Compression (gzip, lz4, zstd)
- Automatic expiry

## Quick Links

- [Quickstart Guide](quickstart.md) - Get started in 5 minutes
- [Data Sources](sources.md) - Loading from files, cloud, HuggingFace
- [Transforms](transforms.md) - Tokenization, chat templates, field ops
- [Dataset Mixing](mixing.md) - Combining multiple datasets
- [Pre-tokenization](pretokenization.md) - Offline tokenization
- [Streaming](streaming.md) - Streaming from HF and GCS
- [Pipeline API](pipeline.md) - Fluent pipeline DSL
- [Caching](caching.md) - Multi-layer caching system
- [Trainer Integration](trainer_integration.md) - Using with EasyDeL trainers

## Installation

EasyData is included with EasyDeL:

```bash
pip install easydel

# For cloud storage
pip install easydel[gcs]  # Google Cloud Storage
pip install easydel[s3]   # Amazon S3
```

## Example: Large-Scale Pre-training

```python
from datasets import load_dataset
from easydel.data import (
    block_mixture_interleave,
    MixedShardedSource,
    HFDatasetShardedSource,
    WeightScheduler,
    WeightSchedulePoint,
)
import easydel as ed

# Load datasets with streaming
code_ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
text_ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

# Simple mixing
mixed = block_mixture_interleave(
    [code_ds, text_ds],
    weights={"code": 0.3, "text": 0.7},
    block_size=1000,
    seed=42,
    stop="restart",
)

# Or with dynamic scheduling
scheduler = WeightScheduler([
    WeightSchedulePoint(step=0, weights={"code": 0.2, "text": 0.8}),
    WeightSchedulePoint(step=100000, weights={"code": 0.5, "text": 0.5}),
], interpolation="cosine")

mixed = MixedShardedSource(
    sources={
        "code": HFDatasetShardedSource(code_ds),
        "text": HFDatasetShardedSource(text_ds),
    },
    weight_scheduler=scheduler,
)

# Train
trainer = ed.SFTTrainer(
    model=model,
    train_dataset=mixed,
    processing_class=tokenizer,
    arguments=ed.SFTConfig(max_length=2048),
)
trainer.train()
```
