# Pipeline API

The Pipeline API provides a fluent interface for building complex data processing pipelines with composable stages.

## Overview

```python
from easydel.data import Pipeline, PipelineConfig, DatasetConfig

# Create pipeline
config = PipelineConfig(datasets=[DatasetConfig(data_files="data/*.parquet")])
pipeline = Pipeline.from_config(config)

# Chain stages
result = (
    pipeline
    .source()      # Load data sources
    .tokenize()    # Apply tokenization
    .mix()         # Mix multiple datasets
    .pack()        # Pack sequences
    .load()        # Create data loader
    .build()       # Get final iterator
)

# Use in training
for batch in result:
    train_step(batch)
```

## Pipeline Stages

| Stage | Description | Input | Output |
|-------|-------------|-------|--------|
| `source()` | Load data from files/Hub | - | dict[name, ShardedDataSource] |
| `tokenize()` | Apply tokenization | dict[name, Source] | dict[name, TokenizedSource] |
| `mix()` | Combine multiple datasets | dict[name, Source] | dict["mixed", MixedSource] |
| `pack()` | Pack into fixed-length | dict[name, Source] | dict[name, PackedSource] |
| `save()` | Save to disk | dict[name, Source] | dict[name, Source] |
| `load()` | Create data loaders | dict[name, Source] | dict[name, AsyncDataLoader] |
| `build()` | Get final iterator | dict[name, Loader] | Iterator |

## PipelineConfig

The main configuration object:

```python
from easydel.data import (
    PipelineConfig,
    DatasetConfig,
    TokenizeStageConfig,
    MixStageConfig,
    PackStageConfig,
    LoadStageConfig,
    SaveStageConfig,
)

config = PipelineConfig(
    # Required: at least one dataset
    datasets=[
        DatasetConfig(
            name="dataset1",
            data_files="data/*.parquet",
            tokenizer="meta-llama/Llama-2-7b",
        ),
    ],

    # Global settings
    default_tokenizer="meta-llama/Llama-2-7b",  # Fallback tokenizer
    streaming=True,  # Enable streaming mode
    seed=42,  # Random seed

    # Stage configurations
    tokenize=TokenizeStageConfig(max_length=2048),
    mix=MixStageConfig(weights={"ds1": 0.5, "ds2": 0.5}),
    pack=PackStageConfig(enabled=True, seq_length=2048),
    load=LoadStageConfig(batch_size=8),
    save=SaveStageConfig(enabled=True, output_dir="./output"),
)
```

## DatasetConfig

Per-dataset configuration:

```python
from easydel.data import DatasetConfig, TokenizerConfig

config = DatasetConfig(
    # Source (required)
    data_files="data/*.parquet",  # Path, glob, or list

    # Identity
    name="my_dataset",  # Auto-generated if not provided

    # Source options
    type="parquet",  # json, parquet, csv, arrow, huggingface, txt
    split="train",   # For HuggingFace datasets
    num_rows=10000,  # Limit rows (optional)

    # Per-dataset tokenization
    tokenizer="meta-llama/Llama-2-7b",  # Or TokenizerConfig
    tokenizer_kwargs={"use_fast": True},

    # Per-dataset caching
    cache_path="./cache/my_dataset",
    cache_enabled=True,

    # Per-dataset saving
    save_path="./output/my_dataset",
    save_format="parquet",

    # Content mapping
    content_field="text",  # Field to tokenize
    additional_fields=["metadata"],  # Fields to preserve
    format_callback=my_transform_fn,  # Custom transform
    format_fields={"old_name": "new_name"},  # Rename fields
)
```

## Stage Details

### Source Stage

Loads data from configured sources:

```python
pipeline = Pipeline.from_config(config)
pipeline.source()

# Access loaded sources
sources = pipeline.get_data()
# {"dataset1": ParquetShardedSource, "dataset2": JsonShardedSource}
```

### Tokenize Stage

Applies tokenization with per-dataset settings:

```python
from easydel.data import TokenizeStageConfig

config = TokenizeStageConfig(
    default_tokenizer="meta-llama/Llama-2-7b",
    max_length=2048,
    batch_size=1000,  # Batch tokenization
    num_workers=4,    # Parallel workers
    cache_tokenized=True,
    remove_columns=["text"],  # Remove after tokenizing
)

pipeline.tokenize(config)  # Or use from PipelineConfig
```

### Mix Stage

Combines multiple datasets:

```python
from easydel.data import MixStageConfig, WeightSchedulePoint

config = MixStageConfig(
    # Static weights
    weights={"code": 0.3, "text": 0.7},

    # Or dynamic schedule
    weight_schedule=[
        WeightSchedulePoint(step=0, weights={"code": 0.2, "text": 0.8}),
        WeightSchedulePoint(step=10000, weights={"code": 0.5, "text": 0.5}),
    ],
    weight_schedule_type="linear",  # step, linear, cosine

    block_size=1000,
    stop_strategy="restart",  # restart, first_exhausted, all_exhausted
    seed=42,
)

pipeline.mix(config)
```

### Pack Stage

Packs sequences into fixed-length chunks:

```python
from easydel.data import PackStageConfig

config = PackStageConfig(
    enabled=True,
    seq_length=2048,
    eos_token_id=2,
    pad_token_id=0,
    strategy="greedy",  # greedy, pool, first_fit
    num_packers=4,      # For pool strategy
    include_segment_ids=True,
    shuffle_packed=True,
    shuffle_buffer_factor=10,
)

pipeline.pack(config)
```

**Packing Strategies:**

| Strategy | Description | Efficiency | Overhead |
|----------|-------------|------------|----------|
| `greedy` | Simple concatenation | Medium | Low |
| `pool` | Multiple packers for better fit | High | Medium |
| `first_fit` | Bin-packing algorithm | Highest | High |

### Save Stage

Saves processed data to disk:

```python
from easydel.data import SaveStageConfig

config = SaveStageConfig(
    enabled=True,
    output_dir="./processed_data",
    format="parquet",  # parquet, arrow, jsonl
    num_shards=100,
    compression="zstd",  # none, gzip, lz4, zstd
    max_shard_size="500MB",
    overwrite=False,

    # Push to HuggingFace Hub
    push_to_hub=False,
    hub_repo_id="username/dataset",
    hub_private=True,
    hub_token="hf_xxx",
)

pipeline.save(config)
```

### Load Stage

Creates async data loaders:

```python
from easydel.data import LoadStageConfig

config = LoadStageConfig(
    batch_size=8,
    prefetch_enabled=True,
    prefetch_workers=2,
    prefetch_buffer_size=4,
    shuffle_buffer_size=10000,
    drop_last=True,
    prefetch_to_device=False,  # JAX pre-sharding
)

pipeline.load(config)
```

## Full Pipeline Examples

### SFT Training Pipeline

```python
from easydel.data import (
    Pipeline,
    PipelineConfig,
    DatasetConfig,
    PackStageConfig,
    LoadStageConfig,
)

config = PipelineConfig(
    datasets=[
        DatasetConfig(
            data_files="conversations/*.jsonl",
            tokenizer="meta-llama/Llama-2-7b-chat-hf",
            content_field="messages",  # Chat data
        ),
    ],
    pack=PackStageConfig(
        enabled=True,
        seq_length=2048,
        eos_token_id=2,
    ),
    load=LoadStageConfig(
        batch_size=8,
        shuffle_buffer_size=10000,
    ),
)

pipeline = Pipeline.from_config(config)
for batch in pipeline.source().tokenize().pack().load().build():
    sft_train_step(batch)
```

### Multi-Dataset Pre-training

```python
config = PipelineConfig(
    datasets=[
        DatasetConfig(
            name="code",
            data_files="gs://bucket/code/*.parquet",
            tokenizer="bigcode/starcoder",
        ),
        DatasetConfig(
            name="text",
            data_files="gs://bucket/text/*.parquet",
            tokenizer="meta-llama/Llama-2-7b",
        ),
        DatasetConfig(
            name="math",
            data_files="gs://bucket/math/*.parquet",
            tokenizer="meta-llama/Llama-2-7b",
        ),
    ],
    mix=MixStageConfig(
        weights={"code": 0.4, "text": 0.5, "math": 0.1},
        block_size=2000,
    ),
    pack=PackStageConfig(
        enabled=True,
        seq_length=4096,
        strategy="pool",
    ),
    load=LoadStageConfig(batch_size=32),
)

pipeline = Pipeline.from_config(config)
for batch in pipeline.source().tokenize().mix().pack().load().build():
    pretrain_step(batch)
```

### Pre-tokenization Pipeline

```python
config = PipelineConfig(
    datasets=[
        DatasetConfig(
            data_files="raw_data/*.jsonl",
            tokenizer="meta-llama/Llama-2-7b",
            save_path="./tokenized_data",
        ),
    ],
    tokenize=TokenizeStageConfig(max_length=2048),
    save=SaveStageConfig(
        enabled=True,
        format="parquet",
        compression="zstd",
    ),
)

# Just tokenize and save
Pipeline.from_config(config).source().tokenize().save().build()
```

## Accessing Pipeline State

```python
pipeline = Pipeline.from_config(config)
pipeline.source().tokenize()

# Get current data
data = pipeline.get_data()
# {"dataset1": TokenizedShardedSource, ...}

# Get context
context = pipeline.get_context()
print(context.seed)
print(context.config)

# Get applied stages
stages = pipeline.get_stages()
# ["source", "tokenize"]
```

## Error Handling

```python
from easydel.data import Pipeline, PipelineConfig

config = PipelineConfig(datasets=[...])
pipeline = Pipeline.from_config(config)

# Validate config before running
errors = config.validate()
if errors:
    for error in errors:
        print(f"Config error: {error}")

# source() must be called first
try:
    pipeline.tokenize()  # Error!
except RuntimeError as e:
    print(e)  # "Call source() before other pipeline stages"
```

## Custom Stages

Extend the pipeline with custom stages:

```python
from easydel.data.core.protocols import BaseStage, PipelineContext, ShardedDataSource

class MyCustomStage(BaseStage):
    def __init__(self, config):
        super().__init__(config)
        self._config = config

    @property
    def name(self) -> str:
        return "my_stage"

    def process(
        self,
        data: dict[str, ShardedDataSource],
        context: PipelineContext,
    ) -> dict[str, ShardedDataSource]:
        # Transform data
        result = {}
        for name, source in data.items():
            # Apply your transformation
            result[name] = MyTransformedSource(source, self._config)
        return result
```

## Factory Functions

### create_pipeline

Quick pipeline creation:

```python
from easydel.data import create_pipeline, DatasetConfig

pipeline = create_pipeline(
    datasets=[
        DatasetConfig(data_files="data/*.parquet"),
        {"data_files": "more_data/*.jsonl"},  # Dict also works
    ],
    default_tokenizer="meta-llama/Llama-2-7b",
    streaming=True,
)
```

### tokenize_and_save

One-liner for tokenization:

```python
from easydel.data import tokenize_and_save

tokenize_and_save(
    data_files="data/*.jsonl",
    tokenizer="meta-llama/Llama-2-7b",
    output_path="./tokenized",
    output_format="parquet",
    max_length=2048,
)
```

## Next Steps

- [Transforms](transforms.md) - Custom transforms in pipeline
- [Dataset Mixing](mixing.md) - Advanced mixing strategies
- [Caching](caching.md) - Cache pipeline outputs
