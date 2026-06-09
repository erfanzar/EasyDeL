# Dataset Mixing

EasyData provides powerful dataset mixing capabilities with static weights, dynamic scheduling, and block-based deterministic mixing.

## Quick Start

### Simple Mixing with HuggingFace Datasets

```python
from datasets import load_dataset
from easydel.data import block_mixture_interleave

# Load datasets
code_ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
text_ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
math_ds = load_dataset("hendrycks/competition_math", split="train")

# Mix with weights using dict (recommended - explicit mapping)
mixed = block_mixture_interleave(
    datasets={"code": code_ds, "text": text_ds, "math": math_ds},
    weights={"code": 0.4, "text": 0.5, "math": 0.1},
    block_size=1000,
    seed=42,
    stop="restart",
)

# Use with trainer
trainer = ed.SFTTrainer(train_dataset=mixed, ...)
```

## block_mixture_interleave

The simplest way to mix HuggingFace datasets.

```python
from easydel.data import block_mixture_interleave

# Dict format with explicit name-to-dataset mapping
mixed = block_mixture_interleave(
    datasets={"code": ds1, "text": ds2, "math": ds3},
    weights={"code": 0.5, "text": 0.3, "math": 0.2},
    block_size=1000,
    seed=42,
    stop="restart",
)

# Equal weights
mixed = block_mixture_interleave(
    datasets={"code": ds1, "text": ds2},
    weights=None,  # Equal 50/50
)
```

**Parameters:**

| Parameter    | Type         | Description                                                          |
| ------------ | ------------ | -------------------------------------------------------------------- |
| `datasets`   | dict         | Dict mapping names to datasets: `{"code": code_ds, "text": text_ds}` |
| `weights`    | dict or None | Dict mapping names to weights (keys must match), None = equal        |
| `block_size` | int          | Number of examples per mixing block                                  |
| `seed`       | int          | Random seed for shuffling within blocks                              |
| `stop`       | str          | `"restart"` to loop, `"first_exhausted"` to stop                     |

**How it works:**

1. Divides training into blocks of `block_size` examples
2. Within each block, samples according to weights
3. Shuffles within block for variety
4. Uses deterministic RNG per block for reproducibility

## MixedShardedSource

For ShardedDataSource-based mixing with more control.

```python
from easydel.data import MixedShardedSource, HFDatasetShardedSource

# Wrap datasets as ShardedDataSource
source1 = HFDatasetShardedSource(dataset1)
source2 = HFDatasetShardedSource(dataset2)

# Mix with static weights
mixed = MixedShardedSource(
    sources={"code": source1, "text": source2},
    weights={"code": 0.3, "text": 0.7},
    block_size=1000,
    seed=42,
    stop_strategy="restart",
)

# Iterate
for example in mixed.open_shard(mixed.shard_names[0]):
    print(example["__source__"])  # Shows which dataset
```

**Parameters:**

| Parameter          | Type            | Default   | Description                                         |
| ------------------ | --------------- | --------- | --------------------------------------------------- |
| `sources`          | dict            | Required  | Name to ShardedDataSource mapping                   |
| `weights`          | dict            | None      | Static weights (None = equal)                       |
| `block_size`       | int             | 1000      | Examples per mixing block                           |
| `seed`             | int             | None      | Random seed                                         |
| `stop_strategy`    | str             | "restart" | `"restart"`, `"first_exhausted"`, `"all_exhausted"` |
| `weight_scheduler` | WeightScheduler | None      | Dynamic weight scheduler                            |

## Dynamic Weight Scheduling

Change dataset weights during training:

```python
from easydel.data import (
    MixedShardedSource,
    HFDatasetShardedSource,
    WeightScheduler,
    WeightSchedulePoint,
)

# Define schedule
scheduler = WeightScheduler(
    schedule=[
        WeightSchedulePoint(step=0, weights={"code": 0.2, "text": 0.8}),
        WeightSchedulePoint(step=10000, weights={"code": 0.5, "text": 0.5}),
        WeightSchedulePoint(step=50000, weights={"code": 0.8, "text": 0.2}),
    ],
    interpolation="linear",  # "step", "linear", or "cosine"
)

# Create sources
code_source = HFDatasetShardedSource(code_dataset)
text_source = HFDatasetShardedSource(text_dataset)

# Mix with scheduler
mixed = MixedShardedSource(
    sources={"code": code_source, "text": text_source},
    weight_scheduler=scheduler,
    block_size=1000,
)
```

### Interpolation Types

| Type       | Description                                  |
| ---------- | -------------------------------------------- |
| `"step"`   | Jump to new weights at each schedule point   |
| `"linear"` | Linearly interpolate between schedule points |
| `"cosine"` | Smooth cosine annealing between points       |

### Visualizing Schedule

```python
scheduler = WeightScheduler(
    schedule=[
        WeightSchedulePoint(step=0, weights={"code": 0.2, "text": 0.8}),
        WeightSchedulePoint(step=10000, weights={"code": 0.5, "text": 0.5}),
        WeightSchedulePoint(step=50000, weights={"code": 0.8, "text": 0.2}),
    ],
    interpolation="linear",
)

# Check weights at any step
print(scheduler.get_weights(0))      # {"code": 0.2, "text": 0.8}
print(scheduler.get_weights(5000))   # {"code": 0.35, "text": 0.65}
print(scheduler.get_weights(10000))  # {"code": 0.5, "text": 0.5}
print(scheduler.get_weights(30000))  # {"code": 0.65, "text": 0.35}
```

## Stop Strategies

| Strategy            | Behavior                                     |
| ------------------- | -------------------------------------------- |
| `"restart"`         | Loop exhausted datasets (infinite iteration) |
| `"first_exhausted"` | Stop when any dataset is exhausted           |
| `"all_exhausted"`   | Stop when all datasets are exhausted         |

```python
# Infinite training - datasets loop
mixed = MixedShardedSource(sources, stop_strategy="restart")

# Stop at smallest dataset
mixed = MixedShardedSource(sources, stop_strategy="first_exhausted")

# Train until all data seen at least once
mixed = MixedShardedSource(sources, stop_strategy="all_exhausted")
```

## Pipeline API Mixing

Using the Pipeline fluent API:

```python
from easydel.data import (
    Pipeline,
    PipelineConfig,
    DatasetConfig,
    MixStageConfig,
)

config = PipelineConfig(
    datasets=[
        DatasetConfig(
            name="code",
            data_files="code_data/*.parquet",
            tokenizer="meta-llama/Llama-2-7b",
        ),
        DatasetConfig(
            name="text",
            data_files="text_data/*.parquet",
            tokenizer="meta-llama/Llama-2-7b",
        ),
    ],
    mix=MixStageConfig(
        weights={"code": 0.3, "text": 0.7},
        block_size=1000,
        stop_strategy="restart",
    ),
)

pipeline = Pipeline.from_config(config)
for batch in pipeline.source().tokenize().mix().load().build():
    train_step(batch)
```

### MixStageConfig

```python
from easydel.data import MixStageConfig, WeightSchedulePoint

config = MixStageConfig(
    weights={"code": 0.3, "text": 0.7},  # Static weights
    weight_schedule=[  # Or dynamic schedule
        WeightSchedulePoint(step=0, weights={"code": 0.2, "text": 0.8}),
        WeightSchedulePoint(step=10000, weights={"code": 0.5, "text": 0.5}),
    ],
    weight_schedule_type="linear",  # "step", "linear", "cosine"
    block_size=1000,
    stop_strategy="restart",
    seed=42,
)
```

## CompositeShardedSource

For mixing without weights (simple concatenation):

```python
from easydel.data import CompositeShardedSource, ParquetShardedSource

source1 = ParquetShardedSource("data1/*.parquet")
source2 = ParquetShardedSource("data2/*.parquet")

# Simple concatenation (no mixing)
combined = CompositeShardedSource([source1, source2])

# All shards from both sources
for shard in combined.shard_names:
    for example in combined.open_shard(shard):
        process(example)
```

## Legacy DatasetMixture API

For backward compatibility:

```python
from easydel.data import DatasetMixture, TextDatasetInform, build_dataset

mixture = DatasetMixture(
    informs=[
        TextDatasetInform(
            type="parquet",
            data_files="data1/*.parquet",
            content_field="text",
        ),
        TextDatasetInform(
            type="json",
            data_files="data2/*.json",
            content_field="content",
        ),
    ],
    mixture_weights={"ds1": 0.7, "ds2": 0.3},
    block_mixture=True,
    mixture_block_size=2048,
    streaming=True,
    seed=42,
)

dataset = build_dataset(mixture)
```

## Best Practices

### 1. Choose Block Size Carefully

```python
# Smaller blocks = more interleaving but more overhead
mixed = block_mixture_interleave(..., block_size=100)

# Larger blocks = less overhead but more bursty
mixed = block_mixture_interleave(..., block_size=10000)

# Recommended: 1000-2000 for good balance
mixed = block_mixture_interleave(..., block_size=1000)
```

### 2. Use Dynamic Scheduling for Curriculum

```python
# Start with easier data, gradually increase difficulty
scheduler = WeightScheduler(
    schedule=[
        WeightSchedulePoint(step=0, weights={"easy": 0.9, "hard": 0.1}),
        WeightSchedulePoint(step=50000, weights={"easy": 0.5, "hard": 0.5}),
        WeightSchedulePoint(step=100000, weights={"easy": 0.1, "hard": 0.9}),
    ],
    interpolation="cosine",  # Smooth transition
)
```

### 3. Handle Imbalanced Datasets

```python
# Small dataset with important data
scheduler = WeightScheduler(
    schedule=[
        # High weight initially to ensure exposure
        WeightSchedulePoint(step=0, weights={"small_important": 0.5, "large": 0.5}),
        # Reduce after sufficient exposure
        WeightSchedulePoint(step=10000, weights={"small_important": 0.1, "large": 0.9}),
    ],
)
```

### 4. Track Data Source

```python
mixed = MixedShardedSource(sources, ...)

for example in mixed.open_shard(mixed.shard_names[0]):
    source_name = example.get("__source__")  # Added automatically
    if source_name == "code":
        # Apply code-specific processing
        pass
```

## Reproducibility

Block-based mixing is deterministic given the same seed:

```python
# Same seed = same sequence
mixed1 = block_mixture_interleave(datasets, seed=42)
mixed2 = block_mixture_interleave(datasets, seed=42)

# Different seeds = different sequences
mixed3 = block_mixture_interleave(datasets, seed=123)
```

For distributed training, ensure workers use consistent seeds:

```python
# All workers use same seed for consistent global ordering
mixed = MixedShardedSource(
    sources=sources,
    seed=42,  # Same across all workers
)
```

## Next Steps

- [Pre-tokenization](pretokenization.md) - Tokenize before mixing
- [Pipeline API](pipeline.md) - Full pipeline with mixing
- [Streaming](streaming.md) - Stream mixed datasets from cloud
