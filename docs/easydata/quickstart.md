# EasyData Quickstart

Get started with EasyData in 5 minutes.

## Basic Usage

### Loading a HuggingFace Dataset

```python
from datasets import load_dataset
import easydel as ed

# Load dataset from HuggingFace Hub
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# Use directly with trainer
trainer = ed.DPOTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    arguments=dpo_config,
)
trainer.train()
```

### Loading Local Files

```python
from easydel.data import JsonShardedSource, ParquetShardedSource

# JSON/JSONL files
source = JsonShardedSource("data/*.jsonl")

# Parquet files (supports GCS/S3)
source = ParquetShardedSource("gs://bucket/data/*.parquet")

# Iterate over data
for example in source.open_shard(source.shard_names[0]):
    print(example)
```

### Mixing Multiple Datasets

```python
from datasets import load_dataset
from easydel.data import block_mixture_interleave

# Load multiple datasets
code_ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
text_ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

# Mix with custom weights (dict for explicit mapping)
mixed = block_mixture_interleave(
    datasets={"code": code_ds, "text": text_ds},
    weights={"code": 0.3, "text": 0.7},
    block_size=1000,
    seed=42,
    stop="restart",  # Loop when exhausted
)

# Use with trainer
trainer = ed.SFTTrainer(train_dataset=mixed, ...)
```

## Common Patterns

### SFT Training with Chat Data

```python
from datasets import load_dataset
import easydel as ed

# Load conversational dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Trainer handles chat template application internally
trainer = ed.SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    arguments=ed.SFTConfig(
        max_sequence_length=2048,
        # Chat template applied automatically
    ),
)
```

### DPO with Preference Data

```python
from datasets import load_dataset
import easydel as ed

# Load preference dataset (chosen/rejected format)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# DPOTrainer handles preprocessing internally
trainer = ed.DPOTrainer(
    model=policy_model,
    reference_model=ref_model,
    train_dataset=dataset,  # Raw preference data
    processing_class=tokenizer,
    arguments=ed.DPOConfig(
        max_prompt_length=512,
        max_completion_length=512,
    ),
)
```

### GRPO with Reward Functions

```python
from datasets import load_dataset
import easydel as ed

# Load dataset with prompts
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

def my_reward_fn(prompts, completions, **kwargs):
    # Return list of reward scores
    return [1.0 if "correct" in c else 0.0 for c in completions]

trainer = ed.GRPOTrainer(
    model=model,
    reward_funcs=my_reward_fn,
    train_dataset=dataset,  # Prompts extracted automatically
    processing_class=tokenizer,
    arguments=ed.GRPOConfig(
        num_return_sequences=4,
        max_prompt_length=512,
        max_completion_length=512,
    ),
)
```

## Pre-tokenization Pipeline

For large-scale training, pre-tokenize and save:

```python
from easydel.data import (
    Pipeline,
    PipelineConfig,
    DatasetConfig,
    SaveStageConfig,
    TokenizeStageConfig,
)

# Configure pipeline
config = PipelineConfig(
    datasets=[
        DatasetConfig(
            data_files="data/*.jsonl",
            tokenizer="meta-llama/Llama-2-7b",
            content_field="text",
        )
    ],
    tokenize=TokenizeStageConfig(max_length=2048),
    save=SaveStageConfig(
        enabled=True,
        output_dir="./tokenized_data",
        format="parquet",
    ),
)

# Run pipeline
Pipeline.from_config(config).source().tokenize().save().build()

# Later: load pre-tokenized data
from easydel.data import ParquetShardedSource
source = ParquetShardedSource("./tokenized_data/*.parquet")
```

## Streaming from Cloud

```python
from easydel.data import ParquetShardedSource

# GCS with automatic retry
source = ParquetShardedSource(
    "gs://my-bucket/training-data/*.parquet",
    storage_options={"token": "cloud"},  # Uses default credentials
)

# S3
source = ParquetShardedSource(
    "s3://my-bucket/data/*.parquet",
    storage_options={
        "key": "ACCESS_KEY",
        "secret": "SECRET_KEY",
    },
)

# Iterate with automatic retry on failures
for shard in source.shard_names:
    for example in source.open_shard(shard):
        process(example)
```

## Token Packing

Pack sequences for efficient training:

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
            data_files="./tokenized/*.parquet",
            type="parquet",
        )
    ],
    pack=PackStageConfig(
        enabled=True,
        seq_length=2048,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        strategy="greedy",  # or "pool", "first_fit"
        include_segment_ids=True,
    ),
)

pipeline = Pipeline.from_config(config)
for batch in pipeline.source().pack().load().build():
    # batch contains packed sequences with segment_ids
    train_step(batch)
```

## Dynamic Weight Scheduling

Change dataset mix ratios during training:

```python
from easydel.data import (
    MixedShardedSource,
    HFDatasetShardedSource,
    WeightScheduler,
    WeightSchedulePoint,
)

# Create sources
code_source = HFDatasetShardedSource(code_dataset)
text_source = HFDatasetShardedSource(text_dataset)

# Define schedule: shift from text-heavy to code-heavy
scheduler = WeightScheduler(
    schedule=[
        WeightSchedulePoint(step=0, weights={"code": 0.2, "text": 0.8}),
        WeightSchedulePoint(step=10000, weights={"code": 0.5, "text": 0.5}),
        WeightSchedulePoint(step=50000, weights={"code": 0.8, "text": 0.2}),
    ],
    interpolation="linear",  # Smooth transition
)

# Create mixed source
mixed = MixedShardedSource(
    sources={"code": code_source, "text": text_source},
    weight_scheduler=scheduler,
    block_size=1000,
)
```

## Next Steps

- [Data Sources](sources.md) - All supported data formats
- [Transforms](transforms.md) - Tokenization and preprocessing
- [Dataset Mixing](mixing.md) - Advanced mixing strategies
- [Pipeline API](pipeline.md) - Full pipeline reference
