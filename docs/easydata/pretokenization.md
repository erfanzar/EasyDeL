# Pre-tokenization

Pre-tokenizing datasets before training provides significant benefits:

- **Faster training startup**: No tokenization during training
- **Reduced compute costs**: Tokenize once, train many times
- **Better disk utilization**: Store efficient binary formats
- **Easier debugging**: Inspect tokenized data offline

## Quick Start

### Using tokenize_and_save

The simplest way to pre-tokenize:

```python
from easydel.data import tokenize_and_save

# Tokenize and save in one call
tokenize_and_save(
    data_files="data/*.jsonl",
    tokenizer="meta-llama/Llama-2-7b",
    output_path="./tokenized_data",
    output_format="parquet",  # parquet, arrow, or jsonl
    max_length=2048,
)

# Later: load pre-tokenized data
from easydel.data import ParquetShardedSource
source = ParquetShardedSource("./tokenized_data/*.parquet")
```

### Using Pipeline API

For more control:

```python
from easydel.data import (
    Pipeline,
    PipelineConfig,
    DatasetConfig,
    TokenizeStageConfig,
    SaveStageConfig,
)

config = PipelineConfig(
    datasets=[
        DatasetConfig(
            data_files="data/*.jsonl",
            tokenizer="meta-llama/Llama-2-7b",
            content_field="text",
            save_path="./tokenized/dataset1",
        ),
        DatasetConfig(
            data_files="gs://bucket/data/*.parquet",
            tokenizer="meta-llama/Llama-2-7b",
            content_field="content",
            save_path="./tokenized/dataset2",
        ),
    ],
    tokenize=TokenizeStageConfig(
        max_length=2048,
        batch_size=1000,  # Batch tokenization for speed
    ),
    save=SaveStageConfig(
        enabled=True,
        format="parquet",
        compression="zstd",  # Optional compression
    ),
)

# Run tokenization pipeline
Pipeline.from_config(config).source().tokenize().save().build()
```

## Per-Dataset Configuration

Each dataset can have its own tokenizer and settings:

```python
config = PipelineConfig(
    datasets=[
        DatasetConfig(
            name="code",
            data_files="code_data/*.jsonl",
            tokenizer="bigcode/starcoder",  # Code tokenizer
            tokenizer_kwargs={"use_fast": True},
            content_field="code",
            save_path="./tokenized/code",
        ),
        DatasetConfig(
            name="text",
            data_files="text_data/*.parquet",
            tokenizer="meta-llama/Llama-2-7b",  # Text tokenizer
            content_field="text",
            save_path="./tokenized/text",
        ),
    ],
)
```

## TokenizerConfig Options

```python
from easydel.data import DatasetConfig, TokenizerConfig

config = DatasetConfig(
    data_files="data/*.jsonl",
    tokenizer=TokenizerConfig(
        name_or_path="meta-llama/Llama-2-7b",
        max_length=2048,
        truncation=True,
        padding=False,  # or "max_length", "longest"
        add_special_tokens=True,
        return_attention_mask=True,
        trust_remote_code=True,
    ),
)
```

## Output Formats

### Parquet (Recommended)

Best for large-scale training:

```python
SaveStageConfig(
    format="parquet",
    compression="zstd",  # Good compression ratio
    num_shards=100,      # Split into shards
)
```

**Advantages:**

- Efficient column access
- Row-group level seeking for resume
- Wide tool support (pandas, DuckDB, etc.)

### Arrow IPC

For maximum read speed:

```python
SaveStageConfig(
    format="arrow",
    num_shards=50,
)
```

**Advantages:**

- Zero-copy reads
- Fastest format for iteration

### JSONL

For debugging and portability:

```python
SaveStageConfig(
    format="jsonl",
    compression="gzip",
)
```

**Advantages:**

- Human readable
- Easy to inspect

## Processing Chat Data

For conversational datasets, apply chat template before tokenization:

```python
from easydel.data import (
    Pipeline,
    PipelineConfig,
    DatasetConfig,
    ChatTemplateTransform,
)

config = PipelineConfig(
    datasets=[
        DatasetConfig(
            data_files="conversations/*.jsonl",
            tokenizer="meta-llama/Llama-2-7b-chat-hf",
            # Chat template applied automatically if messages field exists
        ),
    ],
)

# Or manually with transform
from transformers import AutoTokenizer
from easydel.data import JsonShardedSource, TransformedShardedSource

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
source = JsonShardedSource("conversations/*.jsonl")
transformed = TransformedShardedSource(
    source,
    transform=ChatTemplateTransform(tokenizer),
)
```

## Large-Scale Tokenization

### Parallel Processing

```python
from easydel.data import DatasetConfig

config = DatasetConfig(
    data_files="data/*.parquet",
    tokenizer="meta-llama/Llama-2-7b",
)

# Pipeline uses multiple workers automatically
```

### Distributed with Ray

```python
from easydel.data import PipelineConfig, RayConfig

config = PipelineConfig(
    datasets=[...],
    ray=RayConfig(
        enabled=True,
        num_workers=8,
        resources_per_worker={"CPU": 2},
    ),
)
```

### Cloud Storage

```python
# Read from GCS, write to GCS
config = PipelineConfig(
    datasets=[
        DatasetConfig(
            data_files="gs://input-bucket/data/*.parquet",
            tokenizer="meta-llama/Llama-2-7b",
            save_path="gs://output-bucket/tokenized",
        ),
    ],
    save=SaveStageConfig(enabled=True, format="parquet"),
)
```

## Token Packing

Combine tokenization with packing:

```python
from easydel.data import (
    PipelineConfig,
    DatasetConfig,
    PackStageConfig,
    SaveStageConfig,
)

config = PipelineConfig(
    datasets=[
        DatasetConfig(
            data_files="data/*.jsonl",
            tokenizer="meta-llama/Llama-2-7b",
        ),
    ],
    pack=PackStageConfig(
        enabled=True,
        seq_length=2048,
        eos_token_id=2,
        pad_token_id=0,
        strategy="greedy",  # or "pool", "first_fit"
        include_segment_ids=True,
    ),
    save=SaveStageConfig(
        enabled=True,
        output_dir="./packed_data",
        format="parquet",
    ),
)

Pipeline.from_config(config).source().tokenize().pack().save().build()
```

## Loading Pre-tokenized Data

### With Trainer

```python
from easydel.data import ParquetShardedSource
import easydel as ed

# Load pre-tokenized data
source = ParquetShardedSource("./tokenized/*.parquet")

# Use with trainer (wraps automatically)
trainer = ed.SFTTrainer(
    model=model,
    train_dataset=source,  # Works directly
    processing_class=tokenizer,
    arguments=ed.SFTConfig(...),
)
```

### Manual Iteration

```python
from easydel.data import ParquetShardedSource

source = ParquetShardedSource("./tokenized/*.parquet")

for shard in source.shard_names:
    for example in source.open_shard(shard):
        input_ids = example["input_ids"]
        attention_mask = example["attention_mask"]
        # Process...
```

### With AsyncDataLoader

```python
from easydel.data import ParquetShardedSource, AsyncDataLoader

source = ParquetShardedSource("./tokenized/*.parquet")

loader = AsyncDataLoader(
    source=source,
    batch_size=8,
    prefetch_enabled=True,
    prefetch_buffer_size=4,
)

for batch in loader:
    # batch["input_ids"] is numpy array with shape (batch_size, seq_len)
    train_step(batch)
```

## Verifying Tokenized Data

### Check format

```python
import pyarrow.parquet as pq

# Check schema
pf = pq.ParquetFile("./tokenized/shard_0.parquet")
print(pf.schema)
# input_ids: list<int32>
# attention_mask: list<int32>

# Check sample
table = pf.read_row_group(0)
print(table.to_pandas().head())
```

### Decode and verify

```python
from transformers import AutoTokenizer
from easydel.data import ParquetShardedSource

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
source = ParquetShardedSource("./tokenized/*.parquet")

# Check first few examples
for i, example in enumerate(source.open_shard(source.shard_names[0])):
    if i >= 3:
        break
    decoded = tokenizer.decode(example["input_ids"])
    print(f"Example {i}: {decoded[:100]}...")
```

## Best Practices

### 1. Version Your Tokenized Data

```md
tokenized/
├── v1_llama2_2048/
│   ├── config.json  # Save tokenizer + settings
│   └── *.parquet
├── v2_llama2_4096/
│   ├── config.json
│   └── *.parquet
```

### 2. Save Metadata

```python
import json

metadata = {
    "tokenizer": "meta-llama/Llama-2-7b",
    "max_length": 2048,
    "truncation": True,
    "num_examples": 1000000,
    "created_at": "2024-01-15",
}

with open("./tokenized/metadata.json", "w") as f:
    json.dump(metadata, f)
```

### 3. Use Appropriate Shard Sizes

```python
# ~500MB per shard for cloud storage
SaveStageConfig(
    max_shard_size="500MB",
)

# Or explicit shard count
SaveStageConfig(
    num_shards=100,
)
```

### 4. Validate Before Training

```python
# Quick validation script
from easydel.data import ParquetShardedSource

source = ParquetShardedSource("./tokenized/*.parquet")

# Check all shards readable
for shard in source.shard_names:
    count = 0
    for example in source.open_shard(shard):
        count += 1
        if count >= 10:  # Just check first 10
            break
    print(f"{shard}: OK ({count}+ examples)")
```

## Next Steps

- [Streaming](streaming.md) - Stream pre-tokenized data
- [Dataset Mixing](mixing.md) - Mix pre-tokenized datasets
- [Caching](caching.md) - Cache processed data
