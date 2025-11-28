# Streaming Data

Streaming enables training on datasets too large to fit in memory. EasyData supports streaming from local files, cloud storage, and HuggingFace Hub.

## Why Streaming?

| Approach | Memory Usage | Startup Time | Best For |
|----------|--------------|--------------|----------|
| Full Load | ~Dataset Size | Slow | Small datasets (<10GB) |
| Streaming | ~Batch Size | Fast | Large datasets (>10GB) |

## HuggingFace Hub Streaming

### Basic Streaming

```python
from datasets import load_dataset

# Stream from HuggingFace Hub (never downloads full dataset)
dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    split="train",
    streaming=True,  # Key: enables streaming
)

# Use with trainer
trainer = ed.SFTTrainer(
    train_dataset=dataset,
    ...
)
```

### With HuggingFaceShardedSource

```python
from easydel.data import HuggingFaceShardedSource

source = HuggingFaceShardedSource(
    dataset_name="HuggingFaceFW/fineweb",
    split="train",
    streaming=True,
    cache_dir="/data/hf_cache",  # Optional local cache
)

# Iterate without loading to memory
for example in source.open_shard(source.shard_names[0]):
    process(example)
```

### Streaming with Subsets

```python
from easydel.data import HuggingFaceShardedSource

# The Stack with specific language
source = HuggingFaceShardedSource(
    dataset_name="bigcode/the-stack",
    split="train",
    subset="python",  # Only Python code
    streaming=True,
)

# C4 with specific configuration
source = HuggingFaceShardedSource(
    dataset_name="allenai/c4",
    split="train",
    subset="en",  # English only
    streaming=True,
)
```

## Cloud Storage Streaming

### Google Cloud Storage (GCS)

```python
from easydel.data import ParquetShardedSource

# Stream from GCS
source = ParquetShardedSource(
    "gs://my-bucket/training-data/*.parquet",
    storage_options={"token": "cloud"},  # Uses default credentials
)

# With service account
source = ParquetShardedSource(
    "gs://my-bucket/data/*.parquet",
    storage_options={"token": "/path/to/service-account.json"},
)

# Column projection for efficiency
source = ParquetShardedSource(
    "gs://my-bucket/data/*.parquet",
    storage_options={"token": "cloud"},
    columns=["input_ids", "attention_mask"],  # Only load needed columns
)
```

### Amazon S3

```python
from easydel.data import ParquetShardedSource

# Stream from S3
source = ParquetShardedSource(
    "s3://my-bucket/data/*.parquet",
    storage_options={
        "key": "ACCESS_KEY_ID",
        "secret": "SECRET_ACCESS_KEY",
        "client_kwargs": {"region_name": "us-west-2"},
    },
)

# Public bucket (anonymous access)
source = ParquetShardedSource(
    "s3://public-bucket/data/*.parquet",
    storage_options={"anon": True},
)
```

### Azure Blob Storage

```python
from easydel.data import ParquetShardedSource

source = ParquetShardedSource(
    "az://container/data/*.parquet",
    storage_options={
        "account_name": "myaccount",
        "account_key": "mykey",
    },
)
```

## JSON/JSONL Streaming

```python
from easydel.data import JsonShardedSource

# Stream JSONL from GCS
source = JsonShardedSource(
    "gs://bucket/data/*.jsonl",
    storage_options={"token": "cloud"},
)

# Stream large JSON files
source = JsonShardedSource(
    "gs://bucket/data/*.json",
    jsonl=False,  # JSON array format
    storage_options={"token": "cloud"},
)
```

## Automatic Retry

All cloud sources have automatic retry for transient failures:

```python
from easydel.data import ParquetShardedSource

# Built-in: 3 retries with exponential backoff
source = ParquetShardedSource("gs://bucket/data/*.parquet")

# Retry happens automatically on network errors
for shard in source.shard_names:
    for example in source.open_shard(shard):  # Retries on failure
        process(example)
```

## Prefetching

Overlap data loading with training using async data loader:

```python
from easydel.data import ParquetShardedSource, AsyncDataLoader

source = ParquetShardedSource("gs://bucket/data/*.parquet")

loader = AsyncDataLoader(
    source=source,
    batch_size=8,
    prefetch_enabled=True,
    prefetch_workers=2,
    prefetch_buffer_size=4,  # Batches to prefetch
)

# Training loop - data arrives while computing
for batch in loader:
    train_step(batch)  # Next batch loading in background
```

## Shuffling Streamed Data

Streaming prevents global shuffle, but buffer shuffle provides randomization:

```python
from datasets import load_dataset

# HuggingFace streaming shuffle
dataset = load_dataset("dataset", streaming=True)
dataset = dataset.shuffle(buffer_size=10000, seed=42)
```

```python
from easydel.data import AsyncDataLoader

# EasyData shuffle during loading
loader = AsyncDataLoader(
    source=source,
    shuffle_buffer_size=10000,  # Examples to buffer for shuffle
    seed=42,
)
```

### Shuffle Buffer Considerations

| Buffer Size | Memory | Randomization Quality |
|-------------|--------|----------------------|
| 1000 | ~10MB | Low (local patterns remain) |
| 10000 | ~100MB | Medium (good for most cases) |
| 100000 | ~1GB | High (nearly global shuffle) |

## Resumable Streaming

All sources support resuming from checkpoints:

```python
from easydel.data import ParquetShardedSource

source = ParquetShardedSource("gs://bucket/data/*.parquet")

# Save checkpoint
checkpoint = {
    "shard": current_shard,
    "row": current_row,
}

# Resume from checkpoint
for example in source.open_shard_at_row(checkpoint["shard"], checkpoint["row"]):
    process(example)
    # Update checkpoint periodically
```

### Parquet Row-Group Resume

Parquet sources use row-group metadata for efficient seeking:

```python
# Parquet file with 100 row groups of 10000 rows each
# Resume at row 550000 = row group 55, row 0 in group

source = ParquetShardedSource("data.parquet")
# Automatically seeks to correct row group
for example in source.open_shard_at_row("data.parquet", row=550000):
    process(example)
```

## Distributed Streaming

Assign different shards to different workers:

```python
from easydel.data import ParquetShardedSource

source = ParquetShardedSource("gs://bucket/data/*.parquet")

# Get all shard names
all_shards = source.shard_names

# Assign to workers
worker_id = 0
num_workers = 8
worker_shards = all_shards[worker_id::num_workers]

# Each worker processes its shards
for shard in worker_shards:
    for example in source.open_shard(shard):
        process(example)
```

## Streaming with Transforms

Apply transforms while streaming:

```python
from easydel.data import (
    HuggingFaceShardedSource,
    TransformedShardedSource,
    ChatTemplateTransform,
)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Stream from HuggingFace
source = HuggingFaceShardedSource(
    "tatsu-lab/alpaca",
    streaming=True,
)

# Apply chat template on the fly
transformed = TransformedShardedSource(
    source,
    transform=ChatTemplateTransform(tokenizer),
)

# Memory-efficient: transforms applied during iteration
for example in transformed.open_shard(transformed.shard_names[0]):
    print(example["text"])
```

## Streaming Pipeline

Full pipeline with streaming:

```python
from easydel.data import (
    Pipeline,
    PipelineConfig,
    DatasetConfig,
    LoadStageConfig,
)

config = PipelineConfig(
    datasets=[
        DatasetConfig(
            data_files="gs://bucket/data/*.parquet",
            type="parquet",
        ),
    ],
    streaming=True,  # Enable streaming mode
    load=LoadStageConfig(
        batch_size=8,
        prefetch_enabled=True,
        prefetch_buffer_size=4,
        shuffle_buffer_size=10000,
    ),
)

pipeline = Pipeline.from_config(config)
for batch in pipeline.source().load().build():
    train_step(batch)
```

## Performance Optimization

### 1. Use Parquet with Column Projection

```python
# Only load needed columns
source = ParquetShardedSource(
    "gs://bucket/data/*.parquet",
    columns=["input_ids", "attention_mask"],  # Skip unused columns
)
```

### 2. Optimal Shard Size

```md
Ideal shard size: 100MB - 500MB
- Too small: Too many files, high overhead
- Too large: Slow resume, high memory for seeking
```

### 3. Increase Prefetch Buffer

```python
loader = AsyncDataLoader(
    source=source,
    prefetch_buffer_size=8,  # More batches in flight
)
```

### 4. Use Fast Storage for Cache

```python
source = HuggingFaceShardedSource(
    "dataset",
    streaming=True,
    cache_dir="/nvme/hf_cache",  # Fast local SSD
)
```

## Monitoring Streaming

```python
import time
from easydel.data import ParquetShardedSource

source = ParquetShardedSource("gs://bucket/data/*.parquet")

start = time.time()
count = 0
for shard in source.shard_names:
    for example in source.open_shard(shard):
        count += 1
        if count % 10000 == 0:
            elapsed = time.time() - start
            rate = count / elapsed
            print(f"Processed {count} examples, {rate:.1f} ex/sec")
```

## Troubleshooting

### Slow Streaming from Cloud

```python
# Check: Are you using column projection?
source = ParquetShardedSource(
    "gs://bucket/*.parquet",
    columns=["input_ids"],  # Add this
)

# Check: Is prefetching enabled?
loader = AsyncDataLoader(
    source=source,
    prefetch_enabled=True,  # Should be True
)
```

### Memory Growing During Streaming

```python
# Check: Is shuffle buffer too large?
loader = AsyncDataLoader(
    source=source,
    shuffle_buffer_size=10000,  # Reduce if OOM
)

# Check: Are you holding references?
for example in source.open_shard(shard):
    process(example)  # Don't store examples
```

### Network Errors

```python
# Automatic retry handles most issues
# For persistent errors, check:
# 1. Credentials valid?
# 2. Network connectivity?
# 3. Rate limiting?
```

## Next Steps

- [Dataset Mixing](mixing.md) - Mix streamed datasets
- [Caching](caching.md) - Cache streamed data
- [Trainer Integration](trainer_integration.md) - Use with trainers
