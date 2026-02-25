# Data Sources

EasyData supports loading data from multiple sources with a unified `ShardedDataSource` interface.

## Supported Formats

| Format          | Class                      | Extensions        | Cloud Support |
| --------------- | -------------------------- | ----------------- | ------------- |
| Parquet         | `ParquetShardedSource`     | `.parquet`, `.pq` | GCS, S3, HTTP |
| JSON/JSONL      | `JsonShardedSource`        | `.json`, `.jsonl` | GCS, S3, HTTP |
| Arrow IPC       | `ArrowShardedSource`       | `.arrow`          | GCS, S3, HTTP |
| CSV             | `CsvShardedSource`         | `.csv`            | GCS, S3, HTTP |
| Plain Text      | `TextShardedSource`        | `.txt`            | GCS, S3, HTTP |
| HuggingFace Hub | `HuggingFaceShardedSource` | -                 | Native        |
| HF Dataset      | `HFDatasetShardedSource`   | -                 | -             |

## ParquetShardedSource

Best for large-scale training with efficient row-group level seeking.

```python
from easydel.data import ParquetShardedSource

# Local files with glob pattern
source = ParquetShardedSource("data/*.parquet")

# Google Cloud Storage
source = ParquetShardedSource(
    "gs://my-bucket/training/*.parquet",
    storage_options={"token": "cloud"},  # Default credentials
)

# AWS S3
source = ParquetShardedSource(
    "s3://my-bucket/data/*.parquet",
    storage_options={
        "key": "ACCESS_KEY_ID",
        "secret": "SECRET_ACCESS_KEY",
        "endpoint_url": "https://s3.amazonaws.com",  # Optional
    },
)

# Load specific columns only
source = ParquetShardedSource(
    "data/*.parquet",
    columns=["input_ids", "attention_mask"],  # Memory efficient
)

# Iteration
for shard_name in source.shard_names:
    for example in source.open_shard(shard_name):
        print(example)

# Resume from specific row (uses row-group metadata)
for example in source.open_shard_at_row("data/shard_0.parquet", row=10000):
    print(example)
```

### Parquet Features

- **Row-group level seeking**: Efficient resume without reading entire file
- **Column projection**: Load only needed columns
- **Automatic retry**: 3 retries with exponential backoff for cloud
- **Parallel shard access**: Distribute shards across workers

## JsonShardedSource

For JSON array files or JSONL (one JSON per line).

```python
from easydel.data import JsonShardedSource

# JSONL files (default)
source = JsonShardedSource("data/*.jsonl")

# JSON array files
source = JsonShardedSource("data/*.json", jsonl=False)

# GCS with credentials
source = JsonShardedSource(
    "gs://bucket/data/*.jsonl",
    storage_options={"token": "/path/to/service-account.json"},
)

# Iteration
for shard in source.shard_names:
    for record in source.open_shard(shard):
        print(record)
```

### JSON vs JSONL

**JSONL** (recommended for large datasets):

```json
{"text": "First example"}
{"text": "Second example"}
{"text": "Third example"}
```

**JSON array** (simpler but loads entire file):

```json
[
  {"text": "First example"},
  {"text": "Second example"},
  {"text": "Third example"}
]
```

## ArrowShardedSource

For Arrow IPC format files.

```python
from easydel.data import ArrowShardedSource

source = ArrowShardedSource("data/*.arrow")

# With cloud storage
source = ArrowShardedSource(
    "gs://bucket/arrow-data/*.arrow",
    storage_options={"token": "cloud"},
)
```

## CsvShardedSource

For CSV files with headers.

```python
from easydel.data import CsvShardedSource

# Standard CSV
source = CsvShardedSource("data/*.csv")

# TSV (tab-separated)
source = CsvShardedSource("data/*.tsv", delimiter="\t")

# Custom storage options
source = CsvShardedSource(
    "s3://bucket/data/*.csv",
    storage_options={"key": "...", "secret": "..."},
)
```

## TextShardedSource

For plain text files where each line becomes a record.

```python
from easydel.data import TextShardedSource

# Each line becomes {"text": "line content"}
source = TextShardedSource("data/*.txt")

# Custom field name
source = TextShardedSource(
    "data/*.txt",
    text_field="content",  # {"content": "line content"}
)
```

## HuggingFaceShardedSource

For streaming directly from HuggingFace Hub.

```python
from easydel.data import HuggingFaceShardedSource

# Basic usage
source = HuggingFaceShardedSource(
    dataset_name="HuggingFaceFW/fineweb",
    split="train",
    streaming=True,
)

# With subset/configuration
source = HuggingFaceShardedSource(
    dataset_name="bigcode/the-stack",
    split="train",
    subset="python",  # Language subset
    streaming=True,
    cache_dir="/data/hf_cache",
)

# Non-streaming (loads to memory)
source = HuggingFaceShardedSource(
    dataset_name="tatsu-lab/alpaca",
    split="train",
    streaming=False,  # Full download
)
```

## HFDatasetShardedSource

Wraps an existing HuggingFace Dataset as ShardedDataSource.

```python
from datasets import load_dataset
from easydel.data import HFDatasetShardedSource, wrap_hf_dataset

# Load HuggingFace dataset
hf_ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# Wrap as ShardedDataSource
source = HFDatasetShardedSource(hf_ds)

# Or use the convenience function
source = wrap_hf_dataset(hf_ds)

# Works with IterableDataset too
hf_streaming = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
source = wrap_hf_dataset(hf_streaming)
```

## CompositeShardedSource

Combine multiple sources of different types.

```python
from easydel.data import (
    CompositeShardedSource,
    ParquetShardedSource,
    JsonShardedSource,
)

# Combine different formats
parquet_source = ParquetShardedSource("data/parquet/*.parquet")
json_source = JsonShardedSource("data/json/*.jsonl")

combined = CompositeShardedSource([parquet_source, json_source])

# All shards from both sources
print(combined.shard_names)  # ['data/parquet/0.parquet', 'data/json/0.jsonl', ...]
```

## Auto-detection with create_source

Automatically detect and create appropriate source:

```python
from easydel.data import create_source, DatasetConfig

# From config
config = DatasetConfig(
    data_files="data/*.parquet",  # Auto-detects parquet
)
source = create_source(config)

# HuggingFace detection
config = DatasetConfig(
    data_files="trl-lib/ultrafeedback_binarized",
    type="huggingface",
    split="train",
)
source = create_source(config)
```

## Cloud Storage Authentication

### Google Cloud Storage (GCS)

```python
# Default credentials (recommended for GCE/Cloud Run)
storage_options = {"token": "cloud"}

# Service account JSON
storage_options = {"token": "/path/to/service-account.json"}

# Inline credentials
storage_options = {"token": {"type": "service_account", ...}}

source = ParquetShardedSource("gs://bucket/data/*.parquet", storage_options=storage_options)
```

### Amazon S3

```python
# Access keys
storage_options = {
    "key": "ACCESS_KEY_ID",
    "secret": "SECRET_ACCESS_KEY",
}

# With region
storage_options = {
    "key": "ACCESS_KEY_ID",
    "secret": "SECRET_ACCESS_KEY",
    "client_kwargs": {"region_name": "us-west-2"},
}

# Anonymous access (public buckets)
storage_options = {"anon": True}

source = ParquetShardedSource("s3://bucket/data/*.parquet", storage_options=storage_options)
```

### HTTP/HTTPS

```python
# Public URLs
source = ParquetShardedSource("https://example.com/data/*.parquet")

# With auth headers
storage_options = {
    "headers": {"Authorization": "Bearer TOKEN"},
}
source = JsonShardedSource("https://api.example.com/data.jsonl", storage_options=storage_options)
```

## Glob Patterns

All sources support glob patterns:

```python
# Single directory
source = ParquetShardedSource("data/*.parquet")

# Recursive
source = ParquetShardedSource("data/**/*.parquet")

# Multiple patterns
source = ParquetShardedSource(["data/2023/*.parquet", "data/2024/*.parquet"])

# Character ranges
source = JsonShardedSource("data/shard_[0-9].jsonl")
```

## Resumable Iteration

All sources support resuming from a specific position:

```python
# Get checkpoint
checkpoint = {"shard": "data/shard_5.parquet", "row": 12345}

# Resume iteration
source = ParquetShardedSource("data/*.parquet")
for example in source.open_shard_at_row(checkpoint["shard"], checkpoint["row"]):
    # Process example
    # Save new checkpoint periodically
    pass
```

## Performance Tips

1. **Use Parquet for large datasets**: Best compression and column projection
2. **Enable streaming for HuggingFace**: Avoids downloading entire dataset
3. **Column projection**: Load only needed columns with `columns=`
4. **Shard distribution**: Assign different shards to different workers
5. **Prefetch**: Use `AsyncDataLoader` for overlap with training

## Next Steps

- [Transforms](transforms.md) - Apply transformations to loaded data
- [Streaming](streaming.md) - Detailed streaming guide
- [Pipeline API](pipeline.md) - Chain sources with processing
