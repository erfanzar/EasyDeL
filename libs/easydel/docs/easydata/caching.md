# Caching

EasyData provides a multi-layer caching system inspired by Levanter's TreeCache for efficient data processing.

## Overview

The caching system provides:

- **Memory Cache**: Fast LRU cache for recently accessed data
- **Disk Cache**: Persistent cache with compression and expiry
- **TreeCacheManager**: Hierarchical combination of both layers
- **Dataset Cache**: Specialized cache for HuggingFace datasets

## TreeCacheManager

The main caching interface combining memory and disk layers:

```python
from easydel.data import TreeCacheManager

# Create cache manager
cache = TreeCacheManager(
    cache_dir="./cache",
    memory_size=100,      # Max items in memory
    disk_expiry=86400,    # 24 hours
    compression="zstd",   # none, gzip, lz4, zstd
)

# Store data
cache.put("my_key", {"input_ids": [1, 2, 3]})

# Retrieve data
result = cache.get("my_key")
if result:
    data, metadata = result
    print(data)

# Check existence
if cache.contains("my_key"):
    print("Cached!")

# Invalidate
cache.invalidate("my_key")  # Single key
cache.invalidate()          # Clear all
```

## Cache-on-Compute Pattern

Use `get_or_compute` for lazy caching:

```python
from easydel.data import TreeCacheManager, CacheMetadata

cache = TreeCacheManager("./cache")

def expensive_tokenization(text):
    # Expensive operation
    return tokenizer(text)["input_ids"]

# Will compute and cache if not exists
result = cache.get_or_compute(
    key="text_123",
    compute_fn=lambda: expensive_tokenization("Hello world"),
    metadata=CacheMetadata(
        source_hash="abc123",
        tokenizer_hash="llama2",
    ),
)
```

## Cache Keys

Generate consistent cache keys from configuration:

```python
from easydel.data import TreeCacheManager

# From config dictionary
key = TreeCacheManager.compute_key(
    config={
        "tokenizer": "meta-llama/Llama-2-7b",
        "max_length": 2048,
        "data_file": "data.jsonl",
    },
    prefix="tokenized",
)
# "tokenized_a1b2c3d4e5f6"

# With content hash
key = TreeCacheManager.compute_key(
    config={"tokenizer": "llama"},
    prefix="example",
    include_content_hash=True,
    content="Hello world",
)
# "example_a1b2c3d4_e5f6g7h8"
```

## CacheMetadata

Track cache validity:

```python
from easydel.data import CacheMetadata

metadata = CacheMetadata(
    version="1.0",
    source_hash="abc123",         # Hash of source data
    tokenizer_hash="llama2_7b",   # Hash of tokenizer
    transform_hash="v2",          # Hash of transforms
    num_examples=100000,
    config_hash="xyz789",
    extra={"custom": "data"},
)

# Check validity
if metadata.is_valid_for(config_hash="xyz789", source_hash="abc123"):
    print("Cache is valid")
```

## Memory Cache (LRU)

Fast in-memory cache with automatic eviction:

```python
from easydel.data import MemoryCache

cache = MemoryCache(max_size=1000)

# Basic operations
cache.put("key1", {"data": "value"})
result = cache.get("key1")  # Returns (data, metadata) or None

# Statistics
stats = cache.stats
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Size: {stats['size']}/{stats['max_size']}")
```

## Disk Cache

Persistent cache with compression:

```python
from easydel.data import DiskCache

cache = DiskCache(
    cache_dir="./disk_cache",
    compression="zstd",      # Best compression ratio
    expiry_seconds=86400,    # Auto-expire after 24 hours
)

# Operations
cache.put("key", large_data)
result = cache.get("key")  # Returns (data, metadata) or None

# Manual expiry check
if cache.contains("key"):  # Also checks expiry
    result = cache.get("key")
```

### Compression Options

| Compression | Speed   | Ratio | Use Case               |
| ----------- | ------- | ----- | ---------------------- |
| `none`      | Fastest | 1.0x  | SSDs, small data       |
| `gzip`      | Slow    | ~3x   | Maximum compatibility  |
| `lz4`       | Fast    | ~2x   | Balanced (recommended) |
| `zstd`      | Medium  | ~3.5x | Best ratio             |

## Dataset Cache

Specialized cache for HuggingFace datasets:

```python
from easydel.data import DatasetCache
from datasets import Dataset

cache = DatasetCache("./dataset_cache")

# Cache a dataset
dataset = Dataset.from_dict({"text": ["hello", "world"]})
cache.put("my_dataset", dataset)

# Load from cache
cached_dataset = cache.get("my_dataset")

# Check and invalidate
if cache.contains("my_dataset"):
    cache.invalidate("my_dataset")
```

## Pipeline Integration

### CacheStageConfig

Configure caching in pipelines:

```python
from easydel.data import PipelineConfig, CacheStageConfig

config = PipelineConfig(
    datasets=[...],
    cache=CacheStageConfig(
        enabled=True,
        cache_type="hierarchical",  # memory, disk, hierarchical
        cache_dir=".cache/easydel_pipeline",
        memory_cache_size=100,
        disk_cache_expiry=86400,
        compression="lz4",
        hash_fn="combined",  # content, path, combined
    ),
)
```

### Per-Dataset Caching

```python
from easydel.data import DatasetConfig

config = DatasetConfig(
    data_files="data/*.parquet",
    cache_path="./cache/my_dataset",  # Dataset-specific cache
    cache_enabled=True,
)
```

## Caching Strategies

### 1. Tokenization Cache

Cache tokenized data to avoid re-tokenization:

```python
from easydel.data import TreeCacheManager, CacheMetadata
from transformers import AutoTokenizer

cache = TreeCacheManager("./tokenizer_cache")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

def tokenize_with_cache(text: str, text_id: str):
    key = f"tokenized_{text_id}"

    result = cache.get(key)
    if result:
        return result[0]

    tokenized = tokenizer(text, max_length=2048, truncation=True)
    cache.put(
        key,
        dict(tokenized),
        metadata=CacheMetadata(
            tokenizer_hash="llama2_7b",
            num_examples=1,
        ),
    )
    return tokenized
```

### 2. Processed Dataset Cache

Cache entire processed datasets:

```python
from easydel.data import DatasetCache

cache = DatasetCache("./processed_datasets")

def get_processed_dataset(name: str, process_fn):
    # Check cache
    cached = cache.get(name)
    if cached:
        print(f"Loaded {name} from cache")
        return cached

    # Process and cache
    print(f"Processing {name}...")
    dataset = process_fn()
    cache.put(name, dataset)
    return dataset
```

### 3. Checkpoint-Aware Caching

Include training state in cache key:

```python
from easydel.data import TreeCacheManager

cache = TreeCacheManager("./training_cache")

def get_cached_batch(step: int, shard: str, row: int):
    key = f"batch_{shard}_{row}"

    result = cache.get(key)
    if result:
        return result[0]

    # Compute batch
    batch = load_and_process_batch(shard, row)
    cache.put(key, batch)
    return batch
```

## Cache Invalidation

### Automatic Invalidation

```python
from easydel.data import TreeCacheManager, CacheMetadata

cache = TreeCacheManager("./cache")

def get_with_validation(key: str, expected_config_hash: str):
    result = cache.get_or_compute(
        key=key,
        compute_fn=lambda: compute_data(),
        metadata=CacheMetadata(config_hash=expected_config_hash),
        validate_fn=lambda meta: meta.config_hash == expected_config_hash,
    )
    return result
```

### Manual Invalidation

```python
# Single key
cache.invalidate("outdated_key")

# Pattern-based (iterate and invalidate)
for key in ["key1", "key2", "key3"]:
    if cache.contains(key):
        cache.invalidate(key)

# Clear all
cache.invalidate()
```

### Expiry-Based

```python
from easydel.data import DiskCache

# Auto-expire after 1 hour
cache = DiskCache(
    cache_dir="./cache",
    expiry_seconds=3600,
)

# Old entries automatically invalidated on access
```

## Best Practices

### 1. Use Appropriate Cache Location

```python
# Fast SSD for frequently accessed data
cache = TreeCacheManager("/nvme/cache")

# Network storage for shared cache
cache = TreeCacheManager("/shared/cache")

# Memory-only for ephemeral data
from easydel.data import MemoryCache
cache = MemoryCache(max_size=1000)
```

### 2. Include Version in Cache Keys

```python
CACHE_VERSION = "v2"

key = TreeCacheManager.compute_key(
    config={"tokenizer": "llama", "version": CACHE_VERSION},
)
```

### 3. Monitor Cache Statistics

```python
cache = TreeCacheManager("./cache")

# Periodically log stats
stats = cache.stats
print(f"Memory: {stats['memory']}")
print(f"Disk: {stats['disk']}")

# Check hit rates
memory_hits = stats['memory']['hit_rate']
if memory_hits < 0.5:
    print("Consider increasing memory cache size")
```

### 4. Clean Up Old Caches

```python
import shutil
from pathlib import Path

def cleanup_old_caches(cache_dir: str, max_age_days: int = 7):
    import time

    cache_path = Path(cache_dir)
    now = time.time()
    max_age_seconds = max_age_days * 86400

    for item in cache_path.iterdir():
        if item.is_dir():
            age = now - item.stat().st_mtime
            if age > max_age_seconds:
                shutil.rmtree(item)
                print(f"Removed old cache: {item}")
```

## Troubleshooting

### Cache Not Being Used

```python
# Check cache contains expected key
key = TreeCacheManager.compute_key(config)
print(f"Looking for key: {key}")
print(f"Cache contains: {cache.contains(key)}")

# Verify metadata matches
result = cache.get(key)
if result:
    data, meta = result
    print(f"Cached metadata: {meta}")
```

### Disk Space Issues

```python
# Set size limit
cache = DiskCache(
    cache_dir="./cache",
    expiry_seconds=3600,  # Short expiry
)

# Or use memory-only
cache = MemoryCache(max_size=100)
```

### Stale Cache

```python
# Always include version/hash in metadata
metadata = CacheMetadata(
    config_hash=compute_config_hash(),
    tokenizer_hash=compute_tokenizer_hash(),
)

# Use validation function
cache.get_or_compute(
    key=key,
    compute_fn=compute,
    validate_fn=lambda m: m.config_hash == current_hash,
)
```

## Next Steps

- [Pipeline API](pipeline.md) - Use caching in pipelines
- [Streaming](streaming.md) - Cache streamed data
- [Trainer Integration](trainer_integration.md) - Caching with trainers
