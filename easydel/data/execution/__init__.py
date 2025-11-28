# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Execution layer: loading, saving, caching, and pipeline orchestration."""

from .cache import (
    CacheLayer,
    CacheMetadata,
    DatasetCache,
    DiskCache,
    MemoryCache,
    TreeCacheManager,
)
from .loader import (
    AsyncDataLoader,
    LoadStage,
    PrefetchIterator,
    ShardingSpec,
    batch_iterator,
    collate_batch,
    create_data_iterator,
    preshard_batch,
)
from .pipeline import (
    Pipeline,
    build_dataset,
    create_pipeline,
    pretokenize,
    tokenize_and_save,
)
from .save import (
    ArrowWriter,
    DatasetWriter,
    JsonlWriter,
    ParquetWriter,
    SaveStage,
    WriteStats,
    create_writer,
    parse_size,
    save_dataset,
    save_iterator,
)

__all__ = [
    # Save
    "ArrowWriter",
    # Loader
    "AsyncDataLoader",
    # Cache
    "CacheLayer",
    "CacheMetadata",
    "DatasetCache",
    "DatasetWriter",
    "DiskCache",
    "JsonlWriter",
    "LoadStage",
    "MemoryCache",
    "ParquetWriter",
    # Pipeline
    "Pipeline",
    "PrefetchIterator",
    "SaveStage",
    "ShardingSpec",
    "TreeCacheManager",
    "WriteStats",
    "batch_iterator",
    "build_dataset",
    "collate_batch",
    "create_data_iterator",
    "create_pipeline",
    "create_writer",
    "parse_size",
    "preshard_batch",
    "pretokenize",
    "save_dataset",
    "save_iterator",
    "tokenize_and_save",
]
