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

"""Data source implementations for various formats.

This module provides ShardedDataSource implementations for:
- Parquet files (with row-group level seeking)
- JSON/JSONL files
- Arrow IPC files
- CSV files
- Plain text files
- HuggingFace Hub datasets
- Composite sources (combining multiple sources)
"""

from .base import (
    ArrowShardedSource,
    CompositeShardedSource,
    CsvShardedSource,
    HuggingFaceShardedSource,
    JsonShardedSource,
    ParquetShardedSource,
    ParquetShardInfo,
    TextShardedSource,
    create_source,
    expand_data_files,
    load_for_inform,
)
from .hf_wrapper import (
    HFDatasetShardedSource,
    wrap_hf_dataset,
)

__all__ = [
    "ArrowShardedSource",
    "CompositeShardedSource",
    "CsvShardedSource",
    "HFDatasetShardedSource",
    "HuggingFaceShardedSource",
    "JsonShardedSource",
    "ParquetShardInfo",
    "ParquetShardedSource",
    "TextShardedSource",
    "create_source",
    "expand_data_files",
    "load_for_inform",
    "wrap_hf_dataset",
]
