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

"""Data management utilities for EasyDeL.

This module provides comprehensive tools for loading, mixing, and managing datasets
for training and evaluation. It supports both text and visual datasets with flexible
mixing strategies, streaming capabilities, and efficient data processing pipelines.

Key Features:
    - Flexible dataset loading from HuggingFace Hub and local files
    - Multiple dataset formats support (JSON, Parquet, CSV, Arrow, TSV, TXT)
    - Dataset mixing with custom ratios and strategies
    - Streaming and non-streaming modes
    - Token packing for efficient training
    - Automatic preprocessing and tokenization
    - Block-deterministic mixing for reproducible training

Example:
    >>> from easydel.utils.data_managers import DatasetMixture, TextDatasetInform
    >>> from easydel.utils.data_managers import build_dataset
    >>>
    >>> # Single dataset from HuggingFace Hub
    >>> inform = TextDatasetInform(
    ...     type="huggingface",
    ...     data_files="squad",
    ...     content_field="context"
    ... )
    >>>
    >>> # Mixed datasets with custom ratios
    >>> mixture = DatasetMixture(
    ...     informs=[
    ...         TextDatasetInform(type="json", data_files="data1.json"),
    ...         TextDatasetInform(type="parquet", data_files="data2.parquet"),
    ...     ],
    ...     batch_size=32,
    ...     shuffle_buffer_size=10000,
    ... )
    >>> dataset = build_dataset(mixture)
"""

from . import sources, utils
from .loader import create_data_iterator
from .mixture import block_mixture_interleave
from .pack import pack_constant_length, pack_pre_tokenized
from .pipeline import build_dataset
from .types import DatasetLoadError, DatasetMixture, DatasetType, TextDatasetInform, VisualDatasetInform

__all__ = [
    "DatasetLoadError",
    "DatasetMixture",
    "DatasetType",
    "TextDatasetInform",
    "VisualDatasetInform",
    "block_mixture_interleave",
    "build_dataset",
    "create_data_iterator",
    "pack_constant_length",
    "pack_pre_tokenized",
    "sources",
    "utils",
]

