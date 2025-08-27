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

Provides tools for loading, mixing, and managing datasets for training
and evaluation. Supports both text and visual datasets with flexible
mixing strategies.

Classes:
    DataManager: Main dataset management class
    DatasetMixture: Dataset mixing configuration
    DatasetType: Enum for dataset types
    TextDatasetInform: Text dataset information
    VisualDatasetInform: Visual dataset information
    DatasetLoadError: Dataset loading exception

Key Features:
    - Flexible dataset loading from HuggingFace
    - Dataset mixing with custom ratios
    - Support for text and visual modalities
    - Streaming and non-streaming modes
    - Automatic preprocessing and tokenization

Example:
    >>> from easydel.utils.data_managers import DataManager, DatasetMixture
    >>>
    >>> # Single dataset
    >>> manager = DataManager(
    ...     dataset_name="squad",
    ...     tokenizer=tokenizer
    ... )
    >>>
    >>> # Mixed datasets
    >>> mixture = DatasetMixture(
    ...     datasets=["dataset1", "dataset2"],
    ...     ratios=[0.7, 0.3]
    ... )
    >>> manager = DataManager(mixture=mixture)
    >>>
    >>> # Load and iterate
    >>> dataloader = manager.get_dataloader(batch_size=32)
    >>> for batch in dataloader:
    ...     process(batch)
"""

from .manager import DataManager
from .types import DatasetLoadError, DatasetMixture, DatasetType, TextDatasetInform, VisualDatasetInform

__all__ = (
    "DataManager",
    "DatasetLoadError",
    "DatasetMixture",
    "DatasetType",
    "TextDatasetInform",
    "VisualDatasetInform",
)
