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

"""Core protocols, configurations, and types for the data pipeline."""

from .config import (
    CacheStageConfig,
    DatasetConfig,
    LoadStageConfig,
    MixStageConfig,
    ObservabilityConfig,
    PackStageConfig,
    PipelineConfig,
    RayConfig,
    SaveStageConfig,
    SourceStageConfig,
    TokenizerConfig,
    TokenizeStageConfig,
    WeightSchedulePoint,
    get_dataset_name,
    merge_tokenizer_config,
)
from .protocols import (
    AsyncDataset,
    AsyncDatasetProtocol,
    BaseStage,
    MappedShardedDataSource,
    PipelineContext,
    PipelineStage,
    ResumeState,
    ShardedDataSource,
    ShardInfo,
)
from .types import (
    BaseDatasetInform,
    DatasetLoadError,
    DatasetMixture,
    DatasetType,
    TextDatasetInform,
    VisualDatasetInform,
)

__all__ = [
    # Protocols
    "AsyncDataset",
    "AsyncDatasetProtocol",
    # Types (legacy)
    "BaseDatasetInform",
    "BaseStage",
    # Config
    "CacheStageConfig",
    "DatasetConfig",
    "DatasetLoadError",
    "DatasetMixture",
    "DatasetType",
    "LoadStageConfig",
    "MappedShardedDataSource",
    "MixStageConfig",
    "ObservabilityConfig",
    "PackStageConfig",
    "PipelineConfig",
    "PipelineContext",
    "PipelineStage",
    "RayConfig",
    "ResumeState",
    "SaveStageConfig",
    "ShardInfo",
    "ShardedDataSource",
    "SourceStageConfig",
    "TextDatasetInform",
    "TokenizeStageConfig",
    "TokenizerConfig",
    "VisualDatasetInform",
    "WeightSchedulePoint",
    "get_dataset_name",
    "merge_tokenizer_config",
]
