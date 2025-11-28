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
    - Per-dataset tokenization and save paths
    - Dynamic weight scheduling for training
    - Async data loading with JAX sharding
    - Ray distributed preprocessing
    - Comprehensive transform DSL (map, filter, field ops, chat templates)

Package Structure:
    - core/: Foundation layer (protocols, config, types)
    - sources/: Data source implementations
    - transforms/: Transform DSL and processing stages
    - execution/: Data loading, saving, caching, pipeline
    - distributed/: Ray integration

Example (Legacy API):
    >>> from easydel.data import DatasetMixture, TextDatasetInform
    >>> from easydel.data import build_dataset
    >>>
    >>> mixture = DatasetMixture(
    ...     informs=[TextDatasetInform(type="json", data_files="data.json")],
    ...     batch_size=32,
    ... )
    >>> dataset = build_dataset(mixture)

Example (New Pipeline API):
    >>> from easydel.data import Pipeline, DatasetConfig, PipelineConfig
    >>>
    >>> config = PipelineConfig(
    ...     datasets=[
    ...         DatasetConfig(
    ...             data_files="data/*.json",
    ...             tokenizer="meta-llama/Llama-2-7b",
    ...             save_path="/output/tokenized",
    ...         )
    ...     ],
    ... )
    >>> pipeline = Pipeline.from_config(config)
    >>> for batch in pipeline.source().tokenize().pack().load().build():
    ...     train_step(batch)

Example (New Transform API):
    >>> from easydel.data import JsonShardedSource
    >>> from easydel.data.transforms import ChatTemplateTransform
    >>>
    >>> source = JsonShardedSource("conversations.jsonl")
    >>> # Apply chat template to convert messages to text
    >>> formatted = source.apply_chat_template(tokenizer)
"""

from . import core, distributed, execution, sources, transforms, utils
from .core import (
    AsyncDataset,
    BaseDatasetInform,
    BaseStage,
    CacheStageConfig,
    DatasetConfig,
    DatasetLoadError,
    DatasetMixture,
    DatasetType,
    LoadStageConfig,
    MixStageConfig,
    ObservabilityConfig,
    PackStageConfig,
    PipelineConfig,
    PipelineContext,
    RayConfig,
    ResumeState,
    SaveStageConfig,
    ShardedDataSource,
    ShardInfo,
    SourceStageConfig,
    TextDatasetInform,
    TokenizerConfig,
    TokenizeStageConfig,
    VisualDatasetInform,
    WeightSchedulePoint,
)
from .execution import (
    ArrowWriter,
    AsyncDataLoader,
    CacheLayer,
    CacheMetadata,
    DatasetCache,
    DiskCache,
    JsonlWriter,
    LoadStage,
    MemoryCache,
    ParquetWriter,
    Pipeline,
    PrefetchIterator,
    SaveStage,
    ShardingSpec,
    TreeCacheManager,
    WriteStats,
    batch_iterator,
    build_dataset,
    collate_batch,
    create_data_iterator,
    create_pipeline,
    create_writer,
    preshard_batch,
    pretokenize,
    save_dataset,
    save_iterator,
    tokenize_and_save,
)
from .sources import (
    ArrowShardedSource,
    CompositeShardedSource,
    CsvShardedSource,
    HFDatasetShardedSource,
    HuggingFaceShardedSource,
    JsonShardedSource,
    ParquetShardedSource,
    TextShardedSource,
    create_source,
    expand_data_files,
    wrap_hf_dataset,
)
from .transforms import (
    # Base
    AddField,
    ChainedTransform,
    ChatTemplateTransform,
    CombineFields,
    ConvertInputOutputToChatML,
    ConvertToChatML,
    DropFields,
    ExtractField,
    FilterByField,
    FilterNonEmpty,
    FilterTransform,
    FirstFitPacker,
    GreedyPacker,
    MapField,
    MapTransform,
    MaybeApplyChatTemplate,
    MixedShardedSource,
    MixedShardState,
    MixStage,
    PackedSequence,
    PackedShardedSource,
    PackStage,
    PoolPacker,
    RenameFields,
    SelectFields,
    TokenizedShardedSource,
    TokenizerManager,
    TokenizeStage,
    Transform,
    TransformedShardedSource,
    WeightScheduler,
    block_mixture_interleave,
    pack_constant_length,
    pack_pre_tokenized,
    tokenize_dataset_config,
)

# Trainer-specific transforms (lazy import from trainers to avoid circular imports)
_TRAINER_TRANSFORMS = {
    "BCOPreprocessTransform",
    "CPOPreprocessTransform",
    "DPOPreprocessTransform",
    "GRPOPreprocessTransform",
    "KTOPreprocessTransform",
    "ORPOPreprocessTransform",
    "RewardPreprocessTransform",
    "SFTPreprocessTransform",
}

__all__ = [
    "AddField",
    "ArrowShardedSource",
    "ArrowWriter",
    "AsyncDataLoader",
    "AsyncDataset",
    "BCOPreprocessTransform",
    "BaseDatasetInform",
    "BaseStage",
    "CPOPreprocessTransform",
    "CacheLayer",
    "CacheMetadata",
    "CacheStageConfig",
    "ChainedTransform",
    "ChatTemplateTransform",
    "CombineFields",
    "CompositeShardedSource",
    "ConvertInputOutputToChatML",
    "ConvertToChatML",
    "CsvShardedSource",
    "DPOPreprocessTransform",
    "DatasetCache",
    "DatasetConfig",
    "DatasetLoadError",
    "DatasetMixture",
    "DatasetType",
    "DiskCache",
    "DropFields",
    "ExtractField",
    "FilterByField",
    "FilterNonEmpty",
    "FilterTransform",
    "FirstFitPacker",
    "GRPOPreprocessTransform",
    "GreedyPacker",
    "HFDatasetShardedSource",
    "HuggingFaceShardedSource",
    "JsonShardedSource",
    "JsonlWriter",
    "KTOPreprocessTransform",
    "LoadStage",
    "LoadStageConfig",
    "MapField",
    "MapTransform",
    "MaybeApplyChatTemplate",
    "MemoryCache",
    "MixStage",
    "MixStageConfig",
    "MixedShardState",
    "MixedShardedSource",
    "ORPOPreprocessTransform",
    "ObservabilityConfig",
    "PackStage",
    "PackStageConfig",
    "PackedSequence",
    "PackedShardedSource",
    "ParquetShardedSource",
    "ParquetWriter",
    "Pipeline",
    "PipelineConfig",
    "PipelineContext",
    "PoolPacker",
    "PrefetchIterator",
    "RayConfig",
    "RenameFields",
    "ResumeState",
    "RewardPreprocessTransform",
    "SFTPreprocessTransform",
    "SaveStage",
    "SaveStageConfig",
    "SelectFields",
    "ShardInfo",
    "ShardedDataSource",
    "ShardingSpec",
    "SourceStageConfig",
    "TextDatasetInform",
    "TextShardedSource",
    "TokenizeStage",
    "TokenizeStageConfig",
    "TokenizedShardedSource",
    "TokenizerConfig",
    "TokenizerManager",
    "Transform",
    "TransformedShardedSource",
    "TreeCacheManager",
    "VisualDatasetInform",
    "WeightSchedulePoint",
    "WeightScheduler",
    "WriteStats",
    "batch_iterator",
    "block_mixture_interleave",
    "build_dataset",
    "collate_batch",
    "core",
    "create_data_iterator",
    "create_pipeline",
    "create_source",
    "create_writer",
    "distributed",
    "execution",
    "expand_data_files",
    "pack_constant_length",
    "pack_pre_tokenized",
    "preshard_batch",
    "pretokenize",
    "save_dataset",
    "save_iterator",
    "sources",
    "tokenize_and_save",
    "tokenize_dataset_config",
    "transforms",
    "utils",
    "wrap_hf_dataset",
]


def __getattr__(name: str):
    """Lazy import trainer transforms for backwards compatibility."""
    if name in _TRAINER_TRANSFORMS:
        from easydel.trainers import prompt_transforms

        return getattr(prompt_transforms, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
