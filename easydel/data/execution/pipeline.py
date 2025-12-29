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

"""Fluent API pipeline builder for data processing.

This module provides:
- Pipeline class with fluent API for building data pipelines
- Stage-based composition (source -> tokenize -> cache -> mix -> pack -> load)
- Per-dataset configuration support
- Easy creation from PipelineConfig
"""

from __future__ import annotations

import logging
import os
import typing as tp

from ..core.config import (
    DatasetConfig,
    LoadStageConfig,
    MixStageConfig,
    PackStageConfig,
    PipelineConfig,
    SaveStageConfig,
    TokenizeStageConfig,
)
from ..core.protocols import PipelineContext, ShardedDataSource
from ..core.types import DatasetMixture, TextDatasetInform
from ..sources import create_source, load_for_inform
from ..transforms.mixture import MixStage, block_mixture_interleave
from ..transforms.pack import PackStage, pack_constant_length, pack_pre_tokenized
from ..transforms.tokenize import TokenizeStage
from ..utils import align_columns_intersection, is_streaming, wrap_format_callback
from .loader import AsyncDataLoader, LoadStage
from .save import SaveStage, WriteStats

if tp.TYPE_CHECKING:
    from collections.abc import Iterator

    from datasets import Dataset as DS


logger = logging.getLogger(__name__)


class Pipeline:
    """Fluent API for building data processing pipelines.

    Provides a chainable interface for creating complex data pipelines
    with per-dataset configuration support.

    Example:
        >>> config = PipelineConfig(
        ...     datasets=[
        ...         DatasetConfig(
        ...             data_files="data/*.json",
        ...             tokenizer="meta-llama/Llama-2-7b",
        ...             save_path="/output/tokenized",
        ...         )
        ...     ],
        ...     pack=PackStageConfig(enabled=True, seq_length=2048),
        ... )
        >>> pipeline = Pipeline.from_config(config)
        >>> for batch in pipeline.source().tokenize().pack().load().build():
        ...     train_step(batch)
    """

    def __init__(self, config: PipelineConfig):
        """Initialize Pipeline.

        Args:
            config: Pipeline configuration.
        """
        self._config = config
        self._context = PipelineContext(config=config, seed=config.seed)
        self._data: dict[str, ShardedDataSource] | None = None
        self._stages: list[str] = []

    @classmethod
    def from_config(cls, config: PipelineConfig | dict) -> "Pipeline":
        """Create a pipeline from configuration.

        Args:
            config: PipelineConfig or dict with configuration.

        Returns:
            New Pipeline instance.
        """
        if isinstance(config, dict):
            # Convert dict to PipelineConfig
            datasets = [DatasetConfig(**ds) if isinstance(ds, dict) else ds for ds in config.get("datasets", [])]
            config = PipelineConfig(
                datasets=datasets,
                default_tokenizer=config.get("default_tokenizer"),
                streaming=config.get("streaming", True),
                seed=config.get("seed"),
                source=config.get("source"),
                tokenize=TokenizeStageConfig(**config.get("tokenize", {})) if config.get("tokenize") else None,
                cache=config.get("cache"),
                mix=MixStageConfig(**config.get("mix", {})) if config.get("mix") else None,
                pack=PackStageConfig(**config.get("pack", {})) if config.get("pack") else None,
                load=LoadStageConfig(**config.get("load", {})) if config.get("load") else None,
                save=SaveStageConfig(**config.get("save", {})) if config.get("save") else None,
            )
        return cls(config)

    def source(self) -> "Pipeline":
        """Load datasets from their sources.

        Creates ShardedDataSource instances for each dataset in the config.

        Returns:
            Self for chaining.
        """
        if self._data is not None:
            raise RuntimeError("source() has already been called")

        self._data = {}
        for i, ds_config in enumerate(self._config.datasets):
            name = ds_config.name or f"dataset_{i}"
            source = create_source(ds_config)
            self._data[name] = source
            logger.info(f"Loaded source for dataset '{name}'")

        self._stages.append("source")
        return self

    def tokenize(self, config: TokenizeStageConfig | None = None) -> "Pipeline":
        """Apply tokenization to all datasets.

        Uses per-dataset tokenizer configuration when available.

        Args:
            config: Optional override for tokenization config.

        Returns:
            Self for chaining.
        """
        self._ensure_data()

        stage_config = config or self._config.tokenize
        stage = TokenizeStage(stage_config)
        self._data = stage.process(self._data, self._context)
        self._stages.append("tokenize")
        return self

    def mix(self, config: MixStageConfig | None = None) -> "Pipeline":
        """Mix multiple datasets into one.

        Supports static weights and dynamic weight scheduling.

        Args:
            config: Optional override for mix config.

        Returns:
            Self for chaining.
        """
        self._ensure_data()

        if len(self._data) <= 1:
            logger.info("Only one dataset, skipping mix stage")
            self._stages.append("mix")
            return self

        stage_config = config or self._config.mix
        stage = MixStage(stage_config)
        self._data = stage.process(self._data, self._context)
        self._stages.append("mix")
        return self

    def pack(self, config: PackStageConfig | None = None) -> "Pipeline":
        """Pack sequences into fixed-length chunks.

        Supports multiple packing strategies (greedy, pool, first_fit).

        Args:
            config: Optional override for pack config.

        Returns:
            Self for chaining.
        """
        self._ensure_data()

        stage_config = config or self._config.pack
        stage = PackStage(stage_config)
        self._data = stage.process(self._data, self._context)
        self._stages.append("pack")
        return self

    def save(self, config: SaveStageConfig | None = None) -> "Pipeline":
        """Save datasets to disk.

        Uses per-dataset save paths when available.

        Args:
            config: Optional override for save config.

        Returns:
            Self for chaining.
        """
        self._ensure_data()

        stage_config = config or self._config.save
        stage = SaveStage(stage_config)
        self._data = stage.process(self._data, self._context)
        self._stages.append("save")
        return self

    def load(self, config: LoadStageConfig | None = None) -> "Pipeline":
        """Create data loaders with batching and prefetching.

        Args:
            config: Optional override for load config.

        Returns:
            Self for chaining.
        """
        self._ensure_data()

        stage_config = config or self._config.load
        stage = LoadStage(stage_config)
        self._data = stage.process(self._data, self._context)
        self._stages.append("load")
        return self

    def build(self) -> "Iterator[dict] | AsyncDataLoader":
        """Build and return the final data iterator.

        Returns:
            Iterator or AsyncDataLoader depending on pipeline configuration.
        """
        self._ensure_data()

        # If we have a single loader, return it directly
        if len(self._data) == 1:
            return next(iter(self._data.values()))

        # Return the mixed/combined result
        return next(iter(self._data.values()))

    def get_data(self) -> dict[str, tp.Any]:
        """Get the current pipeline data.

        Returns:
            Dictionary mapping dataset names to their current state.
        """
        return self._data or {}

    def get_context(self) -> PipelineContext:
        """Get the pipeline context.

        Returns:
            Pipeline context with configuration and metrics.
        """
        return self._context

    def get_stages(self) -> list[str]:
        """Get the list of applied stages.

        Returns:
            List of stage names in order.
        """
        return self._stages.copy()

    def _ensure_data(self):
        """Ensure data has been loaded."""
        if self._data is None:
            raise RuntimeError("Call source() before other pipeline stages")


def create_pipeline(
    datasets: list[DatasetConfig | dict],
    default_tokenizer: str | None = None,
    **kwargs,
) -> Pipeline:
    """Create a pipeline from a list of dataset configurations.

    Args:
        datasets: List of DatasetConfig or dicts.
        default_tokenizer: Default tokenizer for all datasets.
        **kwargs: Additional PipelineConfig options.

    Returns:
        New Pipeline instance.
    """
    ds_configs = [DatasetConfig(**ds) if isinstance(ds, dict) else ds for ds in datasets]
    config = PipelineConfig(
        datasets=ds_configs,
        default_tokenizer=default_tokenizer,
        **kwargs,
    )
    return Pipeline(config)


def tokenize_and_save(
    data_files: str | os.PathLike | list[str | os.PathLike],
    tokenizer: str,
    output_path: str,
    output_format: str = "parquet",
    max_length: int = 2048,
) -> None:
    """Tokenize a dataset and save to disk.

    Convenience function for the common pattern of tokenizing and saving.

    Args:
        data_files: Path(s) to input data.
        tokenizer: Tokenizer name or path.
        output_path: Output directory.
        output_format: Output format (parquet, arrow, jsonl).
        max_length: Maximum sequence length.
    """
    config = PipelineConfig(
        datasets=[
            DatasetConfig(
                data_files=data_files,
                tokenizer=tokenizer,
                save_path=output_path,
                save_format=output_format,
            )
        ],
        tokenize=TokenizeStageConfig(max_length=max_length),
        save=SaveStageConfig(enabled=True, format=output_format),
    )

    Pipeline.from_config(config).source().tokenize().save().build()
    logger.info(f"Tokenized and saved to {output_path}")


def pretokenize(
    source: "ShardedDataSource",
    transform: tp.Any,
    output_path: str,
    output_format: str = "parquet",
    max_shard_size: str = "500MB",
    compression: str | None = "snappy",
    num_proc: int | None = None,
    show_progress: bool = True,
) -> WriteStats:
    """Pretokenize a data source using a trainer transform and save to disk.

    This is a convenience function for preprocessing datasets with trainer-specific
    transforms like SFTPreprocessTransform, DPOPreprocessTransform, etc. The transform
    handles all preprocessing (chat template, tokenization, label creation) in one pass.

    Args:
        source: ShardedDataSource to pretokenize.
        transform: Trainer transform (e.g., SFTPreprocessTransform, DPOPreprocessTransform).
            Must be a callable that takes an example dict and returns a tokenized dict.
        output_path: Directory to save pretokenized data.
        output_format: Output format - "parquet" (default), "arrow", or "jsonl".
        max_shard_size: Maximum size per output shard (e.g., "500MB", "1GB").
        compression: Compression algorithm (default: "snappy" for parquet).
        num_proc: Number of parallel processes (currently unused, reserved for future).
        show_progress: Whether to show progress information.

    Returns:
        WriteStats with num_examples, num_shards, total_bytes, output_paths.

    Example:
        >>> from transformers import AutoTokenizer
        >>> from easydel.data import HuggingFaceShardedSource, pretokenize
        >>> from easydel.trainers import SFTPreprocessTransform
        >>>
        >>> # Load tokenizer and create transform
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        >>> transform = SFTPreprocessTransform(
        ...     tokenizer=tokenizer,
        ...     max_length=2048,
        ...     mask_prompt=True,
        ... )
        >>>
        >>> # Create source and pretokenize
        >>> source = HuggingFaceShardedSource("tatsu-lab/alpaca")
        >>> stats = pretokenize(source, transform, "./pretokenized_alpaca")
        >>> print(f"Saved {stats.num_examples} examples")

    Example with mixed datasets:
        >>> from easydel.data import MixedShardedSource
        >>> from easydel.trainers import DPOPreprocessTransform
        >>>
        >>> # Create mixed source
        >>> sources = [
        ...     HuggingFaceShardedSource("Anthropic/hh-rlhf"),
        ...     HuggingFaceShardedSource("argilla/ultrafeedback-binarized"),
        ... ]
        >>> mixed = MixedShardedSource(
        ...     sources=sources,
        ...     weights=[0.5, 0.5],
        ...     block_size=1024,
        ... )
        >>>
        >>> # Pretokenize for DPO
        >>> transform = DPOPreprocessTransform(tokenizer=tokenizer, max_length=2048)
        >>> stats = pretokenize(mixed, transform, "./pretokenized_dpo")
    """
    from ..transforms.source import TransformedShardedSource
    from .save import save_dataset

    _ = num_proc  # Reserved for future parallel processing

    if show_progress:
        logger.info(f"Pretokenizing with {transform.__class__.__name__}...")
        logger.info(f"Output: {output_path} ({output_format})")

    # Wrap source with transform
    transformed_source = TransformedShardedSource(source, transform)

    # Save to disk
    stats = save_dataset(
        source=transformed_source,
        output_path=output_path,
        format=output_format,
        max_shard_size=max_shard_size,
        compression=compression,
    )

    if show_progress:
        logger.info(
            f"Pretokenization complete: {stats.num_examples:,} examples, "
            f"{stats.num_shards} shards, {stats.total_bytes / 1024 / 1024:.2f} MB"
        )

    return stats


def build_dataset(mixture: DatasetMixture) -> "DS":
    """Build a unified dataset from a DatasetMixture configuration.

    This is the main entry point for creating datasets. It handles loading
    multiple data sources, applying transformations, mixing datasets with
    various strategies, and optionally packing sequences for efficient training.

    The pipeline supports:
    - Loading from HuggingFace Hub and local files
    - Field renaming and custom format callbacks
    - Multiple mixing strategies (standard interleave or block-deterministic)
    - Optional token packing (pre-tokenized or on-the-fly)
    - Streaming and non-streaming modes

    Args:
        mixture: DatasetMixture configuration object containing all settings
            for dataset loading, processing, and mixing.

    Returns:
        A Dataset or IterableDataset ready for training, with all transformations
        and mixing strategies applied.

    Example:
        >>> from easydel.data import DatasetMixture, TextDatasetInform
        >>>
        >>> # Simple single dataset
        >>> mixture = DatasetMixture(
        ...     informs=[TextDatasetInform(type="json", data_files="data.json")],
        ...     batch_size=32
        ... )
        >>> dataset = build_dataset(mixture)
        >>>
        >>> # Complex multi-dataset mixture with packing
        >>> mixture = DatasetMixture(
        ...     informs=[
        ...         TextDatasetInform(type="parquet", data_files="dataset1/*.parquet"),
        ...         TextDatasetInform(type="json", data_files="dataset2.json"),
        ...     ],
        ...     block_mixture=True,
        ...     mixture_weights={"dataset1": 0.7, "dataset2": 0.3},
        ...     pack_tokens=True,
        ...     pack_seq_length=2048,
        ... )
        >>> dataset = build_dataset(mixture)
    """
    per_ds = []
    content_target = mixture.text_target_field

    for inform in mixture.informs:
        ds = load_for_inform(inform, mixture)

        if getattr(inform, "format_fields", None):
            mapping_local = dict(inform.format_fields)

            def rename_fields(ex, _mapping=mapping_local):
                for old_name, new_name in _mapping.items():
                    if old_name in ex:
                        ex[new_name] = ex.pop(old_name)
                for k in list(ex.keys()):
                    v = ex[k]
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        ex[k] = [{(_mapping.get(kk) or kk): vv for kk, vv in d.items()} for d in v]
                return ex

            ds = ds.map(rename_fields, batched=False)

        if getattr(inform, "format_callback", None):
            fmt = wrap_format_callback(inform.format_callback, getattr(inform, "content_field", "content"))

            try:
                ex0 = next(iter(ds.take(1))) if is_streaming(ds) else ds[0]
            except (StopIteration, IndexError) as e:
                raise ValueError(
                    f"Cannot apply format_callback to empty dataset: {getattr(inform, 'data_files', 'unknown')}"
                ) from e
            after = fmt(dict(ex0))
            cols_to_remove = list(set(ex0.keys()) - set(after.keys()))
            ds = ds.map(fmt, batched=False, remove_columns=cols_to_remove or None)

        if isinstance(inform, TextDatasetInform):
            keep = {content_target}
            addl = getattr(inform, "additional_fields", None) or []
            keep.update(addl)

            content_field = inform.content_field
            addl_fields = tuple(addl or ())

            def to_target(ex, _content_field=content_field, _addl=addl_fields, _target=content_target):
                if _content_field is None:
                    return ex
                try:
                    out = {_target: ex[_content_field]}
                except KeyError as e:
                    raise KeyError(f"Missing content field '{_content_field}'. Available keys: {list(ex.keys())}") from e
                for f in _addl:
                    if f in ex:
                        out[f] = ex[f]
                return out

            ds = ds.map(to_target, batched=False)
            try:
                ds = ds.select_columns(list(keep))
            except (ValueError, KeyError, AttributeError):
                # Column selection not supported for this dataset type
                pass

        per_ds.append(ds)

    if mixture.streaming:
        if getattr(mixture, "block_mixture", False):
            weights = None
            if mixture.mixture_weights and len(mixture.mixture_weights) == len(per_ds):
                weights = mixture.mixture_weights
            mixed = block_mixture_interleave(
                per_ds,
                weights=weights,
                block_size=getattr(mixture, "mixture_block_size", 2048),
                seed=mixture.seed or 0,
                stop=getattr(mixture, "stop_strategy", "restart"),
            )
        else:
            from datasets import interleave_datasets

            mixed = interleave_datasets(per_ds, seed=mixture.seed, stopping_strategy="first_exhausted")
            if mixture.shuffle_buffer_size:
                mixed = mixed.shuffle(buffer_size=mixture.shuffle_buffer_size, seed=mixture.seed)
    else:
        per_ds = align_columns_intersection(per_ds)
        from datasets import concatenate_datasets

        mixed = concatenate_datasets(per_ds)
        if mixture.shuffle_buffer_size:
            mixed = mixed.shuffle(seed=mixture.seed)

    if getattr(mixture, "pack_tokens", False):
        from datasets import IterableDataset

        gen = pack_pre_tokenized(
            iter(mixed),
            seq_length=mixture.pack_seq_length or 1024,
            eos_token_id=mixture.pack_eos_token_id,
            batch_size=mixture.batch_size,
            shuffle=mixture.pack_shuffle,
            buffer_factor=mixture.pack_shuffle_buffer_factor,
        )
        return IterableDataset.from_generator(gen)

    if getattr(mixture, "pack_on_the_fly", False):
        if mixture.tokenize_callback is None:
            raise ValueError("pack_on_the_fly=True requires mixture.tokenize_callback")
        from datasets import IterableDataset

        gen = pack_constant_length(
            iter(mixed),
            tokenize_fn=mixture.tokenize_callback,
            seq_length=mixture.pack_seq_length or 1024,
            eos_token_id=mixture.pack_eos_token_id,
            batch_size=mixture.batch_size,
            shuffle=mixture.pack_shuffle,
            buffer_factor=mixture.pack_shuffle_buffer_factor,
        )
        return IterableDataset.from_generator(gen)

    if mixture.batch_size and mixture.batch_size > 1 and is_streaming(mixed):
        mixed = mixed.batch(mixture.batch_size)

    return mixed
