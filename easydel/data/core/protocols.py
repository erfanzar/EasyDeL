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

"""Core protocols and abstractions for the data management pipeline.

This module defines the base interfaces for:
- Pipeline stages (Source, Tokenize, Cache, Mix, Pack, Load, Save)
- Async datasets with JAX integration
- Sharded data sources with URL support
- Pipeline context for shared state
"""

from __future__ import annotations

import typing as tp
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from jax.sharding import NamedSharding

    from .config import PipelineConfig

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
DatasetLike = tp.Union["Dataset", "IterableDataset", tp.Iterator[dict]]


@dataclass
class ShardInfo:
    """Metadata about a data shard."""

    shard_id: int
    shard_name: str
    num_rows: int | None = None
    byte_size: int | None = None
    url: str | None = None
    checksum: str | None = None


class ShardedDataSource(ABC, Generic[T_co]):
    """Abstract base class for shard-based data sources.

    Inspired by Levanter's ShardedDataSource, this provides:
    - URL-based shard discovery and loading
    - Resumable iteration with shard-level checkpoints
    - Distributed-friendly shard assignment

    Example:
        >>> source = ParquetShardedSource("gs://bucket/data/*.parquet")
        >>> for shard_name in source.shard_names[:10]:
        ...     for example in source.open_shard(shard_name):
        ...         process(example)
    """

    @property
    @abstractmethod
    def shard_names(self) -> Sequence[str]:
        """Return list of shard identifiers (URLs or paths)."""
        ...

    @abstractmethod
    def num_shards(self) -> int:
        """Return total number of shards."""
        ...

    @abstractmethod
    def open_shard(self, shard_name: str) -> Iterator[T_co]:
        """Open a shard and return an iterator over its examples."""
        ...

    def get_shard_info(self, shard_name: str) -> ShardInfo | None:
        """Get metadata about a specific shard. Optional."""
        return None

    def open_shard_at_row(
        self,
        shard_name: str,
        row: int,
    ) -> Iterator[T_co]:
        """Open a shard and skip to a specific row for resumption.

        Default implementation skips rows sequentially. Subclasses can
        override for more efficient seeking (e.g., Parquet row groups).
        """
        it = self.open_shard(shard_name)
        for _ in range(row):
            next(it, None)
        return it

    def iter_shards(
        self,
        shard_indices: Sequence[int] | None = None,
        start_shard: int = 0,
        start_row: int = 0,
    ) -> Iterator[T_co]:
        """Iterate over shards with optional resume support.

        Args:
            shard_indices: Subset of shard indices to iterate (for distributed).
            start_shard: Index to start from (for resumption).
            start_row: Row within start_shard to begin (for resumption).

        Yields:
            Examples from all specified shards.
        """
        indices = shard_indices if shard_indices is not None else range(self.num_shards())

        for i, shard_idx in enumerate(indices):
            if i < start_shard:
                continue

            shard_name = self.shard_names[shard_idx]

            if i == start_shard and start_row > 0:
                shard_iter = self.open_shard_at_row(shard_name, start_row)
            else:
                shard_iter = self.open_shard(shard_name)

            yield from shard_iter

    def map(
        self,
        fn: tp.Callable[[T_co], T],
    ) -> "MappedShardedDataSource[T]":
        """Apply a function to each example (lazy)."""
        return MappedShardedDataSource(self, fn)

    def filter(
        self,
        predicate: tp.Callable[[T_co], bool],
    ) -> "ShardedDataSource[T_co]":
        """Filter examples by predicate (lazy).

        Args:
            predicate: Function that returns True for examples to keep.

        Returns:
            TransformedShardedSource with filter applied.
        """
        from ..transforms import FilterTransform, TransformedShardedSource

        return TransformedShardedSource(self, FilterTransform(predicate))

    def __len__(self) -> int:
        """Return total number of examples across all shards.

        Raises:
            TypeError: If the source doesn't support length (streaming).
        """
        raise TypeError(f"{type(self).__name__} has no len()")

    def transform(
        self,
        transform: "tp.Any",  # Transform type
    ) -> "ShardedDataSource":
        """Apply a transform or chain of transforms.

        Args:
            transform: Transform object to apply.

        Returns:
            TransformedShardedSource with transform applied.
        """
        from ..transforms import TransformedShardedSource

        return TransformedShardedSource(self, transform)

    def rename_fields(
        self,
        mapping: dict[str, str],
    ) -> "ShardedDataSource":
        """Rename fields in examples.

        Args:
            mapping: Dictionary mapping old field names to new names.

        Returns:
            TransformedShardedSource with rename applied.
        """
        from ..transforms import RenameFields, TransformedShardedSource

        return TransformedShardedSource(self, RenameFields(mapping))

    def select_fields(
        self,
        fields: list[str],
    ) -> "ShardedDataSource":
        """Select only specified fields.

        Args:
            fields: List of field names to keep.

        Returns:
            TransformedShardedSource with selection applied.
        """
        from ..transforms import SelectFields, TransformedShardedSource

        return TransformedShardedSource(self, SelectFields(fields))

    def drop_fields(
        self,
        fields: list[str],
    ) -> "ShardedDataSource":
        """Drop specified fields.

        Args:
            fields: List of field names to remove.

        Returns:
            TransformedShardedSource with drop applied.
        """
        from ..transforms import DropFields, TransformedShardedSource

        return TransformedShardedSource(self, DropFields(fields))

    def apply_chat_template(
        self,
        tokenizer: tp.Any,
        messages_field: str = "messages",
        output_field: str = "text",
        **kwargs,
    ) -> "ShardedDataSource":
        """Apply chat template to convert messages to formatted text.

        Args:
            tokenizer: HuggingFace tokenizer with chat template.
            messages_field: Field containing the messages list.
            output_field: Field to store the formatted text.
            **kwargs: Additional arguments for ChatTemplateTransform.

        Returns:
            TransformedShardedSource with chat template applied.
        """
        from ..transforms import ChatTemplateTransform, TransformedShardedSource

        transform = ChatTemplateTransform(
            tokenizer=tokenizer,
            messages_field=messages_field,
            output_field=output_field,
            **kwargs,
        )
        return TransformedShardedSource(self, transform)


class MappedShardedDataSource(ShardedDataSource[T], Generic[T]):
    """Lazily mapped sharded data source."""

    def __init__(
        self,
        source: ShardedDataSource,
        fn: tp.Callable[[Any], T],
    ):
        self._source = source
        self._fn = fn

    @property
    def shard_names(self) -> Sequence[str]:
        return self._source.shard_names

    def num_shards(self) -> int:
        return self._source.num_shards()

    def open_shard(self, shard_name: str) -> Iterator[T]:
        for example in self._source.open_shard(shard_name):
            yield self._fn(example)

    def __len__(self) -> int:
        """Return length of underlying source."""
        return len(self._source)


@runtime_checkable
class AsyncDatasetProtocol(Protocol[T_co]):
    """Protocol for async datasets that integrate with JAX's execution model."""

    async def aget(self, index: int) -> T_co:
        """Asynchronously get item at index."""
        ...

    async def __aiter__(self) -> AsyncIterator[T_co]:
        """Async iteration over the dataset."""
        ...

    def get_output_sharding(self) -> Mapping[str, "NamedSharding"] | None:
        """Return the sharding specification for output batches.

        Returns None if no specific sharding is configured.
        """
        ...

    @property
    def is_exhausted(self) -> bool:
        """Check if dataset iteration is complete."""
        ...


class AsyncDataset(ABC, Generic[T]):
    """Async-first dataset with dual sync/async interface.

    Designed for JAX integration with:
    - Pre-sharding during prefetch
    - Proper PRNG key handling
    - Memory-efficient streaming
    """

    @abstractmethod
    async def aget(self, index: int) -> T:
        """Asynchronously get item at index."""
        ...

    @abstractmethod
    async def __aiter__(self) -> AsyncIterator[T]:
        """Async iteration over the dataset."""
        ...

    async def abatch(self, indices: Sequence[int]) -> list[T]:
        """Get multiple items concurrently."""
        import asyncio

        return list(await asyncio.gather(*[self.aget(i) for i in indices]))

    def get_output_sharding(self) -> Mapping[str, "NamedSharding"] | None:
        """Return sharding specification for output batches."""
        return None

    @property
    def is_exhausted(self) -> bool:
        """Check if dataset iteration is complete."""
        return False

    # Synchronous interface (wraps async)
    def get(self, index: int) -> T:
        """Synchronously get item at index."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.aget(index))

    def __iter__(self) -> Iterator[T]:
        """Sync iteration (wraps async)."""
        import asyncio

        async def _collect():
            results = []
            async for item in self:
                results.append(item)
            return results

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return iter(loop.run_until_complete(_collect()))


@runtime_checkable
class PipelineStage(Protocol):
    """Protocol for all pipeline stages."""

    @property
    def name(self) -> str:
        """Stage name for logging and metrics."""
        ...

    def process(
        self,
        data: Any,
        context: "PipelineContext",
    ) -> Any:
        """Process data through this stage.

        Args:
            data: Input data (dataset, iterator, or dict of datasets)
            context: Pipeline context with config and state

        Returns:
            Processed data
        """
        ...

    def validate_config(self, config: dict) -> bool:
        """Validate stage configuration."""
        ...


class BaseStage(ABC):
    """Base class for pipeline stages with common functionality."""

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._metrics: dict[str, Any] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name."""
        ...

    @abstractmethod
    def process(
        self,
        data: Any,
        context: "PipelineContext",
    ) -> Any:
        """Process data through this stage."""
        ...

    def validate_config(self, config: dict) -> bool:
        """Validate configuration. Override for custom validation."""
        return True

    def get_metrics(self) -> dict[str, Any]:
        """Return stage metrics."""
        return self._metrics.copy()

    def _update_metric(self, key: str, value: Any):
        """Update a metric value."""
        self._metrics[key] = value

    def _get_dataset_config(
        self,
        dataset_id: str,
        context: "PipelineContext",
    ) -> dict:
        """Get configuration for a specific dataset."""
        datasets = context.config.get("datasets", [])
        for i, ds_cfg in enumerate(datasets):
            ds_name = ds_cfg.get("name", f"dataset_{i}")
            if ds_name == dataset_id:
                return ds_cfg
        return {}


@dataclass
class PipelineContext:
    """Shared context passed through pipeline stages.

    Maintains:
    - Global configuration
    - Shared tokenizers (cached)
    - Metrics from all stages
    - Random state for reproducibility
    """

    config: "PipelineConfig"
    seed: int | None = None
    step: int = 0
    epoch: int = 0
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Cached resources
    _tokenizers: dict[str, Any] = field(default_factory=dict, repr=False)
    _cache_manager: Any = field(default=None, repr=False)

    def get_tokenizer(self, name_or_path: str) -> Any:
        """Get or create a tokenizer (cached)."""
        if name_or_path not in self._tokenizers:
            from transformers import AutoTokenizer

            self._tokenizers[name_or_path] = AutoTokenizer.from_pretrained(
                name_or_path,
                trust_remote_code=True,
            )
        return self._tokenizers[name_or_path]

    def update_step(self, step: int):
        """Update current pipeline step."""
        self.step = step

    def update_epoch(self, epoch: int):
        """Update current epoch."""
        self.epoch = epoch

    def record_metric(self, stage: str, key: str, value: Any):
        """Record a metric for a stage."""
        if stage not in self.metrics:
            self.metrics[stage] = {}
        self.metrics[stage][key] = value

    def get_metrics(self) -> dict[str, dict[str, Any]]:
        """Get all recorded metrics."""
        return self.metrics.copy()

    @property
    def cache_manager(self) -> Any:
        """Get cache manager (lazy initialization)."""
        if self._cache_manager is None:
            from ..execution.cache import TreeCacheManager

            cache_config = self.config.get("cache", {})
            cache_dir = cache_config.get("cache_dir", ".cache/easydel_pipeline")
            self._cache_manager = TreeCacheManager(cache_dir=cache_dir)
        return self._cache_manager


@dataclass
class ResumeState:
    """State for resuming pipeline iteration.

    Stores enough information to resume from a checkpoint:
    - Which shard we were on
    - Which row within that shard
    - Epoch and step counts
    """

    shard_index: int = 0
    row_index: int = 0
    epoch: int = 0
    step: int = 0
    dataset_states: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "shard_index": self.shard_index,
            "row_index": self.row_index,
            "epoch": self.epoch,
            "step": self.step,
            "dataset_states": self.dataset_states,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ResumeState":
        """Deserialize from dictionary."""
        return cls(
            shard_index=data.get("shard_index", 0),
            row_index=data.get("row_index", 0),
            epoch=data.get("epoch", 0),
            step=data.get("step", 0),
            dataset_states=data.get("dataset_states", {}),
        )
