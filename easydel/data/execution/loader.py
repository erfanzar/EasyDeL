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

"""Data loading with async prefetching and JAX sharding.

This module provides:
- Async data loading with ThreadPoolExecutor
- Pre-sharding during prefetch for JAX optimization
- Configurable batching and buffering
- Integration with the pipeline system
"""

from __future__ import annotations

import asyncio
import logging
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread

import numpy as np

from ..core.config import LoadStageConfig
from ..core.protocols import AsyncDataset, BaseStage, PipelineContext, ShardedDataSource

if tp.TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from jax.sharding import NamedSharding

logger = logging.getLogger(__name__)


def collate_batch(examples: list[dict]) -> dict[str, np.ndarray]:
    """Collate a list of examples into a batch.

    Args:
        examples: List of dictionaries with same keys.

    Returns:
        Dictionary with stacked numpy arrays.
    """
    if not examples:
        return {}

    keys = examples[0].keys()
    batch = {}

    for key in keys:
        values = [ex[key] for ex in examples]
        # Stack if they're arrays, otherwise create object array
        try:
            batch[key] = np.stack(values, axis=0)
        except (ValueError, TypeError):
            batch[key] = np.array(values, dtype=object)

    return batch


def batch_iterator(
    source: "Iterator[dict]",
    batch_size: int,
    drop_last: bool = True,
) -> "Iterator[dict[str, np.ndarray]]":
    """Create batches from an iterator.

    Args:
        source: Iterator yielding dictionaries.
        batch_size: Number of examples per batch.
        drop_last: Whether to drop incomplete final batch.

    Yields:
        Batched dictionaries with numpy arrays.
    """
    batch = []
    for item in source:
        batch.append(item)
        if len(batch) >= batch_size:
            yield collate_batch(batch)
            batch = []

    if batch and not drop_last:
        yield collate_batch(batch)


class PrefetchIterator:
    """Iterator with thread-based prefetching.

    Uses a background thread to prefetch batches into a queue,
    keeping the GPU/TPU fed with data.
    """

    def __init__(
        self,
        source: "Iterator",
        buffer_size: int = 4,
        num_workers: int = 2,
    ):
        """Initialize PrefetchIterator.

        Args:
            source: Source iterator.
            buffer_size: Size of prefetch buffer.
            num_workers: Number of worker threads (kept for API compat).
        """
        del num_workers  # unused, kept for API compatibility
        self._source = source
        self._buffer = Queue(maxsize=buffer_size)
        self._stop_event = Event()
        self._sentinel = object()
        self._worker = None
        self._started = False

    def _prefetch_worker(self):
        """Background worker that prefetches items."""
        try:
            for item in self._source:
                if self._stop_event.is_set():
                    break
                self._buffer.put(item)
        except Exception as e:
            self._buffer.put(e)
        finally:
            self._buffer.put(self._sentinel)

    def _start(self):
        """Start the prefetch worker."""
        if not self._started:
            self._worker = Thread(
                target=self._prefetch_worker,
                daemon=True,
            )
            self._worker.start()
            self._started = True

    def __iter__(self):
        return self

    def __next__(self):
        self._start()

        try:
            item = self._buffer.get(timeout=60.0)
        except Empty:
            raise StopIteration from None

        if item is self._sentinel:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        """Stop prefetching and cleanup."""
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)


@dataclass
class ShardingSpec:
    """Specification for sharding batch arrays across devices."""

    mesh: tp.Any = None  # jax.sharding.Mesh
    partition_specs: dict[str, tp.Any] = None  # field -> PartitionSpec

    def apply(self, batch: dict[str, np.ndarray]) -> dict[str, tp.Any]:
        """Apply sharding to a batch.

        Args:
            batch: Dictionary of numpy arrays.

        Returns:
            Dictionary with jax.Array values properly sharded.
        """
        if self.mesh is None:
            return batch

        import jax
        from jax.sharding import NamedSharding, PartitionSpec

        result = {}
        for key, arr in batch.items():
            spec = self.partition_specs.get(key, PartitionSpec())
            sharding = NamedSharding(self.mesh, spec)
            result[key] = jax.device_put(arr, sharding)

        return result


def preshard_batch(
    batch: dict[str, np.ndarray],
    sharding_map: "Mapping[str, NamedSharding] | None",
) -> dict[str, tp.Any]:
    """Pre-shard a batch according to sharding specifications.

    Args:
        batch: Dictionary of numpy arrays.
        sharding_map: Mapping of field names to NamedSharding.

    Returns:
        Dictionary with sharded jax arrays.
    """
    if sharding_map is None:
        return batch

    import jax

    result = {}
    for key, arr in batch.items():
        sharding = sharding_map.get(key)
        if sharding is not None:
            result[key] = jax.device_put(arr, sharding)
        else:
            result[key] = arr

    return result


class AsyncDataLoader(AsyncDataset[dict]):
    """Async-first data loader with prefetching and optional sharding.

    Designed for JAX integration with:
    - Thread-based prefetching
    - Pre-sharding during prefetch
    - Proper async/await interface
    """

    def __init__(
        self,
        source: ShardedDataSource,
        batch_size: int = 8,
        prefetch_enabled: bool = True,
        prefetch_workers: int = 2,
        prefetch_buffer_size: int = 4,
        shuffle_buffer_size: int | None = None,
        drop_last: bool = True,
        sharding_map: "Mapping[str, NamedSharding] | None" = None,
        seed: int | None = None,
    ):
        """Initialize AsyncDataLoader.

        Args:
            source: Data source to load from.
            batch_size: Number of examples per batch.
            prefetch_enabled: Whether to enable prefetching.
            prefetch_workers: Number of prefetch workers.
            prefetch_buffer_size: Size of prefetch buffer.
            shuffle_buffer_size: Buffer size for shuffling.
            drop_last: Whether to drop incomplete final batch.
            sharding_map: Mapping for pre-sharding.
            seed: Random seed.
        """
        self._source = source
        self._batch_size = batch_size
        self._prefetch_enabled = prefetch_enabled
        self._prefetch_workers = prefetch_workers
        self._prefetch_buffer_size = prefetch_buffer_size
        self._shuffle_buffer_size = shuffle_buffer_size
        self._drop_last = drop_last
        self._sharding_map = sharding_map
        self._seed = seed
        self._exhausted = False

    async def aget(self, _index: int) -> dict:
        """Async get is not supported for streaming sources."""
        raise NotImplementedError("AsyncDataLoader does not support random access")

    async def __aiter__(self):
        """Async iteration over batches."""
        # Create the underlying iterator
        iterator = self._create_iterator()

        # Wrap in executor for async
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        def get_next():
            try:
                return next(iterator)
            except StopIteration:
                return None

        try:
            while True:
                batch = await loop.run_in_executor(executor, get_next)
                if batch is None:
                    self._exhausted = True
                    break
                yield batch
        finally:
            executor.shutdown(wait=False)

    def _create_iterator(self) -> "Iterator[dict]":
        """Create the base iterator with batching and prefetching."""

        # Chain all shards
        def iter_examples():
            for shard_name in self._source.shard_names:
                yield from self._source.open_shard(shard_name)

        examples = iter_examples()

        # Shuffle if configured
        if self._shuffle_buffer_size:
            examples = self._shuffle_stream(examples, self._shuffle_buffer_size)

        # Batch
        batches = batch_iterator(examples, self._batch_size, self._drop_last)

        if self._sharding_map:
            batches = (preshard_batch(b, self._sharding_map) for b in batches)

        # Prefetch
        if self._prefetch_enabled:
            return PrefetchIterator(
                batches,
                buffer_size=self._prefetch_buffer_size,
                num_workers=self._prefetch_workers,
            )

        return batches

    def _shuffle_stream(
        self,
        stream: "Iterator[dict]",
        buffer_size: int,
    ) -> "Iterator[dict]":
        """Shuffle a stream using reservoir sampling."""
        import random

        if self._seed is not None:
            random.seed(self._seed)

        buffer = []
        for item in stream:
            if len(buffer) < buffer_size:
                buffer.append(item)
            else:
                idx = random.randrange(0, buffer_size)
                yield buffer[idx]
                buffer[idx] = item

        random.shuffle(buffer)
        yield from buffer

    def get_output_sharding(self) -> "Mapping[str, NamedSharding] | None":
        """Return the sharding specification for output batches."""
        return self._sharding_map

    @property
    def is_exhausted(self) -> bool:
        """Check if the dataset iteration is complete."""
        return self._exhausted

    def __iter__(self) -> "Iterator[dict]":
        """Synchronous iteration interface."""
        return self._create_iterator()


class LoadStage(BaseStage):
    """Pipeline stage for data loading."""

    def __init__(self, config: LoadStageConfig | None = None):
        """Initialize LoadStage.

        Args:
            config: Loading stage configuration.
        """
        super().__init__(config.__dict__ if config else {})
        self._stage_config = config or LoadStageConfig()

    @property
    def name(self) -> str:
        return "load"

    def process(
        self,
        data: dict[str, ShardedDataSource],
        context: PipelineContext,
    ) -> dict[str, AsyncDataLoader]:
        """Create data loaders from sources.

        Args:
            data: Dictionary mapping dataset names to sources.
            context: Pipeline context.

        Returns:
            Dictionary with AsyncDataLoader instances.
        """
        result = {}

        for ds_name, source in data.items():
            loader = AsyncDataLoader(
                source=source,
                batch_size=self._stage_config.batch_size,
                prefetch_enabled=self._stage_config.prefetch_enabled,
                prefetch_workers=self._stage_config.prefetch_workers,
                prefetch_buffer_size=self._stage_config.prefetch_buffer_size,
                shuffle_buffer_size=self._stage_config.shuffle_buffer_size,
                drop_last=self._stage_config.drop_last,
                seed=context.seed,
            )
            result[ds_name] = loader
            logger.info(f"Created loader for '{ds_name}' with batch_size={self._stage_config.batch_size}")

        return result


def create_data_iterator(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    prefetch: bool = True,
    prefetch_workers: int = 2,
    prefetch_buffer: int = 4,
    shuffle_buffer: int | None = None,
    seed: int | None = None,
) -> tp.Iterator:
    """Create an efficient data iterator with optional batching and prefetching.

    Creates an iterator over a dataset with support for shuffling, batching,
    and thread-based prefetching for improved performance.

    Args:
        dataset: Dataset to iterate over (Dataset or IterableDataset).
        batch_size: Number of examples per batch.
        shuffle: Whether to shuffle the data (default: True).
        drop_last: Whether to drop the last incomplete batch (default: False).
        prefetch: Whether to enable thread-based prefetching (default: True).
        prefetch_workers: Number of prefetch worker threads (default: 2, unused).
        prefetch_buffer: Size of prefetch buffer queue (default: 4).
        shuffle_buffer: Buffer size for streaming datasets (default: 10000).
        seed: Random seed for shuffling (default: None).

    Returns:
        Iterator yielding batches of data or individual examples if batch_size=1.
    """
    del prefetch_workers  # unused, kept for API compatibility
    if shuffle:
        try:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer or 10000, seed=seed)
        except TypeError:
            dataset = dataset.shuffle(seed=seed)

    def _batched(it, bs):
        batch = []
        for item in it:
            batch.append(item)
            if len(batch) >= bs:
                yield batch
                batch = []
        if batch and not drop_last:
            yield batch

    it = iter(dataset)
    if batch_size and batch_size > 1:
        it = _batched(it, batch_size)

    if prefetch:
        _SENTINEL = object()
        buffer = Queue(maxsize=prefetch_buffer)
        stop_event = Event()

        def prefetch_worker(source_iter, buf, stop_evt):
            try:
                for item in source_iter:
                    if stop_evt.is_set():
                        break
                    buf.put(item)
            except Exception as e:
                buf.put(e)
            finally:
                buf.put(_SENTINEL)

        worker = Thread(
            target=prefetch_worker,
            args=(it, buffer, stop_event),
            daemon=True,
        )
        worker.start()

        def _gen():
            try:
                while True:
                    try:
                        item = buffer.get(timeout=60.0)
                    except Empty:
                        continue

                    if item is _SENTINEL:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                stop_event.set()
                worker.join(timeout=1.0)

        return _gen()

    return it
