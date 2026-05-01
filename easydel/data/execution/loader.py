# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Thread

import numpy as np

from easydel.infra.sharding import MeshLike

from ..core.config import LoadStageConfig
from ..core.protocols import AsyncDataset, BaseStage, PipelineContext, ShardedDataSource

if tp.TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from jax.sharding import NamedSharding

logger = logging.getLogger(__name__)


def collate_batch(examples: list[dict]) -> dict[str, np.ndarray]:
    """Stack a list of identically-keyed example dicts into a batched dict-of-arrays.

    For each key, attempts a vectorised :func:`numpy.stack` along axis 0;
    when that fails (heterogeneous shapes/dtypes — e.g. variable-length
    token lists in unpacked data), falls back to a numpy object array
    so the values are still preserved in batch shape without raising.

    Args:
        examples: List of per-row dicts. All entries are assumed to
            share the same key set; only ``examples[0].keys()`` is
            consulted for which fields to materialise. An empty list
            short-circuits to an empty dict.

    Returns:
        dict[str, np.ndarray]: Mapping from each key to either a stacked
        array (when all values are array-compatible) or an object array
        (fallback for ragged/unstackable values).
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
    """Group rows from a streaming source into fixed-size batches and collate each.

    Walks ``source`` accumulating rows into a buffer; once the buffer
    reaches ``batch_size`` it is collated via :func:`collate_batch` and
    yielded, and the buffer is reset. Whether the final, possibly
    incomplete batch is yielded is controlled by ``drop_last``.

    Args:
        source: Iterator yielding per-row dicts (typically the output
            of a sharded source or transform chain).
        batch_size: Number of rows to combine into one batch. Must be
            positive.
        drop_last: When ``True`` (the default), trailing rows that do
            not fill a complete batch are discarded. When ``False``,
            they are emitted as a final, smaller batch — useful for
            evaluation where shape stability is not required.

    Yields:
        dict[str, np.ndarray]: Each batch as a key-aligned dict of
        numpy arrays produced by :func:`collate_batch`.
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
    """Background-thread prefetcher that keeps the trainer-thread fed.

    Wraps an upstream iterator with a single-thread producer that
    drains it into a bounded :class:`~queue.Queue`. The consumer
    (typically the training loop) pulls from the queue via the regular
    iterator protocol, so I/O and CPU preprocessing in the source
    overlap with model execution. A sentinel object signals end of
    stream, and exceptions raised by the source are placed onto the
    queue and re-raised on the consumer side so errors are not
    silently swallowed.
    """

    def __init__(
        self,
        source: "Iterator",
        buffer_size: int = 4,
        num_workers: int = 2,
    ):
        """Configure the prefetch buffer without starting the background thread.

        Args:
            source: Iterator to consume in the background. Drained
                lazily — no I/O happens until iteration starts.
            buffer_size: Maximum number of pre-produced batches kept
                in the queue at a time. Bounds memory usage.
            num_workers: Currently ignored; kept for API compatibility
                with earlier multi-worker implementations. Only one
                worker thread is started regardless.
        """
        del num_workers  # unused, kept for API compatibility
        self._source = source
        self._buffer = Queue(maxsize=buffer_size)
        self._stop_event = Event()
        self._sentinel = object()
        self._worker = None
        self._started = False

    def _prefetch_worker(self):
        """Background-thread body that drains the source into the prefetch queue.

        Iterates the source in a loop, pushing each item onto the
        buffer. Cooperatively checks ``self._stop_event`` so the
        producer can be torn down on demand. Exceptions from the
        source are captured and forwarded onto the queue so the
        consumer can re-raise them in its own context. A sentinel
        object is always pushed last (in the ``finally`` block) so
        the consumer terminates cleanly regardless of how the source
        ended.
        """
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
        """Lazily spawn the producer thread on the first call to :meth:`__next__`.

        Safe to call repeatedly — only the first invocation starts the
        thread; subsequent calls return without doing anything. Keeps
        :class:`PrefetchIterator` cheap to construct (e.g. in tests
        that never iterate it).
        """
        if not self._started:
            self._worker = Thread(
                target=self._prefetch_worker,
                daemon=True,
            )
            self._worker.start()
            self._started = True

    def __iter__(self):
        """Iterator protocol: this class is its own iterator.

        Returns:
            PrefetchIterator: ``self``, so ``for x in iterator`` works
            without an additional iter call.
        """
        return self

    def __next__(self):
        """Pop the next item from the prefetch queue, blocking briefly if empty.

        Lazily starts the producer thread on the first call and uses a
        60-second blocking ``get`` to pull from the queue. The
        sentinel object signals end-of-stream; an :class:`Exception`
        instance signals the producer hit an error and the same
        exception is re-raised here.

        Returns:
            Any: The next item produced by the upstream source.

        Raises:
            StopIteration: When the source is exhausted (sentinel
                received) or the queue blocks for longer than 60s
                without any activity (treated as end-of-stream rather
                than a hang).
            Exception: Whatever exception type the upstream source
                raised; transparently propagated through the queue.
        """
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
        """Signal the producer thread to stop and join with a 1-second timeout.

        Idempotent: safe to call from multiple sites (e.g. consumer
        teardown plus context-manager exit). Uses a daemon thread, so
        even if the join times out the worker will not block process
        exit.
        """
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)


@dataclass
class ShardingSpec:
    """Per-field :class:`jax.sharding.PartitionSpec` map applied to numpy batches.

    Used by :meth:`apply` (and indirectly by ``preshard_batch``) to
    move every array in a batch onto its target devices using
    ``jax.device_put`` with an MPMD-aware
    :func:`spectrax.get_corrected_named_sharding`. Keeping this as a
    declarative struct (rather than baking sharding into each field)
    lets the same batch flow through different sharding regimes
    cheaply during PP/MPMD experiments.

    Attributes:
        mesh (MeshLike | None): Spectrax/JAX mesh on which the
            partition specs are realised. ``None`` disables sharding —
            :meth:`apply` returns the batch untouched.
        partition_specs (dict[str, Any]): Map from batch field name to
            :class:`jax.sharding.PartitionSpec` (typed as ``Any`` to
            avoid an eager JAX import). Fields without an entry default
            to a fully replicated spec.
    """

    mesh: MeshLike | None = None
    partition_specs: dict[str, tp.Any] = field(default_factory=dict)  # field -> PartitionSpec

    def apply(self, batch: dict[str, np.ndarray]) -> dict[str, tp.Any]:
        """Move every array in ``batch`` onto devices according to the spec.

        For each field, builds the per-array
        :class:`jax.sharding.NamedSharding` via
        :func:`spectrax.get_corrected_named_sharding` (which is
        MPMD-aware: it routes the requested spec through the resolver
        onto the active per-stage submesh, avoiding the bug where a
        hand-rolled ``NamedSharding`` collapses pipeline submeshes back
        to the full mesh) and calls ``jax.device_put``.

        Args:
            batch: Mapping of field names to numpy arrays.

        Returns:
            dict[str, Any]: Dict with the same keys, but values are
            sharded ``jax.Array`` instances. When :attr:`mesh` is
            ``None`` the original batch is returned unchanged.
        """
        if self.mesh is None:
            return batch

        import jax
        import spectrax as spx
        from jax.sharding import PartitionSpec

        # # @erfanzar NOTE: get_corrected_named_sharding is MPMD-aware -- it
        # routes the spec through the resolver onto the per-stage submesh and
        # sanitizes for the concrete shape.  Hand-rolling NamedSharding(mesh,
        # spec) would collapse PP submeshes back to the full mesh, the same
        # bug class as init_tx had.
        result = {}
        for key, arr in batch.items():
            spec = self.partition_specs.get(key, PartitionSpec())
            sharding = spx.get_corrected_named_sharding(tuple(arr.shape), spec, raise_mesh_error=False)
            result[key] = jax.device_put(arr, sharding)

        return result


def preshard_batch(
    batch: dict[str, np.ndarray],
    sharding_map: "Mapping[str, NamedSharding] | None",
) -> dict[str, tp.Any]:
    """Pre-place each batch field onto devices using a precomputed sharding map.

    Lighter-weight alternative to :class:`ShardingSpec.apply` for use
    inside the prefetch loop: the caller has already resolved
    :class:`jax.sharding.NamedSharding` instances (typically from
    :meth:`AsyncDataset.get_output_sharding`) and just needs to apply
    them via ``jax.device_put``. Fields missing from the map are left
    as their original numpy arrays so the host-to-device transfer can
    happen later in the trainer thread.

    Args:
        batch: Mapping of field names to numpy arrays produced by
            :func:`collate_batch` (or equivalent).
        sharding_map: Resolved ``{field: NamedSharding}`` map. ``None``
            disables pre-sharding; the batch is returned unchanged.

    Returns:
        dict[str, Any]: Dict with the same keys; values are
        ``jax.Array`` instances when a sharding was supplied for the
        key, otherwise the unchanged numpy array.
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
    """Streaming async loader that batches a :class:`ShardedDataSource` for JAX trainers.

    Stitches together shard iteration, optional reservoir-sampling
    shuffle, fixed-size batching, optional pre-sharding, and a
    background prefetch thread to deliver host-collated (or device-placed)
    batches to the training step. Implements
    :class:`AsyncDataset[dict]` so it plugs into the rest of the
    pipeline without manual conversion. Note that random-access
    :meth:`aget` is not supported — this loader is streaming-only.
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
        """Capture loader configuration without performing any I/O.

        All work is deferred until iteration starts via
        :meth:`__aiter__` or :meth:`__iter__`, so constructing the
        loader is cheap.

        Args:
            source: Underlying :class:`ShardedDataSource` whose shards
                are concatenated to form the row stream.
            batch_size: Number of rows combined into one batch.
            prefetch_enabled: When ``True``, wraps the batch stream in
                a :class:`PrefetchIterator` so I/O overlaps with the
                training step.
            prefetch_workers: Forwarded to :class:`PrefetchIterator`;
                see that class for the (currently advisory) semantics.
            prefetch_buffer_size: Queue depth used by the prefetch
                iterator.
            shuffle_buffer_size: Reservoir size for streaming shuffle.
                ``None`` disables shuffling.
            drop_last: Whether to drop a trailing partial batch
                (preserves static shape required by JIT).
            sharding_map: Optional ``{field: NamedSharding}`` map used
                to pre-shard each batch via :func:`preshard_batch`.
            seed: RNG seed for the shuffle reservoir; ``None`` keeps
                Python's default (non-deterministic) randomness.
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
        """Random access is not supported by this streaming loader.

        Args:
            _index: Ignored; kept for protocol compatibility.

        Raises:
            NotImplementedError: Always; AsyncDataLoader is streaming-only.
        """
        raise NotImplementedError("AsyncDataLoader does not support random access")

    async def __aiter__(self):
        """Yield successive batches over an async iterator backed by a thread pool.

        Builds the underlying sync batch iterator with
        :meth:`_create_iterator`, then runs each ``next()`` call on a
        single-thread executor via ``loop.run_in_executor`` so the
        event loop stays responsive even when the upstream pipeline
        blocks on I/O. Sets :attr:`is_exhausted` to ``True`` once the
        underlying iterator drains.

        Yields:
            dict[str, Any]: Each batch dict, optionally with
            pre-sharded ``jax.Array`` values when ``sharding_map`` was
            supplied.
        """
        # Create the underlying iterator
        iterator = self._create_iterator()

        # Wrap in executor for async
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        def get_next():
            """Inline closure: produce the next batch on a worker thread.

            Captures ``iterator`` from the enclosing scope so the
            executor can advance it. Returning ``None`` on
            :class:`StopIteration` lets the awaiter use a regular
            falsy check instead of catching ``StopIteration`` across
            the executor boundary (which Python forbids).

            Returns:
                dict | None: The next batch, or ``None`` when the
                upstream iterator is exhausted.
            """
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
        """Compose the full sync iterator pipeline used by both async and sync entry points.

        Pipeline order (each stage optional except shard iter and
        batching):

        1. Iterate every shard from :attr:`_source`.
        2. Reservoir-sample shuffle when ``shuffle_buffer_size`` is set.
        3. Group into ``batch_size`` chunks via :func:`batch_iterator`.
        4. Pre-shard with :func:`preshard_batch` when ``sharding_map``
           is supplied.
        5. Wrap in a :class:`PrefetchIterator` when ``prefetch_enabled``.

        Returns:
            Iterator[dict]: Iterator yielding batch dicts ready to feed
            into the training step.
        """

        # Chain all shards
        def iter_examples():
            """Inline closure that flattens shard iteration into a single example stream.

            Walks every shard name in :attr:`_source.shard_names` in
            order, opens it via :meth:`ShardedDataSource.open_shard`,
            and yields its examples before moving to the next.

            Yields:
                dict: Individual rows from every shard, in shard
                order then in shard-local order.
            """
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
        """Reservoir-sample shuffle for streaming sources of unknown length.

        Fills a fixed-size reservoir with the first ``buffer_size``
        examples; thereafter, for each new example, picks a random
        slot to evict (yielding the evicted item and replacing it
        in-place). When the source is exhausted, the remaining buffer
        is shuffled and drained. Memory is bounded by ``buffer_size``;
        shuffle quality improves with larger buffers.

        Args:
            stream: Iterator yielding the rows to shuffle.
            buffer_size: Reservoir size — examples kept in memory at
                a time. Trades memory for shuffle quality.

        Yields:
            dict: Rows from ``stream`` in pseudo-random order.
        """
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
        """Expose the loader's sharding map so downstream consumers can pre-place tensors.

        Implements the
        :meth:`AsyncDatasetProtocol.get_output_sharding` hook.

        Returns:
            Mapping[str, NamedSharding] | None: Whatever was passed to
            the constructor as ``sharding_map``; ``None`` when
            sharding was not configured.
        """
        return self._sharding_map

    @property
    def is_exhausted(self) -> bool:
        """End-of-stream flag set when the async iterator finishes.

        Returns:
            bool: ``True`` once :meth:`__aiter__` has drained the
            underlying iterator and exited normally; ``False`` before
            iteration starts or while it is still running.
        """
        return self._exhausted

    def __iter__(self) -> "Iterator[dict]":
        """Synchronous bridge: return the underlying batch iterator directly.

        Bypasses the async/event-loop wrapper, which is convenient
        for single-process inference and tests but does not benefit
        from any async parallelism.

        Returns:
            Iterator[dict]: The same iterator returned by
            :meth:`_create_iterator`.
        """
        return self._create_iterator()


class LoadStage(BaseStage):
    """Pipeline stage that converts ``{name: source}`` mappings into async loaders.

    The final stage of the typed pipeline: takes a dict of named
    :class:`ShardedDataSource` instances (the output of preceding
    stages) and wraps each in a configured :class:`AsyncDataLoader`,
    propagating batching, shuffle, prefetch, and sharding settings
    from :class:`LoadStageConfig`. The :attr:`PipelineContext.seed` is
    threaded into each loader so reproducibility is preserved across
    constituents.
    """

    def __init__(self, config: LoadStageConfig | None = None):
        """Capture the stage configuration and forward to the base stage.

        Args:
            config: :class:`LoadStageConfig` controlling batching,
                prefetch, shuffle, and sharding behaviour.
                Defaulted to an empty :class:`LoadStageConfig` when
                ``None`` is supplied so the stage is usable in tests.
        """
        super().__init__(config.__dict__ if config else {})
        self._stage_config = config or LoadStageConfig()

    @property
    def name(self) -> str:
        """Stage identifier used in metric and log namespaces.

        Returns:
            str: The constant string ``"load"``.
        """
        return "load"

    def process(
        self,
        data: dict[str, ShardedDataSource],
        context: PipelineContext,
    ) -> dict[str, AsyncDataLoader]:
        """Wrap each named source into an :class:`AsyncDataLoader`.

        Settings on :attr:`_stage_config` apply uniformly to every
        loader; the per-pipeline RNG seed is read from
        ``context.seed`` so the same pipeline run remains
        reproducible across all constituents.

        Args:
            data: ``{dataset_name: ShardedDataSource}`` map produced
                by earlier stages.
            context: Shared pipeline context; only ``context.seed`` is
                consulted at this stage.

        Returns:
            dict[str, AsyncDataLoader]: ``{dataset_name: loader}`` with
            one loader per input source. Order matches ``data``.
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
) -> Iterator:
    """One-shot helper: shuffle, batch, and prefetch any HuggingFace-like dataset.

    Convenience function that wraps the most common dataset-iteration
    pattern: optionally shuffle (using whichever ``shuffle`` signature
    the input dataset supports — streaming or map-style), optionally
    batch into lists, and optionally drape a background prefetch
    thread over the result for I/O overlap. Unlike
    :class:`AsyncDataLoader`, this function takes already-loaded
    ``datasets.Dataset``/``IterableDataset`` objects rather than a
    :class:`ShardedDataSource`.

    Args:
        dataset: Source dataset; must expose ``shuffle`` and ``__iter__``
            in the HuggingFace ``datasets`` style.
        batch_size: Number of rows per emitted batch. ``batch_size <= 1``
            yields individual examples instead of lists.
        shuffle: Run dataset-native shuffling before batching.
        drop_last: Whether the trailing partial batch is dropped.
        prefetch: Wrap the result in a background-thread prefetcher
            with a 60-second blocking get and exception forwarding.
        prefetch_workers: Currently unused; kept for API parity with
            :class:`PrefetchIterator`.
        prefetch_buffer: Queue depth for the prefetch buffer.
        shuffle_buffer: Reservoir size for streaming-dataset shuffles
            (passed through to ``Dataset.shuffle(buffer_size=...)``);
            fallback default of 10000 used when ``None``. Map-style
            datasets that do not accept ``buffer_size`` get a plain
            seeded shuffle instead.
        seed: RNG seed for the shuffle.

    Returns:
        Iterator: Iterator yielding either lists of length
        ``batch_size`` (when ``batch_size > 1``) or individual rows.
        Wrapped in a generator with prefetching when ``prefetch`` is
        ``True``.
    """
    del prefetch_workers  # unused, kept for API compatibility
    if shuffle:
        try:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer or 10000, seed=seed)
        except TypeError:
            dataset = dataset.shuffle(seed=seed)

    def _batched(it, bs):
        """Inline closure: group rows from an iterator into fixed-size lists.

        Captures ``drop_last`` from the enclosing scope so the
        trailing-partial-batch policy follows what the caller asked
        for in :func:`create_data_iterator`.

        Args:
            it: Source iterator yielding individual rows.
            bs: Number of rows per emitted list.

        Yields:
            list: Lists of length ``bs`` while iteration is going;
            the trailing partial list is yielded only when
            ``drop_last`` is ``False``.
        """
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
            """Background-thread body for the prefetch helper inside :func:`create_data_iterator`.

            Mirrors :meth:`PrefetchIterator._prefetch_worker`: pushes
            each produced item onto the queue, captures and forwards
            any source-side exception, and finally pushes a
            module-local ``_SENTINEL`` so the consumer terminates.

            Args:
                source_iter: Iterator to drain; advanced one item per
                    iteration of the loop.
                buf: ``Queue`` shared with the consumer, used for both
                    items and exception propagation.
                stop_evt: ``Event`` polled cooperatively before each
                    push so the worker can exit early on consumer
                    teardown.
            """
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
            """Inline generator that drains the prefetch queue and re-raises errors.

            Reads from ``buffer`` with a 60-second blocking ``get``.
            Treats the captured ``_SENTINEL`` as end-of-stream and
            transparently re-raises any :class:`Exception` instance
            it pulls off the queue (forwarded by ``prefetch_worker``).
            Always sets ``stop_event`` and joins the worker on exit
            via ``finally`` so consumer shutdown is clean even on
            early break.

            Yields:
                Any: The next item from the prefetch worker.

            Raises:
                Exception: Whatever exception the source raised,
                    transparently forwarded.
            """
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
