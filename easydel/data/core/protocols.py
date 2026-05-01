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

"""Core protocols and abstractions for the data management pipeline.

This module defines the base interfaces for:
- Pipeline stages (Source, Tokenize, Cache, Mix, Pack, Load, Save)
- Async datasets with JAX integration
- Sharded data sources with URL support
- Pipeline context for shared state
"""

from __future__ import annotations

import importlib
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
    from jax.sharding import NamedSharding

    from .config import PipelineConfig

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
DatasetLike = "Dataset | IterableDataset | Iterator[dict]"


@dataclass
class ShardInfo:
    """Lightweight metadata descriptor for a single shard of a sharded dataset.

    Returned (optionally) from :meth:`ShardedDataSource.get_shard_info` so
    callers can plan parallelism, distribute shards across workers, or
    detect corruption without opening the underlying file. All numeric
    fields are nullable because some sources cannot determine the row
    count or byte size cheaply (e.g. line-delimited JSON over HTTP).

    Attributes:
        shard_id (int): Zero-based positional index of this shard within
            the source's :attr:`ShardedDataSource.shard_names` sequence.
        shard_name (str): The string identifier passed to
            :meth:`ShardedDataSource.open_shard` — typically a filesystem
            path or URL.
        num_rows (int | None): Number of examples in the shard when known
            (e.g. from a parquet footer). ``None`` when the source cannot
            cheaply enumerate rows.
        byte_size (int | None): On-disk size of the shard in bytes, or
            ``None`` when unavailable.
        url (str | None): Fully qualified URL/URI for the shard payload
            (``gs://``, ``s3://``, ``hf://``, …); separated from
            ``shard_name`` so opaque identifiers and concrete locations
            can coexist.
        checksum (str | None): Integrity hash for the shard contents
            (e.g. an MD5 or SHA256 hex digest); used by caches and
            distributed runners to detect drift.
    """

    shard_id: int
    shard_name: str
    num_rows: int | None = None
    byte_size: int | None = None
    url: str | None = None
    checksum: str | None = None


class ShardedDataSource(ABC, Generic[T_co]):
    """Abstract base class for the EasyDeL shard-oriented data source API.

    Modelled after Levanter's ``ShardedDataSource``, this contract is the
    fundamental abstraction the rest of the data layer is built on. A
    sharded source exposes a fixed list of opaque ``shard_names``
    (typically file paths or URIs), and an :meth:`open_shard` factory that
    returns an iterator of in-memory examples for any one of them. The
    base class additionally bolts on a fluent transform DSL —
    :meth:`map`, :meth:`filter`, :meth:`rename_fields`, etc. — that
    returns wrapped sources implementing the same protocol, so chains of
    transforms can be composed without losing the resume/distributed
    properties of the underlying shards.

    Resumption: :meth:`open_shard_at_row` and :meth:`iter_shards` together
    let trainers checkpoint at ``(shard_index, row)`` granularity.
    Subclasses are encouraged to override ``open_shard_at_row`` for
    sources that natively support seeking (e.g. Parquet row groups).

    Distribution: callers can pass a subset of shard indices to
    :meth:`iter_shards` to fan a single source out across multiple
    workers without coordination.

    Example:
        >>> source = ParquetShardedSource("gs://bucket/data/*.parquet")
        >>> for shard_name in source.shard_names[:10]:
        ...     for example in source.open_shard(shard_name):
        ...         process(example)
    """

    @property
    @abstractmethod
    def shard_names(self) -> Sequence[str]:
        """Stable, ordered list of shard identifiers exposed by this source.

        Each returned name is an opaque token accepted by
        :meth:`open_shard` — usually a filesystem path or URI but it may
        also be an internal tag (e.g. ``"shard-0023"``). The order must
        be deterministic across calls so that ``shard_id`` indices remain
        meaningful for resumption.

        Returns:
            Sequence[str]: Ordered shard identifiers. May be empty when
            the source has no data.
        """
        ...

    @abstractmethod
    def num_shards(self) -> int:
        """Return the count of shards available from this source.

        Returns:
            int: Number of shards; equal to ``len(self.shard_names)``.
        """
        ...

    @abstractmethod
    def open_shard(self, shard_name: str) -> Iterator[T_co]:
        """Open a shard and yield its examples one at a time.

        Args:
            shard_name: One of the values returned by
                :attr:`shard_names`. Passing an unknown name should raise
                ``KeyError`` or an analogous error from the concrete
                implementation.

        Returns:
            Iterator[T_co]: Iterator producing examples in shard-local
            order. Implementations are free to be lazy — no I/O is
            required until iteration starts.
        """
        ...

    def get_shard_info(self, shard_name: str) -> ShardInfo | None:
        """Best-effort metadata lookup for a single shard.

        Default implementation returns ``None`` because most sources do
        not eagerly compute counts/sizes/checksums; concrete sources
        backed by formats with cheap metadata (Parquet, Arrow) are
        expected to override.

        Args:
            shard_name: Identifier of the shard whose metadata is
                requested. Must be a member of :attr:`shard_names`.

        Returns:
            ShardInfo | None: Populated info object when available,
            otherwise ``None``.
        """
        return None

    def open_shard_at_row(
        self,
        shard_name: str,
        row: int,
    ) -> Iterator[T_co]:
        """Open a shard and skip the first ``row`` examples for fast resume.

        The default implementation iterates and discards leading rows —
        correct but O(row). Subclasses backed by random-access formats
        (Parquet row groups, Arrow record batches, indexed shards) should
        override to seek directly.

        Args:
            shard_name: Identifier of the shard to open.
            row: Number of leading rows to discard before yielding.
                ``0`` is equivalent to :meth:`open_shard`.

        Returns:
            Iterator[T_co]: Iterator over the remaining examples after
            ``row`` items have been skipped.
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
        """Iterate over a (possibly filtered) subset of shards with resume support.

        Combines distributed shard assignment (``shard_indices``) with
        checkpoint-aware resumption (``start_shard`` + ``start_row``).
        The iteration order matches the ordering of ``shard_indices`` (or
        ``range(num_shards())`` when omitted), and exactly one shard —
        the one at position ``start_shard`` — is opened with
        :meth:`open_shard_at_row`; the rest start from the beginning.

        Args:
            shard_indices: Explicit subset of indices into
                :attr:`shard_names` to iterate, in the desired order.
                ``None`` iterates every shard.
            start_shard: Position within ``shard_indices`` (or the full
                shard list) at which iteration should begin; earlier
                positions are skipped entirely.
            start_row: Number of rows to skip inside the shard at
                ``start_shard``; ignored for subsequent shards.

        Yields:
            T_co: Examples from each visited shard, in shard-local order.
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
        """Wrap this source so ``fn`` is lazily applied to every example.

        No data is read or transformed until the wrapper is iterated; the
        wrapper preserves the underlying shard layout and resume
        semantics, so distributed/checkpointed iteration still works.

        Args:
            fn: Pure callable applied to each example as it streams
                through. Should not have side effects since it may be
                invoked multiple times (e.g. across resumed runs).

        Returns:
            MappedShardedDataSource[T]: A new sharded source whose
            elements are ``fn(x)`` for every ``x`` produced by ``self``.
        """
        return MappedShardedDataSource(self, fn)

    def filter(
        self,
        predicate: tp.Callable[[T_co], bool],
    ) -> "ShardedDataSource[T_co]":
        """Wrap this source so only examples passing ``predicate`` are yielded.

        Filtering is lazy — predicates are evaluated as data streams
        through. Internally constructs a
        :class:`~easydel.data.transforms.source.TransformedShardedSource`
        wrapping a :class:`~easydel.data.transforms.filter_ops.FilterTransform`
        so the result is itself a :class:`ShardedDataSource`.

        Args:
            predicate: Callable returning ``True`` for examples to keep
                and ``False`` to drop. Must be deterministic for
                reproducible iteration.

        Returns:
            ShardedDataSource[T_co]: A new sharded source emitting the
            subset of examples that satisfy ``predicate``.
        """
        filter_ops = importlib.import_module("easydel.data.transforms.filter_ops")
        source_mod = importlib.import_module("easydel.data.transforms.source")
        filter_transform = filter_ops.FilterTransform
        transformed_source = source_mod.TransformedShardedSource

        transformed = transformed_source(
            tp.cast("ShardedDataSource[dict[str, Any]]", self),
            filter_transform(tp.cast(tp.Callable[[dict[str, Any]], bool], predicate)),
        )
        return tp.cast("ShardedDataSource[T_co]", transformed)

    def __len__(self) -> int:
        """Total number of examples across all shards (when finite-known).

        The base implementation is a strict opt-out: streaming sources
        and any source whose total count is not statically known raise
        ``TypeError``. Concrete sources backed by formats with row counts
        in their metadata (Parquet, Arrow) override this.

        Raises:
            TypeError: Always, in the default base-class implementation.
                Streaming/unknown-length sources do not support ``len()``.
        """
        raise TypeError(f"{type(self).__name__} has no len()")

    def transform(
        self,
        transform: "tp.Any",  # Transform type
    ) -> "ShardedDataSource":
        """Wrap this source with an arbitrary :class:`Transform` instance.

        General-purpose escape hatch when the more specific helpers
        (:meth:`map`, :meth:`filter`, :meth:`rename_fields`,
        :meth:`apply_chat_template`, …) do not match. The transform may
        be a single :class:`Transform` or a
        :class:`~easydel.data.transforms.base.ChainedTransform`.

        Args:
            transform: A transform implementing the
                :class:`~easydel.data.transforms.base.Transform` protocol
                (per-row callable with ``__call__`` taking a row dict).

        Returns:
            ShardedDataSource: A new sharded source applying
            ``transform`` lazily during iteration.
        """
        source_mod = importlib.import_module("easydel.data.transforms.source")
        transformed_source = source_mod.TransformedShardedSource
        return transformed_source(tp.cast("ShardedDataSource[dict[str, Any]]", self), transform)

    def rename_fields(
        self,
        mapping: dict[str, str],
    ) -> "ShardedDataSource":
        """Rename keys in each row according to ``mapping``.

        Useful for aligning schemas across mixed datasets so a single
        downstream tokenizer/transform can operate uniformly. Backed by
        :class:`~easydel.data.transforms.field_ops.RenameFields`.

        Args:
            mapping: ``{old_key: new_key}`` rename map applied in-place
                to every row dict. Keys that do not appear in a row are
                silently ignored.

        Returns:
            ShardedDataSource: A new sharded source whose rows have had
            their keys renamed.
        """
        field_ops = importlib.import_module("easydel.data.transforms.field_ops")
        source_mod = importlib.import_module("easydel.data.transforms.source")
        rename_fields = field_ops.RenameFields
        transformed_source = source_mod.TransformedShardedSource
        return transformed_source(
            tp.cast("ShardedDataSource[dict[str, Any]]", self),
            rename_fields(mapping),
        )

    def select_fields(
        self,
        fields: list[str],
    ) -> "ShardedDataSource":
        """Project rows down to a chosen subset of fields.

        Backed by
        :class:`~easydel.data.transforms.field_ops.SelectFields`.
        Useful before tokenization to drop large columns that would
        otherwise be carried through the pipeline unnecessarily.

        Args:
            fields: Names of the keys to retain in each row dict.
                Missing keys produce no entry in the output (no error).

        Returns:
            ShardedDataSource: A new sharded source whose rows contain
            only the listed fields.
        """
        field_ops = importlib.import_module("easydel.data.transforms.field_ops")
        source_mod = importlib.import_module("easydel.data.transforms.source")
        select_fields = field_ops.SelectFields
        transformed_source = source_mod.TransformedShardedSource
        return transformed_source(
            tp.cast("ShardedDataSource[dict[str, Any]]", self),
            select_fields(fields),
        )

    def drop_fields(
        self,
        fields: list[str],
    ) -> "ShardedDataSource":
        """Remove the listed keys from every row (the complement of :meth:`select_fields`).

        Backed by :class:`~easydel.data.transforms.field_ops.DropFields`.

        Args:
            fields: Names of keys to delete from each row dict. Names
                that are not present in a particular row are silently
                ignored.

        Returns:
            ShardedDataSource: A new sharded source whose rows no longer
            contain the named fields.
        """
        field_ops = importlib.import_module("easydel.data.transforms.field_ops")
        source_mod = importlib.import_module("easydel.data.transforms.source")
        drop_fields = field_ops.DropFields
        transformed_source = source_mod.TransformedShardedSource
        return transformed_source(
            tp.cast("ShardedDataSource[dict[str, Any]]", self),
            drop_fields(fields),
        )

    def apply_chat_template(
        self,
        tokenizer: tp.Any,
        messages_field: str = "messages",
        output_field: str = "text",
        **kwargs,
    ) -> "ShardedDataSource":
        """Render conversational rows into model-ready text using the tokenizer's chat template.

        Reads structured ``messages`` arrays (``[{"role": ..., "content": ...}, ...]``)
        from each row, runs them through ``tokenizer.apply_chat_template``,
        and writes the resulting flat string back into ``output_field``.
        Backed by
        :class:`~easydel.data.transforms.chat_template.ChatTemplateTransform`.

        Args:
            tokenizer: A HuggingFace tokenizer (or compatible object)
                exposing ``apply_chat_template``. Must define a chat
                template — error is raised at iteration time otherwise.
            messages_field: Row key holding the list of message dicts.
                Defaults to ``"messages"``, matching the OpenAI/HF
                convention.
            output_field: Row key under which the rendered string is
                stored. Defaults to ``"text"`` so downstream tokenization
                stages can pick it up without configuration.
            **kwargs: Extra keyword arguments forwarded verbatim to
                :class:`ChatTemplateTransform` — typically
                ``add_generation_prompt``, ``tokenize=False``, or custom
                template flags.

        Returns:
            ShardedDataSource: A new sharded source emitting rows
            augmented with a rendered ``output_field``.
        """
        chat_template_mod = importlib.import_module("easydel.data.transforms.chat_template")
        source_mod = importlib.import_module("easydel.data.transforms.source")
        chat_template_transform = chat_template_mod.ChatTemplateTransform
        transformed_source = source_mod.TransformedShardedSource

        transform = chat_template_transform(
            tokenizer=tokenizer,
            messages_field=messages_field,
            output_field=output_field,
            **kwargs,
        )
        return transformed_source(tp.cast("ShardedDataSource[dict[str, Any]]", self), transform)


class MappedShardedDataSource(ShardedDataSource[T], Generic[T]):
    """Lightweight :class:`ShardedDataSource` wrapper that maps a function over rows.

    Constructed by :meth:`ShardedDataSource.map`. The wrapper forwards
    shard discovery and metadata to the underlying source unchanged and
    only intercepts :meth:`open_shard` to inject a per-row map. Because
    iteration laziness is preserved, the mapped source can still be used
    with the resume/distribute machinery on the base class.
    """

    def __init__(
        self,
        source: ShardedDataSource,
        fn: tp.Callable[[Any], T],
    ):
        """Capture the underlying source and the per-example map function.

        Args:
            source: The wrapped :class:`ShardedDataSource`. Iteration is
                delegated to it; only the per-row payload is transformed.
            fn: Pure callable invoked once per yielded example. Should
                be deterministic so repeated/resumed iteration produces
                identical streams.
        """
        self._source = source
        self._fn = fn

    @property
    def shard_names(self) -> Sequence[str]:
        """Pass-through to the wrapped source's shard list.

        Returns:
            Sequence[str]: Identical to the wrapped source — the map
            transform does not alter shard layout.
        """
        return self._source.shard_names

    def num_shards(self) -> int:
        """Pass-through to the wrapped source's shard count.

        Returns:
            int: ``self._source.num_shards()``.
        """
        return self._source.num_shards()

    def open_shard(self, shard_name: str) -> Iterator[T]:
        """Open the underlying shard and yield ``fn(x)`` for each example.

        Args:
            shard_name: One of :attr:`shard_names`. Forwarded verbatim
                to the wrapped source.

        Yields:
            T: ``self._fn(x)`` for each ``x`` produced by the wrapped
            source's iterator over ``shard_name``.
        """
        for example in self._source.open_shard(shard_name):
            yield self._fn(example)

    def __len__(self) -> int:
        """Length of the underlying source (no rows are added or dropped).

        Returns:
            int: ``len(self._source)``.

        Raises:
            TypeError: If the wrapped source is streaming and does not
                support ``len()``.
        """
        return len(self._source)


@runtime_checkable
class AsyncDatasetProtocol(Protocol[T_co]):
    """Structural protocol for datasets exposing an async-first JAX-friendly API.

    Anything implementing this protocol can be plugged into the
    :class:`~easydel.data.execution.loader.AsyncDataLoader`. The protocol
    requires both random access (``aget``) and iteration (``__aiter__``)
    in async form, plus two metadata hooks the loader uses to coordinate
    with the JAX trainer:

    * :meth:`get_output_sharding` advertises desired
      :class:`jax.sharding.NamedSharding` for each batch field so the
      prefetch worker can host->device place tensors in parallel with
      the training step.
    * :attr:`is_exhausted` lets the loader detect end-of-stream without
      relying on iterator exhaustion semantics, which matters for
      restartable streaming sources.
    """

    async def aget(self, index: int) -> T_co:
        """Asynchronously retrieve the dataset item at ``index``.

        Args:
            index: Zero-based row index. Implementations may raise for
                out-of-range indices.

        Returns:
            T_co: The materialised dataset item at the given index.
        """
        ...

    async def __aiter__(self) -> AsyncIterator[T_co]:
        """Asynchronously iterate over the entire dataset.

        Returns:
            AsyncIterator[T_co]: Iterator over every example in the
            dataset's preferred order.
        """
        ...

    def get_output_sharding(self) -> Mapping[str, "NamedSharding"] | None:
        """Sharding hints used by the loader to pre-place batch fields onto devices.

        Returns:
            Mapping[str, NamedSharding] | None: Per-field
            :class:`jax.sharding.NamedSharding` when sharding is
            requested, else ``None`` to leave host->device transfer to
            the trainer.
        """
        ...

    @property
    def is_exhausted(self) -> bool:
        """Cheap check the loader uses to decide when to stop polling for data.

        Returns:
            bool: ``True`` once the dataset has produced its final item
            and will yield nothing more, even on a fresh iterator.
        """
        ...


class AsyncDataset(ABC, Generic[T]):
    """Concrete base class for async-first datasets with a sync compatibility layer.

    Subclasses implement :meth:`aget` and :meth:`__aiter__`; the base
    class provides :meth:`abatch` (concurrent gather), and synchronous
    bridges (:meth:`get`, :meth:`__iter__`) that drive the async API
    from a freshly created event loop when no running loop is detected.
    The sharding/exhausted hooks default to safe no-op values so simple
    subclasses do not need to override them.

    The class is purpose-built for JAX trainers: subclasses are expected
    to support pre-sharding during prefetch, proper JAX PRNG key
    handling, and memory-efficient streaming.
    """

    @abstractmethod
    async def aget(self, index: int) -> T:
        """Asynchronously retrieve the dataset item at ``index``.

        Args:
            index: Zero-based row index. Subclasses decide how to
                handle out-of-range or non-deterministic random-access
                semantics (raise, wrap, or return default).

        Returns:
            T: The fully materialised dataset item.
        """
        ...

    @abstractmethod
    async def __aiter__(self) -> AsyncIterator[T]:
        """Asynchronously iterate over the dataset's examples.

        Returns:
            AsyncIterator[T]: Iterator over every example in the
            subclass's preferred order.
        """
        ...

    async def abatch(self, indices: Sequence[int]) -> list[T]:
        """Concurrently fetch several items via :meth:`aget` and return them in input order.

        Implemented with :func:`asyncio.gather`, which is correct for
        I/O-bound :meth:`aget` implementations but may not provide
        speedup if ``aget`` is CPU-bound.

        Args:
            indices: Indices to fetch. Must be safe to pass to
                :meth:`aget` in arbitrary order.

        Returns:
            list[T]: Items aligned positionally with ``indices``.
        """
        import asyncio

        return list(await asyncio.gather(*[self.aget(i) for i in indices]))

    def get_output_sharding(self) -> Mapping[str, "NamedSharding"] | None:
        """Per-field sharding hints; default returns ``None`` (no pre-sharding).

        Subclasses that want the loader to pre-shard batches should
        override and return a mapping from batch field name to
        :class:`jax.sharding.NamedSharding`.

        Returns:
            Mapping[str, NamedSharding] | None: ``None`` by default.
        """
        return None

    @property
    def is_exhausted(self) -> bool:
        """Cheap end-of-stream signal; defaults to ``False`` (always more data).

        Subclasses backed by finite streams should override this to
        report ``True`` once they have nothing left.

        Returns:
            bool: ``False`` by default.
        """
        return False

    # Synchronous interface (wraps async)
    def get(self, index: int) -> T:
        """Synchronous wrapper that drives :meth:`aget` via a temporary event loop.

        Reuses the running loop when called from inside one; otherwise
        creates a new one with :func:`asyncio.new_event_loop`. Note that
        calling this from inside a running loop may not work as expected
        since :meth:`asyncio.AbstractEventLoop.run_until_complete` cannot
        be reentered.

        Args:
            index: Zero-based item index, forwarded to :meth:`aget`.

        Returns:
            T: The item produced by :meth:`aget`.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.aget(index))

    def __iter__(self) -> Iterator[T]:
        """Materialise the async iterator into a list and return a sync iterator.

        Buffers the full async iteration into memory before yielding,
        which is fine for finite, modestly sized datasets used in tests
        but not appropriate for production streaming workloads.

        Returns:
            Iterator[T]: Sync iterator over the collected items.
        """
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
    """Structural protocol every pipeline stage implements.

    A stage is the unit of work that flows data from one shape into
    another inside :class:`~easydel.data.execution.pipeline.Pipeline`. It
    must expose a :attr:`name` (used in metric/log output) and a
    :meth:`process` method that takes the upstream data plus a shared
    :class:`PipelineContext` and returns a value to feed into the next
    stage. ``validate_config`` is an optional pre-flight check the
    pipeline runner consults before starting.
    """

    @property
    def name(self) -> str:
        """Short identifier for the stage, used in logs and metric keys.

        Returns:
            str: Human-readable stage name (e.g. ``"source"``,
            ``"tokenize"``).
        """
        ...

    def process(
        self,
        data: Any,
        context: "PipelineContext",
    ) -> Any:
        """Run the stage's transformation on the data flowing through it.

        Args:
            data: Whatever the upstream stage produced — typically a
                dataset, iterator, or ``{name: dataset}`` mapping.
            context: Shared :class:`PipelineContext` carrying global
                configuration, cached tokenizers, step/epoch counters,
                and metrics. Stages may both read from and record into
                the context.

        Returns:
            Any: The data forwarded to the next stage. Type depends on
            the concrete stage.
        """
        ...

    def validate_config(self, config: dict) -> bool:
        """Pre-flight check on the stage's portion of the pipeline config.

        Args:
            config: Stage-local configuration dictionary (typically the
                serialised form of one of the ``*StageConfig``
                dataclasses).

        Returns:
            bool: ``True`` when the configuration is acceptable, ``False``
            to signal the pipeline runner to abort.
        """
        ...


class BaseStage(ABC):
    """Concrete base class for pipeline stages with shared infrastructure.

    Provides storage for the stage-local configuration dict, an internal
    metrics map (consumed by :meth:`get_metrics`), and a helper for
    looking up :class:`DatasetConfig` entries by id from the
    :class:`PipelineContext`. Subclasses are required to implement only
    :attr:`name` and :meth:`process`; ``validate_config`` defaults to
    permissive behaviour and can be overridden when stricter checks are
    needed.
    """

    def __init__(self, config: dict | None = None):
        """Capture the stage's configuration dict and initialise metric storage.

        Args:
            config: Stage-local configuration dictionary, typically
                derived from one of the ``*StageConfig`` dataclasses.
                ``None`` is treated as an empty config so every stage
                can be constructed without arguments in tests.
        """
        self._config = config or {}
        self._metrics: dict[str, Any] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier emitted alongside this stage's logs and metrics.

        Returns:
            str: Stage name (e.g. ``"tokenize"``).
        """
        ...

    @abstractmethod
    def process(
        self,
        data: Any,
        context: "PipelineContext",
    ) -> Any:
        """Apply the stage's transformation and forward the result.

        Args:
            data: Output from the upstream stage; concrete shape is
                stage-dependent.
            context: Shared :class:`PipelineContext` for cross-stage
                state (tokenizers, metrics, config).

        Returns:
            Any: The transformed data passed to the next stage.
        """
        ...

    def validate_config(self, config: dict) -> bool:
        """Permissive default validator; subclasses override for strict checks.

        Args:
            config: Stage-local configuration dictionary.

        Returns:
            bool: ``True`` always in the base class. Subclasses should
            return ``False`` (or raise) when they detect an unusable
            configuration.
        """
        return True

    def get_metrics(self) -> dict[str, Any]:
        """Snapshot the metrics this stage has recorded so far.

        Returns:
            dict[str, Any]: Copy of the stage's ``key -> value`` metric
            map. Mutating the returned dict does not affect future
            recordings.
        """
        return self._metrics.copy()

    def _update_metric(self, key: str, value: Any):
        """Record or overwrite a metric value on this stage.

        Args:
            key: Metric name; later returned via :meth:`get_metrics`.
            value: Arbitrary metric payload (counter, latency, summary
                dict, etc.).
        """
        self._metrics[key] = value

    def _get_dataset_config(
        self,
        dataset_id: str,
        context: "PipelineContext",
    ) -> dict:
        """Resolve the per-dataset config dict for ``dataset_id`` from the context.

        Iterates the ``datasets`` list on the pipeline configuration and
        matches by either explicit ``name`` or the auto-assigned
        ``dataset_{i}`` fallback used by :class:`PipelineConfig`.

        Args:
            dataset_id: Identifier to look up — either the user-supplied
                name or the implicit ``dataset_{i}`` form.
            context: Pipeline context whose ``config`` mapping carries
                the dataset declarations.

        Returns:
            dict: Matching dataset configuration dict, or an empty dict
            when no entry matches (callers can treat empty as "use
            defaults").
        """
        datasets = context.config.get("datasets", [])
        for i, ds_cfg in enumerate(datasets):
            ds_name = ds_cfg.get("name", f"dataset_{i}")
            if ds_name == dataset_id:
                return ds_cfg
        return {}


@dataclass
class PipelineContext:
    """Mutable cross-stage state container passed through every pipeline stage.

    A single :class:`PipelineContext` is created at the start of a
    pipeline run and threaded through each :class:`PipelineStage`'s
    :meth:`process` call. It owns four kinds of shared state: the
    immutable :class:`PipelineConfig`, monotonically increasing step/epoch
    counters that stages can read for scheduling decisions, a per-stage
    metrics map, and lazily-built shared resources (tokenizers, cache
    manager) that would be wasteful to instantiate per stage.

    Attributes:
        config (PipelineConfig): The pipeline configuration this run is
            executing against. Stages should treat it as read-only.
        seed (int | None): Master RNG seed propagated from
            :attr:`PipelineConfig.seed`; consulted by stages that need
            reproducible behaviour.
        step (int): Pipeline step counter, advanced by stages or the
            outer training loop via :meth:`update_step`. Used by
            curriculum / weight-schedule logic.
        epoch (int): Epoch counter, advanced via :meth:`update_epoch`.
        metrics (dict[str, dict[str, Any]]): Nested ``{stage_name:
            {metric_key: value}}`` map populated by
            :meth:`record_metric`.
        _tokenizers (dict[str, Any]): Cache of already-loaded tokenizers
            keyed by ``name_or_path``; populated lazily by
            :meth:`get_tokenizer`.
        _cache_manager (Any): Lazily constructed
            :class:`~easydel.data.execution.cache.TreeCacheManager`
            returned by :attr:`cache_manager`.
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
        """Return a shared tokenizer, loading and caching it on first request.

        Subsequent calls with the same ``name_or_path`` return the same
        instance, avoiding duplicate downloads/initialisation across
        stages. Tokenizers are loaded with ``trust_remote_code=True`` so
        the context works with novel architectures shipped via the Hub.

        Args:
            name_or_path: Identifier accepted by
                ``AutoTokenizer.from_pretrained`` — Hub repo id or local
                path.

        Returns:
            Any: A ``transformers.PreTrainedTokenizerBase`` (or
            compatible) instance. Typed as ``Any`` to keep the
            ``transformers`` import out of the module's import graph.
        """
        if name_or_path not in self._tokenizers:
            from transformers import AutoTokenizer

            self._tokenizers[name_or_path] = AutoTokenizer.from_pretrained(
                name_or_path,
                trust_remote_code=True,
            )
        return self._tokenizers[name_or_path]

    def update_step(self, step: int):
        """Advance the pipeline-level step counter.

        Used by stages with step-dependent behaviour (e.g. weight
        schedules in :class:`MixStage`).

        Args:
            step: New absolute step value to record. The context does
                not enforce monotonicity; callers must do so.
        """
        self.step = step

    def update_epoch(self, epoch: int):
        """Advance the pipeline-level epoch counter.

        Args:
            epoch: New absolute epoch value to record.
        """
        self.epoch = epoch

    def record_metric(self, stage: str, key: str, value: Any):
        """Store a single metric value under the given stage namespace.

        Existing values for the same ``(stage, key)`` pair are
        overwritten. The first time a stage records a metric the
        per-stage subdict is created on demand.

        Args:
            stage: Stage name acting as the metric namespace; matches
                :attr:`PipelineStage.name`.
            key: Metric identifier within the stage namespace.
            value: Arbitrary metric payload (counter, dict, summary).
        """
        if stage not in self.metrics:
            self.metrics[stage] = {}
        self.metrics[stage][key] = value

    def get_metrics(self) -> dict[str, dict[str, Any]]:
        """Return a shallow copy of every recorded metric, grouped by stage.

        Returns:
            dict[str, dict[str, Any]]: Top-level keys are stage names,
            inner keys are metric names. The outer dict is copied; the
            inner dicts are still aliased to the live recordings, so
            callers should treat the result as read-only.
        """
        return self.metrics.copy()

    @property
    def cache_manager(self) -> Any:
        """Lazily-built shared :class:`TreeCacheManager` for the pipeline run.

        Constructs the manager on first access using ``cache.cache_dir``
        from the active configuration (defaulting to
        ``".cache/easydel_pipeline"``). Subsequent accesses return the
        same instance so different stages can share its on-disk
        invalidation namespace.

        Returns:
            Any: The :class:`~easydel.data.execution.cache.TreeCacheManager`
            instance for this context.
        """
        if self._cache_manager is None:
            from ..execution.cache import TreeCacheManager

            cache_config = self.config.get("cache", {})
            cache_dir = cache_config.get("cache_dir", ".cache/easydel_pipeline")
            self._cache_manager = TreeCacheManager(cache_dir=cache_dir)
        return self._cache_manager


@dataclass
class ResumeState:
    """Serializable checkpoint of pipeline iteration position.

    Captures the minimum information needed to restart a sharded
    pipeline run without reprocessing already-consumed examples:
    coarse-grained shard and row indices for the global stream, and a
    per-dataset state map for sources (such as the mixer) that maintain
    independent positions for each input. The struct is JSON-friendly
    via :meth:`to_dict` / :meth:`from_dict` so it can be persisted next
    to model checkpoints.

    Attributes:
        shard_index (int): Position within the active source's
            ``shard_names`` list at which iteration should resume; the
            target shard is opened at ``row_index`` rather than from the
            top.
        row_index (int): Number of leading rows to skip in the shard
            identified by :attr:`shard_index`. Used directly with
            :meth:`ShardedDataSource.open_shard_at_row`.
        epoch (int): Epoch counter at the time of the checkpoint;
            mirrored back to :attr:`PipelineContext.epoch` on resume.
        step (int): Step counter at the time of the checkpoint;
            mirrored back to :attr:`PipelineContext.step` on resume.
        dataset_states (dict[str, dict]): Opaque per-dataset state used
            by composite sources (e.g.
            :class:`~easydel.data.transforms.mixture.MixedShardedSource`)
            to persist additional metadata such as RNG state or
            per-source progress; keyed by dataset ``name``.
    """

    shard_index: int = 0
    row_index: int = 0
    epoch: int = 0
    step: int = 0
    dataset_states: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Render the resume state into a plain dict suitable for JSON serialisation.

        The returned dict contains only built-in Python types so it can
        be persisted with :func:`json.dump` alongside other checkpoint
        artefacts.

        Returns:
            dict: Mapping with ``shard_index``, ``row_index``, ``epoch``,
            ``step``, and ``dataset_states`` keys mirroring the dataclass
            fields.
        """
        return {
            "shard_index": self.shard_index,
            "row_index": self.row_index,
            "epoch": self.epoch,
            "step": self.step,
            "dataset_states": self.dataset_states,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ResumeState":
        """Build a :class:`ResumeState` from the dict produced by :meth:`to_dict`.

        Missing keys default to zero / empty so partial checkpoints from
        older pipeline versions remain readable.

        Args:
            data: Dictionary previously produced by :meth:`to_dict` (or
                a subset thereof).

        Returns:
            ResumeState: Populated instance with any unspecified fields
            defaulted.
        """
        return cls(
            shard_index=data.get("shard_index", 0),
            row_index=data.get("row_index", 0),
            epoch=data.get("epoch", 0),
            step=data.get("step", 0),
            dataset_states=data.get("dataset_states", {}),
        )
