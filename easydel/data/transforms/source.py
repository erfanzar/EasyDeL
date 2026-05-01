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

"""Sharded source wrappers.

This module provides:
- TransformedShardedSource: applies transforms during iteration
- LimitedShardedSource: caps the total number of rows exposed by a source
"""

from __future__ import annotations

import itertools
import typing as tp
from collections.abc import Iterator, Sequence
from dataclasses import replace

from ..core.protocols import ShardedDataSource, ShardInfo
from .base import ExpandTransform, Transform


class TransformedShardedSource(ShardedDataSource[dict]):
    """:class:`ShardedDataSource` wrapper that applies a transform lazily on iteration.

    Pass-through for shard discovery, metadata, and resumption — only
    iteration is intercepted so the wrapped source's distributed/
    checkpointable properties survive. Distinguishes the two
    transform shapes at call time: regular :class:`Transform` instances
    have their single result forwarded (or dropped on ``None``), while
    :class:`ExpandTransform` instances have every yielded item
    forwarded individually so a 1-to-many transform integrates
    transparently.

    Built by :meth:`ShardedDataSource.transform` /
    :meth:`ShardedDataSource.filter` and friends.

    Example:
        >>> source = JsonShardedSource("data.jsonl")
        >>> transform = RenameFields({"old": "new"}) >> FilterNonEmpty(["new"])
        >>> transformed = TransformedShardedSource(source, transform)
        >>> for example in transformed.iter_shards():
        ...     process(example)
    """

    def __init__(self, source: ShardedDataSource[dict], transform: Transform | ExpandTransform):
        """Capture the underlying source and the transform to apply on iteration.

        Args:
            source: Wrapped :class:`ShardedDataSource`. All
                shard-discovery and metadata calls are forwarded to
                it unchanged.
            transform: Either a :class:`Transform` (one-in / one-out
                or filter) or an :class:`ExpandTransform` (one-in /
                many-out). The shape is detected at iteration time
                via ``isinstance``.
        """
        self._source = source
        self._transform = transform

    @property
    def shard_names(self) -> Sequence[str]:
        """Return shard names from the underlying source.

        Returns:
            Pass-through of ``self._source.shard_names``.
        """
        return self._source.shard_names

    def num_shards(self) -> int:
        """Return the number of shards from the underlying source.

        Returns:
            Pass-through of ``self._source.num_shards()``.
        """
        return self._source.num_shards()

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        """Open a shard and apply transforms during iteration.

        Args:
            shard_name: Name of the shard to open.

        Yields:
            Transformed examples (filtered examples are skipped).
        """
        for example in self._source.open_shard(shard_name):
            if isinstance(self._transform, ExpandTransform):
                # ExpandTransform: yields multiple examples
                yield from self._transform(example)
            else:
                # Regular Transform: yields single example or None
                result = self._transform(example)
                if result is not None:  # Handle filter transforms
                    yield result

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        """Open a shard at a specific row and apply transforms.

        Note: Row counting is based on the underlying source, not the
        transformed output. Filtered/expanded examples may affect row alignment.

        Args:
            shard_name: Name of the shard to open.
            row: Row number to start from.

        Yields:
            Transformed examples (filtered examples are skipped).
        """
        for example in self._source.open_shard_at_row(shard_name, row):
            if isinstance(self._transform, ExpandTransform):
                # ExpandTransform: yields multiple examples
                yield from self._transform(example)
            else:
                # Regular Transform: yields single example or None
                result = self._transform(example)
                if result is not None:
                    yield result

    def get_shard_info(self, shard_name: str) -> tp.Any:
        """Pass through ``ShardInfo`` from the underlying source.

        Args:
            shard_name: Shard identifier to look up.

        Returns:
            ``ShardInfo`` for the shard, or whatever the underlying
            source returns (potentially ``None``).
        """
        return self._source.get_shard_info(shard_name)

    def __len__(self) -> int:
        """Return length of the underlying source.

        Warning:
            May overcount when filter transforms are applied.

        Returns:
            ``len(self._source)``.
        """
        return len(self._source)

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            ``"TransformedShardedSource(<source>, <transform>)"``.
        """
        return f"TransformedShardedSource({self._source!r}, {self._transform!r})"


class LimitedShardedSource(ShardedDataSource[dict]):
    """ShardedDataSource wrapper that exposes at most ``max_rows`` examples.

    The limit is applied across shards in order. When shard metadata includes
    row counts, those are used directly. Otherwise prior shards are counted only
    when later shard access makes that necessary, and counting stops as soon as
    the global row budget is exhausted.

    Note: This class is **not** thread-safe. Shard resolution mutates internal
    state and must be driven from a single thread.
    """

    def __init__(self, source: ShardedDataSource[dict], max_rows: int):
        """Initialize a ``LimitedShardedSource`` wrapper.

        Args:
            source: Underlying data source whose iteration is bounded.
            max_rows: Maximum total number of rows to expose across all
                shards (clamped to ``>= 0``).
        """
        self._source = source
        self._max_rows = max(int(max_rows), 0)
        self._shard_names = tuple(source.shard_names)
        self._shard_name_to_index = {name: idx for idx, name in enumerate(self._shard_names)}
        self._exact_shard_sizes: dict[str, int] = {}
        self._resolved_prefix_rows: dict[str, int] = {}
        self._resolved_prefix_count = 0
        self._remaining_before_index = [self._max_rows]

    def _count_shard_rows_up_to(self, shard_name: str, limit: int) -> int:
        """Count rows in ``shard_name``, stopping early once ``limit`` is exceeded.

        Args:
            shard_name: Shard whose rows are counted.
            limit: Maximum number of rows to count exactly.

        Returns:
            Exact row count when the shard has ``<= limit`` rows,
            or ``limit + 1`` as a sentinel meaning "at least ``limit + 1``
            rows" (avoids iterating the entire shard when we only need
            to know it exceeds the budget).
        """
        if limit <= 0:
            return 0
        count = 0
        for _ in self._source.open_shard(shard_name):
            count += 1
            if count > limit:
                return count
        return count

    def _get_known_shard_size(self, shard_name: str) -> int | None:
        """Return the cached or metadata-reported row count for a shard.

        Args:
            shard_name: Shard identifier.

        Returns:
            Exact row count when known, otherwise ``None``.
        """
        if shard_name in self._exact_shard_sizes:
            return self._exact_shard_sizes[shard_name]
        info = self._source.get_shard_info(shard_name)
        if info is None or info.num_rows is None:
            return None
        size = int(info.num_rows)
        self._exact_shard_sizes[shard_name] = size
        return size

    def _resolve_prefix_until(self, shard_index: int) -> None:
        """Compute exposed row counts for shards ``[0, shard_index)``.

        Walks earlier shards in order, consulting metadata or counting
        rows when necessary, and updates ``_remaining_before_index`` so
        later lookups can be answered cheaply.

        Args:
            shard_index: Stop after resolving the prefix up to (but not
                including) this index.
        """
        target = min(max(shard_index, 0), len(self._shard_names))
        while self._resolved_prefix_count < target:
            shard_name = self._shard_names[self._resolved_prefix_count]
            remaining = self._remaining_before_index[-1]
            exposed_rows = 0
            if remaining > 0:
                shard_rows = self._get_known_shard_size(shard_name)
                if shard_rows is None:
                    counted_rows = self._count_shard_rows_up_to(shard_name, remaining)
                    if counted_rows <= remaining:
                        shard_rows = counted_rows
                        self._exact_shard_sizes[shard_name] = shard_rows
                        exposed_rows = shard_rows
                    else:
                        exposed_rows = remaining
                else:
                    exposed_rows = min(shard_rows, remaining)
            self._resolved_prefix_rows[shard_name] = exposed_rows
            self._resolved_prefix_count += 1
            self._remaining_before_index.append(max(0, remaining - exposed_rows))

    def _get_shard_limit(self, shard_name: str) -> int:
        """Compute the maximum rows exposable for a single shard.

        Args:
            shard_name: Shard identifier.

        Returns:
            Number of rows allowed from this shard, accounting for the
            global ``max_rows`` budget consumed by earlier shards.
        """
        shard_index = self._shard_name_to_index[shard_name]
        self._resolve_prefix_until(shard_index)
        remaining = self._remaining_before_index[shard_index]
        if remaining <= 0:
            return 0
        if shard_name in self._resolved_prefix_rows:
            return self._resolved_prefix_rows[shard_name]
        shard_rows = self._get_known_shard_size(shard_name)
        if shard_rows is None:
            return remaining
        return min(shard_rows, remaining)

    @property
    def shard_names(self) -> Sequence[str]:
        """Return the underlying source's shard names verbatim.

        Returns:
            Tuple of shard identifiers captured at construction time.
        """
        return self._shard_names

    def num_shards(self) -> int:
        """Return the number of shards exposed by this source.

        Returns:
            Length of the captured shard-name list.
        """
        return len(self._shard_names)

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        """Open a shard, capping iteration at this shard's row budget.

        Args:
            shard_name: Shard identifier.

        Returns:
            Iterator yielding at most ``_get_shard_limit(shard_name)``
            rows.
        """
        return itertools.islice(self._source.open_shard(shard_name), self._get_shard_limit(shard_name))

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        """Open a shard at ``row`` while respecting the row budget.

        Args:
            shard_name: Shard identifier.
            row: Row offset within the shard.

        Returns:
            Iterator yielding remaining rows from ``row`` up to the
            shard's exposed limit; an empty iterator when ``row`` is
            already past the limit.
        """
        shard_limit = self._get_shard_limit(shard_name)
        if row >= shard_limit:
            return iter(())
        return itertools.islice(self._source.open_shard_at_row(shard_name, row), shard_limit - row)

    def get_shard_info(self, shard_name: str) -> ShardInfo | None:
        """Return shard metadata adjusted for this source's row budget.

        Args:
            shard_name: Shard identifier.

        Returns:
            ``ShardInfo`` whose ``num_rows`` reflects the truncated
            count, falling back to a plain ``ShardInfo`` if the
            underlying source provides no metadata.
        """
        shard_index = self._shard_name_to_index[shard_name]
        self._resolve_prefix_until(shard_index)
        base_info = self._source.get_shard_info(shard_name)
        remaining = self._remaining_before_index[shard_index]
        if remaining <= 0:
            shard_limit: int | None = 0
        elif shard_name in self._resolved_prefix_rows:
            shard_limit = self._resolved_prefix_rows[shard_name]
        else:
            shard_rows = self._get_known_shard_size(shard_name)
            shard_limit = None if shard_rows is None else min(shard_rows, remaining)
        if base_info is None:
            return ShardInfo(
                shard_id=shard_index,
                shard_name=shard_name,
                num_rows=shard_limit,
            )
        try:
            return replace(base_info, num_rows=shard_limit)
        except TypeError:
            return ShardInfo(
                shard_id=getattr(base_info, "shard_id", shard_index),
                shard_name=getattr(base_info, "shard_name", shard_name),
                num_rows=shard_limit,
                byte_size=getattr(base_info, "byte_size", None),
                url=getattr(base_info, "url", None),
                checksum=getattr(base_info, "checksum", None),
            )

    def __len__(self) -> int:
        """Return the actual number of rows this wrapper will yield.

        Returns:
            ``min(max_rows, total source rows)``.
        """
        self._resolve_prefix_until(len(self._shard_names))
        return self._max_rows - self._remaining_before_index[-1]

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            ``"LimitedShardedSource(<source>, max_rows=N)"``.
        """
        return f"LimitedShardedSource({self._source!r}, max_rows={self._max_rows})"
