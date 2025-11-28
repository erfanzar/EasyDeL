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

"""Transformed sharded source wrapper.

This module provides:
- TransformedShardedSource: ShardedDataSource wrapper that applies transforms during iteration
"""

from __future__ import annotations

import typing as tp
from collections.abc import Iterator, Sequence

from ..core.protocols import ShardedDataSource
from .base import Transform


class TransformedShardedSource(ShardedDataSource[dict]):
    """ShardedDataSource wrapper that applies transforms during iteration.

    Supports lazy evaluation - transforms are applied as examples are yielded.
    Handles filter transforms by skipping filtered examples.

    Example:
        >>> source = JsonShardedSource("data.jsonl")
        >>> transform = RenameFields({"old": "new"}) >> FilterNonEmpty(["new"])
        >>> transformed = TransformedShardedSource(source, transform)
        >>> for example in transformed.iter_shards():
        ...     process(example)
    """

    def __init__(self, source: ShardedDataSource[dict], transform: Transform):
        """Initialize TransformedShardedSource.

        Args:
            source: Underlying data source.
            transform: Transform (or chain of transforms) to apply.
        """
        self._source = source
        self._transform = transform

    @property
    def shard_names(self) -> Sequence[str]:
        """Return shard names from underlying source."""
        return self._source.shard_names

    def num_shards(self) -> int:
        """Return number of shards from underlying source."""
        return self._source.num_shards()

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        """Open a shard and apply transforms during iteration.

        Args:
            shard_name: Name of the shard to open.

        Yields:
            Transformed examples (filtered examples are skipped).
        """
        for example in self._source.open_shard(shard_name):
            result = self._transform(example)
            if result is not None:  # Handle filter transforms
                yield result

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        """Open a shard at a specific row and apply transforms.

        Note: Row counting is based on the underlying source, not the
        transformed output. Filtered examples may affect row alignment.

        Args:
            shard_name: Name of the shard to open.
            row: Row number to start from.

        Yields:
            Transformed examples (filtered examples are skipped).
        """
        for example in self._source.open_shard_at_row(shard_name, row):
            result = self._transform(example)
            if result is not None:
                yield result

    def get_shard_info(self, shard_name: str) -> tp.Any:
        """Get shard info from underlying source."""
        return self._source.get_shard_info(shard_name)

    def __repr__(self) -> str:
        return f"TransformedShardedSource({self._source!r}, {self._transform!r})"
