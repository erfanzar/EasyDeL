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

"""HuggingFace Dataset wrapper as ShardedDataSource.

This module provides HFDatasetShardedSource which wraps any HuggingFace
Dataset or IterableDataset as a ShardedDataSource, enabling trainers to
work with a unified data interface.
"""

from __future__ import annotations

import typing as tp
from collections.abc import Iterator, Sequence

from ..core.protocols import ShardedDataSource, ShardInfo

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset


class HFDatasetShardedSource(ShardedDataSource[dict]):
    """Wraps HuggingFace Dataset/IterableDataset as ShardedDataSource.

    This adapter allows trainers to work with a single data type internally
    while accepting both HF datasets and ShardedDataSource from users.
    HF datasets are treated as a single shard for simplicity.

    Example:
        >>> from datasets import load_dataset
        >>> hf_ds = load_dataset("tatsu-lab/alpaca", split="train")
        >>> source = HFDatasetShardedSource(hf_ds)
        >>> for example in source.iter_shards():
        ...     process(example)

    Example with streaming:
        >>> hf_ds = load_dataset("tiiuae/falcon-refinedweb", split="train", streaming=True)
        >>> source = HFDatasetShardedSource(hf_ds)
        >>> for example in source.iter_shards():
        ...     process(example)
    """

    def __init__(
        self,
        dataset: "Dataset | IterableDataset",
        name: str | None = None,
    ):
        """Initialize HFDatasetShardedSource.

        Args:
            dataset: HuggingFace Dataset or IterableDataset to wrap.
            name: Optional name for the shard. Defaults to "hf_dataset".
        """
        self._dataset = dataset
        self._name = name or "hf_dataset"
        self._is_iterable = self._check_is_iterable(dataset)
        self._length: int | None = None

        # Cache length for non-iterable datasets
        if not self._is_iterable:
            try:
                self._length = len(dataset)
            except (TypeError, AttributeError):
                pass

    @staticmethod
    def _check_is_iterable(dataset) -> bool:
        """Check if dataset is an IterableDataset."""
        try:
            from datasets import IterableDataset

            return isinstance(dataset, IterableDataset)
        except ImportError:
            # Fallback: check if it has __len__
            return not hasattr(dataset, "__len__")

    @property
    def shard_names(self) -> Sequence[str]:
        """Return shard names. HF datasets are treated as single shard."""
        return [f"{self._name}:0"]

    def num_shards(self) -> int:
        """Return number of shards. HF datasets are treated as single shard."""
        return 1

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        """Open the shard and return iterator over examples.

        Args:
            shard_name: Shard identifier (ignored since single shard).

        Yields:
            Individual examples from the dataset as dictionaries.
        """
        if self._is_iterable:
            # IterableDataset - just iterate
            yield from self._dataset
        else:
            # Regular Dataset - index access
            for i in range(len(self._dataset)):
                yield self._dataset[i]

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        """Open shard starting at a specific row.

        Args:
            shard_name: Shard identifier (ignored since single shard).
            row: Row index to start from.

        Yields:
            Examples starting from the specified row.
        """
        if self._is_iterable:
            # IterableDataset - skip rows
            for i, example in enumerate(self._dataset):
                if i >= row:
                    yield example
        else:
            # Regular Dataset - direct indexing
            for i in range(row, len(self._dataset)):
                yield self._dataset[i]

    def get_shard_info(self, shard_name: str) -> ShardInfo | None:
        """Get metadata about the shard.

        Args:
            shard_name: Shard identifier.

        Returns:
            ShardInfo with available metadata.
        """
        return ShardInfo(
            shard_id=0,
            shard_name=shard_name,
            num_rows=self._length,
        )

    @property
    def is_streaming(self) -> bool:
        """Check if this is a streaming (IterableDataset) source."""
        return self._is_iterable

    @property
    def estimated_length(self) -> int | None:
        """Return estimated number of examples, if known."""
        return self._length

    def __len__(self) -> int:
        """Return number of examples in the dataset.

        Raises:
            TypeError: If dataset is streaming (IterableDataset) and length is unknown.
        """
        if self._length is not None:
            return self._length
        raise TypeError("Streaming HuggingFace datasets don't support len()")

    def __repr__(self) -> str:
        ds_type = "IterableDataset" if self._is_iterable else "Dataset"
        length_str = f", length={self._length}" if self._length else ""
        return f"HFDatasetShardedSource({ds_type}{length_str})"


def wrap_hf_dataset(
    dataset: "Dataset | IterableDataset | ShardedDataSource",
) -> ShardedDataSource:
    """Wrap a HuggingFace dataset as ShardedDataSource if needed.

    This is a convenience function for trainers to normalize input datasets.

    Args:
        dataset: Either a HF Dataset, IterableDataset, or existing ShardedDataSource.

    Returns:
        ShardedDataSource wrapping the input.

    Raises:
        TypeError: If dataset type is not supported.

    Example:
        >>> # In trainer __init__:
        >>> self._train_source = wrap_hf_dataset(dataset_train)
    """
    if isinstance(dataset, ShardedDataSource):
        return dataset

    # Check for HF datasets
    try:
        from datasets import Dataset, IterableDataset

        if isinstance(dataset, (Dataset, IterableDataset)):
            return HFDatasetShardedSource(dataset)
    except ImportError:
        pass

    # Check if it looks like a Dataset (duck typing)
    if hasattr(dataset, "__iter__") and (hasattr(dataset, "__len__") or hasattr(dataset, "__getitem__")):
        return HFDatasetShardedSource(dataset)

    raise TypeError(
        f"Unsupported dataset type: {type(dataset)}. "
        "Expected HuggingFace Dataset, IterableDataset, or ShardedDataSource."
    )
