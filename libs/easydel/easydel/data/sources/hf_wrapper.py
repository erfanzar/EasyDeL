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

"""HuggingFace Dataset wrapper as ShardedDataSource.

This module provides HFDatasetShardedSource which wraps any HuggingFace
Dataset or IterableDataset as a ShardedDataSource, enabling trainers to
work with a unified data interface.
"""

from __future__ import annotations

import typing as tp
from collections.abc import Iterator, Mapping, Sequence

from ..core.protocols import ShardedDataSource, ShardInfo

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]


class HFDatasetShardedSource(ShardedDataSource[dict]):
    """Adapter that exposes any pre-loaded HF dataset as a single-shard sharded source.

    The trainers and pipeline stages in EasyDeL expect their inputs as
    :class:`ShardedDataSource` instances; this wrapper lets users pass
    in already-built ``datasets.Dataset`` or ``datasets.IterableDataset``
    objects and have them participate in the same machinery. The
    wrapper exposes one synthetic shard, picks the right random-access
    strategy at construction time (indexing for in-memory ``Dataset``,
    iteration for streaming ``IterableDataset``), and caches the
    length of in-memory datasets so :meth:`__len__` and
    :meth:`get_shard_info` are cheap.

    For dataset families where multiple file-level shards exist on
    the Hub, prefer :class:`HuggingFaceShardedSource` which exposes
    real shard granularity. Use this class when you already have a
    materialised dataset object.

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
        """Detect the dataset shape and (when possible) cache its length.

        Streaming detection happens up front via
        :meth:`_check_is_iterable` so :meth:`open_shard` can pick the
        right access pattern without reflecting on every call. For
        in-memory datasets, ``len(dataset)`` is queried once and
        memoised on ``self._length``; failures (some custom dataset
        objects don't implement ``__len__``) are caught silently and
        the source falls back to streaming-style iteration.

        Args:
            dataset: HuggingFace ``Dataset`` (in-memory) or
                ``IterableDataset`` (streaming) to wrap.
            name: Optional shard label embedded in the synthetic
                shard name (``"{name}:0"``); useful when several
                wrapped sources are composed and need distinguishable
                shard ids. Defaults to ``"hf_dataset"``.
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
        """Decide whether to treat ``dataset`` as a streaming or random-access source.

        Prefers an :class:`isinstance` check against
        ``datasets.IterableDataset`` when the ``datasets`` package is
        importable. Falls back to a duck-typed check (treats anything
        without ``__len__`` as streaming) when the import fails — this
        keeps the helper usable in environments that wrap HF-like
        objects without depending on the package.

        Args:
            dataset: The dataset object to classify.

        Returns:
            bool: ``True`` for streaming/iterable datasets, ``False``
            for indexable in-memory ones.
        """
        try:
            from datasets import IterableDataset  # pyright: ignore[reportMissingTypeStubs]

            return isinstance(dataset, IterableDataset)
        except ImportError:
            # Fallback: check if it has __len__
            return not hasattr(dataset, "__len__")

    @staticmethod
    def _to_example(value: tp.Any) -> dict[str, tp.Any]:
        """Coerce HF row payloads into plain Python dicts for downstream consumers.

        HuggingFace's ``Dataset.__getitem__`` and iteration may return
        rich row proxies depending on the format flag and column types.
        Rather than rely on those proxies' shape, this helper
        normalises into a plain ``dict`` so the rest of the pipeline
        can be schema-agnostic. Mirrors
        :func:`_coerce_example` in :mod:`base`.

        Args:
            value: Whatever the underlying dataset yielded.

        Returns:
            dict[str, Any]: Plain dict; original ``dict`` instances
            are returned unchanged (caller owns mutation).

        Raises:
            TypeError: When ``value`` is not mapping-like at all.
        """
        if isinstance(value, dict):
            return value
        if isinstance(value, Mapping):
            return dict(value)
        if hasattr(value, "items"):
            try:
                return dict(value.items())
            except Exception:
                pass
        raise TypeError(f"Expected mapping-like dataset row, got {type(value).__name__}")

    @property
    def shard_names(self) -> Sequence[str]:
        """Return shard names; HF datasets expose a single synthetic shard.

        Returns:
            One-element list of ``"{name}:0"``.
        """
        return [f"{self._name}:0"]

    def num_shards(self) -> int:
        """Return the constant shard count of one.

        Returns:
            Always ``1``.
        """
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
            for example in self._dataset:
                yield self._to_example(example)
        else:
            # Regular Dataset - index access
            for i in range(len(self._dataset)):
                yield self._to_example(self._dataset[i])

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
                    yield self._to_example(example)
        else:
            # Regular Dataset - direct indexing
            for i in range(row, len(self._dataset)):
                yield self._to_example(self._dataset[i])

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
        """Whether the wrapped dataset is a streaming ``IterableDataset``.

        Returns:
            True for streaming datasets, False for in-memory ``Dataset``.
        """
        return self._is_iterable

    @property
    def estimated_length(self) -> int | None:
        """Return the cached length, if available.

        Returns:
            Number of examples for in-memory datasets, otherwise ``None``.
        """
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
        """Return a developer-friendly representation.

        Returns:
            ``"HFDatasetShardedSource(Dataset|IterableDataset, length=N)"``.
        """
        ds_type = "IterableDataset" if self._is_iterable else "Dataset"
        length_str = f", length={self._length}" if self._length else ""
        return f"HFDatasetShardedSource({ds_type}{length_str})"


def wrap_hf_dataset(
    dataset: "Dataset | IterableDataset | ShardedDataSource",
) -> ShardedDataSource:
    """Coerce any supported dataset shape into a :class:`ShardedDataSource`.

    The trainers in :mod:`easydel.trainers` accept user inputs as any
    of: a real :class:`ShardedDataSource` (passed through unchanged),
    a HuggingFace ``Dataset``/``IterableDataset`` (wrapped via
    :class:`HFDatasetShardedSource`), or any duck-typed object that
    looks iterable plus indexable/sized. This helper is the single
    entry point that performs that normalisation so each trainer
    does not need to repeat the type plumbing.

    Args:
        dataset: Either a :class:`ShardedDataSource`, a HuggingFace
            dataset, or any object with ``__iter__`` plus either
            ``__len__`` or ``__getitem__``.

    Returns:
        ShardedDataSource: ``dataset`` itself when already a
        :class:`ShardedDataSource`; otherwise a fresh
        :class:`HFDatasetShardedSource` wrapping it.

    Raises:
        TypeError: When ``dataset`` is none of the supported shapes.

    Example:
        >>> # In trainer __init__:
        >>> self._train_source = wrap_hf_dataset(dataset_train)
    """
    if isinstance(dataset, ShardedDataSource):
        return dataset

    # Check for HF datasets
    try:
        from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

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
