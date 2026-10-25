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

"""Row-level random access over parquet corpora, without a format conversion step.

:class:`ParquetRandomAccessSource` implements grain's ``RandomAccessDataSource`` protocol
(``__len__`` + ``__getitem__``) over a set of parquet files. A cumulative ``(file, row_group)``
index is built once from footer metadata (parallel scan, optionally served from the row-group
index sidecar under ``metadata_cache_dir``), and ``__getitem__`` bisects a global row index to its
row group, decodes that group as an Arrow table and returns one row dict. Resume to any global row
is therefore O(log groups) with no data read for skipped rows, and feeding the source to
``grain.MapDataset.source(...).shuffle(seed)`` yields a true row-level global shuffle.

Cost model: a random row costs a full row-group decode on a cache miss. A small per-process LRU of
decoded groups (``cache_groups``, default 4) keeps sequential-ish access from re-decoding the same
group per row; under a global shuffle the hit rate is near zero by construction and the cache is
just bounded memory. The cache is per process and per source, so peak memory is roughly
``sources x worker processes x cache_groups x decoded group size``: with many sources, several
grain workers and ~1 GiB groups a cache of 4 can exhaust host memory, which is why the bound is a
constructor argument. Write corpora with bounded row groups (see
:mod:`easydel.data.execution.save`) to keep both cost terms small.

An oversized-row-group preflight runs while indexing: every file whose largest group exceeds the
decode guard is named up front, before any accelerator time is spent, instead of surfacing as a
``MemoryError`` inside a data worker later. The per-read guard remains as a backstop.
"""

from __future__ import annotations

import bisect
import threading
import typing as tp
from collections import OrderedDict

from .base import ParquetShardedSource
from .rowgroup_shuffle import (
    _decode_guard_bytes,
    _project_row,
    enumerate_parquet_row_groups,
    max_row_group_bytes,
    read_parquet_row_group_table,
)

if tp.TYPE_CHECKING:
    import os

__all__ = ("DEFAULT_CACHE_GROUPS", "ParquetRandomAccessSource")

DEFAULT_CACHE_GROUPS: tp.Final[int] = 4


class ParquetRandomAccessSource(ParquetShardedSource):
    """Grain ``RandomAccessDataSource`` over parquet files (footer-count index + bisect).

    Picklable for grain worker processes: the decoded-group cache, its lock and the open file
    handles are dropped on pickle (each worker re-caches locally); the row index, a cheap list of
    ``(path, row_group, rows)``, travels with the pickle when already built so workers skip the
    footer scan. When not yet built, each worker builds it lazily on first access.

    It remains a :class:`ParquetShardedSource`, so the streaming ``open_shard`` /
    ``open_shard_at_row`` surface keeps working per file.

    Example:
        >>> src = ParquetRandomAccessSource("gs://bucket/data/*.parquet", cache_groups=2)
        >>> len(src), src[12345]["input_ids"][:4]
    """

    _HANDLES_PER_THREAD = 8

    def __init__(
        self,
        data_files: str | os.PathLike | list[str | os.PathLike],
        storage_options: dict | None = None,
        columns: list[str] | None = None,
        *,
        cache_groups: int = DEFAULT_CACHE_GROUPS,
        max_row_group_bytes: int | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        metadata_cache_dir: str | None = None,
        process_index: int | None = None,
        footer_scan_workers: int | None = None,
    ):
        """Capture settings; the corpus index is built lazily on first ``len``/``[]``.

        Args:
            data_files: Glob, path list or directory (see :func:`expand_data_files`).
            storage_options: ``fsspec`` storage options.
            columns: Optional column projection.
            cache_groups: Decoded row groups cached per process (``>= 1``; the current group is
                always held).
            max_row_group_bytes: Whole-group decode guard used by both the indexing preflight and
                every read; ``None`` uses 2 GiB, ``0`` disables both.
            max_retries: Transient-I/O retries per footer/group read (``None``: 2).
            retry_delay: Linear backoff base in seconds between retries (``None``: 1.5).
            metadata_cache_dir: Root for the row-group index / byte-size / shard-info sidecars
                (``None``: off).
            process_index: Process index; only ``0``/``None`` writes remote sidecars.
            footer_scan_workers: Threads for the cold footer scan (``None``: 32).

        Raises:
            ValueError: When ``cache_groups < 1``.
        """
        super().__init__(
            data_files,
            storage_options=storage_options,
            columns=columns,
            max_row_group_bytes=max_row_group_bytes,
            max_retries=max_retries,
            retry_delay=retry_delay,
            metadata_cache_dir=metadata_cache_dir,
            process_index=process_index,
        )
        cache_groups = int(cache_groups)
        if cache_groups < 1:
            raise ValueError(f"cache_groups must be >= 1 (got {cache_groups}); the current group is always held.")
        self._cache_groups = cache_groups
        self._footer_scan_workers = footer_scan_workers
        self._group_index: list[tuple[str, int, int]] | None = None  # (path, row_group, rows), FILE order
        self._group_starts: list[int] | None = None  # cumulative row offsets aligned with _group_index
        self._total_rows: int | None = None
        self._group_cache: OrderedDict[tuple[str, int], tuple[tp.Any, bool]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._handles = threading.local()  # per-thread {path: (file_handle, ParquetFile)} LRU

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_group_cache"] = OrderedDict()  # decoded tables are heavy and per-process by design
        state["_cache_lock"] = None
        state["_handles"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._cache_lock = threading.Lock()
        self._handles = threading.local()

    def _cached_parquet_file(self, path: str):
        """Per-thread LRU of open ``ParquetFile`` handles.

        Footer parsing dominates small-row-group random access, so handles are reused across reads.
        Thread-local because a pyarrow handle's stream position is not safe under concurrent reads;
        grain read threads are a small fixed pool, so the handle count stays bounded.
        """
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs]

        cache = getattr(self._handles, "lru", None)
        if cache is None:
            cache = self._handles.lru = OrderedDict()
        entry = cache.get(path)
        if entry is None:
            handle = self._open_file(path).open()  # keep the entered binary handle for the LRU lifetime
            entry = (handle, pq.ParquetFile(handle))
            cache[path] = entry
            while len(cache) > self._HANDLES_PER_THREAD:
                old_handle, _ = cache.popitem(last=False)[1]
                try:
                    old_handle.close()
                except Exception:
                    pass
        else:
            cache.move_to_end(path)
        return entry[1]

    def _drop_cached_parquet_file(self, path: str) -> None:
        cache = getattr(self._handles, "lru", None)
        if cache is not None and path in cache:
            handle, _ = cache.pop(path)
            try:
                handle.close()
            except Exception:
                pass

    def _preflight_group_sizes(self, groups: list[tuple[str, int, int]]) -> None:
        """Name every file whose largest row group exceeds the decode guard, before any read."""
        cap = _decode_guard_bytes(self)
        if cap <= 0:
            return
        paths = sorted({g[0] for g in groups})
        biggest = max_row_group_bytes(self, paths)
        offenders = [f"{p} largest rg {biggest[p] / 2**30:.2f} GiB" for p in paths if biggest.get(p, 0) > cap]
        if offenders:
            raise ValueError(
                f"{type(self).__name__}: {len(offenders)} file(s) have row groups exceeding the decode guard "
                f"({cap / 2**30:.2f} GiB) and would fail mid-run: "
                + "; ".join(offenders[:8])
                + (" ..." if len(offenders) > 8 else "")
                + ". Exclude these files, rewrite them with a smaller row_group_size, or raise "
                "max_row_group_bytes deliberately."
            )

    def _index(self) -> tuple[list[tuple[str, int, int]], list[int], int]:
        if self._group_index is None:
            groups = enumerate_parquet_row_groups(self)
            starts: list[int] = []
            total = 0
            for _path, _rg, rows in groups:
                starts.append(total)
                total += rows
            if total == 0:
                raise ValueError(f"{type(self).__name__}: corpus {self._files!r} contains 0 rows.")
            self._preflight_group_sizes(groups)
            self._group_index = groups
            self._group_starts = starts
            self._total_rows = total
        return self._group_index, self._group_starts, self._total_rows  # type: ignore[return-value]

    @property
    def delivers_all_metadata_rows(self) -> bool:
        """Always ``True``: oversized groups raise instead of being skipped."""
        return True

    def locate(self, idx: int) -> tuple[str, int, int]:
        """Map a global row index to ``(path, row_group, row_within_group)`` without reading data.

        Args:
            idx: Global row index (negative indices count from the end).

        Returns:
            tuple[str, int, int]: File, row group and offset inside that group.

        Raises:
            IndexError: When ``idx`` is out of range.
        """
        groups, starts, total = self._index()
        idx = idx.__index__()
        if idx < 0:
            idx += total
        if not 0 <= idx < total:
            raise IndexError(f"row index {idx} out of range for {total} rows")
        k = bisect.bisect_right(starts, idx) - 1
        path, rg, _ = groups[k]
        return path, rg, idx - starts[k]

    def __len__(self) -> int:
        return self._index()[2]

    def __getitem__(self, idx: int) -> dict:
        groups, starts, total = self._index()
        idx = idx.__index__()
        if idx < 0:
            idx += total
        if not 0 <= idx < total:
            raise IndexError(f"row index {idx} out of range for {total} rows")
        k = bisect.bisect_right(starts, idx) - 1
        path, rg, n_rows = groups[k]
        key = (path, rg)
        with self._cache_lock:
            cached = self._group_cache.get(key)
            if cached is not None:
                self._group_cache.move_to_end(key)
        if cached is None:
            # Decode outside the lock: concurrent misses on one group may decode it twice
            # (idempotent) but never serialise unrelated reads. A cached handle serves the first
            # attempt; on any failure it is dropped (possibly poisoned) and retries reopen.
            try:
                cached = read_parquet_row_group_table(
                    self, path, rg, expected_rows=n_rows, parquet_file=self._cached_parquet_file(path)
                )
            except Exception:
                self._drop_cached_parquet_file(path)
                raise
            with self._cache_lock:
                self._group_cache[key] = cached
                self._group_cache.move_to_end(key)
                while len(self._group_cache) > self._cache_groups:
                    self._group_cache.popitem(last=False)
        table, needs_row_projection = cached
        # slice(i, 1).to_pylist() builds a fresh dict per call: callers may mutate it freely.
        row = table.slice(idx - starts[k], 1).to_pylist()[0]
        if needs_row_projection:
            row = _project_row(self._columns, row)
        return row

    def __repr__(self) -> str:
        return f"{type(self).__name__}(files={len(self._files)}, cache_groups={self._cache_groups})"
