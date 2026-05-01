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

"""ShardedDataSource implementations for various data formats.

This module provides Levanter-inspired ShardedDataSource implementations:
- ParquetShardedSource: Parquet files with row-group level seeking
- JsonShardedSource: JSON/JSONL files
- ArrowShardedSource: Arrow IPC files
- HuggingFaceShardedSource: HuggingFace Hub datasets
- TextShardedSource: Plain text files
- CompositeShardedSource: Combine multiple sources

Each source supports:
- URL-based shard discovery (local, GCS, S3, HTTP)
- Resumable iteration with shard/row checkpoints
- Lazy loading for memory efficiency
"""

from __future__ import annotations

import os
import typing as tp
from collections.abc import Mapping
from dataclasses import dataclass

from ejkernel.loggings import get_logger

from ..core.protocols import ShardedDataSource, ShardInfo
from ..utils import glob_files, with_retry

if tp.TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ..core.config import DatasetConfig

logger = get_logger(__name__)


def _is_pathlike(s: str) -> bool:
    """Heuristic that distinguishes filesystem/URL strings from Hub repo identifiers.

    Used by :func:`create_source` and :func:`load_for_inform` to decide
    whether to route a string to a file-format reader or to the
    HuggingFace Hub. The check is intentionally conservative — only
    strings that clearly look like paths (containing a URL scheme,
    starting with ``/``, ``./``, or ``../``) are treated as file
    locations; everything else is left for the Hub branch.

    Args:
        s: Candidate string from a user-facing config field.

    Returns:
        bool: ``True`` when ``s`` looks like a filesystem path or URI;
        ``False`` for things that probably name a Hub dataset.
    """
    return "://" in s or s.startswith("/") or s.startswith("./") or s.startswith("../")


def _fix_missing_dot(pattern: str) -> str:
    """Repair common typos in user-supplied glob patterns where the dot is missing.

    Patterns like ``"*parquet"`` and ``"*jsonl"`` are very common
    user mistakes — they either fail to expand or expand wrongly
    (``*parquet`` matches ``my_parquet_data``). This helper rewrites
    them to the canonical ``"*.parquet"``, ``"*.jsonl"`` form before
    expansion.

    Args:
        pattern: Glob pattern as written by the user.

    Returns:
        str: Corrected glob pattern. Unchanged when none of the
        recognised typos appear or when the corrected form already
        appears in the input.
    """
    fixes = {
        "*parquet": "*.parquet",
        "*jsonl": "*.jsonl",
        "*json": "*.json",
        "*csv": "*.csv",
        "*arrow": "*.arrow",
        "*pq": "*.pq",
    }
    out = pattern
    for bad, good in fixes.items():
        if bad in out and good not in out:
            out = out.replace(bad, good)
    return out


def _detect_format(files: list[str]) -> str:
    """Pick a single dataset format string from a heterogeneous file list.

    Walks each file in turn and matches its lowercased name against a
    priority-ordered list of extensions. The first hit wins, so
    mixed-extension lists snap to the highest-priority format
    (``arrow`` > ``parquet`` > ``jsonl`` > ``json`` > ``csv`` > ``pq``
    > ``txt``). Falls back to ``"arrow"`` when no recognised
    extension is found, matching the most common in-memory caller
    behaviour.

    Args:
        files: File paths/URIs whose extensions are inspected. Only
            the trailing portion is consulted.

    Returns:
        str: One of ``"json"``, ``"parquet"``, ``"csv"``, ``"txt"``,
        or ``"arrow"`` — the same vocabulary recognised by
        :func:`create_source`.
    """
    exts_priority = [".arrow", ".parquet", ".jsonl", ".json", ".csv", ".pq", ".txt"]

    for f in files:
        f_lower = f.lower()
        for ext in exts_priority:
            if f_lower.endswith(ext):
                if ext in (".json", ".jsonl"):
                    return "json"
                elif ext in (".parquet", ".pq"):
                    return "parquet"
                elif ext == ".csv":
                    return "csv"
                elif ext == ".txt":
                    return "txt"
                else:
                    return "arrow"
    return "arrow"  # Default


def _coerce_example(example: tp.Any) -> dict[str, tp.Any]:
    """Coerce arbitrary mapping-like row representations into a plain ``dict``.

    HuggingFace ``datasets`` and various readers can yield rows in
    several shapes (plain dicts, ``Mapping`` proxies, custom row
    objects exposing ``items()``). The transform DSL and downstream
    pipeline assume plain ``dict`` rows, so this helper homogenises
    them at source-iteration boundaries.

    Args:
        example: The candidate row object to coerce.

    Returns:
        dict[str, Any]: The row as a plain dict. ``dict`` instances
        are returned without copying (caller owns mutation); other
        types are materialised into a fresh dict.

    Raises:
        TypeError: When ``example`` is not mapping-like at all
            (e.g. a tuple, scalar, or class without an ``items()``
            method).
    """
    if isinstance(example, dict):
        return example
    if isinstance(example, Mapping):
        return dict(example)
    if hasattr(example, "items"):
        try:
            return dict(example.items())
        except Exception:
            pass
    raise TypeError(f"Expected mapping-like dataset row, got {type(example).__name__}")


def expand_data_files(data_files: str | os.PathLike | list[str | os.PathLike]) -> list[str]:
    """Resolve glob patterns and bare directory paths into a sorted list of files.

    Each input element is processed by an inner ``expand_one`` helper:

    * Glob meta-characters (``*``, ``?``, ``[``) trigger an immediate
      :func:`glob_files` expansion.
    * Otherwise, every recognised dataset extension (``.arrow``,
      ``.parquet``, ``.jsonl``, ``.json``, ``.csv``, ``.pq``,
      ``.txt``) is tried — first as a literal path match, then as a
      ``"<dir>/**/*<ext>"`` recursive search. The first hit wins.

    Results across all inputs are deduplicated and sorted so the order
    is deterministic across runs (important for shard-id-based
    resumption). Common typos like ``*parquet`` are repaired via
    :func:`_fix_missing_dot` before expansion.

    Args:
        data_files: Single path/pattern, ``os.PathLike``, or list of
            either. ``hf://`` and other URI schemes are forwarded
            verbatim to the underlying glob via fsspec.

    Returns:
        list[str]: Deduplicated, sorted list of file paths/URIs.

    Raises:
        FileNotFoundError: When the expansion produces no matches at
            all — e.g. typo'd glob, empty directory, missing file.
    """
    exts_priority = [".arrow", ".parquet", ".jsonl", ".json", ".csv", ".pq", ".txt"]

    def expand_one(p: str) -> list[str]:
        p = _fix_missing_dot(p)
        if any(ch in p for ch in ("*", "?", "[")):
            return glob_files(p)
        # Try each extension in priority order
        for ext in exts_priority:
            if p.endswith(ext):
                return glob_files(p)
            patt = p.rstrip("/") + f"/**/*{ext}"
            m = glob_files(patt)
            if m:
                return m
        return []

    if isinstance(data_files, (str, os.PathLike)):
        files = expand_one(os.fspath(data_files))
    else:
        files = []
        for p in data_files:
            files.extend(expand_one(os.fspath(p)))

    if not files:
        raise FileNotFoundError(f"No files matched: {data_files}")

    return sorted(set(files))


@dataclass
class ParquetShardInfo(ShardInfo):
    """Parquet-specific :class:`ShardInfo` carrying row-group counts.

    Extends :class:`ShardInfo` with ``num_row_groups`` so
    :meth:`ParquetShardedSource.open_shard_at_row` can seek directly to
    the row group containing a target row instead of iterating from
    the start. Inherits :attr:`shard_id`, :attr:`shard_name`,
    :attr:`num_rows`, :attr:`byte_size`, :attr:`url`, and
    :attr:`checksum` semantics from the base.

    Attributes:
        num_row_groups (int): Number of Parquet row groups in the
            shard, read from the file footer; used to bisect into the
            correct row group during resumption. Defaults to ``0`` for
            constructions that do not yet have metadata.
    """

    num_row_groups: int = 0


class ParquetShardedSource(ShardedDataSource[dict]):
    """:class:`ShardedDataSource` reader for Apache Parquet files via PyArrow.

    Each Parquet file is treated as a shard. Iteration walks the file's
    row groups in order and yields per-row dicts; seeking jumps
    directly to the row group containing the target row, exploiting
    Parquet's metadata to avoid linear scans during resume. Cloud URIs
    work transparently through fsspec, with optional per-source
    storage options. Column selection is supported up-front so
    expensive columns can be skipped at read time.

    Example:
        >>> source = ParquetShardedSource("gs://bucket/data/*.parquet")
        >>> for example in source.open_shard(source.shard_names[0]):
        ...     process(example)
    """

    def __init__(
        self,
        data_files: str | os.PathLike | list[str | os.PathLike],
        storage_options: dict | None = None,
        columns: list[str] | None = None,
    ):
        """Resolve the shard list and capture reader settings.

        File discovery happens immediately via :func:`expand_data_files`
        so a missing dataset is reported at construction time rather
        than on first iteration.

        Args:
            data_files: Glob pattern, list of paths, or directory passed
                to :func:`expand_data_files`.
            storage_options: ``fsspec`` storage options forwarded on
                every open (credentials, project ids, …). ``None``
                uses fsspec defaults.
            columns: Optional Parquet column projection — only the
                listed columns are read from disk. ``None`` reads all
                columns.
        """
        self._files = expand_data_files(data_files)
        self._storage_options = storage_options or {}
        self._columns = columns
        self._shard_info_cache: dict[str, ParquetShardInfo] = {}

    @property
    def shard_names(self) -> "Sequence[str]":
        """Return the parquet file paths backing this source.

        Returns:
            Sequence of expanded file paths used as shard identifiers.
        """
        return self._files

    def num_shards(self) -> int:
        """Return the number of underlying parquet files.

        Returns:
            Count of parquet files.
        """
        return len(self._files)

    def _open_file(self, path: str):
        """Open a parquet file via fsspec, honoring storage options.

        Args:
            path: Path or URL to the parquet file.

        Returns:
            Opened binary file handle (fsspec context manager).
        """
        import fsspec  # pyright: ignore[reportMissingTypeStubs]

        return fsspec.open(path, "rb", **self._storage_options)

    @with_retry(max_retries=3, initial_delay=1.0)  # pyright: ignore[reportUntypedFunctionDecorator]
    def get_shard_info(self, shard_name: str) -> ParquetShardInfo:
        """Get metadata about a Parquet shard including row group count.

        Results are cached to avoid repeated file reads.

        Args:
            shard_name: Path or URL to the Parquet file.

        Returns:
            ParquetShardInfo with file metadata.
        """
        if shard_name in self._shard_info_cache:
            return self._shard_info_cache[shard_name]

        import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs]

        with self._open_file(shard_name) as fh:
            pf = pq.ParquetFile(fh)
            info = ParquetShardInfo(
                shard_id=self._files.index(shard_name),
                shard_name=shard_name,
                num_rows=pf.metadata.num_rows,
                num_row_groups=pf.num_row_groups,
            )
            self._shard_info_cache[shard_name] = info
            return info

    @with_retry(max_retries=3, initial_delay=1.0)  # pyright: ignore[reportUntypedFunctionDecorator]
    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open a Parquet shard and iterate over rows.

        Reads row groups sequentially and yields individual rows as dictionaries.

        Args:
            shard_name: Path or URL to the Parquet file.

        Yields:
            Individual rows as dictionaries.
        """
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs]

        with self._open_file(shard_name) as fh:
            pf = pq.ParquetFile(fh)
            for rg_idx in range(pf.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=self._columns)
                cols = table.to_pydict()
                if not cols:
                    continue
                n = len(next(iter(cols.values()), []))
                for i in range(n):
                    yield {k: v[i] for k, v in cols.items()}

    def open_shard_at_row(self, shard_name: str, row: int) -> "Iterator[dict]":
        """Open a Parquet shard starting at a specific row.

        Uses row group metadata to skip directly to the relevant row group,
        avoiding sequential scanning of earlier rows.

        Args:
            shard_name: Path or URL to the Parquet file.
            row: Zero-based row index to start from.

        Yields:
            Rows starting from the specified position.
        """
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs]

        with self._open_file(shard_name) as fh:
            pf = pq.ParquetFile(fh)
            cumulative_rows = 0
            start_rg = 0
            skip_in_rg = row

            # Find the row group containing the target row
            for rg_idx in range(pf.num_row_groups):
                rg_rows = pf.metadata.row_group(rg_idx).num_rows
                if cumulative_rows + rg_rows > row:
                    start_rg = rg_idx
                    skip_in_rg = row - cumulative_rows
                    break
                cumulative_rows += rg_rows

            # Iterate from the target row group
            for rg_idx in range(start_rg, pf.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=self._columns)
                cols = table.to_pydict()
                if not cols:
                    continue
                n = len(next(iter(cols.values()), []))

                start_i = skip_in_rg if rg_idx == start_rg else 0
                for i in range(start_i, n):
                    yield {k: v[i] for k, v in cols.items()}

    def __len__(self) -> int:
        """Return total number of rows across all parquet files.

        Returns:
            Sum of ``num_rows`` reported by each shard's metadata.
        """
        total = 0
        for shard_name in self._files:
            info = self.get_shard_info(shard_name)
            total += info.num_rows
        return total

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            ``"ParquetShardedSource(files=N, columns=...)"``.
        """
        return f"ParquetShardedSource(files={len(self._files)}, columns={self._columns})"


class JsonShardedSource(ShardedDataSource[dict]):
    """:class:`ShardedDataSource` reader for JSON and JSON Lines files.

    Supports two on-disk shapes selected by the ``jsonl`` constructor
    flag: line-delimited JSON (one object per line, the default) and
    a single top-level JSON value (object or array). For arrays each
    element is yielded as a row; for objects the entire object is
    yielded as a single row. Cloud URIs work via fsspec.
    """

    def __init__(
        self,
        data_files: str | os.PathLike | list[str | os.PathLike],
        storage_options: dict | None = None,
        jsonl: bool = True,
    ):
        """Resolve the shard list and capture reader settings.

        Args:
            data_files: Glob pattern, list of paths, or directory passed
                to :func:`expand_data_files`.
            storage_options: ``fsspec`` storage options forwarded on
                every open. ``None`` uses fsspec defaults.
            jsonl: When ``True`` (the default) treat files as JSON
                Lines (one JSON value per line). When ``False`` parse
                each file as a single JSON value and yield array
                elements (for arrays) or the value itself (for
                objects).
        """
        self._files = expand_data_files(data_files)
        self._storage_options = storage_options or {}
        self._jsonl = jsonl

    @property
    def shard_names(self) -> "Sequence[str]":
        """Return the JSON/JSONL file paths backing this source.

        Returns:
            Sequence of file paths used as shard identifiers.
        """
        return self._files

    def num_shards(self) -> int:
        """Return the number of underlying JSON files.

        Returns:
            Count of JSON files.
        """
        return len(self._files)

    def _open_file(self, path: str):
        """Open a text-mode file with fsspec.

        Args:
            path: Path or URL to the file.

        Returns:
            Opened text file handle (fsspec context manager).
        """
        import fsspec  # pyright: ignore[reportMissingTypeStubs]

        return fsspec.open(path, "r", **self._storage_options)

    @with_retry(max_retries=3, initial_delay=1.0)  # pyright: ignore[reportUntypedFunctionDecorator]
    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open a JSON/JSONL shard and iterate over records.

        Args:
            shard_name: Path or URL to the JSON file.

        Yields:
            Decoded JSON records as dictionaries (single records for
            JSONL, all elements when the file is a JSON array, or a
            single yielded record when it's a JSON object).
        """
        import json

        with self._open_file(shard_name) as fh:
            if self._jsonl:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            else:
                data = json.load(fh)
                if isinstance(data, list):
                    yield from data
                else:
                    yield data

    def open_shard_at_row(self, shard_name: str, row: int) -> "Iterator[dict]":
        """Open a JSON/JSONL shard starting at a specific row.

        Args:
            shard_name: Path or URL to the JSON file.
            row: Zero-based row index to skip to.

        Yields:
            Decoded JSON records starting at ``row``.
        """
        import json

        with self._open_file(shard_name) as fh:
            if self._jsonl:
                for i, line in enumerate(fh):
                    if i < row:
                        continue
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            else:
                data = json.load(fh)
                if isinstance(data, list):
                    yield from data[row:]
                elif row == 0:
                    yield data

    def __len__(self) -> int:
        """Total number of JSON records across all shards (cached after first call).

        JSON has no random-access metadata so the first invocation
        scans every file; the result is cached on ``self._cached_len``
        for subsequent calls.

        Returns:
            int: Total record count across every file.
        """
        if not hasattr(self, "_cached_len"):
            total = 0
            for shard_name in self._files:
                for _ in self.open_shard(shard_name):
                    total += 1
            self._cached_len = total
        return self._cached_len

    def __repr__(self) -> str:
        """Concise developer-facing repr summarising the source.

        Returns:
            str: ``"JsonShardedSource(files=N, format='jsonl' | 'json')"``.
        """
        fmt = "jsonl" if self._jsonl else "json"
        return f"JsonShardedSource(files={len(self._files)}, format={fmt!r})"


class ArrowShardedSource(ShardedDataSource[dict]):
    """:class:`ShardedDataSource` reader for Apache Arrow IPC (Feather) files.

    Each Arrow IPC file is one shard; iteration walks the file's
    record batches and yields one dict per row. Cloud paths work via
    fsspec. Compared to :class:`ParquetShardedSource`, Arrow IPC is
    faster to load and lossless across PyArrow versions but lacks
    the row-group seeking machinery, so this class does not implement
    :meth:`open_shard_at_row` (the base-class linear default applies).
    """

    def __init__(
        self,
        data_files: str | os.PathLike | list[str | os.PathLike],
        storage_options: dict | None = None,
    ):
        """Resolve the shard list and capture fsspec storage options.

        Args:
            data_files: Glob pattern, list of paths, or directory passed
                to :func:`expand_data_files`.
            storage_options: ``fsspec`` storage options forwarded on
                every open. ``None`` uses fsspec defaults.
        """
        self._files = expand_data_files(data_files)
        self._storage_options = storage_options or {}

    @property
    def shard_names(self) -> "Sequence[str]":
        """Return the Arrow IPC file paths backing this source.

        Returns:
            Sequence of file paths used as shard identifiers.
        """
        return self._files

    def num_shards(self) -> int:
        """Return the number of underlying Arrow files.

        Returns:
            Count of Arrow IPC files.
        """
        return len(self._files)

    @with_retry(max_retries=3, initial_delay=1.0)  # pyright: ignore[reportUntypedFunctionDecorator]
    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open an Arrow IPC shard and iterate over rows.

        Args:
            shard_name: Path or URL to the Arrow IPC file.

        Yields:
            Individual rows as dictionaries.
        """
        import fsspec  # pyright: ignore[reportMissingTypeStubs]
        import pyarrow.ipc as ipc  # pyright: ignore[reportMissingTypeStubs]

        with fsspec.open(shard_name, "rb", **self._storage_options) as fh:
            reader = ipc.open_file(fh)
            for batch_idx in range(reader.num_record_batches):
                batch = reader.get_batch(batch_idx)
                cols = batch.to_pydict()
                if not cols:
                    continue
                n = len(next(iter(cols.values()), []))
                for i in range(n):
                    yield {k: v[i] for k, v in cols.items()}

    def __len__(self) -> int:
        """Total number of rows across all shards (cached after first call).

        Linear-scan fallback because Arrow IPC reading happens through
        :meth:`open_shard`; the result is cached on ``self._cached_len``.

        Returns:
            int: Total record count across every file.
        """
        if not hasattr(self, "_cached_len"):
            total = 0
            for shard_name in self._files:
                for _ in self.open_shard(shard_name):
                    total += 1
            self._cached_len = total
        return self._cached_len

    def __repr__(self) -> str:
        """Concise developer-facing repr summarising the source.

        Returns:
            str: ``"ArrowShardedSource(files=N)"``.
        """
        return f"ArrowShardedSource(files={len(self._files)})"


class CsvShardedSource(ShardedDataSource[dict]):
    """:class:`ShardedDataSource` reader for CSV files using :class:`csv.DictReader`.

    Each CSV file is one shard. Rows are yielded as dicts keyed by
    the header columns. Custom delimiters are supported via the
    ``delimiter`` constructor argument so the same class handles
    TSV (``"\\t"``) data. Cloud paths work via fsspec; large files
    are read line-by-line so memory usage is bounded.
    """

    def __init__(
        self,
        data_files: str | os.PathLike | list[str | os.PathLike],
        storage_options: dict | None = None,
        delimiter: str = ",",
    ):
        """Resolve the shard list and capture reader settings.

        Args:
            data_files: Glob pattern, list of paths, or directory passed
                to :func:`expand_data_files`.
            storage_options: ``fsspec`` storage options forwarded on
                every open.
            delimiter: Field separator passed to
                :class:`csv.DictReader`. Defaults to ``","``; pass
                ``"\\t"`` for TSV.
        """
        self._files = expand_data_files(data_files)
        self._storage_options = storage_options or {}
        self._delimiter = delimiter

    @property
    def shard_names(self) -> "Sequence[str]":
        """Return the CSV file paths backing this source.

        Returns:
            Sequence of file paths used as shard identifiers.
        """
        return self._files

    def num_shards(self) -> int:
        """Return the number of underlying CSV files.

        Returns:
            Count of CSV files.
        """
        return len(self._files)

    @with_retry(max_retries=3, initial_delay=1.0)  # pyright: ignore[reportUntypedFunctionDecorator]
    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open a CSV shard and iterate over rows.

        Args:
            shard_name: Path or URL to the CSV file.

        Yields:
            Each row as a ``dict`` keyed by header column.
        """
        import csv

        import fsspec  # pyright: ignore[reportMissingTypeStubs]

        with fsspec.open(shard_name, "r", **self._storage_options) as fh:
            reader = csv.DictReader(fh, delimiter=self._delimiter)
            yield from reader

    def __len__(self) -> int:
        """Total number of records across all CSV shards (cached after first call).

        First call counts every record line by line; subsequent calls
        return the cached value.

        Returns:
            int: Total record count across every file.
        """
        if not hasattr(self, "_cached_len"):
            total = 0
            for shard_name in self._files:
                for _ in self.open_shard(shard_name):
                    total += 1
            self._cached_len = total
        return self._cached_len

    def __repr__(self) -> str:
        """Concise developer-facing repr summarising the source.

        Returns:
            str: ``"CsvShardedSource(files=N, delimiter=...)"``.
        """
        return f"CsvShardedSource(files={len(self._files)}, delimiter={self._delimiter!r})"


class TextShardedSource(ShardedDataSource[dict]):
    """:class:`ShardedDataSource` reader for plain text files (one row per line).

    Each line is yielded as a dict with a single key (configurable
    via ``text_field``, default ``"text"``) holding the line's
    contents with trailing newline characters stripped. Useful for
    quickly wrapping raw corpora as a sharded source for tokenization.
    """

    def __init__(
        self,
        data_files: str | os.PathLike | list[str | os.PathLike],
        storage_options: dict | None = None,
        text_field: str = "text",
    ):
        """Resolve the shard list and capture reader settings.

        Args:
            data_files: Glob pattern, list of paths, or directory passed
                to :func:`expand_data_files`.
            storage_options: ``fsspec`` storage options forwarded on
                every open.
            text_field: Row key under which each line is stored.
                Defaults to ``"text"`` so downstream tokenize stages
                pick it up without configuration.
        """
        self._files = expand_data_files(data_files)
        self._storage_options = storage_options or {}
        self._text_field = text_field

    @property
    def shard_names(self) -> "Sequence[str]":
        """Return the text file paths backing this source.

        Returns:
            Sequence of file paths used as shard identifiers.
        """
        return self._files

    def num_shards(self) -> int:
        """Return the number of underlying text files.

        Returns:
            Count of text files.
        """
        return len(self._files)

    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open a text shard and iterate over lines.

        Args:
            shard_name: Path or URL to the text file.

        Yields:
            Dictionaries of the form ``{text_field: line}`` with trailing
            newlines stripped.
        """
        import fsspec  # pyright: ignore[reportMissingTypeStubs]

        with fsspec.open(shard_name, "r", **self._storage_options) as fh:
            for line in fh:
                yield {self._text_field: line.rstrip("\n\r")}

    def __len__(self) -> int:
        """Total number of lines across all text shards (cached after first call).

        First call walks every file line by line; subsequent calls
        return the cached count.

        Returns:
            int: Total line count.
        """
        if not hasattr(self, "_cached_len"):
            total = 0
            for shard_name in self._files:
                for _ in self.open_shard(shard_name):
                    total += 1
            self._cached_len = total
        return self._cached_len

    def __repr__(self) -> str:
        """Concise developer-facing repr summarising the source.

        Returns:
            str: ``"TextShardedSource(files=N, text_field=...)"``.
        """
        return f"TextShardedSource(files={len(self._files)}, text_field={self._text_field!r})"


class HuggingFaceShardedSource(ShardedDataSource[dict]):
    """:class:`ShardedDataSource` adapter around ``datasets.load_dataset``.

    Wraps a single HuggingFace dataset split into a sharded source
    with one synthetic shard. The underlying ``load_dataset`` call is
    deferred until the first iteration to avoid expensive downloads
    during pipeline construction. Supports both streaming and
    in-memory modes; the streaming path lets the source iterate
    arbitrarily large datasets without materialising them.

    For more granular sharding (one shard per parquet file, etc.) use
    :class:`HFDatasetShardedSource` from :mod:`hf_wrapper` instead.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        subset: str | None = None,
        streaming: bool = True,
        cache_dir: str | None = None,
    ):
        """Capture the load_dataset arguments without performing any I/O.

        Args:
            dataset_name: Hub repo id or local path forwarded as the
                ``path`` positional argument to
                ``datasets.load_dataset``.
            split: Split label (``"train"``, ``"validation"``, …)
                forwarded as ``split=`` to ``load_dataset``.
            subset: Configuration/subset name (the ``name=`` argument
                to ``load_dataset``); ``None`` selects the default
                config.
            streaming: When ``True`` (default), uses the streaming
                loader; when ``False`` materialises the split into
                RAM/cache.
            cache_dir: Local cache directory forwarded to
                ``load_dataset``; ``None`` uses the HF default.
        """
        self._dataset_name = dataset_name
        self._split = split
        self._subset = subset
        self._streaming = streaming
        self._cache_dir = cache_dir
        self._dataset = None

    @property
    def shard_names(self) -> "Sequence[str]":
        """Return a single synthetic shard name for the HF split.

        Returns:
            One-element list of ``"{dataset_name}:{split}"``.
        """
        # HuggingFace datasets are treated as a single shard
        return [f"{self._dataset_name}:{self._split}"]

    def num_shards(self) -> int:
        """Return the constant shard count of one.

        Returns:
            Always ``1``.
        """
        return 1

    def _load_dataset(self):
        """Lazily load the underlying ``datasets`` object on first access.

        Returns:
            The loaded HuggingFace ``Dataset`` or ``IterableDataset``.
        """
        if self._dataset is None:
            from datasets import load_dataset  # pyright: ignore[reportMissingTypeStubs]

            self._dataset = load_dataset(
                self._dataset_name,
                name=self._subset,
                split=self._split,
                streaming=self._streaming,
                cache_dir=self._cache_dir,
            )
        return self._dataset

    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open the HuggingFace dataset shard.

        Args:
            shard_name: Ignored; only one synthetic shard is exposed.

        Yields:
            Examples coerced to plain dicts.
        """
        ds = self._load_dataset()
        for example in ds:
            yield _coerce_example(example)

    def open_shard_at_row(self, shard_name: str, row: int) -> "Iterator[dict]":
        """Open the HuggingFace dataset starting at a specific row.

        Args:
            shard_name: Ignored; only one synthetic shard is exposed.
            row: Number of leading rows to skip.

        Yields:
            Examples coerced to plain dicts, starting at ``row``.
        """
        ds = self._load_dataset()
        if self._streaming:
            it = iter(ds)
            for _ in range(row):
                next(it, None)
            for example in it:
                yield _coerce_example(example)
        else:
            # Non-streaming: use select
            for example in ds.select(range(row, len(ds))):
                yield _coerce_example(example)

    def __len__(self) -> int:
        """Length of the underlying dataset (only valid in non-streaming mode).

        Returns:
            int: ``len(dataset)`` from the materialised
            ``datasets.Dataset``.

        Raises:
            TypeError: When ``streaming=True`` was passed at
                construction — streaming HF datasets do not support
                ``len()``.
        """
        if self._streaming:
            raise TypeError("Streaming HuggingFace datasets don't support len()")
        ds = self._load_dataset()
        return len(ds)

    def __repr__(self) -> str:
        """Concise developer-facing repr summarising the source.

        Returns:
            str: ``"HuggingFaceShardedSource(name, split=..., subset=...)"``.
        """
        subset_str = f", subset={self._subset!r}" if self._subset else ""
        return f"HuggingFaceShardedSource({self._dataset_name!r}, split={self._split!r}{subset_str})"


class CompositeShardedSource(ShardedDataSource[dict]):
    """Concatenation source that exposes several wrapped sources as a single shard list.

    Useful for combining datasets stored in different formats (e.g. a
    Parquet corpus plus a JSONL corpus) without writing a custom
    reader. Every wrapped source contributes its shards in the order
    they appear; an internal ``shard_map`` records which underlying
    source owns each merged shard so :meth:`open_shard` and
    :meth:`open_shard_at_row` can dispatch correctly.

    Note that :class:`CompositeShardedSource` is plain concatenation —
    for weighted/round-robin mixing use
    :class:`~easydel.data.transforms.mixture.MixedShardedSource`.
    """

    def __init__(self, sources: list[ShardedDataSource]):
        """Build the merged shard list from the wrapped sources.

        Iterates each input source's :attr:`ShardedDataSource.shard_names`
        in declaration order and records ``(source_index, local_shard_index)``
        in :attr:`_shard_map` for later dispatch.

        Args:
            sources: List of pre-constructed
                :class:`ShardedDataSource` instances to concatenate.
        """
        self._sources = sources
        self._shard_map: list[tuple[int, int]] = []  # (source_idx, shard_idx)
        self._shard_names: list[str] = []

        for src_idx, source in enumerate(sources):
            for shard_idx, shard_name in enumerate(source.shard_names):
                self._shard_map.append((src_idx, shard_idx))
                self._shard_names.append(shard_name)

    @property
    def shard_names(self) -> "Sequence[str]":
        """Return the flattened shard names across all wrapped sources.

        Returns:
            Sequence of shard identifiers in the order they were registered.
        """
        return self._shard_names

    def num_shards(self) -> int:
        """Return the total number of shards across all wrapped sources.

        Returns:
            Sum of shard counts.
        """
        return len(self._shard_names)

    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Dispatch to the wrapped source that owns ``shard_name``.

        Args:
            shard_name: Shard identifier.

        Returns:
            Iterator of examples produced by the underlying source.
        """
        idx = self._shard_names.index(shard_name)
        src_idx, _ = self._shard_map[idx]
        return self._sources[src_idx].open_shard(shard_name)

    def open_shard_at_row(self, shard_name: str, row: int) -> "Iterator[dict]":
        """Dispatch to the wrapped source's row-seek implementation.

        Args:
            shard_name: Shard identifier.
            row: Row index to skip to.

        Returns:
            Iterator of examples starting at ``row``.
        """
        idx = self._shard_names.index(shard_name)
        src_idx, _ = self._shard_map[idx]
        return self._sources[src_idx].open_shard_at_row(shard_name, row)

    def __len__(self) -> int:
        """Return total number of examples across all wrapped sources.

        Returns:
            Sum of ``len(source)`` for each wrapped source.
        """
        return sum(len(source) for source in self._sources)

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            ``"CompositeShardedSource(sources=N, shards=M)"``.
        """
        return f"CompositeShardedSource(sources={len(self._sources)}, shards={len(self._shard_names)})"


def create_source(config: "DatasetConfig") -> ShardedDataSource:
    """Construct the right :class:`ShardedDataSource` subclass for a :class:`DatasetConfig`.

    Decision tree:

    1. If ``config.type`` is ``"huggingface"`` / ``"hf"`` and
       ``data_files`` does not look pathlike, return a
       :class:`HuggingFaceShardedSource`.
    2. Otherwise, try to expand ``config.data_files`` into local files.
       If expansion fails, fall back to treating the string as a Hub
       dataset name.
    3. Pick the file-format reader from ``config.type`` when set, or
       infer it from extensions via :func:`_detect_format`.

    Args:
        config: Dataset declaration produced by the user (or
            :class:`PipelineConfig`).

    Returns:
        ShardedDataSource: One of :class:`ParquetShardedSource`,
        :class:`JsonShardedSource`, :class:`CsvShardedSource`,
        :class:`TextShardedSource`, :class:`ArrowShardedSource`, or
        :class:`HuggingFaceShardedSource`.

    Raises:
        FileNotFoundError: When ``config.data_files`` points at no
            existing files and is not a string that could be a Hub
            dataset id.
    """
    data_files = config.data_files
    source_type = config.type
    storage_options = None  # Could be added to DatasetConfig

    # Check if it's a HuggingFace dataset
    if source_type in ("huggingface", "hf"):
        if isinstance(data_files, str) and not _is_pathlike(data_files):
            return HuggingFaceShardedSource(
                dataset_name=data_files,
                split=config.split,
                subset=config.dataset_split_name,
            )

    # Expand files and detect format
    try:
        files = expand_data_files(data_files)
    except FileNotFoundError:
        # Might be a HuggingFace dataset
        if isinstance(data_files, str):
            return HuggingFaceShardedSource(
                dataset_name=data_files,
                split=config.split,
                subset=config.dataset_split_name,
            )
        raise

    # Use specified type or detect from files
    if source_type in ("json", "jsonl"):
        fmt = "json"
    elif source_type == "parquet":
        fmt = "parquet"
    elif source_type == "csv":
        fmt = "csv"
    elif source_type == "arrow":
        fmt = "arrow"
    elif source_type == "txt":
        fmt = "txt"
    else:
        fmt = _detect_format(files)

    # Create appropriate source
    if fmt == "parquet":
        return ParquetShardedSource(files, storage_options)
    elif fmt == "json":
        jsonl = any(f.endswith(".jsonl") for f in files)
        return JsonShardedSource(files, storage_options, jsonl=jsonl)
    elif fmt == "csv":
        return CsvShardedSource(files, storage_options)
    elif fmt == "txt":
        return TextShardedSource(files, storage_options, text_field=config.content_field)
    else:
        return ArrowShardedSource(files, storage_options)


def _detect_builder_and_files(data_files: str | os.PathLike | list[str | os.PathLike]) -> tuple[str, list[str]]:
    """Legacy helper that pairs format inference with file expansion.

    Backward-compatibility shim used by :func:`load_for_inform`. Runs
    :func:`expand_data_files` to materialise the list of paths and
    then :func:`_detect_format` to pick a builder name in the
    HuggingFace ``datasets`` vocabulary (``json``, ``parquet``,
    ``csv``, ``txt``, ``arrow``). New callers should prefer
    :func:`create_source` or use the two helpers directly.

    Args:
        data_files: Single path/pattern, list, or directory passed
            verbatim to :func:`expand_data_files`.

    Returns:
        tuple[str, list[str]]: ``(builder_name, expanded_files)``.
    """
    files = expand_data_files(data_files)
    fmt = _detect_format(files)
    return fmt, files


def load_for_inform(inform, mixture):
    """Materialise one constituent of a :class:`DatasetMixture` as a HuggingFace dataset.

    Used by :func:`build_dataset` (the legacy mixture API) to load
    each :class:`BaseDatasetInform` entry into either a
    :class:`datasets.Dataset` or :class:`datasets.IterableDataset`.
    Routes Hub-style identifiers through ``load_dataset(path=name)``
    and file-based informs through ``load_dataset(path=builder,
    data_files=files)``. When the builtin ``parquet`` builder fails
    on schemas containing nested ``List`` features, falls back to a
    custom streaming reader (:func:`_iter_parquet_rows`) implemented
    directly on top of fsspec/pyarrow.

    Honours :attr:`inform.num_rows` by truncating the loaded dataset
    via ``.take`` (streaming) or ``.select`` (in-memory).

    Args:
        inform: :class:`TextDatasetInform` or
            :class:`VisualDatasetInform` describing the constituent —
            its data files, format type, split, and optional row
            limit.
        mixture: Surrounding :class:`DatasetMixture` whose
            :attr:`cache_dir`, :attr:`streaming`, and :attr:`seed`
            settings are read.

    Returns:
        ``datasets.Dataset`` or ``datasets.IterableDataset``: The
        loaded dataset, possibly row-limited, ready for the mixer.
    """
    from datasets import IterableDataset, load_dataset  # pyright: ignore[reportMissingTypeStubs]

    t = str(inform.get_str_type())
    df = inform.data_files

    def _apply_num_rows_limit(dataset):
        """Inline helper: truncate a loaded dataset to ``inform.num_rows`` if set.

        Captures ``inform`` and ``mixture`` from :func:`load_for_inform`'s
        scope. Picks the right truncation API for the dataset shape:
        :meth:`Dataset.select` for in-memory, :meth:`take` for streaming
        (or anything else exposing ``take``). When no limit is
        configured, the dataset is returned untouched.

        Args:
            dataset: Either a ``datasets.Dataset`` (in-memory) or a
                ``datasets.IterableDataset`` (streaming).

        Returns:
            Dataset or IterableDataset: The dataset truncated to at
            most :attr:`inform.num_rows` rows, or the original
            dataset when no limit was set.

        Raises:
            ValueError: When :attr:`inform.num_rows` is negative.
        """
        num_rows = getattr(inform, "num_rows", None)
        if num_rows is None:
            return dataset

        limit = int(num_rows)
        if limit < 0:
            raise ValueError("num_rows must be >= 0")

        if mixture.streaming:
            if hasattr(dataset, "take"):
                return dataset.take(limit)
            logger.warning("num_rows=%d requested but dataset has no .take() method; returning unlimited.", limit)
            return dataset

        if hasattr(dataset, "select"):
            return dataset.select(range(min(limit, len(dataset))))

        if hasattr(dataset, "take"):
            return dataset.take(limit)
        return dataset

    # Create source and convert to HF dataset
    if t in {"huggingface", "hf"} and isinstance(df, str) and not _is_pathlike(df):
        dataset = load_dataset(
            path=df,
            name=inform.dataset_split_name,
            split=inform.split or "train",
            cache_dir=mixture.cache_dir,
            streaming=mixture.streaming,
            num_proc=None if mixture.streaming else 1,
        )
        return _apply_num_rows_limit(dataset)

    # File-based loading
    builder, files = _detect_builder_and_files(df)
    specified_builder = t if t in {"json", "jsonl", "csv", "parquet", "arrow"} else None
    builder = specified_builder or builder

    def _iter_parquet_rows(files: list[str]):
        """Inline fallback reader: stream parquet rows directly via fsspec + pyarrow.

        Activated when ``datasets.load_dataset(path="parquet", ...)``
        rejects the file schema (typically because it contains
        nested ``List`` features that the builtin loader's feature
        inference can't handle). Walks each file's row groups in
        turn and yields per-row dicts.

        Args:
            files: Parquet file paths or fsspec URIs.

        Yields:
            dict: One row dict per record across every file, in
            file/row-group order.
        """
        import fsspec  # pyright: ignore[reportMissingTypeStubs]
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs]

        for path in files:
            with fsspec.open(path, "rb") as fh:
                pf = pq.ParquetFile(fh)
                for rg in range(pf.num_row_groups):
                    table = pf.read_row_group(rg)
                    cols = table.to_pydict()
                    if not cols:
                        continue
                    n = len(next(iter(cols.values()), []))
                    for i in range(n):
                        yield {k: v[i] for k, v in cols.items()}

    try:
        dataset = load_dataset(
            path="json" if builder in {"json", "jsonl"} else builder,
            data_files=files,
            split=inform.split or "train",
            cache_dir=mixture.cache_dir,
            streaming=mixture.streaming,
            num_proc=None if mixture.streaming else 1,
        )
        return _apply_num_rows_limit(dataset)
    except ValueError as e:
        msg = str(e)
        if builder == "parquet" and ("Feature type 'List' not found" in msg or "from_dict" in msg):
            dataset = IterableDataset.from_generator(lambda: _iter_parquet_rows(files))
            return _apply_num_rows_limit(dataset)
        raise
