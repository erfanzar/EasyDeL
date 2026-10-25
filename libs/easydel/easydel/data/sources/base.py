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

import itertools
import os
import typing as tp
from collections.abc import Mapping
from dataclasses import dataclass

from ejkernel.loggings import get_logger

from ..core.protocols import ShardedDataSource, ShardInfo
from ..core.types import _uncompressed_filename
from ..utils import glob_files, with_retry

if tp.TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ..core.config import DatasetConfig, SourceStageConfig

logger = get_logger(__name__)

_PYARROW_NESTED_CHUNKED_ARRAY_ERROR = "Nested data conversions not implemented for chunked array outputs"
_PARQUET_SLOW_NESTED_COLUMNS = frozenset({"image_embeds"})
_PARQUET_NESTED_FALLBACK_BATCH_SIZE = 256
_PARQUET_NESTED_FALLBACK_BATCH_SIZES = (256, 64, 16, 1)
# Per-batch byte budget for streaming embed-heavy row groups, and the projected
# row-group size above which whole-row-group reads are not even attempted
# (Arrow cannot represent >2 GiB of nested binary as a single array, so such
# reads are guaranteed to fail after wasting the full decode).
_PARQUET_NESTED_TARGET_BATCH_BYTES = 256 * 2**20
_PARQUET_NESTED_CHUNK_SAFE_BYTES = 1_500_000_000
# Row groups whose projected payload exceeds this are streamed in byte-budgeted batches instead of decoded whole.
_PARQUET_MAX_SAFE_ROW_GROUP_BYTES_DEFAULT = 24 * 2**30


def _parquet_max_safe_row_group_bytes() -> int:
    """Default decode-memory guard threshold (bytes) for sources without an explicit override.

    Per-source overrides are passed as ``ParquetShardedSource(max_row_group_bytes=...)``.
    """
    return _PARQUET_MAX_SAFE_ROW_GROUP_BYTES_DEFAULT


# Errors that reopen a shard stream; aiohttp's ClientError is not an OSError, so list it explicitly.
try:  # aiohttp is a gcsfs dependency; absent on non-GCS deployments
    from aiohttp import ClientError as _AiohttpClientError

    _PARQUET_TRANSIENT_IO_ERRORS: tuple[type[BaseException], ...] = (
        IOError,
        OSError,
        TimeoutError,
        ConnectionError,
        _AiohttpClientError,
    )
except Exception:  # pragma: no cover - environment-dependent
    _PARQUET_TRANSIENT_IO_ERRORS = (IOError, OSError, TimeoutError, ConnectionError)
_PARQUET_STREAM_MAX_REOPENS = 4


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
    exts_priority = [".arrow", ".parquet", ".jsonl", ".json", ".csv", ".tsv", ".pq", ".txt"]

    for f in files:
        f_lower = _uncompressed_filename(f)
        for ext in exts_priority:
            if f_lower.endswith(ext):
                if ext in (".json", ".jsonl"):
                    return "json"
                elif ext in (".parquet", ".pq"):
                    return "parquet"
                elif ext in (".csv", ".tsv"):
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


def expand_data_files(
    data_files: str | os.PathLike | list[str | os.PathLike], storage_options: dict | None = None
) -> list[str]:
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
    exts_priority = [".arrow", ".parquet", ".jsonl", ".json", ".csv", ".tsv", ".pq", ".txt"]
    exts_priority = [ext + suffix for ext in exts_priority for suffix in ("", ".gz", ".zst", ".bz2", ".xz")]

    def expand_glob(pattern: str) -> list[str]:
        if not storage_options:
            return glob_files(pattern)
        import fsspec

        fs, path = fsspec.core.url_to_fs(pattern, **storage_options)
        matches = fs.glob(path, recursive=True)
        protocol = fs.protocol
        if isinstance(protocol, tuple | list):
            protocol = protocol[0]
        if protocol not in ("file", "local", ""):
            return [fs.unstrip_protocol(match) for match in matches]
        return matches

    def expand_one(p: str) -> list[str]:
        """Expand a single path / pattern argument into concrete file paths.

        Inline closure of :func:`expand_data_files`. Captures
        ``exts_priority`` from the enclosing scope. Glob meta-characters
        force a direct :func:`glob_files` call; otherwise the helper
        first treats ``p`` as a file path with a dataset extension, then
        as a directory and tries ``"<dir>/**/*<ext>"`` recursive matches
        in priority order. Returns an empty list when no match is found
        — :func:`expand_data_files` then aggregates and decides whether
        to raise :class:`FileNotFoundError`.

        Args:
            p: Single path/pattern (already converted to ``str``).

        Returns:
            list[str]: Matching files, possibly empty.
        """
        p = _fix_missing_dot(p)
        if any(ch in p for ch in ("*", "?", "[")):
            return expand_glob(p)
        # Try each extension in priority order
        for ext in exts_priority:
            if p.endswith(ext):
                return expand_glob(p)
            patt = p.rstrip("/") + f"/**/*{ext}"
            m = expand_glob(patt)
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
        max_row_group_bytes: int | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        *,
        metadata_cache_dir: str | None = None,
        process_index: int | None = None,
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
            max_row_group_bytes: Decode-memory guard threshold in bytes.
                Row groups whose projected payload exceeds it are streamed
                in byte-budgeted batches with a warning (PyArrow decodes a row group atomically and
                can kill the host on oversized ones). ``0`` disables the
                guard; ``None`` uses the built-in default (24 GiB).
            max_retries: Number of transient-I/O retries for footer reads
                and mid-stream reopens. ``None`` keeps the built-in budget.
            retry_delay: Base delay in seconds for the exponential backoff
                between retries. ``None`` keeps the built-in 1 second.
            metadata_cache_dir: Optional directory or URI prefix (``gs://...``)
                for durable footer-metadata caches: the shard-info cache
                (:class:`~easydel.data.shard_info_cache.ShardInfoDiskCache`)
                that backs :meth:`get_shard_info`, and the row-group index /
                byte-size sidecars used by the row-group shuffle and
                random-access sources. Restarts then skip the footer walk.
                ``None`` (default) disables every durable cache.
            process_index: This process's index in a multi-host job; only index
                ``0`` (or ``None``, unknown) writes shared remote cache objects.
                Reads are unrestricted.
        """
        from ..shard_info_cache import cache_for_files

        self._files = expand_data_files(data_files, storage_options=storage_options)
        self._storage_options = storage_options or {}
        self._columns = columns
        self._max_row_group_bytes = None if max_row_group_bytes is None else int(max_row_group_bytes)
        self._max_retries = None if max_retries is None else max(int(max_retries), 0)
        self._retry_delay = None if retry_delay is None else max(float(retry_delay), 0.0)
        self._metadata_cache_dir = (metadata_cache_dir or "").strip() or None
        self._process_index = None if process_index is None else int(process_index)
        self._shard_info_cache: dict[str, ParquetShardInfo] = {}
        # Durable backing store for the dict above; inert unless ``metadata_cache_dir`` is set and
        # fail-open in every path, so an unconfigured source reads exactly as before.
        self._shard_info_disk = cache_for_files(
            self._files,
            self._storage_options,
            root=self._metadata_cache_dir,
            process_index=self._process_index,
        )
        self._warned_projection_fallback = False
        self._force_single_threaded_reads = False

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

    @staticmethod
    def _is_nested_projection_error(exc: Exception) -> bool:
        """Return whether *exc* is PyArrow's nested projected-read bug."""
        return type(exc).__name__ == "ArrowNotImplementedError" and _PYARROW_NESTED_CHUNKED_ARRAY_ERROR in str(exc)

    def _warn_projection_fallback_once(self) -> None:
        """Log the nested Parquet fallback warning at most once per source."""
        if self._warned_projection_fallback:
            return
        logger.warning(
            "Projected Parquet row-group reads failed for nested columns; retrying with single-threaded Parquet reads."
        )
        self._warned_projection_fallback = True

    def _guard_bytes(self) -> int:
        """Return the effective decode-memory guard threshold for this source.

        The per-source ``max_row_group_bytes`` constructor argument wins; otherwise the module default
        from :func:`_parquet_max_safe_row_group_bytes` applies.

        Returns:
            int: Threshold in bytes; ``0`` (or negative) disables the guard.
        """
        if self._max_row_group_bytes is not None:
            return self._max_row_group_bytes
        return _parquet_max_safe_row_group_bytes()

    def _call_with_retries(self, fn: tp.Callable[..., tp.Any], *args: tp.Any) -> tp.Any:
        """Call ``fn(*args)`` with this source's transient-I/O retry budget.

        Args:
            fn: Callable performing a footer/metadata read.
            *args: Positional arguments forwarded to ``fn``.

        Returns:
            Whatever ``fn`` returns.
        """
        retries = 3 if self._max_retries is None else self._max_retries
        delay = 1.0 if self._retry_delay is None else self._retry_delay
        return with_retry(max_retries=retries, initial_delay=delay)(fn)(*args)

    @staticmethod
    def _table_to_rows(table: tp.Any) -> "Iterator[dict]":
        """Yield row dictionaries from an Arrow table.

        Numeric columns (primitives and list-of-primitive) are materialized as
        numpy views sliced zero-copy from the Arrow buffers instead of Python
        object lists: ``to_pydict`` on token/embedding columns creates millions
        of Python ints/floats per batch, and that allocation churn holds the
        GIL on dataloader threads long enough to stall step dispatch
        fleet-wide (measured 13-25s put-phase stalls on v5p-2048, 2026-07-04).
        Strings, structs (e.g. ``messages``), and anything unexpected keep the
        original Python materialization.
        """
        try:
            numpy_rows = ParquetShardedSource._table_to_rows_numpy(table)
        except Exception:
            numpy_rows = None
        if numpy_rows is not None:
            # Deliberately outside the try: all fallible materialization ran at
            # call time above, so iteration can only fail for reasons (e.g.
            # MemoryError) that must PROPAGATE — falling back mid-stream would
            # re-yield from row 0 and silently duplicate rows already consumed.
            yield from numpy_rows
            return
        try:
            cols = table.to_pydict()
        except Exception as exc:
            if not ParquetShardedSource._is_nested_projection_error(exc):
                raise
            combine_chunks = getattr(table, "combine_chunks", None)
            if combine_chunks is None:
                raise
            cols = combine_chunks().to_pydict()
        if not cols:
            return
        n = len(next(iter(cols.values()), []))
        for i in range(n):
            yield {k: v[i] for k, v in cols.items()}

    @staticmethod
    def _table_to_rows_numpy(table: tp.Any) -> "Iterator[dict]":
        """Numpy-first row materialization (see ``_table_to_rows``).

        A plain function (not a generator): every fallible step — the Arrow →
        numpy/pylist column materialization — runs eagerly at call time, so
        the caller can still fall back to the legacy path before a single row
        has been yielded. The returned iterator only slices pre-materialized
        buffers.
        """
        import pyarrow as pa  # pyright: ignore[reportMissingTypeStubs]

        n = int(table.num_rows)
        if n == 0:
            return iter(())
        fast_list: dict[str, tuple[tp.Any, tp.Any]] = {}
        fast_flat: dict[str, tp.Any] = {}
        slow: dict[str, list] = {}
        for name in table.column_names:
            column = table.column(name)
            arr = column.combine_chunks() if hasattr(column, "combine_chunks") else column
            atype = arr.type
            if (
                (pa.types.is_list(atype) or pa.types.is_large_list(atype))
                and pa.types.is_primitive(atype.value_type)
                and not pa.types.is_boolean(atype.value_type)
                and arr.null_count == 0
                # arr.null_count only counts null LISTS; a null ELEMENT inside a
                # non-null list survives it, and to_numpy() on the flattened
                # child would then silently upcast the ENTIRE column chunk to
                # float64 with NaN in place of the null — every downstream
                # int cast turns that into a garbage token id with no error
                # anywhere. Element-level nulls take the pylist route instead.
                and arr.values.null_count == 0
            ):
                values = arr.values.to_numpy(zero_copy_only=False)
                offsets = arr.offsets.to_numpy(zero_copy_only=False)
                fast_list[name] = (offsets, values)
            elif pa.types.is_primitive(atype) and not pa.types.is_boolean(atype) and arr.null_count == 0:
                fast_flat[name] = arr.to_numpy(zero_copy_only=False)
            else:
                slow[name] = arr.to_pylist()
        del table

        def _rows() -> "Iterator[dict]":
            for i in range(n):
                row: dict[str, tp.Any] = {}
                for name, (offsets, values) in fast_list.items():
                    row[name] = values[offsets[i] : offsets[i + 1]]
                for name, flat in fast_flat.items():
                    row[name] = flat[i]
                for name, pylist in slow.items():
                    row[name] = pylist[i]
                yield row

        return _rows()

    @staticmethod
    def _column_output_key(column: str) -> str:
        """Return the top-level row key produced for a projected column."""
        return column.split(".", 1)[0]

    @classmethod
    def _projection_pairs(cls, columns: list[str]) -> list[tuple[str, str]]:
        """Return deduplicated ``(parquet_column, row_key)`` projection pairs."""
        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for column in columns:
            key = cls._column_output_key(column)
            if key not in seen:
                pairs.append((column, key))
                seen.add(key)
        return pairs

    @classmethod
    def _uses_slow_nested_projection(cls, columns: list[str] | None) -> bool:
        """Return whether projected reads should skip PyArrow's slow failing path."""
        if columns is None:
            return False
        return any(cls._column_output_key(column) in _PARQUET_SLOW_NESTED_COLUMNS for column in columns)

    def _iter_row_group_batches(
        self,
        pf: tp.Any,
        rg_idx: int,
        columns: list[str] | None,
        *,
        batch_size: int | None = None,
    ) -> "Iterator[dict]":
        """Yield row dictionaries from a row group through RecordBatch reads."""
        kwargs = {"row_groups": [rg_idx], "columns": columns, "use_threads": False}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        for batch in pf.iter_batches(**kwargs):
            yield from self._table_to_rows(batch)

    def _projected_row_group_bytes(self, pf: tp.Any, rg_idx: int) -> tuple[int, int]:
        """Return ``(uncompressed_bytes, num_rows)`` for the projected columns of one row group.

        Uses only footer metadata (no data reads). When no projection is set,
        counts every column.
        """
        rg = pf.metadata.row_group(rg_idx)
        roots: set[str] | None = None
        if self._columns is not None:
            roots = {self._column_output_key(c) for c in self._columns}
        total = 0
        for c in range(rg.num_columns):
            col = rg.column(c)
            if roots is None or col.path_in_schema.split(".", 1)[0] in roots:
                total += col.total_uncompressed_size
        return total, rg.num_rows

    def _nested_batch_ladder(self, pf: tp.Any, rg_idx: int) -> tuple[int, ...]:
        """Return descending batch sizes targeting a bounded per-batch byte budget.

        A row-count ladder alone is unsafe for embed-heavy rows (tens of MiB per
        row): 256 rows of `area_bucket=1536` VL data is a multi-GiB batch. Sizing
        from the row group's own metadata keeps every attempt near
        ``_PARQUET_NESTED_TARGET_BATCH_BYTES`` regardless of row weight.
        """
        proj_bytes, n_rows = self._projected_row_group_bytes(pf, rg_idx)
        if n_rows <= 0 or proj_bytes <= 0:
            return _PARQUET_NESTED_FALLBACK_BATCH_SIZES
        per_row = max(proj_bytes // n_rows, 1)
        first = int(_PARQUET_NESTED_TARGET_BATCH_BYTES // per_row)
        first = max(1, min(first, _PARQUET_NESTED_FALLBACK_BATCH_SIZE))
        sizes: list[int] = []
        while True:
            sizes.append(max(first, 1))
            if first <= 1:
                break
            first //= 4
        return tuple(sizes)

    def _iter_row_group_batches_with_fallback(
        self,
        pf: tp.Any,
        rg_idx: int,
        columns: list[str] | None,
        *,
        batch_sizes: tuple[int, ...] | None = None,
    ) -> "Iterator[dict]":
        """Yield row-group batches, reducing batch size for nested Arrow bugs.

        Retries are resume-safe: rows already delivered by a partial attempt are
        skipped (never re-yielded) when a smaller batch size is retried, so a
        mid-stream Arrow failure cannot silently duplicate rows downstream.
        """
        last_nested_error: Exception | None = None
        emitted = 0
        for batch_size in batch_sizes or _PARQUET_NESTED_FALLBACK_BATCH_SIZES:
            try:
                row_iter = self._iter_row_group_batches(
                    pf,
                    rg_idx,
                    columns,
                    batch_size=batch_size,
                )
                for row in itertools.islice(row_iter, emitted, None):
                    emitted += 1
                    yield row
                return
            except Exception as exc:
                if not self._is_nested_projection_error(exc):
                    raise
                last_nested_error = exc
        if last_nested_error is not None:
            raise last_nested_error

    def _iter_row_group_columns(self, pf: tp.Any, rg_idx: int, columns: list[str]) -> "Iterator[dict]":
        """Yield rows by reading projected columns one at a time."""
        pairs = self._projection_pairs(columns)
        if not pairs:
            return

        iterators = [
            (
                parquet_column,
                key,
                self._iter_row_group_batches(
                    pf,
                    rg_idx,
                    [parquet_column],
                    batch_size=_PARQUET_NESTED_FALLBACK_BATCH_SIZE,
                ),
            )
            for parquet_column, key in pairs
        ]

        while True:
            chunks: list[tuple[str, str, list[tp.Any]]] = []
            any_values = False
            for parquet_column, key, row_iter in iterators:
                values = [
                    row[key] for row in itertools.islice(row_iter, _PARQUET_NESTED_FALLBACK_BATCH_SIZE) if key in row
                ]
                if values:
                    any_values = True
                chunks.append((parquet_column, key, values))

            if not any_values:
                return

            expected_rows = len(chunks[0][2])
            for parquet_column, _key, values in chunks[1:]:
                if len(values) != expected_rows:
                    raise ValueError(
                        f"Projected Parquet column {parquet_column!r} yielded {len(values)} rows in a fallback "
                        f"batch; expected {expected_rows}."
                    )

            for i in range(expected_rows):
                yield {key: values[i] for _parquet_column, key, values in chunks}

    def _iter_row_group_rows(self, pf: tp.Any, rg_idx: int) -> "Iterator[dict]":
        """Yield rows from one row group, avoiding PyArrow's nested table bug.

        Row groups whose projected payload cannot form a single Arrow array
        (> ``_PARQUET_NESTED_CHUNK_SAFE_BYTES``, e.g. packed VL ``image_embeds``
        data at tens of MiB per row) are routed directly to byte-budgeted
        streaming batches: whole-row-group reads of such groups are guaranteed
        to fail with the nested-chunked-array error only after paying the full
        decode, and row-count-based batches can transiently allocate many GiB.
        """
        skip_projected_read = self._uses_slow_nested_projection(self._columns)
        proj_bytes, _n_rows = self._projected_row_group_bytes(pf, rg_idx)
        _max_safe_bytes = self._guard_bytes()
        oversized = 0 < _max_safe_bytes < proj_bytes
        warned = getattr(pf, "_ed_oversized_warned", None)
        if warned is None:
            warned = set()
            try:
                pf._ed_oversized_warned = warned
            except AttributeError:
                pass
        if oversized and rg_idx not in warned:
            warned.add(rg_idx)
            logger.warning(
                f"Oversized parquet row group {rg_idx} ({_n_rows} rows, projected payload "
                f"{proj_bytes / 2**30:.1f} GiB > {_max_safe_bytes / 2**30:.1f} GiB guard): streaming it in "
                "byte-budgeted batches instead of a whole-group decode. Rewrite the file with a smaller "
                "row_group_size for throughput."
            )
        heavy = oversized or skip_projected_read or proj_bytes > _PARQUET_NESTED_CHUNK_SAFE_BYTES
        use_threads = not self._force_single_threaded_reads
        if not heavy:
            try:
                table = pf.read_row_group(
                    rg_idx,
                    columns=self._columns,
                    use_threads=use_threads,
                )
                yield from self._table_to_rows(table)
                return
            except Exception as exc:
                if not self._is_nested_projection_error(exc):
                    raise

            self._warn_projection_fallback_once()
            self._force_single_threaded_reads = True

            if use_threads:
                try:
                    table = pf.read_row_group(rg_idx, columns=self._columns, use_threads=False)
                    yield from self._table_to_rows(table)
                    return
                except Exception as exc:
                    if not self._is_nested_projection_error(exc):
                        raise
        else:
            self._warn_projection_fallback_once()

        last_nested_error: Exception | None = None
        try:
            yield from self._iter_row_group_batches_with_fallback(
                pf,
                rg_idx,
                self._columns,
                batch_sizes=self._nested_batch_ladder(pf, rg_idx),
            )
            return
        except Exception as exc:
            if not self._is_nested_projection_error(exc):
                raise
            last_nested_error = exc

        if self._columns is None:
            raise last_nested_error
        yield from self._iter_row_group_columns(pf, rg_idx, self._columns)

    def get_shard_info(self, shard_name: str) -> ParquetShardInfo:
        """Get metadata about a Parquet shard including row group count.

        Results are cached in two layers: the per-process dict, and behind it
        the optional durable :class:`~easydel.data.shard_info_cache.ShardInfoDiskCache`
        (enabled by ``metadata_cache_dir``), which survives restarts. The
        durable layer validates every hit against a live listing and fails
        open, so its worst case is the plain footer read.

        Args:
            shard_name: Path or URL to the Parquet file.

        Returns:
            ParquetShardInfo with file metadata.
        """
        if shard_name in self._shard_info_cache:
            return self._shard_info_cache[shard_name]
        # The durable cache (``metadata_cache_dir``) takes its stat baseline inside this ``get``, BEFORE
        # the footer read below, so a shard replaced mid-walk is banked against its old stat and rejected.
        # ``shard_id`` is a property of this source's file list and is never served from it.
        cached = self._shard_info_disk.get(shard_name)
        if cached is not None:
            num_rows, num_row_groups = cached
            info = ParquetShardInfo(
                shard_id=self._files.index(shard_name),
                shard_name=shard_name,
                num_rows=num_rows,
                num_row_groups=num_row_groups,
            )
            self._shard_info_cache[shard_name] = info
            return info
        return self._call_with_retries(self._read_shard_info, shard_name)

    def _read_shard_info(self, shard_name: str) -> ParquetShardInfo:
        """Read one shard's footer metadata and cache it (no retry; see :meth:`get_shard_info`)."""
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
        self._shard_info_disk.record(shard_name, info.num_rows, info.num_row_groups)
        return info

    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open a Parquet shard and iterate over rows.

        Reads row groups sequentially and yields individual rows as dictionaries.
        Transient I/O failures mid-stream resume at the next unread row (a
        ``with_retry`` decorator is inert on generator functions — calling one
        only creates the generator, so the old decorator never actually
        retried anything here).

        Args:
            shard_name: Path or URL to the Parquet file.

        Yields:
            Individual rows as dictionaries.
        """
        yield from self._open_shard_resumable(shard_name, 0)

    @property
    def delivers_all_metadata_rows(self) -> bool:
        """Whether iteration delivers every physical row counted by ``num_rows``.

        Always ``True``: row groups above the decode-memory guard are streamed in
        byte-budgeted batches, never skipped.

        Returns:
            bool: ``True``.
        """
        return True

    def open_shard_at_row(self, shard_name: str, row: int) -> "Iterator[dict]":
        """Open a Parquet shard starting at a specific delivered row.

        Uses row group metadata to skip directly to the relevant row group,
        avoiding sequential scanning of earlier rows. Transient I/O failures
        mid-stream resume at the next unread row.

        Args:
            shard_name: Path or URL to the Parquet file.
            row: Zero-based delivered-row index to start from.

        Yields:
            Rows starting from the specified position.
        """
        yield from self._open_shard_resumable(shard_name, None, delivered_row=row)

    def _open_shard_resumable(
        self, shard_name: str, start_row: int | None, *, delivered_row: int = 0
    ) -> "Iterator[dict]":
        """Stream rows from ``start_row``, reopening at the next unread row on transient I/O errors.

        Long-running streaming jobs hold shard file handles open for minutes to
        hours; a single dropped connection mid-read must not kill the consumer
        (the trainer's prefetch thread). Each retryable failure reopens the
        shard at the first row not yet delivered, so the consumer sees no
        duplicates and no gaps. Successful progress resets the retry budget;
        repeated failures with no progress propagate.
        """
        import time as _time

        # ``start_row=None``: translate ``delivered_row`` to a physical row on the first attempt.
        position = None if start_row is None else int(start_row)
        max_reopens = _PARQUET_STREAM_MAX_REOPENS if self._max_retries is None else self._max_retries
        base_delay = 1.0 if self._retry_delay is None else self._retry_delay
        failures_without_progress = 0
        while True:
            try:
                if position is None:
                    position = self._physical_row_for_delivered_row(shard_name, delivered_row)
                for abs_row, example in self._iter_shard_rows(shard_name, position):
                    position = abs_row + 1
                    failures_without_progress = 0
                    yield example
                return
            except _PARQUET_TRANSIENT_IO_ERRORS as exc:
                failures_without_progress += 1
                if failures_without_progress > max_reopens:
                    raise
                delay = min(base_delay * (2 ** (failures_without_progress - 1)), 30.0)
                logger.warning(
                    f"Transient I/O error streaming {shard_name!r} at row {position} "
                    f"({type(exc).__name__}: {str(exc)[:120]}); reopening at that row in {delay:.0f}s "
                    f"(attempt {failures_without_progress}/{max_reopens})."
                )
                _time.sleep(delay)

    def _physical_row_for_delivered_row(self, shard_name: str, row: int) -> int:
        """Translate a delivered-row offset into a file-physical row index.

        Every row group is delivered (oversized ones are streamed, not skipped), so the two coincide.

        Args:
            shard_name: Path or URL to the Parquet file.
            row: Zero-based delivered-row offset.

        Returns:
            int: ``row`` clamped at zero.
        """
        del shard_name
        return max(int(row), 0)

    def _iter_shard_rows(self, shard_name: str, row: int) -> "Iterator[tuple[int, dict]]":
        """Single-pass ``(absolute_row_index, row)`` stream starting at ``row``.

        Yields the file-physical row index alongside each row so callers can
        resume by position. A ``row`` at or past the end of the file yields nothing.
        """
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs]

        row = max(int(row), 0)
        with self._open_file(shard_name) as fh:
            pf = pq.ParquetFile(fh)
            rg_starts: list[int] = []
            cumulative_rows = 0
            for rg_idx in range(pf.num_row_groups):
                rg_starts.append(cumulative_rows)
                cumulative_rows += pf.metadata.row_group(rg_idx).num_rows
            if row >= cumulative_rows:
                return

            start_rg = 0
            skip_in_rg = row
            for rg_idx in range(pf.num_row_groups - 1, -1, -1):
                if rg_starts[rg_idx] <= row:
                    start_rg = rg_idx
                    skip_in_rg = row - rg_starts[rg_idx]
                    break

            for rg_idx in range(start_rg, pf.num_row_groups):
                start_i = skip_in_rg if rg_idx == start_rg else 0
                base = rg_starts[rg_idx]
                for i, example in enumerate(self._iter_row_group_rows(pf, rg_idx)):
                    if i >= start_i:
                        yield base + i, example

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
        self._files = expand_data_files(data_files, storage_options=storage_options)
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

        return fsspec.open(path, "r", **{"compression": "infer", **self._storage_options})

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
                # ``row`` counts delivered records, and open_shard skips blank lines, so only
                # non-blank lines advance the seek counter (r11 S13-3).
                delivered = 0
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if delivered >= row:
                        yield json.loads(line)
                    delivered += 1
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

    Each Arrow IPC file or stream is one shard; iteration walks its
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
        self._files = expand_data_files(data_files, storage_options=storage_options)
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
            # HuggingFace datasets write IPC streams, whereas Feather and
            # ipc.new_file use the file container with ARROW1 magic.
            is_file = fh.read(6) == b"ARROW1"
            fh.seek(0)
            if is_file:
                reader = ipc.open_file(fh)
                batches = (reader.get_batch(index) for index in range(reader.num_record_batches))
            else:
                batches = ipc.open_stream(fh)
            for batch in batches:
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
        infer_types: bool = False,
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
            infer_types: Infer numeric, boolean, and null columns with the
                pandas/Arrow conversion used by HuggingFace's CSV reader.
                False keeps string-valued rows.
        """
        self._files = expand_data_files(data_files, storage_options=storage_options)
        self._storage_options = storage_options or {}
        self._delimiter = delimiter
        self._infer_types = infer_types

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

    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open a CSV shard and iterate over rows.

        Args:
            shard_name: Path or URL to the CSV file.

        Yields:
            Each row as a ``dict`` keyed by header column.
        """
        import csv

        import fsspec  # pyright: ignore[reportMissingTypeStubs]

        with fsspec.open(shard_name, "r", **{"compression": "infer", **self._storage_options}) as fh:
            if self._infer_types:
                import pandas as pd
                import pyarrow as pa

                with pd.read_csv(fh, sep=self._delimiter, chunksize=10_000) as batches:
                    for batch in batches:
                        yield from pa.Table.from_pandas(batch, preserve_index=False).to_pylist()
            else:
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
        self._files = expand_data_files(data_files, storage_options=storage_options)
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

        with fsspec.open(shard_name, "r", **{"compression": "infer", **self._storage_options}) as fh:
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


def _load_hf_dataset(path: str, *, streaming: bool, **kwargs):
    """Load HF data without asynchronous Parquet reads outliving a streaming reader.

    Arrow dataset scanners can leave native I/O tasks calling Python-backed files
    during shutdown after early-stopped training. Use synchronous Parquet batches
    for streaming while retaining HF discovery, schemas, transforms and sharding.
    Materialized datasets, other formats, and high-level encrypted Parquet scans
    retain HF's normal loading behavior.

    Args:
        path: Hub identifier, local dataset directory, or HF builder name.
        streaming: Whether to return a lazy iterable dataset.
        **kwargs: Remaining ``datasets.load_dataset`` arguments used by the source.

    Returns:
        A Hugging Face Dataset or IterableDataset.

    Raises:
        NotImplementedError: If streaming is combined with ``num_proc``.
    """
    from datasets import load_dataset, load_dataset_builder

    if not streaming:
        return load_dataset(path, streaming=False, **kwargs)
    if kwargs.pop("num_proc", None) is not None:
        raise NotImplementedError("Streaming datasets do not support num_proc.")
    split = kwargs.pop("split", None)
    builder = load_dataset_builder(path, **kwargs)
    if builder.info.builder_name == "parquet":
        import inspect

        from datasets import DownloadConfig

        from ._hf_parquet import SynchronousParquetTables

        # datasets 3.x has a different, files-only generator protocol. Leave that
        # supported legacy API intact rather than assuming the current HF layout.
        if "row_groups_list" not in inspect.signature(builder._generate_tables).parameters:
            return builder.as_streaming_dataset(split=split)
        options = builder.config.fragment_scan_options
        # High-level encryption configs require Arrow's filesystem/key-material
        # setup. Preserve that specialized HF path rather than dropping options.
        if options is None or options.decryption_config is None:
            builder._generate_tables = SynchronousParquetTables(
                builder.config,
                builder.info,
                DownloadConfig(token=builder.token, storage_options=builder.storage_options),
            )
    return builder.as_streaming_dataset(split=split)


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
        storage_options: dict | None = None,
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
        self._storage_options = storage_options
        self._dataset = None

    @property
    def shard_names(self) -> "Sequence[str]":
        """Return a single synthetic shard name for the HF split.

        Returns:
            One-element list of ``"{dataset_name}:{split}"``.
        """
        # HuggingFace datasets are treated as a single shard. The subset is
        # part of the identity: omitting it made two subsets of one dataset
        # collide inside CompositeShardedSource.
        if self._subset:
            return [f"{self._dataset_name}:{self._subset}:{self._split}"]
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
            self._dataset = _load_hf_dataset(
                self._dataset_name,
                name=self._subset,
                split=self._split,
                streaming=self._streaming,
                cache_dir=self._cache_dir,
                **({"storage_options": self._storage_options} if self._storage_options is not None else {}),
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

        seen: dict[str, int] = {}
        for src_idx, source in enumerate(sources):
            for shard_idx, shard_name in enumerate(source.shard_names):
                if shard_name in seen:
                    raise ValueError(
                        f"CompositeShardedSource shard name collision: {shard_name!r} is exposed by both "
                        f"source {seen[shard_name]} and source {src_idx}. Name-based dispatch would read "
                        "the first owner twice and the second never; give the sources distinct shard "
                        "names (e.g. distinct HF subsets/paths)."
                    )
                seen[shard_name] = src_idx
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

    @property
    def delivers_all_metadata_rows(self) -> bool:
        """Whether every wrapped source delivers all of its metadata rows.

        Returns:
            bool: ``all(source.delivers_all_metadata_rows)`` over the wrapped sources.
        """
        return all(source.delivers_all_metadata_rows for source in self._sources)

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


class RetryingShardedSource(ShardedDataSource[dict]):
    """Wrapper that reopens a shard at the next unread row after transient I/O errors.

    Formats without their own resilience (JSON, CSV, Arrow, text) read through a single file handle;
    on remote storage a dropped connection would otherwise kill the consumer. Each retryable failure
    reopens the shard at the first row not yet delivered, so the consumer sees no duplicates and no
    gaps. Successful progress resets the retry budget; repeated failures with no progress propagate.
    """

    def __init__(self, source: ShardedDataSource, max_retries: int = 3, retry_delay: float = 1.0):
        """Capture the wrapped source and the retry policy.

        Args:
            source: Source whose shard reads are made resilient.
            max_retries: Reopen attempts allowed without forward progress.
            retry_delay: Base delay in seconds; doubles per consecutive failure (capped at 30s).
        """
        self._source = source
        self._max_retries = max(int(max_retries), 0)
        self._retry_delay = max(float(retry_delay), 0.0)

    @property
    def shard_names(self) -> "Sequence[str]":
        """Return the wrapped source's shard names."""
        return self._source.shard_names

    def num_shards(self) -> int:
        """Return the wrapped source's shard count."""
        return self._source.num_shards()

    def get_shard_info(self, shard_name: str) -> ShardInfo | None:
        """Read metadata with the same retry policy as row streaming."""
        read = with_retry(max_retries=self._max_retries, initial_delay=self._retry_delay)(self._source.get_shard_info)
        return read(shard_name)

    @property
    def delivers_all_metadata_rows(self) -> bool:
        """Forward the wrapped source's flag (retrying never drops or adds rows)."""
        return self._source.delivers_all_metadata_rows

    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Stream ``shard_name`` from the start, resuming after transient failures."""
        yield from self._open_resumable(shard_name, 0)

    def open_shard_at_row(self, shard_name: str, row: int) -> "Iterator[dict]":
        """Stream ``shard_name`` from delivered row ``row``, resuming after transient failures."""
        yield from self._open_resumable(shard_name, max(int(row), 0))

    def _open_resumable(self, shard_name: str, position: int) -> "Iterator[dict]":
        """Yield rows from ``position``; reopen at the next undelivered row on transient I/O errors."""
        import time as _time

        failures = 0
        while True:
            try:
                rows = (
                    self._source.open_shard(shard_name)
                    if position == 0
                    else self._source.open_shard_at_row(shard_name, position)
                )
                for example in rows:
                    position += 1
                    failures = 0
                    yield example
                return
            except _PARQUET_TRANSIENT_IO_ERRORS as exc:
                failures += 1
                if failures > self._max_retries:
                    raise
                delay = min(self._retry_delay * (2 ** (failures - 1)), 30.0)
                logger.warning(
                    f"Transient I/O error streaming {shard_name!r} at row {position} "
                    f"({type(exc).__name__}: {str(exc)[:120]}); reopening in {delay:.0f}s "
                    f"(attempt {failures}/{self._max_retries})."
                )
                _time.sleep(delay)

    def __len__(self) -> int:
        """Length of the wrapped source."""
        return len(self._source)

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return f"RetryingShardedSource({self._source!r}, max_retries={self._max_retries})"


class MaterializedShardedSource(ShardedDataSource[dict]):
    """Non-streaming adapter: every shard of the wrapped source is loaded into RAM up front.

    Implements ``streaming=False`` semantics for file-backed sources
    (:attr:`SourceStageConfig.streaming` / :attr:`PipelineConfig.streaming`): the entire split is
    materialised before downstream stages run, so later passes (epochs, resumes) never re-read storage.
    """

    def __init__(self, source: ShardedDataSource):
        """Read every shard of ``source`` into memory.

        Args:
            source: Source to materialise.
        """
        self._names = list(source.shard_names)
        self._rows: dict[str, list[dict]] = {name: list(source.open_shard(name)) for name in self._names}

    @property
    def shard_names(self) -> "Sequence[str]":
        """Return the shard names of the materialised source."""
        return self._names

    def num_shards(self) -> int:
        """Return the number of materialised shards."""
        return len(self._names)

    def get_shard_info(self, shard_name: str) -> ShardInfo | None:
        """Return exact in-memory row counts for ``shard_name``."""
        return ShardInfo(
            shard_id=self._names.index(shard_name),
            shard_name=shard_name,
            num_rows=len(self._rows[shard_name]),
        )

    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Iterate the in-memory rows of ``shard_name``."""
        return iter(self._rows[shard_name])

    def open_shard_at_row(self, shard_name: str, row: int) -> "Iterator[dict]":
        """Iterate the in-memory rows of ``shard_name`` starting at ``row`` (O(1) seek)."""
        return itertools.islice(self._rows[shard_name], max(int(row), 0), None)

    def __len__(self) -> int:
        """Total number of in-memory rows."""
        return sum(len(rows) for rows in self._rows.values())

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return f"MaterializedShardedSource(shards={len(self._names)}, rows={len(self)})"


def create_source(
    config: "DatasetConfig",
    source_config: "SourceStageConfig | None" = None,
    streaming: bool | None = None,
    infer_csv_types: bool = False,
) -> ShardedDataSource:
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

    The optional stage settings are honoured as follows:

    * ``source_config.dask_storage_options`` become the ``fsspec`` ``storage_options`` of every
      file-backed source.
    * ``source_config.cloud_max_retries`` / ``cloud_retry_delay`` set the transient-I/O retry budget
      (Parquet retries internally; other cloud-backed formats are wrapped in
      :class:`RetryingShardedSource`).
    * streaming: ``streaming`` (typically :attr:`PipelineConfig.streaming`) and
      ``source_config.streaming`` must both be true for streaming reads. When either is false, Hub
      datasets are loaded non-streaming and file sources are materialised in memory per shard
      (:class:`MaterializedShardedSource`).
    * ``config.max_row_group_bytes`` sets the Parquet decode-memory guard.
    * ``source_config.metadata_cache_dir`` / ``process_index`` enable the durable Parquet
      shard-info cache (only process ``0``/unknown writes it).

    Args:
        config: Dataset declaration produced by the user (or
            :class:`PipelineConfig`).
        source_config: Optional :class:`SourceStageConfig` supplying streaming, retry and storage
            options. ``None`` uses the defaults.
        streaming: Optional global streaming default (:attr:`PipelineConfig.streaming`). ``None``
            means "streaming".
        infer_csv_types: Opt into HuggingFace-compatible CSV type/null inference.
            Defaults to False to preserve string-valued pipeline CSV rows.

    Returns:
        ShardedDataSource: One of :class:`ParquetShardedSource`,
        :class:`JsonShardedSource`, :class:`CsvShardedSource`,
        :class:`TextShardedSource`, :class:`ArrowShardedSource`, or
        :class:`HuggingFaceShardedSource` (possibly wrapped by the retry/materialise adapters).

    Raises:
        FileNotFoundError: When ``config.data_files`` points at no
            existing files and is not a string that could be a Hub
            dataset id.
    """
    data_files = config.data_files
    source_type = config.type
    storage_options = None
    max_retries: int | None = None
    retry_delay: float | None = None
    metadata_cache_dir: str | None = None
    process_index: int | None = None
    use_streaming = True if streaming is None else bool(streaming)
    if source_config is not None:
        use_streaming = use_streaming and bool(source_config.streaming)
        if source_config.dask_storage_options:
            storage_options = dict(source_config.dask_storage_options)
        max_retries = int(source_config.cloud_max_retries)
        retry_delay = float(source_config.cloud_retry_delay)
        metadata_cache_dir = getattr(source_config, "metadata_cache_dir", None)
        process_index = getattr(source_config, "process_index", None)

    def hub_source() -> ShardedDataSource:
        kwargs = {"storage_options": storage_options} if storage_options else {}
        result = HuggingFaceShardedSource(
            dataset_name=data_files,
            split=config.split,
            subset=config.dataset_split_name,
            streaming=use_streaming,
            **kwargs,
        )
        if max_retries is not None:
            result = RetryingShardedSource(result, max_retries=max_retries, retry_delay=retry_delay or 0.0)
        if not use_streaming:
            result = MaterializedShardedSource(result)
        return result

    if source_type in ("huggingface", "hf"):
        if isinstance(data_files, str) and not _is_pathlike(data_files):
            return hub_source()

    try:
        files = expand_data_files(data_files, storage_options=storage_options)
    except FileNotFoundError:
        if isinstance(data_files, str):
            return hub_source()
        raise

    # Use specified type or detect from files
    if source_type in ("json", "jsonl"):
        fmt = "json"
    elif source_type == "parquet":
        fmt = "parquet"
    elif source_type in ("csv", "tsv"):
        fmt = "csv"
    elif source_type == "arrow":
        fmt = "arrow"
    elif source_type == "txt":
        fmt = "txt"
    else:
        fmt = _detect_format(files)

    source: ShardedDataSource
    if fmt == "parquet":
        source = ParquetShardedSource(
            files,
            storage_options,
            max_row_group_bytes=getattr(config, "max_row_group_bytes", None),
            max_retries=max_retries,
            retry_delay=retry_delay,
            metadata_cache_dir=metadata_cache_dir,
            process_index=process_index,
        )
    else:
        if fmt == "json":
            jsonl = source_type == "jsonl" or any(_uncompressed_filename(f).endswith(".jsonl") for f in files)
            source = JsonShardedSource(files, storage_options, jsonl=jsonl)
        elif fmt == "csv":
            delimiter = (
                "\t" if source_type == "tsv" or any(_uncompressed_filename(f).endswith(".tsv") for f in files) else ","
            )
            source = CsvShardedSource(files, storage_options, delimiter=delimiter, infer_types=infer_csv_types)
        elif fmt == "txt":
            source = TextShardedSource(files, storage_options, text_field=config.content_field)
        else:
            source = ArrowShardedSource(files, storage_options)
        if max_retries is not None and any("://" in str(f) and not str(f).startswith("file://") for f in files):
            source = RetryingShardedSource(
                source,
                max_retries=max_retries,
                retry_delay=1.0 if retry_delay is None else retry_delay,
            )

    if not use_streaming:
        source = MaterializedShardedSource(source)
    return source


def _detect_builder_and_files(
    data_files: str | os.PathLike | list[str | os.PathLike], storage_options: dict | None = None
) -> tuple[str, list[str]]:
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
        storage_options: Optional credentials and filesystem settings for expansion.

    Returns:
        tuple[str, list[str]]: ``(builder_name, expanded_files)``.
    """
    files = expand_data_files(data_files, storage_options=storage_options)
    fmt = _detect_format(files)
    return fmt, files


def load_for_inform(inform, mixture):
    """Materialise one constituent of a :class:`DatasetMixture` as a HuggingFace dataset.

    Used by :func:`build_dataset` (the legacy mixture API) to load
    each :class:`BaseDatasetInform` entry into either a
    :class:`datasets.Dataset` or :class:`datasets.IterableDataset`.
    Routes Hub-style identifiers through ``load_dataset(path=name)``.
    File-based informs use optimized fsspec-backed sharded readers when
    ``mixture.use_fast_loader`` is true, and HuggingFace's standard file
    builders otherwise. When the builtin ``parquet`` builder fails
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
    from datasets import IterableDataset  # pyright: ignore[reportMissingTypeStubs]

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

    def _apply_file_split(dataset):
        """Apply HF file-builder split semantics before the independent raw-row cap.

        A flat list of file paths defines only the ``train`` split. Reuse
        HuggingFace's public instruction parser for absolute/percentage slices,
        negative bounds, rounding, and concatenation rather than treating a
        split expression as an unused source label. Streaming file builders
        accept only a plain split name, so retain their validation behavior.
        """
        split = str(inform.split or "train")
        if mixture.streaming:
            if split != "train":
                raise ValueError(f"Bad split: {split}. Available splits: ['train']")
            return dataset
        if split in {"train", "all"}:
            return dataset

        from datasets import ReadInstruction

        instructions = ReadInstruction.from_spec(split).to_absolute({"train": len(dataset)})
        # HF's Arrow reader uses take=-1 as a sentinel for the remaining
        # rows, including when reversed bounds produce that exact length.
        # Other reversed bounds select nothing; zero-width-only instructions
        # are rejected before reading. Preserve all three distinct cases.
        if all(instruction.from_ == instruction.to for instruction in instructions):
            raise ValueError(f'Instruction "{split}" corresponds to no data!')
        ranges = [
            range(instruction.from_, len(dataset) if instruction.to - instruction.from_ == -1 else instruction.to)
            for instruction in instructions
        ]
        indices = ranges[0] if len(ranges) == 1 else itertools.chain.from_iterable(ranges)
        return dataset.select(indices)

    # Create source and convert to HF dataset
    if t in {"huggingface", "hf"} and isinstance(df, str) and not _is_pathlike(df):
        dataset = _load_hf_dataset(
            path=df,
            name=inform.dataset_split_name,
            split=inform.split or "train",
            cache_dir=mixture.cache_dir,
            streaming=mixture.streaming,
            num_proc=None if mixture.streaming else 1,
        )
        return _apply_num_rows_limit(dataset)

    storage_options = getattr(mixture, "dask_storage_options", None) or {}
    if getattr(mixture, "use_fast_loader", False):
        from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]

        from ..core.config import DatasetConfig, SourceStageConfig

        source = create_source(
            DatasetConfig(
                data_files=df,
                type=t,
                split=inform.split or "train",
                content_field=getattr(inform, "content_field", None) or "text",
            ),
            source_config=SourceStageConfig(
                streaming=True,
                dask_storage_options=storage_options,
                cloud_max_retries=mixture.cloud_max_retries,
                cloud_retry_delay=mixture.cloud_retry_delay,
            ),
            infer_csv_types=True,
        )

        def rows():
            """Yield file rows lazily, preserving the source's shard order."""
            for shard_name in source.shard_names:
                yield from source.open_shard(shard_name)

        if mixture.streaming:
            dataset = IterableDataset.from_generator(rows)
        else:
            # Dataset.from_list infers columns only from its first row. Build
            # the complete key union before Arrow conversion so late optional
            # fields survive filtering/formatting and additional_fields selection.
            materialized = list(rows())
            keys = dict.fromkeys(key for row in materialized for key in row)
            dataset = Dataset.from_dict({key: [row.get(key) for row in materialized] for key in keys})
        return _apply_num_rows_limit(_apply_file_split(dataset))

    builder, files = _detect_builder_and_files(df, storage_options=storage_options)
    specified_builder = t if t in {"json", "jsonl", "csv", "parquet", "arrow", "tsv", "txt"} else None
    builder = specified_builder or builder
    is_tsv = builder in {"csv", "tsv"} and (t == "tsv" or any(_uncompressed_filename(f).endswith(".tsv") for f in files))
    builder_kwargs = {"delimiter": "\t"} if is_tsv else {}

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
            with fsspec.open(path, "rb", **storage_options) as fh:
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
        dataset = _load_hf_dataset(
            path={"jsonl": "json", "tsv": "csv", "txt": "text"}.get(builder, builder),
            data_files=files,
            split=inform.split or "train",
            cache_dir=mixture.cache_dir,
            streaming=mixture.streaming,
            num_proc=None if mixture.streaming else 1,
            storage_options=storage_options or None,
            **builder_kwargs,
        )
        return _apply_num_rows_limit(dataset)
    except ValueError as e:
        msg = str(e)
        if builder == "parquet" and ("Feature type 'List' not found" in msg or "from_dict" in msg):
            if mixture.streaming:
                dataset = IterableDataset.from_generator(lambda: _iter_parquet_rows(files))
            else:
                # Every other non-streaming path returns a map-style Dataset;
                # substituting an IterableDataset here broke len()/select()
                # consumers far from the cause.
                from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]

                dataset = Dataset.from_generator(lambda: _iter_parquet_rows(files))
            return _apply_num_rows_limit(_apply_file_split(dataset))
        raise
