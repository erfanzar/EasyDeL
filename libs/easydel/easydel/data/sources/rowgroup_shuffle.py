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

"""Row-group-granular global shuffle over parquet corpora, plus the shared row-group index.

The streaming reservoir (:class:`~easydel.data.transforms.source.ShuffledShardedSource`) only
decorrelates rows within a window of ``buffer_size`` rows; rows far apart in file order never swap,
so batches stay local samples of each dataset. :class:`RowGroupShuffledParquetSource` is the
practical middle ground: it enumerates every ``(file, row_group)`` pair, visits them in a seeded
global permutation and shuffles rows within each group as it is read. I/O stays sequential per row
group while sample order becomes globally decorrelated at row-group granularity. With
``group_pool_size=K > 1`` rows are drawn from K open groups at once, so a batch averages ~K content
clusters instead of draining one group at a time.

Determinism: the emitted order is a pure function of ``(seed, corpus, group_pool_size)``; seeking
(:meth:`RowGroupShuffledParquetSource.open_shard_at_row`) replays the same permutation using row
counts only, so checkpoint fast-forward is exact and reads no skipped data. The source exposes ONE
synthetic shard, so a mixer's per-epoch shard reshuffle cannot vary its order; build it with a
different ``seed`` per epoch for epoch-varying permutations.

Row-group index sidecar: :func:`enumerate_parquet_row_groups` footer-scans every file, which is
latency-bound (one blocking object-store GET per file) and re-paid on every restart. Two remedies:

* the cold scan runs on a bounded thread pool (``footer_scan_workers``, default 32): a
  3.3k-file corpus at ~2.9 s per serial footer read is hours of pure latency, and the reads are
  independent;
* with ``metadata_cache_dir`` set, results are persisted per file set and revalidated against one
  bulk listing per parent directory (size / mtime / generation per file), so a warm start rescans
  only new or changed files. Remote sidecars are written by process 0 only (concurrent rewrites of
  one object from many hosts trip object-store mutation rate limits); local sidecars are written by
  every process. Write failures degrade to a warning.

The oversized-row-group preflight (:func:`max_row_group_bytes`) is cached the same way, so a warm
start does not pay a second full footer pass for it.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import posixpath
import time
import typing as tp

import numpy as np
from ejkernel.loggings import get_logger

from ..core.protocols import ShardInfo
from ..shard_info_cache import _fs_for, _has_discriminator, _stat_key
from .base import _PARQUET_TRANSIENT_IO_ERRORS, ParquetShardedSource

if tp.TYPE_CHECKING:
    import os
    from collections.abc import Callable, Iterator, Sequence

logger = get_logger(__name__)

__all__ = (
    "DEFAULT_FOOTER_SCAN_WORKERS",
    "DEFAULT_MAX_DECODE_GROUP_BYTES",
    "RowGroupShuffledParquetSource",
    "enumerate_parquet_row_groups",
    "max_row_group_bytes",
    "read_parquet_row_group_rows",
    "read_parquet_row_group_table",
)

DEFAULT_MAX_DECODE_GROUP_BYTES: tp.Final[int] = 2 * 2**30
DEFAULT_FOOTER_SCAN_WORKERS: tp.Final[int] = 32

_DEFAULT_READ_ATTEMPTS = 3
_DEFAULT_READ_BACKOFF = 1.5
_RG_INDEX_VERSION = 1
_SYNTHETIC_SHARD = "rg_shuffled_0"


def _decode_guard_bytes(source: ParquetShardedSource) -> int:
    """Whole-group decode guard for ``source``: its ``max_row_group_bytes`` or the 2 GiB default."""
    value = getattr(source, "_max_row_group_bytes", None)
    return DEFAULT_MAX_DECODE_GROUP_BYTES if value is None else int(value)


def _read_attempts(source: ParquetShardedSource) -> int:
    retries = getattr(source, "_max_retries", None)
    return _DEFAULT_READ_ATTEMPTS if retries is None else max(int(retries), 0) + 1


def _read_backoff(source: ParquetShardedSource) -> float:
    delay = getattr(source, "_retry_delay", None)
    return _DEFAULT_READ_BACKOFF if delay is None else float(delay)


def _project_row(columns: list[str] | None, row: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """Apply a top-level column projection to one row dict (nested names keep their root key)."""
    if columns is None:
        return row
    projected: dict[str, tp.Any] = {}
    for column in columns:
        key = column if column in row else column.split(".", 1)[0]
        if key in row and key not in projected:
            projected[key] = row[key]
    return projected


def _is_remote(uri: str) -> bool:
    return "://" in uri and not uri.startswith("file://")


def _sidecar_writer(uri: str, process_index: int | None) -> bool:
    """Return whether this process may rewrite the sidecar at ``uri``.

    Local paths: every process writes its own copy. Remote object-store paths: process ``0`` (or an
    unknown index, ``None``) only, because many hosts rewriting one object at launch trips the
    per-object mutation rate limit and stalls the slow host past collective launch timeouts.
    """
    if not _is_remote(uri):
        return True
    return process_index in (None, 0)


def _sidecar_uri(base: str, kind: str, files: Sequence[str]) -> str:
    digest = hashlib.sha256("\n".join(sorted(files)).encode()).hexdigest()[:20]
    return f"{base.rstrip('/')}/{kind}_v{_RG_INDEX_VERSION}_{digest}.json.gz"


def _bulk_identities(source: ParquetShardedSource, files: Sequence[str]) -> tuple[dict[str, str], Callable]:
    """Return ``({normalised path: identity}, normaliser)`` from one detail listing per parent dir.

    The identity is the JSON of :func:`~easydel.data.shard_info_cache._stat_key` (size, mtime,
    generation). Entries with no discriminator are omitted, i.e. treated as uncacheable.
    """
    try:
        fs, _ = _fs_for(files[0], getattr(source, "_storage_options", None))
    except Exception:
        return {}, lambda p: p
    norm = getattr(fs, "_strip_protocol", None) or (lambda p: p)
    out: dict[str, str] = {}
    for parent in sorted({posixpath.dirname(f) for f in files}):
        try:
            listing = fs.ls(parent, detail=True)
        except Exception:
            continue  # missing parent -> those files are simply rescanned
        for entry in listing:
            try:
                key = _stat_key(entry)
                if _has_discriminator(key):
                    out[norm(entry["name"])] = json.dumps(list(key))
            except Exception:
                continue
    return out, norm


def _read_sidecar(uri: str, storage_options: dict | None, label: str) -> dict:
    try:
        fs, path = _fs_for(uri, storage_options)
        if not fs.exists(path):
            return {}
        with fs.open(path, "rb") as fh:
            payload = json.loads(gzip.decompress(fh.read()))
        if isinstance(payload, dict) and payload.get("version") == _RG_INDEX_VERSION:
            entries = payload.get("entries", {})
            return entries if isinstance(entries, dict) else {}
    except Exception as exc:  # corrupt/unreadable sidecar -> full rescan, loudly
        logger.warning("%s sidecar unreadable (%s: %s); rescanning.", label, uri, exc)
    return {}


def _write_sidecar(uri: str, entries: dict, storage_options: dict | None, process_index: int | None, label: str):
    if not _sidecar_writer(uri, process_index):
        logger.info("%s sidecar: shared write left to process 0 (%s).", label, uri)
        return
    try:
        blob = gzip.compress(json.dumps({"version": _RG_INDEX_VERSION, "entries": entries}).encode())
        fs, path = _fs_for(uri, storage_options)
        parent = path.rsplit("/", 1)[0]
        if parent and parent != path:
            try:
                fs.makedirs(parent, exist_ok=True)
            except Exception:
                pass
        with fs.open(path, "wb") as fh:
            fh.write(blob)
        logger.info("%s sidecar written: %s (%d files).", label, uri, len(entries))
    except Exception as exc:  # never fatal
        logger.warning("%s sidecar write failed (%s): %s", label, uri, exc)


def _footer_scan_workers(requested: int | None, n_files: int) -> int:
    workers = DEFAULT_FOOTER_SCAN_WORKERS if requested is None else int(requested)
    return max(1, min(max(workers, 1), n_files))


def _scan_paths_parallel(paths: Sequence[str], fn: Callable[[str], tp.Any], workers: int | None) -> dict:
    """``{path: fn(path)}`` over a bounded thread pool. A worker exception propagates (a short index is
    worse than a loud failure: it would silently shorten the corpus)."""
    paths = list(paths)
    if not paths:
        return {}
    n = _footer_scan_workers(workers, len(paths))
    if n == 1:
        return {p: fn(p) for p in paths}
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n) as pool:
        return dict(zip(paths, pool.map(fn, paths), strict=True))


def _read_metadata(source: ParquetShardedSource, path: str):
    """Footer metadata for one file with the source's transient-I/O retry budget."""
    import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs]

    attempts = _read_attempts(source)
    for attempt in range(attempts):
        try:
            with source._open_file(path) as handle:
                return pq.ParquetFile(handle).metadata
        except _PARQUET_TRANSIENT_IO_ERRORS:
            if attempt >= attempts - 1:
                raise
            time.sleep(_read_backoff(source) * (attempt + 1))
    raise RuntimeError("unreachable")  # pragma: no cover


def _scan_one_file(source: ParquetShardedSource, path: str) -> list[list[int]]:
    """Footer-scan one file -> ``[[row_group, num_rows], ...]`` for non-empty groups."""
    md = _read_metadata(source, path)
    return [[rg, int(md.row_group(rg).num_rows)] for rg in range(md.num_row_groups) if md.row_group(rg).num_rows > 0]


def _biggest_group_bytes(source: ParquetShardedSource, path: str) -> int:
    md = _read_metadata(source, path)
    return max((int(md.row_group(rg).total_byte_size) for rg in range(md.num_row_groups)), default=0)


def _cached_per_file_scan(
    source: ParquetShardedSource,
    files: list[str],
    *,
    kind: str,
    value_key: str,
    scan: Callable[[str], tp.Any],
    cache_dir: str | None,
    process_index: int | None,
    scan_workers: int | None,
) -> dict[str, tp.Any]:
    """Shared engine: ``{path: scan(path)}``, served per file from a validated sidecar when possible."""
    if not cache_dir:
        return _scan_paths_parallel(files, scan, scan_workers)

    storage_options = getattr(source, "_storage_options", None)
    identities, norm = _bulk_identities(source, files)
    uri = _sidecar_uri(cache_dir, kind, files)
    cached = _read_sidecar(uri, storage_options, kind)

    values: dict[str, tp.Any] = {}
    entries: dict[str, dict] = {}
    to_scan: list[str] = []
    for path in files:
        ident = identities.get(norm(path))
        hit = cached.get(path)
        if ident is not None and isinstance(hit, dict) and hit.get("identity") == ident and value_key in hit:
            values[path] = hit[value_key]
            entries[path] = hit
        else:
            to_scan.append(path)

    if not to_scan:
        logger.info("%s sidecar: HIT, %d files, footer scan skipped.", kind, len(files))
        return values

    logger.info(
        "%s sidecar: scanning %d/%d files (cache %s).",
        kind,
        len(to_scan),
        len(files),
        "MISS" if len(to_scan) == len(files) else "partial",
    )
    scanned = _scan_paths_parallel(to_scan, scan, scan_workers)
    for path in to_scan:
        values[path] = scanned[path]
        ident = identities.get(norm(path))
        if ident is not None:  # unlistable files are served this run but never banked
            entries[path] = {"identity": ident, value_key: scanned[path]}
    _write_sidecar(uri, entries, storage_options, process_index, kind)
    return values


def enumerate_parquet_row_groups(
    source: ParquetShardedSource,
    *,
    cache_dir: str | None = None,
    process_index: int | None = None,
    scan_workers: int | None = None,
) -> list[tuple[str, int, int]]:
    """Enumerate non-empty ``(path, row_group, num_rows)`` in FILE order from footer metadata.

    Shared by the row-group shuffle stream and the random-access source, so both index the corpus
    with identical retry semantics.

    Args:
        source: Parquet source whose files are indexed.
        cache_dir: Sidecar root; defaults to the source's ``metadata_cache_dir``. Falsy disables it.
        process_index: Writer gate for remote sidecars; defaults to the source's ``process_index``.
        scan_workers: Footer-scan threads; defaults to the source's ``footer_scan_workers`` or 32.

    Returns:
        list[tuple[str, int, int]]: Groups in file order, then row-group order.
    """
    files = list(source._files)
    scanned = _cached_per_file_scan(
        source,
        files,
        kind="rg_index",
        value_key="groups",
        scan=lambda p: _scan_one_file(source, p),
        cache_dir=cache_dir if cache_dir is not None else getattr(source, "_metadata_cache_dir", None),
        process_index=process_index if process_index is not None else getattr(source, "_process_index", None),
        scan_workers=scan_workers if scan_workers is not None else getattr(source, "_footer_scan_workers", None),
    )
    return [(path, int(rg), int(rows)) for path in files for rg, rows in scanned[path]]


def max_row_group_bytes(
    source: ParquetShardedSource,
    files: Sequence[str],
    *,
    cache_dir: str | None = None,
    process_index: int | None = None,
    scan_workers: int | None = None,
) -> dict[str, int]:
    """``{path: largest row-group total_byte_size}``, sidecar-cached per file like the row-group index.

    Without the cache this preflight would be a second full footer pass after an index served from
    its sidecar. A changed file rescans only itself.

    Args:
        source: Parquet source the files belong to.
        files: Paths to measure.
        cache_dir: Sidecar root; defaults to the source's ``metadata_cache_dir``.
        process_index: Writer gate for remote sidecars; defaults to the source's ``process_index``.
        scan_workers: Footer-scan threads; defaults to the source's ``footer_scan_workers`` or 32.

    Returns:
        dict[str, int]: Largest row group in bytes per file.
    """
    files = list(files)
    if not files:
        return {}
    scanned = _cached_per_file_scan(
        source,
        files,
        kind="rg_bytes",
        value_key="max_bytes",
        scan=lambda p: _biggest_group_bytes(source, p),
        cache_dir=cache_dir if cache_dir is not None else getattr(source, "_metadata_cache_dir", None),
        process_index=process_index if process_index is not None else getattr(source, "_process_index", None),
        scan_workers=scan_workers if scan_workers is not None else getattr(source, "_footer_scan_workers", None),
    )
    return {path: int(scanned[path]) for path in files}


def _check_group_size(source: ParquetShardedSource, pf, path: str, row_group: int) -> None:
    limit = _decode_guard_bytes(source)
    if limit > 0:
        nbytes = int(pf.metadata.row_group(row_group).total_byte_size)
        if nbytes > limit:
            raise MemoryError(
                f"Row group {row_group} of {path} is {nbytes / 2**30:.2f} GiB (> max_row_group_bytes={limit}); "
                "decoding it whole risks the host. Rewrite the file with a smaller row_group_size or raise "
                "max_row_group_bytes on the source."
            )


def _readable_columns(source: ParquetShardedSource, pf) -> list[str] | None:
    """Projection pushed to the reader: only columns the file has.

    pyarrow silently drops unknown names from a projected read (0-column table, rows intact), so
    unknown names are filtered here and projected post-read instead.
    """
    columns = source._columns
    if columns is None:
        return None
    available = set(pf.schema_arrow.names)
    kept = [c for c in columns if c.split(".", 1)[0] in available]
    return kept or None


def _retrying_group_read(source, path: str, row_group: int, expected_rows, parquet_file, decode):
    """Run ``decode(ParquetFile)`` with transient-I/O retries and the metadata row-count check.

    ``parquet_file`` (a cached handle) serves the first attempt only; retries reopen fresh, since a
    transient error may have poisoned the cached handle.
    """
    import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs]

    attempts = _read_attempts(source)
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            if parquet_file is not None and attempt == 0:
                result, n = decode(parquet_file)
            else:
                with source._open_file(path) as handle:
                    result, n = decode(pq.ParquetFile(handle))
            if expected_rows is not None and n != expected_rows:
                raise RuntimeError(
                    f"Row group {row_group} of {path} read {n} rows but metadata promised {expected_rows}; "
                    "the file changed since indexing and seek arithmetic would silently misalign. "
                    "Rebuild the source."
                )
            return result
        except _PARQUET_TRANSIENT_IO_ERRORS as exc:
            last_exc = exc
            logger.warning(f"transient read error on {path} rg={row_group} (attempt {attempt + 1}): {exc!r}")
            if attempt < attempts - 1:
                time.sleep(_read_backoff(source) * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def read_parquet_row_group_table(
    source: ParquetShardedSource,
    path: str,
    row_group: int,
    expected_rows: int | None = None,
    parquet_file=None,
):
    """Read one row group as an Arrow table with the size guard, retry and row-count check.

    For random-access consumers that use one row per lookup: materialising a whole group to Python
    dicts is GIL-bound and was measured at 137 ms (2.9k-row group) to 1.4 s (20k rows) per lookup,
    while the columnar decode releases the GIL and one ``table.slice(i, 1).to_pylist()`` is cheap.

    Args:
        source: Owning source (projection, guard and retry settings).
        path: File path.
        row_group: Row-group index.
        expected_rows: Row count promised by the index; a mismatch raises.
        parquet_file: Optional pre-opened ``pq.ParquetFile`` for the first attempt.

    Returns:
        tuple: ``(table, row_projection_needed)``; when True the caller projects each row.
    """

    def _decode(pf):
        _check_group_size(source, pf, path, row_group)
        read_columns = _readable_columns(source, pf)
        try:
            table = pf.read_row_group(row_group, columns=read_columns)
            if read_columns is not None and table.num_columns == 0 and table.num_rows > 0:
                raise RuntimeError(
                    f"Projected read of {path} rg={row_group} returned 0 columns for {table.num_rows} rows."
                )
            return (table, read_columns is None and source._columns is not None), table.num_rows
        except Exception as exc:
            if source._columns is not None and source._is_nested_projection_error(exc):
                source._warn_projection_fallback_once()
                table = pf.read_row_group(row_group)
                return (table, True), table.num_rows
            raise

    return _retrying_group_read(source, path, row_group, expected_rows, parquet_file, _decode)


def read_parquet_row_group_rows(
    source: ParquetShardedSource,
    path: str,
    row_group: int,
    expected_rows: int | None = None,
    parquet_file=None,
) -> list[dict]:
    """Read one row group as row dicts with the size guard, retry and row-count check.

    Raises instead of skipping oversized groups: callers key seek offsets and random-access indices
    to metadata row counts, so dropping a group would silently misalign every later lookup.

    Args:
        source: Owning source (projection, guard and retry settings).
        path: File path.
        row_group: Row-group index.
        expected_rows: Row count promised by the index; a mismatch raises.
        parquet_file: Optional pre-opened ``pq.ParquetFile`` for the first attempt.

    Returns:
        list[dict]: The group's rows in file order.
    """
    columns = source._columns

    def _decode(pf):
        _check_group_size(source, pf, path, row_group)
        read_columns = _readable_columns(source, pf)
        try:
            table = pf.read_row_group(row_group, columns=read_columns)
            if read_columns is not None and table.num_columns == 0 and table.num_rows > 0:
                raise RuntimeError(
                    f"Projected read of {path} rg={row_group} returned 0 columns for {table.num_rows} rows."
                )
            rows = list(source._table_to_rows(table))
            if read_columns is None and columns is not None:
                rows = [_project_row(columns, r) for r in rows]
        except Exception as exc:
            if columns is not None and source._is_nested_projection_error(exc):
                source._warn_projection_fallback_once()
                table = pf.read_row_group(row_group)
                rows = [_project_row(columns, r) for r in source._table_to_rows(table)]
            else:
                raise
        return rows, len(rows)

    return _retrying_group_read(source, path, row_group, expected_rows, parquet_file, _decode)


class RowGroupShuffledParquetSource(ParquetShardedSource):
    """:class:`ParquetShardedSource` variant emitting a globally permuted row-group stream.

    Exposes a single synthetic shard (``"rg_shuffled_0"``) whose order is a seeded permutation over
    every row group of every file, with optional within-group row shuffling and K-way group
    interleaving. ``open_shard_at_row`` walks the same permutation, so resume is exact.

    Example:
        >>> src = RowGroupShuffledParquetSource("gs://bucket/data/*.parquet", seed=0, group_pool_size=32)
        >>> for row in src.open_shard(src.shard_names[0]):
        ...     process(row)
    """

    def __init__(
        self,
        data_files: str | os.PathLike | list[str | os.PathLike],
        storage_options: dict | None = None,
        columns: list[str] | None = None,
        *,
        seed: int = 0,
        shuffle_within_group: bool = True,
        emit_original_index: bool = False,
        group_pool_size: int = 1,
        max_row_group_bytes: int | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        metadata_cache_dir: str | None = None,
        process_index: int | None = None,
        footer_scan_workers: int | None = None,
    ):
        """Capture the shuffle settings; the corpus is indexed lazily on first use.

        Args:
            data_files: Glob, path list or directory (see :func:`expand_data_files`).
            storage_options: ``fsspec`` storage options.
            columns: Optional column projection.
            seed: Permutation seed; the emitted order is a pure function of it.
            shuffle_within_group: Permute rows inside each group (seeded per group).
            emit_original_index: Add ``__orig_row__`` (file-order corpus position) to each row.
            group_pool_size: ``K``; ``1`` drains one group at a time, ``K > 1`` draws each row from
                a uniformly chosen one of K open groups (memory: at most K decoded groups).
            max_row_group_bytes: Whole-group decode guard; ``None`` uses 2 GiB, ``0`` disables.
            max_retries: Transient-I/O retries per footer/group read (``None``: 2).
            retry_delay: Linear backoff base in seconds between those retries (``None``: 1.5).
            metadata_cache_dir: Root for the row-group index / shard-info sidecars (``None``: off).
            process_index: Process index; only ``0``/``None`` writes remote sidecars.
            footer_scan_workers: Threads for the cold footer scan (``None``: 32).
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
        self._footer_scan_workers = footer_scan_workers
        self._rg_seed = int(seed)
        self._shuffle_within_group = bool(shuffle_within_group)
        self._emit_original_index = bool(emit_original_index)
        self._group_pool_size = max(1, int(group_pool_size))
        self._group_index: list[tuple[str, int, int]] | None = None  # permuted (path, rg, rows)
        self._orig_starts: dict[tuple[str, int], int] = {}  # (path, rg) -> file-order corpus offset

    def _groups(self) -> list[tuple[str, int, int]]:
        if self._group_index is None:
            groups = enumerate_parquet_row_groups(self)
            offset = 0
            for path, rg, rows in groups:
                self._orig_starts[(path, rg)] = offset
                offset += rows
            order = np.random.default_rng(self._rg_seed).permutation(len(groups))
            self._group_index = [groups[i] for i in order]
        return self._group_index

    @property
    def shard_names(self) -> Sequence[str]:
        """The single synthetic shard name."""
        return [_SYNTHETIC_SHARD]

    def num_shards(self) -> int:
        """Return 1 (a method, per the :class:`ShardedDataSource` contract)."""
        return 1

    @property
    def delivers_all_metadata_rows(self) -> bool:
        """Always ``True``: oversized groups raise instead of being skipped."""
        return True

    def get_shard_info(self, shard_name: str) -> ShardInfo:
        """Return the synthetic shard's info (total corpus row count)."""
        return ShardInfo(shard_id=0, shard_name=_SYNTHETIC_SHARD, num_rows=len(self))

    def _read_group_rows(self, path: str, row_group: int, expected_rows: int | None = None) -> list[dict]:
        return read_parquet_row_group_rows(self, path, row_group, expected_rows=expected_rows)

    def _group_row_order(self, group_idx: int, n_rows: int) -> np.ndarray:
        if not self._shuffle_within_group or n_rows < 2:
            return np.arange(n_rows)
        return np.random.default_rng((self._rg_seed, 0x5247, int(group_idx))).permutation(n_rows)

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        """Iterate the whole permuted stream."""
        yield from self.open_shard_at_row(shard_name, 0)

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        """Iterate the permuted stream, skipping exactly the first ``row`` rows without reading them."""
        skip = max(int(row), 0)
        if self._group_pool_size > 1:
            yield from self._open_pooled_at_row(skip)
            return
        for group_idx, (path, rg, n_rows) in enumerate(self._groups()):
            if skip >= n_rows:
                skip -= n_rows
                continue
            rows = self._read_group_rows(path, rg, expected_rows=n_rows)
            order = self._group_row_order(group_idx, len(rows))
            base = self._orig_starts.get((path, rg), 0)
            for i in order[skip:]:
                out = rows[int(i)]
                if self._emit_original_index:
                    out["__orig_row__"] = base + int(i)
                yield out
            skip = 0

    def _open_pooled_at_row(self, skip: int) -> Iterator[dict]:
        """K-way group-pool interleave over the same permuted deck.

        Per emitted row: draw a uniform slot, emit that group's next (within-group shuffled) row,
        and when a group empties pop it and refill from the deck in order. One ``rng.integers`` draw
        per row makes the order a pure function of ``(seed, deck, K)``; seeking replays the
        draw/pop/refill sequence on counts alone and decodes nothing for skipped rows.
        """
        deck = self._groups()
        pool_k = min(self._group_pool_size, len(deck))
        rng = np.random.default_rng((self._rg_seed, 0x504F4F4C))
        slots: list[list] = []  # [deck_idx, taken, rows | None, order | None, base]; rows lazy
        next_deck = 0

        def _refill() -> None:
            nonlocal next_deck
            while len(slots) < pool_k and next_deck < len(deck):
                slots.append([next_deck, 0, None, None, 0])
                next_deck += 1

        _refill()
        while slots:
            m = int(rng.integers(len(slots)))
            slot = slots[m]
            deck_idx = slot[0]
            n_rows = deck[deck_idx][2]
            if skip > 0:
                skip -= 1
                slot[1] += 1
            else:
                if slot[2] is None:
                    path, rg, exp = deck[deck_idx]
                    slot[2] = self._read_group_rows(path, rg, expected_rows=exp)
                    slot[3] = self._group_row_order(deck_idx, len(slot[2]))
                    slot[4] = self._orig_starts.get((path, rg), 0)
                i = int(slot[3][slot[1]])
                slot[1] += 1
                out = slot[2][i]
                if self._emit_original_index:
                    out["__orig_row__"] = slot[4] + i
                yield out
            if slot[1] >= n_rows:
                slots.pop(m)
                _refill()

    def __len__(self) -> int:
        return sum(rows for _, _, rows in self._groups())

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(files={len(self._files)}, seed={self._rg_seed}, "
            f"group_pool_size={self._group_pool_size})"
        )
