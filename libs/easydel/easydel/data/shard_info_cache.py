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

"""Durable parquet shard-metadata cache, validated at the point of serving.

:meth:`ParquetShardedSource.get_shard_info` reads each shard's parquet footer to learn
``num_rows`` / ``num_row_groups``. Its own cache is a per-process dict, so the whole footer walk is
re-paid by every process start on every host. A footer read over GCS measured a median of ~0.16 s
in-region, so a ~3.6k-shard mixture costs ~10 minutes per uncontended host before the first step,
and more when many hosts walk the same corpus at once. This module gives that dict a durable,
optional backing store. Nothing else about the read path changes.

Correctness rules (``num_rows`` drives positioning and resume, so a wrong value does not crash: it
silently changes which rows a run trains on):

* **Only file-intrinsic numbers are stored.** ``shard_id`` is ``files.index(name)``, a property of
  one source's file list, and is recomputed by the caller.
* **Validation is ``(size, mtime, generation)`` from one bulk listing per parent directory.**
  ``generation`` (gcsfs) / ``etag`` (s3fs) is the exact object identity and changes on every
  rewrite, including same-size rewrites inside one mtime tick. Missing components compare equal
  only when absent on both sides, and an entry with no component at all is refused on write and on
  read (it would validate against nothing).
* **The check runs on every served hit** (:meth:`ShardInfoDiskCache.get`), not only in
  :meth:`~ShardInfoDiskCache.load`. Any path that fills the entry map (load, flush merges, …)
  therefore cannot make an unvalidated record servable.
* **The stat baseline predates the first footer read.** It is taken in :meth:`load` and memoised;
  :meth:`record` refuses to bank a shard when no baseline exists. A shard replaced mid-walk is then
  banked against its *old* stat and rejected next start, instead of pairing a post-swap stat with a
  pre-swap row count (an entry that would look valid forever).
* **Proven-stale records are evicted, not re-persisted.** Flushes merge with the stored object so
  concurrent writers add to each other; records the live listing proves stale (listed, different
  stat) or that can never validate are dropped. Records the listing does not cover are kept: a
  transient listing failure must not empty the cache.
* **Fail-open everywhere.** A missing, unreadable, corrupt or wrongly shaped object, an unlistable
  backend or a failed write all degrade to the original footer read and never raise.

Writer election is explicit: pass ``process_index``. Index ``0`` (or ``None`` when the caller does
not know it) writes; other indices only read. Electing several writers is safe because serving
re-validates every hit: a redundant writer costs at most a lost update (one footer read later),
never a wrong ``num_rows``. Electing one avoids a write stampede against a single object.

The object is keyed on a digest of the sorted file list, so two spellings of one set share an
object and a grown set cannot reuse a smaller set's entries. The price: appending one shard mints a
new object and the next run re-walks everything once. :func:`prewarm` (also ``python -m
easydel.data.shard_info_cache --root ... --prewarm GLOB``) pays the walk from a single machine.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import socket
import typing as tp
import weakref

from ejkernel.loggings import get_logger

__all__ = (
    "DEFAULT_FLUSH_EVERY",
    "ShardInfoDiskCache",
    "cache_for_files",
    "prewarm",
)

logger = get_logger(__name__)

_SCHEMA: tp.Final[int] = 2
DEFAULT_FLUSH_EVERY: tp.Final[int] = 256
_GENERATION_KEYS: tp.Final[tuple[str, ...]] = ("generation", "etag", "ETag")
_MTIME_KEYS: tp.Final[tuple[str, ...]] = ("mtime", "LastModified", "updated")

_LIVE_CACHES: weakref.WeakSet = weakref.WeakSet()
_ATEXIT_REGISTERED: bool = False


def _fs_for(path: str, storage_options: dict | None):
    """Resolve ``path`` to ``(fsspec filesystem, stripped path)``.

    Args:
        path: Local path or URI.
        storage_options: ``fsspec`` storage options (credentials, project ids, …).

    Returns:
        tuple: ``(AbstractFileSystem, str)`` as returned by ``fsspec.core.url_to_fs``.
    """
    import fsspec  # pyright: ignore[reportMissingTypeStubs]

    return fsspec.core.url_to_fs(path, **(storage_options or {}))


def _stat_key(entry: tp.Mapping) -> tuple:
    """Normalise a listing entry to ``(size, mtime, generation)`` across fsspec backends.

    Backends disagree on the mtime field name (``mtime`` / ``updated`` / ``LastModified``) and type
    (float epoch, ISO string, tz-aware ``datetime``). Non-JSON types are stringified here so they
    can be persisted and compared after a round trip. Presence is tested against ``None``, not
    truthiness: an mtime of ``0`` or an empty etag is a real value.

    Args:
        entry: One ``fs.ls(..., detail=True)`` record.

    Returns:
        tuple: ``(int | None, int | float | str | None, str | None)``.

    Raises:
        Exception: On a malformed row (e.g. non-numeric ``size``); the caller drops that entry.
    """
    size = entry.get("size")
    size = int(size) if size is not None else None

    mtime = None
    for key in _MTIME_KEYS:
        value = entry.get(key)
        if value is not None:
            mtime = value if isinstance(value, int | float | str) else str(value)
            break

    generation = None
    for key in _GENERATION_KEYS:
        value = entry.get(key)
        if value is not None:
            generation = str(value)
            break

    return (size, mtime, generation)


def _has_discriminator(stat: tp.Sequence | None) -> bool:
    """Return whether a stat key can distinguish one version of an object from another.

    Args:
        stat: A ``(size, mtime, generation)`` triple, or ``None``.

    Returns:
        bool: ``True`` when at least one component is present.
    """
    return stat is not None and any(part is not None for part in stat)


def _tmp_suffix() -> str:
    """Build a staging-file suffix unique across hosts, not merely across processes.

    PIDs restart at small numbers in every container, so a shared POSIX/NFS root needs the host in
    the name or two hosts could interleave bytes into one staging path.

    Returns:
        str: ``<sanitised host>-<pid>-<random>``.
    """
    try:
        host = socket.gethostname() or "nohost"
    except Exception:
        host = "nohost"
    host = re.sub(r"[^A-Za-z0-9_.-]", "-", host)[:64] or "nohost"
    return f"{host}-{os.getpid()}-{os.urandom(4).hex()}"


def _valid_counts(rec: tp.Any) -> bool:
    """Return whether a stored record carries two usable counts.

    Booleans are refused (``isinstance(True, int)`` is true, and a JSON ``true`` would otherwise be
    served as ``num_rows=1``), as are negatives. Every real parquet file has at least one row group:
    a zero-row table still writes ``(0, 1)``; the only ``(0, 0)`` producer is a writer opened and
    closed without rows (a crashed producer), so ``num_row_groups == 0`` is refused. Refusing costs
    one footer read.

    Args:
        rec: A candidate record from the stored object.

    Returns:
        bool: ``True`` when both counts are non-negative, non-boolean integers and coherent.
    """
    if not isinstance(rec, dict):
        return False
    rows, groups = rec.get("num_rows"), rec.get("num_row_groups")
    for value in (rows, groups):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return False
    return groups >= 1


class ShardInfoDiskCache:
    """Durable ``shard_name -> (num_rows, num_row_groups)`` map keyed by the object's stat.

    Inert unless ``root`` is set. Every method fails open: on any error the caller is told "no
    entry", which costs a footer read and never raises.
    """

    def __init__(
        self,
        files: tp.Sequence[str],
        storage_options: dict | None = None,
        *,
        root: str | None = None,
        process_index: int | None = None,
        flush_every: int = DEFAULT_FLUSH_EVERY,
    ):
        """Bind the cache to one source's file list.

        Args:
            files: The source's expanded shard paths; the object name is derived from this list.
            storage_options: ``fsspec`` storage options for the shard listings and for the cache
                object's own I/O.
            root: Directory or URI prefix (``gs://bucket/prefix``) holding cache objects. ``None``
                or ``""`` disables the cache.
            process_index: This process's index in a multi-host job. ``0`` or ``None`` (unknown)
                may write the shared object; any other index only reads.
            flush_every: Persist after this many newly learned entries (``>= 1``).
        """
        self._files = list(files)
        self._storage_options = storage_options or {}
        self._root = (root or "").strip()
        if self._root and "://" not in self._root and not os.path.isabs(self._root):
            logger.warning("shard-info cache root %r is relative; it resolves against each process's CWD.", self._root)
        self._process_index = None if process_index is None else int(process_index)
        self._flush_every = max(1, int(flush_every))
        self.enabled = bool(self._root) and bool(self._files)
        self._entries: dict[str, dict] = {}  # loaded records; re-validated on every serve
        self._learned: dict[str, dict] = {}  # discovered this process, not yet persisted
        self._loaded = False
        self._stats: dict[str, tuple] | None = None
        self._forced_writer = False
        digest = hashlib.sha256("\n".join(sorted(self._files)).encode()).hexdigest()[:32]
        self._uri = f"{self._root.rstrip('/')}/shardinfo-v{_SCHEMA}-{digest}.json" if self.enabled else ""

    @property
    def uri(self) -> str:
        """Location of this file set's cache object (``""`` when disabled)."""
        return self._uri

    def force_writer(self) -> None:
        """Write regardless of ``process_index``. Used by :func:`prewarm` and by tests."""
        self._forced_writer = True

    def _may_write(self) -> bool:
        """Return whether this instance may persist.

        Returns:
            bool: ``True`` when forced, or when ``process_index`` is ``0`` or unknown.
        """
        return self._forced_writer or self._process_index in (None, 0)

    def _current_stats(self) -> dict[str, tuple]:
        """Return ``(size, mtime, generation)`` per file from one detail listing per parent dir.

        Memoised: the baseline is taken once, before the first footer read, and reused. Every path
        is indexed under both the backend's spelling (gcsfs lists ``bucket/key``) and the
        protocol-qualified one (``gs://bucket/key``) so lookups match ``expand_data_files`` output.

        Returns:
            dict: path -> stat key; empty when the backend cannot be listed at all.
        """
        if self._stats is not None:
            return self._stats
        stats: dict[str, tuple] = {}
        try:
            fs, _ = _fs_for(self._files[0], self._storage_options)
            parents = sorted({f.rsplit("/", 1)[0] for f in self._files})
            for parent in parents:
                try:
                    listing = fs.ls(parent, detail=True)
                except Exception:
                    continue
                for entry in listing:
                    # Per entry: one malformed row costs one shard's stat, not the directory's.
                    try:
                        name = entry.get("name")
                        if not name:
                            continue
                        key = _stat_key(entry)
                        stats[name] = key
                        try:
                            stats[fs.unstrip_protocol(name)] = key
                        except Exception:
                            pass
                    except Exception:
                        continue
        except Exception:
            stats = {}
        self._stats = stats
        return stats

    def _stat_of(self, shard_name: str) -> tuple | None:
        """Look up one shard's stat key in the bulk listing.

        Args:
            shard_name: The path as the source spells it.

        Returns:
            tuple | None: The stat key, or ``None`` when the object was not listed.
        """
        stats = self._current_stats()
        if shard_name in stats:
            return stats[shard_name]
        return stats.get(shard_name.split("://", 1)[-1])

    def _entry_is_fresh(self, shard_name: str, rec: tp.Any) -> bool:
        """Return whether ``rec`` may be served for ``shard_name`` against the live stat baseline.

        The only validation gate; it runs on the serving path so no way of populating the entry
        map can bypass it. Never raises.

        Args:
            shard_name: The path as the source spells it.
            rec: A candidate record.

        Returns:
            bool: ``True`` when the record is well formed and its stat key matches the object's.
        """
        try:
            if not _valid_counts(rec):
                return False
            want = (rec.get("size"), rec.get("mtime"), rec.get("generation"))
            got = self._stat_of(shard_name)
            if not _has_discriminator(want) or not _has_discriminator(got):
                return False
            return list(got) == list(want)  # list(): a JSON round trip turns tuples into lists
        except Exception:
            return False

    def _read_object(self) -> dict:
        """Return the stored ``shard_name -> record`` mapping, or ``{}``. Never raises.

        The shape is checked rather than assumed: a valid JSON document of the wrong shape must
        cost a footer walk, not an ``AttributeError`` (which ``with_retry`` would not catch).

        Returns:
            dict: The raw stored mapping; records are not yet validated.
        """
        try:
            fs, path = _fs_for(self._uri, self._storage_options)
            if not fs.exists(path):
                return {}
            with fs.open(path, "rb") as fh:
                blob = json.loads(fh.read().decode())
            if not isinstance(blob, dict) or blob.get("schema") != _SCHEMA:
                return {}
            stored = blob.get("shards")
            return stored if isinstance(stored, dict) else {}
        except Exception:
            return {}

    def load(self) -> None:
        """Take the stat baseline, then populate the entry map from the stored object. Never raises."""
        if self._loaded or not self.enabled:
            self._loaded = True
            return
        self._loaded = True
        # Before the object is read, and therefore before the caller can read any footer.
        self._current_stats()
        stored = self._read_object()
        kept = 0
        try:
            items = list(stored.items())
        except Exception:
            return
        for name, rec in items:
            try:
                if isinstance(rec, dict) and self._entry_is_fresh(name, rec):
                    self._entries[name] = rec
                    kept += 1
            except Exception:
                continue
        if kept:
            logger.info(
                "shard-info cache: %d/%d entries valid from %s (%d shards in this source).",
                kept,
                len(stored),
                self._uri,
                len(self._files),
            )

    def get(self, shard_name: str) -> tuple[int, int] | None:
        """Return a validated ``(num_rows, num_row_groups)``, or ``None`` to fall back to the footer.

        Every hit is re-validated against the live stat baseline, whatever put it in the map.

        Args:
            shard_name: The path as the source spells it.

        Returns:
            tuple | None: The two file-intrinsic numbers (never ``shard_id``).
        """
        if not self.enabled:
            return None
        try:
            if not self._loaded:
                self.load()
            rec = self._entries.get(shard_name)
            if rec is None or not self._entry_is_fresh(shard_name, rec):
                return None
            return int(rec["num_rows"]), int(rec["num_row_groups"])
        except Exception:
            return None

    def _keep_on_disk(self, shard_name: str, rec: tp.Any) -> bool:
        """Return whether a stored record should survive this flush's merge.

        Eviction needs proof: a record is dropped when the listing shows the object with a
        different stat, or when it could never validate (bad counts, no discriminator). A record
        whose shard the listing does not cover is kept; it is refused at serve time regardless.

        Args:
            shard_name: Key as stored in the object.
            rec: The stored record.

        Returns:
            bool: ``True`` to carry it through the merge.
        """
        try:
            if not _valid_counts(rec):
                return False
            want = (rec.get("size"), rec.get("mtime"), rec.get("generation"))
            if not _has_discriminator(want):
                return False
            if self._stats is None:
                return True
            got = self._stat_of(shard_name)
            if not _has_discriminator(got):
                return True  # not listed => unproven, not stale
            return list(got) == list(want)
        except Exception:
            return True

    def record(self, shard_name: str, num_rows: int, num_row_groups: int) -> None:
        """Note a footer just read, for the incremental / end-of-process flush. Never raises.

        Args:
            shard_name: The path as the source spells it.
            num_rows: Row count from the parquet footer.
            num_row_groups: Row-group count from the parquet footer.
        """
        if not self.enabled or not self._may_write():
            return
        try:
            if self._stats is None:
                # Any stat taken now would postdate the footer read: the poisoning pairing.
                logger.debug("shard-info cache: no stat baseline for %s; not banking it.", shard_name)
                return
            stat = self._stat_of(shard_name)
            if not _has_discriminator(stat):
                return
            if isinstance(num_rows, bool) or isinstance(num_row_groups, bool):
                return
            rec = {
                "size": stat[0],
                "mtime": stat[1],
                "generation": stat[2],
                "num_rows": int(num_rows),
                "num_row_groups": int(num_row_groups),
            }
            if not _valid_counts(rec):
                return
            self._learned[shard_name] = rec
        except Exception:
            return
        if len(self._learned) >= self._flush_every:
            self.flush()

    def persisted_count(self) -> int:
        """Count this source's shards that are banked on the backend and would actually serve.

        Returns:
            int: Shards of the file list with a servable stored record; ``0`` when disabled.
        """
        if not self.enabled:
            return 0
        try:
            stored = self._read_object()
            return sum(1 for name in self._files if self._entry_is_fresh(name, stored.get(name)))
        except Exception:
            return 0

    def flush(self, *, force_writer: bool = False) -> bool:
        """Persist learned entries merged with the stored object. Never raises.

        The merged object is written to a unique staging path and moved into place. On object
        stores ``mv`` is copy + delete, so the two steps are not atomic together, but the
        destination object finalizes atomically: readers never see a half-written object, and the
        realistic concurrent-writer outcome is a lost update (a footer read later), never a wrong
        count. The served map is NOT replaced by the merged object; ``get`` keeps validating.

        Args:
            force_writer: Bypass the writer election for this and later calls on this instance.

        Returns:
            bool: Whether an object was written.
        """
        if force_writer:
            self.force_writer()
        if not self.enabled:
            return False
        if not self._may_write():
            self._learned = {}
            return False
        if not self._learned:
            return False
        try:
            merged = {name: rec for name, rec in self._read_object().items() if self._keep_on_disk(name, rec)}
            merged.update({name: rec for name, rec in self._entries.items() if self._keep_on_disk(name, rec)})
            merged.update(self._learned)
            payload = json.dumps({"schema": _SCHEMA, "shards": merged}, sort_keys=True).encode()
            fs, path = _fs_for(self._uri, self._storage_options)
            parent = path.rsplit("/", 1)[0]
            if parent and parent != path:
                try:
                    fs.makedirs(parent, exist_ok=True)
                except Exception:
                    pass
            tmp = f"{path}.tmp-{_tmp_suffix()}"
            with fs.open(tmp, "wb") as fh:
                fh.write(payload)
            try:
                fs.mv(tmp, path)
            except Exception:
                with fs.open(path, "wb") as fh:  # backends without a working move
                    fh.write(payload)
                try:
                    fs.rm(tmp)
                except Exception:
                    pass
            logger.info("shard-info cache: wrote %d entries (%d new) to %s.", len(merged), len(self._learned), self._uri)
            self._learned = {}
            return True
        except Exception as exc:
            logger.warning("shard-info cache: write skipped (%s: %s).", type(exc).__name__, exc)
            return False


def _flush_live_caches() -> None:
    """Flush every live enabled cache. Registered once per process; never raises."""
    for cache in list(_LIVE_CACHES):
        try:
            cache.flush()
        except Exception:
            continue


def _register_atexit_once() -> None:
    """Register :func:`_flush_live_caches` exactly once per process."""
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    try:
        atexit.register(_flush_live_caches)
        _ATEXIT_REGISTERED = True
    except Exception:
        pass


def cache_for_files(
    files: tp.Sequence[str],
    storage_options: dict | None = None,
    *,
    root: str | None = None,
    process_index: int | None = None,
    flush_every: int = DEFAULT_FLUSH_EVERY,
) -> ShardInfoDiskCache:
    """Construct a cache (inert when ``root`` is unset) and register it for the exit flush.

    The single process-wide ``atexit`` flush covers the tail of the walk after the last incremental
    flush. Caches are held weakly, so many sources neither register many handlers nor stay alive
    until exit. A process killed with SIGKILL runs no exit handler; the incremental flush covers it.

    Args:
        files: The source's expanded shard paths.
        storage_options: ``fsspec`` storage options for listings and object I/O.
        root: Cache root directory / URI prefix; ``None`` disables the cache.
        process_index: Writer election input; see :class:`ShardInfoDiskCache`.
        flush_every: Incremental flush interval in newly learned entries.

    Returns:
        ShardInfoDiskCache: Enabled or inert.
    """
    cache = ShardInfoDiskCache(files, storage_options, root=root, process_index=process_index, flush_every=flush_every)
    if cache.enabled:
        _register_atexit_once()
        _LIVE_CACHES.add(cache)
    return cache


def prewarm(patterns: tp.Sequence[str], root: str, storage_options: dict | None = None) -> int:
    """Read every shard's footer once and bank it so no training process pays the walk.

    Args:
        patterns: Globs to walk; each becomes its own cache object.
        root: Cache root directory / URI prefix.
        storage_options: ``fsspec`` storage options.

    Returns:
        int: Number of shards banked and servable across all patterns.

    Raises:
        ValueError: When ``root`` is empty.
    """
    from .sources.base import ParquetShardedSource

    if not root:
        raise ValueError("prewarm needs a cache root.")
    total = 0
    for pattern in patterns:
        source = ParquetShardedSource(pattern, storage_options=storage_options, metadata_cache_dir=root)
        cache = source._shard_info_disk
        cache.force_writer()  # before the walk: incremental flushes must write too
        names = list(source.shard_names)
        logger.info("prewarm %s: %d shards.", pattern, len(names))
        for i, name in enumerate(names, 1):
            source.get_shard_info(name)
            if i % 200 == 0:
                logger.info("prewarm %s: %d/%d.", pattern, i, len(names))
        cache.flush(force_writer=True)
        banked = cache.persisted_count()
        total += banked
        logger.info("prewarm %s: %d/%d shards banked and servable.", pattern, banked, len(names))
    logger.info("prewarm done: %d shards banked.", total)
    return total


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Prewarm the parquet shard-info cache.")
    ap.add_argument("--root", required=True, help="Cache root directory or URI prefix.")
    ap.add_argument("--prewarm", nargs="+", required=True, metavar="GLOB")
    args = ap.parse_args()
    prewarm(args.prewarm, args.root)
