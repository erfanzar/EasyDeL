# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OCDBT checkpoint layout: one coalesced TensorStore kvstore per tree prefix.

The default layout writes one zarr store per array, i.e. one object per device shard. The OCDBT
layout keeps the same relative array paths as entries of a single B+tree store per prefix, so the
object count tracks bytes rather than ``arrays x shards``::

    {root}/{prefix}.ocdbt/manifest.ocdbt
    {root}/{prefix}.ocdbt/d/...                   # data files
    {root}/{prefix}.ocdbt/ocdbt.process_{i}/...   # multi-process staging stores

Every process writes its arrays inside one atomic ``ts.Transaction`` that is committed before the
call returns. Multi-process saves stage into per-process stores; after a cross-process barrier
process 0 merges them into the prefix store. The merged manifest references the staging data
files, so staging directories are part of the store and are only removed with the checkpoint.

The transaction keeps this process's whole payload resident in host memory until the commit.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import time
import typing as tp
from dataclasses import dataclass

import jax
import tensorstore as ts
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import serialization as array_serialization
from jax.experimental.array_serialization import tensorstore_impl as ts_impl

from spectrax._internal.logging import get_logger

from . import _fs

logger = get_logger(__name__)

STORE_SUFFIX = ".ocdbt"
STAGE_DIR_FORMAT = "ocdbt.process_{}"
MANIFEST_NAME = "manifest.ocdbt"
DEFAULT_DATA_FILE_MB = 64
_UNPREFIXED_STORE = "_root"
_FALLBACK_CONTEXT_SPEC: dict[str, tp.Any] = {
    "file_io_concurrency": {"limit": 128},
    "data_copy_concurrency": {"limit": 128},
    "cache_pool": {"total_bytes_limit": 10_000_000_000},
    "cache_pool#remote": {"total_bytes_limit": 10_000_000_000},
}
_fallback_context: ts.Context | None = None

_T = tp.TypeVar("_T")

__all__ = [
    "DEFAULT_DATA_FILE_MB",
    "OcdbtSaveOptions",
    "backend_for_prefix",
    "default_context",
    "finalize",
    "manifest_exists",
    "read_spec",
    "record_backend",
    "run_sync",
    "save_context",
    "save_prefix",
    "serialize_arrays",
    "stage_dir",
    "store_dir",
    "write_specs",
]


@dataclass(frozen=True)
class OcdbtSaveOptions:
    """Write-side tuning for the OCDBT layout.

    Attributes:
        data_file_mb: Target size of each OCDBT data file in MiB. Smaller files upload with more
            parallelism; larger files produce fewer objects.
        io_concurrency: Optional TensorStore ``file_io_concurrency`` limit for the save.
        copy_concurrency: Optional TensorStore ``data_copy_concurrency`` limit for the save.
        gcs_concurrency: Optional TensorStore ``gcs_request_concurrency`` limit for the save.
    """

    data_file_mb: int = DEFAULT_DATA_FILE_MB
    io_concurrency: int | None = None
    copy_concurrency: int | None = None
    gcs_concurrency: int | None = None

    def __post_init__(self) -> None:
        """Validate that every set field is a positive integer.

        Raises:
            ValueError: If a field is not a positive ``int``.
        """
        for name in ("data_file_mb", "io_concurrency", "copy_concurrency", "gcs_concurrency"):
            value = getattr(self, name)
            if value is None and name != "data_file_mb":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"OcdbtSaveOptions.{name} must be a positive int, got {value!r}")

    @property
    def target_data_file_size(self) -> int:
        """Target OCDBT data-file size in bytes."""
        return self.data_file_mb * 1024 * 1024

    def context_overrides(self) -> dict[str, dict[str, int]]:
        """Return the TensorStore context resources this options object overrides.

        Returns:
            Mapping of resource name to resource config; empty when no concurrency is set.
        """
        cfg: dict[str, dict[str, int]] = {}
        if self.io_concurrency is not None:
            cfg["file_io_concurrency"] = {"limit": self.io_concurrency}
        if self.copy_concurrency is not None:
            cfg["data_copy_concurrency"] = {"limit": self.copy_concurrency}
        if self.gcs_concurrency is not None:
            cfg["gcs_request_concurrency"] = {"limit": self.gcs_concurrency}
        return cfg


def run_sync(factory: tp.Callable[[], tp.Awaitable[_T]]) -> _T:
    """Run a coroutine to completion from synchronous code.

    ``asyncio.run`` refuses to start while the calling thread already runs an event loop (a
    notebook, an async server). In that case the coroutine runs on a fresh loop in a worker
    thread and this call blocks until it finishes.

    Args:
        factory: Zero-argument callable returning the coroutine to run.

    Returns:
        The coroutine's result.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="spectrax-ocdbt") as pool:
        return pool.submit(lambda: asyncio.run(factory())).result()


def default_context() -> ts.Context:
    """Return JAX's module-level TensorStore context, or an equivalent fallback.

    Returns:
        The shared :class:`tensorstore.Context` used by JAX array serialization.
    """
    global _fallback_context
    for context in (getattr(array_serialization, "TS_CONTEXT", None), getattr(ts_impl, "_TS_CONTEXT", None)):
        if isinstance(context, ts.Context):
            return context
    if _fallback_context is None:
        logger.warning("JAX no longer exposes its TensorStore context; OCDBT saves use spectrax's fallback context.")
        _fallback_context = ts.Context(_FALLBACK_CONTEXT_SPEC)
    return _fallback_context


def _context_json(context: ts.Context) -> dict[str, tp.Any]:
    """Return a context's resource config as JSON, falling back to the known JAX defaults."""
    try:
        return dict(context.spec.to_json())
    except Exception as exc:
        logger.warning(f"Could not read TensorStore context config ({exc}); using spectrax's fallback config.")
        return dict(_FALLBACK_CONTEXT_SPEC)


def save_context(options: OcdbtSaveOptions | None = None) -> ts.Context:
    """Build the single TensorStore context shared by every array write of one save.

    Overrides are layered on top of JAX's context config: a fresh ``ts.Context`` does not inherit
    JAX's resources for keys it omits, so building it from the overrides alone would silently drop
    JAX's concurrency limits and cache pools.

    Args:
        options: Write options; ``None`` or options without concurrency overrides reuse JAX's context.

    Returns:
        One context instance to pass to every write of the save.
    """
    base = default_context()
    overrides = (options or OcdbtSaveOptions()).context_overrides()
    if not overrides:
        return base
    merged = _context_json(base)
    merged.update(overrides)
    return ts.Context(merged)


def store_dir(root: str, prefix: str | None) -> str:
    """Return the OCDBT store directory for a checkpoint root and tree prefix.

    Args:
        root: Checkpoint directory.
        prefix: Tree prefix (``"model"``, ``"tx"``), or ``None`` for the unprefixed tree.

    Returns:
        Path of the prefix's OCDBT store.
    """
    return _fs.joinpath(str(root), f"{prefix or _UNPREFIXED_STORE}{STORE_SUFFIX}")


def stage_dir(root: str, prefix: str | None, process_idx: int) -> str:
    """Return the per-process staging store path, nested inside :func:`store_dir`.

    Args:
        root: Checkpoint directory.
        prefix: Tree prefix, or ``None``.
        process_idx: JAX process index owning the staging store.

    Returns:
        Path of the staging store.
    """
    return _fs.joinpath(store_dir(root, prefix), STAGE_DIR_FORMAT.format(int(process_idx)))


def manifest_exists(root: str, prefix: str | None) -> bool:
    """Return whether the prefix's OCDBT store has a manifest.

    Args:
        root: Checkpoint directory.
        prefix: Tree prefix, or ``None``.

    Returns:
        ``True`` when ``manifest.ocdbt`` exists in the prefix store.
    """
    return _fs.exists(_fs.joinpath(store_dir(root, prefix), MANIFEST_NAME), refresh=True)


def _base_kvstore(path: str) -> dict[str, tp.Any]:
    """Build the TensorStore kvstore spec of the directory holding an OCDBT store.

    Args:
        path: Store directory: a local path, ``file://``, ``gs://`` or ``s3://`` URL.

    Returns:
        A ``file``, ``gcs`` or ``s3`` kvstore spec.

    Raises:
        ValueError: For other URL schemes or a URL without a bucket.
    """
    path = str(path)
    for scheme, driver in (("gs://", "gcs"), ("s3://", "s3")):
        if path.startswith(scheme):
            bucket, _, key = path[len(scheme) :].partition("/")
            if not bucket:
                raise ValueError(f"OCDBT store path needs a bucket: {path!r}")
            return {"driver": driver, "bucket": bucket, "path": key.rstrip("/")}
    if path.startswith("file://"):
        path = path[len("file://") :]
    elif _fs.is_remote_path(path):
        raise ValueError(f"The OCDBT layout supports local, gs:// and s3:// paths; got {path!r}")
    return {"driver": "file", "path": os.path.abspath(path)}


def _ocdbt_kvstore(store: str, entry_key: str | None = None, data_file_bytes: int | None = None) -> dict[str, tp.Any]:
    """Build an OCDBT kvstore spec over a store directory.

    Args:
        store: OCDBT store directory.
        entry_key: Optional key inside the store (an array's relative path).
        data_file_bytes: Optional write-side ``target_data_file_size``.

    Returns:
        OCDBT kvstore spec.
    """
    kv: dict[str, tp.Any] = {"driver": "ocdbt", "base": _base_kvstore(store)}
    if entry_key is not None:
        kv["path"] = entry_key
    if data_file_bytes is not None:
        kv["target_data_file_size"] = int(data_file_bytes)
    return kv


def write_specs(
    root: str,
    prefix: str | None,
    relpaths: tp.Sequence[str],
    *,
    options: OcdbtSaveOptions | None = None,
    process_idx: int | None = None,
    process_count: int | None = None,
) -> list[dict[str, tp.Any]]:
    """Build zarr3-over-OCDBT write specs for a prefix's arrays.

    Zarr metadata is left out; ``async_serialize`` derives it from each array's sharding.

    Args:
        root: Checkpoint directory.
        prefix: Tree prefix, or ``None``.
        relpaths: Array paths relative to *root*, used as OCDBT entry keys.
        options: Write options supplying the data-file size.
        process_idx: Overrides ``jax.process_index()``.
        process_count: Overrides ``jax.process_count()``.

    Returns:
        One TensorStore spec per relative path.
    """
    options = options or OcdbtSaveOptions()
    count = jax.process_count() if process_count is None else int(process_count)
    index = jax.process_index() if process_idx is None else int(process_idx)
    store = store_dir(root, prefix) if count <= 1 else stage_dir(root, prefix, index)
    return [
        {"driver": "zarr3", "kvstore": _ocdbt_kvstore(store, rel, options.target_data_file_size)} for rel in relpaths
    ]


def read_spec(root: str, prefix: str | None, relpath: str) -> dict[str, tp.Any]:
    """Build the read spec of one array in an OCDBT checkpoint.

    Args:
        root: Checkpoint directory.
        prefix: Tree prefix, or ``None``.
        relpath: Array path recorded in ``tensorstore_index.json``.

    Returns:
        TensorStore spec dictionary.
    """
    return {"driver": "zarr3", "kvstore": _ocdbt_kvstore(store_dir(root, prefix), relpath)}


def serialize_arrays(
    arrays: tp.Sequence[tp.Any],
    specs: tp.Sequence[dict[str, tp.Any]],
    *,
    context: ts.Context | None = None,
) -> None:
    """Write arrays under one atomic transaction and commit it before returning.

    ``GlobalAsyncCheckpointManager.serialize`` forwards a transaction but never commits it, so the
    writes are issued here directly.

    Args:
        arrays: Arrays to write, positionally matching *specs*.
        specs: Write specs from :func:`write_specs`.
        context: Shared TensorStore context; defaults to :func:`default_context`.

    Raises:
        ValueError: If *arrays* and *specs* differ in length.
    """
    if len(arrays) != len(specs):
        raise ValueError(f"arrays ({len(arrays)}) and specs ({len(specs)}) length mismatch")
    if not arrays:
        return
    context = context if context is not None else default_context()
    transaction = ts.Transaction(atomic=True)

    async def _write() -> None:
        await asyncio.gather(
            *[
                ts_impl.async_serialize(array, dict(spec), context=context, transaction=transaction)
                for array, spec in zip(arrays, specs, strict=True)
            ]
        )
        await transaction.commit_async()

    started = time.monotonic()
    try:
        run_sync(_write)
    except BaseException:
        if not transaction.commit_started:
            transaction.abort()
        raise
    logger.debug(f"OCDBT wrote {len(arrays)} arrays in {time.monotonic() - started:.2f}s")


def finalize(
    root: str,
    prefix: str | None,
    *,
    process_count: int | None = None,
    context: ts.Context | None = None,
) -> None:
    """Merge the per-process staging stores into the prefix store.

    No-op for single-process saves. Call on one process only, after every process has committed
    its staging store (see :func:`save_prefix`). Staging stores are kept: the merged manifest
    references their data files.

    Args:
        root: Checkpoint directory.
        prefix: Tree prefix, or ``None``.
        process_count: Overrides ``jax.process_count()``.
        context: TensorStore context; defaults to :func:`default_context`.

    Raises:
        RuntimeError: If the installed TensorStore cannot copy kvstore ranges.
    """
    count = jax.process_count() if process_count is None else int(process_count)
    if count <= 1:
        return
    if not hasattr(ts.KvStore, "experimental_copy_range_to"):
        raise RuntimeError("Merging multi-process OCDBT saves needs tensorstore with KvStore.experimental_copy_range_to")
    context = context if context is not None else default_context()
    combined_spec = _ocdbt_kvstore(store_dir(root, prefix))
    stage_specs = [_ocdbt_kvstore(stage_dir(root, prefix, i)) for i in range(count)]

    async def _merge() -> None:
        combined, stages = await asyncio.gather(
            ts.KvStore.open(combined_spec, context=context),
            asyncio.gather(*[ts.KvStore.open(spec, context=context) for spec in stage_specs]),
        )
        transaction = ts.Transaction()
        await asyncio.gather(
            *[stage.experimental_copy_range_to(combined.with_transaction(transaction)) for stage in stages]
        )
        await transaction.commit_async()

    run_sync(_merge)
    logger.info(f"Merged {count} OCDBT staging stores for prefix={prefix!r} under {root}")


def save_prefix(
    root: str,
    prefix: str | None,
    relpaths: tp.Sequence[str],
    arrays: tp.Sequence[tp.Any],
    *,
    options: OcdbtSaveOptions | None = None,
) -> None:
    """Write a prefix's arrays in the OCDBT layout and, multi-process, merge the staging stores.

    An existing store for the prefix is removed first. Returns on every process only once the
    merged store is readable: a barrier after each process's commit gates the merge on process 0,
    and a second barrier gates the return. Multi-process saves need a store shared by all hosts
    (``gs://``, ``s3://`` or a shared filesystem), since process 0 reads every staging store.

    Args:
        root: Checkpoint directory.
        prefix: Tree prefix, or ``None``.
        relpaths: Array paths relative to *root*.
        arrays: Arrays positionally matching *relpaths*.
        options: Write options.

    Raises:
        ValueError: If *relpaths* and *arrays* differ in length.
    """
    if len(relpaths) != len(arrays):
        raise ValueError(f"relpaths ({len(relpaths)}) and arrays ({len(arrays)}) length mismatch")
    count = jax.process_count()
    tag = prefix or _UNPREFIXED_STORE
    store = store_dir(root, prefix)
    # A re-save replaces the whole store, staging stores included, so no stale entries survive.
    if jax.process_index() == 0 and _fs.exists(store, refresh=True):
        _fs.rm(store, recursive=True)
    if count > 1:
        multihost_utils.sync_global_devices(f"spectrax:ocdbt:cleared:{tag}")
    context = save_context(options)
    specs = write_specs(root, prefix, relpaths, options=options, process_count=count)
    serialize_arrays(arrays, specs, context=context)
    if count > 1:
        multihost_utils.sync_global_devices(f"spectrax:ocdbt:committed:{tag}")
        if jax.process_index() == 0:
            finalize(root, prefix, process_count=count, context=context)
        multihost_utils.sync_global_devices(f"spectrax:ocdbt:merged:{tag}")


def record_backend(index_data: dict, prefix: str | None, ocdbt: bool) -> None:
    """Record (or clear) a prefix's OCDBT backend in ``tensorstore_index.json`` data.

    Args:
        index_data: Parsed index dictionary, mutated in place.
        prefix: Tree prefix, or ``None``.
        ocdbt: Whether the prefix was just written in the OCDBT layout.
    """
    name = prefix or _UNPREFIXED_STORE
    if ocdbt:
        index_data.setdefault("backends", {})[name] = {"backend": "ocdbt", "store": f"{name}{STORE_SUFFIX}"}
        return
    backends = index_data.get("backends")
    if isinstance(backends, dict):
        backends.pop(name, None)
        if not backends:
            index_data.pop("backends", None)


def backend_for_prefix(index_data: dict | None, prefix: str | None, default: str = "tensorstore") -> str:
    """Return the storage backend recorded for a prefix.

    Args:
        index_data: Parsed ``tensorstore_index.json`` contents, or ``None``.
        prefix: Tree prefix, or ``None``.
        default: Backend reported when the index records none for the prefix.

    Returns:
        ``"ocdbt"`` or ``"tensorstore"`` (per-array zarr).
    """
    backends = (index_data or {}).get("backends") or {}
    entry = backends.get(prefix or _UNPREFIXED_STORE) if isinstance(backends, dict) else None
    if isinstance(entry, dict) and entry.get("backend"):
        return str(entry["backend"])
    return default
