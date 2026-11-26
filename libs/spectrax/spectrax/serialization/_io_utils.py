# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded, leak-free TensorStore checkpoint I/O helpers.

Provides:
* :func:`release_host_memory` — returns freed heap pages to the OS under both
  preloaded ``libtcmalloc`` (``MallocExtension_ReleaseFreeMemory``) and glibc
  (``malloc_trim(0)``).
* :func:`zero_cache_context` — a TensorStore :class:`~tensorstore.Context` with
  zero-byte cache pools so one-shot checkpoint reads and writes never pin
  arrays permanently in JAX's module-global 10 GB ``_TS_CONTEXT``.
* :func:`transfer_shard_to_host_no_leak` / :func:`patched_shard_transfer` —
  stages shard-to-host copies through an ephemeral ``may_alias=False``
  ``pinned_host`` buffer and deletes that buffer immediately once the host
  copy is materialized, preventing persistent host-value caching on live
  ``pinned_host`` PJRT buffers.
* :func:`plan_replica_writes` — deterministically load-balances replicated
  leaves across all replicas so every data-parallel replica shares checkpoint
  write memory and I/O equally instead of concentrating 100% on ``replica_id=0``.
* :func:`bounded_serialize_zarr` — streams per-array zarr writes under a strict
  per-process in-flight byte budget, committing each array before releasing its
  budget and reclaiming host heap pages afterwards.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import ctypes
import gc
import itertools
import os
import threading
import typing as tp
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
from jax._src import distributed as jax_distributed
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import serialization as array_ser
from jax.experimental.array_serialization import tensorstore_impl as ts_impl
from jax.sharding import SingleDeviceSharding

from spectrax._internal.logging import get_logger

from . import _fs

logger = get_logger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


DEFAULT_SAVE_CONCURRENT_GB: float = _env_float("SPECTRAX_SAVE_CONCURRENT_GB", 4.0)
DEFAULT_SAVE_IO_CONCURRENCY: int = _env_int("SPECTRAX_SAVE_IO_CONCURRENCY", 64)
DEFAULT_SAVE_COPY_CONCURRENCY: int = _env_int("SPECTRAX_SAVE_COPY_CONCURRENCY", 64)

_TRANSFER_LOCK = threading.Lock()
_TRANSFER_DEPTH = 0
_TRANSFER_SAVED: tp.Any = None
_BARRIER_COUNTER = itertools.count()

_tcmalloc_release_fn: tp.Any = None
_tcmalloc_probed = False
_glibc_trim_fn: tp.Any = None
_glibc_probed = False


def release_host_memory() -> None:
    """Run Python GC and return freed heap pages to the OS.

    Supports both preloaded ``libtcmalloc`` (where ``libc.malloc_trim(0)`` is a
    no-op) and standard glibc ``malloc``.
    """
    global _tcmalloc_release_fn, _tcmalloc_probed, _glibc_trim_fn, _glibc_probed

    gc.collect()

    if not _tcmalloc_probed:
        _tcmalloc_probed = True
        try:
            proc = ctypes.CDLL(None)
            fn = getattr(proc, "MallocExtension_ReleaseFreeMemory", None)
            if fn is not None:
                fn.argtypes = []
                fn.restype = None
                _tcmalloc_release_fn = fn
        except (OSError, AttributeError):
            _tcmalloc_release_fn = None

    if _tcmalloc_release_fn is not None:
        try:
            _tcmalloc_release_fn()
        except Exception:
            pass

    if not _glibc_probed:
        _glibc_probed = True
        try:
            libc = ctypes.CDLL("libc.so.6")
            fn = getattr(libc, "malloc_trim", None)
            if fn is not None:
                fn.argtypes = [ctypes.c_size_t]
                fn.restype = ctypes.c_int
                _glibc_trim_fn = fn
        except (OSError, AttributeError):
            _glibc_trim_fn = None

    if _glibc_trim_fn is not None:
        try:
            _glibc_trim_fn(0)
        except Exception:
            pass


def zero_cache_context_spec(
    *,
    io_concurrency: int | None = None,
    copy_concurrency: int | None = None,
) -> dict[str, dict[str, int]]:
    """Build a TensorStore context JSON spec with zero-byte cache pools."""
    return {
        "file_io_concurrency": {
            "limit": max(1, int(io_concurrency if io_concurrency is not None else DEFAULT_SAVE_IO_CONCURRENCY))
        },
        "data_copy_concurrency": {
            "limit": max(1, int(copy_concurrency if copy_concurrency is not None else DEFAULT_SAVE_COPY_CONCURRENCY))
        },
        "cache_pool": {"total_bytes_limit": 0},
        "cache_pool#remote": {"total_bytes_limit": 0},
    }


def zero_cache_context(
    *,
    io_concurrency: int | None = None,
    copy_concurrency: int | None = None,
) -> ts.Context:
    """Create a TensorStore context with zero-byte cache pools for one-shot checkpoint I/O."""
    return ts.Context(
        zero_cache_context_spec(
            io_concurrency=io_concurrency,
            copy_concurrency=copy_concurrency,
        )
    )


async def transfer_shard_to_host_no_leak(shard) -> np.ndarray:
    """Transfer a JAX shard to a host NumPy array without leaking host caches on live buffers.

    On accelerators with ``pinned_host`` memory, calling ``np.array`` or
    ``_single_device_array_to_np_array_did_copy`` directly on a live
    ``pinned_host`` shard attaches a persistent host copy inside the live PJRT
    buffer until that buffer is deleted. Staging through an ephemeral
    ``may_alias=False`` copy and explicitly deleting that temporary buffer as
    soon as the copied NumPy array is obtained keeps both live buffers and the
    pinned-host pool free of retained copies.
    """
    data = shard.data
    ephemeral = None
    if any(m.kind == "pinned_host" for m in shard.device.addressable_memories()):
        sharding = SingleDeviceSharding(shard.device, memory_kind="pinned_host")
        data = jax.device_put(data, sharding, may_alias=False)
        ephemeral = data
    else:
        data.copy_to_host_async()
    await asyncio.sleep(0)
    if hasattr(data, "_single_device_array_to_np_array_did_copy"):
        arr, did_copy = data._single_device_array_to_np_array_did_copy()
        if ephemeral is not None and did_copy:
            ephemeral.delete()
        return arr
    return np.array(data, copy=False)


@contextlib.contextmanager
def patched_shard_transfer():
    """Temporarily install :func:`transfer_shard_to_host_no_leak` on JAX's TensorStore backend."""
    global _TRANSFER_DEPTH, _TRANSFER_SAVED
    with _TRANSFER_LOCK:
        if _TRANSFER_DEPTH == 0:
            _TRANSFER_SAVED = ts_impl._transfer_shard_to_host
            ts_impl._transfer_shard_to_host = transfer_shard_to_host_no_leak
        _TRANSFER_DEPTH += 1
    try:
        yield
    finally:
        with _TRANSFER_LOCK:
            _TRANSFER_DEPTH -= 1
            if _TRANSFER_DEPTH == 0:
                ts_impl._transfer_shard_to_host = _TRANSFER_SAVED
                _TRANSFER_SAVED = None


class ByteBudget:
    """Async per-process in-flight byte limiter that does not permanently inflate on oversize items."""

    def __init__(self, max_bytes: int):
        self._max_bytes = max(1, int(max_bytes))
        self._in_flight = 0
        self._cond = asyncio.Condition()

    async def acquire(self, requested_bytes: int) -> None:
        need = max(0, int(requested_bytes))
        if need == 0:
            return
        async with self._cond:
            if need >= self._max_bytes:
                await self._cond.wait_for(lambda: self._in_flight == 0)
            else:
                await self._cond.wait_for(lambda: self._in_flight + need <= self._max_bytes)
            self._in_flight += need

    async def release(self, released_bytes: int) -> None:
        done = max(0, int(released_bytes))
        if done == 0:
            return
        async with self._cond:
            self._in_flight = max(0, self._in_flight - done)
            self._cond.notify_all()


def _index_key(idx: tuple[tp.Any, ...]) -> tuple[tp.Any, ...]:
    return tuple((s.start, s.stop, s.step) if isinstance(s, slice) else s for s in idx)


def _sharding_replica_info(
    sharding: tp.Any,
    shape: tuple[int, ...],
    cache: dict[tuple[tp.Any, tuple[int, ...]], tuple[int, dict[int, int]]],
) -> tuple[int, dict[int, int]]:
    """Return ``(n_replicas, {replica_id: lowest_writer_process_index})`` for a sharding/shape pair."""
    key = (sharding, shape)
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        dim = sharding.devices_indices_map(shape)
        counts: dict[tuple[tp.Any, ...], int] = {}
        replica_primary: dict[int, int] = {}
        for dev, idx in dim.items():
            hk = _index_key(idx)
            rid = counts.get(hk, 0)
            counts[hk] = rid + 1
            proc = int(dev.process_index)
            prev = replica_primary.get(rid)
            if prev is None or proc < prev:
                replica_primary[rid] = proc
        unique_counts = set(counts.values())
        n_rep = next(iter(unique_counts)) if len(unique_counts) == 1 else 1
        if n_rep <= 1:
            replica_primary = {0: replica_primary.get(0, 0)}
    except Exception:
        n_rep = 1
        replica_primary = {0: 0}
    res = (n_rep, replica_primary)
    cache[key] = res
    return res


def plan_replica_writes(
    arrays: Sequence[tp.Any],
    *,
    balance_replicas: bool = True,
) -> tuple[list[int], list[int | None]]:
    """Deterministically assign ``(replica_id, primary_host)`` for each array across all processes.

    Every process sees the same global array order, shapes, dtypes, and shardings, so all
    processes agree on the exact same assignment without any communication. When
    ``balance_replicas=True``, replicated arrays are greedily assigned to the least-loaded
    replica (by global byte size), spreading host memory and I/O evenly across all replicas.
    """
    pcount = jax.process_count()
    if pcount <= 1:
        return [0] * len(arrays), [None] * len(arrays)

    cache: dict[tuple[tp.Any, tuple[int, ...]], tuple[int, dict[int, int]]] = {}
    loads: dict[int, list[int]] = {}
    replica_ids: list[int] = []
    primary_hosts: list[int | None] = []

    for arr in arrays:
        if not isinstance(arr, jax.Array) or arr.is_fully_addressable:
            replica_ids.append(0)
            primary_hosts.append(0)
            continue
        sharding = getattr(arr, "sharding", None)
        if sharding is None:
            replica_ids.append(0)
            primary_hosts.append(0)
            continue
        n_rep, replica_primary = _sharding_replica_info(sharding, tuple(arr.shape), cache)
        if balance_replicas and n_rep > 1:
            bucket = loads.get(n_rep)
            if bucket is None:
                bucket = [0] * n_rep
                loads[n_rep] = bucket
            rid = min(range(n_rep), key=bucket.__getitem__)
            bucket[rid] += int(arr.size) * int(arr.dtype.itemsize)
        else:
            rid = 0
        replica_ids.append(rid)
        primary_hosts.append(replica_primary.get(rid, 0))

    return replica_ids, primary_hosts


def local_write_bytes(arr: tp.Any, replica_id: int) -> int:
    """Return the exact bytes this process will serialize for ``arr`` at ``replica_id``."""
    itemsize = int(arr.dtype.itemsize)
    if isinstance(arr, jax.Array):
        total = 0
        for shard in arr.addressable_shards:
            if shard.replica_id == replica_id:
                total += int(shard.data.size) * itemsize
        return total
    return int(arr.size) * itemsize if replica_id == 0 else 0


def effective_save_dtype(arr: tp.Any, dtype: tp.Any) -> tp.Any:
    """Return the dtype ``arr`` will have when serialized with optional floating cast ``dtype``."""
    if dtype is not None and jnp.issubdtype(arr.dtype, jnp.floating):
        return jnp.dtype(dtype)
    return arr.dtype


def _run_async(factory: Callable[[], tp.Awaitable[tp.Any]]) -> tp.Any:
    """Run an async coroutine even if an event loop is already active on the calling thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="spectrax-save") as pool:
        return pool.submit(lambda: asyncio.run(factory())).result()


def _barrier(tag: str, timeout_secs: int) -> None:
    if jax.process_count() <= 1:
        return
    key = f"spectrax:{tag}:{next(_BARRIER_COUNTER)}"
    client = jax_distributed.global_state.client
    if client is not None:
        client.wait_at_barrier(key, int(timeout_secs) * 1000)
    else:
        multihost_utils.sync_global_devices(key)


def _stream_zarr_batch(
    arrays: Sequence[tp.Any],
    specs: Sequence[dict[str, tp.Any]],
    replica_ids: Sequence[int],
    primary_hosts: Sequence[int | None],
    *,
    context: ts.Context,
    max_in_flight: int,
    release_memory: bool,
) -> None:
    pidx = jax.process_index()
    release_stride = max(4 * (1024**3), 2 * max_in_flight)

    async def _write_all() -> None:
        budget = ByteBudget(max_in_flight)
        completed_since_release = 0

        async def _write_one(arr, spec, replica_id: int, primary_host: int | None) -> None:
            nonlocal completed_since_release
            lbytes = local_write_bytes(arr, replica_id)
            if lbytes == 0 and primary_host is not None and pidx != primary_host:
                return
            await budget.acquire(lbytes)
            try:
                await ts_impl.async_serialize(
                    arr,
                    spec,
                    commit_future=None,
                    context=context,
                    primary_host=primary_host,
                    replica_id=replica_id,
                )
            finally:
                await budget.release(lbytes)
                if release_memory and lbytes > 0:
                    completed_since_release += lbytes
                    if completed_since_release >= release_stride:
                        completed_since_release = 0
                        release_host_memory()

        await asyncio.gather(
            *(
                _write_one(arr, spec, rid, ph)
                for arr, spec, rid, ph in zip(arrays, specs, replica_ids, primary_hosts, strict=True)
            )
        )

    _run_async(_write_all)


def _deterministic_cast_groups(arrays: Sequence[tp.Any], target_dtype: tp.Any, max_bytes_per_process: int) -> list[list[int]]:
    """Group array indices identically on all processes so lockstep dtype casts stay memory-bounded."""
    pcount = max(1, jax.process_count())
    budget_global = max(1, int(max_bytes_per_process)) * pcount
    out: list[list[int]] = []
    cur: list[int] = []
    cur_bytes = 0
    for i, arr in enumerate(arrays):
        n = int(arr.size) * int(effective_save_dtype(arr, target_dtype).itemsize)
        if cur and cur_bytes + n > budget_global:
            out.append(cur)
            cur, cur_bytes = [], 0
        cur.append(i)
        cur_bytes += n
    if cur:
        out.append(cur)
    return out


def bounded_serialize_zarr(
    arrays: Sequence[tp.Any],
    paths: Sequence[str],
    *,
    dtype: tp.Any = None,
    concurrent_gb: float | None = None,
    balance_replicas: bool = True,
    io_concurrency: int | None = None,
    copy_concurrency: int | None = None,
    context: ts.Context | None = None,
    on_commit_callback: Callable[[], None] | None = None,
    timeout_secs: int = 400,
    release_memory: bool = True,
) -> None:
    """Serialize arrays to per-array zarr stores with bounded host RAM and replica load balancing."""
    if len(arrays) != len(paths):
        raise ValueError(f"arrays ({len(arrays)}) and paths ({len(paths)}) length mismatch")

    pcount = jax.process_count()
    pidx = jax.process_index()

    if arrays:
        budget_gb = DEFAULT_SAVE_CONCURRENT_GB if concurrent_gb is None else float(concurrent_gb)
        max_in_flight = max(64 * 1024 * 1024, int(budget_gb * (1024**3)))

        replica_ids, planned_primaries = plan_replica_writes(arrays, balance_replicas=balance_replicas)
        ts_ctx = (
            context
            if context is not None
            else zero_cache_context(io_concurrency=io_concurrency, copy_concurrency=copy_concurrency)
        )
        specs = [array_ser.get_tensorstore_spec(p) for p in paths]
        primary_hosts = [
            (ph if (pcount > 1 and _fs.is_remote_path(p)) else None)
            for p, ph in zip(paths, planned_primaries, strict=True)
        ]

        target_dtype = jnp.dtype(dtype) if dtype is not None else None
        needs_cast = target_dtype is not None and any(
            jnp.issubdtype(a.dtype, jnp.floating) and a.dtype != target_dtype for a in arrays
        )

        with patched_shard_transfer():
            if not needs_cast:
                _stream_zarr_batch(
                    arrays,
                    specs,
                    replica_ids,
                    primary_hosts,
                    context=ts_ctx,
                    max_in_flight=max_in_flight,
                    release_memory=release_memory,
                )
            else:
                # Cast deterministically in lockstep batches across all processes so SPMD
                # casts never deadlock while still avoiding full-tree memory doubling.
                for group in _deterministic_cast_groups(arrays, target_dtype, max_in_flight):
                    batch_arrays = []
                    ephemeral_casts = []
                    for idx in group:
                        arr = arrays[idx]
                        if jnp.issubdtype(arr.dtype, jnp.floating) and arr.dtype != target_dtype:
                            casted = arr.astype(target_dtype)
                            batch_arrays.append(casted)
                            ephemeral_casts.append(casted)
                        else:
                            batch_arrays.append(arr)
                    try:
                        _stream_zarr_batch(
                            batch_arrays,
                            [specs[idx] for idx in group],
                            [replica_ids[idx] for idx in group],
                            [primary_hosts[idx] for idx in group],
                            context=ts_ctx,
                            max_in_flight=max_in_flight,
                            release_memory=release_memory,
                        )
                    finally:
                        for casted in ephemeral_casts:
                            if hasattr(casted, "delete"):
                                casted.delete()
                        del batch_arrays, ephemeral_casts
                        if release_memory:
                            release_host_memory()

        if release_memory:
            release_host_memory()

    _barrier("zarr_committed", timeout_secs=timeout_secs)
    if pidx == 0 and on_commit_callback is not None:
        on_commit_callback()
    _barrier("zarr_done", timeout_secs=timeout_secs)
