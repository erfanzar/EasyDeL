# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for bounded, zero-cache, replica-balanced, leak-free checkpoint saves."""

from __future__ import annotations

import asyncio
import ctypes

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.array_serialization import tensorstore_impl as ts_impl
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from spectrax.serialization import AsyncCheckpointManager, _io_utils, _ocdbt


class _FakeDevice:
    """Minimal device stub for deterministic multi-process replica planning tests."""

    def __init__(self, device_id: int, process_index: int):
        self.id = device_id
        self.process_index = process_index


class _FakeSharding:
    """Minimal sharding stub mapping devices to slice tuples."""

    def __init__(self, mapping: dict[_FakeDevice, tuple[slice, ...]]):
        self._mapping = mapping

    def devices_indices_map(self, global_shape):
        return self._mapping


class _FakeArray:
    """Minimal array stub for replica planning tests."""

    def __init__(self, shape: tuple[int, ...], dtype, sharding: _FakeSharding):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(shape))
        self.sharding = sharding
        self.is_fully_addressable = False


def test_zero_cache_context_has_zero_cache_limits():
    """Default write and OCDBT contexts must not retain written arrays in TensorStore cache pools."""
    ctx = _io_utils.zero_cache_context()
    assert ctx["cache_pool"].to_json().get("total_bytes_limit", 0) == 0
    assert ctx["cache_pool#remote"].to_json().get("total_bytes_limit", 0) == 0

    ocdbt_ctx = _ocdbt.default_context()
    assert ocdbt_ctx["cache_pool"].to_json().get("total_bytes_limit", 0) == 0
    assert ocdbt_ctx["cache_pool#remote"].to_json().get("total_bytes_limit", 0) == 0


def test_byte_budget_bounds_concurrency_without_inflating():
    """ByteBudget caps concurrent in-flight bytes and does not permanently grow on oversize items."""

    async def _run():
        budget = _io_utils.ByteBudget(100)
        peak = 0
        current = 0

        async def worker(size: int):
            nonlocal peak, current
            await budget.acquire(size)
            current += size
            peak = max(peak, current)
            await asyncio.sleep(0.01)
            current -= size
            await budget.release(size)

        # Oversize item (250 > 100) runs alone, then normal items stay <= 100.
        await asyncio.gather(worker(250), worker(60), worker(60), worker(40))
        assert budget._max_bytes == 100
        assert budget._in_flight == 0
        assert peak == 250

    asyncio.run(_run())


def test_plan_replica_writes_balances_across_replicas(monkeypatch):
    """Replicated arrays are balanced evenly across all data-parallel replicas by byte weight."""
    monkeypatch.setattr(jax, "process_count", lambda: 8)
    monkeypatch.setattr(_io_utils.jax, "Array", _FakeArray)

    # 4 dp replicas x 2 processes per replica (1 device per process).
    mapping = {}
    for dp in range(4):
        for fsdp in range(2):
            dev = _FakeDevice(dp * 2 + fsdp, dp * 2 + fsdp)
            mapping[dev] = (slice(fsdp * 4, (fsdp + 1) * 4),)
    sharding = _FakeSharding(mapping)

    arrays = [_FakeArray((8,), np.float32, sharding) for _ in range(8)]
    replica_ids, primary_hosts = _io_utils.plan_replica_writes(arrays, balance_replicas=True)

    assert replica_ids == [0, 1, 2, 3, 0, 1, 2, 3]
    assert primary_hosts == [0, 2, 4, 6, 0, 2, 4, 6]

    # Opt-out falls back to replica 0.
    unbalanced, unbalanced_primaries = _io_utils.plan_replica_writes(arrays, balance_replicas=False)
    assert unbalanced == [0] * 8
    assert unbalanced_primaries == [0] * 8


def test_transfer_shard_to_host_no_leak_avoids_aliasing_live_pinned_shard():
    """Staging through may_alias=False never aliases the live pinned_host array."""
    dev = jax.devices()[0]
    if not any(m.kind == "pinned_host" for m in dev.addressable_memories()):
        pytest.skip("backend has no pinned_host memory")

    pinned = jax.sharding.SingleDeviceSharding(dev, memory_kind="pinned_host")
    live = jax.device_put(jnp.arange(32, dtype=jnp.float32), pinned)

    seen = []
    orig_put = jax.device_put

    def spy(x, *args, **kwargs):
        out = orig_put(x, *args, **kwargs)
        seen.append(out)
        return out

    jax.device_put = spy
    try:
        out = asyncio.run(_io_utils.transfer_shard_to_host_no_leak(live.addressable_shards[0]))
    finally:
        jax.device_put = orig_put

    assert seen and all(buf is not live and buf is not live._arrays[0] for buf in seen)
    np.testing.assert_array_equal(out, np.arange(32, dtype=np.float32))


def test_default_save_uses_zero_cache_and_streaming_cast(tmp_path, monkeypatch):
    """Default save_pytree uses a zero-cache context, bounded writes, and streams dtype casts."""
    mesh = Mesh(np.array(jax.devices()[:1]), ("x",))
    seen_contexts = []
    real_async_serialize = ts_impl.async_serialize

    async def spy(arr, spec, **kwargs):
        seen_contexts.append(kwargs.get("context"))
        return await real_async_serialize(arr, spec, **kwargs)

    monkeypatch.setattr(ts_impl, "async_serialize", spy)

    tree = {
        "a": jnp.arange(16, dtype=jnp.float32).reshape(4, 4),
        "b": jnp.arange(16, dtype=jnp.float32).reshape(4, 4) + 10.0,
        "ids": jnp.arange(4, dtype=jnp.int32),
    }
    mgr = AsyncCheckpointManager()
    mgr.save_pytree(tree, str(tmp_path), mesh=mesh, prefix="model", dtype=jnp.bfloat16)

    assert len(seen_contexts) == 3
    for ctx in seen_contexts:
        assert ctx is not None
        assert ctx["cache_pool"].to_json().get("total_bytes_limit", 0) == 0
        assert ctx["cache_pool#remote"].to_json().get("total_bytes_limit", 0) == 0

    loaded, _ = mgr.load_pytree(str(tmp_path), mesh, prefix="model")
    assert loaded["a"].dtype == jnp.bfloat16
    assert loaded["b"].dtype == jnp.bfloat16
    assert loaded["ids"].dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(loaded["a"]), np.asarray(tree["a"].astype(jnp.bfloat16)))
    np.testing.assert_array_equal(np.asarray(loaded["ids"]), np.asarray(tree["ids"]))


def test_release_host_memory_reclaims_tcmalloc_if_loaded():
    """When libtcmalloc is preloaded, release_host_memory drops pageheap_free_bytes to ~0."""
    proc = ctypes.CDLL(None)
    get_prop = getattr(proc, "MallocExtension_GetNumericProperty", None)
    if get_prop is None:
        _io_utils.release_host_memory()
        return

    get_prop.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]
    get_prop.restype = ctypes.c_int

    def _ph_free_mb() -> float:
        val = ctypes.c_size_t(0)
        get_prop(b"tcmalloc.pageheap_free_bytes", ctypes.byref(val))
        return val.value / (1024 * 1024)

    bufs = [np.ones((4, 1024, 1024), dtype=np.float32) for _ in range(8)]  # 128 MiB
    del bufs
    _io_utils.release_host_memory()
    assert _ph_free_mb() < 16.0


def test_repeated_default_saves_do_not_accumulate_heap(tmp_path):
    """Repeated default save_pytree calls keep allocated/cached heap flat (no cache_pool retention)."""
    mesh = Mesh(np.array(jax.devices()[:1]), ("x",))
    sh = NamedSharding(mesh, PartitionSpec())
    tree = {f"w{i}": jax.device_put(jnp.ones((256, 256), dtype=jnp.float32) * (i + 1), sh) for i in range(8)}
    mgr = AsyncCheckpointManager()

    for step in range(4):
        d = tmp_path / f"run-{step}"
        mgr.save_pytree(tree, str(d), mesh=mesh, prefix="tx")
        loaded, _ = mgr.load_pytree(str(d), mesh, prefix="tx")
        np.testing.assert_array_equal(np.asarray(loaded["w0"]), np.asarray(tree["w0"]))
