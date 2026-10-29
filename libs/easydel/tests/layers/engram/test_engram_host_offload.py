# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""GatedEngram with ``table_memory_kind="pinned_host"``: placement, bit-identical outputs, device-memory saving.

The table lives in pinned host memory (a non-trainable ``buffers`` leaf at the same ``embed.weight`` path); the lookup
runs under ``compute_on("device_host")`` and moves only the gathered rows to device memory.
"""

import gc
import time

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.engram import GatedEngram, GatedEngramConfig
from jax import numpy as jnp

ROWS = 3_000_000
HEAD_DIM = 64
COLS = 2
HIDDEN = 32
HC = 2
TABLE_BYTES = ROWS * HEAD_DIM * 4


@pytest.fixture(autouse=True)
def _require_pinned_host():
    device = jax.devices()[0]
    if "pinned_host" not in {memory.kind for memory in device.addressable_memories()}:
        pytest.skip(f"backend {device.platform!r} exposes no pinned_host memory")


@pytest.fixture(scope="module")
def weights():
    rng = np.random.default_rng(0)
    return {
        "table": rng.standard_normal((ROWS, HEAD_DIM), dtype=np.float32),
        "kv": (rng.standard_normal((COLS * HEAD_DIM, HIDDEN * (HC + 1))) * 0.1).astype(np.float32),
        "q": rng.uniform(0.5, 1.5, size=(HC, HIDDEN)).astype(np.float32),
        "k": rng.uniform(0.5, 1.5, size=(HC, HIDDEN)).astype(np.float32),
    }


def _config(kind: str) -> GatedEngramConfig:
    return GatedEngramConfig(
        hidden_size=HIDDEN, hc_mult=HC, table_rows=ROWS, num_hash_cols=COLS, head_dim=HEAD_DIM, table_memory_kind=kind
    )


def _build(kind: str, weights) -> GatedEngram:
    module = GatedEngram(
        _config(kind),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    module.kv_proj.weight.value = jnp.asarray(weights["kv"])
    module.q_weight.value = jnp.asarray(weights["q"])
    module.k_weight.value = jnp.asarray(weights["k"])
    if kind == "pinned_host":
        device = jax.devices()[0]
        host = jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host")
        module.embed.weight.value = jax.device_put(weights["table"], host)
    else:
        module.embed.weight.value = jnp.asarray(weights["table"])
    jax.block_until_ready(module.embed.weight.value)
    return module


def _inputs(batch: int, seq: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    streams = rng.normal(size=(batch, seq, HC, HIDDEN)).astype(np.float32)
    ids = rng.integers(0, ROWS, size=(batch, seq, COLS)).astype(np.int32)
    return jnp.asarray(streams), jnp.asarray(ids)


def _collections(module) -> dict[str, str]:
    _, state = spx.export(module)
    return {path: collection for collection, path, _ in state.items()}


def test_default_keeps_the_table_a_device_parameter(weights):
    module = _build("device", weights)
    assert module.table_memory_kind == "device"
    assert _collections(module)["embed.weight"] == "parameters"
    assert module.embed.weight.value.sharding.memory_kind == "device"


def test_host_table_is_a_pinned_host_buffer(weights):
    module = _build("pinned_host", weights)
    assert module.embed.weight.value.sharding.memory_kind == "pinned_host"
    collections = _collections(module)
    assert collections["embed.weight"] == "buffers"
    assert {collections[name] for name in ("kv_proj.weight", "q_weight", "k_weight")} == {"parameters"}


def test_place_table_on_host_is_idempotent_and_moves_a_device_table(weights):
    module = _build("device", weights)
    module.place_table_on_host()
    module.place_table_on_host()
    assert module.embed.weight.value.sharding.memory_kind == "pinned_host"
    streams, ids = _inputs(2, 4)
    expected = _build("device", weights)(streams, ids)
    np.testing.assert_array_equal(np.asarray(module(streams, ids)), np.asarray(expected))


@pytest.mark.parametrize("seq", [8, 1])  # training-style forward and a decode step
def test_outputs_bit_identical_to_device_table(weights, seq):
    streams, ids = _inputs(2, seq)
    device, host = _build("device", weights), _build("pinned_host", weights)
    expected = np.asarray(device(streams, ids))
    assert np.abs(expected - np.asarray(streams)).max() > 1e-4  # the engram contributes
    np.testing.assert_array_equal(np.asarray(host(streams, ids)), expected)
    np.testing.assert_array_equal(np.asarray(jax.jit(host)(streams, ids)), expected)
    np.testing.assert_array_equal(np.asarray(jax.jit(device)(streams, ids)), expected)


def test_gradient_through_the_streams_matches(weights):
    streams, ids = _inputs(2, 6)
    device, host = _build("device", weights), _build("pinned_host", weights)
    grads = [np.asarray(jax.grad(lambda s, m=m: m(s, ids).sum())(streams)) for m in (device, host)]
    assert np.isfinite(grads[1]).all()
    np.testing.assert_array_equal(grads[0], grads[1])


def _bytes_in_use() -> int:
    gc.collect()
    return int(jax.devices()[0].memory_stats()["bytes_in_use"])


def test_host_table_frees_device_memory(weights):
    base_device = _bytes_in_use()
    device = _build("device", weights)
    delta_device = _bytes_in_use() - base_device
    del device
    base_host = _bytes_in_use()
    host = _build("pinned_host", weights)
    delta_host = _bytes_in_use() - base_host
    assert host.embed.weight.value.sharding.memory_kind == "pinned_host"
    print(
        f"[engram-offload] table {TABLE_BYTES / 2**20:.0f} MiB; device-table module +{delta_device / 2**20:.1f} MiB, "
        f"host-table module +{delta_host / 2**20:.1f} MiB on device 0"
    )
    assert delta_device - delta_host >= 0.8 * TABLE_BYTES, (delta_device, delta_host, TABLE_BYTES)


def _median_ms(fn, *args, repeats: int = 30) -> float:
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return float(np.median(times)) * 1e3


def test_lookup_timing_report(weights):
    streams, ids = _inputs(8, 512)
    timings = {kind: _median_ms(jax.jit(_build(kind, weights)), streams, ids) for kind in ("device", "pinned_host")}
    print(
        f"[engram-offload] jitted forward, 8x512 tokens: device table {timings['device']:.3f} ms, "
        f"pinned_host table {timings['pinned_host']:.3f} ms"
    )
    assert all(np.isfinite(value) and value > 0 for value in timings.values())
