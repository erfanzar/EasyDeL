# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""GatedEngram ``table_dtype="fp8"``: fp8 e4m3 rows + e8m0 byte scales, row-sharded, optionally in pinned host memory.

Computation tests (TPU): lookup equals a NumPy dequant reference exactly (eager, jit, and export/bind under jit with
replicated or batch-sharded ids), the host table lives in host RAM at its unpadded size and costs no device memory,
and device tables are row-sharded and padded to the shard count.
"""

import gc
import tempfile
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.engram import GatedEngram, GatedEngramConfig, engram_table_padded_rows, load_engram_tables
from easydel.layers.engram._layer import _dequantize_rows
from jax import numpy as jnp

ROWS = 1_000_003  # not divisible by the device count: exercises row padding
HEAD_DIM = 64
COLS = 3
HIDDEN = 32
HC = 2


@pytest.fixture(autouse=True)
def _require_pinned_host():
    device = jax.devices()[0]
    if "pinned_host" not in {memory.kind for memory in device.addressable_memories()}:
        pytest.skip(f"backend {device.platform!r} exposes no pinned_host memory")


@pytest.fixture(scope="module")
def tables():
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((ROWS, HEAD_DIM), dtype=np.float32).astype(ml_dtypes.float8_e4m3fn)
    scale = rng.integers(120, 135, size=(ROWS, HEAD_DIM // 32), dtype=np.uint8)
    return weight, scale


def _dequant(weight: np.ndarray, scale: np.ndarray, ids: np.ndarray) -> np.ndarray:
    rows = weight[ids].astype(np.float32).reshape(*ids.shape, HEAD_DIM // 32, 32)
    factor = (2.0 ** (scale[ids].astype(np.int32) - 127)).astype(np.float32)
    return (rows * factor[..., None]).reshape(*ids.shape, HEAD_DIM)


def _config(kind: str, dtype: str) -> GatedEngramConfig:
    return GatedEngramConfig(
        hidden_size=HIDDEN,
        hc_mult=HC,
        table_rows=ROWS,
        num_hash_cols=COLS,
        head_dim=HEAD_DIM,
        table_memory_kind=kind,
        table_dtype=dtype,
    )


def _module(kind: str, dtype: str, kv: np.ndarray) -> GatedEngram:
    module = GatedEngram(
        _config(kind, dtype),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    module.kv_proj.weight.value = jnp.asarray(kv)
    return module


def _ids(batch: int, seq: int) -> np.ndarray:
    return np.random.default_rng(3).integers(0, ROWS, size=(batch, seq, COLS)).astype(np.int32)


@pytest.mark.parametrize("kind", ["device", "pinned_host"])
@pytest.mark.parametrize("seq", [8, 1])
def test_lookup_equals_numpy_dequant_reference_exactly(tables, kind, seq):
    weight, scale = tables
    kv = np.random.default_rng(1).standard_normal((COLS * HEAD_DIM, HIDDEN * (HC + 1))).astype(np.float32) * 0.1
    module = _module(kind, "fp8", kv)
    module.set_table(weight, scale)
    ids = _ids(2, seq)
    expected = _dequant(weight, scale, ids)
    np.testing.assert_array_equal(np.asarray(module._lookup(jnp.asarray(ids))), expected)
    np.testing.assert_array_equal(np.asarray(jax.jit(module._lookup)(jnp.asarray(ids))), expected)

    # the full layer matches a param-mode layer holding the dequantized table
    streams = jnp.asarray(np.random.default_rng(2).normal(size=(2, seq, HC, HIDDEN)).astype(np.float32))
    reference = _module("device", "param", kv)
    reference.embed.weight.value = jnp.asarray(_dequant(weight, scale, np.arange(ROWS)[None]).reshape(ROWS, HEAD_DIM))
    np.testing.assert_array_equal(
        np.asarray(module(streams, jnp.asarray(ids))), np.asarray(reference(streams, jnp.asarray(ids)))
    )


@pytest.mark.parametrize("seq", [8, 1])
def test_host_gathered_rows_forward_equals_in_program_lookup(tables, seq):
    """Serving path: rows gathered on the host (``gather_host_rows``) and passed as ``gathered`` give the same output
    as the in-program ``pure_callback`` lookup, bit for bit, and the gathered bytes are the table's own rows."""
    weight, scale = tables
    kv = np.random.default_rng(1).standard_normal((COLS * HEAD_DIM, HIDDEN * (HC + 1))).astype(np.float32) * 0.1
    module = _module("pinned_host", "fp8", kv)
    module.set_table(weight, scale)
    assert module.uses_host_table
    ids = _ids(2, seq)
    rows, scales = module.gather_host_rows(ids)
    assert rows.dtype == np.uint8 and rows.shape == (2, seq, COLS, HEAD_DIM)
    assert scales.dtype == np.uint8 and scales.shape == (2, seq, COLS, HEAD_DIM // 32)
    np.testing.assert_array_equal(rows, weight.view(np.uint8)[ids])
    np.testing.assert_array_equal(scales, scale[ids])

    # The device-side dequantization of the host-gathered bytes is bit-exact against the NumPy reference.
    fp8 = jax.lax.bitcast_convert_type(jnp.asarray(rows), jnp.float8_e4m3fn)
    dequant = jax.jit(_dequantize_rows)(fp8, jnp.asarray(scales))
    np.testing.assert_array_equal(np.asarray(dequant), _dequant(weight, scale, ids))

    # The full forward (both sides compiled) matches the in-program lookup; the projection/gate math is shared, so
    # only XLA fusion order can differ (measured <= 7.6e-6 abs at |x| ~ 1e2 for seq=1).
    streams = jnp.asarray(np.random.default_rng(2).normal(size=(2, seq, HC, HIDDEN)).astype(np.float32))
    expected = jax.jit(lambda s, i: module(s, i))(streams, jnp.asarray(ids))
    got = jax.jit(lambda s, r, c: module(s, gathered=(r, c)))(streams, jnp.asarray(rows), jnp.asarray(scales))
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=2e-5, atol=1e-5)
    with pytest.raises(ValueError, match="hash_ids or gathered"):
        module(streams)


def test_gather_host_rows_rejects_device_tables(tables):
    weight, scale = tables
    module = _module("device", "fp8", np.zeros((COLS * HEAD_DIM, HIDDEN * (HC + 1)), np.float32))
    module.set_table(weight, scale)
    assert not module.uses_host_table
    with pytest.raises(ValueError, match="pinned_host"):
        module.gather_host_rows(_ids(1, 1))


def test_host_table_is_sharded_pinned_host_buffers_with_no_device_cost(tables):
    weight, scale = tables
    kv = np.zeros((COLS * HEAD_DIM, HIDDEN * (HC + 1)), np.float32)
    gc.collect()
    before = int(jax.devices()[0].memory_stats()["bytes_in_use"])
    module = _module("pinned_host", "fp8", kv)
    module.set_table(weight, scale)
    jax.block_until_ready((module.embed.weight.value, module.embed.scale.value))
    gc.collect()
    delta = int(jax.devices()[0].memory_stats()["bytes_in_use"]) - before

    # the table lives in host RAM (unpadded); the module keeps only 1-row placeholder buffers
    assert module.embed.weight.value.shape == (1, HEAD_DIM)
    assert module.embed.scale.value.shape == (1, HEAD_DIM // 32)
    stored_w, stored_s = module.fp8_table_numpy()
    assert stored_w.shape == (ROWS, HEAD_DIM) and stored_s.shape == (ROWS, HEAD_DIM // 32)
    np.testing.assert_array_equal(stored_w.view(np.uint8), weight.view(np.uint8))
    np.testing.assert_array_equal(stored_s, scale)
    _, state = spx.export(module)
    kinds = {path: collection for collection, path, _ in state.items()}
    assert kinds["embed.weight"] == kinds["embed.scale"] == "buffers"
    assert delta < 1 << 20, delta  # only the kv_proj leaf lands on device; the ~64 MiB table does not


def test_padding_rows_are_zero_weight_and_unit_scale(tables):
    weight, scale = tables
    # device tables are row-sharded and padded to the shard count; host tables are unpadded (see the test above)
    module = _module("device", "fp8", np.zeros((COLS * HEAD_DIM, HIDDEN * (HC + 1)), np.float32))
    module.set_table(weight, scale)
    padded = engram_table_padded_rows(ROWS, jax.device_count())
    if padded == ROWS:
        pytest.skip("device count divides the row count")
    stored_w, stored_s = module.fp8_table_numpy()
    assert stored_w.shape == (padded, HEAD_DIM) and stored_s.shape == (padded, HEAD_DIM // 32)
    np.testing.assert_array_equal(stored_w[:ROWS].view(np.uint8), weight.view(np.uint8))
    np.testing.assert_array_equal(stored_s[:ROWS], scale)
    assert (stored_w[ROWS:].astype(np.float32) == 0).all() and (stored_s[ROWS:] == 1).all()


def test_default_is_unchanged_param_table():
    assert _config("device", "param").table_dtype == "param"
    module = _module("device", "param", np.zeros((COLS * HEAD_DIM, HIDDEN * (HC + 1)), np.float32))
    _, state = spx.export(module)
    assert {path: collection for collection, path, _ in state.items()}["embed.weight"] == "parameters"
    assert module.embed.weight.value.shape == (ROWS, HEAD_DIM)


def _write_synthetic_checkpoint(root: Path, weight: np.ndarray, scale: np.ndarray, layer: int) -> list[str]:
    import json
    import struct

    def write(path: Path, name: str, dtype: str, arr: np.ndarray) -> None:
        raw = np.ascontiguousarray(arr.view(np.uint8)).tobytes()
        header = {name: {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [0, len(raw)]}}
        blob = json.dumps(header).encode()
        blob += b" " * (-len(blob) % 8)
        path.write_bytes(struct.pack("<Q", len(blob)) + blob + raw)

    write(root / "w.safetensors", f"layers.{layer}.engram.embed.weight", "F8_E4M3", weight)
    write(root / "s.safetensors", f"layers.{layer}.engram.embed.scale", "F8_E8M0", scale)
    return [str(root / "w.safetensors"), str(root / "s.safetensors")]


@pytest.mark.parametrize("mmap", [False, True])
def test_loader_fills_the_host_table_from_safetensors(tables, mmap):
    weight, scale = tables
    kv = np.random.default_rng(1).standard_normal((COLS * HEAD_DIM, HIDDEN * (HC + 1))).astype(np.float32) * 0.1

    class _Holder:  # minimal model: base.layers[layer].engram
        pass

    engram = _module("pinned_host", "fp8", kv)
    layer0, layer1 = _Holder(), _Holder()
    layer0.engram, layer1.engram = None, engram
    model = _Holder()
    model.layers = [layer0, layer1]
    with tempfile.TemporaryDirectory() as tmp:
        paths = _write_synthetic_checkpoint(Path(tmp), weight, scale, layer=1)
        assert load_engram_tables(model, paths, mmap=mmap) == {1: ROWS}
        # with mmap=True the table is backed by the files, so every lookup must happen while they exist
        ids = _ids(2, 4)
        np.testing.assert_array_equal(np.asarray(engram._lookup(jnp.asarray(ids))), _dequant(weight, scale, ids))
        assert engram.table_memory_kind == "pinned_host"
        stored_w, stored_s = engram.fp8_table_numpy()
        np.testing.assert_array_equal(stored_w.view(np.uint8), weight.view(np.uint8))
        np.testing.assert_array_equal(stored_s, scale)
        assert engram.embed.weight.value.shape == (1, HEAD_DIM)
        del stored_w, stored_s
        engram.release_host_table()


@pytest.mark.parametrize("ids_spec", ["replicated", "batch_sharded"])
def test_host_table_lookup_through_export_bind_under_jit(tables, ids_spec):
    """The model path: state passed into jit, module rebound from its GraphDef, ids sharded over a mesh."""
    weight, scale = tables
    kv = np.random.default_rng(1).standard_normal((COLS * HEAD_DIM, HIDDEN * (HC + 1))).astype(np.float32) * 0.1
    module = _module("pinned_host", "fp8", kv)
    module.set_table(weight, scale)
    graphdef, state = spx.export(module)

    @jax.jit
    def lookup(state, ids):
        return spx.bind(graphdef, state)._lookup(ids)

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("dp",))
    spec = jax.sharding.PartitionSpec() if ids_spec == "replicated" else jax.sharding.PartitionSpec("dp")
    ids = _ids(jax.device_count() * 2, 4)
    got = lookup(state, jax.device_put(jnp.asarray(ids), jax.sharding.NamedSharding(mesh, spec)))
    np.testing.assert_array_equal(np.asarray(got), _dequant(weight, scale, ids))


@pytest.mark.parametrize("ids_spec", ["replicated", "batch_sharded"])
def test_host_lookup_narrow_rows_on_multi_device_mesh(ids_spec):
    """Narrow host rows (32-byte fp8 rows, 1-byte scale rows, 4 hash columns) in a multi-device jit.

    On TPU an 8-bit ``pure_callback`` result with a minor dim under 128 bytes came back with scrambled rows (hash
    columns 1..3) once the ids were committed to a multi-device mesh; the lookup now returns uint32-packed bytes.
    """
    if jax.device_count() < 2:
        pytest.skip("needs a multi-device mesh")
    rows, head_dim, cols = 997, 32, 4
    rng = np.random.default_rng(5)
    weight = (rng.standard_normal((rows, head_dim)) * 0.5).astype(ml_dtypes.float8_e4m3fn)
    scale = rng.integers(125, 130, size=(rows, head_dim // 32), dtype=np.uint8)
    module = GatedEngram(
        GatedEngramConfig(
            hidden_size=HIDDEN,
            hc_mult=HC,
            table_rows=rows,
            num_hash_cols=cols,
            head_dim=head_dim,
            table_memory_kind="pinned_host",
            table_dtype="fp8",
        ),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    module.set_table(weight, scale)
    try:
        ids = rng.integers(0, rows, size=(jax.device_count(), 32, cols)).astype(np.int32)
        flat = ids.reshape(-1)
        factor = (2.0 ** (scale[flat].astype(np.int32) - 127)).astype(np.float32)
        expected = (weight[flat].astype(np.float32).reshape(-1, 1, 32) * factor[..., None]).reshape(*ids.shape, head_dim)
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("dp",))
        spec = jax.sharding.PartitionSpec() if ids_spec == "replicated" else jax.sharding.PartitionSpec("dp")
        placed = jax.device_put(jnp.asarray(ids), jax.sharding.NamedSharding(mesh, spec))
        np.testing.assert_array_equal(np.asarray(jax.jit(module._lookup)(placed)), expected)
    finally:
        module.release_host_table()
