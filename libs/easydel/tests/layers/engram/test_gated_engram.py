# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""GatedEngram: gate / value arithmetic against a NumPy reference, masking, dtypes and parameter layout."""

import jax
import numpy as np
import spectrax as spx
from easydel.layers.engram import GatedEngram, GatedEngramConfig
from jax import numpy as jnp

HIDDEN = 6
HC = 2
ROWS = 20
COLS = 3
HEAD_DIM = 4
EPS = 1e-6
BATCH = 2
SEQ = 5


def _config():
    return GatedEngramConfig(
        hidden_size=HIDDEN, hc_mult=HC, table_rows=ROWS, num_hash_cols=COLS, head_dim=HEAD_DIM, eps=EPS
    )


def _build(dtype=jnp.float32):
    module = GatedEngram(
        _config(), dtype=dtype, param_dtype=dtype, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    rng = np.random.default_rng(11)
    weights = {
        "table": rng.normal(size=(ROWS, HEAD_DIM)).astype(np.float32),
        "kv": rng.normal(size=(COLS * HEAD_DIM, HIDDEN * (HC + 1))).astype(np.float32) * 0.5,
        "q": rng.uniform(0.5, 1.5, size=(HC, HIDDEN)).astype(np.float32),
        "k": rng.uniform(0.5, 1.5, size=(HC, HIDDEN)).astype(np.float32),
    }
    module.embed.weight.value = jnp.asarray(weights["table"], dtype)
    module.kv_proj.weight.value = jnp.asarray(weights["kv"], dtype)
    module.q_weight.value = jnp.asarray(weights["q"], dtype)
    module.k_weight.value = jnp.asarray(weights["k"], dtype)
    return module, weights


def _inputs(seed=3):
    rng = np.random.default_rng(seed)
    streams = rng.normal(size=(BATCH, SEQ, HC, HIDDEN)).astype(np.float32)
    ids = rng.integers(0, ROWS, size=(BATCH, SEQ, COLS)).astype(np.int32)
    return streams, ids


def _reference(weights, streams, ids, mask=None):
    w = {name: value.astype(np.float64) for name, value in weights.items()}
    h = streams.astype(np.float64)
    batch, seq = ids.shape[:2]
    rows = w["table"][ids].reshape(batch, seq, -1)
    kv = rows @ w["kv"]
    key = kv[..., : HC * HIDDEN].reshape(batch, seq, HC, HIDDEN)
    value = kv[..., HC * HIDDEN :]
    weight = w["q"] * w["k"]
    rstd = 1.0 / np.sqrt((h * h).mean(-1) + EPS) * (1.0 / np.sqrt((key * key).mean(-1) + EPS))
    dot = (h * weight * key).sum(-1) * rstd * HIDDEN**-0.5
    gate = 1.0 / (1.0 + np.exp(-np.copysign(np.sqrt(np.maximum(np.abs(dot), 1e-6)), dot)))
    if mask is not None:
        gate = np.where(mask[..., None], gate, 0.0)
    return h + gate[..., None] * value[:, :, None, :]


def test_forward_matches_numpy():
    module, weights = _build()
    streams, ids = _inputs()
    expected = _reference(weights, streams, ids)
    for got in (
        module(jnp.asarray(streams), jnp.asarray(ids)),
        jax.jit(module)(jnp.asarray(streams), jnp.asarray(ids)),
    ):
        assert got.shape == streams.shape and got.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-4, atol=1e-5)
    assert not np.allclose(expected, streams)  # the gate actually contributes


def test_token_mask_zeroes_the_gate():
    module, weights = _build()
    streams, ids = _inputs()
    mask = np.random.default_rng(4).random((BATCH, SEQ)) > 0.4
    mask[0, 0], mask[0, 1] = True, False
    got = np.asarray(module(jnp.asarray(streams), jnp.asarray(ids), jnp.asarray(mask)))
    np.testing.assert_allclose(got, _reference(weights, streams, ids, mask), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(got[~mask], streams[~mask], rtol=1e-6, atol=1e-6)


def test_bfloat16_keeps_stream_dtype_and_shape():
    module, _ = _build(jnp.bfloat16)
    streams, ids = _inputs()
    got = module(jnp.asarray(streams, jnp.bfloat16), jnp.asarray(ids))
    assert got.shape == streams.shape and got.dtype == jnp.bfloat16
    assert np.isfinite(np.asarray(got, np.float32)).all()


def test_parameter_leaves_and_shapes():
    module, _ = _build()
    _, state = spx.export(module)
    shapes = {}
    for collection, path, leaf in state.items():
        if collection == "parameters":
            shapes[path] = tuple((leaf.value if hasattr(leaf, "value") else leaf).shape)
    assert shapes == {
        "embed.weight": (ROWS, HEAD_DIM),
        "kv_proj.weight": (COLS * HEAD_DIM, HIDDEN * (HC + 1)),
        "q_weight": (HC, HIDDEN),
        "k_weight": (HC, HIDDEN),
    }
