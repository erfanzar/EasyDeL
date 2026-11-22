# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""GatedPoolCompressor: NumPy parity (ratios 1/2/4), decode == full, open-group fp32 state."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.compressed_window import GatedPoolCompressor, GatedPoolCompressorConfig
from jax import numpy as jnp

HIDDEN = 8
HEAD_DIM = 4
EPS = 1e-6
BATCH = 2


def _build(ratio: int):
    config = GatedPoolCompressorConfig(hidden_size=HIDDEN, head_dim=HEAD_DIM, ratio=ratio, rms_norm_eps=EPS)
    module = GatedPoolCompressor(
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    rng = np.random.default_rng(100 + ratio)
    weights = {
        "kv": rng.normal(size=(HIDDEN, HEAD_DIM)).astype(np.float32),
        "gate": rng.normal(size=(HIDDEN, HEAD_DIM)).astype(np.float32),
        "norm": rng.uniform(0.5, 1.5, size=(HEAD_DIM,)).astype(np.float32),
    }
    module.kv_proj.weight.value = jnp.asarray(weights["kv"])
    if ratio > 1:
        module.gate_proj.weight.value = jnp.asarray(weights["gate"])
    module.kv_norm.weight.value = jnp.asarray(weights["norm"])
    return module, weights


def _rms_norm(x, weight):
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + EPS) * weight


def _reference_latents(x, weights, ratio):
    """Straightforward NumPy softmax-gated pooling over complete groups (float64)."""
    batch, seq, _ = x.shape
    groups = seq // ratio
    x = x.astype(np.float64)[:, : groups * ratio]
    kv = x @ weights["kv"].astype(np.float64)
    norm = weights["norm"].astype(np.float64)
    if ratio == 1:
        return _rms_norm(kv, norm)
    gate = x @ weights["gate"].astype(np.float64)
    kv = kv.reshape(batch, groups, ratio, -1)
    gate = gate.reshape(batch, groups, ratio, -1)
    expo = np.exp(gate - gate.max(axis=2, keepdims=True))
    probs = expo / expo.sum(axis=2, keepdims=True)
    return _rms_norm((kv * probs).sum(axis=2), norm)


def _tokens(seq, seed=7):
    return np.random.default_rng(seed).normal(size=(BATCH, seq, HIDDEN)).astype(np.float32)


@pytest.mark.parametrize("ratio", [1, 2, 4])
@pytest.mark.parametrize("seq", [8, 11])
def test_forward_matches_numpy(ratio, seq):
    module, weights = _build(ratio)
    x = _tokens(seq)
    expected = _reference_latents(x, weights, ratio)
    for got in (module(jnp.asarray(x)), jax.jit(module)(jnp.asarray(x))):
        assert got.shape == (BATCH, seq // ratio, HEAD_DIM)
        assert got.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-4, atol=1e-5)


def test_forward_returns_none_without_a_complete_group():
    module, _ = _build(4)
    assert module(jnp.asarray(_tokens(3))) is None


@pytest.mark.parametrize("ratio", [1, 2, 3])
def test_decode_steps_equal_full_forward_and_leave_open_group_state(ratio):
    seq = 10
    module, weights = _build(ratio)
    x = _tokens(seq)
    full = _reference_latents(x, weights, ratio)
    kv_state = score_state = None
    if ratio > 1:
        kv_state = jnp.zeros((BATCH, ratio, HEAD_DIM), jnp.float32)
        score_state = jnp.zeros((BATCH, ratio, HEAD_DIM), jnp.float32)
    for t in range(seq):
        position = jnp.full((BATCH,), t, jnp.int32)
        latent, kv_state, score_state, closed, entry = module.decode_step(
            jnp.asarray(x[:, t : t + 1]), kv_state, score_state, position
        )
        assert latent.shape == (BATCH, 1, HEAD_DIM)
        should_close = (t + 1) % ratio == 0
        np.testing.assert_array_equal(np.asarray(closed), np.full((BATCH,), should_close))
        np.testing.assert_array_equal(np.asarray(entry), np.full((BATCH,), t // ratio))
        if should_close:
            np.testing.assert_allclose(np.asarray(latent[:, 0]), full[:, t // ratio], rtol=1e-4, atol=1e-5)
    remainder = seq % ratio
    if ratio > 1 and remainder:
        assert kv_state.dtype == jnp.float32 and score_state.dtype == jnp.float32
        tail_kv, tail_score = module.project(jnp.asarray(x[:, seq - remainder :]))
        np.testing.assert_allclose(np.asarray(kv_state[:, :remainder]), np.asarray(tail_kv), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(score_state[:, :remainder]), np.asarray(tail_score), rtol=1e-5, atol=1e-6)


def test_decode_step_handles_rows_at_different_positions():
    ratio = 4
    module, weights = _build(ratio)
    x = _tokens(8)
    full = _reference_latents(x, weights, ratio)
    positions = np.array([7, 5], np.int32)  # row 0 closes group 1, row 1 is mid-group
    kv_state = np.zeros((BATCH, ratio, HEAD_DIM), np.float32)
    score_state = np.zeros((BATCH, ratio, HEAD_DIM), np.float32)
    for row, pos in enumerate(positions):
        start = (pos // ratio) * ratio
        if pos > start:
            kv, score = module.project(jnp.asarray(x[row : row + 1, start:pos]))
            kv_state[row, : pos - start] = np.asarray(kv[0])
            score_state[row, : pos - start] = np.asarray(score[0])
    hidden = np.stack([x[row, pos] for row, pos in enumerate(positions)])[:, None]
    latent, new_kv, _, closed, entry = module.decode_step(
        jnp.asarray(hidden), jnp.asarray(kv_state), jnp.asarray(score_state), jnp.asarray(positions)
    )
    np.testing.assert_array_equal(np.asarray(closed), [True, False])
    np.testing.assert_array_equal(np.asarray(entry), [1, 1])
    np.testing.assert_allclose(np.asarray(latent[0, 0]), full[0, 1], rtol=1e-4, atol=1e-5)
    projected_kv, _ = module.project(jnp.asarray(hidden))
    # Each row wrote its own token into slot position % ratio.
    np.testing.assert_allclose(np.asarray(new_kv[0, 3]), np.asarray(projected_kv[0, 0]), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(new_kv[1, 1]), np.asarray(projected_kv[1, 0]), rtol=1e-5, atol=1e-6)


def test_parameter_leaves_follow_the_ratio():
    for ratio, expected in (
        (1, {"kv_proj.weight", "kv_norm.weight"}),
        (2, {"kv_proj.weight", "gate_proj.weight", "kv_norm.weight"}),
    ):
        module, _ = _build(ratio)
        _, state = spx.export(module)
        paths = {path for collection, path, _ in state.items() if collection == "parameters"}
        assert paths == expected


def test_config_rejects_non_positive_ratio():
    with pytest.raises(ValueError):
        GatedPoolCompressorConfig(hidden_size=HIDDEN, head_dim=HEAD_DIM, ratio=0)
