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

"""``gated_pool_compress`` operation: float64 NumPy parity and decode / extend / windows agreement.

Reference: entry ``g`` is ``sum(kv * softmax(gate + position_bias))`` over group ``g``'s ``rate`` slots; a two-series
stream pools ``[Ca of group g - 1 | Cb of group g]`` over ``2 * rate`` slots (group 0's Ca half: zero kv, -inf gate).
"""

import jax
import numpy as np
import pytest
from easydel.operations import OperationRegistry
from easydel.operations.kernels import GatedPoolCompressOp, gated_pool_compress_op
from jax import numpy as jnp

BATCH = 3
DIM = 4


def _softmax(x, axis):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _reference(kv, gate, bias, rate, two_series):
    """float64 pooled entries ``[B, G, D]`` and (two-series) the last group's Ca kv / biased gate."""
    batch, length, feat = kv.shape
    groups = length // rate
    kv = kv[:, : groups * rate].reshape(batch, groups, rate, feat).astype(np.float64)
    gate = gate[:, : groups * rate].reshape(batch, groups, rate, feat).astype(np.float64) + bias.astype(np.float64)
    if not two_series:
        return np.sum(kv * _softmax(gate, axis=2), axis=2), None, None
    dim = feat // 2
    prev_kv = np.concatenate([np.zeros((batch, 1, rate, dim)), kv[:, :-1, :, :dim]], axis=1)
    prev_gate = np.concatenate([np.full((batch, 1, rate, dim), -np.inf), gate[:, :-1, :, :dim]], axis=1)
    slot_kv = np.concatenate([prev_kv, kv[..., dim:]], axis=2)
    slot_gate = np.concatenate([prev_gate, gate[..., dim:]], axis=2)
    return np.sum(slot_kv * _softmax(slot_gate, axis=2), axis=2), kv[:, -1, :, :dim], gate[:, -1, :, :dim]


def _inputs(rate, two_series, groups, seed):
    feat = 2 * DIM if two_series else DIM
    rng = np.random.default_rng(seed)
    kv = rng.normal(size=(BATCH, groups * rate, feat)).astype(np.float32)
    gate = rng.normal(size=(BATCH, groups * rate, feat)).astype(np.float32)
    bias = (rng.normal(size=(rate, feat)) * 0.5).astype(np.float32)
    return kv, gate, bias, feat


def _empty_state(rate, feat, two_series):
    state = dict(
        buffer_kv=jnp.zeros((BATCH, rate, feat), jnp.float32),
        buffer_gate=jnp.zeros((BATCH, rate, feat), jnp.float32),
    )
    if two_series:
        state["overlap_kv"] = jnp.zeros((BATCH, rate, DIM), jnp.float32)
        state["overlap_gate"] = jnp.full((BATCH, rate, DIM), -jnp.inf, jnp.float32)
    return state


def test_registered_under_its_name():
    assert OperationRegistry.get("gated_pool_compress") is GatedPoolCompressOp
    assert gated_pool_compress_op() is gated_pool_compress_op()


@pytest.mark.parametrize(("rate", "two_series"), [(1, False), (4, False), (2, True), (3, True)])
def test_windows_match_numpy(rate, two_series):
    kv, gate, bias, _ = _inputs(rate, two_series, groups=5, seed=rate)
    out = gated_pool_compress_op()(
        "windows",
        jnp.asarray(kv),
        jnp.asarray(gate),
        rate,
        position_bias=jnp.asarray(bias),
        series_dim=DIM if two_series else None,
    )
    pooled, ca_kv, ca_gate = _reference(kv, gate, bias, rate, two_series)
    np.testing.assert_allclose(np.asarray(out.pooled, np.float64), pooled, rtol=2e-5, atol=2e-5)
    if two_series:
        np.testing.assert_allclose(np.asarray(out.overlap_kv, np.float64), ca_kv, rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(out.overlap_gate, np.float64), ca_gate, rtol=0, atol=1e-6)
    else:
        assert out.overlap_kv is None and out.overlap_gate is None


@pytest.mark.parametrize(("rate", "two_series"), [(4, False), (2, True), (3, True)])
def test_decode_stream_emits_the_window_entries(rate, two_series):
    groups = 4
    kv, gate, bias, feat = _inputs(rate, two_series, groups, seed=10 + rate)
    pooled_ref, ca_kv, ca_gate = _reference(kv, gate, bias, rate, two_series)
    state = _empty_state(rate, feat, two_series)
    op = jax.jit(gated_pool_compress_op(), static_argnums=(0, 3))
    emitted = np.zeros_like(pooled_ref)
    for t in range(groups * rate):
        out = op(
            "decode",
            jnp.asarray(kv[:, t : t + 1]),
            jnp.asarray(gate[:, t : t + 1]),
            rate,
            position_bias=jnp.asarray(bias),
            position=jnp.full((BATCH,), t, jnp.int32),
            **state,
        )
        state = {k: getattr(out, k) for k in state}
        closed = np.asarray(out.closed)
        assert bool(closed.all()) == ((t + 1) % rate == 0)
        assert np.all(np.asarray(out.entry) == t // rate)
        if closed.all():
            emitted[:, t // rate] = np.asarray(out.pooled, np.float64)
    np.testing.assert_allclose(emitted, pooled_ref, rtol=2e-5, atol=2e-5)
    if two_series:
        np.testing.assert_allclose(np.asarray(state["overlap_kv"], np.float64), ca_kv, atol=1e-6)
        np.testing.assert_allclose(np.asarray(state["overlap_gate"], np.float64), ca_gate, atol=1e-6)


def test_decode_invalid_rows_leave_state_untouched():
    rate = 3
    kv, gate, bias, feat = _inputs(rate, True, 2, seed=3)
    state = _empty_state(rate, feat, True)
    valid = jnp.asarray([True, False, True])
    for t in range(rate):
        out = gated_pool_compress_op()(
            "decode",
            jnp.asarray(kv[:, t : t + 1]),
            jnp.asarray(gate[:, t : t + 1]),
            rate,
            position_bias=jnp.asarray(bias),
            position=jnp.full((BATCH,), t, jnp.int32),
            valid=valid,
            **state,
        )
        state = {k: getattr(out, k) for k in state}
    assert np.asarray(out.closed).tolist() == [True, False, True]
    for name, value in state.items():
        np.testing.assert_array_equal(np.asarray(value)[1], np.asarray(_empty_state(rate, feat, True)[name])[1])


@pytest.mark.parametrize(("rate", "two_series"), [(4, False), (2, True), (3, True)])
def test_extend_equals_per_token_decode(rate, two_series):
    """Ragged blocks per row (a valid prefix each) leave the same state and emit the same groups as decode."""
    block = 5
    prefix = np.array([0, 2, 1], np.int32)
    block_valid = (np.array([5, 1, 0], np.int32), np.array([3, 5, 4], np.int32))
    total = int(prefix.max()) + 2 * block
    kv, gate, bias, feat = _inputs(rate, two_series, groups=-(-total // rate), seed=20 + rate)
    rows = np.arange(BATCH)
    op = gated_pool_compress_op()

    def decode(state, start, n_valid, steps, emitted):
        for col in range(steps):
            pos = start + col
            take = np.clip(pos, 0, kv.shape[1] - 1)
            out = op(
                "decode",
                jnp.asarray(kv[rows, take][:, None]),
                jnp.asarray(gate[rows, take][:, None]),
                rate,
                position_bias=jnp.asarray(bias),
                position=jnp.asarray(pos, jnp.int32),
                valid=jnp.asarray(col < n_valid),
                **state,
            )
            state = {k: getattr(out, k) for k in state}
            for b in np.flatnonzero(np.asarray(out.closed)):
                emitted[(b, int(np.asarray(out.entry)[b]))] = np.asarray(out.pooled)[b]
        return state

    state0 = _empty_state(rate, feat, two_series)
    ref = decode(dict(state0), np.zeros(BATCH, np.int32), prefix, int(prefix.max()), {})
    got = dict(ref)
    start = prefix.copy()
    for n_valid in block_valid:
        ref_emitted: dict = {}
        ref = decode(ref, start, n_valid, block, ref_emitted)
        positions = start[:, None] + np.arange(block)[None, :]
        take = np.clip(positions, 0, kv.shape[1] - 1)
        out = op(
            "extend",
            jnp.asarray(kv[rows[:, None], take]),
            jnp.asarray(gate[rows[:, None], take]),
            rate,
            position_bias=jnp.asarray(bias),
            positions=jnp.asarray(positions, jnp.int32),
            valid=jnp.asarray(np.arange(block)[None, :] < n_valid[:, None]),
            **got,
        )
        got = {k: getattr(out, k) for k in got}
        closed, entry, pooled = np.asarray(out.closed), np.asarray(out.entry), np.asarray(out.pooled)
        got_emitted = {(b, int(entry[b, g])): pooled[b, g] for b, g in zip(*np.nonzero(closed), strict=True)}
        assert set(got_emitted) == set(ref_emitted)
        for key, value in ref_emitted.items():
            np.testing.assert_allclose(got_emitted[key], value, rtol=1e-5, atol=1e-5, err_msg=str(key))
        for name in state0:
            np.testing.assert_allclose(np.asarray(got[name]), np.asarray(ref[name]), rtol=1e-6, atol=1e-6, err_msg=name)
        start = start + n_valid
    assert len(ref_emitted) > 0
