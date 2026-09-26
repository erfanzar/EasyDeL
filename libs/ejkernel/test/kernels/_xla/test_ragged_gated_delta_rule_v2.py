# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Tests for ragged gated delta rule v2 XLA helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.ragged_gated_delta_rule_v2 import ragged_gated_delta_rule_v2 as xla_gdn_v2
from ejkernel.kernels._xla.ragged_gated_delta_rule_v2._xla_impl_fwd import (
    ragged_gated_delta_rule_mixed_prefill,
)


def test_mixed_prefill_zeros_missing_initial_state():
    """Fresh prefills must not consume stale recurrent slots."""
    tokens, heads, dim = 4, 2, 4
    keys = jax.random.split(jax.random.PRNGKey(0), 6)
    query = jax.random.normal(keys[0], (tokens, heads, dim), dtype=jnp.float32)
    key = jax.random.normal(keys[1], (tokens, heads, dim), dtype=jnp.float32)
    value = jax.random.normal(keys[2], (tokens, heads, dim), dtype=jnp.float32)
    b = jax.random.normal(keys[3], (tokens, heads), dtype=jnp.float32)
    a = jax.random.normal(keys[4], (tokens, heads), dtype=jnp.float32)
    recurrent_state = jax.random.normal(keys[5], (1, heads, dim, dim), dtype=jnp.float32)
    zero_state = jnp.zeros_like(recurrent_state)
    A_log = jnp.zeros((heads,), dtype=jnp.float32)
    dt_bias = jnp.zeros((heads,), dtype=jnp.float32)
    query_start_loc = jnp.array([0, tokens], dtype=jnp.int32)
    state_indices = jnp.array([0], dtype=jnp.int32)
    distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

    state_fresh, out_fresh = ragged_gated_delta_rule_mixed_prefill(
        query,
        key,
        value,
        b,
        a,
        A_log,
        dt_bias,
        query_start_loc,
        recurrent_state,
        state_indices,
        distribution,
        has_initial_state=jnp.array([False]),
        chunk_size=2,
        mask_initial_state=True,
    )
    state_zero, out_zero = ragged_gated_delta_rule_mixed_prefill(
        query,
        key,
        value,
        b,
        a,
        A_log,
        dt_bias,
        query_start_loc,
        zero_state,
        state_indices,
        distribution,
        has_initial_state=jnp.array([True]),
        chunk_size=2,
        mask_initial_state=True,
    )
    state_carried, out_carried = ragged_gated_delta_rule_mixed_prefill(
        query,
        key,
        value,
        b,
        a,
        A_log,
        dt_bias,
        query_start_loc,
        recurrent_state,
        state_indices,
        distribution,
        has_initial_state=jnp.array([True]),
        chunk_size=2,
        mask_initial_state=True,
    )

    assert jnp.allclose(out_fresh, out_zero, atol=1e-5)
    assert jnp.allclose(state_fresh, state_zero, atol=1e-5)
    assert not jnp.allclose(out_fresh, out_carried, atol=1e-5)
    assert not jnp.allclose(state_fresh, state_carried, atol=1e-5)


def test_mixed_prefill_can_skip_initial_state_mask_on_hot_path():
    """The default hot path should ignore the mask and consume the state."""
    tokens, heads, dim = 4, 2, 4
    keys = jax.random.split(jax.random.PRNGKey(1), 6)
    query = jax.random.normal(keys[0], (tokens, heads, dim), dtype=jnp.float32)
    key = jax.random.normal(keys[1], (tokens, heads, dim), dtype=jnp.float32)
    value = jax.random.normal(keys[2], (tokens, heads, dim), dtype=jnp.float32)
    b = jax.random.normal(keys[3], (tokens, heads), dtype=jnp.float32)
    a = jax.random.normal(keys[4], (tokens, heads), dtype=jnp.float32)
    recurrent_state = jax.random.normal(keys[5], (1, heads, dim, dim), dtype=jnp.float32)
    A_log = jnp.zeros((heads,), dtype=jnp.float32)
    dt_bias = jnp.zeros((heads,), dtype=jnp.float32)
    query_start_loc = jnp.array([0, tokens], dtype=jnp.int32)
    state_indices = jnp.array([0], dtype=jnp.int32)
    distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

    _, out_unmasked = ragged_gated_delta_rule_mixed_prefill(
        query,
        key,
        value,
        b,
        a,
        A_log,
        dt_bias,
        query_start_loc,
        recurrent_state,
        state_indices,
        distribution,
        has_initial_state=jnp.array([False]),
        chunk_size=2,
        mask_initial_state=False,
    )
    _, out_carried = ragged_gated_delta_rule_mixed_prefill(
        query,
        key,
        value,
        b,
        a,
        A_log,
        dt_bias,
        query_start_loc,
        recurrent_state,
        state_indices,
        distribution,
        has_initial_state=jnp.array([True]),
        chunk_size=2,
        mask_initial_state=True,
    )

    assert jnp.allclose(out_unmasked, out_carried, atol=1e-5)


def _naive_ragged_gdr(mixed_qkv, b, a, pool, A_log, dt_bias, lengths, state_indices, *, n_kq, n_v, d_k, d_v):
    """Float64 per-row sequential gated-delta-rule reference.

    Row ``r`` runs its ``lengths[r]`` tokens one by one from slot
    ``state_indices[r]``; empty rows and tokens past ``sum(lengths)`` are
    ignored. Returns ``(pool, outputs)`` with ``outputs`` zero for padding.
    """
    x = np.asarray(jnp.asarray(mixed_qkv, jnp.float32), np.float64)
    b = np.asarray(jnp.asarray(b, jnp.float32), np.float64)
    a = np.asarray(jnp.asarray(a, jnp.float32), np.float64)
    A_log = np.asarray(jnp.asarray(A_log, jnp.float32), np.float64)
    dt_bias = np.asarray(jnp.asarray(dt_bias, jnp.float32), np.float64)
    pool = np.array(pool, np.float64)
    key_dim = n_kq * d_k
    outputs = np.zeros((x.shape[0], n_v * d_v), np.float64)

    def l2norm(v):
        return v / np.sqrt(np.sum(v * v, axis=-1, keepdims=True) + 1e-6)

    start = 0
    for row, length in enumerate(lengths):
        h = pool[state_indices[row]].copy()
        for t in range(start, start + length):
            q = np.repeat(x[t, :key_dim].reshape(n_kq, d_k), n_v // n_kq, axis=0)
            k = np.repeat(x[t, key_dim : 2 * key_dim].reshape(n_kq, d_k), n_v // n_kq, axis=0)
            v = x[t, 2 * key_dim :].reshape(n_v, d_v)
            q = l2norm(q) * d_k**-0.5
            k = l2norm(k)
            beta = 1.0 / (1.0 + np.exp(-b[t]))
            g = -np.exp(A_log) * np.logaddexp(0.0, a[t] + dt_bias)
            h = h * np.exp(g)[:, None, None]
            v_new = beta[:, None] * (v - np.einsum("hd,hdm->hm", k, h))
            h = h + k[:, :, None] * v_new[:, None, :]
            outputs[t] = np.einsum("hd,hdm->hm", q, h).reshape(-1)
        pool[state_indices[row]] = h
        start += length
    return pool, outputs


def _make_ragged_inputs(seed, num_tokens, num_slots, *, n_kq, n_v, d_k, d_v, dtype=jnp.float32):
    """Random packed GDR inputs plus an fp32 state pool (numpy, never donated)."""
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    mixed_qkv = jax.random.normal(keys[0], (num_tokens, 2 * n_kq * d_k + n_v * d_v), dtype=jnp.float32).astype(dtype)
    b = jax.random.normal(keys[1], (num_tokens, n_v), dtype=jnp.float32).astype(dtype)
    a = jax.random.normal(keys[2], (num_tokens, n_v), dtype=jnp.float32).astype(dtype)
    pool = np.asarray(jax.random.normal(keys[3], (num_slots, n_v, d_k, d_v), dtype=jnp.float32)) * 0.5
    A_log = jnp.log(jnp.linspace(0.5, 2.0, n_v, dtype=jnp.float32))
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)
    return mixed_qkv, b, a, pool, A_log, dt_bias


def _row_layout(lengths, decode_only):
    """``query_start_loc`` and the positional ``distribution`` for row ``lengths``."""
    query_start_loc = jnp.asarray(np.concatenate([[0], np.cumsum(lengths)]), dtype=jnp.int32)
    num_rows = len(lengths)
    decode_end = num_rows if decode_only else sum(1 for length in lengths if length == 1)
    distribution = jnp.array([decode_end, num_rows, num_rows], dtype=jnp.int32)
    return query_start_loc, distribution


def _assert_matches_reference(
    state, out, ref_state, ref_out, pool, lengths, state_indices, *, state_tol, out_tol, zero_padding=True
):
    """Written slots match the reference, all other slots are bit-identical, padding outputs are zero.

    ``zero_padding=False`` skips the padding-output check (the chunked prefill
    path leaves padding outputs unspecified).
    """
    state = np.asarray(state)
    out = np.asarray(jnp.asarray(out, jnp.float32))
    written = {int(state_indices[row]) for row, length in enumerate(lengths) if length > 0}
    for slot in range(pool.shape[0]):
        if slot in written:
            np.testing.assert_allclose(state[slot], ref_state[slot], atol=state_tol, rtol=state_tol)
        else:
            np.testing.assert_array_equal(state[slot], pool[slot])
    num_valid = int(sum(lengths))
    np.testing.assert_allclose(out[:num_valid], ref_out[:num_valid], atol=out_tol, rtol=out_tol)
    if zero_padding:
        np.testing.assert_array_equal(out[num_valid:], 0.0)


@pytest.mark.parametrize("lengths", [(0, 1, 1, 0), (1, 0, 1, 1), (1, 1, 0, 0)])
def test_decode_only_skips_empty_rows(lengths):
    """Empty rows must not consume another row's token or touch any slot."""
    n_kq, n_v, d_k, d_v = 2, 2, 8, 8
    num_tokens, num_slots = 6, 6
    state_indices = np.array([4, 1, 5, 2], dtype=np.int32)
    mixed_qkv, b, a, pool, A_log, dt_bias = _make_ragged_inputs(
        0, num_tokens, num_slots, n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v
    )
    query_start_loc, distribution = _row_layout(lengths, decode_only=True)

    state, out = xla_gdn_v2(
        mixed_qkv,
        b,
        a,
        jnp.asarray(pool),
        A_log,
        dt_bias,
        query_start_loc,
        jnp.asarray(state_indices),
        distribution,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=4,
    )
    ref_state, ref_out = _naive_ragged_gdr(
        mixed_qkv, b, a, pool, A_log, dt_bias, lengths, state_indices, n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v
    )
    _assert_matches_reference(state, out, ref_state, ref_out, pool, lengths, state_indices, state_tol=1e-4, out_tol=1e-4)


@pytest.mark.parametrize("lengths", [(0, 3, 1, 5), (3, 0, 5, 1), (3, 1, 5, 0), (0, 4, 0, 2, 0)])
def test_mixed_prefill_skips_empty_rows(lengths):
    """Empty rows inside the row prefix must not clobber other rows' initial or final states."""
    n_kq, n_v, d_k, d_v = 2, 2, 8, 8
    num_tokens = int(sum(lengths)) + 3
    num_slots = len(lengths) + 2
    state_indices = np.arange(num_slots, dtype=np.int32)[::-1][: len(lengths)].copy()
    mixed_qkv, b, a, pool, A_log, dt_bias = _make_ragged_inputs(
        1, num_tokens, num_slots, n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v
    )
    query_start_loc, distribution = _row_layout(lengths, decode_only=False)

    state, out = xla_gdn_v2(
        mixed_qkv,
        b,
        a,
        jnp.asarray(pool),
        A_log,
        dt_bias,
        query_start_loc,
        jnp.asarray(state_indices),
        distribution,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=4,
    )
    ref_state, ref_out = _naive_ragged_gdr(
        mixed_qkv, b, a, pool, A_log, dt_bias, lengths, state_indices, n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v
    )
    # The chunked path runs Q/K/V in bf16, hence the looser tolerance.
    _assert_matches_reference(
        state, out, ref_state, ref_out, pool, lengths, state_indices, state_tol=5e-2, out_tol=5e-2, zero_padding=False
    )


def test_decode_padded_tokens_beyond_rows_do_not_write_state():
    """Padding tokens past the last row (more tokens than rows) must not overwrite the last row's slot."""
    n_kq, n_v, d_k, d_v = 2, 2, 8, 8
    lengths = (1, 1, 1)
    num_tokens, num_slots = 8, 4
    state_indices = np.array([0, 1, 2], dtype=np.int32)
    mixed_qkv, b, a, pool, A_log, dt_bias = _make_ragged_inputs(
        2, num_tokens, num_slots, n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v
    )
    query_start_loc, distribution = _row_layout(lengths, decode_only=True)

    state, out = xla_gdn_v2(
        mixed_qkv,
        b,
        a,
        jnp.asarray(pool),
        A_log,
        dt_bias,
        query_start_loc,
        jnp.asarray(state_indices),
        distribution,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=4,
    )
    ref_state, ref_out = _naive_ragged_gdr(
        mixed_qkv, b, a, pool, A_log, dt_bias, lengths, state_indices, n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v
    )
    _assert_matches_reference(state, out, ref_state, ref_out, pool, lengths, state_indices, state_tol=1e-4, out_tol=1e-4)


def test_decode_slow_decay_matches_float32_recurrence():
    """Slow-decay heads over many bf16 decode steps must track an fp32 recurrence with an fp32 state pool.

    ``exp(g)`` with ``|g| < 2**-9`` rounds to exactly 1.0 in bf16, so a kernel
    that rounds the decay (or the state) to the activation dtype drifts
    linearly away from the reference. A tiny ``beta`` makes the state
    dynamics decay-dominated so the check is sensitive to exactly that.
    """
    n_kq, n_v, d_k, d_v = 2, 2, 8, 8
    lengths = (1, 1)
    num_steps = 128
    num_slots = 3
    state_indices = np.array([2, 0], dtype=np.int32)
    target_g = np.array([-1.5e-3, -8e-4], dtype=np.float64)
    A_log = jnp.asarray(np.log(-target_g / np.log(2.0)), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)
    a = jnp.zeros((len(lengths), n_v), dtype=jnp.bfloat16)
    b = jnp.full((len(lengths), n_v), -6.0, dtype=jnp.bfloat16)
    query_start_loc, distribution = _row_layout(lengths, decode_only=True)
    pool = np.asarray(jax.random.normal(jax.random.PRNGKey(3), (num_slots, n_v, d_k, d_v), dtype=jnp.float32))

    state = jnp.asarray(pool)
    ref_state = pool.astype(np.float64)
    for step in range(num_steps):
        mixed_qkv = jax.random.normal(
            jax.random.PRNGKey(100 + step), (len(lengths), 2 * n_kq * d_k + n_v * d_v), dtype=jnp.float32
        ).astype(jnp.bfloat16)
        state, out = xla_gdn_v2(
            mixed_qkv,
            b,
            a,
            state,
            A_log,
            dt_bias,
            query_start_loc,
            jnp.asarray(state_indices),
            distribution,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            chunk_size=4,
        )
        ref_state, ref_out = _naive_ragged_gdr(
            mixed_qkv, b, a, ref_state, A_log, dt_bias, lengths, state_indices, n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v
        )
        assert state.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(jnp.asarray(out, jnp.float32)), ref_out, atol=2e-2, rtol=2e-2)

    np.testing.assert_allclose(np.asarray(state), ref_state, atol=1e-3, rtol=1e-3)
    np.testing.assert_array_equal(np.asarray(state)[1], pool[1])


@pytest.mark.parametrize("lengths", [(1, 1, 0), (3, 0, 2)])
def test_state_pool_dtype_is_preserved(lengths):
    """A bf16 runtime must not round-trip the fp32 state pool through bf16 on either branch."""
    n_kq, n_v, d_k, d_v = 2, 2, 8, 8
    num_tokens = int(sum(lengths)) + 2
    num_slots = len(lengths) + 1
    state_indices = np.arange(len(lengths), dtype=np.int32)
    mixed_qkv, b, a, pool, A_log, dt_bias = _make_ragged_inputs(
        4, num_tokens, num_slots, n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v, dtype=jnp.bfloat16
    )
    decode_only = max(lengths) <= 1
    query_start_loc, distribution = _row_layout(lengths, decode_only=decode_only)

    state, out = xla_gdn_v2(
        mixed_qkv,
        b,
        a,
        jnp.asarray(pool),
        A_log,
        dt_bias,
        query_start_loc,
        jnp.asarray(state_indices),
        distribution,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=4,
        runtime_dtype=jnp.bfloat16,
    )
    assert state.dtype == jnp.float32
    assert out.dtype == jnp.bfloat16
    ref_state, ref_out = _naive_ragged_gdr(
        mixed_qkv, b, a, pool, A_log, dt_bias, lengths, state_indices, n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v
    )
    _assert_matches_reference(
        state,
        out,
        ref_state,
        ref_out,
        pool,
        lengths,
        state_indices,
        state_tol=1e-4 if decode_only else 5e-2,
        out_tol=5e-2,
        zero_padding=decode_only,
    )
