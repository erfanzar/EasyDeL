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
    # The chunked path now runs in fp32 by default and zeroes trailing padding outputs.
    _assert_matches_reference(
        state, out, ref_state, ref_out, pool, lengths, state_indices, state_tol=1e-4, out_tol=1e-4, zero_padding=True
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
        zero_padding=True,
    )


def test_mixed_prefill_zeros_nan_padding_and_oob_state_indices():
    """NaN trailing padding tokens and OOB state_indices must not leak into outputs or state."""
    n_kq, n_v, d_k, d_v = 2, 4, 8, 8
    lengths = (3, 2, 2)
    num_valid = sum(lengths)
    num_tokens = num_valid + 4
    num_slots = 4
    # Row 1 has an OOB negative state index; row 0 and row 2 have valid slots 1 and 3.
    state_indices = np.array([1, -1, 3], dtype=np.int32)
    mixed_qkv, b, a, pool, A_log, dt_bias = _make_ragged_inputs(
        11, num_tokens, num_slots, n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v
    )
    # Poison trailing padding tokens with NaNs to verify scatter-drop isolation.
    mixed_qkv = mixed_qkv.at[num_valid:].set(jnp.nan)
    b = b.at[num_valid:].set(jnp.nan)
    a = a.at[num_valid:].set(jnp.nan)
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

    out_np = np.asarray(out)
    state_np = np.asarray(state)
    assert np.all(np.isfinite(out_np))
    assert np.all(np.isfinite(state_np))
    # Trailing padding tokens and OOB row 1 tokens [3:5] must be exact zeros.
    np.testing.assert_array_equal(out_np[num_valid:], 0.0)
    np.testing.assert_array_equal(out_np[3:5], 0.0)
    # Untouched pool slots 0 and 2 remain bit-identical.
    np.testing.assert_array_equal(state_np[0], pool[0])
    np.testing.assert_array_equal(state_np[2], pool[2])

    # Compare valid rows 0 and 2 against the float64 reference.
    ref_state0, ref_out0 = _naive_ragged_gdr(
        mixed_qkv[:3], b[:3], a[:3], pool, A_log, dt_bias, (3,), np.array([1]), n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v
    )
    ref_state2, ref_out2 = _naive_ragged_gdr(
        mixed_qkv[5:7], b[5:7], a[5:7], pool, A_log, dt_bias, (2,), np.array([3]), n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v
    )
    np.testing.assert_allclose(out_np[:3], ref_out0, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out_np[5:7], ref_out2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(state_np[1], ref_state0[1], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(state_np[3], ref_state2[3], atol=1e-4, rtol=1e-4)


def test_ragged_gdn_conv_decode_validation_pool_bounds_and_parity():
    """Verify ragged_gdn_conv_decode interface validation, per-pool drop bounds, and reference parity."""
    from ejkernel.kernels._registry import kernel_registry
    from ejkernel.kernels._xla.ragged_causal_conv1d import ragged_causal_conv1d
    from ejkernel.kernels._xla.ragged_gated_delta_rule_v2 import ragged_gdn_conv_decode, ragged_gdn_conv_decode_xla

    assert kernel_registry.validate_signatures("ragged_gdn_conv_decode")

    n_kq, n_v, d_k, d_v, d_conv = 2, 4, 8, 8, 4
    conv_dim = 2 * n_kq * d_k + n_v * d_v
    num_tokens, num_slots = 4, 5
    keys = jax.random.split(jax.random.PRNGKey(21), 7)
    conv_input = jax.random.normal(keys[0], (num_tokens, conv_dim), dtype=jnp.float32)
    b_raw = jax.random.normal(keys[1], (num_tokens, n_v), dtype=jnp.float32)
    a_raw = jax.random.normal(keys[2], (num_tokens, n_v), dtype=jnp.float32)
    conv_weight = jax.random.normal(keys[3], (conv_dim, d_conv), dtype=jnp.float32)
    conv_pool = jax.random.normal(keys[4], (num_slots, conv_dim, d_conv), dtype=jnp.float32)
    rec_pool = jax.random.normal(keys[5], (num_slots, n_v, d_k, d_v), dtype=jnp.float32) * 0.5
    A_log = jnp.log(jnp.linspace(0.5, 1.5, n_v, dtype=jnp.float32))
    dt_bias = jnp.linspace(-0.2, 0.2, n_v, dtype=jnp.float32)
    token_slots = jnp.array([2, 0, -1, 4], dtype=jnp.int32)
    token_init = jnp.array([True, False, True, True], dtype=jnp.bool_)

    # 1. Interface rejects mismatched pool sizes via @jaxtyped and invalid divisibility / token_slots via ValueError.
    bigger_conv_pool = jax.random.normal(keys[6], (num_slots + 2, conv_dim, d_conv), dtype=jnp.float32)
    with pytest.raises((TypeError, ValueError)):
        ragged_gdn_conv_decode(
            conv_input,
            b_raw,
            a_raw,
            conv_weight,
            bigger_conv_pool,
            rec_pool,
            A_log,
            dt_bias,
            token_slots,
            token_init,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
    with pytest.raises(ValueError, match="divisible"):
        ragged_gdn_conv_decode_xla(
            conv_input,
            b_raw,
            a_raw,
            conv_weight,
            conv_pool,
            rec_pool,
            A_log,
            dt_bias,
            token_slots,
            token_init,
            n_kq=3,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
    with pytest.raises(ValueError, match="token_slots"):
        ragged_gdn_conv_decode_xla(
            conv_input,
            b_raw,
            a_raw,
            conv_weight,
            conv_pool,
            rec_pool,
            A_log,
            dt_bias,
            token_slots[:2],
            token_init,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )

    # 2. Direct XLA helper with conv_pool.shape[0] > recurrent_pool.shape[0] must NOT corrupt conv_pool[num_rec_slots].
    out_mismatch, cp_mismatch, rp_mismatch = ragged_gdn_conv_decode_xla(
        conv_input,
        b_raw,
        a_raw,
        conv_weight,
        bigger_conv_pool,
        rec_pool,
        A_log,
        dt_bias,
        token_slots,
        token_init,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
    )
    np.testing.assert_array_equal(np.asarray(cp_mismatch[num_slots]), np.asarray(bigger_conv_pool[num_slots]))
    np.testing.assert_array_equal(np.asarray(cp_mismatch[num_slots + 1]), np.asarray(bigger_conv_pool[num_slots + 1]))
    np.testing.assert_array_equal(np.asarray(out_mismatch[2]), 0.0)
    np.testing.assert_array_equal(np.asarray(rp_mismatch[1]), np.asarray(rec_pool[1]))

    # 3. Valid tokens match standalone ragged_causal_conv1d + xla_gdn_v2 decode_only.
    out, updated_cp, updated_rp = ragged_gdn_conv_decode(
        conv_input,
        b_raw,
        a_raw,
        conv_weight,
        conv_pool,
        rec_pool,
        A_log,
        dt_bias,
        token_slots,
        token_init,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
    )
    # Build equivalent reference inputs for the 3 live tokens (indices 0, 1, 3) with slot 0 zeroed for token_init=False.
    live_idx = jnp.array([0, 1, 3])
    live_slots = token_slots[live_idx]
    ref_cp_in = conv_pool.at[0].set(0.0)
    ref_rp_in = rec_pool.at[0].set(0.0)
    q_start = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    dist = jnp.array([3, 3, 3], dtype=jnp.int32)
    conv_out_ref, cp_ref = ragged_causal_conv1d(
        conv_input[live_idx],
        ref_cp_in,
        conv_weight,
        q_start,
        live_slots,
        dist,
        d_conv=d_conv,
        apply_silu=True,
    )
    rp_ref, gdn_out_ref = xla_gdn_v2(
        conv_out_ref,
        b_raw[live_idx],
        a_raw[live_idx],
        ref_rp_in,
        A_log,
        dt_bias,
        q_start,
        live_slots,
        dist,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
    )
    np.testing.assert_allclose(np.asarray(out[live_idx]).reshape(3, -1), np.asarray(gdn_out_ref), atol=1e-4, rtol=1e-4)
    np.testing.assert_array_equal(np.asarray(out[2]), 0.0)
    for slot in (0, 2, 4):
        np.testing.assert_allclose(np.asarray(updated_cp[slot]), np.asarray(cp_ref[slot]), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(updated_rp[slot]), np.asarray(rp_ref[slot]), atol=1e-4, rtol=1e-4)
    for untouched in (1, 3):
        np.testing.assert_array_equal(np.asarray(updated_cp[untouched]), np.asarray(conv_pool[untouched]))
        np.testing.assert_array_equal(np.asarray(updated_rp[untouched]), np.asarray(rec_pool[untouched]))


def test_ragged_gdn_conv_decode_head_sharded_export_and_fallback_mesh():
    """Verify public export, canonical conv_pool interleave/deinterleave, and FSDP-only axis fallback."""
    from ejkernel.modules.operations import ragged_gdn_conv_decode_head_sharded
    from ejkernel.modules.operations.ragged_gated_delta_rule_v2 import (
        _deinterleave_concatenated_tensor_from_sharding,
        _reorder_concatenated_tensor_for_sharding,
    )
    from jax.sharding import Mesh

    n_kq, n_v, d_k, d_v, d_conv = 4, 8, 8, 8, 4
    key_dim = n_kq * d_k
    val_dim = n_v * d_v
    conv_dim = 2 * key_dim + val_dim
    split_sizes = (key_dim, key_dim, val_dim)

    # Round-trip check for interleave / deinterleave inverse.
    arr = jnp.arange(3 * conv_dim * d_conv, dtype=jnp.float32).reshape(3, conv_dim, d_conv)
    interleaved = _reorder_concatenated_tensor_for_sharding(arr, split_sizes, 4, 1)
    restored = _deinterleave_concatenated_tensor_from_sharding(interleaved, split_sizes, 4, 1)
    np.testing.assert_array_equal(np.asarray(restored), np.asarray(arr))

    num_tokens, num_slots = 4, 5
    keys = jax.random.split(jax.random.PRNGKey(31), 6)
    conv_input = jax.random.normal(keys[0], (num_tokens, conv_dim), dtype=jnp.float32)
    b_raw = jax.random.normal(keys[1], (num_tokens, n_v), dtype=jnp.float32)
    a_raw = jax.random.normal(keys[2], (num_tokens, n_v), dtype=jnp.float32)
    conv_weight = jax.random.normal(keys[3], (conv_dim, d_conv), dtype=jnp.float32)
    conv_pool = jax.random.normal(keys[4], (num_slots, conv_dim, d_conv), dtype=jnp.float32)
    rec_pool = jax.random.normal(keys[5], (num_slots, n_v, d_k, d_v), dtype=jnp.float32) * 0.5
    A_log = jnp.log(jnp.linspace(0.5, 1.5, n_v, dtype=jnp.float32))
    dt_bias = jnp.linspace(-0.2, 0.2, n_v, dtype=jnp.float32)
    token_slots = jnp.array([3, 1, -1, 0], dtype=jnp.int32)
    token_init = jnp.array([True, False, True, True], dtype=jnp.bool_)

    # Reference unsharded output.
    ref_out, ref_cp, ref_rp = ragged_gdn_conv_decode_head_sharded(
        conv_input,
        b_raw,
        a_raw,
        conv_weight,
        conv_pool,
        rec_pool,
        A_log,
        dt_bias,
        token_slots,
        token_init,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        platform="xla",
    )

    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires >= 2 host devices for multi-device shard_map fallback test")

    n_dev = 4 if len(devices) >= 4 else 2
    mesh = Mesh(np.array(devices[:n_dev]).reshape(n_dev, 1), ("fsdp", "tp"))

    # Case 2: head_axis="tp" has size 1; allow_axis_fallback=True falls back to "fsdp" with canonical conv_pool.
    fb_out, fb_cp, fb_rp = ragged_gdn_conv_decode_head_sharded(
        conv_input,
        b_raw,
        a_raw,
        conv_weight,
        conv_pool,
        rec_pool,
        A_log,
        dt_bias,
        token_slots,
        token_init,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        mesh=mesh,
        head_axis="tp",
        allow_axis_fallback=True,
        platform="xla",
    )
    np.testing.assert_allclose(np.asarray(fb_out), np.asarray(ref_out), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(fb_cp), np.asarray(ref_cp), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(fb_rp), np.asarray(ref_rp), atol=1e-5, rtol=1e-5)

    # Case 3: allow_axis_fallback=False runs unsharded and also matches reference.
    un_out, un_cp, un_rp = ragged_gdn_conv_decode_head_sharded(
        conv_input,
        b_raw,
        a_raw,
        conv_weight,
        conv_pool,
        rec_pool,
        A_log,
        dt_bias,
        token_slots,
        token_init,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        mesh=mesh,
        head_axis="tp",
        allow_axis_fallback=False,
        platform="xla",
    )
    np.testing.assert_allclose(np.asarray(un_out), np.asarray(ref_out), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(un_cp), np.asarray(ref_cp), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(un_rp), np.asarray(ref_rp), atol=1e-5, rtol=1e-5)
