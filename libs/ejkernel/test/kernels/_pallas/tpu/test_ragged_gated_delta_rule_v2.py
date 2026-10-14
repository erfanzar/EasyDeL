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

"""TPU Pallas registration tests for ragged GDN v2."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.ragged_gated_delta_rule_v2 import (
    ragged_gated_delta_rule_v2 as pallas_gdn_v2,
)
from ejkernel.kernels._registry import kernel_registry
from ejkernel.kernels._xla.ragged_gated_delta_rule_v2 import ragged_gated_delta_rule_v2 as xla_gdn_v2


def test_ragged_gated_delta_rule_v2_pallas_is_registered():
    """Assert the Pallas TPU GDN v2 kernel is registered with a valid signature.

    Looks up ``"ragged_gated_delta_rule_v2"`` in the kernel registry for
    ``platform="pallas", backend="tpu"`` and asserts the returned impl is the
    imported ``pallas_gdn_v2`` callable, then asserts
    ``kernel_registry.validate_signatures`` accepts all registered impls of that
    operation (i.e. the Pallas and reference signatures agree). Runs without a
    TPU since it only inspects registry metadata.
    """
    impl = kernel_registry.get("ragged_gated_delta_rule_v2", platform="pallas", backend="tpu")
    assert impl is pallas_gdn_v2
    assert kernel_registry.validate_signatures("ragged_gated_delta_rule_v2")


@pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="TPU-only Pallas execution")
def test_ragged_gated_delta_rule_v2_pallas_matches_xla_decode_smoke():
    """Smoke-test that the Pallas TPU GDN v2 decode matches the XLA reference.

    Builds a tiny decode-only batch of 2 single-token requests with one KQ and
    one V head (``d_k = d_v = 128``): a fused ``mixed_qkv`` projection of shape
    ``[tokens, 2 * key_dim + value_dim]``, zero gate inputs ``a``/``b``, zero
    ``A_log``/``dt_bias``, a per-token CSR offset array, slot ``state_indices``,
    and a ``distribution`` descriptor. It runs both ``xla_gdn_v2`` and
    ``pallas_gdn_v2`` from fresh zero recurrent states and asserts the decode
    outputs and updated states agree within ``atol=1e-2``. TPU-only via the
    function-level ``skipif``.
    """
    tokens, n_kq, n_v, d_k, d_v = 2, 1, 1, 128, 128
    key_dim = n_kq * d_k
    value_dim = n_v * d_v
    mixed_qkv = (
        jax.random.normal(
            jax.random.PRNGKey(0),
            (tokens, 2 * key_dim + value_dim),
            dtype=jnp.float32,
        )
        * 0.05
    )
    b = jnp.zeros((tokens, n_v), dtype=jnp.float32)
    a = jnp.zeros((tokens, n_v), dtype=jnp.float32)
    A_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)
    query_start_loc = jnp.array([0, 1, 2], dtype=jnp.int32)
    state_indices = jnp.array([0, 1], dtype=jnp.int32)
    distribution = jnp.array([2, 2, 2], dtype=jnp.int32)

    def fresh_state():
        """Allocate a zero-initialized recurrent GDN state for one kernel call.

        Returns:
            A zero ``float32`` array of shape ``[tokens, n_v, d_k, d_v]`` giving
            each call (XLA and Pallas) its own independent starting state so the
            two runs are compared from identical initial conditions.
        """
        return jnp.zeros((tokens, n_v, d_k, d_v), dtype=jnp.float32)

    state_xla, out_xla = xla_gdn_v2(
        mixed_qkv,
        b,
        a,
        fresh_state(),
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=2,
    )
    state_pallas, out_pallas = pallas_gdn_v2(
        mixed_qkv,
        b,
        a,
        fresh_state(),
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=2,
    )

    assert jnp.allclose(out_pallas, out_xla, atol=1e-2)
    assert jnp.allclose(state_pallas, state_xla, atol=1e-2)


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


def _run_pallas_case(lengths, num_tokens, num_slots, state_indices, *, seed, chunk_size=16, **kwargs):
    """Run the Pallas v2 kernel (bf16 activations, fp32 pool) and the fp64 reference on one ragged layout."""
    n_kq, n_v, d_k, d_v = 1, 2, 128, 128
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    mixed_qkv = jax.random.normal(keys[0], (num_tokens, 2 * n_kq * d_k + n_v * d_v), dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    b = jax.random.normal(keys[1], (num_tokens, n_v), dtype=jnp.float32).astype(jnp.bfloat16)
    a = jax.random.normal(keys[2], (num_tokens, n_v), dtype=jnp.float32).astype(jnp.bfloat16)
    pool = np.asarray(jax.random.normal(keys[3], (num_slots, n_v, d_k, d_v), dtype=jnp.float32)) * 0.1
    A_log = jnp.log(jnp.array([0.5, 2.0], dtype=jnp.float32))
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)
    query_start_loc = jnp.asarray(np.concatenate([[0], np.cumsum(lengths)]), dtype=jnp.int32)
    num_rows = len(lengths)
    decode_end = num_rows if max(lengths) <= 1 else sum(1 for length in lengths if length == 1)
    distribution = jnp.array([decode_end, num_rows, num_rows], dtype=jnp.int32)

    state, out = pallas_gdn_v2(
        mixed_qkv,
        b,
        a,
        jnp.asarray(pool),
        A_log,
        dt_bias,
        query_start_loc,
        jnp.asarray(state_indices, dtype=jnp.int32),
        distribution,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=chunk_size,
        **kwargs,
    )
    ref_state, ref_out = _naive_ragged_gdr(
        mixed_qkv, b, a, pool, A_log, dt_bias, lengths, state_indices, n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v
    )
    assert state.dtype == jnp.float32
    state = np.asarray(state)
    written = {int(state_indices[row]) for row, length in enumerate(lengths) if length > 0}
    for slot in range(num_slots):
        if slot in written:
            np.testing.assert_allclose(state[slot], ref_state[slot], atol=3e-2, rtol=3e-2)
        else:
            np.testing.assert_array_equal(state[slot], pool[slot])
    num_valid = int(sum(lengths))
    out = np.asarray(jnp.asarray(out, jnp.float32))
    np.testing.assert_allclose(out[:num_valid], ref_out[:num_valid], atol=3e-2, rtol=3e-2)
    return out, num_valid


@pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="TPU-only Pallas execution")
@pytest.mark.parametrize("num_slots", [8, 16])
def test_ragged_gated_delta_rule_v2_pallas_decode_skips_empty_rows(num_slots):
    """Decode with empty rows (start/middle/end) and padding tokens, on both Pallas decode layouts.

    ``num_slots == num_tokens`` runs the kernel over the whole pool in slot
    space; ``num_slots > num_tokens`` runs it over per-token gathered states.
    """
    lengths = (0, 1, 1, 0, 1, 1, 1, 0)
    state_indices = np.random.default_rng(0).permutation(num_slots)[: len(lengths)]
    out, num_valid = _run_pallas_case(lengths, 8, num_slots, state_indices, seed=5)
    np.testing.assert_array_equal(out[num_valid:], 0.0)


@pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="TPU-only Pallas execution")
@pytest.mark.parametrize(
    "lengths, use_recurrent_scan_prefill",
    [
        ((2, 0, 1, 6, 0), False),
        ((0, 5, 0, 3, 7, 0), False),
        ((0, 5, 0, 3, 7, 0), True),
        # Recurrent-scan layouts: a leading decode row keeps the chunked branch under the switch; a short
        # tail swallowed into the previous request's transition block must not reopen that request.
        ((2, 0, 1, 6, 0), True),
        ((3, 5), True),
        ((16, 0, 16), True),
        ((40, 1, 70), True),
    ],
)
def test_ragged_gated_delta_rule_v2_pallas_mixed_prefill_skips_empty_rows(lengths, use_recurrent_scan_prefill):
    """Chunked and recurrent-scan prefill must skip empty rows without clobbering neighbours.

    The recurrent-scan kernel receives un-activated ``mixed_qkv`` and must apply SiLU only when asked.
    """
    num_slots = len(lengths) + 2
    state_indices = np.arange(num_slots)[::-1][: len(lengths)].copy()
    _run_pallas_case(
        lengths,
        int(sum(lengths)) + 4,
        num_slots,
        state_indices,
        seed=6,
        use_recurrent_scan_prefill=use_recurrent_scan_prefill,
    )


@pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="TPU-only Pallas execution")
def test_ragged_gated_delta_rule_v2_pallas_decode_slow_decay_matches_float32_recurrence():
    """Slow-decay heads over many bf16 decode steps must track an fp32 recurrence.

    ``exp(g)`` with ``|g| < 2**-9`` rounds to exactly 1.0 in bf16; a tiny
    ``beta`` keeps the state decay-dominated so any rounding of the decay or of
    the fp32 state pool shows up as a large drift.
    """
    n_kq, n_v, d_k, d_v = 1, 2, 128, 128
    num_rows = num_slots = 8
    lengths = (1,) * num_rows
    state_indices = np.arange(num_rows, dtype=np.int32)
    target_g = np.array([-1.5e-3, -8e-4], dtype=np.float64)
    A_log = jnp.asarray(np.log(-target_g / np.log(2.0)), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)
    a = jnp.zeros((num_rows, n_v), dtype=jnp.bfloat16)
    b = jnp.full((num_rows, n_v), -6.0, dtype=jnp.bfloat16)
    query_start_loc = jnp.arange(num_rows + 1, dtype=jnp.int32)
    distribution = jnp.array([num_rows, num_rows, num_rows], dtype=jnp.int32)
    pool = np.asarray(jax.random.normal(jax.random.PRNGKey(7), (num_slots, n_v, d_k, d_v), dtype=jnp.float32))

    state = jnp.asarray(pool)
    ref_state = pool.astype(np.float64)
    for step in range(64):
        mixed_qkv = jax.random.normal(
            jax.random.PRNGKey(200 + step), (num_rows, 2 * n_kq * d_k + n_v * d_v), dtype=jnp.float32
        ).astype(jnp.bfloat16)
        state, _ = pallas_gdn_v2(
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
            chunk_size=16,
        )
        ref_state, _ = _naive_ragged_gdr(
            mixed_qkv, b, a, ref_state, A_log, dt_bias, lengths, state_indices, n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v
        )

    assert state.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(state), ref_state, atol=2e-3, rtol=2e-3)
