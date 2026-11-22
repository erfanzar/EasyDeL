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

"""XLA tests for the GDN speculative-window state scan."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from ejkernel.kernels._xla.gdn_spec_window import gdn_spec_window_states

NKQ, NV, DK, DV = 2, 4, 8, 8
KEY_DIM = NKQ * DK
QKV_DIM = 2 * KEY_DIM + NV * DV


def _make_inputs(batch=3, steps=4, seed=0, dtype=jnp.float32):
    """Random window inputs plus a mixed validity mask."""
    key = jax.random.PRNGKey(seed)
    kx, kb, ka, kal, ks = jax.random.split(key, 5)
    mixed = jax.random.normal(kx, (batch, steps, QKV_DIM), dtype=dtype) * 0.3
    b = jax.random.normal(kb, (batch, steps, NV), dtype=dtype)
    a = jax.random.normal(ka, (batch, steps, NV), dtype=dtype)
    A_log = jax.random.normal(kal, (NV,), dtype=jnp.float32) * 0.1
    dt_bias = jnp.ones((NV,), dtype=jnp.float32)
    state = jax.random.normal(ks, (batch, NV, DK, DV), dtype=jnp.float32) * 0.05
    valid = jnp.asarray(
        np.array([[1] * steps, [1] * (steps - 2) + [0, 0], [0] * steps][:batch] or [[1] * steps]),
        dtype=bool,
    )
    return mixed, b, a, A_log, dt_bias, valid, state


def _reference(mixed, b, a, A_log, dt_bias, valid, state, apply_silu=False):
    """Independent per-token reference of the sequential gated delta rule."""
    batch, steps, _ = mixed.shape
    x = jax.nn.silu(mixed.astype(jnp.float32)).astype(mixed.dtype) if apply_silu else mixed
    outs = []
    s = np.array(state, dtype=np.float32, copy=True)
    for t in range(steps):
        row = np.asarray(x[:, t], dtype=np.float32)
        k = row[:, KEY_DIM : 2 * KEY_DIM].reshape(batch, NKQ, DK)
        v = row[:, 2 * KEY_DIM :].reshape(batch, NV, DV)
        k = np.repeat(k, NV // NKQ, axis=1)
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        beta = 1.0 / (1.0 + np.exp(-np.asarray(b[:, t], dtype=np.float32)))
        alpha = np.asarray(a[:, t], dtype=np.float32) + np.asarray(dt_bias)[None, :]
        softplus = np.log1p(np.exp(alpha))
        g = np.exp(-np.exp(np.asarray(A_log))[None, :] * softplus)
        for bi in range(batch):
            if not bool(valid[bi, t]):
                continue
            for h in range(NV):
                k_mem = k[bi, h] @ s[bi, h]
                v_new = beta[bi, h] * (v[bi, h] - g[bi, h] * k_mem)
                s[bi, h] = s[bi, h] * g[bi, h] + np.outer(k[bi, h], v_new)
        outs.append(s.copy())
    return np.stack(outs, axis=0)


def test_matches_independent_reference():
    """XLA impl matches a per-(batch,head) numpy reference including masking."""
    mixed, b, a, A_log, dt_bias, valid, state = _make_inputs()
    got = np.asarray(gdn_spec_window_states(mixed, b, a, A_log, dt_bias, valid, state, n_kq=NKQ, n_v=NV, d_k=DK, d_v=DV))
    ref = _reference(mixed, b, a, A_log, dt_bias, valid, state)
    np.testing.assert_allclose(got, ref, rtol=2e-5, atol=2e-5)


def test_invalid_steps_freeze_state():
    """Rows with valid=False must carry the state through unchanged."""
    mixed, b, a, A_log, dt_bias, _, state = _make_inputs(batch=2, steps=3)
    valid = jnp.asarray(np.array([[True, False, True], [False, False, False]]))
    got = np.asarray(gdn_spec_window_states(mixed, b, a, A_log, dt_bias, valid, state, n_kq=NKQ, n_v=NV, d_k=DK, d_v=DV))
    # Batch row 1 is fully invalid: every stacked step equals the entry state.
    for t in range(3):
        np.testing.assert_allclose(got[t, 1], np.asarray(state[1], dtype=np.float32), rtol=0, atol=0)
    # Batch row 0 step 1 invalid: state after step 1 equals state after step 0.
    np.testing.assert_allclose(got[1, 0], got[0, 0], rtol=0, atol=0)


def test_apply_silu_path():
    """apply_silu=True equals pre-activating the rows outside the op."""
    mixed, b, a, A_log, dt_bias, valid, state = _make_inputs(seed=3)
    got = np.asarray(
        gdn_spec_window_states(
            mixed, b, a, A_log, dt_bias, valid, state, n_kq=NKQ, n_v=NV, d_k=DK, d_v=DV, apply_silu=True
        )
    )
    pre = jax.nn.silu(mixed.astype(jnp.float32)).astype(mixed.dtype)
    ref = np.asarray(
        gdn_spec_window_states(
            pre, b, a, A_log, dt_bias, valid, state, n_kq=NKQ, n_v=NV, d_k=DK, d_v=DV, apply_silu=False
        )
    )
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


def test_combined_ragged_gdn_window_matches_two_op_composition():
    """The fused ragged-GDN+window entry point equals the two-op composition."""
    from ejkernel.modules.operations.ragged_gated_delta_rule_v2 import (
        ragged_gated_delta_rule_v2,
        ragged_gated_delta_rule_v2_with_window_states,
    )

    base, cc = 4, 3
    seq = 16
    key = jax.random.PRNGKey(7)
    kx, kb, ka, kal, ks = jax.random.split(key, 5)
    mixed = jax.random.normal(kx, (seq, QKV_DIM), dtype=jnp.float32) * 0.3
    b = jax.random.normal(kb, (seq, NV), dtype=jnp.float32)
    a = jax.random.normal(ka, (seq, NV), dtype=jnp.float32)
    A_log = jax.random.normal(kal, (NV,), dtype=jnp.float32) * 0.1
    dt_bias = jnp.ones((NV,), dtype=jnp.float32)
    pool = jax.random.normal(ks, (base * (1 + cc), NV, DK, DV), dtype=jnp.float32) * 0.05

    sched = np.array([cc, cc, cc, 0], dtype=np.int32)
    qsl = np.zeros((base + 1,), dtype=np.int32)
    qsl[1:] = np.cumsum(sched)
    starts = qsl[:base]
    state_indices = jnp.arange(base, dtype=jnp.int32)
    distribution = jnp.asarray([0, 3, 3], dtype=jnp.int32)
    has_init = jnp.asarray([True, True, True, False])
    positions = np.clip(starts[:, None] + np.arange(cc)[None, :], 0, seq - 1).astype(np.int32)
    valid = jnp.asarray(np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool))

    common = dict(n_kq=NKQ, n_v=NV, d_k=DK, d_v=DV, chunk_size=8, use_qk_norm_in_gdn=True, platform="xla")
    ref_state, ref_out = ragged_gated_delta_rule_v2(
        mixed,
        b,
        a,
        jnp.array(pool, copy=True),
        A_log,
        dt_bias,
        jnp.asarray(qsl),
        state_indices,
        distribution,
        has_init,
        **common,
    )
    from ejkernel.kernels._xla.gdn_spec_window import gdn_spec_window_states as window_ref

    flat = positions.reshape(-1)
    ref_stack = window_ref(
        jnp.asarray(mixed)[flat].reshape(base, cc, -1),
        jnp.asarray(b)[flat].reshape(base, cc, -1),
        jnp.asarray(a)[flat].reshape(base, cc, -1),
        A_log,
        dt_bias,
        valid,
        pool[:base],
        n_kq=NKQ,
        n_v=NV,
        d_k=DK,
        d_v=DV,
    )

    got_state, got_out, got_stack = ragged_gated_delta_rule_v2_with_window_states(
        mixed,
        b,
        a,
        jnp.array(pool, copy=True),
        A_log,
        dt_bias,
        jnp.asarray(qsl),
        state_indices,
        distribution,
        jnp.asarray(positions),
        valid,
        has_init,
        num_base_slots=base,
        **common,
    )
    np.testing.assert_allclose(np.asarray(got_state), np.asarray(ref_state), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(got_out), np.asarray(ref_out), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(got_stack), np.asarray(ref_stack), rtol=0, atol=0)
