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

"""XLA tests for gated_delta_rule."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from ejkernel.kernels._xla.gated_delta_rule import gated_delta_rule


def _make_inputs(batch=1, seq_len=8, heads=2, qk_dim=8, v_dim=8, dtype=jnp.float32, seed=0):
    """Create test inputs in public API shape [batch, seq_len, heads, dim]."""
    key = jax.random.PRNGKey(seed)
    kq, kk, kv, kb, kd = jax.random.split(key, 5)

    q = jax.random.normal(kq, (batch, seq_len, heads, qk_dim), dtype=dtype)
    k = jax.random.normal(kk, (batch, seq_len, heads, qk_dim), dtype=dtype)
    v = jax.random.normal(kv, (batch, seq_len, heads, v_dim), dtype=dtype)
    beta = jax.nn.sigmoid(jax.random.normal(kb, (batch, seq_len, heads), dtype=dtype))
    decay = jax.random.normal(kd, (batch, seq_len, heads), dtype=dtype) * -0.01

    return q, k, v, beta, decay


def _make_internal(batch=1, seq_len=8, heads=2, qk_dim=8, v_dim=8, dtype=jnp.float32, seed=0):
    """Create test inputs in internal shape [batch, heads, seq_len, dim]."""
    q, k, v, beta, decay = _make_inputs(batch, seq_len, heads, qk_dim, v_dim, dtype, seed)
    return (
        q.transpose(0, 2, 1, 3),
        k.transpose(0, 2, 1, 3),
        v.transpose(0, 2, 1, 3),
        beta.transpose(0, 2, 1),
        decay.transpose(0, 2, 1),
    )


def test_interface_returns_correct_shapes():
    batch, seq_len, heads, qk_dim, v_dim = 1, 16, 2, 8, 8
    q, k, v, beta, decay = _make_inputs(batch, seq_len, heads, qk_dim, v_dim)

    out, state = gated_delta_rule(q, k, v, beta, decay, chunk_size=8)

    assert out.shape == (batch, seq_len, heads, v_dim)
    assert state.shape == (batch, heads, qk_dim, v_dim)


def test_chunked_and_recurrent_agree():
    batch, seq_len, heads, qk_dim, v_dim = 1, 16, 2, 8, 8
    q, k, v, beta, decay = _make_inputs(batch, seq_len, heads, qk_dim, v_dim)

    out_chunk, _state_chunk = gated_delta_rule(q, k, v, beta, decay, chunk_size=8, use_chunked=True)
    out_rec, _state_rec = gated_delta_rule(q, k, v, beta, decay, use_chunked=False)

    max_diff = float(jnp.max(jnp.abs(out_chunk - out_rec)))
    assert max_diff < 0.25, f"chunked vs recurrent max diff: {max_diff}"


def test_no_decay():
    batch, seq_len, heads, qk_dim, v_dim = 1, 8, 2, 8, 8
    q, k, v, beta, _ = _make_inputs(batch, seq_len, heads, qk_dim, v_dim)

    out, _state = gated_delta_rule(q, k, v, beta, None, chunk_size=8)
    assert out.shape == (batch, seq_len, heads, v_dim)
    assert jnp.all(jnp.isfinite(out))


def test_state_continuation():
    """Verify that chunked(full) == chunked(prefix) + single_step(suffix, state)."""
    batch, heads, qk_dim, v_dim = 1, 2, 8, 8
    q, k, v, beta, decay = _make_inputs(batch, 9, heads, qk_dim, v_dim)

    # Full sequence
    out_full, _ = gated_delta_rule(q, k, v, beta, decay, chunk_size=8)

    # Prefix + state
    _, state = gated_delta_rule(
        q[:, :8],
        k[:, :8],
        v[:, :8],
        beta[:, :8],
        decay[:, :8],
        chunk_size=8,
    )

    # Single step with state
    out_last, _ = gated_delta_rule(
        q[:, 8:],
        k[:, 8:],
        v[:, 8:],
        beta[:, 8:],
        decay[:, 8:],
        initial_state=state,
    )

    max_diff = float(jnp.max(jnp.abs(out_full[:, 8:] - out_last)))
    assert max_diff < 0.5, f"state continuation max diff: {max_diff}"


def test_bfloat16_inputs():
    batch, seq_len, heads, qk_dim, v_dim = 1, 8, 2, 8, 8
    q, k, v, beta, decay = _make_inputs(batch, seq_len, heads, qk_dim, v_dim, dtype=jnp.bfloat16)

    out, _state = gated_delta_rule(q, k, v, beta, decay, chunk_size=8)
    assert out.shape == (batch, seq_len, heads, v_dim)
    assert jnp.all(jnp.isfinite(out))


def test_chunk_fwd_backward():
    batch, seq_len, heads, qk_dim, v_dim = 1, 8, 2, 8, 8
    q, k, v, beta, decay = _make_inputs(batch, seq_len, heads, qk_dim, v_dim, dtype=jnp.float32)

    def loss_fn(q, k, v, beta, decay):
        out, _ = gated_delta_rule(q, k, v, beta, decay, chunk_size=8, use_chunked=True)
        return jnp.sum(out)

    grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v, beta, decay)
    inputs = [q, k, v, beta, decay]
    for i, g in enumerate(grads):
        assert g.shape == inputs[i].shape, f"grad {i} shape mismatch"
        assert jnp.all(jnp.isfinite(g)), f"grad {i} has non-finite values"


def _naive_gdr_reference(q, k, v, beta, decay):
    """Float64 token-by-token gated delta rule (public [B, T, H, D] layout, qk L2-normalised).

    S_t = S_{t-1} * exp(g_t) + k_t (beta_t (v_t - S_{t-1}' k_t))^T,  o_t = S_t^T q_t / sqrt(K).
    """
    import numpy as np

    q, k, v, beta, decay = (np.asarray(jnp.asarray(x, jnp.float32), np.float64) for x in (q, k, v, beta, decay))
    B, T, H, K = q.shape
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) / np.sqrt(K)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    state = np.zeros((B, H, K, v.shape[-1]))
    out = np.zeros(v.shape)
    for t in range(T):
        state = state * np.exp(decay[:, t])[..., None, None]
        pred = np.einsum("bhk,bhkv->bhv", k[:, t], state)
        delta = beta[:, t][..., None] * (v[:, t] - pred)
        state = state + np.einsum("bhk,bhv->bhkv", k[:, t], delta)
        out[:, t] = np.einsum("bhk,bhkv->bhv", q[:, t], state)
    return out


def test_bf16_decay_cumsum_runs_in_fp32():
    """bf16 inputs with strong decay: the in-chunk log-decay cumsum must not be rounded to bf16.

    With g_t ~ -1 over a 64-token chunk the cumsum reaches ~-64..-96 where bf16 spacing is
    0.25-0.5, so exp(g_i - g_j) between neighbours was off by tens of percent.
    """
    import numpy as np

    batch, seq_len, heads, qk_dim, v_dim = 1, 128, 2, 16, 16
    q, k, v, beta, _ = _make_inputs(batch, seq_len, heads, qk_dim, v_dim, dtype=jnp.float32, seed=3)
    decay = -jax.random.uniform(jax.random.PRNGKey(5), (batch, seq_len, heads), minval=0.5, maxval=1.5)
    q, k, v, beta, decay = (x.astype(jnp.bfloat16) for x in (q, k, v, beta, decay))

    out, _ = gated_delta_rule(q, k, v, beta, decay, chunk_size=64)
    ref = _naive_gdr_reference(q, k, v, beta, decay)

    err = np.abs(np.asarray(out, np.float32) - ref).max() / np.abs(ref).max()
    assert err < 0.05, f"relative max error vs float64 recurrence: {err}"
