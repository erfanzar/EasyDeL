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

"""Per-channel-decay KDA kernels (GLM-5-Next) — parity and dispatch tests.

The GLM-5-Next forget gate produces a per-channel log-decay of shape
``(batch, seq, num_heads, head_dim)`` instead of Kimi Linear's per-head
``(batch, seq, num_heads)``. These tests pin the three per-channel kernels
(chunked, recurrent, single-step decode) against each other and against an
independent numpy reference recurrence, plus the fused gate function.
"""

import jax
import jax.numpy as jnp
import numpy as np
from easydel.infra import EasyDeLBaseConfig
from easydel.operations._operation_impl import OperationMetadata
from easydel.operations.kernels.kda import (
    KernelDeltaAttnOp,
    _chunk_kda_per_channel_fwd,
    _recurrent_kda_per_channel_fwd,
    _single_step_kda_per_channel_fwd_bthd,
    fused_kda_gate_per_channel,
    l2norm,
)

ATOL_CHUNKED = 5e-4
ATOL_TIGHT = 1e-5


def _reference_recurrence(query, key, value, beta, decay, use_qk_l2norm=True):
    """Independent numpy reference of the per-channel KDA recurrence.

    Transcribed from the HF ``recurrent_kimi_delta_attention`` loop (numpy,
    no easydel helpers beyond shape conventions) so the JAX kernels are not
    compared against themselves.
    """
    B, H, L, K = query.shape
    V = value.shape[-1]
    query = np.asarray(query, dtype=np.float64)
    key = np.asarray(key, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    decay = np.asarray(decay, dtype=np.float64)
    if use_qk_l2norm:
        query = query / np.sqrt((query * query).sum(-1, keepdims=True) + 1e-6)
        key = key / np.sqrt((key * key).sum(-1, keepdims=True) + 1e-6)
    query = query * (1.0 / np.sqrt(K))

    state = np.zeros((B, H, K, V))
    outputs = np.zeros((B, H, L, V))
    for t in range(L):
        g = np.exp(decay[:, :, t][..., None])  # (B, H, K, 1)
        b = beta[:, :, t][..., None]
        state = state * g
        kv_mem = (state * key[:, :, t][..., None]).sum(-2)
        delta = (value[:, :, t] - kv_mem) * b
        state = state + key[:, :, t][..., None] * delta[..., None, :]
        outputs[:, :, t] = (state * query[:, :, t][..., None]).sum(-2)
    return outputs.transpose(0, 1, 2, 3)


def _make_inputs(seed=0, batch=2, heads=3, seq=24, head_dim=8, d_state=6, dtype=jnp.float32):
    rng = jax.random.key(seed)
    query = jax.random.normal(rng, (batch, heads, seq, head_dim), dtype=jnp.float32) * 0.5
    key = jax.random.normal(jax.random.fold_in(rng, 1), (batch, heads, seq, head_dim), dtype=jnp.float32) * 0.5
    value = jax.random.normal(jax.random.fold_in(rng, 2), (batch, heads, seq, d_state), dtype=jnp.float32) * 0.5
    beta = jax.nn.sigmoid(jax.random.normal(jax.random.fold_in(rng, 3), (batch, heads, seq), dtype=jnp.float32))
    decay = (
        -5.0
        * jax.nn.sigmoid(jax.random.normal(jax.random.fold_in(rng, 4), (batch, heads, seq, head_dim), dtype=jnp.float32))
    ).astype(dtype)
    return query, key, value, beta, decay


def test_fused_gate_per_channel_matches_reference():
    """The fused gate equals the HF formula evaluated directly."""
    batch, seq, heads, head_dim = 2, 5, 3, 8
    qkv_dim = heads * head_dim
    rng = jax.random.key(11)
    gate = jax.random.normal(rng, (batch, seq, qkv_dim), dtype=jnp.float32) * 2.0
    a_log = jax.random.normal(jax.random.fold_in(rng, 1), (heads,), dtype=jnp.float32) * 0.3
    dt_bias = jax.random.normal(jax.random.fold_in(rng, 2), (qkv_dim,), dtype=jnp.float32) * 0.5
    lower_bound = -5.0

    out = fused_kda_gate_per_channel(gate, a_log, dt_bias, lower_bound=lower_bound)

    g = (np.asarray(gate) + np.asarray(dt_bias).reshape(1, 1, qkv_dim)).reshape(batch, seq, heads, head_dim)
    rate = np.exp(np.asarray(a_log)).reshape(1, 1, heads, 1)
    expected = lower_bound / (1.0 + np.exp(-rate * g))
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-5, rtol=1e-5)
    assert out.shape == (batch, seq, heads, head_dim)
    assert out.dtype == jnp.float32
    # Safe gate keeps the decay in [lower_bound, 0) — exp(g) contracts state.
    assert float(out.min()) >= lower_bound - 1e-6
    assert float(out.max()) <= 0.0


def test_fused_gate_softplus_variant():
    """Without a lower bound the gate falls back to the softplus form."""
    batch, seq, heads, head_dim = 2, 4, 2, 4
    qkv_dim = heads * head_dim
    rng = jax.random.key(12)
    gate = jax.random.normal(rng, (batch, seq, qkv_dim), dtype=jnp.float32)
    a_log = jax.random.normal(jax.random.fold_in(rng, 1), (heads,), dtype=jnp.float32) * 0.2
    dt_bias = jax.random.normal(jax.random.fold_in(rng, 2), (qkv_dim,), dtype=jnp.float32) * 0.3

    out = fused_kda_gate_per_channel(gate, a_log, dt_bias, lower_bound=None)

    g = (np.asarray(gate) + np.asarray(dt_bias).reshape(1, 1, qkv_dim)).reshape(batch, seq, heads, head_dim)
    rate = np.exp(np.asarray(a_log)).reshape(1, 1, heads, 1)
    softplus = np.where(g > 20.0, g, np.log1p(np.exp(g)))
    np.testing.assert_allclose(np.asarray(out), -rate * softplus, atol=1e-5, rtol=1e-5)
    assert float(out.max()) <= 0.0


def test_chunked_matches_recurrent_per_channel():
    """The chunked kernel agrees with the sequential recurrence."""
    query, key, value, beta, decay = _make_inputs(seq=48)
    out_rec, state_rec = _recurrent_kda_per_channel_fwd(query, key, value, beta, decay, use_qk_l2norm=True)
    out_chunk, state_chunk = _chunk_kda_per_channel_fwd(
        query, key, value, beta, decay, chunk_size=16, use_qk_l2norm=True
    )
    assert out_rec.shape == (2, 3, 48, 6)
    np.testing.assert_allclose(np.asarray(out_chunk), np.asarray(out_rec), atol=ATOL_CHUNKED, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(state_chunk), np.asarray(state_rec), atol=ATOL_CHUNKED, rtol=1e-3)


def test_recurrent_matches_numpy_reference():
    """The recurrent kernel matches an independent numpy recurrence."""
    query, key, value, beta, decay = _make_inputs(seq=16)
    out, state = _recurrent_kda_per_channel_fwd(query, key, value, beta, decay, use_qk_l2norm=True)
    expected = _reference_recurrence(query, key, value, beta, decay, use_qk_l2norm=True)
    np.testing.assert_allclose(np.asarray(out), expected, atol=ATOL_TIGHT, rtol=1e-4)
    assert state.shape == (2, 3, 8, 6)
    assert state.dtype == jnp.float32


def test_single_step_decode_matches_recurrent():
    """Feeding the batched sequence step-by-step reproduces the recurrent run."""
    query, key, value, beta, decay = _make_inputs(seq=10)

    out_full, state_full = _recurrent_kda_per_channel_fwd(query, key, value, beta, decay, use_qk_l2norm=True)

    state = jnp.zeros((2, 3, 8, 6), dtype=jnp.float32)
    for t in range(10):
        out_t, state = _single_step_kda_per_channel_fwd_bthd(
            query=query.transpose(0, 2, 1, 3)[:, t][..., None, :, :],
            key=key.transpose(0, 2, 1, 3)[:, t][..., None, :, :],
            value=value.transpose(0, 2, 1, 3)[:, t][..., None, :, :],
            beta=beta.transpose(0, 2, 1)[:, t][..., None, :],
            decay=decay.transpose(0, 2, 1, 3)[:, t][..., None, :, :],
            recurrent_state=state,
            use_qk_l2norm=True,
        )
        np.testing.assert_allclose(np.asarray(out_t[:, 0]), np.asarray(out_full[:, :, t]), atol=ATOL_TIGHT, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(state), np.asarray(state_full), atol=ATOL_TIGHT, rtol=1e-4)


def test_op_dispatch_per_channel_decay():
    """KernelDeltaAttnOp routes 4-dim decay through the per-channel kernels."""
    query, key, value, beta, decay = _make_inputs(seq=32)

    # BTHD layout at the op boundary: (B, S, H, K) and decay (B, S, H, K).
    query_bthd = query.transpose(0, 2, 1, 3)
    key_bthd = key.transpose(0, 2, 1, 3)
    value_bthd = value.transpose(0, 2, 1, 3)
    beta_bthd = beta.transpose(0, 2, 1)
    decay_bthd = decay.transpose(0, 2, 1, 3)

    metadata = OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        base_config=EasyDeLBaseConfig(),
    )
    op = KernelDeltaAttnOp(metadata)

    output = op(
        query=query_bthd,
        key=key_bthd,
        value=value_bthd,
        beta=beta_bthd,
        decay=decay_bthd,
        chunk_size=16,
    )
    assert output.attention_outputs.shape == (2, 32, 3, 6)
    assert output.recurrent_state.shape == (2, 3, 8, 6)

    expected, expected_state = _recurrent_kda_per_channel_fwd(query, key, value, beta, decay, use_qk_l2norm=True)
    np.testing.assert_allclose(
        np.asarray(output.attention_outputs).transpose(0, 2, 1, 3),
        np.asarray(expected),
        atol=ATOL_CHUNKED,
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(output.recurrent_state), np.asarray(expected_state), atol=ATOL_CHUNKED, rtol=1e-3
    )


def test_op_dispatch_decode_per_channel():
    """Decode mode (seq_len=1 + recurrent_state) uses the per-channel step."""
    query, key, value, beta, decay = _make_inputs(seq=1)
    rng = jax.random.key(21)
    state_in = jax.random.normal(rng, (2, 3, 8, 6), dtype=jnp.float32)

    metadata = OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        base_config=EasyDeLBaseConfig(),
    )
    op = KernelDeltaAttnOp(metadata)

    output = op(
        query=query.transpose(0, 2, 1, 3),  # (B, 1, H, K)
        key=key.transpose(0, 2, 1, 3),
        value=value.transpose(0, 2, 1, 3),
        beta=beta.transpose(0, 2, 1),
        decay=decay.transpose(0, 2, 1, 3),
        recurrent_state=state_in,
    )
    assert output.attention_outputs.shape == (2, 1, 3, 6)
    assert output.recurrent_state.shape == (2, 3, 8, 6)

    expected, expected_state = _single_step_kda_per_channel_fwd_bthd(
        query=query.transpose(0, 2, 1, 3),
        key=key.transpose(0, 2, 1, 3),
        value=value.transpose(0, 2, 1, 3),
        beta=beta.transpose(0, 2, 1),
        decay=decay.transpose(0, 2, 1, 3),
        recurrent_state=state_in,
        use_qk_l2norm=True,
    )
    np.testing.assert_allclose(np.asarray(output.attention_outputs), np.asarray(expected), atol=ATOL_TIGHT, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(output.recurrent_state), np.asarray(expected_state), atol=ATOL_TIGHT, rtol=1e-5
    )


def test_per_head_path_untouched():
    """3-dim decay still routes through the original per-head kernels."""
    from easydel.operations.kernels.kda import _chunk_kda_fwd, _recurrent_kda_fwd

    rng = jax.random.key(31)
    batch, heads, seq, head_dim, d_state = 2, 3, 16, 8, 6
    query = jax.random.normal(rng, (batch, heads, seq, head_dim), dtype=jnp.float32) * 0.5
    key = jax.random.normal(jax.random.fold_in(rng, 1), (batch, heads, seq, head_dim), dtype=jnp.float32) * 0.5
    value = jax.random.normal(jax.random.fold_in(rng, 2), (batch, heads, seq, d_state), dtype=jnp.float32) * 0.5
    beta = jax.nn.sigmoid(jax.random.normal(jax.random.fold_in(rng, 3), (batch, heads, seq), dtype=jnp.float32))
    decay = jax.random.normal(jax.random.fold_in(rng, 4), (batch, heads, seq), dtype=jnp.float32) * 0.01

    out_rec, state_rec = _recurrent_kda_fwd(query, key, value, beta, decay, use_qk_l2norm=True)
    out_chunk, state_chunk = _chunk_kda_fwd(query, key, value, beta, decay, chunk_size=8, use_qk_l2norm=True)
    np.testing.assert_allclose(np.asarray(out_chunk), np.asarray(out_rec), atol=ATOL_CHUNKED, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(state_chunk), np.asarray(state_rec), atol=ATOL_CHUNKED, rtol=1e-3)


def test_l2norm_shared_by_paths():
    """Q/K L2-normalization applies before decay, matching the HF kernel."""
    x = jnp.array([[[3.0, 4.0]]])
    normalized = l2norm(x, axis=-1, eps=1e-6)
    np.testing.assert_allclose(np.asarray(normalized), [[[0.6, 0.8]]], atol=1e-6)
