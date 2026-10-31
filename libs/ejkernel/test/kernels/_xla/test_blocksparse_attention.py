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

"""Tests for XLA block-sparse attention fallback implementation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.attention import attention
from ejkernel.kernels._xla.blocksparse_attention import blocksparse_attention


def _bhtd_to_bthd(x: jax.Array) -> jax.Array:
    return jnp.transpose(x, (0, 2, 1, 3))


@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        (jnp.float32, 2e-4, 2e-4),
        (jnp.bfloat16, 2e-2, 2e-2),
    ],
)
@pytest.mark.parametrize("use_jit", [False, True])
def test_full_attention_matches_vanilla(dtype, rtol, atol, use_jit):
    key = jax.random.PRNGKey(0)
    key, kq, kk, kv = jax.random.split(key, 4)

    batch, num_heads, seq_len, head_dim = 2, 4, 16, 8
    q = jax.random.normal(kq, (batch, num_heads, seq_len, head_dim), dtype=dtype)
    k = jax.random.normal(kk, (batch, num_heads, seq_len, head_dim), dtype=dtype)
    v = jax.random.normal(kv, (batch, num_heads, seq_len, head_dim), dtype=dtype)

    if use_jit:
        bs_fn = jax.jit(lambda q, k, v: blocksparse_attention(q, k, v, causal=False))
        out_bs = bs_fn(q, k, v)
    else:
        out_bs = blocksparse_attention(q, k, v, causal=False)

    def _vanilla(q_bhtd, k_bhtd, v_bhtd):
        out, _ = attention(_bhtd_to_bthd(q_bhtd), _bhtd_to_bthd(k_bhtd), _bhtd_to_bthd(v_bhtd), dtype=dtype)
        return out

    vanilla_fn = jax.jit(_vanilla) if use_jit else _vanilla
    out_ref = vanilla_fn(q, k, v)

    assert out_bs.shape == (batch, num_heads, seq_len, head_dim)
    assert jnp.isfinite(out_bs).all()
    assert jnp.allclose(_bhtd_to_bthd(out_bs), out_ref, rtol=rtol, atol=atol)


def test_segment_ids_match_attention_mask():
    key = jax.random.PRNGKey(1)
    key, kq, kk, kv = jax.random.split(key, 4)

    batch, num_heads, seq_len, head_dim = 1, 2, 12, 8
    q = jax.random.normal(kq, (batch, num_heads, seq_len, head_dim), dtype=jnp.float32)
    k = jax.random.normal(kk, (batch, num_heads, seq_len, head_dim), dtype=jnp.float32)
    v = jax.random.normal(kv, (batch, num_heads, seq_len, head_dim), dtype=jnp.float32)

    # Two segments: [0..5] and [6..11]
    seg = jnp.array([[0] * 6 + [1] * 6], dtype=jnp.int32)
    out_bs = blocksparse_attention(q, k, v, q_segment_ids=seg, kv_segment_ids=seg, causal=False)

    mask = seg[:, :, None] == seg[:, None, :]  # (B, T, T)
    mask = mask[:, None, :, :]  # (B, 1, T, T)

    out_ref, _ = attention(_bhtd_to_bthd(q), _bhtd_to_bthd(k), _bhtd_to_bthd(v), attention_mask=mask, dtype=jnp.float32)

    assert jnp.allclose(_bhtd_to_bthd(out_bs), out_ref, rtol=2e-4, atol=2e-4)


def test_sliding_window_and_causal_match_vanilla():
    key = jax.random.PRNGKey(2)
    key, kq, kk, kv = jax.random.split(key, 4)

    batch, num_heads, seq_len, head_dim = 1, 4, 24, 8
    q = jax.random.normal(kq, (batch, num_heads, seq_len, head_dim), dtype=jnp.float32)
    k = jax.random.normal(kk, (batch, num_heads, seq_len, head_dim), dtype=jnp.float32)
    v = jax.random.normal(kv, (batch, num_heads, seq_len, head_dim), dtype=jnp.float32)

    window = 4
    out_bs = blocksparse_attention(q, k, v, causal=True, sliding_window=(window, 0))

    out_ref, _ = attention(
        _bhtd_to_bthd(q),
        _bhtd_to_bthd(k),
        _bhtd_to_bthd(v),
        causal=True,
        sliding_window=(window, 0),
        dtype=jnp.float32,
    )

    assert jnp.isfinite(out_bs).all()
    assert jnp.allclose(_bhtd_to_bthd(out_bs), out_ref, rtol=3e-4, atol=3e-4)


def test_softmax_aux_per_head_matches_reference():
    key = jax.random.PRNGKey(4)
    key, kq, kk, kv, ks = jax.random.split(key, 5)

    batch, num_heads, seq_len, head_dim = 2, 4, 16, 8
    q = jax.random.normal(kq, (batch, num_heads, seq_len, head_dim), dtype=jnp.float32)
    k = jax.random.normal(kk, (batch, num_heads, seq_len, head_dim), dtype=jnp.float32)
    v = jax.random.normal(kv, (batch, num_heads, seq_len, head_dim), dtype=jnp.float32)
    softmax_aux = jax.random.normal(ks, (num_heads,), dtype=jnp.float32)

    out_bs = blocksparse_attention(q, k, v, causal=False, softmax_aux=softmax_aux)

    softmax_scale = head_dim**-0.5
    logits = jnp.einsum("bhtd,bhkd->bhtk", q * softmax_scale, k, optimize=True)
    sinks = jnp.broadcast_to(softmax_aux[None, :, None, None], (batch, num_heads, seq_len, 1))
    combined = jnp.concatenate([logits, sinks], axis=-1)
    probs = jax.nn.softmax(combined.astype(jnp.float32), axis=-1)
    weights = probs[..., :seq_len]
    out_ref = jnp.einsum("bhtk,bhkd->bhtd", weights, v, optimize=True)

    assert jnp.allclose(out_bs, out_ref, rtol=2e-4, atol=2e-4)


def test_offset_query_positions_are_not_remasked_by_local_triangle():
    """Causal masking must follow the supplied positions, not array indices (r10 S04-1).

    One query at position 2 sees keys at positions 0, 1, 2. With zero Q/K the weights are uniform,
    so the output is the mean value 2.0; a second local-index ``tril`` would keep only key 0 (1.0).
    """
    head_dim = 8
    q = jnp.zeros((1, 1, 1, head_dim), dtype=jnp.float32)
    k = jnp.zeros((1, 1, 3, head_dim), dtype=jnp.float32)
    v = jnp.broadcast_to(jnp.asarray([1.0, 2.0, 3.0])[None, None, :, None], (1, 1, 3, head_dim))
    out = blocksparse_attention(
        q,
        k,
        v,
        q_positions=jnp.asarray([[2]], dtype=jnp.int32),
        kv_positions=jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
        causal=True,
    )
    assert jnp.allclose(out, 2.0, atol=1e-5), out


def test_disabled_sinks_have_zero_not_nan_gradient():
    """All--inf (disabled) sink rows must reduce with a zero, finite gradient (r13 S03-1 / S04-1).

    The shared reducer ``_sinks.reduce_sinks_to_per_head`` differentiated disabled rows through
    ``log(0)`` and returned NaN even for a zero incoming cotangent.
    """
    q = jnp.zeros((1, 1, 1, 1), jnp.float32)
    k = jnp.zeros((1, 1, 1, 1), jnp.float32)
    v = jnp.ones((1, 1, 1, 1), jnp.float32)

    def grad_for(q, k, v, aux):
        return jax.grad(lambda a: jnp.sum(blocksparse_attention(q, k, v, causal=False, softmax_aux=a)))(aux)

    # Shared disabled sinks (Hq == Hkv == 1): output is one, sink gradient is exactly zero.
    disabled = jnp.full((2,), -jnp.inf, jnp.float32)
    out = blocksparse_attention(q, k, v, causal=False, softmax_aux=disabled)
    assert jnp.allclose(out, 1.0, atol=1e-6), out
    g = grad_for(q, k, v, disabled)
    assert jnp.all(jnp.isfinite(g)) and float(jnp.max(jnp.abs(g))) == 0.0, g

    # Mixed per-head layout: disabled row -> 0, -inf entry of a live row -> 0, live entry -> nonzero.
    kq, kk, kv = jax.random.split(jax.random.PRNGKey(9), 3)
    q2 = jax.random.normal(kq, (1, 2, 8, 8), jnp.float32)
    k2 = jax.random.normal(kk, (1, 2, 8, 8), jnp.float32)
    v2 = jax.random.normal(kv, (1, 2, 8, 8), jnp.float32)
    mixed = jnp.asarray([[-jnp.inf, -jnp.inf], [0.5, -jnp.inf]], jnp.float32)
    g = grad_for(q2, k2, v2, mixed)
    assert jnp.all(jnp.isfinite(g)), g
    assert float(jnp.max(jnp.abs(g[0]))) == 0.0 and float(g[1, 1]) == 0.0, g
    assert abs(float(g[1, 0])) > 1e-4, g


def test_sink_reducer_only_disables_all_negative_infinity_rows():
    """+inf / NaN sink rows are not 'disabled': they keep their logsumexp value (r14 S03-1)."""
    from ejkernel.kernels._sinks import reduce_sinks_to_per_head

    def reduce(x):
        return np.asarray(reduce_sinks_to_per_head(jnp.asarray(x, jnp.float32), num_q_heads=2, num_kv_heads=2))

    assert np.all(reduce([[np.inf], [np.inf]]) == np.inf)
    assert np.all(np.isnan(reduce([[np.nan], [np.nan]])))
    assert np.all(reduce([[np.inf, -np.inf], [np.inf, -np.inf]]) == np.inf)
    assert np.all(reduce([[-np.inf, -np.inf], [-np.inf, -np.inf]]) == -np.inf)
    # 2-D (Hq, 1) and the equivalent 1-D per-head layout agree, including non-finite rows.
    np.testing.assert_array_equal(reduce([[np.inf], [0.5]]), reduce([np.inf, 0.5]))


def test_gradients_are_finite():
    key = jax.random.PRNGKey(3)
    key, kq, kk, kv = jax.random.split(key, 4)

    batch, num_heads, seq_len, head_dim = 1, 2, 8, 8
    q = jax.random.normal(kq, (batch, num_heads, seq_len, head_dim), dtype=jnp.float32)
    k = jax.random.normal(kk, (batch, num_heads, seq_len, head_dim), dtype=jnp.float32)
    v = jax.random.normal(kv, (batch, num_heads, seq_len, head_dim), dtype=jnp.float32)

    def loss_fn(q, k, v):
        out = blocksparse_attention(q, k, v, causal=True, sliding_window=3)
        return jnp.sum(out**2)

    dq, dk, dv = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
    assert dq.shape == q.shape and dk.shape == k.shape and dv.shape == v.shape
    assert jnp.isfinite(dq).all()
    assert jnp.isfinite(dk).all()
    assert jnp.isfinite(dv).all()
