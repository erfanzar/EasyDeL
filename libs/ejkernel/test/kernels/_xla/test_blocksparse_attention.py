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
