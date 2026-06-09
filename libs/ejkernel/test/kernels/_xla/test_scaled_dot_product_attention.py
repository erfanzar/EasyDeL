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

"""Tests for XLA scaled dot-product attention wrapper."""

import jax
import jax.numpy as jnp

from ejkernel.kernels._xla.scaled_dot_product_attention import scaled_dot_product_attention


def _naive_sdpa(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    softmax_scale: float | None,
    causal: bool,
    sliding_window: int | tuple[int, int] | None,
    attention_mask: jax.Array | None,
    bias: jax.Array | None,
) -> jax.Array:
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    b, tq, hq, d = q.shape
    _, tk, hk, _ = k.shape
    if hk == 1 and hq > 1:
        k = jnp.broadcast_to(k, (b, tk, hq, d))
        v = jnp.broadcast_to(v, (b, tk, hq, d))

    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * jnp.asarray(softmax_scale, q.dtype)

    if sliding_window is not None:
        if isinstance(sliding_window, int):
            left, right = sliding_window, sliding_window
        else:
            left, right = sliding_window
        q_pos = jnp.arange(tq)[:, None]
        k_pos = jnp.arange(tk)[None, :]
        window_mask = (k_pos >= q_pos - left) & (k_pos <= q_pos + right)
        logits = jnp.where(window_mask[None, None, :, :], logits, jnp.finfo(logits.dtype).min)

    if causal:
        causal_mask = jnp.tril(jnp.ones((tq, tk), dtype=bool))
        logits = jnp.where(causal_mask[None, None, :, :], logits, jnp.finfo(logits.dtype).min)

    if attention_mask is not None:
        if attention_mask.ndim == 4:
            m = attention_mask[:, :1]  # (B, 1, Q, K)
        else:
            raise ValueError("attention_mask must be rank-4 in this test")
        logits = jnp.where(m, logits, jnp.finfo(logits.dtype).min)

    if bias is not None:
        logits = logits + bias

    weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(q.dtype)
    out = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return out


def test_matches_naive_basic():
    key = jax.random.PRNGKey(0)
    key, kq, kk, kv = jax.random.split(key, 4)

    b, t, h, d = 2, 16, 4, 8
    q = jax.random.normal(kq, (b, t, h, d), dtype=jnp.float32)
    k = jax.random.normal(kk, (b, t, h, d), dtype=jnp.float32)
    v = jax.random.normal(kv, (b, t, h, d), dtype=jnp.float32)

    out = scaled_dot_product_attention(q, k, v)
    ref = _naive_sdpa(q, k, v, softmax_scale=None, causal=False, sliding_window=None, attention_mask=None, bias=None)

    assert out.shape == ref.shape
    assert jnp.allclose(out, ref, rtol=5e-4, atol=5e-4)


def test_matches_naive_causal_and_window_with_mask_and_bias_init():
    key = jax.random.PRNGKey(1)
    key, kq, kk, kv, kb = jax.random.split(key, 5)

    b, t, h, d = 1, 12, 4, 8
    q = jax.random.normal(kq, (b, t, h, d), dtype=jnp.float32)
    k = jax.random.normal(kk, (b, t, h, d), dtype=jnp.float32)
    v = jax.random.normal(kv, (b, t, h, d), dtype=jnp.float32)

    # Mask out the last two keys.
    attention_mask = jnp.ones((b, 1, t, t), dtype=bool)
    attention_mask = attention_mask.at[:, :, :, -2:].set(False)

    bias = jax.random.normal(kb, (b, h, t, t), dtype=jnp.float32) * 0.01

    out = scaled_dot_product_attention(
        q,
        k,
        v,
        attention_mask=attention_mask,
        init_bias=lambda: bias,
        causal=True,
        sliding_window=(3, 0),
    )
    ref = _naive_sdpa(
        q,
        k,
        v,
        softmax_scale=None,
        causal=True,
        sliding_window=(3, 0),
        attention_mask=attention_mask,
        bias=bias,
    )

    assert jnp.allclose(out, ref, rtol=8e-4, atol=8e-4)


def test_supports_mqa():
    key = jax.random.PRNGKey(2)
    key, kq, kk, kv = jax.random.split(key, 4)

    b, t, hq, hk, d = 1, 8, 4, 1, 8
    q = jax.random.normal(kq, (b, t, hq, d), dtype=jnp.float32)
    k = jax.random.normal(kk, (b, t, hk, d), dtype=jnp.float32)
    v = jax.random.normal(kv, (b, t, hk, d), dtype=jnp.float32)

    out = scaled_dot_product_attention(q, k, v, causal=True)
    assert out.shape == (b, t, hq, d)
    assert jnp.isfinite(out).all()
