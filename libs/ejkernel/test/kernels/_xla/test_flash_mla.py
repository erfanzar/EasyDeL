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

"""Tests for XLA Flash Multi-head Latent Attention (MLA) implementation."""

import math

import jax
import jax.numpy as jnp

from ejkernel.kernels._xla.flash_mla import flash_mla


def _repeat_kv_for_gqa(x: jax.Array, q_heads: int) -> jax.Array:
    kv_heads = int(x.shape[2])
    if kv_heads == q_heads:
        return x
    reps = q_heads // kv_heads
    return jnp.repeat(x, reps, axis=2)


def _mla_reference(
    *,
    query: jax.Array,
    key_value: jax.Array,
    w_kc: jax.Array,
    w_vc: jax.Array,
    b_q: jax.Array | None,
    b_k: jax.Array | None,
    softmax_scale: float | None,
    causal: bool,
) -> jax.Array:
    q_f32 = query.astype(jnp.float32)
    kv_f32 = key_value.astype(jnp.float32)
    w_kc_f32 = w_kc.astype(jnp.float32)
    w_vc_f32 = w_vc.astype(jnp.float32)

    _b, t, q_heads, _q_dim = q_f32.shape
    d_nope = int(w_kc_f32.shape[2])

    k_nope = jnp.einsum("btr,rhd->bthd", kv_f32, w_kc_f32, optimize=True)
    v = jnp.einsum("btr,rhd->bthd", kv_f32, w_vc_f32, optimize=True)
    k_nope = _repeat_kv_for_gqa(k_nope, q_heads)
    v = _repeat_kv_for_gqa(v, q_heads)

    if b_k is None:
        logits = jnp.einsum("bqhd,bkhd->bhqk", q_f32, k_nope, optimize=True)
        d_scale = float(d_nope)
    elif b_q is None:
        rope_dim = int(b_k.shape[2])
        q_nope = q_f32[..., :d_nope]
        q_rope = q_f32[..., d_nope : d_nope + rope_dim]
        logits_nope = jnp.einsum("bqhd,bkhd->bhqk", q_nope, k_nope, optimize=True)
        logits_rope = jnp.einsum("bqhd,bkd->bhqk", q_rope, b_k.astype(jnp.float32), optimize=True)
        logits = logits_nope + logits_rope
        d_scale = float(d_nope + rope_dim)
    else:
        logits_nope = jnp.einsum("bqhd,bkhd->bhqk", q_f32, k_nope, optimize=True)
        logits_rope = jnp.einsum("bqd,bkd->bqk", b_q.astype(jnp.float32), b_k.astype(jnp.float32), optimize=True)
        logits = logits_nope + logits_rope[:, None, :, :]
        d_scale = float(d_nope + int(b_k.shape[2]))

    scale = float(softmax_scale) if softmax_scale is not None else float(1.0 / math.sqrt(d_scale))
    logits = logits * jnp.asarray(scale, dtype=jnp.float32)

    if causal:
        q_idx = jnp.arange(t)[:, None]
        k_idx = jnp.arange(t)[None, :]
        mask = k_idx <= q_idx
        logits = jnp.where(mask[None, None, :, :], logits, jnp.finfo(logits.dtype).min)

    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bhqk,bkhd->bqhd", weights, v, optimize=True)
    return out.astype(query.dtype)


def test_flash_mla_matches_reference_rope_gqa_causal():
    key = jax.random.PRNGKey(0)
    key, kq, kkv, kwk, kwv, krope = jax.random.split(key, 6)

    b, t, q_heads, kv_heads = 2, 8, 4, 2
    kv_lora_rank = 6
    d_nope, rope_dim = 12, 4
    head_dim = d_nope + rope_dim

    query = jax.random.normal(kq, (b, t, q_heads, head_dim), dtype=jnp.float32)
    key_value = jax.random.normal(kkv, (b, t, kv_lora_rank), dtype=jnp.float32)
    w_kc = jax.random.normal(kwk, (kv_lora_rank, kv_heads, d_nope), dtype=jnp.float32)
    w_vc = jax.random.normal(kwv, (kv_lora_rank, kv_heads, head_dim), dtype=jnp.float32)
    b_k = jax.random.normal(krope, (b, t, rope_dim), dtype=jnp.float32)

    out = flash_mla(query, key_value, w_kc, w_vc, b_k=b_k, causal=True)
    ref = _mla_reference(
        query=query,
        key_value=key_value,
        w_kc=w_kc,
        w_vc=w_vc,
        b_q=None,
        b_k=b_k,
        softmax_scale=None,
        causal=True,
    )

    assert out.shape == ref.shape
    assert jnp.allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_flash_mla_matches_reference_shared_rope_term():
    key = jax.random.PRNGKey(1)
    key, kq, kkv, kwk, kwv, kbq, kbk = jax.random.split(key, 7)

    b, t, q_heads, kv_heads = 1, 7, 3, 3
    kv_lora_rank = 5
    d_nope, rope_dim = 8, 4

    query = jax.random.normal(kq, (b, t, q_heads, d_nope), dtype=jnp.float32)
    key_value = jax.random.normal(kkv, (b, t, kv_lora_rank), dtype=jnp.float32)
    w_kc = jax.random.normal(kwk, (kv_lora_rank, kv_heads, d_nope), dtype=jnp.float32)
    w_vc = jax.random.normal(kwv, (kv_lora_rank, kv_heads, d_nope), dtype=jnp.float32)
    b_q = jax.random.normal(kbq, (b, t, rope_dim), dtype=jnp.float32)
    b_k = jax.random.normal(kbk, (b, t, rope_dim), dtype=jnp.float32)

    out = flash_mla(query, key_value, w_kc, w_vc, b_q=b_q, b_k=b_k, causal=False)
    ref = _mla_reference(
        query=query,
        key_value=key_value,
        w_kc=w_kc,
        w_vc=w_vc,
        b_q=b_q,
        b_k=b_k,
        softmax_scale=None,
        causal=False,
    )

    assert out.shape == ref.shape
    assert jnp.allclose(out, ref, rtol=1e-4, atol=1e-4)
