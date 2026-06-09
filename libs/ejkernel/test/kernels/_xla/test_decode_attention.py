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

import math

import jax
import jax.numpy as jnp

from ejkernel.kernels._xla.decode_attention import decode_attention


def _ref_decode_attention(
    query: jax.Array,
    key_buffer: jax.Array,
    value_buffer: jax.Array,
    req_to_tokens: jax.Array,
    seq_lens: jax.Array,
    *,
    softmax_scale: float,
    page_size: int,
    logits_soft_cap: float | None,
) -> tuple[jax.Array, jax.Array]:
    batch, num_q_heads, head_dim = map(int, query.shape)
    _, num_kv_heads, _ = map(int, key_buffer.shape)
    kv_group = num_q_heads // num_kv_heads

    out = []
    lse = []
    for b in range(batch):
        seq_len = int(seq_lens[b])
        num_pages = (seq_len + page_size - 1) // page_size
        pages = req_to_tokens[b, :num_pages]

        k = key_buffer.reshape(-1, page_size, num_kv_heads, head_dim)[pages]
        v = value_buffer.reshape(-1, page_size, num_kv_heads, head_dim)[pages]
        k = k.reshape(num_pages * page_size, num_kv_heads, head_dim)[:seq_len]
        v = v.reshape(num_pages * page_size, num_kv_heads, head_dim)[:seq_len]

        if kv_group != 1:
            k = jnp.repeat(k, kv_group, axis=1)
            v = jnp.repeat(v, kv_group, axis=1)

        q = query[b].astype(jnp.float32)
        logits = jnp.einsum("hd,khd->hk", q, k.astype(jnp.float32)) * float(softmax_scale)
        if logits_soft_cap is not None and logits_soft_cap > 0:
            logits = float(logits_soft_cap) * jnp.tanh(logits / float(logits_soft_cap))

        w = jax.nn.softmax(logits, axis=-1)
        o = jnp.einsum("hk,khd->hd", w, v.astype(jnp.float32)).astype(query.dtype)
        out.append(o)
        lse.append(jax.scipy.special.logsumexp(logits, axis=-1))

    return jnp.stack(out, axis=0), jnp.stack(lse, axis=0)


def test_decode_attention_xla_matches_reference_basic():
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    batch = 4
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 64
    page_size = 8
    max_pages = 4
    num_kv_splits = 4

    total_pages = batch * max_pages
    key_buffer = jax.random.normal(kk, (total_pages * page_size, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    value_buffer = jax.random.normal(kv, (total_pages * page_size, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    query = jax.random.normal(kq, (batch, num_q_heads, head_dim), dtype=jnp.bfloat16)

    req_to_tokens = jnp.arange(total_pages, dtype=jnp.int32).reshape(batch, max_pages)
    seq_lens = jnp.array([3, 17, 23, 29], dtype=jnp.int32)

    scale = 1.0 / math.sqrt(head_dim)

    out, lse = decode_attention(
        query,
        key_buffer,
        value_buffer,
        req_to_tokens,
        seq_lens,
        softmax_scale=scale,
        num_kv_splits=num_kv_splits,
        page_size=page_size,
        logits_soft_cap=20.0,
    )
    ref_out, ref_lse = _ref_decode_attention(
        query,
        key_buffer,
        value_buffer,
        req_to_tokens,
        seq_lens,
        softmax_scale=scale,
        page_size=page_size,
        logits_soft_cap=20.0,
    )

    assert jnp.allclose(out.astype(jnp.float32), ref_out.astype(jnp.float32), rtol=1e-2, atol=2e-2)
    assert jnp.allclose(lse, ref_lse, rtol=1e-3, atol=1e-3)
