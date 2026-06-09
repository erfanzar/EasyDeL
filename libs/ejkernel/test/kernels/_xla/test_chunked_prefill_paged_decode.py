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

from ejkernel.kernels._xla.chunked_prefill_paged_decode import chunked_prefill_paged_decode


def _write_tokens_to_cache(
    cache: jax.Array,
    *,
    tokens: jax.Array,
    seq_idx: int,
    block_tables: jax.Array,
    block_size: int,
    start_pos: int,
) -> jax.Array:
    for i in range(int(tokens.shape[0])):
        pos = start_pos + i
        blk = int(pos // block_size)
        within = int(pos % block_size)
        phys = int(block_tables[seq_idx, blk])
        cache = cache.at[phys, within].set(tokens[i])
    return cache


def _ref_chunked_prefill(
    queries: jax.Array,
    k_full: list[jax.Array],
    v_full: list[jax.Array],
    q_lens: list[int],
    context_lens: list[int],
    *,
    softmax_scale: float,
    sliding_window: int | None,
    logits_soft_cap: float | None,
) -> jax.Array:
    num_seqs = len(q_lens)
    outs = []
    cursor = 0
    for s in range(num_seqs):
        q_len = int(q_lens[s])
        ctx_len = int(context_lens[s])
        int(k_full[s].shape[0])
        q = queries[cursor : cursor + q_len]
        cursor += q_len

        k = k_full[s]
        v = v_full[s]
        if q.shape[1] != k.shape[1]:
            rep = q.shape[1] // k.shape[1]
            k = jnp.repeat(k, rep, axis=1)
            v = jnp.repeat(v, rep, axis=1)

        for t in range(q_len):
            abs_pos = ctx_len + t
            right = abs_pos + 1
            left = 0
            if sliding_window is not None and sliding_window > 0:
                left = max(0, right - int(sliding_window))

            logits = jnp.einsum("hd,khd->hk", q[t].astype(jnp.float32), k[left:right].astype(jnp.float32)) * float(
                softmax_scale
            )
            if logits_soft_cap is not None and logits_soft_cap > 0:
                logits = float(logits_soft_cap) * jnp.tanh(logits / float(logits_soft_cap))
            w = jax.nn.softmax(logits, axis=-1)
            o = jnp.einsum("hk,khd->hd", w, v[left:right].astype(jnp.float32)).astype(queries.dtype)
            outs.append(o)

    out = jnp.stack(outs, axis=0)
    assert out.shape == queries.shape
    return out


def test_chunked_prefill_paged_decode_xla_updates_cache_and_matches_reference():
    key = jax.random.PRNGKey(0)
    kq, kk_ctx, kv_ctx, kk_new, kv_new = jax.random.split(key, 5)

    num_seqs = 3
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 32
    block_size = 8

    q_lens = [4, 1, 3]
    context_lens = [5, 2, 0]
    kv_lens = jnp.array([c + q for c, q in zip(context_lens, q_lens, strict=True)], dtype=jnp.int32)

    cu = [0]
    for q in q_lens:
        cu.append(cu[-1] + int(q))
    query_start_loc = jnp.array(cu, dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])

    max_kv = int(kv_lens.max())
    max_blocks_per_seq = (max_kv + block_size - 1) // block_size
    num_blocks_total = num_seqs * max_blocks_per_seq
    block_tables = jnp.arange(num_blocks_total, dtype=jnp.int32).reshape(num_seqs, max_blocks_per_seq)

    key_cache = jnp.zeros((num_blocks_total, block_size, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    value_cache = jnp.zeros_like(key_cache)

    ctx_keys: list[jax.Array] = []
    ctx_vals: list[jax.Array] = []
    for s, ctx_len in enumerate(context_lens):
        k_ctx = jax.random.normal(jax.random.fold_in(kk_ctx, s), (ctx_len, num_kv_heads, head_dim), dtype=jnp.bfloat16)
        v_ctx = jax.random.normal(jax.random.fold_in(kv_ctx, s), (ctx_len, num_kv_heads, head_dim), dtype=jnp.bfloat16)
        ctx_keys.append(k_ctx)
        ctx_vals.append(v_ctx)
        key_cache = _write_tokens_to_cache(
            key_cache, tokens=k_ctx, seq_idx=s, block_tables=block_tables, block_size=block_size, start_pos=0
        )
        value_cache = _write_tokens_to_cache(
            value_cache, tokens=v_ctx, seq_idx=s, block_tables=block_tables, block_size=block_size, start_pos=0
        )

    queries = jax.random.normal(kq, (total_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16)
    keys = jax.random.normal(kk_new, (total_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    values = jax.random.normal(kv_new, (total_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)

    scale = 1.0 / math.sqrt(head_dim)
    out, new_kc, new_vc = chunked_prefill_paged_decode(
        queries,
        keys,
        values,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        softmax_scale=scale,
        causal=True,
        sliding_window=16,
        logits_soft_cap=20.0,
    )

    k_full: list[jax.Array] = []
    v_full: list[jax.Array] = []
    cursor = 0
    for s, q_len in enumerate(q_lens):
        k_chunk = keys[cursor : cursor + q_len]
        v_chunk = values[cursor : cursor + q_len]
        cursor += q_len
        k_full.append(jnp.concatenate([ctx_keys[s], k_chunk], axis=0))
        v_full.append(jnp.concatenate([ctx_vals[s], v_chunk], axis=0))

    ref = _ref_chunked_prefill(
        queries,
        k_full,
        v_full,
        q_lens,
        context_lens,
        softmax_scale=scale,
        sliding_window=16,
        logits_soft_cap=20.0,
    )

    assert jnp.allclose(out.astype(jnp.float32), ref.astype(jnp.float32), rtol=1e-2, atol=2e-2)

    # Verify the updated cache stores the full K/V sequence for each request.
    for s, kv_len in enumerate(map(int, kv_lens.tolist())):
        num_blocks = (kv_len + block_size - 1) // block_size
        blk_ids = block_tables[s, :num_blocks]
        stored_k = new_kc[blk_ids].reshape(num_blocks * block_size, num_kv_heads, head_dim)[:kv_len]
        stored_v = new_vc[blk_ids].reshape(num_blocks * block_size, num_kv_heads, head_dim)[:kv_len]
        assert jnp.allclose(stored_k.astype(jnp.float32), k_full[s].astype(jnp.float32))
        assert jnp.allclose(stored_v.astype(jnp.float32), v_full[s].astype(jnp.float32))
