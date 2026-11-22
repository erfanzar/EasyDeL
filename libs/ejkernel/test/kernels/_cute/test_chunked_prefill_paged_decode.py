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

import importlib
import importlib.util
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_has_cutlass = importlib.util.find_spec("cutlass") is not None
if not _has_cutlass:
    pytest.skip("CUTLASS CuTe DSL not installed", allow_module_level=True)
if jax.devices()[0].platform != "gpu":
    pytest.skip("CUTE tests require GPU backend", allow_module_level=True)

_cute_chunked_prefill_module = importlib.import_module("ejkernel.kernels._cute.chunked_prefill_paged_decode")
cute_chunked_prefill_paged_decode = _cute_chunked_prefill_module.chunked_prefill_paged_decode
_xla_chunked_prefill_module = importlib.import_module("ejkernel.kernels._xla.chunked_prefill_paged_decode")
xla_chunked_prefill_paged_decode = _xla_chunked_prefill_module.chunked_prefill_paged_decode


def _write_tokens_to_cache(
    cache: jax.Array,
    *,
    tokens: jax.Array,
    seq_idx: int,
    block_tables: jax.Array,
    block_size: int,
    start_pos: int,
) -> jax.Array:
    """Write token rows into a block-tabled cache for a single sequence."""
    for i in range(int(tokens.shape[0])):
        pos = start_pos + i
        block_idx = int(pos // block_size)
        within = int(pos % block_size)
        phys = int(block_tables[seq_idx, block_idx])
        cache = cache.at[phys, within].set(tokens[i])
    return cache


def test_chunked_prefill_paged_decode_cute_matches_xla():
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

    query_start_loc = jnp.array([0, q_lens[0], q_lens[0] + q_lens[1], sum(q_lens)], dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])

    max_kv = int(kv_lens.max())
    max_blocks_per_seq = (max_kv + block_size - 1) // block_size
    num_blocks_total = num_seqs * max_blocks_per_seq
    block_tables = jnp.arange(num_blocks_total, dtype=jnp.int32).reshape(num_seqs, max_blocks_per_seq)

    key_cache = jnp.zeros((num_blocks_total, block_size, num_kv_heads, head_dim), dtype=jnp.float16)
    value_cache = jnp.zeros_like(key_cache)

    for seq_idx, ctx_len in enumerate(context_lens):
        ctx_keys = jax.random.normal(
            jax.random.fold_in(kk_ctx, seq_idx),
            (ctx_len, num_kv_heads, head_dim),
            dtype=jnp.float16,
        )
        ctx_values = jax.random.normal(
            jax.random.fold_in(kv_ctx, seq_idx),
            (ctx_len, num_kv_heads, head_dim),
            dtype=jnp.float16,
        )
        key_cache = _write_tokens_to_cache(
            key_cache,
            tokens=ctx_keys,
            seq_idx=seq_idx,
            block_tables=block_tables,
            block_size=block_size,
            start_pos=0,
        )
        value_cache = _write_tokens_to_cache(
            value_cache,
            tokens=ctx_values,
            seq_idx=seq_idx,
            block_tables=block_tables,
            block_size=block_size,
            start_pos=0,
        )

    queries = jax.random.normal(kq, (total_tokens, num_q_heads, head_dim), dtype=jnp.float16)
    keys = jax.random.normal(kk_new, (total_tokens, num_kv_heads, head_dim), dtype=jnp.float16)
    values = jax.random.normal(kv_new, (total_tokens, num_kv_heads, head_dim), dtype=jnp.float16)

    scale = 1.0 / math.sqrt(head_dim)

    out_cute, key_cache_cute, value_cache_cute = cute_chunked_prefill_paged_decode(
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
        seq_threshold_3d=0,
        num_par_softmax_segments=8,
    )
    out_xla, key_cache_xla, value_cache_xla = xla_chunked_prefill_paged_decode(
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

    out_cute, key_cache_cute, value_cache_cute = (
        jax.block_until_ready(out_cute),
        jax.block_until_ready(key_cache_cute),
        jax.block_until_ready(value_cache_cute),
    )
    out_xla, key_cache_xla, value_cache_xla = (
        jax.block_until_ready(out_xla),
        jax.block_until_ready(key_cache_xla),
        jax.block_until_ready(value_cache_xla),
    )

    np.testing.assert_allclose(
        np.asarray(out_cute, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(np.asarray(key_cache_cute), np.asarray(key_cache_xla), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(value_cache_cute), np.asarray(value_cache_xla), rtol=0.0, atol=0.0)


def test_chunked_prefill_paged_decode_cute_jit_matches_xla():
    key = jax.random.PRNGKey(11)
    kq, kk, kv = jax.random.split(key, 3)

    num_seqs = 2
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 32
    block_size = 8
    q_lens = [2, 1]

    kv_lens = jnp.array(q_lens, dtype=jnp.int32)
    query_start_loc = jnp.array([0, q_lens[0], sum(q_lens)], dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])

    max_kv = int(kv_lens.max())
    max_blocks_per_seq = (max_kv + block_size - 1) // block_size
    num_blocks_total = num_seqs * max_blocks_per_seq
    block_tables = jnp.arange(num_blocks_total, dtype=jnp.int32).reshape(num_seqs, max_blocks_per_seq)

    queries = jax.random.normal(kq, (total_tokens, num_q_heads, head_dim), dtype=jnp.float16)
    keys = jax.random.normal(kk, (total_tokens, num_kv_heads, head_dim), dtype=jnp.float16)
    values = jax.random.normal(kv, (total_tokens, num_kv_heads, head_dim), dtype=jnp.float16)
    key_cache = jnp.zeros((num_blocks_total, block_size, num_kv_heads, head_dim), dtype=jnp.float16)
    value_cache = jnp.zeros_like(key_cache)
    scale = 1.0 / math.sqrt(head_dim)

    @jax.jit
    def _run(q, k, v, kc, vc):
        return cute_chunked_prefill_paged_decode(
            q,
            k,
            v,
            kc,
            vc,
            kv_lens,
            block_tables,
            query_start_loc,
            softmax_scale=scale,
            causal=True,
            sliding_window=16,
            logits_soft_cap=20.0,
        )

    out_cute, key_cache_cute, value_cache_cute = _run(
        queries,
        keys,
        values,
        key_cache,
        value_cache,
    )
    out_xla, key_cache_xla, value_cache_xla = xla_chunked_prefill_paged_decode(
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

    out_cute, key_cache_cute, value_cache_cute = (
        jax.block_until_ready(out_cute),
        jax.block_until_ready(key_cache_cute),
        jax.block_until_ready(value_cache_cute),
    )
    out_xla, key_cache_xla, value_cache_xla = (
        jax.block_until_ready(out_xla),
        jax.block_until_ready(key_cache_xla),
        jax.block_until_ready(value_cache_xla),
    )

    np.testing.assert_allclose(
        np.asarray(out_cute, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(np.asarray(key_cache_cute), np.asarray(key_cache_xla), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(value_cache_cute), np.asarray(value_cache_xla), rtol=0.0, atol=0.0)
