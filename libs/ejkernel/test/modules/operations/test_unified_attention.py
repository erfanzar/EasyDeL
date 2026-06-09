from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ejkernel.modules.operations import unified_attention


def test_unified_attention_basic_and_optional_features_run():
    num_seqs = 2
    q_heads = 4
    kv_heads = 2
    head_dim = 32
    block_size = 8

    kv_lens = jnp.array([16, 9], dtype=jnp.int32)
    q_lens = [4, 1]
    max_kv = int(kv_lens.max())
    max_blocks_per_seq = math.ceil(max_kv / block_size)
    num_blocks_total = num_seqs * max_blocks_per_seq

    block_tables = jnp.arange(num_blocks_total, dtype=jnp.int32).reshape(num_seqs, max_blocks_per_seq)

    query_start_loc = jnp.array([0, q_lens[0], sum(q_lens)], dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])

    queries = jax.random.normal(jax.random.PRNGKey(0), (total_tokens, q_heads, head_dim), dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    key_cache = jax.random.normal(
        jax.random.PRNGKey(1), (num_blocks_total, block_size, kv_heads, head_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    value_cache = jax.random.normal(
        jax.random.PRNGKey(2), (num_blocks_total, block_size, kv_heads, head_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)

    out = unified_attention(queries, key_cache, value_cache, kv_lens, block_tables, query_start_loc, platform="xla")
    assert out.shape == (total_tokens, q_heads, head_dim)

    alibi = jnp.linspace(0.0, 1.0, q_heads, dtype=jnp.bfloat16)
    qq_bias = jnp.zeros((max(q_lens), max(q_lens)), dtype=jnp.float32)
    sink = jnp.zeros((q_heads,), dtype=jnp.bfloat16)

    out2 = unified_attention(
        queries,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        alibi,
        qq_bias,
        sink,
        causal=True,
        sliding_window=12,
        logits_soft_cap=10.0,
        platform="xla",
    )
    assert out2.shape == (total_tokens, q_heads, head_dim)
