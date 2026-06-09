from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec

from ejkernel.modules.operations import chunked_prefill_paged_decode


def test_chunked_prefill_paged_decode_module_runs_xla():
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    num_seqs = 2
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 32
    block_size = 8

    q_lens = [3, 1]
    kv_lens = jnp.array(q_lens, dtype=jnp.int32)
    query_start_loc = jnp.array([0, q_lens[0], sum(q_lens)], dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])

    max_kv = int(kv_lens.max())
    max_blocks_per_seq = (max_kv + block_size - 1) // block_size
    num_blocks_total = num_seqs * max_blocks_per_seq
    block_tables = jnp.arange(num_blocks_total, dtype=jnp.int32).reshape(num_seqs, max_blocks_per_seq)

    queries = jax.random.normal(kq, (total_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16)
    keys = jax.random.normal(kk, (total_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    values = jax.random.normal(kv, (total_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)

    key_cache = jnp.zeros((num_blocks_total, block_size, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    value_cache = jnp.zeros_like(key_cache)

    out, new_kc, new_vc = chunked_prefill_paged_decode(
        queries,
        keys,
        values,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
        sliding_window=16,
        logits_soft_cap=20.0,
        platform="xla",
    )

    assert out.shape == (total_tokens, num_q_heads, head_dim)
    assert new_kc.shape == key_cache.shape
    assert new_vc.shape == value_cache.shape


def test_chunked_prefill_paged_decode_module_shard_map_runs_xla():
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    num_seqs = 2
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 32
    block_size = 8

    q_lens = [3, 1]
    kv_lens = jnp.array(q_lens, dtype=jnp.int32)
    query_start_loc = jnp.array([0, q_lens[0], sum(q_lens)], dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])

    max_kv = int(kv_lens.max())
    max_blocks_per_seq = (max_kv + block_size - 1) // block_size
    num_blocks_total = num_seqs * max_blocks_per_seq
    block_tables = jnp.arange(num_blocks_total, dtype=jnp.int32).reshape(num_seqs, max_blocks_per_seq)

    queries = jax.random.normal(kq, (total_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16)
    keys = jax.random.normal(kk, (total_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    values = jax.random.normal(kv, (total_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)

    key_cache = jnp.zeros((num_blocks_total, block_size, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    value_cache = jnp.zeros_like(key_cache)

    mesh = Mesh(jax.devices()[:1], ("x",))
    in_specs = (
        PartitionSpec(None, None, None),
        PartitionSpec(None, None, None),
        PartitionSpec(None, None, None),
        PartitionSpec(None, None, None, None),
        PartitionSpec(None, None, None, None),
        PartitionSpec(None),
        PartitionSpec(None, None),
        PartitionSpec(None),
        None,
        None,
    )
    out_specs = (
        PartitionSpec(None, None, None),
        PartitionSpec(None, None, None, None),
        PartitionSpec(None, None, None, None),
    )

    out, new_kc, new_vc = chunked_prefill_paged_decode(
        queries,
        keys,
        values,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
        sliding_window=16,
        logits_soft_cap=20.0,
        platform="xla",
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )

    assert out.shape == (total_tokens, num_q_heads, head_dim)
    assert new_kc.shape == key_cache.shape
    assert new_vc.shape == value_cache.shape
