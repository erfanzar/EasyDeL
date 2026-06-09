from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec

from ejkernel.modules.operations import decode_attention


def test_decode_attention_module_runs_xla():
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    batch = 2
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 32
    page_size = 4
    max_pages = 4

    total_pages = batch * max_pages
    query = jax.random.normal(kq, (batch, num_q_heads, head_dim), dtype=jnp.bfloat16)
    key_buffer = jax.random.normal(kk, (total_pages * page_size, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    value_buffer = jax.random.normal(kv, (total_pages * page_size, num_kv_heads, head_dim), dtype=jnp.bfloat16)

    req_to_tokens = jnp.arange(total_pages, dtype=jnp.int32).reshape(batch, max_pages)
    seq_lens = jnp.array([7, 11], dtype=jnp.int32)

    out, lse = decode_attention(
        query,
        key_buffer,
        value_buffer,
        req_to_tokens,
        seq_lens,
        softmax_scale=1.0 / math.sqrt(head_dim),
        num_kv_splits=4,
        page_size=page_size,
        logits_soft_cap=20.0,
        platform="xla",
    )
    assert out.shape == (batch, num_q_heads, head_dim)
    assert lse.shape == (batch, num_q_heads)


def test_decode_attention_module_shard_map_runs_xla():
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    batch = 2
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 32
    page_size = 4
    max_pages = 4

    total_pages = batch * max_pages
    query = jax.random.normal(kq, (batch, num_q_heads, head_dim), dtype=jnp.bfloat16)
    key_buffer = jax.random.normal(kk, (total_pages * page_size, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    value_buffer = jax.random.normal(kv, (total_pages * page_size, num_kv_heads, head_dim), dtype=jnp.bfloat16)

    req_to_tokens = jnp.arange(total_pages, dtype=jnp.int32).reshape(batch, max_pages)
    seq_lens = jnp.array([7, 11], dtype=jnp.int32)

    mesh = Mesh(jax.devices()[:1], ("x",))
    in_specs = (
        PartitionSpec(None, None, None),
        PartitionSpec(None, None, None),
        PartitionSpec(None, None, None),
        PartitionSpec(None, None),
        PartitionSpec(None),
    )
    out_specs = (PartitionSpec(None, None, None), PartitionSpec(None, None))

    out, lse = decode_attention(
        query,
        key_buffer,
        value_buffer,
        req_to_tokens,
        seq_lens,
        softmax_scale=1.0 / math.sqrt(head_dim),
        num_kv_splits=4,
        page_size=page_size,
        logits_soft_cap=20.0,
        platform="xla",
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )
    assert out.shape == (batch, num_q_heads, head_dim)
    assert lse.shape == (batch, num_q_heads)
