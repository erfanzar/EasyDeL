from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ejkernel.errors import EjkernelRuntimeError
from ejkernel.modules.operations import page_attention

from ._utils import assert_allclose, dense_attention_reference, device_platform


def _gather_kv(
    cache: jax.Array, indices: jax.Array, *, context_len: int, block_size: int
) -> jax.Array:  # [kv_len, kv_heads, head_dim]
    valid = indices[indices >= 0]
    pages = cache[valid]  # [num_pages, kv_heads, block_size, head_dim]
    kv_heads = pages.shape[1]
    head_dim = pages.shape[3]
    flat = pages.transpose(1, 0, 2, 3).reshape(kv_heads, -1, head_dim)[:, :context_len, :]
    return flat.transpose(1, 0, 2)


def test_page_attention_xla_matches_dense_reference_variable_contexts():
    jax.random.PRNGKey(0)
    num_seqs, q_heads, kv_heads, head_dim = 3, 4, 2, 32
    block_size, pages_per_seq = 8, 4
    num_blocks = num_seqs * pages_per_seq

    q = jax.random.normal(jax.random.PRNGKey(1), (num_seqs, q_heads, head_dim), dtype=jnp.float32).astype(jnp.bfloat16)
    k_cache = jax.random.normal(
        jax.random.PRNGKey(2), (num_blocks, kv_heads, block_size, head_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    v_cache = jax.random.normal(
        jax.random.PRNGKey(3), (num_blocks, kv_heads, block_size, head_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)

    block_tables = -jnp.ones((num_seqs, pages_per_seq), dtype=jnp.int32)
    block_tables = block_tables.at[0, :3].set(jnp.array([0, 1, 2], dtype=jnp.int32))
    block_tables = block_tables.at[1, :2].set(jnp.array([4, 5], dtype=jnp.int32))
    block_tables = block_tables.at[2, :].set(jnp.array([8, 9, 10, 11], dtype=jnp.int32))

    context_lens = jnp.array([23, 10, 32], dtype=jnp.int32)
    scale = head_dim**-0.5

    out = page_attention(
        q,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
        attn_scale=scale,
        platform="xla",
    )
    assert out.shape == (num_seqs, q_heads, head_dim)

    refs = []
    for i in range(num_seqs):
        k_i = _gather_kv(k_cache, block_tables[i], context_len=int(context_lens[i]), block_size=block_size)[None, ...]
        v_i = _gather_kv(v_cache, block_tables[i], context_len=int(context_lens[i]), block_size=block_size)[None, ...]
        q_i = q[i][None, None, :, :]
        ref_i, _ = dense_attention_reference(
            q_i,
            k_i,
            v_i,
            softmax_scale=scale,
        )
        refs.append(ref_i[0, 0])
    ref = jnp.stack(refs, axis=0)

    assert_allclose(out, ref, atol=0.2)


def test_page_attention_xla_rejects_attn_logits_soft_cap():
    key = jax.random.PRNGKey(9)
    num_seqs, q_heads, kv_heads, head_dim = 1, 4, 2, 32
    block_size, pages_per_seq = 8, 2
    num_blocks = num_seqs * pages_per_seq

    q = jax.random.normal(key, (num_seqs, q_heads, head_dim), dtype=jnp.float32).astype(jnp.bfloat16)
    k_cache = jax.random.normal(
        jax.random.PRNGKey(10), (num_blocks, kv_heads, block_size, head_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    v_cache = jax.random.normal(
        jax.random.PRNGKey(11), (num_blocks, kv_heads, block_size, head_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    block_tables = jnp.arange(num_blocks, dtype=jnp.int32).reshape(num_seqs, pages_per_seq)
    context_lens = jnp.array([block_size * pages_per_seq], dtype=jnp.int32)

    with pytest.raises(EjkernelRuntimeError):
        page_attention(q, k_cache, v_cache, context_lens, block_tables, attn_logits_soft_cap=10.0, platform="xla")


@pytest.mark.skipif(device_platform() != "tpu", reason="TPU-only pallas path coverage")
def test_page_attention_inline_seq_dim_pallas_parity_on_tpu():
    key = jax.random.PRNGKey(10)
    num_seqs, q_heads, kv_heads, head_dim = 2, 4, 2, 128
    block_size, pages_per_seq = 128, 2
    num_blocks = num_seqs * pages_per_seq

    q = jax.random.normal(key, (num_seqs, q_heads, head_dim), dtype=jnp.float32).astype(jnp.bfloat16)
    k_cache = jax.random.normal(
        jax.random.PRNGKey(11), (num_blocks, kv_heads, block_size, head_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    v_cache = jax.random.normal(
        jax.random.PRNGKey(12), (num_blocks, kv_heads, block_size, head_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    block_tables = jnp.arange(num_blocks, dtype=jnp.int32).reshape(num_seqs, pages_per_seq)
    context_lens = jnp.array([192, 256], dtype=jnp.int32)

    out_xla = page_attention(q, k_cache, v_cache, context_lens, block_tables, platform="xla")
    out_inline = page_attention(q, k_cache, v_cache, context_lens, block_tables, platform="pallas", inline_seq_dim=True)
    out_noinline = page_attention(
        q, k_cache, v_cache, context_lens, block_tables, platform="pallas", inline_seq_dim=False
    )

    assert_allclose(out_inline, out_xla, atol=0.25)
    assert_allclose(out_noinline, out_xla, atol=0.25)
