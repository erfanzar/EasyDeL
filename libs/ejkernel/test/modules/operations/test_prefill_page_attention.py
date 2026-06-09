from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ejkernel.modules.operations import prefill_page_attention

from ._utils import assert_allclose, dense_attention_reference, device_platform


def _gather_prefill_kv(cache: jax.Array, page_indices: jax.Array, *, context_len: int) -> jax.Array:
    # cache: [kv_heads, total_pages, page_size, head_dim]
    pages = cache[:, page_indices]  # [kv_heads, num_pages, page_size, head_dim]
    kv_heads, num_pages, page_size, head_dim = pages.shape
    flat = pages.reshape(kv_heads, num_pages * page_size, head_dim)[:, :context_len, :]
    return flat.transpose(1, 0, 2)  # [context_len, kv_heads, head_dim]


def test_prefill_page_attention_xla_matches_dense_reference():
    num_heads, num_kv_heads, head_dim = 4, 2, 32
    page_size, total_pages = 8, 6
    page_indices = jnp.array([0, 1, 3], dtype=jnp.int32)
    chunk_size = 6
    context_len = 18

    q = jax.random.normal(jax.random.PRNGKey(0), (chunk_size, num_heads, head_dim), dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    k_cache = jax.random.normal(
        jax.random.PRNGKey(1), (num_kv_heads, total_pages, page_size, head_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    v_cache = jax.random.normal(
        jax.random.PRNGKey(2), (num_kv_heads, total_pages, page_size, head_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)

    scale = head_dim**-0.5
    soft_cap = 8.0
    sliding_window = 10

    out = prefill_page_attention(
        q,
        k_cache,
        v_cache,
        jnp.array([context_len], dtype=jnp.int32),
        page_indices,
        softmax_scale=scale,
        attn_logits_soft_cap=soft_cap,
        sliding_window=sliding_window,
        platform="xla",
    )
    assert out.shape == q.shape

    k = _gather_prefill_kv(k_cache, page_indices, context_len=context_len)[None, ...]
    v = _gather_prefill_kv(v_cache, page_indices, context_len=context_len)[None, ...]
    q4 = q[None, ...]

    q_abs = (context_len - chunk_size) + jnp.arange(chunk_size, dtype=jnp.int32)
    kv_pos = jnp.arange(context_len, dtype=jnp.int32)
    mask = kv_pos[None, :] <= q_abs[:, None]
    mask = mask & (kv_pos[None, :] >= (q_abs[:, None] - sliding_window + 1))
    mask = mask[None, None, :, :]

    ref_out, _ = dense_attention_reference(q4, k, v, attention_mask=mask, softmax_scale=scale, logits_soft_cap=soft_cap)
    assert_allclose(out, ref_out[0], atol=0.25)


@pytest.mark.skipif(device_platform() != "tpu", reason="TPU-only cross-backend comparison (pallas vs xla)")
def test_prefill_page_attention_pallas_matches_xla_on_tpu():
    num_heads, num_kv_heads, head_dim = 1, 1, 128
    page_size, total_pages = 32, 1
    page_indices = jnp.array([0], dtype=jnp.int32)
    chunk_size = 32
    context_len = 32

    q = jax.random.normal(jax.random.PRNGKey(10), (chunk_size, num_heads, head_dim), dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    k_cache = jax.random.normal(
        jax.random.PRNGKey(11), (num_kv_heads, total_pages, page_size, head_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    v_cache = jax.random.normal(
        jax.random.PRNGKey(12), (num_kv_heads, total_pages, page_size, head_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)

    out_xla = prefill_page_attention(
        q,
        k_cache,
        v_cache,
        jnp.array([context_len], dtype=jnp.int32),
        page_indices,
        sliding_window=32,
        attn_logits_soft_cap=10.0,
        platform="xla",
    )
    out_pallas = prefill_page_attention(
        q,
        k_cache,
        v_cache,
        jnp.array([context_len], dtype=jnp.int32),
        page_indices,
        sliding_window=32,
        attn_logits_soft_cap=10.0,
        platform="pallas",
    )

    assert_allclose(out_pallas, out_xla, atol=0.35)
