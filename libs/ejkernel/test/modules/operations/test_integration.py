from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.modules.operations import attention, flash_attention, mean_pooling, scaled_dot_product_attention

from ._utils import assert_allclose, rand_qkv


def test_attention_variants_agree_small_xla():
    key = jax.random.PRNGKey(0)
    q, k, v = rand_qkv(key, batch=1, q_len=16, kv_len=16, q_heads=4, kv_heads=2, head_dim=32, dtype=jnp.bfloat16)

    out_flash = flash_attention(q, k, v, causal=True, platform="xla")
    out_sdpa = scaled_dot_product_attention(q, k, v, causal=True, platform="xla")
    out_attn = attention(q, k, v, None, None, None, causal=True)[0]

    assert_allclose(out_flash, out_sdpa, atol=0.2)
    assert_allclose(out_flash, out_attn, atol=0.2)


def test_flash_attention_then_mean_pooling_shape():
    key = jax.random.PRNGKey(1)
    q, k, v = rand_qkv(key, batch=2, q_len=8, kv_len=8, q_heads=4, kv_heads=4, head_dim=16, dtype=jnp.bfloat16)
    attn_out = flash_attention(q, k, v, causal=True, platform="xla")
    pooled = mean_pooling(attn_out.reshape(2, 8, 4 * 16), platform="xla")
    assert pooled.shape == (2, 64)
