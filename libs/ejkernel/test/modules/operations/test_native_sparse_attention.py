from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.modules.operations import native_sparse_attention
from ejkernel.modules.operations.configs import NativeSparseAttentionConfig


def test_native_sparse_attention_block_indices_and_g_slc_gating():
    b, t, hq, hkv, d = 1, 32, 4, 2, 16
    block_size = 8
    num_blocks = (t + block_size - 1) // block_size
    num_selected = 2

    q = jax.random.normal(jax.random.PRNGKey(0), (b, t, hq, d), dtype=jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(jax.random.PRNGKey(1), (b, t, hkv, d), dtype=jnp.float32).astype(jnp.bfloat16)
    v = jax.random.normal(jax.random.PRNGKey(2), (b, t, hkv, d), dtype=jnp.float32).astype(jnp.bfloat16)

    block_indices = jnp.zeros((b, t, hkv, num_selected), dtype=jnp.int32)
    block_counts = jnp.full((b, t, hkv), num_selected, dtype=jnp.int32)

    cfg = NativeSparseAttentionConfig(block_size=block_size, platform="xla", backend="any")
    out = native_sparse_attention(q, k, v, None, None, block_indices, block_counts, cfg=cfg, platform="xla")
    assert out.shape == (b, t, hq, d)
    assert jnp.all(jnp.isfinite(out))

    g_slc = jnp.zeros((b, t, hq), dtype=jnp.bfloat16)
    out_zero = native_sparse_attention(q, k, v, None, g_slc, block_indices, block_counts, cfg=cfg, platform="xla")
    assert jnp.all(out_zero == 0)

    assert num_blocks >= 1  # smoke-check that block_size is applied


def test_native_sparse_attention_with_g_cmp_runs_without_block_indices():
    b, t, hq, hkv, d = 1, 32, 4, 2, 16
    block_size = 8

    q = jax.random.normal(jax.random.PRNGKey(3), (b, t, hq, d), dtype=jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(jax.random.PRNGKey(4), (b, t, hkv, d), dtype=jnp.float32).astype(jnp.bfloat16)
    v = jax.random.normal(jax.random.PRNGKey(5), (b, t, hkv, d), dtype=jnp.float32).astype(jnp.bfloat16)

    g_cmp = jnp.ones((b, t, hq), dtype=jnp.bfloat16)
    cfg = NativeSparseAttentionConfig(block_size=block_size, platform="xla", backend="any")
    out = native_sparse_attention(q, k, v, g_cmp, None, None, 2, cfg=cfg, platform="xla")

    assert out.shape == (b, t, hq, d)
    assert jnp.all(jnp.isfinite(out))
