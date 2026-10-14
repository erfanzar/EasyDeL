from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from ejkernel.modules.operations import scaled_dot_product_attention
from ejkernel.types import MaskInfo

from ._utils import assert_allclose, dense_attention_reference, rand_qkv


def test_scaled_dot_product_attention_matches_dense_reference_basic():
    key = jax.random.PRNGKey(0)
    q, k, v = rand_qkv(key, batch=2, q_len=8, kv_len=10, q_heads=4, kv_heads=2, head_dim=32, dtype=jnp.bfloat16)

    out = scaled_dot_product_attention(q, k, v, platform="xla")
    ref_out, _ = dense_attention_reference(q, k, v)
    assert out.shape == (2, 8, 4, 32)
    assert_allclose(out, ref_out, atol=0.15)


def test_scaled_dot_product_attention_bias_causal_and_sliding_window_match_reference():
    key = jax.random.PRNGKey(1)
    q, k, v = rand_qkv(key, batch=1, q_len=12, kv_len=12, q_heads=4, kv_heads=4, head_dim=16, dtype=jnp.bfloat16)
    bias = jax.random.normal(jax.random.PRNGKey(2), (1, 4, 12, 12), dtype=jnp.float32).astype(jnp.bfloat16)

    out = scaled_dot_product_attention(q, k, v, bias, None, None, causal=True, sliding_window=(6, 0), platform="xla")
    ref_out, _ = dense_attention_reference(q, k, v, bias=bias, causal=True, sliding_window=(6, 0))
    assert_allclose(out, ref_out, atol=0.18)


def test_scaled_dot_product_attention_mask_info_attention_mask_matches_reference():
    key = jax.random.PRNGKey(3)
    q, k, v = rand_qkv(key, batch=2, q_len=8, kv_len=8, q_heads=4, kv_heads=4, head_dim=16, dtype=jnp.bfloat16)
    mask = jnp.ones((2, 1, 8, 8), dtype=jnp.bool_)
    mask = mask.at[:, :, :, 6:].set(False)
    # Ensure no query row is fully masked.
    mask = mask.at[0, :, 4:, 1:].set(False)
    mask_info = MaskInfo.from_attention_mask(mask)

    out = scaled_dot_product_attention(q, k, v, None, None, None, mask_info=mask_info, platform="xla")
    ref_out, _ = dense_attention_reference(q, k, v, attention_mask=mask)
    assert_allclose(out, ref_out, atol=0.15)


def _hf_sliding_window_mask(batch: int, length: int, window: int) -> jax.Array:
    """``transformers.masking_utils`` causal sliding window: the query plus the previous ``window - 1`` keys."""
    q_idx = jnp.arange(length)[:, None]
    kv_idx = jnp.arange(length)[None, :]
    mask = (kv_idx <= q_idx) & (kv_idx > q_idx - window)
    return jnp.broadcast_to(mask[None, None], (batch, 1, length, length))


@pytest.mark.parametrize("sliding_window", [1, 3, 8])
def test_scaled_dot_product_attention_int_sliding_window_is_hf_window_size(sliding_window: int):
    """An int window ``W`` keeps exactly ``W`` keys (itself + ``W - 1`` previous), like HF and the paged kernels."""
    q, k, v = rand_qkv(
        jax.random.PRNGKey(20), batch=2, q_len=24, kv_len=24, q_heads=4, kv_heads=4, head_dim=32, dtype=jnp.float32
    )
    with jax.default_matmul_precision("highest"):
        out = scaled_dot_product_attention(
            q, k, v, None, None, None, causal=True, sliding_window=sliding_window, softmax_scale=0.5, platform="xla"
        )
        ref_out, _ = dense_attention_reference(
            q, k, v, attention_mask=_hf_sliding_window_mask(2, 24, sliding_window), softmax_scale=0.5
        )
    assert_allclose(out, ref_out, atol=2e-3)
