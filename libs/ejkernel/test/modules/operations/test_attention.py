from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.modules.operations import attention
from ejkernel.types import MaskInfo

from ._utils import assert_allclose, dense_attention_reference, rand_qkv


def test_attention_matches_dense_reference_gqa_and_softmax_aux():
    key = jax.random.PRNGKey(0)
    q, k, v = rand_qkv(
        key,
        batch=2,
        q_len=8,
        kv_len=12,
        q_heads=4,
        kv_heads=2,
        head_dim=16,
        dtype=jnp.bfloat16,
    )
    softmax_aux = jax.random.normal(jax.random.PRNGKey(1), (3,), dtype=jnp.float32).astype(jnp.bfloat16)

    out = attention(q, k, v, None, None, softmax_aux, causal=True, softmax_scale=16**-0.5)[0]
    ref_out, _ = dense_attention_reference(q, k, v, causal=True, softmax_scale=16**-0.5, softmax_aux=softmax_aux)

    assert out.shape == (2, 8, 4, 16)
    assert_allclose(out, ref_out, atol=0.12)


def test_attention_mask_info_attention_mask_matches_reference():
    key = jax.random.PRNGKey(2)
    q, k, v = rand_qkv(key, batch=2, q_len=8, kv_len=8, q_heads=4, kv_heads=4, head_dim=16, dtype=jnp.bfloat16)

    mask = jnp.ones((2, 1, 8, 8), dtype=jnp.bool_)
    mask = mask.at[0, :, 4:, :2].set(False)
    mask = mask.at[1, :, :, 6:].set(False)
    mask_info = MaskInfo.from_attention_mask(mask)

    out = attention(q, k, v, None, None, None, mask_info=mask_info)[0]
    ref_out, _ = dense_attention_reference(q, k, v, attention_mask=mask)

    assert out.shape == (2, 8, 4, 16)
    assert_allclose(out, ref_out, atol=0.12)


def test_attention_bias_sliding_window_and_logits_soft_cap_match_reference():
    key = jax.random.PRNGKey(3)
    q, k, v = rand_qkv(key, batch=1, q_len=12, kv_len=12, q_heads=4, kv_heads=2, head_dim=32, dtype=jnp.bfloat16)

    bias = jax.random.normal(jax.random.PRNGKey(4), (1, 4, 12, 12), dtype=jnp.float32).astype(jnp.bfloat16)
    sliding_window = (5, 2)
    logits_soft_cap = 10.0
    scale = 32**-0.5

    out = attention(
        q,
        k,
        v,
        bias,
        None,
        None,
        softmax_scale=scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
    )[0]
    ref_out, _ = dense_attention_reference(
        q,
        k,
        v,
        bias=bias,
        softmax_scale=scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
    )
    assert_allclose(out, ref_out, atol=0.15)


def test_attention_dropout_and_init_bias_runs():
    key = jax.random.PRNGKey(5)
    q, k, v = rand_qkv(key, batch=2, q_len=8, kv_len=8, q_heads=4, kv_heads=4, head_dim=16, dtype=jnp.bfloat16)

    def init_bias():
        return jnp.zeros((2, 4, 8, 8), dtype=jnp.bfloat16)

    out1 = attention(
        q,
        k,
        v,
        None,
        jax.random.PRNGKey(0),
        None,
        init_bias=init_bias,
        deterministic=False,
        dropout_prob=0.25,
    )[0]
    out2 = attention(
        q,
        k,
        v,
        None,
        jax.random.PRNGKey(1),
        None,
        init_bias=init_bias,
        deterministic=False,
        dropout_prob=0.25,
    )[0]

    assert out1.shape == q.shape
    assert out2.shape == q.shape
    assert not jnp.allclose(out1, out2)
