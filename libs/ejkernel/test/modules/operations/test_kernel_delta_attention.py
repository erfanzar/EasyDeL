from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.modules.operations import kernel_delta_attention

from ._utils import assert_allclose


def test_kernel_delta_attention_chunked_and_recurrent_match_and_return_state():
    batch, seq_len, heads, qk_dim, v_dim = 1, 8, 2, 8, 8
    key = jax.random.PRNGKey(0)
    kq, kk, kv, kb, kd = jax.random.split(key, 5)

    q = jax.random.normal(kq, (batch, seq_len, heads, qk_dim), dtype=jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(kk, (batch, seq_len, heads, qk_dim), dtype=jnp.float32).astype(jnp.bfloat16)
    v = jax.random.normal(kv, (batch, seq_len, heads, v_dim), dtype=jnp.float32).astype(jnp.bfloat16)
    beta = jax.nn.sigmoid(jax.random.normal(kb, (batch, seq_len, heads), dtype=jnp.float32)).astype(jnp.bfloat16)
    decay = (jax.random.normal(kd, (batch, seq_len, heads), dtype=jnp.float32) * -0.01).astype(jnp.bfloat16)

    out_chunked, state = kernel_delta_attention(
        q, k, v, beta, decay, return_state=True, use_chunked=True, platform="xla"
    )
    out_recurrent = kernel_delta_attention(q, k, v, beta, decay, return_state=False, use_chunked=False, platform="xla")

    assert out_chunked.shape == (batch, seq_len, heads, v_dim)
    assert out_recurrent.shape == (batch, seq_len, heads, v_dim)
    assert state.shape == (batch, heads, qk_dim, v_dim)

    assert_allclose(out_chunked, out_recurrent, atol=0.25)

    q1 = jax.random.normal(jax.random.PRNGKey(1), (batch, 1, heads, qk_dim), dtype=jnp.float32).astype(jnp.bfloat16)
    k1 = jax.random.normal(jax.random.PRNGKey(2), (batch, 1, heads, qk_dim), dtype=jnp.float32).astype(jnp.bfloat16)
    v1 = jax.random.normal(jax.random.PRNGKey(3), (batch, 1, heads, v_dim), dtype=jnp.float32).astype(jnp.bfloat16)
    beta1 = jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(4), (batch, 1, heads), dtype=jnp.float32)).astype(
        jnp.bfloat16
    )
    decay1 = (jax.random.normal(jax.random.PRNGKey(5), (batch, 1, heads), dtype=jnp.float32) * -0.01).astype(
        jnp.bfloat16
    )

    out_next, state_next = kernel_delta_attention(
        q1, k1, v1, beta1, decay1, state, return_state=True, use_chunked=False, platform="xla"
    )
    assert out_next.shape == (batch, 1, heads, v_dim)
    assert state_next.shape == (batch, heads, qk_dim, v_dim)
