from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.modules.operations import recurrent_attention


def test_recurrent_attention_basic_shape():
    key = jax.random.PRNGKey(0)
    b, t, hq, hkv, d = 2, 8, 4, 4, 16
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (b, t, hq, d), dtype=jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(kk, (b, t, hkv, d), dtype=jnp.float32).astype(jnp.bfloat16)
    v = jax.random.normal(kv, (b, t, hkv, d), dtype=jnp.float32).astype(jnp.bfloat16)

    out = recurrent_attention(q, k, v, platform="xla")
    assert out.shape == (b, t, hq, d)


def test_recurrent_attention_gating_reverse_and_return_state():
    key = jax.random.PRNGKey(1)
    b, t, hq, hkv, d = 1, 8, 4, 4, 16
    kq, kk, kv, kg = jax.random.split(key, 4)
    q = jax.random.normal(kq, (b, t, hq, d), dtype=jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(kk, (b, t, hkv, d), dtype=jnp.float32).astype(jnp.bfloat16)
    v = jax.random.normal(kv, (b, t, hkv, d), dtype=jnp.float32).astype(jnp.bfloat16)
    g = jax.random.normal(kg, (b, t, hq, d), dtype=jnp.float32).astype(jnp.bfloat16)
    g_gamma = jnp.ones((b, hq), dtype=jnp.bfloat16)

    out, state = recurrent_attention(q, k, v, g, g_gamma, reverse=True, return_state=True, platform="xla")
    assert out.shape == (b, t, hq, d)
    assert state.shape == (b, hq, d, d)

    out2 = recurrent_attention(
        q[:, :1], k[:, :1], v[:, :1], None, None, None, None, state, return_state=False, platform="xla"
    )
    assert out2.shape == (b, 1, hq, d)
