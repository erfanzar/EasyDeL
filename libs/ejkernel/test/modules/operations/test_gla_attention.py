from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.modules.operations import gla_attention


def test_gla_attention_basic_and_return_state():
    key = jax.random.PRNGKey(0)
    b, t, hq, hkv, d = 2, 8, 4, 4, 16
    kq, kk, kv, kg = jax.random.split(key, 4)
    q = jax.random.normal(kq, (b, t, hq, d), dtype=jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(kk, (b, t, hkv, d), dtype=jnp.float32).astype(jnp.bfloat16)
    v = jax.random.normal(kv, (b, t, hkv, d), dtype=jnp.float32).astype(jnp.bfloat16)
    g = jax.random.normal(kg, (b, t, hq, d), dtype=jnp.float32).astype(jnp.bfloat16)
    g_gamma = jnp.ones((b, hq), dtype=jnp.bfloat16)

    out = gla_attention(q, k, v, platform="xla")
    assert out.shape == (b, t, hq, d)

    out2, state = gla_attention(q, k, v, g, g_gamma, return_state=True, reverse=True, platform="xla")
    assert out2.shape == (b, t, hq, d)
    assert state.shape == (b, hq, d, d)
