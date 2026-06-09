from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.modules.operations import lightning_attention


def test_lightning_attention_basic_and_return_state():
    key = jax.random.PRNGKey(0)
    b, t, hq, hkv, d = 1, 8, 4, 4, 16
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (b, t, hq, d), dtype=jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(kk, (b, t, hkv, d), dtype=jnp.float32).astype(jnp.bfloat16)
    v = jax.random.normal(kv, (b, t, hkv, d), dtype=jnp.float32).astype(jnp.bfloat16)

    out = lightning_attention(q, k, v, layer_idx=0, num_layers=4, platform="xla")
    assert out.shape == (b, t, hq, d)

    out2, state = lightning_attention(
        q,
        k,
        v,
        None,
        None,
        layer_idx=3,
        num_layers=4,
        softmax_scale=d**-0.5,
        reverse=True,
        return_state=True,
        platform="xla",
    )
    assert out2.shape == (b, t, hq, d)
    assert state.shape == (b, hq, d, d)
