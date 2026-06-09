from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.modules.operations import mean_pooling

from ._utils import assert_allclose


def test_mean_pooling_fixed_matches_jnp_mean():
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 8, 16), dtype=jnp.float32).astype(jnp.bfloat16)
    out = mean_pooling(x, platform="xla")
    ref = jnp.mean(x, axis=1)
    assert out.shape == (3, 16)
    assert_allclose(out, ref, atol=0.0)


def test_mean_pooling_varlen_matches_manual_and_gradient():
    lengths = [3, 1, 4]
    cu = [0]
    for l in lengths:
        cu.append(cu[-1] + l)
    cu_seqlens = jnp.array(cu, dtype=jnp.int32)
    total = int(cu_seqlens[-1])

    x = jax.random.normal(jax.random.PRNGKey(1), (total, 8), dtype=jnp.float32).astype(jnp.bfloat16)
    out = mean_pooling(x, cu_seqlens, platform="xla")

    refs = []
    for i, _ in enumerate(lengths):
        start = int(cu_seqlens[i])
        end = int(cu_seqlens[i + 1])
        refs.append(jnp.mean(x[start:end], axis=0))
    ref = jnp.stack(refs, axis=0)

    assert out.shape == (len(lengths), 8)
    assert_allclose(out, ref, atol=0.0)

    def loss_fn(inp):
        return jnp.sum(mean_pooling(inp, cu_seqlens, platform="xla"))

    dx = jax.grad(loss_fn)(x.astype(jnp.float32))
    expected = []
    for _, l in enumerate(lengths):
        expected.append(jnp.ones((l, 8), dtype=jnp.float32) / float(l))
    expected = jnp.concatenate(expected, axis=0)
    assert_allclose(dx, expected, atol=0.0)
