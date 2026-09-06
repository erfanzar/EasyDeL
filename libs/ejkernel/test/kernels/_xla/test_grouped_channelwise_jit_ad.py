"""Differentiating a cached compiled call must not retain traced operands."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import grouped_matmul_channelwise


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_autodiff_after_compiling_grouped_call(bits):
    x = jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 7
    codes = (jnp.arange(24).reshape(2, 4, 3) % 7 - 3).astype(jnp.int4)
    scales = jnp.ones((2, 1, 3), jnp.float32) * 0.2
    groups = jnp.array([3, 5], jnp.int32)
    f = jax.jit(
        lambda a, b, c, d: grouped_matmul_channelwise(
            a, b, c, d, activation_bits=bits, preferred_element_type=jnp.float32
        )
    )
    f(x, codes, scales, groups).block_until_ready()
    def actual(a):
        return f(a, codes, scales, groups)
    def direct(a):
        return grouped_matmul_channelwise(
            a, codes, scales, groups, activation_bits=bits, preferred_element_type=jnp.float32
        )
    _, got = jax.jvp(actual, (x,), (jnp.ones_like(x),))
    _, want = jax.jvp(direct, (x,), (jnp.ones_like(x),))
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    gv = jax.grad(lambda a: jnp.sum(actual(a)))(x)
    wv = jax.grad(lambda a: jnp.sum(direct(a)))(x)
    np.testing.assert_allclose(gv, wv, rtol=1e-6, atol=1e-6)
