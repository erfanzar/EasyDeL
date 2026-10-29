"""Public Pallas A4/A8 streaming parity and compiled AD."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import grouped_matmul_channelwise


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("m,k,n", [(3, 128, 128), (24, 160, 256), (80, 640, 256)])
@pytest.mark.parametrize("empty", [False, True])
def test_pallas_integer_streaming(bits, m, k, n, empty):
    rng = np.random.default_rng(61)
    x = jnp.asarray(rng.normal(size=(m, k)), jnp.bfloat16)
    groups = jnp.array([0, 0, 0] if empty else [m // 2, 0, m - m // 2 - 1], jnp.int32)
    valid = int(groups.sum())
    x = x.at[valid:].set(jnp.nan)
    w = jnp.asarray(rng.integers(-7, 8, (3, k, n), dtype=np.int8)).astype(jnp.int4 if bits == 4 else jnp.int8)
    s = jnp.asarray(rng.uniform(0.001, 0.1, (3, 1, n)), jnp.float32)

    def make(platform):
        return jax.jit(
            lambda a, b, c, g: grouped_matmul_channelwise(a, b, c, g, activation_bits=bits, platform=platform)
        )

    fast = make("pallas")
    slow = make("xla")
    np.testing.assert_array_equal(fast(x, w, s, groups), slow(x, w, s, groups))

    def derivatives(f):
        return jax.jit(jax.grad(lambda a, b, c, g: f(a, b, c, g).astype(jnp.float32).sum(), argnums=(0, 2)))(
            x, w, s, groups
        )

    for a, b in zip(derivatives(fast), derivatives(slow), strict=True):
        np.testing.assert_allclose(a, b, rtol=0.001, atol=0.001)
    dx = jnp.full_like(x, 0.125)
    ds = jnp.full_like(s, 0.01)
    jf = jax.jvp(lambda a, c: fast(a, w, c, groups), (x, s), (dx, ds))[1]
    jr = jax.jvp(lambda a, c: slow(a, w, c, groups), (x, s), (dx, ds))[1]
    np.testing.assert_allclose(jf, jr, rtol=0.001, atol=0.001)
