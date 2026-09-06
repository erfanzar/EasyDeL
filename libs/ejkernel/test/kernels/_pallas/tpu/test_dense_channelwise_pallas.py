"""Explicit dense Pallas integer path: full-range operands and compiled AD."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import channelwise_quantized_matmul


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("m", [64, 128, 192])
def test_dense_pallas_explicit_matches_xla(bits, m):
    rng = np.random.default_rng(68)
    x = jnp.asarray(rng.normal(size=(m, 256)), jnp.bfloat16).at[0].set(0)
    bound = (1 << (bits - 1)) - 1
    x = x.at[1, :8].set(jnp.array([1, -3, 5, -7, 9, -11, 13, 2 * bound], jnp.bfloat16))
    w = jnp.asarray(rng.integers(-(1 << (bits - 1)), 1 << (bits - 1), (256, 256), dtype=np.int16)).astype(
        jnp.int4 if bits == 4 else jnp.int8
    )
    s = jnp.asarray(rng.uniform(-0.05, 0.05, (1, 256)), jnp.float32)

    def make(platform):
        return jax.jit(
            lambda a, b, c: channelwise_quantized_matmul(
                a, b, c, quantize_activations=True, activation_bits=bits, prefill_threshold=0, platform=platform
            )
        )

    fast = make("pallas")
    ref = make("xla")
    np.testing.assert_array_equal(fast(x, w, s), ref(x, w, s))
    dx = jnp.full_like(x, 0.125)
    ds = jnp.full_like(s, 0.01)
    np.testing.assert_array_equal(
        jax.jvp(lambda a, c: fast(a, w, c), (x, s), (dx, ds))[1], jax.jvp(lambda a, c: ref(a, w, c), (x, s), (dx, ds))[1]
    )

    def grad(f):
        return jax.grad(lambda a, c: f(a, w, c).astype(jnp.float32).sum(), argnums=(0, 1))(x, s)

    for a, b in zip(grad(fast), grad(ref), strict=True):
        np.testing.assert_allclose(a, b, rtol=0.001, atol=0.001)
