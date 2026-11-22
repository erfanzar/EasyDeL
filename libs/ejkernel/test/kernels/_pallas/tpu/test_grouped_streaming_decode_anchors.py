"""Full-output comparisons on tuned matrix families with skewed/empty routes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import grouped_matmul_channelwise


@pytest.mark.parametrize("bits,abits", [(4, 16), (8, 16), (4, 4), (8, 8)])
@pytest.mark.parametrize("m,k,n", [(24, 2560, 1280), (80, 640, 2560)])
def test_decode_anchor_full_output(bits, abits, m, k, n):
    rng = np.random.default_rng(65)
    x = jnp.asarray(rng.normal(size=(m, k)), jnp.bfloat16)
    w = jnp.asarray(rng.integers(-(1 << (bits - 1)), 1 << (bits - 1), (128, k, n), dtype=np.int8)).astype(
        jnp.int4 if bits == 4 else jnp.int8
    )
    s = jnp.asarray(rng.uniform(-0.025, 0.025, (128, 1, n)), jnp.float32)
    gs = np.zeros(128, np.int32)
    valid = m - 3
    gs[0] = valid // 2
    gs[17] = valid - valid // 2
    g = jnp.asarray(gs)
    x = x.at[valid:].set(jnp.nan)

    def make(platform):
        return jax.jit(
            lambda a, b, c, d: grouped_matmul_channelwise(
                a, b, c, d, activation_bits=abits, platform=platform, preferred_element_type=jnp.float32
            )
        )

    actual = np.asarray(make("pallas")(x, w, s, g))
    expected = np.asarray(make("xla")(x, w, s, g))
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[valid:], 0)
    if abits != 16:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=5e-4)
