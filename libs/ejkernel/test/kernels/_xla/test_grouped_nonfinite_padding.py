"""Unused expert scales must not poison the documented zero padding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import grouped_matmul_channelwise


@pytest.mark.parametrize("platform", ["xla", "pallas"])
@pytest.mark.parametrize("bits", [4, 8, 16])
def test_unused_nonfinite_scale_padding_and_ad(platform, bits):
    if platform == "pallas" and jax.default_backend() != "tpu":
        pytest.skip("TPU streaming")
    x = jnp.ones((32, 128), jnp.bfloat16)
    w = jnp.ones((3, 128, 128), jnp.int4 if bits == 4 else jnp.int8)
    s = jnp.ones((3, 1, 128), jnp.float32).at[1:].set(jnp.nan)
    g = jnp.array([3, 0, 0], jnp.int32)
    f = jax.jit(lambda a, c: grouped_matmul_channelwise(a, w, c, g, activation_bits=bits, platform=platform))
    y, dy = jax.jvp(f, (x, s), (jnp.ones_like(x), jnp.ones_like(s)))
    assert np.isfinite(y).all()
    assert np.isfinite(dy).all()
    np.testing.assert_array_equal(y[3:], 0)
    np.testing.assert_array_equal(dy[3:], 0)
    gx, gs = jax.grad(lambda a, c: f(a, c).astype(jnp.float32).sum(), argnums=(0, 1))(x, s)
    assert np.isfinite(gx).all()
    assert np.isfinite(gs).all()
    np.testing.assert_array_equal(gx[3:], 0)
    np.testing.assert_array_equal(gs[1:], 0)
