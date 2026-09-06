"""Static dispatch tails and unassigned experts have finite zero gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import grouped_matmul_channelwise


@pytest.mark.parametrize("platform,bits", [("xla", 16), ("xla", 4), ("xla", 8), ("pallas", 16)])
@pytest.mark.parametrize("sizes", [(2, 1), (0, 0)])
@pytest.mark.parametrize("poison_tail", [False, True])
def test_dispatch_padding_and_empty_groups(platform, bits, sizes, poison_tail):
    if platform == "pallas" and jax.default_backend() != "tpu":
        pytest.skip("real TPU streaming implementation required")
    x = jnp.ones((8, 128), jnp.bfloat16)
    if poison_tail:
        x = x.at[sum(sizes) :].set(jnp.nan)
    codes = jnp.ones((2, 128, 128), jnp.int4)
    scales = jnp.full((2, 1, 128), 0.25, jnp.float32)
    groups = jnp.array(sizes, jnp.int32)
    f = jax.jit(
        lambda a, s: grouped_matmul_channelwise(
            a, codes, s, groups, activation_bits=bits, platform=platform, preferred_element_type=jnp.float32
        )
    )
    got = f(x, scales)
    want = np.zeros((8, 128), np.float32)
    want[: sum(sizes)] = 32
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    gx, gs = jax.jit(jax.grad(lambda a, s: f(a, s).sum(), argnums=(0, 1)))(x, scales)
    assert np.isfinite(gx).all() and np.isfinite(gs).all()
    np.testing.assert_array_equal(gx[sum(sizes) :], 0)
    expected_scale = np.asarray(sizes, dtype=np.float32)[:, None, None] * 128 * np.ones((2, 1, 128), np.float32)
    np.testing.assert_allclose(gs, expected_scale, rtol=1e-6, atol=1e-6)
