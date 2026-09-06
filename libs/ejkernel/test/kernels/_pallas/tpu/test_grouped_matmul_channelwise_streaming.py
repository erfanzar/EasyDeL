"""Real TPU gate for opt-in streaming A16 (never emulate hardware on CPU)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import grouped_matmul_channelwise

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires real TPU")


def make_inputs(bits, shape):
    e, m, k, n = shape
    rng = np.random.default_rng(38)
    x = jnp.asarray(rng.normal(size=(m, k)), jnp.bfloat16)
    bound = (1 << (bits - 1)) - 1
    host = rng.integers(-bound, bound + 1, size=(e, k, n), dtype=np.int8)
    codes = jnp.asarray(host).astype(jnp.int4 if bits == 4 else jnp.int8)
    scales = jnp.asarray(rng.uniform(-0.02, 0.02, (e, 1, n)), jnp.float32)
    sizes = np.zeros(e, np.int32)
    # Include empty groups, skew, and group boundaries not aligned to tile M.
    sizes[0] = m // 2 + 1
    sizes[-1] = m - sizes[0]
    groups = jnp.asarray(sizes)
    return x, codes, scales, groups, host, sizes


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("shape", [(4, 16, 160, 256), (128, 24, 2560, 1280), (128, 80, 640, 2560)])
def test_streaming_primal_real_shapes(bits, shape):
    x, codes, scales, groups, host, sizes = make_inputs(bits, shape)
    expected = np.zeros((shape[1], shape[3]), np.float32)
    start = 0
    for g, rows in enumerate(sizes):
        if rows:
            expected[start : start + rows] = (
                np.asarray(x[start : start + rows], np.float32) @ host[g].astype(np.float32) * np.asarray(scales[g])
            )
        start += rows
    f = jax.jit(
        lambda a, b, c, d: grouped_matmul_channelwise(a, b, c, d, platform="pallas", preferred_element_type=jnp.float32)
    )
    np.testing.assert_allclose(f(x, codes, scales, groups), expected, rtol=1e-4, atol=2e-4)


@pytest.mark.parametrize("bits", [4, 8])
def test_streaming_jvp_vjp_after_cached_jit(bits):
    x, codes, scales, groups, _, _ = make_inputs(bits, (4, 16, 160, 256))
    f = jax.jit(
        lambda a, b, c, d: grouped_matmul_channelwise(a, b, c, d, platform="pallas", preferred_element_type=jnp.float32)
    )
    def ref(a, b, c, d):
        return grouped_matmul_channelwise(a, b, c, d, platform="xla", preferred_element_type=jnp.float32)
    f(x, codes, scales, groups).block_until_ready()
    dx = jnp.ones_like(x) * 0.375
    ds = jnp.ones_like(scales) * 0.01712345
    _, got = jax.jvp(lambda a, c: f(a, codes, c, groups), (x, scales), (dx, ds))
    _, want = jax.jvp(lambda a, c: ref(a, codes, c, groups), (x, scales), (dx, ds))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=2e-4)
    actual = jax.jit(jax.grad(lambda a, b, c, d: f(a, b, c, d).sum(), argnums=(0, 2)))
    expected = jax.jit(jax.grad(lambda a, b, c, d: ref(a, b, c, d).sum(), argnums=(0, 2)))
    for got, want in zip(actual(x, codes, scales, groups), expected(x, codes, scales, groups), strict=False):
        np.testing.assert_allclose(got.astype(jnp.float32), want.astype(jnp.float32), rtol=1e-5, atol=2e-4)


def test_explicit_tiles_and_default_bf16_output():
    x, codes, scales, groups, _, _ = make_inputs(4, (4, 16, 160, 256))
    f = jax.jit(lambda a, b, c, d: grouped_matmul_channelwise(a, b, c, d, platform="pallas", tiling=(16, 128, 128)))
    got = f(x, codes, scales, groups)
    want = grouped_matmul_channelwise(x, codes, scales, groups)
    assert got.dtype == jnp.bfloat16
    np.testing.assert_allclose(got.astype(jnp.float32), want.astype(jnp.float32), rtol=0.01, atol=0.01)
