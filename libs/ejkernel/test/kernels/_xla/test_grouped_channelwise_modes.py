"""Independent NumPy reference for the public CHANNELWISE operation."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.modules import grouped_matmul_channelwise


def reference(x, codes, scales, sizes, bits):
    x = np.asarray(x, dtype=np.float32)
    codes = np.asarray(codes, dtype=np.int32)
    out = np.zeros((len(x), codes.shape[-1]), np.float32)
    start = 0
    for expert, size in enumerate(sizes):
        rows = x[start : start + size]
        if bits == 16:
            out[start : start + size] = (rows @ codes[expert].astype(np.float32)) * scales[expert]
        else:
            qmax = 2 ** (bits - 1) - 1
            amax = np.max(np.abs(rows), axis=-1, keepdims=True)
            scale = np.where(amax == 0, 1.0, amax / qmax)
            q = np.clip(np.rint(rows / scale), -qmax, qmax).astype(np.int32)
            out[start : start + size] = (q @ codes[expert]).astype(np.float32) * scale * scales[expert]
        start += size
    return out


@pytest.mark.parametrize("weight_bits,bits", [(4, 16), (8, 16), (4, 4), (8, 8), (4, 8)])
@pytest.mark.parametrize("sizes", [(2, 0, 5), (0, 7, 0)])
@pytest.mark.parametrize("compiled", [False, True])
def test_modes(weight_bits, bits, sizes, compiled):
    rng = np.random.default_rng(71)
    x = rng.normal(size=(7, 16)).astype(np.float32)
    x[1] = 0
    c = rng.integers(-(2 ** (weight_bits - 1)), 2 ** (weight_bits - 1), (3, 16, 5), dtype=np.int8)
    s = rng.uniform(0.01, 0.3, (3, 1, 5)).astype(np.float32)
    fn = functools.partial(grouped_matmul_channelwise, activation_bits=bits, preferred_element_type=jnp.float32)
    if compiled:
        fn = jax.jit(fn)
    result = fn(
        jnp.asarray(x),
        jnp.asarray(c, dtype=jnp.int4 if weight_bits == 4 else jnp.int8),
        jnp.asarray(s),
        jnp.asarray(sizes, jnp.int32),
    )
    np.testing.assert_allclose(result, reference(x, c, s, sizes, bits), rtol=2e-5, atol=2e-5)
    assert result.dtype == jnp.float32


@pytest.mark.parametrize("bits,dtype", [(3, jnp.int8), (32, jnp.int4), (4, jnp.int8), (8, jnp.uint8), (16, jnp.float32)])
def test_invalid(bits, dtype):
    with pytest.raises((ValueError, TypeError)):
        grouped_matmul_channelwise(
            jnp.ones((2, 4)), jnp.ones((1, 4, 3), dtype), jnp.ones((1, 1, 3)), jnp.array([2]), activation_bits=bits
        )


@pytest.mark.parametrize("weight_bits,bits", [(4, 16), (8, 16), (4, 4), (8, 8), (4, 8)])
def test_ad_contract(weight_bits, bits):
    rng = np.random.default_rng(43)
    x = rng.normal(size=(5, 8)).astype(np.float32)
    x[0] = 0
    c = rng.integers(-7, 8, (3, 8, 4), dtype=np.int8)
    s = rng.uniform(0.01, 0.3, (3, 1, 4)).astype(np.float32)
    sizes = np.array([2, 0, 3], np.int32)
    codes = jnp.asarray(c, jnp.int4 if weight_bits == 4 else jnp.int8)
    def fn(x, s):
        return grouped_matmul_channelwise(
            x, codes, s, jnp.asarray(sizes), activation_bits=bits, preferred_element_type=jnp.float32
        )
    dx = rng.normal(size=x.shape).astype(np.float32)
    ds = rng.normal(size=s.shape).astype(np.float32)
    _, tangent = jax.jit(lambda x, s, dx, ds: jax.jvp(fn, (x, s), (dx, ds)))(
        jnp.asarray(x), jnp.asarray(s), jnp.asarray(dx), jnp.asarray(ds)
    )
    expected = reference(dx, c, s, sizes, 16) + reference(x, c, ds, sizes, bits)
    np.testing.assert_allclose(tangent, expected, rtol=2e-5, atol=2e-5)
    gx, gs = jax.jit(jax.grad(lambda x, s: fn(x, s).sum(), argnums=(0, 1)))(jnp.asarray(x), jnp.asarray(s))
    row_weights = np.repeat(c.astype(np.float32) * s, sizes, axis=0)
    np.testing.assert_allclose(gx, row_weights.sum(axis=-1), rtol=2e-5, atol=2e-5)
    expected_gs = np.zeros_like(s)
    base = reference(x, c, np.ones_like(s), sizes, bits)
    start = 0
    for expert, size in enumerate(sizes):
        expected_gs[expert, 0] = base[start : start + size].sum(axis=0)
        start += size
    np.testing.assert_allclose(gs, expected_gs, rtol=2e-5, atol=2e-5)


def test_default_bfloat16():
    out = grouped_matmul_channelwise(
        jnp.ones((2, 4), jnp.bfloat16), jnp.ones((1, 4, 3), jnp.int4), jnp.ones((1, 1, 3)), jnp.array([2], jnp.int32)
    )
    assert out.dtype == jnp.bfloat16
    np.testing.assert_array_equal(out.astype(jnp.float32), np.full((2, 3), 4.0, np.float32))


def test_registry_and_exports():
    from ejkernel.modules.operations import grouped_matmul_channelwise as operation
    from ejkernel.modules.operations.grouped_matmul import GroupedMatmulChannelwise
    from ejkernel.ops import Kernel

    assert operation is grouped_matmul_channelwise
    assert isinstance(GroupedMatmulChannelwise(), Kernel)
    assert callable(kernel_registry.get("grouped_matmul_channelwise", platform=Platform.XLA, backend=Backend.ANY))
