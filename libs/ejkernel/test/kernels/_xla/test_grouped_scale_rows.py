"""Small decode scale expansion must match repeat without scatter lowering."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

impl = importlib.import_module("ejkernel.kernels._pallas.tpu.grouped_matmul_channelwise._interface")


@pytest.mark.parametrize(
    "sizes,count",
    [
        ([0, 3, 0, 2], 8),
        ([0, 0, 0, 0], 8),
        ([8, 0, 0, 0], 8),
        ([2, 0, 3, 1], 6),
        ([0, 2, 0, 1, 0], 8),
        ([0], 1),
        ([1], 1),
        ([0, 0, 4, 0, 0], 4),
    ],
)
def test_scale_row_expansion(sizes, count):
    expand = getattr(impl, "_expand_channel_scales", None)
    assert callable(expand), "missing bounded decode scale expansion"
    s = jnp.arange(len(sizes) * 128, dtype=jnp.float32).reshape(len(sizes), 1, 128) / 128
    gs = jnp.array(sizes, jnp.int32)
    f = jax.jit(lambda a, b: expand(a, b, count))
    def ref(a, b):
        return jnp.repeat(a[:, 0, :], b, axis=0, total_repeat_length=count)
    np.testing.assert_array_equal(f(s, gs), ref(s, gs))
    np.testing.assert_array_equal(jax.grad(lambda a: f(a, gs).sum())(s), jax.grad(lambda a: ref(a, gs).sum())(s))
    hlo = f.lower(s, gs).compiler_ir(dialect="stablehlo").operation.get_asm()
    assert "stablehlo.scatter" not in hlo


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32, jnp.float64])
@pytest.mark.parametrize("count", [8, 256])
def test_shared_scale_expansion_preserves_dtype_and_fallback(dtype, count):
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("requires JAX_ENABLE_X64=1")
    from ejkernel.kernels._xla.grouped_matmul_quant._scale_rows import expand_group_scales

    s = (jnp.arange(384, dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32).reshape(3, 1, 128) / 127).astype(
        dtype
    )
    gs = jnp.array([0, 3, 2], jnp.int32)
    actual = jax.jit(lambda a, b: expand_group_scales(a, b, count))(s, gs)
    expected = jnp.repeat(s[:, 0, :], gs, axis=0, total_repeat_length=count)
    assert actual.dtype == s.dtype
    np.testing.assert_array_equal(actual, expected)
