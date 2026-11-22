"""Large W4A4 streaming outputs match exact widened integer arithmetic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.grouped_matmul_quant._channelwise import _dot, _mask_rows
from ejkernel.kernels._xla.grouped_matmul_quant._scale_rows import expand_group_scales
from ejkernel.kernels._xla.quantized_matmul._integer_quantization import quantize_rows
from ejkernel.modules import grouped_matmul_channelwise

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU prefill regression")


@pytest.mark.parametrize("shape", [(2560, 1280), (640, 2560)])
@pytest.mark.parametrize("route", ["balanced", "one", "empty"])
@pytest.mark.parametrize("platform", ["pallas", "xla"])
def test_w4_prefill_streaming(shape, route, platform):
    k, n = shape
    m = 1280 if route == "empty" else 81920
    e = 128
    rng = np.random.default_rng(82)
    sizes = np.zeros(e, np.int32)
    if route == "balanced":
        sizes[:] = 160
    elif route == "one":
        sizes[73] = 20477
    valid = int(sizes.sum())
    g = jnp.asarray(sizes)
    x = jnp.asarray(rng.normal(size=(m, k)), jnp.bfloat16).at[valid:].set(jnp.nan)
    w = jnp.asarray(rng.integers(-8, 8, (e, k, n), dtype=np.int8)).astype(jnp.int4)
    s = jnp.asarray(rng.uniform(-0.05, 0.05, (e, 1, n)), jnp.float32)

    def ref(a, b, c, d):
        q, rs = quantize_rows(_mask_rows(a, d), 4)
        raw = _dot(q.astype(jnp.int8), b.astype(jnp.int8), d, jnp.int32, None)
        return (raw.astype(jnp.float32) * rs * expand_group_scales(c, d, a.shape[0])).astype(jnp.bfloat16)

    f = jax.jit(lambda a, b, c, d: grouped_matmul_channelwise(a, b, c, d, activation_bits=4, platform=platform))
    actual = np.asarray(f(x, w, s, g))
    expected = np.asarray(jax.jit(ref)(x, w, s, g))
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()
    assert np.count_nonzero(actual[valid:]) == 0
