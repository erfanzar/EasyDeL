"""Output scaling must occur after the full raw integer contraction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.grouped_matmulv3._pallas_impl import (
    TileSizes,
)
from ejkernel.kernels._pallas.tpu.grouped_matmulv3._pallas_impl import (
    grouped_matmulv3_pallas_impl as gmm,
)

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="real TPU epilogue")


@pytest.mark.parametrize("bits,k", [(4, 128), (4, 160), (8, 384)])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_output_epilogue_partial_rows_and_k(bits, k, dtype):
    rng = np.random.default_rng(205)
    m, e, n = 96, 5, 256
    x = jnp.asarray(rng.integers(-7, 8, (m, k)), jnp.int8)
    w = jnp.asarray(rng.integers(-8, 8, (e, k, n)), jnp.int4 if bits == 4 else jnp.int8)
    sizes = np.array([3, 0, 29, 13, 5], np.int32)
    g = jnp.asarray(sizes)
    row = rng.uniform(-0.02, 0.02, (m, 1)).astype(np.float32)
    channel = rng.uniform(-0.1, 0.1, (e, 1, n)).astype(np.float32)
    channel[1] = np.nan  # empty expert must not leak into adjacent sublanes
    row[5] = np.nan
    r = jnp.asarray(row)
    s = jnp.asarray(channel)
    f = jax.jit(
        lambda a, b, c, d, h: gmm(
            a,
            b,
            h,
            tile_info=TileSizes(32, 128, 128),
            maybe_quantize_lhs=False,
            acc_dtype=jnp.int32,
            preferred_element_type=dtype,
            output_row_scale=c,
            output_channel_scale=d,
        )
    )
    actual = np.asarray(f(x, w, r, s, g)).astype(np.float32)
    expected = np.zeros((m, n), np.float32)
    start = 0
    for expert, count in enumerate(sizes):
        end = start + count
        raw = np.asarray(x[start:end], np.int32) @ np.asarray(w[expert], np.int32)
        expected[start:end] = raw.astype(np.float32) * row[start:end] * channel[expert]
        start = end
    expected = np.asarray(jnp.asarray(expected, dtype)).astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert np.count_nonzero(actual[start:]) == 0


def test_output_epilogue_signed_zero_and_empty():
    x = jnp.zeros((64, 128), jnp.int8)
    w = jnp.ones((3, 128, 128), jnp.int4)
    r = jnp.ones((64, 1), jnp.float32)
    s = -jnp.ones((3, 1, 128), jnp.float32)
    f = jax.jit(
        lambda g: gmm(
            x,
            w,
            g,
            tile_info=TileSizes(32, 128, 128),
            maybe_quantize_lhs=False,
            acc_dtype=jnp.int32,
            preferred_element_type=jnp.bfloat16,
            output_row_scale=r,
            output_channel_scale=s,
        )
    )
    out = np.asarray(f(jnp.array([3, 13, 1], jnp.int32))).astype(np.float32)
    assert np.signbit(out[:17]).all(), "partial-row merge must preserve owned negative zero"
    np.testing.assert_array_equal(f(jnp.zeros((3,), jnp.int32)), 0)


def test_full_range_int8_accumulates_before_float_conversion():
    rng = np.random.default_rng(207)
    m, e, k, n = 64, 3, 2304, 128
    hx = rng.integers(112, 128, (m, k), dtype=np.int8)
    hw = rng.integers(112, 128, (e, k, n), dtype=np.int8)
    sizes = np.array([3, 13, 1], np.int32)
    row = rng.uniform(0.01, 0.03, (m, 1)).astype(np.float32)
    channel = rng.uniform(-0.2, 0.2, (e, 1, n)).astype(np.float32)
    result = gmm(
        jnp.asarray(hx),
        jnp.asarray(hw),
        jnp.asarray(sizes),
        tile_info=TileSizes(32, 128, 128),
        acc_dtype=jnp.int32,
        maybe_quantize_lhs=False,
        preferred_element_type=jnp.float32,
        output_row_scale=jnp.asarray(row),
        output_channel_scale=jnp.asarray(channel),
    )
    expected = np.zeros((m, n), np.float32)
    start = 0
    for expert, count in enumerate(sizes):
        end = start + count
        raw = hx[start:end].astype(np.int64) @ hw[expert].astype(np.int64)
        assert (raw > 2**24).all()
        expected[start:end] = raw.astype(np.float32) * row[start:end] * channel[expert]
        start = end
    np.testing.assert_array_equal(result, expected)
