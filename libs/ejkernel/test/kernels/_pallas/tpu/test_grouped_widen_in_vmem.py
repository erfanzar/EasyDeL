"""Exact INT4-grid computation with INT8 arithmetic and packed RHS storage."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.grouped_matmulv3._pallas_impl import TileSizes, grouped_matmulv3_pallas_impl


@pytest.mark.parametrize("bits", [4, 8])
def test_native_integer_lhs_with_streamed_rhs(bits):
    rng = np.random.default_rng(95)
    x = jnp.asarray(rng.integers(-7, 8, (32, 128), dtype=np.int8))
    w = jnp.asarray(rng.integers(-7, 8, (4, 128, 256), dtype=np.int8)).astype(jnp.int4 if bits == 4 else jnp.int8)
    g = jnp.array([8, 0, 8, 8], jnp.int32)
    got = jax.jit(
        lambda a, b, c: grouped_matmulv3_pallas_impl(
            a,
            b,
            c,
            maybe_quantize_lhs=False,
            tile_info=TileSizes(tile_m=64, tile_k=128, tile_n=256),
            preferred_element_type=jnp.float32,
            acc_dtype=jnp.float32,
        )
    )(x, w, g)
    hx = np.asarray(x).astype(np.int32)
    hw = np.asarray(w).astype(np.int32)
    expected = np.concatenate(
        [hx[:8] @ hw[0], hx[8:16] @ hw[2], hx[16:24] @ hw[3], np.zeros((8, 256), np.int32)], axis=0
    )
    np.testing.assert_array_equal(got, expected)


def test_integer_lhs_rejects_in_kernel_fractional_rhs_scales():
    x = jnp.ones((32, 128), jnp.int8)
    w = jnp.ones((1, 128, 128), jnp.int4)
    scales = jnp.full((1, 1, 1, 128), 0.125, jnp.float32)
    with pytest.raises(ValueError, match=r"integer lhs.*outside"):
        grouped_matmulv3_pallas_impl(
            x,
            w,
            jnp.array([32], jnp.int32),
            rhs_scale=scales,
            maybe_quantize_lhs=False,
            tile_info=TileSizes(tile_m=32, tile_k=128, tile_n=128),
            preferred_element_type=jnp.float32,
            acc_dtype=jnp.float32,
        ).block_until_ready()
