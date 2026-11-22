"""Avoid expensive floating remainder in rowwise integer parity correction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.quantized_matmul._integer_quantization import quantize_rows


@pytest.mark.parametrize("bits", [4, 8])
def test_parity_correction_needs_no_floating_remainder(bits):
    hlo = jax.jit(lambda a: quantize_rows(a, bits)).lower(jax.ShapeDtypeStruct((128, 2048), jnp.bfloat16)).as_text()
    assert "stablehlo.remainder" not in hlo


@pytest.mark.parametrize("bits", [4, 8])
def test_parity_grid_codes_remain_exact(bits):
    bound = (1 << (bits - 1)) - 1
    grid = np.linspace(-bound, bound, 4 * bound + 1, dtype=np.float32)
    a = jnp.asarray(np.tile(grid, (8, 1)), jnp.bfloat16)
    codes, scale = jax.jit(lambda x: quantize_rows(x, bits))(a)
    np.testing.assert_array_equal(
        np.asarray(codes).astype(np.int8), np.round(np.asarray(a).astype(np.float64)).astype(np.int8)
    )
    np.testing.assert_array_equal(scale, np.ones((8, 1), np.float32))
