"""XLA grouped wrappers should not build decode scale maps using scatter."""

import jax
import jax.numpy as jnp
import pytest
from ejkernel.modules import grouped_matmul_channelwise, grouped_matmul_w8a8


@pytest.mark.parametrize("mode", ["a16", "a4", "a8", "legacy"])
def test_decode_scale_mapping_has_no_scatter(mode):
    x = jnp.ones((8, 128), jnp.bfloat16)
    w = jnp.ones((3, 128, 128), jnp.int4 if mode == "a4" else jnp.int8)
    s = jnp.ones((3, 1, 128), jnp.float32)
    g = jnp.array([2, 0, 3], jnp.int32)
    if mode == "legacy":
        f = jax.jit(grouped_matmul_w8a8)
    else:
        bits = {"a16": 16, "a4": 4, "a8": 8}[mode]
        f = jax.jit(lambda a, b, c, d: grouped_matmul_channelwise(a, b, c, d, activation_bits=bits))
    hlo = f.lower(x, w, s, g).compiler_ir(dialect="stablehlo").operation.get_asm()
    assert "stablehlo.scatter" not in hlo
