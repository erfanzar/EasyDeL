"""Measured prefill avoids materializing the raw MxN INT32 result."""

import jax
import jax.numpy as jnp
import pytest
from ejkernel.modules import grouped_matmul_channelwise

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU executable memory")


@pytest.mark.parametrize("bits", [4, 8])
def test_measured_prefill_uses_bounded_temporary_buffers(bits):
    args = (
        jax.ShapeDtypeStruct((81920, 640), jnp.bfloat16),
        jax.ShapeDtypeStruct((128, 640, 2560), jnp.int4 if bits == 4 else jnp.int8),
        jax.ShapeDtypeStruct((128, 1, 2560), jnp.float32),
        jax.ShapeDtypeStruct((128,), jnp.int32),
    )
    exe = (
        jax.jit(lambda a, b, c, d: grouped_matmul_channelwise(a, b, c, d, activation_bits=bits, platform="pallas"))
        .lower(*args)
        .compile()
    )
    assert exe.memory_analysis().temp_size_in_bytes < 256 * 1024 * 1024
