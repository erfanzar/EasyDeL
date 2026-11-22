"""Joint primal/JVP must preserve the documented BF16 activation-dot boundary."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.quantized_matmul._channelwise import channelwise_quantized_matmul


@pytest.mark.parametrize("bits", [4, 8])
def test_joint_jvp_preserves_bf16_dot_rounding(bits):
    rng = np.random.default_rng(62)
    x = jnp.asarray(rng.normal(size=(128, 256)), jnp.bfloat16)
    w = jnp.asarray(rng.integers(-(1 << (bits - 1)), 1 << (bits - 1), (256, 512), dtype=np.int16)).astype(
        jnp.int4 if bits == 4 else jnp.int8
    )
    s = jnp.asarray(rng.uniform(-0.1, 0.1, (1, 512)), jnp.float32)
    dx = jnp.full_like(x, 0.125)
    ds = jnp.zeros_like(s)
    f = jax.jit(
        lambda a, b, c: channelwise_quantized_matmul(
            a, b, c, quantize_activations=True, activation_bits=bits, prefill_threshold=0
        )
    )
    primal, tangent = jax.jvp(lambda a, c: f(a, w, c), (x, s), (dx, ds))
    assert np.isfinite(np.asarray(primal)).all()
    dot = np.asarray(dx).astype(np.float32) @ np.asarray(w).astype(np.float32)
    rounded_dot = np.asarray(jnp.asarray(dot, jnp.bfloat16)).astype(np.float32)
    expected = jnp.asarray(rounded_dot * np.asarray(s), jnp.bfloat16)
    np.testing.assert_array_equal(tangent, expected)
