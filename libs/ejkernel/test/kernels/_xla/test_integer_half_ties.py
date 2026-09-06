"""BF16 half-way activation values must round to even across CPU/TPU."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import channelwise_quantized_matmul, grouped_matmul_channelwise


@pytest.mark.parametrize("grouped", [False, True])
def test_bf16_half_ties_round_to_even(grouped):
    h = np.zeros((8, 32), np.float32)
    h[:, :5] = [3.515625, 1.7578125, -1.7578125, -3.515625, 0]
    x = jnp.asarray(h, jnp.bfloat16)
    q = jnp.eye(32, dtype=jnp.int4)
    if grouped:
        def f(a):
            return grouped_matmul_channelwise(
                    a,
                    q[None],
                    jnp.ones((1, 1, 32)),
                    jnp.array([8], jnp.int32),
                    activation_bits=4,
                    preferred_element_type=jnp.float32,
                )
    else:
        def f(a):
            return channelwise_quantized_matmul(
                    a, q, jnp.ones((1, 32)), quantize_activations=True, activation_bits=4, prefill_threshold=0
                )
    got = np.asarray(jax.jit(f)(x)).astype(np.float32)
    want = np.zeros_like(h)
    want[:, :5] = np.array([7, 4, -4, -7, 0], np.float32) * (3.515625 / 7)
    if not grouped:
        want = np.asarray(jnp.asarray(want, jnp.bfloat16)).astype(np.float32)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
