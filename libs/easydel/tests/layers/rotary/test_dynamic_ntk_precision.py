"""Dynamic-NTK cache accuracy against an independent float64 reference."""

import jax
import numpy as np
import pytest
from easydel.layers.rotary import compute_basic_inv_frequencies, compute_dynamic_frequencies, get_frequencies
from jax import numpy as jnp


@pytest.mark.parametrize("base", [4096.0, 1000000.0])
@pytest.mark.parametrize("rotary_dim", [12, 64, 128])
def test_jitted_dynamic_cache_preserves_float32_phase_accuracy(base, rotary_dim):
    original, length, factor = 32, 128, 2.0
    scaling = {"rope_type": "dynamic", "factor": factor, "original_max_position_embeddings": original}
    cache = jax.jit(lambda: get_frequencies(rotary_dim, rotary_dim, length, base, scaling))()
    positions = np.arange(length, dtype=np.float64)
    context = np.maximum(positions + 1, original)
    adjusted_base = base * (factor * context / original - (factor - 1)) ** (rotary_dim / (rotary_dim - 2))
    inverse = adjusted_base[:, None] ** (-np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
    angles = positions[:, None] * inverse
    expected = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1)
    assert cache.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(cache), expected, rtol=2e-5, atol=2e-5)
    # Preserve the original-context contract as well as the extended rows.
    unscaled = get_frequencies(rotary_dim, rotary_dim, original, base)
    np.testing.assert_allclose(np.asarray(cache[:original]), np.asarray(unscaled), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("base", [4096.0, 1000000.0])
def test_jitted_unscaled_spectrum_is_correctly_rounded_float32(base):
    rotary_dim = 128
    expected = np.power(base, -np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim).astype(np.float32)
    actual = jax.jit(lambda: compute_basic_inv_frequencies(base, rotary_dim))()
    assert actual.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_dynamic_cache_preserves_traced_base_and_factor_derivatives():
    # Static cache construction must not turn array-valued public arguments
    # into Python scalars or drop their gradients. Small positions keep this
    # regression focused on the transformation contract, not long-phase error.
    length, original, dim = 8, 4, 4

    def evaluate(base, factor):
        return compute_dynamic_frequencies(base, dim, original, factor, max_positions=length)

    base, factor = 4096.0, 2.0
    result = jax.jit(evaluate)(jnp.asarray(base), jnp.asarray(factor))
    positions = np.arange(length, dtype=np.float64)
    context = np.maximum(positions + 1, original)
    stretch = factor * context / original - (factor - 1)
    spectrum = base ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)
    inverse = spectrum[None, :] * stretch[:, None] ** (-np.arange(0, dim, 2) / (dim - 2))
    angles = positions[:, None] * inverse
    expected = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=2e-5, atol=2e-5)
    base_grad, factor_grad = jax.jit(jax.grad(lambda b, f: evaluate(b, f).sum(), argnums=(0, 1)))(
        jnp.asarray(base), jnp.asarray(factor)
    )
    angle_grad = -np.sin(angles) + np.cos(angles)
    expected_base_grad = np.sum(angle_grad * angles * (-np.arange(0, dim, 2) / dim) / base)
    expected_factor_grad = np.sum(
        angle_grad * angles * (-np.arange(0, dim, 2) / (dim - 2)) * ((context / original - 1) / stretch)[:, None]
    )
    np.testing.assert_allclose(np.asarray(base_grad), expected_base_grad, rtol=2e-5, atol=1e-8)
    np.testing.assert_allclose(np.asarray(factor_grad), expected_factor_grad, rtol=2e-5, atol=2e-5)
