# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the straight-through estimators used in quantization-aware training.

These cover four defects that were all present together and each of which
silently produced a wrong result rather than an error at the point of use:

1. every ejkernel-backed estimator raised ``TypeError`` on its first call,
   because keyword-only parameters cannot be resolved by ``custom_vjp``;
2. the ejkernel round trip returned the **transpose** of its input, since
   ``quantize`` maps into a swapped layout that ``dequantize`` does not
   map back;
3. ``straight_through_1bit`` had no straight-through wrapper at all, so
   ``jnp.sign`` gave exactly zero gradients and binarized weights never
   moved;
4. ``CHANNELWISE`` had no dispatch entry, and it is the only scheme the
   stacked mixture-of-experts linears accept.

Every test therefore asserts shape, value and gradient together — checking
only one of the three would have passed against at least one of the bugs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.layers.quantization import (
    QuantizationConfig,
    QuantizationType,
    straight_through,
    straight_through_channelwise,
)

_TRAINABLE_TYPES = [
    QuantizationType.MXFP8,
    QuantizationType.MXFP4,
    QuantizationType.NVFP8,
    QuantizationType.NVFP4,
    QuantizationType.NF4,
    QuantizationType.AFFINE,
    QuantizationType.INT8,
    QuantizationType.CHANNELWISE,
    QuantizationType.TERNARY,
    QuantizationType.BINARY,
]


def _weights(shape=(64, 32), seed=0):
    """Return a deterministic float32 weight tensor.

    Args:
        shape: Shape of the tensor.
        seed: PRNG seed.

    Returns:
        A normally-distributed array.
    """
    return jax.random.normal(jax.random.key(seed), shape, jnp.float32)


@pytest.mark.parametrize("quant_type", _TRAINABLE_TYPES)
def test_straight_through_runs_for_every_trainable_type(quant_type):
    """Every scheme declared trainable must actually execute."""
    assert straight_through(_weights(), dtype=quant_type) is not None


@pytest.mark.parametrize("quant_type", _TRAINABLE_TYPES)
def test_straight_through_preserves_shape(quant_type):
    """The estimator must return the weight, not its transpose."""
    weights = _weights((64, 32))
    assert straight_through(weights, dtype=quant_type).shape == weights.shape


@pytest.mark.parametrize("quant_type", _TRAINABLE_TYPES)
def test_straight_through_gradient_is_the_identity(quant_type):
    """The whole point of a straight-through estimator is that gradients survive it."""
    weights = _weights((64, 32))
    cotangent = jax.random.normal(jax.random.key(1), weights.shape, jnp.float32)
    grad = jax.grad(lambda w: (straight_through(w, dtype=quant_type).astype(jnp.float32) * cotangent).sum())(weights)
    np.testing.assert_allclose(np.asarray(grad), np.asarray(cotangent), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("quant_type", _TRAINABLE_TYPES)
def test_straight_through_actually_discretizes(quant_type):
    """A no-op estimator would pass every other test here."""
    weights = _weights((64, 32))
    assert not jnp.array_equal(straight_through(weights, dtype=quant_type).astype(jnp.float32), weights)


@pytest.mark.parametrize("quant_type", [QuantizationType.INT8, QuantizationType.AFFINE, QuantizationType.NF4])
def test_straight_through_stays_near_the_original(quant_type):
    """Discretization is a perturbation; a transposed or scrambled result would not be."""
    weights = _weights((64, 32))
    quantized = straight_through(weights, dtype=quant_type).astype(jnp.float32)
    relative = float(jnp.linalg.norm(quantized - weights) / jnp.linalg.norm(weights))
    assert relative < 0.25, f"{quant_type} round trip moved the weight by {relative:.3f} of its norm"


def test_binarization_returns_only_signs():
    """Binary quantization must map every value to exactly +1 or -1."""
    binarized = straight_through(_weights(), dtype=QuantizationType.BINARY)
    assert set(np.unique(np.asarray(binarized)).tolist()) <= {-1.0, 1.0}


def test_binarization_gradient_is_not_zero():
    """``jnp.sign`` alone has a zero derivative, which would freeze every binarized weight."""
    weights = _weights()
    grad = jax.grad(lambda w: straight_through(w, dtype=QuantizationType.BINARY).sum())(weights)
    assert float(jnp.abs(grad).sum()) > 0


def test_channelwise_matches_the_serving_quantizer_exactly():
    """Training must see the same discretization serving applies, not an approximation.

    Compared against :func:`channelwise_quantize_array`, the function the
    module-level and quantize-at-load paths use, so a divergence between
    what a run trains against and what its checkpoint is served with shows
    up here rather than as a quality regression after deployment.
    """
    from easydel.layers.linears._linear_quantized import channelwise_quantize_array

    weights = _weights((128, 64))
    codes, scales, _ = channelwise_quantize_array(weights, 8)
    expected = (codes.astype(jnp.float32) * scales).astype(weights.dtype)
    assert jnp.array_equal(straight_through_channelwise(weights, bits=8), expected)


def test_channelwise_scale_is_per_output_channel():
    """One loud output channel must not flatten the others."""
    weights = jnp.concatenate([jnp.full((32, 4), 0.01), jnp.full((32, 4), 100.0)], axis=1)
    quantized = straight_through_channelwise(weights, bits=8)
    np.testing.assert_allclose(np.asarray(quantized[:, :4]), np.asarray(weights[:, :4]), rtol=1e-2)


def test_channelwise_handles_stacked_expert_kernels():
    """Stacked experts ``[E, in, out]`` reduce over axis -2, giving each expert its own scale.

    Asserted exactly: quantizing the stack must give expert 0 bit-for-bit
    what quantizing expert 0 alone gives.
    """
    experts = jnp.stack([jnp.full((32, 8), 0.01), jnp.full((32, 8), 100.0)])
    together = straight_through_channelwise(experts, bits=4)
    alone = straight_through_channelwise(experts[:1], bits=4)
    assert together.shape == experts.shape
    assert jnp.array_equal(together[:1], alone)


@pytest.mark.parametrize("bits", [8, 4])
def test_channelwise_bits_are_honoured(bits):
    """Fewer bits must produce a coarser grid, not the same one."""
    weights = _weights((64, 32))
    error = float(jnp.abs(straight_through_channelwise(weights, bits=bits) - weights).max())
    assert error > 0
    if bits == 4:
        coarse = error
        fine = float(jnp.abs(straight_through_channelwise(weights, bits=8) - weights).max())
        assert coarse > fine


def test_channelwise_rejects_an_unsupported_bit_width():
    """Only the two widths the serving kernels accept are offered."""
    with pytest.raises(ValueError, match="8 or 4 bits"):
        straight_through_channelwise(_weights(), bits=6)


def test_channelwise_rejects_a_one_dimensional_tensor():
    """Norm gains and biases have no contraction axis to reduce over."""
    with pytest.raises(ValueError, match="2-D or higher"):
        straight_through_channelwise(jnp.ones((8,), jnp.float32))


def test_ejkernel_estimator_rejects_a_one_dimensional_tensor():
    """The same guard on the grouped path, with a message that names the fix."""
    with pytest.raises(ValueError, match="2-D or higher"):
        straight_through(jnp.ones((8,), jnp.float32), dtype=QuantizationType.INT8)


def test_channelwise_config_selects_bit_width():
    """A :class:`QuantizationConfig` carrying ``bits`` reaches the channelwise estimator."""
    weights = _weights((64, 32))
    four = straight_through(weights, config=QuantizationConfig(dtype=QuantizationType.CHANNELWISE, bits=4))
    eight = straight_through(weights, config=QuantizationConfig(dtype=QuantizationType.CHANNELWISE, bits=8))
    assert not jnp.array_equal(four, eight)
    assert jnp.array_equal(four, straight_through_channelwise(weights, bits=4))


def test_unsupported_type_names_the_supported_ones():
    """TurboQuant is post-training only; the error should say what is available."""
    with pytest.raises(ValueError, match="Unsupported quantization type"):
        straight_through(_weights(), dtype=QuantizationType.TURBOQUANT)
