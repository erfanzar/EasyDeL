# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""XLA kernel tests for fused conv-state decode."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.kernels._registry import kernel_registry
from ejkernel.kernels._xla.fused_conv_decode import fused_conv_decode


def test_fused_conv_decode_kernel_is_registered_and_signature_compatible():
    """Assert the ``fused_conv_decode`` XLA kernel is registered with a consistent signature.

    Looks the operation up in the global :data:`kernel_registry` for ``platform="xla"`` and
    ``backend="any"``, asserting a non-``None`` implementation is returned, then asserts that
    :meth:`kernel_registry.validate_signatures` reports the registered backends for the op share a
    compatible signature.

    Raises:
        AssertionError: If no XLA implementation is registered, or if the cross-backend signature
            validation fails.
    """
    impl = kernel_registry.get("fused_conv_decode", platform="xla", backend="any")
    assert impl is not None
    assert kernel_registry.validate_signatures("fused_conv_decode")


def test_fused_conv_decode_xla_matches_manual_reference():
    """Check the XLA ``fused_conv_decode`` against a hand-written shift + conv + SiLU reference.

    Builds a small ``[num_slots, conv_dim, d_conv]`` conv state of ``arange`` values, all-ones new
    tokens, and a constant ``0.25`` depthwise kernel, then calls :func:`fused_conv_decode` with the
    default SiLU activation. The expected state is the input rolled left by one with ``new_tokens``
    appended as the newest column; the expected output is the per-channel weighted sum over the
    window followed by :func:`jax.nn.silu`. Verifies the returned state and output shapes
    (``[num_slots, conv_dim, d_conv]`` and ``[num_slots, conv_dim]``) and that both match the
    reference within :func:`jnp.allclose` tolerance.

    Raises:
        AssertionError: If the updated state or convolution output mismatches the manual reference
            in shape or value.
    """
    num_slots, conv_dim, d_conv = 2, 4, 3
    conv_state = jnp.arange(num_slots * conv_dim * d_conv, dtype=jnp.float32).reshape(num_slots, conv_dim, d_conv)
    new_tokens = jnp.ones((num_slots, conv_dim), dtype=jnp.float32)
    kernel = jnp.ones((conv_dim, d_conv), dtype=jnp.float32) * 0.25

    updated_state, conv_output = fused_conv_decode(
        conv_state,
        new_tokens,
        kernel,
        output_dtype=jnp.float32,
    )

    expected_state = jnp.concatenate(
        [conv_state[..., 1:], new_tokens[..., None]],
        axis=-1,
    )
    expected_output = jnp.sum(expected_state * kernel[None, :, :], axis=-1)
    expected_output = jax.nn.silu(expected_output)

    assert updated_state.shape == conv_state.shape
    assert conv_output.shape == (num_slots, conv_dim)
    assert jnp.allclose(updated_state, expected_state)
    assert jnp.allclose(conv_output, expected_output)


def test_fused_conv_decode_xla_no_activation():
    """Check the XLA ``fused_conv_decode`` with ``activation=None`` skips the nonlinearity.

    Calls :func:`fused_conv_decode` with ``activation=None`` on all-ones state, doubled new tokens,
    and a unit kernel, and asserts the convolution output equals the raw per-channel windowed sum
    (the shift-then-sum reference) with no SiLU applied. The updated state is ignored.

    Raises:
        AssertionError: If the convolution output does not match the unactivated reference sum.
    """
    num_slots, conv_dim, d_conv = 1, 2, 2
    conv_state = jnp.ones((num_slots, conv_dim, d_conv), dtype=jnp.float32)
    new_tokens = jnp.ones((num_slots, conv_dim), dtype=jnp.float32) * 2
    kernel = jnp.ones((conv_dim, d_conv), dtype=jnp.float32)

    _, conv_output = fused_conv_decode(
        conv_state,
        new_tokens,
        kernel,
        output_dtype=jnp.float32,
        activation=None,
    )

    expected_state = jnp.concatenate(
        [conv_state[..., 1:], new_tokens[..., None]],
        axis=-1,
    )
    expected_output = jnp.sum(expected_state * kernel[None, :, :], axis=-1)

    assert jnp.allclose(conv_output, expected_output)
