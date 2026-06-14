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

"""XLA/JAX reference implementation for fused conv-state decode."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array


def _shift_conv_state_left(
    conv_state: jnp.ndarray,
    new_value: jnp.ndarray,
) -> jnp.ndarray:
    """Shift a convolution cache left by one along the temporal axis and append a new column.

    Drops the oldest temporal slice (index 0 of the last axis) and appends ``new_value`` as the
    newest slice, keeping the window length ``d_conv`` constant.

    Args:
        conv_state: Convolution cache of shape ``[num_slots, conv_dim, d_conv]``; the last axis is
            the temporal window ordered oldest-to-newest.
        new_value: New per-channel values to append, shape ``[num_slots, conv_dim]``.

    Returns:
        The updated cache of shape ``[num_slots, conv_dim, d_conv]`` whose final column is
        ``new_value`` and whose remaining columns are the trailing ``d_conv - 1`` columns of the
        input ``conv_state``.
    """
    return jnp.concatenate(
        [
            jax.lax.dynamic_slice_in_dim(conv_state, 1, conv_state.shape[-1] - 1, axis=-1),
            new_value[..., None],
        ],
        axis=-1,
    )


def _apply_manual_depthwise_conv(
    conv_state: jnp.ndarray,
    kernel: jnp.ndarray,
    *,
    output_dtype: jnp.dtype,
    activation: Callable[[Array], Array] | None = jax.nn.silu,
) -> jnp.ndarray:
    """Compute a depthwise (per-channel) convolution over the cached temporal window.

    Multiplies the cache by the per-channel kernel and sums over the ``d_conv`` window. The kernel
    is broadcast across the ``num_slots`` axis. Computation runs in the promoted dtype of
    ``conv_state`` and ``kernel`` to limit precision loss before casting to ``output_dtype``.

    Args:
        conv_state: Convolution cache of shape ``[num_slots, conv_dim, d_conv]``.
        kernel: Depthwise convolution weights of shape ``[conv_dim, d_conv]``.
        output_dtype: Dtype the result is cast to before returning.
        activation: Elementwise activation applied to the summed output, or ``None`` to skip it.
            Defaults to ``jax.nn.silu``.

    Returns:
        The convolution output of shape ``[num_slots, conv_dim]`` (activation applied when
        provided) cast to ``output_dtype``.
    """
    compute_dtype = jnp.promote_types(conv_state.dtype, kernel.dtype)
    conv_output = jnp.sum(
        conv_state.astype(compute_dtype) * kernel.astype(compute_dtype)[None, :, :],
        axis=-1,
    )
    if activation is not None:
        conv_output = activation(conv_output)
    return conv_output.astype(output_dtype)


def fused_conv_decode(
    conv_state: jnp.ndarray,
    new_tokens: jnp.ndarray,
    kernel: jnp.ndarray,
    *,
    output_dtype: jnp.dtype,
    activation: Callable[[Array], Array] | None = jax.nn.silu,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fused conv-state shift, depthwise convolution, and activation.

    Args:
        conv_state: Current conv state ``[num_slots, conv_dim, d_conv]``.
        new_tokens: New token embeddings ``[num_slots, conv_dim]``.
        kernel: Depthwise conv kernel ``[conv_dim, d_conv]``.
        output_dtype: Desired dtype for the convolution output.
        activation: Activation function to apply after convolution.

    Returns:
        ``(updated_state, conv_output)`` with shapes
        ``[num_slots, conv_dim, d_conv]`` and ``[num_slots, conv_dim]``.
    """
    updated_state = _shift_conv_state_left(conv_state, new_tokens)
    conv_output = _apply_manual_depthwise_conv(
        updated_state,
        kernel,
        output_dtype=output_dtype,
        activation=activation,
    )
    return updated_state, conv_output
