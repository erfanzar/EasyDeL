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

"""Depthwise-convolution state helpers for linear-attention modules.

Linear-attention architectures (GatedDeltaNet, KDA, Mamba, etc.) typically use a
short causal depthwise convolution to inject local context before the recurrent
kernel.  During training/prefill this convolution runs normally, but during
single-token decode it must be implemented as a rolling state buffer plus a
manual dot-product over the cached window.

This module extracts that logic into reusable free functions so that every
linear-attention model can share a single, tested implementation instead of
inlining it in each ``__call__``.

Key Functions:
    ``shift_conv_state_left``
        Shift a ``[batch, dim, d_conv]`` state buffer one position left and
        insert the newest token at the rightmost slot.

    ``apply_manual_depthwise_conv``
        Compute the depthwise-conv output from the cached state window and
        the conv kernel, with optional activation (default: SiLU).

    ``apply_conv_with_state``
        All-in-one helper that handles both the training path (full ``nn.Conv``)
        and the decode path (shift + manual conv), returning the updated state
        in both cases.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx as nn
from jaxtyping import Array, Float

from easydel.layers.norms import lowfloats


def shift_conv_state_left(
    conv_state: Array,
    new_value: Array,
) -> Array:
    """Shift a causal convolution cache one position left and append the newest token.

    This implements the rolling-buffer update for depthwise causal convolution
    state during autoregressive decoding.  The oldest (leftmost) time step is
    discarded and ``new_value`` is placed at the rightmost position.

    Args:
        conv_state: Existing convolution state of shape
            ``[batch, dim, d_conv]``.
        new_value: The new token's features, shape ``[batch, dim]``.

    Returns:
        Updated convolution state with the same shape as ``conv_state``.
    """
    return jnp.concatenate(
        (
            jax.lax.dynamic_slice_in_dim(conv_state, 1, conv_state.shape[-1] - 1, axis=-1),
            new_value[..., None],
        ),
        axis=-1,
    )


def apply_manual_depthwise_conv(
    conv_state: Array,
    kernel: Array,
    *,
    output_dtype: jnp.dtype,
    activation: tp.Callable[[Array], Array] | None = jax.nn.silu,
) -> Array:
    """Compute a depthwise convolution from the cached state and kernel.

    During decode (``seq_len == 1``), the full ``nn.Conv`` is replaced by a
    manual element-wise multiply-and-reduce over the state buffer.  This avoids
    the overhead of JAX's general convolution for a single output position.

    Upcasts to ``float32`` when either operand is a low-precision float
    (fp8 / fp4) to preserve numerical fidelity through the SiLU activation.

    Args:
        conv_state: Convolution state of shape ``[batch, dim, d_conv]``.
        kernel: Depthwise convolution kernel of shape ``[dim, d_conv]``
            (already transposed from Flax's ``[kernel_size, 1, dim]`` layout).
        output_dtype: Data type to cast the output to.
        activation: Pointwise activation applied after the dot product.
            Defaults to ``jax.nn.silu``.  Pass ``None`` to skip.

    Returns:
        Convolution output of shape ``[batch, dim]``, cast to ``output_dtype``.
    """
    if conv_state.dtype in lowfloats or kernel.dtype in lowfloats:
        compute_dtype = jnp.float32
    else:
        compute_dtype = jnp.promote_types(conv_state.dtype, kernel.dtype)
    conv_output = jnp.sum(
        conv_state.astype(compute_dtype) * kernel.astype(compute_dtype)[None, :, :],
        axis=-1,
    )
    if activation is not None:
        conv_output = activation(conv_output)
    return conv_output.astype(output_dtype)


def apply_conv_with_state(
    x: Float[Array, "batch seq_len dim"],
    conv_layer: nn.Conv,
    conv_state: Float[Array, "batch dim d_conv"] | None,
    *,
    is_inference: bool,
    d_conv: int,
    output_dtype: jnp.dtype,
    activation: tp.Callable[[Array], Array] | None = jax.nn.silu,
    reuse_partial_state: bool = False,
) -> tuple[
    Float[Array, "batch seq_len dim"],
    Float[Array, "batch dim d_conv"] | None,
]:
    """Run a causal depthwise convolution and maintain its rolling state.

    Transparently handles two execution modes:

    **Decode** (``is_inference=True`` and ``conv_state`` is not ``None``):
        Shifts the state buffer left, inserts the single input token, then
        computes the output via :func:`apply_manual_depthwise_conv`.

    **Train / prefill** (all other cases):
        Runs the full ``nn.Conv`` layer, then captures the trailing ``d_conv``
        inputs as the new state for future decode steps.

    The ``reuse_partial_state`` flag controls what happens when the input
    sequence is shorter than ``d_conv`` during prefill:

    - ``False`` (default, Qwen3Next convention): pad the state with zeros on
      the left.
    - ``True`` (KDA convention): keep the rightmost portion of the existing
      ``conv_state`` as prefix.

    Args:
        x: Input tensor of shape ``[batch, seq_len, dim]``.
        conv_layer: A Flax ``nn.Conv`` module configured for depthwise causal
            convolution.
        conv_state: Previous convolution state of shape
            ``[batch, dim, d_conv]``, or ``None`` if no cache exists yet.
        is_inference: Whether the model is in single-token decode mode.
        d_conv: Convolution kernel / state window size.
        output_dtype: Data type for the convolution output.
        activation: Pointwise activation applied to the convolution output.
            Defaults to ``jax.nn.silu``.
        reuse_partial_state: When ``True`` and ``seq_len < d_conv``, preserve
            the rightmost cached prefix instead of zero-padding.

    Returns:
        A tuple of ``(output, new_state)`` where:

        - ``output`` has shape ``[batch, seq_len, dim]``.
        - ``new_state`` has shape ``[batch, dim, d_conv]`` (or ``None`` when
          no cache was provided and the layer is in training mode without
          cache).
    """
    _, seq_len, _ = x.shape

    if is_inference and conv_state is not None:
        new_state = shift_conv_state_left(conv_state, x[:, 0, :].astype(conv_state.dtype))
        kernel = conv_layer.kernel.value.squeeze(1).T
        output = apply_manual_depthwise_conv(
            new_state,
            kernel,
            output_dtype=output_dtype,
            activation=activation,
        )[:, None, :]
        return output, new_state

    # Match the decode path by promoting the conv result before the activation.
    output = conv_layer(x).astype(output_dtype)
    if activation is not None:
        output = activation(output).astype(output_dtype)

    if seq_len >= d_conv:
        new_state = x[:, -d_conv:, :].transpose(0, 2, 1)
    else:
        x_transposed = x.transpose(0, 2, 1)
        if reuse_partial_state and conv_state is not None:
            new_state = jnp.concatenate(
                (
                    conv_state[:, :, seq_len:],
                    x_transposed,
                ),
                axis=-1,
            )
        else:
            new_state = jnp.pad(
                x_transposed,
                ((0, 0), (0, 0), (d_conv - seq_len, 0)),
            )

    return output, new_state
