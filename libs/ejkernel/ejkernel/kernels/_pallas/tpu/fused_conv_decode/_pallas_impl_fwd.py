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

"""TPU Pallas implementation for fused conv-state decode."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float


def _fused_conv_decode_kernel(
    conv_state_ref,
    new_tokens_ref,
    kernel_ref,
    updated_state_ref,
    conv_output_ref,
):
    """Pallas kernel that fuses conv-state shift, depthwise convolution, and SiLU.

    This kernel is designed for TPU execution inside a ``pallas_call`` with
    grid ``(conv_dim_tiles,)``. It tiles over the ``conv_dim`` axis in chunks
    determined by the caller's ``conv_tile`` block size, processing all slots
    within each tile. Each grid step handles one ``conv_dim`` tile.

    For each tile the kernel performs three fused operations:

    1. Shift: drops the oldest token from the conv state and appends the new
       token along the ``d_conv`` axis.
    2. Depthwise conv: computes the per-channel dot product of the updated
       state with the kernel weights.
    3. SiLU activation: applies the SiLU (Swish) non-linearity to the conv
       output.

    All computation is done in float32; results are cast back to the output ref
    dtypes on write.

    Args:
        conv_state_ref: Input ref holding the current conv-state tile of shape
            ``[num_slots, conv_tile, d_conv]``. Read only.
        new_tokens_ref: Input ref holding the new-token tile of shape
            ``[num_slots, conv_tile]``. Read only.
        kernel_ref: Input ref holding the depthwise kernel tile of shape
            ``[conv_tile, d_conv]``. Read only.
        updated_state_ref: Output ref written with the shifted conv state of
            shape ``[num_slots, conv_tile, d_conv]``.
        conv_output_ref: Output ref written with the post-activation convolution
            result of shape ``[num_slots, conv_tile]``.

    Returns:
        None. Results are written in place to ``updated_state_ref`` and
        ``conv_output_ref``.
    """
    _tile_idx = pl.program_id(0)
    state = conv_state_ref[:, :, :].astype(jnp.float32)
    token = new_tokens_ref[:, :].astype(jnp.float32)
    kern = kernel_ref[:, :].astype(jnp.float32)

    new_state = jnp.concatenate([state[:, :, 1:], token[:, :, None]], axis=2)
    conv_out = jnp.sum(new_state * kern[None, :, :], axis=-1)
    conv_out = conv_out * jax.nn.sigmoid(conv_out)

    updated_state_ref[:, :, :] = new_state.astype(updated_state_ref.dtype)
    conv_output_ref[:, :] = conv_out.astype(conv_output_ref.dtype)


def fused_conv_decode(
    conv_state: Float[Array, "num_slots conv_dim d_conv"],
    new_tokens: Float[Array, "num_slots conv_dim"],
    kernel: Float[Array, "conv_dim d_conv"],
    *,
    output_dtype: jnp.dtype,
) -> tuple[
    Float[Array, "num_slots conv_dim d_conv"],
    Float[Array, "num_slots conv_dim"],
]:
    """Pallas-accelerated fused conv-state update targeting TPU.

    Tiles over the ``conv_dim`` axis in chunks of ``conv_tile = min(conv_dim, 128)``,
    which must divide ``conv_dim``. All slots are processed within each tile.

    Args:
        conv_state: Current conv state ``[num_slots, conv_dim, d_conv]``, where
            the last axis holds the most recent ``d_conv`` tokens per channel.
        new_tokens: New token embeddings ``[num_slots, conv_dim]`` to append.
        kernel: Depthwise conv kernel ``[conv_dim, d_conv]`` (per-channel
            weights).
        output_dtype: Desired dtype for the convolution output array.

    Returns:
        A tuple ``(updated_state, conv_output)`` where ``updated_state`` is the
        shifted conv state ``[num_slots, conv_dim, d_conv]`` (same dtype as
        ``conv_state``) and ``conv_output`` is the post-SiLU convolution result
        ``[num_slots, conv_dim]`` with dtype ``output_dtype``.

    Raises:
        ValueError: If ``conv_dim`` is not divisible by the chosen tile size
            (``min(conv_dim, 128)``).
    """
    num_slots, conv_dim, d_conv = conv_state.shape
    conv_tile = min(conv_dim, 128)
    if conv_dim % conv_tile != 0:
        raise ValueError(f"conv_dim={conv_dim} not divisible by {conv_tile}")
    num_tiles = conv_dim // conv_tile

    out_state_shape = jax.ShapeDtypeStruct((num_slots, conv_dim, d_conv), dtype=conv_state.dtype)
    out_conv_shape = jax.ShapeDtypeStruct((num_slots, conv_dim), dtype=output_dtype)

    updated_state, conv_output = pl.pallas_call(
        _fused_conv_decode_kernel,
        grid=(num_tiles,),
        in_specs=[
            pl.BlockSpec((num_slots, conv_tile, d_conv), lambda t: (0, t, 0)),
            pl.BlockSpec((num_slots, conv_tile), lambda t: (0, t)),
            pl.BlockSpec((conv_tile, d_conv), lambda t: (t, 0)),
        ],
        out_specs=[
            pl.BlockSpec((num_slots, conv_tile, d_conv), lambda t: (0, t, 0)),
            pl.BlockSpec((num_slots, conv_tile), lambda t: (0, t)),
        ],
        out_shape=[out_state_shape, out_conv_shape],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(conv_state, new_tokens, kernel)

    return updated_state, conv_output
