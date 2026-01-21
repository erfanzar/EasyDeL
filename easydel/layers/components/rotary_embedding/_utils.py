# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@jax.named_scope("easydel-rotary-yarn-find-correction-dim")
def _yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    """
    Calculates the correction dimension for YaRN scaling.

    Internal helper function for YaRN.

    Args:
        num_rotations (int): Number of rotations.
        dim (int): The dimension of the embeddings.
        base (float, optional): The base value for positional encoding. Defaults to 10000.
        max_position_embeddings (int, optional): The maximum sequence length. Defaults to 2048.

    Returns:
        float: The calculated correction dimension.
    """
    return (
        dim
        * jnp.log(
            max_position_embeddings / (num_rotations * 2 * jnp.pi),
        )
    ) / (2 * jnp.log(base))


@jax.named_scope("easydel-rotary-yarn-find-correction-range")
def _yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> tuple[int, int]:
    """
    Finds the correction range for YaRN scaling based on low and high rotation frequencies.

    Internal helper function for YaRN.

    Args:
        low_rot (int): Lower rotation frequency boundary.
        high_rot (int): Higher rotation frequency boundary.
        dim (int): The dimension of the embeddings.
        base (float, optional): The base value for positional encoding. Defaults to 10000.
        max_position_embeddings (int, optional): The maximum sequence length. Defaults to 2048.

    Returns:
        tp.Tuple[int, int]: A tuple containing the lower and upper bounds of the correction range,
                            clipped between 0 and dim-1.
    """
    hr = jnp.ceil(
        _yarn_find_correction_dim(
            high_rot,
            dim,
            base,
            max_position_embeddings,
        )
    )
    lr = jnp.floor(
        _yarn_find_correction_dim(
            low_rot,
            dim,
            base,
            max_position_embeddings,
        )
    )
    return jax.lax.max(lr, 0.0), jax.lax.min(hr, jnp.array(dim - 1, dtype=jnp.float32))


@jax.named_scope("easydel-rotary-yarn-linear-ramp-mask")
def _yarn_linear_ramp_mask(
    low: float,
    high: float,
    dim: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """
    Creates a linear ramp mask for YaRN scaling.

    Internal helper function for YaRN. Generates a mask that ramps linearly from 0 to 1
    between the `low` and `high` dimension indices.

    Args:
        low (float): The starting dimension index for the ramp.
        high (float): The ending dimension index for the ramp.
        dim (int): The total dimension of the mask.
        dtype (jnp.dtype): The data type for the mask array.

    Returns:
        jnp.ndarray: A 1D array of shape (dim,) representing the linear ramp mask,
                     clipped between 0 and 1.
    """
    high = jax.lax.cond(low == high, lambda x: x + 0.001, lambda x: x, high)
    linear_func = (jnp.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = jnp.clip(linear_func, 0, 1)
    return ramp_func


@jax.named_scope("easydel-rotary-yarn-get-mscale")
def _yarn_get_mscale(scale: float = 1) -> float:
    """
    Calculates the mscale factor for YaRN context extension method.

    Internal helper function for YaRN.

    Args:
        scale (float, optional): The scaling factor. Defaults to 1.

    Returns:
        float: The calculated mscale value. Returns 1.0 if scale <= 1.
    """
    if scale <= 1:
        return 1.0
    return 0.1 * jnp.log(scale) + 1.0


@jax.named_scope("easydel-rotary-rotate-neox")
def _rotate_neox(x: Float[Array, "... seq_len head_dim"]) -> Float[Array, "... seq_len head_dim"]:
    """
    Applies the Neox-style rotation to the input array.

    Splits the last dimension in half and concatenates the negated second half
    with the first half.

    Args:
        x (jnp.ndarray): The input array.

    Returns:
        jnp.ndarray: The rotated array.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


@jax.named_scope("easydel-rotary-rotate-gptj")
def _rotate_gptj(x: Float[Array, "... seq_len head_dim"]) -> Float[Array, "... seq_len head_dim"]:
    """
    Applies the GPT-J-style rotation to the input array.

    Interleaves the negated odd-indexed elements with the even-indexed elements.

    Args:
        x (jnp.ndarray): The input array.

    Returns:
        jnp.ndarray: The rotated array.
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = jnp.stack((-x2, x1), axis=-1)
    return x.reshape((*x.shape[:-2], -1))


@jax.named_scope("easydel-rotary-apply-rotary-emb")
def _apply_rotary_emb(
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    is_neox_style: bool,
) -> jnp.ndarray:
    """
    Applies rotary positional embedding to the input tensor.

    Args:
        x (jnp.ndarray): Input tensor, e.g., query or key. Expected shape
                         [..., num_tokens, head_size] or similar.
        cos (jnp.ndarray): Cosine components of the embedding. Expected shape
                           compatible for broadcasting with `x` after rotation,
                           e.g., [..., num_tokens, head_size//2].
        sin (jnp.ndarray): Sine components of the embedding. Expected shape
                           compatible for broadcasting with `x` after rotation,
                           e.g., [..., num_tokens, head_size//2].
        is_neox_style (bool): Whether to use Neox-style rotation (`_rotate_neox`)
                              or GPT-J-style rotation (`_rotate_gptj`).

    Returns:
        jnp.ndarray: The tensor with rotary embeddings applied.
    """
    cos = cos[:, :, None].astype(x.dtype)
    sin = sin[:, :, None].astype(x.dtype)
    assert sin.ndim == x.ndim
    if is_neox_style:
        x1, x2 = jnp.split(x, 2, axis=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox_style:
        return jnp.concatenate((o1, o2), axis=-1)
    else:
        return jnp.stack((o1, o2), axis=-1).reshape(x.shape)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    """
    Calculates the mscale factor, potentially used by Deepseek-YaRN or similar methods.

    Allows specifying an additional `mscale` parameter compared to `_yarn_get_mscale`.

    Args:
        scale (float, optional): The scaling factor. Defaults to 1.
        mscale (float, optional): An additional scaling parameter. Defaults to 1.

    Returns:
        float: The calculated mscale value. Returns 1.0 if scale <= 1.
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0
