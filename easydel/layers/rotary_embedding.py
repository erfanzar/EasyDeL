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
import typing as tp

import chex
import jax
import jax.numpy as jnp
from flax import nnx as nn
from jaxtyping import Array, Float

from easydel.utils.compiling_utils import ejit


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


AVAILABLE_ROPE_TYPES = {}
"""A dictionary to store registered RoPE (Rotary Position Embedding) types and their configurations."""


def rope_wraper(type):  # noqa
    """
    A decorator factory that registers a RotaryEmbedding class under a specific type name.

    This allows retrieving RoPE configurations by type name later. It also sets
    basic __str__ and __repr__ for the decorated class.

    Args:
        type (str): The name to register the RoPE class under (e.g., "linear", "yarn").

    Returns:
        Callable: A decorator function that takes a RotaryEmbedding class, registers it,
                  and returns the class.
    """

    def w(rope: RotaryEmbedding):
        """
        Decorator function that registers the RoPE class.

        Args:
            rope (RotaryEmbedding): The RotaryEmbedding class to register.

        Returns:
            RotaryEmbedding: The registered RotaryEmbedding class.
        """
        properties = {k: v for k, v in rope.__dict__.items()}
        AVAILABLE_ROPE_TYPES[type] = properties
        rope.__str__ = lambda cls: str(cls.__class__.__name__)
        rope.__repr__ = lambda cls: repr(cls.__class__.__name__)
        rope._type = type
        return rope

    return w


@jax.named_scope("easydel-rotary-compute-basic-inv-frequencies")
def compute_basic_inv_frequencies(base: int, rotary_dim: int):
    """
    Computes the inverse frequencies for standard RoPE.

    Args:
        base (int): The base value for the geometric progression of frequencies.
        rotary_dim (int): The dimension of the rotary embeddings.

    Returns:
        jnp.ndarray: An array of inverse frequencies of shape (rotary_dim // 2,).
    """
    return 1.0 / (base ** (jnp.arange(0, rotary_dim, 2, dtype="f4") / rotary_dim))


@jax.named_scope("easydel-rotary-compute-yarn-inv-frequencies")
def compute_yarn_inv_frequencies(
    base: float,
    rotary_dim: int,
    beta_fast: float,
    beta_slow: float,
    max_position_embeddings: int,
    scaling_factor: float,
    extrapolation_factor: float,
) -> jnp.ndarray:
    """
    Computes the inverse frequencies for YaRN scaled RoPE.

    Combines interpolation and extrapolation frequencies based on correction ranges.

    Args:
        base (float): The base value for positional encoding.
        rotary_dim (int): The dimension of the rotary embeddings.
        beta_fast (float): YaRN parameter for faster rotating dimensions.
        beta_slow (float): YaRN parameter for slower rotating dimensions.
        max_position_embeddings (int): Original maximum sequence length before scaling.
        scaling_factor (float): The factor by which the context length is scaled.
        extrapolation_factor (float): YaRN parameter controlling extrapolation strength.

    Returns:
        jnp.ndarray: An array of YaRN-adjusted inverse frequencies of shape (rotary_dim // 2,).
    """
    pos_freqs = base ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
    low, high = _yarn_find_correction_range(
        low_rot=beta_fast,
        high_rot=beta_slow,
        dim=rotary_dim,
        base=base,
        max_position_embeddings=max_position_embeddings,
    )
    inv_frequencies_mask = (
        1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=jnp.float32)
    ) * extrapolation_factor
    inv_frequencies = inv_freq_interpolation * (1 - inv_frequencies_mask) + inv_freq_extrapolation * inv_frequencies_mask
    return inv_frequencies


@jax.named_scope("easydel-rotary-compute-llama3-inv-frequencies")
def compute_llama3_inv_frequencies(
    base,
    rotary_dim,
    low_freq_factor,
    high_freq_factor,
    orig_max_position,
    scaling_factor,
):
    """
    Computes the inverse frequencies for Llama3-style scaled RoPE.

    Adjusts frequencies based on wavelength thresholds and a smoothing factor.

    Args:
        base (float): The base value for positional encoding.
        rotary_dim (int): The dimension of the rotary embeddings.
        low_freq_factor (float): Factor for adjusting low-frequency components.
        high_freq_factor (float): Factor for adjusting high-frequency components.
        orig_max_position (int): Original maximum sequence length before scaling.
        scaling_factor (float): The overall scaling factor applied.

    Returns:
        jnp.ndarray: An array of Llama3-adjusted inverse frequencies of shape (rotary_dim // 2,).
    """
    inv_freqs = compute_basic_inv_frequencies(base, rotary_dim)
    low_freq_wavelen = orig_max_position / low_freq_factor
    high_freq_wavelen = orig_max_position / high_freq_factor

    wave_len = 2 * jnp.pi / inv_freqs
    if low_freq_factor != high_freq_factor:
        smooth = (orig_max_position / wave_len - low_freq_factor) / (high_freq_factor - low_freq_factor)
    else:
        smooth = 0
    new_freqs = jnp.where(
        wave_len < high_freq_wavelen,
        inv_freqs,
        jnp.where(
            wave_len > low_freq_wavelen,
            inv_freqs / scaling_factor,
            (1 - smooth) * inv_freqs / scaling_factor + smooth * inv_freqs,
        ),
    )
    return new_freqs


@jax.named_scope("easydel-rotary-compute-basic-frequencies")
def compute_basic_frequencies(
    base: int,
    rotary_dim: int,
    max_position_embeddings: int,
):
    """
    Computes the basic RoPE frequencies (cos and sin values) for all positions.

    Args:
        base (int): The base value for the geometric progression of frequencies.
        rotary_dim (int): The dimension of the rotary embeddings.
        max_position_embeddings (int): The maximum sequence length.

    Returns:
        jnp.ndarray: A frequency cache tensor of shape
                     (max_position_embeddings, rotary_dim). Contains concatenated
                     cos and sin values.
    """
    inv = compute_basic_inv_frequencies(base, rotary_dim)
    freqs = jnp.einsum(
        "i,j -> ij",
        jnp.arange(max_position_embeddings, dtype=jnp.float32),
        inv,
    )
    freqs = jnp.concatenate([jnp.cos(freqs), jnp.sin(freqs)], axis=-1)
    return freqs


@jax.named_scope("easydel-rotary-compute-linear-frequencies")
def compute_linear_frequencies(
    base: int,
    rotary_dim: int,
    max_position_embeddings: int,
    scaling_factors: list[float],
):
    """
    Computes RoPE frequencies using linear scaling for potentially multiple factors.

    This function computes frequency caches for each scaling factor and concatenates them.
    Note: This implementation seems designed for a specific use case where different
    parts of a sequence might use different scaling factors, determined by offsets.
    If only one scaling factor is used, it behaves like standard linear scaling.

    Args:
        base (int): The base value for the geometric progression of frequencies.
        rotary_dim (int): The dimension of the rotary embeddings.
        max_position_embeddings (int): The base maximum sequence length before scaling.
        scaling_factors (tp.Union[tp.List[float], float]): A single scaling factor or a list
                                                            of scaling factors.

    Returns:
        jnp.ndarray: A frequency cache tensor. If multiple scaling factors are provided,
                     the caches are concatenated along the position dimension. Shape
                     is (total_scaled_length, rotary_dim).
    """
    if not isinstance(scaling_factors, list):
        scaling_factors = [scaling_factors]
    inv_freq = compute_basic_inv_frequencies(
        base=base,
        rotary_dim=rotary_dim,
    )
    cache_list: list[jnp.ndarray] = []
    offsets: list[int] = []

    for scaling_factor in scaling_factors:
        max_len = max_position_embeddings * scaling_factor
        t = jnp.arange(max_len, dtype=jnp.float32)
        t = t / scaling_factor

        freqs = jnp.einsum("i,j -> ij", t, inv_freq)
        cache = jnp.concatenate([jnp.cos(freqs), jnp.sin(freqs)], axis=-1)
        if not cache_list:
            offset = 0
        else:
            last_offset = offsets[-1]
            next_max_len = cache_list[-1].shape[0]
            offset = last_offset + next_max_len
        offsets.append(offset)
        cache_list.append(cache)

    assert len(scaling_factors) == len(offsets)
    return jnp.concatenate(cache_list, axis=0)


@jax.named_scope("easydel-rotary-compute-dynamic-frequencies")
def compute_dynamic_frequencies(
    base: int,
    rotary_dim: int,
    max_position_embeddings: int,
    scaling_factor: float,
):
    """
    Computes RoPE frequencies using Dynamic NTK scaling.

    Adjusts the 'base' dynamically based on the scaling factor.

    Args:
        base (int): The initial base value before dynamic adjustment.
        rotary_dim (int): The dimension of the rotary embeddings.
        max_position_embeddings (int): The base maximum sequence length before scaling.
        scaling_factor (float): The scaling factor applied to the sequence length.

    Returns:
        jnp.ndarray: A frequency cache tensor of shape
                     (max_position_embeddings * scaling_factor, rotary_dim).
    """
    max_length = max_position_embeddings * scaling_factor
    base = base * ((scaling_factor * max_length / max_position_embeddings) - (scaling_factor - 1)) ** (
        rotary_dim / (rotary_dim - 2)
    )
    inv_frequencies = compute_basic_inv_frequencies(base=base, rotary_dim=rotary_dim)
    times = jnp.arange(max_length, dtype=jnp.float32)
    frequencies = jnp.einsum("i,j -> ij", times, inv_frequencies)
    return jnp.concatenate([jnp.cos(frequencies), jnp.sin(frequencies)], -1)


@jax.named_scope("easydel-rotary-compute-yarn-frequencies")
def compute_yarn_frequencies(
    base: float,
    rotary_dim: int,
    beta_fast: float,
    beta_slow: float,
    max_position_embeddings: int,
    scaling_factor: float,
    extrapolation_factor: float,
    attn_factor: float,
) -> jnp.ndarray:
    """
    Computes RoPE frequencies using the YaRN scaling method.

    Includes adjustments based on YaRN parameters and applies an mscale factor.

    Args:
        base (float): The base value for positional encoding.
        rotary_dim (int): The dimension of the rotary embeddings.
        beta_fast (float): YaRN parameter for faster rotating dimensions.
        beta_slow (float): YaRN parameter for slower rotating dimensions.
        max_position_embeddings (int): Original maximum sequence length before scaling.
        scaling_factor (float): The factor by which the context length is scaled.
        extrapolation_factor (float): YaRN parameter controlling extrapolation strength.
        attn_factor (float): YaRN parameter scaling the attention outputs.

    Returns:
        jnp.ndarray: A frequency cache tensor of shape
                     (max_position_embeddings * scaling_factor, rotary_dim).
    """
    inv_freq = compute_yarn_inv_frequencies(
        base=base,
        rotary_dim=rotary_dim,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        max_position_embeddings=max_position_embeddings,
        scaling_factor=scaling_factor,
        extrapolation_factor=extrapolation_factor,
    )
    t = jnp.arange(max_position_embeddings * scaling_factor, dtype=jnp.float32)
    freqs = jnp.einsum("i,j -> ij", t, inv_freq)
    mscale = _yarn_get_mscale(scaling_factor) * attn_factor
    cos = jnp.cos(freqs) * mscale
    sin = jnp.sin(freqs) * mscale
    return jnp.concatenate([cos, sin], axis=-1)


@jax.named_scope("easydel-rotary-compute-phi3-frequencies")
def compute_phi3_frequencies(
    base,
    head_size,
    rotary_dim,
    max_position_embeddings,
    original_max_position_embeddings,
    short_factor,
    long_factor,
):
    """
    Computes RoPE frequencies using the Phi-3 LongRoPE scaling method.

    Applies different scaling factors based on whether the target length is
    shorter or longer than the original max length. Includes a scaling factor
    adjustment based on the ratio of target length to original length.

    Args:
        base (float): The base value for positional encoding.
        head_size (int): The dimension of each attention head.
        rotary_dim (int): The dimension of the rotary embeddings. Must equal head_size for Phi-3.
        max_position_embeddings (int): The target maximum sequence length after scaling.
        original_max_position_embeddings (int): Original maximum sequence length before scaling.
        short_factor (tp.List[float]): Scaling factors for frequencies when
                                        max_position_embeddings <= original_max_position_embeddings.
        long_factor (tp.List[float]): Scaling factors for frequencies when
                                       max_position_embeddings > original_max_position_embeddings.

    Returns:
        jnp.ndarray: A frequency cache tensor of shape (1, max_position_embeddings, rotary_dim).

    Raises:
        ValueError: If rotary_dim does not equal head_size.
    """
    if rotary_dim != head_size:
        raise ValueError(f"rotary_dim != head_size ({rotary_dim}!={head_size})")
    if max_position_embeddings > original_max_position_embeddings:
        ext_factors = jnp.array(long_factor, dtype=jnp.float32)
    else:
        ext_factors = jnp.array(short_factor, dtype=jnp.float32)

    inv_freq_shape = jnp.arange(0, head_size, 2, dtype=jnp.int32).astype(jnp.float32) / head_size
    inv_freq = 1.0 / (ext_factors * (base**inv_freq_shape))

    inv_freq_expanded = jnp.expand_dims(inv_freq, (0, 2)).astype(jnp.float32)
    position_ids = jnp.arange(max_position_embeddings, dtype=jnp.int32).reshape(1, -1)
    position_ids_expanded = jnp.expand_dims(position_ids, 1).astype(jnp.float32)

    freqs = (inv_freq_expanded @ position_ids_expanded).swapaxes(1, 2)
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    scale = max_position_embeddings / original_max_position_embeddings
    if scale <= 1.0:
        scaling_factor = 1.0
    else:
        scaling_factor = math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))

    cos = jnp.cos(emb) * scaling_factor
    sin = jnp.sin(emb) * scaling_factor
    return jnp.concatenate([cos, sin], axis=-1)


@jax.named_scope("easydel-rotary-compute-llama3-frequencies")
def compute_llama3_frequencies(
    base,
    rotary_dim,
    low_freq_factor,
    high_freq_factor,
    scaling_factor,
    max_position_embeddings: int,
):
    """
    Computes RoPE frequencies using the Llama3 scaling method.

    Args:
        base (float): The base value for positional encoding.
        rotary_dim (int): The dimension of the rotary embeddings.
        low_freq_factor (float): Factor for adjusting low-frequency components.
        high_freq_factor (float): Factor for adjusting high-frequency components.
        scaling_factor (float): The overall scaling factor applied.
        max_position_embeddings (int): Original maximum sequence length (referred to as
                                       `orig_max_position` in `compute_llama3_inv_frequencies`).
                                       This defines the length of the frequency cache.

    Returns:
        jnp.ndarray: A frequency cache tensor of shape (max_position_embeddings, rotary_dim).
    """
    inv = compute_llama3_inv_frequencies(
        base,
        rotary_dim,
        low_freq_factor,
        high_freq_factor,
        max_position_embeddings,
        scaling_factor,
    )
    freqs = jnp.einsum(
        "i,j -> ij",
        jnp.arange(max_position_embeddings, dtype=jnp.float32),
        inv,
    )
    freqs = jnp.concatenate([jnp.cos(freqs), jnp.sin(freqs)], axis=-1)
    return freqs


@jax.named_scope("easydel-rotary-compute-deepseek-frequencies")
def compute_deepseek_frequencies(
    base,
    rotary_dim,
    scaling_factor,
    extrapolation_factor,
    beta_fast,
    beta_slow,
    max_position_embeddings,
    mscale,
    mscale_all_dim,
    attn_factor,
) -> jnp.ndarray:
    """
    Computes RoPE frequencies using the Deepseek-YaRN scaling method.

    Similar to YaRN but potentially uses different mscale calculation parameters.

    Args:
        base (float): The base value for positional encoding.
        rotary_dim (int): The dimension of the rotary embeddings.
        scaling_factor (float): The factor by which the context length is scaled.
        extrapolation_factor (float): YaRN parameter controlling extrapolation strength.
        beta_fast (int): YaRN parameter for faster rotating dimensions.
        beta_slow (int): YaRN parameter for slower rotating dimensions.
        max_position_embeddings (int): Original maximum sequence length before scaling.
        mscale (float): Parameter for `yarn_get_mscale` calculation.
        mscale_all_dim (float): Parameter for `yarn_get_mscale` calculation.
        attn_factor (float): Scaling factor applied to attention outputs.

    Returns:
        jnp.ndarray: A frequency cache tensor of shape
                     (max_position_embeddings * scaling_factor, rotary_dim).
    """
    pos_freqs = base ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
    low, high = _yarn_find_correction_range(
        beta_fast,
        beta_slow,
        rotary_dim,
        base,
        max_position_embeddings,
    )
    inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=jnp.float32)) * extrapolation_factor
    inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

    t = jnp.arange(
        max_position_embeddings * scaling_factor,
        dtype=jnp.float32,
    )
    freqs = jnp.einsum("i,j -> ij", t, inv_freq)
    mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(scaling_factor, mscale_all_dim) * attn_factor

    return jnp.concatenate([jnp.cos(freqs) * mscale, jnp.sin(freqs) * mscale], axis=-1)


@jax.named_scope("easydel-rotary-apply-basic-rope")
def apply_basic_rope(
    query: jax.Array,
    key: jax.Array,
    positions: jax.Array,
    frequencies: jax.Array,
    rotary_dim: int,
    is_neox_style: bool,
    offsets: jax.Array = None,
    dtype: jnp.dtype = jnp.float32,
):
    """
    Applies standard or partially applied RoPE to query and key tensors.

    Selects frequencies based on positions (and optional offsets), then applies
    the rotation using `_apply_rotary_emb`. Handles cases where RoPE is
    applied only to a subset of the head dimension (`rotary_dim < query.shape[-1]`).

    Args:
        query (jax.Array): Query tensor. Shape [..., sequence_length, num_heads, head_dim].
        key (jax.Array): Key tensor. Shape [..., sequence_length, num_heads, head_dim].
        positions (jax.Array): Array of positions for lookup in the frequency cache. Shape [sequence_length].
        frequencies (jax.Array): Precomputed frequency cache. Shape [max_length, rotary_dim_freq].
        rotary_dim (int): The dimension up to which RoPE is applied.
        is_neox_style (bool): Whether to use Neox-style rotation.
        offsets (jax.Array, optional): Optional offsets to add to positions. Defaults to None.
        dtype (jnp.dtype, optional): Output dtype. Defaults to jnp.float32.

    Returns:
        tp.Tuple[jax.Array, jax.Array]: The rotated query and key tensors with the specified dtype.
    """
    if offsets is not None:
        positions = positions + offsets
    cos, sin = jnp.split(frequencies[positions], 2, -1)
    if rotary_dim != query.shape[-1]:
        query_rot = _apply_rotary_emb(query[..., :rotary_dim], cos, sin, is_neox_style)
        query = jnp.concatenate((query_rot, query[..., rotary_dim:]), axis=-1)
        key_rot = _apply_rotary_emb(key[..., :rotary_dim], cos, sin, is_neox_style)
        key = jnp.concatenate((key_rot, key[..., rotary_dim:]), axis=-1)
        return query.astype(dtype), key.astype(dtype)
    else:
        query = _apply_rotary_emb(query, cos, sin, is_neox_style)
        key = _apply_rotary_emb(key, cos, sin, is_neox_style)
        return query.astype(dtype), key.astype(dtype)


@jax.named_scope("easydel-rotary-apply-phi3-rope")
def apply_phi3_rope(
    query,
    key,
    positions,
    frequencies,
    offsets: jax.Array = None,
    dtype: jnp.dtype = jnp.float32,
):
    """
    Applies Phi-3 LongRoPE to query and key tensors.

    Uses a specific rotation application style (`_rotate_neox`) assumed by Phi-3.

    Args:
        query (jax.Array): Query tensor. Shape [batch_size, sequence_length, num_heads, head_dim].
        key (jax.Array): Key tensor. Shape [batch_size, sequence_length, num_heads, head_dim].
        positions (jax.Array): Array of positions. Shape [sequence_length].
        frequencies (jax.Array): Precomputed Phi-3 frequency cache.
                                 Shape [1, max_length, rotary_dim].
        offsets (jax.Array, optional): Optional offsets to add to positions. Defaults to None.
        dtype (jnp.dtype, optional): Output dtype. Defaults to jnp.float32.

    Returns:
        tp.Tuple[jax.Array, jax.Array]: The rotated query and key tensors with the specified dtype.
    """
    positions = positions
    if offsets is not None:
        positions = positions + offsets
    emb = frequencies[0, positions]
    cos, sin = jnp.split(emb, 2, axis=-1)
    cos = jnp.expand_dims(cos, 2)
    sin = jnp.expand_dims(sin, 2)

    with jax.default_matmul_precision("float32"):
        query_rot = query * cos + _rotate_neox(query) * sin
        key_rot = key * cos + _rotate_neox(key) * sin

    return query_rot.astype(dtype), key_rot.astype(dtype)


@rope_wraper("default")
class RotaryEmbedding(nn.Module):
    """
    Standard Rotary Positional Embedding (RoPE) module.

    Attributes:
        head_size (int): The dimension size of each attention head.
        rotary_dim (int): The dimension size of the rotary embeddings applied. Can be <= head_size.
        max_position_embeddings (int): The maximum sequence length the model can handle.
        base (int): The base value for calculating frequencies.
        is_neox_style (bool): Flag indicating whether to use Neox-style rotation.
        dtype (jnp.dtype): Data type for computations.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
    ):
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

    @jax.named_scope("easydel-rope-embedding")
    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        offsets: jnp.ndarray | None = None,
        frequencies: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """__call__ pass for the rotary embedding."""
        with jax.ensure_compile_time_eval():
            if frequencies is None:
                frequencies = compute_basic_frequencies(
                    base=self.base,
                    rotary_dim=self.rotary_dim,
                    max_position_embeddings=self.max_position_embeddings,
                )
            if hasattr(frequencies, "value"):
                frequencies = frequencies.value
            return apply_basic_rope(
                query=query,
                key=key,
                positions=positions,
                frequencies=frequencies,
                rotary_dim=self.rotary_dim,
                is_neox_style=self.is_neox_style,
                offsets=offsets,
                dtype=self.dtype,
            )


@rope_wraper("linear")
class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """
    RotaryEmbedding extended with Linear Scaling.

    Linearly scales the position indices before calculating frequencies.

    Attributes:
        scaling_factors (tp.Union[tp.List[float], float]): The factor(s) to scale positions by.
        Inherits other attributes from RotaryEmbedding.
    """

    def __init__(
        self,
        scaling_factors: list[float] | float,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
    ):
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )
        self.scaling_factors = scaling_factors

    @jax.named_scope("easydel-rope-linear-scaling")
    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        offsets: jnp.ndarray | None = None,
        frequencies: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """__call__ pass for the rotary embedding."""
        with jax.ensure_compile_time_eval():
            if frequencies is None:
                frequencies = compute_linear_frequencies(
                    base=self.base,
                    rotary_dim=self.rotary_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factors=self.scaling_factors,
                )
            if hasattr(frequencies, "value"):
                frequencies = frequencies.value
            return apply_basic_rope(
                query=query,
                key=key,
                positions=positions,
                frequencies=frequencies,
                rotary_dim=self.rotary_dim,
                is_neox_style=self.is_neox_style,
                offsets=offsets,
                dtype=self.dtype,
            )


@rope_wraper("dynamic")
class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """
    RotaryEmbedding extended with Dynamic NTK scaling.

    Dynamically adjusts the `base` parameter based on the scaling factor.

    Attributes:
        scaling_factor (float): The scaling factor applied to sequence length and base calculation.
        Inherits other attributes from RotaryEmbedding.
    """

    def __init__(
        self,
        scaling_factor: list[float] | float,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
    ):
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )
        self.scaling_factor = scaling_factor

    @jax.named_scope("easydel-rope-dynamic-ntk-scaling")
    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        offsets: jnp.ndarray | None = None,
        frequencies: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """__call__ pass for the rotary embedding."""
        with jax.ensure_compile_time_eval():
            if frequencies is None:
                frequencies = compute_dynamic_frequencies(
                    base=self.base,
                    rotary_dim=self.rotary_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=self.scaling_factor,
                )
            if hasattr(frequencies, "value"):
                frequencies = frequencies.value
            return apply_basic_rope(
                query=query,
                key=key,
                positions=positions,
                frequencies=frequencies,
                rotary_dim=self.rotary_dim,
                is_neox_style=self.is_neox_style,
                offsets=offsets,
                dtype=self.dtype,
            )


@rope_wraper("yarn")
class YaRNScalingRotaryEmbedding(RotaryEmbedding):
    """
    RotaryEmbedding extended with the YaRN (Yet another RoPE extensioN method) scaling.

    Combines interpolation and extrapolation with frequency correction and magnitude scaling.

    Attributes:
        scaling_factor (tp.Union[float, int]): The primary scaling factor for context length.
        extrapolation_factor (float): Controls the strength of extrapolation correction.
        attn_factor (float): Scales the output attention values.
        beta_fast (int): YaRN parameter for high-frequency dimensions correction range.
        beta_slow (int): YaRN parameter for low-frequency dimensions correction range.
        Inherits other attributes from RotaryEmbedding. Note: `max_position_embeddings`
        in the parent init likely refers to the *original* max length for YaRN calculations.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
        scaling_factor: float | int = 1.0,
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ):
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )

        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

    @jax.named_scope("easydel-rope-yarn-scaling")
    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        offsets: jnp.ndarray | None = None,
        frequencies: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """__call__ pass for the rotary embedding."""
        with jax.ensure_compile_time_eval():
            if frequencies is None:
                frequencies = compute_yarn_frequencies(
                    base=self.base,
                    rotary_dim=self.rotary_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=self.scaling_factor,
                    beta_fast=self.beta_fast,
                    beta_slow=self.beta_slow,
                    extrapolation_factor=self.extrapolation_factor,
                    attn_factor=self.attn_factor,
                )
            if hasattr(frequencies, "value"):
                frequencies = frequencies.value
            return apply_basic_rope(
                query=query,
                key=key,
                positions=positions,
                frequencies=frequencies,
                rotary_dim=self.rotary_dim,
                is_neox_style=self.is_neox_style,
                offsets=offsets,
                dtype=self.dtype,
            )


@rope_wraper("longrope")
class Phi3LongRoPEScaledRotaryEmbedding(nn.Module):
    """
    RotaryEmbedding using the Phi-3 LongRoPE scaling method.

    Applies different frequency scaling factors (`short_factor`, `long_factor`)
    depending on the target sequence length relative to the original maximum.
    Requires `rotary_dim` to be equal to `head_size`.

    Attributes:
        head_size (int): Dimension of each attention head. Must equal rotary_dim.
        rotary_dim (int): Dimension subjected to rotary embedding. Must equal head_size.
        max_position_embeddings (int): The target maximum sequence length after scaling.
        original_max_position_embeddings (int): Original maximum sequence length before scaling.
        base (int): Base for frequency calculation.
        is_neox_style (bool): Flag indicating whether Neox-style rotation is assumed (used by `apply_phi3_rope`).
        dtype (jnp.dtype): Data type for computations.
        short_factor (tp.List[float]): Scaling factors applied when target length <= original max length.
        long_factor (tp.List[float]): Scaling factors applied when target length > original max length.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        original_max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
        short_factor: list[float],
        long_factor: list[float],
    ):
        super().__init__()

        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.short_factor = short_factor
        self.long_factor = long_factor

    @jax.named_scope("easydel-rope-phi3-long")
    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        offsets: jnp.ndarray | None = None,
        frequencies: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """__call__ pass for the rotary embedding."""
        with jax.ensure_compile_time_eval():
            if frequencies is None:
                frequencies = compute_phi3_frequencies(
                    base=self.base,
                    head_size=self.head_size,
                    rotary_dim=self.rotary_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    original_max_position_embeddings=self.original_max_position_embeddings,
                    short_factor=self.short_factor,
                    long_factor=self.long_factor,
                )
            if hasattr(frequencies, "value"):
                frequencies = frequencies.value
            return apply_phi3_rope(
                query=query,
                key=key,
                positions=positions,
                frequencies=frequencies,
                offsets=offsets,
                dtype=self.dtype,
            )


@rope_wraper("llama3")
class Llama3RotaryEmbedding(RotaryEmbedding):
    """
    RotaryEmbedding implementing the Llama-3 scaling method.

    Adjusts frequencies based on wavelength thresholds (`low_freq_factor`, `high_freq_factor`)
    and applies an overall scaling factor.

    Attributes:
        scaling_factor (float): Overall scaling factor.
        low_freq_factor (float): Factor related to low frequency wavelength threshold.
        high_freq_factor (float): Factor related to high frequency wavelength threshold.
        orig_max_position (int): Original maximum sequence length before scaling.
        Inherits other attributes from RotaryEmbedding.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        orig_max_position: int,
    ):
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )

        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position

    @jax.named_scope("easydel-rope-llama3")
    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        offsets: jnp.ndarray | None = None,
        frequencies: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """__call__ pass for the rotary embedding."""
        with jax.ensure_compile_time_eval():
            if frequencies is None:
                frequencies = compute_llama3_frequencies(
                    base=self.base,
                    rotary_dim=self.rotary_dim,
                    low_freq_factor=self.low_freq_factor,
                    high_freq_factor=self.high_freq_factor,
                    scaling_factor=self.scaling_factor,
                    max_position_embeddings=self.orig_max_position,
                )
            if hasattr(frequencies, "value"):
                frequencies = frequencies.value

            return apply_basic_rope(
                query=query,
                key=key,
                positions=positions,
                frequencies=frequencies,
                rotary_dim=self.rotary_dim,
                is_neox_style=self.is_neox_style,
                offsets=offsets,
                dtype=self.dtype,
            )


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
    return 0.1 * mscale * jnp.log(scale) + 1.0


@rope_wraper("deepseek_yarn")
class DeepseekScalingRotaryEmbedding(nn.Module):
    """
    RotaryEmbedding implementing a YaRN-like scaling method, potentially from Deepseek models.

    Uses YaRN parameters (`beta_fast`, `beta_slow`, `extrapolation_factor`) and includes
    additional m-scale parameters (`mscale`, `mscale_all_dim`). This version has a custom
    `__call__` method differing slightly from `apply_basic_rope`.

    Attributes:
        head_size (int): Dimension of each attention head.
        rotary_dim (int): Dimension subjected to rotary embedding.
        max_position_embeddings (int): Original maximum sequence length before scaling.
        base (int): Base for frequency calculation.
        is_neox_style (bool): Use Neox rotation if True, GPT-J otherwise.
        dtype (jnp.dtype): Data type for embeddings.
        scaling_factor (float): Primary scaling factor.
        extrapolation_factor (float): YaRN extrapolation factor.
        attn_factor (float): Attention scaling factor.
        beta_fast (int): YaRN parameter.
        beta_slow (int): YaRN parameter.
        mscale (float): Parameter for m-scale calculation.
        mscale_all_dim (float): Parameter for m-scale calculation.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
        scaling_factor: float,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ):
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim

    @jax.named_scope("easydel-rope-deepseek")
    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        offsets: jnp.ndarray | None = None,
        frequencies: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if frequencies is None:
            frequencies = compute_deepseek_frequencies(
                self.base,
                self.rotary_dim,
                self.scaling_factor,
                self.extrapolation_factor,
                self.beta_fast,
                self.beta_slow,
                self.max_position_embeddings,
                self.mscale,
                self.mscale_all_dim,
                self.attn_factor,
            )
        cos, sin = jnp.split(frequencies[positions], 2, -1)
        if offsets is not None:
            positions += offsets
        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]

        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

        target_sc_shape = (query.shape[0], -1, 1, self.rotary_dim)
        if self.is_neox_style:
            cos = cos.repeat(2, axis=1).reshape(target_sc_shape)
            sin = sin.repeat(2, axis=1).reshape(target_sc_shape)
        else:
            cos = cos.repeat_interleave(2, axis=1).reshape(target_sc_shape)
            sin = sin.repeat_interleave(2, axis=1).reshape(target_sc_shape)
        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = jnp.concatenate((query_rot, query_pass), axis=-1)
            key = jnp.concatenate((key_rot, key_pass), axis=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: dict[str, tp.Any] | None = None,
    dtype: jnp.dtype | None = None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    """
    Factory function to create and return a RotaryEmbedding instance based on configuration.

    Selects the appropriate RoPE class (standard, linear, dynamic, YaRN, Llama3, Phi3, Deepseek)
    based on the `rope_scaling` dictionary.

    Args:
        head_size (int): Dimension of each attention head.
        rotary_dim (int): Base dimension for rotary embedding (before partial factor).
        max_position (int): Maximum sequence length the model should support (target length).
        base (int): Base value for frequency calculation.
        is_neox_style (bool, optional): Use Neox rotation style. Defaults to True.
        rope_scaling (tp.Optional[tp.Dict[str, tp.Any]], optional): Dictionary specifying the
            type and parameters of RoPE scaling. If None or 'rope_type' is 'default',
            uses standard RoPE. Keys like 'rope_type', 'factor',
            'original_max_position_embeddings', etc., are used. Defaults to None.
        dtype (tp.Optional[jnp.dtype], optional): Data type for embeddings. Defaults to jnp.float32.
        partial_rotary_factor (float, optional): Factor to reduce the rotary dimension
            (e.g., 0.5 applies RoPE to half the dimensions). Defaults to 1.0.

    Returns:
        RotaryEmbedding: An instance of the configured RotaryEmbedding subclass.

    Raises:
        ValueError: If `rope_scaling` specifies an unknown `rope_type`.
    """
    if dtype is None:
        dtype = jnp.float32  # Default JAX dtype

    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)

    if rope_scaling is None:
        rotary_emb = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )
    else:
        scaling_type = rope_scaling["rope_type"]
        if scaling_type == "llama3":
            scaling_factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling["low_freq_factor"]
            high_freq_factor = rope_scaling["high_freq_factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            rotary_emb = Llama3RotaryEmbedding(
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position,
                base=base,
                is_neox_style=is_neox_style,
                dtype=dtype,
                scaling_factor=scaling_factor,
                low_freq_factor=low_freq_factor,
                high_freq_factor=high_freq_factor,
                orig_max_position=original_max_position,
            )
        elif scaling_type == "default":
            rotary_emb = RotaryEmbedding(
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position,
                base=base,
                is_neox_style=is_neox_style,
                dtype=dtype,
            )
        elif scaling_type == "linear":
            scaling_factor = rope_scaling["factor"]
            rotary_emb = LinearScalingRotaryEmbedding(
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position,
                base=base,
                is_neox_style=is_neox_style,
                scaling_factors=scaling_factor,
                dtype=dtype,
            )
        elif scaling_type == "dynamic":
            scaling_factor = rope_scaling["factor"]
            rotary_emb = DynamicNTKScalingRotaryEmbedding(
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position,
                base=base,
                is_neox_style=is_neox_style,
                scaling_factor=scaling_factor,
                dtype=dtype,
            )
        elif scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling.get("original_max_position_embeddings", max_position)
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow")
            }
            rotary_emb = YaRNScalingRotaryEmbedding(
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position_embeddings=original_max_position,
                base=base,
                is_neox_style=is_neox_style,
                scaling_factor=scaling_factor,
                dtype=dtype,
                **extra_kwargs,
            )
        elif scaling_type == "deepseek_yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim")
            }
            rotary_emb = DeepseekScalingRotaryEmbedding(
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position_embeddings=original_max_position,
                base=base,
                is_neox_style=is_neox_style,
                scaling_factor=scaling_factor,
                dtype=dtype,
                **extra_kwargs,
            )
        elif scaling_type == "longrope":
            short_factor = rope_scaling["short_factor"]
            long_factor = rope_scaling["long_factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]

            rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position,
                original_max_position_embeddings=original_max_position,
                base=base,
                is_neox_style=is_neox_style,
                dtype=dtype,
                short_factor=short_factor,
                long_factor=long_factor,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    return rotary_emb


@ejit(
    static_argnames=[
        "head_size",
        "rotary_dim",
        "max_position",
        "base",
        "rope_scaling",
        "partial_rotary_factor",
    ],
)
def get_frequencies(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    rope_scaling: dict[str, tp.Any] | None = None,
    partial_rotary_factor: float = 1.0,
) -> jax.Array:
    """
    Computes and returns the RoPE frequency cache based on configuration.

    Selects the appropriate frequency computation function (basic, linear, dynamic,
    YaRN, Llama3, Phi3, Deepseek) based on the `rope_scaling` dictionary.
    This function is JIT-compiled for performance, with relevant parameters marked static.

    Args:
        head_size (int): Dimension of each attention head (needed for some scaling types like Phi3).
        rotary_dim (int): Base dimension for rotary embedding (before partial factor).
        max_position (int): Maximum sequence length for which to compute frequencies.
                            This might be the original or target length depending on scaling type.
        base (int): Base value for frequency calculation.
        rope_scaling (tp.Optional[tp.Dict[str, tp.Any]], optional): Dictionary specifying the
            type and parameters of RoPE scaling. Determines which frequency function to call.
            Defaults to None (uses `compute_basic_frequencies`).
        partial_rotary_factor (float, optional): Factor to reduce the rotary dimension.
            Defaults to 1.0.

    Returns:
        jax.Array: The computed frequency cache tensor. Shape depends on the scaling method,
                   typically [computed_length, rotary_dim_effective].

    Raises:
        ValueError: If `rope_scaling` specifies an unknown `rope_type`.
    """
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)

    if rope_scaling is None:
        frequencies = compute_basic_frequencies(
            base=base,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position,
        )
    else:
        scaling_type = rope_scaling["rope_type"]

        if scaling_type == "llama3":
            scaling_factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling["low_freq_factor"]
            high_freq_factor = rope_scaling["high_freq_factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            frequencies = compute_llama3_frequencies(
                base=base,
                rotary_dim=rotary_dim,
                low_freq_factor=low_freq_factor,
                high_freq_factor=high_freq_factor,
                scaling_factor=scaling_factor,
                max_position_embeddings=original_max_position,
            )

        elif scaling_type == "default":
            frequencies = compute_basic_frequencies(
                base=base,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position,
            )
        elif scaling_type == "linear":
            scaling_factors = rope_scaling["factor"]
            frequencies = compute_linear_frequencies(
                base=base,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position,
                scaling_factors=scaling_factors,
            )
        elif scaling_type == "dynamic":
            scaling_factor = rope_scaling["factor"]
            frequencies = compute_dynamic_frequencies(
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position,
                base=base,
                scaling_factor=scaling_factor,
            )
        elif scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling.get("original_max_position_embeddings", max_position)  # for gpt_oss
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow")
            }

            frequencies = compute_yarn_frequencies(
                base=base,
                rotary_dim=rotary_dim,
                beta_fast=extra_kwargs.get("beta_fast", 32),
                beta_slow=extra_kwargs.get("beta_slow", 1),
                max_position_embeddings=original_max_position,
                scaling_factor=scaling_factor,
                extrapolation_factor=extra_kwargs.get("extrapolation_factor", 1.0),
                attn_factor=extra_kwargs.get("attn_factor", extra_kwargs.get("attention_factor", 1)),
            )
        elif scaling_type == "deepseek_yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim")
            }
            frequencies = compute_deepseek_frequencies(
                base,
                rotary_dim,
                scaling_factor,
                extra_kwargs.get("extrapolation_factor", 1.0),
                extra_kwargs.get("beta_fast", 32),
                extra_kwargs.get("beta_slow", 1),
                original_max_position,
                extra_kwargs["mscale"],
                extra_kwargs["mscale_all_dim"],
                extra_kwargs.get("attn_factor", extra_kwargs.get("attention_factor", 1)),
            )
        elif scaling_type == "longrope":
            short_factor = rope_scaling["short_factor"]
            long_factor = rope_scaling["long_factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            extra_kwargs = {k: v for k, v in rope_scaling.items() if k in ("short_mscale", "long_mscale")}

            frequencies = compute_phi3_frequencies(
                base=base,
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position,
                original_max_position_embeddings=original_max_position,
                short_factor=short_factor,
                long_factor=long_factor,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    return frequencies


@ejit(
    static_argnames=[
        "head_size",
        "rotary_dim",
        "max_position",
        "base",
        "rope_scaling",
        "partial_rotary_factor",
    ],
)
def get_inv_frequencies(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    rope_scaling: dict[str, tp.Any] | None = None,
    partial_rotary_factor: float = 1.0,
) -> jax.Array:
    """
    Computes and returns just the inverse frequencies for RoPE based on configuration.

    Similar to `get_frequencies` but returns only the inverse frequencies without
    computing the full frequency cache (no cos/sin transformation).

    Args:
        head_size (int): Dimension of each attention head (needed for some scaling types like Phi3).
        rotary_dim (int): Base dimension for rotary embedding (before partial factor).
        max_position (int): Maximum sequence length the model should support.
        base (int): Base value for frequency calculation.
        rope_scaling (tp.Optional[tp.Dict[str, tp.Any]], optional): Dictionary specifying the
            type and parameters of RoPE scaling. Determines which frequency function to call.
            Defaults to None (uses basic inverse frequencies).
        partial_rotary_factor (float, optional): Factor to reduce the rotary dimension.
            Defaults to 1.0.

    Returns:
        jax.Array: The computed inverse frequencies. Shape is typically (rotary_dim // 2,).

    Raises:
        ValueError: If `rope_scaling` specifies an unknown `rope_type`.
    """
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)

    if rope_scaling is None:
        inv_frequencies = compute_basic_inv_frequencies(base=base, rotary_dim=rotary_dim)
    else:
        scaling_type = rope_scaling["rope_type"]

        if scaling_type == "llama3":
            scaling_factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling["low_freq_factor"]
            high_freq_factor = rope_scaling["high_freq_factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            inv_frequencies = compute_llama3_inv_frequencies(
                base=base,
                rotary_dim=rotary_dim,
                low_freq_factor=low_freq_factor,
                high_freq_factor=high_freq_factor,
                orig_max_position=original_max_position,
                scaling_factor=scaling_factor,
            )

        elif scaling_type == "default":
            inv_frequencies = compute_basic_inv_frequencies(base=base, rotary_dim=rotary_dim)
        elif scaling_type == "linear":
            inv_frequencies = compute_basic_inv_frequencies(base=base, rotary_dim=rotary_dim)
        elif scaling_type == "dynamic":
            scaling_factor = rope_scaling["factor"]
            adjusted_base = base * ((scaling_factor * max_position / max_position) - (scaling_factor - 1)) ** (
                rotary_dim / (rotary_dim - 2)
            )
            inv_frequencies = compute_basic_inv_frequencies(base=adjusted_base, rotary_dim=rotary_dim)
        elif scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            extra_kwargs = {
                k: v for k, v in rope_scaling.items() if k in ("extrapolation_factor", "beta_fast", "beta_slow")
            }
            extrapolation_factor = extra_kwargs.get("extrapolation_factor", 1.0)
            beta_fast = extra_kwargs.get("beta_fast", 32)
            beta_slow = extra_kwargs.get("beta_slow", 1)
            inv_frequencies = compute_yarn_inv_frequencies(
                base=base,
                rotary_dim=rotary_dim,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
                max_position_embeddings=original_max_position,
                scaling_factor=scaling_factor,
                extrapolation_factor=extrapolation_factor,
            )
        elif scaling_type == "deepseek_yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            extra_kwargs = {
                k: v for k, v in rope_scaling.items() if k in ("extrapolation_factor", "beta_fast", "beta_slow")
            }
            extrapolation_factor = extra_kwargs.get("extrapolation_factor", 1.0)
            beta_fast = extra_kwargs.get("beta_fast", 32)
            beta_slow = extra_kwargs.get("beta_slow", 1)
            pos_freqs = base ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim)
            inv_freq_extrapolation = 1.0 / pos_freqs
            inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
            low, high = _yarn_find_correction_range(beta_fast, beta_slow, rotary_dim, base, original_max_position)
            inv_freq_mask = (
                1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=jnp.float32)
            ) * extrapolation_factor
            inv_frequencies = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        elif scaling_type == "longrope":
            short_factor = rope_scaling["short_factor"]
            long_factor = rope_scaling["long_factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            if max_position > original_max_position:
                ext_factors = jnp.array(long_factor, dtype=jnp.float32)
            else:
                ext_factors = jnp.array(short_factor, dtype=jnp.float32)

            inv_freq_shape = jnp.arange(0, head_size, 2, dtype=jnp.int32).astype(jnp.float32) / head_size
            inv_frequencies = 1.0 / (ext_factors * (base**inv_freq_shape))
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    return inv_frequencies


# Example usage
if __name__ == "__main__":
    head_size = 64
    rotary_dim = 64
    max_position = 2048
    base = 10000
    is_neox_style = True
    dtype = jnp.float32

    rope_scaling = {
        "rope_type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 1024,
        "extrapolation_factor": 1.0,
        "attn_factor": 1.0,
        "beta_fast": 32,
        "beta_slow": 1,
    }

    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style, rope_scaling, dtype)
    freq = get_frequencies(head_size, rotary_dim, max_position, base, rope_scaling)


@chex.dataclass
class RopeConfig:
    """
    Configuration class for RoPE (Rotary Position Embedding) parameters.

    Stores the configuration related to RoPE type and its scaling parameters,
    making it easy to manage and pass around RoPE settings.

    Attributes:
        rope_type (str): The type of RoPE scaling to use (e.g., "default", "linear", "yarn", "llama3").
                         Defaults to "default".
        factor (tp.Optional[float]): General scaling factor used by some types (linear, dynamic, yarn, llama3).
        low_freq_factor (tp.Optional[float]): Specific factor for Llama3 scaling.
        high_freq_factor (tp.Optional[float]): Specific factor for Llama3 scaling.
        original_max_position_embeddings (tp.Optional[int]): Original context window size,
                                                            required by some scaling methods
                                                            (yarn, llama3, phi3, deepseek).
        long_factor (tp.Optional[float]): Specific factor for Phi3 LongRoPE scaling (used for lengths > original).
        short_factor (tp.Optional[float]): Specific factor for Phi3 LongRoPE scaling (used for lengths <= original).
        long_mscale (tp.Optional[float]): Potentially used by variants like Phi3. (Not used in current `get_rope`).
        short_mscale (tp.Optional[float]): Potentially used by variants like Phi3. (Not used in current `get_rope`).
        # Add other potential scaling parameters here as needed (e.g., from YaRN, Deepseek)
        extrapolation_factor (tp.Optional[float]): YaRN/Deepseek parameter.
        attn_factor (tp.Optional[float]): YaRN/Deepseek parameter.
        beta_fast (tp.Optional[int]): YaRN/Deepseek parameter.
        beta_slow (tp.Optional[int]): YaRN/Deepseek parameter.
        mscale (tp.Optional[float]): Deepseek parameter.
        mscale_all_dim (tp.Optional[float]): Deepseek parameter.
    """

    rope_type: str = "default"
    factor: float | None = None
    low_freq_factor: float | None = None
    high_freq_factor: float | None = None
    original_max_position_embeddings: int | None = None
    long_factor: float | None = None
    short_factor: float | None = None
    long_mscale: float | None = None
    short_mscale: float | None = None

    @classmethod
    def from_dict(cls, config_dict: dict[str, tp.Any]) -> RopeConfig:
        """
        Create a RopeConfig instance from a dictionary.

        Handles potential alias 'type' for 'rope_type'.

        Args:
            config_dict (tp.Dict[str, tp.Any]): Dictionary containing RoPE configuration.

        Returns:
            RopeConfig: An instance populated from the dictionary.
        """
        return cls(
            rope_type=config_dict.get("rope_type") or config_dict.get("type", "default"),
            factor=config_dict.get("factor"),
            low_freq_factor=config_dict.get("low_freq_factor"),
            high_freq_factor=config_dict.get("high_freq_factor"),
            original_max_position_embeddings=config_dict.get("original_max_position_embeddings"),
            long_factor=config_dict.get("long_factor"),
            short_factor=config_dict.get("short_factor"),
            long_mscale=config_dict.get("long_mscale"),
            short_mscale=config_dict.get("short_mscale"),
        )

    def to_dict(self) -> dict[str, tp.Any]:
        """
        Convert the RopeConfig instance to a dictionary.

        Filters out attributes with None values. The dictionary is made hashable
        using a custom class for potential use with JIT compilation contexts
        (though making the dict itself static in `get_frequencies` is preferred).

        Returns:
            tp.Dict[str, tp.Any]: A hashable dictionary containing non-None configuration values.
        """
        from easydel.utils.compiling_utils import hash_fn

        class rope_scaling(dict):
            """A dictionary subclass that is hashable."""

            __hash__ = hash_fn

        scale = rope_scaling({k: v for k, v in self.__dict__.items() if v is not None})
        return scale
