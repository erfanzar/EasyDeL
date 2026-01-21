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

from ._utils import (
    _apply_rotary_emb,
    _rotate_neox,
    _yarn_find_correction_range,
    _yarn_get_mscale,
    _yarn_linear_ramp_mask,
    yarn_get_mscale,
)


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
    freqs = jnp.einsum("i,j -> ij", jnp.arange(max_position_embeddings, dtype=jnp.float32), inv)
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
    # DeepSeek mscale calculation
    attention_factor = (
        yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(scaling_factor, mscale_all_dim) * attn_factor
    )

    # Standard RoPE format: concatenate cos and sin
    return jnp.concatenate([jnp.cos(freqs) * attention_factor, jnp.sin(freqs) * attention_factor], axis=-1)


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
        return query, key
    else:
        query = _apply_rotary_emb(query, cos, sin, is_neox_style)
        key = _apply_rotary_emb(key, cos, sin, is_neox_style)
        return query, key
