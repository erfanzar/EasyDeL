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

"""Computation functions for Rotary Position Embeddings (RoPE).

This module provides functions for computing inverse frequencies and frequency
caches for various RoPE scaling methods, as well as functions for applying
rotary embeddings to query and key tensors.

Supported RoPE scaling methods:
    - Basic/Default: Standard RoPE with no scaling.
    - Linear: Linearly scaled positions for extended context.
    - Dynamic NTK: Dynamically adjusted base for context extension.
    - YaRN: Yet another RoPE extensioN method with interpolation/extrapolation.
    - Llama3: Llama-3 style scaling with wavelength-based adjustments.
    - Phi3 LongRoPE: Phi-3 style scaling with short/long factors.
    - Deepseek: Deepseek-YaRN variant with additional mscale parameters.

Example:
    >>> import jax.numpy as jnp
    >>> from easydel.layers.rotary._compute_fns import (
    ...     compute_basic_frequencies,
    ...     apply_basic_rope,
    ... )
    >>> # Compute frequency cache
    >>> freqs = compute_basic_frequencies(base=10000, rotary_dim=64, max_position_embeddings=2048)
    >>> # Apply RoPE to query and key
    >>> query = jnp.ones((1, 128, 8, 64))  # [batch, seq, heads, head_dim]
    >>> key = jnp.ones((1, 128, 8, 64))
    >>> positions = jnp.arange(128)
    >>> q_rot, k_rot = apply_basic_rope(query, key, positions, freqs, rotary_dim=64, is_neox_style=True)
"""

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
    """Compute the unscaled RoPE inverse-frequency vector ``θ_i = 1 / base^(2i/d)``.

    This is the geometric progression at the heart of the original RoPE
    formulation (Su et al. 2021): given a head dimension ``rotary_dim`` and a
    base period ``base`` (typically 10 000 for vanilla RoPE, 500 000 for
    Llama-3, 1 000 000 for some long-context models), the frequencies span
    a geometric series with the high-frequency rotation handled by index 0
    and the slowest rotation by index ``rotary_dim/2 - 1``.

    Used directly by the unscaled RoPE path and as the *unscaled baseline*
    by Llama-3, dynamic-NTK and some YaRN variants.

    Args:
        base: Frequency-progression base (the ``θ`` in the paper). Larger
            values flatten the spectrum and extend the effective context
            length without re-training; defaults vary by model family.
        rotary_dim: Total rotary feature dimension. Must be even since each
            pair of features rotates together; the returned vector has
            ``rotary_dim // 2`` entries (one per rotation plane).

    Returns:
        Float32 array of shape ``(rotary_dim // 2,)`` containing ``1/θ_i``,
        ready to be outer-product'd with positions to form the rotation
        angles.
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
    truncate: bool = True,
) -> jnp.ndarray:
    """Compute YaRN-adjusted inverse frequencies (interpolated + extrapolated mix).

    YaRN (Peng et al., 2023) decomposes RoPE rotations into "fast" and
    "slow" frequency bands defined by how many full rotations a given
    inverse-frequency completes within the original training window. The
    method then blends two regimes:

    * **Interpolation** — divide the raw inverse frequencies by
      ``scaling_factor`` (Position-Interpolation, Chen et al. 2023) so all
      rotations slow down proportionally.
    * **Extrapolation** — keep the raw inverse frequencies untouched
      (matches the trained behaviour on the high-frequency dimensions).

    The blend is gated by a smooth linear ramp built from
    ``_yarn_find_correction_range(beta_fast, beta_slow, ...)`` and weighted
    by ``extrapolation_factor`` so the user can tune how aggressively the
    high-frequency dims are extrapolated. The result is the inverse
    frequencies; positions and the magnitude scale ``mscale`` are applied
    by :func:`compute_yarn_frequencies`.

    Args:
        base: Base ``θ`` for the unscaled spectrum (typically 10 000 or
            larger).
        rotary_dim: Total rotary feature dimension; the returned vector has
            ``rotary_dim // 2`` entries.
        beta_fast: Number of full rotations across the original context
            beyond which the dimension is treated as "fast" — those dims
            extrapolate (no scaling). Typical YaRN value is 32.
        beta_slow: Number of full rotations within which the dimension is
            "slow" — those dims interpolate (full scaling). Typical 1.
        max_position_embeddings: Original (pre-extension) training context
            length; sets the wavelength reference for the ramp.
        scaling_factor: Target context-length multiplier; divides the
            interpolation branch.
        extrapolation_factor: Scalar in ``[0, 1]`` weighting how much the
            extrapolation regime contributes; ``1.0`` is full YaRN.
        truncate: Whether to floor/ceil the correction band to integer plane
            indices before building the ramp (transformers' ``truncate`` rope
            parameter; default ``True`` preserves classic YaRN).

    Returns:
        Float32 array of shape ``(rotary_dim // 2,)`` containing the blended
        ``1/θ_i`` ready for use in the cos/sin cache build.
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
        truncate=truncate,
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
    """Compute Llama-3 piecewise frequency scaling (the official 8B/70B method).

    Llama-3's RoPE extension is a wavelength-piecewise scheme: each rotation
    plane has a wavelength ``λ_i = 2π / θ_i``, and its inverse frequency is
    rescaled by one of three rules depending on where ``λ_i`` falls:

    * ``λ_i < orig_max_position / high_freq_factor`` (very short
      wavelength, "high-frequency band") — left untouched. These planes
      complete many rotations within the original context and don't need
      adjustment.
    * ``λ_i > orig_max_position / low_freq_factor`` (very long wavelength,
      "low-frequency band") — divided by ``scaling_factor``, full
      Position-Interpolation.
    * In between — a linear interpolation between the two regimes,
      controlled by a smoothing factor that goes from 0 at the
      high-frequency boundary to 1 at the low-frequency boundary.

    Args:
        base: RoPE base period for the unscaled spectrum.
        rotary_dim: Total rotary feature dimension.
        low_freq_factor: Wavelength-cutoff at which the low-frequency
            boundary sits (``λ = orig_max_position / low_freq_factor``).
            Llama-3 uses ``1``.
        high_freq_factor: Same idea for the high-frequency boundary; Llama-3
            uses ``4`` (wavelengths shorter than ``orig/4`` are unscaled).
        orig_max_position: Original training context (Llama-3 uses 8 192).
        scaling_factor: Target context-length multiplier (Llama-3 uses
            ``8`` to extend 8K to 64K).

    Returns:
        Float32 array of shape ``(rotary_dim // 2,)`` of the wavelength-
        piecewise-rescaled inverse frequencies.
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
    """Compute the unscaled RoPE cos/sin cache for all positions.

    Builds the frequency table by outer-producing positions ``[0, max_position)``
    with :func:`compute_basic_inv_frequencies` and concatenating ``[cos, sin]``
    along the last axis — the layout consumed by :func:`apply_basic_rope`.

    Args:
        base: Geometric-progression base ``θ`` (typically 10 000).
        rotary_dim: Rotary feature dimension (must be even).
        max_position_embeddings: Maximum sequence length to pre-compute.

    Returns:
        Float32 array of shape ``(max_position_embeddings, rotary_dim)``
        whose first half is ``cos`` and second half is ``sin``.
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
    """Compute linearly-scaled RoPE frequencies (Position Interpolation).

    Applies Chen et al.'s 2023 Position-Interpolation trick: each position is
    divided by ``scaling_factor`` before the outer product with the unscaled
    inverse frequencies, slowing every rotation plane uniformly so a model
    trained with ``max_position_embeddings`` can be evaluated on
    ``scaling_factor * max_position_embeddings`` tokens.

    When ``scaling_factors`` is a list, the frequency cache is built once per
    factor and the results are concatenated along the position axis — used by
    consumers that store multiple scaling regimes back-to-back (e.g. for
    routing different parts of the sequence to different factors).

    Args:
        base: Geometric-progression base ``θ``.
        rotary_dim: Rotary feature dimension (must be even).
        max_position_embeddings: Pre-scaling context length.
        scaling_factors: A single scaling factor or a list of factors; each
            factor contributes ``max_position_embeddings * factor`` positions
            to the concatenated cache.

    Returns:
        Float32 array of shape ``(sum_i max_position_embeddings * factor_i,
        rotary_dim)`` with the concatenated ``[cos | sin]`` layout.

    Raises:
        ValueError: If the per-factor offsets list and ``scaling_factors``
            list disagree in length (an internal-invariant safeguard).
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

    if len(scaling_factors) != len(offsets):
        raise ValueError(f"scaling_factors length ({len(scaling_factors)}) must match offsets length ({len(offsets)})")
    return jnp.concatenate(cache_list, axis=0)


@jax.named_scope("easydel-rotary-compute-dynamic-frequencies")
def compute_dynamic_frequencies(
    base: int,
    rotary_dim: int,
    max_position_embeddings: int,
    scaling_factor: float,
):
    """Compute Dynamic-NTK-scaled RoPE frequencies.

    Implements the NTK-aware scaling from the original blog post: instead of
    shrinking the *positions* (linear/PI) the *base* itself is increased so
    that the high-frequency dimensions are perturbed less than the
    low-frequency ones. For a sequence length ``L`` the adjusted base is
    ``base * ((scaling_factor * L / max_position_embeddings) - (scaling_factor - 1))
    ** (rotary_dim / (rotary_dim - 2))`` and it is only applied once
    ``L > max_position_embeddings``.

    transformers recomputes the base on every forward from the *current*
    sequence length (``max(position_ids) + 1``), which a static position-indexed
    cache cannot reproduce exactly. This cache therefore uses, for row ``p``,
    the base HF would use for a sequence of length ``p + 1``:

    * rows ``p < max_position_embeddings`` use the unscaled base, so every
      sequence that fits in the original window matches HF exactly;
    * rows beyond it use ``base(p + 1)``, which is what HF applies to each new
      token when decoding incrementally past the original window. A single
      forward over a longer sequence differs from HF, which rotates *all*
      positions with ``base(L)``.

    Args:
        base: Pre-adjustment base ``θ``.
        rotary_dim: Rotary feature dimension (must be even).
        max_position_embeddings: Pre-scaling context length.
        scaling_factor: Target context-length multiplier.

    Returns:
        Float32 array of shape ``(max_position_embeddings * scaling_factor,
        rotary_dim)`` containing the ``[cos | sin]`` cache.
    """
    max_length = int(max_position_embeddings * scaling_factor)
    times = jnp.arange(max_length, dtype=jnp.float32)
    seq_len = jnp.maximum(times + 1.0, max_position_embeddings)
    ntk_base = base * ((scaling_factor * seq_len / max_position_embeddings) - (scaling_factor - 1)) ** (
        rotary_dim / (rotary_dim - 2)
    )
    # Keep the in-window rows bit-identical to the unscaled table.
    bases = jnp.where(seq_len > max_position_embeddings, ntk_base, jnp.float32(base))
    exponents = jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim
    inv_frequencies = 1.0 / (bases[:, None] ** exponents[None, :])
    frequencies = times[:, None] * inv_frequencies
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
    truncate: bool = True,
    attention_factor: float | None = None,
    max_positions: int | None = None,
) -> jnp.ndarray:
    """Compute YaRN-scaled RoPE frequencies with the attention magnitude rescaling.

    Wraps :func:`compute_yarn_inv_frequencies` (interpolation/extrapolation
    blend) with the YaRN ``mscale`` factor — a log-derived scalar that
    rescales the cos/sin magnitudes to keep attention-score variance roughly
    constant as the context grows. The result is the position-outer-producted
    cache.

    Args:
        base: Base ``θ`` for the unscaled spectrum.
        rotary_dim: Rotary feature dimension (must be even).
        beta_fast: YaRN fast-band boundary (typical 32).
        beta_slow: YaRN slow-band boundary (typical 1).
        max_position_embeddings: Original (pre-extension) training context.
        scaling_factor: Target context-length multiplier.
        extrapolation_factor: Scalar in ``[0, 1]`` weighting the
            extrapolation branch.
        attn_factor: User-tunable multiplier composed with the YaRN mscale.
        truncate: Whether to floor/ceil the correction band to integer plane
            indices before building the ramp (transformers' ``truncate`` rope
            parameter; default ``True`` preserves classic YaRN).
        attention_factor: HF ``attention_factor``. When given it replaces
            ``mscale * attn_factor`` as the cos/sin magnitude, like transformers.
        max_positions: Number of cache rows. Defaults to
            ``max_position_embeddings * scaling_factor``.

    Returns:
        Float32 array of shape ``(max_positions, rotary_dim)`` with the
        ``[cos | sin]`` layout, each half already multiplied by the magnitude
        scale (``attention_factor`` or ``mscale * attn_factor``).
    """
    inv_freq = compute_yarn_inv_frequencies(
        base=base,
        rotary_dim=rotary_dim,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        max_position_embeddings=max_position_embeddings,
        scaling_factor=scaling_factor,
        extrapolation_factor=extrapolation_factor,
        truncate=truncate,
    )
    if max_positions is None:
        max_positions = max_position_embeddings * scaling_factor
    t = jnp.arange(max_positions, dtype=jnp.float32)
    freqs = jnp.einsum("i,j -> ij", t, inv_freq)
    if attention_factor is not None:
        mscale = attention_factor
    else:
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
    factor: float | None = None,
    attention_factor: float | None = None,
    short_mscale: float | None = None,
    long_mscale: float | None = None,
):
    """Compute Phi-3 LongRoPE frequencies (per-dimension scaling factors).

    Phi-3's RoPE extension does not use a smooth ramp like YaRN; instead it
    multiplies each inverse-frequency element by a learned scalar from
    either ``short_factor`` or ``long_factor``. After the multiplied cos/sin
    cache is built, a magnitude rescale is applied: ``attention_factor`` when
    given, otherwise ``sqrt(1 + log(s)/log(orig_max))`` with ``s = factor``
    (or ``max_position_embeddings / original_max_position_embeddings``).

    transformers picks the factor set per forward from the current sequence
    length (``long`` iff ``max(position_ids) + 1 > original_max_position_embeddings``).
    A static position-indexed cache cannot reproduce that exactly, so row
    ``p`` uses the set HF would use for a sequence of length ``p + 1``:
    ``short_factor`` for ``p < original_max_position_embeddings`` (exact HF
    match for every sequence that fits in the original window) and
    ``long_factor`` beyond it (HF's incremental-decode behaviour). A single
    forward longer than the original window differs from HF, which then
    rotates *all* positions with ``long_factor``. ``short_mscale`` /
    ``long_mscale`` (PhiMoE) follow the same per-row selection.

    Partial rotary (``rotary_dim < head_size``, e.g. Phi-4-mini) is supported:
    the factor lists have ``rotary_dim // 2`` entries and only the first
    ``rotary_dim`` channels are rotated by :func:`apply_phi3_rope`.

    Args:
        base: Base ``θ`` for the unscaled spectrum.
        head_size: Per-head dimension (unused; kept for API compatibility).
        rotary_dim: Rotary feature dimension (``head_size * partial_rotary_factor``).
        max_position_embeddings: Post-scaling target context length (cache rows).
        original_max_position_embeddings: Original training context.
        short_factor: Per-pair scaling vector for positions inside the
            original window.
        long_factor: Per-pair scaling vector for positions beyond it.
        factor: Optional explicit context-extension factor used for the
            default magnitude scale.
        attention_factor: Optional explicit magnitude scale (overrides the
            inferred one).
        short_mscale: Optional magnitude scale for rows inside the original
            window (PhiMoE); requires ``long_mscale``.
        long_mscale: Optional magnitude scale for rows beyond the original
            window (PhiMoE); requires ``short_mscale``.

    Returns:
        Float32 array of shape ``(1, max_position_embeddings, 2*rotary_dim)``
        with the ``[cos | sin]`` layout pre-scaled by the LongRoPE
        magnitude factor.
    """
    del head_size
    inv_freq_shape = jnp.arange(0, rotary_dim, 2, dtype=jnp.int32).astype(jnp.float32) / rotary_dim
    short_inv_freq = 1.0 / (jnp.array(short_factor, dtype=jnp.float32) * (base**inv_freq_shape))
    long_inv_freq = 1.0 / (jnp.array(long_factor, dtype=jnp.float32) * (base**inv_freq_shape))

    positions = jnp.arange(max_position_embeddings, dtype=jnp.int32)
    use_long = (positions >= original_max_position_embeddings)[:, None]
    position_values = positions.astype(jnp.float32)[:, None]
    freqs = jnp.where(use_long, position_values * long_inv_freq[None, :], position_values * short_inv_freq[None, :])
    emb = jnp.concatenate((freqs, freqs), axis=-1)

    if short_mscale is not None and long_mscale is not None:
        scaling_factor = jnp.where(use_long, jnp.float32(long_mscale), jnp.float32(short_mscale))
    elif attention_factor is not None:
        scaling_factor = attention_factor
    else:
        scale = factor if factor is not None else max_position_embeddings / original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))

    cos = jnp.cos(emb) * scaling_factor
    sin = jnp.sin(emb) * scaling_factor
    return jnp.concatenate([cos, sin], axis=-1)[None]


@jax.named_scope("easydel-rotary-compute-llama3-frequencies")
def compute_llama3_frequencies(
    base,
    rotary_dim,
    low_freq_factor,
    high_freq_factor,
    scaling_factor,
    max_position_embeddings: int,
    orig_max_position: int | None = None,
):
    """Compute Llama-3 wavelength-piecewise RoPE frequencies.

    Builds the cos/sin cache by outer-producing positions with
    :func:`compute_llama3_inv_frequencies` (wavelength-piecewise scaled
    inverse frequencies).

    Args:
        base: Base ``θ`` for the unscaled spectrum (Llama-3 uses 500 000).
        rotary_dim: Rotary feature dimension (must be even).
        low_freq_factor: Low-frequency boundary parameter (Llama-3 uses 1).
        high_freq_factor: High-frequency boundary parameter (Llama-3 uses 4).
        scaling_factor: Target context-length multiplier.
        max_position_embeddings: Length of the produced cache (the
            post-extension context; transformers rotates any position).
        orig_max_position: Original training context length (Llama-3 uses
            8192) that sets the wavelength bands. Defaults to
            ``max_position_embeddings`` for backward compatibility.

    Returns:
        Float32 array of shape ``(max_position_embeddings, rotary_dim)``
        with the standard ``[cos | sin]`` layout.
    """
    if orig_max_position is None:
        orig_max_position = max_position_embeddings
    inv = compute_llama3_inv_frequencies(
        base,
        rotary_dim,
        low_freq_factor,
        high_freq_factor,
        orig_max_position,
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
    attention_factor: float | None = None,
    max_positions: int | None = None,
    truncate: bool = True,
) -> jnp.ndarray:
    """Compute DeepSeek-YaRN-scaled RoPE frequencies with two-mscale rescaling.

    Same interpolation/extrapolation blend as standard YaRN, but the attention
    magnitude factor is computed as the *ratio* of two ``yarn_get_mscale``
    evaluations parameterised by ``mscale`` and ``mscale_all_dim`` — a
    DeepSeek-specific tweak that decouples the per-dim and all-dim YaRN
    correction curves.

    Args:
        base: Base ``θ`` for the unscaled spectrum.
        rotary_dim: Rotary feature dimension (must be even).
        scaling_factor: Target context-length multiplier.
        extrapolation_factor: Scalar in ``[0, 1]`` weighting the
            extrapolation branch.
        beta_fast: YaRN fast-band boundary.
        beta_slow: YaRN slow-band boundary.
        max_position_embeddings: Original training context length.
        mscale: Per-dim mscale exponent.
        mscale_all_dim: All-dim mscale exponent (denominator of the ratio).
        attn_factor: User-tunable multiplier composed with the YaRN mscale.
        attention_factor: HF ``attention_factor``. When given it replaces the
            ``mscale`` ratio (times ``attn_factor``), like transformers.
        max_positions: Number of cache rows. Defaults to
            ``max_position_embeddings * scaling_factor``.
        truncate: HF YaRN ``truncate`` flag for the correction band.

    Returns:
        Float32 array of shape ``(max_positions, rotary_dim)`` with the
        ``[cos | sin]`` layout, magnitudes rescaled by the DeepSeek attention
        factor.
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
        truncate=truncate,
    )
    inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=jnp.float32)) * extrapolation_factor
    inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

    if max_positions is None:
        max_positions = max_position_embeddings * scaling_factor
    t = jnp.arange(max_positions, dtype=jnp.float32)
    freqs = jnp.einsum("i,j -> ij", t, inv_freq)
    if attention_factor is None:
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
    offsets: jax.Array | None = None,
    dtype: jnp.dtype = jnp.float32,
):
    """Apply Phi-3 LongRoPE rotation to query and key tensors.

    Looks up the precomputed Phi-3 cos/sin cache at ``positions`` (plus
    optional ``offsets``), splits the cached embedding into cos/sin halves,
    and applies Neox-style rotation (``x * cos + rotate_neox(x) * sin``) under
    a ``float32`` matmul-precision context to preserve numerical fidelity.
    When the cache is narrower than the head (partial rotary, e.g. Phi-4-mini)
    only the first ``cos.shape[-1]`` channels are rotated and the rest pass
    through, like transformers' Phi-3 ``apply_rotary_pos_emb``.

    Args:
        query: Query tensor of shape
            ``[batch_size, sequence_length, num_heads, head_dim]``.
        key: Key tensor with the same shape as ``query``.
        positions: Position indices of shape ``[batch_size, sequence_length]``.
        frequencies: Phi-3 cache produced by
            :func:`compute_phi3_frequencies`; shape ``[1, max_length, 2*rotary_dim]``.
        offsets: Optional per-position offset to add before look-up.
        dtype: Output dtype (cast applied at the end).

    Returns:
        Tuple ``(query_rot, key_rot)`` cast to ``dtype``.
    """
    positions = positions
    if offsets is not None:
        positions = positions + offsets
    emb = frequencies[0, positions]
    cos, sin = jnp.split(emb, 2, axis=-1)
    cos = jnp.expand_dims(cos, 2)
    sin = jnp.expand_dims(sin, 2)
    rotary_dim = cos.shape[-1]

    with jax.default_matmul_precision("float32"):
        if rotary_dim < query.shape[-1]:
            query_rot = query[..., :rotary_dim] * cos + _rotate_neox(query[..., :rotary_dim]) * sin
            key_rot = key[..., :rotary_dim] * cos + _rotate_neox(key[..., :rotary_dim]) * sin
            query_rot = jnp.concatenate((query_rot, query[..., rotary_dim:]), axis=-1)
            key_rot = jnp.concatenate((key_rot, key[..., rotary_dim:]), axis=-1)
        else:
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
    offsets: jax.Array | None = None,
    dtype: jnp.dtype = jnp.float32,
):
    """Apply (optionally partial) RoPE rotation to query and key tensors.

    Looks up the cos/sin cache at ``positions + offsets``, then applies
    :func:`_apply_rotary_emb` to the first ``rotary_dim`` channels. When
    ``rotary_dim < head_dim``, the un-rotated tail is concatenated back so
    callers can use the partial-RoPE pattern (rotate only the lower portion
    of each head, leave NoPE channels at the end).

    Args:
        query: Query tensor of shape
            ``[..., sequence_length, num_heads, head_dim]``.
        key: Key tensor with the same shape as ``query``.
        positions: Position indices of shape ``[sequence_length]``.
        frequencies: Precomputed cos/sin cache with ``[cos | sin]`` layout
            along the last axis.
        rotary_dim: Number of channels (from index 0) to rotate. Must be
            even.
        is_neox_style: When ``True`` use Neox interleaving (pair adjacent
            halves), otherwise the GPT-J pairing.
        offsets: Optional per-position offset to add before look-up.
        dtype: Note: this argument is currently *unused*; the returned
            arrays inherit the dtype of ``query`` / ``key``. Kept for API
            parity with :func:`apply_phi3_rope`.

    Returns:
        Tuple ``(query, key)`` with the same shape as the inputs.
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
