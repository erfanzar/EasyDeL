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

"""Factory functions for creating RoPE embeddings and computing frequencies.

This module provides high-level factory functions that create RotaryEmbedding
instances and compute frequency caches based on configuration dictionaries.
These functions serve as the primary entry points for using RoPE in models.

Functions:
    get_rope: Factory function to create RotaryEmbedding instances.
    get_frequencies: Compute frequency cache based on scaling configuration.
    get_inv_frequencies: Compute inverse frequencies based on scaling configuration.

The functions support the following RoPE scaling types via the `rope_scaling` dict:
    - "default": Standard RoPE with no scaling.
    - "linear": Linear position scaling.
    - "dynamic": Dynamic NTK scaling.
    - "yarn": YaRN (Yet another RoPE extensioN) scaling.
    - "deepseek_yarn": Deepseek variant of YaRN scaling.
    - "longrope": Phi-3 LongRoPE scaling with short/long factors.
    - "llama3": Llama-3 style wavelength-based scaling.
    - "mrope": Multi-modal RoPE for vision-language models (Qwen2/3-VL).

Example:
    >>> from easydel.layers.rotary import get_rope, get_frequencies
    >>> # Create a standard RoPE embedding
    >>> rope = get_rope(head_size=64, rotary_dim=64, max_position=2048, base=10000)
    >>> # Create a YaRN-scaled RoPE embedding
    >>> rope_scaling = {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 2048}
    >>> rope_yarn = get_rope(head_size=64, rotary_dim=64, max_position=4096, base=10000, rope_scaling=rope_scaling)
    >>> # Get pre-computed frequencies
    >>> freqs = get_frequencies(head_size=64, rotary_dim=64, max_position=2048, base=10000)
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp

from easydel.utils import ejit

from ._compute_fns import (
    compute_basic_frequencies,
    compute_basic_inv_frequencies,
    compute_deepseek_frequencies,
    compute_dynamic_frequencies,
    compute_linear_frequencies,
    compute_llama3_frequencies,
    compute_llama3_inv_frequencies,
    compute_phi3_frequencies,
    compute_yarn_frequencies,
    compute_yarn_inv_frequencies,
)
from ._modules import (
    DeepseekScalingRotaryEmbedding,
    DynamicNTKScalingRotaryEmbedding,
    LinearScalingRotaryEmbedding,
    Llama3RotaryEmbedding,
    MultiModalRotaryEmbedding,
    Phi3LongRoPEScaledRotaryEmbedding,
    RotaryEmbedding,
    YaRNScalingRotaryEmbedding,
)
from ._utils import _yarn_find_correction_range, _yarn_linear_ramp_mask


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
    """Build the rotary-embedding module specified by ``rope_scaling``.

    Reads ``rope_scaling["rope_type"]`` and dispatches to the matching
    :class:`spx.Module` subclass in :mod:`._modules`. Also detects the
    HuggingFace Qwen2-VL convention where ``rope_type="default"`` is paired
    with an ``mrope_section`` field and rewrites it to ``"mrope"``. When
    ``rope_scaling is None`` returns the unscaled :class:`RotaryEmbedding`.

    Args:
        head_size: Per-head attention dimension.
        rotary_dim: Rotary feature dimension before applying
            ``partial_rotary_factor``.
        max_position: Target (post-extension) context length.
        base: Geometric-progression base ``θ``.
        is_neox_style: Neox vs GPT-J rotation layout. Defaults to ``True``.
        rope_scaling: Configuration dict carrying ``rope_type`` plus the
            per-method fields (factor, original_max_position_embeddings,
            beta_fast/slow, mscale, short/long factor, mrope_section, …).
            ``None`` means unscaled.
        dtype: Output dtype of the embeddings. Defaults to ``float32``.
        partial_rotary_factor: Scalar in ``(0, 1]`` shrinking ``rotary_dim``
            (e.g. ``0.5`` rotates only the lower half of each head).

    Returns:
        Configured :class:`RotaryEmbedding` subclass instance.

    Raises:
        ValueError: If ``rope_scaling["rope_type"]`` is not one of the
            registered names, or if a YaRN entry omits both ``factor`` and
            ``scaling_factor``.
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
        # HuggingFace Qwen2-VL uses rope_type='default' with mrope_section for mRoPE
        # We detect this and switch to mrope for proper multimodal rotary embedding
        if "mrope_section" in rope_scaling.keys():
            scaling_type = "mrope"

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
            scaling_factor = rope_scaling.get("factor", rope_scaling.get("scaling_factor"))
            if scaling_factor is None:
                raise ValueError("YaRN rope_scaling must contain 'factor' or 'scaling_factor' key")
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
        elif scaling_type == "mrope":
            rotary_emb = MultiModalRotaryEmbedding(
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position,
                base=base,
                is_neox_style=is_neox_style,
                dtype=dtype,
                mrope_section=rope_scaling.get("mrope_section"),
                mrope_interleaved=rope_scaling.get("mrope_interleaved", False),
                repetition_style=rope_scaling.get("repetition_style", False),
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    return rotary_emb  # pyright: ignore[reportReturnType]


@ejit(  # pyright: ignore[reportUntypedFunctionDecorator]
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
    """Compute the cos/sin frequency cache for the configured RoPE variant.

    Reads ``rope_scaling["rope_type"]`` and dispatches to the matching
    ``compute_*_frequencies`` function in :mod:`._compute_fns`. The Qwen2-VL
    ``rope_type="default" + mrope_section`` convention is auto-rewritten to
    ``"mrope"`` (which falls back to the basic cache; the MRoPE module
    performs the THW interleaving at call time). Wrapped in :func:`ejit` with
    the heavy fields marked static so a JIT cache miss only happens when
    geometry changes.

    Args:
        head_size: Per-head attention dimension (needed by the Phi-3 path).
        rotary_dim: Rotary feature dimension before applying
            ``partial_rotary_factor``.
        max_position: Length of the produced cache (target context for the
            length-extending variants).
        base: Geometric-progression base ``θ``.
        rope_scaling: Configuration dict; ``None`` means unscaled.
        partial_rotary_factor: Scalar in ``(0, 1]`` shrinking ``rotary_dim``.

    Returns:
        Frequency cache as a JAX array. Layout is
        ``[length, 2*rotary_dim_effective]`` with ``[cos | sin]`` along the
        last axis (except Phi-3 which prepends a leading axis of size 1).

    Raises:
        ValueError: If ``rope_scaling["rope_type"]`` is not registered.
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
        # HuggingFace Qwen2-VL uses rope_type='default' with mrope_section for mRoPE
        # We detect this and switch to mrope for proper multimodal rotary embedding
        if "mrope_section" in rope_scaling.keys():
            scaling_type = "mrope"

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
                if k in ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim")
            }

            # Check if this is DeepSeek-style YaRN (has mscale and mscale_all_dim parameters)
            if "mscale" in extra_kwargs and "mscale_all_dim" in extra_kwargs:
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
            else:
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
        elif scaling_type == "mrope":
            # Use basic cache; interleaving handled inside the MRoPE class
            frequencies = compute_basic_frequencies(
                base=base,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    return frequencies


@ejit(  # pyright: ignore[reportUntypedFunctionDecorator]
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
    """Compute the inverse-frequency vector for the configured RoPE variant.

    Like :func:`get_frequencies` but stops *before* the outer product with
    positions and the cos/sin transformation. Some callers (notably ones that
    apply RoPE on-the-fly inside a fused attention kernel) want just the
    ``θ_i`` vector and build the cache themselves. The Phi-3 (``longrope``)
    branch uses ``head_size`` rather than ``rotary_dim`` to size the inverse-
    frequency layout, matching :func:`compute_phi3_frequencies`.

    Args:
        head_size: Per-head attention dimension (needed by the Phi-3 path).
        rotary_dim: Rotary feature dimension before applying
            ``partial_rotary_factor``.
        max_position: Used by the ``longrope`` branch to choose between
            ``short_factor`` and ``long_factor``.
        base: Geometric-progression base ``θ``.
        rope_scaling: Configuration dict; ``None`` means unscaled.
        partial_rotary_factor: Scalar in ``(0, 1]`` shrinking ``rotary_dim``.

    Returns:
        Float32 JAX array of inverse frequencies, typically of shape
        ``(rotary_dim // 2,)`` (Phi-3 uses ``head_size // 2``).

    Raises:
        ValueError: If ``rope_scaling["rope_type"]`` is not registered.
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

    return jnp.asarray(inv_frequencies)
