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
from collections.abc import Mapping
from functools import wraps

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
    AVAILABLE_ROPE_TYPES,
    RotaryEmbedding,
)


class _StaticScaling(Mapping):
    """Immutable, hashable snapshot of a public RoPE configuration mapping."""

    def __init__(self, mapping):
        def freeze(value):
            if isinstance(value, Mapping):
                return _StaticScaling(value)
            if isinstance(value, (list, tuple)):
                return tuple(freeze(item) for item in value)
            return value

        self._items = tuple(sorted((key, freeze(value)) for key, value in mapping.items()))
        rope_type = mapping.get("rope_type", "default")
        self._implementation = AVAILABLE_ROPE_TYPES.get(rope_type)

    def __getitem__(self, key):
        for name, value in self._items:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self):
        return len(self._items)

    def __hash__(self):
        return hash((self._items, self._implementation))

    def __repr__(self):
        # ejit's persistent signature must not contain this snapshot's
        # process-local object address.
        return f"{type(self).__name__}({dict(self)!r}, implementation={self._implementation!r})"

    def __eq__(self, other):
        return (
            isinstance(other, _StaticScaling)
            and self._items == other._items
            and self._implementation is other._implementation
        )


def _hashable_rope_scaling(function):
    """Snapshot dict/list configuration before it reaches the static JIT cache."""

    @wraps(function)
    def wrapped(head_size, rotary_dim, max_position, base, rope_scaling=None, partial_rotary_factor=1.0):
        scaling = None if rope_scaling is None else _StaticScaling(rope_scaling)
        return function(head_size, rotary_dim, max_position, base, scaling, partial_rotary_factor)

    return wrapped


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
    with an ``mrope_section`` field. Multimodal layout composes with the
    requested frequency scaling rather than replacing it. When
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

    scaling = {} if rope_scaling is None else rope_scaling
    scaling_type = scaling.get("rope_type", "default")
    multimodal = "mrope_section" in scaling or scaling_type == "mrope"
    try:
        rope_class = AVAILABLE_ROPE_TYPES[scaling_type]
    except KeyError as error:
        raise ValueError(
            f"Unknown RoPE scaling type {scaling_type}. Registered rope types: {sorted(AVAILABLE_ROPE_TYPES)}."
        ) from error

    kwargs = dict(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position,
        base=base,
        is_neox_style=is_neox_style,
        dtype=dtype,
    )
    if multimodal:
        # Axis layout and frequency scaling are independent. Keep both in
        # the registered multimodal module, including direct (uncached) calls.
        return AVAILABLE_ROPE_TYPES["mrope"](
            **kwargs,
            mrope_section=scaling.get("mrope_section"),
            mrope_interleaved=scaling.get("mrope_interleaved", False),
            repetition_style=scaling.get("repetition_style", False),
            rope_scaling=dict(scaling),
        )
    if scaling_type == "llama3":
        kwargs.update(
            scaling_factor=scaling["factor"],
            low_freq_factor=scaling["low_freq_factor"],
            high_freq_factor=scaling["high_freq_factor"],
            orig_max_position=scaling["original_max_position_embeddings"],
        )
    elif scaling_type == "linear":
        kwargs["scaling_factors"] = scaling["factor"]
    elif scaling_type == "dynamic":
        original_max_position = scaling.get("original_max_position_embeddings", max_position)
        kwargs.update(
            scaling_factor=scaling["factor"],
            max_position_embeddings=original_max_position,
            max_positions=max(int(max_position), int(original_max_position * scaling["factor"])),
        )
    elif scaling_type in ("yarn", "deepseek_yarn"):
        scaling_factor = scaling.get("factor", scaling.get("scaling_factor"))
        if scaling_factor is None:
            raise ValueError("YaRN rope_scaling must contain 'factor' or 'scaling_factor' key")
        kwargs.update(
            scaling_factor=scaling_factor,
            max_position_embeddings=scaling.get("original_max_position_embeddings", max_position),
        )
        extra_keys = {"extrapolation_factor", "attn_factor", "beta_fast", "beta_slow", "truncate"}
        extra_keys.update(("mscale", "mscale_all_dim") if scaling_type == "deepseek_yarn" else ("attention_factor",))
        kwargs.update({key: value for key, value in scaling.items() if key in extra_keys})
    elif scaling_type == "longrope":
        kwargs.update(
            original_max_position_embeddings=scaling["original_max_position_embeddings"],
            short_factor=scaling["short_factor"],
            long_factor=scaling["long_factor"],
        )
    elif scaling_type != "default":
        # Custom registered classes receive the common geometry plus their config fields.
        kwargs.update({key: value for key, value in scaling.items() if key != "rope_type"})
    return rope_class(**kwargs)


@_hashable_rope_scaling
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
    ``compute_*_frequencies`` function in :mod:`._compute_fns`. Multimodal
    layout fields do not alter the requested spectrum; the MRoPE module
    performs THW selection at call time. Explicit ``rope_type="mrope"``
    retains the basic spectrum. Wrapped in :func:`ejit` with
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

    scaling_type = "default" if rope_scaling is None else rope_scaling.get("rope_type", "default")
    rope_class = AVAILABLE_ROPE_TYPES.get(scaling_type)
    if rope_class is None:
        raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    compute_frequencies = getattr(rope_class, "compute_frequencies", None)
    if compute_frequencies is not None:
        return compute_frequencies(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=base,
            rope_scaling=rope_scaling,
        )

    if rope_scaling is None:
        frequencies = compute_basic_frequencies(
            base=base,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position,
        )
    else:
        if scaling_type == "llama3":
            scaling_factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling["low_freq_factor"]
            high_freq_factor = rope_scaling["high_freq_factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            # The cache must cover every served position (HF rotates any
            # position); ``original_max_position`` only sets the wavelength bands.
            frequencies = compute_llama3_frequencies(
                base=base,
                rotary_dim=rotary_dim,
                low_freq_factor=low_freq_factor,
                high_freq_factor=high_freq_factor,
                scaling_factor=scaling_factor,
                max_position_embeddings=max_position,
                orig_max_position=original_max_position,
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
            original_max_position = rope_scaling.get("original_max_position_embeddings", max_position)
            frequencies = compute_dynamic_frequencies(
                rotary_dim=rotary_dim,
                max_position_embeddings=original_max_position,
                base=base,
                scaling_factor=scaling_factor,
                max_positions=max(int(max_position), int(original_max_position * scaling_factor)),
            )
        elif scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling.get("original_max_position_embeddings", max_position)  # for gpt_oss
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim")
            }
            # An explicit HF ``attention_factor`` replaces the inferred mscale.
            attention_factor = rope_scaling.get("attention_factor")
            # Never stop the cache short of the served context.
            max_positions = max(int(max_position), int(original_max_position * scaling_factor))
            truncate = bool(rope_scaling.get("truncate", True))

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
                    extra_kwargs.get("attn_factor", 1),
                    attention_factor=attention_factor,
                    max_positions=max_positions,
                    truncate=truncate,
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
                    attn_factor=extra_kwargs.get("attn_factor", 1),
                    truncate=truncate,
                    attention_factor=attention_factor,
                    max_positions=max_positions,
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
                extra_kwargs.get("attn_factor", 1),
                attention_factor=rope_scaling.get("attention_factor"),
                max_positions=max(int(max_position), int(original_max_position * scaling_factor)),
                truncate=bool(rope_scaling.get("truncate", True)),
            )
        elif scaling_type == "longrope":
            short_factor = rope_scaling["short_factor"]
            long_factor = rope_scaling["long_factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            frequencies = compute_phi3_frequencies(
                base=base,
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position,
                original_max_position_embeddings=original_max_position,
                short_factor=short_factor,
                long_factor=long_factor,
                factor=rope_scaling.get("factor"),
                attention_factor=rope_scaling.get("attention_factor"),
                short_mscale=rope_scaling.get("short_mscale"),
                long_mscale=rope_scaling.get("long_mscale"),
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


@_hashable_rope_scaling
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
    branch sizes the inverse frequencies by ``rotary_dim`` (partial rotary
    aware), matching :func:`compute_phi3_frequencies`.

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
        ``(rotary_dim // 2,)``.

    Raises:
        ValueError: If ``rope_scaling["rope_type"]`` is not registered.
    """
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)

    if rope_scaling is None:
        inv_frequencies = compute_basic_inv_frequencies(base=base, rotary_dim=rotary_dim)
    else:
        scaling_type = rope_scaling.get("rope_type", "default")

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

        elif scaling_type in ("default", "mrope"):
            inv_frequencies = compute_basic_inv_frequencies(base=base, rotary_dim=rotary_dim)
        elif scaling_type == "linear":
            factor = rope_scaling["factor"]
            if isinstance(factor, (list, tuple)):
                if len(factor) != 1:
                    raise ValueError("A single inverse-frequency vector requires one linear scaling factor.")
                factor = factor[0]
            if factor <= 0:
                raise ValueError("The linear scaling factor must be positive.")
            inv_frequencies = compute_basic_inv_frequencies(base=base, rotary_dim=rotary_dim) / factor
        elif scaling_type == "dynamic":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling.get("original_max_position_embeddings", max_position)
            sequence_length = max(max_position, original_max_position)
            adjusted_base = base * (
                (scaling_factor * sequence_length / original_max_position) - (scaling_factor - 1)
            ) ** (rotary_dim / (rotary_dim - 2))
            inv_frequencies = compute_basic_inv_frequencies(base=adjusted_base, rotary_dim=rotary_dim)
        elif scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            # Same fallback as ``get_frequencies`` (gpt-oss style configs omit the key).
            original_max_position = rope_scaling.get("original_max_position_embeddings", max_position)
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
                truncate=bool(rope_scaling.get("truncate", True)),
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
            inv_frequencies = compute_yarn_inv_frequencies(
                base=base,
                rotary_dim=rotary_dim,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
                max_position_embeddings=original_max_position,
                scaling_factor=scaling_factor,
                extrapolation_factor=extrapolation_factor,
                truncate=bool(rope_scaling.get("truncate", True)),
            )
        elif scaling_type == "longrope":
            short_factor = rope_scaling["short_factor"]
            long_factor = rope_scaling["long_factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            if max_position > original_max_position:
                ext_factors = jnp.array(long_factor, dtype=jnp.float32)
            else:
                ext_factors = jnp.array(short_factor, dtype=jnp.float32)

            inv_freq_shape = jnp.arange(0, rotary_dim, 2, dtype=jnp.int32).astype(jnp.float32) / rotary_dim
            inv_frequencies = 1.0 / (ext_factors * (base**inv_freq_shape))
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    return jnp.asarray(inv_frequencies)


def partial_rope_frequencies(
    head_dim: int,
    max_position: int,
    base: float,
    partial_rotary_factor: float = 1.0,
    rope_type: str = "default",
    factor: float = 1.0,
) -> jax.Array:
    """Cos/sin cache ``[max_position, head_dim]`` for a rotary covering only part of each head (Gemma-4 style).

    - ``rope_type="proportional"`` (HF ``_compute_proportional_rope_parameters``): the first
      ``int(partial_rotary_factor * head_dim // 2)`` frequencies use exponent denominator ``head_dim`` (the FULL head),
      the rest are zero (cos 1, sin 0), and all are divided by ``factor``.
    - otherwise, ``partial_rotary_factor < 1``: a standard table over ``int(head_dim * partial_rotary_factor)``
      channels, padded with cos 1 / sin 0 up to ``head_dim // 2`` pairs.
    - otherwise: the standard full-width table.

    Args:
        head_dim: Per-head dimension.
        max_position: Number of positions in the cache.
        base: RoPE ``theta``.
        partial_rotary_factor: Rotated fraction of the head.
        rope_type: ``"proportional"`` or anything else.
        factor: Frequency divisor of the proportional variant.

    Returns:
        ``concatenate([cos, sin], -1)`` of shape ``[max_position, head_dim]``.
    """
    if rope_type == "proportional":
        rope_angles = int(partial_rotary_factor * head_dim // 2)
        inv_freq = 1.0 / (base ** (jnp.arange(0, 2 * rope_angles, 2, dtype=jnp.float32) / head_dim))
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = jnp.concatenate((inv_freq, jnp.zeros((nope_angles,), dtype=jnp.float32)), axis=0)
        inv_freq = inv_freq / factor
        phase = jnp.arange(max_position, dtype=jnp.float32)[:, None] * inv_freq[None, :]
        return jnp.concatenate((jnp.cos(phase), jnp.sin(phase)), axis=-1)
    if partial_rotary_factor < 1.0:
        rotated_dim = int(head_dim * partial_rotary_factor)
        rotated = get_frequencies(
            head_size=rotated_dim,
            rotary_dim=rotated_dim,
            max_position=max_position,
            base=base,
            rope_scaling=None,
            partial_rotary_factor=1.0,
        )
        cos, sin = jnp.split(rotated, 2, axis=-1)
        pass_dim = head_dim // 2 - cos.shape[-1]
        return jnp.concatenate(
            (
                jnp.concatenate((cos, jnp.ones((cos.shape[0], pass_dim), dtype=cos.dtype)), axis=-1),
                jnp.concatenate((sin, jnp.zeros((sin.shape[0], pass_dim), dtype=sin.dtype)), axis=-1),
            ),
            axis=-1,
        )
    return get_frequencies(head_size=head_dim, rotary_dim=head_dim, max_position=max_position, base=base)
