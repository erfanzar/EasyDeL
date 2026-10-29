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

"""Utility functions for Rotary Position Embeddings (RoPE).

This module provides low-level utility functions used in the computation and
application of rotary positional embeddings. These functions are primarily
internal helpers used by the main computation and module classes.

Functions:
    _yarn_find_correction_dim: Calculate correction dimension for YaRN scaling.
    _yarn_find_correction_range: Find correction range bounds for YaRN scaling.
    _yarn_linear_ramp_mask: Create linear ramp mask for YaRN scaling.
    _rotate_neox: Apply Neox-style rotation to tensor.
    _rotate_gptj: Apply GPT-J-style rotation to tensor.
    _apply_rotary_emb: Apply rotary embedding to input tensor.
    yarn_get_mscale: Calculate mscale with additional parameter (Deepseek variant).

Note:
    Functions prefixed with underscore are considered internal helpers.
    The public `yarn_get_mscale` function is exported for external use.
"""

from __future__ import annotations

import math
import typing as tp
from collections.abc import Mapping
from copy import deepcopy

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float


def rope_scaling_from_rope_parameters(
    rope_parameters: Mapping[str, tp.Any] | None,
    rope_scaling: Mapping[str, tp.Any] | None,
    default: Mapping[str, tp.Any] | None = None,
) -> dict[str, tp.Any] | None:
    """Normalize HF RoPE parameters into an EasyDeL scaling mapping.

    Explicit ``rope_scaling`` takes precedence and retains every supplied key.
    Otherwise the supported scaling and multimodal fields are copied from
    ``rope_parameters``. The legacy ``type`` spelling aliases ``rope_type``;
    an explicit ``rope_type`` always wins. Neither input is modified.

    Args:
        rope_parameters: HF-style RoPE parameters, or ``None``.
        rope_scaling: Explicit EasyDeL scaling mapping, or ``None``.
        default: Fallback when both inputs are absent. A deep copy is returned
            so mutable presets (such as ``mrope_section``) remain independent.

    Returns:
        Normalized scaling dictionary, or ``None`` if both inputs and the
        default are absent.
    """
    if rope_scaling is not None:
        result = dict(rope_scaling)
        if "type" in result and "rope_type" not in result:
            result["rope_type"] = result["type"]
        return result
    if rope_parameters is None:
        return None if default is None else deepcopy(dict(default))

    from ._configs import RopeConfig

    # The schema owns the supported fields and legacy aliases, including
    # magnitude/correction overrides. Do not maintain a second lossy whitelist.
    return RopeConfig.from_dict(dict(rope_parameters)).to_dict()


def rotate_half(x: Array) -> Array:
    """Return the NeoX split-half rotation partner ``[-x2, x1]``."""
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope_split_half(
    x: Array,
    cos: Array,
    sin: Array,
    *,
    unsqueeze_dim: int | None = -2,
    compute_dtype: jnp.dtype | None = jnp.float32,
) -> Array:
    """NeoX split-half RoPE ``x * cos + rotate_half(x) * sin`` on the leading ``cos.shape[-1]`` channels.

    Channels past ``cos.shape[-1]`` pass through unchanged (partial rotary); a full-width table takes the plain path.

    Args:
        x: Tensor to rotate, e.g. ``[seq, heads, head_dim]``.
        cos: Cosine table, e.g. ``[seq, rot_dim]``.
        sin: Sine table, same shape as ``cos``.
        unsqueeze_dim: Axis inserted into ``cos`` / ``sin`` to broadcast over heads (``None`` = already broadcastable).
        compute_dtype: Dtype the rotation runs in; the result is cast back to ``x``'s dtype. ``None`` computes in the
            operands' own dtypes and returns their promoted dtype.

    Returns:
        The rotated tensor.
    """
    if unsqueeze_dim is not None:
        cos = jnp.expand_dims(cos, unsqueeze_dim)
        sin = jnp.expand_dims(sin, unsqueeze_dim)
    orig_dtype = x.dtype
    if compute_dtype is not None:
        x, cos, sin = x.astype(compute_dtype), cos.astype(compute_dtype), sin.astype(compute_dtype)
    rot_dim = cos.shape[-1]
    if rot_dim == x.shape[-1]:
        out = (x * cos) + (rotate_half(x) * sin)
    else:
        x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        out = jnp.concatenate([(x_rot * cos) + (rotate_half(x_rot) * sin), x_pass], axis=-1)
    return out if compute_dtype is None else out.astype(orig_dtype)


@jax.named_scope("easydel-rotary-yarn-find-correction-dim")
def _yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float | Array:
    """Solve for the rotation-plane index whose wavelength matches ``num_rotations``.

    YaRN's piecewise scaling regions are parameterised by *number of full
    rotations* completed within the original training window
    (``beta_fast`` / ``beta_slow`` are stated in those units). For a given
    rotation budget ``num_rotations`` and base period ``base``, this helper
    inverts the standard RoPE wavelength formula
    ``λ_i = 2π * base ** (2i/dim)`` for ``i`` such that the corresponding
    plane completes exactly ``num_rotations`` cycles in
    ``max_position_embeddings`` tokens — i.e. it solves
    ``max_position_embeddings / λ_i = num_rotations`` for the (continuous,
    fractional) ``i``.

    Args:
        num_rotations: Target number of full rotations across the original
            training window.
        dim: Total rotary feature dimension.
        base: RoPE base period (typically 10000 for unscaled).
        max_position_embeddings: Original (pre-extension) training context
            length.

    Returns:
        Continuous (possibly fractional) rotation-plane index ``i`` that
        :func:`_yarn_find_correction_range` will floor/ceil to integer
        boundaries.
    """
    return (
        dim
        * jnp.log(
            max_position_embeddings / (num_rotations * 2 * jnp.pi),
        )
    ) / (2 * jnp.log(base))


@jax.named_scope("easydel-rotary-yarn-find-correction-range")
def _yarn_find_correction_range(  # pyright: ignore[reportUnusedFunction]
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
    truncate: bool = True,
) -> tuple[int | Array, int | Array]:
    """Compute the YaRN correction band ``[low, high]`` from rotation budgets.

    Calls :func:`_yarn_find_correction_dim` once per rotation budget
    (``low_rot``, ``high_rot``), then — when ``truncate`` is set — floors the
    low end and ceils the high end so the band covers every plane *fully*
    extrapolated and *fully* interpolated. Result is clipped to
    ``[0, dim - 1]`` to keep downstream indexing safe.

    Args:
        low_rot: Lower rotation budget (rotations within the original
            context that still count as "slow").
        high_rot: Upper rotation budget (rotations beyond which the plane
            is "fast").
        dim: Total rotary feature dimension.
        base: RoPE base period.
        max_position_embeddings: Original training context length.
        truncate: Whether to floor/ceil the band to integer plane indices
            (matches transformers' ``truncate`` rope parameter; default
            ``True`` preserves the classic YaRN behaviour).

    Returns:
        Tuple ``(low, high)`` with ``low <= high``, both clipped to
        ``[0, dim - 1]``, ready to be consumed by
        :func:`_yarn_linear_ramp_mask`.
    """
    hr = _yarn_find_correction_dim(
        high_rot,
        dim,
        base,
        max_position_embeddings,
    )
    lr = _yarn_find_correction_dim(
        low_rot,
        dim,
        base,
        max_position_embeddings,
    )
    if truncate:
        hr = jnp.ceil(hr)
        lr = jnp.floor(lr)
    return jax.lax.max(lr, 0.0), jax.lax.min(hr, jnp.array(dim - 1, dtype=jnp.float32))


@jax.named_scope("easydel-rotary-yarn-linear-ramp-mask")
def _yarn_linear_ramp_mask(  # pyright: ignore[reportUnusedFunction]
    low: float,
    high: float,
    dim: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Build the YaRN linear ramp blending interpolation and extrapolation.

    Produces a length-``dim`` vector that is ``0`` for indices below
    ``low``, ``1`` for indices above ``high``, and linearly interpolates in
    between. YaRN multiplies this mask against the per-plane extrapolation
    weight so the blend transitions smoothly from "purely interpolated" on
    the slow dims to "purely extrapolated" on the fast dims. A tiny epsilon
    is added when ``low == high`` to avoid a divide-by-zero.

    Args:
        low: Ramp start (output is ``0`` at and below this index).
        high: Ramp end (output is ``1`` at and above this index).
        dim: Length of the produced 1-D mask.
        dtype: Output dtype.

    Returns:
        1-D array of shape ``(dim,)`` containing values in ``[0, 1]``.
    """
    high = jax.lax.cond(low == high, lambda x: x + 0.001, lambda x: x, high)
    linear_func = (jnp.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = jnp.clip(linear_func, 0, 1)
    return ramp_func


@jax.named_scope("easydel-rotary-rotate-neox")
def _rotate_neox(x: Float[Array, "... seq_len head_dim"]) -> Float[Array, "... seq_len head_dim"]:  # pyright: ignore[reportUnusedFunction]
    """Apply the Neox-style 90-degree rotation to the last axis.

    Splits the last axis into two halves and returns
    ``concat(-second_half, first_half)`` — the Neox / Llama convention used
    by most published RoPE implementations. Combined with element-wise
    ``x * cos + rotate_neox(x) * sin`` this realises the planar rotation
    where each pair ``(x_i, x_{i + d/2})`` is treated as one complex number.

    Args:
        x: Tensor whose last axis is split and rotated.

    Returns:
        Tensor with the same shape as ``x``, rotated 90° in each pairwise
        plane.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


@jax.named_scope("easydel-rotary-rotate-gptj")
def _rotate_gptj(x: Float[Array, "... seq_len head_dim"]) -> Float[Array, "... seq_len head_dim"]:  # pyright: ignore[reportUnusedFunction]
    """Apply the GPT-J-style 90-degree rotation to the last axis.

    The GPT-J convention pairs adjacent elements ``(x_{2k}, x_{2k+1})``
    rather than the halves; this helper produces
    ``interleave(-x_odd, x_even)`` so that downstream
    ``x * cos + rotate_gptj(x) * sin`` realises the same planar rotation as
    Neox-style, just with a different physical interleaving.

    Args:
        x: Tensor whose last axis is split into even/odd lanes and rotated.

    Returns:
        Tensor with the same shape as ``x``, rotated 90° in each pairwise
        plane.
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = jnp.stack((-x2, x1), axis=-1)
    return x.reshape((*x.shape[:-2], -1))


@jax.named_scope("easydel-rotary-apply-rotary-emb")
def _apply_rotary_emb(  # pyright: ignore[reportUnusedFunction]
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    is_neox_style: bool,
) -> jnp.ndarray:
    """Apply RoPE to a query- or key-shaped tensor.

    Splits ``x`` into two halves according to ``is_neox_style`` (halved for
    Neox, even/odd lanes for GPT-J), then computes the standard 2-D rotation
    ``(x1*cos - x2*sin, x2*cos + x1*sin)`` and re-assembles into the
    original layout. ``cos`` / ``sin`` are gathered up to the heads axis via
    ``cos[:, :, None]`` so they broadcast across all heads.

    Args:
        x: Query or key tensor; the last axis is rotated.
        cos: Pre-gathered cosine components for the active positions.
        sin: Pre-gathered sine components for the active positions.
        is_neox_style: ``True`` splits the last axis into halves (Neox),
            ``False`` interleaves even/odd lanes (GPT-J). Must match the
            layout of ``cos`` / ``sin``.

    Returns:
        Tensor with the same shape as ``x`` carrying RoPE-rotated values
        in the rotated axis.

    Raises:
        ValueError: If ``sin.ndim`` does not match ``x.ndim`` after the
            broadcast-axis expansion (sanity check).
    """
    cos = cos[:, :, None].astype(x.dtype)
    sin = sin[:, :, None].astype(x.dtype)
    if sin.ndim != x.ndim:
        raise ValueError(f"sin.ndim ({sin.ndim}) must match x.ndim ({x.ndim})")
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
    """DeepSeek-YaRN magnitude rescale: ``1 + 0.1 * mscale * log(scale)``.

    DeepSeek's variant of YaRN parametrises the logits magnitude rescale
    with an extra coefficient ``mscale`` that defaults to ``1.0`` for the
    full-dim path and to a different DeepSeek-specific constant for the
    rotated half. The actual factor used at attention time is the *ratio*
    of two calls — see :func:`compute_deepseek_frequencies` for the
    ``yarn_get_mscale(s, mscale) / yarn_get_mscale(s, mscale_all_dim)``
    pattern.

    Args:
        scale: Context-length extension factor; ``<= 1`` returns ``1.0``.
        mscale: DeepSeek's additional per-call coefficient — typically
            ``config.mscale`` for the rotated half and ``config.mscale_all_dim``
            for the unrotated half.

    Returns:
        Scalar magnitude multiplier (a Python float — the function uses
        ``math.log`` rather than ``jnp.log`` because it is invoked at
        cache-build time on Python floats from the config).
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def apply_rope_interleaved(x, cos, sin):
    """Apply RoPE where the rotated channels are even/odd interleaved.

    GPT-J-style pairing: channel ``2i`` and ``2i+1`` form one rotation pair, as
    opposed to the NeoX/split-half layout where ``i`` pairs with ``i + d/2``.
    DeepSeek-V2/V3 and the GLM/Mistral MLA descendants all use this form, and
    each had written it out identically.

    Args:
        x: Tensor whose trailing axis carries interleaved rotated channels.
        cos: Per-position cosines, broadcastable to ``x``.
        sin: Per-position sines, broadcastable to ``x``.

    Returns:
        Rotated tensor with the same shape as ``x``.
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return jnp.stack((o1, o2), axis=-1).reshape(x.shape)


def cos_sin_from_inv_frequencies(
    inv_freq: Array,
    position_ids: Array,
    dtype: jnp.dtype,
    attention_scaling: float = 1.0,
) -> tuple[Array, Array]:
    """Half-width ``(cos, sin)`` tables at ``position_ids`` (one entry per rotation pair).

    Args:
        inv_freq: Inverse frequencies ``[rotary_dim // 2]``.
        position_ids: Integer positions ``[batch, seq]``.
        dtype: Output dtype.
        attention_scaling: Magnitude factor multiplied into both tables (YaRN).

    Returns:
        ``(cos, sin)``, each ``[batch, seq, rotary_dim // 2]``.
    """
    inv = jnp.asarray(inv_freq, dtype=jnp.float32)
    freqs = position_ids.astype(jnp.float32)[..., None] * inv[None, None, :]
    return (jnp.cos(freqs) * attention_scaling).astype(dtype), (jnp.sin(freqs) * attention_scaling).astype(dtype)


def apply_rope_interleaved_partial(x: Array, cos: Array, sin: Array, unsqueeze_dim: int = 1) -> Array:
    """Interleaved (GPT-J pair) RoPE on the *trailing* ``2 * cos.shape[-1]`` channels of ``x``, in float32.

    Leading channels pass through. ``cos`` / ``sin`` are half width (one entry per pair), as produced by
    :func:`cos_sin_from_inv_frequencies`; pass ``-sin`` for the inverse rotation. The rotation is written pair-major
    (the rope slice reshaped to explicit ``(half, 2)`` pairs) so cos/sin are never repeat-interleaved::

        out[2k]   = x1 * cos[k] - x2 * sin[k]
        out[2k+1] = x1 * sin[k] + x2 * cos[k]

    which equals :func:`apply_rope_interleaved` over the slice, computed in float32 and cast back to ``x.dtype``.

    Args:
        x: Input ``[..., seq, dim]`` with a broadcastable head axis at ``unsqueeze_dim``.
        cos: Half-width cosines ``[batch, seq, rope_dim // 2]``.
        sin: Half-width sines ``[batch, seq, rope_dim // 2]``.
        unsqueeze_dim: Axis at which cos/sin gain a head axis (1 for ``[B, H, S, D]``, 2 for ``[B, S, H, D]``).

    Returns:
        Array with the shape and dtype of ``x``.
    """
    half = cos.shape[-1]
    rope_dim = 2 * half
    cos = jnp.expand_dims(cos, unsqueeze_dim).astype(jnp.float32)
    sin = jnp.expand_dims(sin, unsqueeze_dim).astype(jnp.float32)
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    pairs = rope.astype(jnp.float32).reshape(*rope.shape[:-1], half, 2)
    x1 = pairs[..., 0]
    x2 = pairs[..., 1]
    rotated = jnp.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), axis=-1)
    rotated = rotated.reshape(rope.shape).astype(x.dtype)
    return jnp.concatenate([nope, rotated], axis=-1)


def gather_mla_cos_sin(frequencies, position_ids, dtype):
    """Split an MLA frequency table into per-position ``(cos, sin)``.

    The stored table is a full-width ``concatenate([freqs, freqs], -1)``; MLA
    layers gather it at ``position_ids`` and split it back into halves, adding a
    head axis so it broadcasts over ``[batch, heads, seq, dim]``. Three families
    carried this block character-for-character.

    Args:
        frequencies: Frequency table, or a parameter wrapper exposing ``.value``.
        position_ids: Positions to gather, shape ``[batch, seq]``.
        dtype: Dtype the returned tables are cast to.

    Returns:
        ``(cos, sin)``, each ``[batch, 1, seq, dim // 2]``.
    """
    table = frequencies.value if hasattr(frequencies, "value") else frequencies
    freqs = table[position_ids]
    cos, sin = jnp.split(freqs, 2, -1)
    return cos[:, None, :, :].astype(dtype), sin[:, None, :, :].astype(dtype)
