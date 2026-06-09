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
    _yarn_get_mscale: Calculate mscale factor for YaRN context extension.
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

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


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
) -> tuple[int | Array, int | Array]:
    """Compute the integer YaRN correction band ``[low, high]`` from rotation budgets.

    Calls :func:`_yarn_find_correction_dim` once per rotation budget
    (``low_rot``, ``high_rot``), then floors the low end and ceils the high
    end so the band covers every plane *fully* extrapolated and *fully*
    interpolated. Result is clipped to ``[0, dim - 1]`` to keep downstream
    indexing safe.

    Args:
        low_rot: Lower rotation budget (rotations within the original
            context that still count as "slow").
        high_rot: Upper rotation budget (rotations beyond which the plane
            is "fast").
        dim: Total rotary feature dimension.
        base: RoPE base period.
        max_position_embeddings: Original training context length.

    Returns:
        Tuple ``(low, high)`` with ``low <= high``, both clipped to
        ``[0, dim - 1]``, ready to be consumed by
        :func:`_yarn_linear_ramp_mask`.
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


@jax.named_scope("easydel-rotary-yarn-get-mscale")
def _yarn_get_mscale(scale: float = 1) -> float | Array:  # pyright: ignore[reportUnusedFunction]
    """Compute YaRN's logits-magnitude rescale ``1 + 0.1 * log(scale)``.

    YaRN proposes that, after extending the context length by ``scale``,
    the *magnitudes* of attention logits should be amplified by a small
    factor that grows logarithmically with the extension. This compensates
    for the softer attention distributions that result from interpolating
    the high-frequency rotation planes. The published constant ``0.1`` is
    used; ``scale <= 1`` is a no-op.

    Args:
        scale: Context-length multiplier (e.g. ``8.0`` to extend an 8K
            model to 64K). Values ``<= 1`` short-circuit to ``1.0``.

    Returns:
        Scalar magnitude multiplier applied to cos/sin in
        :func:`compute_yarn_frequencies`.
    """
    if scale <= 1:
        return 1.0
    return 0.1 * jnp.log(scale) + 1.0


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
