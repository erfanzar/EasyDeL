# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""N-dimensional convolution over channels-last inputs.

Both :func:`conv` and :func:`conv_transpose` follow the
``(N, *spatial, C)`` layout used throughout SpectraX (matching Flax /
Keras / TF). The kernel layout is ``(*kernel_spatial, C_in_per_group, C_out)``,
so a 2-D 3x3 conv with 16 input channels (single-group) and 32 output
channels has kernel shape ``(3, 3, 16, 32)``.

The dimension-spec strings used inside both functions
(``"NHWC"`` / ``"HWIO"``) are constructed from a fixed alphabet so the
helpers work for 1-D, 2-D, 3-D, and arbitrary higher-N spatial inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from ..core._typing import Array, ArrayLike

PaddingSpec = str | Sequence[tuple[int, int]]
"""Padding specification accepted by :func:`conv` / :func:`conv_transpose`.

Either:

* a string padding mode forwarded to JAX (``"SAME"`` keeps the spatial
  output the same size as the input under unit stride; ``"VALID"`` drops
  any partial windows), or
* an explicit per-spatial-axis sequence of ``(lo, hi)`` integer pairs
  giving the number of zero-padded elements on each side.
"""


def conv(
    x: ArrayLike,
    w: ArrayLike,
    b: ArrayLike | None = None,
    *,
    stride: int | Sequence[int] = 1,
    padding: PaddingSpec = "VALID",
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
) -> Array:
    """Apply an N-D convolution.

    Layout conventions:

    * Input ``x`` has shape ``(N, *spatial, C_in)``.
    * Kernel ``w`` has shape ``(*kernel_spatial, C_in // groups, C_out)``.
    * Output has shape ``(N, *spatial_out, C_out)``.

    Args:
        x: Input tensor.
        w: Convolution kernel.
        b: Optional bias of shape ``(C_out,)`` added to the output.
        stride: Per-axis stride (int broadcasts to all axes).
        padding: See :data:`PaddingSpec`.
        dilation: Per-axis kernel dilation (atrous convolution).
        groups: Depthwise-style grouping. Must divide ``C_in``.

    Returns:
        The convolved tensor with optional bias.
    """
    xa = jnp.asarray(x)
    wa = jnp.asarray(w)
    n_spatial = xa.ndim - 2
    if isinstance(stride, int):
        stride = (stride,) * n_spatial
    if isinstance(dilation, int):
        dilation = (dilation,) * n_spatial
    spatial_dims = "".join("HWDTUVXY"[:n_spatial])
    lhs_spec = "N" + spatial_dims + "C"
    rhs_spec = spatial_dims + "IO"
    out_spec = lhs_spec
    dim_numbers = jax.lax.conv_dimension_numbers(xa.shape, wa.shape, (lhs_spec, rhs_spec, out_spec))
    y = jax.lax.conv_general_dilated(
        lhs=xa,
        rhs=wa,
        window_strides=tuple(stride),
        padding=padding,
        rhs_dilation=tuple(dilation),
        dimension_numbers=dim_numbers,
        feature_group_count=groups,
    )
    if b is not None:
        y = y + jnp.asarray(b)
    return y


def conv_transpose(
    x: ArrayLike,
    w: ArrayLike,
    b: ArrayLike | None = None,
    *,
    stride: int | Sequence[int] = 1,
    padding: PaddingSpec = "VALID",
    dilation: int | Sequence[int] = 1,
) -> Array:
    """Apply an N-D transposed convolution (fractionally-strided conv).

    Layout mirrors :func:`conv`:

    * Input ``x``: ``(N, *spatial, C_in)``.
    * Kernel ``w``: ``(*kernel_spatial, C_in, C_out)``.
    * Output: ``(N, *spatial_out, C_out)``.

    Forwards to :func:`jax.lax.conv_transpose` with
    ``transpose_kernel=False`` so the kernel dimension layout matches the
    forward :func:`conv`. Strides act as *upsampling* factors (the
    spatial output is roughly ``stride * spatial_in``); dilations enlarge
    the kernel's receptive field.

    Args:
        x: Input tensor, shape ``(N, *spatial, C_in)``.
        w: Convolution kernel, shape ``(*kernel_spatial, C_in, C_out)``.
        b: Optional bias of shape ``(C_out,)`` added to the output.
        stride: Per-axis upsampling factor (int broadcasts to all axes).
        padding: See :data:`PaddingSpec`.
        dilation: Per-axis kernel dilation.

    Returns:
        The transposed-convolved tensor of shape
        ``(N, *spatial_out, C_out)`` with optional bias added.
    """
    xa = jnp.asarray(x)
    wa = jnp.asarray(w)
    n_spatial = xa.ndim - 2
    if isinstance(stride, int):
        stride = (stride,) * n_spatial
    if isinstance(dilation, int):
        dilation = (dilation,) * n_spatial
    spatial_dims = "".join("HWDTUVXY"[:n_spatial])
    lhs_spec = "N" + spatial_dims + "C"
    rhs_spec = spatial_dims + "IO"
    out_spec = lhs_spec
    dim_numbers = (lhs_spec, rhs_spec, out_spec)
    y = jax.lax.conv_transpose(
        lhs=xa,
        rhs=wa,
        strides=tuple(stride),
        padding=padding,
        rhs_dilation=tuple(dilation),
        dimension_numbers=dim_numbers,
        transpose_kernel=False,
    )
    if b is not None:
        y = y + jnp.asarray(b)
    return y
