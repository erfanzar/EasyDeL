# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dense (fully connected) matrix multiplication with optional bias.

Implemented via :func:`jax.lax.dot_general` rather than
:func:`jax.numpy.matmul` so the contraction can be expressed without
the extra dispatch and reshape overhead of the general matmul entry
point. The dtype-promotion logic on :func:`linear` follows the
convention used by mainstream training paths: the higher-precision
operand is downcast to the lower-precision one to keep the accelerator
on its native fast matmul path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..core._typing import Array, ArrayLike


def linear(x: ArrayLike, w: ArrayLike, b: ArrayLike | None = None) -> Array:
    """Dense multiply with optional bias: ``y = x @ w + b``.

    Contracts the trailing axis of ``x`` with the leading axis of ``w``
    via :func:`jax.lax.dot_general` directly — cheaper than
    :func:`jax.numpy.matmul` for this fixed ``(batch..., in) @ (in, out)``
    shape because it avoids the dispatch overhead of the general
    matmul entry point.

    Dtype handling mirrors mainstream TPU/GPU training practice: when
    the input and weight have mismatched floating-point precision, the
    *higher*-precision operand is downcast to match the lower-precision
    one. Downcasting the weight rather than upcasting the activation
    keeps the accelerator on its native low-precision matmul path
    (bf16 or fp16 matmul with an fp32 accumulator) and avoids
    materializing an upcast of the (usually much larger) activation
    tensor. If the input is the higher-precision operand it gets
    downcast to the weight's dtype instead — symmetrical. Non-floating
    dtypes or matching dtypes skip the conversion entirely.

    The bias (if supplied) is cast to the matmul output's dtype before
    the addition, so a stored-in-fp32 bias composes cleanly with a
    bf16 matmul result without a surprise upcast on the residual
    connection.

    Args:
        x: Input array whose trailing axis is the contraction ("in")
            dimension. All leading axes pass through unchanged.
        w: Weight matrix of shape ``(in, out)``.
        b: Optional bias broadcastable to ``(..., out)``.

    Returns:
        The dense product ``x @ w`` (with ``b`` added when supplied),
        in the promoted dtype selected above.
    """
    xa = jnp.asarray(x)
    wa = jnp.asarray(w)
    x_dtype = xa.dtype
    w_dtype = wa.dtype
    if x_dtype != w_dtype and jnp.issubdtype(x_dtype, jnp.floating) and jnp.issubdtype(w_dtype, jnp.floating):
        if jnp.finfo(x_dtype).bits < jnp.finfo(w_dtype).bits:
            wa = wa.astype(x_dtype)
        else:
            xa = xa.astype(w_dtype)
    y = jax.lax.dot_general(xa, wa, (((xa.ndim - 1,), (0,)), ((), ())))
    if b is not None:
        ba = jnp.asarray(b)
        if ba.dtype != y.dtype and jnp.issubdtype(ba.dtype, jnp.floating) and jnp.issubdtype(y.dtype, jnp.floating):
            ba = ba.astype(y.dtype)
        y = y + ba
    return y
