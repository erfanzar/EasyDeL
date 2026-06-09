# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small shared utilities for the functional ops in :mod:`spectrax.functional`."""

from __future__ import annotations

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType

__all__ = ["promote_dtype"]


def promote_dtype(
    *arrays: ArrayLike,
    dtype: DType | None = None,
) -> tuple[Array, ...]:
    """Cast every input array to a common dtype.

    Used by layers that mix activations and parameters of different
    precisions (e.g. fp32 weights with bf16 inputs). When ``dtype`` is
    ``None`` the result dtype is derived by folding
    :func:`jax.numpy.promote_types` across the inputs in order, which
    matches NumPy's "promote everything to the widest" rule.

    Args:
        *arrays: Arbitrary array-like inputs.
        dtype: Optional explicit target dtype. ``None`` (default)
            triggers automatic promotion across the inputs.

    Returns:
        A tuple of arrays in the same order as the inputs, each cast to
        the resolved dtype. An empty input tuple returns ``()``.
    """
    arrs = [jnp.asarray(a) for a in arrays]
    if dtype is None:
        if not arrs:
            return ()
        out_dtype = arrs[0].dtype
        for a in arrs[1:]:
            out_dtype = jnp.promote_types(out_dtype, a.dtype)
    else:
        out_dtype = dtype
    return tuple(a.astype(out_dtype) for a in arrs)
