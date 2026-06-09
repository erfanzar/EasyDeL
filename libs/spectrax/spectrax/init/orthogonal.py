# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Orthogonal initializer obtained via a QR decomposition of random noise.

Useful for recurrent / very-deep architectures where preserving an
orthogonal Jacobian helps stability. The implementation reshapes the
target weight to a 2-D matrix ``(shape[0], prod(shape[1:]))``, draws an
i.i.d. Gaussian, runs ``np.linalg.qr``, and then sign-corrects the
diagonal of ``R`` so the result is uniformly distributed over the
orthogonal/Stiefel manifold (Mezzadri, 2007).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from ..core._typing import Array, DType, Initializer, PRNGKey, Shape


def orthogonal(gain: float = 1.0) -> Initializer:
    """Return an orthogonal initializer (Saxe et al., 2014).

    For 2-D-or-higher shapes the initializer:

    1. Reshapes the target weight to ``(shape[0], prod(shape[1:]))``.
    2. Draws an i.i.d. Gaussian matrix in ``float32`` and runs a QR
       decomposition (transposing first if it is wider than it is tall,
       which keeps QR cheap).
    3. Multiplies ``Q`` by ``sign(diag(R))`` so the distribution is
       uniform over the orthogonal manifold.
    4. Restores the original shape and casts to ``dtype``, scaling by
       ``gain``.

    For ranks below 2 (where "orthogonal" is not well-defined) it falls
    back to scaled :func:`jax.random.normal` noise.

    The QR runs in NumPy on the host (``np.linalg.qr``), so the output
    is a host array; the SpectraX variable system handles the device
    placement when the value is assigned to a :class:`Parameter`.

    Args:
        gain: Scalar multiplier applied to the resulting orthonormal
            matrix. Defaults to ``1.0``.

    Returns:
        An :class:`~spectrax.typing.Initializer` returning the
        orthogonal weight of the requested shape and dtype.
    """

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Materialize an orthogonal (or, for low rank, scaled-normal) weight of ``shape``.

        Args:
            key: Logical key, path segment, or PRNG key used by the operation.
            shape: Array shape requested by the initializer or helper.
            dtype: Array dtype requested for the produced value.

        Returns:
            Result described by this helper.
        """
        if len(shape) < 2:
            return jax.random.normal(key, shape, dtype=dtype) * gain
        flat: tuple[int, int] = (shape[0], math.prod(shape[1:]))
        a = np.asarray(jax.random.normal(key, flat, dtype=jnp.float32))
        q, r = np.linalg.qr(a if flat[0] >= flat[1] else a.T)
        d = np.sign(np.diag(r))
        q = q * d
        if flat[0] < flat[1]:
            q = q.T
        return jnp.asarray(np.asarray(gain * q.reshape(shape), dtype=np.dtype(dtype)))

    return init
