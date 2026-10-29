# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inverted-dropout primitive.

Implements training-time dropout where the kept activations are scaled
by ``1 / (1 - rate)`` so that the expectation matches the no-dropout
case. The "deterministic" toggle (used by inference paths) returns the
input unchanged. The PRNG key is required at the call site rather than
being implicitly drawn from a default stream — :class:`spectrax.nn.Dropout`
wraps this primitive and pulls a key from the active :class:`Rngs`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, PRNGKey


def dropout(
    x: ArrayLike,
    rate: float,
    *,
    key: PRNGKey | None = None,
    deterministic: bool = False,
) -> Array:
    """Apply inverted dropout.

    With probability ``rate`` each element is zeroed; the rest are
    scaled by ``1 / (1 - rate)`` so expectations are preserved.

    Args:
        x: Input tensor.
        rate: Dropout probability in ``[0, 1)``.
        key: Required PRNG key when ``deterministic`` is ``False`` and
            ``rate`` is positive.
        deterministic: When ``True``, return ``x`` unchanged.

    Returns:
        The (optionally) dropped-out tensor.

    Raises:
        ValueError: If a key is needed but not supplied.
    """
    xa = jnp.asarray(x)
    if deterministic or rate == 0.0:
        return xa
    if key is None:
        raise ValueError("dropout requires a PRNG key when not deterministic")
    keep_rate = 1.0 - rate
    mask = jax.random.bernoulli(key, keep_rate, shape=xa.shape)
    return jnp.where(mask, xa / keep_rate, jnp.zeros_like(xa))
