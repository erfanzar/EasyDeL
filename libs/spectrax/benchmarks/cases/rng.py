# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""RNG / split-merge pressure with a dropout-heavy MLP."""

from __future__ import annotations

from collections.abc import Callable

import jax
from flax import nnx

import spectrax as spx

from .. import models


def build():
    """Build RNG / split-merge pressure benchmark cases.

    Returns:
        Dictionary mapping case name to ``(spectrax_fn, nnx_fn)`` pairs.
    """
    cases: dict[str, tuple[Callable, Callable]] = {}

    spx_mdl, x = models.spx_dropout_mlp()
    nnx_mdl, _ = models.nnx_dropout_mlp()

    rngs_spx = spx.Rngs(42)

    @spx.jit
    def spx_fwd(m, rngs, x):
        """Jitted spectrax forward with explicit RNGs."""
        return m(x, rngs=rngs)

    @nnx.jit
    def nnx_fwd(m, x):
        """Jitted nnx forward."""
        return m(x)

    jax.block_until_ready(spx_fwd(spx_mdl, rngs_spx, x))
    jax.block_until_ready(nnx_fwd(nnx_mdl, x))

    cases["rng/dropout_mlp"] = (
        lambda: jax.block_until_ready(spx_fwd(spx_mdl, rngs_spx, x)),
        lambda: jax.block_until_ready(nnx_fwd(nnx_mdl, x)),
    )

    return cases
