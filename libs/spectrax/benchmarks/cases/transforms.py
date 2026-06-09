# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Transform wiring overhead: jit dispatch, grad, vmap, scan, remat."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

import spectrax as spx

from .. import models


def build():
    """Build transform wiring overhead benchmark cases.

    Returns:
        Dictionary mapping case name to ``(spectrax_fn, nnx_fn)`` pairs.
    """
    cases: dict[str, tuple[Callable, Callable]] = {}

    spx_mdl, x = models.spx_mlp()
    nnx_mdl, _ = models.nnx_mlp()

    @spx.jit
    def spx_fwd(m, x):
        """Jitted spectrax forward pass."""
        return m(x)

    @nnx.jit
    def nnx_fwd(m, x):
        """Jitted nnx forward pass."""
        return m(x)

    jax.block_until_ready(spx_fwd(spx_mdl, x))
    jax.block_until_ready(nnx_fwd(nnx_mdl, x))

    cases["jit_dispatch/mlp"] = (
        lambda: jax.block_until_ready(spx_fwd(spx_mdl, x)),
        lambda: jax.block_until_ready(nnx_fwd(nnx_mdl, x)),
    )

    def spx_loss(m, x):
        """Scalar loss for spectrax: sum of model output."""
        return m(x).sum()

    def nnx_loss(m, x):
        """Scalar loss for nnx: sum of model output."""
        return m(x).sum()

    spx_grad = spx.grad(spx_loss)
    nnx_grad = nnx.grad(nnx_loss)
    jax.block_until_ready(jax.tree.leaves(spx_grad(spx_mdl, x))[0])
    jax.block_until_ready(jax.tree.leaves(nnx_grad(nnx_mdl, x))[0])
    cases["grad/mlp"] = (
        lambda: jax.block_until_ready(jax.tree.leaves(spx_grad(spx_mdl, x))[0]),
        lambda: jax.block_until_ready(jax.tree.leaves(nnx_grad(nnx_mdl, x))[0]),
    )

    spx_vg = spx.value_and_grad(spx_loss)
    nnx_vg = nnx.value_and_grad(nnx_loss)
    jax.block_until_ready(spx_vg(spx_mdl, x)[0])
    jax.block_until_ready(nnx_vg(nnx_mdl, x)[0])
    cases["value_and_grad/mlp"] = (
        lambda: jax.block_until_ready(spx_vg(spx_mdl, x)[0]),
        lambda: jax.block_until_ready(nnx_vg(nnx_mdl, x)[0]),
    )

    x[0]

    spx.vmap(lambda m, x: m(x[None])[0], in_axes=(None, 0) if False else 0)
    x_big = jnp.ones((64, 1024), dtype=jnp.float32)

    @spx.jit
    def spx_batched(m, x):
        """Jitted spectrax batched forward via vmap."""
        return jax.vmap(m)(x[:, None, :])[:, 0]

    @nnx.jit
    def nnx_batched(m, x):
        """Jitted nnx batched forward via vmap."""
        return jax.vmap(m)(x[:, None, :])[:, 0]

    try:
        jax.block_until_ready(spx_batched(spx_mdl, x_big))
        jax.block_until_ready(nnx_batched(nnx_mdl, x_big))
        cases["vmap/mlp"] = (
            lambda: jax.block_until_ready(spx_batched(spx_mdl, x_big)),
            lambda: jax.block_until_ready(nnx_batched(nnx_mdl, x_big)),
        )
    except Exception:
        pass

    return cases
