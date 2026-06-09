# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end training step: forward + loss + grad + optimizer update."""

from __future__ import annotations

from collections.abc import Callable

import jax
from flax import nnx

import spectrax as spx

from .. import models


def _sgd_update_spx(state, grads, lr: float = 0.01):
    """Apply an SGD update to a spectrax :class:`State`.

    Args:
        state: Current parameter state.
        grads: Gradient state from :func:`spectrax.grad`.
        lr: Learning rate.

    Returns:
        Updated :class:`State`.
    """
    from spectrax import State

    new = {}
    for c, d in state.raw().items():
        nd = {}
        for p, v in d.items():
            g = grads.get(c, p) if hasattr(grads, "get") else None
            if g is not None:
                nd[p] = v - lr * g
            else:
                nd[p] = v
        new[c] = nd
    return State(new)


def build():
    """Build end-to-end training-step benchmark cases for spectrax and nnx.

    Returns:
        Dictionary mapping case name to ``(spectrax_fn, nnx_fn)`` pairs.
    """
    cases: dict[str, tuple[Callable, Callable]] = {}

    for name, factory in [("mlp12x1024", "mlp"), ("xfmr_d512", "transformer"), ("conv", "conv")]:
        spx_mdl, x = getattr(models, f"spx_{factory}")()
        nnx_mdl, _ = getattr(models, f"nnx_{factory}")()

        @spx.jit
        def spx_step(m, x):
            """Jitted spectrax training step: compute grads of ``m(x).sum()``.

            Args:
                m: spectrax module.
                x: Input tensor.

            Returns:
                Gradient state.
            """

            def loss_fn(m, x):
                """Scalar loss: sum of model output."""
                return m(x).sum()

            grads = spx.grad(loss_fn)(m, x)
            return grads

        @nnx.jit
        def nnx_step(m, x):
            """Jitted nnx training step: compute grads of ``m(x).sum()``.

            Args:
                m: nnx module.
                x: Input tensor.

            Returns:
                Gradient state.
            """

            def loss_fn(m, x):
                """Scalar loss: sum of model output."""
                return m(x).sum()

            grads = nnx.grad(loss_fn)(m, x)
            return grads

        jax.block_until_ready(jax.tree.leaves(spx_step(spx_mdl, x))[0])
        jax.block_until_ready(jax.tree.leaves(nnx_step(nnx_mdl, x))[0])

        cases[f"train_step/{name}"] = (
            (lambda m=spx_mdl, x=x, f=spx_step: jax.block_until_ready(jax.tree.leaves(f(m, x))[0])),
            (lambda m=nnx_mdl, x=x, f=nnx_step: jax.block_until_ready(jax.tree.leaves(f(m, x))[0])),
        )

    return cases
