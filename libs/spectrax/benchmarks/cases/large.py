# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Large-model benchmarks (~1B-param stacked transformer).

Gated by --cases large so it doesn't run in CPU smoke tests. Designed
for TPU/GPU; on CPU you can still run it but compile and step times
will be dominated by the CPU backend.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
from flax import nnx

import spectrax as spx

from .. import models


def build():
    """Build large-model benchmark cases (~1B-param transformer).

    Returns:
        Dictionary mapping case name to ``(spectrax_fn, nnx_fn)`` pairs.
    """
    cases: dict[str, tuple[Callable, Callable]] = {}

    spx_mdl, x = models.spx_transformer_1b()
    nnx_mdl, _ = models.nnx_transformer_1b()

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

    cases["xfmr_1b/forward"] = (
        lambda: jax.block_until_ready(spx_fwd(spx_mdl, x)),
        lambda: jax.block_until_ready(nnx_fwd(nnx_mdl, x)),
    )

    @spx.jit
    def spx_step(m, x):
        """Jitted spectrax training step."""

        def loss(m, x):
            """Scalar loss: sum of model output."""
            return m(x).sum()

        return spx.grad(loss)(m, x)

    @nnx.jit
    def nnx_step(m, x):
        """Jitted nnx training step."""

        def loss(m, x):
            """Scalar loss: sum of model output."""
            return m(x).sum()

        return nnx.grad(loss)(m, x)

    jax.block_until_ready(jax.tree.leaves(spx_step(spx_mdl, x))[0])
    jax.block_until_ready(jax.tree.leaves(nnx_step(nnx_mdl, x))[0])

    cases["xfmr_1b/train_step"] = (
        (lambda m=spx_mdl, x=x, f=spx_step: jax.block_until_ready(jax.tree.leaves(f(m, x))[0])),
        (lambda m=nnx_mdl, x=x, f=nnx_step: jax.block_until_ready(jax.tree.leaves(f(m, x))[0])),
    )

    spx_fp8_mdl, x_fp8 = models.spx_fp8_transformer_1b()

    @spx.jit(mutable="fp8_meta")
    def spx_fp8_fwd(m, x):
        """Jitted FP8 forward pass."""
        return m(x)

    @spx.jit(mutable="fp8_meta")
    def spx_fp8_step(m, x):
        """Jitted FP8 training step."""

        def loss(m, x):
            """Scalar loss: sum of model output."""
            return m(x).sum()

        return spx.grad(loss)(m, x)

    jax.block_until_ready(spx_fp8_fwd(spx_fp8_mdl, x_fp8))
    jax.block_until_ready(jax.tree.leaves(spx_fp8_step(spx_fp8_mdl, x_fp8))[0])

    cases["xfmr_1b_spx_fp8_vs_nnx_bf16/forward"] = (
        lambda: jax.block_until_ready(spx_fp8_fwd(spx_fp8_mdl, x_fp8)),
        lambda: jax.block_until_ready(nnx_fwd(nnx_mdl, x)),
    )
    cases["xfmr_1b_spx_fp8_vs_nnx_bf16/train_step"] = (
        (lambda m=spx_fp8_mdl, x=x_fp8, f=spx_fp8_step: jax.block_until_ready(jax.tree.leaves(f(m, x))[0])),
        (lambda m=nnx_mdl, x=x, f=nnx_step: jax.block_until_ready(jax.tree.leaves(f(m, x))[0])),
    )

    return cases
