# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Convert a FP32 :class:`spx.nn.Linear` to a :class:`spx.nn.Fp8Linear`.

Allocates a fresh :class:`Fp8Linear` of the same shape, copies the
FP32 weight/bias over with :func:`spx.update`, and inspects the
six :class:`spx.nn.Fp8Meta` scale / amax-history cells owned by the
embedded :class:`spx.nn.Fp8DotGeneral`. A single forward+backward
through :func:`spx.jit` refreshes the meta tensors in place.

Run::

    python -m examples.04_surgery.03_fp8_cast
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax import nn
from spectrax.core.state import State


def build_fp8_clone(fp32: nn.Linear, rngs: spx.Rngs) -> nn.Fp8Linear:
    """Allocate an :class:`Fp8Linear` with shape matching ``fp32``."""
    return nn.Fp8Linear(fp32.in_features, fp32.out_features, rngs=rngs)


def copy_weights(src: nn.Linear, dst: nn.Fp8Linear) -> None:
    """Copy ``weight`` / ``bias`` from ``src`` into ``dst`` via :func:`spx.update`."""
    _, src_state = spx.export(src)
    patch = State()
    patch = patch.set("parameters", "weight", src_state.get("parameters", "weight"))
    patch = patch.set("parameters", "bias", src_state.get("parameters", "bias"))
    spx.update(dst, patch)


def show_fp8_meta(module: spx.Module, tag: str) -> None:
    """Print every :class:`Fp8Meta` cell with its scale / amax summary."""
    print(f"--- {tag} ---")
    for path, var in spx.iter_variables(module, select=spx.of_type(nn.Fp8Meta)):
        v = var.value
        stat = f"val={float(v[0]):.4f}" if v.shape == (1,) else f"amax_max={float(jnp.max(v)):.4f}"
        print(f"  {path:40s} shape={v.shape!s:10s} {stat}")


def main():
    """FP32 -> FP8 cast and demonstrate meta updates under jit."""
    fp32 = nn.Linear(8, 4, rngs=spx.Rngs(0))
    fp8 = build_fp8_clone(fp32, rngs=spx.Rngs(1))
    copy_weights(fp32, fp8)

    show_fp8_meta(fp8, "freshly constructed Fp8Linear")

    x = jax.random.normal(jax.random.PRNGKey(2), (4, 8))

    @spx.jit(mutable="fp8_meta")
    def step(m, x):
        """Forward pass whose backward refreshes the fp8_meta collection."""
        return jnp.sum(m(x) ** 2)

    loss, _grads = jax.value_and_grad(lambda m: step(m, x))(fp8)
    jax.block_until_ready(loss)
    print(f"fp8 forward loss: {float(loss):.4f}")
    show_fp8_meta(fp8, "after one forward (amax history populated)")

    y_fp32 = fp32(x)
    y_fp8 = fp8(x)
    print(f"y shapes: fp32={y_fp32.shape} fp8={y_fp8.shape}")
    print(f"max abs diff (expected nonzero from fp8 qdq): {float(jnp.max(jnp.abs(y_fp32 - y_fp8))):.4f}")


if __name__ == "__main__":
    main()
