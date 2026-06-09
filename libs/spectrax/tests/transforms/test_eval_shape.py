# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :mod:`spectrax.transforms.eval_shape`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax.core.module import Module
from spectrax.core.variable import Buffer
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


class _Accum(Module):
    """Tiny module with one mutable buffer for shape-eval tests."""

    acc: Buffer

    def __init__(self) -> None:
        """Initialize with acc."""
        super().__init__()
        self.acc = Buffer(jnp.zeros((), dtype=jnp.float32), kind="batch_stats")

    def forward(self, x):
        """Run the forward pass."""
        self.acc.value = self.acc.value + 1.0
        return x + self.acc.value


def test_eval_shape_accepts_module_positional_arg():
    """`spx.eval_shape` accepts live modules as positional inputs."""
    m = Linear(4, 3, rngs=Rngs(0))
    x = jax.ShapeDtypeStruct((2, 4), jnp.float32)

    out = spx.eval_shape(lambda mod, x: mod(x), m, x)

    assert isinstance(out, jax.ShapeDtypeStruct)
    assert out.shape == (2, 3)
    assert out.dtype == jnp.float32


def test_eval_shape_accepts_module_keyword_arg():
    """Keyword-positioned module arguments are split and rebound too."""
    m = Linear(4, 3, rngs=Rngs(0))
    x = jax.ShapeDtypeStruct((2, 4), jnp.float32)

    def fn(x, *, mod):
        """Helper function."""
        return mod(x)

    out = spx.eval_shape(fn, x, mod=m)

    assert isinstance(out, jax.ShapeDtypeStruct)
    assert out.shape == (2, 3)


def test_eval_shape_does_not_write_back_mutations():
    """Abstract variable writes stay local to the shape-eval trace."""
    m = _Accum()
    x = jax.ShapeDtypeStruct((), jnp.float32)

    out = spx.eval_shape(lambda mod, x: mod(x), m, x)

    assert isinstance(out, jax.ShapeDtypeStruct)
    assert out.shape == ()
    assert float(m.acc.value) == 0.0


def test_eval_shape_can_return_abstract_module():
    """Model construction under `spx.eval_shape` yields abstract leaves."""
    abs_model = spx.eval_shape(lambda: Linear(4, 3, rngs=Rngs(0)))

    assert isinstance(abs_model, Linear)
    _gdef, state = spx.export(abs_model)
    weight = state["parameters"]["weight"]
    bias = state["parameters"]["bias"]
    assert isinstance(weight, jax.ShapeDtypeStruct)
    assert weight.shape == (4, 3)
    assert isinstance(bias, jax.ShapeDtypeStruct)
    assert bias.shape == (3,)
