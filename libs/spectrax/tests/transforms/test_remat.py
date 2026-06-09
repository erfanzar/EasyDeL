# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.transforms.remat`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import spectrax as spx
from spectrax.core.module import Module
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


def test_remat_forward_equivalent_to_non_remat():
    """Remat preserves forward output."""
    m = Linear(4, 4, rngs=Rngs(0))
    x = jnp.ones((2, 4))
    eager = m(x)
    out = spx.remat(lambda m, x: m(x))(m, x)
    assert jnp.allclose(eager, out)


def test_remat_gradient_matches_non_remat():
    """Remat preserves gradient values (up to numerical tolerance)."""

    def plain_loss(m, x):
        """Loss without rematerialization."""
        return jnp.mean(m(x) ** 2)

    def remat_loss(m, x):
        """Same loss with the forward inside :func:`spectrax.remat`."""
        y = spx.remat(lambda m, x: m(x))(m, x)
        return jnp.mean(y**2)

    m = Linear(3, 3, rngs=Rngs(0))
    x = jnp.ones((2, 3))
    g_plain = spx.grad(plain_loss)(m, x)
    g_remat = spx.grad(remat_loss)(m, x)
    for path in g_plain["parameters"]:
        assert jnp.allclose(g_plain["parameters"][path], g_remat["parameters"][path], atol=1e-5)


def test_remat_decorator_form():
    """Factory-style decorator also works."""

    @spx.remat
    def fn(m, x):
        """Helper function."""
        return m(x)

    m = Linear(4, 4, rngs=Rngs(0))
    out = fn(m, jnp.ones((2, 4)))
    assert out.shape == (2, 4)


def test_remat_rejects_sequence_prevent_cse():
    """The public type matches JAX: ``prevent_cse`` is a single bool."""
    with pytest.raises(TypeError, match="prevent_cse"):
        spx.remat(lambda x: x, prevent_cse=[True, False])


def test_remat_module_class_accepts_unhashable_mutable_selector():
    """Class remat cache keys normalize list/dict selectors before hashing."""

    class Block(Module):
        """Fixture block module for testing."""

        def __init__(self):
            """Initialize with fc."""
            super().__init__()
            self.fc = Linear(4, 4, rngs=Rngs(0))

        def forward(self, x):
            """Run the forward pass."""
            return self.fc(x)

    remat_block = spx.remat(Block, mutable=["buffers"])

    assert issubclass(remat_block, Block)


class StatefulRematModule(Module):
    """Module that uses string/boolean flags in its forward pass."""

    def __init__(self):
        """Initialize with fc."""
        super().__init__()
        self.fc = Linear(4, 4, rngs=Rngs(0))

    def forward(self, x, *, mode: str = "train", output_attentions: bool = False):
        """Run the forward pass."""
        y = self.fc(x)
        if mode == "eval":
            y = y * 0.5
        if output_attentions:
            return y, jnp.ones_like(y)
        return y


def test_remat_auto_static_kwargs():
    """String/bool kwargs are automatically treated as static by remat."""
    m = StatefulRematModule()

    @spx.remat
    def fn(model, x, *, mode, output_attentions):
        """Helper function."""
        return model(x, mode=mode, output_attentions=output_attentions)

    x = jnp.ones((2, 4))
    out = fn(m, x, mode="eval", output_attentions=False)
    expected = m(x, mode="eval", output_attentions=False)
    assert jnp.allclose(out, expected)


def test_remat_gradient_with_auto_static_kwargs():
    """Gradients through remat with auto-static string kwargs are correct."""
    m = StatefulRematModule()

    @spx.remat
    def loss(model, x, *, mode):
        """Compute the loss."""
        return jnp.mean(model(x, mode=mode) ** 2)

    x = jnp.ones((2, 4))
    g = spx.grad(loss)(m, x, mode="train")
    assert "parameters" in g
    assert "fc" in g["parameters"]


def test_remat_module_class_with_auto_static_kwargs():
    """Wrapping a module class whose forward takes string kwargs works."""

    class Block(Module):
        """Fixture block module for testing."""

        def __init__(self):
            """Initialize with fc."""
            super().__init__()
            self.fc = Linear(4, 4, rngs=Rngs(0))

        def forward(self, x, *, mode: str = "train"):
            """Run the forward pass."""
            return self.fc(x) * (0.5 if mode == "eval" else 1.0)

    RematBlock = spx.remat(Block)
    b = RematBlock()
    x = jnp.ones((2, 4))
    out = b(x, mode="eval")
    cached = getattr(b, "_spx_remat_forward", None)
    out_again = b(x, mode="eval")
    assert out.shape == (2, 4)
    assert out_again.shape == (2, 4)
    assert getattr(b, "_spx_remat_forward", None) is cached
