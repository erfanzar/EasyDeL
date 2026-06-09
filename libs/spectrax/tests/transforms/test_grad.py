# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.transforms.grad`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import spectrax as spx
from spectrax.core.module import Module
from spectrax.core.variable import Buffer
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


def _loss_fn(m, x):
    """Mean-squared-error loss used by multiple tests."""
    return jnp.mean(m(x) ** 2)


class _Accum(Module):
    """Tiny module with one mutable buffer for autodiff mutation tests."""

    acc: Buffer

    def __init__(self) -> None:
        """Initialize with acc."""
        super().__init__()
        self.acc = Buffer(jnp.zeros((), dtype=jnp.float32), kind="batch_stats")

    def forward(self, x):
        """Run the forward pass."""
        return x + self.acc.value


def test_grad_returns_state_matching_parameters_shape():
    """``spx.grad`` returns a :class:`State` shaped like ``parameters``."""
    m = Linear(4, 4, rngs=Rngs(0))
    x = jnp.ones((2, 4))
    grads = spx.grad(_loss_fn)(m, x)
    assert isinstance(grads, spx.State)
    for path, val in grads["parameters"].items():
        assert val.shape == spx.export(m)[1]["parameters"][path].shape


def test_value_and_grad_returns_pair():
    """``value_and_grad`` returns ``(loss, grads)``."""
    m = Linear(4, 4, rngs=Rngs(0))
    x = jnp.ones((2, 4))
    loss, grads = spx.value_and_grad(_loss_fn)(m, x)
    assert jnp.ndim(loss) == 0
    assert isinstance(grads, spx.State)


def test_grad_raises_without_module_arg():
    """``spx.grad`` requires at least one :class:`Module` argument."""

    def pure_loss(x):
        """Compute the loss."""
        return jnp.sum(x)

    with pytest.raises(TypeError):
        spx.grad(pure_loss)(jnp.ones(3))


def test_grad_argnum_selects_module():
    """``argnum=`` lets the user pick which argument is the differentiated module."""

    def loss(x, m):
        """Compute the loss."""
        return jnp.mean(m(x) ** 2)

    m = Linear(4, 4, rngs=Rngs(0))
    grads = spx.grad(loss, argnum=1)(jnp.ones((2, 4)), m)
    assert "parameters" in grads.collections()


def test_grad_has_aux_returns_aux_tuple():
    """With ``has_aux=True`` the inner fn returns ``(loss, aux)``."""

    def loss(m, x):
        """Compute the loss."""
        y = m(x)
        return jnp.mean(y**2), y.shape

    m = Linear(4, 4, rngs=Rngs(0))
    (_value, aux), grads = spx.value_and_grad(loss, has_aux=True)(m, jnp.ones((2, 4)))
    assert aux == (2, 4)
    assert "parameters" in grads.collections()


def test_grad_has_aux_on_grad_returns_aux_with_grads():
    """``grad(..., has_aux=True)`` returns ``(grads, aux)``."""

    def loss(m, x):
        """Compute and return the jnp.mean(m(x) ** 2), "tag" loss."""
        return jnp.mean(m(x) ** 2), "tag"

    m = Linear(4, 4, rngs=Rngs(0))
    grads, aux = spx.grad(loss, has_aux=True)(m, jnp.ones((2, 4)))
    assert aux == "tag"
    assert "parameters" in grads.collections()


def test_grad_wrt_selector_narrows_to_weight_only():
    """A selector-based ``wrt`` restricts the gradient collection."""
    m = Linear(4, 4, rngs=Rngs(0))
    sel = spx.select().where_variable(lambda v, p: p.endswith("weight"))
    grads = spx.grad(_loss_fn, wrt=sel)(m, jnp.ones((2, 4)))
    for _c, d in grads.raw().items():
        for p in d:
            assert p.endswith("weight")


def test_grad_decorator_form():
    """Both inline and decorator forms are supported."""

    @spx.grad
    def loss(m, x):
        """Compute the loss."""
        return jnp.mean(m(x) ** 2)

    grads = loss(Linear(4, 4, rngs=Rngs(0)), jnp.ones((2, 4)))
    assert "parameters" in grads.collections()


def test_vjp_returns_pullback_with_state_for_module():
    """The pullback returns a ``State`` cotangent for module primals."""
    m = Linear(4, 4, rngs=Rngs(0))
    x = jnp.ones((2, 4))

    y, pullback = spx.vjp(lambda m, x: m(x), m, x)
    grads_m, grads_x = pullback(jnp.ones_like(y))

    assert isinstance(grads_m, spx.State)
    assert "parameters" in grads_m.collections()
    assert grads_x.shape == x.shape


def test_vjp_has_aux_returns_aux():
    """``has_aux=True`` returns ``(out, pullback, aux)``."""
    m = Linear(4, 4, rngs=Rngs(0))
    x = jnp.ones((2, 4))

    def fn(mod, x):
        """Helper function."""
        y = mod(x)
        return y.sum(), y.shape

    out, pullback, aux = spx.vjp(fn, m, x, has_aux=True)
    grads_m, grads_x = pullback(jnp.array(1.0, dtype=out.dtype))

    assert aux == (2, 4)
    assert isinstance(grads_m, spx.State)
    assert grads_x.shape == x.shape


def test_vjp_mutable_primal_updates_live_module():
    """Primal-state updates are written back when declared mutable."""
    m = _Accum()
    x = jnp.array(3.0, dtype=jnp.float32)

    def fn(mod, x):
        """Helper function."""
        mod.acc.value = mod.acc.value + 2.0
        return mod(x)

    out, pullback = spx.vjp(fn, m, x, mutable="batch_stats")
    grads_m, grads_x = pullback(jnp.array(1.0, dtype=out.dtype))

    assert float(m.acc.value) == 2.0
    assert isinstance(grads_m, spx.State)
    assert grads_x.shape == ()


def test_vjp_decorator_rejects_kwargs():
    """Wrapped ``vjp`` keeps a JAX-like positional-arguments API."""

    @spx.vjp
    def fn(mod, x):
        """Helper function."""
        return mod(x)

    m = Linear(4, 4, rngs=Rngs(0))
    with pytest.raises(TypeError, match="does not support keyword arguments"):
        fn(m=m, x=jnp.ones((2, 4)))


def test_jvp_accepts_module_tangent():
    """Module tangents may be supplied as another module pytree."""
    m = Linear(4, 4, rngs=Rngs(0))
    x = jnp.ones((2, 4))
    m_tangent = jax.tree.map(jnp.zeros_like, m)
    x_tangent = jnp.ones_like(x)

    out, tangent_out = spx.jvp(lambda mod, x: mod(x), (m, x), (m_tangent, x_tangent))

    assert out.shape == tangent_out.shape == (2, 4)


def test_jvp_has_aux_returns_aux():
    """``spx.jvp(..., has_aux=True)`` returns ``(out, tangent_out, aux)``."""
    m = Linear(4, 4, rngs=Rngs(0))
    x = jnp.ones((2, 4))
    m_tangent = jax.tree.map(jnp.zeros_like, m)
    x_tangent = jnp.ones_like(x)

    def fn(mod, x):
        """Helper function."""
        y = mod(x)
        return y.sum(), y.shape

    out, tangent_out, aux = spx.jvp(fn, (m, x), (m_tangent, x_tangent), has_aux=True)
    assert jnp.ndim(out) == 0
    assert jnp.ndim(tangent_out) == 0
    assert aux == (2, 4)


def test_jvp_mutable_primal_updates_live_module():
    """Primal-state updates are written back when declared mutable."""
    m = _Accum()

    def fn(mod, x):
        """Helper function."""
        mod.acc.value = mod.acc.value + x
        return mod(x)

    out, tangent_out = spx.jvp(
        fn,
        (m, jnp.array(2.0, dtype=jnp.float32)),
        (spx.export(_Accum())[1], jnp.array(1.0, dtype=jnp.float32)),
        mutable="batch_stats",
    )
    assert float(m.acc.value) == 2.0
    assert jnp.shape(out) == ()
    assert jnp.shape(tangent_out) == ()
