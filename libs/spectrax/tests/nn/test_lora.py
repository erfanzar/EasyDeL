# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :mod:`spectrax.nn.lora` — Low-Rank Adaptation layers.

Exercises :class:`~spectrax.nn.LoRA`, :class:`~spectrax.nn.LoRALinear`,
:class:`~spectrax.nn.LoraParameter`, and the :func:`~spectrax.nn.wrap_lora`
helper. Verifies shape, zero-init, alpha scaling, state-collection
layout, export/bind round-trip, base-module composition, and gradient
isolation via ``wrt="lora"``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import spectrax as spx
import spectrax.nn as spx_nn


def test_lora_shapes_and_zero_init():
    """Factor shapes are ``(d_in, rank)`` / ``(rank, d_out)`` and B starts at zero.

    With ``lora_b`` initialized to zero the delta ``x @ A @ B`` is
    exactly zero on the first call — the canonical LoRA property that
    lets a wrapped pretrained model match its pre-adapter behaviour
    before training begins.
    """
    lora = spx_nn.LoRA(d_in=8, rank=4, d_out=16, rngs=spx.Rngs(0))
    assert lora.lora_a.value.shape == (8, 4)
    assert lora.lora_b.value.shape == (4, 16)
    x = jnp.ones((2, 8), dtype=jnp.float32)
    y = lora(x)
    assert y.shape == (2, 16)
    assert jnp.allclose(y, 0.0)


def test_lora_delta_equals_x_A_B():
    """After overwriting A and B with non-trivial values, output == ``(x @ A) @ B``."""
    lora = spx_nn.LoRA(8, 4, 16, rngs=spx.Rngs(0))
    A = jax.random.normal(jax.random.PRNGKey(1), (8, 4), dtype=jnp.float32)
    B = jax.random.normal(jax.random.PRNGKey(2), (4, 16), dtype=jnp.float32)
    lora.lora_a.value = A
    lora.lora_b.value = B
    x = jax.random.normal(jax.random.PRNGKey(3), (3, 8), dtype=jnp.float32)
    y = lora(x)
    assert jnp.allclose(y, (x @ A) @ B, atol=1e-5)


def test_lora_alpha_scaling():
    """``alpha`` applies a ``delta * (alpha / rank)`` scaling.

    With rank=4, alpha=16, and all-ones factors, the un-scaled row
    value is ``sum_over_rank(x @ A @ B) = 8 * 4 * 1 = 32``; scaling
    by ``16 / 4 = 4`` yields the expected 128.
    """
    rank = 4
    alpha = 16.0
    lora = spx_nn.LoRA(8, rank, 16, alpha=alpha, rngs=spx.Rngs(0))
    lora.lora_a.value = jnp.ones((8, rank), dtype=jnp.float32)
    lora.lora_b.value = jnp.ones((rank, 16), dtype=jnp.float32)
    x = jnp.ones((2, 8), dtype=jnp.float32)
    y = lora(x)
    assert float(y[0, 0]) == pytest.approx(128.0)


def test_lora_rank_must_be_positive():
    """Rank ≤ 0 is rejected at construction with :class:`ValueError`."""
    with pytest.raises(ValueError, match="rank must be positive"):
        spx_nn.LoRA(8, 0, 16, rngs=spx.Rngs(0))
    with pytest.raises(ValueError, match="rank must be positive"):
        spx_nn.LoRA(8, -2, 16, rngs=spx.Rngs(0))


def test_lora_weights_live_in_lora_collection():
    """Both factors land in the ``"lora"`` collection; no ``"parameters"`` leaks."""
    lora = spx_nn.LoRA(8, 4, 16, rngs=spx.Rngs(0))
    _, state = spx.export(lora)
    assert "lora" in state.raw()
    assert set(state.raw()["lora"].keys()) == {"lora_a", "lora_b"}
    assert "parameters" not in state.raw() or not state.raw()["parameters"]


def test_lora_export_bind_roundtrip():
    """Exporting and re-binding preserves both factor tensors byte-for-byte."""
    lora = spx_nn.LoRA(8, 4, 16, alpha=8.0, rngs=spx.Rngs(0))
    lora.lora_a.value = lora.lora_a.value + 1.0
    gdef, state = spx.export(lora)
    rebuilt = spx.bind(gdef, state)
    assert isinstance(rebuilt, spx_nn.LoRA)
    assert jnp.array_equal(rebuilt.lora_a.value, lora.lora_a.value)
    assert jnp.array_equal(rebuilt.lora_b.value, lora.lora_b.value)


def test_lora_with_base_module_adds_delta():
    """``LoRA(base_module=base)`` returns ``base(x) + delta(x)`` (delta=0 at init)."""
    base = spx_nn.Linear(8, 16, rngs=spx.Rngs(1))
    lora = spx_nn.LoRA(8, 4, 16, base_module=base, rngs=spx.Rngs(2))
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 8), dtype=jnp.float32)
    assert jnp.allclose(lora(x), base(x), atol=1e-6)


def test_wrap_lora_reads_in_out_features():
    """``wrap_lora`` reads ``in_features`` / ``out_features`` off the base layer."""
    base = spx_nn.Linear(32, 64, rngs=spx.Rngs(0))
    wrapper = spx_nn.wrap_lora(base, rank=8, rngs=spx.Rngs(1))
    assert wrapper.d_in == 32
    assert wrapper.d_out == 64
    assert wrapper.rank == 8
    x = jnp.ones((2, 32), dtype=jnp.float32)
    assert jnp.allclose(wrapper(x), base(x), atol=1e-6)


def test_wrap_lora_rejects_base_without_in_out_features():
    """Bases lacking ``in_features`` / ``out_features`` raise a clear TypeError."""

    class Thing(spx.Module):
        """Minimal module missing the feature-count attributes."""

        def forward(self, x):
            """Identity forward used only to satisfy the :class:`Module` contract."""
            return x

    with pytest.raises(TypeError, match="in_features/out_features"):
        spx_nn.wrap_lora(Thing(), rank=4, rngs=spx.Rngs(0))


def test_grad_trains_only_lora_not_base():
    """``spx.grad(wrt="lora")`` returns gradients for the adapter only."""
    base = spx_nn.Linear(8, 16, rngs=spx.Rngs(1))
    lora = spx_nn.LoRA(8, 4, 16, base_module=base, rngs=spx.Rngs(2))
    x = jnp.ones((2, 8), dtype=jnp.float32)

    @spx.jit
    def step(m, x):
        """Differentiate a sum-of-squares loss w.r.t. the ``"lora"`` collection only."""

        def loss(m, x):
            """Sum-of-squares scalar loss."""
            return (m(x) ** 2).sum()

        return spx.grad(loss, wrt="lora")(m, x)

    grads = step(lora, x)
    raw = grads.raw()
    assert "lora" in raw
    assert set(raw["lora"].keys()) == {"lora_a", "lora_b"}
    assert "parameters" not in raw or not raw["parameters"]


def test_lora_linear_shape_and_state_layout():
    """:class:`LoRALinear` places base under ``base.*`` and adapter under ``lora.*``."""
    ll = spx_nn.LoRALinear(8, 16, rank=4, rngs=spx.Rngs(0))
    x = jnp.ones((2, 8), dtype=jnp.float32)
    y = ll(x)
    assert y.shape == (2, 16)
    _, state = spx.export(ll)
    assert "parameters" in state.raw()
    assert "lora" in state.raw()
    params = set(state.raw()["parameters"].keys())
    lora_paths = set(state.raw()["lora"].keys())
    assert params == {"base"}
    assert state.raw()["parameters"]["base"]["weight"] is not None
    assert state.raw()["parameters"]["base"]["bias"] is not None
    assert lora_paths == {"lora"}
    assert state.raw()["lora"]["lora"]["lora_a"] is not None
    assert state.raw()["lora"]["lora"]["lora_b"] is not None


def test_lora_linear_equals_base_at_init():
    """At init ``lora_b`` is zero so :class:`LoRALinear` reproduces the base."""
    ll = spx_nn.LoRALinear(8, 16, rank=4, rngs=spx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 8), dtype=jnp.float32)
    base_only = ll.base(x)
    assert jnp.allclose(ll(x), base_only, atol=1e-6)


def test_lora_linear_grad_separates_collections():
    """``spx.grad(wrt="lora")`` on :class:`LoRALinear` ignores the base's parameters."""
    ll = spx_nn.LoRALinear(8, 16, rank=4, rngs=spx.Rngs(0))
    x = jnp.ones((2, 8), dtype=jnp.float32)

    @spx.jit
    def step(m, x):
        """Differentiate a sum-of-squares loss targeting the adapter collection."""

        def loss(m, x):
            """Sum-of-squares scalar loss."""
            return (m(x) ** 2).sum()

        return spx.grad(loss, wrt="lora")(m, x)

    grads = step(ll, x)
    raw = grads.raw()
    assert raw.get("lora")
    if "parameters" in raw:
        assert not raw["parameters"]


def test_lora_grad_can_target_both_collections():
    """``spx.grad(wrt=("parameters", "lora"))`` differentiates both groups jointly."""
    base = spx_nn.Linear(8, 16, rngs=spx.Rngs(1))
    lora = spx_nn.LoRA(8, 4, 16, base_module=base, rngs=spx.Rngs(2))
    x = jnp.ones((2, 8), dtype=jnp.float32)

    @spx.jit
    def step(m, x):
        """Joint differentiation across parameters and lora collections."""

        def loss(m, x):
            """Sum-of-squares scalar loss."""
            return (m(x) ** 2).sum()

        return spx.grad(loss, wrt=("parameters", "lora"))(m, x)

    grads = step(lora, x)
    raw = grads.raw()
    assert raw.get("parameters")
    assert raw.get("lora")


def test_lora_parameter_default_kind_is_lora():
    """A freshly-constructed :class:`LoraParameter` lands in ``"lora"``."""
    v = spx_nn.LoraParameter(jnp.zeros((3,), dtype=jnp.float32))
    assert v.kind == "lora"
