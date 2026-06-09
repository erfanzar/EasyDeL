# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :meth:`State.flatten` / :meth:`State.from_flat`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import spectrax as spx
from spectrax.core.graph import export
from spectrax.core.state import State
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


def test_flatten_produces_slash_separated_keys():
    """Flat keys have form ``collection/path``."""
    m = Linear(2, 2, rngs=Rngs(0))
    _gdef, state = export(m)
    flat = state.flatten()
    for k in flat:
        assert "/" in k


def test_flatten_from_flat_roundtrip():
    """``from_flat(flatten(state))`` reconstructs the state."""
    m = Linear(2, 2, rngs=Rngs(0))
    _gdef, state = export(m)
    restored = State.from_flat(state.flatten())
    for c in state:
        assert c in restored
        for p, v in state.raw()[c].items():
            assert jnp.array_equal(v, restored.raw()[c][p])


def test_from_flat_rejects_malformed_keys():
    """Keys without ``/`` raise."""
    with pytest.raises(ValueError):
        State.from_flat({"bad_key": jnp.zeros(())})


def test_state_call_abi_roundtrip():
    """``StateCallABI`` round-trips through a tuple of leaves."""
    m = Linear(2, 2, rngs=Rngs(0))
    _gdef, state = export(m)

    abi = state.call_abi()
    leaves = abi.flatten(state)
    restored = abi.unflatten(leaves)

    assert isinstance(restored, State)
    for collection, path, value in state.items():
        assert jnp.array_equal(value, restored.get(collection, path))


def test_state_call_abi_rejects_wrong_structure():
    """A cached call ABI is tied to one State pytree structure."""
    m = Linear(2, 2, rngs=Rngs(0))
    _gdef, state = export(m)
    abi = spx.state_call_abi(state)

    changed = state.set("parameters", "extra", jnp.zeros(()), copy=True)

    with pytest.raises(ValueError, match="different pytree structure"):
        abi.flatten(changed)
    with pytest.raises(ValueError, match="leaf count mismatch"):
        abi.unflatten(abi.flatten(state)[:-1])


def test_state_call_abi_inside_jit():
    """Flat state leaves can be passed through a jitted serving-style call."""
    m = Linear(2, 2, rngs=Rngs(0))
    gdef, state = export(m)
    abi = state.call_abi()
    x = jnp.ones((1, 2))
    expected = m(x)

    @jax.jit
    def step(state_leaves, inputs):
        rebound = spx.bind(gdef, abi.unflatten(state_leaves))
        return rebound(inputs)

    actual = step(abi.flatten(state), x)

    assert jnp.allclose(actual, expected)


def test_spx_jit_flattens_state_argument_automatically():
    """``spx.jit`` lowers top-level State args through the flat call ABI."""
    m = Linear(2, 2, rngs=Rngs(0))
    gdef, state = export(m)
    x = jnp.ones((1, 2))
    expected = m(x)

    @spx.jit
    def step(model_state, inputs):
        rebound = spx.bind(gdef, model_state)
        return rebound(inputs)

    actual = step(state, x)

    assert jnp.allclose(actual, expected)
    cache_key = next(iter(step._spx_compile_cache))
    assert cache_key[-1][0][:3] == ("arg", 0, len(state.call_leaves()))


def test_spx_jit_flattens_state_kwarg_automatically():
    """Keyword State args get the same automatic ABI flattening."""
    m = Linear(2, 2, rngs=Rngs(0))
    gdef, state = export(m)
    x = jnp.ones((1, 2))
    expected = m(x)

    @spx.jit
    def step(inputs, *, model_state):
        rebound = spx.bind(gdef, model_state)
        return rebound(inputs)

    actual = step(x, model_state=state)

    assert jnp.allclose(actual, expected)
    cache_key = next(iter(step._spx_compile_cache))
    assert cache_key[-1][0][:3] == ("kwarg", "model_state", len(state.call_leaves()))


def test_spx_jit_state_arg_cache_invalidates_on_mutation():
    """Automatic State leaf caching follows normal State mutation APIs."""
    state = spx.State({"parameters": {"w": jnp.asarray(1.0)}})

    @spx.jit
    def read_weight(model_state):
        return model_state.get("parameters", "w")

    first = read_weight(state)
    state.set("parameters", "w", jnp.asarray(2.0))
    second = read_weight(state)

    assert jnp.allclose(first, 1.0)
    assert jnp.allclose(second, 2.0)
