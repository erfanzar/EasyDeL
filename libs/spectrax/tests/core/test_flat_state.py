# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :meth:`State.flatten` / :meth:`State.from_flat`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
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


@pytest.fixture
def sharded_state_inputs():
    """Unequal State sizes expose accidental params/aux sharding swaps."""
    params = State({"parameters": {"w": jnp.asarray(2.0), "b": jnp.asarray(3.0), "c": jnp.asarray(5.0)}})
    aux = State({"buffers": {"a": jnp.asarray(7.0), "b": jnp.asarray(11.0)}})
    inputs = jnp.asarray([1.0, 2.0])
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    specs = (jax.tree.map(lambda _: sharding, params), jax.tree.map(lambda _: sharding, aux), sharding)
    return params, aux, inputs, specs


def _state_affine_output(params, aux, inputs):
    """Use every leaf of two independently shaped States."""
    return (
        params.get("parameters", "w") * inputs
        + params.get("parameters", "b")
        + params.get("parameters", "c")
        + aux.get("buffers", "a")
        + aux.get("buffers", "b")
    )


@pytest.mark.parametrize("lower_first", [False, True], ids=["call-first", "lower-first"])
@pytest.mark.parametrize(
    "static_options",
    [
        {"static_argnums": 0},
        {"static_argnums": -4},
        {"static_argnames": "scale"},
    ],
    ids=["positive", "negative", "inferred-position"],
)
def test_spx_jit_state_shardings_after_leading_static(sharded_state_inputs, static_options, lower_first):
    """State shardings exclude static args, in calls and ahead-of-time lowering."""
    params, aux, inputs, specs = sharded_state_inputs

    def step(scale, params, aux, inputs):
        if scale != 2:
            raise ValueError("scale must be static and equal to two")
        return scale * _state_affine_output(params, aux, inputs)

    wrapped = spx.jit(step, in_shardings=specs, **static_options)
    if lower_first:
        # Lowered executables expose the flat, dynamic-only ABI, not live States.
        compiled = wrapped.lower(2, params, aux, inputs).compile()
        assert jnp.allclose(compiled(params.call_leaves(), aux.call_leaves(), inputs), jnp.asarray([56.0, 60.0]))
    assert jnp.allclose(wrapped(2, params, aux, inputs), jnp.asarray([56.0, 60.0]))
    params.set("parameters", "w", jnp.asarray(4.0))
    assert jnp.allclose(wrapped(2, params, aux, inputs), jnp.asarray([60.0, 68.0]))
    if lower_first:
        assert jnp.allclose(compiled(params.call_leaves(), aux.call_leaves(), inputs), jnp.asarray([60.0, 68.0]))


@pytest.mark.parametrize("lower_first", [False, True], ids=["call-first", "lower-first"])
@pytest.mark.parametrize(
    "static_options",
    [
        {"static_argnums": (0, 2)},
        {"static_argnums": (-5, -3)},
        {"static_argnums": (0, -3)},
        {"static_argnames": ("scale", "offset")},
        {"static_argnums": (0, 2), "static_argnames": ()},
    ],
    ids=["interleaved", "negative", "mixed-sign", "inferred-positions", "explicit-both"],
)
def test_spx_jit_state_shardings_with_interleaved_statics(sharded_state_inputs, static_options, lower_first):
    """Multiple statics shift each State by a different number of positions."""
    params, aux, inputs, specs = sharded_state_inputs

    def step(scale, params, offset, aux, inputs):
        if scale != 2 or offset != 13:
            raise ValueError("scale and offset must be static")
        return scale * _state_affine_output(params, aux, inputs) + offset

    wrapped = spx.jit(step, in_shardings=specs, **static_options)
    if lower_first:
        compiled = wrapped.lower(2, params, 13, aux, inputs).compile()
        assert jnp.allclose(compiled(params.call_leaves(), aux.call_leaves(), inputs), jnp.asarray([69.0, 73.0]))
    assert jnp.allclose(wrapped(2, params, 13, aux, inputs), jnp.asarray([69.0, 73.0]))
    params.set("parameters", "w", jnp.asarray(4.0))
    assert jnp.allclose(wrapped(2, params, 13, aux, inputs), jnp.asarray([73.0, 81.0]))


@pytest.mark.parametrize("lower_first", [False, True], ids=["call-first", "lower-first"])
@pytest.mark.parametrize("prefix", ["state-none", "none", "scalar", "global-none", "global-scalar"])
def test_spx_jit_static_state_sharding_prefixes(sharded_state_inputs, prefix, lower_first):
    """Unspecified trees and scalar prefixes keep their existing JAX semantics."""
    params, aux, inputs, specs = sharded_state_inputs
    if prefix == "state-none":
        in_shardings = (jax.tree.map(lambda _: None, params), jax.tree.map(lambda _: None, aux), None)
    elif prefix == "none":
        in_shardings = (None, None, None)
    elif prefix == "scalar":
        in_shardings = (specs[2], specs[2], specs[2])
    elif prefix == "global-none":
        in_shardings = None
    else:
        in_shardings = specs[2]

    def step(scale, params, offset, aux, inputs):
        return scale * _state_affine_output(params, aux, inputs) + offset

    wrapped = spx.jit(step, static_argnums=(0, 2), in_shardings=in_shardings)
    if lower_first:
        compiled = wrapped.lower(2, params, 13, aux, inputs).compile()
        assert jnp.allclose(compiled(params.call_leaves(), aux.call_leaves(), inputs), jnp.asarray([69.0, 73.0]))
    assert jnp.allclose(wrapped(2, params, 13, aux, inputs), jnp.asarray([69.0, 73.0]))


@pytest.mark.parametrize("lower_first", [False, True], ids=["call-first", "lower-first"])
def test_spx_jit_explicit_static_options_disable_inference(sharded_state_inputs, lower_first):
    """A named static passed positionally stays dynamic when nums are explicit."""
    params, aux, inputs, specs = sharded_state_inputs

    def step(scale, params, offset, aux, inputs):
        if scale != 2:
            raise ValueError("scale must be static")
        return scale * _state_affine_output(params, aux, inputs) + offset

    wrapped = spx.jit(
        step,
        static_argnums=(0,),
        static_argnames=("offset",),
        in_shardings=(specs[0], specs[2], specs[1], specs[2]),
    )
    offset = jnp.asarray(13.0)
    if lower_first:
        compiled = wrapped.lower(2, params, offset, aux, inputs).compile()
        actual = compiled(params.call_leaves(), offset, aux.call_leaves(), inputs)
        assert jnp.allclose(actual, jnp.asarray([69.0, 73.0]))
    assert jnp.allclose(wrapped(2, params, offset, aux, inputs), jnp.asarray([69.0, 73.0]))
    assert jnp.allclose(wrapped(2, params, jnp.asarray(17.0), aux, inputs), jnp.asarray([73.0, 77.0]))


@pytest.mark.parametrize("lower", [False, True], ids=["call", "lower"])
def test_spx_jit_static_state_sharding_mismatch_is_rejected(sharded_state_inputs, lower):
    """Genuine State ABI leaf-count mismatches are still rejected."""
    params, aux, inputs, specs = sharded_state_inputs

    def step(scale, params, aux, inputs):
        return scale * _state_affine_output(params, aux, inputs)

    wrapped = spx.jit(step, static_argnums=0, in_shardings=(specs[1], specs[1], specs[2]))
    call = wrapped.lower if lower else wrapped
    with pytest.raises(ValueError, match=r"StateCallABI\.flatten_sharding leaf count mismatch: expected 3, got 2"):
        call(2, params, aux, inputs)


@pytest.mark.parametrize("lower_first", [False, True], ids=["call-first", "lower-first"])
def test_spx_jit_negative_static_index_uses_actual_call_length(sharded_state_inputs, lower_first):
    """An omitted default must not shift a negative index relative to live args."""
    params, aux, inputs, specs = sharded_state_inputs

    def step(scale, params, aux, inputs, offset=13):
        if scale != 2:
            raise ValueError("scale must be static")
        return scale * _state_affine_output(params, aux, inputs) + offset

    wrapped = spx.jit(step, static_argnums=-4, in_shardings=specs)
    if lower_first:
        compiled = wrapped.lower(2, params, aux, inputs).compile()
        assert jnp.allclose(compiled(params.call_leaves(), aux.call_leaves(), inputs), jnp.asarray([69.0, 73.0]))
    assert jnp.allclose(wrapped(2, params, aux, inputs), jnp.asarray([69.0, 73.0]))


@pytest.mark.parametrize("by_name", [False, True], ids=["infer-name", "infer-position"])
def test_spx_jit_inferred_static_state_is_not_flattened(by_name):
    """A static State is a hashable object, not a tuple of unhashable arrays."""
    params = State({"parameters": {"w": jnp.asarray(3.0)}})
    config = State({"options": {"scale": jnp.asarray(2.0)}})

    def step(params, config):
        return params.get("parameters", "w") * config.get("options", "scale")

    if by_name:
        wrapped = spx.jit(step, static_argnames="config")
        assert jnp.allclose(wrapped(params, config), 6.0)
    else:
        wrapped = spx.jit(step, static_argnums=1)
        assert jnp.allclose(wrapped(params, config=config), 6.0)
    params.set("parameters", "w", jnp.asarray(5.0))
    if by_name:
        assert jnp.allclose(wrapped(params, config), 10.0)
    else:
        assert jnp.allclose(wrapped(params, config=config), 10.0)


def test_spx_jit_static_state_distinct_partitioned_shardings():
    """Preserve distinct four-device State layouts through a leading static arg."""
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("requires four devices for a 2x2 mesh")
    mesh = jax.sharding.Mesh(np.asarray(devices[:4]).reshape(2, 2), ("rows", "cols"))
    rows = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("rows", None))
    cols = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "cols"))
    tiles = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("rows", "cols"))
    transposed_tiles = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("cols", "rows"))
    param_specs = State({"parameters": {"w": rows, "b": cols, "c": tiles}})
    aux_specs = State({"buffers": {"a": transposed_tiles, "b": rows}})
    grid = np.arange(16, dtype=np.float32).reshape(4, 4)
    params = State(
        {
            "parameters": {
                "w": jax.device_put(grid + 2, rows),
                "b": jax.device_put(grid + 3, cols),
                "c": jax.device_put(grid + 5, tiles),
            }
        }
    )
    aux = State({"buffers": {"a": jax.device_put(grid + 7, transposed_tiles), "b": jax.device_put(grid + 11, rows)}})
    inputs = jax.device_put(grid + 1, tiles)

    def step(scale, params, aux, inputs):
        if scale != 2:
            raise ValueError("scale must be static and equal to two")
        return scale * _state_affine_output(params, aux, inputs)

    wrapped = spx.jit(
        step,
        static_argnums=0,
        in_shardings=(param_specs, aux_specs, tiles),
        out_shardings=transposed_tiles,
    )
    # Exercise the cold lowering path as well as normal calls. The executable
    # takes flat dynamic State leaves; sharding is still per original State leaf.
    compiled = wrapped.lower(2, params, aux, inputs).compile()
    expected = 2 * ((grid + 2) * (grid + 1) + 4 * grid + 26)
    for actual in (compiled(params.call_leaves(), aux.call_leaves(), inputs), wrapped(2, params, aux, inputs)):
        np.testing.assert_array_equal(jax.device_get(actual), expected)
        assert actual.sharding.is_equivalent_to(transposed_tiles, ndim=2)
        assert not actual.sharding.is_fully_replicated

    params.set("parameters", "w", jax.device_put(grid + 4, rows))
    expected_changed = 2 * ((grid + 4) * (grid + 1) + 4 * grid + 26)
    for actual in (wrapped(2, params, aux, inputs), compiled(params.call_leaves(), aux.call_leaves(), inputs)):
        np.testing.assert_array_equal(jax.device_get(actual), expected_changed)
        assert actual.sharding.is_equivalent_to(transposed_tiles, ndim=2)
        assert not actual.sharding.is_fully_replicated
