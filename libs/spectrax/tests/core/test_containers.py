# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.core.containers`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import spectrax as spx
from spectrax.core.containers import (
    ModuleDict,
    ModuleList,
    ParameterList,
    Sequential,
    StackedModuleList,
    _build_scan_plan_from_modules,
    _build_scan_plan_from_stacked,
)
from spectrax.core.module import Module
from spectrax.core.variable import Parameter
from spectrax.nn.linear import Linear


def _linear(rngs_seed=0):
    """Return a small :class:`Linear` for tests."""
    from spectrax.rng.rngs import Rngs

    return Linear(4, 4, rngs=Rngs(rngs_seed))


class IndexedBlock(Module):
    """Tiny module with per-layer static index metadata."""

    _spx_scan_safe_static_fields = frozenset({"layer_idx"})

    def __init__(self, layer_idx: int):
        """Initialize with layer_idx, weight."""
        super().__init__()
        self.layer_idx = layer_idx
        self.weight = Parameter(jnp.asarray(float(layer_idx + 1), dtype=jnp.float32))

    def forward(self, x):
        """Run the forward pass."""
        return x + self.weight.value


class StaticScaleBlock(Module):
    """Tiny module whose static scale changes computation."""

    def __init__(self, scale: int):
        """Initialize with scale, weight."""
        super().__init__()
        self.scale = scale
        self.weight = Parameter(jnp.asarray(1.0, dtype=jnp.float32))

    def forward(self, x):
        """Run the forward pass."""
        return x * self.scale + self.weight.value


class HeterogeneousA(Module):
    """First heterogeneous module shape for segmented scan tests."""

    def __init__(self):
        """Initialize with weight."""
        super().__init__()
        self.weight = Parameter(jnp.ones((1,)))

    def forward(self, x):
        """Run the forward pass."""
        return x * self.weight.value


class HeterogeneousB(Module):
    """Second heterogeneous module shape for segmented scan tests."""

    def __init__(self):
        """Initialize with other."""
        super().__init__()
        self.other = Parameter(jnp.ones((1,)))

    def forward(self, x):
        """Run the forward pass."""
        return x + self.other.value


class MutatingOpaqueImpl:
    """Opaque helper that records runtime-only attributes."""

    def __init__(self):
        """Initialize with scale."""
        self.scale = 1.0


class OpaqueMutationBlock(Module):
    """Block whose opaque helper mutates during forward."""

    def __init__(self):
        """Initialize with impl, weight."""
        super().__init__()
        self.impl = spx.Opaque(MutatingOpaqueImpl())
        self.weight = Parameter(jnp.asarray(1.0, dtype=jnp.float32))

    def forward(self, x):
        """Run the forward pass."""
        self.impl.RuntimeOnlyType = type(self)
        return x + self.weight.value * self.impl.scale


def test_modulelist_construction_and_len():
    """Construct a :class:`ModuleList` from an iterable."""
    ml = ModuleList([_linear(0), _linear(1)])
    assert len(ml) == 2


def test_modulelist_indexing():
    """Integer indexing returns the stored module."""
    a, b = _linear(0), _linear(1)
    ml = ModuleList([a, b])
    assert ml[0] is a
    assert ml[1] is b


def test_modulelist_slice_returns_same_type():
    """Slicing returns a new :class:`ModuleList` with the same items."""
    a, b, c = _linear(0), _linear(1), _linear(2)
    ml = ModuleList([a, b, c])
    sliced = ml[:2]
    assert isinstance(sliced, ModuleList)
    assert list(sliced) == [a, b]


def test_modulelist_iteration():
    """``__iter__`` yields items in order."""
    items = [_linear(i) for i in range(3)]
    ml = ModuleList(items)
    assert list(iter(ml)) == items


def test_modulelist_append_valid():
    """Appending a :class:`Module` succeeds."""
    ml = ModuleList()
    m = _linear(0)
    ml.append(m)
    assert ml[0] is m


def test_modulelist_append_rejects_non_module():
    """Appending a non-Module raises :class:`TypeError`."""
    ml = ModuleList()
    with pytest.raises(TypeError):
        ml.append("not a module")


def test_modulelist_extend():
    """``extend`` validates every item."""
    ml = ModuleList()
    ml.extend([_linear(0), _linear(1)])
    assert len(ml) == 2


def test_modulelist_not_callable():
    """:class:`ModuleList` is not a callable layer."""
    ml = ModuleList()
    with pytest.raises(RuntimeError):
        ml.forward(jnp.zeros(1))


def test_modulelist_stack_scan_matches_loop():
    """StackedModuleList scans homogeneous layers without changing numerics."""
    layers = ModuleList([_linear(i) for i in range(3)])
    stacked = layers.stack()
    x = jnp.ones((2, 4))
    expected = x
    for layer in layers:
        expected = layer(expected)

    out = stacked.scan(lambda layer, carry: layer(carry), x)

    assert isinstance(stacked, StackedModuleList)
    assert len(stacked) == len(layers)
    assert jnp.allclose(out, expected, atol=1e-5)


def test_modulelist_scan_trace_matches_scan():
    """``trace=True`` executes the Python-loop path with scan-equivalent output."""
    layers = ModuleList([_linear(i) for i in range(3)])
    x = jnp.ones((2, 4))

    scanned = layers.scan(lambda layer, carry: layer(carry), x)
    traced = layers.scan(lambda layer, carry: layer(carry), x, trace=True)

    assert jnp.allclose(traced, scanned, atol=1e-5)


def test_modulelist_scan_default_uses_real_scan_plan():
    """Default unstacked scans should still use the real scan planner."""
    layers = ModuleList([IndexedBlock(i) for i in range(3)])
    x = jnp.asarray(0.0, dtype=jnp.float32)

    scanned = layers.scan(lambda layer, carry: layer(carry), x)
    traced = layers.scan(lambda layer, carry: layer(carry), x, trace=True)

    assert getattr(layers, "_spx_scan_plan_cache", None) is not None
    assert jnp.allclose(scanned, traced)


def test_modulelist_scan_explicit_unroll_uses_rolled_scan_plan():
    """Explicit unroll keeps the real ``lax.scan`` path available."""
    layers = ModuleList([IndexedBlock(i) for i in range(3)])
    x = jnp.asarray(0.0, dtype=jnp.float32)

    rolled = layers.scan(lambda layer, carry: layer(carry), x, unroll=1)
    traced = layers.scan(lambda layer, carry: layer(carry), x, trace=True)

    assert getattr(layers, "_spx_scan_plan_cache", None) is not None
    assert jnp.allclose(rolled, traced)


def test_modulelist_scan_unroll_zero_is_preserved_as_full_unroll():
    """``unroll=0`` is a real JAX mode and must not be rewritten to 1."""
    layers = ModuleList([IndexedBlock(i) for i in range(3)])
    x = jnp.asarray(0.0, dtype=jnp.float32)

    fully_unrolled = layers.scan(lambda layer, carry: layer(carry), x, unroll=0)
    traced = layers.scan(lambda layer, carry: layer(carry), x, trace=True)

    assert jnp.allclose(fully_unrolled, traced)


def test_modulelist_scan_rejects_negative_unroll():
    """Negative unroll values are user errors, not hidden JAX crashes."""
    layers = ModuleList([IndexedBlock(i) for i in range(3)])

    with pytest.raises(ValueError, match="unroll"):
        layers.scan(lambda layer, carry: layer(carry), jnp.asarray(0.0), unroll=-1)


def test_modulelist_scan_accepts_per_layer_index_static():
    """Layer-index statics should not force slow per-layer graph dispatch."""

    layers = ModuleList([IndexedBlock(i) for i in range(3)])
    x = jnp.asarray(0.0, dtype=jnp.float32)
    plan = _build_scan_plan_from_modules(list(layers), context="test")

    scanned = layers.scan(lambda layer, carry: layer(carry), x)
    rolled = layers.scan(lambda layer, carry: layer(carry), x, unroll=1)
    traced = layers.scan(lambda layer, carry: layer(carry), x, trace=True)

    assert plan.lowering == "single_template"
    assert jnp.allclose(scanned, traced)
    assert jnp.allclose(rolled, traced)


def test_modulelist_scan_segments_behavior_changing_static_runs():
    """Behavior-changing statics form graph families instead of unsafe templates."""

    layers = ModuleList([StaticScaleBlock(2), StaticScaleBlock(2), StaticScaleBlock(3), StaticScaleBlock(3)])
    x = jnp.asarray(1.0, dtype=jnp.float32)
    plan = _build_scan_plan_from_modules(list(layers), context="test")

    scanned = layers.scan(lambda layer, carry: layer(carry), x)
    traced = layers.scan(lambda layer, carry: layer(carry), x, trace=True)

    assert plan.lowering == "segmented_templates"
    assert [segment.length for segment in plan.segments] == [2, 2]
    assert jnp.allclose(scanned, traced)


def test_stacked_modulelist_scan_supports_segmented_static_families():
    """Pre-stacked containers keep segmented multi-graph scan semantics."""

    layers = ModuleList([StaticScaleBlock(2), StaticScaleBlock(2), StaticScaleBlock(3)]).stack()
    x = jnp.asarray(1.0, dtype=jnp.float32)

    scanned = layers.scan(lambda layer, carry: layer(carry), x)
    traced = layers.scan(lambda layer, carry: layer(carry), x, trace=True)

    assert jnp.allclose(scanned, traced)


def test_stacked_modulelist_scan_plan_cache_keeps_fresh_values():
    """Cached segmentation metadata must not freeze old parameter values."""

    layers = ModuleList([IndexedBlock(0), IndexedBlock(1)]).stack()
    x = jnp.asarray(0.0, dtype=jnp.float32)

    first = layers.scan(lambda layer, carry: layer(carry), x)
    layers.v0.value = layers.v0.value + 1.0
    second = layers.scan(lambda layer, carry: layer(carry), x)

    assert layers._spx_scan_plan_cache is not None
    assert jnp.allclose(second, first + 2.0)


def test_stacked_modulelist_family_keys_survive_opaque_runtime_mutation():
    """Runtime-only opaque mutations must not split homogeneous scan plans."""

    layers = ModuleList([OpaqueMutationBlock() for _ in range(3)]).stack()
    x = jnp.asarray(0.0, dtype=jnp.float32)

    first = layers.scan(lambda layer, carry: layer(carry), x)
    plan = _build_scan_plan_from_stacked(
        layers._spx_item_gdefs,
        layers._stacked_state(),
        context="test",
        family_keys=layers._spx_item_family_keys,
    )
    second = layers.scan(lambda layer, carry: layer(carry), x)

    assert [segment.length for segment in plan.segments] == [3]
    assert jnp.allclose(first, second)


def test_stacked_modulelist_scan_trace_matches_scan_under_jit():
    """Stacked scan keeps a traceable loop fallback for debugging paths."""
    layers = ModuleList([_linear(i) for i in range(3)]).stack()
    x = jnp.ones((2, 4))

    def run(stacked_layers, xb):
        """Run helper."""
        scanned = stacked_layers.scan(lambda layer, carry: layer(carry), xb)
        traced = stacked_layers.scan(lambda layer, carry: layer(carry), xb, trace=True)
        return scanned, traced

    scanned, traced = jax.jit(run)(layers, x)
    assert jnp.allclose(traced, scanned, atol=1e-5)


def test_stacked_modulelist_is_trainable_pytree():
    """Gradients flow through the stacked layer-axis leaves."""
    layers = ModuleList([_linear(i) for i in range(2)]).stack()
    x = jnp.ones((2, 4))

    def loss(stacked_layers, xb):
        """Compute the loss."""
        return stacked_layers.scan(lambda layer, carry: layer(carry), xb).sum()

    value, grads = spx.value_and_grad(loss)(layers, x)

    assert value.shape == ()
    assert "parameters" in grads
    assert all(leaf.shape[0] == 2 for _collection, _path, leaf in grads.items())


def test_stacked_modulelist_jit_forward():
    """JAX sees a stacked module list as normal model state."""
    layers = ModuleList([_linear(i) for i in range(2)]).stack()
    x = jnp.ones((2, 4))
    out = jax.jit(lambda stacked_layers, xb: stacked_layers.scan(lambda layer, carry: layer(carry), xb))(layers, x)
    assert out.shape == (2, 4)


def test_stacked_modulelist_rejects_heterogeneous_graphs():
    """Stacked scans require identical per-layer graph definitions."""

    class A(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize with weight."""
            super().__init__()
            self.weight = Parameter(jnp.ones((1,)))

        def forward(self, x):
            """Run the forward pass."""
            return x * self.weight.value

    class B(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize with other."""
            super().__init__()
            self.other = Parameter(jnp.ones((1,)))

        def forward(self, x):
            """Run the forward pass."""
            return x + self.other.value

    with pytest.raises(ValueError, match="compatible graph topology"):
        ModuleList([A(), B()]).stack()


def test_modulelist_scan_segments_heterogeneous_graphs():
    """Plain ModuleList.scan can segment mixed graph definitions."""

    layers = ModuleList([HeterogeneousA(), HeterogeneousB(), HeterogeneousA()])
    x = jnp.ones((1,))
    scanned = layers.scan(lambda layer, carry: layer(carry), x)
    traced = layers.scan(lambda layer, carry: layer(carry), x, trace=True)
    assert jnp.allclose(scanned, traced)


def test_modulelist_graph_children_integer_keys():
    """``_spx_graph_children`` yields integer keys."""
    ml = ModuleList([_linear(0), _linear(1)])
    keys = [k for k, _ in ml._spx_graph_children()]
    assert keys == [0, 1]


def test_sequential_forwards_through_items():
    """``Sequential`` threads input through each child in order."""
    seq = Sequential(_linear(0), _linear(1))
    x = jnp.zeros((2, 4))
    y = seq(x)
    assert y.shape == (2, 4)


def test_sequential_rejects_non_module():
    """Non-Modules in the constructor raise via validation on append."""
    with pytest.raises(TypeError):
        Sequential(_linear(0), "bogus")


def test_sequential_kwargs_passthrough_or_fallback():
    """``Sequential`` passes kwargs where accepted, falls back otherwise."""

    class AcceptsKwargs(Module):
        """Module that consumes arbitrary kwargs."""

        def forward(self, x, **kwargs):
            """Return ``x + 1`` regardless of kwargs."""
            return x + 1

    class PositionalOnly(Module):
        """Module whose ``forward`` rejects extra kwargs."""

        def forward(self, x):
            """Return ``x * 2``."""
            return x * 2

    seq = Sequential(AcceptsKwargs(), PositionalOnly())
    y = seq(jnp.asarray([1.0]), some_kwarg=True)
    assert float(y[0]) == 4.0


def test_parameterlist_only_accepts_parameters():
    """``ParameterList`` rejects non-:class:`Parameter` items."""
    pl = ParameterList()
    pl.append(Parameter(jnp.zeros(3)))
    assert len(pl) == 1
    with pytest.raises(TypeError):
        pl.append(jnp.zeros(3))


def test_parameterlist_not_callable():
    """:class:`ParameterList` is not a callable layer."""
    with pytest.raises(RuntimeError):
        ParameterList().forward(None)


def test_moduledict_setitem_and_getitem():
    """String-keyed setitem/getitem round-trips a module."""
    md = ModuleDict()
    m = _linear(0)
    md["a"] = m
    assert md["a"] is m


def test_moduledict_construction_from_mapping():
    """Construct from a plain dict of modules."""
    a, b = _linear(0), _linear(1)
    md = ModuleDict({"a": a, "b": b})
    assert md["a"] is a and md["b"] is b


def test_moduledict_rejects_non_string_key():
    """Non-string keys raise :class:`TypeError`."""
    md = ModuleDict()
    with pytest.raises(TypeError):
        md[1] = _linear(0)


def test_moduledict_rejects_non_module_value():
    """Non-Module values raise :class:`TypeError`."""
    md = ModuleDict()
    with pytest.raises(TypeError):
        md["x"] = "bogus"


def test_moduledict_contains_len_iter():
    """Standard collection protocols on :class:`ModuleDict`."""
    md = ModuleDict({"a": _linear(0)})
    assert "a" in md
    assert "b" not in md
    assert len(md) == 1
    assert list(md) == ["a"]


def test_moduledict_keys_values_items():
    """``keys``/``values``/``items`` return views of the stored dict."""
    a, b = _linear(0), _linear(1)
    md = ModuleDict({"a": a, "b": b})
    assert set(md.keys()) == {"a", "b"}
    assert set(md.values()) == {a, b}
    assert dict(md.items()) == {"a": a, "b": b}


def test_moduledict_graph_children_string_keys():
    """``_spx_graph_children`` yields string keys in insertion order."""
    md = ModuleDict({"first": _linear(0), "second": _linear(1)})
    keys = [k for k, _ in md._spx_graph_children()]
    assert keys == ["first", "second"]


def test_moduledict_not_callable():
    """:class:`ModuleDict` is not a callable layer."""
    with pytest.raises(RuntimeError):
        ModuleDict().forward(None)


def test_containers_have_no_static_fields():
    """Containers override ``_spx_static_fields`` to empty."""
    assert ModuleList()._spx_static_fields() == {}
    assert Sequential()._spx_static_fields() == {}
    assert ModuleDict()._spx_static_fields() == {}
    assert ParameterList()._spx_static_fields() == {}
