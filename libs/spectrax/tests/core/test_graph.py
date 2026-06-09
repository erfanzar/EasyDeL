# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.core.graph`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import spectrax
from spectrax.core.errors import CyclicGraphError
from spectrax.core.graph import (
    GraphDef,
    ModuleNode,
    VarNode,
    bind,
    clone,
    export,
    live_variables,
    tree_state,
    update,
)
from spectrax.core.module import Module
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


class TinyModel(Module):
    """Module with a single :class:`Linear` child used across graph tests."""

    def __init__(self, rngs: Rngs):
        """Install ``self.fc`` as a ``Linear(4, 4)``."""
        super().__init__()
        self.fc = Linear(4, 4, rngs=rngs)

    def forward(self, x, **_):
        """Apply the child."""
        return self.fc(x)


class TiedModel(Module):
    """Module with a tied-weight alias to exercise sharing detection."""

    def __init__(self, rngs: Rngs):
        """Create ``self.fc`` and alias it as ``self.fc_tied``."""
        super().__init__()
        self.fc = Linear(4, 4, rngs=rngs)
        self.fc_tied = self.fc

    def forward(self, x, **_):
        """Apply the shared layer twice and sum."""
        return self.fc(x) + self.fc_tied(x)


class RuntimeDispatchModel(Module):
    """Module that routes through private runtime attrs."""

    def __init__(self, rngs: Rngs):
        """Initialize with left, right."""
        super().__init__()
        self.left = Linear(4, 4, rngs=rngs)
        self.right = Linear(4, 4, rngs=rngs)
        self._target = "right"
        self._scale = lambda x: x * 2

    def forward(self, x):
        """Run the forward pass."""
        return self._scale(getattr(self, self._target)(x))


class RuntimeTargetModel(Module):
    """Module whose private routing attr may be updated after export."""

    def __init__(self, rngs: Rngs):
        """Initialize with left, right."""
        super().__init__()
        self.left = Linear(4, 4, rngs=rngs)
        self.right = Linear(4, 4, rngs=rngs)
        self._target = "left"

    def forward(self, x):
        """Run the forward pass."""
        return getattr(self, self._target)(x)


def test_export_rejects_non_module():
    """:func:`export` rejects non-:class:`Module` inputs."""
    with pytest.raises(TypeError):
        export(42)


def test_export_returns_graphdef_and_state():
    """:func:`export` returns a ``(GraphDef, State)`` pair."""
    g, s = export(TinyModel(Rngs(0)))
    assert isinstance(g, GraphDef)
    assert g.nodes
    assert "parameters" in s


def test_graphdef_is_hashable_and_equal_across_seeds():
    """Two structurally-equal modules yield equal :class:`GraphDef` values."""
    ga, _ = export(TinyModel(Rngs(0)))
    gb, _ = export(TinyModel(Rngs(1)))
    assert ga == gb
    assert hash(ga) == hash(gb)


def test_graphdef_nodes_tuple_immutable():
    """:class:`GraphDef.nodes` is a tuple (frozen structure)."""
    g, _ = export(TinyModel(Rngs(0)))
    assert isinstance(g.nodes, tuple)


def test_graphdef_root_indexes_into_nodes():
    """``graphdef.root`` points at a valid :class:`ModuleNode`."""
    g, _ = export(TinyModel(Rngs(0)))
    assert isinstance(g.nodes[g.root], ModuleNode)


def test_graphdef_var_refs_contains_one_entry_per_variable():
    """Every unique Variable appears once in ``var_refs``."""
    g, _ = export(TinyModel(Rngs(0)))
    assert len(g.var_refs) == len(g.var_canonical)


def test_graphdef_inequality_on_different_structure():
    """Different structure yields different :class:`GraphDef`."""
    from spectrax.core.module import Module as _Module

    class Other(_Module):
        """Different structure than :class:`TinyModel`."""

        def __init__(self, rngs):
            """Initialize with head."""
            super().__init__()
            self.head = Linear(4, 8, rngs=rngs)

        def forward(self, x):
            """Run the forward pass."""
            return self.head(x)

    g1, _ = export(TinyModel(Rngs(0)))
    g2, _ = export(Other(Rngs(0)))
    assert g1 != g2


def test_graphdef_equality_with_non_graphdef():
    """Comparing with a non-GraphDef returns ``NotImplemented`` -> ``False``."""
    g, _ = export(TinyModel(Rngs(0)))
    assert (g == 42) is False


def test_module_node_is_frozen():
    """:class:`ModuleNode` is a frozen dataclass."""
    m = ModuleNode(class_name="X", static_fields=(), children=(), container_kind="module")
    with pytest.raises(AttributeError):
        m.class_name = "Y"


def test_var_node_is_frozen():
    """:class:`VarNode` is a frozen dataclass."""
    v = VarNode(class_name="X", collection="parameters", metadata=())
    with pytest.raises(AttributeError):
        v.class_name = "Y"


def test_canonical_path_lookup():
    """``GraphDef.canonical_path`` returns the canonical path for a ref_id."""
    g, _ = export(TinyModel(Rngs(0)))
    first_rid = g.var_canonical[0][0]
    assert g.canonical_path(first_rid) == g.var_canonical[0][1]


def test_canonical_path_missing_raises_keyerror():
    """Missing ref_id raises :class:`KeyError`."""
    g, _ = export(TinyModel(Rngs(0)))
    with pytest.raises(KeyError):
        g.canonical_path(9999)


def test_export_detects_cycle():
    """A self-referential module raises :class:`CyclicGraphError`."""

    class Cyc(Module):
        """Module that creates a cycle after construction."""

        def __init__(self, rngs):
            """Initialize with fc."""
            super().__init__()
            self.fc = Linear(4, 4, rngs=rngs)

        def forward(self, x):
            """Run the forward pass."""
            return self.fc(x)

    m = Cyc(Rngs(0))
    m.self_ref = m
    with pytest.raises(CyclicGraphError):
        export(m)


def test_export_records_shared_paths():
    """Shared modules are recorded in ``shared_paths``."""
    g, _ = export(TiedModel(Rngs(0)))
    assert g.shared_paths


def test_shared_variable_appears_once_in_state():
    """A tied weight has a single leaf in ``state['parameters']``."""
    _, s = export(TiedModel(Rngs(0)))
    weight_paths = [p for c, p, _ in s.items() if c == "parameters" and p.endswith("weight")]
    assert len(weight_paths) == 1


def test_bind_roundtrip_identical_output():
    """``bind(export(m))`` reproduces the same forward outputs."""
    m = TinyModel(Rngs(0))
    g, s = export(m)
    m2 = bind(g, s)
    x = jnp.ones((2, 4))
    assert jnp.allclose(m(x), m2(x))


def test_bind_preserves_sharing():
    """Sharing is preserved across ``export``/``bind``."""
    m = TiedModel(Rngs(0))
    g, s = export(m)
    m2 = bind(g, s)
    assert m2.fc is m2.fc_tied


def test_bind_creates_fresh_variables():
    """``bind`` allocates new :class:`Variable` instances with fresh ids."""
    m = TinyModel(Rngs(0))
    g, s = export(m)
    m2 = bind(g, s)
    m_ids = {id(v) for _, v in live_variables(m)}
    m2_ids = {id(v) for _, v in live_variables(m2)}
    assert m_ids.isdisjoint(m2_ids)


def test_clone_severs_variable_identity():
    """``clone`` produces a module sharing no :class:`Variable` with the source."""
    m = TinyModel(Rngs(0))
    c = clone(m)
    m_ids = {id(v) for _, v in live_variables(m)}
    c_ids = {id(v) for _, v in live_variables(c)}
    assert m_ids.isdisjoint(c_ids)


def test_clone_preserves_outputs():
    """``clone`` preserves forward output numerically."""
    m = TinyModel(Rngs(0))
    c = clone(m)
    x = jnp.ones((2, 4))
    assert jnp.allclose(m(x), c(x))


def test_bind_restores_private_runtime_attrs_generically():
    """Single-underscore runtime attrs survive export/bind without per-class hooks."""
    m = RuntimeDispatchModel(Rngs(0))
    gdef, state = export(m)
    rebound = bind(gdef, state)
    assert rebound._target == "right"
    assert "_target" not in rebound._spx_attr_order
    assert jnp.allclose(rebound(jnp.ones((2, 4))), m(jnp.ones((2, 4))))


def test_private_runtime_attr_update_invalidates_export_cache():
    """Changing a private runtime attr is reflected by the next export."""
    m = RuntimeTargetModel(Rngs(0))
    export(m)
    m._target = "right"
    gdef, state = export(m)
    rebound = bind(gdef, state)
    assert rebound._target == "right"


def test_update_writes_state_back():
    """``update`` writes new state leaves into the module's variables."""
    m = TinyModel(Rngs(0))
    _, s = export(m)
    new_weight = jnp.ones_like(s["parameters"]["fc"]["weight"])
    new_state = s.set("parameters", "fc.weight", new_weight, copy=True)
    update(m, new_state)
    assert jnp.array_equal(m.fc.weight.value, new_weight)


def test_update_ignores_extra_paths():
    """Extra entries in the provided state are ignored."""
    m = TinyModel(Rngs(0))
    _, s = export(m)
    s2 = s.set("parameters", "does.not.exist", jnp.zeros(1), copy=True)
    update(m, s2)


def test_exported_state_set_writes_through_to_live_module():
    """Live-backed exported states update module variables on in-place set."""
    m = TinyModel(Rngs(0))
    _, s = export(m)
    new_weight = jnp.full_like(s["parameters"]["fc"]["weight"], 3.0)
    out = s.set("parameters", "fc.weight", new_weight)
    assert out is s
    assert jnp.array_equal(m.fc.weight.value, new_weight)


def test_exported_state_indexed_assignment_writes_through_to_live_module():
    """Live-backed exported states update module variables on indexed assignment."""
    m = TinyModel(Rngs(0))
    _, s = export(m)
    new_weight = jnp.full_like(s["parameters"]["fc"]["weight"], 4.0)
    s["parameters"]["fc"]["weight"] = new_weight
    assert jnp.array_equal(m.fc.weight.value, new_weight)


def test_exported_state_collection_assignment_writes_through_to_live_module():
    """Replacing a live-backed collection syncs matching live leaves."""
    m = TinyModel(Rngs(0))
    _, s = export(m)
    new_weight = jnp.full_like(s["parameters"]["fc"]["weight"], 5.0)
    new_bias = jnp.full_like(s["parameters"]["fc"]["bias"], 6.0)
    s["parameters"] = {"fc": {"weight": new_weight, "bias": new_bias}}
    assert jnp.array_equal(m.fc.weight.value, new_weight)
    assert jnp.array_equal(m.fc.bias.value, new_bias)


def test_exported_state_map_writes_through_to_live_module():
    """Live-backed exported states update module variables on in-place map."""
    m = TinyModel(Rngs(0))
    _, s = export(m)
    out = s.map(lambda path, value: value + 2 if path == "fc.bias" else value, "parameters")
    assert out is s
    assert jnp.array_equal(m.fc.bias.value, s["parameters"]["fc"]["bias"])


def test_exported_state_copy_does_not_write_through():
    """Detached state copies leave the live module untouched."""
    m = TinyModel(Rngs(0))
    _, s = export(m)
    original = m.fc.weight.value
    copied = s.map(lambda value: value + 1, "parameters", copy=True)
    assert copied is not s
    assert jnp.array_equal(m.fc.weight.value, original)


def test_tree_state_returns_only_state():
    """:func:`tree_state` returns just the :class:`State` half."""
    m = TinyModel(Rngs(0))
    assert tree_state(m).paths() == export(m)[1].paths()


def test_live_variables_orders_by_canonical_path():
    """:func:`live_variables` sorts by canonical path for reproducibility."""
    m = TinyModel(Rngs(0))
    paths = [p for p, _ in live_variables(m)]
    assert paths == sorted(paths)


def test_live_variables_dedups_shared():
    """Shared variables appear once in :func:`live_variables`."""
    m = TiedModel(Rngs(0))
    out = live_variables(m)
    assert len({id(v) for _, v in out}) == len(out)


def test_paths_stable_across_repeated_exports():
    """Repeated :func:`export` produces equal :class:`GraphDef` values."""
    m = TinyModel(Rngs(0))
    g1, _ = export(m)
    g2, _ = export(m)
    assert g1 == g2


class ModelWithEmptyModuleList(Module):
    """Module containing an empty ModuleList to exercise edge cases."""

    def __init__(self):
        """Initialize with fc, layers."""
        super().__init__()
        self.fc = Linear(4, 4, rngs=Rngs(0))
        self.layers = spectrax.nn.ModuleList([])

    def forward(self, x):
        """Run the forward pass."""
        return self.fc(x)


def test_export_does_not_crash_on_empty_modulelist():
    """Exporting a model with an empty ``ModuleList`` must not raise."""
    m = ModelWithEmptyModuleList()
    g, s = export(m)
    assert isinstance(g, GraphDef)
    assert "parameters" in s


def test_bind_roundtrip_on_empty_modulelist():
    """Bind after export works even when a ``ModuleList`` has no items."""
    m = ModelWithEmptyModuleList()
    g, s = export(m)
    m2 = bind(g, s)
    x = jnp.ones((2, 4))
    assert jnp.allclose(m(x), m2(x))
