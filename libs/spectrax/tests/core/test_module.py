# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.core.module`."""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import pytest

from spectrax.core.context import get as scope_get
from spectrax.core.module import Module, Opaque
from spectrax.core.policy import Policy
from spectrax.core.static import Static
from spectrax.core.variable import Parameter
from spectrax.transforms.jit import jit


class Leaf(Module):
    """Tiny fixture module with one parameter and one static."""

    def __init__(self):
        """Install one parameter and one static hyperparameter."""
        super().__init__()
        self.w = Parameter(jnp.zeros(3))
        self.hp = 42

    def forward(self, x):
        """Return ``x + self.w.value``."""
        return x + self.w.value


class Parent(Module):
    """Fixture nesting two :class:`Leaf` children."""

    def __init__(self):
        """Install two ``Leaf`` children."""
        super().__init__()
        self.a = Leaf()
        self.b = Leaf()

    def forward(self, x):
        """Chain the two leaves."""
        return self.b(self.a(x))


class ScopeScale(Module):
    """Fixture that reads a multiplier from ``spx.scope``."""

    def forward(self, x):
        """Scale ``x`` by the active ``scale`` context value."""
        return x * scope_get("scale")


def test_init_sets_private_slots():
    """:class:`Module.__init__` initializes all private slots."""
    m = Leaf()
    assert m._spx_attr_order == ["w", "hp"]
    assert "hp" in m._spx_static and m._spx_static["hp"] == 42
    assert m._spx_training is True
    assert m._spx_fwd_hooks == []
    assert m._spx_pre_hooks == []
    assert m._spx_policy is None
    assert m._spx_opaque == {}


def test_setattr_records_attr_order():
    """Attribute assignment order is recorded in ``_spx_attr_order``."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize with x, y, z."""
            super().__init__()
            self.x = 1
            self.y = 2
            self.z = 3

        def forward(self, x):
            """Run the forward pass."""
            return x

    m = M()
    assert m._spx_attr_order == ["x", "y", "z"]


def test_setattr_static_scalar_goes_to_static_dict():
    """Static scalars land in ``_spx_static``."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize with dim, name."""
            super().__init__()
            self.dim = 8
            self.name = "test"

        def forward(self, x):
            """Run the forward pass."""
            return x

    m = M()
    assert m._spx_static == {"dim": 8, "name": "test"}


def test_setattr_module_child_not_in_static():
    """Module children are not recorded as static fields."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize with leaf."""
            super().__init__()
            self.leaf = Leaf()

        def forward(self, x):
            """Run the forward pass."""
            return self.leaf(x)

    m = M()
    assert "leaf" not in m._spx_static


def test_setattr_variable_child():
    """Variable children are in ``_spx_attr_order`` but not ``_spx_static``."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize with p."""
            super().__init__()
            self.p = Parameter(jnp.zeros(1))

        def forward(self, x):
            """Run the forward pass."""
            return x

    m = M()
    assert "p" in m._spx_attr_order
    assert "p" not in m._spx_static


def test_setattr_auto_wraps_non_static_in_opaque():
    """Non-static non-Module non-Variable values are auto-wrapped in :class:`Opaque`."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize the instance."""
            super().__init__()

        def forward(self, x):
            """Run the forward pass."""
            return x

    m = M()
    m.arbitrary = [1, 2, 3]
    assert "arbitrary" in m._spx_opaque
    assert m.arbitrary == [1, 2, 3]
    assert m._spx_opaque["arbitrary"].value == [1, 2, 3]


def test_setattr_hashable_object_defaults_to_opaque():
    """Config-like user objects should stay usable but out of static fields."""

    class Config:
        """Simple configuration object for testing."""

        hidden_size = 128

    class M(Module):
        """Fixture module for testing."""

        def __init__(self, config):
            """Initialize with config."""
            super().__init__()
            self.config = config

        def forward(self, x):
            """Run the forward pass."""
            return x

    config = Config()
    m = M(config)

    assert m.config is config
    assert m.config.hidden_size == 128
    assert "config" in m._spx_opaque
    assert m._spx_opaque["config"].value is config
    assert "config" not in m._spx_static


def test_setattr_private_bypasses_discipline():
    """Names starting with ``_`` are stored without type checks."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize with ."""
            super().__init__()
            self._private = [1, 2, 3]

        def forward(self, x):
            """Run the forward pass."""
            return x

    m = M()
    assert m._private == [1, 2, 3]
    assert "_private" not in m._spx_attr_order


def test_setattr_explicit_static_stores_public_value():
    """``Static(value)`` tracks static metadata while exposing ``value`` publicly."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize with activation."""
            super().__init__()
            self.activation = Static("gelu")

        def forward(self, x):
            """Run the forward pass."""
            return x

    m = M()
    assert m.activation == "gelu"
    assert m._spx_static["activation"] == Static("gelu")


def test_setattr_opaque_escape_hatch():
    """``Opaque(value)`` bypasses the type check and is tracked separately."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize with cb."""
            super().__init__()
            self.cb = Opaque(lambda y: y)

        def forward(self, x):
            """Run the forward pass."""
            return x

    m = M()
    assert "cb" in m._spx_opaque
    assert callable(m.cb)
    assert callable(m._spx_opaque["cb"].value)


def test_module_does_not_override_getattribute():
    """Avoid a broad ``__getattribute__ -> Any`` that hides subclass properties from IDEs."""
    assert Module.__getattribute__ is object.__getattribute__


def test_opaque_repr():
    """:class:`Opaque` has a compact repr showing the wrapped type."""
    o = Opaque(lambda: None)
    assert "Opaque(function)" in repr(o) or "Opaque" in repr(o)


def test_setattr_policy_requires_policy_instance():
    """``.policy`` assignment requires :class:`Policy` or ``None``."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize the instance."""
            super().__init__()

        def forward(self, x):
            """Run the forward pass."""
            return x

    m = M()
    m.policy = Policy(compute_dtype=jnp.bfloat16)
    assert m._spx_policy is not None
    m.policy = None
    assert m._spx_policy is None
    with pytest.raises(TypeError):
        m.policy = "not a policy"


def test_delattr_removes_from_tracking():
    """``__delattr__`` removes entries from attr_order / static / opaque."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize with x."""
            super().__init__()
            self.x = 1

        def forward(self, x):
            """Run the forward pass."""
            return x

    m = M()
    del m.x
    assert "x" not in m._spx_attr_order
    assert "x" not in m._spx_static


def test_graph_children_yields_modules_and_variables():
    """``_spx_graph_children`` yields both sub-modules and variables."""
    m = Parent()
    keys = {k for k, _ in m._spx_graph_children()}
    assert keys == {"a", "b"}


def test_graph_children_yields_in_declaration_order():
    """Children are yielded in declaration order."""
    m = Parent()
    keys = [k for k, _ in m._spx_graph_children()]
    assert keys == ["a", "b"]


def test_static_fields_returns_copy():
    """``_spx_static_fields`` returns a shallow copy."""
    m = Leaf()
    fields = m._spx_static_fields()
    fields["extra"] = 1
    assert "extra" not in m._spx_static


def test_train_eval_toggle():
    """``train()`` / ``eval()`` toggle the ``training`` flag."""
    m = Leaf()
    assert m.training is True
    m.eval()
    assert m.training is False
    m.train()
    assert m.training is True
    m.train(False)
    assert m.training is False


def test_train_propagates_recursively():
    """``train(False)`` on a parent also toggles every child."""
    p = Parent()
    p.eval()
    assert p.training is False
    assert p.a.training is False
    assert p.b.training is False


def test_train_returns_self_for_chaining():
    """``train()`` / ``eval()`` return ``self`` for method chaining."""
    m = Leaf()
    assert m.train() is m
    assert m.eval() is m


def test_call_invokes_forward_and_returns_its_output():
    """``module(x)`` calls :meth:`forward` and returns its output."""
    m = Leaf()
    out = m(jnp.asarray([1.0, 2.0, 3.0]))
    assert jnp.array_equal(out, jnp.asarray([1.0, 2.0, 3.0]))


def test_forward_default_raises():
    """The base-class :meth:`forward` raises :class:`NotImplementedError`."""

    class M(Module):
        """Fixture module for testing."""

        pass

    m = M()
    with pytest.raises(NotImplementedError):
        m(None)


def test_pre_hook_can_rewrite_args():
    """A pre-hook returning ``(args, kwargs)`` overrides the call arguments."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize the instance."""
            super().__init__()

        def forward(self, x):
            """Run the forward pass."""
            return x * 10

    m = M()
    m.register_forward_pre_hook(lambda mod, args, kwargs: ((jnp.asarray(5.0),), kwargs))
    out = m(jnp.asarray(1.0))
    assert int(out) == 50


def test_post_hook_can_replace_output():
    """A post-hook returning non-``None`` replaces the module output."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize the instance."""
            super().__init__()

        def forward(self, x):
            """Run the forward pass."""
            return x

    m = M()
    m.register_forward_hook(lambda mod, a, kw, out: out + 1.0)
    assert float(m(jnp.asarray(0.0))) == 1.0


def test_post_hook_returning_none_is_passthrough():
    """A post-hook returning ``None`` leaves the output unchanged."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize the instance."""
            super().__init__()

        def forward(self, x):
            """Run the forward pass."""
            return x + 1

    m = M()
    calls = []
    m.register_forward_hook(lambda *_: calls.append(1) or None)
    out = m(jnp.asarray(0.0))
    assert int(out) == 1
    assert calls == [1]


def test_hook_handle_remove():
    """:meth:`Handle.remove` detaches a hook from its module's list."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize the instance."""
            super().__init__()

        def forward(self, x):
            """Run the forward pass."""
            return x

    m = M()
    calls = []
    handle = m.register_forward_hook(lambda *_: calls.append(1) or None)
    m(jnp.asarray(0.0))
    handle.remove()
    m(jnp.asarray(0.0))
    assert calls == [1]


def test_hook_handle_remove_twice_is_noop():
    """Removing an already-removed handle does not raise."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize the instance."""
            super().__init__()

        def forward(self, x):
            """Run the forward pass."""
            return x

    m = M()
    handle = m.register_forward_hook(lambda *_: None)
    handle.remove()
    handle.remove()


def test_register_context_scope_values_visible_and_removable():
    """Registered scope values are active for each forward call."""

    class M(Module):
        """Fixture module for testing."""

        def forward(self, x):
            """Run the forward pass."""
            return x + scope_get("bias")

    m = M()
    handle = m.register_context(bias=jnp.asarray(2.0))

    assert float(m(jnp.asarray(3.0))) == 5.0

    handle.remove()
    with pytest.raises(KeyError):
        m(jnp.asarray(3.0))


def test_register_context_enters_keyword_context_manager_each_call():
    """Context-manager keyword values are both entered and scoped."""

    class CounterContext:
        """Fixture class for testing."""

        def __init__(self):
            """Initialize with depth, entered, exited."""
            self.depth = 0
            self.entered = 0
            self.exited = 0

        def __enter__(self):
            """Enter the runtime context."""
            self.depth += 1
            self.entered += 1
            return self

        def __exit__(self, exc_type, exc, tb):
            """Exit the runtime context."""
            self.depth -= 1
            self.exited += 1
            return False

    ctx = CounterContext()

    class M(Module):
        """Fixture module for testing."""

        def forward(self, x):
            """Run the forward pass."""
            assert ctx.depth == 1
            assert scope_get("mesh") is ctx
            return x + ctx.entered

    m = M()
    m.register_context(mesh=ctx)

    assert float(m(jnp.asarray(1.0))) == 2.0
    assert float(m(jnp.asarray(1.0))) == 3.0
    assert ctx.depth == 0
    assert ctx.entered == 2
    assert ctx.exited == 2


def test_register_context_factory_creates_fresh_contexts():
    """Factories let one-shot context managers be recreated per call."""
    events = []

    @contextlib.contextmanager
    def factory():
        """Factory function for context managers."""
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    class M(Module):
        """Fixture module for testing."""

        def forward(self, x):
            """Run the forward pass."""
            return x + len(events)

    m = M()
    m.register_context(factory)

    assert float(m(jnp.asarray(1.0))) == 2.0
    assert float(m(jnp.asarray(1.0))) == 4.0
    assert events == ["enter", "exit", "enter", "exit"]


def test_registered_contexts_survive_pytree_roundtrip():
    """Flatten/unflatten copies call contexts alongside hooks and policy."""
    m = ScopeScale()
    m.register_context(scale=3.0)

    cloned = jax.tree_util.tree_map(lambda leaf: leaf, m)
    assert float(cloned(jnp.asarray(2.0))) == 6.0


def test_registered_contexts_work_under_spx_jit():
    """Call contexts are entered inside spectrax transform bodies."""
    m = ScopeScale()
    m.register_context(scale=4.0)

    out = jit(lambda module, x: module(x))(m, jnp.asarray(2.0))
    assert float(out) == 8.0


def test_output_dtype_policy_casts_result():
    """An ``output_dtype`` on the policy casts the forward output."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize the instance."""
            super().__init__()

        def forward(self, x):
            """Run the forward pass."""
            return jnp.asarray(x, dtype=jnp.float32)

    m = M()
    m.policy = Policy(output_dtype=jnp.bfloat16)
    out = m(jnp.ones(2))
    assert out.dtype == jnp.bfloat16


def test_sow_creates_variable_on_first_call():
    """First ``sow`` creates a sow-slot :class:`Variable`."""
    m = Leaf()
    m.sow("intermediates", "h", jnp.asarray(3.0))
    assert hasattr(m, "sow_intermediates_h")


def test_sow_last_reduce_overwrites():
    """``reduce='last'`` overwrites on subsequent calls."""
    m = Leaf()
    m.sow("intermediates", "h", jnp.asarray(1.0))
    m.sow("intermediates", "h", jnp.asarray(2.0))
    assert float(m.sow_intermediates_h.value) == 2.0


def test_sow_sum_accumulates():
    """``reduce='sum'`` adds to the running value."""
    m = Leaf()
    m.sow("intermediates", "h", jnp.asarray(1.0), reduce="sum")
    m.sow("intermediates", "h", jnp.asarray(2.5), reduce="sum")
    assert float(m.sow_intermediates_h.value) == 3.5


def test_sow_stack_concatenates():
    """``reduce='stack'`` concatenates along a new leading axis."""
    m = Leaf()
    m.sow("intermediates", "h", jnp.asarray([1.0, 2.0]), reduce="stack")
    m.sow("intermediates", "h", jnp.asarray([3.0, 4.0]), reduce="stack")
    assert m.sow_intermediates_h.value.shape == (2, 2)


def test_sow_unknown_reduce_raises():
    """An unknown reduce strategy raises :class:`ValueError`."""
    m = Leaf()
    m.sow("intermediates", "h", jnp.asarray(1.0))
    with pytest.raises(ValueError):
        m.sow("intermediates", "h", jnp.asarray(2.0), reduce="bogus")


def test_init_wraps_int_seed_as_rngs():
    """``init(int)`` wraps the seed into an :class:`Rngs` and attaches it."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize the instance."""
            super().__init__()

        def forward(self, x):
            """Run the forward pass."""
            return x

    from spectrax.rng.rngs import Rngs

    m = M()
    m.init(0)
    assert isinstance(m.rngs, Rngs)


def test_init_replaces_existing_rngs_and_invalidates_graph():
    """A second ``init(seed)`` must not silently keep the first RNG module."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize the instance."""
            super().__init__()

        def forward(self, x):
            """Run the forward pass."""
            return x

    from spectrax.core.graph import export
    from spectrax.rng.rngs import Rngs

    m = M()
    m.init(0)
    _, s0 = export(m)
    m.init(42)
    _, s1 = export(m)

    assert not jnp.array_equal(s0.get("rng", "rngs.default"), s1.get("rng", "rngs.default"))
    assert jnp.array_equal(m.rngs.key(), Rngs(42).key())


def test_init_without_rngs_does_not_attach():
    """``init()`` with ``rngs=None`` does not attach an ``rngs`` attribute."""

    class M(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize the instance."""
            super().__init__()

        def forward(self, x):
            """Run the forward pass."""
            return x

    m = M()
    m.init(None)
    assert not hasattr(m, "rngs")


def test_repr_returns_string():
    """``repr(module)`` returns a non-empty string (treescope or ascii)."""
    m = Leaf()
    repr(m)


def test_training_mode_preserved_through_export_bind():
    """``train(False)`` survives an ``export`` / ``bind`` round-trip."""
    from spectrax.core.graph import bind, export

    m = Parent()
    assert m.training is True
    assert m.a.training is True

    m.eval()
    assert m.training is False
    assert m.a.training is False
    assert m.b.training is False

    g, s = export(m)
    m2 = bind(g, s)

    assert m2.training is False, "root training should be False after bind"
    assert m2.a.training is False, "child training should be False after bind"
    assert m2.b.training is False, "child training should be False after bind"


def test_training_mode_true_preserved_through_export_bind():
    """``train(True)`` also survives an ``export`` / ``bind`` round-trip."""
    from spectrax.core.graph import bind, export

    m = Parent()
    m.eval()
    m.train()
    assert m.training is True

    g, s = export(m)
    m2 = bind(g, s)
    assert m2.training is True
    assert m2.a.training is True


def test_eval_invalidation_bumps_export_cache():
    """Calling ``eval()`` invalidates the cached graph so next export is fresh."""
    from spectrax.core.graph import export

    m = Parent()
    g1, _s1 = export(m)

    m.eval()
    g2, _s2 = export(m)

    root_opaque = dict(g2.nodes[0].opaque)
    assert "_spx_training" in root_opaque
    assert root_opaque["_spx_training"] is False

    root_opaque_1 = dict(g1.nodes[0].opaque)
    assert "_spx_training" not in root_opaque_1


def test_freeze_unfreeze_invalidate_exported_collections():
    """Changing variable kind must invalidate cached GraphDef/state collection names."""
    from spectrax.core.graph import export

    m = Leaf()
    export(m)

    m.freeze()
    _g_frozen, s_frozen = export(m)
    assert "parameters" not in s_frozen.collections()
    assert "buffers" in s_frozen.collections()

    m.unfreeze()
    _g_unfrozen, s_unfrozen = export(m)
    assert "parameters" in s_unfrozen.collections()


def test_module_dict_setitem_invalidates_parent_export_cache():
    """Adding a ModuleDict item after export should be visible on the next export."""
    from spectrax.core.containers import ModuleDict
    from spectrax.core.graph import export

    class HasDict(Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize with layers."""
            super().__init__()
            self.layers = ModuleDict({"a": Leaf()})

        def forward(self, x):
            """Run the forward pass."""
            return self.layers["a"](x)

    m = HasDict()
    export(m)

    m.layers["b"] = Leaf()
    _g, s = export(m)

    assert ("parameters", "layers.b.w") in set(s.paths())
