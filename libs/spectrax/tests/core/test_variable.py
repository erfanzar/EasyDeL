# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.core.variable`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectrax.core.sharding import Sharding
from spectrax.core.stage_assignment import (
    PIPELINE_STAGE_METADATA_KEY,
    assign_stage,
    resolve_stage_rank,
)
from spectrax.core.variable import KINDS_BUILTIN, Buffer, DeferredParameter, Parameter, Variable


def test_variable_default_kind_is_buffers():
    """A bare :class:`Variable` defaults to kind ``'buffers'``."""
    v = Variable(jnp.zeros(3))
    assert v.kind == "buffers"


def test_variable_kind_override():
    """The ``kind=`` keyword wins over the class default."""
    v = Variable(jnp.zeros(3), kind="cache")
    assert v.kind == "cache"


def test_variable_pytree_unflatten_preserves_metadata_and_identity():
    """Standalone variable pytree round-trips preserve metadata and ref identity."""
    v = Variable(jnp.zeros(3), kind="cache", metadata={"sharding": ("tp",), "tie_group": "shared"})
    leaves, treedef = jax.tree_util.tree_flatten(v)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rebuilt.kind == "cache"
    assert rebuilt.metadata == v.metadata
    assert rebuilt.sharding.axis_names == ("tp",)
    assert rebuilt.ref_id == v.ref_id
    rebuilt.add_observer(lambda *_args: None)
    assert len(rebuilt._observers) == 1


def test_deferred_parameter_pytree_roundtrip_preserves_lazy_state():
    """Deferred variables should not lose their private shape/init state through pytrees."""

    def init(_rngs, shape, dtype):
        """Initialization helper."""
        return jnp.ones(shape, dtype)

    v = DeferredParameter((None, 4), init, None, jnp.float32, axis_names=("batch", "hidden"))
    v.resolve_shape((3, 4))
    leaves, treedef = jax.tree_util.tree_flatten(v)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(rebuilt, DeferredParameter)
    assert rebuilt.axis_names == ("batch", "hidden")
    assert rebuilt._deferred_shape_spec == (None, 4)
    assert rebuilt._deferred_resolved_shape == (3, 4)
    assert not rebuilt.is_materialized
    assert rebuilt.value.shape == (3, 4)


def test_variable_allocates_unique_ref_ids():
    """Fresh variables get unique ``ref_id`` values."""
    a = Variable(jnp.zeros(()))
    b = Variable(jnp.zeros(()))
    assert a.ref_id != b.ref_id


def test_variable_explicit_ref_id_preserved():
    """An explicit ``ref_id`` is adopted verbatim."""
    v = Variable(jnp.zeros(()), ref_id=999)
    assert v.ref_id == 999


def test_variable_metadata_deep_copied():
    """Constructor takes a shallow copy of the metadata dict."""
    meta = {"tag": 1}
    v = Variable(jnp.zeros(()), metadata=meta)
    meta["tag"] = 2
    assert v.metadata["tag"] == 1


def test_assign_stage_stamps_stage_owned_variable_kinds():
    """Active ``assign_stage`` metadata is inherited by stage-owned Variables."""
    with assign_stage(total=8, current=3):
        v = Variable(jnp.zeros(()))
        p = Parameter(jnp.zeros(()))
        b = Buffer(jnp.zeros(()))

    assert v.metadata[PIPELINE_STAGE_METADATA_KEY] == (3, 8)
    assert p.metadata[PIPELINE_STAGE_METADATA_KEY] == (3, 8)
    assert b.metadata[PIPELINE_STAGE_METADATA_KEY] == (3, 8)
    assert p.stage_assignment == (3, 8)
    assert p.stage_index == 3
    assert p.stage_count == 8
    assert resolve_stage_rank(p.metadata[PIPELINE_STAGE_METADATA_KEY], 4) == 1
    assert p.resolved_stage_index(4) == 1


def test_value_read_and_write():
    """``value`` round-trips through the reference cell."""
    v = Variable(jnp.zeros(2))
    v.value = jnp.ones(2)
    assert jnp.array_equal(v.value, jnp.ones(2))


def test_observer_fires_on_write():
    """A registered observer receives ``(var, old, new)`` on each write."""
    v = Variable(jnp.zeros(2))
    seen: list[tuple] = []
    v.add_observer(lambda var, old, new: seen.append((id(var), tuple(old), tuple(new))))
    v.value = jnp.ones(2)
    assert len(seen) == 1
    assert seen[0][0] == id(v)
    assert seen[0][1] == (0.0, 0.0)
    assert seen[0][2] == (1.0, 1.0)


def test_observer_exceptions_are_swallowed():
    """An observer raising does not propagate out of the write path."""
    v = Variable(jnp.zeros(()))

    def bad(_var, _old, _new):
        """Bad input helper."""
        raise RuntimeError("boom")

    v.add_observer(bad)
    v.value = jnp.ones(())
    assert jnp.array_equal(v.value, jnp.ones(()))


def test_observer_removal():
    """Removing an observer stops it firing."""
    v = Variable(jnp.zeros(()))
    calls: list[int] = []

    def fn(_var, _o, _n):
        """Helper function."""
        return calls.append(1)

    v.add_observer(fn)
    v.remove_observer(fn)
    v.value = jnp.ones(())
    assert calls == []


def test_observer_remove_when_absent_noop():
    """Removing an unregistered observer is a no-op (no error)."""
    v = Variable(jnp.zeros(()))
    v.remove_observer(lambda *_: None)


def test_raw_get_bypasses_read_hook():
    """``_raw_get`` returns the underlying value regardless of hooks."""
    v = Variable(jnp.ones(3))
    assert jnp.array_equal(v._raw_get(), jnp.ones(3))


def test_raw_set_bypasses_observers():
    """``_raw_set`` writes the value without invoking observers."""
    v = Variable(jnp.zeros(()))
    calls: list[int] = []
    v.add_observer(lambda *_: calls.append(1))
    v._raw_set(jnp.ones(()))
    assert calls == []
    assert jnp.array_equal(v.value, jnp.ones(()))


def test_array_protocol_via_numpy():
    """``np.asarray(var)`` yields the underlying array."""
    v = Variable(jnp.asarray([1.0, 2.0, 3.0]))
    assert np.allclose(np.asarray(v), [1.0, 2.0, 3.0])


def test_array_protocol_with_dtype_cast():
    """``np.asarray(var, dtype=...)`` casts the value."""
    v = Variable(jnp.asarray([1.0, 2.0]))
    out = np.asarray(v, dtype=np.float16)
    assert out.dtype == np.float16


def test_jax_array_protocol():
    """``jnp.asarray(var)`` still works via ``__array__`` fallback."""
    v = Variable(jnp.arange(4))
    assert jnp.array_equal(jnp.asarray(v), v.value)


def test_shape_dtype_ndim_size_properties():
    """Shape / dtype / ndim / size delegate to the stored array."""
    v = Variable(jnp.zeros((2, 3), dtype=jnp.float32))
    assert v.shape == (2, 3)
    assert v.dtype == jnp.float32
    assert v.ndim == 2
    assert v.size == 6


def test_astype_returns_raw_array():
    """``astype`` returns a plain array (not a :class:`Variable`)."""
    v = Variable(jnp.ones(3))
    out = v.astype(jnp.int32)
    assert not isinstance(out, Variable)
    assert out.dtype == jnp.int32


def test_unary_ops_delegate_to_value():
    """``+var``, ``-var``, ``abs(var)`` return arrays of correct sign."""
    v = Variable(jnp.asarray([-1.0, 2.0]))
    assert jnp.array_equal(+v, v.value)
    assert jnp.array_equal(-v, -v.value)
    assert jnp.array_equal(abs(v), jnp.abs(v.value))


def test_binary_ops_variable_and_scalar():
    """Arithmetic between a Variable and a scalar uses ``.value``."""
    v = Variable(jnp.asarray([1.0, 2.0, 3.0]))
    assert jnp.array_equal(v + 1.0, v.value + 1.0)
    assert jnp.array_equal(1.0 + v, 1.0 + v.value)
    assert jnp.array_equal(v * 2.0, v.value * 2.0)
    assert jnp.array_equal(2.0 * v, 2.0 * v.value)
    assert jnp.array_equal(v - 0.5, v.value - 0.5)
    assert jnp.array_equal(10.0 - v, 10.0 - v.value)
    assert jnp.array_equal(v / 2.0, v.value / 2.0)
    assert jnp.array_equal(6.0 / v, 6.0 / v.value)


def test_binary_ops_variable_and_variable():
    """Arithmetic between two Variables uses their ``.value`` pair."""
    a = Variable(jnp.asarray([1.0, 2.0]))
    b = Variable(jnp.asarray([3.0, 4.0]))
    assert jnp.array_equal(a + b, jnp.asarray([4.0, 6.0]))
    assert jnp.array_equal(a * b, jnp.asarray([3.0, 8.0]))


def test_floor_mod_pow():
    """Floor-div / mod / pow delegate correctly."""
    v = Variable(jnp.asarray([5.0, 7.0]))
    assert jnp.array_equal(v // 2.0, jnp.asarray([2.0, 3.0]))
    assert jnp.array_equal(v % 2.0, jnp.asarray([1.0, 1.0]))
    assert jnp.array_equal(v**2, jnp.asarray([25.0, 49.0]))
    assert jnp.array_equal(2.0**v, 2.0**v.value)


def test_matmul_and_rmatmul():
    """``@`` delegates in both directions."""
    w = Variable(jnp.arange(6.0).reshape(2, 3))
    x = jnp.ones((3,))
    y = jnp.ones((2,))
    assert jnp.array_equal(w @ x, w.value @ x)
    assert jnp.array_equal(y @ w, y @ w.value)


def test_getitem_indexes_value():
    """``var[i]`` indexes the stored array."""
    v = Variable(jnp.arange(5))
    assert int(v[2]) == 2
    assert jnp.array_equal(v[1:3], jnp.asarray([1, 2]))


def test_equality_on_ref_id():
    """Two :class:`Variable` s are equal iff their ``ref_id`` matches."""
    a = Variable(jnp.zeros(()))
    b = Variable(jnp.zeros(()), ref_id=a.ref_id)
    c = Variable(jnp.zeros(()))
    assert a == b
    assert a != c


def test_equality_with_non_variable_returns_notimplemented():
    """Comparing with a non-Variable hits ``NotImplemented`` (so ``==`` is ``False``)."""
    v = Variable(jnp.zeros(()))
    assert (v == 42) is False


def test_variables_are_hashable():
    """Variables work as dict keys / set members."""
    a = Variable(jnp.zeros(()))
    d = {a: "ok"}
    assert d[a] == "ok"


def test_variable_bool_delegates():
    """``bool(var)`` delegates to ``bool(var.value)`` for scalars."""
    assert bool(Variable(jnp.asarray(True))) is True
    assert bool(Variable(jnp.asarray(False))) is False


def test_variable_repr_has_shape_and_dtype():
    """The default repr includes class name, kind, shape, dtype, ref id."""
    v = Variable(jnp.zeros((2, 2), dtype=jnp.float32))
    r = repr(v)
    assert "Variable" in r and "kind='buffers'" in r and "shape=(2, 2)" in r


def test_variable_repr_falls_back_on_failure():
    """If the stored value has no ``.shape``, repr still works."""
    v = Variable(None)
    assert "Variable" in repr(v)


def test_parameter_default_kind_is_parameters():
    """A :class:`Parameter` defaults to kind ``'parameters'``."""
    p = Parameter(jnp.zeros(3))
    assert p.kind == "parameters"


def test_parameter_non_trainable_routes_to_buffers():
    """``trainable=False`` routes the :class:`Parameter` into ``'buffers'``."""
    p = Parameter(jnp.zeros(3), trainable=False)
    assert p.kind == "buffers"


def test_parameter_dtype_coercion():
    """``dtype=`` casts the stored array."""
    p = Parameter(jnp.zeros(3, dtype=jnp.float32), dtype=jnp.float16)
    assert p.value.dtype == jnp.float16


def test_parameter_axis_names_stored_in_metadata():
    """``axis_names=`` land in :attr:`Variable.metadata` and on :attr:`axis_names`."""
    p = Parameter(jnp.zeros((4, 8)), axis_names=("in", "out"))
    assert p.metadata["axis_names"] == ("in", "out")
    assert p.axis_names == ("in", "out")


def test_parameter_compound_axis_names_stored_in_metadata():
    """Compound ``axis_names`` entries are preserved for sharding consumers."""
    p = Parameter(jnp.zeros((4, 8)), axis_names=(("fsdp", "sp"), "tp"))
    assert p.metadata["axis_names"] == (("fsdp", "sp"), "tp")
    assert p.axis_names == (("fsdp", "sp"), "tp")


def test_parameter_sharding_as_tuple():
    """Sharding passed as a tuple is normalized to a :class:`Sharding`."""
    p = Parameter(jnp.zeros((4, 8)), sharding=("dp", "mp"))
    assert isinstance(p.sharding, Sharding)
    assert p.sharding.axis_names == ("dp", "mp")


def test_parameter_sharding_as_object():
    """A :class:`Sharding` object is stored as-is."""
    s = Sharding(axis_names=("dp",))
    p = Parameter(jnp.zeros(4), sharding=s)
    assert p.sharding is s


def test_buffer_default_kind_and_explicit_override():
    """:class:`Buffer` defaults to ``'buffers'``; ``kind=`` overrides."""
    b1 = Buffer(jnp.zeros(3))
    b2 = Buffer(jnp.zeros(3), kind="batch_stats")
    assert b1.kind == "buffers"
    assert b2.kind == "batch_stats"


def test_buffer_with_axis_names():
    """:class:`Buffer` also records axis-name metadata."""
    b = Buffer(jnp.zeros(4), axis_names=("out",))
    assert b.metadata["axis_names"] == ("out",)


def test_buffer_sharding_none_returns_none():
    """``Variable.sharding`` returns ``None`` when no sharding is set."""
    b = Buffer(jnp.zeros(3))
    assert b.sharding is None
    assert b.axis_names is None


def test_kinds_builtin_is_tuple_of_strings():
    """:data:`KINDS_BUILTIN` exposes the reserved collection names."""
    assert isinstance(KINDS_BUILTIN, tuple)
    for k in ("parameters", "buffers", "batch_stats", "cache", "intermediates", "rng"):
        assert k in KINDS_BUILTIN


@pytest.mark.parametrize(
    "factory",
    [
        lambda: jnp.zeros(3),
        lambda: np.zeros(3),
        lambda: [1.0, 2.0, 3.0],
        lambda: (1.0, 2.0, 3.0),
        lambda: 1.5,
    ],
)
def test_parameter_accepts_various_value_shapes(factory):
    """Parameters construct from arrays, lists, tuples, and Python scalars."""
    p = Parameter(factory())
    assert p.value is not None


def test_variable_jax_array_protocol():
    """Variable implements ``__jax_array__`` so it works with ``jnp.split`` et al."""
    p = Parameter(jnp.arange(6.0))
    a, b, c = jnp.split(p, 3)
    assert jnp.allclose(a, jnp.array([0.0, 1.0]))
    assert jnp.allclose(b, jnp.array([2.0, 3.0]))
    assert jnp.allclose(c, jnp.array([4.0, 5.0]))


def test_variable_jax_array_inside_jit():
    """``__jax_array__`` works inside a jitted function when PyTree registration is active."""
    p = Parameter(jnp.arange(6.0))

    @jax.jit
    def f(x):
        """Helper function."""
        a, b, c = jnp.split(x, 3)
        return a + b + c

    result = f(p)
    expected = jnp.array([6.0, 9.0])
    assert jnp.allclose(result, expected)


def test_parameter_subclass_inherits_jax_array():
    """Subclasses of Variable (created after import) also get ``__jax_array__``."""

    class MyParam(Parameter):
        """Fixture class for testing."""

        pass

    p = MyParam(jnp.ones(4))
    assert jnp.allclose(jnp.sum(p), 4.0)


def test_variable_getattr_delegates_transpose():
    """``var.T`` delegates to the underlying array."""
    p = Parameter(jnp.arange(6.0).reshape(2, 3))
    assert p.T.shape == (3, 2)


def test_variable_getattr_delegates_reshape():
    """``var.reshape(...)`` delegates to the underlying array."""
    p = Parameter(jnp.arange(6.0))
    assert jnp.allclose(p.reshape(2, 3), jnp.arange(6.0).reshape(2, 3))


def test_variable_getattr_delegates_swapaxes():
    """``var.swapaxes(...)`` delegates to the underlying array."""
    p = Parameter(jnp.arange(6.0).reshape(1, 2, 3))
    assert p.swapaxes(0, 2).shape == (3, 2, 1)


def test_variable_getattr_private_raises():
    """Private attributes raise AttributeError instead of delegating."""
    p = Parameter(jnp.ones(3))
    with pytest.raises(AttributeError):
        _ = p._nonexistent
