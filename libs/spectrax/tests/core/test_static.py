# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.core.static`."""

from __future__ import annotations

from spectrax.core.static import Static, is_static_scalar


def test_static_stores_value():
    """``Static(x).value`` returns the wrapped value."""
    s = Static("gelu")
    assert s.value == "gelu"


def test_static_repr():
    """``repr(Static(x))`` has the expected form."""
    assert repr(Static(3)) == "Static(3)"
    assert repr(Static("gelu")) == "Static('gelu')"


def test_static_equality_structural():
    """Two ``Static`` markers are equal iff their values are."""
    assert Static(1) == Static(1)
    assert Static("a") == Static("a")
    assert Static(1) != Static(2)


def test_static_not_equal_to_bare_value():
    """``Static(x) != x`` — the wrapper is a distinct type."""
    assert Static(1) != 1
    assert Static("a") != "a"


def test_static_hashable():
    """``Static`` instances are hashable and usable in sets/dicts."""
    s = {Static(1), Static(2), Static(1)}
    assert len(s) == 2
    d = {Static("k"): "v"}
    assert d[Static("k")] == "v"


def test_static_nested_equality():
    """Nested ``Static`` markers compare structurally."""
    assert Static(Static(1)) == Static(Static(1))


def test_is_static_scalar_none():
    """``None`` is a static scalar."""
    assert is_static_scalar(None)


def test_is_static_scalar_primitives():
    """Python primitive numerics / strings are static scalars."""
    for v in (True, 0, 1, 3.14, 2j, "hello", b"bytes"):
        assert is_static_scalar(v)


def test_is_static_scalar_static_marker():
    """A :class:`Static` is itself a static scalar."""
    assert is_static_scalar(Static("x"))


def test_is_static_scalar_tuple_of_primitives():
    """Tuples of primitives are static scalars."""
    assert is_static_scalar(("a", 1, 2.0, None))


def test_is_static_scalar_nested_tuple():
    """Nested tuples of primitives are static scalars."""
    assert is_static_scalar(("a", (1, 2), (None, (3,))))


def test_is_static_scalar_frozenset_of_primitives():
    """Frozensets of primitives are static scalars."""
    assert is_static_scalar(frozenset({1, 2, 3}))


def test_is_static_scalar_rejects_list():
    """Lists are mutable, hence not static."""
    assert not is_static_scalar([1, 2, 3])


def test_is_static_scalar_rejects_dict():
    """Dicts are mutable, hence not static."""
    assert not is_static_scalar({"a": 1})


def test_is_static_scalar_rejects_set():
    """Non-frozen sets are mutable, hence not static."""
    assert not is_static_scalar({1, 2, 3})


def test_is_static_scalar_rejects_arbitrary_object():
    """Arbitrary user objects are not static."""

    class Bag:
        """Fixture class for testing."""

        pass

    assert not is_static_scalar(Bag())


def test_is_static_scalar_tuple_with_list_is_not_static():
    """A tuple containing a non-static element is not static."""
    assert not is_static_scalar(("a", [1]))
