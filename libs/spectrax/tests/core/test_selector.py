# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.core.selector`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from spectrax.core.errors import SelectorError
from spectrax.core.selector import Selector, as_selector, select
from spectrax.core.state import State
from spectrax.core.variable import Buffer, Parameter
from spectrax.nn.linear import Linear
from spectrax.nn.norm import BatchNorm1d
from spectrax.rng.rngs import Rngs


class SelectorFixture:
    """Container producing a small canonical module for selector tests."""

    def __init__(self):
        """Build a module with a Linear + a BatchNorm1d layer."""
        from spectrax.core.module import Module

        class Net(Module):
            """Tiny net: one ``Linear`` and one ``BatchNorm1d``."""

            def __init__(self, *, rngs):
                """Create the children."""
                super().__init__()
                self.fc = Linear(4, 8, rngs=rngs)
                self.bn = BatchNorm1d(8)

            def forward(self, x, **_):
                """Apply fc -> bn."""
                return self.bn(self.fc(x))

        self.model = Net(rngs=Rngs(0))


@pytest.fixture
def fixture():
    """Return a fresh :class:`SelectorFixture`."""
    return SelectorFixture()


def test_select_returns_empty_selector():
    """:func:`select` returns a plain empty :class:`Selector`."""
    s = select()
    assert isinstance(s, Selector)
    assert s == Selector()


def test_at_instances_of_narrows_to_class(fixture):
    """``at_instances_of(Linear)`` picks only Linear descendants' variables."""
    matches = select().at_instances_of(Linear).apply(fixture.model)
    for path, _ in matches:
        assert path.startswith("fc.")


def test_variables_narrows_to_collection(fixture):
    """``variables('parameters')`` keeps only ``parameters`` variables."""
    matches = select().variables("parameters").apply(fixture.model)
    assert all(v.kind == "parameters" for _, v in matches)
    assert matches


def test_exclude_variables_drops_collection(fixture):
    """``exclude_variables`` removes matching kinds."""
    matches = select().exclude_variables("batch_stats").apply(fixture.model)
    for _, v in matches:
        assert v.kind != "batch_stats"


def test_at_path_single_segment_glob(fixture):
    """``at_path('fc.*')`` matches direct children of ``fc``."""
    matches = select().at_path("fc.*").apply(fixture.model)
    assert all(p.count(".") == 1 and p.startswith("fc.") for p, _ in matches)


def test_at_path_double_star_glob(fixture):
    """``at_path('**')`` matches every variable."""
    matches_all = select().apply(fixture.model)
    matches_star = select().at_path("**").apply(fixture.model)
    assert {p for p, _ in matches_star} == {p for p, _ in matches_all}


def test_at_path_exact_match(fixture):
    """An exact glob matches only that path."""
    matches = select().at_path("fc.weight").apply(fixture.model)
    assert [p for p, _ in matches] == ["fc.weight"]


def test_where_module_predicate(fixture):
    """``where_module`` filters modules by a custom predicate."""
    matches = select().where_module(lambda m, p: isinstance(m, Linear)).apply(fixture.model)
    for p, _ in matches:
        assert p.startswith("fc.")


def test_where_variable_predicate(fixture):
    """``where_variable`` filters variables by a custom predicate."""
    matches = select().where_variable(lambda v, p: p.endswith("weight")).apply(fixture.model)
    assert all(p.endswith("weight") for p, _ in matches)


def test_where_general_applies_to_both():
    """``where`` applies the same predicate to modules and variables."""
    sel = select().where(lambda target, path: "fc" in path)
    assert isinstance(sel, Selector)


def test_invert_negates_variable_match(fixture):
    """``~selector`` inverts the variable-match decision."""
    sel = select().variables("parameters")
    normal = {p for p, _ in sel.apply(fixture.model)}
    inverted = {p for p, _ in (~sel).apply(fixture.model)}
    assert normal.isdisjoint(inverted)


def test_or_union_combines_matches(fixture):
    """``a | b`` matches variables matched by either operand."""
    a = select().variables("parameters")
    b = select().variables("batch_stats")
    union = a | b
    ids = {(p, v.kind) for p, v in union.apply(fixture.model)}
    assert any(k == "parameters" for _, k in ids)
    assert any(k == "batch_stats" for _, k in ids)


def test_apply_rejects_non_module():
    """``apply`` raises :class:`TypeError` on non-Module input."""
    with pytest.raises(TypeError):
        select().apply("not a module")


def test_apply_deduplicates_shared_variables():
    """Shared :class:`Variable` s are reported once by canonical path."""
    from spectrax.core.module import Module

    class Tied(Module):
        """Module with a tied parameter."""

        def __init__(self):
            """Initialize with a, b."""
            super().__init__()
            self.a = Parameter(jnp.zeros(2))
            self.b = self.a

        def forward(self, x):
            """Run the forward pass."""
            return x

    m = Tied()
    matches = select().apply(m)
    refs = [v.ref_id for _, v in matches]
    assert len(refs) == len(set(refs))


def test_partition_state_splits_by_selector(fixture):
    """``partition_state`` splits by ``(collection, path)`` match."""
    from spectrax.core.graph import export

    _, state = export(fixture.model)
    matched, rest = select().variables("parameters").partition_state(fixture.model, state)
    assert "parameters" in matched
    assert "parameters" not in rest
    assert "batch_stats" in rest


def test_partition_state_covers_every_leaf(fixture):
    """Every leaf in the source state lives in exactly one partition."""
    from spectrax.core.graph import export

    _, state = export(fixture.model)
    matched, rest = select().variables("parameters").partition_state(fixture.model, state)
    total = {(c, p) for c, p in state.paths()}
    combined = {(c, p) for c, p in matched.paths()} | {(c, p) for c, p in rest.paths()}
    assert total == combined


def test_set_writes_to_every_match(fixture):
    """``set`` writes ``fn(var)`` to every matched variable."""
    sel = select().variables("parameters")
    sel.set(fixture.model, lambda v: jnp.zeros_like(v.value))
    for _, v in sel.apply(fixture.model):
        assert jnp.array_equal(v.value, jnp.zeros_like(v.value))


def test_as_selector_none():
    """``None`` sugar yields a "match nothing" selector."""
    s = as_selector(None)
    assert isinstance(s, Selector)


def test_as_selector_returns_selector_unchanged():
    """An existing :class:`Selector` passes through."""
    s = select().variables("parameters")
    assert as_selector(s) is s


def test_as_selector_string_sugar():
    """A string is treated as a collection name."""
    s = as_selector("parameters")
    assert s.variable_kinds == ("parameters",)


def test_as_selector_iterable_of_strings():
    """An iterable of strings becomes a collection-kind filter."""
    s = as_selector(["parameters", "batch_stats"])
    assert set(s.variable_kinds) == {"parameters", "batch_stats"}


def test_as_selector_callable_becomes_variable_predicate():
    """A callable becomes a variable predicate."""

    def pred(v, p):
        """Predicate helper."""
        return True

    s = as_selector(pred)
    assert s.variable_where == (pred,)


def test_as_selector_rejects_invalid_input():
    """Unrecognized inputs raise :class:`SelectorError`."""
    with pytest.raises(SelectorError):
        as_selector(42)


def test_selector_predicate_exception_wraps_as_selector_error(fixture):
    """If a predicate raises, the error is wrapped in :class:`SelectorError`."""

    def bad(_v, _p):
        """Bad input helper."""
        raise RuntimeError("oops")

    with pytest.raises(SelectorError):
        select().where_variable(bad).apply(fixture.model)


def test_empty_selector_matches_every_variable(fixture):
    """A fully-empty :class:`Selector` matches every variable."""
    matches = select().apply(fixture.model)
    for _, v in matches:
        assert isinstance(v, (Parameter, Buffer))
    assert matches


def test_partition_state_returns_two_states(fixture):
    """``partition_state`` always returns two :class:`State` objects."""
    from spectrax.core.graph import export

    _, state = export(fixture.model)
    matched, rest = select().variables("parameters").partition_state(fixture.model, state)
    assert isinstance(matched, State) and isinstance(rest, State)
