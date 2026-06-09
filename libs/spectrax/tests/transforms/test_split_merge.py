# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.transforms.split_merge` helpers."""

from __future__ import annotations

import jax.numpy as jnp

from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs
from spectrax.transforms.split_merge import (
    apply_mutations,
    locate_modules,
    make_pure,
    resolve_mutable,
    splice_modules,
    strip_modules,
)


def test_locate_modules_finds_positional_and_keyword():
    """Modules in args and kwargs are both located."""
    m = Linear(4, 4, rngs=Rngs(0))
    refs = locate_modules((m, jnp.ones(1)), {"other": m})
    kinds = [r.kind for r in refs]
    assert "arg" in kinds
    assert "kwarg" in kinds


def test_strip_modules_replaces_with_none():
    """Module entries become ``None`` placeholders."""
    m = Linear(4, 4, rngs=Rngs(0))
    args, kwargs = strip_modules((m, 5), {"k": m, "x": 42})
    assert args[0] is None
    assert args[1] == 5
    assert kwargs["k"] is None
    assert kwargs["x"] == 42


def test_splice_modules_reinserts_modules():
    """``splice`` restores modules into their original slots."""
    m = Linear(4, 4, rngs=Rngs(0))
    refs = locate_modules((m,), {})
    stripped_args, stripped_kwargs = strip_modules((m,), {})
    spliced_args, _ = splice_modules(stripped_args, stripped_kwargs, refs, [m])
    assert spliced_args[0] is m


def test_make_pure_reexports_states():
    """The pure function returns new states after calling the user fn."""
    m = Linear(4, 4, rngs=Rngs(0))
    refs = locate_modules((m,), {})
    states = tuple(r.state for r in refs)
    stripped_args, stripped_kwargs = strip_modules((m,), {})
    pure = make_pure(lambda m: m(jnp.ones((1, 4))), refs)
    out, new_states = pure(states, stripped_args, stripped_kwargs)
    assert out.shape == (1, 4)
    assert len(new_states) == 1


def test_resolve_mutable_none_returns_none():
    """``mutable=None`` resolves to ``None``."""
    assert resolve_mutable(None) is None


def test_resolve_mutable_empty_tuple_returns_none():
    """``()`` and ``[]`` are both treated as "no mutation"."""
    assert resolve_mutable(()) is None
    assert resolve_mutable([]) is None


def test_resolve_mutable_string_becomes_selector():
    """A string is coerced via :func:`as_selector`."""
    sel = resolve_mutable("batch_stats")
    assert sel is not None


def test_apply_mutations_noop_when_nothing_changed():
    """``apply_mutations`` is a no-op when the state is unchanged."""
    m = Linear(4, 4, rngs=Rngs(0))
    refs = locate_modules((m,), {})
    apply_mutations(refs, [refs[0].state], None)
