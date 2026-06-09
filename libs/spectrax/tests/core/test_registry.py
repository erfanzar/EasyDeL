# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.core.registry`."""

from __future__ import annotations

import pytest

from spectrax.core.module import Module
from spectrax.core.registry import qualified_name, resolve_class
from spectrax.core.variable import Parameter, Variable


class OuterFixture(Module):
    """Module-level fixture used to verify qualified-name resolution."""

    class Inner:
        """Nested class used to exercise dotted qualnames."""


NONE_FIXTURE = None


def test_qualified_name_builtin_class():
    """``qualified_name`` returns ``module.Qualname`` for a known class."""
    assert qualified_name(Variable) == "spectrax.core.variable.Variable"
    assert qualified_name(Parameter) == "spectrax.core.variable.Parameter"


def test_qualified_name_module_fixture():
    """A module-level class in this file resolves to its import path."""
    assert qualified_name(OuterFixture) == "tests.core.test_registry.OuterFixture"


def test_qualified_name_nested_class():
    """A nested class retains its dotted qualname."""
    assert qualified_name(OuterFixture.Inner) == "tests.core.test_registry.OuterFixture.Inner"


def test_resolve_class_builtin():
    """``resolve_class`` round-trips a built-in spectrax class."""
    cls = resolve_class("spectrax.core.variable.Variable")
    assert cls is Variable


def test_resolve_class_roundtrip_module_level():
    """A module-level fixture class round-trips through the resolver."""
    cls = resolve_class(qualified_name(OuterFixture))
    assert cls is OuterFixture


def test_resolve_class_roundtrip_nested():
    """A nested fixture class round-trips through the resolver."""
    cls = resolve_class(qualified_name(OuterFixture.Inner))
    assert cls is OuterFixture.Inner


def test_resolve_class_rejects_bare_name():
    """A name with no dot (no module) raises :class:`ImportError`."""
    with pytest.raises(ImportError):
        resolve_class("Variable")


def test_resolve_class_rejects_missing_module():
    """A name whose module cannot be imported raises."""
    with pytest.raises(ImportError):
        resolve_class("no_such.module.Thing")


def test_resolve_class_rejects_missing_attr():
    """A name whose attribute is missing raises :class:`ImportError`."""
    with pytest.raises(ImportError):
        resolve_class("spectrax.core.variable.Nonexistent")


def test_resolve_class_rejects_non_class_object():
    """Resolving a non-class object (e.g. a function) raises :class:`TypeError`."""
    with pytest.raises(TypeError):
        resolve_class("spectrax.core.registry.resolve_class")


def test_resolve_class_none_attribute_is_resolved_then_rejected_as_non_class():
    """A real ``None`` attribute should not be reported as missing."""
    with pytest.raises(TypeError):
        resolve_class("tests.core.test_registry.NONE_FIXTURE")
