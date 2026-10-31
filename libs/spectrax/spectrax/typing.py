# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public type aliases and protocols for spectrax users.

Re-exports from :mod:`spectrax.core._typing`. Import these when typing
code that interacts with the spectrax API so callers and implementers
agree on the shapes of arrays, dtypes, initializers, and hook
callables.
"""

from __future__ import annotations

from .core._typing import (
    Array,
    ArrayLike,
    DType,
    ForwardHook,
    ForwardPreHook,
    Initializer,
    ModulePredicate,
    Path,
    PathComponent,
    PRNGKey,
    PyTree,
    Shape,
    VariableObserver,
    VariablePredicate,
)

__all__ = [
    "Array",
    "ArrayLike",
    "DType",
    "ForwardHook",
    "ForwardPreHook",
    "Initializer",
    "ModulePredicate",
    "PRNGKey",
    "Path",
    "PathComponent",
    "PyTree",
    "Shape",
    "VariableObserver",
    "VariablePredicate",
]
