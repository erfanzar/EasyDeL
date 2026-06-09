# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Structural inspection helpers exposing the :class:`~spectrax.State` view.

These helpers wrap the export half of :func:`spectrax.export` for
callers that only need the leaf-by-leaf state tree (rather than the
full ``(GraphDef, State)`` pair).
"""

from __future__ import annotations

from ..core._typing import DType
from ..core.graph import export
from ..core.graph import tree_state as _tree_state
from ..core.module import Module
from ..core.state import State

__all__ = ["paths_and_shapes", "tree_state"]


def tree_state(module: Module) -> State:
    """Return the :class:`~spectrax.State` half of :func:`spectrax.export`.

    A convenience wrapper for callers that only need the state tree
    (e.g. checkpointing, parameter inspection) and don't want the
    accompanying :class:`~spectrax.GraphDef`.

    Args:
        module: The live module to extract state from.

    Returns:
        The :class:`~spectrax.State` of ``module``.
    """
    return _tree_state(module)


def paths_and_shapes(module: Module) -> list[tuple[str, str, tuple[int, ...], DType]]:
    """List every leaf in ``module`` as ``(collection, path, shape, dtype)``.

    Iterates the module's :class:`~spectrax.State` items, captures
    each variable's shape and dtype, and sorts the result by
    ``(collection, path)`` so the output order is stable across runs
    and convenient to diff.

    Args:
        module: The module to introspect.

    Returns:
        A sorted list of ``(collection, path, shape, dtype)`` tuples,
        one per leaf variable.
    """
    _gdef, state = export(module)
    out: list[tuple[str, str, tuple[int, ...], DType]] = []
    for c, p, v in state.items():
        shape = tuple(getattr(v, "shape", ()))
        dtype = getattr(v, "dtype", None)
        out.append((c, p, shape, dtype))
    return sorted(out, key=lambda r: (r[0], r[1]))
