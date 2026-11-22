# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PyTorch-style pretty-print for :class:`~spectrax.Module` instances.

The format for each module is::

    ClassName(static_field=value, ...)(
      (child_name): <child repr>
      ...
    )

Leaf modules (no children) collapse to a single line. Shared submodules
are printed once; subsequent appearances render as ``<shared>`` to keep
the output finite.
"""

from __future__ import annotations

from ..core.module import Module
from ..core.variable import Variable

__all__ = ["repr_module"]


_INDENT = "  "


def repr_module(module: Module) -> str:
    """Render ``module`` as a PyTorch-style multi-line tree of submodules.

    Static hyperparameters appear in the head ``ClassName(...)`` and
    every nested submodule is printed on its own indented line under
    ``(child_name): ...``. Modules referenced more than once are
    rendered in full on first encounter and as ``ClassName(<shared>)``
    on subsequent visits, keeping the output finite even for
    weight-tied graphs.

    Args:
        module: The module to render.

    Returns:
        A multi-line string suitable for ``print``.
    """
    seen: set[int] = set()
    return _render(module, seen, indent=0)


def _render(module: Module, seen: set[int], indent: int) -> str:
    """Recursive worker for :func:`repr_module`.

    Args:
        module: The module currently being rendered.
        seen: Set of already-rendered ``id(module)`` values; used to
            short-circuit on shared submodules.
        indent: Current indentation depth in units of :data:`_INDENT`.

    Returns:
        The rendered text for ``module`` and all children below it.
    """
    cls = type(module).__name__
    head_args = _head_args(module)
    head = f"{cls}({head_args})"

    mid = id(module)
    if mid in seen:
        return f"{cls}(<shared>)"
    seen.add(mid)

    module_children = [(k, c) for k, c in _iter_children(module) if isinstance(c, Module)]
    if not module_children:
        return head

    pad = _INDENT * (indent + 1)
    lines: list[str] = [f"{cls}("] if not head_args else [f"{cls}({head_args})("]
    for key, child in module_children:
        sub = _render(child, seen, indent + 1)
        lines.append(f"{pad}({key}): {sub}")
    lines.append(f"{_INDENT * indent})")
    return "\n".join(lines)


def _head_args(module: Module) -> str:
    """Render the module's static hyperparameters as ``"k1=v1, k2=v2"``.

    Pulls the static field map via :meth:`Module._spx_static_fields`
    and ``repr`` s each value. Returns the empty string when the
    module exposes no static fields or the lookup raises.

    Args:
        module: The module whose static fields to render.

    Returns:
        A comma-separated ``key=value`` string, or ``""``.
    """
    try:
        static = module._spx_static_fields()
    except Exception:
        return ""
    if not static:
        return ""
    return ", ".join(f"{k}={v!r}" for k, v in static.items())


def _iter_children(module: Module):
    """Yield ``(key, child)`` pairs for every graph child of ``module``.

    Delegates to :meth:`Module._spx_graph_children`; on any exception
    the generator yields nothing rather than propagating, so a partial
    or malformed module still renders.

    Args:
        module: The module whose graph children to iterate.

    Yields:
        ``(key, child)`` tuples in the order returned by the module.
    """
    try:
        yield from module._spx_graph_children()
    except Exception:
        return


def _format_key(key: object) -> str:
    """Wrap a graph-child key in parentheses, PyTorch-style.

    Used by callers that want ``(name)`` / ``(0)`` rather than the bare
    ``name`` / ``0``.

    Args:
        key: The child key (typically a string attribute name or an
            integer list index).

    Returns:
        ``f"({key})"``.
    """
    return f"({key})"


def _render_variable(var: Variable) -> str:
    """Build a one-line repr for a :class:`~spectrax.Variable` leaf.

    Includes the variable subclass name, its collection (``kind``),
    shape, and dtype. Falls back to a kind-only form if the leaf is
    pre-init or otherwise lacks a shape/dtype.

    Args:
        var: The variable to render.

    Returns:
        A single-line repr suitable for embedding in a tree dump.
    """
    try:
        shape = tuple(getattr(var, "shape", ()))
        dtype = getattr(var, "dtype", "")
        return f"{type(var).__name__}(kind={var.kind!r}, shape={shape}, dtype={dtype})"
    except Exception:
        return f"{type(var).__name__}(kind={var.kind!r})"


def _ascii_tree(module: Module, *, indent: int = 0) -> str:
    """Legacy entry point that delegates to :func:`_render` with a fresh ``seen`` set.

    Kept so tests and downstream code that import the private helper
    by name continue to work; new callers should use
    :func:`repr_module` instead.

    Args:
        module: The module to render.
        indent: Initial indentation level.

    Returns:
        The same text :func:`repr_module` would produce starting at
        the given indentation.
    """
    return _render(module, set(), indent)
