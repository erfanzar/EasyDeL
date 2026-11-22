# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":func:`display` — rich, optionally-interactive rendering of a module.

Uses `treescope` when available (rich HTML in notebooks, ANSI in TTYs)
and falls back to plain :func:`spectrax.inspect.repr.repr_module` text
otherwise.
"""

from __future__ import annotations

from ..core.module import Module
from .repr import repr_module

try:
    import treescope as _treescope
except ImportError:
    _treescope = None

__all__ = ["display"]


def display(module: Module, *, roundtrip: bool = False) -> None:
    """Pretty-print ``module`` to the active output (notebook or stdout).

    When `treescope` is importable, renders the module via
    :func:`treescope.display` (interactive HTML with collapsible nodes
    in a notebook, ANSI in a terminal). When `treescope` is missing or
    raises, falls back to printing
    :func:`~spectrax.inspect.repr.repr_module`.

    Args:
        module: The module to render.
        roundtrip: Forwarded to :func:`treescope.display` as
            ``roundtrip_mode=``. When ``True``, the rendering attempts
            to produce a Python-evaluable repr.

    Returns:
        ``None``. The function exists for its side effect of rendering.
    """
    if _treescope is None:
        print(repr_module(module))
        return
    try:
        _treescope.display(module, roundtrip_mode=roundtrip)
    except Exception:
        print(repr_module(module))
