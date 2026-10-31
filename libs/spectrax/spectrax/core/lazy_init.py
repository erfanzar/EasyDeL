# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicit lazy-initialization controls."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

__all__ = ["lazy_init"]


_LAZY_STATE: threading.local = threading.local()


def _explicit_lazy_mode() -> bool:
    """Return ``True`` when construction happens under :func:`lazy_init`.

    Returns:
        Return ``True`` when construction happens under :func:`lazy_init`.
    """
    return bool(getattr(_LAZY_STATE, "explicit_lazy", False))


def _materialization_allowed() -> bool:
    """Return ``True`` when lazy modules may materialize on this thread.

    Returns:
        Return ``True`` when lazy modules may materialize on this thread.
    """
    return bool(getattr(_LAZY_STATE, "allow_materialization", False))


@contextlib.contextmanager
def lazy_init() -> Iterator[None]:
    """Build modules in explicit-lazy mode.

    Lazy-capable modules created under this context do not silently
    materialize on first forward. Call ``module.sequential_init(...)``
    (or ``module.init(...example_inputs...)``) to materialize them
    explicitly.

    Yields:
        Control passes to the caller's ``with`` body.
    """
    prev = _explicit_lazy_mode()
    _LAZY_STATE.explicit_lazy = True
    try:
        yield
    finally:
        _LAZY_STATE.explicit_lazy = prev


@contextlib.contextmanager
def _allow_materialization() -> Iterator[None]:
    """Temporarily allow lazy modules to materialize.

    Returns:
        Result described by this helper.
    """
    prev = _materialization_allowed()
    _LAZY_STATE.allow_materialization = True
    try:
        yield
    finally:
        _LAZY_STATE.allow_materialization = prev
