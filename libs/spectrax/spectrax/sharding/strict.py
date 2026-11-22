# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Strict sharding mode: turn silent sharding fallbacks into errors.

By default several sharding helpers degrade quietly so the same model code
runs on many mesh shapes: :func:`~spectrax.sharding.with_sharding_constraint`
becomes a no-op when the MPMD stage cannot be resolved, and spec sanitization
drops missing or non-divisible axes. Strict mode raises
:class:`~spectrax.ShardingError` instead, so accidental replication shows up
at trace time rather than as an out-of-memory error later.

Enable it for a block::

    with spx.sharding.strict_sharding():
        ...

or for the whole process with :func:`set_strict_sharding`. The innermost
context manager wins over the process-wide setting.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

__all__ = ["is_strict_sharding", "set_strict_sharding", "strict_sharding"]

_LOCAL = threading.local()
_GLOBAL = False


def is_strict_sharding() -> bool:
    """Return whether strict sharding is active for the current thread."""
    stack = getattr(_LOCAL, "stack", None)
    if stack:
        return stack[-1]
    return _GLOBAL


def set_strict_sharding(enabled: bool) -> None:
    """Set the process-wide default used outside any :func:`strict_sharding` block.

    Args:
        enabled: ``True`` to raise on sharding fallbacks everywhere.
    """
    global _GLOBAL
    _GLOBAL = bool(enabled)


@contextlib.contextmanager
def strict_sharding(enabled: bool = True) -> Iterator[None]:
    """Enable (or, with ``enabled=False``, disable) strict sharding in the block.

    Args:
        enabled: Strict mode for the dynamic extent of the block.

    Yields:
        ``None``.
    """
    stack = getattr(_LOCAL, "stack", None)
    if stack is None:
        stack = []
        _LOCAL.stack = stack
    stack.append(bool(enabled))
    try:
        yield
    finally:
        stack.pop()
