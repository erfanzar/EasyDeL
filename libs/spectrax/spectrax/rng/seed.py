# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Thread-local default-:class:`Rngs` context manager.

Layers optionally fall back to a default :class:`~spectrax.Rngs` when
none is passed explicitly to their constructor (via
:func:`spectrax.rng.resolve_rngs`). ``spectrax.seed(n)`` pushes such a
default onto a thread-local stack so user code can opt into implicit
RNG for the duration of a block without losing the option of explicit
``rngs=`` arguments elsewhere.

The stack is per-thread (backed by :class:`threading.local`), so
multiple worker threads each get an independent default. ``seed()``
contexts may be nested arbitrarily; only the innermost context's
:class:`Rngs` is exposed to :func:`default_rngs`.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

from .rngs import Rngs

__all__ = ["default_rngs", "has_default_rngs", "seed"]


_state = threading.local()


def _stack() -> list[Rngs]:
    """Return the thread-local :class:`Rngs` stack, initializing it lazily.

    Each call returns the same list object for the calling thread, so
    appends and pops are observed across all callers within that thread
    while remaining isolated from other threads.

    Returns:
        Return the thread-local :class:`Rngs` stack, initializing it lazily.
    """
    s = getattr(_state, "stack", None)
    if s is None:
        s = []
        _state.stack = s
    return s


@contextlib.contextmanager
def seed(n: int | Rngs) -> Iterator[Rngs]:
    """Push an :class:`Rngs` as the thread-local default for this block.

    Within the ``with`` block, layers that take an optional ``rngs=``
    argument and call :func:`spectrax.rng.resolve_rngs` will pick up the
    pushed :class:`Rngs` automatically. ``seed`` blocks may be nested;
    the innermost one wins. On block exit (normal or via exception) the
    stack is popped.

    Args:
        n: Either an ``int`` seed (wrapped as ``Rngs(n)``) or an
            existing :class:`Rngs` pushed verbatim.

    Yields:
        The :class:`Rngs` that is active for the duration of the block.
    """
    rngs = n if isinstance(n, Rngs) else Rngs(n)
    stack = _stack()
    stack.append(rngs)
    try:
        yield rngs
    finally:
        stack.pop()


def default_rngs() -> Rngs:
    """Return the current thread-local default :class:`Rngs`.

    The "current" default is the innermost :func:`seed` context on this
    thread.

    Returns:
        The active :class:`Rngs` instance.

    Raises:
        RuntimeError: If no :func:`seed` context is active on this
            thread.
    """
    stack = _stack()
    if not stack:
        raise RuntimeError(
            "No default Rngs active. Pass rngs=... explicitly or wrap the block with `with spectrax.seed(n): ...`."
        )
    return stack[-1]


def has_default_rngs() -> bool:
    """Return ``True`` iff a :func:`seed` context is currently active on this thread.

    Used by :func:`spectrax.rng.resolve_rngs` to decide whether to fall
    back to :func:`default_rngs` or raise.

    Returns:
        Return ``True`` iff a :func:`seed` context is currently active on this thread.
    """
    return bool(_stack())
