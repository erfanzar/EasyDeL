# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dynamic-scope context data for spectrax.

Thread values through the call stack without explicit argument passing.
``spx.scope(**values)`` is a context manager that pushes a frame onto
a per-task :mod:`contextvars` stack; ``spx.scope.get(key)`` anywhere
in the enclosed call stack returns the innermost value bound to
``key`` (inner frames shadow outer ones).

Integration with :func:`spectrax.jit`:

* **Static values** (Python scalars, strings, tuples of statics) are
  folded into the jit compile cache key — different static snapshots
  trigger distinct compiles, so conditional branches on a context flag
  specialize correctly.
* **Array values** (``jax.Array``, NumPy arrays, anything with
  ``.shape``) are auto-lifted into the jit input pytree as tracers;
  the traced value is reinstated as a scope frame inside the traced
  body so deep ``spx.scope.get(...)`` calls see tracers rather than
  baked-in constants.

The no-scope hot path costs a single :meth:`contextvars.ContextVar.get`
(~50 ns) so long-running training loops that never touch the scope API
pay essentially nothing.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["get", "partition", "scope", "snapshot"]


_STACK: contextvars.ContextVar[tuple[dict[str, object], ...]] = contextvars.ContextVar(
    "spectrax_scope_stack", default=()
)
"""Stack of scope frames. Outer frames are at lower indices; inner at
higher. :func:`get` walks from inner -> outer so the innermost binding
wins.
"""


_MISSING: object = object()
"""Sentinel distinguishing ``default=None`` from an omitted default."""


@contextmanager
def _enter(values: dict[str, object]) -> Iterator[None]:
    """Push ``values`` onto the scope stack and pop on exit.

    The implementation detail behind :func:`scope`. Uses
    :mod:`contextvars` so the stack is per-task (asyncio-safe) and
    inherited by JAX-traced function bodies.

    Args:
        values: Values consumed by the helper.

    Returns:
        Result described by this helper.
    """
    token = _STACK.set((*_STACK.get(), values))
    try:
        yield
    finally:
        _STACK.reset(token)


def scope(**values: object) -> object:
    """Enter a new scope frame carrying ``values`` until the block exits.

    Usage::

        with spx.scope(mask=attn_mask, is_training=False):
            loss = model(x)

    Inside the block, :func:`get` resolves the bound keys. Nesting is
    supported; inner values shadow outer bindings.

    The call returns a context manager. Calling :func:`scope` without
    keyword arguments is a no-op (pushes an empty frame) — this is
    occasionally useful when you want a cancellation boundary on the
    stack without actually binding anything.

    Args:
        **values: Arbitrary key/value pairs to bind for the duration
            of the context block.

    Returns:
        A context manager that unwinds the scope frame when exited.
    """
    return _enter(values)


def get(key: str, default: object = _MISSING) -> object:
    """Look up ``key`` in the innermost scope frame that binds it.

    Args:
        key: Name to look up.
        default: Value to return if no active frame binds ``key``.
            When omitted, raises :class:`KeyError` on a miss.

    Returns:
        The bound value.

    Raises:
        KeyError: No active frame binds ``key`` and no ``default``
            was supplied.
    """
    stack = _STACK.get()
    for i in range(len(stack) - 1, -1, -1):
        frame = stack[i]
        if key in frame:
            return frame[key]
    if default is _MISSING:
        raise KeyError(
            f"spx.scope.get({key!r}): no active scope binds this key. "
            f"Wrap the call site in `with spx.scope({key}=...):` or pass "
            f"`default=` to get a fallback."
        )
    return default


def snapshot() -> dict[str, object]:
    """Flatten the active scope stack into a single ``{key: value}`` dict.

    Inner frames overwrite outer ones on key collision (matching
    :func:`get` semantics). The returned dict is a fresh copy; mutating
    it does not affect the scope stack.

    Returns:
        A dict containing the merged key/value bindings of every active
        scope frame.
    """
    stack = _STACK.get()
    if not stack:
        return {}
    out: dict[str, object] = {}
    for frame in stack:
        out.update(frame)
    return out


def _is_array_like(v: object) -> bool:
    """Return ``True`` iff ``v`` should be treated as a traced value.

    Traced values get lifted into :func:`spectrax.jit`'s pytree-input
    tuple and materialize inside the traced body as tracers; static
    values get folded into the compile cache key. The discriminator is
    deliberately permissive — anything with a ``shape`` attribute is
    array-like.

    Args:
        v: V value consumed by this operation.

    Returns:
        Return ``True`` iff ``v`` should be treated as a traced value.
    """
    return hasattr(v, "shape") or hasattr(v, "dtype")


def partition(snap: dict[str, object]) -> tuple[dict[str, object], tuple[tuple[str], ...]]:
    """Split a scope snapshot into (traced, static) halves.

    Args:
        snap: Output of :func:`snapshot` (or any equivalent dict).

    Returns:
        A pair ``(traced, static)``:

        * ``traced`` — ``{name: array}`` dict of array-like values to
          flow as pytree inputs into :func:`spectrax.jit`.
        * ``static`` — sorted tuple of ``(name, value)`` pairs of
          hashable-by-value scalars to fold into the compile cache key.
    """
    if not snap:
        return {}, ()
    traced: dict[str, object] = {}
    static: list[tuple[str]] = []
    for k, v in snap.items():
        if _is_array_like(v):
            traced[k] = v
        else:
            static.append((k, v))
    static.sort()
    return traced, tuple(static)


scope.get = get
scope.snapshot = snapshot
scope.partition = partition
