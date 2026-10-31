# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dtype policy describing parameter storage and compute precision.

A :class:`Policy` attached to a module via ``module.policy = Policy(...)``
controls three orthogonal dtypes for that subtree:

* ``param_dtype`` — the dtype parameters are stored as.
* ``compute_dtype`` — the dtype reads are cast to before use (enabling
  bfloat16 compute with fp32 storage, for example).
* ``output_dtype`` — the dtype the module's output is cast to, applied
  automatically in :meth:`Module.__call__`.

All three default to ``None`` (meaning "don't touch"); combine them to
implement mixed-precision regimes without touching layer code.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import jax.numpy as jnp

from ._typing import Array, ArrayLike, DType

__all__ = ["Policy", "current_policy", "push_policy"]


_POLICY_STACK: threading.local = threading.local()


def _get_stack() -> list["Policy"]:
    """Return (or create) the thread-local policy stack.

    Returns:
        Return (or create) the thread-local policy stack.
    """
    s = getattr(_POLICY_STACK, "stack", None)
    if s is None:
        s = []
        _POLICY_STACK.stack = s
    return s


def current_policy() -> "Policy | None":
    """Return the innermost active :class:`Policy`, or ``None``.

    Returns:
        The currently-enforced dtype policy, or ``None`` when no policy
        has been pushed.
    """
    s = _get_stack()
    return s[-1] if s else None


@contextlib.contextmanager
def push_policy(policy: "Policy | None") -> Iterator[None]:
    """Push ``policy`` onto the stack for the body, popping on exit.

    ``None`` is a no-op (so callers can guardlessly push the result of
    ``module._spx_policy``).

    Args:
        policy: The :class:`Policy` to activate, or ``None`` for a no-op.

    Yields:
        Control passes to the caller's ``with`` body.
    """
    if policy is None:
        yield
        return
    stack = _get_stack()
    stack.append(policy)
    try:
        yield
    finally:
        stack.pop()


@dataclass(frozen=True)
class Policy:
    """Immutable dtype policy for a module subtree.

    Attributes:
        param_dtype: Storage dtype for parameters. ``None`` means "use the
          layer's declared dtype".
        compute_dtype: Dtype inputs are cast to inside layer forwards.
          ``None`` means "no cast".
        output_dtype: Dtype applied to the forward output in
          :meth:`Module.__call__`. ``None`` means "no cast".
    """

    param_dtype: DType | None = None
    compute_dtype: DType | None = None
    output_dtype: DType | None = None

    def cast_param(self, x: ArrayLike) -> Array:
        """Cast ``x`` to :attr:`compute_dtype` (or identity if unset).

        Args:
            x: Array-like value to cast.

        Returns:
            The casted :class:`Array`, or ``x`` unchanged when
            :attr:`compute_dtype` is ``None``.
        """
        if self.compute_dtype is None:
            return jnp.asarray(x)
        return jnp.asarray(x, dtype=self.compute_dtype)

    def cast_output(self, x: ArrayLike) -> Array:
        """Cast ``x`` to :attr:`output_dtype` (or identity if unset).

        Args:
            x: Array-like value to cast.

        Returns:
            The casted :class:`Array`, or ``x`` unchanged when
            :attr:`output_dtype` is ``None``.
        """
        if self.output_dtype is None:
            return jnp.asarray(x)
        return jnp.asarray(x, dtype=self.output_dtype)

    def storage_dtype(self, fallback: DType | None) -> DType | None:
        """Return the dtype parameters should be stored as.

        If :attr:`param_dtype` is set it takes precedence; otherwise
        ``fallback`` is returned (typically the layer's declared dtype).

        Args:
            fallback: The dtype to fall back to when :attr:`param_dtype`
                is ``None``.

        Returns:
            The resolved storage dtype, or ``None`` when both
            :attr:`param_dtype` and ``fallback`` are ``None``.
        """
        return self.param_dtype if self.param_dtype is not None else fallback
