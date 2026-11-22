# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Variable write observers (eager only).

Use :func:`register_variable_hook` to attach an observer to an
individual :class:`~spectrax.Variable`. The returned
:class:`_VarHookHandle` carries a :meth:`~_VarHookHandle.remove`
method so hooks can be cleanly detached when no longer needed.

Observers do not fire under spectrax transforms — writes there are
intercepted by the transform machinery before observers would
otherwise run.
"""

from __future__ import annotations

from ..core._typing import VariableObserver
from ..core.variable import Variable

__all__ = ["register_variable_hook"]


class _VarHookHandle:
    """Handle returned by :func:`register_variable_hook`.

    Holds a reference to the target :class:`~spectrax.Variable` and
    the registered observer so that :meth:`remove` can detach the
    observer at any later point. Uses ``__slots__`` to avoid a
    per-instance ``__dict__`` (these handles are cheap and may be
    created in tight loops).

    Attributes:
        _var: The variable the observer is attached to.
        _fn: The observer callable.
    """

    __slots__ = ("_fn", "_var")

    _var: Variable
    _fn: VariableObserver

    def __init__(self, var: Variable, fn: VariableObserver) -> None:
        """Record the target variable and the observer.

        Args:
            var: The variable the observer is attached to.
            fn: The observer callable.
        """
        self._var = var
        self._fn = fn

    def remove(self) -> None:
        """Detach the observer from the variable.

        Idempotent in practice: if the observer has already been
        removed, the underlying ``remove_observer`` call is a no-op
        (or may raise depending on the variable implementation).
        """
        self._var.remove_observer(self._fn)


def register_variable_hook(var: Variable, fn: VariableObserver) -> _VarHookHandle:
    """Register ``fn`` to be called on every successful eager write to ``var``.

    The observer is invoked as ``fn(var, old, new)`` after the write
    has been applied; exceptions raised inside the observer are
    swallowed by :class:`~spectrax.Variable`'s write path so a buggy
    observer cannot derail user code.

    Args:
        var: The variable to observe.
        fn: The observer callable matching
            :class:`spectrax.typing.VariableObserver`.

    Returns:
        A :class:`_VarHookHandle` whose :meth:`~_VarHookHandle.remove`
        method detaches the observer.
    """
    var.add_observer(fn)
    return _VarHookHandle(var, fn)
