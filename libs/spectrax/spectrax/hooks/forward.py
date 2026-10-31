# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Functional wrappers for forward pre-hook and post-hook registration.

These are thin function wrappers around the
:meth:`~spectrax.Module.register_forward_pre_hook` and
:meth:`~spectrax.Module.register_forward_hook` methods for users who
prefer a functional call site::

    handle = register_forward_hook(module, my_hook)
    ...
    handle.remove()

Both hooks fire only in eager mode; under a spectrax transform they
are skipped with a single warning per module.
"""

from __future__ import annotations

from ..core._typing import ForwardHook, ForwardPreHook
from ..core.module import Module, _HookHandle

Handle = _HookHandle
"""Alias for the hook handle type returned by the register functions."""

__all__ = ["Handle", "register_forward_hook", "register_forward_pre_hook"]


def register_forward_pre_hook(module: Module, fn: ForwardPreHook) -> Handle:
    """Attach ``fn`` to ``module`` as a forward pre-hook.

    The hook is called as ``fn(module, args, kwargs)`` immediately
    before :meth:`Module.forward`. Return ``None`` to leave the call
    arguments alone, or return a ``(new_args, new_kwargs)`` tuple to
    rewrite them.

    Args:
        module: The target module.
        fn: A pre-hook callable matching
            :class:`spectrax.typing.ForwardPreHook`.

    Returns:
        A :class:`Handle` whose ``remove()`` method detaches the hook.
    """
    return module.register_forward_pre_hook(fn)


def register_forward_hook(module: Module, fn: ForwardHook) -> Handle:
    """Attach ``fn`` to ``module`` as a forward post-hook.

    The hook is called as ``fn(module, args, kwargs, out)`` after
    :meth:`Module.forward` returns. Return ``None`` to keep the
    original output, or return a value to replace it.

    Args:
        module: The target module.
        fn: A post-hook callable matching
            :class:`spectrax.typing.ForwardHook`.

    Returns:
        A :class:`Handle` whose ``remove()`` method detaches the hook.
    """
    return module.register_forward_hook(fn)
