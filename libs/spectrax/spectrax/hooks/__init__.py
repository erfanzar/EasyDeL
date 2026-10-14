# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Forward hooks and variable-write observers for :class:`~spectrax.Module`.

Two hook flavours are exposed:

* Forward pre- and post-hooks attached to a module via
  :func:`register_forward_pre_hook` and :func:`register_forward_hook`.
  They fire in eager mode around :meth:`Module.__call__` and are
  suppressed (with a single warning per module) under a spectrax
  transform — use ``self.sow("intermediates", ...)`` inside a
  transform when you need to capture intermediates safely.
* Variable write observers attached via :func:`register_variable_hook`.
  They fire on successful eager writes to a :class:`~spectrax.Variable`
  and likewise do not fire under transforms.

Each registration returns a handle whose ``remove()`` method detaches
the hook.
"""

from .forward import Handle, register_forward_hook, register_forward_pre_hook
from .variable import register_variable_hook

__all__ = [
    "Handle",
    "register_forward_hook",
    "register_forward_pre_hook",
    "register_variable_hook",
]
