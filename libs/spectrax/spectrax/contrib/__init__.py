# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opt-in helpers that depend on third-party libraries.

Anything in this subpackage lives behind an optional install (e.g.
``pip install spectrax-lib[contrib]``) and must import gracefully
when its dependency is missing — the typical pattern is to import the
optional dep at module level, capture any :class:`ImportError`, and
defer the failure to the first call into the helper that actually
needs it.

Currently exposes :class:`Optimizer` and :class:`MultiOptimizer`, both
of which wrap :mod:`optax` for spectrax's module/state split.
"""

from .optimizer import MultiOptimizer, Optimizer

__all__ = ["MultiOptimizer", "Optimizer"]
