# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Introspection helpers for spectrax modules.

This subpackage groups the read-only utilities used to inspect a live
:class:`~spectrax.Module`: PyTorch-style :func:`repr_module`, the
treescope-backed :func:`display`, the text :func:`summary` /
:func:`tabulate` reports, parameter and byte counters
(:func:`count_parameters`, :func:`count_bytes`,
:func:`format_parameters`), the XLA-cost-model wrapper
:func:`hlo_cost`, and the :class:`~spectrax.State` accessor
:func:`tree_state`.

All helpers are non-mutating; functions that need to compile under
:func:`jax.eval_shape` or :func:`jax.jit` (notably :func:`summary`,
:func:`tabulate`, and :func:`hlo_cost`) suppress forward / variable
hooks for the duration so introspection cannot trigger user-visible
side effects.
"""

from .counting import format_parameters
from .display import display
from .repr import repr_module
from .summary import summary
from .tabulate import count_bytes, count_parameters, hlo_cost, tabulate
from .tree import tree_state

__all__ = [
    "count_bytes",
    "count_parameters",
    "display",
    "format_parameters",
    "hlo_cost",
    "repr_module",
    "summary",
    "tabulate",
    "tree_state",
]
