# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware JAX transforms.

Every callable exported from this package is a spectrax counterpart to
a JAX transform: :func:`eval_shape`, :func:`jit`, :func:`grad`,
:func:`value_and_grad`, :func:`jvp`, :func:`vjp`, :func:`vmap`,
:func:`scan`, :func:`associative_scan`, :func:`remat`, and the
control-flow primitives :func:`cond`, :func:`switch`, :func:`while_loop`,
:func:`fori_loop`, :func:`remat_scan`. The :class:`StateAxes` helper
plus :func:`split_rngs` / :func:`split_stream_keys` round out the
RNG-aware vmapping toolkit.

All transforms share the split/merge shim defined in
:mod:`spectrax.transforms.split_merge`. The shim:

1. Locates :class:`~spectrax.Module` arguments via
   :func:`~spectrax.transforms.split_merge.locate_and_strip`.
2. Exports each to a ``(GraphDef, State)`` pair.
3. Wraps the user function in a pure ``(states, args, kwargs) -> (out,
   new_states)`` callable consumable by the underlying JAX transform.
4. After the transform returns, copies declared-mutable leaves back to
   the live module via
   :func:`~spectrax.transforms.split_merge.apply_mutations`.

:func:`eval_shape` is the read-only exception: it snapshots module
inputs but throws away ``new_states`` so abstract updates never reach
the live modules.
"""

from .control_flow import cond, fori_loop, remat_scan, switch, while_loop
from .eval_shape import eval_shape
from .grad import grad, jvp, value_and_grad, vjp
from .jit import jit
from .remat import remat
from .rng_axes import StateAxes, split_rngs, split_stream_keys
from .scan import associative_scan, scan
from .vmap import vmap

__all__ = [
    "StateAxes",
    "associative_scan",
    "cond",
    "eval_shape",
    "fori_loop",
    "grad",
    "jit",
    "jvp",
    "remat",
    "remat_scan",
    "scan",
    "split_rngs",
    "split_stream_keys",
    "switch",
    "value_and_grad",
    "vjp",
    "vmap",
    "while_loop",
]
