# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":class:`PipelineStage` dataclass and helpers.

A :class:`PipelineStage` is the unit the pipeline runtimes consume:
each stage is a Python callable plus the pytree of parameters it
operates on, optionally carrying a per-call state (KV cache, RNN
hidden, etc.). One :class:`PipelineStage` corresponds to one
*logical* stage of the pipeline; how those logical stages map onto
physical mesh ranks is the runtime's job, not the dataclass'.

The MPMD runtime (:func:`spectrax.runtime.mpmd.sxcall`) accepts lists
of these directly.  SPMD ``PipelineSequential`` execution uses the same
logical stage concept after extracting stages from the module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["PipelineStage", "_is_empty_state"]


def _is_empty_state(s: object) -> bool:
    """Detect the "no state" sentinel used by stateless stages.

    The pipeline runtimes accept either ``()`` or ``None`` to mean
    "this stage carries no state across calls" so users don't have to
    pick a single canonical spelling. This helper normalises both
    forms into a single boolean check used wherever the runtime needs
    to decide whether to thread per-stage state through ``shard_map``
    in/out specs, ``device_put`` calls, and the per-microbatch loop.

    Args:
        s: A pytree-like value to test, typically a stage's
            ``init_state`` or the state output of a previous call.

    Returns:
        ``True`` if ``s`` is the empty-state sentinel (``None`` or
        ``()``); ``False`` otherwise. Note that this only checks the
        two known sentinels — non-empty pytrees that happen to flatten
        to zero leaves are still treated as a real state.
    """
    return s is None or s == ()


@dataclass
class PipelineStage:
    """A single logical stage of a pipeline.

    Bundles the three pieces of data the pipeline runtimes need to
    invoke a stage: its forward callable, the parameter pytree it
    closes over, and the initial state pytree (if any). The runtime
    is responsible for placing :attr:`parameters` and :attr:`init_state`
    on the right devices and for threading the carry between stages;
    this dataclass is purely descriptive.

    The expected callable signature is
    ``fn(parameters, state, x) -> (y, new_state)`` where:

    * ``parameters`` is the same pytree as :attr:`parameters` (the
      runtime hands it back unchanged each call so JAX can shard it).
    * ``state`` is whatever value was last returned as ``new_state``
      (or :attr:`init_state` on the first call). Stateless stages set
      :attr:`init_state` to ``()`` or ``None`` and ignore the argument.
    * ``x`` is the activation flowing in from the upstream stage (or
      the model input on stage 0).

    Attributes:
        fn: The stage's forward callable. Must be a pure function of
            its three arguments.
        parameters: The parameter pytree consumed by ``fn``. Shape
            and dtype determine whether a runtime can stack stages on
            a shared SPMD axis.
        init_state: Initial state pytree. Defaults to ``()`` for
            stateless stages; see :func:`_is_empty_state` for the
            sentinel handling.
    """

    fn: Callable[[object, object, object], tuple[object, object]]
    parameters: object
    init_state: object = ()
