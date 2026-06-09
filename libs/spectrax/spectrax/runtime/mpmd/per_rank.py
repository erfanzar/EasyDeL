# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-rank compiled pipeline programs — fewer jits per training step.

Fold Python-orchestrated pipeline dispatch into a small number of
compiled programs per rank, shrinking the per-step jit dispatch count
from ``O(T x n)`` (one per :class:`Action`) to ``O(n)``. Each rank
gets a **forward-sweep** jit that runs every ``FWD`` assigned to that
rank back-to-back, and a **backward-sweep** jit that does the same
for the rank's ``BWD`` actions plus the terminal-rank loss/g_y seed.

**Scope (honest)**:

* Supports **GPipe** only. GPipe's clean "all forwards, then all
  backwards" structure lets a rank execute its entire forward sweep
  with no cross-rank cotangent dependency, and its entire backward
  sweep with no cross-rank activation dependency. Non-GPipe schedules
  (Std1F1B, 1F1B variants, ZeroBubble, Interleaved, DualPipeV)
  **interleave forwards and backwards** across ranks per time step,
  so a single-rank one-shot program would deadlock on cotangents
  from ranks that haven't run yet. That case fundamentally requires
  explicit send/recv (e.g. NCCL P2P) inside the compiled program —
  which requires NCCL P2P or shard_map + ppermute — and is not
available here.

* Supports **flat** schedules only (``virtual_stages_per_rank == 1``).

* **No cross-rank compute overlap yet.** The Python driver still
  waits for each rank's program to finish before transporting
  activations to the next rank. True overlap requires send/recv
  primitives (NCCL on GPU) or ``shard_map`` + ``lax.ppermute``
  under one fused HLO. The win this module delivers is purely
  dispatch-count reduction (``2·n`` jits per step vs ``2·n·M`` in
  :func:`sxcall`'s scheduled path).

**Entry points**:

* :func:`compile_per_rank_fwd` — per-rank forward-sweep jit.
* :func:`compile_per_rank_bwd` — per-rank backward-sweep jit
  (terminal rank also computes loss + g_y seed).
* :func:`run_gpipe_per_rank` — driver that stitches the two sweeps
  together with :func:`jax.device_put` transports between ranks.
* :func:`extract_rank_actions` — introspection helper.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from ..schedules import Action, FusedTask, GPipe, Phase, Schedule

__all__ = [
    "compile_per_rank_bwd",
    "compile_per_rank_fwd",
    "extract_rank_actions",
    "run_gpipe_per_rank",
]


def extract_rank_actions(schedule: Schedule, rank: int, n_stages: int) -> list[Action]:
    """Return the ordered :class:`Action` list that ``rank`` performs.

    Walks the schedule grid row-by-row and collects every action at
    column ``rank``. :class:`FusedTask` cells are expanded into their
    component actions — per-rank compilation is orthogonal to
    :class:`FusedTask` fusion (fusion happens inside
    :func:`~spectrax.runtime.mpmd.sxcall` instead).

    Args:
        schedule: The :class:`Schedule` whose grid to walk.
        rank: Physical pipeline rank to extract actions for
            (``0 <= rank < n_stages``).
        n_stages: Number of physical pipeline stages (passed to
            :meth:`Schedule.build`).

    Returns:
        :class:`Action` s for ``rank``, in time-step order. Empty if
        ``rank`` has no assigned actions.
    """
    grid = schedule.build(n_stages)
    actions: list[Action] = []
    for row in grid:
        cell = row[rank]
        if cell is None:
            continue
        if isinstance(cell, FusedTask):
            actions.append(cell.fwd)
            actions.append(cell.bwd)
        else:
            actions.append(cell)
    return actions


def compile_per_rank_fwd(
    rank: int,
    schedule: Schedule,
    n_stages: int,
    microbatches: int,
    fwd_fn: Callable[..., object],
) -> Callable[..., object]:
    """Compile this rank's entire forward sweep into one :func:`jax.jit`.

    Returns a jitted callable that, in a single compiled HLO, runs
    every :class:`Phase.FWD` action assigned to ``rank`` in schedule
    order. For :class:`GPipe` this is exactly ``M`` forwards, each
    consuming the corresponding incoming microbatch and producing a
    single outgoing activation — stacked into a ``(M, ...)`` output.

    Args:
        rank: Physical pipeline rank.
        schedule: Must be a :class:`GPipe` schedule; a
            :class:`NotImplementedError` is raised for other types.
        n_stages: Number of physical pipeline stages.
        microbatches: ``M`` — microbatches per step.
        fwd_fn: ``(params, rest, x) -> y`` — one microbatch forward on
            this rank's stage.

    Returns:
        Jitted callable with signature::

            fwd_step(params, rest, mb_inputs)
                -> (mb_outputs, mb_saved_inputs)

        where ``mb_inputs`` and ``mb_outputs`` are ``(M, *mb_shape)``
        stacks and ``mb_saved_inputs`` is the same stack returned so
        the matching backward sweep can consume the exact inputs
        observed here (avoids recomputation).
    """
    if not isinstance(schedule, GPipe):
        raise NotImplementedError(
            f"compile_per_rank_fwd currently supports GPipe only; "
            f"got {type(schedule).__name__}. Non-GPipe schedules need "
            f"send/recv inside the compiled program (NCCL), which is "
            f"not available here."
        )
    actions = [a for a in extract_rank_actions(schedule, rank, n_stages) if a.phase == Phase.FWD]
    if len(actions) != microbatches:
        raise ValueError(f"rank {rank} has {len(actions)} FWD actions; expected {microbatches} for GPipe.")
    fwd_order = [a.microbatch for a in actions]

    @jax.jit
    def fwd_step(params, rest, mb_inputs):
        """Run every FWD action for this rank inside a single jitted HLO.

        Iterates through ``fwd_order`` (the rank's FWD actions in
        schedule order) and indexes ``mb_inputs`` per microbatch. Each
        call's output is placed back into the matching microbatch slot
        and the resulting per-microbatch outputs are stacked into a
        single ``(M, ...)`` array so the next rank can transport one
        contiguous activation buffer instead of M individual arrays.

        Args:
            params: This rank's stage parameters (placed on the
                rank-local sub-mesh).
            rest: This rank's non-parameter state.
            mb_inputs: Microbatched inputs of shape ``(M, *mb_shape)``.

        Returns:
            ``(mb_outputs, mb_inputs)`` where ``mb_outputs`` is a
            ``(M, *out_shape)`` stack and ``mb_inputs`` is returned
            unchanged so the matching backward sweep can consume the
            exact inputs observed here without recomputation.
        """
        with jax.named_scope(f"spectrax/mpmd/per_rank/forward_rank_{rank}"):
            mb_outputs = [None] * microbatches
            for mb in fwd_order:
                with jax.named_scope(f"spectrax/mpmd/per_rank/forward_rank_{rank}/microbatch_{mb}"):
                    y = fwd_fn(params, rest, mb_inputs[mb])
                mb_outputs[mb] = y
            return jnp.stack(tuple(mb_outputs), axis=0), mb_inputs

    return fwd_step


def compile_per_rank_bwd(
    rank: int,
    schedule: Schedule,
    n_stages: int,
    microbatches: int,
    bwd_fn: Callable[..., object],
    loss_and_g_y: Callable[..., object] | None = None,
    *,
    is_terminal: bool = False,
) -> Callable[..., object]:
    """Compile this rank's entire backward sweep into one :func:`jax.jit`.

    Returns a jitted callable that, in a single compiled HLO, runs
    every :class:`Phase.BWD` action assigned to ``rank``. The terminal
    rank additionally computes the scalar loss and the initial
    cotangent via ``loss_and_g_y`` before the first backward.

    Args:
        rank: Physical pipeline rank.
        schedule: Must be a :class:`GPipe` schedule; raises
            :class:`NotImplementedError` for other types.
        n_stages: Number of physical pipeline stages.
        microbatches: ``M`` — microbatches per step.
        bwd_fn: ``(params, rest, x, g_y) -> (g_params, g_x)`` — one
            microbatch's VJP on this rank.
        loss_and_g_y: ``(y, *targets) -> (loss_scalar, g_y)`` — only
            consulted on the terminal rank.
        is_terminal: ``True`` iff ``rank`` is the terminal stage and
            should compute the loss / g_y seed.

    Returns:
        Jitted callable with signature::

            bwd_step(params, rest, mb_saved_inputs, mb_saved_outputs,
                     mb_incoming_cots, *mb_targets)
                -> (g_params, mb_outgoing_cots, loss_sum)

        Non-terminal ranks ignore ``mb_saved_outputs`` and
        ``mb_targets`` and return ``loss_sum == 0``.
    """
    if not isinstance(schedule, GPipe):
        raise NotImplementedError(f"compile_per_rank_bwd currently supports GPipe only; got {type(schedule).__name__}.")
    if is_terminal and loss_and_g_y is None:
        raise ValueError(f"compile_per_rank_bwd(rank={rank}, is_terminal=True) requires loss_and_g_y to be provided.")
    actions = [a for a in extract_rank_actions(schedule, rank, n_stages) if a.phase == Phase.BWD]
    if len(actions) != microbatches:
        raise ValueError(f"rank {rank} has {len(actions)} BWD actions; expected {microbatches} for GPipe.")
    bwd_order = [a.microbatch for a in actions]

    @jax.jit
    def bwd_step(params, rest, mb_saved_inputs, mb_saved_outputs, mb_incoming_cots, *mb_targets):
        """Run every BWD for this rank back-to-back (terminal: loss + g_y).

        Args:
            params: Parameter mapping or primitive parameter dictionary.
            rest: Rest value consumed by this operation.
            mb_saved_inputs: Mb saved inputs value consumed by this operation.
            mb_saved_outputs: Mb saved outputs value consumed by this operation.
            mb_incoming_cots: Mb incoming cots value consumed by this operation.
            *mb_targets: Additional positional arguments forwarded to the wrapped callable or backend.
        """
        with jax.named_scope(f"spectrax/mpmd/per_rank/backward_rank_{rank}"):
            mb_outgoing_cots = jnp.zeros_like(mb_saved_inputs)
            g_params = jax.tree.map(jnp.zeros_like, params)
            loss_sum = None

            for mb in bwd_order:
                with jax.named_scope(f"spectrax/mpmd/per_rank/backward_rank_{rank}/microbatch_{mb}"):
                    if is_terminal:
                        y_out = mb_saved_outputs[mb]
                        loss_mb, g_y = loss_and_g_y(y_out, *(t[mb] for t in mb_targets))
                        loss_sum = loss_mb if loss_sum is None else loss_sum + loss_mb
                    else:
                        g_y = mb_incoming_cots[mb]
                    x_in = mb_saved_inputs[mb]
                    g_p, g_x = bwd_fn(params, rest, x_in, g_y)
                    g_params = jax.tree.map(lambda a, b: a + b, g_params, g_p)
                    mb_outgoing_cots = mb_outgoing_cots.at[mb].set(g_x)

            if loss_sum is None:
                loss_sum = jnp.zeros((), dtype=jnp.float32)
            return g_params, mb_outgoing_cots, loss_sum

    return bwd_step


def run_gpipe_per_rank(
    *,
    n_stages: int,
    microbatches: int,
    schedule: GPipe,
    fwd_fns: list[Callable[..., object]],
    bwd_fns: list[Callable[..., object]],
    loss_and_g_y: Callable[..., object],
    stage_params: list[object],
    stage_rest: list[object],
    stage_shardings: list[object],
    xs: jax.Array,
    target_args: tuple[jax.Array, ...],
) -> tuple[jax.Array, tuple[object, ...]]:
    """Execute a GPipe training step with 2n per-rank compiled programs.

    Drives one forward-sweep jit and one backward-sweep jit per rank.
    Cross-rank transports happen via :func:`jax.device_put` between
    programs. Correctness matches
    :func:`~spectrax.runtime.mpmd.sxcall`; performance win is bounded
    by the fraction of step time spent in Python dispatch — small at
    realistic training scales.

    Args:
        n_stages: Number of physical pipeline ranks.
        microbatches: ``M``.
        schedule: Must be :class:`GPipe`.
        fwd_fns, bwd_fns: Per-rank forward / backward jits used inside
            each rank's compiled sweep.
        loss_and_g_y: ``(y, *targets) -> (loss_scalar, g_y)``; consumed
            only inside the terminal rank's backward sweep.
        stage_params, stage_rest: Per-rank parameter / non-param state,
            pre-placed on the rank's sub-mesh.
        stage_shardings: Per-rank :class:`NamedSharding` for placing
            transports between ranks.
        xs: Microbatched inputs, shape ``(M, per_mb_batch, ...)``.
        target_args: Microbatched targets.

    Returns:
        ``(mean_loss, per_rank_grads)`` — same return shape as
        :func:`~spectrax.runtime.mpmd.sxcall`.
    """
    fwd_programs = [
        compile_per_rank_fwd(
            rank=r,
            schedule=schedule,
            n_stages=n_stages,
            microbatches=microbatches,
            fwd_fn=fwd_fns[r],
        )
        for r in range(n_stages)
    ]
    bwd_programs = [
        compile_per_rank_bwd(
            rank=r,
            schedule=schedule,
            n_stages=n_stages,
            microbatches=microbatches,
            bwd_fn=bwd_fns[r],
            loss_and_g_y=loss_and_g_y if r == n_stages - 1 else None,
            is_terminal=(r == n_stages - 1),
        )
        for r in range(n_stages)
    ]

    saved_inputs_per_rank: list[jax.Array] = []
    saved_outputs_per_rank: list[jax.Array] = []
    mb_curr = xs

    for r in range(n_stages):
        mb_curr_r = jax.device_put(mb_curr, stage_shardings[r])
        mb_out, mb_in = fwd_programs[r](stage_params[r], stage_rest[r], mb_curr_r)
        saved_inputs_per_rank.append(mb_in)
        saved_outputs_per_rank.append(mb_out)
        mb_curr = mb_out

    zero_cots = jnp.zeros_like(xs)
    grads: list[object] = [None] * n_stages
    loss_sum = jnp.zeros((), dtype=jnp.float32)
    mb_cots = zero_cots

    for r in range(n_stages - 1, -1, -1):
        if r == n_stages - 1:
            targets_r = tuple(jax.device_put(t, stage_shardings[r]) for t in target_args)
            mb_cots_r = jnp.zeros_like(saved_outputs_per_rank[r])
        else:
            targets_r = ()
            mb_cots_r = jax.device_put(mb_cots, stage_shardings[r])
        g_p, mb_out_cots, loss_r = bwd_programs[r](
            stage_params[r],
            stage_rest[r],
            saved_inputs_per_rank[r],
            saved_outputs_per_rank[r],
            mb_cots_r,
            *targets_r,
        )
        grads[r] = g_p
        loss_sum = loss_sum + loss_r
        mb_cots = mb_out_cots

    inv_m = 1.0 / jnp.asarray(microbatches, dtype=loss_sum.dtype)
    mean_loss = loss_sum * inv_m
    mean_grads = jax.tree.map(lambda g: g * inv_m, tuple(grads))
    return mean_loss, mean_grads
