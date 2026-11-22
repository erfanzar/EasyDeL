# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fusion helpers: fuse adjacent FWD/BWD or BWD_I/BWD_W pairs in schedule grids."""

from __future__ import annotations

from .base import Action, FusedTask, Phase


def fuse_1f1b_steady_state(
    grid: list[list[Action | FusedTask | None]],
) -> list[list[Action | FusedTask | None]]:
    """Collapse adjacent ``FWD(mb_A) -> BWD(mb_B)`` on the same rank into a :class:`FusedTask`.

    Scans the grid top-to-bottom, left-to-right, looking for a
    :class:`Action` with :attr:`Phase.FWD` at ``(t, rank, virt)``
    followed immediately (at ``(t+1, rank, virt)``) by a
    :class:`Action` with :attr:`Phase.BWD`. When both cells match and
    no other action on ``rank`` sits between them, the FWD cell is
    replaced with :class:`FusedTask(fwd, bwd)` and the BWD cell is
    cleared to ``None``. Grid size and action ordering are otherwise
    preserved — the runtime still iterates the same time steps, but
    fused slots dispatch once instead of twice.

    The transformation is safe for 1F1B-family schedules because:

    * The FWD produces ``y`` for downstream ranks; under async
      dispatch, the downstream consumer waits on the fused jit's
      output future either way.
    * The BWD reads ``saved_inputs[loc][mb_B]`` (from an earlier time
      step) and ``recv_cots[loc][mb_B]`` (cotangent that arrived from
      rank+1). Both are already available at ``t`` before we fire
      the fused jit.

    Leaves other cells untouched: non-adjacent FWD/BWD pairs, terminal
    rank's FWD (which needs the loss + ``d_loss/d_y`` between fwd and
    bwd), and any cell on a mismatched ``virtual_stage``.

    Args:
        grid: A schedule grid from :meth:`Schedule.build`. Cells are
            :class:`Action` / :class:`FusedTask` / ``None``.

    Returns:
        A new grid with fusable FWD-BWD pairs replaced by
        :class:`FusedTask`. Same shape as the input. The input is not
        mutated.
    """
    rows = [list(row) for row in grid]
    n_rows = len(rows)
    if n_rows < 2:
        return rows
    n_ranks = len(rows[0]) if rows else 0
    for t in range(n_rows - 1):
        for r in range(n_ranks):
            a = rows[t][r]
            b = rows[t + 1][r]
            if not isinstance(a, Action) or not isinstance(b, Action):
                continue
            if a.phase != Phase.FWD or b.phase != Phase.BWD:
                continue
            if a.virtual_stage != b.virtual_stage:
                continue
            rows[t][r] = FusedTask(fwd=a, bwd=b, virtual_stage=a.virtual_stage)
            rows[t + 1][r] = None
    return rows


def fuse_zerobubble_bwd_pair(
    grid: list[list[Action | FusedTask | None]],
) -> list[list[Action | FusedTask | None]]:
    """Collapse adjacent ``BWD_I(mb_A) -> BWD_W(mb_A)`` on the same rank into a :class:`FusedTask`.

    Zero-bubble H1 splits a backward into :attr:`Phase.BWD_I`
    (input gradient, critical path) and :attr:`Phase.BWD_W` (weight
    gradient, fills bubbles). When both halves of the *same*
    microbatch land on adjacent time steps on the same rank, dispatch
    them as one jit rather than two. The runtime still uses the
    split primitives (XLA DCEs the unused tangent output of each
    half-call) but pays only one trace/dispatch cost per pair.

    Args:
        grid: A schedule grid from :meth:`Schedule.build` (typically
            :meth:`ZeroBubbleH1.build`).

    Returns:
        A new grid with fusable ``(BWD_I, BWD_W)`` pairs replaced by
        :class:`FusedTask(fwd=bwd_i, bwd=bwd_w)` — we reuse
        :class:`FusedTask`'s slots: the ``fwd`` slot holds the
        :attr:`BWD_I` action and the ``bwd`` slot holds the
        :attr:`BWD_W` action. Runtime inspects phases to decide
        dispatch.
    """
    rows = [list(row) for row in grid]
    n_rows = len(rows)
    if n_rows < 2:
        return rows
    n_ranks = len(rows[0]) if rows else 0
    for t in range(n_rows - 1):
        for r in range(n_ranks):
            a = rows[t][r]
            b = rows[t + 1][r]
            if not isinstance(a, Action) or not isinstance(b, Action):
                continue
            if a.phase != Phase.BWD_I or b.phase != Phase.BWD_W:
                continue
            if a.microbatch != b.microbatch:
                continue
            if a.virtual_stage != b.virtual_stage:
                continue
            rows[t][r] = FusedTask(fwd=a, bwd=b, virtual_stage=a.virtual_stage)
            rows[t + 1][r] = None
    return rows
