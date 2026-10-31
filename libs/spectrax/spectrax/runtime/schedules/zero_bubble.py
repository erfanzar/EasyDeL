# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Zero-bubble H1 schedule (Qi et al., ICLR 2024).

The H1 variant of zero-bubble pipeline parallelism splits each
backward into an input-gradient half (:attr:`Phase.BWD_I`) and a
weight-gradient half (:attr:`Phase.BWD_W`). The former stays on the
critical path; the latter is reorderable and slots into what would
otherwise be bubble time, driving the schedule's idle ratio toward
zero without raising peak activation memory.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Action, Phase, Schedule
from .one_f_one_b import Std1F1B


@dataclass
class ZeroBubbleH1(Schedule):
    """Zero-bubble H1 schedule (Qi et al., 2023, arxiv:2401.10241).

    The H1 variant splits each stage's backward into two:

    * :attr:`Phase.BWD_I` — **input gradient**, which is on the
      critical path (next-upstream stage waits for it).
    * :attr:`Phase.BWD_W` — **weight gradient**, which can run at any
      time after the BWD_I of the same stage x microbatch.

    The BWD_W work fills what would otherwise be bubble slots in
    :class:`Std1F1B`, driving the pipeline bubble ratio towards zero
    at the cost of slightly higher scheduling complexity.

    ``W-grads`` are not critical-path, so this schedule slots them
    into the warmup tail and cooldown head. Per-stage peak activation
    memory is the same as 1F1B.

    Reference: Penghui Qi, Xinyi Wan, Guangxing Huang, Min Lin.
    *Zero Bubble Pipeline Parallelism*. ICLR 2024.
    """

    def build(self, n_stages: int) -> list[list[Action | None]]:
        """Emit the H1 split-backward grid.

        Uses the "ZB-H1" variant: schedule the critical-path
        BWD_I actions as in 1F1B, then inject BWD_W actions into the
        warmup/cooldown bubbles. Sufficient BWD_W slots exist because
        the two halves of each backward are the same duration (the
        ZB paper's balance assumption).

        Implementation: start from the 1F1B skeleton and rewrite each
        BWD into a BWD_I plus a pending BWD_W for the same
        ``(stage, microbatch)``. A BWD_W for ``(s, mb)`` can be placed
        in any empty slot at stage ``s`` at time ``t >= bwd_i_time(s,
        mb)``; we scan stage-``s`` slots and fill the earliest
        eligible bubble, extending the grid with new rows if we run
        out of room.

        Args:
            n_stages: N stages value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        n, m = n_stages, self.microbatches
        n_stages_ref = n
        base = Std1F1B(m).build(n)

        bwd_i_time: dict[tuple[int, int], int] = {}
        grid: list[list[Action | None]] = []
        pending_w: dict[int, list[int]] = {s: [] for s in range(n)}

        for t, row in enumerate(base):
            new_row: list[Action | None] = [None] * n
            for s, action in enumerate(row):
                if action is None:
                    new_row[s] = None
                elif action.phase == Phase.FWD:
                    new_row[s] = action
                elif action.phase == Phase.BWD:
                    new_row[s] = Action(Phase.BWD_I, action.microbatch)
                    bwd_i_time[(s, action.microbatch)] = t
                    pending_w[s].append(action.microbatch)
            grid.append(new_row)

        for s in range(n):
            queue = list(pending_w[s])
            if not queue:
                continue
            t = 0
            while queue:
                while t >= len(grid):
                    grid.append([None] * n)
                if grid[t][s] is None:
                    placed = False
                    for i, mb in enumerate(queue):
                        if bwd_i_time[(s, mb)] < t:
                            grid[t][s] = Action(Phase.BWD_W, mb)
                            queue.pop(i)
                            placed = True
                            break
                    if not placed and t >= len(grid) - 1:
                        grid.append([None] * n)
                t += 1
                if t > 4 * (m + n_stages_ref) + 1:
                    raise RuntimeError(
                        f"ZeroBubbleH1: failed to place BWD_W for stage {s}, queue={queue}; please file a bug."
                    )

        while grid and all(c is None for c in grid[-1]):
            grid.pop()
        return grid

    def peak_activations(self, n_stages: int) -> int:
        """Same peak as :class:`Std1F1B` — splitting BWD does not change lifetime.

        BWD_W reads the same saved input as BWD_I, so injecting BWD_W
        into bubble slots does not extend any activation's lifetime.
        The per-stage peak is still bounded by ``n_stages``.

        Args:
            n_stages: Number of physical pipeline ranks.

        Returns:
            ``n_stages``.
        """
        return n_stages
