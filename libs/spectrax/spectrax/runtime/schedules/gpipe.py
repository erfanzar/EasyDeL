# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GPipe schedule (Huang et al., 2019): all forwards, then all backwards.

The simplest pipeline schedule. Every microbatch's forward runs through
every stage before any backward starts, so each stage holds ``M``
saved activations at its memory peak. The bubble shape is the
classic GPipe trapezoid: ``(n_stages - 1)`` empty slots at the start
of forward and at the end of backward.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Action, Phase, Schedule


@dataclass
class GPipe(Schedule):
    """GPipe schedule (Huang et al., 2019): all forwards then all backwards.

    Structure for ``n_stages`` stages and ``M`` microbatches::

        FWD:
          t=0: stage 0 on mb 0
          t=1: stage 0 on mb 1; stage 1 on mb 0
          …
          t=M+n-2: drain complete
        BWD:
          all microbatches' backwards, reverse of forward

    Pros:

    * Simplest schedule; easy to debug.
    * Natural activation-checkpointing pairing (each stage saves M
      activations, but the schedule is symmetric so `lax.remat` just
      works).

    Cons:

    * Peak activation memory scales as ``O(n_stages · M)`` — every
      stage holds every microbatch's activation until the backward
      pass starts.
    * ``2(n_stages - 1)`` idle slots at both ends of the schedule
      (the "bubble").
    """

    def build(self, n_stages: int) -> list[list[Action | None]]:
        """Emit the all-fwd / all-bwd grid.

        Forward phase: stage ``s`` runs microbatch ``t - s`` at time
        ``t`` while ``0 <= t - s < m``. Backward phase mirrors this
                with reversed stage order — stage ``s`` runs backward on
        microbatch ``t - (n - 1 - s)``.

        Args:
            n_stages: N stages value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        n, m = n_stages, self.microbatches
        fwd_steps = m + n - 1
        bwd_steps = m + n - 1
        grid: list[list[Action | None]] = []

        for t in range(fwd_steps):
            row: list[Action | None] = [None] * n
            for s in range(n):
                mb = t - s
                if 0 <= mb < m:
                    row[s] = Action(Phase.FWD, mb)
            grid.append(row)

        for t in range(bwd_steps):
            row = [None] * n
            for s in range(n):
                mb = t - (n - 1 - s)
                if 0 <= mb < m:
                    row[s] = Action(Phase.BWD, mb)
            grid.append(row)

        return grid

    def peak_activations(self, n_stages: int) -> int:
        """Peak live activations per stage equals :attr:`microbatches`.

        Every stage saves its activation for every microbatch's
        forward, then releases them as the matching backwards run.
        Memory is dominated by the all-fwd-first ordering.

        Args:
            n_stages: Number of physical pipeline ranks (unused for
                GPipe; the peak is per-stage and ``M``-bound).

        Returns:
            ``self.microbatches``.
        """
        return self.microbatches
