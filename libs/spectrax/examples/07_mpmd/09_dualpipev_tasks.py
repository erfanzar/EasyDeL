# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inspect the per-rank task list for DualPipe-V.

Builds the rank-centric task sequence for each pipeline rank and
prints task counts, fused-task counts, and the first few tasks per
rank. Demonstrates the :func:`dualpipev_tasks` utility.

Run::

    python -m examples.07_mpmd.09_dualpipev_tasks
"""

from __future__ import annotations

from spectrax.runtime.schedules import DualPipeV, FusedTask, dualpipev_tasks
from spectrax.runtime.schedules.base import Action


def main():
    """Build and inspect DualPipe-V per-rank task lists for 4 ranks, 8 microbatches."""
    n_pp = 4
    n_mubatches = 8

    print(f"DualPipeV task lists: {n_pp} ranks, {n_mubatches} microbatches")
    print(f"Schedule bubble ratio: {DualPipeV(microbatches=n_mubatches).bubble_ratio(n_pp):.2%}")
    print()

    for rank in range(n_pp):
        tasks = dualpipev_tasks(mpmd_dim=n_pp, mpmd_idx=rank, n_mubatches=n_mubatches)
        fused = sum(1 for t in tasks if isinstance(t, FusedTask))
        plain = sum(1 for t in tasks if isinstance(t, Action))
        print(f"rank {rank}: {len(tasks)} tasks ({fused} fused, {plain} plain)")
        for t in tasks[:4]:
            if isinstance(t, FusedTask):
                print(
                    f"    FusedTask(fwd={t.fwd.phase.value}/mb{t.fwd.microbatch}, "
                    f"bwd={t.bwd.phase.value}/mb{t.bwd.microbatch})"
                )
            else:
                print(f"    Action({t.phase.value}, mb={t.microbatch}, vs={t.virtual_stage})")
        if len(tasks) > 4:
            print(f"    ... ({len(tasks) - 4} more)")
        print()


if __name__ == "__main__":
    main()
