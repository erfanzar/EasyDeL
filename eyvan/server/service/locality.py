# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Place work where its large inputs probably already are.

A training job that pulls a 400 GB dataset spends real minutes doing it, and
spends them again on every cell that has not seen the dataset before. The
placement that avoids the pull is worth preferring -- and Eyvan already has the
mechanism to prefer it: ``WorkloadSpec.preferred_cells`` is the fourth key of the
scheduler's lexicographic ``PlacementScore``. What was missing is the *index* --
which cells have already materialized which asset.

**Two signals, and the reported one wins.** A cell that ran a job declaring
asset X twenty minutes ago very probably still has X on local disk, and that
inference costs nothing -- the control plane already records which workload held
which cell and when. But it is a guess about a disk the control plane cannot
see, and an agent *can* see it. So an agent that reports its materialized assets
ranks ahead of every inference, and inference still covers every cell whose
agent reports nothing.

Deliberately additive rather than a replacement. An agent that never learns to
report leaves the fleet exactly where it was, which is what makes this safe to
deploy before every agent is upgraded.

**Being wrong is cheap, and that is what makes the inference acceptable.** The
output feeds a *preference*, never a requirement. A stale entry costs one cache
miss; it cannot make a workload unschedulable, cannot strand it in a zone, and
cannot survive the TTL. Any design where a wrong guess costs more than a pull
would need real reporting instead.
"""

from __future__ import annotations

import typing as tp
from collections import defaultdict
from datetime import datetime, timedelta

from eyvan.records import CellRecord, CellStageRecord, LeaseRecord, WorkloadRecord
from eyvan.server.service.stage import confirmed_stage_index

#: How long a cell is assumed to still hold an asset after a job that used it
#: last held the cell. Six hours is chosen against the failure it causes: too
#: long merely wastes one pull, while too short reverts to today's behaviour of
#: no locality at all. Nothing here is load-bearing enough to warrant a setting
#: until a fleet reports the guess is wrong.
DEFAULT_LOCALITY_TTL_S = 6 * 3_600

#: Most cells to suggest per workload. ``preferred_cells`` is a ranked list the
#: scheduler walks, and a long one buys nothing: past the first few, the cells
#: are old enough that the guess is weak anyway.
MAX_PREFERRED_CELLS = 8


def build_index(
    workloads: tp.Sequence[WorkloadRecord],
    leases: tp.Sequence[LeaseRecord],
    *,
    now: datetime,
    ttl_s: float = DEFAULT_LOCALITY_TTL_S,
    cells: tp.Sequence[CellRecord] = (),
    stages: tp.Sequence[CellStageRecord] = (),
) -> dict[str, tuple[str, ...]]:
    """Map each named asset to the cells that most recently worked with it.

    Cells are ordered most-recent-first, so the head of the list is the strongest
    guess. Ties break on cell id, so two leases stamped in the same instant do not
    make placement depend on dict ordering.

    Three signals, strongest first. A **confirmed managed stage** is the only one
    that is evidence: this control plane asked an executor for the bytes, was told
    they arrived, and recorded the generation it was told about. An **agent
    report** is a claim about a disk nobody here measured. **Lease history** is an
    inference from "a job naming this asset held this cell recently". Every stage
    that is not confirmed -- staging, unknown, evicting, blocked, or confirmed
    against a generation this cell no longer runs -- contributes nothing at all,
    which is the point: scoring an unproven stage as a hit would send a job to a
    cold cell on the strength of a row that never claimed the bytes were there.
    """

    cutoff = now - timedelta(seconds=ttl_s)
    assets_by_workload = {
        workload.id: workload.spec.named_assets for workload in workloads if workload.spec.named_assets
    }
    seen: dict[tuple[str, str], datetime] = {}
    for lease in leases:
        assets = assets_by_workload.get(lease.workload_id)
        if not assets:
            continue
        # The lease's most recent activity, not its creation: a long-running job
        # that started a week ago still had the asset on disk five minutes ago.
        stamp = lease.updated_at or lease.created_at
        if stamp < cutoff:
            continue
        for asset in assets:
            key = (asset, lease.cell_id)
            if key not in seen or seen[key] < stamp:
                seen[key] = stamp

    ranked: defaultdict[str, list[tuple[datetime, str]]] = defaultdict(list)
    for (asset, cell_id), stamp in seen.items():
        ranked[asset].append((stamp, cell_id))
    index = {
        asset: [cell_id for _stamp, cell_id in sorted(entries, key=lambda item: (item[0], item[1]), reverse=True)]
        for asset, entries in ranked.items()
    }

    # Reported assets go to the front, in cell-id order so placement never
    # depends on how the cells happened to be listed. An agent that can see the
    # disk beats an inference about it, and a cell that appears in both signals
    # must appear once -- a duplicate would count twice in the ranking and
    # silently outweigh a cell holding more of the workload's assets.
    reported: defaultdict[str, list[str]] = defaultdict(list)
    for cell in sorted(cells, key=lambda record: record.id):
        for asset in cell.materialized_assets:
            reported[asset].append(cell.id)
    for asset, cell_ids in reported.items():
        inferred = [cell_id for cell_id in index.get(asset, ()) if cell_id not in set(cell_ids)]
        index[asset] = [*cell_ids, *inferred]

    # Confirmed stages go ahead of both, and for the same de-duplication reason:
    # a cell that is proven to hold the asset must appear exactly once, or it
    # counts twice and outranks a cell holding more of the workload's assets.
    for asset, cell_ids in confirmed_stage_index(stages, cells).items():
        weaker = [cell_id for cell_id in index.get(asset, ()) if cell_id not in set(cell_ids)]
        index[asset] = [*cell_ids, *weaker]
    return {asset: tuple(entries) for asset, entries in index.items()}


def preferred_cells_for(
    assets: tp.Sequence[str],
    index: tp.Mapping[str, tp.Sequence[str]],
    *,
    limit: int = MAX_PREFERRED_CELLS,
) -> tuple[str, ...]:
    """Rank cells by how many of this workload's assets each already holds.

    A cell holding three of four assets beats one holding one, and among equals
    the one seen more recently wins -- so a job wanting a dataset *and* a model
    checkpoint is drawn toward the cell that has both rather than to whichever
    single asset happened to be indexed first.
    """

    if not assets:
        return ()
    hits: defaultdict[str, int] = defaultdict(int)
    best_rank: dict[str, int] = {}
    for asset in assets:
        for rank, cell_id in enumerate(index.get(asset, ())):
            hits[cell_id] += 1
            best_rank[cell_id] = min(best_rank.get(cell_id, rank), rank)
    ordered = sorted(hits, key=lambda cell_id: (-hits[cell_id], best_rank[cell_id], cell_id))
    return tuple(ordered[:limit])


__all__ = (
    "DEFAULT_LOCALITY_TTL_S",
    "MAX_PREFERRED_CELLS",
    "build_index",
    "preferred_cells_for",
)
