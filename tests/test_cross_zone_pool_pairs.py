# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Provider-neutral correlated label constraints."""

from dataclasses import replace

import pytest

from eyvan.domain import (
    CapacityCell,
    CapacityClass,
    CheckpointPolicy,
    PlacementMode,
    PriorityBand,
    ResourceGang,
    ResourceVector,
    Scheduler,
    WorkerResourceScope,
    WorkloadIdentity,
    WorkloadSpec,
)


def test_allowed_label_combinations_are_canonical_and_optional() -> None:
    base = WorkloadSpec(
        workload_id="workload",
        identity=WorkloadIdentity("org", "project", "user"),
        gangs=(ResourceGang("gang", 1, ResourceVector({"cpu": 1})),),
    )
    assert base.allowed_label_combinations == ()
    assert base == replace(base, allowed_label_combinations=())
    pairs = ((("template", "b"), ("pool", "a")), (("pool", "b"), ("template", "a")))
    constrained = replace(base, allowed_label_combinations=pairs)
    assert constrained == replace(base, allowed_label_combinations=tuple(reversed(pairs)))
    assert hash(constrained) == hash(replace(base, allowed_label_combinations=tuple(reversed(pairs))))
    with pytest.raises(ValueError, match="conflicting"):
        replace(base, allowed_label_combinations=((("pool", "a"), ("pool", "b")),))


@pytest.mark.parametrize("mode", ["atomic", "elastic", "group"])
def test_label_combinations_are_enforced_in_all_placement_modes(mode: str) -> None:
    workload = WorkloadSpec(
        workload_id="workload",
        identity=WorkloadIdentity("org", "project", "user"),
        gangs=(ResourceGang("gang", 1, ResourceVector(cpu=1)),),
        placement=PlacementMode.ELASTIC if mode == "elastic" else PlacementMode.ATOMIC,
        cell_group_count=2 if mode == "group" else 1,
        worker_resource_scope=WorkerResourceScope.PER_WORKER,
        allowed_label_combinations=((("pool", "a"), ("template", "a")), (("pool", "b"), ("template", "b"))),
    )
    cells = tuple(
        CapacityCell(
            f"cell-{index}",
            ResourceVector(cpu=2),
            CapacityClass.ON_DEMAND,
            labels=(("pool", "a"), ("template", "b")),
        )
        for index in range(2)
    )
    refused = Scheduler().schedule([workload], cells)
    assert not refused.placements
    assert not refused.preemptions
    paired = tuple(replace(cell, labels=(("pool", "a"), ("template", "a"))) for cell in cells)
    accepted = Scheduler().schedule([workload], paired)
    assert len(accepted.placements) == 1


def test_crossed_label_pairs_are_not_preemption_candidates() -> None:
    victim = WorkloadSpec(
        "victim",
        WorkloadIdentity("org", "project", "user"),
        (ResourceGang("gang", 1, ResourceVector(cpu=1)),),
        priority=PriorityBand.BATCH,
        checkpoint_policy=CheckpointPolicy(allow_forced_termination=True),
    )
    incoming = replace(
        victim,
        workload_id="incoming",
        priority=PriorityBand.INTERACTIVE,
        allowed_label_combinations=((("pool", "a"), ("template", "a")), (("pool", "b"), ("template", "b"))),
    )
    cell = CapacityCell("cell", ResourceVector(cpu=1), CapacityClass.SPOT, labels=(("pool", "a"), ("template", "b")))
    scheduler = Scheduler()
    lease = scheduler.schedule([victim], [cell]).placements[0].lease
    refused = scheduler.schedule([incoming], [cell], leases=[lease])
    assert not refused.placements
    assert not refused.preemptions
    paired = replace(cell, labels=(("pool", "a"), ("template", "a")))
    accepted = scheduler.schedule([incoming], [paired], leases=[lease])
    assert len(accepted.preemptions) == 1
