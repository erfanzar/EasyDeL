# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Preempting a workload that holds leases on several cells.

Multi-cell gangs used to be excluded from preemption entirely -- ``revocable``
was gated on ``cell_group_count == 1`` -- because the protocol could not
checkpoint across cells and, more importantly, because victim selection is
per-cell: it draws candidates from ``leases_for_cell``, so a gang member on the
chosen cell could be stopped while its siblings on other cells kept running.

That leaves the survivors executing a job that can never complete, holding
accelerators nobody will reclaim: strictly worse than not preempting at all.

Now that ``NativeGangExecutor.checkpoint`` runs the hook on every cell under one
shared checkpoint ID, the remaining requirement is atomicity in the scheduler:
**every sibling lease is taken, or the candidate is rejected.** These tests pin
that, and the two properties that make it correct -- that capacity freed on
another cell does not count toward this placement, and that an exclusive cell is
only free once its own leases are all victims.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from test_scheduler import _cell, _identity, _workload

from eyvan.domain.models import (
    CapacityLease,
    CheckpointPolicy,
    PriorityBand,
    ResourceVector,
)
from eyvan.domain.scheduler import Scheduler

GANG = "gang-workload"


def _member(lease_id: str, cell_id: str, *, workload_id: str = GANG, revocable: bool = True) -> CapacityLease:
    """One lease of a multi-cell gang, checkpointable so it is eligible."""

    return CapacityLease(
        lease_id=lease_id,
        workload_id=workload_id,
        attempt=1,
        identity=_identity("tenant"),
        cell_id=cell_id,
        resources=ResourceVector(cpu=8),
        priority=PriorityBand.BATCH,
        hourly_cost=Decimal(0),
        reserved_cost=Decimal(0),
        resource_hours_reserved=ResourceVector(),
        checkpoint_policy=CheckpointPolicy(allow_forced_termination=True),
        revocable=revocable,
    )


def _plan(leases: tuple[CapacityLease, ...], *, cells=None):
    """Schedule one production-band job needing a full cell."""

    result = Scheduler().schedule(
        (_workload("urgent", resources=ResourceVector(cpu=8), priority=PriorityBand.PRODUCTION),),
        cells if cells is not None else (_cell("cell-a"), _cell("cell-b")),
        leases=leases,
    )
    return result


def test_preempting_one_gang_member_takes_every_sibling() -> None:
    """The property the old gate existed to protect, now enforced directly."""

    leases = (_member("lease-a", "cell-a"), _member("lease-b", "cell-b"))

    result = _plan(leases)

    assert result.preemptions, "a checkpointable gang on spare capacity should be preemptible"
    victims = {victim.lease_id for plan in result.preemptions for victim in plan.victims}
    assert victims == {"lease-a", "lease-b"}, "a strict subset of a gang must never be selected"


def test_an_unrevocable_sibling_protects_the_whole_gang() -> None:
    """All-or-nothing cuts both ways: if one member may not be stopped, none is.

    Otherwise the gang would be dismembered by the very rule meant to keep it
    whole.
    """

    leases = (_member("lease-a", "cell-a"), _member("lease-b", "cell-b", revocable=False))

    result = _plan(leases)

    victims = {victim.lease_id for plan in result.preemptions for victim in plan.victims}
    assert "lease-a" not in victims


def test_a_single_cell_workload_is_unaffected() -> None:
    """The ordinary case must keep behaving exactly as before."""

    solo = _member("lease-solo", "cell-a", workload_id="solo-workload")

    result = _plan((solo,), cells=(_cell("cell-a"),))

    victims = {victim.lease_id for plan in result.preemptions for victim in plan.victims}
    assert victims == {"lease-solo"}


def test_freeing_a_sibling_elsewhere_does_not_make_this_cell_fit() -> None:
    """Only capacity released on the placement cell counts toward fitting.

    Sized so the distinction decides the outcome: cell-a can offer 2 free plus
    2 reclaimable from its gang member, and the request needs 9. Counting the
    sibling's 8 on cell-b would make the plan look feasible and admit a workload
    onto a cell with no room for it.
    """

    pinned = CapacityLease(
        lease_id="pinned",
        workload_id="other",
        attempt=1,
        identity=_identity("tenant"),
        cell_id="cell-a",
        resources=ResourceVector(cpu=6),
        priority=PriorityBand.PRODUCTION,
        hourly_cost=Decimal(0),
        reserved_cost=Decimal(0),
        resource_hours_reserved=ResourceVector(),
        checkpoint_policy=CheckpointPolicy(),
        revocable=False,
    )
    big_sibling = CapacityLease(
        lease_id="lease-b",
        workload_id=GANG,
        attempt=1,
        identity=_identity("tenant"),
        cell_id="cell-b",
        resources=ResourceVector(cpu=8),
        priority=PriorityBand.BATCH,
        hourly_cost=Decimal(0),
        reserved_cost=Decimal(0),
        resource_hours_reserved=ResourceVector(),
        checkpoint_policy=CheckpointPolicy(allow_forced_termination=True),
        revocable=True,
    )
    small = replace(_member("lease-a", "cell-a"), resources=ResourceVector(cpu=2))

    result = Scheduler().schedule(
        (_workload("urgent", resources=ResourceVector(cpu=9), priority=PriorityBand.PRODUCTION),),
        (_cell("cell-a", capacity=ResourceVector(cpu=10)), _cell("cell-b", capacity=ResourceVector(cpu=8))),
        leases=(pinned, small, big_sibling),
    )

    for plan in result.preemptions:
        assert plan.cell_id != "cell-a", (
            "cell-a can free at most 4 cpu locally for a 9 cpu request; "
            "counting the sibling's capacity on cell-b made it look feasible"
        )


def test_a_gang_is_not_selected_twice_when_both_members_are_eligible() -> None:
    """Both siblings sit in the candidate pool; the plan must list each once."""

    leases = (_member("lease-a", "cell-a"), _member("lease-b", "cell-a"))

    result = _plan(leases, cells=(_cell("cell-a", capacity=ResourceVector(cpu=16)),))

    for plan in result.preemptions:
        ids = [victim.lease_id for victim in plan.victims]
        assert len(ids) == len(set(ids)), f"duplicate victims in plan: {ids}"


def test_a_gang_is_preemptible_on_exclusive_cells() -> None:
    """The case that matters on real hardware: every native cell is exclusive.

    ``to_core_cell`` marks native-executor cells ``exclusive_workloads``, so a
    TPU cell always is. The exclusive check compared ``len(selected)`` against
    the cell's own lease count -- and ``selected`` now also carries gang siblings
    from other cells, so a two-cell gang made the count 2 against a cell holding
    1 and the plan was rejected. Gangs preempted fine on shared cells and never
    on the only kind Eyvan actually provisions.
    """

    member = _member("g-a", "cell-a")
    sibling = _member("g-b", "cell-b")

    result = Scheduler().schedule(
        (_workload("urgent", resources=ResourceVector(cpu=8), priority=PriorityBand.PRODUCTION),),
        (
            _cell("cell-a", capacity=ResourceVector(cpu=8), exclusive_workloads=True),
            _cell("cell-b", capacity=ResourceVector(cpu=8), exclusive_workloads=True),
        ),
        leases=(member, sibling),
    )

    assert result.preemptions, "a gang on exclusive cells must still be preemptible"
    victims = {victim.lease_id for plan in result.preemptions for victim in plan.victims}
    assert victims == {"g-a", "g-b"}


def test_an_exclusive_cell_keeps_an_unrelated_tenant_until_it_is_a_victim() -> None:
    """Emptying an exclusive cell means every lease *on it*, not a sibling count."""

    member = replace(_member("g-a", "cell-a"), resources=ResourceVector(cpu=4))
    sibling = replace(_member("g-b", "cell-b"), resources=ResourceVector(cpu=4))
    neighbour = CapacityLease(
        lease_id="neighbour",
        workload_id="unrelated",
        attempt=1,
        identity=_identity("tenant"),
        cell_id="cell-a",
        resources=ResourceVector(cpu=4),
        priority=PriorityBand.BATCH,
        hourly_cost=Decimal(0),
        reserved_cost=Decimal(0),
        resource_hours_reserved=ResourceVector(),
        checkpoint_policy=CheckpointPolicy(allow_forced_termination=True),
        revocable=True,
    )

    result = Scheduler().schedule(
        (_workload("urgent", resources=ResourceVector(cpu=8), priority=PriorityBand.PRODUCTION),),
        (
            _cell("cell-a", capacity=ResourceVector(cpu=8), exclusive_workloads=True),
            _cell("cell-b", capacity=ResourceVector(cpu=4), exclusive_workloads=True),
        ),
        leases=(member, sibling, neighbour),
    )

    for plan in result.preemptions:
        if plan.cell_id != "cell-a":
            continue
        assert "neighbour" in {victim.lease_id for victim in plan.victims}, (
            "cell-a was claimed while an unrelated lease still held it"
        )
