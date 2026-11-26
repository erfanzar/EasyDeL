# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Preemption eligibility for non-checkpointable spot capacity.

The scheduler requires a victim to be ``checkpointable or
allow_forced_termination``. The service adapter never set the second flag, so a
non-checkpointable spot lease was revocable but never *eligible*: a
higher-priority workload got PREEMPTION_UNSAFE and that capacity was never
reclaimed. These tests pin the policy that fixes it, and pin the boundary that
on-demand capacity is still protected.

The same flag must key off the class the lease actually *landed* on -- the class
``revocable`` already used. A portfolio workload that requested on-demand and
landed on a spot cell stayed revocable-but-ineligible under the requested-class
policy, so its spot capacity was never reclaimable either; the tests at the end
pin that direction too.
"""

from __future__ import annotations

from datetime import timedelta

from eyvan.domain.models import InfeasibilityCode
from eyvan.records import (
    CapacityClass,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    PriorityBand,
    ResourceGang,
    ResourceHours,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.service.scheduling import CoreSchedulingService, _checkpoint, to_core_lease


def _spec(
    *,
    capacity_class: CapacityClass,
    checkpointable: bool = False,
    band: PriorityBand = PriorityBand.BATCH,
    additional_capacity_classes: tuple[CapacityClass, ...] = (),
) -> WorkloadDefinition:
    return WorkloadDefinition(
        name="job",
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=8, memory_gib=8)),
        capacity_class=capacity_class,
        additional_capacity_classes=additional_capacity_classes,
        priority_band=band,
        checkpointable=checkpointable,
        checkpoint_hook="hook" if checkpointable else None,
        checkpoint_output_uri="gs://bucket/ckpt" if checkpointable else None,
    )


def _record(workload_id: str, spec: WorkloadDefinition, state) -> WorkloadRecord:
    return WorkloadRecord(
        id=workload_id,
        org_id="acme",
        project_id="research",
        spec=spec,
        state=state,
        created_by="alice",
    )


def test_spot_capacity_is_force_terminable() -> None:
    policy = _checkpoint(_record("wl", _spec(capacity_class=CapacityClass.SPOT), None or "pending"))

    assert policy.allow_forced_termination is True


def test_on_demand_capacity_is_not_force_terminable() -> None:
    """On-demand is only preemptible when the workload can checkpoint."""

    policy = _checkpoint(_record("wl", _spec(capacity_class=CapacityClass.ON_DEMAND), "pending"))

    assert policy.allow_forced_termination is False


def test_checkpointable_spot_keeps_its_checkpoint_mode() -> None:
    """Enabling forced termination must not disable checkpoint-on-preemption."""

    policy = _checkpoint(_record("wl", _spec(capacity_class=CapacityClass.SPOT, checkpointable=True), "pending"))

    assert policy.checkpointable is True
    assert policy.allow_forced_termination is True


def test_a_higher_priority_workload_reclaims_non_checkpointable_spot_capacity() -> None:
    """The behaviour that was broken: this used to report PREEMPTION_UNSAFE."""

    victim_spec = _spec(capacity_class=CapacityClass.SPOT, band=PriorityBand.BATCH)
    victim = _record("wl-victim", victim_spec, WorkloadState.RUNNING)
    claimant = _record(
        "wl-claimant",
        _spec(capacity_class=CapacityClass.SPOT, band=PriorityBand.PRODUCTION),
        WorkloadState.PENDING,
    )
    cell = CellRecord(
        id="cell-a",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=8, memory_gib=8),
        allocatable=ResourceVector(cpu=8, memory_gib=8),
        capacity_class=CapacityClass.SPOT,
    )
    lease = LeaseRecord(
        id="lease-victim",
        workload_id=victim.id,
        cell_id=cell.id,
        cell_generation=1,
        org_id=victim.org_id,
        project_id=victim.project_id,
        resources=victim_spec.gang,
        hourly_cost=1.0,
        reserved_cost=1.0,
        resource_hours_reserved=ResourceHours(cpu=8.0, memory_gib=8.0, gpu=0.0, tpu=0.0),
        generation=1,
        fencing_token="fence",
        expires_at=utcnow() + timedelta(minutes=5),
    )

    result = CoreSchedulingService().schedule([victim, claimant], [cell], leases=[lease])

    preempted = [plan for plan in result.preemptions if plan.workload_id == claimant.id]
    unsafe = [
        item
        for item in result.infeasibilities
        if item.workload_id == claimant.id and item.code is InfeasibilityCode.PREEMPTION_UNSAFE
    ]
    assert preempted, "a higher-priority workload must be able to reclaim spot capacity"
    assert not unsafe
    assert [victim.workload_id for victim in preempted[0].victims] == [victim.id]


def _lease(workload: WorkloadRecord) -> LeaseRecord:
    return LeaseRecord(
        id="lease-1",
        workload_id=workload.id,
        cell_id="cell-a",
        org_id=workload.org_id,
        project_id=workload.project_id,
        state=LeaseState.ACTIVE,
        resources=workload.spec.gang,
        hourly_cost=1.0,
        reserved_cost=1.0,
        resource_hours_reserved=ResourceHours(cpu=8.0, memory_gib=8.0, gpu=0.0, tpu=0.0),
        generation=1,
        fencing_token="fence",
        expires_at=utcnow() + timedelta(minutes=5),
    )


def test_a_spot_landed_lease_is_force_terminable_even_when_it_requested_on_demand() -> None:
    """The landed cell decides eligibility, not the requested class.

    ``revocable`` keyed off the landed class while the checkpoint policy kept
    keying off the requested one, so this lease was revocable yet failed the
    ``checkpointable or allow_forced_termination`` victim gate: the scheduler
    answered PREEMPTION_UNSAFE for capacity interruptible by definition.
    """

    workload = _record("wl", _spec(capacity_class=CapacityClass.ON_DEMAND), WorkloadState.RUNNING)
    lease = to_core_lease(_lease(workload), workload, landed_capacity_class=CapacityClass.SPOT)

    assert lease.revocable is True
    assert lease.checkpoint_policy.allow_forced_termination is True


def test_a_lease_landed_on_on_demand_is_never_force_terminable() -> None:
    """The same rule protects the other direction: spot request, on-demand landing."""

    workload = _record("wl", _spec(capacity_class=CapacityClass.SPOT), WorkloadState.RUNNING)
    lease = to_core_lease(_lease(workload), workload, landed_capacity_class=CapacityClass.ON_DEMAND)

    assert lease.revocable is False
    assert lease.checkpoint_policy.allow_forced_termination is False


def test_without_a_landed_class_the_requested_class_decides() -> None:
    workload = _record("wl", _spec(capacity_class=CapacityClass.SPOT), WorkloadState.RUNNING)
    lease = to_core_lease(_lease(workload), workload)

    assert lease.checkpoint_policy.allow_forced_termination is True


def test_a_higher_priority_workload_reclaims_spot_capacity_a_portfolio_landed_on() -> None:
    """The portfolio case the requested-class policy missed.

    The victim requested on-demand with spot as an additional class and landed
    on the spot cell; under the requested-class policy its checkpoint policy was
    never force-terminable and the claimant saw PREEMPTION_UNSAFE.
    """

    victim_spec = _spec(
        capacity_class=CapacityClass.ON_DEMAND,
        additional_capacity_classes=(CapacityClass.SPOT,),
        band=PriorityBand.BATCH,
    )
    victim = _record("wl-victim", victim_spec, WorkloadState.RUNNING)
    claimant = _record(
        "wl-claimant",
        _spec(capacity_class=CapacityClass.SPOT, band=PriorityBand.PRODUCTION),
        WorkloadState.PENDING,
    )
    cell = CellRecord(
        id="cell-a",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=8, memory_gib=8),
        allocatable=ResourceVector(cpu=8, memory_gib=8),
        capacity_class=CapacityClass.SPOT,
    )
    lease = _lease(victim)

    result = CoreSchedulingService().schedule([victim, claimant], [cell], leases=[lease])

    preempted = [plan for plan in result.preemptions if plan.workload_id == claimant.id]
    unsafe = [
        item
        for item in result.infeasibilities
        if item.workload_id == claimant.id and item.code is InfeasibilityCode.PREEMPTION_UNSAFE
    ]
    assert preempted, "spot capacity is reclaimable no matter which class was requested"
    assert not unsafe
    assert [victim_item.workload_id for victim_item in preempted[0].victims] == [victim.id]
