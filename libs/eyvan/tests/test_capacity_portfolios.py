# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Capacity portfolios: the scheduler already had them; the adapter hid them.

``WorkloadSpec.allowed_capacity_classes`` and ``capacity_class_preference`` have
always existed in the pure core, normalized and deduplicated, defaulting to
``(RESERVED, ON_DEMAND, SPOT)`` -- and ``capacity_class_rank`` has always been the
**fourth key** of the lexicographic ``PlacementScore``. The only thing collapsing
that to one class was the service adapter. So this is a public-API and adapter
change, not a scheduler change, and the tests are shaped accordingly: they assert
what the adapter now hands the scheduler, and what the scheduler already did with
it once it could see it.

Two things get special attention. ``capacity_class_rank`` had **zero test
coverage** before this -- the entire ordered-preference mechanism -- which is
defect D2 and is why nothing noticed it was unreachable. And ``to_core_lease``
computes ``revocable`` from the *landed* cell's class rather than the requested
one, without which a workload that lists ``spot`` in its portfolio and lands on
an **on-demand** cell becomes force-terminable: a silent promise violation.
"""

from __future__ import annotations

import pytest

from eyvan.domain import CapacityClass as CoreCapacityClass
from eyvan.records import (
    CapacityClass,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    PolicyRecord,
    ResourceGang,
    ResourceLimits,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.service.scheduling import (
    CoreSchedulingService,
    landed_capacity_classes,
    to_core_lease,
    to_core_workload,
)


def _spec(**overrides: object) -> WorkloadDefinition:
    defaults: dict[str, object] = {
        "name": "train",
        "execution": CommandExecution(kind="command", command="python train.py"),
        "gang": ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0, tpu=4)),
    }
    return WorkloadDefinition(**{**defaults, **overrides})


def _workload(spec: WorkloadDefinition, workload_id: str = "wl-1") -> WorkloadRecord:
    return WorkloadRecord(id=workload_id, org_id="acme", project_id="research", spec=spec, created_by="alice")


def _cell(
    cell_id: str,
    *,
    capacity_class: CapacityClass,
    region: str = "us-east5",
    zone: str | None = None,
    accelerator_type: str = "v5p-16",
) -> CellRecord:
    return CellRecord(
        id=cell_id,
        provider="manual",
        capacity_backend="manual",
        region=region,
        zone=zone,
        state=CellState.READY,
        capacity=ResourceLimits(cpu=8.0, memory_gib=64.0, tpu=8),
        allocatable=ResourceLimits(cpu=8.0, memory_gib=64.0, tpu=8),
        unit_prices={"tpu": 1.0},
        accelerator_type=accelerator_type,
        capacity_class=capacity_class,
    )


def _lease(cell_id: str, workload_id: str = "wl-1") -> LeaseRecord:
    return LeaseRecord(
        id="lease-1",
        workload_id=workload_id,
        cell_id=cell_id,
        org_id="acme",
        project_id="research",
        state=LeaseState.ACTIVE,
        resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
        generation=1,
        fencing_token="token-1",
        expires_at=utcnow(),
    )


# --- 3a: widening the classes -----------------------------------------------


def test_a_spec_with_no_additional_classes_is_still_a_single_class_request() -> None:
    """Additive by design: every existing spec keeps its exact behaviour."""

    core = to_core_workload(_workload(_spec(capacity_class=CapacityClass.ON_DEMAND)))

    assert core.capacity_class_preference == (CoreCapacityClass.ON_DEMAND,)
    assert core.allowed_capacity_classes == frozenset({CoreCapacityClass.ON_DEMAND})


def test_the_portfolio_reaches_the_scheduler_in_declared_order() -> None:
    """Declared order, not alphabetical.

    The core *repairs* a short preference by appending the remaining allowed
    classes sorted by name, so an order that happens to match alphabetical order
    cannot tell a preserved preference from a discarded one. These three are
    deliberately declared out of alphabetical order.
    """

    spec = _spec(
        capacity_class=CapacityClass.SPOT,
        additional_capacity_classes=(CapacityClass.RESERVED, CapacityClass.ON_DEMAND),
    )

    core = to_core_workload(_workload(spec))

    assert core.capacity_class_preference == (
        CoreCapacityClass.SPOT,
        CoreCapacityClass.RESERVED,
        CoreCapacityClass.ON_DEMAND,
    )
    assert len(core.allowed_capacity_classes) == 3


def test_a_repeated_class_is_refused_at_the_boundary() -> None:
    """A duplicate would make ``capacity_class_rank`` ambiguous for that class."""

    with pytest.raises(ValueError, match="repeat"):
        _spec(capacity_class=CapacityClass.SPOT, additional_capacity_classes=(CapacityClass.SPOT,))


# --- D2: capacity_class_rank had zero coverage ------------------------------


def test_the_preferred_class_wins_among_otherwise_identical_cells() -> None:
    """D2: this ranking mechanism shipped with no test at all."""

    scheduler = CoreSchedulingService()
    spec = _spec(
        capacity_class=CapacityClass.RESERVED,
        additional_capacity_classes=(CapacityClass.ON_DEMAND, CapacityClass.SPOT),
    )
    cells = [
        _cell("cell-spot", capacity_class=CapacityClass.SPOT),
        _cell("cell-on-demand", capacity_class=CapacityClass.ON_DEMAND),
        _cell("cell-reserved", capacity_class=CapacityClass.RESERVED),
    ]

    result = scheduler.schedule([_workload(spec)], cells)

    assert [placement.lease.cell_id for placement in result.placements] == ["cell-reserved"]


def test_reversing_the_preference_reverses_the_placement() -> None:
    """Otherwise the previous test could pass on cell ordering alone."""

    scheduler = CoreSchedulingService()
    spec = _spec(
        capacity_class=CapacityClass.SPOT,
        additional_capacity_classes=(CapacityClass.ON_DEMAND, CapacityClass.RESERVED),
    )
    cells = [
        _cell("cell-spot", capacity_class=CapacityClass.SPOT),
        _cell("cell-on-demand", capacity_class=CapacityClass.ON_DEMAND),
        _cell("cell-reserved", capacity_class=CapacityClass.RESERVED),
    ]

    result = scheduler.schedule([_workload(spec)], cells)

    assert [placement.lease.cell_id for placement in result.placements] == ["cell-spot"]


def test_a_second_preference_beats_a_third_one() -> None:
    """Also declared out of alphabetical order, so a repaired tail would lose."""

    scheduler = CoreSchedulingService()
    spec = _spec(
        capacity_class=CapacityClass.SPOT,
        additional_capacity_classes=(CapacityClass.RESERVED, CapacityClass.ON_DEMAND),
    )
    cells = [
        _cell("cell-on-demand", capacity_class=CapacityClass.ON_DEMAND),
        _cell("cell-reserved", capacity_class=CapacityClass.RESERVED),
    ]

    result = scheduler.schedule([_workload(spec)], cells)

    assert [placement.lease.cell_id for placement in result.placements] == ["cell-reserved"]


def test_a_class_outside_the_portfolio_is_not_placed_on() -> None:
    scheduler = CoreSchedulingService()
    spec = _spec(capacity_class=CapacityClass.RESERVED)

    result = scheduler.schedule([_workload(spec)], [_cell("cell-spot", capacity_class=CapacityClass.SPOT)])

    assert result.placements == ()


def test_a_second_choice_class_is_used_when_the_first_is_absent() -> None:
    """The point of a portfolio: the run lands instead of waiting."""

    scheduler = CoreSchedulingService()
    spec = _spec(capacity_class=CapacityClass.RESERVED, additional_capacity_classes=(CapacityClass.SPOT,))

    result = scheduler.schedule([_workload(spec)], [_cell("cell-spot", capacity_class=CapacityClass.SPOT)])

    assert [placement.lease.cell_id for placement in result.placements] == ["cell-spot"]


# --- 3a: the revocable trap -------------------------------------------------


def test_a_spot_request_landing_on_an_on_demand_cell_is_not_force_terminable() -> None:
    """The trap. Without the landed class, this lease reads as preemptible.

    Nothing about an on-demand cell is preemptible, so marking its lease
    force-terminable is a promise violation the workload never agreed to -- and it
    is silent, because the lease record carries no capacity class of its own.
    """

    spec = _spec(
        capacity_class=CapacityClass.SPOT,
        additional_capacity_classes=(CapacityClass.ON_DEMAND,),
        checkpointable=False,
    )

    landed_on_demand = to_core_lease(_lease("cell-a"), _workload(spec), None, CapacityClass.ON_DEMAND)
    landed_spot = to_core_lease(_lease("cell-a"), _workload(spec), None, CapacityClass.SPOT)

    assert landed_on_demand.revocable is False
    assert landed_spot.revocable is True


def test_omitting_the_landed_class_falls_back_to_the_requested_one() -> None:
    """Exactly right for the single-class case, which is every existing caller."""

    spec = _spec(capacity_class=CapacityClass.SPOT, checkpointable=False)

    assert to_core_lease(_lease("cell-a"), _workload(spec)).revocable is True


def test_a_checkpointable_workload_stays_revocable_on_any_class() -> None:
    """Checkpointing is a property of the workload, not of the capacity it got."""

    spec = _spec(
        capacity_class=CapacityClass.ON_DEMAND,
        checkpointable=True,
        checkpoint_hook="save",
        checkpoint_output_uri="gs://acme/ckpt",
    )

    lease = to_core_lease(_lease("cell-a"), _workload(spec), None, CapacityClass.RESERVED)

    assert lease.revocable is True


def test_policy_still_overrides_the_landed_class() -> None:
    """A project that disabled preemption keeps it disabled however the run landed."""

    spec = _spec(capacity_class=CapacityClass.SPOT, checkpointable=False)
    policy = PolicyRecord(org_id="acme", project_id="research", preemption_enabled=False, updated_by="root")

    lease = to_core_lease(_lease("cell-a"), _workload(spec), policy, CapacityClass.SPOT)

    assert lease.revocable is False


def test_the_join_that_makes_the_landed_class_reachable() -> None:
    """A lease record carries no capacity class; this join is where it comes from.

    Asserted on the named helper rather than through a placement outcome. The
    *decision* it feeds is covered directly above, and a scheduling fixture large
    enough to observe a preemption would end up testing the preemption planner
    rather than this wiring -- a first attempt at that produced zero preemptions
    and asserted nothing at all.
    """

    cells = [
        _cell("cell-spot", capacity_class=CapacityClass.SPOT),
        _cell("cell-on-demand", capacity_class=CapacityClass.ON_DEMAND),
    ]

    landed = landed_capacity_classes(cells)

    assert landed == {
        "cell-spot": CapacityClass.SPOT,
        "cell-on-demand": CapacityClass.ON_DEMAND,
    }
    # And it composes with the decision: one lease, one workload, opposite answers
    # depending purely on which cell it landed on.
    spec = _spec(capacity_class=CapacityClass.SPOT, checkpointable=False)
    assert to_core_lease(_lease("cell-a"), _workload(spec), None, landed["cell-on-demand"]).revocable is False
    assert to_core_lease(_lease("cell-a"), _workload(spec), None, landed["cell-spot"]).revocable is True


# --- 3b/3c: widening zone, region, accelerator ------------------------------


def test_allowed_regions_reach_the_scheduler_as_label_values() -> None:
    core = to_core_workload(_workload(_spec(allowed_regions=("us-east5", "europe-west4"))))

    # The core normalizes each value tuple, so compare membership rather than
    # declaration order: the scheduler treats these as a set of acceptable values.
    regions = {label: set(values) for label, values in core.allowed_label_values}["region"]
    assert regions == {"us-east5", "europe-west4"}


def test_allowed_zones_and_accelerator_types_do_too() -> None:
    spec = _spec(allowed_zones=("us-east5-a",), allowed_accelerator_types=("v5p-16", "v5p-32"))

    core = to_core_workload(_workload(spec))

    by_label = {label: set(values) for label, values in core.allowed_label_values}
    assert by_label["zone"] == {"us-east5-a"}
    assert by_label["accelerator_type"] == {"v5p-16", "v5p-32"}


def test_a_pin_and_a_list_are_mutually_exclusive() -> None:
    """Otherwise "which wins" is decided by whichever check happens to run first."""

    with pytest.raises(ValueError, match="never both"):
        _spec(region="us-east5", allowed_regions=("europe-west4",))
    with pytest.raises(ValueError, match="never both"):
        _spec(accelerator_type="v5p-16", allowed_accelerator_types=("v5p-32",))


def test_a_workload_list_takes_precedence_over_the_project_policy_list() -> None:
    """A policy says what a project *may* use; a spec says what this run accepts."""

    policy = PolicyRecord(
        org_id="acme",
        project_id="research",
        allowed_regions=("us-east5", "europe-west4", "asia-northeast1"),
        updated_by="root",
    )

    core = to_core_workload(_workload(_spec(allowed_regions=("us-east5",))), policy)

    region_values = [set(values) for label, values in core.allowed_label_values if label == "region"]
    assert region_values == [{"us-east5"}]


def test_a_run_only_lands_in_a_region_it_listed() -> None:
    scheduler = CoreSchedulingService()
    spec = _spec(allowed_regions=("europe-west4",), capacity_class=CapacityClass.ON_DEMAND)
    cells = [
        _cell("cell-us", capacity_class=CapacityClass.ON_DEMAND, region="us-east5"),
        _cell("cell-eu", capacity_class=CapacityClass.ON_DEMAND, region="europe-west4"),
    ]

    result = scheduler.schedule([_workload(spec)], cells)

    assert [placement.lease.cell_id for placement in result.placements] == ["cell-eu"]


def test_widening_accelerator_types_does_not_make_a_fixed_shape_job_portable() -> None:
    """The documented limitation, asserted so it cannot be quietly assumed away.

    ``_static_cell_failure`` requires ``worker_count`` equality and ``fits_within``,
    so a whole-slice request that fills a v5p-16 does not fit a differently-sized
    cell just because both names appear in the list. This is plumbing; shape
    adaptation is not built.
    """

    scheduler = CoreSchedulingService()
    spec = _spec(
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0, tpu=8)),
        allowed_accelerator_types=("v5p-16", "v5p-32"),
        capacity_class=CapacityClass.ON_DEMAND,
    )
    too_small = _cell("cell-small", capacity_class=CapacityClass.ON_DEMAND, accelerator_type="v5p-32")
    too_small = too_small.model_copy(update={"allocatable": ResourceLimits(cpu=8.0, memory_gib=64.0, tpu=4)})

    result = scheduler.schedule([_workload(spec)], [too_small])

    assert result.placements == (), "a listed name does not resize the request"


# --- 3d: reservations are free once the portfolio widens --------------------


def test_reserved_capacity_needs_no_new_concept() -> None:
    """CapacityClass.RESERVED already exists and already ranks first by default."""

    scheduler = CoreSchedulingService()
    spec = _spec(
        capacity_class=CapacityClass.RESERVED,
        additional_capacity_classes=(CapacityClass.ON_DEMAND,),
    )
    cells = [
        _cell("cell-on-demand", capacity_class=CapacityClass.ON_DEMAND),
        _cell("cell-reserved", capacity_class=CapacityClass.RESERVED),
    ]

    result = scheduler.schedule([_workload(spec)], cells)

    assert [placement.lease.cell_id for placement in result.placements] == ["cell-reserved"]


# --- 3e: failover on queue-admission denial ---------------------------------


@pytest.mark.asyncio
async def test_a_denial_does_not_terminally_fail_a_workload_with_other_pools() -> None:
    """A denial says this shape in this zone is unavailable, not that the run is."""

    from eyvan.records import AutoscaleBlockReason, CapacityPoolRecord
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence import InMemoryRepository
    from eyvan.server.service import ControlPlaneService

    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository=repository)
    controller = EyvanController(repository=repository, service=service)
    spec = _spec(
        capacity_class=CapacityClass.SPOT,
        additional_capacity_classes=(CapacityClass.ON_DEMAND,),
        allowed_regions=("us-east5", "europe-west4"),
    )
    workload = _workload(spec)
    denied = _cell("cell-denied", capacity_class=CapacityClass.SPOT).model_copy(
        update={"pool_id": "research/spot", "autoscale_request_workload_id": workload.id}
    )
    async with repository.transaction() as tx:
        await tx.put_workload(workload)
        await tx.put_cell(denied)
        for pool_id, capacity_class, region in (
            ("research/spot", CapacityClass.SPOT, "us-east5"),
            ("research/on-demand", CapacityClass.ON_DEMAND, "europe-west4"),
        ):
            await tx.put_capacity_pool(
                CapacityPoolRecord(
                    id=pool_id,
                    fleet_name="research",
                    node_type="v5p-16",
                    provider="manual",
                    capacity_backend="manual",
                    region=region,
                    agent_endpoint="http://agent:8080",
                    capacity=ResourceLimits(cpu=8.0, memory_gib=64.0, tpu=8),
                    allocatable=ResourceLimits(cpu=8.0, memory_gib=64.0, tpu=8),
                    unit_prices={"tpu": 1.0},
                    accelerator_type="v5p-16",
                    capacity_class=capacity_class,
                )
            )

    async with repository.transaction() as tx:
        await controller._fail_autoscale_workload_for_queue_admission_denial(tx, denied, "zone is full")

    async with repository.transaction() as tx:
        stored = await tx.get_workload(workload.id)
    assert stored is not None
    assert stored.state is WorkloadState.PENDING, "another pool remains; the run is not over"
    assert stored.autoscale_block_reason is AutoscaleBlockReason.QUEUE_ADMISSION_DENIED
    assert "other admissible pools remain" in (stored.scheduling_reason or "")


@pytest.mark.asyncio
async def test_a_denial_still_fails_a_workload_with_nowhere_else_to_go() -> None:
    """The single-candidate case keeps its terminal failure; the change is additive."""

    from eyvan.records import CapacityPoolRecord
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence import InMemoryRepository
    from eyvan.server.service import ControlPlaneService

    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository=repository)
    controller = EyvanController(repository=repository, service=service)
    workload = _workload(_spec(capacity_class=CapacityClass.SPOT))
    denied = _cell("cell-denied", capacity_class=CapacityClass.SPOT).model_copy(
        update={"pool_id": "research/spot", "autoscale_request_workload_id": workload.id}
    )
    async with repository.transaction() as tx:
        await tx.put_workload(workload)
        await tx.put_cell(denied)
        await tx.put_capacity_pool(
            CapacityPoolRecord(
                id="research/spot",
                fleet_name="research",
                node_type="v5p-16",
                provider="manual",
                capacity_backend="manual",
                region="us-east5",
                agent_endpoint="http://agent:8080",
                capacity=ResourceLimits(cpu=8.0, memory_gib=64.0, tpu=8),
                allocatable=ResourceLimits(cpu=8.0, memory_gib=64.0, tpu=8),
                unit_prices={"tpu": 1.0},
                accelerator_type="v5p-16",
                capacity_class=CapacityClass.SPOT,
            )
        )

    async with repository.transaction() as tx:
        await controller._fail_autoscale_workload_for_queue_admission_denial(tx, denied, "zone is full")

    async with repository.transaction() as tx:
        stored = await tx.get_workload(workload.id)
    assert stored is not None
    assert stored.state is WorkloadState.FAILED


@pytest.mark.asyncio
async def test_a_disabled_alternative_pool_does_not_count() -> None:
    """A pool an operator turned off is not somewhere the run could still land."""

    from eyvan.records import CapacityPoolRecord
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence import InMemoryRepository
    from eyvan.server.service import ControlPlaneService

    repository = InMemoryRepository()
    await repository.open()
    controller = EyvanController(repository=repository, service=ControlPlaneService(repository=repository))
    workload = _workload(
        _spec(capacity_class=CapacityClass.SPOT, additional_capacity_classes=(CapacityClass.ON_DEMAND,))
    )
    denied = _cell("cell-denied", capacity_class=CapacityClass.SPOT).model_copy(
        update={"pool_id": "research/spot", "autoscale_request_workload_id": workload.id}
    )
    async with repository.transaction() as tx:
        await tx.put_workload(workload)
        await tx.put_cell(denied)
        await tx.put_capacity_pool(
            CapacityPoolRecord(
                id="research/off",
                fleet_name="research",
                node_type="v5p-16",
                provider="manual",
                capacity_backend="manual",
                region="us-east5",
                agent_endpoint="http://agent:8080",
                capacity=ResourceLimits(cpu=8.0, memory_gib=64.0, tpu=8),
                allocatable=ResourceLimits(cpu=8.0, memory_gib=64.0, tpu=8),
                unit_prices={"tpu": 1.0},
                accelerator_type="v5p-16",
                capacity_class=CapacityClass.ON_DEMAND,
                enabled=False,
            )
        )

    async with repository.transaction() as tx:
        await controller._fail_autoscale_workload_for_queue_admission_denial(tx, denied, "zone is full")

    async with repository.transaction() as tx:
        stored = await tx.get_workload(workload.id)
    assert stored is not None
    assert stored.state is WorkloadState.FAILED
