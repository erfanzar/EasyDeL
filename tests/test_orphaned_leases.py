# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reclaiming a lease that no live workload claims.

The orphaned-*workload* sweep settles by workload. So a lease its workload
finished with, vanished from, or retried away from is reachable from no path at
all: it keeps a cell allocated and its cost unsettled indefinitely -- capacity
the fleet pays for and no tenant is charged for.

The gap was found by the invariants module rather than by reading. Wiring
``check_fleet`` into a chaos scenario turned "the workload settled correctly"
into "the fleet is whole", and the second assertion failed on a lease the first
could not see.

The dangerous direction here is the opposite one. Releasing a lease that is
*not* orphaned kills live work on real accelerators, so the tests that assert
silence carry as much weight as the ones that assert reclamation.
"""

from __future__ import annotations

from datetime import timedelta

from eyvan.config import EyvanSettings
from eyvan.records import (
    BudgetRecord,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.invariants import Invariant, check_fleet
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService

ADMIN = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _settings() -> EyvanSettings:
    return EyvanSettings()


def _stale() -> object:
    """A moment safely older than the sweep's grace period."""

    return utcnow() - timedelta(seconds=_settings().lease_ttl_s * 4)


async def _fleet(
    *,
    workload_state: WorkloadState | None = WorkloadState.RUNNING,
    claims_lease: bool = True,
    lease_state: LeaseState = LeaseState.ACTIVE,
    lease_updated_at=None,
):
    """Build one cell, one lease, and optionally the workload that claims it."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service, settings=_settings())
    async with repository.transaction() as tx:
        await tx.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(cpu=4),
                allocatable=ResourceVector(cpu=4),
                endpoint="http://agent:8090",
            )
        )
        await tx.put_budget(
            BudgetRecord(org_id="acme", project_id="research", limit=1000.0, spent=0.0, updated_by="root")
        )
        if workload_state is not None:
            await tx.put_workload(
                WorkloadRecord(
                    id="wl-a",
                    org_id="acme",
                    project_id="research",
                    spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
                    state=workload_state,
                    created_by="alice",
                    lease_id="lease-a" if claims_lease else None,
                    lease_ids=("lease-a",) if claims_lease else (),
                )
            )
        await tx.put_lease(
            LeaseRecord(
                id="lease-a",
                workload_id="wl-a",
                cell_id="cell-a",
                org_id="acme",
                project_id="research",
                resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
                state=lease_state,
                fencing_token="cell-a:1",
                hourly_cost=10.0,
                expires_at=utcnow() + timedelta(hours=1),
                updated_at=lease_updated_at or _stale(),
            )
        )
    return repository, controller


async def _lease(repository) -> LeaseRecord | None:
    async with repository.transaction() as tx:
        return await tx.get_lease("lease-a")


# --------------------------------------------------------------------------- reclaimed


async def test_a_lease_whose_workload_finished_is_released() -> None:
    """The leak the invariants module surfaced.

    Nothing else reaches this lease: the workload sweep looks at workloads, and
    this workload is already correctly terminal.
    """

    repository, controller = await _fleet(workload_state=WorkloadState.SUCCEEDED)

    assert await controller.settle_orphaned_leases_once() == 1

    lease = await _lease(repository)
    assert lease is not None
    assert lease.state is LeaseState.RELEASED
    assert lease.settled_at is not None


async def test_a_lease_whose_workload_vanished_is_released() -> None:
    """A missing workload is the same leak with less evidence."""

    repository, controller = await _fleet(workload_state=None)

    assert await controller.settle_orphaned_leases_once() == 1

    lease = await _lease(repository)
    assert lease is not None
    assert lease.state is LeaseState.RELEASED


async def test_a_lease_a_live_workload_moved_on_from_is_released() -> None:
    """The third shape, and the one a retry produces routinely.

    The workload is alive and healthy on different capacity. Only the lease it
    left behind is stranded, so a check that looked only at workload state would
    call this fleet clean.
    """

    repository, controller = await _fleet(workload_state=WorkloadState.RUNNING, claims_lease=False)

    assert await controller.settle_orphaned_leases_once() == 1

    lease = await _lease(repository)
    assert lease is not None
    assert lease.state is LeaseState.RELEASED


async def test_the_reclaimed_cost_is_charged_rather_than_discarded() -> None:
    """Releasing without settling would trade a capacity leak for an accounting one.

    The fleet would stop paying for the cell and the tenant would never be
    charged for the hours it held -- which I5 would then report as a budget
    disagreeing with its settled leases.
    """

    repository, controller = await _fleet(workload_state=WorkloadState.SUCCEEDED)

    assert await controller.settle_orphaned_leases_once() == 1

    async with repository.transaction() as tx:
        budget = await tx.get_budget("acme", "research")
        lease = await tx.get_lease("lease-a")
    assert lease is not None
    assert budget is not None
    assert lease.settled_cost > 0
    assert budget.spent == lease.settled_cost


async def test_the_sweep_closes_the_invariant_that_found_it() -> None:
    """The whole point, asserted through the checker rather than the record.

    Before: I2 reports an active lease on a finished workload. After: nothing.
    """

    repository, controller = await _fleet(workload_state=WorkloadState.SUCCEEDED)

    before = await check_fleet(repository)
    assert [violation.invariant for violation in before] == [Invariant.NO_LEASES_OUTLIVING_WORKLOADS]

    await controller.settle_orphaned_leases_once()

    assert await check_fleet(repository) == ()


# --------------------------------------------------------------------------- left alone


async def test_a_lease_a_running_workload_still_claims_is_untouched() -> None:
    """The dangerous direction.

    Releasing this kills live work on real accelerators, so it matters more than
    any reclamation the sweep performs.
    """

    repository, controller = await _fleet(workload_state=WorkloadState.RUNNING)

    assert await controller.settle_orphaned_leases_once() == 0

    lease = await _lease(repository)
    assert lease is not None
    assert lease.state is LeaseState.ACTIVE


async def test_a_freshly_updated_lease_waits_out_its_grace_period() -> None:
    """A dispatch mid-way through attaching a lease looks exactly like an orphan.

    Nothing should write a lease and its workload in separate transactions, but
    the cost of being wrong is destroying live work -- so one cycle of patience
    removes the entire class of "a race made this look orphaned".
    """

    repository, controller = await _fleet(workload_state=WorkloadState.SUCCEEDED, lease_updated_at=utcnow())

    assert await controller.settle_orphaned_leases_once() == 0

    lease = await _lease(repository)
    assert lease is not None
    assert lease.state is LeaseState.ACTIVE


async def test_an_already_released_lease_is_not_settled_twice() -> None:
    """Double settlement would charge a tenant twice for the same hours."""

    _repository, controller = await _fleet(workload_state=WorkloadState.SUCCEEDED, lease_state=LeaseState.RELEASED)

    assert await controller.settle_orphaned_leases_once() == 0


async def test_a_direct_call_on_a_settled_lease_does_nothing() -> None:
    """The sweep filters on active states, so this guard is only reachable directly.

    It still has to hold: a second settlement charges a tenant twice for the same
    hours, and the sweep is not the only caller a future phase might add.
    """

    repository, controller = await _fleet(workload_state=WorkloadState.SUCCEEDED, lease_state=LeaseState.RELEASED)
    async with repository.transaction() as tx:
        lease = await tx.get_lease("lease-a")
    assert lease is not None

    settled = await controller.service.settle_orphaned_lease(
        ADMIN, "lease-a", reason="orphaned: test", expected_generation=lease.generation
    )

    assert settled is False
    async with repository.transaction() as tx:
        budget = await tx.get_budget("acme", "research")
    assert budget is not None
    assert budget.spent == 0.0


async def test_a_lease_re_attached_between_the_scan_and_the_write_is_left_alone() -> None:
    """The race the generation check exists for.

    The scan sees an orphan; a dispatch attaches the lease before the write. A
    sweep that trusted its scan would release capacity a workload had just been
    given.
    """

    repository, controller = await _fleet(workload_state=WorkloadState.SUCCEEDED)
    async with repository.transaction() as tx:
        lease = await tx.get_lease("lease-a")
        assert lease is not None
        # Only the generation moves. The workload stays terminal, so the lease is
        # still orphaned by every other measure -- which makes the generation
        # check the one thing standing between the scan and a release.
        await tx.put_lease(lease.model_copy(update={"generation": lease.generation + 1}))

    settled = await controller.service.settle_orphaned_lease(
        ADMIN, "lease-a", reason="orphaned: test", expected_generation=lease.generation
    )

    assert settled is False
    current = await _lease(repository)
    assert current is not None
    assert current.state is LeaseState.ACTIVE


# --------------------------------------------------------------------------- reachable


def test_the_sweep_runs_in_the_leader_loop() -> None:
    """A reclamation phase nothing schedules reclaims nothing.

    Which is how the gap it fixes came to exist: the workload sweep was wired
    and its lease counterpart was never written, so half the condition was swept
    on a timer and the other half never at all.
    """

    controller = EyvanController(InMemoryRepository(), ControlPlaneService(InMemoryRepository()))
    phases = [phase.name for phase in controller._leader_phases()]

    assert "settle-orphan-leases" in phases
    # After the workload sweep: a workload settled there becomes terminal, which
    # is exactly what makes its leftover leases visible here in the same cycle.
    assert phases.index("settle-orphan-leases") > phases.index("settle-orphans")
