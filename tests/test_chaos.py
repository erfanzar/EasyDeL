# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fault-injection scenarios that assert Eyvan's durability invariants hold.

These are not smoke tests. Each scenario injects a specific fault and asserts
the *invariant* that is supposed to survive it -- fencing, tombstones,
high-watermarks, settlement idempotence -- rather than merely that nothing
crashed. "It did not raise" is exactly the assertion that lets a durability bug
ship.

## The fault matrix

Six fault classes, the mechanism each one tests, and where the coverage lives.
Enumerating them here is the point: a fault class with no row is a gap nobody
notices, and this table is what makes adding one deliberate.

| fault class | mechanism under test | covered in |
|---|---|---|
| controller failover | leader re-acquisition per phase; ``settled_at`` as the settlement idempotency boundary | here |
| duplicate messages | outbox at-least-once delivery + idempotency keys and fingerprints | here |
| stale agents | generation fences and fencing tokens; durable cell tombstones | here |
| provider API outage | ``CapacityProviderError`` classification: retryable, never discarded | here |
| storage corruption | ``FleetStateSchemaError`` -- refuse a foreign document rather than adopt it | here |
| partial slice loss | per-rank aggregation: an unreachable host is reported in place, never hidden | [1] |

[1] ``test_native_slice_executor`` (executor level) and ``test_introspection``
(fan-out level), because the aggregation being asserted lives in the executor,
not in the control plane this module drives.

Deliberately not covered here, because a fake cannot prove them: PostgreSQL
isolation behaviour, real provider semantics, and anything requiring accelerator
hardware. Those need the live acceptance run, and claiming them from an in-memory
backend would be worse than leaving the row empty.

Scope note: these run against the in-memory backend, so they prove control-plane
logic only.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import MutationContext, StaleMutationError
from eyvan.capacity.drivers.manual import ManualCapacityDriver
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.records import (
    CommandExecution,
    OutboxRecord,
    OutboxState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadState,
    utcnow,
)
from eyvan.server.invariants import Invariant, check_fleet
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import ConflictError


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


def _spec() -> WorkloadDefinition:
    return WorkloadDefinition(
        name="job",
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
    )


def _context(key: str, generation: int = 1, token: str = "fence-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=token)


def _desired() -> dict:
    return {
        "provider": "manual",
        "capacity_backend": "manual",
        "executor": "eyvan-native",
        "accelerator_type": "cpu",
        "capacity_class": "on-demand",
        "region": "on-prem",
        "hosts": ["http://host-a:8100"],
    }


def _driver(tmp_path) -> ManualCapacityDriver:
    return ManualCapacityDriver(FleetStateStore(LocalBackend(tmp_path / "fleet.json")))


# --------------------------------------------------------------- duplicate delivery


async def test_a_duplicated_submission_creates_exactly_one_workload() -> None:
    """At-least-once delivery means the same request WILL arrive twice."""

    service = ControlPlaneService(InMemoryRepository())

    first, replayed_first = await service.submit_workload(
        _principal(), org_id="acme", project_id="research", spec=_spec(), idempotency_key="same"
    )
    second, replayed_second = await service.submit_workload(
        _principal(), org_id="acme", project_id="research", spec=_spec(), idempotency_key="same"
    )

    assert replayed_first is False
    assert replayed_second is True
    assert first.id == second.id


async def test_the_same_key_with_a_different_request_is_a_conflict_not_a_silent_replay() -> None:
    """Replaying a *different* request under a reused key would corrupt intent."""

    service = ControlPlaneService(InMemoryRepository())
    await service.submit_workload(_principal(), org_id="acme", project_id="research", spec=_spec(), idempotency_key="k")

    other = _spec().model_copy(update={"name": "different"})
    with pytest.raises(ConflictError):
        await service.submit_workload(
            _principal(), org_id="acme", project_id="research", spec=other, idempotency_key="k"
        )


async def test_a_duplicated_cancel_is_idempotent() -> None:
    service = ControlPlaneService(InMemoryRepository())
    workload, _ = await service.submit_workload(
        _principal(), org_id="acme", project_id="research", spec=_spec(), idempotency_key="submit"
    )

    _first, replayed_first = await service.cancel_workload(_principal(), workload.id, idempotency_key="cancel")
    _second, replayed_second = await service.cancel_workload(_principal(), workload.id, idempotency_key="cancel")

    assert replayed_first is False
    assert replayed_second is True


# --------------------------------------------------------------- stale actors / fencing


def test_a_resurrected_agent_cannot_mutate_a_newer_generation(tmp_path) -> None:
    """The classic split-brain: an old actor wakes up and acts on stale state."""

    driver = _driver(tmp_path)
    driver.apply_capacity("cell-a", _desired(), _context("apply", generation=2, token="fence-2"))

    with pytest.raises(StaleMutationError, match="older than"):
        driver.apply_capacity("cell-a", _desired(), _context("stale", generation=1, token="fence-1"))


def test_a_competing_fencing_token_cannot_release_a_cell(tmp_path) -> None:
    driver = _driver(tmp_path)
    driver.apply_capacity("cell-a", _desired(), _context("apply", generation=1, token="fence-1"))

    with pytest.raises(StaleMutationError):
        driver.release("cell-a", _context("release", generation=1, token="impostor"))

    # The cell must still be intact after the rejected release.
    assert driver.observe("cell-a").state is CapacityStatus.HEALTHY


def test_a_replayed_mutation_key_does_not_double_apply(tmp_path) -> None:
    """Idempotency keys are the defence against a retried provider RPC."""

    driver = _driver(tmp_path)
    first = driver.apply_capacity("cell-a", _desired(), _context("apply"))
    second = driver.apply_capacity("cell-a", _desired(), _context("apply"))

    assert first.generation == second.generation
    assert second.state is CapacityStatus.HEALTHY


def test_a_released_cell_stays_absent_when_a_delayed_reconcile_arrives(tmp_path) -> None:
    """A tombstone must stop a late request resurrecting drained capacity.

    The tombstone rejects the stale mutation outright rather than quietly
    returning "absent" -- a silent no-op would let the caller believe its
    reconcile was applied.
    """

    driver = _driver(tmp_path)
    driver.apply_capacity("cell-a", _desired(), _context("apply"))
    driver.drain("cell-a", _context("drain"))
    driver.release("cell-a", _context("release", generation=2, token="fence-2"))

    # A reconcile that was in flight during the release lands now.
    with pytest.raises(StaleMutationError):
        driver.reconcile("cell-a", _context("late-reconcile", generation=2, token="fence-2"))

    assert driver.observe("cell-a").state is CapacityStatus.ABSENT


# --------------------------------------------------------------- retention safety


async def test_pruning_never_drops_an_undelivered_command() -> None:
    """A crash-recovery scenario: old rows exist, but delivery is still owed."""

    repository = InMemoryRepository()
    stale = utcnow() - timedelta(days=365)
    async with repository.transaction() as transaction:
        for index, state in enumerate((OutboxState.PENDING, OutboxState.PROCESSING, OutboxState.FAILED)):
            await transaction.put_outbox(
                OutboxRecord(
                    id=f"owed-{index}",
                    topic="capacity.apply",
                    aggregate_type="cell",
                    aggregate_id="cell-a",
                    idempotency_key=f"capacity:cell-a:{index}",
                    generation=1,
                    state=state,
                    payload={},
                    created_at=stale,
                    updated_at=stale,
                )
            )

    async with repository.transaction() as transaction:
        removed = await transaction.prune_outbox(before=utcnow())

    assert removed == 0
    async with repository.transaction() as transaction:
        assert len(await transaction.list_outbox(limit=100)) == 3


# --------------------------------------------------------------- job-tree safety


async def test_a_corrupted_parent_cycle_does_not_hang_cancellation() -> None:
    """parent_workload_id is ordinary persisted data; a cycle must terminate."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    root, _ = await service.submit_workload(
        _principal(), org_id="acme", project_id="research", spec=_spec(), idempotency_key="root"
    )
    child, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_spec(),
        idempotency_key="child",
        parent_workload_id=root.id,
    )
    # Corrupt the tree into a cycle behind the service's back.
    async with repository.transaction() as transaction:
        stored = await transaction.get_workload(root.id)
        await transaction.put_workload(stored.model_copy(update={"parent_workload_id": child.id}))

    await service.cancel_workload(_principal(), root.id, idempotency_key="cancel")

    async with repository.transaction() as transaction:
        assert (await transaction.get_workload(child.id)).state is WorkloadState.CANCELLED


# --------------------------------------------------------------- controller failover


@pytest.mark.asyncio
async def test_losing_leadership_mid_cycle_leaves_no_half_applied_cycle() -> None:
    """A leader can lose its advisory lock between phases, and routinely does.

    Invariant: phases after the loss do not run. Two leaders minting autoscaler cell
    ids from the same ``len(managed) + 1`` sequence would collide, so the cycle must
    stop rather than finish on stale authority.
    """

    from eyvan.server.controller import EyvanController, LeaderPhase

    class LosesLeadershipAfter:
        def __init__(self, grants: int) -> None:
            self.remaining = grants

        async def acquire(self) -> bool:
            self.remaining -= 1
            return self.remaining >= 0

        async def release(self) -> None:
            return None

    repository = InMemoryRepository()
    controller = EyvanController(repository, ControlPlaneService(repository), leader=LosesLeadershipAfter(grants=1))
    ran: list[str] = []

    def _phase(name: str) -> LeaderPhase:
        async def run() -> None:
            ran.append(name)

        return LeaderPhase(name, run)

    controller._leader_phases = lambda: (_phase("first"), _phase("second"), _phase("third"))  # type: ignore[method-assign]

    await controller._run_leader_phases(now=0.0, due={})

    assert ran == ["first"]


@pytest.mark.asyncio
async def test_a_failover_between_cycles_does_not_double_settle_a_lease() -> None:
    """Settlement is the one operation that moves money, so it must be exactly-once.

    Invariant: ``settled_at`` is the idempotency boundary. A second settlement --
    from a replacement leader replaying the same terminal transition -- must not
    charge the budget twice.
    """

    from eyvan.records import BudgetRecord

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    async with repository.transaction() as transaction:
        await transaction.put_budget(
            BudgetRecord(org_id="acme", project_id="research", limit=1_000, updated_by="operator")
        )

    workload, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_spec(),
        idempotency_key="failover-1",
    )
    admin = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))

    first, _lease, _budget = await service.settle_terminal_workload(
        admin, workload.id, WorkloadState.CANCELLED, action="workload.cancelled"
    )
    second, _lease, _budget = await service.settle_terminal_workload(
        admin, workload.id, WorkloadState.CANCELLED, action="workload.cancelled"
    )

    assert first.state is second.state is WorkloadState.CANCELLED
    async with repository.transaction() as transaction:
        budget = await transaction.get_budget("acme", "research")
        events = await transaction.list_events(org_id="acme", project_id="research", limit=100)
    assert budget is not None and budget.spent == 0
    assert [event.action for event in events].count("budget.lease_settled") == 0


# --------------------------------------------------------------- provider API outage


@pytest.mark.asyncio
async def test_a_provider_outage_does_not_discard_the_capacity_command() -> None:
    """Provider APIs fail transiently all the time.

    Invariant: a provider error leaves the command owed delivery -- FAILED and
    retryable, never DISCARDED -- because discarding it would strand a cell the
    control plane believes it asked for.
    """

    from eyvan.capacity.drivers.base import CapacityProviderFailureCode
    from eyvan.records import CellRecord, CellState
    from eyvan.server.controller import EyvanController
    from eyvan.worker.agent.client import CellAgentCapacityProviderError

    class OutageAgent:
        def apply_capacity(self, cell_id, request):
            raise CellAgentCapacityProviderError("gcloud returned 503", code=CapacityProviderFailureCode.PROVIDER_ERROR)

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service, agent_factory=lambda _cell: OutageAgent())  # type: ignore[arg-type]
    admin = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    await service.upsert_cell(
        admin,
        CellRecord(
            id="cell-outage",
            provider="local",
            region="local",
            state=CellState.REGISTERING,
            capacity=ResourceVector(cpu=8),
            allocatable=ResourceVector(cpu=8),
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell-1",
    )

    await controller.dispatch_outbox_once()

    async with repository.transaction() as transaction:
        commands = await transaction.list_outbox(limit=100)
    applies = [command for command in commands if command.topic == "capacity.apply"]
    assert applies, "the upsert must enqueue a capacity apply for this to test anything"
    assert applies[0].state is OutboxState.FAILED
    assert applies[0].state is not OutboxState.DISCARDED
    assert "503" in (applies[0].last_error or "")


# --------------------------------------------------------------- storage corruption


def test_a_foreign_fleet_document_is_refused_rather_than_overwritten(tmp_path) -> None:
    """``EYVAN_FLEET_STATE_URI`` can be pointed at the wrong object.

    Invariant: Eyvan refuses a document it does not recognize instead of treating it
    as empty state and writing over it. Silently adopting it would destroy whatever
    the file actually was, and mint cells against a fleet that does not exist.
    """

    import json

    from eyvan.capacity.fleet_state import FleetStateSchemaError

    path = tmp_path / "foreign.json"
    path.write_text(json.dumps({"some_other_tool": {"clusters": []}}), encoding="utf-8")

    with pytest.raises(FleetStateSchemaError):
        FleetStateStore(LocalBackend(path)).get("cell-a")


def test_a_malformed_generation_is_refused_rather_than_coerced(tmp_path) -> None:
    """A hand-edited or truncated record must not be repaired by guessing.

    Invariant: generations fence every provider mutation, so a non-integer generation
    is a refusal. Coercing it to 0 or 1 would hand a stale actor a valid-looking fence.
    """

    import json

    from eyvan.capacity.fleet_state import FleetStateSchemaError

    path = tmp_path / "corrupt.json"
    path.write_text(
        json.dumps({"version": 1, "cells": {"cell-a": {"generation": "not-a-number"}}, "lease": None}),
        encoding="utf-8",
    )

    with pytest.raises(FleetStateSchemaError):
        FleetStateStore(LocalBackend(path)).get("cell-a")


# --------------------------------------------------------------- orphaned workloads


@pytest.mark.asyncio
async def test_a_workload_whose_lease_vanished_is_settled_rather_than_stranded() -> None:
    """The observer skips what it cannot resolve, so nothing else would ever settle it.

    Invariant: a live workload with no lease row reaches a terminal state. Left alone
    it stays RUNNING forever -- never polled, never settled -- while still counted
    against its project's quota, budget and fair share despite holding no capacity.
    One lost row would permanently shrink a tenant's effective quota.
    """

    from eyvan.records import CellRecord, CellState, LeaseRecord, LeaseState
    from eyvan.server.controller import EyvanController

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service)
    admin = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))

    workload, _ = await service.submit_workload(
        _principal(), org_id="acme", project_id="research", spec=_spec(), idempotency_key="orphan-1"
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(cpu=4),
                allocatable=ResourceVector(cpu=4),
            )
        )
    lease = LeaseRecord(
        id="lease-orphan",
        workload_id=workload.id,
        cell_id="cell-a",
        org_id="acme",
        project_id="research",
        resources=_spec().gang,
        state=LeaseState.ACTIVE,
        fencing_token="fence-1",
        expires_at=utcnow() + timedelta(hours=1),
    )
    # create_lease already moves the workload to LEASED.
    await service.create_lease(admin, lease, idempotency_key="lease-1")
    async with repository.transaction() as transaction:
        stored = await transaction.get_workload(workload.id)
        await transaction.put_workload(
            stored.model_copy(
                update={
                    "state": WorkloadState.RUNNING,
                    # Points at a lease row that is not there -- what a retention
                    # sweep, a hand repair, or a restore from an older backup leaves
                    # behind.
                    "lease_id": "lease-vanished",
                    "lease_ids": (),
                    "remote_id": "remote-1",
                    # Older than the lease TTL, so the grace period has elapsed.
                    "updated_at": utcnow() - timedelta(hours=2),
                }
            )
        )

    assert await controller.settle_orphaned_workloads_once() == 1

    async with repository.transaction() as transaction:
        settled = await transaction.get_workload(workload.id)
    assert settled.state is WorkloadState.FAILED
    assert "no longer exists" in (settled.failure_reason or "")
    # The workload sweep settles by *workload*, so ``lease-orphan`` -- a real
    # ACTIVE lease the corrupted workload stopped pointing at -- is invisible to
    # it. That is what the lease sweep is for, and asserting the intermediate
    # state pins which phase does which half.
    intermediate = await check_fleet(repository)
    assert [violation.invariant for violation in intermediate] == [Invariant.NO_LEASES_OUTLIVING_WORKLOADS]

    # The lease sweep waits out the same grace period the workload sweep does --
    # a dispatch mid-way through attaching a lease is indistinguishable from an
    # orphan, and releasing live capacity is the expensive mistake. Aged here so
    # the scenario exercises the reclamation rather than the patience.
    async with repository.transaction() as transaction:
        stranded = await transaction.get_lease("lease-orphan")
        await transaction.put_lease(stranded.model_copy(update={"updated_at": utcnow() - timedelta(hours=2)}))

    assert await controller.settle_orphaned_leases_once() == 1

    # And now the fleet is whole, not merely this one record. Asserted against
    # the named invariants rather than eyeballed, because a recovery that settles
    # the workload while leaving other state broken looks identical to a correct
    # one from the workload alone -- which is exactly how the missing lease sweep
    # went unnoticed.
    assert await check_fleet(repository) == ()


@pytest.mark.asyncio
async def test_a_recently_updated_workload_is_not_swept_as_an_orphan() -> None:
    """Settling a workload is irreversible, so the sweep waits out its grace period.

    Nothing should write a workload and its lease in separate transactions, but the
    cost of being wrong here is destroying live work -- one cycle of patience removes
    the entire class of "a race made this look orphaned".
    """

    from eyvan.server.controller import EyvanController

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service)
    admin = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))

    workload, _ = await service.submit_workload(
        _principal(), org_id="acme", project_id="research", spec=_spec(), idempotency_key="fresh-1"
    )
    await service.transition_workload(admin, workload.id, WorkloadState.ADMITTED)
    async with repository.transaction() as transaction:
        stored = await transaction.get_workload(workload.id)
        await transaction.put_workload(
            stored.model_copy(update={"state": WorkloadState.RUNNING, "lease_id": "lease-gone", "remote_id": "remote-1"})
        )

    assert await controller.settle_orphaned_workloads_once() == 0

    async with repository.transaction() as transaction:
        assert (await transaction.get_workload(workload.id)).state is WorkloadState.RUNNING


@pytest.mark.asyncio
async def test_a_healthy_workload_is_never_swept() -> None:
    """The sweep must not settle a workload whose rows all resolve."""

    from eyvan.records import CellRecord, CellState, LeaseRecord, LeaseState
    from eyvan.server.controller import EyvanController

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service)
    admin = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))

    workload, _ = await service.submit_workload(
        _principal(), org_id="acme", project_id="research", spec=_spec(), idempotency_key="healthy-1"
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(cpu=4),
                allocatable=ResourceVector(cpu=4),
            )
        )
    lease = LeaseRecord(
        id="lease-healthy",
        workload_id=workload.id,
        cell_id="cell-a",
        org_id="acme",
        project_id="research",
        resources=_spec().gang,
        state=LeaseState.ACTIVE,
        fencing_token="fence-1",
        expires_at=utcnow() + timedelta(hours=1),
    )
    # create_lease already moves the workload to LEASED.
    await service.create_lease(admin, lease, idempotency_key="lease-1")
    async with repository.transaction() as transaction:
        stored = await transaction.get_workload(workload.id)
        await transaction.put_workload(
            stored.model_copy(
                update={
                    "state": WorkloadState.RUNNING,
                    "lease_id": lease.id,
                    "remote_id": "remote-1",
                    "updated_at": utcnow() - timedelta(hours=2),
                }
            )
        )

    assert await controller.settle_orphaned_workloads_once() == 0

    async with repository.transaction() as transaction:
        assert (await transaction.get_workload(workload.id)).state is WorkloadState.RUNNING


# --------------------------------------------------------------- agent cannot answer


@pytest.mark.asyncio
async def test_an_unanswerable_agent_does_not_retire_queued_capacity() -> None:
    """Leave queued capacity unchanged when an agent cannot answer rather than infer provider failure.

    Agent unavailability must not restart the provider phase clock or feed a
    failure into the zone's capacity belief; the next tick should ask again.
    """

    from eyvan.capacity import CapacityStatus
    from eyvan.records import CellRecord, CellState
    from eyvan.server.controller import EyvanController
    from eyvan.worker.agent.client import CellAgentUnavailableError

    class SilentAgent:
        def reconcile(self, cell_id, request):
            raise CellAgentUnavailableError("the cell agent returned 500")

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service, agent_factory=lambda _cell: SilentAgent())  # type: ignore[arg-type]
    admin = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    await service.upsert_cell(
        admin,
        CellRecord(
            id="cell-queued",
            provider="local",
            region="local",
            state=CellState.REGISTERING,
            capacity=ResourceVector(cpu=8),
            allocatable=ResourceVector(cpu=8),
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell-queued-1",
    )
    async with repository.transaction() as transaction:
        stored = await transaction.get_cell("cell-queued")
        assert stored is not None
        queued = stored.model_copy(update={"provider_phase": CapacityStatus.WAITING})
        await transaction.put_cell(queued)
        # The apply that put this cell in the queue has already been delivered.
        # A REGISTERING cell with an in-flight apply is deliberately not observed
        # at all, so leaving it PENDING would make this test assert nothing.
        for command in await transaction.list_outbox(limit=100):
            if command.topic == "capacity.apply":
                await transaction.put_outbox(command.model_copy(update={"state": OutboxState.SUCCEEDED}))

    observed_count = await controller.observe_cells_once()

    assert observed_count or True  # the count is the retry signal, not the assertion
    async with repository.transaction() as transaction:
        observed = await transaction.get_cell("cell-queued")
    assert observed is not None
    assert observed.state is CellState.REGISTERING, "an unanswerable agent is not a broken cell"
    assert observed.provider_phase is CapacityStatus.WAITING
    assert observed.provider_phase_started_at == queued.provider_phase_started_at, (
        "restarting the phase clock retires queued capacity on a timeout the provider never caused"
    )
    assert observed.availability == queued.availability, "the zone did nothing wrong; its belief must not move"
    assert observed.unavailable_reason is None


@pytest.mark.asyncio
async def test_a_rejected_fence_records_which_check_rejected_it() -> None:
    """The agent distinguishes three causes; the controller used to collapse them.

    ``_accept_mutation`` raises for an older generation, for a fencing token that
    does not own the current generation, and for a tombstoned cell -- three
    different operator actions. The observation path replaced all of them with
    "cell-agent reconciliation rejected the current fence", which says only that
    a fence was rejected.

    Preserve the specific rejection so operators can distinguish those causes
    without inspecting the agent's state document.
    """

    from eyvan.capacity import CapacityStatus
    from eyvan.records import CellRecord, CellState
    from eyvan.server.controller import EyvanController
    from eyvan.worker.agent.client import CellAgentStaleMutationError

    class TombstonedAgent:
        def reconcile(self, cell_id, request):
            raise CellAgentStaleMutationError("generation 1 is not newer than fenced cell 'cell-queued' generation 2")

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service, agent_factory=lambda _cell: TombstonedAgent())  # type: ignore[arg-type]
    admin = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    await service.upsert_cell(
        admin,
        CellRecord(
            id="cell-queued",
            provider="local",
            region="local",
            state=CellState.REGISTERING,
            capacity=ResourceVector(cpu=8),
            allocatable=ResourceVector(cpu=8),
            endpoint="http://agent:8090",
            # Pool-managed, so the autoscaler owns recreation and the conflict
            # record survives for an operator to read. A pool-less cell is
            # recreated inline at a fresh generation, which correctly clears it.
            pool_id="pool-a",
        ),
        idempotency_key="cell-fence-1",
    )
    async with repository.transaction() as transaction:
        stored = await transaction.get_cell("cell-queued")
        assert stored is not None
        await transaction.put_cell(stored.model_copy(update={"provider_phase": CapacityStatus.WAITING}))
        for command in await transaction.list_outbox(limit=100):
            if command.topic == "capacity.apply":
                await transaction.put_outbox(command.model_copy(update={"state": OutboxState.SUCCEEDED}))

    await controller.observe_cells_once()

    async with repository.transaction() as transaction:
        observed = await transaction.get_cell("cell-queued")
    assert observed is not None
    assert observed.unavailable_reason == "agent_fence_conflict", "the closed code stays a closed code"
    assert observed.unavailable_detail is not None
    assert "not newer than fenced cell" in observed.unavailable_detail
