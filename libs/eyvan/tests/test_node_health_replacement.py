# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A failed attempt that names a dead machine replaces the machine.

Eyvan knows what a failed *attempt* is and nothing about a bad *machine*, so a
TPU host that never initializes is asked to run the job again -- and again --
until the workload's user-code budget is gone. All the while it holds a lease,
which keeps its cell undeletable, which is half of why a stuck ``eyvan down``
needs a break-glass at all.

What is under test here is the promotion and, at least as much, its limits:

* the failure is charged to the *capacity* budget, not the user's, because the
  user's code never ran;
* exactly one host is quarantined -- a gang's observation is every rank's text
  concatenated, and one dead TPU must not take its healthy peers with it;
* an ordinary failure is left completely alone;
* a zone handing out broken machines exhausts a bounded budget and is reported,
  rather than provisioning and destroying capacity forever;
* and none of it happens at all until an operator turns it on.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from test_controller import FakeAgent, GangFakeAgent, snapshot

from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.config import EyvanSettings
from eyvan.records import (
    AuditEvent,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    ResourceGang,
    ResourceVector,
    WorkloadAttemptTopology,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    WorkloadTopologyCell,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.worker.agent.models import WorkloadObservation
from eyvan.worker.runtime.node_health import (
    HOST_FAULT_AUDIT_ENTITY,
    HOST_FAULT_HOST_QUARANTINED,
    HOST_FAULT_REASON,
    HOST_FAULT_REPLACEMENT_DECLINED,
    HostFaultCode,
    HostReplacementBudget,
    parse_host_fault_marker,
)

IOMMU = "libtpu: Couldn't open iommu group /dev/vfio/3: No such device"
TPU_INIT = "RuntimeError: Failed to initialize TPU system: 4 of 8 chips did not respond"
USER_BUG = "Traceback (most recent call last): ValueError: shapes do not match"


class FailingAgent(FakeAgent):
    """A cell agent whose single-cell execution reports one failure message."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self.remote_status = ExecutionWorkloadStatus.FAILED
        self.remote_message = message


class FailingGangAgent(GangFakeAgent):
    """A gang whose aggregate observation is every rank's own text."""

    def __init__(self, ranks: dict[int, str]) -> None:
        super().__init__()
        self.ranks = ranks

    def gang_workload(self, coordinator_cell_id, workload_id, remote_id):
        self.gang_status_calls.append((coordinator_cell_id, workload_id, remote_id))
        return WorkloadObservation(
            remote_id=remote_id,
            status=ExecutionWorkloadStatus.FAILED,
            message="; ".join(f"rank {rank}=failed: {text}" for rank, text in sorted(self.ranks.items())),
        )


def _cell(cell_id: str, *, zone: str = "us-east5-a", pool_id: str | None = None, processes: int = 1) -> CellRecord:
    return CellRecord(
        id=cell_id,
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="gcp-project-a",
        region="us-east5",
        zone=zone,
        state=CellState.READY,
        capacity=ResourceVector(cpu=32, memory_gib=64, tpu=8),
        allocatable=ResourceVector(cpu=32, memory_gib=64, tpu=8),
        accelerator_type="v5p-16",
        endpoint="http://agent:8090",
        pool_id=pool_id,
        runtime_process_count=processes,
    )


async def _attempt(
    agent,
    cells: list[CellRecord],
    *,
    budget: HostReplacementBudget,
    freeze_topology: bool = True,
) -> EyvanController:
    """One running attempt holding a lease on each of ``cells``."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(lease_ttl_s=120),
        agent_factory=lambda _cell: agent,
        host_replacement_budget=budget,
    )
    lease_ids = [f"lease-{index}" for index, _cell in enumerate(cells)]
    async with repository.transaction() as transaction:
        for index, cell in enumerate(cells):
            await transaction.put_cell(cell)
            await transaction.put_lease(
                LeaseRecord(
                    id=lease_ids[index],
                    workload_id="wl-1",
                    cell_id=cell.id,
                    org_id="acme",
                    project_id="research",
                    state=LeaseState.ACTIVE,
                    resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1, tpu=8)),
                    generation=1,
                    cell_generation=cell.generation,
                    fencing_token=f"token-{index}",
                    expires_at=utcnow() + timedelta(hours=1),
                )
            )
        await transaction.put_workload(
            WorkloadRecord(
                id="wl-1",
                org_id="acme",
                project_id="research",
                state=WorkloadState.RUNNING,
                spec=WorkloadDefinition(
                    name="train",
                    execution=CommandExecution(command="python train.py"),
                    gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1, tpu=8)),
                    max_failure_retries=3,
                    max_retries=3,
                ),
                lease_id=lease_ids[0],
                lease_ids=tuple(lease_ids) if len(cells) > 1 else (),
                remote_id="remote-1",
                created_by="alice",
            )
        )
        if freeze_topology:
            await transaction.put_workload_topology(
                WorkloadAttemptTopology.freeze(
                    workload_id="wl-1",
                    attempt=1,
                    generation=1,
                    cells=[
                        WorkloadTopologyCell(
                            cell_id=cell.id,
                            generation=cell.generation,
                            runtime_process_count=cell.runtime_process_count,
                            global_rank_offset=0,
                        )
                        for cell in cells
                    ],
                )
            )
    return controller


async def _cells(controller: EyvanController) -> dict[str, CellRecord]:
    _workloads, cells, _leases, _commands = await snapshot(controller.repository)
    return {cell.id: cell for cell in cells}


async def _workload(controller: EyvanController) -> WorkloadRecord:
    workloads, _cells, _leases, _commands = await snapshot(controller.repository)
    return next(item for item in workloads if item.id == "wl-1")


async def _host_fault_events(controller: EyvanController):
    async with controller.repository.transaction() as transaction:
        return await transaction.list_events(entity_type=HOST_FAULT_AUDIT_ENTITY, limit=100)


# --------------------------------------------------------------------- promotion


@pytest.mark.asyncio
async def test_a_host_fault_quarantines_the_machine_instead_of_retrying_the_job() -> None:
    """The point of the workstream: a dead TPU is not a failed attempt."""

    controller = await _attempt(
        FailingAgent(TPU_INIT), [_cell("cell-a")], budget=HostReplacementBudget(max_replacements=2)
    )

    await controller.observe_workloads_once()

    cell = (await _cells(controller))["cell-a"]
    assert cell.state is CellState.UNAVAILABLE
    assert cell.provider_phase is CapacityStatus.NEEDS_RECREATE
    assert cell.unavailable_reason == HOST_FAULT_REASON
    proof = parse_host_fault_marker(cell.unavailable_reason, cell.unavailable_detail)
    assert proof is not None
    assert proof.code is HostFaultCode.TPU_SYSTEM_INIT_FAILED
    assert proof.evidence == TPU_INIT
    assert proof.workload_id == "wl-1"
    workload = await _workload(controller)
    assert workload.failure_count == 0, "the user's code never ran; its retry budget must not pay for the machine"
    assert workload.state is WorkloadState.RUNNING


@pytest.mark.asyncio
async def test_the_workload_is_requeued_against_the_capacity_budget_and_its_lease_returns() -> None:
    """The lease is what was blocking ``eyvan down``; the quarantine frees it."""

    controller = await _attempt(FailingAgent(IOMMU), [_cell("cell-a")], budget=HostReplacementBudget(max_replacements=2))

    await controller.observe_workloads_once()
    await controller.observe_workloads_once()

    workload = await _workload(controller)
    assert workload.state is WorkloadState.PENDING
    assert workload.capacity_loss_count == 1
    assert workload.failure_count == 0
    _workloads, _inventory, leases, _commands = await snapshot(controller.repository)
    assert all(lease.state in {LeaseState.RELEASED, LeaseState.EXPIRED} for lease in leases)


@pytest.mark.asyncio
async def test_the_quarantined_host_is_replaced_rather_than_returned_to_service() -> None:
    """NEEDS_RECREATE is what the existing recovery loop acts on: new machine, same cell."""

    controller = await _attempt(FailingAgent(IOMMU), [_cell("cell-a")], budget=HostReplacementBudget(max_replacements=2))
    await controller.observe_workloads_once()
    await controller.observe_workloads_once()

    assert await controller.reconcile_unmanaged_capacity_recoveries_once() == 1

    cell = (await _cells(controller))["cell-a"]
    assert cell.state is CellState.REGISTERING
    assert cell.generation == 2, "a replacement is a new capacity generation, not a retry of the old one"
    _workloads, _inventory, _leases, commands = await snapshot(controller.repository)
    assert [command.topic for command in commands if command.generation == 2] == ["capacity.apply"]


@pytest.mark.asyncio
async def test_an_ordinary_failure_still_retries_the_workload_and_keeps_its_host() -> None:
    """Most failures are the user's. Nothing about them may change."""

    controller = await _attempt(
        FailingAgent(USER_BUG), [_cell("cell-a")], budget=HostReplacementBudget(max_replacements=2)
    )

    await controller.observe_workloads_once()

    assert (await _cells(controller))["cell-a"].state is CellState.READY
    workload = await _workload(controller)
    assert workload.failure_count == 1
    assert workload.capacity_loss_count == 0
    assert await _host_fault_events(controller) == []


@pytest.mark.asyncio
async def test_replacement_is_off_until_an_operator_turns_it_on() -> None:
    """It changes what happens to a failed attempt, so it is opted into."""

    controller = await _attempt(FailingAgent(TPU_INIT), [_cell("cell-a")], budget=HostReplacementBudget())

    await controller.observe_workloads_once()

    assert (await _cells(controller))["cell-a"].state is CellState.READY
    assert (await _workload(controller)).failure_count == 1


# --------------------------------------------------------------------- attribution


@pytest.mark.asyncio
async def test_one_ranks_host_fault_quarantines_exactly_that_cell() -> None:
    """A gang reports every rank's text at once. Only one machine is broken."""

    agent = FailingGangAgent({0: USER_BUG, 1: IOMMU})
    controller = await _attempt(
        agent,
        [_cell("cell-a"), _cell("cell-b")],
        budget=HostReplacementBudget(max_replacements=4),
    )

    await controller.observe_workloads_once()

    cells = await _cells(controller)
    assert cells["cell-b"].state is CellState.UNAVAILABLE
    assert cells["cell-b"].unavailable_reason == HOST_FAULT_REASON
    assert cells["cell-a"].state is CellState.READY, "rank 1's dead TPU took a healthy peer with it"
    assert cells["cell-a"].unavailable_reason is None
    quarantined = [event for event in await _host_fault_events(controller)]
    assert [event.entity_id for event in quarantined] == ["cell-b"]


@pytest.mark.asyncio
async def test_a_gang_fault_with_no_frozen_rank_space_quarantines_nothing() -> None:
    """No proven rank map means the owning host is unknown, and unknown is not all of them."""

    agent = FailingGangAgent({0: USER_BUG, 1: IOMMU})
    controller = await _attempt(
        agent,
        [_cell("cell-a"), _cell("cell-b")],
        budget=HostReplacementBudget(max_replacements=4),
        freeze_topology=False,
    )

    await controller.observe_workloads_once()

    cells = await _cells(controller)
    assert [cell.state for cell in cells.values()] == [CellState.READY, CellState.READY]
    assert await _host_fault_events(controller) == []


@pytest.mark.asyncio
async def test_two_broken_hosts_in_one_gang_are_both_quarantined() -> None:
    """Attribution is per rank, so two dead machines are two dead machines."""

    agent = FailingGangAgent({0: TPU_INIT, 1: IOMMU})
    controller = await _attempt(
        agent,
        [_cell("cell-a"), _cell("cell-b")],
        budget=HostReplacementBudget(max_replacements=4),
    )

    await controller.observe_workloads_once()

    cells = await _cells(controller)
    assert {cell_id for cell_id, cell in cells.items() if cell.state is CellState.UNAVAILABLE} == {
        "cell-a",
        "cell-b",
    }


# --------------------------------------------------------------------- the budget


@pytest.mark.asyncio
async def test_a_budget_stops_a_zone_of_bad_hosts_from_becoming_a_replacement_loop() -> None:
    """A whole bad zone must not turn into unbounded provisioning at full price."""

    agent = FailingGangAgent({0: TPU_INIT, 1: IOMMU})
    controller = await _attempt(
        agent,
        [_cell("cell-a", pool_id="research/v5p-16"), _cell("cell-b", pool_id="research/v5p-16")],
        budget=HostReplacementBudget(max_replacements=1),
    )

    await controller.observe_workloads_once()

    cells = await _cells(controller)
    quarantined = [cell_id for cell_id, cell in cells.items() if cell.state is CellState.UNAVAILABLE]
    assert quarantined == ["cell-a"], "the budget is one host, and one host was replaced"
    assert cells["cell-b"].state is CellState.READY
    assert cells["cell-b"].unavailable_reason is None


@pytest.mark.asyncio
async def test_exhaustion_is_reported_with_the_evidence_and_the_bound_it_hit() -> None:
    """Silently declining to replace a machine Eyvan knows is dead is the worse failure."""

    agent = FailingGangAgent({0: TPU_INIT, 1: IOMMU})
    controller = await _attempt(
        agent,
        [_cell("cell-a", pool_id="research/v5p-16"), _cell("cell-b", pool_id="research/v5p-16")],
        budget=HostReplacementBudget(max_replacements=1, window_s=1800),
    )

    await controller.observe_workloads_once()

    events = await _host_fault_events(controller)
    assert [event.action for event in events] == [
        HOST_FAULT_HOST_QUARANTINED,
        HOST_FAULT_REPLACEMENT_DECLINED,
    ]
    declined = events[1]
    assert declined.entity_id == "cell-b"
    assert declined.payload["declined"] == "exhausted"
    assert declined.payload["scope"] == "pool:research/v5p-16"
    assert declined.payload["spent"] == 1
    assert declined.payload["budget"] == 1
    assert declined.payload["code"] == HostFaultCode.TPU_IOMMU_GROUP_UNAVAILABLE.value
    assert declined.payload["evidence"] == IOMMU


@pytest.mark.asyncio
async def test_a_budget_is_counted_per_scope_not_across_the_fleet() -> None:
    """One pool's bad rack must not stop another pool from replacing its own host."""

    agent = FailingGangAgent({0: TPU_INIT, 1: IOMMU})
    controller = await _attempt(
        agent,
        [_cell("cell-a", pool_id="research/v5p-16"), _cell("cell-b", pool_id="production/v5p-16")],
        budget=HostReplacementBudget(max_replacements=1),
    )

    await controller.observe_workloads_once()

    cells = await _cells(controller)
    assert cells["cell-a"].state is CellState.UNAVAILABLE
    assert cells["cell-b"].state is CellState.UNAVAILABLE


@pytest.mark.asyncio
async def test_replacing_the_same_host_twice_for_one_execution_charges_the_budget_once() -> None:
    """The observation loop is at-least-once; the ledger it charges must not be."""

    controller = await _attempt(
        FailingAgent(TPU_INIT), [_cell("cell-a")], budget=HostReplacementBudget(max_replacements=2)
    )
    await controller.observe_workloads_once()
    async with controller.repository.transaction() as transaction:
        cell = await transaction.get_cell("cell-a")
        assert cell is not None
        await transaction.put_cell(cell.model_copy(update={"state": CellState.READY, "provider_phase": None}))

    await controller.observe_workloads_once()

    events = await _host_fault_events(controller)
    assert [event.action for event in events] == [HOST_FAULT_HOST_QUARANTINED]


async def _charge(controller: EyvanController, *, scope: str, age_s: float) -> None:
    """Record one past host replacement against ``scope``, as the controller would."""

    async with controller.repository.transaction() as transaction:
        await transaction.append_event(
            AuditEvent(
                id=f"evt-{scope}-{age_s}",
                actor="system:controller",
                entity_type=HOST_FAULT_AUDIT_ENTITY,
                entity_id="cell-earlier",
                action=HOST_FAULT_HOST_QUARANTINED,
                timestamp=utcnow() - timedelta(seconds=age_s),
                payload={"scope": scope},
            )
        )


@pytest.mark.asyncio
async def test_a_replacement_charged_before_this_controller_started_still_counts() -> None:
    """The ledger outlives the process; a restart must not hand back a fresh budget."""

    controller = await _attempt(
        FailingAgent(TPU_INIT),
        [_cell("cell-a", pool_id="research/v5p-16")],
        budget=HostReplacementBudget(max_replacements=1, window_s=3600),
    )
    await _charge(controller, scope="pool:research/v5p-16", age_s=600)

    await controller.observe_workloads_once()

    assert (await _cells(controller))["cell-a"].state is CellState.READY
    assert [event.action for event in await _host_fault_events(controller)][-1] == HOST_FAULT_REPLACEMENT_DECLINED


@pytest.mark.asyncio
async def test_a_replacement_charged_outside_the_window_no_longer_counts() -> None:
    """The bound is a rate, not a lifetime cap on a pool.

    Same ledger entry as the test above and the same budget of one; only its age
    differs, and that is what decides whether this host is replaced.
    """

    controller = await _attempt(
        FailingAgent(TPU_INIT),
        [_cell("cell-a", pool_id="research/v5p-16")],
        budget=HostReplacementBudget(max_replacements=1, window_s=3600),
    )
    await _charge(controller, scope="pool:research/v5p-16", age_s=7200)

    await controller.observe_workloads_once()

    assert (await _cells(controller))["cell-a"].state is CellState.UNAVAILABLE


@pytest.mark.asyncio
async def test_a_replacement_charged_to_another_scope_does_not_count() -> None:
    """Otherwise one pool's bad rack silently freezes every other pool's node health."""

    controller = await _attempt(
        FailingAgent(TPU_INIT),
        [_cell("cell-a", pool_id="research/v5p-16")],
        budget=HostReplacementBudget(max_replacements=1, window_s=3600),
    )
    await _charge(controller, scope="pool:production/v5p-16", age_s=60)

    await controller.observe_workloads_once()

    assert (await _cells(controller))["cell-a"].state is CellState.UNAVAILABLE


# --------------------------------------------------------------------- the belief


@pytest.mark.asyncio
async def test_the_fault_is_recorded_against_the_zone_that_produced_it() -> None:
    """Routing learns from this; charging the wrong zone steers work away from a good one."""

    agent = FailingGangAgent({0: USER_BUG, 1: IOMMU})
    controller = await _attempt(
        agent,
        [_cell("cell-a", zone="us-east5-a"), _cell("cell-b", zone="us-east5-c")],
        budget=HostReplacementBudget(max_replacements=4),
    )
    before = (await _cells(controller))["cell-b"].availability

    await controller.observe_workloads_once()

    cells = await _cells(controller)
    faulted = cells["cell-b"].availability
    assert faulted.failure_evidence > before.failure_evidence
    assert faulted.consecutive_failures == 1
    assert faulted.cooldown_until is not None
    assert cells["cell-a"].availability.cooldown_until is None, "a healthy zone was charged for another zone's fault"
    assert cells["cell-a"].availability.consecutive_failures == 0


@pytest.mark.asyncio
async def test_a_host_fault_backs_off_harder_than_an_ordinary_provider_error() -> None:
    """Each retry costs a full provision *and* a destroy, so the first wait is long."""

    controller = await _attempt(
        FailingAgent(TPU_INIT), [_cell("cell-a")], budget=HostReplacementBudget(max_replacements=2)
    )

    await controller.observe_workloads_once()

    cell = (await _cells(controller))["cell-a"]
    assert cell.availability.cooldown_until is not None
    wait_s = (cell.availability.cooldown_until - utcnow()).total_seconds()
    assert 240 <= wait_s <= 300


@pytest.mark.asyncio
async def test_a_pool_record_is_not_required_for_any_of_this() -> None:
    """Manual and loopback fleets have no pool, and still have machines that break."""

    pool = CapacityPoolRecord(
        id="research/v5p-16",
        fleet_name="research",
        node_type="v5p-16",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="gcp-project-a",
        region="us-east5",
        zone="us-east5-a",
        capacity=ResourceVector(cpu=32, tpu=8),
        allocatable=ResourceVector(cpu=32, tpu=8),
        accelerator_type="v5p-16",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
    )
    controller = await _attempt(
        FailingAgent(TPU_INIT), [_cell("cell-a")], budget=HostReplacementBudget(max_replacements=1)
    )
    async with controller.repository.transaction() as transaction:
        await transaction.put_capacity_pool(pool)

    await controller.observe_workloads_once()

    events = await _host_fault_events(controller)
    assert [event.payload["scope"] for event in events] == ["unpooled:gcp/us-east5-a"]


@pytest.mark.asyncio
async def test_a_spend_that_could_not_be_established_refuses_rather_than_assuming_room() -> None:
    """An unknown budget is not an available one, and a refusal only costs capacity.

    The scan is bounded so one decision cannot read an unbounded ledger. A fleet
    that fills it is already in the outage this bound exists for, and guessing
    the cheaper answer there is how a bound becomes decorative.
    """

    controller = await _attempt(
        FailingAgent(TPU_INIT),
        [_cell("cell-a", pool_id="research/v5p-16")],
        budget=HostReplacementBudget(max_replacements=1000, window_s=3600),
    )
    async with controller.repository.transaction() as transaction:
        for index in range(600):
            await transaction.append_event(
                AuditEvent(
                    id=f"evt-noise-{index}",
                    actor="system:controller",
                    entity_type=HOST_FAULT_AUDIT_ENTITY,
                    entity_id="cell-elsewhere",
                    action=HOST_FAULT_REPLACEMENT_DECLINED,
                    payload={"scope": "pool:production/v5p-16"},
                )
            )

    await controller.observe_workloads_once()

    assert (await _cells(controller))["cell-a"].state is CellState.READY
    async with controller.repository.transaction() as transaction:
        recorded = await transaction.list_events(entity_type=HOST_FAULT_AUDIT_ENTITY, entity_id="cell-a", limit=10)
    assert [event.action for event in recorded] == [HOST_FAULT_REPLACEMENT_DECLINED]
    assert recorded[0].payload["declined"] == "unknown"


@pytest.mark.asyncio
async def test_a_quarantined_host_is_not_observed_back_into_service() -> None:
    """The provider never knew. A queued resource stays ACTIVE while its chips do not."""

    agent = FailingAgent(TPU_INIT)
    controller = await _attempt(agent, [_cell("cell-a")], budget=HostReplacementBudget(max_replacements=1))
    await controller.observe_workloads_once()
    assert (await _cells(controller))["cell-a"].state is CellState.UNAVAILABLE
    async with controller.repository.transaction() as transaction:
        quarantined = await transaction.get_cell("cell-a")
        assert quarantined is not None
        # Skip the belief cooldown rather than waiting it out: the quarantine is
        # what must hold here, not the backoff that also happens to.
        await transaction.put_cell(
            quarantined.model_copy(
                update={"availability": quarantined.availability.model_copy(update={"cooldown_until": None})}
            )
        )
    agent.capacity_state = CapacityStatus.HEALTHY

    assert await controller.observe_cells_once() == 0

    cell = (await _cells(controller))["cell-a"]
    assert cell.state is CellState.UNAVAILABLE
    assert cell.unavailable_reason == HOST_FAULT_REASON


@pytest.mark.asyncio
async def test_recreating_the_cell_returns_it_to_ordinary_observation() -> None:
    """The quarantine is on a machine, not on a name; a new generation is a new machine."""

    agent = FailingAgent(TPU_INIT)
    controller = await _attempt(agent, [_cell("cell-a")], budget=HostReplacementBudget(max_replacements=1))
    await controller.observe_workloads_once()
    await controller.observe_workloads_once()
    await controller.reconcile_unmanaged_capacity_recoveries_once()
    agent.capacity_state = CapacityStatus.HEALTHY
    await controller.dispatch_outbox_once()

    assert await controller.observe_cells_once() == 1

    cell = (await _cells(controller))["cell-a"]
    assert cell.state is CellState.READY
    assert cell.unavailable_reason is None
    assert cell.generation == 2
