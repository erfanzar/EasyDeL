# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Two native gangs are never handed one rendezvous endpoint.

A gang's ranks meet at ``<coordinator host>:<port>``. The coordinator is the
lexicographically first of the gang's leased cells, and both ports are fixed
constants -- 12345 for the JAX coordinator, 12346 for GPU collectives -- because
a deployment's firewall has to be able to open them once and keep working.
Choosing arbitrary ports instead could route coordination to an address the
deployment's firewall does not permit.

Fixed ports mean two gangs whose coordinators land on one machine are handed the
*same* socket, and a rank of one gang can register with the other gang's
coordinator. That was masked rather than closed: ``server/service/scheduling.py``
marks every ``native-local`` and ``eyvan-native`` cell ``exclusive_workloads``,
and ``domain/scheduler.py`` will not place a second workload on a busy exclusive
cell -- so the safety was a consequence of a flag rather than a decision, and it
disappeared silently the moment the flag did.

What is asserted here: dispatch refuses the second gang and says which flag it
expected to have been protected by; a gang with no neighbour is dispatched with
exactly the bytes it has always been dispatched with; and with exclusivity on,
the scheduler still never produces the placement in the first place.
"""

from __future__ import annotations

import typing as tp
from datetime import timedelta

from eyvan.config import EyvanSettings
from eyvan.domain import ResourceVector as DomainResourceVector
from eyvan.domain import Scheduler
from eyvan.domain.models import ResourceGang as DomainResourceGang
from eyvan.domain.models import WorkloadIdentity, WorkloadSpec
from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    OutboxRecord,
    OutboxState,
    ResourceGang,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService, to_core_cell
from eyvan.worker.distributed_environment import (
    DEFAULT_GPU_RENDEZVOUS_PORT,
    DEFAULT_NATIVE_COORDINATOR_PORT,
    plan_gpu_devices,
)

# --------------------------------------------------------------------------- harness


class _RecordingAgent:
    """Every submission the control plane actually put on the wire."""

    def __init__(self) -> None:
        self.gang_submissions: list[tuple[str, tp.Mapping[str, object]]] = []

    def submit_gang(self, coordinator_cell_id, request):
        self.gang_submissions.append((coordinator_cell_id, dict(request.spec)))
        from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
        from eyvan.worker.agent.models import WorkloadObservation

        return WorkloadObservation(
            remote_id=request.spec["submission_id"],
            status=ExecutionWorkloadStatus.SUBMITTED,
        )


def _native_cell(cell_id: str, *, hosts: tuple[str, ...] = ()) -> CellRecord:
    """One ready ``eyvan-native`` cell, optionally naming its worker hosts.

    ``provider_options["hosts"]`` is what the manual driver copies into the
    observation's ``native_worker_endpoints``, which is what the agent forms the
    rendezvous address from -- so it is the control plane's only view of which
    machine a cell's ranks land on.
    """

    return CellRecord(
        id=cell_id,
        provider="manual" if hosts else "local",
        capacity_backend="manual" if hosts else None,
        region="dc1" if hosts else "local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=8, memory_gib=32),
        allocatable=ResourceVector(cpu=8, memory_gib=32),
        executor="eyvan-native",
        endpoint="http://agent:8090",
        runtime_process_count=len(hosts) or 1,
        provider_options={"hosts": list(hosts)} if hosts else {},
    )


def _spec(cells: int) -> WorkloadDefinition:
    return WorkloadDefinition(
        name="job",
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
        required_executor="eyvan-native",
        cell_group_count=cells,
    )


async def _put_gang(
    repository: InMemoryRepository,
    workload_id: str,
    cell_ids: tp.Sequence[str],
    *,
    state: WorkloadState,
    lease_state: LeaseState,
) -> None:
    """Store one multi-cell native workload and its per-cell leases."""

    spec = _spec(len(cell_ids))
    lease_ids = [f"{workload_id}-lease-{index}" for index, _ in enumerate(cell_ids)]
    workload = WorkloadRecord(
        id=workload_id,
        org_id="acme",
        project_id="research",
        spec=spec,
        state=state,
        lease_id=lease_ids[0],
        lease_ids=lease_ids,
        created_by="alice",
    )
    async with repository.transaction() as transaction:
        for index, cell_id in enumerate(cell_ids):
            cell = await transaction.get_cell(cell_id)
            assert cell is not None
            await transaction.put_lease(
                LeaseRecord(
                    id=lease_ids[index],
                    workload_id=workload_id,
                    cell_id=cell_id,
                    cell_generation=cell.generation,
                    org_id=workload.org_id,
                    project_id=workload.project_id,
                    state=lease_state,
                    resources=spec.gang,
                    generation=workload.generation,
                    fencing_token=f"fence-{workload_id}-{cell_id}",
                    group_id=f"group-{workload_id}" if len(cell_ids) > 1 else None,
                    group_index=index,
                    expires_at=utcnow() + timedelta(minutes=5),
                )
            )
        await transaction.put_workload(workload)


async def _dispatchable(
    cells: tp.Sequence[CellRecord],
    *,
    dispatching: tp.Sequence[str],
    neighbours: tp.Sequence[tuple[str, tp.Sequence[str], WorkloadState, LeaseState]] = (),
) -> tuple[EyvanController, _RecordingAgent]:
    """A control plane holding one queued gang dispatch and any live neighbours."""

    repository = InMemoryRepository()
    agent = _RecordingAgent()
    controller = EyvanController(
        repository,
        ControlPlaneService(repository, scheduler=CoreSchedulingService()),
        settings=EyvanSettings(dev_auth_enabled=True, lease_ttl_s=60),
        agent_factory=lambda _cell: agent,
    )
    async with repository.transaction() as transaction:
        for cell in cells:
            await transaction.put_cell(cell)
    for workload_id, cell_ids, workload_state, lease_state in neighbours:
        await _put_gang(
            repository,
            workload_id,
            cell_ids,
            state=workload_state,
            lease_state=lease_state,
        )
    await _put_gang(
        repository,
        "wl-dispatch",
        dispatching,
        state=WorkloadState.LEASED,
        lease_state=LeaseState.PENDING,
    )
    async with repository.transaction() as transaction:
        await transaction.put_outbox(
            OutboxRecord(
                id="cmd-dispatch",
                topic="workload.dispatch",
                aggregate_type="workload",
                aggregate_id="wl-dispatch",
                idempotency_key="dispatch-job",
                generation=1,
                payload={
                    "workload_id": "wl-dispatch",
                    "lease_id": "wl-dispatch-lease-0",
                    "lease_ids": [f"wl-dispatch-lease-{index}" for index, _ in enumerate(dispatching)],
                    "cell_id": dispatching[0],
                },
            )
        )
    return controller, agent


async def _settled(controller: EyvanController) -> tuple[WorkloadRecord, OutboxRecord]:
    async with controller.repository.transaction() as transaction:
        workload = await transaction.get_workload("wl-dispatch")
        command = await transaction.get_outbox("cmd-dispatch")
    assert workload is not None and command is not None
    return workload, command


# ------------------------------------------------------------------ the refusal


async def test_a_second_gang_on_one_coordinator_cell_is_refused_and_names_the_flag() -> None:
    """The collision exclusivity was silently masking, now stated out loud.

    Both gangs' coordinator is ``cell-a``, so both would be told to meet at that
    machine's 12345 and 12346 -- and a rank of the new gang could register with
    the running gang's coordinator, which fails as a wrong answer rather than as
    an error. Refused before the agent is contacted at all.
    """

    controller, agent = await _dispatchable(
        [_native_cell("cell-a"), _native_cell("cell-b")],
        dispatching=("cell-a", "cell-b"),
        neighbours=[("wl-running", ("cell-a", "cell-b"), WorkloadState.RUNNING, LeaseState.ACTIVE)],
    )

    assert await controller.dispatch_outbox_once() == 1

    workload, _command = await _settled(controller)
    assert agent.gang_submissions == [], "no colliding environment may reach a worker"
    assert workload.remote_id is None
    assert workload.state is WorkloadState.FAILED
    reason = workload.failure_reason or ""
    assert "exclusive_workloads" in reason, "the operator must be told which flag was supposed to prevent this"
    assert "wl-running" in reason and "cell-a" in reason
    assert str(DEFAULT_NATIVE_COORDINATOR_PORT) in reason and str(DEFAULT_GPU_RENDEZVOUS_PORT) in reason


async def test_two_cells_that_name_one_machine_are_refused_even_though_the_cells_differ() -> None:
    """Per-cell exclusivity cannot see this one, and it is the same collision.

    ``cell-a`` and ``cell-c`` are distinct cells whose first worker endpoint is
    the same host, so their gangs' rendezvous addresses are byte-identical while
    every ``exclusive_workloads`` check passes.
    """

    controller, agent = await _dispatchable(
        [
            _native_cell("cell-a", hosts=("http://10.0.0.10:8100",)),
            _native_cell("cell-b", hosts=("http://10.0.0.20:8100",)),
            _native_cell("cell-c", hosts=("http://10.0.0.10:8100",)),
            _native_cell("cell-d", hosts=("http://10.0.0.40:8100",)),
        ],
        dispatching=("cell-a", "cell-b"),
        neighbours=[("wl-running", ("cell-c", "cell-d"), WorkloadState.RUNNING, LeaseState.ACTIVE)],
    )

    assert await controller.dispatch_outbox_once() == 1

    workload, _command = await _settled(controller)
    assert agent.gang_submissions == []
    assert workload.state is WorkloadState.FAILED
    assert "10.0.0.10" in (workload.failure_reason or "")
    assert "exclusive_workloads" in (workload.failure_reason or "")


async def test_the_refusal_is_terminal_rather_than_a_command_that_retries_forever() -> None:
    """A retrying command holds the lease and keeps its cell undeletable.

    The outbox backs off forever and has no dead letter, so the refusal is
    settled as a workload failure with the reason attached: the capacity comes
    back, and the operator gets a sentence naming what to change.
    """

    controller, _agent = await _dispatchable(
        [_native_cell("cell-a"), _native_cell("cell-b")],
        dispatching=("cell-a", "cell-b"),
        neighbours=[("wl-running", ("cell-a", "cell-b"), WorkloadState.RUNNING, LeaseState.ACTIVE)],
    )

    assert await controller.dispatch_outbox_once() == 1

    _workload, command = await _settled(controller)
    assert command.state is OutboxState.SUCCEEDED
    assert await controller.dispatch_outbox_once() == 0, "nothing is left to retry"
    async with controller.repository.transaction() as transaction:
        leases = [lease for lease in await transaction.list_leases() if lease.workload_id == "wl-dispatch"]
    assert leases and all(lease.state is LeaseState.RELEASED for lease in leases), "the capacity is given back"


# ------------------------------------------------------ what is *not* a conflict


async def test_a_single_cell_workload_beside_a_gang_is_not_a_conflict() -> None:
    """A lone workload forms no rendezvous, so there is nothing to be met at.

    Only ``NativeGangExecutor`` writes ``JAX_COORDINATOR_ADDRESS`` and
    ``MASTER_ADDR`` for a cell group. Refusing on a neighbour that never binds
    those ports would fail workloads for a collision that cannot occur.
    """

    controller, agent = await _dispatchable(
        [_native_cell("cell-a"), _native_cell("cell-b")],
        dispatching=("cell-a", "cell-b"),
        neighbours=[("wl-single", ("cell-a",), WorkloadState.RUNNING, LeaseState.ACTIVE)],
    )

    assert await controller.dispatch_outbox_once() == 1

    workload, _command = await _settled(controller)
    assert workload.state is WorkloadState.DISPATCHING
    assert [cell_id for cell_id, _spec in agent.gang_submissions] == ["cell-a"]


async def test_a_terminal_gang_whose_lease_has_not_settled_yet_is_not_a_conflict() -> None:
    """Its ranks are gone; the endpoint is free.

    Lease release trails the workload's terminal transition by a settlement pass.
    Refusing during that window would permanently fail a workload over a few
    seconds of bookkeeping.
    """

    controller, agent = await _dispatchable(
        [_native_cell("cell-a"), _native_cell("cell-b")],
        dispatching=("cell-a", "cell-b"),
        neighbours=[("wl-finished", ("cell-a", "cell-b"), WorkloadState.SUCCEEDED, LeaseState.ACTIVE)],
    )

    assert await controller.dispatch_outbox_once() == 1

    workload, _command = await _settled(controller)
    assert workload.state is WorkloadState.DISPATCHING
    assert len(agent.gang_submissions) == 1


async def test_a_gang_redispatched_after_a_restart_does_not_collide_with_itself() -> None:
    """Its own live leases name its own endpoint; that is not a second gang."""

    controller, agent = await _dispatchable(
        [_native_cell("cell-a"), _native_cell("cell-b")],
        dispatching=("cell-a", "cell-b"),
    )

    assert await controller.dispatch_outbox_once() == 1

    workload, _command = await _settled(controller)
    assert workload.state is WorkloadState.DISPATCHING
    assert len(agent.gang_submissions) == 1


# ---------------------------------------------------- a lone gang is unchanged


async def test_a_lone_gang_is_dispatched_exactly_as_before() -> None:
    """The dispatch a fleet makes today must not move at all."""

    controller, agent = await _dispatchable(
        [_native_cell("cell-a"), _native_cell("cell-b")],
        dispatching=("cell-a", "cell-b"),
    )

    assert await controller.dispatch_outbox_once() == 1

    workload, command = await _settled(controller)
    assert workload.state is WorkloadState.DISPATCHING
    assert command.state is OutboxState.SUCCEEDED
    coordinator_cell_id, spec = agent.gang_submissions[0]
    assert coordinator_cell_id == "cell-a"
    assert spec["submission_id"] == workload.remote_id
    assert spec["entrypoint"]


def test_a_lone_gangs_rendezvous_bytes_are_the_ones_that_ship() -> None:
    """The exact values a rank reads, pinned against a constant that moved file.

    The JAX coordinator port now has one definition, shared with the dispatcher
    that refuses to hand it out twice. Same number, same bytes -- asserted rather
    than assumed, because "identical" is the whole claim.
    """

    import httpx

    from eyvan.capacity.drivers.base import MutationContext
    from eyvan.worker.runtime.native_gang import NativeGangExecutor, NativeGangMember

    submitted: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.content)
        submitted.append(dict(payload["spec"]["environment"]))
        return httpx.Response(200, json={"remote_id": "gang-job"})

    endpoints = ("http://10.12.0.10:8100", "http://10.12.0.20:8100")
    clients = [httpx.Client(transport=httpx.MockTransport(handler), base_url=endpoint) for endpoint in endpoints]
    members = tuple(
        NativeGangMember(
            cell_id=cell_id,
            endpoints=(endpoint,),
            secret=f"secret-{cell_id}",
            context=MutationContext(idempotency_key=f"submit-{cell_id}", generation=7, fencing_token=f"fence-{cell_id}"),
            gpu_devices_per_host=8,
            gpu_model="nvidia-h100-80gb",
        )
        for cell_id, endpoint in zip(("cell-a", "cell-b"), endpoints, strict=True)
    )
    try:
        executor = NativeGangExecutor(members, clients=clients)
        assert executor.submit({"submission_id": "gang-job", "entrypoint": "python train.py"}) == "gang-job"
    finally:
        for client in clients:
            client.close()

    assert len(submitted) == 2
    for environment in submitted:
        assert environment["JAX_COORDINATOR_ADDRESS"] == "10.12.0.10:12345"
        assert environment["JAX_COORDINATOR_PORT"] == "12345"
        assert environment["MASTER_ADDR"] == "10.12.0.10"
        assert environment["MASTER_PORT"] == "12346"


def test_the_two_rendezvous_ports_still_differ_and_are_the_ones_a_firewall_opens() -> None:
    """One definition each, and they are not the same socket.

    A workload may initialize JAX *and* a torch process group; two services
    asking for one port is a failure that reads as a network fault.
    """

    assert DEFAULT_NATIVE_COORDINATOR_PORT == 12345
    assert DEFAULT_GPU_RENDEZVOUS_PORT == 12346
    assert plan_gpu_devices([1, 1]).rank_environment(0, rendezvous_host="10.0.0.1")["MASTER_PORT"] == "12346"


# ------------------------------------------------- exclusivity on is unchanged


def test_with_exclusivity_on_the_scheduler_never_produces_the_colliding_placement() -> None:
    """The mask still masks; the refusal is the floor under it, not a replacement.

    ``to_core_cell`` marks every native cell ``exclusive_workloads``, and the
    scheduler will not place a second workload on a busy one -- so a fleet with
    the shipped defaults never reaches the dispatcher's refusal at all.
    """

    record = _native_cell("cell-a")
    cell = to_core_cell(record)
    assert cell.exclusive_workloads is True

    identity = WorkloadIdentity("acme", "research", "alice")
    workloads = tuple(
        WorkloadSpec(
            workload_id,
            identity,
            (DomainResourceGang("workers", 1, DomainResourceVector(cpu=1, memory_gib=1)),),
            runtime="eyvan-native",
        )
        for workload_id in ("first", "second")
    )

    result = Scheduler().schedule(workloads, (cell,))

    assert [placement.workload_id for placement in result.placements] == ["first"]
    assert result.infeasibilities[0].workload_id == "second"
    assert "exactly one active workload" in result.infeasibilities[0].cells[0].message


def test_a_native_cell_carries_the_flag_the_refusal_message_names() -> None:
    """Pinned because the refusal's sentence names this flag by name.

    If ``to_core_cell`` ever stops setting it -- for a native cell with declared
    hosts as well as one without -- the message an operator reads would point at
    a mechanism that is no longer there.
    """

    assert to_core_cell(_native_cell("cell-a")).exclusive_workloads is True
    assert to_core_cell(_native_cell("cell-b", hosts=("http://10.0.0.20:8100",))).exclusive_workloads is True
