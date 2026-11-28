# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bounded fan-out: a ceiling that refuses, and waves that defer.

A ``count`` field is eleven characters and an unbounded number of accelerator
jobs. Two separate properties are asserted here, because they are two different
promises:

**The ceiling refuses.** Above it, the submitter is told at submit, holding the
request, with nothing created. Not at dispatch, where the only remaining move is
a retry loop behind a held lease.

**The waves defer, and say so.** Under the ceiling, every child is created and
durably owed execution, but only a bounded number compete at once. A silently
truncated fan-out would be worse than a refusal, so what is asserted is that the
children all exist, that the ledger says how many are waiting, and that the
waiting ones are invisible to the scheduler until an earlier one finishes.
"""

from __future__ import annotations

import json

import pytest

from eyvan.config import EyvanSettings
from eyvan.records import (
    CapacityClass,
    CellRecord,
    CellState,
    CommandExecution,
    ExpansionAdmissionState,
    ExpansionMode,
    Principal,
    QuotaRecord,
    ResourceGang,
    ResourceLimits,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadState,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.batch import BATCH_INDEX_ENV
from eyvan.server.service.expansion import (
    ExpansionLimits,
    ExpansionRefused,
    plan_expansion,
    quota_headroom_children,
    wave_size_for,
)
from eyvan.server.service.scheduling import CoreSchedulingService


def _principal(org_id: str = "acme", project_id: str = "research") -> Principal:
    return Principal(
        subject=f"{org_id}-operator",
        grants=(
            RoleGrant(role=Role.PROJECT_OPERATOR, org_id=org_id, project_id=project_id),
            RoleGrant(role=Role.FLEET_ADMIN),
        ),
    )


def _template(*, name: str = "shard", cpu: float = 1.0) -> WorkloadDefinition:
    return WorkloadDefinition(
        name=name,
        execution=CommandExecution(command="python eval.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=cpu, memory_gib=1)),
        capacity_class=CapacityClass.ON_DEMAND,
    )


def _cell(cell_id: str, *, cpu: float = 1.0) -> CellRecord:
    return CellRecord(
        id=cell_id,
        provider="manual",
        capacity_backend="manual",
        region="us-east5",
        state=CellState.READY,
        capacity=ResourceLimits(cpu=cpu, memory_gib=8.0),
        allocatable=ResourceLimits(cpu=cpu, memory_gib=8.0),
        unit_prices={"cpu": 0.1},
        capacity_class=CapacityClass.ON_DEMAND,
    )


async def _service(*, ceiling: int = 1000, wave: int = 0, scheduler: bool = False) -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    return ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService() if scheduler else None,
        expansion_limits=ExpansionLimits(ceiling=ceiling, wave_size=wave),
    )


def _controller(service: ControlPlaneService, **settings) -> EyvanController:
    return EyvanController(service.repository, service, settings=EyvanSettings(**settings))


async def _finish(service: ControlPlaneService, *workload_ids: str) -> None:
    """Force children terminal, so their place in the wave is free.

    Written directly rather than run through a lease: what the wave phase reads
    is that a child is no longer occupying, and staging a full lifecycle would
    make this depend on placement, which has its own tests.
    """

    async with service.repository.transaction() as tx:
        for workload_id in workload_ids:
            stored = await tx.get_workload(workload_id)
            assert stored is not None
            await tx.put_workload(stored.model_copy(update={"state": WorkloadState.SUCCEEDED}))


async def _wave_admissions(service: ControlPlaneService) -> list[str]:
    """Return the workload id of every durable wave-admission event, in order."""

    async with service.repository.transaction() as tx:
        events = await tx.list_events(limit=1000)
    return [event.entity_id for event in events if event.action == "workload.wave_admitted"]


# --------------------------------------------------------------------------- the ceiling


def test_the_ceiling_refuses_with_the_numbers_the_submitter_needs() -> None:
    """ "Too big" is not actionable; "you asked for 5000, you may have 50" is."""

    with pytest.raises(ExpansionRefused) as refusal:
        plan_expansion(5000, limits=ExpansionLimits(ceiling=50), subject="batch")

    message = str(refusal.value)
    assert "5000" in message and "50" in message
    assert "EYVAN_MAX_EXPANSION_CHILDREN" in message


def test_a_request_may_lower_the_ceiling_and_never_raise_it() -> None:
    """How many accelerators one call may claim is a deployment decision."""

    limits = ExpansionLimits(ceiling=100)

    assert limits.effective_ceiling(10) == 10
    assert limits.effective_ceiling(10_000) == 100
    assert limits.effective_ceiling(None) == 100


async def test_an_expansion_above_the_ceiling_is_refused_at_submit_creating_nothing() -> None:
    """Refused where the submitter is, not at dispatch behind a held lease."""

    service = await _service(ceiling=4)

    with pytest.raises(ExpansionRefused, match="exceeds the 4-child admission ceiling"):
        await service.submit_batch(
            _principal(),
            org_id="acme",
            project_id="research",
            template=_template(),
            count=5,
            idempotency_key="batch-1",
        )

    async with service.repository.transaction() as tx:
        assert await tx.list_workloads() == []
        assert await tx.list_expansion_admissions() == []


async def test_the_ceiling_bounds_a_dag_and_a_group_by_the_same_number() -> None:
    """One deployment ceiling, not three shapes each with their own."""

    from eyvan.server.service.dag import DagNode
    from eyvan.server.service.groups import GroupRole

    service = await _service(ceiling=2)

    with pytest.raises(ExpansionRefused, match="DAG of 3 children"):
        await service.submit_dag(
            _principal(),
            org_id="acme",
            project_id="research",
            nodes=[DagNode(name=f"n{index}", template=_template()) for index in range(3)],
            idempotency_key="dag-1",
        )
    with pytest.raises(ExpansionRefused, match="group of 3 children"):
        await service.submit_group(
            _principal(),
            org_id="acme",
            project_id="research",
            roles=[
                GroupRole(name="a", template=_template(name="a"), count=2),
                GroupRole(name="b", template=_template(name="b"), count=1),
            ],
            idempotency_key="group-1",
        )

    async with service.repository.transaction() as tx:
        assert await tx.list_workloads() == []


async def test_the_default_ceiling_is_the_limit_this_package_already_enforced() -> None:
    """An operator who configures nothing must see no change at all."""

    service = await _service()

    parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=6,
        idempotency_key="batch-1",
    )

    assert all(child.schedulable for child in children), "no waves unless an operator asks for them"
    async with service.repository.transaction() as tx:
        assert await tx.get_expansion_admission(parent.id) is None, "no durable plan a rollback could not read"


# --------------------------------------------------------------------------- waves


async def test_a_bounded_expansion_creates_every_child_and_admits_one_wave() -> None:
    """Created and owed, not truncated: the difference the caller is told about."""

    service = await _service(ceiling=100, wave=3)

    parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=8,
        idempotency_key="batch-1",
    )

    assert len(children) == 8, "every task exists; a silent truncation would be worse than a refusal"
    assert [child.schedulable for child in children] == [True, True, True, False, False, False, False, False]
    admission = await service.get_expansion_admission(_principal(), parent.id)
    assert admission is not None
    assert (admission.mode, admission.wave_size, admission.released_children) == (ExpansionMode.WAVES, 3, 3)
    assert admission.deferred_children == 5
    assert "waves of 3" in admission.describe()


async def test_a_deferred_child_is_invisible_to_the_scheduler_until_its_wave_opens() -> None:
    """The whole point: the queue carries a bounded slice of one array."""

    service = await _service(ceiling=100, wave=2, scheduler=True)
    async with service.repository.transaction() as tx:
        for index in range(6):
            await tx.put_cell(_cell(f"cell-{index}"))
    _parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=6,
        idempotency_key="batch-1",
    )

    result = await _controller(service).reconcile_once()

    assert len(result.considered_order) == 2, "four cells sat free rather than let one array take them"
    assert {placement.workload_id for placement in result.placements} == {children[0].id, children[1].id}


async def test_the_next_wave_opens_only_as_earlier_children_stop_competing() -> None:
    service = await _service(ceiling=100, wave=3)
    _parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=8,
        idempotency_key="batch-1",
    )
    controller = _controller(service)

    assert await controller.release_expansion_waves_once() == 0, "the wave is full"

    await _finish(service, children[0].id, children[1].id)

    assert await controller.release_expansion_waves_once() == 2
    async with service.repository.transaction() as tx:
        stored = [await tx.get_workload(child.id) for child in children]
    assert [item.schedulable for item in stored if item is not None] == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]


async def test_children_are_released_in_task_order_not_identifier_order() -> None:
    """Task 3 runs before task 7; ids are random, so ordering must not use them."""

    service = await _service(ceiling=100, wave=2)
    _parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=6,
        idempotency_key="batch-1",
    )
    await _finish(service, children[0].id, children[1].id)

    await _controller(service).release_expansion_waves_once()

    async with service.repository.transaction() as tx:
        stored = [await tx.get_workload(child.id) for child in children]
    released = [item.spec.environment[BATCH_INDEX_ENV] for item in stored if item is not None and item.schedulable]
    assert released == ["0", "1", "2", "3"]


async def test_a_finished_expansion_closes_its_ledger_rather_than_being_rescanned() -> None:
    """The leader's phase must not scan a plan that can never release anything."""

    service = await _service(ceiling=100, wave=4)
    parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=6,
        idempotency_key="batch-1",
    )
    controller = _controller(service)
    await _finish(service, *(child.id for child in children[:4]))
    assert await controller.release_expansion_waves_once() == 2

    assert await controller.release_expansion_waves_once() == 0

    async with service.repository.transaction() as tx:
        admission = await tx.get_expansion_admission(parent.id)
        assert admission is not None
        assert admission.state is ExpansionAdmissionState.COMPLETE
        assert await tx.list_expansion_admissions(state=ExpansionAdmissionState.ADMITTING) == []


async def test_cancelling_the_array_closes_its_ledger_instead_of_releasing_forever() -> None:
    """A cancelled array's remaining children must not be re-admitted one by one."""

    service = await _service(ceiling=100, wave=2)
    parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=6,
        idempotency_key="batch-1",
    )

    await service.cancel_workload(_principal(), parent.id, idempotency_key="cancel-1")
    released = await _controller(service).release_expansion_waves_once()

    assert released == 0
    async with service.repository.transaction() as tx:
        admission = await tx.get_expansion_admission(parent.id)
        assert admission is not None and admission.state is ExpansionAdmissionState.COMPLETE
        stored = [await tx.get_workload(child.id) for child in children]
    assert all(item is not None and item.state is WorkloadState.CANCELLED for item in stored)


# --------------------------------------------------------------------------- quota interaction


def test_quota_headroom_is_the_binding_dimension_and_unknown_when_it_cannot_be_known() -> None:
    """Room for two 4-TPU children under a 16-TPU quota with 8 already leased."""

    assert quota_headroom_children({"tpu": 16.0}, {"tpu": 8.0}, {"tpu": 4.0, "cpu": 1.0}) == 2
    assert quota_headroom_children({"tpu": 16.0, "cpu": 5.0}, {"tpu": 0.0, "cpu": 4.0}, {"tpu": 4.0, "cpu": 1.0}) == 1
    # No quota, and a quota over a dimension this child does not use, are both
    # unknown -- never zero, which would throttle a tenant nobody limited.
    assert quota_headroom_children({}, {}, {"cpu": 1.0}) is None
    assert quota_headroom_children({"gpu": 8.0}, {}, {"cpu": 1.0}) is None


def test_a_full_quota_still_admits_one_child_rather_than_deadlocking() -> None:
    """A wave of zero is not backpressure; it is an expansion that never starts."""

    limits = ExpansionLimits(ceiling=100, wave_size=10)

    assert wave_size_for(20, limits=limits, quota_headroom=0) == 1
    assert wave_size_for(20, limits=limits, quota_headroom=3) == 3
    assert wave_size_for(20, limits=limits, quota_headroom=99) == 10
    assert wave_size_for(20, limits=ExpansionLimits(ceiling=100), quota_headroom=None) == 20


async def test_the_wave_narrows_to_what_the_projects_quota_can_hold() -> None:
    """A quota with room for two children admits two, not the configured ten."""

    service = await _service(ceiling=100, wave=10)
    async with service.repository.transaction() as tx:
        await tx.put_quota(
            QuotaRecord(
                org_id="acme",
                project_id="research",
                limits=ResourceLimits(cpu=4.0),
                updated_by="operator",
            )
        )

    parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(cpu=2.0),
        count=6,
        idempotency_key="batch-1",
    )

    assert sum(1 for child in children if child.schedulable) == 2
    admission = await service.get_expansion_admission(_principal(), parent.id)
    assert admission is not None and admission.wave_size == 2


async def test_an_unquotaed_project_keeps_the_configured_wave() -> None:
    """Unknown headroom narrows nothing: no quota is not a quota of zero."""

    service = await _service(ceiling=100, wave=4)

    _parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(cpu=2.0),
        count=9,
        idempotency_key="batch-1",
    )

    assert sum(1 for child in children if child.schedulable) == 4


# --------------------------------------------------------------------------- restart


async def test_a_restart_mid_expansion_neither_loses_nor_duplicates_a_child() -> None:
    """The plan is durable, so a new leader resumes it rather than re-deciding it.

    Every wave admission writes an audit event, so "exactly once per child" is a
    property of persisted rows rather than of anything held in memory: a child
    released twice would appear twice, and one lost would never appear.
    """

    service = await _service(ceiling=100, wave=3)
    parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=9,
        idempotency_key="batch-1",
    )
    repository = service.repository

    released_total = 0
    for wave in range(3):
        await _finish(service, *(child.id for child in children[wave * 3 : wave * 3 + 3]))
        # A new service and a new controller over the same durable state: nothing
        # in this process survives the restart, and the plan still does.
        restarted = ControlPlaneService(repository, expansion_limits=ExpansionLimits(ceiling=1, wave_size=1))
        released_total += await _controller(restarted).release_expansion_waves_once()

    assert released_total == 6, "the six deferred children were released, and only once each"
    admitted = await _wave_admissions(service)
    assert sorted(admitted) == sorted(child.id for child in children[3:])
    assert len(set(admitted)) == len(admitted), "no child was released twice"
    async with repository.transaction() as tx:
        stored = [await tx.get_workload(child.id) for child in children]
        admission = await tx.get_expansion_admission(parent.id)
    assert all(item is not None and item.schedulable for item in stored)
    assert admission is not None and admission.released_children == 9


async def test_a_replayed_submission_does_not_open_a_second_plan() -> None:
    """Two ledgers over one array would be two leaders releasing two plans."""

    service = await _service(ceiling=100, wave=2)
    kwargs = dict(
        org_id="acme",
        project_id="research",
        template=_template(),
        count=5,
        idempotency_key="batch-1",
    )

    parent, _ = await service.submit_batch(_principal(), **kwargs)
    replayed_parent, replayed = await service.submit_batch(_principal(), **kwargs)

    assert replayed_parent.id == parent.id
    assert [child.schedulable for child in replayed] == [True, True, False, False, False]
    async with service.repository.transaction() as tx:
        assert len(await tx.list_expansion_admissions()) == 1


# --------------------------------------------------------------------------- fair share


async def test_a_second_tenant_is_still_placed_while_a_large_expansion_runs() -> None:
    """Fair share is preserved, and the queue carries a bounded slice of the array.

    Both halves matter. A bound that starved the other tenant would be a worse
    failure than the flood, and a placement that let one array occupy the whole
    queue is the flood.
    """

    service = await _service(ceiling=1000, wave=3, scheduler=True)
    async with service.repository.transaction() as tx:
        for index in range(4):
            await tx.put_cell(_cell(f"cell-{index}"))
    _parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=12,
        idempotency_key="batch-1",
    )
    neighbour, _ = await service.submit_workload(
        _principal("globex", "analytics"),
        org_id="globex",
        project_id="analytics",
        spec=_template(name="neighbour"),
        idempotency_key="workload-1",
    )

    result = await _controller(service).reconcile_once()

    placed = {placement.workload_id for placement in result.placements}
    assert neighbour.id in placed, "the other tenant was placed while the array ran"
    array_candidates = {child.id for child in children} & set(result.considered_order)
    assert len(array_candidates) == 3, "one array may occupy at most its wave of the shared queue"


# --------------------------------------------------------------------------- durability


async def test_the_plan_survives_the_local_backend_rather_than_being_dropped(tmp_path) -> None:
    """The document version moves rather than silently losing a collection.

    A dropped ledger is an expansion whose deferred children are either stranded
    or released all at once after one restart, so the local store gets a version
    that can carry it -- and an older build refuses that file by version instead
    of reading it and discarding the plan.
    """

    from eyvan.server.persistence.local import (
        LOCAL_CONTROL_PLANE_V4_SCHEMA_VERSION,
        LocalJsonRepository,
    )

    path = tmp_path / "control-plane.json"
    repository = LocalJsonRepository(path)
    await repository.open()
    service = ControlPlaneService(repository, expansion_limits=ExpansionLimits(ceiling=100, wave_size=2))
    parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=5,
        idempotency_key="batch-1",
    )

    # The bytes on disk, not a decoded convenience: the version an older build
    # would refuse is the thing being asserted.
    document = json.loads(path.read_text(encoding="utf-8"))
    # The first repository holds the exclusive owner lock, exactly as a running
    # server does; a restart releases it before the next one opens the file.
    await repository.close()
    reopened = LocalJsonRepository(path)
    await reopened.open()
    async with reopened.transaction() as tx:
        restored = await tx.get_expansion_admission(parent.id)
    await reopened.close()

    assert document["version"] == LOCAL_CONTROL_PLANE_V4_SCHEMA_VERSION
    assert len(document["expansion_admissions"]) == 1
    assert restored is not None
    assert (restored.total_children, restored.wave_size, restored.released_children) == (5, 2, 2)
    assert restored.workload_id == parent.id
    assert len(children) == 5


async def test_a_lowered_ceiling_does_not_refuse_an_array_that_already_exists() -> None:
    """A replay must not report a failure the durable state contradicts.

    The ceiling is checked after the replay branch for exactly this: an operator
    lowering it must bound the *next* submission, not make an already-created
    array unretriable and tell its submitter nothing was created.
    """

    repository = InMemoryRepository()
    await repository.open()
    kwargs = dict(
        org_id="acme",
        project_id="research",
        template=_template(),
        count=6,
        idempotency_key="batch-1",
    )
    generous = ControlPlaneService(repository, expansion_limits=ExpansionLimits(ceiling=10))
    parent, children = await generous.submit_batch(_principal(), **kwargs)

    tightened = ControlPlaneService(repository, expansion_limits=ExpansionLimits(ceiling=2))
    replayed_parent, replayed = await tightened.submit_batch(_principal(), **kwargs)

    assert replayed_parent.id == parent.id
    assert [child.id for child in replayed] == [child.id for child in children]
    with pytest.raises(ExpansionRefused, match="exceeds the 2-child admission ceiling"):
        await tightened.submit_batch(_principal(), **{**kwargs, "idempotency_key": "batch-2"})
