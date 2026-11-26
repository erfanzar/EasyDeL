# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A DAG's fan-out is bounded wherever it happens, and survives a restart.

A DAG with five hundred independent roots floods the queue exactly like an array
does, so the same wave bounds it. So does one root fanning out to five hundred
dependents: the dependency is what decides a node is *eligible*, and the wave is
what decides how many eligible nodes compete at once. Both paths read the same
plan and count the same occupancy, so a stage completing cannot exceed the bound
its expansion was admitted under.

What the wave *phase* still does not do is release a dependent node. Its gate is
its predecessor, and releasing it early would run a stage before the stage it
waits for. This asserts both halves so neither is mistaken for the other later.
"""

from __future__ import annotations

from eyvan.config import EyvanSettings
from eyvan.records import (
    CommandExecution,
    CoordinationKind,
    ExpansionAdmissionState,
    ExpansionMode,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadState,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.dag import DagNode
from eyvan.server.service.expansion import ExpansionLimits


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(
            RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),
            RoleGrant(role=Role.FLEET_ADMIN),
        ),
    )


def _other_principal() -> Principal:
    """A second tenant, in its own organization and project."""

    return Principal(
        subject="bob",
        grants=(
            RoleGrant(role=Role.PROJECT_OPERATOR, org_id="globex", project_id="ops"),
            RoleGrant(role=Role.FLEET_ADMIN),
        ),
    )


def _template(name: str = "stage") -> WorkloadDefinition:
    return WorkloadDefinition(
        name=name,
        execution=CommandExecution(command="python stage.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
    )


async def _service(*, wave: int = 0, ceiling: int = 1000) -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    return ControlPlaneService(
        repository,
        expansion_limits=ExpansionLimits(ceiling=ceiling, wave_size=wave),
    )


def _controller(service: ControlPlaneService) -> EyvanController:
    return EyvanController(service.repository, service, settings=EyvanSettings())


async def _succeed(service: ControlPlaneService, *workload_ids: str) -> None:
    async with service.repository.transaction() as tx:
        for workload_id in workload_ids:
            stored = await tx.get_workload(workload_id)
            assert stored is not None
            await tx.put_workload(stored.model_copy(update={"state": WorkloadState.SUCCEEDED}))


def _by_name(nodes) -> dict[str, object]:
    return {node.spec.name: node for node in nodes}


async def test_a_wide_dags_roots_are_admitted_in_waves() -> None:
    service = await _service(wave=2)

    parent, nodes = await service.submit_dag(
        _principal(),
        org_id="acme",
        project_id="research",
        nodes=[DagNode(name=f"root-{index}", template=_template()) for index in range(5)],
        idempotency_key="dag-1",
        name_prefix="wide",
    )

    assert parent.coordination_kind is CoordinationKind.DAG
    assert [node.schedulable for node in nodes] == [True, True, False, False, False]
    admission = await service.get_expansion_admission(_principal(), parent.id)
    assert admission is not None
    assert (admission.kind, admission.mode, admission.wave_size) == (CoordinationKind.DAG, ExpansionMode.WAVES, 2)
    assert admission.total_children == 5, "the plan governs the roots, which are the entry fan-out"


async def test_a_dependent_node_is_never_released_by_the_wave_phase() -> None:
    """Its gate is its predecessor. Releasing it early would run a stage out of order."""

    service = await _service(wave=1)
    parent, nodes = await service.submit_dag(
        _principal(),
        org_id="acme",
        project_id="research",
        nodes=[
            DagNode(name="prepare", template=_template()),
            DagNode(name="train", template=_template(), depends_on=("prepare",)),
            DagNode(name="evaluate", template=_template(), depends_on=("train",)),
        ],
        idempotency_key="dag-1",
        name_prefix="pipeline",
    )
    controller = _controller(service)
    stages = _by_name(nodes)

    # A plan is opened because the DAG is larger than one wave; the wave phase
    # still must not touch a dependent, whose gate is its predecessor.
    assert [node.schedulable for node in nodes] == [True, False, False]
    async with service.repository.transaction() as tx:
        admission = await tx.get_expansion_admission(parent.id)
    assert admission is not None and admission.total_children == 3

    assert await controller.release_expansion_waves_once() == 0
    async with service.repository.transaction() as tx:
        stored = await tx.get_workload(stages["pipeline-train"].id)
    assert stored is not None and not stored.schedulable, "a dependent node is not a wave admission"

    await _succeed(service, stages["pipeline-prepare"].id)
    assert await controller.release_ready_dag_nodes_once() == 1
    async with service.repository.transaction() as tx:
        stored = await tx.get_workload(stages["pipeline-train"].id)
    assert stored is not None and stored.schedulable, "the dependency path still releases it"


async def test_a_dependency_release_counts_against_the_next_wave() -> None:
    """A node the DAG phase opened is occupying the queue, wave or no wave.

    The two paths are not independent: if the wave ignored nodes released by
    dependency satisfaction, an expansion would exceed its own bound as soon as
    one stage finished.
    """

    service = await _service(wave=2)
    _parent, nodes = await service.submit_dag(
        _principal(),
        org_id="acme",
        project_id="research",
        nodes=[
            DagNode(name="root-a", template=_template()),
            DagNode(name="root-b", template=_template()),
            DagNode(name="root-c", template=_template()),
            DagNode(name="after-a", template=_template(), depends_on=("root-a",)),
        ],
        idempotency_key="dag-1",
        name_prefix="mixed",
    )
    controller = _controller(service)
    stages = _by_name(nodes)

    await _succeed(service, stages["mixed-root-a"].id)
    assert await controller.release_ready_dag_nodes_once() == 1

    # root-a finished, freeing one place; after-a took it. root-b still occupies
    # the other, so root-c waits.
    assert await controller.release_expansion_waves_once() == 0
    async with service.repository.transaction() as tx:
        stored = await tx.get_workload(stages["mixed-root-c"].id)
    assert stored is not None and not stored.schedulable

    await _succeed(service, stages["mixed-after-a"].id)

    assert await controller.release_expansion_waves_once() == 1
    async with service.repository.transaction() as tx:
        stored = await tx.get_workload(stages["mixed-root-c"].id)
    assert stored is not None and stored.schedulable


async def test_gated_dependents_keep_the_wave_ledger_open() -> None:
    """Closing the plan while a DAG still owes releases is a flood valve.

    The wave phase releases roots only, so a roots-only waiting count reads
    empty the moment every root is out -- while dependents are still gated. A
    ledger that closes there hands the DAG phase ``wave_size=None``, and its
    next release runs every ready stage at once, which is the queue flood the
    wave exists to bound.
    """

    service = await _service(wave=1)
    parent, nodes = await service.submit_dag(
        _principal(),
        org_id="acme",
        project_id="research",
        nodes=[
            DagNode(name="prepare", template=_template()),
            DagNode(name="train", template=_template(), depends_on=("prepare",)),
            DagNode(name="score", template=_template(), depends_on=("prepare",)),
        ],
        idempotency_key="dag-1",
        name_prefix="fanout",
    )
    controller = _controller(service)
    stages = _by_name(nodes)

    # The root is out; both dependents are still gated, so the plan stays open.
    assert await controller.release_expansion_waves_once() == 0
    async with service.repository.transaction() as tx:
        admission = await tx.get_expansion_admission(parent.id)
    assert admission is not None and admission.state is ExpansionAdmissionState.ADMITTING

    # Prepare finished and freed the one slot. The DAG phase may release one
    # dependent under the bound -- not both, which a closed ledger would have
    # allowed.
    await _succeed(service, stages["fanout-prepare"].id)
    assert await controller.release_ready_dag_nodes_once() == 1
    dependents = [stages["fanout-train"], stages["fanout-score"]]
    async with service.repository.transaction() as tx:
        schedulable = [node.spec.name for node in dependents if (await tx.get_workload(node.id)).schedulable]
    assert len(schedulable) == 1, "one wave slot opened, so one dependent competes"

    # The second dependent waits for the first to finish, and the ledger
    # closes only once every child has been released.
    async with service.repository.transaction() as tx:
        admission = await tx.get_expansion_admission(parent.id)
    assert admission is not None and admission.state is ExpansionAdmissionState.ADMITTING
    async with service.repository.transaction() as tx:
        done = [node.id for node in dependents if (await tx.get_workload(node.id)).schedulable]
    await _succeed(service, *done)
    assert await controller.release_ready_dag_nodes_once() == 1
    assert await controller.release_expansion_waves_once() == 0
    async with service.repository.transaction() as tx:
        admission = await tx.get_expansion_admission(parent.id)
    assert admission is not None and admission.state is ExpansionAdmissionState.COMPLETE


async def test_dag_progress_is_derivable_after_a_control_plane_restart() -> None:
    """Nothing about a half-admitted DAG lives in the leader's memory."""

    service = await _service(wave=2)
    parent, nodes = await service.submit_dag(
        _principal(),
        org_id="acme",
        project_id="research",
        nodes=[DagNode(name=f"root-{index}", template=_template()) for index in range(6)],
        idempotency_key="dag-1",
        name_prefix="wide",
    )
    repository = service.repository
    stages = _by_name(nodes)
    await _succeed(service, stages["wide-root-0"].id, stages["wide-root-1"].id)

    # A new service and controller over the same durable rows, holding the plan
    # this process never saw.
    restarted = ControlPlaneService(repository)
    released = await _controller(restarted).release_expansion_waves_once()

    assert released == 2
    async with repository.transaction() as tx:
        stored = [await tx.get_workload(node.id) for node in nodes]
        admission = await tx.get_expansion_admission(parent.id)
    assert [item.schedulable for item in stored if item is not None] == [True, True, True, True, False, False]
    assert admission is not None
    assert admission.released_children == 4
    assert admission.state is ExpansionAdmissionState.ADMITTING


async def _competing(service: ControlPlaneService, parent_id: str) -> list[str]:
    """Return the names of one DAG's children that are holding a queue place."""

    async with service.repository.transaction() as tx:
        children = await tx.list_child_workloads(parent_id)
    return sorted(
        child.spec.name
        for child in children
        if child.schedulable
        and child.state not in {WorkloadState.SUCCEEDED, WorkloadState.FAILED, WorkloadState.CANCELLED}
    )


async def _release_events(service: ControlPlaneService) -> list[str]:
    """Return every workload a DAG release was recorded for, with repeats."""

    async with service.repository.transaction() as tx:
        events = await tx.list_events(entity_type="workload", limit=1000)
    return [event.entity_id for event in events if event.action == "workload.dag_released"]


async def test_a_completing_stage_does_not_exceed_the_wave() -> None:
    """The defect: a stage finishing used to release every dependent it unblocked.

    One root fanning out to a wide second stage is the same flood as a wide entry
    fan-out, and the wave the expansion was admitted under has to bound both.
    """

    service = await _service(wave=2)
    parent, nodes = await service.submit_dag(
        _principal(),
        org_id="acme",
        project_id="research",
        nodes=[
            DagNode(name="root-a", template=_template()),
            DagNode(name="root-b", template=_template()),
            DagNode(name="root-c", template=_template()),
            *[DagNode(name=f"fan-{index}", template=_template(), depends_on=("root-a",)) for index in range(4)],
        ],
        idempotency_key="dag-1",
        name_prefix="wide",
    )
    controller = _controller(service)
    stages = _by_name(nodes)
    assert await _competing(service, parent.id) == ["wide-root-a", "wide-root-b"]

    await _succeed(service, stages["wide-root-a"].id)

    # root-a's place is free and root-b still holds the other, so exactly one of
    # the four dependents it unblocked may compete.
    assert await controller.release_ready_dag_nodes_once() == 1
    assert await _competing(service, parent.id) == ["wide-fan-0", "wide-root-b"]
    assert await controller.release_ready_dag_nodes_once() == 0
    assert await _competing(service, parent.id) == ["wide-fan-0", "wide-root-b"]

    await _succeed(service, stages["wide-root-b"].id)
    assert await controller.release_ready_dag_nodes_once() == 1
    assert await _competing(service, parent.id) == ["wide-fan-0", "wide-fan-1"]


async def test_an_interrupted_dependent_release_neither_loses_nor_duplicates_a_node() -> None:
    """Nothing about a half-released stage lives in the leader that released it.

    The whole decision -- the plan, the occupancy, and which node is next -- is
    read back out of durable rows, so a replacement leader resumes at the node
    the dead one stopped at instead of re-releasing what it already flipped or
    skipping what it had not.
    """

    service = await _service(wave=2)
    repository = service.repository
    parent, nodes = await service.submit_dag(
        _principal(),
        org_id="acme",
        project_id="research",
        nodes=[
            DagNode(name="root-a", template=_template()),
            DagNode(name="root-b", template=_template()),
            DagNode(name="root-c", template=_template()),
            *[DagNode(name=f"fan-{index}", template=_template(), depends_on=("root-a",)) for index in range(5)],
        ],
        idempotency_key="dag-1",
        name_prefix="pipeline",
    )
    stages = _by_name(nodes)
    await _succeed(service, stages["pipeline-root-a"].id, stages["pipeline-root-b"].id)

    drained: list[str] = []
    for _ in range(5):
        # A fresh service and controller over the same rows on every pass: the
        # leader that released the previous nodes is gone, and nothing it knew
        # survived except the rows it committed.
        restarted = ControlPlaneService(repository)
        await _controller(restarted).release_ready_dag_nodes_once()
        competing = await _competing(restarted, parent.id)
        assert len(competing) <= 2, "a restart must not release past the wave"
        drained.extend(competing)
        await _succeed(restarted, *[stages[name].id for name in competing])

    released = await _release_events(service)
    assert sorted(released) == sorted(set(released)), "no node was released twice"
    assert set(drained) == {f"pipeline-fan-{index}" for index in range(5)}, "no node was lost"
    async with repository.transaction() as tx:
        stored = [await tx.get_workload(stages[f"pipeline-fan-{index}"].id) for index in range(5)]
    assert all(item is not None and item.schedulable for item in stored)


async def test_a_dag_plans_over_every_node_so_dependents_are_bounded_too() -> None:
    """The case that actually floods a queue.

    Bounding the entry fan-out alone left one root fanning out to N dependents
    entirely unbounded: the roots fitted in a wave, so the DAG was admitted
    immediately, no plan was written, and every dependent was released the
    moment its predecessor succeeded. A dependent's predecessor says *when* it
    may run and nothing about *how many* may run at once.
    """

    service = await _service(wave=2)
    parent, nodes = await service.submit_dag(
        _principal(),
        org_id="acme",
        project_id="research",
        nodes=[
            DagNode(name="root", template=_template()),
            *[DagNode(name=f"fan-{index}", template=_template(), depends_on=("root",)) for index in range(4)],
        ],
        idempotency_key="dag-1",
        name_prefix="immediate",
    )
    controller = _controller(service)

    # A plan now exists precisely because the DAG is larger than one wave, even
    # though its single root was never going to flood anything by itself.
    async with service.repository.transaction() as tx:
        admission = await tx.get_expansion_admission(parent.id)
    assert admission is not None
    assert admission.total_children == 5

    await _succeed(service, _by_name(nodes)["immediate-root"].id)

    released = await controller.release_ready_dag_nodes_once()

    assert released < 4, "the dependents were released without a bound"
    assert len(await _competing(service, parent.id)) <= 2


async def test_a_second_tenant_is_unaffected_while_a_large_dag_drains() -> None:
    """The bound is per expansion, so one tenant's backlog is not another's."""

    service = await _service(wave=1)
    big, big_nodes = await service.submit_dag(
        _principal(),
        org_id="acme",
        project_id="research",
        nodes=[
            *[DagNode(name=f"root-{index}", template=_template()) for index in range(8)],
            *[DagNode(name=f"fan-{index}", template=_template(), depends_on=("root-0",)) for index in range(8)],
        ],
        idempotency_key="dag-big",
        name_prefix="big",
    )
    small, small_nodes = await service.submit_dag(
        _other_principal(),
        org_id="globex",
        project_id="ops",
        nodes=[
            DagNode(name="prepare", template=_template()),
            DagNode(name="spare-a", template=_template()),
            DagNode(name="spare-b", template=_template()),
            DagNode(name="train", template=_template(), depends_on=("prepare",)),
        ],
        idempotency_key="dag-small",
        name_prefix="small",
    )
    controller = _controller(service)
    async with service.repository.transaction() as tx:
        assert (await tx.get_expansion_admission(big.id)).wave_size == 1
        assert (await tx.get_expansion_admission(small.id)).wave_size == 1
    await _succeed(service, _by_name(big_nodes)["big-root-0"].id, _by_name(small_nodes)["small-prepare"].id)

    assert await controller.release_ready_dag_nodes_once() == 2

    # Each DAG gets exactly the one place its own wave allows: the small one's
    # stage is not held behind the large one's eight-deep backlog, and the large
    # one does not borrow the small one's place.
    assert await _competing(service, big.id) == ["big-fan-0"]
    assert await _competing(service, small.id) == ["small-train"]


async def test_an_unbounded_dag_keeps_todays_shape_exactly() -> None:
    """With no wave configured, every root is schedulable and no plan is written."""

    service = await _service()

    parent, nodes = await service.submit_dag(
        _principal(),
        org_id="acme",
        project_id="research",
        nodes=[
            DagNode(name="left", template=_template()),
            DagNode(name="right", template=_template()),
            DagNode(name="merge", template=_template(), depends_on=("left", "right")),
        ],
        idempotency_key="dag-1",
        name_prefix="fan",
    )

    stages = _by_name(nodes)
    assert stages["fan-left"].schedulable and stages["fan-right"].schedulable
    assert not stages["fan-merge"].schedulable
    async with service.repository.transaction() as tx:
        assert await tx.get_expansion_admission(parent.id) is None
