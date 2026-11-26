# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Small DAGs of whole workloads.

A DAG is the batch shape plus one relation. The interesting behaviour is entirely
in what happens at the edges: when a predecessor succeeds, when one fails, and what
a caller is prevented from declaring in the first place.
"""

from __future__ import annotations

import pytest

from eyvan.records import (
    CommandExecution,
    CoordinationKind,
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
from eyvan.server.service.dag import DagError, DagNode, validate


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


def _template(name: str = "stage") -> WorkloadDefinition:
    return WorkloadDefinition(
        name=name,
        execution=CommandExecution(command="python stage.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
    )


def _pipeline() -> tuple[DagNode, ...]:
    """prepare -> train -> {evaluate, export}."""

    return (
        DagNode("prepare", _template()),
        DagNode("train", _template(), depends_on=("prepare",)),
        DagNode("evaluate", _template(), depends_on=("train",)),
        DagNode("export", _template(), depends_on=("train",)),
    )


# --------------------------------------------------------------------------- validation


def test_topological_order_places_every_predecessor_first() -> None:
    """The caller creates records in this order, so predecessor ids always resolve."""

    order = validate(_pipeline())

    assert order.index("prepare") < order.index("train")
    assert order.index("train") < order.index("evaluate")
    assert order.index("train") < order.index("export")


def test_the_order_is_deterministic_so_a_replay_derives_the_same_keys() -> None:
    """Child idempotency keys are derived from position, so order must be stable."""

    assert validate(_pipeline()) == validate(_pipeline())


def test_a_cycle_is_rejected_with_the_nodes_that_form_it() -> None:
    nodes = (
        DagNode("a", _template(), depends_on=("b",)),
        DagNode("b", _template(), depends_on=("a",)),
    )

    with pytest.raises(DagError, match="cycle"):
        validate(nodes)


def test_a_self_edge_is_rejected() -> None:
    with pytest.raises(DagError, match="cannot depend on itself"):
        validate((DagNode("a", _template(), depends_on=("a",)),))


def test_a_dependency_outside_the_request_is_rejected() -> None:
    """Names, not ids: that restriction is what makes cycles detectable at submit.

    An arbitrary workload id could point across scopes, form a cycle no single
    request can see, or outlive whatever created it.
    """

    with pytest.raises(DagError, match="not a node in this request"):
        validate((DagNode("a", _template(), depends_on=("wl_somewhere_else",)),))


def test_duplicate_node_names_are_rejected() -> None:
    with pytest.raises(DagError, match="unique"):
        validate((DagNode("a", _template()), DagNode("a", _template())))


def test_an_empty_or_oversized_dag_is_rejected() -> None:
    with pytest.raises(DagError, match="at least one node"):
        validate(())
    with pytest.raises(DagError, match="exceeds"):
        validate(tuple(DagNode(f"n{index}", _template()) for index in range(5)), max_fan_out=4)


def test_an_over_deep_chain_is_rejected() -> None:
    chain = [DagNode("n0", _template())]
    for index in range(1, 40):
        chain.append(DagNode(f"n{index}", _template(), depends_on=(f"n{index - 1}",)))

    with pytest.raises(DagError, match="deep"):
        validate(tuple(chain))


# --------------------------------------------------------------------------- submission


async def test_only_root_nodes_are_schedulable_at_submission() -> None:
    """The gate is the existing ``schedulable`` flag the scheduler already filters on."""

    service = ControlPlaneService(InMemoryRepository())

    parent, nodes = await service.submit_dag(
        _principal(), org_id="acme", project_id="research", nodes=_pipeline(), idempotency_key="dag-1"
    )

    by_name = {node.spec.name: node for node in nodes}
    assert parent.coordination_kind is CoordinationKind.DAG
    assert parent.schedulable is False
    assert by_name["dag-prepare"].schedulable is True
    assert by_name["dag-train"].schedulable is False
    assert by_name["dag-train"].depends_on == (by_name["dag-prepare"].id,)
    assert by_name["dag-evaluate"].depends_on == (by_name["dag-train"].id,)


async def test_a_retried_dag_submission_replays_rather_than_duplicating() -> None:
    service = ControlPlaneService(InMemoryRepository())
    kwargs = dict(org_id="acme", project_id="research", nodes=_pipeline(), idempotency_key="dag-1")

    first_parent, first_nodes = await service.submit_dag(_principal(), **kwargs)
    second_parent, second_nodes = await service.submit_dag(_principal(), **kwargs)

    assert first_parent.id == second_parent.id
    assert {node.id for node in first_nodes} == {node.id for node in second_nodes}


# --------------------------------------------------------------------------- release


async def test_a_node_is_released_only_once_every_predecessor_succeeded() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service)
    _parent, nodes = await service.submit_dag(
        _principal(), org_id="acme", project_id="research", nodes=_pipeline(), idempotency_key="dag-1"
    )
    by_name = {node.spec.name: node for node in nodes}

    assert await controller.release_ready_dag_nodes_once() == 0

    async with repository.transaction() as transaction:
        prepare = await transaction.get_workload(by_name["dag-prepare"].id)
        await transaction.put_workload(prepare.model_copy(update={"state": WorkloadState.SUCCEEDED}))

    assert await controller.release_ready_dag_nodes_once() == 1

    async with repository.transaction() as transaction:
        train = await transaction.get_workload(by_name["dag-train"].id)
        evaluate = await transaction.get_workload(by_name["dag-evaluate"].id)
    assert train.schedulable is True
    assert evaluate.schedulable is False, "a grandchild waits for its own predecessor"


async def test_a_failed_predecessor_cancels_its_dependents_rather_than_stranding_them() -> None:
    """A stranded node is the orphan defect again: PENDING forever, never settled.

    CANCELLED also states the truth -- this never ran because what it waited for did
    not succeed -- where a silent PENDING says nothing at all.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service)
    _parent, nodes = await service.submit_dag(
        _principal(), org_id="acme", project_id="research", nodes=_pipeline(), idempotency_key="dag-1"
    )
    by_name = {node.spec.name: node for node in nodes}

    async with repository.transaction() as transaction:
        prepare = await transaction.get_workload(by_name["dag-prepare"].id)
        await transaction.put_workload(prepare.model_copy(update={"state": WorkloadState.FAILED}))

    assert await controller.release_ready_dag_nodes_once() == 1

    async with repository.transaction() as transaction:
        train = await transaction.get_workload(by_name["dag-train"].id)
    assert train.state is WorkloadState.CANCELLED
    assert train.schedulable is False


async def test_a_fan_in_node_waits_for_every_branch() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service)
    nodes = (
        DagNode("left", _template()),
        DagNode("right", _template()),
        DagNode("merge", _template(), depends_on=("left", "right")),
    )
    _parent, created = await service.submit_dag(
        _principal(), org_id="acme", project_id="research", nodes=nodes, idempotency_key="dag-1"
    )
    by_name = {node.spec.name: node for node in created}

    async with repository.transaction() as transaction:
        left = await transaction.get_workload(by_name["dag-left"].id)
        await transaction.put_workload(left.model_copy(update={"state": WorkloadState.SUCCEEDED}))
    assert await controller.release_ready_dag_nodes_once() == 0

    async with repository.transaction() as transaction:
        right = await transaction.get_workload(by_name["dag-right"].id)
        await transaction.put_workload(right.model_copy(update={"state": WorkloadState.SUCCEEDED}))
    assert await controller.release_ready_dag_nodes_once() == 1

    async with repository.transaction() as transaction:
        assert (await transaction.get_workload(by_name["dag-merge"].id)).schedulable is True


async def test_a_gated_node_is_invisible_to_the_scheduler() -> None:
    """The gate has to be the flag placement already respects, not a parallel one."""

    from eyvan.server.service.scheduling import CoreSchedulingService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    _parent, nodes = await service.submit_dag(
        _principal(), org_id="acme", project_id="research", nodes=_pipeline(), idempotency_key="dag-1"
    )
    by_name = {node.spec.name: node for node in nodes}

    async with repository.transaction() as transaction:
        workloads = await transaction.list_workloads()
        cells = await transaction.list_cells()
    considered = set(service.scheduler.schedule(workloads, cells).considered_order)

    assert considered == {by_name["dag-prepare"].id}
