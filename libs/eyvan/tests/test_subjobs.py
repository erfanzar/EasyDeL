# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Job trees: parent/child relations and cancellation cascade.

Nothing in Eyvan modelled a relation between two workloads before this. The
design decision these tests pin is that a tree is a *relation and a cascade*,
not a fourth fair-share level: a child inherits its parent's
(org, project, principal) and competes as an ordinary sibling.
"""

from __future__ import annotations

import pytest

from eyvan.records import (
    CommandExecution,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import ConflictError, NotFoundError
from eyvan.server.service.subjobs import (
    MAX_JOB_TREE_DEPTH,
    JobTreeError,
    JobTreeTooDeepError,
    collect_descendants,
)


def _principal(project: str = "research") -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id=project),),
    )


def _spec(name: str = "job") -> WorkloadDefinition:
    return WorkloadDefinition(
        name=name,
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
    )


async def _submit(
    service: ControlPlaneService,
    key: str,
    *,
    parent: str | None = None,
    project: str = "research",
) -> WorkloadRecord:
    workload, _ = await service.submit_workload(
        _principal(project),
        org_id="acme",
        project_id=project,
        spec=_spec(key),
        idempotency_key=key,
        parent_workload_id=parent,
    )
    return workload


async def _tree(service: ControlPlaneService) -> tuple[WorkloadRecord, WorkloadRecord, WorkloadRecord]:
    root = await _submit(service, "root")
    child = await _submit(service, "child", parent=root.id)
    grandchild = await _submit(service, "grandchild", parent=child.id)
    return root, child, grandchild


async def test_a_child_records_its_parent_and_inherits_the_scope() -> None:
    service = ControlPlaneService(InMemoryRepository())
    root, child, _ = await _tree(service)

    assert child.parent_workload_id == root.id
    assert (child.org_id, child.project_id) == (root.org_id, root.project_id)
    # The child competes as an ordinary sibling: same fair-share leaf.
    assert child.created_by == root.created_by


async def test_children_are_listed_for_their_parent() -> None:
    service = ControlPlaneService(InMemoryRepository())
    root, child, grandchild = await _tree(service)

    direct = await service.list_child_workloads(_principal(), root.id)

    assert [item.id for item in direct] == [child.id]
    assert [item.id for item in await service.list_child_workloads(_principal(), child.id)] == [grandchild.id]


async def test_cancelling_a_parent_cascades_to_the_whole_subtree() -> None:
    """A launcher's children have no meaning once the launcher is gone."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    root, child, grandchild = await _tree(service)

    await service.cancel_workload(_principal(), root.id, idempotency_key="cancel-root")

    async with repository.transaction() as transaction:
        states = {item: (await transaction.get_workload(item)).state for item in (root.id, child.id, grandchild.id)}
    assert states == {
        root.id: WorkloadState.CANCELLED,
        child.id: WorkloadState.CANCELLED,
        grandchild.id: WorkloadState.CANCELLED,
    }


async def test_cascade_can_be_disabled_for_a_single_workload() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    root, child, _ = await _tree(service)

    await service.cancel_workload(_principal(), root.id, idempotency_key="cancel-root", cascade=False)

    async with repository.transaction() as transaction:
        assert (await transaction.get_workload(root.id)).state is WorkloadState.CANCELLED
        assert (await transaction.get_workload(child.id)).state is WorkloadState.PENDING


async def test_cancelling_a_child_leaves_its_parent_running() -> None:
    """The cascade is strictly downward."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    root, child, _ = await _tree(service)

    await service.cancel_workload(_principal(), child.id, idempotency_key="cancel-child")

    async with repository.transaction() as transaction:
        assert (await transaction.get_workload(root.id)).state is WorkloadState.PENDING


async def test_a_terminal_parent_cannot_gain_new_children() -> None:
    """Otherwise a child outlives its parent and keeps spending the budget."""

    service = ControlPlaneService(InMemoryRepository())
    root = await _submit(service, "root")
    await service.cancel_workload(_principal(), root.id, idempotency_key="cancel-root")

    with pytest.raises(ConflictError, match="cannot gain new children"):
        await _submit(service, "late-child", parent=root.id)


async def test_an_unknown_parent_is_rejected() -> None:
    service = ControlPlaneService(InMemoryRepository())

    with pytest.raises(NotFoundError, match="parent workload"):
        await _submit(service, "orphan", parent="wl-does-not-exist")


async def test_a_child_is_refused_when_the_parent_generation_moved_since_the_capability_was_verified() -> None:
    """The route verifies the token against the parent it read, then admits later.

    A retry or a kick bumps the parent's generation, so the admitting transaction
    has to re-check the generation the capability named: without that, a
    superseded attempt's token could attach children to the current attempt.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    root = await _submit(service, "root")
    # What the route's first transaction read: generation 1, capable of children.
    read = root.model_copy(update={"spec": root.spec.model_copy(update={"can_submit_children": True})})
    async with repository.transaction() as tx:
        await tx.put_workload(read.model_copy(update={"generation": 2}))

    with pytest.raises(ConflictError, match="generation"):
        await service.submit_child_workload(
            parent=read,
            spec=_spec("child"),
            idempotency_key="child-1",
            expected_parent_generation=read.generation,
        )
    assert await service.list_child_workloads(_principal(), root.id) == []

    # Opt-in fence: without an expected generation, admission proceeds as before.
    child, replayed = await service.submit_child_workload(
        parent=read,
        spec=_spec("child"),
        idempotency_key="child-2",
    )
    assert replayed is False
    assert child.parent_workload_id == root.id


async def test_a_cross_project_parent_is_rejected() -> None:
    """A child inherits the parent's budget; crossing projects would misattribute spend."""

    service = ControlPlaneService(InMemoryRepository())
    root = await _submit(service, "root", project="research")

    other = Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="other"),),
    )
    with pytest.raises(JobTreeError, match="not the child's"):
        await service.submit_workload(
            other,
            org_id="acme",
            project_id="other",
            spec=_spec("cross"),
            idempotency_key="cross",
            parent_workload_id=root.id,
        )


# --------------------------------------------------------------------------- traversal


async def test_traversal_terminates_on_a_cycle() -> None:
    """parent_workload_id is an ordinary persisted field, so a cycle is possible."""

    graph = {"a": ["b"], "b": ["c"], "c": ["a"]}

    async def load(parent_id: str):
        return [
            WorkloadRecord(
                id=child,
                org_id="acme",
                project_id="research",
                spec=_spec(child),
                created_by="alice",
                parent_workload_id=parent_id,
            )
            for child in graph.get(parent_id, [])
        ]

    descendants = await collect_descendants("a", load)

    assert [item.id for item in descendants] == ["b", "c"]


async def test_traversal_refuses_an_unbounded_chain() -> None:
    async def load(parent_id: str):
        index = int(parent_id.removeprefix("n"))
        child = f"n{index + 1}"
        return [
            WorkloadRecord(
                id=child,
                org_id="acme",
                project_id="research",
                spec=_spec(child),
                created_by="alice",
                parent_workload_id=parent_id,
            )
        ]

    # A 409, not a 400: the request was well formed and the *stored* tree is the
    # problem, so the caller is told how to get out rather than to fix input.
    with pytest.raises(JobTreeTooDeepError, match="deeper than") as excinfo:
        await collect_descendants("n0", load, max_depth=5)

    assert "cascade=false" in str(excinfo.value)


# --------------------------------------------------------------------------- HTTP surfacing


async def test_a_cross_project_parent_is_a_client_error_not_a_500(tmp_path) -> None:
    """``JobTreeError`` subclassed a bare ``ValueError``, so no handler matched.

    The API has handlers for ``EyvanServiceError``, ``RequestValidationError``,
    ``ValidationError``, and ``TransientRepositoryError`` -- none of which caught
    this. A caller naming a parent in another project got an opaque 500 instead
    of being told what was wrong with their request.
    """

    from fastapi.testclient import TestClient

    from eyvan.config import EyvanSettings
    from eyvan.server.api.app import create_app

    client = TestClient(
        create_app(settings=EyvanSettings(dev_auth_enabled=True, artifact_store_uri=str(tmp_path / "managed"))),
        raise_server_exceptions=False,
    )
    headers = {
        "X-Eyvan-User": "alice",
        "X-Eyvan-Org": "acme",
        "X-Eyvan-Project": "research",
        "X-Eyvan-Roles": "operator",
    }
    spec = {
        "name": "parent",
        "execution": {"kind": "command", "command": "python parent.py"},
    }
    parent = client.post(
        "/api/v1/workloads",
        headers={**headers, "Idempotency-Key": "parent-1"},
        json={"org_id": "acme", "project_id": "research", "spec": spec},
    )
    assert parent.status_code == 201

    # Same org, different project: the child would spend one project's budget
    # against another's quota.
    crossed = client.post(
        "/api/v1/workloads",
        headers={**headers, "X-Eyvan-Project": "other", "Idempotency-Key": "child-1"},
        json={
            "org_id": "acme",
            "project_id": "other",
            "spec": {**spec, "name": "child"},
            "parent_workload_id": parent.json()["id"],
        },
    )

    assert crossed.status_code == 400, crossed.text
    assert "research" in crossed.json()["detail"]


async def test_a_tree_too_deep_to_walk_can_still_be_cancelled_alone() -> None:
    """Otherwise such a parent could never be cancelled at all.

    Every cancel attempt failed on the traversal before reaching the workload the
    caller named, and ``cascade=False`` was not exposed on the route -- so the
    workload was permanently un-cancellable and held its lease forever.
    """

    from eyvan.records import CommandExecution, WorkloadDefinition, WorkloadState
    from eyvan.server.persistence import InMemoryRepository
    from eyvan.server.service import ControlPlaneService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    principal = _principal()

    def _definition(name: str) -> WorkloadDefinition:
        return WorkloadDefinition(name=name, execution=CommandExecution(command="python chain.py"))

    root, _ = await service.submit_workload(
        principal,
        org_id="acme",
        project_id="research",
        spec=_definition("n0"),
        idempotency_key="n0",
    )
    parent_id = root.id
    for index in range(1, MAX_JOB_TREE_DEPTH + 2):
        child, _ = await service.submit_workload(
            principal,
            org_id="acme",
            project_id="research",
            spec=_definition(f"n{index}"),
            idempotency_key=f"n{index}",
            parent_workload_id=parent_id,
        )
        parent_id = child.id

    with pytest.raises(JobTreeTooDeepError):
        await service.cancel_workload(principal, root.id, idempotency_key="cancel-cascade")

    cancelled, _ = await service.cancel_workload(
        principal,
        root.id,
        idempotency_key="cancel-alone",
        cascade=False,
    )

    assert cancelled.state is WorkloadState.CANCELLED
