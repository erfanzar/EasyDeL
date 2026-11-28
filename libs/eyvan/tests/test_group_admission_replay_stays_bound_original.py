# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Group admission replay stays bound to the original lease identities."""

from datetime import timedelta

import pytest
from test_group_lease_lifecycle import repository as _repository_fixture

from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    PlacementMode,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkerResourceScope,
    WorkloadDefinition,
    WorkloadState,
    utcnow,
)
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import ConflictError

repository = _repository_fixture
ADMIN = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
OPERATOR = Principal(
    subject="alice", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),)
)


async def _group(service, repository, workload, *, name, count):
    members = []
    for index in range(count):
        cell = CellRecord(
            id=f"{name}-cell-{index}",
            provider="manual",
            capacity_backend="manual",
            executor="eyvan-native",
            region="local",
            state=CellState.READY,
            capacity=ResourceVector(tpu=6),
            allocatable=ResourceVector(tpu=6),
            worker_count=6 // count,
        )
        async with repository.transaction() as tx:
            await tx.put_cell(cell)
        members.append(
            LeaseRecord(
                id=f"{name}-{index}",
                workload_id=workload.id,
                cell_id=cell.id,
                org_id=workload.org_id,
                project_id=workload.project_id,
                generation=workload.generation,
                resources=workload.spec.gang,
                fencing_token=f"{name}-fence-{index}",
                group_id=name,
                group_index=index,
                expires_at=utcnow() + timedelta(minutes=5),
            )
        )
    members = tuple(members)
    stored, replayed = await service.create_lease_group(ADMIN, members, idempotency_key=name)
    assert stored == members and not replayed
    return members


async def _original_group(repository):
    service = ControlPlaneService(repository)
    workload, replayed = await service.submit_workload(
        OPERATOR,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="replay-group",
            provider="manual",
            required_executor="eyvan-native",
            execution=CommandExecution(command="true"),
            gang=ResourceGang(replicas=1, per_replica=ResourceVector(tpu=1)),
            placement=PlacementMode.ELASTIC,
            worker_resource_scope=WorkerResourceScope.PER_WORKER,
            worker_count=6,
            max_failure_retries=1,
        ),
        idempotency_key="submit",
    )
    assert not replayed
    original = await _group(service, repository, workload, name="original", count=2)
    return service, workload, original


async def _snapshot(repository):
    async with repository.transaction() as tx:
        return (
            await tx.list_workloads(),
            await tx.list_leases(),
            await tx.list_events(),
            await tx.list_outbox(),
        )


@pytest.mark.parametrize("replacement_count", [0, 2, 3], ids=["pending", "same-size", "different-size"])
async def test_group_replay_returns_original_members_after_failure_retry(repository, replacement_count):
    service, workload, original = await _original_group(repository)
    replay, replayed = await service.create_lease_group(ADMIN, original, idempotency_key="original")
    assert replayed and replay == original

    resumed, settled, _ = await service.settle_failed_workload(ADMIN, workload.id, expected_generation=1)
    assert resumed.state is WorkloadState.PENDING and resumed.generation == 2
    assert resumed.lease_ids == () and resumed.lease_id is None
    assert settled is not None and settled.state is LeaseState.RELEASED
    if replacement_count:
        await _group(service, repository, resumed, name="replacement", count=replacement_count)

    before = await _snapshot(repository)
    async with repository.transaction() as tx:
        original_settled = tuple([await tx.get_lease(member.id) for member in original])
    assert all(member is not None and member.settled_at is not None for member in original_settled)
    replay, replayed = await service.create_lease_group(ADMIN, original, idempotency_key="original")
    assert replayed
    assert replay == original_settled
    assert tuple(member.id for member in replay) == ("original-0", "original-1")
    assert all(member.generation == 1 and member.state is LeaseState.RELEASED for member in replay)
    assert await _snapshot(repository) == before


@pytest.mark.parametrize(
    "updates",
    [
        {"group_id": "replacement"},
        {"group_index": 2},
        {"workload_id": "another-workload"},
        {"org_id": "another-org"},
        {"project_id": "another-project"},
        {"generation": 2},
        {"cell_id": "another-cell"},
        {"cell_generation": 2},
        {"fencing_token": "replacement-fence"},
    ],
    ids=["group", "index", "workload", "org", "project", "generation", "cell", "cell-generation", "fence"],
)
async def test_group_replay_refuses_original_ids_rebound_to_other_ownership(repository, updates):
    service, _, original = await _original_group(repository)
    async with repository.transaction() as tx:
        await tx.put_lease(original[0].model_copy(update=updates))
    before = await _snapshot(repository)
    with pytest.raises(ConflictError, match="original lease group"):
        await service.create_lease_group(ADMIN, original, idempotency_key="original")
    assert await _snapshot(repository) == before


async def test_group_replay_still_rejects_a_changed_request(repository):
    service, _, original = await _original_group(repository)
    changed = (original[0].model_copy(update={"fencing_token": "different-request"}), original[1])
    before = await _snapshot(repository)
    with pytest.raises(ConflictError):
        await service.create_lease_group(ADMIN, changed, idempotency_key="original")
    assert await _snapshot(repository) == before
