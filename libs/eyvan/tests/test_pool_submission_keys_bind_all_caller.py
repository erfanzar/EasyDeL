# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pool submission keys must bind all caller-owned execution intent."""

from __future__ import annotations

from typing import Any

import pytest

from eyvan.records import (
    CallableExecution,
    CapacityPoolRecord,
    PlacementMode,
    Principal,
    PriorityBand,
    ResourceVector,
    Role,
    RoleGrant,
    WorkspaceRef,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
from eyvan.server.service.errors import ConflictError


async def _setup() -> tuple[ControlPlaneService, InMemoryRepository, Principal, Principal, CapacityPoolRecord]:
    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    operator = Principal(
        subject="operator",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )
    admin = Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    resources = ResourceVector(cpu=4, memory_gib=16, tpu=4)
    pool = CapacityPoolRecord(
        id="fingerprint-pool",
        fleet_name="research-fleet",
        node_type="v5p-workers",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="test-project",
        region="us-east5",
        zone="us-east5-a",
        capacity=resources,
        allocatable=resources,
        accelerator_type="v5p-8",
        capacity_class="spot",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        max_cells=4,
    )
    await service.upsert_capacity_pool(admin, pool, idempotency_key="pool")
    return service, repository, operator, admin, pool


async def _submit(service: ControlPlaneService, operator: Principal, route: str, key: str, **intent: Any):
    arguments = {
        "fleet_name": "research-fleet",
        "org_id": "acme",
        "project_id": "research",
        "idempotency_key": key,
        **intent,
    }
    if route == "callable":
        return await service.submit_pool_callable(
            operator,
            accelerator_type="v5p-8",
            pods=2,
            execution=CallableExecution(
                artifact_uri="gs://trusted/train.pkl",
                sha256="a" * 64,
                python="3.11",
                result_uri="gs://trusted/results/result.pkl",
            ),
            **arguments,
        )
    return await service.submit_pool_workload(
        operator,
        node_type="v5p-workers",
        command="python train.py",
        **arguments,
    )


_CHECKPOINT = {
    "checkpointable": True,
    "checkpoint_hook": "python checkpoint.py",
    "checkpoint_output_uri": "gs://checkpoints/run-a",
}
_INTENT_CHANGES = [
    pytest.param({}, {"max_preemption_retries": 0}, id="preemption-retry-cap"),
    pytest.param({}, _CHECKPOINT, id="checkpoint-contract"),
    pytest.param(_CHECKPOINT, {**_CHECKPOINT, "checkpoint_hook": "python checkpoint_v2.py"}, id="checkpoint-hook"),
    pytest.param(_CHECKPOINT, {**_CHECKPOINT, "checkpoint_output_uri": "gs://checkpoints/run-b"}, id="checkpoint-uri"),
    pytest.param({}, {"can_submit_children": True}, id="child-authority"),
    pytest.param({}, {"priority_band": PriorityBand.BATCH}, id="priority-band"),
    pytest.param({}, {"intra_band_priority": 10}, id="intra-band-priority"),
    pytest.param(_CHECKPOINT, {**_CHECKPOINT, "checkpoint_grace_seconds": 30}, id="checkpoint-grace"),
    pytest.param({}, {"estimated_duration_seconds": 600}, id="duration"),
    pytest.param({}, {"named_assets": ("dataset-v2",)}, id="named-assets"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["callable", "command"])
@pytest.mark.parametrize(("original_intent", "changed_intent"), _INTENT_CHANGES)
async def test_pool_replay_conflicts_with_changed_execution_intent(
    route: str, original_intent: dict[str, Any], changed_intent: dict[str, Any]
) -> None:
    service, repository, operator, _admin, _pool = await _setup()
    original, replayed = await _submit(service, operator, route, "original", **original_intent)
    assert not replayed
    replay, replayed = await _submit(service, operator, route, "original", **original_intent)
    assert replayed and replay.id == original.id and replay.spec == original.spec
    # The changed request is independently admitted under a fresh key first:
    # validation failures or incomplete checkpoint fixtures cannot count as proof.
    changed, replayed = await _submit(service, operator, route, "changed", **changed_intent)
    assert not replayed and changed.spec != original.spec
    with pytest.raises(ConflictError, match="different request"):
        await _submit(service, operator, route, "original", **changed_intent)
    async with repository.transaction() as tx:
        stored = await tx.get_workload(original.id)
        assert stored is not None and stored.spec == original.spec
        assert len(await tx.list_workloads()) == 2


@pytest.mark.asyncio
async def test_callable_atomic_replay_conflicts_with_elastic_contract() -> None:
    service, repository, operator, _admin, _pool = await _setup()
    original, _ = await _submit(service, operator, "callable", "original", atomic=False)
    assert original.spec.placement is PlacementMode.ELASTIC
    assert original.spec.worker_count == 2
    changed, _ = await _submit(service, operator, "callable", "changed", atomic=True)
    assert changed.spec.placement is PlacementMode.ATOMIC
    with pytest.raises(ConflictError, match="different request"):
        await _submit(service, operator, "callable", "original", atomic=True)
    async with repository.transaction() as tx:
        stored = await tx.get_workload(original.id)
        assert stored is not None and stored.spec == original.spec


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["uri", "signature"])
async def test_command_workspace_reference_is_not_only_its_digest(field: str) -> None:
    service, _repository, operator, _admin, _pool = await _setup()
    workspace = WorkspaceRef(uri="gs://trusted/workspace-a.tar.gz", sha256="b" * 64, signature="c" * 64)
    changed_workspace = WorkspaceRef.model_validate(
        {**workspace.model_dump(), field: "gs://trusted/workspace-b.tar.gz" if field == "uri" else "d" * 64}
    )
    original, _ = await _submit(service, operator, "command", "original", workspace=workspace)
    changed, _ = await _submit(service, operator, "command", "changed", workspace=changed_workspace)
    assert original.spec.workspace != changed.spec.workspace
    assert original.spec.workspace.sha256 == changed.spec.workspace.sha256
    with pytest.raises(ConflictError, match="different request"):
        await _submit(service, operator, "command", "original", workspace=changed_workspace)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["callable", "command"])
async def test_identical_replay_survives_pool_template_change_and_disable(route: str) -> None:
    service, repository, operator, admin, pool = await _setup()
    original, replayed = await _submit(service, operator, route, "original", **_CHECKPOINT)
    assert not replayed
    changed_pool = pool.model_copy(update={"provider_options": {"runtime_version": "new-runtime"}})
    assert changed_pool.template_fingerprint != pool.template_fingerprint
    await service.upsert_capacity_pool(admin, changed_pool, idempotency_key="pool-changed")
    await service.disable_capacity_pool(admin, pool.id, idempotency_key="pool-disabled")
    replay, replayed = await _submit(service, operator, route, "original", **_CHECKPOINT)
    assert replayed and replay.id == original.id and replay.spec == original.spec
    async with repository.transaction() as tx:
        assert len(await tx.list_workloads()) == 1
