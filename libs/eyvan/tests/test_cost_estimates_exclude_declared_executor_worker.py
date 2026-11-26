# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cost estimates exclude declared executor and worker-width incompatibilities."""

from __future__ import annotations

import pytest

from eyvan.records import (
    CapacityPoolRecord,
    CommandExecution,
    Principal,
    ResourceGang,
    ResourceLimits,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService


def _pool(pool_id: str, *, executor: str = "eyvan-native", backend: str = "gcp-tpu") -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name="research",
        node_type="v5p-8",
        provider="gcp",
        capacity_backend=backend,
        provider_project="gcp-project-a",
        region="us-east5",
        zone="us-east5-a",
        capacity=ResourceLimits(tpu=8),
        allocatable=ResourceLimits(tpu=8),
        unit_prices={"tpu": 2.0},
        accelerator_type="v5p-8",
        executor=executor,
        agent_endpoint="http://agent:8080",
    )


def _spec(*, workers: int = 2, executor: str | None = "eyvan-native") -> WorkloadDefinition:
    return WorkloadDefinition(
        name="train",
        execution=CommandExecution(kind="command", command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(tpu=8)),
        accelerator_type="v5p-8",
        provider="gcp",
        region="us-east5",
        required_executor=executor,
        worker_count=workers,
        estimated_duration_seconds=3600,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "executor", "include_compatible"),
    [
        ("gcp-tpu", "process", False),
        ("gcp-tpu", "process", True),
        ("gcp-compute", "eyvan-native", False),
        ("gcp-compute", "eyvan-native", True),
        ("manual", "eyvan-native", False),
        ("manual", "eyvan-native", True),
    ],
)
async def test_multislice_estimate_skips_incompatible_declared_shapes(
    backend: str, executor: str, include_compatible: bool
) -> None:
    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository=repository)
    admin = Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    reader = Principal(
        subject="alice", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),)
    )
    pools = [_pool("research/incompatible", executor=executor, backend=backend)]
    if include_compatible:
        pools.append(_pool("research/compatible"))
    try:
        for pool in pools:
            stored, replayed = await service.upsert_capacity_pool(admin, pool, idempotency_key=pool.id)
            assert stored.executor == pool.executor
            assert not replayed
        async with repository.transaction() as tx:
            assert await tx.list_cells() == [], "an estimate must not require any live capacity"
            before_events = await tx.list_events()

        estimate = await service.estimate_workload(reader, org_id="acme", project_id="research", spec=_spec())

        expected = ["research/compatible"] if include_compatible else []
        assert [candidate.pool_id for candidate in estimate.candidates] == expected
        assert estimate.unpriced_pool_ids == ()
        if include_compatible:
            candidate = estimate.candidates[0]
            assert candidate.cells == 1
            assert candidate.hourly_cost == 32.0
            assert candidate.estimated_cost == 32.0
        async with repository.transaction() as tx:
            assert await tx.list_cells() == []
            assert await tx.list_workloads(org_id="acme", project_id="research") == []
            assert await tx.list_leases(org_id="acme", project_id="research") == []
            assert await tx.list_events() == before_events
    finally:
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("required_executor", ["eyvan-native", "process", None])
async def test_single_worker_estimate_respects_optional_required_executor(required_executor: str | None) -> None:
    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository=repository)
    admin = Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    reader = Principal(
        subject="alice", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),)
    )
    try:
        for executor in ("eyvan-native", "process"):
            pool = _pool(f"research/{executor}", executor=executor)
            await service.upsert_capacity_pool(admin, pool, idempotency_key=pool.id)

        estimate = await service.estimate_workload(
            reader, org_id="acme", project_id="research", spec=_spec(workers=1, executor=required_executor)
        )

        expected = (
            ["research/eyvan-native", "research/process"]
            if required_executor is None
            else [f"research/{required_executor}"]
        )
        assert [candidate.pool_id for candidate in estimate.candidates] == expected
        assert all(candidate.hourly_cost == 16.0 for candidate in estimate.candidates)
    finally:
        await repository.close()
