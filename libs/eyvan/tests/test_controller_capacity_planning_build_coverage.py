# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression contracts for controller capacity planning and build coverage."""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan._internal.build_identity import build_identity
from eyvan.capacity.providers.kubernetes.api import KUEUE_API, ApiObservation
from eyvan.records import BudgetRecord, CellState, CommandExecution, ResourceLimits, WorkloadDefinition, utcnow
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.federation import BudgetGrantRecord, QuotaGrantRecord
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.worker.agent.client import CellAgentUnavailableError
from tests.test_autoscale_pool_capacity import _cluster_queue, _ClusterApi, _controller, _pass
from tests.test_capacity_autoscaler import cpu_pool, fleet_admin, operator
from tests.test_fleet_builds import _Agent, _agent_build, _cell, _pool, _run_builds, _sweep


@pytest.mark.asyncio
@pytest.mark.parametrize("reserved,first_launches", [("32", 0), ("24", 2)])
async def test_withheld_initial_capacity_remains_requested(reserved: str, first_launches: int) -> None:
    api = _ClusterApi(cluster_queue=_cluster_queue(nominal="32", reserved=reserved))
    controller, repository = await _controller(_pass(api))

    first = await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        pool = await transaction.get_capacity_pool("research/k8s")
        cells = await transaction.list_cells()
    assert first["launched"] == first_launches
    assert len(cells) == first_launches
    assert pool is not None and pool.initialized_at is None

    api.observations[f"{KUEUE_API}/clusterqueues/research-cq"] = ApiObservation(
        payload=_cluster_queue(nominal="32", reserved="0"), readable=True
    )
    second = await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        pool = await transaction.get_capacity_pool("research/k8s")
        cells = await transaction.list_cells()
    assert second["launched"] == 4 - first_launches
    assert sum(cell.worker_count for cell in cells) == 4
    assert pool is not None and pool.initialized_at is not None


@pytest.mark.asyncio
async def test_skipped_initial_upserts_do_not_initialize_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    controller, repository = await _controller(_pass(_ClusterApi(cluster_queue=_cluster_queue())))

    async def superseded(*_args, **_kwargs):
        return None, False

    monkeypatch.setattr(controller, "_autoscale_upsert_cell", superseded)
    result = await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        pool = await transaction.get_capacity_pool("research/k8s")
        cells = await transaction.list_cells()
    assert result["launched"] == 0
    assert cells == []
    assert pool is not None and pool.initialized_at is None


@pytest.mark.asyncio
async def test_changed_pool_generation_skips_initial_upserts_and_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    controller, repository = await _controller(_pass(_ClusterApi(cluster_queue=_cluster_queue())))
    upsert = controller._autoscale_upsert_cell
    changed = False

    async def change_pool_before_upsert(pool, *args, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            await controller.upsert_capacity_pool(
                fleet_admin(), pool.model_copy(update={"min_cells": 1}), idempotency_key="new-pool-generation"
            )
        return await upsert(pool, *args, **kwargs)

    monkeypatch.setattr(controller, "_autoscale_upsert_cell", change_pool_before_upsert)
    first = await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        pool = await transaction.get_capacity_pool("research/k8s")
        cells = await transaction.list_cells()
    assert first["launched"] == 0 and cells == []
    assert pool is not None and pool.generation == 2 and pool.initialized_at is None
    second = await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        pool = await transaction.get_capacity_pool("research/k8s")
        cells = await transaction.list_cells()
    assert second["launched"] == 4 and len(cells) == 4
    assert pool is not None and pool.initialized_at is not None


@pytest.mark.asyncio
async def test_stale_shrink_does_not_drain_new_warm_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    pool = cpu_pool(idle_timeout_s=0)
    await controller.upsert_capacity_pool(fleet_admin(), pool, idempotency_key="pool-v1")
    cell = pool.make_cell("idle", worker_count=1).model_copy(update={"state": CellState.READY})
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
    persist = controller._persist_shrink_clocks

    async def change_floor_after_plan(pools, plans, now):
        await persist(pools, plans, now)
        await controller.upsert_capacity_pool(
            fleet_admin(), pool.model_copy(update={"min_cells": 1, "initial_cells": 1}), idempotency_key="pool-v2"
        )

    monkeypatch.setattr(controller, "_persist_shrink_clocks", change_floor_after_plan)
    result = await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        current_pool = await transaction.get_capacity_pool(pool.id)
        current_cell = await transaction.get_cell(cell.id)
        commands = await transaction.list_outbox()
    assert current_pool is not None and current_pool.generation == 2 and current_pool.min_cells == 1
    assert current_cell is not None and current_cell.state is CellState.READY
    assert result["drained"] == 0
    assert not any(command.topic == "capacity.drain" for command in commands)


async def _federated_controller():
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService(), plane_id="child")
    controller = EyvanController(repository, service)
    pool = cpu_pool(hourly_cpu_price=10, idle_timeout_s=0)
    await controller.upsert_capacity_pool(fleet_admin(), pool, idempotency_key="pool")
    async with repository.transaction() as transaction:
        await transaction.put_budget(BudgetRecord(org_id="acme", project_id="research", limit=1000, updated_by="root"))
    workload, _ = await service.submit_workload(
        operator(),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="pending",
            execution=CommandExecution(command="true"),
            estimated_duration_seconds=3600,
            required_labels={"eyvan_pool": pool.id},
        ),
        idempotency_key="workload",
    )
    return controller, repository, workload


async def _restrict_federation(repository: InMemoryRepository, kind: str, *, restrictive: bool = True) -> None:
    now = utcnow()
    common = dict(
        id="grant",
        org_id="acme",
        project_id="research",
        home_plane_id="home",
        grantee_plane_id="child",
        issued_at=now - timedelta(seconds=1),
        expires_at=now + timedelta(hours=1),
    )
    async with repository.transaction() as transaction:
        if kind == "budget":
            await transaction.put_budget_grant(BudgetGrantRecord(**common, amount=1 if restrictive else 100))
        else:
            await transaction.put_quota_grant(
                # Grants are complete authority vectors, not sparse local
                # quotas: the default workload also requests 1 GiB of memory.
                QuotaGrantRecord(**common, limits=ResourceLimits(cpu=0.5 if restrictive else 4, memory_gib=4))
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["budget", "quota"])
async def test_federation_refusal_does_not_launch_workload_capacity(kind: str) -> None:
    controller, repository, _workload = await _federated_controller()
    await _restrict_federation(repository, kind)
    result = await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert result["launched"] == 0
    assert cells == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["budget", "quota"])
async def test_tightened_federation_releases_unusable_pending_reservation(kind: str) -> None:
    controller, repository, workload = await _federated_controller()
    first = await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert first["launched"] == 1
    assert len(cells) == 1 and cells[0].autoscale_request_workload_id == workload.id
    await _restrict_federation(repository, kind)
    result = await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        cell = await transaction.get_cell(cells[0].id)
        commands = await transaction.list_outbox()
    assert result["drained"] == 1
    assert cell is not None and cell.autoscale_request_workload_id is None and cell.state is CellState.DRAINING
    assert any(command.topic == "capacity.drain" for command in commands)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["budget", "quota"])
async def test_permissive_federation_retains_waiting_workload_reservation(kind: str) -> None:
    controller, repository, workload = await _federated_controller()
    await _restrict_federation(repository, kind, restrictive=False)
    assert (await controller.reconcile_capacity_pools_once())["launched"] == 1
    second = await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert second == {"launched": 0, "drained": 0, "deleted": 0}
    assert len(cells) == 1 and cells[0].autoscale_request_workload_id == workload.id


@pytest.mark.asyncio
@pytest.mark.parametrize("with_skew", [False, True])
async def test_unreachable_agent_prevents_agreement_but_not_proven_skew(
    monkeypatch: pytest.MonkeyPatch,
    with_skew: bool,
) -> None:
    pool = _pool()
    cells = [
        _cell("cell-a", pool=pool, endpoint="http://agent-a:8090"),
        _cell("cell-b", pool=pool, endpoint="http://agent-b:8090"),
    ]
    agents = {
        "http://agent-a:8090": _Agent(_agent_build(build_identity().source_digest)),
        "http://agent-b:8090": _Agent(CellAgentUnavailableError("unavailable")),
    }
    if with_skew:
        cells.append(_cell("cell-c", pool=pool, endpoint="http://agent-c:8090"))
        agents["http://agent-c:8090"] = _Agent(_agent_build("sha256:stale"))
    report = await _sweep(agents, cells=cells, pools=[pool])
    assert report["unreachable"] == [{"endpoint": "http://agent-b:8090", "detail": "unavailable"}]
    assert report["in_agreement"] is (False if with_skew else None)
    result, _client = _run_builds(monkeypatch, report)
    assert result.exit_code != 0, result.output


@pytest.mark.asyncio
@pytest.mark.parametrize("second_workers", ["stale", "unreadable", "matching"])
async def test_shared_agent_probes_workers_of_every_live_cell(
    monkeypatch: pytest.MonkeyPatch,
    second_workers: str,
) -> None:
    pool_a = _pool("research/a")
    pool_b = _pool("research/b")
    endpoint = pool_a.agent_endpoint
    cells = [
        _cell("cell-a", pool=pool_a, endpoint=endpoint),
        _cell("cell-b", pool=pool_b, endpoint=endpoint),
        _cell("cell-deleted", pool=pool_b, endpoint=endpoint, state=CellState.DELETED),
    ]
    plane_digest = build_identity().source_digest

    class SharedAgent(_Agent):
        def __init__(self):
            super().__init__(_agent_build(plane_digest))
            self.build_calls = 0
            self.worker_ids = []

        def build(self):
            self.build_calls += 1
            return super().build()

        def worker_builds(self, cell_id):
            self.worker_ids.append(cell_id)
            if cell_id == "cell-b" and second_workers == "unreadable":
                raise CellAgentUnavailableError("worker unavailable")
            digest = "sha256:stale" if cell_id == "cell-b" and second_workers == "stale" else plane_digest
            return [{"rank": 0, "build": {"source_digest": digest}}]

    agent = SharedAgent()
    report = await _sweep({endpoint: agent}, cells=cells, pools=[pool_a, pool_b])
    expected = {"stale": False, "unreadable": None, "matching": True}[second_workers]
    assert report["in_agreement"] is expected
    assert {entry["cell_id"] for entry in report["agents"]} == {"cell-a", "cell-b"}
    assert {entry["pool_id"] for entry in report["agents"]} == {pool_a.id, pool_b.id}
    assert report["unscanned_pools"] == []
    assert agent.build_calls == 1
    assert agent.worker_ids == ["cell-a", "cell-b"]
    result, _client = _run_builds(monkeypatch, report)
    assert (result.exit_code == 0) is (expected is True), result.output
