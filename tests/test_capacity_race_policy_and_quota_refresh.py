# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Provisioning races obey current eligibility; retained pools share fresh quota."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
import yaml
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan._internal.compat import UTC
from eyvan.capacity.autoscale import AutoscaleConfigCache, AutoscaleDefaults, AutoscaleGenerationService
from eyvan.capacity.planning import CapacityPlanner, CapacitySimulation
from eyvan.capacity.providers.gcp.quota_discovery import TpuQuotaAvailability
from eyvan.config import EyvanSettings
from eyvan.records import (
    CapacityClass,
    CapacityPoolRecord,
    CellState,
    CommandExecution,
    OutboxState,
    PolicyRecord,
    Principal,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService


def _principal(role: Role) -> Principal:
    return Principal(subject=role.value, grants=(RoleGrant(role=role, org_id="acme", project_id="research"),))


def _pool(name: str, region: str = "us-east5", **updates) -> CapacityPoolRecord:
    return CapacityPoolRecord.model_validate(
        {
            "id": f"research/{name}",
            "fleet_name": "research",
            "node_type": name,
            "provider": "gcp",
            "capacity_backend": "gcp-tpu",
            "provider_project": "offline-project",
            "region": region,
            "zone": f"{region}-a",
            "capacity": {"cpu": 32, "tpu": 8},
            "allocatable": {"cpu": 32, "tpu": 8},
            "accelerator_type": "v6e-16",
            "capacity_class": "spot",
            "executor": "eyvan-native",
            "agent_endpoint": "http://agent.invalid:8090",
            "max_cells": 8,
            "launch_batch": 8,
            **updates,
        }
    )


def _spec(primary: CapacityPoolRecord, alternates: tuple[CapacityPoolRecord, ...], **updates) -> WorkloadDefinition:
    return WorkloadDefinition.model_validate(
        {
            "name": "offline-racer",
            "execution": CommandExecution(command="true"),
            "gang": {"replicas": 1, "per_replica": {"cpu": 32, "memory_gib": 0, "tpu": 8}},
            "provider": "gcp",
            "capacity_class": "spot",
            "required_executor": "eyvan-native",
            "required_labels": {
                "eyvan_pool": primary.id,
                "eyvan_pool_alternates": ",".join(pool.id for pool in alternates),
            },
            **updates,
        }
    )


@pytest.mark.parametrize(
    "restriction", ["policy-region", "workload-region", "disabled", "spot", "resources", "executor"]
)
@pytest.mark.parametrize("acquiring", [False, True])
def test_race_skips_ineligible_alternate_and_uses_next_eligible_slot(restriction: str, acquiring: bool) -> None:
    primary = _pool("primary")
    forbidden = _pool("forbidden", "europe-west4")
    allowed = _pool("allowed")
    policies = ()
    spec_updates = {}
    if restriction == "policy-region":
        policies = (
            PolicyRecord(
                org_id="acme", project_id="research", allowed_regions=("us-east5",), updated_by="project-admin"
            ),
        )
    elif restriction == "workload-region":
        spec_updates = {"allowed_regions": ("us-east5",)}
    elif restriction == "disabled":
        forbidden = forbidden.model_copy(update={"enabled": False})
    elif restriction == "spot":
        forbidden = forbidden.model_copy(update={"capacity_class": CapacityClass.ON_DEMAND})
    elif restriction == "resources":
        forbidden = _pool("forbidden", "europe-west4", capacity={"cpu": 16, "tpu": 8}, allocatable={"cpu": 16, "tpu": 8})
    elif restriction == "executor":
        forbidden = forbidden.model_copy(update={"executor": "native-local"})
    pools = (primary, forbidden, allowed)
    workload = WorkloadRecord(
        id="racer",
        org_id="acme",
        project_id="research",
        created_by="operator",
        spec=_spec(primary, (forbidden, allowed), **spec_updates),
    )
    cells = (
        (
            primary.make_cell("acquiring").model_copy(
                update={"state": CellState.REGISTERING, "autoscale_request_workload_id": workload.id}
            ),
        )
        if acquiring
        else ()
    )
    now = utcnow()
    simulation = CapacitySimulation.build(pools, cells, (workload,), (), now, policies=policies)
    assert forbidden.id not in {
        simulation.pool_by_cell_id[cell_id] for cell_id in simulation.virtual_worker_count_by_cell_id
    }
    scheduled = CoreSchedulingService().schedule((workload,), simulation.cells, policies=policies)
    assert len(scheduled.placements) == 1
    assert {simulation.pool_by_cell_id[lease.cell_id] for lease in scheduled.placements[0].all_leases} == {primary.id}
    demands = simulation.demand(pools, (), scheduled, (workload,), now, race_width=2)
    plans = CapacityPlanner(
        pools, {pool.id: [cell for cell in cells if cell.pool_id == pool.id] for pool in pools}, now
    ).plan(demands)
    assert demands[forbidden.id].virtual_workload_ids == (), "ineligible alternate must not become provisioning demand"
    assert plans[forbidden.id].launch_worker_counts == ()
    assert plans[allowed.id].launch_workload_ids == (workload.id,), "skipped alternate must not consume a race slot"


@pytest.fixture(params=("memory", "postgres"))
async def repository(request):
    if request.param == "memory":
        yield InMemoryRepository()
        return
    # The session fixture starts a disposable loopback PostgreSQL, never a provider.
    url = request.getfixturevalue("postgres_url")
    schema = f"capacity_candidate_{uuid4().hex}"
    admin_engine = create_async_engine(url)
    async with admin_engine.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    try:
        await create_schema(engine)
        repository = SqlAlchemyRepository(engine)
        await repository.open()
        yield repository
    finally:
        await engine.dispose()
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin_engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("race_width", [1, 2])
@pytest.mark.parametrize("tighten_after_submission", [False, True])
async def test_current_policy_prevents_forbidden_durable_cell_and_apply_intent(
    repository,
    race_width: int,
    tighten_after_submission: bool,
) -> None:
    primary, forbidden = _pool("primary"), _pool("forbidden", "europe-west4")
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(
        repository,
        service,
        settings=replace(EyvanSettings.from_env(), capacity_race_width=race_width),
    )
    await service.apply_fleet(_principal(Role.FLEET_ADMIN), "research", (primary, forbidden), idempotency_key="fleet")
    policy = PolicyRecord(
        org_id="acme", project_id="research", allowed_regions=("us-east5",), updated_by="project-admin"
    )
    await service.put_policy(
        _principal(Role.PROJECT_ADMIN),
        policy.model_copy(update={"allowed_regions": ()}) if tighten_after_submission else policy,
        idempotency_key="initial-policy",
    )
    workload, _ = await service.submit_workload(
        _principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=_spec(primary, (forbidden,)),
        idempotency_key="workload",
    )
    if tighten_after_submission:
        await service.put_policy(_principal(Role.PROJECT_ADMIN), policy, idempotency_key="tightened-policy")
    outcome = await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
        outbox = await transaction.list_outbox(state=OutboxState.PENDING)
        persisted = await transaction.get_workload(workload.id)
        leases = await transaction.list_leases()
    applies = [record for record in outbox if record.topic == "capacity.apply"]
    assert [(cell.pool_id, cell.region, cell.state, cell.autoscale_request_workload_id) for cell in cells] == [
        (primary.id, "us-east5", CellState.REGISTERING, workload.id)
    ], "policy must prevent durable forbidden provisioning, not just lease placement"
    assert {record.aggregate_id for record in applies} == {cells[0].id}
    assert len(applies) == 1
    assert outcome == {"launched": 1, "drained": 0, "deleted": 0}
    assert persisted is not None and persisted.state is WorkloadState.PENDING
    assert leases == []


class _Discovery:
    def __init__(self, budget: int | None, *, unrelated: bool = True, usage_known: bool = True):
        self.rows = tuple(
            TpuQuotaAvailability(
                zone="us-east5-a",
                accelerator_type=flavor,
                generation=flavor.partition("-")[0],
                quota_tpus=budget,
                available_tpus=budget if usage_known else None,
                preemptible=True,
                metric=f"offline-{flavor.partition('-')[0]}",
                usage_observed_at=datetime.now(UTC) if budget is not None and usage_known else None,
            )
            for flavor in (("v6e-16", "v6e-32", "v5p-16") if unrelated else ("v6e-16", "v6e-32"))
        )
        # All observations in a shared dimension must have identical provenance.
        self.rows = tuple(replace(row, usage_observed_at=self.rows[0].usage_observed_at) for row in self.rows)

    def discover(
        self,
        *,
        project: str,
        zones: Iterable[str] = (),
        generations: Iterable[str] = (),
        accelerator_types: Iterable[str] = (),
        preemptible: bool = True,
    ) -> tuple[TpuQuotaAvailability, ...]:
        assert project == "offline-project" and preemptible
        return tuple(row for row in self.rows if not tuple(generations) or row.generation in generations)


@pytest.mark.parametrize("new_budget", [32, 128])
def test_denial_refresh_persists_current_group_quota_and_retains_identity(tmp_path, new_budget: int) -> None:
    root = tmp_path / "cache"
    cache = AutoscaleConfigCache(root)
    defaults = AutoscaleDefaults(project="offline-project", fleet_prefix="research")
    original = AutoscaleGenerationService(_Discovery(64)).generate(cache=cache, defaults=defaults).entries[0]
    old_other = original.config.pools["v5p-16"].model_dump()
    denial = cache.record_queue_admission_denial(
        project="offline-project",
        zone="us-east5-a",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
        observed_at=datetime.now(UTC) - timedelta(hours=1),
    )
    old_denial_bytes = cache._queue_admission_denials.path.read_bytes()
    service = AutoscaleGenerationService(_Discovery(new_budget, unrelated=False))
    preview = service.generate(cache=cache, defaults=defaults, generations=("v6e",), persist=False)
    assert preview.entries[0].config.pools["v6e-32"].quota_limit == new_budget
    assert cache.list()[0].config.pools["v6e-32"].quota_limit == 64
    result = service.generate(cache=cache, defaults=defaults, generations=("v6e",))
    assert result.entries[0].config.pools["v6e-32"].quota_limit == new_budget
    reloaded = AutoscaleConfigCache(root)
    stored = reloaded.list()[0]
    assert stored.path == original.path and stored.config.name == original.config.name
    assert set(stored.config.pools) == {"v6e-16", "v6e-32", "v5p-16"}
    assert {stored.config.pools[name].quota_limit for name in ("v6e-16", "v6e-32")} == {new_budget}
    assert stored.config.pools["v5p-16"].model_dump() == old_other
    assert reloaded.list_queue_admission_denials() == (denial,)
    assert reloaded._queue_admission_denials.path.read_bytes() == old_denial_bytes
    assert set(reloaded.list_eligible()[0].config.pools) == {"v6e-32", "v5p-16"}
    assert reloaded.list_eligible()[0].config.pools["v6e-32"].quota_limit == new_budget
    quota = next(quota for quota in stored.generation_quotas if quota.generation == "v6e")
    assert quota.quota_tpus == quota.available_tpus == new_budget
    assert reloaded.clear_queue_admission_denials(denial.key) == 1
    recovered = AutoscaleConfigCache(root).list_eligible()[0]
    assert recovered.config.pools["v6e-16"].quota_limit == new_budget
    assert recovered.config.pools["v6e-16"].quota_units_per_cell == original.config.pools["v6e-16"].quota_units_per_cell
    raw = yaml.safe_load(stored.path.read_text())
    assert {raw["pools"][name]["quota_limit"] for name in ("v6e-16", "v6e-32")} == {new_budget}


def test_unlimited_quota_is_not_manufactured_as_zero_or_finite(tmp_path) -> None:
    cache = AutoscaleConfigCache(tmp_path / "cache")
    defaults = AutoscaleDefaults(project="offline-project")
    generated = AutoscaleGenerationService(_Discovery(None)).generate(cache=cache, defaults=defaults)
    assert generated.has_entries
    assert all(pool.quota_limit is None for pool in cache.list()[0].config.pools.values())
    assert all(quota.available_tpus is None and quota.quota_tpus is None for quota in cache.list()[0].generation_quotas)


def test_unknown_live_balance_remains_unknown_and_cannot_generate_eligible_pools(tmp_path) -> None:
    cache = AutoscaleConfigCache(tmp_path / "cache")
    discovery = _Discovery(64, usage_known=False)
    result = AutoscaleGenerationService(discovery).generate(
        cache=cache, defaults=AutoscaleDefaults(project="offline-project")
    )
    assert all(row.available_tpus is None and row.usage_observed_at is None for row in discovery.rows)
    assert not result.has_entries
    assert all(candidate.reason.name == "LIVE_SPOT_QUOTA_USAGE_UNAVAILABLE" for candidate in result.skipped)
    assert cache.list() == cache.list_eligible() == ()


def test_retained_pool_floor_validation_fails_before_yaml_or_index_write(tmp_path) -> None:
    cache = AutoscaleConfigCache(tmp_path / "cache")
    defaults = AutoscaleDefaults(project="offline-project", min_workers=1)
    original = (
        AutoscaleGenerationService(_Discovery(64, unrelated=False)).generate(cache=cache, defaults=defaults).entries[0]
    )
    index_path = cache.root / "index.json"
    old_yaml, old_index = original.path.read_bytes(), index_path.read_bytes()
    cache.record_queue_admission_denial(
        project="offline-project",
        zone="us-east5-a",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
    )
    # Fresh 32 fits the eligible flavor's floor, but not both retained floors.
    with pytest.raises(ValueError):
        AutoscaleGenerationService(_Discovery(32, unrelated=False)).generate(cache=cache, defaults=defaults)
    assert original.path.read_bytes() == old_yaml
    assert index_path.read_bytes() == old_index


def test_eligible_race_still_obeys_shared_provider_quota() -> None:
    primary, alternate = (
        _pool(name, quota_group="shared", quota_limit=16, quota_units_per_cell=16) for name in ("primary", "alternate")
    )
    pools = (primary, alternate)
    workload = WorkloadRecord(
        id="quota-racer",
        org_id="acme",
        project_id="research",
        created_by="operator",
        spec=_spec(primary, (alternate,)),
    )
    now = utcnow()
    simulation = CapacitySimulation.build(pools, (), (workload,), (), now)
    scheduled = CoreSchedulingService().schedule((workload,), simulation.cells)
    demands = simulation.demand(pools, (), scheduled, (workload,), now, race_width=2)
    assert all(demand.virtual_workload_ids == (workload.id,) for demand in demands.values())
    plans = CapacityPlanner(pools, {pool.id: [] for pool in pools}, now).plan(demands)
    assert sum(sum(plan.launch_worker_counts) * 16 for plan in plans.values()) == 16
