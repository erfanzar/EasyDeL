# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import asyncio
import importlib
import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from eyvan._internal.json_types import JsonObject, model_json
from eyvan.records import (
    AuditEvent,
    BudgetRecord,
    CapacityPoolRecord,
    CellRecord,
    CommandExecution,
    IdempotencyRecord,
    LeaseRecord,
    LeaseState,
    OutboxRecord,
    OutboxState,
    PolicyRecord,
    QuotaRecord,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.persistence import SCHEMA_REVISION, Base, InMemoryRepository, SqlAlchemySession
from eyvan.server.persistence.base import TransientRepositoryError
from eyvan.server.persistence.sqlalchemy import CellRow, OutboxRow, SqlAlchemyRepository, create_schema


def _legacy_multislice_cell() -> JsonObject:
    return {
        "id": "legacy-v5p-group",
        "provider": "gcp",
        "capacity_backend": "gcp-tpu",
        "provider_project": "project-a",
        "region": "us-east5",
        "zone": "us-east5-a",
        "capacity": {"tpu": 16},
        "allocatable": {"tpu": 16},
        "accelerator_type": "v5p-16",
        "executor": "eyvan-native",
        "provider_options": {"node_count": 2, "node_prefix": "legacy-v5p-group-node"},
    }


def _workload(workload_id: str = "wl-1") -> WorkloadRecord:
    return WorkloadRecord(
        id=workload_id,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
        created_by="alice",
    )


def _workload_record(workload_id: str, org_id: str, project_id: str, state: WorkloadState) -> WorkloadRecord:
    return WorkloadRecord(
        id=workload_id,
        org_id=org_id,
        project_id=project_id,
        state=state,
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
        created_by="alice",
    )


def _lease_record(
    lease_id: str,
    org_id: str,
    project_id: str,
    workload_id: str,
    state: LeaseState = LeaseState.ACTIVE,
) -> LeaseRecord:
    return LeaseRecord(
        id=lease_id,
        workload_id=workload_id,
        cell_id=f"cell-{lease_id}",
        org_id=org_id,
        project_id=project_id,
        state=state,
        resources=_workload_record(workload_id, org_id, project_id, WorkloadState.LEASED).spec.gang,
        fencing_token=f"token-{lease_id}",
        expires_at=utcnow() + timedelta(minutes=5),
    )


def _pool_record(pool_id: str, fleet_name: str, node_type: str = "v5p-16") -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name=fleet_name,
        node_type=node_type,
        provider="gcp",
        provider_project="gcp-project-a",
        region="us-east5",
        zone="us-east5-a",
        capacity=ResourceVector(tpu=8),
        allocatable=ResourceVector(tpu=8),
        agent_endpoint="http://agent:8090",
    )


def _policy_record(org_id: str, project_id: str) -> PolicyRecord:
    return PolicyRecord(org_id=org_id, project_id=project_id, updated_by="admin")


def _budget_record(org_id: str, project_id: str) -> BudgetRecord:
    return BudgetRecord(org_id=org_id, project_id=project_id, limit=100.0, updated_by="admin")


def _quota_record(org_id: str, project_id: str | None, principal_id: str | None) -> QuotaRecord:
    return QuotaRecord(
        org_id=org_id,
        project_id=project_id,
        principal_id=principal_id,
        limits=ResourceVector(tpu=8),
        updated_by="admin",
    )


def _migration_module():
    return importlib.import_module("eyvan.server.persistence.migrations.versions.0004_query_indexes")


_EXPECTED_FILTER_COLUMNS: dict[str, tuple[tuple[str, int], ...]] = {
    "workloads": (("org_id", 128), ("project_id", 128), ("state", 32)),
    "cells": (("state", 32), ("pool_id", 192)),
    "capacity_pools": (("fleet_name", 96),),
    "leases": (("org_id", 128), ("project_id", 128), ("workload_id", 96), ("state", 32)),
    "policies": (("org_id", 128),),
    "quotas": (("org_id", 128), ("project_id", 128), ("principal_id", 128)),
    "budgets": (("org_id", 128),),
}


async def _sql_list(method: str, rows: list, filters: dict):
    scalars_result = Mock()
    scalars_result.all.return_value = list(rows)
    database_session = AsyncMock(spec=AsyncSession)
    database_session.scalars.return_value = scalars_result
    result = await getattr(SqlAlchemySession(database_session), method)(**filters)
    statement = database_session.scalars.await_args.args[0]
    compiled = str(statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    return result, compiled


async def _sql_rows_for_records(put_method: str, records: list) -> list:
    write_session = AsyncMock(spec=AsyncSession)
    writer = SqlAlchemySession(write_session)
    for record in records:
        await getattr(writer, put_method)(record)
    return [call.args[0] for call in write_session.merge.await_args_list]


@pytest.mark.asyncio
async def test_transaction_rolls_back_records_and_events() -> None:
    repository = InMemoryRepository()

    with pytest.raises(RuntimeError, match="rollback"):
        async with repository.transaction() as transaction:
            await transaction.put_workload(_workload())
            await transaction.append_event(
                AuditEvent(
                    id="evt-1",
                    actor="alice",
                    org_id="acme",
                    project_id="research",
                    entity_type="workload",
                    entity_id="wl-1",
                    action="submitted",
                )
            )
            raise RuntimeError("rollback")

    async with repository.transaction() as transaction:
        assert await transaction.get_workload("wl-1") is None
        assert await transaction.list_events() == []


@pytest.mark.asyncio
async def test_in_memory_repository_roundtrips_an_upgraded_legacy_multislice_cell() -> None:
    repository = InMemoryRepository()
    upgraded = CellRecord.model_validate(_legacy_multislice_cell())

    async with repository.transaction() as transaction:
        await transaction.put_cell(upgraded)

    async with repository.transaction() as transaction:
        restored = await transaction.get_cell(upgraded.id)

    assert restored is not None
    assert restored.worker_count == 2
    assert restored.model_dump(mode="json")["worker_count"] == 2


@pytest.mark.asyncio
async def test_progress_queries_select_exact_autoscale_cells_and_all_current_commands() -> None:
    repository = InMemoryRepository()
    requested = CellRecord.model_validate({**_legacy_multislice_cell(), "pool_id": "research-v5p-16"}).model_copy(
        update={"id": "requested-cell", "autoscale_request_workload_id": "wl-request"}
    )
    other = requested.model_copy(update={"id": "other-cell", "autoscale_request_workload_id": "wl-other"})

    async with repository.transaction() as transaction:
        await transaction.put_cell(other)
        await transaction.put_cell(requested)
        for index in range(101):
            aggregate_id = f"cell-{index:03d}"
            await transaction.put_outbox(
                OutboxRecord(
                    id=f"current-{index:03d}",
                    topic="capacity.apply",
                    aggregate_type="cell",
                    aggregate_id=aggregate_id,
                    idempotency_key=f"current-key-{index:03d}",
                    generation=2,
                )
            )
            await transaction.put_outbox(
                OutboxRecord(
                    id=f"stale-{index:03d}",
                    topic="capacity.apply",
                    aggregate_type="cell",
                    aggregate_id=aggregate_id,
                    idempotency_key=f"stale-key-{index:03d}",
                    generation=1,
                )
            )
        await transaction.put_outbox(
            OutboxRecord(
                id="release-current",
                topic="capacity.release",
                aggregate_type="cell",
                aggregate_id="cell-000",
                idempotency_key="release-current-key",
                generation=2,
            )
        )

    async with repository.transaction() as transaction:
        cells = await transaction.list_cells_for_autoscale_request("wl-request")
        commands = await transaction.list_outbox_for_aggregates(
            "cell",
            {f"cell-{index:03d}": 2 for index in range(101)},
            topic="capacity.apply",
        )

    assert [cell.id for cell in cells] == ["requested-cell"]
    assert len(commands) == 101
    assert {command.id for command in commands} == {f"current-{index:03d}" for index in range(101)}


@pytest.mark.asyncio
async def test_sql_progress_queries_filter_json_fields_and_current_generations() -> None:
    requested = CellRecord.model_validate({**_legacy_multislice_cell(), "pool_id": "research-v5p-16"}).model_copy(
        update={"id": "requested-cell", "autoscale_request_workload_id": "wl-request"}
    )
    current = OutboxRecord(
        id="current-command",
        topic="capacity.apply",
        aggregate_type="cell",
        aggregate_id="requested-cell",
        idempotency_key="current-command-key",
        generation=3,
    )
    stale = current.model_copy(update={"id": "stale-command", "generation": 2})
    cell_rows = Mock()
    cell_rows.all.return_value = [CellRow(id=requested.id, data=model_json(requested))]
    command_rows = Mock()
    command_rows.all.return_value = [
        OutboxRow(
            id=current.id,
            state=current.state.value,
            available_at=current.available_at,
            claimed_until=current.claimed_until,
            data=model_json(current),
        ),
        OutboxRow(
            id=stale.id,
            state=stale.state.value,
            available_at=stale.available_at,
            claimed_until=stale.claimed_until,
            data=model_json(stale),
        ),
    ]
    database_session = AsyncMock(spec=AsyncSession)
    database_session.scalars.side_effect = [cell_rows, command_rows]
    session = SqlAlchemySession(database_session)

    cells = await session.list_cells_for_autoscale_request("wl-request")
    commands = await session.list_outbox_for_aggregates(
        "cell",
        {"requested-cell": 3},
        topic="capacity.apply",
    )

    assert [cell.id for cell in cells] == ["requested-cell"]
    assert [command.id for command in commands] == ["current-command"]
    cell_statement = database_session.scalars.await_args_list[0].args[0]
    outbox_statement = database_session.scalars.await_args_list[1].args[0]
    cell_sql = str(cell_statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    outbox_sql = str(outbox_statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    assert "autoscale_request_workload_id" in cell_sql
    assert "'wl-request'" in cell_sql
    assert "aggregate_type" in outbox_sql
    assert "aggregate_id" in outbox_sql
    assert "topic" in outbox_sql
    assert "'capacity.apply'" in outbox_sql
    assert "LIMIT" not in outbox_sql


@pytest.mark.asyncio
async def test_sql_session_loads_a_legacy_multislice_cell_after_restart() -> None:
    database_session = AsyncMock(spec=AsyncSession)
    database_session.get.return_value = CellRow(id="legacy-v5p-group", data=_legacy_multislice_cell())

    restored = await SqlAlchemySession(database_session).get_cell("legacy-v5p-group")

    assert restored is not None
    assert restored.worker_count == 2
    assert restored.provider_options["node_count"] == 2


@pytest.mark.asyncio
async def test_event_wait_observes_only_committed_events() -> None:
    repository = InMemoryRepository()
    waiter = asyncio.create_task(repository.wait_for_events(after_sequence=0, timeout=1))
    await asyncio.sleep(0)

    async with repository.transaction() as transaction:
        await transaction.append_event(
            AuditEvent(
                id="evt-1",
                actor="alice",
                org_id="acme",
                project_id="research",
                entity_type="workload",
                entity_id="wl-1",
                action="submitted",
            )
        )
        assert not waiter.done()

    events = await waiter
    assert [(event.sequence, event.action) for event in events] == [(1, "submitted")]


@pytest.mark.asyncio
async def test_event_wait_returns_empty_after_timeout() -> None:
    repository = InMemoryRepository()

    events = await repository.wait_for_events(after_sequence=0, timeout=0.001)

    assert events == []


@pytest.mark.asyncio
async def test_quota_crud_uses_exact_hierarchical_scope() -> None:
    repository = InMemoryRepository()
    project_quota = QuotaRecord(
        org_id="acme",
        project_id="research",
        limits=ResourceVector(tpu=32),
        weight=2,
        updated_by="admin",
    )
    user_quota = project_quota.model_copy(update={"principal_id": "alice", "limits": ResourceVector(tpu=8)})

    async with repository.transaction() as transaction:
        await transaction.put_quota(project_quota)
        await transaction.put_quota(user_quota)

    async with repository.transaction() as transaction:
        assert await transaction.get_quota("acme", "research", None) == project_quota
        assert await transaction.get_quota("acme", "research", "alice") == user_quota
        assert len(await transaction.list_quotas(org_id="acme", project_id="research")) == 2
        assert await transaction.delete_quota("acme", "research", "alice")

    async with repository.transaction() as transaction:
        assert await transaction.get_quota("acme", "research", "alice") is None


@pytest.mark.asyncio
async def test_outbox_claims_are_atomic_and_reclaim_abandoned_work() -> None:
    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        for index in range(4):
            await transaction.put_outbox(
                OutboxRecord(
                    id=f"cmd-{index}",
                    topic="test",
                    aggregate_type="workload",
                    aggregate_id=f"wl-{index}",
                    idempotency_key=f"key-{index}",
                    generation=1,
                )
            )

    first, second = await asyncio.gather(
        repository.claim_outbox(worker_id="worker-a", limit=2),
        repository.claim_outbox(worker_id="worker-b", limit=2),
    )

    assert len(first) == len(second) == 2
    assert {record.id for record in first}.isdisjoint(record.id for record in second)
    assert {record.claimed_by for record in first + second} == {"worker-a", "worker-b"}


@pytest.mark.asyncio
async def test_outbox_claim_snapshot_never_includes_commands_added_after_the_snapshot() -> None:
    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        await transaction.put_outbox(
            OutboxRecord(
                id="initial",
                topic="test",
                aggregate_type="workload",
                aggregate_id="wl-initial",
                idempotency_key="key-initial",
                generation=1,
            )
        )

    snapshot = await repository.snapshot_claimable_outbox_ids(limit=10)

    async with repository.transaction() as transaction:
        await transaction.put_outbox(
            OutboxRecord(
                id="later",
                topic="test",
                aggregate_type="workload",
                aggregate_id="wl-later",
                idempotency_key="key-later",
                generation=1,
            )
        )

    claimed = await repository.claim_outbox_id(
        worker_id="worker",
        command_id=snapshot[0],
        lease_seconds=60,
    )

    assert snapshot == ("initial",)
    assert claimed is not None and claimed.id == "initial"
    async with repository.transaction() as transaction:
        later = await transaction.get_outbox("later")
    assert later is not None and later.state is OutboxState.PENDING


@pytest.mark.asyncio
async def test_outbox_claim_skips_unbounded_succeeded_history() -> None:
    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        for index in range(81):
            await transaction.put_outbox(
                OutboxRecord(
                    id=f"done-{index:03d}",
                    topic="test",
                    aggregate_type="workload",
                    aggregate_id=f"wl-{index}",
                    idempotency_key=f"done-key-{index}",
                    generation=1,
                    state=OutboxState.SUCCEEDED,
                )
            )
        await transaction.put_outbox(
            OutboxRecord(
                id="pending",
                topic="test",
                aggregate_type="workload",
                aggregate_id="wl-pending",
                idempotency_key="pending-key",
                generation=1,
            )
        )

    claimed = await repository.claim_outbox(worker_id="worker", limit=20)

    assert [record.id for record in claimed] == ["pending"]


@pytest.mark.asyncio
async def test_outbox_claim_reclaims_only_expired_processing_records() -> None:
    repository = InMemoryRepository()
    expired_at = utcnow() - timedelta(seconds=1)
    async with repository.transaction() as transaction:
        for record_id, state in (
            ("processing", OutboxState.PROCESSING),
            ("succeeded", OutboxState.SUCCEEDED),
            ("discarded", OutboxState.DISCARDED),
        ):
            await transaction.put_outbox(
                OutboxRecord(
                    id=record_id,
                    topic="test",
                    aggregate_type="workload",
                    aggregate_id=f"wl-{record_id}",
                    idempotency_key=f"key-{record_id}",
                    generation=1,
                    state=state,
                    claimed_by="expired-worker",
                    claimed_until=expired_at,
                )
            )

    claimed = await repository.claim_outbox(worker_id="new-worker", limit=10)

    assert [record.id for record in claimed] == ["processing"]
    assert claimed[0].claimed_by == "new-worker"
    async with repository.transaction() as transaction:
        succeeded = await transaction.get_outbox("succeeded")
        discarded = await transaction.get_outbox("discarded")
    assert succeeded is not None and succeeded.state is OutboxState.SUCCEEDED
    assert discarded is not None and discarded.state is OutboxState.DISCARDED


def test_sqlalchemy_metadata_matches_query_index_revision() -> None:
    assert "outbox" in Base.metadata.tables
    assert "capacity_pools" in Base.metadata.tables

    migration = _migration_module()
    assert migration.down_revision == "0003_capacity_pools"
    assert migration._FILTER_COLUMNS == _EXPECTED_FILTER_COLUMNS

    for table, columns in _EXPECTED_FILTER_COLUMNS.items():
        table_metadata = Base.metadata.tables[table]
        index_names = {index.name for index in table_metadata.indexes}
        for column, length in columns:
            assert column in table_metadata.c
            assert table_metadata.c[column].type.length == length
            assert f"ix_{table}_{column}" in index_names


def test_the_orm_stamps_the_head_of_the_migration_chain() -> None:
    """``create_schema`` writes ``SCHEMA_REVISION`` into ``schema_revision``.

    If that is not the chain head, a freshly created development database is
    stamped at a revision alembic believes still needs upgrading -- so the next
    ``eyvan-migrate upgrade`` replays migrations against tables that already have
    their final shape. Pinning a literal revision here instead just meant the
    assertion had to be edited on every migration, which is not a check.
    """

    import pkgutil

    import eyvan.server.persistence.migrations.versions as versions

    modules = [
        importlib.import_module(f"{versions.__name__}.{info.name}") for info in pkgutil.iter_modules(versions.__path__)
    ]
    revisions = {module.revision for module in modules}
    superseded: set[str] = set()
    for module in modules:
        down_revision = module.down_revision
        if isinstance(down_revision, tuple):
            superseded.update(down_revision)
        elif down_revision is not None:
            superseded.add(down_revision)
    heads = revisions - superseded

    assert heads == {SCHEMA_REVISION}


def test_retention_columns_are_indexed_and_present_in_the_orm() -> None:
    """Retention must be able to page on the column it filters.

    Without an indexed, typed copy of these timestamps the prune paged by one
    column and filtered by another inside the JSON blob, so expired rows behind
    a full LIMIT window were never reached -- see ``test_sql_retention``.
    """

    for table, column in (("outbox", "updated_at"), ("idempotency_keys", "created_at")):
        table_metadata = Base.metadata.tables[table]
        assert column in table_metadata.c
        assert {index.name for index in table_metadata.indexes} >= {f"ix_{table}_{column}"}


@pytest.mark.asyncio
async def test_sql_put_populates_typed_query_columns() -> None:
    database_session = AsyncMock(spec=AsyncSession)
    session = SqlAlchemySession(database_session)

    workload = _workload_record("wl-1", "acme", "research", WorkloadState.RUNNING)
    lease = _lease_record("lease-1", "acme", "research", "wl-1")
    cell = CellRecord.model_validate(
        {**_legacy_multislice_cell(), "id": "cell-1", "pool_id": "research/v5p-16", "state": "ready"}
    )
    pool = _pool_record("research/v5p-16", "research")
    policy = _policy_record("acme", "research")
    quota = _quota_record("acme", None, None)
    budget = _budget_record("acme", "research")

    await session.put_workload(workload)
    await session.put_lease(lease)
    await session.put_cell(cell)
    await session.put_capacity_pool(pool)
    await session.put_policy(policy)
    await session.put_quota(quota)
    await session.put_budget(budget)

    workload_row, lease_row, cell_row, pool_row, policy_row, quota_row, budget_row = [
        call.args[0] for call in database_session.merge.await_args_list
    ]

    assert (workload_row.org_id, workload_row.project_id, workload_row.state) == (
        "acme",
        "research",
        "running",
    )
    assert (lease_row.org_id, lease_row.project_id, lease_row.workload_id, lease_row.state) == (
        "acme",
        "research",
        "wl-1",
        "active",
    )
    assert (cell_row.state, cell_row.pool_id) == ("ready", "research/v5p-16")
    assert pool_row.fleet_name == "research"
    assert policy_row.org_id == "acme"
    assert (quota_row.org_id, quota_row.project_id, quota_row.principal_id) == ("acme", None, None)
    assert budget_row.org_id == "acme"


@pytest.mark.asyncio
async def test_sql_list_filters_match_reference_results_for_mixed_records() -> None:
    workloads = [
        _workload_record("wl-1", "acme", "research", WorkloadState.PENDING),
        _workload_record("wl-2", "acme", "research", WorkloadState.RUNNING),
        _workload_record("wl-3", "acme", "platform", WorkloadState.PENDING),
        _workload_record("wl-4", "globex", "research", WorkloadState.PENDING),
    ]
    leases = [
        _lease_record("lease-1", "acme", "research", "wl-1"),
        _lease_record("lease-2", "acme", "platform", "wl-3"),
        _lease_record("lease-3", "globex", "research", "wl-4"),
        _lease_record("lease-4", "acme", "research", "wl-2"),
    ]
    quotas = [
        _quota_record("acme", "research", "alice"),
        _quota_record("acme", "research", None),
        _quota_record("acme", None, None),
        _quota_record("globex", "research", "bob"),
    ]
    pools = [
        _pool_record("research/v5p-16", "research", "v5p-16"),
        _pool_record("research/v6e-4", "research", "v6e-4"),
        _pool_record("globex/v5p-8", "globex", "v5p-8"),
    ]
    policies = [
        _policy_record("acme", "research"),
        _policy_record("acme", "platform"),
        _policy_record("globex", "research"),
    ]
    budgets = [
        _budget_record("acme", "research"),
        _budget_record("acme", "platform"),
        _budget_record("globex", "research"),
    ]

    reference = InMemoryRepository()
    async with reference.transaction() as transaction:
        for workload in workloads:
            await transaction.put_workload(workload)
        for lease in leases:
            await transaction.put_lease(lease)
        for quota in quotas:
            await transaction.put_quota(quota)
        for pool in pools:
            await transaction.put_capacity_pool(pool)
        for policy in policies:
            await transaction.put_policy(policy)
        for budget in budgets:
            await transaction.put_budget(budget)

    workload_rows = await _sql_rows_for_records("put_workload", workloads)
    lease_rows = await _sql_rows_for_records("put_lease", leases)
    quota_rows = await _sql_rows_for_records("put_quota", quotas)
    pool_rows = await _sql_rows_for_records("put_capacity_pool", pools)
    policy_rows = await _sql_rows_for_records("put_policy", policies)
    budget_rows = await _sql_rows_for_records("put_budget", budgets)
    rows_by_method = {
        "list_workloads": workload_rows,
        "list_leases": lease_rows,
        "list_quotas": quota_rows,
        "list_capacity_pools": pool_rows,
        "list_policies": policy_rows,
        "list_budgets": budget_rows,
    }

    cases = (
        ("list_workloads", {}, []),
        ("list_workloads", {"org_id": "acme"}, ("workloads.org_id = 'acme'",)),
        (
            "list_workloads",
            {"org_id": "acme", "project_id": "research"},
            ("workloads.org_id = 'acme'", "workloads.project_id = 'research'"),
        ),
        (
            "list_workloads",
            {"org_id": "acme", "project_id": "research", "state": WorkloadState.RUNNING},
            (
                "workloads.org_id = 'acme'",
                "workloads.project_id = 'research'",
                "workloads.state = 'running'",
            ),
        ),
        ("list_workloads", {"state": WorkloadState.PENDING}, ("workloads.state = 'pending'",)),
        ("list_leases", {"org_id": "acme"}, ("leases.org_id = 'acme'",)),
        (
            "list_leases",
            {"org_id": "acme", "project_id": "research"},
            ("leases.org_id = 'acme'", "leases.project_id = 'research'"),
        ),
        ("list_leases", {"workload_id": "wl-2"}, ("leases.workload_id = 'wl-2'",)),
        ("list_leases", {"project_id": "research"}, ("leases.project_id = 'research'",)),
        ("list_quotas", {"org_id": "acme"}, ("quotas.org_id = 'acme'",)),
        (
            "list_quotas",
            {"org_id": "acme", "project_id": "research"},
            ("quotas.org_id = 'acme'", "quotas.project_id = 'research'"),
        ),
        ("list_quotas", {"principal_id": "alice"}, ("quotas.principal_id = 'alice'",)),
        ("list_quotas", {}, []),
        ("list_capacity_pools", {"fleet_name": "research"}, ("capacity_pools.fleet_name = 'research'",)),
        ("list_capacity_pools", {}, []),
        ("list_policies", {"org_id": "acme"}, ("policies.org_id = 'acme'",)),
        ("list_budgets", {"org_id": "acme"}, ("budgets.org_id = 'acme'",)),
    )

    async with reference.transaction() as transaction:
        for method, filters, fragments in cases:
            reference_result = await getattr(transaction, method)(**filters)
            sql_result, compiled = await _sql_list(method, rows_by_method[method], filters)
            assert sql_result == reference_result, f"{method}({filters}) diverged from reference"
            if filters:
                for fragment in fragments:
                    assert fragment in compiled, f"{method} missing expected pushdown fragment {fragment!r}"
            else:
                assert "WHERE" not in compiled, f"{method} should not add a WHERE clause when unfiltered"


def test_migration_0004_backfills_typed_columns_from_json_payloads(monkeypatch) -> None:
    migration = _migration_module()

    added_columns: list[tuple[str, str]] = []
    created_indexes: list[tuple[str, str, tuple[str, ...]]] = []
    updates: list[str] = []

    monkeypatch.setattr(
        "alembic.op.add_column",
        lambda table, column, *args, **kwargs: added_columns.append((table, column.name)),
    )
    monkeypatch.setattr(
        "alembic.op.create_index",
        lambda name, table, columns, *args, **kwargs: created_indexes.append((name, table, tuple(columns))),
    )
    monkeypatch.setattr("alembic.op.execute", lambda statement, *args, **kwargs: updates.append(str(statement)))

    migration.upgrade()

    assert added_columns == [
        (table, column) for table, columns in _EXPECTED_FILTER_COLUMNS.items() for column, _ in columns
    ]
    assert created_indexes == [
        (f"ix_{table}_{column}", table, (column,))
        for table, columns in _EXPECTED_FILTER_COLUMNS.items()
        for column, _ in columns
    ]
    assert len(updates) == len(_EXPECTED_FILTER_COLUMNS)

    database = sqlite3.connect(":memory:")
    for table, columns in _EXPECTED_FILTER_COLUMNS.items():
        column_sql = ", ".join(f"{name} TEXT" for name, _ in columns)
        database.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY, data TEXT, {column_sql})")

    payloads: dict[str, list[tuple[str, JsonObject]]] = {
        "workloads": [
            ("wl-1", model_json(_workload_record("wl-1", "acme", "research", WorkloadState.RUNNING))),
            ("wl-legacy", {"id": "wl-legacy", "org_id": "globex"}),
        ],
        "cells": [
            (
                "cell-1",
                model_json(
                    CellRecord.model_validate(
                        {
                            **_legacy_multislice_cell(),
                            "id": "cell-1",
                            "pool_id": "research/v5p-16",
                            "state": "ready",
                        }
                    )
                ),
            ),
            ("cell-legacy", {"id": "cell-legacy"}),
        ],
        "capacity_pools": [("pool-1", model_json(_pool_record("pool-1", "research")))],
        "leases": [("lease-1", model_json(_lease_record("lease-1", "acme", "research", "wl-1")))],
        "policies": [("policy-1", model_json(_policy_record("acme", "research")))],
        "quotas": [
            ("quota-1", model_json(_quota_record("acme", None, None))),
            ("quota-2", model_json(_quota_record("acme", "research", "alice"))),
        ],
        "budgets": [("budget-1", model_json(_budget_record("acme", "research")))],
    }
    for table, rows in payloads.items():
        for row_id, data in rows:
            database.execute(f"INSERT INTO {table} (id, data) VALUES (?, ?)", (row_id, json.dumps(data)))

    for update in updates:
        database.execute(update)

    workload_columns = [name for name, _ in _EXPECTED_FILTER_COLUMNS["workloads"]]
    assert database.execute(f"SELECT id, {', '.join(workload_columns)} FROM workloads ORDER BY id").fetchall() == [
        ("wl-1", "acme", "research", "running"),
        ("wl-legacy", "globex", None, None),
    ]
    assert database.execute("SELECT id, org_id, project_id, principal_id FROM quotas ORDER BY id").fetchall() == [
        ("quota-1", "acme", None, None),
        ("quota-2", "acme", "research", "alice"),
    ]


def test_migration_0004_downgrade_drops_exactly_added_indexes_and_columns(monkeypatch) -> None:
    migration = _migration_module()

    dropped_indexes: list[tuple[str, str | None]] = []
    dropped_columns: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "alembic.op.drop_index",
        lambda name, table_name=None, **kwargs: dropped_indexes.append((name, table_name)),
    )
    monkeypatch.setattr(
        "alembic.op.drop_column",
        lambda table, column, **kwargs: dropped_columns.append((table, column)),
    )

    migration.downgrade()

    expected_indexes = {
        (f"ix_{table}_{column}", table) for table, columns in _EXPECTED_FILTER_COLUMNS.items() for column, _ in columns
    }
    expected_columns = {(table, column) for table, columns in _EXPECTED_FILTER_COLUMNS.items() for column, _ in columns}
    assert set(dropped_indexes) == expected_indexes
    assert set(dropped_columns) == expected_columns


class _FlakyBeginSessions:
    """A session factory whose first commits fail like SERIALIZABLE does."""

    def __init__(self, failures: int, sqlstate: str = "40001") -> None:
        self.failures = failures
        self.sqlstate = sqlstate
        self.begins = 0

    @asynccontextmanager
    async def begin(self):
        self.begins += 1
        if self.begins <= self.failures:

            class _Orig(OSError):
                pgcode = self.sqlstate

            raise DBAPIError("SELECT 1", {}, _Orig("conflict"))
        yield object()


class _RaisingBody:
    """A body that fails inside the transaction, the way caller bugs do."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.runs = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def run(self):
        raise self.error


def test_transaction_retries_a_serialization_failure_in_service(tmp_path) -> None:
    repository = SqlAlchemyRepository.__new__(SqlAlchemyRepository)
    sessions = _FlakyBeginSessions(failures=2)
    repository._sessions = sessions

    seen: list[object] = []

    async def scenario() -> None:
        async with repository.transaction() as session:
            seen.append(session)

    asyncio.run(scenario())

    # Three attempts were made; only the successful one reached the body --
    # a failed begin or commit never ran it.
    assert sessions.begins == 3
    assert len(seen) == 1


def test_transaction_gives_up_after_its_attempt_budget(tmp_path) -> None:
    repository = SqlAlchemyRepository.__new__(SqlAlchemyRepository)
    repository._sessions = _FlakyBeginSessions(failures=99)

    async def scenario() -> None:
        async with repository.transaction():
            pass

    try:
        asyncio.run(scenario())
    except TransientRepositoryError:
        pass
    else:
        raise AssertionError("an exhausted retry budget must surface as transient")
    assert repository._sessions.begins == SqlAlchemyRepository.MAX_TRANSACTION_ATTEMPTS


def test_transaction_does_not_retry_a_body_failure() -> None:
    repository = SqlAlchemyRepository.__new__(SqlAlchemyRepository)
    sessions = _FlakyBeginSessions(failures=0)
    repository._sessions = sessions

    class _Orig(OSError):
        pgcode = "40001"

    body = _RaisingBody(DBAPIError("INSERT 1", {}, _Orig("duplicate key")))

    async def scenario() -> None:
        async with repository.transaction():
            await body.run()

    try:
        asyncio.run(scenario())
    except TransientRepositoryError:
        pass
    except DBAPIError as raw:  # pragma: no cover - contract guard
        raise AssertionError("a body conflict must surface translated, not raw") from raw
    else:
        raise AssertionError("a caller's own failure must not be swallowed")
    # Exactly one begin: a body failure cannot be retried by re-yielding, but
    # it is still translated so callers see the typed retryable error.
    assert sessions.begins == 1


class _FlakyCommitSessions:
    """A session factory whose commit -- ``begin``'s ``__aexit__`` -- conflicts.

    PostgreSQL SERIALIZABLE raises 40001 at COMMIT most often, so this is the
    shape the retry path has to answer for.
    """

    def __init__(self, sqlstate: str = "40001") -> None:
        self.sqlstate = sqlstate
        self.begins = 0

    @asynccontextmanager
    async def begin(self):
        self.begins += 1
        yield object()

        class _Orig(OSError):
            pgcode = self.sqlstate

        raise DBAPIError("COMMIT", {}, _Orig("conflict"))


def test_transaction_translates_a_commit_time_conflict_without_a_second_yield() -> None:
    repository = SqlAlchemyRepository.__new__(SqlAlchemyRepository)
    sessions = _FlakyCommitSessions()
    repository._sessions = sessions

    seen: list[object] = []

    async def scenario() -> None:
        async with repository.transaction() as session:
            seen.append(session)

    try:
        asyncio.run(scenario())
    except TransientRepositoryError:
        pass
    except RuntimeError as raw:  # pragma: no cover - contract guard
        raise AssertionError("a commit-time conflict must surface as the typed transient error") from raw
    else:
        raise AssertionError("a commit-time conflict must not be swallowed")

    # The body had already run when the conflict surfaced, so the service
    # cannot retry it: exactly one begin and exactly one yield. A second yield
    # is what turned this failure into RuntimeError("generator didn't stop").
    assert sessions.begins == 1
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_idempotency_reads_run_on_a_non_postgresql_dialect(tmp_path) -> None:
    """``pg_advisory_xact_lock`` is a PostgreSQL function, not a portable one.

    ``get_idempotency`` executed the advisory lock unconditionally, so every
    idempotency-scoped request on a SQLite-backed development plane crashed
    with "no such function: hashtextextended" -- including ``put_idempotency``,
    which reads before writing. The lock is taken only where it exists.
    """

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'eyvan.db'}")
    await create_schema(engine)
    repository = SqlAlchemyRepository(engine)
    record = IdempotencyRecord(
        scope="acme/research/alice",
        operation="workload.submit",
        key="submit-1",
        fingerprint="a" * 64,
        resource_type="workload",
        resource_id="wl-1",
        created_at=utcnow(),
    )
    try:
        async with repository.transaction() as transaction:
            await transaction.put_idempotency(record)

        async with repository.transaction() as transaction:
            stored = await transaction.get_idempotency("acme/research/alice", "workload.submit", "submit-1")
            assert await transaction.get_idempotency("acme/research/alice", "workload.submit", "other") is None
        assert stored == record
    finally:
        await engine.dispose()


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow


@pytest.mark.asyncio
async def test_a_rolled_back_write_into_a_nested_container_leaves_no_trace() -> None:
    """Isolate rolled-back nested-container writes while transactions share immutable records.

    Progress reports are nested by workload and rank, so container copies must
    preserve rollback isolation without deep-copying the full state.
    """

    from eyvan.records import UserProgressReport

    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        await transaction.put_workload(_workload())
        await transaction.put_progress_report("wl-1", UserProgressReport(rank=0, attempt=1, step=1))

    with pytest.raises(RuntimeError, match="rollback"):
        async with repository.transaction() as transaction:
            await transaction.put_progress_report("wl-1", UserProgressReport(rank=1, attempt=1, step=1))
            raise RuntimeError("rollback")

    async with repository.transaction() as transaction:
        assert [report.rank for report in await transaction.list_progress_reports("wl-1")] == [0]


@pytest.mark.asyncio
async def test_a_transaction_shares_committed_records_rather_than_copying_them() -> None:
    """The records are frozen, so copying them bought nothing and cost a core."""

    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        await transaction.put_workload(_workload())

    async with repository.transaction() as first:
        seen = await first.get_workload("wl-1")
    async with repository.transaction() as second:
        assert await second.get_workload("wl-1") is seen


@pytest.mark.asyncio
async def test_a_replaced_record_with_nested_changes_is_isolated_by_rollback() -> None:
    """The supported way to change nested data: a new record, rolled back cleanly."""

    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        await transaction.put_workload(_workload())

    with pytest.raises(RuntimeError, match="rollback"):
        async with repository.transaction() as transaction:
            loaded = await transaction.get_workload("wl-1")
            assert loaded is not None
            spec = loaded.spec.model_copy(update={"environment": {**loaded.spec.environment, "INJECTED": "x"}})
            await transaction.put_workload(loaded.model_copy(update={"spec": spec}))
            raise RuntimeError("rollback")

    async with repository.transaction() as transaction:
        fresh = await transaction.get_workload("wl-1")
    assert fresh is not None and "INJECTED" not in fresh.spec.environment


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True], ids=["commit", "rollback"])
async def test_editing_a_committed_record_in_place_is_refused(fails: bool) -> None:
    """Structural sharing cannot isolate an in-place edit, so the suite forbids one."""

    from eyvan.server.persistence.state import RecordMutatedInPlaceError

    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        await transaction.put_workload(_workload())

    expected = RuntimeError if fails else RecordMutatedInPlaceError
    with pytest.raises(expected) as caught:
        async with repository.transaction() as transaction:
            loaded = await transaction.get_workload("wl-1")
            assert loaded is not None
            loaded.spec.environment["INJECTED"] = "in place"
            await transaction.put_workload(_workload("wl-2"))
            if fails:
                raise RuntimeError("rollback")

    if fails:
        # The transaction's own failure is what propagates; the guard only annotates it.
        assert str(caught.value) == "rollback"
        assert any("in place" in note for note in getattr(caught.value, "__notes__", ()))
    async with repository.transaction() as transaction:
        # Refused before commit: the legitimate write in the same transaction is not persisted.
        assert await transaction.get_workload("wl-2") is None
