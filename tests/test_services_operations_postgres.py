# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Service health and decisions on the backend production actually runs.

Every other test of these two tables uses the in-memory repository, which is
fast and proves the logic. It does not prove that the migration and the ORM
agree about the columns, that the per-service decision sequence is really a key
and not a convention, or that rolling the migration back leaves a workload
readable -- and a durable audit is worth nothing without all three.

The upgrade is applied from ``0017_merge_stage_rot``, the head this change was
written against, so what runs here is the transition an operator will run rather
than a schema created from today's metadata.

The rollback assertion is the load-bearing one. Health counters and the decision
log are separate tables precisely so a rolled-back replica loses them and
nothing else; fields on the workload document -- which is ``extra='forbid'`` --
would have made every workload in the fleet unreadable on the way back.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from alembic import command
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.records import (
    CommandExecution,
    Principal,
    Role,
    RoleGrant,
    ServiceDecisionKind,
    ServiceDecisionRecord,
    ServiceObservation,
    ServiceProbeOutcome,
    ServiceReplicaHealthRecord,
    WorkloadDefinition,
    WorkloadRecord,
)
from eyvan.server.persistence.migrate import build_alembic_config
from eyvan.server.persistence.sqlalchemy import SCHEMA_REVISION, SqlAlchemyRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.services import ServicePolicy, plan_autoscale

_PREVIOUS_HEAD = "0017_merge_stage_rot"
NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)


@pytest.fixture
async def database(postgres_url: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EYVAN_DATABASE_URL", raising=False)
    engine = create_async_engine(postgres_url, poolclass=None)
    async with engine.begin() as connection:
        await connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await connection.execute(text("CREATE SCHEMA public"))
    try:
        yield engine
    finally:
        await engine.dispose()


def _config(engine):
    url = str(engine.url.render_as_string(hide_password=False))
    config = build_alembic_config(url)
    config.set_main_option("sqlalchemy.url", url)
    return config


async def _upgrade(engine, revision: str) -> None:
    await asyncio.to_thread(command.upgrade, _config(engine), revision)


async def _downgrade(engine, revision: str) -> None:
    await asyncio.to_thread(command.downgrade, _config(engine), revision)


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(
            RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),
            RoleGrant(role=Role.FLEET_ADMIN),
        ),
    )


def _observed(load: float, *, age_s: float = 0.0, now: datetime | None = None) -> ServiceObservation:
    return ServiceObservation(
        live_replicas=1,
        healthy_replicas=1,
        unhealthy_replicas=0,
        unknown_replicas=0,
        loaded_replicas=1,
        average_load=load,
        oldest_load_observed_at=(now or NOW) - timedelta(seconds=age_s),
    )


def _health(replica_id: str, *, failures: int = 0, unknowns: int = 0, load: float | None = None):
    outcome = ServiceProbeOutcome.SUCCESS
    if unknowns:
        outcome = ServiceProbeOutcome.UNKNOWN
    elif failures:
        outcome = ServiceProbeOutcome.FAILURE
    return ServiceReplicaHealthRecord(
        service_id="svc-1",
        replica_id=replica_id,
        replica_generation=1,
        endpoint=f"http://10.0.0.1:8080/{replica_id}",
        last_outcome=outcome,
        consecutive_failures=failures,
        consecutive_unknowns=unknowns,
        last_success_at=None if failures or unknowns else NOW,
        last_failure_at=NOW if failures else None,
        observed_at=NOW,
        observed_load=load,
        load_observed_at=None if load is None else NOW,
    )


def _decision(sequence: int, *, kind: ServiceDecisionKind = ServiceDecisionKind.SCALE_OUT, decided_at=NOW):
    return ServiceDecisionRecord(
        service_id="svc-1",
        sequence=sequence,
        kind=kind,
        from_replicas=1,
        to_replicas=1 if kind is ServiceDecisionKind.REPLACE_REPLICA else 2,
        replica_id="replica-1" if kind is ServiceDecisionKind.REPLACE_REPLICA else None,
        replica_generation=1 if kind is ServiceDecisionKind.REPLACE_REPLICA else None,
        reason=f"decision {sequence}",
        observed=_observed(0.95),
        decided_at=decided_at,
    )


@pytest.mark.asyncio
async def test_health_counters_survive_a_restart_with_unknown_still_apart_from_failed(database) -> None:
    """A restart that collapsed the two would tear down whatever it could not reach."""

    await _upgrade(database, "head")
    repository = SqlAlchemyRepository(database)
    await repository.open()
    async with repository.transaction() as transaction:
        await transaction.put_service_replica_health(_health("replica-1", failures=3))
        await transaction.put_service_replica_health(_health("replica-2", unknowns=9))

    restarted = SqlAlchemyRepository(database)
    await restarted.open()
    async with restarted.transaction() as transaction:
        stored = await transaction.list_service_replica_health("svc-1")

    assert [record.replica_id for record in stored] == ["replica-1", "replica-2"]
    assert (stored[0].consecutive_failures, stored[0].consecutive_unknowns) == (3, 0)
    assert (stored[1].consecutive_failures, stored[1].consecutive_unknowns) == (0, 9)
    assert stored[1].last_outcome is ServiceProbeOutcome.UNKNOWN


@pytest.mark.asyncio
async def test_one_replica_upserts_exactly_its_own_row(database) -> None:
    await _upgrade(database, "head")
    repository = SqlAlchemyRepository(database)
    await repository.open()
    async with repository.transaction() as transaction:
        await transaction.put_service_replica_health(_health("replica-1", failures=1))
        await transaction.put_service_replica_health(_health("replica-2", failures=1))
    async with repository.transaction() as transaction:
        await transaction.put_service_replica_health(_health("replica-1", failures=2))
        removed = await transaction.delete_service_replica_health("svc-1", "replica-2")
        missing = await transaction.delete_service_replica_health("svc-1", "replica-404")

    assert (removed, missing) == (True, False)
    async with repository.transaction() as transaction:
        stored = await transaction.list_service_replica_health("svc-1")
    assert [(record.replica_id, record.consecutive_failures) for record in stored] == [("replica-1", 2)]


@pytest.mark.asyncio
async def test_the_decision_sequence_is_a_key_and_not_a_convention(database) -> None:
    """Two leaders minting the same sequence must collide, not overwrite.

    The loser's whole decision is rolled back, which is what stops one committed
    explanation quietly replacing another.
    """

    await _upgrade(database, "head")
    repository = SqlAlchemyRepository(database)
    await repository.open()
    async with repository.transaction() as transaction:
        await transaction.put_service_decision(_decision(1))

    async with database.connect() as connection:
        with pytest.raises(IntegrityError):
            async with connection.begin():
                await connection.execute(
                    text("INSERT INTO service_decisions (service_id, sequence, kind, data) VALUES (:s, 1, :k, :d)"),
                    {"s": "svc-1", "k": "scale_in", "d": "{}"},
                )


@pytest.mark.asyncio
async def test_a_non_positive_sequence_is_refused_by_the_database(database) -> None:
    """The audit is append-only and ordered; sequence zero is neither."""

    await _upgrade(database, "head")

    async with database.connect() as connection:
        with pytest.raises(IntegrityError):
            async with connection.begin():
                await connection.execute(
                    text("INSERT INTO service_decisions (service_id, sequence, kind, data) VALUES (:s, 0, :k, :d)"),
                    {"s": "svc-1", "k": "scale_out", "d": "{}"},
                )


@pytest.mark.asyncio
async def test_the_newest_decision_comes_back_first_and_bounded(database) -> None:
    await _upgrade(database, "head")
    repository = SqlAlchemyRepository(database)
    await repository.open()
    async with repository.transaction() as transaction:
        for sequence in range(1, 6):
            await transaction.put_service_decision(_decision(sequence))

    async with repository.transaction() as transaction:
        newest = await transaction.list_service_decisions("svc-1", limit=2)
        dropped = await transaction.delete_service_decisions_before("svc-1", sequence=4)
        remaining = await transaction.list_service_decisions("svc-1")

    assert [item.sequence for item in newest] == [5, 4]
    assert dropped == 3
    assert [item.sequence for item in remaining] == [5, 4]


@pytest.mark.asyncio
async def test_a_cooldown_read_off_disk_refuses_a_second_leader(database) -> None:
    """The cooldown is the newest committed row, so it is not a leader's memory.

    A control plane that restarted would otherwise resize immediately, which for
    a rate limit is exactly a restart loop.
    """

    await _upgrade(database, "head")
    repository = SqlAlchemyRepository(database)
    await repository.open()
    now = datetime.now(UTC)
    async with repository.transaction() as transaction:
        await transaction.put_service_decision(_decision(1, decided_at=now - timedelta(seconds=30)))

    restarted = SqlAlchemyRepository(database)
    await restarted.open()
    async with restarted.transaction() as transaction:
        (newest,) = await transaction.list_service_decisions("svc-1", limit=1)
    policy = ServicePolicy(
        replicas=2,
        revision="v1",
        health_path="/healthz",
        autoscale=True,
        min_replicas=1,
        max_replicas=8,
        scale_cooldown_s=300.0,
    )

    decision = plan_autoscale(policy, _observed(0.99, now=now), now=now, last_decision=newest)

    assert decision.changes is False
    assert "cooling down" in decision.reason


@pytest.mark.asyncio
async def test_the_kind_is_a_real_column_an_operator_can_query(database) -> None:
    """ "When did this service last scale out" is answerable in SQL at 3am."""

    await _upgrade(database, "head")
    repository = SqlAlchemyRepository(database)
    await repository.open()
    async with repository.transaction() as transaction:
        await transaction.put_service_decision(_decision(1, kind=ServiceDecisionKind.SCALE_OUT))
        await transaction.put_service_decision(_decision(2, kind=ServiceDecisionKind.REPLACE_REPLICA))

    async with database.connect() as connection:
        rows = (
            await connection.execute(
                text("SELECT sequence, kind FROM service_decisions WHERE service_id = :s ORDER BY sequence"),
                {"s": "svc-1"},
            )
        ).all()

    assert [(row.sequence, row.kind) for row in rows] == [(1, "scale_out"), (2, "replace_replica")]


@pytest.mark.asyncio
async def test_the_tables_arrive_and_depart_without_making_a_workload_unreadable(database) -> None:
    """Additive both ways: a rollback loses the probe counters and the audit,
    and nothing else.

    ``WorkloadRecord`` is ``extra='forbid'``, so a field this change had added to
    the workload document would fail this assertion rather than a production
    restart.
    """

    await _upgrade(database, "head")
    repository = SqlAlchemyRepository(database)
    await repository.open()
    service = ControlPlaneService(repository)
    workload, _replayed = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="api", execution=CommandExecution(command="python serve.py")),
        idempotency_key="workload-1",
    )
    async with repository.transaction() as transaction:
        await transaction.put_service_replica_health(_health("replica-1", failures=2))
        await transaction.put_service_decision(_decision(1))

    await _downgrade(database, _PREVIOUS_HEAD)

    async with database.connect() as connection:
        version = (await connection.execute(text("SELECT version_num FROM alembic_version"))).scalar_one()
        tables = {
            row[0]
            for row in (
                await connection.execute(
                    text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
                )
            ).all()
        }
        rows = (await connection.execute(text("SELECT data FROM workloads"))).scalars().all()

    assert version == _PREVIOUS_HEAD
    assert {"service_replica_health", "service_decisions"}.isdisjoint(tables)
    assert {"cell_stages", "expansion_admissions", "workload_topologies"} <= tables
    decoded = [WorkloadRecord.model_validate(row) for row in rows]
    assert {record.id for record in decoded} == {workload.id}


@pytest.mark.asyncio
async def test_the_migration_chain_ends_at_the_revision_this_build_declares(database) -> None:
    """One head, and the one the ORM refuses to start without."""

    await _upgrade(database, "head")

    async with database.connect() as connection:
        versions = (await connection.execute(text("SELECT version_num FROM alembic_version"))).scalars().all()

    assert versions == [SCHEMA_REVISION]
