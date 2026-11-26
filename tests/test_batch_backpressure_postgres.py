# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The fan-out admission ledger on the backend production actually runs.

Every other test of bounded fan-out uses the in-memory repository, which is fast
and proves the logic. It does not prove that the migration and the ORM agree
about the table, that the identity fits its key column, or that a rolled-back
replica is unaffected -- and those are exactly the properties a durable plan is
worth nothing without.

The upgrade is applied from ``0015_attempt_topology``, the head this change was
written against, so what is exercised is the transition an operator will run and
not a schema created from today's metadata.
"""

from __future__ import annotations

import asyncio

import pytest
from alembic import command
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.records import (
    CommandExecution,
    ExpansionAdmissionState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.migrate import build_alembic_config
from eyvan.server.persistence.sqlalchemy import SCHEMA_REVISION, SqlAlchemyRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.expansion import ExpansionLimits

_PREVIOUS_HEAD = "0015_attempt_topology"


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


def _template() -> WorkloadDefinition:
    return WorkloadDefinition(
        name="shard",
        execution=CommandExecution(command="python eval.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
    )


@pytest.mark.asyncio
async def test_a_bounded_expansion_resumes_from_postgres_after_a_restart(database) -> None:
    """The plan, the ledger, and the next wave all come back off disk."""

    await _upgrade(database, "head")
    repository = SqlAlchemyRepository(database)
    await repository.open()
    service = ControlPlaneService(repository, expansion_limits=ExpansionLimits(ceiling=100, wave_size=2))

    parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=5,
        idempotency_key="batch-1",
    )
    async with repository.transaction() as transaction:
        for child in children[:2]:
            stored = await transaction.get_workload(child.id)
            assert stored is not None
            await transaction.put_workload(stored.model_copy(update={"state": WorkloadState.SUCCEEDED}))

    # A second control plane over the same database, holding none of the first
    # one's memory.
    restarted = SqlAlchemyRepository(database)
    await restarted.open()
    released = await EyvanController(restarted, ControlPlaneService(restarted)).release_expansion_waves_once()

    assert released == 2
    async with restarted.transaction() as transaction:
        admission = await transaction.get_expansion_admission(parent.id)
        stored = [await transaction.get_workload(child.id) for child in children]
    assert admission is not None
    assert (admission.wave_size, admission.released_children) == (2, 4)
    assert admission.state is ExpansionAdmissionState.ADMITTING
    assert [item.schedulable for item in stored if item is not None] == [True, True, True, True, False]

    async with database.connect() as connection:
        row = (
            await connection.execute(
                text(
                    "SELECT id, workload_id, state, total_children, wave_size "
                    "FROM expansion_admissions WHERE workload_id = :workload_id"
                ),
                {"workload_id": parent.id},
            )
        ).one()
    assert row.state == ExpansionAdmissionState.ADMITTING.value
    assert (row.total_children, row.wave_size) == (5, 2)
    assert len(row.id) <= 32, "the ledger identity has to fit the key column on every backend"


@pytest.mark.asyncio
async def test_the_ledger_arrives_and_departs_without_making_a_workload_unreadable(database) -> None:
    """Additive both ways: a rollback loses the plan and nothing else.

    This is the whole reason the plan is a table rather than fields on the
    workload document. The rows are decoded here exactly as a rolled-back replica
    would decode them -- ``WorkloadRecord`` is ``extra='forbid'``, so a field this
    change had added would fail this assertion rather than a production restart.
    """

    await _upgrade(database, "head")
    repository = SqlAlchemyRepository(database)
    await repository.open()
    service = ControlPlaneService(repository)
    workload, _replayed = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_template(),
        idempotency_key="workload-1",
    )
    bounded = ControlPlaneService(repository, expansion_limits=ExpansionLimits(ceiling=100, wave_size=1))
    parent, children = await bounded.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=3,
        idempotency_key="batch-1",
    )

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
    assert "expansion_admissions" not in tables
    assert "workload_topologies" in tables
    decoded = [WorkloadRecord.model_validate(row) for row in rows]
    assert {record.id for record in decoded} == {workload.id, parent.id, *(child.id for child in children)}


@pytest.mark.asyncio
async def test_the_migration_chain_ends_at_the_revision_this_build_declares(database) -> None:
    await _upgrade(database, "head")

    async with database.connect() as connection:
        versions = (await connection.execute(text("SELECT version_num FROM alembic_version"))).scalars().all()

    assert versions == [SCHEMA_REVISION]
