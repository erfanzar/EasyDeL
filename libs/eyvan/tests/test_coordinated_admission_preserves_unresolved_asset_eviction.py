# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coordinated admission preserves unresolved asset eviction reservations."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.records import CapacityClass, CellStageState, WorkloadState
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.groups import GroupRole
from eyvan.server.service.scheduling import CoreSchedulingService
from tests.test_groups_atomic_placement import _cell, _principal, _template
from tests.test_locality_stage import _stage


@pytest.fixture(params=["memory", "local", "postgres"])
async def repository(request, tmp_path):
    if request.param != "postgres":
        repo = InMemoryRepository() if request.param == "memory" else LocalJsonRepository(tmp_path / "state.json")
        await repo.open()
        try:
            yield repo
        finally:
            await repo.close()
        return
    url = request.getfixturevalue("postgres_url")
    schema = f"group_eviction_{uuid.uuid4().hex}"
    admin = create_async_engine(url)
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    try:
        await create_schema(engine)
        yield SqlAlchemyRepository(engine)
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


@pytest.mark.parametrize("eviction", ["same-generation", "old-generation", "other-asset"])
async def test_coordinated_group_revalidates_eviction_before_writing_any_member(repository, eviction: str) -> None:
    service = ControlPlaneService(repository=repository, scheduler=CoreSchedulingService())
    generation = 2 if eviction == "old-generation" else 1
    cells = [
        _cell(name, cpu=1).model_copy(update={"generation": generation, "capacity_class": CapacityClass.SPOT})
        for name in ["a", "b"]
    ]
    roles = tuple(
        GroupRole(
            name=name,
            count=1,
            template=_template(name=name).model_copy(
                update={"named_assets": ("asset",), "capacity_class": CapacityClass.SPOT}
            ),
        )
        for name in ["first", "second"]
    )
    parent, members = await service.submit_group(
        _principal(), org_id="acme", project_id="research", roles=roles, idempotency_key="group-eviction"
    )
    stage = _stage("b", "other" if eviction == "other-asset" else "asset", state=CellStageState.EVICTING)
    async with repository.transaction() as tx:
        for cell in cells:
            await tx.put_cell(cell)
        await tx.put_cell_stage(stage)
        before_events = await tx.list_events()

    placed = await service.place_group_atomically(_principal(), coordinator_id=parent.id, lease_ttl_s=60)

    async with repository.transaction() as tx:
        stored = [await tx.get_workload(member.id) for member in members]
        leases = await tx.list_leases()
        dispatches = [row for row in await tx.list_outbox() if row.topic == "workload.dispatch"]
        assert await tx.get_cell_stage("b", stage.asset) == stage
        if eviction == "same-generation":
            assert placed == (), "a later blocked member must not commit earlier members' proposals"
            assert leases == [] and dispatches == []
            assert stored == members
            assert await tx.list_events() == before_events
        else:
            assert set(placed) == {member.id for member in members}
            assert len(leases) == len(dispatches) == 2
            assert all(row is not None and row.state is WorkloadState.LEASED for row in stored)
