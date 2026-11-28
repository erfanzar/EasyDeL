# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Independent API-only instances read committed state from one PostgreSQL primary."""

import httpx
from sqlalchemy.ext.asyncio import create_async_engine
from test_postgres_semantics import _migrated
from test_postgres_semantics import database as database

from eyvan._internal.platform import EyvanPlatformConfig
from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.records import CommandExecution, Principal, Role, RoleGrant, WorkloadDefinition
from eyvan.server.api.app import create_app
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository
from eyvan.server.service import ControlPlaneService


async def test_an_api_only_replica_sees_another_instances_committed_updates(database, tmp_path):
    writer = await _migrated(database)
    reader_engine = create_async_engine(database.url)
    reader = SqlAlchemyRepository(reader_engine)
    await reader.open()
    try:
        service = ControlPlaneService(writer)
        admin = Principal(subject="writer", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
        settings = EyvanSettings(
            repository_backend=RepositoryBackend.POSTGRES,
            controller_enabled=False,
            bootstrap_token="replica-test-token",
            artifact_token_key="k" * 32,
            artifact_store_uri=str(tmp_path),
        )
        app = create_app(settings=settings, repository=reader, platform_config=EyvanPlatformConfig())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://replica") as client:
            headers = {"Authorization": "Bearer replica-test-token"}
            workload, _ = await service.submit_workload(
                admin,
                org_id="acme",
                project_id="research",
                spec=WorkloadDefinition(name="replica-read", execution=CommandExecution(command="true")),
                idempotency_key="replica-submit",
            )
            initial = await client.get(f"/api/v1/workloads/{workload.id}", headers=headers)
            assert initial.status_code == 200
            assert initial.json()["state"] == "pending"
            await service.cancel_workload(admin, workload.id, idempotency_key="replica-cancel")
            updated = await client.get(f"/api/v1/workloads/{workload.id}", headers=headers)
            assert updated.status_code == 200
            assert updated.json()["state"] == "cancelled"
    finally:
        await reader.close()
        await writer.close()
        await reader_engine.dispose()
