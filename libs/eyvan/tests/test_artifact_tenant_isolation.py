# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adversarial same-URI tenant isolation through persistence, service, and API."""

from __future__ import annotations

import typing as tp

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.artifacts.managed import callable_uri
from eyvan.config import EyvanSettings
from eyvan.records import (
    ArtifactReferenceRecord,
    ManagedArtifactRecord,
    ManagedArtifactState,
    Principal,
    Role,
    RoleGrant,
    artifact_reference_id,
    utcnow,
)
from eyvan.server.api.app import create_app
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema

_URI = callable_uri("a" * 64)


@pytest.fixture(params=("memory", "sqlite"))
async def repository(request: pytest.FixtureRequest, tmp_path) -> tp.AsyncIterator[object]:
    if request.param == "memory":
        repository = InMemoryRepository()
        await repository.open()
        yield repository
        return
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tenant-isolation.db'}")
    await create_schema(engine)
    repository = SqlAlchemyRepository(engine)
    await repository.open()
    try:
        yield repository
    finally:
        await engine.dispose()


def _principal(project_id: str) -> Principal:
    return Principal(
        subject=f"admin-{project_id}",
        grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id=project_id),),
    )


def _headers(project_id: str) -> dict[str, str]:
    return {
        "X-Eyvan-User": f"admin-{project_id}",
        "X-Eyvan-Roles": "project_admin",
        "X-Eyvan-Org": "acme",
        "X-Eyvan-Project": project_id,
    }


async def _seed(service) -> None:
    await service.index_managed_artifact(org_id="acme", project_id="alpha", uri=_URI, size_bytes=11, sha256="a" * 64)
    await service.index_managed_artifact(org_id="acme", project_id="beta", uri=_URI, size_bytes=29, sha256="a" * 64)
    async with service.repository.transaction() as transaction:
        for project_id, workload_id in (("alpha", "wl-alpha"), ("beta", "wl-beta")):
            record = await transaction.get_managed_artifact("acme", project_id, _URI)
            assert record is not None
            now = utcnow()
            await transaction.put_managed_artifact(
                ManagedArtifactRecord.model_validate(
                    {
                        **record.model_dump(mode="python"),
                        "state": ManagedArtifactState.ACTIVE,
                        "object_version": f"test:{project_id}",
                        "activated_at": now,
                        "updated_at": now,
                        "quarantine_reason": None,
                    }
                )
            )
            await transaction.put_artifact_reference(
                ArtifactReferenceRecord(
                    id=artifact_reference_id("acme", project_id, _URI, workload_id),
                    uri=_URI,
                    workload_id=workload_id,
                    org_id="acme",
                    project_id=project_id,
                )
            )


async def _exercise_same_uri_contract(repository, tmp_path) -> None:
    app = create_app(
        settings=EyvanSettings(
            dev_auth_enabled=True,
            artifact_store_uri=str(tmp_path / "objects"),
            artifact_token_key="k" * 32,
        ),
        repository=repository,
    )
    service = app.state.service
    await _seed(service)

    # Service reads never merge the two ledger rows or reference sets.
    alpha_usage = await service.artifact_usage(_principal("alpha"), org_id="acme", project_id="alpha")
    beta_usage = await service.artifact_usage(_principal("beta"), org_id="acme", project_id="beta")
    assert (alpha_usage.artifact_count, alpha_usage.stored_bytes) == (1, 11)
    assert (beta_usage.artifact_count, beta_usage.stored_bytes) == (1, 29)
    assert [
        reference.workload_id
        for reference in await service.artifact_references(
            _principal("alpha"), org_id="acme", project_id="alpha", uri=_URI
        )
    ] == ["wl-alpha"]
    assert [
        reference.workload_id
        for reference in await service.artifact_references(
            _principal("beta"), org_id="acme", project_id="beta", uri=_URI
        )
    ] == ["wl-beta"]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        held = await client.post(
            "/api/v1/artifacts/holds",
            headers=_headers("alpha"),
            json={
                "org_id": "acme",
                "project_id": "alpha",
                "uri": _URI,
                "reason": "alpha evidence",
            },
        )
        assert held.status_code == 200, held.text
        alpha_references = await client.get(
            "/api/v1/artifacts/references",
            headers=_headers("alpha"),
            params={"org_id": "acme", "project_id": "alpha", "uri": _URI},
        )
        beta_references = await client.get(
            "/api/v1/artifacts/references",
            headers=_headers("beta"),
            params={"org_id": "acme", "project_id": "beta", "uri": _URI},
        )
        cross_tenant_probe = await client.get(
            "/api/v1/artifacts/references",
            headers=_headers("alpha"),
            params={"org_id": "acme", "project_id": "beta", "uri": _URI},
        )

    assert [item["workload_id"] for item in alpha_references.json()] == ["wl-alpha"]
    assert [item["workload_id"] for item in beta_references.json()] == ["wl-beta"]
    assert cross_tenant_probe.status_code == 403

    async with repository.transaction() as transaction:
        alpha = await transaction.get_managed_artifact("acme", "alpha", _URI)
        beta = await transaction.get_managed_artifact("acme", "beta", _URI)
    assert alpha is not None and alpha.hold is not None and alpha.hold.reason == "alpha evidence"
    assert beta is not None and beta.hold is None
    assert beta.size_bytes == 29


@pytest.mark.asyncio
async def test_same_uri_is_isolated_on_memory_and_sqlite(repository, tmp_path) -> None:
    await _exercise_same_uri_contract(repository, tmp_path)


@pytest.mark.asyncio
async def test_same_uri_is_isolated_on_postgresql(postgres_url: str, tmp_path) -> None:
    """Runs only when the shared PostgreSQL fixture can satisfy its env gate."""

    engine = create_async_engine(postgres_url, poolclass=None)
    async with engine.begin() as connection:
        await connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await connection.execute(text("CREATE SCHEMA public"))
    await create_schema(engine)
    repository = SqlAlchemyRepository(engine)
    await repository.open()
    try:
        await _exercise_same_uri_contract(repository, tmp_path)
    finally:
        await engine.dispose()
