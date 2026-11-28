# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""API correlation identifies server-owned work and never echoes a caller hint."""

from dataclasses import replace

import pytest
from starlette.testclient import TestClient

from eyvan._internal.platform import EyvanPlatformConfig
from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.records import WorkloadRecord
from eyvan.server.api.app import create_app


@pytest.mark.parametrize("enabled", [False, True])
def test_workload_response_correlation_is_opt_in_and_authoritative(monkeypatch, tmp_path, enabled):
    monkeypatch.setenv("EYVAN_CORRELATION_ENABLED", "1" if enabled else "0")
    settings = replace(
        EyvanSettings.from_env(),
        repository_backend=RepositoryBackend.MEMORY,
        controller_enabled=False,
        dev_auth_enabled=False,
        oidc_issuer=None,
        bootstrap_token="test-admin",
        artifact_token_key="k" * 32,
        artifact_store_uri=str(tmp_path),
    )
    app = create_app(settings=settings, platform_config=EyvanPlatformConfig())
    with TestClient(app) as client:
        headers = {
            "Authorization": "Bearer test-admin",
            "Idempotency-Key": "submit",
            "X-Eyvan-Correlation-ID": "caller-spoof",
        }
        created = client.post(
            "/api/v1/workloads",
            headers=headers,
            json={
                "org_id": "test",
                "project_id": "test",
                "spec": {"name": "correlation", "execution": {"kind": "command", "command": "true"}},
            },
        )
        assert created.status_code == 201, created.text
        identifier = created.json()["id"]
        assert created.headers.get("X-Eyvan-Correlation-ID") == (identifier if enabled else None)
        loaded = client.get(f"/api/v1/workloads/{identifier}", headers=headers)
        assert loaded.status_code == 200
        assert loaded.headers.get("X-Eyvan-Correlation-ID") == (identifier if enabled else None)
        assert app.state.service.correlation_enabled is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_single_workload_mutations_keep_the_workload_correlation(monkeypatch, tmp_path, enabled):
    monkeypatch.setenv("EYVAN_CORRELATION_ENABLED", "1" if enabled else "0")
    settings = replace(
        EyvanSettings.from_env(),
        repository_backend=RepositoryBackend.MEMORY,
        controller_enabled=False,
        dev_auth_enabled=False,
        oidc_issuer=None,
        bootstrap_token="test-admin",
        artifact_token_key="k" * 32,
        artifact_store_uri=str(tmp_path),
    )
    app = create_app(settings=settings, platform_config=EyvanPlatformConfig())
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer test-admin", "Idempotency-Key": "submit"}
        created = client.post(
            "/api/v1/workloads",
            headers=headers,
            json={
                "org_id": "test",
                "project_id": "test",
                "spec": {"name": "correlation", "execution": {"kind": "command", "command": "true"}},
            },
        )
        assert created.status_code == 201, created.text
        workload_id = created.json()["id"]
        record = WorkloadRecord.model_validate(client.get(f"/api/v1/workloads/{workload_id}", headers=headers).json())

        async def fake_settle(*args, **kwargs):
            del args, kwargs
            return record, None, None

        async def fake_finalize(*args, **kwargs):
            del args, kwargs
            return record, False

        app.state.service.settle_preempted_workload = fake_settle
        app.state.service.force_finalize_stuck_workload = fake_finalize
        expected = workload_id if enabled else None

        kicked = client.post(f"/api/v1/workloads/{workload_id}/kick", headers=headers)
        assert kicked.status_code == 200, kicked.text
        assert kicked.headers.get("X-Eyvan-Correlation-ID") == expected
        finalized = client.post(
            f"/api/v1/workloads/{workload_id}/force-finalize",
            headers={**headers, "Idempotency-Key": "finalize"},
            json={"reason": "operator test"},
        )
        assert finalized.status_code == 200, finalized.text
        assert finalized.headers.get("X-Eyvan-Correlation-ID") == expected


@pytest.mark.parametrize("enabled", [False, True])
async def test_lease_and_workload_driven_cell_reads_share_the_workload_identity(tmp_path, enabled):
    from datetime import timedelta

    import httpx

    from eyvan.records import (
        CapacityPoolRecord,
        CellRecord,
        CommandExecution,
        LeaseRecord,
        ResourceLimits,
        WorkloadDefinition,
        utcnow,
    )
    from eyvan.server.persistence import InMemoryRepository

    repository = InMemoryRepository()
    await repository.open()
    spec = WorkloadDefinition(name="reference", execution=CommandExecution(command="true"))
    workload = WorkloadRecord(id="wl_reference", org_id="test", project_id="test", created_by="owner", spec=spec)
    pool = CapacityPoolRecord(
        id="pool",
        fleet_name="fleet",
        node_type="cpu",
        provider="local",
        region="local",
        capacity=ResourceLimits(cpu=1),
        allocatable=ResourceLimits(cpu=1),
        agent_endpoint="http://agent.invalid",
    )
    cell = CellRecord(
        id="cell",
        provider="local",
        region="local",
        pool_id=pool.id,
        capacity=pool.capacity,
        allocatable=pool.allocatable,
        autoscale_request_workload_id=workload.id,
    )
    lease = LeaseRecord(
        id="lease",
        workload_id=workload.id,
        cell_id=cell.id,
        org_id="test",
        project_id="test",
        resources=spec.gang,
        fencing_token="fence",
        expires_at=utcnow() + timedelta(minutes=5),
    )
    async with repository.transaction() as tx:
        await tx.put_capacity_pool(pool)
        await tx.put_workload(workload)
        await tx.put_cell(cell)
        await tx.put_lease(lease)
    settings = EyvanSettings(
        controller_enabled=False,
        correlation_enabled=enabled,
        bootstrap_token="test-admin",
        artifact_token_key="k" * 32,
        artifact_store_uri=str(tmp_path),
    )
    app = create_app(settings=settings, repository=repository, platform_config=EyvanPlatformConfig())
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for path in ("/api/v1/leases/lease", "/api/v1/cells/cell"):
                response = await client.get(path, headers={"Authorization": "Bearer test-admin"})
                assert response.status_code == 200
                assert response.headers.get("X-Eyvan-Correlation-ID") == (workload.id if enabled else None)
    finally:
        await repository.close()
