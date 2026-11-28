# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The status/result path uses the invited user's live grant, not project READ."""

import asyncio

from fastapi.testclient import TestClient
from test_team_routes import ADMIN, ORIGIN, PASSWORD, _onboard

from eyvan.artifacts.managed import callable_uri, result_uri
from eyvan.config import EyvanSettings
from eyvan.records import (
    CallableExecution,
    ResourceGang,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.server.api.app import create_app

SCOPE = {"org_id": "acme", "project_id": "r"}


def _invite_and_login(http: TestClient, subject: str) -> dict[str, str]:
    email = f"{subject}@example.com"
    token = _onboard(http, email, subject)["token"]
    accepted = http.post("/api/v1/team/invites/accept", json={"token": token, "password": PASSWORD})
    assert accepted.status_code == 200
    http.cookies.clear()
    login = http.post("/api/v1/team/login", json={"email": email, "password": PASSWORD, "token_name": "sdk"})
    assert login.status_code == 200
    return {"Authorization": f"Bearer {login.json()['token']}"}


def test_invited_submitter_can_fetch_only_own_current_scoped_status_and_result(tmp_path) -> None:
    settings = EyvanSettings(
        team_mode=True, dev_auth_enabled=True, artifact_store_uri=str(tmp_path), artifact_token_key="k" * 32
    )
    app = create_app(settings=settings)
    http = TestClient(app, base_url=ORIGIN)
    alice = _invite_and_login(http, "alice")
    bob = _invite_and_login(http, "bob")
    rules = http.put("/api/v1/team/users/alice/rules", headers=ADMIN, json={"max_cpu_units": 1})
    assert rules.status_code == 200
    spec = WorkloadDefinition(
        name="compute",
        execution=CallableExecution(
            artifact_uri=callable_uri("a" * 64), sha256="a" * 64, python="3.13", result_uri=result_uri("requested")
        ),
        gang=ResourceGang(per_replica=ResourceVector(cpu=1)),
    )
    submitted = http.post(
        "/api/v1/workloads",
        headers={**alice, "Idempotency-Key": "job"},
        json={**SCOPE, "spec": spec.model_dump(mode="json")},
    )
    assert submitted.status_code == 201, submitted.text
    workload = WorkloadRecord.model_validate(submitted.json())
    status = f"/api/v1/workloads/{workload.id}"
    execution = workload.spec.execution
    assert isinstance(execution, CallableExecution)
    result_id = f"{workload.id}-g1"
    assert execution.result_uri == result_uri(result_id)
    result = f"/api/v1/artifacts/results/{result_id}"

    async def store(record: WorkloadRecord) -> None:
        async with app.state.service.repository.transaction() as tx:
            await tx.put_workload(record)

    asyncio.run(store(workload.model_copy(update={"state": WorkloadState.RUNNING})))
    transfer = app.state.artifact_access.result_upload_transfer(execution.result_uri, workload_id=workload.id, **SCOPE)
    uploaded = http.put(transfer.url, content=b"result bytes", headers={"X-Eyvan-Artifact-Token": transfer.token})
    assert uploaded.status_code == 200, uploaded.text
    asyncio.run(store(workload.model_copy(update={"state": WorkloadState.SUCCEEDED})))
    owner_status = http.get(status, headers=alice).json()
    assert WorkloadRecord.model_validate(owner_status).state is WorkloadState.SUCCEEDED
    assert {"remote_id", "lease_ids", "dispatch_reservation", "service_endpoint"}.isdisjoint(owner_status)
    assert "lease_ids" in http.get(status, headers=ADMIN).json()
    download = http.get(result, params=SCOPE, headers=alice)
    assert download.status_code == 200 and download.content == b"result bytes"
    assert download.headers["Cache-Control"] == "private, no-store"

    for url, params in ((status, {}), (result, SCOPE)):
        assert http.get(url, params=params, headers=bob).status_code == 403
        assert http.get(url, params=params).status_code == 401
    assert http.get("/api/v1/workloads", params=SCOPE, headers=alice).status_code == 403
    for suffix in ("logs", "progress", "children"):
        assert http.get(f"{status}/{suffix}", headers=alice).status_code == 403
    other = {**SCOPE, "project_id": "other"}
    assert http.get(result, params=other, headers=alice).status_code == 403
    # Even with the same owner and result ID, a caller-supplied scope cannot override the row.
    asyncio.run(store(workload.model_copy(update=other)))
    assert http.get(status, headers=alice).status_code == 403
    assert http.get(result, params=SCOPE, headers=alice).status_code == 403
    asyncio.run(store(workload.model_copy(update={"generation": 2})))
    assert http.get(result, params=SCOPE, headers=alice).status_code == 403
    asyncio.run(store(workload))
    revoked = http.delete(
        "/api/v1/grants", headers=ADMIN, params={**SCOPE, "subject": "alice", "role": "project_submitter"}
    )
    assert revoked.status_code == 204
    assert http.get(status, headers=alice).status_code == 403
    assert http.get(result, params=SCOPE, headers=alice).status_code == 403


def test_owner_exception_is_off_without_team_mode(tmp_path) -> None:
    app = create_app(
        settings=EyvanSettings(dev_auth_enabled=True, artifact_store_uri=str(tmp_path), artifact_token_key="k" * 32)
    )
    http = TestClient(app)
    headers = {"X-Eyvan-User": "alice", "X-Eyvan-Roles": "submitter", "X-Eyvan-Org": "acme", "X-Eyvan-Project": "r"}
    submitted = http.post(
        "/api/v1/workloads",
        headers={**headers, "Idempotency-Key": "job"},
        json={**SCOPE, "spec": {"name": "command", "execution": {"kind": "command", "command": "true"}}},
    )
    assert submitted.status_code == 201, submitted.text
    workload_id = submitted.json()["id"]
    assert http.get(f"/api/v1/workloads/{workload_id}", headers=headers).status_code == 403
    assert http.get(f"/api/v1/artifacts/results/{workload_id}-g1", params=SCOPE, headers=headers).status_code == 403
