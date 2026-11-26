# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Research plans: persistence in every backend, and the HTTP lifecycle with ownership refusals."""

from __future__ import annotations

import asyncio
import json

import pytest
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.records import Principal, ResearchPlanRecord, Role, RoleGrant
from eyvan.server.api.app import create_app

SPEC = {"name": "probe-job", "execution": {"kind": "command", "command": "echo hi"}}
ADMIN = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
ALICE = {"X-Eyvan-User": "alice", "X-Eyvan-Roles": "project_operator", "X-Eyvan-Org": "acme", "X-Eyvan-Project": "r"}
BOB = {"X-Eyvan-User": "bob", "X-Eyvan-Roles": "project_operator", "X-Eyvan-Org": "acme", "X-Eyvan-Project": "r"}
SUBMITTER = {"X-Eyvan-User": "sam", "X-Eyvan-Roles": "project_submitter", "X-Eyvan-Org": "acme", "X-Eyvan-Project": "r"}
BUDGET = {"max_jobs": 4, "max_chip_hours": 10, "max_wall_seconds": 3600}


def _plan_body(jobs: int = 2, **extra) -> dict:
    return {
        "title": "lr sweep",
        "goal": "find a learning rate",
        "org_id": "acme",
        "project_id": "r",
        "jobs": [{"name": f"job-{index}", "spec": {**SPEC, "name": f"job-{index}"}} for index in range(jobs)],
        "budget": BUDGET,
        **extra,
    }


def _client() -> TestClient:
    settings = EyvanSettings(dev_auth_enabled=True, controller_enabled=False, artifact_token_key="k" * 32)
    return TestClient(create_app(settings=settings), base_url="http://127.0.0.1:8080")


@pytest.mark.parametrize("backend", ["memory", "local", "sql"])
def test_research_plans_persist_in_every_backend(backend: str, tmp_path) -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    from eyvan.server.persistence.local import LocalJsonRepository
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
    from eyvan.server.service.control_plane import ControlPlaneService

    alice = Principal(subject="alice", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="r"),))
    admin = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))

    async def run() -> None:
        engine = None
        if backend == "memory":
            repository = InMemoryRepository()
        elif backend == "local":
            repository = LocalJsonRepository(tmp_path / "control-plane.json")
        else:
            engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'plane.db'}")
            await create_schema(engine)
            repository = SqlAlchemyRepository(engine)
        await repository.open()
        service = ControlPlaneService(repository)
        first = await service.propose_research_plan(
            alice, title="sweep", org_id="acme", project_id="r", jobs=[{"name": "a", "spec": SPEC}], budget=BUDGET
        )
        second = await service.propose_research_plan(
            alice, title="other", org_id="acme", project_id="r", jobs=[{"name": "b", "spec": SPEC}], budget=BUDGET
        )
        await service.approve_research_plan(alice, first.id)
        await service.stop_research_plan(alice, second.id, reason="changed my mind")
        await repository.close()
        if backend == "memory":
            stored = await service.list_research_plans(admin)
        else:
            if backend == "local":
                reopened = LocalJsonRepository(tmp_path / "control-plane.json")
            else:
                reopened = SqlAlchemyRepository(engine)
            await reopened.open()
            reopened_service = ControlPlaneService(reopened)
            stored = await reopened_service.list_research_plans(admin)
            running = await reopened_service.list_research_plans(alice, state="approved")
            assert [item.id for item in running] == [first.id]
            await reopened.close()
            if backend == "local":
                assert json.loads((tmp_path / "control-plane.json").read_text())["version"] == 11
        if engine is not None:
            await engine.dispose()
        by_id = {item.id: item for item in stored}
        assert set(by_id) == {first.id, second.id}
        assert by_id[first.id].state == "approved" and by_id[first.id].approved_by == "alice"
        assert by_id[first.id].approval_grants == (RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="r"),)
        assert by_id[second.id].state == "stopped" and by_id[second.id].stop_reason == "changed my mind"
        assert by_id[first.id].jobs[0].spec == SPEC

    asyncio.run(run())


def test_v10_document_refuses_a_state_holding_research_plans() -> None:
    from eyvan.server.persistence.local import LocalControlPlaneV10Document, local_control_plane_document
    from eyvan.server.persistence.state import ControlPlaneState

    plan = ResearchPlanRecord(
        id="rp_0123456789abcdef",
        subject="alice",
        org_id="acme",
        project_id="r",
        title="t",
        jobs=({"name": "a", "spec": SPEC},),
        budget=BUDGET,
    )
    state = ControlPlaneState(research_plans={plan.id: plan})
    with pytest.raises(ValueError, match="v11"):
        LocalControlPlaneV10Document.from_state(state)
    assert local_control_plane_document(state).version == 11
    assert local_control_plane_document(ControlPlaneState()).version == 2


def test_research_plan_record_refuses_more_jobs_than_its_budget() -> None:
    with pytest.raises(ValueError, match="max_jobs"):
        ResearchPlanRecord(
            id="rp_0123456789abcdef",
            subject="alice",
            org_id="acme",
            project_id="r",
            title="t",
            jobs=tuple({"name": f"j{index}", "spec": SPEC} for index in range(3)),
            budget={**BUDGET, "max_jobs": 2},
        )


def test_propose_approve_show_and_stop_over_http() -> None:
    http = _client()
    proposed = http.post("/api/v1/research/plans", json=_plan_body(), headers=ALICE)
    assert proposed.status_code == 201, proposed.text
    plan = proposed.json()
    assert plan["state"] == "proposed" and plan["subject"] == "alice" and plan["jobs_planned"] == 2
    assert [job["state"] for job in plan["jobs"]] == ["not_submitted", "not_submitted"]
    assert plan["remaining"] == {"jobs": 4, "chip_hours": 10.0}

    listed = http.get("/api/v1/research/plans", headers=ALICE)
    assert [item["id"] for item in listed.json()["plans"]] == [plan["id"]]

    approved = http.post(f"/api/v1/research/plans/{plan['id']}/approve", headers=ALICE)
    assert approved.status_code == 200, approved.text
    assert approved.json()["state"] == "approved" and approved.json()["approved_by"] == "alice"
    again = http.post(f"/api/v1/research/plans/{plan['id']}/approve", headers=ALICE)
    assert again.status_code == 409

    shown = http.get(f"/api/v1/research/plans/{plan['id']}", headers=ALICE)
    assert shown.status_code == 200 and [note["kind"] for note in shown.json()["notes"]] == ["proposed", "approved"]

    stopped = http.post(f"/api/v1/research/plans/{plan['id']}/stop", json={"reason": "enough"}, headers=ALICE)
    assert stopped.status_code == 200, stopped.text
    assert stopped.json()["state"] == "stopped" and stopped.json()["stop_reason"] == "enough"
    assert http.post(f"/api/v1/research/plans/{plan['id']}/stop", headers=ALICE).status_code == 409


def test_other_users_get_404_and_admins_see_and_stop_but_never_approve() -> None:
    http = _client()
    plan = http.post("/api/v1/research/plans", json=_plan_body(), headers=ALICE).json()
    path = f"/api/v1/research/plans/{plan['id']}"

    assert http.get(path, headers=BOB).status_code == 404
    assert http.post(f"{path}/approve", headers=BOB).status_code == 404
    assert http.post(f"{path}/stop", headers=BOB).status_code == 404
    assert http.get("/api/v1/research/plans", headers=BOB).json() == {"plans": []}
    assert http.get("/api/v1/research/plans/rp_ffffffffffffffff", headers=ALICE).status_code == 404

    assert [item["id"] for item in http.get("/api/v1/research/plans", headers=ADMIN).json()["plans"]] == [plan["id"]]
    assert http.get(path, headers=ADMIN).status_code == 200
    refused = http.post(f"{path}/approve", headers=ADMIN)
    assert refused.status_code == 403 and "only the person who proposed" in refused.json()["detail"]
    stopped = http.post(f"{path}/stop", headers=ADMIN)
    assert stopped.status_code == 200 and stopped.json()["state"] == "stopped"


def test_proposals_are_validated_now_and_bounded_by_the_budget() -> None:
    http = _client()
    too_many = http.post("/api/v1/research/plans", json=_plan_body(jobs=5), headers=ALICE)
    assert too_many.status_code == 400 and "max_jobs=4" in too_many.json()["detail"]
    bad_spec = _plan_body(jobs=1)
    bad_spec["jobs"][0]["spec"] = {"name": "x", "execution": {"kind": "nope"}}
    invalid = http.post("/api/v1/research/plans", json=bad_spec, headers=ALICE)
    assert invalid.status_code == 400 and "invalid workload spec" in invalid.json()["detail"]
    bad_budget = http.post(
        "/api/v1/research/plans", json={**_plan_body(), "budget": {**BUDGET, "max_chip_hours": 0}}, headers=ALICE
    )
    assert bad_budget.status_code == 422
    viewer = {**ALICE, "X-Eyvan-Roles": "project_viewer"}
    assert http.post("/api/v1/research/plans", json=_plan_body(), headers=viewer).status_code == 403


def test_a_submit_only_user_cannot_approve_what_the_plane_could_not_stop() -> None:
    http = _client()
    proposed = http.post("/api/v1/research/plans", json=_plan_body(), headers=SUBMITTER)
    assert proposed.status_code == 201, proposed.text
    refused = http.post(f"/api/v1/research/plans/{proposed.json()['id']}/approve", headers=SUBMITTER)
    assert refused.status_code == 403 and "operate" in refused.json()["detail"]
