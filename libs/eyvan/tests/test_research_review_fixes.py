# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Enforce research-plan authority, budget, cancellation, iteration and shutdown boundaries."""

from __future__ import annotations

import asyncio
import json
import typing as tp
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest

from eyvan.config import EyvanSettings
from eyvan.identity.assistant_keys import AssistantKeySealer
from eyvan.records import Principal, Role, RoleGrant, RoleGrantRecord, WorkloadState, utcnow
from eyvan.server.api.app import create_app
from eyvan.server.api.research_runner import ResearchRunner
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.errors import ConflictError, InputError

ORG, PROJECT = "acme", "r"
OPERATOR = RoleGrant(role=Role.PROJECT_OPERATOR, org_id=ORG, project_id=PROJECT)
SUBMITTER = RoleGrant(role=Role.PROJECT_SUBMITTER, org_id=ORG, project_id=PROJECT)
ALICE = Principal(subject="alice", grants=(OPERATOR,))
ADMIN = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _spec(name: str, **extra: tp.Any) -> dict[str, tp.Any]:
    return {"name": name, "execution": {"kind": "command", "command": f"echo {name}"}, **extra}


def _jobs(*names: str) -> list[dict[str, tp.Any]]:
    return [{"name": name, "spec": _spec(name)} for name in names]


def _budget(**overrides: tp.Any) -> dict[str, tp.Any]:
    return {"max_jobs": 4, "max_chip_hours": 100, "max_wall_seconds": 36_000, **overrides}


def _service() -> ControlPlaneService:
    service = ControlPlaneService(InMemoryRepository())
    service.assistant_sealer = AssistantKeySealer(b"k" * 32)  # type: ignore[attr-defined]
    return service


async def _propose(service: ControlPlaneService, principal: Principal = ALICE, **kwargs: tp.Any):
    fields = {"title": "sweep", "org_id": ORG, "project_id": PROJECT, "jobs": _jobs("a"), "budget": _budget()}
    fields.update(kwargs)
    return await service.propose_research_plan(principal, **fields)


async def _workloads(service: ControlPlaneService):
    async with service.repository.transaction() as tx:
        return await tx.list_workloads(org_id=ORG, project_id=PROJECT)


async def _set_state(service: ControlPlaneService, state: WorkloadState) -> None:
    async with service.repository.transaction() as tx:
        for workload in await tx.list_workloads(org_id=ORG, project_id=PROJECT):
            await tx.put_workload(workload.model_copy(update={"state": state}))


async def _plan(service: ControlPlaneService, plan_id: str):
    return await service.get_research_plan(ADMIN, plan_id)


async def _durable_owner(service: ControlPlaneService, grant: RoleGrant = OPERATOR) -> tuple[Principal, RoleGrantRecord]:
    record = RoleGrantRecord(subject="alice", grant=grant, granted_by="root")
    await service.issue_role_grant(ADMIN, record, idempotency_key=f"grant-{grant.role.value}")
    return Principal(subject="alice", grants=await service.active_grants_for("alice")), record


def test_a_plan_whose_owner_lost_authority_is_never_reported_stopped_while_its_job_runs() -> None:
    async def run() -> None:
        service = _service()
        alice, record = await _durable_owner(service)
        plan = await _propose(service, alice)
        await service.approve_research_plan(alice, plan.id)
        t0 = utcnow()
        await service.step_research_plans(t0)
        await _set_state(service, WorkloadState.RUNNING)
        assert await service.revoke_role_grant(ADMIN, record.key)

        await service.step_research_plans(t0 + timedelta(seconds=15))
        await service.step_research_plans(t0 + timedelta(seconds=30))
        current = await _plan(service, plan.id)
        assert current.state == "stopping", "the plan must not claim it stopped a job it could not cancel"
        assert [item.state for item in await _workloads(service)] == [WorkloadState.RUNNING]
        stuck = [note for note in current.notes if note.kind == "cancel_failed"]
        assert len(stuck) == 1 and "still running" in stuck[0].text  # repeated ticks do not flood notes

        # A fleet admin's stop actually cancels it, with the admin's own authority.
        stopped = await service.stop_research_plan(ADMIN, plan.id, reason="owner left")
        assert stopped.state == "failed"
        assert [item.state for item in await _workloads(service)] == [WorkloadState.CANCELLING]
        assert any(note.kind == "cancelled" and "as root" in note.text for note in stopped.notes)

    asyncio.run(run())


def test_an_owner_who_can_submit_but_not_operate_gets_nothing_submitted() -> None:
    async def run() -> None:
        service = _service()
        alice, record = await _durable_owner(service)
        plan = await _propose(service, alice)
        await service.approve_research_plan(alice, plan.id)
        assert await service.revoke_role_grant(ADMIN, record.key)
        await _durable_owner(service, SUBMITTER)
        await service.step_research_plans(utcnow())
        current = await _plan(service, plan.id)
        assert current.state == "failed" and "submit and operate" in (current.stop_reason or "")
        assert await _workloads(service) == []

    asyncio.run(run())


def test_approval_freezes_only_what_the_credential_carried_never_a_revoked_durable_grant() -> None:
    async def run() -> None:
        service = _service()
        stale, record = await _durable_owner(service)
        plan = await _propose(service, stale)
        assert await service.revoke_role_grant(ADMIN, record.key)
        # Authenticated before the revocation committed: the principal still
        # holds the grant, but the credential itself carried none.
        approved = await service.approve_research_plan(stale, plan.id, credential_grants=())
        assert approved.approval_grants == ()
        await service.step_research_plans(utcnow())
        assert (await _plan(service, plan.id)).state == "failed"
        assert await _workloads(service) == []

    asyncio.run(run())


def test_research_jobs_may_not_submit_children_that_would_escape_the_budget() -> None:
    async def run() -> None:
        service = _service()
        with pytest.raises(InputError, match="child workloads"):
            await _propose(service, jobs=[{"name": "fan", "spec": _spec("fan", can_submit_children=True)}])

    asyncio.run(run())


class _Provider:
    def __init__(self, script: list[dict[str, tp.Any]]) -> None:
        self.script = script
        self.requests: list[dict[str, tp.Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        message = self.script.pop(0) if self.script else {"role": "assistant", "content": "done"}
        return httpx.Response(200, json={"choices": [{"message": message}]})


def _tool(name: str, arguments: dict[str, tp.Any]) -> dict[str, tp.Any]:
    call = {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
    return {"role": "assistant", "content": None, "tool_calls": [call]}


async def _iterating(service: ControlPlaneService, transport: httpx.AsyncBaseTransport, **runner: tp.Any):
    provider = await service.create_assistant_provider(
        ADMIN, name="fake", base_url="http://provider.test/v1", models=["m1"], api_key="sk-provider-secret-12345"
    )
    app = SimpleNamespace(state=SimpleNamespace(assistant_transport=transport))
    plan = await _propose(service, iterate=True, provider_id=provider.id)
    await service.approve_research_plan(ALICE, plan.id)
    return plan, ResearchRunner(service, app=tp.cast(tp.Any, app), interval_s=0.01, **runner)


def test_a_model_planned_round_cannot_add_secret_mounts_the_approved_round_did_not_have() -> None:
    async def run() -> None:
        service = _service()
        mounted = {"name": "leak", "spec": _spec("leak", secret_mounts={"TOKEN": "gcp-secret://p/t"})}
        fake = _Provider([_tool("propose_jobs", {"jobs": [mounted, *_jobs("ok")], "rationale": "x"})])
        plan, runner = await _iterating(service, httpx.MockTransport(fake))
        t0 = utcnow()
        await runner.tick(t0)
        await _set_state(service, WorkloadState.SUCCEEDED)
        await runner.tick(t0 + timedelta(seconds=15))
        current = await _plan(service, plan.id)
        assert [job.name for job in current.jobs] == ["a", "ok"]
        assert any(note.kind == "rejected" and "mounts secrets" in note.text for note in current.notes)

    asyncio.run(run())


def test_an_iteration_call_is_bounded_and_counts_as_one_failure() -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "late"}}]})

    async def run() -> None:
        service = _service()
        plan, runner = await _iterating(service, httpx.MockTransport(slow), iteration_timeout_s=0.05)
        t0 = utcnow()
        await runner.tick(t0)
        await _set_state(service, WorkloadState.SUCCEEDED)
        started = asyncio.get_running_loop().time()
        await runner.tick(t0 + timedelta(seconds=15))
        assert asyncio.get_running_loop().time() - started < 2
        current = await _plan(service, plan.id)
        assert current.state == "running" and current.iteration_errors == 1
        assert any("did not answer within" in note.text for note in current.notes)

    asyncio.run(run())


def test_a_replica_that_lost_leadership_stops_stepping_plans() -> None:
    async def run() -> None:
        service = _service()
        plan = await _propose(service)
        await service.approve_research_plan(ALICE, plan.id)

        async def lost() -> bool:
            return False

        await service.step_research_plans(utcnow(), still_leading=lost)
        assert await _workloads(service) == []
        assert (await _plan(service, plan.id)).state == "approved"

    asyncio.run(run())


def test_a_deferred_late_cancel_reopens_the_stop_instead_of_orphaning_the_workload() -> None:
    async def run() -> None:
        service = _service()
        plan = await _propose(service)
        await service.approve_research_plan(ALICE, plan.id)
        original_submit = service.submit_workload
        original_cancel = service.cancel_workload

        async def submit_then_get_stopped(*args: tp.Any, **kwargs: tp.Any):
            result = await original_submit(*args, **kwargs)
            # A stop lands while this step is still submitting.
            async with service.repository.transaction() as tx:
                current = await tx.get_research_plan(plan.id)
                await tx.put_research_plan(
                    current.model_copy(update={"state": "stopped", "finished_at": utcnow(), "updated_at": utcnow()})
                )
            return result

        async def busy(*args: tp.Any, **kwargs: tp.Any):
            raise ConflictError("a preemption is in flight")

        service.submit_workload = submit_then_get_stopped  # type: ignore[method-assign]
        service.cancel_workload = busy  # type: ignore[method-assign]
        t0 = utcnow()
        await service.step_research_plans(t0)
        reopened = await _plan(service, plan.id)
        assert reopened.state == "stopping" and reopened.stop_target == "stopped"
        assert [item.state for item in await _workloads(service)] == [WorkloadState.PENDING]

        service.cancel_workload = original_cancel  # type: ignore[method-assign]
        await service.step_research_plans(t0 + timedelta(seconds=15))
        assert (await _plan(service, plan.id)).state == "stopped"
        assert [item.state for item in await _workloads(service)] == [WorkloadState.CANCELLED]

    asyncio.run(run())


def test_shutdown_waits_for_in_flight_assistant_saves_before_closing_the_store() -> None:
    async def run() -> None:
        app = create_app(
            settings=EyvanSettings(dev_auth_enabled=True, artifact_token_key="k" * 32, controller_enabled=False)
        )
        landed: list[bool] = []

        async def save() -> None:
            await asyncio.sleep(0.2)
            landed.append(True)

        async with app.router.lifespan_context(app):
            app.state.assistant_pending_saves = {asyncio.ensure_future(save())}
        assert landed == [True]

    asyncio.run(run())
