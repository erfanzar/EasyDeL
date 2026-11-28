# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The research runner: acts as the owner, never double-submits, enforces the budget, stops within a tick.

Time is injected (``step_research_plans(now)``), workload outcomes and leases are
written straight into the in-memory store, and the iteration model is a scripted
OpenAI-compatible fake behind ``httpx.MockTransport``. Nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json
import typing as tp
from datetime import timedelta
from types import SimpleNamespace

import httpx

from eyvan.identity.assistant_keys import AssistantKeySealer
from eyvan.records import (
    LeaseRecord,
    LeaseState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    RoleGrantRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.api.research_runner import ResearchRunner
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service._control_plane.research import lease_chip_hours
from eyvan.server.service.control_plane import ControlPlaneService

ORG, PROJECT = "acme", "r"
OPERATOR = RoleGrant(role=Role.PROJECT_OPERATOR, org_id=ORG, project_id=PROJECT)
ALICE = Principal(subject="alice", grants=(OPERATOR,))
ADMIN = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _spec(name: str) -> dict[str, tp.Any]:
    return {"name": name, "execution": {"kind": "command", "command": f"echo {name}"}}


def _jobs(*names: str) -> list[dict[str, tp.Any]]:
    return [{"name": name, "spec": _spec(name)} for name in names]


def _budget(**overrides: tp.Any) -> dict[str, tp.Any]:
    return {"max_jobs": 4, "max_chip_hours": 100, "max_wall_seconds": 36_000, **overrides}


def _service() -> ControlPlaneService:
    service = ControlPlaneService(InMemoryRepository())
    service.assistant_sealer = AssistantKeySealer(b"k" * 32)  # type: ignore[attr-defined]
    return service


async def _approved(service: ControlPlaneService, principal: Principal = ALICE, **kwargs: tp.Any):
    fields = {"title": "sweep", "org_id": ORG, "project_id": PROJECT, "jobs": _jobs("a", "b"), "budget": _budget()}
    fields.update(kwargs)
    plan = await service.propose_research_plan(principal, **fields)
    return await service.approve_research_plan(principal, plan.id)


async def _workloads(service: ControlPlaneService):
    async with service.repository.transaction() as tx:
        return await tx.list_workloads(org_id=ORG, project_id=PROJECT)


async def _set_state(service: ControlPlaneService, state: WorkloadState, *ids: str) -> None:
    async with service.repository.transaction() as tx:
        for workload in await tx.list_workloads(org_id=ORG, project_id=PROJECT):
            if not ids or workload.id in ids:
                await tx.put_workload(workload.model_copy(update={"state": state}))


async def _lease(service: ControlPlaneService, workload_id: str, started, *, chips: int = 4) -> None:
    lease = LeaseRecord(
        id=f"lease-{workload_id}",
        workload_id=workload_id,
        cell_id="cell-1",
        org_id=ORG,
        project_id=PROJECT,
        state=LeaseState.ACTIVE,
        resources=ResourceGang(per_replica=ResourceVector(tpu=chips)),
        fencing_token="fence",
        expires_at=started + timedelta(days=30),
        created_at=started,
        run_started_at=started,
    )
    async with service.repository.transaction() as tx:
        await tx.put_lease(lease)


async def _plan(service: ControlPlaneService, plan_id: str):
    return await service.get_research_plan(ADMIN, plan_id)


def test_runner_submits_as_the_owner_and_finishes_when_jobs_end() -> None:
    async def run() -> None:
        service = _service()
        plan = await _approved(service)
        t0 = utcnow()
        assert await service.step_research_plans(t0) == 1
        workloads = await _workloads(service)
        assert len(workloads) == 2 and {item.created_by for item in workloads} == {"alice"}
        current = await _plan(service, plan.id)
        assert current.state == "running" and current.spent.jobs_submitted == 2
        assert [item.workload_id for item in current.submitted] == [item.id for item in workloads]

        await service.step_research_plans(t0 + timedelta(seconds=15))
        assert (await _plan(service, plan.id)).state == "running"
        await _set_state(service, WorkloadState.SUCCEEDED)
        await service.step_research_plans(t0 + timedelta(seconds=30))
        finished = await _plan(service, plan.id)
        assert finished.state == "finished" and finished.finished_at is not None
        assert len(await _workloads(service)) == 2
        # Nothing to do once every plan is terminal: the tick reads and returns.
        assert await service.step_research_plans(t0 + timedelta(seconds=45)) == 0
        async with service.repository.transaction() as tx:
            actions = [event.action for event in await tx.list_events(entity_type="research_plan", limit=50)]
        assert actions[:3] == ["research_plan.proposed", "research_plan.approved", "research_plan.submitted"]
        assert "research_plan.started" in actions and actions[-1] == "research_plan.finished"

    asyncio.run(run())


def test_a_crash_between_submitting_and_recording_never_double_submits() -> None:
    async def run() -> None:
        service = _service()
        plan = await _approved(service)
        original = service._commit_research_step
        calls = {"n": 0}

        async def crash_once(*args: tp.Any, **kwargs: tp.Any):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("plane died after submitting")
            return await original(*args, **kwargs)

        service._commit_research_step = crash_once  # type: ignore[method-assign]
        t0 = utcnow()
        await service.step_research_plans(t0)
        assert len(await _workloads(service)) == 2
        assert (await _plan(service, plan.id)).submitted == ()
        # A "restarted" plane: a fresh service over the same store steps again.
        restarted = ControlPlaneService(service.repository)
        await restarted.step_research_plans(t0 + timedelta(seconds=15))
        workloads = await _workloads(service)
        assert len(workloads) == 2
        current = await _plan(service, plan.id)
        assert sorted(item.workload_id for item in current.submitted) == sorted(item.id for item in workloads)
        assert sum("(replayed)" in note.text for note in current.notes) == 2

    asyncio.run(run())


def test_chip_hours_over_budget_cancels_running_jobs() -> None:
    async def run() -> None:
        service = _service()
        plan = await _approved(service, budget=_budget(max_chip_hours=2))
        t0 = utcnow()
        await service.step_research_plans(t0)
        workloads = await _workloads(service)
        await _set_state(service, WorkloadState.RUNNING)
        await _lease(service, workloads[0].id, t0)
        # 4 chips for 20 minutes: 1.33 chip-hours, still inside the budget.
        await service.step_research_plans(t0 + timedelta(minutes=20))
        assert (await _plan(service, plan.id)).state == "running"
        # 4 chips for 31 minutes: 2.07 >= 2 chip-hours.
        await service.step_research_plans(t0 + timedelta(minutes=31))
        current = await _plan(service, plan.id)
        assert current.state == "over_budget" and "chip-hour budget" in (current.stop_reason or "")
        assert current.spent.chip_hours >= 2
        assert {item.state for item in await _workloads(service)} == {WorkloadState.CANCELLING}

    asyncio.run(run())


def test_lease_chip_hours_never_undercounts_a_settled_lease() -> None:
    t0 = utcnow()
    lease = LeaseRecord(
        id="l1",
        workload_id="w1",
        cell_id="c1",
        org_id=ORG,
        project_id=PROJECT,
        state=LeaseState.RELEASED,
        resources=ResourceGang(replicas=2, per_replica=ResourceVector(tpu=4)),
        fencing_token="f",
        expires_at=t0 + timedelta(days=1),
        created_at=t0,
        settled_at=t0 + timedelta(hours=1),
        settlement_reason="done",
        accelerator_hours_settled={"tpu": 3.0},
    )
    # Settled figure 3, held time 1h x 8 chips = 8: the larger wins, whatever now is.
    assert lease_chip_hours(lease, t0 + timedelta(days=5)) == 8.0
    assert lease_chip_hours(lease.model_copy(update={"accelerator_hours_settled": {"tpu": 9.0}}), t0) == 9.0
    active = lease.model_copy(
        update={
            "state": LeaseState.ACTIVE,
            "settled_at": None,
            "settlement_reason": None,
            "accelerator_hours_settled": {},
        }
    )
    assert lease_chip_hours(active, t0 + timedelta(minutes=30)) == 4.0


def test_the_wall_clock_deadline_cancels_and_ends_over_budget() -> None:
    async def run() -> None:
        service = _service()
        plan = await _approved(service, budget=_budget(max_wall_seconds=600))
        t0 = utcnow()
        await service.step_research_plans(t0)
        await _set_state(service, WorkloadState.RUNNING)
        await service.step_research_plans(t0 + timedelta(seconds=599))
        assert (await _plan(service, plan.id)).state == "running"
        await service.step_research_plans(t0 + timedelta(seconds=600))
        current = await _plan(service, plan.id)
        assert current.state == "over_budget" and "wall-clock" in (current.stop_reason or "")
        assert {item.state for item in await _workloads(service)} == {WorkloadState.CANCELLING}

    asyncio.run(run())


def test_stop_cancels_running_jobs_at_once_and_a_stopping_plan_settles_in_one_tick() -> None:
    async def run() -> None:
        service = _service()
        plan = await _approved(service)
        t0 = utcnow()
        await service.step_research_plans(t0)
        await _set_state(service, WorkloadState.RUNNING)
        stopped = await service.stop_research_plan(ALICE, plan.id, reason="enough")
        assert stopped.state == "stopped" and stopped.stop_reason == "enough"
        assert {item.state for item in await _workloads(service)} == {WorkloadState.CANCELLING}
        async with service.repository.transaction() as tx:
            outbox = [item for item in tx._state.outbox.values() if item.topic == "workload.cancel"]  # type: ignore[attr-defined]
        assert len(outbox) == 2

        # A plan left in ``stopping`` (e.g. the stop request's own cancel was
        # interrupted) is finished by the next tick, as its owner.
        other = await _approved(service, jobs=_jobs("c"))
        await service.step_research_plans(t0 + timedelta(seconds=15))
        [running] = [item for item in await _workloads(service) if item.spec.name == "c"]
        await _set_state(service, WorkloadState.RUNNING, running.id)
        async with service.repository.transaction() as tx:
            current = await tx.get_research_plan(other.id)
            await tx.put_research_plan(
                current.model_copy(update={"state": "stopping", "stop_target": "stopped", "stop_reason": "x"})
            )
        await service.step_research_plans(t0 + timedelta(seconds=30))
        assert (await _plan(service, other.id)).state == "stopped"
        [cancelled] = [item for item in await _workloads(service) if item.id == running.id]
        assert cancelled.state is WorkloadState.CANCELLING

    asyncio.run(run())


def test_a_revoked_owner_gets_no_new_jobs_submitted() -> None:
    async def run() -> None:
        service = _service()
        record = RoleGrantRecord(subject="alice", grant=OPERATOR, granted_by="root")
        await service.issue_role_grant(ADMIN, record, idempotency_key="grant-alice")
        # Authenticated the way the API builds a principal: durable grants only.
        durable = Principal(subject="alice", grants=await service.active_grants_for("alice"))
        plan = await _approved(service, durable)
        assert plan.approval_grants == ()  # durable authority is re-read, never frozen into the plan
        assert await service.revoke_role_grant(ADMIN, record.key)
        await service.step_research_plans(utcnow())
        current = await _plan(service, plan.id)
        assert current.state == "failed" and "no longer allowed" in (current.stop_reason or "")
        assert await _workloads(service) == []

    asyncio.run(run())


def test_the_runner_never_uses_admin_rights_even_for_an_admin_owner() -> None:
    async def run() -> None:
        service = _service()
        plan = await _approved(service, ADMIN)
        assert plan.approval_grants == (OPERATOR,)
        principal = await service._research_principal(plan, utcnow())
        assert principal.grants == (OPERATOR,)
        assert not service.authorizer.allows(principal, __import__("eyvan.records").records.Permission.MANAGE_FLEET)

    asyncio.run(run())


class _Provider:
    """A scripted OpenAI-compatible provider; records every request it gets."""

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


def test_iteration_proposes_a_second_round_capped_by_the_budget() -> None:
    async def run() -> None:
        service = _service()
        provider = await service.create_assistant_provider(
            ADMIN, name="fake", base_url="http://provider.test/v1", models=["m1"], api_key="sk-provider-secret-12345"
        )
        fake = _Provider(
            [
                _tool("propose_jobs", {"jobs": _jobs("c", "d", "e"), "rationale": "try lower lr"}),
                {"role": "assistant", "content": "planned"},
            ]
        )
        app = SimpleNamespace(state=SimpleNamespace(assistant_transport=httpx.MockTransport(fake)))
        runner = ResearchRunner(service, app=tp.cast(tp.Any, app), interval_s=0.01)
        plan = await _approved(
            service, jobs=_jobs("a"), budget=_budget(max_jobs=3), iterate=True, provider_id=provider.id
        )
        t0 = utcnow()
        await runner.tick(t0)
        await _set_state(service, WorkloadState.FAILED)
        await runner.tick(t0 + timedelta(seconds=15))
        current = await _plan(service, plan.id)
        assert current.current_round == 1 and [job.name for job in current.jobs] == ["a", "c", "d"]
        assert any("dropped 1 proposed job" in note.text for note in current.notes)
        sent = fake.requests[0]
        assert [tool["function"]["name"] for tool in sent["tools"]] == ["propose_jobs", "finish"]
        summary = json.loads(sent["messages"][1]["content"])
        assert summary["jobs"][0]["state"] == "failed" and summary["jobs_you_may_still_add"] == 2

        await runner.tick(t0 + timedelta(seconds=30))
        workloads = await _workloads(service)
        assert len(workloads) == 3 and {item.created_by for item in workloads} == {"alice"}
        await _set_state(service, WorkloadState.SUCCEEDED)
        await runner.tick(t0 + timedelta(seconds=45))
        done = await _plan(service, plan.id)
        assert done.state == "finished" and done.spent.jobs_submitted == 3 and done.spent.rounds == 2
        assert len(fake.requests) == 2  # no third call: the budget left no room for another round
        assert len(await _workloads(service)) == 3

    asyncio.run(run())


def test_iterate_without_a_provider_finishes_after_the_initial_jobs() -> None:
    async def run() -> None:
        service = _service()
        runner = ResearchRunner(service, interval_s=0.01)
        plan = await _approved(service, iterate=True)
        t0 = utcnow()
        await runner.tick(t0)
        await _set_state(service, WorkloadState.SUCCEEDED)
        await runner.tick(t0 + timedelta(seconds=15))
        current = await _plan(service, plan.id)
        assert current.state == "finished" and "cannot iterate" in current.notes[-1].text
        assert len(await _workloads(service)) == 2

    asyncio.run(run())


def test_the_runner_loop_only_ticks_while_it_leads_and_stops_promptly() -> None:
    class Follower:
        async def acquire(self) -> bool:
            return False

    async def run() -> None:
        service = _service()
        await _approved(service)
        stop = asyncio.Event()
        follower = ResearchRunner(service, leader=Follower(), interval_s=0.01)
        task = asyncio.create_task(follower.run(stop))
        await asyncio.sleep(0.05)
        assert await _workloads(service) == []
        stop.set()
        await asyncio.wait_for(task, 1)

        stop = asyncio.Event()
        leader = ResearchRunner(service, interval_s=0.01)
        task = asyncio.create_task(leader.run(stop))
        for _ in range(100):
            if len(await _workloads(service)) == 2:
                break
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, 1)
        assert len(await _workloads(service)) == 2

    asyncio.run(run())
