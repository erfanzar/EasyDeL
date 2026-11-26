# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Research-mode HTTP routes: propose, approve, inspect and stop a budgeted research plan.

Every route is owner-scoped: a plan belongs to the person who proposed it, and
someone else's plan answers 404 exactly like one that does not exist. Fleet
admins (``MANAGE_FLEET``) may list, read and stop every plan, but only the owner
approves one, and the plane always runs it as the owner, never as the admin.
"""

from __future__ import annotations

import typing as tp

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from eyvan.config import EyvanSettings
from eyvan.records import (
    MAX_RESEARCH_GOAL_CHARS,
    MAX_RESEARCH_JOBS,
    Principal,
    ResearchBudget,
    ResearchPlanRecord,
    WorkloadRecord,
)
from eyvan.server.api.auth import authenticated_principal

if tp.TYPE_CHECKING:
    from eyvan.server.service import ControlPlaneService

#: Module level, not a closure: with postponed annotations FastAPI resolves
#: dependency annotations against module globals.
_Principal = tp.Annotated[Principal, Depends(authenticated_principal)]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProposeResearchBody(_Body):
    title: str = Field(min_length=1, max_length=200)
    goal: str = Field(default="", max_length=MAX_RESEARCH_GOAL_CHARS)
    org_id: str = Field(default="default", min_length=1, max_length=128)
    project_id: str = Field(default="default", min_length=1, max_length=128)
    #: Each ``{name, spec}``; ``spec`` is the workload document ``POST /api/v1/workloads`` accepts.
    jobs: list[dict[str, tp.Any]] = Field(min_length=1, max_length=MAX_RESEARCH_JOBS)
    budget: ResearchBudget
    iterate: bool = False
    provider_id: str | None = Field(default=None, max_length=32)
    model: str | None = Field(default=None, max_length=128)


class StopResearchBody(_Body):
    reason: str = Field(default="", max_length=2000)


def _iso(value: tp.Any) -> str | None:
    return None if value is None else value.isoformat()


def plan_summary(plan: ResearchPlanRecord) -> dict[str, tp.Any]:
    """The compact list view of one plan: lifecycle, budget and spend, without job specs or notes."""

    return {
        "id": plan.id,
        "title": plan.title,
        "subject": plan.subject,
        "org_id": plan.org_id,
        "project_id": plan.project_id,
        "state": plan.state,
        "iterate": plan.iterate,
        "provider_id": plan.provider_id,
        "model": plan.model,
        "current_round": plan.current_round,
        "jobs_planned": len(plan.jobs),
        "budget": plan.budget.model_dump(mode="json"),
        "spent": plan.spent.model_dump(mode="json"),
        "stop_reason": plan.stop_reason,
        "approved_at": _iso(plan.approved_at),
        "approved_by": plan.approved_by,
        "started_at": _iso(plan.started_at),
        "finished_at": _iso(plan.finished_at),
        "created_at": _iso(plan.created_at),
        "updated_at": _iso(plan.updated_at),
    }


def plan_document(plan: ResearchPlanRecord, workloads: tp.Mapping[str, WorkloadRecord]) -> dict[str, tp.Any]:
    """One plan in full: every job with its workload's current state, spend against budget, and notes."""

    submitted = {item.job_index: item for item in plan.submitted}
    jobs: list[dict[str, tp.Any]] = []
    for index, job in enumerate(plan.jobs):
        row: dict[str, tp.Any] = {"index": index, "name": job.name, "round": job.round, "spec": job.spec}
        item = submitted.get(index)
        if item is None:
            row["state"] = "not_submitted"
        elif item.workload_id is None:
            row.update(state="refused", error=item.error, submitted_at=_iso(item.submitted_at))
        else:
            workload = workloads.get(item.workload_id)
            row.update(
                workload_id=item.workload_id,
                state=workload.state.value if workload is not None else "unknown",
                failure_reason=workload.failure_reason if workload is not None else None,
                submitted_at=_iso(item.submitted_at),
            )
            if item.loss is not None:
                row["loss"] = {
                    "final": item.loss.final,
                    "last_step": item.loss.last_step,
                    "total_steps": item.loss.total_steps,
                    "points": [list(point) for point in item.loss.points],
                }
        jobs.append(row)
    return {
        **plan_summary(plan),
        "goal": plan.goal,
        "jobs": jobs,
        "workload_ids": [item.workload_id for item in plan.submitted if item.workload_id is not None],
        "remaining": {
            "jobs": max(plan.budget.max_jobs - plan.spent.jobs_submitted, 0),
            "chip_hours": max(plan.budget.max_chip_hours - plan.spent.chip_hours, 0.0),
        },
        "notes": [{"t": _iso(note.t), "kind": note.kind, "text": note.text} for note in plan.notes],
    }


def register_research_routes(app: FastAPI, *, settings: EyvanSettings, service: ControlPlaneService) -> None:
    """Mount the research-plan routes (owner-scoped; fleet admins see and stop all)."""

    del settings

    @app.post("/api/v1/research/plans", status_code=201, tags=["research"])
    async def propose_research_plan(body: ProposeResearchBody, principal: _Principal) -> dict[str, tp.Any]:
        """Propose a plan; every job spec is validated now. Nothing runs until its owner approves it."""

        plan = await service.propose_research_plan(
            principal,
            title=body.title,
            goal=body.goal,
            org_id=body.org_id,
            project_id=body.project_id,
            jobs=body.jobs,
            budget=body.budget,
            iterate=body.iterate,
            provider_id=body.provider_id,
            model=body.model,
        )
        return plan_document(plan, {})

    @app.get("/api/v1/research/plans", tags=["research"])
    async def list_research_plans(principal: _Principal, state: str | None = None) -> dict[str, tp.Any]:
        """Your plans (fleet admins see everyone's), oldest first."""

        plans = await service.list_research_plans(principal, state=state)
        return {"plans": [plan_summary(plan) for plan in plans]}

    @app.get("/api/v1/research/plans/{plan_id}", tags=["research"])
    async def get_research_plan(plan_id: str, principal: _Principal) -> dict[str, tp.Any]:
        """One plan: its jobs and their workloads' states, spend against budget, and notes."""

        plan, workloads = await service.describe_research_plan(principal, plan_id)
        return plan_document(plan, workloads)

    @app.post("/api/v1/research/plans/{plan_id}/approve", tags=["research"])
    async def approve_research_plan(plan_id: str, request: Request, principal: _Principal) -> dict[str, tp.Any]:
        """Approve a proposed plan (owner only); the plane then submits and iterates it as you."""

        await service.approve_research_plan(
            principal, plan_id, credential_grants=getattr(request.state, "credential_grants", None)
        )
        plan, workloads = await service.describe_research_plan(principal, plan_id)
        return plan_document(plan, workloads)

    @app.post("/api/v1/research/plans/{plan_id}/stop", tags=["research"])
    async def stop_research_plan(
        plan_id: str, principal: _Principal, body: StopResearchBody | None = None
    ) -> dict[str, tp.Any]:
        """Stop a plan (owner or fleet admin); its running workloads are cancelled now."""

        await service.stop_research_plan(principal, plan_id, reason=body.reason if body is not None else "")
        plan, workloads = await service.describe_research_plan(principal, plan_id)
        return plan_document(plan, workloads)


__all__ = ("plan_document", "plan_summary", "register_research_routes")
