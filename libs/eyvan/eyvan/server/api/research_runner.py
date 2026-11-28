# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The research-plan runner: a leader-only loop that steps approved plans and asks the model for new rounds.

The deterministic part -- submitting as the owner, accounting chip-hours,
enforcing the budget, cancelling on stop -- lives in the service
(:meth:`ControlPlaneService.step_research_plans`). This module adds only what the
service may not import: the assistant provider machinery used to plan the next
round. The model sees a compact summary and has exactly two tools,
``propose_jobs`` and ``finish``; whatever it proposes is validated and capped by
the remaining budget in the service, and it never submits anything itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import typing as tp
from datetime import datetime
from types import SimpleNamespace

from fastapi import FastAPI, Request

from eyvan.records import MAX_RESEARCH_JOBS, Principal, ResearchPlanRecord, utcnow
from eyvan.server.api.assistant import chat_model_for
from eyvan.server.api.assistant_stream import collect_turn, run_turn_stream
from eyvan.server.service._control_plane.research import research_job
from eyvan.server.service.errors import InputError, NotFoundError, UnavailableError

if tp.TYPE_CHECKING:
    from eyvan.server.service import ControlPlaneService

logger = logging.getLogger(__name__)

#: How often the runner steps active plans. A stop request does not wait for
#: this: the stop route cancels the plan's workloads itself.
DEFAULT_RESEARCH_INTERVAL_S = 15.0
#: Wall-clock bound on one iteration call; a timeout counts as one failed iteration.
DEFAULT_ITERATION_TIMEOUT_S = 180.0
#: Provider rounds one iteration may take (bounded again by the provider's own max_steps).
MAX_ITERATION_STEPS = 4
#: Upper bound on the summary sent to the model.
MAX_ITERATION_PROMPT_CHARS = 24_000

ITERATION_SYSTEM_PROMPT = (
    "You run an approved research plan on Eyvan for its owner. The plane has run the jobs below; read their "
    "states, failure reasons and progress metrics, then decide the next step. Call propose_jobs with a small "
    "next round of jobs (each {name, spec} where spec is an Eyvan workload spec like the ones shown) and a "
    "short rationale, or call finish with a short summary when the goal is met or nothing useful remains. "
    "Stay inside the remaining budget; the plane drops anything beyond it. Never repeat a job that already "
    "failed for the same reason."
)

_JOB_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "spec": {"type": "object"}},
    "required": ["name", "spec"],
}
ITERATION_TOOL_SPECS: tuple[dict[str, tp.Any], ...] = (
    {
        "name": "propose_jobs",
        "description": "Plan the next round: workload specs to run, within the remaining budget.",
        "parameters": {
            "type": "object",
            "properties": {
                "jobs": {"type": "array", "items": _JOB_SCHEMA, "minItems": 1},
                "rationale": {"type": "string"},
            },
            "required": ["jobs", "rationale"],
            "additionalProperties": False,
        },
    },
    {
        "name": "finish",
        "description": "End the research plan with a short summary of what was learned.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
)


class _Leader(tp.Protocol):
    async def acquire(self) -> bool: ...


class _IterationTools:
    """The two tools of one iteration turn; records the first valid decision."""

    def __init__(self, round_index: int) -> None:
        self.drafts: list[dict[str, tp.Any]] = []
        self.requests: list[tp.Any] = []
        self.outcome: dict[str, tp.Any] | None = None
        self._round = round_index

    async def call(self, name: str, arguments: dict[str, tp.Any]) -> tuple[str, bool, tp.Any]:
        if self.outcome is not None:
            return name, False, {"error": "a decision was already recorded; reply with a one-line summary"}
        if name == "finish":
            self.outcome = {"finish": str(arguments.get("summary") or "")[:1500]}
            return name, True, {"finished": True}
        if name != "propose_jobs":
            return name, False, {"error": f"unknown tool {name!r}; use propose_jobs or finish"}
        raw = arguments.get("jobs")
        if not isinstance(raw, list) or not raw:
            return name, False, {"error": "jobs must be a non-empty list of {name, spec}"}
        valid: list[dict[str, tp.Any]] = []
        errors: list[str] = []
        for item in raw[:MAX_RESEARCH_JOBS]:
            try:
                research_job(item, round_index=self._round)
            except InputError as exc:
                errors.append(str(exc)[:300])
            else:
                valid.append(item)
        if not valid:
            # Let the model correct itself within the same turn.
            return name, False, {"error": "no proposed job had a valid workload spec", "details": errors[:5]}
        self.outcome = {"jobs": valid, "rationale": str(arguments.get("rationale") or "")[:1000]}
        return name, True, {"accepted_for_review": len(valid), "rejected": errors[:5]}


def iteration_prompt(plan: ResearchPlanRecord, summary: list[dict[str, tp.Any]]) -> str:
    """The compact, bounded user message describing the plan and every job's outcome."""

    room = plan.budget.max_jobs - len(plan.jobs)
    document = {
        "title": plan.title,
        "goal": plan.goal,
        "round_finished": plan.current_round + 1,
        "budget": plan.budget.model_dump(mode="json"),
        "spent": plan.spent.model_dump(mode="json"),
        "jobs_you_may_still_add": max(room, 0),
        "jobs": summary,
    }
    text = json.dumps(document, default=str, separators=(",", ":"))
    if len(text) > MAX_ITERATION_PROMPT_CHARS:
        text = text[:MAX_ITERATION_PROMPT_CHARS] + "...(truncated)"
    return text


class ResearchRunner:
    """Steps active research plans on an interval, only while this replica leads.

    ``leader`` is the controller's leader elector, so a multi-replica plane
    (PostgreSQL advisory lock) runs plans on exactly one replica. ``interval_s``
    and ``clock`` are constructor arguments, not environment knobs.
    """

    def __init__(
        self,
        service: ControlPlaneService,
        *,
        app: FastAPI | None = None,
        leader: _Leader | None = None,
        interval_s: float = DEFAULT_RESEARCH_INTERVAL_S,
        clock: tp.Callable[[], datetime] = utcnow,
        iteration_timeout_s: float = DEFAULT_ITERATION_TIMEOUT_S,
        read_logs: tp.Callable[[str], tp.Awaitable[str]] | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("the research runner interval must be positive")
        if iteration_timeout_s <= 0:
            raise ValueError("the iteration timeout must be positive")
        self.iteration_timeout_s = iteration_timeout_s
        self.service = service
        self.app = app
        self.leader = leader
        self.interval_s = interval_s
        self.clock = clock
        #: Reads a final job's output so the plan records the loss it printed.
        self.read_logs = read_logs

    async def tick(self, now: datetime | None = None) -> int:
        """Step every active plan once; return how many were looked at."""

        leader = self.leader
        if now is None and self.clock is not utcnow:
            now = self.clock()  # an injected clock (tests) pins the whole tick
        return await self.service.step_research_plans(
            now,
            iterate=self.iterate,
            # Re-checked before every plan, so a replica that lost leadership
            # during a slow iteration call stops stepping plans at once.
            still_leading=leader.acquire if leader is not None else None,
            read_logs=self.read_logs,
        )

    async def run(self, stop: asyncio.Event) -> None:
        """Tick until ``stop`` is set; a failing tick is logged and the loop goes on."""

        while not stop.is_set():
            try:
                if self.leader is None or await self.leader.acquire():
                    await self.tick()
            except Exception:
                logger.exception("research runner tick failed; retrying on the next interval")
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.interval_s)
            except TimeoutError:
                pass

    async def iterate(
        self, plan: ResearchPlanRecord, principal: Principal, summary: list[dict[str, tp.Any]]
    ) -> dict[str, tp.Any] | None:
        """Ask the plan's provider, as its owner, for the next round or a finish."""

        try:
            record, model, key = await self.service.resolve_assistant_provider(
                principal, provider_id=plan.provider_id, model=plan.model
            )
        except (NotFoundError, InputError, UnavailableError) as exc:
            return {"unavailable": str(exc)[:300]}
        # chat_model_for reads only ``request.app.state`` (the test transport and
        # Claude Code executable seams); there is no request in a background loop.
        shim = tp.cast(Request, SimpleNamespace(app=self.app or SimpleNamespace(state=SimpleNamespace())))
        chat = chat_model_for(record, model, key, shim)
        tools = _IterationTools(plan.current_round + 1)
        events = run_turn_stream(
            chat,
            tools,
            [{"role": "user", "content": iteration_prompt(plan, summary)}],
            system=ITERATION_SYSTEM_PROMPT,
            tool_specs=list(ITERATION_TOOL_SPECS),
            max_steps=min(record.max_steps, MAX_ITERATION_STEPS),
        )
        try:
            # Bounded: plans are stepped one after another, so an unbounded
            # provider call would delay every other plan's budget checks.
            turn = await asyncio.wait_for(collect_turn(events), timeout=self.iteration_timeout_s)
        except TimeoutError:
            raise TimeoutError(f"the iteration model did not answer within {self.iteration_timeout_s:g}s") from None
        if tools.outcome is None:
            reply = str(turn.get("reply") or "").strip()[:500]
            return {"finish": f"the model made no decision ({reply or 'no answer'})"}
        return tools.outcome


__all__ = (
    "DEFAULT_ITERATION_TIMEOUT_S",
    "DEFAULT_RESEARCH_INTERVAL_S",
    "ITERATION_SYSTEM_PROMPT",
    "ITERATION_TOOL_SPECS",
    "ResearchRunner",
    "iteration_prompt",
)
