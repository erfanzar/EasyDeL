# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Budgeted research plans: approve jobs plus a budget once; the plane runs and iterates them as that user.

A plan is proposed (validated now), approved once by its owner, and then driven
by :meth:`ResearchPlansMixin.step_research_plans`, which a leader-only runner
calls on an interval. Each step acts strictly as the plan's owner: the
principal is rebuilt every time from the owner's *current* durable grants plus
the narrowed authority the approving request carried, then narrowed again to
submit/operate in the plan's own project. A revoked grant or a disabled account
therefore stops new submissions on the next step, and no step ever holds admin
rights.

The budget is enforced here, deterministically, never by a model: a plan never
plans more jobs than ``max_jobs``; once the conservative chip-hour estimate or
the wall-clock deadline is reached every still-running workload is cancelled
and the plan ends ``over_budget``. Iteration is a callback (the API-side
runner talks to the assistant provider), so this service stays transport-free.
"""

from __future__ import annotations

import logging
import secrets
import typing as tp
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from pydantic import ValidationError

from eyvan.records import (
    MAX_RESEARCH_JOBS,
    MAX_RESEARCH_NOTE_CHARS,
    MAX_RESEARCH_NOTES,
    MAX_RESEARCH_ROUNDS,
    RESEARCH_PLAN_ACTIVE_STATES,
    RESEARCH_PLAN_TERMINAL_STATES,
    ResearchBudget,
    ResearchJob,
    ResearchLoss,
    ResearchNote,
    ResearchPlanRecord,
    ResearchSpend,
    ResearchSubmission,
)
from eyvan.server.service._control_plane.shared import (
    ACTIVE_LEASE_STATES,
    TERMINAL_WORKLOAD_STATES,
    AuditEvent,
    AuthorizationError,
    ConflictError,
    InputError,
    LeaseRecord,
    NotFoundError,
    Permission,
    Principal,
    Role,
    RoleGrant,
    UnavailableError,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    _new_id,
    utcnow,
)
from eyvan.server.service.errors import EyvanServiceError
from eyvan.server.service.research_losses import ResearchLogReader, read_research_losses

logger = logging.getLogger(__name__)

#: Plans one person may hold before they are approved, finished or stopped.
MAX_OPEN_RESEARCH_PLANS_PER_SUBJECT = 20
#: Consecutive iteration-model failures after which a plan finishes instead of retrying.
MAX_RESEARCH_ITERATION_ERRORS = 3
_OPEN_STATES = frozenset({"proposed", *RESEARCH_PLAN_ACTIVE_STATES})

#: ``(plan, owner principal, per-job summary) -> outcome``. The outcome is
#: ``{"jobs": [{name, spec}], "rationale": str}``, ``{"finish": summary}``,
#: ``{"unavailable": reason}`` or ``None`` (no model available); raising counts
#: as one failed iteration.
ResearchIterator: tp.TypeAlias = tp.Callable[
    [ResearchPlanRecord, Principal, list[dict[str, tp.Any]]], tp.Awaitable[dict[str, tp.Any] | None]
]


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _read_view(repository: tp.Any) -> tp.AsyncContextManager[tp.Any]:
    """A read that never queues behind a commit: the committed snapshot where the store offers one."""

    snapshot = getattr(repository, "snapshot", None)
    return tp.cast(tp.AsyncContextManager[tp.Any], snapshot() if snapshot is not None else repository.transaction())


def _note(now: datetime, kind: str, text: str) -> ResearchNote:
    return ResearchNote(t=now, kind=kind, text=text[:MAX_RESEARCH_NOTE_CHARS])


#: How far back a repeated note is recognised and not written again.
_NOTE_DEDUPE_WINDOW = 20


def _fresh_notes(existing: tp.Sequence[ResearchNote], notes: tp.Sequence[ResearchNote]) -> list[ResearchNote]:
    """Drop notes that repeat a recent one (same kind and text).

    A step that keeps retrying (a deferred cancel, an unavailable store) would
    otherwise add the same line every tick, rewrite the whole local document
    each time and push the plan's real history out of its bounded note window.
    """

    recent = {(item.kind, item.text) for item in existing[-_NOTE_DEDUPE_WINDOW:]}
    fresh: list[ResearchNote] = []
    for item in notes:
        if (item.kind, item.text) not in recent:
            recent.add((item.kind, item.text))
            fresh.append(item)
    return fresh


def _brief(exc: BaseException) -> str:
    """A short, input-free description of a validation or service error."""

    if isinstance(exc, ValidationError):
        parts = [
            f"{'.'.join(str(part) for part in error['loc']) or 'value'}: {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)[:3]
        ]
        return "; ".join(parts)[:500]
    return str(exc)[:500]


def _event(
    actor: str, plan: ResearchPlanRecord, action: str, payload: dict[str, tp.Any] | None = None, *, previous: str | None
) -> AuditEvent:
    return AuditEvent(
        id=_new_id("evt"),
        actor=actor,
        entity_type="research_plan",
        entity_id=plan.id,
        action=action,
        org_id=plan.org_id,
        project_id=plan.project_id,
        previous_state=previous,
        new_state=plan.state,
        payload=payload or {},
    )


def research_job(
    raw: object,
    *,
    round_index: int,
    allowed_secret_mounts: tp.Mapping[str, str] | None = None,
) -> ResearchJob:
    """Validate one ``{name, spec}`` whose spec is a workload the submit route accepts.

    A research job may not submit children: their leases would sit outside the
    plan's chip-hour accounting, so the budget could never stop them. When
    ``allowed_secret_mounts`` is given (a later, model-planned round), the job
    may only mount secrets the human-approved round already mounted.
    """

    if not isinstance(raw, dict):
        raise InputError("each research job must be an object with a name and a spec")
    spec = raw.get("spec")
    if not isinstance(spec, dict):
        raise InputError("each research job needs a spec object")
    label = raw.get("name") or spec.get("name")
    try:
        definition = WorkloadDefinition.model_validate(spec)
    except ValueError as exc:
        raise InputError(f"research job {label!r} has an invalid workload spec: {_brief(exc)}") from None
    if definition.can_submit_children:
        raise InputError(
            f"research job {label!r} may not submit child workloads: their usage would escape the plan's budget"
        )
    if allowed_secret_mounts is not None:
        extra = {
            name: reference
            for name, reference in definition.secret_mounts.items()
            if allowed_secret_mounts.get(name) != reference
        }
        if extra:
            raise InputError(
                f"research job {label!r} mounts secrets the approved plan did not ({', '.join(sorted(extra))}); "
                "a later round may only reuse the approved round's secret mounts"
            )
    try:
        return ResearchJob(name=str(raw.get("name") or definition.name).strip()[:200], spec=spec, round=round_index)
    except ValueError as exc:
        raise InputError(f"invalid research job {label!r}: {_brief(exc)}") from None


def lease_chip_hours(lease: LeaseRecord, now: datetime) -> float:
    """Chip-hours one lease used, never less than any figure the plane recorded for it.

    Chips are the gang's GPUs plus TPUs. A settled lease counts the larger of
    its settled accelerator hours and its whole held time (queue included)
    times its chips. An open lease counts everything from creation to ``now``:
    reserved capacity is spent capacity, and an estimate that waits for
    settlement would let a running plan overshoot its budget.
    """

    gang = lease.resources
    chips = float((gang.per_replica.tpu + gang.per_replica.gpu) * gang.replicas)
    start = _aware(lease.created_at)
    if lease.settled_at is not None:
        held = max((_aware(lease.settled_at) - start).total_seconds(), 0.0) / 3_600
        return max(float(sum(lease.accelerator_hours_settled.values())), held * chips)
    end = now if lease.state in ACTIVE_LEASE_STATES else _aware(lease.updated_at)
    return max((end - start).total_seconds(), 0.0) / 3_600 * chips


@dataclass(frozen=True)
class _Observed:
    workload: WorkloadRecord | None
    chip_hours: float


class ResearchPlansMixin:
    """Owner-scoped research plans and the deterministic step the runner drives."""

    # ------------------------------------------------------------- helpers

    def _research_admin(self, principal: Principal) -> bool:
        return bool(self.authorizer.allows(principal, Permission.MANAGE_FLEET))  # type: ignore[attr-defined]

    def _visible_research_plan(
        self, principal: Principal, plan: ResearchPlanRecord | None, plan_id: str
    ) -> ResearchPlanRecord:
        # Someone else's plan answers exactly like one that does not exist.
        if plan is None or (plan.subject != principal.subject and not self._research_admin(principal)):
            raise NotFoundError(f"research plan {plan_id!r} not found")
        return plan

    def _project_authority(self, principal: Principal, org_id: str, project_id: str) -> tuple[RoleGrant, ...]:
        """The narrowest grant carrying what ``principal`` may do in one project: operate, submit or nothing."""

        allows = self.authorizer.allows  # type: ignore[attr-defined]
        if not allows(principal, Permission.SUBMIT, org_id=org_id, project_id=project_id):
            return ()
        role = (
            Role.PROJECT_OPERATOR
            if allows(principal, Permission.OPERATE, org_id=org_id, project_id=project_id)
            else Role.PROJECT_SUBMITTER
        )
        return (RoleGrant(role=role, org_id=org_id, project_id=project_id),)

    async def _research_principal(self, plan: ResearchPlanRecord, now: datetime) -> Principal:
        """The owner, as authentication would build them now, narrowed to this plan's project.

        Durable grants are re-read through ``active_grants_for`` (the resolver
        authentication itself uses), so a revocation counts from the next step.
        A disabled account has no authority at all. Admin roles never survive
        the narrowing: the plane submits exactly as a project operator would.
        """

        async with _read_view(self.repository) as tx:  # type: ignore[attr-defined]
            user = await tx.get_user(plan.subject)
        if user is not None and user.disabled:
            return Principal(subject=plan.subject)
        durable = await self.active_grants_for(plan.subject, now=now)  # type: ignore[attr-defined]
        merged: list[RoleGrant] = []
        for grant in (*plan.approval_grants, *durable):
            if grant not in merged:
                merged.append(grant)
        whole = Principal(subject=plan.subject, grants=tuple(merged))
        return Principal(subject=plan.subject, grants=self._project_authority(whole, plan.org_id, plan.project_id))

    # --------------------------------------------------------------- reads

    async def list_research_plans(self, principal: Principal, *, state: str | None = None) -> list[ResearchPlanRecord]:
        """Fleet admins see every plan; everyone else sees only their own."""

        subject = None if self._research_admin(principal) else principal.subject
        async with _read_view(self.repository) as tx:  # type: ignore[attr-defined]
            return list(
                await tx.list_research_plans(subject=subject, states=None if state is None else frozenset({state}))
            )

    async def get_research_plan(self, principal: Principal, plan_id: str) -> ResearchPlanRecord:
        async with _read_view(self.repository) as tx:  # type: ignore[attr-defined]
            return self._visible_research_plan(principal, await tx.get_research_plan(plan_id), plan_id)

    async def describe_research_plan(
        self, principal: Principal, plan_id: str
    ) -> tuple[ResearchPlanRecord, dict[str, WorkloadRecord]]:
        """One plan plus the current record of every workload it submitted."""

        async with _read_view(self.repository) as tx:  # type: ignore[attr-defined]
            plan = self._visible_research_plan(principal, await tx.get_research_plan(plan_id), plan_id)
            workloads: dict[str, WorkloadRecord] = {}
            for item in plan.submitted:
                if item.workload_id is not None:
                    workload = await tx.get_workload(item.workload_id)
                    if workload is not None:
                        workloads[item.workload_id] = workload
        return plan, workloads

    # ----------------------------------------------------------- mutations

    async def propose_research_plan(
        self,
        principal: Principal,
        *,
        title: str,
        org_id: str,
        project_id: str,
        jobs: tp.Sequence[object],
        budget: ResearchBudget | tp.Mapping[str, tp.Any],
        goal: str = "",
        iterate: bool = False,
        provider_id: str | None = None,
        model: str | None = None,
    ) -> ResearchPlanRecord:
        """Record a plan in ``proposed``; every job spec is validated as a workload now."""

        self.authorizer.require(principal, Permission.SUBMIT, org_id=org_id, project_id=project_id)  # type: ignore[attr-defined]
        try:
            limits = budget if isinstance(budget, ResearchBudget) else ResearchBudget.model_validate(dict(budget))
        except ValueError as exc:
            raise InputError(f"invalid research budget: {_brief(exc)}") from None
        if not jobs:
            raise InputError("a research plan needs at least one job")
        if len(jobs) > limits.max_jobs:
            raise InputError(f"the plan lists {len(jobs)} jobs but its budget allows max_jobs={limits.max_jobs}")
        planned = tuple(research_job(item, round_index=0) for item in jobs)
        now = utcnow()
        try:
            plan = ResearchPlanRecord(
                id=f"rp_{secrets.token_hex(8)}",
                subject=principal.subject,
                org_id=org_id,
                project_id=project_id,
                title=" ".join(title.split())[:200],
                goal=goal.strip(),
                jobs=planned,
                budget=limits,
                iterate=iterate,
                provider_id=provider_id,
                model=model,
                notes=(_note(now, "proposed", f"{principal.subject} proposed {len(planned)} job(s)"),),
                created_at=now,
                updated_at=now,
            )
        except ValueError as exc:
            raise InputError(f"invalid research plan: {_brief(exc)}") from None
        async with self.repository.transaction() as tx:  # type: ignore[attr-defined]
            open_plans = await tx.list_research_plans(subject=principal.subject, states=_OPEN_STATES)
            if len(open_plans) >= MAX_OPEN_RESEARCH_PLANS_PER_SUBJECT:
                raise ConflictError(
                    f"{principal.subject} already has {len(open_plans)} open research plans; stop or finish some first"
                )
            await tx.put_research_plan(plan)
            await tx.append_event(
                _event(principal.subject, plan, "research_plan.proposed", {"jobs": len(planned)}, previous=None)
            )
        return plan

    async def approve_research_plan(
        self,
        principal: Principal,
        plan_id: str,
        *,
        credential_grants: tp.Sequence[RoleGrant] | None = None,
    ) -> ResearchPlanRecord:
        """Approve a proposed plan (its owner only); the runner starts it on its next step.

        The approver must be able to submit *and* operate in the plan's project,
        because the plane must be able to stop what it starts. Authority the
        approving credential carried (rather than the plane's durable grants,
        which are re-read on every step) is recorded narrowed to that project.

        ``credential_grants`` are the grants the authenticated credential itself
        carried, before durable grants were merged in; the API passes them. Only
        a direct service caller falls back to inferring them by subtracting a
        fresh durable read, which can misclassify a grant revoked in between.
        """

        durable = (
            None if credential_grants is not None else await self.active_grants_for(principal.subject)  # type: ignore[attr-defined]
        )
        async with self.repository.transaction() as tx:  # type: ignore[attr-defined]
            plan = self._visible_research_plan(principal, await tx.get_research_plan(plan_id), plan_id)
            if plan.subject != principal.subject:
                raise AuthorizationError("only the person who proposed a research plan may approve it")
            if plan.state != "proposed":
                raise ConflictError(f"research plan {plan_id} is already {plan.state}")
            for permission in (Permission.SUBMIT, Permission.OPERATE):
                self.authorizer.require(principal, permission, org_id=plan.org_id, project_id=plan.project_id)  # type: ignore[attr-defined]
            if credential_grants is not None:
                credential = Principal(subject=principal.subject, grants=tuple(credential_grants))
            else:
                credential = Principal(
                    subject=principal.subject,
                    grants=tuple(grant for grant in principal.grants if grant not in (durable or ())),
                )
            now = utcnow()
            approved = plan.model_copy(
                update={
                    "state": "approved",
                    "approved_at": now,
                    "approved_by": principal.subject,
                    "approval_grants": self._project_authority(credential, plan.org_id, plan.project_id),
                    "notes": (*plan.notes, _note(now, "approved", f"approved by {principal.subject}"))[
                        -MAX_RESEARCH_NOTES:
                    ],
                    "updated_at": now,
                }
            )
            await tx.put_research_plan(approved)
            await tx.append_event(_event(principal.subject, approved, "research_plan.approved", previous=plan.state))
        return approved

    async def stop_research_plan(self, principal: Principal, plan_id: str, *, reason: str = "") -> ResearchPlanRecord:
        """Stop a plan (owner or fleet admin): its running workloads are cancelled as the owner, now."""

        cleaned = " ".join(reason.split())[:2000]
        async with self.repository.transaction() as tx:  # type: ignore[attr-defined]
            plan = self._visible_research_plan(principal, await tx.get_research_plan(plan_id), plan_id)
            if plan.state in RESEARCH_PLAN_TERMINAL_STATES:
                raise ConflictError(f"research plan {plan_id} is already {plan.state}")
            updated = plan
            if plan.state != "stopping":
                now = utcnow()
                text = f"stop requested by {principal.subject}" + (f": {cleaned}" if cleaned else "")
                update: dict[str, tp.Any] = {
                    "stop_reason": cleaned or f"stopped by {principal.subject}",
                    "stop_requested_by": principal.subject,
                    "notes": (*plan.notes, _note(now, "stop", text))[-MAX_RESEARCH_NOTES:],
                    "updated_at": now,
                }
                if plan.state == "proposed":
                    # Nothing was ever submitted, so there is nothing to cancel.
                    update.update(state="stopped", finished_at=now)
                else:
                    update.update(state="stopping", stop_target="stopped")
                updated = plan.model_copy(update=update)
                await tx.put_research_plan(updated)
                await tx.append_event(
                    _event(
                        principal.subject,
                        updated,
                        "research_plan.stop_requested",
                        {"reason": updated.stop_reason},
                        previous=plan.state,
                    )
                )
        if updated.state == "stopping":
            # The caller's own authority is the fallback: an admin stopping a
            # disabled owner's plan must actually cancel its jobs.
            driven = await self._drive_research_stop(plan_id, utcnow(), actor=principal)
            return driven or updated
        return updated

    # ------------------------------------------------------------- runner

    async def step_research_plans(
        self,
        now: datetime | None = None,
        *,
        iterate: ResearchIterator | None = None,
        still_leading: tp.Callable[[], tp.Awaitable[bool]] | None = None,
        read_logs: ResearchLogReader | None = None,
    ) -> int:
        """Advance every approved, running or stopping plan once; return how many were looked at.

        One indexed read when nothing is active. A plan whose step fails is
        logged and retried on the next step; it never stops the others.
        ``still_leading`` is re-checked before each plan, so a replica that lost
        leadership mid-tick (a slow iteration call) stops stepping plans. With
        no fixed ``now`` each plan is measured against the current time, so one
        slow plan never makes the next plan's spend look smaller.
        """

        async with _read_view(self.repository) as tx:  # type: ignore[attr-defined]
            plans = await tx.list_research_plans(states=RESEARCH_PLAN_ACTIVE_STATES)
        for plan in plans:
            if still_leading is not None and not await still_leading():
                break
            moment = _aware(now or utcnow())
            try:
                if plan.state == "stopping":
                    await self._drive_research_stop(plan.id, moment)
                else:
                    await self._advance_research_plan(plan, moment, iterate, read_logs)
            except Exception:
                logger.exception("research plan %s step failed; it is retried on the next step", plan.id)
        return len(plans)

    async def _observe_research(self, plan: ResearchPlanRecord, now: datetime) -> dict[str, _Observed]:
        observed: dict[str, _Observed] = {}
        async with _read_view(self.repository) as tx:  # type: ignore[attr-defined]
            for item in plan.submitted:
                if item.workload_id is None:
                    continue
                workload = await tx.get_workload(item.workload_id)
                leases = await tx.list_leases(workload_id=item.workload_id)
                observed[item.workload_id] = _Observed(workload, sum(lease_chip_hours(lease, now) for lease in leases))
        return observed

    @staticmethod
    def _research_budget_breach(
        plan: ResearchPlanRecord, chip_hours: float, now: datetime, started_at: datetime
    ) -> tuple[str, str] | None:
        if chip_hours >= plan.budget.max_chip_hours:
            return (
                "over_budget",
                f"chip-hour budget reached: {chip_hours:.3f} of {plan.budget.max_chip_hours:g} chip-hours",
            )
        if now >= _aware(started_at) + timedelta(seconds=plan.budget.max_wall_seconds):
            return "over_budget", f"wall-clock budget of {plan.budget.max_wall_seconds}s reached"
        return None

    async def _advance_research_plan(
        self,
        plan: ResearchPlanRecord,
        now: datetime,
        iterate: ResearchIterator | None,
        read_logs: ResearchLogReader | None = None,
    ) -> None:
        principal = await self._research_principal(plan, now)
        started_at = plan.started_at or now
        notes: list[ResearchNote] = []
        submissions: list[ResearchSubmission] = []
        if plan.state == "approved":
            notes.append(_note(now, "started", f"running as {plan.subject}"))
        observed = await self._observe_research(plan, now)
        waiting_losses: set[int] = set()
        losses = await read_research_losses(plan, observed, read_logs, now, notes, waiting_losses)
        if losses:
            # The iteration model (same step) compares jobs on these.
            plan = plan.model_copy(
                update={
                    "submitted": tuple(
                        item.model_copy(update={"loss": losses[item.job_index]}) if item.job_index in losses else item
                        for item in plan.submitted
                    )
                }
            )
        chip_hours = max(plan.spent.chip_hours, sum(item.chip_hours for item in observed.values()))
        stop = self._research_budget_breach(plan, chip_hours, now, started_at)
        allows = self.authorizer.allows  # type: ignore[attr-defined]
        if stop is None and not all(
            allows(principal, permission, org_id=plan.org_id, project_id=plan.project_id)
            for permission in (Permission.SUBMIT, Permission.OPERATE)
        ):
            # OPERATE too: the plane must be able to cancel what it submits, or
            # the budget could not be enforced on those jobs.
            stop = (
                "failed",
                f"{plan.subject} is no longer allowed to submit and operate in {plan.org_id}/{plan.project_id}",
            )
        if stop is None:
            stop = await self._submit_research_jobs(plan, principal, now, submissions, notes, observed)
        update: dict[str, tp.Any] = {"started_at": started_at, "state": "running"}
        if stop is not None:
            target, why = stop
            update.update(state="stopping", stop_target=target, stop_reason=why)
            notes.append(_note(now, target, why))
        elif not waiting_losses:
            # A round is only judged once every final job's loss is read (or its
            # grace has run out): finishing first would lose the comparison.
            update.update(
                await self._research_round_outcome(
                    plan, principal, now, submissions, observed, chip_hours, notes, iterate
                )
            )
        notes = _fresh_notes(plan.notes, notes)
        if (
            not submissions
            and not notes
            and not losses
            and set(update) == {"started_at", "state"}
            and update["state"] == plan.state
            and chip_hours - plan.spent.chip_hours < plan.budget.max_chip_hours * 0.01
        ):
            # Nothing a reader would notice changed: skip the write (on the
            # local store every commit rewrites the whole document).
            return
        committed = await self._commit_research_step(plan, now, update, submissions, notes, chip_hours, losses)
        if committed is not None and committed.state == "stopping":
            await self._drive_research_stop(plan.id, now)

    async def _submit_research_jobs(
        self,
        plan: ResearchPlanRecord,
        principal: Principal,
        now: datetime,
        submissions: list[ResearchSubmission],
        notes: list[ResearchNote],
        observed: dict[str, _Observed],
    ) -> tuple[str, str] | None:
        """Submit the current round's unsubmitted jobs; return a stop target if the owner lost authority.

        The idempotency key is the plan and job index, so a step that crashes
        after submitting but before recording replays the same workload rather
        than submitting it twice.
        """

        done = {item.job_index for item in plan.submitted}
        submitted = sum(1 for item in plan.submitted if item.workload_id is not None)
        for index, job in enumerate(plan.jobs):
            if index in done or job.round > plan.current_round:
                continue
            if submitted >= plan.budget.max_jobs:
                notes.append(_note(now, "budget", f"job {index} not submitted: max_jobs={plan.budget.max_jobs}"))
                break
            try:
                spec = WorkloadDefinition.model_validate(job.spec)
                workload, replayed = await self.submit_workload(  # type: ignore[attr-defined]
                    principal,
                    org_id=plan.org_id,
                    project_id=plan.project_id,
                    spec=spec,
                    idempotency_key=f"research:{plan.id}:{index}",
                )
            except AuthorizationError as exc:
                return "failed", f"{plan.subject} may no longer submit this plan's jobs: {_brief(exc)}"
            except UnavailableError as exc:
                notes.append(_note(now, "retry", f"job {index} not submitted yet: {_brief(exc)}"))
                break
            except (EyvanServiceError, ValueError) as exc:
                submissions.append(
                    ResearchSubmission(job_index=index, round=job.round, error=_brief(exc), submitted_at=now)
                )
                notes.append(_note(now, "refused", f"job {index} {job.name!r} was refused: {_brief(exc)}"))
                continue
            submissions.append(
                ResearchSubmission(job_index=index, round=job.round, workload_id=workload.id, submitted_at=now)
            )
            observed[workload.id] = _Observed(workload, observed.get(workload.id, _Observed(None, 0.0)).chip_hours)
            submitted += 1
            suffix = " (replayed)" if replayed else ""
            notes.append(_note(now, "submitted", f"job {index} {job.name!r} -> {workload.id}{suffix}"))
        return None

    async def _research_summary(
        self,
        plan: ResearchPlanRecord,
        submissions: dict[int, ResearchSubmission],
        observed: dict[str, _Observed],
    ) -> list[dict[str, tp.Any]]:
        """A compact per-job summary for the iteration model: spec, state, failure and latest progress."""

        rows: list[dict[str, tp.Any]] = []
        async with _read_view(self.repository) as tx:  # type: ignore[attr-defined]
            for index, job in enumerate(plan.jobs):
                row: dict[str, tp.Any] = {"index": index, "name": job.name, "round": job.round, "spec": job.spec}
                item = submissions.get(index)
                if item is None:
                    row["state"] = "not_submitted"
                elif item.workload_id is None:
                    row.update(state="refused", error=item.error)
                else:
                    seen = observed.get(item.workload_id)
                    workload = seen.workload if seen is not None else None
                    row.update(
                        workload_id=item.workload_id,
                        state=workload.state.value if workload is not None else "unknown",
                        failure_reason=workload.failure_reason if workload is not None else None,
                        attempts=workload.attempt if workload is not None else None,
                        chip_hours=round(seen.chip_hours, 4) if seen is not None else None,
                    )
                    if item.loss is not None:
                        row["loss"] = {
                            "final": item.loss.final,
                            "last_step": item.loss.last_step,
                            "total_steps": item.loss.total_steps,
                        }
                    reports = await tx.list_progress_reports(item.workload_id)
                    if reports:
                        latest = max(reports, key=lambda report: (report.reported_at, -report.rank))
                        row["progress"] = {
                            "step": latest.step,
                            "total": latest.total,
                            "message": latest.message,
                            "metrics": dict(list(latest.metrics.items())[:32]),
                        }
                rows.append(row)
        return rows

    async def _research_round_outcome(
        self,
        plan: ResearchPlanRecord,
        principal: Principal,
        now: datetime,
        submissions: list[ResearchSubmission],
        observed: dict[str, _Observed],
        chip_hours: float,
        notes: list[ResearchNote],
        iterate: ResearchIterator | None,
    ) -> dict[str, tp.Any]:
        """Once every job of the current round is final: plan another round, or finish."""

        by_index = {item.job_index: item for item in (*plan.submitted, *submissions)}
        current = [index for index, job in enumerate(plan.jobs) if job.round == plan.current_round]
        for index in current:
            item = by_index.get(index)
            if item is None:
                return {}
            if item.workload_id is not None:
                seen = observed.get(item.workload_id)
                if seen is None or seen.workload is None or seen.workload.state not in TERMINAL_WORKLOAD_STATES:
                    return {}
        finish: dict[str, tp.Any] = {"state": "finished", "finished_at": now}
        round_number = plan.current_round + 1
        if not plan.iterate:
            notes.append(_note(now, "finished", f"all {len(plan.jobs)} job(s) reached a final state"))
            return finish
        approved_mounts: dict[str, str] = {}
        for job in plan.jobs:
            if job.round == 0:
                try:
                    approved_mounts.update(WorkloadDefinition.model_validate(job.spec).secret_mounts)
                except ValueError:
                    continue  # a stored spec that no longer validates approves nothing
        limit = plan.budget.max_rounds or MAX_RESEARCH_ROUNDS
        room = min(plan.budget.max_jobs, MAX_RESEARCH_JOBS) - len(plan.jobs)
        if plan.current_round + 1 >= limit or room <= 0:
            notes.append(_note(now, "finished", f"round {round_number} done; the budget leaves no room for another"))
            return finish
        if iterate is None:
            notes.append(
                _note(now, "finished", f"no iteration model runs on this plane; finished after round {round_number}")
            )
            return finish
        summary = await self._research_summary(plan, by_index, observed)
        try:
            outcome = await iterate(plan, principal, summary)
        except Exception as exc:
            errors = plan.iteration_errors + 1
            text = f"the iteration model failed ({type(exc).__name__}: {str(exc)[:300]})"
            if errors >= MAX_RESEARCH_ITERATION_ERRORS:
                notes.append(_note(now, "finished", f"{text}; giving up after {errors} attempts"))
                return {**finish, "iteration_errors": errors}
            notes.append(_note(now, "iteration_error", f"{text}; retrying on the next step"))
            return {"iteration_errors": errors}
        if outcome is None or "unavailable" in outcome:
            why = (outcome or {}).get("unavailable") or "no assistant provider is available"
            notes.append(_note(now, "finished", f"cannot iterate ({why}); finished after round {round_number}"))
            return finish
        if "finish" in outcome:
            notes.append(_note(now, "finished", f"the iteration model finished: {str(outcome['finish'])[:1500]}"))
            return finish
        proposed = outcome.get("jobs")
        accepted: list[ResearchJob] = []
        dropped = 0
        for raw in proposed if isinstance(proposed, list) else []:
            if len(accepted) >= room:
                dropped += 1
                continue
            try:
                accepted.append(research_job(raw, round_index=round_number, allowed_secret_mounts=approved_mounts))
            except InputError as exc:
                notes.append(_note(now, "rejected", f"proposed job rejected: {exc}"))
        if dropped:
            notes.append(
                _note(now, "budget", f"dropped {dropped} proposed job(s) beyond max_jobs={plan.budget.max_jobs}")
            )
        if not accepted:
            notes.append(_note(now, "finished", "the iteration model proposed no valid jobs; finished"))
            return finish
        rationale = str(outcome.get("rationale") or "").strip()
        names = ", ".join(job.name for job in accepted)
        notes.append(
            _note(now, "round", f"round {round_number + 1} planned ({len(accepted)} job(s): {names}). {rationale}")
        )
        return {
            "jobs": (*plan.jobs, *accepted),
            "current_round": round_number,
            "iteration_errors": 0,
            "_round_planned": len(accepted),
        }

    async def _commit_research_step(
        self,
        seen: ResearchPlanRecord,
        now: datetime,
        update: dict[str, tp.Any],
        submissions: list[ResearchSubmission],
        notes: list[ResearchNote],
        chip_hours: float,
        losses: tp.Mapping[int, ResearchLoss] | None = None,
    ) -> ResearchPlanRecord | None:
        """Write one step, merged over whatever a concurrent stop changed meanwhile.

        Submissions, notes and spend always land (a submitted workload must
        never be forgotten); lifecycle changes land only if the plan's state is
        still the one this step started from, so a stop is never overwritten.
        """

        planned = update.pop("_round_planned", 0)
        async with self.repository.transaction() as tx:  # type: ignore[attr-defined]
            current = await tx.get_research_plan(seen.id)
            if current is None:
                return None
            # Compare-and-set on the last write, not only the state: a second
            # replica (or a stop) that wrote meanwhile wins the lifecycle fields,
            # so two steps can never both plan a round over each other.
            raced = current.state != seen.state or current.updated_at != seen.updated_at
            known = {item.job_index for item in current.submitted}
            stored = tuple(
                item.model_copy(update={"loss": losses[item.job_index]})
                if losses and item.loss is None and item.workload_id is not None and item.job_index in losses
                else item
                for item in current.submitted
            )
            submitted = (*stored, *(item for item in submissions if item.job_index not in known))
            fields: dict[str, tp.Any] = {
                "submitted": submitted,
                "notes": (*current.notes, *_fresh_notes(current.notes, notes))[-MAX_RESEARCH_NOTES:],
                "updated_at": now,
            }
            if not raced:
                fields.update(update)
            fields["spent"] = ResearchSpend(
                chip_hours=max(current.spent.chip_hours, chip_hours),
                jobs_submitted=sum(1 for item in submitted if item.workload_id is not None),
                rounds=fields.get("current_round", current.current_round) + 1,
            )
            updated = ResearchPlanRecord.model_validate(current.model_copy(update=fields).model_dump())
            await tx.put_research_plan(updated)
            new_ids = [item.workload_id for item in submissions if item.workload_id is not None]
            if new_ids:
                await tx.append_event(
                    _event(
                        updated.subject,
                        updated,
                        "research_plan.submitted",
                        {"workload_ids": new_ids},
                        previous=current.state,
                    )
                )
            if planned and not raced:
                await tx.append_event(
                    _event(
                        updated.subject,
                        updated,
                        "research_plan.round_planned",
                        {"round": updated.current_round, "jobs": planned},
                        previous=current.state,
                    )
                )
            if updated.state != current.state:
                action = {"running": "started"}.get(updated.state, updated.state)
                await tx.append_event(
                    _event(
                        updated.subject,
                        updated,
                        f"research_plan.{action}",
                        {"reason": updated.stop_reason} if updated.stop_reason else {},
                        previous=current.state,
                    )
                )
        if raced and updated.state in RESEARCH_PLAN_TERMINAL_STATES and new_ids:
            # Stopped while this step was submitting: cancel what it just started.
            async with _read_view(self.repository) as tx:  # type: ignore[attr-defined]
                late = {workload_id: await tx.get_workload(workload_id) for workload_id in new_ids}
            principal = await self._research_principal(updated, now)
            late_notes, retry, stuck = await self._cancel_research_workloads(updated, principal, late, now)
            for note in late_notes:
                logger.warning("research plan %s: %s", updated.id, note.text)
            if retry or stuck:
                # A terminal plan is never stepped again, so a deferred cancel
                # would orphan the workload: reopen the stop instead.
                async with self.repository.transaction() as tx:  # type: ignore[attr-defined]
                    current = await tx.get_research_plan(updated.id)
                    if current is not None and current.state in RESEARCH_PLAN_TERMINAL_STATES:
                        reopened = current.model_copy(
                            update={
                                "state": "stopping",
                                "stop_target": current.state,
                                "finished_at": None,
                                "notes": (*current.notes, *_fresh_notes(current.notes, late_notes))[
                                    -MAX_RESEARCH_NOTES:
                                ],
                                "updated_at": now,
                            }
                        )
                        await tx.put_research_plan(reopened)
                        await tx.append_event(
                            _event(updated.subject, reopened, "research_plan.stopping", previous=current.state)
                        )
                        updated = reopened
        return updated

    async def _cancel_research_workloads(
        self,
        plan: ResearchPlanRecord,
        principal: Principal,
        workloads: dict[str, WorkloadRecord | None],
        now: datetime,
        *,
        fallback: Principal | None = None,
    ) -> tuple[list[ResearchNote], bool, bool]:
        """Cancel every unfinished workload; return notes, whether to retry, and whether any is stuck.

        The owner's own authority is tried first. When the owner can no longer
        operate (a disabled account, a revoked grant) ``fallback`` -- the person
        who asked for the stop, e.g. a fleet admin -- is tried next. A workload
        nobody could cancel is reported as stuck, so the plan is never settled
        as stopped while its jobs still run.
        """

        notes: list[ResearchNote] = []
        retry = False
        stuck = False
        for workload_id, workload in workloads.items():
            if workload is None or workload.state in TERMINAL_WORKLOAD_STATES:
                continue
            if workload.state is WorkloadState.CANCELLING:
                continue
            actors = [principal] + ([fallback] if fallback is not None and fallback != principal else [])
            for position, actor in enumerate(actors):
                try:
                    await self.cancel_workload(  # type: ignore[attr-defined]
                        actor, workload_id, idempotency_key=f"research-stop:{plan.id}:{workload_id}"
                    )
                except (ConflictError, UnavailableError) as exc:
                    # A preemption in flight, or a store hiccup: try again next step.
                    retry = True
                    notes.append(_note(now, "retry", f"cancel of {workload_id} deferred: {_brief(exc)}"))
                    break
                except EyvanServiceError as exc:
                    if position + 1 < len(actors):
                        continue
                    stuck = True
                    notes.append(
                        _note(
                            now,
                            "cancel_failed",
                            f"could not cancel {workload_id} as {actor.subject}: {_brief(exc)}; "
                            "it is still running -- a project operator or fleet admin must stop it",
                        )
                    )
                    break
                else:
                    by = "" if actor is principal else f" as {actor.subject}"
                    notes.append(_note(now, "cancelled", f"cancelled {workload_id}{by}"))
                    break
        return notes, retry, stuck

    async def _drive_research_stop(
        self, plan_id: str, now: datetime, *, actor: Principal | None = None
    ) -> ResearchPlanRecord | None:
        """Cancel a stopping plan's workloads as its owner, then settle it in its stop target.

        The plan stays ``stopping`` -- never reported stopped -- while any of its
        workloads could not be cancelled; each later step retries.
        """

        async with _read_view(self.repository) as tx:  # type: ignore[attr-defined]
            plan = await tx.get_research_plan(plan_id)
            if plan is None or plan.state != "stopping":
                return tp.cast(ResearchPlanRecord | None, plan)
            workloads = {
                item.workload_id: await tx.get_workload(item.workload_id)
                for item in plan.submitted
                if item.workload_id is not None
            }
        principal = await self._research_principal(plan, now)
        notes, retry, stuck = await self._cancel_research_workloads(plan, principal, workloads, now, fallback=actor)
        async with self.repository.transaction() as tx:  # type: ignore[attr-defined]
            current = await tx.get_research_plan(plan_id)
            if current is None or current.state != "stopping":
                return tp.cast(ResearchPlanRecord | None, current)
            if any(item.workload_id is not None and item.workload_id not in workloads for item in current.submitted):
                retry = True  # a step submitted more while this one cancelled
            fresh = _fresh_notes(current.notes, notes)
            fields: dict[str, tp.Any] = {"notes": (*current.notes, *fresh)[-MAX_RESEARCH_NOTES:], "updated_at": now}
            if not fresh and (retry or stuck):
                # Nothing new to say and still waiting: skip the whole-document write.
                return tp.cast(ResearchPlanRecord, current)
            if not retry and not stuck:
                fields.update(state=current.stop_target or "stopped", finished_at=now)
            updated = current.model_copy(update=fields)
            await tx.put_research_plan(updated)
            if updated.state != current.state:
                await tx.append_event(
                    _event(
                        current.stop_requested_by or current.subject,
                        updated,
                        f"research_plan.{updated.state}",
                        {"reason": updated.stop_reason},
                        previous=current.state,
                    )
                )
        return updated


__all__ = (
    "MAX_OPEN_RESEARCH_PLANS_PER_SUBJECT",
    "MAX_RESEARCH_ITERATION_ERRORS",
    "ResearchIterator",
    "ResearchPlansMixin",
    "lease_chip_hours",
    "research_job",
)
