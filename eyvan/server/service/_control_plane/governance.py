# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Team governance: per-user fleet rules, approval holds and borrow/reclaim.

Off unless the service is built with ``governance_enabled`` (team mode). The
decisions themselves live in :mod:`eyvan.domain.governance`; this mixin only
collects facts inside a transaction and applies the verdict.

A held submission is an ordinary PENDING workload stored ``schedulable=False``
-- the gate the scheduler and the capacity planner already honour -- plus an
``approval`` request for the fleet admins. Approving flips the gate; declining
or withdrawing cancels the workload. No new workload state reaches the wire.
"""

from __future__ import annotations

import secrets
import typing as tp
import weakref

from eyvan.domain.governance import (
    HELD_WORKLOAD_KEY,
    REFUSE,
    RULE_UNCHANGED,
    AdmissionVerdict,
    QuotaUnits,
    ReclaimPlan,
    RuleUnchanged,
    RunningJob,
    UserRules,
    admission_verdict,
    has_approval_gate,
    plan_reclaim,
    requested_tpu_units,
    requested_units,
    targeted_pool_ids,
    team_tier,
    units_in_use,
)
from eyvan.records import TeamRequestRecord, TeamTier, UserRecord
from eyvan.server.service._control_plane.shared import (
    TERMINAL_WORKLOAD_STATES,
    AuditEvent,
    AuthorizationError,
    CallableExecution,
    ConflictError,
    InputError,
    NotFoundError,
    Permission,
    Principal,
    RepositorySession,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    _new_id,
    utcnow,
)

#: The resolution string that releases a held workload.
HOLD_APPROVED = "approved"

#: Workload states that still hold or wait for capacity, for the hard TPU cap.
_UNFINISHED_STATES = frozenset(WorkloadState) - TERMINAL_WORKLOAD_STATES

#: Running TPU usage per owner inside one transaction, keyed by the session.
#: Counted once, then raised by every workload admitted in that transaction, so
#: a batch, DAG or group admitting N children reads the owner's rows once. The
#: entry dies with the session; a retried transaction gets a fresh session.
_USAGE_IN_TX: weakref.WeakKeyDictionary[tp.Any, dict[str, QuotaUnits]] = weakref.WeakKeyDictionary()

#: States in which a workload holds capacity, for borrow accounting.
_ACTIVE_STATES = frozenset(
    {
        WorkloadState.LEASED,
        WorkloadState.DISPATCHING,
        WorkloadState.RUNNING,
        WorkloadState.CHECKPOINTING,
    }
)


def user_rules(user: UserRecord) -> UserRules:
    return UserRules(
        allowed_fleets=user.allowed_fleets,
        approval_tpu_threshold=user.approval_tpu_threshold,
        guaranteed_tpu_units=user.guaranteed_tpu_units,
        max_tpu_units=user.max_tpu_units,
        max_cpu_units=user.max_cpu_units,
        max_gpu_units=user.max_gpu_units,
    )


class GovernanceMixin:
    """Transactional governance operations."""

    governance_enabled: bool

    # -------------------------------------------------------- tier and cap

    @staticmethod
    async def _team_tier_in(tx: RepositorySession, user: UserRecord) -> TeamTier:
        """Derive one account's tier from its durable grants and ``trusted`` flag."""

        return team_tier(user.trusted, await tx.list_role_grants(subject=user.subject), utcnow())

    @staticmethod
    async def _lock_resource_caps(tx: RepositorySession, subject: str) -> None:
        """Serialize every cap decision for one person, across all of their orgs and projects.

        The idempotency lookup takes a transaction-scoped PostgreSQL advisory lock
        on the key (the same mechanism ``_lock_admission_scope`` uses), so two
        cap decisions for one person never overlap. It is a backstop, not the
        guarantee: the transaction's snapshot predates the lock, so a waiter that
        resumes reads stale usage, and it is PostgreSQL's SERIALIZABLE conflict
        detection that aborts it (retryable, surfaced as 503 + Retry-After) rather
        than admitting it. The memory and local stores run whole transactions
        serially, so there the count is always current.
        """

        await tx.get_idempotency("fleet", "governance.tpu_cap", subject)

    @staticmethod
    async def _units_in_use(tx: RepositorySession, subject: str, *, exclude_id: str | None = None) -> QuotaUnits:
        owned = await tx.list_workloads_created_by(subject, states=_UNFINISHED_STATES)
        return units_in_use(owned, subject, exclude_id=exclude_id)

    async def _require_caps(
        self,
        tx: RepositorySession,
        user: UserRecord,
        requested: QuotaUnits,
        *,
        exclude_id: str | None = None,
        prefix: str = "",
    ) -> None:
        """Check every configured dimension under one owner-wide lock and snapshot.

        Cache admissions within a batch; approval and retry exclude the re-entering
        workload and read fresh usage. Never charge a coordinator for its children.
        """

        rules = user_rules(user)
        if not any(cap is not None and getattr(requested, name) > 0 for name, cap in rules.caps.items()):
            return
        running = _USAGE_IN_TX.setdefault(tx, {}) if exclude_id is None else None
        in_use = running.get(user.subject) if running is not None else None
        if in_use is None:
            await self._lock_resource_caps(tx, user.subject)
            in_use = await self._units_in_use(tx, user.subject, exclude_id=exclude_id)
        refusal = rules.cap_refusal(in_use, requested)
        if refusal is not None:
            raise ConflictError(prefix + refusal)
        if running is not None:
            running[user.subject] = in_use + requested

    @staticmethod
    async def _restricted_target_fleets(tx: RepositorySession, rules: UserRules, spec: WorkloadDefinition) -> list[str]:
        """Resolve every possible target when an allowlist applies, including alternates."""

        fleets: list[str] = []
        if rules.allowed_fleets is not None:
            for pool_id in targeted_pool_ids(spec):
                pool = await tx.get_capacity_pool(pool_id)
                if pool is None:
                    raise AuthorizationError(f"cannot verify the fleet of unresolved pool {pool_id!r}")
                fleets.append(pool.fleet_name)
        return fleets

    async def _governance_readmission(
        self, tx: RepositorySession, workload: WorkloadRecord, prefix: str, *, approving: bool = False
    ) -> None:
        """Re-check the owner, not the operator retrying or releasing the job."""

        if not self.governance_enabled:
            return
        owner = await tx.get_user(workload.created_by)
        if owner is not None:
            if owner.disabled:
                raise AuthorizationError(f"{prefix}owner {owner.subject!r} is disabled")
            rules = user_rules(owner)
            fleets = await self._restricted_target_fleets(tx, rules, workload.spec)
            refusal = rules.fleet_refusal(fleets, pinned=bool(fleets))
            if refusal is not None:
                raise AuthorizationError(prefix + refusal)
            if not approving and workload.parent_workload_id is None and workload.coordination_kind is None:
                reason = rules.approval_reason(requested_tpu_units(workload.spec))
                if reason is not None and await self._team_tier_in(tx, owner) is TeamTier.USER:
                    raise ConflictError(f"{prefix}new attempt {reason}; resubmit for fresh approval")
        if workload.coordination_kind is not None:
            return  # The coordinator holds no resources and never executes its template.
        if isinstance(workload.spec.execution, CallableExecution):
            await self._require_team_callable(
                tx,
                subject=workload.created_by,
                org_id=workload.org_id,
                project_id=workload.project_id,
                spec=workload.spec,
            )
        if owner is not None:
            await self._require_caps(tx, owner, requested_units(workload.spec), exclude_id=workload.id, prefix=prefix)

    async def _require_team_callable(
        self, tx: RepositorySession, *, subject: str, org_id: str, project_id: str, spec: WorkloadDefinition | None
    ) -> None:
        """Authorize uploads provisionally; only a concrete reservation authorizes execution."""

        if not self.governance_enabled:
            return
        user = await tx.get_user(subject)
        if user is None:
            self.callable_trust.require(org_id, project_id)  # type: ignore[attr-defined]
            return
        if user.disabled:
            raise AuthorizationError(f"{subject!r} is disabled and may not run callable workloads")
        if await self._team_tier_in(tx, user) is TeamTier.USER:
            refusal = user_rules(user).callable_refusal(requested_units(spec) if spec is not None else None)
            if refusal is not None:
                raise AuthorizationError(f"{subject!r} may not run callable workloads: {refusal}")
            if spec is not None:
                for pool_id in targeted_pool_ids(spec):
                    if await tx.get_capacity_pool(pool_id) is None:
                        raise AuthorizationError(f"unresolved callable pool {pool_id!r}")

    async def team_tpu_cap(self, principal: Principal) -> dict[str, tp.Any] | None:
        """Return the caller's tier, hard TPU cap and current usage, or ``None`` off team mode."""

        if not getattr(self, "governance_enabled", False):
            return None
        async with self.repository.transaction() as tx:  # type: ignore[attr-defined]
            user = await tx.get_user(principal.subject)
            if user is None:
                return None
            tier = await self._team_tier_in(tx, user)
            in_use = await self._units_in_use(tx, user.subject)
        report = {"tier": tier.value, "max_tpu_units": user.max_tpu_units, "tpu_units_in_use_or_queued": in_use.tpu}
        if user.max_cpu_units is not None:
            report.update(max_cpu_units=user.max_cpu_units, cpu_units_in_use_or_queued=float(in_use.cpu))
        if user.max_gpu_units is not None:
            report.update(max_gpu_units=user.max_gpu_units, gpu_units_in_use_or_queued=in_use.gpu)
        return report

    # ------------------------------------------------------------- rules

    async def set_user_rules(
        self,
        principal: Principal,
        subject: str,
        *,
        allowed_fleets: tp.Sequence[str] | None | RuleUnchanged = RULE_UNCHANGED,
        approval_tpu_threshold: int | None | RuleUnchanged = RULE_UNCHANGED,
        guaranteed_tpu_units: int | None | RuleUnchanged = RULE_UNCHANGED,
        max_tpu_units: int | None | RuleUnchanged = RULE_UNCHANGED,
        max_gpu_units: int | None | RuleUnchanged = RULE_UNCHANGED,
        max_cpu_units: float | None | RuleUnchanged = RULE_UNCHANGED,
    ) -> UserRecord:
        """Update one account's rules (fleet admins only).

        ``None`` clears a rule; a rule left at ``RULE_UNCHANGED`` (omitted by the
        caller) keeps its stored value, so a client that predates a rule cannot
        clear it by not knowing it exists.
        """

        if not self.authorizer.allows(principal, Permission.MANAGE_FLEET):  # type: ignore[attr-defined]
            raise AuthorizationError(f"{principal.subject!r} may not set governance rules")
        async with self.repository.transaction() as tx:  # type: ignore[attr-defined]
            user = await tx.get_user(subject)
            if user is None:
                raise NotFoundError(f"user {subject!r} not found")
            if isinstance(allowed_fleets, RuleUnchanged):
                fleets = user.allowed_fleets
            else:
                fleets = None if allowed_fleets is None else tuple(sorted({name.strip() for name in allowed_fleets}))
            rules: dict[str, tp.Any] = {
                "allowed_fleets": fleets,
                "approval_tpu_threshold": user.approval_tpu_threshold
                if isinstance(approval_tpu_threshold, RuleUnchanged)
                else approval_tpu_threshold,
                "guaranteed_tpu_units": user.guaranteed_tpu_units
                if isinstance(guaranteed_tpu_units, RuleUnchanged)
                else guaranteed_tpu_units,
                "max_tpu_units": user.max_tpu_units if isinstance(max_tpu_units, RuleUnchanged) else max_tpu_units,
                "max_cpu_units": user.max_cpu_units if isinstance(max_cpu_units, RuleUnchanged) else max_cpu_units,
                "max_gpu_units": user.max_gpu_units if isinstance(max_gpu_units, RuleUnchanged) else max_gpu_units,
            }
            try:
                updated = UserRecord.model_validate(
                    {
                        **user.model_dump(mode="json"),
                        "password_hash": user.password_hash,
                        **rules,
                        "updated_at": utcnow(),
                    }
                )
            except ValueError as exc:
                raise InputError(str(exc)) from exc
            await tx.put_user(updated)
            await tx.append_event(
                AuditEvent(
                    id=_new_id("evt"),
                    actor=principal.subject,
                    entity_type="user",
                    entity_id=subject,
                    action="user.rules_updated",
                    payload={
                        "allowed_fleets": list(fleets) if fleets is not None else None,
                        "approval_tpu_threshold": updated.approval_tpu_threshold,
                        "guaranteed_tpu_units": updated.guaranteed_tpu_units,
                        "max_tpu_units": updated.max_tpu_units,
                        **(
                            {"max_cpu_units": updated.max_cpu_units}
                            if updated.max_cpu_units is not None or not isinstance(max_cpu_units, RuleUnchanged)
                            else {}
                        ),
                        **(
                            {"max_gpu_units": updated.max_gpu_units}
                            if updated.max_gpu_units is not None or not isinstance(max_gpu_units, RuleUnchanged)
                            else {}
                        ),
                    },
                )
            )
            return updated

    # --------------------------------------------------------- admission

    async def _governance_admission(
        self,
        tx: RepositorySession,
        *,
        spec: WorkloadDefinition,
        subject: str,
        operation: str,
        top_level: bool,
        holds_capacity: bool = True,
    ) -> AdmissionVerdict | None:
        """Return the verdict for one new submission, or ``None`` when no rule applies.

        Raises ``AuthorizationError`` for a fleet the person may not use and
        ``ConflictError`` for an over-threshold submission that cannot be held
        or one that would break their hard TPU cap. The ``admin`` and
        ``trusted`` tiers are never held, but the cap binds every tier.
        """

        if not getattr(self, "governance_enabled", False):
            return None
        user = await tx.get_user(subject)
        if user is None:
            return None
        rules = user_rules(user)
        if rules.empty:
            return None
        fleets = await self._restricted_target_fleets(tx, rules, spec)
        reservation = requested_units(spec)
        tpu_units = reservation.tpu
        exempt = rules.approval_tpu_threshold is not None and await self._team_tier_in(tx, user) is not TeamTier.USER
        verdict = admission_verdict(
            rules,
            tpu_units=tpu_units,
            fleets=fleets,
            pinned=bool(fleets),
            operation=operation,
            top_level=top_level,
            exempt_from_approval=exempt,
        )
        if verdict.action == REFUSE:
            if verdict.refusal == "fleet":
                raise AuthorizationError(verdict.reason)
            raise ConflictError(verdict.reason)
        # Held jobs count too: approving one later must not be how a cap is passed.
        # A coordinator (batch/DAG/group/service parent) holds nothing itself;
        # each of its children is checked as it is stored.
        if holds_capacity:
            await self._require_caps(tx, user, reservation)
        return verdict

    async def _hold_for_approval(
        self,
        tx: RepositorySession,
        principal: Principal,
        workload: WorkloadRecord,
        verdict: AdmissionVerdict,
    ) -> TeamRequestRecord:
        """File the approval request that owns one held workload."""

        request = TeamRequestRecord(
            id=f"req_{secrets.token_hex(8)}",
            kind="approval",
            subject=workload.created_by,
            org_id=workload.org_id,
            project_id=workload.project_id,
            summary=f"Run {workload.spec.name} ({verdict.tpu_units} TPU units)"[:2000],
            details={
                HELD_WORKLOAD_KEY: workload.id,
                "tpu_units": verdict.tpu_units,
                "reason": verdict.reason,
            },
        )
        await tx.put_team_request(request)
        await tx.append_event(
            AuditEvent(
                id=_new_id("evt"),
                actor=principal.subject,
                entity_type="team_request",
                entity_id=request.id,
                action="team_request.filed",
                org_id=request.org_id,
                project_id=request.project_id,
                payload={"kind": request.kind, HELD_WORKLOAD_KEY: workload.id},
            )
        )
        await self._audit(  # type: ignore[attr-defined]
            tx,
            principal,
            workload,
            "workload.approval_held",
            new_state=workload.state.value,
            payload={"request_id": request.id, "tpu_units": verdict.tpu_units},
        )
        return request

    async def _apply_approval_resolution(
        self,
        tx: RepositorySession,
        principal: Principal,
        request: TeamRequestRecord,
        resolution: str,
    ) -> None:
        """Release (approved) or cancel (declined/withdrawn) the workload a request holds."""

        workload_id = request.details.get(HELD_WORKLOAD_KEY)
        if request.kind != "approval" or not isinstance(workload_id, str):
            return
        workload = await tx.get_workload(workload_id)
        if workload is None or workload.state is not WorkloadState.PENDING or not has_approval_gate(workload):
            # Already cancelled by its owner, or not a governance hold at all.
            return
        if resolution == HOLD_APPROVED:
            await self._governance_readmission(
                tx, workload, f"cannot release {workload.id} for {workload.created_by}: ", approving=True
            )
            released = workload.model_copy(
                update={
                    "schedulable": True,
                    "scheduling_reason": f"approved by {principal.subject}; waiting for scheduling",
                    "updated_at": utcnow(),
                }
            )
            await tx.put_workload(released)
            await self._audit(  # type: ignore[attr-defined]
                tx, principal, released, "workload.approved", payload={"request_id": request.id}
            )
            return
        action = "workload.approval_withdrawn" if resolution == "withdrawn" else "workload.approval_declined"
        await self._cancel_one(tx, principal, workload, action)  # type: ignore[attr-defined]

    # ---------------------------------------------------- borrow / reclaim

    async def _reclaim_inputs(
        self, tx: RepositorySession
    ) -> tuple[list[RunningJob], dict[str, int], dict[str, WorkloadRecord]]:
        active = [workload for workload in await tx.list_workloads() if workload.state in _ACTIVE_STATES]
        ordered = sorted(active, key=lambda item: (item.created_at, item.id))
        jobs = [
            RunningJob(
                workload_id=workload.id,
                subject=workload.created_by,
                tpu_units=requested_tpu_units(workload.spec),
                priority_band=workload.spec.priority_band.value,
                started_order=index,
                reclaimable=workload.state is WorkloadState.RUNNING,
            )
            for index, workload in enumerate(ordered)
        ]
        guarantees = {
            user.subject: user.guaranteed_tpu_units
            for user in await tx.list_users()
            if user.guaranteed_tpu_units is not None
        }
        return jobs, guarantees, {workload.id: workload for workload in active}

    async def plan_reclaim_for(self, principal: Principal, *, tpu_units: int) -> ReclaimPlan:
        """Return the deterministic plan for ``principal`` reclaiming ``tpu_units`` (read-only)."""

        if not getattr(self, "governance_enabled", False):
            raise NotFoundError("governance is not enabled on this plane")
        async with self.repository.transaction() as tx:  # type: ignore[attr-defined]
            jobs, guarantees, _ = await self._reclaim_inputs(tx)
        return plan_reclaim(principal.subject, tpu_units, jobs, guarantees)

    async def reclaim_borrowed(
        self, principal: Principal, *, tpu_units: int, reason: str, idempotency_key: str
    ) -> tuple[ReclaimPlan, list[str]]:
        """Reclaim borrowed capacity for ``principal`` by the deterministic rule.

        The rule, not the caller, picks the victims, and only someone still
        within their own guarantee may ask. Each victim goes through the
        ordinary checkpoint-first ``reclaim_workload`` command, issued under a
        narrow ``reclaim:<subject>`` identity: the caller never gains general
        fleet authority, and the audit trail names who triggered it.

        Returns the plan and the victim ids a reclaim command was issued for.
        A victim that stopped running between planning and issuing is skipped
        (its capacity is already free) rather than failing the whole request.
        """

        plan = await self.plan_reclaim_for(principal, tpu_units=tpu_units)
        if not plan.allowed:
            raise ConflictError(plan.reason)
        reason = reason.strip() or "borrowed capacity reclaimed"
        actor = Principal(
            subject=f"reclaim:{principal.subject}",
            grants=(RoleGrant(role=Role.FLEET_ADMIN),),
        )
        issued: list[str] = []
        for workload_id in plan.victims:
            try:
                await self.reclaim_workload(  # type: ignore[attr-defined]
                    actor,
                    workload_id,
                    idempotency_key=f"{idempotency_key}:{workload_id}",
                    reason=f"{reason} (for {principal.subject})"[:500],
                )
            except ConflictError:
                continue
            issued.append(workload_id)
        return plan, issued


__all__ = ("HELD_WORKLOAD_KEY", "GovernanceMixin", "user_rules")
