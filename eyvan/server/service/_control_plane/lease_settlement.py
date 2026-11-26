# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from uuid import uuid4

from eyvan.server.service._control_plane.shared import (
    ACTIVE_LEASE_STATES,
    TERMINAL_WORKLOAD_STATES,
    AuditEvent,
    AuthorizationError,
    BudgetGrantRecord,
    BudgetRecord,
    ConflictError,
    IdempotencyRecord,
    InvalidTransition,
    JsonObject,
    LeaseRecord,
    LeaseState,
    OrganizationRecord,
    OutboxState,
    Permission,
    PolicyRecord,
    Principal,
    ProjectRecord,
    QuotaGrantRecord,
    QuotaRecord,
    RepositorySession,
    Role,
    WorkloadDispatchReservation,
    WorkloadRecord,
    WorkloadState,
    WorkloadTransitionUpdates,
    _decimal_sum,
    _scope,
    costing,
    datetime,
    ensure_lease_transition,
    ensure_workload_transition,
    hashlib,
    json,
    tp,
    utcnow,
)
from eyvan.server.service.scheduling import rolled_budget


class LeaseSettlementMixin:
    """Transactional lease settlement operations."""

    async def settle_orphaned_lease(
        self,
        principal: Principal,
        lease_id: str,
        *,
        reason: str,
        expected_generation: int,
    ) -> bool:
        """Release one lease that no live workload claims, and charge what it held.

        The counterpart to ``settle_orphaned_workloads_once``, which settles by
        *workload* and therefore cannot see this: a lease whose workload finished,
        vanished, or stopped referencing it is reachable from no other path. It
        keeps its cell allocated and its cost unsettled indefinitely -- capacity
        the fleet is paying for and no tenant is charged for.

        Re-verified inside the transaction rather than trusted from the scan. A
        lease that a dispatch re-attached between the two is not orphaned, and
        releasing it would kill live work; the generation check makes that a
        no-op instead of a race.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            lease = await tx.get_lease(lease_id)
            if lease is None or lease.state not in ACTIVE_LEASE_STATES:
                return False
            if lease.generation != expected_generation:
                return False
            workload = await tx.get_workload(lease.workload_id)
            if not self._lease_is_orphaned(lease, workload):
                return False
            # ``_settle_lease`` persists the lease and the budget itself, so this
            # only decides *whether* to call it. Re-writing what it returned would
            # be redundant at best, and a mutation test proved it was never
            # load-bearing.
            _settled, _budget, changed = await self._settle_lease(tx, principal, workload, lease, reason=reason)
            return changed

    @staticmethod
    def _lease_is_orphaned(lease: LeaseRecord, workload: WorkloadRecord | None) -> bool:
        """Return whether no live workload still claims this lease.

        Three ways, and the third is the one the workload sweep cannot reach: the
        workload is gone, the workload is terminal, or the workload is live and
        has moved on to different leases. A retry or a preemption resume produces
        exactly that third shape.
        """

        if workload is None or workload.state in TERMINAL_WORKLOAD_STATES:
            return True
        claimed = set(workload.lease_ids) | ({workload.lease_id} if workload.lease_id else set())
        return lease.id not in claimed

    async def _settle_lease(
        self,
        tx: RepositorySession,
        principal: Principal,
        workload: WorkloadRecord | None,
        lease: LeaseRecord,
        *,
        reason: str,
    ) -> tuple[LeaseRecord, BudgetRecord | None, bool]:
        """Release one lease and charge what it actually held.

        ``workload`` is optional because a lease can outlive its workload record
        -- that is the condition the orphaned-lease sweep exists to clear, and
        refusing to settle without a record would leave the capacity charged and
        held forever. The lease already carries the workload id, which is all
        the audit payload ever needed.
        """

        scope = _scope(lease.org_id, lease.project_id)
        await tx.get_idempotency(scope, "budget.settle", scope)
        budget = await tx.get_budget(lease.org_id, lease.project_id)
        if lease.settled_at is not None:
            return lease, budget, False
        if lease.state not in {LeaseState.RELEASED, LeaseState.EXPIRED}:
            try:
                ensure_lease_transition(lease.state, LeaseState.RELEASED)
            except InvalidTransition as exc:
                raise ConflictError(str(exc)) from exc
        now = utcnow()
        # Settle what was actually held, not what was estimated at submission.
        # `settled_cost = reserved_cost` charged the submitter's guess: a 6-hour
        # estimate preempted after 10 minutes billed 6 hours, and a 1-hour estimate
        # that ran 40 hours billed 1 -- and the second direction is the one that
        # quietly overruns a hard budget.
        queued_seconds, running_seconds = self._lease_clock(lease, now)
        elapsed_hours = (queued_seconds + running_seconds) / 3_600
        settled_cost = costing.cost_for_hours(lease.hourly_cost, elapsed_hours)
        hours = self._accelerator_hours(lease, elapsed_hours, workload)
        if budget is not None:
            # The occurrence and its actual charged window are persisted in the
            # same transaction, before either the lease or budget is published.
            budget = self._rolled_budget(budget, now)
            if budget.settlement_period_id is None or budget.period_opening_spent is None:
                # Start prospective attribution at the retained balance. This
                # also covers lifetime/unknown periods, which never reset spend.
                budget = budget.model_copy(
                    update={
                        "settlement_period_id": uuid4().hex,
                        "period_opening_spent": budget.spent,
                    }
                )
        settled = lease.model_copy(
            update={
                "state": lease.state if lease.state is LeaseState.EXPIRED else LeaseState.RELEASED,
                "settled_cost": settled_cost,
                "accelerator_hours_settled": hours,
                "settlement_reason": reason,
                "settled_at": now,
                "settlement_budget_period_id": budget.settlement_period_id if budget is not None else None,
                "queued_seconds": queued_seconds,
                "running_seconds": running_seconds,
                "updated_at": now,
            }
        )
        await tx.put_lease(settled)
        await self._audit_scoped(
            tx,
            principal,
            org_id=lease.org_id,
            project_id=lease.project_id,
            entity_type="lease",
            entity_id=lease.id,
            action="lease.settled",
            previous_state=lease.state.value,
            new_state=settled.state.value,
            payload={
                "workload_id": lease.workload_id,
                "reason": reason,
                "settled_cost": settled.settled_cost,
                "accelerator_hours_settled": hours,
            },
        )
        if budget is None:
            return settled, None, True
        # The budget above was rolled before recording the lease attribution.
        accelerator_hours_spent = dict(budget.accelerator_hours_spent)
        for accelerator, quantity in hours.items():
            accelerator_hours_spent[accelerator] = _decimal_sum(
                accelerator_hours_spent.get(accelerator, 0),
                quantity,
            )
        spent = _decimal_sum(budget.spent, settled_cost)
        crossed = budget.thresholds_crossed_by(spent)
        updated_budget = budget.model_copy(
            update={
                "spent": spent,
                "accelerator_hours_spent": accelerator_hours_spent,
                "alerted_thresholds": tuple(sorted({*budget.alerted_thresholds, *crossed})),
                "revision": budget.revision + 1,
                "updated_by": principal.subject,
                "updated_at": now,
            }
        )
        await tx.put_budget(updated_budget)
        for threshold in crossed:
            # One event per newly crossed threshold, in the same transaction as the
            # spend that crossed it. An alert emitted outside this transaction could
            # survive a rollback and warn about spend that never happened.
            await self._audit_scoped(
                tx,
                principal,
                org_id=lease.org_id,
                project_id=lease.project_id,
                entity_type="budget",
                entity_id=_scope(lease.org_id, lease.project_id),
                action="budget.threshold_crossed",
                payload={
                    "threshold": threshold,
                    "spent": updated_budget.spent,
                    "limit": updated_budget.limit,
                    "period": updated_budget.period,
                    "hard_limit": updated_budget.hard_limit,
                },
            )
        await self._audit_scoped(
            tx,
            principal,
            org_id=lease.org_id,
            project_id=lease.project_id,
            entity_type="budget",
            entity_id=_scope(lease.org_id, lease.project_id),
            action="budget.lease_settled",
            payload={
                "lease_id": lease.id,
                "cost_increment": settled_cost,
                "accelerator_hours_increment": hours,
                "spent": updated_budget.spent,
                "accelerator_hours_spent": updated_budget.accelerator_hours_spent,
            },
        )
        return settled, updated_budget, True

    @staticmethod
    def _lease_clock(lease: LeaseRecord, settled_at: datetime) -> tuple[float, float]:
        """Return ``(queued_seconds, running_seconds)`` for one settled lease.

        Queued is capacity reserved for this workload but not yet executing its
        payload; running is from the lease's first entry into ACTIVE. A lease that
        never ran is pure queue, and both are clamped at zero so a clock skew cannot
        mint negative time.
        """

        held = max((settled_at - lease.created_at).total_seconds(), 0.0)
        if lease.run_started_at is None:
            return held, 0.0
        running = max((settled_at - lease.run_started_at).total_seconds(), 0.0)
        return max(held - running, 0.0), min(running, held)

    @staticmethod
    def _accelerator_hours(
        lease: LeaseRecord,
        elapsed_hours: float,
        workload: WorkloadRecord | None = None,
    ) -> dict[str, float]:
        """Scale the lease's reserved resource shape by hours actually held.

        ``resource_hours_reserved`` is the *shape* multiplied by the submitted
        duration estimate, so dividing it back out recovers the per-hour vector and
        re-scaling by real elapsed time gives settled accelerator-hours. The
        price-based division needs a priced pool; on an unpriced pool the workload's
        submitted duration estimate is the same divisor, so the record supplies it.
        When neither is available there is nothing to divide out and the result is
        the reserved shape unchanged.
        """

        reserved = lease.resource_hours_reserved
        values = {"gpu": float(reserved.gpu), "tpu": float(reserved.tpu), **reserved.custom}
        estimated_hours = lease.reserved_cost / lease.hourly_cost if lease.hourly_cost else 0.0
        if not estimated_hours and workload is not None:
            # The same figure the scheduler multiplied the shape by
            # (``estimated_duration_seconds`` is immutable on a submitted spec), so
            # this divides out exactly what scheduling multiplied in -- on a
            # zero-priced pool where ``reserved_cost`` cannot recover it.
            estimated_hours = workload.spec.estimated_duration_seconds / 3_600
        if not estimated_hours:
            return {name: quantity for name, quantity in values.items() if quantity}
        return costing.scale(values, elapsed_hours / estimated_hours)

    @staticmethod
    def _period_seconds(period: str) -> float | None:
        """Return the length of a budget period, or None when it never rolls."""

        return {
            "hourly": 3_600.0,
            "daily": 86_400.0,
            "weekly": 604_800.0,
            "monthly": 2_592_000.0,
            "quarterly": 7_776_000.0,
            "yearly": 31_536_000.0,
        }.get(period.strip().lower())

    @classmethod
    def _rolled_budget(cls, budget: BudgetRecord, now: datetime) -> BudgetRecord:
        """Reset a budget whose declared period has elapsed.

        ``period`` was a string nothing acted on. That was survivable only while
        ``spent`` held a duration estimate; once it became actual elapsed cost it is a
        true monotone total, so without this a "monthly" budget is a lifetime budget
        and every alert threshold fires exactly once, ever.
        """

        return rolled_budget(budget, now)

    async def _transition_workload(
        self,
        tx: RepositorySession,
        principal: Principal,
        workload: WorkloadRecord,
        target: WorkloadState,
        action: str,
        *,
        updates: WorkloadTransitionUpdates | None = None,
    ) -> WorkloadRecord:
        try:
            ensure_workload_transition(workload.state, target)
        except InvalidTransition as exc:
            raise ConflictError(str(exc)) from exc
        now = utcnow()
        transition_updates: dict[str, tp.Any] = {"state": target, "updated_at": now, **(updates or {})}
        if target in TERMINAL_WORKLOAD_STATES:
            transition_updates["dispatch_reservation"] = None
        updated = workload.model_copy(update=transition_updates)
        await tx.put_workload(updated)
        if target in TERMINAL_WORKLOAD_STATES:
            # Every terminal path funnels through here -- ordinary settlement,
            # cancellation, the batch and DAG coordinators -- so releasing claims
            # at this one point is what makes the reference index complete rather
            # than complete-for-the-paths-someone-remembered.
            await self._release_artifact_references(tx, updated, now=now)
        await self._audit(
            tx,
            principal,
            updated,
            action,
            previous_state=workload.state.value,
            new_state=target.value,
        )
        return updated

    async def _idempotent_resource(
        self,
        tx: RepositorySession,
        scope: str,
        operation: str,
        key: str,
        fingerprint: str,
    ) -> IdempotencyRecord | None:
        existing = await tx.get_idempotency(scope, operation, key)
        if existing is not None and existing.fingerprint != fingerprint:
            raise ConflictError("Idempotency-Key was already used with a different request")
        return existing

    @staticmethod
    async def _dispatch_reservation_may_have_started(
        tx: RepositorySession,
        workload: WorkloadRecord,
    ) -> bool:
        """Return whether requeueing could orphan a cell-agent submission."""

        reservation = workload.dispatch_reservation
        lease_ids = workload.lease_ids or (() if workload.lease_id is None else (workload.lease_id,))
        if (
            reservation is None
            or reservation.generation != workload.generation
            or reservation.workload_attempt != workload.attempt
            or reservation.lease_ids != lease_ids
        ):
            return False
        command = await tx.get_outbox(reservation.outbox_id)
        persisted: WorkloadDispatchReservation | None = None
        if command is not None:
            try:
                persisted = WorkloadDispatchReservation.model_validate(command.payload.get("dispatch_reservation"))
            except Exception:
                persisted = None
        return bool(
            command is not None
            and command.topic == "workload.dispatch"
            and command.aggregate_type == "workload"
            and command.aggregate_id == workload.id
            and command.generation == workload.generation
            and command.state in {OutboxState.PENDING, OutboxState.PROCESSING, OutboxState.FAILED}
            and persisted == reservation
        )

    @staticmethod
    async def _preemption_execution_started(tx: RepositorySession, workload: WorkloadRecord) -> bool:
        if workload.preemption_id is None:
            return False
        command = await tx.get_outbox(workload.preemption_id)
        return bool(
            command is not None
            and command.topic == "workload.preempt"
            and command.state is not OutboxState.SUCCEEDED
            and command.payload.get("execution_started")
            and not command.payload.get("execution_completed")
        )

    async def _bind_idempotency(
        self,
        tx: RepositorySession,
        scope: str,
        operation: str,
        key: str,
        fingerprint: str,
        resource: (
            WorkloadRecord
            | LeaseRecord
            | PolicyRecord
            | BudgetRecord
            | QuotaRecord
            | BudgetGrantRecord
            | OrganizationRecord
            | ProjectRecord
        ),
    ) -> None:
        # Explicit on every branch, including the last. This used to fall through
        # to "budget" for anything unrecognized, which silently bound a new record
        # type to its *scope* instead of its own id -- so a replay looked up the
        # wrong resource and issued a second one.
        if isinstance(resource, WorkloadRecord):
            resource_type, resource_id = "workload", resource.id
        elif isinstance(resource, LeaseRecord):
            resource_type, resource_id = "lease", resource.id
        elif isinstance(resource, PolicyRecord):
            resource_type, resource_id = "policy", _scope(resource.org_id, resource.project_id)
        elif isinstance(resource, QuotaRecord):
            resource_type = "quota"
            resource_id = self._quota_id(resource.org_id, resource.project_id, resource.principal_id)
        elif isinstance(resource, BudgetGrantRecord):
            resource_type, resource_id = "budget_grant", resource.id
        elif isinstance(resource, QuotaGrantRecord):
            resource_type, resource_id = "quota_grant", resource.id
        elif isinstance(resource, BudgetRecord):
            resource_type, resource_id = "budget", _scope(resource.org_id, resource.project_id)
        elif isinstance(resource, ProjectRecord):
            # Before OrganizationRecord: a project carries an org_id too, and
            # ordering these the other way would bind every project under its
            # organization's id, so a replay would return the wrong record.
            resource_type, resource_id = "project", _scope(resource.org_id, resource.id)
        elif isinstance(resource, OrganizationRecord):
            resource_type, resource_id = "organization", resource.id
        else:
            raise TypeError(f"no idempotency binding is defined for {type(resource).__name__}")
        await tx.put_idempotency(
            IdempotencyRecord(
                scope=scope,
                operation=operation,
                key=key,
                fingerprint=fingerprint,
                resource_type=resource_type,
                resource_id=resource_id,
            )
        )

    @staticmethod
    def _quota_id(org_id: str, project_id: str | None, principal_id: str | None) -> str:
        payload = json.dumps((org_id, project_id, principal_id), ensure_ascii=True, separators=(",", ":")).encode()
        return f"quota_{hashlib.sha256(payload).hexdigest()}"

    def _require_quota_admin(self, principal: Principal, org_id: str, project_id: str | None) -> None:
        if project_id is not None:
            self.authorizer.require(principal, Permission.MANAGE_PROJECT, org_id=org_id, project_id=project_id)
            return
        if not any(
            grant.role == Role.FLEET_ADMIN or (grant.role == Role.ORG_ADMIN and grant.org_id == org_id)
            for grant in principal.grants
        ):
            raise AuthorizationError(f"{principal.subject!r} lacks organization quota administration")

    async def _audit(
        self,
        tx: RepositorySession,
        principal: Principal,
        workload: WorkloadRecord,
        action: str,
        *,
        previous_state: str | None = None,
        new_state: str | None = None,
        payload: JsonObject | None = None,
    ) -> AuditEvent:
        return await self._audit_scoped(
            tx,
            principal,
            org_id=workload.org_id,
            project_id=workload.project_id,
            entity_type="workload",
            entity_id=workload.id,
            action=action,
            previous_state=previous_state,
            new_state=new_state,
            payload=payload,
        )
