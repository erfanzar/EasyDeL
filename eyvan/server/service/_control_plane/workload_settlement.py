# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.server.service._control_plane.shared import (
    TERMINAL_WORKLOAD_STATES,
    BudgetRecord,
    CellState,
    ConflictError,
    InputError,
    LeaseRecord,
    LeaseState,
    OutboxRecord,
    OutboxState,
    Permission,
    Principal,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)


class WorkloadSettlementMixin:
    """Transactional workload settlement operations."""

    async def transition_workload(
        self,
        principal: Principal,
        workload_id: str,
        target: WorkloadState,
        *,
        action: str = "workload.state_changed",
        remote_id: str | None = None,
        checkpoint_uri: str | None = None,
        failure_reason: str | None = None,
        expected_generation: int | None = None,
        expected_lease_id: str | None = None,
        expected_state: WorkloadState | None = None,
    ) -> WorkloadRecord:
        """Apply a reconciler-observed workload transition with audit history."""

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            workload = await self._require_workload(tx, workload_id)
            self._validate_expected_workload(workload, expected_generation, expected_lease_id, expected_state)
            updates = {
                key: value
                for key, value in {
                    "remote_id": remote_id,
                    "checkpoint_uri": checkpoint_uri,
                    "failure_reason": failure_reason,
                }.items()
                if value is not None
            }
            return await self._transition_workload(tx, principal, workload, target, action, updates=updates)

    async def settle_terminal_workload(
        self,
        principal: Principal,
        workload_id: str,
        target: WorkloadState,
        *,
        action: str = "workload.terminal",
        remote_id: str | None = None,
        checkpoint_uri: str | None = None,
        failure_reason: str | None = None,
        expected_generation: int | None = None,
        expected_lease_id: str | None = None,
        expected_state: WorkloadState | None = None,
        expected_cell_generation: int | None = None,
        expected_cell_state: CellState | None = None,
        expected_unavailable_reason: str | None = None,
        expected_cell_id: str | None = None,
        #: Finish a workload whose lease rows are gone. Off by default: a missing
        #: lease is normally a real inconsistency worth refusing, and only the
        #: controller's orphan sweep -- which exists precisely to clear that state --
        #: has the standing to say the absence is the point.
        settle_without_leases: bool = False,
        #: A pre-admission refusal may settle only while its exact delivery still
        #: owns the claim and no submit reservation has appeared. Ordinary terminal
        #: observations retain their existing semantics when this is omitted.
        expected_dispatch_command: OutboxRecord | None = None,
    ) -> tuple[WorkloadRecord, LeaseRecord | None, BudgetRecord | None]:
        """Atomically finish a workload and settle its current lease reservation.

        A lease's durable ``settled_at`` marker is the idempotency boundary. Replaying
        the same observed terminal state therefore returns the current records without
        incrementing budget spend or emitting duplicate settlement events.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        if target not in TERMINAL_WORKLOAD_STATES:
            raise InputError(f"{target.value} is not a terminal workload state")
        async with self.repository.transaction() as tx:
            await self._lock_settlement(tx, workload_id)
            workload = await self._require_workload(tx, workload_id)
            self._validate_expected_workload(workload, expected_generation, expected_lease_id, expected_state)
            if expected_dispatch_command is not None:
                command = await tx.get_outbox(expected_dispatch_command.id)
                if (
                    workload.dispatch_reservation is not None
                    or command is None
                    or command.topic != "workload.dispatch"
                    or command.aggregate_type != "workload"
                    or command.aggregate_id != workload.id
                    or command.generation != workload.generation
                    or command.attempts != expected_dispatch_command.attempts
                    or command.state is not OutboxState.PROCESSING
                    or command.claimed_by is None
                    or command.claimed_by != expected_dispatch_command.claimed_by
                    or command.claimed_until is None
                    or command.claimed_until <= utcnow()
                    or "dispatch_reservation" in command.payload
                ):
                    raise ConflictError("dispatch refusal no longer owns an unreserved workload and live claim")
            await self._lock_admission_scope(tx, workload.org_id, workload.project_id)
            await self._validate_expected_cell_snapshot(
                tx,
                workload,
                expected_cell_generation,
                expected_cell_state,
                expected_unavailable_reason,
                expected_cell_id,
            )
            updates = {
                key: value
                for key, value in {
                    "remote_id": remote_id,
                    "checkpoint_uri": checkpoint_uri,
                    "failure_reason": failure_reason,
                }.items()
                if value is not None
            }
            if workload.state is not target:
                workload = await self._transition_workload(
                    tx,
                    principal,
                    workload,
                    target,
                    action,
                    updates=updates,
                )
            try:
                leases = await self._current_leases(tx, workload)
            except ConflictError:
                if not settle_without_leases:
                    raise
                # The lease rows are gone, which is exactly why this workload could
                # not be settled by any ordinary path. There is nothing to settle
                # against -- no reserved cost to charge, no capacity to release -- so
                # the workload is finished on its own.
                leases = ()
            if not leases:
                return workload, None, await tx.get_budget(workload.org_id, workload.project_id)
            settled_leases: list[LeaseRecord] = []
            budget: BudgetRecord | None = None
            for lease in leases:
                settled_lease, budget, _ = await self._settle_lease(
                    tx,
                    principal,
                    workload,
                    lease,
                    reason=target.value,
                )
                settled_leases.append(settled_lease)
            return workload, settled_leases[0], budget

    async def settle_preempted_workload(
        self,
        principal: Principal,
        workload_id: str,
        *,
        checkpoint_uri: str | None = None,
        scheduling_reason: str = "preempted; waiting to resume",
        action: str = "workload.preempted",
        expected_generation: int | None = None,
        expected_lease_id: str | None = None,
        expected_state: WorkloadState | None = None,
        require_preemption_owner: bool = False,
        provider_preempted: bool = False,
        expected_cell_generation: int | None = None,
        expected_cell_state: CellState | None = None,
        expected_unavailable_reason: str | None = None,
        expected_cell_id: str | None = None,
        recycle: bool = False,
    ) -> tuple[WorkloadRecord, LeaseRecord | None, BudgetRecord | None]:
        """Atomically settle a preempted lease and requeue its workload.

        ``provider_preempted`` charges the restart to the preemption budget
        instead of the capacity-loss one. Not the same thing as
        ``require_preemption_owner``, which is about *Eyvan's own* priority
        preemption -- one is the fleet taking capacity back from a lower-priority
        job, the other is the cloud taking it back from the fleet.

        ``recycle`` charges neither budget. It is the plane retiring a cell
        that runs a stale build -- maintenance the workload did not cause and
        must not pay for, and charging it would be worse than unfair: both
        budgets default to zero, so the recycling would never run at all for
        a submission that did not opt into retries. The bound lives with the
        caller, which only reaches this path after draining the cell that
        made the requeue necessary.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            await self._lock_settlement(tx, workload_id)
            workload = await self._require_workload(tx, workload_id)
            self._validate_expected_workload(workload, expected_generation, expected_lease_id, expected_state)
            await self._lock_admission_scope(tx, workload.org_id, workload.project_id)
            await self._validate_expected_cell_snapshot(
                tx,
                workload,
                expected_cell_generation,
                expected_cell_state,
                expected_unavailable_reason,
                expected_cell_id,
            )
            if workload.state is WorkloadState.PENDING and workload.lease_id is None:
                return workload, None, await tx.get_budget(workload.org_id, workload.project_id)
            if await self._dispatch_reservation_may_have_started(tx, workload):
                raise ConflictError("cannot requeue while a workload dispatch may have started")
            if workload.state not in {
                WorkloadState.LEASED,
                WorkloadState.DISPATCHING,
                WorkloadState.RUNNING,
                WorkloadState.CHECKPOINTING,
                WorkloadState.CANCELLING,
            }:
                raise ConflictError(f"cannot requeue a preempted {workload.state.value} workload")
            leases = await self._current_leases(tx, workload)
            if not leases:
                raise ConflictError("preempted workload does not have a current lease")
            if require_preemption_owner and any(lease.revocation_reason != "preemption" for lease in leases):
                raise ConflictError("preemption no longer owns the workload lease")
            settled_leases: list[LeaseRecord] = []
            budget: BudgetRecord | None = None
            for lease in leases:
                settled_lease, budget, _ = await self._settle_lease(
                    tx,
                    principal,
                    workload,
                    lease,
                    reason="preempted",
                )
                settled_leases.append(settled_lease)
            now = utcnow()
            next_generation = workload.generation + 1
            resumed = workload.model_copy(
                update={
                    "state": WorkloadState.PENDING,
                    "attempt": workload.attempt + 1,
                    **(
                        {}
                        if recycle
                        else (
                            {"preemption_count": workload.preemption_count + 1}
                            if provider_preempted
                            else {"capacity_loss_count": workload.capacity_loss_count + 1}
                        )
                    ),
                    "generation": next_generation,
                    "spec": self.result_spec_for_generation(
                        workload.spec,
                        workload_id=workload.id,
                        generation=next_generation,
                    ),
                    "lease_id": None,
                    "lease_ids": (),
                    "dispatch_reservation": None,
                    "remote_id": None,
                    "service_endpoint": None,
                    "checkpoint_uri": checkpoint_uri,
                    "autoscale_block_reason": None,
                    "queue_admission_denial": None,
                    "scheduling_reason": scheduling_reason,
                    "updated_at": now,
                }
            )
            await self._put_workload_keeping_references(tx, resumed)
            await self._audit(
                tx,
                principal,
                resumed,
                action,
                previous_state=workload.state.value,
                new_state=resumed.state.value,
                payload={"checkpoint_uri": checkpoint_uri, "released_lease_ids": [lease.id for lease in settled_leases]},
            )
            return resumed, settled_leases[0], budget

    async def settle_failed_workload(
        self,
        principal: Principal,
        workload_id: str,
        *,
        checkpoint_uri: str | None = None,
        scheduling_reason: str = "execution failed; waiting to retry",
        action: str = "workload.failure_retry_queued",
        expected_generation: int | None = None,
        expected_lease_id: str | None = None,
        expected_state: WorkloadState | None = None,
        expected_cell_generation: int | None = None,
        expected_cell_state: CellState | None = None,
        expected_unavailable_reason: str | None = None,
        expected_cell_id: str | None = None,
    ) -> tuple[WorkloadRecord, LeaseRecord | None, BudgetRecord | None]:
        """Atomically settle a failed execution's lease and requeue its workload.

        This is the user-code failure class: it consumes the failure retry
        budget (``spec.max_failure_retries``), never the capacity-loss budget,
        and preserves any committed checkpoint so the retry resumes from it.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            await self._lock_settlement(tx, workload_id)
            workload = await self._require_workload(tx, workload_id)
            self._validate_expected_workload(workload, expected_generation, expected_lease_id, expected_state)
            await self._lock_admission_scope(tx, workload.org_id, workload.project_id)
            await self._validate_expected_cell_snapshot(
                tx,
                workload,
                expected_cell_generation,
                expected_cell_state,
                expected_unavailable_reason,
                expected_cell_id,
            )
            if workload.state is WorkloadState.PENDING and workload.lease_id is None:
                return workload, None, await tx.get_budget(workload.org_id, workload.project_id)
            if await self._dispatch_reservation_may_have_started(tx, workload):
                raise ConflictError("cannot requeue while a workload dispatch may have started")
            if workload.state not in {
                WorkloadState.LEASED,
                WorkloadState.DISPATCHING,
                WorkloadState.RUNNING,
                WorkloadState.CHECKPOINTING,
            }:
                raise ConflictError(f"cannot requeue a failed {workload.state.value} workload")
            if workload.failure_count >= workload.spec.max_failure_retries:
                raise ConflictError("workload failure retry limit has been exhausted")
            leases = await self._current_leases(tx, workload)
            if not leases:
                raise ConflictError("failed workload does not have a current lease")
            settled_leases: list[LeaseRecord] = []
            budget: BudgetRecord | None = None
            for lease in leases:
                settled_lease, budget, _ = await self._settle_lease(
                    tx,
                    principal,
                    workload,
                    lease,
                    reason="failed",
                )
                settled_leases.append(settled_lease)
            now = utcnow()
            next_generation = workload.generation + 1
            resumed = workload.model_copy(
                update={
                    "state": WorkloadState.PENDING,
                    "attempt": workload.attempt + 1,
                    "failure_count": workload.failure_count + 1,
                    "generation": next_generation,
                    "spec": self.result_spec_for_generation(
                        workload.spec,
                        workload_id=workload.id,
                        generation=next_generation,
                    ),
                    "lease_id": None,
                    "lease_ids": (),
                    "dispatch_reservation": None,
                    "remote_id": None,
                    "service_endpoint": None,
                    "checkpoint_uri": checkpoint_uri,
                    "autoscale_block_reason": None,
                    "queue_admission_denial": None,
                    "scheduling_reason": scheduling_reason,
                    "updated_at": now,
                }
            )
            await self._put_workload_keeping_references(tx, resumed)
            await self._audit(
                tx,
                principal,
                resumed,
                action,
                previous_state=workload.state.value,
                new_state=resumed.state.value,
                payload={"checkpoint_uri": checkpoint_uri, "released_lease_ids": [lease.id for lease in settled_leases]},
            )
            return resumed, settled_leases[0], budget

    async def requeue_unstarted_workload(
        self,
        principal: Principal,
        workload_id: str,
        *,
        scheduling_reason: str,
        expected_generation: int,
        expected_lease_id: str,
        expected_cell_generation: int,
        expected_cell_state: CellState,
        expected_unavailable_reason: str | None,
        expected_cell_id: str | None = None,
    ) -> tuple[WorkloadRecord, LeaseRecord]:
        """Release an undispatched lease without charging its reservation."""

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            await self._lock_settlement(tx, workload_id)
            workload = await self._require_workload(tx, workload_id)
            self._validate_expected_workload(
                workload,
                expected_generation,
                expected_lease_id,
                WorkloadState.LEASED,
            )
            await self._lock_admission_scope(tx, workload.org_id, workload.project_id)
            await self._validate_expected_cell_snapshot(
                tx,
                workload,
                expected_cell_generation,
                expected_cell_state,
                expected_unavailable_reason,
                expected_cell_id,
            )
            if await self._dispatch_reservation_may_have_started(tx, workload):
                raise ConflictError("cannot requeue while a workload dispatch may have started")
            if workload.state is not WorkloadState.LEASED or workload.remote_id is not None:
                raise ConflictError("only an undispatched leased workload can be requeued without settlement")
            leases = await self._current_leases(tx, workload)
            if not leases or any(lease.state not in {LeaseState.PENDING, LeaseState.QUARANTINED} for lease in leases):
                raise ConflictError("undispatched workload does not own a releasable pending lease")
            now = utcnow()
            released_leases = tuple(
                lease.model_copy(
                    update={
                        "state": LeaseState.RELEASED,
                        "settlement_reason": "unstarted",
                        "settled_at": now,
                        "updated_at": now,
                    }
                )
                for lease in leases
            )
            next_generation = workload.generation + 1
            resumed = workload.model_copy(
                update={
                    "state": WorkloadState.PENDING,
                    "generation": next_generation,
                    "spec": self.result_spec_for_generation(
                        workload.spec,
                        workload_id=workload.id,
                        generation=next_generation,
                    ),
                    "lease_id": None,
                    "lease_ids": (),
                    "dispatch_reservation": None,
                    "remote_id": None,
                    "service_endpoint": None,
                    "autoscale_block_reason": None,
                    "queue_admission_denial": None,
                    "scheduling_reason": scheduling_reason,
                    "updated_at": now,
                }
            )
            for released in released_leases:
                await tx.put_lease(released)
            await self._put_workload_keeping_references(tx, resumed)
            await self._audit_scoped(
                tx,
                principal,
                org_id=workload.org_id,
                project_id=workload.project_id,
                entity_type="lease",
                entity_id=leases[0].id,
                action="lease.released_unstarted",
                previous_state=leases[0].state.value,
                new_state=released_leases[0].state.value,
            )
            await self._audit(
                tx,
                principal,
                resumed,
                "workload.dispatch_requeued",
                previous_state=workload.state.value,
                new_state=resumed.state.value,
                payload={"released_lease_ids": [lease.id for lease in released_leases]},
            )
            return resumed, released_leases[0]
