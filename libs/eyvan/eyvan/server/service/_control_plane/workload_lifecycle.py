# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.domain.governance import has_approval_gate
from eyvan.server.service._control_plane.shared import (
    ACTIVE_LEASE_STATES,
    STUCK_CANCEL_AFTER_S,
    TERMINAL_WORKLOAD_STATES,
    CellState,
    ConflictError,
    IdempotencyRecord,
    InputError,
    JsonValue,
    LeaseState,
    NotFoundError,
    OutboxState,
    Permission,
    Principal,
    RepositorySession,
    WorkloadRecord,
    WorkloadState,
    _fingerprint,
    _host_fault_proof,
    _scope,
    _validate_idempotency_key,
    collect_descendants,
    datetime,
    ensure_workload_transition,
    hashlib,
    timedelta,
    utcnow,
)


class WorkloadLifecycleMixin:
    """Transactional workload lifecycle operations."""

    async def cancel_workload(
        self, principal: Principal, workload_id: str, *, idempotency_key: str, cascade: bool = True
    ) -> tuple[WorkloadRecord, bool]:
        """Request cancellation; pending work is cancelled synchronously.

        Cancelling a parent cancels its whole subtree by default: a launcher's
        children have no independent meaning once the launcher is gone, and
        leaving them running would silently keep spending the project's budget.
        Pass ``cascade=False`` to cancel exactly one workload.
        """

        key = _validate_idempotency_key(idempotency_key)
        async with self.repository.transaction() as tx:
            workload = await self._require_workload(tx, workload_id)
            self.authorizer.require(
                principal, Permission.OPERATE, org_id=workload.org_id, project_id=workload.project_id
            )
            scope = _scope(workload.org_id, workload.project_id)
            # ``cascade`` is part of what was asked for, so it is part of what a
            # replay must match. Omitting it made a full cancel replayed over an
            # earlier ``cascade=False`` call return the first call's handle and do
            # nothing: the caller is told the cancel succeeded while every
            # descendant keeps running, holding the capacity the cancel was meant
            # to release.
            fingerprint = _fingerprint({"workload_id": workload_id, "cascade": cascade})
            existing = await self._idempotent_resource(tx, scope, "workload.cancel", key, fingerprint)
            if existing is not None:
                return workload, True
            if workload.state in {WorkloadState.SUCCEEDED, WorkloadState.FAILED}:
                raise ConflictError(f"cannot cancel a {workload.state.value} workload")
            updated = await self._cancel_one(tx, principal, workload, "workload.cancel_requested")
            if cascade:
                # Descendants are cancelled through the same path as the root, so
                # a RUNNING child receives a `workload.cancel` command and is
                # actually stopped. Transitioning its state alone would leave it
                # holding its accelerator and spending the project's budget while
                # merely *recorded* as cancelling.
                for child in await collect_descendants(workload_id, tx.list_child_workloads):
                    if child.state in TERMINAL_WORKLOAD_STATES:
                        continue
                    await self._cancel_one(tx, principal, child, "workload.cancel_cascaded")
            await self._bind_idempotency(tx, scope, "workload.cancel", key, fingerprint, updated)
            return updated, False

    async def _cancel_one(
        self,
        tx: RepositorySession,
        principal: Principal,
        workload: WorkloadRecord,
        action: str,
    ) -> WorkloadRecord:
        """Transition one workload toward cancellation and enqueue its stop command.

        A PENDING workload has nothing running, so it cancels synchronously.
        Anything further along needs a durable ``workload.cancel`` command: that
        outbox record is the only thing the controller's dispatcher reacts to,
        so omitting it leaves the remote process alive.
        """

        # Every member of a cascade must respect the preemption owner, not
        # only its root. Refusal rolls back the whole cancellation transaction,
        # including earlier ancestors, before any ownership markers are lost.
        if await self._preemption_execution_started(tx, workload):
            raise ConflictError("preemption is already executing; retry cancellation after it completes")
        target = WorkloadState.CANCELLED if workload.state is WorkloadState.PENDING else WorkloadState.CANCELLING
        updated = await self._transition_workload(
            tx,
            principal,
            workload,
            target,
            action,
            updates={
                "preemption_id": None,
                "reclaim_deadline_at": None,
                "reclaim_requested_by": None,
            },
        )
        if target is WorkloadState.CANCELLING:
            for lease in await self._current_leases(tx, updated):
                if lease.revocation_reason == "preemption":
                    await tx.put_lease(lease.model_copy(update={"revocation_reason": None, "updated_at": utcnow()}))
            await self._outbox(
                tx,
                topic="workload.cancel",
                aggregate_type="workload",
                aggregate_id=updated.id,
                idempotency_key=f"cancel:{updated.id}:{updated.generation}",
                generation=updated.generation,
                payload={
                    "workload_id": updated.id,
                    "lease_id": updated.lease_id,
                    "lease_ids": list(updated.lease_ids),
                    "remote_id": updated.remote_id,
                },
            )
        return updated

    async def checkpoint_workload(
        self, principal: Principal, workload_id: str, *, idempotency_key: str
    ) -> tuple[WorkloadRecord, bool]:
        """Request an execution-cell checkpoint for a running workload."""

        key = _validate_idempotency_key(idempotency_key)
        async with self.repository.transaction() as tx:
            workload = await self._require_workload(tx, workload_id)
            self.authorizer.require(
                principal, Permission.OPERATE, org_id=workload.org_id, project_id=workload.project_id
            )
            scope = _scope(workload.org_id, workload.project_id)
            fingerprint = _fingerprint({"workload_id": workload_id})
            existing = await self._idempotent_resource(tx, scope, "workload.checkpoint", key, fingerprint)
            if existing is not None:
                return workload, True
            if not workload.spec.checkpointable:
                raise InputError("workload is not checkpointable")
            if workload.spec.cell_group_count > 1:
                # A multi-*cell* group spans independently fenced cells, so a hook
                # would run under separate mutation contexts with no shared
                # cluster. A multi-*host* slice is different and now supported:
                # NativeSliceExecutor fans the hook across ranks that inherit
                # their own JAX identity. Only the cross-cell case stays rejected.
                raise InputError("checkpoint hooks are not supported for multi-cell TPU pod groups")
            updated = await self._transition_workload(
                tx, principal, workload, WorkloadState.CHECKPOINTING, "workload.checkpoint_requested"
            )
            await self._outbox(
                tx,
                topic="workload.checkpoint",
                aggregate_type="workload",
                aggregate_id=updated.id,
                idempotency_key=(
                    f"checkpoint:{updated.id}:{updated.generation}:{hashlib.sha256(key.encode()).hexdigest()[:16]}"
                ),
                generation=updated.generation,
                payload={
                    "workload_id": updated.id,
                    "lease_id": updated.lease_id,
                    "remote_id": updated.remote_id,
                    "hook": updated.spec.checkpoint_hook,
                    "output_uri": updated.spec.checkpoint_output_uri,
                },
            )
            await self._bind_idempotency(tx, scope, "workload.checkpoint", key, fingerprint, updated)
            return updated, False

    async def reclaim_workload(
        self,
        principal: Principal,
        workload_id: str,
        *,
        idempotency_key: str,
        reason: str,
        grace_seconds: int | None = None,
        requested_by: str | None = None,
    ) -> tuple[WorkloadRecord, bool]:
        """Take one running workload's capacity back without ending the workload.

        An administrative preemption: the controller checkpoints the workload
        (when it is checkpointable, has a hook and runs on one cell), stops its
        remote process, and requeues it to PENDING at a new generation so it
        resumes from the checkpoint. Not a cancel -- the workload stays runnable.

        State choice. The workload moves to CHECKPOINTING when a hook will run
        and to CANCELLING otherwise -- exactly the two states priority
        preemption uses for its victims, and both are allowed from RUNNING and
        accepted by ``settle_preempted_workload``. In the same transaction every
        current lease is marked ``revocation_reason="preemption"`` (ACTIVE leases
        become REVOKING; QUARANTINED ones keep their state, since the lease state
        machine has no QUARANTINED -> REVOKING edge). That marker is what keeps
        an in-flight reclaim from being misread: observation treats SUCCEEDED,
        FAILED, STOPPED, LOST and NOT_FOUND as preemption-owned and leaves them to
        the command, so the STOPPED that the reclaim itself causes is never
        settled as CANCELLED or FAILED; lease recovery and host-fault attribution
        skip the lease; the priority planner never picks a REVOKING lease as a
        victim; and a lost cell requeues the workload through the
        preemption-owned recovery path. ``preemption_id`` stays None, which is
        how a reclaim differs from Eyvan's own priority preemption. A user cancel
        still wins: ``_cancel_one`` clears the marker, after which the reclaim
        command no longer owns the lease and retires as stale.

        Multi-cell groups are stopped without a hook (checkpoint hooks are not
        supported across independently fenced cells), keeping whatever
        ``checkpoint_uri`` the workload already recorded.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        key = _validate_idempotency_key(idempotency_key)
        reason = reason.strip()
        if not reason:
            raise InputError("reclaim reason must not be empty")
        if len(reason) > 500:
            raise InputError("reclaim reason must be at most 500 characters")
        if grace_seconds is not None and (isinstance(grace_seconds, bool) or grace_seconds <= 0):
            raise InputError("reclaim grace_seconds must be a positive integer")
        async with self.repository.transaction() as tx:
            await self._lock_settlement(tx, workload_id)
            workload = await self._require_workload(tx, workload_id)
            scope = _scope(workload.org_id, workload.project_id)
            fingerprint = _fingerprint({"workload_id": workload_id, "reason": reason})
            existing = await self._idempotent_resource(tx, scope, "workload.reclaim", key, fingerprint)
            if existing is not None:
                return workload, True
            if workload.state is not WorkloadState.RUNNING:
                raise ConflictError(f"only a running workload can be reclaimed; this one is {workload.state.value}")
            if workload.preemption_id is not None:
                raise ConflictError("workload is part of a priority preemption; retry after it completes")
            if workload.remote_id is None:
                raise ConflictError("running workload has no remote execution to stop")
            leases = await self._current_leases(tx, workload)
            if not leases:
                raise ConflictError("running workload holds no lease")
            for lease in leases:
                if (
                    lease.state not in {LeaseState.ACTIVE, LeaseState.QUARANTINED}
                    or lease.revocation_reason is not None
                    or lease.generation != workload.generation
                ):
                    raise ConflictError(f"lease {lease.id!r} is already being revoked")
            spec = workload.spec
            checkpoint = bool(spec.checkpointable and spec.checkpoint_hook and spec.cell_group_count == 1)
            target = WorkloadState.CHECKPOINTING if checkpoint else WorkloadState.CANCELLING
            ensure_workload_transition(workload.state, target)
            now = utcnow()
            default_grace = int(getattr(self, "default_reclaim_grace_s", 600))
            if grace_seconds is not None:
                effective_grace_s = grace_seconds
            elif spec.checkpoint_grace_seconds > 0:
                effective_grace_s = spec.checkpoint_grace_seconds
            else:
                effective_grace_s = default_grace if default_grace > 0 else 600
            deadline_at = now + timedelta(seconds=effective_grace_s)
            requester = (
                requested_by.strip()
                if isinstance(requested_by, str) and requested_by.strip()
                else (
                    principal.subject.removeprefix("reclaim:")
                    if principal.subject.startswith("reclaim:")
                    else principal.subject
                )
            )
            updated = workload.model_copy(
                update={
                    "state": target,
                    "scheduling_reason": f"reclaim requested by {principal.subject}: {reason}",
                    "reclaim_deadline_at": deadline_at,
                    "reclaim_requested_by": requester,
                    "updated_at": now,
                }
            )
            await tx.put_workload(updated)
            for lease in leases:
                await tx.put_lease(
                    lease.model_copy(
                        update={
                            "state": LeaseState.REVOKING if lease.state is LeaseState.ACTIVE else lease.state,
                            "revocation_reason": "preemption",
                            "updated_at": now,
                        }
                    )
                )
            lease_ids: list[JsonValue] = [lease.id for lease in leases]
            # What an aborted checkpoint phase restores each lease to.
            lease_states: dict[str, JsonValue] = {lease.id: lease.state.value for lease in leases}
            command = await self._outbox(
                tx,
                topic="workload.reclaim",
                aggregate_type="workload",
                aggregate_id=updated.id,
                idempotency_key=(
                    f"reclaim:{updated.id}:{updated.generation}:{hashlib.sha256(key.encode()).hexdigest()[:16]}"
                ),
                generation=updated.generation,
                payload={
                    "workload_id": updated.id,
                    "lease_id": updated.lease_id,
                    "lease_ids": lease_ids,
                    "lease_states": lease_states,
                    "remote_id": updated.remote_id,
                    "checkpoint": checkpoint,
                    "hook": spec.checkpoint_hook if checkpoint else None,
                    "output_uri": spec.checkpoint_output_uri if checkpoint else None,
                    "grace_period_seconds": spec.checkpoint_grace_seconds,
                    "reclaim_grace_seconds": effective_grace_s,
                    "reclaim_deadline_at": deadline_at.isoformat(),
                    "reclaim_deadline_epoch_s": deadline_at.timestamp(),
                    "reclaim_policy": spec.reclaim_policy,
                    "requested_by": requester,
                    "reason": reason,
                    "actor": principal.subject,
                },
            )
            await self._audit(
                tx,
                principal,
                updated,
                "workload.reclaim_requested",
                previous_state=workload.state.value,
                new_state=updated.state.value,
                payload={
                    "reason": reason,
                    "checkpoint": checkpoint,
                    "lease_ids": lease_ids,
                    "command_id": command.id,
                    "grace_seconds": effective_grace_s,
                    "deadline_at": deadline_at.isoformat(),
                    "policy": spec.reclaim_policy,
                    "requested_by": requester,
                },
            )
            await self._bind_idempotency(tx, scope, "workload.reclaim", key, fingerprint, updated)
            return updated, False

    @staticmethod
    def _manual_retry_budget_update(workload: WorkloadRecord) -> dict[str, int]:
        """Consume one retry from the budget matching the workload's failure class.

        A manual retry of a ``FAILED`` workload draws from the failure budget
        first; a ``CANCELLED`` workload draws from the capacity-loss budget
        first. When the matching budget is exhausted the other budget may
        cover the retry, and when both are exhausted the retry is rejected.
        """

        budgets: tuple[tuple[str, int, int], ...]
        if workload.state is WorkloadState.FAILED:
            budgets = (
                ("failure_count", workload.failure_count, workload.spec.max_failure_retries),
                ("capacity_loss_count", workload.capacity_loss_count, workload.spec.max_retries),
            )
        else:
            budgets = (
                ("capacity_loss_count", workload.capacity_loss_count, workload.spec.max_retries),
                ("failure_count", workload.failure_count, workload.spec.max_failure_retries),
            )
        for field_name, count, limit in budgets:
            if count < limit:
                return {field_name: count + 1}
        raise ConflictError("workload retry limit has been exhausted")

    async def retry_workload(
        self, principal: Principal, workload_id: str, *, idempotency_key: str
    ) -> tuple[WorkloadRecord, bool]:
        """Return a failed or cancelled workload to pending with a new generation."""

        key = _validate_idempotency_key(idempotency_key)
        async with self.repository.transaction() as tx:
            workload = await self._require_workload(tx, workload_id)
            self.authorizer.require(
                principal, Permission.OPERATE, org_id=workload.org_id, project_id=workload.project_id
            )
            scope = _scope(workload.org_id, workload.project_id)
            fingerprint = _fingerprint({"workload_id": workload_id})
            existing = await self._idempotent_resource(tx, scope, "workload.retry", key, fingerprint)
            if existing is not None:
                return workload, True
            if workload.state not in {WorkloadState.FAILED, WorkloadState.CANCELLED}:
                raise ConflictError("only failed or cancelled workloads can be retried")
            if has_approval_gate(workload):
                raise ConflictError("workload was never released from approval; resubmit it as a new workload")
            budget_update = self._manual_retry_budget_update(workload)
            ensure_workload_transition(workload.state, WorkloadState.PENDING)
            # Team governance (off unless team mode): a retry re-queues work, so
            # the owner's callable rule and hard caps apply as at admission.
            await self._governance_readmission(tx, workload, f"cannot retry {workload.id}: ")  # type: ignore[attr-defined]
            now = utcnow()
            next_generation = workload.generation + 1
            updated = workload.model_copy(
                update={
                    "state": WorkloadState.PENDING,
                    "attempt": workload.attempt + 1,
                    **budget_update,
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
                    "failure_reason": None,
                    "autoscale_block_reason": None,
                    "queue_admission_denial": None,
                    "scheduling_reason": None,
                    "reclaim_deadline_at": None,
                    "reclaim_requested_by": None,
                    "updated_at": now,
                }
            )
            await self._put_workload_keeping_references(tx, updated)
            await self._audit(
                tx,
                principal,
                updated,
                "workload.retried",
                previous_state=workload.state.value,
                new_state=updated.state.value,
                payload={"attempt": updated.attempt, "generation": updated.generation},
            )
            await self._bind_idempotency(tx, scope, "workload.retry", key, fingerprint, updated)
            return updated, False

    async def force_finalize_stuck_workload(
        self,
        principal: Principal,
        workload_id: str,
        *,
        reason: str,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> tuple[WorkloadRecord, bool]:
        """Settle a workload whose stop the worker will never confirm.

        A worker that never confirms ``/stop`` can leave a workload CANCELLING
        with an active lease. That lease blocks cell deletion, so fleet teardown
        cannot complete. ``force_finalize_preemption`` attests preemption victims
        and does not resolve an ordinary cancel; this is the explicit escape path.

        **This declares a process dead that nobody confirmed stopping.** That is
        why a cell is drained rather than returned to service: the workload may
        still be running on it, and scheduling something else onto a host in that
        state would be worse than the deadlock. The capacity is given up, not
        recycled.

        *Unless something did confirm it.* A cell that node health has quarantined
        for this execution carries the host fault it matched, and those faults are
        the ones where the accelerator runtime is dead -- so the process that
        needed it is not still running. That cell is **reclaimed**: it is left on
        the replacement path node health already put it on, rather than
        additionally surrendered out of a caution the evidence has answered. Every
        cell is judged on its own record, so a gang whose hosts disagree drains the
        ones nobody could account for and reclaims the ones that are accounted
        for. The audit entry names which happened to each cell and why -- an
        operator reading "we gave up a v5p" is owed the difference between
        "because we could not know" and "because we knew".

        Refused unless the stop has genuinely stopped making progress -- a live
        cancel in flight is not this, and a break-glass that also works on
        healthy work is one that gets used on healthy work.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        key = _validate_idempotency_key(idempotency_key)
        reason = reason.strip()
        if not reason:
            raise InputError("force-finalize reason must not be empty")
        moment = now or utcnow()
        fingerprint = _fingerprint({"workload_id": workload_id, "reason": reason})
        async with self.repository.transaction() as tx:
            existing = await self._idempotent_resource(tx, "fleet", "workload.force_finalize", key, fingerprint)
            workload = await self._require_workload(tx, workload_id)
            if existing is not None:
                return workload, True
            if workload.state in TERMINAL_WORKLOAD_STATES:
                raise ConflictError(f"workload {workload_id!r} is already {workload.state.value}")
            if workload.state is not WorkloadState.CANCELLING:
                raise ConflictError(
                    f"workload {workload_id!r} is {workload.state.value}; only a stuck cancellation is "
                    "force-finalized, because this gives up the cell rather than returning it"
                )
            stuck = [
                command
                for command in await tx.list_outbox_for_aggregates(
                    "workload", {workload_id: workload.generation}, topic="workload.cancel"
                )
                if command.state in {OutboxState.FAILED, OutboxState.PENDING, OutboxState.PROCESSING}
                and command.created_at <= moment - timedelta(seconds=STUCK_CANCEL_AFTER_S)
            ]
            if not stuck:
                raise ConflictError(
                    f"workload {workload_id!r} has no cancel older than {STUCK_CANCEL_AFTER_S:g}s; "
                    "a cancellation still in flight is not stuck"
                )
            for command in stuck:
                await tx.put_outbox(
                    command.model_copy(
                        update={
                            "state": OutboxState.DISCARDED,
                            "claimed_by": None,
                            "claimed_until": None,
                            "last_error": f"force-finalized: {reason}",
                            "updated_at": moment,
                        }
                    )
                )
            held: set[str] = set()
            for lease in await tx.list_leases(workload_id=workload_id):
                if lease.state not in ACTIVE_LEASE_STATES:
                    continue
                await self._settle_lease(tx, principal, workload, lease, reason="force-finalized")
                held.add(lease.cell_id)
            drained: set[str] = set()
            reclaimed: set[str] = set()
            outcomes: list[JsonValue] = []
            for cell_id in sorted(held):
                cell = await tx.get_cell(cell_id)
                if cell is None or cell.state in {CellState.DELETED, CellState.DELETING}:
                    continue
                proof = _host_fault_proof(cell, workload)
                if proof is None:
                    drained.add(cell_id)
                    await tx.put_cell(cell.model_copy(update={"state": CellState.DRAINING, "updated_at": moment}))
                    outcomes.append(
                        {
                            "cell_id": cell_id,
                            "outcome": "drained",
                            "reason": "nothing confirmed the process stopped, so the host is given up",
                        }
                    )
                    continue
                reclaimed.add(cell_id)
                outcomes.append(
                    {
                        "cell_id": cell_id,
                        "outcome": "reclaimed",
                        "reason": f"{proof.diagnosis}, so the process that needed it is gone",
                        "host_fault": proof.code.value,
                        "evidence": proof.evidence,
                    }
                )
            settled = workload.model_copy(
                update={
                    "state": WorkloadState.CANCELLED,
                    "dispatch_reservation": None,
                    "failure_reason": (
                        f"force-finalized after an unconfirmed stop: {reason}"
                        if drained or not reclaimed
                        else f"force-finalized after a host fault accounted for every cell it held: {reason}"
                    ),
                    "updated_at": moment,
                }
            )
            await tx.put_workload(settled)
            # A terminal write outside ``_transition_workload`` -- kept direct so
            # the force-finalize audit record stays one scoped event with its
            # capacity outcomes -- so it releases artifact references itself, or
            # the CANCELLED workload's active claims would pin its artifacts
            # against reclamation forever.
            await self._release_artifact_references(tx, settled, now=moment)
            await self._bind_idempotency(tx, "fleet", "workload.force_finalize", key, fingerprint, settled)
            await self._audit_scoped(
                tx,
                principal,
                org_id=workload.org_id,
                project_id=workload.project_id,
                entity_type="workload",
                entity_id=workload_id,
                action="workload.force_finalized",
                payload={
                    "reason": reason,
                    "discarded_commands": sorted(command.id for command in stuck),
                    "drained_cells": sorted(drained),
                    "reclaimed_cells": sorted(reclaimed),
                    "capacity_outcomes": outcomes,
                },
            )
            return settled, False

    async def force_finalize_preemption(
        self,
        principal: Principal,
        command_id: str,
        *,
        reason: str,
        idempotency_key: str,
    ) -> tuple[WorkloadRecord, bool]:
        """Attest that every unresolved victim is terminal and retry settlement."""

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        key = _validate_idempotency_key(idempotency_key)
        reason = reason.strip()
        if not reason:
            raise InputError("force-finalize reason must not be empty")
        fingerprint = _fingerprint({"command_id": command_id, "reason": reason})
        async with self.repository.transaction() as tx:
            existing = await self._idempotent_resource(tx, "fleet", "preemption.force_finalize", key, fingerprint)
            command = await tx.get_outbox(command_id)
            if command is None or command.topic != "workload.preempt":
                raise NotFoundError(f"preemption command {command_id!r} was not found")
            target = await self._require_workload(tx, str(command.payload["target_workload_id"]))
            if existing is not None:
                return target, True
            if command.payload.get("execution_completed"):
                raise ConflictError("preemption command is already complete")
            if str(command.payload.get("phase")) != "stopping":
                raise ConflictError("only a preemption in the stop phase can be force-finalized")
            victims = command.payload.get("victims")
            if not isinstance(victims, list) or not victims:
                raise ConflictError("preemption command has no durable victim set")
            outcomes = {
                str(lease_id): str(outcome)
                for lease_id, outcome in dict(command.payload.get("victim_outcomes") or {}).items()
            }
            for victim in victims:
                if not isinstance(victim, dict) or "lease_id" not in victim:
                    raise ConflictError("preemption command contains an invalid victim")
                outcomes.setdefault(str(victim["lease_id"]), "preempted")
            now = utcnow()
            await tx.put_outbox(
                command.model_copy(
                    update={
                        "state": OutboxState.FAILED,
                        "claimed_by": None,
                        "claimed_until": None,
                        "available_at": now,
                        "last_error": "force-finalize requested",
                        "payload": {
                            **command.payload,
                            "victim_outcomes": outcomes,
                            "stopped_victims": list(outcomes),
                            "force_finalize_reason": reason,
                            "force_finalize_actor": principal.subject,
                        },
                        "updated_at": now,
                    }
                )
            )
            await tx.put_idempotency(
                IdempotencyRecord(
                    scope="fleet",
                    operation="preemption.force_finalize",
                    key=key,
                    fingerprint=fingerprint,
                    resource_type="workload",
                    resource_id=target.id,
                )
            )
            await self._audit_scoped(
                tx,
                principal,
                org_id=target.org_id,
                project_id=target.project_id,
                entity_type="preemption",
                entity_id=command.id,
                action="preemption.force_finalize_requested",
                payload={"reason": reason, "victim_count": len(victims)},
            )
            return target, False
