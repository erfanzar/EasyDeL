# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The multi-phase preemption protocol and its checkpoint hooks."""

from __future__ import annotations

import asyncio
import typing as tp
from datetime import timedelta

from pydantic import ValidationError

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.records import (
    CellRecord,
    CellState,
    LeaseRecord,
    LeaseState,
    OutboxRecord,
    OutboxState,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller._support import (
    _PREEMPTION_TERMINAL_WORKLOAD_STATUSES,
    PreemptionCheckpointError,
    StaleOutboxCommand,
    _LaunchedCheckpointHook,
    _PreemptionCheckpointUris,
    _PreemptionStoppedVictims,
    _PreemptionVictim,
    _PreemptionVictimOutcome,
    _PreemptionVictimOutcomes,
    logger,
)
from eyvan.server.persistence.base import RepositorySession
from eyvan.server.service.control_plane import storage_scope
from eyvan.server.service.errors import InputError
from eyvan.server.service.state import ACTIVE_LEASE_STATES
from eyvan.worker.agent.client import (
    CellAgentClient,
    CellAgentStaleMutationError,
)
from eyvan.worker.agent.models import (
    CheckpointRequest,
    MutationModel,
    WorkloadMutationRequest,
)


class _ControllerPreemption:
    """The multi-phase preemption protocol and its checkpoint hooks.

    A concern of :class:`~eyvan.server.controller.EyvanController`; not
    instantiated on its own.
    """

    async def _dispatch_preemption(self, command: OutboxRecord) -> None:
        prepared = await self._prepare_preemption_plan(command)
        if prepared is None:
            raise StaleOutboxCommand("preemption plan is no longer admissible")
        if prepared.payload.get("execution_completed"):
            return

        phase = str(prepared.payload.get("phase") or "checkpointing")
        if phase == "checkpointing":
            checkpoint_uris: dict[str, str | None] = {}
            victim_outcomes: dict[str, _PreemptionVictimOutcome] = {}
            launched_hooks: list[_LaunchedCheckpointHook] = []
            try:
                for victim in self._preemption_victims(prepared):
                    prepared = await self._renew_preemption_claim(prepared)
                    state = await self._load_preemption_victim(prepared, victim)
                    if state is None:
                        continue
                    workload, lease, cell = state
                    checkpoint_uri: str | None = None
                    agent = self._agent_factory(cell)
                    if workload.remote_id:
                        source = await asyncio.to_thread(agent.workload, cell.id, workload.remote_id)
                        if source.checkpoint_uri:
                            checkpoint_uri = source.checkpoint_uri
                        if source.status in _PREEMPTION_TERMINAL_WORKLOAD_STATUSES:
                            victim_outcomes[lease.id] = self._preemption_outcome(source.status)
                            # A gang's members share one workload id: a sibling
                            # that checkpointed first owns the recorded URI, and
                            # this rank only proves the workload reached a
                            # terminal state on its own cell.
                            checkpoint_uris[workload.id] = checkpoint_uri or checkpoint_uris.get(workload.id)
                            continue
                    if victim.checkpoint and victim.hook and workload.remote_id:
                        mutation = MutationModel(
                            idempotency_key=f"{prepared.idempotency_key}:{lease.id}",
                            generation=victim.generation,
                            fencing_token=lease.fencing_token,
                            correlation_id=prepared.correlation_id,
                        )
                        request = CheckpointRequest(
                            mutation=mutation,
                            hook=victim.hook,
                            output_uri=victim.output_uri,
                        )
                        checkpoint = await asyncio.to_thread(
                            agent.checkpoint,
                            cell.id,
                            workload.remote_id,
                            request,
                        )
                        # Record it before waiting: a hook that times out is still
                        # running and still holding the victim's accelerators, and
                        # the abort path below is the only thing that can retire it.
                        launched_hooks.append(
                            _LaunchedCheckpointHook(
                                cell=cell,
                                remote_id=checkpoint.remote_id,
                                lease_id=lease.id,
                                generation=victim.generation,
                                fencing_token=lease.fencing_token,
                            )
                        )
                        await self._wait_for_checkpoint(
                            agent,
                            cell.id,
                            checkpoint.remote_id,
                            timeout_s=victim.grace_period_seconds or self.settings.checkpoint_grace_s,
                        )
                        checkpoint_uri = checkpoint.checkpoint_uri or victim.output_uri
                    # Keep a sibling's real URI: a rank that finished without
                    # producing one must not erase what another rank saved.
                    checkpoint_uris[workload.id] = checkpoint_uri or checkpoint_uris.get(workload.id)
                prepared = await self._advance_preemption_to_stopping(
                    prepared,
                    checkpoint_uris,
                    victim_outcomes,
                )
            except StaleOutboxCommand:
                raise
            except Exception as exc:
                # Rolling back durable state restores the victims to RUNNING. Any
                # hook still executing would then share the accelerators with the
                # trainer it was meant to checkpoint, and could still write output
                # under a plan that no longer exists.
                retire_errors = await self._retire_checkpoint_hooks(prepared, launched_hooks)
                reason = "; ".join([str(exc), *retire_errors])
                if await self._abort_preemption_checkpoint_phase(prepared, reason):
                    raise StaleOutboxCommand(f"preemption checkpoint phase aborted: {reason}") from exc
                prepared = await self._renew_preemption_claim(prepared)

        if str(prepared.payload.get("phase")) != "stopping":
            raise RuntimeError("started preemption plan has an invalid durable phase")
        stop_errors: list[str] = []
        for victim in self._preemption_victims(prepared):
            prepared = await self._renew_preemption_claim(prepared)
            outcomes = self._preemption_victim_outcomes(prepared)
            if victim.lease_id in outcomes:
                continue
            try:
                outcome = _PreemptionVictimOutcome.PREEMPTED
                state = await self._load_preemption_victim(prepared, victim)
                if state is not None:
                    workload, lease, cell = state
                    if workload.remote_id:
                        agent = self._agent_factory(cell)
                        observation = await asyncio.to_thread(agent.workload, cell.id, workload.remote_id)
                        status = observation.status
                        if status not in _PREEMPTION_TERMINAL_WORKLOAD_STATUSES:
                            request = WorkloadMutationRequest(
                                mutation=MutationModel(
                                    idempotency_key=f"{prepared.idempotency_key}:{lease.id}",
                                    generation=victim.generation,
                                    fencing_token=lease.fencing_token,
                                    correlation_id=prepared.correlation_id,
                                )
                            )
                            await asyncio.to_thread(agent.stop, cell.id, workload.remote_id, request)
                            status = await self._wait_for_workload_terminal(
                                agent,
                                cell.id,
                                workload.remote_id,
                                timeout_s=self.settings.preemption_stop_grace_s,
                            )
                        outcome = self._preemption_outcome(status)
                prepared = await self._record_preemption_victim_stopped(
                    prepared.id,
                    victim.lease_id,
                    outcome,
                )
            except StaleOutboxCommand:
                raise
            except CellAgentStaleMutationError:
                raise
            except Exception as exc:
                stop_errors.append(f"{victim.lease_id}: {exc}")
        if stop_errors:
            raise RuntimeError("preemption stop phase incomplete: " + "; ".join(stop_errors))

        checkpoint_uris = self._preemption_checkpoint_uris(prepared)
        await self._finalize_preemption_plan(prepared, checkpoint_uris)

    @staticmethod
    def _preemption_outcome(status: ExecutionWorkloadStatus) -> _PreemptionVictimOutcome:
        if status is ExecutionWorkloadStatus.SUCCEEDED:
            return _PreemptionVictimOutcome.SUCCEEDED
        if status is ExecutionWorkloadStatus.FAILED:
            return _PreemptionVictimOutcome.FAILED
        if status in {
            ExecutionWorkloadStatus.STOPPED,
            ExecutionWorkloadStatus.LOST,
            ExecutionWorkloadStatus.NOT_FOUND,
        }:
            return _PreemptionVictimOutcome.PREEMPTED
        raise RuntimeError(f"preemption outcome requires a terminal workload status, got {status.value}")

    @staticmethod
    def _preemption_victims(command: OutboxRecord) -> tuple[_PreemptionVictim, ...]:
        raw = command.payload.get("victims")
        if not isinstance(raw, list) or not raw:
            raise StaleOutboxCommand("preemption plan has no victim set")
        try:
            return tuple(_PreemptionVictim.model_validate(item) for item in raw)
        except ValidationError as exc:
            raise StaleOutboxCommand("preemption plan contains an invalid victim") from exc

    @staticmethod
    def _preemption_checkpoint_uris(command: OutboxRecord) -> dict[str, str | None]:
        raw = command.payload.get("checkpoint_uris", {})
        try:
            return _PreemptionCheckpointUris.model_validate(raw).root
        except ValidationError as exc:
            raise StaleOutboxCommand("preemption plan contains invalid checkpoint URIs") from exc

    @staticmethod
    def _preemption_victim_outcomes(command: OutboxRecord) -> dict[str, _PreemptionVictimOutcome]:
        raw = command.payload.get("victim_outcomes", {})
        try:
            return _PreemptionVictimOutcomes.model_validate(raw).root
        except ValidationError as exc:
            raise StaleOutboxCommand("preemption plan contains invalid victim outcomes") from exc

    @staticmethod
    def _preemption_stopped_victims(command: OutboxRecord) -> list[str]:
        raw = command.payload.get("stopped_victims", [])
        try:
            return _PreemptionStoppedVictims.model_validate(raw).root
        except ValidationError as exc:
            raise StaleOutboxCommand("preemption plan contains an invalid stopped-victim set") from exc

    async def _prepare_preemption_plan(self, command: OutboxRecord) -> OutboxRecord | None:
        async with self.repository.transaction() as transaction:
            current = await transaction.get_outbox(command.id)
            if current is None:
                raise StaleOutboxCommand("preemption command no longer exists")
            self._require_preemption_claim(current)
            if current.payload.get("execution_started"):
                return current

            victims = self._preemption_victims(current)
            victim_workload_ids = sorted({victim.workload_id for victim in victims})
            for workload_id in victim_workload_ids:
                await self.service._lock_settlement(transaction, workload_id)

            workload_ids = sorted({str(current.payload["target_workload_id"]), *victim_workload_ids})
            workloads = {workload_id: await transaction.get_workload(workload_id) for workload_id in workload_ids}
            lease_ids = sorted({victim.lease_id for victim in victims})
            leases = {lease_id: await transaction.get_lease(lease_id) for lease_id in lease_ids}
            target = workloads.get(str(current.payload["target_workload_id"]))

            scopes = sorted(
                {(candidate.org_id, candidate.project_id) for candidate in workloads.values() if candidate is not None}
            )
            for org_id, project_id in scopes:
                await self.service._lock_admission_scope(transaction, org_id, project_id)

            stale = (
                target is None
                or target.state is not WorkloadState.PENDING
                or target.generation != int(current.payload["target_generation"])
                or target.attempt != int(current.payload["target_attempt"])
                or target.preemption_id != current.id
                or self.service.scheduler is None
            )
            for victim in victims:
                workload = workloads.get(victim.workload_id)
                lease = leases.get(victim.lease_id)
                stale = stale or (
                    workload is None
                    or lease is None
                    or workload.generation != victim.generation
                    or victim.lease_id not in self._workload_lease_ids(workload)
                    or workload.preemption_id != current.id
                    or lease.generation != victim.generation
                    or lease.cell_id != victim.cell_id
                    or lease.revocation_reason != "preemption"
                    or workload.state not in {WorkloadState.CHECKPOINTING, WorkloadState.CANCELLING}
                    or lease.state not in {LeaseState.REVOKING, LeaseState.QUARANTINED}
                )

            if not stale and target is not None and self.service.scheduler is not None:
                cell = await transaction.get_cell(str(current.payload["cell_id"]))
                stale = cell is None or cell.state is not CellState.READY
                if not stale:
                    # A gang's siblings occupy different cells. Fence each
                    # allocation against its own generation, not the target's.
                    for lease in leases.values():
                        member_cell = None if lease is None else await transaction.get_cell(lease.cell_id)
                        if (
                            lease is None
                            or member_cell is None
                            or member_cell.state is not CellState.READY
                            or lease.cell_generation != member_cell.generation
                        ):
                            stale = True
                            break
                policy = await transaction.get_policy(target.org_id, target.project_id)
                if not stale and policy is not None:
                    try:
                        self.service._validate_workload_policy(target.spec, policy)
                    except InputError:
                        stale = True
                if not stale:
                    budget = await transaction.get_budget(target.org_id, target.project_id)
                    active_leases = [
                        candidate
                        for candidate in await transaction.list_leases()
                        if candidate.state in ACTIVE_LEASE_STATES
                    ]
                    active_workload_ids = {candidate.workload_id for candidate in active_leases}
                    validation_workloads = [
                        candidate
                        for candidate in await transaction.list_workloads()
                        if candidate.id == target.id or candidate.id in active_workload_ids
                    ]
                    refreshed = self.service.scheduler.schedule(
                        validation_workloads,
                        await transaction.list_cells(),
                        leases=active_leases,
                        budgets=() if budget is None else (budget,),
                        quotas=await transaction.list_quotas(org_id=target.org_id),
                        policies=await transaction.list_policies(),
                        grants=await transaction.list_budget_grants() if self.service.plane_id else [],
                        quota_grants=await transaction.list_quota_grants() if self.service.plane_id else [],
                        plane_id=self.service.plane_id,
                    )
                    current_plan = next(
                        (candidate for candidate in refreshed.preemptions if candidate.workload_id == target.id),
                        None,
                    )
                    expected_lease_ids = tuple(victim.lease_id for victim in victims)
                    stale = (
                        current_plan is None
                        or current_plan.cell_id != str(current.payload["cell_id"])
                        or tuple(victim.lease_id for victim in current_plan.victims) != expected_lease_ids
                    )

            if stale:
                await self._rollback_preemption_plan(transaction, current, workloads, leases)
                return None

            prepared = current.model_copy(
                update={
                    "payload": {
                        **current.payload,
                        "execution_started": True,
                        "execution_started_at": utcnow().isoformat(),
                        "phase": "checkpointing",
                        "checkpoint_uris": {},
                        "victim_outcomes": {},
                        "stopped_victims": [],
                    },
                    "updated_at": utcnow(),
                }
            )
            await transaction.put_outbox(prepared)
            return prepared

    def _require_preemption_claim(self, command: OutboxRecord) -> None:
        if (
            command.state is not OutboxState.PROCESSING
            or command.claimed_by != self.worker_id
            or command.claimed_until is None
            or command.claimed_until <= utcnow()
        ):
            raise StaleOutboxCommand("preemption command claim expired or moved to another worker")

    async def _renew_preemption_claim(self, command: OutboxRecord) -> OutboxRecord:
        """Renew the current preemption command through the shared claim fence."""

        if command.topic != "workload.preempt":
            raise StaleOutboxCommand(f"outbox command {command.id!r} is not a preemption command")
        return await self._renew_outbox_claim(command)

    async def _advance_preemption_to_stopping(
        self,
        command: OutboxRecord,
        checkpoint_uris: dict[str, str | None],
        victim_outcomes: dict[str, _PreemptionVictimOutcome],
    ) -> OutboxRecord:
        async with self.repository.transaction() as transaction:
            current = await transaction.get_outbox(command.id)
            if current is None:
                raise StaleOutboxCommand("preemption command no longer exists")
            self._require_preemption_claim(current)
            phase = str(current.payload.get("phase") or "checkpointing")
            if phase == "stopping":
                return current
            if phase != "checkpointing" or current.payload.get("execution_completed"):
                raise StaleOutboxCommand("preemption checkpoint phase is no longer active")
            advanced = current.model_copy(
                update={
                    "payload": {
                        **current.payload,
                        "phase": "stopping",
                        "checkpoint_uris": checkpoint_uris,
                        "victim_outcomes": {lease_id: outcome.value for lease_id, outcome in victim_outcomes.items()},
                        "stopped_victims": self._preemption_stopped_victims(current),
                        "stopping_started_at": utcnow().isoformat(),
                    },
                    "claimed_until": utcnow() + timedelta(seconds=max(self.settings.outbox_claim_ttl_s, 1)),
                    "updated_at": utcnow(),
                }
            )
            await transaction.put_outbox(advanced)
            return advanced

    async def _record_preemption_victim_stopped(
        self,
        command_id: str,
        lease_id: str,
        outcome: _PreemptionVictimOutcome,
    ) -> OutboxRecord:
        async with self.repository.transaction() as transaction:
            current = await transaction.get_outbox(command_id)
            if current is None:
                raise StaleOutboxCommand("preemption command no longer exists")
            self._require_preemption_claim(current)
            if str(current.payload.get("phase")) != "stopping" or current.payload.get("execution_completed"):
                raise StaleOutboxCommand("preemption stop phase is no longer active")
            stopped = list(dict.fromkeys([*self._preemption_stopped_victims(current), lease_id]))
            outcomes = {**self._preemption_victim_outcomes(current), lease_id: outcome}
            updated = current.model_copy(
                update={
                    "payload": {
                        **current.payload,
                        "stopped_victims": stopped,
                        "victim_outcomes": {
                            victim_lease_id: victim_outcome.value for victim_lease_id, victim_outcome in outcomes.items()
                        },
                    },
                    "claimed_until": utcnow() + timedelta(seconds=max(self.settings.outbox_claim_ttl_s, 1)),
                    "updated_at": utcnow(),
                }
            )
            await transaction.put_outbox(updated)
            return updated

    async def _rollback_preemption_plan(
        self,
        transaction: RepositorySession,
        command: OutboxRecord,
        workloads: dict[str, WorkloadRecord | None],
        leases: dict[str, LeaseRecord | None],
    ) -> None:
        now = utcnow()
        for victim in self._preemption_victims(command):
            workload = workloads.get(victim.workload_id)
            lease = leases.get(victim.lease_id)
            if (
                workload is None
                or lease is None
                or workload.generation != victim.generation
                or lease.id not in self._workload_lease_ids(workload)
                or workload.preemption_id != command.id
                or lease.revocation_reason != "preemption"
                or workload.state not in {WorkloadState.CHECKPOINTING, WorkloadState.CANCELLING}
                or lease.state not in {LeaseState.REVOKING, LeaseState.QUARANTINED}
            ):
                continue
            restored_workload = workload.model_copy(
                update={"state": victim.workload_state, "preemption_id": None, "updated_at": now}
            )
            restored_lease = lease.model_copy(
                update={"state": victim.lease_state, "revocation_reason": None, "updated_at": now}
            )
            await transaction.put_workload(restored_workload)
            await transaction.put_lease(restored_lease)
            if (
                victim.workload_state is WorkloadState.LEASED
                and workload.remote_id is None
                and lease.id == workload.lease_id
            ):
                await self.service._outbox(
                    transaction,
                    topic="workload.dispatch",
                    aggregate_type="workload",
                    aggregate_id=workload.id,
                    idempotency_key=f"dispatch:{lease.id}:{lease.generation}:rollback:{command.id}",
                    generation=lease.generation,
                    payload={"workload_id": workload.id, "lease_id": lease.id, "cell_id": lease.cell_id},
                )

        target = workloads.get(str(command.payload["target_workload_id"]))
        if target is not None and target.state is WorkloadState.PENDING and target.preemption_id == command.id:
            await transaction.put_workload(
                target.model_copy(
                    update={
                        "preemption_id": None,
                        "scheduling_reason": "preemption invalidated; waiting for scheduling",
                        "updated_at": now,
                    }
                )
            )

    async def _retire_checkpoint_hooks(
        self,
        command: OutboxRecord,
        hooks: tp.Sequence[_LaunchedCheckpointHook],
    ) -> list[str]:
        """Retire every hook this attempt launched; return the failures as text.

        Best effort by design. A hook that cannot be retired is worse than one
        that can, but it is strictly better to roll the victims back to RUNNING
        and surface the stray unit in ``aborted_reason`` than to leave them
        pinned in CHECKPOINTING because cleanup failed.
        """

        errors: list[str] = []
        for hook in hooks:
            request = WorkloadMutationRequest(
                mutation=MutationModel(
                    idempotency_key=f"{command.idempotency_key}:{hook.lease_id}:hook-retire",
                    generation=hook.generation,
                    fencing_token=hook.fencing_token,
                    correlation_id=command.correlation_id,
                )
            )
            try:
                await asyncio.to_thread(
                    self._agent_factory(hook.cell).retire,
                    hook.cell.id,
                    hook.remote_id,
                    request,
                )
            except Exception as exc:  # reported, never raised
                errors.append(f"checkpoint hook {hook.remote_id} was not retired: {exc}")
                logger.warning(
                    "orphaned checkpoint hook %s on cell %s after aborted preemption %s: %s",
                    hook.remote_id,
                    hook.cell.id,
                    command.id,
                    exc,
                )
        return errors

    async def _abort_preemption_checkpoint_phase(self, command: OutboxRecord, reason: str) -> bool:
        victims = self._preemption_victims(command)
        async with self.repository.transaction() as transaction:
            current = await transaction.get_outbox(command.id)
            if current is None:
                raise StaleOutboxCommand("preemption command no longer exists")
            self._require_preemption_claim(current)
            if str(current.payload.get("phase") or "checkpointing") != "checkpointing":
                return False
            for workload_id in sorted({victim.workload_id for victim in victims}):
                await self.service._lock_settlement(transaction, workload_id)
            workload_ids = sorted(
                {str(command.payload["target_workload_id"]), *(victim.workload_id for victim in victims)}
            )
            workloads = {workload_id: await transaction.get_workload(workload_id) for workload_id in workload_ids}
            leases = {victim.lease_id: await transaction.get_lease(victim.lease_id) for victim in victims}
            scopes = sorted(
                {(workload.org_id, workload.project_id) for workload in workloads.values() if workload is not None}
            )
            for org_id, project_id in scopes:
                await self.service._lock_admission_scope(transaction, org_id, project_id)
            await self._rollback_preemption_plan(transaction, current, workloads, leases)
            await transaction.put_outbox(
                current.model_copy(
                    update={
                        "payload": {
                            **current.payload,
                            "execution_completed": True,
                            "aborted_reason": reason[:1000],
                            "execution_completed_at": utcnow().isoformat(),
                        },
                        "updated_at": utcnow(),
                    }
                )
            )
            return True

    async def _load_preemption_victim(
        self,
        command: OutboxRecord,
        victim: _PreemptionVictim,
    ) -> tuple[WorkloadRecord, LeaseRecord, CellRecord] | None:
        async with self.repository.transaction() as transaction:
            workload = await transaction.get_workload(victim.workload_id)
            lease = await transaction.get_lease(victim.lease_id)
            if (
                workload is not None
                and lease is not None
                and lease.state is LeaseState.RELEASED
                and workload.generation > victim.generation
                and lease.id not in self._workload_lease_ids(workload)
            ):
                return None
            cell = await transaction.get_cell(victim.cell_id)
            if workload is None or lease is None or cell is None:
                raise RuntimeError("started preemption plan references a missing workload, lease, or cell")
            if (
                workload.generation != victim.generation
                or lease.id not in self._workload_lease_ids(workload)
                or workload.preemption_id != command.id
                or lease.generation != workload.generation
                or lease.cell_id != victim.cell_id
                or lease.revocation_reason != "preemption"
                or workload.state not in {WorkloadState.CHECKPOINTING, WorkloadState.CANCELLING}
                or lease.state not in {LeaseState.REVOKING, LeaseState.QUARANTINED}
            ):
                raise RuntimeError("started preemption plan lost ownership of a victim")
            return workload, lease, cell

    async def _finalize_preemption_plan(
        self,
        command: OutboxRecord,
        checkpoint_uris: dict[str, str | None],
    ) -> None:
        victims = self._preemption_victims(command)
        members_by_workload: dict[str, list[_PreemptionVictim]] = {}
        for victim in victims:
            members_by_workload.setdefault(victim.workload_id, []).append(victim)
        # A victim whose execution is observed SUCCEEDED or FAILED is settled
        # terminal inside the transaction below, and settlement releases its
        # lease -- which frees the cell for deletion and destroys the journal.
        # Archive the tail first, the same best-effort step _release_terminal
        # takes on every ordinary terminal path, and for the same reason: a job
        # that ran must still be recorded as finished even if its logs could not
        # be saved. Outside the transaction: the archive opens its own.
        outcomes = self._preemption_victim_outcomes(command)
        finished_outcomes = {_PreemptionVictimOutcome.SUCCEEDED, _PreemptionVictimOutcome.FAILED}
        if any(outcome in finished_outcomes for outcome in outcomes.values()):
            async with self.repository.transaction() as transaction:
                finished = [
                    await transaction.get_workload(workload_id)
                    for workload_id, members in members_by_workload.items()
                    if any(outcomes.get(member.lease_id) in finished_outcomes for member in members)
                ]
            for workload in finished:
                if workload is None:
                    continue
                try:
                    await self._archive_terminal_logs(workload)
                except Exception:
                    logger.warning("could not archive logs for %s before release", workload.id, exc_info=True)
        async with self.repository.transaction() as transaction:
            current_command = await transaction.get_outbox(command.id)
            if current_command is None:
                raise StaleOutboxCommand("preemption command no longer exists")
            self._require_preemption_claim(current_command)
            if current_command.payload.get("execution_completed"):
                return
            outcomes = self._preemption_victim_outcomes(current_command)
            expected_lease_ids = {victim.lease_id for victim in victims}
            if str(current_command.payload.get("phase")) != "stopping" or not expected_lease_ids <= outcomes.keys():
                raise RuntimeError("preemption plan cannot finalize before every victim is terminal")
            for workload_id in sorted({victim.workload_id for victim in victims}):
                await self.service._lock_settlement(transaction, workload_id)
            workloads = {victim.workload_id: await transaction.get_workload(victim.workload_id) for victim in victims}
            leases = {victim.lease_id: await transaction.get_lease(victim.lease_id) for victim in victims}
            scopes = sorted(
                {(workload.org_id, workload.project_id) for workload in workloads.values() if workload is not None}
            )
            for org_id, project_id in scopes:
                await self.service._lock_admission_scope(transaction, org_id, project_id)
                scope = storage_scope(org_id, project_id)
                await transaction.get_idempotency(scope, "budget.settle", scope)

            for victim in victims:
                workload = workloads.get(victim.workload_id)
                lease = leases.get(victim.lease_id)
                if (
                    workload is not None
                    and lease is not None
                    and lease.state is LeaseState.RELEASED
                    and workload.generation > victim.generation
                    and lease.id not in self._workload_lease_ids(workload)
                ):
                    continue
                if (
                    workload is None
                    or lease is None
                    or workload.generation != victim.generation
                    or lease.id not in self._workload_lease_ids(workload)
                    or workload.preemption_id != command.id
                    or lease.revocation_reason != "preemption"
                    or workload.state not in {WorkloadState.CHECKPOINTING, WorkloadState.CANCELLING}
                    or lease.state not in {LeaseState.REVOKING, LeaseState.QUARANTINED}
                ):
                    raise RuntimeError("started preemption plan cannot settle a victim it no longer owns")
                members = members_by_workload[workload.id]
                member_outcomes = {outcomes[member.lease_id] for member in members}
                # A gang succeeds only when every member succeeded. One failed
                # member fails the gang; otherwise lost members require a retry.
                if _PreemptionVictimOutcome.FAILED in member_outcomes:
                    outcome = _PreemptionVictimOutcome.FAILED
                elif member_outcomes == {_PreemptionVictimOutcome.SUCCEEDED}:
                    outcome = _PreemptionVictimOutcome.SUCCEEDED
                else:
                    outcome = _PreemptionVictimOutcome.PREEMPTED
                settled, _, _ = await self.service._settle_lease(
                    transaction,
                    self._principal,
                    workload,
                    lease,
                    reason=outcome.value,
                )
                # Settle each reservation, but change the workload/generation
                # only after its last member. Replaying remains one transaction.
                if victim.lease_id != members[-1].lease_id:
                    continue
                if outcome in {_PreemptionVictimOutcome.SUCCEEDED, _PreemptionVictimOutcome.FAILED}:
                    target_state = (
                        WorkloadState.SUCCEEDED
                        if outcome is _PreemptionVictimOutcome.SUCCEEDED
                        else WorkloadState.FAILED
                    )
                    terminal = await self.service._transition_workload(
                        transaction,
                        self._principal,
                        workload,
                        target_state,
                        f"workload.{outcome.value}_during_preemption",
                    )
                    terminal = terminal.model_copy(update={"preemption_id": None, "updated_at": utcnow()})
                    await transaction.put_workload(terminal)
                    continue
                now = utcnow()
                checkpoint_uri = checkpoint_uris.get(workload.id) or workload.checkpoint_uri
                next_generation = workload.generation + 1
                resumed = workload.model_copy(
                    update={
                        "state": WorkloadState.PENDING,
                        "attempt": workload.attempt + 1,
                        "generation": next_generation,
                        "spec": self.service.result_spec_for_generation(
                            workload.spec,
                            workload_id=workload.id,
                            generation=next_generation,
                        ),
                        "lease_id": None,
                        # Same reset the sibling resume path performs: the
                        # released generation's lease group belongs to that
                        # generation. Keeping it with no leader lease would
                        # violate the leader-first shape _current_leases
                        # enforces and strand the requeued workload.
                        "lease_ids": (),
                        "dispatch_reservation": None,
                        "remote_id": None,
                        "service_endpoint": None,
                        "checkpoint_uri": checkpoint_uri,
                        "scheduling_reason": "preempted; waiting to resume",
                        "preemption_id": None,
                        "updated_at": now,
                    }
                )
                # ``_put_workload_keeping_references``, not ``put_workload``: this
                # path bumps the generation, and ``result_spec_for_generation``
                # mints a *new* managed result URI for the resumed attempt. Its
                # helper documents itself as used by every path that bumps the
                # generation, and this one did not -- so the resumed attempt owned
                # an artifact address nothing referenced, and a collector could
                # reclaim the object the attempt was about to write.
                await self.service._put_workload_keeping_references(transaction, resumed)
                await self.service._audit(
                    transaction,
                    self._principal,
                    resumed,
                    "workload.preempted",
                    previous_state=workload.state.value,
                    new_state=resumed.state.value,
                    payload={"checkpoint_uri": checkpoint_uri, "released_lease_id": settled.id},
                )

            await transaction.put_outbox(
                current_command.model_copy(
                    update={
                        "payload": {
                            **current_command.payload,
                            "execution_completed": True,
                            "execution_completed_at": utcnow().isoformat(),
                        },
                        "updated_at": utcnow(),
                    }
                )
            )
            target = await transaction.get_workload(str(command.payload["target_workload_id"]))
            if target is not None and target.preemption_id == command.id:
                await transaction.put_workload(target.model_copy(update={"preemption_id": None, "updated_at": utcnow()}))

    async def _wait_for_checkpoint(
        self,
        agent: CellAgentClient,
        cell_id: str,
        checkpoint_job_id: str,
        *,
        timeout_s: int,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + max(timeout_s, 1)
        while True:
            observation = await asyncio.to_thread(agent.workload, cell_id, checkpoint_job_id)
            status = observation.status
            if status is ExecutionWorkloadStatus.SUCCEEDED:
                return
            if status in {ExecutionWorkloadStatus.FAILED, ExecutionWorkloadStatus.STOPPED}:
                raise PreemptionCheckpointError(f"checkpoint hook {checkpoint_job_id!r} ended in {status.value}")
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"checkpoint hook {checkpoint_job_id!r} exceeded {timeout_s}s grace")
            await asyncio.sleep(1)

    async def _wait_for_workload_terminal(
        self,
        agent: CellAgentClient,
        cell_id: str,
        remote_id: str,
        *,
        timeout_s: int,
    ) -> ExecutionWorkloadStatus:
        deadline = asyncio.get_running_loop().time() + max(timeout_s, 1)
        while True:
            observation = await asyncio.to_thread(agent.workload, cell_id, remote_id)
            if observation.status in _PREEMPTION_TERMINAL_WORKLOAD_STATUSES:
                return observation.status
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"remote workload {remote_id!r} did not stop within {timeout_s}s grace")
            await asyncio.sleep(1)
