# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The administrative reclaim of one running workload: checkpoint, stop, requeue."""

from __future__ import annotations

import asyncio
import typing as tp
from datetime import timedelta

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.records import (
    CellRecord,
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
    StaleOutboxCommand,
    _PreemptionVictimOutcome,
    logger,
)
from eyvan.server.persistence.base import RepositorySession
from eyvan.server.service.state import ensure_workload_transition
from eyvan.worker.agent.client import CellAgentClient, CellAgentStaleMutationError
from eyvan.worker.agent.models import (
    CheckpointRequest,
    GangReclaimNoticeRequest,
    GangWorkloadMutationRequest,
    MutationModel,
    ReclaimNoticeRequest,
    WorkloadMutationRequest,
)

if tp.TYPE_CHECKING:
    from eyvan.config import EyvanSettings
    from eyvan.records import Principal
    from eyvan.server.persistence.base import Repository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.worker.agent.models import GangWorkloadMember

#: Workload states a reclaim command owns: CHECKPOINTING while its hook still has
#: to run, CANCELLING once the checkpoint is recorded (or when there is none).
_RECLAIM_WORKLOAD_STATES = frozenset({WorkloadState.CHECKPOINTING, WorkloadState.CANCELLING})
_RECLAIM_LEASE_STATES = frozenset({LeaseState.REVOKING, LeaseState.QUARANTINED})

_Members = tuple[tuple[LeaseRecord, CellRecord], ...]


class _ControllerReclaim:
    """Deliver ``workload.reclaim``: checkpoint, stop, then requeue one workload.

    A concern of :class:`~eyvan.server.controller.EyvanController`; not
    instantiated on its own.

    The workload's own state is the durable phase marker. CHECKPOINTING means
    the hook has not been recorded yet; the transition to CANCELLING commits the
    checkpoint URI, so a retried command never re-runs a completed checkpoint.
    Every write is fenced on the command's generation, its lease set and the
    ``revocation_reason="preemption"`` marker that ``reclaim_workload`` placed on
    the leases; a user cancel clears that marker and thereby wins.

    Checkpoint-phase failure policy: *abort*. Any failure before the remote
    process is stopped -- hook FAILED/STOPPED, grace timeout, an unreachable
    agent -- retires the hook it launched (best effort), returns the workload to
    RUNNING and its leases to their prior state, and records
    ``workload.reclaim_aborted`` with the reason. Stopping anyway would discard
    the progress the reclaim exists to keep, and the process is still healthy;
    the operator can re-issue the reclaim. Once the stop phase begins nothing is
    rolled back: a partially stopped process cannot safely be called RUNNING, so
    stop failures and timeouts retry the command instead.
    """

    if tp.TYPE_CHECKING:
        repository: Repository
        service: ControlPlaneService
        settings: EyvanSettings
        worker_id: str
        _principal: Principal
        _agent_factory: tp.Callable[[CellRecord], CellAgentClient]

        @staticmethod
        def _workload_lease_ids(workload: WorkloadRecord) -> tuple[str, ...]: ...

        @staticmethod
        def _ordered_gang_member_pairs(members: tp.Sequence[tuple[LeaseRecord, CellRecord]]) -> _Members: ...

        @classmethod
        def _gang_members(
            cls,
            members: tp.Sequence[tuple[LeaseRecord, CellRecord]],
            *,
            idempotency_prefix: str,
            correlation_id: str | None = None,
        ) -> tuple[GangWorkloadMember, ...]: ...

        @staticmethod
        def _preemption_outcome(status: ExecutionWorkloadStatus) -> _PreemptionVictimOutcome: ...

        async def _wait_for_checkpoint(
            self, agent: CellAgentClient, cell_id: str, checkpoint_job_id: str, *, timeout_s: int
        ) -> None: ...

        async def _release_terminal(
            self,
            workload: WorkloadRecord,
            lease: LeaseRecord,
            target: WorkloadState,
            action: str,
            *,
            failure_reason: str | None = None,
            expected_dispatch_command: OutboxRecord | None = None,
        ) -> None: ...

        async def _recover_stale_workload_command(
            self, command: OutboxRecord, *, expected_state: WorkloadState, diagnostic: str
        ) -> None: ...

        async def _discard_outbox(self, command: OutboxRecord, *, provider_diagnostic: str | None = None) -> None: ...

    async def _dispatch_reclaim(self, command: OutboxRecord) -> None:
        workload, members = await self._load_reclaim_state(command)
        status, message, observed_uri = await self._reclaim_remote_status(workload, members)
        if status in _PREEMPTION_TERMINAL_WORKLOAD_STATUSES:
            # Finished on its own before notice/hook ran: settle that outcome.
            await self._settle_reclaim(command, status, message, checkpoint_uri=observed_uri)
            return

        grace_s = self._reclaim_grace_seconds(command, workload)
        notice_delivered = await self._deliver_reclaim_notice(command, workload, members, grace_s=grace_s)

        if workload.state is WorkloadState.CHECKPOINTING:
            checkpoint_uri = await self._run_reclaim_checkpoint(command, workload, members)
            workload = await self._record_reclaim_checkpoint(command, checkpoint_uri)

        status, message, observed_uri = await self._reclaim_remote_status(workload, members)
        if status not in _PREEMPTION_TERMINAL_WORKLOAD_STATUSES:
            if notice_delivered:
                status, message, observed_uri, exited = await self._wait_for_graceful_reclaim_exit(
                    workload, members, timeout_s=grace_s
                )
            else:
                exited = False
            if not exited and status not in _PREEMPTION_TERMINAL_WORKLOAD_STATUSES:
                await self._stop_reclaimed_workload(command, workload, members)
                status, message, observed_uri = await self._wait_for_reclaim_terminal(workload, members)
        await self._settle_reclaim(command, status, message, checkpoint_uri=observed_uri)

    # ------------------------------------------------------------- ownership

    @staticmethod
    def _reclaim_lease_ids(command: OutboxRecord) -> tuple[str, ...]:
        raw = command.payload.get("lease_ids")
        if not isinstance(raw, list) or not raw or not all(isinstance(item, str) and item for item in raw):
            raise StaleOutboxCommand("reclaim command has an invalid lease set")
        return tuple(str(item) for item in raw)

    def _require_reclaim_claim(self, command: OutboxRecord) -> None:
        if (
            command.state is not OutboxState.PROCESSING
            or command.claimed_by != self.worker_id
            or command.claimed_until is None
            or command.claimed_until <= utcnow()
        ):
            raise StaleOutboxCommand("reclaim command claim expired or moved to another worker")

    async def _owned_reclaim_records(
        self, tx: RepositorySession, command: OutboxRecord
    ) -> tuple[WorkloadRecord, tuple[LeaseRecord, ...]] | None:
        """Return the workload and leases only while this command still owns them."""

        lease_ids = self._reclaim_lease_ids(command)
        workload = await tx.get_workload(command.aggregate_id)
        leases = [await tx.get_lease(lease_id) for lease_id in lease_ids]
        if workload is None or any(lease is None for lease in leases):
            return None
        concrete = tuple(lease for lease in leases if lease is not None)
        if (
            workload.generation != command.generation
            or workload.state not in _RECLAIM_WORKLOAD_STATES
            or workload.preemption_id is not None
            or self._workload_lease_ids(workload) != lease_ids
        ):
            return None
        for lease in concrete:
            if (
                lease.workload_id != workload.id
                or lease.generation != command.generation
                or lease.revocation_reason != "preemption"
                or lease.state not in _RECLAIM_LEASE_STATES
            ):
                return None
        return workload, concrete

    async def _load_reclaim_state(self, command: OutboxRecord) -> tuple[WorkloadRecord, _Members]:
        async with self.repository.transaction() as tx:
            owned = await self._owned_reclaim_records(tx, command)
            if owned is None:
                raise StaleOutboxCommand("reclaim command no longer owns the workload generation and its leases")
            workload, leases = owned
            cells = [await tx.get_cell(lease.cell_id) for lease in leases]
        concrete_cells = tuple(cell for cell in cells if cell is not None)
        if len(concrete_cells) != len(leases):
            # Not stale: the orphan sweep owns a workload whose cell vanished.
            raise RuntimeError(f"reclaim of {workload.id!r} references a missing execution cell")
        if workload.remote_id is None:
            raise RuntimeError(f"reclaim of {workload.id!r} has no remote execution to stop")
        return workload, tuple(zip(leases, concrete_cells, strict=True))

    # ------------------------------------------------------------ remote I/O

    def _reclaim_grace_seconds(self, command: OutboxRecord, workload: WorkloadRecord) -> int:
        for key in ("reclaim_grace_seconds", "grace_period_seconds"):
            raw = command.payload.get(key)
            if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
                return raw
        if workload.spec.checkpoint_grace_seconds > 0:
            return workload.spec.checkpoint_grace_seconds
        default_grace = self.settings.default_reclaim_grace_s
        return default_grace if default_grace > 0 else 600

    async def _deliver_reclaim_notice(
        self,
        command: OutboxRecord,
        workload: WorkloadRecord,
        members: _Members,
        *,
        grace_s: int,
    ) -> bool:
        """Deliver an atomic JSON reclaim notice and graceful signal to single or gang members."""

        remote_id = str(workload.remote_id)
        now = utcnow()
        deadline_at_dt = workload.reclaim_deadline_at or (now + timedelta(seconds=grace_s))
        raw_deadline = command.payload.get("reclaim_deadline_at")
        deadline_at = raw_deadline if isinstance(raw_deadline, str) and raw_deadline else deadline_at_dt.isoformat()
        raw_epoch = command.payload.get("reclaim_deadline_epoch_s")
        deadline_epoch_s = (
            float(raw_epoch)
            if isinstance(raw_epoch, (int, float)) and not isinstance(raw_epoch, bool)
            else deadline_at_dt.timestamp()
        )
        raw_policy = command.payload.get("reclaim_policy") or workload.spec.reclaim_policy
        policy: tp.Literal["requeue", "stop"] = "stop" if raw_policy == "stop" else "requeue"
        raw_requested_by = command.payload.get("requested_by") or workload.reclaim_requested_by
        requested_by = raw_requested_by if isinstance(raw_requested_by, str) and raw_requested_by else None
        reason = str(command.payload.get("reason") or "reclaim requested")[:500]
        checkpoint_uri = workload.spec.checkpoint_output_uri or workload.checkpoint_uri

        if len(members) > 1:
            ordered = self._ordered_gang_member_pairs(members)
            coordinator_lease, coordinator_cell = ordered[0]
            agent = self._agent_factory(coordinator_cell)
            notifier = getattr(agent, "reclaim_notice_gang", None)
            if not callable(notifier):
                return False
            gang_request = GangReclaimNoticeRequest(
                members=self._gang_members(
                    ordered,
                    idempotency_prefix=f"{command.idempotency_key}:notice",
                    correlation_id=command.correlation_id,
                ),
                reason=reason,
                requested_by=requested_by,
                deadline_at=deadline_at,
                deadline_epoch_s=deadline_epoch_s,
                grace_seconds=grace_s,
                policy=policy,
                checkpoint_uri=checkpoint_uri,
                workload_id=workload.id,
            )
            observation = await asyncio.to_thread(
                notifier,
                coordinator_lease.cell_id,
                workload.id,
                remote_id,
                gang_request,
            )
            return observation is not None

        lease, cell = members[0]
        agent = self._agent_factory(cell)
        notifier = getattr(agent, "reclaim_notice", None)
        if not callable(notifier):
            return False
        request = ReclaimNoticeRequest(
            mutation=MutationModel(
                idempotency_key=f"{command.idempotency_key}:{lease.id}:notice",
                generation=lease.generation,
                fencing_token=lease.fencing_token,
                correlation_id=command.correlation_id,
            ),
            reason=reason,
            requested_by=requested_by,
            deadline_at=deadline_at,
            deadline_epoch_s=deadline_epoch_s,
            grace_seconds=grace_s,
            policy=policy,
            checkpoint_uri=checkpoint_uri,
            workload_id=workload.id,
        )
        observation = await asyncio.to_thread(notifier, cell.id, remote_id, request)
        return observation is not None

    async def _reclaim_remote_status(
        self, workload: WorkloadRecord, members: _Members
    ) -> tuple[ExecutionWorkloadStatus, str | None, str | None]:
        remote_id = str(workload.remote_id)
        if len(members) > 1:
            coordinator_lease, coordinator_cell = self._ordered_gang_member_pairs(members)[0]
            observation = await asyncio.to_thread(
                self._agent_factory(coordinator_cell).gang_workload,
                coordinator_lease.cell_id,
                workload.id,
                remote_id,
            )
        else:
            _lease, cell = members[0]
            observation = await asyncio.to_thread(self._agent_factory(cell).workload, cell.id, remote_id)
        return observation.status, observation.message, observation.checkpoint_uri

    async def _wait_for_graceful_reclaim_exit(
        self,
        workload: WorkloadRecord,
        members: _Members,
        *,
        timeout_s: int,
    ) -> tuple[ExecutionWorkloadStatus, str | None, str | None, bool]:
        """Poll for a graceful checkpoint-and-exit until the reclaim grace deadline expires."""

        loop = asyncio.get_running_loop()
        poll_s = 0.05 if timeout_s <= 5 else 1.0
        deadline = loop.time() + max(timeout_s, 1)
        while True:
            status, message, checkpoint_uri = await self._reclaim_remote_status(workload, members)
            if status in _PREEMPTION_TERMINAL_WORKLOAD_STATUSES:
                return status, message, checkpoint_uri, True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return status, message, checkpoint_uri, False
            await asyncio.sleep(min(poll_s, remaining))

    async def _wait_for_reclaim_terminal(
        self, workload: WorkloadRecord, members: _Members
    ) -> tuple[ExecutionWorkloadStatus, str | None, str | None]:
        timeout_s = self.settings.preemption_stop_grace_s
        deadline = asyncio.get_running_loop().time() + max(timeout_s, 1)
        while True:
            status, message, checkpoint_uri = await self._reclaim_remote_status(workload, members)
            if status in _PREEMPTION_TERMINAL_WORKLOAD_STATUSES:
                return status, message, checkpoint_uri
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"reclaimed workload {workload.id!r} did not stop within {timeout_s}s grace")
            await asyncio.sleep(0.1 if timeout_s <= 5 else 1.0)

    async def _run_reclaim_checkpoint(self, command: OutboxRecord, workload: WorkloadRecord, members: _Members) -> str:
        """Run the hook and wait for it; on any failure abort the reclaim and raise."""

        hook = command.payload.get("hook")
        output_uri = command.payload.get("output_uri")
        if not members or not isinstance(hook, str) or not hook:
            raise StaleOutboxCommand("reclaim checkpoint phase requires at least one cell and a hook")
        if output_uri is not None and not isinstance(output_uri, str):
            raise StaleOutboxCommand("reclaim checkpoint output URI must be a string when set")
        grace = command.payload.get("grace_period_seconds")
        timeout_s = grace if isinstance(grace, int) and not isinstance(grace, bool) and grace > 0 else 0
        timeout_s = timeout_s or self.settings.checkpoint_grace_s
        remote_id = str(workload.remote_id)
        if len(members) > 1:
            ordered = self._ordered_gang_member_pairs(members)
            coordinator_lease, coordinator_cell = ordered[0]
            agent = self._agent_factory(coordinator_cell)
            checkpoint_gang = getattr(agent, "checkpoint_gang", None)
            if not callable(checkpoint_gang):
                raise StaleOutboxCommand("reclaim gang checkpoint hook is not supported by cell agent")
            try:
                gang_members = self._gang_members(
                    ordered,
                    idempotency_prefix=f"{command.idempotency_key}:checkpoint",
                    correlation_id=command.correlation_id,
                )
                observation = await asyncio.to_thread(
                    checkpoint_gang,
                    coordinator_lease.cell_id,
                    workload.id,
                    remote_id,
                    gang_members,
                    hook=hook,
                    output_uri=output_uri,
                )
                await self._wait_for_checkpoint(
                    agent, coordinator_lease.cell_id, observation.remote_id, timeout_s=timeout_s
                )
            except (StaleOutboxCommand, CellAgentStaleMutationError):
                raise
            except Exception as exc:
                reason = f"checkpoint failed: {exc}"
                if await self._abort_reclaim(command, reason):
                    raise StaleOutboxCommand(f"reclaim aborted: {reason}") from exc
                raise StaleOutboxCommand("reclaim lost ownership during its checkpoint phase") from exc
            return observation.checkpoint_uri or output_uri or ""

        lease, cell = members[0]
        agent = self._agent_factory(cell)
        hook_id: str | None = None
        try:
            request = CheckpointRequest(
                mutation=MutationModel(
                    idempotency_key=f"{command.idempotency_key}:{lease.id}:checkpoint",
                    generation=lease.generation,
                    fencing_token=lease.fencing_token,
                    correlation_id=command.correlation_id,
                ),
                hook=hook,
                output_uri=output_uri,
            )
            observation = await asyncio.to_thread(agent.checkpoint, cell.id, remote_id, request)
            # Recorded before waiting: a hook that outlives its grace still holds
            # the accelerators, and only the abort below can retire it.
            hook_id = observation.remote_id
            await self._wait_for_checkpoint(agent, cell.id, observation.remote_id, timeout_s=timeout_s)
        except (StaleOutboxCommand, CellAgentStaleMutationError):
            raise
        except Exception as exc:
            errors: list[str] = []
            if hook_id is not None:
                retire = WorkloadMutationRequest(
                    mutation=MutationModel(
                        idempotency_key=f"{command.idempotency_key}:{lease.id}:hook-retire",
                        generation=lease.generation,
                        fencing_token=lease.fencing_token,
                        correlation_id=command.correlation_id,
                    )
                )
                try:
                    await asyncio.to_thread(agent.retire, cell.id, hook_id, retire)
                except Exception as retire_exc:  # reported, never raised
                    errors.append(f"checkpoint hook {hook_id} was not retired: {retire_exc}")
                    logger.warning("orphaned checkpoint hook %s after aborted reclaim %s", hook_id, command.id)
            reason = "; ".join([f"checkpoint failed: {exc}", *errors])
            if await self._abort_reclaim(command, reason):
                raise StaleOutboxCommand(f"reclaim aborted: {reason}") from exc
            raise StaleOutboxCommand("reclaim lost ownership during its checkpoint phase") from exc
        return observation.checkpoint_uri or output_uri or ""

    async def _stop_reclaimed_workload(self, command: OutboxRecord, workload: WorkloadRecord, members: _Members) -> None:
        remote_id = str(workload.remote_id)
        if len(members) > 1:
            ordered = self._ordered_gang_member_pairs(members)
            gang_request = GangWorkloadMutationRequest(
                members=self._gang_members(
                    ordered,
                    idempotency_prefix=f"{command.idempotency_key}:stop",
                    correlation_id=command.correlation_id,
                )
            )
            coordinator_lease, coordinator_cell = ordered[0]
            await asyncio.to_thread(
                self._agent_factory(coordinator_cell).stop_gang,
                coordinator_lease.cell_id,
                workload.id,
                remote_id,
                gang_request,
            )
            return
        lease, cell = members[0]
        request = WorkloadMutationRequest(
            mutation=MutationModel(
                idempotency_key=f"{command.idempotency_key}:{lease.id}:stop",
                generation=lease.generation,
                fencing_token=lease.fencing_token,
                correlation_id=command.correlation_id,
            )
        )
        await asyncio.to_thread(self._agent_factory(cell).stop, cell.id, remote_id, request)

    # ------------------------------------------------------- durable phases

    async def _record_reclaim_checkpoint(self, command: OutboxRecord, checkpoint_uri: str) -> WorkloadRecord:
        """Commit the checkpoint and move to the stop phase (CHECKPOINTING -> CANCELLING)."""

        async with self.repository.transaction() as tx:
            current = await tx.get_outbox(command.id)
            if current is None:
                raise StaleOutboxCommand("reclaim command no longer exists")
            self._require_reclaim_claim(current)
            await self.service._lock_settlement(tx, command.aggregate_id)
            owned = await self._owned_reclaim_records(tx, command)
            if owned is None or owned[0].state is not WorkloadState.CHECKPOINTING:
                raise StaleOutboxCommand("reclaim lost ownership before recording its checkpoint")
            workload, _leases = owned
            ensure_workload_transition(workload.state, WorkloadState.CANCELLING)
            recorded = checkpoint_uri or workload.checkpoint_uri
            updated = workload.model_copy(
                update={
                    "state": WorkloadState.CANCELLING,
                    "checkpoint_uri": recorded,
                    "scheduling_reason": "reclaim: checkpoint saved; stopping",
                    "updated_at": utcnow(),
                }
            )
            await tx.put_workload(updated)
            await self.service._audit(
                tx,
                self._principal,
                updated,
                "workload.reclaim_checkpointed",
                previous_state=workload.state.value,
                new_state=updated.state.value,
                payload={"checkpoint_uri": recorded, "command_id": command.id},
            )
            return updated

    async def _abort_reclaim(self, command: OutboxRecord, reason: str) -> bool:
        """Return a reclaim still in its checkpoint phase to RUNNING; False if not owned."""

        async with self.repository.transaction() as tx:
            current = await tx.get_outbox(command.id)
            if current is None:
                raise StaleOutboxCommand("reclaim command no longer exists")
            self._require_reclaim_claim(current)
            await self.service._lock_settlement(tx, command.aggregate_id)
            owned = await self._owned_reclaim_records(tx, command)
            if owned is None or owned[0].state is not WorkloadState.CHECKPOINTING:
                return False
            workload, leases = owned
            await self.service._lock_admission_scope(tx, workload.org_id, workload.project_id)
            ensure_workload_transition(workload.state, WorkloadState.RUNNING)
            now = utcnow()
            restored = workload.model_copy(
                update={
                    "state": WorkloadState.RUNNING,
                    "scheduling_reason": f"reclaim aborted: {reason}"[:1000],
                    "reclaim_deadline_at": None,
                    "reclaim_requested_by": None,
                    "updated_at": now,
                }
            )
            await tx.put_workload(restored)
            prior_states = command.payload.get("lease_states")
            for lease in leases:
                prior = prior_states.get(lease.id) if isinstance(prior_states, dict) else None
                # A REVOKING lease returns to what it was; one the expiry sweep
                # quarantined meanwhile stays quarantined until observation
                # renews it, exactly as any other expired lease would.
                state = lease.state
                if lease.state is LeaseState.REVOKING:
                    state = LeaseState.QUARANTINED if prior == LeaseState.QUARANTINED.value else LeaseState.ACTIVE
                await tx.put_lease(
                    lease.model_copy(update={"state": state, "revocation_reason": None, "updated_at": now})
                )
            await self.service._audit(
                tx,
                self._principal,
                restored,
                "workload.reclaim_aborted",
                previous_state=workload.state.value,
                new_state=restored.state.value,
                payload={"reason": reason[:1000], "command_id": command.id},
            )
            await tx.put_outbox(
                current.model_copy(
                    update={
                        "payload": {**current.payload, "aborted_reason": reason[:1000]},
                        "updated_at": now,
                    }
                )
            )
            return True

    async def _clear_reclaim_markers(
        self,
        workload_id: str,
        *,
        expected_generation: int | None = None,
        scheduling_reason: str | None = None,
    ) -> None:
        async with self.repository.transaction() as tx:
            current = await tx.get_workload(workload_id)
            if current is None:
                return
            if expected_generation is not None and current.generation != expected_generation:
                return
            updates: dict[str, tp.Any] = {
                "reclaim_deadline_at": None,
                "reclaim_requested_by": None,
                "updated_at": utcnow(),
            }
            if scheduling_reason is not None:
                updates["scheduling_reason"] = scheduling_reason
            await tx.put_workload(current.model_copy(update=updates))

    async def _settle_reclaim(
        self,
        command: OutboxRecord,
        status: ExecutionWorkloadStatus,
        message: str | None,
        *,
        checkpoint_uri: str | None = None,
    ) -> None:
        """Settle the stopped attempt: requeue, stop cleanly, or record the outcome it reached itself."""

        outcome = self._preemption_outcome(status)
        workload, members = await self._load_reclaim_state(command)
        lease = members[0][0]
        effective_checkpoint_uri = checkpoint_uri or workload.checkpoint_uri
        if outcome is _PreemptionVictimOutcome.SUCCEEDED:
            await self._release_terminal(workload, lease, WorkloadState.SUCCEEDED, "workload.succeeded_during_reclaim")
            await self._clear_reclaim_markers(workload.id, expected_generation=workload.generation)
            return
        if outcome is _PreemptionVictimOutcome.FAILED:
            await self._release_terminal(
                workload,
                lease,
                WorkloadState.FAILED,
                "workload.failed_during_reclaim",
                failure_reason=message or "execution backend reported failure during reclaim",
            )
            await self._clear_reclaim_markers(workload.id, expected_generation=workload.generation)
            return
        actor = str(command.payload.get("actor") or "an administrator")
        reason = str(command.payload.get("reason") or "no reason given")
        policy = str(command.payload.get("reclaim_policy") or workload.spec.reclaim_policy or "requeue")
        if policy == "stop":
            await self.service.settle_terminal_workload(
                self._principal,
                workload.id,
                WorkloadState.CANCELLED,
                action="workload.reclaimed_stopped",
                checkpoint_uri=effective_checkpoint_uri,
                expected_generation=command.generation,
                expected_lease_id=lease.id,
                expected_state=workload.state,
            )
            await self._clear_reclaim_markers(
                workload.id,
                expected_generation=command.generation,
                scheduling_reason=f"reclaimed by {actor}: {reason}; stopped with checkpoint preserved",
            )
            return
        resumed, _lease, _budget = await self.service.settle_preempted_workload(
            self._principal,
            workload.id,
            checkpoint_uri=effective_checkpoint_uri,
            scheduling_reason=f"reclaimed by {actor}: {reason}; waiting to resume",
            action="workload.reclaimed",
            expected_generation=command.generation,
            expected_lease_id=lease.id,
            expected_state=workload.state,
            require_preemption_owner=True,
            provider_preempted=True,
        )
        await self._clear_reclaim_markers(resumed.id, expected_generation=resumed.generation)

    async def _recover_stale_reclaim_command(self, command: OutboxRecord, *, diagnostic: str) -> None:
        """Handle an agent fence conflict the way cancel/checkpoint commands do.

        A conflict means the agent holds a newer fence than this attempt, so its
        remote ownership is uncertain; the shared recovery settles it as an
        agent-fence failure. Only while this reclaim still owns the workload --
        a cancel that took over owns the outcome instead.
        """

        async with self.repository.transaction() as tx:
            owned = await self._owned_reclaim_records(tx, command)
        if owned is None:
            await self._discard_outbox(command, provider_diagnostic=diagnostic)
            return
        await self._recover_stale_workload_command(command, expected_state=owned[0].state, diagnostic=diagnostic)
