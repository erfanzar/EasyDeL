# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Committing what the pure scheduler decided."""

from __future__ import annotations

import secrets
import uuid
from datetime import timedelta

from eyvan._internal.correlation import canonical_correlation_id
from eyvan._internal.identifiers import stable_id
from eyvan._internal.json_types import JsonObject, model_json
from eyvan.capacity.acquisition_backoff import is_acquisition_backoff_reason
from eyvan.domain import (
    PlacementDecision,
    PreemptionActionKind,
    PreemptionPlan,
    SchedulingResult,
)
from eyvan.records import (
    AuditEvent,
    AutoscaleBlockReason,
    CellState,
    LeaseRecord,
    LeaseState,
    OutboxRecord,
    OutboxState,
    ResourceGang,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller._support import (
    _ACQUISITION_EXHAUSTED_SCHEDULING_REASON,
    _SHARED_TPU_QUOTA_SCHEDULING_REASON,
    _api_resources,
    _PreemptionVictim,
)
from eyvan.server.persistence.base import RepositorySession
from eyvan.server.service.control_plane import storage_scope
from eyvan.server.service.errors import InputError
from eyvan.server.service.state import ACTIVE_LEASE_STATES


class _ControllerScheduling:
    """Committing what the pure scheduler decided.

    A concern of :class:`~eyvan.server.controller.EyvanController`; not
    instantiated on its own.
    """

    async def _commit_placement(self, decision: PlacementDecision) -> None:
        if len(decision.all_leases) > 1:
            await self._commit_group_placement(decision)
            return
        async with self.repository.transaction() as transaction:
            workload = await transaction.get_workload(decision.workload_id)
            cell = await transaction.get_cell(decision.lease.cell_id)
        if (
            workload is None
            or workload.state is not WorkloadState.PENDING
            or workload.preemption_id is not None
            or cell is None
            or cell.state is not CellState.READY
            or cell.autoscale_request_workload_id not in {None, workload.id}
        ):
            return
        lease = decision.lease
        record = LeaseRecord(
            id=lease.lease_id,
            workload_id=workload.id,
            cell_id=lease.cell_id,
            cell_generation=cell.generation,
            org_id=workload.org_id,
            project_id=workload.project_id,
            resources=ResourceGang(
                replicas=workload.spec.gang.replicas,
                per_replica=workload.spec.gang.per_replica,
                required_topology=workload.spec.gang.required_topology,
            ),
            hourly_cost=float(lease.hourly_cost),
            reserved_cost=float(lease.reserved_cost),
            resource_hours_reserved=_api_resources(lease.resource_hours_reserved),
            generation=workload.generation,
            fencing_token=secrets.token_urlsafe(32),
            tpu_chips=lease.tpu_chips or None,
            expires_at=utcnow() + timedelta(seconds=self.settings.lease_ttl_s),
        )
        await self.service.create_lease(
            self._principal,
            record,
            idempotency_key=f"schedule:{record.id}:{record.generation}",
        )

    async def _commit_group_placement(self, decision: PlacementDecision) -> None:
        """Persist every member of a spread-cell pod placement atomically."""

        async with self.repository.transaction() as transaction:
            workload = await transaction.get_workload(decision.workload_id)
            cells = {lease.cell_id: await transaction.get_cell(lease.cell_id) for lease in decision.all_leases}
        if (
            workload is None
            or workload.state is not WorkloadState.PENDING
            or workload.preemption_id is not None
            or len(cells) != len(decision.all_leases)
            or any(
                cell is None
                or cell.state is not CellState.READY
                or cell.autoscale_request_workload_id not in {None, workload.id}
                for cell in cells.values()
            )
        ):
            return
        group_id = stable_id("pod_group", workload.id, workload.attempt, workload.generation)
        records: list[LeaseRecord] = []
        for index, proposed in enumerate(decision.all_leases):
            cell = cells[proposed.cell_id]
            assert cell is not None
            records.append(
                LeaseRecord(
                    id=proposed.lease_id,
                    workload_id=workload.id,
                    cell_id=proposed.cell_id,
                    cell_generation=cell.generation,
                    org_id=workload.org_id,
                    project_id=workload.project_id,
                    resources=ResourceGang(
                        replicas=workload.spec.gang.replicas,
                        per_replica=workload.spec.gang.per_replica,
                        required_topology=workload.spec.gang.required_topology,
                    ),
                    hourly_cost=float(proposed.hourly_cost),
                    reserved_cost=float(proposed.reserved_cost),
                    resource_hours_reserved=_api_resources(proposed.resource_hours_reserved),
                    generation=workload.generation,
                    fencing_token=secrets.token_urlsafe(32),
                    group_id=group_id,
                    group_index=index,
                    expires_at=utcnow() + timedelta(seconds=self.settings.lease_ttl_s),
                )
            )
        await self.service.create_lease_group(
            self._principal,
            tuple(records),
            idempotency_key=f"schedule-group:{group_id}",
        )

    async def _commit_preemption(self, plan: PreemptionPlan) -> None:
        now = utcnow()
        async with self.repository.transaction() as transaction:
            target = await transaction.get_workload(plan.workload_id)
            if target is None or target.state is not WorkloadState.PENDING or self.service.scheduler is None:
                return
            if target.preemption_id is not None:
                existing = await transaction.get_outbox(target.preemption_id)
                if existing is not None and existing.state is not OutboxState.SUCCEEDED:
                    return
            active_leases = [lease for lease in await transaction.list_leases() if lease.state in ACTIVE_LEASE_STATES]
            lease_snapshots = {lease.id: lease for lease in active_leases}
            victims: list[tuple[LeaseRecord, WorkloadRecord]] = []
            for proposed in plan.victims:
                snapshot = lease_snapshots.get(proposed.lease_id)
                if snapshot is None:
                    return
                workload = await transaction.get_workload(snapshot.workload_id)
                lease = await transaction.get_lease(snapshot.id)
                if workload is None or lease is None:
                    return
                victims.append((lease, workload))

            await transaction.get_idempotency(
                "fleet",
                "admission.org",
                storage_scope(target.org_id, ""),
            )
            await transaction.get_idempotency(
                "fleet",
                "admission.project",
                storage_scope(target.org_id, target.project_id),
            )
            victim_scopes = sorted({(workload.org_id, workload.project_id) for _, workload in victims})
            for org_id, project_id in victim_scopes:
                if (org_id, project_id) == (target.org_id, target.project_id):
                    continue
                await transaction.get_idempotency("fleet", "admission.org", storage_scope(org_id, ""))
                await transaction.get_idempotency(
                    "fleet",
                    "admission.project",
                    storage_scope(org_id, project_id),
                )
            cell = await transaction.get_cell(plan.cell_id)
            if cell is None or cell.state is not CellState.READY:
                return
            policy = await transaction.get_policy(target.org_id, target.project_id)
            if policy is not None:
                try:
                    self.service._validate_workload_policy(target.spec, policy)
                except InputError:
                    return
            budget = await transaction.get_budget(target.org_id, target.project_id)
            active_workload_ids = {active.workload_id for active in active_leases}
            validation_workloads = [
                workload
                for workload in await transaction.list_workloads()
                if workload.id == target.id or workload.id in active_workload_ids
            ]
            refreshed = self.service.scheduler.schedule(
                validation_workloads,
                await transaction.list_cells(),
                leases=active_leases,
                budgets=() if budget is None else (budget,),
                quotas=await transaction.list_quotas(org_id=target.org_id),
                policies=await transaction.list_policies(),
                # Same federation inputs the plan pass used; a preemption plan
                # that only fits under the grant bound must not revalidate
                # against the looser local limit.
                grants=await transaction.list_budget_grants() if self.service.plane_id else [],
                quota_grants=await transaction.list_quota_grants() if self.service.plane_id else [],
                plane_id=self.service.plane_id,
            )
            current_plan = next(
                (candidate for candidate in refreshed.preemptions if candidate.workload_id == target.id),
                None,
            )
            if current_plan is None or current_plan.cell_id != plan.cell_id:
                return
            if tuple(victim.lease_id for victim in current_plan.victims) != tuple(
                victim.lease_id for victim in plan.victims
            ):
                return
            plan = current_plan

            by_lease = {lease.id: (lease, workload) for lease, workload in victims}
            for proposed in plan.victims:
                lease, workload = by_lease[proposed.lease_id]
                # Membership, not equality with the leader. A workload holding a
                # lease *group* records the group in ``lease_ids`` and only the
                # leader in ``lease_id``, and the scheduler deliberately names
                # every sibling as a victim -- preempting half a gang bills for
                # capacity that cannot make progress. Testing ``lease_id`` meant
                # the first sibling failed and the whole plan was abandoned, so a
                # multi-cell gang could never be preempted at all: the
                # high-priority workload waited forever while the metrics counted
                # a preemption that never happened.
                held = set(workload.lease_ids) or ({workload.lease_id} if workload.lease_id else set())
                if (
                    lease.state not in {LeaseState.PENDING, LeaseState.ACTIVE, LeaseState.QUARANTINED}
                    or lease.id not in held
                    or workload.generation != lease.generation
                    or workload.state not in {WorkloadState.LEASED, WorkloadState.DISPATCHING, WorkloadState.RUNNING}
                ):
                    return

            command_id = stable_id(
                "cmd_preempt_plan",
                target.id,
                target.attempt,
                target.generation,
                uuid.uuid4().hex,
            )
            target = target.model_copy(
                update={
                    "scheduling_reason": "waiting for checkpoint-aware preemption",
                    "preemption_id": command_id,
                    "updated_at": now,
                }
            )
            await transaction.put_workload(target)
            victim_payloads: list[JsonObject] = []
            for victim in plan.victims:
                lease, workload = by_lease[victim.lease_id]
                victim_lease_state = lease.state
                victim_workload_state = workload.state
                lease = lease.model_copy(
                    update={
                        "state": LeaseState.REVOKING,
                        "revocation_reason": "preemption",
                        "updated_at": now,
                    }
                )
                should_checkpoint = (
                    victim.action is PreemptionActionKind.CHECKPOINT_AND_STOP and workload.remote_id is not None
                )
                next_state = WorkloadState.CHECKPOINTING if should_checkpoint else WorkloadState.CANCELLING
                workload = workload.model_copy(
                    update={"state": next_state, "preemption_id": command_id, "updated_at": now}
                )
                await transaction.put_lease(lease)
                await transaction.put_workload(workload)
                victim_payloads.append(
                    model_json(
                        _PreemptionVictim(
                            workload_id=workload.id,
                            lease_id=lease.id,
                            cell_id=lease.cell_id,
                            generation=workload.generation,
                            checkpoint=should_checkpoint,
                            grace_period_seconds=victim.grace_period_seconds,
                            hook=workload.spec.checkpoint_hook if should_checkpoint else None,
                            output_uri=workload.spec.checkpoint_output_uri if should_checkpoint else None,
                            workload_state=victim_workload_state,
                            lease_state=victim_lease_state,
                        )
                    )
                )
                await transaction.append_event(
                    AuditEvent(
                        id=f"evt_{uuid.uuid4().hex}",
                        actor=self._principal.subject,
                        org_id=workload.org_id,
                        project_id=workload.project_id,
                        entity_type="workload",
                        entity_id=workload.id,
                        action="workload.preemption_started",
                        new_state=next_state.value,
                        payload={"target_workload_id": target.id, "lease_id": lease.id},
                    )
                )
            await transaction.put_outbox(
                OutboxRecord(
                    id=command_id,
                    topic="workload.preempt",
                    correlation_id=canonical_correlation_id(target.id) if self.service.correlation_enabled else None,
                    aggregate_type="workload",
                    aggregate_id=target.id,
                    idempotency_key=command_id,
                    generation=target.generation,
                    payload={
                        "target_workload_id": target.id,
                        "target_generation": target.generation,
                        "target_attempt": target.attempt,
                        "cell_id": plan.cell_id,
                        "execution_started": False,
                        "victims": victim_payloads,
                    },
                )
            )

    async def _record_infeasibilities(self, result: SchedulingResult) -> None:
        async with self.repository.transaction() as transaction:
            for failure in result.infeasibilities:
                workload = await transaction.get_workload(failure.workload_id)
                if workload is None or workload.state is not WorkloadState.PENDING:
                    continue
                if workload.autoscale_block_reason is not None:
                    continue
                if is_acquisition_backoff_reason(workload.scheduling_reason):
                    # The capacity pass owns this sentence and clears it itself;
                    # "no cell satisfies" would hide why no cell exists.
                    continue
                if await self._has_active_autoscale_capacity(transaction, workload):
                    continue
                reason = f"{failure.code.value}: {failure.message}"
                if workload.scheduling_reason != reason:
                    await transaction.put_workload(
                        workload.model_copy(update={"scheduling_reason": reason, "updated_at": utcnow()})
                    )

    @staticmethod
    async def _has_active_autoscale_capacity(
        transaction: RepositorySession,
        workload: WorkloadRecord,
    ) -> bool:
        """Return whether an owned physical cell is still being provisioned."""

        return any(
            cell.state is CellState.REGISTERING
            for cell in await transaction.list_cells_for_autoscale_request(workload.id)
        )

    async def _record_acquisition_exhausted(
        self,
        workloads: tuple[WorkloadRecord, ...],
        exhausted_workload_ids: frozenset[str],
    ) -> None:
        """Say once, durably, that every zone a workload accepts stopped granting.

        Without this the workload stays PENDING with no reason while the
        autoscaler asks the same zone again each time belief decays -- twenty
        cells over nineteen hours, in the case that prompted it. A pending job
        that will not progress has to say so; retrying silently forever is
        indistinguishable from working.

        Cleared the moment a pool grants again, because the bound is on
        consecutive failures and one success ends the run.
        """

        pending = tuple(sorted({workload.id for workload in workloads if workload.state is WorkloadState.PENDING}))
        if not pending:
            return
        async with self.repository.transaction() as transaction:
            for workload_id in pending:
                workload = await transaction.get_workload(workload_id)
                if workload is None or workload.state is not WorkloadState.PENDING:
                    continue
                exhausted = workload.id in exhausted_workload_ids
                already = workload.autoscale_block_reason is AutoscaleBlockReason.CAPACITY_ACQUISITION_EXHAUSTED
                if exhausted and not already:
                    await transaction.put_workload(
                        workload.model_copy(
                            update={
                                "autoscale_block_reason": AutoscaleBlockReason.CAPACITY_ACQUISITION_EXHAUSTED,
                                "scheduling_reason": _ACQUISITION_EXHAUSTED_SCHEDULING_REASON,
                                "updated_at": utcnow(),
                            }
                        )
                    )
                elif already and not exhausted:
                    reason = workload.scheduling_reason
                    await transaction.put_workload(
                        workload.model_copy(
                            update={
                                "autoscale_block_reason": None,
                                "scheduling_reason": (
                                    None if reason == _ACQUISITION_EXHAUSTED_SCHEDULING_REASON else reason
                                ),
                                "updated_at": utcnow(),
                            }
                        )
                    )

    async def _record_autoscale_quota_blocks(
        self,
        workloads: tuple[WorkloadRecord, ...],
        blocked_workload_ids: frozenset[str],
    ) -> None:
        """Persist the local shared-Spot-quota outcome for current pending workloads."""

        pending_workload_ids = tuple(
            sorted({workload.id for workload in workloads if workload.state is WorkloadState.PENDING})
        )
        if not pending_workload_ids:
            return

        async with self.repository.transaction() as transaction:
            for workload_id in pending_workload_ids:
                workload = await transaction.get_workload(workload_id)
                if workload is None or workload.state is not WorkloadState.PENDING:
                    continue

                quota_blocked = workload.id in blocked_workload_ids
                if quota_blocked:
                    if (
                        workload.autoscale_block_reason is AutoscaleBlockReason.SHARED_TPU_QUOTA
                        and workload.scheduling_reason == _SHARED_TPU_QUOTA_SCHEDULING_REASON
                    ):
                        continue
                    await transaction.put_workload(
                        workload.model_copy(
                            update={
                                "autoscale_block_reason": AutoscaleBlockReason.SHARED_TPU_QUOTA,
                                "queue_admission_denial": None,
                                "scheduling_reason": _SHARED_TPU_QUOTA_SCHEDULING_REASON,
                                "updated_at": utcnow(),
                            }
                        )
                    )
                    continue

                if workload.autoscale_block_reason is not AutoscaleBlockReason.SHARED_TPU_QUOTA:
                    continue
                scheduling_reason = workload.scheduling_reason
                if scheduling_reason == _SHARED_TPU_QUOTA_SCHEDULING_REASON:
                    scheduling_reason = None
                await transaction.put_workload(
                    workload.model_copy(
                        update={
                            "autoscale_block_reason": None,
                            "queue_admission_denial": None,
                            "scheduling_reason": scheduling_reason,
                            "updated_at": utcnow(),
                        }
                    )
                )
