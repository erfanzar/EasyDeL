# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.records import CellStageState
from eyvan.server.service._control_plane.shared import (
    ACTIVE_LEASE_STATES,
    AuditEvent,
    CellRecord,
    CellState,
    ConflictError,
    IdempotencyRecord,
    InputError,
    InvalidTransition,
    JsonObject,
    LeaseRecord,
    LeaseState,
    NotFoundError,
    Permission,
    PlacementMode,
    Principal,
    RepositorySession,
    WorkloadRecord,
    WorkloadState,
    _fingerprint,
    _new_id,
    _resource_quantities,
    _scope,
    _validate_idempotency_key,
    datetime,
    ensure_lease_transition,
    ensure_workload_transition,
    tp,
    utcnow,
)


class LeasesMixin:
    """Transactional leases operations."""

    @staticmethod
    async def _require_no_asset_eviction(tx: RepositorySession, workload: WorkloadRecord, cell: CellRecord) -> None:
        """Refuse use of an asset with an unresolved same-host deletion.

        Call only after locking the cell through ``get_cell`` in this transaction:
        eviction marking takes that same lock before checking live holders.
        Missing/ordinary unknown stages remain cache misses, not admission gates.
        """

        for asset in sorted(set(workload.spec.named_assets)):
            stage = await tx.get_cell_stage(cell.id, asset)
            if stage is not None and stage.cell_generation == cell.generation and stage.state is CellStageState.EVICTING:
                raise ConflictError(f"cell {cell.id!r} has an unresolved asset eviction")

    async def create_lease(
        self, principal: Principal, lease: LeaseRecord, *, idempotency_key: str
    ) -> tuple[LeaseRecord, bool]:
        """Persist a scheduler lease and bind it atomically to its workload."""

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        key = _validate_idempotency_key(idempotency_key)
        scope = _scope(lease.org_id, lease.project_id)
        fingerprint = _fingerprint(lease.model_copy(update={"created_at": datetime.min, "updated_at": datetime.min}))
        async with self.repository.transaction() as tx:
            existing = await self._idempotent_resource(tx, scope, "lease.create", key, fingerprint)
            if existing is not None:
                stored = await tx.get_lease(existing.resource_id)
                if stored is None:
                    raise ConflictError("idempotency record points to a missing lease")
                return stored, True
            workload = await self._require_workload(tx, lease.workload_id)
            await self._lock_admission_scope(tx, workload.org_id, workload.project_id)
            if workload.org_id != lease.org_id or workload.project_id != lease.project_id:
                raise InputError("lease scope does not match its workload")
            if workload.state is not WorkloadState.PENDING:
                raise ConflictError(f"workload {workload.id!r} is {workload.state.value}, not pending")
            if workload.preemption_id is not None:
                raise ConflictError("workload is reserved by an executing preemption plan")
            if lease.generation != workload.generation:
                raise ConflictError(
                    f"lease generation {lease.generation} does not match workload generation {workload.generation}"
                )
            cell = await tx.get_cell(lease.cell_id)
            if cell is None:
                raise NotFoundError(f"cell {lease.cell_id!r} was not found")
            if cell.state is not CellState.READY:
                raise ConflictError(f"cell {lease.cell_id!r} is {cell.state.value}, not ready for a new lease")
            if lease.cell_generation != cell.generation:
                raise ConflictError(
                    f"lease cell generation {lease.cell_generation} does not match cell generation {cell.generation}"
                )
            self._require_autoscale_request_ownership(cell, workload.id)
            await self._require_no_asset_eviction(tx, workload, cell)
            if self.scheduler is not None:
                policy = await tx.get_policy(workload.org_id, workload.project_id)
                if policy is not None:
                    self._validate_workload_policy(workload.spec, policy)
                budget = await tx.get_budget(workload.org_id, workload.project_id)
                active_leases = [
                    candidate for candidate in await tx.list_leases() if candidate.state in ACTIVE_LEASE_STATES
                ]
                active_workload_ids = {active.workload_id for active in active_leases}
                validation_workloads = [
                    candidate
                    for candidate in await tx.list_workloads()
                    if candidate.id == workload.id or candidate.id in active_workload_ids
                ]
                validation = self.scheduler.schedule(
                    validation_workloads,
                    await tx.list_cells(),
                    leases=active_leases,
                    budgets=() if budget is None else (budget,),
                    quotas=await tx.list_quotas(org_id=workload.org_id),
                    policies=() if policy is None else (policy,),
                    # The plan pass tightens a grantee plane's limits through its
                    # grants; revalidation must re-derive with the same inputs,
                    # or the authoritative check enforces weaker limits than
                    # the decision it is confirming.
                    grants=await tx.list_budget_grants() if self.plane_id else [],
                    quota_grants=await tx.list_quota_grants() if self.plane_id else [],
                    plane_id=self.plane_id,
                )
                placement = next(
                    (candidate for candidate in validation.placements if candidate.workload_id == workload.id),
                    None,
                )
                if placement is None or placement.lease.cell_id != lease.cell_id:
                    reason = (
                        validation.infeasibilities[0].message
                        if validation.infeasibilities
                        else "placement decision is no longer immediately admissible"
                    )
                    raise ConflictError(reason)
                if placement.lease.lease_id != lease.id:
                    raise ConflictError("lease identifier does not match the current placement decision")
                # Revalidation re-ran the scheduler against every active lease, so
                # these chips are disjoint from every other share on the cell now.
                if tuple(lease.tpu_chips or ()) != placement.lease.tpu_chips:
                    raise ConflictError("lease TPU chips do not match the current placement decision")
                if lease.resources != workload.spec.gang:
                    raise ConflictError("lease resources do not match the current workload gang")
                if lease.hourly_cost != float(placement.lease.hourly_cost) or lease.reserved_cost != float(
                    placement.lease.reserved_cost
                ):
                    raise ConflictError("lease cost does not match current cell pricing")
                actual_hours = {
                    name: quantity
                    for name, quantity in _resource_quantities(lease.resource_hours_reserved).items()
                    if quantity
                }
                expected_hours = {
                    name: float(quantity) for name, quantity in placement.lease.resource_hours_reserved.items()
                }
                if actual_hours != expected_hours:
                    raise ConflictError("lease resource-hour reservation is stale")
            workload = await self._transition_workload(
                tx,
                principal,
                workload,
                WorkloadState.ADMITTED,
                "workload.admitted",
            )
            ensure_workload_transition(workload.state, WorkloadState.LEASED)
            now = utcnow()
            await self._clear_autoscale_request_ownership(tx, cell, workload.id, now)
            await tx.put_lease(lease)
            updated_workload = workload.model_copy(
                update={
                    "state": WorkloadState.LEASED,
                    "lease_id": lease.id,
                    "dispatch_reservation": None,
                    "preemption_id": None,
                    "autoscale_block_reason": None,
                    "queue_admission_denial": None,
                    "scheduling_reason": None,
                    "updated_at": now,
                }
            )
            await tx.put_workload(updated_workload)
            await self._outbox(
                tx,
                topic="workload.dispatch",
                aggregate_type="workload",
                aggregate_id=updated_workload.id,
                idempotency_key=f"dispatch:{lease.id}:{lease.generation}",
                generation=lease.generation,
                payload={
                    "workload_id": updated_workload.id,
                    "lease_id": lease.id,
                    "cell_id": lease.cell_id,
                },
            )
            await self._bind_idempotency(tx, scope, "lease.create", key, fingerprint, lease)
            await tx.append_event(
                AuditEvent(
                    id=_new_id("evt"),
                    actor=principal.subject,
                    org_id=lease.org_id,
                    project_id=lease.project_id,
                    entity_type="lease",
                    entity_id=lease.id,
                    action="lease.created",
                    new_state=lease.state.value,
                    payload={"workload_id": lease.workload_id, "cell_id": lease.cell_id},
                )
            )
            await self._audit(
                tx,
                principal,
                updated_workload,
                "workload.leased",
                previous_state=workload.state.value,
                new_state=updated_workload.state.value,
                payload={"lease_id": lease.id},
            )
            return lease, False

    async def create_lease_group(
        self,
        principal: Principal,
        leases: tp.Sequence[LeaseRecord],
        *,
        idempotency_key: str,
    ) -> tuple[tuple[LeaseRecord, ...], bool]:
        """Persist a complete spread-cell placement in one transaction.

        The first member remains the legacy workload ``lease_id`` leader, but
        ``lease_ids`` is the authoritative ordered membership list. A caller
        never receives a partially persisted TPU pod group.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        members = tuple(leases)
        if len(members) < 2:
            raise InputError("a lease group requires at least two members")
        key = _validate_idempotency_key(idempotency_key)
        leader = members[0]
        scope = _scope(leader.org_id, leader.project_id)
        fingerprint = _fingerprint(
            {"leases": [member.model_dump(mode="json", exclude={"created_at", "updated_at"}) for member in members]}
        )
        workload_ids = {member.workload_id for member in members}
        cell_ids = [member.cell_id for member in members]
        group_ids = {member.group_id for member in members}
        if len(workload_ids) != 1:
            raise InputError("lease group members must belong to one workload")
        if len(cell_ids) != len(set(cell_ids)):
            raise InputError("lease group members must use distinct cells")
        if len(group_ids) != 1 or None in group_ids:
            raise InputError("lease group members must share a non-empty group_id")
        if tuple(member.group_index for member in members) != tuple(range(len(members))):
            raise InputError("lease group members must have contiguous group_index values starting at zero")
        if any(member.org_id != leader.org_id or member.project_id != leader.project_id for member in members):
            raise InputError("lease group members must share one organization and project")

        async with self.repository.transaction() as tx:
            existing = await self._idempotent_resource(tx, scope, "lease.group.create", key, fingerprint)
            if existing is not None:
                workload = await tx.get_workload(existing.resource_id)
                if workload is None:
                    raise ConflictError("idempotency record points to a missing workload")
                # The fingerprint binds this request's original identities; the
                # workload's current membership can belong to a later retry.
                identity_fields = (
                    "id",
                    "workload_id",
                    "org_id",
                    "project_id",
                    "group_id",
                    "group_index",
                    "generation",
                    "cell_id",
                    "cell_generation",
                    "fencing_token",
                )
                stored: list[LeaseRecord] = []
                for member in members:
                    original = await tx.get_lease(member.id)
                    if original is None:
                        raise ConflictError("idempotency record points to an incomplete original lease group")
                    if existing.resource_id != member.workload_id or any(
                        getattr(original, field) != getattr(member, field) for field in identity_fields
                    ):
                        raise ConflictError("idempotency record does not match the original lease group")
                    stored.append(original)
                return tuple(stored), True

            workload = await self._require_workload(tx, leader.workload_id)
            await self._lock_admission_scope(tx, workload.org_id, workload.project_id)
            if workload.org_id != leader.org_id or workload.project_id != leader.project_id:
                raise InputError("lease group scope does not match its workload")
            if workload.state is not WorkloadState.PENDING:
                raise ConflictError(f"workload {workload.id!r} is {workload.state.value}, not pending")
            if workload.preemption_id is not None:
                raise ConflictError("workload is reserved by an executing preemption plan")
            if workload.spec.placement is not PlacementMode.ELASTIC and workload.spec.cell_group_count != len(members):
                raise ConflictError(f"workload requires {workload.spec.cell_group_count} cells, not {len(members)}")
            if any(member.generation != workload.generation for member in members):
                raise ConflictError("lease group generation does not match the workload generation")

            cells: dict[str, CellRecord] = {}
            for member in members:
                cell = await tx.get_cell(member.cell_id)
                if cell is None:
                    raise NotFoundError(f"cell {member.cell_id!r} was not found")
                if cell.state is not CellState.READY:
                    raise ConflictError(f"cell {member.cell_id!r} is {cell.state.value}, not ready for a new lease")
                if member.cell_generation != cell.generation:
                    raise ConflictError(
                        f"lease cell generation {member.cell_generation} does not match "
                        f"cell generation {cell.generation}"
                    )
                self._require_autoscale_request_ownership(cell, workload.id)
                await self._require_no_asset_eviction(tx, workload, cell)
                cells[cell.id] = cell

            if workload.spec.placement is PlacementMode.ELASTIC:
                # An elastic pack spans however many cells its warm supply
                # dictated; the invariant is the worker total, not a member
                # count fixed at submission time.
                packed = sum(cell.worker_count for cell in cells.values())
                if packed != workload.spec.worker_count:
                    raise ConflictError(
                        f"elastic workload requires {workload.spec.worker_count} workers across its cells, not {packed}"
                    )

            if self.scheduler is not None:
                policy = await tx.get_policy(workload.org_id, workload.project_id)
                if policy is not None:
                    self._validate_workload_policy(workload.spec, policy)
                budget = await tx.get_budget(workload.org_id, workload.project_id)
                active_leases = [
                    candidate for candidate in await tx.list_leases() if candidate.state in ACTIVE_LEASE_STATES
                ]
                active_workload_ids = {active.workload_id for active in active_leases}
                validation_workloads = [
                    candidate
                    for candidate in await tx.list_workloads()
                    if candidate.id == workload.id or candidate.id in active_workload_ids
                ]
                validation = self.scheduler.schedule(
                    validation_workloads,
                    await tx.list_cells(),
                    leases=active_leases,
                    budgets=() if budget is None else (budget,),
                    quotas=await tx.list_quotas(org_id=workload.org_id),
                    policies=() if policy is None else (policy,),
                    # The plan pass tightens a grantee plane's limits through its
                    # grants; revalidation must re-derive with the same inputs,
                    # or the authoritative check enforces weaker limits than
                    # the decision it is confirming.
                    grants=await tx.list_budget_grants() if self.plane_id else [],
                    quota_grants=await tx.list_quota_grants() if self.plane_id else [],
                    plane_id=self.plane_id,
                )
                placement = next(
                    (candidate for candidate in validation.placements if candidate.workload_id == workload.id),
                    None,
                )
                if placement is None or len(placement.all_leases) != len(members):
                    reason = (
                        validation.infeasibilities[0].message
                        if validation.infeasibilities
                        else "pod group placement is no longer immediately admissible"
                    )
                    raise ConflictError(reason)
                expected = {candidate.lease_id: candidate for candidate in placement.all_leases}
                if set(expected) != {member.id for member in members}:
                    raise ConflictError("lease group identifiers do not match the current placement decision")
                for member in members:
                    candidate = expected[member.id]
                    if candidate.cell_id != member.cell_id:
                        raise ConflictError("lease group cell does not match the current placement decision")
                    if member.resources != workload.spec.gang:
                        raise ConflictError("lease group resources do not match the current workload gang")
                    if member.hourly_cost != float(candidate.hourly_cost) or member.reserved_cost != float(
                        candidate.reserved_cost
                    ):
                        raise ConflictError("lease group cost does not match current cell pricing")
                    actual_hours = {
                        name: quantity
                        for name, quantity in _resource_quantities(member.resource_hours_reserved).items()
                        if quantity
                    }
                    expected_hours = {
                        name: float(quantity) for name, quantity in candidate.resource_hours_reserved.items()
                    }
                    if actual_hours != expected_hours:
                        raise ConflictError("lease group resource-hour reservation is stale")

            workload = await self._transition_workload(
                tx,
                principal,
                workload,
                WorkloadState.ADMITTED,
                "workload.admitted",
            )
            ensure_workload_transition(workload.state, WorkloadState.LEASED)
            now = utcnow()
            for member in members:
                cell = cells[member.cell_id]
                await self._clear_autoscale_request_ownership(tx, cell, workload.id, now)
                await tx.put_lease(member)
            updated_workload = workload.model_copy(
                update={
                    "state": WorkloadState.LEASED,
                    "lease_id": leader.id,
                    "lease_ids": tuple(member.id for member in members),
                    "dispatch_reservation": None,
                    "preemption_id": None,
                    "autoscale_block_reason": None,
                    "queue_admission_denial": None,
                    "scheduling_reason": None,
                    "updated_at": now,
                }
            )
            await tx.put_workload(updated_workload)
            await self._outbox(
                tx,
                topic="workload.dispatch",
                aggregate_type="workload",
                aggregate_id=updated_workload.id,
                idempotency_key=f"dispatch:{leader.id}:{leader.generation}",
                generation=leader.generation,
                payload={
                    "workload_id": updated_workload.id,
                    "lease_id": leader.id,
                    "lease_ids": list(updated_workload.lease_ids),
                    "cell_id": leader.cell_id,
                    "cell_ids": cell_ids,
                },
            )
            await tx.put_idempotency(
                IdempotencyRecord(
                    scope=scope,
                    operation="lease.group.create",
                    key=key,
                    fingerprint=fingerprint,
                    resource_type="workload",
                    resource_id=updated_workload.id,
                )
            )
            for member in members:
                await tx.append_event(
                    AuditEvent(
                        id=_new_id("evt"),
                        actor=principal.subject,
                        org_id=member.org_id,
                        project_id=member.project_id,
                        entity_type="lease",
                        entity_id=member.id,
                        action="lease.created",
                        new_state=member.state.value,
                        payload={
                            "workload_id": member.workload_id,
                            "cell_id": member.cell_id,
                            "group_id": member.group_id,
                            "group_index": member.group_index,
                        },
                    )
                )
            await self._audit(
                tx,
                principal,
                updated_workload,
                "workload.leased",
                previous_state=workload.state.value,
                new_state=updated_workload.state.value,
                payload={"lease_ids": list(updated_workload.lease_ids)},
            )
            return members, False

    async def transition_lease(
        self, principal: Principal, lease_id: str, target: LeaseState, *, action: str = "lease.state_changed"
    ) -> LeaseRecord:
        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            lease = await tx.get_lease(lease_id)
            if lease is None:
                raise NotFoundError(f"lease {lease_id!r} was not found")
            try:
                ensure_lease_transition(lease.state, target)
            except InvalidTransition as exc:
                raise ConflictError(str(exc)) from exc
            now = utcnow()
            changes: JsonObject | dict[str, object] = {"state": target, "updated_at": now}
            if target is LeaseState.ACTIVE and lease.run_started_at is None:
                # Same run boundary the controller's dispatch path stamps. Recorded
                # here too so every route into ACTIVE starts the settlement clock,
                # rather than only the one the controller happens to take.
                changes["run_started_at"] = now
            updated = lease.model_copy(update=changes)
            await tx.put_lease(updated)
            await tx.append_event(
                AuditEvent(
                    id=_new_id("evt"),
                    actor=principal.subject,
                    org_id=lease.org_id,
                    project_id=lease.project_id,
                    entity_type="lease",
                    entity_id=lease.id,
                    action=action,
                    previous_state=lease.state.value,
                    new_state=target.value,
                )
            )
            return updated
