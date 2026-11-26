# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.server.service._control_plane.shared import (
    _CELL_LIFECYCLE_TOPIC_BY_STATE,
    AuditEvent,
    AuthorizationError,
    CellRecord,
    CellState,
    ConflictError,
    IdempotencyRecord,
    InputError,
    LeaseState,
    NotFoundError,
    Permission,
    PlacementMode,
    Principal,
    WorkerResourceScope,
    _capacity_apply_payload,
    _changed_cell_capacity_identity,
    _fingerprint,
    _is_active_capacity_apply,
    _new_id,
    _resource_quantities,
    _validate_idempotency_key,
    cell_matches_template,
    datetime,
    utcnow,
)


class CellsMixin:
    """Transactional cells operations."""

    async def upsert_cell(
        self, principal: Principal, cell: CellRecord, *, idempotency_key: str
    ) -> tuple[CellRecord, bool]:
        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        key = _validate_idempotency_key(idempotency_key)
        fingerprint = _fingerprint(cell.model_copy(update={"created_at": datetime.min, "updated_at": datetime.min}))
        async with self.repository.transaction() as tx:
            existing_key = await self._idempotent_resource(tx, "fleet", "cell.upsert", key, fingerprint)
            if existing_key is not None:
                existing_cell = await tx.get_cell(existing_key.resource_id)
                if existing_cell is None:
                    raise ConflictError("idempotency record points to a missing cell")
                return existing_cell, True
            existing = await tx.get_cell(cell.id)
            if (
                existing is not None
                and (existing.billing_started_at is not None or existing.billed_seconds > 0 or existing.idle_seconds > 0)
                and (existing.unit_prices != cell.unit_prices or existing.hourly_cost() != cell.hourly_cost())
            ):
                # The ledger stores elapsed time, not historical rate intervals.
                # Until such a ledger exists, a new rate would reprice old hours.
                raise ConflictError("cannot change a cell billing rate after billing has started")
            endpoint_changed = existing is not None and existing.endpoint != cell.endpoint
            if existing is not None and existing.state is not CellState.DELETED:
                changed_identity = _changed_cell_capacity_identity(existing, cell)
                if changed_identity:
                    raise ConflictError(
                        f"cell capacity identity cannot change before deletion: {', '.join(changed_identity)}"
                    )
                if existing.state in {CellState.DRAINING, CellState.DELETING}:
                    raise ConflictError(f"cannot update a {existing.state.value} cell")
                active_leases = [
                    lease
                    for lease in await tx.list_leases()
                    if lease.cell_id == cell.id
                    and lease.state
                    in {LeaseState.PENDING, LeaseState.ACTIVE, LeaseState.REVOKING, LeaseState.QUARANTINED}
                ]
                if active_leases and existing.endpoint != cell.endpoint:
                    raise ConflictError("cell agent endpoint cannot change while leases are active")
                requested: dict[str, float] = {}
                for lease in active_leases:
                    workload = await self._require_workload(tx, lease.workload_id)
                    worker_multiplier = 1
                    if workload.spec.worker_resource_scope is WorkerResourceScope.PER_WORKER:
                        worker_multiplier = (
                            existing.worker_count
                            if workload.spec.placement is PlacementMode.ELASTIC
                            else workload.spec.worker_count
                        )
                    per_replica = _resource_quantities(lease.resources.per_replica)
                    for resource, quantity in per_replica.items():
                        requested[resource] = requested.get(resource, 0) + (
                            quantity * lease.resources.replicas * worker_multiplier
                        )
                allocatable = _resource_quantities(cell.allocatable)
                exceeded = sorted(
                    resource for resource, quantity in requested.items() if quantity > allocatable.get(resource, 0)
                )
                if exceeded:
                    raise ConflictError(f"updated cell capacity cannot contain active lease resource {exceeded[0]!r}")
                if cell.max_replica_resources is not None:
                    maximum = _resource_quantities(cell.max_replica_resources)
                    for lease in active_leases:
                        oversized = sorted(
                            resource
                            for resource, quantity in _resource_quantities(lease.resources.per_replica).items()
                            if quantity > maximum.get(resource, 0)
                        )
                        if oversized:
                            raise ConflictError(
                                f"updated per-replica capacity cannot contain active lease resource {oversized[0]!r}"
                            )
                if cell.max_gang_replicas is not None:
                    oversized_gang = next(
                        (lease for lease in active_leases if lease.resources.replicas > cell.max_gang_replicas),
                        None,
                    )
                    if oversized_gang is not None:
                        raise ConflictError(f"updated replica limit cannot contain active lease {oversized_gang.id!r}")
            now = utcnow()
            template_changed_while_unprovisioned = (
                existing is not None
                and existing.pool_id is not None
                and existing.state is CellState.REGISTERING
                and existing.provider_phase is None
                and not cell_matches_template(existing, cell)
            )
            requires_provision = (
                existing is None
                or existing.state is CellState.DELETED
                or endpoint_changed
                or template_changed_while_unprovisioned
            )
            stored = cell.model_copy(
                update={
                    "generation": (
                        1 if existing is None else existing.generation + 1 if requires_provision else existing.generation
                    ),
                    "state": CellState.REGISTERING if requires_provision else existing.state,
                    "created_at": now if existing is None else existing.created_at,
                    "availability": cell.availability if existing is None else existing.availability,
                    "provider_phase": None if requires_provision else existing.provider_phase,
                    "provider_failure_code": None if requires_provision else existing.provider_failure_code,
                    # Provider-owned rank width is observation state, not
                    # declarative template metadata. A labels/prices update can
                    # happen while leases are active and must not reset it to
                    # the model's legacy single-host default.
                    "runtime_process_count": (
                        cell.runtime_process_count
                        if existing is None or requires_provision
                        else existing.runtime_process_count
                    ),
                    "provider_phase_started_at": None if requires_provision else existing.provider_phase_started_at,
                    "unavailable_reason": None if requires_provision else existing.unavailable_reason,
                    "last_heartbeat_at": None if requires_provision else existing.last_heartbeat_at,
                    "billed_seconds": 0.0 if existing is None else existing.billed_seconds,
                    "idle_seconds": 0.0 if existing is None else existing.idle_seconds,
                    "billing_started_at": (None if requires_provision else existing.billing_started_at),
                    "billing_ticked_at": None if requires_provision else existing.billing_ticked_at,
                    "updated_at": now,
                }
            )
            await tx.put_cell(stored)
            if requires_provision:
                await self._outbox(
                    tx,
                    topic="capacity.apply",
                    aggregate_type="cell",
                    aggregate_id=stored.id,
                    idempotency_key=f"capacity:{stored.id}:{stored.generation}",
                    generation=stored.generation,
                    payload=_capacity_apply_payload(stored),
                )
            await tx.put_idempotency(
                IdempotencyRecord(
                    scope="fleet",
                    operation="cell.upsert",
                    key=key,
                    fingerprint=fingerprint,
                    resource_type="cell",
                    resource_id=stored.id,
                )
            )
            await tx.append_event(
                AuditEvent(
                    id=_new_id("evt"),
                    actor=principal.subject,
                    entity_type="cell",
                    entity_id=stored.id,
                    action="cell.registered" if existing is None else "cell.updated",
                    previous_state=None if existing is None else existing.state.value,
                    new_state=stored.state.value,
                    payload={"generation": stored.generation},
                )
            )
            return stored, False

    async def recreate_cell(
        self,
        principal: Principal,
        cell_id: str,
        *,
        expected_generation: int,
        idempotency_key: str,
    ) -> tuple[CellRecord, bool]:
        """Rotate an unavailable unleased cell to a fresh fenced capacity generation."""

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        key = _validate_idempotency_key(idempotency_key)
        fingerprint = _fingerprint({"cell_id": cell_id, "expected_generation": expected_generation})
        async with self.repository.transaction() as tx:
            existing_key = await self._idempotent_resource(tx, "fleet", "cell.recreate", key, fingerprint)
            cell = await tx.get_cell(cell_id)
            if cell is None:
                raise NotFoundError(f"cell {cell_id!r} was not found")
            if existing_key is not None:
                return cell, True
            if cell.generation != expected_generation:
                raise ConflictError(
                    f"cell {cell.id!r} is generation {cell.generation}, expected {expected_generation} for recreation"
                )
            if cell.state is not CellState.UNAVAILABLE:
                raise ConflictError(f"cell {cell.id!r} is {cell.state.value}, not unavailable for recreation")
            active = [
                lease
                for lease in await tx.list_leases()
                if lease.cell_id == cell.id and lease.state not in {LeaseState.RELEASED, LeaseState.EXPIRED}
            ]
            if active:
                raise ConflictError(f"cell {cell.id!r} still owns {len(active)} active lease(s)")
            now = utcnow()
            recreated = cell.model_copy(
                update={
                    "state": CellState.REGISTERING,
                    "generation": cell.generation + 1,
                    "provider_phase": None,
                    "provider_failure_code": None,
                    "provider_phase_started_at": None,
                    "unavailable_reason": None,
                    "last_heartbeat_at": None,
                    "billing_started_at": None,
                    "billing_ticked_at": None,
                    "updated_at": now,
                }
            )
            await tx.put_cell(recreated)
            await self._outbox(
                tx,
                topic="capacity.apply",
                aggregate_type="cell",
                aggregate_id=recreated.id,
                idempotency_key=f"capacity:{recreated.id}:{recreated.generation}",
                generation=recreated.generation,
                payload=_capacity_apply_payload(recreated),
            )
            await tx.put_idempotency(
                IdempotencyRecord(
                    scope="fleet",
                    operation="cell.recreate",
                    key=key,
                    fingerprint=fingerprint,
                    resource_type="cell",
                    resource_id=recreated.id,
                )
            )
            await tx.append_event(
                AuditEvent(
                    id=_new_id("evt"),
                    actor=principal.subject,
                    entity_type="cell",
                    entity_id=recreated.id,
                    action="cell.recreation_requested",
                    previous_state=cell.state.value,
                    new_state=recreated.state.value,
                    payload={"generation": recreated.generation},
                )
            )
            return recreated, False

    async def requeue_unprovisioned_cell(
        self,
        principal: Principal,
        cell_id: str,
        *,
        expected_generation: int,
    ) -> tuple[CellRecord, bool]:
        """Fence and requeue a registering cell that has no live capacity apply command.

        The absence of an active apply command is the durable recovery condition.
        This can happen when a fleet template supersedes an already claimed
        command. Bumping the cell generation ensures a delayed predecessor can
        never provision the repaired template.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            cell = await tx.get_cell(cell_id)
            if cell is None:
                raise NotFoundError(f"cell {cell_id!r} was not found")
            if cell.generation != expected_generation:
                raise ConflictError(
                    f"cell {cell.id!r} is generation {cell.generation}, expected {expected_generation} for provisioning"
                )
            if cell.state is not CellState.REGISTERING or cell.provider_phase is not None:
                raise ConflictError(f"cell {cell.id!r} is not an unprovisioned registering cell")
            # Scoped to this one cell and generation rather than scanning the
            # whole outbox: a limit-bounded scan could miss the very command
            # this check exists to find, and then requeue a live provision.
            commands = await tx.list_outbox_for_aggregates("cell", {cell.id: cell.generation}, topic="capacity.apply")
            if any(_is_active_capacity_apply(command, cell) for command in commands):
                return cell, True
            now = utcnow()
            requeued = cell.model_copy(update={"generation": cell.generation + 1, "updated_at": now})
            await tx.put_cell(requeued)
            await self._outbox(
                tx,
                topic="capacity.apply",
                aggregate_type="cell",
                aggregate_id=requeued.id,
                idempotency_key=f"capacity:{requeued.id}:{requeued.generation}",
                generation=requeued.generation,
                payload=_capacity_apply_payload(requeued),
            )
            await tx.append_event(
                AuditEvent(
                    id=_new_id("evt"),
                    actor=principal.subject,
                    entity_type="cell",
                    entity_id=requeued.id,
                    action="cell.provisioning_requeued",
                    previous_state=cell.state.value,
                    new_state=requeued.state.value,
                    payload={"generation": requeued.generation},
                )
            )
            return requeued, False

    async def refence_cell_lifecycle_command(
        self,
        principal: Principal,
        cell_id: str,
        *,
        expected_generation: int,
        expected_state: CellState,
        idempotency_key: str,
    ) -> tuple[CellRecord, bool]:
        """Replace a stale drain or release command with a strictly newer cell fence."""

        topic = _CELL_LIFECYCLE_TOPIC_BY_STATE.get(expected_state)
        if topic is None:
            raise InputError(f"cell state {expected_state.value!r} has no fenced lifecycle command")
        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        key = _validate_idempotency_key(idempotency_key)
        fingerprint = _fingerprint(
            {
                "cell_id": cell_id,
                "expected_generation": expected_generation,
                "expected_state": expected_state.value,
            }
        )
        async with self.repository.transaction() as tx:
            existing_key = await self._idempotent_resource(tx, "fleet", "cell.lifecycle_refence", key, fingerprint)
            cell = await tx.get_cell(cell_id)
            if cell is None:
                raise NotFoundError(f"cell {cell_id!r} was not found")
            if existing_key is not None:
                return cell, True
            if cell.generation != expected_generation:
                raise ConflictError(
                    f"cell {cell.id!r} is generation {cell.generation}, expected {expected_generation} "
                    "for lifecycle refence"
                )
            if cell.state is not expected_state:
                raise ConflictError(
                    f"cell {cell.id!r} is {cell.state.value}, expected {expected_state.value} for lifecycle refence"
                )
            now = utcnow()
            refenced = cell.model_copy(update={"generation": cell.generation + 1, "updated_at": now})
            await tx.put_cell(refenced)
            await self._outbox(
                tx,
                topic=topic,
                aggregate_type="cell",
                aggregate_id=refenced.id,
                idempotency_key=f"{topic}:{refenced.id}:{refenced.generation}",
                generation=refenced.generation,
                payload={"cell_id": refenced.id},
            )
            await tx.put_idempotency(
                IdempotencyRecord(
                    scope="fleet",
                    operation="cell.lifecycle_refence",
                    key=key,
                    fingerprint=fingerprint,
                    resource_type="cell",
                    resource_id=refenced.id,
                )
            )
            await tx.append_event(
                AuditEvent(
                    id=_new_id("evt"),
                    actor=principal.subject,
                    entity_type="cell",
                    entity_id=refenced.id,
                    action="cell.lifecycle_refenced",
                    previous_state=cell.state.value,
                    new_state=refenced.state.value,
                    payload={"generation": refenced.generation, "topic": topic},
                )
            )
            return refenced, False

    def _require_capacity_administration(
        self,
        principal: Principal,
        *,
        declared_owner: str | None,
        existing_owner: str | None,
    ) -> None:
        """Require authority over one pool, by fleet or by ownership.

        A fleet administrator may do anything, as before. An organization
        administrator may administer a pool its own organization owns -- which
        is the whole point of ``MANAGE_OWN_CAPACITY``: sizing capacity you pay
        for should not require rights over every other tenant's cells and the
        provider configuration underneath them.

        Both the existing and the declared owner are checked, and they are
        different refusals:

        * a pool someone else owns cannot be *taken over* by declaring your own
          organization on an update, and
        * a pool you own cannot be *handed away* -- to another organization, or
          to ``None``, which would make it shared capacity you can no longer
          reach and nobody asked the fleet operator about.

        Unowned pools stay fleet-only. ``owner_org_id`` defaults to ``None``, so
        every pool that existed before ownership did keeps exactly the authority
        it had.
        """

        if self.authorizer.allows(principal, Permission.MANAGE_FLEET):
            return
        if existing_owner is None and declared_owner is None:
            raise AuthorizationError(f"{principal.subject!r} lacks fleet administration for unowned capacity")
        for org_id, failure in (
            (existing_owner, "is owned by another organization"),
            (declared_owner, "would be owned by another organization"),
        ):
            if org_id is None:
                # A None on either side means the pool is shared before or after
                # this change, and moving capacity in or out of shared ownership
                # is a fleet decision.
                raise AuthorizationError(f"{principal.subject!r} cannot move capacity in or out of shared ownership")
            if not self.authorizer.allows(principal, Permission.MANAGE_OWN_CAPACITY, org_id=org_id):
                raise AuthorizationError(f"{principal.subject!r} cannot administer capacity that {failure}")
