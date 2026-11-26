# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan._internal import fleet_init_script
from eyvan.records import WorkloadRecord
from eyvan.server.service._control_plane.shared import (
    AuditEvent,
    CapacityClass,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    ConflictError,
    IdempotencyRecord,
    InputError,
    JsonObject,
    LeaseState,
    NotFoundError,
    Permission,
    Principal,
    _fingerprint,
    _new_id,
    _pool_declaration,
    _validate_idempotency_key,
    datetime,
    tp,
    utcnow,
    validate_capacity_pool_quota_groups,
)


def _fleet_pool_declaration(pool: CapacityPoolRecord) -> JsonObject:
    """Exclude controller-owned shrink state as well as initialization bookkeeping."""

    declaration = _pool_declaration(pool)
    declaration["demand_below_supply_since"] = None
    return declaration


def init_script_unsupported_reason(pool: CapacityPoolRecord) -> str | None:
    """Why a pool's new cells would never run a fleet init script, or ``None`` when they do.

    Only the GCP driver renders ``runtime.init_commands`` into a new host's
    bootstrap. An attached TPU or SSH host already exists: its worker was
    installed by ``eyvan gcp connect``, which is where its setup runs.
    """

    if pool.provider != "gcp":
        return f"pool {pool.id!r} uses provider {pool.provider!r}, whose hosts do not run fleet init scripts"
    if (pool.capacity_backend or "").strip().lower() == "gcp-attach":
        return (
            f"pool {pool.id!r} is an attached TPU: Eyvan never creates its hosts, so a script here would never run; "
            "pass setup to `eyvan gcp connect --init-command` instead"
        )
    return None


def _carry_init_script(pool: CapacityPoolRecord, previous: CapacityPoolRecord | None) -> CapacityPoolRecord:
    """Keep a dashboard-written init script across a fleet YAML re-apply that does not mention it.

    ``eyvan up`` and ``eyvan gcp connect`` regenerate fleet YAML from their own
    profile and know nothing of the script; without this every re-apply would
    silently delete it. A YAML that carries its own managed entry wins, and the
    dashboard (or the init-script API) is how one is removed.
    """

    if previous is None or any(fleet_init_script.is_entry(item) for item in pool.runtime.init_commands):
        return pool
    script = fleet_init_script.extract(previous.runtime.init_commands)
    if script is None:
        return pool
    runtime = pool.runtime.model_copy(
        update={"init_commands": fleet_init_script.with_script(pool.runtime.init_commands, script)}
    )
    return pool.model_copy(update={"runtime": runtime})


class FleetsMixin:
    """Transactional fleets operations."""

    async def upsert_capacity_pool(
        self,
        principal: Principal,
        pool: CapacityPoolRecord,
        *,
        idempotency_key: str,
    ) -> tuple[CapacityPoolRecord, bool]:
        """Create or update one declarative autoscaling node type."""

        key = _validate_idempotency_key(idempotency_key)
        fingerprint = _fingerprint(
            pool.model_copy(update={"generation": 1, "created_at": datetime.min, "updated_at": datetime.min})
        )
        async with self.repository.transaction() as tx:
            existing_key = await self._idempotent_resource(tx, "fleet", "capacity-pool.upsert", key, fingerprint)
            replayed: CapacityPoolRecord | None = None
            if existing_key is not None:
                replayed = await tx.get_capacity_pool(existing_key.resource_id)
                if replayed is None:
                    raise ConflictError("idempotency record points to a missing capacity pool")
            existing = await tx.get_capacity_pool(pool.id)
            # Authorized inside the transaction, against the stored owner rather
            # than the one the request asserts: reading it outside would leave a
            # window in which a pool changes hands between the check and the write.
            self._require_capacity_administration(
                principal,
                declared_owner=pool.owner_org_id,
                existing_owner=None if existing is None else existing.owner_org_id,
            )
            if existing is not None and pool.owner_org_id != existing.owner_org_id:
                # Reassignment is a fleet act even for a fleet admin coming
                # through this route: it changes who may administer the pool
                # from here on, which deserves its own deliberate call rather
                # than being a side effect of an update that changed max_cells.
                self.authorizer.require(principal, Permission.MANAGE_FLEET)
            if replayed is not None:
                # The replay runs the same ownership gate as the original call:
                # the pool may have changed hands or the caller's grant may
                # have been revoked since, and a replay must not become an
                # authorization-skipping read of the current configuration.
                return replayed, True
            # A pool joining (or editing) a shared-quota group must agree with
            # the group it lands in. The planner spends the *minimum* member
            # limit, so a silently disagreeing upsert would tighten the whole
            # group's budget without telling anyone; rejecting it here turns
            # that into a loud, per-pool configuration error instead.
            if stored_quota_group := pool.quota_group:
                peers = [
                    candidate
                    for candidate in await tx.list_capacity_pools(fleet_name=pool.fleet_name)
                    if candidate.quota_group == stored_quota_group and candidate.id != pool.id
                ]
                try:
                    validate_capacity_pool_quota_groups([*peers, pool])
                except ValueError as exc:
                    raise InputError(str(exc)) from exc
            now = utcnow()
            stored = pool.model_copy(
                update={
                    "generation": 1 if existing is None else existing.generation + 1,
                    "created_at": now if existing is None else existing.created_at,
                    "updated_at": now,
                }
            )
            await tx.put_capacity_pool(stored)
            await tx.put_idempotency(
                IdempotencyRecord(
                    scope="fleet",
                    operation="capacity-pool.upsert",
                    key=key,
                    fingerprint=fingerprint,
                    resource_type="capacity_pool",
                    resource_id=stored.id,
                )
            )
            await tx.append_event(
                AuditEvent(
                    id=_new_id("evt"),
                    actor=principal.subject,
                    entity_type="capacity_pool",
                    entity_id=stored.id,
                    action="capacity_pool.created" if existing is None else "capacity_pool.updated",
                    payload={
                        "fleet_name": stored.fleet_name,
                        "node_type": stored.node_type,
                        "generation": stored.generation,
                    },
                )
            )
            return stored, False

    async def apply_fleet(
        self,
        principal: Principal,
        fleet_name: str,
        pools: tp.Sequence[CapacityPoolRecord],
        *,
        idempotency_key: str,
    ) -> tuple[list[CapacityPoolRecord], bool]:
        """Atomically apply every node type in one declarative fleet.

        Fleet administration only, deliberately, even though single-pool
        mutations now accept an owning organization. This is a *declarative*
        apply: pools absent from the request are disabled, so an organization
        admin running it would take down every pool in the fleet they do not
        own. Per-pool ``upsert``/``disable`` are the ownership-scoped surface.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        if not fleet_name or fleet_name != fleet_name.strip():
            raise InputError("fleet name must be non-empty and trimmed")
        if not pools:
            raise InputError("fleet configuration must contain at least one capacity pool")
        if any(pool.fleet_name != fleet_name for pool in pools):
            raise InputError("capacity pool fleet_name must match the fleet URL")
        if len({pool.id for pool in pools}) != len(pools):
            raise InputError("fleet configuration contains duplicate capacity pool IDs")
        try:
            validate_capacity_pool_quota_groups(pools)
        except ValueError as exc:
            raise InputError(str(exc)) from exc
        key = _validate_idempotency_key(idempotency_key)
        normalized = [
            {**_fleet_pool_declaration(pool), "enabled": pool.enabled}
            for pool in sorted(pools, key=lambda item: item.id)
        ]
        fingerprint = _fingerprint({"fleet_name": fleet_name, "pools": normalized})
        async with self.repository.transaction() as tx:
            existing_key = await self._idempotent_resource(tx, "fleet", "fleet.apply", key, fingerprint)
            if existing_key is not None:
                return await tx.list_capacity_pools(fleet_name=fleet_name), True
            existing = {pool.id: pool for pool in await tx.list_capacity_pools(fleet_name=fleet_name)}
            # Pool IDs are global persistence keys, not fleet-local names. Check
            # all supplied IDs before writes or pruning so an apply cannot move
            # another fleet's template or reset its fencing generation.
            for pool in pools:
                globally_stored = await tx.get_capacity_pool(pool.id)
                if globally_stored is not None and globally_stored.fleet_name != fleet_name:
                    raise ConflictError(
                        f"capacity pool {pool.id!r} already belongs to fleet {globally_stored.fleet_name!r}"
                    )
            now = utcnow()
            supplied_ids = {pool.id for pool in pools}
            pools = [_carry_init_script(pool, existing.get(pool.id)) for pool in pools]
            for pool in pools:
                previous = existing.get(pool.id)
                # An apply that declares exactly what is already stored changes
                # nothing and must bump nothing. The generation is what cells are
                # fenced against, so bumping it invalidates every cell of the pool
                # -- including requests still in flight at the provider, which are
                # then drained and re-launched. Re-applying an unchanged fleet
                # therefore threw away in-flight capacity and asked for it again:
                # twenty cells in one zone across nineteen hours, one per apply,
                # each granted and released without ever becoming READY.
                #
                # Applying a fleet enables its supplied pools. A disabled stored
                # pool is therefore a change even when its other fields match.
                # Runtime clocks are not declarations and cannot invalidate cells
                # or re-arm one-time initialization on an unchanged apply.
                if (
                    previous is not None
                    and previous.enabled
                    and _fleet_pool_declaration(previous) == _fleet_pool_declaration(pool)
                ):
                    continue
                if (
                    previous is not None
                    and previous.enabled
                    and previous.chip_sharing != pool.chip_sharing
                    and _fleet_pool_declaration(previous.model_copy(update={"chip_sharing": pool.chip_sharing}))
                    == _fleet_pool_declaration(pool)
                ):
                    # Chip sharing is a scheduling policy, not a different
                    # machine: flip it on the pool and its live cells in place.
                    # Bumping the generation would fence and replace every cell
                    # (an attached TPU included) just to change who may share it.
                    await tx.put_capacity_pool(
                        previous.model_copy(update={"chip_sharing": pool.chip_sharing, "updated_at": now})
                    )
                    for cell in await tx.list_cells():
                        if cell.pool_id == pool.id and cell.state is not CellState.DELETED:
                            await tx.put_cell(
                                cell.model_copy(update={"chip_sharing": pool.chip_sharing, "updated_at": now})
                            )
                    continue
                await tx.put_capacity_pool(
                    pool.model_copy(
                        update={
                            "enabled": True,
                            "initialized_at": None,
                            "generation": 1 if previous is None else previous.generation + 1,
                            "created_at": now if previous is None else previous.created_at,
                            "updated_at": now,
                        }
                    )
                )
            for pool_id, previous in existing.items():
                if pool_id in supplied_ids or not previous.enabled:
                    continue
                await tx.put_capacity_pool(
                    previous.model_copy(
                        update={
                            "enabled": False,
                            "min_cells": 0,
                            "max_cells": 0,
                            # Zeroed with the others, and not an afterthought:
                            # the record requires min <= initial <= max, so a
                            # pool that declared a floor kept `initial_cells` at
                            # 1 and became invalid the moment anything
                            # re-validated it. `model_copy` does not re-run
                            # validators, so the bad record was written silently
                            # and surfaced later as a 422 on an unrelated call --
                            # with the fleet still enabled and still billing.
                            "initial_cells": 0,
                            "generation": previous.generation + 1,
                            "updated_at": now,
                        }
                    )
                )
            await tx.put_idempotency(
                IdempotencyRecord(
                    scope="fleet",
                    operation="fleet.apply",
                    key=key,
                    fingerprint=fingerprint,
                    resource_type="fleet",
                    resource_id=fleet_name,
                )
            )
            await tx.append_event(
                AuditEvent(
                    id=_new_id("evt"),
                    actor=principal.subject,
                    entity_type="fleet",
                    entity_id=fleet_name,
                    action="fleet.applied",
                    payload={"capacity_pools": sorted(supplied_ids)},
                )
            )
            return await tx.list_capacity_pools(fleet_name=fleet_name), False

    async def get_fleet_init_script(self, principal: Principal, fleet_name: str) -> JsonObject:
        """Return a fleet's dashboard-managed init script and whether its pools can run one."""

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            pools = await tx.list_capacity_pools(fleet_name=fleet_name)
        if not pools:
            raise NotFoundError(f"fleet {fleet_name!r} was not found")
        scripts = {fleet_init_script.extract(pool.runtime.init_commands) for pool in pools}
        unsupported = [reason for pool in pools if (reason := init_script_unsupported_reason(pool))]
        script = next((item for item in sorted(scripts, key=lambda item: item or "") if item), None)
        return {
            "fleet": fleet_name,
            "script": script or "",
            # Pools edited by hand (YAML) can disagree; the dashboard says so.
            "consistent": len(scripts) == 1,
            "supported": not unsupported,
            "unsupported_reasons": unsupported,
            "other_init_commands": sorted(
                {
                    command
                    for pool in pools
                    for command in pool.runtime.init_commands
                    if not fleet_init_script.is_entry(command)
                }
            ),
            "pools": [pool.id for pool in pools],
            "max_bytes": fleet_init_script.MAX_SCRIPT_BYTES,
        }

    async def set_fleet_init_script(
        self,
        principal: Principal,
        fleet_name: str,
        script: str,
        *,
        idempotency_key: str,
    ) -> tuple[JsonObject, bool]:
        """Replace (or with an empty script, remove) the init script on every pool of a fleet.

        The script becomes one marked ``runtime.init_commands`` entry. A changed
        runtime is a changed cell template, so the pool generation advances and
        existing cells are replaced as the autoscaler retires them -- idle cells
        first; a cell holding a lease is never drained for this. New cells run
        the script before their worker starts.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        try:
            normalized = fleet_init_script.normalize(script)
        except fleet_init_script.InitScriptError as exc:
            raise InputError(str(exc)) from exc
        key = _validate_idempotency_key(idempotency_key)
        fingerprint = _fingerprint({"fleet_name": fleet_name, "script": normalized})
        async with self.repository.transaction() as tx:
            existing_key = await self._idempotent_resource(tx, "fleet", "fleet.init_script", key, fingerprint)
            pools = await tx.list_capacity_pools(fleet_name=fleet_name)
            if not pools:
                raise NotFoundError(f"fleet {fleet_name!r} was not found")
            if existing_key is not None:
                replayed = True
            else:
                replayed = False
                unsupported = [reason for pool in pools if (reason := init_script_unsupported_reason(pool))]
                if unsupported and normalized:
                    raise InputError("; ".join(unsupported))
                now = utcnow()
                changed: list[str] = []
                for pool in pools:
                    commands = fleet_init_script.with_script(pool.runtime.init_commands, normalized or None)
                    if commands == pool.runtime.init_commands:
                        continue
                    runtime = pool.runtime.model_copy(update={"init_commands": commands})
                    try:
                        # Re-validated: model_copy would skip the runtime's own checks.
                        stored = CapacityPoolRecord.model_validate(
                            pool.model_copy(
                                update={
                                    "runtime": runtime,
                                    "generation": pool.generation + 1,
                                    "initialized_at": None,
                                    "updated_at": now,
                                }
                            ).model_dump()
                        )
                    except ValueError as exc:
                        raise InputError(f"pool {pool.id!r} rejects this init script: {exc}") from exc
                    await tx.put_capacity_pool(stored)
                    changed.append(pool.id)
                await tx.put_idempotency(
                    IdempotencyRecord(
                        scope="fleet",
                        operation="fleet.init_script",
                        key=key,
                        fingerprint=fingerprint,
                        resource_type="fleet",
                        resource_id=fleet_name,
                    )
                )
                await tx.append_event(
                    AuditEvent(
                        id=_new_id("evt"),
                        actor=principal.subject,
                        entity_type="fleet",
                        entity_id=fleet_name,
                        action="fleet.init_script_set" if normalized else "fleet.init_script_removed",
                        # A digest, not the script: setup scripts often carry tokens.
                        payload={
                            "capacity_pools": changed,
                            "sha256_16": fleet_init_script.digest(normalized),
                            "bytes": len(normalized.encode("utf-8")),
                        },
                    )
                )
        return await self.get_fleet_init_script(principal, fleet_name), replayed

    async def list_submit_fleets(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
    ) -> list[str]:
        """Return sorted runnable fleet names without exposing pool topology."""

        self.authorizer.require(principal, Permission.SUBMIT, org_id=org_id, project_id=project_id)
        async with self.repository.transaction() as tx:
            return sorted(
                {pool.fleet_name for pool in await tx.list_capacity_pools() if pool.enabled and pool.max_cells > 0}
            )

    async def list_capacity_pools(
        self,
        principal: Principal,
        *,
        fleet_name: str | None = None,
    ) -> list[CapacityPoolRecord]:
        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            return await tx.list_capacity_pools(fleet_name=fleet_name)

    async def get_capacity_pool(self, principal: Principal, *, pool_id: str) -> CapacityPoolRecord:
        """Return one capacity pool record."""

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            pool = await tx.get_capacity_pool(pool_id)
            if pool is None:
                raise NotFoundError(f"capacity pool {pool_id!r} was not found")
            return pool

    async def list_capacity(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        accelerator_type: str | None = None,
        capacity_class: CapacityClass | None = None,
    ) -> list[CapacityPoolRecord]:
        """Return enabled pools a caller may submit to, optionally filtered by shape."""

        self.authorizer.require(principal, Permission.SUBMIT, org_id=org_id, project_id=project_id)
        async with self.repository.transaction() as tx:
            pools = await tx.list_capacity_pools()
        result: list[CapacityPoolRecord] = []
        for pool in pools:
            if not pool.enabled or pool.max_cells <= 0:
                continue
            if accelerator_type is not None and pool.accelerator_type != accelerator_type:
                continue
            if capacity_class is not None and pool.capacity_class != capacity_class:
                continue
            result.append(pool)
        return result

    async def disable_capacity_pool(
        self,
        principal: Principal,
        pool_id: str,
        *,
        idempotency_key: str,
    ) -> tuple[CapacityPoolRecord, bool]:
        """Disable a pool so its managed cells drain to zero."""

        key = _validate_idempotency_key(idempotency_key)
        fingerprint = _fingerprint({"pool_id": pool_id, "enabled": False})
        async with self.repository.transaction() as tx:
            existing_key = await self._idempotent_resource(tx, "fleet", "capacity-pool.disable", key, fingerprint)
            pool = await tx.get_capacity_pool(pool_id)
            if pool is None:
                raise NotFoundError(f"capacity pool {pool_id!r} was not found")
            self._require_capacity_administration(
                principal,
                declared_owner=pool.owner_org_id,
                existing_owner=pool.owner_org_id,
            )
            if existing_key is not None:
                return pool, True
            stored = pool.model_copy(
                update={
                    "enabled": False,
                    "min_cells": 0,
                    "max_cells": 0,
                    # Same reason as the prune in ``apply_fleet`` and the sweep in
                    # ``disable_fleet``: the record requires min <= initial <= max,
                    # so a pool that declared a floor kept ``initial_cells`` above
                    # the new max of 0 and became invalid on re-read.
                    "initial_cells": 0,
                    "generation": pool.generation + 1,
                    "updated_at": utcnow(),
                }
            )
            await tx.put_capacity_pool(stored)
            await tx.put_idempotency(
                IdempotencyRecord(
                    scope="fleet",
                    operation="capacity-pool.disable",
                    key=key,
                    fingerprint=fingerprint,
                    resource_type="capacity_pool",
                    resource_id=stored.id,
                )
            )
            await tx.append_event(
                AuditEvent(
                    id=_new_id("evt"),
                    actor=principal.subject,
                    entity_type="capacity_pool",
                    entity_id=stored.id,
                    action="capacity_pool.disabled",
                    payload={"fleet_name": stored.fleet_name, "node_type": stored.node_type},
                )
            )
            return stored, False

    async def disable_fleet(
        self,
        principal: Principal,
        fleet_name: str,
        *,
        idempotency_key: str,
    ) -> tuple[list[CapacityPoolRecord], bool]:
        """Disable every pool in a fleet so the autoscaler drains it to zero."""

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        key = _validate_idempotency_key(idempotency_key)
        fingerprint = _fingerprint({"fleet_name": fleet_name, "enabled": False})
        async with self.repository.transaction() as tx:
            existing_key = await self._idempotent_resource(tx, "fleet", "fleet.disable", key, fingerprint)
            pools = await tx.list_capacity_pools(fleet_name=fleet_name)
            if not pools:
                raise NotFoundError(f"fleet {fleet_name!r} was not found")
            if existing_key is not None:
                return pools, True
            now = utcnow()
            for pool in pools:
                if not pool.enabled and pool.max_cells == 0:
                    continue
                await tx.put_capacity_pool(
                    pool.model_copy(
                        update={
                            "enabled": False,
                            "min_cells": 0,
                            "max_cells": 0,
                            # Same reason as the prune in ``apply_fleet``: the
                            # record requires min <= initial <= max, so leaving a
                            # floor's ``initial_cells`` at 1 wrote a pool that is
                            # invalid on re-read. ``eyvan down`` then failed with
                            # a 422 naming a field the operator never set, and
                            # the fleet stayed enabled and billing.
                            "initial_cells": 0,
                            "generation": pool.generation + 1,
                            "updated_at": now,
                        }
                    )
                )
            await tx.put_idempotency(
                IdempotencyRecord(
                    scope="fleet",
                    operation="fleet.disable",
                    key=key,
                    fingerprint=fingerprint,
                    resource_type="fleet",
                    resource_id=fleet_name,
                )
            )
            await tx.append_event(
                AuditEvent(
                    id=_new_id("evt"),
                    actor=principal.subject,
                    entity_type="fleet",
                    entity_id=fleet_name,
                    action="fleet.disabled",
                )
            )
            return await tx.list_capacity_pools(fleet_name=fleet_name), False

    async def list_cells(self, principal: Principal) -> list[CellRecord]:
        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            return await tx.list_cells()

    async def get_cell(self, principal: Principal, cell_id: str) -> CellRecord:
        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            cell = await tx.get_cell(cell_id)
            if cell is None:
                raise NotFoundError(f"cell {cell_id!r} was not found")
            return cell

    async def drain_cell(
        self,
        principal: Principal,
        cell_id: str,
        *,
        idempotency_key: str,
        expected_workload: WorkloadRecord | None = None,
        expected_cell_generation: int | None = None,
        expected_pool_generation: int | None = None,
        require_empty: bool = False,
    ) -> tuple[CellRecord, bool]:
        """Stop new placement and ask the cell agent to drain capacity."""

        return await self._change_cell_state(
            principal,
            cell_id,
            target=CellState.DRAINING,
            topic="capacity.drain",
            operation="cell.drain",
            idempotency_key=idempotency_key,
            expected_workload=expected_workload,
            expected_cell_generation=expected_cell_generation,
            expected_pool_generation=expected_pool_generation,
            require_empty=require_empty,
        )

    async def delete_cell(
        self,
        principal: Principal,
        cell_id: str,
        *,
        idempotency_key: str,
        expected_cell_generation: int | None = None,
        expected_pool_generation: int | None = None,
    ) -> tuple[CellRecord, bool]:
        """Release provider capacity after every lease has left the cell."""

        return await self._change_cell_state(
            principal,
            cell_id,
            target=CellState.DELETING,
            topic="capacity.release",
            operation="cell.delete",
            idempotency_key=idempotency_key,
            require_empty=True,
            bump_generation=True,
            expected_cell_generation=expected_cell_generation,
            expected_pool_generation=expected_pool_generation,
        )

    async def _change_cell_state(
        self,
        principal: Principal,
        cell_id: str,
        *,
        target: CellState,
        topic: str,
        operation: str,
        idempotency_key: str,
        require_empty: bool = False,
        bump_generation: bool = False,
        expected_workload: WorkloadRecord | None = None,
        expected_cell_generation: int | None = None,
        expected_pool_generation: int | None = None,
    ) -> tuple[CellRecord, bool]:
        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        key = _validate_idempotency_key(idempotency_key)
        declaration: JsonObject = {"cell_id": cell_id, "target": target.value}
        if expected_workload is not None:
            declaration["expected_workload"] = expected_workload.model_dump(mode="json")
        if expected_cell_generation is not None:
            declaration["expected_cell_generation"] = expected_cell_generation
        if expected_pool_generation is not None:
            declaration["expected_pool_generation"] = expected_pool_generation
        if require_empty and operation == "cell.drain":
            declaration["require_empty"] = True
        fingerprint = _fingerprint(declaration)
        async with self.repository.transaction() as tx:
            existing_key = await self._idempotent_resource(tx, "fleet", operation, key, fingerprint)
            if expected_workload is not None:
                current_workload = await tx.get_workload(expected_workload.id)
                if current_workload is None or any(
                    getattr(current_workload, name) != getattr(expected_workload, name)
                    for name in ("generation", "state", "lease_id", "lease_ids", "remote_id")
                ):
                    raise ConflictError("execution ownership changed before cell drain")
            cell = await tx.get_cell(cell_id)
            if cell is None:
                raise NotFoundError(f"cell {cell_id!r} was not found")
            if expected_cell_generation is not None and cell.generation != expected_cell_generation:
                raise ConflictError("cell generation changed before capacity mutation")
            if expected_pool_generation is not None:
                pool = await tx.get_capacity_pool(cell.pool_id) if cell.pool_id else None
                if pool is None or pool.generation != expected_pool_generation:
                    raise ConflictError("capacity pool changed before autoscale mutation")
            if expected_workload is not None:
                owned_ids = set(expected_workload.lease_ids or ())
                if expected_workload.lease_id is not None:
                    owned_ids.add(expected_workload.lease_id)
                owned = False
                for lease_id in owned_ids:
                    lease = await tx.get_lease(lease_id)
                    if (
                        lease is not None
                        and lease.cell_id == cell_id
                        and lease.workload_id == expected_workload.id
                        and lease.state not in {LeaseState.RELEASED, LeaseState.EXPIRED}
                    ):
                        owned = True
                if not owned or cell.state is not CellState.READY:
                    raise ConflictError("execution no longer owns ready capacity")
            if existing_key is not None:
                return cell, True
            if require_empty:
                active = [
                    lease
                    for lease in await tx.list_leases()
                    if lease.cell_id == cell_id and lease.state not in {LeaseState.RELEASED, LeaseState.EXPIRED}
                ]
                if active:
                    raise ConflictError(f"cell {cell_id!r} still owns {len(active)} active lease(s)")
            stored = cell.model_copy(
                update={
                    "state": target,
                    "generation": cell.generation + int(bump_generation),
                    "updated_at": utcnow(),
                }
            )
            await tx.put_cell(stored)
            await self._outbox(
                tx,
                topic=topic,
                aggregate_type="cell",
                aggregate_id=stored.id,
                idempotency_key=f"{operation}:{stored.id}:{stored.generation}",
                generation=stored.generation,
                payload={"cell_id": stored.id},
            )
            await tx.put_idempotency(
                IdempotencyRecord(
                    scope="fleet",
                    operation=operation,
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
                    action=topic,
                    previous_state=cell.state.value,
                    new_state=target.value,
                )
            )
            return stored, False
