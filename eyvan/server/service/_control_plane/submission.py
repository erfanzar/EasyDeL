# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.records import TpuChipRequest, WorkloadSetup
from eyvan.server.service._control_plane.shared import (
    DEFAULT_MAX_RETRIES,
    CommandExecution,
    ConflictError,
    InputError,
    NotFoundError,
    Permission,
    Principal,
    PriorityBand,
    WorkloadDefinition,
    WorkloadRecord,
    WorkspaceRef,
    _fingerprint,
    _flex_start_refusal,
    _multislice_refusal,
    _scope,
    _validate_idempotency_key,
    tp,
)


class SubmissionMixin:
    """Transactional submission operations."""

    async def submit_workload(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        spec: WorkloadDefinition,
        idempotency_key: str,
        parent_workload_id: str | None = None,
    ) -> tuple[WorkloadRecord, bool]:
        """Create a workload or replay the resource bound to the same request key.

        ``parent_workload_id`` places the new workload in a job tree. A child
        inherits its parent's scope and competes as an ordinary sibling; the
        relation exists for cancellation cascade and lineage, not scheduling.
        """

        self.authorizer.require(principal, Permission.SUBMIT, org_id=org_id, project_id=project_id)
        self._authorize_callable_spec(spec, org_id=org_id, project_id=project_id)
        key = _validate_idempotency_key(idempotency_key)
        scope = _scope(org_id, project_id)
        # Fingerprint the spec as submitted, before locality fills anything in. A
        # resolved preference depends on fleet history, so folding it into the
        # fingerprint would make the same request hash differently minute to minute
        # and turn every replay into a conflict.
        fingerprint = _fingerprint(
            {
                "org_id": org_id,
                "project_id": project_id,
                "spec": spec.model_dump(mode="json"),
                "parent_workload_id": parent_workload_id,
            }
        )
        spec = await self.resolve_locality_preferences(spec)
        async with self.repository.transaction() as tx:
            return await self._store_workload(
                tx,
                principal,
                org_id=org_id,
                project_id=project_id,
                spec=spec,
                scope=scope,
                operation="workload.submit",
                key=key,
                fingerprint=fingerprint,
                parent_workload_id=parent_workload_id,
            )

    async def submit_pool_workload(
        self,
        principal: Principal,
        *,
        fleet_name: str,
        node_type: str,
        org_id: str,
        project_id: str,
        command: str,
        pods: int = 1,
        zone: str | None = None,
        name: str | None = None,
        environment: tp.Mapping[str, str] | None = None,
        secret_mounts: tp.Mapping[str, str] | None = None,
        metadata: tp.Mapping[str, str] | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_failure_retries: int = 0,
        max_preemption_retries: int = 10,
        workspace: WorkspaceRef | None = None,
        checkpointable: bool = False,
        checkpoint_hook: str | None = None,
        checkpoint_output_uri: str | None = None,
        can_submit_children: bool = False,
        priority_band: PriorityBand = PriorityBand.INTERACTIVE,
        intra_band_priority: int = 0,
        checkpoint_grace_seconds: int = 0,
        estimated_duration_seconds: int = 3600,
        named_assets: tuple[str, ...] = (),
        dependencies: tuple[str, ...] = (),
        tpu_chips: TpuChipRequest | None = None,
        setup: WorkloadSetup | None = None,
        idempotency_key: str,
    ) -> tuple[WorkloadRecord, bool]:
        """Queue one command on a logical fleet node type, deriving all placement intent."""

        self.authorizer.require(principal, Permission.SUBMIT, org_id=org_id, project_id=project_id)
        if tpu_chips is not None and pods != 1:
            raise InputError("a TPU chip request runs on one host; it cannot be combined with pods > 1")
        key = _validate_idempotency_key(idempotency_key)
        scope = _scope(org_id, project_id)
        operation = "workload.submit_pool"
        fingerprint = _fingerprint(
            {
                "fleet_name": fleet_name,
                "node_type": node_type,
                "org_id": org_id,
                "project_id": project_id,
                "command": command,
                # Both are part of what was asked for, so a replay that ignored
                # them would return a one-cell job to a caller who asked for
                # four, or an unpinned job to one who pinned a zone.
                "pods": pods,
                "zone": zone,
                "name": name,
                "environment": dict(environment or {}),
                "secret_mounts": dict(secret_mounts or {}),
                "metadata": dict(metadata or {}),
                "max_retries": max_retries,
                "max_failure_retries": max_failure_retries,
                "max_preemption_retries": max_preemption_retries,
                "checkpointable": checkpointable,
                "checkpoint_hook": checkpoint_hook,
                "checkpoint_output_uri": checkpoint_output_uri,
                "can_submit_children": can_submit_children,
                "priority_band": priority_band.value,
                "intra_band_priority": intra_band_priority,
                "checkpoint_grace_seconds": checkpoint_grace_seconds,
                "estimated_duration_seconds": estimated_duration_seconds,
                "named_assets": list(named_assets),
                # Part of the fingerprint: the same command against a different
                # working tree is a different submission, and a replay that
                # ignored the tree would silently run yesterday's code.
                "workspace": workspace.model_dump(mode="json") if workspace is not None else None,
                # Part of what was asked for: the same command with a different
                # dependency set is a different job, and a replay that ignored
                # them would hand back a run built against yesterday's packages.
                "dependencies": list(dependencies),
                # Only when asked for, so every whole-slice fingerprint is unchanged.
                **({"tpu_chips": tpu_chips.model_dump(mode="json")} if tpu_chips is not None else {}),
                # Likewise: a different setup step is a different job.
                **({"setup": setup.model_dump(mode="json")} if setup is not None else {}),
            }
        )
        async with self.repository.transaction() as tx:
            replayed = await self._replayed_workload(
                tx,
                scope=scope,
                operation=operation,
                key=key,
                fingerprint=fingerprint,
            )
            if replayed is not None:
                return replayed, True

            fleet_pools = await self._fleet_pools(tx, fleet_name)
            # Enabled up front, as on the callable path: the pools a submission can
            # match are the pools capacity offers, so a disabled pool neither
            # creates a spurious ambiguity nor wins the primary slot.
            matching_pools = [pool for pool in fleet_pools if pool.enabled and pool.node_type == node_type]
            if not matching_pools:
                # Fall back to the hardware the caller asked for. A pool is
                # named by its shape unless an operator named it otherwise, so
                # requiring the pool name meant a user had to learn a naming
                # scheme to run anything -- while `ray job submit` and
                # `iris job run --tpu v5litepod-16` ask only for hardware.
                matching_pools = [pool for pool in fleet_pools if pool.enabled and pool.accelerator_type == node_type]
                if not matching_pools:
                    raise NotFoundError(f"fleet node type {fleet_name}/{node_type} was not found")
            if zone is not None:
                # A constraint, so it is applied before alternates are chosen
                # and it removes them: a job pinned to a zone that quietly ran
                # in another one is the failure this flag exists to prevent.
                pinned = [pool for pool in matching_pools if pool.zone == zone]
                if not pinned:
                    available = sorted({pool.zone for pool in matching_pools if pool.zone is not None})
                    detail = f"; it has pools in {', '.join(available)}" if available else ""
                    raise NotFoundError(f"fleet node type {fleet_name}/{node_type} has no pool in zone {zone}{detail}")
                matching_pools = pinned
            alternate_pool_ids: tuple[str, ...] = ()
            alternate_pool_templates: tuple[str, ...] = ()
            if len(matching_pools) > 1:
                # Several pools of one shape is cross-zone capacity, not a
                # misconfiguration, exactly as on the callable path: they are
                # alternates when interchangeable, and ambiguous when placing in
                # either would change the admission width the workload runs on.
                candidates = sorted(matching_pools, key=lambda candidate: candidate.id)
                primary = candidates[0]
                interchangeable = [
                    candidate
                    for candidate in candidates[1:]
                    if candidate.allocatable == primary.allocatable and candidate.topology == primary.topology
                ]
                if not interchangeable:
                    raise ConflictError(
                        f"fleet node type {fleet_name}/{node_type} is ambiguous: its pools differ in admission "
                        "width or topology, so they are not interchangeable. Name one pool directly, or make "
                        "them match to use them as cross-zone alternates."
                    )
                matching_pools = [primary]
                alternate_pool_ids = tuple(candidate.id for candidate in interchangeable)
                alternate_pool_templates = tuple(candidate.template_fingerprint for candidate in interchangeable)
            pool = matching_pools[0]
            if pool.max_cells == 0:
                raise ConflictError(f"fleet node type {fleet_name}/{node_type} has max_workers set to zero")
            if pool.max_replica_resources is not None and pool.max_replica_resources != pool.allocatable:
                raise ConflictError(
                    f"fleet node type {fleet_name}/{node_type} partitions workers with max_replica_resources; "
                    "use the advanced workload submission API"
                )
            if refusal := _flex_start_refusal(
                pool,
                estimated_duration_seconds=estimated_duration_seconds,
                checkpointable=checkpointable,
            ):
                raise ConflictError(f"fleet node type {fleet_name}/{node_type} cannot run this job: {refusal}")
            if pods > 1 and (unsupported := _multislice_refusal(pool)):
                # Refused here, naming the pool, rather than left to the record's
                # own validator: that one raises while the spec is being built,
                # so the caller was told their *request* was malformed when the
                # truth is that this pool cannot host a multislice job at all.
                raise ConflictError(
                    f"fleet node type {fleet_name}/{node_type} cannot run --pods {pods}: {unsupported}. "
                    "Several slices allocated together are a TPU multislice shape."
                )
            if pods > pool.max_cells:
                # The same ceiling the callable path enforces: the planner launches
                # width-``pods`` cells only while ``max_cells`` minus live worker
                # units allows, so a wider job accepted here would sit PENDING
                # forever, holding its place in the queue against work that could
                # actually run.
                raise ConflictError(
                    f"fleet node type {fleet_name}/{node_type} is capped at {pool.max_cells} workers, "
                    f"but the job requested pods={pods}"
                )

            if tpu_chips is not None:
                self._require_chip_sharing_pool(pool, tpu_chips, fleet_name=fleet_name)
                # Chip indexes name hardware on this pool's slice only.
                alternate_pool_ids = ()
                alternate_pool_templates = ()
            spec = self._pinned_pool_workload_spec(
                pool=pool,
                tpu_chips=tpu_chips,
                alternate_pool_ids=alternate_pool_ids,
                alternate_pool_templates=alternate_pool_templates,
                name=name if name is not None else f"{fleet_name}/{node_type}",
                execution=CommandExecution(command=command),
                environment=dict(environment or {}),
                secret_mounts=dict(secret_mounts or {}),
                metadata=dict(metadata or {}),
                max_retries=max_retries,
                max_failure_retries=max_failure_retries,
                max_preemption_retries=max_preemption_retries,
                # The same mapping the callable path uses, deliberately: `pods`
                # is the number of native workers allocated together, so
                # `--pods 2` and ``TpuConfig(pods=2)`` describe one shape. Two
                # spellings of "two slices" that produced different specs would
                # be the worst possible outcome of adding this flag.
                cell_group_count=1,
                worker_count=pods,
                workspace=workspace,
                checkpointable=checkpointable,
                checkpoint_hook=checkpoint_hook,
                checkpoint_output_uri=checkpoint_output_uri,
                can_submit_children=can_submit_children,
                priority_band=priority_band,
                intra_band_priority=intra_band_priority,
                checkpoint_grace_seconds=checkpoint_grace_seconds,
                estimated_duration_seconds=estimated_duration_seconds,
                named_assets=named_assets,
                dependencies=dependencies,
                setup=setup,
            )
            return await self._store_workload(
                tx,
                principal,
                org_id=org_id,
                project_id=project_id,
                spec=spec,
                scope=scope,
                operation=operation,
                key=key,
                fingerprint=fingerprint,
            )
