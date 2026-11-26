# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.server.service._control_plane.shared import (
    CellRecord,
    CellState,
    ConflictError,
    LeaseRecord,
    NotFoundError,
    RepositorySession,
    WorkloadRecord,
    WorkloadState,
    _scope,
    datetime,
)


class TransactionsMixin:
    """Transactional transactions operations."""

    @staticmethod
    def _require_autoscale_request_ownership(cell: CellRecord, workload_id: str) -> None:
        """Reject a lease that would take capacity requested by another workload."""

        owner_id = cell.autoscale_request_workload_id
        if owner_id is not None and owner_id != workload_id:
            raise ConflictError("cell is reserved for a different autoscale workload")

    @classmethod
    async def _clear_autoscale_request_ownership(
        cls,
        tx: RepositorySession,
        cell: CellRecord,
        workload_id: str,
        now: datetime,
    ) -> None:
        """Release a one-job provisioning attribution once a cell becomes shared supply."""

        cls._require_autoscale_request_ownership(cell, workload_id)
        if cell.autoscale_request_workload_id is None:
            return
        await tx.put_cell(
            cell.model_copy(
                update={
                    "autoscale_request_workload_id": None,
                    "updated_at": now,
                }
            )
        )

    async def _require_workload(self, tx: RepositorySession, workload_id: str) -> WorkloadRecord:
        workload = await tx.get_workload(workload_id)
        if workload is None:
            raise NotFoundError(f"workload {workload_id!r} was not found")
        return workload

    @staticmethod
    def _validate_expected_workload(
        workload: WorkloadRecord,
        expected_generation: int | None,
        expected_lease_id: str | None,
        expected_state: WorkloadState | None = None,
    ) -> None:
        if expected_generation is not None and workload.generation != expected_generation:
            raise ConflictError(
                f"workload generation {workload.generation} does not match observed generation {expected_generation}"
            )
        if expected_lease_id is not None and workload.lease_id != expected_lease_id:
            raise ConflictError(
                f"workload lease {workload.lease_id!r} does not match observed lease {expected_lease_id!r}"
            )
        if expected_state is not None and workload.state is not expected_state:
            raise ConflictError(
                f"workload state {workload.state.value} does not match observed state {expected_state.value}"
            )

    @staticmethod
    async def _lock_settlement(tx: RepositorySession, workload_id: str) -> None:
        # PostgreSQL's repository acquires a transaction advisory lock as part of
        # idempotency lookup. The durable lease marker remains the source of truth.
        await tx.get_idempotency("fleet", "workload.settle", workload_id)

    @staticmethod
    async def _lock_admission_scope(
        tx: RepositorySession,
        org_id: str,
        project_id: str | None,
    ) -> None:
        await tx.get_idempotency("fleet", "admission.org", _scope(org_id, ""))
        if project_id is not None:
            await tx.get_idempotency("fleet", "admission.project", _scope(org_id, project_id))

    @staticmethod
    async def _current_leases(tx: RepositorySession, workload: WorkloadRecord) -> tuple[LeaseRecord, ...]:
        """Return the complete active lease group in deterministic rank order."""

        lease_ids = workload.lease_ids or (() if workload.lease_id is None else (workload.lease_id,))
        if not lease_ids:
            return ()
        if workload.lease_id != lease_ids[0]:
            raise ConflictError("workload leader lease does not match its lease group")
        leases = tuple([await tx.get_lease(lease_id) for lease_id in lease_ids])
        if any(lease is None for lease in leases):
            missing = lease_ids[next(index for index, lease in enumerate(leases) if lease is None)]
            raise ConflictError(f"workload references missing lease {missing!r}")
        concrete = tuple(lease for lease in leases if lease is not None)
        for index, lease in enumerate(concrete):
            if (
                lease.workload_id != workload.id
                or lease.org_id != workload.org_id
                or lease.project_id != workload.project_id
            ):
                raise ConflictError("workload current lease has inconsistent ownership")
            if len(concrete) > 1 and lease.group_index != index:
                raise ConflictError("workload lease group has an inconsistent member order")
        if len(concrete) > 1:
            group_ids = {lease.group_id for lease in concrete}
            if len(group_ids) != 1 or None in group_ids:
                raise ConflictError("workload lease group has inconsistent membership")
        return concrete

    @classmethod
    async def _validate_expected_cell_snapshot(
        cls,
        tx: RepositorySession,
        workload: WorkloadRecord,
        expected_generation: int | None,
        expected_state: CellState | None,
        expected_unavailable_reason: str | None,
        expected_cell_id: str | None = None,
    ) -> None:
        if expected_state is None:
            return
        leases = await cls._current_leases(tx, workload)
        if not leases:
            raise ConflictError("workload no longer owns the observed cell lease")
        lease = (
            next((candidate for candidate in leases if candidate.cell_id == expected_cell_id), None)
            if expected_cell_id is not None
            else leases[0]
        )
        if lease is None:
            raise ConflictError("observed cell is not a member of the workload lease group")
        cell = await tx.get_cell(lease.cell_id)
        if cell is None:
            return
        if expected_generation is not None and cell.generation != expected_generation:
            raise ConflictError("execution cell generation changed after the loss observation")
        if cell.state is not expected_state or cell.unavailable_reason != expected_unavailable_reason:
            raise ConflictError("execution cell recovered or changed after the loss observation")
