# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Applying, draining, and releasing capacity through the cell agent."""

from __future__ import annotations

import asyncio
import time

from eyvan.capacity import (
    PENDING_CAPACITY_STATUSES,
    READY_CAPACITY_STATUSES,
    CapacityStatus,
)
from eyvan.capacity.drivers.base import provision_outcome
from eyvan.domain import (
    ProvisionOutcome,
)
from eyvan.records import (
    CellRecord,
    CellState,
    OutboxRecord,
    OutboxState,
    utcnow,
)
from eyvan.server.controller._support import (
    _ACTIVE_CAPACITY_APPLY_STATES,
    _CAPACITY_APPLY_PROVIDER_RPC_STARTED_KEY,
    CapacityApplySuperseded,
    StaleOutboxCommand,
    _provider_phase_start,
)
from eyvan.server.persistence.base import RepositorySession
from eyvan.worker.agent.client import (
    CellAgentStaleMutationError,
)
from eyvan.worker.agent.models import (
    CapacityRequest,
    WorkloadMutationRequest,
)


class _ControllerCapacityCommands:
    """Applying, draining, and releasing capacity through the cell agent.

    A concern of :class:`~eyvan.server.controller.EyvanController`; not
    instantiated on its own.
    """

    async def _owned_capacity_apply_command(
        self,
        transaction: RepositorySession,
        command: OutboxRecord,
    ) -> OutboxRecord:
        """Load an exact, unexpired capacity-apply claim owned by this controller."""

        current = await transaction.get_outbox(command.id)
        now = utcnow()
        if (
            current is None
            or current.topic != "capacity.apply"
            or current.aggregate_type != command.aggregate_type
            or current.aggregate_id != command.aggregate_id
            or current.generation != command.generation
            or current.state is not OutboxState.PROCESSING
            or current.claimed_by != self.worker_id
            or current.claimed_until is None
            or current.claimed_until <= now
        ):
            raise StaleOutboxCommand(f"capacity apply command {command.id!r} claim expired or moved to another worker")
        return current

    @staticmethod
    def _capacity_apply_provider_rpc_is_proven_unstarted(command: OutboxRecord) -> bool:
        """Return whether this command reached the durable pre-RPC boundary only."""

        return command.payload.get(_CAPACITY_APPLY_PROVIDER_RPC_STARTED_KEY) is False

    async def _clear_unstarted_capacity_apply_requesting(
        self,
        transaction: RepositorySession,
        command: OutboxRecord,
        cell: CellRecord,
    ) -> CellRecord | None:
        """Clear a local request marker only when no provider RPC could have started."""

        current_command = await self._owned_capacity_apply_command(transaction, command)
        if (
            not self._capacity_apply_provider_rpc_is_proven_unstarted(current_command)
            or cell.generation != command.generation
            or cell.state is not CellState.REGISTERING
            or cell.provider_phase is not CapacityStatus.REQUESTING
        ):
            return None
        cleared = cell.model_copy(
            update={
                "provider_phase": None,
                "provider_phase_started_at": None,
                "updated_at": utcnow(),
            }
        )
        await transaction.put_cell(cleared)
        return cleared

    async def _recover_superseded_capacity_apply_template(
        self,
        transaction: RepositorySession,
        command: OutboxRecord,
        cell: CellRecord,
    ) -> CapacityApplySuperseded | None:
        """Recover a proven pre-RPC marker and return a template supersession."""

        try:
            await self._require_current_pool_template(transaction, cell)
        except CapacityApplySuperseded as superseded:
            await self._clear_unstarted_capacity_apply_requesting(transaction, command, cell)
            return superseded
        return None

    async def _dispatch_capacity(self, command: OutboxRecord) -> None:
        cell = await self._load_cell(command)
        async with self._capacity_operation(cell):
            await self._dispatch_capacity_locked(command)

    async def _dispatch_capacity_locked(self, command: OutboxRecord) -> None:
        cell = await self._load_cell(command)
        if cell.state is CellState.UNAVAILABLE:
            raise CapacityApplySuperseded(f"cell {cell.id!r} is unavailable after a prior capacity apply failure")
        if cell.state is not CellState.REGISTERING:
            raise StaleOutboxCommand(f"cell {cell.id!r} is {cell.state.value}, not registering")
        mutation = self._mutation(command, f"cell:{cell.id}:{cell.generation}")
        desired = {
            key: value
            for key, value in command.payload.items()
            if key != _CAPACITY_APPLY_PROVIDER_RPC_STARTED_KEY and value is not None
        }
        cell = await self._mark_capacity_apply_requesting(command, cell)
        await self._mark_capacity_apply_provider_rpc_started(command, cell)
        started = time.perf_counter()
        try:
            observation = await asyncio.to_thread(
                self._agent_factory(cell).apply_capacity,
                cell.id,
                CapacityRequest(mutation=mutation, desired=desired),
            )
        except CellAgentStaleMutationError:
            raise
        except Exception as exc:
            superseded: CapacityApplySuperseded | None = None
            async with self.repository.transaction() as transaction:
                current = await transaction.get_cell(cell.id)
                if (
                    current is None
                    or current.generation != command.generation
                    or current.state is not CellState.REGISTERING
                ):
                    diagnostic = str(exc).strip() or None
                    superseded = CapacityApplySuperseded(
                        "capacity apply lost ownership while the remote mutation was running",
                        provider_diagnostic=diagnostic,
                    )
                else:
                    template_superseded = await self._recover_superseded_capacity_apply_template(
                        transaction,
                        command,
                        current,
                    )
                    if template_superseded is not None:
                        diagnostic = str(exc).strip() or None
                        superseded = CapacityApplySuperseded(
                            str(template_superseded),
                            provider_diagnostic=diagnostic,
                        )
            if superseded is not None:
                raise superseded from exc
            raise
        remote_state = observation.state
        if remote_state in READY_CAPACITY_STATUSES:
            state = CellState.READY
            availability = self._observe_capacity_belief(
                cell,
                ProvisionOutcome.SUCCESS,
                latency_s=time.perf_counter() - started,
            )
        elif remote_state in PENDING_CAPACITY_STATUSES:
            state = CellState.REGISTERING
            availability = cell.availability
        else:
            state = CellState.UNAVAILABLE
            # A successful observation can still report a provider loss, notably
            # an attached Spot TPU reclaimed between connect and the first apply.
            # Keep the historical fallback for older agents without a typed reason.
            outcome = (
                provision_outcome(observation.failure_code, remote_state)
                if observation.failure_code is not None
                else ProvisionOutcome.QUOTA_ERROR
                if remote_state is CapacityStatus.HALTED_QUOTA
                else ProvisionOutcome.STOCKOUT
            )
            availability = self._observe_capacity_belief(cell, outcome)
        observed_at = utcnow()
        transition: tuple[CellRecord, CellRecord] | None = None
        superseded: CapacityApplySuperseded | None = None
        async with self.repository.transaction() as transaction:
            current = await transaction.get_cell(cell.id)
            if (
                current is not None
                and current.generation == command.generation
                and current.state is CellState.REGISTERING
            ):
                template_superseded = await self._recover_superseded_capacity_apply_template(
                    transaction,
                    command,
                    current,
                )
                if template_superseded is not None:
                    superseded = template_superseded
                else:
                    phase_started_at = _provider_phase_start(current, remote_state, observed_at)
                    updated = current.model_copy(
                        update={
                            "state": state,
                            "provider_phase": remote_state,
                            "provider_failure_code": observation.failure_code,
                            "observed_tpu_vms": observation.observed_tpu_vms,
                            "runtime_process_count": (
                                observation.runtime_process_count
                                or observation.observed_tpu_vms
                                or current.runtime_process_count
                            ),
                            "provider_phase_started_at": phase_started_at,
                            "availability": availability,
                            "billing_started_at": (
                                observed_at
                                if state is CellState.READY and current.billing_started_at is None
                                else current.billing_started_at
                            ),
                            "billing_ticked_at": (
                                observed_at
                                if state is CellState.READY and current.billing_started_at is None
                                else current.billing_ticked_at
                            ),
                            "last_heartbeat_at": observed_at,
                            "updated_at": observed_at,
                        }
                    )
                    await transaction.put_cell(updated)
                    transition = current, updated
            else:
                superseded = CapacityApplySuperseded("capacity apply lost ownership before its observation was recorded")
        if superseded is not None:
            raise superseded
        if transition is not None:
            self._capacity_lifecycle.transition(*transition, reason=observation.detail or None)

    async def _mark_capacity_apply_requesting(self, command: OutboxRecord, cell: CellRecord) -> CellRecord:
        """Persist Eyvan's pre-provider request phase before a slow agent RPC."""

        transition: tuple[CellRecord, CellRecord] | None = None
        superseded: CapacityApplySuperseded | None = None
        async with self.repository.transaction() as transaction:
            current = await transaction.get_cell(cell.id)
            if current is None or current.generation != command.generation or current.state is not CellState.REGISTERING:
                raise CapacityApplySuperseded("capacity apply lost ownership before capacity requesting began")
            superseded = await self._recover_superseded_capacity_apply_template(transaction, command, current)
            if superseded is None:
                current_command = await self._owned_capacity_apply_command(transaction, command)
                if current.provider_phase not in {None, CapacityStatus.REQUESTING}:
                    raise CapacityApplySuperseded(
                        f"cell {current.id!r} entered provider phase "
                        f"{current.provider_phase.value} before capacity apply"
                    )
                if current.provider_phase is CapacityStatus.REQUESTING:
                    return current
                observed_at = utcnow()
                if current_command.payload.get(_CAPACITY_APPLY_PROVIDER_RPC_STARTED_KEY) is True:
                    raise CapacityApplySuperseded(
                        f"capacity apply {command.id!r} may have started a provider RPC before requesting was persisted"
                    )
                payload = dict(current_command.payload)
                payload[_CAPACITY_APPLY_PROVIDER_RPC_STARTED_KEY] = False
                await transaction.put_outbox(
                    current_command.model_copy(
                        update={
                            "payload": payload,
                            "last_error": None,
                            "failure_code": None,
                            "updated_at": observed_at,
                        }
                    )
                )
                updated = current.model_copy(
                    update={
                        "provider_phase": CapacityStatus.REQUESTING,
                        "provider_failure_code": None,
                        "provider_phase_started_at": _provider_phase_start(
                            current,
                            CapacityStatus.REQUESTING,
                            observed_at,
                        ),
                        "updated_at": observed_at,
                    }
                )
                await transaction.put_cell(updated)
                transition = current, updated
        if superseded is not None:
            raise superseded
        if transition is not None:
            self._capacity_lifecycle.transition(*transition, reason="sending capacity request to cell agent")
            return transition[1]
        raise RuntimeError("capacity requesting transition unexpectedly had no persisted cell")

    async def _mark_capacity_apply_provider_rpc_started(self, command: OutboxRecord, cell: CellRecord) -> None:
        """Durably mark the point after which a provider request may have started."""

        superseded: CapacityApplySuperseded | None = None
        async with self.repository.transaction() as transaction:
            current = await transaction.get_cell(cell.id)
            if current is None or current.generation != command.generation or current.state is not CellState.REGISTERING:
                raise CapacityApplySuperseded("capacity apply lost ownership before its provider RPC began")
            superseded = await self._recover_superseded_capacity_apply_template(transaction, command, current)
            if superseded is None:
                if current.provider_phase is not CapacityStatus.REQUESTING:
                    raise CapacityApplySuperseded(
                        f"cell {current.id!r} left capacity requesting before the provider RPC began"
                    )
                current_command = await self._owned_capacity_apply_command(transaction, command)
                if current_command.payload.get(_CAPACITY_APPLY_PROVIDER_RPC_STARTED_KEY) is True:
                    return
                payload = dict(current_command.payload)
                payload[_CAPACITY_APPLY_PROVIDER_RPC_STARTED_KEY] = True
                await transaction.put_outbox(
                    current_command.model_copy(update={"payload": payload, "updated_at": utcnow()})
                )
        if superseded is not None:
            raise superseded

    async def _dispatch_capacity_drain(self, command: OutboxRecord) -> None:
        cell = await self._load_cell(command)
        async with self._capacity_operation(cell):
            await self._dispatch_capacity_drain_locked(command)

    async def _dispatch_capacity_drain_locked(self, command: OutboxRecord) -> None:
        cell = await self._load_cell(command)
        request = WorkloadMutationRequest(mutation=self._mutation(command, f"cell:{cell.id}:{cell.generation}"))
        await asyncio.to_thread(self._agent_factory(cell).drain, cell.id, request)
        async with self.repository.transaction() as transaction:
            current = await transaction.get_cell(cell.id)
            if current is not None and current.generation == command.generation and current.state is CellState.DRAINING:
                await transaction.put_cell(
                    current.model_copy(update={"state": CellState.DRAINING, "updated_at": utcnow()})
                )

    async def _dispatch_capacity_release(self, command: OutboxRecord) -> None:
        cell = await self._load_cell(command)
        async with self._capacity_operation(cell):
            await self._dispatch_capacity_release_locked(command)

    async def _dispatch_capacity_release_locked(self, command: OutboxRecord) -> None:
        cell = await self._load_cell(command)
        request = WorkloadMutationRequest(mutation=self._mutation(command, f"cell:{cell.id}:{cell.generation}"))
        await asyncio.to_thread(self._agent_factory(cell).release, cell.id, request)
        transition: tuple[CellRecord, CellRecord] | None = None
        async with self.repository.transaction() as transaction:
            current = await transaction.get_cell(cell.id)
            if current is not None and current.generation == command.generation and current.state is CellState.DELETING:
                updated = current.model_copy(update={"state": CellState.DELETED, "updated_at": utcnow()})
                await transaction.put_cell(updated)
                transition = current, updated
        if transition is not None:
            self._capacity_lifecycle.transition(*transition, reason="provider capacity released")

    @staticmethod
    def _is_active_capacity_apply(command: OutboxRecord) -> bool:
        """Return whether a capacity apply may still enter or own a provider RPC."""

        return command.topic == "capacity.apply" and command.state in _ACTIVE_CAPACITY_APPLY_STATES
