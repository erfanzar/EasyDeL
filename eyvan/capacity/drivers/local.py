# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fenced single-host capacity driver for Eyvan development agents."""

from __future__ import annotations

import typing as tp
from datetime import datetime
from threading import RLock

from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonValue
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CellCapabilities, CellObservation, MutationContext, StaleMutationError
from eyvan.capacity.fleet_state import FleetCellNotRegisteredError, FleetCellRecord, FleetStateStore, StaleFenceError


class LocalCapacityDriver:
    """Materialize a logical cell on the local agent host for development only.

    The driver deliberately makes no claim about CPU, GPU, TPU, cgroup, or
    container isolation.  It exists so the control plane, managed callable
    transfer, scheduler, and native-local executor can be exercised without
    talking to a cloud provider.
    """

    provider_name = "local"
    execution_backend = "native-local"
    accelerator_types = ("cpu", "gpu", "tpu")
    capacity_classes = ("on-demand", "spot", "reserved")

    def __init__(self, state: FleetStateStore | None = None):
        self.state = state or FleetStateStore.from_uri(None)
        self._mutation_lock = RLock()

    def capabilities(self) -> CellCapabilities:
        """Describe the intentionally unisolated local development cell."""

        return CellCapabilities(
            provider=self.provider_name,
            executor=self.execution_backend,
            accelerator_types=self.accelerator_types,
            capacity_classes=self.capacity_classes,
            capacity_backends=("local",),
            supports_callables=True,
            max_gang_replicas=1,
            # Local development runs on the operator's own machine, where a shell is
            # already available without asking Eyvan for one.
            supports_exec=False,
            resource_enforcement="none",
        )

    def observe(self, cell_id: str) -> CellObservation:
        """Return the durable local cell observation, or an absent result."""

        record = self.state.get(cell_id)
        if record is None:
            return CellObservation(
                cell_id=cell_id,
                state=CapacityStatus.ABSENT,
                generation=0,
                observed_at=datetime.now(UTC),
            )
        return self._observation(record)

    @classmethod
    def _observation(cls, record: FleetCellRecord) -> CellObservation:
        """Return the local cell's observation, deliberately without an accelerator.

        Every other driver publishes ``cell_accelerator`` so a gang can be pinned
        to exact devices. This one does not, and the omission is the honest
        answer rather than an oversight: an operator declaring
        ``accelerator_type: a3-highgpu-8g`` on a development cell is naming the
        fleet they are modelling, not eight H100s in the machine this agent runs
        on, and this driver makes no isolation claim that would let it be asked.
        Absence reads as *unknown* everywhere it is read, which is correct here.
        """

        return CellObservation(
            cell_id=record.name,
            state=record.state,
            generation=record.generation,
            execution_endpoint=record.execution_endpoint,
            observed_at=datetime.now(UTC),
            runtime_process_count=1,
            metadata={
                "provider": cls.provider_name,
                "capacity_backend": cls.provider_name,
                "resource_id": f"local:{record.name}",
                "desired_state": record.desired_state,
                "provider_generation": record.provider_generation,
                **record.metadata,
            },
        )

    @staticmethod
    def _accept_mutation(record: FleetCellRecord, context: MutationContext) -> bool:
        if context.generation < record.generation:
            raise StaleMutationError(
                f"generation {context.generation} is older than cell {record.name!r} generation {record.generation}"
            )
        if (
            context.generation == record.generation
            and record.fencing_token is not None
            and record.fencing_token != context.fencing_token
        ):
            raise StaleMutationError(f"fencing token does not own cell {record.name!r} generation {record.generation}")
        return context.idempotency_key not in record.idempotency_keys

    @staticmethod
    def _stamp(record: FleetCellRecord, context: MutationContext) -> None:
        record.generation = context.generation
        record.fencing_token = context.fencing_token
        record.idempotency_keys = [*record.idempotency_keys, context.idempotency_key][-64:]

    @staticmethod
    def _required_string(desired: tp.Mapping[str, JsonValue], name: str) -> str:
        value = desired.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"local capacity field {name!r} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _optional_string(desired: tp.Mapping[str, JsonValue], name: str) -> str | None:
        value = desired.get(name)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"local capacity field {name!r} must be a non-empty string when set")
        return value.strip()

    def _validate_desired(self, desired: tp.Mapping[str, JsonValue]) -> tuple[str, str, str, str | None]:
        provider = self._required_string(desired, "provider")
        if provider != self.provider_name:
            raise ValueError(f"local capacity requires provider {self.provider_name!r}, got {provider!r}")
        backend = self._required_string(desired, "capacity_backend")
        if backend != self.provider_name:
            raise ValueError(f"local capacity requires capacity_backend {self.provider_name!r}, got {backend!r}")
        executor = self._required_string(desired, "executor")
        if executor != self.execution_backend:
            raise ValueError(f"local capacity requires executor {self.execution_backend!r}, got {executor!r}")
        accelerator_type = self._required_string(desired, "accelerator_type")
        capacity_class = self._required_string(desired, "capacity_class").replace("_", "-")
        if capacity_class not in self.capacity_classes:
            choices = ", ".join(self.capacity_classes)
            raise ValueError(f"unsupported local capacity class {capacity_class!r}; choose one of {choices}")
        region = self._required_string(desired, "region")
        zone = self._optional_string(desired, "zone")
        return accelerator_type, capacity_class, region, zone

    def apply_capacity(
        self,
        cell_id: str,
        desired: tp.Mapping[str, JsonValue],
        context: MutationContext,
    ) -> CellObservation:
        """Declare a healthy logical cell on this agent without cloud provisioning."""

        accelerator_type, capacity_class, region, zone = self._validate_desired(desired)
        project = self._optional_string(desired, "project")
        agent_endpoint = self._optional_string(desired, "agent_endpoint")
        with self._mutation_lock, self.state.exclusive():

            def mutate(record: FleetCellRecord) -> None:
                if not self._accept_mutation(record, context):
                    return
                record.project = project
                record.zone = zone
                record.accelerator_type = accelerator_type
                record.capacity_class = capacity_class
                record.desired_state = "up"
                record.state = CapacityStatus.HEALTHY
                record.execution_endpoint = "local"
                record.provider_generation = max(record.provider_generation, 1)
                record.metadata["capacity_backend"] = self.provider_name
                record.metadata["region"] = region
                if agent_endpoint is None:
                    record.metadata.pop("agent_endpoint", None)
                else:
                    record.metadata["agent_endpoint"] = agent_endpoint
                self._stamp(record, context)

            try:
                record = self.state.mutate_or_create_record(cell_id, mutate, generation=context.generation)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
        return self._observation(record)

    def reconcile(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Keep an owned local cell healthy while its agent remains alive."""

        with self._mutation_lock, self.state.exclusive():

            def mutate(record: FleetCellRecord) -> None:
                if not self._accept_mutation(record, context):
                    return
                record.desired_state = "up"
                record.state = CapacityStatus.HEALTHY
                record.execution_endpoint = "local"
                self._stamp(record, context)

            try:
                record = self.state.mutate_record(cell_id, mutate, generation=context.generation)
            except FleetCellNotRegisteredError:
                return self.observe(cell_id)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
        return self._observation(record)

    def drain(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Fence new placement while retaining the local record for release."""

        with self._mutation_lock, self.state.exclusive():

            def mutate(record: FleetCellRecord) -> None:
                record.desired_state = "down"
                record.state = CapacityStatus.DRAINING
                self._stamp(record, context)

            try:
                record = self.state.drain_record(cell_id, context.generation, context.fencing_token, mutate)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
        if record is None:
            return CellObservation(
                cell_id=cell_id,
                state=CapacityStatus.DRAINING,
                generation=context.generation,
                observed_at=datetime.now(UTC),
            )
        return self._observation(record)

    def release(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Remove a local capacity record after the controller deletes its cell."""

        with self._mutation_lock, self.state.exclusive():

            def owned(record: FleetCellRecord) -> bool:
                if self._accept_mutation(record, context):
                    self._stamp(record, context)
                if record.generation != context.generation or record.fencing_token != context.fencing_token:
                    raise StaleMutationError(f"cell {cell_id!r} changed ownership before local release")
                return True

            try:
                self.state.tombstone_record(cell_id, context.generation, context.fencing_token, owned)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
        return CellObservation(
            cell_id=cell_id,
            state=CapacityStatus.RELEASED,
            generation=context.generation,
            observed_at=datetime.now(UTC),
        )


__all__ = ("LocalCapacityDriver",)
