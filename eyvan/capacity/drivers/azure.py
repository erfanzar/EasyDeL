# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Azure capacity: one virtual machine per cell, fenced by tags.

The third cloud, and the one that tests whether the boundary actually holds. AWS
was written against a protocol the GCP driver had just been cleaned up to
expose; if that cleanup was real, a second one should be shorter and duller. It
is: the same six methods, the same ownership discipline, the same conformance
suite, and no edit anywhere that decides.

Two things are genuinely different from EC2, and both simplify:

**The VM name is the idempotency key.** A ``PUT`` to the same resource path is
the same machine, so there is no client token to derive and no window in which a
retry might build a second one. What EC2 needs a token to answer, ARM answers by
construction.

**Eviction is a policy Eyvan chooses.** A Spot VM defaults to being deallocated
rather than deleted -- it stops, and keeps its disks, its NIC and its name. That
is the worst of both: no compute, still billing, and a machine no scheduler can
place work on. This driver asks for ``Delete``, which is the same decision as
one-time spot requests on EC2 and for the same reason.

**Verified against a fake ARM, not against Azure.** Nothing here has run against
a subscription. "Supported" means it passes the twelve-property conformance
suite against a substrate that behaves as the documented REST API says ARM does.
"""

from __future__ import annotations

import typing as tp
from datetime import datetime
from threading import RLock

from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import (
    CellCapabilities,
    CellObservation,
    MutationContext,
    StaleMutationError,
)
from eyvan.capacity.fleet_state import FleetCellNotRegisteredError, FleetCellRecord, FleetStateStore, StaleFenceError
from eyvan.capacity.providers.azure.api import AzureApi, AzureVirtualMachine, parse_nic_private_ip
from eyvan.capacity.providers.azure.machines import (
    CELL_TAG,
    FENCE_TAG,
    GENERATION_TAG,
    is_live,
    observation_metadata,
    vm_body,
    vm_name,
)

#: Provisioning states in which the machine is up and can be scheduled onto.
_READY_STATES = frozenset({"Succeeded"})

#: States in which it exists and is not ready yet.
_PENDING_STATES = frozenset({"Creating", "Updating"})


class AzureVmCapacityDriver:
    """Provision one Azure virtual machine per Eyvan capacity cell."""

    provider_name = "azure-vm"
    execution_backend = "eyvan-native"
    accelerator_types = ("cpu", "gpu")
    capacity_classes = ("on-demand", "spot", "reserved")

    def __init__(
        self,
        state: FleetStateStore | None = None,
        *,
        api: AzureApi | None = None,
        location: str = "",
    ) -> None:
        if api is None:  # pragma: no cover - constructed with an api in every test
            raise ValueError(
                "the Azure driver requires an ARM client; build one with "
                "HttpAzureApi(subscription_id=..., resource_group=..., location=..., token=...)"
            )
        self.state = state or FleetStateStore.from_uri(None)
        self.api = api
        self.location = location or getattr(api, "location", "")
        self._mutation_lock = RLock()

    def capabilities(self) -> CellCapabilities:
        """Describe VM-backed capacity.

        ``resource_enforcement`` is ``"none"`` for the reason EC2's is: a VM is a
        whole machine and Eyvan does not put a cgroup around the workload on it.
        """

        return CellCapabilities(
            provider=self.provider_name,
            executor=self.execution_backend,
            accelerator_types=self.accelerator_types,
            capacity_classes=self.capacity_classes,
            capacity_backends=("azure-vm",),
            supports_callables=True,
            supports_checkpoint=True,
            # One VM per cell. A scale set would make a multi-machine cell
            # possible; claiming it before the driver builds one would let the
            # durable model accept a gang nothing can allocate.
            max_gang_replicas=1,
            supports_exec=False,
            resource_enforcement="none",
        )

    # ---- durable record helpers, shared in shape with the other drivers -----

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
            raise ValueError(f"azure capacity field {name!r} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _options(desired: tp.Mapping[str, JsonValue]) -> JsonObject:
        for key in ("provider_config", "provider_options"):
            value = desired.get(key)
            if is_json_object(value):
                return value
        return {}

    def _validate_desired(self, desired: tp.Mapping[str, JsonValue]) -> JsonObject:
        """Return the flattened VM document, or refuse it by field name."""

        backend = self._required_string(desired, "capacity_backend").replace("_", "-")
        if backend not in {"azure", "azure-vm"}:
            raise ValueError(f"azure capacity requires capacity_backend 'azure-vm', got {backend!r}")
        executor = self._required_string(desired, "executor")
        if executor != self.execution_backend:
            raise ValueError(f"azure capacity requires executor {self.execution_backend!r}, got {executor!r}")
        capacity_class = self._required_string(desired, "capacity_class").replace("_", "-")
        if capacity_class not in self.capacity_classes:
            choices = ", ".join(self.capacity_classes)
            raise ValueError(f"unsupported azure capacity class {capacity_class!r}; choose one of {choices}")

        options = self._options(desired)
        flattened: JsonObject = {**options, **{key: value for key, value in desired.items() if key not in options}}
        flattened["capacity_class"] = capacity_class.replace("-", "_")
        for name in ("vm_size", "network_interface_id"):
            if not isinstance(flattened.get(name), str) or not str(flattened[name]).strip():
                raise ValueError(f"azure capacity field {name!r} must be a non-empty string")
        return flattened

    # ---- observation --------------------------------------------------------

    def observe(self, cell_id: str) -> CellObservation:
        """Return the cell's live state, reconciled against the durable record."""

        record = self.state.get(cell_id)
        if record is None:
            return CellObservation(
                cell_id=cell_id,
                state=CapacityStatus.ABSENT,
                generation=0,
                observed_at=datetime.now(UTC),
            )
        if self._disowned(record):
            # Hardware carrying this cell's name that its fence disowns. The
            # controller must replace the cell rather than schedule onto it.
            return CellObservation(
                cell_id=record.name,
                state=CapacityStatus.NEEDS_RECREATE,
                generation=record.generation,
                execution_endpoint=None,
                observed_at=datetime.now(UTC),
                metadata={
                    "provider": self.provider_name,
                    "capacity_backend": self.provider_name,
                    "resource_id": f"azure:{record.name}",
                    "detail": "Azure holds a VM tagged for this cell that its fencing token disowns",
                },
            )
        return self._observation(record, self._current_vm(record))

    def _current_vm(self, record: FleetCellRecord) -> AzureVirtualMachine | None:
        """Return the VM this incarnation owns, if ARM still has one."""

        machine = self.api.get_vm(vm_name(record.name))
        if machine is None or not is_live(machine):
            return None
        if machine.tags.get(GENERATION_TAG) != str(record.generation):
            return None
        if record.fencing_token is not None and machine.tags.get(FENCE_TAG) != record.fencing_token:
            return None
        return machine

    def _disowned(self, record: FleetCellRecord) -> list[AzureVirtualMachine]:
        """Return live VMs tagged for this cell that its fence disowns."""

        disowned = []
        for machine in self.api.list_vms():
            if machine.tags.get(CELL_TAG) != record.name or not is_live(machine):
                continue
            same_generation = machine.tags.get(GENERATION_TAG) == str(record.generation)
            same_fence = record.fencing_token is None or machine.tags.get(FENCE_TAG) == record.fencing_token
            if not (same_generation and same_fence):
                disowned.append(machine)
        return disowned

    def _resolved_private_ip(self, machine: AzureVirtualMachine) -> str | None:
        """Return the VM's private address, resolving through its NIC if needed.

        Real ARM VM documents carry interface *references* only -- the address
        lives one GET away on the network interface. A payload that already
        embeds an address short-circuits; a provider that cannot ask changes
        nothing.
        """

        if machine.private_ip:
            return machine.private_ip
        get_nic = getattr(self.api, "get_network_interface", None)
        if not callable(get_nic):
            return None
        for nic_id in machine.nic_ids:
            address = parse_nic_private_ip(get_nic(nic_id))
            if address:
                return address
        return None

    def _observation(self, record: FleetCellRecord, machine: AzureVirtualMachine | None) -> CellObservation:
        metadata: JsonObject = {
            "provider": self.provider_name,
            "capacity_backend": self.provider_name,
            "resource_id": f"azure:{record.name}",
            "desired_state": record.desired_state,
            "provider_generation": record.provider_generation,
            **record.metadata,
        }
        if record.desired_state != "up":
            # Drain has decided. A `Succeeded` VM is exactly what a draining cell
            # has, and deriving the state from it would undo the drain on the
            # next observation.
            return CellObservation(
                cell_id=record.name,
                state=record.state,
                generation=record.generation,
                execution_endpoint=None,
                observed_at=datetime.now(UTC),
                runtime_process_count=1,
                metadata=metadata,
            )

        state = record.state
        endpoint = record.execution_endpoint
        if machine is not None:
            metadata.update(observation_metadata(machine))
            if machine.eviction_reason:
                metadata["preemption_reason"] = machine.eviction_reason
            if machine.provisioning_state in _READY_STATES:
                state = CapacityStatus.HEALTHY
                endpoint = endpoint or self._resolved_private_ip(machine)
            elif machine.provisioning_state in _PENDING_STATES:
                state = CapacityStatus.WAITING
        elif record.state is not CapacityStatus.ABSENT:
            state = CapacityStatus.WAITING
            endpoint = None

        return CellObservation(
            cell_id=record.name,
            state=state,
            generation=record.generation,
            execution_endpoint=endpoint,
            observed_at=datetime.now(UTC),
            runtime_process_count=1,
            metadata=metadata,
        )

    # ---- mutations ----------------------------------------------------------

    def apply_capacity(
        self,
        cell_id: str,
        desired: tp.Mapping[str, JsonValue],
        context: MutationContext,
    ) -> CellObservation:
        """Create this cell's VM, or adopt the one this incarnation owns."""

        document = self._validate_desired(desired)
        with self._mutation_lock, self.state.exclusive():

            def claim(record: FleetCellRecord) -> None:
                if not self._accept_mutation(record, context):
                    return
                record.desired_state = "up"
                record.zone = str(document.get("zone") or "") or None
                record.accelerator_type = str(desired.get("accelerator_type") or "cpu")
                record.capacity_class = str(document["capacity_class"]).replace("_", "-")
                record.metadata["capacity_backend"] = self.provider_name
                record.metadata["region"] = str(document.get("region") or self.location)
                self._stamp(record, context)

            try:
                record = self.state.mutate_or_create_record(cell_id, claim, generation=context.generation)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc

        existing = self._current_vm(record)
        if existing is None:
            # The name is the idempotency key, so this either returns the machine
            # that exists or creates the one that does not. There is no window in
            # which a retry builds a second.
            existing = self.api.create_or_update_vm(
                vm_name(cell_id),
                vm_body(cell_id, document, context, location=str(document.get("region") or self.location)),
            )

        with self._mutation_lock, self.state.exclusive():

            def record_vm(current: FleetCellRecord) -> None:
                current.metadata["azure_vm_name"] = existing.name
                current.provider_generation = max(current.provider_generation, context.generation)
                current.state = (
                    CapacityStatus.HEALTHY if existing.provisioning_state in _READY_STATES else CapacityStatus.WAITING
                )
                if existing.provisioning_state in _READY_STATES:
                    current.execution_endpoint = self._resolved_private_ip(existing)

            try:
                record = self.state.mutate_record(cell_id, record_vm, generation=context.generation)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
        return self._observation(record, existing)

    def reconcile(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Re-read ARM and keep the durable record agreeing with it."""

        record = self.state.get(cell_id)
        if record is None:
            return self.observe(cell_id)
        machine = self._current_vm(record)
        with self._mutation_lock, self.state.exclusive():

            def mutate(current: FleetCellRecord) -> None:
                if not self._accept_mutation(current, context):
                    return
                if machine is None:
                    current.execution_endpoint = None
                    current.metadata.pop("azure_vm_name", None)
                    if current.desired_state == "up":
                        current.state = CapacityStatus.WAITING
                else:
                    current.metadata["azure_vm_name"] = machine.name
                    if machine.provisioning_state in _READY_STATES:
                        current.state = CapacityStatus.HEALTHY
                        current.execution_endpoint = self._resolved_private_ip(machine)
                self._stamp(current, context)

            try:
                record = self.state.mutate_record(cell_id, mutate, generation=context.generation)
            except FleetCellNotRegisteredError:
                return self.observe(cell_id)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
        return self._observation(record, machine)

    def drain(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Stop new placement while leaving the VM for release to delete."""

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
        return self._observation(record, self._current_vm(record))

    def release(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Delete this cell's VM and forget it.

        The fence is checked before anything is deleted, and the substrate is
        freed before the record is tombstoned. The first order stops a stale
        replica destroying a live cell; the second stops a crash between the two
        leaving a VM nothing points at.
        """

        record = self.state.get(cell_id)
        if record is not None:
            if context.generation < record.generation or (
                context.generation == record.generation
                and record.fencing_token is not None
                and record.fencing_token != context.fencing_token
            ):
                raise StaleMutationError(
                    f"generation {context.generation} does not own cell {cell_id!r} "
                    f"generation {record.generation}; refusing to delete its VM"
                )
            for machine in self.api.list_vms():
                if machine.tags.get(CELL_TAG) == cell_id and is_live(machine):
                    self.api.delete_vm(machine.name)

        with self._mutation_lock, self.state.exclusive():

            def owned(current: FleetCellRecord) -> bool:
                if self._accept_mutation(current, context):
                    self._stamp(current, context)
                if current.generation != context.generation or current.fencing_token != context.fencing_token:
                    raise StaleMutationError(f"cell {cell_id!r} changed ownership before release")
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


__all__ = ("AzureVmCapacityDriver",)
