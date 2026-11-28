# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fixed-host capacity for machines Eyvan does not provision.

A manual cell is a set of hosts an operator already owns -- on-premises
machines, a reserved lab rack, a long-lived VM -- that already run the Eyvan
native worker. The driver provisions nothing and deletes nothing: ``apply`` and
``reconcile`` record and re-assert the declared hosts, and ``release`` forgets
them. Everything the operator gets is therefore the control plane's:
admission, fair share, budgets, leases, fencing, and audit.

This is deliberately the *cheapest* second capacity backend. It exercises the
whole ``CapacityDriver`` contract against something that is not GCP and needs no
cloud credentials, which is what makes it an honest test of whether those
contracts generalize.

Because it declares the ``eyvan-native`` executor, its observations must carry
the metadata that execution resolution requires: ``management_uid`` and
``provider_generation`` for per-cell worker-token derivation, plus either
``native_worker_endpoints`` (multi-host) or a single execution endpoint.
"""

from __future__ import annotations

import typing as tp
from datetime import datetime
from threading import RLock

from eyvan._internal.compat import UTC
from eyvan._internal.identifiers import stable_id
from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import (
    CELL_ACCELERATOR_METADATA_KEY,
    CellCapabilities,
    CellObservation,
    MutationContext,
    StaleMutationError,
    describe_cell_accelerator,
)
from eyvan.capacity.fleet_state import (
    FleetCellNotRegisteredError,
    FleetCellRecord,
    FleetStateStore,
    StaleFenceError,
)

_MAX_MANUAL_HOSTS = 512


class ManualCapacityDriver:
    """Adopt operator-declared hosts as one Eyvan capacity cell.

    The driver never starts, stops, or deletes a machine. A drained or released
    cell simply stops being schedulable; the hosts keep running whatever their
    owner put on them.
    """

    provider_name = "manual"
    execution_backend = "eyvan-native"
    accelerator_types = ("cpu", "gpu", "tpu")
    capacity_classes = ("on-demand", "reserved")

    def __init__(self, state: FleetStateStore | None = None, *, native_worker_port: int = 8100):
        self.state = state or FleetStateStore.from_uri(None)
        self.native_worker_port = native_worker_port
        self._mutation_lock = RLock()

    def capabilities(self) -> CellCapabilities:
        """Describe fixed-host capacity.

        ``resource_enforcement`` is ``"backend"``: unlike the local development
        driver, the native worker really does fence each workload into its own
        systemd cgroup on these hosts.

        Spot is absent from ``capacity_classes`` on purpose -- a machine the
        operator owns is never preemptible by a provider, so advertising it
        would let the scheduler make a promise about interruption that has no
        meaning here.
        """

        return CellCapabilities(
            provider=self.provider_name,
            executor=self.execution_backend,
            accelerator_types=self.accelerator_types,
            capacity_classes=self.capacity_classes,
            capacity_backends=("manual",),
            supports_callables=True,
            supports_checkpoint=True,
            # The operator owns these hosts, so a diagnostic beside a stuck job is
            # something they could already do by other means.
            supports_exec=True,
            resource_enforcement="backend",
        )

    def observe(self, cell_id: str) -> CellObservation:
        """Return the durable manual observation, or an absent result."""

        record = self.state.get(cell_id)
        if record is None:
            return CellObservation(
                cell_id=cell_id,
                state=CapacityStatus.ABSENT,
                generation=0,
                observed_at=datetime.now(UTC),
            )
        return self._observation(record)

    def _observation(self, record: FleetCellRecord) -> CellObservation:
        endpoints = record.metadata.get("native_worker_endpoints")
        hosts = tuple(endpoints) if isinstance(endpoints, list) else ()
        metadata: JsonObject = {
            "provider": self.provider_name,
            "capacity_backend": self.provider_name,
            "resource_id": f"manual:{record.name}",
            "desired_state": record.desired_state,
            "provider_generation": record.provider_generation,
            **record.metadata,
        }
        # The operator's declaration is the evidence that these hosts are that
        # shape; the catalog is the authority for what the shape *is*. Eyvan
        # still describes nothing it has no row for, so declaring a type this
        # build does not know leaves the cell exactly as unknown as it is today
        # rather than turning a name into a device count. Derived either way,
        # including the removal: a copy persisted by some other build must not
        # outlive this build's ability to state it.
        accelerator = describe_cell_accelerator(accelerator_type=record.accelerator_type)
        if accelerator is None:
            metadata.pop(CELL_ACCELERATOR_METADATA_KEY, None)
        else:
            metadata[CELL_ACCELERATOR_METADATA_KEY] = accelerator.as_metadata()
        return CellObservation(
            cell_id=record.name,
            state=record.state,
            generation=record.generation,
            execution_endpoint=hosts[0] if len(hosts) == 1 else record.execution_endpoint,
            observed_at=datetime.now(UTC),
            runtime_process_count=len(hosts) if hosts else 1,
            metadata=metadata,
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
            raise ValueError(f"manual capacity field {name!r} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _optional_string(desired: tp.Mapping[str, JsonValue], name: str) -> str | None:
        value = desired.get(name)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"manual capacity field {name!r} must be a non-empty string when set")
        return value.strip()

    @staticmethod
    def _nested_options(desired: tp.Mapping[str, JsonValue]) -> JsonObject:
        """Return the provider-specific options block.

        The control plane nests everything provider-specific under
        ``provider_options`` (see ``_capacity_apply_payload``), so a driver that
        reads its own fields from the top level is unreachable through the
        controller no matter how well it works in isolation. The GCP driver has
        the same helper for the same reason.
        """

        for key in ("provider_config", "provider_options"):
            value = desired.get(key)
            if is_json_object(value):
                return value
        return {}

    def _hosts(self, desired: tp.Mapping[str, JsonValue]) -> tuple[str, ...]:
        # Accept either placement: nested is what the controller sends, top-level
        # is what a direct driver caller (and every existing test) uses.
        raw = desired.get("hosts")
        if raw is None:
            raw = self._nested_options(desired).get("hosts")
        if not isinstance(raw, list) or not raw:
            raise ValueError("manual capacity requires a non-empty 'hosts' list of native worker endpoints")
        if len(raw) > _MAX_MANUAL_HOSTS:
            raise ValueError(f"manual capacity supports at most {_MAX_MANUAL_HOSTS} hosts per cell")
        hosts: list[str] = []
        for entry in raw:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError("every manual capacity host must be a non-empty string")
            host = entry.strip()
            if not host.startswith(("http://", "https://")):
                raise ValueError(f"manual capacity host {host!r} must be an http(s) native worker endpoint")
            hosts.append(host)
        if len(set(hosts)) != len(hosts):
            raise ValueError("manual capacity hosts must not contain duplicates")
        return tuple(hosts)

    def _validate_desired(
        self, desired: tp.Mapping[str, JsonValue]
    ) -> tuple[str, str, str, str | None, tuple[str, ...]]:
        provider = self._required_string(desired, "provider")
        if provider != self.provider_name:
            raise ValueError(f"manual capacity requires provider {self.provider_name!r}, got {provider!r}")
        backend = self._required_string(desired, "capacity_backend")
        if backend != self.provider_name:
            raise ValueError(f"manual capacity requires capacity_backend {self.provider_name!r}, got {backend!r}")
        executor = self._required_string(desired, "executor")
        if executor != self.execution_backend:
            raise ValueError(f"manual capacity requires executor {self.execution_backend!r}, got {executor!r}")
        accelerator_type = self._required_string(desired, "accelerator_type")
        capacity_class = self._required_string(desired, "capacity_class").replace("_", "-")
        if capacity_class not in self.capacity_classes:
            choices = ", ".join(self.capacity_classes)
            raise ValueError(
                f"unsupported manual capacity class {capacity_class!r}; choose one of {choices}. "
                "Operator-owned hosts are never provider-preemptible, so spot is not offered."
            )
        region = self._required_string(desired, "region")
        zone = self._optional_string(desired, "zone")
        return accelerator_type, capacity_class, region, zone, self._hosts(desired)

    def apply_capacity(
        self,
        cell_id: str,
        desired: tp.Mapping[str, JsonValue],
        context: MutationContext,
    ) -> CellObservation:
        """Adopt the declared hosts as one schedulable cell.

        Nothing is provisioned. The cell is reported HEALTHY because the hosts
        are asserted to exist; the agent's own execution readiness probe is what
        decides whether their native workers actually answer, and downgrades the
        cell to EXECUTION_CONNECTING when they do not.
        """

        accelerator_type, capacity_class, region, zone, hosts = self._validate_desired(desired)
        agent_endpoint = self._optional_string(desired, "agent_endpoint")
        with self._mutation_lock, self.state.exclusive():

            def mutate(record: FleetCellRecord) -> None:
                if not self._accept_mutation(record, context):
                    return
                record.zone = zone
                record.accelerator_type = accelerator_type
                record.capacity_class = capacity_class
                record.desired_state = "up"
                record.state = CapacityStatus.HEALTHY
                record.execution_endpoint = hosts[0]
                record.provider_generation = max(record.provider_generation, 1)
                record.metadata["capacity_backend"] = self.provider_name
                record.metadata["region"] = region
                record.metadata["native_worker_endpoints"] = list(hosts)
                # Derived, not random: re-applying the same cell must reproduce
                # the same UID so the per-cell worker token stays stable and
                # already-running workers keep authenticating.
                record.metadata["management_uid"] = stable_id("manual", cell_id)
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
        """Re-assert an owned manual cell without touching its hosts."""

        with self._mutation_lock, self.state.exclusive():

            def mutate(record: FleetCellRecord) -> None:
                if not self._accept_mutation(record, context):
                    return
                record.desired_state = "up"
                record.state = CapacityStatus.HEALTHY
                self._stamp(record, context)

            try:
                record = self.state.mutate_record(cell_id, mutate, generation=context.generation)
            except FleetCellNotRegisteredError:
                return self.observe(cell_id)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
        return self._observation(record)

    def drain(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Stop scheduling onto these hosts while keeping the record for release."""

        with self._mutation_lock, self.state.exclusive():

            def mutate(record: FleetCellRecord) -> None:
                # A superseded controller must not be able to drain: the fence
                # is checked exactly as onprem-ssh checks it.
                if not self._accept_mutation(record, context):
                    return
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
        """Forget the declared hosts. The machines themselves are untouched."""

        with self._mutation_lock, self.state.exclusive():

            def owned(record: FleetCellRecord) -> bool:
                if self._accept_mutation(record, context):
                    self._stamp(record, context)
                if record.generation != context.generation or record.fencing_token != context.fencing_token:
                    raise StaleMutationError(f"cell {cell_id!r} changed ownership before manual release")
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


__all__ = ("ManualCapacityDriver",)
