# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fenced claims on existing Cloud TPU VMs, without cloud mutations."""

from __future__ import annotations

import contextlib
import logging
import typing as tp
from datetime import datetime
from threading import RLock

from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan._internal.network import format_http_ip_endpoint
from eyvan._internal.ssh import SshDestination, SshError, SshTunnelManager
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import (
    CELL_ACCELERATOR_METADATA_KEY,
    CapacityProviderError,
    CapacityProviderFailureCode,
    CellCapabilities,
    CellObservation,
    MutationContext,
    StaleMutationError,
    describe_cell_accelerator,
)
from eyvan.capacity.fleet_state import (
    FleetCellNotRegisteredError,
    FleetCellRecord,
    FleetStateSchemaError,
    FleetStateStore,
    StaleFenceError,
)
from eyvan.capacity.providers.gcp.tpu_nodes import (
    ATTACH_CAPACITY_BACKEND,
    ENDPOINT_MODES,
    OPTION_ENDPOINT_MODE,
    OPTION_NODE_ID,
    OPTION_NODE_NAME,
    OPTION_SSH_ENTRY,
    OPTION_TUNNEL_PORTS,
    OPTION_WORKER_PORT,
    TpuApiError,
    TpuNode,
    TpuNodeApi,
    TpuNodeParseError,
    attach_management_uid,
    validate_node_name,
)
from eyvan.capacity.providers.gcp.tpu_ssh import SshTpuNodeReader, attach_tunnel_spec

if tp.TYPE_CHECKING:
    from eyvan.capacity.providers.gcp.native_worker_ingress import NativeWorkerPublicIngress

logger = logging.getLogger(__name__)


class TpuNodeReader(tp.Protocol):
    """Anything that reads one node with :meth:`TpuNodeApi.get_node`'s contract."""

    def get_node(self, project: str, zone: str, name: str) -> TpuNode | None: ...


def _valid_port(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and 1 <= value <= 65535


_PROVISIONING = frozenset({"CREATING", "STARTING", "RESTARTING", "REPAIRING", "REIMAGING", "UNHIDING"})
_TERMINAL = frozenset({"STOPPED", "STOPPING", "TERMINATED", "DELETING", "HIDING", "HIDDEN"})


class GcpAttachCapacityDriver:
    """Observe one existing TPU slice; drain/release only forget its local claim.

    Provider generation is always one: the CLI derives worker credentials from
    the node's immutable identity before the control plane creates a cell ID.
    A node recreated under the same name must never inherit that credential.
    """

    provider_name = "gcp"
    capacity_kind = "attach"
    execution_backend = "eyvan-native"
    capacity_classes = ("spot", "on-demand", "reserved")

    def __init__(
        self,
        state: FleetStateStore | None = None,
        *,
        api: TpuNodeApi | None = None,
        native_worker_port: int = 8100,
        ssh_reader_factory: tp.Callable[[SshDestination], TpuNodeReader] | None = None,
        tunnels: SshTunnelManager | None = None,
        native_worker_public_ingress: NativeWorkerPublicIngress | None = None,
    ) -> None:
        if not _valid_port(native_worker_port):
            raise ValueError("native_worker_port must be a TCP port between 1 and 65535")
        self.state = state or FleetStateStore.from_uri(None)
        self.api = api or TpuNodeApi()
        self.native_worker_port = native_worker_port
        # A node the operator's own credential cannot list is read through one of
        # its hosts instead; which reader applies is recorded per claim, not fixed.
        self.ssh_reader_factory = ssh_reader_factory or (lambda entry: SshTpuNodeReader(entry))
        self.tunnels = tunnels or SshTunnelManager()
        self._ssh_readers: dict[str, TpuNodeReader] = {}
        self._mutation_lock = RLock()
        # The profile-owned /32 rule `eyvan gcp connect` opened for a public
        # claim. Re-ensured on every reconcile, exactly as the provisioning
        # driver does, so a deleted rule or a changed laptop IP heals itself.
        self.native_worker_public_ingress = native_worker_public_ingress

    def capabilities(self) -> CellCapabilities:
        """Declare native execution on a slice with its real purchase class."""

        return CellCapabilities(
            provider=self.provider_name,
            executor=self.execution_backend,
            accelerator_types=("tpu",),
            capacity_classes=self.capacity_classes,
            capacity_backends=(ATTACH_CAPACITY_BACKEND,),
            supports_callables=True,
            supports_checkpoint=True,
            supports_exec=True,
            max_gang_replicas=1,
            resource_enforcement="none",
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
    def _required_string(document: tp.Mapping[str, JsonValue], name: str) -> str:
        value = document.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"gcp-attach capacity field {name!r} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _nested_options(desired: tp.Mapping[str, JsonValue]) -> JsonObject:
        for key in ("provider_config", "provider_options"):
            value = desired.get(key)
            if isinstance(value, dict) and is_json_object(value):
                return value
        return {}

    @staticmethod
    def _access_options(options: tp.Mapping[str, JsonValue], mode: str) -> JsonObject:
        """Validate how this claim is read (API or SSH) and reached (address or tunnel)."""

        access: JsonObject = {}
        entry = options.get(OPTION_SSH_ENTRY)
        if entry is not None:
            if not isinstance(entry, str):
                raise ValueError("gcp-attach ssh_entry must be an SSH destination string")
            access[OPTION_SSH_ENTRY] = SshDestination.parse(entry).target
        worker_port = options.get(OPTION_WORKER_PORT)
        if worker_port is not None:
            if not _valid_port(worker_port):
                raise ValueError("gcp-attach worker_port must be a TCP port between 1 and 65535")
            access[OPTION_WORKER_PORT] = worker_port
        ports = options.get(OPTION_TUNNEL_PORTS)
        if mode == "tunnel":
            if OPTION_SSH_ENTRY not in access:
                raise ValueError("gcp-attach endpoint_mode tunnel requires ssh_entry")
            if not isinstance(ports, list) or not ports or not all(_valid_port(port) for port in ports):
                raise ValueError("gcp-attach tunnel_ports must be a non-empty list of TCP ports")
            if len(set(tp.cast(list[int], ports))) != len(ports):
                raise ValueError("gcp-attach tunnel_ports must be distinct")
            access[OPTION_TUNNEL_PORTS] = list(ports)
        elif ports is not None:
            raise ValueError("gcp-attach tunnel_ports apply only to endpoint_mode tunnel")
        return access

    def _validate_desired(self, desired: tp.Mapping[str, JsonValue]) -> tuple[str, str, str, str, str, str, str]:
        for field, expected in (
            ("provider", self.provider_name),
            ("capacity_backend", ATTACH_CAPACITY_BACKEND),
            ("executor", self.execution_backend),
        ):
            if self._required_string(desired, field) != expected:
                raise ValueError(f"gcp-attach capacity requires {field} {expected!r}")
        project = self._required_string(desired, "project")
        zone = self._required_string(desired, "zone")
        # Reject path/query injection before constructing the provider resource URL.
        for field, value in (("project", project), ("zone", zone)):
            if not all(character.isascii() and (character.isalnum() or character in "-:") for character in value):
                raise ValueError(f"gcp-attach capacity field {field!r} is not a valid GCP identifier")
        if "-" not in zone:
            raise ValueError("gcp-attach zone must include its region and zone suffix")
        accelerator = self._required_string(desired, "accelerator_type").lower()
        capacity_class = self._required_string(desired, "capacity_class").replace("_", "-")
        if capacity_class not in self.capacity_classes:
            raise ValueError("gcp-attach capacity_class must be spot, on-demand, or reserved")
        options = self._nested_options(desired)
        name = validate_node_name(self._required_string(options, OPTION_NODE_NAME))
        node_id = self._required_string(options, OPTION_NODE_ID)
        mode = self._required_string(options, OPTION_ENDPOINT_MODE)
        if mode not in ENDPOINT_MODES:
            raise ValueError(f"gcp-attach endpoint_mode must be one of {', '.join(ENDPOINT_MODES)}")
        node_count = options.get("node_count", 1)
        if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count != 1:
            raise ValueError("gcp-attach supports a single TPU slice only (node_count=1)")
        workers = desired.get("worker_count", 1)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers != 1:
            raise ValueError("gcp-attach supports a single TPU slice only (worker_count=1)")
        return project, zone, accelerator, capacity_class, name, node_id, mode

    def _reader(self, entry: str | None) -> TpuNodeReader:
        if entry is None:
            return self.api
        reader = self._ssh_readers.get(entry)
        if reader is None:
            reader = self._ssh_readers[entry] = self.ssh_reader_factory(SshDestination.parse(entry))
        return reader

    def _node(self, project: str, zone: str, name: str, entry: str | None = None) -> TpuNode | None:
        try:
            return self._reader(entry).get_node(project, zone, name)
        except TpuApiError as exc:
            # Provider response text is untrusted and can contain credentials.
            diagnostic = "could not read attached TPU node" + (f" over SSH to {entry}" if entry else "")
            if exc.status is not None:
                diagnostic += f" (HTTP {exc.status})"
            raise CapacityProviderError(
                diagnostic,
                code=(
                    CapacityProviderFailureCode.PROVIDER_UNAVAILABLE
                    if exc.retryable
                    else CapacityProviderFailureCode.CONFIGURATION
                ),
                retryable=exc.retryable,
            ) from None
        except TpuNodeParseError:
            raise CapacityProviderError(
                "attached TPU API returned an invalid node document",
                code=CapacityProviderFailureCode.CONFIGURATION,
                retryable=False,
            ) from None

    def _metadata(self, record: FleetCellRecord, node: TpuNode | None = None) -> JsonObject:
        metadata: JsonObject = {
            **record.metadata,
            "provider": self.provider_name,
            "capacity_backend": ATTACH_CAPACITY_BACKEND,
            "capacity_kind": self.capacity_kind,
            "resource_id": f"gcp-attach:{record.project}/{record.zone}/{record.metadata[OPTION_NODE_NAME]}",
            "management_uid": record.metadata["management_uid"],
            "provider_generation": 1,
            "project": record.project,
            "zone": record.zone,
            "region": (record.zone or "").rsplit("-", 1)[0],
            "capacity_class": node.capacity_class.value.replace("_", "-") if node else record.capacity_class,
            "tpu_config": {"node_count": 1, "accelerator_type": record.accelerator_type},
            "native_worker_endpoints": [],
        }
        accelerator = describe_cell_accelerator(accelerator_type=record.accelerator_type)
        if accelerator is None:
            metadata.pop(CELL_ACCELERATOR_METADATA_KEY, None)
        else:
            metadata[CELL_ACCELERATOR_METADATA_KEY] = accelerator.as_metadata()
        return metadata

    def _observation(
        self,
        record: FleetCellRecord,
        node: TpuNode | None = None,
        *,
        state: CapacityStatus | None = None,
        failure_code: CapacityProviderFailureCode | None = None,
        detail: str = "",
        connect_tunnel: bool = True,
    ) -> CellObservation:
        metadata = self._metadata(record, node)
        endpoints: tuple[str, ...] = ()
        if state is None:
            if node is None:
                state = CapacityStatus.NEEDS_RECREATE
                failure_code = (
                    CapacityProviderFailureCode.PREEMPTED
                    if record.capacity_class == "spot"
                    else CapacityProviderFailureCode.PROVIDER_ERROR
                )
                detail = "attached TPU no longer exists; re-run eyvan gcp connect"
            elif node.node_id != record.metadata[OPTION_NODE_ID]:
                state = CapacityStatus.NEEDS_RECREATE
                failure_code = CapacityProviderFailureCode.PROVIDER_ERROR
                detail = "TPU was recreated under the same name; re-run eyvan gcp connect"
            elif (
                node.accelerator_type != record.accelerator_type
                or node.capacity_class.value.replace("_", "-") != record.capacity_class
            ):
                state = CapacityStatus.NEEDS_RECREATE
                failure_code = CapacityProviderFailureCode.CONFIGURATION
                detail = "attached TPU accelerator type or capacity class no longer matches its claim"
            elif node.state == "PREEMPTED" or (node.state == "DELETING" and node.spot):
                state = CapacityStatus.NEEDS_RECREATE
                failure_code = CapacityProviderFailureCode.PREEMPTED
                detail = "attached TPU was reclaimed by the provider"
            elif node.state in _TERMINAL:
                state = CapacityStatus.NEEDS_RECREATE
                detail = f"attached TPU is {node.state}"
            elif node.state in _PROVISIONING:
                state = CapacityStatus.PROVISIONING
            elif node.state == "READY":
                if node.health not in (None, "HEALTHY"):
                    state = CapacityStatus.UNAVAILABLE
                    detail = "attached TPU is not healthy"
                else:
                    addresses = node.endpoint_ips(str(record.metadata[OPTION_ENDPOINT_MODE]))
                    state = CapacityStatus.HEALTHY if addresses else CapacityStatus.PROVISIONING
                    if addresses:
                        endpoints, problem = self._endpoints(record, addresses, connect_tunnel=connect_tunnel)
                        if problem is not None:
                            state, failure_code, detail = problem
            else:
                state = CapacityStatus.UNAVAILABLE
                detail = "attached TPU state is unknown"
        metadata["native_worker_endpoints"] = list(endpoints)
        return CellObservation(
            cell_id=record.name,
            state=state,
            generation=record.generation,
            execution_endpoint=endpoints[0] if endpoints else None,
            detail=detail,
            observed_at=datetime.now(UTC),
            metadata=metadata,
            observed_tpu_vms=len(node.hosts) if node else None,
            runtime_process_count=len(node.hosts) if node else None,
            failure_code=failure_code,
        )

    def _worker_port(self, record: FleetCellRecord) -> int:
        port = record.metadata.get(OPTION_WORKER_PORT)
        return port if isinstance(port, int) and _valid_port(port) else self.native_worker_port

    def _tunnel_spec(self, record: FleetCellRecord, addresses: tuple[str, ...]) -> tp.Any:
        ports = record.metadata.get(OPTION_TUNNEL_PORTS)
        entry = record.metadata.get(OPTION_SSH_ENTRY)
        if not isinstance(ports, list) or not isinstance(entry, str) or len(ports) != len(addresses):
            return None
        return attach_tunnel_spec(
            management_uid=str(record.metadata["management_uid"]),
            entry=SshDestination.parse(entry),
            internal_ips=addresses,
            worker_port=self._worker_port(record),
            local_ports=[int(tp.cast(int, port)) for port in ports],
        )

    def _endpoints(
        self, record: FleetCellRecord, addresses: tuple[str, ...], *, connect_tunnel: bool
    ) -> tuple[tuple[str, ...], tuple[CapacityStatus, CapacityProviderFailureCode, str] | None]:
        """Return worker endpoints, or why none are usable right now."""

        if record.metadata.get(OPTION_ENDPOINT_MODE) != "tunnel":
            port = self._worker_port(record)
            return tuple(format_http_ip_endpoint(address, port) for address in addresses), None
        spec = self._tunnel_spec(record, addresses)
        if spec is None:
            return (), (
                CapacityStatus.UNAVAILABLE,
                CapacityProviderFailureCode.CONFIGURATION,
                "attached TPU host count no longer matches its tunnel ports; re-run eyvan gcp connect",
            )
        if connect_tunnel:
            # The agent keeps the tunnel alive: an observation is exactly the
            # moment it must be up, and a dead one (laptop slept, network moved)
            # is replaced here rather than reported healthy.
            try:
                self.tunnels.ensure(spec)
            except (SshError, OSError, ValueError) as exc:
                return (), (
                    CapacityStatus.UNOBSERVED,
                    CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
                    f"SSH tunnel to the attached TPU is down: {exc}",
                )
        return tuple(f"http://{forward.bind}:{forward.local_port}" for forward in spec.forwards), None

    def _refresh_public_ingress(self, record: FleetCellRecord, node: TpuNode) -> None:
        """Re-ensure the profile's worker ingress rule for a public, API-reached claim.

        Only when the node actually carries the profile tag, which is what
        ``eyvan gcp connect`` added and what the rule targets: a node without it
        is not reached through this rule, and creating one for it would open a
        port nothing asked for. A failure is logged, never raised -- the
        readiness probe is what decides whether the worker is reachable, and
        the next reconcile tries again.
        """

        ingress = self.native_worker_public_ingress
        tag = getattr(ingress, "network_tag", None)
        if (
            ingress is None
            or not isinstance(tag, str)
            or tag not in node.tags
            or record.metadata.get(OPTION_ENDPOINT_MODE) != "public"
            or record.metadata.get(OPTION_SSH_ENTRY)
            or not isinstance(record.project, str)
            or not record.project
        ):
            return
        try:
            ingress.ensure(project=record.project, network=node.network, port=self._worker_port(record))
        except (CapacityProviderError, OSError, ValueError) as exc:
            logger.warning("attached cell %s worker ingress could not be ensured: %s", record.name, exc)

    def _read_observation(self, record: FleetCellRecord, *, refresh_ingress: bool = False) -> CellObservation:
        if record.desired_state == "down":
            return self._observation(record, state=CapacityStatus.DRAINING)
        try:
            node = self._node(
                str(record.project),
                str(record.zone),
                str(record.metadata[OPTION_NODE_NAME]),
                tp.cast("str | None", record.metadata.get(OPTION_SSH_ENTRY)),
            )
        except CapacityProviderError as exc:
            return self._observation(
                record,
                state=CapacityStatus.UNOBSERVED if exc.retryable else CapacityStatus.UNAVAILABLE,
                failure_code=exc.code,
                detail=exc.diagnostic,
            )
        if refresh_ingress and node is not None:
            self._refresh_public_ingress(record, node)
        return self._observation(record, node)

    def observe(self, cell_id: str) -> CellObservation:
        """Read current provider state, never reusing stale execution endpoints."""

        record = self.state.get(cell_id)
        if record is not None:
            return self._read_observation(record)
        tombstone = self.state.cell_tombstone(cell_id)
        metadata = tombstone.get("metadata") if tombstone else None
        generation = tombstone.get("generation") if tombstone else 0
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise FleetStateSchemaError("attached cell tombstone has an invalid generation")
        return CellObservation(
            cell_id=cell_id,
            state=(
                CapacityStatus.DRAINING
                if tombstone and tombstone.get("state") == CapacityStatus.DRAINING
                else CapacityStatus.RELEASED
                if tombstone
                else CapacityStatus.ABSENT
            ),
            generation=generation,
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
            observed_at=datetime.now(UTC),
        )

    def apply_capacity(
        self, cell_id: str, desired: tp.Mapping[str, JsonValue], context: MutationContext
    ) -> CellObservation:
        """Validate the existing node and persist one fenced attach claim."""

        project, zone, accelerator, capacity_class, name, node_id, mode = self._validate_desired(desired)
        access = self._access_options(self._nested_options(desired), mode)
        entry = tp.cast("str | None", access.get(OPTION_SSH_ENTRY))
        with self._mutation_lock, self.state.exclusive():
            existing = self.state.get(cell_id)
            if existing is not None:
                self._accept_mutation(existing, context)
            tombstone = self.state.cell_tombstone(cell_id)
            fenced_generation = tombstone.get("generation") if tombstone else None
            if isinstance(fenced_generation, int) and context.generation <= fenced_generation:
                raise StaleMutationError(f"cell {cell_id!r} was already drained or released at this generation")
            node = self._node(project, zone, name, entry)
            if node is None:
                raise CapacityProviderError(
                    "attached TPU no longer exists; re-run eyvan gcp connect",
                    code=CapacityProviderFailureCode.CONFIGURATION,
                    retryable=False,
                )
            if node.node_id != node_id:
                raise CapacityProviderError(
                    "TPU was recreated under the same name; re-run eyvan gcp connect",
                    code=CapacityProviderFailureCode.CONFIGURATION,
                    retryable=False,
                )
            if node.accelerator_type != accelerator or node.capacity_class.value.replace("_", "-") != capacity_class:
                raise CapacityProviderError(
                    "attached TPU accelerator_type or capacity_class does not match the provider",
                    code=CapacityProviderFailureCode.CONFIGURATION,
                    retryable=False,
                )
            uid = attach_management_uid(project=project, zone=zone, name=name, node_id=node_id)

            def mutate(record: FleetCellRecord) -> None:
                if record.metadata.get("capacity_kind") not in (None, self.capacity_kind):
                    raise ValueError("gcp-attach cannot take over another capacity kind")
                old_uid = record.metadata.get("management_uid")
                if old_uid is not None and old_uid != uid:
                    raise ValueError("gcp-attach cannot retarget an existing cell; release its claim first")
                if not self._accept_mutation(record, context):
                    return
                record.project = project
                record.zone = zone
                record.accelerator_type = accelerator
                record.capacity_class = capacity_class
                record.provider_generation = 1
                record.desired_state = "up"
                record.metadata.update(
                    {
                        "provider": self.provider_name,
                        "capacity_backend": ATTACH_CAPACITY_BACKEND,
                        "capacity_kind": self.capacity_kind,
                        "management_uid": uid,
                        OPTION_NODE_NAME: name,
                        OPTION_NODE_ID: node_id,
                        OPTION_ENDPOINT_MODE: mode,
                    }
                )
                for key in (OPTION_SSH_ENTRY, OPTION_TUNNEL_PORTS, OPTION_WORKER_PORT):
                    if key in access:
                        record.metadata[key] = access[key]
                    else:
                        record.metadata.pop(key, None)
                # No tunnel is started while the fleet-state lock is held.
                record.state = self._observation(record, node, connect_tunnel=False).state
                self._stamp(record, context)

            try:
                record = self.state.mutate_or_create_record(cell_id, mutate, generation=context.generation)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
        return self._observation(record, node)

    def reconcile(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Fence and re-observe a claim; never provision replacement hardware."""

        with self._mutation_lock, self.state.exclusive():

            def mutate(record: FleetCellRecord) -> None:
                if self._accept_mutation(record, context):
                    self._stamp(record, context)

            try:
                record = self.state.mutate_record(cell_id, mutate, generation=context.generation)
            except FleetCellNotRegisteredError:
                return self.observe(cell_id)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
        # Provider reads (an SSH round trip for SSH-reached nodes), ingress and
        # tunnel upkeep run outside the fleet-state lock; the fenced record is
        # already persisted.
        return self._read_observation(record, refresh_ingress=True)

    def drain(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Keep the local claim, but make it unschedulable without provider I/O."""

        with self._mutation_lock, self.state.exclusive():

            def mutate(record: FleetCellRecord) -> None:
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
            return self.observe(cell_id)
        return self._observation(record, state=CapacityStatus.DRAINING)

    def release(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Tombstone the claim; the TPU and its workers remain untouched."""

        with self._mutation_lock, self.state.exclusive():
            previous = self.state.cell_tombstone(cell_id)
            released_metadata: JsonObject = {
                "capacity_kind": self.capacity_kind,
                "metadata": previous.get("metadata", {}) if previous else {},
            }

            def owned(record: FleetCellRecord) -> bool:
                if self._accept_mutation(record, context):
                    self._stamp(record, context)
                if record.generation != context.generation or record.fencing_token != context.fencing_token:
                    raise StaleMutationError(f"cell {cell_id!r} changed ownership before attach release")
                released_metadata["metadata"] = self._metadata(record)
                return True

            try:
                self.state.tombstone_record(
                    cell_id, context.generation, context.fencing_token, owned, metadata=released_metadata
                )
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
        released = released_metadata.get("metadata")
        if is_json_object(released) and released.get(OPTION_ENDPOINT_MODE) == "tunnel":
            # The claim is gone, so nothing will use the forwards; the TPU itself is untouched.
            with contextlib.suppress(Exception):
                self.tunnels.stop(f"attach-{released['management_uid']}")
        return self.observe(cell_id)


__all__ = ("GcpAttachCapacityDriver",)
