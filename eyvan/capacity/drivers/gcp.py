# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fenced Eyvan capacity drivers for mixed Google Cloud fleets."""

from __future__ import annotations

import hashlib
import os
import socket
import typing as tp
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from threading import RLock

from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object, model_json

if tp.TYPE_CHECKING:
    from eyvan.capacity.providers.gcp.native_worker_ingress import FirewallNetworkReference
from eyvan.capacity import READY_CAPACITY_STATUSES, CapacityStatus
from eyvan.capacity.drivers.base import (
    CELL_ACCELERATOR_METADATA_KEY,
    CapacityProviderError,
    CapacityProviderFailureCode,
    CellCapabilities,
    CellObservation,
    DriverBusyError,
    MutationContext,
    StaleMutationError,
    canonical_execution_backend,
    describe_cell_accelerator,
)
from eyvan.capacity.fleet_state import (
    FleetCellNotRegisteredError,
    FleetCellRecord,
    FleetLeaseBusyError,
    FleetStateStore,
    StaleFenceError,
)
from eyvan.capacity.providers.gcp import (
    ComputeProviderConfig,
    GcloudComputeProvider,
    GcloudDefaultComputeServiceAccountResolver,
    GcloudNativeWorkerArtifactAccessStore,
    GcloudNativeWorkerSecretStore,
    GcloudTpuProvider,
    GcsArtifactBucket,
    NativeWorkerEndpointMode,
    NativeWorkerPublicIngress,
    ProviderObservation,
    compute_instance_name,
    endpoint_mode_from_metadata,
    normalize_compute_config,
)
from eyvan.capacity.providers.gcp.queued_resources import (
    TpuProviderConfig,
    bootstrap_egress_choices,
    normalize_tpu_config,
)
from eyvan.config import EyvanSettings
from eyvan.records import CapacityClass, NativeContainerConfig, NativeRuntimeConfig

#: One lock per (fleet-state document, cell), shared by every driver built over
#: that document in this process -- the GCP umbrella builds a TPU and a Compute
#: driver over the same one, and a cell is only ever the other's by mistake.
_CELL_MUTATION_LOCKS: dict[tuple[str, str], RLock] = {}
_CELL_MUTATION_LOCK_USERS: dict[tuple[str, str], int] = {}
_CELL_MUTATION_LOCKS_GUARD = RLock()


@contextmanager
def _cell_mutation_lock(scope: str, cell_id: str) -> tp.Iterator[None]:
    key = (scope, cell_id)
    with _CELL_MUTATION_LOCKS_GUARD:
        lock = _CELL_MUTATION_LOCKS.setdefault(key, RLock())
        # Register before waiting: a waiter already holds a reference to this
        # lock and must exclude later operations, even across cell resurrection.
        _CELL_MUTATION_LOCK_USERS[key] = _CELL_MUTATION_LOCK_USERS.get(key, 0) + 1
    try:
        with lock:
            yield
    finally:
        with _CELL_MUTATION_LOCKS_GUARD:
            remaining = _CELL_MUTATION_LOCK_USERS[key] - 1
            if remaining:
                _CELL_MUTATION_LOCK_USERS[key] = remaining
            else:
                _CELL_MUTATION_LOCK_USERS.pop(key)
                _CELL_MUTATION_LOCKS.pop(key)


def _forget_cell_mutation_lock(scope: str, cell_id: str) -> None:
    """Evict only an unused lock, never a holder or waiter's shared reference."""

    with _CELL_MUTATION_LOCKS_GUARD:
        key = (scope, cell_id)
        if not _CELL_MUTATION_LOCK_USERS.get(key, 0):
            _CELL_MUTATION_LOCKS.pop(key, None)


#: Record metadata naming the process whose provider operation is in flight for
#: a cell. The in-process lock cannot see another agent sharing the same
#: fleet-state document -- or the outgoing agent of a redeploy, still inside a
#: ``gcloud ... create`` while its successor takes the retry -- and before the
#: fleet lease stopped covering provider calls, that lease was what kept them
#: apart. The claim is written in the same fenced compare-and-swap that claims
#: the cell, so two processes cannot both hold it.
PROVIDER_OPERATION_KEY = "provider_operation"
#: A claim nobody released -- its process crashed -- stops counting after this,
#: or at once when its owner is a dead process on this host.
PROVIDER_OPERATION_TTL_S = 1800.0


def _provider_operation_lapsed(claim: JsonObject, now: datetime) -> bool:
    expires_at = claim.get("expires_at")
    try:
        if not isinstance(expires_at, str) or datetime.fromisoformat(expires_at) <= now:
            return True
    except ValueError:
        return True
    owner = claim.get("owner")
    host, _, rest = owner.partition(":") if isinstance(owner, str) else ("", "", "")
    pid_text = rest.partition(":")[0]
    if host == socket.gethostname() and pid_text.isdigit():
        try:
            os.kill(int(pid_text), 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
    return False


class CapacityProvider(tp.Protocol):
    """Provider operations required by a fenced GCP capacity driver."""

    def ensure(self, record: FleetCellRecord) -> ProviderObservation: ...

    def delete(
        self,
        resource_id: str,
        *,
        project: str,
        zone: str,
        management_uid: str | None = None,
    ) -> None: ...


class NativeWorkerSecretStore(tp.Protocol):
    """Cell-scoped secret lifecycle required by native GCP workers."""

    def ensure(
        self,
        *,
        project: str,
        management_uid: str,
        value: str,
        reader_service_account: str,
    ) -> str: ...

    def delete(self, *, project: str, management_uid: str) -> None: ...


class NativeWorkerArtifactAccessStore(tp.Protocol):
    """Bucket IAM lifecycle required by profile-owned native GCP workers."""

    def ensure(
        self,
        *,
        project: str,
        artifact_bucket: GcsArtifactBucket,
        reader_service_account: str,
    ) -> None: ...


class ServiceAccountResolver(tp.Protocol):
    """Resolve the VM identity used when no service account is configured."""

    def resolve(self, *, project: str) -> str: ...


@dataclass(frozen=True, slots=True)
class NormalizedCapacityRequest:
    """Validated request document plus its concrete provider configuration."""

    document: JsonObject
    runtime: NativeRuntimeConfig
    tpu_config: TpuProviderConfig | None = None
    compute_config: ComputeProviderConfig | None = None


def _native_runtime_from_desired(desired: tp.Mapping[str, JsonValue]) -> NativeRuntimeConfig:
    """Validate the typed runtime carried alongside provider creation options."""

    value = desired.get("runtime")
    if value is None:
        return NativeRuntimeConfig()
    return NativeRuntimeConfig.model_validate(value)


def _native_runtime_from_record(record: FleetCellRecord) -> NativeRuntimeConfig:
    """Load the runtime pinned to a persisted capacity incarnation."""

    value = record.metadata.get("native_runtime")
    if value is None:
        return NativeRuntimeConfig()
    if not is_json_object(value):
        raise ValueError(f"cell {record.name!r} has an invalid persisted native runtime")
    return NativeRuntimeConfig.model_validate(value)


def _required_string(document: tp.Mapping[str, JsonValue], name: str) -> str:
    value = document.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"capacity field {name!r} must be a non-empty string")
    return value


def _optional_string(document: tp.Mapping[str, JsonValue], name: str) -> str | None:
    value = document.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"capacity field {name!r} must be a string when set")
    return value


def _management_uid(record: FleetCellRecord) -> str | None:
    value = record.metadata.get("management_uid")
    return value if isinstance(value, str) and value else None


def _native_runtime_setup(runtime: NativeRuntimeConfig) -> str | None:
    """Render declarative setup after Eyvan installs its native worker."""

    if runtime.is_empty:
        return None
    from eyvan.worker.runtime.native_bootstrap import render_native_runtime_setup

    return render_native_runtime_setup(runtime)


def _persist_native_worker_endpoint_mode(record: FleetCellRecord, mode: NativeWorkerEndpointMode) -> None:
    """Persist public routing only when the profile-managed ingress rule owns it."""

    if mode is NativeWorkerEndpointMode.PROFILE_MANAGED_PUBLIC:
        record.metadata["native_worker_endpoint_mode"] = mode.value
    else:
        record.metadata.pop("native_worker_endpoint_mode", None)


class GcpTpuCapacityDriver:
    """Reconcile GCP TPU queued resources behind Eyvan fences."""

    provider_name = "gcp-tpu"
    capacity_kind = "tpu"
    accelerator_types = ("tpu",)
    capacity_classes = ("spot", "on-demand", "reserved", "guaranteed")

    def __init__(
        self,
        state: FleetStateStore | None = None,
        *,
        provider: CapacityProvider | None = None,
        execution_backend: str | None = None,
        native_worker_port: int | None = None,
        native_worker_package: str | None = None,
        native_worker_package_sha256: str | None = None,
        native_worker_package_generation: str | None = None,
        native_worker_wheelhouse: str | None = None,
        native_worker_wheelhouse_sha256: str | None = None,
        native_worker_wheelhouse_generation: str | None = None,
        native_worker_state_dir: str | None = None,
        native_worker_shared_secret: str | None = None,
        native_worker_secrets: NativeWorkerSecretStore | None = None,
        native_worker_service_account_resolver: ServiceAccountResolver | None = None,
        native_worker_artifact_root_uri: str | None = None,
        native_worker_artifact_access: NativeWorkerArtifactAccessStore | None = None,
        native_worker_public_ingress: NativeWorkerPublicIngress | None = None,
    ):
        settings = EyvanSettings.from_env()
        self.state = state or FleetStateStore.from_uri(settings.fleet_state_uri)
        self.provider: CapacityProvider = provider or GcloudTpuProvider()
        self.execution_backend = canonical_execution_backend(execution_backend or settings.execution_backend)
        self.supports_exec = settings.capacity_supports_exec
        self.native_worker_port = native_worker_port if native_worker_port is not None else settings.native_worker_port
        self.native_worker_package = native_worker_package or settings.native_worker_package
        self.native_worker_package_sha256 = native_worker_package_sha256 or settings.native_worker_package_sha256
        self.native_worker_package_generation = (
            native_worker_package_generation or settings.native_worker_package_generation
        )
        self.native_worker_wheelhouse = native_worker_wheelhouse or settings.native_worker_wheelhouse
        self.native_worker_wheelhouse_sha256 = (
            native_worker_wheelhouse_sha256 or settings.native_worker_wheelhouse_sha256
        )
        self.native_worker_wheelhouse_generation = (
            native_worker_wheelhouse_generation or settings.native_worker_wheelhouse_generation
        )
        self.native_worker_state_dir = native_worker_state_dir or settings.native_worker_state_dir
        self.native_worker_shared_secret = (
            native_worker_shared_secret
            if native_worker_shared_secret is not None
            else settings.native_worker_root_secret
        )
        self.native_worker_secrets = native_worker_secrets or GcloudNativeWorkerSecretStore()
        self.native_worker_service_account_resolver = (
            native_worker_service_account_resolver or GcloudDefaultComputeServiceAccountResolver()
        )
        self.native_worker_artifact_bucket = (
            GcsArtifactBucket.from_artifact_root_uri(native_worker_artifact_root_uri)
            if native_worker_artifact_root_uri is not None
            else None
        )
        self.native_worker_artifact_access = native_worker_artifact_access or GcloudNativeWorkerArtifactAccessStore()
        self.native_worker_public_ingress = native_worker_public_ingress
        self._mutation_lock = RLock()

    def capabilities(self) -> CellCapabilities:
        return CellCapabilities(
            provider=self.provider_name,
            executor=self.execution_backend,
            accelerator_types=self.accelerator_types,
            capacity_classes=self.capacity_classes,
            # Exactly the ``capacity_backend`` values _desired_kind accepts:
            # routing reads this to decide whether this agent can serve a
            # requested backend, so overclaiming would route cells here
            # only to fail validation.
            capacity_backends=("gcp", "gcp-tpu", "tpu"),
            supports_callables=True,
            max_gang_replicas=1 if self.execution_backend in {"native-local", "eyvan-native"} else None,
            # Stated rather than defaulted: an operator turns this on for a fleet
            # they are willing to have diagnostics run on. Advertise that policy
            # rather than silently disabling the runtime's exec capability.
            supports_exec=self.supports_exec,
            resource_enforcement="none" if self.execution_backend in {"native-local", "eyvan-native"} else "backend",
        )

    @contextmanager
    def _cell_mutation(self, cell_id: str) -> tp.Iterator[None]:
        """Serialize mutations of *one* cell without serializing the fleet.

        Holding ``_mutation_lock`` or the fleet-state lease across
        ``provider.ensure`` would queue unrelated cells behind a provider call
        that can take minutes on a Spot insert.

        The fleet lock covers only record claims and persists, each a
        fenced compare-and-swap. What still must not overlap is two mutations
        of the *same* cell -- two creates racing one resource id. This lock,
        always taken before the fleet lock, excludes threads in this process;
        the durable provider-operation claim excludes other agent processes.
        """

        with _cell_mutation_lock(self.state.backend.mutation_scope, cell_id):
            yield

    def _claim_provider_operation(self, record: FleetCellRecord) -> None:
        """Record that this process is about to talk to the provider for ``record``.

        Called inside the fleet-locked claim, so it is part of that fenced
        compare-and-swap. Raises ``DriverBusyError`` -- 409, retried by the
        controller -- while another live process holds the cell.
        """

        self._check_provider_operation(record)
        record.metadata[PROVIDER_OPERATION_KEY] = {
            "owner": self.state.owner,
            "expires_at": (datetime.now(UTC) + timedelta(seconds=PROVIDER_OPERATION_TTL_S)).isoformat(),
        }

    def _check_provider_operation(self, record: FleetCellRecord) -> None:
        """Refuse a foreign provider operation inside a fleet-locked mutation."""

        # A crashed agent on another host blocks even release until the 1800 s
        # PROVIDER_OPERATION_TTL_S expires; a dead same-host PID lapses at once.
        now = datetime.now(UTC)
        claim = record.metadata.get(PROVIDER_OPERATION_KEY)
        if (
            isinstance(claim, dict)
            and claim.get("owner") != self.state.owner
            and not _provider_operation_lapsed(claim, now)
        ):
            raise DriverBusyError(
                f"cell {record.name!r} has a provider operation in flight in another agent process "
                f"({claim.get('owner')}); retry once it completes"
            )

    def _release_provider_operation(self, cell_id: str) -> None:
        """Clear this process's claim; a lost race or a released cell leaves nothing to clear."""

        def clear(record: FleetCellRecord) -> None:
            claim = record.metadata.get(PROVIDER_OPERATION_KEY)
            if isinstance(claim, dict) and claim.get("owner") == self.state.owner:
                record.metadata.pop(PROVIDER_OPERATION_KEY, None)

        try:
            with self._mutation_lock, self.state.exclusive():
                if self.state.get(cell_id) is not None:
                    self.state.mutate_record(cell_id, clear)
        except (FleetCellNotRegisteredError, StaleFenceError, DriverBusyError, FleetLeaseBusyError):
            # Best effort: an unreleased claim lapses on its own.
            pass

    def observe(self, cell_id: str) -> CellObservation:
        record = self.state.get(cell_id)
        if record is None:
            return CellObservation(
                cell_id=cell_id,
                state=CapacityStatus.ABSENT,
                generation=0,
                observed_at=datetime.now(UTC),
            )
        disowned = self._disowned_incarnations(record)
        if disowned is not None and disowned:
            # Hardware carrying this cell's resource-id scheme but a generation
            # the fence disowns. The controller has to be told to *replace* the
            # cell rather than schedule onto it: reporting the record's own
            # state here would send work to capacity nobody is tracking.
            return CellObservation(
                cell_id=record.name,
                state=CapacityStatus.NEEDS_RECREATE,
                generation=record.generation,
                execution_endpoint=None,
                observed_at=datetime.now(UTC),
                observed_tpu_vms=GcpTpuCapacityDriver._observed_tpu_vms(record),
                runtime_process_count=GcpTpuCapacityDriver._runtime_process_count(record),
                failure_code=GcpTpuCapacityDriver._failure_code(record),
                metadata={
                    **GcpTpuCapacityDriver._observation(record).metadata,
                    "disowned_incarnations": disowned,
                },
            )
        return self._observation(record)

    def _disowned_incarnations(self, record: FleetCellRecord) -> tuple[str, ...] | None:
        """Return provider incarnations of this cell the fence does not own.

        Consulted only when the provider can enumerate them; ``None`` means
        the substrate could not be asked at all. A failing listing raises its
        typed provider error rather than dissolving into "looks fine": an
        observation made over a broken link must carry the reason it is
        inconclusive, not present stale record state as fact.
        """

        listing = getattr(self.provider, "list_cell_incarnations", None)
        if not callable(listing) or not record.project or not record.zone or not record.qr_id:
            return None
        incarnations = listing(project=record.project, zone=record.zone, cell_id=record.name)
        prefix = f"{record.name}-r"
        return tuple(
            incarnation
            for incarnation in incarnations
            if isinstance(incarnation, str)
            and incarnation != record.qr_id
            and incarnation.startswith(prefix)
            # Anchor the incarnation suffix: ``{cell}-r`` followed by digits
            # and nothing else. A sibling cell whose own id extends the prefix
            # (``alpha`` and ``alpha-r1`` are both legal operator-chosen ids)
            # must never be reported as this cell's disowned incarnation.
            and incarnation[len(prefix) :].isdigit()
        )

    @staticmethod
    def _observation(record: FleetCellRecord, *, detail: str = "") -> CellObservation:
        metadata: JsonObject = {
            **{key: value for key, value in record.metadata.items() if key != PROVIDER_OPERATION_KEY},
            "provider": record.metadata.get("provider", record.metadata.get("capacity_backend", "gcp-tpu")),
            "project": record.project,
            "zone": record.zone,
            "accelerator_type": record.accelerator_type,
            "capacity_class": record.capacity_class,
            "capacity_backend": record.metadata.get("capacity_backend", "gcp-tpu"),
            "resource_id": record.qr_id,
            "qr_id": record.qr_id,
            "desired_state": record.desired_state,
            "provider_generation": record.provider_generation,
            "head_ip": record.head_ip,
        }
        # Derived last and unconditionally, including the removal: a copy
        # persisted by some other build must never outrank this build's catalog,
        # and a build that cannot describe the cell must not leave one standing.
        accelerator = GcpTpuCapacityDriver._accelerator_metadata(record)
        if accelerator is None:
            metadata.pop(CELL_ACCELERATOR_METADATA_KEY, None)
        else:
            metadata[CELL_ACCELERATOR_METADATA_KEY] = accelerator
        return CellObservation(
            cell_id=record.name,
            state=record.state,
            generation=record.generation,
            execution_endpoint=record.execution_endpoint,
            detail=detail,
            observed_at=datetime.now(UTC),
            observed_tpu_vms=GcpTpuCapacityDriver._observed_tpu_vms(record),
            runtime_process_count=GcpTpuCapacityDriver._runtime_process_count(record),
            failure_code=GcpTpuCapacityDriver._failure_code(record),
            metadata=metadata,
        )

    @staticmethod
    def _accelerator_metadata(record: FleetCellRecord) -> JsonObject | None:
        """Return what the hardware catalog says this cell's hosts carry.

        Two keys are offered to the catalog because a GCP cell can be named two
        ways. A TPU cell's ``accelerator_type`` *is* the catalog row (``v5p-16``).
        A GPU cell's is the GPU model the pool declared (``nvidia-l4``), which
        the catalog deliberately does not describe -- the model fixes neither the
        host nor the device count -- so its row is found through the machine type
        the instance was actually created as.

        The configured GPU count is passed as a contradiction check, not as a
        source. It is worth checking because it is provider-verified:
        ``GcloudComputeProvider._validate_instance`` refuses an instance whose
        ``guestAccelerators`` differ from the configured type and count, so a
        cell that reached HEALTHY carries what its config says.

        Args:
            record: The durable cell record being observed.

        Returns:
            The accelerator block, or ``None`` when this build cannot state one.
            ``None`` means unknown, never "no accelerator".
        """

        machine_type: str | None = None
        attached_device_count: int | None = None
        config = record.metadata.get("compute_config")
        if is_json_object(config):
            raw_machine_type = config.get("machine_type")
            machine_type = raw_machine_type if isinstance(raw_machine_type, str) else None
            raw_gpu_count = config.get("gpu_count")
            if isinstance(raw_gpu_count, int) and not isinstance(raw_gpu_count, bool) and raw_gpu_count > 0:
                attached_device_count = raw_gpu_count
        accelerator = describe_cell_accelerator(
            accelerator_type=record.accelerator_type,
            machine_type=machine_type,
            attached_device_count=attached_device_count,
        )
        return None if accelerator is None else accelerator.as_metadata()

    @staticmethod
    def _failure_code(record: FleetCellRecord) -> CapacityProviderFailureCode | None:
        """Recover a persisted provider reason, ignoring one this build cannot name."""

        value = record.metadata.get("provider_failure_code")
        if not isinstance(value, str):
            return None
        try:
            return CapacityProviderFailureCode(value)
        except ValueError:
            # A newer agent's vocabulary. Treated as "no specific reason" rather
            # than as an error: the observation is still usable, and refusing it
            # would strand a cell during a rolling upgrade.
            return None

    @staticmethod
    def _observed_tpu_vms(record: FleetCellRecord) -> int | None:
        """Return the exact VM count reported by GCP for a healthy TPU allocation."""

        value = record.metadata.get("observed_tpu_vms")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return None

    @staticmethod
    def _runtime_process_count(record: FleetCellRecord) -> int | None:
        """Return the last exact execution width, including legacy TPU records."""

        value = record.metadata.get("runtime_process_count")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return GcpTpuCapacityDriver._observed_tpu_vms(record)

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

    def _persist_provider_observation(
        self,
        cell_id: str,
        report: ProviderObservation,
        context: MutationContext,
        expected_record: FleetCellRecord,
    ) -> CellObservation:
        def mutate(record: FleetCellRecord) -> None:
            if record.generation != context.generation or record.fencing_token != context.fencing_token:
                raise StaleMutationError(f"cell {cell_id!r} changed ownership while its provider operation was running")
            record.state = report.state
            if report.head_ip is not None:
                record.head_ip = report.head_ip
            elif report.state not in READY_CAPACITY_STATUSES:
                record.head_ip = None
            if report.execution_endpoint is not None:
                record.execution_endpoint = report.execution_endpoint
            elif report.state not in READY_CAPACITY_STATUSES:
                record.execution_endpoint = None
            if report.worker_endpoints:
                record.metadata["native_worker_endpoints"] = list(report.worker_endpoints)
            elif report.state not in READY_CAPACITY_STATUSES:
                record.metadata.pop("native_worker_endpoints", None)
            if report.failure_code is not None and report.state not in READY_CAPACITY_STATUSES:
                # Through the durable record rather than returned directly,
                # because ``observe`` re-reads the record without calling the
                # provider: a reason that lived only on this call would be
                # invisible to the next observation of the same cell.
                record.metadata["provider_failure_code"] = report.failure_code.value
            else:
                record.metadata.pop("provider_failure_code", None)
            if report.state in READY_CAPACITY_STATUSES and report.observed_tpu_vms is not None:
                record.metadata["observed_tpu_vms"] = report.observed_tpu_vms
            elif report.state not in READY_CAPACITY_STATUSES:
                record.metadata.pop("observed_tpu_vms", None)
            if report.runtime_process_count is not None:
                count = report.runtime_process_count
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    raise ValueError("provider reported an invalid runtime process count")
                # Deliberately retained through a later partial/unreachable
                # observation. Recounting only endpoints that still answer would
                # shift every following cell's global log ranks.
                record.metadata["runtime_process_count"] = count
            record.metadata.pop("recreate_intent", None)

        record = self._persist_owned_cell(
            cell_id,
            context,
            expected_record,
            phase="provider observation could be persisted",
            cleanup_capacity=True,
            mutate=mutate,
        )
        return self._observation(record, detail=report.detail)

    def _persist_owned_cell(
        self,
        cell_id: str,
        context: MutationContext,
        expected_record: FleetCellRecord,
        *,
        phase: str,
        cleanup_capacity: bool,
        mutate: tp.Callable[[FleetCellRecord], None],
    ) -> FleetCellRecord:
        """Persist an external operation only while its cell incarnation remains owned."""

        try:
            return self.state.mutate_record(cell_id, mutate, generation=context.generation)
        except FleetCellNotRegisteredError:
            self._fence_missing_cell(
                cell_id,
                context,
                phase=phase,
                expected_record=expected_record,
                cleanup_capacity=cleanup_capacity,
            )
        except StaleFenceError as exc:
            raise StaleMutationError(str(exc)) from exc

    def _fence_missing_cell(
        self,
        cell_id: str,
        context: MutationContext,
        *,
        phase: str,
        expected_record: FleetCellRecord | None = None,
        cleanup_capacity: bool = False,
    ) -> tp.NoReturn:
        """Fence a vanished cell generation and roll back only resources it still owns."""

        try:
            fenced = self.state.fence_missing_record(cell_id, context.generation, context.fencing_token)
        except StaleFenceError as exc:
            raise StaleMutationError(str(exc)) from exc
        if not fenced:
            raise StaleMutationError(f"cell {cell_id!r} no longer belongs to this generation before {phase}")
        if expected_record is not None:
            try:
                if cleanup_capacity:
                    self._delete_owned_capacity(expected_record)
                self._release_auxiliary_resources(expected_record)
            except Exception as exc:
                raise StaleMutationError(
                    f"cell {cell_id!r} was removed before {phase}; old generation was fenced but cleanup failed "
                    f"with {type(exc).__name__}"
                ) from exc
        raise StaleMutationError(f"cell {cell_id!r} was removed before {phase}; old generation was fenced")

    def _ensure_owned(
        self,
        cell_id: str,
        context: MutationContext,
        *,
        recreate_terminal: bool = False,
    ) -> CellObservation:
        record = self.state.get(cell_id)
        if record is None:
            self._fence_missing_cell(cell_id, context, phase="capacity reconciliation could begin")
        if record.desired_state != "up":
            return self._observation(record, detail=f"capacity desired state is {record.desired_state}")
        if recreate_terminal:
            self._resume_recreate(record, context)
            resumed_record = self.state.get(cell_id)
            if resumed_record is None:
                self._fence_missing_cell(
                    cell_id,
                    context,
                    phase="capacity recreation could resume",
                    expected_record=record,
                    cleanup_capacity=True,
                )
            record = resumed_record
        self._refresh_native_worker_public_ingress(record)
        self._finalize_provider_config(cell_id, context)
        finalized_record = self.state.get(cell_id)
        if finalized_record is None:
            self._fence_missing_cell(
                cell_id,
                context,
                phase="native worker bootstrap could be persisted",
                expected_record=record,
            )
        record = finalized_record
        report = self.provider.ensure(record)
        if recreate_terminal and report.state is CapacityStatus.NEEDS_RECREATE:
            if record.state is not CapacityStatus.NEEDS_RECREATE:
                return self._persist_provider_observation(cell_id, report, context, record)
            record = self._plan_recreate(record, context)
            self._delete_previous_capacity(record)
            report = self.provider.ensure(record)
        return self._persist_provider_observation(cell_id, report, context, record)

    def _resume_recreate(self, record: FleetCellRecord, context: MutationContext) -> None:
        intent = record.metadata.get("recreate_intent")
        if not isinstance(intent, dict):
            return
        if record.generation != context.generation or record.fencing_token != context.fencing_token:
            raise StaleMutationError(f"cell {record.name!r} recreate intent is owned by another generation")
        self._delete_previous_capacity(record)

    def _plan_recreate(self, record: FleetCellRecord, context: MutationContext) -> FleetCellRecord:
        previous_qr_id = record.qr_id

        def plan(current: FleetCellRecord) -> None:
            if current.generation != context.generation or current.fencing_token != context.fencing_token:
                raise StaleMutationError(f"cell {record.name!r} changed ownership before capacity recreation")
            current.provider_generation += 1
            current.qr_id = self._resource_id(current.name, current.provider_generation)
            current.state = CapacityStatus.RECREATING
            current.head_ip = None
            current.execution_endpoint = None
            current.metadata.pop("native_worker_endpoints", None)
            current.metadata.pop("observed_tpu_vms", None)
            current.metadata["recreate_intent"] = {
                "previous_qr_id": previous_qr_id,
                "target_qr_id": current.qr_id,
                "cell_generation": context.generation,
            }

        replacement = self._persist_owned_cell(
            record.name,
            context,
            record,
            phase="capacity recreation could be planned",
            cleanup_capacity=True,
            mutate=plan,
        )
        if self.capacity_kind != "tpu" or self.execution_backend != "eyvan-native":
            return replacement
        config = replacement.metadata.get("tpu_config")
        if not isinstance(config, dict):
            raise ValueError("native TPU recreation requires a persisted provider configuration")
        setup_script = replacement.metadata.get("native_worker_setup_script")
        if setup_script is not None and not isinstance(setup_script, str):
            raise ValueError("native TPU recreation has an invalid setup script")
        runtime = _native_runtime_from_record(replacement)
        refreshed = normalize_tpu_config(config)
        endpoint_mode = self._configure_native_tpu_worker(
            replacement,
            refreshed,
            before_start=setup_script,
            runtime_setup=_native_runtime_setup(runtime),
            runtime_env=runtime.env,
            runtime_container=runtime.container,
            container_artifact_mounts=tuple(artifact.destination for artifact in runtime.artifacts),
            runtime_python=runtime.python,
        )

        def persist(current: FleetCellRecord) -> None:
            if current.generation != context.generation or current.fencing_token != context.fencing_token:
                raise StaleMutationError(f"cell {record.name!r} changed ownership before native TPU recreation")
            if current.provider_generation != replacement.provider_generation:
                raise StaleMutationError(
                    f"cell {record.name!r} changed provider generation before native TPU recreation"
                )
            current.metadata["tpu_config"] = refreshed
            current.metadata["native_worker_port"] = self.native_worker_port
            current.metadata["execution_backend"] = self.execution_backend
            current.metadata["native_worker_secret_id"] = replacement.metadata["native_worker_secret_id"]
            current.metadata["native_worker_bootstrap_generation"] = replacement.provider_generation
            current.metadata.pop("native_worker_bootstrap_key", None)
            _persist_native_worker_endpoint_mode(current, endpoint_mode)

        return self._persist_owned_cell(
            replacement.name,
            context,
            replacement,
            phase="native worker recreation could be persisted",
            cleanup_capacity=True,
            mutate=persist,
        )

    def _delete_previous_capacity(self, record: FleetCellRecord) -> None:
        intent = record.metadata.get("recreate_intent")
        if not isinstance(intent, dict):
            return
        previous_qr_id = intent.get("previous_qr_id")
        if isinstance(previous_qr_id, str) and previous_qr_id and record.project and record.zone:
            self.provider.delete(
                previous_qr_id,
                project=record.project,
                zone=record.zone,
                management_uid=_management_uid(record),
            )

    def _delete_owned_capacity(self, record: FleetCellRecord) -> None:
        """Delete every resource named by a lost cell incarnation."""

        if not record.project or not record.zone:
            return
        recreate_intent = record.metadata.get("recreate_intent")
        previous_qr_id = recreate_intent.get("previous_qr_id") if isinstance(recreate_intent, dict) else None
        resource_ids = {
            resource_id for resource_id in (previous_qr_id, record.qr_id) if isinstance(resource_id, str) and resource_id
        }
        for resource_id in sorted(resource_ids):
            self.provider.delete(
                resource_id,
                project=record.project,
                zone=record.zone,
                management_uid=_management_uid(record),
            )

    def reconcile(self, cell_id: str, context: MutationContext) -> CellObservation:
        with self._cell_mutation(cell_id):
            claimed = False
            # Acquisition sits inside the guard: the claim is durable as soon as
            # the fleet-locked mutation commits, which can precede an interrupt at
            # the lock's exit. A refused claim is never ours, so it is not cleared.
            try:
                with self._mutation_lock, self.state.exclusive():

                    def claim(record: FleetCellRecord) -> None:
                        nonlocal claimed
                        if self._accept_mutation(record, context):
                            self._stamp(record, context)
                        self._claim_provider_operation(record)
                        claimed = True

                    try:
                        self.state.mutate_record(
                            cell_id,
                            claim,
                            generation=context.generation,
                        )
                    except FleetCellNotRegisteredError:
                        return self.observe(cell_id)
                    except StaleFenceError as exc:
                        raise StaleMutationError(str(exc)) from exc
                # Provider I/O outside the fleet lock; see ``_cell_mutation``.
                return self._ensure_owned(cell_id, context, recreate_terminal=True)
            finally:
                if claimed:
                    self._release_provider_operation(cell_id)

    def apply_capacity(
        self,
        cell_id: str,
        desired: tp.Mapping[str, JsonValue],
        context: MutationContext,
    ) -> CellObservation:
        normalized = self._normalize_desired(desired)
        document = normalized.document
        project = _required_string(document, "project")
        zone = _required_string(document, "zone")
        accelerator_type = _required_string(document, "accelerator_type")
        capacity_class_value = document.get("capacity_class", "spot")
        if not isinstance(capacity_class_value, str):
            raise ValueError("capacity field 'capacity_class' must be a string")
        capacity_class = capacity_class_value.replace("_", "-")
        if capacity_class not in self.capacity_classes:
            # Two different failures, and the difference is what an operator does
            # next. A class Eyvan does not have is a typo; a class Eyvan models
            # and this backend has not implemented is a gap, and saying so keeps
            # somebody from concluding the fleet YAML is wrong when it is not.
            known = {candidate.value.replace("_", "-") for candidate in CapacityClass}
            if capacity_class in known:
                raise ValueError(
                    f"{self.provider_name} does not yet provision {capacity_class!r} capacity. Eyvan models the "
                    "class and this backend has not implemented it, so the pool is refused rather than quietly "
                    f"provisioned as something else. Supported here: {', '.join(sorted(self.capacity_classes))}"
                )
            raise ValueError(f"unsupported {self.provider_name} capacity class {capacity_class!r}")
        executor_value = document.get("executor")
        if executor_value is not None and not isinstance(executor_value, str):
            raise ValueError("capacity field 'executor' must be a string when set")
        requested_executor = canonical_execution_backend(executor_value or self.execution_backend)
        if requested_executor != self.execution_backend:
            raise ValueError(
                f"cell requests executor {requested_executor!r}, but this agent provides {self.execution_backend!r}"
            )
        with self._cell_mutation(cell_id):
            claimed = False
            # See ``reconcile``: the claim is acquired inside the guard that clears it.
            try:
                with self._mutation_lock, self.state.exclusive():

                    def mutate(record: FleetCellRecord) -> None:
                        nonlocal claimed
                        accepted = self._accept_mutation(record, context)
                        self._claim_provider_operation(record)
                        claimed = True
                        if not accepted:
                            return
                        persisted_kind = record.metadata.get("capacity_kind")
                        if persisted_kind not in {None, self.capacity_kind}:
                            raise ValueError(
                                f"cell {cell_id!r} is already managed by {persisted_kind!r} capacity; "
                                "delete it before changing"
                            )
                        record.project = project
                        record.zone = zone
                        record.accelerator_type = accelerator_type
                        record.capacity_class = capacity_class
                        record.runtime_version = _optional_string(document, "runtime_version")
                        record.execution_endpoint = _optional_string(document, "execution_endpoint")
                        agent_endpoint = _optional_string(document, "agent_endpoint")
                        if agent_endpoint is not None:
                            record.metadata["agent_endpoint"] = agent_endpoint
                        record.metadata["capacity_backend"] = self.provider_name
                        record.metadata["capacity_kind"] = self.capacity_kind
                        record.metadata.setdefault("management_uid", uuid.uuid4().hex)
                        if record.qr_id is None:
                            record.provider_generation += 1
                            record.qr_id = self._resource_id(cell_id, record.provider_generation)
                        self._persist_provider_config(record, normalized)
                        record.desired_state = "up"
                        self._stamp(record, context)

                    try:
                        self.state.mutate_or_create_record(cell_id, mutate, generation=context.generation)
                    except StaleFenceError as exc:
                        raise StaleMutationError(str(exc)) from exc
                # Provider I/O outside the fleet lock; see ``_cell_mutation``.
                return self._ensure_owned(cell_id, context)
            finally:
                # Only this process's own claim is ever cleared, so clearing after
                # a mutation that raised before persisting is a harmless no-op.
                if claimed:
                    self._release_provider_operation(cell_id)

    def _validate_normalized_document(self, document: JsonObject, runtime: NativeRuntimeConfig) -> None:
        """Validate execution/runtime fields common to GCP capacity kinds."""

        if not runtime.is_empty and self.execution_backend != "eyvan-native":
            raise ValueError("native runtime configuration requires execution backend eyvan-native")
        document["runtime"] = model_json(runtime)
        backend_value = document.get("capacity_backend")
        if backend_value is not None and not isinstance(backend_value, str):
            raise ValueError("capacity field 'capacity_backend' must be a string when set")
        requested_backend = (backend_value or self.provider_name).strip().lower().replace("_", "-")
        if requested_backend not in {"gcp", self.provider_name, self.capacity_kind}:
            raise ValueError(f"{self.provider_name} driver cannot satisfy capacity backend {requested_backend!r}")

    def _normalize_desired(self, desired: tp.Mapping[str, JsonValue]) -> NormalizedCapacityRequest:
        document: JsonObject = dict(desired)
        runtime = _native_runtime_from_desired(document)
        if runtime.vm_image is not None:
            raise ValueError("runtime.vm_image requires the gcp-compute capacity backend")
        self._validate_normalized_document(document, runtime)
        config = normalize_tpu_config(document)
        document["runtime_version"] = config["runtime_version"]
        return NormalizedCapacityRequest(document=document, runtime=runtime, tpu_config=config)

    def _persist_provider_config(
        self,
        record: FleetCellRecord,
        desired: NormalizedCapacityRequest,
    ) -> None:
        config = desired.tpu_config
        if config is None:
            raise ValueError("normalized TPU provider configuration is missing")
        persisted_config = config.copy()
        if self.execution_backend == "eyvan-native":
            setup_script = persisted_config.get("startup_script")
            if not desired.runtime.is_empty and setup_script is not None:
                raise ValueError("use runtime.init_commands instead of provider_options.startup_script")
            record.metadata["native_worker_setup_script"] = setup_script
            record.metadata["native_runtime"] = model_json(desired.runtime)
            record.metadata["native_worker_port"] = self.native_worker_port
            record.metadata["execution_backend"] = self.execution_backend
            record.metadata.pop("native_worker_secret_id", None)
            record.metadata.pop("native_worker_bootstrap_generation", None)
            record.metadata.pop("native_worker_bootstrap_key", None)
            record.metadata.pop("native_worker_endpoint_mode", None)
        else:
            record.metadata.pop("native_worker_port", None)
            record.metadata.pop("execution_backend", None)
            record.metadata.pop("native_worker_secret_id", None)
            record.metadata.pop("native_worker_setup_script", None)
            record.metadata.pop("native_runtime", None)
            record.metadata.pop("native_worker_endpoint_mode", None)
        record.metadata["tpu_config"] = persisted_config

    def _bootstrap_key(self, record: FleetCellRecord) -> str:
        """Return what must be unchanged for a host's bootstrap to still be current.

        The provider generation alone is not enough. The staged worker wheel is
        baked into the startup script at bootstrap time, and re-staging a wheel
        does not change the generation. Artifact identity must participate so
        re-staged code reaches existing cells.

        The identity includes both artifact URIs, their verified content hashes,
        and any GCS generation pins. Changing any supplied byte source rewrites
        the script, while identical qualified artifacts do not churn the host.

        The script's own behavior revision is pinned for the same reason:
        identical artifacts with a fixed script still means a new bootstrap,
        otherwise a bootstrap bug fix would never reach existing cells.
        """

        from eyvan.worker.runtime.native_bootstrap import NATIVE_WORKER_BOOTSTRAP_REVISION

        identity = "|".join(
            (
                NATIVE_WORKER_BOOTSTRAP_REVISION,
                self.native_worker_package or "",
                self.native_worker_package_sha256 or "",
                self.native_worker_package_generation or "",
                self.native_worker_wheelhouse or "",
                self.native_worker_wheelhouse_sha256 or "",
                self.native_worker_wheelhouse_generation or "",
            )
        )
        return f"{record.provider_generation}:{hashlib.sha256(identity.encode()).hexdigest()[:16]}"

    def _finalize_provider_config(self, cell_id: str, context: MutationContext) -> None:
        """Finish any provider setup that must run outside durable-state CAS callbacks."""

        if self.capacity_kind != "tpu" or self.execution_backend != "eyvan-native":
            return
        record = self.state.get(cell_id)
        if record is None:
            self._fence_missing_cell(cell_id, context, phase="native worker bootstrap could begin")
        if record.generation != context.generation or record.fencing_token != context.fencing_token:
            raise StaleMutationError(f"cell {cell_id!r} changed ownership before native TPU bootstrap")
        if record.metadata.get("native_worker_bootstrap_key") == self._bootstrap_key(record):
            return
        config = record.metadata.get("tpu_config")
        if not isinstance(config, dict):
            raise ValueError("native TPU bootstrap requires a persisted provider configuration")
        setup_script = record.metadata.get("native_worker_setup_script")
        if setup_script is not None and not isinstance(setup_script, str):
            raise ValueError("native TPU bootstrap has an invalid setup script")
        runtime = _native_runtime_from_record(record)
        configured = normalize_tpu_config(config)
        endpoint_mode = self._configure_native_tpu_worker(
            record,
            configured,
            before_start=setup_script,
            runtime_setup=_native_runtime_setup(runtime),
            runtime_env=runtime.env,
            runtime_container=runtime.container,
            container_artifact_mounts=tuple(artifact.destination for artifact in runtime.artifacts),
            runtime_python=runtime.python,
        )

        def persist(current: FleetCellRecord) -> None:
            if current.generation != context.generation or current.fencing_token != context.fencing_token:
                raise StaleMutationError(f"cell {cell_id!r} changed ownership before native TPU bootstrap")
            if current.provider_generation != record.provider_generation:
                raise StaleMutationError(f"cell {cell_id!r} changed provider generation before native TPU bootstrap")
            current.metadata["tpu_config"] = configured
            current.metadata["native_worker_port"] = self.native_worker_port
            current.metadata["execution_backend"] = self.execution_backend
            current.metadata["native_worker_secret_id"] = record.metadata["native_worker_secret_id"]
            current.metadata["native_worker_bootstrap_generation"] = record.provider_generation
            current.metadata["native_worker_bootstrap_key"] = self._bootstrap_key(record)
            _persist_native_worker_endpoint_mode(current, endpoint_mode)

        self._persist_owned_cell(
            cell_id,
            context,
            record,
            phase="native worker bootstrap could be persisted",
            cleanup_capacity=False,
            mutate=persist,
        )

    def _native_worker_service_account(
        self,
        record: FleetCellRecord,
        config: TpuProviderConfig | ComputeProviderConfig,
    ) -> str:
        """Return an explicit override or GCP's default VM identity for secret access."""

        configured = config.get("service_account")
        if configured is not None:
            if not isinstance(configured, str) or not configured.strip():
                raise ValueError("native worker provider_options.service_account must be a non-empty string when set")
            return configured
        if not isinstance(record.project, str) or not record.project:
            raise ValueError("native worker bootstrap requires a GCP project")
        service_account = self.native_worker_service_account_resolver.resolve(project=record.project)
        if not isinstance(service_account, str) or not service_account.strip():
            raise ValueError("native worker default service-account resolver returned an empty identity")
        return service_account

    def _ensure_native_worker_artifact_access(
        self,
        record: FleetCellRecord,
        service_account: str,
    ) -> None:
        """Grant the profile-owned worker identity narrow access to its GCS artifact bucket."""

        artifact_bucket = self.native_worker_artifact_bucket
        if artifact_bucket is None:
            return
        if not isinstance(record.project, str) or not record.project:
            raise ValueError("native worker artifact access requires a GCP project")
        self.native_worker_artifact_access.ensure(
            project=record.project,
            artifact_bucket=artifact_bucket,
            reader_service_account=service_account,
        )

    def _configure_native_worker_public_ingress(
        self,
        record: FleetCellRecord,
        config: TpuProviderConfig | ComputeProviderConfig,
        *,
        has_public_ip: bool,
    ) -> NativeWorkerEndpointMode:
        """Attach the profile-owned firewall tag before public worker creation."""

        ingress = self.native_worker_public_ingress
        if not has_public_ip or ingress is None:
            return NativeWorkerEndpointMode.PRIVATE
        if not isinstance(record.project, str) or not record.project:
            raise ValueError("profile-managed native worker ingress requires a GCP project")
        tag = ingress.ensure(
            project=record.project,
            network=config["network"],
            port=self.native_worker_port,
        )
        if not isinstance(tag, str) or not tag or tag != tag.strip():
            raise ValueError("profile-managed native worker ingress returned an invalid network tag")
        if tag not in config["tags"]:
            config["tags"] = [*config["tags"], tag]
        return NativeWorkerEndpointMode.PROFILE_MANAGED_PUBLIC

    def _refresh_native_worker_public_ingress(self, record: FleetCellRecord) -> None:
        """Update the managed source /32 when the local public IPv4 changes."""

        try:
            endpoint_mode = endpoint_mode_from_metadata(record.metadata)
        except ValueError as exc:
            raise ValueError(f"cell {record.name!r} has an invalid native worker endpoint mode") from exc
        if endpoint_mode is NativeWorkerEndpointMode.PRIVATE:
            return
        if self.execution_backend != "eyvan-native":
            raise ValueError(f"cell {record.name!r} has profile-managed public routing without eyvan-native execution")
        ingress = self.native_worker_public_ingress
        if ingress is None:
            raise ValueError(
                f"cell {record.name!r} requires its profile-managed public ingress rule, "
                "but this agent has no runtime profile"
            )
        if not isinstance(record.project, str) or not record.project:
            raise ValueError(f"cell {record.name!r} requires a GCP project for profile-managed public ingress")
        if self.capacity_kind == "tpu":
            raw_config = record.metadata.get("tpu_config")
            if not isinstance(raw_config, dict):
                raise ValueError(f"cell {record.name!r} has no TPU provider configuration for public ingress")
            config = normalize_tpu_config(raw_config)
            if config["internal_ips"]:
                raise ValueError(f"cell {record.name!r} marks a private TPU as profile-managed public")
            network = config["network"]
        else:
            raw_config = record.metadata.get("compute_config")
            if not isinstance(raw_config, dict):
                raise ValueError(f"cell {record.name!r} has no Compute provider configuration for public ingress")
            config = normalize_compute_config(raw_config)
            if config["internal_ip_only"]:
                raise ValueError(f"cell {record.name!r} marks a private Compute VM as profile-managed public")
            network = config["network"]
        ingress.ensure(project=record.project, network=network, port=self.native_worker_port)

    def _configure_native_tpu_worker(
        self,
        record: FleetCellRecord,
        config: TpuProviderConfig,
        *,
        before_start: str | None,
        runtime_setup: str | None = None,
        runtime_env: tp.Mapping[str, str] | None = None,
        runtime_container: NativeContainerConfig | None = None,
        container_artifact_mounts: tp.Sequence[str] = (),
        runtime_python: str | None = None,
    ) -> NativeWorkerEndpointMode:
        """Inject a shared native-worker bootstrap into every TPU VM in a queued resource."""

        from eyvan.worker.runtime.native_bootstrap import build_native_worker_startup_script, derive_native_worker_token

        if not isinstance(self.native_worker_shared_secret, str) or not self.native_worker_shared_secret.strip():
            raise ValueError("eyvan-native TPU capacity requires EYVAN_NATIVE_WORKER_ROOT_SECRET")
        if not isinstance(self.native_worker_package, str) or not self.native_worker_package.strip():
            raise ValueError(
                "eyvan-native TPU capacity requires EYVAN_NATIVE_WORKER_PACKAGE; "
                "use a gs:// wheel with its SHA-256 and object generation"
            )
        if not isinstance(record.project, str) or not record.project:
            raise ValueError("native TPU bootstrap requires a GCP project")
        management_uid = record.metadata.get("management_uid")
        if not isinstance(management_uid, str) or not management_uid:
            raise ValueError("native TPU bootstrap requires an Eyvan management UID")
        service_account = self._native_worker_service_account(record, config)
        egress = config.get("bootstrap_egress")
        if not isinstance(egress, str) or not egress:
            raise ValueError(
                "eyvan-native TPU capacity requires provider_options.bootstrap_egress "
                f"(one of: {bootstrap_egress_choices()})"
            )
        if config.get("internal_ips", False) and egress == "public-internet":
            raise ValueError("internal-only eyvan-native TPU capacity cannot use bootstrap_egress=public-internet")
        endpoint_mode = self._configure_native_worker_public_ingress(
            record,
            config,
            has_public_ip=not config["internal_ips"],
        )
        wheelhouse = None
        if egress == "private-google-access":
            if not self.native_worker_package.startswith("gs://"):
                raise ValueError(
                    "private-google-access native TPU bootstrap requires "
                    "EYVAN_NATIVE_WORKER_PACKAGE to be a verified gs:// wheel"
                )
            if not isinstance(self.native_worker_wheelhouse, str) or not self.native_worker_wheelhouse.strip():
                raise ValueError(
                    "private-google-access eyvan-native TPU capacity requires EYVAN_NATIVE_WORKER_WHEELHOUSE "
                    "with every worker dependency wheel"
                )
            if not self.native_worker_wheelhouse.startswith("gs://"):
                raise ValueError("EYVAN_NATIVE_WORKER_WHEELHOUSE must be a gs:// zip archive")
            wheelhouse = self.native_worker_wheelhouse
        scopes = list(config["scopes"])
        cloud_platform_scope = "https://www.googleapis.com/auth/cloud-platform"
        if cloud_platform_scope not in scopes:
            scopes.append(cloud_platform_scope)
        config["scopes"] = scopes
        self._ensure_native_worker_artifact_access(record, service_account)
        worker_token = derive_native_worker_token(
            self.native_worker_shared_secret,
            management_uid,
            provider_generation=record.provider_generation,
        )
        if worker_token is None:
            raise ValueError("eyvan-native TPU capacity requires EYVAN_NATIVE_WORKER_ROOT_SECRET")
        secret_resource = self.native_worker_secrets.ensure(
            project=record.project,
            management_uid=management_uid,
            value=worker_token,
            reader_service_account=service_account,
        )
        config["startup_script"] = build_native_worker_startup_script(
            package=self.native_worker_package,
            package_sha256=self.native_worker_package_sha256,
            package_generation=self.native_worker_package_generation,
            port=self.native_worker_port,
            state_dir=self.native_worker_state_dir,
            worker_secret_resource=secret_resource,
            wheelhouse=wheelhouse,
            wheelhouse_sha256=self.native_worker_wheelhouse_sha256,
            wheelhouse_generation=self.native_worker_wheelhouse_generation,
            before_start=before_start,
            runtime_setup=runtime_setup,
            runtime_env=runtime_env,
            runtime_container=runtime_container,
            container_artifact_mounts=container_artifact_mounts,
            runtime_python=runtime_python,
        )
        record.metadata["native_worker_port"] = self.native_worker_port
        record.metadata["execution_backend"] = self.execution_backend
        record.metadata["native_worker_secret_id"] = secret_resource.rsplit("/versions/", 1)[0]
        return endpoint_mode

    @staticmethod
    def _resource_id(cell_id: str, provider_generation: int) -> str:
        return f"{cell_id}-r{provider_generation}"

    def drain(self, cell_id: str, context: MutationContext) -> CellObservation:
        with self._cell_mutation(cell_id):
            return self._drain_claimed(cell_id, context)

    def _drain_claimed(self, cell_id: str, context: MutationContext) -> CellObservation:
        with self._mutation_lock, self.state.exclusive():

            def mutate(record: FleetCellRecord) -> None:
                accepted = self._accept_mutation(record, context)
                # No provider I/O, but advancing this fence would disown an
                # in-flight create. The fleet lock spans this entire mutation.
                self._check_provider_operation(record)
                if not accepted:
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
        """Release one cell's provider capacity; see :meth:`_release_claimed`."""

        with self._cell_mutation(cell_id):
            result = self._release_claimed(cell_id, context)
        if self.state.get(cell_id) is None:
            # A newer generation may already be acquiring the same cell lock.
            _forget_cell_mutation_lock(self.state.backend.mutation_scope, cell_id)
        return result

    def _release_claimed(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Release one cell's provider capacity, holding the fleet lock only to claim it.

        Holding ``_mutation_lock`` and the durable fleet-state lease across
        provider deletes would serialize all releases sharing a document. The
        resulting queue can exceed client timeouts, causing retries that add
        load even after provider capacity is gone.

        The lock exists to make the *record* transition safe, not to serialise
        Google. So claim under it, delete outside it, and tombstone under it
        again. The fence survives the gap by construction: the claim stamps this
        generation and fencing token, and ``still_owned`` re-checks both before
        the tombstone, which is exactly the check it was written for. A rival
        mutation that takes the record in between loses the tombstone rather
        than corrupting it.
        """

        claim_taken: dict[str, bool] = {}
        released_project: str | None = None
        released_uid: str | None = None
        # Acquisition sits inside the guard (see ``reconcile``): the claim is
        # durable once the fleet-locked mutation commits, before the lock exits.
        try:
            record = self._claim_for_release(cell_id, context, claim_taken)
            if isinstance(record, CellObservation):
                return record
            if record is not None:
                # Outside the fleet lock: the durable provider claim protects
                # this cell through both deletion and auxiliary cleanup.
                self._delete_owned_capacity(record)
                self._release_auxiliary_resources(record)
                uid = record.metadata.get("management_uid")
                if isinstance(record.project, str) and isinstance(uid, str) and uid:
                    released_project, released_uid = record.project, uid
            result = self._tombstone_after_release(
                cell_id,
                context,
                claimed=record is not None,
                extra_metadata={"management_uid": released_uid} if released_uid is not None else None,
            )
        finally:
            # Success removes the record; failure must allow another agent to retry.
            if claim_taken:
                self._release_provider_operation(cell_id)
        if released_project is not None:
            # After the tombstone lands: the sweep reads tombstone metadata.
            self._sweep_orphaned_native_worker_secrets(released_project)
        return result

    def _claim_for_release(
        self,
        cell_id: str,
        context: MutationContext,
        claim_taken: dict[str, bool] | None = None,
    ) -> FleetCellRecord | CellObservation | None:
        """Mark one cell DELETING under this fence, or report it already released.

        ``claim_taken`` is set as soon as this process's provider claim is on the
        record, so the caller can clear it even if the lock's exit is interrupted.
        """

        with self._mutation_lock, self.state.exclusive():
            owned: dict[str, FleetCellRecord] = {}

            def claim(record: FleetCellRecord) -> None:
                accepted = self._accept_mutation(record, context)
                self._claim_provider_operation(record)
                if claim_taken is not None:
                    claim_taken["claim"] = True
                if not accepted:
                    if (
                        record.state is CapacityStatus.DELETING
                        and record.generation == context.generation
                        and record.fencing_token == context.fencing_token
                    ):
                        owned["record"] = record
                        return
                    # The idempotency key was already consumed by another mutation while
                    # the record still names live capacity. _accept_mutation has already
                    # validated the fence, so fall through and re-stamp the release
                    # instead of silently leaking the provider resources.
                record.desired_state = "down"
                record.state = CapacityStatus.DELETING
                self._stamp(record, context)
                owned["record"] = record

            try:
                self.state.mutate_record(cell_id, claim, generation=context.generation)
            except FleetCellNotRegisteredError:
                pass
            except StaleFenceError as exc:
                tombstone = self.state.cell_tombstone(cell_id)
                if (
                    self.state.get(cell_id) is None
                    and tombstone is not None
                    and int(tombstone["generation"]) == context.generation
                    and tombstone.get("fencing_token") == context.fencing_token
                ):
                    return self._released(cell_id, context.generation)
                raise StaleMutationError(str(exc)) from exc

            record = owned.get("record")
            if record is None and self.state.get(cell_id) is not None:
                return self.observe(cell_id)
            return record

    def _tombstone_after_release(
        self,
        cell_id: str,
        context: MutationContext,
        *,
        claimed: bool,
        extra_metadata: tp.Mapping[str, str] | None = None,
    ) -> CellObservation:
        """Retire the record, refusing if anything took ownership during the deletes."""

        def still_owned(current: FleetCellRecord) -> bool:
            return current.generation == context.generation and current.fencing_token == context.fencing_token

        with self._mutation_lock, self.state.exclusive():
            # Re-read rather than trusting the pre-delete snapshot: the point of
            # dropping the lock is that something else may have run, and this is
            # where that is caught.
            if not claimed and self.state.get(cell_id) is not None:
                return self.observe(cell_id)
            try:
                removed = self.state.tombstone_record(
                    cell_id,
                    context.generation,
                    context.fencing_token,
                    still_owned,
                    metadata=extra_metadata,
                )
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
            if not removed:
                raise StaleMutationError(f"cell {cell_id!r} changed ownership during release")
            return self._released(cell_id, context.generation)

    def _release_auxiliary_resources(self, record: FleetCellRecord) -> None:
        """Release provider resources coupled to a cell after capacity deletion."""

        if self.capacity_kind != "tpu" or self.execution_backend != "eyvan-native" or not record.project:
            return
        management_uid = record.metadata.get("management_uid")
        if not isinstance(management_uid, str) or not management_uid:
            raise ValueError("native TPU cleanup requires an Eyvan management UID")
        self.native_worker_secrets.delete(project=record.project, management_uid=management_uid)
        self._release_unused_native_ingress_rules(record)

    def _release_unused_native_ingress_rules(self, record: FleetCellRecord) -> None:
        """Delete managed worker ingress rules whose last user this cell was.

        The rule is per (network tag, port) and deliberately survives
        scale-to-zero -- ``ensure`` never deletes so the next cell can reach
        its worker -- which is also why nothing else ever deleted one: rules
        outlived their runtime forever and accumulated toward the project's
        firewall cap (issue-register E22). This consults the fleet-state
        document: a rule dies here only when no *other* registered native
        cell of this runtime still targets the same network, tag, and port,
        and deletion re-confirms ownership itself. Best-effort by contract:
        the capacity is already gone, so a failed sweep must never fail the
        release.
        """

        network_tag = getattr(self.native_worker_public_ingress, "network_tag", None)
        if not isinstance(network_tag, str) or not network_tag or not isinstance(record.project, str):
            return
        from eyvan.capacity.providers.gcp.native_worker_ingress import (
            list_managed_ingress_rules,
            release_managed_ingress_rule,
        )

        try:
            rules = list_managed_ingress_rules(project=record.project)
        except (CapacityProviderError, OSError, ValueError):
            return
        for rule in rules:
            if rule.network_tag != network_tag or rule.port != self.native_worker_port:
                continue
            if self._native_cell_remains_for_network(record, rule.network):
                continue
            try:
                release_managed_ingress_rule(project=record.project, rule_name=rule.name)
            except (CapacityProviderError, OSError):
                # One failure means GCP-side trouble; stop sweeping rather
                # than hammering a provider that just stopped answering.
                return

    def _sweep_orphaned_native_worker_secrets(self, project: str) -> None:
        """Delete native-worker secrets left behind by interrupted releases.

        A release deletes the cell's secret *before* it tombstones the record,
        so a crash between those two steps can leak the secret. Releases stamp
        the cell's management UID into its tombstone, and every later release
        re-checks those uids: a secret whose uid this document released but
        which still exists is swept here. Uids this runtime never released are
        never candidates, so another profile sharing the project cannot be
        damaged by a sweep keyed to tombstones it did not write. Best-effort
        like the ingress sweep: the release already succeeded.
        """

        sweep = getattr(self.native_worker_secrets, "list_managed", None)
        if project == "" or not callable(sweep):
            return
        try:
            document, _ = self.state.backend.read()
        except (CapacityProviderError, OSError, ValueError):
            return
        tombstones = document.get("cell_tombstones")
        cells = document.get("cells")
        if not isinstance(tombstones, dict) or not isinstance(cells, dict):
            return

        def record_uid(value: object) -> str | None:
            # Live records nest the uid under metadata; tombstones flatten
            # their extra keys alongside the generation fence.
            uid: object = None
            if isinstance(value, dict):
                metadata = value.get("metadata")
                uid = metadata.get("management_uid") if isinstance(metadata, dict) else value.get("management_uid")
            return uid if isinstance(uid, str) and uid else None

        live = {uid for value in cells.values() if (uid := record_uid(value))}
        candidates: set[str] = set()
        for value in tombstones.values():
            uid = record_uid(value)
            if uid is not None and uid not in live:
                candidates.add(uid)
        if not candidates:
            return
        try:
            managed = sweep(project=project)
        except (CapacityProviderError, OSError):
            return
        for uid, _secret_id in managed:
            if uid not in candidates:
                continue
            try:
                self.native_worker_secrets.delete(project=project, management_uid=uid)
            except (CapacityProviderError, OSError):
                return

    def _native_cell_remains_for_network(self, released: FleetCellRecord, network: FirewallNetworkReference) -> bool:
        """Return whether a registered native cell still needs rules for ``network``."""

        from eyvan.capacity.providers.gcp.native_worker_ingress import FirewallNetworkReference
        from eyvan.capacity.providers.gcp.tpu_nodes import OPTION_ENDPOINT_MODE, OPTION_SSH_ENTRY

        try:
            for other in self.state.load().values():
                if other.name == released.name or other.project != released.project:
                    continue
                if other.metadata.get("capacity_kind") == "attach":
                    # An attached TPU reaches its worker through this same
                    # profile rule (``eyvan gcp connect`` adds the tag), but its
                    # record never carries execution_backend. Releasing the last
                    # provisioned cell must not delete the attached TPU's rule.
                    # Its VPC is not persisted, so a public, API-reached attach
                    # claim in this project keeps the rule -- the safe direction.
                    if other.metadata.get(OPTION_ENDPOINT_MODE) == "public" and not other.metadata.get(OPTION_SSH_ENTRY):
                        return True
                    continue
                if other.metadata.get("execution_backend") != "eyvan-native":
                    continue
                if other.state in (CapacityStatus.DELETING, CapacityStatus.RELEASED):
                    continue
                # Read the owning kind's config; legacy records may lack the
                # kind marker. A normalized null network means the default VPC,
                # not the absence of a remaining ingress user.
                kind = other.metadata.get("capacity_kind")
                if kind == "compute" or (kind is None and "compute_config" in other.metadata):
                    config = other.metadata.get("compute_config")
                elif kind is None or kind == "tpu":
                    config = other.metadata.get("tpu_config")
                else:
                    return True
                if not isinstance(config, dict):
                    return True
                configured_network = config.get("network")
                if configured_network is not None and not isinstance(configured_network, str):
                    return True
                other_network = FirewallNetworkReference.from_config(
                    capacity_project=other.project, network=configured_network
                )
                if other_network.canonical_name == network.canonical_name:
                    return True
        except (CapacityProviderError, OSError, ValueError):
            # An unreadable sibling must not cause a deletion: treat it as a
            # remaining user and keep the rule.
            return True
        return False

    @staticmethod
    def _released(cell_id: str, generation: int) -> CellObservation:
        return CellObservation(
            cell_id=cell_id,
            state=CapacityStatus.RELEASED,
            generation=generation,
            observed_at=datetime.now(UTC),
        )


class GcpComputeCapacityDriver(GcpTpuCapacityDriver):
    """Reconcile CPU, GPU, and integrated TPU Compute instances behind Eyvan fences."""

    provider_name = "gcp-compute"
    capacity_kind = "compute"
    accelerator_types = ("cpu", "gpu", "tpu")
    capacity_classes = ("spot", "on-demand", "reserved")

    def capabilities(self) -> CellCapabilities:
        """Describe Compute-backed capacity with this backend's own names.

        Inherited otherwise from the TPU driver, whose ``capacity_backends``
        would advertise TPU queueing this driver cannot do.
        """

        return replace(super().capabilities(), capacity_backends=("gcp", "gcp-compute", "compute"))

    def __init__(
        self,
        state: FleetStateStore | None = None,
        *,
        provider: CapacityProvider | None = None,
        execution_backend: str | None = None,
        native_worker_port: int | None = None,
        native_worker_package: str | None = None,
        native_worker_package_sha256: str | None = None,
        native_worker_package_generation: str | None = None,
        native_worker_wheelhouse: str | None = None,
        native_worker_wheelhouse_sha256: str | None = None,
        native_worker_wheelhouse_generation: str | None = None,
        native_worker_state_dir: str | None = None,
        native_worker_shared_secret: str | None = None,
        native_worker_secrets: NativeWorkerSecretStore | None = None,
        native_worker_service_account_resolver: ServiceAccountResolver | None = None,
        native_worker_artifact_root_uri: str | None = None,
        native_worker_artifact_access: NativeWorkerArtifactAccessStore | None = None,
        native_worker_public_ingress: NativeWorkerPublicIngress | None = None,
    ):
        super().__init__(
            state,
            provider=provider or GcloudComputeProvider(),
            execution_backend=execution_backend,
            native_worker_port=native_worker_port,
            native_worker_package=native_worker_package,
            native_worker_package_sha256=native_worker_package_sha256,
            native_worker_package_generation=native_worker_package_generation,
            native_worker_wheelhouse=native_worker_wheelhouse,
            native_worker_wheelhouse_sha256=native_worker_wheelhouse_sha256,
            native_worker_wheelhouse_generation=native_worker_wheelhouse_generation,
            native_worker_state_dir=native_worker_state_dir,
            native_worker_shared_secret=native_worker_shared_secret,
            native_worker_secrets=native_worker_secrets,
            native_worker_service_account_resolver=native_worker_service_account_resolver,
            native_worker_artifact_root_uri=native_worker_artifact_root_uri,
            native_worker_artifact_access=native_worker_artifact_access,
            native_worker_public_ingress=native_worker_public_ingress,
        )
        self.native_worker_secrets = native_worker_secrets or GcloudNativeWorkerSecretStore()

    def _normalize_desired(self, desired: tp.Mapping[str, JsonValue]) -> NormalizedCapacityRequest:
        runtime = _native_runtime_from_desired(desired)
        document: JsonObject = dict(desired)
        if runtime.vm_image is not None:
            nested_values = [name for name in ("provider_config", "provider_options") if document.get(name) is not None]
            if len(nested_values) > 1:
                raise ValueError("set only one of provider_config or provider_options")
            option_name = nested_values[0] if nested_values else "provider_options"
            raw_options = document.get(option_name) or {}
            if not is_json_object(raw_options):
                raise ValueError("Compute provider options must be a mapping")
            options: JsonObject = dict(raw_options)
            configured_image = options.get("image")
            if configured_image is not None and configured_image != runtime.vm_image:
                raise ValueError("runtime.vm_image conflicts with Compute provider option 'image'")
            if configured_image is None and any(key in options for key in ("image_family", "image_project")):
                raise ValueError("runtime.vm_image conflicts with Compute image_family or image_project")
            options["image"] = runtime.vm_image
            document[option_name] = options
        config = normalize_compute_config(document)
        capacity_class_value = desired.get("capacity_class", "on-demand")
        if not isinstance(capacity_class_value, str):
            raise ValueError("capacity field 'capacity_class' must be a string")
        capacity_class = capacity_class_value.replace("_", "-")
        if capacity_class == "reserved" and config["reservation"] is None:
            raise ValueError("reserved Compute capacity requires provider option 'reservation'")
        accelerator_type = config["gpu_type"] or desired.get("accelerator_type") or config["tpu_type"]
        if accelerator_type is not None and not isinstance(accelerator_type, str):
            raise ValueError("capacity field 'accelerator_type' must be a string when set")
        document["capacity_class"] = capacity_class
        document["accelerator_type"] = accelerator_type or config["machine_type"]
        self._validate_normalized_document(document, runtime)
        return NormalizedCapacityRequest(document=document, runtime=runtime, compute_config=config)

    def _persist_provider_config(
        self,
        record: FleetCellRecord,
        desired: NormalizedCapacityRequest,
    ) -> None:
        config = desired.compute_config
        if config is None:
            raise ValueError("normalized Compute provider configuration is missing")
        persisted_config = config.copy()
        if self.execution_backend == "eyvan-native":
            setup_script = persisted_config["startup_script"]
            if not desired.runtime.is_empty and setup_script is not None:
                raise ValueError("use runtime.init_commands instead of provider_options.startup_script")
            record.metadata["native_worker_setup_script"] = setup_script
            record.metadata["native_runtime"] = model_json(desired.runtime)
            record.metadata["native_worker_port"] = self.native_worker_port
            record.metadata["execution_backend"] = self.execution_backend
            record.metadata.pop("native_worker_secret_id", None)
            record.metadata.pop("native_worker_bootstrap_generation", None)
            record.metadata.pop("native_worker_bootstrap_key", None)
            record.metadata.pop("native_worker_endpoint_mode", None)
        else:
            record.metadata.pop("native_worker_port", None)
            record.metadata.pop("execution_backend", None)
            record.metadata.pop("native_worker_secret_id", None)
            record.metadata.pop("native_worker_setup_script", None)
            record.metadata.pop("native_runtime", None)
            record.metadata.pop("native_worker_endpoint_mode", None)
        record.runtime_version = None
        record.metadata["compute_config"] = persisted_config

    def _finalize_provider_config(self, cell_id: str, context: MutationContext) -> None:
        if self.execution_backend != "eyvan-native":
            return
        record = self.state.get(cell_id)
        if record is None:
            self._fence_missing_cell(cell_id, context, phase="native worker bootstrap could begin")
        if record.generation != context.generation or record.fencing_token != context.fencing_token:
            raise StaleMutationError(f"cell {cell_id!r} changed ownership before native worker bootstrap")
        if record.metadata.get("native_worker_bootstrap_key") == self._bootstrap_key(record):
            return
        config = record.metadata.get("compute_config")
        if not isinstance(config, dict):
            raise ValueError("native worker bootstrap requires a persisted Compute provider configuration")
        setup_script = record.metadata.get("native_worker_setup_script")
        if setup_script is not None and not isinstance(setup_script, str):
            raise ValueError("native worker bootstrap has an invalid setup script")
        runtime = _native_runtime_from_record(record)
        configured = normalize_compute_config(config)
        endpoint_mode = self._configure_native_worker(
            record,
            configured,
            before_start=setup_script,
            runtime_setup=_native_runtime_setup(runtime),
            runtime_env=runtime.env,
            runtime_container=runtime.container,
            container_artifact_mounts=tuple(artifact.destination for artifact in runtime.artifacts),
            runtime_python=runtime.python,
        )

        def persist(current: FleetCellRecord) -> None:
            if current.generation != context.generation or current.fencing_token != context.fencing_token:
                raise StaleMutationError(f"cell {cell_id!r} changed ownership before native worker bootstrap")
            if current.provider_generation != record.provider_generation:
                raise StaleMutationError(f"cell {cell_id!r} changed provider generation before native worker bootstrap")
            current.metadata["compute_config"] = configured
            current.metadata["native_worker_port"] = self.native_worker_port
            current.metadata["execution_backend"] = self.execution_backend
            current.metadata["native_worker_secret_id"] = record.metadata["native_worker_secret_id"]
            current.metadata["native_worker_bootstrap_generation"] = record.provider_generation
            current.metadata["native_worker_bootstrap_key"] = self._bootstrap_key(record)
            _persist_native_worker_endpoint_mode(current, endpoint_mode)

        self._persist_owned_cell(
            cell_id,
            context,
            record,
            phase="native worker bootstrap could be persisted",
            cleanup_capacity=False,
            mutate=persist,
        )

    def _configure_native_worker(
        self,
        record: FleetCellRecord,
        config: ComputeProviderConfig,
        *,
        before_start: str | None,
        runtime_setup: str | None = None,
        runtime_env: tp.Mapping[str, str] | None = None,
        runtime_container: NativeContainerConfig | None = None,
        container_artifact_mounts: tp.Sequence[str] = (),
        runtime_python: str | None = None,
    ) -> NativeWorkerEndpointMode:
        """Inject a private worker bootstrap without writing its token to VM metadata."""

        from eyvan.worker.runtime.native_bootstrap import build_native_worker_startup_script, derive_native_worker_token

        if not isinstance(self.native_worker_shared_secret, str) or not self.native_worker_shared_secret.strip():
            raise ValueError("eyvan-native Compute capacity requires EYVAN_NATIVE_WORKER_ROOT_SECRET")
        if not isinstance(self.native_worker_package, str) or not self.native_worker_package.strip():
            raise ValueError(
                "eyvan-native Compute capacity requires EYVAN_NATIVE_WORKER_PACKAGE; "
                "use a gs:// wheel with its SHA-256 and object generation"
            )
        if not isinstance(record.project, str) or not record.project:
            raise ValueError("native worker bootstrap requires a GCP project")
        management_uid = record.metadata.get("management_uid")
        if not isinstance(management_uid, str) or not management_uid:
            raise ValueError("native worker bootstrap requires an Eyvan management UID")
        service_account = self._native_worker_service_account(record, config)
        egress = config.get("bootstrap_egress")
        if not isinstance(egress, str) or not egress:
            raise ValueError(
                "eyvan-native Compute capacity requires provider_options.bootstrap_egress "
                f"(one of: {bootstrap_egress_choices()})"
            )
        if config["internal_ip_only"] and egress == "public-internet":
            raise ValueError("internal-only eyvan-native Compute capacity cannot use bootstrap_egress=public-internet")
        endpoint_mode = self._configure_native_worker_public_ingress(
            record,
            config,
            has_public_ip=not config["internal_ip_only"],
        )
        wheelhouse = None
        if egress == "private-google-access":
            if not self.native_worker_package.startswith("gs://"):
                raise ValueError(
                    "private-google-access native bootstrap requires "
                    "EYVAN_NATIVE_WORKER_PACKAGE to be a verified gs:// wheel"
                )
            if not isinstance(self.native_worker_wheelhouse, str) or not self.native_worker_wheelhouse.strip():
                raise ValueError(
                    "private-google-access eyvan-native Compute capacity requires EYVAN_NATIVE_WORKER_WHEELHOUSE "
                    "with every worker dependency wheel"
                )
            if not self.native_worker_wheelhouse.startswith("gs://"):
                raise ValueError("EYVAN_NATIVE_WORKER_WHEELHOUSE must be a gs:// zip archive")
            wheelhouse = self.native_worker_wheelhouse

        scopes = list(config["scopes"])
        cloud_platform_scope = "https://www.googleapis.com/auth/cloud-platform"
        if cloud_platform_scope not in scopes:
            scopes.append(cloud_platform_scope)
        config["scopes"] = scopes
        self._ensure_native_worker_artifact_access(record, service_account)
        worker_token = derive_native_worker_token(
            self.native_worker_shared_secret,
            management_uid,
            provider_generation=record.provider_generation,
        )
        if worker_token is None:
            raise ValueError("eyvan-native Compute capacity requires EYVAN_NATIVE_WORKER_ROOT_SECRET")
        secret_resource = self.native_worker_secrets.ensure(
            project=record.project,
            management_uid=management_uid,
            value=worker_token,
            reader_service_account=service_account,
        )
        config["startup_script"] = build_native_worker_startup_script(
            package=self.native_worker_package,
            package_sha256=self.native_worker_package_sha256,
            package_generation=self.native_worker_package_generation,
            port=self.native_worker_port,
            state_dir=self.native_worker_state_dir,
            worker_secret_resource=secret_resource,
            wheelhouse=wheelhouse,
            wheelhouse_sha256=self.native_worker_wheelhouse_sha256,
            wheelhouse_generation=self.native_worker_wheelhouse_generation,
            before_start=before_start,
            runtime_setup=runtime_setup,
            runtime_env=runtime_env,
            runtime_container=runtime_container,
            container_artifact_mounts=container_artifact_mounts,
            runtime_python=runtime_python,
        )
        record.metadata["native_worker_port"] = self.native_worker_port
        record.metadata["execution_backend"] = self.execution_backend
        record.metadata["native_worker_secret_id"] = secret_resource.rsplit("/versions/", 1)[0]
        return endpoint_mode

    def _plan_recreate(self, record: FleetCellRecord, context: MutationContext) -> FleetCellRecord:
        replacement = super()._plan_recreate(record, context)
        if self.execution_backend != "eyvan-native":
            return replacement
        config = replacement.metadata.get("compute_config")
        if not isinstance(config, dict):
            raise ValueError("native worker recreation requires a persisted Compute provider configuration")
        refreshed = normalize_compute_config(config)
        setup_script = replacement.metadata.get("native_worker_setup_script")
        if setup_script is not None and not isinstance(setup_script, str):
            raise ValueError("native worker recreation has an invalid setup script")
        runtime = _native_runtime_from_record(replacement)
        endpoint_mode = self._configure_native_worker(
            replacement,
            refreshed,
            before_start=setup_script,
            runtime_setup=_native_runtime_setup(runtime),
            runtime_env=runtime.env,
            runtime_container=runtime.container,
            container_artifact_mounts=tuple(artifact.destination for artifact in runtime.artifacts),
            runtime_python=runtime.python,
        )

        def persist(current: FleetCellRecord) -> None:
            if current.generation != context.generation or current.fencing_token != context.fencing_token:
                raise StaleMutationError(f"cell {record.name!r} changed ownership before native worker recreation")
            current.metadata["compute_config"] = refreshed
            current.metadata["native_worker_port"] = self.native_worker_port
            current.metadata["execution_backend"] = self.execution_backend
            current.metadata["native_worker_secret_id"] = replacement.metadata["native_worker_secret_id"]
            current.metadata["native_worker_bootstrap_generation"] = replacement.provider_generation
            current.metadata.pop("native_worker_bootstrap_key", None)
            _persist_native_worker_endpoint_mode(current, endpoint_mode)

        return self._persist_owned_cell(
            replacement.name,
            context,
            replacement,
            phase="native worker recreation could be persisted",
            cleanup_capacity=True,
            mutate=persist,
        )

    def _release_auxiliary_resources(self, record: FleetCellRecord) -> None:
        if self.execution_backend != "eyvan-native" or not record.project:
            return
        management_uid = record.metadata.get("management_uid")
        if not isinstance(management_uid, str) or not management_uid:
            raise ValueError("native worker cleanup requires an Eyvan management UID")
        self.native_worker_secrets.delete(project=record.project, management_uid=management_uid)
        self._release_unused_native_ingress_rules(record)

    @staticmethod
    def _resource_id(cell_id: str, provider_generation: int) -> str:
        return compute_instance_name(cell_id, provider_generation)


class GcpCapacityDriver:
    """Route a mixed GCP fleet to TPU or Compute capacity per desired cell."""

    _MODE_ALIASES: tp.ClassVar[dict[str, str]] = {
        "gcp": "auto",
        "auto": "auto",
        "gcp-tpu": "tpu",
        "tpu": "tpu",
        "gcp-compute": "compute",
        "compute": "compute",
        "gcp-attach": "attach",
        "attach": "attach",
    }

    def __init__(
        self,
        state: FleetStateStore | None = None,
        *,
        tpu_provider: CapacityProvider | None = None,
        compute_provider: CapacityProvider | None = None,
        execution_backend: str | None = None,
        native_worker_port: int | None = None,
        native_worker_package: str | None = None,
        native_worker_package_sha256: str | None = None,
        native_worker_package_generation: str | None = None,
        native_worker_wheelhouse: str | None = None,
        native_worker_wheelhouse_sha256: str | None = None,
        native_worker_wheelhouse_generation: str | None = None,
        native_worker_state_dir: str | None = None,
        native_worker_shared_secret: str | None = None,
        native_worker_secrets: NativeWorkerSecretStore | None = None,
        native_worker_service_account_resolver: ServiceAccountResolver | None = None,
        native_worker_artifact_root_uri: str | None = None,
        native_worker_artifact_access: NativeWorkerArtifactAccessStore | None = None,
        native_worker_public_ingress: NativeWorkerPublicIngress | None = None,
        mode: str = "auto",
        attach_api: tp.Any = None,
    ) -> None:
        from eyvan.capacity.drivers.gcp_attach import GcpAttachCapacityDriver

        settings = EyvanSettings.from_env()
        self.state = state or FleetStateStore.from_uri(settings.fleet_state_uri)
        normalized_mode = mode.strip().lower().replace("_", "-")
        try:
            self.mode = self._MODE_ALIASES[normalized_mode]
        except KeyError:
            choices = ", ".join(sorted(name for name in self._MODE_ALIASES if name.startswith("gcp")))
            raise ValueError(f"unknown GCP capacity mode {mode!r}; available modes: {choices}") from None
        selected_execution = execution_backend or settings.execution_backend
        self.tpu = GcpTpuCapacityDriver(
            self.state,
            provider=tpu_provider,
            execution_backend=selected_execution,
            native_worker_port=native_worker_port,
            native_worker_package=native_worker_package,
            native_worker_package_sha256=native_worker_package_sha256,
            native_worker_package_generation=native_worker_package_generation,
            native_worker_wheelhouse=native_worker_wheelhouse,
            native_worker_wheelhouse_sha256=native_worker_wheelhouse_sha256,
            native_worker_wheelhouse_generation=native_worker_wheelhouse_generation,
            native_worker_state_dir=native_worker_state_dir,
            native_worker_shared_secret=native_worker_shared_secret,
            native_worker_secrets=native_worker_secrets,
            native_worker_service_account_resolver=native_worker_service_account_resolver,
            native_worker_artifact_root_uri=native_worker_artifact_root_uri,
            native_worker_artifact_access=native_worker_artifact_access,
            native_worker_public_ingress=native_worker_public_ingress,
        )
        self.compute = GcpComputeCapacityDriver(
            self.state,
            provider=compute_provider,
            execution_backend=selected_execution,
            native_worker_port=native_worker_port,
            native_worker_package=native_worker_package,
            native_worker_package_sha256=native_worker_package_sha256,
            native_worker_package_generation=native_worker_package_generation,
            native_worker_wheelhouse=native_worker_wheelhouse,
            native_worker_wheelhouse_sha256=native_worker_wheelhouse_sha256,
            native_worker_wheelhouse_generation=native_worker_wheelhouse_generation,
            native_worker_state_dir=native_worker_state_dir,
            native_worker_shared_secret=native_worker_shared_secret,
            native_worker_secrets=native_worker_secrets,
            native_worker_service_account_resolver=native_worker_service_account_resolver,
            native_worker_artifact_root_uri=native_worker_artifact_root_uri,
            native_worker_artifact_access=native_worker_artifact_access,
            native_worker_public_ingress=native_worker_public_ingress,
        )
        self.attach = GcpAttachCapacityDriver(
            self.state,
            api=attach_api,
            native_worker_port=native_worker_port if native_worker_port is not None else settings.native_worker_port,
            native_worker_public_ingress=native_worker_public_ingress,
        )
        self.execution_backend = self.attach.execution_backend if self.mode == "attach" else self.tpu.execution_backend
        # Delegated rather than re-read: a router that disagreed with the
        # driver it routes to would admit exec the backend then refuses.
        self.supports_exec = self.tpu.supports_exec

    def capabilities(self) -> CellCapabilities:
        if self.mode == "tpu":
            return self.tpu.capabilities()
        if self.mode == "compute":
            return self.compute.capabilities()
        if self.mode == "attach":
            return self.attach.capabilities()
        # Only "auto" reaches here: the pinned modes returned above.
        backends = ("gcp", "gcp-tpu", "gcp-compute", "gcp-attach", "tpu", "compute")
        return CellCapabilities(
            provider="gcp",
            executor=self.execution_backend,
            accelerator_types=("cpu", "gpu", "tpu"),
            capacity_classes=("spot", "on-demand", "reserved", "guaranteed"),
            capacity_backends=backends,
            supports_callables=True,
            max_gang_replicas=1 if self.execution_backend in {"native-local", "eyvan-native"} else None,
            # Stated rather than defaulted: an operator turns this on for a fleet
            # they are willing to have diagnostics run on. Advertise that policy
            # rather than silently disabling the runtime's exec capability.
            supports_exec=self.supports_exec,
            resource_enforcement="none" if self.execution_backend in {"native-local", "eyvan-native"} else "backend",
        )

    @staticmethod
    def _nested_options(desired: tp.Mapping[str, JsonValue]) -> JsonObject:
        for key in ("provider_config", "provider_options"):
            value = desired.get(key)
            if is_json_object(value):
                return value
        return {}

    def _desired_kind(self, desired: tp.Mapping[str, JsonValue]) -> str:
        backend_value = desired.get("capacity_backend")
        if backend_value is not None and not isinstance(backend_value, str):
            raise ValueError("capacity field 'capacity_backend' must be a string when set")
        requested_backend = (backend_value or "gcp").strip().lower().replace("_", "-")
        if requested_backend in {"gcp-tpu", "tpu"}:
            selected = "tpu"
        elif requested_backend == "gcp-attach":
            selected = "attach"
        elif requested_backend in {"gcp-compute", "compute"}:
            selected = "compute"
        elif requested_backend != "gcp":
            raise ValueError(f"GCP driver cannot satisfy capacity backend {requested_backend!r}")
        else:
            kind_value = desired.get("capacity_kind") or desired.get("kind")
            if kind_value is not None and not isinstance(kind_value, str):
                raise ValueError("capacity field 'capacity_kind' must be a string when set")
            requested_kind = (kind_value or "").strip().lower()
            if requested_kind in {"cpu", "gpu", "compute"}:
                selected = "compute"
            elif requested_kind == "tpu":
                selected = "tpu"
            else:
                options = self._nested_options(desired)
                accelerator_value = desired.get("accelerator_type")
                if accelerator_value is not None and not isinstance(accelerator_value, str):
                    raise ValueError("capacity field 'accelerator_type' must be a string when set")
                accelerator_type = (accelerator_value or "").lower()
                compute_keys = (
                    "machine_type",
                    "gpu_type",
                    "gpu_count",
                    "tpu_type",
                    "tpu_count",
                    "image",
                    "image_family",
                )
                selected = (
                    "compute"
                    if accelerator_type in {"cpu", "gpu"}
                    or accelerator_type.startswith("nvidia-")
                    or any(key in desired or key in options for key in compute_keys)
                    else "tpu"
                )
        if self.mode != "auto" and selected != self.mode:
            raise ValueError(f"gcp-{self.mode} agent cannot satisfy {selected!r} capacity")
        return selected

    def _record_kind(self, cell_id: str) -> str:
        """Route record-backed operations to the driver that owns the cell.

        ``capacity_kind`` is stamped by the owning driver at apply time and
        cross-kind takeover is refused, so in practice the first check
        decides. The remaining markers are redundancy against a record whose
        metadata was damaged after apply: ``capacity_backend``, then the
        persisted provider-config key each kind writes. The final default is
        deliberate -- creates never consult this method (they route from the
        requested spec), TPU was the only kind before markers existed, and
        refusing to route an ambiguous record would leave it impossible to
        observe or release.
        """
        if self.mode != "auto":
            return self.mode
        record = self.state.get(cell_id)
        if record is None:
            tombstone = self.state.cell_tombstone(cell_id)
            if tombstone is not None and tombstone.get("capacity_kind") == "attach":
                return "attach"
            return "tpu"
        kind = record.metadata.get("capacity_kind")
        if kind in {"tpu", "compute", "attach"}:
            return str(kind)
        backend = record.metadata.get("capacity_backend")
        if isinstance(backend, str) and len(backend.strip()) > 0:
            normalized = backend.strip().lower().replace("_", "-")
            if normalized == "gcp-compute":
                return "compute"
            if normalized == "gcp-tpu":
                return "tpu"
            if normalized == "gcp-attach":
                return "attach"
        if "compute_config" in record.metadata:
            return "compute"
        if "tpu_config" in record.metadata:
            return "tpu"
        return "tpu"

    def _driver(self, kind: str) -> tp.Any:
        if kind == "attach":
            return self.attach
        return self.compute if kind == "compute" else self.tpu

    def observe(self, cell_id: str) -> CellObservation:
        return self._driver(self._record_kind(cell_id)).observe(cell_id)

    def reconcile(self, cell_id: str, context: MutationContext) -> CellObservation:
        return self._driver(self._record_kind(cell_id)).reconcile(cell_id, context)

    def apply_capacity(
        self,
        cell_id: str,
        desired: tp.Mapping[str, JsonValue],
        context: MutationContext,
    ) -> CellObservation:
        return self._driver(self._desired_kind(desired)).apply_capacity(cell_id, desired, context)

    def drain(self, cell_id: str, context: MutationContext) -> CellObservation:
        return self._driver(self._record_kind(cell_id)).drain(cell_id, context)

    def release(self, cell_id: str, context: MutationContext) -> CellObservation:
        return self._driver(self._record_kind(cell_id)).release(cell_id, context)


__all__ = ("GcpCapacityDriver", "GcpComputeCapacityDriver", "GcpTpuCapacityDriver")
