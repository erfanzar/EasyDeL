# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Direct gcloud integration for TPU queued-resource capacity."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import typing as tp
from dataclasses import dataclass, field

from eyvan._internal.compat import StrEnum
from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object, is_json_value
from eyvan._internal.network import format_http_ip_endpoint, normalize_ip_address
from eyvan._internal.subprocess_env import provider_cli_env
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import (
    CapacityProviderError,
    CapacityProviderFailureCode,
    provider_failure_from_diagnostic,
)
from eyvan.capacity.fleet_state import FleetCellRecord
from eyvan.capacity.hardware_catalog import shape_for
from eyvan.capacity.providers.gcp.native_worker_ingress import NativeWorkerEndpointMode, endpoint_mode_from_metadata

RUNTIME_VERSION_BY_FAMILY = {
    "v4": "tpu-ubuntu2204-base",
    "v5e": "v2-alpha-tpuv5-lite",
    "v5litepod": "v2-alpha-tpuv5-lite",
    "v5p": "v2-alpha-tpuv5",
    "v6e": "v2-alpha-tpuv6e",
}
#: How a native worker VM reaches Eyvan's artifacts and control plane. There is
#: no default: each choice needs infrastructure the operator has to have built
#: (a Cloud NAT gateway, Private Google Access on the subnet, or an external
#: address), so guessing one produces a VM that provisions and then never
#: registers. Also the canonical source for the Compute path and the fleet
#: schema, which used to keep their own copies of this set.
BOOTSTRAP_EGRESS_MODES: tuple[str, ...] = ("cloud-nat", "private-google-access", "public-internet")


def bootstrap_egress_choices() -> str:
    """Render the legal ``bootstrap_egress`` values for an operator-facing message."""

    return ", ".join(BOOTSTRAP_EGRESS_MODES)


class GcpQueuedResourceState(StrEnum):
    """Cloud TPU queued-resource states accepted at the gcloud boundary."""

    STATE_UNSPECIFIED = "STATE_UNSPECIFIED"
    CREATING = "CREATING"
    ACCEPTED = "ACCEPTED"
    PROVISIONING = "PROVISIONING"
    FAILED = "FAILED"
    DELETING = "DELETING"
    ACTIVE = "ACTIVE"
    SUSPENDING = "SUSPENDING"
    SUSPENDED = "SUSPENDED"
    WAITING_FOR_RESOURCES = "WAITING_FOR_RESOURCES"


class GcpTpuNodeState(StrEnum):
    """Cloud TPU node states accepted at the gcloud boundary."""

    STATE_UNSPECIFIED = "STATE_UNSPECIFIED"
    CREATING = "CREATING"
    READY = "READY"
    RESTARTING = "RESTARTING"
    REIMAGING = "REIMAGING"
    DELETING = "DELETING"
    REPAIRING = "REPAIRING"
    STOPPED = "STOPPED"
    STOPPING = "STOPPING"
    STARTING = "STARTING"
    PREEMPTED = "PREEMPTED"
    TERMINATED = "TERMINATED"
    HIDING = "HIDING"
    HIDDEN = "HIDDEN"
    UNHIDING = "UNHIDING"
    UNKNOWN = "UNKNOWN"  # Exact Cloud TPU wire value, never an Eyvan fallback.


TERMINAL_STATES = frozenset(
    {
        GcpQueuedResourceState.FAILED,
        GcpQueuedResourceState.DELETING,
        GcpQueuedResourceState.SUSPENDING,
        GcpQueuedResourceState.SUSPENDED,
    }
)
PENDING_STATES = frozenset(
    {
        GcpQueuedResourceState.ACCEPTED,
        GcpQueuedResourceState.CREATING,
        GcpQueuedResourceState.WAITING_FOR_RESOURCES,
    }
)
TERMINAL_NODE_STATES = frozenset(
    {
        GcpTpuNodeState.DELETING,
        GcpTpuNodeState.STOPPED,
        GcpTpuNodeState.STOPPING,
        GcpTpuNodeState.PREEMPTED,
        GcpTpuNodeState.TERMINATED,
        GcpTpuNodeState.HIDING,
        GcpTpuNodeState.HIDDEN,
    }
)
#: Terminal node states that GCP moves a *reclaimed spot* TPU through.
#:
#: ``PREEMPTED`` is the honest signal, but it is transient: GCP deletes the node
#: it reclaimed, so whether Eyvan ever sees ``PREEMPTED`` is a race between the
#: reclamation and the reconcile interval. ``DELETING`` can be the first state
#: observed after Spot reclamation and must follow the same recovery path.
SPOT_RECLAMATION_NODE_STATES = frozenset(
    {
        GcpTpuNodeState.PREEMPTED,
        GcpTpuNodeState.DELETING,
    }
)
_TPU_OPTION_KEYS = frozenset(
    {
        "bootstrap_egress",
        "internal_ips",
        "metadata",
        "network",
        "node_count",
        "node_prefix",
        "runtime_version",
        "scopes",
        "service_account",
        "startup_script",
        "subnetwork",
        "tags",
    }
)


class TpuProviderConfig(tp.TypedDict):
    """Validated provider options used to create one TPU queued resource."""

    runtime_version: str | None
    network: str | None
    subnetwork: str | None
    internal_ips: bool
    service_account: str | None
    scopes: list[str]
    tags: list[str]
    metadata: dict[str, str]
    node_count: tp.NotRequired[int]
    node_prefix: tp.NotRequired[str]
    startup_script: tp.NotRequired[str]
    bootstrap_egress: tp.NotRequired[str]


ProviderOptionValue: tp.TypeAlias = JsonValue


class GcloudError(CapacityProviderError):
    """A gcloud command failed with its diagnostic preserved."""


class GcloudQueueAdmissionDeniedError(GcloudError):
    """GCP rejected a queued-resource request before it entered its capacity queue."""

    def __init__(self, diagnostic: str) -> None:
        super().__init__(diagnostic, code=CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED)


def default_runtime_version(accelerator_type: str) -> str:
    family = accelerator_type.split("-", 1)[0].lower()
    try:
        return RUNTIME_VERSION_BY_FAMILY[family]
    except KeyError:
        raise ValueError(
            f"no default TPU runtime is known for {accelerator_type!r}; set runtime_version explicitly"
        ) from None


def _string(value: ProviderOptionValue, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"TPU option {name!r} must be a non-empty trimmed string")
    return value


def _script(value: ProviderOptionValue, name: str, *, optional: bool = False) -> str | None:
    """Validate a non-empty script while preserving its intentional whitespace."""

    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"TPU option {name!r} must be a non-empty NUL-free script")
    return value


def _string_sequence(value: ProviderOptionValue, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"TPU option {name!r} must be a list of strings")
    result: list[str] = []
    for item in value:
        parsed = _string(item, name)
        if parsed is None:
            raise AssertionError(f"required TPU option {name!r} passed validation without a value")
        result.append(parsed)
    return result


def _string_mapping(value: ProviderOptionValue, name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not is_json_object(value):
        raise ValueError(f"TPU option {name!r} must be a string mapping")
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not raw_key.strip() or raw_key != raw_key.strip():
            raise ValueError(f"TPU option {name!r} keys must be non-empty trimmed strings")
        item = _string(raw_value, f"{name}[{raw_key}]")
        if item is None:
            raise AssertionError(f"required TPU option {name!r} passed validation without a value")
        result[raw_key] = item
    return result


def normalize_tpu_config(desired: tp.Mapping[str, ProviderOptionValue]) -> TpuProviderConfig:
    """Validate and normalize provider options for one TPU queued resource."""

    nested_values = [
        desired.get(name) for name in ("provider_config", "provider_options") if desired.get(name) is not None
    ]
    if len(nested_values) > 1:
        raise ValueError("set only one of provider_config or provider_options")
    nested = nested_values[0] if nested_values else {}
    if not is_json_object(nested):
        raise ValueError("TPU provider options must be a mapping")
    unknown = sorted(str(key) for key in nested if key not in _TPU_OPTION_KEYS)
    if unknown:
        # Naming the accepted keys costs one line here and removes a round trip
        # through the source for whoever typed the wrong one.
        raise ValueError(
            f"unknown TPU provider option: {unknown[0]}; accepted keys are {', '.join(sorted(_TPU_OPTION_KEYS))}"
        )
    options: dict[str, ProviderOptionValue] = dict(nested)
    for key in _TPU_OPTION_KEYS:
        if key in desired:
            options[key] = desired[key]

    internal_ips = options.get("internal_ips", False)
    if not isinstance(internal_ips, bool):
        raise ValueError("TPU option 'internal_ips' must be a boolean")
    node_count = options.get("node_count", 1)
    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 1:
        raise ValueError("TPU option 'node_count' must be a positive integer")
    node_prefix = _string(options.get("node_prefix"), "node_prefix", optional=True)
    if node_count == 1 and node_prefix is not None:
        raise ValueError("TPU option 'node_prefix' requires node_count greater than one")
    metadata = _string_mapping(options.get("metadata"), "metadata")
    reserved_metadata = sorted(key for key in metadata if key.lower().startswith("eyvan-"))
    if reserved_metadata:
        raise ValueError(f"TPU metadata key {reserved_metadata[0]!r} is reserved by Eyvan")

    normalized: TpuProviderConfig = {
        "runtime_version": _string(options.get("runtime_version"), "runtime_version", optional=True),
        "network": _string(options.get("network"), "network", optional=True),
        "subnetwork": _string(options.get("subnetwork"), "subnetwork", optional=True),
        "internal_ips": internal_ips,
        "service_account": _string(options.get("service_account"), "service_account", optional=True),
        "scopes": _string_sequence(options.get("scopes"), "scopes"),
        "tags": _string_sequence(options.get("tags"), "tags"),
        "metadata": metadata,
    }
    if node_count != 1:
        normalized["node_count"] = node_count
    if node_prefix is not None:
        normalized["node_prefix"] = node_prefix
    startup_script = _script(options.get("startup_script"), "startup_script", optional=True)
    if startup_script is not None:
        normalized["startup_script"] = startup_script
    bootstrap_egress = _string(options.get("bootstrap_egress"), "bootstrap_egress", optional=True)
    if bootstrap_egress is not None:
        bootstrap_egress = bootstrap_egress.lower().replace("_", "-")
        if bootstrap_egress not in BOOTSTRAP_EGRESS_MODES:
            raise ValueError(f"TPU option 'bootstrap_egress' must be one of: {bootstrap_egress_choices()}")
        normalized["bootstrap_egress"] = bootstrap_egress
    return normalized


def _mapping_flag(values: tp.Mapping[str, str]) -> str:
    items = [f"{key}={value}" for key, value in sorted(values.items())]
    if all("," not in item for item in items):
        return ",".join(items)
    for delimiter in ("|", ";", "~", ":"):
        if all(delimiter not in item for item in items):
            return f"^{delimiter}^{delimiter.join(items)}"
    raise ValueError("TPU metadata contains unsupported delimiter characters")


@dataclass(frozen=True, slots=True)
class QueuedResourceSpec:
    """Inputs for one TPU queued-resource request."""

    name: str
    accelerator_type: str
    zone: str
    project: str
    runtime_version: str | None = None
    capacity_class: str = "spot"
    node_id: str | None = None
    node_count: int = 1
    node_prefix: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    network: str | None = None
    subnetwork: str | None = None
    internal_ips: bool = False
    service_account: str | None = None
    scopes: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    startup_script: str | None = None
    bootstrap_egress: str | None = None

    def create_args(self, *, qr_id: str | None = None, startup_script_path: str | None = None) -> list[str]:
        if isinstance(self.node_count, bool) or not isinstance(self.node_count, int) or self.node_count < 1:
            raise ValueError("node_count must be a positive integer")
        if self.node_count == 1 and self.node_prefix is not None:
            raise ValueError("node_prefix requires node_count greater than one")
        args = [
            "compute",
            "tpus",
            "queued-resources",
            "create",
            qr_id or self.name,
            "--zone",
            self.zone,
            "--project",
            self.project,
            "--accelerator-type",
            self.accelerator_type,
            "--runtime-version",
            self.runtime_version or default_runtime_version(self.accelerator_type),
        ]
        if self.node_count == 1:
            args.extend(("--node-id", self.node_id or self.name))
        else:
            args.extend(("--node-count", str(self.node_count)))
            args.extend(("--node-prefix", self.node_prefix or qr_id or self.name))
        if self.capacity_class == "spot":
            args.append("--spot")
        elif self.capacity_class == "reserved":
            args.append("--reserved")
        elif self.capacity_class == "guaranteed":
            args.append("--guaranteed")
        elif self.capacity_class != "on-demand":
            raise ValueError(f"unsupported TPU capacity class {self.capacity_class!r}")
        if self.labels:
            args.extend(("--labels", _mapping_flag(self.labels)))
        if self.metadata:
            args.extend(("--metadata", _mapping_flag(self.metadata)))
        if startup_script_path:
            args.extend(("--metadata-from-file", f"startup-script={startup_script_path}"))
        if self.network:
            args.extend(("--network", self.network))
        if self.subnetwork:
            args.extend(("--subnetwork", self.subnetwork))
        if self.internal_ips:
            args.append("--internal-ips")
        if self.service_account:
            args.extend(("--service-account", self.service_account))
        if self.scopes:
            args.extend(("--scopes", ",".join(self.scopes)))
        if self.tags:
            args.extend(("--tags", ",".join(self.tags)))
        return args


def _queued_resource_failure(raw: JsonObject) -> str | None:
    """Return the provider's own words for a FAILED queued resource, if it gave any.

    GCP records the cause in ``state.failedData.error``. Preserve that diagnostic
    rather than reporting only that the queued resource is terminal.
    """

    state = raw.get("state")
    failed = state.get("failedData") if is_json_object(state) else None
    error = failed.get("error") if is_json_object(failed) else None
    if not is_json_object(error):
        return None
    message = error.get("message")
    if not isinstance(message, str) or not message.strip():
        return None
    code = error.get("code")
    words = " ".join(message.split())
    return f"{words} (code {code})" if isinstance(code, int) and not isinstance(code, bool) else words


@dataclass(frozen=True, slots=True)
class QueuedResource:
    qr_id: str
    state: GcpQueuedResourceState
    node_ids: tuple[str, ...]
    accelerator_type: str | None
    raw: JsonObject = field(repr=False)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


@dataclass(frozen=True, slots=True)
class ProviderObservation:
    """Normalized provider state returned to the capacity driver."""

    state: CapacityStatus
    detail: str = ""
    #: Why the provider is in this state, when it said something specific. The
    #: prose in ``detail`` is for a human; this is what the control plane routes
    #: on, so a reason it can act on never has to be recovered by reading text.
    failure_code: CapacityProviderFailureCode | None = None
    head_ip: str | None = None
    execution_endpoint: str | None = None
    worker_endpoints: tuple[str, ...] = ()
    observed_tpu_vms: int | None = None
    runtime_process_count: int | None = None


@dataclass(frozen=True, slots=True)
class TpuNodeObservation:
    """Validated state and worker addresses for one Cloud TPU node."""

    node_id: str
    state: GcpTpuNodeState
    worker_ips: tuple[str, ...]


Runner = tp.Callable[[tp.Sequence[str]], subprocess.CompletedProcess[str]]


def decode_gcloud_json(output: str, *, resource: str) -> JsonValue:
    """Decode and validate one gcloud JSON response at the subprocess boundary."""

    try:
        value = json.loads(output or "{}")
    except json.JSONDecodeError as exc:
        raise GcloudError(f"{resource} returned invalid JSON: {exc.msg}") from exc
    if not is_json_value(value):
        raise GcloudError(f"{resource} returned a value outside the JSON data model")
    return value


def decode_gcloud_object(output: str, *, resource: str) -> JsonObject:
    """Decode a gcloud response that must be a JSON object."""

    value = decode_gcloud_json(output, resource=resource)
    if not is_json_object(value):
        raise GcloudError(f"{resource} did not return a JSON object")
    return value


def _runner(args: tp.Sequence[str]) -> subprocess.CompletedProcess[str]:
    timeout_s = max(float(os.getenv("EYVAN_GCLOUD_TPU_TIMEOUT_S", "600")), 1.0)
    return subprocess.run(
        ["gcloud", *args], check=False, capture_output=True, text=True, timeout=timeout_s, env=provider_cli_env()
    )


class GcloudTpuProvider:
    """Idempotent TPU queued-resource lifecycle without a Ray dependency."""

    def __init__(self, *, runner: Runner = _runner, delete_wait_timeout_s: float = 600):
        self._runner = runner
        self.delete_wait_timeout_s = delete_wait_timeout_s

    def _run(self, args: tp.Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Run one gcloud call, surfacing its subprocess timeout as a typed provider failure."""
        try:
            return self._runner(args)
        except subprocess.TimeoutExpired as exc:
            raise GcloudError(str(exc), code=CapacityProviderFailureCode.PROVIDER_ERROR) from exc

    @staticmethod
    def _not_found(result: subprocess.CompletedProcess[str]) -> bool:
        detail = f"{result.stdout}\n{result.stderr}".lower()
        return result.returncode != 0 and ("not_found" in detail or "not found" in detail or "404" in detail)

    @staticmethod
    def _parse_qr(raw: JsonObject) -> QueuedResource:
        name_value = raw.get("name")
        name = name_value if isinstance(name_value, str) and name_value else "?"
        state_value = raw.get("state")
        if is_json_object(state_value):
            state_value = state_value.get("state")
        if not isinstance(state_value, str) or not state_value:
            raise GcloudError(f"queued resource {name!r} did not report a state")
        try:
            state = GcpQueuedResourceState(state_value.upper())
        except ValueError as exc:
            raise GcloudError(f"queued resource {name!r} reported unsupported state {state_value!r}") from exc
        tpu = raw.get("tpu")
        tpu_document = tpu if is_json_object(tpu) else {}
        spec_values = tpu_document.get("nodeSpec")
        specs = spec_values if isinstance(spec_values, list) else []
        node_ids: list[str] = []
        accelerator_type: str | None = None
        for spec_value in specs:
            if not is_json_object(spec_value):
                continue
            node_id = spec_value.get("nodeId")
            if isinstance(node_id, str) and node_id:
                node_ids.append(node_id)
            node_value = spec_value.get("node")
            if accelerator_type is None and is_json_object(node_value):
                candidate = node_value.get("acceleratorType")
                if isinstance(candidate, str) and candidate:
                    accelerator_type = candidate
        return QueuedResource(name.rsplit("/", 1)[-1], state, tuple(node_ids), accelerator_type, raw.copy())

    def describe_qr(self, qr_id: str, *, project: str, zone: str) -> QueuedResource | None:
        args = (
            "compute",
            "tpus",
            "queued-resources",
            "describe",
            qr_id,
            "--zone",
            zone,
            "--project",
            project,
            "--format=json",
        )
        result = self._run(args)
        if self._not_found(result):
            return None
        if result.returncode != 0:
            raise GcloudError((result.stderr or result.stdout).strip() or f"could not describe {qr_id}")
        value = decode_gcloud_object(result.stdout, resource=f"queued resource {qr_id!r}")
        return self._parse_qr(value)

    def list_qr_names(self, *, project: str, zone: str, prefix: str) -> tuple[str, ...]:
        """Return ids of this zone's queued resources whose name carries ``prefix``.

        This is the substrate-side leak detector: a queued resource is billed
        from the moment its provision request is admitted, so one that no
        record points at must be findable without knowing its id.
        """

        args = (
            "compute",
            "tpus",
            "queued-resources",
            "list",
            "--zone",
            zone,
            "--project",
            project,
            f"--filter=name~{prefix}",
            "--format=value(name)",
        )
        result = self._run(args)
        if result.returncode != 0:
            raise GcloudError(
                (result.stderr or result.stdout).strip() or f"could not list queued resources matching {prefix!r}"
            )
        ids: list[str] = []
        for line in result.stdout.splitlines():
            name = line.strip()
            if not name:
                continue
            # Full names look like projects/P/locations/Z/queuedResources/<id>.
            ids.append(name.rstrip("/").rsplit("/", 1)[-1])
        return tuple(ids)

    def list_cell_incarnations(self, *, project: str, zone: str, cell_id: str) -> tuple[str, ...]:
        """Return every queued-resource id this zone holds for ``cell_id``."""

        return self.list_qr_names(project=project, zone=zone, prefix=f"{cell_id}-r")

    def describe_node(self, node_id: str, *, project: str, zone: str) -> JsonObject | None:
        args = (
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            node_id,
            "--zone",
            zone,
            "--project",
            project,
            "--format=json",
        )
        result = self._run(args)
        if self._not_found(result):
            return None
        if result.returncode != 0:
            raise GcloudError((result.stderr or result.stdout).strip() or f"could not describe TPU node {node_id}")
        return decode_gcloud_object(result.stdout, resource=f"TPU node {node_id!r}")

    def create(self, spec: QueuedResourceSpec, *, qr_id: str, recorded_qr_id: str | None = None) -> QueuedResource:
        existing = self.describe_qr(qr_id, project=spec.project, zone=spec.zone)
        if existing is not None:
            self._validate_qr(existing, spec, recorded_qr_id=recorded_qr_id)
            return existing
        try:
            if spec.startup_script:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as script:
                    script.write(spec.startup_script)
                    script.flush()
                    result = self._run(spec.create_args(qr_id=qr_id, startup_script_path=script.name))
            else:
                result = self._run(spec.create_args(qr_id=qr_id))
        except GcloudError:
            # A gcloud call that times out may still have landed; give it the
            # same re-check a failed command gets before surfacing the failure.
            existing = self.describe_qr(qr_id, project=spec.project, zone=spec.zone)
            if existing is not None:
                self._validate_qr(existing, spec, recorded_qr_id=recorded_qr_id)
                return existing
            raise
        if result.returncode != 0:
            existing = self.describe_qr(qr_id, project=spec.project, zone=spec.zone)
            if existing is not None:
                self._validate_qr(existing, spec, recorded_qr_id=recorded_qr_id)
                return existing
            diagnostic = (result.stderr or result.stdout).strip() or f"could not create {qr_id}"
            if self._is_queue_admission_denied(diagnostic):
                raise GcloudQueueAdmissionDeniedError(diagnostic)
            classified = provider_failure_from_diagnostic(diagnostic)
            if classified is None:
                raise GcloudError(diagnostic)
            raise GcloudError(diagnostic, code=classified)
        created = self.describe_qr(qr_id, project=spec.project, zone=spec.zone)
        if created is None:
            raise GcloudError(f"queued resource {qr_id!r} was not visible after creation")
        self._validate_qr(created, spec, recorded_qr_id=recorded_qr_id)
        return created

    @staticmethod
    def _is_queue_admission_denied(diagnostic: str) -> bool:
        """Return whether GCP explicitly rejected entry to its TPU capacity queue.

        ``PERMISSION_DENIED`` is broad: it can describe project, network, or
        resource permissions unrelated to queue admission. GCP's explicit
        queue-admission diagnostic is the only one promoted to the typed
        failure because the control plane can give it dedicated guidance.
        """

        normalized = " ".join(diagnostic.casefold().split())
        # "Queueing is not supported" is the same answer from the other side: the
        # zone has no queue for this accelerator type at all, so a retry can never
        # be admitted either. Classify it explicitly to avoid retrying a zone
        # that cannot accept the request.
        return (
            "does not have permission to submit requests into this queue" in normalized
            or "queueing is not supported for accelerator type" in normalized
        )

    @staticmethod
    def _exposes_ownership_fields(qr: QueuedResource) -> bool:
        """Return whether the provider surfaced any ownership marks to compare.

        Cloud TPU does not echo ``--labels`` or ``--metadata`` back through
        ``queued-resources describe`` or ``tpu-vm describe`` for a provisioned
        queued resource: the node comes back with ``labels: null`` and
        ``metadata: {}`` even though the startup script from that same metadata
        ran.

        So "no ownership fields at all" is not evidence of a foreign resource --
        it is the provider declining to answer, and it is the normal case here.
        """

        if is_json_object(qr.raw.get("labels")):
            return True
        tpu_value = qr.raw.get("tpu")
        tpu = tpu_value if is_json_object(tpu_value) else {}
        node_specs_value = tpu.get("nodeSpec")
        node_specs = node_specs_value if isinstance(node_specs_value, list) else []
        for spec_value in node_specs:
            if is_json_object(spec_value):
                node_value = spec_value.get("node")
                if is_json_object(node_value) and is_json_object(node_value.get("labels")):
                    return True
        return False

    @staticmethod
    def _validate_qr(qr: QueuedResource, spec: QueuedResourceSpec, *, recorded_qr_id: str | None = None) -> None:
        expected_uid = spec.labels.get("eyvan-uid")
        if not expected_uid:
            raise GcloudError(f"queued resource {qr.qr_id!r} is not owned by this Eyvan cell")
        if GcloudTpuProvider._exposes_ownership_fields(qr):
            # The provider answered, so its answer is authoritative in both
            # directions: a mismatch here is a genuinely foreign resource.
            if not GcloudTpuProvider._has_ownership_labels(
                qr,
                management_uid=expected_uid,
                cell_id=spec.labels.get("eyvan-cell"),
            ):
                raise GcloudError(f"queued resource {qr.qr_id!r} is not owned by this Eyvan cell")
        elif recorded_qr_id != qr.qr_id:
            # No marks to compare, so fall back to the only other authority:
            # Eyvan minted this exact id under a fence and recorded it. Without
            # this, an ACTIVE queued resource Eyvan created itself is declared
            # foreign the moment it becomes usable -- capacity stranded and
            # billing, with the workload stuck pending. Treating "the provider
            # said nothing" as "this is someone else's" is the same
            # unknown-is-not-a-value mistake as the rest of this class.
            raise GcloudError(f"queued resource {qr.qr_id!r} is not owned by this Eyvan cell")
        if qr.accelerator_type is not None and qr.accelerator_type != spec.accelerator_type:
            raise GcloudError(
                f"queued resource {qr.qr_id!r} requests {qr.accelerator_type!r}, expected {spec.accelerator_type!r}"
            )
        if not qr.node_ids:
            return
        if len(qr.node_ids) != spec.node_count:
            raise GcloudError(
                f"queued resource {qr.qr_id!r} owns {len(qr.node_ids)} TPU nodes, expected {spec.node_count}"
            )
        if spec.node_count == 1:
            expected_node_id = spec.node_id or spec.name
            if qr.node_ids != (expected_node_id,):
                raise GcloudError(f"queued resource {qr.qr_id!r} does not own TPU node {expected_node_id!r}")
        elif spec.node_prefix is not None and any(not node_id.startswith(spec.node_prefix) for node_id in qr.node_ids):
            raise GcloudError(f"queued resource {qr.qr_id!r} does not own the expected TPU node prefix")

    @staticmethod
    def _has_ownership_labels(
        qr: QueuedResource,
        *,
        management_uid: str,
        cell_id: str | None = None,
    ) -> bool:
        """Return whether every available provider ownership label matches Eyvan's record."""

        def matches(labels: JsonValue | None) -> bool:
            return (
                is_json_object(labels)
                and labels.get("eyvan-uid") == management_uid
                and (cell_id is None or labels.get("eyvan-cell") == cell_id)
            )

        top_level_labels = qr.raw.get("labels")
        tpu_value = qr.raw.get("tpu")
        tpu = tpu_value if is_json_object(tpu_value) else {}
        node_specs_value = tpu.get("nodeSpec")
        node_specs = node_specs_value if isinstance(node_specs_value, list) else []
        node_labels: list[JsonValue | None] = []
        for spec_value in node_specs:
            if not is_json_object(spec_value):
                continue
            node_value = spec_value.get("node")
            if is_json_object(node_value):
                node_labels.append(node_value.get("labels"))
        if is_json_object(top_level_labels):
            return matches(top_level_labels) and all(
                not is_json_object(labels) or matches(labels) for labels in node_labels
            )
        return bool(node_labels) and all(matches(labels) for labels in node_labels)

    def delete(
        self,
        qr_id: str,
        *,
        project: str,
        zone: str,
        management_uid: str | None = None,
    ) -> None:
        existing = self.describe_qr(qr_id, project=project, zone=zone)
        if existing is None:
            return
        if not isinstance(management_uid, str) or not management_uid:
            raise GcloudError(f"refusing to delete queued resource {qr_id!r} without its Eyvan ownership UID")
        if self._exposes_ownership_fields(existing):
            # The provider answered; a mismatch means this is someone else's.
            if not self._has_ownership_labels(existing, management_uid=management_uid):
                raise GcloudError(f"refusing to delete queued resource {qr_id!r} without its Eyvan ownership UID")
        # Otherwise the provider exposes nothing to compare, which for a
        # *provisioned* Cloud TPU queued resource is always the case: labels and
        # metadata are not echoed back once it is ACTIVE. Refusing then does not
        # protect anything -- it guarantees a leak, because every TPU Eyvan
        # creates becomes undeletable by Eyvan the moment it becomes usable.
        #
        # That is exactly how ACTIVE v5p-128 queued resources survived for days
        # with nothing running on them: cells released while still WAITING went
        # away cleanly, and every one that reached ACTIVE could never be
        # reclaimed. The caller is deleting an id it recorded on its own fenced
        # cell record, which is the remaining authority.
        args = (
            "compute",
            "tpus",
            "queued-resources",
            "delete",
            qr_id,
            "--zone",
            zone,
            "--project",
            project,
            "--quiet",
            "--force",
        )
        try:
            result = self._run(args)
        except GcloudError:
            # A gcloud call that times out may still have deleted the resource;
            # give it the same re-check a failed command gets.
            if self.describe_qr(qr_id, project=project, zone=zone) is not None:
                raise
            return
        if result.returncode != 0:
            if self.describe_qr(qr_id, project=project, zone=zone) is not None:
                raise GcloudError((result.stderr or result.stdout).strip() or f"could not delete {qr_id}")
            return
        deadline = time.monotonic() + self.delete_wait_timeout_s
        while True:
            try:
                if self.describe_qr(qr_id, project=project, zone=zone) is None:
                    return
            except GcloudError:
                # A transient describe failure must not abandon the delete
                # wait; keep polling until the deadline.
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GcloudError(
                    f"queued resource {qr_id!r} still exists after delete "
                    f"(waited {self.delete_wait_timeout_s:g} seconds for its removal)",
                    code=CapacityProviderFailureCode.PROVIDER_ERROR,
                )
            time.sleep(min(2, remaining))

    @staticmethod
    def _network_ips(
        node_id: str,
        node: JsonObject,
        *,
        endpoint_mode: NativeWorkerEndpointMode,
    ) -> tuple[str, ...]:
        endpoints_value = node.get("networkEndpoints")
        if endpoints_value is None:
            return ()
        if not isinstance(endpoints_value, list):
            raise GcloudError(f"TPU node {node_id!r} returned invalid network endpoints")
        addresses: list[str] = []
        incomplete = False
        for endpoint_value in endpoints_value:
            if not is_json_object(endpoint_value):
                raise GcloudError(f"TPU node {node_id!r} returned an invalid network endpoint")
            if endpoint_mode is NativeWorkerEndpointMode.PROFILE_MANAGED_PUBLIC:
                access_config = endpoint_value.get("accessConfig")
                if access_config is None:
                    incomplete = True
                    continue
                if not is_json_object(access_config):
                    raise GcloudError(f"TPU node {node_id!r} returned an invalid public network endpoint")
                ip_address = access_config.get("externalIp")
                address_label = "external IP address"
            else:
                ip_address = endpoint_value.get("ipAddress")
                address_label = "IP address"
            if ip_address is None:
                incomplete = True
                continue
            if not isinstance(ip_address, str) or not ip_address or ip_address != ip_address.strip():
                raise GcloudError(f"TPU node {node_id!r} returned a network endpoint without a valid {address_label}")
            try:
                address = normalize_ip_address(ip_address)
            except ValueError as exc:
                raise GcloudError(
                    f"TPU node {node_id!r} returned a network endpoint without a valid {address_label}"
                ) from exc
            addresses.append(address)
        if incomplete:
            return ()
        return tuple(addresses)

    @classmethod
    def _observe_node(
        cls,
        node_id: str,
        node: JsonObject,
        record: FleetCellRecord,
        *,
        endpoint_mode: NativeWorkerEndpointMode,
    ) -> TpuNodeObservation:
        state_value = node.get("state")
        if not isinstance(state_value, str) or not state_value:
            raise GcloudError(f"TPU cell {record.name!r} contains a node without a state")
        try:
            state = GcpTpuNodeState(state_value.upper())
        except ValueError as exc:
            raise GcloudError(
                f"TPU cell {record.name!r} contains a node with unsupported state {state_value!r}"
            ) from exc
        return TpuNodeObservation(
            node_id=node_id,
            state=state,
            worker_ips=cls._network_ips(node_id, node, endpoint_mode=endpoint_mode),
        )

    @staticmethod
    def _spec(record: FleetCellRecord) -> QueuedResourceSpec:
        if not (record.project and record.zone and record.accelerator_type):
            raise ValueError(f"cell {record.name!r} is missing project, zone, or accelerator_type")
        management_uid = record.metadata.get("management_uid")
        if not isinstance(management_uid, str) or not management_uid:
            raise GcloudError(f"cell {record.name!r} has no Eyvan management UID")
        config = record.metadata.get("tpu_config")
        if config is None:
            config = normalize_tpu_config({"runtime_version": record.runtime_version})
        if not isinstance(config, dict):
            raise GcloudError(f"cell {record.name!r} has invalid TPU provider configuration")
        normalized = normalize_tpu_config(config)
        metadata = normalized["metadata"]
        if normalized.get("node_count", 1) > 1 and normalized.get("node_prefix") is None:
            normalized["node_prefix"] = f"{record.qr_id or record.name}-node"
        return QueuedResourceSpec(
            name=record.name,
            accelerator_type=record.accelerator_type,
            zone=record.zone,
            project=record.project,
            capacity_class=record.capacity_class,
            labels={
                "eyvan-cell": record.name.replace("_", "-").lower(),
                "eyvan-uid": management_uid,
            },
            metadata={**metadata, "eyvan-uid": management_uid},
            runtime_version=normalized["runtime_version"],
            node_count=normalized.get("node_count", 1),
            node_prefix=normalized.get("node_prefix"),
            network=normalized["network"],
            subnetwork=normalized["subnetwork"],
            internal_ips=normalized["internal_ips"],
            service_account=normalized["service_account"],
            scopes=tuple(normalized["scopes"]),
            tags=tuple(normalized["tags"]),
            startup_script=normalized.get("startup_script"),
            bootstrap_egress=normalized.get("bootstrap_egress"),
        )

    def ensure(self, record: FleetCellRecord) -> ProviderObservation:
        if record.desired_state != "up":
            return ProviderObservation(CapacityStatus.DOWN, "desired state is down")
        try:
            endpoint_mode = endpoint_mode_from_metadata(record.metadata)
        except ValueError as exc:
            raise GcloudError(f"cell {record.name!r} has an invalid native worker endpoint mode") from exc
        spec = self._spec(record)
        management_uid = spec.labels["eyvan-uid"]
        qr_id = record.qr_id or record.next_qr_id()
        qr = self.describe_qr(qr_id, project=record.project, zone=record.zone)
        if qr is not None:
            self._validate_qr(qr, spec, recorded_qr_id=record.qr_id)
        if qr is None:
            # A previously observed allocation disappeared. Its credential may
            # already have been exposed: let the driver's fenced replacement
            # advance the provider generation and bootstrap before provisioning.
            if record.state not in {CapacityStatus.UNOBSERVED, CapacityStatus.RECREATING}:
                return ProviderObservation(CapacityStatus.NEEDS_RECREATE, "queued resource is absent")
            # A same-named singleton TPU VM without a queued resource is not
            # ours to adopt. Multi-slice node names are generated from the QR
            # prefix, so the QR ownership label is its only safe authority.
            if spec.node_count == 1:
                node_id = spec.node_id or spec.name
                if self.describe_node(node_id, project=record.project, zone=record.zone) is not None:
                    raise GcloudError(f"TPU node {node_id!r} exists without its owned queued resource")
            qr = self.create(spec, qr_id=qr_id, recorded_qr_id=record.qr_id)
            return self._queued_resource_observation(qr, created=True)
        # Allocation lifecycle is authoritative even while its children still
        # appear READY. Never publish endpoints from a terminating QR, and keep
        # its provider failure diagnostic/classification for the replacement.
        if qr.terminal:
            return self._queued_resource_observation(qr)
        node_ids = qr.node_ids
        if not node_ids and spec.node_count == 1:
            node_ids = (spec.node_id or spec.name,)
        nodes: list[TpuNodeObservation] = []
        for node_id in node_ids:
            node = self.describe_node(node_id, project=record.project, zone=record.zone)
            if node is None:
                continue
            self._validate_node(node, record, management_uid, owned_qr_node=node_id in qr.node_ids)
            nodes.append(self._observe_node(node_id, node, record, endpoint_mode=endpoint_mode))
        if qr.state is GcpQueuedResourceState.ACTIVE and len(nodes) < spec.node_count:
            return ProviderObservation(
                CapacityStatus.NEEDS_RECREATE,
                f"active queued resource is missing TPU worker allocations ({len(nodes)}/{spec.node_count} visible)",
            )
        if nodes:
            node_states = tuple(node.state for node in nodes)
            worker_ips = tuple(ip for node in nodes for ip in node.worker_ips)
            if len(worker_ips) != len(set(worker_ips)):
                raise GcloudError(f"TPU cell {record.name!r} has duplicate worker network endpoints")
            if any(state in TERMINAL_NODE_STATES for state in node_states):
                terminal = ", ".join(sorted({state.value for state in node_states if state in TERMINAL_NODE_STATES}))
                # PREEMPTED is the one terminal state that says something about
                # the *purchase class* rather than the allocation, and it is the
                # ordinary end of a spot TPU. Reported typed so the control
                # plane does not record a spot reclamation as evidence that the
                # pool is out of capacity -- which is what it did while every
                # non-ready observation collapsed to one belief.
                preempted = any(state is GcpTpuNodeState.PREEMPTED for state in node_states)
                if not preempted and spec.capacity_class == "spot":
                    # Reaching here means ``desired_state`` is still "up": Eyvan
                    # wants this cell and did not ask for it to go away. On spot
                    # that is a reclamation, and reading it as a generic capacity
                    # loss charges it to ``max_retries`` -- which a spot job sets
                    # to 0 precisely because it expects to be reclaimed -- and so
                    # kills the workload instead of resuming it on the preemption
                    # budget it still has.
                    #
                    # Narrow on purpose: STOPPED, TERMINATED and the HIDDEN pair
                    # are left generic, because those say something happened to
                    # the host rather than to the purchase, and calling them
                    # reclamations would suppress the failure evidence that lets
                    # a genuinely bad zone accumulate a cooldown.
                    #
                    # This is the rule ``compute_instances`` already applies to a
                    # Spot VM that reached TERMINATED without Eyvan stopping it,
                    # limited the same way. TPU was the sibling still requiring
                    # the provider to spell the reclamation out.
                    preempted = any(state in SPOT_RECLAMATION_NODE_STATES for state in node_states)
                return ProviderObservation(
                    CapacityStatus.NEEDS_RECREATE,
                    f"TPU VM is terminal ({terminal})",
                    failure_code=CapacityProviderFailureCode.PREEMPTED if preempted else None,
                )
            ambiguous_states = {
                GcpTpuNodeState.STATE_UNSPECIFIED,
                GcpTpuNodeState.UNKNOWN,
            }
            if any(state in ambiguous_states for state in node_states):
                detail = ", ".join(sorted({state.value for state in node_states if state in ambiguous_states}))
                return ProviderObservation(CapacityStatus.UNAVAILABLE, f"TPU VM state is not definitive ({detail})")
            all_ready = len(nodes) == spec.node_count and all(state is GcpTpuNodeState.READY for state in node_states)
            ready_nodes_with_endpoints = sum(
                node.state is GcpTpuNodeState.READY and bool(node.worker_ips) for node in nodes
            )
            # Cloud TPU returns the complete endpoint set for every READY TPU
            # node. The provider response, not a static shape table, is the
            # authority for how many TPU VMs belong to an allocation.
            if not all_ready or ready_nodes_with_endpoints != spec.node_count:
                return ProviderObservation(
                    CapacityStatus.PROVISIONING,
                    f"TPU workers are provisioning ({len(nodes)}/{spec.node_count} worker allocations and "
                    f"{len(worker_ips)} TPU VM endpoints visible across "
                    f"{ready_nodes_with_endpoints}/{spec.node_count} ready allocations)",
                    head_ip=worker_ips[0] if worker_ips else None,
                )
            port = record.metadata.get("native_worker_port", 8100)
            if isinstance(port, bool):
                raise GcloudError(f"cell {record.name!r} has an invalid native worker port")
            try:
                worker_port = int(port)
            except (TypeError, ValueError) as exc:
                raise GcloudError(f"cell {record.name!r} has an invalid native worker port") from exc
            if worker_port <= 0 or worker_port > 65535:
                raise GcloudError(f"cell {record.name!r} has an invalid native worker port")
            endpoints = tuple(format_http_ip_endpoint(ip, worker_port) for ip in worker_ips)
            # The provider stays the authority for how many *slices* an
            # allocation has; the catalog is the authority for how many *hosts*
            # a slice has, which the provider response alone cannot contradict
            # because a short response looks exactly like a small slice.
            #
            # Without this, a v5p-16 that came back with one host of two reports
            # HEALTHY and hands the workload a silently half-sized JAX cluster:
            # the run starts, forms a smaller mesh than it asked for, and the
            # only symptom is throughput nobody can explain.
            #
            # Only checked for types whose host count this build knows. For the
            # rest the behaviour is unchanged, which is why an unknown shape is
            # not an outage.
            shape = shape_for(spec.accelerator_type)
            if shape is not None and shape.hosts is not None:
                expected_vms = spec.node_count * shape.hosts
                if len(endpoints) != expected_vms:
                    return ProviderObservation(
                        CapacityStatus.PROVISIONING,
                        f"{len(endpoints)} TPU VM endpoints are visible but {spec.accelerator_type} has "
                        f"{shape.hosts} hosts per slice, so {expected_vms} are expected across "
                        f"{spec.node_count} allocations",
                        head_ip=worker_ips[0] if worker_ips else None,
                    )
            native = record.metadata.get("execution_backend") == "eyvan-native"
            return ProviderObservation(
                CapacityStatus.HEALTHY,
                f"{len(nodes)} TPU worker allocations and {len(endpoints)} TPU VMs are ready",
                head_ip=worker_ips[0],
                execution_endpoint=endpoints[0] if native else None,
                worker_endpoints=endpoints if native else (),
                observed_tpu_vms=len(endpoints),
                runtime_process_count=len(endpoints),
            )
        return self._queued_resource_observation(qr)

    @staticmethod
    def _queued_resource_observation(qr: QueuedResource, *, created: bool = False) -> ProviderObservation:
        """Map one provider-confirmed queued-resource state without inventing capacity progress."""

        prefix = "queued resource was created; " if created else ""
        if qr.state in PENDING_STATES:
            return ProviderObservation(CapacityStatus.WAITING, f"{prefix}GCP reports {qr.state}")
        if qr.state is GcpQueuedResourceState.PROVISIONING:
            return ProviderObservation(CapacityStatus.PROVISIONING, f"{prefix}GCP is provisioning the queued resource")
        if qr.state is GcpQueuedResourceState.ACTIVE:
            return ProviderObservation(CapacityStatus.ACTIVE, f"{prefix}GCP reports it active; TPU workers are starting")
        if qr.terminal:
            failure = _queued_resource_failure(qr.raw)
            if failure is None:
                return ProviderObservation(
                    CapacityStatus.NEEDS_RECREATE, f"{prefix}queued resource is terminal ({qr.state})"
                )
            return ProviderObservation(
                CapacityStatus.NEEDS_RECREATE,
                f"{prefix}queued resource is terminal ({qr.state}): {failure}",
                failure_code=provider_failure_from_diagnostic(failure),
            )
        return ProviderObservation(CapacityStatus.UNAVAILABLE, f"{prefix}GCP reports queued resource {qr.state}")

    @staticmethod
    def _validate_node(
        node: JsonObject, record: FleetCellRecord, management_uid: str, *, owned_qr_node: bool = False
    ) -> None:
        owned = False
        for marks in (node.get("labels"), node.get("metadata")):
            if marks is None:
                continue
            if not is_json_object(marks):
                raise GcloudError(f"TPU node {record.name!r} is not owned by this Eyvan cell")
            if "eyvan-uid" in marks:
                if marks["eyvan-uid"] != management_uid:
                    raise GcloudError(f"TPU node {record.name!r} is not owned by this Eyvan cell")
                owned = True
            if "eyvan-cell" in marks and marks["eyvan-cell"] != record.name.replace("_", "-").lower():
                raise GcloudError(f"TPU node {record.name!r} is not owned by this Eyvan cell")
        # An already validated QR can authorize only its explicit children.
        # A same-named fallback node is not membership evidence. When provider
        # marks are omitted, require the child's accelerator shape as well.
        if not owned and not (owned_qr_node and node.get("acceleratorType") == record.accelerator_type):
            raise GcloudError(f"TPU node {record.name!r} is not owned by this Eyvan cell")
        accelerator_type = node.get("acceleratorType")
        if accelerator_type is not None and not isinstance(accelerator_type, str):
            raise GcloudError(f"TPU node {record.name!r} returned an invalid accelerator type")
        if accelerator_type is not None and accelerator_type != record.accelerator_type:
            raise GcloudError(f"TPU node {record.name!r} is {accelerator_type!r}, expected {record.accelerator_type!r}")
