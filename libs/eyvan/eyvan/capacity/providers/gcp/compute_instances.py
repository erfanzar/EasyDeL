# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Direct gcloud integration for CPU, GPU, and TPU Compute Engine capacity."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import tempfile
import threading
import time
import typing as tp
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from eyvan._internal.compat import StrEnum
from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan._internal.network import format_http_ip_endpoint, normalize_ip_address
from eyvan._internal.subprocess_env import provider_cli_env
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CapacityProviderFailureCode, provider_failure_from_diagnostic
from eyvan.capacity.fleet_state import FleetCellRecord
from eyvan.capacity.providers.gcp.native_worker_ingress import NativeWorkerEndpointMode, endpoint_mode_from_metadata
from eyvan.capacity.providers.gcp.queued_resources import (
    BOOTSTRAP_EGRESS_MODES,
    GcloudError,
    ProviderObservation,
    ProviderOptionValue,
    Runner,
    bootstrap_egress_choices,
    decode_gcloud_object,
)

_DEFAULT_CPU_IMAGE = ("ubuntu-2404-lts-amd64", "ubuntu-os-cloud")
_DEFAULT_GPU_IMAGE = ("ubuntu-accelerator-2404-amd64-with-nvidia-580", "ubuntu-os-accelerator-images")
_DEFAULT_TPU_IMAGE = (
    "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e",
    "ubuntu-os-accelerator-images",
)
_INTEGRATED_ACCELERATOR_MACHINE_PREFIXES = ("a2-", "a3-", "a4-", "a4x-", "g2-", "g4-", "ct5lp-", "ct5p-", "ct6e-")
_COMPUTE_OPTION_KEYS = frozenset(
    {
        "bootstrap_egress",
        "boot_disk_size_gb",
        "boot_disk_type",
        "gpu_count",
        "gpu_type",
        "image",
        "image_family",
        "image_project",
        "internal_ip_only",
        "machine_type",
        "maintenance_policy",
        "metadata",
        "network",
        "reservation",
        "scopes",
        "service_account",
        "startup_script",
        "subnetwork",
        "tags",
        "tpu_count",
        "tpu_type",
    }
)
_INSTANCE_NAME_PATTERN = re.compile(r"[^a-z0-9-]+")
_NATIVE_SECRET_NAME_PATTERN = re.compile(r"[^a-z0-9-]+")
_PROJECT_NUMBER_PATTERN = re.compile(r"^[1-9][0-9]*$")
_GCS_BUCKET_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{1,61}[a-z0-9])?$")


class GcpComputeInstanceStatus(StrEnum):
    """Compute Engine instance states accepted at the gcloud boundary."""

    PROVISIONING = "PROVISIONING"
    STAGING = "STAGING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    SUSPENDING = "SUSPENDING"
    SUSPENDED = "SUSPENDED"
    REPAIRING = "REPAIRING"
    TERMINATED = "TERMINATED"


class ComputeProviderConfig(tp.TypedDict):
    """Validated provider options used to create one Compute instance."""

    bootstrap_egress: str | None
    machine_type: str
    gpu_type: str | None
    gpu_count: int
    tpu_type: str | None
    tpu_count: int
    image: str | None
    image_family: str | None
    image_project: str | None
    boot_disk_size_gb: int
    boot_disk_type: str
    network: str | None
    subnetwork: str | None
    service_account: str | None
    scopes: list[str]
    tags: list[str]
    metadata: dict[str, str]
    startup_script: str | None
    internal_ip_only: bool
    maintenance_policy: str | None
    reservation: str | None


@dataclass(frozen=True, slots=True)
class GcsArtifactBucket:
    """One concrete GCS bucket derived from an Eyvan artifact-root URI."""

    name: str

    @property
    def uri(self) -> str:
        """Return the canonical bucket URI accepted by ``gcloud storage``."""

        return f"gs://{self.name}"

    @classmethod
    def from_artifact_root_uri(cls, artifact_root_uri: str) -> GcsArtifactBucket:
        """Derive a bucket-scoped IAM target from a concrete GCS artifact root."""

        if (
            not isinstance(artifact_root_uri, str)
            or artifact_root_uri != artifact_root_uri.strip()
            or "\x00" in artifact_root_uri
        ):
            raise ValueError("native worker artifact root must be a concrete gs://bucket[/prefix] URI")
        parsed = urlsplit(artifact_root_uri)
        bucket = parsed.netloc
        if (
            parsed.scheme != "gs"
            or not bucket
            or parsed.query
            or parsed.fragment
            or _GCS_BUCKET_PATTERN.fullmatch(bucket) is None
        ):
            raise ValueError("native worker artifact root must be a concrete gs://bucket[/prefix] URI")
        raw_path = parsed.path.rstrip("/")
        if raw_path:
            if not raw_path.startswith("/"):
                raise ValueError("native worker artifact root must be a concrete gs://bucket[/prefix] URI")
            segments = raw_path[1:].split("/")
            if any(
                not segment
                or segment in {".", ".."}
                or any(character in segment for character in ("\\", "*", "?", "[", "]"))
                for segment in segments
            ):
                raise ValueError("native worker artifact root must be a concrete gs://bucket[/prefix] URI")
        return cls(name=bucket)


def compute_instance_name(cell_id: str, provider_generation: int) -> str:
    """Return a deterministic, valid, collision-resistant Compute instance name."""

    normalized = _INSTANCE_NAME_PATTERN.sub("-", cell_id.strip().lower()).strip("-") or "cell"
    if not normalized[0].isalpha():
        normalized = f"cell-{normalized}"
    digest = hashlib.sha256(cell_id.encode()).hexdigest()[:8]
    suffix = f"-{digest}-r{provider_generation}"
    return f"{normalized[: 63 - len(suffix)].rstrip('-')}{suffix}"


def _cell_label(cell_id: str) -> str:
    normalized = _INSTANCE_NAME_PATTERN.sub("-", cell_id.strip().lower()).strip("-") or "cell"
    if len(normalized) <= 63:
        return normalized
    suffix = f"-{hashlib.sha256(cell_id.encode()).hexdigest()[:8]}"
    return f"{normalized[: 63 - len(suffix)].rstrip('-')}{suffix}"


def native_worker_secret_id(management_uid: str) -> str:
    """Return a stable Secret Manager identifier for one Eyvan cell."""

    if not isinstance(management_uid, str) or not management_uid.strip():
        raise ValueError("native worker management UID must be a non-empty string")
    normalized = _NATIVE_SECRET_NAME_PATTERN.sub("-", management_uid.lower()).strip("-")
    if not normalized:
        raise ValueError("native worker management UID cannot produce a secret identifier")
    digest = hashlib.sha256(management_uid.encode()).hexdigest()[:8]
    prefix = "eyvan-native-"
    suffix = f"-{digest}"
    return f"{prefix}{normalized[: 63 - len(prefix) - len(suffix)].rstrip('-')}{suffix}"


class GcloudDefaultComputeServiceAccountResolver:
    """Resolve the service account GCP attaches when a VM omits one explicitly."""

    def __init__(self, *, runner: Runner | None = None):
        self._runner = runner or _runner

    def _run(self, args: tp.Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Run one gcloud call, surfacing its subprocess timeout as a typed provider failure."""
        try:
            return self._runner(args)
        except subprocess.TimeoutExpired as exc:
            raise GcloudError(str(exc), code=CapacityProviderFailureCode.PROVIDER_ERROR) from exc

    def resolve(self, *, project: str) -> str:
        """Return the project's default Compute Engine service-account email.

        GCP attaches this identity to Compute Engine VMs and Cloud TPU queued
        resources when their create request omits ``--service-account``. Eyvan
        needs the concrete email only to grant a cell-scoped Secret Manager
        reader binding; it deliberately leaves capacity creation account-free.
        """

        if not isinstance(project, str) or not project.strip() or project != project.strip():
            raise ValueError("GCP project must be a non-empty trimmed string")
        result = self._run(("projects", "describe", project, "--format=value(projectNumber)"))
        if result.returncode != 0:
            raise GcloudError(
                (result.stderr or result.stdout).strip()
                or f"could not resolve the default Compute Engine service account for project {project!r}"
            )
        project_number = result.stdout.strip()
        if _PROJECT_NUMBER_PATTERN.fullmatch(project_number) is None:
            raise GcloudError(
                f"could not resolve the default Compute Engine service account for project {project!r}: "
                "gcloud did not return a numeric project number"
            )
        return f"{project_number}-compute@developer.gserviceaccount.com"


def _string(value: ProviderOptionValue, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"Compute option {name!r} must be a non-empty trimmed string")
    return value


def _string_sequence(value: ProviderOptionValue, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Compute option {name!r} must be a list of strings")
    result: list[str] = []
    for item in value:
        parsed = _string(item, name)
        if parsed is None:
            raise AssertionError(f"required Compute option {name!r} passed validation without a value")
        result.append(parsed)
    return result


def _string_mapping(value: ProviderOptionValue, name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not is_json_object(value):
        raise ValueError(f"Compute option {name!r} must be a string mapping")
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not raw_key.strip() or raw_key != raw_key.strip():
            raise ValueError(f"Compute option {name!r} keys must be non-empty trimmed strings")
        item = _string(raw_value, f"{name}[{raw_key}]")
        if item is None:
            raise AssertionError(f"required Compute option {name!r} passed validation without a value")
        result[raw_key] = item
    return result


def normalize_compute_config(desired: tp.Mapping[str, ProviderOptionValue]) -> ComputeProviderConfig:
    """Validate and normalize provider options for one Compute Engine instance."""

    nested_values = [
        desired.get(name) for name in ("provider_config", "provider_options") if desired.get(name) is not None
    ]
    if len(nested_values) > 1:
        raise ValueError("set only one of provider_config or provider_options")
    nested = nested_values[0] if nested_values else {}
    if not is_json_object(nested):
        raise ValueError("Compute provider options must be a mapping")
    unknown = sorted(str(key) for key in nested if key not in _COMPUTE_OPTION_KEYS)
    if unknown:
        # Naming the accepted keys costs one line here and removes a round trip
        # through the source for whoever typed the wrong one.
        raise ValueError(
            f"unknown Compute provider option: {unknown[0]}; accepted keys are {', '.join(sorted(_COMPUTE_OPTION_KEYS))}"
        )
    options: dict[str, ProviderOptionValue] = dict(nested)
    for key in _COMPUTE_OPTION_KEYS:
        if key in desired:
            options[key] = desired[key]

    machine_type = _string(options.get("machine_type"), "machine_type")
    gpu_type = _string(options.get("gpu_type"), "gpu_type", optional=True)
    raw_gpu_count = options.get("gpu_count", 0)
    if isinstance(raw_gpu_count, bool) or not isinstance(raw_gpu_count, int) or raw_gpu_count < 0:
        raise ValueError("Compute option 'gpu_count' must be a non-negative integer")
    tpu_type = _string(options.get("tpu_type"), "tpu_type", optional=True)
    raw_tpu_count = options.get("tpu_count", 0)
    if isinstance(raw_tpu_count, bool) or not isinstance(raw_tpu_count, int) or raw_tpu_count < 0:
        raise ValueError("Compute option 'tpu_count' must be a non-negative integer")
    gpu_configured = gpu_type is not None or raw_gpu_count > 0
    tpu_configured = tpu_type is not None or raw_tpu_count > 0
    if gpu_configured and tpu_configured:
        raise ValueError("GPU and TPU Compute options are mutually exclusive")
    if (gpu_type is None) != (raw_gpu_count == 0):
        raise ValueError("gpu_type and a positive gpu_count must be provided together")
    if (tpu_type is None) != (raw_tpu_count == 0):
        raise ValueError("tpu_type and a positive tpu_count must be provided together")

    image = _string(options.get("image"), "image", optional=True)
    image_family = _string(options.get("image_family"), "image_family", optional=True)
    image_project = _string(options.get("image_project"), "image_project", optional=True)
    if image and image_family:
        raise ValueError("Compute options image and image_family are mutually exclusive")
    if image and image_project:
        raise ValueError("image_project applies only to image_family")
    if image is None and image_family is None:
        if tpu_type:
            image_family, image_project = _DEFAULT_TPU_IMAGE
        elif gpu_type:
            image_family, image_project = _DEFAULT_GPU_IMAGE
        else:
            image_family, image_project = _DEFAULT_CPU_IMAGE
    elif image_family and image_project is None:
        raise ValueError("image_family requires image_project")

    raw_disk_size = options.get("boot_disk_size_gb", 100)
    minimum_disk_size = 40 if gpu_type or tpu_type else 10
    if isinstance(raw_disk_size, bool) or not isinstance(raw_disk_size, int) or raw_disk_size < minimum_disk_size:
        raise ValueError(f"Compute option 'boot_disk_size_gb' must be an integer of at least {minimum_disk_size}")
    internal_ip_only = options.get("internal_ip_only", False)
    if not isinstance(internal_ip_only, bool):
        raise ValueError("Compute option 'internal_ip_only' must be a boolean")
    maintenance_policy = _string(options.get("maintenance_policy"), "maintenance_policy", optional=True)
    if maintenance_policy is not None:
        maintenance_policy = maintenance_policy.upper()
        if maintenance_policy not in {"MIGRATE", "TERMINATE"}:
            raise ValueError("maintenance_policy must be MIGRATE or TERMINATE")
        if gpu_type and maintenance_policy != "TERMINATE":
            raise ValueError("GPU instances require maintenance_policy TERMINATE")
        if tpu_type and maintenance_policy != "TERMINATE":
            raise ValueError("TPU instances require maintenance_policy TERMINATE")

    metadata = _string_mapping(options.get("metadata"), "metadata")
    reserved_metadata = sorted(key for key in metadata if key.startswith("eyvan-"))
    if reserved_metadata:
        raise ValueError(f"Compute metadata key {reserved_metadata[0]!r} is reserved by Eyvan")
    startup_script = options.get("startup_script")
    if startup_script is not None and (not isinstance(startup_script, str) or not startup_script.strip()):
        raise ValueError("Compute option 'startup_script' must be a non-empty string")
    bootstrap_egress = _string(options.get("bootstrap_egress"), "bootstrap_egress", optional=True)
    if bootstrap_egress is not None:
        bootstrap_egress = bootstrap_egress.lower().replace("_", "-")
        if bootstrap_egress not in BOOTSTRAP_EGRESS_MODES:
            raise ValueError(f"Compute option 'bootstrap_egress' must be one of: {bootstrap_egress_choices()}")

    if machine_type is None:
        raise AssertionError("required Compute machine type passed validation without a value")
    boot_disk_type = _string(options.get("boot_disk_type", "pd-balanced"), "boot_disk_type")
    if boot_disk_type is None:
        raise AssertionError("required Compute boot disk type passed validation without a value")
    return {
        "bootstrap_egress": bootstrap_egress,
        "machine_type": machine_type,
        "gpu_type": gpu_type,
        "gpu_count": raw_gpu_count,
        "tpu_type": tpu_type,
        "tpu_count": raw_tpu_count,
        "image": image,
        "image_family": image_family,
        "image_project": image_project,
        "boot_disk_size_gb": raw_disk_size,
        "boot_disk_type": boot_disk_type,
        "network": _string(options.get("network"), "network", optional=True),
        "subnetwork": _string(options.get("subnetwork"), "subnetwork", optional=True),
        "service_account": _string(options.get("service_account"), "service_account", optional=True),
        "scopes": _string_sequence(options.get("scopes"), "scopes"),
        "tags": _string_sequence(options.get("tags"), "tags"),
        "metadata": metadata,
        "startup_script": startup_script,
        "internal_ip_only": internal_ip_only,
        "maintenance_policy": maintenance_policy,
        "reservation": _string(options.get("reservation"), "reservation", optional=True),
    }


def _mapping_flag(values: tp.Mapping[str, str]) -> str:
    items = [f"{key}={value}" for key, value in sorted(values.items())]
    if all("," not in item for item in items):
        return ",".join(items)
    for delimiter in ("|", ";", "~", ":"):
        if all(delimiter not in item for item in items):
            return f"^{delimiter}^{delimiter.join(items)}"
    raise ValueError("Compute metadata contains unsupported delimiter characters")


@dataclass(frozen=True, slots=True)
class ComputeInstanceSpec:
    """Inputs for one CPU, GPU, or TPU Compute Engine instance."""

    name: str
    project: str
    zone: str
    machine_type: str
    capacity_class: str = "on-demand"
    gpu_type: str | None = None
    gpu_count: int = 0
    tpu_type: str | None = None
    tpu_count: int = 0
    image: str | None = None
    image_family: str | None = None
    image_project: str | None = None
    boot_disk_size_gb: int = 100
    boot_disk_type: str = "pd-balanced"
    network: str | None = None
    subnetwork: str | None = None
    service_account: str | None = None
    scopes: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    startup_script: str | None = None
    internal_ip_only: bool = False
    maintenance_policy: str | None = None
    reservation: str | None = None

    def __post_init__(self) -> None:
        gpu_configured = self.gpu_type is not None or self.gpu_count > 0
        tpu_configured = self.tpu_type is not None or self.tpu_count > 0
        if gpu_configured and tpu_configured:
            raise ValueError("GPU and TPU Compute options are mutually exclusive")
        if (self.gpu_type is None) != (self.gpu_count == 0):
            raise ValueError("gpu_type and a positive gpu_count must be provided together")
        if (self.tpu_type is None) != (self.tpu_count == 0):
            raise ValueError("tpu_type and a positive tpu_count must be provided together")
        if tpu_configured and not self.machine_type.startswith(_INTEGRATED_ACCELERATOR_MACHINE_PREFIXES):
            raise ValueError(
                "TPU Compute capacity requires an integrated TPU machine type (ct5lp-*, ct5p-*, or ct6e-*); "
                "--accelerator is the guest-attached GPU path"
            )

    def create_args(self, *, startup_script_path: str | None = None) -> list[str]:
        args = [
            "compute",
            "instances",
            "create",
            self.name,
            "--zone",
            self.zone,
            "--project",
            self.project,
            "--machine-type",
            self.machine_type,
            "--boot-disk-size",
            f"{self.boot_disk_size_gb}GB",
            "--boot-disk-type",
            self.boot_disk_type,
        ]
        if self.image:
            args.extend(("--image", self.image))
        elif self.image_family:
            args.extend(("--image-family", self.image_family, "--image-project", self.image_project or ""))
        accelerator_type = self.gpu_type or self.tpu_type
        if self.gpu_type and not self.machine_type.startswith(_INTEGRATED_ACCELERATOR_MACHINE_PREFIXES):
            args.extend(("--accelerator", f"type={self.gpu_type},count={self.gpu_count}"))
        maintenance_policy = self.maintenance_policy
        if accelerator_type or self.capacity_class == "spot":
            maintenance_policy = "TERMINATE"
        if maintenance_policy:
            args.extend(("--maintenance-policy", maintenance_policy))
        if self.capacity_class == "spot":
            args.extend(("--provisioning-model", "SPOT", "--instance-termination-action", "DELETE"))
        elif self.capacity_class == "reserved":
            if not self.reservation:
                raise ValueError("reserved Compute capacity requires a reservation")
            args.extend(("--reservation-affinity", "specific", "--reservation", self.reservation))
        elif self.capacity_class != "on-demand":
            raise ValueError(f"unsupported Compute capacity class {self.capacity_class!r}")
        if self.labels:
            args.extend(("--labels", _mapping_flag(self.labels)))
        if self.metadata:
            args.extend(("--metadata", _mapping_flag(self.metadata)))
        if startup_script_path:
            args.extend(("--metadata-from-file", f"startup-script={startup_script_path}"))
        if self.network:
            args.extend(("--network", self.network))
        if self.subnetwork:
            args.extend(("--subnet", self.subnetwork))
        if self.service_account:
            args.extend(("--service-account", self.service_account))
        if self.scopes:
            args.extend(("--scopes", ",".join(self.scopes)))
        if self.tags:
            args.extend(("--tags", ",".join(self.tags)))
        if self.internal_ip_only:
            args.append("--no-address")
        return args


@dataclass(frozen=True, slots=True)
class ComputeInstance:
    """Normalized Compute Engine instance state."""

    name: str
    status: GcpComputeInstanceStatus
    machine_type: str | None
    accelerators: tuple[tuple[str, int], ...]
    raw: JsonObject = field(repr=False)


def _runner(args: tp.Sequence[str]) -> subprocess.CompletedProcess[str]:
    timeout_s = max(float(os.getenv("EYVAN_GCLOUD_COMPUTE_TIMEOUT_S", "600")), 1.0)
    return subprocess.run(
        ["gcloud", *args], check=False, capture_output=True, text=True, timeout=timeout_s, env=provider_cli_env()
    )


class GcloudNativeWorkerSecretStore:
    """Create, rotate, and delete cell-scoped native-worker Secret Manager values."""

    #: Permissions every eyvan-native bootstrap needs on the target project.
    #: ``create`` covers the cell-scoped secret; ``setIamPolicy`` covers the
    #: reader binding. Both are exercised by ``ensure`` on every new cell.
    _BIND_PERMISSIONS = ("secretmanager.secrets.create", "secretmanager.secrets.setIamPolicy")

    def __init__(self, *, runner: Runner = _runner):
        self._runner = runner
        self._preflight_ok: set[str] = set()

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

    def _describe(self, secret_id: str, *, project: str) -> JsonObject | None:
        result = self._run(("secrets", "describe", secret_id, "--project", project, "--format=json"))
        if self._not_found(result):
            return None
        if result.returncode != 0:
            raise GcloudError((result.stderr or result.stdout).strip() or f"could not describe secret {secret_id!r}")
        return decode_gcloud_object(result.stdout, resource=f"secret {secret_id!r}")

    @staticmethod
    def _validate_ownership(secret_id: str, value: JsonObject, management_uid: str) -> None:
        labels = value.get("labels")
        if (
            not is_json_object(labels)
            or labels.get("eyvan-managed") != "true"
            or labels.get("eyvan-uid") != management_uid
        ):
            raise GcloudError(f"secret {secret_id!r} is not owned by this Eyvan cell")

    def caller_bind_state(self, *, project: str) -> tuple[bool, str | None]:
        """Return ``(allowed, denial)`` for binding cell secrets in ``project``.

        ``(True, None)`` means every required permission is held. ``(False,
        message)`` names the missing permissions and the grant that fixes
        them. ``(False, None)`` means the state is unknown — no token, or the
        API was unreachable — and by this project's rule unknown is reported
        as unknown, never as allowed.

        The check is a read-only project ``testIamPermissions`` request to
        Cloud Resource Manager. Secret Manager exposes that method on secrets,
        not on projects; using its project path silently produced an unknown
        result instead of a useful preflight.
        """

        if project in self._preflight_ok:
            return True, None
        token_result = self._run(("auth", "print-access-token"))
        token = (token_result.stdout or "").strip()
        if token_result.returncode != 0 or not token:
            return False, None
        request = urllib.request.Request(
            f"https://cloudresourcemanager.googleapis.com/v1/projects/{project}:testIamPermissions",
            data=json.dumps({"permissions": list(self._BIND_PERMISSIONS)}).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return False, None
        allowed = decoded.get("permissions") if is_json_object(decoded) else None
        missing = [permission for permission in self._BIND_PERMISSIONS if not (allowed and permission in allowed)]
        if missing:
            return False, (
                "the active gcloud account cannot bind native-worker secrets in project "
                f"{project!r} (missing: {', '.join(missing)}). Eyvan-native bootstrap needs a one-time "
                "grant; ask a project owner to run: gcloud projects add-iam-policy-binding "
                f"{project} --member=user:<your-account> --role=roles/secretmanager.admin — "
                "or authenticate as a principal that already holds it (for example a project "
                "owner). Non-native executors are unaffected."
            )
        self._preflight_ok.add(project)
        return True, None

    def ensure_caller_can_bind(self, *, project: str) -> None:
        """Fail fast when the active gcloud account cannot bind cell secrets.

        ``ensure`` calls this before creating anything. When the state is
        unknown (no token, API unreachable), it stays silent and lets
        ``ensure`` fail closed with the provider's own diagnostic — the one
        rule that must survive every refactor of this check.
        """

        _allowed, denial = self.caller_bind_state(project=project)
        if denial is not None:
            # A missing IAM grant is configuration, not a provider fault: no
            # retry of the same request can succeed until someone grants it.
            raise GcloudError(denial, code=CapacityProviderFailureCode.CONFIGURATION)

    def ensure(
        self,
        *,
        project: str,
        management_uid: str,
        value: str,
        reader_service_account: str,
    ) -> str:
        """Ensure a cell-owned secret contains ``value`` and the VM can read it."""

        self.ensure_caller_can_bind(project=project)
        secret_id = native_worker_secret_id(management_uid)
        existing = self._describe(secret_id, project=project)
        if existing is None:
            create = self._run(
                (
                    "secrets",
                    "create",
                    secret_id,
                    "--project",
                    project,
                    "--replication-policy=automatic",
                    "--labels",
                    f"eyvan-managed=true,eyvan-uid={management_uid}",
                )
            )
            if create.returncode != 0:
                existing = self._describe(secret_id, project=project)
                if existing is None:
                    raise GcloudError(
                        (create.stderr or create.stdout).strip()
                        or f"could not create native worker secret {secret_id!r}"
                    )
        if existing is not None:
            self._validate_ownership(secret_id, existing, management_uid)

        descriptor, path = tempfile.mkstemp(prefix=".eyvan-native-secret-")
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            version = self._run(("secrets", "versions", "add", secret_id, "--project", project, "--data-file", path))
            if version.returncode != 0:
                raise GcloudError(
                    (version.stderr or version.stdout).strip() or f"could not rotate native worker secret {secret_id!r}"
                )
        finally:
            Path(path).unlink(missing_ok=True)

        binding = self._run(
            (
                "secrets",
                "add-iam-policy-binding",
                secret_id,
                "--project",
                project,
                "--member",
                f"serviceAccount:{reader_service_account}",
                "--role",
                "roles/secretmanager.secretAccessor",
                "--quiet",
            )
        )
        if binding.returncode != 0:
            raise GcloudError(
                (binding.stderr or binding.stdout).strip()
                or f"could not grant native worker access to secret {secret_id!r}"
            )
        return f"projects/{project}/secrets/{secret_id}/versions/latest"

    def delete(self, *, project: str, management_uid: str) -> None:
        """Delete a cell-owned native-worker secret after its VM is gone."""

        secret_id = native_worker_secret_id(management_uid)
        existing = self._describe(secret_id, project=project)
        if existing is None:
            return
        self._validate_ownership(secret_id, existing, management_uid)
        deleted = self._run(("secrets", "delete", secret_id, "--project", project, "--quiet"))
        if deleted.returncode != 0 and self._describe(secret_id, project=project) is not None:
            raise GcloudError((deleted.stderr or deleted.stdout).strip() or f"could not delete secret {secret_id!r}")

    def list_managed(self, *, project: str) -> tuple[tuple[str, str], ...]:
        """Return every ``(management_uid, secret_id)`` this runtime owns in ``project``.

        Ownership is decided exactly as ``ensure`` decides it when creating:
        the ``eyvan-managed=true`` label carries the decision and
        ``eyvan-uid`` names the cell. Entries missing either mark are skipped
        rather than guessed about -- a janitor must never act on an ambiguous
        identity in a project it shares.
        """

        result = self._run(
            ("secrets", "list", "--project", project, "--filter=labels.eyvan-managed=true", "--format=json")
        )
        if result.returncode != 0:
            raise GcloudError(
                (result.stderr or result.stdout).strip() or f"could not list native worker secrets in {project!r}"
            )
        try:
            decoded = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise GcloudError(f"native worker secret listing returned invalid JSON: {exc.msg}") from exc
        if not isinstance(decoded, list):
            raise GcloudError("native worker secret listing did not return a JSON array")
        managed: list[tuple[str, str]] = []
        for entry in decoded:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            labels = entry.get("labels")
            if not isinstance(name, str) or not isinstance(labels, dict):
                continue
            uid = labels.get("eyvan-uid")
            if not isinstance(uid, str) or not uid:
                continue
            managed.append((uid, name.rsplit("/", 1)[-1]))
        return tuple(managed)


class GcloudNativeWorkerArtifactAccessStore:
    """Grant native workers the minimum bucket access needed for Eyvan artifacts."""

    _ROLES: tp.ClassVar[tuple[str, str]] = (
        "roles/storage.objectViewer",
        "roles/storage.objectCreator",
    )

    #: Bucket IAM is a read-modify-write guarded by an etag: two grants racing on
    #: one bucket make the loser fail ``HTTPError 412``. Retry these conflicts
    #: rather than misclassifying them as missing permissions.
    _PRECONDITION_ATTEMPTS: tp.ClassVar[int] = 6
    _PRECONDITION_BACKOFF_S: tp.ClassVar[float] = 0.5
    _BUCKET_LOCKS: tp.ClassVar[dict[str, threading.Lock]] = {}
    _BUCKET_LOCKS_GUARD: tp.ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, *, runner: Runner = _runner, sleep: tp.Callable[[float], None] = time.sleep):
        self._runner = runner
        self._sleep = sleep

    def _run(self, args: tp.Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Run one gcloud call, surfacing its subprocess timeout as a typed provider failure."""
        try:
            return self._runner(args)
        except subprocess.TimeoutExpired as exc:
            raise GcloudError(str(exc), code=CapacityProviderFailureCode.PROVIDER_ERROR) from exc

    @classmethod
    def _bucket_lock(cls, bucket_uri: str) -> threading.Lock:
        with cls._BUCKET_LOCKS_GUARD:
            return cls._BUCKET_LOCKS.setdefault(bucket_uri, threading.Lock())

    @staticmethod
    def _lost_precondition(result: subprocess.CompletedProcess[str]) -> bool:
        # The HTTP status as gcloud prints it, not any "412": EIDs, project
        # numbers and request ids contain those digits too, and a permission
        # failure mistaken for a race is retried and then misreported.
        detail = f"{result.stdout}\n{result.stderr}".casefold()
        return "httperror 412" in detail or "precondition" in detail

    def ensure(
        self,
        *,
        project: str,
        artifact_bucket: GcsArtifactBucket,
        reader_service_account: str,
    ) -> None:
        """Ensure the worker can read inputs and create result objects in one bucket.

        ``gcloud storage buckets add-iam-policy-binding`` is idempotent, so this
        remains safe across controller retries and replacement capacity.

        The matching grant is deliberately never revoked on release. The reader
        identity is the worker VM's service account -- by default the
        project-wide Compute default service account, which every other VM in
        the project may also be using -- and these roles are scoped to exactly
        one artifact bucket. Revoking on release therefore risks breaking
        unrelated workloads that share the identity to save one narrow binding
        on one bucket, so the grant is left in place and auditable through
        normal IAM tooling. Revisit only if workers move to dedicated
        per-cell identities.
        """

        member = f"serviceAccount:{reader_service_account}"
        # In-process grants to one bucket take turns; the bounded retry below is
        # for the race this lock cannot see -- another agent, or a human.
        with self._bucket_lock(artifact_bucket.uri):
            for role in self._ROLES:
                self._grant(project=project, bucket_uri=artifact_bucket.uri, member=member, role=role)

    def _grant(self, *, project: str, bucket_uri: str, member: str, role: str) -> None:
        args = (
            "storage",
            "buckets",
            "add-iam-policy-binding",
            bucket_uri,
            "--member",
            member,
            "--role",
            role,
            "--condition=None",
            f"--project={project}",
            "--quiet",
        )
        for attempt in range(1, self._PRECONDITION_ATTEMPTS + 1):
            result = self._run(args)
            if result.returncode == 0:
                return
            if not self._lost_precondition(result) or attempt == self._PRECONDITION_ATTEMPTS:
                break
            self._sleep(self._PRECONDITION_BACKOFF_S * attempt * (1 + random.random()))
        diagnostic = (result.stderr or result.stdout).strip()
        if self._lost_precondition(result):
            raise GcloudError(
                f"could not configure native worker artifact access on {bucket_uri}: the bucket's IAM policy kept "
                f"changing underneath {self._PRECONDITION_ATTEMPTS} attempts to grant {role} (concurrent "
                f"modification, not a permission problem): {diagnostic}",
                code=CapacityProviderFailureCode.PROVIDER_ERROR,
            )
        raise GcloudError(
            f"could not configure native worker artifact access on {bucket_uri}; Eyvan needs permission to grant "
            f"{role} to the worker identity before provisioning: {diagnostic or 'gcloud gave no diagnostic'}"
        )


class GcloudComputeProvider:
    """Idempotent, ownership-checked Compute Engine instance lifecycle."""

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
    def _parse_instance(raw: JsonObject) -> ComputeInstance:
        machine_type = raw.get("machineType")
        accelerators: list[tuple[str, int]] = []
        accelerator_values = raw.get("guestAccelerators")
        for accelerator_value in accelerator_values if isinstance(accelerator_values, list) else []:
            if not is_json_object(accelerator_value):
                continue
            raw_type = accelerator_value.get("acceleratorType")
            accelerator_type = raw_type.rsplit("/", 1)[-1] if isinstance(raw_type, str) else ""
            if accelerator_type:
                raw_count = accelerator_value.get("acceleratorCount", 0)
                if isinstance(raw_count, bool) or not isinstance(raw_count, (str, int, float)):
                    raise GcloudError("Compute instance returned an invalid accelerator count")
                try:
                    accelerator_count = int(raw_count)
                except ValueError as exc:
                    raise GcloudError("Compute instance returned an invalid accelerator count") from exc
                accelerators.append((accelerator_type, accelerator_count))
        name_value = raw.get("name")
        status_value = raw.get("status")
        if not isinstance(status_value, str) or not status_value:
            raise GcloudError("Compute instance did not report a status")
        try:
            status = GcpComputeInstanceStatus(status_value.upper())
        except ValueError as exc:
            raise GcloudError(f"Compute instance reported unsupported status {status_value!r}") from exc
        return ComputeInstance(
            name=(name_value if isinstance(name_value, str) and name_value else "?").rsplit("/", 1)[-1],
            status=status,
            machine_type=machine_type.rsplit("/", 1)[-1] if isinstance(machine_type, str) and machine_type else None,
            accelerators=tuple(sorted(accelerators)),
            raw=raw.copy(),
        )

    def describe(self, name: str, *, project: str, zone: str) -> ComputeInstance | None:
        args = ("compute", "instances", "describe", name, "--zone", zone, "--project", project, "--format=json")
        result = self._run(args)
        if self._not_found(result):
            return None
        if result.returncode != 0:
            raise GcloudError((result.stderr or result.stdout).strip() or f"could not describe Compute instance {name}")
        value = decode_gcloud_object(result.stdout, resource=f"Compute instance {name!r}")
        return self._parse_instance(value)

    def absence_cause(
        self, name: str, *, project: str, zone: str
    ) -> tuple[str, CapacityProviderFailureCode | None] | None:
        """Return what the provider's own operation log says happened to a vanished instance.

        A Spot VM is created with ``instanceTerminationAction=DELETE``, so both a
        reclamation and an insert that failed after the create call returned
        (the waiter interrupted, or GCE giving up after ``PROVISIONING``) leave
        the same evidence behind: no instance. The zone's operation log can
        distinguish the causes and preserve the provider's diagnostic instead
        of reducing both to a generic absence.

        Best effort by design: any failure to read the log answers ``None`` and
        the caller keeps its generic observation. Diagnosis must never be the
        reason a replacement does not happen.
        """

        args = (
            "compute",
            "operations",
            "list",
            "--project",
            project,
            "--zones",
            zone,
            f"--filter=targetLink ~ /instances/{name}$",
            "--sort-by=~insertTime",
            "--limit=5",
            "--format=json",
        )
        try:
            result = self._run(args)
        except GcloudError:
            return None
        if result.returncode != 0:
            return None
        try:
            operations = json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            return None
        if not isinstance(operations, list):
            return None
        for operation in operations:
            if not is_json_object(operation):
                continue
            operation_type = operation.get("operationType")
            if operation_type == "compute.instances.preempted":
                return (
                    "Compute instance was reclaimed by the provider (Spot preemption)",
                    CapacityProviderFailureCode.PREEMPTED,
                )
            if operation_type != "insert":
                continue
            error = operation.get("error")
            errors = error.get("errors") if is_json_object(error) else None
            messages = [
                " ".join(part for part in (item.get("code"), item.get("message")) if isinstance(part, str) and part)
                for item in (errors if isinstance(errors, list) else [])
                if is_json_object(item)
            ]
            diagnostic = "; ".join(message for message in messages if message)
            if not diagnostic:
                return None
            return (f"Compute instance creation failed: {diagnostic}", provider_failure_from_diagnostic(diagnostic))
        return None

    @staticmethod
    def _validate_instance(instance: ComputeInstance, spec: ComputeInstanceSpec) -> None:
        labels_value = instance.raw.get("labels")
        labels = labels_value if is_json_object(labels_value) else {}
        expected_uid = spec.labels.get("eyvan-uid")
        if not expected_uid or labels.get("eyvan-uid") != expected_uid:
            raise GcloudError(f"Compute instance {instance.name!r} is not owned by this Eyvan cell")
        if labels.get("eyvan-cell") != spec.labels.get("eyvan-cell"):
            raise GcloudError(f"Compute instance {instance.name!r} has a different Eyvan cell identity")
        if instance.machine_type is not None and instance.machine_type != spec.machine_type:
            raise GcloudError(
                f"Compute instance {instance.name!r} uses {instance.machine_type!r}, expected {spec.machine_type!r}"
            )
        accelerator_type = spec.gpu_type or spec.tpu_type
        accelerator_count = spec.gpu_count if spec.gpu_type else spec.tpu_count
        expected = () if accelerator_type is None else ((accelerator_type, accelerator_count),)
        if instance.accelerators != expected:
            raise GcloudError(
                f"Compute instance {instance.name!r} has accelerators {instance.accelerators!r}, expected {expected!r}"
            )
        scheduling_value = instance.raw.get("scheduling")
        scheduling = scheduling_value if is_json_object(scheduling_value) else {}
        if spec.capacity_class == "spot":
            provisioning_model = scheduling.get("provisioningModel")
            termination_action = scheduling.get("instanceTerminationAction")
            maintenance_action = scheduling.get("onHostMaintenance")
            observed = tuple(
                value.upper() if isinstance(value, str) else ""
                for value in (provisioning_model, termination_action, maintenance_action)
            )
            # ``preemptible`` is a legacy Standard-instance field. Spot VMs
            # are identified by provisioningModel plus their termination and
            # maintenance behavior; GCE may omit the legacy boolean entirely.
            expected_spot = ("SPOT", "DELETE", "TERMINATE")
            if observed != expected_spot:
                raise GcloudError(
                    f"Compute instance {instance.name!r} has scheduling {observed!r}, expected Spot {expected_spot!r}"
                )

    def create(self, spec: ComputeInstanceSpec) -> ComputeInstance:
        existing = self.describe(spec.name, project=spec.project, zone=spec.zone)
        if existing is not None:
            self._validate_instance(existing, spec)
            return existing
        if spec.startup_script:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as script:
                script.write(spec.startup_script)
                script.flush()
                result = self._run(spec.create_args(startup_script_path=script.name))
        else:
            result = self._run(spec.create_args())
        if result.returncode != 0:
            existing = self.describe(spec.name, project=spec.project, zone=spec.zone)
            if existing is not None:
                self._validate_instance(existing, spec)
                return existing
            diagnostic = (result.stderr or result.stdout).strip() or f"could not create Compute instance {spec.name}"
            classified = provider_failure_from_diagnostic(diagnostic)
            if classified is None:
                raise GcloudError(diagnostic)
            raise GcloudError(diagnostic, code=classified)
        created = self.describe(spec.name, project=spec.project, zone=spec.zone)
        if created is None:
            raise GcloudError(f"Compute instance {spec.name!r} was not visible after creation")
        self._validate_instance(created, spec)
        return created

    def delete(self, name: str, *, project: str, zone: str, management_uid: str | None = None) -> None:
        existing = self.describe(name, project=project, zone=zone)
        if existing is None:
            return
        labels_value = existing.raw.get("labels")
        labels = labels_value if is_json_object(labels_value) else {}
        if not isinstance(management_uid, str) or not management_uid or labels.get("eyvan-uid") != management_uid:
            raise GcloudError(f"refusing to delete Compute instance {name!r} without its Eyvan ownership UID")
        args = ("compute", "instances", "delete", name, "--zone", zone, "--project", project, "--quiet")
        result = self._run(args)
        if result.returncode != 0:
            if self.describe(name, project=project, zone=zone) is not None:
                raise GcloudError(
                    (result.stderr or result.stdout).strip() or f"could not delete Compute instance {name}"
                )
            return
        deadline = time.monotonic() + self.delete_wait_timeout_s
        while True:
            try:
                if self.describe(name, project=project, zone=zone) is None:
                    return
            except (GcloudError, subprocess.TimeoutExpired):
                # A transient describe failure must not abandon the delete
                # wait; keep polling until the deadline.
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Typed like the TPU path: a bare TimeoutError carries no
                # retryability or failure code, so the controller cannot tell
                # it from a real failure.
                raise GcloudError(
                    f"Compute instance {name!r} still exists after delete "
                    f"(waited {self.delete_wait_timeout_s:g} seconds for its removal)",
                    code=CapacityProviderFailureCode.PROVIDER_ERROR,
                )
            time.sleep(min(2, remaining))

    @staticmethod
    def _head_ip(instance: ComputeInstance, *, endpoint_mode: NativeWorkerEndpointMode) -> str | None:
        """Select the private or profile-managed public address for one instance."""

        interface_values = instance.raw.get("networkInterfaces")
        for interface_value in interface_values if isinstance(interface_values, list) else []:
            if not is_json_object(interface_value):
                continue
            access_values = interface_value.get("accessConfigs")
            if endpoint_mode is NativeWorkerEndpointMode.PRIVATE:
                network_ip = interface_value.get("networkIP")
                if isinstance(network_ip, str) and network_ip:
                    try:
                        return normalize_ip_address(network_ip)
                    except ValueError as exc:
                        raise GcloudError("Compute instance returned an invalid network IP address") from exc
            for access_value in access_values if isinstance(access_values, list) else []:
                if not is_json_object(access_value):
                    continue
                nat_ip = access_value.get("natIP")
                if isinstance(nat_ip, str) and nat_ip:
                    try:
                        return normalize_ip_address(nat_ip)
                    except ValueError as exc:
                        raise GcloudError("Compute instance returned an invalid NAT IP address") from exc
        return None

    @staticmethod
    def _execution_endpoint(record: FleetCellRecord, head_ip: str | None) -> str | None:
        """Derive the selected native-worker route for an Eyvan-owned VM."""

        if record.metadata.get("execution_backend") != "eyvan-native" or not head_ip:
            return None
        raw_port = record.metadata.get("native_worker_port", 8100)
        if isinstance(raw_port, bool):
            raise GcloudError(f"cell {record.name!r} has an invalid native worker port")
        try:
            port = int(raw_port)
        except (TypeError, ValueError) as exc:
            raise GcloudError(f"cell {record.name!r} has an invalid native worker port") from exc
        if port <= 0 or port > 65535:
            raise GcloudError(f"cell {record.name!r} has an invalid native worker port")
        return format_http_ip_endpoint(head_ip, port)

    @staticmethod
    def _spec(record: FleetCellRecord) -> ComputeInstanceSpec:
        config = record.metadata.get("compute_config")
        if not isinstance(config, dict):
            raise GcloudError(f"cell {record.name!r} has no Compute provider configuration")
        management_uid = record.metadata.get("management_uid")
        if not isinstance(management_uid, str) or not management_uid:
            raise GcloudError(f"cell {record.name!r} has no Eyvan management UID")
        if not (record.project and record.zone and record.qr_id):
            raise ValueError(f"cell {record.name!r} is missing project, zone, or Compute instance ID")
        normalized = normalize_compute_config(config)
        return ComputeInstanceSpec(
            name=record.qr_id,
            project=record.project,
            zone=record.zone,
            machine_type=normalized["machine_type"],
            capacity_class=record.capacity_class,
            gpu_type=normalized["gpu_type"],
            gpu_count=normalized["gpu_count"],
            tpu_type=normalized["tpu_type"],
            tpu_count=normalized["tpu_count"],
            image=normalized["image"],
            image_family=normalized["image_family"],
            image_project=normalized["image_project"],
            boot_disk_size_gb=normalized["boot_disk_size_gb"],
            boot_disk_type=normalized["boot_disk_type"],
            network=normalized["network"],
            subnetwork=normalized["subnetwork"],
            service_account=normalized["service_account"],
            scopes=tuple(normalized["scopes"]),
            tags=tuple(normalized["tags"]),
            labels={"eyvan-cell": _cell_label(record.name), "eyvan-uid": management_uid},
            metadata={**normalized["metadata"], "eyvan-uid": management_uid},
            startup_script=normalized["startup_script"],
            internal_ip_only=normalized["internal_ip_only"],
            maintenance_policy=normalized["maintenance_policy"],
            reservation=normalized["reservation"],
        )

    def ensure(self, record: FleetCellRecord) -> ProviderObservation:
        if record.desired_state != "up":
            return ProviderObservation(CapacityStatus.DOWN, "desired state is down")
        try:
            endpoint_mode = endpoint_mode_from_metadata(record.metadata)
        except ValueError as exc:
            raise GcloudError(f"cell {record.name!r} has an invalid native worker endpoint mode") from exc
        spec = self._spec(record)
        instance = self.describe(spec.name, project=spec.project, zone=spec.zone)
        if instance is None:
            # A previously observed instance disappeared. Do not recreate the
            # same generation: a Spot VM may already have exposed its worker
            # credential, and callers need the driver's fenced replacement
            # path to rotate the resource ID and worker token first.
            if record.state not in {CapacityStatus.UNOBSERVED, CapacityStatus.RECREATING}:
                cause = self.absence_cause(spec.name, project=spec.project, zone=spec.zone)
                if cause is None:
                    return ProviderObservation(CapacityStatus.NEEDS_RECREATE, "Compute instance is absent")
                detail, failure_code = cause
                return ProviderObservation(CapacityStatus.NEEDS_RECREATE, detail, failure_code=failure_code)
            instance = self.create(spec)
            return ProviderObservation(CapacityStatus.WAITING, f"Compute instance created ({instance.status})")
        self._validate_instance(instance, spec)
        head_ip = self._head_ip(instance, endpoint_mode=endpoint_mode)
        if instance.status is GcpComputeInstanceStatus.RUNNING and head_ip:
            return ProviderObservation(
                CapacityStatus.HEALTHY,
                "Compute instance is running",
                head_ip=head_ip,
                execution_endpoint=self._execution_endpoint(record, head_ip),
                runtime_process_count=1,
            )
        if instance.status is GcpComputeInstanceStatus.RUNNING:
            return ProviderObservation(
                CapacityStatus.ACTIVE, "Compute instance is running without an observed IP address"
            )
        if instance.status in {
            GcpComputeInstanceStatus.PROVISIONING,
            GcpComputeInstanceStatus.STAGING,
            GcpComputeInstanceStatus.REPAIRING,
        }:
            return ProviderObservation(
                CapacityStatus.PROVISIONING,
                f"Compute instance is {instance.status.lower()}",
                head_ip=head_ip,
            )
        if instance.status in {GcpComputeInstanceStatus.STOPPING, GcpComputeInstanceStatus.SUSPENDING}:
            return ProviderObservation(
                CapacityStatus.DRAINING, f"Compute instance is {instance.status.lower()}", head_ip=head_ip
            )
        if instance.status in {GcpComputeInstanceStatus.TERMINATED, GcpComputeInstanceStatus.SUSPENDED}:
            # GCE has no distinct PREEMPTED status the way Cloud TPU does: a
            # reclaimed Spot VM simply reaches TERMINATED. What distinguishes it
            # is the purchase class, which ``_validate_instance`` has already
            # confirmed against the spec -- an Eyvan-owned Spot instance that
            # stopped without Eyvan stopping it was reclaimed.
            #
            # Limited on purpose to Spot and to TERMINATED. A Standard instance
            # reaching TERMINATED is a guest shutdown or a host failure, and a
            # SUSPENDED instance is neither; claiming preemption for those would
            # be the same guess this exists to remove.
            preempted = spec.capacity_class == "spot" and instance.status is GcpComputeInstanceStatus.TERMINATED
            return ProviderObservation(
                CapacityStatus.NEEDS_RECREATE,
                f"Compute instance is terminal ({instance.status})",
                failure_code=CapacityProviderFailureCode.PREEMPTED if preempted else None,
            )
        return ProviderObservation(CapacityStatus.UNAVAILABLE, f"Compute instance is {instance.status}", head_ip=head_ip)


__all__ = (
    "ComputeInstance",
    "ComputeInstanceSpec",
    "ComputeProviderConfig",
    "GcloudComputeProvider",
    "GcloudDefaultComputeServiceAccountResolver",
    "GcloudNativeWorkerArtifactAccessStore",
    "GcloudNativeWorkerSecretStore",
    "GcpComputeInstanceStatus",
    "GcsArtifactBucket",
    "compute_instance_name",
    "native_worker_secret_id",
    "normalize_compute_config",
)
