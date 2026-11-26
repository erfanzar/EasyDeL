# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Stable control-plane records shared by services and persistence adapters."""

from __future__ import annotations

import hashlib
import json
import re
import typing as tp
from datetime import datetime, timedelta
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic import JsonValue as PydanticJsonValue

from eyvan._internal.compat import UTC, StrEnum
from eyvan._internal.correlation_ids import CORRELATION_ID_WIRE_PATTERN
from eyvan._internal.json_types import is_json_object
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import (
    CapacityProviderFailureCode,
    backend_capabilities,
    canonical_execution_backend,
    max_multi_worker_cell_width,
    providers_hosting_multi_worker_cells,
)
from eyvan.domain import costing
from eyvan.domain.models import InfeasibilityCode, PlacementMode, WorkerResourceScope
from eyvan.identity.rotation import CredentialRotationState
from eyvan.secrets import validate_secret_reference
from eyvan.worker.distributed_environment import (
    DISTRIBUTED_TOPOLOGY_ENVIRONMENT_NAMES,
    is_distributed_topology_environment_name,
    is_runtime_owned_environment_name,
)

JsonDocument: tp.TypeAlias = dict[str, PydanticJsonValue]
"""A recursively JSON-compatible document accepted by Pydantic models."""

_SAFE_FLEET_COMPONENT = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_MAX_RETRIES = 3
#: Widest multi-machine cell any known backend builds. Derived from the backend
#: capability table rather than written here, so the durable model stops being
#: the place a provider's ceiling is recorded. Kept under its original name
#: because it is imported across the package and by tests.
MAX_GCP_TPU_MULTISLICE_WORKERS = backend_capabilities("gcp-tpu").max_workers_per_cell
WORKER_COUNT_LABEL = "eyvan_worker_count"
#: Additional pool ids a workload will accept, comma separated. ``eyvan_pool``
#: stays the first choice, so a spec written before this existed behaves
#: identically.
POOL_ALTERNATES_LABEL = "eyvan_pool_alternates"
#: The template fingerprint of each alternate, comma separated and aligned with
#: ``POOL_ALTERNATES_LABEL``. ``eyvan_pool_template`` pins the *primary* pool's
#: template, and a template includes the zone, so without these an alternate's
#: cells could never satisfy the pin that keeps a rollout from mixing templates.
POOL_ALTERNATE_TEMPLATES_LABEL = "eyvan_pool_alternate_templates"
POOL_LABEL = "eyvan_pool"
POOL_TEMPLATE_LABEL = "eyvan_pool_template"


def candidate_pool_templates(labels: tp.Mapping[str, str]) -> dict[str, str | None]:
    """Return each pool this workload may run in, and the template it must carry there.

    ``None`` means the workload pins no template in that pool. An alternate
    submitted before alternates' templates were recorded is left out: the
    scheduler keeps such a workload to its first choice's template, so counting
    the alternate's cells as its capacity would report supply it cannot use.
    """

    primary = labels.get(POOL_LABEL)
    if not primary:
        return {}
    candidates: dict[str, str | None] = {primary: labels.get(POOL_TEMPLATE_LABEL)}
    alternates = [value.strip() for value in (labels.get(POOL_ALTERNATES_LABEL) or "").split(",") if value.strip()]
    templates = [
        value.strip() for value in (labels.get(POOL_ALTERNATE_TEMPLATES_LABEL) or "").split(",") if value.strip()
    ]
    if len(templates) == len(alternates):
        for pool_id, template in zip(alternates, templates, strict=True):
            candidates.setdefault(pool_id, template)
    return candidates


CALLABLE_SIGNING_KEY_ENV = "EYVAN_CALLABLE_SIGNING_KEY"
REMOTE_CALLABLE_SIGNATURE_ERROR = (
    "Eyvan remote callable signatures are not supported because no controller-managed verifier is available"
)
REMOTE_CALLABLE_SIGNING_KEY_ERROR = (
    "Eyvan remote callable signing keys are not supported because Eyvan never forwards caller secrets to workers. "
    "Remove signing_key or EYVAN_CALLABLE_SIGNING_KEY; controller-managed callable signing is not available yet."
)

# These values are injected by the controller or native slice executor.  A
# fleet runtime cannot own them, otherwise a submitted program could break its
# fencing or distributed JAX rendezvous contract.
_RESERVED_RUNTIME_ENVIRONMENT_NAMES = frozenset(
    {
        "HOME",
        "LOGNAME",
        "PATH",
        "USER",
        "VIRTUAL_ENV",
        "XDG_CACHE_HOME",
        "EYVAN_WORKLOAD_ID",
        "EYVAN_ATTEMPT",
        "EYVAN_RESUME_CHECKPOINT_URI",
    }
    | DISTRIBUTED_TOPOLOGY_ENVIRONMENT_NAMES
)
_PROTECTED_CONTAINER_MOUNT_SOURCES = (
    PurePosixPath("/etc/eyvan-native"),
    PurePosixPath("/run/eyvan-native"),
)
_AUTOMATIC_CONTAINER_MOUNT_TARGETS = frozenset(
    {
        "/opt/eyvan-native/oci-callable-runner",
        "/opt/eyvan-native/multihost",
        "/var/lib/eyvan/workload",
    }
)


def native_container_mount_targets_overlap(left: str, right: str) -> bool:
    """Return whether two absolute container mount targets overlap.

    Docker permits nested bind mounts, but that lets a later mount shadow files
    from an earlier one.  Eyvan uses bind mounts for its workload directory,
    multi-host rendezvous data, and the OCI callable bridge, so target paths
    must be disjoint rather than merely distinct strings.
    """

    left_path = PurePosixPath(left)
    right_path = PurePosixPath(right)
    return left_path == right_path or left_path in right_path.parents or right_path in left_path.parents


def _first_overlapping_mount_target(targets: tp.Sequence[str]) -> str | None:
    """Return the later target in the first overlapping pair, if any."""

    for index, target in enumerate(targets):
        if any(native_container_mount_targets_overlap(target, prior) for prior in targets[:index]):
            return target
    return None


def utcnow() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(UTC)


def _tpu_node_prefix(cell_id: str) -> str:
    """Return a deterministic GCP-safe node prefix for one physical cell."""

    normalized = re.sub(r"[^a-z0-9-]+", "-", cell_id.lower()).strip("-") or "eyvan-cell"
    direct = f"{normalized}-node"
    if len(direct) <= 63:
        return direct
    digest = hashlib.sha256(cell_id.encode()).hexdigest()[:8]
    return f"{normalized[:49].rstrip('-')}-{digest}-node"


def _validate_string_map(
    values: tp.Mapping[str, PydanticJsonValue],
    *,
    field_name: str,
    forbidden_keys: frozenset[str] = frozenset(),
    validate_string_values: bool = True,
) -> None:
    for key, value in values.items():
        if not key or key != key.strip():
            raise ValueError(f"{field_name} keys must be non-empty and trimmed")
        if key.casefold() in {candidate.casefold() for candidate in forbidden_keys}:
            raise ValueError(f"{field_name} key {key!r} collides with a built-in field")
        if validate_string_values and isinstance(value, str) and (not value or value != value.strip()):
            raise ValueError(f"{field_name} values must be non-empty and trimmed")


def _validate_strings(values: tp.Iterable[str], *, field_name: str) -> None:
    if any(not value or value != value.strip() for value in values):
        raise ValueError(f"{field_name} values must be non-empty and trimmed")


def _environment_name_is_reserved(name: str) -> bool:
    """Return whether Eyvan rather than a workload owns an environment name."""

    return (
        name in _RESERVED_RUNTIME_ENVIRONMENT_NAMES
        or name.startswith("EYVAN_SECRET_")
        or is_distributed_topology_environment_name(name)
    )


def validate_secret_mount_contract(
    environment: tp.Mapping[str, str],
    secret_mounts: tp.Mapping[str, str],
) -> None:
    """Apply the ordinary workload-environment contract to secret mounts."""

    _validate_string_map(secret_mounts, field_name="secret mounts")
    for name, reference in secret_mounts.items():
        if not _ENVIRONMENT_NAME.fullmatch(name):
            raise ValueError("secret mount names must be valid shell identifiers")
        if is_runtime_owned_environment_name(name):
            raise ValueError(f"secret mount name {name!r} is reserved by Eyvan")
        if name in environment:
            raise ValueError(f"secret mount name {name!r} collides with the workload environment")
        if "\x00" in reference or "\n" in reference or "\r" in reference:
            raise ValueError(f"secret mount reference for {name!r} must not contain NUL bytes or newlines")
        validate_secret_reference(reference)


def _validate_string(value: str | None, *, field_name: str, optional: bool = False) -> None:
    if value is None and optional:
        return
    if value is None or not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty and trimmed")


def resource_quantities(resources: ResourceVector) -> dict[str, float]:
    """Flatten a resource vector to ``{dimension: quantity}``, custom names included.

    Public because federated quota headroom has to hold in *every* dimension at
    once, and the alternative was a fourth copy of these five lines.
    """

    return {
        "cpu": resources.cpu,
        "memory_gib": resources.memory_gib,
        "gpu": resources.gpu,
        "tpu": resources.tpu,
        **resources.custom,
    }


class EyvanModel(BaseModel):
    """Base model for persisted and API-facing control-plane data."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class NativeRuntimeArtifact(EyvanModel):
    """A GCS artifact staged on every native capacity host before work starts."""

    source: str
    destination: str
    recursive: bool = False

    @model_validator(mode="after")
    def validate_artifact(self) -> NativeRuntimeArtifact:
        if not self.source.startswith("gs://") or not self.source[5:].split("/", 1)[0]:
            raise ValueError("native runtime artifact source must be a gs:// URI")
        if "\x00" in self.source:
            raise ValueError("native runtime artifact source must not contain NUL bytes")
        destination = PurePosixPath(self.destination)
        if not destination.is_absolute():
            raise ValueError("native runtime artifact destination must be an absolute path")
        if "\x00" in self.destination or ".." in destination.parts:
            raise ValueError("native runtime artifact destination must not contain NUL bytes or '..'")
        if str(destination) == "/" or str(destination).startswith("//"):
            raise ValueError("native runtime artifact destination must not be root or start with '//'")
        return self


class NativeContainerMount(EyvanModel):
    """One explicit host path made available to an Eyvan-native OCI workload."""

    source: str
    target: str
    read_only: bool = True

    @model_validator(mode="after")
    def validate_mount(self) -> NativeContainerMount:
        for field_name, value in (("source", self.source), ("target", self.target)):
            if value != value.strip() or "\x00" in value:
                raise ValueError(f"native container mount {field_name} must be trimmed and NUL-free")
            path = PurePosixPath(value)
            if not path.is_absolute():
                raise ValueError(f"native container mount {field_name} must be an absolute path")
            if ".." in path.parts or str(path) == "/" or str(path).startswith("//"):
                raise ValueError(
                    f"native container mount {field_name} must not be root, start with '//', or contain '..'"
                )
            if field_name == "source" and any(
                path == protected or protected in path.parents for protected in _PROTECTED_CONTAINER_MOUNT_SOURCES
            ):
                raise ValueError(f"native container mount source {value!r} exposes Eyvan-native credentials")
        return self


class NativeContainerConfig(EyvanModel):
    """Structured OCI execution settings for an Eyvan-native workload.

    This intentionally exposes only stable runtime properties. Arbitrary Docker
    command-line flags are not part of Eyvan's durable fleet API.
    """

    image: str
    pull_policy: Literal["always", "if-not-present", "never"] = "if-not-present"
    working_dir: str | None = None
    privileged: bool = False
    mounts: tuple[NativeContainerMount, ...] = ()

    @model_validator(mode="after")
    def validate_container(self) -> NativeContainerConfig:
        image = self.image
        if (
            image != image.strip()
            or "\x00" in image
            or any(character.isspace() for character in image)
            or image.startswith(("-", "/", "."))
            or "://" in image
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@:+-]*", image)
        ):
            raise ValueError("native container image must be an OCI image reference")
        if self.working_dir is not None:
            working_dir = self.working_dir
            if working_dir != working_dir.strip() or "\x00" in working_dir:
                raise ValueError("native container working_dir must be trimmed and NUL-free")
            path = PurePosixPath(working_dir)
            if not path.is_absolute() or ".." in path.parts or str(path) == "/" or str(path).startswith("//"):
                raise ValueError("native container working_dir must be an absolute non-root path without '..'")
        targets = [str(PurePosixPath(mount.target)) for mount in self.mounts]
        if len(targets) != len(set(targets)):
            raise ValueError("native container mounts must not share a target")
        conflicting_target = _first_overlapping_mount_target(targets)
        if conflicting_target is not None:
            raise ValueError(
                f"native container mount target {conflicting_target!r} overlaps another configured mount target"
            )
        return self


class NativeRuntimeConfig(EyvanModel):
    """Declarative host setup and workload defaults for Eyvan-native cells.

    The configuration is deliberately host-oriented. ``vm_image`` selects a
    GCP Compute OS image, while ``container`` describes OCI execution handled
    by the Eyvan-native runtime.
    """

    env: dict[str, str] = Field(default_factory=dict)
    init_commands: tuple[str, ...] = ()
    artifacts: tuple[NativeRuntimeArtifact, ...] = ()
    vm_image: str | None = None
    container: NativeContainerConfig | None = None
    #: The ``major.minor`` Python these cells run workloads on, when the operator
    #: states it. A serialized callable is pinned to the interpreter that pickled
    #: it and cannot load on another, and that mismatch was only discovered on the
    #: accelerator: a live multislice v5p-16 provisioned two slices, dispatched to
    #: four ranks, and every rank died with "callable requires Python 3.12;
    #: current runtime is 3.10" -- a fact knowable at submission, paid for with a
    #: full provisioning cycle.
    #:
    #: Optional because a fleet that runs only command workloads has no
    #: interpreter contract to state, and inventing one would refuse work that is
    #: fine. Declared rather than discovered because a scale-from-zero pool has no
    #: cell to ask when the submission arrives.
    python: str | None = None

    @field_validator("python")
    @classmethod
    def _valid_python(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parts = value.split(".")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError("runtime python must be a 'major.minor' version such as '3.11'")
        return value

    @property
    def is_empty(self) -> bool:
        """Return whether this runtime changes neither bootstrap nor workloads."""

        return not (self.env or self.init_commands or self.artifacts or self.vm_image or self.container is not None)

    def merged(self, override: NativeRuntimeConfig | None) -> NativeRuntimeConfig:
        """Apply a node-type runtime over fleet defaults deterministically."""

        if override is None:
            return self
        return NativeRuntimeConfig(
            env={**self.env, **override.env},
            init_commands=(*self.init_commands, *override.init_commands),
            artifacts=(*self.artifacts, *override.artifacts),
            vm_image=override.vm_image if override.vm_image is not None else self.vm_image,
            container=override.container if override.container is not None else self.container,
            python=override.python if override.python is not None else self.python,
        )

    @model_validator(mode="after")
    def validate_runtime(self) -> NativeRuntimeConfig:
        for name, value in self.env.items():
            if not _ENVIRONMENT_NAME.fullmatch(name):
                raise ValueError("native runtime environment names must be valid shell identifiers")
            if name in _RESERVED_RUNTIME_ENVIRONMENT_NAMES or name.startswith("EYVAN_"):
                raise ValueError(f"native runtime environment variable {name!r} is reserved by Eyvan")
            if "\x00" in value:
                raise ValueError(f"native runtime environment variable {name!r} must not contain NUL bytes")
        for command in self.init_commands:
            if not command.strip():
                raise ValueError("native runtime init command must be non-empty")
            if "\x00" in command:
                raise ValueError("native runtime init command must not contain NUL bytes")
        destinations = [artifact.destination for artifact in self.artifacts]
        conflicting_destination = _first_overlapping_mount_target(destinations)
        if conflicting_destination is not None:
            raise ValueError(
                f"native runtime artifact destination {conflicting_destination!r} overlaps another artifact destination"
            )
        if self.container is not None:
            automatic_artifact_target = next(
                (
                    destination
                    for destination in destinations
                    if any(
                        native_container_mount_targets_overlap(destination, automatic_target)
                        for automatic_target in _AUTOMATIC_CONTAINER_MOUNT_TARGETS
                    )
                ),
                None,
            )
            if automatic_artifact_target is not None:
                raise ValueError(
                    "native runtime artifact destination "
                    f"{automatic_artifact_target!r} conflicts with an Eyvan automatic mount"
                )
            reserved_targets = (*_AUTOMATIC_CONTAINER_MOUNT_TARGETS, *destinations)
            conflicting_target = next(
                (
                    mount.target
                    for mount in self.container.mounts
                    if any(
                        native_container_mount_targets_overlap(mount.target, reserved_target)
                        for reserved_target in reserved_targets
                    )
                ),
                None,
            )
            if conflicting_target is not None:
                raise ValueError(
                    f"native container mount target {conflicting_target!r} conflicts with an Eyvan automatic mount"
                )
        if self.vm_image is not None:
            if not self.vm_image.strip():
                raise ValueError("native runtime vm_image must be non-empty when set")
            if "\x00" in self.vm_image:
                raise ValueError("native runtime vm_image must not contain NUL bytes")
        return self


class WarmWindow(EyvanModel):
    """A recurring window during which a pool keeps a raised cell floor.

    Scale-to-zero saves real money overnight and costs it at 09:00, when every user
    pays the full provisioning latency simultaneously. A window is the smallest thing
    that fixes that without predicting anything.

    Time is a **fixed UTC offset**, not a named zone. Eyvan ships no timezone
    database, and a DST-aware window it cannot actually honour would be worse than
    one whose limits are stated: a fleet in a DST region shifts by an hour twice a
    year, which is visible and adjustable, rather than silently wrong.
    """

    min_cells: int = Field(ge=0, le=10_000)
    #: Half-open ``[start_hour, end_hour)`` so adjacent windows do not overlap by one
    #: hour. ``end_hour`` may be less than ``start_hour`` to wrap past midnight.
    start_hour: int = Field(ge=0, le=23)
    end_hour: int = Field(ge=0, le=23)
    #: ISO weekdays (Monday is 1). Empty means every day.
    days: tuple[int, ...] = ()
    utc_offset_minutes: int = Field(default=0, ge=-1440, le=1440)

    @model_validator(mode="after")
    def validate_window(self) -> WarmWindow:
        if self.start_hour == self.end_hour:
            raise ValueError("warm window start_hour and end_hour must differ; a zero-length window never applies")
        if any(day < 1 or day > 7 for day in self.days):
            raise ValueError("warm window days must be ISO weekdays from 1 (Monday) through 7 (Sunday)")
        if len(set(self.days)) != len(self.days):
            raise ValueError("warm window days must not repeat")
        return self

    def covers(self, moment: datetime) -> bool:
        """Return whether ``moment`` falls inside this window; naive times are UTC."""

        utc = moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
        local = utc + timedelta(minutes=self.utc_offset_minutes)
        start_day = local
        if self.start_hour < self.end_hour:
            inside = self.start_hour <= local.hour < self.end_hour
        else:
            inside = local.hour >= self.start_hour or local.hour < self.end_hour
            # The post-midnight segment belongs to the preceding local start day.
            if local.hour < self.end_hour:
                start_day = local - timedelta(days=1)
        return inside and (not self.days or start_day.isoweekday() in self.days)


class CoordinationKind(StrEnum):
    """What a coordinator record coordinates, and therefore how it settles.

    A coordinator owns children and has no lease of its own, so nothing but a
    dedicated controller phase can ever settle it. Naming the kind explicitly --
    rather than inferring "coordinator" from ``schedulable is False`` -- keeps that
    phase from capturing unrelated unschedulable records as they are introduced.
    """

    BATCH = "batch"
    DAG = "dag"
    #: A long-running service: a coordinator whose children are *replicas*.
    #: Settled by nobody. A batch parent finishes when its children do; a
    #: service's children are periodically all-terminal by design, which is the
    #: moment a replacement is due -- so the batch settler must not see it.
    SERVICE = "service"
    #: Differently-shaped members that admit together. Distinct from BATCH because
    #: the settler must not declare a group finished from a subset of its members,
    #: and distinct from DAG because there are no edges between them -- only the
    #: shared admission gate.
    GROUP = "group"


class WorkloadState(StrEnum):
    """Durable lifecycle states for a whole workload."""

    PENDING = "pending"
    ADMITTED = "admitted"
    LEASED = "leased"
    DISPATCHING = "dispatching"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class AutoscaleBlockReason(StrEnum):
    """Durable reason an autoscaler cannot add a workload's required capacity."""

    SHARED_TPU_QUOTA = "shared_tpu_quota"
    QUEUE_ADMISSION_DENIED = "queue_admission_denied"
    #: Every pool this workload accepts has failed to acquire capacity too many
    #: times in a row. Without a terminal state the autoscaler asks the same zone
    #: again every time the belief cooldown expires -- observed producing twenty
    #: cells for one workload over nineteen hours, one an hour, because
    #: ``max_backoff_s`` caps the cooldown at an hour and nothing counted the
    #: attempts. Retrying forever is indistinguishable from working, so the
    #: workload is told instead.
    CAPACITY_ACQUISITION_EXHAUSTED = "capacity_acquisition_exhausted"


#: Said when every pool a workload accepts has failed to acquire capacity
#: ``MAX_CONSECUTIVE_ACQUISITION_FAILURES`` times running. Names the remedy,
#: because the job is not going to progress where it is and the operator has to
#: choose a different zone or shape.
ACQUISITION_EXHAUSTED_SCHEDULING_REASON = (
    "every zone this job accepts has failed to grant capacity five times running; "
    "choose another zone or shape, or wait for the provider to free some"
)


class CellState(StrEnum):
    """Observed availability of an execution cell."""

    REGISTERING = "registering"
    READY = "ready"
    DRAINING = "draining"
    DELETING = "deleting"
    UNAVAILABLE = "unavailable"
    DELETED = "deleted"


class LeaseState(StrEnum):
    """Fenced ownership states for capacity leases."""

    PENDING = "pending"
    ACTIVE = "active"
    REVOKING = "revoking"
    RELEASED = "released"
    EXPIRED = "expired"
    QUARANTINED = "quarantined"


class OutboxState(StrEnum):
    """Delivery state for a transactional backend command."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DISCARDED = "discarded"


class Role(StrEnum):
    """Organization and project roles understood by the control plane."""

    FLEET_ADMIN = "fleet_admin"
    ORG_ADMIN = "org_admin"
    PROJECT_ADMIN = "project_admin"
    PROJECT_OPERATOR = "project_operator"
    PROJECT_SUBMITTER = "project_submitter"
    PROJECT_VIEWER = "project_viewer"


class Permission(StrEnum):
    """Actions enforced by the service authorization policy."""

    READ = "read"
    SUBMIT = "submit"
    OPERATE = "operate"
    MANAGE_PROJECT = "manage_project"
    #: Administer capacity an organization owns. Distinct from MANAGE_FLEET,
    #: which reaches every tenant's capacity and the provider configuration
    #: underneath it: an org admin needs to size its own pools without being
    #: handed the ability to delete another tenant's cells.
    MANAGE_OWN_CAPACITY = "manage_own_capacity"
    MANAGE_FLEET = "manage_fleet"


class TeamTier(StrEnum):
    """How much a team-mode account is trusted. Derived, never stored.

    ``admin`` is whoever holds an active ``fleet_admin`` grant; otherwise
    ``trusted`` when the account carries ``UserRecord.trusted``; otherwise
    ``user``. Deriving it keeps the grant table the one source of authority.
    """

    ADMIN = "admin"
    TRUSTED = "trusted"
    USER = "user"


class CapacityClass(StrEnum):
    """Canonical capacity purchase classes used at the API boundary."""

    RESERVED = "reserved"
    ON_DEMAND = "on_demand"
    SPOT = "spot"
    #: Queued until available, then held for a bounded window and returned
    #: whether or not the job has finished. See the domain enum for why the name
    #: describes the shape rather than any provider's product.
    FLEX_START = "flex_start"


class PriorityBand(StrEnum):
    """Strict scheduler priority bands."""

    BATCH = "batch"
    INTERACTIVE = "interactive"
    PRODUCTION = "production"


def normalize_capacity_class(value: tp.Any) -> tp.Any:
    """Normalize accepted API aliases to one durable capacity-class spelling."""

    if isinstance(value, str):
        value = value.strip().lower().replace("-", "_")
    return value


CanonicalCapacityClass = Annotated[CapacityClass, BeforeValidator(normalize_capacity_class)]


class GcpQueueAdmissionDenialTarget(EyvanModel):
    """Exact GCP TPU target rejected before it entered the capacity queue."""

    provider_project: str
    zone: str
    accelerator_type: str
    capacity_class: CanonicalCapacityClass
    #: Provider-denial occurrence, not the time a client reread this target.
    #: Missing on legacy documents; never inferred from a status heartbeat.
    occurred_at: datetime | None = None
    event_id: str | None = Field(default=None, min_length=1, max_length=192)

    @model_serializer(mode="wrap")
    def _wire_document(self, handler):
        document = handler(self)
        for name in ("occurred_at", "event_id"):
            if getattr(self, name) is None:
                document.pop(name, None)
        return document

    @model_validator(mode="after")
    def validate_identity(self) -> GcpQueueAdmissionDenialTarget:
        """Require the immutable provider identity needed for local suppression."""

        _validate_string(self.provider_project, field_name="GCP queue-denial project")
        _validate_string(self.zone, field_name="GCP queue-denial zone")
        _validate_string(self.accelerator_type, field_name="GCP queue-denial accelerator type")
        return self


class RoleGrant(EyvanModel):
    """A role scoped to a fleet, organization, or project."""

    role: Role
    org_id: str | None = Field(default=None, max_length=128)
    project_id: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_scope(self) -> RoleGrant:
        if self.role == Role.FLEET_ADMIN:
            # Fleet administration is global; drop any org/project scope so a
            # stored grant can never appear narrower than the access it gives.
            if self.org_id is not None or self.project_id is not None:
                object.__setattr__(self, "org_id", None)
                object.__setattr__(self, "project_id", None)
            return self
        if not self.org_id:
            raise ValueError(f"{self.role.value} requires an organization scope")
        if self.role == Role.ORG_ADMIN:
            # Same reason as fleet admin above, one level down. Organization
            # administration covers every project in the organization -- the
            # authorizer never consults ``project_id`` for this role -- so a
            # stored grant reading ``org_admin @ acme/research`` would suggest a
            # boundary that does not exist, in an audit log, an admin UI, or a
            # reviewer's head.
            if self.project_id is not None:
                object.__setattr__(self, "project_id", None)
            return self
        if (
            self.role
            in {
                Role.PROJECT_ADMIN,
                Role.PROJECT_OPERATOR,
                Role.PROJECT_SUBMITTER,
                Role.PROJECT_VIEWER,
            }
            and not self.project_id
        ):
            raise ValueError(f"{self.role.value} requires a project scope")
        return self


class RoleGrantRecord(EyvanModel):
    """A durable role grant this control plane issued and can revoke.

    Until this existed, ``eyvan_grants`` in an OIDC token was the *only* source
    of authority, so onboarding a tenant meant configuring an identity provider
    to mint a bespoke claim that deserializes into ``tuple[RoleGrant, ...]`` --
    a claim mentioned nowhere in the README. Anyone whose IdP could not be
    taught a custom claim could not use Eyvan's authorization at all, and
    revoking access meant an IdP change rather than an API call.

    The claim is still honored. This is a second source, not a replacement:
    a deployment that already mints the claim keeps working, and one that cannot
    now has somewhere to put grants. They are unioned, because a grant from
    either source is a grant the operator made.

    Keyed by ``(subject, role, org_id, project_id)`` so re-issuing the same
    grant overwrites rather than accumulating -- there is no meaningful
    difference between holding one copy of a role and holding two.
    """

    subject: str = Field(min_length=1, max_length=256)
    grant: RoleGrant
    #: Time-boxed access, for the case that would otherwise be permanent: a
    #: temporary elevation during an incident. ``None`` is indefinite, which is
    #: the ordinary case and should not be dressed up as an expiry nobody chose.
    #: Grant timestamps require an explicit timezone; ambiguous local times fail
    #: validation before they can confer authority or reach expiry comparisons.
    expires_at: Annotated[datetime, AwareDatetime] | None = None
    reason: str = Field(default="", max_length=512)
    granted_by: str = Field(min_length=1, max_length=256)
    granted_at: Annotated[datetime, AwareDatetime] = Field(default_factory=utcnow)

    @property
    def key(self) -> tuple[str, str, str, str]:
        """Return the identity this grant is stored under."""

        return (self.subject, self.grant.role.value, self.grant.org_id or "", self.grant.project_id or "")

    def active_at(self, moment: datetime) -> bool:
        """Return whether this grant confers authority at one timezone-aware instant."""

        if moment.utcoffset() is None:
            raise ValueError("grant observation time must include a timezone")
        return self.expires_at is None or self.expires_at > moment

    @model_validator(mode="after")
    def validate_grant_record(self) -> RoleGrantRecord:
        _validate_string(self.subject, field_name="grant subject")
        _validate_string(self.granted_by, field_name="grant issuer")
        if self.expires_at is not None and self.expires_at <= self.granted_at:
            raise ValueError("grant expiry must be after the moment it was granted")
        return self


class Principal(EyvanModel):
    """Verified caller identity and its explicit role grants."""

    subject: str = Field(min_length=1)
    display_name: str | None = None
    grants: tuple[RoleGrant, ...] = ()


#: A team-mode user name: short, lower-case, safe in URLs, logs and file names.
TEAM_SUBJECT_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"
_EMAIL_PATTERN = r"^[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s]{1,63}$"


class UserRecord(EyvanModel):
    """One team-mode account on this plane.

    ``password_hash`` is ``None`` for an account that signs in another way
    (OIDC) or has not accepted its invite yet. The raw password is never stored.
    """

    subject: str = Field(pattern=TEAM_SUBJECT_PATTERN)
    email: str = Field(max_length=320, pattern=_EMAIL_PATTERN)
    display_name: str = Field(default="", max_length=128)
    password_hash: str | None = Field(default=None, max_length=512, repr=False)
    disabled: bool = False
    created_by: str = Field(min_length=1, max_length=256)
    created_at: Annotated[datetime, AwareDatetime] = Field(default_factory=utcnow)
    updated_at: Annotated[datetime, AwareDatetime] = Field(default_factory=utcnow)
    #: Governance rules a fleet admin sets for this person. ``None`` means "no
    #: rule", which is every account's default, so a plane whose admin never
    #: sets one admits exactly what it admitted before rules existed.
    #:
    #: ``allowed_fleets``: the only fleets this person's submissions may target.
    allowed_fleets: tuple[str, ...] | None = Field(default=None, max_length=64)
    #: Submissions asking for more TPU units than this (the unit the pool
    #: declares in ``allocatable.tpu``, the same one quotas count) are held for
    #: a fleet admin's approval instead of being scheduled.
    approval_tpu_threshold: int | None = Field(default=None, ge=0, le=1_000_000)
    #: TPU units this person is guaranteed. Usage above it is *borrowed* and may
    #: be reclaimed (checkpoint first, then requeue) by someone still within
    #: their own guarantee. ``None`` means no guarantee: everything is borrowed.
    guaranteed_tpu_units: int | None = Field(default=None, ge=0, le=1_000_000)
    #: The ``trusted`` tier: may run ``@eyvan.execute`` callables in any project
    #: it can submit to and is never held for approval. Fleet admins are the
    #: ``admin`` tier through their grant, not through this flag.
    trusted: bool = False
    #: Hard ceilings across every org/project, in ResourceVector units. A plain
    #: team user needs a positive cap for each dimension their callable reserves.
    max_tpu_units: int | None = Field(default=None, ge=0, le=1_000_000)
    max_gpu_units: int | None = Field(default=None, ge=0, le=1_000_000, strict=True)
    max_cpu_units: float | None = Field(default=None, ge=0, le=1_000_000, strict=True)

    @model_serializer(mode="wrap")
    def _wire_document(self, handler):
        # An account with no rules serializes exactly as it did before rules
        # existed, so stored documents and older readers are unaffected.
        document = handler(self)
        for name in (
            "allowed_fleets",
            "approval_tpu_threshold",
            "guaranteed_tpu_units",
            "max_tpu_units",
            "max_cpu_units",
            "max_gpu_units",
        ):
            if getattr(self, name) is None:
                document.pop(name, None)
        if not self.trusted:
            document.pop("trusted", None)
        return document

    @model_validator(mode="after")
    def validate_user(self) -> UserRecord:
        if self.allowed_fleets is not None:
            _validate_strings(self.allowed_fleets, field_name="allowed fleets")
        if self.email != self.email.lower():
            raise ValueError("user email must be lower-case")
        return self


class InviteRecord(EyvanModel):
    """A single-use invitation; only the SHA-256 of its token is stored."""

    id: str = Field(pattern=r"^inv_[0-9a-f]{16}$")
    email: str = Field(max_length=320, pattern=_EMAIL_PATTERN)
    subject: str = Field(pattern=TEAM_SUBJECT_PATTERN)
    grants: tuple[RoleGrant, ...] = ()
    token_hash: str = Field(pattern=r"^[0-9a-f]{64}$", repr=False)
    created_by: str = Field(min_length=1, max_length=256)
    created_at: Annotated[datetime, AwareDatetime] = Field(default_factory=utcnow)
    expires_at: Annotated[datetime, AwareDatetime]
    accepted_at: Annotated[datetime, AwareDatetime] | None = None
    revoked: bool = False
    #: Copied onto the account on acceptance (the ``trusted`` tier).
    trusted: bool = False

    @model_serializer(mode="wrap")
    def _wire_document(self, handler):
        document = handler(self)
        if not self.trusted:
            document.pop("trusted", None)
        return document

    def open_at(self, moment: datetime) -> bool:
        """Return whether this invite can still be accepted at ``moment``."""

        return not self.revoked and self.accepted_at is None and self.expires_at > moment


class ApiTokenRecord(EyvanModel):
    """A revocable bearer credential for one user; only its SHA-256 is stored.

    ``kind`` separates a long-lived token a person created for the CLI or a
    script (``api``) from a dashboard sign-in (``session``), so listing "my
    tokens" never shows browser sessions and logging out never revokes a CLI.
    """

    id: str = Field(pattern=r"^tok_[0-9a-f]{16}$")
    subject: str = Field(pattern=TEAM_SUBJECT_PATTERN)
    name: str = Field(min_length=1, max_length=64)
    kind: tp.Literal["api", "session"] = "api"
    token_hash: str = Field(pattern=r"^[0-9a-f]{64}$", repr=False)
    prefix: str = Field(min_length=4, max_length=16)
    created_at: Annotated[datetime, AwareDatetime] = Field(default_factory=utcnow)
    expires_at: Annotated[datetime, AwareDatetime] | None = None
    last_used_at: Annotated[datetime, AwareDatetime] | None = None
    revoked_at: Annotated[datetime, AwareDatetime] | None = None

    def active_at(self, moment: datetime) -> bool:
        """Return whether this token authenticates at ``moment``."""

        return self.revoked_at is None and (self.expires_at is None or self.expires_at > moment)


class TeamRequestRecord(EyvanModel):
    """A user's request to the fleet admins: more capacity, or approval for a run.

    Filed by a person (often through the assistant) and resolved by a fleet
    admin. It carries no authority itself: approving one is a decision an admin
    records, and any capacity or quota change still goes through the ordinary
    audited routes.
    """

    id: str = Field(pattern=r"^req_[0-9a-f]{16}$")
    kind: tp.Literal["capacity", "approval"]
    subject: str = Field(min_length=1, max_length=256)
    org_id: str | None = Field(default=None, max_length=128)
    project_id: str | None = Field(default=None, max_length=128)
    summary: str = Field(min_length=1, max_length=2000)
    details: dict[str, tp.Any] = Field(default_factory=dict)
    state: tp.Literal["open", "approved", "declined", "withdrawn"] = "open"
    created_at: Annotated[datetime, AwareDatetime] = Field(default_factory=utcnow)
    resolved_at: Annotated[datetime, AwareDatetime] | None = None
    resolved_by: str | None = Field(default=None, max_length=256)
    resolution_note: str = Field(default="", max_length=2000)


#: One model name as an OpenAI-compatible endpoint spells it.
ASSISTANT_MODEL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/@+\[\]-]{0,127}$"
#: The base URL a Claude Code provider records: it is a local CLI, never an HTTP endpoint.
CLAUDE_CODE_BASE_URL = "claude-code://local"


class AssistantProviderRecord(EyvanModel):
    """An LLM endpoint a fleet admin configured for the fleet assistant.

    ``api_key_ciphertext`` is the provider key sealed with a key derived from
    the plane's own server secret; the plaintext is never stored, returned,
    logged or audited. ``None`` means the endpoint takes no key.
    """

    id: str = Field(pattern=r"^aprv_[0-9a-f]{16}$")
    name: str = Field(min_length=1, max_length=64)
    #: Built-in preset this provider was created from (``None`` for a hand-entered endpoint).
    preset: str | None = Field(default=None, max_length=64)
    base_url: str = Field(min_length=1, max_length=512)
    models: tuple[str, ...] = Field(min_length=1, max_length=32)
    default_model: str
    max_steps: int = Field(default=6, ge=1, le=20)
    enabled: bool = True
    api_key_ciphertext: str | None = Field(default=None, max_length=8192, repr=False)
    created_by: str = Field(min_length=1, max_length=256)
    created_at: Annotated[datetime, AwareDatetime] = Field(default_factory=utcnow)
    updated_at: Annotated[datetime, AwareDatetime] = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_provider(self) -> AssistantProviderRecord:
        _validate_string(self.name, field_name="assistant provider name")
        from urllib.parse import urlsplit

        if self.preset == "claude-code" or self.base_url == CLAUDE_CODE_BASE_URL:
            # Claude Code runs as the plane host's own `claude` CLI, not over HTTP.
            if self.preset != "claude-code" or self.base_url != CLAUDE_CODE_BASE_URL:
                raise ValueError(f"the claude-code preset and base_url {CLAUDE_CODE_BASE_URL!r} go together")
            if self.api_key_ciphertext is not None:
                raise ValueError("a Claude Code provider uses the host's own sign-in and stores no key")
            parsed = None
        else:
            parsed = urlsplit(self.base_url)
        if parsed is not None and (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an http(s) URL without credentials, query or fragment")
        if len(set(self.models)) != len(self.models):
            raise ValueError("assistant provider models must be unique")
        for model in self.models:
            if not re.fullmatch(ASSISTANT_MODEL_PATTERN, model):
                raise ValueError(f"invalid model name {model!r}")
        if self.default_model not in self.models:
            raise ValueError("default_model must be one of the provider's models")
        return self


#: Bounds on one saved assistant conversation, so a chat cannot grow a record without limit.
MAX_CONVERSATION_MESSAGES = 200
MAX_CONVERSATION_MESSAGE_CHARS = 32000
MAX_CONVERSATION_TRACE_ITEMS = 64


class AssistantMessageRecord(EyvanModel):
    """One visible turn of a saved assistant conversation.

    ``trace`` holds what the assistant did to answer (tool names, outcomes and
    short result summaries); ``drafts`` and ``requests`` are what it produced.
    None of it carries a provider key.
    """

    id: str = Field(pattern=r"^msg_[0-9a-f]{16}$")
    role: tp.Literal["user", "assistant"]
    content: str = Field(max_length=MAX_CONVERSATION_MESSAGE_CHARS)
    created_at: Annotated[datetime, AwareDatetime] = Field(default_factory=utcnow)
    provider_id: str | None = Field(default=None, max_length=32)
    model: str | None = Field(default=None, max_length=128)
    trace: tuple[dict[str, tp.Any], ...] = Field(default=(), max_length=MAX_CONVERSATION_TRACE_ITEMS)
    drafts: tuple[dict[str, tp.Any], ...] = Field(default=(), max_length=32)
    requests: tuple[dict[str, tp.Any], ...] = Field(default=(), max_length=32)
    #: Set when the turn ended in an error or was stopped, so history shows it honestly.
    error: str | None = Field(default=None, max_length=2000)
    #: Research plans the turn proposed (summaries; the plan record is the source of truth).
    plans: tuple[dict[str, tp.Any], ...] = Field(default=(), max_length=8)

    @model_serializer(mode="wrap")
    def _wire_document(self, handler):
        # A turn that proposed no plan serializes exactly as before the field
        # existed, so stored chats and older ``extra="forbid"`` readers agree.
        document = handler(self)
        if not self.plans and isinstance(document, dict):
            document.pop("plans", None)
        return document


class AssistantConversationRecord(EyvanModel):
    """A user's saved chat with the fleet assistant; only its owner may read or change it."""

    id: str = Field(pattern=r"^conv_[0-9a-f]{16}$")
    subject: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=200)
    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    provider_id: str | None = Field(default=None, max_length=32)
    model: str | None = Field(default=None, max_length=128)
    messages: tuple[AssistantMessageRecord, ...] = Field(default=(), max_length=MAX_CONVERSATION_MESSAGES)
    created_at: Annotated[datetime, AwareDatetime] = Field(default_factory=utcnow)
    updated_at: Annotated[datetime, AwareDatetime] = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_conversation(self) -> AssistantConversationRecord:
        _validate_string(self.title, field_name="assistant conversation title")
        if len({message.id for message in self.messages}) != len(self.messages):
            raise ValueError("assistant conversation messages must have unique IDs")
        return self


#: Bounds on one research plan, so neither a plan nor the model iterating on it
#: can grow a record without limit.
MAX_RESEARCH_JOBS = 64
MAX_RESEARCH_ROUNDS = 20
MAX_RESEARCH_NOTES = 200
MAX_RESEARCH_NOTE_CHARS = 2000
MAX_RESEARCH_GOAL_CHARS = 8000
MAX_RESEARCH_JOB_SPEC_BYTES = 64 * 1024
#: Thirty days: a research plan is a bounded experiment, not a standing service.
MAX_RESEARCH_WALL_SECONDS = 30 * 86_400
ResearchPlanState: tp.TypeAlias = Literal[
    "proposed", "approved", "running", "stopping", "finished", "stopped", "failed", "over_budget"
]
#: States the plane never moves a plan out of.
RESEARCH_PLAN_TERMINAL_STATES: frozenset[str] = frozenset({"finished", "stopped", "failed", "over_budget"})
#: States the research runner acts on each tick.
RESEARCH_PLAN_ACTIVE_STATES: frozenset[str] = frozenset({"approved", "running", "stopping"})


class ResearchJob(EyvanModel):
    """One job of a research plan: a name and the workload spec the submit route accepts.

    The spec is kept as the JSON document that was validated as a
    ``WorkloadDefinition`` when it was planned, so a later schema change never
    makes a stored plan unreadable; it is validated again at submission.
    """

    name: str = Field(min_length=1, max_length=200)
    spec: JsonDocument
    #: Which round planned it: 0 for the approved jobs, then one per iteration.
    round: int = Field(default=0, ge=0, le=MAX_RESEARCH_ROUNDS)

    @model_validator(mode="after")
    def validate_job(self) -> ResearchJob:
        _validate_string(self.name, field_name="research job name")
        if len(json.dumps(self.spec, separators=(",", ":"), default=str)) > MAX_RESEARCH_JOB_SPEC_BYTES:
            raise ValueError(f"a research job spec must be at most {MAX_RESEARCH_JOB_SPEC_BYTES} bytes")
        return self


class ResearchBudget(EyvanModel):
    """What a research plan may spend; enforced by the plane, never by the model."""

    max_jobs: int = Field(ge=1, le=MAX_RESEARCH_JOBS)
    max_chip_hours: float = Field(gt=0, le=10_000_000)
    max_wall_seconds: int = Field(gt=0, le=MAX_RESEARCH_WALL_SECONDS)
    max_rounds: int | None = Field(default=None, ge=1, le=MAX_RESEARCH_ROUNDS)


#: Points of one job's loss curve kept on its plan record.
MAX_RESEARCH_LOSS_POINTS = 50


class ResearchLoss(EyvanModel):
    """The training loss a finished research job printed, read once from its output.

    ``final`` is ``None`` when the output was read and printed no loss -- a
    different fact from a job whose output was never read, which has no
    ``ResearchLoss`` at all.
    """

    final: float | None = None
    last_step: int | None = Field(default=None, ge=0)
    total_steps: int | None = Field(default=None, ge=0)
    #: ``(step or None, loss)``, ascending by step, thinned to the bound.
    points: tuple[tuple[int | None, float], ...] = Field(default=(), max_length=MAX_RESEARCH_LOSS_POINTS)


class ResearchSubmission(EyvanModel):
    """One job the plane submitted (or tried to) for a plan, in submission order."""

    job_index: int = Field(ge=0, lt=MAX_RESEARCH_JOBS)
    round: int = Field(ge=0, le=MAX_RESEARCH_ROUNDS)
    workload_id: str | None = Field(default=None, max_length=96)
    #: Why the plane refused the job (policy, validation); never a credential.
    error: str | None = Field(default=None, max_length=500)
    submitted_at: Annotated[datetime, AwareDatetime]
    #: Recorded once the job's workload is final and its output was read.
    loss: ResearchLoss | None = None

    @model_validator(mode="after")
    def validate_submission(self) -> ResearchSubmission:
        if (self.workload_id is None) == (self.error is None):
            raise ValueError("a research submission has exactly one of workload_id or error")
        if self.loss is not None and self.workload_id is None:
            raise ValueError("only a submitted research job can record a loss")
        return self

    @model_serializer(mode="wrap")
    def _wire_document(self, handler):
        # A submission without a recorded loss serializes exactly as before the
        # field existed, so stored plans stay byte-stable.
        document = handler(self)
        if self.loss is None and isinstance(document, dict):
            document.pop("loss", None)
        return document


class ResearchSpend(EyvanModel):
    """The plane's own conservative accounting of what a plan has used."""

    chip_hours: float = Field(default=0, ge=0)
    jobs_submitted: int = Field(default=0, ge=0)
    rounds: int = Field(default=0, ge=0)


class ResearchNote(EyvanModel):
    """One bounded line of a plan's log: what the plane or the iteration model did and why."""

    t: Annotated[datetime, AwareDatetime]
    kind: str = Field(min_length=1, max_length=32)
    text: str = Field(max_length=MAX_RESEARCH_NOTE_CHARS)


class ResearchPlanRecord(EyvanModel):
    """A budgeted set of jobs a user approved once; the plane runs and iterates it as that user.

    ``approval_grants`` is the authority the approving request carried from its
    own credential (not from the plane's durable grants), narrowed to the plan's
    project and to submit/operate. The runner acts with that plus the owner's
    *current* durable grants, so revoking a durable grant or disabling the
    account takes effect on the next tick. It never acts with admin rights.
    """

    id: str = Field(pattern=r"^rp_[0-9a-f]{16}$")
    subject: str = Field(min_length=1, max_length=256)
    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=200)
    goal: str = Field(default="", max_length=MAX_RESEARCH_GOAL_CHARS)
    jobs: tuple[ResearchJob, ...] = Field(min_length=1, max_length=MAX_RESEARCH_JOBS)
    budget: ResearchBudget
    iterate: bool = False
    provider_id: str | None = Field(default=None, max_length=32)
    model: str | None = Field(default=None, max_length=128)
    state: ResearchPlanState = "proposed"
    current_round: int = Field(default=0, ge=0, le=MAX_RESEARCH_ROUNDS)
    approved_at: Annotated[datetime, AwareDatetime] | None = None
    approved_by: str | None = Field(default=None, max_length=256)
    approval_grants: tuple[RoleGrant, ...] = Field(default=(), max_length=2)
    started_at: Annotated[datetime, AwareDatetime] | None = None
    finished_at: Annotated[datetime, AwareDatetime] | None = None
    stop_reason: str | None = Field(default=None, max_length=2000)
    #: Where a ``stopping`` plan lands once its workloads are cancelled.
    stop_target: Literal["stopped", "over_budget", "failed"] | None = None
    stop_requested_by: str | None = Field(default=None, max_length=256)
    submitted: tuple[ResearchSubmission, ...] = Field(default=(), max_length=MAX_RESEARCH_JOBS)
    spent: ResearchSpend = Field(default_factory=ResearchSpend)
    #: Consecutive failed calls to the iteration model; bounded so a broken
    #: provider ends the plan instead of retrying forever.
    iteration_errors: int = Field(default=0, ge=0, le=100)
    notes: tuple[ResearchNote, ...] = Field(default=(), max_length=MAX_RESEARCH_NOTES)
    created_at: Annotated[datetime, AwareDatetime] = Field(default_factory=utcnow)
    updated_at: Annotated[datetime, AwareDatetime] = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_plan(self) -> ResearchPlanRecord:
        _validate_string(self.title, field_name="research plan title")
        _validate_string(self.subject, field_name="research plan owner")
        if len(self.jobs) > self.budget.max_jobs:
            raise ValueError("a research plan cannot plan more jobs than its max_jobs budget")
        indexes = [item.job_index for item in self.submitted]
        if len(set(indexes)) != len(indexes):
            raise ValueError("a research plan submits each job at most once")
        if any(index >= len(self.jobs) for index in indexes):
            raise ValueError("a research submission must name a planned job")
        if self.current_round > max(job.round for job in self.jobs):
            raise ValueError("a research plan's current round must have planned jobs")
        return self


class ResourceVector(EyvanModel):
    """Scalar resources requested by one workload replica or offered by a cell."""

    cpu: float = Field(default=1.0, ge=0)
    memory_gib: float = Field(default=1.0, ge=0)
    gpu: int = Field(default=0, ge=0)
    tpu: int = Field(default=0, ge=0)
    custom: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_custom(self) -> ResourceVector:
        if any(value < 0 for value in self.custom.values()):
            raise ValueError("custom resource quantities must be non-negative")
        _validate_string_map(
            self.custom,
            field_name="custom resources",
            forbidden_keys=frozenset({"cpu", "memory", "memory_gib", "gpu", "tpu"}),
        )
        return self


class ResourceLimits(ResourceVector):
    """Resource limit vector whose omitted scalar dimensions mean zero."""

    cpu: float = Field(default=0, ge=0)
    memory_gib: float = Field(default=0, ge=0)


ResourceVectorT = tp.TypeVar("ResourceVectorT", bound=ResourceVector)


def scale_resources(resources: ResourceVectorT, multiplier: int) -> ResourceVectorT:
    """Return a resource vector scaled by a positive worker count."""

    if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier < 1:
        raise ValueError("resource multiplier must be a positive integer")
    return resources.model_copy(
        update={
            "cpu": resources.cpu * multiplier,
            "memory_gib": resources.memory_gib * multiplier,
            "gpu": resources.gpu * multiplier,
            "tpu": resources.tpu * multiplier,
            "custom": {name: amount * multiplier for name, amount in resources.custom.items()},
        }
    )


class ResourceHours(ResourceLimits):
    """Fractional resource-time quantities used only for accounting."""

    # Widening int -> float is deliberate (whole-worker counts vs. fractional
    # hours); mypy flags the override because the base fields are int.
    gpu: float = Field(default=0, ge=0)  # type: ignore[assignment]
    tpu: float = Field(default=0, ge=0)  # type: ignore[assignment]


def normalize_resource_hours(value: tp.Any) -> tp.Any:
    if isinstance(value, ResourceVector) and not isinstance(value, ResourceHours):
        return value.model_dump()
    return value


CanonicalResourceHours = Annotated[ResourceHours, BeforeValidator(normalize_resource_hours)]


def normalize_resource_limits(value: tp.Any) -> tp.Any:
    """Convert legacy resource vectors without carrying their workload defaults."""

    if isinstance(value, ResourceVector) and not isinstance(value, ResourceLimits):
        return value.model_dump(exclude_unset=True)
    return value


CanonicalResourceLimits = Annotated[ResourceLimits, BeforeValidator(normalize_resource_limits)]
CanonicalExecutor = Annotated[str, BeforeValidator(canonical_execution_backend)]
CanonicalOptionalExecutor = Annotated[
    str | None,
    BeforeValidator(lambda value: None if value is None else canonical_execution_backend(value)),
]


class ResourceGang(EyvanModel):
    """Atomic group of equal resource bundles."""

    replicas: int = Field(default=1, ge=1)
    per_replica: ResourceVector = Field(default_factory=ResourceVector)
    required_topology: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_topology(self) -> ResourceGang:
        if not any(quantity > 0 for quantity in resource_quantities(self.per_replica).values()):
            raise ValueError("gang per-replica resources must contain a positive quantity")
        _validate_string_map(self.required_topology, field_name="required topology")
        return self


#: Upper bound on one chip request. One allocation lives on one TPU host, and no
#: generation the catalog describes has more chips per host than this.
MAX_TPU_CHIPS_PER_REQUEST = 8


class TpuChipRequest(EyvanModel):
    """A share of one TPU host: ``count`` chips, optionally the exact ``indexes``.

    Indexes are global within the slice -- host ``index // chips_per_host``,
    local chip ``index % chips_per_host`` -- and are kept sorted. Only accepted
    by pools that opt into chip sharing; absent, a workload takes its whole cell
    exactly as before.
    """

    count: int = Field(ge=1, le=MAX_TPU_CHIPS_PER_REQUEST, strict=True)
    #: Strict items: lax validation would quietly read ``True`` as chip 1.
    indexes: tuple[tp.Annotated[int, Field(strict=True)], ...] | None = None

    @model_validator(mode="before")
    @classmethod
    def _sorted_indexes(cls, value: tp.Any) -> tp.Any:
        if isinstance(value, dict) and isinstance(value.get("indexes"), (list, tuple)):
            indexes = value["indexes"]
            if all(isinstance(index, int) and not isinstance(index, bool) for index in indexes):
                value = {**value, "indexes": tuple(sorted(indexes))}
        return value

    @model_validator(mode="after")
    def validate_chips(self) -> TpuChipRequest:
        if self.indexes is not None:
            if any(isinstance(index, bool) or index < 0 for index in self.indexes):
                raise ValueError("TPU chip indexes must be non-negative integers")
            if len(set(self.indexes)) != len(self.indexes):
                raise ValueError("TPU chip indexes must not repeat")
            if len(self.indexes) != self.count:
                raise ValueError("TPU chip count must equal the number of chip indexes")
        return self


#: Bounds on a per-job setup step. An inline script travels inside the workload
#: record, so it is capped like any other persisted text.
MAX_SETUP_SCRIPT_BYTES = 32 * 1024
MAX_SETUP_ARGS = 64
MAX_SETUP_ARG_LENGTH = 4096
_SETUP_SHA256 = re.compile(r"[0-9a-f]{64}")


class WorkloadSetup(EyvanModel):
    """A bash step run on every host of a job, inside the job's own environment.

    Exactly one of ``script`` (inline bash) or ``url`` (an ``https://`` script
    fetched on the host) is set. ``args`` are passed to the script; ``sha256``,
    when given, refuses a fetched script whose bytes differ. The worker runs it
    after the job's ``dependencies`` are installed, with ``EYVAN_PYTHON`` naming
    the job interpreter, and caches the result by script content and args.
    """

    script: str | None = Field(default=None, max_length=MAX_SETUP_SCRIPT_BYTES)
    url: str | None = Field(default=None, max_length=2048)
    args: tuple[str, ...] = ()
    sha256: str | None = None

    @model_serializer(mode="wrap")
    def _wire_document(self, handler):
        document = handler(self)
        for name in ("script", "url", "sha256"):
            if document.get(name) is None:
                document.pop(name, None)
        if not document.get("args"):
            document.pop("args", None)
        return document

    @model_validator(mode="after")
    def validate_setup(self) -> WorkloadSetup:
        if (self.script is None) == (self.url is None):
            raise ValueError("a setup step needs exactly one of an inline script or an https:// URL")
        if self.script is not None:
            if not self.script.strip():
                raise ValueError("a setup script must not be empty")
            if "\x00" in self.script:
                raise ValueError("a setup script must not contain null bytes")
            if len(self.script.encode("utf-8")) > MAX_SETUP_SCRIPT_BYTES:
                raise ValueError(f"a setup script must be at most {MAX_SETUP_SCRIPT_BYTES} bytes")
            if self.sha256 is not None:
                raise ValueError("sha256 pins a fetched script; an inline script is already exact")
        if self.url is not None:
            # https only: the script runs as the job, so a plain-http fetch would
            # let anything on the path between the host and the server run code.
            if not self.url.startswith("https://") or any(c in self.url for c in " \t\r\n\x00"):
                raise ValueError("a setup URL must be an https:// URL without whitespace")
        if self.sha256 is not None and not _SETUP_SHA256.fullmatch(self.sha256):
            raise ValueError("setup sha256 must be 64 lowercase hex characters")
        if len(self.args) > MAX_SETUP_ARGS:
            raise ValueError(f"a setup step takes at most {MAX_SETUP_ARGS} arguments")
        for argument in self.args:
            if len(argument) > MAX_SETUP_ARG_LENGTH or "\x00" in argument:
                raise ValueError(f"setup arguments must be at most {MAX_SETUP_ARG_LENGTH} characters, no null bytes")
        return self


class CommandExecution(EyvanModel):
    """A shell command submitted to a cell execution backend."""

    kind: Literal["command"] = "command"
    command: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_command(self) -> CommandExecution:
        _validate_string(self.command, field_name="command")
        return self


class CallableExecution(EyvanModel):
    """Reference to a callable artifact in trusted object storage."""

    kind: Literal["callable"] = "callable"
    artifact_uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    python: str = Field(pattern=r"^\d+\.\d+$")
    format_version: int = Field(default=1, ge=1)
    signature: str | None = None
    result_uri: str | None = None
    workspace_uri: str | None = None
    workspace_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    #: HMAC of the workspace payload under the same key as ``signature``. A
    #: workspace lands at ``sys.path[0]`` and becomes cwd *before* the callable
    #: is unpickled, so an unsigned bundle can shadow any module the pickle
    #: imports by reference -- which would make ``signature`` decorative.
    workspace_signature: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_uris(self) -> CallableExecution:
        _validate_string(self.artifact_uri, field_name="callable artifact URI")
        _validate_string(self.result_uri, field_name="callable result URI", optional=True)
        _validate_string(self.workspace_uri, field_name="callable workspace URI", optional=True)
        if (self.workspace_uri is None) != (self.workspace_sha256 is None):
            raise ValueError("callable workspace URI and digest must be provided together")
        if self.workspace_signature is not None and self.workspace_uri is None:
            raise ValueError("callable workspace signature requires a workspace URI")
        if self.signature is not None and self.workspace_uri is not None and self.workspace_signature is None:
            # Signing the callable but not its workspace buys nothing: the
            # workspace is on sys.path first and can replace what the signed
            # pickle resolves.
            raise ValueError("a signed callable with a workspace must also sign the workspace bundle")
        return self


WorkloadExecution = Annotated[CommandExecution | CallableExecution, Field(discriminator="kind")]


class WorkspaceRef(EyvanModel):
    """A packaged working tree to materialize on the host before the workload runs.

    Deliberately a property of the *workload*, not of one execution kind. A
    workspace answers "what source is on this host", which is independent of
    whether the process is launched as a shell command or as a pickled callable.

    Modelling it on :class:`CallableExecution` -- where it first appeared,
    because callables needed it first -- made that independence untrue in
    practice: ``CommandExecution`` has only a command string, so ``eyvan run``
    could not ship a working tree at all, and the only way to get your own code
    onto an accelerator was to bake an image or use the Python decorator. That
    was a modelling accident, not a decision.

    The three fields travel together and are validated together: a URI without a
    digest cannot be verified, and a signature without a URI signs nothing.
    """

    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    #: HMAC of the bundle payload. A workspace is extracted and placed ahead of
    #: the workload's own imports, so an unsigned bundle can shadow any module
    #: the workload resolves by name.
    signature: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_workspace_ref(self) -> WorkspaceRef:
        """Reject a reference that cannot be verified on arrival."""

        _validate_string(self.uri, field_name="workspace URI")
        return self


class WorkloadDefinition(EyvanModel):
    """Execution and placement intent accepted by the public API."""

    name: str = Field(min_length=1, max_length=200)
    execution: WorkloadExecution
    #: Source to materialize on the host before the entrypoint runs, for every
    #: execution kind. See :class:`WorkspaceRef`.
    workspace: WorkspaceRef | None = None

    def resolved_workspace(self) -> WorkspaceRef | None:
        """Return the workspace to materialize, from either supported location.

        ``CallableExecution`` carried these fields before they were hoisted onto
        the workload, and specs written against that shape are still accepted --
        a submission is a durable record, so an older client must keep working.
        The spec-level field wins when both are set, because it is the one a
        newer client fills in deliberately.
        """

        if self.workspace is not None:
            return self.workspace
        execution = self.execution
        if execution.kind != "callable" or execution.workspace_uri is None:
            return None
        return WorkspaceRef(
            uri=execution.workspace_uri,
            sha256=str(execution.workspace_sha256),
            signature=execution.workspace_signature,
        )

    gang: ResourceGang = Field(default_factory=ResourceGang)
    environment: dict[str, str] = Field(default_factory=dict)
    #: Packages installed on the host before this job's command runs, layered
    #: over whatever the image provides rather than replacing it. The missing
    #: half of a per-job runtime environment: a workload could already carry its
    #: working tree and its variables, while adding one library meant rebuilding
    #: an image and rolling a fleet.
    #:
    #: Applies to every execution kind. The wrapper installs and then ``exec``s,
    #: so a callable's own worker starts under the prepared environment -- the
    #: interpreter is the host's, with the job's packages layered over it, which
    #: is what the pickled callable's imports resolve against.
    #:
    #: Requirement strings only. An entry starting with ``-`` is refused because
    #: it is an installer *option*, not a package: ``-r requirements.txt`` names
    #: a file on the host that nothing shipped, and ``--index-url`` redirects
    #: the install to somewhere the submitter chose. Both would be honoured by
    #: pip and neither is what a dependency list means.
    dependencies: tuple[str, ...] = ()
    backend_options: JsonDocument = Field(default_factory=dict)
    required_executor: CanonicalOptionalExecutor = None
    metadata: dict[str, str] = Field(default_factory=dict)
    provider: str | None = None
    region: str | None = None
    accelerator_type: str | None = None
    capacity_class: CanonicalCapacityClass = CapacityClass.ON_DEMAND
    priority_band: PriorityBand = PriorityBand.INTERACTIVE
    intra_band_priority: int = Field(default=0, ge=0, le=1000)
    required_labels: dict[str, str] = Field(default_factory=dict)
    preferred_labels: dict[str, str] = Field(default_factory=dict)
    preferred_cells: tuple[str, ...] = ()
    #: Logical names of large inputs this workload pulls -- a dataset URI, a model
    #: checkpoint, a container image. Used only to *prefer* cells that recently ran
    #: something naming the same asset, so a wrong or stale guess costs one pull
    #: and can never make the workload unplaceable. Opaque to Eyvan: two workloads
    #: naming the same string are treated as wanting the same bytes, and nothing
    #: here fetches, validates or resolves the name.
    named_assets: tuple[str, ...] = ()
    #: Additional capacity classes this workload will accept, in preference order
    #: after ``capacity_class``. Additive rather than a replacement list so every
    #: existing spec keeps its exact behaviour: a spec that names none is still a
    #: single-class request, which is what the scheduler saw before portfolios.
    additional_capacity_classes: tuple[CanonicalCapacityClass, ...] = ()
    #: Regions this workload will accept, when it does not pin one. Mutually
    #: exclusive with ``region``: a pin is a statement about *the* region, and a
    #: list beside it would leave "which wins" to whichever check ran first.
    allowed_regions: tuple[str, ...] = ()
    #: Zones this workload will accept. Same rule and same reason as regions.
    allowed_zones: tuple[str, ...] = ()
    #: Accelerator types this workload will accept, when it does not pin one.
    #:
    #: Plumbing only, and the limit is worth stating where it is declared:
    #: ``_static_cell_failure`` requires ``worker_count`` equality and
    #: ``fits_within``, so one ``ResourceGang`` matches a v5p-16 *and* a v5p-32
    #: only if the request is smaller than both -- which for a whole-slice TPU job
    #: it is not. Widening this does not make a fixed-shape job portable across
    #: generations; that needs shape-adaptive gang rewriting, which is not built.
    allowed_accelerator_types: tuple[str, ...] = ()
    estimated_duration_seconds: int = Field(default=3600, ge=1)
    checkpointable: bool = False
    checkpoint_hook: str | None = None
    checkpoint_output_uri: str | None = None
    checkpoint_grace_seconds: int = Field(default=0, ge=0)
    max_retries: int = Field(default=0, ge=0, le=100)
    max_failure_retries: int = Field(default=0, ge=0, le=100)
    #: Restarts allowed after the *provider* reclaimed the capacity, separate
    #: from ``max_retries``.
    #:
    #: A spot reclamation is the expected ending of a spot job and says nothing
    #: about whether the fleet can run it again; a stockout says this pool is
    #: empty, and a bad request says it will fail the same way forever. Charging
    #: all three to one budget means a job on spot spends its whole allowance on
    #: the one failure that was always going to happen, and then gives up the
    #: first time something is genuinely wrong.
    #:
    #: The default is deliberately larger than ``max_retries``: this is the
    #: cheap failure to retry, and a spot job that cannot outlast a handful of
    #: reclamations cannot use spot at all.
    max_preemption_retries: int = Field(default=10, ge=0, le=1000)
    #: Whether this workload receives a capability to submit children of its own.
    #: Off by default: a job that does not fan out should not be handed a
    #: credential it cannot use, and a credential nobody asked for is one nobody
    #: is watching. The gang path issues one token for the attempt, so every rank
    #: receives the same one -- deciding which rank may use it is the SDK's job,
    #: not the token's.
    can_submit_children: bool = False
    #: Secret *references*, never values. ``environment`` is persisted in the
    #: spec and readable by anyone who can read the workload, so passing a
    #: credential through it is the practice this replaces.
    #:
    #: Keyed by the environment variable name the worker will populate, so a
    #: reader can see what a job will be handed without resolving anything.
    secret_mounts: dict[str, str] = Field(default_factory=dict)
    cell_group_count: int = Field(default=1, ge=1, le=10_000)
    worker_count: int = Field(default=1, ge=1, le=10_000, strict=True)
    worker_resource_scope: WorkerResourceScope = WorkerResourceScope.ATOMIC_CELL
    #: How the workers may be distributed: one exact-shape cell (atomic) or
    #: packed from any combination of same-type cells, warm first, provisioning
    #: only the deficit (elastic). Elastic workloads run as one lease group with
    #: one hardware world per cell, and require ``per_worker`` scope.
    placement: PlacementMode = PlacementMode.ATOMIC
    #: A share of one TPU host instead of the whole cell. ``None`` -- the
    #: default and every spec written before chip sharing -- is omitted from the
    #: wire document, so existing payloads and fingerprints are unchanged.
    tpu_chips: TpuChipRequest | None = None
    #: A bash step run on every host before the job, inside its environment.
    #: Omitted from the wire document while unset, like ``tpu_chips``.
    setup: WorkloadSetup | None = None

    @model_serializer(mode="wrap")
    def _wire_document(self, handler):
        document = handler(self)
        if self.tpu_chips is None:
            document.pop("tpu_chips", None)
        if self.setup is None:
            document.pop("setup", None)
        return document

    @model_validator(mode="after")
    def validate_tpu_chips(self) -> WorkloadDefinition:
        if self.tpu_chips is None:
            return self
        if self.worker_count != 1 or self.cell_group_count != 1 or self.gang.replicas != 1:
            raise ValueError("a TPU chip request runs on one host of one cell; pods and replicas must be 1")
        if self.placement is not PlacementMode.ATOMIC:
            raise ValueError("a TPU chip request requires atomic placement")
        if self.gang.per_replica.tpu != self.tpu_chips.count:
            raise ValueError("a TPU chip request must reserve exactly its chip count of TPU")
        return self

    @model_validator(mode="after")
    def validate_placement(self) -> WorkloadDefinition:
        if self.placement is PlacementMode.ELASTIC and self.worker_resource_scope is not WorkerResourceScope.PER_WORKER:
            raise ValueError("elastic placement requires worker_resource_scope=per_worker")
        if self.placement is PlacementMode.ELASTIC and self.cell_group_count != 1:
            raise ValueError("elastic placement and explicit cell groups are mutually exclusive")
        return self

    @model_validator(mode="after")
    def validate_checkpoint(self) -> WorkloadDefinition:
        _validate_string(self.name, field_name="workload name")
        _validate_string(self.provider, field_name="provider", optional=True)
        _validate_string(self.region, field_name="region", optional=True)
        _validate_string(self.accelerator_type, field_name="accelerator type", optional=True)
        _validate_string(self.required_executor, field_name="required executor", optional=True)
        if not self.checkpointable and any(
            value is not None for value in (self.checkpoint_hook, self.checkpoint_output_uri)
        ):
            raise ValueError("checkpoint hook and output URI require checkpointable=true")
        if self.checkpointable and not (self.checkpoint_hook and self.checkpoint_output_uri):
            raise ValueError("checkpointable workloads require a checkpoint hook and durable output URI")
        for requirement in self.dependencies:
            _validate_string(requirement, field_name="dependency")
            if requirement.startswith("-"):
                # An installer option, not a package. `-r requirements.txt`
                # names a file on the host that nothing shipped, and
                # `--index-url` points the install somewhere the submitter
                # chose; pip honours both, and neither is a dependency.
                raise ValueError(
                    f"dependency {requirement!r} is an installer option, not a package; list the packages themselves"
                )
            if any(character in requirement for character in "\x00\n\r"):
                raise ValueError("dependencies must not contain newlines or null bytes")
        _validate_string_map(self.required_labels, field_name="required labels")
        _validate_string_map(self.preferred_labels, field_name="preferred labels")
        _validate_string_map(self.metadata, field_name="metadata")
        _validate_string_map(self.environment, field_name="environment")
        for name, value in self.environment.items():
            if not _ENVIRONMENT_NAME.fullmatch(name):
                raise ValueError("workload environment names must be valid shell identifiers")
            if _environment_name_is_reserved(name):
                raise ValueError(f"workload environment variable {name!r} is reserved by Eyvan")
            if "\x00" in value or "\n" in value or "\r" in value:
                raise ValueError(f"workload environment variable {name!r} must not contain NUL bytes or newlines")
        validate_secret_mount_contract(self.environment, self.secret_mounts)
        _validate_strings(self.preferred_cells, field_name="preferred cells")
        if len(self.preferred_cells) != len(set(self.preferred_cells)):
            raise ValueError("preferred cells must not contain duplicates")
        _validate_strings(self.named_assets, field_name="named assets")
        if len(self.named_assets) != len(set(self.named_assets)):
            raise ValueError("named assets must not contain duplicates")
        portfolio = self.capacity_class_portfolio
        if len(set(portfolio)) != len(portfolio):
            raise ValueError("additional capacity classes must not repeat the primary class or each other")
        for pin, choices, label in (
            (self.region, self.allowed_regions, "region"),
            (self.accelerator_type, self.allowed_accelerator_types, "accelerator type"),
        ):
            if pin and choices:
                raise ValueError(f"a workload pins one {label} or lists several, never both")
        # No zone-pin conflict to check: a spec has no ``zone`` field. Zones are
        # expressed only as a list, because a single zone is already sayable as a
        # one-element list and a second spelling would be a second thing to keep
        # consistent.
        for name, values in (
            ("allowed regions", self.allowed_regions),
            ("allowed zones", self.allowed_zones),
            ("allowed accelerator types", self.allowed_accelerator_types),
        ):
            _validate_strings(values, field_name=name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        declared_worker_count = self.required_labels.get(WORKER_COUNT_LABEL)
        if declared_worker_count is not None and declared_worker_count != str(self.worker_count):
            raise ValueError(f"required label {WORKER_COUNT_LABEL!r} must match worker_count")
        if self.worker_count > 1:
            if self.cell_group_count != 1:
                raise ValueError("multi-worker capacity must use exactly one physical cell")
            if self.provider not in providers_hosting_multi_worker_cells():
                # Asked of the capability table, not of a provider name: a cell
                # spanning several machines is a thing a backend can or cannot
                # build, and a second cloud that can build one must not be
                # refused by a validator in the durable model.
                supported = ", ".join(sorted(providers_hosting_multi_worker_cells())) or "no known provider"
                raise ValueError(f"multi-worker capacity requires a provider that builds it: {supported}")
            if self.required_executor != "eyvan-native":
                raise ValueError("multi-worker capacity requires the eyvan-native executor")
            if self.gang.per_replica.tpu <= 0 or self.gang.per_replica.gpu > 0:
                raise ValueError("multi-worker capacity requires a TPU resource request")
            if self.worker_count > max_multi_worker_cell_width():
                raise ValueError(
                    f"multi-worker cells support at most {max_multi_worker_cell_width()} workers "
                    "across known capacity backends"
                )
        return self

    @property
    def capacity_class_portfolio(self) -> tuple[CanonicalCapacityClass, ...]:
        """Return every acceptable capacity class, most preferred first.

        ``capacity_class`` stays primary and stays the default, so a spec that
        declares no additional classes is the single-class request the scheduler
        already understood. The core scheduler has always ranked a portfolio --
        ``capacity_class_rank`` is the fourth key of ``PlacementScore`` -- and the
        only thing collapsing it to one entry was the adapter.
        """

        return (self.capacity_class, *self.additional_capacity_classes)

    @property
    def resources_per_cell(self) -> ResourceVector:
        """Return the aggregate resource reservation for one physical cell."""

        resources = scale_resources(self.gang.per_replica, self.gang.replicas)
        if self.worker_resource_scope is WorkerResourceScope.PER_WORKER:
            return scale_resources(resources, self.worker_count)
        return resources


class CellRejection(EyvanModel):
    """One cell-specific reason a workload could not be placed there."""

    cell_id: str = Field(max_length=128)
    code: InfeasibilityCode
    message: str = Field(max_length=512)


class CandidateZone(EyvanModel):
    """One pool this workload accepts, and what is known about it.

    The belief state -- cooldowns and consecutive provider failures -- has
    decided which pool a cross-zone workload asks next since alternates
    existed, and said so nowhere. "Why is it waiting on us-east5-b when
    us-central2-b is idle" had no answer short of reading cell records by hand,
    and on a spot-first fleet it is the question asked most often.

    Attributes:
        pool_id: The pool, which for a cross-zone shape is effectively the zone.
        zone: Where it is, when the pool records one.
        preferred: Whether this is the pool the planner would ask next.
        free_owned_cells: Cells this workload already owns here, which it could
            take immediately. The strongest reason to prefer a pool.
        free_shared_cells: Unreserved cells of the right width here.
        cooling_cells: Cells in a provider-failure cooldown. A pool is as cold
            as its coldest cell, which is why one is enough to rank it lower.
        consecutive_failures: The worst consecutive-failure count here.
    """

    pool_id: str = Field(max_length=128)
    zone: str | None = Field(default=None, max_length=128)
    preferred: bool = False
    free_owned_cells: int = Field(default=0, ge=0)
    free_shared_cells: int = Field(default=0, ge=0)
    cooling_cells: int = Field(default=0, ge=0)
    consecutive_failures: int = Field(default=0, ge=0)


class WorkloadExplanation(EyvanModel):
    """Why one workload is or is not currently placeable.

    Computed on demand by re-running the pure scheduler against current state,
    so it always reflects the fleet as it is now rather than a verdict cached
    when the workload was last considered. The scheduler already produces the
    per-cell detail; persisting it would add a second field to keep in sync at
    every ``scheduling_reason`` write site, and would go stale the moment a
    cell changed.

    ``cells`` is bounded for display; ``considered_cell_count`` is the true
    number of rejected cells so a truncated list is never mistaken for the
    whole fleet.
    """

    workload_id: str = Field(max_length=96)
    state: WorkloadState
    placeable: bool
    placement_cell_id: str | None = None
    code: InfeasibilityCode | None = None
    message: str | None = Field(default=None, max_length=512)
    cells: tuple[CellRejection, ...] = ()
    considered_cell_count: int = Field(default=0, ge=0)
    #: Cells the autoscaler is currently building for this workload. The
    #: scheduler only ever sees cells that already exist, so without this a
    #: workload waiting on capacity is indistinguishable from one that can
    #: never be placed -- both report "no cell satisfies all hard workload
    #: constraints", and only one of them is a problem.
    provisioning_cell_ids: tuple[str, ...] = ()
    scheduling_reason: str | None = None
    autoscale_block_reason: AutoscaleBlockReason | None = None
    queue_admission_denial: GcpQueueAdmissionDenialTarget | None = None
    #: How the durable command that carries this workload to its cell is faring.
    #: A workload past PENDING is no longer the scheduler's business, and the
    #: honest answer to "why is it not running" moves to delivery -- which was
    #: previously visible nowhere. A dispatch failed ten times over twenty-one
    #: minutes while this endpoint said only "workload is leased".
    delivery_state: str | None = None
    delivery_attempts: int = Field(default=0, ge=0)
    delivery_error: str | None = Field(default=None, max_length=2000)
    #: Every pool this workload accepts, best choice first, with what the belief
    #: state knows about each. Empty for a workload that names one pool, which
    #: is most of them and for which there is nothing to choose between.
    candidate_zones: tuple[CandidateZone, ...] = ()
    observed_at: datetime = Field(default_factory=utcnow)

    @property
    def truncated_cell_count(self) -> int:
        """Return how many rejected cells were omitted from ``cells``."""

        return max(0, self.considered_cell_count - len(self.cells))


class WorkloadDispatchReservation(EyvanModel):
    """Durable fence around one possibly-started cell-agent submission.

    The reservation is committed before the controller calls the agent and is
    compared again when the response is written back.  ``fence`` also binds
    secret capabilities to this exact delivery: retrying the same outbox command
    replaces it, so capabilities from an abandoned RPC never become live again.
    """

    fence: str = Field(min_length=1, max_length=96)
    outbox_id: str = Field(min_length=1, max_length=96)
    outbox_attempt: int = Field(ge=1)
    generation: int = Field(ge=0)
    workload_attempt: int = Field(ge=1)
    lease_ids: tuple[str, ...] = Field(min_length=1)
    reserved_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_membership(self) -> WorkloadDispatchReservation:
        _validate_strings(self.lease_ids, field_name="dispatch reservation lease ids")
        if len(self.lease_ids) != len(set(self.lease_ids)):
            raise ValueError("dispatch reservation lease ids must not contain duplicates")
        return self


#: Ceiling on how many global ranks one workload attempt may declare.
#:
#: ``WorkloadDefinition.cell_group_count`` and ``CellRecord.runtime_process_count``
#: are each bounded at 10,000, and nothing bounded their *product*: a 10,000-cell
#: group of 10,000-process cells is a hundred million ranks, and every one of them
#: is a cursor position, an archive object, and a progress row. The composed total
#: is what the log, cursor, and archive contracts actually pay for, so the total is
#: what is bounded -- once, before dispatch, rather than per factor.
#:
#: Also deliberately below the opaque log cursor's own entry ceiling. One flat
#: cursor carries one position per expected global rank, so an attempt wider than
#: that ceiling could be dispatched and then never resumed.
MAX_ATTEMPT_RANK_COUNT = 4096


class WorkloadTopologyCell(EyvanModel):
    """One cell's frozen contribution to an attempt's global rank space."""

    cell_id: str = Field(max_length=96)
    #: The cell generation this attempt was dispatched against. A later
    #: generation is a different machine, so a page read against it is not the
    #: same rank space even when the identifier matches.
    generation: int = Field(ge=1)
    #: Runtime/native-worker processes this cell contributed, copied once at
    #: dispatch. Reading ``CellRecord.runtime_process_count`` later instead is
    #: the defect this record exists to remove: an observation may overwrite it,
    #: which silently shifts every following cell's global ranks.
    #:
    #: Bounded exactly like the cell field it copies, deliberately *not* at the
    #: attempt ceiling: a per-factor limit here would report an over-wide attempt
    #: as a malformed cell rather than as the composed width that is actually too
    #: large, which is the diagnosis an operator needs.
    runtime_process_count: int = Field(ge=1, le=10_000, strict=True)
    #: First global rank owned by this cell. ``global = offset + local_rank``.
    global_rank_offset: int = Field(ge=0)
    #: Ordered per-host identities the control plane could observe at dispatch.
    #: Empty when the provider does not expose them -- an unknown order is
    #: recorded as unknown rather than invented, and a later comparison then
    #: falls back to the counts and generations that *are* durable.
    worker_endpoints: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_cell(self) -> WorkloadTopologyCell:
        _validate_string(self.cell_id, field_name="topology cell id")
        _validate_strings(self.worker_endpoints, field_name="topology worker endpoints")
        if self.worker_endpoints and len(self.worker_endpoints) != self.runtime_process_count:
            raise ValueError("topology worker endpoints must match the cell runtime process count")
        if len(set(self.worker_endpoints)) != len(self.worker_endpoints):
            raise ValueError("topology worker endpoints must not contain duplicates")
        return self


class WorkloadAttemptTopology(EyvanModel):
    """The rank space one workload attempt was dispatched against, frozen once.

    Durable and additive: a control plane that predates this record simply never
    reads the table, which is what makes it safe to write during a rolling
    upgrade. The alternative -- a new field on ``WorkloadRecord`` -- is not, because
    the persisted workload document is decoded with ``extra='forbid'`` and the
    first write would make every workload unreadable to a rolled-back replica.

    An attempt with no row is *unknown*, never reconstructed. Recomputing the map
    from today's cells is precisely the bug: a reconciliation that changes one
    cell's process width moves every later cell's ranks, so two pages of one
    attempt disagree about which host is rank N.
    """

    workload_id: str = Field(max_length=96)
    attempt: int = Field(ge=1)
    #: Workload generation at dispatch. Together with ``attempt`` this is the
    #: exact execution the ranks describe.
    generation: int = Field(ge=1)
    cells: tuple[WorkloadTopologyCell, ...] = Field(min_length=1)
    total_rank_count: int = Field(ge=1, le=MAX_ATTEMPT_RANK_COUNT)
    #: ``sha256:<hex>`` over the ordered cell identities and the total width.
    #: Carried on every page so a follower can detect that it is reading a
    #: different rank space than the one its cursor was issued against.
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    frozen_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_topology(self) -> WorkloadAttemptTopology:
        _validate_string(self.workload_id, field_name="topology workload id")
        identifiers = [cell.cell_id for cell in self.cells]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("attempt topology cells must not repeat a cell id")
        if identifiers != sorted(identifiers):
            raise ValueError("attempt topology cells must be ordered by cell id")
        offset = 0
        for cell in self.cells:
            if cell.global_rank_offset != offset:
                raise ValueError("attempt topology rank offsets must be contiguous from zero")
            offset += cell.runtime_process_count
        if offset != self.total_rank_count:
            raise ValueError("attempt topology total rank count must equal the sum of cell widths")
        if self.digest != topology_digest(self.cells, total_rank_count=self.total_rank_count):
            raise ValueError("attempt topology digest does not describe its cells")
        return self

    @classmethod
    def freeze(
        cls,
        *,
        workload_id: str,
        attempt: int,
        generation: int,
        cells: tp.Sequence[WorkloadTopologyCell],
        frozen_at: datetime | None = None,
    ) -> WorkloadAttemptTopology:
        """Build one attempt topology, computing offsets, width, and digest."""

        ordered = sorted(cells, key=lambda cell: cell.cell_id)
        placed: list[WorkloadTopologyCell] = []
        offset = 0
        for cell in ordered:
            placed.append(cell.model_copy(update={"global_rank_offset": offset}))
            offset += cell.runtime_process_count
        if offset > MAX_ATTEMPT_RANK_COUNT:
            raise ValueError(
                f"workload attempt declares {offset} global ranks, above the {MAX_ATTEMPT_RANK_COUNT} rank ceiling"
            )
        frozen = tuple(placed)
        return cls(
            workload_id=workload_id,
            attempt=attempt,
            generation=generation,
            cells=frozen,
            total_rank_count=offset,
            digest=topology_digest(frozen, total_rank_count=offset),
            frozen_at=frozen_at or utcnow(),
        )

    def cell_for_rank(self, rank: int) -> WorkloadTopologyCell | None:
        """Return the frozen cell owning one global rank, or ``None``."""

        for cell in self.cells:
            if cell.global_rank_offset <= rank < cell.global_rank_offset + cell.runtime_process_count:
                return cell
        return None

    def contains_rank(self, rank: int) -> bool:
        """Return whether one global rank exists in this frozen rank space."""

        return isinstance(rank, int) and not isinstance(rank, bool) and 0 <= rank < self.total_rank_count


def topology_digest(cells: tp.Sequence[WorkloadTopologyCell], *, total_rank_count: int) -> str:
    """Return the stable digest of one ordered attempt topology.

    Covers cell identity, generation, width, offset, and any observed worker
    order -- everything a global rank means. Two attempts with the same digest
    name the same physical process for the same rank; two with different digests
    are not comparable and no cursor crosses between them.
    """

    payload = {
        "total_rank_count": total_rank_count,
        "cells": [
            {
                "cell_id": cell.cell_id,
                "generation": cell.generation,
                "runtime_process_count": cell.runtime_process_count,
                "global_rank_offset": cell.global_rank_offset,
                "worker_endpoints": list(cell.worker_endpoints),
            }
            for cell in cells
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class WorkloadRecord(EyvanModel):
    """Persisted workload aggregate."""

    id: str = Field(max_length=96)
    org_id: str = Field(max_length=128)
    project_id: str = Field(max_length=128)
    spec: WorkloadDefinition
    state: WorkloadState = WorkloadState.PENDING
    attempt: int = 1
    failure_count: int = Field(default=0, ge=0)
    capacity_loss_count: int = Field(default=0, ge=0)
    #: Restarts already spent on provider reclamation, charged against
    #: ``spec.max_preemption_retries`` rather than ``spec.max_retries``.
    preemption_count: int = Field(default=0, ge=0)
    generation: int = 1
    lease_id: str | None = None
    lease_ids: tuple[str, ...] = ()
    #: A short-lived, durable boundary around a possibly-started dispatch RPC.
    #: It is intentionally not a secret: possession does not authenticate a
    #: caller, while signed secret capabilities must carry the exact same fence.
    dispatch_reservation: WorkloadDispatchReservation | None = None
    remote_id: str | None = None
    #: Where this replica says it can be reached, once it says so.
    #:
    #: On the workload rather than in a registry table, which is what makes the
    #: liveness rule free: an endpoint is served only while the record carrying
    #: it is RUNNING at the current generation, and a replica that dies takes
    #: its endpoint with it. A separate table would need a TTL or a heartbeat to
    #: reach the same conclusion, and both are a second source of truth about
    #: something the control plane already knows exactly.
    #:
    #: Set by the workload itself after it can serve. That is real readiness --
    #: an authenticated statement by the process that would answer -- rather
    #: than an inference from a state machine that only knows the process
    #: started.
    service_endpoint: str | None = Field(default=None, max_length=512)
    checkpoint_uri: str | None = None
    failure_reason: str | None = None
    scheduling_reason: str | None = None
    autoscale_block_reason: AutoscaleBlockReason | None = None
    queue_admission_denial: GcpQueueAdmissionDenialTarget | None = None
    preemption_id: str | None = Field(default=None, max_length=96)
    #: Parent in a job tree. A child is an ordinary workload that competes as a
    #: normal sibling under the same (org, project, principal) fair-share leaf --
    #: job structure is a relation and a cancellation cascade, not a fourth
    #: scheduling level. See eyvan.server.service.subjobs for why.
    parent_workload_id: str | None = Field(default=None, max_length=96)
    #: False for a record that must never be placed. The scheduler considers every
    #: PENDING workload, so without this a batch of N tasks placed N+1 jobs and the
    #: extra one ran the template command with no task index -- on a full accelerator
    #: slice. Deliberately *not* the test for "is this a coordinator": several planned
    #: shapes are unschedulable without being coordinators (a DAG member waiting on a
    #: predecessor, a workload delegated to a peer control plane), and identifying a
    #: coordinator by the absence of a flag would silently capture all of them.
    schedulable: bool = True
    #: Set only on a record that owns children and derives its state from them.
    #: ``None`` means "not a coordinator" -- a positive, self-describing marker, so a
    #: new unschedulable shape cannot accidentally be settled from a sibling's state.
    coordination_kind: CoordinationKind | None = None
    #: Sibling workloads that must succeed before this one may be placed. Non-empty
    #: implies ``schedulable=False`` until the controller releases it. Dependencies
    #: are between whole workloads and carry no data: results already travel as
    #: ``eyvan://`` result URIs, and adding a value channel here would make the
    #: control plane a data plane.
    depends_on: tuple[str, ...] = ()
    created_by: str = Field(max_length=256)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_identity(self) -> WorkloadRecord:
        _validate_string(self.id, field_name="workload id")
        _validate_string(self.org_id, field_name="organization id")
        _validate_string(self.project_id, field_name="project id")
        _validate_string(self.created_by, field_name="creator id")
        _validate_strings(self.lease_ids, field_name="workload lease ids")
        if len(self.lease_ids) != len(set(self.lease_ids)):
            raise ValueError("workload lease_ids must not contain duplicates")
        if self.lease_ids and self.lease_id != self.lease_ids[0]:
            raise ValueError("workload lease_id must be the first workload lease_ids entry")
        if self.lease_id is None and self.lease_ids:
            raise ValueError("workload lease_ids require lease_id")
        if (
            self.queue_admission_denial is not None
            and self.autoscale_block_reason is not AutoscaleBlockReason.QUEUE_ADMISSION_DENIED
        ):
            raise ValueError("queue_admission_denial requires autoscale_block_reason queue_admission_denied")
        return self


class AvailabilityPolicy(EyvanModel):
    """How hard a pool backs off, and how fast it forgets.

    These were constants, and the wrong ones for somebody. How long to wait
    after a zone says no is a property of *that* zone and that purchase class:
    a spot TPU pool in a contended region is worth asking again in a minute,
    while a pool whose provider is rate-limiting the account is not. A single
    curve compiled into the planner meant an operator watching the wrong
    behaviour had no way to say so, and the only lever left was to delete the
    pool.

    Defaults reproduce the constants exactly, so a fleet that declares nothing
    behaves as it always did.
    """

    #: What one observation retains of the previous evidence. Below 1 the model
    #: forgets, which is the point: a zone that was empty this morning is not
    #: evidence about this afternoon. Nearer 1 remembers longer.
    evidence_decay: float = Field(default=0.98, gt=0, le=1)
    #: The ceiling on the doubling. Without one a pool that failed a dozen times
    #: overnight is still in cooldown when somebody arrives to use it.
    max_backoff_s: int = Field(default=3600, ge=1, le=86_400)
    #: First cooldown per outcome, doubling per consecutive failure. Keyed by
    #: `ProvisionOutcome` value; anything unnamed uses `default_backoff_s`.
    #: A stockout is answered by asking again later, while a host fault is
    #: answered by provisioning a replacement -- a full allocation and a destroy
    #: before anything is learned -- which is why they are not one number.
    backoff_base_s: dict[str, int] = Field(default_factory=lambda: {"stockout": 60, "host_fault": 300})
    default_backoff_s: int = Field(default=15, ge=1, le=86_400)

    @model_validator(mode="after")
    def validate_policy(self) -> AvailabilityPolicy:
        for outcome, seconds in self.backoff_base_s.items():
            if not outcome or outcome != outcome.strip():
                raise ValueError("availability policy outcome names must be non-empty and trimmed")
            if seconds < 1 or seconds > self.max_backoff_s:
                # A base above the ceiling is not a longer cooldown, it is a
                # cooldown the ceiling silently discards -- so the number an
                # operator wrote would mean nothing.
                raise ValueError(
                    f"backoff base for {outcome!r} must be between 1 and max_backoff_s ({self.max_backoff_s})"
                )
        if self.default_backoff_s > self.max_backoff_s:
            raise ValueError("default backoff must not exceed max_backoff_s")
        return self


class CapacityBeliefState(EyvanModel):
    """Persisted empirical availability evidence for one execution cell."""

    success_evidence: float = Field(default=1.0, gt=0)
    failure_evidence: float = Field(default=1.0, gt=0)
    provision_latencies_s: tuple[float, ...] = ()
    interruptions: float = Field(default=0, ge=0)
    successful_hours: float = Field(default=0, ge=0)
    consecutive_failures: int = Field(default=0, ge=0)
    cooldown_until: datetime | None = None
    last_observed_at: datetime | None = None


class CellRecord(EyvanModel):
    """Persisted execution-cell inventory record."""

    id: str = Field(max_length=96)
    provider: str
    capacity_backend: str | None = None
    provider_project: str | None = None
    region: str
    zone: str | None = None
    state: CellState = CellState.REGISTERING
    capacity: CanonicalResourceLimits
    allocatable: CanonicalResourceLimits
    unit_prices: dict[str, float] = Field(default_factory=dict)
    accelerator_type: str | None = None
    capacity_class: CanonicalCapacityClass = CapacityClass.ON_DEMAND
    executor: CanonicalExecutor = "eyvan-native"
    endpoint: str | None = None
    pool_id: str | None = None
    autoscale_request_workload_id: str | None = Field(default=None, max_length=96)
    provider_options: JsonDocument = Field(default_factory=dict)
    runtime: NativeRuntimeConfig = Field(default_factory=NativeRuntimeConfig)
    labels: dict[str, str] = Field(default_factory=dict)
    topology: dict[str, str] = Field(default_factory=dict)
    max_replica_resources: CanonicalResourceLimits | None = None
    max_gang_replicas: int | None = Field(default=None, ge=1)
    worker_count: int = Field(default=1, ge=1, le=10_000, strict=True)
    #: Copied from the pool: whether several chip-sized workloads may share this
    #: cell's hosts. Omitted from the wire document while false.
    chip_sharing: bool = False
    provider_phase: CapacityStatus | None = None
    provider_failure_code: CapacityProviderFailureCode | None = None
    #: First observation of this failure in this provider attempt. Heartbeats
    #: preserve it; legacy failures without attribution remain unknown.
    provider_failure_occurred_at: datetime | None = None
    provider_failure_event_id: str | None = Field(default=None, min_length=1, max_length=192)
    provider_failure_generation: int | None = Field(default=None, ge=1)
    observed_tpu_vms: int | None = Field(default=None, ge=1)
    #: Durable width of this cell in runtime/native-worker/JAX processes.
    #:
    #: ``worker_count`` counts allocation units (TPU slices). A v5p-16 cell with
    #: one allocated slice has ``worker_count == 1`` but two runtime processes.
    #: Log rank offsets must survive a lost endpoint, so they use this persisted
    #: provider-observed width rather than recounting whichever endpoints answer
    #: a later request. One is the backward-compatible single-host default.
    runtime_process_count: int = Field(default=1, ge=1, le=10_000, strict=True)
    #: Named assets this cell's agent reports are materialized on local disk.
    #:
    #: A *stronger* locality signal than lease history, which infers presence
    #: from "a job that declared this asset held this cell recently". The
    #: inference is free and often right; the report is authoritative for the
    #: one thing the control plane cannot see, which is the disk.
    #:
    #: Still only a preference. An agent that over-reports steers placement
    #: toward a cell that then pays the pull anyway -- one cache miss, which is
    #: the same cost a stale inference already carries, and strictly less
    #: authority than the capacity and readiness an agent already reports.
    materialized_assets: tuple[str, ...] = ()
    provider_phase_started_at: datetime | None = None
    unavailable_reason: str | None = None
    #: Free-text cause behind ``unavailable_reason``, which is a closed code that
    #: other paths compare. A fence conflict has three distinct causes and a
    #: provider error has many; collapsing them all to one code means an operator
    #: has to reconstruct the cause from an agent's storage document by hand.
    unavailable_detail: str | None = Field(default=None, max_length=2048)
    availability: CapacityBeliefState = Field(default_factory=CapacityBeliefState)
    #: Copied from the pool at creation, like the runtime and the labels. Held on
    #: the cell so an observation needs nothing but the cell it is about -- and
    #: so a pool whose curve is retuned does not retroactively rewrite how the
    #: machines already running were being judged.
    availability_policy: AvailabilityPolicy = Field(default_factory=AvailabilityPolicy)
    #: How long the provider holds *this* machine before taking it back. Copied
    #: from the pool for the same reason: when a cell goes away is a fact about
    #: the cell, and asking its pool later would answer for a template that may
    #: since have been retuned.
    max_hold_s: int | None = Field(default=None, ge=60)
    generation: int = 1
    idle_since: datetime | None = None
    last_heartbeat_at: datetime | None = None
    #: When the provider started charging for this cell. Set once the provider
    #: resource exists, cleared when it is released. Distinct from ``created_at``:
    #: a queued TPU request is a record Eyvan holds and a bill nobody is paying.
    billing_started_at: datetime | None = None
    #: When the billing clock was last advanced. The accrual loop is at-least-once
    #: and may run late or twice; charging ``now - last_tick`` rather than one
    #: fixed interval makes a missed or repeated tick cost the right amount.
    billing_ticked_at: datetime | None = None
    #: Wall-clock seconds this cell has existed at the provider's expense.
    billed_seconds: float = Field(default=0.0, ge=0)
    #: The subset of ``billed_seconds`` during which no lease held the cell. This
    #: is capacity the fleet paid for and nobody used -- the number an operator
    #: needs to size a pool, and one that no lease-derived ledger can produce,
    #: because an idle cell has no lease to settle.
    idle_seconds: float = Field(default=0.0, ge=0)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @model_serializer(mode="wrap")
    def _wire_document(self, handler):
        document = handler(self)
        for name in ("provider_failure_occurred_at", "provider_failure_event_id", "provider_failure_generation"):
            if getattr(self, name) is None:
                document.pop(name, None)
        if not self.chip_sharing:
            document.pop("chip_sharing", None)
        return document

    def hourly_cost(self) -> float:
        """Return what this cell costs per hour at its declared unit prices.

        Zero when the pool carries no prices -- which is exactly defect D7's
        shape, and why an unpriced pool is a ``doctor`` finding rather than
        something this method can compensate for. A silent zero here would make
        an idle-cost report say the fleet is free.
        """

        capacity = self.capacity
        quantities = {
            "cpu": capacity.cpu,
            "memory_gib": capacity.memory_gib,
            "gpu": float(capacity.gpu),
            "tpu": float(capacity.tpu),
            **capacity.custom,
        }
        return costing.total(*(price * quantities.get(resource, 0.0) for resource, price in self.unit_prices.items()))

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_runtime_widths(cls, value: tp.Any) -> tp.Any:
        """Restore fields omitted by older durable cell documents.

        Healthy TPU records already carried the provider-observed VM count, and
        manual records carried their declared endpoint list. Reusing those
        durable observations preserves an exact rank width across an upgrade;
        records without either remain safely readable as single-host cells.
        """

        if not is_json_object(value):
            return value
        upgraded = dict(value)
        provider_options = value.get("provider_options")
        if "worker_count" not in value and is_json_object(provider_options) and "node_count" in provider_options:
            node_count = provider_options["node_count"]
            if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 1:
                raise ValueError("GCP TPU provider_options.node_count must be a positive integer")
            upgraded["worker_count"] = node_count
        if "runtime_process_count" not in value:
            observed = value.get("observed_tpu_vms")
            if isinstance(observed, int) and not isinstance(observed, bool) and observed > 0:
                upgraded["runtime_process_count"] = observed
            elif value.get("provider") == "manual" and is_json_object(provider_options):
                hosts = provider_options.get("hosts")
                if isinstance(hosts, list) and hosts:
                    upgraded["runtime_process_count"] = len(hosts)
        return upgraded

    @model_validator(mode="after")
    def validate_provider_identity(self) -> CellRecord:
        _validate_string(self.id, field_name="cell id")
        _validate_string(self.provider, field_name="cell provider")
        _validate_string(self.capacity_backend, field_name="capacity backend", optional=True)
        _validate_string(self.provider_project, field_name="provider project", optional=True)
        _validate_string(self.region, field_name="cell region")
        _validate_string(self.zone, field_name="cell zone", optional=True)
        _validate_string(self.accelerator_type, field_name="cell accelerator type", optional=True)
        _validate_string(self.executor, field_name="cell executor")
        _validate_string(self.endpoint, field_name="cell endpoint", optional=True)
        _validate_string(self.pool_id, field_name="capacity pool id", optional=True)
        _validate_string(
            self.autoscale_request_workload_id,
            field_name="autoscale request workload id",
            optional=True,
        )
        if self.autoscale_request_workload_id is not None and self.pool_id is None:
            raise ValueError("autoscale request workload id requires a capacity pool")
        if self.executor in {"native-local", "eyvan-native"} and self.max_gang_replicas not in {None, 1}:
            raise ValueError(f"{self.executor} cells support exactly one gang replica")
        if not self.runtime.is_empty and self.executor != "eyvan-native":
            raise ValueError("native runtime configuration requires executor eyvan-native")
        if self.provider == "gcp" and not (self.provider_project and self.zone):
            raise ValueError("GCP cells require provider_project and zone")
        if self.capacity_backend in {"gcp-tpu", "gcp-attach"} and not self.accelerator_type:
            raise ValueError("GCP TPU cells require accelerator_type")
        tpu_backend = self.capacity_backend in {"gcp-tpu", "tpu"}
        if tpu_backend:
            # ``node_count`` is how a Cloud TPU queued resource states its
            # width; other backends size cells differently (indexed jobs,
            # host lists) and must not be measured with this ruler.
            node_count = self.provider_options.get("node_count", 1)
            if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 1:
                raise ValueError("GCP TPU provider_options.node_count must be a positive integer")
            if node_count != self.worker_count:
                raise ValueError("cell worker_count must match provider_options.node_count")
        node_prefix = self.provider_options.get("node_prefix")
        if self.worker_count == 1 and node_prefix is not None:
            raise ValueError("provider_options.node_prefix requires worker_count greater than one")
        if self.worker_count > 1:
            capabilities = backend_capabilities(self.capacity_backend)
            if not capabilities.multi_worker_cells:
                raise ValueError(
                    f"capacity backend {self.capacity_backend!r} does not build cells spanning several machines"
                )
            if self.executor != "eyvan-native":
                raise ValueError("multi-worker cells require the eyvan-native executor")
            if self.worker_count > capabilities.max_workers_per_cell:
                raise ValueError(
                    f"capacity backend {self.capacity_backend!r} builds at most "
                    f"{capabilities.max_workers_per_cell} workers per cell"
                )
            if tpu_backend and (
                not isinstance(node_prefix, str) or not node_prefix.strip() or node_prefix != node_prefix.strip()
            ):
                raise ValueError("multi-worker GCP TPU cells require provider_options.node_prefix")
        declared_worker_count = self.labels.get(WORKER_COUNT_LABEL)
        if declared_worker_count is not None and declared_worker_count != str(self.worker_count):
            raise ValueError(f"cell label {WORKER_COUNT_LABEL!r} must match worker_count")
        if any(price < 0 for price in self.unit_prices.values()):
            raise ValueError("cell unit prices must be non-negative")
        _validate_string_map(self.unit_prices, field_name="unit prices")
        _validate_string_map(self.provider_options, field_name="provider options", validate_string_values=False)
        if not self.runtime.is_empty and self.provider_options.get("startup_script") is not None:
            raise ValueError("use runtime.init_commands instead of provider_options.startup_script")
        _validate_string_map(self.labels, field_name="cell labels")
        _validate_string_map(self.topology, field_name="cell topology")
        capacity = resource_quantities(self.capacity)
        allocatable = resource_quantities(self.allocatable)
        if not any(quantity > 0 for quantity in capacity.values()):
            raise ValueError("cell capacity must contain a positive resource quantity")
        if not any(quantity > 0 for quantity in allocatable.values()):
            raise ValueError("cell allocatable capacity must contain a positive resource quantity")
        for resource in capacity.keys() | allocatable.keys():
            if allocatable.get(resource, 0) > capacity.get(resource, 0):
                raise ValueError(f"allocatable {resource} must fit within cell capacity")
        if self.max_replica_resources is not None:
            maximum = resource_quantities(self.max_replica_resources)
            if not any(quantity > 0 for quantity in maximum.values()):
                raise ValueError("max per-replica resources must contain a positive quantity")
            for resource in maximum.keys() | allocatable.keys():
                if maximum.get(resource, 0) > allocatable.get(resource, 0):
                    raise ValueError(f"max per-replica {resource} must fit within allocatable capacity")
        return self


class CapacityPoolRecord(EyvanModel):
    """Declarative node type used by the capacity autoscaler."""

    id: str = Field(max_length=192)
    fleet_name: str = Field(max_length=96)
    node_type: str = Field(max_length=96)
    #: Organization accountable for this pool, or ``None`` for shared capacity.
    #: Without it "an org admin may manage its own pools" had no referent: a
    #: pool belonged to nobody, so the only workable rule was fleet-admin-only.
    #: ``None`` keeps every existing pool shared and fleet-administered, so
    #: ownership is something an operator opts a pool into.
    owner_org_id: str | None = Field(default=None, max_length=128)
    provider: str
    capacity_backend: str | None = None
    provider_project: str | None = None
    region: str
    zone: str | None = None
    capacity: CanonicalResourceLimits
    allocatable: CanonicalResourceLimits
    unit_prices: dict[str, float] = Field(default_factory=dict)
    accelerator_type: str | None = None
    capacity_class: CanonicalCapacityClass = CapacityClass.ON_DEMAND
    executor: CanonicalExecutor = "eyvan-native"
    agent_endpoint: str
    provider_options: JsonDocument = Field(default_factory=dict)
    runtime: NativeRuntimeConfig = Field(default_factory=NativeRuntimeConfig)
    labels: dict[str, str] = Field(default_factory=dict)
    topology: dict[str, str] = Field(default_factory=dict)
    #: How this pool backs off when the provider says no, and how fast it
    #: forgets. Declared rather than compiled in: the right curve for a
    #: contended spot TPU zone is not the right curve for a rate-limited
    #: account, and there was no way to say so.
    availability_policy: AvailabilityPolicy = Field(default_factory=AvailabilityPolicy)
    #: How long this pool's capacity is held before the provider takes it back,
    #: whatever the job is doing. The property that makes ``flex_start`` a
    #: distinct class rather than a discount: a wait, then a window.
    #:
    #: Required for a flex-start pool, because the window exists whether or not
    #: anybody wrote it down -- and a scheduler that does not know it places
    #: eight-hour jobs on four-hour capacity and finds out at hour four.
    #: Permitted on other classes for a bounded commitment; absent means the
    #: capacity is held until Eyvan releases it.
    max_hold_s: int | None = Field(default=None, ge=60)
    max_replica_resources: CanonicalResourceLimits | None = None
    max_gang_replicas: int | None = Field(default=None, ge=1)
    #: Let several workloads that each ask for some chips (``TpuConfig(chips=...)``)
    #: share one cell's TPU hosts, each seeing only its own chips. Off by
    #: default, which keeps one workload per native cell; omitted from the wire
    #: document while false.
    chip_sharing: bool = False
    quota_group: str | None = Field(default=None, max_length=192)
    quota_limit: int | None = Field(default=None, ge=0)
    quota_units_per_cell: int | None = Field(default=None, ge=1)
    min_cells: int = Field(default=0, ge=0)
    #: Recurring windows that raise the floor above ``min_cells``. Scale-to-zero is
    #: correct overnight and wrong at the start of a working day, when every user pays
    #: the full provisioning latency at once.
    warm_windows: tuple[WarmWindow, ...] = ()
    initial_cells: int | None = Field(default=None, ge=0)
    max_cells: int = Field(default=1, ge=0, le=10_000)
    #: Seconds a cell holds capacity after its last lease leaves, so the next job
    #: reuses the slice instead of re-queuing for one. Reacquisition can take
    #: longer than the idle gap, especially in a contended zone. An hour spans
    #: the gap between jobs in an interactive session and avoids repeated cold
    #: starts. Holding longer costs nothing where the capacity is
    #: granted rather than billed. Set zero to release on the first sweep where
    #: holding a scarce slice costs someone else more than re-queuing costs you.
    idle_timeout_s: float = Field(default=3600, ge=0)
    queue_timeout_s: float | None = Field(default=None, gt=0)
    provisioning_timeout_s: float = Field(default=1800, gt=0)
    upscaling_speed: float = Field(default=1.0, gt=0)
    launch_batch: int = Field(default=5, ge=1)
    #: How long demand must stay below supply before autoscaling shrinks this
    #: pool. Without it one quiet minute starts draining cells that the next
    #: busy minute has to buy back at full cold-start price.
    shrink_stabilization_s: float = Field(default=300.0, ge=0)
    #: When the current shortfall below supply was first seen. ``None`` means
    #: supply is not in surplus (or no controller has judged it yet); the
    #: planner holds today's shape until this ages past
    #: ``shrink_stabilization_s``.
    demand_below_supply_since: datetime | None = None
    enabled: bool = True
    initialized_at: datetime | None = None
    generation: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @model_serializer(mode="wrap")
    def _wire_document(self, handler):
        document = handler(self)
        if not self.chip_sharing:
            document.pop("chip_sharing", None)
        return document

    @model_validator(mode="after")
    def validate_pool(self) -> CapacityPoolRecord:
        if self.capacity_class is CapacityClass.FLEX_START and self.max_hold_s is None:
            raise ValueError(
                "a flex_start pool must declare max_hold_s: the provider returns the capacity after a "
                "bounded window whether or not the job is finished, and a scheduler that does not know "
                "the window places work that cannot finish on it"
            )
        for field_name, value in (
            ("capacity pool id", self.id),
            ("fleet name", self.fleet_name),
            ("node type", self.node_type),
            ("capacity provider", self.provider),
            ("capacity region", self.region),
            ("cell executor", self.executor),
            ("agent endpoint", self.agent_endpoint),
        ):
            _validate_string(value, field_name=field_name)
        if not _SAFE_FLEET_COMPONENT.fullmatch(self.fleet_name):
            raise ValueError("fleet name must contain only lowercase letters, digits, and hyphens")
        if not _SAFE_FLEET_COMPONENT.fullmatch(self.node_type):
            raise ValueError("node type must contain only lowercase letters, digits, and hyphens")
        _validate_string(self.provider_project, field_name="provider project", optional=True)
        _validate_string(self.capacity_backend, field_name="capacity backend", optional=True)
        _validate_string(self.zone, field_name="capacity zone", optional=True)
        _validate_string(self.accelerator_type, field_name="accelerator type", optional=True)
        if self.min_cells > self.max_cells:
            raise ValueError("capacity pool min_cells must not exceed max_cells")
        if self.initial_cells is not None and not (self.min_cells <= self.initial_cells <= self.max_cells):
            raise ValueError("capacity pool initial_cells must be between min_cells and max_cells")
        if self.capacity_backend == "gcp-tpu":
            node_count = self.provider_options.get("node_count", 1)
            if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count != 1:
                raise ValueError("GCP TPU capacity pool provider_options.node_count must be exactly one when set")
            if self.provider_options.get("node_prefix") is not None:
                raise ValueError("GCP TPU capacity pool provider_options.node_prefix is autoscaler-owned")
        quota_fields = (self.quota_group, self.quota_limit, self.quota_units_per_cell)
        if any(value is not None for value in quota_fields):
            if self.quota_group is None or self.quota_limit is None or self.quota_units_per_cell is None:
                raise ValueError("quota_group, quota_limit, and quota_units_per_cell must be configured together")
            _validate_string(self.quota_group, field_name="GCP TPU quota group")
            if self.provider != "gcp" or self.capacity_backend != "gcp-tpu":
                raise ValueError("shared TPU quota budgets require provider gcp with capacity_backend gcp-tpu")
            if self.initial_target * self.quota_units_per_cell > self.quota_limit:
                raise ValueError("capacity pool initial quota demand exceeds quota_limit")
        CellRecord(
            id="capacity-pool-template",
            provider=self.provider,
            capacity_backend=self.capacity_backend,
            provider_project=self.provider_project,
            region=self.region,
            zone=self.zone,
            capacity=self.capacity,
            allocatable=self.allocatable,
            unit_prices=self.unit_prices,
            accelerator_type=self.accelerator_type,
            capacity_class=self.capacity_class,
            executor=self.executor,
            endpoint=self.agent_endpoint,
            pool_id=self.id,
            provider_options=self.provider_options,
            runtime=self.runtime,
            labels=self.labels,
            topology=self.topology,
            max_replica_resources=self.max_replica_resources,
            max_gang_replicas=self.max_gang_replicas,
        )
        return self

    @property
    def initial_target(self) -> int:
        return self.min_cells if self.initial_cells is None else self.initial_cells

    def warm_floor(self, now: datetime) -> int:
        """Return the cell floor in force at ``now``.

        The highest matching window wins rather than the first, so overlapping
        windows compose predictably however they are ordered -- an operator adding a
        "conference week" window on top of "business hours" gets the larger of the
        two, not whichever happens to be listed first.
        """

        # ``min_cells`` participates in the max rather than acting as a fallback: a
        # window only ever *raises* the floor, so one declaring fewer cells than the
        # pool's own baseline must not lower it.
        return max((self.min_cells, *(window.min_cells for window in self.warm_windows if window.covers(now))))

    @property
    def template_fingerprint(self) -> str:
        """Return the immutable capacity identity used to pin new workloads during rollouts."""

        payload = self.model_dump(
            mode="json",
            include={
                "provider",
                "capacity_backend",
                "provider_project",
                "region",
                "zone",
                "capacity",
                "accelerator_type",
                "capacity_class",
                "executor",
                "agent_endpoint",
                "provider_options",
                "runtime",
            },
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def hosts_multi_worker_cells(self) -> bool:
        """Return whether this pool can mint a cell wider than one worker.

        The readable form of the refusals :meth:`make_cell` raises, and the one
        every caller that must *decide* rather than *attempt* should ask. A
        multi-worker cell is one Cloud TPU multislice queued resource, which the
        GCP TPU backend is the only thing that can create and the native executor
        the only thing that can fence -- so a pool failing this can never host the
        shape, whatever else is true of it.

        It exists as a predicate because the alternative is calling ``make_cell``
        to find out: that answers by raising, and the planner asks this question
        inside a leader phase where a raise stops the control plane.
        """

        return backend_capabilities(self.capacity_backend).multi_worker_cells and self.executor == "eyvan-native"

    def make_cell(
        self,
        cell_id: str,
        *,
        worker_count: int = 1,
        autoscale_request_workload_id: str | None = None,
    ) -> CellRecord:
        """Materialize one managed cell containing ``worker_count`` complete TPU workers."""

        if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
            raise ValueError("capacity cell worker_count must be a positive integer")
        provider_options = dict(self.provider_options)
        provider_options.pop("node_count", None)
        if worker_count > 1:
            if self.provider != "gcp" or self.capacity_backend != "gcp-tpu":
                raise ValueError("multi-worker cells require a GCP TPU capacity pool")
            if self.executor != "eyvan-native":
                raise ValueError("multi-worker cells require the eyvan-native executor")
            if worker_count > MAX_GCP_TPU_MULTISLICE_WORKERS:
                raise ValueError(f"GCP TPU multislice worker_count must not exceed {MAX_GCP_TPU_MULTISLICE_WORKERS}")
            provider_options["node_count"] = worker_count
            provider_options["node_prefix"] = _tpu_node_prefix(cell_id)

        return CellRecord(
            id=cell_id,
            provider=self.provider,
            capacity_backend=self.capacity_backend,
            provider_project=self.provider_project,
            region=self.region,
            zone=self.zone,
            capacity=scale_resources(self.capacity, worker_count),
            allocatable=scale_resources(self.allocatable, worker_count),
            unit_prices=self.unit_prices,
            accelerator_type=self.accelerator_type,
            capacity_class=self.capacity_class,
            executor=self.executor,
            endpoint=self.agent_endpoint,
            pool_id=self.id,
            autoscale_request_workload_id=autoscale_request_workload_id,
            provider_options=provider_options,
            runtime=self.runtime,
            availability_policy=self.availability_policy,
            max_hold_s=self.max_hold_s,
            labels={
                **self.labels,
                "eyvan_fleet": self.fleet_name,
                "eyvan_pool": self.id,
                "eyvan_pool_template": self.template_fingerprint,
                WORKER_COUNT_LABEL: str(worker_count),
            },
            topology=self.topology,
            max_replica_resources=(
                scale_resources(self.max_replica_resources, worker_count)
                if self.max_replica_resources is not None
                else None
            ),
            max_gang_replicas=self.max_gang_replicas,
            worker_count=worker_count,
            chip_sharing=self.chip_sharing,
        )


def validate_capacity_pool_quota_groups(pools: tp.Iterable[CapacityPoolRecord]) -> None:
    """Validate finite GCP TPU budgets shared by multiple autoscaling pools.

    A group represents one provider quota dimension, normally a GCP project,
    zone, generation, and purchase class. Different TPU shapes may consume a
    different number of units from that shared ceiling.
    """

    grouped: dict[str, list[CapacityPoolRecord]] = {}
    for pool in pools:
        if pool.quota_group is not None:
            grouped.setdefault(pool.quota_group, []).append(pool)

    for group, members in grouped.items():
        limits = {pool.quota_limit for pool in members}
        if len(limits) != 1:
            raise ValueError(f"quota group {group!r} must use one quota_limit across every pool")
        scopes = {
            (
                pool.provider,
                pool.capacity_backend,
                pool.provider_project,
                pool.region,
                pool.zone,
                pool.capacity_class.value,
            )
            for pool in members
        }
        if len(scopes) != 1:
            raise ValueError(f"quota group {group!r} cannot span distinct GCP TPU quota scopes")
        quota_limit = next(iter(limits))
        assert quota_limit is not None
        initial_units = 0
        for pool in members:
            quota_units_per_cell = pool.quota_units_per_cell
            assert quota_units_per_cell is not None
            initial_units += pool.initial_target * quota_units_per_cell
        if initial_units > quota_limit:
            raise ValueError(
                f"quota group {group!r} initial capacity requires {initial_units} quota units, "
                f"but quota_limit is {quota_limit}"
            )


class LeaseRecord(EyvanModel):
    """Fenced assignment of one workload to one execution cell."""

    id: str = Field(max_length=96)
    workload_id: str = Field(max_length=96)
    cell_id: str = Field(max_length=96)
    cell_generation: int = Field(default=1, ge=1)
    org_id: str = Field(max_length=128)
    project_id: str = Field(max_length=128)
    state: LeaseState = LeaseState.PENDING
    resources: ResourceGang
    hourly_cost: float = Field(default=0, ge=0)
    reserved_cost: float = Field(default=0, ge=0)
    resource_hours_reserved: CanonicalResourceHours = Field(default_factory=ResourceHours)
    settled_cost: float = Field(default=0, ge=0)
    accelerator_hours_settled: dict[str, float] = Field(default_factory=dict)
    settlement_reason: str | None = None
    settled_at: datetime | None = None
    #: Budget accounting epoch charged by this settlement. None means no
    #: attributable budget charge (including documents written before tracking).
    settlement_budget_period_id: str | None = Field(default=None, min_length=1, max_length=64)
    #: Write-once, stamped the first time this lease enters ACTIVE -- which the
    #: dispatch transaction already commits atomically with the workload's move to
    #: DISPATCHING, so ACTIVE *is* this system's run boundary and no new plumbing is
    #: needed to observe it. `created_at -> run_started_at` is queued time (capacity
    #: reserved, payload not started) and `run_started_at -> settled_at` is running
    #: time. None at settlement means the lease never ran and is pure queue.
    run_started_at: datetime | None = None
    #: Settled clock, recorded so cost is auditable without re-deriving it from
    #: timestamps that later records may not carry.
    queued_seconds: float = Field(default=0, ge=0)
    running_seconds: float = Field(default=0, ge=0)
    generation: int = 1
    fencing_token: str
    group_id: str | None = Field(default=None, max_length=128)
    group_index: int = Field(default=0, ge=0)
    revocation_reason: Literal["preemption"] | None = None
    #: Global indexes of the TPU chips this lease holds on a chip-sharing cell.
    #: ``None`` means the whole cell, and is omitted from the wire document.
    tpu_chips: tuple[int, ...] | None = None
    expires_at: datetime
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @model_serializer(mode="wrap")
    def _wire_document(self, handler):
        document = handler(self)
        if self.settlement_budget_period_id is None:
            document.pop("settlement_budget_period_id", None)
        if self.tpu_chips is None:
            document.pop("tpu_chips", None)
        return document

    @model_validator(mode="after")
    def validate_settlement(self) -> LeaseRecord:
        _validate_string(self.id, field_name="lease id")
        _validate_string(self.workload_id, field_name="lease workload id")
        _validate_string(self.cell_id, field_name="lease cell id")
        _validate_string(self.org_id, field_name="lease organization id")
        _validate_string(self.project_id, field_name="lease project id")
        _validate_string(self.fencing_token, field_name="lease fencing token")
        _validate_string(self.group_id, field_name="lease group id", optional=True)
        if any(value < 0 for value in self.accelerator_hours_settled.values()):
            raise ValueError("settled accelerator hours must be non-negative")
        if self.settled_at is None and (
            self.settled_cost or self.accelerator_hours_settled or self.settlement_reason is not None
        ):
            raise ValueError("lease settlement values require settled_at")
        if self.settled_at is not None and self.settlement_reason is None:
            raise ValueError("settled leases require a settlement reason")
        return self


class PolicyRecord(EyvanModel):
    """Placement policy for an organization project."""

    org_id: str = Field(max_length=128)
    project_id: str = Field(max_length=128)
    allowed_providers: tuple[str, ...] = ()
    allowed_regions: tuple[str, ...] = ()
    allowed_capacity_classes: tuple[CanonicalCapacityClass, ...] = ()
    denied_accelerator_types: tuple[str, ...] = ()
    max_priority: int = Field(default=1000, ge=0, le=1000)
    #: Scheme/authority/path-segment roots this project's workloads may mount.
    #: Empty means none, which is the safe default and not the permissive one.
    #:
    #: This is the security-critical field. Without it, any submitter in a
    #: project can name any secret the *agent's* identity can read -- and the
    #: agent's identity is broad by construction, because it provisions
    #: capacity. That is a privilege escalation from "may submit jobs" to "may
    #: read every secret in the project", available to anyone who can write a
    #: spec.
    allowed_secret_references: tuple[str, ...] = ()
    preemption_enabled: bool = True
    revision: int = 1
    updated_by: str
    updated_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_policy_values(self) -> PolicyRecord:
        _validate_string(self.org_id, field_name="policy organization id")
        _validate_string(self.project_id, field_name="policy project id")
        _validate_string(self.updated_by, field_name="policy updater id")
        _validate_strings(self.allowed_providers, field_name="allowed providers")
        _validate_strings(self.allowed_regions, field_name="allowed regions")
        _validate_strings(self.denied_accelerator_types, field_name="denied accelerator types")
        _validate_strings(self.allowed_secret_references, field_name="allowed secret references")
        for reference in self.allowed_secret_references:
            validate_secret_reference(reference, policy_root=True)
        return self


class ScopeQuota(EyvanModel):
    """One scope of the quota hierarchy, and whether anything is declared there.

    Attributes:
        scope: ``org``, ``project`` or ``principal``.
        org_id / project_id / principal_id: the scope's identity.
        quota: the record declared at this scope, or ``None`` when nothing is.
            Absence means this scope imposes no limit -- which is what admission
            does with it -- and is a different answer from a limit of zero.
    """

    scope: tp.Literal["org", "project", "principal"]
    org_id: str = Field(max_length=128)
    project_id: str | None = Field(default=None, max_length=128)
    principal_id: str | None = Field(default=None, max_length=128)
    quota: QuotaRecord | None = None


class EffectiveQuota(EyvanModel):
    """Every quota that applies to one scope, and whether any of them bind.

    Admission walks the whole chain and enforces *every* quota it finds, so
    "the quota that applies" is not one row. Reporting the narrowest stored row
    would answer a question nobody asked, and reporting a 404 when nothing is
    stored -- which is what this used to do -- calls the ordinary state of a
    fresh fleet an error.

    ``unlimited`` is the honest reading of an empty hierarchy: no scope declares
    a limit, so nothing here constrains the workload. It is stated rather than
    left to be inferred from an empty list, because an empty list is exactly
    what a caller would also see if the read had silently failed.
    """

    org_id: str = Field(max_length=128)
    project_id: str | None = Field(default=None, max_length=128)
    principal_id: str | None = Field(default=None, max_length=128)
    scopes: tuple[ScopeQuota, ...] = ()

    @property
    def unlimited(self) -> bool:
        """Return whether no scope in the chain declares a limit."""

        return all(entry.quota is None for entry in self.scopes)

    @model_validator(mode="after")
    def validate_chain(self) -> EffectiveQuota:
        """Keep the reported chain in the order admission walks it."""

        order = [entry.scope for entry in self.scopes]
        if order != sorted(order, key=("org", "project", "principal").index):
            raise ValueError("effective quota scopes must be ordered org, project, principal")
        return self


class QuotaRecord(EyvanModel):
    """Hierarchical resource quota and DRF weight."""

    org_id: str = Field(max_length=128)
    project_id: str | None = Field(default=None, max_length=128)
    principal_id: str | None = Field(default=None, max_length=128)
    limits: CanonicalResourceLimits
    constrained_resources: tuple[str, ...] = ()
    #: Ceiling on indexed managed-artifact bytes in this scope. Separate from
    #: ``limits`` because that vector is *schedulable* capacity resolved against
    #: cells, while artifact bytes are storage the control plane holds for the
    #: project whether or not anything is running. ``None`` means unlimited,
    #: which is the historical behaviour and stays the default.
    artifact_bytes_limit: int | None = Field(default=None, ge=0)
    weight: float = Field(default=1.0, gt=0)
    revision: int = 1
    updated_by: str
    updated_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="before")
    @classmethod
    def infer_constrained_resources(cls, value: tp.Any) -> tp.Any:
        if not isinstance(value, tp.Mapping) or "constrained_resources" in value:
            return value
        data = dict(value)
        limits = data.get("limits")
        names: set[str] = set()
        if isinstance(limits, ResourceVector):
            names.update(limits.model_fields_set - {"custom"})
            if "custom" in limits.model_fields_set:
                names.update(limits.custom)
        elif isinstance(limits, tp.Mapping):
            names.update(str(name) for name in limits if name != "custom")
            custom = limits.get("custom")
            if isinstance(custom, tp.Mapping):
                names.update(str(name) for name in custom)
        data["constrained_resources"] = tuple(sorted(names))
        return data

    @model_validator(mode="after")
    def validate_scope(self) -> QuotaRecord:
        _validate_string(self.org_id, field_name="quota organization id")
        _validate_string(self.project_id, field_name="quota project id", optional=True)
        _validate_string(self.principal_id, field_name="quota principal id", optional=True)
        _validate_string(self.updated_by, field_name="quota updater id")
        _validate_strings(self.constrained_resources, field_name="quota constrained resources")
        available = {"cpu", "memory_gib", "gpu", "tpu", *self.limits.custom}
        unknown = sorted(set(self.constrained_resources) - available)
        if unknown:
            raise ValueError(f"quota constrained resource {unknown[0]!r} is missing from limits")
        if self.principal_id is not None and self.project_id is None:
            raise ValueError("principal quota scope requires project_id")
        return self


class OrganizationRecord(EyvanModel):
    """A tenant this control plane knows about.

    Until this existed, ``org_id`` was a free-form string on every request and a
    tenant was whatever anyone typed. Nothing could answer "which organizations
    exist", so nothing could list them, bill them, or refuse a typo -- a
    ``PUT /quotas`` naming ``acmee`` created a quota for a tenant that would
    never submit anything, and it looked exactly like success.

    Registration is deliberately thin. It is a name and an owner, not a
    settings object: quotas, budgets and policies are already their own records
    scoped by ``org_id``, and duplicating any of them here would create two
    places to look and one of them wrong.
    """

    id: str = Field(max_length=128)
    display_name: str = Field(default="", max_length=200)
    #: Subject of the principal accountable for this tenant. Not a grant --
    #: authority still comes from :class:`RoleGrant` -- but the answer to "who
    #: do we talk to", which an audit trail cannot reconstruct after the fact.
    owner: str = Field(min_length=1, max_length=256)
    labels: dict[str, str] = Field(default_factory=dict)
    #: Registered-but-inactive rather than deleted. Deleting an organization
    #: whose workloads, leases and audit events still reference it would orphan
    #: all of them; disabling refuses new work while leaving history readable.
    enabled: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    created_by: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_organization(self) -> OrganizationRecord:
        _validate_string(self.id, field_name="organization id")
        _validate_string(self.owner, field_name="organization owner")
        _validate_string(self.created_by, field_name="organization creator")
        _validate_string_map(self.labels, field_name="organization labels")
        return self


class ProjectRecord(EyvanModel):
    """A project inside one registered organization.

    Scoped by ``(org_id, id)`` because project names are only unique within a
    tenant: two organizations both having a ``research`` project is the ordinary
    case, and a globally unique project id would make the second one impossible
    to create for reasons no user could see.
    """

    org_id: str = Field(max_length=128)
    id: str = Field(max_length=128)
    display_name: str = Field(default="", max_length=200)
    owner: str = Field(min_length=1, max_length=256)
    labels: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    created_by: str = Field(min_length=1, max_length=256)

    @property
    def key(self) -> tuple[str, str]:
        """Return the scoped identity this record is stored under."""

        return self.org_id, self.id

    @model_validator(mode="after")
    def validate_project(self) -> ProjectRecord:
        _validate_string(self.org_id, field_name="project organization id")
        _validate_string(self.id, field_name="project id")
        _validate_string(self.owner, field_name="project owner")
        _validate_string(self.created_by, field_name="project creator")
        _validate_string_map(self.labels, field_name="project labels")
        return self


class BudgetRecord(EyvanModel):
    """Monetary admission budget for an organization project."""

    org_id: str = Field(max_length=128)
    project_id: str = Field(max_length=128)
    limit: float = Field(ge=0)
    spent: float = Field(default=0, ge=0)
    accelerator_hour_limits: dict[str, float] = Field(default_factory=dict)
    accelerator_hours_spent: dict[str, float] = Field(default_factory=dict)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    period: str = "monthly"
    #: Start of the window ``spent`` accumulates over. Nothing used to reset ``spent``,
    #: which was harmless only while it held a duration *estimate* that ran roughly as
    #: often high as low. Once settlement became actual elapsed cost, ``spent`` became a
    #: true monotone total and ``period`` a lie: every alert threshold would fire once
    #: in the fleet's lifetime and never again. None means "not yet rolled"; the first
    #: settlement stamps it.
    period_started_at: datetime | None = None
    #: Server-owned accounting epoch and retained spend at its opening. This
    #: baseline need not be zero: initializing a legacy window retains spend.
    #: Missing attribution is unknown, never an inferred zero opening balance.
    settlement_period_id: str | None = Field(default=None, min_length=1, max_length=64)
    period_opening_spent: float | None = Field(default=None, ge=0)
    #: Fractions of ``limit`` that emit an audit event the first time spend crosses
    #: them in a period. Warning before the wall is the point: a hard budget that
    #: speaks only by refusing admission tells an operator at the exact moment when
    #: the only remaining option is to raise the limit.
    alert_thresholds: tuple[float, ...] = ()
    #: Thresholds already announced in the current period, cleared when it rolls.
    #: Without this, every settlement past 80% re-announces 80%, and an alert that
    #: repeats on every job is an alert nobody reads.
    alerted_thresholds: tuple[float, ...] = ()
    hard_limit: bool = True
    revision: int = 1
    updated_by: str
    updated_at: datetime = Field(default_factory=utcnow)

    @model_serializer(mode="wrap")
    def _wire_document(self, handler):
        document = handler(self)
        for name in ("settlement_period_id", "period_opening_spent"):
            if getattr(self, name) is None:
                document.pop(name, None)
        return document

    @model_validator(mode="after")
    def validate_spend(self) -> BudgetRecord:
        _validate_string(self.org_id, field_name="budget organization id")
        _validate_string(self.project_id, field_name="budget project id")
        _validate_string(self.currency, field_name="budget currency")
        if self.currency != "USD":
            raise ValueError("Eyvan v0 budgets support only USD")
        _validate_string(self.period, field_name="budget period")
        _validate_string(self.updated_by, field_name="budget updater id")
        if any(value < 0 for value in self.accelerator_hour_limits.values()):
            raise ValueError("accelerator-hour limits must be non-negative")
        if any(value < 0 for value in self.accelerator_hours_spent.values()):
            raise ValueError("accelerator-hour spend must be non-negative")
        _validate_string_map(self.accelerator_hour_limits, field_name="accelerator-hour limits")
        _validate_string_map(self.accelerator_hours_spent, field_name="accelerator-hour spend")
        if any(not 0 < threshold <= 1 for threshold in self.alert_thresholds):
            raise ValueError("budget alert thresholds must be fractions in (0, 1]")
        if len(set(self.alert_thresholds)) != len(self.alert_thresholds):
            raise ValueError("budget alert thresholds must not repeat")
        unknown = sorted(set(self.alerted_thresholds) - set(self.alert_thresholds))
        if unknown:
            raise ValueError(f"budget records an alert for undeclared threshold {unknown[0]}")
        return self

    @property
    def utilization(self) -> float:
        """Return spend as a fraction of the limit, or zero for a zero limit.

        Zero rather than infinity: a zero-limit budget refuses everything, so its
        utilization is a number nobody acts on, and dividing by zero here would
        break every report that lists budgets alongside one another.
        """

        return 0.0 if self.limit <= 0 else self.spent / self.limit

    def thresholds_crossed_by(self, spent: float) -> tuple[float, ...]:
        """Return declared thresholds ``spent`` reaches that have not yet fired."""

        if self.limit <= 0:
            return ()
        fraction = spent / self.limit
        already = set(self.alerted_thresholds)
        return tuple(
            sorted(
                threshold for threshold in self.alert_thresholds if threshold <= fraction and threshold not in already
            )
        )


class PoolEstimate(EyvanModel):
    """What one pool would charge for one workload, and where the price came from."""

    pool_id: str
    accelerator_type: str | None = None
    capacity_class: str
    region: str
    zone: str | None = None
    #: Cells this workload needs in this pool, after ``worker_count`` scaling.
    cells: int = Field(ge=1)
    hourly_cost: float = Field(ge=0)
    estimated_cost: float = Field(ge=0)
    #: Where the price was resolved from, or ``None`` when the pool declares none.
    #: Reported rather than defaulted to zero: an unpriced pool estimating "free"
    #: is defect D7 wearing a friendly face.
    price_source: str | None = None


class WorkloadEstimate(EyvanModel):
    """Pre-submission cost estimate for one workload definition.

    Answers "what will this cost, and will my budget take it" *before* a record
    exists -- the question every operator asks and that previously required
    submitting and reading the resulting lease's ``reserved_cost``.

    Every figure is an estimate over ``estimated_duration_seconds``, which is a
    user-supplied number. The estimate is only as honest as that input, and the
    field is named to say so rather than presenting a projection as a quote.
    """

    org_id: str
    project_id: str
    estimated_duration_seconds: int = Field(ge=1)
    #: Candidate pools, cheapest first. Empty means nothing in the fleet matches
    #: the request -- a different answer from "it costs zero".
    candidates: tuple[PoolEstimate, ...] = ()
    #: Current spend and limit at estimate time, when a budget exists.
    budget_limit: float | None = None
    budget_spent: float | None = None
    budget_hard_limit: bool = True
    #: Pools whose estimate would not fit the remaining hard budget. Named rather
    #: than filtered out: "your budget refuses the cheap option too" is the useful
    #: message, and silently dropping candidates would look like no capacity.
    budget_blocked_pool_ids: tuple[str, ...] = ()
    #: Pools matched but declaring no price. Their estimate is zero and meaningless.
    unpriced_pool_ids: tuple[str, ...] = ()

    @property
    def cheapest(self) -> PoolEstimate | None:
        """Return the cheapest candidate, or ``None`` when nothing matched."""

        return self.candidates[0] if self.candidates else None


class QueueTrendPoint(EyvanModel):
    """One time bucket of the submitted-vs-admitted trend."""

    started_at: datetime
    submitted: int = Field(ge=0)
    admitted: int = Field(ge=0)


class QueueTrend(EyvanModel):
    """Queue pressure over recent time, derived from the audit ledger.

    Bucketed by wall clock, not by event count: the question is "is the queue
    growing", and count-based buckets compress exactly the quiet periods that
    answer it.
    """

    observed_at: datetime
    bucket_seconds: float = Field(gt=0)
    points: tuple[QueueTrendPoint, ...] = ()


class FederatedBudgetStatus(EyvanModel):
    """One tenant's spend across every plane, as this plane understands it.

    The peer figures are reported, not observed: each is as fresh as that peer's
    last report. That is why this number informs and alerts and never admits --
    admission stays local, against a grant already bounded when it was issued, so
    a stale report can make this report wrong without making spend unbounded.
    """

    org_id: str
    project_id: str
    home_plane_id: str | None = None
    limit: float = Field(ge=0)
    #: Spend on this plane, from its own settled leases.
    local_spent: float = Field(ge=0)
    #: Currently authorized on peers, whether or not they have used it.
    granted_away: float = Field(ge=0)
    reported_peer_spend: float = Field(ge=0)
    total_spent: float = Field(ge=0)
    active_grants: int = Field(ge=0)
    #: Active grants whose holder has never reported. Named because it is the
    #: difference between "the peers are idle" and "the peers are not talking",
    #: and a total that silently mixes the two is worse than no total.
    stale_reports: int = Field(ge=0)


class BudgetStatus(EyvanModel):
    """A budget's spend, its period window, and where it is heading.

    Two of these fields revive dashboard surfaces that could never populate:
    ``forecast`` was typed and never set, so "Current forecast" always read
    Unavailable and the alert block behind it could not fire; ``period_ends_at``
    is what makes ``spent`` interpretable at all, since a number with no window
    attached cannot be compared to a limit that has one.
    """

    org_id: str
    project_id: str
    limit: float = Field(ge=0)
    spent: float = Field(ge=0)
    currency: str = "USD"
    period: str = "monthly"
    period_started_at: datetime | None = None
    period_ends_at: datetime | None = None
    hard_limit: bool = True
    #: Projected end-of-period spend: current spend plus what every active lease
    #: will cost for the rest of the window. Derived, never stored -- a stored
    #: forecast is a number that goes stale the moment a lease starts or ends.
    forecast: float | None = None
    alert_thresholds: tuple[float, ...] = ()
    alerted_thresholds: tuple[float, ...] = ()

    @property
    def utilization(self) -> float:
        """Return spend as a fraction of the limit, or zero for a zero limit."""

        return 0.0 if self.limit <= 0 else self.spent / self.limit

    @property
    def forecast_exceeds_limit(self) -> bool:
        """Return whether this budget is projected to run out before its period does."""

        return self.forecast is not None and self.limit > 0 and self.forecast > self.limit


class AuditEvent(EyvanModel):
    """Immutable, ordered audit event emitted by a committed transaction."""

    id: str = Field(max_length=96)
    sequence: int = 0
    timestamp: datetime = Field(default_factory=utcnow)
    actor: str
    org_id: str | None = Field(default=None, max_length=128)
    project_id: str | None = Field(default=None, max_length=128)
    entity_type: str = Field(max_length=64)
    entity_id: str = Field(max_length=256)
    action: str
    previous_state: str | None = None
    new_state: str | None = None
    payload: JsonDocument = Field(default_factory=dict)


#: Cell states in which the provider resource exists and is being charged for.
#: REGISTERING is deliberately excluded: a queued TPU request is a record Eyvan
#: holds and a bill nobody is paying, and charging it would make queue time --
#: the one genuinely free interval -- look like spend. DELETING is included
#: because the resource still exists until the delete completes.
BILLABLE_CELL_STATES = frozenset({CellState.READY, CellState.DRAINING, CellState.UNAVAILABLE, CellState.DELETING})


class PoolWarmth(EyvanModel):
    """Whether one pool's warm floor is currently held, and what it costs.

    A warm window is a promise an operator wrote and nothing reported on. Two
    things go wrong invisibly: the floor is not met, so the 09:00 latency the
    window exists to prevent happens anyway; or the floor is met and the cells sit
    unused, which is *correct* and looks identical to waste in every other view.

    That second case is why this joins the billing clock. Warm capacity is
    deliberately idle. Without a way to name it, an operator reading idle cost
    cannot tell the capacity they asked to keep hot from the capacity their
    packing is wasting, and the only safe reaction to an unexplained idle number
    is to delete the warm floor -- undoing the fix.
    """

    pool_id: str
    #: The floor in force at the reported moment: ``min_cells`` raised by any
    #: window covering it. Zero means scale-to-zero is permitted right now.
    floor: int = Field(ge=0)
    #: Whether a *window* is what raised the floor, as opposed to the pool's
    #: standing ``min_cells``. An operator debugging "why is this pool not empty
    #: at 03:00" needs these distinguished.
    window_active: bool = False
    #: Ready cells that currently count toward the floor.
    warm_cells: int = Field(ge=0)
    #: Floor cells that are requested but not ready yet. The planner counts them
    #: as supply, so it will not launch more; they are not warm until they serve.
    provisioning_cells: int = Field(default=0, ge=0)
    #: Warm cells holding no lease. Not waste: this is the capacity the floor
    #: exists to reserve, and it is the number that must be subtracted from a
    #: fleet's idle cost before that cost means anything.
    warm_idle_cells: int = Field(ge=0)
    hourly_cost: float = Field(default=0.0, ge=0)

    @property
    def satisfied(self) -> bool:
        """Return whether the pool currently holds its floor."""

        return self.warm_cells >= self.floor


class RegionCarbon(EyvanModel):
    """What is known about one region's grid intensity, including nothing.

    ``intensity`` is null and ``reason`` is populated whenever there is no
    usable reading, which today is always: no provider ships. Reported rather
    than omitted, because an operator asking about carbon deserves the answer
    "nothing is configured" instead of an empty list they have to interpret.
    """

    region: str
    #: Grams of CO2e per kWh, or null. Never zero as a stand-in -- zero claims a
    #: carbon-free grid, which is the strongest statement the field can make.
    intensity: float | None = None
    #: Who measured it and when. Both null exactly when ``intensity`` is.
    source: str | None = None
    observed_at: datetime | None = None
    #: Why there is no reading, from the closed ``CarbonUnavailableReason`` set.
    reason: str | None = None
    detail: str = ""


class CarbonReport(EyvanModel):
    """Grid intensity across the regions the fleet actually occupies.

    Deliberately not part of placement. A carbon number that silently steered
    scheduling would produce decisions that look deliberate and are not, so this
    is a report and only a report until someone brings a source they trust; a
    test asserts the placement path does not import the module behind it.
    """

    observed_at: datetime = Field(default_factory=utcnow)
    regions: tuple[RegionCarbon, ...] = ()

    @property
    def configured(self) -> bool:
        """Whether any region produced a usable reading."""

        return any(region.intensity is not None for region in self.regions)


class PoolLaunchLatency(EyvanModel):
    """How long one pool takes to hand over usable capacity, and on what evidence.

    The scheduler has ranked pools by this number since it existed -- ``p95`` of
    the same ``provision_latencies_s`` samples -- and nothing ever showed it to
    the person deciding which pool to ask for. "Why did my job land in the slow
    region" had no answer that did not involve reading the scheduler.

    ``samples`` is reported beside every percentile and is not decoration. A p95
    over three provisions is not a p95, and a report that presented one as
    equivalent to a p95 over three hundred would be inviting exactly the
    confident wrong decision this exists to prevent.
    """

    pool_id: str
    accelerator_type: str | None = None
    #: Nearest-rank p95 over the retained samples, in seconds. ``None`` when the
    #: pool has never successfully provisioned -- distinct from zero, which is a
    #: pool that provisions instantly.
    p95_seconds: float | None = None
    median_seconds: float | None = None
    slowest_seconds: float | None = None
    #: How many provisions back the figures above.
    samples: int = Field(default=0, ge=0)
    #: Cells currently held back by a cooldown after repeated failures. A pool
    #: whose latency looks excellent because it has stopped being asked is not a
    #: fast pool.
    cells_in_cooldown: int = Field(default=0, ge=0)


class LaunchLatencyReport(EyvanModel):
    """Provisioning latency across every pool, at one moment."""

    observed_at: datetime = Field(default_factory=utcnow)
    pools: tuple[PoolLaunchLatency, ...] = ()

    @property
    def unmeasured(self) -> tuple[PoolLaunchLatency, ...]:
        """Return pools with no successful provision to report on.

        Surfaced separately because "no data" and "fast" are the same shape in a
        table sorted by latency, and one of them is a pool nobody should be
        planning around.
        """

        return tuple(pool for pool in self.pools if pool.samples == 0)


class WarmthReport(EyvanModel):
    """Warm-floor status across every pool, at one moment."""

    observed_at: datetime = Field(default_factory=utcnow)
    pools: tuple[PoolWarmth, ...] = ()

    @property
    def unsatisfied(self) -> tuple[PoolWarmth, ...]:
        """Return pools whose floor is declared and not currently held."""

        return tuple(pool for pool in self.pools if not pool.satisfied)

    @property
    def warm_hourly_cost(self) -> float:
        """Return what holding every floor costs per hour."""

        return costing.total(*(pool.hourly_cost for pool in self.pools))


class CellBillingAccrual(EyvanModel):
    """What one accrual pass added to the fleet's cell clocks."""

    cells: int = Field(ge=0)
    accrued_seconds: float = Field(ge=0)
    idle_seconds: float = Field(ge=0)
    accrued_cost: float = Field(ge=0)
    idle_cost: float = Field(ge=0)


class CellBillingReport(EyvanModel):
    """Accrued and idle cell cost over a fleet or one pool."""

    pool_id: str | None = None
    cells: int = Field(ge=0)
    billed_seconds: float = Field(ge=0)
    idle_seconds: float = Field(ge=0)
    accrued_cost: float = Field(ge=0)
    idle_cost: float = Field(ge=0)
    #: Cells contributing zero cost because their pool declares no unit prices.
    #: Reported rather than hidden: without it, defect D7's fleet reads as a
    #: fleet that genuinely costs nothing.
    unpriced_cells: int = Field(ge=0)


class SubjectUsageRecord(EyvanModel):
    """What one person used, and what it cost.

    Eyvan settled cost per lease and reported it per pool, so the fleet's
    spending was answerable and "who spent it" was not -- a shared project could
    see its bill and not its breakdown. Every input already existed: a lease
    carries settled cost and accelerator hours, and the workload it belongs to
    carries who submitted it.

    ``unpriced_leases`` is reported rather than folded into zero. A pool with no
    declared unit price contributes nothing to ``settled_cost``, and without
    this a heavy user of unpriced capacity is indistinguishable from an idle
    one -- which is the reading that gets somebody's quota cut.
    """

    subject: str
    org_id: str
    project_id: str
    workloads: int = Field(ge=0)
    #: Terminal workloads only. A run still going has not settled, and counting
    #: it would make the total move backwards when it finishes cheaper than its
    #: reservation.
    settled_workloads: int = Field(ge=0)
    leases: int = Field(ge=0)
    accelerator_hours: float = Field(ge=0)
    running_seconds: float = Field(ge=0)
    queued_seconds: float = Field(ge=0)
    settled_cost: float = Field(ge=0)
    unpriced_leases: int = Field(ge=0)
    first_submitted_at: datetime | None = None
    last_submitted_at: datetime | None = None


class ProjectUsageReport(EyvanModel):
    """Per-subject usage across one project, newest activity first."""

    org_id: str
    project_id: str
    observed_at: datetime
    subjects: tuple[SubjectUsageRecord, ...] = ()

    @property
    def total_cost(self) -> float:
        """Return what the whole project settled over the same window."""

        return sum(subject.settled_cost for subject in self.subjects)


class WorkloadSpendRecord(EyvanModel):
    """What one run used and was charged, with the on-demand alternative.

    Computed by the control plane rather than by whoever is asking, because the
    honest answer needs pool unit prices and those are a fleet read. Assembling
    it in the client meant `eyvan workload cost` listed capacity pools, so the
    one command that answers "what did this cost me" was refused to everybody
    who could not administer the fleet -- while its own help said nothing about
    that.

    ``priced`` is the field that matters. An unpriced pool settles to zero, so
    ``settled`` means nothing when this is false, and reporting `$0.00` would
    present the absence of a price as the absence of a bill.
    """

    workload_id: str
    state: str
    #: Measured, so always known even when nothing here can be priced.
    accelerator_hours: dict[str, float] = Field(default_factory=dict)
    settled: float = Field(default=0, ge=0)
    priced: bool = False
    reserved: float = Field(default=0, ge=0)
    leases: int = Field(default=0, ge=0)
    #: The shape and class this run held, from the pool it was pinned to.
    shape: str | None = None
    capacity_class: str | None = None
    #: An on-demand price for the same shape, and where it came from. Never
    #: invented: Eyvan carries no price catalogue and no spot-discount constant,
    #: so this is a price some on-demand pool declares or nothing at all, and
    #: ``on_demand_source`` says which.
    on_demand_hourly: float | None = Field(default=None, ge=0)
    on_demand_source: str = ""


class WebhookRecord(EyvanModel):
    """Where one project wants its lifecycle events pushed.

    The audit stream has always been readable, over SSE, by a client willing to
    hold a connection open. That is the wrong shape for the things people
    actually want notified -- a Slack workspace, a pager, somebody's small
    script -- so "tell me when my job finishes" meant writing a poller, and a
    poller is how a control plane acquires a hundred clients asking the same
    question every five seconds.

    The secret is write-only. It is the key the receiver verifies the signature
    with; anything that reads a subscription back reads it redacted, because a
    signature anybody can forge proves nothing about who sent the request.
    """

    id: str = Field(max_length=96)
    org_id: str = Field(max_length=128)
    project_id: str = Field(max_length=128)
    url: str = Field(min_length=1, max_length=2048)
    #: Key for the HMAC every delivery carries. Never returned by a read.
    secret: str = Field(min_length=16, max_length=256)
    #: Entity types and actions this subscription wants. Empty means every one,
    #: which is the useful default: a filter that must be written before
    #: anything arrives is a filter most people get wrong once and never fix.
    entity_types: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    enabled: bool = True
    #: The audit sequence this subscription has been delivered through. Durable
    #: so a restarted plane resumes rather than replaying a week of history into
    #: somebody's incident channel.
    delivered_sequence: int = Field(default=0, ge=0)
    consecutive_failures: int = Field(default=0, ge=0)
    #: Why delivery stopped, when it did. A subscription that silently went
    #: quiet is indistinguishable from a fleet with nothing happening in it.
    disabled_reason: str | None = Field(default=None, max_length=512)
    created_by: str = Field(max_length=256)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_webhook(self) -> WebhookRecord:
        _validate_string(self.id, field_name="webhook id")
        _validate_string(self.url, field_name="webhook url")
        for name in (*self.entity_types, *self.actions):
            _validate_string(name, field_name="webhook filter")
        if self.enabled and self.disabled_reason is not None:
            raise ValueError("an enabled webhook has no disabled reason")
        return self

    def redacted(self) -> WebhookRecord:
        """Return this subscription with its signing key removed."""

        return self.model_copy(update={"secret": "*" * 16})


class ArtifactCollection(EyvanModel):
    """What one collection pass reclaimed, or would reclaim in a dry run."""

    #: URIs whose bytes were removed, or would be. Named rather than counted: an
    #: operator approving a first real run needs to see *which* objects, and a
    #: count cannot be checked against expectations.
    collected_uris: tuple[str, ...] = ()
    reclaimed_bytes: int = Field(default=0, ge=0)
    #: Indexed artifacts examined and kept, by reason. The kept count is what makes
    #: a surprisingly small collection diagnosable without re-deriving the index.
    held: int = Field(default=0, ge=0)
    referenced: int = Field(default=0, ge=0)
    within_grace: int = Field(default=0, ge=0)
    #: True when the bounded scan could not examine the complete active/claimed set.
    capped: bool = False
    #: True when nothing was deleted and the pass only reported.
    dry_run: bool = True


class ArtifactUsage(EyvanModel):
    """Indexed artifact bytes in one scope, and the ceiling they are measured against."""

    org_id: str
    project_id: str | None = None
    artifact_count: int = Field(ge=0)
    stored_bytes: int = Field(ge=0)
    #: ``None`` when no quota applies, which is not the same as a quota of zero.
    quota_bytes: int | None = Field(default=None, ge=0)


class ManagedArtifactState(StrEnum):
    """Durable lifecycle of one scoped managed object."""

    PENDING_UPLOAD = "pending_upload"
    ACTIVE = "active"
    DELETING = "deleting"
    DELETED = "deleted"
    QUARANTINED = "quarantined"


class ArtifactInventoryStatus(StrEnum):
    """Relationship between one scoped index row and one physical object."""

    MATCHED = "matched"
    PHYSICAL_ONLY = "physical_only"
    INDEX_ONLY = "index_only"
    SIZE_MISMATCH = "size_mismatch"
    VERSION_MISMATCH = "version_mismatch"


class ArtifactInventoryEntry(EyvanModel):
    """One non-destructive comparison at an explicit tenant-scoped URI."""

    uri: str = Field(min_length=1, max_length=512)
    status: ArtifactInventoryStatus
    indexed_size_bytes: int | None = Field(default=None, ge=0)
    observed_size_bytes: int | None = Field(default=None, ge=0)
    lifecycle_state: ManagedArtifactState | None = None
    indexed_object_version: str | None = Field(default=None, min_length=1, max_length=256)
    #: GCS generation or local file identity captured with observed_size_bytes.
    observed_object_version: str | None = Field(default=None, min_length=1, max_length=256)
    #: True only when an explicitly acknowledged exact-version reconciliation
    #: activated a physical-only or quarantined row. Physical bytes are untouched.
    reindexed: bool = False


class ArtifactInventoryReport(EyvanModel):
    """One bounded, scope-explicit inventory page."""

    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    entries: tuple[ArtifactInventoryEntry, ...] = ()
    next_cursor: str | None = Field(default=None, max_length=512)
    dry_run: bool = True
    reindexed_count: int = Field(default=0, ge=0)


class ArtifactHold(EyvanModel):
    """An explicit claim that keeps one artifact out of any future collection.

    References answer "is something still going to run this". A hold answers a
    question references cannot: "a human still needs this" -- a callable under
    investigation, a result cited outside Eyvan, a workspace pinned for a repro.
    Without it the only way to protect such an object is to keep a workload alive
    around it, which spends capacity to express a bookkeeping fact.

    One hold per artifact, deliberately. A hold *set* would need its own identity,
    ownership and expiry semantics, and the failure it prevents -- two people each
    believing the other's hold protects the object -- is not one this package has
    seen. The single hold names its owner and reason, so a second person finds a
    stated claim to argue with rather than an anonymous flag.
    """

    reason: str = Field(min_length=1, max_length=512)
    held_by: str = Field(min_length=1, max_length=128)
    created_at: datetime = Field(default_factory=utcnow)
    #: ``None`` is an indefinite hold. Permitted because the honest lifetime of
    #: "cited in a paper" is indefinite, and an expiry invented to avoid saying so
    #: would delete the object on a date nobody chose.
    expires_at: datetime | None = None

    def active_at(self, moment: datetime) -> bool:
        """Return whether this hold is in force at ``moment``."""

        return self.expires_at is None or moment < self.expires_at


class ManagedArtifactRecord(EyvanModel):
    """Fenced ledger entry for one tenant-scoped physical object.

    ``record_generation`` fences stale database actors while ``object_version``
    fences provider mutations.  Rows are retained as zero-byte tombstones after
    deletion; a later upload advances the generation instead of resurrecting an
    old claim in place.
    """

    uri: str = Field(max_length=512)
    kind: str = Field(max_length=32)
    org_id: str = Field(max_length=128)
    project_id: str = Field(max_length=128)
    state: ManagedArtifactState = ManagedArtifactState.QUARANTINED
    record_generation: int = Field(default=1, ge=1)
    size_bytes: int = Field(ge=0)
    sha256: str | None = Field(default=None, max_length=64)
    object_version: str | None = Field(default=None, max_length=256)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    upload_started_at: datetime | None = None
    activated_at: datetime | None = None
    deletion_claim_id: str | None = Field(default=None, max_length=96)
    deletion_started_at: datetime | None = None
    deleted_at: datetime | None = None
    quarantine_reason: str | None = Field(default="legacy object version has not been verified", max_length=512)
    last_accessed_at: datetime | None = None
    hold: ArtifactHold | None = None
    superseded_at: datetime | None = None
    superseded_by_uri: str | None = Field(default=None, max_length=512)

    @property
    def charged_bytes(self) -> int:
        """Return bytes conservatively charged to quota in this state."""

        return 0 if self.state is ManagedArtifactState.DELETED else self.size_bytes

    def collectable_at(self, moment: datetime, *, has_live_reference: bool) -> bool:
        """Return whether an active object is currently eligible for a claim."""

        if self.state is not ManagedArtifactState.ACTIVE or has_live_reference:
            return False
        return self.hold is None or not self.hold.active_at(moment)

    @model_validator(mode="after")
    def validate_artifact(self) -> ManagedArtifactRecord:
        _validate_string(self.uri, field_name="managed artifact uri")
        _validate_string(self.kind, field_name="managed artifact kind")
        _validate_string(self.org_id, field_name="managed artifact organization id")
        _validate_string(self.project_id, field_name="managed artifact project id")
        _validate_string(self.sha256, field_name="managed artifact digest", optional=True)
        _validate_string(self.object_version, field_name="managed artifact object version", optional=True)
        _validate_string(self.deletion_claim_id, field_name="managed artifact deletion claim", optional=True)
        _validate_string(self.quarantine_reason, field_name="managed artifact quarantine reason", optional=True)

        if self.sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("managed artifact digest must be a lowercase SHA-256")
        if (
            self.state
            in {
                ManagedArtifactState.PENDING_UPLOAD,
                ManagedArtifactState.ACTIVE,
                ManagedArtifactState.DELETING,
            }
            and self.sha256 is None
        ):
            raise ValueError("managed artifact lifecycle states require an expected SHA-256")
        if self.state is ManagedArtifactState.PENDING_UPLOAD:
            if self.upload_started_at is None or self.object_version is not None:
                raise ValueError("pending managed artifacts require an upload start and no object version")
        elif self.state is ManagedArtifactState.ACTIVE:
            if self.object_version is None or self.activated_at is None:
                raise ValueError("active managed artifacts require an object version and activation time")
        elif self.state is ManagedArtifactState.DELETING:
            if self.object_version is None or self.deletion_claim_id is None or self.deletion_started_at is None:
                raise ValueError("deleting managed artifacts require a versioned durable claim")
        elif self.state is ManagedArtifactState.DELETED:
            if self.size_bytes != 0 or self.deleted_at is None:
                raise ValueError("deleted managed artifacts must be zero-byte tombstones with a deletion time")
        elif self.state is ManagedArtifactState.QUARANTINED and self.quarantine_reason is None:
            raise ValueError("quarantined managed artifacts require a reason")

        if self.state is not ManagedArtifactState.QUARANTINED and self.quarantine_reason is not None:
            raise ValueError("only quarantined managed artifacts may carry a quarantine reason")
        if self.state is not ManagedArtifactState.DELETING and (
            self.deletion_claim_id is not None or self.deletion_started_at is not None
        ):
            raise ValueError("only deleting managed artifacts may retain a deletion claim")
        if self.state is not ManagedArtifactState.DELETED and self.deleted_at is not None:
            raise ValueError("only deleted managed artifacts may carry a deletion time")
        return self


class ArtifactReferenceRecord(EyvanModel):
    """One workload's claim on one artifact: the reverse index.

    Nothing could previously answer "which workloads use this callable?", so
    "unreferenced" was not a computable property and no collector could be
    written safely. A reference is created when a workload that names an
    artifact is stored, and released when that workload reaches a terminal
    state -- but the row is *kept*, because the history of who referenced what is
    what makes a later deletion auditable.
    """

    #: Deterministic display identity over the complete scope. Durable uniqueness
    #: is the four typed fields below, so correctness does not depend on this
    #: truncated digest never colliding.
    id: str = Field(max_length=200)
    uri: str = Field(max_length=512)
    workload_id: str = Field(max_length=96)
    org_id: str = Field(max_length=128)
    project_id: str = Field(max_length=128)
    created_at: datetime = Field(default_factory=utcnow)
    #: Set when the referencing workload became terminal. A URI with no
    #: unreleased reference is a *candidate* for collection, never a safety
    #: proof. Serialized, version-conditional deletion is a later migration.
    released_at: datetime | None = None

    @model_validator(mode="after")
    def validate_reference(self) -> ArtifactReferenceRecord:
        _validate_string(self.id, field_name="artifact reference id")
        _validate_string(self.uri, field_name="artifact reference uri")
        _validate_string(self.workload_id, field_name="artifact reference workload id")
        _validate_string(self.org_id, field_name="artifact reference organization id")
        _validate_string(self.project_id, field_name="artifact reference project id")
        return self


def artifact_reference_id(org_id: str, project_id: str, uri: str, workload_id: str) -> str:
    """Return the deterministic display id for one scoped artifact reference.

    Derived rather than generated so that a replayed submission produces the
    same value. Persistence is keyed by ``(org_id, project_id, uri,
    workload_id)`` itself, so a digest collision cannot merge two references.
    """

    digest = hashlib.sha256(f"{org_id}\n{project_id}\n{uri}\n{workload_id}".encode()).hexdigest()
    return f"ref-{digest[:40]}"


class IdempotencyRecord(EyvanModel):
    """Durable response pointer for one scoped mutating request."""

    scope: str = Field(max_length=256)
    operation: str = Field(max_length=96)
    key: str = Field(max_length=200)
    fingerprint: str
    resource_type: str
    resource_id: str
    created_at: datetime = Field(default_factory=utcnow)


class OutboxRecord(EyvanModel):
    """A fenced, at-least-once backend command committed with domain state."""

    id: str = Field(max_length=96)
    topic: str
    aggregate_type: str
    aggregate_id: str
    idempotency_key: str
    generation: int = Field(ge=0)
    payload: JsonDocument = Field(default_factory=dict)
    state: OutboxState = OutboxState.PENDING
    attempts: int = Field(default=0, ge=0)
    available_at: datetime = Field(default_factory=utcnow)
    claimed_by: str | None = None
    claimed_until: datetime | None = None
    last_error: str | None = None
    failure_code: CapacityProviderFailureCode | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    correlation_id: str | None = Field(default=None, pattern=CORRELATION_ID_WIRE_PATTERN)
    traceparent: str | None = Field(default=None, max_length=55, pattern=r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")

    @model_serializer(mode="wrap")
    def _wire_document(self, handler):
        document = handler(self)
        if self.correlation_id is None:
            document.pop("correlation_id", None)
        if self.traceparent is None:
            document.pop("traceparent", None)
        return document


class WorkloadProgressPhase(StrEnum):
    """Operator-facing phase for one workload while capacity is acquired and used."""

    WAITING_FOR_AUTOSCALER = "waiting_for_autoscaler"
    REQUESTING_CAPACITY = "requesting_capacity"
    WAITING_FOR_CAPACITY = "waiting_for_capacity"
    PROVISIONING = "provisioning"
    CAPACITY_ALLOCATED = "capacity_allocated"
    NATIVE_WORKERS_CONNECTING = "native_workers_connecting"
    WAITING_FOR_SCHEDULER = "waiting_for_scheduler"
    WAITING_FOR_DISPATCH = "waiting_for_dispatch"
    READY_FOR_DISPATCH = "ready_for_dispatch"
    PROVIDER_FAILURE = "provider_failure"
    QUEUE_ADMISSION_DENIED = "queue_admission_denied"
    QUOTA_BLOCKED = "quota_blocked"
    PREEMPTED = "preempted"
    DISPATCHING = "dispatching"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    CANCELLING = "cancelling"
    TERMINAL = "terminal"


class WorkloadCapacityRequest(EyvanModel):
    """Exact capacity request after gang-replica and physical-cell expansion."""

    cells: int = Field(ge=1)
    workers_per_cell: int = Field(ge=1)
    resources_per_cell: ResourceVector
    total_resources: ResourceVector


class WorkloadPoolProgress(EyvanModel):
    """Stable identity of the exact capacity pool pinned by a workload."""

    id: str
    fleet_name: str
    node_type: str
    provider: str
    provider_project: str | None = None
    region: str
    zone: str | None
    accelerator_type: str | None
    capacity_class: CanonicalCapacityClass


class WorkloadCapacityPhaseProgress(EyvanModel):
    """Current cells grouped by their durable Eyvan and provider phases."""

    state: CellState
    provider_phase: CapacityStatus | None
    cells: int = Field(ge=0)
    workers: int = Field(ge=0)


class WorkloadCapacitySummary(EyvanModel):
    """Exact-pool capacity counts for a workload's requested physical shape."""

    active_cells: int = Field(ge=0)
    active_workers: int = Field(ge=0)
    ready_cells: int = Field(ge=0)
    ready_workers: int = Field(ge=0)
    requesting_cells: int = Field(default=0, ge=0)
    requesting_workers: int = Field(default=0, ge=0)
    waiting_cells: int = Field(ge=0)
    waiting_workers: int = Field(ge=0)
    provisioning_cells: int = Field(ge=0)
    provisioning_workers: int = Field(ge=0)
    lost_cells: int = Field(ge=0)
    lost_workers: int = Field(ge=0)
    observed_tpu_vms: int | None = Field(default=None, ge=1)
    phases: tuple[WorkloadCapacityPhaseProgress, ...] = ()


class CapacityRetryProgress(EyvanModel):
    """Durable state of the newest relevant capacity-apply command."""

    state: OutboxState
    attempts: int = Field(ge=0)
    next_attempt_at: datetime | None
    diagnostic: str | None = Field(default=None, max_length=2000)
    failure_code: CapacityProviderFailureCode | None = None


class UserProgressReport(EyvanModel):
    """One rank's latest self-reported progress.

    **Latest per rank, no history.** Upserted on ``(workload_id, attempt, rank)``,
    so it is idempotent by overwrite, needs no idempotency key, and cannot grow
    without bound. Unbounded user-metric history in the control-plane database is
    the thing to refuse now rather than after it has filled a disk.
    """

    rank: int = Field(ge=0)
    attempt: int = Field(ge=1)
    step: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=0)
    message: str = Field(default="", max_length=512)
    #: Free-form user metrics. Reported and displayed; deliberately **not**
    #: exported as Prometheus series, because unbounded label cardinality coming
    #: from user code is a self-inflicted outage.
    metrics: dict[str, float] = Field(default_factory=dict)
    reported_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_report(self) -> UserProgressReport:
        if self.total is not None and self.step is not None and self.step > self.total:
            raise ValueError("progress step must not exceed total")
        _validate_string_map({name: str(value) for name, value in self.metrics.items()}, field_name="metrics")
        return self

    @property
    def fraction(self) -> float | None:
        """Return completion as a fraction, or ``None`` when it is not known.

        ``None`` rather than zero: a job that reported a step but no total knows
        how far it has come and not how far it has to go, and rendering that as
        0% is worse than rendering nothing.
        """

        if self.step is None or not self.total:
            return None
        return min(1.0, self.step / self.total)


class WorkloadProgress(EyvanModel):
    """Read-only autoscaling progress for one submitted workload."""

    workload_id: str
    workload_state: WorkloadState
    phase: WorkloadProgressPhase
    scheduling_reason: str | None
    failure_reason: str | None
    autoscale_block_reason: AutoscaleBlockReason | None = None
    queue_admission_denial: GcpQueueAdmissionDenialTarget | None = None
    request: WorkloadCapacityRequest
    pool: WorkloadPoolProgress | None
    capacity: WorkloadCapacitySummary
    retry: CapacityRetryProgress | None
    #: What the workload itself reported, newest per rank. Exposed on the route
    #: the dashboard and ``client/callables.py`` already poll, so a training loop
    #: reporting its step needs no new polling loop anywhere.
    user_progress: tuple[UserProgressReport, ...] = ()


#: Longest logical credential name a rotation ledger row may carry. Sized to the
#: identifier columns elsewhere in this schema rather than to today's two names,
#: so adding a third credential never needs a migration.
MAX_CREDENTIAL_NAME_LENGTH = 96


class CredentialRotationRecord(EyvanModel):
    """One rotation of one signing credential, and how it ended.

    Durable in a table of its own, and additive for the same reason
    :class:`WorkloadAttemptTopology` is: a field on an existing persisted record
    would be decoded with ``extra='forbid'`` by a rolled-back replica, and the
    first rotation would make that record unreadable. A replica that predates
    this table never selects from it and simply does not rotate.

    **No material, and nothing derived from material.** Not the new key, not the
    old one, and deliberately not a digest either: a digest of a credential is an
    offline cracking target that an audit reader, a backup, and a support bundle
    would all carry. A generation number is enough to say which material is in
    effect, and it is the only thing here that refers to it.

    The row is what lets two replicas agree, so ``previous_not_after`` is the
    definite point at which the superseded generation stops verifying **fleet
    wide** rather than per process.
    """

    id: str = Field(max_length=32)
    #: Logical credential, not a key: which of a deployment's credentials this
    #: rotation is about.
    credential: str = Field(max_length=MAX_CREDENTIAL_NAME_LENGTH)
    #: The generation this rotation produces. Generation 1 is the material a
    #: deployment started with, which is not a rotation, so a row is always >= 2.
    generation: int = Field(ge=2)
    #: Held by whoever began the rotation and required to resolve it. Two
    #: operators reacting to one alert produce one rotation and one refusal.
    fence: str = Field(max_length=64)
    state: CredentialRotationState
    #: Seconds the superseded generation keeps verifying after a commit. Chosen
    #: when the rotation begins so a commit cannot silently widen it.
    overlap_s: float = Field(ge=0)
    started_at: datetime = Field(default_factory=utcnow)
    #: When the outcome was recorded -- including when it was recorded as
    #: unknown. ``None`` only while the rotation is still in flight.
    resolved_at: datetime | None = None
    #: Absolute instant the previous generation stops verifying. Set only on a
    #: commit: an abandoned rotation retires nothing, and an inconclusive one
    #: does not know what it retired.
    previous_not_after: datetime | None = None
    #: Operator-safe text. Never a provider message, never a path, never a value.
    detail: str = Field(default="", max_length=512)

    @model_validator(mode="after")
    def validate_rotation(self) -> CredentialRotationRecord:
        _validate_string(self.id, field_name="credential rotation id")
        _validate_string(self.credential, field_name="credential rotation credential")
        _validate_string(self.fence, field_name="credential rotation fence")
        if self.state is CredentialRotationState.PENDING:
            if self.resolved_at is not None or self.previous_not_after is not None:
                raise ValueError("an in-flight credential rotation has no outcome yet")
            return self
        if self.resolved_at is None:
            raise ValueError("a resolved credential rotation must record when its outcome was observed")
        if self.state is CredentialRotationState.COMMITTED:
            if self.previous_not_after is None:
                raise ValueError("a committed credential rotation must retire the previous generation at a point")
            return self
        if self.previous_not_after is not None:
            # Neither of the two remaining states retires anything: an abandoned
            # rotation changed nothing, and an inconclusive one cannot claim to
            # know what it changed.
            raise ValueError("only a committed credential rotation retires the previous generation")
        return self

    @property
    def is_in_flight(self) -> bool:
        """Return whether this rotation is still waiting for an outcome."""

        return self.state is CredentialRotationState.PENDING

    @property
    def blocks_further_rotation(self) -> bool:
        """Return whether this row prevents another rotation from beginning.

        In flight, because two rotations must not interleave; inconclusive,
        because stacking a second unknown on an unresolved one is how a
        deployment ends up unable to say which material is live.
        """

        return self.state in {CredentialRotationState.PENDING, CredentialRotationState.INCONCLUSIVE}


#: Longest named-asset string one durable stage row may key on.
#:
#: ``WorkloadDefinition.named_assets`` is an opaque list of strings -- a dataset
#: URI, a checkpoint path, an image reference -- and nothing bounded them,
#: because until now nothing keyed durable state on one. A staged asset is a
#: primary key in every backend, so the bound is the width the column has to
#: reserve. Deliberately generous next to real object-store URIs and still far
#: below any index key limit.
MAX_CELL_STAGE_ASSET_LENGTH = 512


class CellStageState(StrEnum):
    """What this control plane can prove about one asset's copy on one cell.

    Three of the five are the same statement -- *presence is unproven* -- kept
    apart because the operator response differs: ``STAGING`` and ``EVICTING`` say
    an action is running, ``UNKNOWN`` says one ended without an answer, and
    ``BLOCKED`` says this control plane has stopped asking. Only ``STAGED``
    claims the bytes are there, and only ``STAGED`` may be scored as a cache hit.

    There is no state for *proven absent*: that is the absence of a row. A
    confirmed eviction deletes the record rather than tombstoning it, so the
    table stays bounded by what is actually on disk somewhere.
    """

    #: A staging action is in flight. Bytes may or may not be on disk.
    STAGING = "staging"
    #: An executor reported the exact bytes present at this cell generation.
    STAGED = "staged"
    #: An action ended without an answer. Not absent -- unproven.
    UNKNOWN = "unknown"
    #: An eviction is in flight. Bytes may or may not still be on disk.
    EVICTING = "evicting"
    #: No further automatic action will be taken. Presence is still unproven.
    BLOCKED = "blocked"


#: States in which the asset is not proven present. Every one of them charges
#: the cell budget, because a disk does not care what this control plane can
#: prove: bytes that *might* be there occupy space that might not be free.
CELL_STAGE_UNPROVEN_STATES = frozenset(
    {
        CellStageState.STAGING,
        CellStageState.UNKNOWN,
        CellStageState.EVICTING,
        CellStageState.BLOCKED,
    }
)


class CellStageRecord(EyvanModel):
    """One named asset's managed copy on one execution cell.

    Durable and additive, in its own table. The alternative -- a field on
    ``CellRecord`` -- is not available: the persisted cell document is decoded
    with ``extra='forbid'``, so the first write would make every cell unreadable
    to a rolled-back replica, and a control plane that cannot read its cells
    schedules nothing.

    Distinct from ``CellRecord.materialized_assets``, which is an agent's
    unaccounted claim carried in free-form observation metadata. That claim has
    no size, no digest, no generation and no lifecycle, so nothing can evict
    against it or bound it. This record is the managed half: every field here is
    either observed by an executor or decided by this control plane, and the two
    are never mixed -- ``confirmed_at`` is set only by an observation.
    """

    cell_id: str = Field(max_length=96)
    asset: str = Field(max_length=MAX_CELL_STAGE_ASSET_LENGTH)
    #: The cell generation this row describes. A generation bump is a different
    #: machine, so a row naming an older one describes a disk that no longer
    #: exists: it is never a hit and never charged, whatever its state says.
    cell_generation: int = Field(ge=1)
    state: CellStageState
    #: Bytes this stage is charged against its cell's budget -- the ceiling the
    #: action was admitted under while presence is unproven, and the observed
    #: size once an executor has measured it. One number so the budget is
    #: comparable across states; the honest difference is ``observed_bytes``.
    reserved_bytes: int = Field(ge=0)
    #: What an executor actually measured, or ``None`` when nobody has. Never
    #: inferred from the ceiling: an unmeasured stage reports unmeasured.
    observed_bytes: int | None = Field(default=None, ge=0)
    #: Content digest as reported by the executor. ``None`` means the executor
    #: does not compute one for this asset, which is ordinary for a directory
    #: tree -- presence is still proven, content is simply not verified.
    digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    #: When an executor last proved the bytes present. Set only with
    #: ``STAGED`` and cleared with every other state, so "confirmed" cannot
    #: survive the observation that stopped confirming it.
    confirmed_at: datetime | None = None
    #: When a workload declaring this asset last held a lease on this cell.
    #: The eviction policy's clock; not evidence of presence.
    last_used_at: datetime | None = None
    #: Completed staging attempts charged against the attempt budget. Bounds the
    #: retry of a stage that keeps ending without an answer, so an asset that can
    #: never be fetched stops consuming a leader phase forever.
    attempts: int = Field(default=0, ge=0)
    #: Why the last action ended as it did. Free text, and the only place an
    #: operator can read the executor's own words.
    detail: str | None = Field(default=None, max_length=2048)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_stage(self) -> CellStageRecord:
        _validate_string(self.cell_id, field_name="cell stage cell id")
        _validate_string(self.asset, field_name="cell stage asset")
        if (self.state is CellStageState.STAGED) != (self.confirmed_at is not None):
            raise ValueError("a cell stage is confirmed exactly when it is staged")
        if self.state is CellStageState.STAGED and self.observed_bytes is None:
            raise ValueError("a staged cell stage must carry the size an executor observed")
        if self.state is CellStageState.STAGED and self.observed_bytes != self.reserved_bytes:
            raise ValueError("a staged cell stage must be charged exactly the bytes an executor observed")
        return self

    @property
    def presence_is_unproven(self) -> bool:
        """Return whether nothing here claims the bytes are on a disk.

        The half of the design that is easy to get backwards: an unproven stage
        is never a cache hit and always a budget charge. Reading it off one named
        set rather than off an inline comparison is what makes adding a state a
        decision -- a new member either charges without hitting or it does not,
        and the set is where that is written down.
        """

        return self.state in CELL_STAGE_UNPROVEN_STATES

    def is_confirmed_on(self, cell: CellRecord) -> bool:
        """Return whether this row proves the asset is on *this* cell right now.

        Both halves matter. A row that is not ``STAGED`` never proved presence,
        and a row from an older generation proved it about a machine that has
        since been replaced. Either way the answer is "unknown", and unknown is
        never a cache hit.
        """

        return not self.presence_is_unproven and self.cell_id == cell.id and self.cell_generation == cell.generation


class ExpansionMode(StrEnum):
    """How one fan-out expansion was admitted, as reported to its submitter.

    Durable and wire-visible, so it lives beside the record that carries it
    rather than in the service that computes it.
    """

    #: Every child competes for capacity from the moment it is created.
    IMMEDIATE = "immediate"
    #: Children compete in bounded waves; the rest are created, durably owed
    #: execution, and invisible to the scheduler until an earlier wave drains.
    WAVES = "waves"


class ExpansionAdmissionState(StrEnum):
    """Whether an expansion ledger still owes any child an admission."""

    ADMITTING = "admitting"
    #: Every child this ledger governs has been released, or can never be:
    #: an absorbing state, so a settled expansion stops being scanned.
    COMPLETE = "complete"


class ExpansionAdmissionRecord(EyvanModel):
    """The durable admission plan one fan-out expansion was accepted under.

    Additive and in its own table on purpose. The alternative -- fields on the
    coordinator workload -- is not available: the persisted workload document is
    decoded with ``extra='forbid'``, so the first write would make every workload
    unreadable to a rolled-back replica, and a control plane that cannot read its
    workloads schedules nothing.

    Durable rather than in-memory for the reason waves exist at all: a leader
    that dies between two waves must resume the *same* plan. Recomputing it from
    configuration would silently re-plan a running expansion whenever an operator
    changed a setting, and recomputing it from live quota would make an
    expansion's own progress change its remaining plan.

    ``released_children`` is a ledger, never the decision input. Which child is
    released next is read from the children themselves, so a leader that crashed
    after flipping a child and before committing the counter releases nothing
    twice.
    """

    #: Derived from the coordinator id, so a replayed submission rebinds the same
    #: row instead of opening a second plan for one expansion.
    id: str = Field(max_length=32)
    #: The coordinator whose children this plan governs.
    workload_id: str = Field(max_length=96)
    org_id: str = Field(max_length=128)
    project_id: str = Field(max_length=128)
    kind: CoordinationKind
    total_children: int = Field(ge=1)
    #: How many children may compete at once. Equal to ``total_children`` when
    #: the expansion was admitted immediately.
    wave_size: int = Field(ge=1)
    #: How many children this ledger has made schedulable so far, including the
    #: first wave admitted inside the submitting transaction.
    released_children: int = Field(default=0, ge=0)
    #: The ceiling this expansion was held to, kept so an operator can see what
    #: an already-admitted expansion was judged against after the setting moves.
    ceiling: int = Field(ge=1)
    mode: ExpansionMode
    state: ExpansionAdmissionState = ExpansionAdmissionState.ADMITTING
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_admission(self) -> ExpansionAdmissionRecord:
        _validate_string(self.id, field_name="expansion admission id")
        _validate_string(self.workload_id, field_name="expansion admission workload id")
        _validate_string(self.org_id, field_name="expansion admission organization id")
        _validate_string(self.project_id, field_name="expansion admission project id")
        if self.released_children > self.total_children:
            raise ValueError("expansion admission cannot release more children than it created")
        if self.wave_size > self.total_children:
            raise ValueError("expansion admission wave cannot exceed the children it governs")
        if self.mode is ExpansionMode.IMMEDIATE and self.wave_size != self.total_children:
            raise ValueError("an immediately admitted expansion has one wave holding every child")
        return self

    @property
    def deferred_children(self) -> int:
        """Children created and owed execution that this ledger has not released."""

        return self.total_children - self.released_children

    def describe(self) -> str:
        """Return the one-line, operator-readable summary of this admission."""

        if self.mode is ExpansionMode.IMMEDIATE:
            return f"admitted {self.total_children} children immediately (ceiling {self.ceiling})"
        return (
            f"admitted {self.total_children} children in waves of {self.wave_size}: "
            f"{self.released_children} released, {self.deferred_children} waiting "
            f"(ceiling {self.ceiling})"
        )


def expansion_admission_id(workload_id: str) -> str:
    """Return the stable ledger id for one coordinator.

    Derived rather than random so a replayed submission finds the row it already
    wrote, and short enough to be a 32-character key in every backend.
    """

    return f"exp_{hashlib.sha256(workload_id.encode()).hexdigest()[:28]}"


# --------------------------------------------------------------------------- managed services
#
# Everything below describes *operating* a service -- proving a replica answers,
# replacing one whose claim on its host is gone, and changing how many there
# are. All of it lives in tables of its own rather than in fields on
# ``WorkloadRecord``: that document is decoded with ``extra='forbid'``, so the
# first write of a new field makes every workload unreadable to a rolled-back
# replica, and a control plane that cannot read its workloads schedules nothing.


class ServiceProbeOutcome(StrEnum):
    """What one completed health probe established, including "nothing"."""

    #: The endpoint answered and said it was serving.
    SUCCESS = "success"
    #: The endpoint answered and said it was not. A definite negative.
    FAILURE = "failure"
    #: The probe did not complete: a timeout, a refused connection, a DNS
    #: failure, an unreadable body. Deliberately *not* a failure -- the control
    #: plane learned nothing about the process, only about its own reachability
    #: of it, and tearing a replica down on that is how a network blip becomes
    #: an outage.
    UNKNOWN = "unknown"


class ServiceReplicaHealthState(StrEnum):
    """The health verdict derived from a replica's durable probe counters."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ServiceReplicaHealthRecord(EyvanModel):
    """Durable probe history for one replica of one service.

    Keyed by ``(service_id, replica_id)`` and carrying the *replica generation*
    the counters were gathered at. A requeued replica is a different process on
    a different host, so its predecessor's failures say nothing about it; the
    generation is what makes "these counters are stale" representable instead of
    silently carrying a verdict across a restart.
    """

    service_id: str = Field(max_length=96)
    replica_id: str = Field(max_length=96)
    replica_generation: int = Field(ge=1)
    #: The address that was probed, recorded so an operator can tell a replica
    #: that moved from one that stopped answering.
    endpoint: str = Field(default="", max_length=512)
    last_outcome: ServiceProbeOutcome = ServiceProbeOutcome.UNKNOWN
    last_detail: str = Field(default="", max_length=512)
    #: Definite failures since the last definite success. An unknown probe
    #: neither raises this nor clears it: raising it would record "I could not
    #: reach this" as "this said no", and clearing it would grant a recovery the
    #: endpoint never demonstrated. Only a probe that completed moves it.
    consecutive_failures: int = Field(default=0, ge=0)
    #: Probes in a row that established nothing. Reported, never acted on.
    consecutive_unknowns: int = Field(default=0, ge=0)
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    #: When any probe last completed, whatever it established.
    observed_at: datetime = Field(default_factory=utcnow)
    #: Utilization the replica reported about itself, in ``[0, 1]``. ``None``
    #: means the replica did not report one, which is the only honest default:
    #: the control plane cannot see inside a serving process, and inventing a
    #: number here would be inventing the input an autoscaler multiplies.
    observed_load: float | None = Field(default=None, ge=0.0, le=1.0)
    load_observed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_health(self) -> ServiceReplicaHealthRecord:
        _validate_string(self.service_id, field_name="service health service id")
        _validate_string(self.replica_id, field_name="service health replica id")
        if (self.observed_load is None) != (self.load_observed_at is None):
            raise ValueError("a reported service replica load requires the moment it was reported")
        if self.last_outcome is ServiceProbeOutcome.SUCCESS and self.last_success_at is None:
            raise ValueError("a successful service probe requires its success timestamp")
        if self.last_outcome is ServiceProbeOutcome.FAILURE and self.last_failure_at is None:
            raise ValueError("a failed service probe requires its failure timestamp")
        return self

    @property
    def key(self) -> tuple[str, str]:
        """Return the complete durable identity of one replica's health."""

        return (self.service_id, self.replica_id)


class ServiceDecisionKind(StrEnum):
    """What a recorded service decision actually did."""

    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    REPLACE_REPLICA = "replace_replica"


class ServiceObservation(EyvanModel):
    """The durable facts one service decision was taken from.

    Stored beside the decision rather than recomputed later. A decision explained
    by today's state is not explained at all: by the time anyone reads it, the
    replicas it counted are gone.
    """

    live_replicas: int = Field(ge=0)
    healthy_replicas: int = Field(ge=0)
    unhealthy_replicas: int = Field(ge=0)
    #: Replicas whose health could not be established. Carried separately from
    #: unhealthy on purpose: collapsing the two is the defect this whole record
    #: set exists to make impossible.
    unknown_replicas: int = Field(ge=0)
    #: Replicas that reported a load, and the mean of what they reported.
    loaded_replicas: int = Field(ge=0)
    average_load: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Oldest load observation that was counted, so staleness is auditable.
    oldest_load_observed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_observation(self) -> ServiceObservation:
        if (self.average_load is None) != (self.loaded_replicas == 0):
            raise ValueError("an average service load requires at least one reporting replica")
        if self.loaded_replicas and self.oldest_load_observed_at is None:
            raise ValueError("counted service load observations require their oldest timestamp")
        return self


class ServiceDecisionRecord(EyvanModel):
    """One committed, explained decision about a service.

    Append-only per service and bounded per service, so the audit cannot become
    the table that fills the disk. The newest row is also the durable cooldown
    clock: a decision and the moment it may next be revisited are the same fact,
    and keeping them apart is how a cooldown ends up not surviving a restart.
    """

    service_id: str = Field(max_length=96)
    sequence: int = Field(ge=1)
    kind: ServiceDecisionKind
    #: Declared replica counts before and after. Equal for a replacement, which
    #: changes which replicas exist rather than how many.
    from_replicas: int = Field(ge=0)
    to_replicas: int = Field(ge=0)
    #: The replaced replica and the generation it was replaced at, for a
    #: replacement. ``None`` for a scaling decision.
    replica_id: str | None = Field(default=None, max_length=96)
    replica_generation: int | None = Field(default=None, ge=1)
    reason: str = Field(min_length=1, max_length=512)
    observed: ServiceObservation
    decided_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_decision(self) -> ServiceDecisionRecord:
        _validate_string(self.service_id, field_name="service decision service id")
        _validate_string(self.replica_id, field_name="service decision replica id", optional=True)
        if self.kind is ServiceDecisionKind.REPLACE_REPLICA:
            if self.replica_id is None or self.replica_generation is None:
                raise ValueError("a replacement decision must name the replica and the generation it was fenced at")
            if self.from_replicas != self.to_replicas:
                raise ValueError("a replacement decision does not change the declared replica count")
        else:
            if self.replica_id is not None or self.replica_generation is not None:
                raise ValueError("a scaling decision is about a count, not about one replica")
            if self.kind is ServiceDecisionKind.SCALE_OUT and self.to_replicas <= self.from_replicas:
                raise ValueError("a scale-out decision must raise the declared replica count")
            if self.kind is ServiceDecisionKind.SCALE_IN and self.to_replicas >= self.from_replicas:
                raise ValueError("a scale-in decision must lower the declared replica count")
        return self

    @property
    def key(self) -> tuple[str, int]:
        """Return the complete durable identity of one decision."""

        return (self.service_id, self.sequence)
