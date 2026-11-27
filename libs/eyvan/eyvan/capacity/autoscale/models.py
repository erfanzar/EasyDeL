# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed inputs, outputs, and stable document contracts for autoscale."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from eyvan._internal.compat import UTC, StrEnum
from eyvan._internal.json_types import JsonObject, JsonValue
from eyvan.capacity.fleet_config import EyvanFleetConfig, validate_fleet_name
from eyvan.capacity.providers.gcp.tpu_types import parse_tpu_accelerator_type
from eyvan.records import CapacityClass

_ZONE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?-[a-z][a-z0-9]*$")
_GCP_PROJECT_ID = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")

DEFAULT_AUTOSCALE_CACHE_ROOT = Path("~/.eyvan/autoscale")
DEFAULT_AUTOSCALE_DEFAULTS_PATH = DEFAULT_AUTOSCALE_CACHE_ROOT / "defaults.yaml"


class AutoscaleSkipReason(StrEnum):
    """Canonical reasons why a discovered TPU type is not emitted as a pool."""

    COMPUTE_BACKED_TPU_VM = "compute-backed TPU VM is excluded from queued-resource autoscale"
    CACHED_QUEUE_ADMISSION_DENIAL = "a previous GCP queue-admission denial blocks this exact Spot worker"
    DOCUMENTED_UNSUPPORTED_TPU_ZONE = "Google documents this TPU generation as unsupported in this zone"
    UNSUPPORTED_ACCELERATOR_TYPE = "Eyvan cannot derive a typed queued-resource worker shape"
    LIVE_SPOT_QUOTA_USAGE_UNAVAILABLE = "current Spot quota usage is unavailable"
    SHARED_SPOT_QUOTA_TOO_SMALL = "available shared Spot quota cannot fit one complete worker"
    ZONE_CONFIGURATION_INVALID = "selected TPU types cannot form a valid autoscale zone configuration"


def autoscale_fleet_name(fleet_prefix: str, *parts: str) -> str:
    """Return a stable fleet name from user-facing autoscale components."""

    base = "-".join((fleet_prefix, *parts)).lower().replace("_", "-")
    if len(base) <= 63:
        candidate = base
    else:
        suffix = f"-{hashlib.sha256(base.encode()).hexdigest()[:8]}"
        candidate = f"{base[: 63 - len(suffix)].rstrip('-')}{suffix}"
    return validate_fleet_name(candidate)


def _optional_trimmed(value: str | None, field_name: str) -> str | None:
    """Return a normalized optional string or raise an actionable validation error."""

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string when set")
    return value


def _document_optional_string(value: JsonValue, field_name: str) -> str | None:
    """Read one optional string from an already-decoded defaults document."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty trimmed string when set")
    return value


def _document_required_string(
    value: JsonValue,
    field_name: str,
) -> str:
    """Read one required string from an already-decoded defaults document."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty trimmed string")
    return value


def _document_integer(
    value: JsonValue,
    field_name: str,
) -> int:
    """Read one integer from an already-decoded defaults document."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer from 0 through 10000")
    return value


def validate_gcp_project_id(value: str) -> str:
    """Validate one GCP project ID and reject resource URIs early."""

    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("GCP project must be a non-empty trimmed project ID")
    if value.startswith("gs://"):
        raise ValueError(f"GCP project must be a project ID such as 'example-project-1', not bucket URI {value!r}")
    if _GCP_PROJECT_ID.fullmatch(value) is None:
        raise ValueError("GCP project must be a 6-30 character lowercase project ID")
    return value


@dataclass(frozen=True, slots=True)
class QueueAdmissionDenialKey:
    """Exact GCP Spot TPU identity blocked by a prior queue-admission denial.

    The denial is intentionally narrower than quota: it applies to one GCP
    project, zone, accelerator type, and Spot purchase class only. A denial
    must never hide an unrelated TPU type or location.
    """

    project: str
    zone: str
    accelerator_type: str
    capacity_class: CapacityClass

    def __post_init__(self) -> None:
        validate_gcp_project_id(self.project)
        if not _ZONE.fullmatch(self.zone):
            raise ValueError(f"invalid GCP zone {self.zone!r}")
        if parse_tpu_accelerator_type(self.accelerator_type) is None:
            raise ValueError(f"invalid GCP TPU accelerator type {self.accelerator_type!r}")
        if not isinstance(self.capacity_class, CapacityClass):
            raise ValueError("capacity_class must be a CapacityClass")
        if self.capacity_class is not CapacityClass.SPOT:
            raise ValueError("queue-admission denial caching supports only Spot TPU workers")

    def as_document(self) -> JsonObject:
        """Return the stable owner-only cache identity document."""

        return {
            "accelerator_type": self.accelerator_type,
            "capacity_class": self.capacity_class.value,
            "project": self.project,
            "zone": self.zone,
        }


@dataclass(frozen=True, slots=True)
class QueueAdmissionDenial:
    """One durable local record of a rejected GCP Spot TPU queue request.

    Entries expire after ``QueueAdmissionDenialCache.DEFAULT_TTL_S`` (24
    hours) and are evicted lazily on the next read, so a denied shape becomes
    selectable again without hand-clearing the cache. They filter only
    quota-based discovery and automatic fleet selection; an explicitly
    configured workload request is still sent to GCP.
    """

    key: QueueAdmissionDenialKey
    observed_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.key, QueueAdmissionDenialKey):
            raise ValueError("key must be a QueueAdmissionDenialKey")
        if not isinstance(self.observed_at, datetime) or self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be a timezone-aware datetime")
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(UTC))

    def as_document(self) -> JsonObject:
        """Return the stable owner-only cache record document."""

        document = self.key.as_document()
        document["observed_at"] = self.observed_at.isoformat()
        return document


@dataclass(frozen=True, slots=True)
class AutoscaleDefaults:
    """YAML-backed operator defaults used to generate complete TPU fleet files.

    The project remains optional only so ``autoscale init`` can repair a
    partially written file interactively. GCP worker identity and cell-agent
    routing are control-plane concerns, never user-profile settings.
    """

    project: str | None = None
    fleet_prefix: str = "eyvan"
    bootstrap_egress: str = "public-internet"
    #: The Python a worker on these images runs, as ``major.minor``. Declared so
    #: the control plane can refuse a callable pickled on a different
    #: interpreter *before* provisioning: that check has existed since a live
    #: multislice v5p-16 died on it in every rank, and it was unreachable from a
    #: generated fleet because nothing here filled the field in. Left unset the
    #: behaviour is unchanged and the mismatch is discovered on the accelerator.
    runtime_python: str | None = None
    #: Shell run on every host after the worker environment exists and before
    #: the worker unit starts. That window is the only place a fleet can install
    #: what a declared interpreter needs: a venv built on a Python the image does
    #: not ship cannot use the image's preinstalled JAX and libtpu, so the fleet
    #: brings its own.
    runtime_init_commands: tuple[str, ...] = ()
    #: A default OCI image reference, ideally an immutable digest, emitted as
    #: the ``container`` block of every generated fleet. One line here is how an
    #: organisation pins one blessed runtime image for every zone anyone will
    #: ever generate -- "the A1 image" -- instead of hand-authoring OCI fleet
    #: YAML per zone. Pair it with ``runtime_init_commands`` that make the host
    #: container-ready (docker installed, ``gcloud auth configure-docker`` for
    #: the image's registry host); the generator states the image, not the host
    #: preparation, because pretending to derive shell that depends on the
    #: image's registry would hide the one command an operator must own. Left
    #: unset, generated fleets keep the venv model exactly as before.
    runtime_container_image: str | None = None
    min_workers: int = 0
    max_workers: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "project", _optional_trimmed(self.project, "project"))
        fleet_prefix = _optional_trimmed(self.fleet_prefix, "fleet_prefix")
        bootstrap_egress = _optional_trimmed(self.bootstrap_egress, "bootstrap_egress")
        object.__setattr__(self, "runtime_python", _optional_trimmed(self.runtime_python, "runtime_python"))
        object.__setattr__(
            self, "runtime_container_image", _optional_trimmed(self.runtime_container_image, "runtime_container_image")
        )
        if fleet_prefix is None:
            raise ValueError("fleet_prefix must be a non-empty trimmed string")
        if bootstrap_egress is None:
            raise ValueError("bootstrap_egress must be a non-empty trimmed string")
        object.__setattr__(self, "fleet_prefix", fleet_prefix)
        object.__setattr__(self, "bootstrap_egress", bootstrap_egress)
        if (
            isinstance(self.min_workers, bool)
            or not isinstance(self.min_workers, int)
            or not 0 <= self.min_workers <= 10_000
        ):
            raise ValueError("min_workers must be an integer from 0 through 10000")
        if (
            isinstance(self.max_workers, bool)
            or not isinstance(self.max_workers, int)
            or not 0 <= self.max_workers <= 10_000
        ):
            raise ValueError("max_workers must be an integer from 0 through 10000")
        if self.max_workers and self.min_workers > self.max_workers:
            raise ValueError("max_workers must be zero or at least min_workers")

    @property
    def missing_required_fields(self) -> tuple[str, ...]:
        """Return identity fields that must be present before generation."""

        fields = (("project", self.project),)
        return tuple(name for name, value in fields if value is None)

    @property
    def complete(self) -> bool:
        """Return whether this profile can generate a fleet without prompting."""

        return not self.missing_required_fields

    def validate_for_generation(self) -> None:
        """Validate complete user-owned values before contacting a provider."""

        missing = self.missing_required_fields
        if missing:
            raise ValueError(f"autoscale defaults is missing {', '.join(missing)}")
        assert self.project is not None
        validate_gcp_project_id(self.project)

    def as_document(self) -> JsonObject:
        """Return the stable, human-editable defaults YAML document."""

        return {
            "version": 1,
            "project": self.project,
            "fleet_prefix": self.fleet_prefix,
            "bootstrap_egress": self.bootstrap_egress,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            **({} if self.runtime_python is None else {"runtime_python": self.runtime_python}),
            **({} if not self.runtime_init_commands else {"runtime_init_commands": list(self.runtime_init_commands)}),
            **(
                {} if self.runtime_container_image is None else {"runtime_container_image": self.runtime_container_image}
            ),
        }

    @classmethod
    def from_document(cls, value: JsonObject) -> AutoscaleDefaults:
        """Parse one strict defaults YAML mapping."""

        if not isinstance(value, dict):
            raise ValueError("autoscale defaults YAML must be a mapping")
        allowed = {
            "version",
            "project",
            # Kept only to read profiles written by Eyvan 0.1. Both fields
            # are control-plane owned and deliberately omitted when saving.
            "agent_endpoint",
            "worker_service_account",
            "fleet_prefix",
            "bootstrap_egress",
            "min_workers",
            "max_workers",
            "runtime_python",
            "runtime_init_commands",
            "runtime_container_image",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown autoscale defaults field {unknown[0]!r}")
        version = value.get("version", 1)
        if isinstance(version, bool) or not isinstance(version, int) or version != 1:
            raise ValueError("autoscale defaults version must be 1")
        project = _document_optional_string(value.get("project"), "project")
        fleet_prefix = _document_required_string(value.get("fleet_prefix", "eyvan"), "fleet_prefix")
        bootstrap_egress = _document_required_string(
            value.get("bootstrap_egress", "public-internet"), "bootstrap_egress"
        )
        min_workers = _document_integer(value.get("min_workers", 0), "min_workers")
        max_workers = _document_integer(value.get("max_workers", 0), "max_workers")
        runtime_python = _document_optional_string(value.get("runtime_python"), "runtime_python")
        runtime_container_image = _document_optional_string(
            value.get("runtime_container_image"), "runtime_container_image"
        )
        raw_commands = value.get("runtime_init_commands", [])
        if not isinstance(raw_commands, list) or any(not isinstance(item, str) for item in raw_commands):
            raise ValueError("runtime_init_commands must be a list of shell strings")
        return cls(
            project=project,
            fleet_prefix=fleet_prefix,
            bootstrap_egress=bootstrap_egress,
            min_workers=min_workers,
            max_workers=max_workers,
            runtime_python=runtime_python,
            runtime_init_commands=tuple(raw_commands),
            runtime_container_image=runtime_container_image,
        )


@dataclass(frozen=True, slots=True)
class TpuProfile:
    """One complete TPU worker shape with all host admission resources.

    Discovered accelerator names and quota suffixes establish only TPU count;
    they do not establish a host's CPU or RAM. Callers constructing a profile
    outside the shipped catalog must therefore provide positive memory
    explicitly rather than inheriting a zero or guessed fallback.
    """

    accelerator_type: str
    cpu: int
    memory_gib: float
    tpu: int

    def __post_init__(self) -> None:
        if parse_tpu_accelerator_type(self.accelerator_type) is None:
            raise ValueError(f"unsupported TPU accelerator type {self.accelerator_type!r}")
        if isinstance(self.cpu, bool) or not isinstance(self.cpu, int) or self.cpu <= 0:
            raise ValueError("cpu must be a positive integer")
        if (
            isinstance(self.memory_gib, bool)
            or not isinstance(self.memory_gib, int | float)
            or not math.isfinite(self.memory_gib)
            or self.memory_gib <= 0
        ):
            raise ValueError("memory_gib must be a positive number")
        if isinstance(self.tpu, bool) or not isinstance(self.tpu, int) or self.tpu <= 0:
            raise ValueError("tpu must be a positive integer")

    @property
    def generation(self) -> str:
        """Return the accelerator family used by generation filters."""

        parsed = parse_tpu_accelerator_type(self.accelerator_type)
        if parsed is None:
            raise ValueError(f"unsupported TPU accelerator type {self.accelerator_type!r}")
        return parsed.generation

    @property
    def quota_tpus_per_worker(self) -> int:
        """Return the Cloud TPU quota units consumed by one worker allocation."""

        parsed = parse_tpu_accelerator_type(self.accelerator_type)
        if parsed is None:
            raise ValueError(f"unsupported TPU accelerator type {self.accelerator_type!r}")
        return parsed.size


def _validate_observed_quota_balance(
    *,
    quota_tpus: int | None,
    available_tpus: int | None,
    usage_observed_at: datetime | None,
) -> None:
    """Validate one finite live quota balance or an explicitly unlimited quota."""

    if quota_tpus is not None and (isinstance(quota_tpus, bool) or not isinstance(quota_tpus, int) or quota_tpus < 0):
        raise ValueError("quota_tpus must be a non-negative integer or None")
    if available_tpus is not None and (
        isinstance(available_tpus, bool) or not isinstance(available_tpus, int) or available_tpus < 0
    ):
        raise ValueError("available_tpus must be a non-negative integer or None")
    if quota_tpus is None:
        if available_tpus is not None or usage_observed_at is not None:
            raise ValueError("unlimited quota must not carry a finite usage observation")
        return
    if available_tpus is None:
        if usage_observed_at is not None:
            raise ValueError("usage_observed_at requires available_tpus")
        return
    if available_tpus > quota_tpus:
        raise ValueError("available_tpus must not exceed quota_tpus")
    if usage_observed_at is not None and (
        not isinstance(usage_observed_at, datetime) or usage_observed_at.tzinfo is None
    ):
        raise ValueError("usage_observed_at must be timezone-aware when set")


@dataclass(frozen=True, slots=True)
class QuotaSnapshot:
    """One read-only provider quota observation for a TPU shape in one zone.

    ``quota_tpus`` is a provider quota limit. ``available_tpus`` is the live,
    conservative balance used to constrain Eyvan's allocation budget. A
    finite quota without ``available_tpus`` has unknown live usage and cannot
    safely be used for quota-based autoscale generation.
    """

    zone: str
    accelerator_type: str
    generation: str
    quota_tpus: int | None
    preemptible: bool = True
    source_metric: str | None = None
    available_tpus: int | None = None
    usage_observed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not _ZONE.fullmatch(self.zone):
            raise ValueError(f"invalid GCP zone {self.zone!r}")
        if not self.accelerator_type or self.accelerator_type != self.accelerator_type.strip():
            raise ValueError("accelerator_type must be a non-empty trimmed string")
        if not self.generation or self.generation != self.generation.strip():
            raise ValueError("generation must be a non-empty trimmed string")
        _validate_observed_quota_balance(
            quota_tpus=self.quota_tpus,
            available_tpus=self.available_tpus,
            usage_observed_at=self.usage_observed_at,
        )
        if not isinstance(self.preemptible, bool):
            raise ValueError("preemptible must be a boolean")
        if self.source_metric is not None and (
            not isinstance(self.source_metric, str)
            or not self.source_metric.strip()
            or self.source_metric != self.source_metric.strip()
        ):
            raise ValueError("source_metric must be a non-empty trimmed string or None")
        parsed = parse_tpu_accelerator_type(self.accelerator_type)
        if parsed is not None and self.generation != parsed.generation:
            raise ValueError("generation must match accelerator_type")

    def as_dict(self) -> JsonObject:
        """Return a JSON-safe representation for cache metadata and the CLI."""

        document: JsonObject = {
            "accelerator_type": self.accelerator_type,
            "available_tpus": self.available_tpus,
            "generation": self.generation,
            "preemptible": self.preemptible,
            "quota_tpus": self.quota_tpus,
            "source_metric": self.source_metric,
            "zone": self.zone,
        }
        if self.usage_observed_at is not None:
            document["usage_observed_at"] = self.usage_observed_at.isoformat()
        return document


@dataclass(frozen=True, slots=True)
class AutoscaleRequest:
    """Operator inputs shared by generated single-zone fleet configurations."""

    project: str
    zone: str
    profile: TpuProfile
    fleet_prefix: str = "eyvan"
    min_workers: int = 0
    max_workers: int = 0
    bootstrap_egress: str = "public-internet"
    #: The Python a worker on these images runs, as ``major.minor``. Declared so
    #: the control plane can refuse a callable pickled on a different
    #: interpreter *before* provisioning: that check has existed since a live
    #: multislice v5p-16 died on it in every rank, and it was unreachable from a
    #: generated fleet because nothing here filled the field in. Left unset the
    #: behaviour is unchanged and the mismatch is discovered on the accelerator.
    runtime_python: str | None = None
    runtime_init_commands: tuple[str, ...] = ()
    #: Default OCI image for the generated fleet's ``container`` block, when the
    #: operator's profile pins one blessed runtime image for every zone. The
    #: host-side preparation (docker, registry auth) stays in
    #: ``runtime_init_commands`` where the operator owns it.
    runtime_container_image: str | None = None

    def __post_init__(self) -> None:
        validate_gcp_project_id(self.project)
        if not self.fleet_prefix or self.fleet_prefix != self.fleet_prefix.strip():
            raise ValueError("fleet_prefix must be a non-empty trimmed string")
        if not self.bootstrap_egress or self.bootstrap_egress != self.bootstrap_egress.strip():
            raise ValueError("bootstrap_egress must be a non-empty trimmed string")
        if not _ZONE.fullmatch(self.zone):
            raise ValueError(f"invalid GCP zone {self.zone!r}")
        if isinstance(self.min_workers, bool) or not isinstance(self.min_workers, int) or self.min_workers < 0:
            raise ValueError("min_workers must be a non-negative integer")
        if isinstance(self.max_workers, bool) or not isinstance(self.max_workers, int) or self.max_workers < 0:
            raise ValueError("max_workers must be a non-negative integer")

    @property
    def fleet_name(self) -> str:
        """Return a deterministic valid fleet name for this zone/profile pair."""

        return autoscale_fleet_name(self.fleet_prefix, self.profile.accelerator_type, self.zone)


@dataclass(frozen=True, slots=True)
class AutoscaleZoneRequest:
    """Operator inputs for one zone fleet containing every selected TPU pool.

    A GCP TPU quota applies to a generation in a zone, not independently to
    every accelerator shape in that generation. Keeping selected profiles
    together makes that relationship explicit in generated cache metadata.
    """

    project: str
    zone: str
    profiles: tuple[TpuProfile, ...]
    fleet_prefix: str = "eyvan"
    min_workers: int = 0
    max_workers: int = 0
    bootstrap_egress: str = "public-internet"
    #: The Python a worker on these images runs, as ``major.minor``. Declared so
    #: the control plane can refuse a callable pickled on a different
    #: interpreter *before* provisioning: that check has existed since a live
    #: multislice v5p-16 died on it in every rank, and it was unreachable from a
    #: generated fleet because nothing here filled the field in. Left unset the
    #: behaviour is unchanged and the mismatch is discovered on the accelerator.
    runtime_python: str | None = None
    runtime_init_commands: tuple[str, ...] = ()
    #: Default OCI image for the generated fleet's ``container`` block, when the
    #: operator's profile pins one blessed runtime image for every zone. The
    #: host-side preparation (docker, registry auth) stays in
    #: ``runtime_init_commands`` where the operator owns it.
    runtime_container_image: str | None = None

    def __post_init__(self) -> None:
        validate_gcp_project_id(self.project)
        if not self.fleet_prefix or self.fleet_prefix != self.fleet_prefix.strip():
            raise ValueError("fleet_prefix must be a non-empty trimmed string")
        if not self.bootstrap_egress or self.bootstrap_egress != self.bootstrap_egress.strip():
            raise ValueError("bootstrap_egress must be a non-empty trimmed string")
        if not _ZONE.fullmatch(self.zone):
            raise ValueError(f"invalid GCP zone {self.zone!r}")
        if isinstance(self.min_workers, bool) or not isinstance(self.min_workers, int) or self.min_workers < 0:
            raise ValueError("min_workers must be a non-negative integer")
        if isinstance(self.max_workers, bool) or not isinstance(self.max_workers, int) or self.max_workers < 0:
            raise ValueError("max_workers must be a non-negative integer")
        profiles = tuple(self.profiles)
        if not profiles:
            raise ValueError("a zone autoscale fleet requires at least one TPU profile")
        if not all(isinstance(profile, TpuProfile) for profile in profiles):
            raise ValueError("profiles must contain TpuProfile values")
        profiles = tuple(sorted(profiles, key=lambda profile: profile.accelerator_type))
        names = [profile.accelerator_type for profile in profiles]
        if len(names) != len(set(names)):
            raise ValueError("a zone autoscale fleet cannot contain the same accelerator type twice")
        object.__setattr__(self, "profiles", profiles)

    @property
    def fleet_name(self) -> str:
        """Return the stable single fleet name for this GCP zone."""

        return autoscale_fleet_name(self.fleet_prefix, self.zone)


@dataclass(frozen=True, slots=True)
class GeneratedAutoscaleConfig:
    """A validated public YAML document plus its provenance."""

    config: EyvanFleetConfig
    document: JsonObject
    profile: TpuProfile
    quota: QuotaSnapshot | None = None

    @property
    def fleet_name(self) -> str:
        """Return the generated fleet name."""

        return self.config.name


@dataclass(frozen=True, slots=True)
class GenerationQuotaSnapshot:
    """A live Spot TPU quota balance shared by types in one generation/zone."""

    zone: str
    generation: str
    quota_tpus: int | None
    preemptible: bool = True
    source_metric: str | None = None
    shared: bool = False
    available_tpus: int | None = None
    usage_observed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not _ZONE.fullmatch(self.zone):
            raise ValueError(f"invalid GCP zone {self.zone!r}")
        if not self.generation or self.generation != self.generation.strip():
            raise ValueError("generation must be a non-empty trimmed string")
        _validate_observed_quota_balance(
            quota_tpus=self.quota_tpus,
            available_tpus=self.available_tpus,
            usage_observed_at=self.usage_observed_at,
        )
        if not isinstance(self.preemptible, bool):
            raise ValueError("preemptible must be a boolean")
        if self.source_metric is not None and (
            not isinstance(self.source_metric, str)
            or not self.source_metric.strip()
            or self.source_metric != self.source_metric.strip()
        ):
            raise ValueError("source_metric must be a non-empty trimmed string or None")
        if not isinstance(self.shared, bool):
            raise ValueError("shared must be a boolean")

    def as_dict(self) -> JsonObject:
        """Return cache and CLI provenance without attributing it to one type."""

        document: JsonObject = {
            "available_tpus": self.available_tpus,
            "generation": self.generation,
            "preemptible": self.preemptible,
            "quota_tpus": self.quota_tpus,
            "shared": self.shared,
            "source_metric": self.source_metric,
            "zone": self.zone,
        }
        if self.usage_observed_at is not None:
            document["usage_observed_at"] = self.usage_observed_at.isoformat()
        return document


@dataclass(frozen=True, slots=True)
class GeneratedZoneAutoscaleConfig:
    """One zone fleet plus its typed pool and generation-quota provenance."""

    config: EyvanFleetConfig
    document: JsonObject
    profiles: tuple[TpuProfile, ...]
    generation_quotas: tuple[GenerationQuotaSnapshot, ...] = ()

    @property
    def fleet_name(self) -> str:
        """Return the generated zone fleet name."""

        return self.config.name


@dataclass(frozen=True, slots=True)
class CachedAutoscaleConfig:
    """One generated config persisted below the Eyvan autoscale cache."""

    path: Path
    config: EyvanFleetConfig
    quota: QuotaSnapshot | None
    generated_at: str | None
    public_max_workers: int | None = None
    pool_max_workers: tuple[tuple[str, int], ...] = ()
    generation_quotas: tuple[GenerationQuotaSnapshot, ...] = ()

    @property
    def worker_limits(self) -> tuple[int, ...]:
        """Return public worker ceilings in the same stable order as pool views.

        The fleet schema normalizes public ``max_workers: 0`` to an internal
        cell ceiling. This property preserves the original public value when
        cache metadata or generated YAML retains it, so terminal views never
        need to serialize then parse :meth:`as_dict`.
        """

        known_limits = dict(self.pool_max_workers)
        limits: list[int] = []
        for pool_name, pool in sorted(self.config.pools.items()):
            limit = known_limits.get(pool_name)
            if limit is None and len(self.config.pools) == 1:
                limit = self.public_max_workers
            limits.append(pool.max_cells if limit is None else limit)
        return tuple(limits)

    def as_dict(self) -> JsonObject:
        """Return concise JSON-safe details for a terminal selector."""

        pool_max_workers = dict(self.pool_max_workers)
        pools: list[JsonValue] = []
        for pool_name, pool in sorted(self.config.pools.items()):
            public_max_workers = pool_max_workers.get(pool_name)
            if public_max_workers is None and len(self.config.pools) == 1:
                public_max_workers = self.public_max_workers
            pools.append(
                {
                    "accelerator_type": pool.accelerator_type,
                    "capacity_class": (pool.capacity_class or self.config.capacity_class).value,
                    "max_workers": public_max_workers if public_max_workers is not None else pool.max_cells,
                    "min_workers": pool.min_cells,
                    "zone": pool.zone or self.config.provider.zone,
                }
            )
        generation_quotas: list[JsonValue] = [quota.as_dict() for quota in self.generation_quotas]
        payload: JsonObject = {
            "fleet": self.config.name,
            "generated_at": self.generated_at,
            "generation_quotas": generation_quotas,
            "path": str(self.path),
            "pools": pools,
            "zone": self.config.provider.zone,
        }
        if self.quota is not None:
            payload["quota"] = self.quota.as_dict()
        return payload


@dataclass(frozen=True, slots=True)
class AutoscaleGenerationRequest:
    """Typed input for discovery-backed zone fleet generation."""

    defaults: AutoscaleDefaults
    zones: tuple[str, ...] = ()
    generations: tuple[str, ...] = ()
    persist: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.defaults, AutoscaleDefaults):
            raise ValueError("defaults must be an AutoscaleDefaults value")
        if not isinstance(self.persist, bool):
            raise ValueError("persist must be a boolean")


@dataclass(frozen=True, slots=True)
class SkippedAutoscaleCandidate:
    """One concise group of candidates omitted by autoscale generation.

    The discovery API can expose a large catalog of valid accelerator strings.
    Eyvan never serializes one diagnostic per omitted string: normal exclusions
    are grouped by zone, TPU generation, and canonical reason.
    """

    reason: AutoscaleSkipReason
    scope: Literal["generation", "zone"]
    zone: str
    generation: str | None = None
    generations: tuple[str, ...] = ()
    type_count: int = 0
    detail: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, AutoscaleSkipReason):
            raise ValueError("reason must be an AutoscaleSkipReason")
        if self.scope not in {"generation", "zone"}:
            raise ValueError("scope must be 'generation' or 'zone'")
        if not _ZONE.fullmatch(self.zone):
            raise ValueError(f"invalid GCP zone {self.zone!r}")
        if isinstance(self.type_count, bool) or not isinstance(self.type_count, int) or self.type_count <= 0:
            raise ValueError("type_count must be a positive integer")
        if self.detail is not None and (
            not isinstance(self.detail, str) or not self.detail.strip() or self.detail != self.detail.strip()
        ):
            raise ValueError("detail must be a non-empty trimmed string when set")
        if self.scope == "generation":
            if self.generation is None or self.generations:
                raise ValueError("generation skips require only generation")
            if not self.generation.strip() or self.generation != self.generation.strip():
                raise ValueError("generation must be a non-empty trimmed string")
            return
        if self.generation is not None or not self.generations:
            raise ValueError("zone skips require generations")
        if any(not generation.strip() or generation != generation.strip() for generation in self.generations):
            raise ValueError("generations must contain non-empty trimmed strings")

    def as_dict(self) -> JsonObject:
        """Return one bounded machine-readable omission diagnostic."""

        if self.scope == "generation":
            if self.generation is None:
                raise AssertionError("validated generation skip is missing generation")
            payload: JsonObject = {
                "generation": self.generation,
                "reason": self.reason.value,
                "scope": self.scope,
                "type_count": self.type_count,
                "zone": self.zone,
            }
        elif self.scope == "zone":
            payload = {
                "generations": list(self.generations),
                "reason": self.reason.value,
                "scope": self.scope,
                "type_count": self.type_count,
                "zone": self.zone,
            }
        else:
            raise AssertionError("validated autoscale skip has an unknown scope")
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


__all__ = (
    "DEFAULT_AUTOSCALE_CACHE_ROOT",
    "DEFAULT_AUTOSCALE_DEFAULTS_PATH",
    "AutoscaleDefaults",
    "AutoscaleGenerationRequest",
    "AutoscaleRequest",
    "AutoscaleSkipReason",
    "AutoscaleZoneRequest",
    "CachedAutoscaleConfig",
    "GeneratedAutoscaleConfig",
    "GeneratedZoneAutoscaleConfig",
    "GenerationQuotaSnapshot",
    "QueueAdmissionDenial",
    "QueueAdmissionDenialKey",
    "QuotaSnapshot",
    "SkippedAutoscaleCandidate",
    "TpuProfile",
    "autoscale_fleet_name",
    "validate_gcp_project_id",
)
