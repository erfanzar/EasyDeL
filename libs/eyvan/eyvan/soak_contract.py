# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Closed, bounded identities shared by the P8 observer and control plane.

The privileged invariant endpoint sees tenant IDs, object names, locations and
provider metadata.  None of those values is receipt evidence.  This module owns
the deliberately small vocabulary and the domain-separated opaque identifiers
that are allowed to cross the receipt boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
import typing as tp
from dataclasses import dataclass

SAFE_EXECUTION_KINDS: tp.Final[tuple[str, ...]] = ("callable", "command", "other")
SAFE_COORDINATION_KINDS: tp.Final[tuple[str, ...]] = ("none", "batch", "dag", "group", "service", "other")
SAFE_EXECUTORS: tp.Final[tuple[str, ...]] = ("any", "native-local", "eyvan-native", "ray-jobs", "other")
SAFE_CAPACITY_BACKENDS: tp.Final[tuple[str, ...]] = (
    "local",
    "manual",
    "gcp",
    "gcp-attach",
    "gcp-compute",
    "gcp-tpu",
    "onprem-ssh",
    "aws-ec2",
    "azure-vm",
    "kubernetes",
    "other",
)
SAFE_PROVIDERS: tp.Final[tuple[str, ...]] = (
    "local",
    "manual",
    "gcp",
    "onprem-ssh",
    "aws",
    "azure",
    "kubernetes",
    "other",
)
SAFE_RESOURCE_CLASSES: tp.Final[tuple[str, ...]] = ("cpu", "gpu", "tpu", "other")
SAFE_CAPACITY_CLASSES: tp.Final[tuple[str, ...]] = ("reserved", "on_demand", "spot", "other")
SAFE_FAILURE_CATEGORIES: tp.Final[tuple[str, ...]] = (
    "authorization",
    "filesystem",
    "internal",
    "invalid_response",
    "request",
    "server",
    "timeout",
    "unreachable",
)
AVAILABILITY_BUDGET_CATEGORIES: tp.Final[frozenset[str]] = frozenset({"timeout", "unreachable"})

MAX_COVERAGE_ENTRIES: tp.Final[int] = 64
MAX_EXPECTED_IDENTITIES: tp.Final[int] = 32
MAX_PARTICIPANTS: tp.Final[int] = 64
MAX_SERVER_VIOLATIONS: tp.Final[int] = 128
MAX_BUDGET_SCOPES: tp.Final[int] = 2_048
MAX_SAFE_COUNTER: tp.Final[int] = 2**63 - 1

SOAK_PROOF_METADATA_KEY: tp.Final[str] = "eyvan.soak.proof_id"
SOAK_CONFIG_METADATA_KEY: tp.Final[str] = "eyvan.soak.config_sha256"

_OPAQUE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_RECORD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+\-]{0,95}")


def _require_enum(name: str, value: object, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} is not a supported safe identity value")
    return value


def is_opaque_id(value: object) -> bool:
    """Return whether ``value`` is one canonical 256-bit opaque identifier."""

    return isinstance(value, str) and _OPAQUE_ID.fullmatch(value) is not None


def require_opaque_id(name: str, value: object) -> str:
    """Validate and return one canonical opaque identifier."""

    if not is_opaque_id(value):
        raise ValueError(f"{name} must be a lowercase sha256 opaque identifier")
    return tp.cast(str, value)


def opaque_id(domain: str, *parts: object) -> str:
    """Return a domain-separated, fixed-width, non-echoing identifier."""

    hasher = hashlib.sha256()
    hasher.update(domain.encode("ascii", errors="strict"))
    hasher.update(b"\0")
    for part in parts:
        encoded = str(part).encode("utf-8", errors="replace")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return f"sha256:{hasher.hexdigest()}"


def safe_scope_id(*parts: object) -> str:
    """Return an opaque longitudinal budget-scope identifier."""

    return opaque_id("eyvan-soak-budget-scope-v1", *parts)


def safe_period_id(period_started_at: object) -> str:
    """Return a stable period identity without exporting its source text."""

    marker = "uninitialized" if period_started_at is None else period_started_at
    return opaque_id("eyvan-soak-budget-period-v1", marker)


def safe_subject_id(invariant: object, subject: object) -> str:
    """Return an opaque receipt subject; raw diagnostic text is discarded."""

    return opaque_id("eyvan-soak-violation-subject-v1", invariant, subject)


def safe_enum(
    value: object,
    allowed: tuple[str, ...],
    *,
    aliases: tp.Mapping[str, str] | None = None,
    missing: str = "other",
) -> str:
    """Map a runtime value to a closed enum without retaining the input."""

    raw = getattr(value, "value", value)
    if raw is None:
        raw = missing
    if not isinstance(raw, str):
        return "other"
    normalized = raw.strip().lower().replace("_", "-")
    if aliases is not None:
        normalized = aliases.get(normalized, normalized).replace("_", "-")
    canonical = {item.replace("_", "-"): item for item in allowed}
    return canonical.get(normalized, "other")


@dataclass(frozen=True, order=True, slots=True)
class CapabilityIdentity:
    """Allowlisted workload capability attached to an executed lease."""

    execution: str
    checkpointable: bool
    coordination: str
    required_executor: str

    def __post_init__(self) -> None:
        _require_enum("execution", self.execution, SAFE_EXECUTION_KINDS)
        if not isinstance(self.checkpointable, bool):
            raise ValueError("checkpointable must be boolean")
        _require_enum("coordination", self.coordination, SAFE_COORDINATION_KINDS)
        _require_enum("required_executor", self.required_executor, SAFE_EXECUTORS)

    @property
    def classified(self) -> bool:
        return "other" not in self.as_dict().values()

    def as_dict(self) -> dict[str, object]:
        return {
            "execution": self.execution,
            "checkpointable": self.checkpointable,
            "coordination": self.coordination,
            "required_executor": self.required_executor,
        }

    @classmethod
    def from_dict(cls, value: object) -> CapabilityIdentity:
        if not isinstance(value, dict) or set(value) != {
            "execution",
            "checkpointable",
            "coordination",
            "required_executor",
        }:
            raise ValueError("capability identity has unexpected fields")
        return cls(
            execution=_require_enum("execution", value["execution"], SAFE_EXECUTION_KINDS),
            checkpointable=value["checkpointable"],
            coordination=_require_enum("coordination", value["coordination"], SAFE_COORDINATION_KINDS),
            required_executor=_require_enum("required_executor", value["required_executor"], SAFE_EXECUTORS),
        )


@dataclass(frozen=True, order=True, slots=True)
class BackendIdentity:
    """Allowlisted capacity/execution backend pair."""

    capacity_backend: str
    executor: str

    def __post_init__(self) -> None:
        _require_enum("capacity_backend", self.capacity_backend, SAFE_CAPACITY_BACKENDS)
        _require_enum("executor", self.executor, SAFE_EXECUTORS)

    @property
    def classified(self) -> bool:
        return "other" not in self.as_dict().values()

    def as_dict(self) -> dict[str, str]:
        return {"capacity_backend": self.capacity_backend, "executor": self.executor}

    @classmethod
    def from_dict(cls, value: object) -> BackendIdentity:
        if not isinstance(value, dict) or set(value) != {"capacity_backend", "executor"}:
            raise ValueError("backend identity has unexpected fields")
        return cls(
            capacity_backend=_require_enum("capacity_backend", value["capacity_backend"], SAFE_CAPACITY_BACKENDS),
            executor=_require_enum("executor", value["executor"], SAFE_EXECUTORS),
        )


@dataclass(frozen=True, order=True, slots=True)
class ProviderIdentity:
    """Allowlisted provider class with no project, location, URI, or label."""

    provider: str
    resource_class: str
    capacity_class: str

    def __post_init__(self) -> None:
        _require_enum("provider", self.provider, SAFE_PROVIDERS)
        _require_enum("resource_class", self.resource_class, SAFE_RESOURCE_CLASSES)
        _require_enum("capacity_class", self.capacity_class, SAFE_CAPACITY_CLASSES)

    @property
    def classified(self) -> bool:
        return "other" not in self.as_dict().values()

    def as_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "resource_class": self.resource_class,
            "capacity_class": self.capacity_class,
        }

    @classmethod
    def from_dict(cls, value: object) -> ProviderIdentity:
        if not isinstance(value, dict) or set(value) != {"provider", "resource_class", "capacity_class"}:
            raise ValueError("provider identity has unexpected fields")
        return cls(
            provider=_require_enum("provider", value["provider"], SAFE_PROVIDERS),
            resource_class=_require_enum("resource_class", value["resource_class"], SAFE_RESOURCE_CLASSES),
            capacity_class=_require_enum("capacity_class", value["capacity_class"], SAFE_CAPACITY_CLASSES),
        )


@dataclass(frozen=True, order=True, slots=True)
class ExecutionIdentity:
    """One coherent workload/backend/provider execution path."""

    capability: CapabilityIdentity
    backend: BackendIdentity
    provider: ProviderIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.capability, CapabilityIdentity):
            raise ValueError("execution capability must be a CapabilityIdentity")
        if not isinstance(self.backend, BackendIdentity):
            raise ValueError("execution backend must be a BackendIdentity")
        if not isinstance(self.provider, ProviderIdentity):
            raise ValueError("execution provider must be a ProviderIdentity")

    @property
    def classified(self) -> bool:
        return self.capability.classified and self.backend.classified and self.provider.classified

    def as_dict(self) -> dict[str, object]:
        return {
            "capability": self.capability.as_dict(),
            "backend": self.backend.as_dict(),
            "provider": self.provider.as_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> ExecutionIdentity:
        if not isinstance(value, dict) or set(value) != {"capability", "backend", "provider"}:
            raise ValueError("execution identity has unexpected fields")
        return cls(
            capability=CapabilityIdentity.from_dict(value["capability"]),
            backend=BackendIdentity.from_dict(value["backend"]),
            provider=ProviderIdentity.from_dict(value["provider"]),
        )


@dataclass(frozen=True, order=True, slots=True)
class ParticipantBinding:
    """Exact raw workload/lease/cell join, retained only inside the request."""

    workload_id: str
    lease_id: str
    cell_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("workload_id", self.workload_id),
            ("lease_id", self.lease_id),
            ("cell_id", self.cell_id),
        ):
            if not isinstance(value, str) or _RECORD_ID.fullmatch(value) is None:
                raise ValueError(f"{name} is not a safe bounded record identifier")

    def as_request_dict(self) -> dict[str, str]:
        return {"workload_id": self.workload_id, "lease_id": self.lease_id, "cell_id": self.cell_id}

    @classmethod
    def from_dict(cls, value: object) -> ParticipantBinding:
        if not isinstance(value, dict) or set(value) != {"workload_id", "lease_id", "cell_id"}:
            raise ValueError("participant binding has unexpected fields")
        return cls(
            workload_id=value["workload_id"],
            lease_id=value["lease_id"],
            cell_id=value["cell_id"],
        )

    def opaque_id(self, proof_id: str) -> str:
        require_opaque_id("proof_id", proof_id)
        return opaque_id("eyvan-soak-participant-v1", proof_id, self.workload_id, self.lease_id, self.cell_id)


@dataclass(frozen=True, order=True, slots=True)
class ExpectedExecution:
    """Expected joint identity for one exact participant join."""

    participant: ParticipantBinding
    identity: ExecutionIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.participant, ParticipantBinding):
            raise ValueError("expected execution requires a participant binding")
        if not isinstance(self.identity, ExecutionIdentity):
            raise ValueError("expected execution requires a joint identity")
        validate_expected_identity(self.identity)

    def as_receipt_dict(self, proof_id: str) -> dict[str, object]:
        return {"participant_id": self.participant.opaque_id(proof_id), **self.identity.as_dict()}


@dataclass(frozen=True, order=True, slots=True)
class ParticipantExecutionIdentity:
    """Receipt-safe participant plus its coherent observed execution identity."""

    participant_id: str
    identity: ExecutionIdentity

    def __post_init__(self) -> None:
        require_opaque_id("participant_id", self.participant_id)
        if not isinstance(self.identity, ExecutionIdentity):
            raise ValueError("participant execution requires a joint identity")

    def as_dict(self) -> dict[str, object]:
        return {"participant_id": self.participant_id, **self.identity.as_dict()}


def participant_manifest_sha256(proof_id: str, participants: tp.Iterable[ParticipantBinding]) -> str:
    """Hash the exact sorted participant set without exporting its raw IDs."""

    require_opaque_id("proof_id", proof_id)
    unique = tuple(sorted(set(participants)))
    if not unique or len(unique) > MAX_PARTICIPANTS:
        raise ValueError("participant manifest must contain a bounded non-empty exact set")
    manifest = json.dumps(
        [participant.as_request_dict() for participant in unique],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(b"eyvan-soak-participant-manifest-v1\0" + proof_id.encode("ascii") + manifest).hexdigest()
    return f"sha256:{digest}"


def validate_expected_identity(
    identity: CapabilityIdentity | BackendIdentity | ProviderIdentity | ExecutionIdentity,
) -> None:
    """Reject the runtime-only ``other`` sentinel from qualification inputs."""

    if not identity.classified:
        raise ValueError("qualification expectations cannot use the 'other' identity sentinel")


__all__ = (
    "AVAILABILITY_BUDGET_CATEGORIES",
    "MAX_BUDGET_SCOPES",
    "MAX_COVERAGE_ENTRIES",
    "MAX_EXPECTED_IDENTITIES",
    "MAX_PARTICIPANTS",
    "MAX_SAFE_COUNTER",
    "MAX_SERVER_VIOLATIONS",
    "SAFE_CAPACITY_BACKENDS",
    "SAFE_CAPACITY_CLASSES",
    "SAFE_COORDINATION_KINDS",
    "SAFE_EXECUTION_KINDS",
    "SAFE_EXECUTORS",
    "SAFE_FAILURE_CATEGORIES",
    "SAFE_PROVIDERS",
    "SAFE_RESOURCE_CLASSES",
    "SOAK_CONFIG_METADATA_KEY",
    "SOAK_PROOF_METADATA_KEY",
    "BackendIdentity",
    "CapabilityIdentity",
    "ExecutionIdentity",
    "ExpectedExecution",
    "ParticipantBinding",
    "ParticipantExecutionIdentity",
    "ProviderIdentity",
    "is_opaque_id",
    "opaque_id",
    "participant_manifest_sha256",
    "require_opaque_id",
    "safe_enum",
    "safe_period_id",
    "safe_scope_id",
    "safe_subject_id",
    "validate_expected_identity",
)
