# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
#
# Portions adapted from the Iris cluster manager in marin-community/marin
# (lib/iris/src/iris/cluster — Copyright The Marin Authors, Apache-2.0).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Types, validation, and the TPU topology table for the capacity layer.

Adapted from Iris (``iris.cluster.platforms.gcp.service`` /
``iris.cluster.platforms.types`` / ``iris.cluster.tpu_topology``), reduced to
the queued-resource surface eray needs and extended with spot semantics:
Iris only provisions *reserved* capacity through queued resources, while
eray's fleet is primarily ``--spot``.

The error hierarchy is load-bearing: :class:`QuotaExhaustedError` is how a
zone stockout is distinguished from a real failure, which is what lets the
pool loop fall through to the next declared zone.
"""

from __future__ import annotations

import datetime
import re
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from ..provision.qr import PENDING_STATES, TERMINAL_STATES

# ============================================================================
# Errors (ported from iris.cluster.platforms.types)
# ============================================================================


class InfraError(Exception):
    """Base for infrastructure operation failures."""


class QuotaExhaustedError(InfraError):
    """No capacity or quota in the requested zone. Try another zone or wait."""


class ResourceNotFoundError(InfraError):
    """The requested resource does not exist."""


class InfraUnavailableError(InfraError):
    """Transient infrastructure failure. Retry with backoff."""


# ============================================================================
# Capacity tiers
# ============================================================================


class CapacityType(StrEnum):
    """Capacity tier for a queued-resource request.

    Values match eray's existing ``eray qr create --capacity`` vocabulary
    (:data:`eray.provision.qr.CapacityT`), which is a superset of Iris's
    ``CapacityType`` (Iris has no distinct ``guaranteed`` tier and never
    requests spot through queued resources).
    """

    SPOT = "spot"
    ON_DEMAND = "on-demand"
    RESERVED = "reserved"
    GUARANTEED = "guaranteed"


# ============================================================================
# QR lifecycle state sets
# ============================================================================

#: States on the way to ACTIVE (re-exported from eray.provision.qr — the
#: single source of truth shared with the gcloud-based fleet path).
QR_PENDING_STATES = PENDING_STATES
#: States that will never progress to ACTIVE on their own (FAILED, SUSPENDED).
QR_DEAD_STATES = TERMINAL_STATES
#: States that count as holding-or-acquiring capacity.
QR_LIVE_STATES = frozenset(QR_PENDING_STATES | {"ACTIVE"})
#: Capacity being torn down (spot preemption in progress). Not live — the
#: pool provisions a replacement immediately — but not yet deleted either;
#: it is reclaimed once it lands in SUSPENDED.
QR_DYING_STATES = frozenset({"SUSPENDING"})

# ============================================================================
# GCP constants and label keys
# ============================================================================

#: GCP zones where TPUs are available (ported from iris; extend the service's
#: ``valid_zones`` attribute at runtime for zones GCP adds later).
KNOWN_GCP_ZONES: frozenset[str] = frozenset(
    {
        "us-central1-a",
        "us-central1-b",
        "us-central1-c",
        "us-central1-f",
        "us-central2-b",
        "us-east1-b",
        "us-east1-d",
        "us-east5-a",
        "us-east5-b",
        "us-east5-c",
        "us-west1-a",
        "us-west1-c",
        "us-west4-a",
        "us-south1-a",
        "europe-west4-a",
        "europe-west4-b",
        "asia-northeast1-b",
    }
)

#: Label key marking a QR as owned by an eray capacity pool; the value is the
#: pool name. Pool membership is rediscovered from this label (cloud state is
#: the source of truth), so a crashed loop resumes with zero local state.
CAPACITY_POOL_LABEL = "eray-capacity-pool"
#: Label key recording the capacity tier the QR was requested with.
CAPACITY_TYPE_LABEL = "eray-capacity-type"

# GCP label key/value constraints (ported from iris).
_LABEL_KEY_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
_LABEL_VALUE_RE = re.compile(r"^[a-z0-9_-]{0,63}$")

# GCP resource name constraints (ported from iris).
_RESOURCE_NAME_RE = re.compile(r"^[a-z]([a-z0-9-]*[a-z0-9])?$")
MAX_RESOURCE_NAME_LENGTH = 63


# ============================================================================
# Data types (ported from iris, QR-only surface)
# ============================================================================


@dataclass(frozen=True)
class TpuCreateRequest:
    """Parameters for creating a TPU slice via a queued resource.

    Attributes:
        name: Queued-resource id; also the TPU node id (``node_id`` overrides).
        zone: GCP zone.
        accelerator_type: e.g. ``"v5p-128"``.
        runtime_version: TPU VM runtime version, e.g. ``"v2-alpha-tpuv5"``.
        capacity_type: Capacity tier; selects the QR scheduling block
            (``spot`` / ``guaranteed`` / plain on-demand queueing).
        labels: Node labels (also used for pool rediscovery).
        metadata: TPU VM metadata key/values.
        node_id: TPU node name; defaults to ``name``.
        service_account: Service-account email to attach to the node.
        network: VPC network; GCP defaults to ``"default"`` when None.
        subnetwork: VPC subnetwork.
        enable_external_ip: Allocate external IPs (False → internal only).
        valid_until_duration: Optional auto-expiry for the request, e.g.
            ``"72h"`` — the QR FAILs instead of queueing forever.
    """

    name: str
    zone: str
    accelerator_type: str
    runtime_version: str
    capacity_type: CapacityType = CapacityType.SPOT
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    node_id: str | None = None
    service_account: str | None = None
    network: str | None = None
    subnetwork: str | None = None
    enable_external_ip: bool = True
    valid_until_duration: str | None = None

    def resolved_node_id(self) -> str:
        """The TPU node id to request (explicit or the QR name)."""
        return self.node_id or self.name


@dataclass(frozen=True)
class QueuedResourceInfo:
    """Status of a GCP queued resource, as observed via describe/list.

    Attributes:
        name: Queued-resource id (short name).
        state: Lifecycle state (``WAITING_FOR_RESOURCES``, ``ACTIVE``, ...).
        zone: GCP zone the QR lives in.
        labels: Node labels, when reported.
        create_time: Creation time as a Unix epoch (seconds), or None when
            not reported. Drives the stuck-in-``WAITING_FOR_RESOURCES``
            zone-rotation timer.
        accelerator_type: Requested accelerator type, when reported.
        node_ids: TPU node ids attached to this QR, when reported.
    """

    name: str
    state: str
    zone: str = ""
    labels: dict[str, str] | None = None
    create_time: float | None = None
    accelerator_type: str | None = None
    node_ids: tuple[str, ...] = ()

    @property
    def is_live(self) -> bool:
        """True while the QR is holding or acquiring capacity."""
        return self.state in QR_LIVE_STATES

    @property
    def is_dead(self) -> bool:
        """True when the QR will never progress to ACTIVE on its own."""
        return self.state in QR_DEAD_STATES


# ============================================================================
# Validation (ported from iris)
# ============================================================================


def validate_resource_name(name: str, resource_kind: str) -> None:
    """Validate a GCP resource name (lowercase alphanumeric/hyphens, ≤63).

    Args:
        name: Candidate resource name.
        resource_kind: Human label for error messages ("queued resource").

    Raises:
        ValueError: On length or character violations.
    """
    if len(name) > MAX_RESOURCE_NAME_LENGTH:
        raise ValueError(f"{resource_kind} name exceeds {MAX_RESOURCE_NAME_LENGTH} chars: {name!r}")
    if not _RESOURCE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid {resource_kind} name (must be lowercase alphanumeric/hyphens, start with letter): {name!r}"
        )


def validate_labels(labels: dict[str, str]) -> None:
    """Validate GCP label keys/values.

    Args:
        labels: Label mapping to validate.

    Raises:
        ValueError: On the first invalid key or value.
    """
    for key, val in labels.items():
        if not _LABEL_KEY_RE.match(key):
            raise ValueError(f"Invalid label key: {key!r}")
        if not _LABEL_VALUE_RE.match(val):
            raise ValueError(f"Invalid label value for {key!r}: {val!r}")


def validate_zone(zone: str, valid_zones: set[str] | frozenset[str]) -> None:
    """Validate a zone against the known-TPU-zones set.

    Args:
        zone: Zone name to check.
        valid_zones: Acceptable zones (typically the service's mutable copy
            of :data:`KNOWN_GCP_ZONES`).

    Raises:
        InfraError: If the zone is not in the set.
    """
    if zone not in valid_zones:
        raise InfraError(f"Zone {zone!r} not available (known TPU zones: {sorted(valid_zones)})")


def validate_tpu_create(
    request: TpuCreateRequest,
    valid_zones: set[str] | frozenset[str],
    valid_types: set[str] | frozenset[str],
) -> None:
    """Validate a queued-resource create request before any API call.

    Args:
        request: The create request.
        valid_zones: Acceptable zones.
        valid_types: Acceptable accelerator type names.

    Raises:
        ValueError: On invalid name, labels, or empty runtime version.
        InfraError: On an unknown zone.
        ResourceNotFoundError: On an unknown accelerator type.
    """
    validate_resource_name(request.name, "queued resource")
    validate_resource_name(request.resolved_node_id(), "TPU node")
    validate_zone(request.zone, valid_zones)
    if request.accelerator_type not in valid_types:
        raise ResourceNotFoundError(f"Unknown accelerator type: {request.accelerator_type!r}")
    if not request.runtime_version:
        raise ValueError("runtime_version must be non-empty")
    validate_labels(request.labels)


def generate_qr_suffix() -> str:
    """Generate a unique queued-resource name suffix: ``YYYYMMDD-HHMM-<hex6>``.

    Pool slices get fresh names on every create (ported iris convention):
    identity lives in labels, so a replacement never has to wait for its dead
    predecessor's name to free up.

    Returns:
        The suffix string (length 20 with the joining hyphen accounted for
        by callers).
    """
    now = datetime.datetime.now(datetime.UTC)
    return f"{now.strftime('%Y%m%d-%H%M')}-{uuid.uuid4().hex[:6]}"


def parse_duration_s(text: str) -> float:
    """Parse a compact duration string like ``"72h"``, ``"30m"``, ``"3600s"``.

    Args:
        text: Number with a ``s``/``m``/``h``/``d`` suffix (no suffix means
            seconds).

    Returns:
        The duration in seconds.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    text = text.strip().lower()
    multipliers = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    if text and text[-1] in multipliers:
        return float(text[:-1]) * multipliers[text[-1]]
    return float(text)


# ============================================================================
# TPU topology table (ported from iris.cluster.tpu_topology)
# ============================================================================


@dataclass(frozen=True)
class TpuTopologyInfo:
    """TPU topology configuration.

    Attributes:
        name: Accelerator type name, e.g. ``"v5p-128"``.
        chip_count: Total TPU chips in the slice.
        host_count: Physical hosts in the slice.
        vm_count: Worker VMs in the slice.
        chips_per_vm: Chips visible to each worker VM.
    """

    name: str
    chip_count: int
    host_count: int
    vm_count: int
    chips_per_vm: int


TPU_TOPOLOGIES: tuple[TpuTopologyInfo, ...] = (
    # https://cloud.google.com/tpu/docs/v4
    TpuTopologyInfo("v4-8", 4, 1, 1, 4),
    TpuTopologyInfo("v4-16", 8, 2, 2, 4),
    TpuTopologyInfo("v4-32", 16, 4, 4, 4),
    TpuTopologyInfo("v4-64", 32, 8, 8, 4),
    TpuTopologyInfo("v4-128", 64, 16, 16, 4),
    TpuTopologyInfo("v4-256", 128, 32, 32, 4),
    TpuTopologyInfo("v4-512", 256, 64, 64, 4),
    TpuTopologyInfo("v4-1024", 512, 128, 128, 4),
    TpuTopologyInfo("v4-2048", 1024, 256, 256, 4),
    TpuTopologyInfo("v4-4096", 2048, 512, 512, 4),
    # https://cloud.google.com/tpu/docs/v5e
    TpuTopologyInfo("v5litepod-1", 1, 1, 1, 1),
    TpuTopologyInfo("v5litepod-2", 2, 1, 1, 2),
    TpuTopologyInfo("v5litepod-4", 4, 1, 1, 4),
    TpuTopologyInfo("v5litepod-8", 8, 1, 1, 8),
    TpuTopologyInfo("v5litepod-16", 16, 2, 4, 4),
    TpuTopologyInfo("v5litepod-32", 32, 4, 8, 4),
    TpuTopologyInfo("v5litepod-64", 64, 8, 16, 4),
    TpuTopologyInfo("v5litepod-128", 128, 16, 32, 4),
    TpuTopologyInfo("v5litepod-256", 256, 32, 64, 4),
    # https://cloud.google.com/tpu/docs/v5p
    TpuTopologyInfo("v5p-8", 4, 1, 1, 4),
    TpuTopologyInfo("v5p-16", 8, 2, 2, 4),
    TpuTopologyInfo("v5p-32", 16, 4, 4, 4),
    TpuTopologyInfo("v5p-64", 32, 8, 8, 4),
    TpuTopologyInfo("v5p-128", 64, 16, 16, 4),
    TpuTopologyInfo("v5p-256", 128, 32, 32, 4),
    TpuTopologyInfo("v5p-512", 256, 64, 64, 4),
    TpuTopologyInfo("v5p-1024", 512, 128, 128, 4),
    TpuTopologyInfo("v5p-2048", 1024, 256, 256, 4),
    TpuTopologyInfo("v5p-4096", 2048, 512, 512, 4),
    TpuTopologyInfo("v5p-8192", 4096, 1024, 1024, 4),
    TpuTopologyInfo("v5p-12288", 6144, 1536, 1536, 4),
    # https://cloud.google.com/tpu/docs/v6e
    TpuTopologyInfo("v6e-1", 1, 1, 1, 1),
    TpuTopologyInfo("v6e-4", 4, 1, 1, 4),
    TpuTopologyInfo("v6e-8", 8, 1, 1, 8),
    TpuTopologyInfo("v6e-16", 16, 4, 4, 4),
    TpuTopologyInfo("v6e-32", 32, 8, 8, 4),
    TpuTopologyInfo("v6e-64", 64, 16, 16, 4),
    TpuTopologyInfo("v6e-128", 128, 32, 32, 4),
    TpuTopologyInfo("v6e-256", 256, 64, 64, 4),
)

#: Accelerator type names derived from the topology registry.
KNOWN_TPU_TYPES: frozenset[str] = frozenset(t.name for t in TPU_TOPOLOGIES)


def get_tpu_topology(tpu_type: str) -> TpuTopologyInfo:
    """Get TPU topology by accelerator type name.

    Args:
        tpu_type: e.g. ``"v5p-128"``.

    Returns:
        The matching :class:`TpuTopologyInfo`.

    Raises:
        ValueError: If the type is unknown.
    """
    for config in TPU_TOPOLOGIES:
        if config.name == tpu_type:
            return config
    raise ValueError(f"Unknown TPU type: {tpu_type}")
