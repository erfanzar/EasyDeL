# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pydantic v2 request and response schemas for the Eyvan HTTP API."""

from __future__ import annotations

import collections
import ipaddress
import re
import socket
import typing as tp
from datetime import datetime
from typing import Annotated
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from eyvan._internal.compat import StrEnum
from eyvan.capacity.hardware_catalog import validate_accelerator_write_boundary
from eyvan.capacity.providers.gcp.quota_discovery import TpuQuotaAvailability
from eyvan.capacity.providers.gcp.tpu_types import parse_tpu_accelerator_type
from eyvan.records import (
    DEFAULT_MAX_RETRIES,
    AvailabilityPolicy,
    BudgetRecord,
    CallableExecution,
    CanonicalCapacityClass,
    CapacityClass,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    JsonDocument,
    NativeRuntimeConfig,
    OrganizationRecord,
    Permission,
    PolicyRecord,
    PriorityBand,
    ProjectRecord,
    QuotaRecord,
    ResourceLimits,
    Role,
    RoleGrant,
    RoleGrantRecord,
    ServiceProbeOutcome,
    ServiceReplicaHealthState,
    TpuChipRequest,
    WarmWindow,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadSetup,
    WorkspaceRef,
    validate_secret_mount_contract,
)
from eyvan.server.api.server_startup import ServerControllerMode
from eyvan.server.service.batch import DEFAULT_MAX_FAN_OUT
from eyvan.server.service.errors import ServiceErrorCode
from eyvan.server.service.services import ServicePolicy


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PlatformAgentNotConfiguredError(ValueError):
    """Raised when a pool omits routing and the platform has no default agent."""


class WorkloadSubmitRequest(ApiModel):
    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    spec: WorkloadDefinition
    #: Optional parent in a job tree. The child inherits the parent's scope and
    #: is cancelled with it; it does not become a fair-share level of its own.
    parent_workload_id: str | None = Field(default=None, max_length=96)


class BatchSubmitRequest(ApiModel):
    """Fan out one template into ``count`` ordinary durable workloads."""

    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    template: WorkloadDefinition
    count: int = Field(ge=1)
    #: Task names become ``{name_prefix}-{index}``. Defaults to the template's
    #: own name.
    name_prefix: str = Field(default="", max_length=180)
    #: Per-request ceiling, never above the server's. A caller can only lower
    #: it -- raising the fan-out limit is a deployment decision, not a
    #: request-body one.
    max_fan_out: int = Field(default=DEFAULT_MAX_FAN_OUT, ge=1, le=DEFAULT_MAX_FAN_OUT)


class GroupRoleRequest(ApiModel):
    """One named shape within a group, and how many of it the group needs."""

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]*$")
    template: WorkloadDefinition
    count: int = Field(default=1, ge=1, le=DEFAULT_MAX_FAN_OUT)


class GroupSubmitRequest(ApiModel):
    """Submit differently-shaped workloads that must admit together."""

    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    #: At least two, because a single shape is a batch and batch admits its tasks
    #: independently with no gate to evaluate each cycle.
    roles: tuple[GroupRoleRequest, ...] = Field(min_length=2)
    name_prefix: str = Field(default="", max_length=180)
    max_members: int = Field(default=DEFAULT_MAX_FAN_OUT, ge=2, le=DEFAULT_MAX_FAN_OUT)


class GroupSubmitResponse(ApiModel):
    """The coordinator plus its gated members."""

    parent: WorkloadRecord
    members: list[WorkloadRecord]


class ProgressReportRequest(ApiModel):
    """One rank's self-reported progress.

    Authenticated by the artifact token, not a principal: the caller is a running
    attempt, so there is no ``Principal`` to attribute -- the same reason the
    result-upload route authenticates the way it does.
    """

    rank: int = Field(ge=0)
    attempt: int = Field(ge=1)
    step: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=0)
    message: str = Field(default="", max_length=512)
    metrics: dict[str, float] = Field(default_factory=dict)


class ArtifactInventoryReindexRequest(ApiModel):
    """Explicit additive repair for one previously inspected scope page."""

    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    cursor: str | None = Field(default=None, min_length=1, max_length=512)
    limit: int = Field(default=100, ge=1, le=1000)
    #: Must exactly repeat ``org_id/project_id``. The redundancy is deliberate:
    #: a mutating flag copied from a dry-run command must not silently target a
    #: context-selected tenant the operator did not read.
    acknowledge_scope: str = Field(min_length=3, max_length=257)


class ArtifactHoldRequest(ApiModel):
    """Place an explicit human claim on one scoped indexed artifact."""

    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    uri: str = Field(min_length=1, max_length=512)
    reason: str = Field(min_length=1, max_length=512)
    #: ``None`` is an indefinite hold, which is permitted: the honest lifetime of
    #: "cited in a paper" is indefinite, and an invented expiry would delete the
    #: object on a date nobody chose.
    expires_at: datetime | None = None


class WorkloadEstimateRequest(ApiModel):
    """Ask what one workload definition would cost, without submitting it."""

    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    spec: WorkloadDefinition


class DagNodeRequest(ApiModel):
    """One named stage and the sibling stages it waits for."""

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]*$")
    template: WorkloadDefinition
    #: Names of sibling nodes in this same request. Not workload ids: that
    #: restriction is what makes cycles detectable at submission and keeps every
    #: edge inside one scope.
    depends_on: tuple[str, ...] = ()


class DagSubmitRequest(ApiModel):
    """Submit a small pipeline of dependent workloads in one transaction."""

    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    nodes: tuple[DagNodeRequest, ...] = Field(min_length=1)
    name_prefix: str = Field(default="", max_length=180)
    max_fan_out: int = Field(default=DEFAULT_MAX_FAN_OUT, ge=1, le=DEFAULT_MAX_FAN_OUT)


class BatchSubmitResponse(ApiModel):
    """The coordinator plus its tasks, in task order."""

    #: Not a task: it holds the array and is never scheduled. Cancel it to
    #: cancel every task.
    parent: WorkloadRecord
    tasks: list[WorkloadRecord]


class DagSubmitResponse(ApiModel):
    """The coordinator plus its nodes, in topological order."""

    parent: WorkloadRecord
    nodes: list[WorkloadRecord]


class PoolWorkloadSubmitRequest(ApiModel):
    """Command intent whose infrastructure contract is derived from a fleet node type."""

    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    command: str = Field(min_length=1)
    #: How many whole cells this job occupies at once, allocated atomically.
    #: The multislice knob, and it was pinned to one on this path while the
    #: callable path exposed it -- so a command could not ask for the shape the
    #: decorator could, and every multislice run had to be written in Python or
    #: submitted through the advanced API.
    pods: int = Field(default=1, ge=1)
    #: A constraint rather than a preference: with it set only a pool in that
    #: zone may run this job, and cross-zone alternates are not offered. For
    #: work pinned to data, to a reservation, or to a peer already running
    #: there. Without it the fleet chooses, which is what you usually want.
    zone: str | None = Field(default=None, min_length=1, max_length=128)
    name: str | None = Field(default=None, min_length=1, max_length=200)
    environment: dict[str, str] = Field(default_factory=dict)
    #: Secret *references*, never values. Unlike ``environment``, which is
    #: persisted in the spec, these are resolved at dispatch and the value never
    #: enters a record.
    secret_mounts: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    #: Packages installed on the host before the command runs, layered over
    #: whatever the image provides. The missing half of a per-job runtime
    #: environment: the working tree and the variables could already travel with
    #: a job, while adding one library meant rebuilding an image.
    dependencies: tuple[str, ...] = ()
    max_retries: int = Field(default=DEFAULT_MAX_RETRIES, ge=0)
    max_failure_retries: int = Field(default=0, ge=0)
    max_preemption_retries: int = Field(default=10, ge=0)
    #: Working tree to materialize on the host before the command runs. Uploaded
    #: separately to the content-addressed workspace route, then referenced here.
    workspace: WorkspaceRef | None = None
    #: Whether this job can be checkpointed, and the hook that does it. A
    #: property of the job rather than of the pool it names, which is why they
    #: are here and not derived: the scheduler grants ``ON_PREEMPTION`` only when
    #: both are set, so a spot job missing them restarts from zero on every
    #: reclamation no matter how many preemption retries it is allowed.
    checkpointable: bool = False
    checkpoint_hook: str | None = Field(default=None, max_length=4096)
    checkpoint_output_uri: str | None = Field(default=None, max_length=2048)
    #: Whether this job may submit children of its own, under its owner's
    #: identity. Same capability the explicit-intent path exposes.
    can_submit_children: bool = False
    #: Scheduling intent, which the pool cannot supply. Defaults to the
    #: record's own default, so a long training run submitted here competed
    #: above batch work with no way to declare itself batch.
    priority_band: PriorityBand = PriorityBand.INTERACTIVE
    intra_band_priority: int = Field(default=0, ge=0)
    #: Seconds the checkpoint hook is given before the job is stopped.
    checkpoint_grace_seconds: int = Field(default=0, ge=0)
    #: Settlement policy when reclaimed: requeue for resume or stop with checkpoint preserved.
    reclaim_policy: tp.Literal["requeue", "stop"] = "requeue"
    #: How long the job is expected to run. Every cost figure is an estimate
    #: over this number, so a pool submission that could not state it was
    #: estimated at the default hour whatever it actually was.
    estimated_duration_seconds: int = Field(default=3600, ge=0)
    #: Logical names of large inputs this job pulls. A hint only: cells that
    #: recently ran something naming the same asset are preferred.
    named_assets: tuple[str, ...] = ()
    #: Run on some chips of one host of a chip-sharing pool instead of the
    #: whole slice. Omitted from the wire document while unset.
    tpu_chips: TpuChipRequest | None = None
    #: A bash step run on every host before the command, inside the job's
    #: environment. Omitted from the wire document while unset.
    setup: WorkloadSetup | None = None

    @model_serializer(mode="wrap")
    def _wire_document(self, handler):
        document = handler(self)
        if self.reclaim_policy == "requeue":
            document.pop("reclaim_policy", None)
        if self.tpu_chips is None:
            document.pop("tpu_chips", None)
        if self.setup is None:
            document.pop("setup", None)
        return document

    @model_validator(mode="after")
    def validate_secret_mounts(self) -> PoolWorkloadSubmitRequest:
        """Reject mount names the resulting workload environment cannot hold."""

        validate_secret_mount_contract(self.environment, self.secret_mounts)
        return self


class PoolCallableSubmitRequest(ApiModel):
    """Callable intent whose TPU placement is derived from an enabled Spot fleet pool."""

    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    accelerator_type: str = Field(min_length=1, max_length=128)
    pods: int = Field(default=1, ge=1)
    #: Require one exact-shape allocation (one hardware world spanning every
    #: pod) instead of the default elastic packing across warm same-type cells
    #: with only the deficit provisioned.
    atomic: bool = False
    #: Run on some chips of one host of a chip-sharing pool instead of the
    #: whole slice. Omitted by the SDK when unset, so whole-slice requests are
    #: unchanged on the wire.
    tpu_chips: TpuChipRequest | None = None
    capacity_class: CanonicalCapacityClass = CapacityClass.SPOT
    execution: CallableExecution
    name: str | None = Field(default=None, min_length=1, max_length=200)
    environment: dict[str, str] = Field(default_factory=dict)
    #: Packages installed on the host before the callable worker starts, layered
    #: over what the image provides. The wrapper `exec`s into the worker, so the
    #: pickled callable's imports resolve against them.
    dependencies: tuple[str, ...] = ()
    #: Secret *references*, never values. Unlike ``environment``, which is
    #: persisted in the spec, these are resolved at dispatch and the value never
    #: enters a record.
    secret_mounts: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    max_retries: int = Field(default=DEFAULT_MAX_RETRIES, ge=0)
    max_failure_retries: int = Field(default=0, ge=0)
    max_preemption_retries: int = Field(default=10, ge=0)
    #: The job's own checkpoint contract; see PoolWorkloadSubmitRequest. It
    #: matters most here, because this path defaults to Spot.
    checkpointable: bool = False
    checkpoint_hook: str | None = Field(default=None, max_length=4096)
    checkpoint_output_uri: str | None = Field(default=None, max_length=2048)
    #: Whether this job may submit children of its own, under its owner's
    #: identity. Same capability the explicit-intent path exposes.
    can_submit_children: bool = False
    #: Scheduling intent, which the pool cannot supply. Defaults to the
    #: record's own default, so a long training run submitted here competed
    #: above batch work with no way to declare itself batch.
    priority_band: PriorityBand = PriorityBand.INTERACTIVE
    intra_band_priority: int = Field(default=0, ge=0)
    #: Seconds the checkpoint hook is given before the job is stopped.
    checkpoint_grace_seconds: int = Field(default=0, ge=0)
    #: Settlement policy when reclaimed: requeue for resume or stop with checkpoint preserved.
    reclaim_policy: tp.Literal["requeue", "stop"] = "requeue"
    #: How long the job is expected to run. Every cost figure is an estimate
    #: over this number, so a pool submission that could not state it was
    #: estimated at the default hour whatever it actually was.
    estimated_duration_seconds: int = Field(default=3600, ge=0)
    #: Logical names of large inputs this job pulls. A hint only: cells that
    #: recently ran something naming the same asset are preferred.
    named_assets: tuple[str, ...] = ()
    #: A bash step run on every host before the callable loads, inside the
    #: job's environment. Omitted by the SDK when unset.
    setup: WorkloadSetup | None = None

    @model_validator(mode="after")
    def validate_secret_mounts(self) -> PoolCallableSubmitRequest:
        """Reject mount names the resulting workload environment cannot hold."""

        validate_secret_mount_contract(self.environment, self.secret_mounts)
        return self


class PreemptionForceFinalizeRequest(ApiModel):
    reason: str = Field(min_length=1, max_length=1000)


class CellUpsertRequest(ApiModel):
    id: str = Field(min_length=1, max_length=96)
    provider: str = Field(min_length=1)
    capacity_backend: str | None = None
    provider_project: str | None = None
    region: str = Field(min_length=1)
    zone: str | None = None
    state: CellState = CellState.REGISTERING
    capacity: ResourceLimits
    allocatable: ResourceLimits
    unit_prices: dict[str, float] = Field(default_factory=dict)
    accelerator_type: str | None = None
    capacity_class: CanonicalCapacityClass = CapacityClass.ON_DEMAND
    executor: str = "eyvan-native"
    endpoint: str | None = None
    pool_id: str | None = None
    provider_options: JsonDocument = Field(default_factory=dict)
    runtime: NativeRuntimeConfig = Field(default_factory=NativeRuntimeConfig)
    #: Carried from the pool that created this cell, so an observation about
    #: it needs nothing but the cell itself.
    availability_policy: AvailabilityPolicy = Field(default_factory=AvailabilityPolicy)
    #: How long the provider holds this capacity before taking it back.
    #: Required for a flex_start pool; the record refuses one without it.
    max_hold_s: int | None = Field(default=None, ge=60)
    labels: dict[str, str] = Field(default_factory=dict)
    topology: dict[str, str] = Field(default_factory=dict)
    max_replica_resources: ResourceLimits | None = None
    max_gang_replicas: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_accelerator_semantics(self) -> CellUpsertRequest:
        """Reject impossible accelerator inventory before it becomes durable."""

        validate_accelerator_write_boundary(
            provider=self.provider,
            capacity_backend=self.capacity_backend,
            accelerator_type=self.accelerator_type,
            capacity=self.capacity,
            allocatable=self.allocatable,
            boundary="accelerator cell",
        )
        return self

    def to_record(self) -> CellRecord:
        return CellRecord(**self.model_dump())


class CapacityPoolUpsertRequest(ApiModel):
    id: str = Field(min_length=1, max_length=192)
    fleet_name: str = Field(min_length=1, max_length=96)
    node_type: str = Field(min_length=1, max_length=96)
    provider: str = Field(min_length=1)
    capacity_backend: str | None = None
    provider_project: str | None = None
    provider_account: str | None = Field(default=None, max_length=320)
    region: str = Field(min_length=1)
    zone: str | None = None
    capacity: ResourceLimits
    allocatable: ResourceLimits
    unit_prices: dict[str, float] = Field(default_factory=dict)
    accelerator_type: str | None = None
    capacity_class: CanonicalCapacityClass = CapacityClass.ON_DEMAND
    executor: str = "eyvan-native"
    agent_endpoint: str | None = Field(default=None, min_length=1)
    provider_options: JsonDocument = Field(default_factory=dict)
    runtime: NativeRuntimeConfig = Field(default_factory=NativeRuntimeConfig)
    #: How this pool backs off when the provider refuses it, and how fast it
    #: forgets. Declared in fleet YAML and carried to the record, so a curve
    #: an operator wrote is the curve the planner uses.
    availability_policy: AvailabilityPolicy = Field(default_factory=AvailabilityPolicy)
    #: How long the provider holds this capacity before taking it back.
    #: Required for a flex_start pool; the record refuses one without it.
    max_hold_s: int | None = Field(default=None, ge=60)
    labels: dict[str, str] = Field(default_factory=dict)
    topology: dict[str, str] = Field(default_factory=dict)
    max_replica_resources: ResourceLimits | None = None
    max_gang_replicas: int | None = Field(default=None, ge=1)
    #: Let chip-sized workloads share this pool's cells. Omitted from the wire
    #: document while false, so applies to an older plane are unchanged.
    chip_sharing: bool = False
    quota_group: str | None = Field(default=None, max_length=192)
    quota_limit: int | None = Field(default=None, ge=0)
    quota_units_per_cell: int | None = Field(default=None, ge=1)
    min_cells: int = Field(default=0, ge=0)
    #: Recurring windows during which the pool holds a raised cell floor.
    #:
    #: Honoured by the planner since ``WarmWindow`` was written; until now every
    #: writer was a test, so the feature existed and no operator could reach it.
    warm_windows: tuple[WarmWindow, ...] = ()
    initial_cells: int | None = Field(default=None, ge=0)
    max_cells: int = Field(default=1, ge=0, le=10_000)
    idle_timeout_s: float = Field(default=3600, ge=0)
    queue_timeout_s: float | None = Field(default=None, gt=0)
    provisioning_timeout_s: float = Field(default=1800, gt=0)
    upscaling_speed: float = Field(default=1.0, gt=0)
    launch_batch: int = Field(default=5, ge=1)
    enabled: bool = True

    @model_serializer(mode="wrap")
    def _wire_document(self, handler):
        document = handler(self)
        if not self.chip_sharing:
            document.pop("chip_sharing", None)
        if self.provider_account is None:
            document.pop("provider_account", None)
        return document

    @model_validator(mode="after")
    def validate_accelerator_semantics(self) -> CapacityPoolUpsertRequest:
        """Reject impossible accelerator templates before they become durable.

        ``CapacityPoolRecord`` deliberately remains able to decode legacy rows;
        direct pool upserts and fleet apply both pass through this request model.
        """

        validate_accelerator_write_boundary(
            provider=self.provider,
            capacity_backend=self.capacity_backend,
            accelerator_type=self.accelerator_type,
            capacity=self.capacity,
            allocatable=self.allocatable,
            boundary="accelerator pool",
        )
        return self

    def to_record(self, *, default_agent_endpoint: str | None = None) -> CapacityPoolRecord:
        """Resolve the platform default agent before persisting a pool."""

        agent_endpoint = self.agent_endpoint or default_agent_endpoint
        if agent_endpoint is None:
            raise PlatformAgentNotConfiguredError("no default cell agent is configured by the Eyvan platform")
        return CapacityPoolRecord(**self.model_dump(exclude={"agent_endpoint"}), agent_endpoint=agent_endpoint)


class WebhookCreateRequest(ApiModel):
    """Where a project wants its lifecycle events pushed."""

    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    url: str = Field(min_length=1, max_length=2048)
    #: The key deliveries are signed with. Supplied rather than generated so the
    #: receiver can be configured with it first: a secret this plane invents has
    #: to be read back out of a record that deliberately never returns it.
    secret: str = Field(min_length=16, max_length=256)
    #: Empty means every event, which is the useful default. A filter that has to
    #: be right before anything arrives is one most people get wrong once.
    entity_types: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()


class FleetApplyRequest(ApiModel):
    pools: tuple[CapacityPoolUpsertRequest, ...] = Field(min_length=1)

    def to_records(self, *, default_agent_endpoint: str | None = None) -> tuple[CapacityPoolRecord, ...]:
        """Resolve the platform default agent for every fleet pool."""

        return tuple(pool.to_record(default_agent_endpoint=default_agent_endpoint) for pool in self.pools)


class FleetStatusResponse(ApiModel):
    """Server-side pool and cell state for one applied fleet."""

    name: str
    pools: list[CapacityPoolRecord]
    cells: list[CellRecord]


class FleetInitScriptRequest(ApiModel):
    """A fleet's bash init script; an empty string removes it."""

    # Bounded again (in bytes) by the service; this caps the request body.
    script: str = Field(default="", max_length=64 * 1024)


class CapacityPoolResponse(ApiModel):
    """One available pool advertised by the control plane."""

    fleet: str
    pool: str
    accelerator_type: str
    capacity_class: CanonicalCapacityClass
    region: str
    zone: str | None
    min_cells: int
    max_cells: int
    enabled: bool


class CapacityResponse(ApiModel):
    """Advertised capacity inventory across all applied fleets."""

    pools: list[CapacityPoolResponse]


class TpuQuotaState(StrEnum):
    """What one quota row is actually able to say about free capacity.

    ``USAGE_UNKNOWN`` is the reason this enum exists. Cloud Quotas reports a
    limit for a zone whether or not Cloud Monitoring has a matching usage
    series, and a row that renders a missing usage as zero claims the entire
    quota is free -- the most expensive wrong answer this data can give. The
    state is carried on the wire so no renderer has to infer it from a null.
    """

    KNOWN = "known"
    USAGE_UNKNOWN = "usage_unknown"
    UNLIMITED = "unlimited"


class TpuQuotaStatus(StrEnum):
    """Whether a snapshot describes provider truth, and why not when it does not."""

    OK = "ok"
    UNAVAILABLE = "unavailable"
    UNCONFIGURED = "unconfigured"


def _quota_region(zone: str) -> str:
    """Return the region prefix of a provider zone, or the zone when it has none."""

    region, separator, suffix = zone.rpartition("-")
    return region if separator and region and suffix else zone


class TpuQuotaEntry(ApiModel):
    """One zone and accelerator type's quota ceiling, usage, and free balance.

    Chips are Cloud TPU quota units, which use the public accelerator suffix:
    one ``v6e-8`` slice consumes eight units. ``free_slices`` is that division
    already done, because the question an operator actually has is "how many of
    these fit", not "how many units are left".
    """

    zone: str = Field(min_length=1, max_length=128)
    region: str = Field(min_length=1, max_length=128)
    generation: str = Field(min_length=1, max_length=64)
    accelerator_type: str = Field(min_length=1, max_length=128)
    capacity_class: CanonicalCapacityClass
    metric: str = Field(min_length=1, max_length=256)
    chips_per_slice: int = Field(ge=1)
    #: ``None`` means the provider reports no ceiling, never "zero".
    limit_chips: int | None = Field(default=None, ge=0)
    #: Derived as ``limit - free``: the balance is what the provider observes.
    in_use_chips: int | None = Field(default=None, ge=0)
    free_chips: int | None = Field(default=None, ge=0)
    free_slices: int | None = Field(default=None, ge=0)
    state: TpuQuotaState
    usage_observed_at: datetime | None = None
    #: How old the usage observation was when this snapshot was assembled.
    usage_age_s: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_state(self) -> TpuQuotaEntry:
        """Keep an unknown or unlimited row from carrying an invented number."""

        counted = (self.in_use_chips, self.free_chips, self.free_slices)
        if self.state is TpuQuotaState.UNLIMITED:
            if self.limit_chips is not None:
                raise ValueError("an unlimited quota row must not carry a limit")
            if any(value is not None for value in counted) or self.usage_observed_at is not None:
                raise ValueError("an unlimited quota row must not carry a usage observation")
            return self
        if self.limit_chips is None:
            raise ValueError("a limited quota row requires a limit")
        if self.state is TpuQuotaState.USAGE_UNKNOWN:
            if any(value is not None for value in counted) or self.usage_observed_at is not None:
                raise ValueError("an unknown-usage quota row must not carry a usage number")
            return self
        if any(value is None for value in counted):
            raise ValueError("a known quota row requires usage and free balances")
        assert self.free_chips is not None and self.in_use_chips is not None
        if self.free_chips > self.limit_chips or self.in_use_chips > self.limit_chips:
            raise ValueError("a known quota row must not exceed its own limit")
        return self

    @classmethod
    def from_availability(
        cls,
        availability: TpuQuotaAvailability,
        *,
        capacity_class: CapacityClass,
        observed_at: datetime,
    ) -> TpuQuotaEntry:
        """Convert one provider observation without inventing a missing usage."""

        parsed = parse_tpu_accelerator_type(availability.accelerator_type)
        chips_per_slice = parsed.size if parsed is not None else 1
        common = {
            "zone": availability.zone,
            "region": _quota_region(availability.zone),
            "generation": availability.generation,
            "accelerator_type": availability.accelerator_type,
            "capacity_class": capacity_class,
            "metric": availability.metric,
            "chips_per_slice": chips_per_slice,
        }
        if availability.quota_tpus is None:
            return cls(**common, limit_chips=None, state=TpuQuotaState.UNLIMITED)
        if availability.available_tpus is None:
            return cls(**common, limit_chips=availability.quota_tpus, state=TpuQuotaState.USAGE_UNKNOWN)
        free = availability.available_tpus
        usage_observed_at = availability.usage_observed_at
        return cls(
            **common,
            limit_chips=availability.quota_tpus,
            in_use_chips=availability.quota_tpus - free,
            free_chips=free,
            free_slices=free // chips_per_slice,
            state=TpuQuotaState.KNOWN,
            usage_observed_at=usage_observed_at,
            usage_age_s=(
                None if usage_observed_at is None else max((observed_at - usage_observed_at).total_seconds(), 0.0)
            ),
        )


#: Maximum character length accepted on ``TpuQuotaFailure.reason``.
TPU_QUOTA_FAILURE_REASON_MAX_LENGTH = 2048

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_QUOTA_PERMISSION_PATTERNS = (
    re.compile(r"Permission\s+['\"]([a-z0-9_.-]+\.[a-z0-9_.-]+)['\"]\s+denied", re.IGNORECASE),
    re.compile(r"\bpermission\s*[:=]\s*['\"]?([a-z0-9_.-]+\.[a-z0-9_.-]+)['\"]?", re.IGNORECASE),
    re.compile(r"\bmissing\s+['\"]?([a-z0-9_.-]+\.[a-z0-9_.-]+)['\"]?", re.IGNORECASE),
)
_QUOTA_ACCOUNT_PATTERNS = (
    re.compile(r"authenticated as\s+['\"]?([^\s,;)'\"]+@[^\s,;)'\"]+)['\"]?", re.IGNORECASE),
    re.compile(r"active account(?:\s+specified by[^\n]*?)?\s+['\"]?([^\s,;)'\"]+@[^\s,;)'\"]+)['\"]?", re.IGNORECASE),
    re.compile(r"\baccount\s+['\"]([^\s'\"]+@[^\s'\"]+)['\"]", re.IGNORECASE),
)
_QUOTA_PROJECT_PATTERNS = (
    re.compile(r"projects/([a-z][a-z0-9-]{4,61}[a-z0-9])"),
    re.compile(r"--project[=\s]+['\"]?([a-z][a-z0-9-]{4,61}[a-z0-9])['\"]?"),
    re.compile(r"\bproject\s+['\"]([a-z][a-z0-9-]{4,61}[a-z0-9])['\"]", re.IGNORECASE),
)
_QUOTA_SERVICE_RE = re.compile(r"\b([a-z0-9-]+\.googleapis\.com)\b", re.IGNORECASE)
_NOISE_LINE_PREFIXES = (
    "- '@type':",
    "'@type':",
    '"@type":',
    "error_info_id:",
    "troubleshooter_url:",
    "domain:",
    "metadata:",
    "reason:",
    "detail:",
    "details:",
    "[original error]",
)
_INLINE_NOISE_RE = re.compile(
    r"(?:troubleshooter_url\S*|error_info_id\S*|type\.googleapis\.com/\S+|https?://console\.cloud\.google\.com/iam-admin/troubleshooter\S*|[A-Za-z0-9+/=_-]{48,})",
    re.IGNORECASE,
)


def _bound_text(value: str, limit: int) -> str:
    cleaned = " ".join(value.split()).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _extract_quota_permission(text: str) -> str | None:
    for pattern in _QUOTA_PERMISSION_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return _bound_text(match.group(1), 128)
    return None


def _extract_quota_account(text: str) -> str | None:
    for pattern in _QUOTA_ACCOUNT_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return _bound_text(match.group(1).rstrip("."), 256)
    return None


def _extract_quota_project(text: str, explicit_project: str | None) -> str | None:
    if explicit_project and explicit_project.strip():
        return explicit_project.strip()
    for pattern in _QUOTA_PROJECT_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return match.group(1)
    return None


def _role_for_quota_permission(permission: str | None) -> str:
    if permission:
        lowered = permission.lower()
        if lowered.startswith("monitoring."):
            return "roles/monitoring.viewer"
        if lowered.startswith("tpu."):
            return "roles/tpu.viewer"
    return "roles/cloudquotas.viewer"


def _clean_quota_fallback_summary(text: str) -> str:
    """Strip raw protobuf/ErrorInfo/base64/troubleshooter noise to one plain sentence."""

    plain = _ANSI_ESCAPE_RE.sub("", text)
    kept_lines: list[str] = []
    for raw_line in plain.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(lowered.startswith(prefix) for prefix in _NOISE_LINE_PREFIXES):
            break
        if "type.googleapis.com/" in lowered or "error_info_id" in lowered or "troubleshooter_url" in lowered:
            continue
        kept_lines.append(stripped)
        if len(" ".join(kept_lines)) >= 240:
            break
    candidate = " ".join(kept_lines) if kept_lines else plain.splitlines()[0]
    candidate = _INLINE_NOISE_RE.sub("", candidate)
    candidate = re.sub(r"^gcloud\s+beta\s+quotas\s+info\s+list\s+failed:\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^ERROR:\s*(?:\([^)]+\)\s*)?", "", candidate, flags=re.IGNORECASE)
    candidate = " ".join(candidate.split()).strip(" :-")
    if not candidate:
        return "Provider TPU quota lookup failed."
    if not candidate.endswith((".", "!", "?")):
        candidate = f"{candidate}."
    return _bound_text(candidate, 320)


def _diagnose_tpu_quota_failure(
    raw_reason: str,
    *,
    project: str | None = None,
) -> tuple[str, str, str | None, str | None, str | None]:
    """Classify one provider quota failure into structured, human-readable diagnostics.

    Returns:
        Tuple of ``(code, summary, permission, account, remediation)``.
    """

    plain = _ANSI_ESCAPE_RE.sub("", raw_reason)
    lowered = plain.lower()
    permission = _extract_quota_permission(plain)
    account = _extract_quota_account(plain)
    resolved_project = _extract_quota_project(plain, project)
    project_clause = f" on project {resolved_project}" if resolved_project else ""

    if (
        "no such file or directory" in lowered
        or "executable file not found" in lowered
        or "filenotfounderror" in lowered
        or "winerror 2" in lowered
        or "gcloud: command not found" in lowered
        or ("gcloud" in lowered and "not found" in lowered and "project" not in lowered and "location" not in lowered)
    ):
        return (
            "gcloud_missing",
            "The gcloud CLI is not installed or not on PATH on the control-plane host.",
            permission,
            account,
            "Install the Google Cloud CLI on the control-plane host and authenticate with: gcloud auth login",
        )

    if (
        "did not answer within" in lowered
        or "timed out" in lowered
        or "timeoutexpired" in lowered
        or "deadline_exceeded" in lowered
        or "connecttimeout" in lowered
        or "readtimeout" in lowered
    ):
        return (
            "timeout",
            _bound_text(
                f"TPU quota lookup{project_clause} timed out before GCP responded; "
                "a background read may still complete.",
                512,
            ),
            permission,
            account,
            "Retry in a few seconds, or filter to a single zone to narrow the provider read.",
        )

    if (
        "service_disabled" in lowered
        or "has not been used in project" in lowered
        or "is disabled" in lowered
        or "not enabled for project" in lowered
        or "enable it by visiting" in lowered
    ):
        service_match = _QUOTA_SERVICE_RE.search(plain)
        service = service_match.group(1) if service_match else "cloudquotas.googleapis.com"
        target_flag = f" --project={resolved_project}" if resolved_project else ""
        return (
            "api_disabled",
            _bound_text(f"Required GCP API ({service}) is disabled{project_clause}.", 512),
            permission,
            account,
            _bound_text(f"Enable the API: gcloud services enable {service}{target_flag}", 512),
        )

    if (
        "unauthenticated" in lowered
        or "no credentials" in lowered
        or "not logged in" in lowered
        or "reauthentication" in lowered
        or "invalid_grant" in lowered
        or "token lookup failed" in lowered
        or "could not obtain" in lowered
        or "credentials were not found" in lowered
        or "refresherror" in lowered
        or "gcloud auth login" in lowered
        or "answered 401" in lowered
        or "status 401" in lowered
    ):
        acct_clause = f" for {account}" if account else ""
        login_cmd = f"gcloud auth login {account}" if account else "gcloud auth login"
        return (
            "auth_required",
            _bound_text(
                f"GCP credentials{acct_clause} are missing or expired, "
                f"so Eyvan could not read TPU quotas{project_clause}.",
                512,
            ),
            permission,
            account,
            _bound_text(f"Sign in on the control-plane host with: {login_cmd}", 512),
        )

    if (
        "permission_denied" in lowered
        or "iam_permission_denied" in lowered
        or "does not have permission" in lowered
        or "caller does not have" in lowered
        or "is not authorized" in lowered
        or "answered 403" in lowered
        or "status 403" in lowered
        or permission is not None
    ):
        role = _role_for_quota_permission(permission)
        if account and permission:
            summary = f"Account {account} lacks '{permission}' permission to read TPU quotas{project_clause}."
        elif account:
            summary = f"Account {account} does not have permission to read TPU quotas{project_clause}."
        elif permission:
            summary = f"Missing '{permission}' permission to read TPU quotas{project_clause}."
        else:
            summary = f"Permission denied while reading TPU quotas{project_clause}."
        if resolved_project and account:
            member_kind = "serviceAccount" if account.endswith(".gserviceaccount.com") else "user"
            remediation = (
                f"Switch to an authorized project or account, or grant {role}: "
                f"gcloud projects add-iam-policy-binding {resolved_project} "
                f"--member={member_kind}:{account} --role={role}"
            )
        elif resolved_project:
            remediation = (
                f"Switch to an authorized project or grant {role} on {resolved_project}: "
                f"gcloud projects add-iam-policy-binding {resolved_project} "
                f"--member=user:<EMAIL> --role={role}"
            )
        else:
            remediation = f"Switch to an authorized project or gcloud account, or grant {role} on the target project."
        return (
            "permission_denied",
            _bound_text(summary, 512),
            permission,
            account,
            _bound_text(remediation, 512),
        )

    if (
        "not_found" in lowered
        or "unknown project" in lowered
        or "invalid project" in lowered
        or "project not found" in lowered
        or "does not exist" in lowered
        or "answered 404" in lowered
        or "status 404" in lowered
    ):
        proj_label = f"'{resolved_project}'" if resolved_project else "requested"
        return (
            "project_not_found",
            _bound_text(f"GCP project {proj_label} was not found or is inaccessible.", 512),
            permission,
            account,
            "Verify available GCP projects with: gcloud projects list",
        )

    fallback_remediation = (
        "Verify access directly with: "
        f"gcloud beta quotas info list --service=tpu.googleapis.com --project={resolved_project}"
        if resolved_project
        else "Verify gcloud authentication and network access to Cloud Quotas and Cloud Monitoring APIs."
    )
    return (
        "quota_read_failed",
        _clean_quota_fallback_summary(plain),
        permission,
        account,
        _bound_text(fallback_remediation, 512),
    )


class TpuQuotaFailure(ApiModel):
    """One purchase class whose provider read did not succeed.

    Reported beside the rows that did succeed rather than collapsing the whole
    snapshot: a fleet whose Spot usage is unreadable still has readable
    on-demand quota, and hiding one behind the other is how an operator ends up
    believing a zone is empty.
    """

    capacity_class: CanonicalCapacityClass
    reason: str = Field(min_length=1, max_length=TPU_QUOTA_FAILURE_REASON_MAX_LENGTH)
    #: Machine-readable failure classification for clean UI/CLI presentation.
    code: str = Field(default="quota_read_failed", min_length=1, max_length=64)
    #: Concise 1-sentence plain-English diagnosis stripped of protobuf noise.
    summary: str | None = Field(default=None, max_length=512)
    #: Missing IAM permission when extracted from the provider error.
    permission: str | None = Field(default=None, max_length=128)
    #: Active GCP account named by the provider error, if any.
    account: str | None = Field(default=None, max_length=256)
    #: Concrete actionable next step or CLI command for the operator.
    remediation: str | None = Field(default=None, max_length=512)

    @classmethod
    def from_reason(
        cls,
        *,
        capacity_class: CapacityClass,
        reason: str,
        project: str | None = None,
    ) -> TpuQuotaFailure:
        """Build a failure record, truncating raw provider text and populating structured diagnostics."""

        normalized = reason.strip() or "provider quota discovery failed"
        if len(normalized) > TPU_QUOTA_FAILURE_REASON_MAX_LENGTH:
            normalized = normalized[: TPU_QUOTA_FAILURE_REASON_MAX_LENGTH - 3].rstrip() + "..."
        code, summary, permission, account, remediation = _diagnose_tpu_quota_failure(normalized, project=project)
        return cls(
            capacity_class=capacity_class,
            reason=normalized,
            code=code,
            summary=summary,
            permission=permission,
            account=account,
            remediation=remediation,
        )


class TpuQuotaSnapshot(ApiModel):
    """Provider TPU quota, usage, and free balance for one project.

    ``refreshed_at`` is when the provider was read and ``observed_at`` is when
    this response was assembled; ``age_s`` is the distance between them. A
    cached snapshot therefore states its own age instead of being presented as
    live, which is the difference between "no capacity" and "nobody has looked
    recently".
    """

    observed_at: datetime
    provider: str = Field(default="gcp", min_length=1, max_length=64)
    project: str | None = Field(default=None, max_length=128)
    #: Every provider project this fleet's pools declare, so the caller can ask
    #: about a second one without reading pool records by hand.
    available_projects: list[str] = Field(default_factory=list)
    status: TpuQuotaStatus
    refreshed_at: datetime | None = None
    age_s: float | None = Field(default=None, ge=0)
    cached: bool = False
    ttl_s: float = Field(default=0.0, ge=0)
    entries: list[TpuQuotaEntry] = Field(default_factory=list)
    failures: list[TpuQuotaFailure] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_snapshot(self) -> TpuQuotaSnapshot:
        """Keep a failed or unconfigured snapshot from looking like an empty fleet."""

        if self.status is TpuQuotaStatus.UNAVAILABLE and not self.failures:
            raise ValueError("an unavailable quota snapshot must name at least one failure")
        if self.status is TpuQuotaStatus.UNCONFIGURED and (self.entries or self.project is not None):
            raise ValueError("an unconfigured quota snapshot describes no project")
        if self.entries and self.status is not TpuQuotaStatus.OK:
            raise ValueError("quota rows are only reported by an ok snapshot")
        return self


class PolicyPutRequest(ApiModel):
    allowed_providers: tuple[str, ...] = ()
    allowed_regions: tuple[str, ...] = ()
    allowed_capacity_classes: tuple[CanonicalCapacityClass, ...] = ()
    denied_accelerator_types: tuple[str, ...] = ()
    max_priority: int = Field(default=1000, ge=0, le=1000)
    #: Prefixes of the secret references this project's workloads may mount.
    #: Empty means *none*, unlike every other list here -- see PolicyRecord.
    allowed_secret_references: tuple[str, ...] = ()
    preemption_enabled: bool = True

    def to_record(self, org_id: str, project_id: str) -> PolicyRecord:
        return PolicyRecord(org_id=org_id, project_id=project_id, updated_by="api", **self.model_dump())


class WorkloadExecRequest(ApiModel):
    """One diagnostic command to run beside a running workload."""

    #: A list of arguments, never a shell string: a string would be split by a
    #: shell on the host, which is an injection surface as soon as any part of
    #: it is templated.
    command: tuple[str, ...] = Field(min_length=1)
    timeout_s: float = Field(default=30.0, gt=0, le=600)
    #: ``None`` runs on every rank, which is the default because the question
    #: exec answers is usually "which rank is stuck".
    rank: int | None = Field(default=None, ge=0)


#: Networks a workload-registered service endpoint must never resolve to. The
#: control plane GETs the endpoint itself -- ``HttpServiceHealthProbe`` fetches
#: ``endpoint + health_path`` from the leader's host -- so these name exactly
#: the targets a workload must not aim it at: the plane's own loopback, the
#: link-local range every cloud exposes its metadata service on, and the
#: unique-local range of the host's own private segments.
_ENDPOINT_FORBIDDEN_V4_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
)
_ENDPOINT_FORBIDDEN_V6_NETWORKS = (
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("fc00::/7"),
)


class ServiceEndpointRequest(ApiModel):
    """Where a replica says it can be reached."""

    endpoint: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_endpoint(self) -> ServiceEndpointRequest:
        """Refuse a target only the control plane's own host could answer."""

        parsed = urlparse(self.endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("service endpoint must be an http(s) URL with a hostname")
        try:
            addresses = [ipaddress.ip_address(parsed.hostname)]
        except ValueError:
            try:
                resolved = socket.getaddrinfo(parsed.hostname, None, type=socket.SOCK_STREAM)
            except OSError:
                # Unresolvable names remain registerable. The probe resolves and
                # pins its own checked destination before making a connection.
                return self
            addresses = [ipaddress.ip_address(entry[4][0]) for entry in resolved]
        for address in addresses:
            if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
                address = address.ipv4_mapped
            if address.is_unspecified:
                raise ValueError("service endpoint must not resolve to an unspecified address")
            forbidden = _ENDPOINT_FORBIDDEN_V4_NETWORKS if address.version == 4 else _ENDPOINT_FORBIDDEN_V6_NETWORKS
            if any(address in network for network in forbidden):
                raise ValueError("service endpoint must not resolve to a loopback, link-local, or unique-local address")
        return self


class ServiceReplicaHealth(ApiModel):
    """What is durably known about one replica, with UNKNOWN as its own answer.

    ``health`` has three values and the two counters are carried side by side,
    so an operator can tell "this replica answered that it is not serving" from
    "nobody could reach this replica". Collapsing those into a boolean is the
    defect the durable record exists to prevent, and a transport model that
    re-collapses them would undo it at the last hop.
    """

    workload_id: str
    generation: int = Field(ge=1)
    revision: str
    #: Where the replica registered itself, or ``None`` before it has.
    endpoint: str | None = None
    health: ServiceReplicaHealthState
    #: Whether any probe has ever completed against this replica. ``False`` is
    #: *not probed*: nothing was asked and nothing was learned, which is neither
    #: a failure nor a success, and is why the counters below read zero.
    probed: bool
    last_outcome: ServiceProbeOutcome | None = None
    last_detail: str | None = None
    consecutive_failures: int = Field(default=0, ge=0)
    consecutive_unknowns: int = Field(default=0, ge=0)
    last_success_at: datetime | None = None
    observed_at: datetime | None = None
    #: The load the replica reported about itself. ``None`` means it reported
    #: none, and is never rendered as ``0.0`` -- that would be a number nothing
    #: observed, handed to a reader who cannot tell it from a measurement.
    observed_load: float | None = Field(default=None, ge=0.0, le=1.0)


class ServiceHealthResponse(ApiModel):
    """One managed service's replica health, as an operator can read it.

    The three counts are separate fields that must account for every replica
    exactly once. A summary that reported only ``healthy`` would make an
    unreachable fleet indistinguishable from a dead one at exactly the moment
    the difference decides whether accelerators get destroyed.
    """

    service_id: str
    revision: str
    #: The declared replica count, which is what the service is *supposed* to
    #: have. It is not the length of ``replicas``, and a difference between them
    #: is the reconciler mid-flight rather than an error.
    declared_replicas: int = Field(ge=0)
    #: The path probes are sent to. Empty means this service declares none.
    health_path: str = ""
    #: ``False`` when the service declares no health path. Every replica then
    #: reads ``unknown`` and ``probed=False``, which is *not probed* rather than
    #: unhealthy: this service has never been asked anything.
    probing_enabled: bool = False
    healthy: int = Field(ge=0)
    unhealthy: int = Field(ge=0)
    #: Replicas whose health is not established. Its own count, never folded
    #: into ``unhealthy`` and never silently dropped.
    unknown: int = Field(ge=0)
    replicas: list[ServiceReplicaHealth] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> ServiceHealthResponse:
        if self.healthy + self.unhealthy + self.unknown != len(self.replicas):
            raise ValueError("every reported service replica must be counted in exactly one health bucket")
        return self

    @classmethod
    def build(
        cls,
        *,
        service: WorkloadRecord,
        policy: ServicePolicy,
        replicas: tp.Sequence[tp.Mapping[str, object]],
    ) -> ServiceHealthResponse:
        """Assemble the operator-facing answer from durable rows and policy.

        Every reported entry is validated against :class:`ServiceReplicaHealth`,
        which forbids unknown keys. That makes the coupling loud: a field added
        to the durable read and not to the wire fails here rather than becoming
        one more fact only SQL can see.
        """

        views = [
            ServiceReplicaHealth.model_validate({**replica, "probed": replica.get("last_outcome") is not None})
            for replica in replicas
        ]
        counted = collections.Counter(view.health for view in views)
        return cls(
            service_id=service.id,
            revision=policy.revision,
            declared_replicas=policy.replicas,
            health_path=policy.health_path,
            probing_enabled=bool(policy.health_path),
            healthy=counted[ServiceReplicaHealthState.HEALTHY],
            unhealthy=counted[ServiceReplicaHealthState.UNHEALTHY],
            unknown=counted[ServiceReplicaHealthState.UNKNOWN],
            replicas=views,
        )


class ChildWorkloadSubmitRequest(ApiModel):
    """A workload submitted by a running job.

    Carries only the spec. Scope, identity and parentage all come from the
    authenticated parent -- accepting them here would let a job name a scope its
    parent does not belong to, which is the whole thing the capability prevents.
    """

    spec: WorkloadDefinition


class RoleGrantIssueRequest(ApiModel):
    """Grant one role to one subject, durably.

    The grant's scope travels inside ``grant`` rather than as sibling fields, so
    ``RoleGrant``'s own validation -- fleet admin is unscoped, project roles
    require a project -- applies here instead of being restated and drifting.
    """

    subject: str = Field(min_length=1, max_length=256)
    grant: RoleGrant
    expires_at: datetime | None = None
    reason: str = Field(default="", max_length=512)

    def to_record(self) -> RoleGrantRecord:
        return RoleGrantRecord(granted_by="api", **self.model_dump())


class OrganizationRegisterRequest(ApiModel):
    """Register or update one tenant.

    ``id`` comes from the path, so it is absent here: two spellings of the same
    identity in one request is a mismatch waiting to be resolved silently in
    whichever direction the handler happens to prefer.
    """

    display_name: str = Field(default="", max_length=200)
    owner: str = Field(min_length=1, max_length=256)
    labels: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True

    def to_record(self, org_id: str, *, created_by: str) -> OrganizationRecord:
        return OrganizationRecord(id=org_id, created_by=created_by, **self.model_dump())


class ProjectRegisterRequest(ApiModel):
    """Register or update one project inside a registered organization."""

    display_name: str = Field(default="", max_length=200)
    owner: str = Field(min_length=1, max_length=256)
    labels: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True

    def to_record(self, org_id: str, project_id: str, *, created_by: str) -> ProjectRecord:
        return ProjectRecord(org_id=org_id, id=project_id, created_by=created_by, **self.model_dump())


class PlanePeerPutRequest(ApiModel):
    """Register another control plane this one may grant budget to.

    ``plane_id`` comes from the path for the same reason it does on an
    organization: one identity, spelled once.
    """

    endpoint: str = Field(min_length=1, max_length=512)
    description: str = Field(default="", max_length=512)
    enabled: bool = True


class BudgetGrantIssueRequest(ApiModel):
    """Cut a slice of one tenant's budget for one peer plane.

    ``expires_at`` is required and has no default. A grant that cannot lapse
    turns a home plane going away permanently into unbounded spend on every peer
    that ever held one, so the record refuses to be built without it -- and a
    default here would be this layer quietly choosing how long that risk lasts.
    """

    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    grantee_plane_id: str = Field(min_length=1, max_length=128)
    amount: float = Field(ge=0)
    expires_at: datetime


class QuotaGrantIssueRequest(ApiModel):
    """Cut a slice of one tenant's quota for one peer plane.

    ``limits`` is a resource vector rather than a scalar, and omitted dimensions
    are zero. A slice that silently authorized every resource it forgot to
    mention would be the opposite of a ceiling.
    """

    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    grantee_plane_id: str = Field(min_length=1, max_length=128)
    limits: ResourceLimits
    expires_at: datetime


class BudgetPutRequest(ApiModel):
    limit: float = Field(ge=0)
    accelerator_hour_limits: dict[str, float] = Field(default_factory=dict)
    currency: tp.Literal["USD"] = "USD"
    period: str = "monthly"
    #: Fractions of ``limit`` that emit an event the first time spend crosses them
    #: in a period. Settable here because a warning nobody can configure is a
    #: warning nobody gets. Bounded to ``(0, 1]`` at the request boundary as well
    #: as on the record: the record's check would reject this anyway, but as a 500
    #: from deep inside a service rather than a 422 naming the field.
    alert_thresholds: tuple[Annotated[float, Field(gt=0, le=1)], ...] = ()
    hard_limit: bool = True

    def to_record(self, org_id: str, project_id: str) -> BudgetRecord:
        return BudgetRecord(org_id=org_id, project_id=project_id, updated_by="api", **self.model_dump())


class QuotaPutRequest(ApiModel):
    org_id: str = Field(min_length=1, max_length=128)
    project_id: str | None = Field(default=None, max_length=128)
    principal_id: str | None = Field(default=None, max_length=128)
    limits: ResourceLimits
    #: Ceiling on indexed managed-artifact bytes in this scope. ``None`` is
    #: unlimited, which is the historical behaviour and stays the default.
    artifact_bytes_limit: int | None = Field(default=None, ge=0)
    weight: float = Field(default=1.0, gt=0)

    def to_record(self) -> QuotaRecord:
        return QuotaRecord(
            updated_by="api",
            limits=self.limits,
            **self.model_dump(exclude={"limits"}),
        )


class EffectiveProjectScope(ApiModel):
    """One project the authenticated principal can act in.

    ``permissions`` is evaluated by the same :class:`Authorizer` used at every
    service boundary after claim and durable grants have been unioned. It is not
    a client-side interpretation of an access-token payload.
    """

    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    display_name: str = Field(default="", max_length=200)
    registered: bool = False
    enabled: bool = True
    #: Whether this scope accepts mutations that create or change tenant work.
    #: Separate from ``enabled`` so disabled history remains selectable/readable,
    #: while a legacy unregistered organization stays compatible.
    accepts_new_work: bool = True
    roles: tuple[Role, ...] = ()
    permissions: tuple[Permission, ...] = ()


class SessionResponse(ApiModel):
    """Authenticated identity and its fully resolved effective authority."""

    subject: str = Field(min_length=1, max_length=256)
    display_name: str | None = None
    grants: tuple[RoleGrant, ...] = ()
    global_permissions: tuple[Permission, ...] = ()
    scopes: tuple[EffectiveProjectScope, ...] = ()


class HealthResponse(ApiModel):
    status: str = "ok"
    service: str = "eyvan"
    controller: ServerControllerMode


class BuildResponse(ApiModel):
    """What code this control plane is made of.

    Every component of this system reports one version string, so a caller
    cannot tell from it whether the plane it is talking to is the build a
    receipt was minted against. ``source_digest`` distinguishes them; it is
    ``None``, with ``digest_detail`` saying why, when it could not be computed
    in full -- never a digest over the part of the tree that happened to be
    readable.
    """

    version: str
    origin: str
    origin_detail: str = ""
    source_digest: str | None = None
    digest_detail: str = ""
    file_count: int = 0
    python: str
    platform: str
    controller: ServerControllerMode


class ReadOnlyQueryRequest(ApiModel):
    """One ad-hoc read against this plane's own database."""

    statement: str = Field(min_length=1, max_length=20_000)
    limit: int = Field(default=200, ge=1, le=10_000)


class ReadOnlyQueryResponse(ApiModel):
    """Rows, and whether there were more of them.

    ``truncated`` is reported rather than left to the caller to infer from the
    row count against the limit: those are indistinguishable when a result
    exactly fills the cap, and reading a cut-short answer as a complete one is
    the mistake this surface must not make easy.
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[tp.Any, ...], ...]
    truncated: bool


class ErrorResponse(ApiModel):
    detail: str


class NoMatchingCapacityPoolErrorContext(ApiModel):
    """Fleet inventory returned when callable placement has no exact pool."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    fleet: str
    requested_accelerator_type: str
    available_enabled_accelerator_types: tuple[str, ...]
    #: Fleets that do provide the requested shape, and every fleet in scope.
    #: Defaulted so a client reading an older plane's response still validates.
    fleets_providing_requested_type: tuple[str, ...] = ()
    available_fleets: tuple[str, ...] = ()


class NoMatchingCapacityPoolErrorResponse(ErrorResponse):
    """Structured API response for an unavailable accelerator in one fleet."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    code: tp.Literal[ServiceErrorCode.NO_MATCHING_CAPACITY_POOL] = ServiceErrorCode.NO_MATCHING_CAPACITY_POOL
    context: NoMatchingCapacityPoolErrorContext


class CapacityClassUnavailableErrorContext(ApiModel):
    """Classes a fleet does offer for the shape, when the requested one is absent."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    fleet: str
    requested_accelerator_type: str
    requested_capacity_class: str
    available_capacity_classes: tuple[str, ...]
    #: Offered classes whose every pool is hardware that already runs and is
    #: billed whatever Eyvan does (an attached TPU, operator-owned hosts).
    existing_hardware_capacity_classes: tuple[str, ...] = ()


class CapacityClassUnavailableErrorResponse(ErrorResponse):
    """Structured 409 for a fleet that has the shape but not the requested class."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    code: tp.Literal[ServiceErrorCode.CAPACITY_CLASS_UNAVAILABLE] = ServiceErrorCode.CAPACITY_CLASS_UNAVAILABLE
    context: CapacityClassUnavailableErrorContext


class StorageContextResponse(ApiModel):
    """Detected GCP/TPU runtime context returned by ``GET /api/v1/storage/status``."""

    on_gcp: bool = False
    project: str | None = None
    zone: str | None = None
    region: str | None = None
    accelerator_type: str | None = None
    source: str = "none"


class StorageMonitoringBucketResponse(ApiModel):
    """Configured default monitoring bucket for Eyvan run tracking."""

    bucket: str
    uri: str
    location: str
    prefix: str = "tracking"


class StorageRegionalBucketResponse(ApiModel):
    """Configured regional GCS bucket for checkpoints and datasets."""

    bucket: str
    uri: str
    location: str


class StorageStatusResponse(ApiModel):
    """Non-secret Eyvan storage configuration and detected GCP context."""

    configured: bool
    auto_create: bool
    config_path: str
    project: str | None = None
    monitoring: StorageMonitoringBucketResponse | None = None
    regional: dict[str, StorageRegionalBucketResponse] = Field(default_factory=dict)
    context: StorageContextResponse


class GovernanceSummaryUser(ApiModel):
    """Per-user quota, guarantee, borrowing, and live usage snapshot."""

    subject: str
    tier: str
    guaranteed_tpu_units: int | None = None
    max_tpu_units: int | None = None
    can_borrow: bool
    in_use_tpu_units: int = Field(default=0, ge=0)
    borrowed_tpu_units: int = Field(default=0, ge=0)


class GovernanceActiveReclaim(ApiModel):
    """One currently active graceful reclaim window across the fleet."""

    workload_id: str
    owner: str
    requested_by: str | None = None
    deadline_at: str | None = None
    policy: tp.Literal["requeue", "stop"] = "requeue"
    state: str


class GovernanceSummaryResponse(ApiModel):
    """Plane-wide team quota, borrowing, and active reclaim summary."""

    enabled: bool
    total_guaranteed_tpu_units: int = Field(default=0, ge=0)
    total_in_use_tpu_units: int = Field(default=0, ge=0)
    total_borrowed_tpu_units: int = Field(default=0, ge=0)
    users: list[GovernanceSummaryUser] = Field(default_factory=list)
    active_reclaims: list[GovernanceActiveReclaim] = Field(default_factory=list)
