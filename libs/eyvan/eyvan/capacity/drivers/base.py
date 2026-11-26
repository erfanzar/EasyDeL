# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backend-neutral contracts for capacity and execution cells."""

from __future__ import annotations

import logging
import typing as tp
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime

from typing_extensions import TypedDict

from eyvan._internal.compat import StrEnum
from eyvan._internal.json_types import JsonObject, JsonValue
from eyvan.capacity import CapacityStatus
from eyvan.domain.availability import ProvisionOutcome

logger = logging.getLogger(__name__)


def canonical_execution_backend(name: str) -> str:
    """Return the stable public name for a built-in execution backend."""

    normalized = name.strip().lower().replace("_", "-")
    aliases = {
        "local": "native-local",
        "local-process": "native-local",
        "native": "eyvan-native",
        "native-remote": "eyvan-native",
        "native-worker": "eyvan-native",
        "eyvan": "eyvan-native",
        # Retired: kept as aliases so a persisted workload naming Ray still
        # normalises to one refused backend instead of an unknown-typo error.
        "ray": "ray-jobs",
        "rayjobs": "ray-jobs",
    }
    return aliases.get(normalized, normalized)


def capacity_backend_agent_family(name: str) -> str:
    """Return the group of capacity backends one cell agent can serve at once.

    ``CellAgentService`` reads ``settings.capacity_backend`` once at construction and
    builds a single driver, so an agent serves exactly one *driver* -- but several backend
    names can share one. ``gcp``, ``gcp-tpu`` and ``gcp-compute`` are all modes of the same
    factory (``drivers/registry.py``), which is why a mixed TPU + Compute fleet behind one
    agent endpoint is legitimate and must keep validating.

    Expressed as data here rather than read from the registry on purpose: the registry's
    stated invariant is that selecting one backend never imports another's dependencies, so
    a config-time validator must not import it just to ask this question.
    """

    normalized = name.strip().lower().replace("_", "-")
    families = {
        "gcp": "gcp",
        "gcp-tpu": "gcp",
        "gcp-attach": "gcp",
        "tpu": "gcp",
        "gcp-compute": "gcp",
        "compute": "gcp",
        "kubernetes": "kubernetes",
        "k8s": "kubernetes",
        "aws": "aws",
        "aws-ec2": "aws",
        "ec2": "aws",
        "azure": "azure",
        "azure-vm": "azure",
    }
    return families.get(normalized, normalized)


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """What one capacity backend can actually provide, as data rather than a name.

    The durable model has to refuse shapes no backend can build -- a cell
    spanning several machines, allocated together or not at all, is the only one
    so far -- and it did that by asking whether the provider was GCP. That reads
    as a validation and behaves as a moat: a second cloud could implement
    multislice perfectly and still be refused by a record validator in a file
    that has nothing to do with any provider.

    Stated here, keyed by backend, a new provider declares what it can do and
    the durable model stops being the thing that decides.

    Attributes:
        multi_worker_cells: Whether one cell may span more than one machine.
        max_workers_per_cell: The widest such cell this backend will build.
        existing_hardware: Whether a cell is a claim on machines that already
            run and are billed whether or not Eyvan uses them (attached or
            operator-owned), rather than capacity Eyvan provisions. Running on
            such a cell starts no new spend, whatever its capacity class.
    """

    multi_worker_cells: bool = False
    max_workers_per_cell: int = 1
    existing_hardware: bool = False


#: Every backend's capabilities. A backend absent from this table hosts only
#: single-machine cells, which is the safe reading: a driver that has not said
#: it can allocate a gang atomically almost certainly cannot.
_BACKEND_CAPABILITIES: dict[str, BackendCapabilities] = {
    # One Cloud TPU multislice queued resource. 256 is the provider's own
    # ceiling on workers in a single request, not a number Eyvan chose.
    "gcp-tpu": BackendCapabilities(multi_worker_cells=True, max_workers_per_cell=256),
    # One Compute Engine instance, EC2 instance, or Azure VM per cell today.
    # Declared at the single-machine default on purpose: an entry here is a
    # statement, and "one machine per cell" is the honest one for these.
    "gcp-compute": BackendCapabilities(),
    "aws-ec2": BackendCapabilities(),
    "azure-vm": BackendCapabilities(),
    "local": BackendCapabilities(),
    # One Kubernetes indexed job per cell; 512 is the job controller's own
    # worker ceiling (MAX_INDEXED_JOB_WORKERS), not a number Eyvan chose.
    "kubernetes": BackendCapabilities(multi_worker_cells=True, max_workers_per_cell=512),
    # Operator-owned host sets joined over SSH or declared by hand share the
    # drivers' own 512-host ceilings (onprem_ssh.MAX_HOSTS and the manual
    # driver's limit), which is why the two numbers agree.
    "onprem-ssh": BackendCapabilities(multi_worker_cells=True, max_workers_per_cell=512, existing_hardware=True),
    "manual": BackendCapabilities(multi_worker_cells=True, max_workers_per_cell=512, existing_hardware=True),
    # An already-running TPU VM adopted by ``eyvan gcp connect``: one slice per
    # cell, never created or deleted by Eyvan.
    "gcp-attach": BackendCapabilities(existing_hardware=True),
}


def backend_capabilities(name: str | None) -> BackendCapabilities:
    """Return what ``name`` can build, defaulting to a single-machine backend."""

    if not name:
        return BackendCapabilities()
    return _BACKEND_CAPABILITIES.get(name.strip().lower().replace("_", "-"), BackendCapabilities())


def providers_hosting_multi_worker_cells() -> frozenset[str]:
    """Return every provider some backend of which can build a multi-machine cell.

    Derived from the capability table rather than listed, so a second cloud that
    declares a multislice backend is accepted by the durable model without
    anyone editing it. A workload spec names a provider and not a backend --
    which backend hosts it is a placement decision -- so this is the widest
    honest question the spec-level check can ask.
    """

    return frozenset(
        capacity_backend_agent_family(name)
        for name, capabilities in _BACKEND_CAPABILITIES.items()
        if capabilities.multi_worker_cells
    )


def max_multi_worker_cell_width() -> int:
    """Return the widest multi-machine cell any known backend will build."""

    return max(
        (capabilities.max_workers_per_cell for capabilities in _BACKEND_CAPABILITIES.values()),
        default=1,
    )


class StaleMutationError(RuntimeError):
    """Reject a mutation that does not own the persisted generation fence."""


class DriverBusyError(RuntimeError):
    """A shared driver mutation lease is currently owned elsewhere."""


class CapacityProviderFailureCode(StrEnum):
    """Stable machine-readable reason for a capacity-provider failure."""

    PROVIDER_ERROR = "provider_error"
    QUEUE_ADMISSION_DENIED = "queue_admission_denied"
    #: The provider's control API could not be reached or was overloaded. Distinct
    #: from PROVIDER_ERROR because it is the ordinary state of a cluster during an
    #: upgrade, and discarding the command would drop capacity the control plane
    #: believes it asked for.
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    #: The request will fail identically however many times it is retried --
    #: missing credentials, a namespace that does not exist, a rejected manifest.
    CONFIGURATION = "configuration"
    #: The provider has no capacity of this shape right now. Distinct from
    #: PROVIDER_ERROR because it says something about the *pool* rather than
    #: about the request, and the placement engine routes on that difference.
    STOCKOUT = "stockout"
    #: The project's quota for this shape is exhausted. Retrying in the same
    #: pool cannot succeed until a human raises the limit, so it must not look
    #: like a stockout that another zone might satisfy.
    QUOTA_EXCEEDED = "quota_exceeded"
    #: The provider reclaimed capacity Eyvan already held. Expected on spot, and
    #: the cheapest failure in this enum to retry -- which is precisely why it
    #: must not be charged to the same budget as a stockout or a bad request.
    PREEMPTED = "preempted"


#: How one provider reason updates a pool's availability belief.
#:
#: Explicit and total rather than derived from the failure text. The two places
#: that needed this both reconstructed it by searching an arbitrary provider
#: string for the words "quota" and "capacity" -- which meant a message like
#: "capacity pool not found" (a configuration mistake) was recorded as evidence
#: that a zone had run out of accelerators, and every reason nobody had written
#: a substring for became a stockout by default.
#:
#: ``ProvisionOutcome.HOST_FAULT`` is deliberately unreachable from this map and
#: has no ``CapacityProviderFailureCode``. A host fault is observed from an
#: *execution* -- an attempt's own output naming an accelerator that never came
#: up -- and never from a provider API, which reports a machine it created
#: successfully. Adding a provider code for it would put a value on the agent
#: wire and in ``CellRecord.provider_failure_code`` that no provider raises, that
#: a rolled-back replica could not decode, and that would make
#: :func:`provision_outcome` claim a provider said something it did not.
_PROVISION_OUTCOMES: dict[CapacityProviderFailureCode, ProvisionOutcome] = {
    CapacityProviderFailureCode.STOCKOUT: ProvisionOutcome.STOCKOUT,
    CapacityProviderFailureCode.QUOTA_EXCEEDED: ProvisionOutcome.QUOTA_ERROR,
    # Not a stockout. The provider had the capacity and took it back, which says
    # nothing about whether the pool can satisfy the next request -- and saying
    # otherwise steers placement away from a zone that is working.
    CapacityProviderFailureCode.PREEMPTED: ProvisionOutcome.INTERRUPTED,
    CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED: ProvisionOutcome.PROVIDER_ERROR,
    CapacityProviderFailureCode.PROVIDER_UNAVAILABLE: ProvisionOutcome.PROVIDER_ERROR,
    CapacityProviderFailureCode.CONFIGURATION: ProvisionOutcome.PROVIDER_ERROR,
    CapacityProviderFailureCode.PROVIDER_ERROR: ProvisionOutcome.PROVIDER_ERROR,
}

#: Provider diagnostics that name an exhausted project quota, casefolded.
#: Deliberately short and confident: a wrong quota classification poisons the
#: placement belief in the *opposite* direction (permanently routing away from
#: a zone whose real problem was transient), so anything ambiguous must stay
#: unclassified and fall through to PROVIDER_ERROR.
_DIAGNOSTIC_QUOTA_SIGNATURES = (
    "quota exceeded",
    "quota has been exceeded",
    "quota_exceeded",
    "exceeds allocated quota",
    "quota limit has been reached",
)

#: Provider diagnostics that name a capacity shortage in the requested zone --
#: the provider's stockout vocabulary across Cloud TPU and Compute Engine.
_DIAGNOSTIC_STOCKOUT_SIGNATURES = (
    "stockout",
    "does not have enough resources available",
    "zone_resource_pool_exhausted",
    "resource pool exhausted",
    "insufficient capacity available",
    "no more resources available in zone",
    "capacity is not available in zone",
)


def provider_failure_from_diagnostic(diagnostic: str) -> CapacityProviderFailureCode | None:
    """Return the failure code named by a provider's own diagnostic text.

    ``STOCKOUT`` and ``QUOTA_EXCEEDED`` exist because the placement engine
    routes on them -- a stockout says "another zone might work", a quota halt
    says "no zone will work until a human raises a limit" -- but for their
    entire existence every GCP provisioning failure surfaced as the default
    ``PROVIDER_ERROR``, so the distinction was dead on every path. This reads
    the provider's own words; when it names nothing confidently, return
    ``None`` and let the caller keep its default rather than guessing.
    """

    normalized = " ".join(diagnostic.casefold().split())
    if not normalized:
        return None
    if any(signature in normalized for signature in _DIAGNOSTIC_QUOTA_SIGNATURES):
        return CapacityProviderFailureCode.QUOTA_EXCEEDED
    if any(signature in normalized for signature in _DIAGNOSTIC_STOCKOUT_SIGNATURES):
        return CapacityProviderFailureCode.STOCKOUT
    return None


def provision_outcome(
    failure_code: CapacityProviderFailureCode | None,
    status: CapacityStatus | None = None,
) -> ProvisionOutcome:
    """Return how one provider failure should move a pool's availability belief.

    ``status`` is the fallback for an observation that reported no reason --
    an older agent, or a provider that simply did not say. ``HALTED_QUOTA`` is
    self-describing; anything else unnamed is a provider error, **not** a
    stockout. Defaulting the unknown to "this pool is empty" is how a belief
    that drives placement gets poisoned by causes that have nothing to do with
    capacity.
    """

    if failure_code is not None:
        return _PROVISION_OUTCOMES[failure_code]
    if status is CapacityStatus.HALTED_QUOTA:
        return ProvisionOutcome.QUOTA_ERROR
    return ProvisionOutcome.PROVIDER_ERROR


class CapacityProviderError(RuntimeError):
    """An operator-safe diagnostic returned by an external capacity provider."""

    def __init__(
        self,
        diagnostic: str,
        *,
        code: CapacityProviderFailureCode = CapacityProviderFailureCode.PROVIDER_ERROR,
        retryable: bool | None = None,
    ) -> None:
        normalized = diagnostic.strip()
        self.diagnostic = normalized or "capacity provider operation failed"
        self.code = code
        # Derived from the code unless stated, so existing raisers keep their
        # behaviour and a new one cannot forget to classify itself.
        self.retryable = code is CapacityProviderFailureCode.PROVIDER_UNAVAILABLE if retryable is None else retryable
        super().__init__(self.diagnostic)


class UnsupportedCapabilityError(RuntimeError):
    """A selected backend cannot satisfy a requested capability."""


class ExecutionWorkloadStatus(StrEnum):
    """Backend-neutral state reported for one remote execution.

    ``INDETERMINATE`` means a backend explicitly attempted an observation but
    could not establish a definitive state. It is never a missing-value or
    default fallback.
    """

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    CHECKPOINTING = "CHECKPOINTING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    LOST = "LOST"
    NOT_FOUND = "NOT_FOUND"
    INDETERMINATE = "INDETERMINATE"


class ExecutionWorkloadResult(TypedDict):
    """Stable JSON-compatible workload status payload returned by an executor."""

    remote_id: str
    status: ExecutionWorkloadStatus
    message: str


@dataclass(frozen=True, slots=True)
class ExecutionWorkloadObservation(tp.Mapping[str, JsonValue]):
    """Typed executor observation with the existing mapping wire shape."""

    remote_id: str
    status: ExecutionWorkloadStatus
    message: str = ""

    def as_result(self) -> ExecutionWorkloadResult:
        """Return the stable payload consumed by the agent wire model."""

        return {"remote_id": self.remote_id, "status": self.status, "message": self.message}

    def __getitem__(self, key: str) -> JsonValue:
        """Read a status field through the legacy mapping interface."""

        if key == "remote_id":
            return self.remote_id
        if key == "status":
            return self.status
        if key == "message":
            return self.message
        raise KeyError(key)

    def __iter__(self) -> tp.Iterator[str]:
        """Iterate the stable workload-status field names."""

        return iter(("remote_id", "status", "message"))

    def __len__(self) -> int:
        """Return the fixed number of workload-status fields."""

        return 3


ExecutionWorkloadReport: tp.TypeAlias = ExecutionWorkloadObservation | ExecutionWorkloadResult
"""Typed status result returned by every execution driver.

Most drivers return :class:`ExecutionWorkloadObservation`. Coordinating
drivers may return a typed dictionary that extends ``ExecutionWorkloadResult``
with per-worker details.
"""


@dataclass(frozen=True, slots=True)
class MutationContext:
    """Fencing metadata required for every mutating driver operation."""

    idempotency_key: str
    generation: int
    fencing_token: str
    #: Observability only; never part of ownership or idempotency decisions.
    correlation_id: str | None = field(default=None, compare=False)


#: The agent wire model's bound on an idempotency key.
MAX_IDEMPOTENCY_KEY_LENGTH = 256


def rollback_context(context: MutationContext) -> MutationContext:
    """Return the context a partial-launch rollback retires under.

    Same fence -- the rollback owns exactly what the submission owned -- but a
    key of its own. Retiring under the submission key made a second rollback,
    after the dispatch retry had relaunched under that key, indistinguishable
    from a stale replay of the first: the executor ignored it and left a rank
    running. The nonce comes first so the key stays unique even when a long
    submission key is truncated to the wire bound.
    """

    key = f"rollback:{uuid.uuid4().hex}:{context.idempotency_key}"
    return replace(context, idempotency_key=key[:MAX_IDEMPOTENCY_KEY_LENGTH])


@dataclass(frozen=True, slots=True)
class CellCapabilities:
    """Features and resource shapes supported by a cell driver."""

    provider: str
    executor: str
    accelerator_types: tuple[str, ...]
    capacity_classes: tuple[str, ...]
    #: Capacity backend names this agent's single driver can serve. **Plural**,
    #: because the GCP driver legitimately serves ``gcp``, ``gcp-tpu`` and
    #: ``gcp-compute`` from one process, and Kubernetes serves ``kubernetes`` and
    #: ``k8s``. Empty means "does not declare", so an older agent that predates
    #: this field is skipped rather than treated as serving nothing.
    capacity_backends: tuple[str, ...] = ()
    supports_checkpoint: bool = True
    supports_callables: bool = False
    #: Whether an operator has *permitted* diagnostic exec on this cell.
    #:
    #: Defaulted to **False**, changed from True. A default-on capability that no
    #: production code read is precisely how it became dead: nothing could have
    #: noticed it was wrong. Every declaration site now says so explicitly, and
    #: this is the operator-facing admission gate -- whether a backend *can*
    #: exec is a separate question, answered by ``ExecCapableExecutionDriver``.
    supports_exec: bool = False
    max_gang_replicas: int | None = None
    resource_enforcement: str = "backend"


@dataclass(frozen=True, slots=True)
class CellObservation:
    """Observed state returned during reconciliation."""

    cell_id: str
    state: CapacityStatus
    generation: int
    execution_endpoint: str | None = None
    detail: str = ""
    observed_at: datetime | None = None
    metadata: JsonObject = field(default_factory=dict)
    observed_tpu_vms: int | None = None
    #: Stable number of runtime/JAX processes exposed by this cell. This is not
    #: ``worker_count``: the latter counts allocated slices, while one TPU slice
    #: can expose several native-worker endpoints.
    runtime_process_count: int | None = None
    #: Why the provider is in a non-ready state, when it said something specific.
    #: A code that arrives only on a *raised* provider error is a code that never
    #: describes a successful observation -- and a spot preemption is a
    #: successful observation of a reclaimed machine, not an error.
    failure_code: CapacityProviderFailureCode | None = None


#: Metadata key under which a driver publishes what its cell's hosts carry.
#:
#: A wire *addition*, so it is staged: agents start reporting it before anything
#: reads it, and every reader treats its absence as **unknown**. Absence is not
#: "this cell has no accelerator" -- an agent that predates this key would then
#: be read as evidence about hardware it was never asked about.
CELL_ACCELERATOR_METADATA_KEY: tp.Final = "cell_accelerator"


@dataclass(frozen=True, slots=True)
class CellAccelerator:
    """What the shipped hardware catalog says one cell's hosts carry.

    Derived at observation time from the catalog, never persisted and never
    inferred from the shape of a name: :func:`describe_cell_accelerator` looks a
    type up, and a type with no row produces nothing at all. Parsing ``4`` out
    of ``g2-standard-48`` would be a guess, and a wrong device count is dispatched
    to a rank as an exact device list nobody can correct downstream.

    ``devices_per_host`` is optional because the catalog deliberately leaves it
    out for TPU generations this repository cannot ground. Optional is not zero:
    zero would reach a rank as an empty ``CUDA_VISIBLE_DEVICES``, which is how a
    process ends up on the CPU while its peers wait for it at a collective.

    Nothing here is hardware-qualified. A callable can report success even when
    the host's driver cannot open its GPU. This describes catalog hardware, not
    evidence that a workload can use it.

    ``eyvan.worker.runtime.host_accelerators`` is the counterpart that looks at
    the machine instead of the catalog.

    Attributes:
        kind: The catalog's accelerator kind, ``"gpu"`` or ``"tpu"`` today. Read
            as an open string so a newer agent's vocabulary is unknown rather
            than invalid.
        accelerator_type: The catalog row that produced this description.
        catalog_revision: The catalog revision the numbers came from, so a
            dispatched device count can be traced back to a reviewed row.
        devices_per_host: Accelerators on one host, or ``None`` when the catalog
            does not record it.
        gpu_model: The attached accelerator's provider name, GPU rows only.
    """

    kind: str
    accelerator_type: str
    catalog_revision: str
    devices_per_host: int | None = None
    gpu_model: str | None = None

    def __post_init__(self) -> None:
        """Refuse a description that cannot be acted on.

        Raises:
            ValueError: If a field is empty, untrimmed, not a count, or claims a
                GPU model without the device count that makes it usable.
        """

        for field_name in ("kind", "accelerator_type", "catalog_revision"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"cell accelerator {field_name} must be non-empty and trimmed")
        if self.devices_per_host is not None and (
            isinstance(self.devices_per_host, bool)
            or not isinstance(self.devices_per_host, int)
            or self.devices_per_host <= 0
        ):
            raise ValueError("cell accelerator device count must be a positive integer or absent")
        if self.gpu_model is not None and (
            not isinstance(self.gpu_model, str) or not self.gpu_model.strip() or self.gpu_model != self.gpu_model.strip()
        ):
            raise ValueError("cell accelerator GPU model must be non-empty and trimmed or absent")
        if self.gpu_model is not None and self.kind != "gpu":
            raise ValueError(f"cell accelerator kind {self.kind!r} cannot name a GPU model")
        if self.kind == "gpu" and self.devices_per_host is None:
            raise ValueError("a GPU cell accelerator must state how many devices one host carries")

    @property
    def gpu_devices_per_host(self) -> int | None:
        """Return GPUs on one host, or ``None`` when this is not a GPU shape.

        Named after the field it feeds, ``NativeGangMember.gpu_devices_per_host``,
        so the one value that pins a rank to devices is not re-derived by each
        caller. A TPU cell answers ``None`` here, which is the same answer an
        unknown cell gives, and both mean "not GPU-coordinated".
        """

        return self.devices_per_host if self.kind == "gpu" else None

    def as_metadata(self) -> JsonObject:
        """Return the JSON block a driver publishes in ``CellObservation.metadata``.

        Returns:
            The block keyed by :data:`CELL_ACCELERATOR_METADATA_KEY`. Optional
            fields are omitted rather than written as ``null``, so a reader can
            never mistake "not recorded" for a recorded empty value.
        """

        block: JsonObject = {
            "kind": self.kind,
            "accelerator_type": self.accelerator_type,
            "catalog_revision": self.catalog_revision,
        }
        if self.devices_per_host is not None:
            block["devices_per_host"] = self.devices_per_host
        if self.gpu_model is not None:
            block["gpu_model"] = self.gpu_model
        return block


def describe_cell_accelerator(
    *,
    accelerator_type: str | None,
    machine_type: str | None = None,
    attached_device_count: int | None = None,
) -> CellAccelerator | None:
    """Describe one cell's accelerator from the shipped hardware catalog.

    Two lookup keys because the catalog uses two: TPU rows are keyed by the
    accelerator type a pool already declares (``v5p-16``), while GPU rows are
    keyed by *machine type*, because the GPU model fixes none of the numbers --
    ``g2-standard-4`` and ``g2-standard-96`` carry the same L4 and differ by 8x
    in attached devices. A cell that declares ``nvidia-l4`` is therefore found
    through its machine type or not at all.

    Args:
        accelerator_type: The cell's declared accelerator type, tried first.
        machine_type: The machine type the cell was provisioned as, tried when
            the declared type has no catalog row.
        attached_device_count: Devices the provider was asked for and verified
            on the instance, when one is configured. Used only to *contradict*
            the catalog, never to supply a count the catalog does not have.

    Returns:
        The description, or ``None`` when this build cannot state one. ``None``
        means unknown: the catalog has no row, the row records no host device
        count, or the two available sources disagree. It never means "no
        accelerator", and callers must not turn it into a zero.
    """

    # Imported here rather than at module scope for the reason the catalog
    # itself documents: it is a leaf that reaches into the GCP provider package
    # for TPU name parsing, and this module is imported by the host-side worker
    # runtime, which has no reason to pay for reading a YAML catalog.
    from eyvan.capacity.hardware_catalog import hardware_catalog

    catalog = hardware_catalog()
    shape = None
    for candidate in (accelerator_type, machine_type):
        if isinstance(candidate, str) and candidate.strip():
            shape = catalog.get(candidate)
            if shape is not None:
                break
    if shape is None or shape.devices_per_host is None:
        return None
    if attached_device_count is not None and attached_device_count != shape.devices_per_host:
        # Two sources that should agree do not, and there is no way to tell here
        # which one is wrong. Reporting either would hand a rank an exact device
        # list built on the loser of a coin flip.
        logger.warning(
            "cell accelerator %r is described by catalog row %r as %d devices per host "
            "but %d are attached; reporting the accelerator as unknown",
            accelerator_type,
            shape.accelerator_type,
            shape.devices_per_host,
            attached_device_count,
        )
        return None
    return CellAccelerator(
        kind=shape.kind,
        accelerator_type=shape.accelerator_type,
        catalog_revision=catalog.revision,
        devices_per_host=shape.devices_per_host,
        gpu_model=shape.gpu_model,
    )


def read_cell_accelerator(metadata: tp.Mapping[str, JsonValue]) -> CellAccelerator | None:
    """Read one cell's accelerator description back out of observation metadata.

    Total by construction: every failure returns ``None``, which the whole GPU
    path reads as *unknown* and answers with the behaviour it had before this key
    existed. That is what makes the rollout safe in both directions -- an agent
    older than the key reports nothing, and an agent newer than this reader
    reports a vocabulary it does not have to understand.

    Args:
        metadata: One :class:`CellObservation`'s metadata mapping.

    Returns:
        The description, or ``None`` when it is absent or unreadable. Never a
        zero-device description.
    """

    block = metadata.get(CELL_ACCELERATOR_METADATA_KEY)
    if block is None:
        return None
    if not isinstance(block, dict):
        logger.warning("cell accelerator metadata is not a mapping; reading the accelerator as unknown")
        return None
    try:
        return CellAccelerator(
            kind=block.get("kind"),  # type: ignore[arg-type]
            accelerator_type=block.get("accelerator_type"),  # type: ignore[arg-type]
            catalog_revision=block.get("catalog_revision"),  # type: ignore[arg-type]
            devices_per_host=block.get("devices_per_host"),  # type: ignore[arg-type]
            gpu_model=block.get("gpu_model"),  # type: ignore[arg-type]
        )
    except ValueError as exc:
        logger.warning("cell accelerator metadata is unusable (%s); reading the accelerator as unknown", exc)
        return None


class CapacityDriver(tp.Protocol):
    """Provider lifecycle used by the global capacity reconciler."""

    def capabilities(self) -> CellCapabilities: ...

    def observe(self, cell_id: str) -> CellObservation: ...

    def reconcile(self, cell_id: str, context: MutationContext) -> CellObservation: ...

    def apply_capacity(
        self,
        cell_id: str,
        desired: tp.Mapping[str, JsonValue],
        context: MutationContext,
    ) -> CellObservation: ...

    def drain(self, cell_id: str, context: MutationContext) -> CellObservation: ...

    def release(self, cell_id: str, context: MutationContext) -> CellObservation: ...


class ExecutionDriver(tp.Protocol):
    """Whole-workload operations delegated to an execution cell."""

    def submit(self, spec: tp.Mapping[str, JsonValue], context: MutationContext) -> str: ...

    def status(self, remote_id: str) -> ExecutionWorkloadReport: ...

    def stop(self, remote_id: str, context: MutationContext) -> None: ...

    def retire(self, remote_id: str, context: MutationContext) -> None: ...

    def checkpoint(
        self,
        remote_id: str,
        *,
        hook: str,
        output_uri: str | None,
        context: MutationContext,
    ) -> str: ...

    def logs(self, remote_id: str) -> str: ...


class ExecutionDriverResolver(tp.Protocol):
    """Resolve the configured execution backend for a particular cell."""

    def resolve(self, cell_id: str, observation: CellObservation) -> ExecutionDriver: ...


class CapacityProviderPlugin(tp.Protocol):
    """The object an out-of-tree capacity provider publishes at its entry point.

    This is the whole ABI surface an external distribution has to satisfy. It is
    deliberately three members and no base class: a provider must be able to
    declare which ABI it was built against **without importing Eyvan**, because
    the version check exists precisely for the case where importing this Eyvan
    would fail or would mean something different than the provider expects. A
    module object satisfies this protocol, so the smallest legal plugin is a
    module with two module-level constants and one function.

    Attributes:
        abi_version: The published ABI this provider was written against, as
            ``"MAJOR.MINOR"``. A plain string rather than a shared type for the
            reason above. Mismatches are refused by
            :mod:`eyvan.capacity.drivers.registry`, never adapted.
        backend_names: The ``capacity_backend`` names this plugin serves, in the
            same normalized form the registry uses. Plural because one factory
            legitimately serves several modes, exactly as ``gcp``, ``gcp-tpu``
            and ``gcp-compute`` share one in-tree factory.
    """

    abi_version: str
    backend_names: tp.Sequence[str]

    def create(self, name: str, *, state: tp.Any, settings: tp.Any) -> CapacityDriver:
        """Build the driver for one of ``backend_names``.

        Args:
            name: The normalized backend name being created, so one plugin can
                serve several modes.
            state: The durable, fenced :class:`~eyvan.capacity.fleet_state.FleetStateStore`
                this agent owns. Typed as ``Any`` here only to keep this module
                import-free of the store it describes; the passed object is
                always a ``FleetStateStore``.
            settings: The agent's :class:`~eyvan.config.EyvanSettings`.

        Returns:
            A driver satisfying :class:`CapacityDriver`.
        """
        ...
