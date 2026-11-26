# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Durable callable-job value types shared by execution and SDK adapters."""

from __future__ import annotations

import hashlib
import re
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

from eyvan.artifacts import StoredCallableArtifact
from eyvan.domain.accelerators import sort_accelerator_types
from eyvan.records import (
    CALLABLE_SIGNING_KEY_ENV,
    DEFAULT_MAX_RETRIES,
    REMOTE_CALLABLE_SIGNATURE_ERROR,
    REMOTE_CALLABLE_SIGNING_KEY_ERROR,
    CapacityClass,
    TpuChipRequest,
    WorkloadRecord,
    WorkloadSetup,
    WorkloadState,
)
from eyvan.sdk.client.easydel_setup import EasyDeLSetup

R_co = tp.TypeVar("R_co", covariant=True)

_ACCELERATOR_TYPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def reject_remote_callable_signing_key(signing_key: bytes | None) -> None:
    """Reject caller-owned signing keys that cannot safely reach remote workers."""

    if signing_key:
        raise ValueError(REMOTE_CALLABLE_SIGNING_KEY_ERROR)


def reject_remote_callable_signature(signature: str | None) -> None:
    """Reject artifact signatures that no remote worker can verify safely."""

    if signature is not None:
        raise ValueError(REMOTE_CALLABLE_SIGNATURE_ERROR)


def reject_remote_callable_signing_environment(environment: tp.Mapping[str, str] | None) -> None:
    """Reject the worker-only signing secret from remote callable environments."""

    if environment is not None and CALLABLE_SIGNING_KEY_ENV in environment:
        raise ValueError(REMOTE_CALLABLE_SIGNING_KEY_ERROR)


@tp.runtime_checkable
class AcceleratorConfig(tp.Protocol):
    """What the submission path needs from any hardware request.

    ``TpuConfig``, ``GpuConfig`` and ``CpuConfig`` differ only in which shapes they
    accept, so every consumer -- fleet resolution, pool selection, the callable
    submitter -- works against this instead of one concrete class.
    """

    type: str
    pods: int
    max_retries: int
    max_failure_retries: int
    max_preemption_retries: int
    capacity_class: CapacityClass
    secret_mounts: tp.Mapping[str, str]
    checkpointable: bool
    checkpoint_hook: str | None
    checkpoint_output_uri: str | None
    can_submit_children: bool
    dependencies: tp.Sequence[str]


def _validate_accelerator_request(config: AcceleratorConfig, *, label: str, multi_worker: bool) -> str:
    """Validate the shared fields and return the normalized accelerator type.

    ``multi_worker`` is false for GPU and CPU because ``WorkloadDefinition`` requires
    that ``worker_count > 1`` imply a TPU resource request -- Eyvan's multi-worker
    unit is one atomic Multislice queued resource, which has no GPU or CPU analogue.
    Rejecting ``pods > 1`` here turns that into a clear SDK-level error instead of a
    validation failure deep in the submit path.
    """

    accelerator_type = config.type.strip() if isinstance(config.type, str) else ""
    if not accelerator_type or not _ACCELERATOR_TYPE.fullmatch(accelerator_type):
        raise ValueError(f"{label} type must contain only letters, digits, dots, underscores, or hyphens")
    if isinstance(config.pods, bool) or not isinstance(config.pods, int) or config.pods < 1:
        raise ValueError(f"{label} pods must be a positive integer")
    if not multi_worker and config.pods != 1:
        raise ValueError(f"{label} does not support multi-worker capacity; only TPU pods may exceed 1")
    if not isinstance(config.capacity_class, CapacityClass):
        raise ValueError(f"{label} capacity_class must be a CapacityClass")
    if (
        isinstance(config.max_retries, bool)
        or not isinstance(config.max_retries, int)
        or config.max_retries < 0
        or config.max_retries > 100
    ):
        raise ValueError(f"{label} max_retries must be an integer from 0 through 100")
    # A separate budget, not a synonym. ``max_retries`` pays for capacity the
    # fleet lost -- a preemption, a stockout, a reclaimed spot cell -- which says
    # nothing about the job. ``max_failure_retries`` pays for the job's own
    # non-zero exit, which usually will repeat, which is why it defaults to zero
    # and why merging the two would let a deterministically broken job burn a
    # capacity budget sized for interruptions.
    if (
        isinstance(config.max_failure_retries, bool)
        or not isinstance(config.max_failure_retries, int)
        or config.max_failure_retries < 0
        or config.max_failure_retries > 100
    ):
        raise ValueError(f"{label} max_failure_retries must be an integer from 0 through 100")
    return accelerator_type


@dataclass(frozen=True, slots=True)
class SetupScript:
    """A per-job setup step: a bash script and the arguments it is run with.

    ``source`` is an ``https://`` URL, fetched on each host, or a local file
    path, read here at submission and shipped inline. ``sha256`` pins the
    script's bytes: a fetched or local script that differs is refused.
    """

    source: str
    args: tuple[str, ...] = ()
    sha256: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.args, (str, bytes)) or not isinstance(self.args, (list, tuple)):
            raise ValueError("SetupScript args must be a list of strings")
        if any(not isinstance(argument, str) for argument in self.args):
            raise ValueError("SetupScript args must be strings")
        object.__setattr__(self, "args", tuple(self.args))

    def to_workload_setup(self) -> WorkloadSetup:
        """Return the wire form, reading a local file now so the host needs no copy of it."""

        source = self.source.strip() if isinstance(self.source, str) else ""
        if not source:
            raise ValueError("SetupScript needs an https:// URL or a local file path")
        if source.startswith("https://"):
            return WorkloadSetup(url=source, args=self.args, sha256=self.sha256)
        if "://" in source:
            raise ValueError("a SetupScript URL must be https://; the script runs as the job")
        path = Path(source).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(f"cannot read setup script {source!r}: {error}") from error
        if self.sha256 is not None and hashlib.sha256(text.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError(f"setup script {source!r} does not match its sha256")
        return WorkloadSetup(script=text, args=self.args)


#: Every value a config's ``setup`` field accepts.
SetupValue: tp.TypeAlias = "SetupScript | EasyDeLSetup | WorkloadSetup | str | None"


def workload_setup(setup: SetupValue) -> WorkloadSetup | EasyDeLSetup | None:
    """Normalize a config's ``setup`` value to its wire form.

    A plain string is inline bash, except a single ``https://`` token, which is
    a script URL. Use ``SetupScript`` to pass arguments or pin a hash. An
    ``EasyDeLSetup`` is kept as is: its branch is resolved to a commit when the
    job is submitted, not when the decorator is defined.
    """

    if setup is None or isinstance(setup, (WorkloadSetup, EasyDeLSetup)):
        return setup
    if isinstance(setup, SetupScript):
        return setup.to_workload_setup()
    if isinstance(setup, str):
        stripped = setup.strip()
        if stripped.startswith("https://") and not any(character.isspace() for character in stripped):
            return WorkloadSetup(url=stripped)
        return WorkloadSetup(script=setup)
    raise ValueError("setup must be a bash string, an https:// URL, a SetupScript or an EasyDeLSetup")


def setup_request_of(setup: WorkloadSetup | EasyDeLSetup | None) -> WorkloadSetup | None:
    """Return the wire form of a normalized ``setup``, resolving an ``EasyDeLSetup`` now."""

    if isinstance(setup, EasyDeLSetup):
        return setup.to_workload_setup()
    return setup


#: Marks a ``TpuConfig`` whose caller did not name a capacity class. Resolved to
#: spot in ``__post_init__``, so every reader still sees a real ``CapacityClass``.
_UNSET_CAPACITY_CLASS: tp.Final[CapacityClass] = tp.cast(CapacityClass, object())


@dataclass(frozen=True, slots=True)
class TpuConfig:
    """TPU shape and purchase class requested by an Eyvan callable."""

    type: str
    pods: int = 1
    max_retries: int = DEFAULT_MAX_RETRIES
    #: Retries the job's *own* failure, not lost capacity. Zero by default: a
    #: non-zero exit usually repeats, and re-running it costs an accelerator.
    max_failure_retries: int = 0
    #: Restarts after the provider reclaims the capacity. Larger than
    #: ``max_retries`` on purpose: this is the cheap failure to retry, and a spot
    #: job that cannot outlast a handful of reclamations cannot use spot at all.
    max_preemption_retries: int = 10
    #: Spot when not given. Left unset, it is a preference rather than a demand:
    #: a fleet whose pools for this type are reserved and offer no spot (an
    #: attached TPU the caller already holds) runs the job on that reservation
    #: instead of refusing it. Set explicitly, it is exact and never substituted.
    capacity_class: CapacityClass = _UNSET_CAPACITY_CLASS
    #: Secret references, resolved at dispatch and injected as environment
    #: variables on the cell. Unlike a value passed through ``environment``, a
    #: reference is what gets persisted -- which is the whole point.
    secret_mounts: tp.Mapping[str, str] = field(default_factory=dict)
    #: Whether this job can be checkpointed, and the hook that does it. The
    #: scheduler grants ``ON_PREEMPTION`` only when ``checkpointable`` is set and
    #: a hook is named, so a job that declares neither restarts from zero on
    #: every reclamation -- ``max_preemption_retries`` buys it more attempts, not
    #: a resumption. Off by default because a hook that does not exist is worse
    #: than none: the job would be stopped for a checkpoint it never wrote.
    checkpointable: bool = False
    checkpoint_hook: str | None = None
    checkpoint_output_uri: str | None = None
    #: Whether this job receives a capability to submit children of its own,
    #: under its owner's identity. Off by default: a credential nobody asked
    #: for is one nobody is watching.
    can_submit_children: bool = False
    #: Packages installed on the cell before this callable is loaded, over what
    #: the image provides. The wrapper `exec`s into the callable worker, so the
    #: pickled function's imports resolve against them -- which is the point:
    #: adding one library should not mean rebuilding an image.
    dependencies: tuple[str, ...] = ()
    #: Require one exact-shape allocation -- one provider resource, one hardware
    #: world spanning every pod -- instead of the default elastic packing. The
    #: default reuses warm slices of the same accelerator type and provisions
    #: only the deficit, which is the node-pool behaviour; a job whose math
    #: needs the whole allocation in one JAX world (single-mesh training, an
    #: all-reduce across every chip) must say ``atomic=True``. Elastic workloads
    #: run as one lease group with one hardware world per cell, and the job's
    #: receipt states the packing it got.
    atomic: bool = False
    #: Run on this many chips of one TPU host instead of the whole slice, so
    #: several jobs can share one slice. Any free chips are chosen. Requires a
    #: pool with ``chip_sharing: true``.
    chips: int | None = None
    #: Run on exactly these chips (global indexes within the slice: host
    #: ``index // chips_per_host``). Implies ``chips=len(chips_indexes)``; all
    #: indexes must sit on one host. The job waits while any of them is busy.
    chips_indexes: tuple[int, ...] | None = None
    #: A bash step run on every host before the callable loads, inside the job's
    #: environment (``$EYVAN_PYTHON`` is its interpreter). Inline bash, an
    #: ``https://`` URL, a ``SetupScript`` with args, or the built-in
    #: ``EasyDeLSetup``; cached per host by the script's content and args.
    setup: SetupValue = None
    #: Whether the caller named ``capacity_class``. Derived, never passed: only
    #: an unnamed class may be substituted by a fleet's reserved capacity.
    capacity_class_explicit: bool = field(default=True, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.capacity_class is _UNSET_CAPACITY_CLASS:
            object.__setattr__(self, "capacity_class", CapacityClass.SPOT)
            object.__setattr__(self, "capacity_class_explicit", False)
        normalized = _validate_accelerator_request(self, label="TPU", multi_worker=True)
        object.__setattr__(self, "type", normalized)
        self._validate_chips()
        object.__setattr__(self, "setup", workload_setup(self.setup))

    @property
    def setup_request(self) -> WorkloadSetup | None:
        """Return the wire form of ``setup``, or ``None`` when the job has no setup step."""

        return setup_request_of(tp.cast("WorkloadSetup | EasyDeLSetup | None", self.setup))

    def _validate_chips(self) -> None:
        indexes = self.chips_indexes
        if indexes is not None:
            if isinstance(indexes, (str, bytes)) or not isinstance(indexes, (list, tuple)):
                raise ValueError("TPU chips_indexes must be a list of chip indexes")
            if not indexes:
                raise ValueError("TPU chips_indexes must name at least one chip")
            if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indexes):
                raise ValueError("TPU chips_indexes must be non-negative integers")
            if len(set(indexes)) != len(indexes):
                raise ValueError("TPU chips_indexes must not repeat a chip")
            indexes = tuple(sorted(indexes))
            object.__setattr__(self, "chips_indexes", indexes)
            if self.chips is None:
                object.__setattr__(self, "chips", len(indexes))
        if self.chips is None:
            return
        if isinstance(self.chips, bool) or not isinstance(self.chips, int) or self.chips < 1:
            raise ValueError("TPU chips must be a positive integer")
        if indexes is not None and len(indexes) != self.chips:
            raise ValueError("TPU chips must equal len(chips_indexes) when both are given")
        if self.pods != 1:
            raise ValueError("TPU chips runs on one host of one slice; it cannot be combined with pods > 1")
        if self.atomic:
            raise ValueError("TPU chips already runs on one host; atomic=True does not apply")

    @property
    def chip_request(self) -> TpuChipRequest | None:
        """Return the wire form of this chip request, or ``None`` for a whole-slice job."""

        if self.chips is None:
            return None
        return TpuChipRequest(count=self.chips, indexes=self.chips_indexes)


@dataclass(frozen=True, slots=True)
class GpuConfig:
    """GPU shape and purchase class requested by an Eyvan callable.

    Names a pool's accelerator type (``nvidia-tesla-a100``, ``nvidia-h100-80gb``),
    not a GPU count: the fleet YAML owns shape, provider and machine type, and the
    job names a logical pool. That separation is the same one ``TpuConfig`` keeps,
    and it is why this class carries no ``machine_type`` or ``gpu_count``.

    Defaults to on-demand rather than spot, unlike TPU: GPU spot capacity is
    reclaimed far more aggressively, and a silently preemptible default would
    surprise a caller who wrote one line.
    """

    type: str
    max_retries: int = DEFAULT_MAX_RETRIES
    #: Retries the job's *own* failure, not lost capacity. See ``TpuConfig``.
    max_failure_retries: int = 0
    #: Restarts after the provider reclaims the capacity. See ``TpuConfig``.
    max_preemption_retries: int = 10
    capacity_class: CapacityClass = CapacityClass.ON_DEMAND
    #: Fixed at one. Present so the submission path is shape-agnostic.
    pods: int = 1
    #: Secret references, resolved at dispatch and injected as environment
    #: variables on the cell. Unlike a value passed through ``environment``, a
    #: reference is what gets persisted -- which is the whole point.
    secret_mounts: tp.Mapping[str, str] = field(default_factory=dict)
    #: Whether this job can be checkpointed, and the hook that does it. The
    #: scheduler grants ``ON_PREEMPTION`` only when ``checkpointable`` is set and
    #: a hook is named, so a job that declares neither restarts from zero on
    #: every reclamation -- ``max_preemption_retries`` buys it more attempts, not
    #: a resumption. Off by default because a hook that does not exist is worse
    #: than none: the job would be stopped for a checkpoint it never wrote.
    checkpointable: bool = False
    checkpoint_hook: str | None = None
    checkpoint_output_uri: str | None = None
    #: Whether this job receives a capability to submit children of its own,
    #: under its owner's identity. Off by default: a credential nobody asked
    #: for is one nobody is watching.
    can_submit_children: bool = False
    #: Packages installed on the cell before this callable is loaded, over what
    #: the image provides. The wrapper `exec`s into the callable worker, so the
    #: pickled function's imports resolve against them -- which is the point:
    #: adding one library should not mean rebuilding an image.
    dependencies: tuple[str, ...] = ()
    #: A bash step run before the callable loads. See ``TpuConfig.setup``.
    setup: SetupValue = None

    def __post_init__(self) -> None:
        normalized = _validate_accelerator_request(self, label="GPU", multi_worker=False)
        object.__setattr__(self, "type", normalized)
        object.__setattr__(self, "setup", workload_setup(self.setup))

    @property
    def setup_request(self) -> WorkloadSetup | None:
        """Return the wire form of ``setup``, or ``None`` when the job has no setup step."""

        return setup_request_of(tp.cast("WorkloadSetup | EasyDeLSetup | None", self.setup))


@dataclass(frozen=True, slots=True)
class CpuConfig:
    """CPU shape and purchase class requested by an Eyvan callable."""

    type: str = "cpu"
    max_retries: int = DEFAULT_MAX_RETRIES
    #: Retries the job's *own* failure, not lost capacity. See ``TpuConfig``.
    max_failure_retries: int = 0
    #: Restarts after the provider reclaims the capacity. See ``TpuConfig``.
    max_preemption_retries: int = 10
    capacity_class: CapacityClass = CapacityClass.ON_DEMAND
    #: Fixed at one. Present so the submission path is shape-agnostic.
    pods: int = 1
    #: Secret references, resolved at dispatch and injected as environment
    #: variables on the cell. Unlike a value passed through ``environment``, a
    #: reference is what gets persisted -- which is the whole point.
    secret_mounts: tp.Mapping[str, str] = field(default_factory=dict)
    #: Whether this job can be checkpointed, and the hook that does it. The
    #: scheduler grants ``ON_PREEMPTION`` only when ``checkpointable`` is set and
    #: a hook is named, so a job that declares neither restarts from zero on
    #: every reclamation -- ``max_preemption_retries`` buys it more attempts, not
    #: a resumption. Off by default because a hook that does not exist is worse
    #: than none: the job would be stopped for a checkpoint it never wrote.
    checkpointable: bool = False
    checkpoint_hook: str | None = None
    checkpoint_output_uri: str | None = None
    #: Whether this job receives a capability to submit children of its own,
    #: under its owner's identity. Off by default: a credential nobody asked
    #: for is one nobody is watching.
    can_submit_children: bool = False
    #: Packages installed on the cell before this callable is loaded, over what
    #: the image provides. The wrapper `exec`s into the callable worker, so the
    #: pickled function's imports resolve against them -- which is the point:
    #: adding one library should not mean rebuilding an image.
    dependencies: tuple[str, ...] = ()
    #: A bash step run before the callable loads. See ``TpuConfig.setup``.
    setup: SetupValue = None

    def __post_init__(self) -> None:
        normalized = _validate_accelerator_request(self, label="CPU", multi_worker=False)
        object.__setattr__(self, "type", normalized)
        object.__setattr__(self, "setup", workload_setup(self.setup))

    @property
    def setup_request(self) -> WorkloadSetup | None:
        """Return the wire form of ``setup``, or ``None`` when the job has no setup step."""

        return setup_request_of(tp.cast("WorkloadSetup | EasyDeLSetup | None", self.setup))


@dataclass(frozen=True, slots=True)
class CallableSubmission:
    """Durable callable intent passed from the decorator to an SDK adapter."""

    invocation_id: str
    fleet: str
    org_id: str
    project_id: str
    callable_name: str
    #: The hardware request. Named ``tpu`` for wire compatibility; any
    #: ``AcceleratorConfig`` -- TPU, GPU or CPU -- is accepted.
    tpu: AcceleratorConfig
    artifact: StoredCallableArtifact
    result_uri: str
    #: Content address of an uploaded workspace bundle, when the caller opted in
    #: with ``@execute(..., workspace=...)``. ``None`` keeps the historical
    #: behaviour of shipping only the bound function.
    workspace_uri: str | None = None
    workspace_sha256: str | None = None
    #: HMAC of the bundle under the project signing key. Required whenever the
    #: callable itself is signed: the workspace precedes it on ``sys.path``.
    workspace_signature: str | None = None


@dataclass(frozen=True, slots=True)
class JobCompletion:
    """Terminal state returned by an Eyvan job backend."""

    state: WorkloadState
    succeeded: bool
    message: str | None = None


JobProgressWaiter: tp.TypeAlias = tp.Callable[[str, float | None], JobCompletion]


class JobBackend(tp.Protocol):
    """Operations an SDK adapter supplies to a durable ``EyvanJob``."""

    def wait(self, job_id: str, timeout: float | None = None) -> JobCompletion:
        """Wait for terminal state and describe the outcome."""

        ...

    def status(self, job_id: str) -> WorkloadState:
        """Return the current control-plane state."""

        ...

    def logs(self, job_id: str) -> str:
        """Return currently available workload logs."""

        ...

    def cancel(self, job_id: str) -> WorkloadRecord:
        """Request cancellation and return the updated workload."""

        ...


class EyvanJobError(RuntimeError):
    """Raised when a durable callable job does not succeed."""

    def __init__(self, job_id: str, completion: JobCompletion):
        detail = f": {completion.message}" if completion.message else ""
        super().__init__(f"Eyvan job {job_id!r} finished in state {completion.state!r}{detail}")
        self.job_id = job_id
        self.completion = completion


class EyvanPlacementError(RuntimeError):
    """Raised when no enabled capacity pool can satisfy a callable's TPU request.

    ``fleet`` names the fleet that was already selected. It is ``None`` when the
    request failed while auto-resolving a fleet from a :class:`TpuConfig`, in
    which case ``scope`` names the ``org/project`` that was searched.
    """

    def __init__(
        self,
        *,
        requested_accelerator_type: str,
        available_accelerator_types: tp.Iterable[str],
        fleet: str | None = None,
        scope: str | None = None,
        fleets_providing_requested_type: tp.Iterable[str] = (),
        available_fleets: tp.Iterable[str] = (),
    ) -> None:
        self.fleet = fleet
        self.scope = scope
        self.requested_accelerator_type = requested_accelerator_type
        self.available_accelerator_types = sort_accelerator_types(available_accelerator_types)
        self.fleets_providing_requested_type = tuple(sorted(set(fleets_providing_requested_type)))
        self.available_fleets = tuple(sorted(set(available_fleets)))
        if fleet is not None:
            location = f"in selected fleet {fleet!r}"
            empty_detail = "That fleet has no compatible enabled TPU pools for this request."
        else:
            location = "in any enabled fleet" if scope is None else f"in any enabled fleet for {scope}"
            empty_detail = "No enabled fleet exposes a compatible TPU pool for this request."
        parts = [f"Eyvan cannot run TPU {requested_accelerator_type!r} {location}."]
        if self.available_accelerator_types:
            available = ", ".join(self.available_accelerator_types)
            parts.append(f"Compatible TPU types for this request: {available}.")
            parts.append("Change TpuConfig(type=...) to one of those.")
        else:
            parts.append(empty_detail)
        # Name the fleets that would work. The commonest cause of this error is
        # a zone typed where a fleet name belongs -- `asia-east1-c` for
        # `research-asia-east1-c` -- and telling somebody to "select another
        # fleet" without saying which leaves them exactly as stuck, holding a
        # name they already believe is a fleet.
        if self.fleets_providing_requested_type:
            named = ", ".join(self.fleets_providing_requested_type)
            parts.append(f"Fleets providing {requested_accelerator_type!r}: {named}.")
        elif self.available_fleets:
            named = ", ".join(self.available_fleets)
            parts.append(f"No fleet provides it. Fleets in this project: {named}.")
        # One closing instruction, whichever branch ran, so the guidance is not
        # repeated in different words within a single sentence.
        parts.append(
            "Select a fleet with `eyvan use FLEET`, list them with `eyvan capacity fleets`, "
            "or apply one with `eyvan up CONFIG`."
        )
        super().__init__(" ".join(parts))


@dataclass(frozen=True, slots=True)
class EyvanJob(tp.Generic[R_co]):
    """Durable handle returned by an Eyvan callable submission."""

    id: str
    result_uri: str
    backend: JobBackend = field(repr=False, compare=False)
    result_reader: tp.Callable[[str], R_co] = field(repr=False, compare=False)
    progress_waiter: JobProgressWaiter | None = field(default=None, repr=False, compare=False)

    def wait(self, timeout: float | None = None, *, progress: bool = False) -> JobCompletion:
        """Wait until the workload reaches a terminal state.

        Args:
            timeout: Maximum number of seconds to wait, or ``None`` to wait indefinitely.
            progress: Render control-plane lifecycle updates while waiting when the backend supports them.
        """

        if progress and self.progress_waiter is not None:
            return self.progress_waiter(self.id, timeout)
        return self.backend.wait(self.id, timeout)

    def status(self) -> WorkloadState:
        """Return the current workload state."""

        return self.backend.status(self.id)

    def logs(self) -> str:
        """Return currently available workload logs."""

        return self.backend.logs(self.id)

    def cancel(self) -> WorkloadRecord:
        """Request cancellation through the control plane."""

        return self.backend.cancel(self.id)

    def result(self, timeout: float | None = None) -> R_co:
        """Wait for success with lifecycle updates, then decode the durable result artifact."""

        completion = self.wait(timeout, progress=True)
        if not completion.succeeded:
            raise EyvanJobError(self.id, completion)
        return self.result_reader(self.result_uri)


class CallableSubmitter(tp.Protocol[R_co]):
    """SDK integration boundary for native callable workloads."""

    def submit_callable(self, request: CallableSubmission) -> EyvanJob[R_co]:
        """Create one durable workload from ``request``."""

        ...


__all__ = (
    "AcceleratorConfig",
    "CallableSubmission",
    "CallableSubmitter",
    "CpuConfig",
    "EyvanJob",
    "EyvanJobError",
    "EyvanPlacementError",
    "GpuConfig",
    "JobBackend",
    "JobCompletion",
    "JobProgressWaiter",
    "TpuConfig",
    "reject_remote_callable_signature",
    "reject_remote_callable_signing_environment",
    "reject_remote_callable_signing_key",
)
