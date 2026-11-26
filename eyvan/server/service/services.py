# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A service as a coordinator, not a workload kind and not a restart flag.

Two shapes were available and both are worse.

``execution.kind`` selects *how a process is launched*, and a service launches
identically to anything else -- so a ``kind`` would name something that is not a
launching difference.

A ``restart_policy`` flag is worse still. ``WORKLOAD_TRANSITIONS[SUCCEEDED]`` is
an empty frozenset, ``observe_workloads_once`` polls only the five live states,
and ``_release_terminal`` settles lease and budget with ``settled_at`` as the
idempotency boundary. Making one record restart itself requires a
``SUCCEEDED -> PENDING`` edge, which weakens the terminal contract for *every*
consumer -- ``wait_workload`` returns the moment a state is terminal, and
``EyvanJob.result()`` reads the artifact immediately after.

So: a ``schedulable=False`` parent whose children are ordinary, terminal,
single-attempt workloads, with a leader phase maintaining "N live children of
the target revision". "SUCCEEDED means restart me" then never touches the state
machine, and a rolling update never interacts with fencing tokens because each
replica is a different workload.

The desired state lives in the parent's ``spec.metadata`` behind the typed
accessor here, so stringly-typing is confined to one file. A persisted
``ServiceRecord`` costs a protocol, three backends and a migration, and cannot
be undone cheaply -- worth paying once the shape has stopped moving, not before.
"""

from __future__ import annotations

import json
import math
import typing as tp
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from eyvan.records import (
    CellRecord,
    CellState,
    LeaseRecord,
    LeaseState,
    ServiceDecisionKind,
    ServiceDecisionRecord,
    ServiceObservation,
    ServiceProbeOutcome,
    ServiceReplicaHealthRecord,
    ServiceReplicaHealthState,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.server.service.errors import InputError
from eyvan.server.service.state import TERMINAL_WORKLOAD_STATES

#: Metadata keys the service policy is stored under. Prefixed so a user's own
#: metadata cannot collide with control-plane state.
_REPLICAS = "eyvan.service/replicas"
_REVISION = "eyvan.service/revision"
_MAX_UNAVAILABLE = "eyvan.service/max_unavailable"
_MAX_SURGE = "eyvan.service/max_surge"
_BACKOFF_S = "eyvan.service/restart_backoff_s"
#: Health-probe policy. Every one of these defaults to the behaviour a service
#: registered before this existed already has: no path means no probing, and a
#: service that is never probed is never found unhealthy.
_HEALTH_PATH = "eyvan.service/health_path"
_HEALTH_TIMEOUT_S = "eyvan.service/health_timeout_s"
_HEALTH_FAILURE_THRESHOLD = "eyvan.service/health_failure_threshold"
#: Autoscaling policy. ``_AUTOSCALE`` defaults to off, so a service that does
#: not name it keeps exactly the replica count its operator declared.
_AUTOSCALE = "eyvan.service/autoscale"
_MIN_REPLICAS = "eyvan.service/min_replicas"
_MAX_REPLICAS = "eyvan.service/max_replicas"
_SCALE_COOLDOWN_S = "eyvan.service/scale_cooldown_s"
_SCALE_STEP = "eyvan.service/scale_step"
_SCALE_OUT_LOAD = "eyvan.service/scale_out_load"
_SCALE_IN_LOAD = "eyvan.service/scale_in_load"
_LOAD_MAX_AGE_S = "eyvan.service/load_max_age_s"

#: Replica identity, written onto each child so the reconciler can read a
#: child's revision back without a second record type.
CHILD_REVISION = "eyvan.service/child_revision"

#: Ceiling on how long a crash-looping service waits between attempts. Derived
#: statelessly from the children the reconciler already reads, because every
#: resubmission can trigger a provider capacity request -- a service whose image
#: is broken should not ask for accelerators every second.
_MAX_BACKOFF_S = 300.0

#: Live workload states in which a replica claims to hold capacity. A record in
#: one of these with no usable lease is not "starting up"; it is a record whose
#: claim on a host cannot be true.
_PLACED_WORKLOAD_STATES = frozenset(
    {
        WorkloadState.LEASED,
        WorkloadState.DISPATCHING,
        WorkloadState.RUNNING,
        WorkloadState.CHECKPOINTING,
    }
)
#: Lease states from which nothing can come back. ``QUARANTINED`` is deliberately
#: absent: it can return to ACTIVE, so treating it as death would delete hosts
#: that a lease sweep was about to hand back.
_DEAD_LEASE_STATES = frozenset({LeaseState.RELEASED, LeaseState.EXPIRED})


@dataclass(frozen=True, slots=True)
class ServicePolicy:
    """What a service is supposed to look like."""

    replicas: int
    revision: str
    #: How many replicas may be missing during a roll. Default 1 and
    #: ``max_surge`` default 0, because accelerators are scarce: the honest
    #: default is "stop one, then start one", not "start one, then stop one".
    max_unavailable: int = 1
    max_surge: int = 0
    restart_backoff_s: float = 5.0
    #: Path appended to a replica's registered endpoint to probe it. Empty means
    #: no probing at all, which is what every service registered before health
    #: checks existed has -- and what every service still has unless someone
    #: writes a path down.
    health_path: str = ""
    health_timeout_s: float = 2.0
    #: Definite failures in a row before a replica is unhealthy. Never counts
    #: unknowns: see :func:`apply_probe`.
    health_failure_threshold: int = 3
    #: Autoscaling, off unless explicitly turned on. A control plane that starts
    #: resizing services after an upgrade nobody associated with autoscaling is
    #: a control plane operators stop upgrading.
    autoscale: bool = False
    #: Absolute bounds on the declared replica count. ``None`` means *unset*, not
    #: "one": an autoscaler is refused unless both are written down, because a
    #: defaulted ceiling is a defaulted invoice and a defaulted floor is a
    #: defaulted outage.
    min_replicas: int | None = None
    max_replicas: int | None = None
    scale_cooldown_s: float = 300.0
    #: Ceiling on how much one decision may change the count. Bounded on purpose:
    #: an autoscaler that can go from 2 to 200 in one commit is a way to spend a
    #: quarter's accelerator budget in a minute.
    scale_step: int = 1
    scale_out_load: float = 0.8
    scale_in_load: float = 0.3
    #: How old a load observation may be and still be counted. Past this the
    #: autoscaler refuses to decide rather than scaling on a number that predates
    #: the traffic it is supposed to describe.
    load_max_age_s: float = 60.0

    def __post_init__(self) -> None:
        if self.replicas < 0:
            raise InputError("a service cannot have a negative replica count")
        if not self.revision:
            raise InputError("a service requires a revision")
        if self.max_unavailable < 0 or self.max_surge < 0:
            raise InputError("service roll bounds cannot be negative")
        if self.max_unavailable == 0 and self.max_surge == 0 and self.replicas:
            # Neither stopping first nor starting first is permitted, so a roll
            # could never make progress. Refusing here beats a reconciler that
            # silently does nothing forever.
            raise InputError("a service roll requires max_unavailable or max_surge above zero")
        if not math.isfinite(self.restart_backoff_s) or self.restart_backoff_s < 0:
            raise InputError("a service restart backoff must be finite and nonnegative")
        if self.health_path and not self.health_path.startswith("/"):
            raise InputError("a service health path must be absolute")
        if not math.isfinite(self.health_timeout_s) or self.health_timeout_s <= 0:
            raise InputError("a service health timeout must be finite and positive")
        if self.health_failure_threshold < 1:
            # Zero would make the very first definite failure fatal, which is
            # the opposite of "a bounded run of failures".
            raise InputError("a service health failure threshold must be at least one")
        if self.autoscale and (self.min_replicas is None or self.max_replicas is None):
            raise InputError("an autoscaled service must declare explicit min_replicas and max_replicas")
        if (self.min_replicas is not None and self.min_replicas < 0) or (
            self.max_replicas is not None and self.max_replicas < 0
        ):
            raise InputError("service replica bounds cannot be negative")
        if self.min_replicas is not None and self.max_replicas is not None and self.min_replicas > self.max_replicas:
            raise InputError("a service minimum replica count cannot exceed its maximum")
        if not math.isfinite(self.scale_cooldown_s) or self.scale_cooldown_s < 0:
            raise InputError("a service scale cooldown must be finite and nonnegative")
        if self.scale_step < 1:
            # A cap of zero is an autoscaler that is on and can never act, which
            # reads to an operator as one that is broken.
            raise InputError("a service scale step must be at least one")
        if not 0.0 <= self.scale_in_load < self.scale_out_load <= 1.0:
            raise InputError("service scale thresholds must satisfy 0 <= scale_in < scale_out <= 1")
        if not math.isfinite(self.load_max_age_s) or self.load_max_age_s <= 0:
            raise InputError("a service load observation window must be finite and positive")


@dataclass(frozen=True, slots=True)
class ServicePlan:
    """What the reconciler decided, as data rather than as side effects."""

    submit: int = 0
    cancel: tuple[str, ...] = field(default_factory=tuple)
    wait_seconds: float = 0.0
    detail: str = ""


def service_policy(metadata: tp.Mapping[str, str]) -> ServicePolicy | None:
    """Read a service policy out of a parent's metadata, or ``None``.

    ``None`` rather than a raise for a non-service, because every coordinator's
    metadata passes through here and most of them are not services.
    """

    if _REPLICAS not in metadata:
        return None
    try:
        replicas = int(metadata[_REPLICAS])
        return ServicePolicy(
            replicas=replicas,
            revision=metadata.get(_REVISION, ""),
            max_unavailable=int(metadata.get(_MAX_UNAVAILABLE, 1)),
            max_surge=int(metadata.get(_MAX_SURGE, 0)),
            restart_backoff_s=float(metadata.get(_BACKOFF_S, 5.0)),
            health_path=metadata.get(_HEALTH_PATH, ""),
            health_timeout_s=float(metadata.get(_HEALTH_TIMEOUT_S, 2.0)),
            health_failure_threshold=int(metadata.get(_HEALTH_FAILURE_THRESHOLD, 3)),
            autoscale=_flag(metadata.get(_AUTOSCALE, "")),
            # Absent stays absent. Defaulting a bound would let a service turn
            # autoscaling on and inherit a ceiling nobody chose, which is the one
            # mistake here that spends real money.
            min_replicas=_optional_count(metadata.get(_MIN_REPLICAS)),
            max_replicas=_optional_count(metadata.get(_MAX_REPLICAS)),
            scale_cooldown_s=float(metadata.get(_SCALE_COOLDOWN_S, 300.0)),
            scale_step=int(metadata.get(_SCALE_STEP, 1)),
            scale_out_load=float(metadata.get(_SCALE_OUT_LOAD, 0.8)),
            scale_in_load=float(metadata.get(_SCALE_IN_LOAD, 0.3)),
            load_max_age_s=float(metadata.get(_LOAD_MAX_AGE_S, 60.0)),
        )
    except (TypeError, ValueError) as exc:
        raise InputError(f"service policy is malformed: {exc}") from exc


def _flag(value: str) -> bool:
    """Read one metadata flag, refusing anything that is not clearly a boolean.

    Defaulting an unrecognised value to ``False`` would mean a typo silently
    turns autoscaling off and nothing says so; defaulting it to ``True`` is
    worse. The empty string is the only absent value, and it means off.
    """

    if not value:
        return False
    if value in {"on", "true", "1", "yes"}:
        return True
    if value in {"off", "false", "0", "no"}:
        return False
    raise InputError(f"service autoscale flag {value!r} is neither on nor off")


def _optional_count(value: str | None) -> int | None:
    """Read a bound that may legitimately be absent, without inventing one."""

    return None if value is None else int(value)


def service_metadata(policy: ServicePolicy) -> dict[str, str]:
    """Render a policy back into metadata, so one file owns both directions.

    A key is written only when it says something the reader would not otherwise
    conclude. Two reasons, and neither is brevity:

    * A service declared exactly as services were declared before health checks
      and autoscaling existed produces byte-identical metadata, so an upgrade
      changes no stored record and no API response.
    * Metadata values must be non-empty, so an unset health path has no
      representation other than absence.
    """

    rendered = {
        _REPLICAS: str(policy.replicas),
        _REVISION: policy.revision,
        _MAX_UNAVAILABLE: str(policy.max_unavailable),
        _MAX_SURGE: str(policy.max_surge),
        _BACKOFF_S: str(policy.restart_backoff_s),
    }
    default = ServicePolicy(replicas=policy.replicas, revision=policy.revision)
    optional: tuple[tuple[str, object, object], ...] = (
        (_HEALTH_PATH, policy.health_path, default.health_path),
        (_HEALTH_TIMEOUT_S, policy.health_timeout_s, default.health_timeout_s),
        (_HEALTH_FAILURE_THRESHOLD, policy.health_failure_threshold, default.health_failure_threshold),
        (_AUTOSCALE, "on" if policy.autoscale else "off", "off"),
        (_MIN_REPLICAS, policy.min_replicas, default.min_replicas),
        (_MAX_REPLICAS, policy.max_replicas, default.max_replicas),
        (_SCALE_COOLDOWN_S, policy.scale_cooldown_s, default.scale_cooldown_s),
        (_SCALE_STEP, policy.scale_step, default.scale_step),
        (_SCALE_OUT_LOAD, policy.scale_out_load, default.scale_out_load),
        (_SCALE_IN_LOAD, policy.scale_in_load, default.scale_in_load),
        (_LOAD_MAX_AGE_S, policy.load_max_age_s, default.load_max_age_s),
    )
    for name, value, fallback in optional:
        if value != fallback and value is not None:
            rendered[name] = str(value)
    return rendered


def scaled_service_metadata(metadata: tp.Mapping[str, str], *, replicas: int) -> dict[str, str]:
    """Return one service's metadata with a new declared replica count.

    The declared count stays the single source of truth for how large a service
    is, whether an operator or the autoscaler last wrote it. A parallel
    "autoscaler's desired count" would be a second answer to the one question a
    service reconciler asks, and the two would disagree the first time anyone
    edited the service by hand.
    """

    if replicas < 0:
        raise InputError("a service cannot be scaled to a negative replica count")
    return {**metadata, _REPLICAS: str(replicas)}


def child_revision(child: WorkloadRecord) -> str:
    """Return the revision a replica was started for."""

    return child.spec.metadata.get(CHILD_REVISION, "")


def plan_service(
    policy: ServicePolicy,
    children: tp.Sequence[WorkloadRecord],
    *,
    now: datetime,
) -> ServicePlan:
    """Decide what one service needs, as a pure function of what it has.

    Pure on purpose: reconciliation that reads and writes in the same breath is
    the kind of code where "what would it do here" can only be answered by
    running it against a database. This takes state and returns intent; a thin
    leader-only wrapper performs one action per transaction.
    """

    live = [child for child in children if child.state not in TERMINAL_WORKLOAD_STATES]
    current = [child for child in live if child_revision(child) == policy.revision]
    stale = [child for child in live if child_revision(child) != policy.revision]

    # Scaling to zero is a valid desired state, and it is not a roll: stop
    # everything and stop.
    if policy.replicas == 0:
        if live:
            return ServicePlan(cancel=tuple(child.id for child in live), detail="scaling to zero")
        return ServicePlan(detail="stopped")

    if stale:
        # Occupancy bounds surge; only running replicas contribute availability.
        # A pending replacement cannot justify stopping the old serving process,
        # and a cancellation already in flight has spent its allowance.
        floor = policy.replicas - policy.max_unavailable
        serving = [child for child in live if child.state is WorkloadState.RUNNING]
        allowed_down = max(0, len(serving) - floor)
        retiring = [child.id for child in stale if child.state not in {WorkloadState.RUNNING, WorkloadState.CANCELLING}]
        retiring.extend(child.id for child in stale if child.state is WorkloadState.RUNNING)
        nonserving = sum(child.state not in {WorkloadState.RUNNING, WorkloadState.CANCELLING} for child in stale)
        retiring = retiring[: nonserving + allowed_down]
        if retiring:
            return ServicePlan(cancel=tuple(retiring), detail=f"rolling to {policy.revision}")
        if len(current) + len(stale) < policy.replicas + policy.max_surge:
            return ServicePlan(submit=1, detail=f"surging to {policy.revision}")
        return ServicePlan(detail=f"waiting for a replica to stop before rolling to {policy.revision}")

    missing = policy.replicas - len(current)
    if missing > 0:
        wait = _restart_backoff(policy, children, now=now)
        if wait > 0:
            return ServicePlan(wait_seconds=wait, detail="backing off after repeated replica failures")
        return ServicePlan(submit=missing, detail=f"starting {missing} replica(s)")
    if missing < 0:
        return ServicePlan(cancel=tuple(child.id for child in current[missing:]), detail="scaling in")
    return ServicePlan(detail="steady")


def _restart_backoff(policy: ServicePolicy, children: tp.Sequence[WorkloadRecord], *, now: datetime) -> float:
    """Return how long to wait before replacing a replica that keeps dying.

    Derived from the children already in hand rather than from stored counters:
    a service whose image does not exist would otherwise ask a provider for
    accelerators once per reconcile cycle, forever.

    Only *consecutive recent* failures count. A service that has been up for a
    week and loses one replica restarts immediately, which is the behaviour an
    operator expects and the one a naive failure count would break.
    """

    window = now - timedelta(seconds=_MAX_BACKOFF_S)
    recent = [child for child in children if child.state is WorkloadState.FAILED and child.updated_at >= window]
    if not recent:
        return 0.0
    last = max(child.updated_at for child in recent)
    delay = policy.restart_backoff_s
    if delay > 0:
        steps = len(recent) - 1
        # Compare binary exponents before scaling. Even a subnormal base can
        # need more than 1024 doublings, where constructing 2**steps and then
        # converting it to a float overflows before the ceiling is applied.
        _, base_exponent = math.frexp(delay)
        _, ceiling_exponent = math.frexp(_MAX_BACKOFF_S)
        if base_exponent + steps > ceiling_exponent:
            delay = _MAX_BACKOFF_S
        else:
            delay = min(math.ldexp(delay, steps), _MAX_BACKOFF_S)
    elapsed = (now - last).total_seconds()
    return max(0.0, delay - elapsed)


# --------------------------------------------------------------------------- health
#
# A service today is a registered address. Nothing proves the address answers,
# and the two obvious ways to prove it are both wrong.
#
# Inferring health from the workload state machine proves only that a process
# started -- which for a model server is minutes before it can serve, and stays
# true for as long as a wedged process keeps its file descriptors open.
#
# Treating an unreachable probe as a failure is worse: it makes the control
# plane's own view of the network into a verdict about someone else's process,
# so a partition, a DNS blip or a saturated egress link deletes healthy
# accelerators. That is why the outcome below has three values and not two.


@dataclass(frozen=True, slots=True)
class ServiceProbeResult:
    """What one probe attempt established, and what it could not."""

    outcome: ServiceProbeOutcome
    detail: str = ""
    #: Utilization the replica reported about itself, when it reported one.
    #: ``None`` is not zero: an endpoint that does not report load leaves the
    #: autoscaler without an input, and inventing one is inventing the number a
    #: scaling decision is multiplied by.
    load: float | None = None


class ServiceHealthProbe(tp.Protocol):
    """Ask one replica whether it is serving.

    A protocol rather than a concrete client because the transport is the one
    part of health checking that cannot be tested without a network, and because
    a fleet whose replicas speak something other than HTTP should be able to say
    so without forking the reconciler.
    """

    async def __call__(self, *, endpoint: str, path: str, timeout_s: float) -> ServiceProbeResult: ...


def probe_from_response(status: int, body: str) -> ServiceProbeResult:
    """Classify one *completed* HTTP exchange with a replica.

    A completed exchange is the only thing that can be a definite failure: the
    process was reached, it was asked, and it answered that it is not serving.
    Everything short of that is :func:`probe_unreachable`.
    """

    if 200 <= status < 300:
        return ServiceProbeResult(
            outcome=ServiceProbeOutcome.SUCCESS,
            detail=f"health endpoint answered {status}",
            load=reported_load(body),
        )
    return ServiceProbeResult(
        outcome=ServiceProbeOutcome.FAILURE,
        detail=f"health endpoint answered {status}",
    )


def probe_unreachable(detail: str) -> ServiceProbeResult:
    """Record that a probe established nothing.

    Timeouts, refused connections, TLS errors and DNS failures all land here.
    None of them is evidence about the replica; every one of them is evidence
    about this control plane's ability to reach it, which is a different fact
    and must never be recorded as the first one.
    """

    return ServiceProbeResult(outcome=ServiceProbeOutcome.UNKNOWN, detail=detail[:512])


def reported_load(body: str) -> float | None:
    """Read a replica's self-reported utilization out of a health response.

    Returns ``None`` for anything that is not a number in ``[0, 1]`` -- an empty
    body, a body that is not JSON, a missing key, a string, a NaN. Every one of
    those means the replica did not report its load, and the honest record of
    that is an absence rather than a default.
    """

    try:
        document = json.loads(body)
    except (TypeError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    value = document.get("load")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    load = float(value)
    if not math.isfinite(load) or not 0.0 <= load <= 1.0:
        return None
    return load


def apply_probe(
    previous: ServiceReplicaHealthRecord | None,
    result: ServiceProbeResult,
    *,
    service_id: str,
    replica_id: str,
    replica_generation: int,
    endpoint: str,
    now: datetime,
) -> ServiceReplicaHealthRecord:
    """Fold one probe result into a replica's durable counters.

    Pure, so "what would three failures and a timeout do here" is answerable
    without a network and without a database.

    Two rules carry the whole design:

    * A record gathered at a different replica generation is discarded, not
      merged. A requeued replica is a different process on a different host, and
      carrying its predecessor's failures forward would let a machine that has
      never answered a probe be declared unhealthy on evidence about a machine
      that no longer exists.
    * ``consecutive_failures`` counts definite failures *since the last definite
      success*. An unknown probe neither raises it nor clears it. Raising it
      would record "I could not reach this" as "this said no"; clearing it would
      grant a recovery the endpoint never demonstrated. Doing neither is the only
      option that adds no evidence, which is exactly what an unknown is.
    """

    if previous is not None and (previous.replica_generation != replica_generation or previous.endpoint != endpoint):
        previous = None
    failures = 0 if previous is None else previous.consecutive_failures
    unknowns = 0 if previous is None else previous.consecutive_unknowns
    last_success = None if previous is None else previous.last_success_at
    last_failure = None if previous is None else previous.last_failure_at
    load = None if previous is None else previous.observed_load
    load_at = None if previous is None else previous.load_observed_at
    if result.outcome is ServiceProbeOutcome.SUCCESS:
        failures, unknowns, last_success = 0, 0, now
        if result.load is not None:
            load, load_at = result.load, now
    elif result.outcome is ServiceProbeOutcome.FAILURE:
        failures, unknowns, last_failure = failures + 1, 0, now
    else:
        unknowns += 1
    return ServiceReplicaHealthRecord(
        service_id=service_id,
        replica_id=replica_id,
        replica_generation=replica_generation,
        endpoint=endpoint,
        last_outcome=result.outcome,
        last_detail=result.detail[:512],
        consecutive_failures=failures,
        consecutive_unknowns=unknowns,
        last_success_at=last_success,
        last_failure_at=last_failure,
        observed_at=now,
        observed_load=load,
        load_observed_at=load_at,
    )


def replica_health_state(
    replica: WorkloadRecord,
    health: ServiceReplicaHealthRecord | None,
    policy: ServicePolicy,
) -> ServiceReplicaHealthState:
    """Return what is actually known about one replica's health.

    ``UNKNOWN`` is the default and is returned for every case where the answer
    has not been established: a service nobody probes, a replica nobody has
    probed yet, a replica whose only observations belong to an earlier
    generation, and a replica whose last probe could not complete. None of those
    is a reason to destroy anything.
    """

    if not policy.health_path:
        return ServiceReplicaHealthState.UNKNOWN
    if health is None or health.replica_generation != replica.generation or health.endpoint != replica.service_endpoint:
        return ServiceReplicaHealthState.UNKNOWN
    if health.consecutive_failures >= policy.health_failure_threshold:
        return ServiceReplicaHealthState.UNHEALTHY
    if health.last_outcome is ServiceProbeOutcome.SUCCESS:
        return ServiceReplicaHealthState.HEALTHY
    return ServiceReplicaHealthState.UNKNOWN


# --------------------------------------------------------------------------- replica lifecycle


@dataclass(frozen=True, slots=True)
class ReplicaObservation:
    """One replica and the durable records its claim on a host depends on.

    Whole records rather than pre-digested booleans: the decision below is the
    thing under test, and a test that has to reproduce the digesting first is a
    test of the digesting.
    """

    replica: WorkloadRecord
    #: The replica's current lease, or ``None`` when the row is absent.
    lease: LeaseRecord | None = None
    #: The cell that lease names, or ``None`` when the row is absent.
    cell: CellRecord | None = None
    health: ServiceReplicaHealthRecord | None = None


@dataclass(frozen=True, slots=True)
class ReplicaVerdict:
    """What one replica needs, with the reason it needs it."""

    replica_id: str
    #: The generation the verdict was reached at. The acting transaction refuses
    #: unless the record is still at it, which is what makes a replacement a
    #: replacement rather than a second way to kill a live process.
    generation: int
    replace: bool
    reason: str
    health: ServiceReplicaHealthState = ServiceReplicaHealthState.UNKNOWN


def plan_replica_lifecycle(
    policy: ServicePolicy,
    observations: tp.Sequence[ReplicaObservation],
) -> tuple[ReplicaVerdict, ...]:
    """Decide which replicas no longer own the host they claim, and why.

    Pure, and it returns a verdict for *every* replica rather than only for the
    doomed ones, so "why was this one kept" is answerable from the same call
    that answers "why was this one replaced".

    The rules are all of the same shape: a replica is replaced when a durable
    record proves its claim on a host cannot still be true. Nothing here replaces
    on a *missing* answer -- an unknown health probe, a lease that is merely
    quarantined, a cell that is merely draining -- because a missing answer is
    not proof of anything, and the cost of being wrong is a deleted accelerator.
    """

    verdicts: list[ReplicaVerdict] = []
    for observation in observations:
        verdicts.append(_replica_verdict(policy, observation))
    return tuple(verdicts)


def _replica_verdict(policy: ServicePolicy, observation: ReplicaObservation) -> ReplicaVerdict:
    """Return the verdict for exactly one replica."""

    replica = observation.replica
    health = replica_health_state(replica, observation.health, policy)

    def verdict(replace: bool, reason: str) -> ReplicaVerdict:
        return ReplicaVerdict(
            replica_id=replica.id,
            generation=replica.generation,
            replace=replace,
            reason=reason,
            health=health,
        )

    if replica.state in TERMINAL_WORKLOAD_STATES:
        # Already gone; the replica-count planner starts its successor.
        return verdict(False, f"already {replica.state.value}")
    if replica.state is WorkloadState.CANCELLING:
        return verdict(False, "already retiring")
    if child_revision(replica) != policy.revision:
        # A superseded replica is the roll's business, and taking it here would
        # retire it outside the availability bound the roll exists to hold.
        return verdict(False, "belongs to a superseded revision; the roll owns it")
    if replica.state not in _PLACED_WORKLOAD_STATES:
        return verdict(False, f"waiting to be placed ({replica.state.value})")
    if replica.lease_id is None:
        return verdict(True, f"is {replica.state.value} while holding no lease")
    if observation.lease is None:
        return verdict(True, f"lease {replica.lease_id!r} no longer exists")
    lease = observation.lease
    if lease.workload_id != replica.id:
        return verdict(True, f"lease {lease.id!r} belongs to workload {lease.workload_id!r}")
    if lease.state in _DEAD_LEASE_STATES:
        return verdict(True, f"lease {lease.id!r} is {lease.state.value}")
    if observation.cell is None:
        return verdict(True, f"execution cell {lease.cell_id!r} no longer exists")
    cell = observation.cell
    if cell.generation != lease.cell_generation:
        # The host was recreated under the lease. Whatever is running there now,
        # it is not this replica's process.
        return verdict(
            True,
            f"execution cell {cell.id!r} moved to generation {cell.generation} "
            f"from the {lease.cell_generation} its lease was fenced at",
        )
    if cell.state is CellState.DELETED:
        return verdict(True, f"execution cell {cell.id!r} is deleted")
    if health is ServiceReplicaHealthState.UNHEALTHY:
        assert observation.health is not None
        return verdict(
            True,
            f"failed {observation.health.consecutive_failures} definite health probes in a row "
            f"against a threshold of {policy.health_failure_threshold}",
        )
    return verdict(False, f"holds lease {lease.id!r} on cell {cell.id!r}; health is {health.value}")


# --------------------------------------------------------------------------- autoscaling


@dataclass(frozen=True, slots=True)
class ServiceScaleDecision:
    """A scaling decision, including the decision to do nothing.

    Every path returns one of these with a reason, so "why did it not scale" has
    an answer that is not "read the reconciler". Only the ones that change the
    count are committed; a reason that recurs every five seconds is a log line,
    not a durable row.
    """

    from_replicas: int
    to_replicas: int
    reason: str
    observed: ServiceObservation
    kind: ServiceDecisionKind | None = None

    @property
    def changes(self) -> bool:
        """Return whether this decision alters the declared replica count."""

        return self.kind is not None


def service_observation(
    policy: ServicePolicy,
    observations: tp.Sequence[ReplicaObservation],
) -> ServiceObservation:
    """Summarize the durable facts a scaling decision is allowed to use.

    Only replicas of the *current* revision and only non-terminal ones count. A
    draining old-revision replica still serving traffic is real, but sizing the
    new revision from it means the count jumps the moment the roll finishes.
    """

    live = [
        observation
        for observation in observations
        if observation.replica.state not in TERMINAL_WORKLOAD_STATES
        and child_revision(observation.replica) == policy.revision
    ]
    healthy = unhealthy = unknown = 0
    loads: list[float] = []
    oldest: datetime | None = None
    for observation in live:
        state = replica_health_state(observation.replica, observation.health, policy)
        if state is ServiceReplicaHealthState.HEALTHY:
            healthy += 1
        elif state is ServiceReplicaHealthState.UNHEALTHY:
            unhealthy += 1
        else:
            unknown += 1
        health = observation.health
        if (
            health is not None
            and health.observed_load is not None
            and health.load_observed_at is not None
            and health.replica_generation == observation.replica.generation
            and health.endpoint == observation.replica.service_endpoint
        ):
            loads.append(health.observed_load)
            oldest = health.load_observed_at if oldest is None else min(oldest, health.load_observed_at)
    average = min(1.0, sum(loads) / len(loads)) if loads else None
    return ServiceObservation(
        live_replicas=len(live),
        healthy_replicas=healthy,
        unhealthy_replicas=unhealthy,
        unknown_replicas=unknown,
        loaded_replicas=len(loads),
        average_load=average,
        oldest_load_observed_at=oldest,
    )


def plan_autoscale(
    policy: ServicePolicy,
    observed: ServiceObservation,
    *,
    now: datetime,
    last_decision: ServiceDecisionRecord | None,
) -> ServiceScaleDecision:
    """Decide a service's declared replica count from durable observations only.

    Pure by construction: every input is either policy an operator wrote or a row
    somebody committed, so the whole decision table is testable without a
    provider, a network, or a clock that moves.

    Bounded in four independent ways, because an autoscaler that is wrong is
    wrong at the speed of the control loop:

    * ``min_replicas`` / ``max_replicas`` -- absolute floor and ceiling.
    * ``scale_step`` -- how far one decision may move, whatever the load says.
    * ``scale_cooldown_s`` -- how soon after the *last committed decision*,
      including a replacement, another may be taken.
    * staleness -- a decision is refused outright unless every live replica has
      reported its load inside the observation window. Scaling on a partial or
      old view is how an autoscaler removes the replicas that were about to
      report the spike.
    """

    declared = policy.replicas

    def hold(reason: str) -> ServiceScaleDecision:
        return ServiceScaleDecision(from_replicas=declared, to_replicas=declared, reason=reason, observed=observed)

    def move(target: int, reason: str) -> ServiceScaleDecision:
        if target == declared:
            return hold(reason)
        return ServiceScaleDecision(
            from_replicas=declared,
            to_replicas=target,
            reason=reason,
            observed=observed,
            kind=ServiceDecisionKind.SCALE_OUT if target > declared else ServiceDecisionKind.SCALE_IN,
        )

    if not policy.autoscale:
        return hold("autoscaling is off for this service")
    minimum, maximum = policy.min_replicas, policy.max_replicas
    if minimum is None or maximum is None:
        # Unreachable through ServicePolicy, which refuses an autoscaled service
        # without both bounds. Kept because an unbounded autoscaler is the one
        # failure here that is measured in money.
        return hold("autoscaling is enabled without explicit replica bounds")
    if last_decision is not None:
        elapsed = (now - last_decision.decided_at).total_seconds()
        if elapsed < policy.scale_cooldown_s:
            remaining = policy.scale_cooldown_s - elapsed
            return hold(
                f"cooling down for another {remaining:.0f}s after the {last_decision.kind.value} "
                f"at sequence {last_decision.sequence}"
            )
    if declared < minimum:
        return move(
            min(declared + policy.scale_step, minimum),
            f"declared {declared} replicas is below the minimum of {minimum}",
        )
    if declared > maximum:
        return move(
            max(declared - policy.scale_step, maximum),
            f"declared {declared} replicas is above the maximum of {maximum}",
        )
    if observed.live_replicas == 0:
        # Nothing is running, so nothing can report load. Saying so beats
        # treating "no observations" as "no load", which scales to the minimum
        # exactly when a service is trying to come back.
        return hold("no live replicas are reporting, so there is no load to scale on")
    if observed.loaded_replicas != observed.live_replicas:
        return hold(
            f"only {observed.loaded_replicas} of {observed.live_replicas} live replicas reported a load; "
            "an incomplete view is not a basis for resizing"
        )
    oldest = observed.oldest_load_observed_at
    assert oldest is not None  # guaranteed by ServiceObservation when loaded_replicas > 0
    age = (now - oldest).total_seconds()
    if age < 0:
        return hold(f"a load observation is dated {abs(age):.0f}s in the future; the clocks disagree")
    if age > policy.load_max_age_s:
        return hold(
            f"the oldest counted load observation is {age:.0f}s old, past the {policy.load_max_age_s:.0f}s window"
        )
    average = observed.average_load
    assert average is not None  # guaranteed by ServiceObservation when loaded_replicas > 0
    if average >= policy.scale_out_load:
        return move(
            min(declared + policy.scale_step, maximum),
            f"mean load {average:.2f} over {observed.loaded_replicas} replicas is at or above "
            f"the scale-out threshold {policy.scale_out_load:.2f}",
        )
    if average <= policy.scale_in_load:
        return move(
            max(declared - policy.scale_step, minimum),
            f"mean load {average:.2f} over {observed.loaded_replicas} replicas is at or below "
            f"the scale-in threshold {policy.scale_in_load:.2f}",
        )
    return hold(f"mean load {average:.2f} is inside the [{policy.scale_in_load:.2f}, {policy.scale_out_load:.2f}] band")


__all__ = (
    "CHILD_REVISION",
    "ReplicaObservation",
    "ReplicaVerdict",
    "ServiceHealthProbe",
    "ServicePlan",
    "ServicePolicy",
    "ServiceProbeResult",
    "ServiceScaleDecision",
    "apply_probe",
    "child_revision",
    "plan_autoscale",
    "plan_replica_lifecycle",
    "plan_service",
    "probe_from_response",
    "probe_unreachable",
    "replica_health_state",
    "reported_load",
    "scaled_service_metadata",
    "service_metadata",
    "service_observation",
    "service_policy",
)
