# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Kueue coexistence: the precondition that makes it safe, made checkable --
and the cluster's real free capacity, read as an observation rather than assumed.

Two admission controllers over the same nodes fight twice, and only one of the
two fights is survivable.

**Kueue vetoing a cell Eyvan already leased** is safe but wasteful: Eyvan's
placement is overridden and the cell sits WAITING.

**Kueue *evicting* an admitted Workload** is not. Eviction destroys a pod with no
checkpoint hook, bypassing Eyvan's entire ``PreemptionPlan`` /
``CHECKPOINT_AND_STOP`` / ``grace_period_seconds`` machinery -- a training job
loses hours of progress and nothing in Eyvan's ledger records why.

So the boundary is:

    **Eyvan delegates *capacity* admission to Kueue and retains *workload*
    admission. A ClusterQueue Eyvan uses must be exclusively Eyvan's, with
    preemption disabled.**

The value of that precondition is that it is *checkable*. An invisible
correctness hazard becomes a refusal at the point of use:
``KubernetesCapacityDriver`` reads the ClusterQueue before creating a Job and
refuses a configuration that could evict Eyvan's work. The driver is where the
check belongs because it runs in-cluster with a ServiceAccount; ``eyvan doctor``
reports the same hazard but can only warn, since it runs on an operator's
machine with no cluster access.

--------------------------------------------------------------------------------

**The second half of this module is cluster capacity awareness (W3).**
``capacity/drivers/kubernetes.py`` states the cost plainly: Eyvan's budgets and
fair share reason about cells it *asked for*, not about the cluster's real free
capacity, so an Eyvan pool can sit WAITING while the cluster is full of
non-Eyvan work. ``CapacityBelief`` learns that from repeated timeouts; nothing
ever *planned* around it.

What is read, and what each reading is worth:

* ``ClusterQueue.spec.resourceGroups[].flavors[].resources[].nominalQuota`` is
  the ceiling Kueue admits against, and
  ``ClusterQueue.status.flavorsReservation`` is what admitted Workloads already
  hold. Free capacity is the difference -- and it is the *exact* number Kueue
  itself subtracts, not an estimate, which is why reservation is used and
  ``flavorsUsage`` (a lagging view of running pods) is not.
* ``ResourceFlavor.spec.nodeLabels`` says which nodes back that quota, so
  ``Node.status.allocatable`` can show a nominal quota that no node can honour.
  That is a real and common misconfiguration and it is worth naming; it is not
  worth *planning* on, because allocatable is a ceiling and nothing here reads
  pods, so it never says how much of a node is free.

The rule the whole design is organised around:

    **An unreadable cluster is UNKNOWN and changes nothing. It is never read as
    full, and never read as empty.**

Three concrete consequences, each of which is a test:

1. A ClusterQueue whose ``status`` has not been populated is UNKNOWN. Reading a
   missing ``flavorsReservation`` as "nothing is reserved" would report an
   entire cluster as free at exactly the moment Eyvan knows least about it.
2. An RBAC refusal is UNKNOWN. A ServiceAccount without ``get clusterqueues`` is
   the ordinary state of a cluster where Eyvan was installed before Kueue.
3. Free capacity in a *cohort* is a lower bound, because borrowing depends on
   other queues Eyvan cannot see. A lower bound proves a cell fits; it never
   proves one does not. Only an exact reading can conclude "full", which is why
   :class:`FlavorHeadroom` carries ``exact`` separately from ``free``.

**Where this plugs in.** The signal is a library surface: it reads, it explains,
and it advises, and it holds no state and schedules nothing. Two call sites turn
it into behaviour, and both live in files outside this change:

* ``capacity/drivers/kubernetes.py``'s ``observe`` calls ``capacity_state(status)``.
  Calling :func:`cell_capacity_state` with an explanation instead is what makes a
  cell WAITING behind a saturated cluster *say so* in ``CellObservation.detail``.
* the autoscale pass would take one reading per Kubernetes pool and pass it to
  ``eyvan.capacity.autoscale.cluster_capacity.advise_launch``, which is
  default-off and returns the caller's own number until an operator enables it.

Neither is wired here, and no behaviour changes without them.
"""

from __future__ import annotations

import math
import re
import typing as tp
from dataclasses import dataclass, field
from datetime import datetime

from eyvan._internal.compat import UTC, StrEnum
from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan.capacity import CapacityStatus
from eyvan.capacity.providers.kubernetes.api import ApiObservation, JobStatus, KubernetesApi, capacity_state
from eyvan.capacity.providers.kubernetes.workers import CELL_LABEL

#: Preemption policies that leave Eyvan's workloads safe. ``Never`` is the only
#: setting under which Kueue will not destroy a running pod out from under the
#: control plane that thinks it owns it.
SAFE_PREEMPTION = frozenset({"Never", ""})

#: Kubernetes quantity suffixes. Binary suffixes are exact powers of two; the
#: decimal ones include the sub-unit suffixes a CPU request uses (``250m``).
_QUANTITY_SUFFIXES: dict[str, float] = {
    "": 1.0,
    "n": 1e-9,
    "u": 1e-6,
    "m": 1e-3,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
    "P": 1e15,
    "E": 1e18,
    "Ki": 2.0**10,
    "Mi": 2.0**20,
    "Gi": 2.0**30,
    "Ti": 2.0**40,
    "Pi": 2.0**50,
    "Ei": 2.0**60,
}
_QUANTITY = re.compile(r"^(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+))(?P<suffix>[a-zA-Z]*)$")

#: How many resources one explanation names before it stops. An explanation an
#: operator will not read is not actionable.
_MAX_EXPLAINED_RESOURCES = 4


@dataclass(frozen=True, slots=True)
class KueueFinding:
    """One reason a ClusterQueue is unsafe for Eyvan to use."""

    cluster_queue: str
    problem: str
    remedy: str


def inspect_cluster_queue(payload: JsonObject | None, name: str) -> tuple[KueueFinding, ...]:
    """Report why one ClusterQueue is unsafe, if it is.

    A missing ClusterQueue is a finding, not an empty result: a pool configured
    against a queue that does not exist will suspend every Job forever, and
    reporting "no problems" for that would be actively misleading.
    """

    if payload is None:
        return (
            KueueFinding(
                cluster_queue=name,
                problem="the ClusterQueue does not exist",
                remedy=f"create ClusterQueue {name!r}, or remove queue_name from the pool's provider_options",
            ),
        )
    spec = payload.get("spec") if is_json_object(payload.get("spec")) else {}
    assert isinstance(spec, dict)
    preemption = spec.get("preemption") if is_json_object(spec.get("preemption")) else {}
    assert isinstance(preemption, dict)
    findings: list[KueueFinding] = []
    within = str(preemption.get("withinClusterQueue", ""))
    cohort = str(preemption.get("reclaimWithinCohort", ""))
    if within not in SAFE_PREEMPTION:
        findings.append(
            KueueFinding(
                cluster_queue=name,
                problem=f"preemption.withinClusterQueue is {within!r}",
                remedy="set it to 'Never'; Kueue eviction destroys pods with no checkpoint hook",
            )
        )
    if cohort not in SAFE_PREEMPTION:
        findings.append(
            KueueFinding(
                cluster_queue=name,
                problem=f"preemption.reclaimWithinCohort is {cohort!r}",
                remedy="set it to 'Never'; another cohort's queue must not reclaim Eyvan's capacity",
            )
        )
    return tuple(findings)


def foreign_workloads(workloads: tp.Sequence[JsonObject]) -> tuple[str, ...]:
    """Return the names of Workloads in the queue that are not Eyvan's.

    Identified by the absence of the cell label rather than by a positive marker
    of foreignness: anything Eyvan created carries the label, so absence is the
    only test that cannot be spoofed into looking like Eyvan's own work.
    """

    foreign: list[str] = []
    for workload in workloads:
        if not is_json_object(workload):
            continue
        metadata = workload.get("metadata") if is_json_object(workload.get("metadata")) else {}
        assert isinstance(metadata, dict)
        labels = metadata.get("labels") if is_json_object(metadata.get("labels")) else {}
        assert isinstance(labels, dict)
        if not labels.get(CELL_LABEL):
            foreign.append(str(metadata.get("name", "<unnamed>")))
    return tuple(sorted(foreign))


# --- cluster capacity: the observed signal -----------------------------------


class ClusterCapacityState(StrEnum):
    """Whether Eyvan established the cluster's free capacity, or did not.

    There is no third value on purpose. "Full" and "empty" are *conclusions*
    drawn from an OBSERVED reading; they are never states in their own right,
    because a state named ``FULL`` is a state something eventually sets when it
    fails to read the cluster.
    """

    OBSERVED = "observed"
    UNKNOWN = "unknown"


def parse_quantity(value: JsonValue) -> float | None:
    """Return one Kubernetes resource quantity as a float, or ``None``.

    ``None`` means *unparseable*, which callers must propagate as UNKNOWN rather
    than substitute with zero: a quantity Eyvan cannot read is not a quantity of
    nothing.

    Args:
        value: A decoded quantity -- ``"4"``, ``4``, ``"250m"``, ``"32Gi"``,
            ``"1e3"`` -- as it appears in an API response.

    Returns:
        The value in base units (cores, bytes, whole accelerators), or ``None``
        when it is absent, negative, non-finite, or not a quantity at all.
    """

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    match = _QUANTITY.match(text)
    if match is not None:
        multiplier = _QUANTITY_SUFFIXES.get(match.group("suffix"))
        if multiplier is not None:
            number = float(match.group("number")) * multiplier
            return number if math.isfinite(number) and number >= 0 else None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) and number >= 0 else None


@dataclass(frozen=True, slots=True)
class FlavorHeadroom:
    """Free capacity for one covered resource in one ResourceFlavor.

    Attributes:
        flavor: ResourceFlavor name the quota is expressed in.
        resource: Kubernetes resource name, such as ``google.com/tpu``.
        nominal_quota: What the ClusterQueue may admit against.
        reserved: What already-admitted Workloads hold, from
            ``status.flavorsReservation``. Kueue's own admission arithmetic uses
            this number.
        exact: Whether ``free`` is the whole truth. ``False`` when the queue sits
            in a cohort and may borrow more than its nominal quota, which makes
            ``free`` a *lower bound*: enough to prove a cell fits, never enough
            to prove one does not.
    """

    flavor: str
    resource: str
    nominal_quota: float
    reserved: float
    exact: bool
    #: Which ``spec.resourceGroups`` entry this came from. Kueue assigns one
    #: flavor per resource *group*, so two resources in the same group must be
    #: satisfied by the same flavor while resources in different groups need
    #: not be. Losing the grouping turns a fit into a guess.
    group: int = 0

    @property
    def free(self) -> float:
        """Return free capacity, floored at zero because a queue cannot owe."""

        return max(self.nominal_quota - self.reserved, 0.0)


@dataclass(frozen=True, slots=True)
class NodeAllocatable:
    """Allocatable totals for the nodes backing one ResourceFlavor.

    This is a *ceiling*, never a balance: nothing here reads pods, so it cannot
    say how much of a node is free. Its job is to expose a nominal quota no node
    can honour, which is a misconfiguration an operator can fix and which
    otherwise presents as a pool that queues forever.

    Attributes:
        state: OBSERVED only when the node list was actually read.
        reason: Why nothing was established, when ``state`` is UNKNOWN.
        allocatable: Resource name to allocatable total over schedulable, Ready
            nodes.
        node_count: Nodes that contributed to ``allocatable``.
        complete: Whether every node in scope was seen and classified. ``False``
            when the list was truncated or a node's readiness was itself
            ``Unknown``, which makes ``allocatable`` a lower bound.
    """

    state: ClusterCapacityState
    reason: str = ""
    allocatable: dict[str, float] = field(default_factory=dict)
    node_count: int = 0
    complete: bool = True


@dataclass(frozen=True, slots=True)
class ClusterQueueCapacity:
    """One ClusterQueue's free capacity, as an observation that may be UNKNOWN.

    Attributes:
        cluster_queue: The queue this describes.
        state: OBSERVED only when at least one covered resource yielded a usable
            nominal quota *and* a reservation. Anything else is UNKNOWN.
        reason: Why nothing was established, when ``state`` is UNKNOWN.
        headroom: Free capacity per ``(flavor, resource)``. A resource missing
            here was not established and must not be read as zero.
        cohort: The queue's cohort, when it is in one. Presence is what makes
            borrowing possible and headroom inexact.
        pending_workloads: ``status.pendingWorkloads`` -- how many Workloads the
            queue is already holding back. Explanation only.
        admitted_workloads: ``status.admittedWorkloads``. Explanation only.
        active: Whether Kueue reports the queue Active. ``None`` when the
            condition is absent, which is unknown rather than inactive.
    """

    cluster_queue: str
    state: ClusterCapacityState
    reason: str = ""
    headroom: tuple[FlavorHeadroom, ...] = ()
    cohort: str = ""
    pending_workloads: int = 0
    admitted_workloads: int = 0
    active: bool | None = None

    def free(self, resource: str, *, flavor: str = "") -> tuple[float, bool] | None:
        """Return ``(free, exact)`` for one resource, or ``None`` if unknown.

        The value is the *best single flavor's* free capacity, not the sum
        across flavors. One resource in one Workload is admitted against one
        ResourceFlavor, so two flavors with two free accelerators each cannot
        between them host a cell that needs four.

        Args:
            resource: Kubernetes resource name.
            flavor: Restrict to one ResourceFlavor. Empty considers every flavor
                that covers the resource.

        Returns:
            ``None`` when this queue established nothing about ``resource``.
            Otherwise the free amount and whether that amount is exact.
        """

        if self.state is not ClusterCapacityState.OBSERVED:
            return None
        matching = [
            entry for entry in self.headroom if entry.resource == resource and (not flavor or entry.flavor == flavor)
        ]
        if not matching:
            return None
        return max(entry.free for entry in matching), all(entry.exact for entry in matching)


def read_cluster_queue_capacity(observation: ApiObservation, name: str) -> ClusterQueueCapacity:
    """Read one ClusterQueue's free capacity, or establish that it is unknown.

    Args:
        observation: The result of :meth:`KubernetesApi.observe_cluster_queue`.
        name: ClusterQueue name, kept so an UNKNOWN reading still says which
            queue it could not read.

    Returns:
        A :class:`ClusterQueueCapacity`. UNKNOWN when the queue was unreadable,
        absent, malformed, or -- the case worth naming -- when its ``status``
        carries no ``flavorsReservation``, since treating that as "nothing is
        reserved" would report a full cluster as entirely free.
    """

    if observation.unreadable:
        return _unknown_queue(name, f"the ClusterQueue could not be read: {observation.reason}")
    payload = observation.payload
    if payload is None:
        return _unknown_queue(name, "the ClusterQueue does not exist")
    spec = payload.get("spec") if is_json_object(payload.get("spec")) else {}
    status = payload.get("status") if is_json_object(payload.get("status")) else {}
    assert isinstance(spec, dict) and isinstance(status, dict)
    reservation = _flavor_resource_totals(status.get("flavorsReservation"))
    if reservation is None:
        return _unknown_queue(name, "the ClusterQueue reports no flavorsReservation yet")
    cohort = str(spec.get("cohort", "") or "")
    headroom: list[FlavorHeadroom] = []
    for index, group in enumerate(_json_objects(spec.get("resourceGroups"))):
        for flavor in _json_objects(group.get("flavors")):
            flavor_name = str(flavor.get("name", "") or "")
            if not flavor_name:
                continue
            for resource in _json_objects(flavor.get("resources")):
                entry = _flavor_headroom(flavor_name, resource, reservation, cohort=cohort, group=index)
                if entry is not None:
                    headroom.append(entry)
    if not headroom:
        return _unknown_queue(name, "the ClusterQueue declares no readable resource quota")
    return ClusterQueueCapacity(
        cluster_queue=name,
        state=ClusterCapacityState.OBSERVED,
        headroom=tuple(headroom),
        cohort=cohort,
        pending_workloads=_non_negative_int(status.get("pendingWorkloads")),
        admitted_workloads=_non_negative_int(status.get("admittedWorkloads")),
        active=_active_condition(status.get("conditions")),
    )


def resource_flavor_node_labels(observation: ApiObservation) -> dict[str, str] | None:
    """Return one ResourceFlavor's ``spec.nodeLabels``, or ``None`` if unknown.

    ``None`` and ``{}`` are different answers. ``{}`` is a flavor that genuinely
    selects every node; ``None`` is a flavor Eyvan could not read, and selecting
    every node on that basis would attribute the whole cluster's allocatable to
    a flavor that may back none of it.
    """

    if observation.unreadable or observation.payload is None:
        return None
    spec = observation.payload.get("spec")
    if not is_json_object(spec):
        return None
    labels = spec.get("nodeLabels")
    if labels is None:
        return {}
    if not is_json_object(labels):
        return None
    return {key: str(value) for key, value in labels.items() if isinstance(value, str)}


def node_label_selector(labels: tp.Mapping[str, str]) -> str:
    """Render node labels as a Kubernetes ``labelSelector`` query value."""

    return ",".join(f"{key}={value}" for key, value in sorted(labels.items()))


def read_node_allocatable(observation: ApiObservation) -> NodeAllocatable:
    """Sum allocatable capacity over the schedulable, Ready nodes in a list.

    Args:
        observation: The result of :meth:`KubernetesApi.observe_nodes`.

    Returns:
        A :class:`NodeAllocatable`. UNKNOWN when the list could not be read.
        OBSERVED but incomplete -- ``complete=False`` -- when the list was
        truncated or a node's ``Ready`` condition is itself ``Unknown``, because
        the total is then a lower bound and a lower bound cannot prove a
        shortage.
    """

    if observation.unreadable:
        return NodeAllocatable(
            state=ClusterCapacityState.UNKNOWN,
            reason=f"the node list could not be read: {observation.reason}",
        )
    payload = observation.payload
    if payload is None:
        return NodeAllocatable(state=ClusterCapacityState.UNKNOWN, reason="the node list does not exist")
    items = payload.get("items")
    if not isinstance(items, list):
        return NodeAllocatable(state=ClusterCapacityState.UNKNOWN, reason="the node list carries no items")
    metadata = payload.get("metadata") if is_json_object(payload.get("metadata")) else {}
    assert isinstance(metadata, dict)
    complete = not str(metadata.get("continue", "") or "")
    totals: dict[str, float] = {}
    counted = 0
    for item in items:
        if not is_json_object(item):
            complete = False
            continue
        readiness = _node_ready(item)
        if readiness is None:
            # The kubelet stopped reporting. Neither usable nor provably absent,
            # so it lowers confidence in the total instead of changing it.
            complete = False
            continue
        if not readiness:
            continue
        allocatable = _node_allocatable(item)
        if allocatable is None:
            complete = False
            continue
        counted += 1
        for resource, amount in allocatable.items():
            totals[resource] = totals.get(resource, 0.0) + amount
    return NodeAllocatable(
        state=ClusterCapacityState.OBSERVED,
        allocatable=totals,
        node_count=counted,
        complete=complete,
    )


@dataclass(frozen=True, slots=True)
class ClusterCapacityReading:
    """One timestamped reading of the capacity behind an Eyvan Kubernetes pool.

    Attributes:
        queue: The ClusterQueue's free capacity, itself possibly UNKNOWN.
        nodes: Allocatable behind the flavor, itself possibly UNKNOWN. A node
            list Eyvan cannot read never degrades a queue reading it could.
        flavor: ResourceFlavor the node reading was scoped to, when one was
            resolved.
        observed_at: When the reading was taken. Required, because a reading
            with no time cannot be told from a stale one.
    """

    queue: ClusterQueueCapacity
    observed_at: datetime
    nodes: NodeAllocatable = field(default_factory=lambda: NodeAllocatable(state=ClusterCapacityState.UNKNOWN))
    flavor: str = ""

    def __post_init__(self) -> None:
        """Normalize the observation time and refuse a naive one."""

        if not isinstance(self.observed_at, datetime) or self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be a timezone-aware datetime")
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(UTC))

    @property
    def state(self) -> ClusterCapacityState:
        """Return the queue's state, which is the only part that plans."""

        return self.queue.state


def observe_cluster_capacity(
    api: KubernetesApi,
    cluster_queue: str,
    *,
    flavor: str = "",
    now: datetime | None = None,
) -> ClusterCapacityReading:
    """Read the free capacity behind one ClusterQueue without ever raising.

    Every read here is a GET against objects Eyvan does not own, so each one can
    fail on its own and each failure degrades only what it was going to
    establish: an unreadable ResourceFlavor leaves the queue reading intact, and
    an unreadable ClusterQueue leaves the whole thing UNKNOWN.

    This makes up to three HTTP calls and must therefore be taken *outside* any
    database transaction or row lock: it describes a cluster, so a caller that
    holds a lease while waiting on someone else's API server has turned a slow
    read into a stuck cell.

    Args:
        api: In-cluster API access.
        cluster_queue: ClusterQueue name from the pool's ``provider_options``.
        flavor: Optional ResourceFlavor to scope node allocatable to. Empty
            skips the node reading rather than summing the whole cluster.
        now: Observation time; defaults to the current UTC time.

    Returns:
        A :class:`ClusterCapacityReading`, never an exception.
    """

    observed_at = now or datetime.now(UTC)
    queue = read_cluster_queue_capacity(api.observe_cluster_queue(cluster_queue), cluster_queue)
    nodes = NodeAllocatable(state=ClusterCapacityState.UNKNOWN, reason="no ResourceFlavor was named")
    if flavor:
        labels = resource_flavor_node_labels(api.observe_resource_flavor(flavor))
        if labels is None:
            nodes = NodeAllocatable(
                state=ClusterCapacityState.UNKNOWN,
                reason=f"ResourceFlavor {flavor!r} could not be read",
            )
        else:
            nodes = read_node_allocatable(api.observe_nodes(label_selector=node_label_selector(labels)))
    return ClusterCapacityReading(queue=queue, nodes=nodes, flavor=flavor, observed_at=observed_at)


# --- cluster capacity: what it lets Eyvan conclude ---------------------------


@dataclass(frozen=True, slots=True)
class CapacityFit:
    """How many cells of one shape the observed cluster is proven to admit.

    Attributes:
        state: UNKNOWN whenever ``admits`` is ``None``.
        admits: A *lower bound* on the number of requested cells the cluster
            will admit, or ``None`` when nothing was established. Never a guess:
            ``0`` means proven-to-fit-none only when ``exact``.
        exact: Whether ``admits`` is also an upper bound. Only an exact fit can
            support the conclusion "the cluster is full"; an inexact one is
            evidence of room, never of its absence.
        limiting_resource: The resource that produced ``admits``.
        reason: Why nothing was established, when ``state`` is UNKNOWN.
    """

    state: ClusterCapacityState
    admits: int | None = None
    exact: bool = False
    limiting_resource: str = ""
    reason: str = ""

    @property
    def blocked(self) -> bool:
        """Return whether the cluster is *proven* to admit nothing right now."""

        return self.state is ClusterCapacityState.OBSERVED and self.admits == 0 and self.exact


def capacity_fit(
    capacity: ClusterQueueCapacity,
    request: tp.Mapping[str, JsonValue],
    *,
    flavor: str = "",
) -> CapacityFit:
    """Return how many cells of one resource shape the queue is proven to admit.

    Kueue assigns one ResourceFlavor per *resource group*, so resources sharing
    a group must come out of the same flavor while resources in different groups
    need not. That structure is preserved rather than flattened: flattening
    would let two flavors with two free accelerators each look like room for a
    four-accelerator cell that fits in neither.

    ``admits`` is always a lower bound on the truth, and ``exact`` says whether
    it is also an upper bound. Only an exact zero means "full". The fit uses the
    best single flavor per group. If multiple flavors can each host a cell,
    different cells may use different flavors, so that fit is not an exact
    multi-cell ceiling. Keep it inexact rather than pooling resources across
    flavors or claiming a batch allocation that this observation does not make.

    Args:
        capacity: An observed or unknown ClusterQueue reading.
        request: One cell's resource requests, exactly as they appear in a Job
            container -- ``{"google.com/tpu": "4", "cpu": "8"}``.
        flavor: ResourceFlavor to prefer. Applied per resource group and only
            where that group offers it -- a flavor that covers accelerators says
            nothing about the group that covers ``cpu``, and treating it as a
            hard filter there would report an unknown fit for a queue that was
            read perfectly well.

    Returns:
        A :class:`CapacityFit`. UNKNOWN when the queue is unknown, when the
        request is empty or unparseable, or when the queue established nothing
        about a requested resource -- an absent resource is not a free one.
    """

    if capacity.state is not ClusterCapacityState.OBSERVED:
        return CapacityFit(state=ClusterCapacityState.UNKNOWN, reason=capacity.reason)
    wanted: dict[str, float] = {}
    for resource, raw in request.items():
        amount = parse_quantity(raw)
        if amount is None:
            return CapacityFit(
                state=ClusterCapacityState.UNKNOWN,
                reason=f"requested quantity for {resource!r} is unreadable",
            )
        if amount > 0:
            wanted[resource] = amount
    if not wanted:
        return CapacityFit(state=ClusterCapacityState.UNKNOWN, reason="the request names no positive resource")
    groups: dict[int, set[str]] = {}
    flavors_by_group: dict[int, set[str]] = {}
    for entry in capacity.headroom:
        flavors_by_group.setdefault(entry.group, set()).add(entry.flavor)
        if entry.resource in wanted:
            groups.setdefault(entry.group, set()).add(entry.resource)
    covered = {resource for resources in groups.values() for resource in resources}
    missing = sorted(set(wanted) - covered)
    if missing:
        return CapacityFit(
            state=ClusterCapacityState.UNKNOWN,
            reason=f"the ClusterQueue declares no quota for {missing[0]!r}",
        )
    admits: int | None = None
    limiting = ""
    exact = True
    for group, resources in sorted(groups.items()):
        candidates = [flavor] if flavor and flavor in flavors_by_group[group] else sorted(flavors_by_group[group])
        best = -1
        best_resource = ""
        fitting_flavors = 0
        for candidate in candidates:
            count, count_resource, count_exact = _flavor_group_fit(capacity, wanted, resources, group, candidate)
            if count is None:
                continue
            exact = exact and count_exact
            if count > 0:
                fitting_flavors += 1
            if count > best:
                best = count
                best_resource = count_resource
        # Separate cells may fit in separate flavors. The best single-flavor
        # count is then only a lower bound, never a batch capacity ceiling.
        exact = exact and fitting_flavors <= 1
        if best < 0:
            return CapacityFit(
                state=ClusterCapacityState.UNKNOWN,
                reason=f"no ResourceFlavor covers every requested resource in group {group}",
            )
        if admits is None or best < admits:
            admits = best
            limiting = best_resource
    if admits is None:
        return CapacityFit(state=ClusterCapacityState.UNKNOWN, reason="the request names no positive resource")
    return CapacityFit(
        state=ClusterCapacityState.OBSERVED,
        admits=admits,
        exact=exact,
        limiting_resource=limiting,
    )


def _flavor_group_fit(
    capacity: ClusterQueueCapacity,
    wanted: tp.Mapping[str, float],
    resources: tp.Collection[str],
    group: int,
    flavor: str,
) -> tuple[int | None, str, bool]:
    """Return how many cells one flavor can host for one resource group.

    Returns:
        ``(count, limiting_resource, exact)``, or ``(None, "", True)`` when this
        flavor does not declare every resource the group must satisfy -- which
        makes it unusable for this request rather than a flavor with no room.
    """

    count: int | None = None
    limiting = ""
    exact = True
    for resource in sorted(resources):
        entry = next(
            (
                item
                for item in capacity.headroom
                if item.group == group and item.flavor == flavor and item.resource == resource
            ),
            None,
        )
        if entry is None:
            return None, "", True
        exact = exact and entry.exact
        # A hair of tolerance so a quantity such as 1500m divided by 250m does
        # not floor to one less than it should; understating free capacity is
        # the direction that reads a cluster as fuller than it was observed to
        # be, which is the inference this module exists to prevent.
        fits = math.floor(entry.free / wanted[resource] + 1e-9)
        if count is None or fits < count:
            count = fits
            limiting = resource
    return count, limiting, exact


@dataclass(frozen=True, slots=True)
class ClusterCapacityExplanation:
    """An operator-facing account of why a cell is or is not admissible.

    Attributes:
        state: Whether anything was established at all.
        blocked: Whether the cluster is *proven* to admit no such cell.
        summary: One line naming what was observed, with numbers.
        remedy: What an operator can do about it. Empty when nothing is wrong.
        fit: The underlying fit, so a caller can re-derive the decision.
    """

    state: ClusterCapacityState
    blocked: bool
    summary: str
    remedy: str = ""
    fit: CapacityFit = field(default_factory=lambda: CapacityFit(state=ClusterCapacityState.UNKNOWN))

    @property
    def detail(self) -> str:
        """Return the single string a cell observation or explain can carry."""

        return f"{self.summary}; {self.remedy}" if self.remedy else self.summary


def explain_cluster_capacity(
    reading: ClusterCapacityReading,
    request: tp.Mapping[str, JsonValue],
) -> ClusterCapacityExplanation:
    """Explain, in the operator's terms, what the cluster has room for.

    This is the whole point of W3 for a human: a pool that sits WAITING because
    the cluster is full of non-Eyvan work currently says "job is suspended
    pending kueue admission", which describes Eyvan's own state and not the
    cause. The explanation below names the exhausted resource, the numbers
    behind it, how many Workloads are already queued, and what to change.

    Args:
        reading: A timestamped cluster capacity reading.
        request: One cell's resource requests.

    Returns:
        A :class:`ClusterCapacityExplanation`. UNKNOWN says so plainly and
        proposes the read that would settle it, because a missing signal that
        presents as a settled one is the failure this package exists to prevent.
    """

    queue = reading.queue
    fit = capacity_fit(queue, request, flavor=reading.flavor)
    if fit.state is not ClusterCapacityState.OBSERVED:
        return ClusterCapacityExplanation(
            state=ClusterCapacityState.UNKNOWN,
            blocked=False,
            summary=(
                f"cluster free capacity for ClusterQueue {queue.cluster_queue!r} is unknown: "
                f"{fit.reason or queue.reason or 'nothing was established'}"
            ),
            remedy=(
                "grant the Eyvan ServiceAccount 'get' on clusterqueues, resourceflavors and nodes, "
                "or leave cluster capacity out of planning; it is not being used as one"
            ),
            fit=fit,
        )
    observed = _observed_summary(queue, request, fit, flavor=reading.flavor)
    if fit.blocked:
        return ClusterCapacityExplanation(
            state=ClusterCapacityState.OBSERVED,
            blocked=True,
            summary=observed,
            remedy=_blocked_remedy(reading, fit),
            fit=fit,
        )
    return ClusterCapacityExplanation(
        state=ClusterCapacityState.OBSERVED,
        blocked=False,
        summary=observed,
        remedy=_unbacked_quota_remedy(reading, fit),
        fit=fit,
    )


def cell_capacity_state(
    status: JobStatus,
    explanation: ClusterCapacityExplanation | None = None,
) -> tuple[CapacityStatus, str]:
    """Return a cell's capacity state, with an actionable detail when queued.

    ``capacity_state`` alone answers "what is this Job doing"; a queued Job
    answers that with "job is suspended pending kueue admission", which is
    Eyvan's own state restated. When the cluster's free capacity *was* observed
    and proves there is no room, the WAITING detail says so instead.

    Args:
        status: The interpreted Job.
        explanation: An observed cluster capacity explanation, or ``None``. The
            default reproduces today's behaviour exactly, so a caller that has
            no capacity signal -- or has one that is UNKNOWN -- is unchanged.

    Returns:
        The Eyvan capacity state and its detail.
    """

    state, detail = capacity_state(status)
    if explanation is None or state is not CapacityStatus.WAITING:
        return state, detail
    if explanation.state is not ClusterCapacityState.OBSERVED or not explanation.blocked:
        return state, detail
    return state, f"{detail}; {explanation.detail}"


# --- cluster capacity: parsing helpers ---------------------------------------


def _unknown_queue(name: str, reason: str) -> ClusterQueueCapacity:
    """Return an UNKNOWN queue reading that says why it established nothing."""

    return ClusterQueueCapacity(cluster_queue=name, state=ClusterCapacityState.UNKNOWN, reason=reason)


def _json_objects(value: JsonValue) -> tuple[JsonObject, ...]:
    """Return the JSON objects in a list, dropping anything else."""

    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if is_json_object(item))


def _flavor_resource_totals(value: JsonValue) -> dict[tuple[str, str], float] | None:
    """Return ``(flavor, resource) -> total`` from a Kueue flavor usage list.

    ``None`` when the list is absent or unusable. That distinction is the single
    most important one in this module: an absent ``flavorsReservation`` means
    the queue has not reported yet, and reading it as "nothing is reserved"
    would present a saturated cluster as an empty one.
    """

    if not isinstance(value, list):
        return None
    totals: dict[tuple[str, str], float] = {}
    for flavor in _json_objects(value):
        flavor_name = str(flavor.get("name", "") or "")
        if not flavor_name:
            return None
        for resource in _json_objects(flavor.get("resources")):
            resource_name = str(resource.get("name", "") or "")
            amount = parse_quantity(resource.get("total"))
            if not resource_name or amount is None:
                return None
            totals[(flavor_name, resource_name)] = amount
    return totals


def _flavor_headroom(
    flavor: str,
    resource: JsonObject,
    reservation: tp.Mapping[tuple[str, str], float],
    *,
    cohort: str,
    group: int,
) -> FlavorHeadroom | None:
    """Return one resource's headroom, or ``None`` when it is not established."""

    name = str(resource.get("name", "") or "")
    nominal = parse_quantity(resource.get("nominalQuota"))
    if not name or nominal is None:
        return None
    reserved = reservation.get((flavor, name))
    if reserved is None:
        return None
    borrowing = parse_quantity(resource.get("borrowingLimit"))
    # Borrowing is only *possible* inside a cohort, and only unbounded when the
    # limit is absent or positive. A cohort member with an explicit zero limit
    # is as exact as a queue with no cohort at all.
    exact = not cohort or borrowing == 0.0
    return FlavorHeadroom(
        flavor=flavor,
        resource=name,
        nominal_quota=nominal,
        reserved=reserved,
        exact=exact,
        group=group,
    )


def _non_negative_int(value: JsonValue) -> int:
    """Return a non-negative integer count, or zero for anything else."""

    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _active_condition(conditions: JsonValue) -> bool | None:
    """Return whether Kueue reports the queue Active, or ``None`` if unstated."""

    for condition in _json_objects(conditions):
        if str(condition.get("type", "")) != "Active":
            continue
        status = str(condition.get("status", ""))
        if status == "True":
            return True
        if status == "False":
            return False
        return None
    return None


def _node_ready(node: JsonObject) -> bool | None:
    """Return whether one node can hold pods, or ``None`` when unknowable.

    ``None`` is the kubelet's own ``Ready=Unknown``: the node stopped reporting,
    so it is neither usable capacity nor provably gone.
    """

    spec = node.get("spec") if is_json_object(node.get("spec")) else {}
    status = node.get("status") if is_json_object(node.get("status")) else {}
    assert isinstance(spec, dict) and isinstance(status, dict)
    if spec.get("unschedulable") is True:
        return False
    for condition in _json_objects(status.get("conditions")):
        if str(condition.get("type", "")) != "Ready":
            continue
        value = str(condition.get("status", ""))
        if value == "True":
            return True
        if value == "False":
            return False
        return None
    return None


def _node_allocatable(node: JsonObject) -> dict[str, float] | None:
    """Return one node's allocatable resources, or ``None`` when unreadable."""

    status = node.get("status") if is_json_object(node.get("status")) else {}
    assert isinstance(status, dict)
    allocatable = status.get("allocatable")
    if not is_json_object(allocatable):
        return None
    totals: dict[str, float] = {}
    for resource, raw in allocatable.items():
        amount = parse_quantity(raw)
        if amount is None:
            return None
        totals[resource] = amount
    return totals


# --- cluster capacity: explanation helpers -----------------------------------


def _format_amount(value: float) -> str:
    """Render a quantity without a trailing ``.0`` on whole accelerator counts."""

    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _observed_summary(
    capacity: ClusterQueueCapacity,
    request: tp.Mapping[str, JsonValue],
    fit: CapacityFit,
    *,
    flavor: str = "",
) -> str:
    """Return the observed-capacity sentence, with the numbers behind it."""

    parts: list[str] = []
    for resource in sorted(request)[:_MAX_EXPLAINED_RESOURCES]:
        available = capacity.free(resource, flavor=flavor)
        if available is None:
            continue
        free, exact = available
        bound = "" if exact else " or more"
        parts.append(f"{resource} free {_format_amount(free)}{bound}")
    queued = f", {capacity.pending_workloads} workload(s) already pending" if capacity.pending_workloads else ""
    verdict = (
        "admits no further cell of this shape"
        if fit.blocked
        else f"admits {fit.admits}{'' if fit.exact else ' or more'} cell(s) of this shape"
    )
    return (
        f"ClusterQueue {capacity.cluster_queue!r} {verdict}: "
        f"{'; '.join(parts) if parts else 'no matching resource quota'}{queued}"
    )


def _blocked_remedy(reading: ClusterCapacityReading, fit: CapacityFit) -> str:
    """Return what an operator can change when the cluster is proven full."""

    queue = reading.queue
    resource = fit.limiting_resource or "the requested resource"
    remedies = [
        f"raise nominalQuota for {resource} on ClusterQueue {queue.cluster_queue!r}",
        f"add nodes backing {resource}",
    ]
    if queue.admitted_workloads:
        remedies.append(f"or wait for {queue.admitted_workloads} admitted workload(s) to finish")
    if queue.active is False:
        remedies.append("and note Kueue reports this ClusterQueue as not Active")
    return ", ".join(remedies) + "."


def _unbacked_quota_remedy(reading: ClusterCapacityReading, fit: CapacityFit) -> str:
    """Return a warning when nominal quota exceeds what the nodes can honour.

    Only stated when the node reading is both OBSERVED and complete. An
    incomplete node list is a lower bound, and a lower bound can always be made
    to look like a shortage -- which is exactly the inference this module
    refuses to draw.
    """

    nodes = reading.nodes
    if nodes.state is not ClusterCapacityState.OBSERVED or not nodes.complete:
        return ""
    resource = fit.limiting_resource
    if not resource:
        return ""
    quota = sum(
        entry.nominal_quota
        for entry in reading.queue.headroom
        if entry.resource == resource and (not reading.flavor or entry.flavor == reading.flavor)
    )
    # A complete node reading that never mentions the resource is a genuine
    # observation of zero -- the flavor's node labels match nothing that carries
    # it. This only ever reaches an explanation string, never a plan input.
    allocatable = nodes.allocatable.get(resource, 0.0)
    if quota <= allocatable:
        return ""
    return (
        f"note ClusterQueue nominalQuota for {resource} is {_format_amount(quota)} but the "
        f"{nodes.node_count} node(s) backing ResourceFlavor {reading.flavor!r} allocate only "
        f"{_format_amount(allocatable)}; the quota is not fully backed by hardware."
    )


__all__ = (
    "SAFE_PREEMPTION",
    "CapacityFit",
    "ClusterCapacityExplanation",
    "ClusterCapacityReading",
    "ClusterCapacityState",
    "ClusterQueueCapacity",
    "FlavorHeadroom",
    "KueueFinding",
    "NodeAllocatable",
    "capacity_fit",
    "cell_capacity_state",
    "explain_cluster_capacity",
    "foreign_workloads",
    "inspect_cluster_queue",
    "node_label_selector",
    "observe_cluster_capacity",
    "parse_quantity",
    "read_cluster_queue_capacity",
    "read_node_allocatable",
    "resource_flavor_node_labels",
)
