# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One cluster reading per Kubernetes pool, taken before anything launches.

:mod:`eyvan.capacity.autoscale.cluster_capacity` decides what an *observed*
cluster allows. It cannot take the observation: it is pure, and the reading is
three GETs against an API server Eyvan does not own. This module is the half
that goes and looks, and it exists to keep three properties that are easy to
lose once a network call enters a planning pass.

**Default-off means no request at all.** With
:attr:`~eyvan.capacity.autoscale.cluster_capacity.ClusterCapacityPolicy.enabled`
false -- the default -- nothing here opens a socket. Not "reads and ignores":
a control plane that upgrades into this code issues exactly the traffic it
issued before, so the switch has no cost to leave off and no surprise to turn
on.

**A reading is taken outside every transaction and row lock.** It describes
someone else's cluster, and holding a lease while waiting on someone else's API
server turns a slow read into a stuck cell. The pass is a plain synchronous
function precisely so a caller must decide where to put it; the shipped caller
runs it on a worker thread between transactions.

**Nothing here can fail a reconciliation.** Every failure is UNKNOWN, and
UNKNOWN returns the caller's own number. A pool whose ServiceAccount cannot read
Kueue scales exactly as it did before this module existed.

**Why the LocalQueue is resolved first.** A pool's ``provider_options.queue_name``
is written to the Job as ``kueue.x-k8s.io/queue-name``, which names a
*LocalQueue*. The quota lives on the *ClusterQueue* that the LocalQueue routes
into, and the two are different objects with different names. Reading capacity
against the pool's own ``queue_name`` would describe whatever ClusterQueue
happened to share that name -- or nothing at all -- and answer with numbers
either way.
"""

from __future__ import annotations

import logging
import os
import typing as tp
from dataclasses import dataclass, replace
from datetime import datetime

from eyvan._internal.compat import UTC
from eyvan.capacity.autoscale.cluster_capacity import (
    DEFAULT_MAX_OBSERVATION_AGE_S,
    ClusterCapacityAdvice,
    ClusterCapacityPolicy,
    advise_launch,
)
from eyvan.capacity.providers.kubernetes.api import KubernetesApi, local_queue_cluster_queue
from eyvan.capacity.providers.kubernetes.kueue import (
    ClusterCapacityReading,
    ClusterCapacityState,
    ClusterQueueCapacity,
    capacity_fit,
    observe_cluster_capacity,
    parse_quantity,
)
from eyvan.records import CapacityPoolRecord

logger = logging.getLogger(__name__)

#: Whether an observed cluster may reduce a launch count. Off by default, which
#: is every Eyvan release before this one: capacity is requested from the cluster
#: and the cluster decides.
CLUSTER_CAPACITY_PLANNING_ENV = "EYVAN_CLUSTER_CAPACITY_PLANNING"

#: Oldest reading still treated as evidence, in seconds. A malformed value is
#: ignored rather than raised: this is read inside a reconciliation pass, and a
#: typo in an environment variable must not stop a fleet from scaling.
CLUSTER_CAPACITY_MAX_AGE_ENV = "EYVAN_CLUSTER_CAPACITY_MAX_AGE_S"

#: Namespace the cell agent creates Jobs in, which is where their LocalQueues
#: live. The same variable the agent reads, so one setting configures both; a
#: pool may override it per-pool through ``provider_options.namespace``.
KUBERNETES_NAMESPACE_ENV = "EYVAN_KUBERNETES_NAMESPACE"

#: Fallback namespace, matching ``KubernetesCapacityDriver``'s own default. A
#: wrong namespace produces a 404, which is a readable "the LocalQueue is not
#: there" and therefore UNKNOWN capacity -- never an empty or a full cluster.
DEFAULT_NAMESPACE = "eyvan"

#: Backend names the Kubernetes driver is registered under.
KUBERNETES_BACKENDS = frozenset({"kubernetes", "k8s"})

_TRUE = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True, slots=True)
class PoolCapacityTarget:
    """What one Kubernetes pool needs read before it launches.

    Attributes:
        pool_id: The pool this describes.
        namespace: Namespace holding the LocalQueue.
        local_queue: LocalQueue name, from the pool's ``queue_name``.
        request: One *pod's* resource request in base units, as
            ``(resource, amount)`` pairs. Parsed once here so an unreadable
            quantity is caught before any GET is spent on it.
        flavor: ResourceFlavor to scope the node reading to. Empty skips it.
    """

    pool_id: str
    namespace: str
    local_queue: str
    request: tuple[tuple[str, float], ...]
    flavor: str = ""

    def cell_request(self, worker_count: int) -> dict[str, float]:
        """Return one whole cell's request: per-pod amounts times its pods.

        A cell is an Indexed Job with ``parallelism == worker_count``, so the
        capacity it takes from a ClusterQueue is the pod request multiplied by
        the worker count. Passing the per-pod request instead would count *pods*
        as cells and let a pool launch several times what fits.

        Args:
            worker_count: Pods in the cell. Values below one are treated as one.

        Returns:
            The cell's request in base units, empty when nothing was readable.
        """

        pods = max(int(worker_count), 1)
        return {resource: amount * pods for resource, amount in self.request}


def kubernetes_capacity_target(
    pool: CapacityPoolRecord,
    *,
    default_namespace: str = DEFAULT_NAMESPACE,
) -> PoolCapacityTarget | None:
    """Return what to read for one pool, or ``None`` when there is nothing to read.

    ``None`` for four distinct reasons, all of which mean the same thing to the
    caller -- take no reading and change no plan:

    * the pool is not Kubernetes-backed;
    * it names no Kueue queue, so Kueue is not admitting it and the cluster's
      ClusterQueue quota is not what it waits on;
    * it declares no resource request, so there is no shape to fit;
    * a declared quantity is unreadable, which is unknown capacity rather than a
      request for nothing -- and is settled here, before a GET is spent on it.

    Args:
        pool: The declarative pool.
        default_namespace: Namespace to assume when the pool does not name one.

    Returns:
        A :class:`PoolCapacityTarget`, or ``None``.
    """

    backend = (pool.capacity_backend or pool.provider or "").strip().lower()
    if backend not in KUBERNETES_BACKENDS:
        return None
    options = pool.provider_options if isinstance(pool.provider_options, dict) else {}
    local_queue = _text(options.get("queue_name"))
    if not local_queue:
        return None
    declared = options.get("resources")
    if not isinstance(declared, dict) or not declared:
        return None
    request: list[tuple[str, float]] = []
    for resource, raw in sorted(declared.items()):
        amount = parse_quantity(raw)
        if amount is None:
            return None
        if amount > 0:
            request.append((str(resource), amount))
    if not request:
        return None
    return PoolCapacityTarget(
        pool_id=pool.id,
        namespace=_text(options.get("namespace")) or default_namespace,
        local_queue=local_queue,
        request=tuple(request),
        flavor=_text(options.get("resource_flavor")),
    )


def cluster_capacity_policy_from_env(
    environ: tp.Mapping[str, str] | None = None,
) -> ClusterCapacityPolicy:
    """Return the operator's cluster-capacity policy, defaulting to disabled.

    Args:
        environ: Environment to read. Defaults to the process environment.

    Returns:
        A :class:`ClusterCapacityPolicy`. An unset, empty, or unrecognized
        enablement value is off, and an out-of-range or non-numeric age falls
        back to the shipped default rather than raising inside a scaling pass.
    """

    source = os.environ if environ is None else environ
    enabled = source.get(CLUSTER_CAPACITY_PLANNING_ENV, "").strip().lower() in _TRUE
    raw = source.get(CLUSTER_CAPACITY_MAX_AGE_ENV, "").strip()
    try:
        max_age_s = float(raw) if raw else DEFAULT_MAX_OBSERVATION_AGE_S
        return ClusterCapacityPolicy(enabled=enabled, max_observation_age_s=max_age_s)
    except ValueError:
        logger.warning(
            "%s=%r is not a usable age; using %ss",
            CLUSTER_CAPACITY_MAX_AGE_ENV,
            raw,
            DEFAULT_MAX_OBSERVATION_AGE_S,
        )
        return ClusterCapacityPolicy(enabled=enabled, max_observation_age_s=DEFAULT_MAX_OBSERVATION_AGE_S)


class ClusterCapacityPass:
    """Read every Kubernetes pool's cluster once, then advise its launches.

    Holds no durable state and remembers nothing between passes: a reading is
    evidence about a cluster *now*, and a cache that outlived one pass would be
    a cache of someone else's admissions.
    """

    def __init__(
        self,
        *,
        policy: ClusterCapacityPolicy | None = None,
        api_factory: tp.Callable[[], KubernetesApi | None] | None = None,
        default_namespace: str = DEFAULT_NAMESPACE,
    ):
        self.policy = policy or ClusterCapacityPolicy()
        self._api_factory = api_factory or _default_api
        self.default_namespace = default_namespace

    @classmethod
    def from_env(cls, environ: tp.Mapping[str, str] | None = None) -> ClusterCapacityPass:
        """Build the pass an operator configured, which by default does nothing."""

        source = os.environ if environ is None else environ
        return cls(
            policy=cluster_capacity_policy_from_env(source),
            default_namespace=source.get(KUBERNETES_NAMESPACE_ENV, "").strip() or DEFAULT_NAMESPACE,
        )

    @property
    def enabled(self) -> bool:
        """Return whether an observed cluster may reduce a launch count."""

        return self.policy.enabled

    def advise_launches(
        self,
        planned: tp.Sequence[tuple[CapacityPoolRecord, tp.Sequence[int]]],
        *,
        now: datetime | None = None,
    ) -> dict[str, ClusterCapacityAdvice]:
        """Return one advice per pool that plans a launch, reading each cluster once.

        Synchronous, and it makes network calls. Callers must run it outside
        every transaction and row lock: it describes a cluster someone else owns.

        One moment stamps both the reading and the plan it feeds. They are
        milliseconds apart in a real pass, and giving the reading a clock of its
        own would leave the freshness bound comparing two different clocks --
        which is how a signal survives its own expiry.

        Exact accepted batches reserve their resource vectors in pool-id order
        against the remaining ClusterQueue headroom. Reservations span LocalQueues
        and preferred flavors where their resource groups share quota, and expire
        with this pass. UNKNOWN advice still leaves the requested count untouched.

        Args:
            planned: ``(pool, worker_counts)`` for each pool about to launch,
                where ``worker_counts`` is one entry per planned cell.
            now: The pass's clock. Defaults to the current UTC time.

        Returns:
            Advice keyed by pool id, holding only pools a reading was taken for.
            An empty mapping -- which is what a disabled policy always returns --
            means every caller keeps its own number.
        """

        if not self.enabled:
            return {}
        targets = [
            (tuple(worker_counts), target)
            for pool, worker_counts in planned
            if worker_counts
            for target in (kubernetes_capacity_target(pool, default_namespace=self.default_namespace),)
            if target is not None
        ]
        if not targets:
            return {}
        moment = now or datetime.now(UTC)
        readings = self.read([target for _counts, target in targets], now=moment)
        # Match the controller's stable launch order. Quota belongs to the
        # ClusterQueue, not the LocalQueue or the pool's preferred flavor: a
        # CPU resource group can be shared by otherwise distinct TPU flavors.
        reserved: dict[str, dict[tuple[str, str], float]] = {}
        advice: dict[str, ClusterCapacityAdvice] = {}
        for counts, target in sorted(targets, key=lambda item: item[1].pool_id):
            reading = readings.get(target.pool_id)
            balance = None
            if reading is not None:
                balance = reserved.setdefault(reading.queue.cluster_queue, {})
                reading = replace(
                    reading,
                    queue=replace(
                        reading.queue,
                        headroom=tuple(
                            replace(entry, reserved=entry.reserved + balance.get((entry.flavor, entry.resource), 0.0))
                            for entry in reading.queue.headroom
                        ),
                    ),
                )
            result = self.advise(target, counts, reading, now=moment)
            advice[target.pool_id] = result
            if result.applied and result.allowed_cells and reading is not None and balance is not None:
                _reserve_launches(
                    reading.queue, target.cell_request(counts[0]), target.flavor, result.allowed_cells, balance
                )
        return advice

    def read(
        self,
        targets: tp.Sequence[PoolCapacityTarget],
        *,
        now: datetime | None = None,
    ) -> dict[str, ClusterCapacityReading | None]:
        """Read each target's cluster once, sharing a queue two pools point at.

        Sharing also requires the same ResourceFlavor: a reading is scoped to
        one flavor, so two pools on one ClusterQueue with different flavors
        are read separately rather than answered with each other's quota.

        Args:
            targets: What to read, one per pool.
            now: Observation time stamped on every reading.

        Returns:
            A reading per pool id, ``None`` where nothing was established.
        """

        api = self._api()
        cluster_queues: dict[tuple[str, str], str | None] = {}
        by_cluster_queue: dict[tuple[str, str], ClusterCapacityReading | None] = {}
        return {target.pool_id: self._read(api, target, cluster_queues, by_cluster_queue, now=now) for target in targets}

    def advise(
        self,
        target: PoolCapacityTarget,
        worker_counts: tp.Sequence[int],
        reading: ClusterCapacityReading | None,
        *,
        now: datetime | None = None,
    ) -> ClusterCapacityAdvice:
        """Return how many of one pool's planned cells the cluster is proven to admit.

        Args:
            target: What was read for this pool.
            worker_counts: One entry per planned cell.
            reading: The cluster reading, or ``None`` when none was established.
            now: Planning time, used to age the reading.

        Returns:
            A :class:`ClusterCapacityAdvice` whose ``allowed_cells`` equals the
            number planned unless a fresh, exact, OBSERVED reading proved
            otherwise.
        """

        requested = len(worker_counts)
        shapes = set(worker_counts)
        if len(shapes) > 1:
            # One fit answers "how many cells of *this* shape", so it bounds a
            # batch only when the batch has one shape. Capping a mixed batch by
            # the largest cell's fit would refuse smaller cells that do fit,
            # which is inventing a shortage out of an unanswered question.
            return ClusterCapacityAdvice(
                requested_cells=requested,
                allowed_cells=requested,
                state=ClusterCapacityState.UNKNOWN,
                applied=False,
                reason="this pass plans cells of more than one shape, which a single fit cannot bound",
            )
        request = target.cell_request(next(iter(shapes), 1))
        return advise_launch(requested, reading, request, policy=self.policy, now=now)

    def _api(self) -> KubernetesApi | None:
        """Return API access, or ``None`` when this process has none."""

        try:
            return self._api_factory()
        except Exception:  # cluster access this process does not have is UNKNOWN
            logger.debug("no Kubernetes API access for the cluster capacity pass", exc_info=True)
            return None

    def _read(
        self,
        api: KubernetesApi | None,
        target: PoolCapacityTarget,
        cluster_queues: dict[tuple[str, str], str | None],
        by_cluster_queue: dict[tuple[str, str], ClusterCapacityReading | None],
        *,
        now: datetime | None = None,
    ) -> ClusterCapacityReading | None:
        """Read one pool's cluster, reusing a reading another pool already took.

        A reading is only shared when both the ClusterQueue and the
        ResourceFlavor match; anything else would fit this pool's request
        against another pool's flavor.

        Every failure returns ``None``, which
        :func:`~eyvan.capacity.autoscale.cluster_capacity.advise_launch` reads as
        "no reading was taken" and answers with the caller's own number.
        """

        if api is None:
            return None
        key = (target.namespace, target.local_queue)
        try:
            if key not in cluster_queues:
                cluster_queues[key] = local_queue_cluster_queue(api.observe_local_queue(*key))
            cluster_queue = cluster_queues[key]
            if not cluster_queue:
                return None
            reading_key = (cluster_queue, target.flavor)
            if reading_key not in by_cluster_queue:
                by_cluster_queue[reading_key] = observe_cluster_capacity(
                    api, cluster_queue, flavor=target.flavor, now=now
                )
            return by_cluster_queue[reading_key]
        except Exception:  # a plan input that can fail a pass is not a plan input
            logger.debug("cluster capacity could not be read for pool %s", target.pool_id, exc_info=True)
            return None


def _reserve_launches(
    queue: ClusterQueueCapacity,
    request: tp.Mapping[str, float],
    flavor: str,
    cells: int,
    reserved: dict[tuple[str, str], float],
) -> None:
    """Reserve an exact accepted batch against one whole flavor per resource group.

    Use the same best-fit flavor choice as ``capacity_fit``. Reservations are
    pass-local overlays; neither the fetched reading nor Kueue is mutated.
    Only applied advice reaches here: unknown, stale, mixed-shape and lower-bound
    decisions must keep their existing semantics rather than invent a shortage.
    """

    for group in sorted({entry.group for entry in queue.headroom}):
        entries = tuple(entry for entry in queue.headroom if entry.group == group)
        wanted = {
            resource: amount for resource, amount in request.items() if any(e.resource == resource for e in entries)
        }
        if not wanted:
            continue
        flavors = sorted({entry.flavor for entry in entries})
        candidates = [flavor] if flavor and flavor in flavors else flavors
        group_queue = replace(queue, headroom=entries)
        best = -1
        selected = ""
        for candidate in candidates:
            fit = capacity_fit(group_queue, wanted, flavor=candidate)
            if fit.admits is not None and fit.admits > best:
                best = fit.admits
                selected = candidate
        for resource, amount in wanted.items():
            key = (selected, resource)
            reserved[key] = reserved.get(key, 0.0) + amount * cells


def _default_api() -> KubernetesApi | None:
    """Return in-cluster API access, or ``None`` when there is none.

    A control plane running outside the cluster it schedules into has no
    ServiceAccount token, and every read degrades to UNKNOWN on its own. Building
    the client is still guarded, because a client that cannot be constructed is
    the same answer arrived at earlier and cheaper.
    """

    return KubernetesApi()


def _text(value: object) -> str:
    """Return a stripped string, or empty for anything that is not one."""

    return value.strip() if isinstance(value, str) else ""


__all__ = (
    "CLUSTER_CAPACITY_MAX_AGE_ENV",
    "CLUSTER_CAPACITY_PLANNING_ENV",
    "DEFAULT_NAMESPACE",
    "KUBERNETES_BACKENDS",
    "KUBERNETES_NAMESPACE_ENV",
    "ClusterCapacityPass",
    "PoolCapacityTarget",
    "cluster_capacity_policy_from_env",
    "kubernetes_capacity_target",
)
