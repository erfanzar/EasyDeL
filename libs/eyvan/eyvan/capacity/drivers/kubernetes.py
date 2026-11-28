# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Kubernetes-backed capacity: one cell is one Indexed Job.

Structurally a near-clone of ``manual.py`` -- the same fencing, the same durable
``FleetStateStore`` record, the same mutation acceptance -- plus calls to the
Kubernetes API. That similarity is the point: the parts that are hard to get
right are backend-neutral, and a second real backend that had to reinvent them
would mean the contracts do not generalize.

**Fencing stays in ``FleetStateStore``.** Kubernetes cannot substitute for it.
``metadata.resourceVersion`` is opaque and bumps on *any* write, including
kubelet status updates, so a compare-and-swap on it produces spurious
``StaleMutationError``s from writes Eyvan did not make. ``metadata.generation``
does not survive recreation. What Kubernetes *does* provide is leak detection: a
generation label and a fencing-token **fingerprint** annotation, so an observed
Job whose fingerprint disagrees with the durable record is a leaked previous
incarnation and gets deleted rather than adopted -- the role ``management_uid``
plays on GCP.

**Kueue coexistence is a boundary, not a compromise.** Two admission controllers
over the same nodes fight twice: Kueue can veto a cell Eyvan already leased, and
Kueue can *evict* an admitted Workload for higher-priority work in another
LocalQueue -- destroying a pod with no checkpoint hook and bypassing Eyvan's
entire ``PreemptionPlan`` / ``CHECKPOINT_AND_STOP`` / ``grace_period_seconds``
machinery. So: **Eyvan delegates capacity admission to Kueue and retains workload
admission, and a ClusterQueue Eyvan uses must be exclusively Eyvan's with
preemption disabled.** The driver verifies that precondition before creating a Job, which is what
turns an invisible correctness hazard into a refusal at the point of use.
``eyvan doctor`` also reports it, but only as a warning: it runs on an
operator's machine and cannot read the cluster.

**Honest cost, and what has been paid down.** Eyvan's budgets and fair share
reason about cells it asked for, not the cluster's real free capacity, so an
Eyvan pool can sit WAITING while the cluster is full of non-Eyvan work.
``CapacityBelief`` *learns* that -- repeated timeouts raise ``route_priority``.
Two things now also *read* it, both through
:mod:`eyvan.capacity.providers.kubernetes.kueue`:

* ``observe`` names the cause on a WAITING cell instead of restating Eyvan's own
  state, resolving the Job's ``kueue.x-k8s.io/queue-name`` label -- a *LocalQueue*
  -- to the ClusterQueue that holds the quota. Every failure in that chain leaves
  the detail exactly as it was: it is a diagnostic, and a diagnostic that can
  break an observation is worse than no diagnostic.
* the autoscale pass (:mod:`eyvan.capacity.autoscale.pool_capacity`) takes one
  reading per Kubernetes pool before launching. That half is default-off and
  issues no request at all until an operator enables it.

Neither ever concludes "full" from a reading it could not take. An unreadable
cluster is UNKNOWN, and UNKNOWN changes nothing.
"""

from __future__ import annotations

import base64
import copy
import logging
import os
import time
import typing as tp
from dataclasses import dataclass
from datetime import datetime
from threading import RLock

from eyvan._internal.compat import UTC
from eyvan._internal.identifiers import stable_id
from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import (
    CapacityProviderError,
    CapacityProviderFailureCode,
    CellCapabilities,
    CellObservation,
    MutationContext,
    StaleMutationError,
)
from eyvan.capacity.fleet_state import (
    FleetCellNotRegisteredError,
    FleetCellRecord,
    FleetStateStore,
    StaleFenceError,
)
from eyvan.capacity.providers.kubernetes.api import (
    QUEUE_NAME_LABEL,
    JobStatus,
    KubernetesApi,
    KubernetesCreateRejected,
    capacity_state,
    interpret_job,
    local_queue_cluster_queue,
)
from eyvan.capacity.providers.kubernetes.kueue import (
    ClusterCapacityExplanation,
    ClusterCapacityReading,
    cell_capacity_state,
    explain_cluster_capacity,
    inspect_cluster_queue,
    observe_cluster_capacity,
)
from eyvan.capacity.providers.kubernetes.workers import (
    API_CLIENT_LABEL,
    CELL_ID_ANNOTATION,
    CELL_LABEL,
    FENCE_ANNOTATION,
    GENERATION_LABEL,
    MAX_INDEXED_JOB_WORKERS,
    cell_label_value,
    fence_fingerprint,
    headless_service_manifest,
    indexed_job_name,
    job_manifest,
    object_name,
    secret_manifest,
    worker_endpoints,
)

#: Where Kueue publishes its ClusterQueue objects.
_KUEUE_API = "/apis/kueue.x-k8s.io/v1beta1"

#: Most workers one Kubernetes cell may contain. A Job's parallelism is the
#: multislice worker count, and the same ceiling that bounds a TPU multislice
#: bounds this: past it, a single failure unit is too large to reason about.
MAX_WORKERS = MAX_INDEXED_JOB_WORKERS

_Revalidate = tp.Callable[[str], None]
_PENDING_CREATES_KEY = "kubernetes_pending_creates"
_TOMBSTONE_CLEANUP_KEY = "kubernetes_cleanup"

#: How long a LocalQueue -> ClusterQueue mapping and a ClusterQueue reading stay
#: usable. One reconcile pass observes every cell in a pool and they all share
#: one queue, so an uncached diagnostic issues two describing GETs *per cell*
#: against an API server Eyvan does not own. Seconds rather than minutes because
#: the reading's whole value is that it describes the cluster *now*: the planner
#: refuses anything older than
#: :data:`~eyvan.capacity.autoscale.cluster_capacity.DEFAULT_MAX_OBSERVATION_AGE_S`
#: anyway, and a stale explanation on a cell detail is a wrong explanation.
CLUSTER_CAPACITY_TTL_S = 5.0

#: Most (namespace, queue) mappings and ClusterQueue readings held at once. An
#: agent serves one namespace and a handful of queues; the bound exists so a
#: misconfigured fleet that names a new queue every pass cannot grow the cache
#: without limit.
_MAX_CACHED_QUEUES = 64

#: Longest cell detail this driver will emit once a capacity explanation has been
#: appended. ``CellRecord.unavailable_detail`` is capped at 2048 and a lifecycle
#: reason lands in a log line, a dashboard panel, and an audit row; an
#: explanation that names four resources with remedies can run long enough to
#: crowd out the state it was meant to qualify.
_MAX_DETAIL = 512

#: Set to a false value to restore the pre-W3 wording exactly: a WAITING cell
#: then says "job is suspended pending kueue admission" and nothing more, and no
#: capacity GET is issued at all. On by default because the addition is a
#: *diagnostic suffix* on a free-text field -- it never changes a cell's state,
#: a decision, or a wire field's meaning -- but an operator who wants the old
#: bytes should not need a rebuild to get them.
CLUSTER_CAPACITY_DETAIL_ENV = "EYVAN_KUBERNETES_CLUSTER_CAPACITY_DETAIL"

_FALSE = frozenset({"0", "false", "no", "off"})

#: Distinguishes "this key holds ``None``" from "this key is not cached". Both
#: caches store ``None`` as a real answer -- an unresolvable queue and an
#: unreadable cluster are exactly the entries worth remembering.
_MISSING = object()

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _CreateOwnership:
    cell_id: str
    generation: int
    fencing_token: str
    provider_generation: int


class KubernetesCapacityDriver:
    """Provision one Indexed Job per Eyvan capacity cell."""

    provider_name = "kubernetes"
    execution_backend = "eyvan-native"
    accelerator_types = ("cpu", "gpu", "tpu")
    capacity_classes = ("on-demand", "reserved", "spot")

    def __init__(
        self,
        state: FleetStateStore | None = None,
        *,
        api: KubernetesApi | None = None,
        namespace: str = "eyvan",
        image: str = "",
        shared_secret: str | None = None,
        #: Whether a WAITING cell's detail may name the cluster's free capacity.
        #: ``None`` reads :data:`CLUSTER_CAPACITY_DETAIL_ENV`, which defaults on.
        cluster_capacity_detail: bool | None = None,
        cluster_capacity_ttl_s: float = CLUSTER_CAPACITY_TTL_S,
    ):
        self.state = state or FleetStateStore.from_uri(None)
        self.api = api or KubernetesApi()
        self.namespace = namespace
        self.image = image
        self.shared_secret = shared_secret
        self._mutation_lock = RLock()
        self.cluster_capacity_detail = (
            os.getenv(CLUSTER_CAPACITY_DETAIL_ENV, "").strip().lower() not in _FALSE
            if cluster_capacity_detail is None
            else cluster_capacity_detail
        )
        self.cluster_capacity_ttl_s = max(float(cluster_capacity_ttl_s), 0.0)
        # Separate from ``_mutation_lock``: this guards two read caches and is
        # never held across an HTTP call, so a slow API server cannot serialize
        # observation behind it.
        self._capacity_lock = RLock()
        self._local_queue_cache: dict[tuple[str, str], tuple[float, str | None]] = {}
        self._reading_cache: dict[str, tuple[float, ClusterCapacityReading | None]] = {}

    def capabilities(self) -> CellCapabilities:
        """Describe Kubernetes-backed capacity.

        ``resource_enforcement`` is ``"backend"``, and that claim is honest *only
        because of exclusivity*: ``to_core_cell`` sets ``exclusive_workloads`` for
        native executors, so a native cell hosts exactly one workload and the
        pod's cgroup **is** that workload's cgroup. If ``exclusive_workloads``
        ever became false for native cells, this must drop to ``"none"``.
        """

        return CellCapabilities(
            provider=self.provider_name,
            executor=self.execution_backend,
            accelerator_types=self.accelerator_types,
            capacity_classes=self.capacity_classes,
            capacity_backends=("kubernetes",),
            supports_callables=True,
            supports_checkpoint=True,
            # ``kubectl exec`` exists regardless; routing it through Eyvan is what
            # makes it fenced and audited.
            supports_exec=True,
            resource_enforcement="backend",
        )

    # ---- durable record helpers, shared in shape with manual.py -------------

    @staticmethod
    def _accept_mutation(record: FleetCellRecord, context: MutationContext) -> bool:
        if context.generation < record.generation:
            raise StaleMutationError(
                f"generation {context.generation} is older than cell {record.name!r} generation {record.generation}"
            )
        if (
            context.generation == record.generation
            and record.fencing_token is not None
            and record.fencing_token != context.fencing_token
        ):
            raise StaleMutationError(f"fencing token does not own cell {record.name!r} generation {record.generation}")
        return context.idempotency_key not in record.idempotency_keys

    @staticmethod
    def _stamp(record: FleetCellRecord, context: MutationContext) -> None:
        record.generation = context.generation
        record.fencing_token = context.fencing_token
        record.idempotency_keys = [*record.idempotency_keys, context.idempotency_key][-64:]

    @staticmethod
    def _nested_options(desired: tp.Mapping[str, JsonValue]) -> JsonObject:
        for key in ("provider_config", "provider_options"):
            value = desired.get(key)
            if is_json_object(value):
                return value
        return {}

    @staticmethod
    def _required_string(desired: tp.Mapping[str, JsonValue], name: str) -> str:
        value = desired.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"kubernetes capacity field {name!r} must be a non-empty string")
        return value.strip()

    def _worker_count(self, desired: tp.Mapping[str, JsonValue]) -> int:
        options = self._nested_options(desired)
        raw = desired.get("worker_count", options.get("worker_count", options.get("node_count", 1)))
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise ValueError("kubernetes capacity worker_count must be a positive integer")
        if raw > MAX_WORKERS:
            raise ValueError(f"kubernetes capacity supports at most {MAX_WORKERS} workers per cell")
        return raw

    def _validate_desired(self, desired: tp.Mapping[str, JsonValue]) -> tuple[str, str, int, JsonObject]:
        backend = self._required_string(desired, "capacity_backend")
        if backend not in {"kubernetes", "k8s"}:
            raise ValueError(f"kubernetes capacity requires capacity_backend 'kubernetes', got {backend!r}")
        executor = self._required_string(desired, "executor")
        if executor != self.execution_backend:
            raise ValueError(f"kubernetes capacity requires executor {self.execution_backend!r}, got {executor!r}")
        accelerator_type = self._required_string(desired, "accelerator_type")
        capacity_class = self._required_string(desired, "capacity_class").replace("_", "-")
        if capacity_class not in self.capacity_classes:
            choices = ", ".join(self.capacity_classes)
            raise ValueError(f"unsupported kubernetes capacity class {capacity_class!r}; choose one of {choices}")
        return accelerator_type, capacity_class, self._worker_count(desired), self._nested_options(desired)

    # ---- observation --------------------------------------------------------

    def observe(self, cell_id: str) -> CellObservation:
        """Return the cell's live state, reconciled against the durable record."""

        record = self.state.get(cell_id)
        if record is None:
            return CellObservation(
                cell_id=cell_id,
                state=CapacityStatus.ABSENT,
                generation=0,
                observed_at=datetime.now(UTC),
            )
        payload = self.api.get_job(self.namespace, indexed_job_name(cell_id))
        status = interpret_job(payload)
        ownership_error = self._job_ownership_error(record, status)
        if ownership_error is not None:
            # Missing metadata is as unsafe as a mismatch. Adopting either would
            # publish endpoints for a Job whose cell generation cannot be proven.
            return self._observation(
                record,
                CapacityStatus.NEEDS_RECREATE,
                ownership_error,
                worker_count=status.parallelism,
                ready=0,
            )
        state, detail = capacity_state(status)
        if record.desired_state != "up" and state in {
            CapacityStatus.HEALTHY,
            CapacityStatus.PROVISIONING,
            CapacityStatus.WAITING,
        }:
            # A drain deliberately keeps the owned Job running. Provider
            # readiness must not restore scheduling or endpoints, but actual
            # loss/failure and unproven ownership still retain their own states.
            state, detail = CapacityStatus.DRAINING, f"cell is draining; {detail}"
        if state is CapacityStatus.WAITING:
            # The one state whose detail restates Eyvan's own state ("suspended
            # pending kueue admission") instead of naming a cause. Every other
            # state already says something an operator can act on, and reading
            # cluster capacity to qualify a HEALTHY cell would spend describing
            # GETs on a question nobody asked.
            detail = self._explained_waiting_detail(status, detail)
        return self._observation(record, state, detail, worker_count=status.parallelism, ready=status.ready)

    # ---- cluster capacity, as a diagnostic that can only add ---------------

    def _explained_waiting_detail(self, status: JobStatus, detail: str) -> str:
        """Return the WAITING detail, with the cluster's free capacity if provable.

        Every failure anywhere in the chain -- no queue label, an unreadable
        LocalQueue, an unreadable ClusterQueue, a queue that has not published
        its reservation, an unparseable request -- returns ``detail`` byte for
        byte. This is a diagnostic layered on top of an observation the control
        plane acts on, and a diagnostic that can break an observation is worse
        than no diagnostic.

        Args:
            status: The interpreted Job.
            detail: Today's detail, which is also the answer whenever the cluster
                establishes nothing.

        Returns:
            ``detail``, or ``detail`` followed by the observed reason there is no
            room, bounded by :data:`_MAX_DETAIL`.
        """

        try:
            explanation = self._cluster_capacity_explanation(status)
            if explanation is None:
                return detail
            _, explained = cell_capacity_state(status, explanation)
        except Exception:  # the diagnostic must never fail the observation
            # Deliberately broad. The read path below is contractually
            # never-raises, so reaching here means something upstream broke its
            # own contract -- and the correct answer to that is still the cell's
            # real state with today's wording, not a failed reconciliation.
            logger.debug("cluster capacity diagnostic failed for a queued Kubernetes cell", exc_info=True)
            return detail
        return explained if len(explained) <= _MAX_DETAIL else f"{explained[: _MAX_DETAIL - 3]}..."

    def _cluster_capacity_explanation(self, status: JobStatus) -> ClusterCapacityExplanation | None:
        """Return an observed cluster explanation for this cell, or ``None``.

        Returns:
            A ``ClusterCapacityExplanation`` when the cluster's free capacity was
            read against the right ClusterQueue for a request this cell actually
            declares, otherwise ``None``.
        """

        if not self.cluster_capacity_detail or not status.queue_label:
            return None
        request = {name: value for name, value in status.resource_requests}
        if not request:
            return None
        namespace = status.namespace or self.namespace
        cluster_queue = self._cluster_queue_for(namespace, status.queue_label)
        if not cluster_queue:
            return None
        reading = self._cluster_capacity_reading(cluster_queue)
        if reading is None:
            return None
        return explain_cluster_capacity(reading, request)

    def _cluster_queue_for(self, namespace: str, local_queue: str) -> str | None:
        """Return the ClusterQueue behind one LocalQueue, cached for a few seconds.

        A failed resolution is cached exactly as a successful one is. A cluster
        where Eyvan has no ``get localqueues`` refuses identically for every cell
        in the pool, and re-asking once per cell would spend the whole cost the
        cache exists to avoid on the answer that establishes nothing.
        """

        key = (namespace, local_queue)
        cached = self._cached(self._local_queue_cache, key)
        if cached is not _MISSING:
            return tp.cast("str | None", cached)
        resolved = local_queue_cluster_queue(self.api.observe_local_queue(namespace, local_queue))
        self._store(self._local_queue_cache, key, resolved)
        return resolved

    def _cluster_capacity_reading(self, cluster_queue: str) -> ClusterCapacityReading | None:
        """Return one ClusterQueue's capacity, cached for a few seconds.

        No ResourceFlavor is named, so no node list is read: the node reading
        only ever produces the "this quota is not backed by hardware" remedy, and
        paying a third GET per pool for a sentence is not a trade a cell
        observation should make.
        """

        cached = self._cached(self._reading_cache, cluster_queue)
        if cached is not _MISSING:
            return tp.cast("ClusterCapacityReading | None", cached)
        reading = observe_cluster_capacity(self.api, cluster_queue)
        self._store(self._reading_cache, cluster_queue, reading)
        return reading

    def _cached(self, cache: dict[tp.Any, tuple[float, tp.Any]], key: tp.Any) -> tp.Any:
        """Return a live cache entry, or ``_MISSING`` when there is none."""

        if self.cluster_capacity_ttl_s <= 0:
            return _MISSING
        now = time.monotonic()
        with self._capacity_lock:
            entry = cache.get(key)
            if entry is None:
                return _MISSING
            expires_at, value = entry
            if expires_at <= now:
                cache.pop(key, None)
                return _MISSING
            return value

    def _store(self, cache: dict[tp.Any, tuple[float, tp.Any]], key: tp.Any, value: tp.Any) -> None:
        """Cache one value for :data:`CLUSTER_CAPACITY_TTL_S`, bounded in size."""

        if self.cluster_capacity_ttl_s <= 0:
            return
        now = time.monotonic()
        with self._capacity_lock:
            for stale, (expires_at, _value) in list(cache.items()):
                if expires_at <= now:
                    cache.pop(stale, None)
            if len(cache) >= _MAX_CACHED_QUEUES and key not in cache:
                cache.pop(next(iter(cache)), None)
            cache[key] = (now + self.cluster_capacity_ttl_s, value)

    def _job_ownership_error(self, record: FleetCellRecord, status: JobStatus) -> str | None:
        """Return why an observed Job cannot be tied to the durable cell fence."""

        if not status.exists:
            return None
        expected_fingerprint = fence_fingerprint(record.fencing_token) if isinstance(record.fencing_token, str) else ""
        expected = (
            status.api_version == "batch/v1"
            and status.kind == "Job"
            and status.name == indexed_job_name(record.name)
            and status.namespace == self.namespace
            and bool(status.uid)
            and bool(status.resource_version)
            and status.cell_label == cell_label_value(record.name)
            and status.cell_id_annotation == record.name
            and status.generation_label == str(record.provider_generation)
            and bool(expected_fingerprint)
            and status.fence_fingerprint == expected_fingerprint
        )
        if expected:
            return None
        return "job ownership metadata is missing or does not match the durable cell fence"

    def _observation(
        self,
        record: FleetCellRecord,
        state: CapacityStatus,
        detail: str,
        *,
        worker_count: int,
        ready: int,
    ) -> CellObservation:
        """Build an observation, publishing endpoints only when every worker is ready.

        Mirrors ``drivers/gcp.py:_persist_provider_observation``: publishing early
        makes the agent's readiness probe resolve a DNS name that has no backing
        pod, and the cell flaps HEALTHY <-> EXECUTION_CONNECTING for as long as
        the slowest worker takes to start.
        """

        declared = worker_count or int(record.metadata.get("worker_count", 1) or 1)
        endpoints = (
            worker_endpoints(record.name, self.namespace, declared)
            if record.desired_state == "up" and declared and ready >= declared
            else ()
        )
        metadata: JsonObject = {
            "provider": self.provider_name,
            "capacity_backend": self.provider_name,
            "resource_id": f"kubernetes:{self.namespace}/{indexed_job_name(record.name)}",
            "desired_state": record.desired_state,
            "provider_generation": record.provider_generation,
            **record.metadata,
            "native_worker_endpoints": list(endpoints),
        }
        return CellObservation(
            cell_id=record.name,
            state=state,
            generation=record.generation,
            execution_endpoint=endpoints[0] if len(endpoints) == 1 else None,
            detail=detail,
            observed_at=datetime.now(UTC),
            metadata=metadata,
            observed_tpu_vms=declared if endpoints else None,
            runtime_process_count=len(endpoints) if endpoints else None,
        )

    # ---- lifecycle ----------------------------------------------------------

    @staticmethod
    def _conflict(kind: str, detail: str) -> CapacityProviderError:
        return CapacityProviderError(
            f"Kubernetes {kind} conflict is not a verified Eyvan replay: {detail}",
            code=CapacityProviderFailureCode.CONFIGURATION,
            retryable=False,
        )

    @staticmethod
    def _retry_reconcile(kind: str, detail: str) -> CapacityProviderError:
        return CapacityProviderError(
            f"Kubernetes {kind} reconciliation must be retried: {detail}",
            code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
            retryable=True,
        )

    def _revalidate_record_owner(
        self,
        cell_id: str,
        *,
        generation: int,
        fencing_token: str,
        provider_generation: int,
        phase: str,
        desired_state: str | None = None,
    ) -> None:
        """Require the live fleet lease and the exact durable cell incarnation.

        A renewable lease is useful only while callers notice losing it. Check
        immediately around Kubernetes mutations, and check the durable record in
        between two lease reads so an ownership change cannot be mistaken for a
        successful provider reconciliation.
        """

        try:
            self.state.require_lease()
            record = self.state.get(cell_id)
            self.state.require_lease()
        except StaleFenceError as exc:
            raise StaleMutationError(f"cell {cell_id!r} lost its fleet mutation lease {phase}") from exc
        expected = (
            record is not None
            and record.generation == generation
            and record.fencing_token == fencing_token
            and record.provider_generation == provider_generation
            and (desired_state is None or record.desired_state == desired_state)
        )
        if not expected:
            raise StaleMutationError(f"cell {cell_id!r} changed durable ownership {phase}")

    @staticmethod
    def _record_owns_create(record: FleetCellRecord, ownership: _CreateOwnership) -> bool:
        return (
            record.generation == ownership.generation
            and record.fencing_token == ownership.fencing_token
            and record.provider_generation == ownership.provider_generation
        )

    @staticmethod
    def _create_intent(path: str, kind: str, ownership: _CreateOwnership, *, status: str) -> JsonObject:
        return {
            "path": path,
            "kind": kind,
            "generation": ownership.generation,
            "provider_generation": ownership.provider_generation,
            "fence_fingerprint": fence_fingerprint(ownership.fencing_token),
            "status": status,
        }

    @staticmethod
    def _pending_creates(record: FleetCellRecord) -> JsonObject:
        value = record.metadata.get(_PENDING_CREATES_KEY)
        if value is None:
            return {}
        if not is_json_object(value):
            raise CapacityProviderError(
                f"cell {record.name!r} has malformed Kubernetes create-intent state",
                code=CapacityProviderFailureCode.CONFIGURATION,
                retryable=False,
            )
        return value

    def _require_no_inflight_creates(self, record: FleetCellRecord, action: str) -> JsonObject:
        """Refuse an ownership transition while an older POST can still land."""

        pending = self._pending_creates(record)
        for intent in pending.values():
            if not is_json_object(intent) or intent.get("status") not in {"inflight", "settled"}:
                raise self._conflict(action, "the durable Kubernetes create intent is malformed")
        if any(intent.get("status") == "inflight" for intent in pending.values()):
            raise self._retry_reconcile(
                action,
                "an older Kubernetes POST is still inflight; refusing to advance durable ownership",
            )
        return pending

    def _begin_create_intent(self, path: str, kind: str, ownership: _CreateOwnership) -> JsonObject:
        """Durably announce a POST before it can reach the API server."""

        intent = self._create_intent(path, kind, ownership, status="inflight")

        def mutate(record: FleetCellRecord) -> None:
            if not self._record_owns_create(record, ownership) or record.desired_state != "up":
                raise StaleMutationError(f"cell {ownership.cell_id!r} changed ownership before {kind} create intent")
            pending = self._pending_creates(record)
            others = [key for key in pending if key != path]
            if others:
                raise self._retry_reconcile(kind, "another Kubernetes create request is still unresolved")
            current = pending.get(path)
            if current is not None and not is_json_object(current):
                raise self._conflict(kind, "the durable create intent is malformed")
            if is_json_object(current) and current.get("status") == "inflight":
                raise self._retry_reconcile(kind, "the previous Kubernetes create request is still unresolved")
            if is_json_object(current) and current.get("status") != "settled":
                raise self._conflict(kind, "the durable create intent has an unknown status")
            pending[path] = intent
            record.metadata[_PENDING_CREATES_KEY] = pending

        try:
            self.state.mutate_record(ownership.cell_id, mutate, generation=ownership.generation)
        except StaleFenceError as exc:
            raise StaleMutationError(f"cell {ownership.cell_id!r} changed ownership before {kind} POST") from exc
        return intent

    def _transition_create_intent(
        self,
        ownership: _CreateOwnership,
        intent: JsonObject,
        *,
        status: str | None,
    ) -> bool:
        """Settle or remove exactly one intent, even after its lease was lost.

        This CAS changes no provider ownership and cannot authorize another POST.
        It is the completion acknowledgement a newer release needs in order to
        know that the old API request can no longer land after an absent GET.
        """

        changed: dict[str, bool] = {}
        path = intent.get("path")
        if not isinstance(path, str):  # pragma: no cover - internal invariant
            return False

        def mutate(record: FleetCellRecord) -> None:
            if not self._record_owns_create(record, ownership):
                changed["ok"] = False
                return
            pending = self._pending_creates(record)
            current = pending.get(path)
            if not is_json_object(current):
                changed["ok"] = False
                return
            expected = {key: value for key, value in intent.items() if key != "status"}
            actual = {key: value for key, value in current.items() if key != "status"}
            if actual != expected:
                changed["ok"] = False
                return
            if status is None:
                pending.pop(path, None)
                if pending:
                    record.metadata[_PENDING_CREATES_KEY] = pending
                else:
                    record.metadata.pop(_PENDING_CREATES_KEY, None)
            else:
                current["status"] = status
                pending[path] = current
                record.metadata[_PENDING_CREATES_KEY] = pending
            changed["ok"] = True

        try:
            self.state.mutate_record(ownership.cell_id, mutate, generation=ownership.generation)
        except (FleetCellNotRegisteredError, StaleFenceError):
            return False
        return bool(changed.get("ok"))

    @classmethod
    def _resource_matches(cls, observed: JsonObject, desired: JsonObject, kind: str) -> bool:
        if kind == "Service":
            return cls._service_matches(observed, desired)
        if kind == "Secret":
            return cls._secret_matches(observed, desired)
        if kind == "Job":
            return cls._job_matches(observed, desired)
        return False

    def _cleanup_settled_create(self, path: str, desired: JsonObject, kind: str) -> bool:
        """Delete only the exact stale object and report whether that incarnation is gone."""

        observed = self.api.get(path)
        if observed is None:
            return True
        try:
            _metadata, uid, resource_version = self._resource_identity(observed, desired, kind)
        except CapacityProviderError:
            # A malformed or replacement object is not the settled create this
            # caller owns. Never delete it merely because it reused the name.
            return True
        if not self._resource_matches(observed, desired, kind):
            return True
        self.api.delete(path, uid=uid, resource_version=resource_version)
        remaining = self.api.get(path)
        if remaining is None:
            return True
        try:
            _metadata, remaining_uid, _remaining_version = self._resource_identity(remaining, desired, kind)
        except CapacityProviderError:
            return True
        return remaining_uid != uid

    def _create_revalidated(
        self,
        collection_path: str,
        desired: JsonObject,
        kind: str,
        revalidate: _Revalidate,
        ownership: _CreateOwnership,
    ) -> bool:
        metadata = desired.get("metadata")
        if not is_json_object(metadata) or not isinstance(metadata.get("name"), str):
            raise self._conflict(kind, "create manifest omitted metadata.name")
        object_path = f"{collection_path}/{metadata['name']}"
        revalidate(f"before {kind} create intent")
        intent = self._begin_create_intent(object_path, kind, ownership)
        revalidate(f"before {kind} create")
        # If this raises without a response, the POST outcome is indeterminate.
        # The durable intent deliberately remains inflight so release refuses to
        # trust an absent preflight while the API server may still store it.
        try:
            created = self.api.create(collection_path, desired)
        except KubernetesCreateRejected:
            # Only a proved completed refusal permits this acknowledgement.
            # CAS-remove exactly this owner/request; never clear a successor's
            # intent or use an absent GET to resolve an indeterminate POST.
            self._transition_create_intent(ownership, intent, status=None)
            raise

        settled = self._transition_create_intent(ownership, intent, status="settled")
        try:
            revalidate(f"after {kind} create")
        except StaleMutationError:
            # The POST has returned, so it cannot land later. Remove only this
            # generation/fence and exact UID/resourceVersion; a newer object is
            # deliberately left untouched. The durable tombstone replay is a
            # second cleanup path if deletion itself is inconclusive.
            if settled and self._cleanup_settled_create(object_path, desired, kind):
                self._transition_create_intent(ownership, intent, status=None)
            raise
        if not settled:
            raise StaleMutationError(f"cell {ownership.cell_id!r} changed ownership after {kind} create")
        if not self._transition_create_intent(ownership, intent, status=None):
            raise StaleMutationError(f"cell {ownership.cell_id!r} changed ownership after {kind} create")
        revalidate(f"after {kind} create intent resolution")
        return created

    @staticmethod
    def _expected_subset(observed: object, expected: object) -> bool:
        """Compare every requested field while tolerating API-server defaults."""

        if isinstance(expected, dict):
            return isinstance(observed, dict) and all(
                key in observed and KubernetesCapacityDriver._expected_subset(observed[key], value)
                for key, value in expected.items()
            )
        if isinstance(expected, list):
            return (
                isinstance(observed, list)
                and len(observed) == len(expected)
                and all(
                    KubernetesCapacityDriver._expected_subset(actual, wanted)
                    for actual, wanted in zip(observed, expected, strict=True)
                )
            )
        return observed == expected

    @classmethod
    def _resource_identity(
        cls,
        payload: object,
        desired: JsonObject,
        kind: str,
    ) -> tuple[JsonObject, str, str]:
        """Validate an exact GET and return metadata plus delete preconditions."""

        if not is_json_object(payload):
            raise cls._conflict(kind, "GET returned a non-object payload")
        expected_metadata = desired.get("metadata")
        metadata = payload.get("metadata")
        if not is_json_object(expected_metadata) or not is_json_object(metadata):
            raise cls._conflict(kind, "GET omitted object metadata")
        if payload.get("apiVersion") != desired.get("apiVersion") or payload.get("kind") != desired.get("kind"):
            raise cls._conflict(kind, "GET returned the wrong resource kind")
        for field in ("name", "namespace"):
            expected = expected_metadata.get(field)
            if not isinstance(expected, str) or not expected or metadata.get(field) != expected:
                raise cls._conflict(kind, f"GET returned the wrong metadata.{field}")
        uid = metadata.get("uid")
        resource_version = metadata.get("resourceVersion")
        if not isinstance(uid, str) or not uid or not isinstance(resource_version, str) or not resource_version:
            raise cls._conflict(kind, "GET omitted uid/resourceVersion fencing metadata")
        return metadata, uid, resource_version

    @classmethod
    def _metadata_matches(cls, observed: JsonObject, desired: JsonObject) -> bool:
        expected_metadata = desired.get("metadata")
        actual_metadata = observed.get("metadata")
        if not is_json_object(expected_metadata) or not is_json_object(actual_metadata):
            return False
        for field in ("name", "namespace"):
            if actual_metadata.get(field) != expected_metadata.get(field):
                return False
        for field in ("labels", "annotations"):
            expected = expected_metadata.get(field)
            if expected is not None and not cls._expected_subset(actual_metadata.get(field), expected):
                return False
        return True

    @classmethod
    def _service_matches(cls, observed: JsonObject, desired: JsonObject) -> bool:
        if not cls._metadata_matches(observed, desired):
            return False
        expected_spec = desired.get("spec")
        actual_spec = observed.get("spec")
        if not is_json_object(expected_spec) or not is_json_object(actual_spec):
            return False
        if not cls._expected_subset(actual_spec, expected_spec):
            return False
        # A selector with extras silently selects no workers; unexpected exposure
        # fields can turn a private headless Service into a public endpoint.
        if actual_spec.get("selector") != expected_spec.get("selector"):
            return False
        if actual_spec.get("type") not in (None, "ClusterIP"):
            return False
        for field in ("externalIPs", "loadBalancerIP", "loadBalancerSourceRanges", "externalName"):
            if actual_spec.get(field) not in (None, [], ""):
                return False
        return True

    @staticmethod
    def _secret_data(payload: JsonObject) -> JsonObject | None:
        data = payload.get("data")
        if is_json_object(data):
            return data
        # The real API returns data, but accepting stringData keeps strict fakes
        # and admission-free fixtures representative without weakening equality.
        string_data = payload.get("stringData")
        if not is_json_object(string_data):
            return None
        encoded: JsonObject = {}
        for key, value in string_data.items():
            if not isinstance(value, str):
                return None
            encoded[key] = base64.b64encode(value.encode()).decode()
        return encoded

    @classmethod
    def _secret_matches(cls, observed: JsonObject, desired: JsonObject) -> bool:
        if not cls._metadata_matches(observed, desired) or observed.get("type") != desired.get("type"):
            return False
        if observed.get("immutable") not in (None, False):
            return False
        expected = cls._secret_data(desired)
        actual = cls._secret_data(observed)
        return expected is not None and actual == expected

    @classmethod
    def _job_matches(cls, observed: JsonObject, desired: JsonObject) -> bool:
        if not cls._metadata_matches(observed, desired):
            return False
        expected_spec = desired.get("spec")
        actual_spec = observed.get("spec")
        if not is_json_object(expected_spec) or not is_json_object(actual_spec):
            return False
        metadata = desired.get("metadata")
        labels = metadata.get("labels") if is_json_object(metadata) else None
        queue_name = labels.get(QUEUE_NAME_LABEL) if is_json_object(labels) else None
        if (
            isinstance(queue_name, str)
            and queue_name
            and expected_spec.get("suspend") is True
            and actual_spec.get("suspend") is False
        ):
            # Kueue owns the admission transition, not the worker identity or
            # security-critical pod spec. Compare an unsuspended, exactly owned
            # queue-backed Job without rewriting it or relaxing any other field.
            expected_spec = {**expected_spec, "suspend": False}
        if not cls._expected_subset(actual_spec, expected_spec):
            return False
        expected_template = expected_spec.get("template")
        actual_template = actual_spec.get("template")
        if not is_json_object(expected_template) or not is_json_object(actual_template):
            return False
        expected_pod = expected_template.get("spec")
        actual_pod = actual_template.get("spec")
        if not is_json_object(expected_pod) or not is_json_object(actual_pod):
            return False
        # Reject admission mutations that could execute different code, expose a
        # host, or mount a different credential. Ordinary Kubernetes defaults
        # (dnsPolicy, schedulerName, terminationGracePeriodSeconds, and generated
        # selector labels) remain allowed.
        for field in ("initContainers", "ephemeralContainers"):
            if field not in expected_pod and actual_pod.get(field) not in (None, []):
                return False
        for field in ("hostNetwork", "hostPID", "hostIPC", "shareProcessNamespace"):
            if field not in expected_pod and actual_pod.get(field) not in (None, False):
                return False
        for field in ("nodeSelector", "affinity", "tolerations", "volumes", "imagePullSecrets", "runtimeClassName"):
            if field not in expected_pod and actual_pod.get(field) not in (None, [], {}):
                return False
        expected_containers = expected_pod.get("containers")
        actual_containers = actual_pod.get("containers")
        if not isinstance(expected_containers, list) or not isinstance(actual_containers, list):
            return False
        if len(expected_containers) != len(actual_containers):
            return False
        for actual, expected in zip(actual_containers, expected_containers, strict=True):
            if not is_json_object(actual) or not is_json_object(expected):
                return False
            for field in (
                "command",
                "args",
                "workingDir",
                "envFrom",
                "securityContext",
                "lifecycle",
                "volumeMounts",
                "volumeDevices",
            ):
                if field not in expected and actual.get(field) not in (None, [], {}):
                    return False
        return True

    def _read_conflict(
        self,
        path: str,
        desired: JsonObject,
        kind: str,
    ) -> tuple[JsonObject, str, str]:
        payload = self.api.get(path)
        if payload is None:
            raise self._retry_reconcile(kind, "the conflicting object disappeared before GET")
        _metadata, uid, resource_version = self._resource_identity(payload, desired, kind)
        return payload, uid, resource_version

    def _delete_fenced(
        self,
        path: str,
        desired: JsonObject,
        kind: str,
        *,
        uid: str,
        resource_version: str,
        revalidate: _Revalidate,
    ) -> None:
        revalidate(f"before {kind} delete")
        self.api.delete(path, uid=uid, resource_version=resource_version)
        revalidate(f"after {kind} delete")
        remaining = self.api.get(path)
        revalidate(f"after {kind} deletion observation")
        if remaining is None:
            return
        _metadata, remaining_uid, _remaining_version = self._resource_identity(remaining, desired, kind)
        if remaining_uid != uid:
            raise self._conflict(kind, "a replacement acquired the name during fenced deletion")
        raise self._retry_reconcile(kind, "foreground deletion is still in progress")

    @staticmethod
    def _release_ownership_manifest(
        kind: str,
        cell_id: str,
        namespace: str,
        *,
        provider_generation: int,
        fencing_fingerprint: str,
    ) -> JsonObject:
        """Return the complete metadata release must prove before deleting."""

        if kind == "Service":
            return headless_service_manifest(
                cell_id,
                namespace,
                provider_generation=provider_generation,
                fence=fencing_fingerprint,
            )
        if kind == "Job":
            api_version = "batch/v1"
            name = indexed_job_name(cell_id)
            labels = {
                API_CLIENT_LABEL: "true",
                CELL_LABEL: cell_label_value(cell_id),
                GENERATION_LABEL: str(provider_generation),
            }
        elif kind == "Secret":
            api_version = "v1"
            name = object_name("eyvan-token", cell_id)
            labels = {
                CELL_LABEL: cell_label_value(cell_id),
                GENERATION_LABEL: str(provider_generation),
            }
        else:  # pragma: no cover - closed internal call surface
            raise ValueError(f"unsupported Kubernetes release kind {kind!r}")
        return tp.cast(
            JsonObject,
            {
                "apiVersion": api_version,
                "kind": kind,
                "metadata": {
                    "name": name,
                    "namespace": namespace,
                    "labels": labels,
                    "annotations": {
                        CELL_ID_ANNOTATION: cell_id,
                        FENCE_ANNOTATION: fencing_fingerprint,
                    },
                },
            },
        )

    def _read_release_resource(
        self,
        path: str,
        desired: JsonObject,
        kind: str,
    ) -> tuple[str, str] | None:
        """Exact-GET and prove one object's ownership before any delete starts."""

        observed = self.api.get(path)
        if observed is None:
            return None
        _metadata, uid, resource_version = self._resource_identity(observed, desired, kind)
        matches = (
            self._service_matches(observed, desired) if kind == "Service" else self._metadata_matches(observed, desired)
        )
        if not matches:
            raise self._conflict(kind, "ownership metadata is missing or differs from the durable cell fence")
        return uid, resource_version

    @classmethod
    def _has_cell_ownership(cls, observed: JsonObject, desired: JsonObject) -> bool:
        actual_metadata = observed.get("metadata")
        expected_metadata = desired.get("metadata")
        if not is_json_object(actual_metadata) or not is_json_object(expected_metadata):
            return False
        actual_labels = actual_metadata.get("labels")
        expected_labels = expected_metadata.get("labels")
        actual_annotations = actual_metadata.get("annotations")
        expected_annotations = expected_metadata.get("annotations")
        return (
            is_json_object(actual_labels)
            and is_json_object(expected_labels)
            and is_json_object(actual_annotations)
            and is_json_object(expected_annotations)
            and isinstance(expected_labels.get(CELL_LABEL), str)
            and isinstance(expected_annotations.get(CELL_ID_ANNOTATION), str)
            and actual_labels.get(CELL_LABEL) == expected_labels.get(CELL_LABEL)
            and actual_annotations.get(CELL_ID_ANNOTATION) == expected_annotations.get(CELL_ID_ANNOTATION)
        )

    @classmethod
    def _service_disposition(cls, observed: JsonObject, desired: JsonObject) -> str:
        """Classify a Service as exact, safely stale, or a name collision."""

        if cls._service_matches(observed, desired):
            return "exact"
        if not cls._has_cell_ownership(observed, desired):
            return "collision"
        actual_metadata = observed.get("metadata")
        expected_metadata = desired.get("metadata")
        assert is_json_object(actual_metadata) and is_json_object(expected_metadata)
        actual_labels = actual_metadata.get("labels")
        expected_labels = expected_metadata.get("labels")
        actual_annotations = actual_metadata.get("annotations")
        expected_annotations = expected_metadata.get("annotations")
        if not all(
            is_json_object(value) for value in (actual_labels, expected_labels, actual_annotations, expected_annotations)
        ):
            return "collision"
        assert is_json_object(actual_labels)
        assert is_json_object(expected_labels)
        assert is_json_object(actual_annotations)
        assert is_json_object(expected_annotations)
        actual_generation = actual_labels.get(GENERATION_LABEL)
        actual_fingerprint = actual_annotations.get(FENCE_ANNOTATION)
        expected_generation = expected_labels.get(GENERATION_LABEL)

        # Upgrade the generation-less Service emitted by older Eyvan versions,
        # but only when its complete security-critical manifest is otherwise exact.
        if actual_generation is None and actual_fingerprint is None:
            legacy = copy.deepcopy(desired)
            legacy_metadata = legacy.get("metadata")
            assert is_json_object(legacy_metadata)
            legacy_labels = legacy_metadata.get("labels")
            legacy_annotations = legacy_metadata.get("annotations")
            assert is_json_object(legacy_labels) and is_json_object(legacy_annotations)
            legacy_labels.pop(GENERATION_LABEL, None)
            legacy_annotations.pop(FENCE_ANNOTATION, None)
            return "stale" if cls._service_matches(observed, legacy) else "collision"

        valid_fingerprint = (
            isinstance(actual_fingerprint, str)
            and len(actual_fingerprint) == 32
            and all(character in "0123456789abcdef" for character in actual_fingerprint)
        )
        if (
            not isinstance(actual_generation, str)
            or not actual_generation.isdecimal()
            or not isinstance(expected_generation, str)
            or not expected_generation.isdecimal()
            or not valid_fingerprint
        ):
            return "collision"
        return "stale" if int(actual_generation) < int(expected_generation) else "collision"

    @classmethod
    def _stale_secret_token(cls, observed: JsonObject, desired: JsonObject) -> bool:
        if not cls._has_cell_ownership(observed, desired) or observed.get("type") != "Opaque":
            return False
        if observed.get("immutable") not in (None, False):
            return False
        data = cls._secret_data(observed)
        if not is_json_object(data) or set(data) != {"token"} or not isinstance(data.get("token"), str):
            return False
        actual_metadata = observed.get("metadata")
        expected_metadata = desired.get("metadata")
        assert is_json_object(actual_metadata) and is_json_object(expected_metadata)
        actual_labels = actual_metadata.get("labels")
        expected_labels = expected_metadata.get("labels")
        actual_annotations = actual_metadata.get("annotations")
        expected_annotations = expected_metadata.get("annotations")
        if not is_json_object(actual_labels) or not is_json_object(expected_labels):
            return False
        actual_generation = actual_labels.get(GENERATION_LABEL)
        expected_generation = expected_labels.get(GENERATION_LABEL)
        actual_fingerprint = actual_annotations.get(FENCE_ANNOTATION) if is_json_object(actual_annotations) else None
        expected_fingerprint = (
            expected_annotations.get(FENCE_ANNOTATION) if is_json_object(expected_annotations) else None
        )
        # Early Secrets with the complete cell ID but no generation fence are
        # repairable, but only after any owned Job is deleted below.
        if actual_generation is None and actual_fingerprint is None:
            return True
        if not isinstance(actual_generation, str) or not actual_generation.isdecimal():
            return False
        if not isinstance(expected_generation, str) or not expected_generation.isdecimal():
            return False
        valid_fingerprint = (
            isinstance(actual_fingerprint, str)
            and len(actual_fingerprint) == 32
            and all(character in "0123456789abcdef" for character in actual_fingerprint)
        )
        if not valid_fingerprint:
            return False
        return int(actual_generation) < int(expected_generation) or (
            actual_generation == expected_generation and actual_fingerprint == expected_fingerprint
        )

    @classmethod
    def _job_disposition(cls, observed: JsonObject, desired: JsonObject) -> str:
        if cls._job_matches(observed, desired):
            return "exact"
        if not cls._has_cell_ownership(observed, desired):
            return "collision"
        actual_metadata = observed.get("metadata")
        expected_metadata = desired.get("metadata")
        assert is_json_object(actual_metadata) and is_json_object(expected_metadata)
        actual_labels = actual_metadata.get("labels")
        expected_labels = expected_metadata.get("labels")
        annotations = actual_metadata.get("annotations")
        if not is_json_object(actual_labels) or not is_json_object(expected_labels) or not is_json_object(annotations):
            return "collision"
        actual_generation = actual_labels.get(GENERATION_LABEL)
        expected_generation = expected_labels.get(GENERATION_LABEL)
        fingerprint = annotations.get(FENCE_ANNOTATION)
        if not isinstance(actual_generation, str) or not actual_generation.isdecimal():
            return "collision"
        if not isinstance(expected_generation, str) or not expected_generation.isdecimal():
            return "collision"
        if (
            int(actual_generation) < int(expected_generation)
            and isinstance(fingerprint, str)
            and len(fingerprint) == 32
            and all(character in "0123456789abcdef" for character in fingerprint)
        ):
            return "stale"
        return "collision"

    def _remove_job_before_owned_rotation(
        self,
        path: str,
        desired: JsonObject,
        revalidate: _Revalidate,
    ) -> None:
        existing = self.api.get(path)
        revalidate("after Job rotation preflight")
        if existing is None:
            return
        _metadata, uid, resource_version = self._resource_identity(existing, desired, "Job")
        disposition = self._job_disposition(existing, desired)
        if disposition == "collision":
            raise self._conflict("Job", "a colliding Job prevents safe dependency rotation")
        self._delete_fenced(
            path,
            desired,
            "Job",
            uid=uid,
            resource_version=resource_version,
            revalidate=revalidate,
        )

    def _ensure_service(
        self,
        collection_path: str,
        object_path: str,
        job_path: str,
        desired: JsonObject,
        desired_job: JsonObject,
        revalidate: _Revalidate,
        ownership: _CreateOwnership,
    ) -> None:
        if self._create_revalidated(collection_path, desired, "Service", revalidate, ownership):
            return
        existing, uid, resource_version = self._read_conflict(object_path, desired, "Service")
        revalidate("after Service conflict observation")
        disposition = self._service_disposition(existing, desired)
        if disposition == "exact":
            return
        if disposition != "stale":
            raise self._conflict("Service", "ownership, generation fence, or security-critical spec differs")
        # The stable DNS object belongs to one provider incarnation at a time.
        # Remove any exactly-owned Job before rotating it so running old workers
        # never observe a Service carrying a newer fence.
        self._remove_job_before_owned_rotation(job_path, desired_job, revalidate)
        self._delete_fenced(
            object_path,
            desired,
            "Service",
            uid=uid,
            resource_version=resource_version,
            revalidate=revalidate,
        )
        if self._create_revalidated(collection_path, desired, "Service", revalidate, ownership):
            return
        replacement, _uid, _resource_version = self._read_conflict(object_path, desired, "Service")
        revalidate("after Service replacement observation")
        if not self._service_matches(replacement, desired):
            raise self._retry_reconcile("Service", "the fenced replacement did not converge")

    def _ensure_secret(
        self,
        collection_path: str,
        object_path: str,
        job_path: str,
        desired: JsonObject,
        desired_job: JsonObject,
        revalidate: _Revalidate,
        ownership: _CreateOwnership,
    ) -> None:
        if self._create_revalidated(collection_path, desired, "Secret", revalidate, ownership):
            return
        existing, uid, resource_version = self._read_conflict(object_path, desired, "Secret")
        revalidate("after Secret conflict observation")
        if self._secret_matches(existing, desired):
            return
        if not self._stale_secret_token(existing, desired):
            raise self._conflict("Secret", "ownership, type, or data shape differs")
        # Never replace a mounted Secret under a running old Job: projected
        # volumes would hand the new credential to the old incarnation.
        self._remove_job_before_owned_rotation(job_path, desired_job, revalidate)
        self._delete_fenced(
            object_path,
            desired,
            "Secret",
            uid=uid,
            resource_version=resource_version,
            revalidate=revalidate,
        )
        if self._create_revalidated(collection_path, desired, "Secret", revalidate, ownership):
            return
        replacement, _uid, _resource_version = self._read_conflict(object_path, desired, "Secret")
        revalidate("after Secret replacement observation")
        if not self._secret_matches(replacement, desired):
            raise self._retry_reconcile("Secret", "the fenced replacement did not converge")

    def _ensure_job(
        self,
        collection_path: str,
        object_path: str,
        desired: JsonObject,
        revalidate: _Revalidate,
        ownership: _CreateOwnership,
    ) -> None:
        if self._create_revalidated(collection_path, desired, "Job", revalidate, ownership):
            return
        existing, uid, resource_version = self._read_conflict(object_path, desired, "Job")
        revalidate("after Job conflict observation")
        disposition = self._job_disposition(existing, desired)
        if disposition == "exact":
            return
        if disposition != "stale":
            raise self._conflict("Job", "cell, generation, fence, or security-critical spec differs")
        self._delete_fenced(
            object_path,
            desired,
            "Job",
            uid=uid,
            resource_version=resource_version,
            revalidate=revalidate,
        )
        if self._create_revalidated(collection_path, desired, "Job", revalidate, ownership):
            return
        replacement, _uid, _resource_version = self._read_conflict(object_path, desired, "Job")
        revalidate("after Job replacement observation")
        if not self._job_matches(replacement, desired):
            raise self._retry_reconcile("Job", "the fenced replacement did not converge")

    def apply_capacity(
        self,
        cell_id: str,
        desired: tp.Mapping[str, JsonValue],
        context: MutationContext,
    ) -> CellObservation:
        """Create this cell's fenced Service and Secret before its Job.

        Order matters. The Service must exist before the pods so their DNS names
        resolve from the first moment, and the Secret must exist before the Job
        so a pod does not start, fail to mount, and burn the ``backoffLimit: 0``
        budget on a race. Rotating either dependency first removes the exactly
        owned predecessor Job, so old workers cannot observe a newer fence or
        credential.

        Deterministic names make a retry find the same objects, but do not
        prove ownership. Every 409 is followed by an exact GET and comparison of
        Eyvan ownership plus the complete security-critical manifest. A stale
        credential is rotated only after its old Job is fenced and gone. The
        driver lock and renewable fleet lease cover the durable write and every
        provider mutation, so release cannot pass an apply that is still capable
        of creating resources.
        """

        accelerator_type, capacity_class, worker_count, options = self._validate_desired(desired)
        try:
            with self._mutation_lock, self.state.exclusive():
                captured: dict[str, object] = {}

                def mutate(record: FleetCellRecord) -> None:
                    self._require_no_inflight_creates(record, "apply")
                    if not self._accept_mutation(record, context):
                        # The durable stamp can commit before any Kubernetes POST.
                        # Replays must resume reconciliation without bumping the
                        # provider generation or rotating the worker token again.
                        captured["generation"] = record.provider_generation
                        return
                    record.accelerator_type = accelerator_type
                    record.capacity_class = capacity_class
                    record.desired_state = "up"
                    record.state = CapacityStatus.PROVISIONING
                    # Bumped on every apply that is not a replay, and the Secret is
                    # rewritten below -- ``derive_native_worker_token`` hashes
                    # (management_uid, provider_generation), so a stale Secret would
                    # authenticate workers of an incarnation this one replaced.
                    record.provider_generation = record.provider_generation + 1
                    record.metadata["capacity_backend"] = self.provider_name
                    record.metadata["worker_count"] = worker_count
                    record.metadata["namespace"] = self.namespace
                    # Namespace included so one cell_id in two clusters cannot
                    # collide on a derived token.
                    record.metadata["management_uid"] = stable_id("kubernetes", self.namespace, cell_id)
                    self._stamp(record, context)
                    captured["generation"] = record.provider_generation

                try:
                    record = self.state.mutate_or_create_record(cell_id, mutate, generation=context.generation)
                except StaleFenceError as exc:
                    raise StaleMutationError(str(exc)) from exc
                generation = int(captured.get("generation", record.provider_generation))

                def revalidate(phase: str) -> None:
                    self._revalidate_record_owner(
                        cell_id,
                        generation=context.generation,
                        fencing_token=context.fencing_token,
                        provider_generation=generation,
                        phase=phase,
                        desired_state="up",
                    )

                ownership = _CreateOwnership(
                    cell_id=cell_id,
                    generation=context.generation,
                    fencing_token=context.fencing_token,
                    provider_generation=generation,
                )
                namespace = self.namespace
                queue_name = self._optional_option(options, "queue_name")
                revalidate("before ClusterQueue validation")
                self._require_safe_queue(queue_name, namespace)
                revalidate("after ClusterQueue validation")

                service = headless_service_manifest(
                    cell_id,
                    namespace,
                    provider_generation=generation,
                    fence=fence_fingerprint(context.fencing_token),
                )
                service_name = object_name("eyvan", cell_id)
                service_collection = f"/api/v1/namespaces/{namespace}/services"
                service_path = f"{service_collection}/{service_name}"

                token = self._worker_token(record, generation)
                secret: JsonObject | None = None
                secret_name: str | None = None
                secret_collection = f"/api/v1/namespaces/{namespace}/secrets"
                secret_path = f"{secret_collection}/{object_name('eyvan-token', cell_id)}"
                if token is not None:
                    secret = secret_manifest(cell_id, namespace, token=token)
                    secret_metadata = secret.get("metadata")
                    assert is_json_object(secret_metadata)
                    labels = secret_metadata.get("labels")
                    assert is_json_object(labels)
                    labels[GENERATION_LABEL] = str(generation)
                    annotations = secret_metadata.get("annotations")
                    assert is_json_object(annotations)
                    annotations[FENCE_ANNOTATION] = fence_fingerprint(context.fencing_token)
                    secret_name = str(secret_metadata["name"])

                job = job_manifest(
                    cell_id=cell_id,
                    namespace=namespace,
                    image=str(options.get("image", self.image)),
                    worker_count=worker_count,
                    provider_generation=generation,
                    fencing_token=context.fencing_token,
                    resources=self._resources(options),
                    secret_name=secret_name,
                    queue_name=queue_name,
                    priority_class=self._optional_option(options, "priority_class"),
                    node_selector=self._mapping_option(options, "node_selector"),
                    tolerations=self._sequence_option(options, "tolerations"),
                )
                job_collection = f"/apis/batch/v1/namespaces/{namespace}/jobs"
                job_path = f"{job_collection}/{indexed_job_name(cell_id)}"

                self._ensure_service(
                    service_collection,
                    service_path,
                    job_path,
                    service,
                    job,
                    revalidate,
                    ownership,
                )
                if secret is not None:
                    self._ensure_secret(
                        secret_collection,
                        secret_path,
                        job_path,
                        secret,
                        job,
                        revalidate,
                        ownership,
                    )
                self._ensure_job(job_collection, job_path, job, revalidate, ownership)
                revalidate("before final observation")
                observation = self.observe(cell_id)
                revalidate("after final observation")
                return observation
        except StaleFenceError as exc:
            raise StaleMutationError(f"cell {cell_id!r} lost its fleet mutation lease during apply") from exc

    def _require_safe_queue(self, queue_name: str | None, namespace: str) -> None:
        """Refuse a ClusterQueue that could evict Eyvan's work.

        Kueue eviction destroys a pod with no checkpoint hook, bypassing the
        whole PreemptionPlan / CHECKPOINT_AND_STOP / grace_period_seconds
        protocol, and the only trace is a training job that lost hours with no
        ledger entry explaining why.

        The check runs here rather than in ``eyvan doctor`` because this is where
        cluster access exists: the driver runs in-cluster with a ServiceAccount,
        while doctor runs on an operator's laptop and can only warn. The
        precondition functions were written and never called from anywhere, and
        two docstrings claimed doctor refused a configuration it merely warned
        about.

        A queue Eyvan cannot read is not treated as safe -- an unreadable
        ClusterQueue is exactly the case where assuming the best silently loses
        the guarantee.

        ``queue_name`` is a *LocalQueue*: it is what ``job_manifest`` writes to
        ``kueue.x-k8s.io/queue-name``, which is the only name a Job may carry.
        This check read it as a ClusterQueue name, so unless a deployment
        happened to give both objects one name the read 404'd and every create
        carrying a queue was refused -- the safety check made the supported
        configuration unusable. The LocalQueue is resolved to the ClusterQueue it
        routes into first, and a LocalQueue that cannot be resolved is refused
        for the same reason an unreadable ClusterQueue is: nothing was
        established, so nothing may be assumed.
        """

        if not queue_name:
            return
        resolved = local_queue_cluster_queue(self.api.observe_local_queue(namespace, queue_name))
        if resolved is None:
            raise CapacityProviderError(
                f"could not resolve LocalQueue {queue_name!r} in namespace {namespace!r} to a ClusterQueue "
                "to verify it cannot evict Eyvan's work",
                code=CapacityProviderFailureCode.CONFIGURATION,
            )
        queue_name = resolved
        response = self.api.request("GET", f"{_KUEUE_API}/clusterqueues/{queue_name}")
        if response.status_code == 404:
            payload = None
        elif response.status_code >= 400:
            raise CapacityProviderError(
                f"could not read ClusterQueue {queue_name!r} to verify it cannot evict Eyvan's work "
                f"(HTTP {response.status_code})",
                code=CapacityProviderFailureCode.CONFIGURATION,
            )
        else:
            decoded = response.json()
            payload = decoded if is_json_object(decoded) else None
        findings = inspect_cluster_queue(payload, queue_name)
        if findings:
            detail = "; ".join(f"{finding.problem} -- {finding.remedy}" for finding in findings)
            raise CapacityProviderError(
                f"ClusterQueue {queue_name!r} is unsafe for Eyvan: {detail}",
                code=CapacityProviderFailureCode.CONFIGURATION,
            )

    def _worker_token(self, record: FleetCellRecord, generation: int) -> str | None:
        """Derive this incarnation's native-worker token, or ``None`` if unconfigured.

        ``None`` is a real answer, not a failure: a cluster with no shared secret
        runs unauthenticated workers, which the agent already supports. Inventing
        a token here would produce one no worker could ever verify.
        """

        from eyvan.worker.runtime.native_bootstrap import derive_native_worker_token

        return derive_native_worker_token(
            self.shared_secret,
            str(record.metadata.get("management_uid", record.name)),
            provider_generation=generation,
        )

    @staticmethod
    def _resources(options: tp.Mapping[str, JsonValue]) -> dict[str, str]:
        raw = options.get("resources")
        if not is_json_object(raw):
            return {}
        return {str(name): str(value) for name, value in raw.items()}

    @staticmethod
    def _optional_option(options: tp.Mapping[str, JsonValue], name: str) -> str | None:
        value = options.get(name)
        return value.strip() if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _mapping_option(options: tp.Mapping[str, JsonValue], name: str) -> dict[str, str]:
        raw = options.get(name)
        if not is_json_object(raw):
            return {}
        return {str(key): str(value) for key, value in raw.items()}

    @staticmethod
    def _sequence_option(options: tp.Mapping[str, JsonValue], name: str) -> tuple[JsonObject, ...]:
        raw = options.get(name)
        if not isinstance(raw, list):
            return ()
        return tuple(item for item in raw if is_json_object(item))

    def reconcile(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Re-assert ownership and re-observe. Creates nothing new."""

        with self._mutation_lock, self.state.exclusive():

            def mutate(record: FleetCellRecord) -> None:
                self._require_no_inflight_creates(record, "reconcile")
                if not self._accept_mutation(record, context):
                    return
                record.desired_state = "up"
                self._stamp(record, context)

            try:
                self.state.mutate_record(cell_id, mutate, generation=context.generation)
            except FleetCellNotRegisteredError:
                return self.observe(cell_id)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
        return self.observe(cell_id)

    def drain(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Stop scheduling onto this cell while leaving its Job running.

        Draining does not delete: workloads already leased on this cell keep
        their pods until they finish, which is exactly what distinguishes a drain
        from a release.
        """

        with self._mutation_lock, self.state.exclusive():

            def mutate(record: FleetCellRecord) -> None:
                # Draining takes capacity out of service, so a superseded
                # controller must not be able to do it -- the same fence check
                # the release tombstone gets, applied before anything moves.
                if not self._accept_mutation(record, context):
                    return
                self._require_no_inflight_creates(record, "drain")
                record.desired_state = "down"
                record.state = CapacityStatus.DRAINING
                self._stamp(record, context)

            try:
                record = self.state.drain_record(cell_id, context.generation, context.fencing_token, mutate)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
        if record is None:
            return CellObservation(
                cell_id=cell_id,
                state=CapacityStatus.DRAINING,
                generation=context.generation,
                observed_at=datetime.now(UTC),
            )
        return self._observation(record, CapacityStatus.DRAINING, "draining", worker_count=0, ready=0)

    @staticmethod
    def _released(cell_id: str, generation: int) -> CellObservation:
        return CellObservation(
            cell_id=cell_id,
            state=CapacityStatus.RELEASED,
            generation=generation,
            observed_at=datetime.now(UTC),
        )

    @staticmethod
    def _cleanup_fingerprint(value: object) -> str | None:
        if isinstance(value, str) and len(value) == 32 and all(character in "0123456789abcdef" for character in value):
            return value
        return None

    def _tombstone_cleanup(
        self,
        cell_id: str,
        context: MutationContext,
        tombstone: JsonObject | None,
    ) -> tuple[int, str] | None:
        """Validate a release replay and return predecessor Kubernetes ownership."""

        if tombstone is None:
            return None
        tombstone_generation = tombstone.get("generation")
        tombstone_token = tombstone.get("fencing_token")
        if isinstance(tombstone_generation, bool) or not isinstance(tombstone_generation, int):
            raise self._conflict("release", "the durable tombstone generation is malformed")
        if context.generation != tombstone_generation:
            relation = "older than" if context.generation < tombstone_generation else "newer than"
            raise StaleMutationError(
                f"generation {context.generation} is {relation} released cell {cell_id!r} generation "
                f"{tombstone_generation}; an exact release replay is required without a live record"
            )
        if tombstone_token != context.fencing_token:
            raise StaleMutationError(
                f"fencing token does not own released cell {cell_id!r} generation {tombstone_generation}"
            )
        cleanup = tombstone.get(_TOMBSTONE_CLEANUP_KEY)
        if cleanup is None:
            return None
        if not is_json_object(cleanup):
            raise self._conflict("release", "the durable Kubernetes cleanup identity is malformed")
        namespace = cleanup.get("namespace")
        provider_generation = cleanup.get("provider_generation")
        fingerprint = self._cleanup_fingerprint(cleanup.get("fence_fingerprint"))
        if (
            namespace != self.namespace
            or isinstance(provider_generation, bool)
            or not isinstance(provider_generation, int)
            or provider_generation < 0
            or fingerprint is None
        ):
            raise self._conflict("release", "the durable Kubernetes cleanup identity is incomplete or out of scope")
        return provider_generation, fingerprint

    def _revalidate_tombstone_owner(
        self,
        cell_id: str,
        context: MutationContext,
        cleanup_identity: tuple[int, str],
        phase: str,
    ) -> None:
        try:
            self.state.require_lease()
            record = self.state.get(cell_id)
            tombstone = self.state.cell_tombstone(cell_id)
            current = self._tombstone_cleanup(cell_id, context, tombstone)
            self.state.require_lease()
        except StaleFenceError as exc:
            raise StaleMutationError(f"cell {cell_id!r} lost its fleet mutation lease {phase}") from exc
        if record is not None or current != cleanup_identity:
            raise StaleMutationError(f"cell {cell_id!r} changed durable release ownership {phase}")

    def _reconcile_release_resources(
        self,
        cell_id: str,
        *,
        provider_generation: int,
        fencing_fingerprint: str,
        revalidate: _Revalidate,
    ) -> None:
        """Preflight every exact predecessor object, then UID/RV-fenced delete."""

        namespace = self.namespace
        resources = (
            (
                "Job",
                f"/apis/batch/v1/namespaces/{namespace}/jobs/{indexed_job_name(cell_id)}",
            ),
            (
                "Secret",
                f"/api/v1/namespaces/{namespace}/secrets/{object_name('eyvan-token', cell_id)}",
            ),
            (
                "Service",
                f"/api/v1/namespaces/{namespace}/services/{object_name('eyvan', cell_id)}",
            ),
        )
        preflight: list[tuple[str, str, JsonObject, tuple[str, str] | None]] = []
        for kind, path in resources:
            revalidate(f"before {kind} release preflight")
            desired = self._release_ownership_manifest(
                kind,
                cell_id,
                namespace,
                provider_generation=provider_generation,
                fencing_fingerprint=fencing_fingerprint,
            )
            identity = self._read_release_resource(path, desired, kind)
            revalidate(f"after {kind} release preflight")
            preflight.append((kind, path, desired, identity))

        for kind, path, desired, identity in preflight:
            if identity is None:
                continue
            uid, resource_version = identity
            self._delete_fenced(
                path,
                desired,
                kind,
                uid=uid,
                resource_version=resource_version,
                revalidate=revalidate,
            )

    def release(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Fenced-delete the Job, Secret, and Service, then tombstone the cell.

        Every object is exact-GET and ownership-checked before the first DELETE,
        so a stale mutation or same-name collision has no provider side effect.
        An inflight create intent makes release retry instead of trusting an
        absent GET while an older POST may still reach storage. The tombstone
        retains the predecessor generation fingerprint so a record-absent replay
        can conservatively remove an exact late object without touching a newer
        generation.
        """

        try:
            with self._mutation_lock, self.state.exclusive():
                record = self.state.get(cell_id)
                if record is None:
                    cleanup_identity = self._tombstone_cleanup(
                        cell_id,
                        context,
                        self.state.cell_tombstone(cell_id),
                    )
                    if cleanup_identity is not None:

                        def revalidate_tombstone(phase: str) -> None:
                            self._revalidate_tombstone_owner(cell_id, context, cleanup_identity, phase)

                        provider_generation, resource_fingerprint = cleanup_identity
                        self._reconcile_release_resources(
                            cell_id,
                            provider_generation=provider_generation,
                            fencing_fingerprint=resource_fingerprint,
                            revalidate=revalidate_tombstone,
                        )
                        revalidate_tombstone("after release replay reconciliation")
                    self.state.require_lease()
                    return self._released(cell_id, context.generation)

                captured: dict[str, object] = {}

                def validate_owner(current: FleetCellRecord) -> None:
                    self._accept_mutation(current, context)  # raises before any Kubernetes call when stale
                    if not isinstance(current.fencing_token, str) or not current.fencing_token:
                        raise StaleMutationError(f"cell {cell_id!r} has no durable Kubernetes ownership fence")
                    self._require_no_inflight_creates(current, "release")
                    captured["generation"] = current.generation
                    captured["provider_generation"] = current.provider_generation
                    captured["fencing_token"] = current.fencing_token

                try:
                    self.state.mutate_record(cell_id, validate_owner, generation=context.generation)
                except FleetCellNotRegisteredError:
                    # The lease should exclude this transition, but treating it as
                    # released without tombstone ownership would reintroduce the
                    # record-absent no-GET hole.
                    raise StaleMutationError(
                        f"cell {cell_id!r} disappeared before Kubernetes release preflight"
                    ) from None
                except StaleFenceError as exc:
                    raise StaleMutationError(str(exc)) from exc

                durable_generation = int(captured["generation"])
                provider_generation = int(captured["provider_generation"])
                resource_fencing_token = str(captured["fencing_token"])
                resource_fingerprint = fence_fingerprint(resource_fencing_token)

                def revalidate(phase: str) -> None:
                    self._revalidate_record_owner(
                        cell_id,
                        generation=durable_generation,
                        fencing_token=resource_fencing_token,
                        provider_generation=provider_generation,
                        phase=phase,
                    )

                self._reconcile_release_resources(
                    cell_id,
                    provider_generation=provider_generation,
                    fencing_fingerprint=resource_fingerprint,
                    revalidate=revalidate,
                )
                revalidate("before durable release tombstone")

                def owned(current: FleetCellRecord) -> bool:
                    self._accept_mutation(current, context)
                    self._stamp(current, context)
                    if current.generation != context.generation or current.fencing_token != context.fencing_token:
                        raise StaleMutationError(f"cell {cell_id!r} changed ownership before kubernetes release")
                    return True

                cleanup_metadata: JsonObject = {
                    _TOMBSTONE_CLEANUP_KEY: {
                        "namespace": self.namespace,
                        "provider_generation": provider_generation,
                        "fence_fingerprint": resource_fingerprint,
                    }
                }
                try:
                    self.state.tombstone_record(
                        cell_id,
                        context.generation,
                        context.fencing_token,
                        owned,
                        metadata=cleanup_metadata,
                    )
                except StaleFenceError as exc:
                    raise StaleMutationError(str(exc)) from exc

                cleanup_identity = (provider_generation, resource_fingerprint)

                def revalidate_tombstone(phase: str) -> None:
                    self._revalidate_tombstone_owner(cell_id, context, cleanup_identity, phase)

                # A second exact pass closes the durable-record transition and is
                # also the path used by every record-absent release replay.
                self._reconcile_release_resources(
                    cell_id,
                    provider_generation=provider_generation,
                    fencing_fingerprint=resource_fingerprint,
                    revalidate=revalidate_tombstone,
                )
                revalidate_tombstone("after durable release reconciliation")
                return self._released(cell_id, context.generation)
        except StaleFenceError as exc:
            raise StaleMutationError(f"cell {cell_id!r} lost its fleet mutation lease during release") from exc


__all__ = ("MAX_WORKERS", "KubernetesCapacityDriver")
