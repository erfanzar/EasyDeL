# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Minimal in-cluster Kubernetes API access, and Job-status interpretation.

**No new dependency.** ``httpx`` is already core, so this talks to the API server
directly rather than pulling in a client library whose object model would then
have to be kept in sync with the manifests Eyvan actually sends.

**The bearer token is re-read on every call.** Projected service-account tokens
rotate -- typically hourly -- and a client that cached the file contents at
construction works for exactly one rotation period and then fails in a way that
looks like an unrelated permissions problem.

**A Job is never replaced.** Several fields of a Job spec are immutable, so a
``replace`` of a live Job returns 422 rather than converging. The lifecycle is
create-if-absent, then a single-field patch guarded by a ``resourceVersion``
precondition, then delete.

**Mutating reads raise; describing reads do not.** ``get``/``create``/``delete``
serve a cell mutation and must not proceed on a guess, so they raise. The
``observe_*`` family serves *description* of a cluster Eyvan does not own, where
the only safe answer to "I could not look" is that nothing was established --
never that the cluster is full and never that it is empty.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from pathlib import Path

import httpx

from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CapacityProviderError, CapacityProviderFailureCode
from eyvan.capacity.providers.kubernetes.workers import (
    CELL_ID_ANNOTATION,
    CELL_LABEL,
    FENCE_ANNOTATION,
    GENERATION_LABEL,
)

#: Longest provider reason kept on an unreadable observation. The whole point of
#: the value is to name a cause an operator can act on; a full API-server error
#: body would land in a cell detail, a log line, and a dashboard panel.
_MAX_REASON = 300

#: Where a projected service-account token is mounted in a pod by default.
DEFAULT_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
DEFAULT_CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"

#: Where Kueue publishes ClusterQueue and ResourceFlavor objects.
KUEUE_API = "/apis/kueue.x-k8s.io/v1beta1"

#: The label Kueue reads to route a Job into a queue, and the one
#: ``workers.job_manifest`` writes. It names a **LocalQueue**, never a
#: ClusterQueue: the two are different objects and only the LocalQueue is
#: namespaced. Reading it as a ClusterQueue name is the mistake that makes a
#: capacity observation describe a queue that does not exist.
QUEUE_NAME_LABEL = "kueue.x-k8s.io/queue-name"

#: Most nodes one capacity observation will read. A node list is only ever used
#: to describe capacity, never to change it, so a cluster larger than this is
#: reported as a partial reading rather than paged through: an observation that
#: costs the API server twenty round trips is an observation an operator turns
#: off.
MAX_OBSERVED_NODES = 500


@dataclass(frozen=True, slots=True)
class ApiObservation:
    """The result of one read whose only purpose is to describe the cluster.

    Separate from :meth:`KubernetesApi.get` on purpose. ``get`` raises, because
    its callers are mutating a cell and must not proceed on a guess. A capacity
    *observation* has the opposite requirement: a cluster Eyvan cannot read is
    an unknown cluster, and an exception there would either abort a
    reconciliation that has nothing to do with cluster capacity or -- worse --
    be caught somewhere that turns "I could not look" into "there is no room".

    Attributes:
        payload: The decoded object, or ``None`` when it does not exist or could
            not be read.
        readable: Whether the answer is authoritative. ``True`` with a ``None``
            payload means the object genuinely is not there; ``False`` means
            nothing was established.
        reason: Why the read established nothing. Empty when ``readable``.
    """

    payload: JsonObject | None
    readable: bool
    reason: str = ""

    @property
    def unreadable(self) -> bool:
        """Return whether this read failed to establish anything at all."""

        return not self.readable


@dataclass(frozen=True, slots=True)
class JobStatus:
    """What the control plane needs to know about one cell's Job."""

    exists: bool
    #: Pods that have reported Ready. Endpoints are published only when this
    #: equals ``parallelism``; publishing early makes the readiness probe resolve
    #: a nonexistent DNS name and the cell flaps HEALTHY <-> EXECUTION_CONNECTING.
    ready: int = 0
    active: int = 0
    failed: int = 0
    parallelism: int = 0
    suspended: bool = False
    #: Set when Kueue refused or evicted the workload, with its own wording kept
    #: so an operator can find it in Kueue's own logs.
    kueue_reason: str = ""
    #: The Job's ``kueue.x-k8s.io/queue-name`` label, which is the **LocalQueue**
    #: this cell was admitted through. Empty when the Job carries no label, which
    #: is every cell in a cluster without Kueue. The ClusterQueue behind it is one
    #: further read (:meth:`KubernetesApi.observe_local_queue`); without this
    #: field a driver observing a queued cell cannot name the queue at all, since
    #: the pool's ``queue_name`` is read only at create time.
    queue_label: str = ""
    #: The cell's per-pod resource request, as a Job container states it. A tuple
    #: of pairs rather than a mapping so the status stays hashable and immutable
    #: like every other field here. Empty when nothing readable was declared,
    #: which must be read as *unknown*, never as a request for nothing.
    resource_requests: tuple[tuple[str, str], ...] = ()
    api_version: str = ""
    kind: str = ""
    name: str = ""
    namespace: str = ""
    uid: str = ""
    cell_label: str = ""
    cell_id_annotation: str = ""
    generation_label: str = ""
    fence_fingerprint: str = ""
    resource_version: str = ""


def read_token(path: str = DEFAULT_TOKEN_PATH) -> str:
    """Return the current service-account bearer token.

    Raises:
        CapacityProviderError: If the token is unreadable, which is a
            configuration failure rather than a transient one -- a pod without a
            mounted token will never grow one.
    """

    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise CapacityProviderError(
            f"Kubernetes service-account token is unreadable at {path}: {exc}",
            code=CapacityProviderFailureCode.CONFIGURATION,
            retryable=False,
        ) from exc


def interpret_job(payload: JsonObject | None) -> JobStatus:
    """Map a Job resource to the fields the driver reasons about.

    ``None`` means the Job is absent, which is a legitimate observation rather
    than an error: it is what ``release`` produces and what ``apply`` starts from.
    """

    if payload is None:
        return JobStatus(exists=False)
    metadata = payload.get("metadata") if is_json_object(payload.get("metadata")) else {}
    spec = payload.get("spec") if is_json_object(payload.get("spec")) else {}
    status = payload.get("status") if is_json_object(payload.get("status")) else {}
    assert isinstance(metadata, dict) and isinstance(spec, dict) and isinstance(status, dict)
    labels = metadata.get("labels") if is_json_object(metadata.get("labels")) else {}
    annotations = metadata.get("annotations") if is_json_object(metadata.get("annotations")) else {}
    assert isinstance(labels, dict) and isinstance(annotations, dict)
    return JobStatus(
        exists=True,
        api_version=str(payload.get("apiVersion", "")),
        kind=str(payload.get("kind", "")),
        name=str(metadata.get("name", "")),
        namespace=str(metadata.get("namespace", "")),
        uid=str(metadata.get("uid", "")),
        cell_label=str(labels.get(CELL_LABEL, "")),
        cell_id_annotation=str(annotations.get(CELL_ID_ANNOTATION, "")),
        ready=_count(status.get("ready")),
        active=_count(status.get("active")),
        failed=_count(status.get("failed")),
        parallelism=_count(spec.get("parallelism")),
        suspended=spec.get("suspend") is True,
        kueue_reason=_kueue_reason(status),
        queue_label=str(labels.get(QUEUE_NAME_LABEL, "")),
        resource_requests=_resource_requests(spec),
        generation_label=str(labels.get(GENERATION_LABEL, "")),
        fence_fingerprint=str(annotations.get(FENCE_ANNOTATION, "")),
        resource_version=str(metadata.get("resourceVersion", "")),
    )


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _resource_requests(spec: tp.Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    """Return the first container's resource request from a Job spec.

    ``requests`` with a fallback to ``limits`` because Kubernetes defaults the
    former to the latter, and ``workers.job_manifest`` writes both.

    The *first* container only. An Eyvan cell has exactly one, and summing over
    a pod Eyvan did not build would need quantity arithmetic that lives in
    :mod:`eyvan.capacity.providers.kubernetes.kueue`. If a future sidecar ever
    appears the answer understates the pod's request, which can only make a
    cluster look roomier than it is -- never fuller, which is the direction that
    would let a capacity reading invent a shortage.

    Args:
        spec: A Job's ``spec``.

    Returns:
        ``(resource, quantity)`` pairs in sorted order, or empty when nothing
        readable was declared. Non-scalar values are dropped rather than
        stringified into a quantity nothing can parse.
    """

    template = spec.get("template") if is_json_object(spec.get("template")) else {}
    assert isinstance(template, dict)
    pod_spec = template.get("spec") if is_json_object(template.get("spec")) else {}
    assert isinstance(pod_spec, dict)
    containers = pod_spec.get("containers")
    if not isinstance(containers, list) or not containers:
        return ()
    container = containers[0]
    if not is_json_object(container):
        return ()
    resources = container.get("resources") if is_json_object(container.get("resources")) else {}
    assert isinstance(resources, dict)
    for key in ("requests", "limits"):
        declared = resources.get(key)
        if not is_json_object(declared):
            continue
        pairs = tuple(
            (str(name), str(value))
            for name, value in sorted(declared.items())
            if isinstance(value, (str, int, float)) and not isinstance(value, bool)
        )
        if pairs:
            return pairs
    return ()


def _kueue_reason(status: tp.Mapping[str, object]) -> str:
    """Return Kueue's own reason string when it refused or evicted the workload."""

    conditions = status.get("conditions")
    if not isinstance(conditions, list):
        return ""
    for condition in conditions:
        if not is_json_object(condition):
            continue
        reason = str(condition.get("reason", ""))
        if reason in {"Inadmissible", "Evicted", "Preempted"} and condition.get("status") == "True":
            return reason
    return ""


def local_queue_cluster_queue(observation: ApiObservation) -> str | None:
    """Return the ClusterQueue one LocalQueue routes into, or ``None``.

    ``None`` covers every way the answer was not established: the LocalQueue was
    unreadable, it does not exist, or it names no ``spec.clusterQueue``. All
    three must stay ``None`` rather than fall back to the LocalQueue's own name
    -- a capacity reading taken against the wrong object is worse than no
    reading, because it answers with numbers.

    Args:
        observation: The result of :meth:`KubernetesApi.observe_local_queue`.

    Returns:
        The ClusterQueue name, or ``None`` when nothing was established.
    """

    if observation.unreadable or observation.payload is None:
        return None
    spec = observation.payload.get("spec")
    if not is_json_object(spec):
        return None
    cluster_queue = spec.get("clusterQueue")
    if not isinstance(cluster_queue, str) or not cluster_queue.strip():
        return None
    return cluster_queue.strip()


def capacity_state(status: JobStatus, *, provisioning: bool = True) -> tuple[CapacityStatus, str]:
    """Map one Job's status to an Eyvan capacity state and a human detail.

    A suspended Job is WAITING rather than PROVISIONING: it is queued behind
    Kueue's admission and nothing is being built yet, so ``queue_timeout_s``
    applies to it exactly as it does to a queued TPU resource.

    A failed Job with ``backoffLimit: 0`` is NEEDS_RECREATE, not UNAVAILABLE:
    nothing will retry it, so the only path forward is a new incarnation with a
    fresh generation.
    """

    del provisioning
    if not status.exists:
        return CapacityStatus.ABSENT, "job does not exist"
    if status.kueue_reason == "Inadmissible":
        return CapacityStatus.WAITING, "kueue reports the workload is inadmissible"
    if status.kueue_reason in {"Evicted", "Preempted"}:
        return CapacityStatus.UNAVAILABLE, f"kueue {status.kueue_reason.lower()} the workload"
    if status.suspended:
        return CapacityStatus.WAITING, "job is suspended pending kueue admission"
    if status.failed > 0:
        return CapacityStatus.NEEDS_RECREATE, f"{status.failed} pod(s) failed with backoffLimit 0"
    if status.parallelism and status.ready >= status.parallelism:
        return CapacityStatus.HEALTHY, f"{status.ready}/{status.parallelism} workers ready"
    return CapacityStatus.PROVISIONING, f"{status.ready}/{status.parallelism or '?'} workers ready"


class KubernetesCreateRejected(CapacityProviderError):
    """A completed POST response proves that Kubernetes refused the create.

    Generic configuration and transport errors do not establish this fact. Only
    the explicitly recognized authorization/validation refusals from ``create``
    use this signal; other errors retain indeterminate completion semantics.
    """


class KubernetesApi:
    """Thin request wrapper over the in-cluster Kubernetes API."""

    def __init__(
        self,
        base_url: str = "https://kubernetes.default.svc",
        *,
        token_path: str = DEFAULT_TOKEN_PATH,
        ca_path: str = DEFAULT_CA_PATH,
        client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.token_path = token_path
        self._client = client or httpx.Client(verify=ca_path if Path(ca_path).exists() else True, timeout=30.0)

    def _headers(self, extra: tp.Mapping[str, str] | None = None) -> dict[str, str]:
        return {"Authorization": f"Bearer {read_token(self.token_path)}", **(extra or {})}

    def request(
        self,
        method: str,
        path: str,
        *,
        json: JsonObject | None = None,
        params: tp.Mapping[str, str] | None = None,
        content_type: str | None = None,
    ) -> httpx.Response:
        """Send one API request, translating transport failures into driver errors."""

        headers = self._headers({"Content-Type": content_type} if content_type else None)
        try:
            return self._client.request(
                method, f"{self.base_url}{path}", json=json, params=dict(params or {}), headers=headers
            )
        except httpx.HTTPError as exc:
            # Retryable: an API server that is briefly unreachable is the ordinary
            # state of a cluster during an upgrade, and discarding the command
            # would drop capacity the control plane believes it asked for.
            raise CapacityProviderError(
                f"Kubernetes API request failed: {exc}",
                code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
                retryable=True,
            ) from exc

    def get(self, path: str) -> JsonObject | None:
        """Return one namespaced object, failing closed on a malformed response."""

        response = self.request("GET", path)
        if response.status_code == 404:
            return None
        payload = self._decode(response)
        if not is_json_object(payload):
            raise CapacityProviderError(
                "Kubernetes API returned a malformed object",
                code=CapacityProviderFailureCode.CONFIGURATION,
                retryable=False,
            )
        return payload

    def get_job(self, namespace: str, name: str) -> JsonObject | None:
        """Return one Job, or ``None`` when it does not exist."""

        return self.get(f"/apis/batch/v1/namespaces/{namespace}/jobs/{name}")

    def observe(self, path: str, *, params: tp.Mapping[str, str] | None = None) -> ApiObservation:
        """Read one object for description only, degrading to unreadable.

        Every failure mode collapses to the same answer -- *nothing was
        established* -- with the cause kept for the operator. That includes RBAC:
        a ServiceAccount without ``get clusterqueues`` is the ordinary state of a
        cluster where Eyvan was installed before Kueue, and it must degrade the
        capacity signal rather than fail an unrelated reconciliation or be
        mistaken for an empty queue.

        Args:
            path: Absolute API path, already namespaced where applicable.
            params: Optional query parameters, such as a label selector.

        Returns:
            An :class:`ApiObservation`. A 404 is a readable "it does not exist";
            anything else that went wrong is unreadable with a reason.
        """

        try:
            response = self.request("GET", path, params=params)
        except CapacityProviderError as exc:
            return ApiObservation(payload=None, readable=False, reason=str(exc)[:_MAX_REASON])
        if response.status_code == 404:
            return ApiObservation(payload=None, readable=True)
        if response.status_code >= 400:
            # Upstream error bodies can echo Secret values. Keep only the HTTP
            # status in operator-visible observations, never arbitrary body text.
            return ApiObservation(
                payload=None,
                readable=False,
                reason=f"HTTP {response.status_code}: Kubernetes API request failed",
            )
        try:
            payload = response.json()
        except ValueError:
            return ApiObservation(payload=None, readable=False, reason="response body is not JSON")
        if not is_json_object(payload):
            return ApiObservation(payload=None, readable=False, reason="response body is not a JSON object")
        return ApiObservation(payload=payload, readable=True)

    def observe_cluster_queue(self, name: str) -> ApiObservation:
        """Read one Kueue ClusterQueue for capacity description only."""

        return self.observe(f"{KUEUE_API}/clusterqueues/{name}")

    def observe_local_queue(self, namespace: str, name: str) -> ApiObservation:
        """Read one Kueue LocalQueue for capacity description only.

        The Job's ``kueue.x-k8s.io/queue-name`` label names a LocalQueue, and a
        LocalQueue is namespaced while the ClusterQueue holding the actual quota
        is not. This is the read that turns the one name a running cell carries
        into the one name the capacity signal can be taken against.

        Args:
            namespace: Namespace the LocalQueue lives in -- the cell's own.
            name: LocalQueue name, from the Job's queue label.

        Returns:
            An :class:`ApiObservation`. Same never-raises contract as every other
            ``observe_*``: a LocalQueue Eyvan cannot read establishes nothing,
            which is neither a full cluster nor an empty one.
        """

        return self.observe(f"{KUEUE_API}/namespaces/{namespace}/localqueues/{name}")

    def observe_resource_flavor(self, name: str) -> ApiObservation:
        """Read one Kueue ResourceFlavor for capacity description only."""

        return self.observe(f"{KUEUE_API}/resourceflavors/{name}")

    def observe_nodes(self, *, label_selector: str = "", limit: int = MAX_OBSERVED_NODES) -> ApiObservation:
        """Read a bounded node list for capacity description only.

        Args:
            label_selector: Kubernetes label selector, normally a ResourceFlavor's
                ``spec.nodeLabels`` rendered as ``key=value`` pairs. Empty reads
                every node.
            limit: Hard ceiling on nodes returned. The response carries
                ``metadata.continue`` when it was truncated, which the reader
                turns into a partial rather than a complete observation.

        Returns:
            An :class:`ApiObservation` holding a ``NodeList``.
        """

        params = {"limit": str(max(1, min(limit, MAX_OBSERVED_NODES)))}
        if label_selector:
            params["labelSelector"] = label_selector
        return self.observe("/api/v1/nodes", params=params)

    def create(self, path: str, manifest: JsonObject) -> bool:
        """Create one object, returning ``False`` for an unverified conflict.

        ``False`` is deliberately *not* success. Kubernetes names are only a
        routing key; a 409 may name a foreign or stale object. Callers must GET
        the exact object and verify its ownership and security-critical content
        before accepting it as an idempotent replay.
        """

        response = self.request("POST", path, json=manifest)
        if response.status_code == 409:
            return False
        try:
            self._decode(response)
        except CapacityProviderError as exc:
            # A completed authorization/validation refusal cannot store this
            # object later. Do not infer that from generic nonretryable errors,
            # timeouts, throttling or server failures, whose outcome is unknown.
            if response.status_code not in {403, 422}:
                raise
            raise KubernetesCreateRejected(str(exc), code=exc.code, retryable=exc.retryable) from exc
        return True

    def delete(
        self,
        path: str,
        *,
        uid: str | None = None,
        resource_version: str | None = None,
    ) -> None:
        """Delete one object with foreground propagation and optional fencing.

        Foreground so the Job's pods are gone before the Job is: a background
        delete returns while pods are still running. UID and resourceVersion
        preconditions prevent a stale reconciler from deleting a replacement
        that acquired the deterministic name after its GET.
        """

        body: JsonObject = {"propagationPolicy": "Foreground"}
        if uid is not None or resource_version is not None:
            if not uid or not resource_version:
                raise ValueError("fenced Kubernetes delete requires both uid and resource_version")
            body["preconditions"] = {"uid": uid, "resourceVersion": resource_version}
        response = self.request("DELETE", path, json=body)
        if response.status_code == 404:
            return
        if response.status_code == 409 and uid is not None:
            raise CapacityProviderError(
                "Kubernetes object changed before its fenced delete",
                code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
                retryable=True,
            )
        self._decode(response)

    @staticmethod
    def _decode(response: httpx.Response) -> object:
        if response.status_code >= 400:
            retryable = response.status_code >= 500 or response.status_code == 429
            # Admission webhooks may echo submitted credentials in any error
            # field. Body truncation or selecting a reason is not redaction.
            raise CapacityProviderError(
                f"Kubernetes API returned HTTP {response.status_code}",
                code=(
                    CapacityProviderFailureCode.PROVIDER_UNAVAILABLE
                    if retryable
                    else CapacityProviderFailureCode.CONFIGURATION
                ),
                retryable=retryable,
            )
        try:
            return response.json()
        except ValueError:
            return {}


__all__ = (
    "DEFAULT_CA_PATH",
    "DEFAULT_TOKEN_PATH",
    "KUEUE_API",
    "MAX_OBSERVED_NODES",
    "QUEUE_NAME_LABEL",
    "ApiObservation",
    "JobStatus",
    "KubernetesApi",
    "KubernetesCreateRejected",
    "capacity_state",
    "interpret_job",
    "local_queue_cluster_queue",
    "read_token",
)
