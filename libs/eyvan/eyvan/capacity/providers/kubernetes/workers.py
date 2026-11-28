# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Manifest builders for Kubernetes-backed Eyvan capacity cells.

**One cell is one ``batch/v1`` Job** with ``completionMode: Indexed``,
``parallelism == completions == worker_count``, and ``backoffLimit: 0``. Four
other mappings were considered and each fails on something structural:

- a *namespace* is not sized, so it cannot report an honest ``ResourceVector``;
- a Kueue ``LocalQueue`` is an admission channel that holds no capacity, and a
  cell must *hold* capacity;
- a pre-provisioned *node pool* is the right physical notion but is not portably
  creatable through the Kubernetes API, and is many failure units rather than one;
- a bare *Pod* needs per-pod group labels and gives no multi-host completion
  semantics.

The Job wins on four counts. A deterministic name from ``cell_id`` makes create
idempotent -- a 409 ``AlreadyExists`` is a replay, the exact analogue of
``FleetCellRecord.qr_id`` on GCP. ``worker_count -> parallelism`` reproduces the
shipped multislice model, so ``native_worker_endpoints`` has N entries and
``runtime/registry.py:resolve`` selects ``NativeSliceExecutor`` **unchanged**. An
Indexed Job plus a headless Service and ``pod.spec.subdomain`` gives stable DNS,
which is load-bearing: ``ExecutionDriverResolver`` caches drivers on the endpoint
fingerprint and closes stale entries, so pod-IP churn would thrash that cache and
could silently repoint an executor at a *different* pod. And release is a Job
delete with ``propagationPolicy: Foreground``, which genuinely frees capacity.

**The token never appears in a manifest.** Workers read it from a Secret, and the
Job carries only a *fingerprint* annotation for leak detection -- annotations are
readable by anyone with ``get`` in the namespace, so the token itself would be
readable too.
"""

from __future__ import annotations

import hashlib
import re
import typing as tp

from eyvan._internal.json_types import JsonObject

#: Kubernetes object names are DNS-1123 labels, at most 63 characters. Eyvan cell
#: ids can reach 96, so long names are truncated and disambiguated by a digest --
#: the same shape ``autoscale_fleet_name`` already uses, rather than a second
#: truncation rule that could collide differently.
MAX_OBJECT_NAME = 63

#: Indexed Jobs append ``-<completion_index>`` to the Job name to form each
#: Pod hostname. Reserve the largest supported suffix in the deterministic Job
#: name even for smaller cells: changing a Job's provider name when worker_count
#: crosses 10 or 100 would strand the previous incarnation during reconciliation.
MAX_INDEXED_JOB_WORKERS = 512
_MAX_INDEX_SUFFIX = f"-{MAX_INDEXED_JOB_WORKERS - 1}"
MAX_INDEXED_JOB_NAME = MAX_OBJECT_NAME - len(_MAX_INDEX_SUFFIX)

#: The only characters Kubernetes DNS-1123 names and Eyvan's cell selector
#: tokens may contain. Keep this explicit: :meth:`str.isalnum` accepts Unicode
#: letters such as ``é``, while the Kubernetes API admits lowercase ASCII only.
_DNS_1123_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")

#: NetworkPolicy grant required for a worker to redeem its token and exchange
#: managed artifacts, results, and progress with the control plane.
API_CLIENT_LABEL = "eyvan.dev/api-client"
#: Label carrying a Kubernetes-safe token for the Eyvan cell. The complete
#: cell ID is retained separately as an annotation because label values are
#: limited to 63 characters while valid Eyvan cell IDs can reach 96.
CELL_LABEL = "eyvan.dev/cell-id"
CELL_ID_ANNOTATION = "eyvan.dev/full-cell-id"
GENERATION_LABEL = "eyvan.dev/provider-generation"
#: Annotation carrying a *fingerprint* of the fencing token, never the token. An
#: observed Job whose fingerprint disagrees with the durable record is a leaked
#: previous incarnation -- the role ``management_uid`` plays on GCP.
FENCE_ANNOTATION = "eyvan.dev/fence-fingerprint"
_IMMUTABLE_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


def _is_dns_1123_label(value: str) -> bool:
    """Return whether ``value`` is one lowercase-ASCII DNS-1123 label."""

    return (
        0 < len(value) <= MAX_OBJECT_NAME
        and value[0] != "-"
        and value[-1] != "-"
        and all(character in _DNS_1123_CHARACTERS for character in value)
    )


def cell_label_value(cell_id: str) -> str:
    """Return a deterministic lowercase-ASCII selector token for a cell ID.

    Existing short DNS-1123 IDs retain their readable spelling. Long or
    otherwise invalid values use 232 bits of SHA-256 under a fixed prefix, which
    is both DNS-safe and collision-resistant. The unhashed ID remains
    authoritative in :data:`CELL_ID_ANNOTATION`.
    """

    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("Kubernetes capacity cell_id must be a non-empty string")
    if _is_dns_1123_label(cell_id):
        return cell_id
    return f"cell-{hashlib.sha256(cell_id.encode()).hexdigest()[:58]}"


def require_immutable_worker_image(image: str) -> str:
    """Return a qualified worker image or reject mutable tags and malformed refs."""

    if not isinstance(image, str) or _IMMUTABLE_IMAGE.fullmatch(image) is None:
        raise ValueError("Kubernetes native worker image must be repository@sha256:<64 lowercase hex characters>")
    return image


def object_name(prefix: str, cell_id: str, *, max_length: int = MAX_OBJECT_NAME) -> str:
    """Return a collision-resistant DNS-1123 name for one cell's object.

    Deterministic: the same cell always produces the same name, which is what
    makes ``create`` idempotent and a 409 a replay rather than a conflict. A
    digest is added whenever case-folding, replacement, or truncation is needed;
    otherwise distinct public cell IDs such as ``cell/a`` and ``cell_a`` would
    collapse onto the same Kubernetes object. ``max_length`` reserves controller-
    generated suffixes such as an Indexed Job's completion index.
    """

    if not isinstance(prefix, str) or not _is_dns_1123_label(prefix):
        raise ValueError("Kubernetes capacity object prefix must be a lowercase ASCII DNS-1123 label")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or not 1 <= max_length <= MAX_OBJECT_NAME:
        raise ValueError(f"Kubernetes capacity object max_length must be between 1 and {MAX_OBJECT_NAME}")
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("Kubernetes capacity cell_id must be a non-empty string")
    raw = f"{prefix}-{cell_id}"
    if len(raw) <= max_length and _is_dns_1123_label(raw):
        return raw
    base = "".join(character if character in _DNS_1123_CHARACTERS else "-" for character in raw.lower()).strip("-")
    suffix = f"-{hashlib.sha256(cell_id.encode()).hexdigest()[:8]}"
    if max_length <= len(suffix):
        raise ValueError("Kubernetes capacity object max_length is too short for its collision hash")
    stem = base[: max_length - len(suffix)].rstrip("-") or prefix[: max_length - len(suffix)].rstrip("-")
    if not stem:
        raise ValueError("Kubernetes capacity object max_length is too short for its prefix")
    return f"{stem}{suffix}"


def indexed_job_name(cell_id: str) -> str:
    """Return a stable Job name with room for every supported Pod index."""

    return object_name("eyvan", cell_id, max_length=MAX_INDEXED_JOB_NAME)


def _indexed_pod_hostname(job_name: str, completion_index: int) -> str:
    """Build and validate the hostname Kubernetes gives one Indexed Job Pod."""

    hostname = f"{job_name}-{completion_index}"
    if not _is_dns_1123_label(hostname):  # pragma: no cover - invariant guard
        raise ValueError(f"Indexed Job Pod hostname {hostname!r} is not a DNS-1123 label")
    return hostname


def fence_fingerprint(fencing_token: str) -> str:
    """Return a non-reversible fingerprint of a fencing token.

    Annotations are readable by anyone with ``get`` in the namespace, so the
    token cannot go there. A fingerprint is enough for the only question asked of
    it: does this observed Job belong to the incarnation the record describes?
    """

    return hashlib.sha256(fencing_token.encode()).hexdigest()[:32]


def headless_service_manifest(
    cell_id: str,
    namespace: str,
    *,
    provider_generation: int | None = None,
    fence: str | None = None,
) -> JsonObject:
    """Return the headless Service that gives Indexed Job pods stable DNS.

    ``clusterIP: None`` plus the pods' ``subdomain`` yields
    ``<job>-<index>.<service>.<namespace>.svc`` -- an address that survives a pod
    restart, which pod IPs do not. Provider reconciliation supplies generation
    and fence together so a stale post-check can distinguish its Service from a
    replacement before issuing a UID/resourceVersion-fenced cleanup.
    """

    if (provider_generation is None) != (fence is None):
        raise ValueError("a fenced Kubernetes Service requires both provider_generation and fence")
    labels = {CELL_LABEL: cell_label_value(cell_id)}
    annotations = {CELL_ID_ANNOTATION: cell_id}
    if provider_generation is not None and fence is not None:
        labels[GENERATION_LABEL] = str(provider_generation)
        annotations[FENCE_ANNOTATION] = fence
    name = object_name("eyvan", cell_id)
    return tp.cast(
        JsonObject,
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": labels,
                "annotations": annotations,
            },
            "spec": {
                "clusterIP": "None",
                # Published before readiness so DNS resolves while the workers are
                # still starting; the control plane decides readiness from Job status,
                # not from whether a name resolves.
                "publishNotReadyAddresses": True,
                "selector": {CELL_LABEL: cell_label_value(cell_id)},
                "ports": [{"name": "native-worker", "port": 8100, "targetPort": 8100}],
            },
        },
    )


def worker_endpoints(cell_id: str, namespace: str, worker_count: int, *, port: int = 8100) -> tuple[str, ...]:
    """Return the stable DNS endpoint of each worker, in index order.

    Index order is the rank order: ``NativeSliceExecutor`` derives per-rank
    checkpoint URIs from position, so a set that reordered between observations
    would silently repoint rank 1's checkpoint at rank 2's.
    """

    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
        raise ValueError("a Kubernetes capacity cell needs at least one worker")
    if worker_count > MAX_INDEXED_JOB_WORKERS:
        raise ValueError(f"a Kubernetes capacity cell supports at most {MAX_INDEXED_JOB_WORKERS} workers")
    job_name = indexed_job_name(cell_id)
    service_name = object_name("eyvan", cell_id)
    return tuple(
        f"http://{_indexed_pod_hostname(job_name, index)}.{service_name}.{namespace}.svc:{port}"
        for index in range(worker_count)
    )


def job_manifest(
    *,
    cell_id: str,
    namespace: str,
    image: str,
    worker_count: int,
    provider_generation: int,
    fencing_token: str,
    resources: tp.Mapping[str, str],
    service_account: str = "eyvan-native-worker",
    secret_name: str | None = None,
    queue_name: str | None = None,
    priority_class: str | None = None,
    node_selector: tp.Mapping[str, str] | None = None,
    tolerations: tp.Sequence[JsonObject] = (),
) -> JsonObject:
    """Return the Job manifest for one capacity cell.

    ``backoffLimit: 0`` because a cell is a *capacity* unit, not a retry unit:
    Eyvan's own retry budgets decide whether failed work runs again, and a Job
    that restarted itself would resurrect capacity the control plane believes it
    released.

    ``suspend`` is set only when a Kueue queue is named. Kueue is opt-in because
    it is not installed in most clusters, and requiring it would make this backend
    untestable for most users; absent it, kube-scheduler and the cluster
    autoscaler place the pods and ``provisioning_timeout_s`` is the only bound.
    """

    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
        raise ValueError("a Kubernetes capacity cell needs at least one worker")
    if worker_count > MAX_INDEXED_JOB_WORKERS:
        raise ValueError(f"a Kubernetes capacity cell supports at most {MAX_INDEXED_JOB_WORKERS} workers")
    image = require_immutable_worker_image(image)
    name = indexed_job_name(cell_id)
    service_name = object_name("eyvan", cell_id)
    _indexed_pod_hostname(name, worker_count - 1)
    labels = {
        API_CLIENT_LABEL: "true",
        CELL_LABEL: cell_label_value(cell_id),
        GENERATION_LABEL: str(provider_generation),
    }
    annotations = {
        CELL_ID_ANNOTATION: cell_id,
        FENCE_ANNOTATION: fence_fingerprint(fencing_token),
    }
    if queue_name:
        labels["kueue.x-k8s.io/queue-name"] = queue_name
    if priority_class:
        labels["kueue.x-k8s.io/priority-class"] = priority_class
    pod_spec: JsonObject = {
        "restartPolicy": "Never",
        "serviceAccountName": service_account,
        "automountServiceAccountToken": False,
        "subdomain": service_name,
        "containers": [
            {
                "name": "native-worker",
                "image": image,
                "imagePullPolicy": "IfNotPresent",
                "ports": [{"name": "native-worker", "containerPort": 8100}],
                "env": [
                    {"name": "EYVAN_CELL_ID", "value": cell_id},
                    {"name": "EYVAN_PROVIDER_GENERATION", "value": str(provider_generation)},
                    # Indexed Jobs expose the completion index here; it is the
                    # worker's rank and must match the endpoint ordering above.
                    {
                        "name": "EYVAN_WORKER_INDEX",
                        "valueFrom": {
                            "fieldRef": {"fieldPath": "metadata.annotations['batch.kubernetes.io/job-completion-index']"}
                        },
                    },
                ],
                "resources": {"limits": dict(resources), "requests": dict(resources)},
            }
        ],
    }
    if secret_name:
        # The token reaches the worker as a mounted Secret. Never as an env value
        # in this manifest and never as an annotation: both are readable by
        # anyone who can read the Job.
        pod_spec["volumes"] = [{"name": "eyvan-worker-token", "secret": {"secretName": secret_name}}]
        containers = pod_spec["containers"]
        assert isinstance(containers, list)
        container = containers[0]
        assert isinstance(container, dict)
        container["volumeMounts"] = [{"name": "eyvan-worker-token", "mountPath": "/var/run/eyvan", "readOnly": True}]
    if node_selector:
        pod_spec["nodeSelector"] = dict(node_selector)
    if tolerations:
        pod_spec["tolerations"] = [dict(toleration) for toleration in tolerations]
    manifest: JsonObject = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": {
            "completionMode": "Indexed",
            "completions": worker_count,
            "parallelism": worker_count,
            "backoffLimit": 0,
            **({"suspend": True} if queue_name else {}),
            "template": {"metadata": {"labels": labels, "annotations": {CELL_ID_ANNOTATION: cell_id}}, "spec": pod_spec},
        },
    }
    return manifest


def secret_manifest(cell_id: str, namespace: str, token: str) -> JsonObject:
    """Return the Secret carrying one cell's derived native-worker token.

    Rewritten on every recreate, because ``derive_native_worker_token`` hashes
    ``(management_uid, provider_generation)`` -- a stale Secret would authenticate
    workers of a previous incarnation, which is exactly what the generation
    exists to prevent.
    """

    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": object_name("eyvan-token", cell_id),
            "namespace": namespace,
            "labels": {CELL_LABEL: cell_label_value(cell_id)},
            "annotations": {CELL_ID_ANNOTATION: cell_id},
        },
        "type": "Opaque",
        "stringData": {"token": token},
    }


__all__ = (
    "API_CLIENT_LABEL",
    "CELL_ID_ANNOTATION",
    "CELL_LABEL",
    "FENCE_ANNOTATION",
    "GENERATION_LABEL",
    "MAX_INDEXED_JOB_NAME",
    "MAX_INDEXED_JOB_WORKERS",
    "MAX_OBJECT_NAME",
    "cell_label_value",
    "fence_fingerprint",
    "headless_service_manifest",
    "indexed_job_name",
    "job_manifest",
    "object_name",
    "require_immutable_worker_image",
    "secret_manifest",
    "worker_endpoints",
)
