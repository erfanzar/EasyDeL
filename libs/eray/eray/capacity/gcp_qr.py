# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
#
# Portions adapted from the Iris cluster manager in marin-community/marin
# (lib/iris/src/iris/cluster/platforms/gcp/service.py — Copyright The Marin
# Authors, Apache-2.0).
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

"""Typed GCP Queued Resource service (``google.cloud.tpu_v2alpha1``).

The API-client counterpart of :mod:`eray.provision.qr` (which shells out to
``gcloud``): same lifecycle states, but structured errors instead of stderr
grepping. :class:`QuotaExhaustedError` vs plain :class:`InfraError` is what
lets the pool loop distinguish "this zone is out of capacity, try the next
one" from "this request is broken, stop".

Differences from the Iris original, on purpose:

- Iris only creates *reserved* QRs (``guaranteed.reserved=True`` hardcoded);
  eray's fleet is spot-first, so :func:`apply_capacity_type` maps the full
  eray tier vocabulary onto the QR scheduling blocks.
- ``queued_resource_delete`` takes ``force`` (eray's watcher semantics)
  instead of always forcing.
- ``QueuedResourceInfo`` carries ``create_time``/``node_ids``/
  ``accelerator_type`` — the pool's zone-rotation timer needs the age.

Requires the ``gcp`` extra: ``pip install eray[gcp]``.
"""

from __future__ import annotations

import logging
import threading
import typing
from typing import Protocol

from .types import (
    KNOWN_GCP_ZONES,
    KNOWN_TPU_TYPES,
    CapacityType,
    InfraError,
    InfraUnavailableError,
    QueuedResourceInfo,
    QuotaExhaustedError,
    ResourceNotFoundError,
    TpuCreateRequest,
    parse_duration_s,
    validate_tpu_create,
)

if typing.TYPE_CHECKING:
    from google.cloud import tpu_v2alpha1

logger = logging.getLogger("eray.capacity.gcp_qr")


def _import_tpu_client():
    """Import ``google.cloud.tpu_v2alpha1``, with an actionable error.

    Returns:
        The imported ``tpu_v2alpha1`` module.

    Raises:
        InfraError: When the ``gcp`` extra is not installed.
    """
    try:
        from google.cloud import tpu_v2alpha1 as mod
    except ImportError as exc:
        raise InfraError(
            "google-cloud-tpu is required for eray.capacity cloud operations; install it with `pip install eray[gcp]`"
        ) from exc
    return mod


class GcpQrService(Protocol):
    """Service boundary for queued-resource operations.

    All methods raise :class:`InfraError` (or a subclass) on failure.
    Implementations: :class:`CloudGcpQrService` (real GCP) and
    :class:`eray.capacity.fake.FakeQrService` (in-memory, for tests and
    dry-runs).
    """

    @property
    def project_id(self) -> str:
        """The GCP project every operation is scoped to."""
        ...

    def queued_resource_create(self, request: TpuCreateRequest) -> None:
        """Create a queued resource. Raises QuotaExhaustedError on stockout/quota."""
        ...

    def queued_resource_describe(self, name: str, zone: str) -> QueuedResourceInfo | None:
        """Describe one queued resource, or None when it does not exist."""
        ...

    def queued_resource_delete(self, name: str, zone: str, *, force: bool = True) -> None:
        """Delete a queued resource (idempotent — missing QRs are ignored)."""
        ...

    def queued_resource_list(
        self, zones: list[str] | None = None, labels: dict[str, str] | None = None
    ) -> list[QueuedResourceInfo]:
        """List queued resources (all zones when ``zones`` is falsy), optionally label-filtered."""
        ...


def capacity_mode(capacity_type: CapacityType) -> str:
    """Map an eray capacity tier onto a QR scheduling block.

    Pure so the mapping is unit-testable without google-cloud-tpu installed.

    Args:
        capacity_type: The requested tier.

    Returns:
        One of ``"spot"`` (Spot block), ``"guaranteed_reserved"``
        (Guaranteed with ``reserved=True``), ``"guaranteed"`` (plain
        Guaranteed), or ``"none"`` (default on-demand queueing).

    Raises:
        ValueError: On an unknown tier.
    """
    mapping = {
        CapacityType.SPOT: "spot",
        CapacityType.RESERVED: "guaranteed_reserved",
        CapacityType.GUARANTEED: "guaranteed",
        CapacityType.ON_DEMAND: "none",
    }
    try:
        return mapping[CapacityType(capacity_type)]
    except (KeyError, ValueError):
        raise ValueError(f"unknown capacity tier {capacity_type!r}") from None


def build_queued_resource(request: TpuCreateRequest, project_id: str) -> tpu_v2alpha1.QueuedResource:
    """Build the ``QueuedResource`` proto for a create request.

    Args:
        request: The validated create request.
        project_id: GCP project id.

    Returns:
        The proto ready for ``create_queued_resource``.
    """
    tpu_v2alpha1 = _import_tpu_client()

    node = tpu_v2alpha1.Node(
        accelerator_type=request.accelerator_type,
        runtime_version=request.runtime_version,
        labels=dict(request.labels),
        metadata=dict(request.metadata),
        network_config=tpu_v2alpha1.NetworkConfig(
            enable_external_ips=request.enable_external_ip,
            network=request.network or "",
            subnetwork=request.subnetwork or "",
        ),
    )
    if request.service_account:
        node.service_account = tpu_v2alpha1.ServiceAccount(email=request.service_account)

    queued_resource = tpu_v2alpha1.QueuedResource(
        tpu=tpu_v2alpha1.QueuedResource.Tpu(
            node_spec=[
                tpu_v2alpha1.QueuedResource.Tpu.NodeSpec(
                    parent=f"projects/{project_id}/locations/{request.zone}",
                    node_id=request.resolved_node_id(),
                    node=node,
                )
            ]
        ),
    )

    mode = capacity_mode(request.capacity_type)
    if mode == "spot":
        queued_resource.spot = tpu_v2alpha1.QueuedResource.Spot()
    elif mode == "guaranteed_reserved":
        queued_resource.guaranteed = tpu_v2alpha1.QueuedResource.Guaranteed(reserved=True)
    elif mode == "guaranteed":
        queued_resource.guaranteed = tpu_v2alpha1.QueuedResource.Guaranteed()

    if request.valid_until_duration:
        import google.protobuf.duration_pb2

        seconds = int(parse_duration_s(request.valid_until_duration))
        queued_resource.queueing_policy = tpu_v2alpha1.QueuedResource.QueueingPolicy(
            valid_until_duration=google.protobuf.duration_pb2.Duration(seconds=seconds)
        )

    return queued_resource


def _labels_match(resource_labels: dict[str, str], required: dict[str, str]) -> bool:
    """True when every required label is present with the required value."""
    return all(resource_labels.get(k) == v for k, v in required.items())


def _zone_from_qr_name(full_name: str, fallback: str) -> str:
    """Extract the zone from a full QR resource name.

    Args:
        full_name: ``projects/<p>/locations/<zone>/queuedResources/<id>``.
        fallback: Zone to return when the name has no ``locations`` segment
            (e.g. a wildcard list result that was already resolved).

    Returns:
        The zone string.
    """
    parts = full_name.split("/")
    for i, part in enumerate(parts):
        if part == "locations" and i + 1 < len(parts):
            return parts[i + 1]
    return fallback


class CloudGcpQrService:
    """:class:`GcpQrService` backed by the typed ``tpu_v2alpha1`` client.

    Credentials come from Application Default Credentials — the attached
    service account on a GCE/TPU VM, or ``gcloud auth application-default
    login`` (optionally with ``--impersonate-service-account``) elsewhere.
    No key files.
    """

    def __init__(self, project_id: str) -> None:
        """Initialize the service.

        Args:
            project_id: GCP project id all operations are scoped to.
        """
        self._project_id = project_id
        self._client_cached: tpu_v2alpha1.TpuClient | None = None
        self._client_lock = threading.Lock()
        #: Mutable copies so callers can admit zones/types GCP adds later.
        self.valid_zones: set[str] = set(KNOWN_GCP_ZONES)
        self.valid_accelerator_types: set[str] = set(KNOWN_TPU_TYPES)

    @property
    def project_id(self) -> str:
        """The GCP project id."""
        return self._project_id

    @property
    def _client(self) -> tpu_v2alpha1.TpuClient:
        # Double-checked under a lock (ported from iris): concurrent
        # reconcile fan-outs would otherwise build (and leak the channel of)
        # a redundant TpuClient per thread.
        if self._client_cached is None:
            with self._client_lock:
                if self._client_cached is None:
                    self._client_cached = _import_tpu_client().TpuClient()
        return self._client_cached

    def _qr_api_error(self, exc: Exception) -> InfraError:
        """Map google-cloud exceptions to the capacity error hierarchy.

        Args:
            exc: Any google-cloud/auth exception.

        Returns:
            The corresponding :class:`InfraError` subclass —
            :class:`InfraUnavailableError` for transient server/transport
            trouble so callers can treat it as "observe failed, act on
            nothing" rather than a real state change.
        """
        import google.api_core.exceptions

        if isinstance(exc, google.api_core.exceptions.NotFound):
            return ResourceNotFoundError(str(exc))
        if isinstance(exc, google.api_core.exceptions.ResourceExhausted):
            return QuotaExhaustedError(str(exc))
        if isinstance(
            exc,
            google.api_core.exceptions.ServiceUnavailable
            | google.api_core.exceptions.InternalServerError
            | google.api_core.exceptions.DeadlineExceeded
            | google.api_core.exceptions.RetryError,
        ):
            return InfraUnavailableError(str(exc))
        return InfraError(str(exc))

    def _call(self, fn):
        """Run a client operation with the full error mapping applied.

        Catches beyond ``GoogleAPICallError``: credential failures
        (``GoogleAuthError`` — e.g. no ADC at the lazy client build) and
        transport-level ``GoogleAPIError`` siblings must surface as
        :class:`InfraError`, not raw tracebacks that kill a watch daemon.
        """
        import google.api_core.exceptions
        import google.auth.exceptions

        try:
            return fn()
        except google.api_core.exceptions.NotFound:
            raise
        except (google.api_core.exceptions.GoogleAPIError, google.auth.exceptions.GoogleAuthError) as exc:
            raise self._qr_api_error(exc) from exc

    def _qr_full_name(self, name: str, zone: str) -> str:
        """Build the full resource name for a QR id in a zone."""
        return f"projects/{self._project_id}/locations/{zone}/queuedResources/{name}"

    def queued_resource_create(self, request: TpuCreateRequest) -> None:
        """Create a queued resource.

        Args:
            request: The create request; validated before any API call.

        Raises:
            QuotaExhaustedError: Zone stockout or quota exhaustion.
            InfraError: Any other API failure.
        """
        validate_tpu_create(request, self.valid_zones, self.valid_accelerator_types)
        queued_resource = build_queued_resource(request, self._project_id)
        parent = f"projects/{self._project_id}/locations/{request.zone}"
        logger.info(
            "Creating queued resource %s (type=%s, zone=%s, capacity=%s)",
            request.name,
            request.accelerator_type,
            request.zone,
            request.capacity_type,
        )
        self._call(
            lambda: self._client.create_queued_resource(
                parent=parent,
                queued_resource=queued_resource,
                queued_resource_id=request.name,
            )
        )

    def queued_resource_describe(self, name: str, zone: str) -> QueuedResourceInfo | None:
        """Describe one queued resource.

        Args:
            name: Queued-resource id.
            zone: GCP zone.

        Returns:
            The info, or None when the QR does not exist.

        Raises:
            InfraError: On non-NotFound API failures.
        """
        import google.api_core.exceptions

        try:
            qr = self._call(lambda: self._client.get_queued_resource(name=self._qr_full_name(name, zone)))
        except google.api_core.exceptions.NotFound:
            return None
        return self._parse_qr(qr, zone)

    def queued_resource_delete(self, name: str, zone: str, *, force: bool = True) -> None:
        """Delete a queued resource (fire-and-forget LRO, idempotent).

        Args:
            name: Queued-resource id.
            zone: GCP zone.
            force: Also delete a provisioned node under the QR (required for
                non-terminal states like ACTIVE).

        Raises:
            InfraError: On non-NotFound API failures.
        """
        import google.api_core.exceptions

        tpu_v2alpha1 = _import_tpu_client()
        logger.info("Deleting queued resource %s (zone=%s, force=%s)", name, zone, force)
        try:
            self._call(
                lambda: self._client.delete_queued_resource(
                    request=tpu_v2alpha1.DeleteQueuedResourceRequest(name=self._qr_full_name(name, zone), force=force)
                )
            )
        except google.api_core.exceptions.NotFound:
            pass

    def queued_resource_list(
        self, zones: list[str] | None = None, labels: dict[str, str] | None = None
    ) -> list[QueuedResourceInfo]:
        """List queued resources, optionally filtered by node labels.

        Args:
            zones: Zones to query; empty/None lists project-wide (the ``-``
                wildcard location, matching ``gcloud --zone=-``).
            labels: Node labels every returned QR must carry.

        Returns:
            Parsed infos (possibly empty).

        Raises:
            InfraUnavailableError: When EVERY queried location failed. A
                caller reconciling desired-vs-observed must be able to tell
                "own nothing" apart from "cannot see" — silently returning
                ``[]`` here would make an API outage look like an empty
                pool and trigger mass-creation. Partial failures on
                explicit multi-zone lists are logged and tolerated.
        """
        zone_list = list(zones) if zones else ["-"]
        results: list[QueuedResourceInfo] = []
        succeeded = 0
        last_error: InfraError | None = None
        for zone in zone_list:
            parent = f"projects/{self._project_id}/locations/{zone}"
            try:
                for qr in self._call(lambda p=parent: list(self._client.list_queued_resources(parent=p))):
                    info = self._parse_qr(qr, zone)
                    if labels and not _labels_match(info.labels or {}, labels):
                        continue
                    results.append(info)
                succeeded += 1
            except InfraError as exc:
                last_error = exc
                logger.warning("Failed to list queued resources in %s: %s", zone, exc)
                continue
        if succeeded == 0 and last_error is not None:
            raise InfraUnavailableError(f"queued-resource list failed for every location: {last_error}") from last_error
        return results

    def _parse_qr(self, qr, query_zone: str) -> QueuedResourceInfo:
        """Parse a ``QueuedResource`` proto into a :class:`QueuedResourceInfo`.

        Args:
            qr: The proto message.
            query_zone: Zone the query was issued against (``-`` resolves
                from the resource name).

        Returns:
            The parsed info.
        """
        short_name = qr.name.rsplit("/", 1)[-1] if qr.name else "?"
        state = qr.state.state.name if qr.state else "UNKNOWN"
        zone = _zone_from_qr_name(qr.name, query_zone) if query_zone == "-" else query_zone

        node_specs = list(qr.tpu.node_spec) if qr.tpu else []
        item_labels = dict(node_specs[0].node.labels) if node_specs and node_specs[0].node else {}
        node_ids = tuple(spec.node_id for spec in node_specs if spec.node_id)
        accelerator_type = None
        for spec in node_specs:
            if spec.node and spec.node.accelerator_type:
                accelerator_type = spec.node.accelerator_type
                break

        create_time: float | None = None
        if qr.create_time:
            epoch = qr.create_time.timestamp()
            if epoch > 0:
                create_time = epoch

        return QueuedResourceInfo(
            name=short_name,
            state=state,
            zone=zone,
            labels=item_labels,
            create_time=create_time,
            accelerator_type=accelerator_type,
            node_ids=node_ids,
        )
