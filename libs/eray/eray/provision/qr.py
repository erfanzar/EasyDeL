# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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


"""TPU Queued Resource management via the gcloud CLI.

Queued Resources are GCP's recommended way to obtain TPU capacity — especially
spot capacity, where a request waits in `WAITING_FOR_RESOURCES` until chips
free up instead of failing immediately. This module wraps
``gcloud compute tpus queued-resources`` as plain functions (subprocess, the
eray convention — no google-api client dependencies) and is the provisioning
primitive under ``eray qr``, ``eray fleet``, and the spot watcher.

QR lifecycle states (observed via ``describe``):
``ACCEPTED/CREATING → WAITING_FOR_RESOURCES → PROVISIONING → ACTIVE`` on the
happy path; ``SUSPENDING → SUSPENDED`` when capacity is torn down (spot
preemption or deletion of the underlying node); ``FAILED`` when provisioning
cannot proceed (quota, invalid request, capacity errors).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Literal

from ..gcp import gcloud, gcloud_json


def _gcloud_or_raise(args: list[str]) -> str:
    """Run gcloud, converting failures into readable RuntimeErrors.

    subprocess.CalledProcessError's str() omits stderr, which is where gcloud
    puts the actual reason (quota, permission, already-exists). Surface it.

    Args:
        args: gcloud argument list.

    Returns:
        gcloud stdout.

    Raises:
        RuntimeError: With gcloud's stderr on non-zero exit.
    """
    import subprocess

    try:
        return gcloud(args)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"gcloud {' '.join(args[:4])} failed: {detail or exc}") from None


CapacityT = Literal["spot", "on-demand", "reserved", "guaranteed"]

RUNTIME_VERSION_BY_FAMILY: dict[str, str] = {
    "v4": "tpu-ubuntu2204-base",
    "v5e": "v2-alpha-tpuv5-lite",
    "v5litepod": "v2-alpha-tpuv5-lite",
    "v5p": "v2-alpha-tpuv5",
    "v6e": "v2-alpha-tpuv6e",
}
"""Default TPU VM runtime version per generation (mirrors the values the
repo's cluster-launcher generator has provisioned with)."""

ERAY_QR_POLL_S = int(os.getenv("ERAY_QR_POLL", "30"))
ERAY_QR_WAIT_TIMEOUT_S = int(os.getenv("ERAY_QR_WAIT_TIMEOUT", str(7 * 24 * 3600)))

#: QR states that will never progress to ACTIVE on their own.
TERMINAL_STATES = frozenset({"SUSPENDED", "FAILED"})
#: QR states on the way to ACTIVE.
PENDING_STATES = frozenset({"ACCEPTED", "CREATING", "WAITING_FOR_RESOURCES", "PROVISIONING"})


def default_runtime_version(accelerator_type: str) -> str:
    """Resolve the default TPU VM runtime version for an accelerator type.

    Args:
        accelerator_type: e.g. ``"v5p-8"`` or ``"v5litepod-16"``.

    Returns:
        The runtime version string for the type's generation.

    Raises:
        ValueError: If the generation has no known default (pass
            ``runtime_version`` explicitly).
    """
    family = accelerator_type.split("-")[0].lower()
    try:
        return RUNTIME_VERSION_BY_FAMILY[family]
    except KeyError:
        raise ValueError(
            f"no default runtime version known for TPU family {family!r} "
            f"(from {accelerator_type!r}); pass runtime_version explicitly"
        ) from None


@dataclass(frozen=True)
class QrSpec:
    """Specification for creating a TPU Queued Resource.

    Attributes:
        name: Queued-resource id AND (single-node case) the TPU node id. The
            node keeps this stable name across re-queues so
            ``eray tpu connect -n <name>`` always works; watchers append a
            generation suffix to the QR id, not the node id.
        accelerator_type: e.g. ``"v5p-64"``.
        zone: GCP zone, e.g. ``"us-central1-a"``.
        project: GCP project id.
        runtime_version: TPU VM runtime version; None resolves the
            generation default from :data:`RUNTIME_VERSION_BY_FAMILY`.
        capacity: ``"spot"`` (default), ``"on-demand"`` (plain queueing, no
            flag), ``"reserved"``, or ``"guaranteed"``.
        node_id: TPU node name; defaults to ``name``.
        node_count: For multi-node QRs (``--node-count``); mutually exclusive
            with an explicit node id per gcloud semantics.
        valid_until_duration: Optional auto-expiry for the request, e.g.
            ``"72h"`` — the QR fails instead of queueing forever.
        labels: Resource labels; eray ownership labels are merged in by
            callers that manage state.
        metadata: TPU VM metadata key/values.
        network: VPC network (gcloud default "default" when None).
        internal_ips: Allocate internal IPs only.
    """

    name: str
    accelerator_type: str
    zone: str
    project: str
    runtime_version: str | None = None
    capacity: CapacityT = "spot"
    node_id: str | None = None
    node_count: int | None = None
    valid_until_duration: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    network: str | None = None
    internal_ips: bool = False

    def resolved_runtime_version(self) -> str:
        """The runtime version to request (explicit or generation default)."""
        return self.runtime_version or default_runtime_version(self.accelerator_type)

    def resolved_node_id(self) -> str:
        """The TPU node id to request (explicit or the QR name)."""
        return self.node_id or self.name


@dataclass(frozen=True)
class QueuedResource:
    """A queued resource as reported by gcloud describe/list.

    Attributes:
        qr_id: Queued-resource id (short name).
        project: GCP project id.
        zone: GCP zone.
        state: Lifecycle state (``WAITING_FOR_RESOURCES``, ``ACTIVE``, ...).
        accelerator_type: Requested accelerator type, when reported.
        node_ids: TPU node ids attached to this QR, when reported.
        raw: Full gcloud JSON payload for anything not surfaced above.
    """

    qr_id: str
    project: str
    zone: str
    state: str
    accelerator_type: str | None
    node_ids: tuple[str, ...]
    raw: dict = field(repr=False)

    @property
    def is_active(self) -> bool:
        """True when the QR's capacity is provisioned."""
        return self.state == "ACTIVE"

    @property
    def is_terminal(self) -> bool:
        """True when the QR will never progress to ACTIVE on its own."""
        return self.state in TERMINAL_STATES


def _parse_state(raw: dict) -> str:
    """Extract the lifecycle state from a QR payload.

    The API nests it (``{"state": {"state": "ACTIVE"}}``); older/other
    surfaces may flatten it. Accept both.

    Args:
        raw: gcloud describe/list JSON for one queued resource.

    Returns:
        The state string, or ``"UNKNOWN"``.
    """
    state = raw.get("state")
    if isinstance(state, dict):
        return str(state.get("state", "UNKNOWN"))
    if isinstance(state, str):
        return state
    return "UNKNOWN"


def _parse_qr(raw: dict, *, project: str, zone: str) -> QueuedResource:
    """Build a QueuedResource from a gcloud JSON payload.

    Args:
        raw: gcloud describe/list JSON for one queued resource.
        project: Project the query was issued against.
        zone: Zone the query was issued against.

    Returns:
        The parsed QueuedResource.
    """
    name = str(raw.get("name", ""))
    qr_id = name.rsplit("/", 1)[-1] if name else "?"
    node_specs = (raw.get("tpu") or {}).get("nodeSpec") or []
    node_ids = tuple(str(spec.get("nodeId")) for spec in node_specs if spec.get("nodeId"))
    accelerator_type = None
    for spec in node_specs:
        acc = (spec.get("node") or {}).get("acceleratorType")
        if acc:
            accelerator_type = str(acc)
            break
    return QueuedResource(
        qr_id=qr_id,
        project=project,
        zone=zone,
        state=_parse_state(raw),
        accelerator_type=accelerator_type,
        node_ids=node_ids,
        raw=raw,
    )


def qr_create_args(spec: QrSpec, *, qr_id: str | None = None) -> list[str]:
    """Build the exact gcloud argv for creating a queued resource.

    Pure so tests can assert the argv without touching gcloud.

    Args:
        spec: The queued-resource specification.
        qr_id: Queued-resource id override (watchers pass
            ``{name}-r{generation}``); defaults to ``spec.name``.

    Returns:
        Argument list after ``gcloud`` (i.e. starting with ``compute``).

    Raises:
        ValueError: If the capacity tier is unknown.
    """
    args = [
        "compute",
        "tpus",
        "queued-resources",
        "create",
        qr_id or spec.name,
        "--zone",
        spec.zone,
        "--project",
        spec.project,
        "--accelerator-type",
        spec.accelerator_type,
        "--runtime-version",
        spec.resolved_runtime_version(),
    ]
    if spec.node_count is not None:
        args += ["--node-count", str(spec.node_count)]
    else:
        args += ["--node-id", spec.resolved_node_id()]
    if spec.capacity == "spot":
        args.append("--spot")
    elif spec.capacity == "reserved":
        args.append("--reserved")
    elif spec.capacity == "guaranteed":
        args.append("--guaranteed")
    elif spec.capacity != "on-demand":
        raise ValueError(f"unknown capacity tier {spec.capacity!r}")
    if spec.valid_until_duration:
        args += ["--valid-until-duration", spec.valid_until_duration]
    if spec.labels:
        args += ["--labels", ",".join(f"{k}={v}" for k, v in sorted(spec.labels.items()))]
    if spec.metadata:
        args += ["--metadata", ",".join(f"{k}={v}" for k, v in sorted(spec.metadata.items()))]
    if spec.network:
        args += ["--network", spec.network]
    if spec.internal_ips:
        args.append("--internal-ips")
    return args


def create_queued_resource(spec: QrSpec, *, qr_id: str | None = None) -> QueuedResource:
    """Create a queued resource and return its current view.

    Args:
        spec: The queued-resource specification.
        qr_id: Queued-resource id override; defaults to ``spec.name``.

    Returns:
        The QueuedResource right after creation (typically
        ``ACCEPTED``/``WAITING_FOR_RESOURCES``).

    Raises:
        RuntimeError: If gcloud fails (already exists, quota, bad flags).
    """
    _gcloud_or_raise(qr_create_args(spec, qr_id=qr_id))
    qr = describe_queued_resource(qr_id or spec.name, project=spec.project, zone=spec.zone)
    if qr is None:
        raise RuntimeError(f"queued resource {qr_id or spec.name} not visible after create")
    return qr


def describe_queued_resource(qr_id: str, *, project: str, zone: str) -> QueuedResource | None:
    """Describe one queued resource.

    Args:
        qr_id: Queued-resource id.
        project: GCP project id.
        zone: GCP zone.

    Returns:
        The QueuedResource, or None if it does not exist.
    """
    import subprocess

    try:
        raw = gcloud_json(
            ["compute", "tpus", "queued-resources", "describe", qr_id, "--zone", zone, "--project", project]
        )
    except subprocess.CalledProcessError as exc:
        stderr = str(exc.stderr or "")
        if "NOT_FOUND" in stderr or "not found" in stderr.lower():
            return None
        raise
    if not isinstance(raw, dict):
        return None
    return _parse_qr(raw, project=project, zone=zone)


def list_queued_resources(*, project: str, zone: str) -> list[QueuedResource]:
    """List queued resources in a zone.

    Args:
        project: GCP project id.
        zone: GCP zone.

    Returns:
        Parsed QueuedResources (possibly empty).
    """
    raw = gcloud_json(["compute", "tpus", "queued-resources", "list", "--zone", zone, "--project", project])
    if not isinstance(raw, list):
        return []
    return [_parse_qr(item, project=project, zone=zone) for item in raw if isinstance(item, dict)]


def delete_queued_resource(qr_id: str, *, project: str, zone: str, force: bool = False, wait: bool = True) -> None:
    """Delete a queued resource.

    Args:
        qr_id: Queued-resource id.
        project: GCP project id.
        zone: GCP zone.
        force: Also delete a provisioned node under the QR (required for
            non-terminal states like ACTIVE).
        wait: Block until the QR is gone (polls describe).

    Raises:
        RuntimeError: If gcloud fails, or the QR still exists after the
            delete wait times out (10 minutes).
    """
    args = ["compute", "tpus", "queued-resources", "delete", qr_id, "--zone", zone, "--project", project, "--quiet"]
    if force:
        args.append("--force")
    _gcloud_or_raise(args)
    if not wait:
        return
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if describe_queued_resource(qr_id, project=project, zone=zone) is None:
            return
        time.sleep(5)
    raise RuntimeError(f"queued resource {qr_id} still exists 10 minutes after delete")


def wait_for_active(
    qr_id: str,
    *,
    project: str,
    zone: str,
    timeout: float = ERAY_QR_WAIT_TIMEOUT_S,
    poll: float = ERAY_QR_POLL_S,
    on_state: object = None,
) -> QueuedResource:
    """Block until a queued resource reaches ACTIVE.

    Spot requests legitimately sit in ``WAITING_FOR_RESOURCES`` for hours or
    days; the default timeout mirrors ``ERAY_SCALE_ADD_TIMEOUT`` (7 days).

    Args:
        qr_id: Queued-resource id.
        project: GCP project id.
        zone: GCP zone.
        timeout: Seconds to wait before giving up.
        poll: Seconds between describes.
        on_state: Optional callable invoked as ``on_state(state: str)`` on
            every state change (progress reporting).

    Returns:
        The ACTIVE QueuedResource.

    Raises:
        RuntimeError: If the QR disappears, reaches a terminal state
            (``FAILED``/``SUSPENDED``), or the timeout expires.
    """
    deadline = time.monotonic() + timeout
    last_state: str | None = None
    while True:
        qr = describe_queued_resource(qr_id, project=project, zone=zone)
        if qr is None:
            raise RuntimeError(f"queued resource {qr_id} disappeared while waiting for ACTIVE")
        if qr.state != last_state:
            last_state = qr.state
            if callable(on_state):
                on_state(qr.state)
        if qr.is_active:
            return qr
        if qr.is_terminal:
            detail = (qr.raw.get("state") or {}).get("failedData") if isinstance(qr.raw.get("state"), dict) else None
            raise RuntimeError(f"queued resource {qr_id} reached terminal state {qr.state}: {detail or 'no detail'}")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"queued resource {qr_id} not ACTIVE after {timeout:.0f}s (state: {qr.state})")
        time.sleep(poll)
