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


"""Fleet reconciliation: make one registered cluster match its desired state.

``ensure_tpu`` is a single idempotent pass — safe to re-run any time:

- desired **down** → nothing to do (capacity teardown is explicit via the
  CLI, never implicit).
- node READY + Ray head reachable → healthy; refresh ``head_ip``.
- node READY + head unreachable → bootstrap (once per generation, when a
  ``bootstrap_cmd`` is configured) and ``connect_tpus``.
- no node, no QR → create the queued resource (spot by default) and
  optionally wait for ACTIVE.
- QR pending (``WAITING_FOR_RESOURCES``/``PROVISIONING``) → report; nothing
  to force.
- QR terminal (``FAILED``/``SUSPENDED``) → report ``NEEDS_RECREATE``; the
  autonomous re-queue loop with budgets/backoff is the watcher's job
  (``eray fleet watch``), not ensure's.

The observe helpers are shared with the watcher.
"""

from __future__ import annotations

import socket
import subprocess
import time
from typing import TYPE_CHECKING

from .qr import QrSpec, create_queued_resource, describe_queued_resource, wait_for_active
from .registry import ClusterRecord, ClusterRegistry

if TYPE_CHECKING:
    from ..cli.utils import TpuInfo

RAY_HEAD_PORT = 6379
#: Remote port the Ray dashboard / Jobs API listens on. Canonical home for
#: this constant so the CLI modules (tpu/dashboard/jobs) import it rather than
#: each re-declaring 8265 under a different name.
RAY_DASHBOARD_PORT = 8265


def head_reachable(ip: str, *, port: int = RAY_HEAD_PORT, timeout: float = 2.0) -> bool:
    """Whether a Ray head answers on the GCS port.

    Args:
        ip: Head internal IP.
        port: Head port (default Ray GCS 6379).
        timeout: TCP connect timeout in seconds.

    Returns:
        True if a TCP connection succeeds.
    """
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def describe_node(name: str, *, project: str, zone: str) -> TpuInfo | None:
    """Describe a TPU node, returning None when it does not exist.

    Args:
        name: TPU node id.
        project: GCP project id.
        zone: GCP zone.

    Returns:
        The TpuInfo, or None on NOT_FOUND. A booting node (no network
        endpoints yet) comes back with its real state and num_hosts 0
        rather than an error, so pollers can wait on it.
    """
    from ..cli.utils import discover_tpu

    try:
        return discover_tpu(name, project, zone, allow_no_ips=True)
    except subprocess.CalledProcessError as exc:
        stderr = str(exc.stderr or "")
        if "NOT_FOUND" in stderr or "not found" in stderr.lower():
            return None
        raise


def _spec_from_record(record: ClusterRecord) -> QrSpec:
    """Build the QR spec a record describes (with eray ownership labels).

    Args:
        record: The cluster record.

    Returns:
        The QrSpec for (re)creating this cluster's capacity.

    Raises:
        ValueError: If the record is missing project/zone/accelerator_type.
    """
    if not (record.project and record.zone and record.accelerator_type):
        raise ValueError(f"cluster {record.name!r} is missing project/zone/accelerator_type in the registry")
    return QrSpec(
        name=record.name,
        accelerator_type=record.accelerator_type,
        zone=record.zone,
        project=record.project,
        runtime_version=record.runtime_version,
        capacity=record.capacity,  # type: ignore[arg-type]
        labels={"eray-cluster": record.name.replace("_", "-").lower()},
    )


def ensure_tpu(
    name: str,
    *,
    registry: ClusterRegistry | None = None,
    connect: bool = True,
    wait_timeout: float | None = None,
    on_event: object = None,
) -> dict:
    """Reconcile one registered cluster toward its desired state (one pass).

    Args:
        name: Registered cluster name.
        registry: Registry to use (default: the configured one).
        connect: Start/repair the Ray cluster when the node is READY but the
            head is unreachable.
        wait_timeout: When capacity had to be requested, block up to this many
            seconds for the QR to reach ACTIVE (None: return immediately with
            the pending state).
        on_event: Optional callable ``on_event(msg: str)`` for progress lines.

    Returns:
        Report dict: ``{"name", "state", "detail", "head_ip"}`` where state is
        one of ``DOWN, HEALTHY, CONNECTED, WAITING, PROVISIONING, ACTIVE,
        NEEDS_RECREATE, NEEDS_BOOTSTRAP, UNREACHABLE``.

    Raises:
        KeyError: If the cluster is not registered.
        RuntimeError: If connecting or provisioning fails outright.
    """
    registry = registry or ClusterRegistry.from_config()
    record = registry.get(name)
    if record is None:
        raise KeyError(f"cluster {name!r} is not registered (eray fleet add ...)")

    def emit(msg: str) -> None:
        if callable(on_event):
            on_event(msg)

    def report(state: str, detail: str = "", head_ip: str | None = None) -> dict:
        registry.mutate_record(name, lambda r: setattr(r, "state", state))
        return {"name": name, "state": state, "detail": detail, "head_ip": head_ip}

    if record.desired_state != "up":
        return report("DOWN", "desired_state is down; nothing to do")

    if record.kind != "qr":
        return report(record.state, f"kind={record.kind} is not reconciled by ensure_tpu")

    node = describe_node(record.name, project=record.project, zone=record.zone)

    # Node exists and is READY → this is a Ray-level concern from here on.
    if node is not None and node.state == "READY":
        head_ip = node.internal_ips[0]
        if head_reachable(head_ip):
            registry.mutate_record(name, lambda r: setattr(r, "head_ip", head_ip))
            return report("HEALTHY", "node READY, Ray head reachable", head_ip)
        if not connect:
            return report("UNREACHABLE", "node READY but Ray head not reachable (connect=False)", head_ip)
        emit(f"node READY, head unreachable — connecting Ray on {record.name}")
        _bootstrap_if_needed(record, node, registry, emit)
        from ..cli.tpu import connect_tpus

        result = connect_tpus(node)

        def _connected(r: ClusterRecord) -> None:
            r.head_ip = result.head_ip
            r.state = "HEALTHY"

        registry.mutate_record(name, _connected)
        return {"name": name, "state": "CONNECTED", "detail": f"{result.num_hosts} hosts", "head_ip": result.head_ip}

    # No READY node: look at the queued resource.
    qr_id = record.qr_id or record.name
    qr = describe_queued_resource(qr_id, project=record.project, zone=record.zone)

    if qr is None:
        emit(f"no node and no QR — requesting {record.capacity} {record.accelerator_type} as {qr_id}")
        spec = _spec_from_record(record)

        def _intent(r: ClusterRecord) -> None:
            r.intent = {"action": "qr_create", "target": qr_id, "ts": time.time()}
            r.qr_id = qr_id

        registry.mutate_record(name, _intent)
        created = create_queued_resource(spec, qr_id=qr_id)
        registry.mutate_record(name, lambda r: setattr(r, "intent", None))
        if wait_timeout:
            emit(f"waiting up to {wait_timeout:.0f}s for {qr_id} to become ACTIVE")
            wait_for_active(qr_id, project=record.project, zone=record.zone, timeout=wait_timeout, on_state=emit)
            return ensure_tpu(name, registry=registry, connect=connect, wait_timeout=None, on_event=on_event)
        return report("WAITING", f"queued resource created ({created.state})")

    if qr.state in ("WAITING_FOR_RESOURCES", "ACCEPTED", "CREATING"):
        return report("WAITING", f"QR {qr.qr_id} is {qr.state}")
    if qr.state == "PROVISIONING":
        return report("PROVISIONING", f"QR {qr.qr_id} is provisioning")
    if qr.is_active:
        # ACTIVE but node not READY (booting or CREATING): report and let the
        # caller re-run; connect will happen once the node is READY.
        return report("ACTIVE", f"QR ACTIVE, node state: {node.state if node else 'missing'}")
    return report(
        "NEEDS_RECREATE",
        f"QR {qr.qr_id} is {qr.state}; run 'eray fleet watch' (or recreate manually) to re-queue capacity",
    )


def _bootstrap_if_needed(record: ClusterRecord, node: TpuInfo, registry: ClusterRegistry, emit) -> None:
    """Run the cluster's bootstrap command once per generation.

    Args:
        record: The cluster record.
        node: The READY TpuInfo.
        registry: Registry for persisting bootstrapped_generation.
        emit: Progress callback.

    Raises:
        RuntimeError: If the bootstrap command fails on any host.
    """
    if not record.bootstrap_cmd or record.bootstrapped_generation == record.generation:
        return
    from ..cli.utils import run_on_all_hosts

    emit(f"bootstrapping {node.num_hosts} host(s) (generation {record.generation})")
    results = run_on_all_hosts(node, record.bootstrap_cmd, timeout=1800)
    failed = [idx for idx, res in results if res.returncode != 0]
    if failed:
        raise RuntimeError(f"bootstrap_cmd failed on host(s) {failed} of {record.name}")
    registry.mutate_record(record.name, lambda r: setattr(r, "bootstrapped_generation", record.generation))


def qr_tunnel_argv(
    record: ClusterRecord, *, remote_port: int, local_port: int, extra_ports: tuple[tuple[int, int], ...] = ()
) -> list[str]:
    """Build the gcloud SSH argv that forwards a QR-kind cluster's head port.

    Ray heads listen on internal VPC IPs, so from outside the VPC (a laptop)
    the dashboard/Jobs API are unreachable directly; this wraps the gcloud
    TPU SSH port-forward (worker 0 is always the head). Shared between
    `eray fleet tunnel` (foreground) and `eray dashboard` (tracked
    background), so the two can't drift.

    Args:
        record: The registered cluster (its name/project/zone identify the
            TPU node).
        remote_port: Port on the head to forward (Ray dashboard: 8265).
        local_port: Local port to bind.
        extra_ports: Additional ``(local_port, remote_port)`` pairs to
            forward over the same SSH connection — e.g. the GCS port
            (`RAY_HEAD_PORT`, 6379) alongside the dashboard port, so
            `ray status`/`ray.init()`-based tools work through one tunnel.

    Returns:
        Full argv for ``gcloud compute tpus tpu-vm ssh ... -- -N -L ...``,
        with one ``-L`` flag per forwarded port.
    """
    argv = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        record.name,
        "--project",
        record.project,
        "--zone",
        record.zone,
        "--worker",
        "0",
        "--",
        "-N",
        "-L",
        f"{local_port}:localhost:{remote_port}",
    ]
    for extra_local, extra_remote in extra_ports:
        argv += ["-L", f"{extra_local}:localhost:{extra_remote}"]
    return argv


def fleet_status(registry: ClusterRegistry | None = None, *, probe: bool = True) -> list[dict]:
    """Summarize every registered cluster.

    Args:
        registry: Registry to read (default: configured).
        probe: Also probe QR state and head reachability live (slower).

    Returns:
        One row per cluster: name, kind, type, zone, desired, state, qr_state,
        head, generation.
    """
    registry = registry or ClusterRegistry.from_config()
    rows = []
    for name, record in sorted(registry.load().items()):
        row = {
            "name": name,
            "kind": record.kind,
            "type": record.accelerator_type or "?",
            "zone": record.zone or "?",
            "desired": record.desired_state,
            "state": record.state,
            "qr_state": "?",
            "head": "?",
            "generation": record.generation,
        }
        if probe and record.kind == "qr" and record.project and record.zone:
            qr = describe_queued_resource(record.qr_id or name, project=record.project, zone=record.zone)
            row["qr_state"] = qr.state if qr else "NOT_FOUND"
            if record.head_ip:
                row["head"] = "up" if head_reachable(record.head_ip) else "down"
        rows.append(row)
    return rows
