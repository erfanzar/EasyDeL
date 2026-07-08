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

"""TPU cluster connection logic: start, stop, status, and health checks.

Supports two connection modes:
    - **gcloud mode**: TPU discovered via gcloud, SSH via `gcloud compute tpus tpu-vm ssh`
    - **direct-IP mode**: IPs provided directly, SSH via plain `ssh`

Both modes produce the same Ray cluster with TPU custom resources.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import shlex
import socket
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass

from ..provision.fleet import RAY_DASHBOARD_PORT, RAY_HEAD_PORT
from .utils import (
    TpuInfo,
    build_ray_resource_flags,
    error,
    info,
    run_on_all_hosts,
    run_on_host,
    success,
    warning,
)

logger = logging.getLogger("eray.cli.tpu")

RAY_TMP_DIR = "/tmp/eray_tmp"
RAY_READINESS_POLL_S = 5


def _wait_for_port(ip: str, port: int, *, attempts: int = 30, interval: float = 2.0) -> bool:
    """Poll a TCP port until it accepts connections or attempts run out.

    Args:
        ip: IP address to connect to.
        port: TCP port number to poll.
        attempts: Maximum number of connection attempts.
        interval: Seconds to wait between attempts.

    Returns:
        True if the port accepts a connection before attempts run out, False otherwise.
    """
    for _ in range(attempts):
        try:
            with socket.create_connection((ip, port), timeout=1):
                return True
        except OSError:
            time.sleep(interval)
    return False


def _ray_bin_preamble(ray_bin: str, *, strict: bool) -> str:
    """Shell preamble that resolves the ray executable into $RAY_BIN on the target host.

    Non-interactive SSH shells (``gcloud ... ssh --command``, plain ``ssh``)
    do not load the interactive rc files, so a venv-installed ray is not on
    PATH there even though bare ``ray`` resolves in the operator's own shell
    (observed live on a v5p-1024: the head, started locally, came up while
    all 127 SSH-started workers failed with ``ray: command not found``).

    When ``ray_bin`` is the bare default ``"ray"``, candidates are tried in
    order: this driver's own venv ray (TPU pod hosts share the filesystem
    layout, and this keeps worker ray versions matched to the head), then
    PATH, then ``~/.local/bin/ray``. An explicit ``ray_bin`` is the sole
    candidate.

    Args:
        ray_bin: Ray binary as passed on the CLI (bare name or path).
        strict: If True, the preamble exits 127 with a diagnostic when no
            candidate resolves (start commands must not proceed); if False it
            falls back to ``ray_bin`` verbatim so best-effort cleanup chains
            (whose ray calls already tolerate failure) keep running.

    Returns:
        A shell snippet to prepend to remote commands; subsequent commands
        invoke ``"$RAY_BIN"``.
    """
    if ray_bin != "ray":
        candidates = [ray_bin]
    else:
        candidates = []
        local_ray = os.path.join(os.path.dirname(sys.executable), "ray")
        if os.path.exists(local_ray):
            candidates.append(local_ray)
        candidates.extend(["ray", "~/.local/bin/ray"])
    # `~/...` must stay unquoted for tilde expansion in the remote shell.
    checks = " || ".join(f"command -v {c if c.startswith('~') else shlex.quote(c)}" for c in candidates)
    if strict:
        tried = ", ".join(candidates)
        return (
            f'RAY_BIN="$({checks})" || '
            f'{{ echo "eray: ray executable not found on this host (tried: {tried}); '
            f'non-interactive SSH shells do not see your venv PATH — pass --ray-bin /abs/path/to/ray" >&2; '
            f"exit 127; }}"
        )
    return f'RAY_BIN="$({checks})" || RAY_BIN={shlex.quote(ray_bin)}'


def _require_reachable_head(ray_address: str, *, timeout: float = 2.0) -> None:
    """Fail fast when no Ray head is listening at the address.

    ``ray.init(address=...)`` against a dead head spends ~107s in GCS
    connection retries (measured live) before erroring with unactionable
    GCS warnings — and the read-only commands (status/resources/health) are
    exactly what an operator runs to check whether the cluster is up. A
    2-second TCP probe converts that into an immediate, actionable error.

    Args:
        ray_address: Ray cluster address (ip:port).
        timeout: TCP connect timeout in seconds.

    Raises:
        RuntimeError: If nothing accepts connections on the head port.
    """
    ip, _, port = ray_address.partition(":")
    try:
        with socket.create_connection((ip, int(port or RAY_HEAD_PORT)), timeout=timeout):
            return
    except OSError as exc:
        raise RuntimeError(
            f"no Ray cluster reachable at {ray_address} ({exc}); start one with 'eray tpu connect'"
        ) from None


@dataclass
class ConnectResult:
    """Result of a TPU cluster connect operation."""

    tpu: TpuInfo
    head_ip: str
    ray_address: str
    dashboard_url: str
    num_hosts: int


def connect_tpus(
    tpu: TpuInfo,
    *,
    ray_bin: str = "ray",
    ray_tmp_dir: str = RAY_TMP_DIR,
    timeout: int = 300,
    user: str | None = None,
    ssh_key: str | None = None,
) -> ConnectResult:
    """Connect all TPU hosts into a Ray cluster.

    Works with both gcloud-discovered TPUs and direct-IP connections.

    This function:
    1. Cleans any existing Ray processes on all hosts.
    2. Starts Ray head on worker 0.
    3. Starts Ray workers on all other hosts, pointing at the head.
    4. Waits for all nodes to register with the cluster.

    Args:
        tpu: TPU info (gcloud-discovered or built from IPs).
        ray_bin: Ray binary on the TPU hosts. The bare default ``"ray"``
            is resolved per host at run time (driver venv ray, then PATH,
            then ``~/.local/bin/ray``) because non-interactive SSH shells
            do not see venv PATH entries.
        ray_tmp_dir: Temp directory for Ray on the hosts.
        timeout: Overall readiness timeout in seconds.
        user: SSH user for direct-IP mode (ignored in gcloud mode).
        ssh_key: SSH key path for direct-IP mode (ignored in gcloud mode).

    Returns:
        ConnectResult with cluster connection details.

    Raises:
        RuntimeError: If Ray fails to start or cluster doesn't become ready.
    """
    head_ip = tpu.internal_ips[0]
    ray_address = f"{head_ip}:{RAY_HEAD_PORT}"
    dashboard_url = f"http://{head_ip}:{RAY_DASHBOARD_PORT}"

    mode = "gcloud" if tpu.is_gcloud_managed else "direct-IP"
    info(f"Connecting {tpu.num_hosts} hosts ({mode}) into Ray cluster...")

    # --- Step 1: Clean existing Ray ---
    info("Cleaning existing Ray processes on all hosts...")
    _cleanup_ray(tpu, ray_bin=ray_bin, user=user, ssh_key=ssh_key)

    # --- Step 2: Start Ray head ---
    head_resources = build_ray_resource_flags(tpu, is_head=True)
    ray_preamble = _ray_bin_preamble(ray_bin, strict=True)
    info(f"Starting Ray head on host 0 ({head_ip})...")
    head_cmd = (
        f"{ray_preamble} && "
        f"export TMPDIR={ray_tmp_dir} RAY_TMPDIR={ray_tmp_dir}/ray && "
        f"mkdir -p $TMPDIR $RAY_TMPDIR && "
        f'"$RAY_BIN" stop --force >/dev/null 2>&1 || true && '
        f'"$RAY_BIN" start --head '
        f"--port={RAY_HEAD_PORT} "
        f"--resources='{head_resources}' "
        f"--node-ip-address={head_ip} "
        f"--dashboard-host=0.0.0.0 "
        f"--disable-usage-stats"
    )
    result = run_on_host(tpu, 0, head_cmd, timeout=120, user=user, ssh_key=ssh_key)
    if result.returncode != 0:
        error("Ray head failed to start on host 0")
        if result.stderr:
            error(result.stderr.strip())
        raise RuntimeError("Ray head failed to start")
    success("Ray head started")

    # --- Step 3: Start Ray workers (parallel fan-out) ---
    worker_resources = build_ray_resource_flags(tpu, is_head=False)
    if tpu.num_hosts > 1:
        # One reachability check for the head port; per-worker checks would
        # serialize the join for no benefit.
        if not _wait_for_port(head_ip, RAY_HEAD_PORT):
            error(f"Ray head port {RAY_HEAD_PORT} not reachable")
            raise RuntimeError(f"Cannot reach head at {ray_address}")

        def _start_worker(host_idx: int):
            worker_ip = tpu.internal_ips[host_idx]
            worker_cmd = (
                f"{ray_preamble} && "
                f"export TMPDIR={ray_tmp_dir} RAY_TMPDIR={ray_tmp_dir}/ray && "
                f"mkdir -p $TMPDIR $RAY_TMPDIR && "
                f'"$RAY_BIN" stop --force >/dev/null 2>&1 || true && '
                f'"$RAY_BIN" start '
                f"--address={ray_address} "
                f"--resources='{worker_resources}' "
                f"--node-ip-address={worker_ip} "
                f"--disable-usage-stats"
            )
            try:
                result = run_on_host(tpu, host_idx, worker_cmd, timeout=120, user=user, ssh_key=ssh_key)
            except Exception as exc:
                # A single bad/unreachable host (e.g. a hung SSH connection raising
                # subprocess.TimeoutExpired) must not escape the pool: an exception
                # here would propagate out of pool.map() and abort the whole join
                # loop before the other hosts' results are collected, losing
                # aggregated failure visibility instead of reporting it.
                result = subprocess.CompletedProcess(args=worker_cmd, returncode=1, stdout="", stderr=str(exc))
            return host_idx, result

        info(f"Starting {tpu.num_hosts - 1} Ray workers in parallel → {ray_address}...")
        failed_workers: list[int] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=tpu.num_hosts - 1) as pool:
            for host_idx, result in pool.map(_start_worker, range(1, tpu.num_hosts)):
                if result.returncode != 0:
                    error(f"Ray worker {host_idx} failed to start")
                    if result.stderr:
                        error(result.stderr.strip())
                    failed_workers.append(host_idx)
        if failed_workers:
            raise RuntimeError(f"{len(failed_workers)} Ray workers failed to start (e.g. {failed_workers[:8]})")
        success(f"All {tpu.num_hosts - 1} Ray workers started")

    # --- Step 4: Wait for cluster readiness ---
    info(f"Waiting for cluster readiness ({tpu.num_hosts} nodes, {ray_address})...")
    if not _wait_for_cluster(ray_address, tpu.num_hosts, timeout=timeout):
        error(f"Ray cluster did not become ready after {timeout}s")
        raise RuntimeError("Cluster readiness timeout")
    success(f"Cluster ready: {tpu.num_hosts} nodes connected")

    return ConnectResult(
        tpu=tpu,
        head_ip=head_ip,
        ray_address=ray_address,
        dashboard_url=dashboard_url,
        num_hosts=tpu.num_hosts,
    )


def disconnect_tpus(
    tpu: TpuInfo,
    *,
    ray_bin: str = "ray",
    user: str | None = None,
    ssh_key: str | None = None,
) -> None:
    """Stop Ray on all TPU hosts.

    Runs `ray stop --force` on every worker and kills stale Ray processes.
    Works with both gcloud and direct-IP modes.

    Args:
        tpu: TPU info (gcloud-discovered or built from IPs).
        ray_bin: Ray binary on the TPU hosts. The bare default ``"ray"``
            is resolved per host at run time (driver venv ray, then PATH,
            then ``~/.local/bin/ray``) because non-interactive SSH shells
            do not see venv PATH entries.
        user: SSH user for direct-IP mode (ignored in gcloud mode).
        ssh_key: SSH key path for direct-IP mode (ignored in gcloud mode).

    Returns:
        None
    """
    info("Stopping Ray on all hosts...")
    _cleanup_ray(tpu, ray_bin=ray_bin, user=user, ssh_key=ssh_key)
    success("Ray stopped on all hosts")


def _cleanup_ray(
    tpu: TpuInfo,
    *,
    ray_bin: str = "ray",
    user: str | None = None,
    ssh_key: str | None = None,
) -> None:
    """Clean Ray processes on all hosts in parallel.

    Stops Ray gracefully, kills any remaining Ray-related processes, and
    checks for lingering ports.

    Args:
        tpu: TPU info containing host IPs.
        ray_bin: Ray binary on the TPU hosts. The bare default ``"ray"``
            is resolved per host at run time (driver venv ray, then PATH,
            then ``~/.local/bin/ray``) because non-interactive SSH shells
            do not see venv PATH entries.
        user: SSH user for direct-IP mode (ignored in gcloud mode).
        ssh_key: SSH key path for direct-IP mode (ignored in gcloud mode).

    Returns:
        None
    """
    cleanup_cmd = (
        f"{_ray_bin_preamble(ray_bin, strict=False)}; "
        f'timeout 60 "$RAY_BIN" stop --force >/tmp/eray-ray-stop.log 2>&1 || true; '
        "pids=$(ps -eo pid=,args= | awk "
        "'/[r]ay start/ || /[r]ay::/ || /[s]ite-packages\\/ray\\// "
        "|| /[r]ay\\/core\\/src\\/ray\\// { print $1 }' | sort -u); "
        'if [ -n "$pids" ]; then kill $pids 2>/dev/null || true; sleep 2; fi; '
        "if command -v ss >/dev/null 2>&1; then "
        f'  remaining=$(ss -ltnp 2>/dev/null | grep -E ":{RAY_HEAD_PORT}|{RAY_DASHBOARD_PORT} " || true); '
        '  if [ -n "$remaining" ]; then echo "WARNING: Ray ports still in use"; fi; '
        "fi; "
        "echo 'cleanup done'"
    )
    results = run_on_all_hosts(tpu, cleanup_cmd, timeout=120, user=user, ssh_key=ssh_key)
    for host_idx, result in results:
        if result.returncode != 0:
            warning(f"Host {host_idx}: cleanup had issues")
        elif result.stdout and "cleanup done" in result.stdout:
            info(f"Host {host_idx}: Ray cleaned")


def _wait_for_cluster(
    ray_address: str, expected_nodes: int, *, timeout: int = 300, dashboard_port: int | None = None
) -> bool:
    """Poll the head's dashboard state API until all expected nodes are alive.

    Deliberately NOT ray.init(): a driver handshake requires the operator's
    Python and Ray versions to match the cluster's exactly, so it can never
    succeed when the operator box connects a cluster it did not bootstrap
    itself (observed live: a 3.13 venv watching a node whose head runs the
    image's system 3.10). The dashboard HTTP API has no such coupling.

    Args:
        ray_address: Ray cluster address (ip:port); the dashboard is assumed
            on the same host.
        expected_nodes: Number of nodes expected to be alive.
        timeout: Maximum time to wait in seconds.
        dashboard_port: Dashboard port override (default RAY_DASHBOARD_PORT).

    Returns:
        True if all expected nodes are registered before the timeout, False otherwise.
    """
    import json
    import urllib.request

    port = dashboard_port or RAY_DASHBOARD_PORT
    url = f"http://{ray_address.split(':')[0]}:{port}/api/v0/nodes?limit=10000"
    deadline = time.monotonic() + timeout
    last_seen = None
    last_error = ""

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                payload = json.loads(resp.read())
            rows = payload["data"]["result"]["result"]
            alive = [n for n in rows if n.get("state") == "ALIVE"]
            total_tpus = sum(n.get("resources_total", {}).get("TPU", 0) for n in alive)
            current = (len(alive), int(total_tpus))
            if current != last_seen:
                info(f"Cluster: {current[0]}/{expected_nodes} nodes, {current[1]} TPU resources")
                last_seen = current
            if len(alive) >= expected_nodes:
                return True
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.debug(f"Waiting for cluster: {last_error}")
        time.sleep(RAY_READINESS_POLL_S)
    if last_error:
        error(f"cluster readiness: last probe error was {last_error}")

    return False


def _exc_str(exc: BaseException) -> str:
    """A never-empty description of an exception.

    Several Ray connection failures (notably ``ConnectionError`` from a
    driver handshake that can't complete over a single tunneled port)
    stringify to ``""``, which surfaced as a bare "Failed: " with no reason.
    Fall back to the type name so the operator always sees *something*.

    Args:
        exc: The exception.

    Returns:
        ``str(exc)`` if non-empty, else the exception's class name.
    """
    return str(exc) or type(exc).__name__


def _ip_sort_key(entry: dict):
    """Sort nodes by dotted-quad IP, non-IPs last."""
    try:
        return (0, tuple(int(part) for part in str(entry["ip"]).split(".")))
    except (ValueError, KeyError):
        return (1, str(entry.get("ip", "")))


def _dashboard_port_for(host: str) -> int:
    """The local port serving the Ray dashboard for `host`.

    Over an ``eray dashboard``/``eray fleet tunnel`` forward the dashboard
    rarely lands on 8265 (it falls back when 8265 is busy on the laptop), and
    the address the resource commands resolve is the *GCS* tunnel's local
    port, not the dashboard's. So when `host` is the loopback (a tunnel),
    look the dashboard's real local port up in the tunnel store; a direct
    cluster IP serves the dashboard on the standard port.

    Args:
        host: The host portion of a resolved cluster address.

    Returns:
        The dashboard port to query on `host` — the tracked tunnel's local
        port when exactly one dashboard tunnel is open and `host` is
        loopback, else `RAY_DASHBOARD_PORT`.
    """
    if host in ("127.0.0.1", "localhost", "::1"):
        from ..provision.tunnel import tunnels_for_remote_port

        dash = tunnels_for_remote_port(RAY_DASHBOARD_PORT)
        if len(dash) == 1:
            return dash[0].local_port
    return RAY_DASHBOARD_PORT


def _dashboard_nodes(ray_address: str, *, dashboard_port: int | None = None, timeout: float = 10.0) -> list[dict]:
    """Node list from the Ray dashboard state API (``/api/v0/nodes``).

    Plain HTTP against the dashboard — no ``ray.init`` driver handshake, so
    it works from a laptop over the dashboard tunnel and needs no
    version-exact client. Each node carries ``resources_total`` and a
    liveness ``state`` (but no availability — the state API doesn't expose
    it). The dashboard is assumed on the same host as ``ray_address``; its
    port is `dashboard_port` when given, else resolved by `_dashboard_port_for`
    (which finds the real local port of an open dashboard tunnel).

    Args:
        ray_address: A cluster address (``host:port``); only the host is used.
        dashboard_port: Dashboard HTTP port; None resolves it per host.
        timeout: Per-request timeout in seconds.

    Returns:
        The node result rows.

    Raises:
        RuntimeError: On any HTTP/parse failure (with a readable reason).
    """
    import json
    import urllib.error
    import urllib.request

    host = ray_address.split(":")[0]
    port = dashboard_port if dashboard_port is not None else _dashboard_port_for(host)
    url = f"http://{host}:{port}/api/v0/nodes?limit=10000"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        return payload["data"]["result"]["result"]
    except (urllib.error.URLError, KeyError, ValueError, OSError) as exc:
        raise RuntimeError(f"dashboard state API unreachable at {url} ({_exc_str(exc)})") from None


def _autoscaler_cluster_status(ray_address: str):
    """The autoscaler's structured cluster status, read from GCS.

    This is exactly what ``ray status`` reads: a pure GCS read of the
    resource state, so it works from a laptop over the GCS tunnel (no
    driver join, unlike ``ray.init``) and — unlike the dashboard state API
    — includes per-resource *usage*, not just totals. Only populated on
    autoscaler-managed clusters (``eray autoscale``); connect-mode/QR
    clusters have no autoscaler, so callers fall back to `_dashboard_nodes`.

    Args:
        ray_address: The GCS address (``host:port``, default GCS port 6379).

    Returns:
        A ``ray.autoscaler.v2.sdk.ClusterStatus``.
    """
    from ray.autoscaler.v2.sdk import get_cluster_status

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return get_cluster_status(ray_address)


def _usage_map(resource_usages) -> dict[str, dict]:
    """``{name: {total, available, used}}`` from autoscaler ResourceUsages."""
    out: dict[str, dict] = {}
    for ru in resource_usages:
        used = ru.used
        total = ru.total
        out[ru.resource_name] = {"total": total, "available": max(total - used, 0.0), "used": used}
    return out


def cluster_status(ray_address: str) -> dict:
    """Cluster status summary, without a ``ray.init`` driver handshake.

    Reads the autoscaler status over GCS (`_autoscaler_cluster_status`);
    on a non-autoscaler cluster, falls back to the dashboard state API
    (`_dashboard_nodes`). Both work from a laptop over the tunnels
    `eray dashboard open` forwards.

    Args:
        ray_address: Ray cluster / GCS address (ip:port).

    Returns:
        A dict with keys: alive_nodes, total_nodes, resources, node_ips,
        dashboard_url.

    Raises:
        RuntimeError: If neither status source is reachable.
    """
    _require_reachable_head(ray_address)
    dashboard_url = f"http://{ray_address.split(':')[0]}:{RAY_DASHBOARD_PORT}"
    errors = []
    try:
        status = _autoscaler_cluster_status(ray_address)
        alive = [*status.active_nodes, *status.idle_nodes]
        resources = {ru.resource_name: ru.total for ru in status.cluster_resource_usage}
        if not alive and not resources:
            # A non-autoscaler cluster can answer the GCS read with an empty
            # status instead of raising; treat that as a miss so the dashboard
            # fallback (which does see the nodes) runs rather than returning
            # an empty summary.
            raise RuntimeError("autoscaler status reported no nodes or resources")
        # failed_nodes keeps a lost node in the total (matching the dashboard
        # fallback's len(rows)), so `Alive < Total` still flags a crash.
        failed = getattr(status, "failed_nodes", None) or []
        return {
            "alive_nodes": len(alive),
            "total_nodes": len(alive) + len(status.pending_nodes) + len(failed),
            "resources": resources,
            "node_ips": [n.ip_address for n in alive],
            "dashboard_url": dashboard_url,
        }
    except Exception as exc:
        errors.append(("autoscaler", exc))

    try:
        rows = _dashboard_nodes(ray_address)
        alive = [n for n in rows if n.get("state") == "ALIVE"]
        resources = {}
        for n in alive:
            for k, v in (n.get("resources_total") or {}).items():
                resources[k] = resources.get(k, 0.0) + v
        return {
            "alive_nodes": len(alive),
            "total_nodes": len(rows),
            "resources": resources,
            "node_ips": [n.get("node_ip", "?") for n in alive],
            "dashboard_url": dashboard_url,
        }
    except Exception as exc:
        errors.append(("dashboard", exc))

    raise RuntimeError("could not read cluster status: " + "; ".join(f"{k}: {_exc_str(e)}" for k, e in errors))


def resource_usage(ray_address: str, *, per_node: bool = False) -> dict:
    """Collect cluster resource totals, availability, and usage.

    Reads the autoscaler status over GCS the way ``ray status`` does
    (`_autoscaler_cluster_status`) — which works from a laptop over the GCS
    tunnel and carries per-resource *usage* (used/total). On a
    non-autoscaler cluster it falls back to the dashboard state API
    (`_dashboard_nodes`), which has totals + liveness only, so ``used`` and
    ``available`` come back ``None`` (rendered as ``?``). Neither path does
    a ``ray.init`` driver handshake, which can't complete over a single
    tunneled port and previously failed with an empty ``ConnectionError``.

    Args:
        ray_address: Ray cluster / GCS address (ip:port).
        per_node: If True, include a per-node breakdown of alive nodes.

    Returns:
        A dict with:
            resources: {name: {"total", "available", "used"}} for the cluster.
            nodes (only when per_node): list of {"ip", "node_id", "resources"}
                sorted by IP, where each node's resources map has the same
                shape (``node:*`` marker resources excluded).

    Raises:
        RuntimeError: If neither status source is reachable.
    """
    _require_reachable_head(ray_address)
    errors = []
    try:
        return _resource_usage_via_autoscaler(ray_address, per_node=per_node)
    except Exception as exc:
        errors.append(("autoscaler", exc))
    try:
        return _resource_usage_via_dashboard(ray_address, per_node=per_node)
    except Exception as exc:
        errors.append(("dashboard", exc))
    raise RuntimeError("could not read resource usage: " + "; ".join(f"{k}: {_exc_str(e)}" for k, e in errors))


def _resource_usage_via_autoscaler(ray_address: str, *, per_node: bool) -> dict:
    """`resource_usage` via the autoscaler status (used/total per resource)."""
    status = _autoscaler_cluster_status(ray_address)
    resources = _usage_map(status.cluster_resource_usage)
    if not resources:
        # Empty means this isn't an autoscaler-managed cluster (the GCS read
        # answered but had nothing); fall through to the dashboard state API
        # instead of reporting an empty resource table.
        raise RuntimeError("autoscaler status reported no resources")
    out: dict = {"resources": resources}
    if per_node:
        nodes = []
        for node in [*status.active_nodes, *status.idle_nodes]:
            usage = node.resource_usage.usage if node.resource_usage else []
            node_resources = _usage_map(ru for ru in usage if not ru.resource_name.startswith("node:"))
            nodes.append({"ip": node.ip_address, "node_id": node.node_id, "resources": node_resources})
        nodes.sort(key=_ip_sort_key)
        out["nodes"] = nodes
    return out


def _resource_usage_via_dashboard(ray_address: str, *, per_node: bool) -> dict:
    """`resource_usage` via the dashboard state API (totals only; used=None)."""
    rows = _dashboard_nodes(ray_address)
    alive = [n for n in rows if n.get("state") == "ALIVE"]

    def _totals(mapping: dict) -> dict:
        return {name: {"total": tot, "available": None, "used": None} for name, tot in mapping.items()}

    cluster: dict[str, float] = {}
    for n in alive:
        for name, tot in (n.get("resources_total") or {}).items():
            cluster[name] = cluster.get(name, 0.0) + tot
    out: dict = {"resources": _totals(cluster)}
    if per_node:
        nodes = []
        for n in alive:
            node_map = {
                name: tot for name, tot in (n.get("resources_total") or {}).items() if not name.startswith("node:")
            }
            nodes.append({"ip": n.get("node_ip", "?"), "node_id": n.get("node_id"), "resources": _totals(node_map)})
        nodes.sort(key=_ip_sort_key)
        out["nodes"] = nodes
    return out


def health_check(ray_address: str, tpu_type: str | None = None) -> list[dict]:
    """Run a health check across the cluster.

    Uses a Ray remote function to run a check on every host, reporting
    JAX devices and host info.

    Unlike ``eray tpu status`` / ``eray resources`` (which read cluster
    state over GCS/HTTP), this runs a JAX device probe *on every worker*,
    so it needs a real ``ray.init`` driver join — which can't complete over
    a single tunneled port. It therefore works only on the cluster itself,
    not from a laptop tunnel; that case raises with guidance rather than an
    opaque connection error.

    Args:
        ray_address: Ray cluster address (ip:port).
        tpu_type: Optional TPU type (e.g. "v4-32") for scheduling.

    Returns:
        A list of per-host health report dicts.

    Raises:
        RuntimeError: If the driver can't join the cluster (e.g. run from a
            laptop over a tunnel instead of on the cluster).
    """
    import ray

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if not ray.is_initialized():
            _require_reachable_head(ray_address)
            try:
                ray.init(address=ray_address, ignore_reinit_error=True, logging_level=logging.ERROR)
            except Exception as exc:
                raise RuntimeError(
                    f"could not join the Ray cluster at {ray_address} ({_exc_str(exc)}); "
                    "`eray tpu health` runs a device probe on every worker and must run on the "
                    "cluster itself — from a laptop use `eray tpu status` / `eray resources` instead"
                ) from None

    @ray.remote
    def check_host():
        import os
        import socket

        report = {
            "host": socket.gethostname(),
            "worker_id": os.environ.get("TPU_WORKER_ID", os.environ.get("TPU_WORKER_INDEX", "")),
        }
        try:
            import jax

            report["jax_devices"] = [str(d) for d in jax.local_devices()]
            report["device_count"] = jax.device_count()
            report["local_device_count"] = jax.local_device_count()
        except ImportError:
            report["jax_devices"] = []
            report["device_count"] = 0
            report["local_device_count"] = 0
        return report

    # Run on all nodes
    refs = [check_host.remote()]
    nodes = ray.nodes()
    alive = [n for n in nodes if n.get("Alive")]
    for _node in alive[1:]:
        refs.append(
            check_host.options(
                resources={"TPU": 0.001} if tpu_type else {},
                scheduling_strategy="SPREAD",
            ).remote()
        )

    results = ray.get(refs)
    return results if isinstance(results, list) else [results]
