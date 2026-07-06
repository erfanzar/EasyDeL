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

RAY_HEAD_PORT = 6379
RAY_DASHBOARD_PORT = 8265
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


def _wait_for_cluster(ray_address: str, expected_nodes: int, *, timeout: int = 300) -> bool:
    """Poll the Ray cluster until all expected nodes are registered.

    Uses ray.init() + ray.nodes() to check cluster topology. Expects
    `expected_nodes` alive nodes, each with TPU resources.

    Args:
        ray_address: Ray cluster address (ip:port).
        expected_nodes: Number of nodes expected to be alive.
        timeout: Maximum time to wait in seconds.

    Returns:
        True if all expected nodes are registered before the timeout, False otherwise.
    """
    import ray

    deadline = time.monotonic() + timeout
    last_seen = None

    while time.monotonic() < deadline:
        try:
            if not ray.is_initialized():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    ray.init(
                        address=ray_address,
                        ignore_reinit_error=True,
                        logging_level=logging.ERROR,
                    )
            alive = [n for n in ray.nodes() if n.get("Alive")]
            total_tpus = sum(n.get("Resources", {}).get("TPU", 0) for n in alive)
            current = (len(alive), int(total_tpus))
            if current != last_seen:
                info(f"Cluster: {current[0]}/{expected_nodes} nodes, {current[1]} TPU resources")
                last_seen = current
            if len(alive) >= expected_nodes:
                return True
        except Exception as exc:
            logger.debug(f"Waiting for cluster: {exc}")
            ray.shutdown()
        time.sleep(RAY_READINESS_POLL_S)

    return False


def cluster_status(ray_address: str) -> dict:
    """Get cluster status summary.

    Args:
        ray_address: Ray cluster address (ip:port).

    Returns:
        A dict with keys: alive_nodes, total_nodes, resources, node_ips,
        dashboard_url.
    """
    import ray

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if not ray.is_initialized():
            _require_reachable_head(ray_address)
            ray.init(address=ray_address, ignore_reinit_error=True, logging_level=logging.ERROR)

    nodes = ray.nodes()
    alive = [n for n in nodes if n.get("Alive")]
    all_resources: dict[str, float] = {}
    for n in alive:
        for k, v in n.get("Resources", {}).items():
            all_resources[k] = all_resources.get(k, 0) + v

    node_ips = [n.get("NodeManagerAddress", "?") for n in alive]

    return {
        "alive_nodes": len(alive),
        "total_nodes": len(nodes),
        "resources": all_resources,
        "node_ips": node_ips,
        "dashboard_url": f"http://{ray_address.split(':')[0]}:{RAY_DASHBOARD_PORT}",
    }


def resource_usage(ray_address: str, *, per_node: bool = False) -> dict:
    """Collect cluster resource totals, availability, and usage.

    Uses the public ``ray.cluster_resources()`` / ``ray.available_resources()``
    pair for the cluster view. A resource that is fully consumed disappears
    from ``available_resources()`` entirely, so usage is computed as
    ``total - available.get(name, 0)``. The per-node view additionally uses
    Ray's private ``available_resources_per_node`` (stable across the 2.x
    line but private API); when it is missing, per-node usage degrades to
    totals with ``available``/``used`` set to None rather than failing.

    Args:
        ray_address: Ray cluster address (ip:port).
        per_node: If True, include a per-node breakdown of alive nodes.

    Returns:
        A dict with:
            resources: {name: {"total", "available", "used"}} for the cluster.
            nodes (only when per_node): list of {"ip", "node_id", "resources"}
                sorted by IP, where each node's resources map has the same
                shape (``node:*`` marker resources excluded).
    """
    import ray

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if not ray.is_initialized():
            _require_reachable_head(ray_address)
            ray.init(address=ray_address, ignore_reinit_error=True, logging_level=logging.ERROR)

    total = ray.cluster_resources()
    available = ray.available_resources()
    resources = {
        name: {
            "total": tot,
            "available": available.get(name, 0.0),
            "used": max(tot - available.get(name, 0.0), 0.0),
        }
        for name, tot in total.items()
    }
    out: dict = {"resources": resources}

    if per_node:
        try:
            from ray._private.state import available_resources_per_node

            per_node_available: dict | None = available_resources_per_node()
        except Exception:
            per_node_available = None

        nodes = []
        for node in ray.nodes():
            if not node.get("Alive"):
                continue
            node_id = node.get("NodeID")
            node_available = None if per_node_available is None else per_node_available.get(node_id, {})
            node_resources = {}
            for name, tot in (node.get("Resources") or {}).items():
                if name.startswith("node:"):
                    continue
                if node_available is None:
                    node_resources[name] = {"total": tot, "available": None, "used": None}
                else:
                    avail = node_available.get(name, 0.0)
                    node_resources[name] = {"total": tot, "available": avail, "used": max(tot - avail, 0.0)}
            nodes.append(
                {
                    "ip": node.get("NodeManagerAddress", "?"),
                    "node_id": node_id,
                    "resources": node_resources,
                }
            )
        def _ip_key(entry: dict):
            try:
                return (0, tuple(int(part) for part in entry["ip"].split(".")))
            except ValueError:
                return (1, entry["ip"])

        nodes.sort(key=_ip_key)
        out["nodes"] = nodes

    return out


def health_check(ray_address: str, tpu_type: str | None = None) -> list[dict]:
    """Run a health check across the cluster.

    Uses a Ray remote function to run a check on every host, reporting
    JAX devices and host info.

    Args:
        ray_address: Ray cluster address (ip:port).
        tpu_type: Optional TPU type (e.g. "v4-32") for scheduling.

    Returns:
        A list of per-host health report dicts.
    """
    import ray

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if not ray.is_initialized():
            _require_reachable_head(ray_address)
            ray.init(address=ray_address, ignore_reinit_error=True, logging_level=logging.ERROR)

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
