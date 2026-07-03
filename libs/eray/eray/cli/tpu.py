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
import socket
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
        ray_bin: Path to the ray binary on TPU hosts.
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
    info(f"Starting Ray head on host 0 ({head_ip})...")
    head_cmd = (
        f"export TMPDIR={ray_tmp_dir} RAY_TMPDIR={ray_tmp_dir}/ray && "
        f"mkdir -p $TMPDIR $RAY_TMPDIR && "
        f"{ray_bin} stop --force >/dev/null 2>&1 || true && "
        f"{ray_bin} start --head "
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
                f"export TMPDIR={ray_tmp_dir} RAY_TMPDIR={ray_tmp_dir}/ray && "
                f"mkdir -p $TMPDIR $RAY_TMPDIR && "
                f"{ray_bin} stop --force >/dev/null 2>&1 || true && "
                f"{ray_bin} start "
                f"--address={ray_address} "
                f"--resources='{worker_resources}' "
                f"--node-ip-address={worker_ip} "
                f"--disable-usage-stats"
            )
            return host_idx, run_on_host(tpu, host_idx, worker_cmd, timeout=120, user=user, ssh_key=ssh_key)

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
            raise RuntimeError(
                f"{len(failed_workers)} Ray workers failed to start (e.g. {failed_workers[:8]})"
            )
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
        ray_bin: Path to the ray binary on TPU hosts.
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
        ray_bin: Path to the ray binary on TPU hosts.
        user: SSH user for direct-IP mode (ignored in gcloud mode).
        ssh_key: SSH key path for direct-IP mode (ignored in gcloud mode).

    Returns:
        None
    """
    cleanup_cmd = (
        f"timeout 60 {ray_bin} stop --force >/tmp/eray-ray-stop.log 2>&1 || true; "
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
