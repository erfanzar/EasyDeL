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

"""Main CLI entry point using click.

Commands:
    eray tpu connect     Connect TPU hosts into a Ray cluster
    eray tpu disconnect  Stop Ray on all TPU hosts
    eray tpu status      Show cluster status
    eray tpu health      Run health check across cluster
    eray resources       Show cluster resources and their usage
    eray tpu list        List TPUs in a zone
    eray version         Print version

Connection modes:
    --tpu-name (gcloud)  Discover IPs from TPU name via gcloud
    --ips (direct)       Provide IPs directly, no gcloud needed
"""

from __future__ import annotations

import importlib.metadata
import json as json_lib
import logging
import sys

import click

from .tpu import (
    cluster_status,
    connect_tpus,
    disconnect_tpus,
    health_check,
    resource_usage,
)
from .utils import (
    TpuInfo,
    check_gcloud,
    detect_local_tpu,
    detect_project,
    discover_tpu,
    error,
    get_active_account,
    info,
    list_tpus_in_project,
    list_tpus_in_zone,
    success,
    warning,
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _validate_gcloud() -> None:
    """Check that gcloud is available and authenticated.

    Verifies the gcloud CLI is installed and the user has an active account.

    Raises:
        click.ClickException: If gcloud is not installed or no account is authenticated.
    """
    if not check_gcloud():
        error("gcloud CLI not found. Install Google Cloud SDK first.")
        raise click.ClickException("gcloud not installed")
    account = get_active_account()
    if not account:
        error("No active gcloud account. Run 'gcloud auth login' first.")
        raise click.ClickException("gcloud not authenticated")


def _resolve_tpu(
    tpu_name: str | None,
    project: str | None,
    zone: str | None,
    ips: str | None,
    tpu_type: str | None,
    user: str | None,
    ssh_key: str | None,
) -> tuple[TpuInfo, str | None, str | None]:
    """Resolve a TpuInfo from either --tpu-name or --ips.

    Validates mutual exclusivity of the two modes and required fields for each.
    In gcloud mode, the TPU is discovered via gcloud. In direct-IP mode, a
    TpuInfo is constructed from the provided IPs.

    Args:
        tpu_name: TPU resource name for gcloud mode, or None.
        project: GCP project ID for gcloud mode, or None.
        zone: GCP zone for gcloud mode, or None.
        ips: Comma-separated host IPs for direct-IP mode, or None.
        tpu_type: Accelerator type (e.g. "v4-32") for direct-IP mode, or None.
        user: SSH user for direct-IP mode, or None.
        ssh_key: SSH key path for direct-IP mode, or None.

    Returns:
        A tuple of (tpu_info, user, ssh_key). user and ssh_key are set only in
        direct-IP mode (gcloud mode returns None for both).

    Raises:
        click.ClickException: If both --tpu-name and --ips are provided, if
            required options are missing for the chosen mode, or if no valid
            IPs are found in --ips.
    """
    if tpu_name and ips:
        raise click.ClickException("--tpu-name and --ips are mutually exclusive. Pick one.")

    if tpu_name:
        if not project or not zone:
            raise click.ClickException("--project and --zone are required with --tpu-name.")
        _validate_gcloud()
        return discover_tpu(tpu_name, project, zone), None, None

    if ips:
        if not tpu_type:
            raise click.ClickException("--tpu-type is required with --ips (e.g. v4-32).")
        ip_list = [ip.strip() for ip in ips.split(",") if ip.strip()]
        if not ip_list:
            raise click.ClickException("--ips contains no valid IPs.")
        info(f"Direct-IP mode: {len(ip_list)} hosts, type={tpu_type}")
        return TpuInfo.from_ips(ip_list, tpu_type), user, ssh_key

    # No flags: when running on a TPU VM, the instance metadata identifies
    # the TPU (name, type, every worker's IP) — nothing else to ask for.
    detected = detect_local_tpu()
    if detected is not None:
        info(
            f"Auto-detected TPU '{detected.name or 'unnamed'}' "
            f"({detected.accelerator_type}, {detected.num_hosts} host(s)) from this VM's metadata"
        )
        return detected, user, ssh_key

    raise click.ClickException(
        "Provide --tpu-name or --ips (with no flags, the TPU is auto-detected only when running on a TPU VM)."
    )


try:
    _VERSION = importlib.metadata.version("eray")
except importlib.metadata.PackageNotFoundError:  # running from source, not installed
    _VERSION = "0.0.0+dev"


@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=False)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
@click.version_option(version=_VERSION, prog_name="eray")
def cli(verbose: bool) -> None:
    """eray — Ray-based TPU cluster management CLI.

    Connect TPU hosts into a Ray cluster, check status, run health checks.

    \b
    Three ways to connect:
      0. On the TPU VM itself (auto-detected, no flags):
         eray tpu connect

      1. By TPU name (gcloud discovers IPs):
         eray tpu connect --tpu-name my-tpu --project proj --zone us-central2-b

      2. By IPs directly (no gcloud needed):
         eray tpu connect --ips 10.0.0.1,10.0.0.2 --tpu-type v4-16

    Args:
        verbose: Enable debug logging when True.

    Returns:
        None
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)


# ── Job subcommands (run / status / logs / stop) ─────────────────

from .jobs import register as _register_job_commands  # noqa: E402

_register_job_commands(cli)


# ── TPU subcommands ──────────────────────────────────────────────


@cli.group()
def tpu() -> None:
    """TPU cluster management commands."""


# Common options reused by connect and disconnect
_tpu_name_opt = click.option("--tpu-name", "-n", default=None, help="TPU resource name (gcloud mode).")
_project_opt = click.option("--project", "-p", default=None, help="GCP project ID (gcloud mode).")
_zone_opt = click.option("--zone", "-z", default=None, help="GCP zone (gcloud mode).")
_ips_opt = click.option("--ips", default=None, help="Comma-separated host IPs (direct mode).")
_tpu_type_opt = click.option("--tpu-type", default=None, help="Accelerator type e.g. v4-32 (direct mode).")
_user_opt = click.option("--user", "-u", default=None, help="SSH user for direct-IP mode.")
_ssh_key_opt = click.option("--ssh-key", default=None, help="SSH key path for direct-IP mode.")


@tpu.command()
@_tpu_name_opt
@_project_opt
@_zone_opt
@_ips_opt
@_tpu_type_opt
@_user_opt
@_ssh_key_opt
@click.option(
    "--ray-bin",
    default="ray",
    help="Ray binary on hosts; default auto-resolves per host (driver venv ray, then PATH, then ~/.local/bin/ray).",
)
@click.option("--ray-tmp-dir", default=None, help="Temp dir for Ray on hosts (default: keep tpu.py default).")
@click.option("--timeout", "-t", default=300, help="Cluster readiness timeout (seconds).")
@click.option(
    "--yes-kill-jobs", is_flag=True, default=False, help="Confirm: this restarts Ray and kills every running job."
)
def bounce(tpu_name, project, zone, ips, tpu_type, user, ssh_key, ray_bin, ray_tmp_dir, timeout, yes_kill_jobs):
    """Restart Ray on every host: stop --force, then reconnect the cluster.

    The permanent fix for a raylet stuck spamming its component log (Ray does
    not rotate raylet/GCS logs): a fresh raylet has no dead-driver retry
    state. Kills all running jobs — hence the explicit confirmation flag.
    """
    if not yes_kill_jobs:
        raise click.UsageError("eray tpu bounce restarts the cluster and kills every running job; pass --yes-kill-jobs.")
    from .tpu import RAY_TMP_DIR

    resolved, direct_user, direct_key = _resolve_tpu(tpu_name, project, zone, ips, tpu_type, user, ssh_key)
    disconnect_tpus(resolved, ray_bin=ray_bin, user=direct_user, ssh_key=direct_key)
    try:
        result = connect_tpus(
            resolved,
            ray_bin=ray_bin,
            ray_tmp_dir=ray_tmp_dir or RAY_TMP_DIR,
            timeout=timeout,
            user=direct_user,
            ssh_key=direct_key,
        )
    except RuntimeError as exc:
        error(str(exc))
        raise click.ClickException(str(exc)) from exc
    success("cluster bounced.")
    info(f"  Ray address:  {result.ray_address}")
    info(f"  Dashboard:    {result.dashboard_url}")


@tpu.command()
@_tpu_name_opt
@_project_opt
@_zone_opt
@_ips_opt
@_tpu_type_opt
@_user_opt
@_ssh_key_opt
@click.option(
    "--ray-bin",
    default="ray",
    help="Ray binary on hosts; default auto-resolves per host (driver venv ray, then PATH, then ~/.local/bin/ray).",
)
@click.option("--ray-tmp-dir", default="/tmp/eray_tmp", help="Temp dir for Ray on hosts.")
@click.option("--timeout", "-t", default=300, help="Cluster readiness timeout (seconds).")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output result as JSON.")
def connect(tpu_name, project, zone, ips, tpu_type, user, ssh_key, ray_bin, ray_tmp_dir, timeout, as_json):
    """Connect TPU hosts into a Ray cluster.

    \b
    Mode 0 — on the TPU VM itself (auto-detected from instance metadata):
        eray tpu connect

    \b
    Mode 1 — by TPU name (gcloud discovers IPs automatically):
        eray tpu connect -n my-tpu -p my-project -z us-central2-b

    \b
    Mode 2 — by IPs directly (no gcloud required):
        eray tpu connect --ips 10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 --tpu-type v4-32

    \b
    The resulting cluster exposes custom Ray resources:
      TPU, TPU-v4, TPU-v4-32-head, accelerator_type:TPU-V4, head-node

    \b
    For direct-IP mode you can also specify SSH credentials:
        eray tpu connect --ips 10.0.0.1,10.0.0.2 --tpu-type v4-16 --user myuser --ssh-key ~/.ssh/id_rsa

    Args:
        tpu_name: TPU resource name (gcloud mode).
        project: GCP project ID (gcloud mode).
        zone: GCP zone (gcloud mode).
        ips: Comma-separated host IPs (direct mode).
        tpu_type: Accelerator type e.g. v4-32 (direct mode).
        user: SSH user for direct-IP mode.
        ssh_key: SSH key path for direct-IP mode.
        ray_bin: Path to ray binary on hosts.
        ray_tmp_dir: Temp dir for Ray on hosts.
        timeout: Cluster readiness timeout in seconds.
        as_json: Output result as JSON.

    Returns:
        None

    Raises:
        click.ClickException: If the connection fails.
    """
    tpu, ssh_user, ssh_key_resolved = _resolve_tpu(tpu_name, project, zone, ips, tpu_type, user, ssh_key)

    mode = "gcloud" if tpu.is_gcloud_managed else "direct-IP"
    info(f"TPU: type={tpu.accelerator_type} | hosts={tpu.num_hosts} | mode={mode}")

    if tpu.is_gcloud_managed and tpu.state != "READY":
        warning(f"TPU state is {tpu.state}, expected READY. Proceeding anyway...")

    try:
        result = connect_tpus(
            tpu,
            ray_bin=ray_bin,
            ray_tmp_dir=ray_tmp_dir,
            timeout=timeout,
            user=ssh_user,
            ssh_key=ssh_key_resolved,
        )
    except RuntimeError as exc:
        error(str(exc))
        raise click.ClickException(str(exc)) from exc

    success("🎉 Ray cluster connected!")
    info(f"  Ray address:  {result.ray_address}")
    info(f"  Dashboard:    {result.dashboard_url}")
    info(f"  Nodes:        {result.num_hosts}")
    info(f"  Head IP:      {result.head_ip}")

    if as_json:
        click.echo(
            json_lib.dumps(
                {
                    "ray_address": result.ray_address,
                    "dashboard_url": result.dashboard_url,
                    "head_ip": result.head_ip,
                    "num_hosts": result.num_hosts,
                    "tpu_type": tpu.accelerator_type,
                    "mode": mode,
                },
                indent=2,
            )
        )


@tpu.command()
@_tpu_name_opt
@_project_opt
@_zone_opt
@_ips_opt
@_tpu_type_opt
@_user_opt
@_ssh_key_opt
@click.option(
    "--ray-bin",
    default="ray",
    help="Ray binary on hosts; default auto-resolves per host (driver venv ray, then PATH, then ~/.local/bin/ray).",
)
def disconnect(tpu_name, project, zone, ips, tpu_type, user, ssh_key, ray_bin):
    """Stop Ray on all TPU hosts.

    Works with both --tpu-name (gcloud) and --ips (direct) modes.

    \b
    Example (gcloud):
        eray tpu disconnect -n my-tpu -p my-project -z us-central2-b

    \b
    Example (direct-IP):
        eray tpu disconnect --ips 10.0.0.1,10.0.0.2 --tpu-type v4-16

    Args:
        tpu_name: TPU resource name (gcloud mode).
        project: GCP project ID (gcloud mode).
        zone: GCP zone (gcloud mode).
        ips: Comma-separated host IPs (direct mode).
        tpu_type: Accelerator type e.g. v4-32 (direct mode).
        user: SSH user for direct-IP mode.
        ssh_key: SSH key path for direct-IP mode.
        ray_bin: Path to ray binary on hosts.

    Returns:
        None

    Raises:
        click.ClickException: If the disconnect fails.
    """
    tpu, ssh_user, ssh_key_resolved = _resolve_tpu(tpu_name, project, zone, ips, tpu_type, user, ssh_key)

    info(f"Stopping Ray on {tpu.num_hosts} hosts...")

    try:
        disconnect_tpus(tpu, ray_bin=ray_bin, user=ssh_user, ssh_key=ssh_key_resolved)
    except Exception as exc:
        error(f"Disconnect failed: {exc}")
        raise click.ClickException(str(exc)) from exc

    success("Ray stopped on all hosts")


def _resolve_address(address: str | None) -> str:
    """Resolve a Ray head address, auto-detecting on a TPU VM.

    Args:
        address: Explicit ``ip:port``, or None to detect the local TPU's
            head (worker 0's internal IP on the standard Ray port).

    Returns:
        The Ray cluster address.

    Raises:
        click.ClickException: If no address was given and this machine is
            not a TPU VM.
    """
    if address:
        return address
    detected = detect_local_tpu()
    if detected is not None:
        resolved = f"{detected.internal_ips[0]}:6379"
        info(f"Auto-detected Ray head {resolved} from this VM's metadata")
        return resolved
    raise click.ClickException(
        "Provide --address (with no flags, the head is auto-detected only when running on a TPU VM)."
    )


@tpu.command()
@click.option("--address", "-a", default=None, help="Ray cluster address (ip:port). Auto-detected on a TPU VM.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def status(address, as_json):
    """Show Ray cluster status.

    \b
    Examples:
        eray tpu status               # on the TPU VM itself
        eray tpu status -a 10.0.0.1:6379

    Args:
        address: Ray cluster address (ip:port), or None to auto-detect.
        as_json: Output as JSON.

    Returns:
        None

    Raises:
        click.ClickException: If the status check fails.
    """
    try:
        result = cluster_status(_resolve_address(address))
    except Exception as exc:
        error(f"Failed to get cluster status: {exc}")
        raise click.ClickException(str(exc)) from exc

    if as_json:
        click.echo(json_lib.dumps(result, indent=2, default=str))
    else:
        info(f"Alive nodes:    {result['alive_nodes']}")
        info(f"Total nodes:    {result['total_nodes']}")
        info(f"Dashboard:      {result['dashboard_url']}")
        info(f"Node IPs:       {', '.join(result['node_ips'])}")
        if result["resources"]:
            info("Resources:")
            for key, val in sorted(result["resources"].items()):
                click.echo(f"    {key:40s} {val}")


@tpu.command()
@click.option("--address", "-a", default=None, help="Ray cluster address (ip:port). Auto-detected on a TPU VM.")
@click.option("--tpu-type", default=None, help="TPU type (e.g. v4-32) for scheduling.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def health(address, tpu_type, as_json):
    """Run a health check across the cluster.

    \b
    Examples:
        eray tpu health               # on the TPU VM itself
        eray tpu health -a 10.0.0.1:6379

    Args:
        address: Ray cluster address (ip:port), or None to auto-detect.
        tpu_type: TPU type (e.g. v4-32) for scheduling.
        as_json: Output as JSON.

    Returns:
        None

    Raises:
        click.ClickException: If the health check fails.
    """
    try:
        reports = health_check(_resolve_address(address), tpu_type=tpu_type)
    except Exception as exc:
        error(f"Health check failed: {exc}")
        raise click.ClickException(str(exc)) from exc

    if as_json:
        click.echo(json_lib.dumps(reports, indent=2, default=str))
    else:
        for report in reports:
            host = report.get("host", "?")
            worker = report.get("worker_id", "?")
            dev_count = report.get("device_count", 0)
            local = report.get("local_device_count", 0)
            devices = report.get("jax_devices", [])
            status_icon = "✅" if dev_count > 0 else "❌"
            info(f"{status_icon} Worker {worker} ({host}): {local} local / {dev_count} total devices")
            if devices and len(devices) <= 8:
                for d in devices:
                    click.echo(f"      {d}")

        total_hosts = len(reports)
        total_devices = sum(r.get("device_count", 0) for r in reports)
        success(f"Health check passed: {total_hosts} hosts, {total_devices} total devices")


# ── Resource usage (top-level: works for any eray-managed Ray cluster) ──

_MEMORY_RESOURCES = {"memory", "object_store_memory"}
_RESOURCE_DISPLAY_ORDER = {"CPU": 0, "GPU": 1, "TPU": 2, "memory": 3, "object_store_memory": 4}


def _fmt_qty(name: str, value: float | None) -> str:
    """Format a resource quantity for display.

    Args:
        name: Resource name; memory-like resources are byte counts.
        value: Quantity, or None when per-node availability is unknown.

    Returns:
        Human-readable string (GiB for memory resources, thousands-separated
        counts otherwise, "?" for unknown).
    """
    if value is None:
        return "?"
    if name in _MEMORY_RESOURCES:
        return f"{value / (1024**3):,.1f}GiB"
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _resource_sort_key(name: str) -> tuple[int, str]:
    """Sort core resources (CPU/GPU/TPU/memory) first, then customs by name."""
    return (_RESOURCE_DISPLAY_ORDER.get(name, len(_RESOURCE_DISPLAY_ORDER)), name)


@cli.command()
@click.option("--address", "-a", default=None, help="Ray cluster address (ip:port). Auto-detected on a TPU VM.")
@click.option("--per-node", "-N", "per_node", is_flag=True, default=False, help="Also show a per-node breakdown.")
@click.option("--all", "show_all", is_flag=True, default=False, help="Include node:* marker resources.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def resources(address, per_node, show_all, as_json):
    """Show cluster resources and how much of each is in use.

    Reports used / total / free and utilization for every cluster resource
    (CPU, TPU/GPU, memory, object store, custom resources). With --per-node,
    adds a per-node usage table to spot busy or idle hosts.

    \b
    Examples:
        eray resources                    # on the TPU VM itself
        eray resources -a 10.0.0.1:6379
        eray resources --per-node
        eray resources --json | jq '.resources.TPU'

    Args:
        address: Ray cluster address (ip:port), or None to auto-detect.
        per_node: Also show a per-node breakdown.
        show_all: Include node:* marker resources.
        as_json: Output as JSON.

    Returns:
        None

    Raises:
        click.ClickException: If the cluster cannot be reached.
    """
    try:
        result = resource_usage(_resolve_address(address), per_node=per_node)
    except Exception as exc:
        error(f"Failed to get resource usage: {exc}")
        raise click.ClickException(str(exc)) from exc

    if not show_all:
        result["resources"] = {k: v for k, v in result["resources"].items() if not k.startswith("node:")}

    if as_json:
        click.echo(json_lib.dumps(result, indent=2, default=str))
        return

    rows = []
    for name in sorted(result["resources"], key=_resource_sort_key):
        res = result["resources"][name]
        util = (res["used"] / res["total"] * 100.0) if res["total"] else 0.0
        rows.append(
            (
                name,
                _fmt_qty(name, res["used"]),
                _fmt_qty(name, res["total"]),
                _fmt_qty(name, res["available"]),
                f"{util:5.1f}%",
            )
        )
    heads = ("RESOURCE", "USED", "TOTAL", "FREE")
    widths = [max([len(row[col]) for row in rows] + [len(head)]) for col, head in enumerate(heads)]
    header = f"{heads[0]:<{widths[0]}}  {heads[1]:>{widths[1]}}  {heads[2]:>{widths[2]}}  {heads[3]:>{widths[3]}}"
    click.echo(f"{header}    UTIL")
    for name, used, total, free, util in rows:
        click.echo(f"{name:<{widths[0]}}  {used:>{widths[1]}}  {total:>{widths[2]}}  {free:>{widths[3]}}  {util}")

    if per_node:
        nodes = result.get("nodes", [])
        cols = [c for c in ("TPU", "GPU", "CPU", "memory") if any(c in n["resources"] for n in nodes)]
        cells = []
        for node in nodes:
            row = [node["ip"]]
            for col in cols:
                res = node["resources"].get(col)
                if res is None:
                    row.append("-")
                else:
                    row.append(f"{_fmt_qty(col, res['used'])}/{_fmt_qty(col, res['total'])}")
            cells.append(row)
        headers = ["NODE"] + [f"{c} used/total" for c in cols]
        node_widths = [max([len(r[i]) for r in cells] + [len(headers[i])]) for i in range(len(headers))]
        click.echo("")
        click.echo("  ".join(f"{h:<{node_widths[i]}}" for i, h in enumerate(headers)))
        for row in cells:
            click.echo("  ".join(f"{v:<{node_widths[i]}}" for i, v in enumerate(row)))


@tpu.command(name="list")
@click.option("--project", "-p", default=None, help="GCP project ID (auto-detected if omitted).")
@click.option("--zone", "-z", default=None, help="Narrow to a specific GCP zone (default: scan all zones).")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def list_tpus(project, zone, as_json):
    """List TPU VMs in the current project.

    By default, scans ALL zones in the project.
    Use --zone to narrow to a specific zone.

    \b
    Examples:
        eray tpu list                          # all zones (auto-detected project)
        eray tpu list -z us-central2-b         # one zone
        eray tpu list -p my-project -z us-central2-b

    Args:
        project: GCP project ID (auto-detected if omitted).
        zone: Narrow to a specific GCP zone (default: scan all zones).
        as_json: Output as JSON.

    Returns:
        None

    Raises:
        click.ClickException: If listing fails.
    """
    _validate_gcloud()

    # Auto-detect project
    if not project:
        project = detect_project()
        if not project:
            error("Could not detect project. Run 'gcloud config set project <PROJECT>' or pass --project.")
            raise click.ClickException("Project not set")
        info(f"Auto-detected project: {project}")

    # Fetch TPUs — default: all zones; --zone narrows
    try:
        if zone:
            info(f"Listing TPUs in {project}/{zone}...")
            tpus = list_tpus_in_zone(project, zone)
        else:
            info(f"Listing TPUs across all zones in {project}...")
            tpus = list_tpus_in_project(project)
    except Exception as exc:
        error(f"Failed to list TPUs: {exc}")
        raise click.ClickException(str(exc)) from exc

    if as_json:
        click.echo(json_lib.dumps(tpus, indent=2))
        return

    if not tpus:
        scope = f"{project}/{zone}" if zone else f"{project} (all zones)"
        info(f"No TPUs found in {scope}")
        return

    # Format output as a table
    click.echo()
    header = f"  {'NAME':<30s} {'TYPE':<12s} {'STATE':<12s} {'HOSTS':>5s}  {'ZONE'}"
    click.echo(header)
    click.echo(f"  {'─' * 30} {'─' * 12} {'─' * 12} {'─' * 5}  {'─' * 20}")

    ready_count = 0
    for t in tpus:
        name = t.get("name", "?").split("/")[-1]
        state = t.get("state", "UNKNOWN")
        acc_type = t.get("acceleratorType", "?").split("/")[-1]
        endpoints = t.get("networkEndpoints", [])
        num_hosts = len(endpoints) if isinstance(endpoints, list) else 0

        icon = "🟢" if state == "READY" else "🔴"
        if state == "READY":
            ready_count += 1

        # Zone: from enriched key (all-zones) or from the --zone arg
        tpu_zone = t.get("zone") or zone or "?"

        click.echo(f"  {icon} {name:<28s} {acc_type:<12s} {state:<12s} {num_hosts:>5d}  {tpu_zone}")

    click.echo()
    scope = f"{project}/{zone}" if zone else f"{project} (all zones)"
    info(f"{ready_count}/{len(tpus)} READY in {scope}")


def main():
    """Console script entry point."""
    cli()


if __name__ == "__main__":
    main()
