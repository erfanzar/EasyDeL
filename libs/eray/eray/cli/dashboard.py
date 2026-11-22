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

"""`eray dashboard` — unified status + tracked tunnels for every eray-managed cluster.

One view over both `eray fleet` (QR-based TPU pods) and `eray autoscale`
(Ray cluster-launcher) clusters: which exist, are they up, is a dashboard
tunnel already open for one, which local port, how long it's been up (or
when it went down). Opening a dashboard spawns a tracked background
port-forward (`provision.tunnel`) instead of blocking the terminal — asking
twice reuses the existing forward instead of colliding on a port, which is
the exact failure `ray dashboard` has no memory to avoid on its own.
"""

from __future__ import annotations

import json as json_lib
import sys
import webbrowser

import click

from ..provision.fleet import RAY_DASHBOARD_PORT, RAY_HEAD_PORT
from ..provision.registry import ClusterRecord, ClusterRegistry
from .utils import info, print_table, success

#: `ray status`/`ray.init()`-based tools (eray resources, eray tpu status)
#: need the GCS port, not the dashboard's — tracked as a second, independent
#: tunnel per cluster under this derived name.
GCS_TUNNEL_SUFFIX = "-gcs"


def _gcs_tunnel_name(name: str) -> str:
    """The tunnel-store key for a cluster's GCS (not dashboard) forward."""
    return f"{name}{GCS_TUNNEL_SUFFIX}"


def _load_records(output_dir=None) -> list[ClusterRecord]:
    """Every known launcher/qr cluster, sorted by name.

    Registered clusters (`eray fleet add`, or a prior `eray autoscale up`)
    plus any generated autoscale profile that was never registered —
    brought up by hand, via a bare `ray up`, or by an eray version before
    `up` started writing to the registry. Unregistered profiles get a
    synthetic in-memory `ClusterRecord` (state ``"UNREGISTERED"``, never
    persisted) built from the on-disk YAML alone, so `eray dashboard` can
    see and open them without requiring `eray autoscale up` first — mirrors
    the same on-disk-plus-registry merge `eray autoscale status` already does.

    Args:
        output_dir: Directory autoscale profiles are discovered in; None uses
            the default. Must match the ``--dir`` `eray autoscale` used, or a
            cluster generated to a custom dir would show in one view and not
            the other.
    """
    from ..provision.launcher import DEFAULT_OUTPUT_DIR, list_profiles

    records = {r.name: r for r in ClusterRegistry.from_config().load().values()}
    for profile in list_profiles(output_dir or DEFAULT_OUTPUT_DIR):
        if profile.name not in records:
            records[profile.name] = ClusterRecord(
                name=profile.name,
                kind="launcher",
                project=profile.project,
                zone=profile.zone,
                config_path=str(profile.path),
                state="UNREGISTERED",
            )
    return sorted(records.values(), key=lambda r: r.name)


def _session_row(record: ClusterRecord, tunnels: dict) -> dict:
    """One display row for a registered cluster (either kind).

    Args:
        record: The registered cluster.
        tunnels: Live tunnels keyed by name (`provision.tunnel.list_tunnels`).

    Returns:
        Row dict: name, kind, status, tunnel, since.
    """
    if record.kind == "launcher":
        from ..provision.launcher import load_profile
        from .autoscale import launcher_row_status

        # record.name is the registry key (a launcher profile's YAML file
        # stem); the live GCE probe must key on the YAML's actual
        # cluster_name field instead, since a renamed/hand-edited profile
        # can let the two diverge (AutoscaleProfile: "the YAML is the
        # source of truth"). config_path is always set once `up` has run.
        profile = load_profile(record.config_path) if record.config_path else None
        cluster_name = profile.cluster_name if profile else record.name

        status, since = launcher_row_status(
            cluster_name,
            project=record.project,
            zone=record.zone,
            state=record.state,
            last_down_ts=record.last_down_ts,
        )
    else:
        from ..provision.fleet import head_reachable
        from .utils import format_ago

        if record.head_ip and head_reachable(record.head_ip):
            status = "RUNNING"
        elif record.head_ip:
            # Head recorded but not responding: report that honestly instead
            # of echoing a possibly-stale record.state — a preempted spot pod
            # would otherwise still read HEALTHY here, contradicting both the
            # launcher branch (which reports DOWN) and `eray fleet status`.
            status = "UNREACHABLE"
        else:
            status = record.state
        since = (
            f"died {format_ago(record.last_down_ts)}" if status != "RUNNING" and record.last_down_ts is not None else "-"
        )

    tunnel = tunnels.get(record.name)
    gcs_tunnel = tunnels.get(_gcs_tunnel_name(record.name))
    parts = []
    if tunnel:
        parts.append(f"127.0.0.1:{tunnel.local_port} (pid {tunnel.pid})")
    if gcs_tunnel:
        parts.append(f"gcs:127.0.0.1:{gcs_tunnel.local_port} (pid {gcs_tunnel.pid})")
    tunnel_str = " ".join(parts) if parts else "-"

    return {"name": record.name, "kind": record.kind, "status": status, "tunnel": tunnel_str, "since": since}


def _collect_sessions(output_dir=None) -> tuple[list[ClusterRecord], list[dict]]:
    """Every registered cluster, with live-probed display rows.

    Each row's probe is an independent gcloud subprocess or a socket
    connect (a few hundred ms to a few seconds); a thread pool runs them
    concurrently so `eray dashboard`/`ls` doesn't scale linearly with fleet
    size, mirroring `run_on_all_hosts` in `cli/utils.py`.

    Args:
        output_dir: Directory autoscale profiles are discovered in (see
            `_load_records`); None uses the default.

    Returns:
        ``(records, rows)`` — records sorted by name, rows in the same order.
    """
    import concurrent.futures

    from ..provision.tunnel import list_tunnels

    records = _load_records(output_dir)
    if not records:
        return records, []
    tunnels = list_tunnels()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(records), 8)) as pool:
        rows = list(pool.map(lambda r: _session_row(r, tunnels), records))
    return records, rows


def _resolve_target(records: list[ClusterRecord], name: str | None) -> ClusterRecord:
    """Resolve a NAME argument (or prompt) to one registered cluster.

    Args:
        records: Every registered cluster.
        name: Explicit name, or None to prompt.

    Returns:
        The chosen record.

    Raises:
        click.ClickException: No clusters registered, an unknown name was
            given, or no selection was made (e.g. a non-interactive call).
    """
    if not records:
        raise click.ClickException("no clusters registered — `eray fleet add` or `eray autoscale up` one first.")
    if name:
        for record in records:
            if record.name == name:
                return record
        known = ", ".join(r.name for r in records)
        raise click.ClickException(f"unknown cluster {name!r} (known: {known})")
    if len(records) == 1:
        info(f"one cluster registered: {records[0].name} — using it.")
        return records[0]
    click.echo("Multiple clusters registered — choose one:")
    for idx, record in enumerate(records, start=1):
        click.echo(f"  {idx}) {record.name}  ({record.kind}, {record.zone or '?'})")
    try:
        choice = click.prompt("select", type=click.IntRange(1, len(records)))
    except click.Abort:
        raise click.ClickException("no cluster selected — pass a name explicitly (non-interactive session?).") from None
    return records[choice - 1]


# A launcher cluster's forward goes through Ray's own cluster-launcher SSH
# (`exec_cluster`, exactly what `ray dashboard` calls) — but with a *list* of
# port-forwards so one SSH process forwards every port. Two concurrent
# `ray dashboard` processes would instead collide on the per-cluster SSH
# ControlMaster socket (observed live: "UNEXPECTED_EOF"/"Command failed").
# argv: <python> -c <snippet> <config> <local1> <remote1> <local2> <remote2> ...
_LAUNCHER_FORWARD_SNIPPET = (
    "import sys; from ray.autoscaler._private.commands import exec_cluster; "
    "a = sys.argv[1:]; cfg = a[0]; ports = a[1:]; "
    "exec_cluster(cfg, port_forward=[(int(ports[i]), int(ports[i + 1])) for i in range(0, len(ports), 2)], "
    "no_config_cache=False)"
)


def _forward_argv(record: ClusterRecord, port_pairs: list[tuple[int, int]]) -> list[str]:
    """Kind-dispatched argv forwarding every ``(local, remote)`` pair in ONE process.

    Args:
        record: The cluster (kind decides `exec_cluster` vs gcloud SSH).
        port_pairs: ``(local_port, remote_port)`` pairs to forward over one
            connection — e.g. the dashboard (8265) and GCS (6379) ports.

    Returns:
        Full subprocess argv.

    Raises:
        click.ClickException: The record is missing what's needed (no
            `config_path` for a launcher-kind cluster, no project/zone for
            a QR-kind one).
    """
    if record.kind == "launcher":
        if not record.config_path:
            raise click.ClickException(
                f"{record.name}: no config_path on record — re-run `eray autoscale up {record.name}`."
            )
        argv = [sys.executable, "-c", _LAUNCHER_FORWARD_SNIPPET, record.config_path]
        for local, remote in port_pairs:
            argv += [str(local), str(remote)]
        return argv
    if not (record.project and record.zone):
        raise click.ClickException(f"{record.name}: missing project/zone in the registry.")
    from ..provision.fleet import qr_tunnel_argv

    (dash_local, dash_remote), *extra = port_pairs
    return qr_tunnel_argv(
        record,
        remote_port=dash_remote,
        local_port=dash_local,
        extra_ports=tuple((local, remote) for local, remote in extra),
    )


def _open_session(record: ClusterRecord, *, open_browser: bool, gcs: bool = True) -> None:
    """Open (or reuse) a dashboard tunnel for one cluster.

    Forwards the dashboard port and (by default) the GCS port over a
    **single** SSH process — one connection with several ``-L`` forwards —
    so `eray resources`/`ray status` (which need the GCS port) work from a
    laptop with no `-a`. Each forwarded port gets its own tracked entry
    (sharing the one forwarder's pid) so lookups and `stop` see both.

    Args:
        record: The cluster to open a dashboard for.
        open_browser: Launch the system browser at the dashboard URL too.
        gcs: Also forward the GCS port (`RAY_HEAD_PORT`, 6379). On by
            default; `--no-gcs` opts out (dashboard port only).

    Raises:
        click.ClickException: The record is missing what's needed to build
            a tunnel (e.g. a launcher record with no `config_path`, or a QR
            record with no project/zone).
    """
    from ..provision import tunnel as tunnel_module

    existing = tunnel_module.get_tunnel(record.name)
    if existing is not None:
        info(f"{record.name}: dashboard already open at http://127.0.0.1:{existing.local_port} (pid {existing.pid})")
        gcs_existing = tunnel_module.get_tunnel(_gcs_tunnel_name(record.name))
        if gcs_existing is not None:
            info(f"{record.name}: GCS tunnel at 127.0.0.1:{gcs_existing.local_port} (for `eray resources`/`ray status`)")
        elif gcs:
            info(
                f"{record.name}: this tunnel has no GCS port — "
                f"`eray dashboard stop {record.name}` then reopen to add it."
            )
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{existing.local_port}")
        return

    dash_local = tunnel_module.find_free_port(RAY_DASHBOARD_PORT)
    port_pairs = [(dash_local, RAY_DASHBOARD_PORT)]
    gcs_local = None
    if gcs:
        # Exclude dash_local so the two forwards carried by one SSH can't be
        # handed the same local port (both fall through to an OS-assigned
        # port when 8265/6379 are already busy on this machine).
        gcs_local = tunnel_module.find_free_port(RAY_HEAD_PORT, exclude=(dash_local,))
        port_pairs.append((gcs_local, RAY_HEAD_PORT))

    argv = _forward_argv(record, port_pairs)
    session = tunnel_module.open_tunnel(
        record.name, argv, kind=record.kind, local_port=dash_local, remote_port=RAY_DASHBOARD_PORT
    )
    success(f"{record.name}: dashboard opening at http://127.0.0.1:{session.local_port} (pid {session.pid})")
    info(f"log: {tunnel_module.LOG_DIR / f'{record.name}.log'}")
    if gcs_local is not None:
        tunnel_module.register_alias(
            _gcs_tunnel_name(record.name),
            pid=session.pid,
            kind=record.kind,
            local_port=gcs_local,
            remote_port=RAY_HEAD_PORT,
        )
        info(f"{record.name}: GCS tunnel at 127.0.0.1:{gcs_local} (for `eray resources`/`ray status`)")
    info("tunnels take ~15-20s to establish (gcloud/ray SSH negotiation) before the ports respond.")
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{session.local_port}")


def register(cli: click.Group) -> None:
    """Register the `dashboard` command group on the root CLI.

    Args:
        cli: The root click group.
    """

    _dir_opt = click.option(
        "--dir",
        "profiles_dir",
        default=None,
        help="Where generated autoscale profiles are discovered (default: ~/.eray/autoscale). "
        "Match `eray autoscale --dir` so both views see the same clusters.",
    )

    @cli.group(invoke_without_command=True)
    @click.option("-o", "--open", "open_browser", is_flag=True, default=False, help="Also open the system browser.")
    @click.option(
        "--gcs/--no-gcs",
        "gcs",
        default=True,
        show_default=True,
        help="Also forward the GCS port (for `ray status`/`eray resources`/`ray.init()`).",
    )
    @_dir_opt
    @click.pass_context
    def dashboard(ctx, open_browser, gcs, profiles_dir):
        """Unified status + tracked dashboard tunnels for every eray-managed cluster.

        \b
        Bare (no subcommand): lists every registered cluster, then — when
        stdin is interactive — prompts you to pick one and opens its
        dashboard (reusing an already-open tunnel instead of colliding).
        Opening also forwards the GCS port by default, so `eray resources`
        and `ray status` work from the laptop with no `-a` (opt out with
        --no-gcs).

        \b
        Examples:
            eray dashboard                # list + pick + open (dashboard + GCS)
            eray dashboard ls             # list only, no opening
            eray dashboard open trainer1  # open a specific one
            eray dashboard open trainer1 --no-gcs  # dashboard tunnel only
            eray dashboard stop --all     # close every tracked tunnel
        """
        if ctx.invoked_subcommand is not None:
            return
        records, rows = _collect_sessions(profiles_dir)
        if not rows:
            info("no clusters registered — `eray fleet add` or `eray autoscale up` one first.")
            return
        print_table(rows)
        if not sys.stdin.isatty():
            return
        click.echo()
        target = _resolve_target(records, None)
        _open_session(target, open_browser=open_browser, gcs=gcs)

    @dashboard.command(name="ls")
    @click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
    @_dir_opt
    def ls(as_json, profiles_dir):
        """List every registered cluster with live status — no opening."""
        _, rows = _collect_sessions(profiles_dir)
        if as_json:
            click.echo(json_lib.dumps(rows, indent=2))
            return
        if not rows:
            info("no clusters registered — `eray fleet add` or `eray autoscale up` one first.")
            return
        print_table(rows)

    @dashboard.command(name="open")
    @click.argument("name", required=False)
    @click.option("-o", "--open", "open_browser", is_flag=True, default=False, help="Also open the system browser.")
    @click.option(
        "--gcs/--no-gcs",
        "gcs",
        default=True,
        show_default=True,
        help="Also forward the GCS port (for `ray status`/`eray resources`/`ray.init()`).",
    )
    @_dir_opt
    def open_cmd(name, open_browser, gcs, profiles_dir):
        """Open (or reuse) a dashboard + GCS tunnel for one cluster."""
        target = _resolve_target(_load_records(profiles_dir), name)
        _open_session(target, open_browser=open_browser, gcs=gcs)

    @dashboard.command(name="stop")
    @click.argument("name", required=False)
    @click.option("--all", "stop_all", is_flag=True, default=False, help="Stop every tracked tunnel.")
    def stop_cmd(name, stop_all):
        """Close a tracked dashboard tunnel."""
        from ..provision.tunnel import list_tunnels, stop_tunnel

        if stop_all:
            names = list(list_tunnels())
            if not names:
                info("no open tunnels.")
                return
            for tunnel_name in names:
                stop_tunnel(tunnel_name)
                success(f"stopped {tunnel_name}")
            return
        if not name:
            raise click.ClickException("pass a cluster name, or --all.")
        stopped_dashboard = stop_tunnel(name)
        stopped_gcs = stop_tunnel(_gcs_tunnel_name(name))
        if stopped_dashboard:
            success(f"stopped {name}")
        if stopped_gcs:
            success(f"stopped {name} (GCS tunnel)")
        if not (stopped_dashboard or stopped_gcs):
            info(f"{name}: no open tunnel.")
