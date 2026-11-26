# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Browse existing TPU VMs and attach them without provisioning hardware."""

from __future__ import annotations

import json
import math
import time
import typing as tp
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from eyvan._internal.ssh import SshDestination
from eyvan.capacity.autoscale.cache import AutoscaleConfigCache
from eyvan.capacity.autoscale.models import AutoscaleDefaults
from eyvan.capacity.providers.gcp.runtime.profile import GcpNativeRuntimeProfileStore
from eyvan.capacity.providers.gcp.runtime.supervisor import runtime_profile_path_for
from eyvan.capacity.providers.gcp.tpu_nodes import GcpProject, TpuApiError, TpuNode, TpuNodeApi, TpuNodeParseError
from eyvan.sdk.commands.autoscale import autoscale_console, is_interactive
from eyvan.sdk.commands.gcp_connect import AttachError, AttachPlan, GcpConnector
from eyvan.sdk.commands.gcp_host_access import resolve_attach_runtime
from eyvan.sdk.commands.state import CliSession
from eyvan.sdk.terminal import print_block


def _project(session: CliSession, api: TpuNodeApi, explicit: str | None) -> str:
    """Choose explicitly, from the local profile, or from visible projects."""

    if explicit:
        return explicit
    try:
        return GcpNativeRuntimeProfileStore(runtime_profile_path_for(session.base_url)).load().project_id
    except FileNotFoundError:
        pass
    projects = api.list_projects()
    if len(projects) == 1:
        return projects[0].project_id
    available = ", ".join(project.project_id for project in projects) or "none visible"
    raise click.ClickException(f"specify --project P to choose a GCP project; available: {available}")


def _load_runtime_defaults(path: Path | None) -> AutoscaleDefaults:
    """Read the same defaults profile ``eyvan up`` derives fleets from; absent means empty."""

    try:
        return AutoscaleConfigCache().load_defaults(path)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


def _destination(value: str) -> SshDestination:
    try:
        return SshDestination.parse(value)
    except ValueError as exc:
        raise click.UsageError(f"--ssh must be [user@]host without options, paths or shell syntax ({exc})") from exc


def _node_document(node: TpuNode) -> dict[str, tp.Any]:
    """Expose inventory, never provider credentials or free-form API metadata."""

    return {
        "project": node.project,
        "name": node.name,
        "zone": node.zone,
        "accelerator_type": node.accelerator_type,
        "state": node.state,
        "capacity_class": node.capacity_class.value,
        "hosts": len(node.hosts),
    }


class _Phase:
    """One visible step: a spinner while it runs, then a ✓/✗ line with its duration."""

    def __init__(self, console: Console, label: str, *, live: bool) -> None:
        self.console = console
        self.label = label
        self.live = live
        self._status: tp.Any = None
        self._started = 0.0

    def __enter__(self) -> _Phase:
        self._started = time.monotonic()
        if self.live:
            self._status = self.console.status(f"{self.label}…", spinner="dots")
            self._status.__enter__()
        return self

    def update(self, text: str) -> None:
        """Show live detail (for example cell states) beside the running step."""

        if self._status is not None:
            self._status.update(f"{self.label}… [dim]{text[:100]}[/]")

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if self._status is not None:
            self._status.__exit__(None, None, None)
        elapsed = time.monotonic() - self._started
        took = f" [dim]{elapsed:.0f}s[/]" if elapsed >= 1 else ""
        mark = "[green]✓[/]" if exc_type is None else "[red]✗[/]"
        self.console.print(f"{mark} {self.label}{took}", highlight=False)
        return False


def _progress(as_json: bool) -> tp.Callable[[str], tp.ContextManager[tp.Any]]:
    """Return the step printer: live spinners on a terminal, plain ✓ lines otherwise, nothing for --json."""

    if as_json:
        return GcpConnector.progress
    console = autoscale_console(err=True)
    live = console.is_terminal
    return lambda label: _Phase(console, label, live=live)


def _finish(session: CliSession, connector: GcpConnector, plan: AttachPlan, *, timeout: float, as_json: bool) -> None:
    """Execute a confirmed plan and print exactly one final result."""

    connector.progress = _progress(as_json)
    result = connector.up(plan, timeout=timeout)
    if as_json:
        click.echo(json.dumps(result, sort_keys=True, indent=2))
        return
    body = Table.grid(padding=(0, 2))
    body.add_column(style="dim")
    body.add_column()
    body.add_row("fleet", f"[bold]{plan.fleet}[/]")
    body.add_row("use it", f'[cyan]init(fleet="{plan.fleet}")[/]')
    body.add_row("config", str(plan.fleet_path))
    print_block(autoscale_console(), Panel(body, title="[bold green]Connected[/]", border_style="green", expand=False))
    for limitation in plan.limitations:
        click.echo(f"Note: {limitation}")


@click.group("gcp")
def gcp() -> None:
    """Browse and connect operator-owned hardware (never create or delete TPUs)."""


@gcp.command("browse")
@click.option("--project", "project_id", help="List only this GCP project; otherwise list all visible projects.")
@click.option("--json", "as_json", is_flag=True, help="List only; never prompt or modify hardware.")
@click.pass_obj
def browse(session: CliSession, project_id: str | None, as_json: bool) -> None:
    """List projects and TPU VMs; in a terminal optionally pick one to connect."""

    api = TpuNodeApi()
    try:
        projects = (GcpProject(project_id, None, None),) if project_id else api.list_projects()
        nodes: list[TpuNode] = []
        unreachable: dict[str, list[str]] = {}
        for project in projects:
            found, missing = api.list_nodes(project.project_id)
            nodes.extend(found)
            if missing:
                unreachable[project.project_id] = list(missing)
    except (TpuApiError, TpuNodeParseError, ValueError) as exc:
        raise click.ClickException(f"eyvan gcp browse discovery failed: {exc}") from exc
    if as_json:
        click.echo(
            json.dumps(
                {
                    "projects": [
                        {"project_id": project.project_id, "name": project.name, "state": project.state}
                        for project in projects
                    ],
                    "nodes": [_node_document(node) for node in nodes],
                    "unreachable": unreachable,
                },
                sort_keys=True,
                indent=2,
            )
        )
        return
    click.echo("Projects: " + (", ".join(project.project_id for project in projects) or "none visible"))
    table = Table(title="Existing TPU VMs", pad_edge=False)
    for column in ("#", "Project", "Name", "Zone", "Type", "State", "Class", "Hosts"):
        table.add_column(column)
    for index, node in enumerate(nodes, 1):
        table.add_row(
            str(index),
            node.project,
            node.name,
            node.zone,
            node.accelerator_type,
            node.state,
            node.capacity_class.value.replace("_", "-"),
            str(len(node.hosts)),
        )
    print_block(autoscale_console(), table)
    if not nodes:
        click.echo("No TPU VMs observed.")
    for project, locations in unreachable.items():
        click.echo(f"Unreachable locations in {project}: {', '.join(locations)} (inventory is incomplete)")
    if not nodes or not is_interactive():
        return
    selected = click.prompt("Select TPU to connect", type=click.IntRange(1, len(nodes)))
    node = nodes[selected - 1]
    connector = GcpConnector(session, api=api)
    try:
        plan = connector.plan_tpu(node)
        if not click.confirm(
            f"Install Eyvan workers and attach {node.project}/{node.zone}/{node.name} as {plan.fleet!r}?"
        ):
            return
        _finish(session, connector, plan, timeout=300.0, as_json=False)
    except AttachError as exc:
        raise click.ClickException(str(exc)) from exc


@gcp.command("connect")
@click.option("--tpu", "tpu_name", metavar="NAME", help="Existing TPU VM node name (never provisioned).")
@click.option(
    "--ssh",
    "ssh_host",
    metavar="[USER@]HOST",
    help="Any host you can SSH into. A TPU VM (found via the TPU API, or by asking the host itself when your "
    "credential cannot see its project) is attached as that whole slice; any other machine is attached as a "
    "prepared native-worker host with CPU and memory read over SSH. The user comes from your SSH config unless given.",
)
@click.option("--project", "project_id", metavar="P", help="GCP project; defaults to the local native-runtime profile.")
@click.option("--zone", metavar="Z", help="TPU zone; required when its name is ambiguous across zones.")
@click.option("--fleet", metavar="NAME", help="Fleet name; default is deterministic for this node/host.")
@click.option(
    "--endpoint-mode",
    type=click.Choice(("public", "internal", "tunnel")),
    default=None,
    help="How the local agent reaches the workers. Default: probed (public when the worker port is reachable "
    "from here, else an SSH tunnel through the --ssh host); public for --tpu.",
)
@click.option("--cpus", type=float, help="For a plain --ssh host: override the CPU count read over SSH.")
@click.option("--memory-gib", type=float, help="For a plain --ssh host: override the memory GiB read over SSH.")
@click.option(
    "--capacity-class",
    type=click.Choice(("reserved", "on_demand")),
    default="reserved",
    show_default=True,
    help="For --ssh only; TPUs retain their actual provider purchase class.",
)
@click.option("--region", default="onprem", show_default=True, help="For --ssh: operator-owned region label.")
@click.option(
    "--timeout",
    type=float,
    default=300.0,
    show_default=True,
    help="Seconds to wait for worker health and fleet readiness.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Read inventory (TPU API, or read-only SSH probes) and print every step/YAML; no mutations or builds.",
)
@click.option(
    "--python",
    "runtime_python",
    metavar="X.Y",
    help="Worker interpreter for TPU hosts. Default: runtime_python from the autoscale defaults profile, "
    "else this machine's Python (callables only load on the interpreter that pickled them).",
)
@click.option(
    "--init-command",
    "init_commands",
    multiple=True,
    help="Shell run on every TPU host after the worker environment exists (repeatable); "
    "appended to runtime_init_commands from the defaults profile.",
)
@click.option(
    "--container-image",
    help="OCI image workloads run in on TPU hosts. Default: runtime_container_image from the defaults profile.",
)
@click.option(
    "--defaults",
    "defaults_path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Read runtime setup from this YAML instead of ~/.eyvan/autoscale/defaults.yaml.",
)
@click.option("-y", "--yes", is_flag=True, hidden=True, help="Accepted for compatibility; connect never prompts.")
@click.option("--json", "as_json", is_flag=True, help="Emit one plan/result JSON document; prompts use stderr.")
@click.pass_obj
def connect(
    session: CliSession,
    tpu_name: str | None,
    ssh_host: str | None,
    project_id: str | None,
    zone: str | None,
    fleet: str | None,
    endpoint_mode: str | None,
    cpus: float | None,
    memory_gib: float | None,
    capacity_class: str,
    region: str,
    timeout: float,
    dry_run: bool,
    runtime_python: str | None,
    init_commands: tuple[str, ...],
    container_image: str | None,
    defaults_path: Path | None,
    yes: bool,
    as_json: bool,
) -> None:
    """Attach --tpu NAME or a prepared --ssh USER@HOST and activate its fleet."""

    if (tpu_name is None) == (ssh_host is None):
        raise click.UsageError("choose exactly one of --tpu NAME or --ssh USER@HOST")
    if not math.isfinite(timeout) or timeout <= 0:
        raise click.UsageError("--timeout must be a positive finite number")
    connector = GcpConnector(session)
    try:
        connector.runtime = resolve_attach_runtime(
            defaults=_load_runtime_defaults(defaults_path),
            python=runtime_python,
            init_commands=init_commands,
            container_image=container_image,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    try:
        if ssh_host is not None:
            # An address that belongs to one of the operator's TPU VMs is attached as
            # that TPU: its shape, hosts and purchase class come from the TPU API, and
            # nothing has to be typed. Anything else is a plain SSH host.
            api = TpuNodeApi()
            connector.api = api
            tpu_node, found_mode, notes = connector.find_tpu_by_address(ssh_host, project=project_id, zone=zone)
            discovered = None
            if tpu_node is not None:
                if cpus is not None or memory_gib is not None or capacity_class != "reserved" or region != "onprem":
                    raise click.UsageError(
                        f"{ssh_host} is TPU VM {tpu_node.name!r}; its shape and class come from the TPU API, so "
                        "--cpus, --memory-gib, --capacity-class and --region do not apply"
                    )
                if not as_json:
                    click.echo(
                        f"{ssh_host} is TPU VM {tpu_node.name} ({tpu_node.accelerator_type}, "
                        f"{tpu_node.capacity_class.value.replace('_', '-')}) in {tpu_node.project}/{tpu_node.zone}",
                        err=True,
                    )
                plan = connector.plan_tpu(tpu_node, fleet=fleet, endpoint_mode=endpoint_mode or found_mode)
            else:
                # Not visible to this machine's credential: ask the host itself. A TPU
                # VM names its node in metadata and its own gcloud reads the node.
                destination = _destination(ssh_host)
                discovered = connector.discover_over_ssh(destination)
            if tpu_node is None and discovered is not None:
                node, worker = discovered
                if cpus is not None or memory_gib is not None or capacity_class != "reserved" or region != "onprem":
                    raise click.UsageError(
                        f"{ssh_host} is TPU VM {node.name!r}; its shape and class come from the host, so "
                        "--cpus, --memory-gib, --capacity-class and --region do not apply"
                    )
                mode = endpoint_mode or connector.endpoint_mode_for(node)
                if not as_json:
                    click.echo(
                        f"{ssh_host} is worker {worker if worker is not None else '?'} of TPU VM {node.name} "
                        f"({node.accelerator_type}, {len(node.hosts)} host(s), "
                        f"{node.capacity_class.value.replace('_', '-')}) in {node.project}/{node.zone}; "
                        f"read over SSH, workers reached via {mode}",
                        err=True,
                    )
                plan = connector.plan_tpu(
                    node, fleet=fleet, endpoint_mode=mode, ssh_entry=destination, entry_worker=worker
                )
            elif tpu_node is None:
                if project_id or zone or endpoint_mode is not None:
                    raise click.UsageError("--project, --zone and --endpoint-mode apply only to TPU VMs")
                plan = connector.plan_ssh(
                    ssh_host,
                    cpus=cpus,
                    memory_gib=memory_gib,
                    fleet=fleet,
                    capacity_class=capacity_class,
                    region=region,
                    notes=notes,
                )
        else:
            if cpus is not None or memory_gib is not None or capacity_class != "reserved" or region != "onprem":
                raise click.UsageError("--cpus, --memory-gib, --capacity-class and --region apply only to --ssh")
            api = TpuNodeApi()
            connector.api = api
            project = _project(session, api, project_id)
            node = connector.resolve(project=project, name=tp.cast(str, tpu_name), zone=zone)
            if endpoint_mode == "tunnel":
                raise click.UsageError("--endpoint-mode tunnel needs an SSH entry host; use --ssh HOST")
            plan = connector.plan_tpu(node, fleet=fleet, endpoint_mode=endpoint_mode or "public")
        if dry_run:
            if as_json:
                click.echo(json.dumps(plan.as_document(), sort_keys=True, indent=2))
            else:
                for line in plan.render_lines():
                    click.echo(line)
            return
        # Connecting is what the command was asked to do; --dry-run is the way to
        # look first. ``--yes`` is still accepted so existing scripts keep working.
        del yes
        _finish(session, connector, plan, timeout=timeout, as_json=as_json)
    except (AttachError, TpuApiError, TpuNodeParseError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@gcp.command("sync")
@click.option("--project", "project_ids", multiple=True, help="Only these GCP projects (repeatable); default all.")
@click.option("--json", "as_json", is_flag=True, help="Emit the discovery index as JSON.")
@click.pass_obj
def sync(session: CliSession, project_ids: tuple[str, ...], as_json: bool) -> None:
    """List every TPU VM your gcloud account can see and give each a ready fleet name.

    Read-only: nothing is installed, tagged or opened. A TPU is connected the
    first time a job names its fleet (``init(fleet=...)``) or by ``eyvan gcp connect``.
    """

    from eyvan.capacity.providers.gcp.tpu_discovery import DiscoveryIndex

    try:
        snapshot = DiscoveryIndex().refresh(projects=project_ids or None)
    except (TpuApiError, TpuNodeParseError, ValueError) as exc:
        raise click.ClickException(f"eyvan gcp sync: could not list GCP projects: {exc}") from exc
    connected = set(_connected_fleets(session))
    if as_json:
        document = snapshot.as_document()
        for tpu in document["tpus"]:
            tpu["connected"] = tpu["fleet"] in connected
        click.echo(json.dumps(document, sort_keys=True, indent=2))
        return
    table = Table(title="TPU VMs available as fleets", pad_edge=False)
    for column in ("Fleet", "Project", "Zone", "TPU", "Type", "Class", "State", "Connected"):
        table.add_column(column)
    for tpu in snapshot.tpus:
        table.add_row(
            tpu.fleet,
            tpu.project,
            tpu.zone,
            tpu.name,
            tpu.accelerator_type,
            tpu.capacity_class.replace("_", "-"),
            tpu.state,
            "yes" if tpu.fleet in connected else "on first use",
        )
    print_block(autoscale_console(), table)
    if not snapshot.tpus:
        click.echo("No TPU VMs observed.")
    for project, reason in sorted(snapshot.skipped.items()):
        click.echo(f"{project}: skipped ({reason})")
    if snapshot.tpus:
        click.echo('Use any of them with init(fleet="<fleet or TPU name>"); it connects on first use.')


def _connected_fleets(session: CliSession) -> list[str]:
    """Fleet names the plane already serves; empty when it cannot be asked."""

    try:
        scope = session.workload_scope()
        payload = session.client.request(
            "GET", "/api/v1/fleets", params={"org_id": scope.org_id, "project_id": scope.project_id}
        )
    except Exception:
        return []
    return [name for name in payload if isinstance(name, str)] if isinstance(payload, list) else []


def register_gcp_commands(main: click.Group) -> None:
    """Register the GCP inventory/attachment group at the composition root."""

    main.add_command(gcp)


__all__ = ("gcp", "register_gcp_commands")
