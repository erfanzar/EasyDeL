# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""`eyvan start` -- the first screen, for somebody who does not know what exists yet.

Every other command assumes you already know the answer to a question this one
asks: which control plane, which fleet, which zone. That is fine on the tenth
day and useless on the first, and equally useless on the day you open a project
somebody else configured. The whole flow here is one question -- *what is
already here, and what do I want to use* -- answered by looking rather than by
being told.

It is a guided console, not a full-screen application. Rich is already a
dependency and a full-screen TUI would need another one; more importantly the
work this does is a short sequence of decisions with real consequences
(connecting to somebody's plane, generating fleet configs, applying capacity),
and a sequence reads better as a transcript you can scroll back through than as
a pane that repaints over its own history.

**Nothing here provisions anything on its own.** It discovers, connects, and
then *offers* the command that would -- printing it first. A first-run tool that
creates cloud resources while you are still working out what the words mean is
the one thing this must not be.
"""

from __future__ import annotations

import shlex
import typing as tp
from dataclasses import dataclass, field

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from eyvan.sdk.commands.head import HeadCommandError, HeadCommandRunner
from eyvan.sdk.commands.head_discovery import DiscoveredHead, configured_project, discover_heads
from eyvan.sdk.terminal import print_block

#: What a local development plane answers on when nobody has said otherwise.
LOCAL_URL = "http://127.0.0.1:8080"


@dataclass(frozen=True, slots=True)
class PlaneTarget:
    """One control plane a person could choose on this screen.

    Attributes:
        kind: ``head`` for a discovered GCP head, ``local`` for a plane already
            answering on this machine, ``new-head`` / ``new-local`` for the two
            things that do not exist yet.
        label: What the row says.
        url: Where it answers, when it does.
        fleet: The fleet name to connect under, when one is known.
        detail: The second column -- status, address, or what choosing it does.
        reachable: Whether something answered. ``False`` is never rendered as
            absence: a stopped head is a head, and hiding it is how somebody
            creates a second one beside it.
        project: GCP project queried during discovery, not a gcloud default.
        instance: Discovered VM name, independent of the display label.
        zone: Discovered VM zone, used to scope recovery commands.
    """

    kind: str
    label: str
    detail: str
    url: str | None = None
    fleet: str | None = None
    reachable: bool = False
    project: str | None = None
    instance: str | None = None
    zone: str | None = None


@dataclass
class StartOutcome:
    """What the flow established, for a caller that wants it as data."""

    project: str | None = None
    heads: list[DiscoveredHead] = field(default_factory=list)
    local_reachable: bool = False
    discovery_error: str | None = None
    targets: list[PlaneTarget] = field(default_factory=list)


def plan_targets(
    heads: tp.Sequence[DiscoveredHead],
    *,
    local_reachable: bool,
    local_url: str = LOCAL_URL,
    project: str | None = None,
) -> list[PlaneTarget]:
    """Turn what was found into the choices offered, in the order they matter.

    Running heads first, because connecting to one is the cheapest useful
    action. Stopped heads are still listed -- and marked -- since the useful
    move there is to start the one that exists rather than create another.
    Creation comes last, so nobody reaches it before seeing what is already
    there.
    """

    running = [head for head in heads if head.status == "RUNNING"]
    stopped = [head for head in heads if head.status != "RUNNING"]

    targets = [
        PlaneTarget(
            kind="head",
            label=f"{head.fleet} — {head.instance}",
            detail=f"{head.zone} · {head.address or 'no internal address'}",
            url=head.url,
            fleet=head.fleet,
            reachable=head.url is not None,
            project=project,
            instance=head.instance,
            zone=head.zone,
        )
        for head in running
    ]

    if local_reachable:
        targets.append(
            PlaneTarget(
                kind="local",
                label="this machine",
                detail=f"a plane is already answering on {local_url}",
                url=local_url,
                reachable=True,
            )
        )

    targets.extend(
        PlaneTarget(
            kind="head",
            label=f"{head.fleet} — {head.instance}",
            detail=f"{head.zone} · {head.status.lower()}, start it before connecting",
            fleet=head.fleet,
            reachable=False,
            project=project,
            instance=head.instance,
            zone=head.zone,
        )
        for head in stopped
    )

    targets.append(
        PlaneTarget(
            kind="new-head",
            label="a new head on GCP",
            detail="one small CPU VM; prints the plan before it creates anything",
        )
    )
    if not local_reachable:
        targets.append(
            PlaneTarget(
                kind="new-local",
                label="this machine, as a head",
                detail="supervised locally; for trying things, not for a team",
            )
        )
    return targets


def render_targets(console: Console, targets: tp.Sequence[PlaneTarget]) -> None:
    """Print the choices as a numbered table.

    Numbered because this *is* a selection, and the numbers are what a person
    types next. They encode nothing else.
    """

    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    table.add_column("", justify="right", style="dim", width=3)
    table.add_column("control plane")
    table.add_column("", style="dim")
    for index, target in enumerate(targets, start=1):
        marker = "" if target.reachable or target.kind.startswith("new") else "·"
        table.add_row(str(index), f"{marker}{target.label}".strip(), target.detail)
    print_block(console, table)


def python_snippet(url: str, fleet: str | None, accelerator: str = "v5p-8") -> str:
    """Return the Python that submits to the plane just connected to.

    Written out rather than described. The gap between "you can submit from
    Python" and a file somebody can run is where a first session usually ends,
    and it is four lines wide.

    ``init`` rather than ``os.environ["EYVAN_URL"]``: pin the selected endpoint
    without changing the user's environment. Even a local plane needs an
    explicit address, since environment and saved-context defaults can name a
    different plane.
    """

    address = f'"{url}"'
    arguments = ", ".join(part for part in (address, f'fleet="{fleet}"' if fleet else "") if part)
    return (
        f"import eyvan\n"
        f"from eyvan import TpuConfig, execute\n\n"
        f"eyvan.init({arguments})\n\n\n"
        f'@execute(TpuConfig(type="{accelerator}"))\n'
        f"def train():\n"
        f"    import jax\n\n"
        f"    print(jax.device_count())\n\n\n"
        f"train()"
    )


def _probe(url: str, timeout_s: float = 3.0) -> bool:
    """Return whether a control plane answers at ``url``."""

    import httpx

    try:
        response = httpx.get(f"{url.rstrip('/')}/api/v1/health", timeout=timeout_s)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def discover(
    project: str | None,
    *,
    runner: HeadCommandRunner | None = None,
    probe: tp.Callable[[str], bool] = _probe,
    local_url: str = LOCAL_URL,
) -> StartOutcome:
    """Look for control planes, without changing anything.

    A discovery failure is carried on the outcome rather than raised. gcloud
    being absent, unauthenticated, or pointed at another project are all
    ordinary on a first run, and none of them should stop the local options
    being offered -- but reporting "no heads" for them would be a lie that
    ends with a duplicate head.
    """

    outcome = StartOutcome(project=project or configured_project(runner))
    outcome.local_reachable = probe(local_url)
    if outcome.project:
        try:
            outcome.heads = discover_heads(outcome.project, runner=runner)
        except HeadCommandError as error:
            outcome.discovery_error = str(error)
    outcome.targets = plan_targets(
        outcome.heads,
        local_reachable=outcome.local_reachable,
        local_url=local_url,
        project=outcome.project,
    )
    return outcome


def _next_steps(target: PlaneTarget) -> list[tuple[str, str]]:
    """Return the commands that follow one choice, as (command, why) pairs."""

    if target.kind == "new-head":
        return [
            ("eyvan head up fleet.yaml --dry-run", "read every gcloud call before one is issued"),
            ("eyvan head up fleet.yaml --yes", "create the head and start it under systemd"),
            ("eyvan start", "come back here; it will be listed"),
        ]
    if target.kind == "new-local":
        return [
            ("eyvan up examples/laptop.yaml", "supervise a plane and apply a no-cost fleet"),
            ("eyvan start", "come back here; it will be listed"),
        ]
    if not target.reachable:
        if not (target.project and target.instance and target.zone):
            return [("eyvan start", "rediscover the head's project, instance, and zone before starting it")]
        return [
            (
                shlex.join(
                    [
                        "gcloud",
                        "compute",
                        "instances",
                        "start",
                        target.instance,
                        "--project",
                        target.project,
                        "--zone",
                        target.zone,
                    ]
                ),
                "start the head that exists",
            ),
            (shlex.join(["eyvan", "start", "--project", target.project]), "connect once it is running"),
        ]
    fleet = target.fleet or "FLEET"
    return [
        (f"eyvan --url {target.url} use {fleet}", "save this plane as the active context"),
        ("eyvan doctor", "confirm the connection and what is missing"),
        ("eyvan capacity quota", "what this project's TPU quota actually is"),
        ("eyvan autoscale init", "declare the identity generated fleets are built with"),
        ("eyvan up --region us-east5", "generate spot fleets for a region and apply one"),
    ]


def render_outcome(console: Console, outcome: StartOutcome) -> None:
    """Print what was found, including the reason nothing was."""

    if outcome.project:
        console.print(f"[dim]GCP project[/dim] {outcome.project}")
    else:
        console.print(
            "[dim]No GCP project configured.[/dim] "
            "Run [bold]gcloud config set project PROJECT[/bold] to have heads discovered here."
        )
    if outcome.discovery_error:
        # Named, never folded into "no heads found". The two look identical on
        # screen and only one of them means it is safe to create another.
        console.print(f"[yellow]Could not list heads:[/yellow] {outcome.discovery_error}")
    console.print()
    render_targets(console, outcome.targets)


def start_flow(
    console: Console,
    *,
    project: str | None,
    runner: HeadCommandRunner | None = None,
    probe: tp.Callable[[str], bool] = _probe,
    choose: tp.Callable[[int], int] | None = None,
) -> StartOutcome:
    """Run the guided first screen and print what to do next.

    Args:
        console: Where output goes.
        project: GCP project to search, or ``None`` to ask gcloud.
        runner: Injected command boundary.
        probe: Injected health check.
        choose: Injected selection, so the flow is testable without a terminal.
            ``None`` prompts.

    Returns:
        What was discovered, so a caller can render it another way.
    """

    outcome = discover(project, runner=runner, probe=probe)
    render_outcome(console, outcome)
    if not outcome.targets:
        return outcome

    pick = (
        choose(len(outcome.targets))
        if choose
        else click.prompt("\nWhich one", type=click.IntRange(1, len(outcome.targets)), default=1)
    )
    target = outcome.targets[pick - 1]

    console.print()
    steps = Table(box=box.SIMPLE, pad_edge=False, show_edge=False, show_header=False)
    # The command column never wraps. A command broken across two lines is one
    # that cannot be copied, which is the only thing this panel is for.
    steps.add_column(style="bold", no_wrap=True, overflow="fold")
    steps.add_column(style="dim")
    for command, why in _next_steps(target):
        steps.add_row(command, why)
    print_block(console, Panel(steps, title=f"[bold]{target.label}[/bold]", border_style="dim", title_align="left"))

    if target.reachable and target.url:
        console.print("\n[dim]…and from Python, against that plane:[/dim]\n")
        print_block(console, Panel(python_snippet(target.url, target.fleet), border_style="dim", box=box.SQUARE))
    return outcome


@click.command("start")
@click.option("--project", help="GCP project to search for heads; defaults to gcloud's configured project.")
@click.option("--json", "as_json", is_flag=True, help="Emit what was discovered and exit without prompting.")
def start(project: str | None, as_json: bool) -> None:
    """Find the control planes that already exist, and connect to one.

    The first screen: what is running in this project, what is running on this
    machine, and what it would take to create either. Discovers, connects, and
    prints the command for anything that would provision -- it creates nothing
    itself.
    """

    import sys

    from eyvan.sdk.commands.common import JsonRenderer
    from eyvan.sdk.terminal import terminal_console

    if as_json:
        outcome = discover(project)
        JsonRenderer().emit(
            {
                "project": outcome.project,
                "local_reachable": outcome.local_reachable,
                "discovery_error": outcome.discovery_error,
                "heads": [
                    {
                        "fleet": head.fleet,
                        "instance": head.instance,
                        "zone": head.zone,
                        "status": head.status,
                        "url": head.url,
                    }
                    for head in outcome.heads
                ],
            }
        )
        return
    start_flow(terminal_console(sys.stdout), project=project)


def register_start_commands(group: click.Group) -> None:
    """Attach the first-run console to the CLI root."""

    group.add_command(start)


__all__ = (
    "LOCAL_URL",
    "PlaneTarget",
    "StartOutcome",
    "discover",
    "plan_targets",
    "python_snippet",
    "register_start_commands",
    "render_outcome",
    "render_targets",
    "start",
    "start_flow",
)
